import gc
import os
import sys
import traceback

import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from viqa import config
from viqa.data import clean_text, load_and_clean_dataset, question_leakage_report
from viqa.generation import generate_predictions
from viqa.hub import hf_login, push_run_to_hub
from viqa.metrics import calculate_metrics, compute_bertscore_with_fallback
from viqa.runtime import detect_runtime, seed_everything
from viqa.training import train_one_model
from viqa.utils import run_name_of, safe_name_of


def _check_context_settings():
    if config.TRAIN_CONTEXT_SOURCE not in config.CONTEXT_FIELDS:
        raise ValueError(f"VIQA_TRAIN_CONTEXT must be one of {list(config.CONTEXT_FIELDS)}")
    if config.TRAIN_CONTEXT_SOURCE == "retrieved" and not config.RETRIEVAL_ENABLED:
        raise ValueError("VIQA_TRAIN_CONTEXT=retrieved requires VIQA_RETRIEVAL=1")


def main():
    _check_context_settings()
    os.makedirs(config.OUTPUT_ROOT, exist_ok=True)
    os.makedirs(config.TOKENIZER_ROOT, exist_ok=True)

    runtime = detect_runtime()
    seed_everything()

    print(f"GPU: {runtime.gpu_name} | VRAM: {runtime.vram_gb:.2f} GB")
    print(f"BF16: {runtime.use_bf16} | FP16: {runtime.use_fp16}")
    print(f"Train batch candidates: {runtime.train_batch_candidates}")
    print(f"Runs planned: {[run_name_of(m, c) for m, _, c in config.RUNS]}")

    hf_push = config.HF_PUSH
    hf_api = hf_owner = None
    if hf_push:
        try:
            hf_api = hf_login()
            hf_owner = hf_api.whoami()["name"]
            print(f"Logged in to Hugging Face Hub as: {hf_owner}")
        except Exception as e:
            print(f"Hugging Face login failed ({e}); disabling push for this run.")
            hf_push = False

    dataset = load_and_clean_dataset()
    print("Leakage:", question_leakage_report(dataset))

    # Retrieval: BGE-M3 -> FAISS -> reranker. Runs once, before any generator is loaded.
    if config.RETRIEVAL_ENABLED and any(use_ctx for _, _, use_ctx in config.RUNS):
        from viqa.retrieval import attach_retrieved_context

        splits = ["test"]
        if config.TRAIN_CONTEXT_SOURCE == "retrieved":
            splits = ["train", "validation", "test"]
        dataset, retrieval_metrics = attach_retrieved_context(dataset, splits)
        rm_df = pd.DataFrame([{"Split": s, **m} for s, m in retrieval_metrics.items()])
        rm_df.to_csv(os.path.join(config.OUTPUT_ROOT, "retrieval_metrics.csv"), index=False, encoding="utf-8-sig")
        print("Retrieval metrics:")
        print(rm_df.drop(columns=["Results_File"], errors="ignore").round(2).to_string())
        gc.collect()
        torch.cuda.empty_cache()

    print(f"Train context: {config.TRAIN_CONTEXT_SOURCE} | test contexts: {config.TEST_CONTEXT_SOURCES}")

    predict_script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "predict.py")
    references = [clean_text(x) for x in dataset["test"]["answer"]]
    all_results = []

    for idx, (model_name, cfg, use_context) in enumerate(config.RUNS, 1):
        run_name = run_name_of(model_name, use_context)
        trainer = tokenizer = None
        print(f"\n=== RUN {idx}/{len(config.RUNS)}: {run_name} ===")
        try:
            train_ctx = config.TRAIN_CONTEXT_SOURCE if use_context else "none"
            test_ctxs = config.TEST_CONTEXT_SOURCES if use_context else ["none"]
            trainer, tokenizer, stats = train_one_model(
                run_name, cfg, use_context, dataset, runtime, config.CONTEXT_FIELDS.get(train_ctx, "context")
            )

            evals = []
            for test_ctx in test_ctxs:
                field = config.CONTEXT_FIELDS.get(test_ctx, "context")
                predictions, inference_time = generate_predictions(
                    trainer.model, tokenizer, cfg, use_context, dataset["test"], runtime, field
                )
                if len(predictions) != len(references):
                    raise RuntimeError("prediction/reference count mismatch")
                metrics = calculate_metrics(predictions, references)

                output = {}
                for col in ("question_id", "question_text", "topic", "subtopic"):
                    if col in dataset["test"].column_names:
                        output[col] = dataset["test"][col]
                if use_context:
                    output["context_given_to_model"] = dataset["test"][field]
                    output["test_context_source"] = [test_ctx] * len(references)
                output["reference_answer"] = references
                output["prediction"] = predictions
                suffix = f"__test-{test_ctx}" if use_context else ""
                pred_path = os.path.join(config.OUTPUT_ROOT, f"{safe_name_of(run_name)}{suffix}_predictions.csv")
                pd.DataFrame(output).to_csv(pred_path, index=False, encoding="utf-8-sig")
                evals.append((test_ctx, predictions, inference_time, metrics, pred_path))

            best_dir = stats["Best_Dir"]
            trainer.model.to("cpu")
            del trainer
            trainer = None
            gc.collect()
            torch.cuda.empty_cache()

            run_rows = []
            for test_ctx, predictions, inference_time, metrics, pred_path in evals:
                bert_f1 = compute_bertscore_with_fallback(predictions, references, runtime.bertscore_batch_start)
                run_rows.append(
                    {
                        "Run": run_name,
                        "Model": model_name,
                        "Use_Context": use_context,
                        "Train_Context": train_ctx,
                        "Test_Context": test_ctx,
                        **metrics,
                        "BERTScore_F1": bert_f1,
                        "Validation_Loss": stats["Validation_Loss"],
                        "Perplexity": stats["Perplexity"],
                        "Training_Time_s": stats["Training_Time_s"],
                        "Inference_Time_s": inference_time,
                        "Inference_ms_per_sample": inference_time / len(predictions) * 1000,
                        "Peak_VRAM_GB": stats["Peak_VRAM_GB"],
                        "Train_Batch": stats["Train_Batch"],
                        "Grad_Accum": stats["Grad_Accum"],
                        "Effective_Batch": stats["Train_Batch"] * stats["Grad_Accum"],
                        "Precision": "BF16" if runtime.use_bf16 else "FP16",
                        "N_Test": len(references),
                        "Predictions_File": os.path.basename(pred_path),
                        "Status": "OK",
                        "HF_Repo": "",
                    }
                )

            if hf_push:
                try:
                    repo_id = push_run_to_hub(
                        hf_api, hf_owner, run_name, cfg, use_context, best_dir,
                        [e[4] for e in evals], predict_script_path, run_rows,
                    )
                    for row in run_rows:
                        row["HF_Repo"] = repo_id
                    print(f"Pushed to https://huggingface.co/{repo_id}")
                except Exception as e:
                    print(f"push_to_hub failed for {run_name}: {e}")

            all_results.extend(run_rows)

        except Exception as e:
            print(f"ERROR {run_name}: {type(e).__name__}: {e}")
            traceback.print_exc()
            all_results.append(
                {
                    "Run": run_name,
                    "Model": model_name,
                    "Use_Context": use_context,
                    "Status": f"ERROR: {type(e).__name__}: {e}",
                }
            )
        finally:
            if trainer is not None:
                trainer.model.to("cpu")
            trainer = tokenizer = None
            gc.collect()
            torch.cuda.empty_cache()

        pd.DataFrame(all_results).to_csv(
            os.path.join(config.OUTPUT_ROOT, "results_summary.csv"), index=False, encoding="utf-8-sig"
        )

    results_df = pd.DataFrame(all_results)
    ok = results_df[results_df["Status"] == "OK"] if "Status" in results_df.columns else pd.DataFrame()
    if len(ok):
        ok = ok.sort_values("Token_F1", ascending=False).reset_index(drop=True)
        cols = [c for c in ("Run", "Train_Context", "Test_Context", "Exact_Match", "Token_F1", "ROUGE_L_F1",
                            "BLEU", "chrF", "BERTScore_F1") if c in ok.columns]
        print(ok[cols].round(2).to_string())
        print("Best run by Token F1:", ok.iloc[0]["Run"], "| test context:", ok.iloc[0].get("Test_Context"))
    else:
        print("No run completed successfully.")

    print("Output folder:", config.OUTPUT_ROOT)


if __name__ == "__main__":
    main()
