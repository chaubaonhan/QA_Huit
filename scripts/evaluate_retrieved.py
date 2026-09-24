"""Re-evaluate already fine-tuned with-context checkpoints on the test set, feeding the generator either the
annotated reference context ("gold") or the BGE-M3 + FAISS + reranker output ("retrieved").

No retraining is needed, so the released Hub checkpoints can be checked directly:

    python scripts/evaluate_retrieved.py --model <owner>/vi-qa-vit5-base-ctx --base ViT5-base
    python scripts/evaluate_retrieved.py --model outputs/ViT5-base__ctx/best_model --base ViT5-base \
        --test_context retrieved

Results are appended to outputs/retrieved_eval_summary.csv; predictions (with the exact context shown
to the model) go to outputs/eval_<model>__test-<context>_predictions.csv.
"""

import argparse
import gc
import os
import sys

import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from viqa import config
from viqa.data import clean_text, load_and_clean_dataset
from viqa.generation import generate_predictions
from viqa.metrics import calculate_metrics, compute_bertscore_with_fallback
from viqa.modeling import set_forced_bos_for_lang
from viqa.retrieval import attach_retrieved_context, build_retriever
from viqa.runtime import detect_runtime, seed_everything
from viqa.utils import safe_name_of


def load_checkpoint(path_or_repo: str, cfg: dict):
    tokenizer = None
    for kwargs in (dict(use_fast=cfg["fast_tokenizer"]), dict(use_fast=False)):
        try:
            tokenizer = AutoTokenizer.from_pretrained(path_or_repo, **kwargs)
            break
        except Exception as e:
            last = e
    if tokenizer is None:
        raise RuntimeError(f"Could not load tokenizer from {path_or_repo}: {last}")
    lang_code = cfg.get("lang_code")
    if lang_code:
        for attr in ("src_lang", "tgt_lang"):
            if hasattr(tokenizer, attr):
                setattr(tokenizer, attr, lang_code)

    model = AutoModelForSeq2SeqLM.from_pretrained(path_or_repo)
    set_forced_bos_for_lang(model, tokenizer, lang_code)
    return tokenizer, model.to("cuda")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="Hub repo id or local checkpoint of a *with-context* run")
    p.add_argument("--base", required=True, choices=list(config.MODELS), help="base model the checkpoint came from")
    p.add_argument("--test_context", nargs="+", default=["gold", "retrieved"], choices=list(config.CONTEXT_FIELDS))
    p.add_argument("--top_k", type=int, default=None)
    p.add_argument("--top_n", type=int, default=None)
    p.add_argument("--no_rerank", action="store_true")
    p.add_argument("--no_bertscore", action="store_true")
    args = p.parse_args()

    runtime = detect_runtime()
    seed_everything()
    cfg = config.MODELS[args.base]

    dataset = load_and_clean_dataset()
    retrieval_metrics = {}
    if "retrieved" in args.test_context:
        retriever = build_retriever(
            dataset, top_k=args.top_k, top_n=args.top_n, use_reranker=False if args.no_rerank else None
        )
        dataset, retrieval_metrics = attach_retrieved_context(dataset, ["test"], retriever)

    test = dataset["test"]
    references = [clean_text(x) for x in test["answer"]]
    tokenizer, model = load_checkpoint(args.model, cfg)

    evals = []
    for src in args.test_context:
        field = config.CONTEXT_FIELDS[src]
        preds, t = generate_predictions(model, tokenizer, cfg, True, test, runtime, field)
        metrics = calculate_metrics(preds, references)

        out = {c: test[c] for c in ("question_id", "question_text", "topic", "subtopic") if c in test.column_names}
        out["context_given_to_model"] = test[field]
        out["test_context_source"] = [src] * len(preds)
        out["reference_answer"] = references
        out["prediction"] = preds
        os.makedirs(config.OUTPUT_ROOT, exist_ok=True)
        pred_path = os.path.join(config.OUTPUT_ROOT, f"eval_{safe_name_of(args.model)}__test-{src}_predictions.csv")
        pd.DataFrame(out).to_csv(pred_path, index=False, encoding="utf-8-sig")
        evals.append((src, preds, t, metrics, pred_path))
        print(f"[{src}] Token_F1={metrics['Token_F1']:.2f} EM={metrics['Exact_Match']:.2f}")

    model.to("cpu")
    del model
    gc.collect()
    torch.cuda.empty_cache()

    rows = []
    for src, preds, t, metrics, pred_path in evals:
        row = {"Checkpoint": args.model, "Model": args.base, "Test_Context": src, **metrics}
        if not args.no_bertscore:
            row["BERTScore_F1"] = compute_bertscore_with_fallback(preds, references, runtime.bertscore_batch_start)
        row["Inference_ms_per_sample"] = t / len(preds) * 1000
        row["N_Test"] = len(preds)
        if src == "retrieved" and "test" in retrieval_metrics:
            rm = retrieval_metrics["test"]
            row["Retriever_Gold_In_Input"] = rm["Gold_Passage_In_Generator_Input"]
            row["Retriever_Top_K"], row["Retriever_Top_N"] = rm["Top_K"], rm["Top_N"]
            row["Reranker"] = rm["rerank_model_id"]
        row["Predictions_File"] = pred_path
        rows.append(row)

    summary = os.path.join(config.OUTPUT_ROOT, "retrieved_eval_summary.csv")
    df = pd.DataFrame(rows)
    if os.path.exists(summary):
        df = pd.concat([pd.read_csv(summary), df], ignore_index=True)
    df.to_csv(summary, index=False, encoding="utf-8-sig")
    print(pd.DataFrame(rows).drop(columns=["Predictions_File"]).round(2).to_string())
    print("Summary:", summary)


if __name__ == "__main__":
    main()
