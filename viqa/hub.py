import os
import re

from . import config


def hf_login():
    token = os.environ.get("HF_TOKEN")
    if not token:
        try:
            from google.colab import userdata

            token = userdata.get("HF_TOKEN")
        except Exception:
            token = None

    from huggingface_hub import HfApi, login

    if token:
        login(token=token)
    else:
        login()
    return HfApi()


_METRIC_KEYS = ("Exact_Match", "Token_F1", "ROUGE_1_F1", "ROUGE_2_F1", "ROUGE_L_F1", "BLEU", "chrF", "BERTScore_F1")
_CTX_LABEL = {
    "none": "question only",
    "gold": "annotated reference context (oracle)",
    "retrieved": "retrieved context (BGE-M3 + FAISS + reranker)",
}


def build_model_card(run_name, repo_id, cfg, use_context, result_rows) -> str:
    if isinstance(result_rows, dict):
        result_rows = [result_rows]
    train_ctx = result_rows[0].get("Train_Context", "gold" if use_context else "none") if result_rows else "none"
    lines = [
        f"# {run_name}",
        "",
        f"Fine-tuned from [`{cfg['id']}`](https://huggingface.co/{cfg['id']}) for Vietnamese",
        f"question answering on `{config.DATASET_ID}`.",
        "",
        f"- Input format: **{'context + question' if use_context else 'question only'}**",
        f"- Context used during fine-tuning: **{_CTX_LABEL.get(train_ctx, train_ctx)}**",
        f"- Preprocess mode: `{cfg['preprocess']}`",
    ]
    if use_context and config.RETRIEVAL_ENABLED:
        lines += [
            f"- Retriever: `{config.EMBED_MODEL_ID}` dense embeddings, FAISS `IndexFlatIP`, top-{config.RETRIEVAL_TOP_K}"
            f" candidates, reranked by `{config.RERANK_MODEL_ID if config.USE_RERANKER else 'none'}`,"
            f" top-{config.RERANK_TOP_N} passages given to the generator",
        ]

    ctxs = [r.get("Test_Context", "none") for r in result_rows]
    lines += ["", "## Metrics (test set)", "", "| Metric | " + " | ".join(_CTX_LABEL.get(c, c) for c in ctxs) + " |",
              "|---|" + "---|" * len(ctxs)]
    for key in _METRIC_KEYS:
        vals = [r.get(key) for r in result_rows]
        if any(isinstance(v, (int, float)) for v in vals):
            lines.append(f"| {key} | " + " | ".join(f"{v:.2f}" if isinstance(v, (int, float)) else "-" for v in vals) + " |")
    if "gold" in ctxs:
        lines += ["", "The *annotated reference context* column is an oracle upper bound; the *retrieved context*"
                      " column is the end-to-end setting where the context comes from the retriever."]

    ctx_flag = ' --use_context --index_dir <faiss_index_dir>' if use_context else ""
    lines += [
        "",
        "## Inference",
        "",
        "`predict.py` is included in this repo:",
        "",
        "```bash",
        f"python predict.py --model {repo_id} --mode {cfg['preprocess']}{ctx_flag} \\",
        '    --question "Việt Nam có bao nhiêu tỉnh thành?"',
        "```",
    ]
    if use_context:
        lines += ["", "`<faiss_index_dir>` is produced by `scripts/build_index.py` in "
                      "[the GitHub repo](https://github.com/chaubaonhan/QA_Huit); alternatively pass `--context \"...\"`."]
    return "\n".join(lines)


def push_run_to_hub(api, owner, run_name, cfg, use_context, best_dir, pred_csv_paths, predict_script_path, result_rows):
    safe = re.sub(r"[^A-Za-z0-9]+", "-", run_name).strip("-").lower()
    repo_id = f"{owner}/{config.HF_REPO_PREFIX}-{safe}"
    if isinstance(pred_csv_paths, str):
        pred_csv_paths = [pred_csv_paths]

    api.create_repo(repo_id=repo_id, private=config.HF_PRIVATE_REPOS, exist_ok=True, repo_type="model")
    api.upload_folder(repo_id=repo_id, folder_path=best_dir, path_in_repo=".", commit_message=f"Upload {run_name}")

    for pred_csv_path in pred_csv_paths:
        if os.path.exists(pred_csv_path):
            api.upload_file(
                repo_id=repo_id,
                path_or_fileobj=pred_csv_path,
                path_in_repo=os.path.basename(pred_csv_path),
                commit_message="Add test-set predictions",
            )
    if os.path.exists(predict_script_path):
        api.upload_file(
            repo_id=repo_id, path_or_fileobj=predict_script_path, path_in_repo="predict.py", commit_message="Add predict.py"
        )

    readme = build_model_card(run_name, repo_id, cfg, use_context, result_rows)
    api.upload_file(
        repo_id=repo_id, path_or_fileobj=readme.encode("utf-8"), path_in_repo="README.md", commit_message="Add model card"
    )
    return repo_id
