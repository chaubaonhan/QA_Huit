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


def build_model_card(run_name, repo_id, cfg, use_context, result_row) -> str:
    ctx_txt = "context + question (with-context)" if use_context else "question only (no-context)"
    lines = [
        f"# {run_name}",
        "",
        f"Fine-tuned from [`{cfg['id']}`](https://huggingface.co/{cfg['id']}) for Vietnamese",
        f"question answering on `{config.DATASET_ID}`.",
        "",
        f"- Input format: **{ctx_txt}**",
        f"- Preprocess mode: `{cfg['preprocess']}`",
        "",
        "## Metrics (test set)",
        "",
        "| Metric | Value |",
        "|---|---|",
    ]
    for key in ("Exact_Match", "Token_F1", "ROUGE_1_F1", "ROUGE_2_F1", "ROUGE_L_F1", "BLEU", "chrF", "BERTScore_F1"):
        if key in result_row and isinstance(result_row[key], (int, float)):
            lines.append(f"| {key} | {result_row[key]:.2f} |")

    ctx_flag = ' --use_context --context "..."' if use_context else ""
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
    return "\n".join(lines)


def push_run_to_hub(api, owner, run_name, cfg, use_context, best_dir, pred_csv_path, predict_script_path, result_row):
    safe = re.sub(r"[^A-Za-z0-9]+", "-", run_name).strip("-").lower()
    repo_id = f"{owner}/{config.HF_REPO_PREFIX}-{safe}"

    api.create_repo(repo_id=repo_id, private=config.HF_PRIVATE_REPOS, exist_ok=True, repo_type="model")
    api.upload_folder(repo_id=repo_id, folder_path=best_dir, path_in_repo=".", commit_message=f"Upload {run_name}")

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

    readme = build_model_card(run_name, repo_id, cfg, use_context, result_row)
    api.upload_file(
        repo_id=repo_id, path_or_fileobj=readme.encode("utf-8"), path_in_repo="README.md", commit_message="Add model card"
    )
    return repo_id
