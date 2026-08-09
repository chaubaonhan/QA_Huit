import re
import unicodedata
from collections import Counter

import numpy as np
import sacrebleu


def normalize_answer(text) -> str:
    text = unicodedata.normalize("NFC", str(text)).replace("_", " ").lower().strip()
    text = "".join(" " if unicodedata.category(c).startswith("P") else c for c in text)
    return re.sub(r"\s+", " ", text).strip()


def exact_match(pred, ref) -> float:
    return float(normalize_answer(pred) == normalize_answer(ref))


def token_f1(pred, ref) -> float:
    p, r = normalize_answer(pred).split(), normalize_answer(ref).split()
    if not p and not r:
        return 1.0
    if not p or not r:
        return 0.0
    same = sum((Counter(p) & Counter(r)).values())
    if same == 0:
        return 0.0
    precision, recall = same / len(p), same / len(r)
    return 2 * precision * recall / (precision + recall)


def _ngrams(tokens, n):
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def rouge_n_f1(pred, ref, n) -> float:
    p, r = normalize_answer(pred).split(), normalize_answer(ref).split()
    pc, rc = Counter(_ngrams(p, n)), Counter(_ngrams(r, n))
    if not pc or not rc:
        return 0.0
    overlap = sum((pc & rc).values())
    precision, recall = overlap / sum(pc.values()), overlap / sum(rc.values())
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def _lcs_len(a, b):
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


def rouge_l_f1(pred, ref) -> float:
    p, r = normalize_answer(pred).split(), normalize_answer(ref).split()
    if not p or not r:
        return 0.0
    lcs = _lcs_len(p, r)
    precision, recall = lcs / len(p), lcs / len(r)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def calculate_metrics(predictions, references) -> dict:
    if not predictions or not references or len(predictions) != len(references):
        raise ValueError("predictions/references must be non-empty and of equal length")

    em = np.mean([exact_match(p, r) for p, r in zip(predictions, references)])
    f1 = np.mean([token_f1(p, r) for p, r in zip(predictions, references)])
    r1 = np.mean([rouge_n_f1(p, r, 1) for p, r in zip(predictions, references)])
    r2 = np.mean([rouge_n_f1(p, r, 2) for p, r in zip(predictions, references)])
    rl = np.mean([rouge_l_f1(p, r) for p, r in zip(predictions, references)])

    try:
        bleu = sacrebleu.corpus_bleu(predictions, [references]).score
    except Exception:
        bleu = 0.0
    try:
        chrf = sacrebleu.corpus_chrf(predictions, [references]).score
    except Exception:
        chrf = 0.0

    return {
        "Exact_Match": em * 100,
        "Token_F1": f1 * 100,
        "ROUGE_1_F1": r1 * 100,
        "ROUGE_2_F1": r2 * 100,
        "ROUGE_L_F1": rl * 100,
        "BLEU": bleu,
        "chrF": chrf,
    }


def compute_bertscore_with_fallback(predictions, references, batch_start: int) -> float:
    if not predictions or not references:
        return 0.0

    import gc

    import torch
    from bert_score import score as bert_score

    bs = batch_start
    while bs >= 1:
        try:
            P, R, F1 = bert_score(
                predictions, references, model_type="xlm-roberta-base", batch_size=bs, device="cuda", verbose=False
            )
            value = float(F1.mean().item()) * 100
            del P, R, F1
            gc.collect()
            torch.cuda.empty_cache()
            return value
        except Exception as e:
            msg = str(e).lower()
            if not (("out of memory" in msg and "cuda" in msg) or "cublas_status_alloc_failed" in msg):
                raise
            gc.collect()
            torch.cuda.empty_cache()
            bs //= 2

    raise RuntimeError("BERTScore OOM even at batch=1")
