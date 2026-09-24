import os
import re
import unicodedata
from functools import lru_cache

from datasets import load_dataset

from . import config

_VNCORENLP = None


def clean_text(text) -> str:
    text = unicodedata.normalize("NFC", str(text))
    return re.sub(r"\s+", " ", text).strip()


def _get_vncorenlp():
    global _VNCORENLP
    if _VNCORENLP is not None:
        return _VNCORENLP
    try:
        import py_vncorenlp

        save_dir = os.path.join(config.OUTPUT_ROOT, "vncorenlp")
        os.makedirs(save_dir, exist_ok=True)
        jar = os.path.join(save_dir, "VnCoreNLP-1.2.jar")
        if not os.path.exists(jar):
            py_vncorenlp.download_model(save_dir=save_dir)
        _VNCORENLP = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=save_dir)
    except Exception:
        _VNCORENLP = None
    return _VNCORENLP


@lru_cache(maxsize=100000)
def _word_segment_cached(text: str) -> str:
    nlp = _get_vncorenlp()
    if nlp is None:
        return text
    try:
        return " ".join(nlp.word_segment(text))
    except Exception:
        return text


def preprocess_text(text, mode: str) -> str:
    text = clean_text(text)
    if mode != "word":
        return text
    return _word_segment_cached(text)


def make_source(example, mode: str, use_context: bool, context_field: str = "context") -> str:
    """Build the generator input.

    context_field selects which context is shown to the model: "context" (annotated reference
    context) or "retrieved_context" (output of viqa.retrieval).
    """
    q = preprocess_text(example["question_text"], mode)
    if use_context:
        if context_field not in example:
            raise KeyError(f"Column '{context_field}' not found; run viqa.retrieval.attach_retrieved_context first")
        c = preprocess_text(example[context_field] or "", mode)
        return f"Câu hỏi: {q}\nNgữ cảnh: {c}"
    return f"Câu hỏi: {q}"


def make_target(answer, mode: str) -> str:
    return preprocess_text(answer, mode)


def _is_valid(x) -> bool:
    q, a = x.get("question_text"), x.get("answer")
    return bool(q) and bool(a) and bool(str(q).strip()) and bool(str(a).strip())


def load_and_clean_dataset(dataset_id: str = config.DATASET_ID):
    raw = load_dataset(dataset_id)

    required = ["question_text", "answer", "context"]
    for split in ("train", "validation", "test"):
        if split not in raw:
            raise ValueError(f"Missing split: {split}")
        for col in required:
            if col not in raw[split].column_names:
                raise ValueError(f"Missing column '{col}' in split '{split}'")

    invalid_test = raw["test"].filter(lambda x: not _is_valid(x))
    if len(invalid_test):
        os.makedirs(config.OUTPUT_ROOT, exist_ok=True)
        invalid_test.to_pandas().to_csv(
            os.path.join(config.OUTPUT_ROOT, "invalid_test_samples.csv"), index=False, encoding="utf-8-sig"
        )

    return raw.filter(_is_valid)


def question_leakage_report(dataset) -> dict:
    def norm(x):
        x = unicodedata.normalize("NFC", str(x)).lower().strip()
        return re.sub(r"\s+", " ", x)

    train_q = set(map(norm, dataset["train"]["question_text"]))
    val_q = set(map(norm, dataset["validation"]["question_text"]))
    test_q = set(map(norm, dataset["test"]["question_text"]))
    return {
        "train_val": len(train_q & val_q),
        "train_test": len(train_q & test_q),
        "val_test": len(val_q & test_q),
    }


def prepare_tokenized(raw_ds, tokenizer, mode: str, use_context: bool, desc: str, context_field: str = "context"):
    def batch_fn(batch):
        n = len(batch["question_text"])
        sources, targets = [], []
        for i in range(n):
            ex = {k: batch[k][i] for k in batch.keys()}
            sources.append(make_source(ex, mode, use_context, context_field))
            targets.append(make_target(ex["answer"], mode))
        inputs = tokenizer(sources, max_length=config.MAX_SOURCE_LENGTH, truncation=True, padding=False)
        labels = tokenizer(text_target=targets, max_length=config.MAX_TARGET_LENGTH, truncation=True, padding=False)
        inputs["labels"] = labels["input_ids"]
        return inputs

    map_bs = 64 if mode == "word" else 512
    return raw_ds.map(batch_fn, batched=True, batch_size=map_bs, remove_columns=raw_ds.column_names, desc=desc)
