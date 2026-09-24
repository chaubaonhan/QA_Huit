import argparse
import re
import unicodedata

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

_VNCORENLP = None


def clean_text(text) -> str:
    text = unicodedata.normalize("NFC", str(text))
    return re.sub(r"\s+", " ", text).strip()


def word_segment(text: str) -> str:
    global _VNCORENLP
    try:
        import os

        import py_vncorenlp

        if _VNCORENLP is None:
            save_dir = os.path.join(os.path.expanduser("~"), ".cache", "vncorenlp")
            os.makedirs(save_dir, exist_ok=True)
            jar = os.path.join(save_dir, "VnCoreNLP-1.2.jar")
            if not os.path.exists(jar):
                py_vncorenlp.download_model(save_dir=save_dir)
            _VNCORENLP = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=save_dir)
        return " ".join(_VNCORENLP.word_segment(text))
    except Exception as e:
        print(f"word segmentation unavailable ({e}); using raw text instead.")
        return text


def build_source(question: str, context: str, mode: str, use_context: bool) -> str:
    q = clean_text(question)
    if mode == "word":
        q = word_segment(q)
    if use_context and context:
        c = clean_text(context)
        if mode == "word":
            c = word_segment(c)
        return f"Câu hỏi: {q}\nNgữ cảnh: {c}"
    return f"Câu hỏi: {q}"


def retrieve_context(question: str, index_dir: str, top_k: int, top_n: int, rerank: bool, rerank_model: str = "") -> str:
    """BGE-M3 -> FAISS -> reranker, using an index directory built by scripts/build_index.py."""
    import json
    import os

    import faiss
    import numpy as np
    from sentence_transformers import CrossEncoder, SentenceTransformer

    with open(os.path.join(index_dir, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    with open(os.path.join(index_dir, "passages.jsonl"), encoding="utf-8") as f:
        passages = [json.loads(line) for line in f if line.strip()]
    index = faiss.read_index(os.path.join(index_dir, "index.faiss"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    query = clean_text(question)
    encoder = SentenceTransformer(meta["embed_model_id"], device=device)
    encoder.max_seq_length = int(meta.get("embed_max_length", 512))
    q = encoder.encode([query], normalize_embeddings=True, convert_to_numpy=True)
    _, ids = index.search(np.ascontiguousarray(q, dtype=np.float32), max(1, min(top_k, index.ntotal)))
    candidates = [int(i) for i in ids[0] if i >= 0]

    if rerank and candidates:
        reranker = CrossEncoder(rerank_model or meta.get("rerank_model_id", "BAAI/bge-reranker-v2-m3"),
                                max_length=512, device=device)
        scores = np.asarray(reranker.predict([[query, passages[i]["text"]] for i in candidates])).reshape(-1)
        order = sorted(range(len(candidates)), key=lambda j: (-float(scores[j]), j))
        candidates = [candidates[j] for j in order]

    return " ".join(passages[i]["text"] for i in candidates[:top_n])


def clean_prediction(text: str, mode: str) -> str:
    text = clean_text(text)
    if mode == "word":
        text = text.replace("_", " ")
    text = re.sub(r"^(trả lời|answer)\s*:\s*", "", text, flags=re.I)
    return clean_text(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HF Hub repo id or a local checkpoint path")
    parser.add_argument("--question", required=True)
    parser.add_argument("--context", default="")
    parser.add_argument("--mode", choices=["raw", "syllable", "word"], default="raw")
    parser.add_argument("--use_context", action="store_true")
    parser.add_argument("--max_source_length", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--index_dir", default="",
                        help="index built by scripts/build_index.py; used to retrieve the context when "
                             "--use_context is set and --context is empty")
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--top_n", type=int, default=3)
    parser.add_argument("--no_rerank", action="store_true")
    parser.add_argument("--rerank_model", default="")
    args = parser.parse_args()

    if args.use_context and not args.context:
        if not args.index_dir:
            parser.error("--use_context needs either --context or --index_dir")
        args.context = retrieve_context(
            args.question, args.index_dir, args.top_k, args.top_n, not args.no_rerank, args.rerank_model
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    source = build_source(args.question, args.context, args.mode, args.use_context)
    enc = tokenizer(source, return_tensors="pt", truncation=True, max_length=args.max_source_length)
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.inference_mode():
        out = model.generate(
            **enc, max_new_tokens=args.max_new_tokens, num_beams=args.num_beams, do_sample=False
        )

    try:
        decoded = tokenizer.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    except TypeError:
        decoded = tokenizer.batch_decode(out, skip_special_tokens=True)

    answer = clean_prediction(decoded[0], args.mode)
    print("Source:", source.replace("\n", " | "))
    print("Answer:", answer)


if __name__ == "__main__":
    main()
