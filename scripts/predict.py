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
    args = parser.parse_args()

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
