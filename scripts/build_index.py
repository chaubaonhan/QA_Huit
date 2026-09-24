"""Build the BGE-M3 + FAISS index, retrieve (and rerank) contexts for every question, and score the retriever.

Outputs (under VIQA_RETRIEVAL_ROOT, default outputs/retrieval/index-<fingerprint>/):
    passages.jsonl, index.faiss, meta.json      the index (reusable by scripts/predict.py --index_dir)
    retrieved_<split>_<hash>.jsonl              per-question candidates, rerank scores and final context
    retrieval_metrics.json                      Recall@k / MRR / answer coverage per split
and outputs/retrieval_metrics.csv.

    python scripts/build_index.py
    python scripts/build_index.py --splits test --top_k 20 --top_n 3
    python scripts/build_index.py --corpus_path my_corpus.jsonl --no_rerank
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from viqa import config
from viqa.data import load_and_clean_dataset
from viqa.retrieval import attach_retrieved_context, build_retriever


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    p.add_argument("--corpus_path", default=None, help="external corpus (.jsonl / .txt); default: dataset contexts")
    p.add_argument("--top_k", type=int, default=None, help=f"FAISS candidates (default {config.RETRIEVAL_TOP_K})")
    p.add_argument("--top_n", type=int, default=None, help=f"passages given to the generator (default {config.RERANK_TOP_N})")
    p.add_argument("--chunk_words", type=int, default=None, help=f"default {config.CHUNK_WORDS}; 0 = no chunking")
    p.add_argument("--chunk_overlap", type=int, default=None, help=f"default {config.CHUNK_OVERLAP}")
    p.add_argument("--no_rerank", action="store_true", help="dense retrieval only (ablation)")
    args = p.parse_args()

    if args.chunk_words is not None:
        config.CHUNK_WORDS = args.chunk_words
    if args.chunk_overlap is not None:
        config.CHUNK_OVERLAP = args.chunk_overlap

    dataset = load_and_clean_dataset()
    retriever = build_retriever(
        dataset, corpus_path=args.corpus_path, top_k=args.top_k, top_n=args.top_n,
        use_reranker=False if args.no_rerank else None,
    )
    dataset, metrics = attach_retrieved_context(dataset, args.splits, retriever)

    df = pd.DataFrame([{"Split": s, **m} for s, m in metrics.items()])
    os.makedirs(config.OUTPUT_ROOT, exist_ok=True)
    out = os.path.join(config.OUTPUT_ROOT, "retrieval_metrics.csv")
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(df.set_index("Split").drop(columns=["Results_File"], errors="ignore").round(2).T.to_string())
    print("Index dir:", retriever.index_dir)
    print("Metrics:", out)


if __name__ == "__main__":
    main()
