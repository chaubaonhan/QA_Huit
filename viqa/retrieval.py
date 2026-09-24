"""Retrieval pipeline used by the with-context runs.

    corpus documents -> word-window passages -> BGE-M3 dense embeddings (L2-normalised)
    -> FAISS IndexFlatIP (exact cosine search) -> top-K candidates per question
    -> BGE cross-encoder reranker -> top-N passages -> `retrieved_context`

When a split is evaluated with `retrieved_context`, the generator never sees the annotated
`context` field: it only receives passages chosen by the retriever from the whole corpus.
The annotated context is used afterwards only to *score* the retriever (Recall@k / MRR).
"""

import hashlib
import json
import os
import re
import time

import numpy as np

from . import config
from .data import clean_text
from .metrics import normalize_answer

GOLD_FIELD = "context"
RETRIEVED_FIELD = "retrieved_context"
RECALL_KS = (1, 3, 5, 10, 20, 50, 100)


def _key(text) -> str:
    return clean_text(text).lower()


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _free_cuda() -> None:
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------

def chunk_text(text, chunk_words: int, overlap: int) -> list:
    """Split a document into overlapping word windows (0 = keep the document whole)."""
    words = clean_text(text).split()
    if not words:
        return []
    if chunk_words <= 0 or len(words) <= chunk_words:
        return [" ".join(words)]
    step = max(1, chunk_words - max(0, overlap))
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start:start + chunk_words]))
        if start + chunk_words >= len(words):
            break
    return chunks


def _read_corpus_file(path: str) -> list:
    docs = []
    if path.endswith(".jsonl"):
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                text = obj.get("text") or obj.get("context") or obj.get("content")
                if text and clean_text(text):
                    docs.append((str(obj.get("id", i)), clean_text(text)))
    else:
        with open(path, encoding="utf-8") as f:
            blocks = re.split(r"\n\s*\n", f.read())
        for i, block in enumerate(blocks):
            if clean_text(block):
                docs.append((f"doc{i}", clean_text(block)))
    if not docs:
        raise ValueError(f"No documents found in corpus file {path}")
    return docs


def load_documents(dataset=None, corpus_path=None, splits=None) -> list:
    """Return [(doc_id, text)].

    Default corpus: every distinct `context` of the given splits. Each question is answered by
    searching this whole pool, not by looking up its own annotated context.
    """
    corpus_path = config.CORPUS_PATH if corpus_path is None else corpus_path
    if corpus_path:
        return _read_corpus_file(corpus_path)

    if dataset is None:
        raise ValueError("Either a dataset or a corpus_path is required")
    splits = config.CORPUS_SPLITS if splits is None else splits
    docs, seen = [], set()
    for split in splits:
        if split not in dataset:
            continue
        for ctx in dataset[split][GOLD_FIELD]:
            text = clean_text(ctx) if ctx else ""
            k = text.lower()
            if not text or k in seen:
                continue
            seen.add(k)
            docs.append((_sha1(k)[:12], text))
    if not docs:
        raise ValueError("The dataset has no non-empty contexts to build a corpus from")
    return docs


def build_passages(docs, chunk_words=None, overlap=None) -> list:
    chunk_words = config.CHUNK_WORDS if chunk_words is None else chunk_words
    overlap = config.CHUNK_OVERLAP if overlap is None else overlap
    passages, seen = [], set()
    for doc_id, text in docs:
        for chunk in chunk_text(text, chunk_words, overlap):
            k = chunk.lower()
            if k in seen:
                continue
            seen.add(k)
            passages.append({"pid": len(passages), "doc_id": doc_id, "text": chunk})
    return passages


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class DenseEncoder:
    """BGE-M3 dense encoder (CLS pooling + L2 normalisation, loaded through sentence-transformers)."""

    def __init__(self, model_id=None, device=None, max_length=None, batch_size=None):
        from sentence_transformers import SentenceTransformer

        self.model_id = model_id or config.EMBED_MODEL_ID
        self.device = device or _device()
        self.model = SentenceTransformer(self.model_id, device=self.device)
        self.model.max_seq_length = max_length or config.EMBED_MAX_LENGTH
        if self.device == "cuda":
            self.model.half()
        self.batch_size = batch_size or config.EMBED_BATCH_SIZE

    def encode(self, texts) -> np.ndarray:
        texts = list(texts)
        emb = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 256,
        )
        return np.ascontiguousarray(emb, dtype=np.float32)


class CrossEncoderReranker:
    """BGE reranker (cross-encoder); a higher score means a more relevant (question, passage) pair."""

    def __init__(self, model_id=None, device=None, max_length=None, batch_size=None):
        from sentence_transformers import CrossEncoder

        self.model_id = model_id or config.RERANK_MODEL_ID
        self.device = device or _device()
        self.model = CrossEncoder(self.model_id, max_length=max_length or config.RERANK_MAX_LENGTH, device=self.device)
        if self.device == "cuda":
            try:
                self.model.model.half()
            except Exception:
                pass
        self.batch_size = batch_size or config.RERANK_BATCH_SIZE

    def score(self, pairs) -> np.ndarray:
        pairs = [list(p) for p in pairs]
        if not pairs:
            return np.zeros(0, dtype=np.float32)
        scores = self.model.predict(
            pairs, batch_size=self.batch_size, show_progress_bar=len(pairs) > 2048, convert_to_numpy=True
        )
        return np.asarray(scores, dtype=np.float32).reshape(-1)


# ---------------------------------------------------------------------------
# FAISS index
# ---------------------------------------------------------------------------

def corpus_fingerprint(passages, embed_model_id: str, max_length: int) -> str:
    h = hashlib.sha1()
    h.update(f"{embed_model_id}|{max_length}\n".encode("utf-8"))
    for p in passages:
        h.update(p["text"].encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


class PassageIndex:
    def __init__(self, passages, index, meta):
        self.passages = passages
        self.index = index
        self.meta = meta

    @classmethod
    def build(cls, passages, encoder, extra_meta=None):
        import faiss

        if not passages:
            raise ValueError("Cannot build an index over an empty corpus")
        t0 = time.time()
        embs = encoder.encode([p["text"] for p in passages])
        index = faiss.IndexFlatIP(embs.shape[1])
        index.add(embs)
        meta = {
            "embed_model_id": encoder.model_id,
            "embed_max_length": getattr(getattr(encoder, "model", None), "max_seq_length", config.EMBED_MAX_LENGTH),
            "rerank_model_id": config.RERANK_MODEL_ID,
            "faiss_index": "IndexFlatIP (exact search, cosine on L2-normalised vectors)",
            "dim": int(embs.shape[1]),
            "n_passages": len(passages),
            "n_documents": len({p["doc_id"] for p in passages}),
            "encode_time_s": round(time.time() - t0, 2),
        }
        meta.update(extra_meta or {})
        return cls(passages, index, meta)

    def save(self, directory: str) -> None:
        import faiss

        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, "passages.jsonl"), "w", encoding="utf-8") as f:
            for p in self.passages:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        faiss.write_index(self.index, os.path.join(directory, "index.faiss"))
        with open(os.path.join(directory, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(self.meta, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, directory: str):
        import faiss

        with open(os.path.join(directory, "passages.jsonl"), encoding="utf-8") as f:
            passages = [json.loads(line) for line in f if line.strip()]
        index = faiss.read_index(os.path.join(directory, "index.faiss"))
        with open(os.path.join(directory, "meta.json"), encoding="utf-8") as f:
            meta = json.load(f)
        if index.ntotal != len(passages):
            raise RuntimeError(f"Corrupt index in {directory}: {index.ntotal} vectors vs {len(passages)} passages")
        return cls(passages, index, meta)

    def search(self, query_embs: np.ndarray, k: int):
        k = max(1, min(k, self.index.ntotal))
        return self.index.search(np.ascontiguousarray(query_embs, dtype=np.float32), k)


def get_or_build_index(passages, encoder=None, root=None, extra_meta=None):
    """Load the cached index for this exact corpus + embedding model, or build and cache it."""
    root = config.RETRIEVAL_ROOT if root is None else root
    model_id = encoder.model_id if encoder is not None else config.EMBED_MODEL_ID
    fp = corpus_fingerprint(passages, model_id, config.EMBED_MAX_LENGTH)
    directory = os.path.join(root, f"index-{fp[:12]}")
    if all(os.path.exists(os.path.join(directory, f)) for f in ("passages.jsonl", "index.faiss", "meta.json")):
        print(f"Loading cached FAISS index from {directory}")
        return PassageIndex.load(directory), directory, encoder

    encoder = encoder or DenseEncoder(model_id)
    print(f"Encoding {len(passages)} passages with {model_id} ...")
    meta = {"fingerprint": fp, "chunk_words": config.CHUNK_WORDS, "chunk_overlap": config.CHUNK_OVERLAP}
    meta.update(extra_meta or {})
    index = PassageIndex.build(passages, encoder, meta)
    index.save(directory)
    print(f"Saved FAISS index ({index.index.ntotal} vectors, dim={index.meta['dim']}) to {directory}")
    return index, directory, encoder


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

class Retriever:
    def __init__(self, index: PassageIndex, index_dir: str = "", encoder=None, reranker=None,
                 top_k=None, top_n=None, use_reranker=None):
        self.index = index
        self.index_dir = index_dir
        self._encoder = encoder
        self._reranker = reranker
        self.top_k = config.RETRIEVAL_TOP_K if top_k is None else top_k
        self.top_n = config.RERANK_TOP_N if top_n is None else top_n
        self.use_reranker = config.USE_RERANKER if use_reranker is None else use_reranker
        if self.top_n > self.top_k:
            raise ValueError(f"top_n ({self.top_n}) must be <= top_k ({self.top_k})")

    @property
    def encoder(self):
        if self._encoder is None:
            self._encoder = DenseEncoder(self.index.meta.get("embed_model_id"))
        return self._encoder

    @property
    def reranker(self):
        if self._reranker is None:
            self._reranker = CrossEncoderReranker()
        return self._reranker

    def settings(self) -> dict:
        return {
            "index_fingerprint": self.index.meta.get("fingerprint", ""),
            "embed_model_id": self.index.meta.get("embed_model_id"),
            "rerank_model_id": self.reranker_id(),
            "top_k": self.top_k,
            "top_n": self.top_n,
        }

    def reranker_id(self) -> str:
        if not self.use_reranker:
            return "none"
        return self._reranker.model_id if self._reranker is not None else config.RERANK_MODEL_ID

    def retrieve(self, questions) -> list:
        queries = [clean_text(q) for q in questions]
        scores, ids = self.index.search(self.encoder.encode(queries), self.top_k)
        results = [
            {"dense": [(int(p), float(s)) for p, s in zip(ids[i], scores[i]) if p >= 0]} for i in range(len(queries))
        ]

        if self.use_reranker:
            pairs = [(queries[i], self.index.passages[pid]["text"]) for i, r in enumerate(results) for pid, _ in r["dense"]]
            rr = self.reranker.score(pairs)
            pos = 0
            for r in results:
                n = len(r["dense"])
                sc = rr[pos:pos + n]
                pos += n
                order = sorted(range(n), key=lambda j: (-float(sc[j]), j))  # ties keep dense order
                r["reranked"] = [(r["dense"][j][0], float(sc[j])) for j in order]
        else:
            for r in results:
                r["reranked"] = list(r["dense"])

        for r in results:
            r["final"] = r["reranked"][: self.top_n]
            r[RETRIEVED_FIELD] = " ".join(self.index.passages[pid]["text"] for pid, _ in r["final"])
        return results

    def release(self) -> None:
        self._encoder = None
        self._reranker = None
        _free_cuda()


def build_retriever(dataset=None, corpus_path=None, **kwargs) -> Retriever:
    docs = load_documents(dataset, corpus_path)
    passages = build_passages(docs)
    source = (corpus_path if corpus_path is not None else config.CORPUS_PATH) or (
        f"{config.DATASET_ID}:context[{','.join(config.CORPUS_SPLITS)}]"
    )
    print(f"Retrieval corpus: {len(docs)} documents -> {len(passages)} passages ({source})")
    index, directory, encoder = get_or_build_index(passages, extra_meta={"corpus_source": source})
    return Retriever(index, directory, encoder=encoder, **kwargs)


# ---------------------------------------------------------------------------
# Evaluation of the retriever itself
# ---------------------------------------------------------------------------

def _first_gold_rank(ranked, passages, gold_key: str):
    """1-based rank of the first passage that belongs to the annotated context (text match)."""
    if not gold_key:
        return None
    for rank, (pid, _) in enumerate(ranked, 1):
        pk = passages[pid]["text"].lower()
        if pk and (pk in gold_key or gold_key in pk):
            return rank
    return None


def _answer_in(text, answer) -> bool:
    a = normalize_answer(answer)
    return bool(a) and a in normalize_answer(text or "")


def compute_retrieval_metrics(results, gold_contexts, answers, passages, top_k: int, top_n: int) -> dict:
    dense_ranks, rerank_ranks = [], []
    for r, gold in zip(results, gold_contexts):
        gk = _key(gold) if gold else ""
        r["gold_rank_dense"] = _first_gold_rank(r["dense"], passages, gk)
        r["gold_rank_reranked"] = _first_gold_rank(r["reranked"], passages, gk)
        dense_ranks.append(r["gold_rank_dense"])
        rerank_ranks.append(r["gold_rank_reranked"])

    def recall(ranks, k):
        return 100.0 * float(np.mean([rk is not None and rk <= k for rk in ranks]))

    def mrr(ranks, k):
        return 100.0 * float(np.mean([1.0 / rk if rk is not None and rk <= k else 0.0 for rk in ranks]))

    m = {"N": len(results), "Top_K": top_k, "Top_N": top_n}
    for k in RECALL_KS:
        if k <= top_k:
            m[f"Dense_Recall@{k}"] = recall(dense_ranks, k)
            m[f"Rerank_Recall@{k}"] = recall(rerank_ranks, k)
    m[f"Dense_MRR@{top_k}"] = mrr(dense_ranks, top_k)
    m[f"Rerank_MRR@{top_k}"] = mrr(rerank_ranks, top_k)
    m["Gold_Passage_In_Generator_Input"] = recall(rerank_ranks, top_n)
    m["Answer_In_Retrieved_Context"] = 100.0 * float(
        np.mean([_answer_in(r[RETRIEVED_FIELD], a) for r, a in zip(results, answers)])
    )
    m["Answer_In_Gold_Context"] = 100.0 * float(np.mean([_answer_in(g, a) for g, a in zip(gold_contexts, answers)]))
    return m


# ---------------------------------------------------------------------------
# Dataset integration
# ---------------------------------------------------------------------------

def _cache_path(retriever: Retriever, split: str, questions) -> str:
    key = json.dumps(retriever.settings(), sort_keys=True) + "\n" + "\n".join(questions)
    return os.path.join(retriever.index_dir or config.RETRIEVAL_ROOT, f"retrieved_{split}_{_sha1(key)[:12]}.jsonl")


def _save_results(path, results, split_ds):
    has_qid = "question_id" in split_ds.column_names
    qids = split_ds["question_id"] if has_qid else [None] * len(results)
    with open(path, "w", encoding="utf-8") as f:
        for i, (r, q) in enumerate(zip(results, split_ds["question_text"])):
            rec = {
                "row": i,
                "question_id": qids[i],
                "question_text": q,
                "dense": r["dense"],
                "reranked": r["reranked"],
                "final_passage_ids": [pid for pid, _ in r["final"]],
                RETRIEVED_FIELD: r[RETRIEVED_FIELD],
                "gold_rank_dense": r.get("gold_rank_dense"),
                "gold_rank_reranked": r.get("gold_rank_reranked"),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _load_results(path, n, top_n):
    with open(path, encoding="utf-8") as f:
        recs = [json.loads(line) for line in f if line.strip()]
    if len(recs) != n:
        return None
    results = []
    for rec in recs:
        dense = [tuple(x) for x in rec["dense"]]
        reranked = [tuple(x) for x in rec["reranked"]]
        results.append(
            {"dense": dense, "reranked": reranked, "final": reranked[:top_n], RETRIEVED_FIELD: rec[RETRIEVED_FIELD]}
        )
    return results


def attach_retrieved_context(dataset, splits=("test",), retriever: Retriever = None, release: bool = True):
    """Add a `retrieved_context` column to the given splits and score the retriever on them.

    Returns (dataset, {split: metrics}). Results are cached as JSONL next to the FAISS index, one
    line per question, so the exact passages given to the generator can be released and audited.
    """
    retriever = retriever or build_retriever(dataset)
    all_metrics = {}
    for split in splits:
        ds = dataset[split]
        questions = [clean_text(q) for q in ds["question_text"]]
        path = _cache_path(retriever, split, questions)
        results = _load_results(path, len(ds), retriever.top_n) if os.path.exists(path) else None
        if results is not None:
            print(f"[{split}] loaded cached retrieval results from {path}")
        else:
            t0 = time.time()
            results = retriever.retrieve(questions)
            print(f"[{split}] retrieved {len(results)} questions in {time.time() - t0:.1f}s")

        metrics = compute_retrieval_metrics(
            results, ds[GOLD_FIELD], ds["answer"], retriever.index.passages, retriever.top_k, retriever.top_n
        )
        metrics.update({k: v for k, v in retriever.settings().items() if k != "index_fingerprint"})
        metrics["Results_File"] = path
        _save_results(path, results, ds)
        all_metrics[split] = metrics

        if RETRIEVED_FIELD in ds.column_names:
            ds = ds.remove_columns(RETRIEVED_FIELD)
        dataset[split] = ds.add_column(RETRIEVED_FIELD, [r[RETRIEVED_FIELD] for r in results])

    if retriever.index_dir:
        with open(os.path.join(retriever.index_dir, "retrieval_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(all_metrics, f, ensure_ascii=False, indent=2)
    if release:
        retriever.release()
    return dataset, all_metrics
