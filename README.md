# Vietnamese Seq2Seq QA Benchmark

Fine-tunes and benchmarks 5 encoder-decoder models — ViT5, BARTpho-syllable, mBART-large-50,
ByT5-small, mT5-base — on [`BaoNhan/FAIR_Dataset_QA`](https://huggingface.co/datasets/BaoNhan/FAIR_Dataset_QA),
each once with `question_text` only and once with `context + question_text` (10 runs total).
Every successful run is pushed to the Hugging Face Hub with a model card, test-set
predictions, and a standalone `predict.py`.

## Setup

```bash
pip install -r requirements.txt
```

A CUDA GPU is required. To push to the Hugging Face Hub, set a token:

```bash
export HF_TOKEN=hf_xxx
```

Without a token, `scripts/train.py` opens an interactive login prompt (Colab/Jupyter).
To skip pushing entirely:

```bash
export VIQA_HF_PUSH=0
```

## Retrieval pipeline (with-context runs)

For the with-context runs the generator can receive two different contexts, and the benchmark reports both:

| `Test_Context` | What the generator sees | Meaning |
|---|---|---|
| `gold` | the annotated `context` field of the dataset | oracle upper bound |
| `retrieved` | top-N passages from BGE-M3 → FAISS → BGE reranker | end-to-end RAG setting |

Retrieval never looks at the question's own annotated context: every question is searched against the
whole corpus. The annotated context is used only afterwards, to score the retriever.

1. **Corpus**: all distinct `context` values of the train/validation/test splits (or your own corpus via
   `VIQA_CORPUS_PATH`), split into overlapping word windows (`VIQA_CHUNK_WORDS=200`, `VIQA_CHUNK_OVERLAP=50`).
2. **Dense retrieval**: passages and questions are embedded with [`BAAI/bge-m3`](https://huggingface.co/BAAI/bge-m3)
   (dense vectors, L2-normalised) and searched with an exact FAISS `IndexFlatIP` (cosine), `VIQA_RETRIEVAL_TOP_K=20`.
3. **Reranking**: the candidates are rescored with the cross-encoder
   [`BAAI/bge-reranker-v2-m3`](https://huggingface.co/BAAI/bge-reranker-v2-m3); the best `VIQA_RERANK_TOP_N=3`
   passages are concatenated (best first) into `retrieved_context`.

Build the index and score the retriever on its own (Recall@k, MRR, answer coverage):

```bash
python scripts/build_index.py                      # all splits
python scripts/build_index.py --splits test --no_rerank   # dense-only ablation
```

Re-evaluate already released with-context checkpoints with retrieved vs. annotated context (no retraining):

```bash
python scripts/evaluate_retrieved.py --model <hf_repo_or_local_ckpt> --base ViT5-base
```

Outputs under `outputs/retrieval/index-<fingerprint>/`: `passages.jsonl`, `index.faiss`, `meta.json`, and one
`retrieved_<split>_<hash>.jsonl` per split with, for every question, the FAISS candidates, reranker scores, the
passages given to the generator and the rank of the annotated context. `outputs/retrieval_metrics.csv`
summarises the retriever.

| Variable | Default | Purpose |
|---|---|---|
| `VIQA_RETRIEVAL` | `1` | `0` = only evaluate with the annotated context |
| `VIQA_TRAIN_CONTEXT` | `gold` | context used to fine-tune with-context runs: `gold` or `retrieved` |
| `VIQA_EMBED_MODEL` / `VIQA_RERANK_MODEL` | `BAAI/bge-m3` / `BAAI/bge-reranker-v2-m3` | retriever models |
| `VIQA_USE_RERANKER` | `1` | `0` = dense retrieval only |
| `VIQA_RETRIEVAL_TOP_K` / `VIQA_RERANK_TOP_N` | `20` / `3` | FAISS candidates / passages given to the generator |
| `VIQA_CORPUS_PATH` | empty | external corpus (`.jsonl` with `id`,`text`, or `.txt` with blank-line-separated docs) |

## Train

```bash
python scripts/train.py
```

Outputs (checkpoints, prediction CSVs, `results_summary.csv`) go to `outputs/`
(override with `VIQA_OUTPUT_ROOT`).

## Inference

```bash
python scripts/predict.py --model <hf_repo_id_or_local_path> --mode raw \
    --question "Việt Nam có bao nhiêu tỉnh thành?" \
    --use_context --context "..."

# or let the retriever find the context (index from scripts/build_index.py)
python scripts/predict.py --model <hf_repo_id_or_local_path> --mode raw \
    --question "Việt Nam có bao nhiêu tỉnh thành?" \
    --use_context --index_dir outputs/retrieval/index-<fingerprint>
```

`--mode` and whether to pass `--use_context` depend on which run produced the
checkpoint — see that run's README on the Hub.

## Configuration

All dataset/model/hyperparameter settings live in `viqa/config.py`:

| Variable | Purpose |
|---|---|
| `MODELS` | model name -> `{id, preprocess, fast_tokenizer, lang_code?}` |
| `CONTEXT_MODES` | `[False, True]`, cross-producted with `MODELS` into `RUNS` |
| `NUM_EPOCHS`, `LEARNING_RATE`, `MAX_SOURCE_LENGTH`, ... | training hyperparameters |
| `HF_PUSH`, `HF_PRIVATE_REPOS`, `HF_REPO_PREFIX` | Hub push behaviour |

## Citation

A paper describing this benchmark is in preparation. If you use this code or the
released checkpoints, please cite it once it is available — the entry below is a
placeholder and will be updated with the final venue, authors, and year.

```bibtex
@misc{TODO_citation_key,
  title        = {TODO: paper title},
  author       = {TODO: author list},
  year         = {TODO},
  howpublished = {TODO: venue / arXiv preprint},
  note         = {Preprint in preparation},
  url          = {https://github.com/chaubaonhan/QA_Huit}
}
```

In the meantime, please cite the repository itself (see `CITATION.cff`).

## Project layout

```
viqa/
  config.py        dataset id, model list, hyperparameters
  runtime.py        GPU detection, batch-size heuristics, seeding
  data.py            dataset loading/cleaning, text preprocessing, tokenization of the dataset
  tokenization.py    tokenizer loading with fallbacks across transformers versions
  modeling.py        model loading (fp32, SDPA / dtype / flax-checkpoint fallbacks)
  training.py        TrainingArguments builder + train loop with CUDA-OOM backoff
  generation.py       batched generation with CUDA-OOM backoff
  metrics.py          Exact Match, token F1, ROUGE-1/2/L, BLEU, chrF, BERTScore
  retrieval.py        corpus chunking, BGE-M3 encoder, FAISS index, BGE reranker, retrieval metrics
  hub.py               Hugging Face Hub push + model card
scripts/
  train.py             entry point, wires the modules above into the benchmark loop
  predict.py           standalone inference script (optionally retrieves context), uploaded to each pushed repo
  build_index.py       builds the FAISS index, retrieves contexts for each split, scores the retriever
  evaluate_retrieved.py  re-evaluates existing checkpoints with retrieved vs. annotated context
```
