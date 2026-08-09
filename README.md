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
  hub.py               Hugging Face Hub push + model card
scripts/
  train.py             entry point, wires the modules above into the benchmark loop
  predict.py           standalone inference script, also uploaded to each pushed repo
```
