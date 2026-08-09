import os

DATASET_ID = "BaoNhan/FAIR_Dataset_QA"

OUTPUT_ROOT = os.environ.get("VIQA_OUTPUT_ROOT", "outputs")
TOKENIZER_ROOT = os.environ.get("VIQA_TOKENIZER_ROOT", "tokenizer_cache")

SEED = 42
NUM_EPOCHS = 3
LEARNING_RATE = 3e-5
MAX_SOURCE_LENGTH = 512
MAX_TARGET_LENGTH = 128
GEN_MAX_NEW_TOKENS = 128
TARGET_EFFECTIVE_BATCH = 32

MODELS = {
    "ViT5-base": {"id": "VietAI/vit5-base", "preprocess": "raw", "fast_tokenizer": True},
    "BARTpho-syllable-base": {"id": "vinai/bartpho-syllable-base", "preprocess": "syllable", "fast_tokenizer": False},
    "mBART-large-50": {"id": "facebook/mbart-large-50", "preprocess": "raw", "fast_tokenizer": True, "lang_code": "vi_VN"},
    "ByT5-small": {"id": "google/byt5-small", "preprocess": "raw", "fast_tokenizer": False},
    "mT5-base": {"id": "google/mt5-base", "preprocess": "raw", "fast_tokenizer": True},
}

CONTEXT_MODES = [False, True]
RUNS = [(name, cfg, use_context) for name, cfg in MODELS.items() for use_context in CONTEXT_MODES]

HF_PUSH = os.environ.get("VIQA_HF_PUSH", "1") == "1"
HF_PRIVATE_REPOS = os.environ.get("VIQA_HF_PRIVATE", "1") == "1"
HF_REPO_PREFIX = os.environ.get("VIQA_HF_REPO_PREFIX", "vi-qa")
