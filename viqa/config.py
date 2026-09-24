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

# ---------------------------------------------------------------------------
# Retrieval for the with-context runs:
#   BGE-M3 dense retrieval -> FAISS (exact inner product) -> BGE reranker -> top-N passages
# ---------------------------------------------------------------------------
RETRIEVAL_ENABLED = os.environ.get("VIQA_RETRIEVAL", "1") == "1"
EMBED_MODEL_ID = os.environ.get("VIQA_EMBED_MODEL", "BAAI/bge-m3")
RERANK_MODEL_ID = os.environ.get("VIQA_RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
USE_RERANKER = os.environ.get("VIQA_USE_RERANKER", "1") == "1"
RETRIEVAL_TOP_K = int(os.environ.get("VIQA_RETRIEVAL_TOP_K", "20"))  # FAISS candidates per question
RERANK_TOP_N = int(os.environ.get("VIQA_RERANK_TOP_N", "3"))  # passages passed to the generator
CHUNK_WORDS = int(os.environ.get("VIQA_CHUNK_WORDS", "200"))  # 0 = one passage per document
CHUNK_OVERLAP = int(os.environ.get("VIQA_CHUNK_OVERLAP", "50"))
EMBED_MAX_LENGTH = 512
EMBED_BATCH_SIZE = 32
RERANK_MAX_LENGTH = 512
RERANK_BATCH_SIZE = 32

# Retrieval corpus. Empty = the de-duplicated `context` field of CORPUS_SPLITS.
# Otherwise a .jsonl file ({"id": ..., "text": ...} per line) or a .txt file (documents separated by blank lines).
CORPUS_PATH = os.environ.get("VIQA_CORPUS_PATH", "")
CORPUS_SPLITS = ("train", "validation", "test")
RETRIEVAL_ROOT = os.environ.get("VIQA_RETRIEVAL_ROOT", os.path.join(OUTPUT_ROOT, "retrieval"))

# Which context the generator is fine-tuned with, and which contexts the test set is evaluated with.
#   "gold"      = annotated reference context (column `context`)       -> oracle upper bound
#   "retrieved" = BGE-M3 + FAISS + reranker output (`retrieved_context`) -> end-to-end RAG setting
CONTEXT_FIELDS = {"gold": "context", "retrieved": "retrieved_context"}
TRAIN_CONTEXT_SOURCE = os.environ.get("VIQA_TRAIN_CONTEXT", "gold")
TEST_CONTEXT_SOURCES = ["gold", "retrieved"] if RETRIEVAL_ENABLED else ["gold"]
