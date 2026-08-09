import os

from transformers import AutoTokenizer

from . import config


def load_tokenizer(cfg: dict):
    model_id = cfg["id"]
    prefer_fast = cfg["fast_tokenizer"]
    cache_dir = os.path.join(config.TOKENIZER_ROOT, model_id.replace("/", "__"))
    os.makedirs(cache_dir, exist_ok=True)

    attempts = [
        dict(use_fast=prefer_fast, trust_remote_code=True),
        dict(use_fast=False, trust_remote_code=True),
        dict(use_fast=False),
        dict(use_fast=False, legacy=True),
    ]

    tok = None
    last_err = None
    for kwargs in attempts:
        try:
            tok = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir, **kwargs)
            print(f"Loaded tokenizer for {model_id} ({kwargs})")
            break
        except Exception as e:
            last_err = e
    if tok is None:
        raise RuntimeError(f"Failed to load tokenizer for {model_id}: {last_err}")

    if tok.pad_token_id is None:
        if tok.eos_token_id is not None:
            tok.pad_token = tok.eos_token
        elif tok.bos_token_id is not None:
            tok.pad_token = tok.bos_token
            tok.pad_token_id = tok.bos_token_id
        else:
            raise ValueError(f"Tokenizer for {model_id} has no pad/eos/bos token")

    lang_code = cfg.get("lang_code")
    if lang_code:
        if hasattr(tok, "src_lang"):
            tok.src_lang = lang_code
        if hasattr(tok, "tgt_lang"):
            tok.tgt_lang = lang_code

    return tok
