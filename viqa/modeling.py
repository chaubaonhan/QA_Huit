import gc
import importlib.util
import subprocess
import sys

import torch
from transformers import AutoModelForSeq2SeqLM

from .runtime import Runtime


def _pip_install(*pkgs):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *pkgs])


def _ensure_flax_available():
    if importlib.util.find_spec("flax") is None or importlib.util.find_spec("jax") is None:
        _pip_install("flax", "jax", "jaxlib")


def _missing_pt_weights(exc: Exception) -> bool:
    msg = str(exc)
    return "pytorch_model.bin" in msg and "safetensors" in msg


def load_model_fp32(model_id: str, runtime: Runtime):
    load_kwargs = dict(low_cpu_mem_usage=True, dtype=torch.float32, trust_remote_code=True, device_map=None)

    def _from_pretrained(**extra):
        try:
            return AutoModelForSeq2SeqLM.from_pretrained(model_id, **load_kwargs, **extra)
        except TypeError as te:
            if "dtype" in str(te):
                alt = dict(load_kwargs)
                alt["torch_dtype"] = alt.pop("dtype")
                return AutoModelForSeq2SeqLM.from_pretrained(model_id, **alt, **extra)
            raise

    model = None
    if runtime.major >= 8:
        try:
            model = _from_pretrained(attn_implementation="sdpa")
        except Exception:
            gc.collect()

    if model is None:
        try:
            model = _from_pretrained()
        except OSError as e:
            if _missing_pt_weights(e):
                _ensure_flax_available()
                model = _from_pretrained(from_flax=True)
            else:
                raise

    if next(model.parameters()).dtype != torch.float32:
        raise RuntimeError(f"{model_id}: model is not FP32")

    model.config.use_cache = False
    if getattr(model.config, "decoder_start_token_id", None) is None:
        if getattr(model.config, "pad_token_id", None) is not None:
            model.config.decoder_start_token_id = model.config.pad_token_id
        elif getattr(model.config, "bos_token_id", None) is not None:
            model.config.decoder_start_token_id = model.config.bos_token_id
        else:
            model.config.decoder_start_token_id = 0

    return model


def set_forced_bos_for_lang(model, tokenizer, lang_code) -> None:
    if lang_code and hasattr(tokenizer, "lang_code_to_id"):
        try:
            model.config.forced_bos_token_id = tokenizer.lang_code_to_id[lang_code]
        except Exception:
            pass
