import re


def run_name_of(model_name: str, use_context: bool) -> str:
    return f"{model_name}__{'ctx' if use_context else 'noctx'}"


def safe_name_of(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def is_cuda_oom(exc: Exception) -> bool:
    import torch

    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    msg = str(exc).lower()
    return ("out of memory" in msg and "cuda" in msg) or "cublas_status_alloc_failed" in msg
