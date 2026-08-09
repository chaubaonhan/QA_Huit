import random
from dataclasses import dataclass

import numpy as np
import torch
from transformers import set_seed

from . import config


@dataclass
class Runtime:
    gpu_name: str
    vram_gb: float
    major: int
    minor: int
    is_l4: bool
    use_bf16: bool
    use_fp16: bool
    train_batch_candidates: list
    gen_batch_start: int
    bertscore_batch_start: int


def detect_runtime() -> Runtime:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required.")

    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    major, minor = torch.cuda.get_device_capability()
    use_bf16 = major >= 8 and torch.cuda.is_bf16_supported()
    use_fp16 = not use_bf16
    is_l4 = "L4" in gpu_name.upper()

    if major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    torch.backends.cudnn.benchmark = True
    try:
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
    except Exception:
        pass

    if is_l4:
        train_batch_candidates, gen_batch_start, bertscore_batch_start = [32, 16, 8, 4, 2, 1], 64, 64
    elif vram_gb >= 70:
        train_batch_candidates, gen_batch_start, bertscore_batch_start = [128, 64, 32, 16, 8, 4, 2, 1], 256, 256
    elif vram_gb >= 35:
        train_batch_candidates, gen_batch_start, bertscore_batch_start = [64, 32, 16, 8, 4, 2, 1], 128, 128
    elif vram_gb >= 20:
        train_batch_candidates, gen_batch_start, bertscore_batch_start = [32, 16, 8, 4, 2, 1], 64, 64
    else:
        train_batch_candidates, gen_batch_start, bertscore_batch_start = [8, 4, 2, 1], 16, 16

    return Runtime(
        gpu_name=gpu_name,
        vram_gb=vram_gb,
        major=major,
        minor=minor,
        is_l4=is_l4,
        use_bf16=use_bf16,
        use_fp16=use_fp16,
        train_batch_candidates=train_batch_candidates,
        gen_batch_start=gen_batch_start,
        bertscore_batch_start=bertscore_batch_start,
    )


def seed_everything(seed: int = config.SEED) -> None:
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
