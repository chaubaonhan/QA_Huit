import re
import time

import torch
from tqdm.auto import tqdm

from . import config
from .data import clean_text, make_source
from .runtime import Runtime
from .utils import is_cuda_oom


def clean_prediction(text: str, mode: str) -> str:
    text = clean_text(text)
    if mode == "word":
        text = text.replace("_", " ")
    text = re.sub(r"^(trả lời|answer)\s*:\s*", "", text, flags=re.I)
    return clean_text(text)


@torch.inference_mode()
def generate_predictions(model, tokenizer, cfg: dict, use_context: bool, test_ds, runtime: Runtime):
    model.eval()
    model.config.use_cache = True
    mode = cfg["preprocess"]
    device = next(model.parameters()).device

    predictions = []
    total_time = 0.0
    i = 0
    gen_bs = runtime.gen_batch_start
    pbar = tqdm(total=len(test_ds), desc="generate")

    while i < len(test_ds):
        current_bs = min(gen_bs, len(test_ds) - i)
        if current_bs <= 0:
            break
        try:
            batch = test_ds[i:i + current_bs]
            sources = [make_source({k: batch[k][j] for k in batch}, mode, use_context) for j in range(current_bs)]
            enc = tokenizer(sources, return_tensors="pt", padding=True, truncation=True, max_length=config.MAX_SOURCE_LENGTH)
            enc = {k: v.to(device) for k, v in enc.items()}

            torch.cuda.synchronize()
            t0 = time.time()
            amp_dtype = torch.bfloat16 if runtime.use_bf16 else torch.float16
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True):
                out = model.generate(
                    **enc, max_new_tokens=config.GEN_MAX_NEW_TOKENS, do_sample=False, num_beams=1, use_cache=True
                )
            torch.cuda.synchronize()
            total_time += time.time() - t0

            try:
                decoded = tokenizer.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            except TypeError:
                decoded = tokenizer.batch_decode(out, skip_special_tokens=True)

            predictions.extend(clean_prediction(x, mode) for x in decoded)
            i += current_bs
            pbar.update(current_bs)

        except Exception as e:
            if not is_cuda_oom(e):
                pbar.close()
                raise
            torch.cuda.empty_cache()
            if gen_bs == 1:
                pbar.close()
                raise RuntimeError(f"Generation OOM at sample {i}") from e
            gen_bs = max(1, gen_bs // 2)

    pbar.close()
    return predictions, total_time
