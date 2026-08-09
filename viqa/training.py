import gc
import inspect
import math
import os
import shutil
import time

import torch
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments

from . import config
from .data import prepare_tokenized
from .modeling import load_model_fp32, set_forced_bos_for_lang
from .runtime import Runtime
from .tokenization import load_tokenizer
from .utils import is_cuda_oom, safe_name_of


def _release_cuda_memory():
    gc.collect()
    torch.cuda.empty_cache()
    if hasattr(torch.cuda, "ipc_collect"):
        torch.cuda.ipc_collect()


def make_training_args(output_dir: str, train_bs: int, runtime: Runtime):
    grad_accum = max(1, config.TARGET_EFFECTIVE_BATCH // train_bs)
    eval_bs = min(64, max(train_bs, train_bs * 2))

    kwargs = dict(
        output_dir=output_dir,
        num_train_epochs=config.NUM_EPOCHS,
        learning_rate=config.LEARNING_RATE,
        per_device_train_batch_size=train_bs,
        per_device_eval_batch_size=eval_bs,
        gradient_accumulation_steps=grad_accum,
        weight_decay=0.01,
        warmup_ratio=0.05,
        lr_scheduler_type="linear",
        logging_steps=50,
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=runtime.use_fp16,
        bf16=runtime.use_bf16,
        gradient_checkpointing=False,
        optim="adamw_torch_fused" if runtime.major >= 8 else "adamw_torch",
        group_by_length=True,
        dataloader_num_workers=4 if runtime.is_l4 else 2,
        dataloader_pin_memory=True,
        predict_with_generate=False,
        report_to="none",
        seed=config.SEED,
        data_seed=config.SEED,
        remove_unused_columns=True,
    )

    params = inspect.signature(Seq2SeqTrainingArguments.__init__).parameters
    kwargs["eval_strategy" if "eval_strategy" in params else "evaluation_strategy"] = "epoch"
    if "tf32" in params:
        kwargs["tf32"] = bool(runtime.major >= 8)
    if "dataloader_persistent_workers" in params:
        kwargs["dataloader_persistent_workers"] = True
    if "dataloader_prefetch_factor" in params:
        kwargs["dataloader_prefetch_factor"] = 4
    if "save_only_model" in params:
        kwargs["save_only_model"] = True

    for k in [k for k in kwargs if k not in params]:
        kwargs.pop(k)

    return Seq2SeqTrainingArguments(**kwargs), grad_accum, eval_bs


def train_one_model(run_name: str, cfg: dict, use_context: bool, dataset, runtime: Runtime):
    model_id, mode = cfg["id"], cfg["preprocess"]
    tokenizer = load_tokenizer(cfg)

    train_ds = prepare_tokenized(dataset["train"], tokenizer, mode, use_context, f"tokenize-train-{run_name}")
    val_ds = prepare_tokenized(dataset["validation"], tokenizer, mode, use_context, f"tokenize-val-{run_name}")

    output_dir = os.path.join(config.OUTPUT_ROOT, safe_name_of(run_name))
    last_oom = None

    for train_bs in runtime.train_batch_candidates:
        if os.path.isdir(output_dir):
            shutil.rmtree(output_dir, ignore_errors=True)
        os.makedirs(output_dir, exist_ok=True)

        model = trainer = collator = None
        try:
            _release_cuda_memory()
            model = load_model_fp32(model_id, runtime)
            set_forced_bos_for_lang(model, tokenizer, cfg.get("lang_code"))

            collator = DataCollatorForSeq2Seq(
                tokenizer=tokenizer, model=model, padding=True, label_pad_token_id=-100, pad_to_multiple_of=8
            )
            args, grad_accum, eval_bs = make_training_args(output_dir, train_bs, runtime)
            trainer = Seq2SeqTrainer(
                model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds, data_collator=collator
            )

            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            t0 = time.time()
            trainer.train()
            torch.cuda.synchronize()
            train_time = time.time() - t0

            eval_metrics = trainer.evaluate()
            eval_loss = float(eval_metrics["eval_loss"])
            peak_vram = torch.cuda.max_memory_allocated() / 1024**3

            best_dir = os.path.join(output_dir, "best_model")
            trainer.save_model(best_dir)
            tokenizer.save_pretrained(best_dir)

            stats = {
                "Validation_Loss": eval_loss,
                "Perplexity": math.exp(eval_loss) if eval_loss < 20 else float("inf"),
                "Training_Time_s": train_time,
                "Peak_VRAM_GB": peak_vram,
                "Train_Batch": train_bs,
                "Grad_Accum": grad_accum,
                "Eval_Batch": eval_bs,
                "Best_Dir": best_dir,
            }
            return trainer, tokenizer, stats

        except Exception as e:
            if not is_cuda_oom(e):
                raise
            last_oom = e
            del trainer, model, collator
            _release_cuda_memory()
            continue

    raise RuntimeError(f"{run_name}: CUDA OOM even at batch=1") from last_oom
