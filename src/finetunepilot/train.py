"""QLoRA training with TRL SFTTrainer + PEFT, with a CPU dry-run path.

Real run (GPU):  4-bit NF4 base model + LoRA adapters, fp16 on T4.
Dry run (CPU):   tiny random model + tokenizer built locally, fp32, 2 steps. It proves the whole
                 code path (data formatting, LoRA, trainer, saving) works, without a GPU or network.

Loss is computed on the completion (the tool call) only, not on the prompt or tool schemas.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from finetunepilot.config import TrainConfig
from finetunepilot.data import build_sft_dataset, token_length_stats


class TrainResult(BaseModel):
    output_dir: str
    final_loss: float | None  # last logged training loss
    train_loss: float | None  # average training loss over the whole run (from the Trainer)
    steps: int
    duration_s: float  # wall time including model loading
    train_runtime_s: float | None
    peak_gpu_memory_mb: float | None  # torch max_memory_allocated; None on CPU
    peak_gpu_memory_reserved_mb: float | None  # closer to what nvidia-smi shows
    loss_history: list[dict[str, float]]
    data_stats: dict[str, int]
    dry_run: bool


def compute_dtype_name(cfg: TrainConfig) -> str:
    if cfg.bf16:
        return "bfloat16"
    if cfg.fp16:
        return "float16"
    return "float32"


def build_bnb_config(cfg: TrainConfig):
    """4-bit NF4 quantization settings (QLoRA). None when load_in_4bit is false."""
    if not cfg.load_in_4bit:
        return None
    import torch
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=getattr(torch, compute_dtype_name(cfg)),
    )


def build_lora_config(cfg: TrainConfig):
    from peft import LoraConfig

    return LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )


def build_sft_config(cfg: TrainConfig, output_dir: str | Path, use_cpu: bool):
    from trl import SFTConfig

    return SFTConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=cfg.grad_accum_steps,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_epochs,
        max_steps=cfg.max_steps,
        warmup_steps=cfg.warmup_ratio,  # transformers 5: a float < 1 means a ratio of total steps
        lr_scheduler_type="cosine",
        logging_steps=cfg.logging_steps,
        max_length=cfg.max_seq_length,
        completion_only_loss=True,  # train on the tool call only
        packing=False,
        fp16=cfg.fp16,
        bf16=cfg.bf16,
        activation_checkpointing=cfg.gradient_checkpointing,  # transformers 5 name
        activation_checkpointing_kwargs=(
            {"use_reentrant": False} if cfg.gradient_checkpointing else None
        ),
        optim="paged_adamw_8bit" if cfg.load_in_4bit else "adamw_torch",
        save_strategy="no",  # we save the final adapter ourselves
        report_to="none",
        seed=cfg.seed,
        data_seed=cfg.seed,
        use_cpu=use_cpu,
        dataloader_pin_memory=not use_cpu,
    )


def _load_real(cfg: TrainConfig) -> tuple[Any, Any]:
    """Load tokenizer + (4-bit) base model on GPU 0."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU found. Real training needs a GPU; use --dry-run on CPU.")
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        quantization_config=build_bnb_config(cfg),
        dtype=getattr(torch, compute_dtype_name(cfg)),
        device_map={"": 0},  # one GPU only, even on Kaggle "T4 x2"
    )
    model.config.use_cache = False  # cache is only useful for generation
    if cfg.load_in_4bit:
        from peft import prepare_model_for_kbit_training

        # Casts norms/embeddings to fp32 for stable training and enables input grads.
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=cfg.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    return model, tokenizer


def _load_dry_run(cfg: TrainConfig) -> tuple[Any, Any]:
    from finetunepilot.dryrun import build_tiny_model, build_tiny_tokenizer

    tokenizer = build_tiny_tokenizer([cfg.dataset_path])
    return build_tiny_model(tokenizer, seed=cfg.seed), tokenizer


def train(
    cfg: TrainConfig, output_dir: str | Path | None = None, dry_run: bool = False
) -> TrainResult:
    """Train LoRA adapters and save adapter + tokenizer to output_dir.

    With dry_run=True, `cfg` is first converted with `cfg.for_dry_run()`.
    Raises ValueError if any row is longer than max_seq_length (the tool call would be cut off).
    """
    import torch
    from peft import get_peft_model
    from transformers import set_seed
    from trl import SFTTrainer

    start = time.perf_counter()
    if dry_run:
        cfg = cfg.for_dry_run()
    out = Path(output_dir or cfg.output_dir)
    set_seed(cfg.seed)

    use_cuda = torch.cuda.is_available() and not dry_run
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()

    model, tokenizer = _load_dry_run(cfg) if dry_run else _load_real(cfg)

    dataset = build_sft_dataset(cfg.dataset_path, tokenizer).remove_columns("id")
    stats = token_length_stats(dataset, tokenizer, cfg.max_seq_length)
    if stats["truncated"]:
        raise ValueError(
            f"{stats['truncated']} of {stats['rows']} rows are longer than max_seq_length="
            f"{cfg.max_seq_length} (longest: {stats['max_tokens']} tokens). Truncation would cut "
            "off the tool call. Raise max_seq_length."
        )

    model = get_peft_model(model, build_lora_config(cfg))
    trainer = SFTTrainer(
        model=model,
        args=build_sft_config(cfg, out, use_cpu=not use_cuda),
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    train_output = trainer.train()

    out.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out))  # saves only the LoRA adapter for a PEFT model
    tokenizer.save_pretrained(str(out))

    history = [
        {"step": float(h["step"]), "loss": float(h["loss"])}
        for h in trainer.state.log_history
        if "loss" in h
    ]
    final_loss = history[-1]["loss"] if history else None
    train_loss = getattr(train_output, "training_loss", None)
    return TrainResult(
        output_dir=str(out),
        final_loss=final_loss,
        train_loss=None if train_loss is None or math.isnan(train_loss) else float(train_loss),
        steps=trainer.state.global_step,
        duration_s=round(time.perf_counter() - start, 2),
        train_runtime_s=train_output.metrics.get("train_runtime"),
        peak_gpu_memory_mb=round(torch.cuda.max_memory_allocated() / 2**20, 1)
        if use_cuda
        else None,
        peak_gpu_memory_reserved_mb=(
            round(torch.cuda.max_memory_reserved() / 2**20, 1) if use_cuda else None
        ),
        loss_history=history,
        data_stats=stats,
        dry_run=dry_run,
    )
