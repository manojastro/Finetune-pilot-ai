"""Pydantic config models (TrainConfig, RunConfig) loaded from YAML.

A YAML config file looks like this:

    name: qwen05b_toolcall
    train:
      base_model: Qwen/Qwen2.5-0.5B-Instruct
      dataset_path: data/sample/train.jsonl
      ...

`RunConfig` holds run-level settings (name, where run logs go) and wraps one `TrainConfig`.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

# Linear layers of Qwen2-style models. LoRA adapters are added to these.
QWEN_LINEAR_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


class TrainConfig(BaseModel):
    """Everything needed to run one QLoRA training job."""

    # extra="forbid" turns a typo in the YAML (e.g. `lora_rank`) into a clear error.
    model_config = ConfigDict(extra="forbid")

    base_model: str = Field(min_length=1)
    dataset_path: str = Field(min_length=1)
    output_dir: str = Field(min_length=1)

    max_seq_length: int = Field(default=1024, ge=32, le=32768)
    per_device_batch_size: int = Field(default=4, ge=1, le=1024)
    grad_accum_steps: int = Field(default=1, ge=1, le=1024)
    learning_rate: float = Field(default=2e-4, ge=1e-6, le=1e-2)
    num_epochs: float = Field(default=3.0, gt=0, le=100)
    # -1 means "use num_epochs". A positive value wins over num_epochs (same rule as HF Trainer).
    max_steps: int = Field(default=-1, ge=-1)
    warmup_ratio: float = Field(default=0.0, ge=0.0, lt=1.0)
    logging_steps: int = Field(default=1, ge=1)

    lora_r: int = Field(default=16, gt=0, le=512)
    lora_alpha: int = Field(default=32, gt=0, le=1024)
    lora_dropout: float = Field(default=0.05, ge=0.0, lt=1.0)
    target_modules: list[str] = Field(
        default_factory=lambda: list(QWEN_LINEAR_MODULES), min_length=1
    )

    load_in_4bit: bool = True
    bf16: bool = False
    fp16: bool = False
    gradient_checkpointing: bool = False
    seed: int = Field(default=42, ge=0)

    @model_validator(mode="after")
    def _check(self) -> TrainConfig:
        if self.bf16 and self.fp16:
            raise ValueError(
                "bf16 and fp16 cannot both be true; pick one (T4 GPUs only support fp16)."
            )
        if self.max_steps == 0:
            raise ValueError("max_steps must be -1 (use num_epochs) or a positive number.")
        return self

    @property
    def effective_batch_size(self) -> int:
        return self.per_device_batch_size * self.grad_accum_steps

    def for_dry_run(self) -> TrainConfig:
        """A copy that runs on CPU in seconds: fp32, no 4-bit, 2 steps, small batch.

        max_seq_length is raised because the tiny dry-run tokenizer has a small vocabulary,
        so the same text takes more tokens than with the real one. Rows must not be truncated,
        or the tool call (the only part we train on) would be cut off.
        """
        return self.model_copy(
            update={
                "load_in_4bit": False,
                "bf16": False,
                "fp16": False,
                "gradient_checkpointing": False,
                "max_steps": 2,
                "per_device_batch_size": 2,
                "grad_accum_steps": 1,
                "max_seq_length": max(self.max_seq_length, 4096),
                "logging_steps": 1,
            }
        )


class RunConfig(BaseModel):
    """Run-level settings plus the training config. This is what a YAML file holds."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_.-]+$")
    runs_dir: str = "runs"
    notes: str = ""
    train: TrainConfig


def diff_configs(old: TrainConfig, new: TrainConfig) -> dict[str, dict]:
    """Fields that differ, as {field: {"from": old_value, "to": new_value}}."""
    a, b = old.model_dump(), new.model_dump()
    return {k: {"from": a[k], "to": b[k]} for k in a if a[k] != b[k]}


def load_run_config(path: str | Path) -> RunConfig:
    """Read a YAML file and validate it. Raises FileNotFoundError or pydantic.ValidationError."""
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping at the top level.")
    return RunConfig.model_validate(data)
