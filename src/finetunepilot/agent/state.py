"""Agent settings, the LLM answer schemas, and the graph state.

The LLM may only touch TUNABLE_FIELDS. Everything else in a TrainConfig (model, data paths,
output dir, 4-bit, precision, seed) is set by code from the session settings.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field, field_validator

from finetunepilot.agent.llm import LLMSettings
from finetunepilot.config import TrainConfig
from finetunepilot.tools.vram import GPU_MEMORY_GIB

TunableField = Literal[
    "learning_rate",
    "num_epochs",
    "lora_r",
    "lora_alpha",
    "lora_dropout",
    "per_device_batch_size",
    "grad_accum_steps",
    "max_seq_length",
    "gradient_checkpointing",
    "warmup_ratio",
]
TUNABLE_FIELDS: tuple[str, ...] = TunableField.__args__  # type: ignore[attr-defined]
NO_BF16_GPUS = {"t4", "p100"}  # these GPUs have no bf16: use fp16


class AgentSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    objective: str = Field(min_length=3)
    base_model: str = Field(min_length=1)
    dataset_dir: str  # must contain train.jsonl and test.jsonl
    gpu: str = "t4"
    max_runs: int = Field(default=3, ge=1, le=10)
    max_minutes: float = Field(default=90, gt=0)
    max_attempts_per_run: int = Field(default=4, ge=1, le=10)
    dry_run: bool = False
    auto_approve: bool = False
    llm: LLMSettings
    runs_dir: str = "runs"
    reports_dir: str = "reports"
    sessions_dir: str = "runs/sessions"
    output_root: str = "outputs/agent"
    seed: int = 42

    @field_validator("gpu")
    @classmethod
    def _known_gpu(cls, value: str) -> str:
        if value not in GPU_MEMORY_GIB:
            raise ValueError(f"unknown gpu {value!r}; choose one of {sorted(GPU_MEMORY_GIB)}")
        return value

    @property
    def train_path(self) -> str:
        return str(Path(self.dataset_dir) / "train.jsonl")

    @property
    def test_path(self) -> str:
        return str(Path(self.dataset_dir) / "test.jsonl")

    @property
    def gpu_memory_gb(self) -> float:
        return GPU_MEMORY_GIB[self.gpu]


def build_train_config(
    settings: AgentSettings, changes: dict[str, Any], output_dir: str
) -> TrainConfig:
    """TrainConfig = code-owned fields + defaults + the LLM's tunable values.
    Raises ValueError / ValidationError on bad values."""
    unknown = sorted(set(changes) - set(TUNABLE_FIELDS))
    if unknown:
        raise ValueError(f"not tunable by the agent: {unknown}; allowed: {list(TUNABLE_FIELDS)}")
    half = "fp16" if settings.gpu in NO_BF16_GPUS else "bf16"
    return TrainConfig(
        base_model=settings.base_model,
        dataset_path=settings.train_path,
        output_dir=output_dir,
        load_in_4bit=True,
        seed=settings.seed,
        logging_steps=2,
        **{half: True},
        **changes,
    )


def tunable_values(cfg: TrainConfig | dict[str, Any]) -> dict[str, Any]:
    data = cfg.model_dump() if isinstance(cfg, TrainConfig) else cfg
    return {k: data[k] for k in TUNABLE_FIELDS}


class PlanProposal(BaseModel):
    """LLM answer for the plan node: values for any tunable fields, and why."""

    model_config = ConfigDict(extra="forbid")

    changes: dict[str, Any]
    reason: str = Field(min_length=10, max_length=1500)

    @field_validator("changes")
    @classmethod
    def _only_tunable(cls, value: dict[str, Any]) -> dict[str, Any]:
        unknown = sorted(set(value) - set(TUNABLE_FIELDS))
        if unknown:
            raise ValueError(f"not tunable: {unknown}; allowed: {list(TUNABLE_FIELDS)}")
        return value


class ConfigChange(BaseModel):
    """LLM answer for the diagnose node: exactly ONE change, and why."""

    model_config = ConfigDict(extra="forbid")

    field: TunableField
    value: Any
    reason: str = Field(min_length=10, max_length=1500)


class AgentState(TypedDict, total=False):
    """Everything the graph carries. Plain JSON-able values only, so it can be checkpointed."""

    session_id: str
    settings: dict[str, Any]
    status: str  # running | completed | failed | rejected
    started_at: float
    last_node: str
    dataset: dict[str, Any]
    baseline: dict[str, Any]
    pending: dict[str, Any]  # {"changes", "reason", "source"} for the next run
    plan_feedback: str | None
    plan_tries: int
    estimate: dict[str, Any]
    runs: list[dict[str, Any]]
    next_change: dict[str, Any] | None
    stop_reason: str | None
    error: str | None
    report_path: str | None
