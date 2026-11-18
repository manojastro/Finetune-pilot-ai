"""The deterministic tools the agent calls: measure data, train, evaluate.

RealToolbox      runs the real jobs (subprocess training with recovery, GPU eval).
SimulatedToolbox (--dry-run) runs the SAME job code paths (pre-check, recovery loop, run logs,
                 scoring) but replaces GPU work with a toy simulation, so the whole agent runs
                 on a laptop in seconds. Its numbers are NOT measurements: every run log it
                 writes has dry_run=true and the report says "simulated".
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Protocol

from finetunepilot.config import RunConfig, TrainConfig
from finetunepilot.data import load_rows
from finetunepilot.eval import EvalResult, EvalSettings
from finetunepilot.jobs import run_eval_job, run_training_job
from finetunepilot.runlog import RunLog
from finetunepilot.runner import AttemptOutcome, measure_data_stats
from finetunepilot.tools.scoring import score_prediction, summarize
from finetunepilot.tools.vram import estimate_vram


class Toolbox(Protocol):
    def data_stats(self, cfg: TrainConfig) -> dict[str, int]: ...

    def train(
        self, run_cfg: RunConfig, *, max_minutes: float, max_attempts: int, gpu_memory_gb: float
    ) -> RunLog: ...

    def evaluate(
        self, base_model: str, adapter: str | None, test_path: str, name: str
    ) -> RunLog: ...


class RealToolbox:
    def __init__(self, runs_dir: str = "runs", say=print):
        self.runs_dir, self.say = runs_dir, say

    def data_stats(self, cfg: TrainConfig) -> dict[str, int]:
        return measure_data_stats(cfg, dry_run=False)

    def train(self, run_cfg, *, max_minutes, max_attempts, gpu_memory_gb) -> RunLog:
        log, _ = run_training_job(
            run_cfg, max_attempts=max_attempts, max_minutes=max_minutes,
            gpu_memory_gb=gpu_memory_gb, runs_dir=self.runs_dir, say=self.say,
        )  # fmt: skip
        return log

    def evaluate(self, base_model, adapter, test_path, name) -> RunLog:
        settings = EvalSettings(base_model=base_model, adapter=adapter, test_path=test_path)
        log, _ = run_eval_job(settings, name=name, runs_dir=self.runs_dir, say=self.say)
        return log


# ----------------------------------------------------------------------------- simulation

SIM_FILE = "simulated_adapter.json"
BASE_QUALITY = 0.15


def simulated_quality(cfg: TrainConfig) -> float:
    """A made-up, smooth 'how good would this config be' score in [0.05, 0.98], used only by
    dry runs so the agent loop has something to react to. Best near lr 3e-4, r 16, >= 4 epochs."""
    q = (
        0.92
        - 0.18 * abs(math.log10(cfg.learning_rate / 3e-4))
        - 0.05 * abs(math.log2(cfg.lora_r / 16))
        - 0.06 * max(0.0, 4 - cfg.num_epochs)
    )
    return round(min(0.98, max(0.05, q)), 4)


def _wrong_output(row, kind: int) -> str:
    target = row.target
    if kind % 3 == 0:  # forgot an argument
        args = dict(list(target.arguments.items())[:-1])
        return (
            "<tool_call>\n"
            + json.dumps({"name": target.name, "arguments": args})
            + "\n</tool_call>"
        )
    if kind % 3 == 1:  # picked another tool
        other = next(n for n in row.tool_names if n != target.name)
        return (
            "<tool_call>\n"
            + json.dumps({"name": other, "arguments": target.arguments})
            + "\n</tool_call>"
        )
    return f"I would call {target.name} for this."  # no JSON at all


def simulated_outputs(test_path: str, quality: float) -> tuple[list, list[str]]:
    rows = load_rows(test_path)
    n_correct = round(quality * len(rows))
    outputs = []
    for i, row in enumerate(rows):
        if i < n_correct:
            payload = {"name": row.target.name, "arguments": row.target.arguments}
            outputs.append("<tool_call>\n" + json.dumps(payload) + "\n</tool_call>")
        else:
            outputs.append(_wrong_output(row, i))
    return rows, outputs


class SimulatedToolbox:
    def __init__(self, runs_dir: str = "runs", say=print):
        self.runs_dir, self.say = runs_dir, say

    def data_stats(self, cfg: TrainConfig) -> dict[str, int]:
        return measure_data_stats(cfg, dry_run=True)  # real code, tiny local tokenizer

    def _attempt_fn(self, gpu_memory_gb: float, seq_len: int | None):
        def attempt(cfg: TrainConfig, out_dir: Path, timeout_s: float) -> AttemptOutcome:
            est = estimate_vram(cfg, gpu_memory_gb, seq_len=seq_len, safety_margin=0.0)
            if not est.fits:
                return AttemptOutcome(
                    status="failed",
                    error=f"OutOfMemoryError: CUDA out of memory (simulated): "
                    f"{est.total_gb:.2f} GiB",
                )
            adapter = out_dir / "adapter"
            adapter.mkdir(parents=True, exist_ok=True)
            (adapter / SIM_FILE).write_text(cfg.model_dump_json(indent=2), encoding="utf-8")
            q = simulated_quality(cfg)
            steps = math.ceil(60 / cfg.effective_batch_size) * math.ceil(cfg.num_epochs)
            loss = round(0.05 + 1.5 * (1 - q), 4)
            return AttemptOutcome(
                status="success",
                result={
                    "output_dir": str(adapter), "final_loss": loss, "train_loss": loss + 0.3,
                    "steps": steps, "duration_s": 0.0, "train_runtime_s": None,
                    "peak_gpu_memory_mb": None, "peak_gpu_memory_reserved_mb": None,
                    "loss_history": [{"step": 1.0, "loss": 2.0},
                                     {"step": float(steps), "loss": loss}],
                    "data_stats": {}, "dry_run": True,
                },
            )  # fmt: skip

        return attempt

    def train(self, run_cfg, *, max_minutes, max_attempts, gpu_memory_gb) -> RunLog:
        stats = self.data_stats(run_cfg.train)
        log, _ = run_training_job(
            run_cfg, max_attempts=max_attempts, max_minutes=max_minutes, dry_run=True,
            gpu_memory_gb=gpu_memory_gb, runs_dir=self.runs_dir, data_stats=stats,
            attempt_fn=self._attempt_fn(gpu_memory_gb, stats.get("max_tokens")), say=self.say,
        )  # fmt: skip
        return log

    def evaluate(self, base_model, adapter, test_path, name) -> RunLog:
        def fake_evaluate(settings: EvalSettings) -> EvalResult:
            if settings.adapter:
                sim = Path(settings.adapter) / SIM_FILE
                quality = simulated_quality(TrainConfig.model_validate_json(sim.read_text()))
            else:
                quality = BASE_QUALITY
            rows, outputs = simulated_outputs(settings.test_path, quality)
            scores = [
                score_prediction(o, r.target.name, r.target.arguments)
                for o, r in zip(outputs, rows, strict=True)
            ]
            predictions = [
                {"id": r.id, "request": r.request, "target": r.target.model_dump(),
                 "output": o, **s.model_dump()}
                for r, o, s in zip(rows, outputs, scores, strict=True)
            ]  # fmt: skip
            return EvalResult(
                summary=summarize(scores, [r.target.name for r in rows]),
                predictions=predictions, duration_s=0.0, peak_gpu_memory_mb=None,
            )  # fmt: skip

        settings = EvalSettings(
            base_model=base_model, adapter=adapter, test_path=test_path, dry_run=True
        )
        log, _ = run_eval_job(
            settings, name=name, runs_dir=self.runs_dir, evaluate_fn=fake_evaluate, say=self.say
        )
        return log


class DryRunLLM:
    """Rule-based stand-in for the LLM in --dry-run: a fixed sensible plan, then a fixed
    sequence of one-field changes. Same interface as the real client; no network."""

    PLAN = {
        "changes": {"learning_rate": 1e-4, "num_epochs": 3, "lora_r": 8, "lora_alpha": 16,
                    "per_device_batch_size": 4, "grad_accum_steps": 2, "max_seq_length": 1024},
        "reason": "Dry run: a conservative first config (low learning rate, small LoRA rank).",
    }  # fmt: skip
    CHANGES = [
        {"field": "learning_rate", "value": 2e-4, "reason": "Dry run: loss still high; raise lr."},
        {"field": "lora_r", "value": 16, "reason": "Dry run: more adapter capacity for arguments."},
        {"field": "num_epochs", "value": 5, "reason": "Dry run: tiny dataset, train longer."},
        {"field": "lora_dropout", "value": 0.1, "reason": "Dry run: regularise a little more."},
    ]

    def __init__(self) -> None:
        self.diagnoses = 0

    def complete(self, system: str, prompt: str) -> str:
        if prompt.startswith("TASK: PLAN"):
            return json.dumps(self.PLAN)
        change = self.CHANGES[self.diagnoses % len(self.CHANGES)]
        self.diagnoses += 1
        return json.dumps(change)
