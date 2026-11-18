"""Run training with a VRAM pre-check, automatic OOM/NaN recovery and a budget cap.

    run_with_recovery(cfg, policy, output_dir=..., attempt_fn=...)

1. Pre-check: estimate VRAM (tools/vram.py) at the longest row in the data. If it won't fit,
   apply the OOM fixes below on paper until it does; if nothing makes it fit, stop before
   launching anything. Skipped with force_oom (that demo needs a real OOM).
2. Run an attempt. By default each attempt is a fresh subprocess (finetunepilot.worker), so a
   CUDA OOM can't leave the next attempt with less GPU memory.
3. On failure, classify the error (tools/log_parser.py) and change the config:
     CUDA_OOM  next fix in this order, one per failure:
                 1. halve per_device_batch_size and double grad_accum_steps (effective batch kept)
                 2. turn on gradient checkpointing
                 3. cut max_seq_length by 25%, never below the longest row (no truncation)
               then keep halving the batch while it is above 1.
     NAN_LOSS  halve the learning rate (not below 1e-6).
     other     stop: retrying would fail the same way.
4. Stop on success, at policy.max_attempts, or when policy.max_minutes is used up.

A fix whose fields are not in policy.allowed_changes is skipped, and every new config is checked
to differ from the base config only in allowed fields.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from finetunepilot.config import TrainConfig, diff_configs
from finetunepilot.runlog import Attempt, now_iso
from finetunepilot.tools.log_parser import classify_error, is_retryable
from finetunepilot.tools.vram import ModelSpec, estimate_vram, get_model_spec

FORCE_OOM_BATCH = 16
MIN_LEARNING_RATE = 1e-6
SEQ_CUT = 0.75
DEFAULT_ALLOWED_CHANGES = [
    "per_device_batch_size",
    "grad_accum_steps",
    "gradient_checkpointing",
    "max_seq_length",
    "learning_rate",
]


class RecoveryPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_attempts: int = Field(default=4, ge=1, le=20)
    max_minutes: float = Field(default=60, gt=0)
    allowed_changes: list[str] = Field(default_factory=lambda: list(DEFAULT_ALLOWED_CHANGES))
    safety_margin: float = Field(default=0.15, ge=0, lt=1)
    gpu_memory_gb: float | None = Field(default=None, gt=0)

    @field_validator("allowed_changes")
    @classmethod
    def _known_fields(cls, value: list[str]) -> list[str]:
        unknown = sorted(set(value) - set(TrainConfig.model_fields))
        if unknown:
            raise ValueError(f"allowed_changes has unknown TrainConfig fields: {unknown}")
        return value


class AttemptOutcome(BaseModel):
    """What one attempt reports back, whatever ran it."""

    status: Literal["success", "failed", "timeout"]
    result: dict[str, Any] | None = None  # TrainResult as a dict
    error: str | None = None  # "ErrorType: message"
    log_tail: str = ""  # traceback / last output lines, used to classify the error
    data_stats: dict[str, int] = Field(default_factory=dict)


# (config, output dir, seconds left) -> outcome
AttemptFn = Callable[[TrainConfig, Path, float], AttemptOutcome]


class RecoveryResult(BaseModel):
    status: Literal["success", "failed"]
    stop_reason: str
    base_config: TrainConfig
    final_config: TrainConfig
    attempts: list[Attempt]
    precheck: dict[str, Any]
    result: dict[str, Any] | None = None


# ----------------------------------------------------------------------------- fixes


@dataclass(frozen=True)
class Fix:
    name: str
    fields: frozenset[str]  # TrainConfig fields this fix may change
    apply: Callable[[TrainConfig, int | None], TrainConfig | None]  # None = not applicable


def _halve_batch(cfg: TrainConfig, _min_seq: int | None) -> TrainConfig | None:
    if cfg.per_device_batch_size <= 1:
        return None
    new_batch = cfg.per_device_batch_size // 2
    # Keep the effective batch (round up if the batch size was odd).
    new_accum = math.ceil(cfg.effective_batch_size / new_batch)
    return cfg.model_copy(
        update={"per_device_batch_size": new_batch, "grad_accum_steps": new_accum}
    )


def _enable_checkpointing(cfg: TrainConfig, _min_seq: int | None) -> TrainConfig | None:
    if cfg.gradient_checkpointing:
        return None
    return cfg.model_copy(update={"gradient_checkpointing": True})


def _cut_seq_length(cfg: TrainConfig, min_seq: int | None) -> TrainConfig | None:
    floor = max(32, min_seq or 32)  # never below the longest row: that would truncate tool calls
    new_len = max(int(cfg.max_seq_length * SEQ_CUT), floor)
    if new_len >= cfg.max_seq_length:
        return None
    return cfg.model_copy(update={"max_seq_length": new_len})


def _halve_lr(cfg: TrainConfig, _min_seq: int | None) -> TrainConfig | None:
    new_lr = cfg.learning_rate / 2
    if new_lr < MIN_LEARNING_RATE:
        return None
    return cfg.model_copy(update={"learning_rate": new_lr})


HALVE_BATCH = Fix(
    "halve batch", frozenset({"per_device_batch_size", "grad_accum_steps"}), _halve_batch
)
GRAD_CHECKPOINTING = Fix(
    "gradient checkpointing", frozenset({"gradient_checkpointing"}), _enable_checkpointing
)
CUT_SEQ = Fix("cut max_seq_length 25%", frozenset({"max_seq_length"}), _cut_seq_length)
HALVE_LR = Fix("halve learning rate", frozenset({"learning_rate"}), _halve_lr)
OOM_LADDER = [HALVE_BATCH, GRAD_CHECKPOINTING, CUT_SEQ]


def describe_change(old: TrainConfig, new: TrainConfig) -> str:
    return ", ".join(f"{k} {v['from']} -> {v['to']}" for k, v in diff_configs(old, new).items())


class FixPlanner:
    """Picks the next fix. Remembers how far down the OOM ladder we already are."""

    def __init__(self, allowed: list[str], min_seq_len: int | None = None):
        self.allowed = set(allowed)
        self.min_seq_len = min_seq_len
        self.position = 0

    def _try(self, fix: Fix, cfg: TrainConfig) -> tuple[TrainConfig, str] | None:
        if not fix.fields <= self.allowed:
            return None
        new = fix.apply(cfg, self.min_seq_len)
        return None if new is None else (new, f"{fix.name}: {describe_change(cfg, new)}")

    def next_oom_fix(self, cfg: TrainConfig) -> tuple[TrainConfig, str] | None:
        while self.position < len(OOM_LADDER):
            fix = OOM_LADDER[self.position]
            self.position += 1
            found = self._try(fix, cfg)
            if found:
                return found
        return self._try(HALVE_BATCH, cfg)  # ladder used up: keep shrinking the batch

    def next_fix(self, error_class: str, cfg: TrainConfig) -> tuple[TrainConfig, str] | None:
        if error_class == "CUDA_OOM":
            return self.next_oom_fix(cfg)
        if error_class == "NAN_LOSS":
            return self._try(HALVE_LR, cfg)
        return None


# ----------------------------------------------------------------------------- pre-check


class WontFitError(RuntimeError):
    def __init__(self, message: str, info: dict[str, Any]):
        super().__init__(message)
        self.info = info


def precheck(
    cfg: TrainConfig,
    policy: RecoveryPolicy,
    gpu_memory_gb: float,
    spec: ModelSpec | None = None,
    seq_len: int | None = None,
) -> tuple[TrainConfig, dict[str, Any]]:
    """Return a config whose VRAM estimate fits, plus a record of what was changed.

    Raises WontFitError if no allowed fix makes it fit. If the model's architecture is unknown
    the check is skipped (and recorded as skipped) rather than guessed.
    """
    try:
        spec = spec or get_model_spec(cfg.base_model)
    except KeyError as e:
        return cfg, {"status": "skipped", "reason": str(e)}

    def est(c: TrainConfig):
        return estimate_vram(
            c, gpu_memory_gb, spec=spec, seq_len=seq_len, safety_margin=policy.safety_margin
        )

    before = est(cfg)
    planner = FixPlanner(policy.allowed_changes, min_seq_len=seq_len)
    fixes: list[str] = []
    current, estimate = cfg, before
    while not estimate.fits:
        step = planner.next_oom_fix(current)
        if step is None:
            info = {
                "status": "wont_fit",
                "estimate_before": before.model_dump(),
                "estimate_after": estimate.model_dump(),
                "fixes": fixes,
            }
            raise WontFitError(
                f"Estimated {estimate.total_gb:.2f} GiB > limit {estimate.limit_gb:.2f} GiB even "
                "after every allowed fix.",
                info,
            )
        current, note = step
        fixes.append(note)
        estimate = est(current)
    return current, {
        "status": "adjusted" if fixes else "fits",
        "gpu_memory_gb": gpu_memory_gb,
        "seq_len": estimate.seq_len,
        "estimate_before": before.model_dump(),
        "estimate_after": estimate.model_dump(),
        "fixes": fixes,
    }


# ----------------------------------------------------------------------------- main loop


def run_with_recovery(
    cfg: TrainConfig,
    policy: RecoveryPolicy,
    *,
    output_dir: str | Path,
    attempt_fn: AttemptFn,
    gpu_memory_gb: float,
    data_stats: dict[str, int] | None = None,
    force_oom: bool = False,
    spec: ModelSpec | None = None,
    clock: Callable[[], float] = time.monotonic,
    on_attempt: Callable[[list[Attempt]], None] | None = None,
) -> RecoveryResult:
    """Train with automatic recovery. See the module docstring for the rules."""
    start = clock()
    deadline = start + policy.max_minutes * 60
    min_seq = (data_stats or {}).get("max_tokens")
    allowed = set(policy.allowed_changes)
    out = Path(output_dir)

    if force_oom:
        cfg = cfg.model_copy(update={"per_device_batch_size": FORCE_OOM_BATCH})
        check = {
            "status": "skipped",
            "reason": f"--force-oom: batch size forced to {FORCE_OOM_BATCH}",
        }
        current, pending_fix = cfg, f"force_oom: per_device_batch_size -> {FORCE_OOM_BATCH}"
    else:
        try:
            current, check = precheck(cfg, policy, gpu_memory_gb, spec=spec, seq_len=min_seq)
        except WontFitError as e:
            return RecoveryResult(
                status="failed", stop_reason=f"won't fit: {e}", base_config=cfg,
                final_config=cfg, attempts=[], precheck=e.info,
            )  # fmt: skip
        pending_fix = ("pre-check: " + "; ".join(check["fixes"])) if check.get("fixes") else None

    base = cfg
    planner = FixPlanner(policy.allowed_changes, min_seq_len=min_seq)
    attempts: list[Attempt] = []
    status: Literal["success", "failed"] = "failed"
    stop_reason = "max attempts reached"
    result: dict[str, Any] | None = None

    for n in range(1, policy.max_attempts + 1):
        remaining = deadline - clock()
        if remaining <= 0:
            stop_reason = f"time budget of {policy.max_minutes:g} min used up before attempt {n}"
            break
        changes = diff_configs(base, current)
        if not set(changes) <= allowed:  # guard: should never trigger
            raise RuntimeError(
                f"Runner tried to change fields outside the policy: {sorted(changes)}"
            )
        attempt = Attempt(
            attempt=n,
            status="running",
            started_at=now_iso(),
            config_changes=changes,
            fix=pending_fix,
        )
        attempts.append(attempt)
        if on_attempt:
            on_attempt(attempts)

        t0 = clock()
        outcome = attempt_fn(current, out / f"attempt-{n}", remaining)
        attempt.duration_s = round(clock() - t0, 2)
        if outcome.data_stats.get("max_tokens"):
            min_seq = planner.min_seq_len = outcome.data_stats["max_tokens"]

        if outcome.status == "success":
            attempt.status, attempt.result = "success", outcome.result or {}
            status, stop_reason, result = "success", f"succeeded on attempt {n}", outcome.result
            break
        if outcome.status == "timeout":
            attempt.status, attempt.error = "timeout", outcome.error
            stop_reason = f"time budget of {policy.max_minutes:g} min used up during attempt {n}"
            break

        info = classify_error(f"{outcome.error or ''}\n{outcome.log_tail}")
        attempt.status = "failed"
        attempt.error_class, attempt.evidence, attempt.error = (
            info.error_class,
            info.evidence,
            outcome.error,
        )
        if on_attempt:
            on_attempt(attempts)
        if not is_retryable(info.error_class):
            stop_reason = f"{info.error_class} is not retryable"
            break
        if n == policy.max_attempts:
            stop_reason = f"max attempts ({policy.max_attempts}) reached"
            break
        step = planner.next_fix(info.error_class, current)
        if step is None:
            stop_reason = f"no allowed fix left for {info.error_class}"
            break
        current, note = step
        pending_fix = f"after {info.error_class}: {note}"

    if on_attempt:
        on_attempt(attempts)
    return RecoveryResult(
        status=status, stop_reason=stop_reason, base_config=base, final_config=current,
        attempts=attempts, precheck=check, result=result,
    )  # fmt: skip


# ----------------------------------------------------------------------------- attempt runners


def make_subprocess_attempt(
    dry_run: bool,
    simulate_oom_gpu_gb: float | None = None,
    seq_len: int | None = None,
    echo: bool = True,
    command: list[str] | None = None,
) -> AttemptFn:
    """An AttemptFn that runs `python -m finetunepilot.worker` in a fresh process.

    Output is streamed to the console (echo) and saved to <attempt dir>/attempt.log. The process
    is killed if it runs past the remaining time budget. `command` overrides the worker command
    (tests use it to fake crashes).
    """

    def attempt(cfg: TrainConfig, out_dir: Path, timeout_s: float) -> AttemptOutcome:
        out_dir.mkdir(parents=True, exist_ok=True)
        request, result_path = out_dir / "request.json", out_dir / "result.json"
        result_path.unlink(missing_ok=True)
        request.write_text(
            json.dumps(
                {
                    "config": cfg.model_dump(),
                    "output_dir": str(out_dir / "adapter"),
                    "dry_run": dry_run,
                    "simulate_oom_gpu_gb": simulate_oom_gpu_gb if dry_run else None,
                    "seq_len": seq_len,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        cmd = command or [
            sys.executable,
            "-m",
            "finetunepilot.worker",
            str(request),
            str(result_path),
        ]
        tail: deque[str] = deque(maxlen=200)
        with (out_dir / "attempt.log").open("w", encoding="utf-8") as log_file:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            )

            def pump() -> None:
                assert proc.stdout is not None
                for line in proc.stdout:
                    tail.append(line)
                    log_file.write(line)
                    if echo:
                        sys.stdout.write(line)

            reader = threading.Thread(target=pump, daemon=True)
            reader.start()
            try:
                proc.wait(timeout=max(timeout_s, 0.001))
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                reader.join(timeout=5)
                return AttemptOutcome(
                    status="timeout",
                    error=f"attempt killed after {timeout_s:.0f}s (time budget)",
                    log_tail="".join(tail),
                )
            reader.join(timeout=5)

        payload: dict[str, Any] = {}
        if result_path.exists():
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        stats = payload.get("data_stats") or {}
        if payload.get("status") == "success":
            return AttemptOutcome(status="success", result=payload["result"], data_stats=stats)
        if payload.get("status") == "failed":
            return AttemptOutcome(
                status="failed",
                error=f"{payload['error_type']}: {payload['error']}",
                log_tail=payload.get("traceback", "") + "\n" + "".join(tail),
                data_stats=stats,
            )
        return AttemptOutcome(  # the process died without reporting (killed, segfault, ...)
            status="failed",
            error=f"worker exited with code {proc.returncode} without a result",
            log_tail="".join(tail),
            data_stats=stats,
        )

    return attempt


def measure_data_stats(cfg: TrainConfig, dry_run: bool) -> dict[str, int]:
    """Token length stats of the training data (needs only the tokenizer, no GPU)."""
    from finetunepilot.data import build_sft_dataset, token_length_stats

    if dry_run:
        from finetunepilot.dryrun import build_tiny_tokenizer

        tokenizer = build_tiny_tokenizer([cfg.dataset_path])
    else:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    return token_length_stats(
        build_sft_dataset(cfg.dataset_path, tokenizer), tokenizer, cfg.max_seq_length
    )


def detect_gpu_memory_gb() -> float | None:
    """Total memory of GPU 0 in GiB, or None without CUDA."""
    import torch

    if not torch.cuda.is_available():
        return None
    return round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2)
