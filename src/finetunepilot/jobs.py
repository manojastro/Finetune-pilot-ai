"""Run one training job or one eval job end to end and write its runs/<run_id>.json.

These are the shared building blocks behind `finetunepilot run`, `finetunepilot eval` and the
agent's train/evaluate nodes, so every path writes the same run log (CLAUDE.md rule 4).
"""

from __future__ import annotations

import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

from finetunepilot.config import RunConfig
from finetunepilot.runlog import RunLog, now_iso, start_run, write_run

Printer = Callable[[str], None]


def _quiet(_message: str) -> None:
    pass


def train_metrics(result: dict[str, Any]) -> dict[str, Any]:
    """The headline numbers of a TrainResult dict."""
    history = result.get("loss_history") or []
    keys = (
        "final_loss",
        "train_loss",
        "steps",
        "train_runtime_s",
        "peak_gpu_memory_mb",
        "peak_gpu_memory_reserved_mb",
    )
    return {
        "first_loss": history[0]["loss"] if history else None,
        **{k: result.get(k) for k in keys},
    }


def _finish(log: RunLog, t0: float) -> None:
    log.finished_at = now_iso()
    log.duration_s = round(time.perf_counter() - t0, 2)


def run_training_job(
    run_cfg: RunConfig,
    *,
    max_attempts: int = 4,
    max_minutes: float = 60,
    dry_run: bool = False,
    force_oom: bool = False,
    gpu_memory_gb: float | None = None,
    output_dir: str | Path | None = None,
    runs_dir: str | Path | None = None,
    attempt_fn=None,
    data_stats: dict[str, int] | None = None,
    echo: bool = True,
    say: Printer = _quiet,
) -> tuple[RunLog, Path]:
    """Train with pre-check + recovery (runner.run_with_recovery) and write the run log.

    `attempt_fn` defaults to one subprocess per attempt; `data_stats` defaults to measuring the
    dataset with the tokenizer. Both can be replaced (the agent's dry run simulates training).
    """
    from finetunepilot.runner import (
        RecoveryPolicy,
        detect_gpu_memory_gb,
        make_subprocess_attempt,
        measure_data_stats,
        run_with_recovery,
    )
    from finetunepilot.tools.vram import GPU_MEMORY_GIB

    cfg = run_cfg.train
    gpu_gb = gpu_memory_gb or detect_gpu_memory_gb() or GPU_MEMORY_GIB["t4"]
    policy = RecoveryPolicy(
        max_attempts=max_attempts, max_minutes=max_minutes, gpu_memory_gb=gpu_gb
    )
    if data_stats is None:
        try:
            data_stats = measure_data_stats(cfg, dry_run)
        except Exception as e:  # the pre-check then falls back to max_seq_length
            say(f"Could not measure token lengths ({e}); using max_seq_length.")
            data_stats = None

    log = start_run(
        "train",
        run_cfg.name,
        dry_run=dry_run,
        seed=cfg.seed,
        config={
            **run_cfg.model_dump(),
            "policy": policy.model_dump(),
            "force_oom": force_oom,
            "data_stats": data_stats,
        },
    )
    runs = Path(runs_dir or run_cfg.runs_dir)
    t0 = time.perf_counter()
    path = write_run(log, runs)
    say(
        f"Run {log.run_id}: up to {max_attempts} attempts / {max_minutes:g} min, "
        f"GPU {gpu_gb:g} GiB, dry_run={dry_run}, force_oom={force_oom}. Log: {path}"
    )

    def save_progress(attempts) -> None:
        log.attempts = attempts
        write_run(log, runs)

    seq_len = (data_stats or {}).get("max_tokens")
    outcome = run_with_recovery(
        cfg,
        policy,
        output_dir=output_dir or Path(cfg.output_dir) / log.run_id,
        attempt_fn=attempt_fn
        or make_subprocess_attempt(dry_run, simulate_oom_gpu_gb=gpu_gb, seq_len=seq_len, echo=echo),
        gpu_memory_gb=gpu_gb,
        data_stats=data_stats,
        force_oom=force_oom,
        on_attempt=save_progress,
    )
    log.attempts = outcome.attempts
    log.precheck = outcome.precheck
    # `train` is the config as given; these are what the runner actually started and ended with
    # (they differ with --force-oom, or after pre-check / recovery fixes).
    log.config["base_train"] = outcome.base_config.model_dump()
    log.config["final_train"] = outcome.final_config.model_dump()
    log.stop_reason = outcome.stop_reason
    log.status = outcome.status
    if outcome.result:
        log.metrics = {**train_metrics(outcome.result), "attempts": len(outcome.attempts)}
        log.artifacts = {"adapter_dir": outcome.result["output_dir"]}
    else:
        log.metrics = {"attempts": len(outcome.attempts)}
        log.errors = [a.error for a in outcome.attempts if a.error] or [outcome.stop_reason]
    _finish(log, t0)
    write_run(log, runs)
    return log, path


def run_eval_job(
    settings,
    *,
    name: str,
    runs_dir: str | Path = "runs",
    evaluate_fn=None,
    say: Printer = _quiet,
) -> tuple[RunLog, Path]:
    """Evaluate (eval.evaluate by default) and write the run log. Never raises for eval errors:
    a failed eval is returned as a log with status "failed"."""
    from finetunepilot.eval import evaluate

    log = start_run(
        "eval", name, dry_run=settings.dry_run, seed=settings.seed, config=settings.model_dump()
    )
    t0 = time.perf_counter()
    path = write_run(log, runs_dir)
    say(f"Eval {log.run_id} started (dry_run={settings.dry_run}). Log: {path}")
    try:
        result = (evaluate_fn or evaluate)(settings)
    except Exception as e:
        log.status = "failed"
        log.errors = [f"{type(e).__name__}: {e}", "".join(traceback.format_exception(e))[-4000:]]
        _finish(log, t0)
        write_run(log, runs_dir)
        return log, path
    log.status = "success"
    log.metrics = {**result.summary, "peak_gpu_memory_mb": result.peak_gpu_memory_mb}
    log.predictions = result.predictions
    _finish(log, t0)
    write_run(log, runs_dir)
    return log, path
