"""Typer command-line interface: `finetunepilot <command>`.

Heavy imports (torch, transformers) happen inside commands so `finetunepilot version` stays fast.
"""

from __future__ import annotations

import time
import traceback
from pathlib import Path
from typing import TYPE_CHECKING

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

from finetunepilot import __version__

if TYPE_CHECKING:
    from finetunepilot.runlog import Attempt, RunLog

app = typer.Typer(
    name="finetunepilot",
    help="Plan, run, diagnose and compare QLoRA fine-tuning experiments.",
    no_args_is_help=True,
)
console = Console()


@app.callback()
def main() -> None:
    """FineTunePilot command-line interface."""
    # An explicit callback keeps commands as subcommands even when there is only one.


@app.command()
def version() -> None:
    """Print the installed FineTunePilot version."""
    typer.echo(f"finetunepilot {__version__}")


# --------------------------------------------------------------------------- train


@app.command()
def train(
    config: Path = typer.Option(..., "--config", "-c", help="YAML run config."),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Tiny random model on CPU, 2 steps. No GPU or download needed."
    ),
    output_dir: Path | None = typer.Option(
        None, help="Where to save the adapter. Default: <train.output_dir>/<run_id>."
    ),
    runs_dir: Path | None = typer.Option(
        None, help="Where to write the run JSON. Default: from config."
    ),
) -> None:
    """Train LoRA adapters and write runs/<run_id>.json."""
    from finetunepilot.config import diff_configs, load_run_config
    from finetunepilot.runlog import Attempt, now_iso, start_run, write_run
    from finetunepilot.train import train as run_training

    try:
        run_cfg = load_run_config(config)
    except FileNotFoundError:
        _exit_usage(f"Config file not found: {config}")
    except (ValidationError, ValueError) as e:
        _exit_usage(f"Invalid config {config}:\n{e}")

    cfg = run_cfg.train
    log = start_run(
        "train", run_cfg.name, dry_run=dry_run, seed=cfg.seed, config=run_cfg.model_dump()
    )
    effective = cfg.for_dry_run() if dry_run else cfg
    attempt = Attempt(
        attempt=1,
        status="running",
        started_at=now_iso(),
        config_changes=diff_configs(cfg, effective),
    )
    log.attempts.append(attempt)
    runs = runs_dir or Path(run_cfg.runs_dir)
    t0 = time.perf_counter()
    path = write_run(log, runs)  # written early, so a crash still leaves a record
    console.print(f"Run [bold]{log.run_id}[/bold] started (dry_run={dry_run}). Log: {path}")

    try:
        result = run_training(
            cfg, output_dir=output_dir or Path(cfg.output_dir) / log.run_id, dry_run=dry_run
        )
    except Exception as e:  # record any failure in the run log, then exit non-zero
        _fail(log, runs, t0, e, "Training", attempt)

    attempt.status = "success"
    attempt.result = result.model_dump()
    log.status = "success"
    log.metrics = _train_metrics(result.model_dump())
    log.artifacts = {"adapter_dir": result.output_dir}
    _finish(log, t0, attempt)
    write_run(log, runs)

    table = _summary_table(log, path)
    for key in ("first_loss", "final_loss", "train_loss", "steps", "peak_gpu_memory_mb"):
        table.add_row(key, _fmt(log.metrics.get(key)))
    table.add_row("adapter", log.artifacts["adapter_dir"])
    console.print(table)


# --------------------------------------------------------------------------- run


@app.command()
def run(
    config: Path = typer.Option(..., "--config", "-c", help="YAML run config."),
    max_attempts: int = typer.Option(4, min=1, max=20, help="Stop after this many attempts."),
    max_minutes: float = typer.Option(60, min=0.1, help="Total time budget for all attempts."),
    force_oom: bool = typer.Option(
        False, "--force-oom", help="Demo: batch size 16 and no pre-check, to trigger OOM recovery."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Tiny model on CPU; OOM is simulated from the VRAM estimate."
    ),
    gpu_memory_gb: float | None = typer.Option(
        None, help="GPU memory in GiB. Default: detected, or a T4 (14.7) if there is no GPU."
    ),
    output_dir: Path | None = typer.Option(
        None, help="Attempt outputs. Default: <train.output_dir>/<run_id>/attempt-N."
    ),
    runs_dir: Path | None = typer.Option(None, help="Where to write the run JSON."),
) -> None:
    """Train with VRAM pre-check, automatic OOM/NaN recovery and a time budget."""
    from finetunepilot.config import load_run_config
    from finetunepilot.runlog import start_run, write_run
    from finetunepilot.runner import (
        RecoveryPolicy,
        detect_gpu_memory_gb,
        make_subprocess_attempt,
        measure_data_stats,
        run_with_recovery,
    )
    from finetunepilot.tools.vram import GPU_MEMORY_GIB

    try:
        run_cfg = load_run_config(config)
    except FileNotFoundError:
        _exit_usage(f"Config file not found: {config}")
    except (ValidationError, ValueError) as e:
        _exit_usage(f"Invalid config {config}:\n{e}")

    cfg = run_cfg.train
    gpu_gb = gpu_memory_gb or detect_gpu_memory_gb() or GPU_MEMORY_GIB["t4"]
    policy = RecoveryPolicy(
        max_attempts=max_attempts, max_minutes=max_minutes, gpu_memory_gb=gpu_gb
    )
    try:
        data_stats = measure_data_stats(cfg, dry_run)
    except Exception as e:  # the pre-check then falls back to max_seq_length
        console.print(
            f"[yellow]Could not measure token lengths ({e}); using max_seq_length.[/yellow]"
        )
        data_stats = None

    log = start_run(
        "train", run_cfg.name, dry_run=dry_run, seed=cfg.seed,
        config={**run_cfg.model_dump(), "policy": policy.model_dump(), "force_oom": force_oom,
                "data_stats": data_stats},
    )  # fmt: skip
    runs = runs_dir or Path(run_cfg.runs_dir)
    t0 = time.perf_counter()
    path = write_run(log, runs)
    console.print(
        f"Run [bold]{log.run_id}[/bold]: up to {max_attempts} attempts / {max_minutes:g} min, "
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
        attempt_fn=make_subprocess_attempt(dry_run, simulate_oom_gpu_gb=gpu_gb, seq_len=seq_len),
        gpu_memory_gb=gpu_gb,
        data_stats=data_stats,
        force_oom=force_oom,
        on_attempt=save_progress,
    )
    log.attempts = outcome.attempts
    log.precheck = outcome.precheck
    # `train` is the YAML as written; these are what the runner actually started and ended with
    # (they differ with --force-oom, or after pre-check / recovery fixes).
    log.config["base_train"] = outcome.base_config.model_dump()
    log.config["final_train"] = outcome.final_config.model_dump()
    log.stop_reason = outcome.stop_reason
    log.status = outcome.status
    if outcome.result:
        log.metrics = {**_train_metrics(outcome.result), "attempts": len(outcome.attempts)}
        log.artifacts = {"adapter_dir": outcome.result["output_dir"]}
    else:
        log.metrics = {"attempts": len(outcome.attempts)}
        log.errors = [a.error for a in outcome.attempts if a.error] or [outcome.stop_reason]
    _finish(log, t0)
    write_run(log, runs)
    _print_attempts(log)
    console.print(f"[bold]{outcome.status}[/bold]: {outcome.stop_reason}. Run log: {path}")
    if outcome.status != "success":
        raise typer.Exit(1)


# --------------------------------------------------------------------------- eval


@app.command("eval")
def eval_cmd(
    base_model: str = typer.Option(..., "--base-model", help="HF model id of the base model."),
    adapter: Path | None = typer.Option(None, "--adapter", help="LoRA adapter dir (from train)."),
    test: Path = typer.Option(Path("data/sample/test.jsonl"), "--test", help="Test JSONL file."),
    name: str | None = typer.Option(None, help="Run name. Default: eval-base / eval-tuned."),
    max_new_tokens: int = typer.Option(128, min=1, help="Fixed generation length cap."),
    batch_size: int = typer.Option(8, min=1),
    load_in_4bit: bool = typer.Option(
        True, "--load-in-4bit/--no-load-in-4bit", help="Load the base model in 4-bit NF4."
    ),
    seed: int = typer.Option(42),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Tiny random model on CPU. Scores are meaningless."
    ),
    runs_dir: Path = typer.Option(Path("runs"), help="Where to write the run JSON."),
    show_failures: int = typer.Option(5, min=0, help="How many wrong rows to print."),
) -> None:
    """Score tool-call accuracy on a test set and write runs/<run_id>.json."""
    from finetunepilot.eval import EvalSettings, default_eval_name, evaluate
    from finetunepilot.runlog import start_run, write_run

    if not test.is_file():
        _exit_usage(f"Test file not found: {test}")
    if adapter is not None and not (adapter / "adapter_config.json").is_file():
        _exit_usage(f"Not a LoRA adapter dir (no adapter_config.json): {adapter}")

    settings = EvalSettings(
        base_model=base_model,
        adapter=str(adapter) if adapter else None,
        test_path=str(test),
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        load_in_4bit=load_in_4bit,
        seed=seed,
        dry_run=dry_run,
    )
    log = start_run(
        "eval", name or default_eval_name(adapter), dry_run=dry_run, seed=seed,
        config=settings.model_dump(),
    )  # fmt: skip
    t0 = time.perf_counter()
    path = write_run(log, runs_dir)
    console.print(f"Eval [bold]{log.run_id}[/bold] started (dry_run={dry_run}). Log: {path}")

    try:
        result = evaluate(settings)
    except Exception as e:
        _fail(log, runs_dir, t0, e, "Evaluation")

    log.status = "success"
    log.metrics = {**result.summary, "peak_gpu_memory_mb": result.peak_gpu_memory_mb}
    log.predictions = result.predictions
    _finish(log, t0)
    write_run(log, runs_dir)

    table = _summary_table(log, path)
    for key in ("n", "json_valid", "tool_name_accuracy", "args_key_match", "args_exact_match",
                "format_ok"):  # fmt: skip
        table.add_row(key, _fmt(log.metrics.get(key)))
    console.print(table)
    _print_per_tool(log.metrics.get("per_tool", {}))
    _print_failures(log.predictions, show_failures)


# --------------------------------------------------------------------------- compare


@app.command()
def compare(
    run_files: list[Path] = typer.Argument(..., help="Two or more runs/<run_id>.json files."),
) -> None:
    """Print a side-by-side table of metrics from run logs (delta = last - first)."""
    from finetunepilot.runlog import read_run
    from finetunepilot.tools.compare import render_comparison

    if len(run_files) < 2:
        _exit_usage("compare needs at least two run files.")
    runs = []
    for path in run_files:
        try:
            runs.append(read_run(path))
        except FileNotFoundError:
            _exit_usage(f"Run file not found: {path}")
        except ValidationError as e:
            _exit_usage(f"Not a valid run log: {path}\n{e}")
    console.print(render_comparison(runs))


# --------------------------------------------------------------------------- helpers


def _exit_usage(message: str) -> None:
    console.print(f"[red]{message}[/red]")
    raise typer.Exit(2)


def _finish(log: RunLog, t0: float, attempt: Attempt | None = None) -> None:
    from finetunepilot.runlog import now_iso

    log.finished_at = now_iso()
    log.duration_s = round(time.perf_counter() - t0, 2)
    if attempt is not None:
        attempt.duration_s = log.duration_s


def _fail(
    log: RunLog, runs_dir: Path, t0: float, e: Exception, what: str, attempt: Attempt | None = None
) -> None:
    """Record an exception in the run log, save it, and exit with code 1."""
    from finetunepilot.runlog import write_run

    message = f"{type(e).__name__}: {e}"
    if attempt is not None:
        attempt.status = "failed"
        attempt.error = message
    log.status = "failed"
    log.errors.append(message)
    log.errors.append("".join(traceback.format_exception(e))[-4000:])
    _finish(log, t0, attempt)
    path = write_run(log, runs_dir)
    console.print(f"[red]{what} failed:[/red] {message}\nRun log: {path}")
    raise typer.Exit(1)


def _train_metrics(result: dict) -> dict:
    history = result.get("loss_history") or []
    return {
        "first_loss": history[0]["loss"] if history else None,
        **{
            k: result.get(k)
            for k in (
                "final_loss", "train_loss", "steps", "train_runtime_s",
                "peak_gpu_memory_mb", "peak_gpu_memory_reserved_mb",
            )
        },
    }  # fmt: skip


def _print_attempts(log: RunLog) -> None:
    pre = log.precheck
    if pre.get("status") in {"fits", "adjusted", "wont_fit"}:
        before, after = pre["estimate_before"], pre["estimate_after"]
        console.print(
            f"VRAM pre-check ({pre['status']}): estimate {before['total_gb']:.2f} GiB -> "
            f"{after['total_gb']:.2f} GiB, limit {after['limit_gb']:.2f} GiB"
        )
    elif pre:
        console.print(f"VRAM pre-check skipped: {pre.get('reason')}")
    table = Table(title="Attempts")
    for col in ("#", "status", "error", "fix applied before this attempt", "seconds"):
        table.add_column(col, overflow="fold")
    for a in log.attempts:
        table.add_row(
            str(a.attempt), a.status, a.error_class or "-", a.fix or "-", str(a.duration_s)
        )
    console.print(table)


def _fmt(value: object) -> str:
    return f"{value:.4f}" if isinstance(value, float) else str(value)


def _summary_table(log: RunLog, path: Path) -> Table:
    table = Table(title=f"Run {log.run_id}", show_header=False)
    table.add_row("status", log.status)
    table.add_row("dry run", str(log.dry_run))
    table.add_row("duration (s)", str(log.duration_s))
    table.add_row("run log", str(path))
    return table


def _print_per_tool(per_tool: dict) -> None:
    if not per_tool:
        return
    table = Table(title="Per tool")
    for col in ("tool", "n", "tool_name_accuracy", "args_exact_match"):
        table.add_column(col)
    for tool, m in per_tool.items():
        table.add_row(tool, str(m["n"]), _fmt(m["tool_name_accuracy"]), _fmt(m["args_exact_match"]))
    console.print(table)


def _print_failures(predictions: list[dict], limit: int) -> None:
    wrong = [p for p in predictions if not p["args_exact_match"]]
    if not wrong or limit == 0:
        return
    console.print(
        f"[bold]{len(wrong)} rows not exactly right[/bold] (showing {min(limit, len(wrong))}):"
    )
    for p in wrong[:limit]:
        console.print(f"- {p['id']}: {p['request']}")
        console.print(f"  expected: {p['target']}")
        console.print(f"  got:      {p['output']!r}", markup=False)


if __name__ == "__main__":
    app()
