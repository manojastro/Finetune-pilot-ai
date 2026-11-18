"""Typer command-line interface: `finetunepilot <command>`."""

from __future__ import annotations

import time
import traceback
from pathlib import Path

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

from finetunepilot import __version__

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
    # Heavy imports (torch, transformers) happen here so `finetunepilot version` stays fast.
    from finetunepilot.config import diff_configs, load_run_config
    from finetunepilot.runlog import (
        Attempt,
        RunLog,
        collect_hardware,
        git_commit,
        library_versions,
        new_run_id,
        now_iso,
        write_run,
    )
    from finetunepilot.train import train as run_training

    try:
        run_cfg = load_run_config(config)
    except FileNotFoundError:
        console.print(f"[red]Config file not found:[/red] {config}")
        raise typer.Exit(2) from None
    except (ValidationError, ValueError) as e:
        console.print(f"[red]Invalid config {config}:[/red]\n{e}")
        raise typer.Exit(2) from None

    cfg = run_cfg.train
    effective = cfg.for_dry_run() if dry_run else cfg
    run_id = new_run_id(f"{run_cfg.name}-dryrun" if dry_run else run_cfg.name)
    out_dir = output_dir or Path(cfg.output_dir) / run_id
    runs = runs_dir or Path(run_cfg.runs_dir)

    log = RunLog(
        run_id=run_id,
        kind="train",
        name=run_cfg.name,
        dry_run=dry_run,
        seed=cfg.seed,
        git_commit=git_commit(),
        library_versions=library_versions(),
        hardware=collect_hardware(),
        config=run_cfg.model_dump(),
    )
    attempt = Attempt(
        attempt=1,
        status="running",
        started_at=now_iso(),
        config_changes=diff_configs(cfg, effective),
    )
    log.attempts.append(attempt)
    t0 = time.perf_counter()
    path = write_run(log, runs)  # written early, so a crash still leaves a record
    console.print(f"Run [bold]{run_id}[/bold] started (dry_run={dry_run}). Log: {path}")

    try:
        result = run_training(cfg, output_dir=out_dir, dry_run=dry_run)
    except Exception as e:  # record any failure in the run log, then exit non-zero
        attempt.status = "failed"
        attempt.error = f"{type(e).__name__}: {e}"
        log.status = "failed"
        log.errors.append(attempt.error)
        log.errors.append("".join(traceback.format_exception(e))[-4000:])
        _finish(log, attempt, t0)
        write_run(log, runs)
        console.print(f"[red]Training failed:[/red] {attempt.error}\nRun log: {path}")
        raise typer.Exit(1) from None

    attempt.status = "success"
    attempt.result = result.model_dump()
    log.status = "success"
    log.metrics = {
        k: getattr(result, k)
        for k in (
            "final_loss", "train_loss", "steps", "train_runtime_s",
            "peak_gpu_memory_mb", "peak_gpu_memory_reserved_mb",
        )
    }  # fmt: skip
    log.metrics["first_loss"] = result.loss_history[0]["loss"] if result.loss_history else None
    log.artifacts = {"adapter_dir": result.output_dir}
    _finish(log, attempt, t0)
    write_run(log, runs)
    _print_summary(log, path)


def _finish(log, attempt, t0: float) -> None:
    from finetunepilot.runlog import now_iso

    log.finished_at = now_iso()
    log.duration_s = round(time.perf_counter() - t0, 2)
    attempt.duration_s = log.duration_s


def _print_summary(log, path: Path) -> None:
    table = Table(title=f"Run {log.run_id}", show_header=False)
    table.add_row("status", log.status)
    table.add_row("dry run", str(log.dry_run))
    for key in ("first_loss", "final_loss", "train_loss", "steps", "peak_gpu_memory_mb"):
        value = log.metrics.get(key)
        table.add_row(key, f"{value:.4f}" if isinstance(value, float) else str(value))
    table.add_row("duration (s)", str(log.duration_s))
    table.add_row("adapter", log.artifacts.get("adapter_dir", "-"))
    table.add_row("run log", str(path))
    console.print(table)


if __name__ == "__main__":
    app()
