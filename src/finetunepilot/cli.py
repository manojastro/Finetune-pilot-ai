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
    from finetunepilot.jobs import train_metrics
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
    log.metrics = train_metrics(result.model_dump())
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
    from finetunepilot.jobs import run_training_job

    try:
        run_cfg = load_run_config(config)
    except FileNotFoundError:
        _exit_usage(f"Config file not found: {config}")
    except (ValidationError, ValueError) as e:
        _exit_usage(f"Invalid config {config}:\n{e}")

    log, path = run_training_job(
        run_cfg,
        max_attempts=max_attempts,
        max_minutes=max_minutes,
        dry_run=dry_run,
        force_oom=force_oom,
        gpu_memory_gb=gpu_memory_gb,
        output_dir=output_dir,
        runs_dir=runs_dir,
        say=console.print,
    )
    _print_attempts(log)
    console.print(f"[bold]{log.status}[/bold]: {log.stop_reason}. Run log: {path}")
    if log.status != "success":
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
    from finetunepilot.eval import EvalSettings, default_eval_name
    from finetunepilot.jobs import run_eval_job

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
    log, path = run_eval_job(
        settings, name=name or default_eval_name(adapter), runs_dir=runs_dir, say=console.print
    )
    if log.status != "success":
        console.print(f"[red]Evaluation failed:[/red] {log.errors[0]}\nRun log: {path}")
        raise typer.Exit(1)

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


# --------------------------------------------------------------------------- agent


@app.command()
def agent(
    objective: str | None = typer.Option(
        None, "--objective", help="What the fine-tune should achieve."
    ),
    model: str = typer.Option("Qwen/Qwen2.5-0.5B-Instruct", "--model", help="Base model (HF id)."),
    dataset: Path = typer.Option(
        Path("data/sample"), "--dataset", help="Dir with train/test.jsonl."
    ),
    gpu: str = typer.Option("t4", "--gpu", help="GPU type: t4, p100, l4, a10g, a100-40gb."),
    max_runs: int = typer.Option(3, min=1, max=10, help="At most this many training runs."),
    max_minutes: float = typer.Option(90, min=1, help="Time budget for the whole session."),
    yes: bool = typer.Option(False, "--yes", help="Approve every run without asking."),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Rule-based fake LLM + simulated training/eval. Runs in seconds."
    ),
    resume: str | None = typer.Option(None, "--resume", help="Continue a session by its id."),
    agent_config: Path = typer.Option(Path("configs/agent.yaml"), help="LLM settings file."),
    runs_dir: Path = typer.Option(Path("runs")),
    reports_dir: Path = typer.Option(Path("reports")),
    sessions_dir: Path = typer.Option(Path("runs/sessions")),
) -> None:
    """Let the agent plan, run, diagnose and compare fine-tuning runs, then write a report."""
    from finetunepilot.agent.graph import load_session_settings, resume_session, run_session
    from finetunepilot.agent.llm import load_llm_settings
    from finetunepilot.agent.state import AgentSettings

    if resume:
        try:
            settings = load_session_settings(resume, sessions_dir)
        except FileNotFoundError:
            _exit_usage(f"No session {resume!r} in {sessions_dir}")
        if yes:
            settings = settings.model_copy(update={"auto_approve": True})
        final = resume_session(resume, _agent_deps(settings), sessions_dir)
    else:
        if not objective:
            _exit_usage("--objective is required (or use --resume <session_id>).")
        for split in ("train.jsonl", "test.jsonl"):
            if not (dataset / split).is_file():
                _exit_usage(f"Missing {dataset / split}")
        try:
            settings = AgentSettings(
                objective=objective, base_model=model, dataset_dir=str(dataset), gpu=gpu,
                max_runs=max_runs, max_minutes=max_minutes, dry_run=dry_run, auto_approve=yes,
                llm=load_llm_settings(agent_config), runs_dir=str(runs_dir),
                reports_dir=str(reports_dir), sessions_dir=str(sessions_dir),
            )  # fmt: skip
        except (ValidationError, ValueError, FileNotFoundError) as e:
            _exit_usage(f"Invalid agent settings: {e}")
        if not dry_run:
            console.print(
                f"[yellow]This session calls the Anthropic API ({settings.llm.model}), which is "
                "billed, and trains on the GPU.[/yellow]"
            )
        final = run_session(settings, _agent_deps(settings))

    console.print(
        f"[bold]Session {final['session_id']}: {final.get('status')}[/bold] - "
        f"{final.get('stop_reason') or final.get('error') or ''}\n"
        f"Report: {final.get('report_path')}"
    )
    if final.get("status") == "failed":
        raise typer.Exit(1)


def _agent_deps(settings):
    from finetunepilot.agent.factory import build_deps

    def approver(text: str) -> bool:
        console.print(text, markup=False)
        if settings.auto_approve:
            console.print("Approved (--yes).")
            return True
        return typer.confirm("Start this run?", default=False)

    return build_deps(settings, approver, say=console.print)


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
