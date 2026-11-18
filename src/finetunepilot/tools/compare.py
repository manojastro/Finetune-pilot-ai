"""Compare two or more run logs side by side.

`compare_metrics` is a pure function (easy to test); `render_comparison` turns it into a
rich Table for the CLI. Only scalar metrics are compared; nested ones like `per_tool` are skipped.
"""

from __future__ import annotations

from typing import Any

from rich.table import Table

from finetunepilot.runlog import RunLog
from finetunepilot.tools.scoring import METRIC_KEYS

# Shown first, in this order; any other scalar metrics follow alphabetically.
PREFERRED_ORDER = [
    *METRIC_KEYS,
    "n",
    "first_loss",
    "final_loss",
    "train_loss",
    "steps",
    "peak_gpu_memory_mb",
    "duration_s",
]


def _is_number(v: Any) -> bool:
    return isinstance(v, int | float) and not isinstance(v, bool)


def compare_metrics(metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per scalar metric: {"metric", "values": [...], "delta"}.

    `delta` is last run minus first run, when both are numbers; otherwise None.
    A run that lacks a metric has None in its slot.
    """
    keys = {k for m in metrics for k, v in m.items() if v is None or _is_number(v)}
    ordered = [k for k in PREFERRED_ORDER if k in keys] + sorted(keys - set(PREFERRED_ORDER))
    rows = []
    for key in ordered:
        values = [m.get(key) for m in metrics]
        first, last = values[0], values[-1]
        delta = round(last - first, 4) if _is_number(first) and _is_number(last) else None
        rows.append({"metric": key, "values": values, "delta": delta})
    return rows


def _fmt(v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def render_comparison(runs: list[RunLog]) -> Table:
    table = Table(title="Run comparison")
    table.add_column("metric")
    for i, run in enumerate(runs, start=1):
        table.add_column(f"{i}. {run.name}", overflow="fold")
    if len(runs) >= 2:
        table.add_column("delta (last - first)")

    def header(label: str, values: list[str]) -> None:
        table.add_row(label, *values, *([""] if len(runs) >= 2 else []))

    header("run id", [r.run_id for r in runs])
    header("kind", [r.kind for r in runs])
    header("status", [r.status for r in runs])
    header("dry run", [str(r.dry_run) for r in runs])
    header("adapter", [str(r.config.get("adapter") or "-") for r in runs])
    table.add_section()

    for row in compare_metrics([r.metrics for r in runs]):
        delta = [] if len(runs) < 2 else [_fmt(row["delta"])]
        table.add_row(row["metric"], *[_fmt(v) for v in row["values"]], *delta)
    if any(r.dry_run for r in runs):
        table.caption = "Includes a dry run: those numbers come from a random tiny model."
    return table
