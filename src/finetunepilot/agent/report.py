"""Write reports/<session_id>.md from the agent state.

Only numbers copied from run logs appear (CLAUDE.md rule 8). Anything missing is shown as
"not measured". A dry-run report says clearly that its numbers are simulated.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

METRICS = ["json_valid", "tool_name_accuracy", "args_exact_match"]


def _fmt(value: Any) -> str:
    if value is None:
        return "not measured"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _changes_text(changes: dict[str, Any] | None) -> str:
    if not changes:
        return "-"
    return ", ".join(f"{k}={v}" for k, v in changes.items())


def _diff(prev: dict[str, Any] | None, new: dict[str, Any]) -> dict[str, Any]:
    if prev is None:
        return {}
    return {k: v for k, v in new.items() if prev.get(k) != v}


def best_run(state: dict[str, Any]) -> dict[str, Any] | None:
    from finetunepilot.agent.nodes import score

    scored = [r for r in state.get("runs", []) if r.get("metrics")]
    return max(scored, key=lambda r: score(r["metrics"]), default=None)


def render_report(state: dict[str, Any]) -> str:
    s = state["settings"]
    runs = state.get("runs", [])
    base = (state.get("baseline") or {}).get("metrics") or {}
    lines = [f"# FineTunePilot session `{state['session_id']}`", ""]
    if s.get("dry_run"):
        lines += [
            "> **DRY RUN.** Training and evaluation were simulated with a toy model of config "
            "quality. These numbers are NOT measurements.",
            "",
        ]
    lines += [
        f"- **Objective:** {s['objective']}",
        f"- **Base model:** `{s['base_model']}` (QLoRA, 4-bit)",
        f"- **Dataset:** `{s['dataset_dir']}`",
        f"- **GPU:** {s['gpu']}",
        f"- **Budget:** {s['max_runs']} runs, {s['max_minutes']:g} minutes",
        f"- **Status:** {state.get('status')}",
        f"- **Stopped because:** {state.get('stop_reason') or state.get('error') or '-'}",
        "",
        "## Results",
        "",
        "| run | config change | train | attempts | final loss | "
        + " | ".join(METRICS)
        + " | run logs |",
        "|---|---|---|---|---|" + "---|" * len(METRICS) + "---|",
        "| base | - (no adapter) | - | - | - | "
        + " | ".join(_fmt(base.get(m)) for m in METRICS)
        + f" | `{(state.get('baseline') or {}).get('eval_run_id', '-')}` |",
    ]
    prev = None
    for r in runs:
        metrics = r.get("metrics") or {}
        change = r["final_tunables"] if prev is None else _diff(prev, r["final_tunables"])
        logs = f"`{r['train_run_id']}`" + (
            f", `{r['eval_run_id']}`" if r.get("eval_run_id") else ""
        )
        change_cell = _changes_text(change) if prev else "initial plan"
        final_loss = _fmt((r.get("train_metrics") or {}).get("final_loss"))
        cells = [str(r["index"]), change_cell, r["train_status"], str(len(r.get("attempts", []))),
                 final_loss, *(_fmt(metrics.get(m)) for m in METRICS), logs]  # fmt: skip
        lines.append("| " + " | ".join(cells) + " |")
        prev = r["final_tunables"]

    best = best_run(state)
    lines += ["", "## Best run", ""]
    if best:
        b, a = best["metrics"].get("args_exact_match"), base.get("args_exact_match")
        delta = (
            f" ({b - a:+.4f} vs base {a:.4f})"
            if isinstance(a, float | int) and b is not None
            else ""
        )
        lines += [
            f"Run {best['index']}: args_exact_match {_fmt(b)}{delta}, "
            f"tool_name_accuracy {_fmt(best['metrics'].get('tool_name_accuracy'))}.",
            f"Adapter: `{best.get('adapter_dir')}`",
        ]
    else:
        lines.append("No run produced evaluated results.")

    lines += ["", "## What changed and why", ""]
    prev = None
    for r in runs:
        lines.append(f"### Run {r['index']} ({r['source']})")
        lines.append(f"- Reason: {r['reason']}")
        shown = r["final_tunables"] if prev is None else _diff(prev, r["final_tunables"])
        lines.append(f"- Config: {_changes_text(shown) if shown else '(no change)'}")
        if r.get("estimate_gb") is not None:
            lines.append(f"- VRAM estimate before training: {r['estimate_gb']} GiB (an estimate)")
        fixes = [a for a in r.get("attempts", []) if a.get("fix")]
        for a in fixes:
            lines.append(f"- Attempt {a['attempt']} ({a['status']}): {a['fix']}")
        failed = [a for a in r.get("attempts", []) if a.get("error_class")]
        for a in failed:
            lines.append(f"- Attempt {a['attempt']} failed with {a['error_class']}")
        lines.append(f"- Training: {r['train_status']} ({r.get('train_stop_reason')})")
        if r.get("diagnosis"):
            d = r["diagnosis"]
            lines.append(f"- Diagnosis: set `{d['field']}` = `{d['value']}`. {d['reason']}")
        prev = r["final_tunables"]
        lines.append("")
    if state.get("error"):
        lines += ["## Error", "", f"`{state['error']}`", ""]
    return "\n".join(lines).rstrip() + "\n"


def write_report(state: dict[str, Any]) -> Path:
    path = Path(state["settings"]["reports_dir"]) / f"{state['session_id']}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_report(state), encoding="utf-8")
    return path
