"""Node functions of the agent graph. Each takes (state, deps) and returns a state update.

    prepare -> baseline -> plan -> estimate -> approve -> train -> evaluate -> diagnose -> decide
                             ^        |                                                  |
                             +- no fit+          (next run: decide -> estimate) <--------+
                                                                       report <- (stop / error)

The LLM is used only in `plan` and `diagnose`, and its answers are validated before use.
All training, evaluation and file work is done by the deterministic toolbox.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from finetunepilot.agent.llm import LLMClient, ask_json
from finetunepilot.agent.prompts import SYSTEM, diagnose_prompt, plan_prompt
from finetunepilot.agent.state import (
    AgentSettings,
    AgentState,
    ConfigChange,
    PlanProposal,
    build_train_config,
    tunable_values,
)
from finetunepilot.agent.toolbox import Toolbox
from finetunepilot.config import RunConfig, TrainConfig
from finetunepilot.data import load_rows
from finetunepilot.tools.vram import estimate_vram

MAX_PLAN_TRIES = 3  # plans rejected by the memory check before giving up
MIN_MINUTES_FOR_A_RUN = 1.0


@dataclass
class AgentDeps:
    llm: LLMClient
    toolbox: Toolbox
    approver: Callable[[str], bool]  # shows the plan, returns True to go ahead
    say: Callable[[str], None] = print
    clock: Callable[[], float] = time.time
    llm_log: Callable[[dict[str, Any]], None] = field(default=lambda entry: None)


def settings_of(state: AgentState) -> AgentSettings:
    return AgentSettings.model_validate(state["settings"])


def score(metrics: dict[str, Any] | None) -> tuple[float, float, float]:
    """How runs are ranked: exact calls first, then tool choice, then valid JSON."""
    m = metrics or {}
    return (
        m.get("args_exact_match") or 0.0,
        m.get("tool_name_accuracy") or 0.0,
        m.get("json_valid") or 0.0,
    )


def minutes_left(state: AgentState, deps: AgentDeps) -> float:
    used = (deps.clock() - state["started_at"]) / 60
    return settings_of(state).max_minutes - used


def _run_dir(state: AgentState, index: int) -> str:
    return str(Path(settings_of(state).output_root) / state["session_id"] / f"run-{index}")


def _config_for(state: AgentState, changes: dict[str, Any], index: int) -> TrainConfig:
    return build_train_config(settings_of(state), changes, _run_dir(state, index))


def _llm_logger(deps: AgentDeps, node: str):
    return lambda entry: deps.llm_log({"node": node, **entry})


# ----------------------------------------------------------------------------- nodes


def prepare(state: AgentState, deps: AgentDeps) -> dict:
    s = settings_of(state)
    train_rows, test_rows = load_rows(s.train_path), load_rows(s.test_path)
    tools: dict[str, int] = {}
    for row in train_rows:
        tools[row.target.name] = tools.get(row.target.name, 0) + 1
    token_stats = deps.toolbox.data_stats(_config_for(state, {}, 0))
    stats = {
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "tool_calls_per_tool": tools,
        **token_stats,
    }
    deps.say(
        f"Dataset: {len(train_rows)} train / {len(test_rows)} test rows, "
        f"longest {stats['max_tokens']} tokens."
    )
    return {
        "dataset": {"stats": stats, "gpu_memory_gb": s.gpu_memory_gb},
        "runs": [],
        "plan_tries": 0,
    }


def baseline(state: AgentState, deps: AgentDeps) -> dict:
    s = settings_of(state)
    deps.say("Evaluating the base model (no adapter) as the score to beat...")
    log = deps.toolbox.evaluate(s.base_model, None, s.test_path, f"{state['session_id']}-base")
    if log.status != "success":
        raise RuntimeError(f"baseline evaluation failed: {log.errors[:1]}")
    return {"baseline": {"eval_run_id": log.run_id, "metrics": log.metrics}}


def plan(state: AgentState, deps: AgentDeps) -> dict:
    tries = state.get("plan_tries", 0) + 1
    if tries > MAX_PLAN_TRIES:
        raise RuntimeError(f"no plan passed the memory check after {MAX_PLAN_TRIES} tries")
    defaults = tunable_values(_config_for(state, {}, 1))
    proposal, _ = ask_json(
        deps.llm, SYSTEM, plan_prompt(state, defaults), PlanProposal,
        check=lambda p: _check_seq(state, _config_for(state, p.changes, 1)),
        log=_llm_logger(deps, "plan"),
    )  # fmt: skip
    deps.say(f"Plan: {proposal.changes} -- {proposal.reason}")
    return {
        "pending": {"changes": proposal.changes, "reason": proposal.reason, "source": "plan"},
        "plan_tries": tries,
        "plan_feedback": None,
    }


def _check_seq(state: AgentState, cfg: TrainConfig) -> TrainConfig:
    longest = state["dataset"]["stats"].get("max_tokens")
    if longest and cfg.max_seq_length < longest:
        raise ValueError(
            f"max_seq_length {cfg.max_seq_length} is below the longest row ({longest} tokens); "
            "rows would be truncated"
        )
    return cfg


def estimate(state: AgentState, deps: AgentDeps) -> dict:
    index = len(state.get("runs", [])) + 1
    cfg = _config_for(state, state["pending"]["changes"], index)
    s = settings_of(state)
    try:
        est = estimate_vram(
            cfg, s.gpu_memory_gb, seq_len=state["dataset"]["stats"].get("max_tokens")
        )
    except KeyError as e:  # architecture not in the built-in table: can't estimate
        return {"estimate": {"fits": True, "skipped": str(e)}}
    info = est.model_dump()
    if est.fits:
        return {"estimate": info}
    feedback = (
        f"estimated {est.total_gb:.2f} GiB but the limit is {est.limit_gb:.2f} GiB "
        f"(batch {cfg.per_device_batch_size}, seq {est.seq_len}, gradient_checkpointing "
        f"{cfg.gradient_checkpointing}; biggest terms: activations "
        f"{est.breakdown_gb['activations']} GiB, logits {est.breakdown_gb['logits']} GiB). "
        "Use a smaller batch with more grad "
        "accumulation, or gradient checkpointing."
    )
    deps.say(f"Memory check failed: {feedback}")
    return {"estimate": info, "plan_feedback": feedback}


def _eta_text(state: AgentState, cfg: TrainConfig) -> str:
    """Time estimate from the measured speed of earlier runs in this session, if any."""
    measured = [
        (r["train_metrics"]["train_runtime_s"], r["train_metrics"]["steps"])
        for r in state.get("runs", [])
        if (r.get("train_metrics") or {}).get("train_runtime_s") and r["train_metrics"].get("steps")
    ]
    rows = state["dataset"]["stats"]["train_rows"]
    steps = math.ceil(rows / cfg.effective_batch_size) * math.ceil(cfg.num_epochs)
    if not measured:
        return f"~{steps} optimizer steps; time not yet measured in this session"
    sec_per_step = sum(t for t, _ in measured) / sum(n for _, n in measured)
    minutes = steps * sec_per_step / 60
    return f"~{steps} steps x {sec_per_step:.1f} s/step (measured) = ~{minutes:.1f} min"


def approve(state: AgentState, deps: AgentDeps) -> dict:
    index = len(state.get("runs", [])) + 1
    pending = state["pending"]
    cfg = _config_for(state, pending["changes"], index)
    est = state.get("estimate", {})
    text = "\n".join(
        [
            f"Run {index} of at most {settings_of(state).max_runs} "
            f"({minutes_left(state, deps):.0f} min left)",
            f"  why: {pending['reason']}",
            f"  config: {tunable_values(cfg)}",
            f"  VRAM estimate: {est.get('total_gb', 'n/a')} GiB "
            f"(limit {est.get('limit_gb', 'n/a')} GiB)",
            f"  time: {_eta_text(state, cfg)}",
        ]
    )
    if deps.approver(text):
        return {}
    return {"status": "rejected", "stop_reason": f"run {index} rejected by the user"}


def train(state: AgentState, deps: AgentDeps) -> dict:
    s = settings_of(state)
    index = len(state.get("runs", [])) + 1
    pending = state["pending"]
    cfg = _config_for(state, pending["changes"], index)
    budget = minutes_left(state, deps)
    run_cfg = RunConfig(name=f"{state['session_id']}-run{index}", runs_dir=s.runs_dir, train=cfg)
    deps.say(f"Training run {index} (budget {budget:.1f} min)...")
    log = deps.toolbox.train(
        run_cfg,
        max_minutes=budget,
        max_attempts=s.max_attempts_per_run,
        gpu_memory_gb=s.gpu_memory_gb,
    )
    final = log.config.get("final_train") or cfg.model_dump()
    run = {
        "index": index,
        "source": pending["source"],
        "reason": pending["reason"],
        "requested": pending["changes"],
        "final_tunables": tunable_values(final),
        "estimate_gb": state.get("estimate", {}).get("total_gb"),
        "train_run_id": log.run_id,
        "train_status": log.status,
        "train_stop_reason": log.stop_reason,
        "attempts": [
            {"attempt": a.attempt, "status": a.status, "error_class": a.error_class, "fix": a.fix}
            for a in log.attempts
        ],
        "train_metrics": log.metrics,
        "adapter_dir": log.artifacts.get("adapter_dir"),
        "train_duration_s": log.duration_s,
    }
    return {"runs": [*state.get("runs", []), run]}


def evaluate(state: AgentState, deps: AgentDeps) -> dict:
    s = settings_of(state)
    runs = [dict(r) for r in state["runs"]]
    run = runs[-1]
    log = deps.toolbox.evaluate(
        s.base_model,
        run["adapter_dir"],
        s.test_path,
        f"{state['session_id']}-run{run['index']}-eval",
    )
    run.update(
        eval_run_id=log.run_id,
        eval_status=log.status,
        metrics=log.metrics if log.status == "success" else None,
    )
    if log.status == "success":
        failures = [p for p in log.predictions if not p.get("args_exact_match")][:5]
        run["failures"] = [{k: p.get(k) for k in ("request", "target", "output")} for p in failures]
        deps.say(f"Run {run['index']}: args_exact_match={log.metrics.get('args_exact_match')}")
    return {"runs": runs}


def diagnose(state: AgentState, deps: AgentDeps) -> dict:
    runs = [dict(r) for r in state["runs"]]
    run = runs[-1]
    current = run["final_tunables"]

    def check(change: ConfigChange) -> dict[str, Any]:
        if current.get(change.field) == change.value:
            raise ValueError(
                f"{change.field} is already {change.value}; suggest a different change"
            )
        new = {**current, change.field: change.value}
        _check_seq(state, _config_for(state, new, run["index"] + 1))  # validates ranges too
        return new

    change, new_tunables = ask_json(
        deps.llm, SYSTEM, diagnose_prompt(state, run, run.get("failures", [])), ConfigChange,
        check=check, log=_llm_logger(deps, "diagnose"),
    )  # fmt: skip
    run["diagnosis"] = change.model_dump()
    deps.say(f"Diagnosis: set {change.field}={change.value} -- {change.reason}")
    return {"runs": runs, "next_change": {**change.model_dump(), "tunables": new_tunables}}


def decide(state: AgentState, deps: AgentDeps) -> dict:
    s = settings_of(state)
    runs = state["runs"]
    last = runs[-1]
    best_before = max(
        [score(state["baseline"]["metrics"])] + [score(r.get("metrics")) for r in runs[:-1]]
    )
    reasons = []
    if last["train_status"] != "success":
        reasons.append(f"run {last['index']} training failed ({last['train_stop_reason']})")
    elif not last.get("metrics"):
        reasons.append(f"run {last['index']} evaluation failed")
    elif not score(last["metrics"]) > best_before:
        reasons.append(f"run {last['index']} did not improve on the best score so far")
    if len(runs) >= s.max_runs:
        reasons.append(f"max runs ({s.max_runs}) reached")
    if minutes_left(state, deps) < MIN_MINUTES_FOR_A_RUN:
        reasons.append(f"time budget ({s.max_minutes:g} min) used up")
    if not state.get("next_change"):
        reasons.append("no next change suggested")
    if reasons:
        return {"stop_reason": "; ".join(reasons)}
    change = state["next_change"]
    return {
        "pending": {
            "changes": change["tunables"],
            "reason": change["reason"],
            "source": "diagnose",
        },
        "next_change": None,
        "stop_reason": None,
    }


def report(state: AgentState, deps: AgentDeps) -> dict:
    from finetunepilot.agent.report import write_report

    status = state.get("status") or "running"
    if state.get("error"):
        status = "failed"
    elif status == "running":
        status = "completed"
    final = {**state, "status": status}
    path = write_report(final)
    deps.say(f"Report: {path}")
    return {"status": status, "report_path": str(path)}


def route_after_estimate(state: AgentState) -> str:
    if state.get("error"):
        return "report"
    return "approve" if state.get("estimate", {}).get("fits") else "plan"


def route_after_approve(state: AgentState) -> str:
    return "report" if state.get("error") or state.get("status") == "rejected" else "train"


def route_after_train(state: AgentState) -> str:
    if state.get("error"):
        return "report"
    return "evaluate" if state["runs"][-1]["train_status"] == "success" else "decide"


def route_after_evaluate(state: AgentState) -> str:
    if state.get("error"):
        return "report"
    return "diagnose" if state["runs"][-1].get("metrics") else "decide"


def route_after_decide(state: AgentState) -> str:
    return "report" if state.get("error") or state.get("stop_reason") else "estimate"


def json_safe(state: AgentState) -> dict[str, Any]:
    return json.loads(json.dumps(state, default=str))
