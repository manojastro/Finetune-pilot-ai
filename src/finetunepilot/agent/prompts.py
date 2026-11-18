"""Prompts for the two LLM decisions: plan a first config, and diagnose a finished run.

Prompts contain only facts produced by our own tools (dataset stats, VRAM estimate, measured
metrics, failed examples) and ask for one small JSON object.
"""

from __future__ import annotations

import json
from typing import Any

SYSTEM = (
    "You are the planning component of FineTunePilot, which fine-tunes small language models "
    "with QLoRA. Other code trains, evaluates and writes files; you never run anything. Your "
    "only job is to choose training hyperparameters and explain results, using the facts you "
    "are given. Reply with a single JSON object and nothing else."
)

FIELD_RULES = """Tunable fields (anything else is set by the system):
- learning_rate: 1e-6 .. 1e-2 (QLoRA usually 5e-5 .. 5e-4)
- num_epochs: > 0
- lora_r: 1 .. 512; lora_alpha: 1 .. 1024 (often 2 x lora_r); lora_dropout: 0 .. <1
- per_device_batch_size: 1 .. 1024; grad_accum_steps: 1 .. 1024
- max_seq_length: 32 .. 32768, and at least the longest training row ({longest} tokens)
- gradient_checkpointing: true/false (saves memory, ~20% slower)
- warmup_ratio: 0 .. <1"""


def _j(data: Any) -> str:
    return json.dumps(data, indent=1, sort_keys=True, default=str)


def plan_prompt(state: dict[str, Any], defaults: dict[str, Any]) -> str:
    s = state["settings"]
    ds = state["dataset"]
    feedback = state.get("plan_feedback")
    parts = [
        "TASK: PLAN the first fine-tuning run.",
        f"Objective: {s['objective']}",
        f"Base model: {s['base_model']} (4-bit QLoRA)",
        f"GPU: {s['gpu']} with {ds['gpu_memory_gb']} GiB usable",
        f"Budget: at most {s['max_runs']} runs and {s['max_minutes']:g} minutes in total.",
        f"Dataset stats:\n{_j(ds['stats'])}",
        f"Base model scores before any training (0..1):\n{_j(state['baseline'].get('metrics'))}",
        f"Default values (used for fields you leave out):\n{_j(defaults)}",
        FIELD_RULES.format(longest=ds["stats"].get("max_tokens", "unknown")),
    ]
    if feedback:
        parts.append(f"Your previous plan was rejected by the memory check: {feedback}")
    parts.append(
        'Answer format: {"changes": {<field>: <value>, ...}, "reason": "<why, 1-3 sentences>"}'
    )
    return "\n\n".join(parts)


def diagnose_prompt(state: dict[str, Any], run: dict[str, Any], failures: list[dict]) -> str:
    s = state["settings"]
    history = [
        {
            "run": r["index"],
            "config": r.get("final_tunables"),
            "args_exact_match": (r.get("metrics") or {}).get("args_exact_match"),
            "tool_name_accuracy": (r.get("metrics") or {}).get("tool_name_accuracy"),
            "final_loss": (r.get("train_metrics") or {}).get("final_loss"),
        }
        for r in state.get("runs", [])
    ]
    metrics = {k: v for k, v in (run.get("metrics") or {}).items() if k != "per_tool"}
    parts = [
        f"TASK: DIAGNOSE run {run['index']} and suggest ONE config change for the next run.",
        f"Objective: {s['objective']}",
        f"Config of this run:\n{_j(run.get('final_tunables'))}",
        f"Training: {_j(run.get('train_metrics'))}",
        f"Test metrics (0..1):\n{_j(metrics)}",
        f"Per tool:\n{_j((run.get('metrics') or {}).get('per_tool'))}",
        f"Base model scores: {_j(state['baseline'].get('metrics'))}",
        f"All runs so far:\n{_j(history)}",
        f"Examples the model got wrong (up to 5):\n{_j(failures)}",
        FIELD_RULES.format(longest=state["dataset"]["stats"].get("max_tokens", "unknown")),
        'Answer format: {"field": "<one tunable field>", "value": <new value>, '
        '"reason": "<what the metrics show and why this change should help>"}',
    ]
    return "\n\n".join(parts)
