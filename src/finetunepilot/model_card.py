"""Fill the Hugging Face model card template from run logs (only measured numbers).

    finetunepilot model-card --train-run runs/A.json --base-eval runs/B.json \
        --tuned-eval runs/C.json --repo-id you/qwen25-05b-toolcall-qlora

Dry-run logs are refused: their numbers are simulated. Missing values read "not yet measured".
Nothing is uploaded; the command prints the exact `hf upload` command to run yourself.
"""

from __future__ import annotations

from importlib.resources import files
from typing import Any

from finetunepilot.runlog import RunLog

NOT_MEASURED = "not yet measured"
METRICS = ["json_valid", "tool_name_accuracy", "args_key_match", "args_exact_match", "format_ok"]
TRAINING_FIELDS = [
    "lora_r", "lora_alpha", "lora_dropout", "target_modules", "learning_rate", "num_epochs",
    "per_device_batch_size", "grad_accum_steps", "max_seq_length", "gradient_checkpointing",
    "load_in_4bit", "fp16", "bf16", "seed",
]  # fmt: skip


class ModelCardError(ValueError):
    pass


def _fmt(value: Any) -> str:
    if value is None:
        return NOT_MEASURED
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, list):
        return ", ".join(map(str, value))
    return str(value)


def _check(log: RunLog, kind: str, label: str) -> None:
    if log.kind != kind:
        raise ModelCardError(f"{label} must be a {kind} run log, got kind={log.kind!r}")
    if log.dry_run:
        raise ModelCardError(f"{label} ({log.run_id}) is a dry run: its numbers are simulated")
    if log.status != "success":
        raise ModelCardError(f"{label} ({log.run_id}) did not succeed (status={log.status})")


def render_model_card(train: RunLog, base_eval: RunLog, tuned_eval: RunLog, repo_id: str) -> str:
    _check(train, "train", "train run")
    _check(base_eval, "eval", "base eval")
    _check(tuned_eval, "eval", "tuned eval")
    if base_eval.config.get("adapter"):
        raise ModelCardError("base eval must be run without an adapter")
    if not tuned_eval.config.get("adapter"):
        raise ModelCardError("tuned eval must be run with the adapter")

    cfg = train.config.get("final_train") or train.config.get("train") or {}
    results_rows = "\n".join(
        f"| `{m}` | {_fmt(base_eval.metrics.get(m))} | {_fmt(tuned_eval.metrics.get(m))} |"
        for m in METRICS
    )
    hw = train.hardware
    training = [(f, _fmt(cfg.get(f))) for f in TRAINING_FIELDS]
    training += [
        ("GPU", _fmt(hw.get("gpu_name"))),
        ("optimizer steps", _fmt(train.metrics.get("steps"))),
        ("first / final logged loss",
         f"{_fmt(train.metrics.get('first_loss'))} / {_fmt(train.metrics.get('final_loss'))}"),
        ("training time (s)", _fmt(train.metrics.get("train_runtime_s"))),
        ("peak GPU memory reserved (MiB)", _fmt(train.metrics.get("peak_gpu_memory_reserved_mb"))),
        ("attempts (OOM recovery)", _fmt(len(train.attempts) or None)),
    ]  # fmt: skip
    template = (files("finetunepilot") / "templates" / "model_card.md").read_text(encoding="utf-8")
    return template.format(
        base_model=cfg.get("base_model", NOT_MEASURED),
        repo_name=repo_id.split("/")[-1],
        repo_id=repo_id,
        test_rows=_fmt(tuned_eval.metrics.get("n")),
        results_rows=results_rows,
        training_rows="\n".join(f"| {k} | {v} |" for k, v in training),
        base_eval_id=base_eval.run_id,
        tuned_eval_id=tuned_eval.run_id,
        train_run_id=train.run_id,
        git_commit=_fmt(train.git_commit),
    )


def upload_command(repo_id: str, adapter_dir: str) -> str:
    """The command to push the adapter + card. training_args.bin is a pickle; leave it out."""
    return (
        f'hf upload {repo_id} "{adapter_dir}" . --private --exclude "training_args.bin" '
        '--commit-message "Add FineTunePilot QLoRA adapter"'
    )
