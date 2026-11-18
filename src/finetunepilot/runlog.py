"""Writes and reads runs/<run_id>.json run logs.

Every training or eval run writes one JSON file (CLAUDE.md rule 4) with: config, hardware,
each attempt, metrics, status, errors, duration, git commit, library versions and seed.
Eval-only runs have an empty `attempts` list and store per-row `predictions`.
"""

from __future__ import annotations

import platform
import secrets
import subprocess
import sys
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

RunStatus = Literal["running", "success", "failed"]

TRACKED_PACKAGES = [
    "finetunepilot", "torch", "transformers", "trl", "peft", "datasets",
    "accelerate", "bitsandbytes",
]  # fmt: skip


class Attempt(BaseModel):
    """One try at training. `finetunepilot run` may make several; `train` makes exactly one."""

    attempt: int = Field(ge=1)
    status: Literal["running", "success", "failed", "timeout"]
    started_at: str
    duration_s: float | None = None
    config_changes: dict[str, Any] = Field(default_factory=dict)  # vs. the run's base config
    fix: str | None = None  # what the runner changed before this attempt, and why
    error_class: str | None = None
    evidence: str | None = None  # the log line that identified error_class
    error: str | None = None
    result: dict[str, Any] = Field(default_factory=dict)


class RunLog(BaseModel):
    run_id: str
    kind: Literal["train", "eval"]
    name: str
    status: RunStatus = "running"
    dry_run: bool = False
    created_at: str = Field(default_factory=lambda: now_iso())
    finished_at: str | None = None
    duration_s: float | None = None
    seed: int | None = None
    git_commit: str | None = None
    library_versions: dict[str, str | None] = Field(default_factory=dict)
    hardware: dict[str, Any] = Field(default_factory=dict)
    config: dict[str, Any] = Field(default_factory=dict)
    attempts: list[Attempt] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)
    artifacts: dict[str, str] = Field(default_factory=dict)
    predictions: list[dict[str, Any]] = Field(default_factory=list)  # eval runs only
    precheck: dict[str, Any] = Field(default_factory=dict)  # VRAM pre-check (finetunepilot run)
    stop_reason: str | None = None  # why a multi-attempt run ended


def start_run(
    kind: Literal["train", "eval"],
    name: str,
    *,
    dry_run: bool,
    seed: int | None,
    config: dict[str, Any],
) -> RunLog:
    """A new 'running' log with the environment filled in (git, versions, hardware)."""
    return RunLog(
        run_id=new_run_id(f"{name}-dryrun" if dry_run else name),
        kind=kind,
        name=name,
        dry_run=dry_run,
        seed=seed,
        git_commit=git_commit(),
        library_versions=library_versions(),
        hardware=collect_hardware(),
        config=config,
    )


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def new_run_id(name: str) -> str:
    """e.g. 'qwen05b_toolcall-20261118-130501-a1b2' (sortable by time, unique)."""
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return f"{name}-{stamp}-{secrets.token_hex(2)}"


def run_path(runs_dir: str | Path, run_id: str) -> Path:
    return Path(runs_dir) / f"{run_id}.json"


def write_run(log: RunLog, runs_dir: str | Path) -> Path:
    """Write the run log atomically (temp file + rename), so a crash never leaves half a file."""
    path = run_path(runs_dir, log.run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(log.model_dump_json(indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def read_run(path: str | Path) -> RunLog:
    return RunLog.model_validate_json(Path(path).read_text(encoding="utf-8"))


def library_versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for pkg in TRACKED_PACKAGES:
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = None
    return out


def git_commit(cwd: str | Path | None = None) -> str | None:
    """Current git commit hash, with '-dirty' if there are uncommitted changes. None if no git."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=cwd, capture_output=True, text=True, check=True,
        ).stdout.strip()  # fmt: skip
    except (OSError, subprocess.CalledProcessError):
        return None
    return f"{sha}-dirty" if dirty else sha


def collect_hardware() -> dict[str, Any]:
    """Describe the machine. GPU fields are None on CPU-only machines."""
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "device": "cpu",
        "gpu_name": None,
        "gpu_count": 0,
        "gpu_memory_gb": None,
        "cuda_version": None,
    }
    try:
        import torch
    except ImportError:
        return info
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info.update(
            device="cuda",
            gpu_name=props.name,
            gpu_count=torch.cuda.device_count(),
            gpu_memory_gb=round(props.total_memory / 1024**3, 2),
            cuda_version=torch.version.cuda,
        )
    return info
