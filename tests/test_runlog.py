"""Run log write/read."""

import json

from finetunepilot.runlog import (
    Attempt,
    RunLog,
    collect_hardware,
    library_versions,
    new_run_id,
    now_iso,
    read_run,
    run_path,
    write_run,
)


def _log(**kw) -> RunLog:
    return RunLog(run_id=new_run_id("t"), kind="train", name="t", **kw)


def test_round_trip(tmp_path) -> None:
    log = _log(
        seed=7,
        config={"train": {"lora_r": 16}},
        attempts=[
            Attempt(
                attempt=1,
                status="success",
                started_at=now_iso(),
                config_changes={"fp16": {"from": True, "to": False}},
                result={"final_loss": 0.5},
            )
        ],
        metrics={"final_loss": 0.5, "peak_gpu_memory_mb": None},
        status="success",
    )
    path = write_run(log, tmp_path / "runs")
    assert path == run_path(tmp_path / "runs", log.run_id)
    assert read_run(path) == log
    assert not list((tmp_path / "runs").glob("*.tmp"))  # atomic write leaves no temp file


def test_file_has_the_required_fields(tmp_path) -> None:
    data = json.loads(write_run(_log(), tmp_path).read_text())
    for key in [
        "config", "hardware", "attempts", "metrics", "status", "errors", "duration_s",
        "git_commit", "library_versions", "seed",
    ]:  # fmt: skip
        assert key in data


def test_eval_run_may_have_no_attempts(tmp_path) -> None:
    log = RunLog(run_id="e1", kind="eval", name="base", status="success")
    assert read_run(write_run(log, tmp_path)).attempts == []


def test_failed_run_keeps_errors(tmp_path) -> None:
    log = _log(status="failed", errors=["RuntimeError: boom"])
    assert read_run(write_run(log, tmp_path)).errors == ["RuntimeError: boom"]


def test_run_ids_are_unique_and_prefixed() -> None:
    ids = {new_run_id("qwen") for _ in range(50)}
    assert len(ids) == 50
    assert all(i.startswith("qwen-") for i in ids)


def test_environment_info_on_cpu() -> None:
    hw = collect_hardware()
    assert hw["device"] in {"cpu", "cuda"}
    if hw["device"] == "cpu":
        assert hw["gpu_name"] is None and hw["gpu_memory_gb"] is None
    versions = library_versions()
    assert versions["torch"] and versions["finetunepilot"]
