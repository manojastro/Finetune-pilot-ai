"""One-attempt worker and the subprocess attempt runner."""

import json
import sys

import pytest

from finetunepilot.config import load_run_config
from finetunepilot.runner import make_subprocess_attempt
from finetunepilot.tools.log_parser import classify_error
from finetunepilot.worker import main, run_attempt
from tests.conftest import CONFIG_PATH


@pytest.fixture
def cfg():
    return load_run_config(CONFIG_PATH).train


def _fake_worker(code: str) -> list[str]:
    return [sys.executable, "-c", code]


def test_in_process_simulated_oom_is_reported(cfg, tmp_path) -> None:
    big = cfg.model_copy(update={"per_device_batch_size": 16})
    request = {"config": big.model_dump(), "output_dir": str(tmp_path / "a"), "dry_run": True,
               "simulate_oom_gpu_gb": 14.7, "seq_len": 762}  # fmt: skip
    assert run_attempt(request, tmp_path / "result.json") == 1
    payload = json.loads((tmp_path / "result.json").read_text())
    assert payload["status"] == "failed" and payload["error_type"] == "OutOfMemoryError"
    assert classify_error(payload["error"]).error_class == "CUDA_OOM"


def test_in_process_bad_config_is_reported(tmp_path) -> None:
    assert run_attempt({"config": {"lora_r": 0}}, tmp_path / "r.json") == 1
    assert json.loads((tmp_path / "r.json").read_text())["error_type"] == "ValidationError"


def test_worker_usage_error() -> None:
    assert main([]) == 2


def test_crash_without_result_is_classified_from_output(cfg, tmp_path) -> None:
    crash = _fake_worker(
        "print('torch.OutOfMemoryError: CUDA out of memory.'); raise SystemExit(3)"
    )
    attempt = make_subprocess_attempt(dry_run=True, echo=False, command=crash)
    outcome = attempt(cfg, tmp_path / "attempt-1", timeout_s=30)
    assert outcome.status == "failed" and "exited with code 3" in outcome.error
    assert classify_error(outcome.log_tail).error_class == "CUDA_OOM"
    assert "CUDA out of memory" in (tmp_path / "attempt-1" / "attempt.log").read_text()
    request = json.loads((tmp_path / "attempt-1" / "request.json").read_text())
    assert request["config"]["per_device_batch_size"] == 4 and request["dry_run"] is True


def test_attempt_is_killed_at_the_time_budget(cfg, tmp_path) -> None:
    attempt = make_subprocess_attempt(
        dry_run=True, echo=False, command=_fake_worker("import time; time.sleep(30)")
    )
    outcome = attempt(cfg, tmp_path / "attempt-1", timeout_s=0.5)
    assert outcome.status == "timeout" and "time budget" in outcome.error


@pytest.mark.slow
def test_real_worker_subprocess_simulated_oom_then_success(cfg, tmp_path) -> None:
    attempt = make_subprocess_attempt(
        dry_run=True, simulate_oom_gpu_gb=14.7, seq_len=762, echo=False
    )
    oom = attempt(
        cfg.model_copy(update={"per_device_batch_size": 16}), tmp_path / "a1", timeout_s=120
    )
    assert oom.status == "failed"
    assert classify_error(f"{oom.error}\n{oom.log_tail}").error_class == "CUDA_OOM"

    ok = attempt(cfg, tmp_path / "a2", timeout_s=120)
    assert ok.status == "success", ok.error
    assert ok.result["steps"] == 2 and ok.data_stats["max_tokens"] > 0
    assert (tmp_path / "a2" / "adapter" / "adapter_model.safetensors").exists()
