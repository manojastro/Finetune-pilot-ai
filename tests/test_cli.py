"""CLI commands."""

import json

import pytest
import yaml
from typer.testing import CliRunner

from finetunepilot import __version__
from finetunepilot.cli import app
from tests.conftest import CONFIG_PATH

runner = CliRunner()


def test_version_command_prints_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.output.strip() == f"finetunepilot {__version__}"


def test_train_missing_config_exits_2(tmp_path) -> None:
    result = runner.invoke(app, ["train", "--config", str(tmp_path / "nope.yaml")])
    assert result.exit_code == 2
    assert "not found" in result.output


def test_train_invalid_config_exits_2(tmp_path) -> None:
    bad = yaml.safe_load(CONFIG_PATH.read_text())
    bad["train"]["lora_r"] = 0
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(bad))
    result = runner.invoke(app, ["train", "--config", str(path)])
    assert result.exit_code == 2
    assert "lora_r" in result.output


def test_failed_training_is_written_to_the_run_log(tmp_path) -> None:
    import torch

    if torch.cuda.is_available():
        pytest.skip("machine has a GPU")
    runs = tmp_path / "runs"
    result = runner.invoke(
        app, ["train", "--config", str(CONFIG_PATH), "--runs-dir", str(runs),
              "--output-dir", str(tmp_path / "out")],
    )  # fmt: skip
    assert result.exit_code == 1
    (path,) = runs.glob("*.json")
    log = json.loads(path.read_text())
    assert log["status"] == "failed"
    assert log["attempts"][0]["status"] == "failed"
    assert "No CUDA GPU" in log["errors"][0]


@pytest.mark.slow
def test_train_dry_run_writes_run_json(tmp_path) -> None:
    runs = tmp_path / "runs"
    result = runner.invoke(
        app,
        ["train", "--config", str(CONFIG_PATH), "--dry-run", "--runs-dir", str(runs),
         "--output-dir", str(tmp_path / "adapter")],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    (path,) = runs.glob("*.json")
    log = json.loads(path.read_text())
    assert log["status"] == "success" and log["dry_run"] is True
    assert "-dryrun-" in log["run_id"]
    assert log["metrics"]["steps"] == 2
    assert log["config"]["train"]["fp16"] is True  # base config is logged as-is ...
    assert log["attempts"][0]["config_changes"]["fp16"] == {
        "from": True,
        "to": False,
    }  # ... + changes
    assert log["artifacts"]["adapter_dir"] == str(tmp_path / "adapter")
    assert (tmp_path / "adapter" / "adapter_model.safetensors").exists()


def test_run_refuses_to_start_when_estimate_cannot_fit(tmp_path) -> None:
    runs = tmp_path / "runs"
    result = runner.invoke(
        app, ["run", "-c", str(CONFIG_PATH), "--dry-run", "--gpu-memory-gb", "1",
              "--runs-dir", str(runs), "--output-dir", str(tmp_path / "out")],
    )  # fmt: skip
    assert result.exit_code == 1, result.output
    (path,) = runs.glob("*.json")
    log = json.loads(path.read_text())
    assert log["status"] == "failed" and log["attempts"] == []
    assert log["stop_reason"].startswith("won't fit") and log["precheck"]["status"] == "wont_fit"


@pytest.mark.slow
def test_run_force_oom_dry_run_recovers(tmp_path) -> None:
    runs = tmp_path / "runs"
    result = runner.invoke(
        app, ["run", "-c", str(CONFIG_PATH), "--dry-run", "--force-oom", "--gpu-memory-gb", "14.7",
              "--runs-dir", str(runs), "--output-dir", str(tmp_path / "out")],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    (path,) = runs.glob("*.json")
    log = json.loads(path.read_text())
    assert log["status"] == "success" and log["stop_reason"] == "succeeded on attempt 3"
    assert [a["error_class"] for a in log["attempts"]] == ["CUDA_OOM", "CUDA_OOM", None]
    assert log["precheck"]["status"] == "skipped" and log["config"]["force_oom"] is True
    assert log["config"]["base_train"]["per_device_batch_size"] == 16
    assert log["config"]["final_train"]["gradient_checkpointing"] is True
    last = log["attempts"][-1]["config_changes"]
    assert last["per_device_batch_size"] == {"from": 16, "to": 8}
    assert log["metrics"]["attempts"] == 3 and log["metrics"]["steps"] == 2
    assert log["artifacts"]["adapter_dir"].endswith("attempt-3/adapter")
