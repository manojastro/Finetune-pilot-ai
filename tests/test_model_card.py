"""Model card rendering from run logs (no upload)."""

import pytest
import yaml
from typer.testing import CliRunner

from finetunepilot.cli import app
from finetunepilot.config import load_run_config
from finetunepilot.model_card import (
    NOT_MEASURED,
    ModelCardError,
    render_model_card,
    upload_command,
)
from finetunepilot.runlog import Attempt, RunLog, write_run
from tests.conftest import CONFIG_PATH


def _logs(adapter="outputs/x/run1/adapter"):
    train_cfg = load_run_config(CONFIG_PATH).train.model_dump()
    train = RunLog(
        run_id="train-1", kind="train", name="t", status="success", git_commit="abc123",
        hardware={"gpu_name": "Tesla T4"}, config={"final_train": train_cfg},
        attempts=[Attempt(attempt=1, status="success", started_at="t")],
        metrics={"steps": 40, "first_loss": 2.1, "final_loss": 0.12, "train_runtime_s": None},
    )  # fmt: skip
    base = RunLog(
        run_id="eval-base-1", kind="eval", name="b", status="success",
        config={"adapter": None}, metrics={"n": 20, "json_valid": 0.55, "args_exact_match": 0.1},
    )  # fmt: skip
    tuned = RunLog(
        run_id="eval-tuned-1", kind="eval", name="t", status="success",
        config={"adapter": adapter}, metrics={"n": 20, "json_valid": 1.0, "args_exact_match": 0.85},
    )  # fmt: skip
    return train, base, tuned


def test_card_contains_only_numbers_from_the_logs() -> None:
    card = render_model_card(*_logs(), repo_id="me/qwen-toolcall")
    front = yaml.safe_load(card.split("---")[1])
    assert front["base_model"] == "Qwen/Qwen2.5-0.5B-Instruct" and front["library_name"] == "peft"
    assert "| `args_exact_match` | 0.1000 | 0.8500 |" in card
    assert "| `json_valid` | 0.5500 | 1.0000 |" in card
    assert f"| `tool_name_accuracy` | {NOT_MEASURED} | {NOT_MEASURED} |" in card
    assert f"| training time (s) | {NOT_MEASURED} |" in card
    assert "| lora_r | 16 |" in card and "| GPU | Tesla T4 |" in card
    assert "eval-base-1" in card and "eval-tuned-1" in card and "abc123" in card
    assert 'PeftModel.from_pretrained(base, "me/qwen-toolcall")' in card
    assert "{" not in card.split("## Results")[1].split("## How to use")[0].replace("{json}", "")


def test_dry_run_and_mismatched_logs_are_refused() -> None:
    train, base, tuned = _logs()
    with pytest.raises(ModelCardError, match="dry run"):
        render_model_card(train.model_copy(update={"dry_run": True}), base, tuned, "me/x")
    with pytest.raises(ModelCardError, match="train run log"):
        render_model_card(base, base, tuned, "me/x")
    with pytest.raises(ModelCardError, match="without an adapter"):
        render_model_card(train, tuned, tuned, "me/x")
    with pytest.raises(ModelCardError, match="did not succeed"):
        render_model_card(train, base, tuned.model_copy(update={"status": "failed"}), "me/x")


def test_upload_command_is_private_and_skips_pickles() -> None:
    cmd = upload_command("me/x", "outputs/a b/adapter")
    assert cmd.startswith('hf upload me/x "outputs/a b/adapter" . --private')
    assert '--exclude "training_args.bin"' in cmd


def test_cli_writes_the_card_and_uploads_nothing(tmp_path) -> None:
    adapter = tmp_path / "adapter"
    train, base, tuned = _logs(adapter=str(adapter))
    paths = [write_run(log, tmp_path / "runs") for log in (train, base, tuned)]
    result = CliRunner().invoke(
        app, ["model-card", "--train-run", str(paths[0]), "--base-eval", str(paths[1]),
              "--tuned-eval", str(paths[2]), "--repo-id", "me/qwen-toolcall"],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert (adapter / "README.md").read_text().startswith("---")
    assert "Nothing was uploaded" in result.output and "hf upload me/qwen-toolcall" in result.output

    args = ["model-card", "--train-run", str(paths[1]), "--base-eval", str(paths[1]),
            "--tuned-eval", str(paths[2]), "--repo-id", "me/x"]  # fmt: skip
    bad = CliRunner().invoke(app, args)
    assert bad.exit_code == 2 and "Cannot write the model card" in bad.output
