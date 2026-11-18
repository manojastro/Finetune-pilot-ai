"""Evaluation: generation bookkeeping, dry run, and CLI."""

import json

import pytest
import torch
from typer.testing import CliRunner

import finetunepilot.eval as eval_mod
from finetunepilot.cli import app
from finetunepilot.data import format_example, load_rows
from finetunepilot.eval import EvalSettings, clean_generation, default_eval_name, evaluate
from tests.conftest import CONFIG_PATH, TEST_PATH

runner = CliRunner()


def test_clean_generation_cuts_at_first_stop_string() -> None:
    assert (
        clean_generation("<tool_call>x</tool_call><|im_end|>junk", ["<|im_end|>"])
        == "<tool_call>x</tool_call>"
    )
    assert clean_generation("a<|endoftext|>b<|im_end|>", ["<|im_end|>", "<|endoftext|>"]) == "a"
    assert clean_generation("  no stop  \n", ["<|im_end|>", ""]) == "no stop"


def test_default_eval_name() -> None:
    assert default_eval_name(None) == "eval-base"
    assert default_eval_name("outputs/x") == "eval-tuned"


class OracleModel(torch.nn.Module):
    """Fake model whose generate() appends the correct completion for each prompt.

    Running it through evaluate() must score 1.0 on everything, which checks the batching,
    left padding, prompt slicing and decoding around the model.
    """

    def __init__(self, tokenizer, rows):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))
        self.tok = tokenizer
        self.answers = {r.request: format_example(r, tokenizer)["completion"] for r in rows}

    def generate(self, input_ids, attention_mask, generation_config):
        new = []
        for ids in input_ids:
            prompt = self.tok.decode(ids, skip_special_tokens=False)
            answer = next(a for req, a in self.answers.items() if req in prompt)
            new.append(
                self.tok(answer, add_special_tokens=False)["input_ids"]
                + [self.tok.eos_token_id] * 3
            )
        width = max(map(len, new))
        new = [seq + [self.tok.pad_token_id] * (width - len(seq)) for seq in new]
        return torch.cat([input_ids, torch.tensor(new)], dim=1)


def test_perfect_model_scores_one(monkeypatch, tiny_tokenizer) -> None:
    rows = load_rows(TEST_PATH)
    monkeypatch.setattr(
        eval_mod, "_load_dry_run", lambda s: (OracleModel(tiny_tokenizer, rows), tiny_tokenizer)
    )
    result = evaluate(
        EvalSettings(base_model="tiny", test_path=str(TEST_PATH), dry_run=True, batch_size=6)
    )
    s = result.summary
    assert s["n"] == 20
    for key in (
        "json_valid",
        "tool_name_accuracy",
        "args_key_match",
        "args_exact_match",
        "format_ok",
    ):
        assert s[key] == 1.0, key
    assert [p["id"] for p in result.predictions] == [
        r.id for r in rows
    ]  # order kept across batches
    assert result.predictions[0]["output"].endswith("</tool_call>")


@pytest.mark.slow
def test_dry_run_evaluate_random_model() -> None:
    result = evaluate(EvalSettings(base_model="tiny", test_path=str(TEST_PATH), dry_run=True))
    assert result.summary["n"] == 20 and len(result.predictions) == 20
    for key in ("json_valid", "tool_name_accuracy", "args_exact_match"):
        assert 0.0 <= result.summary[key] <= 1.0
    assert result.peak_gpu_memory_mb is None
    first = result.predictions[0]
    assert {"id", "request", "target", "output", "json_valid", "args_exact_match"} <= set(first)


def test_real_eval_without_gpu_fails_clearly() -> None:
    if torch.cuda.is_available():
        pytest.skip("machine has a GPU")
    with pytest.raises(RuntimeError, match="No CUDA GPU"):
        evaluate(EvalSettings(base_model="Qwen/Qwen2.5-0.5B-Instruct", test_path=str(TEST_PATH)))


def test_cli_eval_input_errors(tmp_path) -> None:
    missing = runner.invoke(
        app, ["eval", "--base-model", "x", "--test", str(tmp_path / "nope.jsonl")]
    )
    assert missing.exit_code == 2
    not_adapter = runner.invoke(app, ["eval", "--base-model", "x", "--adapter", str(tmp_path)])
    assert not_adapter.exit_code == 2


def test_cli_eval_failure_is_logged(tmp_path) -> None:
    if torch.cuda.is_available():
        pytest.skip("machine has a GPU")
    runs = tmp_path / "runs"
    result = runner.invoke(
        app, ["eval", "--base-model", "Qwen/Qwen2.5-0.5B-Instruct", "--runs-dir", str(runs)]
    )
    assert result.exit_code == 1
    (path,) = runs.glob("*.json")
    log = json.loads(path.read_text())
    assert log["status"] == "failed" and log["kind"] == "eval"


@pytest.mark.slow
def test_cli_train_then_eval_adapter_dry_run(tmp_path) -> None:
    """End to end on CPU: train a dry-run adapter, then evaluate base and tuned, then compare."""
    runs, adapter = tmp_path / "runs", tmp_path / "adapter"
    train = runner.invoke(
        app,
        ["train", "-c", str(CONFIG_PATH), "--dry-run", "--runs-dir", str(runs),
         "--output-dir", str(adapter)],
    )  # fmt: skip
    assert train.exit_code == 0, train.output
    base = runner.invoke(
        app, ["eval", "--base-model", "tiny", "--dry-run", "--runs-dir", str(runs)]
    )
    tuned = runner.invoke(
        app,
        ["eval", "--base-model", "tiny", "--adapter", str(adapter), "--dry-run",
         "--runs-dir", str(runs)],
    )  # fmt: skip
    assert base.exit_code == 0, base.output
    assert tuned.exit_code == 0, tuned.output

    (base_path,) = runs.glob("eval-base-*.json")
    (tuned_path,) = runs.glob("eval-tuned-*.json")
    log = json.loads(tuned_path.read_text())
    assert log["kind"] == "eval" and log["status"] == "success" and log["attempts"] == []
    assert log["config"]["adapter"] == str(adapter)
    assert len(log["predictions"]) == 20 and log["metrics"]["n"] == 20

    cmp = runner.invoke(app, ["compare", str(base_path), str(tuned_path)])
    assert cmp.exit_code == 0 and "args_exact_match" in cmp.output
