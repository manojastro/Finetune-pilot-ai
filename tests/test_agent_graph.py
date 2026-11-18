"""The agent graph end to end, with fake LLMs and simulated training (no GPU, no network)."""

import json

import pytest
from typer.testing import CliRunner

from finetunepilot.agent.graph import SessionStore, resume_session, run_session
from finetunepilot.agent.llm import FakeLLM, LLMSettings
from finetunepilot.agent.nodes import AgentDeps
from finetunepilot.agent.state import AgentSettings, PlanProposal, build_train_config
from finetunepilot.agent.toolbox import DryRunLLM, SimulatedToolbox, simulated_quality
from finetunepilot.cli import app
from tests.conftest import REPO_ROOT

PLAN = {
    "changes": {"learning_rate": 1e-4, "num_epochs": 3, "lora_r": 8},
    "reason": "A safe first config.",
}


def plan_json(**changes) -> str:
    return json.dumps({"changes": changes or PLAN["changes"], "reason": "A safe first config."})


def change_json(field, value) -> str:
    return json.dumps(
        {"field": field, "value": value, "reason": f"Metrics suggest changing {field}."}
    )


def make_settings(tmp_path, **kw) -> AgentSettings:
    base = dict(
        objective="tool calling with valid JSON", base_model="Qwen/Qwen2.5-0.5B-Instruct",
        dataset_dir=str(REPO_ROOT / "data" / "sample"), dry_run=True, auto_approve=True,
        llm=LLMSettings(model="test-model"), runs_dir=str(tmp_path / "runs"),
        reports_dir=str(tmp_path / "reports"), sessions_dir=str(tmp_path / "sessions"),
        output_root=str(tmp_path / "out"),
    )  # fmt: skip
    return AgentSettings(**{**base, **kw})


class CountingToolbox(SimulatedToolbox):
    """Simulated toolbox that counts calls, can move a fake clock, and can crash once."""

    def __init__(self, runs_dir, clock=None, minutes_per_train=0.0, crash_on_train=None):
        super().__init__(runs_dir, say=lambda m: None)
        self.trains, self.evals = 0, 0
        self.clock, self.minutes_per_train, self.crash_on_train = (
            clock,
            minutes_per_train,
            crash_on_train,
        )

    def train(self, run_cfg, **kw):
        self.trains += 1
        if self.crash_on_train == self.trains:
            self.crash_on_train = None
            raise Crash("simulated machine crash")
        if self.clock:
            self.clock.t += self.minutes_per_train * 60
        return super().train(run_cfg, **kw)

    def evaluate(self, *args, **kw):
        self.evals += 1
        return super().evaluate(*args, **kw)


class Crash(BaseException):
    """Not an Exception, so the node wrapper does not catch it: like the process dying."""


class Clock:
    def __init__(self):
        self.t = 1_000_000.0

    def __call__(self):
        return self.t


def deps_for(settings, llm, toolbox=None, approver=lambda text: True, clock=None):
    return AgentDeps(
        llm=llm, toolbox=toolbox or CountingToolbox(settings.runs_dir), approver=approver,
        say=lambda m: None, clock=clock or Clock(),
    )  # fmt: skip


def test_full_session_with_fake_llm(tmp_path) -> None:
    settings = make_settings(tmp_path)
    llm = FakeLLM([plan_json(), change_json("learning_rate", 2e-4), change_json("lora_r", 16),
                   change_json("num_epochs", 5)])  # fmt: skip
    final = run_session(settings, deps_for(settings, llm))
    assert final["status"] == "completed" and not final.get("error")
    assert final["stop_reason"] == "max runs (3) reached"
    runs = final["runs"]
    assert [r["final_tunables"]["learning_rate"] for r in runs] == [1e-4, 2e-4, 2e-4]
    assert runs[2]["final_tunables"]["lora_r"] == 16
    assert [r["source"] for r in runs] == ["plan", "diagnose", "diagnose"]
    assert all(r["train_status"] == "success" and r["metrics"]["n"] == 20 for r in runs)
    assert (
        llm.calls[0]["prompt"].startswith("TASK: PLAN")
        and "TASK: DIAGNOSE" in llm.calls[1]["prompt"]
    )

    report = (tmp_path / "reports" / f"{final['session_id']}.md").read_text()
    assert "DRY RUN" in report and "| base |" in report and "## Best run" in report
    assert "Run 3:" in report and runs[2]["train_run_id"] in report
    summary = SessionStore(settings.sessions_dir).read_summary(final["session_id"])
    assert summary["status"] == "completed" and summary["last_node"] == "report"
    # every train and eval wrote a run log: base eval + 3 x (train + eval)
    assert len(list((tmp_path / "runs").glob("*.json"))) == 7


def test_invalid_llm_json_is_retried_once(tmp_path) -> None:
    settings = make_settings(tmp_path, max_runs=1)
    llm = FakeLLM(["I think lr 1e-4 is good", plan_json(), change_json("lora_r", 16)])
    final = run_session(settings, deps_for(settings, llm))
    assert final["status"] == "completed" and len(final["runs"]) == 1
    log = (tmp_path / "sessions" / f"{final['session_id']}.llm.jsonl").read_text().splitlines()
    assert [json.loads(line)["valid"] for line in log][:2] == [False, True]


def test_invalid_llm_json_twice_fails_the_session_with_a_report(tmp_path) -> None:
    settings = make_settings(tmp_path)
    llm = FakeLLM(['{"changes": {"base_model": "gpt"}, "reason": "use another model"}', "nope"])
    final = run_session(settings, deps_for(settings, llm))
    assert final["status"] == "failed" and "invalid twice" in final["error"]
    assert final["error"].startswith("plan:") and final["runs"] == []
    assert "## Error" in (tmp_path / "reports" / f"{final['session_id']}.md").read_text()


def test_plan_that_does_not_fit_vram_goes_back_to_plan(tmp_path) -> None:
    settings = make_settings(tmp_path, max_runs=1)
    llm = FakeLLM([plan_json(per_device_batch_size=64), plan_json(), change_json("lora_r", 16)])
    final = run_session(settings, deps_for(settings, llm))
    assert final["status"] == "completed"
    assert "rejected by the memory check" in llm.calls[1]["prompt"]
    assert final["runs"][0]["final_tunables"]["per_device_batch_size"] == 4  # the second plan


def test_plans_that_never_fit_end_the_session(tmp_path) -> None:
    settings = make_settings(tmp_path)
    llm = FakeLLM([plan_json(per_device_batch_size=64)] * 3)
    final = run_session(settings, deps_for(settings, llm))
    assert final["status"] == "failed" and "no plan passed the memory check" in final["error"]


def test_time_budget_stops_the_session(tmp_path) -> None:
    settings = make_settings(tmp_path, max_runs=5, max_minutes=60)
    clock = Clock()
    toolbox = CountingToolbox(settings.runs_dir, clock=clock, minutes_per_train=35)
    llm = FakeLLM([plan_json(), change_json("learning_rate", 2e-4), change_json("lora_r", 16)])
    final = run_session(settings, deps_for(settings, llm, toolbox=toolbox, clock=clock))
    assert toolbox.trains == 2 and len(final["runs"]) == 2
    assert "time budget (60 min) used up" in final["stop_reason"]


def test_no_improvement_stops_the_session(tmp_path) -> None:
    settings = make_settings(tmp_path, max_runs=5)
    llm = FakeLLM([plan_json(), change_json("learning_rate", 5e-3), change_json("lora_r", 16)])
    final = run_session(settings, deps_for(settings, llm))
    assert len(final["runs"]) == 2
    assert "run 2 did not improve" in final["stop_reason"]


def test_diagnose_must_change_something(tmp_path) -> None:
    settings = make_settings(tmp_path, max_runs=2)
    llm = FakeLLM([plan_json(), change_json("learning_rate", 1e-4), change_json("lora_r", 16),
                   change_json("num_epochs", 5)])  # fmt: skip
    final = run_session(settings, deps_for(settings, llm))
    assert "already" in llm.calls[2]["prompt"]  # the retry explains the problem
    assert final["runs"][1]["final_tunables"]["lora_r"] == 16


def test_rejected_run_stops_with_a_report(tmp_path) -> None:
    settings = make_settings(tmp_path, auto_approve=False)
    shown = []
    final = run_session(
        settings, deps_for(settings, DryRunLLM(), approver=lambda t: (shown.append(t), False)[1])
    )
    assert final["status"] == "rejected" and final["runs"] == []
    assert "VRAM estimate" in shown[0] and "time not yet measured" in shown[0]
    assert (tmp_path / "reports" / f"{final['session_id']}.md").exists()


def test_resume_after_a_crash_continues_from_the_last_node(tmp_path) -> None:
    settings = make_settings(tmp_path)
    llm = FakeLLM([plan_json(), change_json("learning_rate", 2e-4), change_json("lora_r", 16),
                   change_json("num_epochs", 5)])  # fmt: skip
    toolbox = CountingToolbox(settings.runs_dir, crash_on_train=2)
    with pytest.raises(Crash):
        run_session(settings, deps_for(settings, llm, toolbox=toolbox))
    (summary,) = SessionStore(settings.sessions_dir).list_summaries()
    assert summary["status"] == "running" and len(summary["runs"]) == 1

    final = resume_session(
        summary["session_id"], deps_for(settings, llm, toolbox=toolbox), settings.sessions_dir
    )
    assert final["status"] == "completed" and len(final["runs"]) == 3
    assert toolbox.evals == 4  # base + 3 runs: run 1 and the baseline were not repeated
    again = resume_session(summary["session_id"], deps_for(settings, llm), settings.sessions_dir)
    assert again["status"] == "completed"  # a finished session is returned as is


def test_llm_cannot_change_code_owned_fields(tmp_path) -> None:
    with pytest.raises(ValueError):
        PlanProposal(changes={"base_model": "other"}, reason="switch the model please")
    with pytest.raises(ValueError):
        build_train_config(make_settings(tmp_path), {"load_in_4bit": False}, "out")
    cfg = build_train_config(make_settings(tmp_path), {"lora_r": 32}, "out")
    assert cfg.fp16 and not cfg.bf16 and cfg.load_in_4bit and cfg.lora_r == 32  # t4 -> fp16
    assert build_train_config(make_settings(tmp_path, gpu="l4"), {}, "out").bf16


def test_simulated_quality_prefers_sensible_configs(tmp_path) -> None:
    good = build_train_config(
        make_settings(tmp_path), {"learning_rate": 3e-4, "num_epochs": 5}, "o"
    )
    bad = build_train_config(make_settings(tmp_path), {"learning_rate": 1e-2, "num_epochs": 1}, "o")
    assert 0.05 <= simulated_quality(bad) < simulated_quality(good) <= 0.98


def test_cli_agent_dry_run(tmp_path) -> None:
    result = CliRunner().invoke(
        app,
        ["agent", "--objective", "tool calling", "--dry-run", "--yes", "--max-runs", "2",
         "--runs-dir", str(tmp_path / "runs"), "--reports-dir", str(tmp_path / "reports"),
         "--sessions-dir", str(tmp_path / "sessions")],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    (report,) = (tmp_path / "reports").glob("s-*.md")
    assert "DRY RUN" in report.read_text()


def test_cli_agent_input_errors(tmp_path) -> None:
    runner = CliRunner()
    assert runner.invoke(app, ["agent", "--dry-run"]).exit_code == 2  # no objective
    missing = runner.invoke(app, ["agent", "--objective", "x", "--dataset", str(tmp_path)])
    assert missing.exit_code == 2
    unknown = runner.invoke(app, ["agent", "--resume", "s-nope", "--sessions-dir", str(tmp_path)])
    assert unknown.exit_code == 2
