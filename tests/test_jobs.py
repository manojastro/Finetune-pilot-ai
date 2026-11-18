"""Shared train/eval job helpers."""

from finetunepilot.config import RunConfig, load_run_config
from finetunepilot.eval import EvalSettings
from finetunepilot.jobs import run_eval_job, run_training_job, train_metrics
from finetunepilot.runlog import read_run
from finetunepilot.runner import AttemptOutcome
from tests.conftest import CONFIG_PATH, TEST_PATH


def test_train_metrics() -> None:
    m = train_metrics({"loss_history": [{"step": 1, "loss": 2.0}], "final_loss": 0.5, "steps": 4})
    assert m["first_loss"] == 2.0 and m["final_loss"] == 0.5 and m["peak_gpu_memory_mb"] is None
    assert train_metrics({})["first_loss"] is None


def test_training_job_writes_a_complete_run_log(tmp_path) -> None:
    run_cfg: RunConfig = load_run_config(CONFIG_PATH)
    ok = AttemptOutcome(
        status="success", result={"output_dir": "adapter", "final_loss": 0.4, "steps": 3}
    )
    log, path = run_training_job(
        run_cfg, dry_run=True, gpu_memory_gb=14.7, runs_dir=tmp_path,
        attempt_fn=lambda cfg, out, t: ok, data_stats={"max_tokens": 762},
    )  # fmt: skip
    assert read_run(path) == log
    assert (
        log.status == "success"
        and log.metrics["final_loss"] == 0.4
        and log.metrics["attempts"] == 1
    )
    assert log.precheck["status"] == "fits" and log.artifacts["adapter_dir"] == "adapter"
    assert log.config["final_train"]["lora_r"] == 16 and log.git_commit is not None


def test_failed_eval_job_is_logged_not_raised(tmp_path) -> None:
    def boom(settings):
        raise RuntimeError("GPU fell off")

    settings = EvalSettings(base_model="x", test_path=str(TEST_PATH), dry_run=True)
    log, path = run_eval_job(settings, name="e", runs_dir=tmp_path, evaluate_fn=boom)
    assert log.status == "failed" and "GPU fell off" in log.errors[0]
    assert read_run(path).status == "failed"
