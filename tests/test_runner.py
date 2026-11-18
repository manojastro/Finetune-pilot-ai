"""Recovery loop with scripted (fake) attempts: no GPU, no subprocess."""

import pytest
from pydantic import ValidationError

from finetunepilot.config import load_run_config
from finetunepilot.runner import (
    FORCE_OOM_BATCH,
    AttemptOutcome,
    FixPlanner,
    RecoveryPolicy,
    precheck,
    run_with_recovery,
)
from tests.conftest import CONFIG_PATH

OOM = AttemptOutcome(
    status="failed", error="OutOfMemoryError: CUDA out of memory. Tried to allocate 2.0 GiB"
)
NAN = AttemptOutcome(status="failed", error="NanLossError: NaN loss detected at step 3: loss=nan")
DATA = AttemptOutcome(status="failed", error="ValueError: data/x.jsonl:1: invalid row: bad")
OK = AttemptOutcome(status="success", result={"output_dir": "out", "final_loss": 0.1})
BIG_GPU = 100.0  # the pre-check always passes


class FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


class Script:
    """Returns scripted outcomes in order and records each config it was given."""

    def __init__(self, *outcomes, clock=None, minutes_each=0.0):
        self.outcomes = list(outcomes)
        self.configs, self.timeouts = [], []
        self.clock, self.minutes_each = clock, minutes_each

    def __call__(self, cfg, out_dir, timeout_s):
        self.configs.append(cfg)
        self.timeouts.append(timeout_s)
        if self.clock:
            self.clock.t += self.minutes_each * 60
        return self.outcomes.pop(0)


@pytest.fixture
def cfg():
    return load_run_config(CONFIG_PATH).train  # batch 4, grad accum 2, seq 1024, lr 2e-4


def run(cfg, script, gpu=BIG_GPU, data_stats=None, **policy_kw):
    clock = script.clock or FakeClock()
    return run_with_recovery(
        cfg, RecoveryPolicy(**policy_kw), output_dir="unused", attempt_fn=script,
        gpu_memory_gb=gpu, data_stats=data_stats if data_stats is not None else {"max_tokens": 500},
        clock=clock,
    )  # fmt: skip


def shape(c):
    return (c.per_device_batch_size, c.grad_accum_steps, c.gradient_checkpointing, c.max_seq_length)


def test_oom_fixes_follow_the_ladder_and_keep_effective_batch(cfg) -> None:
    script = Script(OOM, OOM, OOM, OK)
    res = run(cfg, script)
    assert res.status == "success" and res.stop_reason == "succeeded on attempt 4"
    assert [shape(c) for c in script.configs] == [
        (4, 2, False, 1024),
        (2, 4, False, 1024),  # 1. halve batch, double grad accum
        (2, 4, True, 1024),  # 2. gradient checkpointing
        (2, 4, True, 768),  # 3. cut max_seq_length by 25%
    ]
    assert {c.effective_batch_size for c in script.configs} == {8}
    assert [a.status for a in res.attempts] == ["failed", "failed", "failed", "success"]
    assert [a.error_class for a in res.attempts] == ["CUDA_OOM"] * 3 + [None]


def test_after_the_ladder_it_keeps_halving_then_gives_up(cfg) -> None:
    script = Script(*[OOM] * 6)
    res = run(cfg, script, max_attempts=8)
    assert [c.per_device_batch_size for c in script.configs] == [4, 2, 2, 2, 1]
    assert script.configs[-1].grad_accum_steps == 8
    assert res.status == "failed" and res.stop_reason == "no allowed fix left for CUDA_OOM"


def test_seq_cut_never_goes_below_the_longest_row(cfg) -> None:
    script = Script(OOM, OOM, OOM, OK)
    run(cfg, script, data_stats={"max_tokens": 900})
    assert script.configs[3].max_seq_length == 900  # 768 would truncate rows of 900 tokens

    script = Script(OOM, OOM, OOM, OK)
    run(cfg, script, data_stats={"max_tokens": 1024})  # no room to cut: step skipped
    assert [shape(c) for c in script.configs][3] == (1, 8, True, 1024)


def test_longest_row_reported_by_an_attempt_is_used(cfg) -> None:
    first = OOM.model_copy(update={"data_stats": {"max_tokens": 1000}})
    script = Script(first, OOM, OOM, OK)
    run(cfg, script, data_stats={})
    assert script.configs[3].max_seq_length == 1000


def test_nan_loss_halves_learning_rate(cfg) -> None:
    script = Script(NAN, NAN, OK)
    res = run(cfg, script)
    assert [c.learning_rate for c in script.configs] == [2e-4, 1e-4, 5e-5]
    assert res.attempts[1].fix.startswith("after NAN_LOSS: halve learning rate")
    assert res.attempts[0].error_class == "NAN_LOSS"


def test_nan_at_minimum_learning_rate_stops(cfg) -> None:
    res = run(cfg.model_copy(update={"learning_rate": 1.5e-6}), Script(NAN, OK))
    assert res.status == "failed" and res.stop_reason == "no allowed fix left for NAN_LOSS"


def test_non_retryable_error_stops_immediately(cfg) -> None:
    script = Script(DATA, OK)
    res = run(cfg, script)
    assert len(res.attempts) == 1 and res.attempts[0].error_class == "DATASET_ERROR"
    assert res.stop_reason == "DATASET_ERROR is not retryable"
    assert "invalid row" in res.attempts[0].evidence


def test_max_attempts_is_respected(cfg) -> None:
    script = Script(OOM, OOM, OOM)
    res = run(cfg, script, max_attempts=2)
    assert len(script.configs) == 2 and res.stop_reason == "max attempts (2) reached"


def test_time_budget_is_respected(cfg) -> None:
    clock = FakeClock()
    script = Script(OOM, OOM, OK, clock=clock, minutes_each=40)
    res = run(cfg, script, max_minutes=60)
    assert len(script.configs) == 2
    assert script.timeouts == [3600, 1200]  # the second attempt only gets what is left
    assert res.status == "failed" and "used up before attempt 3" in res.stop_reason


def test_timeout_outcome_stops_the_run(cfg) -> None:
    timeout = AttemptOutcome(status="timeout", error="attempt killed after 60s (time budget)")
    res = run(cfg, Script(timeout, OK))
    assert res.attempts[0].status == "timeout" and "during attempt 1" in res.stop_reason


def test_fields_outside_the_policy_are_never_changed(cfg) -> None:
    allowed = ["per_device_batch_size", "grad_accum_steps", "max_seq_length"]
    script = Script(OOM, OOM, OK)
    res = run(cfg, script, allowed_changes=allowed)
    assert not any(c.gradient_checkpointing for c in script.configs)  # that step was skipped
    assert script.configs[2].max_seq_length == 768
    for a in res.attempts:
        assert set(a.config_changes) <= set(allowed)

    res = run(cfg, Script(OOM, OK), allowed_changes=[])
    assert res.stop_reason == "no allowed fix left for CUDA_OOM"


def test_every_attempt_is_logged_with_its_changes(cfg) -> None:
    seen = []
    script = Script(OOM, OK)
    res = run_with_recovery(
        cfg, RecoveryPolicy(), output_dir="unused", attempt_fn=script, gpu_memory_gb=BIG_GPU,
        data_stats={"max_tokens": 500}, clock=FakeClock(),
        on_attempt=lambda attempts: seen.append(len(attempts)),
    )  # fmt: skip
    first, second = res.attempts
    assert first.config_changes == {} and first.fix is None
    assert second.config_changes == {
        "per_device_batch_size": {"from": 4, "to": 2},
        "grad_accum_steps": {"from": 2, "to": 4},
    }
    assert second.fix.startswith("after CUDA_OOM: halve batch")
    assert all(a.duration_s is not None and a.started_at for a in res.attempts)
    assert seen and seen[-1] == 2


def test_precheck_shrinks_a_config_that_would_not_fit(cfg) -> None:
    script = Script(OK)
    res = run(cfg, script, gpu=12.0, data_stats={"max_tokens": 762})
    assert res.precheck["status"] == "adjusted"
    assert (
        res.precheck["estimate_before"]["fits"] is False and res.precheck["estimate_after"]["fits"]
    )
    assert shape(script.configs[0])[:2] == (2, 4)
    assert res.attempts[0].fix.startswith("pre-check: halve batch")


def test_precheck_refuses_to_start_when_nothing_fits(cfg) -> None:
    script = Script(OK)
    res = run(cfg, script, gpu=1.0)
    assert res.status == "failed" and res.stop_reason.startswith("won't fit")
    assert res.attempts == [] and script.configs == []
    assert res.precheck["status"] == "wont_fit"


def test_force_oom_skips_precheck_and_forces_batch(cfg) -> None:
    script = Script(OOM, OK)
    res = run_with_recovery(
        cfg, RecoveryPolicy(), output_dir="unused", attempt_fn=script, gpu_memory_gb=1.0,
        data_stats={"max_tokens": 500}, force_oom=True, clock=FakeClock(),
    )  # fmt: skip
    assert res.precheck["status"] == "skipped"
    assert [c.per_device_batch_size for c in script.configs] == [
        FORCE_OOM_BATCH,
        FORCE_OOM_BATCH // 2,
    ]
    assert res.base_config.per_device_batch_size == FORCE_OOM_BATCH
    assert res.status == "success"


def test_precheck_skipped_for_unknown_architecture(cfg) -> None:
    other = cfg.model_copy(update={"base_model": "someone/unknown-7b"})
    checked, info = precheck(other, RecoveryPolicy(), gpu_memory_gb=1.0)
    assert checked == other and info["status"] == "skipped"


def test_halving_an_odd_batch_rounds_grad_accum_up(cfg) -> None:
    odd = cfg.model_copy(update={"per_device_batch_size": 3, "grad_accum_steps": 2})
    new, note = FixPlanner(RecoveryPolicy().allowed_changes).next_oom_fix(odd)
    assert (new.per_device_batch_size, new.grad_accum_steps) == (1, 6)
    assert "per_device_batch_size 3 -> 1" in note


def test_policy_validation() -> None:
    with pytest.raises(ValidationError):
        RecoveryPolicy(allowed_changes=["not_a_field"])
    with pytest.raises(ValidationError):
        RecoveryPolicy(max_attempts=0)
    with pytest.raises(ValidationError):
        RecoveryPolicy(max_minutes=0)
