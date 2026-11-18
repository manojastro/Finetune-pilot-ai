"""Config loading and validation."""

import pytest
import yaml
from pydantic import ValidationError

from finetunepilot.config import RunConfig, TrainConfig, diff_configs, load_run_config
from tests.conftest import CONFIG_PATH


def test_shipped_config_loads_with_t4_settings() -> None:
    cfg = load_run_config(CONFIG_PATH)
    t = cfg.train
    assert cfg.name == "qwen05b_toolcall"
    assert t.base_model == "Qwen/Qwen2.5-0.5B-Instruct"
    assert t.load_in_4bit is True
    assert t.fp16 is True and t.bf16 is False  # T4 has no bf16
    assert t.effective_batch_size == 8


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("lora_r", 0),
        ("lora_r", -4),
        ("learning_rate", 1e-7),
        ("learning_rate", 0.05),
        ("per_device_batch_size", 0),
        ("grad_accum_steps", 0),
        ("lora_dropout", 1.0),
        ("max_seq_length", 8),
        ("max_steps", 0),
        ("num_epochs", 0),
        ("target_modules", []),
    ],
)
def test_out_of_range_values_are_rejected(train_config_dict, field, value) -> None:
    with pytest.raises(ValidationError):
        TrainConfig(**train_config_dict, **{field: value})


def test_bf16_and_fp16_together_are_rejected(train_config_dict) -> None:
    with pytest.raises(ValidationError, match="bf16 and fp16"):
        TrainConfig(**train_config_dict, bf16=True, fp16=True)


def test_unknown_field_is_rejected(train_config_dict) -> None:
    with pytest.raises(ValidationError):
        TrainConfig(**train_config_dict, lora_rank=8)  # typo of lora_r


def test_load_from_yaml_round_trip(tmp_path, train_config_dict) -> None:
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump({"name": "t1", "train": train_config_dict}))
    cfg = load_run_config(path)
    assert isinstance(cfg, RunConfig)
    assert cfg.train.lora_r == 16  # default
    assert RunConfig.model_validate(cfg.model_dump()) == cfg


def test_yaml_that_is_not_a_mapping_is_rejected(tmp_path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("- just\n- a list\n")
    with pytest.raises(ValueError, match="mapping"):
        load_run_config(path)


def test_for_dry_run_makes_a_cpu_safe_copy(train_config_dict) -> None:
    cfg = TrainConfig(
        **train_config_dict, fp16=True, load_in_4bit=True, gradient_checkpointing=True
    )
    dry = cfg.for_dry_run()
    assert (dry.load_in_4bit, dry.fp16, dry.bf16, dry.gradient_checkpointing) == (
        False, False, False, False,
    )  # fmt: skip
    assert dry.max_steps == 2
    assert cfg.load_in_4bit is True  # original unchanged


def test_diff_configs_lists_only_changed_fields(train_config_dict) -> None:
    a = TrainConfig(**train_config_dict)
    b = a.model_copy(update={"per_device_batch_size": 2, "grad_accum_steps": 2})
    assert diff_configs(a, b) == {
        "per_device_batch_size": {"from": 4, "to": 2},
        "grad_accum_steps": {"from": 1, "to": 2},
    }
    assert diff_configs(a, a) == {}
