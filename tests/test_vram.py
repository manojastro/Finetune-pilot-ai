"""VRAM estimator."""

import pytest

from finetunepilot.config import load_run_config
from finetunepilot.tools.vram import (
    GIB,
    KNOWN_MODEL_SPECS,
    estimate_vram,
    get_model_spec,
    lora_param_count,
    spec_from_hf_config,
)
from tests.conftest import CONFIG_PATH

SPEC = KNOWN_MODEL_SPECS["Qwen/Qwen2.5-0.5B-Instruct"]
ALL = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


@pytest.fixture
def cfg():
    return load_run_config(CONFIG_PATH).train


def test_lora_param_count_for_qwen_05b() -> None:
    # per layer: 16 * (1792 + 1024 + 1024 + 1792 + 5760 + 5760 + 5760) = 366,592; x 24 layers
    assert lora_param_count(SPEC, ALL, r=16) == 8_798_208
    assert lora_param_count(SPEC, ALL, r=8) == 8_798_208 // 2
    assert lora_param_count(SPEC, ["q_proj", "v_proj"], r=16) == 16 * (1792 + 1024) * 24


def test_unknown_target_module_is_rejected() -> None:
    with pytest.raises(ValueError, match="fc1"):
        lora_param_count(SPEC, ["fc1"], r=8)


def test_logits_term_is_batch_seq_vocab_fp32_times_two(cfg) -> None:
    est = estimate_vram(cfg, 14.7, seq_len=762)
    expected = 4 * 762 * 151936 * 4 * 2 / GIB
    assert est.breakdown_gb["logits"] == pytest.approx(expected, abs=1e-3)


def test_breakdown_adds_up_to_total(cfg) -> None:
    est = estimate_vram(cfg, 14.7, seq_len=762)
    assert sum(est.breakdown_gb.values()) == pytest.approx(est.total_gb, abs=0.01)
    assert est.lora_params == 8_798_208


def test_memory_grows_with_batch_and_sequence(cfg) -> None:
    small = estimate_vram(cfg, 14.7, seq_len=256).total_gb
    longer = estimate_vram(cfg, 14.7, seq_len=512).total_gb
    bigger = estimate_vram(
        cfg.model_copy(update={"per_device_batch_size": 8}), 14.7, seq_len=256
    ).total_gb
    assert small < longer and small < bigger


def test_seq_len_defaults_to_and_is_capped_by_max_seq_length(cfg) -> None:
    assert estimate_vram(cfg, 14.7).seq_len == 1024
    assert estimate_vram(cfg, 14.7, seq_len=5000).seq_len == 1024
    assert estimate_vram(cfg, 14.7, seq_len=762).seq_len == 762


def test_gradient_checkpointing_cuts_activations(cfg) -> None:
    off = estimate_vram(cfg, 14.7, seq_len=762).breakdown_gb["activations"]
    on = estimate_vram(cfg.model_copy(update={"gradient_checkpointing": True}), 14.7, seq_len=762)
    assert on.breakdown_gb["activations"] < off / 10
    tokens, h, i, layers = 4 * 762, 896, 4864, 24
    expected = (layers * h + 16.5 * h + 5 * i) * tokens * 2 / GIB
    assert on.breakdown_gb["activations"] == pytest.approx(expected, abs=1e-3)


def test_4bit_weights_are_smaller_and_optimizer_is_8bit(cfg) -> None:
    q = estimate_vram(cfg, 14.7, seq_len=762)
    full = estimate_vram(cfg.model_copy(update={"load_in_4bit": False}), 14.7, seq_len=762)
    assert q.breakdown_gb["weights"] < full.breakdown_gb["weights"]
    # paged 8-bit AdamW: 2 bytes/param vs AdamW: 8 bytes/param (breakdown is rounded to 1 MiB)
    assert q.breakdown_gb["optimizer"] * 4 == pytest.approx(
        full.breakdown_gb["optimizer"], abs=5e-3
    )


def test_fits_uses_safety_margin(cfg) -> None:
    est = estimate_vram(cfg, 14.7, seq_len=762, safety_margin=0.15)
    assert est.limit_gb == pytest.approx(14.7 * 0.85, abs=1e-3)
    assert est.fits == (est.total_gb <= est.limit_gb)
    assert est.headroom_gb == pytest.approx(est.limit_gb - est.total_gb, abs=1e-3)
    tight = estimate_vram(cfg, est.total_gb + 0.01, seq_len=762, safety_margin=0.0)
    assert tight.fits
    assert not estimate_vram(cfg, est.total_gb + 0.01, seq_len=762, safety_margin=0.5).fits
    with pytest.raises(ValueError):
        estimate_vram(cfg, 14.7, safety_margin=1.0)


def test_default_t4_config_fits_at_real_row_length_but_not_at_1024(cfg) -> None:
    assert estimate_vram(cfg, 14.7, seq_len=762).fits
    assert not estimate_vram(cfg, 14.7).fits  # worst case: every row padded to 1024


def test_model_spec_lookup_and_hf_config() -> None:
    assert get_model_spec("Qwen/Qwen2.5-0.5B-Instruct") is SPEC
    with pytest.raises(KeyError, match="spec_from_hf_config"):
        get_model_spec("someone/unknown-7b")
    spec = spec_from_hf_config(
        {"hidden_size": 64, "num_hidden_layers": 2, "intermediate_size": 128, "vocab_size": 1000,
         "num_attention_heads": 4}
    )  # fmt: skip
    assert spec.num_kv_heads == 4 and spec.head_dim == 16 and spec.tie_word_embeddings is False
    assert spec.module_shapes()["k_proj"] == (64, 64)
