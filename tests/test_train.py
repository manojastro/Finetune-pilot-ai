"""Training: real-path settings (built, not run) and a CPU dry run."""

import math

import pytest

from finetunepilot.config import TrainConfig, load_run_config
from finetunepilot.train import build_bnb_config, build_lora_config, build_sft_config, train
from tests.conftest import CONFIG_PATH


@pytest.fixture
def t4_config():
    return load_run_config(CONFIG_PATH).train


def test_qlora_quantization_settings(t4_config) -> None:
    import torch

    bnb = build_bnb_config(t4_config)
    assert bnb.load_in_4bit and bnb.bnb_4bit_quant_type == "nf4"
    assert bnb.bnb_4bit_use_double_quant
    assert bnb.bnb_4bit_compute_dtype == torch.float16
    assert build_bnb_config(t4_config.for_dry_run()) is None


def test_lora_settings(t4_config) -> None:
    lora = build_lora_config(t4_config)
    assert (lora.r, lora.lora_alpha, lora.lora_dropout) == (16, 32, 0.05)
    assert set(lora.target_modules) == {
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    }  # fmt: skip


def test_trainer_settings_for_t4(t4_config, tmp_path) -> None:
    args = build_sft_config(t4_config, tmp_path, use_cpu=False)
    assert args.fp16 and not args.bf16
    assert args.completion_only_loss is True
    assert args.max_length == 1024
    assert args.per_device_train_batch_size == 4 and args.gradient_accumulation_steps == 2
    assert "paged_adamw_8bit" in str(args.optim).lower()
    assert args.report_to in ([], "none", ["none"])


@pytest.mark.slow
def test_loss_is_masked_to_the_tool_call(t4_config, tmp_path, tiny_tokenizer) -> None:
    """Only completion tokens (the tool call + one EOS) have labels; the prompt is ignored."""
    from peft import get_peft_model
    from trl import SFTTrainer

    from finetunepilot.data import build_sft_dataset
    from finetunepilot.dryrun import build_tiny_model

    cfg = t4_config.for_dry_run()
    ds = build_sft_dataset(cfg.dataset_path, tiny_tokenizer).remove_columns("id")
    trainer = SFTTrainer(
        model=get_peft_model(build_tiny_model(tiny_tokenizer), build_lora_config(cfg)),
        args=build_sft_config(cfg, tmp_path, use_cpu=True),
        train_dataset=ds,
        processing_class=tiny_tokenizer,
    )
    batch = trainer.data_collator([trainer.train_dataset[0]])
    labels = batch["labels"][0]
    trained_text = tiny_tokenizer.decode(batch["input_ids"][0][labels != -100])
    assert trained_text == ds[0]["completion"]


@pytest.mark.slow
def test_dry_run_trains_and_saves_adapter(t4_config, tmp_path) -> None:
    result = train(t4_config, output_dir=tmp_path / "adapter", dry_run=True)
    assert result.dry_run and result.steps == 2
    assert result.final_loss is not None and math.isfinite(result.final_loss)
    assert len(result.loss_history) == 2
    assert result.peak_gpu_memory_mb is None  # CPU
    assert result.data_stats["truncated"] == 0
    saved = {p.name for p in (tmp_path / "adapter").iterdir()}
    assert {"adapter_config.json", "adapter_model.safetensors", "tokenizer_config.json"} <= saved


def test_rows_longer_than_max_seq_length_fail_clearly(t4_config, tmp_path, monkeypatch) -> None:
    # for_dry_run() normally raises max_seq_length to 4096; force a tiny limit instead.
    tiny = t4_config.for_dry_run().model_copy(update={"max_seq_length": 64})
    monkeypatch.setattr(TrainConfig, "for_dry_run", lambda self: tiny)
    with pytest.raises(ValueError, match="longer than max_seq_length=64"):
        train(t4_config, output_dir=tmp_path, dry_run=True)


def test_real_training_without_gpu_fails_clearly(t4_config, tmp_path) -> None:
    import torch

    if torch.cuda.is_available():
        pytest.skip("machine has a GPU")
    with pytest.raises(RuntimeError, match="No CUDA GPU"):
        train(t4_config, output_dir=tmp_path, dry_run=False)


def test_nan_loss_callback_stops_training() -> None:
    from types import SimpleNamespace

    from finetunepilot.tools.log_parser import classify_error
    from finetunepilot.train import NanLossError, make_nan_loss_callback

    callback = make_nan_loss_callback()
    state = SimpleNamespace(global_step=7)
    callback.on_log(None, state, None, logs={"loss": 0.5})  # finite: fine
    callback.on_log(None, state, None, logs={"learning_rate": 1e-4})  # no loss: fine
    for bad in (float("nan"), float("inf")):
        with pytest.raises(NanLossError) as exc:
            callback.on_log(None, state, None, logs={"loss": bad})
        assert classify_error(exc.value).error_class == "NAN_LOSS"
