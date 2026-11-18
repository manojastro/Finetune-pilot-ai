"""Error classification."""

import pytest

from finetunepilot.tools.log_parser import classify_error, is_retryable

TORCH_OOM = (
    "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.32 GiB. GPU 0 has a total "
    "capacity of 14.74 GiB of which 1.12 GiB is free."
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (TORCH_OOM, "CUDA_OOM"),
        (
            "RuntimeError: CUDA error: CUBLAS_STATUS_ALLOC_FAILED when calling cublasCreate()",
            "CUDA_OOM",
        ),
        (
            "OutOfMemoryError: CUDA out of memory (simulated in dry run): estimated 41 GiB",
            "CUDA_OOM",
        ),
        ("NanLossError: NaN loss detected at step 12: loss=nan", "NAN_LOSS"),
        ("ValueError: data/sample/train.jsonl:7: invalid row: 1 validation error", "DATASET_ERROR"),
        (
            "ValueError: 3 of 60 rows are longer than max_seq_length=64 (longest: 762 tokens).",
            "DATASET_ERROR",
        ),
        (
            "FileNotFoundError: [Errno 2] No such file or directory: 'data/missing.jsonl'",
            "DATASET_ERROR",
        ),
        ("jinja2.exceptions.TemplateError: Conversation roles must alternate", "TOKENIZER_ERROR"),
        (
            "ValueError: Cannot use chat template functions; tokenizer.chat_template is not set",
            "TOKENIZER_ERROR",
        ),
        ("ConnectionError: Max retries exceeded with url: /api/models", "UNKNOWN"),
    ],  # fmt: skip
)
def test_classifies_real_error_messages(text, expected) -> None:
    info = classify_error(text)
    assert info.error_class == expected
    assert info.evidence and info.evidence in text


def test_evidence_is_the_matching_line_in_a_log() -> None:
    log = (
        "Loading model...\n{'loss': 1.2, 'step': 3}\nTraceback (most recent call last):\n"
        + TORCH_OOM
    )
    info = classify_error(log)
    assert info.error_class == "CUDA_OOM" and info.evidence == TORCH_OOM


def test_traceback_code_lines_do_not_cause_false_matches() -> None:
    log = (
        "Traceback (most recent call last):\n"
        '  File "/x/train.py", line 188, in train\n'
        "    dataset = build_sft_dataset(cfg.dataset_path, tokenizer)\n"
        "ConnectionError: boom\n"
    )
    info = classify_error(log)
    assert info.error_class == "UNKNOWN" and info.evidence == "ConnectionError: boom"


def test_oom_wins_over_other_classes() -> None:
    log = "ValueError: tokenizer warning about chat_template\n" + TORCH_OOM
    assert classify_error(log).error_class == "CUDA_OOM"


def test_exception_objects_and_chains() -> None:
    import torch

    assert classify_error(torch.OutOfMemoryError("CUDA out of memory.")).error_class == "CUDA_OOM"
    try:
        try:
            raise ValueError("data/sample/train.jsonl:3: invalid row: bad")
        except ValueError as e:
            raise RuntimeError("worker failed") from e
    except RuntimeError as outer:
        info = classify_error(outer)
    assert info.error_class == "DATASET_ERROR" and "invalid row" in info.evidence


def test_empty_inputs() -> None:
    assert classify_error(None).error_class == "UNKNOWN"
    assert classify_error("").model_dump() == {"error_class": "UNKNOWN", "evidence": None}


def test_only_oom_and_nan_are_retryable() -> None:
    assert is_retryable("CUDA_OOM") and is_retryable("NAN_LOSS")
    for name in ("DATASET_ERROR", "TOKENIZER_ERROR", "UNKNOWN"):
        assert not is_retryable(name)
