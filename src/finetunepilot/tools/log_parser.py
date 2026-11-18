"""Classify training errors from an exception or log text.

classify_error() returns one of CUDA_OOM, NAN_LOSS, DATASET_ERROR, TOKENIZER_ERROR, UNKNOWN,
plus the line that matched (the evidence). Classes are checked in that order, so an OOM that
happens while tokenizing is still CUDA_OOM. The patterns are plain regexes over real error
messages from torch, bitsandbytes, transformers, datasets and our own code.

Only message lines are searched, never traceback code lines: those are indented in Python
tracebacks and would cause false matches (e.g. any code line that mentions `tokenizer`).
For an exception object, the message chain (the exception, its __cause__ and __context__) is used.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel

ErrorClass = Literal["CUDA_OOM", "NAN_LOSS", "DATASET_ERROR", "TOKENIZER_ERROR", "UNKNOWN"]

PATTERNS: list[tuple[ErrorClass, re.Pattern[str]]] = [
    (
        "CUDA_OOM",
        re.compile(
            r"CUDA out of memory|OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED"
            r"|cudaErrorMemoryAllocation|out of memory on device",
            re.IGNORECASE,
        ),
    ),
    (
        "NAN_LOSS",
        re.compile(r"NanLossError|NaN loss detected|loss (is|became) (nan|inf)", re.IGNORECASE),
    ),
    (
        "DATASET_ERROR",
        re.compile(
            r"invalid row|has no rows|longer than max_seq_length|DatasetGenerationError"
            r"|\.jsonl\b.*(No such file|not found)|(No such file|not found).*\.jsonl\b",
            re.IGNORECASE,
        ),
    ),
    (
        "TOKENIZER_ERROR",
        re.compile(r"chat[_ ]template|TemplateError|TemplateSyntaxError|tokenizer", re.IGNORECASE),
    ),
]

RETRYABLE: set[str] = {"CUDA_OOM", "NAN_LOSS"}


class ErrorInfo(BaseModel):
    error_class: ErrorClass
    evidence: str | None  # the first line that matched, stripped and cut to 300 chars


def _message_lines(source: BaseException | str) -> list[str]:
    if isinstance(source, BaseException):
        lines, seen, exc = [], set(), source
        while exc is not None and id(exc) not in seen:  # walk the exception chain
            seen.add(id(exc))
            lines.extend(f"{type(exc).__name__}: {exc}".splitlines())
            exc = exc.__cause__ or exc.__context__
        return [line.strip() for line in lines if line.strip()]
    # Log text: skip indented lines (traceback frames and source code).
    return [line.strip() for line in source.splitlines() if line.strip() and not line[0].isspace()]


def classify_error(source: BaseException | str | None) -> ErrorInfo:
    """Classify an exception or a block of log text."""
    if source is None:
        return ErrorInfo(error_class="UNKNOWN", evidence=None)
    lines = _message_lines(source)
    for error_class, pattern in PATTERNS:
        for line in lines:
            if pattern.search(line):
                return ErrorInfo(error_class=error_class, evidence=line[:300])
    return ErrorInfo(error_class="UNKNOWN", evidence=lines[-1][:300] if lines else None)


def is_retryable(error_class: str) -> bool:
    """Only OOM and NaN loss have automatic fixes; retrying anything else would fail again."""
    return error_class in RETRYABLE
