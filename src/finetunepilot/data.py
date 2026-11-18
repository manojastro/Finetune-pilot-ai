"""Load the tool-calling dataset and turn rows into prompt/completion training text.

Each JSONL row looks like:
    {"id": "train-000",
     "messages": [{"role": "user", "content": "..."},
                  {"role": "assistant", "tool_calls": [{"type": "function",
                       "function": {"name": "get_weather", "arguments": {"city": "Paris"}}}]}],
     "tools": [<OpenAI-style function schemas>]}

The chat template is applied here, at load time, so the files on disk stay model-agnostic.
For Qwen2.5 the completion is:
    <tool_call>\\n{"name": ..., "arguments": {...}}\\n</tool_call><|im_end|>

Rows are read with the plain `json` module and rendered to strings BEFORE building a
`datasets.Dataset`. If nested dicts went into Arrow directly, Arrow would merge the different
tools' argument dicts into one schema and fill missing keys with null, which would leak
`"unit": null`-style junk into the training text.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class FunctionCall(BaseModel):
    name: str = Field(min_length=1)
    arguments: dict[str, Any]


class ToolCall(BaseModel):
    type: Literal["function"] = "function"
    function: FunctionCall


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant"]
    content: str | None = None
    tool_calls: list[ToolCall] | None = None


class ToolCallRow(BaseModel):
    """One dataset row: the conversation so far, ending with the correct assistant tool call."""

    id: str
    messages: list[Message] = Field(min_length=2)
    tools: list[dict[str, Any]] = Field(min_length=1)

    @property
    def request(self) -> str:
        """The (last) user request text."""
        return next(m.content or "" for m in reversed(self.messages) if m.role == "user")

    @property
    def target(self) -> FunctionCall:
        """The correct tool call: the first tool call of the final assistant message."""
        last = self.messages[-1]
        if last.role != "assistant" or not last.tool_calls:
            raise ValueError(f"Row {self.id}: last message must be an assistant tool call")
        return last.tool_calls[0].function

    @property
    def tool_names(self) -> list[str]:
        return [t["function"]["name"] for t in self.tools]


def load_rows(path: str | Path) -> list[ToolCallRow]:
    """Read and validate a JSONL file. Raises ValueError with the line number on bad rows."""
    path = Path(path)
    rows: list[ToolCallRow] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                row = ToolCallRow.model_validate(json.loads(line))
                _ = row.target  # make sure the last message is a tool call
            except (json.JSONDecodeError, ValueError) as e:
                raise ValueError(f"{path}:{line_no}: invalid row: {e}") from e
            rows.append(row)
    if not rows:
        raise ValueError(f"{path} has no rows")
    return rows


def _messages_for_template(row: ToolCallRow) -> list[dict[str, Any]]:
    return [m.model_dump(exclude_none=True) for m in row.messages]


def build_prompt(row: ToolCallRow, tokenizer: Any) -> str:
    """The model input: system + tools + user turn, ending with the assistant generation prompt."""
    messages = _messages_for_template(row)[:-1]
    return tokenizer.apply_chat_template(
        messages, tools=row.tools, tokenize=False, add_generation_prompt=True
    )


def format_example(row: ToolCallRow, tokenizer: Any) -> dict[str, str]:
    """Split a row into {"prompt", "completion"} strings using the tokenizer's chat template.

    The completion is everything the template adds for the assistant turn, and it ends with the
    EOS token (so TRL does not add a second one). Loss is computed on the completion only.
    """
    full = tokenizer.apply_chat_template(
        _messages_for_template(row), tools=row.tools, tokenize=False
    )
    prompt = build_prompt(row, tokenizer)
    if not full.startswith(prompt):
        raise ValueError(
            f"Row {row.id}: the chat template renders the prompt differently when the answer is "
            "present, so prompt/completion cannot be split cleanly."
        )
    completion = full[len(prompt) :].rstrip("\n")
    eos = tokenizer.eos_token
    if not completion.endswith(eos):
        raise ValueError(f"Row {row.id}: completion does not end with EOS {eos!r}: {completion!r}")
    return {"prompt": prompt, "completion": completion}


def build_sft_dataset(path: str | Path, tokenizer: Any):
    """Load a JSONL file into a `datasets.Dataset` with string columns id/prompt/completion."""
    from datasets import Dataset

    rows = load_rows(path)
    records = [{"id": r.id, **format_example(r, tokenizer)} for r in rows]
    return Dataset.from_list(records)


def token_length_stats(dataset: Any, tokenizer: Any, max_seq_length: int) -> dict[str, int]:
    """Token lengths of prompt+completion. `truncated` counts rows longer than max_seq_length,
    which would lose (part of) the tool call during training."""
    lengths = [
        len(tokenizer(p + c, add_special_tokens=False)["input_ids"])
        for p, c in zip(dataset["prompt"], dataset["completion"], strict=True)
    ]
    return {
        "rows": len(lengths),
        "max_tokens": max(lengths),
        "mean_tokens": round(sum(lengths) / len(lengths)),
        "truncated": sum(n > max_seq_length for n in lengths),
    }
