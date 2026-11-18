"""Pure functions to parse model output and score tool calls. No model needed.

Parsing is lenient so that we measure *what the model meant*, and `format_ok` separately
records whether it used the exact trained format.

Where the JSON payload is looked for, in order:
  1. inside <tool_call> ... </tool_call> (a missing closing tag is allowed)
  2. inside a ``` / ```json code fence
  3. the first JSON object anywhere in the text
In cases 1 and 2 the block itself must start with a JSON object.

Metrics per row:
  json_valid         a JSON object was found and parsed
  tool_name_correct  its "name" equals the target tool name
  args_key_match     tool name correct AND the set of argument keys equals the target's
  args_exact_match   tool name correct AND the arguments equal the target's
                     (key order ignored; 5 == 5.0; "5" != 5; True != 1; strings exact)
  format_ok          the output is exactly <tool_call>\\n{json}\\n</tool_call>, nothing else
"""

from __future__ import annotations

import contextlib
import json
import re
from typing import Any

from pydantic import BaseModel

_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)(?:</tool_call>|$)", re.DOTALL)
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
_EXACT_FORMAT_RE = re.compile(r"\A<tool_call>\n(\{.*\})\n</tool_call>\Z", re.DOTALL)


class ParsedCall(BaseModel):
    json_valid: bool
    name: str | None = None
    arguments: dict[str, Any] | None = None
    num_calls: int = 0  # how many <tool_call> blocks the output had
    error: str | None = None


class RowScore(BaseModel):
    json_valid: bool
    tool_name_correct: bool
    args_key_match: bool
    args_exact_match: bool
    format_ok: bool
    predicted_name: str | None = None
    predicted_arguments: dict[str, Any] | None = None
    error: str | None = None


METRIC_KEYS = [
    "json_valid",
    "tool_name_accuracy",
    "args_key_match",
    "args_exact_match",
    "format_ok",
]


def _first_json_object(text: str) -> Any:
    """Free text: decode the first JSON value that starts at any '{'. Raises ValueError if none."""
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch == "{":
            try:
                value, _ = decoder.raw_decode(text, i)
                return value
            except json.JSONDecodeError:
                continue
    raise ValueError("no JSON object found")


def _decode_block(block: str) -> Any:
    """Inside <tool_call> or a code fence the payload itself must be a JSON object.
    Text after the object is tolerated (e.g. a missing closing tag), a nested object is not."""
    payload = block.strip()
    if not payload.startswith("{"):
        raise ValueError("payload is not a JSON object")
    try:
        value, _ = json.JSONDecoder().raw_decode(payload)
    except json.JSONDecodeError as e:
        raise ValueError(f"invalid JSON: {e.msg}") from None
    return value


def _candidate(text: str) -> tuple[str | None, int]:
    """The block most likely to hold the call payload (None = no tags/fence: search free text),
    and the number of <tool_call> blocks."""
    blocks = _TOOL_CALL_RE.findall(text)
    if blocks:
        return blocks[0], len(blocks)
    fence = _FENCE_RE.search(text)
    if fence:
        return fence.group(1), 0
    return None, 0


def parse_tool_call(text: str) -> ParsedCall:
    """Extract {"name", "arguments"} from raw model output. Never raises."""
    if not text or not text.strip():
        return ParsedCall(json_valid=False, error="empty output")
    block, num_calls = _candidate(text)
    try:
        obj = _first_json_object(text) if block is None else _decode_block(block)
    except ValueError as e:
        return ParsedCall(json_valid=False, num_calls=num_calls, error=str(e))
    if not isinstance(obj, dict):
        return ParsedCall(json_valid=False, num_calls=num_calls, error="JSON is not an object")

    name = obj.get("name")
    args = obj.get("arguments")
    if isinstance(args, str):  # some models emit arguments as a JSON-encoded string
        with contextlib.suppress(json.JSONDecodeError):
            args = json.loads(args)
    return ParsedCall(
        json_valid=True,
        name=name if isinstance(name, str) else None,
        arguments=args if isinstance(args, dict) else None,
        num_calls=num_calls,
        error=None if isinstance(args, dict) else "arguments missing or not an object",
    )


def values_equal(a: Any, b: Any) -> bool:
    """Equality for JSON values: numbers compare by value (5 == 5.0), but bools are not numbers,
    strings never equal numbers, and dicts ignore key order."""
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if isinstance(a, int | float) and isinstance(b, int | float):
        return float(a) == float(b)
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(values_equal(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(values_equal(x, y) for x, y in zip(a, b, strict=True))
    return type(a) is type(b) and a == b


def is_exact_format(text: str) -> bool:
    match = _EXACT_FORMAT_RE.match(text.strip())
    if not match:
        return False
    try:
        return isinstance(json.loads(match.group(1)), dict)
    except json.JSONDecodeError:
        return False


def score_prediction(text: str, target_name: str, target_arguments: dict[str, Any]) -> RowScore:
    parsed = parse_tool_call(text)
    name_ok = parsed.json_valid and parsed.name == target_name
    args = parsed.arguments
    return RowScore(
        json_valid=parsed.json_valid,
        tool_name_correct=name_ok,
        args_key_match=name_ok and args is not None and set(args) == set(target_arguments),
        args_exact_match=name_ok and args is not None and values_equal(args, target_arguments),
        format_ok=is_exact_format(text),
        predicted_name=parsed.name,
        predicted_arguments=args,
        error=parsed.error,
    )


def summarize(scores: list[RowScore], target_names: list[str] | None = None) -> dict[str, Any]:
    """Average each metric over rows (0..1). With target_names, also break down per tool."""
    n = len(scores)
    if n == 0:
        return {"n": 0, **dict.fromkeys(METRIC_KEYS)}

    def rate(rows: list[RowScore], attr: str) -> float:
        return round(sum(getattr(r, attr) for r in rows) / len(rows), 4)

    summary: dict[str, Any] = {
        "n": n,
        "json_valid": rate(scores, "json_valid"),
        "tool_name_accuracy": rate(scores, "tool_name_correct"),
        "args_key_match": rate(scores, "args_key_match"),
        "args_exact_match": rate(scores, "args_exact_match"),
        "format_ok": rate(scores, "format_ok"),
    }
    if target_names is not None:
        if len(target_names) != n:
            raise ValueError("target_names must have one entry per score")
        per_tool: dict[str, dict[str, Any]] = {}
        for tool in sorted(set(target_names)):
            rows = [s for s, t in zip(scores, target_names, strict=True) if t == tool]
            per_tool[tool] = {
                "n": len(rows),
                "tool_name_accuracy": rate(rows, "tool_name_correct"),
                "args_exact_match": rate(rows, "args_exact_match"),
            }
        summary["per_tool"] = per_tool
    return summary
