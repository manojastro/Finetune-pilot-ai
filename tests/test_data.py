"""Sample dataset integrity, loading and chat-template formatting."""

import json

import pytest

from finetunepilot.data import (
    build_prompt,
    build_sft_dataset,
    format_example,
    load_rows,
    token_length_stats,
)
from tests.conftest import TEST_PATH, TRAIN_PATH

JSON_TYPES = {"string": str, "integer": int, "number": (int, float)}


def test_sample_split_sizes() -> None:
    assert len(load_rows(TRAIN_PATH)) == 60
    assert len(load_rows(TEST_PATH)) == 20


def test_no_train_test_overlap() -> None:
    train = {r.request.strip().lower() for r in load_rows(TRAIN_PATH)}
    test = {r.request.strip().lower() for r in load_rows(TEST_PATH)}
    assert not train & test


def test_ids_are_unique_and_every_tool_is_covered() -> None:
    for path in (TRAIN_PATH, TEST_PATH):
        rows = load_rows(path)
        assert len({r.id for r in rows}) == len(rows)
        assert len({r.target.name for r in rows}) == 6


@pytest.mark.parametrize("path", [TRAIN_PATH, TEST_PATH], ids=["train", "test"])
def test_targets_match_the_tool_schemas(path) -> None:
    """Every target names an available tool, has all required args, no unknown args,
    the right JSON types, and valid enum values."""
    for row in load_rows(path):
        tools = {t["function"]["name"]: t["function"]["parameters"] for t in row.tools}
        assert row.target.name in tools, row.id
        schema = tools[row.target.name]
        args = row.target.arguments
        assert set(schema["required"]) <= set(args), row.id
        assert set(args) <= set(schema["properties"]), row.id
        for key, value in args.items():
            prop = schema["properties"][key]
            assert isinstance(value, JSON_TYPES[prop["type"]]), (row.id, key)
            assert not isinstance(value, bool), (row.id, key)
            if "enum" in prop:
                assert value in prop["enum"], (row.id, key)


def test_bad_row_reports_line_number(tmp_path) -> None:
    good = TRAIN_PATH.read_text().splitlines()[0]
    bad = json.loads(good)
    bad["messages"] = bad["messages"][:1]  # no assistant tool call
    path = tmp_path / "bad.jsonl"
    path.write_text(good + "\n" + json.dumps(bad) + "\n")
    with pytest.raises(ValueError, match=r"bad\.jsonl:2"):
        load_rows(path)


def test_invalid_json_line_is_rejected(tmp_path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text("{not json\n")
    with pytest.raises(ValueError, match=r":1: invalid row"):
        load_rows(path)


def test_empty_file_is_rejected(tmp_path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("\n")
    with pytest.raises(ValueError, match="no rows"):
        load_rows(path)


def test_format_example_uses_qwen_tool_call_format(tiny_tokenizer) -> None:
    row = load_rows(TRAIN_PATH)[0]
    ex = format_example(row, tiny_tokenizer)
    # prompt: system with tool schemas, the user request, then the assistant generation prompt
    assert "<tools>" in ex["prompt"]
    assert row.request in ex["prompt"]
    assert ex["prompt"].endswith("<|im_start|>assistant\n")
    assert ex["prompt"] == build_prompt(row, tiny_tokenizer)
    # completion: exactly one tool call, ending with a single EOS
    comp = ex["completion"]
    assert comp.startswith("<tool_call>\n") and comp.endswith("</tool_call><|im_end|>")
    assert comp.count("<|im_end|>") == 1
    payload = json.loads(
        comp.removeprefix("<tool_call>\n").removesuffix("\n</tool_call><|im_end|>")
    )
    assert payload == {"name": row.target.name, "arguments": row.target.arguments}


def test_formatting_does_not_leak_null_arguments(tiny_tokenizer) -> None:
    """Different tools have different argument keys; none may show up as null in the text."""
    ds = build_sft_dataset(TRAIN_PATH, tiny_tokenizer)
    assert ds.column_names == ["id", "prompt", "completion"]
    assert not any("null" in c for c in ds["completion"])
    rows = {r.id: r for r in load_rows(TRAIN_PATH)}
    for row_id, comp in zip(ds["id"], ds["completion"], strict=True):
        inner = json.loads(
            comp.removeprefix("<tool_call>\n").removesuffix("\n</tool_call><|im_end|>")
        )
        assert inner["arguments"] == rows[row_id].target.arguments


def test_token_length_stats_counts_truncated_rows(tiny_tokenizer) -> None:
    ds = build_sft_dataset(TEST_PATH, tiny_tokenizer)
    ok = token_length_stats(ds, tiny_tokenizer, max_seq_length=100_000)
    assert ok["rows"] == 20 and ok["truncated"] == 0
    assert ok["max_tokens"] >= ok["mean_tokens"] > 0
    assert token_length_stats(ds, tiny_tokenizer, max_seq_length=10)["truncated"] == 20
