"""Parsing and scoring of model outputs (pure functions, no model)."""

import pytest

from finetunepilot.tools.scoring import (
    RowScore,
    is_exact_format,
    parse_tool_call,
    score_prediction,
    summarize,
    values_equal,
)

WEATHER = ("get_weather", {"city": "Madrid", "unit": "celsius"})
FX = ("convert_currency", {"amount": 5000, "from_currency": "INR", "to_currency": "GBP"})
EXACT = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Madrid", "unit": "celsius"}}\n</tool_call>'


def score(text, target=WEATHER) -> RowScore:
    return score_prediction(text, *target)


def flags(s: RowScore) -> tuple[bool, bool, bool, bool, bool]:
    return (s.json_valid, s.tool_name_correct, s.args_key_match, s.args_exact_match, s.format_ok)


# (case id, output text, target, expected flags: json, name, keys, exact, format)
CASES = [
    ("exact_trained_format", EXACT, WEATHER, (1, 1, 1, 1, 1)),
    ("bare_json_no_tags", '{"name": "get_weather", "arguments": {"city": "Madrid", "unit": "celsius"}}',
     WEATHER, (1, 1, 1, 1, 0)),
    ("json_code_fence", '```json\n{"name": "get_weather", "arguments": {"city": "Madrid", "unit": "celsius"}}\n```',
     WEATHER, (1, 1, 1, 1, 0)),
    ("plain_code_fence", '```\n{"name": "get_weather", "arguments": {"city": "Madrid", "unit": "celsius"}}\n```',
     WEATHER, (1, 1, 1, 1, 0)),
    ("extra_text_around_tags", "Sure! Here you go:\n" + EXACT + "\nLet me know if you need more.",
     WEATHER, (1, 1, 1, 1, 0)),
    ("prose_then_bare_json", 'I will call {"name": "get_weather", "arguments": {"city": "Madrid", "unit": "celsius"}} now.',
     WEATHER, (1, 1, 1, 1, 0)),
    ("missing_closing_tag", '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Madrid", "unit": "celsius"}}',
     WEATHER, (1, 1, 1, 1, 0)),
    ("truncated_json", '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Mad', WEATHER, (0, 0, 0, 0, 0)),
    ("empty_output", "", WEATHER, (0, 0, 0, 0, 0)),
    ("whitespace_only", "   \n ", WEATHER, (0, 0, 0, 0, 0)),
    ("no_json_at_all", "The weather in Madrid is sunny.", WEATHER, (0, 0, 0, 0, 0)),
    ("python_dict_single_quotes", "<tool_call>\n{'name': 'get_weather', 'arguments': {'city': 'Madrid'}}\n</tool_call>",
     WEATHER, (0, 0, 0, 0, 0)),
    ("nested_object_inside_block_not_accepted", '<tool_call>\nCall: {"name": "get_weather", "arguments": {"city": "Madrid", "unit": "celsius"}}\n</tool_call>',
     WEATHER, (0, 0, 0, 0, 0)),
    ("json_array_not_object", '<tool_call>\n["get_weather", {"city": "Madrid"}]\n</tool_call>', WEATHER, (0, 0, 0, 0, 0)),
    ("wrong_tool", '<tool_call>\n{"name": "search_files", "arguments": {"city": "Madrid", "unit": "celsius"}}\n</tool_call>',
     WEATHER, (1, 0, 0, 0, 1)),
    ("missing_name", '<tool_call>\n{"arguments": {"city": "Madrid", "unit": "celsius"}}\n</tool_call>',
     WEATHER, (1, 0, 0, 0, 1)),
    ("extra_key", '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Madrid", "unit": "celsius", "days": 3}}\n</tool_call>',
     WEATHER, (1, 1, 0, 0, 1)),
    ("missing_optional_key", '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Madrid"}}\n</tool_call>',
     WEATHER, (1, 1, 0, 0, 1)),
    ("right_keys_wrong_value", '<tool_call>\n{"name": "get_weather", "arguments": {"city": "madrid", "unit": "celsius"}}\n</tool_call>',
     WEATHER, (1, 1, 1, 0, 1)),
    ("key_order_ignored", '<tool_call>\n{"arguments": {"unit": "celsius", "city": "Madrid"}, "name": "get_weather"}\n</tool_call>',
     WEATHER, (1, 1, 1, 1, 1)),
    ("int_equals_float", '<tool_call>\n{"name": "convert_currency", "arguments": {"amount": 5000.0, "from_currency": "INR", "to_currency": "GBP"}}\n</tool_call>',
     FX, (1, 1, 1, 1, 1)),
    ("string_number_is_wrong", '<tool_call>\n{"name": "convert_currency", "arguments": {"amount": "5000", "from_currency": "INR", "to_currency": "GBP"}}\n</tool_call>',
     FX, (1, 1, 1, 0, 1)),
    ("arguments_as_json_string", '<tool_call>\n{"name": "get_weather", "arguments": "{\\"city\\": \\"Madrid\\", \\"unit\\": \\"celsius\\"}"}\n</tool_call>',
     WEATHER, (1, 1, 1, 1, 1)),
    ("arguments_not_an_object", '<tool_call>\n{"name": "get_weather", "arguments": "Madrid"}\n</tool_call>',
     WEATHER, (1, 1, 0, 0, 1)),
    ("two_calls_first_is_scored", EXACT + "\n" + EXACT.replace("Madrid", "Seoul"), WEATHER, (1, 1, 1, 1, 0)),
    ("brace_inside_string_value", '<tool_call>\n{"name": "send_email", "arguments": {"to": "a@b.c", "subject": "x}", "body": "{y"}}\n</tool_call>',
     ("send_email", {"to": "a@b.c", "subject": "x}", "body": "{y"}), (1, 1, 1, 1, 1)),
]  # fmt: skip


@pytest.mark.parametrize(
    ("text", "target", "expected"), [c[1:] for c in CASES], ids=[c[0] for c in CASES]
)
def test_score_cases(text, target, expected) -> None:
    assert flags(score(text, target)) == tuple(bool(x) for x in expected)


def test_parse_records_number_of_calls_and_errors() -> None:
    assert parse_tool_call(EXACT + EXACT).num_calls == 2
    assert parse_tool_call("").error == "empty output"
    assert parse_tool_call("nothing here").error == "no JSON object found"
    assert parse_tool_call("<tool_call>[1]</tool_call>").error == "payload is not a JSON object"
    assert parse_tool_call('<tool_call>{"name": </tool_call>').error.startswith("invalid JSON")
    parsed = parse_tool_call(EXACT)
    assert (parsed.name, parsed.arguments) == WEATHER


@pytest.mark.parametrize(
    ("a", "b", "equal"),
    [
        (5, 5.0, True),
        (5, "5", False),
        (True, 1, False),
        (1, True, False),
        (False, False, True),
        (None, None, True),
        ("a", "a", True),
        ([1, 2], [1, 2.0], True),
        ([1, 2], [2, 1], False),
        ({"a": {"b": 1}}, {"a": {"b": 1.0}}, True),
        ({"a": 1}, {"a": 1, "b": 2}, False),
    ],
)
def test_values_equal(a, b, equal) -> None:
    assert values_equal(a, b) is equal


def test_is_exact_format() -> None:
    assert is_exact_format(EXACT)
    assert is_exact_format("  " + EXACT + "\n")  # surrounding whitespace is fine
    assert not is_exact_format(EXACT.replace("\n", " "))
    assert not is_exact_format("<tool_call>\n{bad json}\n</tool_call>")


def test_summarize_rates_and_per_tool() -> None:
    rows = [score(EXACT), score(""), score(EXACT.replace("Madrid", "Paris")), score(EXACT)]
    s = summarize(rows, ["get_weather", "get_weather", "get_weather", "calculate"])
    assert s["n"] == 4
    assert s["json_valid"] == 0.75
    assert s["tool_name_accuracy"] == 0.75
    assert s["args_key_match"] == 0.75
    assert s["args_exact_match"] == 0.5
    assert s["format_ok"] == 0.75
    assert s["per_tool"]["get_weather"] == {
        "n": 3,
        "tool_name_accuracy": 0.6667,
        "args_exact_match": 0.3333,
    }
    assert s["per_tool"]["calculate"]["n"] == 1


def test_summarize_empty_and_mismatched_lengths() -> None:
    assert summarize([])["n"] == 0
    with pytest.raises(ValueError):
        summarize([score(EXACT)], ["a", "b"])
