"""LLM clients and JSON validation (no network)."""

import json
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from finetunepilot.agent.llm import (
    AnthropicClient,
    FakeLLM,
    InvalidLLMOutput,
    LLMError,
    LLMSettings,
    ask_json,
    extract_json_object,
    load_llm_settings,
)
from tests.conftest import REPO_ROOT


class Answer(BaseModel):
    value: int


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"value": 1}', {"value": 1}),
        ('Sure!\n```json\n{"value": 2}\n```\nDone.', {"value": 2}),
        ('Here you go: {"value": 3} -- hope that helps', {"value": 3}),
        ('{"a": {"b": 1}}', {"a": {"b": 1}}),
    ],
)
def test_extract_json_object(text, expected) -> None:
    assert extract_json_object(text) == expected


@pytest.mark.parametrize("text", ["", "no json here", "[1, 2]", "{broken"])
def test_extract_json_object_rejects(text) -> None:
    with pytest.raises(ValueError):
        extract_json_object(text)


def test_ask_json_valid_first_time() -> None:
    llm = FakeLLM(['{"value": 5}'])
    parsed, derived = ask_json(llm, "sys", "prompt", Answer, check=lambda a: a.value * 2)
    assert parsed.value == 5 and derived == 10 and len(llm.calls) == 1


def test_ask_json_retries_once_with_the_error() -> None:
    log = []
    llm = FakeLLM(["not json", '{"value": 7}'])
    parsed, _ = ask_json(llm, "sys", "prompt", Answer, log=log.append)
    assert parsed.value == 7
    assert "rejected" in llm.calls[1]["prompt"] and "no JSON object" in llm.calls[1]["prompt"]
    assert [e["valid"] for e in log] == [False, True]


def test_ask_json_fails_clearly_after_two_bad_answers() -> None:
    llm = FakeLLM(['{"value": "x"}', '{"value": "still bad"}', '{"value": 1}'])
    with pytest.raises(InvalidLLMOutput, match="invalid twice"):
        ask_json(llm, "sys", "prompt", Answer)
    assert len(llm.calls) == 2  # never a third try


def test_ask_json_check_can_reject() -> None:
    def must_be_even(a: Answer):
        if a.value % 2:
            raise ValueError("value must be even")
        return a.value

    llm = FakeLLM(['{"value": 3}', '{"value": 4}'])
    _, derived = ask_json(llm, "sys", "p", Answer, check=must_be_even)
    assert derived == 4 and "must be even" in llm.calls[1]["prompt"]


def test_fake_llm_modes() -> None:
    assert FakeLLM(lambda s, p: p.upper()).complete("s", "hi") == "HI"
    with pytest.raises(LLMError):
        FakeLLM([]).complete("s", "p")


def test_agent_config_file_names_the_model() -> None:
    settings = load_llm_settings(REPO_ROOT / "configs" / "agent.yaml")
    assert settings.model and settings.fallbacks == "default"


class FakeMessages:
    def __init__(self, response):
        self.response, self.kwargs = response, None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return self.response


def _client(response):
    messages, beta = FakeMessages(response), FakeMessages(response)
    return SimpleNamespace(messages=messages, beta=SimpleNamespace(messages=beta)), messages, beta


def _response(text="", stop="end_turn"):
    content = [SimpleNamespace(type="thinking", thinking="")] + (
        [SimpleNamespace(type="text", text=text)] if text else []
    )
    return SimpleNamespace(content=content, stop_reason=stop)


def test_anthropic_client_request_shape() -> None:
    fake, plain, beta = _client(_response('{"value": 1}'))
    settings = LLMSettings(model="some-model", effort="low", max_tokens=1000)
    assert AnthropicClient(settings, client=fake).complete("SYS", "PROMPT") == '{"value": 1}'
    sent = beta.kwargs
    assert sent["model"] == "some-model" and sent["system"] == "SYS"
    assert sent["messages"] == [{"role": "user", "content": "PROMPT"}]
    assert sent["output_config"] == {"effort": "low"} and sent["max_tokens"] == 1000
    assert sent["fallbacks"] == "default" and sent["betas"] == [AnthropicClient.FALLBACK_BETA]
    assert plain.kwargs is None

    fake, plain, beta = _client(_response("ok"))
    AnthropicClient(LLMSettings(model="m", fallbacks=None), client=fake).complete("s", "p")
    assert plain.kwargs is not None and "fallbacks" not in plain.kwargs and beta.kwargs is None


def test_anthropic_client_refusal_and_empty_answers_raise() -> None:
    fake, _, _ = _client(_response(stop="refusal"))
    with pytest.raises(LLMError, match="declined"):
        AnthropicClient(LLMSettings(model="m"), client=fake).complete("s", "p")
    fake, _, _ = _client(_response(stop="max_tokens"))
    with pytest.raises(LLMError, match="Empty answer"):
        AnthropicClient(LLMSettings(model="m"), client=fake).complete("s", "p")


def test_llm_settings_validation() -> None:
    with pytest.raises(ValueError):
        LLMSettings(model="m", effort="extreme")
    assert json.loads(LLMSettings(model="m").model_dump_json())["effort"] == "medium"
