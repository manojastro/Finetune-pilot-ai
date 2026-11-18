"""LLM clients for the agent: one interface, a real Anthropic client and fakes for tests.

The agent only ever asks the LLM for small JSON answers (a config proposal or one config
change). `ask_json` validates every answer with pydantic; an invalid answer gets ONE retry with
the error explained, then fails clearly (CLAUDE.md rule 6).
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol, TypeVar

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    """The LLM call itself failed (API error, refusal, empty answer)."""


class InvalidLLMOutput(RuntimeError):
    """The LLM answered twice, and neither answer passed validation."""


class LLMClient(Protocol):
    def complete(self, system: str, prompt: str) -> str: ...


class LLMSettings(BaseModel):
    """configs/agent.yaml -> llm: {...}. The model name lives in config, not in code."""

    model_config = ConfigDict(extra="forbid")

    model: str = Field(min_length=1)
    effort: str = Field(default="medium", pattern="^(low|medium|high|xhigh|max)$")
    max_tokens: int = Field(default=16000, ge=256, le=64000)
    fallbacks: str | None = "default"  # server-side refusal fallback; None turns it off


def load_llm_settings(path: str | Path) -> LLMSettings:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return LLMSettings.model_validate(data.get("llm", {}))


class AnthropicClient:
    """Calls the Claude Messages API. Credentials come from the environment
    (ANTHROPIC_API_KEY or an `ant auth login` profile) and are never logged."""

    FALLBACK_BETA = "server-side-fallback-2026-07-01"

    def __init__(self, settings: LLMSettings, client: Any = None):
        import anthropic

        self.settings = settings
        self.client = client or anthropic.Anthropic()

    def complete(self, system: str, prompt: str) -> str:
        import anthropic

        s = self.settings
        kwargs: dict[str, Any] = {
            "model": s.model,
            "max_tokens": s.max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": prompt}],
            "output_config": {"effort": s.effort},
        }
        try:
            if s.fallbacks:
                response = self.client.beta.messages.create(
                    betas=[self.FALLBACK_BETA], fallbacks=s.fallbacks, **kwargs
                )
            else:
                response = self.client.messages.create(**kwargs)
        except anthropic.AuthenticationError as e:
            raise LLMError("Anthropic authentication failed: set ANTHROPIC_API_KEY.") from e
        except anthropic.RateLimitError as e:
            raise LLMError(f"Anthropic rate limit hit after SDK retries: {e.message}") from e
        except anthropic.APIStatusError as e:
            raise LLMError(f"Anthropic API error {e.status_code}: {e.message}") from e
        except anthropic.APIConnectionError as e:
            raise LLMError(f"Could not reach the Anthropic API: {e}") from e

        if response.stop_reason == "refusal":
            raise LLMError("The model declined the request (stop_reason=refusal).")
        text = "".join(block.text for block in response.content if block.type == "text").strip()
        if not text:
            raise LLMError(f"Empty answer from the model (stop_reason={response.stop_reason}).")
        return text


class FakeLLM:
    """Test double. Give it a list of scripted answers (returned in order) or a function
    (system, prompt) -> answer. Records every call; never touches the network."""

    def __init__(self, responses: list[str] | Callable[[str, str], str]):
        self.responses = responses
        self.calls: list[dict[str, str]] = []

    def complete(self, system: str, prompt: str) -> str:
        self.calls.append({"system": system, "prompt": prompt})
        if callable(self.responses):
            return self.responses(system, prompt)
        if not self.responses:
            raise LLMError("FakeLLM ran out of scripted responses")
        return self.responses.pop(0)


_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def extract_json_object(text: str) -> dict[str, Any]:
    """The JSON object in an answer: inside a code fence, or the first {...} that parses."""
    fenced = _FENCE.search(text)
    candidates = [fenced.group(1)] if fenced else []
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch == "{":
            try:
                value, _ = decoder.raw_decode(text, i)
            except json.JSONDecodeError:
                continue
            candidates.append(json.dumps(value))
            break
    for candidate in candidates:
        value = json.loads(candidate)
        if isinstance(value, dict):
            return value
    raise ValueError("the answer contains no JSON object")


def ask_json(
    llm: LLMClient,
    system: str,
    prompt: str,
    schema: type[T],
    check: Callable[[T], Any] | None = None,
    log: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[T, Any]:
    """Ask for JSON matching `schema`; `check` may do further validation and return a derived
    value (e.g. the TrainConfig built from a proposal). Retries once, then raises
    InvalidLLMOutput. Returns (parsed answer, check result)."""
    current = prompt
    error = ""
    for attempt in (1, 2):
        answer = llm.complete(system, current)
        try:
            parsed = schema.model_validate(extract_json_object(answer))
            derived = check(parsed) if check else None
        except (ValueError, ValidationError) as e:
            error = str(e)
            if log:
                log({"attempt": attempt, "valid": False, "answer": answer, "error": error})
            current = (
                f"{prompt}\n\nYour previous answer was rejected:\n{error}\n\n"
                "Reply again with ONLY a corrected JSON object."
            )
            continue
        if log:
            log({"attempt": attempt, "valid": True, "answer": answer})
        return parsed, derived
    raise InvalidLLMOutput(f"LLM output invalid twice ({schema.__name__}): {error}")
