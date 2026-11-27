import asyncio
import json
import logging
from typing import Any

import httpx
import pytest

from beyond_answer_confidence import cli
from beyond_answer_confidence.backends import openrouter as orb
from beyond_answer_confidence.backends.base import (
    Request,
    is_fatal,
    parse_response,
    set_refusal_mode,
)
from beyond_answer_confidence.backends.gemini import GeminiBackend
from beyond_answer_confidence.backends.jev import JevBackend
from beyond_answer_confidence.backends.openai import OpenAIBackend
from beyond_answer_confidence.backends.openrouter import (
    OpenRouterAPIError,
    OpenRouterBackend,
    to_body,
    to_state,
)
from beyond_answer_confidence.experiments.base import backend_for
from beyond_answer_confidence.settings import Settings

QUESTIONS: dict[str, Any] = {
    "c": {
        "type": "choice",
        "instructions": "Which?",
        "criteria": {"a": None, "b": "bee"},
    },
    "y": {"type": "noul", "instructions": "Ok?"},
    "s": {"type": "score", "instructions": "Rate", "criteria": ["lo", "hi"]},
}

RESPONSE: dict[str, Any] = {
    "id": "gen-dec-123",
    "model": "typesafe/jev-1.13-20260901",
    "provider": "TypeSafe",
    "answers": {
        "c": {"type": "choice", "choice": "a", "probabilities": {"a": 0.8, "b": 0.2},
              "confidence": 0.8},
        "y": {"type": "noul", "noul": 0.3, "confidence": 0.7},
        "s": {"type": "score", "score": 0.6, "probabilities": {"0": 0.4, "1": 0.6},
              "legend": {"0": "lo", "1": "hi"}, "confidence": 0.6},
    },
    "usage": {"input_tokens": 120, "output_tokens": 0, "cost": 0.000005,
              "input_tokens_details": {"cached_tokens": 0}},
}  # fmt: skip


@pytest.fixture(autouse=True)
def _reset_mode() -> Any:
    yield
    set_refusal_mode("exclude")


def test_to_state() -> None:
    assert to_state("plain") == "plain"
    assert to_state({"b": 1, "a": "é"}) == '{"b":1,"a":"é"}'


def test_to_body_keeps_model_provider_cost() -> None:
    body = to_body(RESPONSE, QUESTIONS)
    assert body["model"] == "typesafe/jev-1.13-20260901"
    assert body["provider"] == "TypeSafe"
    assert body["usage"] == {"input_tokens": 120, "output_tokens": 0, "cost": 0.000005}
    parsed = parse_response(body)
    assert parsed.choices["c"].probabilities == {"a": 0.8, "b": 0.2}
    assert parsed.yes_no["y"].p_yes == 0.3
    assert parsed.scores["s"].score == 0.6


def test_refusal_is_stored_with_options() -> None:
    data = {**RESPONSE, "answers": {**RESPONSE["answers"], "c": {"type": "refusal"}}}
    body = to_body(data, QUESTIONS)
    assert body["answers"]["c"] == {
        "type": "refusal",
        "question_type": "choice",
        "options": ["a", "b"],
    }
    set_refusal_mode("uniform")
    assert parse_response(body).choices["c"].probabilities == {"a": 0.5, "b": 0.5}


def _backend(handler: Any) -> OpenRouterBackend:
    b = OpenRouterBackend(max_retries=3)
    b._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return b


def test_answer_posts_payload_and_parses() -> None:
    seen: list[dict[str, Any]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        assert str(req.url) == orb.ENDPOINT
        seen.append(json.loads(req.content))
        return httpx.Response(200, json=RESPONSE)

    raw = asyncio.run(
        _backend(handler).answer(
            Request({"case": 1}, QUESTIONS, 2), "typesafe/jev-1.13"
        )
    )
    assert seen == [{"model": "typesafe/jev-1.13", "state": '{"case":1}',
                     "questions": QUESTIONS}]  # fmt: skip
    assert raw.request_id == "gen-dec-123"
    assert raw.body["usage"]["input_tokens"] == 120


def test_rate_limit_is_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}
    slept: list[float] = []

    async def fake_sleep(s: float) -> None:
        slept.append(s)

    monkeypatch.setattr("asyncio.sleep", fake_sleep)

    def handler(req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(429, headers={"retry-after": "2"}, text="slow down")
        return httpx.Response(200, json=RESPONSE)

    raw = asyncio.run(_backend(handler).answer(Request({}, QUESTIONS), "x/y"))
    assert calls["n"] == 3
    assert slept == [2.0, 2.0]
    assert raw.body["model"] == "typesafe/jev-1.13-20260901"


def test_payment_required_is_fatal_and_not_retried() -> None:
    calls = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(402, text="insufficient credits")

    with pytest.raises(OpenRouterAPIError) as info:
        asyncio.run(_backend(handler).answer(Request({}, QUESTIONS), "x/y"))
    assert info.value.status == 402
    assert is_fatal(info.value)
    assert calls["n"] == 1


def test_error_messages_redact_key(monkeypatch: pytest.MonkeyPatch) -> None:
    marker = "placeholder-key-xyz"  # pragma: allowlist secret
    monkeypatch.setenv(orb.API_KEY_ENV, marker)

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text=f"bad key {marker}")

    with pytest.raises(OpenRouterAPIError) as info:
        asyncio.run(_backend(handler).answer(Request({}, QUESTIONS), "x/y"))
    assert marker not in str(info.value)
    assert info.value.status == 400
    assert not is_fatal(info.value)


def test_backend_without_key_stops(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(orb.API_KEY_ENV, raising=False)

    async def go() -> None:
        async with OpenRouterBackend():
            pass

    with pytest.raises(OpenRouterAPIError) as info:
        asyncio.run(go())
    assert is_fatal(info.value)


def test_backend_opens_and_closes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(orb.API_KEY_ENV, "placeholder")  # pragma: allowlist secret

    async def go() -> bool:
        b = OpenRouterBackend()
        async with b:
            opened = b._client is not None
        return opened and b._client is None

    assert asyncio.run(go())


def test_backend_requires_context() -> None:
    with pytest.raises(RuntimeError, match="outside"):
        asyncio.run(OpenRouterBackend().answer(Request({}, QUESTIONS), "x/y"))


def test_logger_pinned() -> None:
    assert logging.getLogger("httpx").level == logging.WARNING


def test_routing_and_key_variable() -> None:
    assert (
        backend_for(Settings(model="typesafe/jev-1.13", live=True)) is OpenRouterBackend
    )
    assert (
        backend_for(Settings(model="openai/gpt-6-luna-decisions", live=True))
        is OpenRouterBackend
    )
    assert backend_for(Settings(model="gpt-6-luna", live=True)) is OpenAIBackend
    assert backend_for(Settings(model="gemini-decision-1", live=True)) is GeminiBackend
    assert backend_for(Settings(model="jev-latest", live=True)) is JevBackend
    assert cli.api_key_env("perplexity/pplx-decider-v1.3:free") == "OPENROUTER_API_KEY"
    assert cli.api_key_env("gpt-6-luna") == "OPENAI_API_KEY"


def test_provider_suffix_pins_routing() -> None:
    from beyond_answer_confidence.backends.openrouter import payload_for, split_provider

    assert split_provider("cloudflare/clef-flash@deepinfra") == (
        "cloudflare/clef-flash",
        "deepinfra",
    )
    assert split_provider("typesafe/jev-1.13") == ("typesafe/jev-1.13", None)
    body = payload_for(Request({"a": 1}, QUESTIONS), "cloudflare/clef-flash@deepinfra")
    assert body["model"] == "cloudflare/clef-flash"
    assert body["provider"] == {"order": ["deepinfra"], "allow_fallbacks": False}
    assert "provider" not in payload_for(Request({}, QUESTIONS), "typesafe/jev-1.13")
