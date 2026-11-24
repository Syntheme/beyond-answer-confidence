import asyncio
import logging
from pathlib import Path
from typing import Any

import pytest

from beyond_answer_confidence import cli
from beyond_answer_confidence.backends import openai as oai
from beyond_answer_confidence.backends.base import Request, is_fatal, parse_response
from beyond_answer_confidence.backends.gemini import GeminiBackend
from beyond_answer_confidence.backends.jev import JevBackend
from beyond_answer_confidence.backends.openai import (
    OpenAIAPIError,
    OpenAIBackend,
    to_body,
    to_input,
    to_questions,
)
from beyond_answer_confidence.experiments.base import backend_for
from beyond_answer_confidence.settings import Settings

QUESTIONS: dict[str, Any] = {
    "c": {
        "type": "choice",
        "instructions": {"question": "Which?", "intents": ["a"]},
        "criteria": {"a": None, "b": {"name": "bee"}, "c": "plain"},
    },
    "y": {"type": "noul", "instructions": "Ok?"},
    "s": {"type": "score", "instructions": "Rate", "criteria": ["lo", "mid", "hi"]},
}

RAW: dict[str, Any] = {
    "model": "gpt-6-luna-2026-10-01",
    "answers": [
        {
            "type": "choice",
            "name": "c",
            "choice": "a",
            "confidence": 0.7,
            "probabilities": [
                {"value": "a", "probability": 0.7},
                {"value": "b", "probability": 0.2},
                {"value": "c", "probability": 0.1},
            ],
        },
        {"type": "predicate", "name": "y", "probability": 0.995},
        {
            "type": "score",
            "name": "s",
            "score": 1,
            "confidence": 0.6,
            "probabilities": [
                {"value": 0, "label": "lo", "probability": 0.1},
                {"value": 1, "label": "mid", "probability": 0.6},
                {"value": 2, "label": "hi", "probability": 0.3},
            ],
        },
    ],
    "usage": {
        "input_tokens": 321,
        "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
        "output_tokens": 0,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": 321,
    },
}


def test_to_input_sends_json_natively() -> None:
    assert to_input({"b": 1}) == [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_json", "data": {"b": 1}}],
        }
    ]
    assert to_input("plain")[0]["content"] == [{"type": "input_text", "text": "plain"}]


def test_to_questions_translates_every_type() -> None:
    out = to_questions(QUESTIONS)
    assert [q["name"] for q in out] == ["c", "y", "s"]
    assert out[0] == {
        "name": "c",
        "type": "choice",
        "instructions": '{"question":"Which?","intents":["a"]}',
        "choices": [
            {"value": "a"},
            {"value": "b", "description": '{"name":"bee"}'},
            {"value": "c", "description": "plain"},
        ],
    }
    assert out[1] == {"name": "y", "type": "predicate", "instructions": "Ok?"}
    assert out[2]["levels"] == [{"label": "lo"}, {"label": "mid"}, {"label": "hi"}]
    assert QUESTIONS["c"]["criteria"]["a"] is None  # input untouched


def test_to_questions_rejects_too_many_levels_and_unknown_types() -> None:
    many = {
        "s": {"type": "score", "instructions": "x", "criteria": list("abcdefghijk")}
    }
    with pytest.raises(ValueError, match="at most 10"):
        to_questions(many)
    with pytest.raises(ValueError, match="unknown question type"):
        to_questions({"q": {"type": "other", "instructions": "x"}})


def test_converted_request_passes_sdk_types() -> None:
    from openai.types import DecisionCreateParams  # noqa: F401  (import check)
    from openai.types.decision_create_params import (
        QuestionQuestionParamChoice,
        QuestionQuestionParamPredicate,
        QuestionQuestionParamScore,
    )

    allowed = {
        "choice": set(QuestionQuestionParamChoice.__annotations__),
        "predicate": set(QuestionQuestionParamPredicate.__annotations__),
        "score": set(QuestionQuestionParamScore.__annotations__),
    }
    for q in to_questions(QUESTIONS):
        assert set(q) <= allowed[q["type"]], q


def test_to_body_matches_system_one_shape() -> None:
    body = to_body(RAW, QUESTIONS)
    assert body["model"] == "gpt-6-luna-2026-10-01"
    assert body["usage"] == {"input_tokens": 321, "output_tokens": 0}
    parsed = parse_response(body)
    assert parsed.choices["c"].probabilities == {"a": 0.7, "b": 0.2, "c": 0.1}
    assert parsed.choices["c"].choice == "a"
    assert parsed.choices["c"].confidence == 0.7
    assert parsed.yes_no["y"].p_yes == 0.995
    assert parsed.scores["s"].score == 1
    assert parsed.scores["s"].probabilities == {"0": 0.1, "1": 0.6, "2": 0.3}


def test_to_body_accepts_sdk_models_and_unnamed_answers() -> None:
    from openai.types import Decision

    assert to_body(Decision.model_validate(RAW), QUESTIONS) == to_body(RAW, QUESTIONS)
    unnamed = {**RAW, "answers": [{**a, "name": None} for a in RAW["answers"]]}
    assert to_body(unnamed, QUESTIONS)["answers"] == to_body(RAW, QUESTIONS)["answers"]


def test_to_body_keeps_refusals_with_their_options() -> None:
    raw = {**RAW, "answers": [{"type": "refusal", "name": "c"}, *RAW["answers"][1:]]}
    body = to_body(raw, QUESTIONS)
    assert body["answers"]["c"] == {
        "type": "refusal",
        "question_type": "choice",
        "options": ["a", "b", "c"],
    }
    assert body["answers"]["y"] == {"type": "noul", "noul": 0.995}


def test_to_body_rejects_unknown_answer() -> None:
    with pytest.raises(ValueError, match="unknown answer type"):
        to_body({"answers": [{"type": "other", "name": "c"}]}, QUESTIONS)


class _StatusError(Exception):
    def __init__(self, status: int, code: str | None = None) -> None:
        super().__init__(f"HTTP {status}")
        self.status_code = status
        self.code = code


class _Decisions:
    def __init__(self, result: Any) -> None:
        self.result = result
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result


class _Client:
    def __init__(self, result: Any) -> None:
        self.decisions = _Decisions(result)


def _answer(result: Any) -> tuple[Any, _Decisions]:
    backend = OpenAIBackend()
    client = _Client(result)
    backend._client = client
    req = Request({"case": 1}, QUESTIONS, 2)
    return asyncio.run(backend.answer(req, "gpt-6-luna")), client.decisions


def test_answer_sends_translated_request() -> None:
    raw, decisions = _answer(RAW)
    (call,) = decisions.calls
    assert call["model"] == "gpt-6-luna"
    assert call["input"] == to_input({"case": 1})
    assert call["questions"] == to_questions(QUESTIONS)
    assert raw.body["usage"]["input_tokens"] == 321


@pytest.mark.parametrize(
    ("status", "code", "mapped", "fatal"),
    [
        (401, None, 401, True),
        (403, None, 403, True),
        (429, "insufficient_quota", 402, True),
        (429, "rate_limit_exceeded", 429, False),
        (400, None, 400, False),
    ],
)
def test_answer_maps_errors(
    status: int, code: str | None, mapped: int, fatal: bool
) -> None:
    with pytest.raises(OpenAIAPIError) as info:
        _answer(_StatusError(status, code))
    assert info.value.status == mapped
    assert is_fatal(info.value) is fatal


def test_error_messages_redact_key(monkeypatch: pytest.MonkeyPatch) -> None:
    marker = "placeholder-key-xyz"  # pragma: allowlist secret
    monkeypatch.setenv(oai.API_KEY_ENV, marker)
    with pytest.raises(OpenAIAPIError) as info:
        _answer(RuntimeError(f"echo {marker}"))
    assert marker not in str(info.value)
    assert info.value.__cause__ is None
    monkeypatch.delenv(oai.API_KEY_ENV)
    assert oai.redact("unchanged") == "unchanged"


def test_backend_requires_context() -> None:
    req = Request({}, {"q": {"type": "noul", "instructions": "?"}})
    with pytest.raises(RuntimeError, match="outside"):
        asyncio.run(OpenAIBackend().answer(req, "gpt-6-luna"))


def test_backend_without_key_stops(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(oai.API_KEY_ENV, raising=False)

    async def go() -> None:
        async with OpenAIBackend():
            pass

    with pytest.raises(OpenAIAPIError) as info:
        asyncio.run(go())
    assert is_fatal(info.value)


def test_backend_opens_and_closes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(oai.API_KEY_ENV, "placeholder")  # pragma: allowlist secret

    async def go() -> bool:
        backend = OpenAIBackend()
        async with backend:
            opened = backend._client is not None
        return opened and backend._client is None

    assert asyncio.run(go())


def test_loggers_pinned_to_warning() -> None:
    for name in ("openai", "httpx", "httpcore"):
        assert logging.getLogger(name).level == logging.WARNING


def test_backend_for_routes_by_model() -> None:
    assert backend_for(Settings(live=True, model="gpt-6-luna")) is OpenAIBackend
    assert backend_for(Settings(live=True, model="openai-x")) is OpenAIBackend
    assert backend_for(Settings(live=True, model="gemini-decision-1")) is GeminiBackend
    assert backend_for(Settings(live=True, model="jev-latest")) is JevBackend
    assert backend_for(Settings(live=False, model="gpt-6-luna")) is None


def test_cli_key_variable_for_openai() -> None:
    assert cli.api_key_env("gpt-6-luna") == "OPENAI_API_KEY"


def test_cli_live_openai_needs_openai_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv(cli.API_KEY_ENV, "placeholder")  # pragma: allowlist secret
    monkeypatch.delenv(oai.API_KEY_ENV, raising=False)
    argv = [
        "run",
        "synthetic_worlds",
        "--data-dir",
        str(tmp_path / "d"),
        "--output-dir",
        str(tmp_path / "o"),
        "--model",
        "gpt-6-luna",
        "--live",
        "--max-input-tokens",
        "9",
    ]
    assert cli.main(argv) == 2
    assert oai.API_KEY_ENV in capsys.readouterr().err
