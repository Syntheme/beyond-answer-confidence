import asyncio
import logging
from pathlib import Path
from typing import Any

import pytest

from beyond_answer_confidence import cli
from beyond_answer_confidence.backends import gemini
from beyond_answer_confidence.backends.base import Request, is_fatal, parse_response
from beyond_answer_confidence.backends.gemini import (
    GeminiAPIError,
    GeminiBackend,
    GeminiBlockedError,
    to_body,
    to_input,
)
from beyond_answer_confidence.backends.jev import JevBackend
from beyond_answer_confidence.experiments.base import backend_for
from beyond_answer_confidence.settings import Settings

RAW = {
    "answers": {
        "c": {
            "type": "choice",
            "choice": "a",
            "probabilities": {"a": 0.7, "b": 0.3},
            "confidence": 0.7,
        },
        "y": {"type": "noul", "noul": 0.99, "confidence": 0.98},
        "s": {
            "type": "score",
            "score": 1.2,
            "probabilities": {"0": 0.1, "1": 0.6, "2": 0.3},
            "legend": {"0": "low", "1": "mid", "2": "high"},
            "confidence": 0.6,
        },
    },
    "usage": {
        "total_input_tokens": 321,
        "input_tokens_by_modality": [{"modality": "text", "tokens": 321}],
        "total_output_tokens": 0,
        "total_tokens": 321,
    },
}


def test_to_input_serialises_state() -> None:
    assert to_input("plain") == "plain"
    assert to_input({"b": 1, "a": "é"}) == '{"b":1,"a":"é"}'


def test_to_body_matches_system_one_shape() -> None:
    body = to_body(RAW, "gemini-decision-1")
    assert body["model"] == "gemini-decision-1"
    assert body["usage"] == {"input_tokens": 321, "output_tokens": 0}
    parsed = parse_response(body)
    assert parsed.input_tokens == 321
    assert parsed.choices["c"].probabilities == {"a": 0.7, "b": 0.3}
    assert parsed.choices["c"].choice == "a"
    assert parsed.choices["c"].confidence == 0.7
    assert parsed.yes_no["y"].p_yes == 0.99
    assert parsed.scores["s"].score == 1.2
    assert parsed.scores["s"].probabilities["1"] == 0.6
    assert "legend" not in body["answers"]["s"]


def test_to_body_accepts_sdk_models() -> None:
    from google.genai import decisions

    response = decisions.CreateDecisionResponse.model_validate(RAW)
    assert to_body(response, "m") == to_body(RAW, "m")


def test_to_body_raises_on_block() -> None:
    raw = {"prompt_feedback": {"block_reason": "safety"}}
    with pytest.raises(GeminiBlockedError, match="safety"):
        to_body(raw, "m")


def test_to_body_rejects_unknown_answer() -> None:
    with pytest.raises(ValueError, match="unknown answer type"):
        to_body({"answers": {"q": {"type": "other"}}}, "m")


def test_status_errors_are_fatal_only_for_auth_and_billing() -> None:
    assert is_fatal(GeminiAPIError("x", 403))
    assert is_fatal(GeminiAPIError("x", 401))
    assert not is_fatal(GeminiAPIError("x", 400))
    assert not is_fatal(GeminiAPIError("x", None))


class _StatusError(Exception):
    def __init__(self, code: int) -> None:
        super().__init__(f"HTTP {code}")
        self.status_code = code


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
    backend = GeminiBackend()
    client = _Client(result)
    backend._client = client
    req = Request({"case": 1}, {"q": {"type": "noul", "instructions": "?"}}, 2)
    return asyncio.run(backend.answer(req, "gemini-decision-1")), client.decisions


def test_answer_sends_input_and_questions() -> None:
    raw, decisions = _answer(RAW)
    assert decisions.calls == [
        {
            "model": "gemini-decision-1",
            "input": '{"case":1}',
            "questions": {"q": {"type": "noul", "instructions": "?"}},
        }
    ]
    assert raw.body["usage"]["input_tokens"] == 321
    assert raw.request_id is None


def test_answer_maps_sdk_errors_to_status() -> None:
    with pytest.raises(GeminiAPIError) as info:
        _answer(_StatusError(403))
    assert info.value.status == 403
    assert is_fatal(info.value)


def test_backend_requires_context() -> None:
    req = Request({}, {"q": {"type": "noul", "instructions": "?"}})
    with pytest.raises(RuntimeError, match="outside"):
        asyncio.run(GeminiBackend().answer(req, "m"))


def test_backend_without_key_stops(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(gemini.API_KEY_ENV, raising=False)

    async def go() -> None:
        async with GeminiBackend():
            pass

    with pytest.raises(GeminiAPIError) as info:
        asyncio.run(go())
    assert is_fatal(info.value)


def test_key_not_kept_on_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    marker = "placeholder-value-123"  # pragma: allowlist secret
    monkeypatch.setenv(gemini.API_KEY_ENV, marker)

    async def go() -> list[str]:
        async with GeminiBackend() as backend:
            return [repr(v) for v in vars(backend).values()]

    assert not any(marker in v for v in asyncio.run(go()))


def test_loggers_pinned_to_warning() -> None:
    for name in ("google_genai", "httpx", "httpcore"):
        assert logging.getLogger(name).level == logging.WARNING


def test_backend_for_routes_by_model() -> None:
    assert backend_for(Settings(live=False, model="gemini-decision-1")) is None
    assert backend_for(Settings(live=True, model="gemini-decision-1")) is GeminiBackend
    assert backend_for(Settings(live=True, model="jev-latest")) is JevBackend


def test_cli_key_variable_depends_on_model() -> None:
    assert cli.api_key_env("gemini-decision-1") == "GOOGLE_GEMINI_API_KEY"
    assert cli.api_key_env("jev-latest") == cli.API_KEY_ENV
    assert cli.api_key_env(None) == cli.API_KEY_ENV


def test_cli_live_gemini_needs_gemini_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv(cli.API_KEY_ENV, "placeholder")  # pragma: allowlist secret
    monkeypatch.delenv(gemini.API_KEY_ENV, raising=False)
    argv = [
        "run",
        "synthetic_worlds",
        "--data-dir",
        str(tmp_path / "d"),
        "--output-dir",
        str(tmp_path / "o"),
        "--model",
        "gemini-decision-1",
        "--live",
        "--max-input-tokens",
        "9",
    ]
    assert cli.main(argv) == 2
    assert gemini.API_KEY_ENV in capsys.readouterr().err


def test_to_questions_renders_structured_fields_as_text() -> None:
    from beyond_answer_confidence.backends.gemini import to_questions

    qs: dict[str, Any] = {
        "c": {
            "type": "choice",
            "instructions": {"question": "Which?", "intents": ["a"]},
            "criteria": {"a": None, "b": {"name": "bee"}, "c": "plain"},
        },
        "s": {"type": "score", "instructions": "Rate", "criteria": ["lo", "hi"]},
        "y": {"type": "noul", "instructions": "Ok?"},
    }
    out = to_questions(qs)
    assert out["c"]["instructions"] == '{"question":"Which?","intents":["a"]}'
    assert out["c"]["criteria"] == {"a": "", "b": '{"name":"bee"}', "c": "plain"}
    assert out["s"] == qs["s"]
    assert out["y"] == qs["y"]
    assert qs["c"]["criteria"]["a"] is None  # input untouched


def test_converted_request_passes_sdk_validation() -> None:
    from google.genai import decisions

    from beyond_answer_confidence.backends.gemini import to_questions

    qs = {
        "c": {"type": "choice", "instructions": {"q": 1}, "criteria": {"a": None}},
        "y": {"type": "noul", "instructions": "Ok?"},
    }
    with pytest.raises(ValueError, match="string"):
        decisions.CreateDecisionRequest.model_validate(
            {"input": "x", "model": "m", "questions": qs}
        )
    decisions.CreateDecisionRequest.model_validate(
        {"input": to_input({"k": 1}), "model": "m", "questions": to_questions(qs)}
    )


def test_error_messages_redact_key(monkeypatch: pytest.MonkeyPatch) -> None:
    marker = "placeholder-key-xyz"  # pragma: allowlist secret
    monkeypatch.setenv(gemini.API_KEY_ENV, marker)
    assert gemini.redact(f"bad request ?key={marker}") == "bad request ?key=[redacted]"
    with pytest.raises(GeminiAPIError) as info:
        _answer(RuntimeError(f"echo {marker}"))
    assert marker not in str(info.value)
    assert info.value.__cause__ is None
    assert info.value.__suppress_context__


def test_redact_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(gemini.API_KEY_ENV, raising=False)
    assert gemini.redact("unchanged") == "unchanged"
