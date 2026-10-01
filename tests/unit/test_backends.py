import asyncio
import logging

import pytest

from beyond_answer_confidence.backends.base import Request, parse_response
from beyond_answer_confidence.backends.fake import FakeBackend, fake_body
from beyond_answer_confidence.backends.jev import JevBackend


def test_parse_response_all_answer_types() -> None:
    raw = {
        "model": "m-1",
        "usage": {"input_tokens": 12, "output_tokens": 3},
        "answers": {
            "pick": {
                "type": "choice",
                "choice": "a",
                "probabilities": {"a": 0.7, "b": 0.3},
            },
            "yes": {"type": "noul", "noul": 0.25},
            "rate": {
                "type": "score",
                "score": 1.5,
                "probabilities": {"0": 0.5, "1": 0.5},
            },
        },
    }
    r = parse_response(raw)
    assert r.model == "m-1"
    assert (r.input_tokens, r.output_tokens) == (12, 3)
    assert r.choices["pick"].probabilities == {"a": 0.7, "b": 0.3}
    assert r.choices["pick"].choice == "a"
    assert r.yes_no["yes"].p_yes == 0.25
    assert r.scores["rate"].score == 1.5


def test_parse_response_rejects_unknown_type() -> None:
    with pytest.raises(ValueError, match="unknown answer type"):
        parse_response({"model": "m", "answers": {"x": {"type": "essay"}}})


def test_parse_response_tolerates_missing_usage() -> None:
    assert parse_response({"model": "m", "answers": {}}).input_tokens == 0


def test_fake_body_shapes() -> None:
    body = fake_body(
        {
            "c": {"type": "choice", "criteria": {"x": None, "y": None}},
            "n": {"type": "noul", "instructions": "?"},
            "s": {"type": "score", "criteria": ["lo", "mid", "hi"]},
        },
        "fake",
    )
    r = parse_response(body)
    assert r.choices["c"].probabilities == {"x": 0.75, "y": 0.25}
    assert r.yes_no["n"].p_yes == 0.75
    assert r.scores["s"].score == 1.0


def test_fake_backend_answers() -> None:
    async def go() -> float:
        async with FakeBackend() as fb:
            raw = await fb.answer(
                Request({"text": "hi"}, {"n": {"type": "noul"}}), "fake"
            )
        return parse_response(raw.body).yes_no["n"].p_yes

    assert asyncio.run(go()) == 0.75


def test_jev_backend_requires_context() -> None:
    backend = JevBackend()
    with pytest.raises(RuntimeError, match="outside"):
        asyncio.run(backend.answer(Request({}, {}), "jev-latest"))


def test_sdk_logger_pinned_to_warning() -> None:
    assert logging.getLogger("typesafe_sdk").level == logging.WARNING
