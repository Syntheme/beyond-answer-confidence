import asyncio
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from beyond_answer_confidence import cli
from beyond_answer_confidence.backends.base import (
    MalformedResponseError,
    RawAnswer,
    RefusalError,
    Request,
    parse_response,
    refusal_mode,
    set_refusal_mode,
)
from beyond_answer_confidence.backends.cache import CachedClient

BODY: dict[str, Any] = {
    "model": "m",
    "usage": {"input_tokens": 50, "output_tokens": 0},
    "answers": {
        "c": {"type": "refusal", "question_type": "choice", "options": ["a", "b"]},
        "y": {"type": "refusal", "question_type": "noul", "options": []},
        "s": {"type": "refusal", "question_type": "score", "options": ["l", "m", "h"]},
        "k": {"type": "noul", "noul": 0.9},
    },
}


@pytest.fixture(autouse=True)
def _reset_mode() -> Iterator[None]:
    yield
    set_refusal_mode("exclude")


def test_exclude_mode_raises_refusal_error() -> None:
    assert refusal_mode() == "exclude"
    with pytest.raises(RefusalError) as info:
        parse_response(BODY)
    assert info.value.names == ["c", "y", "s"]
    assert isinstance(info.value, MalformedResponseError)


def test_uniform_mode_reads_refusals_as_no_preference() -> None:
    set_refusal_mode("uniform")
    r = parse_response(BODY)
    assert r.choices["c"].probabilities == {"a": 0.5, "b": 0.5}
    assert r.yes_no["y"].p_yes == 0.5
    assert r.scores["s"].score == 1.0
    assert r.yes_no["k"].p_yes == 0.9


def test_uniform_mode_needs_options() -> None:
    set_refusal_mode("uniform")
    bad = {"answers": {"c": {"type": "refusal", "question_type": "choice"}}}
    with pytest.raises(ValueError, match="refusal without"):
        parse_response(bad)


def test_unknown_mode_rejected() -> None:
    with pytest.raises(ValueError, match="refusal mode"):
        set_refusal_mode("drop")


class _Once:
    """A live backend that returns a refusal body and counts calls."""

    def __init__(self) -> None:
        self.calls = 0

    async def __aenter__(self) -> "_Once":
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def answer(self, request: Request, model: str) -> RawAnswer:
        self.calls += 1
        return RawAnswer(BODY)


def test_refusal_is_cached_counted_and_never_resent(tmp_path: Path) -> None:
    path = tmp_path / "c.jsonl"
    req = Request({"s": 1}, {"c": {"type": "choice", "instructions": "?"}})
    backend = _Once()
    client = CachedClient(path, model="m", max_input_tokens=1000)
    (first,) = asyncio.run(client.ask_many([req], backend))
    assert isinstance(first, RefusalError)
    assert client.spent_input_tokens == 50
    again = CachedClient(path, model="m", max_input_tokens=1000)
    (second,) = asyncio.run(again.ask_many([req], backend))
    assert isinstance(second, RefusalError)
    assert backend.calls == 1
    set_refusal_mode("uniform")
    (third,) = asyncio.run(again.ask_many([req], None))
    assert not isinstance(third, BaseException)
    assert third.response.choices["c"].probabilities == {"a": 0.5, "b": 0.5}


def test_run_flag_sets_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []
    monkeypatch.setattr(
        "beyond_answer_confidence.backends.base.set_refusal_mode", seen.append
    )
    argv = ["run", "synthetic_worlds", "--data-dir", str(tmp_path / "d")]
    argv += ["--output-dir", str(tmp_path / "o"), "--refusals", "uniform"]
    cli.main(argv)
    assert seen == ["uniform"]


def test_analysable_allows_only_refusal_gaps() -> None:
    from beyond_answer_confidence.experiments.base import analysable

    assert analysable(10, 10, {}, None)
    assert analysable(8, 10, {"RefusalError": 2}, None)
    assert not analysable(8, 10, {"RefusalError": 1, "OfflineError": 1}, None)
    assert not analysable(8, 10, {}, None)
    assert not analysable(8, 10, {"RefusalError": 2}, RuntimeError("fatal"))
    assert not analysable(0, 10, {"RefusalError": 10}, None)
