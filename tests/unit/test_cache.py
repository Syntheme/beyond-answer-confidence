import asyncio
import json
from pathlib import Path

import pytest

from beyond_answer_confidence.backends.base import (
    BudgetExceededError,
    FatalAPIError,
    MalformedResponseError,
    OfflineError,
    RawAnswer,
    Request,
    estimate_tokens,
    is_fatal,
)
from beyond_answer_confidence.backends.cache import (
    CachedClient,
    CallRecord,
    request_key,
)
from beyond_answer_confidence.backends.fake import FakeBackend, StatusError, fake_body

QUESTIONS = {
    "intent": {"type": "choice", "criteria": {"INTENT_0": None, "INTENT_1": None}}
}


def req(text: str, replicate: int = 0) -> Request:
    return Request({"text": text}, QUESTIONS, replicate)


def run(
    client: CachedClient,
    requests: list[Request],
    backend: FakeBackend | None,
    concurrency: int = 8,
) -> list[CallRecord | BaseException]:
    return asyncio.run(client.ask_many(requests, backend, concurrency=concurrency))


def test_request_key_is_frozen() -> None:
    # Changing the key format would orphan every existing cache.
    key = request_key(
        "jev-latest",
        {"text": "café order", "n": 2},
        {"q": {"type": "noul", "instructions": "Is it?"}},
        3,
    )
    assert key == "155f3aaf007e907e8428e854a0f1ec7133d394b7db8a8ed0846b44018b981ba5"


def test_request_key_is_canonical_and_replicate_sensitive() -> None:
    assert request_key("m", {"x": 1, "y": 2}, QUESTIONS) == request_key(
        "m", {"y": 2, "x": 1}, QUESTIONS
    )
    assert request_key("m", {}, QUESTIONS, 1) != request_key("m", {}, QUESTIONS)
    assert request_key("other", {}, QUESTIONS) != request_key("m", {}, QUESTIONS)


def test_dedupes_caches_and_keeps_order(tmp_path: Path) -> None:
    backend = FakeBackend()
    client = CachedClient(tmp_path / "c.jsonl", model="m", max_input_tokens=10**6)
    requests = [req("a"), req("b"), req("a"), req("a", replicate=1)]
    out = run(client, requests, backend, concurrency=4)
    assert backend.calls == 3
    assert all(isinstance(r, CallRecord) for r in out)
    assert isinstance(out[0], CallRecord)
    assert isinstance(out[2], CallRecord)
    assert out[0].key == out[2].key
    again = run(client, requests, backend)
    assert backend.calls == 3
    assert all(isinstance(r, CallRecord) and r.cached for r in again)


def test_cache_persists_and_offline_reads_it(tmp_path: Path) -> None:
    path = tmp_path / "c.jsonl"
    live = CachedClient(path, model="m", max_input_tokens=10**6)
    (first,) = run(live, [req("a")], FakeBackend())
    offline = CachedClient(path, model="m")
    assert len(offline) == 1
    assert isinstance(first, CallRecord)
    assert offline.get(req("a")).response == first.response
    assert offline.has(req("a"))
    assert not offline.has(req("z"))
    with pytest.raises(OfflineError):
        offline.get(req("z"))
    out = run(offline, [req("a"), req("z")], None)
    assert isinstance(out[0], CallRecord)
    assert isinstance(out[1], OfflineError)


def test_cache_entry_fields_and_no_secret(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_key = "not-a-real-key-0000"  # pragma: allowlist secret
    monkeypatch.setenv("TYPESAFE_API_KEY", fake_key)
    path = tmp_path / "c.jsonl"
    run(
        CachedClient(path, model="m", max_input_tokens=10**6), [req("a")], FakeBackend()
    )
    text = path.read_text()
    assert fake_key not in text
    assert set(json.loads(text)) == {
        "key", "model", "replicate", "state", "questions",
        "response", "request_id", "latency_s", "timestamp",
    }  # fmt: skip


def test_respects_concurrency(tmp_path: Path) -> None:
    backend = FakeBackend()
    client = CachedClient(tmp_path / "c.jsonl", model="m", max_input_tokens=10**6)
    run(client, [req(str(i)) for i in range(20)], backend, concurrency=3)
    assert 1 < backend.max_concurrent <= 3


def test_failures_are_returned_in_place_and_not_cached(tmp_path: Path) -> None:
    backend = FakeBackend(fail_on=frozenset({"bad"}))
    client = CachedClient(tmp_path / "c.jsonl", model="m", max_input_tokens=10**6)
    out = run(client, [req("ok"), req("bad")], backend)
    assert isinstance(out[0], CallRecord)
    assert isinstance(out[1], ConnectionError)
    assert len(client) == 1


def test_stops_after_fatal_status(tmp_path: Path) -> None:
    backend = FakeBackend(status_on={"2": 402})
    client = CachedClient(tmp_path / "c.jsonl", model="m", max_input_tokens=10**6)
    out = run(client, [req(str(i)) for i in range(6)], backend, concurrency=1)
    assert [type(r).__name__ for r in out[:3]] == [
        "CallRecord",
        "CallRecord",
        "StatusError",
    ]
    assert all(isinstance(r, FatalAPIError) for r in out[3:])
    assert backend.calls == 2
    assert len(client) == 2


def test_non_fatal_status_continues(tmp_path: Path) -> None:
    backend = FakeBackend(status_on={"0": 500})
    client = CachedClient(tmp_path / "c.jsonl", model="m", max_input_tokens=10**6)
    out = run(client, [req(str(i)) for i in range(3)], backend, concurrency=1)
    assert isinstance(out[0], StatusError)
    assert all(isinstance(r, CallRecord) for r in out[1:])


def test_stops_at_budget(tmp_path: Path) -> None:
    # The estimate (~60 tokens) is far below the billed 1,000: the overshoot
    # is bounded by one call's estimation error.
    backend = FakeBackend(tokens_per_call=1_000)
    client = CachedClient(tmp_path / "c.jsonl", model="m", max_input_tokens=2_500)
    out = run(client, [req(str(i)) for i in range(10)], backend, concurrency=1)
    assert sum(isinstance(r, CallRecord) for r in out) == 3
    assert sum(isinstance(r, BudgetExceededError) for r in out) == 7
    assert client.spent_input_tokens == 3_000
    error = 1_000 - estimate_tokens(req("0"))
    assert client.spent_input_tokens <= client.max_input_tokens + error


def test_budget_reserves_the_calls_own_estimate(tmp_path: Path) -> None:
    requests = [req(str(i)) for i in range(10)]
    est = estimate_tokens(requests[0])
    assert all(estimate_tokens(r) == est for r in requests)
    # Accurate estimates: the cap is never exceeded, at any concurrency.
    for concurrency in (1, 4):
        backend = FakeBackend(tokens_per_call=est)
        client = CachedClient(
            tmp_path / f"c{concurrency}.jsonl", model="m", max_input_tokens=3 * est
        )
        out = run(client, requests, backend, concurrency=concurrency)
        assert sum(isinstance(r, CallRecord) for r in out) == 3
        assert client.spent_input_tokens == 3 * est
    # A budget one token short of a call's estimate allows no call at all.
    client = CachedClient(tmp_path / "d.jsonl", model="m", max_input_tokens=est - 1)
    (only,) = run(client, requests[:1], FakeBackend(tokens_per_call=est))
    assert isinstance(only, BudgetExceededError)


def test_concurrent_overshoot_is_bounded_per_slot(tmp_path: Path) -> None:
    requests = [req(str(i)) for i in range(40)]
    est = estimate_tokens(requests[0])
    billed, slots, cap = est + 7, 4, 10 * est
    client = CachedClient(tmp_path / "c.jsonl", model="m", max_input_tokens=cap)
    run(client, requests, FakeBackend(tokens_per_call=billed), concurrency=slots)
    assert client.spent_input_tokens <= cap + slots * (billed - est)


@pytest.mark.parametrize("concurrency", [0, -1, 65])
def test_invalid_concurrency_is_rejected(tmp_path: Path, concurrency: int) -> None:
    client = CachedClient(tmp_path / "c.jsonl", model="m")
    with pytest.raises(ValueError, match="concurrency"):
        run(client, [req("a")], None, concurrency=concurrency)


def test_malformed_lines_are_skipped_and_counted(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "c.jsonl"
    run(
        CachedClient(path, model="m", max_input_tokens=10**6),
        [req("a"), req("b")],
        FakeBackend(),
    )
    good = path.read_text().splitlines()
    marker = "do-not-log-this-content"
    path.write_text(
        f'{good[0]}\nnot json {marker}\n{{"no_key": 1}}\n{good[1]}\n'
        f'{{"key": "{marker}", "resp'
    )
    with caplog.at_level("WARNING"):
        client = CachedClient(path, model="m", max_input_tokens=10**6)
    assert len(client) == 2
    assert client.malformed_lines == 3
    assert "line 2" in caplog.text
    assert "line 5 (truncated final line)" in caplog.text
    assert marker not in caplog.text
    # The next append starts on a new line, so the file stays readable.
    run(client, [req("c")], FakeBackend())
    again = CachedClient(path, model="m")
    assert len(again) == 3
    assert again.malformed_lines == 3


def test_each_entry_is_one_write_on_an_append_descriptor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os

    writes: list[bytes] = []
    real_write = os.write

    def spy(fd: int, data: bytes) -> int:
        writes.append(bytes(data))
        return real_write(fd, data)

    monkeypatch.setattr(os, "write", spy)
    path = tmp_path / "c.jsonl"
    client = CachedClient(path, model="m", max_input_tokens=10**6)
    run(client, [req("a"), req("b"), req("c")], FakeBackend())
    assert len(writes) == 3
    assert all(w.endswith(b"\n") and w.count(b"\n") == 1 for w in writes)
    assert path.read_bytes() == b"".join(writes)


class UnparseableBackend(FakeBackend):
    async def answer(self, request: Request, model: str) -> RawAnswer:
        body = fake_body(request.questions, model, self.tokens_per_call)
        body["answers"] = {"intent": {"type": "mystery"}}
        self.calls += 1
        return RawAnswer(body)


def test_paid_unparsable_response_is_cached_counted_and_returned(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "c.jsonl"
    backend = UnparseableBackend(tokens_per_call=321)
    client = CachedClient(path, model="m", max_input_tokens=10**6)
    with caplog.at_level("WARNING"):
        (out,) = run(client, [req("a")], backend)
    assert isinstance(out, MalformedResponseError)
    assert "could not be parsed" in caplog.text
    assert client.api_calls == 1
    assert client.spent_input_tokens == 321
    assert len(client) == 1
    # Not paid again: the stored body answers (with the same error) offline.
    (again,) = run(client, [req("a")], backend)
    assert isinstance(again, MalformedResponseError)
    assert backend.calls == 1
    offline = CachedClient(path, model="m")
    with pytest.raises(MalformedResponseError):
        offline.get(req("a"))


def test_default_budget_allows_no_paid_calls(tmp_path: Path) -> None:
    backend = FakeBackend()
    client = CachedClient(tmp_path / "c.jsonl", model="m")
    (out,) = run(client, [req("a")], backend)
    assert isinstance(out, BudgetExceededError)
    assert backend.calls == 0


def test_estimate_tokens_grows() -> None:
    assert 0 < estimate_tokens(req("a")) < estimate_tokens(req("a" * 1000))


def test_estimate_tokens_is_conservative() -> None:
    # 1.5 characters per token, rounded up (measured requests used >= 1.7).
    r = req("a" * 1000)
    chars = len(json.dumps({"s": r.state, "q": r.questions}))
    assert estimate_tokens(r) == -(-chars * 2 // 3)
    assert estimate_tokens(r) >= chars / 1.7


def test_is_fatal() -> None:
    assert is_fatal(StatusError(401))
    assert is_fatal(FatalAPIError("x"))
    assert not is_fatal(StatusError(500))
    assert not is_fatal(ValueError("x"))
