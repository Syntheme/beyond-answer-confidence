import json
from collections.abc import Iterable
from pathlib import Path

import pytest

from beyond_answer_confidence.backends.base import Request, estimate_tokens
from beyond_answer_confidence.backends.cache import request_key
from beyond_answer_confidence.reproduce import (
    Coverage,
    Target,
    coverage,
    estimate,
    estimate_all,
    read_cache_keys,
    run_checks,
    task_key,
    verify,
)
from beyond_answer_confidence.tasks.schema import Task

MODEL = "m"


def tasks(n: int, prefix: str = "u") -> list[Task]:
    return [
        Task(f"{prefix}{i}", Request({"i": i, "p": prefix}, {"q": {"type": "noul"}}))
        for i in range(n)
    ]


def write_cache(path: Path, ts: Iterable[Task], *, compact: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for t in ts:
            entry = {"key": task_key(t, MODEL), "response": {}}
            sep = (",", ":") if compact else None
            f.write(json.dumps(entry, separators=sep) + "\n")
        f.write("\n")


def test_task_key_matches_cache_key() -> None:
    t = tasks(1)[0]
    r = t.request
    assert task_key(t, MODEL) == request_key(MODEL, r.state, r.questions, 0)


@pytest.mark.parametrize("compact", [False, True])
def test_read_cache_keys(tmp_path: Path, compact: bool) -> None:
    ts = tasks(3)
    write_cache(tmp_path / "c.jsonl", ts, compact=compact)
    assert read_cache_keys(tmp_path / "c.jsonl") == {task_key(t, MODEL) for t in ts}


def test_read_cache_keys_skips_malformed_lines(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    ts = tasks(3)
    path = tmp_path / "c.jsonl"
    write_cache(path, ts[:2])
    content = "private-looking-content"
    truncated = json.dumps({"key": task_key(ts[2], MODEL), "response": {}})[:80]
    with path.open("a", encoding="utf-8") as f:
        f.write(f'garbage {content}\n[1, 2]\n{{"key": 5}}\n{truncated}')
    with caplog.at_level("WARNING"):
        keys = read_cache_keys(path)
    assert keys == {task_key(t, MODEL) for t in ts[:2]}
    assert "line 4" in caplog.text
    assert "line 7 (truncated final line)" in caplog.text
    assert "4 malformed line(s)" in caplog.text
    assert content not in caplog.text


def test_read_missing_cache_is_empty(tmp_path: Path) -> None:
    assert read_cache_keys(tmp_path / "none.jsonl") == set()


def test_coverage_counts_and_examples() -> None:
    ts = tasks(10)
    keys = {task_key(t, MODEL) for t in ts[:4]}
    c = coverage(iter(ts), MODEL, keys, examples=2)
    assert c == Coverage(10, 4, 6, ["u4", "u5"])
    assert coverage(ts, "other-model", keys).missing == 10


def test_estimate_dedupes_and_counts_tokens() -> None:
    ts = tasks(3)
    dup = [*ts, ts[2]]
    keys = {task_key(ts[0], MODEL)}
    e = estimate(dup, MODEL, keys)
    assert (e.requests, e.cached, e.new_requests) == (4, 1, 2)
    assert e.estimated_input_tokens == sum(estimate_tokens(t.request) for t in ts[1:])


def test_shared_cache_is_read_once(tmp_path: Path) -> None:
    a, b, c = tasks(3, "a"), tasks(2, "b"), tasks(2, "c")
    shared, own = tmp_path / "shared.jsonl", tmp_path / "own.jsonl"
    write_cache(shared, a + b[:1])
    write_cache(own, c)
    reads: list[Path] = []

    def load(path: Path) -> set[str]:
        reads.append(path)
        return read_cache_keys(path)

    targets = [
        Target("a", shared, lambda: a),
        Target("c", own, lambda: c),
        Target("b", shared, lambda: b),
    ]
    out = verify(targets, MODEL, load)
    assert reads == [shared, own]
    assert list(out["experiments"]) == ["a", "c", "b"]
    assert out["experiments"]["b"]["missing"] == 1
    assert (out["requests"], out["missing"], out["ok"]) == (7, 1, False)


def test_verify_ok_and_errors(tmp_path: Path) -> None:
    ts = tasks(2)
    write_cache(tmp_path / "c.jsonl", ts)

    def broken() -> list[Task]:
        raise FileNotFoundError("upstream rows missing")

    good = verify([Target("x", tmp_path / "c.jsonl", lambda: ts)], MODEL)
    assert good["ok"]
    assert good["errors"] == []
    out = verify(
        [
            Target("x", tmp_path / "c.jsonl", lambda: ts),
            Target("y", tmp_path / "c.jsonl", broken),
        ],
        MODEL,
    )
    assert out["errors"] == ["y"]
    assert "upstream rows missing" in out["experiments"]["y"]["error"]
    assert not out["ok"]


def test_run_checks_raises_without_catch(tmp_path: Path) -> None:
    def broken() -> list[Task]:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        run_checks([Target("y", tmp_path / "c.jsonl", broken)], lambda ts, k: 0)


def test_estimate_all(tmp_path: Path) -> None:
    ts = tasks(4)
    write_cache(tmp_path / "c.jsonl", ts[:1])
    out = estimate_all([Target("x", tmp_path / "c.jsonl", lambda: ts)], MODEL)
    assert out["new_requests"] == 3
    assert out["experiments"]["x"]["cached"] == 1
    assert out["estimated_input_tokens"] > 0
