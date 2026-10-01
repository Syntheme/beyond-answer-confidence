from pathlib import Path

import pytest

from beyond_answer_confidence.backends.base import Request
from beyond_answer_confidence.backends.fake import FakeBackend
from beyond_answer_confidence.runner import chunks, run_collect, run_items
from beyond_answer_confidence.tasks.schema import (
    Item,
    Task,
    item_tasks,
    order_collisions,
)


def items() -> list[Item]:
    return [
        Item(
            unit=f"u{i}",
            state={"text": f"case {i}"},
            options=("a", "b", "c"),
            gold="a" if i % 2 == 0 else "b",
            instructions="Pick one.",
            nouls={"known": "Is it known?"},
            info={"group": i % 2},
        )
        for i in range(4)
    ]


def test_item_request_shape() -> None:
    it = Item(
        "u",
        {"text": "x"},
        options=("a", "b"),
        descriptions={"a": "first", "b": "second"},
        instructions="Pick.",
        nouls={"n": "?"},
        scores={"s": ("Rate.", ("low", "high"))},
    )
    r = it.request(2)
    assert r.replicate == 2
    assert list(r.questions) == ["answer", "n", "s"]
    assert r.questions["answer"]["criteria"] == {"a": "first", "b": "second"}
    assert r.questions["s"]["criteria"] == ["low", "high"]


def test_item_tasks_are_replicate_major() -> None:
    tasks = item_tasks(items()[:2], 2)
    assert [(t.unit, t.request.replicate) for t in tasks] == [
        ("u0", 0),
        ("u1", 0),
        ("u0", 1),
        ("u1", 1),
    ]


def test_run_items_end_to_end_then_offline(tmp_path: Path) -> None:
    cache = tmp_path / "cache.jsonl"
    rows, info = run_items(
        items(),
        cache=cache,
        model="fake",
        replicates=2,
        backend=FakeBackend,
        max_input_tokens=10**6,
    )
    assert info["complete_items"] == 4
    assert info["new_api_calls"] == 8
    assert info["models"] == ["fake"]
    first = rows[0]
    assert first["top"] == "a"
    assert first["correct"] is True
    assert first["p_max"] == pytest.approx(0.5 + 0.5 / 3)
    assert first["p_known"] == 0.75
    assert first["rep_tv"] == 0.0
    assert first["group"] == 0
    assert rows[1]["correct"] is False

    rows2, info2 = run_items(items(), cache=cache, model="fake", replicates=2)
    assert rows2 == rows
    assert info2["new_api_calls"] == 0


def test_offline_run_reports_missing_items(tmp_path: Path) -> None:
    rows, info = run_items(items(), cache=tmp_path / "c.jsonl", model="m", replicates=1)
    assert rows == []
    assert info["errors"] == {"OfflineError": 4}


def test_fatal_error_stops_run(tmp_path: Path) -> None:
    def backend() -> FakeBackend:
        return FakeBackend(status_on={"case 1": 402})

    from beyond_answer_confidence.backends.cache import CachedClient

    client = CachedClient(tmp_path / "c.jsonl", model="m", max_input_tokens=10**6)
    out = run_collect(
        item_tasks(items(), 1), client, backend=backend, concurrency=1, chunk_size=2
    )
    assert out.fatal is not None
    assert "402" in out.fatal
    assert sum(len(v) for v in out.records.values()) == 1


def test_order_collisions_rejected(tmp_path: Path) -> None:
    q1 = {"a": {"type": "choice", "criteria": {"x": None, "y": None}}}
    q2 = {"a": {"type": "choice", "criteria": {"y": None, "x": None}}}
    same = [Task("u1", Request({"m": 1}, q1)), Task("u2", Request({"m": 1}, q2))]
    assert order_collisions(same, "m") == 1
    split = [Task("u1", Request({"m": 1}, q1, 0)), Task("u2", Request({"m": 1}, q2, 1))]
    assert order_collisions(split, "m") == 0

    from beyond_answer_confidence.backends.cache import CachedClient

    with pytest.raises(ValueError, match="key order"):
        run_collect(same, CachedClient(tmp_path / "c.jsonl", model="m"))


def test_chunks() -> None:
    assert chunks([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]
