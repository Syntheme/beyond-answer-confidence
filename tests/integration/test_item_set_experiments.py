"""End-to-end runs of the item-set experiments on synthetic tables.

The live backend is replaced by the deterministic fake, the dataset loaders
by tiny synthetic tables and the sentence encoder by hashed vectors, so no
network, key or model is needed. Each experiment runs once "live" (filling a
temporary cache), then again offline from that cache.
"""

import json
import zlib
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
import pytest

from beyond_answer_confidence.backends.cache import request_key
from beyond_answer_confidence.backends.fake import FakeBackend
from beyond_answer_confidence.config import as_dict
from beyond_answer_confidence.data import distractors, loaders
from beyond_answer_confidence.experiments import (
    base,
    benchmarks,
    evidence_sufficiency,
    knowledge_boundary,
    synthetic_worlds,
)
from beyond_answer_confidence.settings import Settings


def hashed_embed(texts: Sequence[str], _d: Path, batch: int = 256) -> np.ndarray:
    rows = [
        np.random.default_rng(zlib.crc32(t.encode())).normal(size=16) for t in texts
    ]
    m = np.array(rows, float).reshape(len(texts), 16)
    out: np.ndarray = m / np.linalg.norm(m, axis=1, keepdims=True)
    return out


@pytest.fixture
def fake_world(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(base, "JevBackend", FakeBackend)
    monkeypatch.setattr(distractors, "embed", hashed_embed)


def settings_for(tmp_path: Path, *, live: bool) -> Settings:
    return Settings(
        data_dir=tmp_path / "data",
        output_dir=tmp_path / "out",
        model="fake-model",
        live=live,
        max_input_tokens=10**9 if live else 0,
    )


def run_twice(module: ModuleType, tmp_path: Path, config: Any) -> dict[str, Any]:
    exp = module.EXPERIMENT
    live = exp.run(settings_for(tmp_path, live=True), config)
    tasks = exp.requests(settings_for(tmp_path, live=False), config)
    unique = {
        request_key(
            "fake-model", t.request.state, t.request.questions, t.request.replicate
        )
        for t in tasks
    }
    assert live["new_api_calls"] == len(unique) > 0  # identical requests go once
    assert "results" in live
    offline = exp.run(settings_for(tmp_path, live=False), config)
    assert offline["new_api_calls"] == 0
    assert json.dumps(offline["results"]) == json.dumps(live["results"])  # NaN-safe
    out = tmp_path / "out" / exp.name
    summary = json.loads((out / "summary.json").read_text())
    assert summary["config"] == as_dict(config)
    rows = [json.loads(x) for x in (out / "rows.jsonl").read_text().splitlines()]
    assert rows
    assert "supported" not in json.dumps(summary)
    return dict(offline)


def test_synthetic_worlds_end_to_end(tmp_path: Path, fake_world: None) -> None:
    cfg = synthetic_worlds.Config(
        scenarios=24, alone_scenarios=4, replicates=2, bootstrap=30
    )
    res = run_twice(synthetic_worlds, tmp_path, cfg)["results"]
    assert res["batching_control"]["D0"]["n"] == 4
    assert set(res["mode_collapse"]["SFp"]) == {"uniform", "skewed", "peaked", "binary"}
    assert res["known_fact"]["n"] == 24
    assert all("p_holm" in res[k] for k in ("known_fact", "past_draws_shift"))
    assert res["by_cell"]["D0"]["mean_replicate_tv"] == 0.0  # the fake is deterministic


def test_synthetic_worlds_reports_incomplete_runs(tmp_path: Path) -> None:
    cfg = synthetic_worlds.Config(scenarios=6, alone_scenarios=1, replicates=1)
    summary = synthetic_worlds.run(settings_for(tmp_path, live=False), cfg)
    assert summary["units_complete"] == 0
    assert "results" not in summary


def popqa_table() -> pd.DataFrame:
    rows = []
    for i in range(20):
        prop = "genre" if i % 2 else "sport"
        rows.append(
            {
                "id": i,
                "subj": f"Subject{i}",
                "prop": prop,
                "obj": f"{prop} {i}",
                "question": f"Which {prop} is linked to Subject{i}?",
                "possible_answers": [f"{prop} {i}"],
                "o_aliases": "",
                "s_pop": i * 7,
            }
        )
    return pd.DataFrame(rows)


def oracle_table(kind: str) -> pd.DataFrame:
    rows = []
    for m in range(14):
        month = f"{2023 + m // 12}-{m % 12 + 1:02d}"
        for i in range(4 if kind == "tf" else 2):
            gold = "Yes" if (m < 8) == (i < 3) else "No"
            rows.append(
                {
                    "question": f"Event {m}-{i}?",
                    "answer": gold if kind == "tf" else "abcd"[i],
                    "choice_a": "w",
                    "choice_b": "x",
                    "choice_c": "y",
                    "choice_d": "z",
                    "date": f"{month}-01",
                    "category": "misc",
                    "month": month,
                }
            )
    return pd.DataFrame(rows)


def test_knowledge_boundary_end_to_end(
    tmp_path: Path, fake_world: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(loaders, "load_popqa", lambda _d: popqa_table())
    monkeypatch.setattr(
        loaders, "load_daily_oracle", lambda kind, _d: oracle_table(kind)
    )
    cfg = knowledge_boundary.Config(fabricated_per_relation=3, bootstrap=20)
    res = run_twice(knowledge_boundary, tmp_path, cfg)["results"]
    assert res["fabricated"]["n"] == 6
    assert res["oracle_change_point"]["first_post_month"] == "2023-09"
    assert "confidence_after_vs_before_cutoff" in res
    assert len(res["oracle_tf_by_month"]) == 14


def hotpot_table() -> pd.DataFrame:
    titles = ["Alpha", "Beta", "Pad1", "Pad2", "Pad3"]
    return pd.DataFrame(
        [
            {
                "id": f"h{i}",
                "type": "comparison",
                "question": f"Compare {i}?",
                "answer": ["yes", "no", "Alpha"][i % 3],
                "context": {
                    "title": np.array(titles),
                    "sentences": [np.array([f"{t} fact {i}."]) for t in titles],
                },
                "supporting_facts": {"title": np.array(["Alpha", "Beta"])},
            }
            for i in range(6)
        ]
    )


def qanta_table(fold: str) -> pd.DataFrame:
    n_q = 6 if fold == "guessdev" else 4
    return pd.DataFrame(
        [
            {
                "qanta_id": f"{fold}{q}",
                "id": f"{fold}{q}-{s}",
                "text": f"Clue {s} about item {q}.",
                "page": f"{fold.title()}_Answer_{q}",
                "category": "Science",
            }
            for q in range(n_q)
            for s in range(1 + q % 3)
        ]
    )


def test_evidence_sufficiency_end_to_end(
    tmp_path: Path, fake_world: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(loaders, "load_hotpot_validation", lambda _d: hotpot_table())
    monkeypatch.setattr(loaders, "load_qanta", lambda fold, _d: qanta_table(fold))
    cfg = evidence_sufficiency.Config(bootstrap=20)
    res = run_twice(evidence_sufficiency, tmp_path, cfg)["results"]
    assert res["hotpot_by_cell"]["closed"]["n"] == 6
    assert set(res["quizbowl_by_k"]) == {1, 2, 3}
    saved = list((tmp_path / "data" / "cache" / "distractors").glob("*.json"))
    assert len(saved) == 1


def benchmark_tables() -> dict[str, pd.DataFrame]:
    words = ["Amber", "Basalt", "Cobalt", "Dune", "Ember", "Flint"]
    nli = pd.DataFrame(
        [
            {
                "uid": f"n{i}",
                "label_count": [60 - 10 * i, 20 + 5 * i, 20 + 5 * i],
                "old_labels": ["entailment", "neutral"],
                "example": {"premise": f"Premise {i}.", "hypothesis": f"Claim {i}."},
                "majority_label": "e",
                "entropy": 0.5 + 0.1 * i,
            }
            for i in range(4)
        ]
    )
    return {
        "selfaware": pd.DataFrame(
            [
                {
                    "question_id": i,
                    "question": f"Q{i}?",
                    "answerable": i % 2 == 0,
                    "source": "s",
                }
                for i in range(6)
            ]
        ),
        "simpleqa": pd.DataFrame(
            [
                {
                    "original_index": i,
                    "problem": f"When {i}?" if i < 3 else f"Who {i}?",
                    "answer": f"{1990 + i}" if i < 3 else words[i],
                    "topic": "t",
                    "answer_type": "Date" if i < 3 else "Person",
                }
                for i in range(6)
            ]
            + [
                {
                    "original_index": 9,
                    "problem": "Who else?",
                    "answer": words[0],
                    "topic": "t",
                    "answer_type": "Person",
                }
            ]
        ),
        "binary": pd.DataFrame(
            [
                {
                    "Question": f"Myth {i}?",
                    "Best Answer": "true thing",
                    "Best Incorrect Answer": "false thing",
                    "Category": "c",
                    "Type": "t",
                }
                for i in range(4)
            ]
        ),
        "mc1": pd.DataFrame(
            [
                {
                    "question": f"Pick {i}?",
                    "mc1_targets": {
                        "choices": np.array(["r", "s", "t"]),
                        "labels": np.array([0, 0, 1]),
                    },
                }
                for i in range(4)
            ]
        ),
        "ambigqa": pd.DataFrame(
            [
                {
                    "id": f"a{i}",
                    "question": f"Vague {i}?",
                    "annotations": {
                        "type": np.array(
                            [
                                ["singleAnswer"],
                                ["multipleQAs"],
                                ["singleAnswer", "multipleQAs"],
                            ][i % 3]
                        )
                    },
                }
                for i in range(6)
            ]
        ),
        "snli": nli,
        "mnli_m": nli.assign(uid=[f"m{i}" for i in range(4)]),
        "alphanli": pd.DataFrame(
            [
                {
                    "uid": f"x{i}",
                    "label_count": [30 + 10 * i, 70 - 10 * i],
                    "example": {"obs1": "o1", "obs2": "o2", "hyp1": "e1", "hyp2": "e2"},
                    "majority_label": 2,
                    "entropy": 0.9 - 0.1 * i,
                }
                for i in range(4)
            ]
        ),
        "triviaqa": pd.DataFrame(
            [
                {
                    "question_id": f"t{i}",
                    "question": f"Trivia {i}?",
                    "answer": {
                        "value": w.upper(),
                        "aliases": np.array([w]),
                        "type": "Free",
                    },
                }
                for i, w in enumerate(words)
            ]
        ),
    }


def test_benchmarks_end_to_end(
    tmp_path: Path, fake_world: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    t = benchmark_tables()
    monkeypatch.setattr(loaders, "load_selfaware", lambda _d: t["selfaware"])
    monkeypatch.setattr(loaders, "load_simpleqa", lambda _d: t["simpleqa"])
    monkeypatch.setattr(loaders, "load_truthfulqa_binary", lambda _d: t["binary"])
    monkeypatch.setattr(loaders, "load_truthfulqa_mc1", lambda _d: t["mc1"])
    monkeypatch.setattr(loaders, "load_ambigqa_validation", lambda _d: t["ambigqa"])
    monkeypatch.setattr(loaders, "load_chaosnli", lambda s, _d: t[s])
    monkeypatch.setattr(loaders, "load_triviaqa_validation", lambda _d: t["triviaqa"])
    cfg = benchmarks.Config(bootstrap=20)
    res = run_twice(benchmarks, tmp_path, cfg)["results"]
    assert set(res["chaosnli"]) == {"alphanli", "mnli_m", "snli"}
    assert res["calibration"]["truthfulqa_binary"]["n"] == 4
    assert res["selfaware_by_source"] == {"s": 0.25}  # fake P(yes) for 'answerable'
