"""End-to-end runs of the follow-up experiments on synthetic data.

The live backend is the deterministic fake, dataset loaders return tiny
synthetic tables (from the item-set tests) and the sentence encoder is
replaced by hashed vectors. Each experiment runs once "live" into a
temporary cache, then offline from it.
"""

import json
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
import pytest
from test_item_set_experiments import (
    benchmark_tables,
    hotpot_table,
    oracle_table,
    popqa_table,
    qanta_table,
    run_twice,
    settings_for,
)

from beyond_answer_confidence.data import intents, loaders
from beyond_answer_confidence.data.intents import Example, IntentDataset
from beyond_answer_confidence.experiments import (
    benchmarks,
    contrastive_facts,
    evidence_sufficiency,
    follow_up_questions,
    inferred_settledness,
    knowledge_boundary,
    open_benchmarks,
    option_count,
    second_look,
    stated_odds,
    synthetic_worlds,
)

pytestmark = pytest.mark.usefixtures("fake_world")


@pytest.fixture
def fake_world(monkeypatch: pytest.MonkeyPatch) -> None:
    from test_item_set_experiments import hashed_embed

    from beyond_answer_confidence.backends.fake import FakeBackend
    from beyond_answer_confidence.data import distractors
    from beyond_answer_confidence.experiments import base

    monkeypatch.setattr(base, "JevBackend", FakeBackend)
    monkeypatch.setattr(distractors, "embed", hashed_embed)


@pytest.fixture
def source_tables(monkeypatch: pytest.MonkeyPatch) -> dict[str, pd.DataFrame]:
    t = benchmark_tables()
    t["ambigqa"] = ambig_table()
    monkeypatch.setattr(loaders, "load_popqa", lambda _d: popqa_table())
    monkeypatch.setattr(
        loaders, "load_daily_oracle", lambda kind, _d: oracle_table(kind)
    )
    monkeypatch.setattr(loaders, "load_hotpot_validation", lambda _d: hotpot_table())
    monkeypatch.setattr(loaders, "load_qanta", lambda fold, _d: qanta_table(fold))
    monkeypatch.setattr(loaders, "load_selfaware", lambda _d: t["selfaware"])
    monkeypatch.setattr(loaders, "load_simpleqa", lambda _d: t["simpleqa"])
    monkeypatch.setattr(loaders, "load_truthfulqa_binary", lambda _d: t["binary"])
    monkeypatch.setattr(loaders, "load_truthfulqa_mc1", lambda _d: t["mc1"])
    monkeypatch.setattr(loaders, "load_ambigqa_validation", lambda _d: t["ambigqa"])
    monkeypatch.setattr(loaders, "load_chaosnli", lambda s, _d: t[s])
    monkeypatch.setattr(loaders, "load_triviaqa_validation", lambda _d: t["triviaqa"])
    return t


def ambig_table() -> pd.DataFrame:
    words = ["Harbor", "Meadow", "Quarry", "Summit", "Tundra", "Valley", "Willow"]
    rows = []
    for i in range(8):
        if i % 2:
            ann = {
                "type": np.array(["multipleQAs"]),
                "qaPairs": [
                    {
                        "answer": [
                            np.array([words[i % 7]]),
                            np.array([words[(i + 1) % 7]]),
                        ]
                    }
                ],
                "answer": [],
            }
        else:
            ann = {
                "type": np.array(["singleAnswer"]),
                "qaPairs": [],
                "answer": [np.array([words[i % 7]])],
            }
        rows.append({"id": f"a{i}", "question": f"Vague {i}?", "annotations": ann})
    return pd.DataFrame(rows)


def test_inferred_settledness_end_to_end(tmp_path: Path) -> None:
    cfg = inferred_settledness.Config(events_per_kind=3, bootstrap=20)
    res = run_twice(inferred_settledness, tmp_path, cfg)["results"]
    assert set(res) >= {"tense", "date_with_today", "lottery_tense_and_date"}
    assert res["tense"]["auroc"] == 0.5  # the fake answers every item alike
    assert set(res["auroc_by_kind"]["tense"]) == {
        "auction",
        "award",
        "court",
        "election",
        "lottery",
        "sports",
    }
    assert res["choice"]["tense"]["share_first_listed"] == 1.0


def test_contrastive_facts_end_to_end(tmp_path: Path) -> None:
    cfg = contrastive_facts.Config(cases_per_policy=3, bootstrap=20)
    res = run_twice(contrastive_facts, tmp_path, cfg)["results"]
    assert res["irrelevant_fact_shift"]["mean"] == 0.0
    assert set(res["by_variant"]) == {
        "base",
        "contradictory",
        "counterfactual",
        "deleted",
        "irrelevant",
    }
    assert res["deleted_share_top_not_cannot_tell"] == 1.0  # fake picks option 1
    assert all(
        "p_holm" in res[k] for k in ("follows_evidence", "enough_flags_undecidable")
    )


def test_stated_odds_end_to_end(tmp_path: Path) -> None:
    cfg = stated_odds.Config(n_scenarios=40, bootstrap=10, averaging_draws=3)
    res = run_twice(stated_odds, tmp_path, cfg)["results"]
    words = res["fair_die_words"]
    assert words["n_orders"] == 720
    # The fake puts 0.5 + 0.5/6 on the first shown option; averaged over all
    # orders every face gets 1/6, so the order-averaged distance is zero.
    assert words["tv"] == pytest.approx(0.0, abs=1e-12)
    die = res["devices"]["die_words"]
    assert die["mean_probability_by_position"][0] == pytest.approx(0.5 + 0.5 / 6)
    assert die["permutation_averaging_tv"]["720"] == pytest.approx(0.0, abs=1e-12)
    assert set(res["relist_by_profile"]) == {"uniform", "binary"}


def test_stated_odds_incomplete_offline(tmp_path: Path) -> None:
    summary = stated_odds.run(settings_for(tmp_path, live=False), stated_odds.Config())
    assert summary["complete_items"] == 0
    assert "results" not in summary


def small_follow_up_config() -> follow_up_questions.Config:
    return follow_up_questions.Config(
        fabricated_per_relation=3,
        yes_no_per_month=4,
        wording_scenarios=4,
        wording_per_quintile=2,
        wording_fabricated_per_relation=2,
        wording_paragraph_questions=3,
        odds_scenarios=6,
        cutoff_month="2023-11",
        bootstrap=20,
    )


def test_follow_up_questions_end_to_end(
    tmp_path: Path, source_tables: dict[str, pd.DataFrame]
) -> None:
    del source_tables
    summary = run_twice(follow_up_questions, tmp_path, small_follow_up_config())
    res = summary["results"]
    assert set(res) == set(follow_up_questions.PROBES)
    assert summary["items_by_probe"] == summary["complete_by_probe"]
    assert set(res["wording"]["settled_wordings"]["wordings"]) == {
        "settled",
        "settled_2",
        "settled_3",
        "settled_4",
        "settled_rev",
    }
    assert set(res["odds_formats"]["by_format"]) == {"noul", "score", "instructed"}
    assert "unknown:post" in res["cutoff"]["by_condition"]
    assert "by_cell" not in res["warned_paragraphs"]  # no evidence rows written


def test_follow_up_questions_partial(
    tmp_path: Path, source_tables: dict[str, pd.DataFrame]
) -> None:
    del source_tables
    summary = follow_up_questions.run(
        settings_for(tmp_path, live=False), small_follow_up_config()
    )
    assert "results" not in summary
    assert summary["complete_by_probe"]["odds_formats"] == 0


def test_open_benchmarks_end_to_end(
    tmp_path: Path, source_tables: dict[str, pd.DataFrame]
) -> None:
    del source_tables
    cfg = open_benchmarks.Config(verification_questions=4, bootstrap=20)
    settings = settings_for(tmp_path, live=True)
    items = open_benchmarks.build_items(settings, cfg)
    reference = [
        {"unit": it.info["benchmark_unit"], "correct": i % 2 == 0}
        for i, it in enumerate(items)
        if it.info["set"] == "simpleqa_idk"
    ]
    base_dir = settings.out(open_benchmarks.BENCHMARKS_EXPERIMENT)
    base_dir.mkdir(parents=True)
    (base_dir / "rows.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in reference)
    )
    res = run_twice(open_benchmarks, tmp_path, cfg)["results"]
    assert res["trivia_verify"]["auroc_true_vs_false"] == 0.5
    assert res["simpleqa_idk"]["abstention_rate"] == 0.0  # the fake picks option A
    assert res["ambig_spread"]["n_ambiguous"] == 4
    assert res["ambiguity_spreads_probability"]["n"] == 4


def test_open_benchmarks_needs_reference_rows(
    tmp_path: Path, source_tables: dict[str, pd.DataFrame]
) -> None:
    del source_tables
    cfg = open_benchmarks.Config(verification_questions=4)
    summary = open_benchmarks.run(settings_for(tmp_path, live=True), cfg)
    assert summary["complete_items"] == summary["items"]
    assert "results" not in summary


def run_live(module: ModuleType, tmp_path: Path, config: Any) -> None:
    summary = module.EXPERIMENT.run(settings_for(tmp_path, live=True), config)
    assert "results" in summary


def test_second_look_end_to_end(
    tmp_path: Path, source_tables: dict[str, pd.DataFrame]
) -> None:
    del source_tables
    with pytest.raises(FileNotFoundError, match="synthetic_worlds"):
        second_look.requests(settings_for(tmp_path, live=False), second_look.Config())
    run_live(
        synthetic_worlds,
        tmp_path,
        synthetic_worlds.Config(scenarios=8, alone_scenarios=2, replicates=1),
    )
    run_live(
        knowledge_boundary,
        tmp_path,
        knowledge_boundary.Config(fabricated_per_relation=3, bootstrap=5),
    )
    run_live(evidence_sufficiency, tmp_path, evidence_sufficiency.Config(bootstrap=5))
    run_live(benchmarks, tmp_path, benchmarks.Config(bootstrap=5))
    cfg = second_look.Config(
        n_scenarios=8,
        fabricated_per_relation=3,
        per_quintile=2,
        cutoff_month="2023-11",
        trivia_questions=4,
        bootstrap=20,
    )
    res = run_twice(second_look, tmp_path, cfg)["results"]
    assert set(res["by_set"]) == {
        "hotpot",
        "oracle_tf",
        "popqa",
        "simpleqa",
        "synthetic",
        "triviaqa",
        "truthfulqa_binary",
    }
    assert set(res["synthetic_ideal_error"]["cells"]) == {"D0", "SFp"}
    assert res["synthetic_by_cell"]["D0"]["mean_ideal"] == 0.25
    rows = [
        json.loads(x) for x in (tmp_path / "out" / "second_look" / "rows.jsonl").open()
    ]
    assert rows
    assert all(r["p_correct"] == 0.75 for r in rows)  # fake P(yes) for 'correct'
    assert all(r["s_chance"] == 4.5 for r in rows)  # fake mid-scale score


def intent_dataset() -> IntentDataset:
    labels = tuple(f"intent_{i}" for i in range(6))
    test = tuple(Example(f"{lab} message {j}", lab) for lab in labels for j in range(2))
    return IntentDataset("clinc150", labels, (), (), test)


def exam_tables() -> dict[str, pd.DataFrame]:
    redux = pd.DataFrame(
        [
            {
                "question": f"Exam {i}?",
                "choices": np.array(["w", "x", "y", "z"]),
                "answer": i % 4,
                "error_type": ["ok", "ok", "wrong_groundtruth", "bad_question_clarity"][
                    i % 4
                ],
                "correct_answer": ["", "", "B", None][i % 4],
                "subject": "anatomy" if i < 4 else "virology",
            }
            for i in range(8)
        ]
    )
    cf = pd.DataFrame(
        [
            {
                "Question": f"Test {i}?",
                "A": "a",
                "B": "b",
                "C": "c",
                "D": "d",
                "Answer": "ABCD"[i % 4],
                "subject": "Math",
            }
            for i in range(4)
        ]
    )
    anli = pd.DataFrame(
        [
            {
                "uid": f"u{i}",
                "premise": f"Premise {i}.",
                "hypothesis": f"Claim {i}.",
                "label": i % 3,
                "round": 1 + i % 3,
            }
            for i in range(6)
        ]
    )
    return {"redux": redux, "cf": cf, "anli": anli}


def test_option_count_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    t = exam_tables()
    monkeypatch.setattr(option_count, "load_mmlu_redux", lambda _d: t["redux"])
    monkeypatch.setattr(option_count, "load_mmlu_cf", lambda _d: t["cf"])
    monkeypatch.setattr(option_count, "load_anli", lambda _d: t["anli"])
    monkeypatch.setattr(intents, "load", lambda _n, _d: intent_dataset())
    cfg = option_count.Config(intents_per_label=2, option_counts=(3, 6), bootstrap=20)
    res = run_twice(option_count, tmp_path, cfg)["results"]
    assert set(res["calibration_by_option_count"]["counts"]) == {3, 6}
    assert res["mmlu_redux_calibration"]["n"] == 4
    assert res["wrong_groundtruth_parsed"]["n"] == 2
    assert set(res["anli_by_round"]) == {1, 2, 3}
    assert 0.0 <= res["clinc_by_K"][3]["accuracy"] <= 1.0
