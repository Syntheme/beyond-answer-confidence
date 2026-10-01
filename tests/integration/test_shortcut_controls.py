"""End-to-end runs of the shortcut controls and the shortcut baselines.

The live backend is the deterministic fake and the dataset loaders return
the synthetic tables of :mod:`shortcut_world`. The experiment runs once
"live" into a temporary cache, then offline from it; the offline analysis
then reads its rows together with generated rows of the other experiments.
"""

import json
from pathlib import Path

import pytest
from shortcut_world import (
    CUTOFF,
    comparison_table,
    news_table,
    popqa_people,
    write_sources,
)
from test_item_set_experiments import run_twice, settings_for

from beyond_answer_confidence.analyses import shortcut_baselines
from beyond_answer_confidence.data import loaders
from beyond_answer_confidence.experiments import shortcut_controls
from beyond_answer_confidence.experiments.shortcut_controls import entity_items


@pytest.fixture
def world(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    from beyond_answer_confidence.backends.fake import FakeBackend
    from beyond_answer_confidence.experiments import base

    monkeypatch.setattr(base, "JevBackend", FakeBackend)
    monkeypatch.setattr(loaders, "load_popqa", lambda _d: popqa_people())
    monkeypatch.setattr(loaders, "load_daily_oracle", lambda kind, _d: news_table(kind))
    monkeypatch.setattr(
        loaders, "load_hotpot_validation", lambda _d: comparison_table()
    )
    write_sources(tmp_path / "out", entity_items(popqa_people(), 10, 0))
    return tmp_path


def controls_config() -> shortcut_controls.Config:
    return shortcut_controls.Config(
        yes_no_per_month=8,
        fabricated_per_relation=10,
        cutoff_month=CUTOFF,
        lookalike_per_relation=3,
        evidence_per_relation=2,
        semantic_orders=2,
        bootstrap=20,
    )


def test_shortcut_controls_end_to_end(world: Path) -> None:
    summary = run_twice(shortcut_controls, world, controls_config())
    res = summary["results"]
    assert list(res) == list(shortcut_controls.PARTS)
    assert summary["parts"]["semantics"]["items"] == 5 * 4 * 2
    assert (
        summary["parts"]["negated"]["items"] == 2 * summary["parts"]["masked"]["items"]
    )
    assert res["masked"]["n"] == res["masked"]["n_post"] * 2  # balanced subset
    assert set(res["masked"]["original"]["known_post_vs_pre"]) == {
        "auroc",
        "ci",
        "n",
        "n_positive",
    }
    assert res["lookalike"]["lookalike"]["n"] == 18
    assert res["evidence"]["n_items"] == 12
    # the fake answers the first option: the stated answer is followed when
    # it happens to be listed first
    assert 0 <= res["evidence"]["follows_stated_answer"] <= 1
    assert len(res["polarity"]["sum_minus_one_ci"]) == 2
    assert res["semantics"]["die_loaded"]["occur"]["stated_p_mode"] == 0.5
    assert set(res["negated"]) >= {"pre", "post", "post_wrong_yes_minus_no"}
    assert set(res["near_day"]) == {"no_date", "fixed_today", "today_just_after"}
    rows = [
        json.loads(x)
        for x in (world / "out" / "shortcut_controls" / "rows.jsonl")
        .read_text()
        .splitlines()
    ]
    assert {r["part"] for r in rows} == set(shortcut_controls.PARTS)
    masked = [r for r in rows if r["part"] == "masked"]
    assert all(r["unit"] == f"masked:{r['base']}" for r in masked)


def test_shortcut_controls_incomplete_offline(world: Path) -> None:
    summary = shortcut_controls.run(settings_for(world, live=False), controls_config())
    assert summary["complete_items"] == 0
    assert "results" not in summary


def test_shortcut_baselines_end_to_end(world: Path) -> None:
    shortcut_controls.run(settings_for(world, live=True), controls_config())
    labels = world / "labels.json"
    news_units = [
        json.loads(x)["base"]
        for x in (world / "out" / "shortcut_controls" / "rows.jsonl")
        .read_text()
        .splitlines()
        if json.loads(x)["part"] == "masked"
    ]
    labels.write_text(
        json.dumps(
            [
                {"unit": u, "label": "P" if i % 2 else "D"}
                for i, u in enumerate(news_units[:20])
            ]
        )
    )
    cfg = shortcut_baselines.Config(
        bootstrap=20,
        yes_no_per_month=8,
        fabricated_per_relation=10,
        lookalike_per_relation=3,
        cases_per_policy=4,
        cutoff_month=CUTOFF,
        recalibration_splits=2,
        null_months=5,
        decision_splits=3,
        min_train_months=6,
        date_edit_labels=str(labels),
    )
    summary = shortcut_baselines.ANALYSIS.run(settings_for(world, live=False), cfg)
    res = summary["results"]
    assert set(res) == {
        "entity_text_baselines",
        "news_text_baselines",
        "second_look_baseline",
        "proper_scores",
        "length_checks",
        "recalibration_groupings",
        "forward_chaining",
        "label_balance",
        "enough_length",
        "paired_known_vs_confidence",
        "grouped_text_baselines",
        "date_routing",
        "enough_decisions",
        "coverage_table",
        "floor_sensitivity",
        "masked_label_safe",
        "date_only_subset",
        "matched_confidence_baselines",
    }
    ent = res["entity_text_baselines"]
    assert ent["n_fabricated"] == 60
    assert ent["all_real"]["known"]["auroc"] > 0.7  # generated with a signal
    assert res["news_text_baselines"]["known"]["auroc"] > 0.7
    assert "benchmarks:bench_x" in res["proper_scores"]["sets"]
    assert "stated_odds:die_loaded" in res["proper_scores"]["stated_odds"]
    assert "synthetic_worlds:D0" in res["proper_scores"]["stated_odds"]
    assert "contrastive_facts" in res["floor_sensitivity"]["sets"]
    assert "contrastive_facts:base" in res["proper_scores"]["sets"]
    assert res["enough_length"]["enough"]["auroc"] > 0.7
    assert set(res["date_routing"]["curve"]) >= {"0.00", "0.50", "1.00"}
    assert res["masked_label_safe"]["hand_labelled_sample"]["n"] == 20
    assert set(res["coverage_table"]) >= {"base|pre", "unknown|post"}
    diff = res["paired_known_vs_confidence"]["news_dates_removed"]
    assert set(diff["known_minus_confidence"]) == {"diff", "ci95", "ci90"}
    saved = json.loads(
        (world / "out" / "shortcut_baselines" / "summary.json").read_text()
    )
    assert saved["config"]["bootstrap"] == 20


def test_shortcut_baselines_labels_file_must_exist(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        shortcut_baselines.read_labels(str(tmp_path / "missing.json"))
    assert shortcut_baselines.read_labels("") is None
