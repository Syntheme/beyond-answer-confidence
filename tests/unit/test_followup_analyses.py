import json
import random
import zlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from beyond_answer_confidence.analyses import accuracy_trend as at
from beyond_answer_confidence.analyses import cue_controls as cue
from beyond_answer_confidence.analyses import distractor_examples as de
from beyond_answer_confidence.analyses import forward_recalibration as fr
from beyond_answer_confidence.analyses.shortcut_baselines import Config as BaseConfig
from beyond_answer_confidence.analyses.shortcut_baselines import Inputs
from beyond_answer_confidence.data import distractors as dis
from beyond_answer_confidence.data import loaders
from beyond_answer_confidence.experiments.base import write_jsonl
from beyond_answer_confidence.settings import Settings
from beyond_answer_confidence.tasks.schema import Item
from beyond_answer_confidence.tasks.shortcuts import (
    PERSON_RELATION_ORDER,
    lookalike_items,
)

CUTOFF = "2024-11"


def _months(start: str, n: int) -> list[str]:
    y, m = int(start[:4]), int(start[5:7])
    out = []
    for _ in range(n):
        out.append(f"{y}-{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def fake_embed(texts: Sequence[str], *_a: object, **_k: object) -> np.ndarray:
    """Deterministic unit vectors seeded by each text (no model)."""
    m = np.array(
        [np.random.default_rng(zlib.crc32(t.encode())).normal(size=16) for t in texts]
    ).reshape(len(texts), 16)
    out: np.ndarray = m / np.linalg.norm(m, axis=1, keepdims=True)
    return out


# --- accuracy_trend -------------------------------------------------------------


def _panel(acc: dict[str, float], n: int = 40, p_max: float = 0.9) -> pd.DataFrame:
    """Exactly ``round(acc * n)`` correct answers per month."""
    rows = []
    for month, a in acc.items():
        k = round(a * n)
        rows += [{"month": month, "correct": i < k, "p_max": p_max} for i in range(n)]
    return pd.DataFrame(rows)


def _step_panel() -> pd.DataFrame:
    months = _months("2024-01", 30)  # 2024-01 .. 2026-06
    return _panel({m: 0.8 if m < "2025-10" else 0.4 for m in months})


def test_quarter() -> None:
    assert at.quarter("2024-01") == "2024-Q1"
    assert at.quarter("2024-03") == "2024-Q1"
    assert at.quarter("2024-04") == "2024-Q2"
    assert at.quarter("2025-12") == "2025-Q4"


def test_by_quarter() -> None:
    tf = pd.DataFrame(
        {
            "month": ["2025-01", "2025-02", "2025-04", "2025-04"],
            "correct": [True, False, True, True],
            "p_max": [0.9, 0.7, 0.6, 0.8],
        }
    )
    q = at.by_quarter(tf)
    assert [r["quarter"] for r in q] == ["2025-Q1", "2025-Q2"]
    assert q[0]["n"] == 2
    assert q[0]["accuracy"] == pytest.approx(0.5)
    assert q[0]["confidence"] == pytest.approx(0.8)
    assert q[0]["gap"] == pytest.approx(0.3)
    assert q[1]["gap"] == pytest.approx(0.7 - 1.0)


def test_change_point_block_finds_a_planted_drop() -> None:
    res = at.change_point_block(_step_panel(), 3, 30, np.random.default_rng(0))
    assert res["months"] == "2024-01..2026-06"
    assert res["min_side"] == 3
    assert res["selected"]["first_post_month"] == "2025-10"
    assert res["selected"]["drop"] == pytest.approx(0.4)
    share = res["selected_month_share"]
    assert sum(share.values()) == pytest.approx(1.0)
    assert share["2025-10"] >= 0.8
    lo, hi = res["drop_ci95"]
    assert 0.2 < lo <= hi < 0.6


def test_trends_recover_a_linear_decline() -> None:
    months = _months("2024-11", 20)
    acc = {m: 0.9 - 0.2 * i / 12 for i, m in enumerate(months)}
    res = at.trends(_panel(acc, n=50), 50, np.random.default_rng(1))
    assert res["months"] == 20
    assert res["accuracy_per_year"] == pytest.approx(-0.2, abs=0.02)
    assert res["gap_per_year"] == pytest.approx(0.2, abs=0.02)
    lo, hi = res["accuracy_per_year_ci95"]
    assert lo <= res["accuracy_per_year"] <= hi
    assert hi < 0
    assert res["gap_per_year_ci95"][0] > 0


def test_accuracy_trend_run_reads_news_rows(tmp_path: Path) -> None:
    tf = _step_panel()
    rows: list[dict[str, Any]] = [
        {str(k): v for k, v in r.items()} | {"set": "oracle_tf"}
        for r in tf.to_dict("records")
    ]
    # Rows of another set must be ignored.
    rows += [{"month": "2026-06", "correct": True, "p_max": 0.5, "set": "other"}] * 50
    settings = Settings(data_dir=tmp_path / "data", output_dir=tmp_path / "out")
    write_jsonl(settings.out("knowledge_boundary") / "rows.jsonl", rows)
    config = at.Config(bootstrap=10, seed=3)
    summary = at.run(settings, config)
    saved = json.loads((settings.out(at.NAME) / "summary.json").read_text())
    assert saved["config"]["bootstrap"] == 10
    res = summary["results"]
    assert res["n"] == len(tf)
    assert res["by_quarter"][0]["quarter"] == "2024-Q1"
    assert len(res["by_quarter"]) == 10
    assert res["change_point_full"]["selected"]["first_post_month"] == "2025-10"
    late = res["change_point_late"]
    assert late["months"] == "2025-01..2026-06"
    assert late["selected"]["first_post_month"] == "2025-10"
    trend = res["trend_after_start"]
    assert trend["months"] == 20
    assert trend["accuracy_per_year"] < 0
    assert trend["gap_per_year"] > 0
    assert saved["results"]["n"] == res["n"]
    # Same seed, same summary.
    assert at.run(settings, config) == summary


# --- cue_controls ---------------------------------------------------------------


def test_one_way_clusters_and_linked_components() -> None:
    donors = [("Ann Bell", "Cy Dunn"), ("Cy Dunn", "Ed Fox"), ("Gil Hale", "Ivy Joy")]
    c = cue.one_way_clusters(["Real One", "Real Two"], donors)
    assert list(c["first_name_donor"]) == [
        "Real One",
        "Real Two",
        "Ann Bell",
        "Cy Dunn",
        "Gil Hale",
    ]
    assert list(c["surname_donor"]) == [
        "Real One",
        "Real Two",
        "Cy Dunn",
        "Ed Fox",
        "Ivy Joy",
    ]
    # The first two made-up names share "Cy Dunn" and join one component.
    linked = list(c["linked"])
    assert linked[2] == linked[3]
    assert linked[4] != linked[2]
    assert len(set(linked)) == 4


def test_name_features() -> None:
    names = ["J. van Berg-Smith", "Ann Bell", "Zoë Ångström", ""]
    ref = ["Ann Bell", "Ann Cole", "Bo Bell"]
    f = cue.name_features(names, ref)
    assert f.shape == (4, 10)
    np.testing.assert_allclose(f[0, :7], [3, 17, 5.0, 1, 1, 1, 0])
    assert f[0, 9] == pytest.approx(2 / 3)
    assert f[1, 7] == pytest.approx(np.log1p(2))  # "ann" twice in the reference
    assert f[1, 8] == pytest.approx(np.log1p(2))  # "bell" twice
    assert f[2, 6] == 1  # non-ASCII
    assert f[2, 7] == 0.0
    assert f[3, 0] == 1  # empty name: one empty token
    assert f[3, 9] == 0.0


def _separable(n: int = 40) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    y = np.arange(n) % 2 == 1
    texts = [
        f"Who is {'zorbly quax' if lab else 'plain person'} number {i}?"
        for i, lab in enumerate(y)
    ]
    groups = np.array([f"g{i // 2}" for i in range(n)])
    emb = rng.normal(size=(n, 4)) + y[:, None] * 3.0
    return texts, y, groups, emb


def test_fit_text_controls_with_permutations() -> None:
    texts, y, groups, emb = _separable()
    feats = np.c_[y * 1.0, np.random.default_rng(1).normal(size=len(y))]
    res = cue.fit_text_controls(
        texts, y, groups, emb, feats, 20, np.random.default_rng(0), permutations=1
    )
    assert set(res) == {
        "char_ngram",
        "word_ngram",
        "sentence_embedding",
        "name_structure",
        "embedding_and_name_structure",
    }
    for r in res.values():
        assert r["auroc"] > 0.9
        assert r["permutation"]["shuffles"] == 1
        assert 0.0 <= r["permutation"]["mean"] <= 1.0
        assert r["permutation"]["p_at_least_observed"] == 0.0
    plain = cue.fit_text_controls(
        texts, y, groups, emb, None, 10, np.random.default_rng(0), clusters=groups
    )
    assert set(plain) == {"char_ngram", "word_ngram", "sentence_embedding"}
    assert "permutation" not in plain["char_ngram"]


def test_oof_keeps_groups_out_of_their_own_fold() -> None:
    n = 30
    y = np.arange(n) % 2 == 0
    groups = np.array([i // 3 for i in range(n)])
    seen: list[tuple[set[int], set[int]]] = []

    def fit(train: np.ndarray, test: np.ndarray) -> np.ndarray:
        seen.append((set(groups[train]), set(groups[test])))
        return np.full(len(test), float(len(seen)))

    out = cue._oof(fit, n, y, groups)
    assert len(seen) == 5
    assert all(not (train_g & test_g) for train_g, test_g in seen)
    # Every item scored once, by its own fold.
    assert set(out) == {1.0, 2.0, 3.0, 4.0, 5.0}


def test_lookalike_block_known_beats_confidence() -> None:
    rng = np.random.default_rng(0)
    n = 60
    y = np.arange(n) % 2 == 1
    known = np.where(y, 0.1, 0.9) + rng.uniform(-0.05, 0.05, n)
    conf = rng.uniform(0.5, 1.0, n)
    relations = np.array([f"r{i % 3}" for i in range(n)])
    people = np.array([f"p{i // 4}" for i in range(n)])
    clusterings = {
        "first_name_donor": people,
        "surname_donor": np.array([f"s{i // 5}" for i in range(n)]),
        "linked": np.array([f"l{i // 10}" for i in range(n)]),
    }
    res = cue.lookalike_block(known, conf, y, clusterings, relations, 20, rng)
    assert res["n_items"] == n
    assert res["n_made_up"] == 30
    assert res["n_clusters"] == {
        "first_name_donor": 15,
        "surname_donor": 12,
        "relation": 3,
    }
    assert res["largest_linked_component_items"] == 10
    for key in ("by_item", "by_first_name_donor", "by_surname_donor", "by_relation"):
        assert res[key]["diff"] > 0.3
        assert res[key]["ci95"][0] > 0
    loro = res["leave_one_relation_out"]
    assert set(loro["diff"]) == {"r0", "r1", "r2"}
    assert 0.3 < loro["min"] <= loro["max"] <= 1.0


FIRSTS = [f"{a}{b}" for a in ("Al", "Bo", "Cy", "Di", "Ed", "Fa") for b in "mnrst"]
LASTS = [f"{a}ford" for a in ("Ash", "Birch", "Cedar", "Dale", "Elm", "Fern")]


def _popqa() -> pd.DataFrame:
    rows = []
    for r, prop in enumerate(PERSON_RELATION_ORDER):
        for i in range(5):
            k = 5 * r + i
            subj = f"{FIRSTS[k]} {LASTS[(r + i) % 6]}"
            rows.append(
                {
                    "id": k,
                    "subj": subj,
                    "prop": prop,
                    "obj": f"{prop} value {i}",
                    "question": f"What is the {prop} of {subj}?",
                    "possible_answers": [f"{prop} value {i}"],
                    "o_aliases": "",
                    "s_pop": i,
                }
            )
    return pd.DataFrame(rows)


def _cue_inputs(per_relation: int) -> tuple[Inputs, pd.DataFrame]:
    popqa = _popqa()
    rng = np.random.default_rng(5)
    entities = [
        Item(
            unit=f"popqa:{r.id}",
            state={"question": r.question},
            info={"prop": r.prop, "s_pop": r.s_pop, "fabricated": False},
        )
        for r in popqa.itertuples()
    ] + [
        Item(
            unit="fab:0",
            state={"question": "Who?"},
            info={"prop": "sport", "s_pop": 0, "fabricated": True},
        )
    ]
    first: list[dict[str, Any]] = [
        {"unit": it.unit, "p_known": float(rng.uniform(0.7, 0.9)), "p_max": 0.8}
        for it in entities
    ]
    look = lookalike_items(popqa, per_relation)
    controls = [
        {
            "part": "lookalike",
            "unit": it.unit,
            "prop": it.info["prop"],
            "subject": it.info["subject"],
            "p_known": float(rng.uniform(0.1, 0.3)),
            "p_max": float(rng.uniform(0.6, 1.0)),
        }
        for it in look
    ]
    news = []
    for m in _months("2024-06", 10):
        post = m >= CUTOFF
        for j in range(6):
            topic = "tariff ruling" if post else "election result"
            unit = f"news:{m}:{j}"
            news.append(
                Item(
                    unit=unit,
                    state={"question": f"Will the {topic} {j} hold by March 2025?"},
                    info={"month": m},
                )
            )
            first.append({"unit": unit, "p_known": 0.5, "p_max": 0.7})
    news.append(Item(unit="news:unscored", state={"question": "?"}, info={}))
    inputs = Inputs(
        rows={"knowledge_boundary": first, "shortcut_controls": controls},
        entities=entities,
        subjects={f"popqa:{r.id}": str(r.subj) for r in popqa.itertuples()},
        news=news,
        hotpot=[],
        contrastive=[],
        lookalike=look,
    )
    return inputs, popqa


def test_donor_map_rebuilds_the_lookalike_names() -> None:
    popqa = _popqa()
    donors = cue.donor_map(popqa, 2)
    items = lookalike_items(popqa, 2)
    assert set(donors) == {it.unit for it in items}
    for it in items:
        a, b = donors[it.unit]
        assert it.info["subject"] == f"{a.split()[0]} {b.split()[1]}"
        assert a.split()[0] != b.split()[0]


def test_cue_controls_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    inputs, popqa = _cue_inputs(3)
    seen: list[BaseConfig] = []

    def load(settings: Settings, base: BaseConfig) -> Inputs:
        seen.append(base)
        return inputs

    monkeypatch.setattr(cue, "load_inputs", load)
    monkeypatch.setattr(loaders, "load_popqa", lambda _d: popqa)
    monkeypatch.setattr(dis, "embed", fake_embed)
    settings = Settings(data_dir=tmp_path / "data", output_dir=tmp_path / "out")
    config = cue.Config(bootstrap=20, lookalike_per_relation=3, cutoff_month=CUTOFF)
    summary = cue.run(settings, config)
    assert seen[0].lookalike_per_relation == 3
    assert seen[0].cutoff_month == CUTOFF
    assert (settings.out(cue.NAME) / "summary.json").exists()
    res = summary["results"]
    look = res["lookalike_vs_obscure_real"]
    block = look["known_minus_confidence"]
    # One obscure real person per relation (least popular fifth), three names.
    assert block["n_items"] == 6 + 18
    assert block["n_made_up"] == 18
    assert block["n_clusters"]["relation"] == 6
    assert block["by_item"]["diff"] > 0.3
    assert block["largest_linked_component_items"] >= 1
    for fold in ("text_only_folds_by_subject", "text_only_folds_by_relation"):
        assert set(look[fold]) == {
            "char_ngram",
            "word_ngram",
            "sentence_embedding",
            "name_structure",
            "embedding_and_name_structure",
        }
        assert all(0.0 <= r["auroc"] <= 1.0 for r in look[fold].values())
    news = res["news_dates_removed"]
    assert news["n_items"] == 60
    assert news["n_months"] == 10
    assert (news["n_months_pre"], news["n_months_post"]) == (5, 5)
    by_month = news["text_only_grouped_by_month"]
    assert set(by_month) == {"char_ngram", "word_ngram", "sentence_embedding"}
    # The topic words differ by period, the dates are masked.
    assert by_month["word_ngram"]["auroc"] > 0.9


# --- forward_recalibration --------------------------------------------------------


def test_latest_date() -> None:
    date, has_date = fr.latest_date("By March 5, 2026, or in 2025?")
    assert date == pytest.approx(6 + 2 / 12)
    assert has_date == 1
    date, has_date = fr.latest_date("Before december 2024?")
    assert date == pytest.approx(4 + 11 / 12)
    assert has_date == 1
    assert fr.latest_date("In 2023 only") == (3.0, 1)
    assert fr.latest_date("No date here") == (0.0, 0)


def _news_frame(
    shift: float = 0.3, n: int = 30, seed: int = 0, cutoff: str = CUTOFF
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for m in _months("2023-11", 24):
        post = m >= cutoff
        for j in range(n):
            p = float(rng.uniform(0.55, 0.95))
            month_name = ["January", "March", "June"][j % 3]
            rows.append(
                {
                    "unit": f"n:{m}:{j}",
                    "question": f"Will the vote {j % 4} pass by {month_name} {m[:4]}?",
                    "month": m,
                    "category": "politics",
                    "gold": "yes",
                    "p_known": float(rng.uniform(0.2, 0.8)),
                    "p_max": p,
                    "correct": bool(rng.random() < p - (shift if post else 0.0)),
                    "top": "yes" if j % 2 else "no",
                    "post": post,
                }
            )
    return pd.DataFrame(rows)


def test_features_blocks() -> None:
    df = _news_frame(n=2).head(4).assign(date=1.5, has_date=1)
    assert fr.features(df, ()).shape == (4, 3)
    assert fr.features(df, ("date",)).shape == (4, 5)
    x = fr.features(df, ("date", "known"))
    assert x.shape == (4, 6)
    # Third column is logit(p_max) for a "yes" answer and 0 for "no".
    yes = (df["top"] == "yes").to_numpy()
    np.testing.assert_allclose(x[:, 2], np.where(yes, x[:, 0], 0.0))
    np.testing.assert_allclose(x[:, 1], yes.astype(float))
    lp = fr._logit(np.array([0.0, 0.5, 1.0]))
    assert lp[1] == 0.0
    assert lp[0] == pytest.approx(-lp[2])
    assert np.isfinite(lp).all()


def test_frozen_isotonic_shows_the_post_gap() -> None:
    df = _news_frame()
    fit = fr.make_models(0.5)["isotonic_p_max"]
    res = fr.evaluate(df, fit, 6)
    frozen = res["frozen_before_cutoff"]
    assert 0.2 < frozen["post_gap"] < 0.4
    assert 0.0 < frozen["post_brier"] < 1.0
    assert frozen["post_log_loss"] > 0
    exp = res["expanding"]
    assert abs(exp["pre_mean_gap"]) < 0.1
    # Refitting on later months closes part of the gap, but not all of it.
    assert 0.0 < exp["post_mean_gap"] < frozen["post_gap"]
    assert exp["post_mean_abs_gap"] >= exp["post_mean_gap"]


def test_no_shift_means_no_post_gap() -> None:
    res = fr.evaluate(_news_frame(shift=0.0), fr.make_models(0.5)["logistic_answer"], 6)
    assert abs(res["frozen_before_cutoff"]["post_gap"]) < 0.08


def test_forward_recalibration_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    df = _news_frame(n=20)
    recs = df.to_dict("records")
    news = [
        Item(
            unit=r["unit"],
            state={"question": r["question"]},
            gold=r["gold"],
            info={"month": r["month"], "category": r["category"]},
        )
        for r in recs
    ]
    first = [
        {
            "unit": r["unit"],
            "p_known": r["p_known"],
            "p_max": r["p_max"],
            "correct": r["correct"],
            "top": r["top"],
            "dist": {"yes": r["p_max"] if r["top"] == "yes" else 1 - r["p_max"]},
        }
        for r in recs
    ]
    seen: list[BaseConfig] = []

    def load(settings: Settings, base: BaseConfig) -> Inputs:
        seen.append(base)
        return Inputs(
            rows={"knowledge_boundary": first},
            entities=[],
            subjects={},
            news=news,
            hotpot=[],
            contrastive=[],
            lookalike=[],
        )

    monkeypatch.setattr(fr, "load_inputs", load)
    settings = Settings(data_dir=tmp_path / "data", output_dir=tmp_path / "out")
    config = fr.Config(cutoff_month=CUTOFF, min_train_months=8, yes_no_per_month=20)
    summary = fr.run(settings, config)
    assert seen[0].cutoff_month == CUTOFF
    assert seen[0].yes_no_per_month == 20
    assert (settings.out(fr.NAME) / "summary.json").exists()
    res = summary["results"]
    assert res["n_pre"] == 12 * 20
    assert res["n_post"] == 12 * 20
    assert res["share_with_date"] == 1.0
    assert 0.2 < res["raw_post_gap"] < 0.4
    assert set(res["models"]) == {
        "isotonic_p_max",
        "logistic_answer",
        "logistic_answer_date",
        "logistic_answer_date_text",
        "logistic_answer_date_known",
    }
    for m in res["models"].values():
        frozen = m["frozen_before_cutoff"]
        assert frozen["post_gap"] > 0.1
        assert 0.0 < frozen["post_brier"] < 1.0


# --- distractor_examples ----------------------------------------------------------


def _item(unit: str, descriptions: dict[str, str] | None, **info: Any) -> Item:
    return Item(
        unit=unit,
        state={"question": f"Q {unit}?"},
        options=("A", "B"),
        gold="B",
        descriptions=descriptions,
        info=info,
    )


def test_describe_counts_and_examples() -> None:
    items = [
        _item("a", {"A": "x", "B": "y"}, perturbed=True),
        _item("b", {"A": "u", "B": "v"}, perturbed=False),
        _item("c", {"A": "s", "B": "t"}),
    ]
    out = de.describe(items, 2, random.Random(0))  # noqa: S311
    assert out["n_items"] == 3
    assert out["share_perturbed"] == 0.5
    assert len(out["examples"]) == 2
    for ex in out["examples"]:
        assert ex["question"] == f"Q {ex['unit']}?"
        assert ex["gold"] == ex["options"][1]
    bare = de.describe([_item("d", None)], 5, random.Random(0))  # noqa: S311
    assert "share_perturbed" not in bare
    assert bare["examples"] == [
        {"unit": "d", "question": "Q d?", "options": ["A", "B"], "gold": "B"}
    ]
    assert de.describe([], 3, random.Random(0)) == {  # noqa: S311
        "n_items": 0,
        "examples": [],
    }


def _hotpot() -> pd.DataFrame:
    titles = ["Alpha Town", "Beta City", "Pad One", "Pad Two", "Pad Three"]
    return pd.DataFrame(
        [
            {
                "id": f"q{i}",
                "type": "comparison",
                "question": f"Question {i}?",
                "answer": answer,
                "context": {
                    "title": np.array(titles),
                    "sentences": [np.array([f"{t} s1. ", f"{t} s2."]) for t in titles],
                },
                "supporting_facts": {"title": np.array(["Alpha Town", "Beta City"])},
            }
            for i, answer in enumerate(["yes", "Alpha Town"])
        ]
    )


def _triviaqa() -> pd.DataFrame:
    golds = ["Paris", "Rome", "Oslo", "Lima", "Cairo", "Quito"]
    return pd.DataFrame(
        [
            {
                "question_id": f"t{i}",
                "question": f"Capital {i}?",
                "answer": {
                    "value": v,
                    "aliases": np.array([v.lower()]),
                    "type": "WikipediaEntity",
                },
            }
            for i, v in enumerate(golds)
        ]
    )


def _simpleqa() -> pd.DataFrame:
    answers = [("12 May 1998", "Date"), ("3 June 2001", "Date")] + [
        (n, "Person") for n in ("Ann Lee", "Bob Ray", "Cy Hale", "Di Moss", "Ed Kerr")
    ]
    return pd.DataFrame(
        [
            {
                "original_index": i,
                "problem": f"p{i}?",
                "answer": a,
                "topic": "t",
                "answer_type": t,
            }
            for i, (a, t) in enumerate(answers)
        ]
    )


def _entity_popqa() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id": i,
                "subj": f"Thing{i}",
                "prop": "genre",
                "obj": f"genre value {i % 5}",
                "question": f"What is the genre of Thing{i}?",
                "possible_answers": [f"genre value {i % 5}"],
                "o_aliases": "",
                "s_pop": i,
            }
            for i in range(8)
        ]
    )


def test_distractor_examples_run_reuses_saved_choices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    popqa = _entity_popqa()
    monkeypatch.setattr(loaders, "load_popqa", lambda _d: popqa)
    monkeypatch.setattr(loaders, "load_triviaqa_validation", lambda _d: _triviaqa())
    monkeypatch.setattr(loaders, "load_simpleqa", lambda _d: _simpleqa())
    monkeypatch.setattr(loaders, "load_hotpot_validation", lambda _d: _hotpot())
    monkeypatch.setattr(dis, "embed", fake_embed)
    settings = Settings(data_dir=tmp_path / "data", output_dir=tmp_path / "out")
    config = de.Config(examples=3, seed=1)
    summary = de.run(settings, config)
    saved = list((settings.data_dir / "cache" / "distractors").glob("*.json"))
    assert len(saved) == 3  # TriviaQA and both SimpleQA answer types
    res = summary["results"]
    assert list(res) == ["popqa", "triviaqa", "simpleqa", "hotpot"]
    assert all(res[k]["method"] == de.METHODS[k] for k in res)
    assert res["popqa"]["n_items"] == 8
    assert res["triviaqa"]["n_items"] == 6
    assert res["simpleqa"]["n_items"] == 7
    assert res["simpleqa"]["share_perturbed"] == pytest.approx(2 / 7)
    assert res["hotpot"]["n_items"] == 8  # two questions, four evidence cells
    assert all(len(res[k]["examples"]) == 3 for k in res)
    golds = dict(zip(popqa["question"], popqa["obj"], strict=True))
    for ex in res["popqa"]["examples"]:
        assert len(ex["options"]) == 4
        assert ex["gold"] == golds[ex["question"]]
        assert ex["gold"] in ex["options"]
    for ex in res["triviaqa"]["examples"] + res["simpleqa"]["examples"]:
        assert len(set(ex["options"])) == 4
        assert ex["gold"] in ex["options"]
    for ex in res["hotpot"]["examples"]:
        assert ex["gold"] in ex["options"]
    assert (settings.out(de.NAME) / "summary.json").exists()

    def broken(*_a: object, **_k: object) -> np.ndarray:
        raise AssertionError("saved distractor choices must be reused")

    monkeypatch.setattr(dis, "embed", broken)
    assert de.run(settings, config) == summary
