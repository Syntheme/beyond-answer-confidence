"""Level merging, out-of-fold baselines and the shortcut-baseline steps."""

import itertools
import math
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from beyond_answer_confidence.analyses import level_merging as lm
from beyond_answer_confidence.analyses import shortcut_baselines as sb
from beyond_answer_confidence.metrics.distributions import expected_scores
from beyond_answer_confidence.metrics.ranking import auroc
from beyond_answer_confidence.stats.crossfit import (
    grouped_oof,
    oof_features,
    oof_text_scores,
)
from beyond_answer_confidence.stats.resampling import auroc_interval


def _news(shift: float, seed: int = 0) -> pd.DataFrame:
    """Quantised p_max; accuracy after the cutoff lower by ``shift``."""
    rng = np.random.default_rng(seed)
    months = [f"{y}-{m:02d}" for y in (2023, 2024, 2025) for m in range(1, 13)]
    rows = []
    for mth in months:
        post = mth >= "2024-11"
        for _ in range(60):
            p = round(float(rng.integers(50, 100)) / 100, 1)
            rows.append(
                {
                    "month": mth,
                    "post": post,
                    "p_max": p,
                    "correct": bool(rng.random() < p - (shift if post else 0.0)),
                    "gold": "yes",
                    "top": "no" if post else "yes",
                    "category": "c",
                }
            )
    return pd.DataFrame(rows)


def test_group_cost_is_min_over_output_value() -> None:
    grid = np.linspace(0, 1, 10001)
    brute = min(10 * abs(c - 0.6) + 4 * abs(c - 0.25) for c in grid)
    assert lm.group_cost(10, 6, 4, 1) == pytest.approx(brute, abs=1e-3)
    assert lm.group_cost(0, 0, 5, 2) == 0.0


def _brute_interval(stats: np.ndarray) -> float:
    m = len(stats)
    best = math.inf
    for cuts in itertools.product([0, 1], repeat=m - 1):
        bounds = [0, *[i + 1 for i, c in enumerate(cuts) if c], m]
        cost = sum(
            lm.group_cost(*stats[a:b].sum(axis=0))
            for a, b in itertools.pairwise(bounds)
        )
        best = min(best, cost)
    return best


def test_interval_partition_matches_brute_force() -> None:
    rng = np.random.default_rng(1)
    for _ in range(20):
        m = 7
        n_pre = rng.integers(1, 20, m).astype(float)
        n_post = rng.integers(1, 20, m).astype(float)
        k_pre = np.floor(n_pre * rng.random(m))
        k_post = np.floor(n_post * rng.random(m))
        stats = np.column_stack([n_pre, k_pre, n_post, k_post])
        exact, groups = lm.best_interval_partition(n_pre, k_pre, n_post, k_post)
        assert exact == pytest.approx(_brute_interval(stats))
        assert groups[0][0] == 0
        assert groups[-1][1] == m
        assert exact <= sum(lm.group_cost(*row) for row in stats) + 1e-9


def test_grouping_search_is_consistent() -> None:
    rng = np.random.default_rng(2)
    n_pre = rng.integers(5, 30, 12).astype(float)
    n_post = rng.integers(5, 30, 12).astype(float)
    k_pre = np.floor(n_pre * 0.8)
    k_post = np.floor(n_post * 0.4)
    err, label = lm.search_groupings(n_pre, k_pre, n_post, k_post, 3, rng)
    stats = np.column_stack([n_pre, k_pre, n_post, k_post])
    recomputed = sum(lm.group_cost(*stats[label == g].sum(axis=0)) for g in range(3))
    assert err == pytest.approx(recomputed)
    assert err >= 0


def test_level_counts() -> None:
    values, n_pre, k_pre, n_post, k_post = lm.level_counts(
        np.array([0.5, 0.5, 0.9, 0.9 + 1e-12]),
        np.array([True, False, True, True]),
        np.array([False, True, False, True]),
    )
    assert values.tolist() == [0.5, 0.9]  # float noise merged
    assert n_pre.tolist() == [1, 1]
    assert k_pre.tolist() == [1, 1]
    assert n_post.tolist() == [1, 1]
    assert k_post.tolist() == [0, 1]


def test_apply_levels_scores_groups_without_training_data() -> None:
    label = np.array([0, 1])
    train = np.array([[10.0, 8.0, 10.0, 6.0], [0.0, 0.0, 0.0, 0.0]])
    test = np.array([[0.0, 0.0, 0.0, 0.0], [4.0, 4.0, 0.0, 0.0]])
    # group 1 has no training data: output = overall training accuracy 14/20
    assert lm.apply_levels(label, train, test) == pytest.approx(4 * abs(0.7 - 1.0))


def test_recalibration_groupings_detects_a_shift() -> None:
    shifted = lm.recalibration_groupings(_news(0.3), np.random.default_rng(0), 3, 10)
    same = lm.recalibration_groupings(_news(0.0), np.random.default_rng(0), 3, 10)
    assert (
        shifted["cross_fitted"]["interval"]["mean"]
        > same["cross_fitted"]["interval"]["mean"]
    )
    assert shifted["cross_fitted_interval_post_gap"]["mean"] > 0.15
    ins = shifted["in_sample"]
    assert ins["interval_partition_exact"] <= ins["valuewise_bound"] + 1e-12
    assert 0 < ins["interval_null_random_months"]["p"] <= 1


def test_forward_chaining_frozen_map_misses_the_shift() -> None:
    res = lm.forward_chaining(_news(0.3), cutoff_month="2024-11")
    assert res["frozen_before_cutoff"]["post_mean_gap"] > 0.2
    assert abs(res["expanding"]["pre_mean_gap"]) < 0.1
    assert len(res["by_month"]) == 36 - 12


def test_label_balance() -> None:
    res = sb.label_balance(_news(0.3), "2024-11")
    assert res["by_period"]["post"]["predicted_no"] == 1.0
    assert res["by_period"]["pre"]["true_no"] == 0.0
    assert math.isnan(res["by_period"]["pre"]["accuracy_when_predicting_no"])
    assert set(res["true_no_by_month_range"]) == {"pre", "post"}
    assert res["category_share_tv_pre_post"] == 0.0


def test_oof_text_scores_find_a_surface_cue() -> None:
    texts = [f"plain name {i}" for i in range(40)] + [
        f"zqxv name {i}" for i in range(40)
    ]
    y = np.r_[np.zeros(40, bool), np.ones(40, bool)]
    assert auroc(oof_text_scores(texts, y), y) > 0.95


def test_oof_features_use_the_informative_feature() -> None:
    rng = np.random.default_rng(0)
    y = rng.random(200) < 0.5
    x = np.column_stack([y + rng.normal(0, 0.5, 200), rng.normal(size=200)])
    assert auroc(oof_features(x, y), y) > 0.8


def test_grouped_oof_keeps_groups_apart() -> None:
    # Each group has one label and the features only identify the group; with
    # grouped folds a model cannot memorise groups.
    rng = np.random.default_rng(1)
    groups = np.repeat(np.arange(40), 5)
    y = np.repeat(rng.random(40) < 0.5, 5)
    x = np.eye(40)[groups]
    scores = grouped_oof(lambda: LogisticRegression(max_iter=500), x, y, groups)
    assert auroc(scores, y) < 0.8


def test_auroc_interval_uses_exact_percentiles() -> None:
    rng = np.random.default_rng(3)
    y = rng.random(300) < 0.4
    s = y + rng.normal(0, 1, 300)
    res = auroc_interval(s, y, 200, np.random.default_rng(7))
    draws = np.random.default_rng(7)
    boot = [
        auroc(s[i], y[i]) for i in (draws.integers(0, 300, 300) for _ in range(200))
    ]
    assert res["ci"] == [
        float(np.nanpercentile(boot, 2.5)),
        float(np.nanpercentile(boot, 97.5)),
    ]
    assert set(res) == {"auroc", "ci", "n", "n_positive"}


def test_expected_scores_are_minimised_by_the_stated_odds() -> None:
    ideal = {"a": 0.5, "b": 0.3, "c": 0.2}
    own = expected_scores(ideal, ideal, sb.EPS)
    assert own["brier"] == pytest.approx(own["ideal_brier"])
    assert own["log_loss"] == pytest.approx(own["ideal_log_loss"])
    mode = expected_scores({"a": 1.0, "b": 0.0, "c": 0.0}, ideal, sb.EPS)
    assert mode["brier"] > own["brier"]
    assert mode["log_loss"] > own["log_loss"]


def test_word_count_walks_nested_states() -> None:
    state = {"question": "two words", "paragraphs": ["a b c", ("d",)], "n": 3}
    assert sb.word_count(state) == 6


def test_proper_scores_and_floors() -> None:
    rows: list[dict[str, Any]] = [
        {"set": "s", "gold": "A", "dist": {"A": 1.0, "B": 0.0}, "correct": True},
        {"set": "s", "gold": "B", "dist": {"A": 1.0, "B": 0.0}, "correct": False},
        {"set": "s", "gold": None, "dist": {"A": 1.0}},
    ]
    tables: dict[str, list[dict[str, Any]]] = {exp: [] for exp in sb.PROPER_SETS}
    tables["knowledge_boundary"] = rows
    res = sb.proper_scores(tables, [], [{"set": "relist"}])
    s = res["sets"]["knowledge_boundary:s"]
    assert s["n"] == 2
    assert s["brier"] == pytest.approx(1.0)  # (0 + 2) / 2
    assert s["log_loss"] == pytest.approx(-math.log(sb.EPS) / 2)
    assert s["zero_on_answer"] == 0.5
    assert s["uniform_brier"] == 0.5
    floors = sb.floor_sensitivity(tables)["sets"]["knowledge_boundary:s"]
    assert floors["log_loss_floor_0.01"] == pytest.approx(-math.log(0.01) / 2)


def test_coverage_table_counts() -> None:
    rows = [
        {
            "cond": "unknown",
            "post": True,
            "top": "unknown",
            "correct": False,
            "p_max": 0.95,
        },
        {"cond": "unknown", "post": True, "top": "no", "correct": False, "p_max": 0.95},
        {"cond": "unknown", "post": True, "top": "yes", "correct": True, "p_max": 0.6},
        {"cond": "unknown", "post": True, "top": "no", "correct": True, "p_max": 0.9},
    ]
    t = sb.coverage_table(rows)["unknown|post"]
    assert t["coverage"] == 0.75
    assert t["accuracy_answered"] == pytest.approx(2 / 3)
    assert t["confident_errors_of_all"] == 0.25
    assert t["confident_errors_of_answered"] == pytest.approx(1 / 3)


def test_enough_decisions_prefers_the_informative_score() -> None:
    rng = np.random.default_rng(0)
    rows = []
    for q in range(60):
        for cell in ("closed", "dose0", "dose1", "dose2"):
            ok = rng.random() < (0.9 if cell == "dose2" else 0.4)
            rows.append(
                {
                    "set": "hotpot",
                    "qid": q,
                    "p_max": float(rng.random()),
                    "p_enough": float(ok) * 0.5 + 0.5 * float(rng.random()),
                    "correct": ok,
                }
            )
    res = sb.enough_decisions(rows, np.random.default_rng(1), splits=5)
    assert res["enough_minus_confidence"]["0.25"]["mean"] > 0.2
    assert res["splits"] == 5
