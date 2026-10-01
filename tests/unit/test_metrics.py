import math
from typing import Any

import numpy as np
import pytest

from beyond_answer_confidence.metrics import calibration, distributions, ranking


def test_normalise_rescales_to_one() -> None:
    assert distributions.normalise({"a": 1.0, "b": 3.0}) == {"a": 0.25, "b": 0.75}


@pytest.mark.parametrize("bad", [{"a": -0.1, "b": 1.1}, {"a": 0.0, "b": 0.0}])
def test_normalise_rejects_invalid(bad: dict[str, float]) -> None:
    with pytest.raises(ValueError, match="probabilities"):
        distributions.normalise(bad)


def test_entropy_uniform_and_point_mass() -> None:
    assert distributions.entropy([0.25] * 4) == pytest.approx(math.log(4))
    assert distributions.entropy([1.0, 0.0, 0.0]) == 0.0
    assert distributions.normalised_entropy([0.25] * 4) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="two options"):
        distributions.normalised_entropy([1.0])


def test_nll_and_brier() -> None:
    p = {"a": 0.8, "b": 0.2}
    assert distributions.nll(p, "a") == pytest.approx(-math.log(0.8))
    assert distributions.nll({"a": 1.0, "b": 0.0}, "b") == pytest.approx(
        -math.log(distributions.EPS)
    )
    assert distributions.nll(p, "missing") == pytest.approx(
        -math.log(distributions.EPS)
    )
    assert distributions.brier(p, "a") == pytest.approx(0.08)
    assert distributions.brier({"a": 1.0, "b": 0.0}, "b") == pytest.approx(2.0)


def test_choice_confidence() -> None:
    # 3 options at 90/6/4 -> (3 * 0.9 - 1) / 2 = 0.85.
    assert distributions.choice_confidence([0.9, 0.06, 0.04]) == pytest.approx(0.85)
    assert distributions.choice_confidence([0.5, 0.5]) == 0.0
    with pytest.raises(ValueError, match="two options"):
        distributions.choice_confidence([1.0])


def test_tv_and_pairwise() -> None:
    assert distributions.tv({"a": 1.0}, {"b": 1.0}) == 1.0
    assert distributions.tv({"a": 0.5, "b": 0.5}, {"a": 0.5, "b": 0.5}) == 0.0
    dists = [{"a": 1.0, "b": 0.0}, {"a": 0.0, "b": 1.0}, {"a": 1.0, "b": 0.0}]
    assert distributions.mean_pairwise_tv(dists) == pytest.approx(2 / 3)
    assert distributions.mean_pairwise_tv(dists[:1]) == 0.0


def test_js_divergence_bounds() -> None:
    p = {"a": 1.0, "b": 0.0}
    q = {"a": 0.0, "b": 1.0}
    assert distributions.js_divergence(p, p) == 0.0
    assert distributions.js_divergence(p, q) == pytest.approx(math.log(2))
    with pytest.raises(ValueError, match="same options"):
        distributions.js_divergence(p, {"a": 1.0})


def test_logit_clips() -> None:
    out = distributions.logit(np.array([0.5, 0.0, 1.0]))
    assert out[0] == 0.0
    assert out[1] == pytest.approx(math.log(1e-3 / (1 - 1e-3)))
    assert out[2] == pytest.approx(-out[1])


def test_expected_scores_perfect_prediction_hits_floor() -> None:
    ideal = {"a": 0.5, "b": 0.5}
    res = distributions.expected_scores(ideal, ideal)
    assert res["brier"] == pytest.approx(res["ideal_brier"]) == pytest.approx(0.5)
    assert res["log_loss"] == pytest.approx(res["ideal_log_loss"])
    assert res["ideal_log_loss"] == pytest.approx(math.log(2))


def test_smece_perfect_and_bad() -> None:
    rng = np.random.default_rng(0)
    p = rng.uniform(0.3, 1.0, 4000)
    y = rng.uniform(size=4000) < p
    assert calibration.smece(p, y) < 0.03
    assert calibration.smece(np.full(1000, 0.99), np.zeros(1000)) > 0.9


def test_ece_known_answer() -> None:
    conf = np.array([0.95, 0.95, 0.55, 0.55])
    ok = np.array([1, 1, 1, 0])
    # bin at 0.95: |1 - 0.95| * 0.5, bin at 0.55: |0.5 - 0.55| * 0.5
    assert calibration.ece(conf, ok, n_bins=10) == pytest.approx(0.05)


def test_reliability_bins_and_merge() -> None:
    conf = np.array([0.2, 0.4, 1.0, 1.0, 1.0, 1.0])
    ok = np.array([0, 1, 1, 1, 0, 1], bool)
    bins = calibration.reliability_bins(conf, ok, n_bins=3)
    assert [b["n"] for b in bins] == [2, 2, 2]
    merged = calibration.merge_tied_bins(bins)
    assert len(merged) == 2
    assert merged[1] == {"confidence": 1.0, "accuracy": 0.75, "n": 4.0}


def test_wilson_and_normal_intervals() -> None:
    lo, hi = calibration.wilson_interval(0.5, 100)
    assert lo == pytest.approx(0.4038, abs=1e-4)
    assert hi == pytest.approx(0.5962, abs=1e-4)
    assert all(math.isnan(v) for v in calibration.wilson_interval(0.5, 0))
    assert calibration.normal_mean_interval(np.array([2.0])) == (2.0, 2.0)
    lo, hi = calibration.normal_mean_interval(np.array([1.0, 3.0]))
    assert (lo + hi) / 2 == pytest.approx(2.0)
    assert hi - lo == pytest.approx(2 * 1.96 * 1.0)


def test_auroc_known_answers() -> None:
    y = np.array([True, True, False, False])
    assert ranking.auroc(np.array([4, 3, 2, 1.0]), y) == 1.0
    assert ranking.auroc(np.array([1, 2, 3, 4.0]), y) == 0.0
    assert ranking.auroc(np.ones(4), y) == 0.5
    assert math.isnan(ranking.auroc(np.ones(3), np.ones(3, bool)))


def test_stratified_auroc_skips_single_class_strata() -> None:
    s = np.array([2, 1, 2, 1, 5.0])
    y = np.array([1, 0, 0, 1, 1], bool)
    g = np.array(["a", "a", "b", "b", "c"])
    res = ranking.stratified_auroc(s, y, g)
    assert res["auroc"] == 0.5
    assert (res["strata_used"], res["rows_covered"], res["rows"]) == (2, 4, 5)


def test_fpr_at_tpr_and_average_precision() -> None:
    s = np.array([0.9, 0.8, 0.7, 0.1])
    y = np.array([True, False, True, False])
    assert ranking.fpr_at_tpr(s, y, tpr=1.0) == 0.5
    assert ranking.average_precision(s, y) == pytest.approx((1 + 2 / 3) / 2)
    assert math.isnan(ranking.average_precision(s, np.zeros(4, bool)))


def test_aurc_and_risk_coverage() -> None:
    score = np.array([0.9, 0.8, 0.7, 0.6])
    ok = np.array([True, True, False, False])
    assert ranking.aurc(score, ok) == pytest.approx((0 + 0 + 1 / 3 + 1 / 2) / 4)
    rc = ranking.risk_coverage(score, ok, coverages=(0.5, 1.0))
    assert rc["aurc"] == pytest.approx(ranking.aurc(score, ok))
    assert rc["risk@50"] == 0.0
    assert rc["risk@100"] == 0.5
    assert rc["augrc"] == pytest.approx((0 + 0 + 1 / 4 + 2 / 4) / 4)


def test_selective_at_counts_confident_errors() -> None:
    score = np.array([0.9, 0.8, 0.7, 0.6])
    ok = np.array([True, False, False, True])
    res = ranking.selective_at(
        score, ok, 0.5, confidence=np.array([0.95, 0.92, 0.5, 0.5])
    )
    assert res == {
        "answered": 2,
        "coverage": 0.5,
        "selective_accuracy": 0.5,
        "errors": 1,
        "confident_errors": 1,
    }


def test_order_metrics_break_ties_by_input_order_by_default() -> None:
    score = np.array([0.9, 0.5, 0.5, 0.1])
    ok = np.array([True, False, True, True])
    swapped = np.array([True, True, False, True])  # tied pair in the other order
    assert ranking.aurc(score, ok) != ranking.aurc(score, swapped)
    assert ranking.risk_coverage(score, ok) != ranking.risk_coverage(score, swapped)
    err, err_swapped = ~ok, ~swapped
    assert ranking.average_precision(score, err) != ranking.average_precision(
        score, err_swapped
    )


def test_average_ties_are_order_independent() -> None:
    score = np.array([0.9, 0.5, 0.5, 0.1])
    ok = np.array([True, False, True, True])
    swapped = np.array([True, True, False, True])
    a = ranking.aurc(score, ok, ties="average")
    assert a == ranking.aurc(score, swapped, ties="average")
    # expectation over the two orders of the tied pair
    assert a == pytest.approx(
        (ranking.aurc(score, ok) + ranking.aurc(score, swapped)) / 2
    )
    rc = ranking.risk_coverage(score, ok, coverages=(0.5,), ties="average")
    assert rc == ranking.risk_coverage(score, swapped, coverages=(0.5,), ties="average")
    assert rc["aurc"] == pytest.approx(a)
    assert rc["risk@50"] == pytest.approx(0.25)
    # without ties the rule makes no difference
    distinct = np.array([0.9, 0.6, 0.5, 0.1])
    assert ranking.aurc(distinct, ok, ties="average") == ranking.aurc(distinct, ok)


def test_average_precision_average_ties_matches_step_definition() -> None:
    from sklearn.metrics import average_precision_score

    rng = np.random.default_rng(0)
    s = rng.integers(0, 4, 50).astype(float)
    y = rng.random(50) < 0.4
    got = ranking.average_precision(s, y, ties="average")
    assert got == pytest.approx(average_precision_score(y, s))
    distinct = rng.random(50)
    assert ranking.average_precision(distinct, y, ties="average") == pytest.approx(
        ranking.average_precision(distinct, y)
    )


def test_ties_argument_is_validated() -> None:
    with pytest.raises(ValueError, match="ties"):
        ranking.aurc(np.ones(2), np.ones(2, bool), ties="random")  # type: ignore[arg-type]


def test_correctness_is_coerced_to_bool() -> None:
    score = np.array([0.9, 0.7, 0.4, 0.2])
    ok_bool = np.array([True, False, True, False])
    variants: list[Any] = [
        [1, 0, 1, 0],
        np.array([1, 0, 1, 0]),
        np.array([1.0, 0.0, 1.0, 0.0]),
    ]
    for ok in variants:
        assert ranking.risk_coverage(score, ok) == ranking.risk_coverage(score, ok_bool)
        assert ranking.aurc(score, ok) == ranking.aurc(score, ok_bool)
        assert ranking.selective_at(
            score, ok, 0.5, confidence=score
        ) == ranking.selective_at(score, ok_bool, 0.5, confidence=score)


def test_fpr_at_tpr_without_one_class_is_nan() -> None:
    s = np.array([0.1, 0.2, 0.3])
    assert math.isnan(ranking.fpr_at_tpr(s, np.zeros(3, bool)))
    assert math.isnan(ranking.fpr_at_tpr(s, np.ones(3, bool)))
    labels: Any = [1, 0, 0]
    assert ranking.fpr_at_tpr(np.array([0.9, 0.1, 0.5]), labels, tpr=1.0) == 0.0


def test_weighted_accuracy_at_splits_ties() -> None:
    score = np.array([1.0, 0.5, 0.5])
    ok = np.array([1, 1, 0])
    w = np.ones(3)
    acc, cov = ranking.weighted_accuracy_at(score, ok, w, 2 / 3)
    assert cov == pytest.approx(2 / 3)
    assert acc == pytest.approx(0.75)  # 1 + half of each tied item
    assert math.isnan(ranking.weighted_accuracy_at(score, ok, np.zeros(3), 0.5)[0])
