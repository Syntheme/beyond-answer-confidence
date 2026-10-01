import numpy as np
import pandas as pd
import pytest

from beyond_answer_confidence.stats import recalibration as rc
from beyond_answer_confidence.stats import smece_bounds as sb


def _calibrated(
    n: int, seed: int = 0, shift: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.3, 1.0, n)
    y = (rng.random(n) < np.clip(p - shift, 0, 1)).astype(float)
    return p, y


def _quantised(shift: float, seed: int = 0) -> pd.DataFrame:
    """Quantised p_max; later-period accuracy lower by ``shift``."""
    rng = np.random.default_rng(seed)
    months = [f"{y}-{m:02d}" for y in (2023, 2024, 2025) for m in range(1, 13)]
    rows = []
    for mth in months:
        post = mth >= "2024-11"
        for _ in range(50):
            p = round(float(rng.integers(50, 100)) / 100, 2)
            rows.append(
                {
                    "month": mth,
                    "post": post,
                    "p_max": p,
                    "correct": bool(rng.random() < p - (shift if post else 0.0)),
                }
            )
    return pd.DataFrame(rows)


# --- SmoothECE bounds -----------------------------------------------------------


def test_smece_bounds_relations() -> None:
    est, n = 0.03, 1000
    boot = np.linspace(0.03, 0.05, 101)  # biased upward by 0.01
    sub = np.linspace(0.025, 0.045, 101)
    b = sb.smece_bounds(est, n, boot, sub)
    assert b["recentred"]["upper"] == pytest.approx(b["percentile"]["upper"] - 0.01)
    assert b["basic"]["upper"] == pytest.approx(2 * est - b["percentile"]["low"])
    for v in b.values():
        assert v["low"] <= v["high"]
    assert sb.smece_bounds(est, n, None, sub).keys() == {"subsample"}
    assert sb.smece_bounds(est, n, boot, None).keys() == {
        "percentile",
        "recentred",
        "basic",
    }


def test_subsample_bound_known_answer() -> None:
    # Half-samples all equal to the estimate give a degenerate interval.
    b = sb.smece_bounds(0.04, 400, None, np.full(50, 0.04))
    assert b["subsample"] == pytest.approx({"upper": 0.04, "low": 0.04, "high": 0.04})


def test_tail_shares_agree_with_bounds() -> None:
    p, y = _calibrated(600)
    chk = sb.smece_with_bounds(p, y, 0.08, 200, np.random.default_rng(1))
    assert chk["n"] == 600
    assert chk["reference"] == 0.08
    for m in sb.SMECE_METHODS:
        below = chk["bounds"][m]["upper"] < 0.08
        assert (chk["tail"][m] < 0.05) == below or abs(
            chk["bounds"][m]["upper"] - 0.08
        ) < 0.005


def test_smece_with_bounds_draw_order() -> None:
    from beyond_answer_confidence.metrics.calibration import smece
    from beyond_answer_confidence.stats.resampling import smece_resamples

    p, y = _calibrated(200, seed=3)
    got = sb.smece_with_bounds(p, y, 0.05, 20, np.random.default_rng(7))
    rng = np.random.default_rng(7)
    boot = smece_resamples(p, y, 20, rng, subsample=False)
    sub = smece_resamples(p, y, 20, rng, subsample=True)
    est = smece(p, y)
    assert got["bounds"] == sb.smece_bounds(est, 200, boot, sub)
    assert got["bootstrap_bias"] == pytest.approx(boot.mean() - est)


# --- scalar maps ---------------------------------------------------------------


def test_temperature_softens_overconfident_distributions() -> None:
    rng = np.random.default_rng(0)
    dists, golds = [], []
    for _ in range(400):
        top = "a" if rng.random() < 0.6 else "b"
        dists.append(
            {"a": 0.95 if top == "a" else 0.05, "b": 0.05 if top == "a" else 0.95}
        )
        golds.append("a")
    assert rc.fit_temperature(dists, golds) > 1.5


def test_temper_known_answer() -> None:
    out = rc.temper([{"a": 0.8, "b": 0.2}], 1.0)
    assert out[0] == pytest.approx({"a": 0.8, "b": 0.2})
    flat = rc.temper([{"a": 0.8, "b": 0.2}], 1e6)
    assert flat[0]["a"] == pytest.approx(0.5, abs=1e-4)


def test_platt_recovers_accuracy_level() -> None:
    rng = np.random.default_rng(1)
    pm = np.full(2000, 0.9)
    ok = rng.random(2000) < 0.7
    ab = rc.fit_platt(pm, ok)
    assert rc.apply_platt(pm, ab).mean() == pytest.approx(0.7, abs=0.02)
    assert rc.apply_platt(np.array([0.5]), (1.0, 0.0))[0] == pytest.approx(0.5)


def test_logistic_map_uses_both_scores() -> None:
    rng = np.random.default_rng(2)
    n = 3000
    pm = rng.uniform(0.5, 1, n)
    pk = rng.uniform(0, 1, n)
    ok = rng.random(n) < np.clip(pk, 0.05, 0.95)
    model = rc.fit_logistic_map([pm, pk], ok)
    assert model.coef_[0][1] > 0.5  # P(known) carries the signal
    pred = rc.predict_logistic_map(model, [pm, pk])
    assert pred.mean() == pytest.approx(ok.mean(), abs=0.02)


def test_split_half_is_disjoint_and_seeded() -> None:
    df = pd.DataFrame({"unit": [f"u{i}" for i in range(10)], "x": range(10)})
    fit, test = rc.split_half(df, "unit", 0)
    assert len(fit) == 5
    assert set(fit["unit"]).isdisjoint(test["unit"])
    again, _ = rc.split_half(df, "unit", 0)
    assert list(again["unit"]) == list(fit["unit"])


def test_cv_logistic_scores_are_out_of_fold() -> None:
    rng = np.random.default_rng(3)
    x = rng.normal(size=(300, 1))
    y = (x[:, 0] + rng.normal(0, 0.5, 300) > 0).astype(int)
    groups = np.repeat(np.arange(30), 10)
    pred = rc.cv_logistic_scores(x, y, groups)
    assert np.corrcoef(pred, x[:, 0])[0, 1] > 0.9
    # A fold whose training labels are all equal predicts their mean.
    const = rc.cv_logistic_scores(x, np.zeros(300, int), groups)
    assert np.all(const == 0)


# --- every function of one score --------------------------------------------------


def test_lower_bound_is_exact_on_a_toy_table() -> None:
    levels = pd.DataFrame(
        {
            "v": [0.6, 0.9],
            "n_pre": [10, 4],
            "n_post": [2, 8],
            "a_pre": [0.6, 1.0],
            "a_post": [0.1, 0.5],
        }
    )
    # min(10, 2) * 0.5 + min(4, 8) * 0.5 = 3, over 24 questions
    assert rc.lower_bound(levels) == pytest.approx(3 / 24)


def test_lower_bound_holds_for_arbitrary_functions() -> None:
    d = _quantised(0.25)
    levels = rc.level_table(d)
    bound = rc.lower_bound(levels)
    rng = np.random.default_rng(1)
    for _ in range(20):
        f = dict(zip(levels["v"], rng.uniform(0, 1, len(levels)), strict=True))
        mapped = d["p_max"].round(9).map(f)
        per = (
            d.assign(m=mapped, ok=d["correct"].astype(float))
            .groupby(["p_max", "post"])
            .agg(n=("ok", "size"), a=("ok", "mean"), m=("m", "first"))
        )
        total = float((per["n"] * (per["m"] - per["a"]).abs()).sum())
        assert total / len(d) >= bound - 1e-12


def test_per_value_map_gaps_known_answer() -> None:
    d = pd.DataFrame(
        {
            "p_max": [0.7, 0.7, 0.7, 0.7],
            "correct": [True, True, False, False],
            "post": [False, False, True, True],
        }
    )
    gaps = rc.per_value_map_gaps(d)
    assert gaps == pytest.approx({"mean_gap_pre": -0.5, "mean_gap_post": 0.5})


def test_permutation_nulls_centre_below_a_real_shift() -> None:
    d = _quantised(0.25)
    obs = rc.lower_bound(rc.level_table(d))
    rng = np.random.default_rng(2)
    within = rc.within_value_permutation_null(d, 20, rng)
    months = rc.random_months_null(d, 20, rng)
    assert within.shape == months.shape == (20,)
    assert obs > within.max()
    assert obs > np.percentile(months, 50)


def test_cross_fitted_maps() -> None:
    d = _quantised(0.25)
    cf = rc.cross_fitted(d, 3, np.random.default_rng(4))
    assert cf["splits"] == 6
    assert cf["common_post"]["mean"] > 0.1
    assert abs(cf["aware_post"]["mean"]) < 0.05
    common = rc.fit_common_map(d)
    # Values seen at fitting time get their pooled accuracy.
    v = float(d["p_max"].iloc[0])
    assert common(np.array([v]))[0] == pytest.approx(
        d.loc[d["p_max"] == v, "correct"].mean()
    )
    # Unseen values fall back to isotonic regression.
    assert 0 <= common(np.array([0.123]))[0] <= 1
    aware = rc.fit_period_aware_map(d)
    out = aware(np.array([0.9, 0.9]), np.array([False, True]))
    assert out[0] > out[1]
