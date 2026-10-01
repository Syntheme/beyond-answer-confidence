from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from beyond_answer_confidence.analyses import abstention as ab
from beyond_answer_confidence.analyses import cutoff_calibration as cc
from beyond_answer_confidence.analyses import knowledge_cutoff as kc
from beyond_answer_confidence.analyses import recalibration as rc
from beyond_answer_confidence.metrics.ranking import weighted_accuracy_at
from beyond_answer_confidence.stats.resampling import month_block_indices

MONTHS = [f"{y}-{m:02d}" for y in (2023, 2024, 2025) for m in range(1, 13)]


def _news(
    shift: float, seed: int = 0, n: int = 60, quantised: bool = False
) -> pd.DataFrame:
    """Equal confidence on both sides; later accuracy lower by ``shift``."""
    rng = np.random.default_rng(seed)
    rows = []
    for mth in MONTHS:
        post = mth >= cc.CUTOFF_MONTH
        for _ in range(n):
            p = (
                round(float(rng.integers(50, 100)) / 100, 2)
                if quantised
                else float(rng.uniform(0.5, 1.0))
            )
            rows.append(
                {
                    "month": mth,
                    "post": post,
                    "p_max": p,
                    "correct": bool(rng.random() < p - (shift if post else 0.0)),
                }
            )
    return pd.DataFrame(rows)


def _cutoff_rows(n: int = 200, prefix: str = "cutoff") -> pd.DataFrame:
    rng = np.random.default_rng(2)
    rows: list[dict[str, Any]] = []
    for i in range(n):
        post = i % 2 == 0
        gold = "yes" if i % 3 else "no"
        wrong = "no" if gold == "yes" else "yes"
        ok = rng.random() < (0.5 if post else 0.8)
        p_unk = rng.uniform(0.6, 1) if post else rng.uniform(0, 0.4)
        month = (
            f"2025-{(i // 2) % 12 + 1:02d}" if post else f"2023-{(i // 2) % 12 + 1:02d}"
        )
        common = {
            "post": post,
            "gold": gold,
            "p_known": float(rng.uniform(0, 0.3) if post else rng.uniform(0.3, 0.7)),
            "month": month,
        }
        rows.append(
            {
                **common,
                "unit": f"{prefix}:base:oracle_tf:{i}",
                "cond": "base",
                "top": gold if ok else wrong,
                "p_max": float(rng.uniform(0.6, 1)),
                "dist": {},
            }
        )
        rest = (1 - p_unk) / 2
        dist = {
            "yes": rest + (0.01 if gold == "yes" else 0),
            "no": rest,
            "unknown": p_unk,
        }
        rows.append(
            {
                **common,
                "unit": f"{prefix}:unknown:oracle_tf:{i}",
                "cond": "unknown",
                "top": "unknown" if p_unk > 0.5 else gold,
                "p_max": max(dist.values()),
                "dist": dist,
            }
        )
    return pd.DataFrame(rows)


# --- accuracy at equal confidence ----------------------------------------------


def test_public_constants() -> None:
    assert cc.CUTOFF_MONTH == "2024-11"
    assert len(cc.COARSE) == 6
    assert cc.FINE[0] == 0.5
    assert cc.FINE[-1] == 1.0001
    assert len(cc.FINE) == 21
    assert len(cc.SHARES) == 21
    assert cc.PANEL_SHARE in cc.REPORT_SHARES
    assert cc.BLOCK_MONTHS == 6


def test_equal_confidence_detects_a_gap_and_no_gap() -> None:
    rng = np.random.default_rng(1)
    res = cc.equal_confidence(_news(0.3), 40, rng)
    assert len(res["bands"]) == len(cc.COARSE) - 1
    assert all(b["difference"] > 0.15 for b in res["bands"])
    assert all(b["difference_ci95"][0] > 0 for b in res["bands"])
    assert res["best_single_map"]["coarse_bands"]["mean_gap_post"] > 0.1
    assert res["clusters"] == {"pre_months": 22, "post_months": 14}
    none = cc.equal_confidence(_news(0.0), 40, rng)
    assert abs(none["best_single_map"]["fine_bands"]["mean_gap_post"]) < 0.05


def test_best_single_map_is_perfect_without_a_shift() -> None:
    d = pd.DataFrame(
        {
            "p_max": [0.55, 0.55, 0.95, 0.95],
            "correct": [True, False, True, True],
            "post": [False, True, False, True],
        }
    )
    gaps = cc.best_single_map_gaps(d, cc.COARSE)
    assert gaps["mean_gap_pre"] == pytest.approx(-0.25)
    assert gaps["mean_gap_post"] == pytest.approx(0.25)
    assert gaps["weighted_abs_band_gap_post"] == pytest.approx(0.25)


def test_band_table_counts() -> None:
    t = cc.band_table(_news(0.0, n=10), cc.COARSE)
    assert len(t) == 5
    assert int(t["n_pre"].sum() + t["n_post"].sum()) == 360
    assert {"acc_pre", "acc_post", "conf_pre", "conf_post"} <= set(t.columns)


# --- prevalence sensitivity ------------------------------------------------------


def test_prevalence_reweighting() -> None:
    x = cc.paired_rows(_cutoff_rows())
    assert len(x["post"]) == 200
    only_pre = cc.at_share(x, 0.0)
    only_post = cc.at_share(x, 1.0)
    assert only_pre["coverage"] > only_post["coverage"]
    assert 0 <= only_post["coverage"] <= 1
    res = cc.prevalence(_cutoff_rows(), 20, np.random.default_rng(3))
    assert set(res["curve"]) == {f"{s:.2f}" for s in cc.SHARES}
    assert f"{cc.PANEL_SHARE:.3f}" in res["at_selected_shares"]
    sel = res["at_selected_shares"][f"{cc.PANEL_SHARE:.3f}"]
    assert sel["achieved_coverage_p_max"] == pytest.approx(sel["coverage"])
    assert res["sample"] == {"post": 100, "pre": 100}
    assert res["shares_where_option_beats_p_known"] == [
        float(k) for k, v in res["curve"].items() if v["option"] > v["p_known"]
    ]


def test_at_share_uses_the_given_indices() -> None:
    x = cc.paired_rows(_cutoff_rows())
    idx = np.arange(len(x["post"]))
    assert cc.at_share(x, 0.3, idx) == cc.at_share(x, 0.3)


def test_weighted_accuracy_matches_coverage_exactly() -> None:
    score = np.array([0.9, 0.8, 0.7])
    ok = np.array([True, False, True])
    w = np.array([1.0, 1.0, 2.0])
    assert weighted_accuracy_at(score, ok, w, 0.25) == pytest.approx((1.0, 0.25))
    assert weighted_accuracy_at(score, ok, w, 0.5) == pytest.approx((0.5, 0.5))
    acc, cov = weighted_accuracy_at(score, ok, w, 0.375)
    assert cov == pytest.approx(0.375)
    assert acc == pytest.approx(1 / 1.5)


def test_month_blocks_keep_period_sizes() -> None:
    months = np.array([f"2024-{m:02d}" for m in range(1, 13) for _ in range(3)])
    post = np.array([m >= "2024-10" for m in months])
    idx = month_block_indices(months, post, np.random.default_rng(6), 2)
    assert int(post[idx].sum()) == int(post.sum())
    assert int((~post[idx]).sum()) == int((~post).sum())


def test_figure_is_written(tmp_path: Path) -> None:
    eq = cc.equal_confidence(_news(0.2), 10, np.random.default_rng(4))
    prev = cc.prevalence(_cutoff_rows(), 5, np.random.default_rng(5))
    out = cc.figure(eq, prev, tmp_path / "f.png")
    assert out.stat().st_size > 10_000


# --- knowledge cutoff --------------------------------------------------------------


def _oracle_rows(step: float = 0.2, known_step: float = 0.1) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    rows = []
    for i, m in enumerate(MONTHS):
        post = m >= "2024-11"
        for _ in range(200):
            acc = 0.75 - (step if post else 0.0)
            rows.append(
                {
                    "month": m,
                    "correct": bool(rng.random() < acc),
                    "p_max": 0.75 + (0.05 if post else 0.0) + rng.normal(0, 0.05),
                    "p_known": 0.4
                    - 0.01 * i
                    - (known_step if post else 0.0)
                    + rng.normal(0, 0.05),
                }
            )
    return pd.DataFrame(rows)


def test_cutoff_reanalysis_on_synthetic_rows() -> None:
    tf = _oracle_rows()
    rng = np.random.default_rng(8)
    assert kc.select(tf)["first_post_month"] in ("2024-10", "2024-11", "2024-12")
    fb = kc.full_procedure_bootstrap(tf, 30, rng)
    assert sum(fb["selected_month_share"].values()) == pytest.approx(1.0)
    assert fb["accuracy_drop_ci95"][0] > 0.1
    assert fb["confidence_difference_tail"] == 1.0
    sp = kc.split_within_months(tf, 5, 50, rng)
    assert sp["splits"] == 5
    assert "post_gap_ci95" in sp["first_split"]
    assert sp["first_split"]["measurement"]["n_post"] > 0
    ks = kc.known_shape(tf, 50, rng)
    assert ks["step_at_cutoff"] == pytest.approx(-0.1, abs=0.03)
    assert ks["slope_per_year"] == pytest.approx(-0.12, abs=0.03)
    stats = kc.cutoff_stats(tf, "2024-11")
    assert stats["n_pre"] + stats["n_post"] == len(tf)
    assert kc.years_since_2020("2021-07") == pytest.approx(1.5)


def test_cutoff_stats_known_answer() -> None:
    tf = pd.DataFrame(
        {
            "month": ["2024-01", "2024-01", "2025-01", "2025-01"],
            "p_max": [0.8, 0.6, 0.9, 0.7],
            "correct": [True, False, False, True],
        }
    )
    s = kc.cutoff_stats(tf, "2024-11")
    assert s["post_gap"] == pytest.approx(0.3)
    assert s["post_minus_pre_confidence"] == pytest.approx(0.1)


# --- recalibration across the cutoff -------------------------------------------------


def test_every_function_detects_a_shift_and_not_its_absence() -> None:
    rng = np.random.default_rng(2)
    shifted = rc.every_function(_news(0.25, n=50, quantised=True), 50, rng)
    assert shifted["null_within_value_permutation"]["p"] < 0.05
    assert shifted["per_value_pooled_map"]["mean_gap_post"] > 0.1
    flat = rc.every_function(_news(0.0, seed=3, n=50, quantised=True), 50, rng)
    assert flat["null_within_value_permutation"]["p"] > 0.05


def test_interaction_model() -> None:
    d = _news(0.25, n=50, quantised=True)
    im = rc.interaction_model(d)
    assert im["wald_period_terms"]["p"] < 0.01
    pa = im["predicted_accuracy"]["p_max=0.9"]
    assert pa["pre"] > pa["post"]


def test_band_unions_cutoffs_and_selection() -> None:
    d = _news(0.25, n=50, quantised=True)
    unions = rc.band_unions(d)
    assert unions["intervals"] > 0
    assert unions["share_positive"] > 0.9
    alt = rc.alternative_cutoffs(d)
    assert set(alt) == set(kc.SENSITIVITY)
    sb = rc.selection_bootstrap(d, 20, np.random.default_rng(5))
    assert len(sb["band_difference_ci95"]) == len(cc.COARSE) - 1
    assert 0 <= sb["share_split_at_boundary"] <= 1


def _boundary_frame() -> pd.DataFrame:
    rng = np.random.default_rng(9)
    n = 1000
    pop = pd.DataFrame(
        {
            "unit": [f"popqa:{i}" for i in range(n)],
            "set": "popqa",
            "s_pop": rng.integers(1, 10_000, n),
            "p_max": rng.uniform(0.3, 1, n).round(2),
            "p_known": rng.uniform(0.01, 0.99, n),
        }
    )
    pop["correct"] = rng.random(n) < pop["p_max"]
    pop["gold"] = "A"
    pop["dist"] = pd.Series(
        [
            {"A": p, "B": 1 - p} if c else {"A": 1 - p, "B": p}
            for p, c in zip(pop["p_max"], pop["correct"], strict=True)
        ],
        index=pop.index,
    )
    fab = pop.head(200).assign(
        set="fabricated", unit=lambda d: "f" + d["unit"], correct=None, gold=None
    )
    tf = _oracle_rows().assign(set="oracle_tf").head(3000)
    tf = pd.concat([tf, _oracle_rows().assign(set="oracle_tf").tail(3000)])
    tf = tf.assign(
        unit=[f"oracle_tf:{i}" for i in range(len(tf))],
        p_max=tf["p_max"].clip(0.5, 0.99),
        p_known=tf["p_known"].clip(0.01, 0.99),
        gold="yes",
    )
    tf["dist"] = pd.Series(
        [
            {"yes": p, "no": 1 - p} if c else {"yes": 1 - p, "no": p}
            for p, c in zip(tf["p_max"], tf["correct"], strict=True)
        ],
        index=tf.index,
    )
    return pd.concat([pop, fab, tf], ignore_index=True)


def _evidence_frame() -> pd.DataFrame:
    rng = np.random.default_rng(10)
    rows = []
    for q in range(80):
        for c in ("closed", "dose0", "dose1", "dose2"):
            p = float(round(rng.uniform(0.3, 1), 2))
            ok = bool(rng.random() < p)
            rows.append(
                {
                    "set": "hotpot",
                    "qid": f"q{q}",
                    "cell": c,
                    "p_max": p,
                    "correct": ok,
                    "gold": "A",
                    "dist": {"A": p, "B": 1 - p} if ok else {"A": 1 - p, "B": p},
                }
            )
    return pd.DataFrame(rows)


def test_held_out_and_known_aware_maps() -> None:
    config = rc.Config(bootstrap=30, smece_resamples=5)
    res = rc.held_out(_boundary_frame(), _evidence_frame(), config)
    assert set(res["sets"]) == {"popqa", "daily_oracle_yes_no", "hotpot"}
    post = res["sets"]["daily_oracle_yes_no"]["post_cutoff"]
    assert post["gap_platt"] > 0.1  # a pre-cutoff map leaves the post gap
    assert "accuracy" not in res["sets"]["popqa"]["made_up_entities"]
    gap = res["post_cutoff_gap_after_platt"]
    assert gap["ci95"][0] > 0.1
    assert gap["tail"] == 0.0
    known = rc.with_known(_boundary_frame(), config)
    edge = known["daily_oracle_yes_no"]["post_cutoff"]
    assert {"raw", "platt", "platt_plus_known"} <= set(edge)
    assert len(known["popqa"]["coef_logit_p_max_logit_known"]) == 2
    assert known["popqa"]["made_up_entities"]["accuracy"] == 0.0


# --- abstention ------------------------------------------------------------------------


def test_abstention_matched_coverage() -> None:
    res = ab.abstention(_cutoff_rows(300), 50, np.random.default_rng(2))
    for part in ("all", "post", "pre"):
        opt = res[part]["not_known_option"]
        assert res[part]["threshold_p_max"]["answered"] == opt["answered"]
        assert res[part]["threshold_p_known"]["answered"] == opt["answered"]
        assert set(res[part]["aurc"]) == {
            "p_max_without_option",
            "p_known_without_option",
            "1-P(not known)_with_option",
        }
    assert res["all"]["items"] == 300


def _trend_rows(step: float) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    months = [f"{y}-{m:02d}" for y in range(2020, 2027) for m in range(1, 13)][:79]
    rows = [
        {
            "month": m,
            "p_max": 0.75 + (0.05 if m >= "2024-11" else 0) + rng.normal(0, 0.05),
            "p_known": 0.4
            - 0.003 * i
            - (step if m >= "2024-11" else 0)
            + rng.normal(0, 0.05),
        }
        for i, m in enumerate(months)
        for _ in range(40)
    ]
    return pd.DataFrame(rows)


def test_separability_and_known_trend() -> None:
    tf = _trend_rows(0.1)
    rng = np.random.default_rng(4)
    sep = ab.separability(tf, 50, rng)
    assert sep["p_max"]["mean_post"] > sep["p_max"]["mean_pre"]
    assert sep["1-p_known"]["auroc_post_vs_pre"] > 0.8
    trend = ab.known_trend(tf, 30, rng)
    assert trend["linear"]["step"] == pytest.approx(-0.1, abs=0.03)
    assert set(trend) >= set(ab.TREND_SPECS)
    no_step = ab.known_trend(_trend_rows(0.0), 30, rng)
    assert abs(no_step["linear"]["step"]) < 0.03


def test_trend_design_shapes() -> None:
    t = np.linspace(0, 6, 79)
    step = (t > 4).astype(float)
    assert ab.trend_design(t, "linear", None).shape == (79, 2)
    assert ab.trend_design(t, "quadratic", step).shape == (79, 4)
    assert ab.trend_design(t, "spline_4", step).shape == (79, 5)


def test_renormalised_readings() -> None:
    from beyond_answer_confidence.tasks.synthetic import make_scenario

    rows: list[dict[str, Any]] = []
    for i in range(20):
        sc = make_scenario(i)
        star = [sc.p_star[o] for o in sc.options]
        rows.append(
            {
                "scenario": i,
                "format": "noul",
                **{f"p_opt{k}": 0.9 * s for k, s in enumerate(star)},
            }
        )
        rows.append(
            {
                "scenario": i,
                "format": "score",
                **{f"s_opt{k}": s * 10 - 0.5 for k, s in enumerate(star)},
            }
        )
        rows.append(
            {
                "scenario": i,
                "format": "instructed",
                "dist": dict(zip(sc.options, star, strict=True)),
            }
        )
    res = ab.renormalised(rows)
    assert res["noul"]["sum"] == pytest.approx(0.9)
    assert res["noul"]["mae_renormalised"] == pytest.approx(0.0, abs=1e-9)
    assert res["noul"]["mae_raw"] > 0
    assert res["instructed"]["tv_renormalised"] == pytest.approx(0.0, abs=1e-9)
    assert res["score"]["kl_renormalised"] == pytest.approx(0.0, abs=1e-6)
