"""The knowledge cutoff with its selection step included in the uncertainty.

Offline; reads the dated yes/no news rows of the knowledge-boundary
experiment. The change point is the month that best splits monthly accuracy
(:func:`beyond_answer_confidence.stats.changepoint.change_point`); statistics computed
at a selected split on the same items understate their uncertainty unless
the selection is repeated. This analysis:

1. **Full-procedure bootstrap.** Resample items within each month, re-select
   the change point and recompute the post-cutoff statistics every time.
2. **Split within months.** Split each month's items in half at random,
   select the change point on one half and compute the statistics on the
   other (months are ordered in time, so the split is by item); repeated
   over many random splits.
3. **Sensitivity.** The same statistics at fixed cutoffs 2024-06 ... 2025-03.
4. **Shape of P(known).** Does the "do you know?" probability step down at
   the cutoff or decline gradually? Item-level regression of P(known) on
   time and a post-cutoff step, with a month-cluster bootstrap, and the
   same two-mean change point applied to monthly P(known).

The statistics at a split are the accuracy on each side, the post-cutoff
calibration gap (mean ``p_max`` minus accuracy) and the post minus pre mean
confidence.

Output: ``<output_dir>/knowledge_cutoff/summary.json``.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from beyond_answer_confidence.analyses.cutoff_calibration import CUTOFF_MONTH, NEWS_SET
from beyond_answer_confidence.analyses.rows import experiment_frame
from beyond_answer_confidence.config import as_dict
from beyond_answer_confidence.experiments.base import Analysis, write_json
from beyond_answer_confidence.settings import Settings
from beyond_answer_confidence.stats.changepoint import change_point
from beyond_answer_confidence.stats.resampling import bootstrap_means, percentile_ci

NAME = "knowledge_cutoff"
SENSITIVITY = (
    "2024-06",
    "2024-07",
    "2024-08",
    "2024-09",
    "2024-10",
    "2024-11",
    "2024-12",
    "2025-01",
    "2025-02",
    "2025-03",
)
"""Fixed alternative cutoffs (first post-cutoff month)."""


@dataclass(frozen=True)
class Config:
    """Options.

    Attributes:
        bootstrap: Bootstrap resamples (one generator for all sections, in
            the order listed in the module docstring).
        splits: Random within-month splits.
        seed: Seed.
        calibration_margin: Equivalence margin for the post-cutoff gap (the
            tail share reports how often a resample falls outside it).
    """

    bootstrap: int = 2000
    splits: int = 200
    seed: int = 0
    calibration_margin: float = 0.05


def cutoff_stats(tf: pd.DataFrame, first_post: str) -> dict[str, float]:
    """Statistics for a given first post-cutoff month.

    Args:
        tf: Dated yes/no rows with ``month``, ``p_max`` and ``correct``.
        first_post: First month counted as post-cutoff.

    Returns:
        Items and accuracy on each side, the post-cutoff calibration gap and
        the post minus pre mean confidence.
    """
    post = tf[tf["month"] >= first_post]
    pre = tf[tf["month"] < first_post]
    return {
        "n_pre": len(pre),
        "n_post": len(post),
        "accuracy_pre": float(pre["correct"].mean()),
        "accuracy_post": float(post["correct"].mean()),
        "post_gap": float((post["p_max"] - post["correct"].astype(float)).mean()),
        "post_minus_pre_confidence": float(post["p_max"].mean() - pre["p_max"].mean()),
    }


def select(tf: pd.DataFrame) -> dict[str, Any]:
    """Change point of monthly accuracy (weighted by items per month).

    Args:
        tf: Dated yes/no rows.

    Returns:
        The :func:`beyond_answer_confidence.stats.changepoint.change_point` result.
    """
    monthly = (
        tf.groupby("month")
        .agg(acc=("correct", "mean"), n=("correct", "size"))
        .sort_index()
    )
    return change_point(list(monthly.index), list(monthly["acc"]), list(monthly["n"]))


def resample_within_months(tf: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Stratified bootstrap: resample items with replacement inside each month.

    Args:
        tf: Rows.
        rng: Random generator (one ``choice`` per month, in month order).

    Returns:
        Resampled rows.
    """
    idx = [
        rng.choice(g, len(g), replace=True)
        for g in tf.groupby("month").indices.values()
    ]
    return tf.iloc[np.concatenate(idx)]


def _intervals(
    post_gap: np.ndarray, conf_diff: np.ndarray, margin: float
) -> dict[str, Any]:
    return {
        "post_gap_ci95": percentile_ci(post_gap),
        "post_gap_outside_margin_tail": max(
            float(np.mean(post_gap >= margin)),
            float(np.mean(post_gap <= -margin)),
        ),
        "confidence_difference_ci95": percentile_ci(conf_diff),
        "confidence_difference_tail": float(np.mean(conf_diff >= 0)),
    }


def full_procedure_bootstrap(
    tf: pd.DataFrame, b: int, rng: np.random.Generator, margin: float = 0.05
) -> dict[str, Any]:
    """Bootstrap that repeats the change-point selection in every resample.

    Args:
        tf: Dated yes/no rows.
        b: Resamples.
        rng: Random generator.
        margin: Equivalence margin for the post-cutoff gap.

    Returns:
        Distribution of the selected month, the share selecting
        :data:`CUTOFF_MONTH`, and selection-aware intervals for the accuracy
        drop, the post-cutoff gap and the confidence difference.
    """
    months, gaps, diffs, drops = [], [], [], []
    for _ in range(b):
        r = resample_within_months(tf, rng)
        cp = select(r)
        s = cutoff_stats(r, cp["first_post_month"])
        months.append(cp["first_post_month"])
        gaps.append(s["post_gap"])
        diffs.append(s["post_minus_pre_confidence"])
        drops.append(cp["drop"])
    counts = pd.Series(months).value_counts()
    return {
        "resamples": b,
        "selected_month_share": {
            str(k): float(v / b) for k, v in counts.sort_index().items()
        },
        "share_selecting_cutoff_month": float(
            np.mean(np.array(months) == CUTOFF_MONTH)
        ),
        "accuracy_drop_ci95": percentile_ci(np.array(drops)),
        **_intervals(np.array(gaps), np.array(diffs), margin),
    }


def split_within_months(
    tf: pd.DataFrame,
    splits: int,
    b: int,
    rng: np.random.Generator,
    margin: float = 0.05,
) -> dict[str, Any]:
    """Select the cutoff on half of each month's items, measure on the other half.

    Args:
        tf: Dated yes/no rows.
        splits: Random splits.
        b: Bootstrap resamples for the measurement half of the first split.
        rng: Random generator.
        margin: Equivalence margin for the post-cutoff gap.

    Returns:
        The first split in full (selected month, statistics on the other
        half and bootstrap intervals) and, over all splits, the distribution
        of the selected month and of the measurement-half statistics.
    """
    runs = []
    first: dict[str, Any] = {}
    for s in range(splits):
        half = np.zeros(len(tf), bool)
        for g in tf.groupby("month").indices.values():
            half[rng.choice(g, len(g) // 2, replace=False)] = True
        find, test = tf[half], tf[~half]
        cp = select(find)
        stats = cutoff_stats(test, cp["first_post_month"])
        runs.append({"month": cp["first_post_month"], **stats})
        if s == 0:
            post = test[test["month"] >= cp["first_post_month"]]
            pre = test[test["month"] < cp["first_post_month"]]
            gap = bootstrap_means(
                (post["p_max"] - post["correct"].astype(float)).to_numpy(), b, rng
            )
            diff = bootstrap_means(post["p_max"].to_numpy(), b, rng) - bootstrap_means(
                pre["p_max"].to_numpy(), b, rng
            )
            first = {
                "selected": cp,
                "measurement": stats,
                **_intervals(gap, diff, margin),
            }
    df = pd.DataFrame(runs)
    return {
        "splits": splits,
        "first_split": first,
        "selected_month_share": {
            str(k): float(v / splits)
            for k, v in df["month"].value_counts().sort_index().items()
        },
        "measurement_post_gap_range": [
            float(df["post_gap"].min()),
            float(df["post_gap"].max()),
        ],
        "measurement_accuracy_drop_mean": float(
            (df["accuracy_pre"] - df["accuracy_post"]).mean()
        ),
        "measurement_conf_diff_range": [
            float(df["post_minus_pre_confidence"].min()),
            float(df["post_minus_pre_confidence"].max()),
        ],
    }


def years_since_2020(month: str) -> float:
    """Time in years from 2020-01 to the start of a month.

    Args:
        month: ``YYYY-MM``.

    Returns:
        Years (months count as twelfths).
    """
    return (int(month[:4]) - 2020) + (int(month[5:7]) - 1) / 12


def known_shape(tf: pd.DataFrame, b: int, rng: np.random.Generator) -> dict[str, Any]:
    """Step versus gradual decline in P(known) around the cutoff.

    Fits ``p_known ~ 1 + t + step`` by least squares on items, where ``t``
    is years since 2020-01 and ``step`` marks months from
    :data:`CUTOFF_MONTH` on; intervals come from a month-cluster bootstrap
    (resamples without both periods are skipped). Also reports the
    pre-cutoff slope alone and the two-mean change point of monthly
    P(known).

    Args:
        tf: Dated yes/no rows with ``month`` and ``p_known``.
        b: Resamples.
        rng: Random generator.

    Returns:
        Coefficients with intervals, monthly means and P(known)'s own change
        point.
    """
    d = tf.copy()
    d["t"] = d["month"].map(years_since_2020)
    d["step"] = (d["month"] >= CUTOFF_MONTH).astype(float)

    def fit(g: pd.DataFrame) -> np.ndarray:
        x = np.column_stack([np.ones(len(g)), g["t"], g["step"]])
        coef = np.linalg.lstsq(x, g["p_known"].to_numpy(float), rcond=None)[0]
        return np.asarray(coef)

    def pre_slope(g: pd.DataFrame) -> float:
        p = g[g["step"] == 0]
        return float(np.polyfit(p["t"], p["p_known"], 1)[0])

    coef = fit(d)
    months = np.array(sorted(d["month"].unique()))
    groups = dict(tuple(d.groupby("month")))
    boots, pre_boots = [], []
    for _ in range(b):
        pick = rng.choice(months, len(months), replace=True)
        r = pd.concat([groups[m] for m in pick])
        if r["step"].nunique() < 2:
            continue
        boots.append(fit(r))
        pre_boots.append(pre_slope(r))
    bc = np.array(boots)
    monthly = d.groupby("month").agg(p_known=("p_known", "mean"), n=("p_known", "size"))
    cp = change_point(list(monthly.index), list(monthly["p_known"]), list(monthly["n"]))
    return {
        "model": f"p_known ~ 1 + years_since_2020 + step(month >= {CUTOFF_MONTH})",
        "slope_per_year": float(coef[1]),
        "slope_per_year_ci95": percentile_ci(bc[:, 1]),
        "step_at_cutoff": float(coef[2]),
        "step_at_cutoff_ci95": percentile_ci(bc[:, 2]),
        "pre_cutoff_slope_per_year": pre_slope(d),
        "pre_cutoff_slope_ci95": percentile_ci(np.array(pre_boots)),
        "p_known_change_point": {
            k: cp[k] for k in ("first_post_month", "pre_accuracy", "post_accuracy")
        },
        "monthly_p_known": {str(m): float(v) for m, v in monthly["p_known"].items()},
        "cluster_resamples": len(bc),
    }


def analyse(tf: pd.DataFrame, config: Config) -> dict[str, Any]:
    """Every section on the dated yes/no rows.

    Args:
        tf: Dated yes/no rows (index reset, boolean ``correct``).
        config: Options.

    Returns:
        ``selection``, ``stats_at_selection``, ``full_procedure_bootstrap``,
        ``split_within_months``, ``sensitivity`` and ``known_shape``.
    """
    rng = np.random.default_rng(config.seed)
    b, margin = config.bootstrap, config.calibration_margin
    selected = select(tf)
    return {
        "selection": selected,
        "stats_at_selection": cutoff_stats(tf, selected["first_post_month"]),
        "full_procedure_bootstrap": full_procedure_bootstrap(tf, b, rng, margin),
        "split_within_months": split_within_months(tf, config.splits, b, rng, margin),
        "sensitivity": {m: cutoff_stats(tf, m) for m in SENSITIVITY},
        "known_shape": known_shape(tf, b, rng),
    }


def run(settings: Settings, config: Config) -> dict[str, Any]:
    """Run the analysis and write the summary.

    Args:
        settings: Run settings.
        config: Options.

    Returns:
        The results.
    """
    rows = experiment_frame(settings, "knowledge_boundary", dtype={"month": str})
    tf = rows[rows["set"] == NEWS_SET].reset_index(drop=True)
    tf["correct"] = tf["correct"].astype(bool)
    res = analyse(tf, config)
    write_json(settings.out(NAME) / "summary.json", {"config": as_dict(config), **res})
    return res


ANALYSIS = Analysis(
    name=NAME,
    summary="Change point with selection-aware intervals; shape of P(known)",
    config_type=Config,
    run=run,
)
