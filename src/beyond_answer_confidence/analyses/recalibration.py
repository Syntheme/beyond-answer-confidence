"""What post-hoc recalibration of the answer probability can and cannot fix.

Offline; reads saved rows. Three sections, each with its own generator
seeded from ``Config.seed``:

1. ``held_out``: temperature and Platt maps fitted on a random half of
   in-distribution items, tested on the other half and at the knowledge
   edge: PopQA -> fabricated entities; dated yes/no news before the cutoff
   -> after it; HotpotQA closed-book and full-evidence questions -> the
   partial-evidence cells. Reports mean confidence, accuracy, SmoothECE and
   the calibration gap before and after, and the post-cutoff gap left by
   the Platt map with an item-bootstrap interval.
2. ``with_known``: the same halves, comparing the Platt map with a logistic
   map that also sees P(known) (``logit p_max`` and ``logit P(known)``).
3. ``across_cutoff``: can *any* function of ``p_max`` be calibrated on both
   sides of the cutoff? ``p_max`` takes few distinct values, so an exact
   lower bound on the mean absolute gap of every function of it is
   available (:func:`beyond_answer_confidence.stats.recalibration.lower_bound`), with
   two permutation nulls (periods shuffled within values; random months as
   the later period). Plus cross-fitted recalibrators (half of the months
   each way), a logistic model ``correct ~ logit(p_max) * post`` with
   month-clustered errors, the pre-post accuracy difference on every union
   of adjacent fine bands, the coarse band differences at alternative
   cutoffs, and band differences with the change point re-selected inside
   a moving-block bootstrap over months.

Inputs: ``knowledge_boundary`` and ``evidence_sufficiency`` rows. Output:
``<output_dir>/recalibration/summary.json``.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from beyond_answer_confidence.analyses.cutoff_calibration import (
    BLOCK_MONTHS,
    COARSE,
    CUTOFF_MONTH,
    FINE,
    NEWS_SET,
    band_table,
)
from beyond_answer_confidence.analyses.knowledge_cutoff import SENSITIVITY
from beyond_answer_confidence.analyses.rows import experiment_frame, records_frame
from beyond_answer_confidence.config import as_dict
from beyond_answer_confidence.experiments.base import Analysis, write_json
from beyond_answer_confidence.metrics.calibration import smece
from beyond_answer_confidence.settings import Settings
from beyond_answer_confidence.stats.changepoint import change_point
from beyond_answer_confidence.stats.recalibration import (
    apply_platt,
    cross_fitted,
    fit_logistic_map,
    fit_platt,
    fit_temperature,
    level_table,
    lower_bound,
    per_value_map_gaps,
    predict_logistic_map,
    random_months_null,
    split_half,
    temper,
    within_value_permutation_null,
)
from beyond_answer_confidence.stats.resampling import (
    bootstrap_means,
    percentile_ci,
    smece_interval,
)

NAME = "recalibration"
MIN_ITEMS = 30
"""Band unions need at least this many items in each period."""


@dataclass(frozen=True)
class Config:
    """Options.

    Attributes:
        bootstrap: Bootstrap resamples.
        seed: Seed of every section's generator and of the half splits.
        permutations: Permutations per null for the every-function bound.
        repeats: Random month splits for the cross-fitted recalibrators.
        smece_resamples: Resamples drawn with each held-out SmoothECE (only
            the estimate is reported; the draws keep the generator's
            sequence).
        gap_reference: Reference calibration gap for the post-cutoff tail
            share.
    """

    bootstrap: int = 2000
    seed: int = 0
    permutations: int = 2000
    repeats: int = 50
    smece_resamples: int = 200
    gap_reference: float = 0.05


# --- 1. held-out temperature and Platt maps --------------------------------------


def _evaluate(
    test: pd.DataFrame,
    t: float,
    ab: tuple[float, float],
    config: Config,
    rng: np.random.Generator,
) -> dict[str, Any]:
    pm = test["p_max"].to_numpy(float)
    tempered = temper(list(test["dist"]), t)
    pm_t = np.array([max(d.values()) for d in tempered])
    pm_p = apply_platt(pm, ab)
    res: dict[str, Any] = {
        "n": len(test),
        "mean_p_max": float(pm.mean()),
        "mean_p_max_temperature": float(pm_t.mean()),
        "mean_platt": float(pm_p.mean()),
    }
    if test["correct"].notna().all():
        ok = test["correct"].to_numpy(bool)

        def smece_point_drawing_interval(conf: np.ndarray) -> float:
            # Only the SmoothECE point estimate is reported, but the full
            # bootstrap interval is still drawn: it keeps the generator's
            # stream, and so every later interval, reproducible.
            chk = smece_interval(conf, ok, config.smece_resamples, rng)
            return float(chk["smece"])

        res |= {
            "accuracy": float(ok.mean()),
            "smece_before": smece_point_drawing_interval(pm),
            "smece_temperature": smece_point_drawing_interval(pm_t),
            "smece_platt": smece_point_drawing_interval(pm_p),
            "gap_before": float(pm.mean() - ok.mean()),
            "gap_platt": float(pm_p.mean() - ok.mean()),
        }
    return res


def _fit(fit: pd.DataFrame) -> tuple[float, tuple[float, float]]:
    t = fit_temperature(list(fit["dist"]), list(fit["gold"]))
    ab = fit_platt(fit["p_max"].to_numpy(float), fit["correct"].to_numpy(bool))
    return t, ab


def held_out(
    boundary: pd.DataFrame, evidence: pd.DataFrame, config: Config
) -> dict[str, Any]:
    """Temperature and Platt maps fitted in-distribution, tested held-out and at the edge.

    Args:
        boundary: Knowledge-boundary rows.
        evidence: Evidence-sufficiency rows.
        config: Options.

    Returns:
        ``sets`` (per edge set: temperature, Platt coefficients and
        before/after statistics) and ``post_cutoff_gap_after_platt``.
    """
    rng = np.random.default_rng(config.seed)
    seed = config.seed
    out: dict[str, Any] = {}
    real = boundary[boundary["set"] == "popqa"]
    fit, test = split_half(real, "unit", seed)
    t, ab = _fit(fit)
    out["popqa"] = {
        "T": t,
        "platt": ab,
        "held_out_real": _evaluate(test, t, ab, config, rng),
        "made_up_entities": _evaluate(
            boundary[boundary["set"] == "fabricated"], t, ab, config, rng
        ),
    }
    tf = boundary[boundary["set"] == NEWS_SET]
    pre = tf[tf["month"] < CUTOFF_MONTH]
    post = tf[tf["month"] >= CUTOFF_MONTH]
    fit, test = split_half(pre, "unit", seed)
    t, ab = _fit(fit)
    post_eval = _evaluate(post, t, ab, config, rng)
    gap = apply_platt(post["p_max"].to_numpy(float), ab) - post["correct"].to_numpy(
        float
    )
    boot = bootstrap_means(gap, config.bootstrap, rng)
    out["daily_oracle_yes_no"] = {
        "T": t,
        "platt": ab,
        "held_out_pre_cutoff": _evaluate(test, t, ab, config, rng),
        "post_cutoff": post_eval,
    }
    hp = evidence[evidence["set"] == "hotpot"]
    ind = hp[hp["cell"].isin(["closed", "dose2"])]
    fit, test = split_half(ind, "qid", seed)
    t, ab = _fit(fit)
    held_q = set(test["qid"])
    out["hotpot"] = {
        "T": t,
        "platt": ab,
        "held_out_closed_and_dose2": _evaluate(test, t, ab, config, rng),
        **{
            f"held_out_{c}": _evaluate(
                hp[(hp["cell"] == c) & hp["qid"].isin(held_q)], t, ab, config, rng
            )
            for c in ("dose0", "dose1")
        },
    }
    return {
        "sets": out,
        "post_cutoff_gap_after_platt": {
            "n": len(gap),
            "mean_gap_after_platt": float(gap.mean()),
            "ci95": percentile_ci(boot),
            "reference": config.gap_reference,
            "tail": float(np.mean(boot <= config.gap_reference)),
        },
    }


# --- 2. a map that also sees P(known) ----------------------------------------------


def _evaluate_known(
    test: pd.DataFrame,
    ab: tuple[float, float],
    model: Any,
    b: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    ok = test["correct"].to_numpy(float)
    raw = test["p_max"].to_numpy(float)
    platt = apply_platt(raw, ab)
    known = predict_logistic_map(model, [test["p_max"], test["p_known"]])
    out: dict[str, Any] = {"n": len(test), "accuracy": float(ok.mean())}
    for name, conf in (("raw", raw), ("platt", platt), ("platt_plus_known", known)):
        gap = conf - ok
        boot = bootstrap_means(gap, b, rng)
        out[name] = {
            "mean_confidence": float(conf.mean()),
            "gap": float(gap.mean()),
            "gap_ci95": percentile_ci(boot),
            "smece": smece(conf, ok),
        }
    return out


def with_known(boundary: pd.DataFrame, config: Config) -> dict[str, Any]:
    """Platt versus Platt plus P(known), fitted in-distribution, tested at the edge.

    Args:
        boundary: Knowledge-boundary rows.
        config: Options (the same half splits as :func:`held_out`).

    Returns:
        Per edge set: the map's coefficients and before/after statistics.
    """
    rng = np.random.default_rng(config.seed)
    b, seed = config.bootstrap, config.seed
    out: dict[str, Any] = {}

    def fit_both(fit: pd.DataFrame) -> tuple[tuple[float, float], Any]:
        ab = fit_platt(fit["p_max"].to_numpy(float), fit["correct"].to_numpy(bool))
        return ab, fit_logistic_map([fit["p_max"], fit["p_known"]], fit["correct"])

    real = boundary[boundary["set"] == "popqa"]
    fit, test = split_half(real, "unit", seed)
    ab, model = fit_both(fit)
    fab = boundary[boundary["set"] == "fabricated"].copy()
    fab["correct"] = False
    out["popqa"] = {
        "coef_logit_p_max_logit_known": model.coef_[0].tolist(),
        "held_out_real": _evaluate_known(test, ab, model, b, rng),
        "made_up_entities": _evaluate_known(fab, ab, model, b, rng),
    }
    tf = boundary[boundary["set"] == NEWS_SET]
    pre = tf[tf["month"] < CUTOFF_MONTH]
    post = tf[tf["month"] >= CUTOFF_MONTH]
    fit, test = split_half(pre, "unit", seed)
    ab, model = fit_both(fit)
    out["daily_oracle_yes_no"] = {
        "coef_logit_p_max_logit_known": model.coef_[0].tolist(),
        "held_out_pre_cutoff": _evaluate_known(test, ab, model, b, rng),
        "post_cutoff": _evaluate_known(post, ab, model, b, rng),
    }
    return out


# --- 3. every function of p_max across the cutoff -----------------------------------


def every_function(
    tf: pd.DataFrame, perms: int, rng: np.random.Generator
) -> dict[str, Any]:
    """The lower bound for every function of ``p_max`` with two permutation nulls.

    Args:
        tf: Dated yes/no rows with ``p_max``, ``correct``, ``post`` and
            ``month``.
        perms: Permutations per null (within-value null first).
        rng: Random generator.

    Returns:
        Distinct values, the observed bound, each null's mean, 95th
        percentile and permutation p-value, and the per-value pooled map's
        gaps.
    """
    levels = level_table(tf)
    obs = lower_bound(levels)
    null_item = within_value_permutation_null(tf, perms, rng)
    null_month = random_months_null(tf, perms, rng)
    shared = levels.dropna(subset=["a_pre", "a_post"])
    return {
        "distinct_values": len(levels),
        "values_seen_in_both_periods": len(shared),
        "items_at_shared_values": int(shared["n_pre"].sum() + shared["n_post"].sum()),
        "lower_bound": obs,
        "null_within_value_permutation": {
            "mean": float(null_item.mean()),
            "p95": float(np.percentile(null_item, 95)),
            "p": float((1 + np.sum(null_item >= obs)) / (perms + 1)),
        },
        "null_random_months": {
            "mean": float(null_month.mean()),
            "p95": float(np.percentile(null_month, 95)),
            "p": float((1 + np.sum(null_month >= obs)) / (perms + 1)),
        },
        "excess_over_within_value_null": float(obs - null_item.mean()),
        "per_value_pooled_map": per_value_map_gaps(tf),
    }


def interaction_model(tf: pd.DataFrame) -> dict[str, Any]:
    """Logistic ``correct ~ logit(p_max) * post`` with month-clustered errors.

    Args:
        tf: Dated yes/no rows with ``p_max``, ``correct``, ``post`` and
            ``month``.

    Returns:
        Coefficients, the joint Wald test of the two period terms, and the
        predicted accuracy per period at ``p_max`` 0.7 and 0.9.
    """
    import statsmodels.api as sm

    lp = np.log(
        np.clip(tf["p_max"], 1e-4, 1 - 1e-4)
        / (1 - np.clip(tf["p_max"], 1e-4, 1 - 1e-4))
    )
    post = tf["post"].astype(float).to_numpy()
    x = np.column_stack([np.ones(len(tf)), lp, post, lp * post])
    groups = pd.factorize(tf["month"])[0]
    fit = sm.GLM(
        tf["correct"].astype(float).to_numpy(), x, family=sm.families.Binomial()
    ).fit(cov_type="cluster", cov_kwds={"groups": groups})
    wald = fit.wald_test(np.array([[0, 0, 1, 0], [0, 0, 0, 1]]), scalar=True)

    def pred(p: float, is_post: float) -> float:
        z = np.log(p / (1 - p))
        eta = fit.params @ np.array([1, z, is_post, z * is_post])
        return float(1 / (1 + np.exp(-eta)))

    return {
        "coefficients": dict(
            zip(
                ["intercept", "logit_p_max", "post", "logit_p_max:post"],
                map(float, fit.params),
                strict=True,
            )
        ),
        "wald_period_terms": {
            "chi2": float(wald.statistic),
            "df": 2,
            "p": float(wald.pvalue),
        },
        "predicted_accuracy": {
            f"p_max={p}": {"pre": pred(p, 0.0), "post": pred(p, 1.0)}
            for p in (0.7, 0.9)
        },
    }


def band_unions(tf: pd.DataFrame) -> dict[str, Any]:
    """Pre-post accuracy difference on every union of adjacent fine bands.

    Args:
        tf: Dated yes/no rows with ``p_max``, ``correct`` and ``post``.

    Returns:
        Number of intervals with at least :data:`MIN_ITEMS` per period, the
        share with a positive difference, and the smallest difference
        overall (with its interval) and from 0.6 up.
    """
    edges = list(FINE)
    out = []
    p, ok, post = (
        tf["p_max"].to_numpy(float),
        tf["correct"].to_numpy(float),
        tf["post"].to_numpy(bool),
    )
    for i in range(len(edges) - 1):
        for j in range(i + 1, len(edges)):
            m = (p >= edges[i]) & (p < edges[j])
            n_pre, n_post = int((m & ~post).sum()), int((m & post).sum())
            if n_pre < MIN_ITEMS or n_post < MIN_ITEMS:
                continue
            out.append(
                {
                    "from": edges[i],
                    "to": min(edges[j], 1.0),
                    "n_pre": n_pre,
                    "n_post": n_post,
                    "difference": float(ok[m & ~post].mean() - ok[m & post].mean()),
                }
            )
    df = pd.DataFrame(out)
    above = df[df["from"] >= 0.6 - 1e-9]
    lo = df.iloc[int(np.argmin(df["difference"].to_numpy()))]
    return {
        "intervals": len(df),
        "share_positive": float((df["difference"] > 0).mean()),
        "min_difference": float(lo["difference"]),
        "min_interval": [float(lo["from"]), float(lo["to"])],
        "intervals_from_0.6": len(above),
        "min_difference_from_0.6": float(above["difference"].min()),
    }


def alternative_cutoffs(tf: pd.DataFrame) -> dict[str, list[float]]:
    """Coarse band differences (pre - post accuracy) at fixed alternative cutoffs.

    Args:
        tf: Dated yes/no rows with ``month``, ``p_max`` and ``correct``.

    Returns:
        Cutoff -> difference per coarse band.
    """
    out = {}
    for c in SENSITIVITY:
        t = band_table(tf.assign(post=tf["month"] >= c), COARSE)
        out[c] = [float(x) for x in (t["acc_pre"] - t["acc_post"])]
    return out


def selection_bootstrap(
    tf: pd.DataFrame, b: int, rng: np.random.Generator
) -> dict[str, Any]:
    """Band differences with the change point re-selected in every resample.

    Within each period, months are drawn in blocks of :data:`BLOCK_MONTHS`
    consecutive months until the period's length is reached; the resampled
    earlier series is followed by the resampled later series, the change
    point is selected on that series (weighted two-mean least squares, as
    for the fixed cutoff), and the coarse band differences are computed
    at the selected split.

    Args:
        tf: Dated yes/no rows with ``post``, ``month``, ``p_max`` and
            ``correct``.
        b: Resamples.
        rng: Random generator.

    Returns:
        Per band the 95 % interval of the difference; the share of resamples
        whose split falls exactly at the period boundary; the shift
        distribution in months.
    """
    by_month = dict(tuple(tf.groupby("month")))
    periods = [
        np.array(sorted(tf.loc[tf["post"] == per, "month"].unique()))
        for per in (False, True)
    ]
    diffs, shifts = [], []
    for _ in range(b):
        series: list[pd.DataFrame] = []
        for ms in periods:
            length = min(BLOCK_MONTHS, len(ms))
            starts = np.arange(len(ms) - length + 1)
            drawn: list[str] = []
            while len(drawn) < len(ms):
                s0 = int(rng.choice(starts))
                drawn.extend(ms[s0 : s0 + length])
            series.extend(by_month[m] for m in drawn[: len(ms)])
        labelled = [g.assign(t=f"{k:03d}") for k, g in enumerate(series)]
        r = pd.concat(labelled)
        monthly = (
            r.groupby("t")
            .agg(acc=("correct", "mean"), n=("correct", "size"))
            .sort_index()
        )
        cp = change_point(list(monthly.index), list(monthly["acc"]), list(monthly["n"]))
        k = int(cp["first_post_month"])
        shifts.append(k - len(periods[0]))
        t = band_table(r.assign(post=r["t"].astype(int) >= k), COARSE)
        diffs.append((t["acc_pre"] - t["acc_post"]).to_numpy(float))
    d = np.array(diffs)
    sh = np.array(shifts)
    return {
        "resamples": b,
        "band_difference_ci95": [
            percentile_ci(d[:, i][~np.isnan(d[:, i])]) for i in range(d.shape[1])
        ],
        "share_split_at_boundary": float(np.mean(sh == 0)),
        "split_shift_months": {
            "p2.5": float(np.percentile(sh, 2.5)),
            "p97.5": float(np.percentile(sh, 97.5)),
        },
    }


def across_cutoff(tf: pd.DataFrame, config: Config) -> dict[str, Any]:
    """Every check of recalibrating ``p_max`` across the cutoff (one generator).

    Args:
        tf: Dated yes/no rows with ``post``, ``month``, ``p_max`` and
            ``correct``.
        config: Options.

    Returns:
        ``every_function``, ``cross_fitted``, ``interaction_model``,
        ``band_unions``, ``alternative_cutoffs`` and ``selection_bootstrap``.
    """
    rng = np.random.default_rng(config.seed)
    return {
        "every_function": every_function(tf, config.permutations, rng),
        "cross_fitted": cross_fitted(tf, config.repeats, rng),
        "interaction_model": interaction_model(tf),
        "band_unions": band_unions(tf),
        "alternative_cutoffs": alternative_cutoffs(tf),
        "selection_bootstrap": selection_bootstrap(tf, config.bootstrap, rng),
    }


def news_frame(settings: Settings) -> pd.DataFrame:
    """Dated yes/no rows with ``post`` and boolean ``correct``.

    Parsed with the default pandas float parser, kept for reproducibility of
    released results (see :mod:`beyond_answer_confidence.analyses.rows`).

    Args:
        settings: Run settings.

    Returns:
        The rows.
    """
    rows = experiment_frame(settings, "knowledge_boundary", dtype={"month": str})
    tf = rows[rows["set"] == NEWS_SET].copy()
    tf["post"] = tf["month"] >= CUTOFF_MONTH
    tf["correct"] = tf["correct"].astype(bool)
    return tf


def run(settings: Settings, config: Config) -> dict[str, Any]:
    """Run the three sections and write the summary.

    Args:
        settings: Run settings.
        config: Options.

    Returns:
        ``held_out``, ``with_known`` and ``across_cutoff`` results.
    """
    boundary = records_frame(settings, "knowledge_boundary")
    evidence = records_frame(settings, "evidence_sufficiency")
    res = {
        "held_out": held_out(boundary, evidence, config),
        "with_known": with_known(boundary, config),
        "across_cutoff": across_cutoff(news_frame(settings), config),
    }
    write_json(settings.out(NAME) / "summary.json", {"config": as_dict(config), **res})
    return res


ANALYSIS = Analysis(
    name=NAME,
    summary="Held-out recalibration, a P(known)-aware map, and every map of p_max",
    config_type=Config,
    run=run,
)
