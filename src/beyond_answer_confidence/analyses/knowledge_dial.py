"""Analysis of the knowledge dial: how uncertainty moves with knowledge.

The unit of analysis is the replicate-mean distribution per item and
condition in the main arm; all intervals are item-bootstrap intervals.

Reported per dataset:

- per-condition means with 95 % intervals (accuracy, P(gold), top-label
  probability, normalised entropy, log loss, Brier, confident-wrong rate)
  and top-label SmoothECE with a bias-recentred interval;
- the dose-response of normalised entropy on ``log2(k + 1)`` examples: a
  mixed model (random intercept per intent, variance component per item)
  and the item-bootstrap mean of within-item OLS slopes (the same estimand
  in this balanced design), plus the paired effect of adding examples to
  intent names;
- behaviour with no knowledge (``L0``): chance level, mean top-label
  probability and the confident-wrong rate, with intervals;
- the length control: paired entropy difference between irrelevant filler
  and no information (``Lirrel - L0``);
- design checks: letter codes vs ordinal codes, other example sets, the
  swap manipulation check and replicate noise.

Every bootstrap draws from one generator in a fixed order (conditions, then
dose-response, no-knowledge and length control, dataset by dataset), so a
given seed reproduces every interval exactly.
"""

import math
import warnings
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from beyond_answer_confidence.metrics.calibration import smece
from beyond_answer_confidence.stats.resampling import (
    bootstrap,
    bootstrap_means,
    percentile_ci,
)

CONFIDENT = 0.5
"""Top-label probability at or above which a wrong answer counts as confident."""
DOSE = {"L0": 0, "L1": 1, "L2": 2, "L4": 4, "L8": 8}
"""Examples per intent in the dose-response conditions."""
CONDITION_ORDER = ("L0", "L1", "L2", "L4", "L8", "Lname", "Lname+8", "Lirrel", "Lswap")
METRICS = ("correct", "p_gold", "p_max", "norm_entropy", "nll", "brier")
CELL_KEYS = ["arm", "item", "condition", "code_style", "example_seed"]


def _cell(key: tuple[Any, ...], grp: pd.DataFrame) -> dict[str, Any]:
    """Replicate-mean statistics of one item x cell."""
    first = grp.iloc[0]
    labels = list(first["probs"])
    mat = np.array([[p[lab] for lab in labels] for p in grp["probs"]], dtype=float)
    mean = mat.mean(axis=0)
    total = mean.sum()
    mean = mean / total if total > 0 else np.full(len(labels), 1 / len(labels))
    gold = first["gold"]
    g = labels.index(gold)
    top = int(np.argmax(mean))
    k = len(labels)
    nz = mean[mean > 0]
    onehot = np.zeros(k)
    onehot[g] = 1.0
    swap = first["swap_target"]
    return {
        **dict(zip(CELL_KEYS, key, strict=True)),
        "gold": gold,
        "K": k,
        "n_reps": len(grp),
        "p_gold": float(mean[g]),
        "p_max": float(mean[top]),
        "correct": top == g,
        "chose_first_listed": labels[top] == first["first_listed"],
        "chose_swap_target": swap is not None and labels[top] == swap,
        "norm_entropy": float(-(nz * np.log(nz)).sum() / math.log(k)),
        "nll": float(-math.log(max(mean[g], 1e-12))),
        "brier": float(((mean - onehot) ** 2).sum()),
        "rep_sd_p_gold": float(mat[:, g].std(ddof=1)) if len(grp) > 1 else math.nan,
    }


def aggregate(rows: pd.DataFrame) -> pd.DataFrame:
    """Average replicates into one distribution per item x cell.

    Args:
        rows: One row per answered request (knowledge-dial row format).

    Returns:
        One row per (arm, item, condition, code_style, example_seed) with the
        replicate-mean distribution's statistics.
    """
    return pd.DataFrame(
        [_cell(key, grp) for key, grp in rows.groupby(CELL_KEYS, sort=False)]
    )


def confident_wrong(cells: pd.DataFrame) -> np.ndarray:
    """Per-item indicator of a confident (``p_max >= 0.5``) wrong top label.

    Args:
        cells: Aggregated cells.

    Returns:
        0/1 floats.
    """
    flags: np.ndarray = ((cells["p_max"] >= CONFIDENT) & ~cells["correct"]).to_numpy(
        float
    )
    return flags


def _smece_entry(
    pm: np.ndarray, ok: np.ndarray, b: int, rng: np.random.Generator
) -> dict[str, Any]:
    """SmoothECE with an item-bootstrap interval recentred on the estimate.

    Resampling with replacement duplicates points, which inflates kernel-
    smoothed calibration error; the bootstrap is shifted by its mean bias.
    """
    est = smece(pm, ok)
    raw = bootstrap(len(pm), lambda idx: smece(pm[idx], ok[idx]), b, rng)
    boot = raw - (raw.mean() - est)
    return {
        "value": est,
        "ci95": list(percentile_ci(boot)),
        "bootstrap_bias": float(raw.mean() - est),
    }


def condition_table(
    cells: pd.DataFrame, b: int, rng: np.random.Generator
) -> dict[str, Any]:
    """Per-condition metrics with item-bootstrap 95 % intervals.

    Draw order per condition: one vectorised draw per metric (in
    :data:`METRICS` order, then the confident-wrong rate), then ``b`` draws
    for SmoothECE.

    Args:
        cells: Aggregated main-arm cells for one dataset.
        b: Bootstrap resamples.
        rng: Random generator.

    Returns:
        Metrics keyed by condition.
    """
    table: dict[str, Any] = {}
    for cond in CONDITION_ORDER:
        c = cells[cells["condition"] == cond].reset_index(drop=True)
        if c.empty:
            continue
        arrays = {m: c[m].to_numpy(float) for m in METRICS}
        arrays["confident_wrong"] = confident_wrong(c)
        entry: dict[str, Any] = {"n": len(c)}
        for name, arr in arrays.items():
            entry[name] = {
                "mean": float(arr.mean()),
                "ci95": list(percentile_ci(bootstrap_means(arr, b, rng))),
            }
        entry["smece"] = _smece_entry(arrays["p_max"], arrays["correct"], b, rng)
        table[cond] = entry
    return table


def within_item_slopes(cells: pd.DataFrame) -> np.ndarray:
    """OLS slope of normalised entropy on ``log2(k + 1)`` for each item.

    With every item observed at every dose (balanced design), the mean of
    these slopes equals the fixed-effect slope of a random-intercept model.
    The slope is per unit of ``log2(k + 1)``, not per doubling of k, and it
    is dominated by the step from no examples to one.

    Args:
        cells: Aggregated main-arm cells.

    Returns:
        One slope per item observed at all five doses.
    """
    d = cells[cells["condition"].isin(DOSE)]
    wide = d.pivot_table(
        index="item", columns="condition", values="norm_entropy"
    ).dropna()
    x = np.log2(np.array([DOSE[c] for c in wide.columns]) + 1)
    xc = x - x.mean()
    y = wide.to_numpy(float)
    slopes: np.ndarray = (y - y.mean(axis=1, keepdims=True)) @ xc / (xc @ xc)
    return slopes


def paired_diff(cells: pd.DataFrame, a: str, b_cond: str) -> np.ndarray:
    """Per-item difference in normalised entropy, condition ``a`` minus ``b_cond``.

    Args:
        cells: Aggregated main-arm cells.
        a: Minuend condition.
        b_cond: Subtrahend condition.

    Returns:
        One difference per item observed under both.
    """
    wide = (
        cells[cells["condition"].isin([a, b_cond])]
        .pivot_table(index="item", columns="condition", values="norm_entropy")
        .dropna()
    )
    diff: np.ndarray = (wide[a] - wide[b_cond]).to_numpy(float)
    return diff


def fit_mixed(cells: pd.DataFrame) -> dict[str, Any]:
    """Fit the dose-response mixed model and record convergence problems.

    ``norm_entropy ~ log2(k + 1)`` over the five dose conditions, with a
    random intercept per intent and a variance component per item.

    Args:
        cells: Aggregated main-arm cells.

    Returns:
        Slope, Wald 95 % interval, one-sided p-value for a negative slope,
        and convergence diagnostics.
    """
    import statsmodels.formula.api as smf
    from scipy.stats import norm
    from statsmodels.tools.sm_exceptions import ConvergenceWarning

    d = cells[cells["condition"].isin(DOSE)].copy()
    d["x"] = np.log2(d["condition"].map(DOSE) + 1)
    d["item"] = d["item"].astype(str)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = smf.mixedlm(
            "norm_entropy ~ x",
            d,
            groups=d["gold"],
            re_formula="1",
            vc_formula={"item": "0 + C(item)"},
        ).fit(reml=True)
    problems = sorted(
        {
            str(w.message)[:80]
            for w in caught
            if issubclass(w.category, ConvergenceWarning | UserWarning)
        }
    )
    slope, se = float(model.params["x"]), float(model.bse["x"])
    return {
        "slope": slope,
        "ci95": [slope - 1.96 * se, slope + 1.96 * se],
        "p_slope_below_zero": float(norm.cdf(slope / se)),
        "converged_cleanly": bool(getattr(model, "converged", False)) and not problems,
        "warnings": problems,
    }


def mean_with_ci(
    values: np.ndarray, b: int, rng: np.random.Generator
) -> dict[str, Any]:
    """Mean with an item-bootstrap 95 % interval (one vectorised draw).

    Args:
        values: One value per item.
        b: Resamples.
        rng: Random generator.

    Returns:
        ``mean``, ``ci95`` and ``n``.
    """
    boot = bootstrap_means(values, b, rng)
    return {
        "mean": float(values.mean()),
        "ci95": list(percentile_ci(boot)),
        "n": len(values),
    }


def dose_response(
    cells: pd.DataFrame, b: int, rng: np.random.Generator
) -> dict[str, Any]:
    """Entropy against the number of examples, and names with vs without examples.

    Args:
        cells: Aggregated main-arm cells for one dataset.
        b: Bootstrap resamples.
        rng: Random generator (two vectorised draws).

    Returns:
        The mixed model, the item-bootstrap slope and the paired
        ``Lname+8 - Lname`` entropy difference.
    """
    mixed = fit_mixed(cells)
    slopes = within_item_slopes(cells)
    slope = mean_with_ci(slopes, b, rng)
    names = mean_with_ci(paired_diff(cells, "Lname+8", "Lname"), b, rng)
    return {
        "mixed_model": mixed,
        "item_bootstrap": {
            "slope": slope["mean"],
            "ci95": slope["ci95"],
            "n_items": slope["n"],
        },
        "names_plus_examples_minus_names": {
            "mean": names["mean"],
            "ci95": names["ci95"],
        },
    }


def no_knowledge(
    cells: pd.DataFrame, b: int, rng: np.random.Generator
) -> dict[str, Any]:
    """Behaviour when codes come with no information (``L0``).

    Args:
        cells: Aggregated main-arm cells for one dataset.
        b: Bootstrap resamples.
        rng: Random generator (two vectorised draws).

    Returns:
        Chance level ``1/K``, mean top-label probability and confident-wrong
        rate with 95 % intervals.
    """
    c = cells[cells["condition"] == "L0"].reset_index(drop=True)
    pmax = mean_with_ci(c["p_max"].to_numpy(float), b, rng)
    cw = mean_with_ci(confident_wrong(c), b, rng)
    return {
        "chance": 1 / int(c["K"].iloc[0]),
        "mean_p_max": pmax["mean"],
        "mean_p_max_ci95": pmax["ci95"],
        "confident_wrong_rate": cw["mean"],
        "confident_wrong_rate_ci95": cw["ci95"],
    }


def length_control(
    cells: pd.DataFrame, b: int, rng: np.random.Generator
) -> dict[str, Any]:
    """Paired entropy difference, irrelevant filler minus no information.

    Args:
        cells: Aggregated main-arm cells for one dataset.
        b: Bootstrap resamples.
        rng: Random generator (one vectorised draw).

    Returns:
        Mean difference with 90 % and 95 % intervals.
    """
    diff = paired_diff(cells, "Lirrel", "L0")
    boot = bootstrap_means(diff, b, rng)
    return {
        "mean_diff": float(diff.mean()),
        "ci90": list(percentile_ci(boot, 0.90)),
        "ci95": list(percentile_ci(boot)),
    }


def _summary(df: pd.DataFrame) -> dict[str, Any]:
    """Per-condition accuracy, top-label probability, entropy, first-listed share."""
    return {
        str(cond): {
            "n": len(g),
            "accuracy": float(g["correct"].mean()),
            "p_max": float(g["p_max"].mean()),
            "norm_entropy": float(g["norm_entropy"].mean()),
            "chose_first_listed": float(g["chose_first_listed"].mean()),
        }
        for cond, g in df.groupby("condition")
    }


def design_checks(agg: pd.DataFrame) -> dict[str, Any]:
    """Descriptive checks of the design (no resampling).

    Args:
        agg: All aggregated cells for one dataset.

    Returns:
        Letter-code arm vs the same items with ordinal codes, the seed arm
        next to the main arm, the swap manipulation check and the median
        replicate spread of P(gold).
    """
    main = agg[agg["arm"] == "main"]
    letters = agg[agg["arm"] == "letters"]
    seeds = agg[agg["arm"] == "seeds"]
    swap = main[main["condition"] == "Lswap"]
    same_items = main[
        main["item"].isin(set(letters["item"]))
        & main["condition"].isin(set(letters["condition"]))
    ]
    seed_frame = pd.concat(
        [
            main[
                main["item"].isin(set(seeds["item"]))
                & main["condition"].isin(set(seeds["condition"]))
            ],
            seeds,
        ]
    )
    return {
        "letters_arm": _summary(letters),
        "ordinal_same_items": _summary(same_items),
        "seed_arm": {
            str(s): _summary(g) for s, g in seed_frame.groupby("example_seed")
        },
        "swap_chose_swap_target": float(swap["chose_swap_target"].mean())
        if len(swap)
        else math.nan,
        "replicate_sd_p_gold_median": float(main["rep_sd_p_gold"].median()),
    }


def analyse_dataset(
    rows: pd.DataFrame, b: int, rng: np.random.Generator
) -> dict[str, Any]:
    """Full analysis of one dataset's rows.

    Args:
        rows: Knowledge-dial rows (one per request).
        b: Bootstrap resamples.
        rng: Random generator, shared across datasets.

    Returns:
        JSON-serialisable results.
    """
    agg = aggregate(rows)
    main = agg[agg["arm"] == "main"].reset_index(drop=True)
    table = condition_table(main, b, rng)
    dose = dose_response(main, b, rng)
    ignorance = no_knowledge(main, b, rng)
    length = length_control(main, b, rng)
    return {
        "models": sorted(set(rows["model"])),
        "conditions": table,
        "dose_response": dose,
        "no_knowledge": ignorance,
        "length_control": length,
        "design_checks": design_checks(agg),
    }


def analyse(
    rows: Mapping[str, pd.DataFrame], b: int = 2000, seed: int = 0
) -> dict[str, Any]:
    """Analyse several datasets with one generator.

    Args:
        rows: Knowledge-dial rows keyed by dataset name (analysed in order).
        b: Bootstrap resamples.
        seed: Bootstrap seed.

    Returns:
        Results keyed by dataset.
    """
    rng = np.random.default_rng(seed)
    return {name: analyse_dataset(df, b, rng) for name, df in rows.items()}
