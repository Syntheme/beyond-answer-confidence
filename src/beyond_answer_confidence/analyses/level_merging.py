"""Recalibration by merging confidence levels, before versus after a cutoff.

A recalibration map that depends on confidence alone gives every item with
the same (or a merged group of) confidence value(s) one output. If accuracy
at equal confidence differs between two periods, no such map can be
calibrated in both. This module measures how much period-conditional error
remains for maps that merge levels:

- :func:`best_interval_partition`: the exact minimum over maps whose level
  sets are intervals of confidence (this covers every monotone map);
- :func:`search_groupings`: a local search over arbitrary groupings;
- :func:`recalibration_groupings`: both, in sample (with a random-month
  null) and cross-fitted on month half-splits;
- :func:`forward_chaining`: isotonic recalibration fitted on earlier months
  only.

All functions are pure; randomness comes from a passed generator.
"""

from typing import Any

import numpy as np
import pandas as pd

from beyond_answer_confidence.tasks.follow_up import CUTOFF_MONTH

LevelCounts = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
"""``(values, n_pre, k_pre, n_post, k_post)`` per distinct confidence value."""


def level_counts(p: np.ndarray, correct: np.ndarray, post: np.ndarray) -> LevelCounts:
    """Per distinct confidence value: counts and correct counts in each period.

    Args:
        p: Confidences (rounded to 1e-9 to merge float noise).
        correct: Correctness.
        post: Whether each item is in the later period.

    Returns:
        ``(values, n_pre, k_pre, n_post, k_post)`` sorted by value.
    """
    v = np.round(np.asarray(p, float), 9)
    ok, po = np.asarray(correct, bool), np.asarray(post, bool)
    values, inv = np.unique(v, return_inverse=True)
    n_pre = np.bincount(inv, weights=~po, minlength=len(values))
    k_pre = np.bincount(inv, weights=ok & ~po, minlength=len(values))
    n_post = np.bincount(inv, weights=po, minlength=len(values))
    k_post = np.bincount(inv, weights=ok & po, minlength=len(values))
    return values, n_pre, k_pre, n_post, k_post


def group_cost(n_pre: float, k_pre: float, n_post: float, k_post: float) -> float:
    """Least period-conditional calibration error of one output level.

    For one output value ``c`` shared by a group of original values, the
    error is ``n_pre |c - a_pre| + n_post |c - a_post|``; its minimum over
    ``c`` is ``min(n_pre, n_post) |a_pre - a_post|``.

    Args:
        n_pre: Items before.
        k_pre: Correct before.
        n_post: Items after.
        k_post: Correct after.

    Returns:
        The minimum (a count of items, not yet divided by N).
    """
    if n_pre == 0 or n_post == 0:
        return 0.0
    return float(min(n_pre, n_post) * abs(k_pre / n_pre - k_post / n_post))


def best_interval_partition(
    n_pre: np.ndarray, k_pre: np.ndarray, n_post: np.ndarray, k_post: np.ndarray
) -> tuple[float, list[tuple[int, int]]]:
    """Exact minimum error over maps whose level sets are intervals of values.

    Every monotone map has interval level sets, so this covers all monotone
    recalibrators (and non-monotone ones whose levels are contiguous).
    Dynamic programming over the sorted distinct values.

    Args:
        n_pre: Per-value counts before.
        k_pre: Per-value correct before.
        n_post: Per-value counts after.
        k_post: Per-value correct after.

    Returns:
        ``(error count, groups)``; groups are half-open index ranges.
    """
    m = len(n_pre)
    cp = [np.concatenate([[0.0], np.cumsum(a)]) for a in (n_pre, k_pre, n_post, k_post)]
    best = np.full(m + 1, np.inf)
    best[0] = 0.0
    back = np.zeros(m + 1, int)
    for j in range(1, m + 1):
        for i in range(j):
            c = best[i] + group_cost(*(a[j] - a[i] for a in cp))
            if c < best[j]:
                best[j], back[j] = c, i
    groups, j = [], m
    while j > 0:
        groups.append((int(back[j]), j))
        j = int(back[j])
    return float(best[m]), groups[::-1]


def search_groupings(
    n_pre: np.ndarray,
    k_pre: np.ndarray,
    n_post: np.ndarray,
    k_post: np.ndarray,
    n_groups: int,
    rng: np.random.Generator,
    sweeps: int = 30,
) -> tuple[float, np.ndarray]:
    """Local search over arbitrary (non-contiguous) groupings of values.

    Starts from equal-mass contiguous groups and moves single values between
    groups while the total error falls. A heuristic upper bound on the best
    in-sample grouping, not an exact optimum.

    Args:
        n_pre: Per-value counts before.
        k_pre: Per-value correct before.
        n_post: Per-value counts after.
        k_post: Per-value correct after.
        n_groups: Number of output levels.
        rng: Random generator (one permutation of the values per sweep).
        sweeps: Maximum passes over all values.

    Returns:
        ``(error count, group label per value)``.
    """
    m = len(n_pre)
    mass = np.cumsum(n_pre + n_post)
    label = np.minimum((mass / mass[-1] * n_groups).astype(int), n_groups - 1)
    tot = np.zeros((n_groups, 4))
    stats = np.column_stack([n_pre, k_pre, n_post, k_post])
    for g in range(n_groups):
        tot[g] = stats[label == g].sum(axis=0)

    def cost(row: np.ndarray) -> float:
        return group_cost(*row)

    for _ in range(sweeps):
        moved = False
        for v in rng.permutation(m):
            g0 = label[v]
            base = cost(tot[g0])
            without = cost(tot[g0] - stats[v])
            best_gain, best_g = 0.0, g0
            for g in range(n_groups):
                if g == g0:
                    continue
                gain = base + cost(tot[g]) - without - cost(tot[g] + stats[v])
                if gain > best_gain + 1e-12:
                    best_gain, best_g = gain, g
            if best_g != g0:
                tot[g0] -= stats[v]
                tot[best_g] += stats[v]
                label[v] = best_g
                moved = True
        if not moved:
            break
    return float(sum(cost(t) for t in tot)), label


def apply_levels(label: np.ndarray, train: np.ndarray, test: np.ndarray) -> float:
    """Held-out period-conditional error of a grouping fitted on ``train``.

    Each group's output is the training value minimising the training error
    (the accuracy of the larger period). A group with no training items (its
    confidence values never occur in the training months) gets the overall
    training accuracy, so every held-out item is scored.

    Args:
        label: Group per distinct value.
        train: Per-value ``(n_pre, k_pre, n_post, k_post)`` counts to fit on.
        test: The same counts to evaluate on.

    Returns:
        ``sum_g n_pre |c_g - a_pre| + n_post |c_g - a_post|`` on ``test``
        (a count of items).
    """
    tot = train.sum(axis=0)
    fallback = (tot[1] + tot[3]) / max(tot[0] + tot[2], 1)
    err = 0.0
    for g in np.unique(label):
        fit = train[label == g].sum(axis=0)
        held = test[label == g].sum(axis=0)
        if held[0] + held[2] == 0:
            continue
        if fit[0] + fit[2] == 0:
            c = fallback
        else:
            a_pre = fit[1] / fit[0] if fit[0] else fit[3] / fit[2]
            a_post = fit[3] / fit[2] if fit[2] else a_pre
            c = a_pre if fit[0] >= fit[2] else a_post
        if held[0]:
            err += held[0] * abs(c - held[1] / held[0])
        if held[2]:
            err += held[2] * abs(c - held[3] / held[2])
    return float(err)


def recalibration_groupings(
    df: pd.DataFrame, rng: np.random.Generator, splits: int = 50, null_reps: int = 200
) -> dict[str, Any]:
    """Merging maps: exact interval optimum, grouping search, cross-fitting, nulls.

    Draw order: the in-sample grouping searches (2, 3, 5, 10, 20 levels), the
    random-month null, then per split the month halves followed by the
    5-, 10- and 20-level searches in each direction.

    Args:
        df: News rows with ``p_max``, ``correct``, ``post`` and ``month``.
        rng: Random generator.
        splits: Month half-splits for cross-fitting (each used both ways).
        null_reps: Random-month nulls for the in-sample interval optimum.

    Returns:
        Mean absolute period-conditional calibration errors (divided by N):
        value-wise bound, interval optimum (with its null distribution) and
        grouping searches in sample; cross-fitted errors and the held-out
        mean gap after the cutoff for the interval map.
    """
    p, ok, post = (
        df["p_max"].to_numpy(),
        df["correct"].to_numpy(bool),
        df["post"].to_numpy(),
    )
    values, *counts = level_counts(p, ok, post)
    n = len(df)
    stats = np.column_stack(counts)
    valuewise = sum(group_cost(*row) for row in stats) / n
    interval, groups = best_interval_partition(*counts)
    searched = {
        g: search_groupings(counts[0], counts[1], counts[2], counts[3], g, rng)[0] / n
        for g in (2, 3, 5, 10, 20)
    }
    months = df["month"].to_numpy()
    uniq = np.unique(months)
    n_post_months = len(np.unique(months[post]))
    null_interval = []
    for _ in range(null_reps):
        fake = np.isin(months, rng.choice(uniq, n_post_months, replace=False))
        _, *c = level_counts(p, ok, fake)
        null_interval.append(best_interval_partition(*c)[0] / n)
    idx = np.searchsorted(values, np.round(p, 9))

    def counts_for(mask: np.ndarray) -> np.ndarray:
        i, o, q = idx[mask], ok[mask], post[mask]
        return np.column_stack(
            [
                np.bincount(i, weights=w, minlength=len(values))
                for w in (~q, o & ~q, q, o & q)
            ]
        )

    pre_m, post_m = np.unique(months[~post]), np.unique(months[post])
    cross: dict[str, list[float]] = {"valuewise": [], "interval": []}
    for g in (5, 10, 20):
        cross[f"search_{g}"] = []
    cross_post_gap: list[float] = []
    for _ in range(splits):
        half = set(rng.choice(pre_m, len(pre_m) // 2, replace=False)) | set(
            rng.choice(post_m, len(post_m) // 2, replace=False)
        )
        in_a = np.array([m in half for m in months])
        for fit_mask, held_mask in ((in_a, ~in_a), (~in_a, in_a)):
            fit, held = counts_for(fit_mask), counts_for(held_mask)
            held_n = held[:, 0].sum() + held[:, 2].sum()
            cross["valuewise"].append(
                apply_levels(np.arange(len(values)), fit, held) / held_n
            )
            _, grp = best_interval_partition(*fit.T)
            lab = np.zeros(len(values), int)
            for gi, (a, bnd) in enumerate(grp):
                lab[a:bnd] = gi
            cross["interval"].append(apply_levels(lab, fit, held) / held_n)
            for g in (5, 10, 20):
                _, lab_s = search_groupings(
                    fit[:, 0], fit[:, 1], fit[:, 2], fit[:, 3], g, rng
                )
                cross[f"search_{g}"].append(apply_levels(lab_s, fit, held) / held_n)
            # held-out mean gap after the cutoff for the interval map
            gap_num, gap_den = 0.0, 0.0
            for gi in np.unique(lab):
                t = fit[lab == gi].sum(axis=0)
                e = held[lab == gi].sum(axis=0)
                if t[0] + t[2] == 0 or e[2] == 0:
                    continue
                c_val = (t[1] + t[3]) / (t[0] + t[2]) if t[0] < t[2] else t[1] / t[0]
                gap_num += e[2] * c_val - e[3]
                gap_den += e[2]
            cross_post_gap.append(gap_num / gap_den if gap_den else float("nan"))
    return {
        "n": n,
        "distinct_values": len(values),
        "in_sample": {
            "valuewise_bound": valuewise,
            "interval_partition_exact": interval / n,
            "interval_groups": len(groups),
            "interval_null_random_months": {
                "mean": float(np.mean(null_interval)),
                "p95": float(np.percentile(null_interval, 95)),
                "p": float(
                    (1 + np.sum(np.array(null_interval) >= interval / n))
                    / (len(null_interval) + 1)
                ),
            },
            "grouping_search": searched,
        },
        "cross_fitted": {
            k: {
                "mean": float(np.mean(v)),
                "range": [float(np.min(v)), float(np.max(v))],
            }
            for k, v in cross.items()
        },
        "cross_fitted_interval_post_gap": {
            "mean": float(np.nanmean(cross_post_gap)),
            "range": [
                float(np.nanmin(cross_post_gap)),
                float(np.nanmax(cross_post_gap)),
            ],
        },
    }


def forward_chaining(
    df: pd.DataFrame, min_train_months: int = 12, cutoff_month: str = CUTOFF_MONTH
) -> dict[str, Any]:
    """Recalibrate on earlier months only and evaluate on later ones.

    Two schemes: *expanding* (fit isotonic regression on all months before
    ``t``, test month ``t``) and *frozen* (fit once on the months before the
    cutoff, test every later month), the deployment case.

    Args:
        df: News rows with ``p_max``, ``correct``, ``post`` and ``month``.
        min_train_months: The first test month needs this many earlier months.
        cutoff_month: First month counted as after the cutoff.

    Returns:
        Mean calibration gap (recalibrated confidence minus accuracy) by
        period and month.
    """
    from sklearn.isotonic import IsotonicRegression

    months = sorted(df["month"].unique())

    def fit(train: pd.DataFrame) -> Any:
        return IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(
            train["p_max"].to_numpy(), train["correct"].to_numpy(float)
        )

    rows = []
    for i, m in enumerate(months):
        if i < min_train_months:
            continue
        test = df[df["month"] == m]
        f = fit(df[df["month"] < m])
        rows.append(
            {
                "month": m,
                "post": m >= cutoff_month,
                "gap": float(
                    f.predict(test["p_max"].to_numpy()).mean() - test["correct"].mean()
                ),
                "raw_gap": float(test["p_max"].mean() - test["correct"].mean()),
            }
        )
    exp = pd.DataFrame(rows)
    frozen_fit = fit(df[~df["post"]])
    later = df[df["post"]]
    grouped: Any = later.groupby("month")
    by_month = grouped.apply(
        lambda g: float(
            frozen_fit.predict(g["p_max"].to_numpy()).mean() - g["correct"].mean()
        ),
        include_groups=False,
    )
    first_post = exp[exp["post"]].head(3)
    return {
        "expanding": {
            "pre_mean_gap": float(exp[~exp["post"]]["gap"].mean()),
            "post_mean_gap": float(exp[exp["post"]]["gap"].mean()),
            "first_three_post_months_gap": [float(x) for x in first_post["gap"]],
            "last_three_post_months_gap": [
                float(x) for x in exp[exp["post"]].tail(3)["gap"]
            ],
            "raw_post_mean_gap": float(exp[exp["post"]]["raw_gap"].mean()),
        },
        "frozen_before_cutoff": {
            "post_mean_gap": float(
                frozen_fit.predict(later["p_max"].to_numpy()).mean()
                - later["correct"].mean()
            ),
            "by_month_range": [float(by_month.min()), float(by_month.max())],
        },
        "by_month": exp.to_dict(orient="records"),
    }
