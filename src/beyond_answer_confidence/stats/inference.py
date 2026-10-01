"""Tests and interval checks for effect sizes.

Bootstrap "tail shares" here are percentile-interval inversions (the share of
replicates on the far side of a reference value), not tests generated under
a null; the t, TOST, Fisher-z and sign-flip functions are proper tests.
"""

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from beyond_answer_confidence.metrics.ranking import auroc
from beyond_answer_confidence.stats.resampling import (
    bootstrap,
    bootstrap_means,
    percentile_ci,
    resample_indices,
)


def holm(pvalues: Sequence[float]) -> list[float]:
    """Holm step-down adjusted p-values.

    Args:
        pvalues: Raw p-values.

    Returns:
        Adjusted p-values in input order.
    """
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def holm_adjust(results: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Attach Holm-adjusted p-values to a family of results.

    Args:
        results: Name -> result with a ``p`` field.

    Returns:
        The results with ``p_holm`` added.
    """
    names = list(results)
    adjusted = holm([float(results[n]["p"]) for n in names])
    return {
        n: {**results[n], "p_holm": a} for n, a in zip(names, adjusted, strict=True)
    }


def tail_share(values: np.ndarray, threshold: float) -> float:
    """Share of bootstrap replicates at or above a threshold.

    Below alpha exactly when the one-sided (1 - alpha) percentile upper
    bound is below the threshold.

    Args:
        values: Bootstrap values.
        threshold: Reference value.

    Returns:
        The share.
    """
    return float(np.mean(values >= threshold))


def mean_interval(
    values: np.ndarray,
    b: int,
    rng: np.random.Generator,
    threshold: float | None = None,
) -> dict[str, Any]:
    """Mean with an item-bootstrap 95 % interval.

    Args:
        values: One value per unit.
        b: Resamples.
        rng: Random generator (one vectorised draw).
        threshold: Optional reference value; adds ``tail``, the share of
            replicates at or above it.

    Returns:
        ``n``, ``mean``, ``ci`` and optionally ``threshold`` and ``tail``.
    """
    boot = bootstrap_means(values, b, rng)
    out: dict[str, Any] = {
        "n": len(values),
        "mean": float(values.mean()),
        "ci": list(percentile_ci(boot)),
    }
    if threshold is not None:
        out["threshold"] = threshold
        out["tail"] = tail_share(boot, threshold)
    return out


def bootstrap_equivalence(
    values: np.ndarray, margin: float, b: int, rng: np.random.Generator
) -> dict[str, Any]:
    """Bootstrap TOST: is the mean paired difference within ``+/- margin``?

    Args:
        values: One paired difference per unit.
        margin: Equivalence margin.
        b: Resamples.
        rng: Random generator (one vectorised draw).

    Returns:
        ``mean``, 90 % interval ``ci90``, ``margin`` and the TOST tail
        share ``p``.
    """
    boot = bootstrap_means(values, b, rng)
    return {
        "n": len(values),
        "mean": float(values.mean()),
        "ci90": list(percentile_ci(boot, 0.90)),
        "margin": margin,
        "p": max(float(np.mean(boot >= margin)), float(np.mean(boot <= -margin))),
    }


def auroc_vs_threshold(
    scores: np.ndarray,
    positive: np.ndarray,
    threshold: float,
    b: int,
    rng: np.random.Generator,
    clusters: np.ndarray | None = None,
) -> dict[str, Any]:
    """AUROC with a (cluster) bootstrap interval and a tail share below a threshold.

    Args:
        scores: Higher = more likely positive.
        positive: Boolean labels.
        threshold: Reference AUROC (e.g. 0.5 for chance).
        b: Resamples.
        rng: Random generator.
        clusters: Cluster id per row (resampled together); defaults to rows.

    Returns:
        ``auroc``, 95 % ``ci``, ``threshold`` and ``p``, the share of
        replicates at or below the threshold.
    """
    if clusters is None:
        clusters = np.arange(len(scores))
    boot = np.array(
        [
            auroc(scores[i], positive[i])
            for i in resample_indices(len(scores), b, rng, clusters)
        ]
    )
    return {
        "n": len(scores),
        "auroc": auroc(scores, positive),
        "ci": list(percentile_ci(boot)),
        "threshold": threshold,
        "p": float(np.nanmean(boot <= threshold)),
    }


def t_below(values: np.ndarray, threshold: float) -> dict[str, Any]:
    """One-sided one-sample t-test that the mean is below a threshold.

    Args:
        values: One value per unit.
        threshold: Threshold.

    Returns:
        Mean, standard error and p-value.
    """
    from scipy import stats

    v = np.asarray(values, float)
    se = float(v.std(ddof=1) / math.sqrt(len(v)))
    t = (v.mean() - threshold) / se if se > 0 else -math.inf
    return {
        "n": len(v),
        "mean": float(v.mean()),
        "se": se,
        "p": float(stats.t.cdf(t, len(v) - 1)),
    }


def t_above(values: np.ndarray, threshold: float) -> dict[str, Any]:
    """One-sided one-sample t-test that the mean is above a threshold.

    Args:
        values: One value per unit.
        threshold: Threshold.

    Returns:
        Mean, standard error and p-value.
    """
    res = t_below(-np.asarray(values, float), -threshold)
    return res | {"mean": -res["mean"]}


def tost_t(values: np.ndarray, margin: float) -> dict[str, Any]:
    """Two one-sided t-tests that the mean lies within ``+/- margin``.

    Args:
        values: One value per unit.
        margin: Equivalence margin.

    Returns:
        Mean, standard error and TOST p-value (the larger one-sided p).
    """
    upper = t_below(values, margin)
    lower = t_above(values, -margin)
    return {
        "n": upper["n"],
        "mean": upper["mean"],
        "se": upper["se"],
        "p": max(upper["p"], lower["p"]),
    }


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation.

    Args:
        a: First variable.
        b: Second variable.

    Returns:
        The correlation (``nan`` if either is constant).
    """
    from scipy.stats import spearmanr

    return float(spearmanr(a, b).statistic)


def spearman_above_fisher(
    x: np.ndarray, y: np.ndarray, threshold: float
) -> dict[str, Any]:
    """One-sided test that Spearman's rho exceeds a threshold (Fisher z).

    Uses the Bonett-Wright standard error ``sqrt((1 + rho^2 / 2) / (n - 3))``.

    Args:
        x: First variable.
        y: Second variable.
        threshold: Threshold for rho.

    Returns:
        Rho, z statistic and p-value.
    """
    from scipy import stats

    rho = spearman(x, y)
    n = len(x)
    se = math.sqrt((1 + rho**2 / 2) / (n - 3))
    z = (math.atanh(rho) - math.atanh(threshold)) / se
    return {"n": n, "rho": rho, "z": z, "p": float(stats.norm.sf(z))}


def sign_flip_below(
    diffs: np.ndarray, b: int, rng: np.random.Generator, threshold: float = 0.0
) -> dict[str, Any]:
    """One-sided sign-flip permutation test that paired differences are < threshold.

    Under the null of a distribution symmetric about ``threshold`` the signs
    of ``diffs - threshold`` are exchangeable.

    Args:
        diffs: One paired difference per unit.
        b: Random sign patterns.
        rng: Random generator.
        threshold: Null centre.

    Returns:
        Mean difference and permutation p-value.
    """
    d = np.asarray(diffs, float) - threshold
    obs = d.mean()
    signs = rng.choice([-1.0, 1.0], size=(b, len(d)))
    null = (signs * d).mean(axis=1)
    return {
        "n": len(d),
        "mean": float(obs + threshold),
        "p": float((1 + np.sum(null <= obs)) / (b + 1)),
    }


def mean_difference_interval(
    a: np.ndarray, c: np.ndarray, b: int, rng: np.random.Generator
) -> dict[str, Any]:
    """Difference of two independent group means, with a bootstrap interval.

    Draw order: one vectorised draw for ``a``, then one for ``c`` (see
    :func:`~beyond_answer_confidence.stats.resampling.bootstrap_means`).

    Args:
        a: Values of the first group.
        c: Values of the second group.
        b: Resamples.
        rng: Random generator.

    Returns:
        ``mean_a``, ``mean_b`` (the second group), ``diff``, 95 % ``ci``
        and ``tail``, the share of replicates with a difference at or
        above 0.
    """
    d = bootstrap_means(a, b, rng) - bootstrap_means(c, b, rng)
    return {
        "mean_a": float(a.mean()),
        "mean_b": float(c.mean()),
        "diff": float(a.mean() - c.mean()),
        "ci": list(percentile_ci(d)),
        "tail": tail_share(d, 0.0),
    }


def spearman_interval(
    x: np.ndarray,
    y: np.ndarray,
    b: int,
    rng: np.random.Generator,
    threshold: float | None = None,
) -> dict[str, Any]:
    """Spearman's rho with an item-bootstrap 95 % interval.

    Draw order: one ``rng.integers(0, n, n)`` call per resample (as in
    :func:`~beyond_answer_confidence.stats.resampling.bootstrap`).

    Args:
        x: First variable.
        y: Second variable.
        b: Resamples.
        rng: Random generator.
        threshold: Optional reference value; adds ``tail``, the share of
            replicates at or below it.

    Returns:
        ``n``, ``rho``, ``ci`` and optionally ``threshold`` and ``tail``.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    boot = bootstrap(len(x), lambda i: spearman(x[i], y[i]), b, rng)
    out: dict[str, Any] = {
        "n": len(x),
        "rho": spearman(x, y),
        "ci": list(percentile_ci(boot)),
    }
    if threshold is not None:
        out["threshold"] = threshold
        out["tail"] = float(np.mean(boot <= threshold))
    return out


def holm_adjust_tails(
    results: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Attach Holm-adjusted values to a family of tests or tail shares.

    Like :func:`holm_adjust`, but each result may carry its value as ``p``
    (tests, equivalence checks) or, failing that, ``tail`` (interval
    inversions such as :func:`mean_interval` with a threshold).

    Args:
        results: Name -> result with a ``p`` or ``tail`` field.

    Returns:
        The results with ``p_holm`` added, in input order.
    """
    names = list(results)
    values = [float(results[n].get("p", results[n].get("tail"))) for n in names]
    return {
        n: {**results[n], "p_holm": a} for n, a in zip(names, holm(values), strict=True)
    }


def mean_interval_lower(
    values: np.ndarray, b: int, rng: np.random.Generator, threshold: float
) -> dict[str, Any]:
    """Mean with an item-bootstrap 95 % interval and a lower tail share.

    The mirror image of :func:`mean_interval` with a threshold: ``tail`` is
    the share of replicates at or *below* the threshold, small when the
    mean is clearly above it.

    Args:
        values: One value per unit.
        b: Resamples.
        rng: Random generator (one vectorised draw).
        threshold: Reference value.

    Returns:
        ``n``, ``mean``, ``ci``, ``threshold`` and ``tail``.
    """
    boot = bootstrap_means(values, b, rng)
    return {
        "n": len(values),
        "mean": float(values.mean()),
        "ci": list(percentile_ci(boot)),
        "threshold": threshold,
        "tail": float(np.mean(boot <= threshold)),
    }
