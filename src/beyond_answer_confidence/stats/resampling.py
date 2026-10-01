"""Bootstrap resampling: items, clusters, month blocks, BCa.

All functions take an explicit ``np.random.Generator`` (or seed) and draw
from it in a documented order, so results are reproducible bit for bit.
"""

from collections.abc import Callable
from typing import Any

import numpy as np

from beyond_answer_confidence.metrics.calibration import smece
from beyond_answer_confidence.metrics.ranking import auroc


def percentile_ci(values: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    """Two-sided percentile interval (NaNs ignored).

    Args:
        values: Bootstrap values.
        level: Coverage.

    Returns:
        ``(low, high)``.
    """
    lo, hi = np.nanpercentile(values, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(lo), float(hi)


def resample_indices(
    n: int, b: int, rng: np.random.Generator, clusters: np.ndarray | None = None
) -> list[np.ndarray]:
    """Row indices for ``b`` item or cluster bootstrap resamples.

    Draw order: one ``rng.integers`` call per resample, over rows (``n``) or
    over distinct clusters.

    Args:
        n: Number of rows.
        b: Resamples.
        rng: Random generator.
        clusters: Optional cluster id per row; whole clusters are resampled.

    Returns:
        One index array per resample.
    """
    if clusters is None:
        return [rng.integers(0, n, n) for _ in range(b)]
    ids, inverse = np.unique(clusters, return_inverse=True)
    members = [np.flatnonzero(inverse == i) for i in range(len(ids))]
    return [
        np.concatenate([members[j] for j in rng.integers(0, len(ids), len(ids))])
        for _ in range(b)
    ]


def bootstrap(
    n: int, stat: Callable[[np.ndarray], float], b: int, rng: np.random.Generator
) -> np.ndarray:
    """Bootstrap a statistic of resampled item indices.

    Args:
        n: Number of items.
        stat: Maps an index array (resampled items) to a value.
        b: Resamples.
        rng: Random generator.

    Returns:
        ``b`` bootstrap values.
    """
    return np.array([stat(rng.integers(0, n, n)) for _ in range(b)], dtype=float)


def bootstrap_means(values: np.ndarray, b: int, rng: np.random.Generator) -> np.ndarray:
    """Bootstrap the mean of per-item values (one vectorised draw).

    Draw order: a single ``rng.integers(0, n, (b, n))`` call. This is *not*
    the same stream as ``b`` separate draws; use :func:`bootstrap` for that.

    Args:
        values: One value per item.
        b: Resamples.
        rng: Random generator.

    Returns:
        ``b`` bootstrap means.
    """
    idx = rng.integers(0, len(values), (b, len(values)))
    means: np.ndarray = values[idx].mean(axis=1)
    return means


def cluster_bootstrap(
    values: np.ndarray,
    clusters: np.ndarray,
    b: int,
    rng: np.random.Generator,
    stat: Callable[[np.ndarray], float] = np.mean,
) -> np.ndarray:
    """Bootstrap a statistic of pooled rows by resampling whole clusters.

    Args:
        values: One value per row.
        clusters: Cluster id per row.
        b: Resamples.
        rng: Random generator.
        stat: Statistic of the pooled rows.

    Returns:
        Bootstrap values.
    """
    v = np.asarray(values, float)
    return np.array(
        [stat(v[i]) for i in resample_indices(len(v), b, rng, clusters)], dtype=float
    )


def auroc_interval(
    score: np.ndarray,
    positive: np.ndarray,
    b: int,
    rng: np.random.Generator,
    clusters: np.ndarray | None = None,
    level: float = 0.95,
) -> dict[str, Any]:
    """AUROC with a percentile bootstrap interval (items or clusters).

    The bounds are taken at ``50 -/+ 50 * level`` percent, which is exact for
    the usual levels (2.5 and 97.5 at 0.95); :func:`percentile_ci` derives
    them differently and can differ in the last bit.

    Args:
        score: Higher = more likely positive.
        positive: Labels.
        b: Resamples.
        rng: Random generator.
        clusters: Optional cluster id per row.
        level: Interval coverage.

    Returns:
        ``auroc``, ``ci`` (low, high), ``n`` and ``n_positive``.
    """
    s, y = np.asarray(score, float), np.asarray(positive, bool)
    boot = np.array(
        [auroc(s[i], y[i]) for i in resample_indices(len(s), b, rng, clusters)]
    )
    half = 50 * level
    return {
        "auroc": auroc(s, y),
        "ci": [
            float(np.nanpercentile(boot, 50 - half)),
            float(np.nanpercentile(boot, 50 + half)),
        ],
        "n": len(s),
        "n_positive": int(y.sum()),
    }


def paired_auroc_difference(
    a: np.ndarray,
    c: np.ndarray,
    positive: np.ndarray,
    b: int,
    rng: np.random.Generator,
    clusters: np.ndarray | None = None,
) -> dict[str, Any]:
    """AUROC(a) - AUROC(c) on the same rows, with a paired bootstrap.

    Args:
        a: First score.
        c: Second score.
        positive: Labels.
        b: Resamples.
        rng: Random generator.
        clusters: Optional cluster id per row.

    Returns:
        ``diff``, 95 % interval ``ci95`` and 90 % interval ``ci90`` (the
        latter gives a two one-sided equivalence check).
    """
    a, c = np.asarray(a, float), np.asarray(c, float)
    y = np.asarray(positive, bool)
    boot = np.array(
        [
            auroc(a[i], y[i]) - auroc(c[i], y[i])
            for i in resample_indices(len(a), b, rng, clusters)
        ]
    )
    return {
        "diff": auroc(a, y) - auroc(c, y),
        "ci95": [
            float(np.nanpercentile(boot, 2.5)),
            float(np.nanpercentile(boot, 97.5)),
        ],
        "ci90": [float(v) for v in np.nanpercentile(boot, [5, 95])],
    }


def month_block_indices(
    month: np.ndarray, period: np.ndarray, rng: np.random.Generator, block: int
) -> np.ndarray:
    """Period-stratified moving-block bootstrap over months.

    Within each period (``False`` first, then ``True``), blocks of ``block``
    consecutive months are drawn with replacement until the period's number
    of months is reached; the items of the drawn months are returned (with
    repeats).

    Args:
        month: Month per item (sortable strings such as ``"2024-11"``).
        period: Boolean period per item (e.g. after a cutoff).
        rng: Random generator.
        block: Block length in months.

    Returns:
        Item indices.
    """
    idx: list[np.ndarray] = []
    for side in (False, True):
        ms = np.array(sorted(set(month[period == side])))
        if len(ms) == 0:
            continue
        members = {m: np.flatnonzero((month == m) & (period == side)) for m in ms}
        length = min(block, len(ms))
        starts = np.arange(len(ms) - length + 1)
        drawn: list[str] = []
        while len(drawn) < len(ms):
            s0 = int(rng.choice(starts))
            drawn.extend(ms[s0 : s0 + length])
        idx.extend(members[m] for m in drawn[: len(ms)])
    return np.concatenate(idx)


def bca_bound(
    values: np.ndarray,
    stat: Callable[..., Any],
    alpha: float,
    side: str,
    b: int,
    seed: int,
) -> float:
    """One-sided BCa bootstrap bound for a statistic of one sample.

    Args:
        values: Sample.
        stat: Vectorised statistic, called as ``stat(x, axis=-1)``.
        alpha: One-sided error rate.
        side: ``"upper"`` or ``"lower"``.
        b: Resamples.
        seed: Seed.

    Returns:
        The bound.
    """
    from scipy import stats

    res = stats.bootstrap(
        (np.asarray(values, float),),
        stat,
        confidence_level=1 - 2 * alpha,
        n_resamples=b,
        method="BCa",
        random_state=np.random.default_rng(seed),
    )
    ci = res.confidence_interval
    return float(ci.high if side == "upper" else ci.low)


def smece_resamples(
    confidence: np.ndarray,
    correct: np.ndarray,
    b: int,
    rng: np.random.Generator,
    *,
    subsample: bool = False,
) -> np.ndarray:
    """SmoothECE on ``b`` resamples.

    Args:
        confidence: Confidences.
        correct: Correctness.
        b: Resamples.
        rng: Random generator.
        subsample: Draw ``n // 2`` points without replacement instead of an
            ordinary bootstrap (``n`` with replacement).

    Returns:
        One SmoothECE per resample.
    """
    p, y = np.asarray(confidence, float), np.asarray(correct, float)
    n = len(p)
    out = np.empty(b)
    for i in range(b):
        idx = (
            rng.choice(n, n // 2, replace=False) if subsample else rng.integers(0, n, n)
        )
        out[i] = smece(p[idx], y[idx])
    return out


def smece_interval(
    confidence: np.ndarray,
    correct: np.ndarray,
    b: int,
    rng: np.random.Generator,
    *,
    max_resamples_above: tuple[int, int] | None = (5000, 200),
    threshold: float | None = None,
) -> dict[str, Any]:
    """SmoothECE with a bias-recentred bootstrap interval.

    Resampling with replacement inflates kernel-smoothed calibration error,
    so the bootstrap distribution is shifted by its mean bias before taking
    percentiles.

    Args:
        confidence: Confidences.
        correct: Correctness.
        b: Resamples.
        rng: Random generator.
        max_resamples_above: ``(n, b_max)``: above ``n`` items draw at most
            ``b_max`` resamples (``None`` for no cap).
        threshold: Optional reference value; if given, the share of
            recentred replicates at or above it is reported as ``tail``.

    Returns:
        ``smece``, recentred 95 % ``ci``, mean confidence, accuracy, the
        number of resamples and (optionally) ``tail``.
    """
    pm, ok = np.asarray(confidence, float), np.asarray(correct, float)
    if max_resamples_above is not None and len(pm) > max_resamples_above[0]:
        b = min(b, max_resamples_above[1])
    est = smece(pm, ok)
    raw = np.array(
        [
            smece(pm[i], ok[i])
            for i in (rng.integers(0, len(pm), len(pm)) for _ in range(b))
        ]
    )
    boot = raw - (raw.mean() - est)
    out: dict[str, Any] = {
        "n": len(pm),
        "smece": est,
        "ci": list(percentile_ci(boot)),
        "mean_confidence": float(pm.mean()),
        "accuracy": float(ok.mean()),
        "resamples": b,
    }
    if threshold is not None:
        out["threshold"] = threshold
        out["tail"] = float(np.mean(boot >= threshold))
    return out


def on_resample(
    stat: Callable[[np.ndarray, np.ndarray], float], x: np.ndarray, y: np.ndarray
) -> Callable[[np.ndarray], float]:
    """Turn a statistic of two paired arrays into a statistic of resampled indices.

    For use with :func:`bootstrap`: ``bootstrap(n, on_resample(f, x, y), b,
    rng)`` resamples the pairs ``(x[i], y[i])`` together.

    Args:
        stat: Statistic of two arrays, e.g. AUROC of scores against labels.
        x: First array.
        y: Second array, paired with ``x``.

    Returns:
        ``idx -> stat(x[idx], y[idx])``.
    """

    def at(idx: np.ndarray) -> float:
        return stat(x[idx], y[idx])

    return at
