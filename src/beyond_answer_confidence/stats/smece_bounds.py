"""Upper bounds for SmoothECE under four resampling schemes.

Resampling with replacement duplicates points, which inflates a
kernel-smoothed calibration error, so the ordinary bootstrap of SmoothECE is
biased upward. The schemes here differ in how they handle that bias:

- ``percentile``: quantiles of the ordinary bootstrap;
- ``recentred``: the bootstrap shifted down by its mean bias;
- ``basic``: reverse percentile, ``2 est - q``;
- ``subsample``: half-samples without replacement (never duplicates points),
  scaled by ``sqrt(m / n)`` around the estimate (Politis-Romano).

:mod:`beyond_answer_confidence.analyses.smece_coverage` measures how often each
scheme's upper bound covers the true value.
"""

import math
from typing import Any

import numpy as np

from beyond_answer_confidence.metrics.calibration import smece
from beyond_answer_confidence.stats.resampling import smece_resamples

SMECE_METHODS = ("percentile", "recentred", "basic", "subsample")
"""Resampling schemes for a SmoothECE upper bound (see :func:`smece_bounds`)."""


def smece_bounds(
    est: float,
    n: int,
    boot: np.ndarray | None,
    sub: np.ndarray | None,
    alpha: float = 0.05,
) -> dict[str, dict[str, float]]:
    """One-sided upper bound and two-sided interval per resampling scheme.

    In a simulation with realistic, quantised confidence distributions and
    one overconfidence shape, the ``percentile`` bound was close to nominal,
    ``recentred`` mildly liberal, and ``basic`` and ``subsample`` far too
    liberal; other miscalibration shapes were not simulated.

    Args:
        est: Full-sample SmoothECE.
        n: Sample size.
        boot: Ordinary bootstrap values (``None`` skips the first three
            schemes).
        sub: Half-sample values (``None`` skips subsampling).
        alpha: One-sided error rate of the upper bound; the two-sided
            interval has coverage ``1 - 2 alpha``.

    Returns:
        Scheme -> ``{"upper", "low", "high"}``.
    """
    lo_q, hi_q = alpha * 100, (1 - alpha) * 100
    out: dict[str, dict[str, float]] = {}
    if boot is not None:
        q_lo, q_hi = np.percentile(boot, [lo_q, hi_q])
        shift = float(boot.mean() - est)
        out["percentile"] = {
            "upper": float(q_hi),
            "low": float(q_lo),
            "high": float(q_hi),
        }
        out["recentred"] = {
            "upper": float(q_hi - shift),
            "low": float(q_lo - shift),
            "high": float(q_hi - shift),
        }
        out["basic"] = {
            "upper": float(2 * est - q_lo),
            "low": float(2 * est - q_hi),
            "high": float(2 * est - q_lo),
        }
    if sub is not None:
        m = n // 2
        root = math.sqrt(m) * (sub - est)
        r_lo, r_hi = np.percentile(root, [lo_q, hi_q])
        out["subsample"] = {
            "upper": float(est - r_lo / math.sqrt(n)),
            "low": float(est - r_hi / math.sqrt(n)),
            "high": float(est - r_lo / math.sqrt(n)),
        }
    return out


def smece_tail_shares(
    est: float, n: int, boot: np.ndarray, sub: np.ndarray, reference: float
) -> dict[str, float]:
    """Interval-inversion tail shares at a reference value, per scheme.

    Each value is the smallest one-sided error rate at which that scheme's
    upper bound falls below ``reference``.

    Args:
        est: Full-sample SmoothECE.
        n: Sample size.
        boot: Ordinary bootstrap values.
        sub: Half-sample values.
        reference: Reference value.

    Returns:
        Scheme -> tail share.
    """
    shift = boot.mean() - est
    root = math.sqrt(n // 2) * (sub - est)
    return {
        "percentile": float(np.mean(boot >= reference)),
        "recentred": float(np.mean(boot - shift >= reference)),
        "basic": float(np.mean(2 * est - boot >= reference)),
        "subsample": float(np.mean(est - root / math.sqrt(n) >= reference)),
    }


def smece_with_bounds(
    confidence: np.ndarray,
    correct: np.ndarray,
    reference: float,
    b: int,
    rng: np.random.Generator,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """SmoothECE with every scheme's bounds and tail share at a reference.

    Draw order: ``b`` ordinary bootstrap resamples, then ``b`` half-samples.

    Args:
        confidence: Confidences.
        correct: Correctness.
        reference: Reference value for the tail shares (e.g. 0.05).
        b: Resamples per scheme.
        rng: Random generator.
        alpha: One-sided error rate of the bounds.

    Returns:
        ``n``, ``smece``, ``resamples``, ``bootstrap_bias``, per-scheme
        ``bounds``, per-scheme ``tail`` shares and the ``reference``.
    """
    p, y = np.asarray(confidence, float), np.asarray(correct, float)
    est = smece(p, y)
    boot = smece_resamples(p, y, b, rng, subsample=False)
    sub = smece_resamples(p, y, b, rng, subsample=True)
    return {
        "n": len(p),
        "smece": est,
        "resamples": b,
        "bootstrap_bias": float(boot.mean() - est),
        "bounds": smece_bounds(est, len(p), boot, sub, alpha),
        "tail": smece_tail_shares(est, len(p), boot, sub, reference),
        "reference": reference,
    }
