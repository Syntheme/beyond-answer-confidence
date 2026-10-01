"""Calibration of confidence against correctness."""

import math

import numpy as np


def smece(confidence: np.ndarray, correct: np.ndarray) -> float:
    """Top-label SmoothECE (Blasiok and Nakkiran, via ``relplot``).

    Args:
        confidence: Confidence per item (e.g. top-label probability).
        correct: Correctness (0/1), or the true P(correct) for a population
            value.

    Returns:
        The smooth ECE.
    """
    import relplot

    return float(
        relplot.smECE(np.asarray(confidence, float), np.asarray(correct, float))
    )


def ece(confidence: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    """Binned expected calibration error with equal-width bins.

    Args:
        confidence: Confidence per item, in [0, 1].
        correct: Correctness (0/1).
        n_bins: Number of bins.

    Returns:
        ``sum_b (n_b / n) |acc_b - conf_b|``.
    """
    c, y = np.asarray(confidence, float), np.asarray(correct, float)
    bins = np.minimum((c * n_bins).astype(int), n_bins - 1)
    total = 0.0
    for b in np.unique(bins):
        m = bins == b
        total += m.sum() / len(c) * abs(float(y[m].mean() - c[m].mean()))
    return float(total)


def reliability_bins(
    confidence: np.ndarray, correct: np.ndarray, n_bins: int = 10
) -> list[dict[str, float]]:
    """Equal-mass reliability bins.

    Args:
        confidence: Confidence per item.
        correct: Correctness per item.
        n_bins: Number of bins.

    Returns:
        Per bin: mean confidence, accuracy and count.
    """
    order = np.argsort(confidence, kind="stable")
    return [
        {
            "confidence": float(confidence[c].mean()),
            "accuracy": float(correct[c].mean()),
            "n": len(c),
        }
        for c in np.array_split(order, n_bins)
        if len(c)
    ]


def merge_tied_bins(
    bins: list[dict[str, float]], ndigits: int = 3
) -> list[dict[str, float]]:
    """Merge neighbouring reliability bins whose mean confidence is identical.

    Equal-mass bins split runs of identical confidence (often exactly 1.0)
    into several bins at the same x position. Merged bins pool their items
    (weighted by ``n``).

    Args:
        bins: ``{"confidence", "accuracy", "n"}`` per bin, sorted by confidence.
        ndigits: Rounding used to decide that two confidences are the same.

    Returns:
        Bins with distinct confidences.
    """
    out: list[dict[str, float]] = []
    for b in bins:
        prev = out[-1] if out else None
        if prev and round(prev["confidence"], ndigits) == round(
            b["confidence"], ndigits
        ):
            n = prev["n"] + b["n"]
            prev["accuracy"] = (
                prev["accuracy"] * prev["n"] + b["accuracy"] * b["n"]
            ) / n
            prev["confidence"] = (
                prev["confidence"] * prev["n"] + b["confidence"] * b["n"]
            ) / n
            prev["n"] = n
        else:
            out.append({k: float(b[k]) for k in ("confidence", "accuracy", "n")})
    return out


def wilson_interval(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion.

    Args:
        p: Observed proportion.
        n: Number of trials.
        z: Normal quantile (1.96 for 95 %).

    Returns:
        ``(low, high)``, or ``(nan, nan)`` when ``n == 0``.
    """
    if n == 0:
        return float("nan"), float("nan")
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def normal_mean_interval(values: np.ndarray, z: float = 1.96) -> tuple[float, float]:
    """Normal-approximation interval for a mean.

    Args:
        values: One value per item.
        z: Normal quantile.

    Returns:
        ``(low, high)``; a single value gives a zero-width interval.
    """
    v = np.asarray(values, float)
    if len(v) < 2:
        return float(v.mean()), float(v.mean())
    se = float(v.std(ddof=1) / np.sqrt(len(v)))
    return float(v.mean() - z * se), float(v.mean() + z * se)
