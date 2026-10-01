"""Change points in a monthly series (e.g. accuracy around a knowledge cutoff)."""

from collections.abc import Sequence
from typing import Any

import numpy as np


def change_point(
    months: Sequence[str],
    accuracy: Sequence[float],
    weights: Sequence[float],
    min_side: int = 6,
) -> dict[str, Any]:
    """Single change point in a monthly series (weighted two-mean least squares).

    Every split leaving at least ``min_side`` months on each side is tried;
    the one minimising the weighted within-side sum of squares wins (the
    earliest on ties).

    Args:
        months: Months in chronological order.
        accuracy: Value per month (e.g. accuracy).
        weights: Weight per month (e.g. items per month).
        min_side: Minimum months on each side of the split.

    Returns:
        ``first_post_month`` (first month after the change), the weighted
        means ``pre_accuracy`` and ``post_accuracy``, and ``drop``
        (pre minus post).

    Raises:
        ValueError: With fewer than ``2 * min_side`` months.
    """
    a, w = np.asarray(accuracy, float), np.asarray(weights, float)
    if len(a) < 2 * min_side:
        raise ValueError(f"need at least {2 * min_side} months, got {len(a)}")
    best: tuple[float, int] | None = None
    for k in range(min_side, len(a) - min_side + 1):
        left = np.average(a[:k], weights=w[:k])
        right = np.average(a[k:], weights=w[k:])
        sse = float(
            (w[:k] * (a[:k] - left) ** 2).sum() + (w[k:] * (a[k:] - right) ** 2).sum()
        )
        if best is None or sse < best[0]:
            best = (sse, k)
    k = best[1] if best is not None else min_side
    pre = float(np.average(a[:k], weights=w[:k]))
    post = float(np.average(a[k:], weights=w[k:]))
    return {
        "first_post_month": months[k],
        "pre_accuracy": pre,
        "post_accuracy": post,
        "drop": pre - post,
    }
