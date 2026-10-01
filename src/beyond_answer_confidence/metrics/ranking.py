"""How well a score ranks items: discrimination and selective prediction.

Ties. :func:`auroc` counts tied scores half (Mann-Whitney) and
:func:`weighted_accuracy_at` takes a crossing tie group fractionally, so both
are independent of input order. :func:`average_precision`, :func:`aurc`,
:func:`risk_coverage` and :func:`selective_at` sort by score with a *stable*
sort, so **items with tied scores are taken in input order** (earlier rows
first). This is the default because released results were computed that
way; with many ties (e.g. quantised probabilities) the result then depends
on row order. Pass ``ties="average"`` to :func:`average_precision`,
:func:`aurc` or :func:`risk_coverage` for an order-independent value.
"""

import math
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np

Ties = Literal["input_order", "average"]
"""Tie handling of the order-based metrics (see the module docstring)."""


def _check_ties(ties: str) -> None:
    if ties not in ("input_order", "average"):
        raise ValueError(f"ties must be 'input_order' or 'average', got {ties!r}")


def _tie_groups(sorted_scores: np.ndarray) -> np.ndarray:
    """Group id per position of a descending-sorted score array."""
    if len(sorted_scores) == 0:
        return np.zeros(0, int)
    change = np.r_[False, sorted_scores[1:] != sorted_scores[:-1]]
    return np.cumsum(change)


def _cumulative(values: np.ndarray, sorted_scores: np.ndarray, ties: str) -> np.ndarray:
    """Cumulative sums of ``values`` (in sorted order) under a tie rule.

    With ``"average"``, the cumulative sum inside each tie group is the
    expectation over random orders of the group: linear interpolation between
    the sums before and after the group.
    """
    cum = np.cumsum(values)
    if ties == "input_order" or len(values) == 0:
        return cum
    groups = _tie_groups(sorted_scores)
    starts = np.flatnonzero(np.r_[True, groups[1:] != groups[:-1]])
    ends = np.r_[starts[1:], len(values)]
    out = np.empty_like(cum, dtype=float)
    for a, b in zip(starts, ends, strict=True):
        before = cum[a - 1] if a else 0.0
        rate = (cum[b - 1] - before) / (b - a)
        out[a:b] = before + rate * np.arange(1, b - a + 1)
    return out


def auroc(scores: np.ndarray, positive: np.ndarray) -> float:
    """Area under the ROC curve (Mann-Whitney), ties counted half.

    Args:
        scores: Higher = more likely positive.
        positive: Boolean labels.

    Returns:
        AUROC in [0, 1], or ``nan`` if only one class is present.
    """
    from scipy.stats import rankdata

    y = np.asarray(positive, bool)
    ranks = rankdata(scores)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def stratified_auroc(
    scores: np.ndarray, positive: np.ndarray, strata: np.ndarray
) -> dict[str, Any]:
    """Size-weighted mean of within-stratum AUROCs (strata with both classes).

    Args:
        scores: Higher = more likely positive.
        positive: Labels.
        strata: Stratum id per row.

    Returns:
        Weighted AUROC, strata used, rows covered and total rows.
    """
    s = np.asarray(scores, float)
    y = np.asarray(positive, bool)
    g = np.asarray(strata)
    total, weight, used = 0.0, 0, 0
    for k in np.unique(g):
        m = g == k
        if 0 < y[m].sum() < m.sum():
            total += auroc(s[m], y[m]) * int(m.sum())
            weight += int(m.sum())
            used += 1
    return {
        "auroc": total / weight if weight else float("nan"),
        "strata_used": used,
        "rows_covered": weight,
        "rows": len(s),
    }


def fpr_at_tpr(scores: np.ndarray, positive: np.ndarray, tpr: float = 0.95) -> float:
    """False-positive rate at the threshold that catches ``tpr`` of positives.

    Args:
        scores: Higher = more likely positive.
        positive: Boolean labels.
        tpr: Target true-positive rate.

    Returns:
        The false-positive rate on negatives, or ``nan`` without positives
        or without negatives.
    """
    s = np.asarray(scores, float)
    y = np.asarray(positive, bool)
    if not y.any() or y.all():
        return float("nan")
    threshold = np.quantile(s[y], 1 - tpr)
    return float(np.mean(s[~y] >= threshold))


def average_precision(
    scores: np.ndarray, positive: np.ndarray, ties: Ties = "input_order"
) -> float:
    """Average precision (area under the precision-recall curve).

    **Ties:** by default, items with tied scores are ranked in input order
    (stable sort), so the value can depend on row order. With
    ``ties="average"``, each group of tied scores is one threshold: precision
    is evaluated after the whole group (the step-wise definition used by
    scikit-learn's ``average_precision_score``), independent of order.

    Args:
        scores: Higher = more likely positive.
        positive: Boolean labels.
        ties: ``"input_order"`` (default) or ``"average"``.

    Returns:
        Average precision, or ``nan`` without positives.
    """
    _check_ties(ties)
    s = np.asarray(scores, float)
    y = np.asarray(positive, bool)
    if not y.any():
        return float("nan")
    order = np.argsort(-s, kind="stable")
    hits = y[order].astype(float)
    if ties == "average":
        cum = np.cumsum(hits)
        last = np.r_[s[order][1:] != s[order][:-1], True]
        tp, k = cum[last], np.flatnonzero(last) + 1
        recall_step = np.diff(np.r_[0.0, tp]) / hits.sum()
        return float((recall_step * tp / k).sum())
    precision = np.cumsum(hits) / np.arange(1, len(hits) + 1)
    return float((precision * hits).sum() / hits.sum())


def aurc(score: np.ndarray, correct: np.ndarray, ties: Ties = "input_order") -> float:
    """Area under the risk-coverage curve (answer the highest score first).

    **Ties:** by default, items with tied scores are answered in input order
    (stable sort), so the value can depend on row order. With
    ``ties="average"``, the risk inside a tie group is its expectation over
    random orders of the group.

    Args:
        score: Higher = answer first.
        correct: Correctness (coerced to boolean).
        ties: ``"input_order"`` (default) or ``"average"``.

    Returns:
        Mean selective error over all coverage levels (lower is better).
    """
    _check_ties(ties)
    s = np.asarray(score, float)
    ok = np.asarray(correct, bool)
    order = np.argsort(-s, kind="stable")
    err = 1 - ok[order].astype(float)
    cum = _cumulative(err, s[order], ties)
    return float((cum / np.arange(1, len(err) + 1)).mean())


def risk_coverage(
    confidence: np.ndarray,
    correct: np.ndarray,
    coverages: Sequence[float] = (0.8, 0.9, 0.95),
    ties: Ties = "input_order",
) -> dict[str, float]:
    """Selective-prediction summary (accept in order of decreasing confidence).

    **Ties:** by default, items with tied confidence are accepted in input
    order (stable sort), so the values can depend on row order. With
    ``ties="average"``, risks inside a tie group are their expectation over
    random orders of the group.

    Args:
        confidence: Per-item confidence.
        correct: Per-item correctness (coerced to boolean).
        coverages: Coverage levels at which to report the selective risk.
        ties: ``"input_order"`` (default) or ``"average"``.

    Returns:
        AURC, AUGRC and ``risk@<percent>`` per coverage level.
    """
    _check_ties(ties)
    conf = np.asarray(confidence, float)
    ok = np.asarray(correct, bool)
    order = np.argsort(-conf, kind="stable")
    errors = (~ok[order]).astype(float)
    n = len(errors)
    cum = _cumulative(errors, conf[order], ties)
    risk = cum / np.arange(1, n + 1)
    out = {"aurc": float(risk.mean()), "augrc": float((cum / n).mean())}
    for c in coverages:
        k = max(1, math.ceil(c * n))
        out[f"risk@{int(c * 100)}"] = float(risk[k - 1])
    return out


def selective_at(
    score: np.ndarray,
    correct: np.ndarray,
    coverage: float,
    confidence: np.ndarray | None = None,
    confident: float = 0.9,
) -> dict[str, Any]:
    """Answer the ``coverage`` share of items with the highest score.

    **Ties:** items with tied scores are answered in input order (stable
    sort), so the result can depend on row order.

    Args:
        score: Higher = answer first.
        correct: Whether each answer would be right (coerced to boolean).
        coverage: Share of items to answer.
        confidence: Answer probability, for counting confident errors.
        confident: Confidence at or above which an error counts as confident.

    Returns:
        Answered count, achieved coverage, selective accuracy, errors and
        (with ``confidence``) confident errors.
    """
    s = np.asarray(score, float)
    n = len(s)
    k = round(coverage * n)
    order = np.argsort(-s, kind="stable")[:k]
    ok = np.asarray(correct, bool)[order]
    res: dict[str, Any] = {
        "answered": k,
        "coverage": k / n if n else 0.0,
        "selective_accuracy": float(ok.mean()) if k else None,
        "errors": int(k - ok.sum()),
    }
    if confidence is not None:
        res["confident_errors"] = int(
            ((~ok.astype(bool)) & (confidence[order] >= confident)).sum()
        )
    return res


def weighted_accuracy_at(
    score: np.ndarray, correct: np.ndarray, weight: np.ndarray, coverage: float
) -> tuple[float, float]:
    """Weighted accuracy when answering the highest-score items up to ``coverage``.

    Coverage is matched exactly: items are taken in descending score order,
    and the group of tied scores that crosses the target is taken
    fractionally (the expected result of random tie-breaking). Items with
    zero weight are left out.

    Args:
        score: Higher = answer first.
        correct: Correctness.
        weight: Item weights.
        coverage: Target share of the total weight to answer.

    Returns:
        ``(accuracy, achieved coverage)``.
    """
    keep = weight > 0
    s, ok, w = score[keep], correct[keep].astype(float), weight[keep]
    total, target = w.sum(), coverage * w.sum()
    if target <= 0:
        return float("nan"), 0.0
    taken_w = taken_ok = 0.0
    for v in np.unique(s)[::-1]:
        grp = s == v
        gw, gok = w[grp].sum(), (w[grp] * ok[grp]).sum()
        frac = min(1.0, (target - taken_w) / gw)
        taken_w += frac * gw
        taken_ok += frac * gok
        if taken_w >= target - 1e-12:
            break
    return float(taken_ok / taken_w), float(taken_w / total)
