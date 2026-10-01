"""Turn answered items into analysis rows (pure; no I/O)."""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from beyond_answer_confidence.backends.cache import CallRecord
from beyond_answer_confidence.metrics.distributions import tv
from beyond_answer_confidence.tasks.schema import Item


def choice_dist(
    records: Sequence[CallRecord], name: str, options: Sequence[str]
) -> dict[str, float]:
    """Replicate-mean distribution of a choice question over fixed options.

    Options the response omits count as 0; the mean is renormalised.

    Args:
        records: One record per replicate.
        name: Question name.
        options: All option keys.

    Returns:
        Option -> probability.
    """
    acc = dict.fromkeys(options, 0.0)
    for rec in records:
        for opt, p in rec.response.choices[name].probabilities.items():
            if opt in acc:
                acc[opt] += p / len(records)
    total = sum(acc.values()) or 1.0
    return {k: v / total for k, v in acc.items()}


def yes_mean(records: Sequence[CallRecord], name: str) -> float:
    """Replicate-mean P(yes) of a yes/no question.

    Args:
        records: One record per replicate.
        name: Question name.

    Returns:
        Mean probability of "yes".
    """
    return float(np.mean([rec.response.yes_no[name].p_yes for rec in records]))


def score_items(
    items: Sequence[Item], complete: Mapping[str, Sequence[CallRecord]]
) -> list[dict[str, Any]]:
    """Score every complete item.

    Args:
        items: Items.
        complete: Unit -> replicate records.

    Returns:
        One row per complete item: metadata, replicate-mean distribution,
        ``p_max``, ``top``, ``correct``, ``p_gold``, normalised entropy
        (``None`` with fewer than two options, where it is undefined),
        replicate spread (mean pairwise total variation), each yes/no
        question's mean P(yes) as ``p_<name>`` and each score's mean as
        ``s_<name>``.

    Raises:
        ValueError: If an item's responses put no probability on any of its
            options (the distribution would be all zeros).
    """
    rows = []
    for it in items:
        recs = complete.get(it.unit)
        if not recs:
            continue
        row: dict[str, Any] = {"unit": it.unit, **it.info, "gold": it.gold}
        if it.options:
            dist = choice_dist(recs, "answer", it.options)
            probs = np.array(list(dist.values()))
            if not probs.any():
                raise ValueError(
                    f"{it.unit}: the responses give no probability to any listed "
                    "option; inspect the cached responses of this item"
                )
            nz = probs[probs > 0]
            # Kept inline rather than metrics.distributions.normalised_entropy:
            # numpy's summation differs from the shared function's in the last
            # bits for about 8 % of released rows.
            norm_entropy = (
                float(-(nz * np.log(nz)).sum() / np.log(len(probs)))
                if len(probs) >= 2
                else None
            )
            reps = [choice_dist([r], "answer", it.options) for r in recs]
            top = max(dist, key=dist.__getitem__)
            row |= {
                "dist": dist,
                "p_max": float(probs.max()),
                "top": top,
                "correct": top == it.gold if it.gold is not None else None,
                "p_gold": dist.get(it.gold, 0.0) if it.gold is not None else None,
                "norm_entropy": norm_entropy,
                "rep_tv": float(
                    np.mean(
                        [tv(a, b) for i, a in enumerate(reps) for b in reps[i + 1 :]]
                    )
                )
                if len(reps) > 1
                else 0.0,
            }
        for name in it.nouls:
            row[f"p_{name}"] = yes_mean(recs, name)
        for name in it.scores:
            row[f"s_{name}"] = float(
                np.mean([r.response.scores[name].score for r in recs])
            )
        row["models"] = sorted({r.response.model for r in recs})
        rows.append(row)
    return rows
