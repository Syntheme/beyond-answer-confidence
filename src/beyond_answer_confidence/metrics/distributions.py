"""Statistics of a single categorical distribution.

Everything here works on probability vectors, never on a backend's own
"confidence" field, whose scale can depend on the number of options.
"""

import math
from collections.abc import Mapping, Sequence
from itertools import combinations

import numpy as np

EPS = 1e-12
"""Floor applied before taking logs of probabilities."""


def normalise(probabilities: Mapping[str, float]) -> dict[str, float]:
    """Rescale probabilities to sum to exactly 1.

    Args:
        probabilities: Non-negative probability per option.

    Returns:
        The same options with probabilities summing to 1.

    Raises:
        ValueError: If any probability is negative or they sum to zero.
    """
    if any(p < 0 for p in probabilities.values()):
        raise ValueError("probabilities must be non-negative")
    total = sum(probabilities.values())
    if total <= 0:
        raise ValueError("probabilities must not all be zero")
    return {k: p / total for k, p in probabilities.items()}


def entropy(probabilities: Sequence[float]) -> float:
    """Shannon entropy in nats.

    Args:
        probabilities: A probability vector.

    Returns:
        ``-sum(p log p)``.
    """
    return -sum(p * math.log(p) for p in probabilities if p > 0)


def normalised_entropy(probabilities: Sequence[float]) -> float:
    """Entropy divided by its maximum ``log K``, so it lies in [0, 1].

    Args:
        probabilities: A probability vector with at least two entries.

    Returns:
        ``H(p) / log K``.

    Raises:
        ValueError: If fewer than two options are given.
    """
    k = len(probabilities)
    if k < 2:
        raise ValueError("need at least two options")
    return entropy(probabilities) / math.log(k)


def nll(probabilities: Mapping[str, float], gold: str, eps: float = EPS) -> float:
    """Negative log-likelihood of the gold option (log score).

    Args:
        probabilities: Probability per option.
        gold: The correct option (missing options count as 0).
        eps: Floor on the probability.

    Returns:
        ``-log max(p(gold), eps)``.
    """
    return -math.log(max(probabilities.get(gold, 0.0), eps))


def brier(probabilities: Mapping[str, float], gold: str) -> float:
    """Multiclass Brier score: squared error against the one-hot gold vector.

    Args:
        probabilities: Probability per option.
        gold: The correct option.

    Returns:
        ``sum_k (p_k - 1[k = gold])^2``, in [0, 2].
    """
    return sum((p - (1.0 if k == gold else 0.0)) ** 2 for k, p in probabilities.items())


def choice_confidence(probabilities: Sequence[float]) -> float:
    """The System One choice confidence, ``(K * p_max - 1) / (K - 1)``.

    Useful only to check that a reported confidence carries no information
    beyond the probabilities.

    Args:
        probabilities: A probability vector with at least two entries.

    Returns:
        The confidence, clipped to [0, 1].

    Raises:
        ValueError: If fewer than two options are given.
    """
    k = len(probabilities)
    if k < 2:
        raise ValueError("need at least two options")
    return min(1.0, max(0.0, (k * max(probabilities) - 1) / (k - 1)))


def tv(p: Mapping[str, float], q: Mapping[str, float]) -> float:
    """Total-variation distance (missing keys count as 0).

    Args:
        p: First distribution.
        q: Second distribution.

    Returns:
        ``0.5 * sum |p - q|``.
    """
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)


def mean_pairwise_tv(dists: Sequence[Mapping[str, float]]) -> float:
    """Mean pairwise total-variation distance, e.g. between replicates.

    Args:
        dists: Distributions over the same options.

    Returns:
        Mean distance (0 with fewer than two distributions).
    """
    pairs = list(combinations(dists, 2))
    if not pairs:
        return 0.0
    return float(np.mean([0.5 * sum(abs(a[k] - b[k]) for k in a) for a, b in pairs]))


def js_divergence(p: Mapping[str, float], q: Mapping[str, float]) -> float:
    """Jensen-Shannon divergence in nats.

    Args:
        p: First distribution, keyed by option.
        q: Second distribution over the same options.

    Returns:
        The divergence, in [0, log 2].

    Raises:
        ValueError: If the option sets differ.
    """
    if p.keys() != q.keys():
        raise ValueError("distributions must share the same options")
    m = {k: (p[k] + q[k]) / 2 for k in p}

    def kl(a: Mapping[str, float]) -> float:
        return sum(a[k] * math.log(a[k] / m[k]) for k in a if a[k] > 0)

    return (kl(p) + kl(q)) / 2


def logit(p: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """Log-odds with the probabilities clipped to ``[eps, 1 - eps]``.

    Args:
        p: Probabilities.
        eps: Clip distance from 0 and 1.

    Returns:
        ``log(p / (1 - p))``.
    """
    q = np.clip(np.asarray(p, float), eps, 1 - eps)
    return np.asarray(np.log(q / (1 - q)))


def expected_scores(
    dist: Mapping[str, float], ideal: Mapping[str, float], eps: float = 1e-4
) -> dict[str, float]:
    """Expected Brier and log loss when the outcome follows ``ideal``.

    Args:
        dist: Predicted distribution.
        ideal: True outcome distribution.
        eps: Log-loss floor.

    Returns:
        Expected Brier and log loss of ``dist``, and of ``ideal`` itself
        (the floor: its Brier and its entropy).
    """
    support = [(k, q) for k, q in ideal.items() if q > 0]
    return {
        "brier": float(sum(q * brier(dist, k) for k, q in support)),
        "log_loss": float(
            sum(q * -np.log(max(dist.get(k, 0.0), eps)) for k, q in support)
        ),
        "ideal_brier": float(sum(q * brier(ideal, k) for k, q in support)),
        "ideal_log_loss": float(sum(-q * np.log(q) for _, q in support)),
    }
