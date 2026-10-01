"""Recalibration maps and what no map of one score can fix.

- Scalar maps fitted on held-in items: temperature scaling of a whole
  distribution and Platt scaling of the top probability.
- Grouped out-of-fold logistic scores, for combining signals without
  in-sample optimism.
- A lower bound on the calibration gap of *every* function of a quantised
  score across two periods, with two permutation nulls.
- Cross-fitted recalibrators: fitted on half of the months, tested on the
  other half, with and without knowledge of the period.

Rows are data frames with ``p_max`` (the score), ``correct``, ``post`` (the
period) and ``month`` columns where needed.
"""

from collections.abc import Callable, Sequence
from typing import Any, cast

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

FLOOR = 1e-4
"""Probability floor for temperature and Platt scaling."""
LEVEL_DIGITS = 9
"""Score values are grouped after rounding to this many digits."""


# --- scalar maps ----------------------------------------------------------------


def temper(dists: Sequence[dict[str, float]], t: float) -> list[dict[str, float]]:
    """Temperature-scale distributions (floored at :data:`FLOOR`, renormalised).

    Args:
        dists: Option -> probability, one per item.
        t: Temperature (> 1 softens).

    Returns:
        The scaled distributions.
    """
    out = []
    for d in dists:
        keys = list(d)
        p = np.maximum(np.array([d[k] for k in keys]), FLOOR) ** (1 / t)
        p = p / p.sum()
        out.append(dict(zip(keys, p.tolist(), strict=True)))
    return out


def fit_temperature(dists: Sequence[dict[str, float]], golds: Sequence[str]) -> float:
    """Temperature minimising the log loss of the gold option.

    Searched on ``log t`` in ``[-3, 3]`` (bounded scalar minimisation).

    Args:
        dists: Distributions.
        golds: Gold option per distribution.

    Returns:
        The temperature.
    """
    from scipy.optimize import minimize_scalar

    def nll(log_t: float) -> float:
        tempered = temper(dists, float(np.exp(log_t)))
        return -float(
            np.mean(
                [np.log(max(d[g], 1e-12)) for d, g in zip(tempered, golds, strict=True)]
            )
        )

    return float(np.exp(minimize_scalar(nll, bounds=(-3, 3), method="bounded").x))


def _clipped_logit(p: ArrayLike) -> np.ndarray:
    q = np.clip(np.asarray(p, float), FLOOR, 1 - FLOOR)
    return np.asarray(np.log(q / (1 - q)))


def fit_platt(p_max: np.ndarray, correct: np.ndarray) -> tuple[float, float]:
    """Logistic map from ``logit(p_max)`` to P(correct) (unpenalised).

    Args:
        p_max: Top probabilities.
        correct: Correctness.

    Returns:
        ``(slope, intercept)``.
    """
    from sklearn.linear_model import LogisticRegression

    x = _clipped_logit(p_max)
    m = LogisticRegression(C=1e6).fit(x[:, None], np.asarray(correct).astype(int))
    return float(m.coef_[0, 0]), float(m.intercept_[0])


def apply_platt(p_max: np.ndarray, ab: tuple[float, float]) -> np.ndarray:
    """Apply a Platt map.

    Args:
        p_max: Top probabilities.
        ab: ``(slope, intercept)``.

    Returns:
        Recalibrated confidence.
    """
    z = ab[0] * _clipped_logit(p_max) + ab[1]
    return np.asarray(1 / (1 + np.exp(-z)))


def fit_logistic_map(features: Sequence[ArrayLike], correct: ArrayLike) -> Any:
    """Unpenalised logistic map from ``logit`` of several scores to P(correct).

    Args:
        features: Probability-valued scores (each clipped at :data:`FLOOR`
            and turned into log-odds).
        correct: Correctness.

    Returns:
        The fitted scikit-learn model (inputs: the same log-odds columns).
    """
    from sklearn.linear_model import LogisticRegression

    x = np.column_stack([_clipped_logit(f) for f in features])
    return LogisticRegression(C=1e6).fit(x, np.asarray(correct).astype(int))


def predict_logistic_map(model: Any, features: Sequence[ArrayLike]) -> np.ndarray:
    """Apply a map from :func:`fit_logistic_map`.

    Args:
        model: The fitted model.
        features: The same scores as at fitting time.

    Returns:
        Predicted P(correct).
    """
    x = np.column_stack([_clipped_logit(f) for f in features])
    return np.asarray(model.predict_proba(x)[:, 1])


def split_half(
    df: pd.DataFrame, key: str, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split rows in two by a random half of the distinct ``key`` values.

    Args:
        df: Rows.
        key: Column whose (string) values are split.
        seed: Seed (a fresh generator; one ``choice`` without replacement over
            the sorted values).

    Returns:
        ``(fit, test)`` rows.
    """
    ids = np.array(sorted(df[key].astype(str).unique()))
    rng = np.random.default_rng(seed)
    fit_ids = set(rng.choice(ids, len(ids) // 2, replace=False))
    mask = df[key].astype(str).isin(fit_ids)
    return df[mask], df[~mask]


def cv_logistic_scores(
    x: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int = 0
) -> np.ndarray:
    """Out-of-fold predicted probabilities from a grouped 5-fold logistic model.

    Folds are grouped, so no group is in both train and test; a fold whose
    training labels are all equal predicts their mean.

    Args:
        x: Features, shape ``(n, d)``.
        y: Binary labels (e.g. errors).
        groups: Group per item.
        seed: Fold seed.

    Returns:
        Out-of-fold probabilities.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from threadpoolctl import threadpool_limits

    pred = np.zeros(len(y))
    folds = GroupKFold(n_splits=5, shuffle=True, random_state=seed)
    # One BLAS thread: multithreaded reductions change the fit in the last
    # digits, so results would depend on the machine's core count.
    with threadpool_limits(limits=1):
        for train, test in folds.split(x, y, groups):
            if len(set(y[train])) < 2:
                pred[test] = y[train].mean()
                continue
            model = LogisticRegression().fit(x[train], y[train])
            pred[test] = model.predict_proba(x[test])[:, 1]
    return pred


# --- every function of one score ---------------------------------------------------


def level_table(tf: pd.DataFrame) -> pd.DataFrame:
    """Items and accuracy per distinct score value and period.

    Args:
        tf: Rows with ``p_max``, ``correct`` and ``post``.

    Returns:
        One row per value ``v``: ``n_pre``, ``n_post``, ``a_pre``, ``a_post``
        (accuracy is NaN for a period without items at that value).
    """
    d = tf.assign(v=tf["p_max"].round(LEVEL_DIGITS), ok=tf["correct"].astype(float))
    g = cast(
        pd.DataFrame,
        d.groupby(["v", "post"])
        .agg(n=("ok", "size"), a=("ok", "mean"))
        .unstack("post"),
    )
    pairs = cast(list[tuple[str, bool]], list(g.columns))
    g.columns = [f"{a}_{'post' if b else 'pre'}" for a, b in pairs]
    return g.fillna({"n_pre": 0, "n_post": 0}).reset_index()


def lower_bound(levels: pd.DataFrame) -> float:
    """Lower bound on the mean absolute calibration gap of any function of the score.

    For any map ``f``, at each value ``v``,
    ``n_pre |f(v) - a_pre(v)| + n_post |f(v) - a_post(v)|`` is at least
    ``min(n_pre, n_post) |a_pre(v) - a_post(v)|``; summed over values and
    divided by the number of items this bounds the mean absolute gap over
    both periods from below.

    Args:
        levels: Output of :func:`level_table`.

    Returns:
        ``sum_v min(n_pre, n_post) |a_pre - a_post| / N`` (values seen in one
        period only contribute 0).
    """
    both = levels.dropna(subset=["a_pre", "a_post"])
    m = np.minimum(both["n_pre"], both["n_post"])
    total = float(levels["n_pre"].sum() + levels["n_post"].sum())
    return float((m * (both["a_pre"] - both["a_post"]).abs()).sum() / total)


def per_value_map_gaps(tf: pd.DataFrame) -> dict[str, float]:
    """Gaps left by the per-value pooled map (the least-squares best common map).

    Args:
        tf: Rows with ``p_max``, ``correct`` and ``post``.

    Returns:
        Mean (map - accuracy) per period, fitted in-sample.
    """
    d = tf.assign(v=tf["p_max"].round(LEVEL_DIGITS), ok=tf["correct"].astype(float))
    mapped = d.groupby("v")["ok"].transform("mean")
    return {
        "mean_gap_pre": float((mapped - d["ok"])[~d["post"]].mean()),
        "mean_gap_post": float((mapped - d["ok"])[d["post"]].mean()),
    }


def within_value_permutation_null(
    tf: pd.DataFrame, perms: int, rng: np.random.Generator
) -> np.ndarray:
    """Null distribution of :func:`lower_bound` with periods shuffled within values.

    Draw order: per permutation, one ``rng.permutation`` per distinct value
    (in sorted order).

    Args:
        tf: Rows with ``p_max``, ``correct`` and ``post``.
        perms: Permutations.
        rng: Random generator.

    Returns:
        One bound per permutation.
    """
    v = tf["p_max"].round(LEVEL_DIGITS).to_numpy()
    post = tf["post"].to_numpy(bool)
    groups = [np.flatnonzero(v == x) for x in np.unique(v)]
    null = np.empty(perms)
    for k in range(perms):
        shuffled = post.copy()
        for g in groups:
            shuffled[g] = rng.permutation(post[g])
        null[k] = lower_bound(level_table(tf.assign(post=shuffled)))
    return null


def random_months_null(
    tf: pd.DataFrame, perms: int, rng: np.random.Generator
) -> np.ndarray:
    """Null distribution of :func:`lower_bound` with random months as the later period.

    Keeps months intact: each permutation labels a random set of as many
    months as the observed later period has.

    Args:
        tf: Rows with ``p_max``, ``correct``, ``post`` and ``month``.
        perms: Permutations.
        rng: Random generator (one ``choice`` without replacement each).

    Returns:
        One bound per permutation.
    """
    months = np.array(sorted(tf["month"].unique()))
    n_post_months = int(tf.loc[tf["post"], "month"].nunique())
    null = np.empty(perms)
    for k in range(perms):
        chosen = set(rng.choice(months, n_post_months, replace=False))
        null[k] = lower_bound(level_table(tf.assign(post=tf["month"].isin(chosen))))
    return null


# --- cross-fitted recalibrators ---------------------------------------------------


def fit_common_map(fit: pd.DataFrame) -> Callable[[np.ndarray], np.ndarray]:
    """Recalibrator of the score alone: per-value pooled accuracy, else isotonic.

    Args:
        fit: Fitting rows with ``p_max`` and ``correct``.

    Returns:
        ``predict(p_max) -> recalibrated confidence``.
    """
    from sklearn.isotonic import IsotonicRegression

    iso = IsotonicRegression(out_of_bounds="clip").fit(
        fit["p_max"], fit["correct"].astype(float)
    )
    per_value = (
        fit.assign(v=fit["p_max"].round(LEVEL_DIGITS))
        .groupby("v")["correct"]
        .mean()
        .to_dict()
    )

    def predict(p: np.ndarray) -> np.ndarray:
        fallback = iso.predict(p)
        return np.array(
            [
                per_value.get(round(float(x), LEVEL_DIGITS), f)
                for x, f in zip(p, fallback, strict=True)
            ]
        )

    return predict


def fit_period_aware_map(
    fit: pd.DataFrame,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Recalibrator that knows the period: one isotonic map per period.

    Args:
        fit: Fitting rows with ``p_max``, ``correct`` and ``post``.

    Returns:
        ``predict(p_max, post) -> recalibrated confidence``.
    """
    from sklearn.isotonic import IsotonicRegression

    models = {
        per: IsotonicRegression(out_of_bounds="clip").fit(
            g["p_max"], g["correct"].astype(float)
        )
        for per, g in fit.groupby("post")
    }

    def predict(p: np.ndarray, post: np.ndarray) -> np.ndarray:
        out = np.empty(len(p))
        for per, m in models.items():
            out[post == per] = m.predict(p[post == per])
        return out

    return predict


def cross_fitted(
    tf: pd.DataFrame, repeats: int, rng: np.random.Generator
) -> dict[str, Any]:
    """Recalibrators fitted on half of the months, tested on the other half.

    Each repeat draws half of each period's months (earlier period first)
    and uses the split in both directions.

    Args:
        tf: Rows with ``p_max``, ``correct``, ``post`` and ``month``.
        repeats: Random month splits.
        rng: Random generator.

    Returns:
        Held-out mean gap per period for the common map (score only) and the
        period-aware map: mean, min and max over splits.
    """
    by_period = {
        per: np.array(sorted(g["month"].unique())) for per, g in tf.groupby("post")
    }
    res: dict[str, list[float]] = {
        k: [] for k in ("common_pre", "common_post", "aware_pre", "aware_post")
    }
    for _ in range(repeats):
        half = set()
        for ms in by_period.values():
            half |= set(rng.choice(ms, len(ms) // 2, replace=False))
        a_mask = tf["month"].isin(half).to_numpy()
        for fit, test in ((tf[a_mask], tf[~a_mask]), (tf[~a_mask], tf[a_mask])):
            p = test["p_max"].to_numpy(float)
            ok = test["correct"].to_numpy(float)
            post = test["post"].to_numpy(bool)
            common = fit_common_map(fit)(p)
            aware = fit_period_aware_map(fit)(p, post)
            for name, pred in (("common", common), ("aware", aware)):
                res[f"{name}_pre"].append(float((pred - ok)[~post].mean()))
                res[f"{name}_post"].append(float((pred - ok)[post].mean()))
    return {
        "splits": repeats * 2,
        **{
            k: {
                "mean": float(np.mean(v)),
                "min": float(np.min(v)),
                "max": float(np.max(v)),
            }
            for k, v in res.items()
        },
    }
