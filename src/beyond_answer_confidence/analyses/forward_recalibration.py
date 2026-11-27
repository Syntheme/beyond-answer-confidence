"""Time-forward recalibration with information beyond top confidence.

On the yes/no news questions, a confidence-only map fitted on earlier months
leaves a large calibration gap after the cutoff. This analysis asks whether
recalibrators that may use more than the top probability do better, under
the same strictly time-forward protocol (:func:`forward_chaining`):

- ``isotonic_p_max``: isotonic regression on p_max (the confidence-only map);
- ``logistic_answer``: logistic regression on logit(p_max), the predicted
  answer and their interaction;
- ``logistic_answer_date``: as above plus the latest year and month written
  in the question (with a no-date indicator), a cue available at deployment;
- ``logistic_answer_date_text``: as above plus word and character n-grams of
  the question;
- ``logistic_answer_date_known``: ``logistic_answer_date`` plus logit
  P(known), the follow-up answer from the same call.

Every model is fitted on months before the test month only (*expanding*),
and once on the months before the cutoff (*frozen*). The gap is the mean
recalibrated probability of the predicted answer minus its accuracy; Brier
score and log loss are reported for the frozen fit after the cutoff.
"""

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from beyond_answer_confidence.analyses.shortcut_baselines import (
    Config as BaselineConfig,
)
from beyond_answer_confidence.analyses.shortcut_baselines import (
    load_inputs,
    news_frame,
)
from beyond_answer_confidence.config import as_dict
from beyond_answer_confidence.experiments.base import Analysis, write_json
from beyond_answer_confidence.settings import Settings
from beyond_answer_confidence.tasks.follow_up import CUTOFF_MONTH

NAME = "forward_recalibration"

EPS = 1e-4
"""Probability floor of the log-loss and logit features."""

MONTHS = {
    m: i + 1
    for i, m in enumerate(
        ["january", "february", "march", "april", "may", "june", "july",
         "august", "september", "october", "november", "december"]
    )
}  # fmt: skip
_DATE = re.compile(
    r"(?:(?P<month>"
    + "|".join(MONTHS)
    + r")\s+(?:\d{1,2},?\s+)?)?(?P<year>(?:19|20)\d\d)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Config:
    """Options.

    Attributes:
        item_seed: Seed of the source item sets.
        yes_no_per_month: News questions per month in the source set.
        cutoff_month: First month counted as after the knowledge cutoff.
        min_train_months: Months before the first expanding test month.
        text_c: Inverse regularisation of the text model.
    """

    item_seed: int = 0
    yes_no_per_month: int = 80
    cutoff_month: str = CUTOFF_MONTH
    min_train_months: int = 12
    text_c: float = 0.5


def latest_date(text: str) -> tuple[float, int]:
    """Latest date written in a question, in years since 2020.

    Args:
        text: Question text.

    Returns:
        ``(years since 2020 of the latest date, 1)``, or ``(0.0, 0)`` without
        a date. A year alone counts as its January.
    """
    found = []
    for m in _DATE.finditer(text):
        month = MONTHS.get((m.group("month") or "").lower(), 1)
        found.append(int(m.group("year")) - 2020 + (month - 1) / 12)
    return (max(found), 1) if found else (0.0, 0)


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.asarray(np.log(p / (1 - p)))


def features(df: pd.DataFrame, extra: tuple[str, ...]) -> np.ndarray:
    """Numeric features of the logistic recalibrators.

    Args:
        df: News frame (see :func:`news_frame`) with ``date``/``has_date``.
        extra: Optional blocks: ``"date"``, ``"known"``.

    Returns:
        Feature matrix.
    """
    lp = _logit(df["p_max"].to_numpy())
    yes = (df["top"].astype(str) == "yes").to_numpy(float)
    cols = [lp, yes, lp * yes]
    if "date" in extra:
        cols += [df["date"].to_numpy(float), df["has_date"].to_numpy(float)]
    if "known" in extra:
        cols.append(_logit(df["p_known"].to_numpy()))
    return np.column_stack(cols)


def make_models(text_c: float) -> dict[str, Callable[[pd.DataFrame], Any]]:
    """Factories: training frame -> fitted predictor of P(answer correct).

    Args:
        text_c: Inverse regularisation of the text model.

    Returns:
        Model name -> factory.
    """
    from scipy.sparse import csr_matrix, hstack
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    def isotonic(train: pd.DataFrame) -> Callable[[pd.DataFrame], np.ndarray]:
        f = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(
            train["p_max"].to_numpy(), train["correct"].to_numpy(float)
        )
        return lambda test: np.asarray(f.predict(test["p_max"].to_numpy()))

    def logistic(extra: tuple[str, ...]) -> Callable[[pd.DataFrame], Any]:
        def fit(train: pd.DataFrame) -> Callable[[pd.DataFrame], np.ndarray]:
            sc = StandardScaler().fit(features(train, extra))
            m = LogisticRegression(C=1.0, max_iter=5000).fit(
                sc.transform(features(train, extra)), train["correct"].to_numpy(int)
            )
            return lambda test: np.asarray(
                m.predict_proba(sc.transform(features(test, extra)))[:, 1]
            )

        return fit

    def with_text(train: pd.DataFrame) -> Callable[[pd.DataFrame], np.ndarray]:
        words = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        chars = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True
        )
        sc = StandardScaler().fit(features(train, ("date",)))

        def x(df: pd.DataFrame, fit: bool) -> Any:
            q = df["question"].astype(str).tolist()
            w = words.fit_transform(q) if fit else words.transform(q)
            c = chars.fit_transform(q) if fit else chars.transform(q)
            num = csr_matrix(sc.transform(features(df, ("date",))))
            return hstack([num, w, c]).tocsr()

        m = LogisticRegression(C=text_c, max_iter=5000).fit(
            x(train, True), train["correct"].to_numpy(int)
        )
        return lambda test: np.asarray(m.predict_proba(x(test, False))[:, 1])

    return {
        "isotonic_p_max": isotonic,
        "logistic_answer": logistic(()),
        "logistic_answer_date": logistic(("date",)),
        "logistic_answer_date_text": with_text,
        "logistic_answer_date_known": logistic(("date", "known")),
    }


def evaluate(
    df: pd.DataFrame,
    fit: Callable[[pd.DataFrame], Callable[[pd.DataFrame], np.ndarray]],
    min_train_months: int,
) -> dict[str, Any]:
    """Expanding and frozen time-forward evaluation of one recalibrator.

    Args:
        df: News frame with ``post``, ``month``, ``correct`` and features.
        fit: Training frame -> predictor.
        min_train_months: Months before the first expanding test month.

    Returns:
        Mean gaps by period (expanding), and the frozen fit's gap, Brier
        score and log loss after the cutoff.
    """
    months = sorted(df["month"].unique())
    rows = []
    for i, m in enumerate(months):
        if i < min_train_months:
            continue
        test = df[df["month"] == m]
        p = fit(df[df["month"] < m])(test)
        rows.append(
            {
                "month": m,
                "post": bool(test["post"].iloc[0]),
                "gap": float(p.mean() - test["correct"].mean()),
            }
        )
    exp = pd.DataFrame(rows)
    later = df[df["post"]]
    p = np.clip(fit(df[~df["post"]])(later), EPS, 1 - EPS)
    y = later["correct"].to_numpy(float)
    return {
        "expanding": {
            "pre_mean_gap": float(exp[~exp["post"]]["gap"].mean()),
            "post_mean_gap": float(exp[exp["post"]]["gap"].mean()),
            "post_mean_abs_gap": float(exp[exp["post"]]["gap"].abs().mean()),
        },
        "frozen_before_cutoff": {
            "post_gap": float(p.mean() - y.mean()),
            "post_brier": float(np.mean((p - y) ** 2)),
            "post_log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
        },
    }


def analyse(df: pd.DataFrame, config: Config) -> dict[str, Any]:
    """Evaluate every recalibrator on one model's news frame.

    Args:
        df: News frame (see :func:`news_frame`).
        config: Options.

    Returns:
        Counts, the raw gap and each recalibrator's evaluation.
    """
    df = df.copy()
    dates = [latest_date(q) for q in df["question"].astype(str)]
    df["date"] = [d for d, _ in dates]
    df["has_date"] = [h for _, h in dates]
    later = df[df["post"]]
    return {
        "n_pre": int((~df["post"]).sum()),
        "n_post": int(df["post"].sum()),
        "share_with_date": float(df["has_date"].mean()),
        "raw_post_gap": float(later["p_max"].mean() - later["correct"].mean()),
        "models": {
            name: evaluate(df, fit, config.min_train_months)
            for name, fit in make_models(config.text_c).items()
        },
    }


def run(settings: Settings, config: Config) -> dict[str, Any]:
    """Build the news frame from the knowledge-boundary rows and evaluate.

    Args:
        settings: Run settings.
        config: Options.

    Returns:
        The summary (``config`` and ``results``).
    """
    base = BaselineConfig(
        item_seed=config.item_seed,
        yes_no_per_month=config.yes_no_per_month,
        cutoff_month=config.cutoff_month,
    )
    inputs = load_inputs(settings, base)
    first: Mapping[str, Any] = {
        str(r["unit"]): r for r in inputs.rows["knowledge_boundary"]
    }
    summary = {
        "config": as_dict(config),
        "results": analyse(news_frame(inputs.news, first, config.cutoff_month), config),
    }
    write_json(settings.out(NAME) / "summary.json", summary)
    return summary


ANALYSIS = Analysis(
    name=NAME,
    summary="Time-forward recalibrators using the answer, dates in the question, "
    "text and P(known)",
    config_type=Config,
    run=run,
)
