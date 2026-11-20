"""Where and how news accuracy changes, without choosing a window by eye.

The change point of :mod:`knowledge_cutoff` keeps at least six months on
each side, so on a panel ending in July 2026 it cannot place a change after
February 2026. This analysis describes the dated yes/no news panel in three
complementary ways:

- **Quarters**: accuracy, mean confidence and their gap per calendar quarter.
- **Change points with shorter sides**: the weighted two-mean change point
  with at least ``min_side`` months per side, on the whole panel and on the
  months from ``late_start`` on, each repeated in a within-month bootstrap
  (selection re-run in every resample) for the distribution of the selected
  month and an interval for the drop.
- **Trends**: least-squares slopes of accuracy and of the gap (confidence
  minus correctness) per year over the months from ``trend_start`` on, with
  month-cluster bootstrap intervals; no window is selected.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from beyond_answer_confidence.analyses.cutoff_calibration import NEWS_SET
from beyond_answer_confidence.analyses.knowledge_cutoff import resample_within_months
from beyond_answer_confidence.analyses.rows import experiment_frame
from beyond_answer_confidence.config import as_dict
from beyond_answer_confidence.experiments.base import Analysis, write_json
from beyond_answer_confidence.settings import Settings
from beyond_answer_confidence.stats.changepoint import change_point
from beyond_answer_confidence.stats.resampling import percentile_ci

NAME = "accuracy_trend"


@dataclass(frozen=True)
class Config:
    """Options.

    Attributes:
        bootstrap: Bootstrap resamples.
        seed: Seed (one generator, used in the order of the module docstring).
        min_side: Minimum months on each side of a change point.
        late_start: First month of the late-panel change-point search.
        trend_start: First month of the trend fits.
    """

    bootstrap: int = 2000
    seed: int = 0
    min_side: int = 3
    late_start: str = "2025-01"
    trend_start: str = "2024-11"


def quarter(month: str) -> str:
    """Calendar quarter of a ``YYYY-MM`` month.

    Args:
        month: Month.

    Returns:
        ``YYYY-Qn``.
    """
    return f"{month[:4]}-Q{(int(month[5:7]) - 1) // 3 + 1}"


def by_quarter(tf: pd.DataFrame) -> list[dict[str, Any]]:
    """Accuracy, mean confidence and gap per quarter.

    Args:
        tf: Dated yes/no rows (``month``, ``correct``, ``p_max``).

    Returns:
        One record per quarter, in order.
    """
    g = tf.assign(q=tf["month"].map(quarter)).groupby("q")
    out = []
    for q, d in g:
        acc, conf = float(d["correct"].mean()), float(d["p_max"].mean())
        out.append(
            {
                "quarter": str(q),
                "n": len(d),
                "accuracy": acc,
                "confidence": conf,
                "gap": conf - acc,
            }
        )
    return out


def _select(tf: pd.DataFrame, min_side: int) -> dict[str, Any]:
    monthly = (
        tf.groupby("month").agg(acc=("correct", "mean"), n=("correct", "size"))
    ).sort_index()
    return change_point(
        list(monthly.index), list(monthly["acc"]), list(monthly["n"]), min_side
    )


def change_point_block(
    tf: pd.DataFrame, min_side: int, b: int, rng: np.random.Generator
) -> dict[str, Any]:
    """Change point and its selection-aware bootstrap.

    Args:
        tf: Dated yes/no rows.
        min_side: Minimum months on each side.
        b: Resamples.
        rng: Random generator.

    Returns:
        The selected split, the share of resamples selecting each month and
        a 95 % interval for the drop at the re-selected split.
    """
    cp = _select(tf, min_side)
    months, drops = [], []
    for _ in range(b):
        r = _select(resample_within_months(tf, rng), min_side)
        months.append(r["first_post_month"])
        drops.append(r["drop"])
    counts = pd.Series(months).value_counts()
    return {
        "months": f"{tf['month'].min()}..{tf['month'].max()}",
        "min_side": min_side,
        "selected": cp,
        "selected_month_share": {
            str(k): float(v / b) for k, v in counts.sort_index().items()
        },
        "drop_ci95": percentile_ci(np.array(drops)),
    }


def _slope(x: np.ndarray, y: np.ndarray) -> float:
    xc = x - x.mean()
    return float((xc * (y - y.mean())).sum() / (xc**2).sum())


def trends(tf: pd.DataFrame, b: int, rng: np.random.Generator) -> dict[str, Any]:
    """Per-year slopes of accuracy and gap, month-cluster bootstrap.

    Args:
        tf: Dated yes/no rows from the trend start on.
        b: Resamples.
        rng: Random generator.

    Returns:
        Slopes with 95 % intervals and the number of months.
    """
    t = tf["month"].map(lambda m: int(m[:4]) + (int(m[5:7]) - 1) / 12).to_numpy()
    acc = tf["correct"].to_numpy(float)
    gap = tf["p_max"].to_numpy(float) - acc
    groups = tf.groupby("month").indices
    keys = sorted(groups, key=str)
    sa, sg = [], []
    for _ in range(b):
        pick = np.concatenate(
            [groups[keys[i]] for i in rng.integers(0, len(keys), len(keys))]
        )
        sa.append(_slope(t[pick], acc[pick]))
        sg.append(_slope(t[pick], gap[pick]))
    return {
        "months": len(keys),
        "accuracy_per_year": _slope(t, acc),
        "accuracy_per_year_ci95": percentile_ci(np.array(sa)),
        "gap_per_year": _slope(t, gap),
        "gap_per_year_ci95": percentile_ci(np.array(sg)),
    }


def analyse(tf: pd.DataFrame, config: Config) -> dict[str, Any]:
    """Quarters, change points and trends of one model's news panel.

    Args:
        tf: Dated yes/no rows.
        config: Options.

    Returns:
        The three descriptions.
    """
    rng = np.random.default_rng(config.seed)
    b = config.bootstrap
    late = tf[tf["month"] >= config.late_start].reset_index(drop=True)
    return {
        "n": len(tf),
        "by_quarter": by_quarter(tf),
        "change_point_full": change_point_block(tf, config.min_side, b, rng),
        "change_point_late": change_point_block(late, config.min_side, b, rng),
        "trend_after_start": trends(
            tf[tf["month"] >= config.trend_start].reset_index(drop=True), b, rng
        ),
    }


def run(settings: Settings, config: Config) -> dict[str, Any]:
    """Read the dated yes/no news rows, analyse and write the summary.

    Args:
        settings: Run settings.
        config: Options.

    Returns:
        The summary (``config`` and ``results``).
    """
    rows = experiment_frame(settings, "knowledge_boundary", dtype={"month": str})
    tf = rows[rows["set"] == NEWS_SET].reset_index(drop=True)
    tf["correct"] = tf["correct"].astype(bool)
    summary = {"config": as_dict(config), "results": analyse(tf, config)}
    write_json(settings.out(NAME) / "summary.json", summary)
    return summary


ANALYSIS = Analysis(
    name=NAME,
    summary="News accuracy by quarter, short-sided change points and trends",
    config_type=Config,
    run=run,
)
