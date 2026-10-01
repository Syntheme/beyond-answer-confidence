"""Does a follow-up question add to the answer probability for flagging errors?

Offline; reads saved rows. For each pool of scored questions, error
detection by ``1 - p_max``, by ``1 - P(<follow-up>)`` and by both combined,
each as out-of-fold probabilities from a grouped 5-fold logistic model
(:func:`beyond_answer_confidence.stats.recalibration.cv_logistic_scores`): AUROC and
AURC per score, and the AUROC gain of the combination over ``p_max`` with an
item-bootstrap interval.

Pools (follow-up question, cross-validation group):

- PopQA plus fabricated entities, where every answer about a fabricated
  entity counts as an error ("do you know?", relation);
- dated yes/no news questions ("do you know?", month);
- HotpotQA and Quizbowl ("is there enough information?", question);
- SimpleQA, TriviaQA and TruthfulQA yes/no ("do you know?", item).

Inputs: ``knowledge_boundary``, ``evidence_sufficiency`` and ``benchmarks``
rows. Output: ``<output_dir>/combined_signals/summary.json``.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from beyond_answer_confidence.analyses.rows import records_frame
from beyond_answer_confidence.config import as_dict
from beyond_answer_confidence.experiments.base import Analysis, write_json
from beyond_answer_confidence.metrics.ranking import aurc, auroc
from beyond_answer_confidence.settings import Settings
from beyond_answer_confidence.stats.recalibration import cv_logistic_scores
from beyond_answer_confidence.stats.resampling import percentile_ci

NAME = "combined_signals"


@dataclass(frozen=True)
class Config:
    """Options.

    Attributes:
        bootstrap: Bootstrap resamples per pool (one generator, pools in
            order).
        seed: Bootstrap seed.
    """

    bootstrap: int = 2000
    seed: int = 0


def combined_signal(
    df: pd.DataFrame, meta: str, groups: np.ndarray, b: int, rng: np.random.Generator
) -> dict[str, Any]:
    """Error detection by ``p_max``, a follow-up question, and both.

    Args:
        df: Rows with ``p_max``, ``p_<meta>`` and a boolean ``error``.
        meta: Follow-up question name (column ``p_<meta>``).
        groups: Cross-validation group per row.
        b: Bootstrap resamples (item bootstrap of the paired AUROC gain).
        rng: Random generator.

    Returns:
        Items, error rate, AUROC and AURC per score, the AUROC gain of both
        over ``p_max`` with its 95 % interval and the share of resamples
        with no gain (``tail``).
    """
    err = df["error"].to_numpy(bool)
    x_p = (1 - df["p_max"].to_numpy(float))[:, None]
    x_m = (1 - df[f"p_{meta}"].to_numpy(float))[:, None]
    s_p = cv_logistic_scores(x_p, err, groups)
    s_m = cv_logistic_scores(x_m, err, groups)
    s_b = cv_logistic_scores(np.column_stack([x_p, x_m]), err, groups)
    n = len(err)
    diffs = []
    for _ in range(b):
        i = rng.integers(0, n, n)
        diffs.append(auroc(s_b[i], err[i]) - auroc(s_p[i], err[i]))
    d = np.array(diffs)

    def area(score: np.ndarray) -> float:
        # Abstain on the highest predicted error first.
        return aurc(-score, ~err)

    return {
        "n": n,
        "error_rate": float(err.mean()),
        "auroc": {
            "p_max": auroc(s_p, err),
            meta: auroc(s_m, err),
            "both": auroc(s_b, err),
        },
        "aurc": {"p_max": area(s_p), meta: area(s_m), "both": area(s_b)},
        "gain_both_minus_p_max": float(auroc(s_b, err) - auroc(s_p, err)),
        "gain_ci95": percentile_ci(d),
        "tail": float(np.mean(d <= 0)),
    }


def pools(
    boundary: pd.DataFrame, evidence: pd.DataFrame, benchmarks: pd.DataFrame
) -> dict[str, tuple[pd.DataFrame, str, str]]:
    """The question pools with their follow-up question and CV group column.

    Args:
        boundary: Knowledge-boundary rows.
        evidence: Evidence-sufficiency rows.
        benchmarks: Benchmark rows.

    Returns:
        Pool name -> ``(rows with error, follow-up name, group column)``.
    """
    out: dict[str, tuple[pd.DataFrame, str, str]] = {}
    pop = boundary[boundary["set"].isin(["popqa", "fabricated"])].copy()
    pop["error"] = pop["fabricated"].astype(bool) | ~pop["correct"].fillna(
        False
    ).astype(bool)
    out["popqa_plus_made_up"] = (pop, "known", "prop")
    tf = boundary[boundary["set"] == "oracle_tf"].copy()
    tf["error"] = ~tf["correct"].astype(bool)
    out["daily_oracle_yes_no"] = (tf, "known", "month")
    for name in ("hotpot", "quizbowl"):
        g = evidence[evidence["set"] == name].copy()
        g["error"] = ~g["correct"].astype(bool)
        out[name] = (g, "enough", "qid")
    for name in ("simpleqa", "triviaqa", "truthfulqa_binary"):
        g = benchmarks[benchmarks["set"] == name].copy()
        g["error"] = ~g["correct"].astype(bool)
        out[name] = (g, "known", "unit")
    return out


def analyse(
    boundary: pd.DataFrame,
    evidence: pd.DataFrame,
    benchmarks: pd.DataFrame,
    b: int,
    seed: int,
) -> dict[str, Any]:
    """Combined-signal results for every pool.

    Args:
        boundary: Knowledge-boundary rows.
        evidence: Evidence-sufficiency rows.
        benchmarks: Benchmark rows.
        b: Bootstrap resamples.
        seed: Seed.

    Returns:
        Pool -> :func:`combined_signal` result.
    """
    rng = np.random.default_rng(seed)
    return {
        name: combined_signal(
            df.reset_index(drop=True), meta, df[group].to_numpy(), b, rng
        )
        for name, (df, meta, group) in pools(boundary, evidence, benchmarks).items()
    }


def run(settings: Settings, config: Config) -> dict[str, Any]:
    """Run the analysis and write the summary.

    Args:
        settings: Run settings.
        config: Options.

    Returns:
        ``{"pools": ...}``.
    """
    res = {
        "pools": analyse(
            records_frame(settings, "knowledge_boundary"),
            records_frame(settings, "evidence_sufficiency"),
            records_frame(settings, "benchmarks"),
            config.bootstrap,
            config.seed,
        )
    }
    write_json(settings.out(NAME) / "summary.json", {"config": as_dict(config), **res})
    return res


ANALYSIS = Analysis(
    name=NAME,
    summary="Error detection by p_max, a follow-up question and both combined",
    config_type=Config,
    run=run,
)
