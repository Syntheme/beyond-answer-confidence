"""Paraphrase stability: does P(known) track how reliable an answer is?

Real PopQA questions are asked in four meaning-preserving wordings (see
:mod:`beyond_answer_confidence.tasks.paraphrases`). Per question, the answer
is *stable* if all four wordings pick the same option and *reliable* if all
four are correct. The analysis compares P(known) with answer confidence
(p_max) as predictors:

- **correctness** of each answer: AUROC of P(known) and of p_max, and their
  paired difference;
- **stability** and **reliability** of each question: AUROC of the mean
  P(known) and of the mean p_max over its wordings, and their difference;
- **beyond p_max**: out-of-fold AUROC for correctness of a logistic model on
  logit p_max alone and on logit p_max plus logit P(known);
- **consistency** of the scores themselves across wordings (mean standard
  deviation per question).

Bootstraps resample subjects (questions about the same subject together);
relation-level sensitivity is reported as leave-one-relation-out ranges.
"""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from beyond_answer_confidence.config import as_dict
from beyond_answer_confidence.experiments.base import (
    Experiment,
    analysable,
    backend_for,
    write_json,
    write_jsonl,
)
from beyond_answer_confidence.runner import run_items
from beyond_answer_confidence.settings import Settings
from beyond_answer_confidence.stats.resampling import (
    auroc_interval,
    paired_auroc_difference,
)
from beyond_answer_confidence.tasks.paraphrases import paraphrase_items
from beyond_answer_confidence.tasks.schema import Item, Task, item_tasks

logger = logging.getLogger(__name__)

NAME = "paraphrase_stability"
EPS = 1e-4


@dataclass(frozen=True)
class Config:
    """Options.

    Attributes:
        replicates: Replicates per item.
        per_relation: Questions per relation.
        seed: Item seed.
        bootstrap: Bootstrap resamples.
        bootstrap_seed: Bootstrap seed.
    """

    replicates: int = 1
    per_relation: int = 25
    seed: int = 0
    bootstrap: int = 2000
    bootstrap_seed: int = 0


def build_items(settings: Settings, config: Config) -> list[Item]:
    """The paraphrased questions.

    Args:
        settings: Run settings (data directory).
        config: Options.

    Returns:
        The items.
    """
    from beyond_answer_confidence.data import loaders

    return paraphrase_items(
        loaders.load_popqa(settings.data_dir), config.per_relation, config.seed
    )


def requests(settings: Settings, config: Config) -> list[Task]:
    """Every task of the experiment.

    Args:
        settings: Run settings.
        config: Options.

    Returns:
        The tasks.
    """
    return item_tasks(build_items(settings, config), config.replicates)


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.asarray(np.log(p / (1 - p)))


def beyond_confidence(
    df: pd.DataFrame, b: int, rng: np.random.Generator
) -> dict[str, Any]:
    """Out-of-fold correctness AUROC with and without P(known).

    Folds are grouped by subject (five folds).

    Args:
        df: Answer rows (``p_max``, ``p_known``, ``correct``, ``subject``).
        b: Bootstrap resamples.
        rng: Random generator.

    Returns:
        AUROC of each model and their paired difference (subject clusters).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold

    y = df["correct"].astype(bool).to_numpy()
    groups = df["subject"].astype(str).to_numpy()
    xs = {
        "p_max": _logit(df["p_max"].to_numpy())[:, None],
        "p_max_and_known": np.c_[
            _logit(df["p_max"].to_numpy()), _logit(df["p_known"].to_numpy())
        ],
    }
    oof = {k: np.empty(len(y)) for k in xs}
    for train, test in GroupKFold(5).split(np.zeros(len(y)), y, groups):
        for k, x in xs.items():
            m = LogisticRegression(max_iter=2000).fit(x[train], y[train])
            oof[k][test] = m.predict_proba(x[test])[:, 1]
    return {
        "p_max_only": auroc_interval(oof["p_max"], y, b, rng, groups),
        "p_max_and_known": auroc_interval(oof["p_max_and_known"], y, b, rng, groups),
        "difference": paired_auroc_difference(
            oof["p_max_and_known"], oof["p_max"], y, b, rng, groups
        ),
    }


def analyse(rows: list[dict[str, Any]], config: Config) -> dict[str, Any]:
    """Correctness, stability, reliability and consistency analyses.

    Args:
        rows: Scored rows (one per question and wording; refused absent).
        config: Options.

    Returns:
        The results.
    """
    rng = np.random.default_rng(config.bootstrap_seed)
    b = config.bootstrap
    df = pd.DataFrame(rows)
    df["correct"] = df["correct"].astype(bool)
    full = df.groupby("question_id").filter(lambda g: g["wording"].nunique() == 4)
    y = full["correct"].to_numpy()
    subj = full["subject"].astype(str).to_numpy()
    known, conf = full["p_known"].to_numpy(float), full["p_max"].to_numpy(float)
    q = full.groupby("question_id").agg(
        subject=("subject", "first"),
        prop=("prop", "first"),
        stable=("top", lambda s: s.nunique() == 1),
        reliable=("correct", "all"),
        accuracy=("correct", "mean"),
        known=("p_known", "mean"),
        conf=("p_max", "mean"),
        known_sd=("p_known", "std"),
        conf_sd=("p_max", "std"),
    )
    qs = q["subject"].astype(str).to_numpy()

    def question_level(target: str) -> dict[str, Any]:
        t = q[target].to_numpy(bool)
        loro = {
            str(r): paired_auroc_difference(
                q["known"].to_numpy()[q["prop"] != r],
                q["conf"].to_numpy()[q["prop"] != r],
                t[q["prop"] != r],
                b,
                rng,
                qs[q["prop"] != r],
            )["diff"]
            for r in sorted(q["prop"].unique())
        }
        return {
            "share": float(t.mean()),
            "auroc_mean_known": auroc_interval(q["known"].to_numpy(), t, b, rng, qs),
            "auroc_mean_p_max": auroc_interval(q["conf"].to_numpy(), t, b, rng, qs),
            "known_minus_p_max": paired_auroc_difference(
                q["known"].to_numpy(), q["conf"].to_numpy(), t, b, rng, qs
            ),
            "leave_one_relation_out_range": [min(loro.values()), max(loro.values())],
        }

    return {
        "n": {
            "questions": len(q),
            "questions_dropped_incomplete": int(df["question_id"].nunique() - len(q)),
            "answers": len(full),
            "subjects": len(set(subj)),
        },
        "accuracy": float(y.mean()),
        "accuracy_by_wording": {
            int(str(k)): float(g["correct"].mean()) for k, g in full.groupby("wording")
        },
        "correctness": {
            "auroc_known": auroc_interval(known, y, b, rng, subj),
            "auroc_p_max": auroc_interval(conf, y, b, rng, subj),
            "known_minus_p_max": paired_auroc_difference(known, conf, y, b, rng, subj),
        },
        "stability": question_level("stable"),
        "reliability": question_level("reliable"),
        "beyond_p_max": beyond_confidence(full, b, rng),
        "consistency": {
            "mean_sd_known": float(q["known_sd"].mean()),
            "mean_sd_p_max": float(q["conf_sd"].mean()),
        },
    }


def run(settings: Settings, config: Config) -> dict[str, Any]:
    """Collect (cache first), score, analyse and write the outputs.

    Args:
        settings: Run settings.
        config: Options.

    Returns:
        The summary.
    """
    rows, book = run_items(
        build_items(settings, config),
        cache=settings.cache_file(NAME),
        model=settings.model,
        replicates=config.replicates,
        backend=backend_for(settings),
        max_input_tokens=settings.max_input_tokens,
        concurrency=settings.concurrency,
        label=NAME,
    )
    out = settings.out(NAME)
    write_jsonl(out / "rows.jsonl", rows)
    summary: dict[str, Any] = {"config": as_dict(config), **book}
    if analysable(
        book["complete_items"], book["items"], book.get("errors"), book.get("fatal")
    ):
        summary["results"] = analyse(rows, config)
    else:
        logger.warning("%s: incomplete items; no analysis", NAME)
    write_json(out / "summary.json", summary)
    return summary


EXPERIMENT = Experiment(
    name=NAME,
    summary="Real questions in four wordings: does P(known) track answer reliability?",
    config_type=Config,
    requests=requests,
    run=run,
)
