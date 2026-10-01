"""Evidence sufficiency: does uncertainty fall as the needed evidence is given?

HotpotQA comparison questions with zero, one or both supporting paragraphs
(plus a closed-book cell), and Quizbowl questions revealed clue by clue
(see :mod:`beyond_answer_confidence.tasks.evidence`). The analysis reports the
per-question slope of normalised entropy against evidence, accuracy gains,
calibration at each evidence level, and whether irrelevant paragraphs change
the entropy.
"""

import logging
import math
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd

from beyond_answer_confidence.config import as_dict
from beyond_answer_confidence.data import loaders
from beyond_answer_confidence.data.distractors import distractor_picker
from beyond_answer_confidence.experiments.base import (
    Experiment,
    backend_for,
    write_json,
    write_jsonl,
)
from beyond_answer_confidence.metrics.ranking import auroc
from beyond_answer_confidence.runner import run_items
from beyond_answer_confidence.settings import Settings
from beyond_answer_confidence.stats.inference import (
    bootstrap_equivalence,
    holm_adjust_tails,
    mean_interval,
)
from beyond_answer_confidence.stats.resampling import smece_interval
from beyond_answer_confidence.tasks.evidence import (
    QUIZBOWL_POOL_FOLDS,
    hotpot_items,
    quizbowl_items,
)
from beyond_answer_confidence.tasks.schema import Item, Task, item_tasks

logger = logging.getLogger(__name__)

NAME = "evidence_sufficiency"


@dataclass(frozen=True)
class Config:
    """Options of the evidence-sufficiency experiment.

    Attributes:
        replicates: Replicates per item.
        seed: Seed for option order and paragraph choice.
        quizbowl_fold: QANTA fold whose questions are asked.
        bootstrap: Bootstrap resamples.
        bootstrap_seed: Bootstrap seed.
        smece_reference: Reference SmoothECE.
        irrelevant_margin: Equivalence margin for the entropy change caused
            by irrelevant paragraphs.
    """

    replicates: int = 3
    seed: int = 0
    quizbowl_fold: str = "guessdev"
    bootstrap: int = 2000
    bootstrap_seed: int = 0
    smece_reference: float = 0.05
    irrelevant_margin: float = 0.05


def build_items(settings: Settings, config: Config) -> list[Item]:
    """Load the datasets and build every item.

    Args:
        settings: Run settings (data directory).
        config: Options.

    Returns:
        HotpotQA items, then Quizbowl items.
    """
    pool = pd.concat(
        [loaders.load_qanta(f, settings.data_dir) for f in QUIZBOWL_POOL_FOLDS]
    )
    return hotpot_items(
        loaders.load_hotpot_validation(settings.data_dir), config.seed
    ) + quizbowl_items(
        loaders.load_qanta(config.quizbowl_fold, settings.data_dir),
        pool,
        distractor_picker(settings.data_dir),
        config.seed,
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


def entropy_slopes(df: pd.DataFrame, x: str, group: str) -> np.ndarray:
    """Least-squares slope of normalised entropy on ``x`` within each group.

    Args:
        df: Rows with ``norm_entropy``, ``x`` and ``group`` columns.
        x: Evidence column.
        group: Grouping column (e.g. question id).

    Returns:
        One slope per group with at least two distinct ``x`` values.
    """
    out = []
    for _, g in df.groupby(group):
        xs = g[x].to_numpy(float)
        if len(set(xs)) < 2:
            continue
        ys = g["norm_entropy"].to_numpy(float)
        out.append(float(np.polyfit(xs, ys, 1)[0]))
    return np.array(out)


def analyse(rows: list[dict[str, Any]], config: Config) -> dict[str, Any]:
    """Entropy slopes, accuracy gains, calibration per evidence level.

    Bootstrap draws (one generator, in this order): HotpotQA slope and
    accuracy gain; Quizbowl slope and accuracy gain; SmoothECE for the six
    evidence cells; irrelevant-paragraph entropy change.

    Args:
        rows: Scored rows.
        config: Options.

    Returns:
        Effect sizes with intervals (tail shares Holm-adjusted across the
        four main comparisons) and descriptive summaries.
    """
    b = config.bootstrap
    rng = np.random.default_rng(config.bootstrap_seed)
    df = pd.DataFrame(rows)
    hp = df[df["set"] == "hotpot"].copy()
    qb = df[df["set"] == "quizbowl"].copy()
    dose = hp[hp["cell"] != "closed"].assign(d=lambda x: x["cell"].str[-1].astype(int))
    wide = hp.pivot(index="qid", columns="cell", values="correct").astype(float)
    ent = hp.pivot(index="qid", columns="cell", values="norm_entropy")

    def below_zero(values: np.ndarray) -> dict[str, Any]:
        return mean_interval(values, b, rng, threshold=0.0)

    hp_slope = below_zero(entropy_slopes(dose, "d", "qid"))
    hp_acc = below_zero((wide["dose0"] - wide["dose2"]).to_numpy())
    grouped: Any = qb.groupby("qid")
    qb_first_last = grouped.apply(
        lambda g: (
            float(g.loc[g["k"].idxmin(), "correct"])
            - float(g.loc[g["k"].idxmax(), "correct"])
        ),
        include_groups=False,
    )
    qb_slope = below_zero(entropy_slopes(qb, "fraction", "qid"))
    qb_acc = below_zero(qb_first_last.to_numpy(float))
    mid = qb[qb["k"] == qb["n_sentences"].map(lambda n: math.ceil(n / 2))]
    cells = {
        "hotpot_dose0": hp[hp["cell"] == "dose0"],
        "hotpot_dose1": hp[hp["cell"] == "dose1"],
        "hotpot_dose2": hp[hp["cell"] == "dose2"],
        "quizbowl_first": qb[qb["k"] == 1],
        "quizbowl_middle": mid,
        "quizbowl_full": qb[qb["k"] == qb["n_sentences"]],
    }
    calibration = {
        name: smece_interval(
            g["p_max"].to_numpy(),
            g["correct"].to_numpy(),
            b,
            rng,
            threshold=config.smece_reference,
        )
        for name, g in cells.items()
    }
    irrelevant = bootstrap_equivalence(
        (ent["dose0"] - ent["closed"]).to_numpy(), config.irrelevant_margin, b, rng
    )
    res: dict[str, Any] = holm_adjust_tails(
        {
            "hotpot_evidence": {
                "entropy_slope": hp_slope,
                "accuracy_dose0_minus_dose2": hp_acc,
                "tail": max(hp_slope["tail"], hp_acc["tail"]),
            },
            "quizbowl_clues": {
                "entropy_slope": qb_slope,
                "accuracy_first_minus_full": qb_acc,
                "tail": max(qb_slope["tail"], qb_acc["tail"]),
            },
            "calibration_by_evidence": {
                "cells": calibration,
                "tail": max(v["tail"] for v in calibration.values()),
            },
            "irrelevant_paragraphs": irrelevant,
        }
    )
    res |= _descriptives(hp, qb, cells["quizbowl_full"])
    return res


def _descriptives(
    hp: pd.DataFrame, qb: pd.DataFrame, qb_full: pd.DataFrame
) -> dict[str, Any]:
    return {
        "hotpot_by_cell": {
            str(c): {
                "n": len(g),
                "accuracy": float(g["correct"].mean()),
                "mean_p_max": float(g["p_max"].mean()),
                "mean_norm_entropy": float(g["norm_entropy"].mean()),
                "mean_p_enough": float(g["p_enough"].mean()),
                "by_kind_accuracy": {
                    str(k): float(x["correct"].mean()) for k, x in g.groupby("kind")
                },
            }
            for c, g in hp.groupby("cell")
        },
        "hotpot_enough_auroc_dose2_vs_rest": auroc(
            hp["p_enough"].to_numpy(), (hp["cell"] == "dose2").to_numpy()
        ),
        "quizbowl_by_k": {
            int(cast(int, k)): {
                "n": len(g),
                "accuracy": float(g["correct"].mean()),
                "mean_p_max": float(g["p_max"].mean()),
                "mean_p_enough": float(g["p_enough"].mean()),
            }
            for k, g in qb.groupby("k")
        },
        "quizbowl_by_fraction_bin": {
            str(k): {
                "n": len(g),
                "accuracy": float(g["correct"].mean()),
                "mean_p_max": float(g["p_max"].mean()),
            }
            for k, g in qb.groupby(
                pd.cut(qb["fraction"], [0, 0.25, 0.5, 0.75, 1.0]), observed=True
            )
        },
        "quizbowl_by_category_full": {
            str(c): float(g["correct"].mean()) for c, g in qb_full.groupby("category")
        },
        "error_auroc": {
            "hotpot_p_max": auroc(hp["p_max"].to_numpy(), hp["correct"].to_numpy(bool)),
            "quizbowl_p_max": auroc(
                qb["p_max"].to_numpy(), qb["correct"].to_numpy(bool)
            ),
            "quizbowl_p_enough": auroc(
                qb["p_enough"].to_numpy(), qb["correct"].to_numpy(bool)
            ),
        },
    }


def run(settings: Settings, config: Config) -> dict[str, Any]:
    """Collect (cache first), score, analyse and write the outputs.

    Args:
        settings: Run settings.
        config: Options.

    Returns:
        The summary (``results`` only when every item is complete).
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
    if book["complete_items"] == book["items"]:
        summary["results"] = analyse(rows, config)
    else:
        logger.warning("%s: incomplete items; no analysis", NAME)
    write_json(out / "summary.json", summary)
    return summary


EXPERIMENT = Experiment(
    name=NAME,
    summary="HotpotQA paragraphs and Quizbowl clues: uncertainty versus evidence",
    config_type=Config,
    requests=requests,
    run=run,
)
