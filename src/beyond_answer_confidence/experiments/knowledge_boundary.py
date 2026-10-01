"""Knowledge boundary: popularity, fabricated entities and time.

Three item sets, all asked as a choice plus the ``known`` yes/no question:

- every PopQA question as four-option multiple choice, graded by subject
  popularity (:mod:`beyond_answer_confidence.tasks.entities`);
- fabricated twins about entities that do not exist (ideal: uniform);
- dated news questions, sampled per month (:mod:`beyond_answer_confidence.tasks.news`);
  a change point in monthly accuracy locates the knowledge cutoff, after
  which confidence should fall.
"""

import logging
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd

from beyond_answer_confidence.config import as_dict
from beyond_answer_confidence.data import loaders
from beyond_answer_confidence.experiments.base import (
    Experiment,
    backend_for,
    write_json,
    write_jsonl,
)
from beyond_answer_confidence.metrics.ranking import auroc
from beyond_answer_confidence.runner import run_items
from beyond_answer_confidence.settings import Settings
from beyond_answer_confidence.stats.changepoint import change_point
from beyond_answer_confidence.stats.inference import (
    bootstrap_equivalence,
    holm_adjust_tails,
    mean_difference_interval,
    mean_interval,
)
from beyond_answer_confidence.stats.resampling import smece_interval
from beyond_answer_confidence.tasks.entities import fabricated_items, popqa_items
from beyond_answer_confidence.tasks.news import (
    multiple_choice_news_items,
    yes_no_news_items,
)
from beyond_answer_confidence.tasks.schema import Item, Task, item_tasks

logger = logging.getLogger(__name__)

NAME = "knowledge_boundary"


@dataclass(frozen=True)
class Config:
    """Options of the knowledge-boundary experiment.

    Attributes:
        replicates: Replicates per item.
        seed: Seed for distractors, option order, names and news sampling.
        fabricated_per_relation: Fabricated items per PopQA relation.
        yes_no_per_month: Yes/no news questions sampled per month.
        multiple_choice_per_month: Four-option news questions per month.
        bootstrap: Bootstrap resamples.
        bootstrap_seed: Bootstrap seed.
        calibration_margin: Equivalence margin for mean confidence minus
            accuracy.
        fabricated_p_max_reference: Reference mean top probability on
            fabricated items (0.25 is uniform).
        smece_reference: Reference SmoothECE.
        min_side_months: Minimum months on each side of the change point.
        min_drop: Accuracy drop at the change point above which the
            before/after comparisons are computed.
    """

    replicates: int = 3
    seed: int = 0
    fabricated_per_relation: int = 100
    yes_no_per_month: int = 80
    multiple_choice_per_month: int = 60
    bootstrap: int = 2000
    bootstrap_seed: int = 0
    calibration_margin: float = 0.05
    fabricated_p_max_reference: float = 0.30
    smece_reference: float = 0.05
    min_side_months: int = 6
    min_drop: float = 0.05


def build_items(settings: Settings, config: Config) -> list[Item]:
    """Load the datasets and build every item.

    Args:
        settings: Run settings (data directory).
        config: Options.

    Returns:
        PopQA items, fabricated twins, yes/no news, four-option news.
    """
    popqa = loaders.load_popqa(settings.data_dir)
    return (
        popqa_items(popqa, config.seed)
        + fabricated_items(popqa, config.fabricated_per_relation, config.seed)
        + yes_no_news_items(
            loaders.load_daily_oracle("tf", settings.data_dir),
            config.yes_no_per_month,
            config.seed,
        )
        + multiple_choice_news_items(
            loaders.load_daily_oracle("mc", settings.data_dir),
            config.multiple_choice_per_month,
            config.seed,
        )
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


def _gap(g: pd.DataFrame) -> np.ndarray:
    """Top probability minus correctness, per item."""
    return (g["p_max"] - g["correct"].astype(float)).to_numpy()


def analyse(rows: list[dict[str, Any]], config: Config) -> dict[str, Any]:
    """Calibration by popularity, fabricated entities and the cutoff.

    Bootstrap draws (one generator, in this order): calibration gap per
    popularity quintile (5); least vs most popular confidence (2);
    fabricated confidence; if the change point drops by at least
    ``min_drop``, calibration gap after it and after-vs-before confidence
    (2); PopQA SmoothECE.

    Args:
        rows: Scored rows.
        config: Options.

    Returns:
        Effect sizes with intervals (tail shares Holm-adjusted across the
        main comparisons) and descriptive summaries.
    """
    b = config.bootstrap
    rng = np.random.default_rng(config.bootstrap_seed)
    df = pd.DataFrame(rows)
    real = df[df["set"] == "popqa"].copy()
    real["quintile"] = pd.qcut(real["s_pop"].rank(method="first"), 5, labels=False)
    fab = df[df["set"] == "fabricated"]
    family: dict[str, dict[str, Any]] = {}
    per_q = {
        int(cast(int, q)): bootstrap_equivalence(
            _gap(g), config.calibration_margin, b, rng
        )
        | {
            "accuracy": float(g["correct"].mean()),
            "mean_p_max": float(g["p_max"].mean()),
        }
        for q, g in real.groupby("quintile")
    }
    family["calibration_by_popularity"] = {
        "quintiles": per_q,
        "p": max(v["p"] for v in per_q.values()),
    }
    family["least_vs_most_popular_confidence"] = mean_difference_interval(
        real[real["quintile"] == 0]["p_max"].to_numpy(),
        real[real["quintile"] == 4]["p_max"].to_numpy(),
        b,
        rng,
    )
    family["fabricated_confidence"] = mean_interval(
        fab["p_max"].to_numpy(), b, rng, threshold=config.fabricated_p_max_reference
    )
    tf = df[df["set"] == "oracle_tf"]
    monthly = (
        tf.groupby("month")
        .agg(acc=("correct", "mean"), n=("correct", "size"))
        .sort_index()
    )
    cp = change_point(
        list(monthly.index),
        list(monthly["acc"]),
        list(monthly["n"]),
        config.min_side_months,
    )
    if cp["drop"] >= config.min_drop:
        post = tf[tf["month"] >= cp["first_post_month"]]
        pre = tf[tf["month"] < cp["first_post_month"]]
        family["calibration_after_cutoff"] = bootstrap_equivalence(
            _gap(post), config.calibration_margin, b, rng
        )
        family["confidence_after_vs_before_cutoff"] = mean_difference_interval(
            post["p_max"].to_numpy(), pre["p_max"].to_numpy(), b, rng
        )
    res: dict[str, Any] = holm_adjust_tails(family)
    res["oracle_change_point"] = cp
    res["popqa_smece"] = smece_interval(
        real["p_max"].to_numpy(),
        real["correct"].to_numpy(),
        b,
        rng,
        threshold=config.smece_reference,
    )
    res["popqa_by_quintile"] = {
        int(cast(int, q)): {
            "s_pop_range": [int(g["s_pop"].min()), int(g["s_pop"].max())],
            "n": len(g),
            "accuracy": float(g["correct"].mean()),
            "mean_p_max": float(g["p_max"].mean()),
            "mean_p_known": float(g["p_known"].mean()),
        }
        for q, g in real.groupby("quintile")
    }
    res["popqa_by_relation"] = {
        str(p): {
            "n": len(g),
            "accuracy": float(g["correct"].mean()),
            "mean_p_max": float(g["p_max"].mean()),
            "fabricated_mean_p_max": float(fab[fab["prop"] == p]["p_max"].mean()),
        }
        for p, g in real.groupby("prop")
    }
    res["popqa_error_auroc"] = {
        "p_max": auroc(real["p_max"].to_numpy(), real["correct"].to_numpy(bool)),
        "p_known": auroc(real["p_known"].to_numpy(), real["correct"].to_numpy(bool)),
    }
    both = pd.concat([real, fab])
    obscure = pd.concat([real[real["quintile"] == 0], fab])
    res["fabricated_detection_auroc"] = {
        "1-p_known": auroc(
            1 - both["p_known"].to_numpy(), both["fabricated"].to_numpy(bool)
        ),
        "1-p_max": auroc(
            1 - both["p_max"].to_numpy(), both["fabricated"].to_numpy(bool)
        ),
        "vs_bottom_quintile_1-p_known": auroc(
            1 - obscure["p_known"].to_numpy(), obscure["fabricated"].to_numpy(bool)
        ),
    }
    res["fabricated"] = {
        "n": len(fab),
        "mean_p_max": float(fab["p_max"].mean()),
        "share_p_max_ge_0.5": float((fab["p_max"] >= 0.5).mean()),
        "mean_p_known": float(fab["p_known"].mean()),
    }
    for name in ("oracle_tf", "oracle_mc"):
        g = df[df["set"] == name]
        res[f"{name}_by_month"] = {
            str(m): {
                "n": len(x),
                "accuracy": float(x["correct"].mean()),
                "mean_p_max": float(x["p_max"].mean()),
                "mean_p_known": float(x["p_known"].mean()),
            }
            for m, x in g.groupby("month")
        }
    return res


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
    summary="Entity popularity, fabricated entities and news after the cutoff",
    config_type=Config,
    requests=requests,
    run=run,
)
