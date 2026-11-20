"""How much do the wrong options matter? Two constructions on the same questions.

TriviaQA and PopQA questions are asked with nearest-neighbour and with
pool-sampled distractors (see
:mod:`beyond_answer_confidence.tasks.distractor_swap`). Per set and arm the
analysis reports accuracy, mean p_max, SmoothECE, the error AUROC of p_max
and of P(known), and selective accuracy at 50 % and 80 % coverage; and the
paired difference between arms (questions resampled).
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
from beyond_answer_confidence.metrics.calibration import smece
from beyond_answer_confidence.metrics.ranking import auroc
from beyond_answer_confidence.runner import run_items
from beyond_answer_confidence.settings import Settings
from beyond_answer_confidence.stats.resampling import percentile_ci
from beyond_answer_confidence.tasks.distractor_swap import (
    popqa_swap_items,
    triviaqa_swap_items,
)
from beyond_answer_confidence.tasks.schema import Item, Task, item_tasks

logger = logging.getLogger(__name__)

NAME = "distractor_swap"
ORIGINAL = {"triviaqa": "nearest", "popqa": "pool"}
"""The construction each set used in the main study."""


@dataclass(frozen=True)
class Config:
    """Options.

    Attributes:
        replicates: Replicates per item.
        per_set: Questions per set.
        seed: Item seed.
        bootstrap: Bootstrap resamples.
        bootstrap_seed: Bootstrap seed.
    """

    replicates: int = 1
    per_set: int = 1000
    seed: int = 0
    bootstrap: int = 2000
    bootstrap_seed: int = 0


def build_items(settings: Settings, config: Config) -> list[Item]:
    """TriviaQA and PopQA items in both arms.

    Args:
        settings: Run settings (data directory).
        config: Options.

    Returns:
        The items.
    """
    from beyond_answer_confidence.data import loaders
    from beyond_answer_confidence.data.distractors import distractor_picker

    d = settings.data_dir
    pick = distractor_picker(d)
    return triviaqa_swap_items(
        loaders.load_triviaqa_validation(d), pick, config.per_set, config.seed
    ) + popqa_swap_items(loaders.load_popqa(d), pick, config.per_set, config.seed)


def requests(settings: Settings, config: Config) -> list[Task]:
    """Every task of the experiment.

    Args:
        settings: Run settings.
        config: Options.

    Returns:
        The tasks.
    """
    return item_tasks(build_items(settings, config), config.replicates)


def arm_metrics(g: pd.DataFrame) -> dict[str, Any]:
    """Accuracy, confidence, calibration and selective accuracy of one arm.

    Args:
        g: Rows of one set and arm.

    Returns:
        The metrics.
    """
    conf = g["p_max"].to_numpy(float)
    ok = g["correct"].astype(bool).to_numpy()
    order = np.argsort(-conf, kind="stable")
    return {
        "n": len(g),
        "accuracy": float(ok.mean()),
        "mean_p_max": float(conf.mean()),
        "smece": float(smece(conf, ok)),
        "error_auroc_p_max": auroc(-conf, ~ok),
        "error_auroc_known": auroc(-g["p_known"].to_numpy(float), ~ok),
        "selective_accuracy_50": float(ok[order[: len(ok) // 2]].mean()),
        "selective_accuracy_80": float(ok[order[: int(len(ok) * 0.8)]].mean()),
    }


def analyse(rows: list[dict[str, Any]], config: Config) -> dict[str, Any]:
    """Per-arm metrics and paired differences per set.

    Args:
        rows: Scored rows.
        config: Options.

    Returns:
        The results.
    """
    rng = np.random.default_rng(config.bootstrap_seed)
    df = pd.DataFrame(rows)
    out: dict[str, Any] = {}
    for name, g in df.groupby("set"):
        both = g.groupby("question_id").filter(lambda h: h["arm"].nunique() == 2)
        wide = both.pivot_table(
            index="question_id",
            columns="arm",
            values=["correct", "p_max"],
            aggfunc="first",
        )
        orig = ORIGINAL[str(name)]
        other = "pool" if orig == "nearest" else "nearest"
        d_acc = (
            wide["correct"][other].astype(float) - wide["correct"][orig].astype(float)
        ).to_numpy()
        d_conf = (wide["p_max"][other] - wide["p_max"][orig]).to_numpy(float)
        idx = [rng.integers(0, len(d_acc), len(d_acc)) for _ in range(config.bootstrap)]
        out[str(name)] = {
            "original_arm": orig,
            "arms": {str(a): arm_metrics(h) for a, h in both.groupby("arm")},
            "other_minus_original": {
                "accuracy": float(d_acc.mean()),
                "accuracy_ci95": percentile_ci(
                    np.array([d_acc[i].mean() for i in idx])
                ),
                "mean_p_max": float(d_conf.mean()),
                "mean_p_max_ci95": percentile_ci(
                    np.array([d_conf[i].mean() for i in idx])
                ),
                "n_questions": len(d_acc),
            },
        }
    return out


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
    summary="TriviaQA and PopQA with nearest vs pool-sampled wrong options",
    config_type=Config,
    requests=requests,
    run=run,
)
