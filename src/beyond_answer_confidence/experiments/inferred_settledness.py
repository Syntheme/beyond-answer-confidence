"""Inferred settledness: is ``settled`` right when the status is only implied?

Made-up events in a past and a future version that differ only in a cue
(see :mod:`beyond_answer_confidence.tasks.settledness`): an explicit statement
(baseline), the tense, or a date with or without today's date. The analysis
reports the AUROC of P(settled) for past versus future versions in each
condition, with events resampled as clusters.
"""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from beyond_answer_confidence.config import as_dict
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
    auroc_vs_threshold,
    holm_adjust_tails,
)
from beyond_answer_confidence.tasks.schema import Item, Task, item_tasks
from beyond_answer_confidence.tasks.settledness import (
    CONDITIONS,
    STATUSES,
    settledness_items,
)

logger = logging.getLogger(__name__)

NAME = "inferred_settledness"


@dataclass(frozen=True)
class Config:
    """Options of the inferred-settledness experiment.

    Attributes:
        replicates: Replicates per item.
        events_per_kind: Events per event kind.
        seed: Event-generation seed.
        bootstrap: Bootstrap resamples.
        bootstrap_seed: Bootstrap seed.
        auroc_reference: Reference AUROC.
    """

    replicates: int = 3
    events_per_kind: int = 60
    seed: int = 0
    bootstrap: int = 2000
    bootstrap_seed: int = 0
    auroc_reference: float = 0.80


def build_items(config: Config) -> list[Item]:
    """Every item.

    Args:
        config: Options.

    Returns:
        Items.
    """
    return settledness_items(config.events_per_kind, config.seed)


def requests(settings: Settings, config: Config) -> list[Task]:
    """Every task of the experiment.

    Args:
        settings: Run settings (unused; the items are generated).
        config: Options.

    Returns:
        The tasks.
    """
    del settings
    return item_tasks(build_items(config), config.replicates)


def analyse(rows: list[dict[str, Any]], config: Config) -> dict[str, Any]:
    """AUROC of P(settled) for past versus future, per condition.

    Bootstrap draws (one generator, in this order): ``tense``,
    ``date_today``, lottery events under ``tense`` and ``date_today``
    (Holm-adjusted family), then ``explicit`` and ``date_only``.

    Args:
        rows: Scored rows.
        config: Options.

    Returns:
        AUROC with cluster-bootstrap intervals and tail shares at or below
        the reference; per-kind AUROC, mean P(settled) and choice summaries.
    """
    rng = np.random.default_rng(config.bootstrap_seed)
    df = pd.DataFrame(rows)
    df["past"] = df["status"] == "past"

    def separation(g: pd.DataFrame) -> dict[str, Any]:
        return auroc_vs_threshold(
            g["p_settled"].to_numpy(float),
            g["past"].to_numpy(bool),
            config.auroc_reference,
            config.bootstrap,
            rng,
            clusters=g["event"].to_numpy(),
        )

    lottery = df[(df["domain"] == "lottery") & df["cond"].isin(["tense", "date_today"])]
    res: dict[str, Any] = holm_adjust_tails(
        {
            "tense": separation(df[df["cond"] == "tense"]),
            "date_with_today": separation(df[df["cond"] == "date_today"]),
            "lottery_tense_and_date": separation(lottery),
        }
    )
    res |= {
        "explicit": separation(df[df["cond"] == "explicit"]),
        "date_only": separation(df[df["cond"] == "date_only"]),
        "auroc_by_kind": {
            c: {
                str(d): auroc(g["p_settled"].to_numpy(float), g["past"].to_numpy(bool))
                for d, g in df[df["cond"] == c].groupby("domain")
            }
            for c in CONDITIONS
        },
        "mean_p_settled": {
            c: {
                s: float(
                    df[(df["cond"] == c) & (df["status"] == s)]["p_settled"].mean()
                )
                for s in STATUSES
            }
            for c in CONDITIONS
        },
        "choice": {
            str(c): {
                "mean_p_max": float(g["p_max"].mean()),
                "share_first_listed": float((g["top"] == g["first_listed"]).mean()),
                "share_p_max_ge_0.5": float((g["p_max"] >= 0.5).mean()),
            }
            for c, g in df.groupby("cond")
        },
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
        build_items(config),
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
    summary="Past or future events cued by tense or date: does settled follow?",
    config_type=Config,
    requests=requests,
    run=run,
)
