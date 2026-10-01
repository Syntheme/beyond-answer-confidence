"""Second look: is an answer judged correct when shown back to the model?

The first answer to each item is the replicate-mean top choice already
scored by an earlier experiment (read from its rows, not asked again). It
is shown back as ``proposed_answer`` with a yes/no question (is it
correct?) and a 0-100 % score question (see
:func:`beyond_answer_confidence.tasks.follow_up.second_look_item`).

Item sets: synthetic scenarios (four cells), PopQA across popularity
quintiles plus fabricated entities, yes/no news around the knowledge
cutoff, HotpotQA with paragraphs, and TriviaQA, SimpleQA and binary
TruthfulQA. The analysis compares P(correct) with the first answer's top
probability as an error detector, and checks it on fabricated entities,
post-cutoff news and synthetic cells with a known ideal.

Needs the rows of ``synthetic_worlds``, ``knowledge_boundary``,
``evidence_sufficiency`` and ``benchmarks`` under the output directory.
"""

import logging
import random
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from beyond_answer_confidence.config import as_dict
from beyond_answer_confidence.data import loaders
from beyond_answer_confidence.data.distractors import distractor_picker
from beyond_answer_confidence.experiments.base import (
    Experiment,
    backend_for,
    read_jsonl,
    source_rows,
    write_json,
    write_jsonl,
)
from beyond_answer_confidence.metrics.calibration import smece
from beyond_answer_confidence.metrics.ranking import auroc
from beyond_answer_confidence.runner import run_items
from beyond_answer_confidence.settings import Settings
from beyond_answer_confidence.stats.inference import holm_adjust_tails, mean_interval
from beyond_answer_confidence.stats.resampling import bootstrap, percentile_ci
from beyond_answer_confidence.tasks.benchmarks import (
    simpleqa_items,
    triviaqa_items,
    truthfulqa_items,
)
from beyond_answer_confidence.tasks.entities import fabricated_items, popqa_items
from beyond_answer_confidence.tasks.evidence import hotpot_items
from beyond_answer_confidence.tasks.follow_up import (
    CUTOFF_MONTH,
    cutoff_sample,
    percent_reading,
    popularity_sample,
    second_look_item,
)
from beyond_answer_confidence.tasks.news import yes_no_news_items
from beyond_answer_confidence.tasks.schema import Item, Task, item_tasks
from beyond_answer_confidence.tasks.synthetic import facts_for, make_scenario

logger = logging.getLogger(__name__)

NAME = "second_look"

SOURCES = (
    "synthetic_worlds",
    "knowledge_boundary",
    "evidence_sufficiency",
    "benchmarks",
)
"""Experiments whose rows hold the first answers."""

SYNTHETIC_CELLS = ("D0", "D3", "SFp", "SP")
"""Synthetic cells re-asked (unknown, decided, stated chances, past draws)."""


@dataclass(frozen=True)
class Config:
    """Options of the second-look experiment.

    Attributes:
        replicates: Replicates per item.
        seed: Seed of the source item sets and of the samples below.
        n_scenarios: Synthetic scenarios.
        fabricated_per_relation: Fabricated entities per relation in the
            source set.
        yes_no_per_month: News questions per month in the source set.
        per_quintile: PopQA items per popularity quintile.
        cutoff_month: First month counted as after the knowledge cutoff.
        trivia_questions: TriviaQA questions sampled.
        bootstrap: Bootstrap resamples.
        bootstrap_seed: Bootstrap seed.
        fabricated_reference: Reference mean P(correct) on fabricated
            entities.
        gap_reference: Reference post-cutoff gap (P(correct) minus accuracy).
        abs_error_reference: Reference mean absolute error from the ideal on
            synthetic cells.
    """

    replicates: int = 3
    seed: int = 0
    n_scenarios: int = 480
    fabricated_per_relation: int = 100
    yes_no_per_month: int = 80
    per_quintile: int = 400
    cutoff_month: str = CUTOFF_MONTH
    trivia_questions: int = 2000
    bootstrap: int = 2000
    bootstrap_seed: int = 0
    fabricated_reference: float = 0.30
    gap_reference: float = 0.10
    abs_error_reference: float = 0.10


def _rows_by_unit(path: Path) -> dict[str, dict[str, Any]]:
    return {r["unit"]: r for r in read_jsonl(path)}


def synthetic_items(
    rows: list[dict[str, Any]], n_scenarios: int, seed: int = 0
) -> list[Item]:
    """Second-look items for four synthetic cells of every scenario.

    The ideal P(correct) is 0.25 for ``D0`` (nothing known), 1 or 0 for
    ``D3`` (decided) and the stated chance of the proposed option otherwise.

    Args:
        rows: Synthetic-worlds rows.
        n_scenarios: Scenarios.
        seed: Scenario seed.

    Returns:
        Items (units ``synthetic:<scenario>:<cell>``).
    """
    by = {(r["scenario"], r["cell"]): r for r in rows if not r["alone"]}
    items = []
    for i in range(n_scenarios):
        sc = make_scenario(i, seed)
        for cell in SYNTHETIC_CELLS:
            r = by[(i, cell)]
            ideal = {"D0": 0.25, "D3": float(r["top"] == sc.gold)}.get(
                cell, sc.p_star[r["top"]]
            )
            unit = f"synthetic:{i}:{cell}"
            it = Item(
                unit=unit,
                state={
                    "facts": facts_for(sc, cell),
                    "question": f"What is {sc.thing}?",
                },
            )
            first = {
                "top": r["top"],
                "p_max": r["p_max"],
                "correct": r["top"] == sc.gold,
            }
            info = {"set": "synthetic", "cell": cell, "scenario": i, "ideal": ideal}
            items.append(second_look_item(it, first, unit, info))
    return items


def build_items(settings: Settings, config: Config) -> list[Item]:
    """Load the datasets and first answers, and build every item.

    Args:
        settings: Run settings (data and output directories).
        config: Options.

    Returns:
        Synthetic, PopQA, news, HotpotQA, TriviaQA, SimpleQA and TruthfulQA
        items.
    """
    d = settings.data_dir
    items = synthetic_items(
        read_jsonl(source_rows(settings, "synthetic_worlds")),
        config.n_scenarios,
        config.seed,
    )

    def look(it: Item, first: Mapping[str, Any], info: Mapping[str, Any]) -> Item:
        return second_look_item(it, first, it.unit, info)

    boundary = _rows_by_unit(source_rows(settings, "knowledge_boundary"))
    popqa = loaders.load_popqa(d)
    entities = popqa_items(popqa, config.seed) + fabricated_items(
        popqa, config.fabricated_per_relation, config.seed
    )
    for it, q in popularity_sample(entities, config.per_quintile, config.seed):
        items.append(
            look(
                it, boundary[it.unit], {"set": "popqa", "quintile": q, "made_up": False}
            )
        )
    items.extend(
        look(it, boundary[it.unit], {"set": "popqa", "quintile": -1, "made_up": True})
        for it in entities
        if it.info["set"] == "fabricated"
    )
    news = yes_no_news_items(
        loaders.load_daily_oracle("tf", d), config.yes_no_per_month, config.seed
    )
    items.extend(
        look(
            it,
            boundary[it.unit],
            {"set": "oracle_tf", "post": it.info["month"] >= config.cutoff_month},
        )
        for it in cutoff_sample(news, config.cutoff_month, config.seed)
    )
    evidence = _rows_by_unit(source_rows(settings, "evidence_sufficiency"))
    items.extend(
        look(it, evidence[it.unit], {"set": "hotpot", "cell": it.info["cell"]})
        for it in hotpot_items(loaders.load_hotpot_validation(d), config.seed)
        if it.info["cell"] != "closed"
    )
    bench = _rows_by_unit(source_rows(settings, "benchmarks"))
    pick = distractor_picker(d)
    trivia = triviaqa_items(loaders.load_triviaqa_validation(d), pick, config.seed)
    rng = random.Random(config.seed)  # noqa: S311  # nosec B311
    items.extend(
        look(it, bench[it.unit], {"set": "triviaqa"})
        for it in rng.sample(trivia, config.trivia_questions)
    )
    items.extend(
        look(it, bench[it.unit], {"set": "simpleqa"})
        for it in simpleqa_items(loaders.load_simpleqa(d), pick, config.seed)
    )
    truthful = truthfulqa_items(
        loaders.load_truthfulqa_binary(d), loaders.load_truthfulqa_mc1(d), config.seed
    )
    items.extend(
        look(it, bench[it.unit], {"set": "truthfulqa_binary"})
        for it in truthful
        if it.info["set"] == "truthfulqa_binary"
    )
    return items


def requests(settings: Settings, config: Config) -> list[Task]:
    """Every task of the experiment (needs the source experiments' rows).

    Args:
        settings: Run settings.
        config: Options.

    Returns:
        The tasks.
    """
    return item_tasks(build_items(settings, config), config.replicates)


def analyse(rows: list[dict[str, Any]], config: Config) -> dict[str, Any]:
    """Second-look P(correct) versus the first answer's top probability.

    Bootstrap draws (one generator, in this order): AUROC gain on PopQA plus
    fabricated entities, then on news; synthetic ``D0`` and ``SFp`` error
    from the ideal; P(correct) on fabricated entities; post-cutoff gap.

    Args:
        rows: Scored rows.
        config: Options.

    Returns:
        Effect sizes with intervals (Holm-adjusted tail shares) and
        per-set, per-cell, per-quintile and per-period summaries.
    """
    b = config.bootstrap
    rng = np.random.default_rng(config.bootstrap_seed)
    df = pd.DataFrame(rows)
    df["chance"] = df["s_chance"].map(percent_reading)
    df["error"] = ~df["first_correct"].astype(bool)

    def gain(g: pd.DataFrame) -> dict[str, Any]:
        err = g["error"].to_numpy(bool)
        s1 = 1 - g["first_p_max"].to_numpy(float)
        s2 = 1 - g["p_correct"].to_numpy(float)
        d = bootstrap(
            len(g), lambda i: auroc(s2[i], err[i]) - auroc(s1[i], err[i]), b, rng
        )
        return {
            "n": len(g),
            "auroc_first_p_max": auroc(s1, err),
            "auroc_p_correct": auroc(s2, err),
            "auroc_score_chance": auroc(1 - g["chance"].to_numpy(float), err),
            "gain": auroc(s2, err) - auroc(s1, err),
            "ci": list(percentile_ci(d)),
            "tail": float(np.mean(d <= 0)),
        }

    pop = df[df["set"] == "popqa"]
    tf = df[df["set"] == "oracle_tf"]
    pools = {"popqa_plus_made_up": gain(pop), "daily_oracle_yes_no": gain(tf)}
    made_up = pop[pop["made_up"].astype(bool)]
    post = tf[tf["post"].astype(bool)]
    syn = df[df["set"] == "synthetic"]
    ideal_error = {
        str(c): mean_interval(
            np.abs(g["p_correct"].to_numpy(float) - g["ideal"].to_numpy(float)),
            b,
            rng,
            threshold=config.abs_error_reference,
        )
        for c, g in syn.groupby("cell")
        if c in ("D0", "SFp")
    }
    res: dict[str, Any] = holm_adjust_tails(
        {
            "error_detection_gain": {
                "pools": pools,
                "tail": max(v["tail"] for v in pools.values()),
            },
            "made_up_entities_p_correct": mean_interval(
                made_up["p_correct"].to_numpy(float),
                b,
                rng,
                threshold=config.fabricated_reference,
            ),
            "post_cutoff_gap": mean_interval(
                (post["p_correct"] - post["first_correct"].astype(float)).to_numpy(),
                b,
                rng,
                threshold=config.gap_reference,
            ),
            "synthetic_ideal_error": {
                "cells": ideal_error,
                "tail": max(v["tail"] for v in ideal_error.values()),
            },
        }
    )
    res |= _descriptives(df, syn, pop, tf)
    return res


def _descriptives(
    df: pd.DataFrame, syn: pd.DataFrame, pop: pd.DataFrame, tf: pd.DataFrame
) -> dict[str, Any]:
    by_set: dict[str, Any] = {}
    for name, g in df.groupby("set"):
        ok = g["first_correct"].to_numpy(float)
        by_set[str(name)] = {
            "n": len(g),
            "first_accuracy": float(ok.mean()),
            "mean_first_p_max": float(g["first_p_max"].mean()),
            "mean_p_correct": float(g["p_correct"].mean()),
            "mean_score_chance": float(g["chance"].mean()),
            "smece_p_correct": smece(g["p_correct"].to_numpy(float), ok),
            "smece_first_p_max": smece(g["first_p_max"].to_numpy(float), ok),
            "error_auroc_p_correct": auroc(
                1 - g["p_correct"].to_numpy(float), ~ok.astype(bool)
            ),
            "error_auroc_first_p_max": auroc(
                1 - g["first_p_max"].to_numpy(float), ~ok.astype(bool)
            ),
            "corr_p_correct_p_max": float(
                np.corrcoef(g["p_correct"], g["first_p_max"])[0, 1]
            ),
        }
    return {
        "by_set": by_set,
        "synthetic_by_cell": {
            str(c): {
                "mean_ideal": float(g["ideal"].mean()),
                "mean_p_correct": float(g["p_correct"].mean()),
                "mean_score_chance": float(g["chance"].mean()),
                "mean_first_p_max": float(g["first_p_max"].mean()),
            }
            for c, g in syn.groupby("cell")
        },
        "popqa_by_quintile": {
            int(cast(int, q)): {
                "accuracy": float(g["first_correct"].mean()),
                "mean_p_correct": float(g["p_correct"].mean()),
                "mean_first_p_max": float(g["first_p_max"].mean()),
            }
            for q, g in pop.groupby("quintile")
        },
        "oracle_by_period": {
            ("post" if p else "pre"): {
                "accuracy": float(g["first_correct"].mean()),
                "mean_p_correct": float(g["p_correct"].mean()),
                "mean_first_p_max": float(g["first_p_max"].mean()),
            }
            for p, g in tf.groupby("post")
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
    summary="First answers shown back: is P(correct) a better error signal?",
    config_type=Config,
    requests=requests,
    run=run,
)
