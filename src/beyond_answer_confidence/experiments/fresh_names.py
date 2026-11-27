"""Fresh made-up names against matched real people: does ``known`` beat confidence?

A confirmation of the look-alike contrast on people the earlier sets did not
use (see :mod:`beyond_answer_confidence.tasks.fresh_names`). The analysis is
fixed in advance:

- **Primary endpoint**: ``D`` = AUROC(1 - P(known)) - AUROC(1 - p_max), made-up
  names positive, with a paired bootstrap whose clusters are first-name
  donors (a real person is their own cluster).
- **Decision rules**: ``known_better`` if the 95 % interval lies above 0,
  ``known_worse`` if it lies below 0, ``equivalent`` if the 90 % interval
  lies inside +/- ``margin``, otherwise ``inconclusive`` (``known_better`` or
  ``known_worse`` take precedence over ``equivalent``).
- **Secondary**: ``D`` clustered by surname donor and by relation, leaving
  out one relation at a time; each score's AUROC; text-only controls fitted
  out of fold with folds grouped by subject (character and word n-grams,
  sentence embeddings, name-structure features, embeddings plus
  name-structure features), each with a permutation reference (the text
  identifies made-up names only if its AUROC beats the label-shuffled
  refits); shares of refused items per group.

Items whose request was refused are left out of every metric (the summary
counts them); a run with ``--refusals uniform`` gives the no-preference
reading.
"""

import logging
import random
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
from beyond_answer_confidence.metrics.ranking import auroc
from beyond_answer_confidence.runner import run_items
from beyond_answer_confidence.settings import Settings
from beyond_answer_confidence.stats.resampling import (
    auroc_interval,
    paired_auroc_difference,
)
from beyond_answer_confidence.tasks.fresh_names import BAND, fresh_items
from beyond_answer_confidence.tasks.schema import Item, Task, item_tasks

logger = logging.getLogger(__name__)

NAME = "fresh_names"


@dataclass(frozen=True)
class Config:
    """Options.

    Attributes:
        replicates: Replicates per item.
        seed: Item seed.
        lookalike_per_relation: Size of the earlier look-alike set, whose
            names are excluded.
        band_low: Lower popularity quantile (exclusive).
        band_high: Upper popularity quantile (inclusive).
        donor_share: Share of each relation's people used as donors.
        margin: Equivalence margin for ``D``.
        bootstrap: Bootstrap resamples.
        bootstrap_seed: Bootstrap seed.
        permutations: Label shuffles for the text-only permutation reference.
    """

    replicates: int = 3
    seed: int = 0
    lookalike_per_relation: int = 100
    band_low: float = BAND[0]
    band_high: float = BAND[1]
    donor_share: float = 0.5
    margin: float = 0.05
    bootstrap: int = 2000
    bootstrap_seed: int = 0
    permutations: int = 50


def build_items(settings: Settings, config: Config) -> list[Item]:
    """The fresh made-up and real items.

    Args:
        settings: Run settings (data directory).
        config: Options.

    Returns:
        The items.
    """
    from beyond_answer_confidence.data import loaders
    from beyond_answer_confidence.tasks import shortcuts

    popqa = loaders.load_popqa(settings.data_dir)
    rng = random.Random(f"{shortcuts._SEED_NAMESPACE}:lookalike")  # noqa: S311  # nosec B311
    earlier = {
        n
        for picks in shortcuts.lookalike_subjects(
            popqa, config.lookalike_per_relation, rng
        ).values()
        for n, _, _ in picks
    }
    return fresh_items(
        popqa,
        earlier,
        config.seed,
        (config.band_low, config.band_high),
        config.donor_share,
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


def verdict(diff: dict[str, Any], margin: float) -> str:
    """Decision for the primary endpoint (see the module docstring).

    Args:
        diff: ``paired_auroc_difference`` result (``ci95``, ``ci90``).
        margin: Equivalence margin.

    Returns:
        ``known_better``, ``known_worse``, ``equivalent`` or ``inconclusive``.
    """
    lo95, hi95 = diff["ci95"]
    lo90, hi90 = diff["ci90"]
    if lo95 > 0:
        return "known_better"
    if hi95 < 0:
        return "known_worse"
    if -margin < lo90 and hi90 < margin:
        return "equivalent"
    return "inconclusive"


def analyse(
    rows: list[dict[str, Any]],
    items: list[Item],
    config: Config,
    embed: Any,
    reference_names: list[str],
) -> dict[str, Any]:
    """Primary endpoint, decision and secondary analyses.

    Args:
        rows: Scored rows (refused items absent).
        items: All items (for counts and question texts).
        config: Options.
        embed: Texts -> sentence embeddings.
        reference_names: PopQA person names for the name-frequency features.

    Returns:
        The results.
    """
    from beyond_answer_confidence.analyses.cue_controls import (
        fit_text_controls,
        name_features,
    )

    rng = np.random.default_rng(config.bootstrap_seed)
    b = config.bootstrap
    df = pd.DataFrame(rows)
    y = (df["set"] == "made_up").to_numpy(bool)
    first = np.where(y, df["first_donor"].astype(str), df["subject"].astype(str))
    last = np.where(y, df["surname_donor"].astype(str), df["subject"].astype(str))
    rel = df["prop"].astype(str).to_numpy()
    a = -df["p_known"].to_numpy(float)
    c = -df["p_max"].to_numpy(float)
    primary = paired_auroc_difference(a, c, y, b, rng, first)
    totals = pd.Series([it.info["set"] for it in items]).value_counts()
    answered = df["set"].value_counts()
    loro = {}
    for r in sorted(set(rel)):
        keep = rel != r
        loro[r] = paired_auroc_difference(
            a[keep], c[keep], y[keep], b, rng, first[keep]
        )["diff"]
    question = {it.unit: str(it.state["question"]) for it in items}
    texts = [question[u] for u in df["unit"]]
    names = df["subject"].astype(str).tolist()
    real = df[~y]
    return {
        "n": {
            "made_up": int(y.sum()),
            "real": int((~y).sum()),
            "refused_share": {
                s: float(1 - answered.get(s, 0) / totals[s]) for s in totals.index
            },
            "first_name_clusters": len(set(first)),
        },
        "primary": {
            "endpoint": "AUROC(1 - P(known)) - AUROC(1 - p_max), made up positive",
            "margin": config.margin,
            **primary,
            "verdict": verdict(primary, config.margin),
        },
        "auroc": {
            "one_minus_known": auroc_interval(a, y, b, rng, first),
            "one_minus_p_max": auroc_interval(c, y, b, rng, first),
        },
        "secondary": {
            "by_surname_donor": paired_auroc_difference(a, c, y, b, rng, last),
            "by_relation": paired_auroc_difference(a, c, y, b, rng, rel),
            "leave_one_relation_out": loro,
            "text_only_folds_by_subject": fit_text_controls(
                texts,
                y,
                np.array(names),
                embed(texts),
                name_features(names, reference_names),
                b,
                rng,
                first,
                config.permutations,
            ),
        },
        "descriptive": {
            "mean_p_known": {
                s: float(g["p_known"].mean()) for s, g in df.groupby("set")
            },
            "mean_p_max": {s: float(g["p_max"].mean()) for s, g in df.groupby("set")},
            "real_accuracy": float(real["correct"].astype(bool).mean()),
            "auroc_known_by_relation": {
                str(r): auroc(
                    -g["p_known"].to_numpy(float), (g["set"] == "made_up").to_numpy()
                )
                for r, g in df.groupby("prop")
            },
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
    from beyond_answer_confidence.data import loaders
    from beyond_answer_confidence.data.distractors import embed
    from beyond_answer_confidence.tasks.entities import PERSON_RELATIONS

    items = build_items(settings, config)
    rows, book = run_items(
        items,
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
        popqa = loaders.load_popqa(settings.data_dir)
        persons = [
            str(s)
            for s, p in zip(popqa["subj"], popqa["prop"], strict=True)
            if p in PERSON_RELATIONS
        ]
        summary["results"] = analyse(
            rows,
            items,
            config,
            lambda texts: embed(texts, settings.data_dir),
            persons,
        )
    else:
        logger.warning("%s: incomplete items; no analysis", NAME)
    write_json(out / "summary.json", summary)
    return summary


EXPERIMENT = Experiment(
    name=NAME,
    summary="Fresh made-up names vs matched real people: known vs confidence "
    "(fixed analysis)",
    config_type=Config,
    requests=requests,
    run=run,
)
