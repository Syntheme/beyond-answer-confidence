"""How free-text benchmarks were turned into multiple choice, with examples.

Three answer sets are converted to four options: PopQA (three wrong answers
sampled from the same relation's answer pool), TriviaQA (the three nearest
other answers in sentence-embedding space) and SimpleQA Verified (perturbed
variants for dates and numbers, otherwise the nearest answers of the same
answer type). HotpotQA's options are its supporting titles (or yes/no).
This analysis records the method, counts and seeded examples per set; it
rebuilds the items from the saved distractor choices and needs no model.
"""

import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from beyond_answer_confidence.config import as_dict
from beyond_answer_confidence.experiments.base import Analysis, write_json
from beyond_answer_confidence.settings import Settings
from beyond_answer_confidence.tasks.schema import Item

NAME = "distractor_examples"

METHODS = {
    "popqa": "three wrong answers sampled from the relation's answer pool "
    "(aliases of the gold answer excluded)",
    "triviaqa": "the three nearest other answers in all-MiniLM-L6-v2 "
    "embedding space (aliases excluded)",
    "simpleqa": "dates and numbers: three perturbed variants of the gold "
    "answer; otherwise the three nearest answers of the same answer type",
    "hotpot": "the question's supporting paragraph titles, or yes/no",
}


@dataclass(frozen=True)
class Config:
    """Options.

    Attributes:
        examples: Examples per set.
        seed: Seed of the example draw (and of the item builders).
    """

    examples: int = 5
    seed: int = 0


def describe(items: Sequence[Item], k: int, rng: random.Random) -> dict[str, Any]:
    """Counts and ``k`` seeded examples of one converted set.

    Args:
        items: The set's items.
        k: Examples.
        rng: Random generator.

    Returns:
        ``n_items``, the share of perturbed items (when recorded) and the
        examples (question, options in display order, gold option text).
    """
    pert = [it.info["perturbed"] for it in items if "perturbed" in it.info]
    out: dict[str, Any] = {"n_items": len(items)}
    if pert:
        out["share_perturbed"] = sum(pert) / len(pert)
    out["examples"] = [
        {
            "unit": it.unit,
            "question": str(it.state.get("question", it.state)),
            "options": [it.descriptions[o] for o in it.options]
            if it.descriptions
            else list(it.options),
            "gold": it.descriptions[it.gold]
            if it.descriptions and it.gold is not None
            else it.gold,
        }
        for it in rng.sample(list(items), min(k, len(items)))
    ]
    return out


def run(settings: Settings, config: Config) -> dict[str, Any]:
    """Rebuild the converted sets and write their descriptions.

    Args:
        settings: Run settings (data directory).
        config: Options.

    Returns:
        The summary (``config`` and ``results``).
    """
    from beyond_answer_confidence.data import loaders
    from beyond_answer_confidence.data.distractors import distractor_picker
    from beyond_answer_confidence.tasks.benchmarks import (
        simpleqa_items,
        triviaqa_items,
    )
    from beyond_answer_confidence.tasks.entities import popqa_items
    from beyond_answer_confidence.tasks.evidence import hotpot_items

    d = settings.data_dir
    pick = distractor_picker(d)
    sets = {
        "popqa": popqa_items(loaders.load_popqa(d), config.seed),
        "triviaqa": triviaqa_items(
            loaders.load_triviaqa_validation(d), pick, config.seed
        ),
        "simpleqa": simpleqa_items(loaders.load_simpleqa(d), pick, config.seed),
        "hotpot": hotpot_items(loaders.load_hotpot_validation(d), config.seed),
    }
    rng = random.Random(f"distractor-examples:{config.seed}")  # noqa: S311  # nosec B311
    results = {
        name: {"method": METHODS[name], **describe(items, config.examples, rng)}
        for name, items in sets.items()
    }
    summary = {"config": as_dict(config), "results": results}
    write_json(settings.out(NAME) / "summary.json", summary)
    return summary


ANALYSIS = Analysis(
    name=NAME,
    summary="Conversion of free-text benchmarks to multiple choice, with examples",
    config_type=Config,
    run=run,
)
