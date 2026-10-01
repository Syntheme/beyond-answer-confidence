"""Follow-up questions on the item sets of the core experiments.

Four probes re-ask existing items in a changed form:

- **wording**: the ``settled``, ``known`` and ``enough`` yes/no questions in
  several paraphrases (some reversed), on synthetic scenarios (including four
  adversarial cells), PopQA and fabricated entities, and HotpotQA
  paragraphs;
- **odds formats**: the stated chances of a synthetic scenario asked as one
  yes/no question per option, one score per option, or an instructed choice;
- **cutoff**: yes/no news questions with today's date and/or an "unknown"
  option;
- **warned paragraphs**: HotpotQA with instructions saying the paragraphs
  may be irrelevant.

A fifth builder, :func:`second_look_item`, shows a first answer back as
``proposed_answer`` and asks whether it is correct. Every prompt string here
is part of the request payload; changing one changes the cache key.
"""

import dataclasses
import random
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from beyond_answer_confidence.tasks.evidence import HOTPOT_ENOUGH_QUESTION
from beyond_answer_confidence.tasks.multiple_choice import KNOWN_QUESTION
from beyond_answer_confidence.tasks.schema import Item
from beyond_answer_confidence.tasks.synthetic import (
    SETTLED_QUESTION,
    Scenario,
    facts_for,
    make_scenario,
)

# --- wording -----------------------------------------------------------------

Wordings = Mapping[str, tuple[str, bool]]
"""Question name -> (question text, reversed); a reversed question means the
opposite of the original, so its score is ``1 - P(yes)``."""

SETTLED_WORDINGS: Wordings = {
    "settled": (SETTLED_QUESTION, False),
    "settled_2": (
        "Has the outcome already been determined, even if it is not stated here?",
        False,
    ),
    "settled_3": (
        "Could someone find out the answer for certain right now, for example by checking records?",
        False,
    ),
    "settled_4": (
        "Is the answer a fact that already exists, rather than something still to be decided?",
        False,
    ),
    "settled_rev": ("Is the answer still open, to be decided in the future?", True),
}
"""Paraphrases of the ``settled`` question (is the answer already fixed?)."""

KNOWN_WORDINGS: Wordings = {
    "known": (KNOWN_QUESTION, False),
    "known_2": ("Is this a question you can answer from your own knowledge?", False),
    "known_3": (
        "Have you seen reliable information about the subject of this question?",
        False,
    ),
    "known_rev": ("Would you need to guess to answer this question?", True),
    "exists": ("Does the subject of this question exist?", False),
}
"""Paraphrases of the ``known`` question; ``exists`` is a related control."""

ENOUGH_WORDINGS: Wordings = {
    "enough": (HOTPOT_ENOUGH_QUESTION, False),
    "enough_2": ("Can the question be answered from the paragraphs alone?", False),
    "enough_3": (
        "Do the paragraphs mention everything needed to answer the question?",
        False,
    ),
    "enough_rev": (
        "Is any information missing from the paragraphs that is needed to answer the question?",
        True,
    ),
}
"""Paraphrases of the ``enough`` question (do the paragraphs suffice?)."""

ADVERSARIAL_CELLS = ("ADVrec", "ADVannounce", "ADVcouncil", "ADVweather")
"""Cells whose wording invites the wrong ``settled`` answer."""

SETTLED_POSITIVE_CELLS = ("D0", "D3", "SP", "ADVrec", "ADVannounce")
"""Scenario cells whose answer is already fixed."""

SETTLED_NEGATIVE_CELLS = ("SFp", "US", "ADVcouncil", "ADVweather")
"""Scenario cells whose answer is still open."""


def adversarial_facts(sc: Scenario, cell: str) -> list[str]:
    """Facts for the adversarial ``settled`` cells.

    ``ADVrec`` (settled by a past random draw) and ``ADVannounce`` (chosen,
    not yet announced) are fixed although they sound open; ``ADVcouncil``
    and ``ADVweather`` are open although they name a decision-maker or a
    rule.

    Args:
        sc: Synthetic scenario.
        cell: One of :data:`ADVERSARIAL_CELLS`.

    Returns:
        Fact sentences (the scenario's fillers first).

    Raises:
        ValueError: For another cell.
    """
    thing = sc.thing[0].upper() + sc.thing[1:]
    facts = list(sc.fillers)
    o1, o2 = sc.options[0], sc.options[1]
    if cell == "ADVrec":
        facts.append(
            f"Chance alone decided {sc.thing}: a random draw was held last spring, "
            "and its result was written in the records, which are not quoted here."
        )
    elif cell == "ADVannounce":
        facts.append(
            f"{thing} will be announced tomorrow; the council chose it last week."
        )
    elif cell == "ADVcouncil":
        facts.append(
            f"{thing} has not been chosen yet. The council will choose it at next week's meeting."
        )
    elif cell == "ADVweather":
        facts.append(
            f"{thing} has not been fixed yet. It will be {o1} if it rains on the day, "
            f"and {o2} otherwise."
        )
    else:
        raise ValueError(cell)
    return facts


def question_texts(wordings: Wordings) -> dict[str, str]:
    """Yes/no question texts of a wording family.

    Args:
        wordings: Name -> (text, reversed).

    Returns:
        Name -> text.
    """
    return {k: v[0] for k, v in wordings.items()}


def popularity_sample(
    entity_items: Sequence[Item], per_quintile: int, seed: int = 0
) -> list[tuple[Item, int]]:
    """Sample real PopQA items evenly across popularity quintiles.

    Quintiles are cut on the rank of subject popularity (``s_pop``; ties by
    order); ``per_quintile`` items are drawn from each with pandas'
    ``sample(random_state=seed)``.

    Args:
        entity_items: PopQA items (info ``set="popqa"``), optionally followed
            by fabricated twins, which are ignored.
        per_quintile: Items per quintile.
        seed: Sampling seed.

    Returns:
        ``(item, quintile)`` pairs, quintile 0 (least popular) first.
    """
    real = [it for it in entity_items if it.info["set"] == "popqa"]
    frame = pd.DataFrame(
        {"i": range(len(real)), "s_pop": [it.info["s_pop"] for it in real]}
    )
    frame["q"] = pd.qcut(frame["s_pop"].rank(method="first"), 5, labels=False)
    picked = frame.groupby("q").sample(per_quintile, random_state=seed)
    return [(real[i], int(q)) for i, q in zip(picked["i"], picked["q"], strict=True)]


def wording_items(
    entity_items: Sequence[Item],
    hotpot: Sequence[Item],
    *,
    n_scenarios: int = 120,
    per_quintile: int = 80,
    fabricated_per_relation: int = 25,
    paragraph_questions: int = 300,
    seed: int = 0,
) -> list[Item]:
    """Items asking every wording of ``settled``, ``known`` and ``enough``.

    Args:
        entity_items: PopQA items followed by their fabricated twins (as
            the knowledge-boundary experiment builds them).
        hotpot: HotpotQA items (all four evidence cells).
        n_scenarios: Synthetic scenarios (each in nine cells).
        per_quintile: Real PopQA items per popularity quintile.
        fabricated_per_relation: Fabricated items kept per relation (those
            whose trailing index is below this).
        paragraph_questions: HotpotQA questions sampled (all cells of each).
        seed: Scenario seed and sampling seed.

    Returns:
        Items with yes/no questions only; info ``part`` is ``settled``,
        ``known`` or ``enough``.
    """
    items = []
    for i in range(n_scenarios):
        sc = make_scenario(i, seed)
        for cell in (*SETTLED_POSITIVE_CELLS, *SETTLED_NEGATIVE_CELLS):
            facts = (
                adversarial_facts(sc, cell)
                if cell in ADVERSARIAL_CELLS
                else facts_for(sc, cell)
            )
            items.append(
                Item(
                    unit=f"wording:settled:{i}:{cell}",
                    state={"facts": facts, "question": f"What is {sc.thing}?"},
                    nouls=question_texts(SETTLED_WORDINGS),
                    info={
                        "part": "settled",
                        "scenario": i,
                        "cell": cell,
                        "positive": cell in SETTLED_POSITIVE_CELLS,
                    },
                )
            )
    fabricated = [
        it
        for it in entity_items
        if it.info["set"] == "fabricated"
        and int(it.unit.rsplit(":", 1)[-1]) < fabricated_per_relation
    ]
    picked = [it for it, _ in popularity_sample(entity_items, per_quintile, seed)]
    for it in [*picked, *fabricated]:
        items.append(  # noqa: PERF401
            Item(
                unit=f"wording:known:{it.unit}",
                state=it.state,
                nouls=question_texts(KNOWN_WORDINGS),
                info={
                    "part": "known",
                    "fabricated": bool(it.info["fabricated"]),
                    "qid": it.unit,
                },
            )
        )
    qids = sorted({it.info["qid"] for it in hotpot})
    chosen = set(random.Random(seed).sample(qids, paragraph_questions))  # noqa: S311  # nosec B311
    for it in hotpot:
        if it.info["qid"] in chosen:
            items.append(  # noqa: PERF401
                Item(
                    unit=f"wording:enough:{it.unit}",
                    state=it.state,
                    nouls=question_texts(ENOUGH_WORDINGS),
                    info={
                        "part": "enough",
                        "qid": it.info["qid"],
                        "cell": it.info["cell"],
                    },
                )
            )
    return items


# --- odds formats --------------------------------------------------------------

PERCENT_LEVELS = tuple(f"{10 * i}–{10 * (i + 1)} %" for i in range(10))  # noqa: RUF001
"""Ten decile bands for a score question (the API allows at most ten levels).
A reading is the band midpoint, ``(expected level + 0.5) / 10``."""


def percent_reading(score: float) -> float:
    """Convert an expected score level on :data:`PERCENT_LEVELS` to a probability.

    Args:
        score: Expected level (0-9).

    Returns:
        Probability in [0.05, 0.95].
    """
    return (score + 0.5) / 10


INSTRUCTED_ODDS = (
    "Give each option the probability that the random draw will produce it."
)
"""Choice instructions asking for the draw's probabilities."""

ODDS_FORMATS = ("noul", "score", "instructed")
"""Formats: per-option yes/no, per-option score, instructed choice."""


def odds_format_items(n_scenarios: int, seed: int = 0) -> list[Item]:
    """Stated-chance cells asked in three formats.

    Args:
        n_scenarios: Synthetic scenarios.
        seed: Scenario seed.

    Returns:
        Three items per scenario (info ``format`` in :data:`ODDS_FORMATS`).
    """
    items = []
    for sc in (make_scenario(i, seed) for i in range(n_scenarios)):
        state = {"facts": facts_for(sc, "SFp"), "question": f"What is {sc.thing}?"}
        info = {"scenario": sc.index, "profile": sc.profile}
        items.append(
            Item(
                unit=f"odds:noul:{sc.index}",
                state=state,
                nouls={
                    f"opt{k}": f"Will the draw give {o}?"
                    for k, o in enumerate(sc.options)
                },
                info={**info, "format": "noul"},
            )
        )
        items.append(
            Item(
                unit=f"odds:score:{sc.index}",
                state=state,
                scores={
                    f"opt{k}": (
                        f"What is the chance that the draw gives {o}?",
                        PERCENT_LEVELS,
                    )
                    for k, o in enumerate(sc.options)
                },
                info={**info, "format": "score"},
            )
        )
        items.append(
            Item(
                unit=f"odds:instructed:{sc.index}",
                state=state,
                options=sc.options,
                instructions=INSTRUCTED_ODDS,
                info={**info, "format": "instructed"},
            )
        )
    return items


def odds_readings(row: Mapping[str, Any], sc: Scenario) -> list[float]:
    """Per-option probability readings of an odds-format row.

    Args:
        row: Scored row of an :func:`odds_format_items` item.
        sc: Its scenario.

    Returns:
        One reading per option, in display order (not normalised).
    """
    n = len(sc.options)
    if row["format"] == "noul":
        return [row[f"p_opt{k}"] for k in range(n)]
    if row["format"] == "score":
        return [percent_reading(row[f"s_opt{k}"]) for k in range(n)]
    return [row["dist"][o] for o in sc.options]


# --- cutoff ------------------------------------------------------------------

CUTOFF_MONTH = "2024-11"
"""Estimated knowledge cutoff: the change point in monthly news accuracy
found by the knowledge-boundary experiment (first month after it)."""

TODAY = "2026-09-28"
"""Date given as ``today`` in the dated cutoff conditions."""

UNKNOWN_DESCRIPTIONS = {
    "yes": "yes",
    "no": "no",
    "unknown": "not known to me: I have no information about how this turned out",
}
"""Option descriptions when an ``unknown`` option is offered."""

CUTOFF_CONDITIONS = ("base", "date", "unknown", "date_unknown")
"""Conditions of the cutoff probe."""


def cutoff_sample(
    news: Sequence[Item], cutoff_month: str = CUTOFF_MONTH, seed: int = 0
) -> list[Item]:
    """Every news item after the cutoff, then as many sampled from before it.

    Args:
        news: News items (info ``month``).
        cutoff_month: First month counted as after the cutoff.
        seed: Seed of ``random.Random(seed).sample`` for the earlier items.

    Returns:
        Post-cutoff items (in input order), then the sampled earlier items.
    """
    post = [it for it in news if it.info["month"] >= cutoff_month]
    pre = [it for it in news if it.info["month"] < cutoff_month]
    return [*post, *random.Random(seed).sample(pre, len(post))]  # noqa: S311  # nosec B311


def cutoff_items(
    news: Sequence[Item],
    *,
    cutoff_month: str = CUTOFF_MONTH,
    today: str = TODAY,
    seed: int = 0,
) -> list[Item]:
    """Yes/no news items after the cutoff and as many before, in four conditions.

    ``base`` is the item unchanged (same request); ``date`` adds ``today``
    to the state; ``unknown`` adds a third option; ``date_unknown`` both.

    Args:
        news: Yes/no news items (info ``month``).
        cutoff_month: First month counted as after the cutoff.
        today: Date given in the dated conditions.
        seed: Seed for the before-cutoff sample.

    Returns:
        Four items per sampled question (units ``cutoff:<cond>:<unit>``).
    """
    items = []
    for it in cutoff_sample(news, cutoff_month, seed):
        info = {**it.info, "post": it.info["month"] >= cutoff_month}
        dated = {**it.state, "today": today}
        items.append(
            dataclasses.replace(
                it, unit=f"cutoff:base:{it.unit}", info={**info, "cond": "base"}
            )
        )
        items.append(
            dataclasses.replace(
                it,
                unit=f"cutoff:date:{it.unit}",
                state=dated,
                info={**info, "cond": "date"},
            )
        )
        for cond, state in (("unknown", it.state), ("date_unknown", dated)):
            items.append(
                dataclasses.replace(
                    it,
                    unit=f"cutoff:{cond}:{it.unit}",
                    state=state,
                    options=("yes", "no", "unknown"),
                    descriptions=UNKNOWN_DESCRIPTIONS,
                    info={**info, "cond": cond},
                )
            )
    return items


# --- warned paragraphs ------------------------------------------------------------

PARAGRAPH_WARNING = (
    "Answer the question. The paragraphs may be irrelevant or incomplete. Use them "
    "only if they actually answer the question; otherwise rely on what you know."
)
"""Choice instructions that warn the paragraphs may not help."""


def warned_paragraph_items(hotpot: Sequence[Item]) -> list[Item]:
    """HotpotQA items with paragraphs, re-asked with :data:`PARAGRAPH_WARNING`.

    Args:
        hotpot: HotpotQA items.

    Returns:
        The items with paragraphs (closed-book cell dropped).
    """
    return [
        dataclasses.replace(
            it, unit=f"warned:{it.unit}", instructions=PARAGRAPH_WARNING
        )
        for it in hotpot
        if it.info["cell"] != "closed"
    ]


# --- second look ------------------------------------------------------------------

CORRECT_QUESTION = "Is the proposed answer the correct answer to the question?"
"""Yes/no question about a shown first answer."""

CHANCE_QUESTION = "How likely is it that the proposed answer is correct?"
"""Score question (on :data:`PERCENT_LEVELS`) about a shown first answer."""


def second_look_item(
    item: Item, first: Mapping[str, Any], unit: str, info: Mapping[str, Any]
) -> Item:
    """Show an item's first answer back and ask whether it is correct.

    Args:
        item: The original item (its state is reused).
        first: Its scored row: ``top``, ``p_max`` and (optionally)
            ``correct``.
        unit: Unit of the new item.
        info: Extra metadata.

    Returns:
        An item with the ``correct`` yes/no and ``chance`` score questions;
        info adds ``first_p_max``, ``first_correct`` and ``proposed``.
    """
    top = first["top"]
    text = item.descriptions[top] if item.descriptions else top
    return Item(
        unit=unit,
        state={**item.state, "proposed_answer": text},
        nouls={"correct": CORRECT_QUESTION},
        scores={"chance": (CHANCE_QUESTION, PERCENT_LEVELS)},
        info={
            **info,
            "first_p_max": first["p_max"],
            "first_correct": bool(first.get("correct") or False),
            "proposed": top,
        },
    )
