"""Surface-cue controls for the follow-up readouts.

High AUROCs for ``known`` could come from cues in the input (dates in a
news question, an unfamiliar-looking name) rather than from what the model
knows. The builders here hold one cue fixed at a time:

- **masked**: yes/no news questions with every date phrase removed
  (:func:`mask_dates`);
- **lookalike**: made-up people whose names recombine real first names and
  surnames of obscure PopQA people, so the name looks real;
- **evidence**: made-up subjects with a sentence stating an answer, or an
  unrelated sentence of the same form;
- **polarity**: the second-look question with the *opposite* of the model's
  own answer proposed;
- **semantics**: random devices asked four ways (which outcome will occur,
  which is most likely, the probability of each, a bet);
- **negated**: the second-look question worded negatively ("is the
  proposed answer wrong?"), with both answers proposed;
- **near_day**: the unchanged news question with today's date set just
  after the event it asks about.

Every prompt string and seed string here is part of the request payload or
decides it; changing one changes the cache keys.
"""

import calendar
import random
import re
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from beyond_answer_confidence.tasks import seeds
from beyond_answer_confidence.tasks.chance import DEVICES, INSTRUCTIONS
from beyond_answer_confidence.tasks.entities import primary_templates
from beyond_answer_confidence.tasks.follow_up import (
    CHANCE_QUESTION,
    CORRECT_QUESTION,
    CUTOFF_MONTH,
    PERCENT_LEVELS,
    cutoff_sample,
)
from beyond_answer_confidence.tasks.multiple_choice import (
    KNOWN_QUESTION,
    LETTERS,
    QA_INSTRUCTIONS,
)
from beyond_answer_confidence.tasks.schema import Item

_SEED_NAMESPACE = seeds.SHORTCUT_CONTROLS

PERSON_RELATION_ORDER = (
    "occupation",
    "place of birth",
    "father",
    "mother",
    "religion",
    "sport",
)
"""PopQA person relations in the order the look-alike names are drawn."""

STUDY_DATE = "2026-09-30"
"""Latest ``today`` used for the near-day items (every label had resolved)."""

WRONG_QUESTION = "Is the proposed answer wrong?"
"""Negative counterpart of :data:`~beyond_answer_confidence.tasks.follow_up.CORRECT_QUESTION`."""

SEMANTIC_DEVICES = (
    "die_words",
    "die_loaded",
    "two_dice_sum",
    "coin_heads",
    "card_suit",
)
"""Devices asked in the four :data:`SEMANTIC_WORDINGS`."""

SEMANTIC_WORDINGS: dict[str, dict[str, str]] = {
    "die_words": {
        "occur": DEVICES["die_words"][0],
        "most_likely": "A fair six-sided die is rolled once. Which number is most likely to come up?",
        "probability": "A fair six-sided die is rolled once. Give the probability that each number comes up.",
        "bet": "A fair six-sided die is rolled once. You win a prize if you pick the number that comes up. Which number do you pick?",
    },
    "die_loaded": {
        "occur": DEVICES["die_loaded"][0],
        "most_likely": "A die is loaded so that it shows six half of the time, and each other number one time in ten. It is rolled once. Which number is most likely to come up?",
        "probability": "A die is loaded so that it shows six half of the time, and each other number one time in ten. It is rolled once. Give the probability that each number comes up.",
        "bet": "A die is loaded so that it shows six half of the time, and each other number one time in ten. It is rolled once. You win a prize if you pick the number that comes up. Which number do you pick?",
    },
    "two_dice_sum": {
        "occur": DEVICES["two_dice_sum"][0],
        "most_likely": "Two fair six-sided dice are rolled. Which sum is most likely?",
        "probability": "Two fair six-sided dice are rolled. Give the probability of each sum.",
        "bet": "Two fair six-sided dice are rolled. You win a prize if you pick their sum. Which sum do you pick?",
    },
    "coin_heads": {
        "occur": DEVICES["coin_heads"][0],
        "most_likely": "A fair coin is flipped three times. Which number of heads is most likely?",
        "probability": "A fair coin is flipped three times. Give the probability of each number of heads.",
        "bet": "A fair coin is flipped three times. You win a prize if you pick the number of heads that comes up. Which number do you pick?",
    },
    "card_suit": {
        "occur": DEVICES["card_suit"][0],
        "most_likely": "A card is drawn from a well-shuffled standard 52-card deck. Which suit is most likely?",
        "probability": "A card is drawn from a well-shuffled standard 52-card deck. Give the probability of each suit.",
        "bet": "A card is drawn from a well-shuffled standard 52-card deck. You win a prize if you pick its suit. Which suit do you pick?",
    },
}
"""Device -> wording -> question; ``occur`` is the stated-odds wording verbatim."""

FACT_TEMPLATES = {
    "author": "{s} was written by {o}.",
    "capital": "The capital of {s} is {o}.",
    "capital of": "{s} is the capital of {o}.",
    "color": "{s} is {o}.",
    "composer": "The music of {s} was composed by {o}.",
    "country": "{s} is in {o}.",
    "director": "{s} was directed by {o}.",
    "father": "The father of {s} is {o}.",
    "genre": "{s} belongs to the {o} genre.",
    "mother": "The mother of {s} is {o}.",
    "occupation": "{s}'s occupation is {o}.",
    "place of birth": "{s} was born in {o}.",
    "producer": "{s} was produced by {o}.",
    "religion": "The religion of {s} is {o}.",
    "screenwriter": "The screenwriter for {s} was {o}.",
    "sport": "{s} plays {o}.",
}
"""Relation -> sentence stating a subject's object."""

NAME_STOPWORDS = frozenset(
    {
        "League",
        "SK",
        "FC",
        "Club",
        "Team",
        "Education",
        "Missionaries",
        "Lutheran",
        "Home",
        "Dark",
        "Came",
        "Dance",
        "Princess",
        "Prince",
        "Archduchess",
        "Archduke",
        "King",
        "Queen",
        "Saint",
        "Sir",
        "Lady",
        "Statistics",
        "Ecclesiastical",
        "Living",
        "Church",
        "University",
        "Now",
        "The",
        "New",
        "Old",
        "Great",
        "Little",
        "Big",
        "Black",
        "White",
    }
)
"""Tokens that mark a PopQA subject as something other than a plain name."""

# --- date phrases --------------------------------------------------------------

MONTH_NAMES = (
    "January|February|March|April|May|June|July|August|September|October|"
    "November|December"
)
SEASONS = "spring|summer|fall|autumn|winter"
ABBREV = r"(?:Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sept?|Oct|Nov|Dec)\."
ORDINAL = "(?:st|n[d]|rd|th)"
"""Ordinal suffixes (a character class keeps the spell checker quiet)."""
WEEKDAYS = "Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday"
_QUAL = (
    r"(?i:early|mid|late|mid-|the end of|the first week of|the second week of|"
    r"the third week of|the last week of|the week of|the (?:trading )?week ending|"
    r"the start of|the beginning of|the first half of|the second half of)"
)
_PREP = r"(?i:by|in|on|during|throughout|before|after|until|since|as of|from|through|within|at)"
_DAY = rf"\d{{1,2}}{ORDINAL}?"
_CORE = (
    # Month [day,] year / season year, optionally led by a weekday and a qualifier
    rf"(?:(?:{WEEKDAYS}),?\s+)?(?:{_QUAL}[- ]?\s*)?"
    rf"(?:(?:{MONTH_NAMES}|{ABBREV}|(?i:{SEASONS}))(?:\s+{_DAY})?,?\s+)?(?:19|20)\d\d\b"
    # Month day [to day] (no year)
    rf"|(?:(?:{WEEKDAYS}),?\s+)?(?:{_QUAL}\s+)?(?:{MONTH_NAMES}|{ABBREV})\s*{_DAY}"
    rf"(?:\s*(?:to|-|\u2013)\s*{_DAY})?\b"
    # day Month [year]
    rf"|\b{_DAY}\s+(?:{MONTH_NAMES})(?:,?\s+(?:19|20)\d\d)?\b"
    # a bare month (capitalised), optionally qualified; "March of Dimes" is a name
    rf"|(?:{_QUAL}[- ]?\s*)?\b(?:{MONTH_NAMES})\b(?! of Dimes)"
    rf"|(?<!\w){ABBREV}"
    rf"|\b(?:{WEEKDAYS})\b"
)
DATE_PATTERN = re.compile(rf"(?:\b{_PREP}\s+)?(?:{_CORE})")
"""A date phrase with its leading preposition, so removing it keeps the
sentence grammatical ("report in March 2024 that" -> "report that"). Month
names match only when capitalised, so the verb "may" is untouched."""
ORPHAN_SPAN = re.compile(
    r"\b(?:through|by|before|until|at|in|on|during)?\s*the\s+"
    r"(?:end|start|beginning|middle|first half|second half|first week|last week)"
    r"\s+of\b(?=\s*(?:[?,.]|due|because|after|before|in|on|by|as|when)\b|\s*[?,.])",
    re.IGNORECASE,
)
"""A span such as "by the end of" left behind once its date is removed."""
DANGLING = re.compile(
    r"\b(?:by|in|on|during|throughout|before|after|until|of|from|between|since)"
    r"\s*(?=[?,.]|\s+(?:after|before|following|when|as|amid|and|or|in|on|by|of|due|because|that|to|be|with|compared)\b)",
    re.IGNORECASE,
)
"""A preposition left without its object once a date is removed."""
YEAR_MONTH = re.compile(rf"(?:({MONTH_NAMES})\s+(?:\d{{1,2}},?\s+)?)?((?:19|20)\d\d)")
"""A year, optionally preceded by a month name (and day)."""
MENTION = re.compile(
    rf"(?:\b({MONTH_NAMES})\s+(?:\d{{1,2}},?\s+)?)?\b((?:19|20)\d\d)\b"
)
"""Like :data:`YEAR_MONTH`, with word boundaries (used by :func:`settled_by`)."""
MONTH_INDEX = {m.lower(): i + 1 for i, m in enumerate(MONTH_NAMES.split("|"))}
"""Lower-case month name -> month number."""


def mask_dates(text: str) -> str:
    """Remove explicit dates and time references from a question.

    Args:
        text: Question.

    Returns:
        The question with date phrases removed and spacing tidied.
    """
    out = DATE_PATTERN.sub("", text)
    for _ in range(2):
        out = re.sub(r"\s{2,}", " ", out)
        out = ORPHAN_SPAN.sub("", out)
        out = DANGLING.sub("", out)
    out = re.sub(r"\s+([?,.])", r"\1", out)
    out = re.sub(r",\s*([?.,])", r"\1", out)
    out = re.sub(r"^[\s,]+", "", out)
    out = re.sub(r"\s{2,}", " ", out).strip()
    return out[:1].upper() + out[1:]


def latest_date(text: str) -> float:
    """The latest year (plus month fraction) mentioned in a question.

    Args:
        text: Question.

    Returns:
        ``year + (month - 1) / 12``, or ``nan`` if no year is mentioned.
    """
    best = float("nan")
    for month, year in YEAR_MONTH.findall(text):
        v = int(year) + (MONTH_INDEX.get(month.lower(), 1) - 1) / 12
        best = v if np.isnan(best) else max(best, v)
    return best


def month_after_end(month: str) -> str:
    """Last day of the month after ``month`` (``YYYY-MM``), as ``YYYY-MM-DD``.

    Args:
        month: A month.

    Returns:
        The date.
    """
    y, m = (int(x) for x in month.split("-"))
    y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return f"{y:04d}-{m:02d}-{calendar.monthrange(y, m)[1]:02d}"


def settled_by(question: str, month: str) -> str:
    """The latest month a question refers to (``YYYY-MM``), never before ``month``.

    A month with a year counts as that month; a bare year (including seasons
    such as "spring 2020") counts as December of that year.

    Args:
        question: Question text.
        month: The question's own month.

    Returns:
        ``YYYY-MM``.
    """
    best = month
    for name, year in MENTION.findall(question):
        m = MONTH_INDEX[name.lower()] if name else 12
        best = max(best, f"{int(year):04d}-{m:02d}")
    return best


# --- names and sentences ----------------------------------------------------------


def is_person_name(name: str) -> bool:
    """Whether a PopQA subject looks like a plain personal name.

    Two or three capitalised alphabetic tokens (hyphens allowed), no particles
    such as "of", no organisation or title words.

    Args:
        name: Subject string.

    Returns:
        ``True`` for names like "Tomas Brenner".
    """
    toks = name.split()
    return (
        2 <= len(toks) <= 3
        and all(t[0].isupper() and t.replace("-", "").isalpha() for t in toks)
        and not NAME_STOPWORDS & set(toks)
    )


def lookalike_subjects(
    popqa: pd.DataFrame, per_relation: int, rng: random.Random
) -> dict[str, list[tuple[str, str, str]]]:
    """Recombined names from obscure real people (least popular fifth).

    Donors are pooled over the person relations; each made-up name joins the
    first name of one donor to the surname(s) of another and is kept only if
    it is not a PopQA subject and has not been drawn before.

    Args:
        popqa: PopQA table (``subj``, ``prop``, ``s_pop``).
        per_relation: Names per relation.
        rng: Random generator.

    Returns:
        Relation -> ``(name, first-name donor, surname donor)`` triples.
    """
    existing = set(popqa["subj"].astype(str))
    donors = sorted(
        {
            s
            for prop in PERSON_RELATION_ORDER
            for s in popqa[
                (popqa["prop"] == prop)
                & (
                    popqa["s_pop"]
                    <= popqa[popqa["prop"] == prop]["s_pop"].quantile(0.2)
                )
            ]["subj"].astype(str)
            if is_person_name(s)
        }
    )
    out: dict[str, list[tuple[str, str, str]]] = {}
    seen: set[str] = set()
    for prop in PERSON_RELATION_ORDER:
        picks: list[tuple[str, str, str]] = []
        while len(picks) < per_relation:
            a, b = rng.sample(donors, 2)
            name = f"{a.split()[0]} {' '.join(b.split()[1:])}"
            if name in existing or name in seen or a.split()[0] == b.split()[0]:
                continue
            seen.add(name)
            picks.append((name, a, b))
        out[prop] = picks
    return out


def fact_sentence(prop: str, subject: str, obj: str) -> str:
    """A sentence stating a subject's object, capitalised.

    Args:
        prop: Relation (a key of :data:`FACT_TEMPLATES`).
        subject: Subject.
        obj: Object.

    Returns:
        The sentence.
    """
    text = FACT_TEMPLATES[prop].format(s=subject, o=obj)
    return text[0].upper() + text[1:]


# --- item builders -----------------------------------------------------------------


def news_subset(
    news: Sequence[Item], cutoff_month: str = CUTOFF_MONTH, seed: int = 0
) -> list[Item]:
    """The news items shared by every news control, in their original order.

    All items from ``cutoff_month`` on, plus an equal seeded sample from
    before (the second-look news subset).

    Args:
        news: Yes/no news items (info ``month``).
        cutoff_month: First month counted as after the cutoff.
        seed: Seed of the earlier sample.

    Returns:
        The chosen items, in the order of ``news``.
    """
    chosen = {it.unit for it in cutoff_sample(news, cutoff_month, seed)}
    return [it for it in news if it.unit in chosen]


def masked_items(subset: Sequence[Item]) -> list[Item]:
    """Date-masked versions of news items.

    Args:
        subset: News items (see :func:`news_subset`).

    Returns:
        Items (units ``masked:<unit>``; info adds ``part`` and ``base``).
    """
    return [
        Item(
            unit=f"masked:{it.unit}",
            state={"question": mask_dates(str(it.state["question"]))},
            options=it.options,
            gold=it.gold,
            instructions=it.instructions,
            nouls=dict(it.nouls),
            info={**it.info, "part": "masked", "base": it.unit},
        )
        for it in subset
    ]


def lookalike_items(popqa: pd.DataFrame, per_relation: int = 100) -> list[Item]:
    """Made-up people with realistic recombined names, as fabricated twins.

    Each uses its relation's most common question template and four objects
    drawn from the relation's pool (none correct).

    Args:
        popqa: PopQA table.
        per_relation: Names per person relation.

    Returns:
        Items (units ``lookalike:<relation>:<j>``, no gold).
    """
    templates = primary_templates(popqa)
    pools = {p: sorted(set(g["obj"])) for p, g in popqa.groupby("prop")}
    rng = random.Random(f"{_SEED_NAMESPACE}:lookalike")  # noqa: S311  # nosec B311
    items = []
    for prop, picks in lookalike_subjects(popqa, per_relation, rng).items():
        for j, (name, _first, _last) in enumerate(picks):
            r = random.Random(f"{_SEED_NAMESPACE}:lookalike:{prop}:{j}")  # noqa: S311  # nosec B311
            items.append(
                Item(
                    unit=f"lookalike:{prop}:{j}",
                    state={"question": templates[prop].format(name)},
                    options=LETTERS[:4],
                    gold=None,
                    instructions=QA_INSTRUCTIONS,
                    descriptions=dict(
                        zip(LETTERS[:4], r.sample(pools[prop], 4), strict=True)
                    ),
                    nouls={"known": KNOWN_QUESTION},
                    info={"part": "lookalike", "prop": prop, "subject": name},
                )
            )
    return items


def evidence_items(fabricated: Sequence[Item], per_relation: int = 25) -> list[Item]:
    """Fabricated items with a sentence stating an answer, or an unrelated one.

    ``per_relation`` fabricated items are sampled per relation. The stated
    arm adds a sentence naming a random option as the answer (that option
    becomes the gold); the unrelated arm adds a sentence of the same form
    about another sampled item.

    Args:
        fabricated: Fabricated twins (info ``prop``, ``subject``).
        per_relation: Items sampled per relation.

    Returns:
        Two items per sampled item (units ``evidence:<unit>:<arm>``).

    Raises:
        ValueError: If a fabricated item has no option descriptions.
    """
    by_prop: dict[str, list[Item]] = {}
    for it in fabricated:
        by_prop.setdefault(str(it.info["prop"]), []).append(it)
    rng = random.Random(f"{_SEED_NAMESPACE}:evidence")  # noqa: S311  # nosec B311
    chosen = [
        it for p in sorted(by_prop) for it in rng.sample(by_prop[p], per_relation)
    ]
    if any(it.descriptions is None for it in chosen):
        raise ValueError("fabricated items need option descriptions")
    items = []
    for k, it in enumerate(chosen):
        desc = dict(it.descriptions or {})
        letter = rng.choice(it.options)
        prop = str(it.info["prop"])
        stated = fact_sentence(prop, str(it.info["subject"]), str(desc[letter]))
        other = chosen[(k + len(chosen) // 2) % len(chosen)]
        other_desc = other.descriptions or {}
        unrelated = fact_sentence(
            str(other.info["prop"]),
            str(other.info["subject"]),
            str(other_desc[other.options[0]]),
        )
        for arm, sentence in (("stated", stated), ("unrelated", unrelated)):
            items.append(
                Item(
                    unit=f"evidence:{it.unit}:{arm}",
                    state={"question": it.state["question"], "paragraphs": [sentence]},
                    options=it.options,
                    gold=letter if arm == "stated" else None,
                    instructions=it.instructions,
                    descriptions=desc,
                    nouls={"known": KNOWN_QUESTION},
                    info={
                        "part": "evidence",
                        "arm": arm,
                        "prop": prop,
                        "base": it.unit,
                    },
                )
            )
    return items


def polarity_items(
    subset: Sequence[Item],
    first: Mapping[str, Mapping[str, Any]],
    cutoff_month: str = CUTOFF_MONTH,
) -> list[Item]:
    """The second look on news, with the opposite of the model's answer proposed.

    Args:
        subset: News items (see :func:`news_subset`).
        first: Scored rows of the unchanged news items by unit (``top``).
        cutoff_month: First month counted as after the cutoff.

    Returns:
        Items (units ``polarity:<unit>``) whose requests differ from the
        second look's only in ``proposed_answer``.
    """
    items = []
    for it in subset:
        own = first[it.unit]["top"]
        flip = "no" if own == "yes" else "yes"
        items.append(
            Item(
                unit=f"polarity:{it.unit}",
                state={**it.state, "proposed_answer": flip},
                nouls={"correct": CORRECT_QUESTION},
                scores={"chance": (CHANCE_QUESTION, PERCENT_LEVELS)},
                info={
                    "part": "polarity",
                    "base": it.unit,
                    "post": it.info["month"] >= cutoff_month,
                    "own": own,
                    "proposed": flip,
                    "gold_answer": it.gold,
                },
            )
        )
    return items


def semantics_items(orders: int = 24) -> list[Item]:
    """Random devices in four wordings, with seeded option orders.

    Each order is its own item; give each its own replicate index (the last
    field of the unit) because the cache key ignores option order.

    Args:
        orders: Option orders per device.

    Returns:
        Items (units ``semantics:<device>:<wording>:<order>``).
    """
    items = []
    for dev in SEMANTIC_DEVICES:
        labels = list(DEVICES[dev][1])
        rng = random.Random(f"{_SEED_NAMESPACE}:order:{dev}")  # noqa: S311  # nosec B311
        drawn = []
        for _ in range(orders):
            o = labels[:]
            rng.shuffle(o)
            drawn.append(tuple(o))
        for wording, question in SEMANTIC_WORDINGS[dev].items():
            for k, order in enumerate(drawn):
                items.append(
                    Item(
                        unit=f"semantics:{dev}:{wording}:{k}",
                        state={"question": question},
                        options=order,
                        instructions=INSTRUCTIONS,
                        info={"part": "semantics", "device": dev, "wording": wording},
                    )
                )
    return items


def negated_items(
    subset: Sequence[Item], cutoff_month: str = CUTOFF_MONTH
) -> list[Item]:
    """Both proposed answers for every news item, with the negative question.

    Args:
        subset: News items (see :func:`news_subset`).
        cutoff_month: First month counted as after the cutoff.

    Returns:
        Two items per news item (units ``negated:<unit>:<yes|no>``).
    """
    return [
        Item(
            unit=f"negated:{it.unit}:{proposed}",
            state={**it.state, "proposed_answer": proposed},
            nouls={"wrong": WRONG_QUESTION},
            info={
                "part": "negated",
                "base": it.unit,
                "post": it.info["month"] >= cutoff_month,
                "proposed": proposed,
                "gold_answer": it.gold,
                "month": it.info["month"],
            },
        )
        for it in subset
        for proposed in ("yes", "no")
    ]


def near_day_items(subset: Sequence[Item], study_date: str = STUDY_DATE) -> list[Item]:
    """The unchanged news questions with today's date just after the event.

    ``today`` is the last day of the month after the latest month the
    question mentions (:func:`settled_by`), capped at ``study_date``.

    Args:
        subset: News items (see :func:`news_subset`).
        study_date: Latest date used.

    Returns:
        Items (units ``near_day:<unit>``).
    """
    return [
        Item(
            unit=f"near_day:{it.unit}",
            state={
                **it.state,
                "today": min(
                    study_date,
                    month_after_end(
                        settled_by(str(it.state["question"]), str(it.info["month"]))
                    ),
                ),
            },
            options=it.options,
            gold=it.gold,
            instructions=it.instructions,
            nouls=dict(it.nouls),
            info={**it.info, "part": "near_day", "base": it.unit},
        )
        for it in subset
    ]
