"""Made-up events whose status (past or future) must be inferred from a cue.

Each event comes in a past and a future version that differ only in one
cue: an explicit statement, the tense, or a date (with or without today's
date). The question is who won; the ``settled`` yes/no question asks
whether the answer is already fixed.
"""

import random
from dataclasses import dataclass

from beyond_answer_confidence.tasks import seeds
from beyond_answer_confidence.tasks.schema import Item
from beyond_answer_confidence.tasks.synthetic import SETTLED_QUESTION, pseudo_word

CONDITIONS = ("explicit", "tense", "date_today", "date_only")
"""Cue conditions, in build order."""

STATUSES = ("past", "future")
"""Event versions."""

TODAY = "Today's date is 29 September 2026."
"""Fact giving today's date in the ``date_today`` condition."""

INSTRUCTIONS = "Choose the correct answer to the question."

MONTHS = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)

_SEED_NAMESPACE = seeds.SETTLEDNESS


@dataclass(frozen=True)
class EventKind:
    """How one kind of event is described.

    Attributes:
        name: Kind id.
        event: Noun phrase with ``{e}`` for the event name.
        past: Tense-only sentence ending for the past version.
        future: Tense-only sentence ending for the future version.
        role: Plural noun for the four contenders.
    """

    name: str
    event: str
    past: str
    future: str
    role: str


EVENT_KINDS = (
    EventKind(
        "sports",
        "the {e} Cup final",
        "was played last spring",
        "will be played next spring",
        "teams",
    ),
    EventKind(
        "election",
        "the {e} mayoral election",
        "was held last spring",
        "will be held next spring",
        "candidates",
    ),
    EventKind(
        "award",
        "the {e} Prize for poetry",
        "was awarded by its jury last spring",
        "will be awarded by its jury next spring",
        "shortlisted poets",
    ),
    EventKind(
        "court",
        "the {e} inheritance case",
        "was heard in court last spring",
        "will be heard in court next spring",
        "claimants",
    ),
    EventKind(
        "auction",
        "the {e} estate auction",
        "took place last spring",
        "will take place next spring",
        "bidders",
    ),
    EventKind(
        "lottery",
        "the {e} lottery draw",
        "took place last spring",
        "will take place next spring",
        "ticket holders",
    ),
)
"""The six event kinds, in build order."""


@dataclass(frozen=True)
class Event:
    """One made-up event shared by its past and future versions.

    Attributes:
        kind: Its kind.
        index: Index within the kind.
        name: Pseudo-word name.
        contenders: Four contenders in display order.
        day: Day of month.
        month: Month name.
        past_year: Year of the past version.
        future_year: Year of the future version.
    """

    kind: EventKind
    index: int
    name: str
    contenders: tuple[str, str, str, str]
    day: int
    month: str
    past_year: int
    future_year: int

    @property
    def phrase(self) -> str:
        """The event's noun phrase."""
        return self.kind.event.format(e=self.name)


def make_event(kind: EventKind, index: int, seed: int = 0) -> Event:
    """Generate one event deterministically.

    Args:
        kind: Event kind.
        index: Index within the kind.
        seed: Global seed.

    Returns:
        The event.
    """
    rng = random.Random(f"{_SEED_NAMESPACE}:{seed}:{kind.name}:{index}")  # noqa: S311  # nosec B311
    names: set[str] = set()
    while len(names) < 4:
        names.add(pseudo_word(rng, 2))
    contenders = sorted(names)
    rng.shuffle(contenders)
    return Event(
        kind=kind,
        index=index,
        name=pseudo_word(rng, 2),
        contenders=(contenders[0], contenders[1], contenders[2], contenders[3]),
        day=rng.randint(1, 28),
        month=rng.choice(MONTHS),
        past_year=rng.randint(1985, 2015),
        future_year=rng.randint(2031, 2045),
    )


def _cap(s: str) -> str:
    return s[0].upper() + s[1:]


def event_facts(ev: Event, condition: str, status: str) -> list[str]:
    """Facts shown for one condition and version.

    Args:
        ev: Event.
        condition: One of :data:`CONDITIONS`.
        status: ``"past"`` or ``"future"``.

    Returns:
        Fact sentences.
    """
    a, b, c, d = ev.contenders
    who = f"The four {ev.kind.role} are {a}, {b}, {c} and {d}."
    past = status == "past"
    if condition == "explicit":
        state = "has already been decided" if past else "has not been decided yet"
        return [f"The winner of {ev.phrase} {state}.", who]
    if condition == "tense":
        ending = ev.kind.past if past else ev.kind.future
        return [f"{_cap(ev.phrase)} {ending}.", who]
    year = ev.past_year if past else ev.future_year
    entry = f"{_cap(ev.phrase)}: {ev.day} {ev.month} {year}."
    return [TODAY, entry, who] if condition == "date_today" else [entry, who]


def event_item(ev: Event, condition: str, status: str) -> Item:
    """The item for one event, condition and version.

    Args:
        ev: Event.
        condition: One of :data:`CONDITIONS`.
        status: ``"past"`` or ``"future"``.

    Returns:
        A choice over the contenders plus the ``settled`` question.
    """
    return Item(
        unit=f"{condition}:{ev.kind.name}:{ev.index}:{status}",
        state={
            "question": f"Who is the winner of {ev.phrase}?",
            "facts": event_facts(ev, condition, status),
        },
        options=ev.contenders,
        instructions=INSTRUCTIONS,
        nouls={"settled": SETTLED_QUESTION},
        info={
            "cond": condition,
            "status": status,
            "domain": ev.kind.name,
            "event": f"{ev.kind.name}:{ev.index}",
            "first_listed": ev.contenders[0],
        },
    )


def settledness_items(events_per_kind: int = 60, seed: int = 0) -> list[Item]:
    """Every item (event-major; each condition and version of an event).

    Args:
        events_per_kind: Events per kind.
        seed: Global seed.

    Returns:
        Items.
    """
    items: list[Item] = []
    for kind in EVENT_KINDS:
        for i in range(events_per_kind):
            ev = make_event(kind, i, seed)
            items.extend(event_item(ev, c, s) for c in CONDITIONS for s in STATUSES)
    return items
