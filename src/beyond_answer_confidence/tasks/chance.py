"""Random devices with known outcome odds, and re-listed synthetic chances.

Each device (dice, coins, cards) is asked once per option order; orders act
as replicates, so position effects average out when all orders are used.
Every order gets its own ``replicate`` index (:func:`order_replicate`),
because the cache key sorts dictionary keys and would otherwise merge
orders.

The re-listed items repeat the tied-chance synthetic cells with the chances
listed in a different order from the options, to separate "shown first"
from "listed first".
"""

import itertools
import random
from collections.abc import Sequence

from beyond_answer_confidence.tasks.schema import Item, Task
from beyond_answer_confidence.tasks.synthetic import (
    CHOICE_INSTRUCTIONS,
    DETERMINED_QUESTION,
    SETTLED_QUESTION,
    Scenario,
    make_scenario,
)

INSTRUCTIONS = "Choose the correct answer to the question."
"""Choice instructions for the device questions."""

WORDS = ("one", "two", "three", "four", "five", "six")
COLOURS = ("red", "green", "blue", "yellow", "white", "black")


def _uniform(opts: Sequence[str]) -> dict[str, float]:
    return {o: 1 / len(opts) for o in opts}


def _two_dice() -> dict[str, float]:
    return {str(s): (6 - abs(s - 7)) / 36 for s in range(2, 13)}


DEVICES: dict[str, tuple[str, dict[str, float], int | None, int]] = {
    "die_words": (
        "What number will come up on a single roll of a fair six-sided die?",
        _uniform(WORDS),
        None,
        1,
    ),
    "die_digits": (
        "What number will come up on a single roll of a fair six-sided die?",
        _uniform(tuple(str(i) for i in range(1, 7))),
        None,
        1,
    ),
    "die_colours": (
        "A fair die has a different colour on each of its six faces. It is rolled once. "
        "Which colour will come up?",
        _uniform(COLOURS),
        None,
        1,
    ),
    "die_loaded": (
        "A die is loaded so that it shows six half of the time, and each other number "
        "one time in ten. It is rolled once. What number will come up?",
        {**dict.fromkeys(WORDS[:5], 0.1), "six": 0.5},
        None,
        1,
    ),
    "two_dice_sum": (
        "Two fair six-sided dice are rolled. What will their sum be?",
        _two_dice(),
        200,
        1,
    ),
    "coin_heads": (
        "A fair coin is flipped three times. How many heads will come up?",
        {"0": 1 / 8, "1": 3 / 8, "2": 3 / 8, "3": 1 / 8},
        None,
        5,
    ),
    "card_suit": (
        "A card is drawn from a well-shuffled standard 52-card deck. What suit will it be?",
        _uniform(("hearts", "diamonds", "clubs", "spades")),
        None,
        5,
    ),
}
"""Name -> (question, ideal distribution, number of seeded orders or ``None``
for all orders, calls per order)."""

RELIST_PROFILES = ("uniform", "binary")
"""Chance profiles with tied top chances (re-listed)."""


def device_items() -> list[tuple[Item, int]]:
    """Items for every device, one per option order.

    Returns:
        ``(item, calls per order)`` pairs; units ``device:<name>:<order>``.
    """
    out = []
    for name, (question, ideal, n_orders, reps) in DEVICES.items():
        labels = list(ideal)
        if n_orders is None:
            orders = list(itertools.permutations(labels))
        else:
            rng = random.Random(f"order:{name}")  # noqa: S311  # nosec B311
            orders = []
            for _ in range(n_orders):
                o = labels[:]
                rng.shuffle(o)
                orders.append(tuple(o))
        for k, order in enumerate(orders):
            item = Item(
                unit=f"device:{name}:{k}",
                state={"question": question},
                options=tuple(order),
                instructions=INSTRUCTIONS,
                info={"set": "device", "device": name, "order": list(order)},
            )
            out.append((item, reps))
    return out


def order_replicate(item: Item, r: int, reps: int) -> int:
    """Replicate index unique to an option order.

    Args:
        item: Device item (unit ends in its order number ``k``).
        r: Call within the order.
        reps: Calls per order.

    Returns:
        ``k * reps + r``.
    """
    k = int(item.unit.rsplit(":", 1)[-1])
    return k * reps + r


def device_tasks(devices: Sequence[tuple[Item, int]]) -> list[Task]:
    """Tasks for device items, each order with its own replicate indices.

    Args:
        devices: Output of :func:`device_items`.

    Returns:
        Item-major tasks.
    """
    return [
        Task(it.unit, it.request(order_replicate(it, r, reps)))
        for it, reps in devices
        for r in range(reps)
    ]


def reordered_chances(sc: Scenario, seed: int = 0) -> tuple[list[str], str]:
    """Stated-chance facts with tied chances listed in a seeded, different order.

    Args:
        sc: A uniform- or binary-profile scenario.
        seed: Seed.

    Returns:
        ``(facts, first_listed_option)``.
    """
    rng = random.Random(f"relist:{seed}:{sc.index}")  # noqa: S311  # nosec B311
    nonzero = [o for o in sc.options if sc.p_star[o] > 0]
    top = max(sc.p_star.values())
    ties = [o for o in nonzero if sc.p_star[o] == top]
    rest = [o for o in nonzero if sc.p_star[o] < top]
    # First listed should differ from first shown whenever the tie allows it.
    candidates = [o for o in ties if o != sc.options[0]] or ties
    first = rng.choice(candidates)
    others = [o for o in ties if o != first]
    rng.shuffle(others)
    listed = [first, *others, *sorted(rest, key=lambda o: -sc.p_star[o])]
    parts = [f"{o} {round(sc.p_star[o] * 100)} %" for o in listed]
    text = ", ".join(parts[:-1]) + f" and {parts[-1]}"
    thing = sc.thing[0].upper() + sc.thing[1:]
    facts = [
        *sc.fillers,
        f"{thing} has not been decided yet. It will be decided by a random draw on the day, with no other influence.",
        f"The chances are: {text}. No other outcome is possible.",
    ]
    return facts, first


def relisted_items(n_scenarios: int, seed: int = 0) -> list[Item]:
    """Re-listed tie cells for the uniform and binary synthetic scenarios.

    Args:
        n_scenarios: Synthetic scenarios to scan.
        seed: Scenario seed (also the re-listing seed).

    Returns:
        Items (units ``relist:<scenario>``).
    """
    items = []
    for i in range(n_scenarios):
        sc = make_scenario(i, seed)
        if sc.profile not in RELIST_PROFILES:
            continue
        facts, first = reordered_chances(sc, seed)
        items.append(
            Item(
                unit=f"relist:{i}",
                state={"facts": facts, "question": f"What is {sc.thing}?"},
                options=sc.options,
                instructions=CHOICE_INSTRUCTIONS,
                nouls={"determined": DETERMINED_QUESTION, "settled": SETTLED_QUESTION},
                info={
                    "set": "relist",
                    "scenario": i,
                    "profile": sc.profile,
                    "first_listed": first,
                    "first_shown": sc.options[0],
                },
            )
        )
    return items
