"""Stated-odds questions under an explicit scoring rule, and new odds profiles.

Two parts, each asked as Choice in two wordings, ``plain`` (the original
"Choose the correct answer to the question.") and ``log_score`` (asks for
the probability of each outcome and states that the score is the logarithm
of the probability given to the outcome that actually happens):

- **Devices**: the dice, coin and card questions of
  :data:`~beyond_answer_confidence.tasks.chance.DEVICES` in seeded option
  orders.
- **Spinners**: new profiles of stated chances (three to six coloured
  sectors in steps of 5 %), each in seeded option orders, plus one request
  per profile asking one yes/no question per outcome ("Will the spinner land
  on red?").

Every option order is its own replicate index (see
:func:`~beyond_answer_confidence.tasks.chance.order_replicate`).
"""

import itertools
import random

from beyond_answer_confidence.tasks.chance import COLOURS, DEVICES, INSTRUCTIONS
from beyond_answer_confidence.tasks.schema import Item

SEED_NAMESPACE = "odds-semantics"
"""Namespace of every random draw in this module."""

LOG_SCORE = (
    "Give each option the probability that it is the actual outcome. Your "
    "score is the logarithm of the probability you give to the outcome that "
    "actually happens, so the best strategy is to state the true chances."
)
"""Choice instructions with an explicit logarithmic scoring rule."""

WORDINGS = {"plain": INSTRUCTIONS, "log_score": LOG_SCORE}
"""Arm -> Choice instructions."""

SPINNER_QUESTION = "Which colour will the spinner land on?"


def _orders(labels: list[str], n: int, key: str) -> list[tuple[str, ...]]:
    perms = list(itertools.permutations(labels))
    if len(perms) <= n:
        return perms
    rng = random.Random(f"{SEED_NAMESPACE}:orders:{key}")  # noqa: S311  # nosec B311
    return rng.sample(perms, n)


def spinner_profiles(n: int = 40, seed: int = 0) -> list[dict[str, float]]:
    """Profiles of stated chances, three to six sectors in steps of 5 %.

    Args:
        n: Profiles (sector counts cycle 3, 4, 5, 6).
        seed: Seed.

    Returns:
        Colour -> chance, in the order stated in the question.
    """
    out = []
    for i in range(n):
        rng = random.Random(f"{SEED_NAMESPACE}:{seed}:spinner:{i}")  # noqa: S311  # nosec B311
        k = 3 + i % 4
        colours = rng.sample(list(COLOURS), k)
        cuts = sorted(rng.sample(range(1, 20), k - 1))
        parts = [b - a for a, b in zip([0, *cuts], [*cuts, 20], strict=True)]
        out.append({c: p / 20 for c, p in zip(colours, parts, strict=True)})
    return out


def spinner_state(profile: dict[str, float]) -> str:
    """The sentence stating a spinner's chances.

    Args:
        profile: Colour -> chance.

    Returns:
        The setup sentence.
    """
    parts = [f"{c} covers {round(p * 100)}%" for c, p in profile.items()]
    listed = ", ".join(parts[:-1]) + f" and {parts[-1]}"
    return (
        f"A spinner is divided into coloured sectors: {listed} of it. It is spun once."
    )


def odds_items(
    device_orders: int = 24, spinners: int = 40, spinner_orders: int = 6, seed: int = 0
) -> list[Item]:
    """Every device, spinner and yes/no item.

    Args:
        device_orders: Option orders per device (all orders if fewer exist).
        spinners: Spinner profiles.
        spinner_orders: Option orders per spinner and wording.
        seed: Seed.

    Returns:
        Items; units end in the option-order index. Info: ``set``
        (``device``, ``spinner`` or ``spinner_yes_no``), ``arm``, ``key``
        (device or profile), ``order`` and ``ideal``.
    """
    items = []
    for name, (question, ideal, _n, _reps) in DEVICES.items():
        for k, order in enumerate(_orders(list(ideal), device_orders, name)):
            for arm, text in WORDINGS.items():
                items.append(
                    Item(
                        unit=f"odds:device:{arm}:{name}:{k}",
                        state={"question": question},
                        options=tuple(order),
                        instructions=text,
                        info={"set": "device", "arm": arm, "key": name,
                              "order": list(order), "ideal": dict(ideal)},
                    )
                )  # fmt: skip
    for i, profile in enumerate(spinner_profiles(spinners, seed)):
        state = {"setup": spinner_state(profile), "question": SPINNER_QUESTION}
        for k, order in enumerate(_orders(list(profile), spinner_orders, f"spin{i}")):
            for arm, text in WORDINGS.items():
                items.append(
                    Item(
                        unit=f"odds:spinner:{arm}:{i}:{k}",
                        state=state,
                        options=tuple(order),
                        instructions=text,
                        info={"set": "spinner", "arm": arm, "key": f"spinner{i}",
                              "order": list(order), "ideal": dict(profile)},
                    )
                )  # fmt: skip
        items.append(
            Item(
                unit=f"odds:spinner_yes_no:{i}:0",
                state={"setup": spinner_state(profile)},
                nouls={f"lands_{c}": f"Will the spinner land on {c}?" for c in profile},
                info={"set": "spinner_yes_no", "arm": "yes_no", "key": f"spinner{i}",
                      "order": list(profile), "ideal": dict(profile)},
            )
        )  # fmt: skip
    return items
