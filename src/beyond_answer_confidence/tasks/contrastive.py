"""Contrastive policy cases: does the answer follow the facts that decide it?

Six made-up policies each decide one of two outcomes from two facts. For
every generated case the label is computed by code, and five variants are
built:

- ``base``: both facts;
- ``counterfactual``: one fact changed so the label flips (one sentence, at
  most :data:`MAX_EDIT_WORDS` words different, checked);
- ``deleted``: a fact that decides the label removed (checked: the rule
  gives different labels for the two values of the removed fact);
- ``irrelevant``: base plus an irrelevant fact;
- ``contradictory``: base plus a sentence contradicting a deciding fact.

Each item asks a three-way choice (the two outcomes and "cannot tell"), a
yes/no question on the outcome and the ``enough`` question.
"""

import difflib
import random
from collections.abc import Callable
from dataclasses import dataclass

from beyond_answer_confidence.tasks import seeds
from beyond_answer_confidence.tasks.schema import Item

MAX_EDIT_WORDS = 8
"""Largest word difference allowed between a fact and its counterfactual."""

CANNOT_TELL = "cannot_tell"
"""Option key for "cannot tell"."""

CANNOT_TELL_DESCRIPTION = "cannot tell from the information given"

ENOUGH_QUESTION = (
    "Do the facts given contain enough information to answer the question for certain?"
)
"""Yes/no question: do the facts decide the answer?"""

VARIANTS = ("base", "counterfactual", "deleted", "irrelevant", "contradictory")
"""Variants of every case, in build order."""

NAMES = (
    "Asha",
    "Ben",
    "Chloe",
    "Dev",
    "Elena",
    "Farid",
    "Grace",
    "Hugo",
    "Ines",
    "Jonas",
    "Kemi",
    "Liam",
    "Mira",
    "Noah",
    "Olga",
    "Pedro",
    "Quinn",
    "Rosa",
    "Sami",
    "Tara",
)
TOWNS = ("Hollowbrook", "Marsden", "Eastwick", "Dunmore", "Kelby", "Ashford")

_SEED_NAMESPACE = seeds.CONTRASTIVE_FACTS


@dataclass(frozen=True)
class Fact:
    """A fact with a sentence for each truth value.

    Attributes:
        true: Sentence when the fact holds (``{n}`` = name, ``{v}`` = number).
        false: Sentence when it does not.
        numeric: ``((lo, hi) when true, (lo, hi) when false)`` for numeric
            facts, else ``None``.
    """

    true: str
    false: str
    numeric: tuple[tuple[int, int], tuple[int, int]] | None = None


@dataclass(frozen=True)
class Policy:
    """A policy that decides one of two outcomes from two facts.

    Attributes:
        name: Policy id.
        policy: The policy text.
        question: Question with ``{n}`` for the person's name.
        outcomes: ``(outcome when the rule holds, outcome otherwise)``.
        outcome_desc: Option descriptions of the two outcomes.
        facts: The two facts.
        rule: Label from the two truth values.
        irrelevant: An irrelevant fact.
    """

    name: str
    policy: str
    question: str
    outcomes: tuple[str, str]
    outcome_desc: tuple[str, str]
    facts: tuple[Fact, Fact]
    rule: Callable[[bool, bool], bool]
    irrelevant: str


def _both(a: bool, b: bool) -> bool:
    return a and b


def _either(a: bool, b: bool) -> bool:
    return a or b


POLICIES = (
    Policy(
        "refund",
        "A refund is given only if the item is returned within 30 days of purchase and has not been used.",
        "Does {n} get a refund?",
        ("refund", "no_refund"),
        ("gets a refund", "does not get a refund"),
        (
            Fact(
                "{n} returned the kettle {v} days after buying it.",
                "{n} returned the kettle {v} days after buying it.",
                ((2, 29), (31, 90)),
            ),
            Fact(
                "The kettle had never been used.",
                "The kettle had been used several times.",
            ),
        ),
        _both,
        "{n} paid for the kettle by card.",
    ),
    Policy(
        "library",
        "A late fee is charged when a book is returned after its due date, unless the borrower works at the library.",
        "Is {n} charged a late fee?",
        ("fee", "no_fee"),
        ("is charged a late fee", "is not charged a late fee"),
        (
            Fact(
                "{n} returned the book {v} days after the due date.",
                "{n} returned the book {v} days before the due date.",
                ((1, 20), (1, 20)),
            ),
            Fact("{n} does not work at the library.", "{n} works at the library."),
        ),
        _both,
        "The book was a novel about sailing.",
    ),
    Policy(
        "lab",
        "Entry to the lab requires both a completed safety course and a valid badge.",
        "Is {n} allowed into the lab?",
        ("allowed", "not_allowed"),
        ("is allowed into the lab", "is not allowed into the lab"),
        (
            Fact(
                "{n} completed the safety course last month.",
                "{n} has not completed the safety course.",
            ),
            Fact(
                "{n}'s badge is valid until next year.",
                "{n}'s badge expired last week.",
            ),
        ),
        _both,
        "{n} usually arrives at the lab before nine.",
    ),
    Policy(
        "grades",
        "A student passes if the exam score is at least 60 or the coursework score is at least 80.",
        "Does {n} pass?",
        ("pass", "fail"),
        ("passes", "fails"),
        (
            Fact(
                "{n} scored {v} on the exam.",
                "{n} scored {v} on the exam.",
                ((60, 95), (20, 59)),
            ),
            Fact(
                "{n} scored {v} on the coursework.",
                "{n} scored {v} on the coursework.",
                ((80, 98), (30, 79)),
            ),
        ),
        _either,
        "{n} sat in the front row during the exam.",
    ),
    Policy(
        "warranty",
        "Repairs are free if the device is less than 24 months old and the damage was not caused by water.",
        "Is {n}'s repair free?",
        ("free", "charged"),
        ("gets a free repair", "is charged for the repair"),
        (
            Fact(
                "{n}'s phone is {v} months old.",
                "{n}'s phone is {v} months old.",
                ((2, 23), (25, 60)),
            ),
            Fact(
                "The damage was caused by a fall onto the floor.",
                "The damage was caused by water.",
            ),
        ),
        _both,
        "{n}'s phone has a blue case.",
    ),
    Policy(
        "shipping",
        "Shipping is free for orders over 50 dollars and for members.",
        "Does {n} get free shipping?",
        ("free_shipping", "paid_shipping"),
        ("gets free shipping", "pays for shipping"),
        (
            Fact(
                "{n}'s order came to {v} dollars.",
                "{n}'s order came to {v} dollars.",
                ((51, 120), (10, 49)),
            ),
            Fact("{n} is a member.", "{n} is not a member."),
        ),
        _either,
        "{n} ordered two books and a lamp.",
    ),
)
"""The six policies, in build order."""


def fact_sentence(fact: Fact, value: bool, name: str, rng: random.Random) -> str:
    """Render a fact for one truth value (numeric facts draw a number).

    Args:
        fact: The fact.
        value: Its truth value.
        name: Person's name.
        rng: Generator for numeric facts (one ``randint`` call).

    Returns:
        The sentence.
    """
    template = fact.true if value else fact.false
    if fact.numeric is None:
        return template.format(n=name)
    lo, hi = fact.numeric[0] if value else fact.numeric[1]
    return template.format(n=name, v=rng.randint(lo, hi))


def word_diff(a: str, b: str) -> int:
    """Number of whitespace-separated words changed between two sentences.

    Args:
        a: First sentence.
        b: Second sentence.

    Returns:
        Words inserted, deleted or replaced (counted on the longer side).
    """
    sm = difflib.SequenceMatcher(a=a.split(), b=b.split())
    return sum(
        max(a_end - a_start, b_end - b_start)
        for op, a_start, a_end, b_start, b_end in sm.get_opcodes()
        if op != "equal"
    )


@dataclass(frozen=True)
class Case:
    """One generated case with all five variants.

    Attributes:
        policy: Policy id.
        index: Case number within the policy.
        name: Person's name.
        values: Truth values of the two facts.
        variants: Variant -> (facts, ideal choice answer).
        flipped: Fact changed in the counterfactual.
        deleted: Fact removed in ``deleted`` (and contradicted in
            ``contradictory``).
    """

    policy: str
    index: int
    name: str
    values: tuple[bool, bool]
    variants: dict[str, tuple[list[str], str]]
    flipped: int
    deleted: int


def _decisive(policy: Policy, values: tuple[bool, bool]) -> tuple[list[int], list[int]]:
    """Facts whose flip changes the label, and facts whose value decides it."""
    label = policy.rule(*values)

    def with_value(k: int, v: bool) -> bool:
        a, b = (v, values[1]) if k == 0 else (values[0], v)
        return policy.rule(a, b)

    flips = [k for k in (0, 1) if with_value(k, not values[k]) != label]
    deciding = [k for k in (0, 1) if with_value(k, True) != with_value(k, False)]
    return flips, deciding


def make_case(policy: Policy, index: int, seed: int = 0) -> Case:
    """Generate one case, redrawing truth values until every variant is defined.

    Args:
        policy: The policy.
        index: Case number.
        seed: Seed.

    Returns:
        The case.

    Raises:
        ValueError: If a counterfactual differs by no words or by more than
            :data:`MAX_EDIT_WORDS`.
    """
    rng = random.Random(f"{_SEED_NAMESPACE}:{seed}:{policy.name}:{index}")  # noqa: S311  # nosec B311
    name = rng.choice(NAMES)
    filler = f"{name} lives in {rng.choice(TOWNS)}."
    while True:
        values = (rng.random() < 0.5, rng.random() < 0.5)
        flips, deciding = _decisive(policy, values)
        if flips and deciding:
            break
    label = policy.rule(*values)
    sents = [
        fact_sentence(f, v, name, rng)
        for f, v in zip(policy.facts, values, strict=True)
    ]
    outcome = policy.outcomes[0] if label else policy.outcomes[1]
    other = policy.outcomes[1] if label else policy.outcomes[0]
    k_flip = rng.choice(flips)
    cf = list(sents)
    cf[k_flip] = fact_sentence(policy.facts[k_flip], not values[k_flip], name, rng)
    if not 0 < word_diff(sents[k_flip], cf[k_flip]) <= MAX_EDIT_WORDS:
        raise ValueError(f"counterfactual edit out of range: {cf[k_flip]!r}")
    k_del = rng.choice(deciding)
    contra = fact_sentence(policy.facts[k_del], not values[k_del], name, rng)
    base = [filler, *sents]
    variants = {
        "base": (base, outcome),
        "counterfactual": ([filler, *cf], other),
        "deleted": (
            [filler, *[s for j, s in enumerate(sents) if j != k_del]],
            CANNOT_TELL,
        ),
        "irrelevant": ([*base, policy.irrelevant.format(n=name)], outcome),
        "contradictory": ([*base, contra], CANNOT_TELL),
    }
    return Case(policy.name, index, name, values, variants, k_flip, k_del)


def case_items(policy: Policy, case: Case) -> list[Item]:
    """The five items of one case.

    Args:
        policy: The case's policy.
        case: The case.

    Returns:
        One item per variant (units ``<policy>:<index>:<variant>``).
    """
    q = policy.question.format(n=case.name)
    desc = {
        policy.outcomes[0]: f"{case.name} {policy.outcome_desc[0]}",
        policy.outcomes[1]: f"{case.name} {policy.outcome_desc[1]}",
        CANNOT_TELL: CANNOT_TELL_DESCRIPTION,
    }
    rule_label = policy.outcomes[0] if policy.rule(*case.values) else policy.outcomes[1]
    return [
        Item(
            unit=f"{policy.name}:{case.index}:{var}",
            state={"policy": policy.policy, "facts": facts},
            options=(policy.outcomes[0], policy.outcomes[1], CANNOT_TELL),
            gold=ideal,
            instructions=f"{q} Use the policy and the facts given.",
            descriptions=desc,
            nouls={"outcome": q, "enough": ENOUGH_QUESTION},
            info={
                "domain": policy.name,
                "case": case.index,
                "variant": var,
                "positive_outcome": policy.outcomes[0],
                "rule_label": rule_label,
                "deleted_fact": case.deleted,
            },
        )
        for var, (facts, ideal) in case.variants.items()
    ]


def contrastive_items(
    cases_per_policy: int = 100, seed: int = 0
) -> tuple[list[Case], list[Item]]:
    """Every case and its items (policy-major, then case, then variant).

    Args:
        cases_per_policy: Cases per policy.
        seed: Seed.

    Returns:
        ``(cases, items)``.
    """
    cases, items = [], []
    for policy in POLICIES:
        for i in range(cases_per_policy):
            case = make_case(policy, i, seed)
            cases.append(case)
            items.extend(case_items(policy, case))
    return cases, items
