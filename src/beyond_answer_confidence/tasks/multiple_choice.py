"""Lettered multiple-choice items with a seeded option order."""

import random
from collections.abc import Mapping, Sequence
from typing import Any

from beyond_answer_confidence.tasks.schema import Item

LETTERS = tuple("ABCDEFGHIJKLMNOP")
"""Option keys, in display order (up to 16 options)."""

QA_INSTRUCTIONS = "Choose the correct answer to the question."
"""Choice instructions for closed-book questions."""

KNOWN_QUESTION = (
    "Do you know the answer to this question for certain, rather than having to guess?"
)
"""Yes/no question: does the model claim to know the answer?"""


def multiple_choice_item(
    unit: str,
    question: str,
    answers: Sequence[str],
    gold_index: int | None,
    rng: random.Random,
    info: Mapping[str, Any],
    *,
    instructions: str = QA_INSTRUCTIONS,
    known_question: str = KNOWN_QUESTION,
) -> Item:
    """Build a lettered multiple-choice item with a ``known`` yes/no question.

    The answers are shuffled with one ``rng.shuffle`` call and labelled
    ``A``, ``B``, ... in the shuffled order.

    Args:
        unit: Analysis-unit id.
        question: Question text (the whole state).
        answers: Answer texts (at most 16).
        gold_index: Position of the correct answer in ``answers``, or
            ``None`` if no answer is correct.
        rng: Random generator.
        info: Metadata for the analysis row.
        instructions: Choice instructions.
        known_question: Text of the ``known`` yes/no question.

    Returns:
        The item.
    """
    order = list(range(len(answers)))
    rng.shuffle(order)
    letters = LETTERS[: len(answers)]
    desc = {letters[pos]: answers[i] for pos, i in enumerate(order)}
    gold = letters[order.index(gold_index)] if gold_index is not None else None
    return Item(
        unit=unit,
        state={"question": question},
        options=letters,
        gold=gold,
        instructions=instructions,
        descriptions=desc,
        nouls={"known": known_question},
        info=info,
    )
