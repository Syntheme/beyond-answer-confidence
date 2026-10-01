"""Standard multiple choice in original option order, and intent choices with K options.

- MMLU-Redux 2.0 (with its annotated error types), MMLU-CF and ANLI, asked
  in the dataset's own option order (some options refer to others by
  letter or position, e.g. "Both A and C");
- a CLINC150 sweep over the number of options: each test utterance is asked
  with its gold intent plus ``K - 1`` random other intents.
"""

import random
import re
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from beyond_answer_confidence.data.intents import IntentDataset, select_items
from beyond_answer_confidence.data.loaders import table_rows
from beyond_answer_confidence.tasks.benchmarks import NLI_INSTRUCTIONS, NLI_OPTIONS
from beyond_answer_confidence.tasks.intents import state_for
from beyond_answer_confidence.tasks.multiple_choice import (
    KNOWN_QUESTION,
    LETTERS,
    QA_INSTRUCTIONS,
)
from beyond_answer_confidence.tasks.schema import Item

OPTION_COUNTS = (5, 10, 25, 50, 100, 150)
"""Numbers of intent options in the sweep."""

INTENT_INSTRUCTIONS = "Which intent does the customer message belong to?"

NLI_LABELS = ("entailment", "neutral", "contradiction")
"""ANLI labels in label-index order (also the option order)."""


def parse_corrected(raw: Any) -> int | None:
    """Parse an MMLU-Redux ``correct_answer`` into a single option index.

    Args:
        raw: Free-text field ("2", "C", "2 and 3", prose ...).

    Returns:
        Index 0-3, or ``None`` unless exactly one option is named.
    """
    if not isinstance(raw, str):
        return None
    s = raw.strip()
    if re.fullmatch(r"[0-3]", s):
        return int(s)
    if re.fullmatch(r"[A-Da-d]", s):
        return "ABCD".index(s.upper())
    return None


def fixed_order_item(
    unit: str,
    question: str,
    choices: Sequence[Any],
    gold: int,
    info: Mapping[str, Any],
) -> Item:
    """A lettered question in the dataset's option order, with ``known``.

    Args:
        unit: Unit id.
        question: Question text.
        choices: Option texts in dataset order.
        gold: Index of the correct option.
        info: Metadata; ``letter_of_index`` (index -> letter) is added.

    Returns:
        The item.
    """
    letters = LETTERS[: len(choices)]
    return Item(
        unit=unit,
        state={"question": question},
        options=letters,
        gold=letters[gold],
        instructions=QA_INSTRUCTIONS,
        descriptions={letters[i]: str(c).strip() for i, c in enumerate(choices)},
        nouls={"known": KNOWN_QUESTION},
        info={
            **info,
            "letter_of_index": {str(i): letters[i] for i in range(len(choices))},
        },
    )


def mmlu_redux_items(redux: pd.DataFrame) -> list[Item]:
    """MMLU-Redux 2.0 items (every question, flawed ones included).

    Args:
        redux: Table from
            :func:`beyond_answer_confidence.data.multiple_choice.load_mmlu_redux`.

    Returns:
        Items with info ``set="mmlu_redux"``, ``subject``, ``error_type`` and
        the parsed ``corrected`` index.
    """
    return [
        fixed_order_item(
            f"redux:{r['subject']}:{i}",
            r["question"],
            list(r["choices"]),
            int(r["answer"]),
            {
                "set": "mmlu_redux",
                "subject": r["subject"],
                "error_type": r["error_type"],
                "corrected": parse_corrected(r["correct_answer"]),
            },
        )
        for i, r in redux.iterrows()
    ]


def mmlu_cf_items(cf: pd.DataFrame) -> list[Item]:
    """MMLU-CF validation items.

    Args:
        cf: Table from :func:`beyond_answer_confidence.data.multiple_choice.load_mmlu_cf`.

    Returns:
        Items with info ``set="mmlu_cf"`` and ``subject``.
    """
    return [
        fixed_order_item(
            f"mmlucf:{r['subject']}:{i}",
            r["Question"],
            [r["A"], r["B"], r["C"], r["D"]],
            "ABCD".index(str(r["Answer"]).strip()),
            {"set": "mmlu_cf", "subject": r["subject"]},
        )
        for i, r in cf.iterrows()
    ]


def anli_items(anli: pd.DataFrame) -> list[Item]:
    """ANLI items (three NLI labels).

    Args:
        anli: Table from :func:`beyond_answer_confidence.data.multiple_choice.load_anli`.

    Returns:
        Items with info ``set="anli"`` and ``round``.
    """
    return [
        Item(
            unit=f"anli:{r.uid}",
            state={"premise": r.premise, "hypothesis": r.hypothesis},
            options=NLI_LABELS,
            gold=NLI_LABELS[int(r.label)],
            instructions=NLI_INSTRUCTIONS,
            descriptions=NLI_OPTIONS,
            info={"set": "anli", "round": int(r.round)},
        )
        for r in table_rows(anli)
    ]


def option_count_items(
    dataset: IntentDataset,
    per_intent: int = 10,
    counts: Sequence[int] = OPTION_COUNTS,
) -> list[Item]:
    """Intent choices with the gold plus ``K - 1`` random other intents.

    The options of each (utterance, K) pair are drawn and shuffled with a
    generator seeded by both, so every K is an independent draw.

    Args:
        dataset: Intent dataset (test split used).
        per_intent: Test utterances per intent.
        counts: Numbers of options.

    Returns:
        Items (units ``clincK:<index>:<K>``; info ``set="clinc_k"``, ``K``,
        ``item``).
    """
    items = []
    for idx in select_items(dataset.test, per_intent):
        ex = dataset.test[idx]
        for k in counts:
            rng = random.Random(f"clinc-k:{idx}:{k}")  # noqa: S311  # nosec B311
            others = rng.sample(
                [lab for lab in dataset.labels if lab != ex.label], k - 1
            )
            opts = [ex.label, *others]
            rng.shuffle(opts)
            items.append(
                Item(
                    unit=f"clincK:{idx}:{k}",
                    state=state_for(ex.text),
                    options=tuple(opts),
                    gold=ex.label,
                    instructions=INTENT_INSTRUCTIONS,
                    info={"set": "clinc_k", "K": k, "item": idx},
                )
            )
    return items
