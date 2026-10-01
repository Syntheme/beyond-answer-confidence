"""Dated questions about real-world news events (Daily Oracle).

Yes/no and four-option questions are sampled per month, so accuracy against
the event month can locate the model's knowledge cutoff (see
:func:`beyond_answer_confidence.stats.changepoint.change_point`).
"""

import pandas as pd

from beyond_answer_confidence.data.loaders import table_rows
from beyond_answer_confidence.tasks.multiple_choice import LETTERS
from beyond_answer_confidence.tasks.schema import Item

NEWS_INSTRUCTIONS = (
    "Choose the correct answer to the question about a real-world event."
)
"""Choice instructions for news questions."""

NEWS_KNOWN_QUESTION = "Do you know for certain how this turned out?"
"""Yes/no question: does the model claim to know the outcome?"""

NEWS_OPTIONS = LETTERS[:4]
"""Option keys of the four-option news questions."""


def parse_yes_no(raw: str) -> str | None:
    """Normalise a yes/no answer label (``"** No."`` -> ``"no"``).

    Args:
        raw: Raw answer label.

    Returns:
        ``"yes"``, ``"no"``, or ``None`` for anything else.
    """
    s = raw.strip(" *\n.").lower()
    return s if s in ("yes", "no") else None


def yes_no_news_items(tf: pd.DataFrame, per_month: int, seed: int = 0) -> list[Item]:
    """Yes/no news questions, up to ``per_month`` sampled per month.

    Args:
        tf: Yes/no table as returned by
            :func:`beyond_answer_confidence.data.loaders.load_daily_oracle` (``"tf"``).
        per_month: Questions sampled per month (all if fewer).
        seed: ``random_state`` of the per-month ``DataFrame.sample``.

    Returns:
        Items with options ``("yes", "no")`` and info ``set="oracle_tf"``,
        ``month``, ``date``, ``category``; units are ``oracle_tf:<row>``,
        where ``<row>`` is the position among rows with a valid label.
    """
    tf = tf.assign(gold=tf["answer"].map(parse_yes_no))
    tf = tf[tf["gold"].notna()].reset_index()
    items: list[Item] = []
    for month, g in tf.groupby("month"):
        take = g.sample(min(per_month, len(g)), random_state=seed)
        items.extend(
            Item(
                unit=f"oracle_tf:{row.index}",
                state={"question": row.question},
                options=("yes", "no"),
                gold=str(row.gold),
                instructions=NEWS_INSTRUCTIONS,
                nouls={"known": NEWS_KNOWN_QUESTION},
                info={
                    "set": "oracle_tf",
                    "month": month,
                    "date": row.date,
                    "category": row.category,
                },
            )
            for row in table_rows(take)
        )
    return items


def multiple_choice_news_items(
    mc: pd.DataFrame, per_month: int, seed: int = 0
) -> list[Item]:
    """Four-option news questions, up to ``per_month`` sampled per month.

    Args:
        mc: Multiple-choice table as returned by
            :func:`beyond_answer_confidence.data.loaders.load_daily_oracle` (``"mc"``).
        per_month: Questions sampled per month (all if fewer).
        seed: ``random_state`` of the per-month ``DataFrame.sample``.

    Returns:
        Items with options ``A``-``D`` and info ``set="oracle_mc"``,
        ``month``, ``date``, ``category``; units are ``oracle_mc:<row>``
        (row position in the full table).
    """
    mc = mc.reset_index()
    mc = mc[mc["answer"].str.strip().str.lower().isin(["a", "b", "c", "d"])]
    items: list[Item] = []
    for month, g in mc.groupby("month"):
        take = g.sample(min(per_month, len(g)), random_state=seed)
        items.extend(
            Item(
                unit=f"oracle_mc:{row.index}",
                state={"question": row.question},
                options=NEWS_OPTIONS,
                gold=str(row.answer).strip().upper(),
                instructions=NEWS_INSTRUCTIONS,
                descriptions={
                    "A": row.choice_a,
                    "B": row.choice_b,
                    "C": row.choice_c,
                    "D": row.choice_d,
                },
                nouls={"known": NEWS_KNOWN_QUESTION},
                info={
                    "set": "oracle_mc",
                    "month": month,
                    "date": row.date,
                    "category": row.category,
                },
            )
            for row in table_rows(take)
        )
    return items
