"""The same questions with a second kind of wrong options.

Two converted sets are asked with both distractor constructions used in the
study, on seeded samples of questions:

- **TriviaQA**: ``nearest`` (the three nearest other answers in
  sentence-embedding space, the original construction) and ``pool`` (three
  answers sampled from the gold answers of other questions with the same
  answer type).
- **PopQA**: ``pool`` (three answers sampled from the relation's answer pool,
  the original construction) and ``nearest`` (the three nearest answers in
  the relation's pool in sentence-embedding space).

Aliases of the gold answer are excluded either way; option order is seeded
per question and arm. Every item asks the ``known`` yes/no question too.
"""

import json
import random
from collections import defaultdict

import pandas as pd

from beyond_answer_confidence.data.distractors import (
    DistractorPicker,
    sample_distractors,
)
from beyond_answer_confidence.tasks.benchmarks import display_case
from beyond_answer_confidence.tasks.entities import relation_pools
from beyond_answer_confidence.tasks.multiple_choice import multiple_choice_item
from beyond_answer_confidence.tasks.schema import Item

SEED_NAMESPACE = "distractor-swap"
"""Namespace of every random draw in this module."""


def triviaqa_swap_items(
    triviaqa: pd.DataFrame, pick: DistractorPicker, n: int = 1000, seed: int = 0
) -> list[Item]:
    """TriviaQA questions with nearest and pool distractors.

    Args:
        triviaqa: TriviaQA validation table.
        pick: Nearest-distractor picker.
        n: Questions sampled (among those both constructions can serve).
        seed: Seed.

    Returns:
        Items with units ``swap:triviaqa:<arm>:<question id>``.
    """
    first = triviaqa.drop_duplicates("question_id").reset_index(drop=True)
    golds = [str(a["value"]) for a in first["answer"]]
    aliases = [list(a["aliases"]) for a in first["answer"]]
    types = [str(a["type"]) for a in first["answer"]]
    near = pick(golds, aliases, golds)
    by_type: dict[str, list[str]] = defaultdict(list)
    for g, t in zip(golds, types, strict=True):
        by_type[t].append(g)
    order = list(range(len(first)))
    random.Random(f"{SEED_NAMESPACE}:{seed}:triviaqa").shuffle(order)  # noqa: S311  # nosec B311
    items: list[Item] = []
    for i in order:
        if len(items) == 2 * n:
            break
        qid = str(first.loc[i, "question_id"])
        r = random.Random(f"{SEED_NAMESPACE}:{seed}:pool:{qid}")  # noqa: S311  # nosec B311
        pool = sample_distractors(golds[i], by_type[types[i]], aliases[i], 3, r)
        nearest = near[i]
        if nearest is None or pool is None:
            continue
        for arm, dis in (("nearest", nearest), ("pool", pool)):
            rng = random.Random(f"{SEED_NAMESPACE}:{seed}:order:{arm}:{qid}")  # noqa: S311  # nosec B311
            items.append(
                multiple_choice_item(
                    f"swap:triviaqa:{arm}:{qid}",
                    str(first.loc[i, "question"]),
                    [display_case(a) for a in (golds[i], *dis)],
                    0,
                    rng,
                    {"set": "triviaqa", "arm": arm, "question_id": qid,
                     "answer_type": types[i]},
                )
            )  # fmt: skip
    return items


def popqa_swap_items(
    popqa: pd.DataFrame, pick: DistractorPicker, n: int = 1000, seed: int = 0
) -> list[Item]:
    """PopQA questions with pool and nearest distractors.

    Args:
        popqa: PopQA table.
        pick: Nearest-distractor picker.
        n: Questions sampled.
        seed: Seed.

    Returns:
        Items with units ``swap:popqa:<arm>:<id>``.
    """
    pools = relation_pools(popqa)
    sample = popqa.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    chosen: list[tuple[int, list[str], list[str]]] = []
    for prop, g in sample.groupby("prop", sort=True):
        records = g.reset_index().to_dict("records")
        golds = [str(rec["obj"]) for rec in records]
        aliases = [
            [str(a) for a in rec["possible_answers"]]
            + json.loads(str(rec["o_aliases"]) or "[]")
            for rec in records
        ]
        near = pick(golds, aliases, pools[str(prop)])
        for rec, al, d in zip(records, aliases, near, strict=True):
            if d is not None:
                chosen.append((int(rec["index"]), al, d))
    chosen.sort()
    items: list[Item] = []
    for idx, al, near_d in chosen:
        if len(items) == 2 * n:
            break
        row = sample.loc[idx]
        draw = random.Random(f"popqa:{seed}:{row['id']}")  # noqa: S311  # nosec B311
        pool_d = sample_distractors(
            str(row["obj"]), pools[str(row["prop"])], al, 3, draw
        )
        if pool_d is None:
            continue
        for arm, dis in (("pool", pool_d), ("nearest", near_d)):
            rng = random.Random(f"{SEED_NAMESPACE}:{seed}:order:{arm}:{row['id']}")  # noqa: S311  # nosec B311
            items.append(
                multiple_choice_item(
                    f"swap:popqa:{arm}:{row['id']}",
                    str(row["question"]),
                    [str(row["obj"]), *dis],
                    0,
                    rng,
                    {"set": "popqa", "arm": arm, "question_id": str(row["id"]),
                     "prop": str(row["prop"]), "s_pop": float(str(row["s_pop"]))},
                )
            )  # fmt: skip
    return items
