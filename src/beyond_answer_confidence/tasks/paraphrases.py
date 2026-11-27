"""Real PopQA questions asked in several meaning-preserving wordings.

For each relation, a seeded sample of questions whose subject is not empty
is asked in four wordings: the relation's most common PopQA template and
three hand-written paraphrases with the same meaning. All wordings of a
question share the same four options (gold + three wrong answers sampled
from the relation's pool) in the same order, so only the wording changes.
Every item also asks the ``known`` yes/no question.
"""

import random

import pandas as pd

from beyond_answer_confidence.data.distractors import sample_distractors
from beyond_answer_confidence.tasks.entities import relation_pools
from beyond_answer_confidence.tasks.multiple_choice import multiple_choice_item
from beyond_answer_confidence.tasks.schema import Item

SEED_NAMESPACE = "paraphrases"
"""Namespace of every random draw in this module."""

PARAPHRASES: dict[str, tuple[str, str, str, str]] = {
    "author": ("Who is the author of {}?", "Who wrote {}?",
               "{} was written by whom?", "Which writer is the author of {}?"),
    "capital": ("What is the capital of {}?", "Which city is the capital of {}?",
                "What city serves as the capital of {}?",
                "The capital of {} is which city?"),
    "capital of": ("What is {} the capital of?", "{} is the capital of which place?",
                   "Of which place is {} the capital?",
                   "Which place has {} as its capital?"),
    "color": ("What color is {}?", "What colour is {}?", "Which color is {}?",
              "{} is what color?"),
    "composer": ("Who was the composer of {}?", "Who composed {}?",
                 "{} was composed by whom?", "Who wrote the music of {}?"),
    "country": ("In what country is {}?", "Which country is {} in?",
                "{} is located in which country?",
                "In which country can {} be found?"),
    "director": ("Who was the director of {}?", "Who directed {}?",
                 "{} was directed by whom?", "Who is credited as the director of {}?"),
    "father": ("Who is the father of {}?", "Who is {}'s father?",
               "Which person is the father of {}?", "{}'s father is who?"),
    "genre": ("What genre is {}?", "Which genre does {} belong to?",
              "What is the genre of {}?", "{} belongs to which genre?"),
    "mother": ("Who is the mother of {}?", "Who is {}'s mother?",
               "Which person is the mother of {}?", "{}'s mother is who?"),
    "occupation": ("What is {}'s occupation?", "What is the occupation of {}?",
                   "What does {} do for a living?", "What is {}'s profession?"),
    "place of birth": ("In what city was {} born?", "Which city was {} born in?",
                       "{} was born in which city?", "What is {}'s city of birth?"),
    "producer": ("Who was the producer of {}?", "Who produced {}?",
                 "{} was produced by whom?", "Who is credited as the producer of {}?"),
    "religion": ("What is the religion of {}?", "What religion does {} follow?",
                 "Which religion is {} associated with?", "What is {}'s religion?"),
    "screenwriter": ("Who was the screenwriter for {}?",
                     "Who wrote the screenplay for {}?",
                     "Who is credited as the screenwriter of {}?",
                     "The screenplay of {} was written by whom?"),
    "sport": ("What sport does {} play?", "Which sport does {} play?",
              "{} plays which sport?", "In what sport does {} compete?"),
}  # fmt: skip
"""Four wordings per relation; the first is PopQA's most common template."""


def paraphrase_items(
    popqa: pd.DataFrame, per_relation: int = 25, seed: int = 0
) -> list[Item]:
    """Every question of the sample in every wording.

    Args:
        popqa: PopQA table.
        per_relation: Questions per relation.
        seed: Seed.

    Returns:
        Items with units ``para:<popqa id>:<k>`` (``k`` = wording) and info
        ``question_id``, ``prop``, ``subject``, ``s_pop`` and ``wording``.
    """
    pools = relation_pools(popqa)
    items = []
    for prop, wordings in PARAPHRASES.items():
        g = popqa[(popqa["prop"] == prop) & (popqa["subj"].astype(str) != "")]
        ids = sorted(g["id"].tolist())
        random.Random(f"{SEED_NAMESPACE}:{seed}:{prop}").shuffle(ids)  # noqa: S311  # nosec B311
        taken = 0
        for qid in ids:
            if taken == per_relation:
                break
            row = g[g["id"] == qid].iloc[0]
            pick = random.Random(f"{SEED_NAMESPACE}:{seed}:options:{qid}")  # noqa: S311  # nosec B311
            aliases = list(row["possible_answers"])
            dis = sample_distractors(str(row["obj"]), pools[prop], aliases, 3, pick)
            if dis is None:
                continue
            taken += 1
            for k, template in enumerate(wordings):
                order = random.Random(f"{SEED_NAMESPACE}:{seed}:order:{qid}")  # noqa: S311  # nosec B311
                items.append(
                    multiple_choice_item(
                        f"para:{qid}:{k}",
                        template.format(str(row["subj"])),
                        [str(row["obj"]), *dis],
                        0,
                        order,
                        {"question_id": int(qid), "prop": prop,
                         "subject": str(row["subj"]),
                         "s_pop": float(str(row["s_pop"])), "wording": k},
                    )
                )  # fmt: skip
    return items
