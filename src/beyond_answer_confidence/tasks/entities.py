"""Entity questions graded by popularity, and fabricated twins.

Every PopQA question becomes a four-option multiple-choice item whose
distractors are other objects of the same relation; subject popularity
(Wikipedia page views) grades how likely the model is to know the fact.
*Fabricated twins* reuse each relation's most common question template and
option pool with a generated subject that refers to nothing: no option is
correct and the ideal distribution is uniform.
"""

import json
import random

import pandas as pd

from beyond_answer_confidence.data.distractors import sample_distractors
from beyond_answer_confidence.data.loaders import table_rows
from beyond_answer_confidence.tasks.multiple_choice import multiple_choice_item
from beyond_answer_confidence.tasks.schema import Item
from beyond_answer_confidence.tasks.synthetic import pseudo_word

PERSON_RELATIONS = frozenset(
    {"occupation", "place of birth", "father", "mother", "religion", "sport"}
)
"""PopQA relations whose subject is a person."""

WORK_RELATIONS = frozenset(
    {"author", "director", "screenwriter", "producer", "composer"}
)
"""PopQA relations whose subject is a creative work."""

_WORK_NOUNS = ("Letters", "Garden", "Winter", "Crossing", "Archive", "Harbour", "Tide")
_PLACE_NOUNS = ("Castle", "Bridge", "Abbey", "Station", "Mill", "Harbour")


def primary_templates(popqa: pd.DataFrame) -> dict[str, str]:
    """Most common question template per relation (subject replaced by ``{}``).

    Args:
        popqa: PopQA table (``prop``, ``subj``, ``question``).

    Returns:
        Relation -> template.
    """
    out = {}
    for prop, g in popqa.groupby("prop"):
        tmpl = g[g["subj"] != ""].apply(
            lambda r: r["question"].replace(str(r["subj"]), "{}"), axis=1
        )
        out[str(prop)] = str(tmpl.value_counts().index[0])
    return out


def fabricated_subject(prop: str, rng: random.Random) -> str:
    """A plausible-looking subject of the right kind that refers to nothing.

    Args:
        prop: PopQA relation.
        rng: Random generator.

    Returns:
        The subject string.

    Raises:
        ValueError: For a relation outside PopQA's sixteen.
    """
    if prop in PERSON_RELATIONS:
        return f"{pseudo_word(rng, 2)} {pseudo_word(rng, 3)}"
    if prop in WORK_RELATIONS:
        return f"The {pseudo_word(rng, 2)} {rng.choice(_WORK_NOUNS)}"
    if prop == "genre":
        return pseudo_word(rng, 3)
    if prop == "capital":
        return f"{pseudo_word(rng, 3)} {rng.choice(['County', 'Province', 'District'])}"
    if prop == "capital of":
        return pseudo_word(rng, 3)
    if prop == "country":
        return f"{pseudo_word(rng, 2)} {rng.choice(_PLACE_NOUNS)}"
    if prop == "color":
        return f"flag of {pseudo_word(rng, 3)}"
    raise ValueError(prop)


def popqa_items(popqa: pd.DataFrame, seed: int = 0) -> list[Item]:
    """PopQA questions as four-option multiple choice.

    Questions whose relation pool has fewer than three usable distractors
    are skipped.

    Args:
        popqa: PopQA table as returned by
            :func:`beyond_answer_confidence.data.loaders.load_popqa`.
        seed: Seed for distractors and option order.

    Returns:
        Items with info ``set="popqa"``, ``prop``, ``s_pop`` and
        ``fabricated=False``.
    """
    pools = relation_pools(popqa)
    items = []
    for row in table_rows(popqa):
        rng = random.Random(f"popqa:{seed}:{row.id}")  # noqa: S311  # nosec B311
        aliases = list(row.possible_answers) + json.loads(str(row.o_aliases) or "[]")
        dis = sample_distractors(str(row.obj), pools[str(row.prop)], aliases, 3, rng)
        if dis is None:
            continue
        items.append(
            multiple_choice_item(
                f"popqa:{row.id}",
                str(row.question),
                [str(row.obj), *dis],
                0,
                rng,
                {
                    "set": "popqa",
                    "prop": row.prop,
                    "s_pop": int(row.s_pop),
                    "fabricated": False,
                },
            )
        )
    return items


def relation_pools(popqa: pd.DataFrame) -> dict[str, list[str]]:
    """Sorted distinct objects per relation (the distractor pools).

    Args:
        popqa: PopQA table.

    Returns:
        Relation -> objects.
    """
    return {str(p): sorted(set(g["obj"])) for p, g in popqa.groupby("prop")}


def fabricated_items(
    popqa: pd.DataFrame, per_relation: int, seed: int = 0
) -> list[Item]:
    """Fabricated twins: PopQA templates and option pools about made-up subjects.

    Args:
        popqa: PopQA table.
        per_relation: Items per relation.
        seed: Seed for names and options.

    Returns:
        Items (no gold) with info ``set="fabricated"``, ``prop``,
        ``subject`` and ``fabricated=True``, relation by relation in sorted
        order.
    """
    pools = relation_pools(popqa)
    templates = primary_templates(popqa)
    items = []
    for prop in sorted(pools):
        for j in range(per_relation):
            rng = random.Random(f"fab:{seed}:{prop}:{j}")  # noqa: S311  # nosec B311
            subj = fabricated_subject(prop, rng)
            opts = rng.sample(pools[prop], 4)
            items.append(
                multiple_choice_item(
                    f"fab:{prop}:{j}",
                    templates[prop].format(subj),
                    opts,
                    None,
                    rng,
                    {
                        "set": "fabricated",
                        "prop": prop,
                        "subject": subj,
                        "fabricated": True,
                    },
                )
            )
    return items
