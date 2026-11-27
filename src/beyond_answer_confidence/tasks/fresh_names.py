"""Fresh realistic made-up names and matched real people.

A confirmation set for the look-alike contrast, built from people the
earlier sets did not use: the second-least-popular fifth (20 to 40 %) of each
person relation in PopQA. Per relation, half of that band is drawn at
random as *donors* and the rest are *real subjects*, so no real subject lends
a name to a made-up one. Each made-up name joins one donor's first name to
another donor's surname(s), as in
:func:`~beyond_answer_confidence.tasks.shortcuts.lookalike_subjects`, but
every donor gives its first name once and its surname once (a shuffled cyclic
pairing), so no name token recurs across made-up names and a text-only model
cannot learn the made-up group from repeated tokens. Pairs that make a PopQA
subject, an earlier look-alike name or a repeated first name are re-drawn.
Each relation gets as many made-up names as real subjects (the larger group
is subsampled).

Both groups are asked with the relation's most common question template and
four options from the relation's answer pool: three wrong answers plus the
gold for a real person, four wrong answers for a made-up one. Every item
also asks the ``known`` yes/no question.
"""

import random
from collections.abc import Collection

import pandas as pd

from beyond_answer_confidence.data.distractors import sample_distractors
from beyond_answer_confidence.tasks.entities import primary_templates, relation_pools
from beyond_answer_confidence.tasks.multiple_choice import multiple_choice_item
from beyond_answer_confidence.tasks.schema import Item
from beyond_answer_confidence.tasks.shortcuts import (
    PERSON_RELATION_ORDER,
    is_person_name,
)

SEED_NAMESPACE = "fresh-names"
"""Namespace of every random draw in this module."""

BAND = (0.2, 0.4)
"""Popularity quantiles (exclusive, inclusive) of the people used."""


def band_people(
    popqa: pd.DataFrame, band: tuple[float, float] = BAND
) -> dict[str, list[str]]:
    """People per person relation whose popularity lies in ``band``.

    Args:
        popqa: PopQA table (``subj``, ``prop``, ``s_pop``).
        band: Lower (exclusive) and upper (inclusive) popularity quantiles,
            per relation.

    Returns:
        Relation -> sorted distinct subject names that look like person names.
    """
    out = {}
    for prop in PERSON_RELATION_ORDER:
        g = popqa[popqa["prop"] == prop]
        lo, hi = g["s_pop"].quantile(band[0]), g["s_pop"].quantile(band[1])
        sel = g[(g["s_pop"] > lo) & (g["s_pop"] <= hi)]["subj"].astype(str)
        out[prop] = sorted({s for s in sel if is_person_name(s)})
    return out


def split_band(
    people: dict[str, list[str]], seed: int = 0, donor_share: float = 0.5
) -> tuple[list[str], dict[str, list[str]]]:
    """Split each relation's band people into donors and real subjects.

    Args:
        people: Relation -> people (see :func:`band_people`).
        seed: Seed.
        donor_share: Share of each relation's people drawn as donors.

    Returns:
        ``(donors pooled over relations, relation -> real subjects)``; a
        person listed under several relations is a donor everywhere if any
        draw makes them one.
    """
    donors: set[str] = set()
    real: dict[str, list[str]] = {}
    for prop in PERSON_RELATION_ORDER:
        names = list(people[prop])
        random.Random(f"{SEED_NAMESPACE}:{seed}:split:{prop}").shuffle(names)  # noqa: S311  # nosec B311
        cut = int(len(names) * donor_share)
        donors.update(names[:cut])
        real[prop] = names[cut:]
    return sorted(donors), {
        p: [s for s in r if s not in donors] for p, r in real.items()
    }


def unique_recombinations(
    donors: list[str], rng: random.Random, exclude: Collection[str], tries: int = 200
) -> list[tuple[str, str, str]]:
    """Made-up names using every donor once as first name and once as surname.

    The donors are shuffled and paired cyclically (each donor's first name
    with the next donor's surname); the shuffle is repeated until no pair
    makes an excluded name, a name repeated in the list, or joins two equal
    first names.

    Args:
        donors: Donor names (at least two).
        rng: Random generator.
        exclude: Names that must not be produced.
        tries: Maximum shuffles.

    Returns:
        ``(name, first-name donor, surname donor)`` per donor.

    Raises:
        RuntimeError: If no valid pairing is found in ``tries`` shuffles.
    """
    for _ in range(tries):
        order = list(donors)
        rng.shuffle(order)
        triples = []
        for i, a in enumerate(order):
            b = order[(i + 1) % len(order)]
            triples.append((f"{a.split()[0]} {' '.join(b.split()[1:])}", a, b))
        names = [t[0] for t in triples]
        if len(set(names)) == len(names) and not any(
            n in exclude or a.split()[0] == b.split()[0] for n, a, b in triples
        ):
            return triples
    raise RuntimeError("no valid donor pairing found")


def fresh_items(
    popqa: pd.DataFrame,
    exclude: Collection[str] = (),
    seed: int = 0,
    band: tuple[float, float] = BAND,
    donor_share: float = 0.5,
) -> list[Item]:
    """Made-up and matched real items for every person relation.

    Every real subject with a usable distractor set is used, and each
    relation gets the same number of made-up names.

    Args:
        popqa: PopQA table.
        exclude: Earlier made-up names that must not recur.
        seed: Seed.
        band: Popularity band (see :func:`band_people`).
        donor_share: Share of each relation's people used as donors.

    Returns:
        Items with units ``fresh:made_up:<relation>:<j>`` (no gold) and
        ``fresh:real:<relation>:<j>``; info ``set`` (``made_up`` or
        ``real``), ``prop``, ``subject``, ``first_donor`` and
        ``surname_donor`` (made up) and ``s_pop`` (real).
    """
    templates = primary_templates(popqa)
    pools = relation_pools(popqa)
    donors, real = split_band(band_people(popqa, band), seed, donor_share)
    real_items: list[Item] = []
    counts: dict[str, int] = {}
    for prop in PERSON_RELATION_ORDER:
        subjects = list(real[prop])
        random.Random(f"{SEED_NAMESPACE}:{seed}:real:{prop}").shuffle(subjects)  # noqa: S311  # nosec B311
        counts[prop] = 0
        for subj in subjects:
            row = popqa[(popqa["subj"] == subj) & (popqa["prop"] == prop)].iloc[0]
            j = counts[prop]
            r = random.Random(f"{SEED_NAMESPACE}:{seed}:real:{prop}:{j}")  # noqa: S311  # nosec B311
            aliases = list(row["possible_answers"])
            dis = sample_distractors(str(row["obj"]), pools[prop], aliases, 3, r)
            if dis is None:
                continue
            real_items.append(
                multiple_choice_item(
                    f"fresh:real:{prop}:{j}",
                    templates[prop].format(subj),
                    [str(row["obj"]), *dis],
                    0,
                    r,
                    {"set": "real", "prop": prop, "subject": subj,
                     "s_pop": float(str(row.at["s_pop"]))},
                )
            )  # fmt: skip
            counts[prop] += 1
    existing = set(popqa["subj"].astype(str)) | set(exclude)
    rng = random.Random(f"{SEED_NAMESPACE}:{seed}:names")  # noqa: S311  # nosec B311
    triples = unique_recombinations(donors, rng, existing)
    total = sum(counts.values())
    if len(triples) < total:  # fewer names than real subjects: subsample real
        keep_share = len(triples) / total
        kept = {p: int(counts[p] * keep_share) for p in counts}
        real_items = [
            it
            for it in real_items
            if int(it.unit.rsplit(":", 1)[1]) < kept[it.info["prop"]]
        ]
        counts = kept
    made_up: list[Item] = []
    pos = 0
    for prop in PERSON_RELATION_ORDER:
        for j in range(counts[prop]):
            name, a, b = triples[pos]
            pos += 1
            r = random.Random(f"{SEED_NAMESPACE}:{seed}:made_up:{prop}:{j}")  # noqa: S311  # nosec B311
            made_up.append(
                multiple_choice_item(
                    f"fresh:made_up:{prop}:{j}",
                    templates[prop].format(name),
                    r.sample(pools[prop], 4),
                    None,
                    r,
                    {"set": "made_up", "prop": prop, "subject": name,
                     "first_donor": a, "surname_donor": b},
                )
            )  # fmt: skip
    return made_up + real_items
