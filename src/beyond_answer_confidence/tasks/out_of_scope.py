"""Out-of-scope detection: messages that fit none of the listed intents.

This measures *schema mismatch*, not epistemic uncertainty: a model may
understand an out-of-scope request perfectly well and simply have no fitting
option. Three readouts per item are asked in one request (batching does not
change answers). Intents appear as opaque ordinal codes carrying the real
intent name, shuffled per item:

- ``closed``: a choice over the in-scope intents only;
- ``with_oos``: the same plus an explicit ``out_of_scope`` option;
- ``fits``: yes/no, "Does any of the listed intents fit the message?".

Data: CLINC150 has its own out-of-scope test split. For Banking77, a seeded
set of intents is held out of the schema; their test examples become
out-of-scope and the other intents stay in scope.
"""

import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from beyond_answer_confidence.backends.base import Request
from beyond_answer_confidence.backends.cache import CallRecord
from beyond_answer_confidence.data.intents import IntentDataset, select_items
from beyond_answer_confidence.tasks import seeds
from beyond_answer_confidence.tasks.intents import assign_codes, state_for

OOS_OPTION = "out_of_scope"
CLOSED_INSTRUCTIONS = "Which intent code does the customer message belong to?"
WITH_OOS_INSTRUCTIONS = (
    "Which intent code does the customer message belong to? "
    "Choose the out_of_scope code if none of the intents fit."
)
FITS_INSTRUCTIONS = "Does any of the listed intents fit the customer message?"


@dataclass(frozen=True)
class ScopeItem:
    """One utterance, whether it is out of scope, and its gold intent.

    Attributes:
        key: Stable item key ``<dataset>:in|oos:<index>`` (its
            :func:`seed_key` seeds the code assignment).
        text: The utterance.
        oos: Whether it is out of scope.
        gold: Gold intent (``None`` when out of scope).
    """

    key: str
    text: str
    oos: bool
    gold: str | None


def item_key(dataset: str, oos: bool, index: int) -> str:
    """Stable key of an item.

    Args:
        dataset: Dataset name.
        oos: Whether the item is out of scope.
        index: Index into the in-scope test split (in scope) or the
            out-of-scope source split.

    Returns:
        ``<dataset>:in|oos:<index>``.
    """
    return f"{dataset}:{'oos' if oos else 'in'}:{index}"


def seed_key(key: str) -> str:
    """The string that seeds an item's code assignment.

    Args:
        key: Item key from :func:`item_key`.

    Returns:
        The key with the fixed seed string inserted after the dataset name.

    Raises:
        ValueError: If the key is not of the form ``<dataset>:<rest>``.
    """
    dataset, sep, rest = key.partition(":")
    if not sep:
        raise ValueError(f"not an out-of-scope item key: {key!r}")
    return f"{dataset}:{seeds.OUT_OF_SCOPE}:{rest}"


def normalise_key(key: str) -> str:
    """Map a seed-form key (see :func:`seed_key`) back to the item key.

    Saved prediction files may carry item ids in seed form; other ids are
    returned unchanged.

    Args:
        key: An item key or evaluation id.

    Returns:
        The item key.
    """
    parts = key.split(":")
    if len(parts) == 4 and parts[1] == seeds.OUT_OF_SCOPE:
        return ":".join([parts[0], *parts[2:]])
    return key


def heldout_intents(labels: Sequence[str], n: int = 10, seed: int = 0) -> list[str]:
    """Pick the intents removed from the schema (Banking77).

    Args:
        labels: All intents.
        n: How many to hold out.
        seed: Selection seed.

    Returns:
        Held-out intents, sorted.
    """
    # Sampling, not crypto.
    rng = random.Random(seed)  # noqa: S311  # nosec B311
    return sorted(rng.sample(sorted(labels), n))


def build_items(
    ds: IntentDataset, per_intent: int = 10, n_heldout: int = 10
) -> tuple[list[str], list[ScopeItem]]:
    """Build the schema and the items for one dataset (test split).

    Args:
        ds: The dataset. With an out-of-scope test split (CLINC150) that
            split is used; otherwise intents are held out.
        per_intent: In-scope items per schema intent (stratified selection).
        n_heldout: Intents held out when there is no out-of-scope split.

    Returns:
        ``(schema_labels, items)``.
    """
    main_idx = select_items(ds.test, per_intent)
    if ds.oos_test:
        schema = list(ds.labels)
        items = [
            ScopeItem(
                item_key(ds.name, False, i), ds.test[i].text, False, ds.test[i].label
            )
            for i in main_idx
        ]
        items += [
            ScopeItem(item_key(ds.name, True, i), ex.text, True, None)
            for i, ex in enumerate(ds.oos_test)
        ]
        return schema, items
    held = set(heldout_intents(ds.labels, n_heldout))
    schema = [lab for lab in ds.labels if lab not in held]
    items = [
        ScopeItem(item_key(ds.name, False, i), ds.test[i].text, False, ds.test[i].label)
        for i in main_idx
        if ds.test[i].label not in held
    ]
    items += [
        ScopeItem(item_key(ds.name, True, i), ex.text, True, None)
        for i, ex in enumerate(ds.test)
        if ex.label in held
    ]
    return schema, items


@dataclass(frozen=True)
class ScopeQuestions:
    """The three questions of one item and their code maps.

    Attributes:
        questions: Question dictionaries (wire format).
        closed: Code -> intent of the closed choice.
        with_oos: Code -> intent (or ``out_of_scope``) of the open choice.
    """

    questions: dict[str, Any]
    closed: dict[str, str]
    with_oos: dict[str, str]


def questions_for(item: ScopeItem, schema: Sequence[str]) -> ScopeQuestions:
    """Build the three questions for one item.

    Args:
        item: The item.
        schema: In-scope intents.

    Returns:
        The questions and code maps.
    """
    seed = seed_key(item.key)
    closed = assign_codes(schema, seed, 0)
    with_oos = assign_codes([*schema, OOS_OPTION], seed, 1)
    questions = {
        "closed": {
            "type": "choice",
            "instructions": CLOSED_INSTRUCTIONS,
            "criteria": {c: {"name": lab} for c, lab in closed.items()},
        },
        "with_oos": {
            "type": "choice",
            "instructions": WITH_OOS_INSTRUCTIONS,
            "criteria": {c: {"name": lab} for c, lab in with_oos.items()},
        },
        "fits": {
            "type": "noul",
            "instructions": {"question": FITS_INSTRUCTIONS, "intents": sorted(schema)},
        },
    }
    return ScopeQuestions(questions, closed, with_oos)


def scope_request(
    item: ScopeItem, questions: ScopeQuestions, replicate: int
) -> Request:
    """The request of one item and replicate.

    Args:
        item: The item.
        questions: Its questions.
        replicate: Replicate index.

    Returns:
        The request.
    """
    return Request(state_for(item.text), questions.questions, replicate)


def _mean_dist(
    records: Sequence[CallRecord], name: str, codes: Mapping[str, str]
) -> dict[str, float]:
    """Replicate-mean distribution of one choice question, keyed by intent."""
    acc: dict[str, float] = defaultdict(float)
    for rec in records:
        for c, p in rec.response.choices[name].probabilities.items():
            acc[codes[c]] += p / len(records)
    total = sum(acc.values()) or 1.0
    return {k: v / total for k, v in acc.items()}


def score_item(
    records: Sequence[CallRecord], questions: ScopeQuestions
) -> dict[str, Any]:
    """Average replicates and derive the scores of one item.

    Args:
        records: One answered request per replicate.
        questions: The item's questions.

    Returns:
        Replicate-mean scores and choices.
    """
    a = _mean_dist(records, "closed", questions.closed)
    b = _mean_dist(records, "with_oos", questions.with_oos)
    fits = float(np.mean([rec.response.yes_no["fits"].p_yes for rec in records]))
    pa = np.array(list(a.values()))
    nz = pa[pa > 0]
    return {
        "closed_choice": max(a, key=a.__getitem__),
        "closed_p_max": float(pa.max()),
        "closed_norm_entropy": float(-(nz * np.log(nz)).sum() / np.log(len(pa))),
        "with_oos_choice": max(b, key=b.__getitem__),
        "p_oos": b[OOS_OPTION],
        "p_fits": fits,
        "models": sorted({rec.response.model for rec in records}),
    }
