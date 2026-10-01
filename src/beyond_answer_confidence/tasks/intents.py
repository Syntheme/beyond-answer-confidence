"""Opaque intent codes with a controlled dose of knowledge.

Each intent is shown under a meaningless code (``INTENT_07``). The utterance
and its gold intent stay the same across conditions; only what the model is
told about each code changes. The aleatoric part of the uncertainty (how
ambiguous the utterance is) is therefore constant within an item, and any
change in the model's uncertainty across conditions is epistemic by
construction.

Every builder here is pure and deterministic: seeds are derived from stable
string keys, so the same item always yields byte-identical requests (and so
the same cache entries).
"""

import hashlib
import random
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

from beyond_answer_confidence.backends.base import Request
from beyond_answer_confidence.backends.cache import CallRecord
from beyond_answer_confidence.data.intents import (
    Example,
    IntentDataset,
    nested_subset,
    select_items,
)
from beyond_answer_confidence.metrics.distributions import normalise, normalised_entropy

INSTRUCTIONS = (
    "Which intent code does the customer message belong to? "
    "Each code may come with a description; use whatever information is given."
)
QUESTION = "intent"
"""Name of the choice question in every request."""


class Condition(StrEnum):
    """What each intent code comes with."""

    L0 = "L0"
    """Nothing: no knowledge, the ideal output is uniform."""
    L1 = "L1"
    """One example utterance per code."""
    L2 = "L2"
    """Two examples per code."""
    L4 = "L4"
    """Four examples per code."""
    L8 = "L8"
    """Eight examples per code."""
    NAME = "Lname"
    """The original intent name."""
    NAME8 = "Lname+8"
    """Name plus eight examples: maximum knowledge."""
    IRREL = "Lirrel"
    """Irrelevant filler utterances of the same length as eight examples."""
    SWAP = "Lswap"
    """Eight examples of a *different* intent per code (manipulation check)."""


EXAMPLE_COUNTS: dict[Condition, int] = {
    Condition.L0: 0,
    Condition.L1: 1,
    Condition.L2: 2,
    Condition.L4: 4,
    Condition.L8: 8,
    Condition.NAME: 0,
    Condition.NAME8: 8,
    Condition.IRREL: 8,
    Condition.SWAP: 8,
}

MAX_EXAMPLES = max(EXAMPLE_COUNTS.values())


class CodeStyle(StrEnum):
    """How opaque intent codes look."""

    ORDINAL = "ordinal"
    """``INTENT_00`` ... listed in numeric order."""
    LETTERS = "letters"
    """``INTENT_QKTV`` ... random consonant strings, no numeric cue."""


_CONSONANTS = "BCDFGHJKLMNPQRSTVWXZ"
_LETTER_CODE_LEN = 4


@dataclass(frozen=True)
class ChoiceSpec:
    """A choice question for one item under one condition.

    Attributes:
        criteria: Code -> description (``None`` when nothing is given).
        code_to_label: Code -> true intent name, in listing order.
        gold_code: Code of the gold intent.
        swap_code: Under ``SWAP``, the code whose examples belong to the gold
            intent (what an evidence-following model should pick); else
            ``None``.
    """

    criteria: dict[str, object]
    code_to_label: dict[str, str]
    gold_code: str
    swap_code: str | None = None

    def question(self) -> dict[str, object]:
        """Return the question in the wire format.

        Returns:
            A ``type="choice"`` question dictionary.
        """
        return {
            "type": "choice",
            "instructions": INSTRUCTIONS,
            "criteria": self.criteria,
        }


def seeded_rng(*parts: object) -> random.Random:
    """A generator seeded from a stable string key.

    Unlike ``hash()``, the seed is the same in every process.

    Args:
        *parts: Key parts, joined with ``|``.

    Returns:
        A seeded generator.
    """
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    # Seeding experimental conditions, not cryptography.
    return random.Random(int.from_bytes(digest[:8], "big"))  # noqa: S311  # nosec B311


def sample_examples(dataset: IntentDataset, seed: int) -> dict[str, list[str]]:
    """Pick ``MAX_EXAMPLES`` pool examples per intent, in a fixed order.

    Conditions with k examples use the first k, so the example sets are
    nested (L1 in L2 in L4 in L8) and the dose-response is not confounded by
    which examples were drawn.

    Args:
        dataset: The dataset whose pool to sample.
        seed: Example-set seed.

    Returns:
        Example texts keyed by intent.

    Raises:
        ValueError: If an intent has fewer than ``MAX_EXAMPLES`` pool examples.
    """
    by_label = dataset.examples_by_label()
    chosen: dict[str, list[str]] = {}
    for label in dataset.labels:
        texts = [ex.text for ex in by_label.get(label, [])]
        if len(texts) < MAX_EXAMPLES:
            raise ValueError(f"intent {label!r} has only {len(texts)} pool examples")
        chosen[label] = seeded_rng("examples", dataset.name, seed, label).sample(
            texts, MAX_EXAMPLES
        )
    return chosen


def assign_codes(
    labels: Sequence[str],
    item_key: str,
    perm_seed: int,
    style: CodeStyle = CodeStyle.ORDINAL,
) -> dict[str, str]:
    """Randomly map opaque codes to intents for one item.

    The intent order is shuffled per item either way. Ordinal codes are
    zero-padded and listed in numeric order; letter codes are random,
    distinct consonant strings (no vowels, so no accidental words) listed in
    the same shuffled order.

    Args:
        labels: Intent names.
        item_key: Stable identifier of the item.
        perm_seed: Permutation seed (vary it to test order invariance).
        style: Code style.

    Returns:
        Code -> intent, in listing order.
    """
    shuffled = list(labels)
    seeded_rng("codes", item_key, perm_seed).shuffle(shuffled)
    if style is CodeStyle.ORDINAL:
        width = len(str(len(labels) - 1))
        return {f"INTENT_{i:0{width}d}": label for i, label in enumerate(shuffled)}
    rng = seeded_rng("letters", item_key, perm_seed)
    codes: list[str] = []
    seen: set[str] = set()
    while len(codes) < len(shuffled):
        code = "INTENT_" + "".join(rng.choices(_CONSONANTS, k=_LETTER_CODE_LEN))
        if code not in seen:
            seen.add(code)
            codes.append(code)
    return dict(zip(codes, shuffled, strict=True))


def derangement(labels: Sequence[str], rng: random.Random) -> dict[str, str]:
    """Map every label to a different label (a permutation without fixed points).

    Args:
        labels: At least two labels.
        rng: Random generator (rejection sampling).

    Returns:
        Label -> another label.
    """
    while True:
        shuffled = list(labels)
        rng.shuffle(shuffled)
        if all(a != b for a, b in zip(labels, shuffled, strict=True)):
            return dict(zip(labels, shuffled, strict=True))


def filler_notes(
    filler: Sequence[str], target_chars: int, rng: random.Random
) -> list[str]:
    """Draw filler utterances (without replacement) until ``target_chars`` is reached.

    Matching characters rather than sentence count keeps the length control
    honest when filler sentences are shorter than the real examples.

    Args:
        filler: Candidate utterances.
        target_chars: Characters to reach.
        rng: Random generator.

    Returns:
        The drawn utterances.

    Raises:
        ValueError: If all the filler together is too short.
    """
    order = list(filler)
    rng.shuffle(order)
    notes: list[str] = []
    total = 0
    for text in order:
        if total >= target_chars:
            break
        notes.append(text)
        total += len(text)
    if total < target_chars:
        raise ValueError("not enough filler text to match the example length")
    return notes


def _description(
    condition: Condition,
    label: str,
    code: str,
    examples: Mapping[str, list[str]],
    filler: Sequence[str],
    source: Mapping[str, str],
    seed_key: tuple[str, int],
) -> object:
    """What one code comes with under one condition."""
    k = EXAMPLE_COUNTS[condition]
    if condition is Condition.L0:
        return None
    if condition is Condition.NAME:
        return {"name": label}
    if condition is Condition.NAME8:
        return {"name": label, "examples": examples[label][:k]}
    if condition is Condition.IRREL:
        target = sum(len(t) for t in examples[label][:k])
        return {
            "notes": filler_notes(filler, target, seeded_rng("filler", *seed_key, code))
        }
    if condition is Condition.SWAP:
        return {"examples": examples[source[label]][:k]}
    return {"examples": examples[label][:k]}


def build_choice(
    item: Example,
    item_key: str,
    dataset: IntentDataset,
    condition: Condition,
    examples: Mapping[str, list[str]],
    filler: Sequence[str],
    perm_seed: int = 0,
    code_style: CodeStyle = CodeStyle.ORDINAL,
) -> ChoiceSpec:
    """Build the choice question for one item under one condition.

    Args:
        item: The utterance and gold intent.
        item_key: Stable item identifier (seeds the code assignment).
        dataset: The dataset (for its label set).
        condition: Knowledge condition.
        examples: Output of :func:`sample_examples`.
        filler: Utterances unrelated to the intents, for ``IRREL``.
        perm_seed: Code-assignment seed.
        code_style: Ordinal or random-letter codes.

    Returns:
        The question and the bookkeeping needed to score it.

    Raises:
        ValueError: If the gold label is not in the dataset's label set, or
            ``IRREL`` is requested with too little filler.
    """
    if item.label not in dataset.labels:
        raise ValueError(f"gold label {item.label!r} not in label set")
    code_to_label = assign_codes(dataset.labels, item_key, perm_seed, code_style)
    label_to_code = {label: code for code, label in code_to_label.items()}

    swap_code: str | None = None
    source: dict[str, str] = {}
    if condition is Condition.SWAP:
        # The code of intent X is shown the examples of intent source[X].
        source = derangement(dataset.labels, seeded_rng("swap", item_key, perm_seed))
        swap_code = next(
            c for c, lab in code_to_label.items() if source[lab] == item.label
        )
    criteria: dict[str, object] = {
        code: _description(
            condition, label, code, examples, filler, source, (item_key, perm_seed)
        )
        for code, label in code_to_label.items()
    }
    return ChoiceSpec(criteria, code_to_label, label_to_code[item.label], swap_code)


def state_for(text: str) -> dict[str, str]:
    """Return the request state for an utterance.

    Args:
        text: The customer message.

    Returns:
        The state dictionary.
    """
    return {"customer_message": text}


def choice_request(item: Example, spec: ChoiceSpec, replicate: int = 0) -> Request:
    """The single-question request for one item and spec.

    Args:
        item: The utterance.
        spec: The choice question.
        replicate: Replicate index.

    Returns:
        The request.
    """
    return Request(state_for(item.text), {QUESTION: spec.question()}, replicate)


def label_probs(
    spec: ChoiceSpec, probabilities: Mapping[str, float]
) -> dict[str, float]:
    """Map a choice answer's code probabilities back to intents (normalised).

    Args:
        spec: The question's bookkeeping.
        probabilities: Code -> probability, as answered.

    Returns:
        Intent -> probability, summing to 1.

    Raises:
        ValueError: If the answer's options differ from the question's codes.
    """
    if probabilities.keys() != spec.code_to_label.keys():
        raise ValueError("answer options do not match question codes")
    return normalise({spec.code_to_label[c]: p for c, p in probabilities.items()})


# --- The knowledge-dial design -------------------------------------------------

EXAMPLE_DEPENDENT = (
    Condition.L1,
    Condition.L2,
    Condition.L4,
    Condition.L8,
    Condition.NAME8,
    Condition.IRREL,
    Condition.SWAP,
)
"""Conditions whose content depends on which examples were drawn."""
LETTER_CONDITIONS = (Condition.L0, Condition.NAME, Condition.L8)
"""Conditions repeated with random-letter codes."""


@dataclass(frozen=True)
class Cell:
    """One planned knowledge-dial request.

    Attributes:
        arm: ``"main"`` (all conditions, ordinal codes, example seed 0),
            ``"seeds"`` (other example sets) or ``"letters"`` (letter codes).
        item: Index into the split.
        condition: Knowledge condition.
        code_style: Code style.
        example_seed: Example-set seed.
        replicate: Replicate index.
    """

    arm: str
    item: int
    condition: Condition
    code_style: CodeStyle
    example_seed: int
    replicate: int


@dataclass(frozen=True)
class Planned:
    """A planned request with the spec needed to score it.

    Attributes:
        cell: The cell.
        spec: The choice question.
        request: The request.
    """

    cell: Cell
    spec: ChoiceSpec
    request: Request


def plan_cells(
    examples: Sequence[Example],
    *,
    per_intent: int,
    subset_per_intent: int,
    replicates: int,
    extra_seeds: Sequence[int] = (1, 2),
) -> list[Cell]:
    """List every knowledge-dial cell (cheap bookkeeping, no request bodies).

    Args:
        examples: The split's examples.
        per_intent: Main-arm items per intent.
        subset_per_intent: Seed- and letter-arm items per intent (a nested
            subset of the main-arm items).
        replicates: Replicates for the main and letter arms.
        extra_seeds: Example-set seeds of the seed arm (one replicate each).

    Returns:
        Cells in a fixed order.
    """
    main_items = select_items(examples, per_intent)
    subset = nested_subset(examples, main_items, subset_per_intent)
    cells: list[Cell] = []
    for item in main_items:
        for cond in Condition:
            cells += [
                Cell("main", item, cond, CodeStyle.ORDINAL, 0, r)
                for r in range(replicates)
            ]
    for item in subset:
        for seed in extra_seeds:
            cells += [
                Cell("seeds", item, cond, CodeStyle.ORDINAL, seed, 0)
                for cond in EXAMPLE_DEPENDENT
            ]
        for cond in LETTER_CONDITIONS:
            cells += [
                Cell("letters", item, cond, CodeStyle.LETTERS, 0, r)
                for r in range(replicates)
            ]
    return cells


class Planner:
    """Turns knowledge-dial cells into requests for one dataset and split."""

    def __init__(
        self,
        dataset: IntentDataset,
        split: str,
        filler: Sequence[str],
        example_seeds: Sequence[int] = (0, 1, 2),
    ) -> None:
        """Create the planner.

        Args:
            dataset: The dataset.
            split: ``"dev"`` or ``"test"``.
            filler: Irrelevant utterances for the length control.
            example_seeds: Example-set seeds that cells may use.
        """
        self.dataset = dataset
        self.split = split
        self.filler = filler
        self.examples = dataset.split(split)
        self.example_sets = {s: sample_examples(dataset, s) for s in example_seeds}

    def item_key(self, item: int) -> str:
        """Stable key of an item (seeds its code assignment).

        Args:
            item: Index into the split.

        Returns:
            ``<dataset>:<split>:<index>``.
        """
        return f"{self.dataset.name}:{self.split}:{item}"

    def planned(self, cell: Cell) -> Planned:
        """Build the request for one cell.

        Args:
            cell: The cell.

        Returns:
            The planned request.
        """
        ex = self.examples[cell.item]
        spec = build_choice(
            ex,
            self.item_key(cell.item),
            self.dataset,
            cell.condition,
            self.example_sets[cell.example_seed],
            self.filler,
            code_style=cell.code_style,
        )
        return Planned(cell, spec, choice_request(ex, spec, cell.replicate))


def dial_row(planned: Planned, record: CallRecord) -> dict[str, Any]:
    """Flatten one answered knowledge-dial request into a row.

    Args:
        planned: The planned request.
        record: Its answer.

    Returns:
        A JSON-serialisable row with probabilities mapped back to intents.
    """
    answer = record.response.choices[QUESTION]
    spec = planned.spec
    codes = list(spec.code_to_label)
    probs = {spec.code_to_label[c]: answer.probabilities[c] for c in codes}
    return {
        **asdict(planned.cell),
        "condition": planned.cell.condition.value,
        "code_style": planned.cell.code_style.value,
        "K": len(codes),
        "gold": spec.code_to_label[spec.gold_code],
        "swap_target": spec.code_to_label[spec.swap_code] if spec.swap_code else None,
        "choice": spec.code_to_label[answer.choice] if answer.choice else None,
        "first_listed": spec.code_to_label[codes[0]],
        "p_gold": answer.probabilities[spec.gold_code],
        "p_max": max(answer.probabilities.values()),
        "norm_entropy": normalised_entropy(list(answer.probabilities.values())),
        "probs": probs,
        "model": record.response.model,
        "cache_key": record.key,
    }
