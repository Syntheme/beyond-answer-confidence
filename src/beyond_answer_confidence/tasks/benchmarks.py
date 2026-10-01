"""Items from public uncertainty benchmarks.

- **SelfAware** (answerable vs unanswerable): yes/no ``answerable``.
- **SimpleQA Verified**: four-option choice; wrong options change the
  year or number of date and number answers, else are the most similar
  answers of the same answer type. Yes/no ``known``.
- **TruthfulQA**: binary (best answer vs best incorrect answer) and MC1
  (all mc1 options). Yes/no ``known``.
- **TriviaQA** (no context): four-option choice with the most similar
  other answers as distractors. Yes/no ``known``.
- **AmbigQA**: yes/no ``single`` (does the question have a single clear
  answer?).
- **ChaosNLI** (SNLI and MNLI: entailment / neutral / contradiction;
  alphaNLI: two explanations), each with 100 human labels; the model's
  distribution is compared with the human label distribution.
"""

import random
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from beyond_answer_confidence.data.distractors import DistractorPicker, perturb_answer
from beyond_answer_confidence.data.loaders import table_rows
from beyond_answer_confidence.tasks.multiple_choice import LETTERS, multiple_choice_item
from beyond_answer_confidence.tasks.schema import Item

ANSWERABLE_QUESTION = "Can this question be answered with a definite, factual answer?"
"""Yes/no question for SelfAware."""

SINGLE_QUESTION = (
    "Does this question have a single clear answer, without needing to be made "
    "more specific?"
)
"""Yes/no question for AmbigQA."""

NLI_INSTRUCTIONS = "What is the relationship between the premise and the hypothesis?"
NLI_OPTIONS = {
    "entailment": "the hypothesis is definitely true given the premise",
    "neutral": "the hypothesis might or might not be true given the premise",
    "contradiction": "the hypothesis is definitely false given the premise",
}
"""Three-way NLI options and their descriptions."""

ABDUCTIVE_INSTRUCTIONS = (
    "Which hypothesis better explains what happened between the two observations?"
)
"""Choice instructions for alphaNLI (abductive NLI)."""

_NLI_LABEL = {"e": "entailment", "n": "neutral", "c": "contradiction"}


def selfaware_items(selfaware: pd.DataFrame) -> list[Item]:
    """SelfAware answerable / unanswerable questions.

    Args:
        selfaware: Table as returned by
            :func:`beyond_answer_confidence.data.loaders.load_selfaware`.

    Returns:
        Items with info ``set="selfaware"``, ``answerable`` and ``source``.
    """
    return [
        Item(
            unit=f"selfaware:{r.question_id}",
            state={"question": r.question},
            nouls={"answerable": ANSWERABLE_QUESTION},
            info={
                "set": "selfaware",
                "answerable": bool(r.answerable),
                "source": r.source,
            },
        )
        for r in table_rows(selfaware)
    ]


def simpleqa_items(
    simpleqa: pd.DataFrame, pick: DistractorPicker, seed: int = 0
) -> list[Item]:
    """SimpleQA Verified as four-option multiple choice.

    Date and number answers get perturbed variants as distractors when
    possible; other answers (and failed perturbations) get the nearest
    answers of the same answer type.

    Args:
        simpleqa: Table as returned by
            :func:`beyond_answer_confidence.data.loaders.load_simpleqa`.
        pick: Distractor picker.
        seed: Seed for perturbations and option order.

    Returns:
        Items with info ``set="simpleqa"``, ``topic``, ``answer_type`` and
        ``perturbed``.
    """
    distractors: dict[Any, list[str] | None] = {}
    perturbed: dict[Any, bool] = {}
    for atype, g in simpleqa.groupby("answer_type"):
        near = pick(list(g["answer"]), [[] for _ in range(len(g))], list(g["answer"]))
        for idx, row, d in zip(g.index, table_rows(g), near, strict=True):
            pert = None
            if atype in ("Date", "Number"):
                rng = random.Random(f"sqa-pert:{seed}:{row.original_index}")  # noqa: S311  # nosec B311
                pert = perturb_answer(str(row.answer), rng)
            distractors[idx] = pert if pert is not None else d
            perturbed[idx] = pert is not None
    items = []
    for idx, row in simpleqa.iterrows():
        dis = distractors[idx]
        if dis is None:
            continue
        rng = random.Random(f"sqa:{seed}:{row['original_index']}")  # noqa: S311  # nosec B311
        items.append(
            multiple_choice_item(
                f"simpleqa:{row['original_index']}",
                row["problem"],
                [row["answer"], *dis],
                0,
                rng,
                {
                    "set": "simpleqa",
                    "topic": row["topic"],
                    "answer_type": row["answer_type"],
                    "perturbed": perturbed[idx],
                },
            )
        )
    return items


def truthfulqa_items(
    binary: pd.DataFrame, mc1: pd.DataFrame, seed: int = 0
) -> list[Item]:
    """TruthfulQA binary (official CSV) and MC1 items.

    Args:
        binary: Table as returned by
            :func:`beyond_answer_confidence.data.loaders.load_truthfulqa_binary`.
        mc1: Table as returned by
            :func:`beyond_answer_confidence.data.loaders.load_truthfulqa_mc1`.
        seed: Seed for option order.

    Returns:
        Binary items (info ``set="truthfulqa_binary"``, ``category``,
        ``type``), then MC1 items with at most 16 options (info
        ``set="truthfulqa_mc1"``, ``k``).
    """
    items = []
    for i, row in binary.iterrows():
        rng = random.Random(f"tqa2:{seed}:{i}")  # noqa: S311  # nosec B311
        items.append(
            multiple_choice_item(
                f"tqa_binary:{i}",
                row["Question"],
                [row["Best Answer"], row["Best Incorrect Answer"]],
                0,
                rng,
                {
                    "set": "truthfulqa_binary",
                    "category": row["Category"],
                    "type": row["Type"],
                },
            )
        )
    for i, row in mc1.iterrows():
        choices = list(row["mc1_targets"]["choices"])
        labels = list(row["mc1_targets"]["labels"])
        gold = labels.index(1)
        answers = [choices[gold]] + [c for j, c in enumerate(choices) if j != gold]
        if len(answers) > len(LETTERS):
            continue
        rng = random.Random(f"tqa1:{seed}:{i}")  # noqa: S311  # nosec B311
        items.append(
            multiple_choice_item(
                f"tqa_mc1:{i}",
                row["question"],
                answers,
                0,
                rng,
                {"set": "truthfulqa_mc1", "k": len(answers)},
            )
        )
    return items


def display_case(answer: str) -> str:
    """Normalise answer casing so it cannot give the gold away.

    About a quarter of TriviaQA gold values are all capitals, while
    distractors drawn from other items often are not. All-caps strings
    longer than four characters are title-cased (short acronyms are kept)
    and trailing full stops removed, for every option alike.

    Args:
        answer: Raw answer string.

    Returns:
        Display string.
    """
    a = answer.strip().rstrip(".")
    return a.title() if a.isupper() and len(a) > 4 else a


def triviaqa_items(
    triviaqa: pd.DataFrame, pick: DistractorPicker, seed: int = 0
) -> list[Item]:
    """TriviaQA (no context) as four-option multiple choice.

    Distractors are picked over all rows (including repeated questions, so
    the saved nearest-distractor choice stays valid); then the first row
    per question is kept.

    Args:
        triviaqa: Table as returned by
            :func:`beyond_answer_confidence.data.loaders.load_triviaqa_validation`.
        pick: Distractor picker.
        seed: Seed for option order.

    Returns:
        Items with info ``set="triviaqa"`` and ``answer_type``.
    """
    golds = [str(a["value"]) for a in triviaqa["answer"]]
    aliases = [list(a["aliases"]) for a in triviaqa["answer"]]
    near = pick(golds, aliases, golds)
    seen: set[str] = set()
    items = []
    for row, gold, dis in zip(table_rows(triviaqa), golds, near, strict=True):
        if dis is None or row.question_id in seen:
            continue
        seen.add(str(row.question_id))
        rng = random.Random(f"tqa:{seed}:{row.question_id}")  # noqa: S311  # nosec B311
        answer = row.answer
        items.append(
            multiple_choice_item(
                f"triviaqa:{row.question_id}",
                str(row.question),
                [display_case(a) for a in (gold, *dis)],
                0,
                rng,
                {"set": "triviaqa", "answer_type": answer["type"]},
            )
        )
    return items


def ambig_label(types: Sequence[str]) -> str:
    """Classify an AmbigQA item by its annotators' judgements.

    Args:
        types: One ``singleAnswer`` / ``multipleQAs`` per annotator.

    Returns:
        ``"single"``, ``"ambiguous"`` or ``"mixed"``.
    """
    kinds = set(types)
    if kinds == {"singleAnswer"}:
        return "single"
    if kinds == {"multipleQAs"}:
        return "ambiguous"
    return "mixed"


def ambigqa_items(ambigqa: pd.DataFrame) -> list[Item]:
    """AmbigQA questions with the ``single`` yes/no question.

    Args:
        ambigqa: Table as returned by
            :func:`beyond_answer_confidence.data.loaders.load_ambigqa_validation`.

    Returns:
        Items with info ``set="ambigqa"`` and ``label`` (see
        :func:`ambig_label`).
    """
    items = []
    for r in table_rows(ambigqa):
        annotations = r.annotations
        items.append(
            Item(
                unit=f"ambigqa:{r.id}",
                state={"question": r.question},
                nouls={"single": SINGLE_QUESTION},
                info={
                    "set": "ambigqa",
                    "label": ambig_label(list(annotations["type"])),
                },
            )
        )
    return items


def _label_distribution(counts: Sequence[int], keys: Sequence[str]) -> dict[str, float]:
    total = sum(counts)
    return dict(zip(keys, (c / total for c in counts), strict=True))


def chaos_items(nli: Mapping[str, pd.DataFrame], abductive: pd.DataFrame) -> list[Item]:
    """ChaosNLI items with the human label distribution attached.

    Args:
        nli: Three-way NLI tables by subset name (``"snli"``, ``"mnli_m"``;
            iterated in the given order), as returned by
            :func:`beyond_answer_confidence.data.loaders.load_chaosnli`.
        abductive: The alphaNLI table.

    Returns:
        Items with info ``set`` (``chaos_nli`` or ``chaos_anli``),
        ``subset``, ``human`` (label distribution of 100 annotators),
        ``human_entropy`` and ``old_dist`` (the original annotators' label
        distribution, where given).
    """
    items = []
    for subset, df in nli.items():
        for r in table_rows(df):
            human = _label_distribution(
                [int(c) for c in r.label_count], list(NLI_OPTIONS)
            )
            old_labels = r.old_labels
            old = [x for x in (old_labels or []) if x in NLI_OPTIONS]
            old_dist = (
                {k: old.count(k) / len(old) for k in NLI_OPTIONS} if old else None
            )
            example = r.example
            majority = r.majority_label
            items.append(
                Item(
                    unit=f"chaos_{subset}:{r.uid}",
                    state={
                        "premise": example["premise"],
                        "hypothesis": example["hypothesis"],
                    },
                    options=tuple(NLI_OPTIONS),
                    gold=_NLI_LABEL.get(majority, majority),
                    instructions=NLI_INSTRUCTIONS,
                    descriptions=NLI_OPTIONS,
                    info={
                        "set": "chaos_nli",
                        "subset": subset,
                        "human": human,
                        "human_entropy": float(r.entropy),
                        "old_dist": old_dist,
                    },
                )
            )
    for r in table_rows(abductive):
        counts = [int(c) for c in r.label_count]
        ex = r.example
        items.append(
            Item(
                unit=f"chaos_anli:{r.uid}",
                state={"observation_1": ex["obs1"], "observation_2": ex["obs2"]},
                options=("1", "2"),
                gold=str(r.majority_label),
                instructions=ABDUCTIVE_INSTRUCTIONS,
                descriptions={"1": ex["hyp1"], "2": ex["hyp2"]},
                info={
                    "set": "chaos_anli",
                    "subset": "alphanli",
                    "human": {
                        "1": counts[0] / sum(counts),
                        "2": counts[1] / sum(counts),
                    },
                    "human_entropy": float(r.entropy),
                    "old_dist": None,
                },
            )
        )
    return items
