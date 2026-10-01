"""Evidence sufficiency on real text: how much of the needed evidence is given.

- **HotpotQA** comparison questions (answer yes/no or one of the two
  compared entities) under four cells: ``closed`` (question only) and three
  cells with exactly two paragraphs, so length is constant: ``dose0`` (two
  of the item's non-supporting paragraphs), ``dose1`` (one supporting and
  one non-supporting) and ``dose2`` (both supporting).
- **Quizbowl** questions revealed one sentence at a time (``k`` = first k
  sentences); the four options are the answer and the three most similar
  answers of the same category.

Every request also asks the yes/no question ``enough``: do the paragraphs
or clues given suffice to answer for certain?
"""

import random
from collections.abc import Sequence
from typing import Any

import pandas as pd

from beyond_answer_confidence.data.distractors import DistractorPicker
from beyond_answer_confidence.data.loaders import table_rows
from beyond_answer_confidence.tasks.multiple_choice import LETTERS
from beyond_answer_confidence.tasks.schema import Item

HOTPOT_CELLS = ("closed", "dose0", "dose1", "dose2")
"""HotpotQA evidence cells, from none to all supporting paragraphs."""

QUIZBOWL_POOL_FOLDS = ("guessdev", "guesstest", "buzzdev", "buzztest")
"""QANTA folds whose answers form the distractor pools."""

HOTPOT_INSTRUCTIONS = (
    "Answer the question, using the paragraphs given if they are relevant."
)
HOTPOT_ENOUGH_QUESTION = "Do the paragraphs given contain enough information to answer the question for certain?"
QUIZBOWL_INSTRUCTIONS = "Which answer do the quiz clues describe?"
QUIZBOWL_ENOUGH_QUESTION = "Do the clues given identify the answer for certain?"

QUIZBOWL_OPTIONS = LETTERS[:4]
"""Option keys of the Quizbowl items."""


def _paragraph(title: str, sentences: Sequence[str]) -> str:
    return f"{title}: {''.join(sentences).strip()}"


def hotpot_items(hotpot: pd.DataFrame, seed: int = 0) -> list[Item]:
    """HotpotQA comparison questions under the four evidence cells.

    Questions without exactly two supporting titles, with fewer than two
    other paragraphs, or whose entity answer is not a supporting title are
    skipped.

    Args:
        hotpot: HotpotQA distractor-setting table as returned by
            :func:`beyond_answer_confidence.data.loaders.load_hotpot_validation`.
        seed: Seed for option order, paragraph choice and paragraph order.

    Returns:
        Four items per question (cells in :data:`HOTPOT_CELLS` order), with
        info ``set="hotpot"``, ``qid``, ``cell`` and ``kind`` (``yes_no`` or
        ``entity``).
    """
    df = hotpot[hotpot["type"] == "comparison"]
    items: list[Item] = []
    for row in table_rows(df):
        context = row.context
        titles = list(context["title"])
        paras = {
            t: _paragraph(t, list(s))
            for t, s in zip(titles, context["sentences"], strict=True)
        }
        support = list(dict.fromkeys(row.supporting_facts["title"]))
        others = [t for t in titles if t not in support]
        if len(support) != 2 or len(others) < 2:
            continue
        answer_text = str(row.answer).strip()
        if answer_text.lower() in ("yes", "no"):
            options: tuple[str, ...] = ("yes", "no")
            gold = answer_text.lower()
            kind = "yes_no"
        else:
            match = [t for t in support if t.lower() == answer_text.lower()]
            if not match:
                continue
            options = tuple(support)
            gold = match[0]
            kind = "entity"
        rng = random.Random(f"hotpot:{seed}:{row.id}")  # noqa: S311  # nosec B311
        opts = list(options)
        rng.shuffle(opts)
        pad = rng.sample(others, 2)
        first = rng.choice(support)
        cells: dict[str, list[str]] = {
            "closed": [],
            "dose0": pad,
            "dose1": [first, pad[0]],
            "dose2": list(support),
        }
        for cell, chosen in cells.items():
            order = list(chosen)
            rng.shuffle(order)
            state: dict[str, Any] = {"question": row.question}
            if order:
                state["paragraphs"] = [paras[t] for t in order]
            items.append(
                Item(
                    unit=f"hotpot:{row.id}:{cell}",
                    state=state,
                    options=tuple(opts),
                    gold=gold,
                    instructions=HOTPOT_INSTRUCTIONS,
                    nouls={"enough": HOTPOT_ENOUGH_QUESTION},
                    info={"set": "hotpot", "qid": row.id, "cell": cell, "kind": kind},
                )
            )
    return items


def _sentence_order(row_id: str) -> int:
    return int(str(row_id).rsplit("-", 1)[-1])


def quizbowl_questions(sentences: pd.DataFrame) -> list[dict[str, Any]]:
    """Group QANTA sentence rows into questions.

    Args:
        sentences: One row per sentence (``qanta_id``, ``id`` ending in
            ``-<position>``, ``text``, ``page``, ``category``).

    Returns:
        Per question (sorted by id): ``qid``, ordered ``sentences``,
        ``page`` (the answer) and ``category``.
    """
    questions: list[dict[str, Any]] = []
    for qid, g in sentences.groupby("qanta_id"):
        g = g.assign(order=g["id"].map(_sentence_order)).sort_values("order")
        questions.append(
            {
                "qid": str(qid),
                "sentences": [str(s).strip() for s in g["text"]],
                "page": str(g["page"].iloc[0]),
                "category": str(g["category"].iloc[0]),
            }
        )
    return questions


def quizbowl_items(
    sentences: pd.DataFrame,
    pool: pd.DataFrame,
    pick: DistractorPicker,
    seed: int = 0,
) -> list[Item]:
    """Quizbowl questions revealed one sentence at a time.

    Distractors are picked per category, from the answers (``page``) of
    that category in ``pool``; questions without three are skipped.

    Args:
        sentences: Sentence rows of the questions to ask.
        pool: Sentence rows of every fold (answers and categories are used).
        pick: Distractor picker (see
            :func:`beyond_answer_confidence.data.distractors.distractor_picker`).
        seed: Seed for option order.

    Returns:
        One item per question and prefix length ``k``, with info
        ``set="quizbowl"``, ``qid``, ``k``, ``n_sentences``, ``fraction``
        and ``category``.
    """
    answers_pool = pool.drop_duplicates("qanta_id")[["page", "category"]]
    questions = quizbowl_questions(sentences)
    distractors: dict[str, list[str] | None] = {}
    for cat, qs in pd.DataFrame(questions).groupby("category"):
        cat_pool = sorted(set(answers_pool[answers_pool["category"] == cat]["page"]))
        picked = pick(
            [p.replace("_", " ") for p in qs["page"]],
            [[] for _ in range(len(qs))],
            [p.replace("_", " ") for p in cat_pool],
        )
        distractors.update(zip(qs["qid"], picked, strict=True))
    items: list[Item] = []
    for q in questions:
        dis = distractors.get(q["qid"])
        if dis is None:
            continue
        rng = random.Random(f"qb:{seed}:{q['qid']}")  # noqa: S311  # nosec B311
        answers = [q["page"].replace("_", " "), *dis]
        order = list(range(4))
        rng.shuffle(order)
        desc = {QUIZBOWL_OPTIONS[pos]: answers[i] for pos, i in enumerate(order)}
        gold = QUIZBOWL_OPTIONS[order.index(0)]
        n = len(q["sentences"])
        items.extend(
            Item(
                unit=f"qb:{q['qid']}:{k}",
                state={"clues": " ".join(q["sentences"][:k])},
                options=QUIZBOWL_OPTIONS,
                gold=gold,
                instructions=QUIZBOWL_INSTRUCTIONS,
                descriptions=desc,
                nouls={"enough": QUIZBOWL_ENOUGH_QUESTION},
                info={
                    "set": "quizbowl",
                    "qid": q["qid"],
                    "k": k,
                    "n_sentences": n,
                    "fraction": k / n,
                    "category": q["category"],
                },
            )
            for k in range(1, n + 1)
        )
    return items
