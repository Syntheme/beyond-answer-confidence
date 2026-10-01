"""Shortcut-control builders, date masking and the per-part analyses."""

import math
import random

import numpy as np
import pandas as pd
import pytest

from beyond_answer_confidence.experiments import shortcut_controls as sc
from beyond_answer_confidence.tasks import shortcuts
from beyond_answer_confidence.tasks.chance import DEVICES
from beyond_answer_confidence.tasks.follow_up import CORRECT_QUESTION
from beyond_answer_confidence.tasks.schema import Item


# Synthetic questions built from made-up words; each exercises one date form.
@pytest.mark.parametrize(
    ("question", "masked"),
    [
        (
            "Will the Zorvik guild trim its blorp dues by the end of March 2024?",
            "Will the Zorvik guild trim its blorp dues?",
        ),
        (
            "Will a flurm cloud drop grells on Wibbleton on Wednesday, December 11, 2024?",
            "Will a flurm cloud drop grells on Wibbleton?",
        ),
        (
            "Will the Plonk troupe stage its 900th snorfle in spring 2020?",
            "Will the Plonk troupe stage its 900th snorfle?",
        ),
        (
            "Will the Quexa gala be held on the announced dates of Sept. 21 to 23?",
            "Will the Quexa gala be held on the announced dates?",
        ),
        (
            "Will the Drabble index rise on October 1, 2025 after the frim report?",
            "Will the Drabble index rise after the frim report?",
        ),
        (
            # the one month-led name the pattern exempts
            "Will the March of Dimes open a Zorvik bureau?",
            "Will the March of Dimes open a Zorvik bureau?",
        ),
        (
            "Will the Zorvik warden say it may snizzle?",
            "Will the Zorvik warden say it may snizzle?",
        ),
        (
            "Will the Zorvik mill reopen in early 2025, as planned?",
            "Will the Zorvik mill reopen, as planned?",
        ),
    ],
)
def test_mask_dates(question: str, masked: str) -> None:
    assert shortcuts.mask_dates(question) == masked


def test_latest_date() -> None:
    assert shortcuts.latest_date(
        "Will X happen by March 2024 or in 2023?"
    ) == pytest.approx(2024 + 2 / 12)
    assert math.isnan(shortcuts.latest_date("Will X happen soon?"))


def test_month_after_end() -> None:
    assert shortcuts.month_after_end("2024-12") == "2025-01-31"
    assert shortcuts.month_after_end("2024-01") == "2024-02-29"


@pytest.mark.parametrize(
    ("question", "month", "expected"),
    [
        ("Will X stage its snorfle in spring 2020?", "2020-01", "2020-12"),
        ("Will the blorp dues rise by the end of March 2024?", "2024-02", "2024-03"),
        ("Will X happen?", "2025-03", "2025-03"),
        ("Did X happen in January 2021?", "2021-05", "2021-05"),
    ],
)
def test_settled_by(question: str, month: str, expected: str) -> None:
    assert shortcuts.settled_by(question, month) == expected


@pytest.mark.parametrize(
    ("name", "ok"),
    [
        ("Brellon Quask", True),
        ("Ama-Lise Vorp", True),
        ("Zorvik Snorfle League", False),
        ("Grell of Wibbleton", False),
        ("Quaskel", False),
        ("Now Field", False),
    ],
)
def test_is_person_name(name: str, ok: bool) -> None:
    assert shortcuts.is_person_name(name) is ok


def test_fact_sentences_are_capitalised() -> None:
    assert shortcuts.fact_sentence("author", "the Logor Archive", "Jane Roe") == (
        "The Logor Archive was written by Jane Roe."
    )
    assert set(shortcuts.FACT_TEMPLATES) >= {"occupation", "sport", "capital of"}


def _people() -> pd.DataFrame:
    first = ["Alda", "Bren", "Corin", "Dace", "Elow", "Fenn", "Garo", "Hesta"]
    last = ["Ashvale", "Brindle", "Corrow", "Dunmere", "Elstow", "Farrow"]
    rows = []
    for r, prop in enumerate(shortcuts.PERSON_RELATION_ORDER):
        for k in range(10):
            subj = f"{first[(k + r) % 8]} {last[(k * 5 + r) % 6]}"
            rows.append(
                {
                    "subj": subj,
                    "prop": prop,
                    "obj": f"{prop} {k % 5}",
                    "question": f"What is the {prop} of {subj}?",
                    "s_pop": k + 1,
                }
            )
    return pd.DataFrame(rows)


def test_lookalike_subjects_recombine_obscure_names() -> None:
    people = _people()
    picks = shortcuts.lookalike_subjects(people, 2, random.Random(0))  # noqa: S311
    assert list(picks) == list(shortcuts.PERSON_RELATION_ORDER)
    names = [n for v in picks.values() for n, _, _ in v]
    assert len(set(names)) == len(names) == 12
    existing = set(people["subj"])
    for name, a, b in (t for v in picks.values() for t in v):
        assert name not in existing
        assert name.split()[0] == a.split()[0] != b.split()[0]
        assert name.split()[1:] == b.split()[1:]


def test_lookalike_items_use_the_relation_template() -> None:
    items = shortcuts.lookalike_items(_people(), 2)
    assert len(items) == 12
    first = items[0]
    assert first.unit == "lookalike:occupation:0"
    assert (
        first.state["question"] == f"What is the occupation of {first.info['subject']}?"
    )
    assert first.options == ("A", "B", "C", "D")
    assert first.gold is None
    assert set(first.nouls) == {"known"}
    assert shortcuts.lookalike_items(_people(), 2) == items  # seeded


def _fabricated() -> list[Item]:
    return [
        Item(
            unit=f"fab:{prop}:{j}",
            state={"question": f"Which {prop} is Zorbel {j}?"},
            options=("A", "B"),
            instructions="Choose.",
            descriptions={"A": f"{prop} one", "B": f"{prop} two"},
            nouls={"known": "Known?"},
            info={"prop": prop, "subject": f"Zorbel {j}"},
        )
        for prop in ("sport", "genre")
        for j in range(4)
    ]


def test_evidence_items_pair_a_stated_and_an_unrelated_sentence() -> None:
    items = shortcuts.evidence_items(_fabricated(), 2)
    assert len(items) == 8
    stated, unrelated = items[0], items[1]
    assert stated.info["arm"] == "stated"
    assert unrelated.info["arm"] == "unrelated"
    assert stated.info["base"] == unrelated.info["base"]
    assert stated.gold in stated.options
    assert unrelated.gold is None
    sentence = stated.state["paragraphs"][0]
    assert stated.descriptions is not None
    assert str(stated.descriptions[stated.gold]) in sentence
    assert unrelated.state["paragraphs"][0] != sentence


def test_evidence_items_need_descriptions() -> None:
    bare = [Item(unit="u", state={}, options=("A",), info={"prop": "sport"})]
    with pytest.raises(ValueError, match="descriptions"):
        shortcuts.evidence_items(bare, 1)


def _news() -> list[Item]:
    return [
        Item(
            unit=f"news:{m}:{i}",
            state={"question": f"Will the Zorvik gala open by March {2023 + m // 12}?"},
            options=("yes", "no"),
            gold="yes" if i else "no",
            instructions="Choose.",
            nouls={"known": "Known?"},
            info={"month": f"{2023 + m // 12}-{m % 12 + 1:02d}"},
        )
        for m in range(18)
        for i in range(2)
    ]


def test_news_subset_is_balanced_and_ordered() -> None:
    news = _news()
    subset = shortcuts.news_subset(news, "2024-01", 0)
    post = [it for it in subset if it.info["month"] >= "2024-01"]
    assert len(subset) == 2 * len(post) == 24
    order = [it.unit for it in news]
    assert [it.unit for it in subset] == sorted(
        (it.unit for it in subset), key=order.index
    )


def test_news_builders() -> None:
    subset = shortcuts.news_subset(_news(), "2024-01", 0)
    masked = shortcuts.masked_items(subset)
    assert masked[0].state == {"question": "Will the Zorvik gala open?"}
    assert masked[0].info["base"] == subset[0].unit
    first = {it.unit: {"top": "yes" if k % 2 else "no"} for k, it in enumerate(subset)}
    pol = shortcuts.polarity_items(subset, first, "2024-01")
    assert all(p.state["proposed_answer"] != first[p.info["base"]]["top"] for p in pol)
    assert pol[0].nouls == {"correct": CORRECT_QUESTION}
    assert pol[0].options == ()
    neg = shortcuts.negated_items(subset, "2024-01")
    by_base: dict[str, set[str]] = {}
    for it in neg:
        by_base.setdefault(str(it.info["base"]), set()).add(str(it.info["proposed"]))
    assert all(v == {"yes", "no"} for v in by_base.values())
    assert neg[0].nouls == {"wrong": shortcuts.WRONG_QUESTION}
    near = shortcuts.near_day_items(subset, "2024-02-15")
    assert all(it.state["today"] <= "2024-02-15" for it in near)
    post = [it for it in near if it.info["month"] >= "2024-01"]
    assert post
    assert all(it.state["today"] == "2024-02-15" for it in post)  # capped
    for it, base in zip(near, subset, strict=True):
        month = str(base.info["month"])
        expected = "2023-04-30" if month <= "2023-03" else None  # "by March 2023"
        if expected is not None:
            assert it.state["today"] == expected
    assert near[0].state["question"] == subset[0].state["question"]


def test_semantics_items_cover_every_wording_and_order() -> None:
    items = shortcuts.semantics_items(3)
    assert len(items) == len(shortcuts.SEMANTIC_DEVICES) * 4 * 3
    first = next(it for it in items if it.info["wording"] == "occur")
    assert first.state["question"] == DEVICES[str(first.info["device"])][0]
    assert len({it.unit for it in items}) == len(items)
    tasks = sc.part_tasks({"semantics": items}, 3)
    assert len(tasks) == len(items)
    assert {t.request.replicate for t in tasks} == {0, 1, 2}


def test_analyse_semantics() -> None:
    rows = [
        {
            "device": "die_loaded",
            "wording": "most_likely",
            "dist": {"one": 0, "two": 0, "three": 0, "four": 0, "five": 0, "six": 1.0},
        }
    ]
    res = sc.analyse_semantics(rows)["die_loaded"]["most_likely"]
    assert res["p_on_mode"] == 1.0
    assert res["tv_to_stated"] == pytest.approx(0.5)
    assert res["top_label"] == "six"


def test_analyse_polarity_counts_coherence() -> None:
    rows = [
        {
            "base": "a",
            "p_correct": 0.2,
            "own": "yes",
            "post": True,
            "gold_answer": "no",
        },
        {
            "base": "b",
            "p_correct": 0.3,
            "own": "no",
            "post": False,
            "gold_answer": "no",
        },
    ]
    own = {"a": {"p_correct": 0.4}, "b": {"p_correct": 0.9}}
    res = sc.analyse_polarity(rows, own, 10, np.random.default_rng(0))
    assert res["post"]["share_rejecting_both"] == 1.0  # 0.4 and 0.2 both < 0.5
    assert res["post"]["implied_answer_yes_share"] == 1.0
    assert res["post"]["implied_answer_accuracy"] == 0.0
    assert res["pre"]["mean_p_correct_when_no_proposed"] == 0.9
    assert res["pre"]["implied_answer_accuracy"] == 1.0
    lo, hi = res["sum_minus_one_ci"]
    assert -0.4 <= lo <= hi <= 0.2


def test_analyse_negated_implied_answers() -> None:
    rows = [
        {"base": "a", "proposed": "yes", "p_wrong": 0.8},
        {"base": "a", "proposed": "no", "p_wrong": 0.1},
        {"base": "b", "proposed": "yes", "p_wrong": 0.2},
        {"base": "b", "proposed": "no", "p_wrong": 0.9},
    ]
    own = {
        "a": {"proposed": "no", "p_correct": 0.9},
        "b": {"proposed": "yes", "p_correct": 0.7},
    }
    opp = {
        "a": {"proposed": "yes", "p_correct": 0.1, "post": True, "gold_answer": "no"},
        "b": {"proposed": "no", "p_correct": 0.2, "post": True, "gold_answer": "yes"},
    }
    res = sc.analyse_negated(rows, own, opp, 10, np.random.default_rng(0))
    post = res["post"]
    assert post["n"] == 2
    assert post["implied_answers_agree"] == 1.0
    assert post["implied_accuracy_negative"] == 1.0
    assert post["coherence_yes_proposed"] == pytest.approx((0.1 + 0.8 + 0.7 + 0.2) / 2)
    assert res["post_wrong_yes_minus_no"]["mean"] == pytest.approx((0.7 - 0.7) / 2)


def test_split_rows_and_tasks() -> None:
    rows = [{"part": "masked", "unit": "m"}, {"part": "near_day", "unit": "n"}]
    parts = sc.split_rows(rows)
    assert list(parts) == list(sc.PARTS)
    assert parts["masked"] == [rows[0]]
    assert parts["lookalike"] == []
    item = Item(unit="x", state={"question": "Q?"}, options=("a", "b"))
    tasks = sc.part_tasks({"masked": [item]}, 3)
    assert [t.request.replicate for t in tasks] == [0, 1, 2]
