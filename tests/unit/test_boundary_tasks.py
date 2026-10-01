import json
import random

import pandas as pd
import pytest

from beyond_answer_confidence.stats.changepoint import change_point
from beyond_answer_confidence.tasks import entities, news
from beyond_answer_confidence.tasks.multiple_choice import (
    KNOWN_QUESTION,
    LETTERS,
    multiple_choice_item,
)

RELATIONS = [
    "author",
    "capital",
    "capital of",
    "color",
    "composer",
    "country",
    "director",
    "father",
    "genre",
    "mother",
    "occupation",
    "place of birth",
    "producer",
    "religion",
    "screenwriter",
    "sport",
]


def entity_table() -> pd.DataFrame:
    rows = []
    for i in range(12):
        prop = "genre" if i < 8 else "sport"
        rows.append(
            {
                "id": 100 + i,
                "subj": f"Thing{i}",
                "prop": prop,
                "obj": f"{prop}-value-{i % 6}",
                "question": f"What is the {prop} of Thing{i}?",
                "possible_answers": [f"{prop}-value-{i % 6}"],
                "o_aliases": json.dumps([f"alias {i}"]) if i % 2 else "",
                "s_pop": 10 * i,
            }
        )
    return pd.DataFrame(rows)


def test_fabricated_subjects_cover_every_relation() -> None:
    rng = random.Random(0)  # noqa: S311
    subjects = [entities.fabricated_subject(p, rng) for p in RELATIONS]
    assert all(subjects)
    assert entities.fabricated_subject("color", rng).startswith("flag of ")
    assert entities.fabricated_subject("author", rng).startswith("The ")
    with pytest.raises(ValueError, match="unknown"):
        entities.fabricated_subject("unknown", rng)


def test_primary_templates_and_pools() -> None:
    df = entity_table()
    assert entities.primary_templates(df) == {
        "genre": "What is the genre of {}?",
        "sport": "What is the sport of {}?",
    }
    pools = entities.relation_pools(df)
    assert pools["sport"] == [
        "sport-value-2",
        "sport-value-3",
        "sport-value-4",
        "sport-value-5",
    ]


def test_popqa_items_have_gold_among_options() -> None:
    df = entity_table()
    items = entities.popqa_items(df)
    # sport has 4 objects: 3 distractors remain for each question.
    assert len(items) == 12
    for it, row in zip(items, df.itertuples(), strict=True):
        assert it.unit == f"popqa:{row.id}"
        assert it.options == LETTERS[:4]
        assert it.gold is not None
        assert it.descriptions is not None
        assert it.descriptions[it.gold] == row.obj
        assert len(set(it.descriptions.values())) == 4
        assert it.info["s_pop"] == row.s_pop
        assert it.nouls == {"known": KNOWN_QUESTION}
    assert entities.popqa_items(df) == items  # seeded


def test_fabricated_items_have_no_gold() -> None:
    items = entities.fabricated_items(entity_table(), per_relation=3)
    assert [it.unit for it in items[:3]] == [
        "fab:genre:0",
        "fab:genre:1",
        "fab:genre:2",
    ]
    assert all(it.gold is None for it in items)
    assert all(it.info["fabricated"] for it in items)
    assert items[0].state["question"].startswith("What is the genre of ")
    assert items[0].info["subject"] in items[0].state["question"]


def test_multiple_choice_item_shuffles_with_seed() -> None:
    a = multiple_choice_item("u", "q?", ["right", "w1", "w2"], 0, random.Random(3), {})  # noqa: S311
    b = multiple_choice_item("u", "q?", ["right", "w1", "w2"], 0, random.Random(3), {})  # noqa: S311
    assert a == b
    assert a.descriptions is not None
    assert a.gold is not None
    assert a.descriptions[a.gold] == "right"
    assert a.options == ("A", "B", "C")
    none = multiple_choice_item("u", "q?", ["x", "y"], None, random.Random(3), {})  # noqa: S311
    assert none.gold is None


def test_parse_yes_no() -> None:
    assert news.parse_yes_no("** No. \n\n**") == "no"
    assert news.parse_yes_no("Yes.") == "yes"
    assert news.parse_yes_no("maybe") is None


def news_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    tf, mc = [], []
    for m in range(1, 4):
        for i in range(5):
            tf.append(
                {
                    "question": f"Will event {m}-{i} happen?",
                    "answer": ["Yes.", "No", "unclear"][i % 3],
                    "date": f"2024-0{m}-1{i}",
                    "category": "misc",
                    "month": f"2024-0{m}",
                }
            )
            mc.append(
                {
                    "question": f"Which of event {m}-{i}?",
                    "answer": ["a", "B ", "x", "d"][i % 4],
                    "choice_a": "one",
                    "choice_b": "two",
                    "choice_c": "three",
                    "choice_d": "four",
                    "date": f"2024-0{m}-1{i}",
                    "category": "misc",
                    "month": f"2024-0{m}",
                }
            )
    return pd.DataFrame(tf), pd.DataFrame(mc)


def test_news_items_sample_per_month() -> None:
    tf, mc = news_tables()
    yn = news.yes_no_news_items(tf, per_month=2)
    assert len(yn) == 6
    assert {it.gold for it in yn} <= {"yes", "no"}
    assert all(it.options == ("yes", "no") for it in yn)
    assert [it.info["month"] for it in yn[::2]] == ["2024-01", "2024-02", "2024-03"]
    assert news.yes_no_news_items(tf, per_month=2) == yn
    everything = news.yes_no_news_items(tf, per_month=100)
    assert len(everything) == 3 * 4  # 'unclear' labels dropped
    four = news.multiple_choice_news_items(mc, per_month=100)
    assert len(four) == 3 * 4  # 'x' labels dropped
    assert {it.gold for it in four} <= set("ABCD")
    assert four[0].descriptions == {"A": "one", "B": "two", "C": "three", "D": "four"}
    assert four[0].nouls == {"known": news.NEWS_KNOWN_QUESTION}


def test_change_point_finds_drop() -> None:
    months = [f"2024-{m:02d}" for m in range(1, 13)] + [
        f"2025-{m:02d}" for m in range(1, 13)
    ]
    cp = change_point(months, [0.9] * 15 + [0.6] * 9, [50] * 24)
    assert cp["first_post_month"] == "2025-04"
    assert cp["pre_accuracy"] == pytest.approx(0.9)
    assert cp["post_accuracy"] == pytest.approx(0.6)
    assert cp["drop"] == pytest.approx(0.3)
    flat = change_point(months, [0.7] * 24, [50] * 24)
    assert flat["drop"] == pytest.approx(0.0)
    assert flat["first_post_month"] == months[6]  # earliest split on ties
    with pytest.raises(ValueError, match="at least 12"):
        change_point(months[:11], [0.5] * 11, [1] * 11)


def test_change_point_respects_weights_and_min_side() -> None:
    months = [str(i) for i in range(8)]
    acc = [1, 1, 1, 0, 0, 0, 0, 0]
    assert change_point(months, acc, [1] * 8, min_side=2)["first_post_month"] == "3"
    assert change_point(months, acc, [1] * 8, min_side=4)["first_post_month"] == "4"
