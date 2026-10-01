from collections.abc import Sequence

import numpy as np
import pandas as pd

from beyond_answer_confidence.tasks import benchmarks as bm
from beyond_answer_confidence.tasks import evidence as ev


def hotpot_table() -> pd.DataFrame:
    rows = []
    for i, answer in enumerate(["yes", "Alpha Town", "Nowhere", "no"]):
        titles = ["Alpha Town", "Beta City", "Pad One", "Pad Two", "Pad Three"]
        rows.append(
            {
                "id": f"q{i}",
                "type": "comparison" if i < 4 else "bridge",
                "question": f"Question {i}?",
                "answer": answer,
                "context": {
                    "title": np.array(titles),
                    "sentences": [np.array([f"{t} s1. ", f"{t} s2."]) for t in titles],
                },
                "supporting_facts": {
                    "title": np.array(["Alpha Town", "Beta City", "Alpha Town"])
                },
            }
        )
    rows.append({**rows[0], "id": "b", "type": "bridge"})
    return pd.DataFrame(rows)


def test_hotpot_items_build_four_cells() -> None:
    items = ev.hotpot_items(hotpot_table())
    # q2's answer is not a supporting title and the bridge row is skipped.
    assert len(items) == 3 * 4
    by = {it.unit: it for it in items}
    assert "paragraphs" not in by["hotpot:q0:closed"].state
    assert by["hotpot:q0:closed"].options in (("yes", "no"), ("no", "yes"))
    assert by["hotpot:q1:dose0"].gold == "Alpha Town"
    assert by["hotpot:q1:dose0"].info["kind"] == "entity"
    dose2 = by["hotpot:q0:dose2"].state["paragraphs"]
    assert sorted(p.split(":")[0] for p in dose2) == ["Alpha Town", "Beta City"]
    assert dose2[0].endswith("s1. " + dose2[0].split(":")[0] + " s2.")
    dose0 = by["hotpot:q0:dose0"].state["paragraphs"]
    assert all(p.startswith("Pad") for p in dose0)
    dose1 = [p.split(":")[0] for p in by["hotpot:q0:dose1"].state["paragraphs"]]
    assert len(dose1) == 2
    assert sum(t in ("Alpha Town", "Beta City") for t in dose1) == 1
    assert ev.hotpot_items(hotpot_table()) == items


def qanta_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "qanta_id": q,
                "id": f"{q}-{2 - s}",
                "text": f"clue {2 - s} of {q} ",
                "page": f"Answer_{q}",
                "category": "Science" if q < 5 else "Lonely",
            }
            for q in range(6)
            for s in range(3)
        ]
    )


def first_k(
    golds: Sequence[str], _ex: Sequence[Sequence[str]], pool: Sequence[str]
) -> list[list[str] | None]:
    out: list[list[str] | None] = []
    for g in golds:
        others = [p for p in pool if p != g]
        out.append(others[:3] if len(others) >= 3 else None)
    return out


def test_quizbowl_questions_order_sentences() -> None:
    qs = ev.quizbowl_questions(qanta_table())
    assert qs[0] == {
        "qid": "0",
        "sentences": ["clue 0 of 0", "clue 1 of 0", "clue 2 of 0"],
        "page": "Answer_0",
        "category": "Science",
    }


def test_quizbowl_items_reveal_clues() -> None:
    t = qanta_table()
    items = ev.quizbowl_items(t, t, first_k)
    # 5 Science questions x 3 prefixes; the lone category has no distractors.
    assert len(items) == 15
    first, last = items[0], items[2]
    assert first.state == {"clues": "clue 0 of 0"}
    assert last.state == {"clues": "clue 0 of 0 clue 1 of 0 clue 2 of 0"}
    assert last.info["fraction"] == 1.0
    assert first.descriptions is not None
    assert first.gold is not None
    assert first.descriptions[first.gold] == "Answer 0"
    assert first.descriptions == last.descriptions


def test_small_benchmark_helpers() -> None:
    assert bm.display_case("JOHN BUCHAN") == "John Buchan"
    assert bm.display_case("USA") == "USA"
    assert bm.display_case("TARTAR.") == "Tartar"
    assert bm.ambig_label(["singleAnswer", "singleAnswer"]) == "single"
    assert bm.ambig_label(["multipleQAs"]) == "ambiguous"
    assert bm.ambig_label(["multipleQAs", "singleAnswer"]) == "mixed"


def test_selfaware_and_ambigqa_items() -> None:
    sa = pd.DataFrame(
        [
            {"question_id": 1, "question": "q1?", "answerable": True, "source": "s"},
            {"question_id": 2, "question": "q2?", "answerable": False, "source": "t"},
        ]
    )
    items = bm.selfaware_items(sa)
    assert [it.unit for it in items] == ["selfaware:1", "selfaware:2"]
    assert items[1].info == {"set": "selfaware", "answerable": False, "source": "t"}
    assert items[0].nouls == {"answerable": bm.ANSWERABLE_QUESTION}
    amb = pd.DataFrame(
        [
            {
                "id": "a",
                "question": "q?",
                "annotations": {"type": np.array(["multipleQAs"])},
            }
        ]
    )
    (it,) = bm.ambigqa_items(amb)
    assert it.info["label"] == "ambiguous"
    assert it.options == ()


def test_simpleqa_items_perturb_dates() -> None:
    df = pd.DataFrame(
        [
            {
                "original_index": i,
                "problem": f"p{i}?",
                "answer": a,
                "topic": "t",
                "answer_type": at,
            }
            for i, (a, at) in enumerate(
                [
                    ("12 May 1998", "Date"),
                    ("no year", "Date"),
                    ("Ann", "Person"),
                    ("Bob", "Person"),
                    ("Cy", "Person"),
                    ("Di", "Person"),
                ]
            )
        ]
    )
    items = bm.simpleqa_items(df, first_k)
    by = {it.unit: it for it in items}
    assert by["simpleqa:0"].info["perturbed"] is True
    assert "simpleqa:1" not in by  # no year and only one other Date answer
    assert by["simpleqa:2"].info["perturbed"] is False
    d = by["simpleqa:0"].descriptions
    assert d is not None
    assert all(v.startswith("12 May ") for v in d.values())


def test_truthfulqa_items() -> None:
    binary = pd.DataFrame(
        [
            {
                "Question": "q?",
                "Best Answer": "good",
                "Best Incorrect Answer": "bad",
                "Category": "c",
                "Type": "t",
            }
        ]
    )
    mc1 = pd.DataFrame(
        [
            {
                "question": "m?",
                "mc1_targets": {
                    "choices": np.array(["x", "y", "z"]),
                    "labels": np.array([0, 1, 0]),
                },
            },
            {
                "question": "big?",
                "mc1_targets": {
                    "choices": np.array([str(i) for i in range(17)]),
                    "labels": np.array([1] + [0] * 16),
                },
            },
        ]
    )
    items = bm.truthfulqa_items(binary, mc1)
    assert [it.unit for it in items] == ["tqa_binary:0", "tqa_mc1:0"]
    assert items[0].descriptions is not None
    assert items[0].gold is not None
    assert items[0].descriptions[items[0].gold] == "good"
    assert items[1].descriptions is not None
    assert items[1].gold is not None
    assert items[1].descriptions[items[1].gold] == "y"
    assert items[1].info == {"set": "truthfulqa_mc1", "k": 3}


def test_triviaqa_items_keep_first_row_per_question() -> None:
    rows = [
        {
            "question_id": f"t{i}",
            "question": f"q{i}?",
            "answer": {
                "value": v,
                "aliases": np.array([v.lower()]),
                "type": "WikipediaEntity",
            },
        }
        for i, v in enumerate(["JOHN BUCHAN", "Paris", "Rome", "Oslo"])
    ]
    df = pd.DataFrame([*rows, rows[0]])
    items = bm.triviaqa_items(df, first_k)
    assert [it.unit for it in items] == [f"triviaqa:t{i}" for i in range(4)]
    d = items[0].descriptions
    assert d is not None
    assert items[0].gold is not None
    assert d[items[0].gold] == "John Buchan"


def test_chaos_items_attach_human_distributions() -> None:
    nli = pd.DataFrame(
        [
            {
                "uid": "u1",
                "label_count": [50, 30, 20],
                "old_labels": ["entailment", "neutral", "bogus"],
                "example": {"premise": "p", "hypothesis": "h"},
                "majority_label": "e",
                "entropy": 1.0,
            },
            {
                "uid": "u2",
                "label_count": [0, 0, 100],
                "old_labels": None,
                "example": {"premise": "p2", "hypothesis": "h2"},
                "majority_label": "c",
                "entropy": 0.0,
            },
        ]
    )
    anli = pd.DataFrame(
        [
            {
                "uid": "a1",
                "label_count": [25, 75],
                "example": {"obs1": "o1", "obs2": "o2", "hyp1": "e1", "hyp2": "e2"},
                "majority_label": 2,
                "entropy": 0.8,
            }
        ]
    )
    items = bm.chaos_items({"snli": nli}, anli)
    assert [it.unit for it in items] == [
        "chaos_snli:u1",
        "chaos_snli:u2",
        "chaos_anli:a1",
    ]
    assert items[0].info["human"] == {
        "entailment": 0.5,
        "neutral": 0.3,
        "contradiction": 0.2,
    }
    assert items[0].info["old_dist"] == {
        "entailment": 0.5,
        "neutral": 0.5,
        "contradiction": 0.0,
    }
    assert items[0].gold == "entailment"
    assert items[1].info["old_dist"] is None
    assert items[1].gold == "contradiction"
    assert items[2].gold == "2"
    assert items[2].info["human"] == {"1": 0.25, "2": 0.75}
    assert items[2].state == {"observation_1": "o1", "observation_2": "o2"}


def test_seeded_builders_are_reproducible() -> None:
    t = qanta_table()
    assert ev.quizbowl_items(t, t, first_k, seed=3) == ev.quizbowl_items(
        t, t, first_k, seed=3
    )
