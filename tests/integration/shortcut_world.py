"""A small synthetic world for the shortcut-control tests.

Tables stand in for the dataset loaders (made-up people, dated news
questions about invented towns, comparison questions with invented
paragraphs), and scored rows of the source experiments are generated with a
seeded generator so that the analyses see non-degenerate signals. No dataset
text is used anywhere.
"""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from beyond_answer_confidence.tasks.chance import DEVICES
from beyond_answer_confidence.tasks.contrastive import contrastive_items
from beyond_answer_confidence.tasks.evidence import hotpot_items
from beyond_answer_confidence.tasks.news import yes_no_news_items
from beyond_answer_confidence.tasks.schema import Item
from beyond_answer_confidence.tasks.shortcuts import PERSON_RELATION_ORDER

CUTOFF = "2024-01"
FIRST = [
    "Alda",
    "Bren",
    "Corin",
    "Dace",
    "Elow",
    "Fenn",
    "Garo",
    "Hesta",
    "Ivor",
    "Jorun",
    "Kael",
    "Lune",
    "Mira",
    "Nold",
    "Orin",
    "Pell",
    "Quin",
    "Rask",
    "Sela",
    "Tove",
]
LAST = [
    "Ashvale",
    "Brindle",
    "Corrow",
    "Dunmere",
    "Elstow",
    "Farrow",
    "Glenholt",
    "Hartwick",
    "Ilsby",
    "Jessop",
    "Kettleby",
    "Larkin",
    "Morrow",
    "Nettle",
    "Oakhurst",
    "Pendle",
    "Quarrel",
    "Rookby",
    "Stannard",
    "Thorne",
]
TOWNS = ("Varn", "Tessel", "Olmere", "Brask", "Quillow", "Dunhollow")
MONTHS = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]


def popqa_people() -> pd.DataFrame:
    """Made-up people in the six person relations (15 each)."""
    rows = []
    i = 0
    for r, prop in enumerate(PERSON_RELATION_ORDER):
        for k in range(15):
            subj = f"{FIRST[(k + 3 * r) % 20]} {LAST[(7 * k + r) % 20]}"
            obj = f"{prop} value {k % 6}"
            rows.append(
                {
                    "id": i,
                    "subj": subj,
                    "prop": prop,
                    "obj": obj,
                    "question": f"What is the {prop} of {subj}?",
                    "possible_answers": [obj],
                    "o_aliases": "",
                    "s_pop": (13 * k + r) % 50 + 1,
                }
            )
            i += 1
    return pd.DataFrame(rows)


def news_table(kind: str = "tf") -> pd.DataFrame:
    """Dated yes/no questions about invented towns, 24 months, 8 per month."""
    del kind
    rows = []
    for m in range(24):
        year, mon = 2023 + m // 12, m % 12 + 1
        month = f"{year}-{mon:02d}"
        for i in range(8):
            town = TOWNS[(m + i) % len(TOWNS)]
            name = MONTHS[mon - 1]
            question = [
                f"Will the {town} council approve plan {i} by {name} {year}?",
                f"Will the {town} fair open on {name} {i + 3}, {year}?",
                f"Will the {town} bridge reopen before the end of {name} {year}?",
                f"Will the {town} market expand in spring {year}?",
                f"Will the {town} library extend its hours on Tuesday?",
                f"Will the {town} ferry resume service?",
                f"Will the {town} choir tour in {name}?",
                f"Will the {town} mill hire staff during {year}?",
            ][i]
            rows.append(
                {
                    "question": question,
                    "answer": "Yes" if (i + m) % 3 else "No",
                    "date": f"{month}-15",
                    "category": "civic" if i % 2 else "economy",
                    "month": month,
                }
            )
    return pd.DataFrame(rows)


def comparison_table() -> pd.DataFrame:
    """Comparison questions with invented paragraphs (30 questions)."""
    titles = ["Alpha", "Beta", "Pad1", "Pad2", "Pad3"]
    return pd.DataFrame(
        [
            {
                "id": f"h{i}",
                "type": "comparison",
                "question": f"Compare item {i} with its rival?",
                "answer": ["yes", "no", "Alpha"][i % 3],
                "context": {
                    "title": np.array(titles),
                    "sentences": [
                        np.array([f"{t} fact {i}." + " extra words" * ((i + j) % 4)])
                        for j, t in enumerate(titles)
                    ],
                },
                "supporting_facts": {"title": np.array(["Alpha", "Beta"])},
            }
            for i in range(30)
        ]
    )


def _dist(options: Sequence[str], rng: np.random.Generator) -> dict[str, float]:
    w = rng.dirichlet(np.ones(len(options)))
    return {o: float(x) for o, x in zip(options, w, strict=True)}


def _choice_fields(
    options: Sequence[str], gold: str | None, rng: np.random.Generator
) -> dict[str, Any]:
    dist = _dist(options, rng)
    top = max(dist, key=dist.__getitem__)
    return {
        "gold": gold,
        "dist": dist,
        "p_max": round(dist[top], 2),
        "top": top,
        "correct": top == gold if gold is not None else None,
    }


def knowledge_rows(
    entities: Sequence[Item], news: Sequence[Item], rng: np.random.Generator
) -> list[dict[str, Any]]:
    """Knowledge-boundary rows: entities and news."""
    rows = []
    for it in entities:
        fab = bool(it.info["fabricated"])
        rows.append(
            {
                "unit": it.unit,
                "set": it.info["set"],
                "fabricated": fab,
                **_choice_fields(it.options, it.gold, rng),
                "p_known": float(rng.uniform(0, 0.6) if fab else rng.uniform(0.3, 1)),
            }
        )
    for it in news:
        post = it.info["month"] >= CUTOFF
        rows.append(
            {
                "unit": it.unit,
                "set": "oracle_tf",
                "month": it.info["month"],
                **_choice_fields(("yes", "no"), it.gold, rng),
                "p_known": float(rng.uniform(0, 0.5) if post else rng.uniform(0.2, 1)),
            }
        )
    return rows


def second_look_rows(
    entities: Sequence[Item],
    first: dict[str, dict[str, Any]],
    news: Sequence[Item],
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    """Second-look rows for entities and news."""
    rows = [
        {
            "unit": it.unit,
            "set": "popqa",
            "made_up": bool(it.info["fabricated"]),
            "first_p_max": first[it.unit]["p_max"],
            "proposed": first[it.unit]["top"],
            "p_correct": float(rng.random()),
            "s_chance": float(rng.uniform(0, 9)),
        }
        for it in entities
    ]
    rows += [
        {
            "unit": it.unit,
            "set": "oracle_tf",
            "post": it.info["month"] >= CUTOFF,
            "first_p_max": first[it.unit]["p_max"],
            "proposed": first[it.unit]["top"],
            "p_correct": float(rng.random()),
            "s_chance": float(rng.uniform(0, 9)),
        }
        for it in news
    ]
    return rows


def cutoff_rows(news: Sequence[Item], rng: np.random.Generator) -> list[dict[str, Any]]:
    """Follow-up cutoff rows (four conditions per news item)."""
    rows = []
    for it in news:
        post = it.info["month"] >= CUTOFF
        for cond in ("base", "date", "unknown", "date_unknown"):
            opts = ("yes", "no", "unknown") if "unknown" in cond else ("yes", "no")
            rows.append(
                {
                    "unit": f"cutoff:{cond}:{it.unit}",
                    "set": "oracle_tf",
                    "month": it.info["month"],
                    "category": it.info["category"],
                    "post": post,
                    "cond": cond,
                    **_choice_fields(opts, it.gold, rng),
                    "p_known": float(rng.random()),
                }
            )
    return rows


def evidence_rows(
    hotpot: Sequence[Item], rng: np.random.Generator
) -> list[dict[str, Any]]:
    """Evidence-sufficiency rows for the comparison items."""
    return [
        {
            "unit": it.unit,
            "set": "hotpot",
            "qid": it.info["qid"],
            "cell": it.info["cell"],
            **_choice_fields(it.options, it.gold, rng),
            "p_enough": float(
                rng.uniform(0.4, 1)
                if it.info["cell"] == "dose2"
                else rng.uniform(0, 0.7)
            ),
        }
        for it in hotpot
    ]


def contrastive_rows(
    items: Sequence[Item], rng: np.random.Generator
) -> list[dict[str, Any]]:
    """Contrastive-facts rows."""
    return [
        {
            "unit": it.unit,
            **it.info,
            **_choice_fields(it.options, it.gold, rng),
            "p_enough": float(rng.random()),
        }
        for it in items
    ]


def synthetic_rows(rng: np.random.Generator) -> list[dict[str, Any]]:
    """Synthetic-worlds rows (two cells, 20 scenarios, plus one alone row)."""
    rows = []
    opts = ("red", "green", "blue", "black")
    for s in range(20):
        profile = "uniform" if s % 2 else "skewed"
        for cell in ("D0", "SFp"):
            ideal = (
                dict.fromkeys(opts, 0.25)
                if profile == "uniform"
                else {"red": 0.7, "green": 0.1, "blue": 0.1, "black": 0.1}
            )
            dist = _dist(opts, rng)
            rows.append(
                {
                    "scenario": s,
                    "cell": cell,
                    "alone": False,
                    "profile": profile,
                    "gold": "green",
                    "dist": dist,
                    "ideal": ideal,
                    "p_max": max(dist.values()),
                    "p_settled": float(
                        rng.uniform(0.3, 1) if cell == "D0" else rng.uniform(0, 0.7)
                    ),
                }
            )
    rows.append({**rows[0], "alone": True})
    return rows


def device_rows(rng: np.random.Generator) -> list[dict[str, Any]]:
    """Stated-odds rows: three orders per device, plus one other set."""
    rows = [
        {
            "unit": f"device:{d}:{k}",
            "set": "device",
            "device": d,
            "dist": _dist(list(v[1]), rng),
        }
        for d, v in DEVICES.items()
        for k in range(3)
    ]
    rows.append({"unit": "relist:0", "set": "relist", "dist": {"a": 1.0}})
    return rows


def benchmark_rows(prefix: str, rng: np.random.Generator) -> list[dict[str, Any]]:
    """Generic scored multiple-choice rows (two sets)."""
    rows = []
    for i in range(12):
        opts = ("A", "B", "C", "D")
        rows.append(
            {
                "unit": f"{prefix}:{i}",
                "set": f"{prefix}_{'x' if i % 2 else 'y'}",
                **_choice_fields(opts, "A", rng),
            }
        )
    rows.append({"unit": f"{prefix}:open", "set": "open", "gold": None})
    return rows


def write_rows(out: Path, name: str, rows: Sequence[dict[str, Any]]) -> None:
    """Write ``<out>/<name>/rows.jsonl``."""
    path = out / name / "rows.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), "utf-8")


def all_news() -> list[Item]:
    """The yes/no news items of the world (8 per month)."""
    return yes_no_news_items(news_table(), 8, 0)


def write_sources(out: Path, entities: Sequence[Item], seed: int = 0) -> None:
    """Write every source experiment's rows except the shortcut controls."""
    rng = np.random.default_rng(seed)
    news = all_news()
    kb = knowledge_rows(entities, news, rng)
    first = {r["unit"]: r for r in kb}
    hotpot = hotpot_items(comparison_table(), 0)
    write_rows(out, "knowledge_boundary", kb)
    write_rows(out, "second_look", second_look_rows(entities, first, news, rng))
    write_rows(out, "follow_up_questions", cutoff_rows(news, rng))
    write_rows(out, "evidence_sufficiency", evidence_rows(hotpot, rng))
    write_rows(
        out,
        "contrastive_facts",
        contrastive_rows(contrastive_items(4, 0)[1], rng),
    )
    write_rows(out, "synthetic_worlds", synthetic_rows(rng))
    write_rows(out, "stated_odds", device_rows(rng))
    write_rows(out, "benchmarks", benchmark_rows("bench", rng))
    write_rows(out, "option_count", benchmark_rows("options", rng))
