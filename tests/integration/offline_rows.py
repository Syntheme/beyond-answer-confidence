"""Synthetic experiment rows for the offline-analysis end-to-end tests (no dataset text)."""

from pathlib import Path
from typing import Any

import numpy as np

from beyond_answer_confidence.experiments.base import write_jsonl
from beyond_answer_confidence.tasks.synthetic import make_scenario

MONTHS = tuple(f"{y}-{m:02d}" for y in (2023, 2024, 2025) for m in range(1, 13))
"""36 months; 2024-11 onwards (14 months) counts as after the cutoff."""
CUTOFF = "2024-11"
LETTERS = ("A", "B", "C", "D")
RELATIONS = ("genre", "author", "capital", "country", "sport", "producer")


def _choice_dist(rng: np.random.Generator, gold: str, p_max: float) -> dict[str, Any]:
    """A four-option row whose top option has ``p_max``; right with prob ``p_max``."""
    right = bool(rng.random() < p_max)
    top = gold if right else next(o for o in LETTERS if o != gold)
    rest = (1 - p_max) / 3
    dist = {o: (p_max if o == top else rest) for o in LETTERS}
    return {"gold": gold, "dist": dist, "p_max": p_max, "top": top, "correct": right}


def news_rows(
    n_per_month: int = 40, shift: float = 0.25, seed: int = 0, prefix: str = ""
) -> list[dict[str, Any]]:
    """Dated yes/no rows with quantised ``p_max``; accuracy drops after the cutoff."""
    rng = np.random.default_rng(seed)
    rows = []
    for i, month in enumerate(MONTHS):
        post = month >= CUTOFF
        for j in range(n_per_month):
            p = round(float(rng.integers(50, 100)) / 100, 2)
            right = bool(rng.random() < p - (shift if post else 0.0))
            gold = "yes" if j % 2 else "no"
            other = "no" if gold == "yes" else "yes"
            top = gold if right else other
            rows.append(
                {
                    "unit": f"{prefix}oracle_tf:{i * 1000 + j}",
                    "set": "oracle_tf",
                    "month": month,
                    "gold": gold,
                    "dist": {top: p, ("no" if top == "yes" else "yes"): 1 - p},
                    "p_max": p,
                    "top": top,
                    "correct": right,
                    "p_known": float(
                        np.clip(
                            0.5
                            - 0.005 * i
                            - (0.1 if post else 0)
                            + rng.normal(0, 0.05),
                            0.01,
                            0.99,
                        )
                    ),
                }
            )
    return rows


def boundary_rows(seed: int = 0) -> list[dict[str, Any]]:
    """PopQA, fabricated-entity and dated news rows."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for i in range(400):
        p = round(float(rng.uniform(0.3, 1.0)), 2)
        rows.append(
            {
                "unit": f"popqa:{i}",
                "set": "popqa",
                "prop": RELATIONS[i % len(RELATIONS)],
                "s_pop": int(rng.integers(1, 10_000)),
                "fabricated": False,
                **_choice_dist(rng, LETTERS[i % 4], p),
                "p_known": float(rng.uniform(0, 1)),
            }
        )
    for i in range(90):
        p = round(float(rng.uniform(0.3, 0.9)), 2)
        rows.append(
            {
                "unit": f"fab:{('genre', 'author', 'capital')[i % 3]}:{i}",
                "set": "fabricated",
                "prop": RELATIONS[i % len(RELATIONS)],
                "fabricated": True,
                "gold": None,
                "dist": {"A": p, "B": (1 - p) / 3, "C": (1 - p) / 3, "D": (1 - p) / 3},
                "p_max": p,
                "top": "A",
                "correct": None,
                "p_known": float(rng.uniform(0, 0.5)),
            }
        )
    return rows + news_rows(seed=seed + 1)


def evidence_rows(seed: int = 0) -> list[dict[str, Any]]:
    """HotpotQA cells and Quizbowl clue prefixes."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for q in range(60):
        for c in ("closed", "dose0", "dose1", "dose2"):
            p = round(float(rng.uniform(0.3, 1.0)), 2)
            rows.append(
                {
                    "unit": f"hotpot:{q}:{c}",
                    "set": "hotpot",
                    "qid": f"q{q}",
                    "cell": c,
                    **_choice_dist(rng, LETTERS[q % 4], p),
                    "p_enough": float(rng.uniform(0, 1)),
                }
            )
    for q in range(40):
        n = 4 + q % 3
        for k in range(1, n + 1):
            p = round(float(rng.uniform(0.25, 1.0)), 2)
            rows.append(
                {
                    "unit": f"quizbowl:{q}:{k}",
                    "set": "quizbowl",
                    "qid": f"z{q}",
                    "k": k,
                    "n_sentences": n,
                    **_choice_dist(rng, LETTERS[q % 4], p),
                    "p_enough": float(rng.uniform(0, 1)),
                }
            )
    return rows


def benchmark_rows(seed: int = 0) -> list[dict[str, Any]]:
    """Factual question-answering rows."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for s in ("simpleqa", "triviaqa", "truthfulqa_binary", "truthfulqa_mc1"):
        for i in range(150):
            p = round(float(rng.uniform(0.25, 1.0)), 2)
            rows.append(
                {
                    "unit": f"{s}:{i}",
                    "set": s,
                    **_choice_dist(rng, LETTERS[i % 4], p),
                    "p_known": float(rng.uniform(0, 1)),
                }
            )
    return rows


def cutoff_rows(n: int = 240, seed: int = 0) -> list[dict[str, Any]]:
    """Plain and "not known"-option rows of the follow-up cutoff sample."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for i in range(n):
        post = i % 2 == 0
        month = MONTHS[22 + (i // 2) % 14] if post else MONTHS[(i // 2) % 22]
        gold = "yes" if i % 3 else "no"
        wrong = "no" if gold == "yes" else "yes"
        ok = bool(rng.random() < (0.5 if post else 0.8))
        p_unk = float(rng.uniform(0.6, 1) if post else rng.uniform(0, 0.4))
        common = {
            "set": "oracle_tf",
            "month": month,
            "post": post,
            "gold": gold,
            "p_known": float(rng.uniform(0, 0.3) if post else rng.uniform(0.3, 0.7)),
        }
        p = round(float(rng.uniform(0.6, 1)), 2)
        rows.append(
            {
                **common,
                "unit": f"cutoff:base:oracle_tf:{i}",
                "cond": "base",
                "top": gold if ok else wrong,
                "p_max": p,
                "dist": {gold if ok else wrong: p, wrong if ok else gold: 1 - p},
            }
        )
        rest = (1 - p_unk) / 2
        dist = {
            "yes": rest + (0.01 if gold == "yes" else 0),
            "no": rest + (0.01 if gold == "no" else 0),
            "unknown": p_unk - 0.01,
        }
        rows.append(
            {
                **common,
                "unit": f"cutoff:unknown:oracle_tf:{i}",
                "cond": "unknown",
                "top": max(dist, key=dist.__getitem__),
                "p_max": max(dist.values()),
                "dist": dist,
            }
        )
    return rows


def odds_rows(n: int = 12) -> list[dict[str, Any]]:
    """Stated-odds readings in the three formats (exact up to a 0.9 factor)."""
    rows: list[dict[str, Any]] = []
    for i in range(n):
        sc = make_scenario(i)
        star = [sc.p_star[o] for o in sc.options]
        rows.append(
            {
                "unit": f"odds:noul:{i}",
                "scenario": i,
                "format": "noul",
                **{f"p_opt{k}": 0.9 * s for k, s in enumerate(star)},
            }
        )
        rows.append(
            {
                "unit": f"odds:score:{i}",
                "scenario": i,
                "format": "score",
                **{f"s_opt{k}": s * 10 - 0.5 for k, s in enumerate(star)},
            }
        )
        rows.append(
            {
                "unit": f"odds:instructed:{i}",
                "scenario": i,
                "format": "instructed",
                "dist": dict(zip(sc.options, star, strict=True)),
            }
        )
    return rows


def dial_rows(n_items: int = 30, seed: int = 0) -> list[dict[str, Any]]:
    """Knowledge-dial rows (three replicates) whose entropy falls with the dose."""
    rng = np.random.default_rng(seed)
    labels = [f"i{j}" for j in range(8)]
    conds = {"L0": 0.2, "L1": 0.35, "L2": 0.45, "L4": 0.55, "L8": 0.65}
    conds |= {"Lname": 0.5, "Lname+8": 0.7}
    rows: list[dict[str, Any]] = []
    for item in range(n_items):
        gold = labels[item % len(labels)]
        for cond, base in conds.items():
            for rep in range(3):
                p = float(np.clip(base + rng.normal(0, 0.05), 0.15, 0.95))
                top = gold if rng.random() < p else labels[(item + 1) % len(labels)]
                probs = dict.fromkeys(labels, (1 - p) / (len(labels) - 1))
                probs[top] = p
                rows.append(
                    {
                        "arm": "main",
                        "item": item,
                        "condition": cond,
                        "code_style": "ordinal",
                        "example_seed": 0,
                        "replicate": rep,
                        "gold": gold,
                        "swap_target": None,
                        "first_listed": labels[0],
                        "probs": probs,
                    }
                )
    return rows


def write_outputs(out: Path) -> None:
    """Write every experiment's synthetic rows under an output directory."""
    write_jsonl(out / "knowledge_boundary" / "rows.jsonl", boundary_rows())
    write_jsonl(out / "evidence_sufficiency" / "rows.jsonl", evidence_rows())
    write_jsonl(out / "benchmarks" / "rows.jsonl", benchmark_rows())
    write_jsonl(out / "follow_up_questions" / "rows.jsonl", cutoff_rows() + odds_rows())
    for i, ds in enumerate(("banking77", "clinc150")):
        write_jsonl(out / "knowledge_dial" / "test" / f"{ds}.jsonl", dial_rows(seed=i))
