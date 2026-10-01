import asyncio
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from beyond_answer_confidence.analyses import combined_signals as cs
from beyond_answer_confidence.analyses import popularity_and_dose as pdz
from beyond_answer_confidence.analyses import replicate_averaging as ra
from beyond_answer_confidence.analyses import smece_coverage as sim
from beyond_answer_confidence.backends.cache import CachedClient
from beyond_answer_confidence.backends.fake import FakeBackend
from beyond_answer_confidence.experiments.base import write_jsonl
from beyond_answer_confidence.stats.smece_bounds import SMECE_METHODS
from beyond_answer_confidence.tasks.schema import Item, item_tasks


def _calibrated(
    n: int, seed: int = 0, shift: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.3, 1.0, n)
    y = (rng.random(n) < np.clip(p - shift, 0, 1)).astype(float)
    return p, y


# --- combined signals ------------------------------------------------------------


def test_combined_signal_gains_when_the_follow_up_is_informative() -> None:
    rng = np.random.default_rng(0)
    n = 600
    known = rng.random(n) < 0.5
    err = np.where(known, rng.random(n) < 0.1, rng.random(n) < 0.7)
    df = pd.DataFrame(
        {
            "p_max": rng.uniform(0.5, 1, n),  # uninformative
            "p_known": np.where(known, 0.8, 0.2) + rng.normal(0, 0.05, n),
            "error": err,
        }
    )
    res = cs.combined_signal(df, "known", np.arange(n) % 40, 50, rng)
    assert res["auroc"]["known"] > 0.7
    assert res["gain_both_minus_p_max"] > 0.1
    assert res["gain_ci95"][0] > 0
    assert res["tail"] == 0.0
    assert res["aurc"]["both"] < res["aurc"]["p_max"]


def test_pools_mark_fabricated_answers_as_errors() -> None:
    boundary = pd.DataFrame(
        [
            {"set": "popqa", "fabricated": False, "correct": True, "prop": "a"},
            {"set": "fabricated", "fabricated": True, "correct": None, "prop": "a"},
            {"set": "oracle_tf", "fabricated": None, "correct": False, "month": "x"},
        ]
    )
    evidence = pd.DataFrame([{"set": "hotpot", "correct": True, "qid": "q"}])
    bench = pd.DataFrame([{"set": "simpleqa", "correct": False, "unit": "u"}])
    pools = cs.pools(boundary, evidence, bench)
    assert list(pools["popqa_plus_made_up"][0]["error"]) == [False, True]
    assert list(pools["daily_oracle_yes_no"][0]["error"]) == [True]
    assert pools["hotpot"][1:] == ("enough", "qid")
    assert pools["simpleqa"][0]["error"].tolist() == [True]
    assert pools["quizbowl"][0].empty


# --- popularity and dose ---------------------------------------------------------------


def test_popqa_quintiles() -> None:
    rng = np.random.default_rng(1)
    n = 500
    pop = pd.DataFrame(
        {
            "set": "popqa",
            "s_pop": rng.integers(1, 10_000, n),
            "p_max": rng.uniform(0.3, 1, n),
        }
    )
    pop["correct"] = rng.random(n) < pop["p_max"]
    q = pdz.popqa_quintiles(pop, 20, rng)
    assert set(q) == {"0", "1", "2", "3", "4"}
    assert sum(v["n"] for v in q.values()) == n
    assert set(q["0"]["bounds"]) == set(SMECE_METHODS)
    assert set(q["0"]["tail"]) == set(SMECE_METHODS)


def test_within_item_slopes_known_answer() -> None:
    wide = pd.DataFrame({"a": [0.0, 1.0], "b": [1.0, 1.0], "c": [2.0, 1.0]})
    s = pdz.within_item_slopes(wide, ["a", "b", "c"], np.array([0.0, 1.0, 2.0]))
    assert s == pytest.approx([1.0, 0.0])


def test_dose_slopes_on_perfect_log_doses() -> None:
    rows = []
    for item in range(10):
        for cond, k in zip(pdz.DOSES, (0, 1, 2, 4, 8), strict=True):
            rows.append(
                {
                    "arm": "main",
                    "item": item,
                    "condition": cond,
                    "norm_entropy": 1.0 - 0.1 * np.log2(k + 1),
                }
            )
    res = pdz.dose_slopes(pd.DataFrame(rows), 20, np.random.default_rng(0))
    assert res["per_unit_log2_k_plus_1_L0_L8"]["mean"] == pytest.approx(-0.1)
    assert res["step_L0_to_L1"]["mean"] == pytest.approx(-0.1)
    assert res["items"] == 10


# --- replicate averaging ------------------------------------------------------------


def test_random_draws_align_items_across_replicates() -> None:
    rng = np.random.default_rng(0)
    n = 200
    frames = []
    for r in range(3):
        p = rng.uniform(0.4, 1, n)
        df = pd.DataFrame(
            {
                "unit": [f"u{i}" for i in range(n)],
                "set": "x",
                "p_max": p,
                "correct": rng.random(n) < p,
            }
        )
        frames.append(df.sample(frac=1, random_state=r))  # different order each

    def group(d: pd.DataFrame) -> ra.Groups:
        return {"x": (d["p_max"].to_numpy(float), d["correct"].to_numpy(float))}

    res = ra.random_draws(frames, ["unit"], group, 20, rng)["x"]
    assert res["draws"] == 20
    assert res["smece"]["min"] <= res["smece"]["mean"] <= res["smece"]["max"]
    with pytest.raises(ValueError, match="same items"):
        ra.random_draws(
            [frames[0], frames[1].iloc[1:], frames[2]], ["unit"], group, 2, rng
        )


def test_summary_structure() -> None:
    p, y = _calibrated(300, seed=10)
    frame = pd.DataFrame(
        {
            "unit": [f"u{i}" for i in range(300)],
            "set": "triviaqa",
            "p_max": p,
            "correct": y.astype(bool),
        }
    )
    modes = {
        m: [frame]
        for m in ("mean3", "single_random", "single_r0", "single_r1", "single_r2")
    }
    modes["mean2"] = [frame, frame, frame]

    def group(d: pd.DataFrame) -> ra.Groups:
        return {"triviaqa": (d["p_max"].to_numpy(float), d["correct"].to_numpy(float))}

    out = ra.summarise_modes(modes, group, 20, np.random.default_rng(0), draws=5)
    res = out["triviaqa"]
    assert res["smece_increase_single_vs_mean3"] == pytest.approx(0.0)
    assert set(res["single_random_bounds"]) == {"smece", "bounds", "bootstrap_bias"}
    assert len(res["single_call_smece_range"]) == 2
    assert res["random_draws"]["draws"] == 5


def _items(n: int = 6) -> list[Item]:
    return [
        Item(
            unit=f"u{i}",
            state={"message": f"synthetic message {i}"},
            options=("a", "b", "c"),
            gold="a",
            instructions="Pick one.",
            nouls={"known": "Do you know?"},
        )
        for i in range(n)
    ]


def _fill(cache: Path, items: list[Item]) -> None:
    client = CachedClient(cache, model="fake", max_input_tokens=10**9)
    tasks = item_tasks(items, ra.REPLICATES)

    async def go() -> None:
        async with FakeBackend() as live:
            await client.ask_many([t.request for t in tasks], live)

    asyncio.run(go())


def test_modes_from_the_cache(tmp_path: Path) -> None:
    items = _items()
    cache = tmp_path / "c.jsonl"
    _fill(cache, items)
    modes = ra.per_call_rows(items, cache, "fake", "t")
    assert set(modes) == {
        "mean3",
        "single_random",
        "mean2",
        "single_r0",
        "single_r1",
        "single_r2",
    }
    assert len(modes["mean2"]) == 3
    assert all(len(rows) == len(items) for sets in modes.values() for rows in sets)
    saved = tmp_path / "rows.jsonl"
    write_jsonl(saved, modes["mean3"][0])
    assert ra.count_mismatches(modes["mean3"][0], saved) == 0
    shifted = [{**r, "p_max": r["p_max"] + 0.1} for r in modes["mean3"][0]]
    assert ra.count_mismatches(shifted, saved) == len(items)
    extra = Item(unit="new", state={"message": "x"}, options=("a", "b"))
    with pytest.raises(RuntimeError, match="not cached"):
        ra.per_call_rows([*items, extra], cache, "fake", "t")


def test_knowledge_dial_modes() -> None:
    rows: list[dict[str, Any]] = []
    for item in range(4):
        for rep in range(3):
            p = 0.4 + 0.1 * rep
            rows.append(
                {
                    "arm": "main",
                    "item": item,
                    "condition": "L1",
                    "code_style": "ordinal",
                    "example_seed": 0,
                    "replicate": rep,
                    "gold": "x",
                    "swap_target": None,
                    "first_listed": "x",
                    "probs": {"x": p, "y": 1 - p},
                }
            )
    rows.append({**rows[0], "arm": "other"})
    modes = ra.knowledge_dial_modes(pd.DataFrame(rows))
    assert len(modes["mean3"][0]) == 4
    assert modes["mean3"][0]["p_max"].iloc[0] == pytest.approx(0.5)
    assert modes["single_r2"][0]["p_max"].iloc[0] == pytest.approx(0.6)
    groups = ra.groups_knowledge_dial(modes["mean3"][0])
    assert len(groups["L1"][0]) == 4
    assert len(groups["L8"][0]) == 0


def test_calibration_groups() -> None:
    kb = pd.DataFrame(
        {
            "set": ["popqa", "oracle_tf", "oracle_tf"],
            "month": [None, "2024-01", "2025-01"],
            "p_max": [0.9, 0.8, 0.7],
            "correct": [True, False, True],
        }
    )
    g = ra.groups_knowledge_boundary(kb)
    assert [len(v[0]) for v in g.values()] == [1, 1, 1]
    ev = pd.DataFrame(
        {
            "set": ["quizbowl"] * 4 + ["hotpot"],
            "k": [1, 2, 3, 4, 0],
            "n_sentences": [4] * 4 + [0],
            "cell": [None] * 4 + ["closed"],
            "p_max": [0.5] * 5,
            "correct": [True] * 5,
        }
    )
    g2 = ra.groups_evidence_sufficiency(ev)
    assert len(g2["quizbowl_middle"][0]) == 1
    assert len(g2["quizbowl_full"][0]) == 1
    assert len(g2["hotpot_closed"][0]) == 1
    oc = pd.DataFrame(
        {
            "set": ["mmlu_redux", "mmlu_redux", "clinc_k", "clinc_k", "anli"],
            "error_type": ["ok", "bad", None, None, None],
            "K": [None, None, 5, 10, None],
            "p_max": [0.5] * 5,
            "correct": [True] * 5,
        }
    )
    g3 = ra.groups_option_count(oc)
    assert len(g3["mmlu_redux"][0]) == 1
    assert {"clinc_K5", "clinc_K10"} <= set(g3)
    tv = pd.DataFrame(
        {"set": "trivia_verify", "p_correct": [0.8, 0.3], "truth": [True, True]}
    )
    conf, ok = ra.groups_verification(tv)["trivia_verify"]
    assert conf == pytest.approx([0.8, 0.7])
    assert ok.tolist() == [1.0, 0.0]


def test_unknown_source_raises(tmp_path: Path) -> None:
    from beyond_answer_confidence.settings import Settings

    with pytest.raises(KeyError, match="unknown source"):
        ra.analyse_source(
            Settings(output_dir=tmp_path), "nope", ra.Config(), np.random.default_rng(0)
        )


# --- SmoothECE coverage simulation ---------------------------------------------------


def test_simulation_pieces() -> None:
    pool = np.random.default_rng(5).uniform(0.4, 1.0, 5000)
    rng = np.random.default_rng(6)
    delta, truth = sim.delta_for(pool, 0.05, rng, 20_000)
    assert truth == pytest.approx(0.05, abs=0.002)
    assert delta > 0
    assert sim.delta_for(pool, 0.0, rng, 1000) == (0.0, 0.0)
    runs = [sim.simulate_once(pool, delta, 400, 30, s) for s in range(3)]
    summ = sim.summarise(truth, runs)
    assert set(summ["methods"]) == set(SMECE_METHODS)
    assert 0 <= summ["methods"]["recentred"]["upper_coverage"] <= 1
    assert summ["sims"] == 3


def test_simulate_in_process_is_deterministic() -> None:
    pools = {"u": np.random.default_rng(7).uniform(0.4, 1.0, 300)}
    config = sim.Config(
        sims=2, resamples=10, workers=1, population_draws=2000, targets=(0.0, 0.05)
    )
    a = sim.simulate(pools, config)
    b = sim.simulate(pools, config)
    assert a == b
    assert [c["target"] for c in a["conditions"]] == [0.0, 0.05]
    assert a["design"]["sims"] == 2
