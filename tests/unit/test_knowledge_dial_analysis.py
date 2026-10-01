import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from intent_helpers import LABELS, dist, row

from beyond_answer_confidence.analyses import knowledge_dial as an
from beyond_answer_confidence.config import load_config
from beyond_answer_confidence.experiments import knowledge_dial

CONFIGS = Path(__file__).resolve().parents[2] / "configs"


def synthetic(n_items: int = 60) -> pd.DataFrame:
    """Knowledge lowers entropy; L0 is confident on a wrong label; Lirrel = L0."""
    rng = np.random.default_rng(1)
    rows = []
    p_by_dose = {
        "L1": 0.6,
        "L2": 0.7,
        "L4": 0.8,
        "L8": 0.9,
        "Lname": 0.7,
        "Lname+8": 0.9,
    }
    for item in range(n_items):
        gold = LABELS[1 + item % (len(LABELS) - 1)]
        other = LABELS[(LABELS.index(gold) + 1) % len(LABELS)]
        for rep in range(2):
            blank = dist(gold, 0.02, LABELS[0], 0.4)
            rows.append(row(item, "L0", blank, gold, rep))
            rows.append(row(item, "Lirrel", blank, gold, rep))
            for cond, p in p_by_dose.items():
                probs = (
                    dist(gold, p) if rng.uniform() < p else dist(gold, 0.0, other, p)
                )
                rows.append(row(item, cond, probs, gold, rep))
            rows.append(row(item, "Lswap", dist(gold, 0.0, other, 0.9), gold, rep))
    return pd.DataFrame(rows)


def test_aggregate_averages_replicates() -> None:
    rows = pd.DataFrame(
        [row(0, "L1", dist("i1", 0.6), "i1", 0), row(0, "L1", dist("i1", 0.8), "i1", 1)]
    )
    agg = an.aggregate(rows)
    assert len(agg) == 1
    assert agg["p_gold"].iloc[0] == pytest.approx(0.7)
    assert agg["correct"].iloc[0]
    assert agg["n_reps"].iloc[0] == 2
    assert agg["rep_sd_p_gold"].iloc[0] == pytest.approx(np.std([0.6, 0.8], ddof=1))
    assert agg["nll"].iloc[0] == pytest.approx(-math.log(0.7))


def test_within_item_slopes_exact() -> None:
    cells = pd.DataFrame(
        [
            {"item": 0, "condition": c, "norm_entropy": 1.0 - 0.1 * math.log2(k + 1)}
            for c, k in an.DOSE.items()
        ]
    )
    assert an.within_item_slopes(cells) == pytest.approx([-0.1])


def test_paired_diff_uses_shared_items() -> None:
    cells = pd.DataFrame(
        [
            {"item": 0, "condition": "A", "norm_entropy": 0.5},
            {"item": 0, "condition": "B", "norm_entropy": 0.2},
            {"item": 1, "condition": "A", "norm_entropy": 0.9},
        ]
    )
    assert an.paired_diff(cells, "A", "B") == pytest.approx([0.3])


def test_confident_wrong() -> None:
    cells = pd.DataFrame({"p_max": [0.4, 0.6, 0.9], "correct": [False, False, True]})
    assert an.confident_wrong(cells).tolist() == [0.0, 1.0, 0.0]


def test_analyse_end_to_end() -> None:
    res = an.analyse({"synth": synthetic()}, b=200)["synth"]
    assert res["dose_response"]["item_bootstrap"]["slope"] < 0
    assert res["dose_response"]["mixed_model"]["slope"] < 0
    lo, hi = res["dose_response"]["item_bootstrap"]["ci95"]
    assert lo <= res["dose_response"]["item_bootstrap"]["slope"] <= hi
    assert res["no_knowledge"]["mean_p_max"] == pytest.approx(0.4, abs=0.01)
    assert res["no_knowledge"]["chance"] == pytest.approx(1 / len(LABELS))
    assert res["no_knowledge"]["confident_wrong_rate"] == 0.0
    assert res["length_control"]["mean_diff"] == pytest.approx(0.0, abs=1e-12)
    assert set(res["conditions"]) == set(an.CONDITION_ORDER)
    assert set(res["conditions"]["L1"]["smece"]) == {"value", "ci95", "bootstrap_bias"}
    assert res["design_checks"]["swap_chose_swap_target"] == 1.0
    assert res["models"] == ["m"]


def test_analyse_is_reproducible_and_seed_dependent() -> None:
    rows = {"synth": synthetic(30)}
    a = an.analyse(rows, b=50, seed=3)
    b = an.analyse(rows, b=50, seed=3)
    c = an.analyse(rows, b=50, seed=4)
    ci = ("synth", "conditions", "L1", "p_max", "ci95")

    def get(res: dict[str, object]) -> object:
        out: object = res
        for k in ci:
            assert isinstance(out, dict)
            out = out[k]
        return out

    assert get(a) == get(b)
    assert get(a) != get(c)


def test_config_file_matches_defaults() -> None:
    cfg = load_config(knowledge_dial.Config, CONFIGS / "knowledge_dial.toml")
    assert cfg == knowledge_dial.Config()
