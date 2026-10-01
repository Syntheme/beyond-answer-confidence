from typing import Any

import numpy as np
import pytest

from beyond_answer_confidence.experiments import (
    benchmarks,
    evidence_sufficiency,
    knowledge_boundary,
)
from beyond_answer_confidence.stats import inference
from beyond_answer_confidence.tasks.evidence import HOTPOT_CELLS


def test_mean_difference_interval() -> None:
    rng = np.random.default_rng(0)
    res = inference.mean_difference_interval(
        np.full(20, 0.3), np.full(30, 0.8), 100, rng
    )
    assert res["diff"] == pytest.approx(-0.5)
    assert res["ci"] == pytest.approx([-0.5, -0.5])
    assert res["tail"] == 0.0
    # Two vectorised draws, in order.
    rng_a, rng_b = np.random.default_rng(1), np.random.default_rng(1)
    a, c = np.arange(5.0), np.arange(7.0)
    got = inference.mean_difference_interval(a, c, 50, rng_a)
    ba = rng_b.integers(0, 5, (50, 5))
    bc = rng_b.integers(0, 7, (50, 7))
    d = a[ba].mean(axis=1) - c[bc].mean(axis=1)
    assert got["tail"] == pytest.approx(float(np.mean(d >= 0)))


def test_spearman_interval() -> None:
    x = np.arange(30.0)
    res = inference.spearman_interval(x, 2 * x + 1, 50, np.random.default_rng(0), 0.3)
    assert res["rho"] == pytest.approx(1.0)
    assert res["ci"] == pytest.approx([1.0, 1.0])
    assert res["tail"] == 0.0
    assert "tail" not in inference.spearman_interval(x, -x, 5, np.random.default_rng(0))


def test_holm_adjust_tails_reads_p_or_tail() -> None:
    out = inference.holm_adjust_tails({"a": {"p": 0.01}, "b": {"tail": 0.04, "x": 1}})
    assert out["a"]["p_holm"] == pytest.approx(0.02)
    assert out["b"]["p_holm"] == pytest.approx(0.04)
    assert out["b"]["x"] == 1


def boundary_rows() -> list[dict[str, Any]]:
    rng = np.random.default_rng(0)
    rows: list[dict[str, Any]] = []
    for i in range(500):
        pm = float(rng.uniform(0.3, 1))
        rows.append(
            {
                "set": "popqa",
                "prop": "genre",
                "s_pop": i,
                "fabricated": False,
                "p_max": pm,
                "correct": bool(rng.random() < pm),
                "p_known": pm,
            }
        )
    rows += [
        {
            "set": "fabricated",
            "prop": "genre",
            "fabricated": True,
            "p_max": 0.3,
            "correct": None,
            "p_known": 0.1,
        }
        for _ in range(100)
    ]
    for m in range(24):
        month = f"{2024 + m // 12}-{m % 12 + 1:02d}"
        for _ in range(20):
            ok = bool(rng.random() < (0.9 if m < 14 else 0.55))
            for name in ("oracle_tf", "oracle_mc"):
                rows.append(
                    {
                        "set": name,
                        "month": month,
                        "p_max": 0.7,
                        "correct": ok,
                        "p_known": 0.5,
                    }
                )
    return rows


def test_boundary_analysis_runs_on_synthetic_rows() -> None:
    cfg = knowledge_boundary.Config(bootstrap=50)
    res = knowledge_boundary.analyse(boundary_rows(), cfg)
    assert res["oracle_change_point"]["first_post_month"] == "2025-03"
    assert res["oracle_change_point"]["drop"] > 0.2
    assert "confidence_after_vs_before_cutoff" in res
    assert set(res["calibration_by_popularity"]["quintiles"]) == {0, 1, 2, 3, 4}
    assert res["fabricated_confidence"]["mean"] == pytest.approx(0.3)
    assert res["fabricated_detection_auroc"]["1-p_known"] == 1.0
    assert all(
        "p_holm" in res[k]
        for k in ("calibration_by_popularity", "fabricated_confidence")
    )
    assert res["popqa_smece"]["resamples"] == 50
    # Same seed, same numbers; no pass/fail flags anywhere.
    assert knowledge_boundary.analyse(boundary_rows(), cfg) == res
    assert "supported" not in str(res)
    # Without a drop, the before/after comparisons are not computed.
    high = knowledge_boundary.Config(bootstrap=50, min_drop=0.9)
    assert "confidence_after_vs_before_cutoff" not in knowledge_boundary.analyse(
        boundary_rows(), high
    )


def evidence_rows() -> list[dict[str, Any]]:
    rng = np.random.default_rng(1)
    rows: list[dict[str, Any]] = []
    for q in range(60):
        for d, cell in enumerate(HOTPOT_CELLS):
            pm = 0.5 + 0.15 * d
            rows.append(
                {
                    "set": "hotpot",
                    "qid": str(q),
                    "cell": cell,
                    "kind": "yes_no",
                    "p_max": pm,
                    "correct": bool(rng.random() < pm),
                    "norm_entropy": 1 - 0.3 * d,
                    "p_enough": 0.2 * d,
                }
            )
        n = 5
        for k in range(1, n + 1):
            pm = 0.25 + 0.15 * k
            rows.append(
                {
                    "set": "quizbowl",
                    "qid": str(q),
                    "k": k,
                    "n_sentences": n,
                    "fraction": k / n,
                    "category": "History",
                    "p_max": pm,
                    "correct": bool(rng.random() < pm),
                    "norm_entropy": 1 - 0.15 * k,
                    "p_enough": 0.1 * k,
                }
            )
    return rows


def test_evidence_analysis_runs_on_synthetic_rows() -> None:
    res = evidence_sufficiency.analyse(
        evidence_rows(), evidence_sufficiency.Config(bootstrap=50)
    )
    assert res["hotpot_evidence"]["entropy_slope"]["mean"] == pytest.approx(-0.3 / 1)
    assert res["quizbowl_clues"]["entropy_slope"]["mean"] == pytest.approx(-0.75)
    assert set(res["calibration_by_evidence"]["cells"]) == {
        "hotpot_dose0",
        "hotpot_dose1",
        "hotpot_dose2",
        "quizbowl_first",
        "quizbowl_middle",
        "quizbowl_full",
    }
    assert res["irrelevant_paragraphs"]["mean"] == pytest.approx(-0.3)
    assert res["hotpot_enough_auroc_dose2_vs_rest"] == 1.0
    assert set(res["quizbowl_by_k"]) == {1, 2, 3, 4, 5}


def test_entropy_slopes_known_answer() -> None:
    import pandas as pd

    df = pd.DataFrame(
        {
            "g": [1, 1, 1, 2, 2, 3],
            "x": [0, 1, 2, 0, 2, 5],
            "norm_entropy": [1, 0.5, 0, 0, 1, 0.3],
        }
    )
    slopes = evidence_sufficiency.entropy_slopes(df, "x", "g")
    assert slopes == pytest.approx([-0.5, 0.5])  # group 3 has a single x


def benchmark_rows() -> list[dict[str, Any]]:
    rng = np.random.default_rng(2)
    rows: list[dict[str, Any]] = []
    for i in range(200):
        answerable = bool(i % 3)
        rows.append(
            {
                "set": "selfaware",
                "answerable": answerable,
                "source": "x",
                "p_answerable": 0.8 if answerable else 0.3,
            }
        )
        rows.append(
            {
                "set": "ambigqa",
                "label": ["single", "ambiguous", "mixed"][i % 3],
                "p_single": 0.8 if i % 3 == 0 else 0.4,
            }
        )
        for name in benchmarks.CALIBRATION_SETS:
            pm = float(rng.uniform(0.3, 1))
            rows.append(
                {
                    "set": name,
                    "answer_type": "Date",
                    "p_max": pm,
                    "correct": bool(rng.random() < pm),
                    "p_known": pm,
                }
            )
        h = float(rng.uniform(0, 1.5))
        human = {"entailment": 0.5, "neutral": 0.3, "contradiction": 0.2}
        rows.append(
            {
                "set": "chaos_nli",
                "subset": "snli",
                "human": human,
                "human_entropy": h,
                "norm_entropy": h / 1.6,
                "dist": human,
                "correct": True,
                "old_dist": {"entailment": 0.6, "neutral": 0.4, "contradiction": 0.0},
            }
        )
        rows.append(
            {
                "set": "chaos_anli",
                "subset": "alphanli",
                "human": {"1": 0.7, "2": 0.3},
                "human_entropy": h,
                "norm_entropy": h,
                "dist": {"1": 0.9, "2": 0.1},
                "correct": True,
                "old_dist": None,
            }
        )
    return rows


def test_benchmark_analysis_runs_on_synthetic_rows() -> None:
    res = benchmarks.analyse(benchmark_rows(), benchmarks.Config(bootstrap=30))
    assert res["selfaware_unanswerable"]["auroc"] == 1.0
    assert res["ambigqa_ambiguous"]["auroc"] == 1.0
    assert res["chaosnli"]["snli"]["model_tv"] == 0.0
    assert res["chaosnli"]["snli"]["five_annotator_tv"] == pytest.approx(0.2)
    assert "five_annotator_tv" not in res["chaosnli"]["alphanli"]
    assert res["chaosnli_entropy_agreement"]["rho"] > 0.99
    assert set(res["calibration"]) == set(benchmarks.CALIBRATION_SETS)
    assert "p_holm" in res["calibration"]["simpleqa"]
    assert "p_holm" not in res["calibration"]["truthfulqa_mc1"]
    pm = np.array([r["p_max"] for r in benchmark_rows() if r["set"] == "simpleqa"])
    cov = res["calibration"]["simpleqa"]["selective"]["0.5"]["coverage"]
    assert cov == pytest.approx(float(np.mean(pm >= 0.5)))
