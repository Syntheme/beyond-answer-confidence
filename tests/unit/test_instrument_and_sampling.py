import math
import random
from pathlib import Path

import numpy as np
import pytest
from intent_helpers import tiny_dataset

from beyond_answer_confidence.analyses import error_detection
from beyond_answer_confidence.backends.base import parse_response
from beyond_answer_confidence.backends.cache import CallRecord
from beyond_answer_confidence.backends.fake import fake_body
from beyond_answer_confidence.config import load_config
from beyond_answer_confidence.experiments import instrument_checks as ic
from beyond_answer_confidence.experiments import out_of_scope
from beyond_answer_confidence.experiments import sampling_frequencies as sf
from beyond_answer_confidence.experiments.comparators import deep_ensemble, gliner

CONFIGS = Path(__file__).resolve().parents[2] / "configs"


@pytest.mark.parametrize(
    ("name", "config_type"),
    [
        ("instrument_checks", ic.Config),
        ("sampling_frequencies", sf.Config),
        ("out_of_scope", out_of_scope.Config),
        ("deep_ensemble", deep_ensemble.TrainConfig),
        ("deep_ensemble_comparison", deep_ensemble.AnalysisConfig),
        ("gliner", gliner.InferenceConfig),
        ("gliner_comparison", gliner.AnalysisConfig),
        ("error_detection", error_detection.Config),
    ],
)
def test_config_files_match_defaults(name: str, config_type: type) -> None:
    assert load_config(config_type, CONFIGS / f"{name}.toml") == config_type()


# --- instrument checks -----------------------------------------------------------


def test_max_abs_diff_and_summarise() -> None:
    assert ic.max_abs_diff(
        [{"a": 0.5, "b": 0.5}, {"a": 0.7, "b": 0.3}]
    ) == pytest.approx(0.2)
    assert ic.max_abs_diff([{"a": 1.0}]) == 0.0
    assert ic.summarise([]) == {}
    assert ic.summarise([1.0, 3.0]) == {
        "n": 2, "min": 1.0, "median": 2.0, "max": 3.0, "mean": 2.0
    }  # fmt: skip


def _fake_pairs(
    probes: list[ic.Probe], confidence_offset: float = 0.0
) -> list[tuple[ic.Probe, CallRecord]]:
    pairs = []
    for p in probes:
        body = fake_body(p.request.questions, "m")
        for ans in body["answers"].values():
            probs = list(ans["probabilities"].values())
            k = len(probs)
            ans["confidence"] = (k * max(probs) - 1) / (k - 1) + confidence_offset
        pairs.append((p, CallRecord("k", parse_response(body), True, 0.5, None)))
    return pairs


def test_checks_on_fake_answers() -> None:
    data = {
        "banking77": tiny_dataset(name="banking77"),
        "clinc150": tiny_dataset(n_labels=6, name="clinc150"),
    }
    builder = ic.ProbeBuilder(data, [ex.text for ex in data["clinc150"].oos_pool])
    config = ic.Config(determinism_items=4, order_items=4, batch_items=2, cost_items=2)
    probes = ic.build_probes(builder, config)
    checks = [p.check for p in probes]
    assert checks.count("determinism") == 4 * config.replicates
    assert checks.count("batched") == 2
    assert checks.count("single") == 2 * len(config.batch_perm_seeds)
    labels = {n: len(ds.labels) for n, ds in data.items()}
    res = ic.analyse(_fake_pairs(probes), labels)
    assert res["determinism"]["deterministic"] is True
    assert res["batching"]["batch_invariant"] is True
    assert res["confidence_formula"]["matches_documented_formula"] is True
    assert set(res["cost"]) == {
        f"{d}/{c}" for d in data for c in config.cost_conditions
    }
    assert res["cost"]["clinc150/L0"]["K"] == 6
    assert res["cost"]["banking77/L0"]["latency_s"]["mean"] == 0.5
    # The fake favours the first listed code, so reassigning codes moves the argmax.
    assert res["order_invariance"]["argmax_agreement_rate"] < 1.0
    off = ic.analyse(_fake_pairs(probes, 0.05), labels)["confidence_formula"]
    assert off["matches_documented_formula"] is False
    assert off["abs_residual"]["max"] == pytest.approx(0.05)


# --- sampling frequencies -------------------------------------------------------


def multinomial_reps(
    p: dict[str, float], n: int, reps: int, seed: int = 0
) -> list[dict[str, float]]:
    rng = random.Random(seed)  # noqa: S311 - simulation
    opts = list(p)
    out = []
    for _ in range(reps):
        counts = dict.fromkeys(opts, 0)
        for o in rng.choices(opts, weights=[p[o] for o in opts], k=n):
            counts[o] += 1
        out.append({o: c / n for o, c in counts.items()})
    return out


def test_pooled_n_recovers_sample_count() -> None:
    reps = multinomial_reps({"a": 0.5, "b": 0.3, "c": 0.2}, n=100, reps=400)
    assert 80 < sf.pooled_n(sf.option_stats(reps)) < 125


def test_pooled_n_edge_cases() -> None:
    assert sf.pooled_n(sf.option_stats([{"a": 0.6, "b": 0.4}] * 5)) == math.inf
    assert math.isnan(sf.pooled_n(sf.option_stats([{"a": 1.0, "b": 0.0}] * 3)))


def test_top2_correlation_is_negative_for_competing_options() -> None:
    reps = multinomial_reps({"a": 0.5, "b": 0.5}, n=100, reps=200)
    assert sf.top2_correlation(reps) < -0.9
    assert math.isnan(sf.top2_correlation([{"a": 0.6, "b": 0.4}] * 4))


def test_build_tasks_and_analyse() -> None:
    ds = tiny_dataset(n_labels=10, name="banking77")
    config = sf.Config()
    tasks = sf.build_tasks(ds, config)
    n_series = len(config.name_positions) + len(config.blank_positions)
    assert len(tasks) == n_series * config.replicates
    assert len({t.request.replicate for t in tasks}) == config.replicates
    first = tasks[0].request.questions["intent"]["criteria"]
    codes = list(first)
    series = {
        "a": [
            dict.fromkeys(codes, 0.0)
            | {codes[0]: round(0.5 + d, 2), codes[1]: round(0.5 - d, 2)}
            for d in (0.05, -0.05) * 15
        ]
    }
    res = sf.analyse(series)
    assert res["per_series"]["a"]["top2_corr"] == pytest.approx(-1.0)
    assert res["per_series"]["a"]["replicates"] == 30
    assert res["sum_range"] == [1.0, 1.0]
    assert res["n_mid_options"] == 2


# --- out-of-scope metrics ---------------------------------------------------------


def test_out_of_scope_metrics_perfect_separation() -> None:
    oos_row = {"oos": True, "gold": None, "closed_choice": "a", "closed_p_max": 0.3,
               "closed_norm_entropy": 0.8, "with_oos_choice": "out_of_scope",
               "p_oos": 0.9, "p_fits": 0.1}  # fmt: skip
    in_row = {"oos": False, "gold": "a", "closed_choice": "a", "closed_p_max": 0.95,
              "closed_norm_entropy": 0.05, "with_oos_choice": "a", "p_oos": 0.01,
              "p_fits": 0.95}  # fmt: skip
    m = out_of_scope.metrics([oos_row] * 20 + [in_row] * 20, b=50)
    assert m["with_oos_p_oos"]["auroc"] == 1.0
    assert m["with_oos_p_oos"]["auroc_ci95"] == [1.0, 1.0]
    assert m["oos_recall_with_oos"] == 1.0
    assert m["in_scope_accuracy_with_oos"] == 1.0
    assert m["fits_1_minus_p"]["fpr_at_95tpr"] == 0.0
    assert m["n_oos"] == 20
    with pytest.raises(ValueError, match="unknown score"):
        out_of_scope.score_value("nope", in_row)


def test_detection_skips_single_class_resamples() -> None:
    score = np.array([0.9, 0.1, 0.2])
    oos = np.array([True, False, False])
    res = out_of_scope.detection(score, oos, 200, np.random.default_rng(0))
    assert res["auroc"] == 1.0
    assert res["auroc_ci95"] == [1.0, 1.0]
