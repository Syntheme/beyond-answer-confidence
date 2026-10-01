"""End-to-end runs of the offline analyses on synthetic experiment rows.

Every analysis reads rows that the experiments would have written (tiny
synthetic tables, no dataset text), writes ``summary.json`` under its own
output directory, and returns the same results.
"""

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from offline_rows import write_outputs

from beyond_answer_confidence.analyses import (
    abstention,
    combined_signals,
    cutoff_calibration,
    knowledge_cutoff,
    popularity_and_dose,
    recalibration,
    replicate_averaging,
    smece_coverage,
)
from beyond_answer_confidence.backends.cache import CachedClient
from beyond_answer_confidence.backends.fake import FakeBackend
from beyond_answer_confidence.config import load_config
from beyond_answer_confidence.experiments.base import write_jsonl
from beyond_answer_confidence.runner import run_items
from beyond_answer_confidence.settings import Settings
from beyond_answer_confidence.tasks.schema import Item, item_tasks

CONFIGS = Path(__file__).resolve().parents[2] / "configs"


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    write_outputs(tmp_path / "out")
    return Settings(data_dir=tmp_path / "data", output_dir=tmp_path / "out")


def _summary(settings: Settings, name: str) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(
        (settings.out(name) / "summary.json").read_text(encoding="utf-8")
    )
    return data


@pytest.mark.parametrize(
    "module",
    [
        abstention,
        combined_signals,
        cutoff_calibration,
        knowledge_cutoff,
        popularity_and_dose,
        recalibration,
        replicate_averaging,
        smece_coverage,
    ],
)
def test_config_files_match_defaults(module: Any) -> None:
    analysis = module.ANALYSIS
    path = CONFIGS / f"{analysis.name}.toml"
    assert load_config(analysis.config_type, path) == analysis.config_type()


def test_cutoff_calibration_end_to_end(settings: Settings) -> None:
    config = cutoff_calibration.Config(bootstrap=10)
    res = cutoff_calibration.ANALYSIS.run(settings, config)
    summary = _summary(settings, "cutoff_calibration")
    assert summary["config"]["bootstrap"] == 10
    assert len(summary["equal_confidence"]["bands"]) == 5
    assert summary["prevalence"]["sample"] == {"post": 120, "pre": 120}
    assert res["prevalence"]["panel_share"] == cutoff_calibration.PANEL_SHARE
    fig = settings.out("cutoff_calibration") / "figures" / "cutoff_calibration.png"
    assert fig.stat().st_size > 10_000


def test_cutoff_calibration_precise_floats(settings: Settings) -> None:
    config = cutoff_calibration.Config(bootstrap=5, figures=False, precise_float=True)
    res = cutoff_calibration.ANALYSIS.run(settings, config)
    assert res["equal_confidence"]["clusters"] == {"pre_months": 22, "post_months": 14}


def test_knowledge_cutoff_end_to_end(settings: Settings) -> None:
    res = knowledge_cutoff.ANALYSIS.run(
        settings, knowledge_cutoff.Config(bootstrap=10, splits=3)
    )
    summary = _summary(settings, "knowledge_cutoff")
    assert set(summary["sensitivity"]) == set(knowledge_cutoff.SENSITIVITY)
    assert res["split_within_months"]["splits"] == 3
    assert "first_post_month" in summary["selection"]


def test_recalibration_end_to_end(settings: Settings) -> None:
    config = recalibration.Config(
        bootstrap=10, permutations=5, repeats=2, smece_resamples=5
    )
    res = recalibration.ANALYSIS.run(settings, config)
    summary = _summary(settings, "recalibration")
    assert set(summary) == {"config", "held_out", "with_known", "across_cutoff"}
    assert res["across_cutoff"]["cross_fitted"]["splits"] == 4
    assert summary["held_out"]["post_cutoff_gap_after_platt"]["n"] > 0


def test_combined_signals_end_to_end(settings: Settings) -> None:
    res = combined_signals.ANALYSIS.run(settings, combined_signals.Config(bootstrap=5))
    assert set(res["pools"]) == {
        "popqa_plus_made_up",
        "daily_oracle_yes_no",
        "hotpot",
        "quizbowl",
        "simpleqa",
        "triviaqa",
        "truthfulqa_binary",
    }
    assert _summary(settings, "combined_signals")["pools"]["hotpot"]["n"] == 240


def test_abstention_end_to_end(settings: Settings) -> None:
    res = abstention.ANALYSIS.run(settings, abstention.Config(bootstrap=5))
    summary = _summary(settings, "abstention")
    assert summary["abstention"]["all"]["items"] == 240
    assert set(res["renormalised"]) == {"noul", "score", "instructed"}
    assert res["renormalised"]["noul"]["n"] == 12


def test_popularity_and_dose_end_to_end(settings: Settings) -> None:
    res = popularity_and_dose.ANALYSIS.run(
        settings, popularity_and_dose.Config(resamples=5)
    )
    assert set(res["smece_by_popularity_quintile"]) == {"0", "1", "2", "3", "4"}
    slopes = _summary(settings, "popularity_and_dose")["entropy_slopes"]
    assert slopes["banking77"]["per_doubling_of_k_L1_L8"]["mean"] < 0
    assert slopes["clinc150"]["items"] == 30


def _tiny_items(_settings: Settings) -> list[Item]:
    return [
        Item(
            unit=f"tiny:{i}",
            state={"message": f"synthetic message {i}"},
            options=("a", "b", "c"),
            gold=("a", "b")[i % 2],
            instructions="Pick one.",
            info={"set": "triviaqa"},
        )
        for i in range(40)
    ]


def test_replicate_averaging_end_to_end(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    items = _tiny_items(settings)
    cache = settings.cache_file("tiny")
    client = CachedClient(cache, model=settings.model, max_input_tokens=10**9)
    tasks = item_tasks(items, replicate_averaging.REPLICATES)
    asyncio.run(client.ask_many([t.request for t in tasks], FakeBackend()))
    rows, _ = run_items(
        items,
        cache=cache,
        model=settings.model,
        replicates=replicate_averaging.REPLICATES,
    )
    write_jsonl(settings.out("tiny") / "rows.jsonl", rows)

    def groups(df: Any) -> replicate_averaging.Groups:
        return {"tiny": (df["p_max"].to_numpy(float), df["correct"].to_numpy(float))}

    monkeypatch.setitem(
        replicate_averaging.EXPERIMENT_SOURCES, "tiny", (_tiny_items, groups)
    )
    config = replicate_averaging.Config(
        sources=("knowledge_dial_banking77", "tiny"), resamples=5, draws=3
    )
    res = replicate_averaging.ANALYSIS.run(settings, config)
    assert res["tiny"]["mean3_mismatches_vs_rows"] == 0
    assert res["tiny"]["tiny"]["smece_increase_single_vs_mean3"] == pytest.approx(0)
    assert set(res["knowledge_dial_banking77"]) == set(
        replicate_averaging.KNOWLEDGE_CONDITIONS
    )
    summary = _summary(settings, "replicate_averaging")
    assert summary["config"]["sources"] == ["knowledge_dial_banking77", "tiny"]


def test_smece_coverage_end_to_end(settings: Settings) -> None:
    config = smece_coverage.Config(
        sims=2, resamples=5, workers=1, population_draws=2000, targets=(0.0, 0.05)
    )
    res = smece_coverage.ANALYSIS.run(settings, config)
    pools = {c["pool"] for c in res["conditions"]}
    assert pools == {"banking77_L8", "clinc150_L1", "quizbowl_first", "triviaqa"}
    assert len(_summary(settings, "smece_coverage")["conditions"]) == 8
