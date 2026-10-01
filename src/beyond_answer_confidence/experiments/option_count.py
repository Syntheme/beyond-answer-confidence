"""Option count: calibration on standard multiple choice and with 5 to 150 options.

MMLU-Redux 2.0 (questions annotated as correct), MMLU-CF, ANLI, and a
CLINC150 sweep over the number of intent options (see
:mod:`beyond_answer_confidence.tasks.options`). The analysis reports top-label
SmoothECE with bias-recentred bootstrap intervals and error-detection AUROC
for each set and each option count, plus how often the model sides with a
corrected key on MMLU-Redux questions whose key is wrong.
"""

import logging
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd

from beyond_answer_confidence.config import as_dict
from beyond_answer_confidence.data import intents
from beyond_answer_confidence.data.multiple_choice import (
    load_anli,
    load_mmlu_cf,
    load_mmlu_redux,
)
from beyond_answer_confidence.experiments.base import (
    Experiment,
    backend_for,
    write_json,
    write_jsonl,
)
from beyond_answer_confidence.metrics.ranking import auroc
from beyond_answer_confidence.runner import run_items
from beyond_answer_confidence.settings import Settings
from beyond_answer_confidence.stats.inference import holm_adjust_tails
from beyond_answer_confidence.stats.resampling import smece_interval
from beyond_answer_confidence.tasks.options import (
    OPTION_COUNTS,
    anli_items,
    mmlu_cf_items,
    mmlu_redux_items,
    option_count_items,
)
from beyond_answer_confidence.tasks.schema import Item, Task, item_tasks

logger = logging.getLogger(__name__)

NAME = "option_count"


@dataclass(frozen=True)
class Config:
    """Options of the option-count experiment.

    Attributes:
        replicates: Replicates per item.
        intents_per_label: CLINC150 test utterances per intent in the sweep.
        option_counts: Numbers of intent options.
        bootstrap: Bootstrap resamples.
        bootstrap_seed: Bootstrap seed.
        smece_reference: Reference SmoothECE.
    """

    replicates: int = 3
    intents_per_label: int = 10
    option_counts: tuple[int, ...] = OPTION_COUNTS
    bootstrap: int = 2000
    bootstrap_seed: int = 0
    smece_reference: float = 0.05


def build_items(settings: Settings, config: Config) -> list[Item]:
    """Load the datasets and build every item.

    Args:
        settings: Run settings (data directory).
        config: Options.

    Returns:
        MMLU-Redux, MMLU-CF, ANLI, then the CLINC150 sweep.
    """
    d = settings.data_dir
    return (
        mmlu_redux_items(load_mmlu_redux(d))
        + mmlu_cf_items(load_mmlu_cf(d))
        + anli_items(load_anli(d))
        + option_count_items(
            intents.load("clinc150", d), config.intents_per_label, config.option_counts
        )
    )


def requests(settings: Settings, config: Config) -> list[Task]:
    """Every task of the experiment.

    Args:
        settings: Run settings.
        config: Options.

    Returns:
        The tasks.
    """
    return item_tasks(build_items(settings, config), config.replicates)


def analyse(rows: list[dict[str, Any]], config: Config) -> dict[str, Any]:
    """SmoothECE and error AUROC per set and per option count.

    Bootstrap draws (one generator, in this order): each option count
    (ascending), MMLU-Redux (correct questions), MMLU-CF, ANLI.

    Args:
        rows: Scored rows.
        config: Options.

    Returns:
        Calibration with intervals (Holm-adjusted tail shares at or above
        the reference) and descriptive summaries.
    """
    rng = np.random.default_rng(config.bootstrap_seed)
    df = pd.DataFrame(rows)

    def calibration(g: pd.DataFrame) -> dict[str, Any]:
        res = smece_interval(
            g["p_max"].to_numpy(),
            g["correct"].to_numpy(),
            config.bootstrap,
            rng,
            threshold=config.smece_reference,
        )
        res["error_auroc_p_max"] = auroc(
            g["p_max"].to_numpy(), g["correct"].to_numpy(bool)
        )
        if "p_known" in g and g["p_known"].notna().all():
            res["error_auroc_p_known"] = auroc(
                g["p_known"].to_numpy(), g["correct"].to_numpy(bool)
            )
        return res

    red = df[df["set"] == "mmlu_redux"]
    ok = red[red["error_type"] == "ok"]
    clinc = df[df["set"] == "clinc_k"]
    per_k = {int(cast(int, k)): calibration(g) for k, g in clinc.groupby("K")}
    res: dict[str, Any] = holm_adjust_tails(
        {
            "mmlu_redux_calibration": calibration(ok),
            "mmlu_cf_calibration": calibration(df[df["set"] == "mmlu_cf"]),
            "anli_calibration": calibration(df[df["set"] == "anli"]),
            "calibration_by_option_count": {
                "counts": per_k,
                "tail": max(v["tail"] for v in per_k.values()),
            },
        }
    )
    res |= _descriptives(df, red, clinc)
    return res


def _descriptives(
    df: pd.DataFrame, red: pd.DataFrame, clinc: pd.DataFrame
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in ("mmlu_redux", "mmlu_cf"):
        g = df[df["set"] == name]
        if name == "mmlu_redux":
            g = g[g["error_type"] == "ok"]
        out[f"{name}_by_subject"] = {
            str(s): {
                "n": len(x),
                "accuracy": float(x["correct"].mean()),
                "mean_p_max": float(x["p_max"].mean()),
            }
            for s, x in g.groupby("subject")
        }
    out["anli_by_round"] = {
        int(cast(int, r)): {
            "n": len(x),
            "accuracy": float(x["correct"].mean()),
            "mean_p_max": float(x["p_max"].mean()),
        }
        for r, x in df[df["set"] == "anli"].groupby("round")
    }
    flawed = red[red["error_type"] != "ok"]
    out["mmlu_redux_flawed"] = {
        str(t): {
            "n": len(x),
            "agrees_with_key": float(x["correct"].mean()),
            "mean_p_max": float(x["p_max"].mean()),
            "mean_p_known": float(x["p_known"].mean()),
        }
        for t, x in flawed.groupby("error_type")
    }
    wg = flawed[
        (flawed["error_type"] == "wrong_groundtruth") & flawed["corrected"].notna()
    ]
    if len(wg):
        picks_fix = [
            top == letters[str(int(fix))]
            for top, letters, fix in zip(
                wg["top"], wg["letter_of_index"], wg["corrected"], strict=True
            )
        ]
        out["wrong_groundtruth_parsed"] = {
            "n": len(wg),
            "share_choosing_corrected": float(np.mean(picks_fix)),
            "share_choosing_original_key": float(wg["correct"].mean()),
        }
    out["clinc_by_K"] = {
        int(cast(int, k)): {
            "accuracy": float(g["correct"].mean()),
            "mean_p_max": float(g["p_max"].mean()),
            "gap": float(g["p_max"].mean() - g["correct"].mean()),
            "mean_norm_entropy": float(g["norm_entropy"].mean()),
        }
        for k, g in clinc.groupby("K")
    }
    return out


def run(settings: Settings, config: Config) -> dict[str, Any]:
    """Collect (cache first), score, analyse and write the outputs.

    Args:
        settings: Run settings.
        config: Options.

    Returns:
        The summary (``results`` only when every item is complete).
    """
    rows, book = run_items(
        build_items(settings, config),
        cache=settings.cache_file(NAME),
        model=settings.model,
        replicates=config.replicates,
        backend=backend_for(settings),
        max_input_tokens=settings.max_input_tokens,
        concurrency=settings.concurrency,
        label=NAME,
    )
    out = settings.out(NAME)
    write_jsonl(out / "rows.jsonl", rows)
    summary: dict[str, Any] = {"config": as_dict(config), **book}
    if book["complete_items"] == book["items"]:
        summary["results"] = analyse(rows, config)
    else:
        logger.warning("%s: incomplete items; no analysis", NAME)
    write_json(out / "summary.json", summary)
    return summary


EXPERIMENT = Experiment(
    name=NAME,
    summary="MMLU-Redux, MMLU-CF, ANLI and 5-150 intent options: calibration",
    config_type=Config,
    requests=requests,
    run=run,
)
