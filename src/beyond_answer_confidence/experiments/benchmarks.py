"""Public uncertainty benchmarks, and agreement with human label spread.

SelfAware (unanswerable questions), SimpleQA Verified, TruthfulQA, TriviaQA
(calibration and error detection), AmbigQA (ambiguous questions) and
ChaosNLI (100 human labels per item; see
:mod:`beyond_answer_confidence.tasks.benchmarks`).
"""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from beyond_answer_confidence.config import as_dict
from beyond_answer_confidence.data import loaders
from beyond_answer_confidence.data.distractors import distractor_picker
from beyond_answer_confidence.experiments.base import (
    Experiment,
    backend_for,
    write_json,
    write_jsonl,
)
from beyond_answer_confidence.metrics.distributions import js_divergence, tv
from beyond_answer_confidence.metrics.ranking import auroc
from beyond_answer_confidence.runner import run_items
from beyond_answer_confidence.settings import Settings
from beyond_answer_confidence.stats.inference import (
    auroc_vs_threshold,
    holm_adjust_tails,
    spearman_interval,
)
from beyond_answer_confidence.stats.resampling import smece_interval
from beyond_answer_confidence.tasks.benchmarks import (
    ambigqa_items,
    chaos_items,
    selfaware_items,
    simpleqa_items,
    triviaqa_items,
    truthfulqa_items,
)
from beyond_answer_confidence.tasks.schema import Item, Task, item_tasks

logger = logging.getLogger(__name__)

NAME = "benchmarks"

CALIBRATION_SETS = ("simpleqa", "truthfulqa_binary", "triviaqa", "truthfulqa_mc1")
"""Multiple-choice sets whose calibration is measured (in bootstrap order)."""


@dataclass(frozen=True)
class Config:
    """Options of the benchmarks experiment.

    Attributes:
        replicates: Replicates per item.
        seed: Seed for perturbations and option order.
        bootstrap: Bootstrap resamples.
        bootstrap_seed: Bootstrap seed.
        smece_reference: Reference SmoothECE.
        selfaware_auroc_reference: Reference AUROC for flagging
            unanswerable questions.
        ambiguity_auroc_reference: Reference AUROC for flagging ambiguous
            questions.
        human_entropy_rho_reference: Reference Spearman correlation between
            model and human label entropy.
    """

    replicates: int = 3
    seed: int = 0
    bootstrap: int = 2000
    bootstrap_seed: int = 0
    smece_reference: float = 0.05
    selfaware_auroc_reference: float = 0.80
    ambiguity_auroc_reference: float = 0.70
    human_entropy_rho_reference: float = 0.30


def build_items(settings: Settings, config: Config) -> list[Item]:
    """Load the datasets and build every item.

    Args:
        settings: Run settings (data directory).
        config: Options.

    Returns:
        SelfAware, SimpleQA, TruthfulQA, AmbigQA, ChaosNLI and TriviaQA items.
    """
    d = settings.data_dir
    pick = distractor_picker(d)
    return (
        selfaware_items(loaders.load_selfaware(d))
        + simpleqa_items(loaders.load_simpleqa(d), pick, config.seed)
        + truthfulqa_items(
            loaders.load_truthfulqa_binary(d),
            loaders.load_truthfulqa_mc1(d),
            config.seed,
        )
        + ambigqa_items(loaders.load_ambigqa_validation(d))
        + chaos_items(
            {s: loaders.load_chaosnli(s, d) for s in ("snli", "mnli_m")},
            loaders.load_chaosnli("alphanli", d),
        )
        + triviaqa_items(loaders.load_triviaqa_validation(d), pick, config.seed)
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


def calibration_summary(
    df: pd.DataFrame, b: int, rng: np.random.Generator, reference: float
) -> dict[str, Any]:
    """SmoothECE with interval, error-detection AUROCs and selective accuracy.

    Args:
        df: Scored multiple-choice rows (``p_max``, ``correct``, ``p_known``).
        b: Bootstrap resamples.
        rng: Random generator.
        reference: Reference SmoothECE.

    Returns:
        :func:`~beyond_answer_confidence.stats.resampling.smece_interval` fields plus
        AUROCs of ``p_max`` and ``p_known`` for correctness, the share of
        confident errors (``p_max >= 0.9``), mean ``p_known`` and coverage
        and accuracy at ``p_max`` thresholds 0.5, 0.7 and 0.9.
    """
    res = smece_interval(
        df["p_max"].to_numpy(), df["correct"].to_numpy(), b, rng, threshold=reference
    )
    res["error_auroc_p_max"] = auroc(
        df["p_max"].to_numpy(), df["correct"].to_numpy(bool)
    )
    res["error_auroc_p_known"] = auroc(
        df["p_known"].to_numpy(), df["correct"].to_numpy(bool)
    )
    res["confident_wrong_rate"] = float(
        ((df["p_max"] >= 0.9) & ~df["correct"].astype(bool)).mean()
    )
    res["mean_p_known"] = float(df["p_known"].mean())
    cov = {}
    for t in (0.5, 0.7, 0.9):
        keep = df["p_max"] >= t
        cov[str(t)] = {
            "coverage": float(keep.mean()),
            "accuracy": float(df[keep]["correct"].mean()) if keep.any() else None,
        }
    res["selective"] = cov
    return res


def _chaos_subset(
    g: pd.DataFrame, b: int, rng: np.random.Generator, reference: float
) -> dict[str, Any]:
    k = len(next(iter(g["human"])))
    uniform = dict.fromkeys(g["human"].iloc[0], 1 / k)
    pairs = list(zip(g["dist"], g["human"], strict=True))
    entry: dict[str, Any] = {
        "n": len(g),
        "accuracy_vs_majority": float(g["correct"].mean()),
        "model_tv": float(np.mean([tv(d, h) for d, h in pairs])),
        "model_jsd": float(np.mean([js_divergence(d, h) for d, h in pairs])),
        "uniform_tv": float(np.mean([tv(uniform, h) for h in g["human"]])),
        "majority_onehot_tv": float(
            np.mean([tv({max(h, key=h.__getitem__): 1.0}, h) for h in g["human"]])
        ),
        "spearman_entropy": spearman_interval(
            g["norm_entropy"].to_numpy(float),
            g["human_entropy"].to_numpy(float),
            b,
            rng,
            threshold=reference,
        ),
        "mean_model_norm_entropy": float(g["norm_entropy"].mean()),
    }
    olds = [
        (d, o, h)
        for d, o, h in zip(g["dist"], g["old_dist"], g["human"], strict=True)
        if o
    ]
    if olds:
        entry["five_annotator_tv"] = float(np.mean([tv(o, h) for _, o, h in olds]))
        entry["model_tv_same_items"] = float(np.mean([tv(d, h) for d, _, h in olds]))
    return entry


def analyse(rows: list[dict[str, Any]], config: Config) -> dict[str, Any]:
    """Flagging, calibration and human-agreement statistics.

    Bootstrap draws (one generator, in this order): SelfAware AUROC;
    SmoothECE for :data:`CALIBRATION_SETS`; AmbigQA AUROC; ChaosNLI (SNLI +
    MNLI) entropy correlation; the entropy correlation per ChaosNLI subset
    (sorted by name).

    Args:
        rows: Scored rows.
        config: Options.

    Returns:
        Effect sizes with intervals (tail shares Holm-adjusted across the
        six main comparisons) and descriptive summaries.
    """
    b = config.bootstrap
    rng = np.random.default_rng(config.bootstrap_seed)
    df = pd.DataFrame(rows)

    sa = df[df["set"] == "selfaware"]
    unanswerable = auroc_vs_threshold(
        1 - sa["p_answerable"].to_numpy(),
        ~sa["answerable"].to_numpy(bool),
        config.selfaware_auroc_reference,
        b,
        rng,
    )
    cal = {
        name: calibration_summary(df[df["set"] == name], b, rng, config.smece_reference)
        for name in CALIBRATION_SETS
    }
    amb = df[(df["set"] == "ambigqa") & (df["label"] != "mixed")]
    ambiguous = auroc_vs_threshold(
        1 - amb["p_single"].to_numpy(),
        (amb["label"] == "ambiguous").to_numpy(),
        config.ambiguity_auroc_reference,
        b,
        rng,
    )
    nli = df[df["set"] == "chaos_nli"]
    agreement = spearman_interval(
        nli["norm_entropy"].to_numpy(float),
        nli["human_entropy"].to_numpy(float),
        b,
        rng,
        threshold=config.human_entropy_rho_reference,
    )
    family = holm_adjust_tails(
        {
            "selfaware_unanswerable": unanswerable,
            "simpleqa": cal["simpleqa"],
            "truthfulqa_binary": cal["truthfulqa_binary"],
            "triviaqa": cal["triviaqa"],
            "ambigqa_ambiguous": ambiguous,
            "chaosnli_entropy_agreement": agreement,
        }
    )
    res: dict[str, Any] = {
        "selfaware_unanswerable": family["selfaware_unanswerable"],
        "calibration": {
            "simpleqa": family["simpleqa"],
            "truthfulqa_binary": family["truthfulqa_binary"],
            "triviaqa": family["triviaqa"],
            "truthfulqa_mc1": cal["truthfulqa_mc1"],
        },
        "ambigqa_ambiguous": family["ambigqa_ambiguous"],
        "chaosnli_entropy_agreement": family["chaosnli_entropy_agreement"],
    }
    res["selfaware_by_source"] = {
        str(s): float(g["p_answerable"].mean()) for s, g in sa.groupby("source")
    }
    res["ambigqa"] = {
        str(k): {"n": len(g), "mean_p_single": float(g["p_single"].mean())}
        for k, g in df[df["set"] == "ambigqa"].groupby("label")
    }
    res["simpleqa_by_answer_type"] = {
        str(t): {
            "n": len(g),
            "accuracy": float(g["correct"].mean()),
            "mean_p_max": float(g["p_max"].mean()),
        }
        for t, g in df[df["set"] == "simpleqa"].groupby("answer_type")
    }
    chaos = df[df["set"].isin(["chaos_nli", "chaos_anli"])]
    res["chaosnli"] = {
        str(subset): _chaos_subset(g, b, rng, config.human_entropy_rho_reference)
        for subset, g in chaos.groupby("subset")
    }
    return res


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
    summary="SelfAware, SimpleQA, TruthfulQA, TriviaQA, AmbigQA and ChaosNLI",
    config_type=Config,
    requests=requests,
    run=run,
)
