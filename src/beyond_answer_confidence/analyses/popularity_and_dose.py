"""SmoothECE per popularity quintile, and entropy slopes per doubling of examples.

Offline; reads saved rows. One generator, sections in order:

1. **PopQA calibration within each popularity quintile.** Calibration in
   the large (mean confidence minus accuracy) per quintile can hide
   miscalibration that cancels out; this adds SmoothECE per quintile with
   bounds under four resampling schemes
   (:func:`beyond_answer_confidence.stats.smece_bounds.smece_with_bounds`).
2. **Knowledge-dial entropy slopes on three scales.** Normalised entropy
   regressed within item on ``log2(k + 1)`` over L0-L8 (slope per unit of
   ``log2(k + 1)``, not per doubling), on ``log2 k`` over L1-L8 (slope per
   doubling of the number of examples ``k``), and the L0 -> L1 step, each
   with an item-bootstrap interval.

Inputs: ``knowledge_boundary`` rows and the knowledge-dial test rows.
Output: ``<output_dir>/popularity_and_dose/summary.json``.
"""

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd

from beyond_answer_confidence.analyses.knowledge_dial import aggregate
from beyond_answer_confidence.analyses.rows import records_frame
from beyond_answer_confidence.config import as_dict
from beyond_answer_confidence.data.intents import DATASETS
from beyond_answer_confidence.experiments.base import Analysis, read_jsonl, write_json
from beyond_answer_confidence.experiments.knowledge_dial import rows_path
from beyond_answer_confidence.settings import Settings
from beyond_answer_confidence.stats.resampling import bootstrap_means, percentile_ci
from beyond_answer_confidence.stats.smece_bounds import smece_with_bounds

NAME = "popularity_and_dose"
DOSES = ("L0", "L1", "L2", "L4", "L8")
"""Knowledge-dial conditions with 0, 1, 2, 4 and 8 examples per intent."""


@dataclass(frozen=True)
class Config:
    """Options.

    Attributes:
        resamples: Resamples per scheme and per interval.
        seed: Seed.
        smece_reference: Reference SmoothECE for the tail shares.
        datasets: Knowledge-dial datasets.
        split: Knowledge-dial split.
    """

    resamples: int = 2000
    seed: int = 0
    smece_reference: float = 0.05
    datasets: tuple[str, ...] = DATASETS
    split: str = "test"


def popqa_quintiles(
    boundary: pd.DataFrame, b: int, rng: np.random.Generator, reference: float = 0.05
) -> dict[str, Any]:
    """SmoothECE per PopQA popularity quintile.

    Quintiles of the subject's page views (``s_pop``), ties broken by row
    order.

    Args:
        boundary: Knowledge-boundary rows.
        b: Resamples per scheme.
        rng: Random generator.
        reference: Reference value for the tail shares.

    Returns:
        Quintile -> accuracy, mean confidence, SmoothECE, bounds per scheme
        and tail shares at ``reference``.
    """
    real = boundary[boundary["set"] == "popqa"].copy()
    real["quintile"] = pd.qcut(real["s_pop"].rank(method="first"), 5, labels=False)
    out = {}
    for q, g in real.groupby("quintile"):
        chk = smece_with_bounds(
            g["p_max"].to_numpy(float), g["correct"].to_numpy(float), reference, b, rng
        )
        out[str(int(cast(int, q)))] = {
            "accuracy": float(g["correct"].mean()),
            "mean_confidence": float(g["p_max"].mean()),
            **{k: chk[k] for k in ("n", "smece", "bounds", "tail")},
        }
    return out


def within_item_slopes(
    wide: pd.DataFrame, cols: list[str], x: np.ndarray
) -> np.ndarray:
    """Least-squares slope of each row's values on ``x``.

    Args:
        wide: One row per item, one column per condition.
        cols: Conditions (columns) in the order of ``x``.
        x: Regressor per condition.

    Returns:
        One slope per item.
    """
    xc = x - x.mean()
    y = wide[cols].to_numpy(float)
    return np.asarray((y - y.mean(axis=1, keepdims=True)) @ xc / (xc @ xc))


def dose_slopes(
    cells: pd.DataFrame, b: int, rng: np.random.Generator
) -> dict[str, Any]:
    """Entropy slopes on three scales for one dataset.

    Args:
        cells: Aggregated knowledge-dial cells (see
            :func:`beyond_answer_confidence.analyses.knowledge_dial.aggregate`).
        b: Resamples.
        rng: Random generator.

    Returns:
        Slope per unit of ``log2(k + 1)`` (L0-L8), per doubling of ``k``
        (L1-L8) and the L1 - L0 step, each with its mean and 95 % interval;
        items with every dose; mean normalised entropy per dose.
    """
    main = cells[cells["arm"] == "main"]
    wide = main.pivot_table(index="item", columns="condition", values="norm_entropy")
    doses = list(DOSES)
    wide = wide[doses].dropna()
    s_pub = within_item_slopes(wide, doses, np.log2(np.array([0, 1, 2, 4, 8]) + 1))
    s_dbl = within_item_slopes(wide, doses[1:], np.log2(np.array([1, 2, 4, 8], float)))
    step = (wide["L1"] - wide["L0"]).to_numpy(float)
    res: dict[str, Any] = {}
    for name, v in (
        ("per_unit_log2_k_plus_1_L0_L8", s_pub),
        ("per_doubling_of_k_L1_L8", s_dbl),
        ("step_L0_to_L1", step),
    ):
        res[name] = {
            "mean": float(v.mean()),
            "ci95": percentile_ci(bootstrap_means(v, b, rng)),
        }
    res["items"] = len(wide)
    res["mean_norm_entropy_by_dose"] = {d: float(wide[d].mean()) for d in doses}
    return res


def run(settings: Settings, config: Config) -> dict[str, Any]:
    """Run both sections and write the summary.

    Args:
        settings: Run settings.
        config: Options.

    Returns:
        ``smece_by_popularity_quintile`` and ``entropy_slopes`` (by dataset).
    """
    rng = np.random.default_rng(config.seed)
    boundary = records_frame(settings, "knowledge_boundary")
    res: dict[str, Any] = {
        "smece_by_popularity_quintile": popqa_quintiles(
            boundary, config.resamples, rng, config.smece_reference
        )
    }
    slopes = {}
    for ds in config.datasets:
        rows = pd.DataFrame(read_jsonl(rows_path(settings, config.split, ds)))
        slopes[ds] = dose_slopes(aggregate(rows), config.resamples, rng)
    res["entropy_slopes"] = slopes
    write_json(settings.out(NAME) / "summary.json", {"config": as_dict(config), **res})
    return res


ANALYSIS = Analysis(
    name=NAME,
    summary="SmoothECE per popularity quintile; entropy slope per doubling",
    config_type=Config,
    run=run,
)
