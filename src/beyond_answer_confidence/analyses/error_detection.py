"""Error detection on the knowledge dial (offline; reads saved outputs).

Uses the knowledge-dial test rows and the deep ensemble's full-data
predictions. Per dataset:

- error detection and selective prediction per condition: AUROC (with an
  item-bootstrap interval) and average precision of ``1 - p_max`` for
  errors, AURC, AUGRC and the selective risk at 80/90/95 % coverage;
- replicate disagreement as a second signal: the mean pairwise
  total-variation distance between replicates, and whether adding it to
  ``1 - p_max`` in a grouped 5-fold cross-validated logistic model flags
  errors better (paired item bootstrap of the AUROC gain);
- equal-mass reliability bins for the knowledge conditions;
- where errors concentrate: the runner-up's share of the residual mass,
  whether wrong answers are ones the ensemble also ranks in its top two,
  and how many confused intent pairs the two share.

Output: ``<output_dir>/error_detection/summary.json``.
"""

from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from beyond_answer_confidence.analyses.knowledge_dial import aggregate
from beyond_answer_confidence.config import as_dict
from beyond_answer_confidence.data.intents import DATASETS, load
from beyond_answer_confidence.experiments.base import Analysis, read_jsonl, write_json
from beyond_answer_confidence.experiments.comparators.deep_ensemble import (
    decompose,
    predictions_path,
)
from beyond_answer_confidence.experiments.knowledge_dial import rows_path
from beyond_answer_confidence.metrics.calibration import reliability_bins
from beyond_answer_confidence.metrics.distributions import mean_pairwise_tv
from beyond_answer_confidence.metrics.ranking import (
    auroc,
    average_precision,
    risk_coverage,
)
from beyond_answer_confidence.settings import Settings
from beyond_answer_confidence.stats.recalibration import cv_logistic_scores
from beyond_answer_confidence.stats.resampling import (
    bootstrap,
    on_resample,
    percentile_ci,
)

NAME = "error_detection"
KNOWLEDGE = ("L1", "L2", "L4", "L8", "Lname", "Lname+8")
ALL_MAIN = ("L0", *KNOWLEDGE, "Lirrel", "Lswap")
STRUCTURE_CONDITIONS = ("Lname", "L8", "Lname+8")
MIN_ERRORS = 10
"""Conditions with fewer errors are skipped by the replicate-signal model."""


@dataclass(frozen=True)
class Config:
    """Options.

    Attributes:
        datasets: Datasets (one bootstrap generator, in order).
        bootstrap: Bootstrap resamples.
        seed: Bootstrap seed.
        fold_seed: Seed of the grouped cross-validation folds.
    """

    datasets: tuple[str, ...] = DATASETS
    bootstrap: int = 2000
    seed: int = 0
    fold_seed: int = 0


def replicate_features(rows: pd.DataFrame) -> pd.DataFrame:
    """Per main-arm cell: replicate spread and whether replicates disagree on the top label.

    Args:
        rows: Knowledge-dial rows (one per request).

    Returns:
        One row per (item, condition) with ``rep_tvd`` and ``rep_flip``.
    """
    main = rows[rows["arm"] == "main"]
    out = []
    for (item, cond), g in main.groupby(["item", "condition"], sort=False):
        dists = list(g["probs"])
        tops = {max(d, key=d.__getitem__) for d in dists}
        out.append(
            {
                "item": item,
                "condition": cond,
                "rep_tvd": mean_pairwise_tv(dists),
                "rep_flip": len(tops) > 1,
            }
        )
    return pd.DataFrame(out)


def error_detection(
    cells: pd.DataFrame, b: int, rng: np.random.Generator
) -> dict[str, Any]:
    """Error detection and selective prediction per condition.

    Args:
        cells: Aggregated main-arm cells for one dataset.
        b: Bootstrap resamples.
        rng: Random generator (``b`` draws per condition).

    Returns:
        Metrics per condition.
    """
    out: dict[str, Any] = {}
    for cond in ALL_MAIN:
        c = cells[cells["condition"] == cond]
        if c.empty:
            continue
        conf = c["p_max"].to_numpy(float)
        correct = c["correct"].to_numpy(bool)
        err = ~correct
        boot = bootstrap(len(c), on_resample(auroc, 1 - conf, err), b, rng)
        out[cond] = {
            "n": len(c),
            "errors": int(err.sum()),
            "accuracy": float(correct.mean()),
            "auroc_error": auroc(1 - conf, err),
            "auroc_error_ci95": list(percentile_ci(boot)),
            "auprc_error": average_precision(1 - conf, err),
            "error_base_rate": float(err.mean()),
            **risk_coverage(conf, correct),
        }
    return out


def _gain(
    both: np.ndarray, base: np.ndarray, err: np.ndarray
) -> Callable[[np.ndarray], float]:
    """``idx -> AUROC(both) - AUROC(base)`` on the resampled rows."""

    def at(idx: np.ndarray) -> float:
        return auroc(both[idx], err[idx]) - auroc(base[idx], err[idx])

    return at


def _signal_entry(
    c: pd.DataFrame, b: int, rng: np.random.Generator, fold_seed: int
) -> dict[str, Any]:
    """Replicate-signal statistics of one condition with enough errors."""
    err = ~c["correct"].to_numpy(bool)
    conf = c["p_max"].to_numpy(float)
    tvd = c["rep_tvd"].to_numpy(float)
    groups = c["gold"].to_numpy()
    base = cv_logistic_scores((1 - conf)[:, None], err, groups, fold_seed)
    both = cv_logistic_scores(np.column_stack([1 - conf, tvd]), err, groups, fold_seed)
    d = bootstrap(len(c), _gain(both, base, err), b, rng)
    return {
        "errors": int(err.sum()),
        "mean_rep_tvd": float(tvd.mean()),
        "rep_flip_rate": float(c["rep_flip"].mean()),
        "auroc_tvd_alone": auroc(tvd, err),
        "auroc_pmax_cv": auroc(base, err),
        "auroc_pmax_plus_tvd_cv": auroc(both, err),
        "gain": auroc(both, err) - auroc(base, err),
        "gain_ci95": list(percentile_ci(d)),
        "share_gain_le_0": float(np.nanmean(d <= 0)),
        "spearman_tvd_entropy": float(
            pd.Series(tvd).corr(c["norm_entropy"], method="spearman")
        ),
    }


def replicate_signal(
    cells: pd.DataFrame,
    feats: pd.DataFrame,
    b: int,
    rng: np.random.Generator,
    fold_seed: int = 0,
) -> dict[str, Any]:
    """Does replicate disagreement flag errors beyond ``p_max``?

    Args:
        cells: Aggregated main-arm cells for one dataset.
        feats: Output of :func:`replicate_features`.
        b: Bootstrap resamples.
        rng: Random generator.
        fold_seed: Seed of the cross-validation folds.

    Returns:
        Per knowledge condition: AUROCs and the paired gain from adding the
        replicate spread.
    """
    m = cells.merge(feats, on=["item", "condition"])
    out: dict[str, Any] = {}
    for cond in KNOWLEDGE:
        c = m[m["condition"] == cond].reset_index(drop=True)
        n_err = int((~c["correct"].to_numpy(bool)).sum())
        if n_err < MIN_ERRORS:
            out[cond] = {"skipped": f"fewer than {MIN_ERRORS} errors", "errors": n_err}
            continue
        out[cond] = _signal_entry(c, b, rng, fold_seed)
    return out


def knowledge_bins(cells: pd.DataFrame) -> dict[str, list[dict[str, float]]]:
    """Equal-mass reliability bins per knowledge condition.

    Args:
        cells: Aggregated main-arm cells.

    Returns:
        Bins keyed by condition.
    """
    return {
        c: reliability_bins(
            cells[cells["condition"] == c]["p_max"].to_numpy(float),
            cells[cells["condition"] == c]["correct"].to_numpy(bool),
        )
        for c in KNOWLEDGE
    }


def _ensemble_tops(path: Path) -> tuple[dict[str, set[str]], dict[str, str]]:
    """The ensemble's top-two and top labels per in-scope test id."""
    z = np.load(path)
    labels, ids = list(z["labels"]), list(z["eval_ids"])
    mean = decompose(z["probs"])["mean"]
    top2 = {
        i: {labels[j] for j in np.argsort(-mean[r])[:2]}
        for r, i in enumerate(ids)
        if i.startswith("test:")
    }
    top1 = {
        i: labels[int(np.argmax(mean[r]))]
        for r, i in enumerate(ids)
        if i.startswith("test:")
    }
    return top2, top1


def _nan_mean(values: list[Any]) -> float:
    return float(np.mean(values)) if values else float("nan")


def _structure_entry(
    g: pd.DataFrame,
    ens_top2: Mapping[str, set[str]],
    ens_pairs: Counter[tuple[str, str]],
    n_labels: int,
) -> dict[str, Any]:
    """Error structure of one condition."""
    runner_share: list[float] = []
    gold_second: list[bool] = []
    in_ens_top2: list[bool] = []
    pairs: Counter[tuple[str, str]] = Counter()
    for item, grp in g.groupby("item"):
        labs = list(grp.iloc[0]["probs"])
        p = np.mean([[d[lab] for lab in labs] for d in grp["probs"]], axis=0)
        order = np.argsort(-p)
        top, second = labs[order[0]], labs[order[1]]
        gold = grp.iloc[0]["gold"]
        rest = 1 - p[order[0]]
        if rest > 1e-9:
            runner_share.append(p[order[1]] / rest)
        if top != gold:
            gold_second.append(second == gold)
            in_ens_top2.append(top in ens_top2.get(f"test:{item}", set()))
            pairs[(gold, top)] += 1
    # Overlap, not a rank correlation: most pairs occur in only one model's
    # errors, so a correlation over the union of pairs is negative by
    # construction and says nothing about agreement.
    shared = set(pairs) & set(ens_pairs)
    n_err = sum(pairs.values())
    return {
        "jev_errors": int(n_err),
        "mean_runner_up_share_of_residual": _nan_mean(runner_share),
        "errors_gold_is_second_choice": _nan_mean(gold_second),
        "errors_in_ensemble_top2": _nan_mean(in_ens_top2),
        "chance_in_ensemble_top2": 2 / (n_labels - 1),
        "distinct_pairs_jev": len(pairs),
        "distinct_pairs_ensemble": len(ens_pairs),
        "shared_pairs": len(shared),
        "jev_errors_on_pairs_ensemble_also_confuses": (
            sum(pairs[p] for p in shared) / n_err if n_err else float("nan")
        ),
        "top_confused_pairs": [
            {"gold": a, "chosen": c, "jev": n, "ensemble": ens_pairs[(a, c)]}
            for (a, c), n in pairs.most_common(8)
        ],
    }


def error_structure(
    rows: pd.DataFrame, ens_path: Path, gold_of: Mapping[int, str], n_labels: int
) -> dict[str, Any]:
    """Where the API's errors concentrate, compared with the full-data ensemble.

    Args:
        rows: Knowledge-dial rows (for replicate-mean distributions).
        ens_path: Full-data ensemble predictions.
        gold_of: Gold intent per test index.
        n_labels: Number of intents (for the chance baseline).

    Returns:
        Per condition: runner-up concentration, ensemble agreement on wrong
        answers and the most confused pairs.
    """
    ens_top2, ens_top1 = _ensemble_tops(ens_path)
    ens_pairs = Counter(
        (gold_of[int(i[5:])], t)
        for i, t in ens_top1.items()
        if t != gold_of[int(i[5:])]
    )
    main = rows[rows["arm"] == "main"]
    return {
        cond: _structure_entry(
            main[main["condition"] == cond], ens_top2, ens_pairs, n_labels
        )
        for cond in STRUCTURE_CONDITIONS
    }


def analyse_dataset(
    settings: Settings, name: str, config: Config, rng: np.random.Generator
) -> dict[str, Any]:
    """All analyses of one dataset.

    Args:
        settings: Run settings (locates the inputs).
        name: Dataset name.
        config: Options.
        rng: Random generator.

    Returns:
        JSON-serialisable results.
    """
    rows = pd.DataFrame(read_jsonl(rows_path(settings, "test", name)))
    cells = aggregate(rows)
    cells = cells[cells["arm"] == "main"].reset_index(drop=True)
    ds = load(name, settings.data_dir)
    detection = error_detection(cells, config.bootstrap, rng)
    signal = replicate_signal(
        cells, replicate_features(rows), config.bootstrap, rng, config.fold_seed
    )
    return {
        "error_detection": detection,
        "replicate_signal": signal,
        "reliability_bins": knowledge_bins(cells),
        "error_structure": error_structure(
            rows,
            predictions_path(settings, name, "full"),
            {i: ex.label for i, ex in enumerate(ds.test)},
            len(ds.labels),
        ),
    }


def run(settings: Settings, config: Config) -> dict[str, Any]:
    """Run every analysis and write the summary.

    Args:
        settings: Run settings.
        config: Options.

    Returns:
        Results keyed by dataset.
    """
    rng = np.random.default_rng(config.seed)
    results = {
        name: analyse_dataset(settings, name, config, rng) for name in config.datasets
    }
    write_json(
        settings.out(NAME) / "summary.json", {"config": as_dict(config), **results}
    )
    return results


ANALYSIS = Analysis(
    name=NAME,
    summary="Error detection, replicate disagreement and error structure",
    config_type=Config,
    run=run,
)
