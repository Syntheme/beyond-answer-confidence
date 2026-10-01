"""Out-of-scope detection with three readouts (see :mod:`beyond_answer_confidence.tasks.out_of_scope`).

Metrics per dataset (out-of-scope is the positive class): AUROC and the
false-positive rate at 95 % true-positive rate for each score, in-scope
accuracy, out-of-scope recall; item-bootstrap 95 % intervals. Replicates are
averaged per item.

Outputs under ``<output_dir>/out_of_scope/``: ``<dataset>_items.jsonl`` (one
scored row per item) and ``summary.json``.
"""

import asyncio
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from beyond_answer_confidence.backends.cache import CachedClient
from beyond_answer_confidence.config import as_dict
from beyond_answer_confidence.data.intents import DATASETS, IntentDataset, load
from beyond_answer_confidence.experiments.base import (
    Experiment,
    backend_for,
    write_json,
    write_jsonl,
)
from beyond_answer_confidence.metrics.ranking import auroc, fpr_at_tpr
from beyond_answer_confidence.runner import collect
from beyond_answer_confidence.settings import Settings
from beyond_answer_confidence.tasks.out_of_scope import (
    OOS_OPTION,
    ScopeItem,
    ScopeQuestions,
    build_items,
    heldout_intents,
    questions_for,
    scope_request,
    score_item,
)
from beyond_answer_confidence.tasks.schema import Task

logger = logging.getLogger(__name__)

NAME = "out_of_scope"
CACHE = "intents"


@dataclass(frozen=True)
class Config:
    """Out-of-scope options.

    Attributes:
        datasets: Datasets to run.
        per_intent: In-scope items per schema intent.
        heldout_intents: Intents held out of the schema when a dataset has
            no out-of-scope split.
        replicates: Replicates per item.
        bootstrap: Bootstrap resamples.
        seed: Bootstrap seed (a fresh generator per dataset).
        chunk_size: Requests per chunk.
    """

    datasets: tuple[str, ...] = DATASETS
    per_intent: int = 10
    heldout_intents: int = 10
    replicates: int = 3
    bootstrap: int = 2000
    seed: int = 0
    chunk_size: int = 1500


SCORES = (
    "closed_1_minus_p_max",
    "closed_norm_entropy",
    "with_oos_p_oos",
    "fits_1_minus_p",
)
"""Out-of-scope scores (higher = more likely out of scope)."""


def score_value(name: str, row: Mapping[str, Any]) -> float:
    """One out-of-scope score of a scored item.

    Args:
        name: One of :data:`SCORES`.
        row: Scored item.

    Returns:
        The score.

    Raises:
        ValueError: For an unknown score name.
    """
    if name == "closed_1_minus_p_max":
        return float(1 - row["closed_p_max"])
    if name == "closed_norm_entropy":
        return float(row["closed_norm_entropy"])
    if name == "with_oos_p_oos":
        return float(row["p_oos"])
    if name == "fits_1_minus_p":
        return float(1 - row["p_fits"])
    raise ValueError(f"unknown score {name!r}")


def _rate(flags: Sequence[bool]) -> float:
    return float(np.mean(flags))


def rates(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Counts, accuracies and recall rates (no resampling).

    Args:
        rows: Scored items.

    Returns:
        Item counts, in-scope accuracy under both choices, and out-of-scope
        and false out-of-scope rates of the open choice and of ``P(fits) < 0.5``.
    """
    in_rows = [r for r in rows if not r["oos"]]
    oos_rows = [r for r in rows if r["oos"]]
    return {
        "n_in_scope": len(in_rows),
        "n_oos": len(oos_rows),
        "in_scope_accuracy_closed": _rate(
            [r["closed_choice"] == r["gold"] for r in in_rows]
        ),
        "in_scope_accuracy_with_oos": _rate(
            [r["with_oos_choice"] == r["gold"] for r in in_rows]
        ),
        "oos_recall_with_oos": _rate(
            [r["with_oos_choice"] == OOS_OPTION for r in oos_rows]
        ),
        "in_scope_false_oos_with_oos": _rate(
            [r["with_oos_choice"] == OOS_OPTION for r in in_rows]
        ),
        "oos_recall_fits_below_half": _rate([r["p_fits"] < 0.5 for r in oos_rows]),
        "in_scope_false_oos_fits_below_half": _rate(
            [r["p_fits"] < 0.5 for r in in_rows]
        ),
    }


def detection(
    score: np.ndarray, oos: np.ndarray, b: int, rng: np.random.Generator
) -> dict[str, Any]:
    """AUROC and FPR at 95 % TPR with item-bootstrap intervals.

    Draw order: ``b`` resamples of ``n`` items; resamples with a single
    class are skipped.

    Args:
        score: Higher = more likely out of scope.
        oos: Out-of-scope labels.
        b: Resamples.
        rng: Random generator.

    Returns:
        ``auroc``, ``auroc_ci95``, ``fpr_at_95tpr`` and ``fpr_at_95tpr_ci95``.
    """
    n = len(score)
    boots_auc, boots_fpr = [], []
    for _ in range(b):
        idx = rng.integers(0, n, n)
        if 0 < oos[idx].sum() < n:
            boots_auc.append(auroc(score[idx], oos[idx]))
            boots_fpr.append(fpr_at_tpr(score[idx], oos[idx]))
    return {
        "auroc": auroc(score, oos),
        "auroc_ci95": [float(x) for x in np.percentile(boots_auc, [2.5, 97.5])],
        "fpr_at_95tpr": fpr_at_tpr(score, oos),
        "fpr_at_95tpr_ci95": [float(x) for x in np.percentile(boots_fpr, [2.5, 97.5])],
    }


def metrics(rows: Sequence[Mapping[str, Any]], b: int, seed: int = 0) -> dict[str, Any]:
    """All metrics of one dataset.

    Args:
        rows: Scored items (with ``oos`` and ``gold``).
        b: Bootstrap resamples.
        seed: Bootstrap seed (a fresh generator).

    Returns:
        Rates and, per score, detection metrics.
    """
    rng = np.random.default_rng(seed)
    oos = np.array([r["oos"] for r in rows])
    out = rates(rows)
    for name in SCORES:
        out[name] = detection(
            np.array([score_value(name, r) for r in rows]), oos, b, rng
        )
    return out


def dataset_plan(
    ds: IntentDataset, config: Config
) -> tuple[list[str], list[tuple[ScopeItem, ScopeQuestions]]]:
    """Schema and items with their questions for one dataset.

    Args:
        ds: The dataset.
        config: Options.

    Returns:
        ``(schema, [(item, questions), ...])``.
    """
    schema, items = build_items(ds, config.per_intent, config.heldout_intents)
    return schema, [(it, questions_for(it, schema)) for it in items]


def plan_tasks(
    built: Sequence[tuple[ScopeItem, ScopeQuestions]], replicates: int
) -> list[Task]:
    """Tasks item by item (replicates of an item are adjacent).

    Args:
        built: Items and their questions.
        replicates: Replicates per item.

    Returns:
        The tasks; the unit is the item key.
    """
    return [
        Task(it.key, scope_request(it, q, r))
        for it, q in built
        for r in range(replicates)
    ]


def requests(settings: Settings, config: Config) -> list[Task]:
    """Every request.

    Args:
        settings: Run settings.
        config: Options.

    Returns:
        The tasks.
    """
    tasks: list[Task] = []
    for name in config.datasets:
        _, built = dataset_plan(load(name, settings.data_dir), config)
        tasks += plan_tasks(built, config.replicates)
    return tasks


def _run_dataset(
    ds: IntentDataset,
    client: CachedClient,
    settings: Settings,
    config: Config,
    out: Path,
) -> tuple[dict[str, Any], dict[str, int], str | None]:
    """Collect, score and summarise one dataset (plus errors and any fatal error)."""
    schema, built = dataset_plan(ds, config)
    got = asyncio.run(
        collect(
            plan_tasks(built, config.replicates),
            client,
            backend_for(settings),
            concurrency=settings.concurrency,
            chunk_size=config.chunk_size,
            label=ds.name,
        )
    )
    rows = [
        {"key": it.key, "oos": it.oos, "gold": it.gold, **score_item(recs, q)}
        for it, q in built
        if len(recs := got.records.get(it.key, [])) == config.replicates
    ]
    write_jsonl(out / f"{ds.name}_items.jsonl", rows)
    meta: dict[str, Any] = {
        "schema_size": len(schema),
        "items": len(built),
        "complete_items": len(rows),
    }
    if not ds.oos_test:
        meta["heldout_intents"] = heldout_intents(ds.labels, config.heldout_intents)
    meta["metrics"] = metrics(rows, config.bootstrap, config.seed)
    meta["models"] = sorted({m for r in rows for m in r["models"]})
    return meta, dict(got.errors), got.fatal


def run(settings: Settings, config: Config) -> dict[str, Any]:
    """Collect, score, analyse and write the summary.

    Args:
        settings: Run settings.
        config: Options.

    Returns:
        The summary.
    """
    client = CachedClient(
        settings.cache_file(CACHE),
        model=settings.model,
        max_input_tokens=settings.max_input_tokens,
    )
    out = settings.out(NAME)
    summary: dict[str, Any] = {"config": as_dict(config)}
    errors: dict[str, int] = {}
    fatal: str | None = None
    for name in config.datasets:
        summary[name], errs, fatal = _run_dataset(
            load(name, settings.data_dir), client, settings, config, out
        )
        for k, v in errs.items():
            errors[k] = errors.get(k, 0) + v
        if fatal is not None:
            logger.warning("stopping after a fatal API error")
            break
    summary |= {
        "errors": errors,
        "fatal": fatal,
        "new_api_calls": client.api_calls,
        "new_input_tokens": client.spent_input_tokens,
    }
    write_json(out / "summary.json", summary)
    return summary


EXPERIMENT = Experiment(
    name=NAME,
    summary="Detecting messages that fit none of the listed intents",
    config_type=Config,
    requests=requests,
    run=run,
)
