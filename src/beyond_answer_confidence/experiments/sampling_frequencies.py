"""Sampling frequencies: are the probabilities counts over a fixed number of samples?

The instrument checks found probabilities quantised to 0.01 and run-to-run
noise. If each probability is a frequency over N parallel samples, then
across replicates of one request each option's probability has variance
about ``p(1 - p) / N`` with the same N everywhere, and options compete
(negative covariance between the top two).

Development items of Banking77 only. Output:
``<output_dir>/sampling_frequencies/summary.json``.
"""

import asyncio
import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from beyond_answer_confidence.backends.cache import CachedClient, CallRecord
from beyond_answer_confidence.config import as_dict
from beyond_answer_confidence.data.intents import IntentDataset, load, spread_indices
from beyond_answer_confidence.experiments.base import (
    Experiment,
    backend_for,
    write_json,
)
from beyond_answer_confidence.runner import collect
from beyond_answer_confidence.settings import Settings
from beyond_answer_confidence.tasks.intents import (
    QUESTION,
    Condition,
    build_choice,
    choice_request,
    sample_examples,
)
from beyond_answer_confidence.tasks.schema import Task

NAME = "sampling_frequencies"
CACHE = "intents"
MID = (0.05, 0.95)
"""Option means inside this range enter the pooled sample-size estimate."""
REFERENCE_N = 100
"""Sample count whose binomial spread is reported for comparison."""


@dataclass(frozen=True)
class Config:
    """Sizes.

    Attributes:
        replicates: Replicates per series.
        grid_items: Size of the evenly spread dev-item grid.
        name_positions: Grid positions asked with intent names (items whose
            top probability was mid-range in the instrument checks).
        blank_positions: Grid positions asked with no information.
    """

    replicates: int = 30
    grid_items: int = 10
    name_positions: tuple[int, ...] = (3, 4, 6, 7, 8)
    blank_positions: tuple[int, ...] = (0, 1, 2, 3, 4)


def option_stats(replicates: Sequence[dict[str, float]]) -> list[dict[str, float]]:
    """Mean and sample variance of each option's probability across replicates.

    Args:
        replicates: One distribution per replicate, same options.

    Returns:
        Per option: ``mean``, ``var`` (ddof = 1). Options with mean 0 are skipped.
    """
    out = []
    for opt in replicates[0]:
        xs = [r[opt] for r in replicates]
        mean = statistics.fmean(xs)
        if mean > 0:
            out.append({"mean": mean, "var": statistics.variance(xs)})
    return out


def pooled_n(
    stats: Sequence[dict[str, float]], lo: float = MID[0], hi: float = MID[1]
) -> float:
    """Pooled effective sample size ``sum p(1-p) / sum var`` over mid-range options.

    Args:
        stats: Output of :func:`option_stats` (possibly concatenated).
        lo: Lower bound on the option mean.
        hi: Upper bound on the option mean.

    Returns:
        The estimate (``inf`` without variance, ``nan`` without mid-range options).
    """
    mids = [s for s in stats if lo <= s["mean"] <= hi]
    if not mids:
        return math.nan
    num = sum(s["mean"] * (1 - s["mean"]) for s in mids)
    den = sum(s["var"] for s in mids)
    return math.inf if den == 0 else num / den


def top2_correlation(replicates: Sequence[dict[str, float]]) -> float:
    """Pearson correlation across replicates between the two highest-mean options.

    Args:
        replicates: One distribution per replicate.

    Returns:
        The correlation, or ``nan`` if either option is constant.
    """
    means = {o: statistics.fmean(r[o] for r in replicates) for o in replicates[0]}
    a, b = sorted(means, key=means.__getitem__, reverse=True)[:2]
    xs, ys = [r[a] for r in replicates], [r[b] for r in replicates]
    try:
        return statistics.correlation(xs, ys)
    except statistics.StatisticsError:
        return math.nan


def build_tasks(ds: IntentDataset, config: Config) -> list[Task]:
    """Every (series, replicate) request; the unit is the series name.

    Args:
        ds: Banking77.
        config: Sizes.

    Returns:
        Tasks, series by series.
    """
    examples = sample_examples(ds, seed=0)
    grid = spread_indices(len(ds.dev), config.grid_items)
    series = [
        *((Condition.NAME, pos) for pos in config.name_positions),
        *((Condition.L0, pos) for pos in config.blank_positions),
    ]
    tasks = []
    for cond, pos in series:
        idx = grid[pos]
        spec = build_choice(ds.dev[idx], f"{ds.name}:dev:{idx}", ds, cond, examples, [])
        unit = f"{cond.value}/dev{idx}"
        tasks += [
            Task(unit, choice_request(ds.dev[idx], spec, rep))
            for rep in range(config.replicates)
        ]
    return tasks


def analyse(series: dict[str, list[dict[str, float]]]) -> dict[str, Any]:
    """Replicate variability per series and pooled.

    Args:
        series: Series name -> one code distribution per answered replicate.

    Returns:
        JSON-serialisable summary.
    """
    per_series: dict[str, Any] = {}
    all_stats: list[dict[str, float]] = []
    sums = [sum(p.values()) for reps in series.values() for p in reps]
    for name, reps in series.items():
        st = option_stats(reps)
        all_stats.extend(st)
        top = max(st, key=lambda s: s["mean"])
        per_series[name] = {
            "replicates": len(reps),
            "top_mean": round(top["mean"], 4),
            "top_sd": round(math.sqrt(top["var"]), 4),
            "binomial_sd_if_N100": round(
                math.sqrt(top["mean"] * (1 - top["mean"]) / REFERENCE_N), 4
            ),
            "pooled_N": pooled_n(st),
            "top2_corr": top2_correlation(reps),
            "distinct_argmax": len({max(r, key=r.__getitem__) for r in reps}),
        }
    mids = [s for s in all_stats if MID[0] <= s["mean"] <= MID[1]]
    ratios = [s["var"] / (s["mean"] * (1 - s["mean"]) / REFERENCE_N) for s in mids]
    return {
        "per_series": per_series,
        "pooled_N_all": pooled_n(all_stats),
        "n_mid_options": len(mids),
        "var_ratio_vs_N100": {
            "median": statistics.median(ratios) if ratios else math.nan,
            "min": min(ratios, default=math.nan),
            "max": max(ratios, default=math.nan),
        },
        "sum_range": [min(sums), max(sums)],
    }


def series_of(
    tasks: Sequence[Task], records: dict[str, list[CallRecord]]
) -> dict[str, list[dict[str, float]]]:
    """Code distributions per series from collected records.

    Args:
        tasks: The tasks (for the series order).
        records: Unit -> answered records.

    Returns:
        Series name -> distributions, in replicate order.
    """
    names = list(dict.fromkeys(t.unit for t in tasks))
    return {
        n: [dict(r.response.choices[QUESTION].probabilities) for r in records[n]]
        for n in names
        if records.get(n)
    }


def requests(settings: Settings, config: Config) -> list[Task]:
    """Every request.

    Args:
        settings: Run settings.
        config: Sizes.

    Returns:
        The tasks.
    """
    return build_tasks(load("banking77", settings.data_dir), config)


def run(settings: Settings, config: Config) -> dict[str, Any]:
    """Collect, analyse and write the summary.

    Args:
        settings: Run settings.
        config: Sizes.

    Returns:
        The summary.
    """
    tasks = requests(settings, config)
    client = CachedClient(
        settings.cache_file(CACHE),
        model=settings.model,
        max_input_tokens=settings.max_input_tokens,
    )
    got = asyncio.run(
        collect(tasks, client, backend_for(settings), concurrency=settings.concurrency)
    )
    summary = {
        "config": as_dict(config),
        **analyse(series_of(tasks, got.records)),
        "errors": sum(got.errors.values()),
        "error_types": sorted(got.errors),
        "response_models": sorted(
            {r.response.model for recs in got.records.values() for r in recs}
        ),
        "new_api_calls": client.api_calls,
        "new_input_tokens": client.spent_input_tokens,
    }
    write_json(settings.out(NAME) / "summary.json", summary)
    return summary


EXPERIMENT = Experiment(
    name=NAME,
    summary="Whether replicate noise matches frequencies over a fixed sample count",
    config_type=Config,
    requests=requests,
    run=run,
)
