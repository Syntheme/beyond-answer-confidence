"""Stated odds: do choice probabilities match the odds of a named random device?

Dice, coins and cards with known outcome probabilities (see
:mod:`beyond_answer_confidence.tasks.chance`), each asked once per option order. The
analysis reports the total variation between the order-averaged
distribution and the true odds, how it shrinks as more orders are
averaged, the mean probability by option position, and, on re-listed
synthetic tie cells, whether the top choice follows the option listed first
in the facts or shown first in the options.
"""

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from beyond_answer_confidence.backends.cache import CachedClient
from beyond_answer_confidence.config import as_dict
from beyond_answer_confidence.experiments.base import (
    Experiment,
    backend_for,
    write_json,
    write_jsonl,
)
from beyond_answer_confidence.metrics.distributions import tv
from beyond_answer_confidence.runner import run_collect, run_summary
from beyond_answer_confidence.settings import Settings
from beyond_answer_confidence.stats.inference import (
    holm_adjust_tails,
    mean_interval_lower,
)
from beyond_answer_confidence.stats.resampling import bootstrap, percentile_ci
from beyond_answer_confidence.tasks.chance import (
    DEVICES,
    RELIST_PROFILES,
    device_items,
    device_tasks,
    relisted_items,
)
from beyond_answer_confidence.tasks.schema import Item, Task, item_tasks
from beyond_answer_confidence.tasks.scoring import score_items

logger = logging.getLogger(__name__)

NAME = "stated_odds"


@dataclass(frozen=True)
class Config:
    """Options of the stated-odds experiment.

    Attributes:
        relist_replicates: Replicates per re-listed synthetic item.
        n_scenarios: Synthetic scenarios scanned for tie cells.
        seed: Scenario and re-listing seed.
        bootstrap: Bootstrap resamples.
        bootstrap_seed: Bootstrap seed.
        tv_reference: Reference total variation from the true odds.
        averaging_sizes: Numbers of orders averaged in the averaging curve
            (all orders are always added).
        averaging_draws: Random subsets per size in the averaging curve.
    """

    relist_replicates: int = 3
    n_scenarios: int = 480
    seed: int = 0
    bootstrap: int = 2000
    bootstrap_seed: int = 0
    tv_reference: float = 0.10
    averaging_sizes: tuple[int, ...] = (1, 2, 4, 8, 16, 64)
    averaging_draws: int = 200


def build(config: Config) -> tuple[list[tuple[Item, int]], list[Item], list[Task]]:
    """Device items, re-listed items and every task.

    Args:
        config: Options.

    Returns:
        ``(devices, relisted, tasks)``; device tasks first.
    """
    devices = device_items()
    relisted = relisted_items(config.n_scenarios, config.seed)
    tasks = device_tasks(devices) + item_tasks(relisted, config.relist_replicates)
    return devices, relisted, tasks


def requests(settings: Settings, config: Config) -> list[Task]:
    """Every task of the experiment.

    Args:
        settings: Run settings (unused; the items are generated).
        config: Options.

    Returns:
        The tasks.
    """
    del settings
    return build(config)[2]


def tv_of_mean(
    dists: Sequence[Mapping[str, float]], ideal: Mapping[str, float]
) -> float:
    """Total variation between the mean of several distributions and the ideal.

    Args:
        dists: Distributions (each covering the ideal's options).
        ideal: True odds.

    Returns:
        The distance.
    """
    mean = {o: float(np.mean([d[o] for d in dists])) for o in ideal}
    return tv(mean, ideal)


def tv_of_mean_interval(
    dists: Sequence[Mapping[str, float]],
    ideal: Mapping[str, float],
    b: int,
    rng: np.random.Generator,
    threshold: float,
) -> dict[str, Any]:
    """Order-averaged total variation with a bootstrap over orders.

    Args:
        dists: One distribution per option order.
        ideal: True odds.
        b: Resamples (one ``rng.integers`` call each).
        rng: Random generator.
        threshold: Reference distance.

    Returns:
        ``n_orders``, ``tv``, 95 % ``ci``, ``mean_distribution``,
        ``threshold`` and ``tail`` (share of replicates at or above it).
    """
    boot = bootstrap(
        len(dists), lambda idx: tv_of_mean([dists[i] for i in idx], ideal), b, rng
    )
    return {
        "n_orders": len(dists),
        "tv": tv_of_mean(dists, ideal),
        "ci": list(percentile_ci(boot)),
        "mean_distribution": {o: float(np.mean([d[o] for d in dists])) for o in ideal},
        "threshold": threshold,
        "tail": float(np.mean(boot >= threshold)),
    }


def analyse(rows: list[dict[str, Any]], config: Config) -> dict[str, Any]:
    """Distance from the true odds, position effects, listed versus shown first.

    Bootstrap draws (one generator, in this order): fair die with number
    words, fair die with colours, listed-first minus shown-first share;
    then the random order subsets of the averaging curve, device by device.

    Args:
        rows: Scored rows.
        config: Options.

    Returns:
        The three effect sizes with intervals (Holm-adjusted tail shares),
        per-device summaries and per-profile re-listing shares.
    """
    b = config.bootstrap
    rng = np.random.default_rng(config.bootstrap_seed)
    by: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        if r["set"] == "device":
            by.setdefault(r["device"], []).append(r)

    def device_tv(name: str) -> dict[str, Any]:
        return tv_of_mean_interval(
            [r["dist"] for r in by[name]], DEVICES[name][1], b, rng, config.tv_reference
        )

    fair_words = device_tv("die_words")
    fair_colours = device_tv("die_colours")
    rel = [r for r in rows if r["set"] == "relist"]
    diff = np.array(
        [
            float(r["top"] == r["first_listed"]) - float(r["top"] == r["first_shown"])
            for r in rel
        ]
    )
    listed = {
        "share_top_is_first_listed": float(
            np.mean([r["top"] == r["first_listed"] for r in rel])
        ),
        "share_top_is_first_shown": float(
            np.mean([r["top"] == r["first_shown"] for r in rel])
        ),
        **mean_interval_lower(diff, b, rng, 0.0),
    }
    res: dict[str, Any] = holm_adjust_tails(
        {
            "fair_die_words": fair_words,
            "fair_die_colours": fair_colours,
            "listed_first_minus_shown_first": listed,
        }
    )
    res["devices"] = {
        name: _device_summary(rs, DEVICES[name][1], config, rng)
        for name, rs in by.items()
    }
    res["relist_by_profile"] = {
        prof: {
            "n": len([r for r in rel if r["profile"] == prof]),
            "share_first_listed": float(
                np.mean(
                    [r["top"] == r["first_listed"] for r in rel if r["profile"] == prof]
                )
            ),
            "share_first_shown": float(
                np.mean(
                    [r["top"] == r["first_shown"] for r in rel if r["profile"] == prof]
                )
            ),
            "mean_p_max": float(
                np.mean([r["p_max"] for r in rel if r["profile"] == prof])
            ),
        }
        for prof in RELIST_PROFILES
    }
    return res


def _device_summary(
    rs: Sequence[Mapping[str, Any]],
    ideal: Mapping[str, float],
    config: Config,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Descriptive summary of one device, including the averaging curve."""
    dists = [r["dist"] for r in rs]
    labels = list(ideal)
    by_pos = [
        float(np.mean([r["dist"][r["order"][p]] for r in rs]))
        for p in range(len(labels))
    ]
    curve = {}
    for k in (*config.averaging_sizes, len(rs)):
        if k > len(rs):
            continue
        draws = config.averaging_draws if k < len(rs) else 1
        curve[str(k)] = float(
            np.mean(
                [
                    tv_of_mean(
                        [dists[i] for i in rng.choice(len(rs), k, replace=False)],
                        ideal,
                    )
                    for _ in range(draws)
                ]
            )
        )
    return {
        "n_calls": len(rs),
        "tv_of_all_order_mean": tv_of_mean(dists, ideal),
        "mean_single_call_tv": float(np.mean([tv(d, ideal) for d in dists])),
        "mean_p_max": float(np.mean([r["p_max"] for r in rs])),
        "mean_probability_by_label": {
            o: float(np.mean([d[o] for d in dists])) for o in labels
        },
        "mean_probability_by_position": by_pos,
        "top_label_counts": {o: int(sum(r["top"] == o for r in rs)) for o in labels},
        "permutation_averaging_tv": curve,
    }


def run(settings: Settings, config: Config) -> dict[str, Any]:
    """Collect (cache first), score, analyse and write the outputs.

    Args:
        settings: Run settings.
        config: Options.

    Returns:
        The summary (``results`` only when every item is complete).
    """
    devices, relisted, tasks = build(config)
    client = CachedClient(
        settings.cache_file(NAME),
        model=settings.model,
        max_input_tokens=settings.max_input_tokens,
    )
    collected = run_collect(
        tasks,
        client,
        backend=backend_for(settings),
        concurrency=settings.concurrency,
        label=NAME,
    )
    expected = {it.unit: reps for it, reps in devices} | {
        it.unit: config.relist_replicates for it in relisted
    }
    rows = score_items(
        [it for it, _ in devices] + relisted, collected.complete(expected)
    )
    out = settings.out(NAME)
    write_jsonl(out / "rows.jsonl", rows)
    summary: dict[str, Any] = {
        "config": as_dict(config),
        "items": len(expected),
        "complete_items": len(rows),
        **run_summary(client, collected),
    }
    if len(rows) == len(expected):
        summary["results"] = analyse(rows, config)
    else:
        logger.warning("%s: incomplete items; no analysis", NAME)
    write_json(out / "summary.json", summary)
    return summary


EXPERIMENT = Experiment(
    name=NAME,
    summary="Dice, coins and cards: choice probabilities versus the true odds",
    config_type=Config,
    requests=requests,
    run=run,
)
