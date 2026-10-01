"""Do bootstrap upper bounds for SmoothECE cover the truth? A simulation.

Resampling with replacement inflates SmoothECE, and recentring the
bootstrap on the estimate shifts it down, which lowers exactly the upper
bound used to call a set calibrated; an invalid recentring would err
towards declaring calibration. This simulation measures how often each
scheme in :func:`beyond_answer_confidence.stats.smece_bounds.smece_bounds` covers the
true value.

Design: confidences are drawn with replacement from real confidence pools
(so quantisation and ties are realistic), at the size of the real set.
Correctness is Bernoulli(g(p)) with an overconfidence shift
``g(p) = clip(p - delta, 0, 1)``; ``delta`` is chosen by bisection so that
the population SmoothECE equals each target (0, 0.03, 0.05, 0.07). The
population value is SmoothECE with the true P(correct) in place of labels
on a large draw. For each condition the output gives, per scheme, the share
of simulations whose one-sided 95 % upper bound is at or above the truth
(nominally >= 0.95), the two-sided coverage, and the share whose upper
bound is below the reference value (at a true value equal to the reference
this is the rate of wrongly calling the set calibrated; nominally <= 0.05).

Pools: knowledge-dial Banking77 L8 and CLINC150 L1 (main arm), Quizbowl
first clue and TriviaQA.

Output: ``<output_dir>/smece_coverage/summary.json``.
"""

import os
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from beyond_answer_confidence.analyses.knowledge_dial import aggregate
from beyond_answer_confidence.analyses.rows import experiment_frame
from beyond_answer_confidence.config import as_dict
from beyond_answer_confidence.experiments.base import Analysis, read_jsonl, write_json
from beyond_answer_confidence.experiments.knowledge_dial import rows_path
from beyond_answer_confidence.metrics.calibration import smece
from beyond_answer_confidence.settings import Settings
from beyond_answer_confidence.stats.resampling import smece_resamples
from beyond_answer_confidence.stats.smece_bounds import SMECE_METHODS, smece_bounds

NAME = "smece_coverage"
KNOWLEDGE_DIAL_POOLS = (("banking77", "L8"), ("clinc150", "L1"))
"""Knowledge-dial (dataset, condition) confidence pools."""

_POOLS: dict[str, np.ndarray] = {}
"""Worker-process copy of the pools (set by :func:`_init`)."""


@dataclass(frozen=True)
class Config:
    """Options.

    Attributes:
        sims: Simulated datasets per condition.
        resamples: Resamples per scheme and simulation.
        seed: Seed (bisection draws and per-simulation seeds).
        workers: Worker processes (0: one fewer than the CPUs; 1: run in
            this process).
        population_draws: Draws for the population SmoothECE.
        targets: Target population SmoothECE values.
        reference: Reference value for the share of upper bounds below it.
        alpha: One-sided error rate of the bounds.
    """

    sims: int = 300
    resamples: int = 500
    seed: int = 0
    workers: int = 0
    population_draws: int = 200_000
    targets: tuple[float, ...] = (0.0, 0.03, 0.05, 0.07)
    reference: float = 0.05
    alpha: float = 0.05


def load_pools(settings: Settings) -> dict[str, np.ndarray]:
    """Real top-label confidences for the near-reference calibration sets.

    Args:
        settings: Run settings (locate the outputs).

    Returns:
        Pool name -> confidences (one per analysis unit).
    """
    pools: dict[str, np.ndarray] = {}
    for ds, cond in KNOWLEDGE_DIAL_POOLS:
        cells = aggregate(pd.DataFrame(read_jsonl(rows_path(settings, "test", ds))))
        c = cells[(cells["arm"] == "main") & (cells["condition"] == cond)]
        pools[f"{ds}_{cond}"] = c["p_max"].to_numpy(float)
    ev = experiment_frame(settings, "evidence_sufficiency")
    qb = ev[(ev["set"] == "quizbowl") & (ev["k"] == 1)]
    pools["quizbowl_first"] = qb["p_max"].to_numpy(float)
    bm = experiment_frame(settings, "benchmarks")
    pools["triviaqa"] = bm[bm["set"] == "triviaqa"]["p_max"].to_numpy(float)
    return pools


def delta_for(
    pool: np.ndarray,
    target: float,
    rng: np.random.Generator,
    population: int = 200_000,
) -> tuple[float, float]:
    """Overconfidence shift whose population SmoothECE matches a target.

    Draws the population first (also for a target of 0), then bisects
    ``delta`` in ``[0, 0.5]`` 30 times.

    Args:
        pool: Confidence pool.
        target: Target population SmoothECE.
        rng: Random generator.
        population: Population draws.

    Returns:
        ``(delta, population SmoothECE at delta)``.
    """
    p = rng.choice(pool, population)
    if target == 0:
        return 0.0, 0.0
    lo, hi = 0.0, 0.5
    for _ in range(30):
        mid = (lo + hi) / 2
        if smece(p, np.clip(p - mid, 0, 1)) < target:
            lo = mid
        else:
            hi = mid
    d = (lo + hi) / 2
    return d, smece(p, np.clip(p - d, 0, 1))


def _init(pools: dict[str, np.ndarray]) -> None:
    _POOLS.update(pools)


def simulate_once(
    pool: np.ndarray, delta: float, n: int, b: int, seed: int, alpha: float = 0.05
) -> dict[str, Any]:
    """Simulate one dataset and compute every scheme's bounds.

    Args:
        pool: Confidence pool.
        delta: Overconfidence shift.
        n: Dataset size.
        b: Resamples per scheme.
        seed: Seed of this simulation.
        alpha: One-sided error rate.

    Returns:
        The estimate and per-scheme bounds.
    """
    rng = np.random.default_rng(seed)
    p = rng.choice(pool, n)
    y = (rng.random(n) < np.clip(p - delta, 0, 1)).astype(float)
    est = smece(p, y)
    boot = smece_resamples(p, y, b, rng, subsample=False)
    sub = smece_resamples(p, y, b, rng, subsample=True)
    return {"est": est, "bounds": smece_bounds(est, n, boot, sub, alpha)}


def _job(args: tuple[str, float, int, int, int, float]) -> dict[str, Any]:
    name, delta, n, b, seed, alpha = args
    return simulate_once(_POOLS[name], delta, n, b, seed, alpha)


def summarise(
    truth: float, runs: Sequence[dict[str, Any]], reference: float = 0.05
) -> dict[str, Any]:
    """Coverage over the simulations of one condition.

    Args:
        truth: Population SmoothECE.
        runs: Output of :func:`simulate_once` per simulation.
        reference: Reference value.

    Returns:
        Mean estimate and bias, and per scheme: one-sided upper coverage,
        two-sided coverage, the share of upper bounds below ``reference``
        and the mean upper bound.
    """
    est = np.array([r["est"] for r in runs])
    out: dict[str, Any] = {
        "truth": truth,
        "sims": len(runs),
        "mean_estimate": float(est.mean()),
        "estimator_bias": float(est.mean() - truth),
        "methods": {},
    }
    for m in SMECE_METHODS:
        up = np.array([r["bounds"][m]["upper"] for r in runs])
        lo = np.array([r["bounds"][m]["low"] for r in runs])
        out["methods"][m] = {
            "upper_coverage": float(np.mean(up >= truth)),
            "two_sided_coverage": float(np.mean((lo <= truth) & (up >= truth))),
            "share_upper_below_reference": float(np.mean(up < reference)),
            "mean_upper": float(up.mean()),
        }
    return out


def simulate(pools: dict[str, np.ndarray], config: Config) -> dict[str, Any]:
    """Run the whole simulation on given pools.

    Args:
        pools: Pool name -> confidences.
        config: Options.

    Returns:
        ``design`` and one entry per (pool, target) in ``conditions``.
    """
    rng = np.random.default_rng(config.seed)
    conditions = []
    for name, pool in pools.items():
        for target in config.targets:
            delta, truth = delta_for(pool, target, rng, config.population_draws)
            conditions.append((name, len(pool), target, delta, truth))
    jobs = [
        (
            name,
            delta,
            n,
            config.resamples,
            config.seed * 1_000_003 + i * 10_007 + s,
            config.alpha,
        )
        for i, (name, n, _, delta, _) in enumerate(conditions)
        for s in range(config.sims)
    ]
    workers = config.workers or max(1, (os.cpu_count() or 2) - 1)
    if workers == 1:
        _init(pools)
        runs = [_job(j) for j in jobs]
    else:
        with ProcessPoolExecutor(workers, initializer=_init, initargs=(pools,)) as ex:
            runs = list(ex.map(_job, jobs, chunksize=4))
    results = []
    for i, (name, n, target, delta, truth) in enumerate(conditions):
        chunk = runs[i * config.sims : (i + 1) * config.sims]
        results.append(
            {
                "pool": name,
                "n": n,
                "target": target,
                "delta": delta,
                **summarise(truth, chunk, config.reference),
            }
        )
    return {
        "design": {
            "sims": config.sims,
            "resamples": config.resamples,
            "reference": config.reference,
            "alpha_one_sided": config.alpha,
            "miscalibration": "g(p) = clip(p - delta, 0, 1)",
            "population_draws": config.population_draws,
        },
        "conditions": results,
    }


def run(settings: Settings, config: Config) -> dict[str, Any]:
    """Load the pools, simulate and write the summary.

    Args:
        settings: Run settings.
        config: Options.

    Returns:
        The simulation results.
    """
    res = simulate(load_pools(settings), config)
    write_json(settings.out(NAME) / "summary.json", {"config": as_dict(config), **res})
    return res


ANALYSIS = Analysis(
    name=NAME,
    summary="Coverage of SmoothECE upper bounds under four resampling schemes",
    config_type=Config,
    run=run,
)
