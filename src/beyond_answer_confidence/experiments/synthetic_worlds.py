"""Synthetic worlds: do probabilities follow what the facts say, and only that?

Made-up facts with known ideal distributions (see
:mod:`beyond_answer_confidence.tasks.synthetic`) separate three situations the model
cannot know from prior knowledge: a fact that is decided but unknown, one
that will be settled by a stated chance, and one settled by chance that has
already happened. The analysis reports the distance from the ideal per
cell, whether irrelevant cues (a stranger's guess, past independent draws)
move the distribution, and whether the ``settled`` and ``determined`` yes/no
answers separate the cells.
"""

import logging
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
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
from beyond_answer_confidence.metrics.ranking import auroc
from beyond_answer_confidence.runner import run_collect, run_summary
from beyond_answer_confidence.settings import Settings
from beyond_answer_confidence.stats.inference import (
    auroc_vs_threshold,
    bootstrap_equivalence,
    holm_adjust_tails,
    mean_interval,
)
from beyond_answer_confidence.stats.resampling import bootstrap_means, percentile_ci
from beyond_answer_confidence.tasks.schema import Task
from beyond_answer_confidence.tasks.synthetic import (
    ALONE_CELLS,
    CELLS,
    DETERMINED_CELLS,
    PROFILES,
    SETTLED_CELLS,
    make_scenario,
    scenario_tasks,
    score_scenarios,
)

logger = logging.getLogger(__name__)

NAME = "synthetic_worlds"

Row = Mapping[str, Any]


@dataclass(frozen=True)
class Config:
    """Options of the synthetic-worlds experiment.

    Attributes:
        scenarios: Number of scenarios.
        alone_scenarios: Leading scenarios also asked without the yes/no
            questions (batching control).
        replicates: Replicates per cell.
        seed: Scenario seed.
        bootstrap: Bootstrap resamples.
        bootstrap_seed: Bootstrap seed.
        tv_reference: Reference total-variation distance from the ideal.
        past_margin: Equivalence margin for the shift toward past draws.
        settled_auroc_reference: Reference AUROC for ``settled`` separating
            decided-but-unknown from future-chance facts.
    """

    scenarios: int = 480
    alone_scenarios: int = 120
    replicates: int = 3
    seed: int = 0
    bootstrap: int = 2000
    bootstrap_seed: int = 0
    tv_reference: float = 0.10
    past_margin: float = 0.05
    settled_auroc_reference: float = 0.80


def requests(settings: Settings, config: Config) -> list[Task]:
    """Every task of the experiment.

    Args:
        settings: Run settings (unused; no data needed).
        config: Options.

    Returns:
        The tasks.
    """
    return scenario_tasks(
        config.scenarios, config.alone_scenarios, config.replicates, config.seed
    )[1]


def _paired(rows: Sequence[Row], a: str, b: str) -> list[tuple[Row, Row]]:
    by = {(r["scenario"], r["cell"]): r for r in rows if not r["alone"]}
    return [
        (by[(s, a)], by[(s, b)])
        for s in sorted({r["scenario"] for r in rows})
        if (s, a) in by and (s, b) in by
    ]


def _shift(pa: Row, pb: Row) -> float:
    """Change of the probability on the nudge target from ``pb`` to ``pa``."""
    target = pa["target"]
    return float(pa["dist"][target] - pb["dist"][target])


def _mean_by(
    rows: Sequence[Row], key: str, value: Callable[[Row], float]
) -> dict[str, float]:
    groups: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        groups[str(r[key])].append(value(r))
    return {k: float(np.mean(v)) for k, v in sorted(groups.items())}


def analyse(rows: Sequence[Row], config: Config) -> dict[str, Any]:
    """Distances from the ideal, cue sensitivity and meta-question separation.

    Bootstrap draws (one generator, in this order): D3, D0, D1, D2, SFp,
    SFc distances; past-draw shift; ``settled`` AUROC; the three nudge
    shifts; the nudge-susceptibility difference.

    Args:
        rows: Rows from :func:`~beyond_answer_confidence.tasks.synthetic.score_scenarios`.
        config: Options.

    Returns:
        Effect sizes with intervals (tail shares Holm-adjusted across the
        five main comparisons) and descriptive summaries.
    """
    b = config.bootstrap
    rng = np.random.default_rng(config.bootstrap_seed)
    main = [r for r in rows if not r["alone"]]

    def cell_values(cell: str, key: str = "tv") -> np.ndarray:
        return np.array([r[key] for r in main if r["cell"] == cell], float)

    def worst(cells: Sequence[str]) -> dict[str, Any]:
        parts = {
            c: mean_interval(cell_values(c), b, rng, threshold=config.tv_reference)
            for c in cells
        }
        return {"cells": parts, "tail": max(p["tail"] for p in parts.values())}

    family: dict[str, dict[str, Any]] = {}
    family["known_fact"] = mean_interval(
        cell_values("D3"), b, rng, threshold=config.tv_reference
    )
    family["partial_and_no_knowledge"] = worst(["D0", "D1", "D2"])
    family["stated_chance"] = worst(["SFp", "SFc"])
    family["past_draws_shift"] = bootstrap_equivalence(
        np.array([_shift(pa, pb) for pa, pb in _paired(main, "SFpast", "SFp")]),
        config.past_margin,
        b,
        rng,
    )
    settled_rows = [r for r in main if r["cell"] in ("D0", "SFp")]
    family["settled_separates_unknown_fact_from_chance"] = auroc_vs_threshold(
        np.array([r["p_settled"] for r in settled_rows]),
        np.array([r["cell"] == "D0" for r in settled_rows]),
        config.settled_auroc_reference,
        b,
        rng,
        clusters=np.array([r["scenario"] for r in settled_rows]),
    )
    res: dict[str, Any] = holm_adjust_tails(family)
    res["by_cell"] = {
        c: {
            "n": len(cell_values(c)),
            "mean_tv": float(cell_values(c).mean()),
            "mean_p_max": float(cell_values(c, "p_max").mean()),
            "mean_p_settled": float(cell_values(c, "p_settled").mean()),
            "mean_p_determined": float(cell_values(c, "p_determined").mean()),
            "mean_replicate_tv": float(cell_values(c, "rep_tv").mean()),
        }
        for c in CELLS
        if len(cell_values(c))
    }
    res["tv_by_cell_and_profile"] = {
        c: _mean_by([r for r in main if r["cell"] == c], "profile", lambda r: r["tv"])
        for c in CELLS
    }
    res["tv_by_cell_and_domain"] = {
        c: _mean_by([r for r in main if r["cell"] == c], "domain", lambda r: r["tv"])
        for c in CELLS
    }
    res["mass_on_ruled_out"] = {
        cell: float(
            np.mean(
                [
                    sum(p for o, p in r["dist"].items() if r["ideal"][o] == 0)
                    for r in main
                    if r["cell"] == cell
                ]
            )
        )
        for cell in ("D1", "D2")
    }
    d0 = [r for r in main if r["cell"] == "D0"]
    res["unknown_fact"] = {
        "confident_wrong_rate": float(
            np.mean([r["p_max"] >= 0.5 and r["top"] != r["gold"] for r in d0])
        ),
        "accuracy": float(np.mean([r["top"] == r["gold"] for r in d0])),
    }
    res["determined_auroc_stated_vs_rest"] = auroc(
        np.array([r["p_determined"] for r in main]),
        np.array([r["cell"] in DETERMINED_CELLS for r in main]),
    )
    res["settled_auroc_all_cells"] = auroc(
        np.array([r["p_settled"] for r in main]),
        np.array([r["cell"] in SETTLED_CELLS for r in main]),
    )
    sp_sf = [r for r in main if r["cell"] in ("SP", "SFp")]
    res["settled_auroc_past_vs_future_chance"] = auroc(
        np.array([r["p_settled"] for r in sp_sf]),
        np.array([r["cell"] == "SP" for r in sp_sf]),
    )
    uni = [(a, c) for a, c in _paired(main, "D0", "SFp") if a["profile"] == "uniform"]
    res["uniform_profile_unknown_fact_vs_chance"] = {
        "n": len(uni),
        "first_order_tv_between": float(
            np.mean([tv(a["dist"], c["dist"]) for a, c in uni])
        )
        if uni
        else None,
        "settled_auroc": auroc(
            np.array([x["p_settled"] for a, c in uni for x in (a, c)]),
            np.array([x["cell"] == "D0" for a, c in uni for x in (a, c)]),
        )
        if uni
        else None,
    }
    res["nudge_shifts"] = _nudge_shifts(main, b, rng)
    res["batching_control"] = _batching(rows, main)
    res["mode_collapse"] = mode_collapse(rows, config.seed)
    return res


def _nudge_shifts(
    main: Sequence[Row], b: int, rng: np.random.Generator
) -> dict[str, Any]:
    shifts: dict[str, Any] = {}
    for nudged, base in (("D0nudge", "D0"), ("SFnudge", "SFp"), ("D3nudge", "D3")):
        diff = np.array([_shift(pa, pb) for pa, pb in _paired(main, nudged, base)])
        shifts[nudged] = {
            "n": len(diff),
            "mean_shift_to_guess": float(diff.mean()),
            "ci": list(percentile_ci(bootstrap_means(diff, b, rng))),
        }
    pairs_d0 = {
        pa["scenario"]: _shift(pa, pb) for pa, pb in _paired(main, "D0nudge", "D0")
    }
    pairs_sf = {
        pa["scenario"]: _shift(pa, pb) for pa, pb in _paired(main, "SFnudge", "SFp")
    }
    common = sorted(set(pairs_d0) & set(pairs_sf))
    diff = np.array([pairs_d0[s] - pairs_sf[s] for s in common])
    shifts["unknown_fact_minus_chance_susceptibility"] = {
        "n": len(diff),
        "mean": float(diff.mean()) if len(diff) else None,
        "ci": list(percentile_ci(bootstrap_means(diff, b, rng))) if len(diff) else None,
    }
    return shifts


def _batching(rows: Sequence[Row], main: Sequence[Row]) -> dict[str, Any]:
    alone = {(r["scenario"], r["cell"]): r for r in rows if r["alone"]}
    out = {}
    for cell in ALONE_CELLS:
        tvs = [
            tv(r["dist"], alone[(r["scenario"], cell)]["dist"])
            for r in main
            if r["cell"] == cell and (r["scenario"], cell) in alone
        ]
        noise = [r["rep_tv"] for r in main if r["cell"] == cell]
        out[cell] = {
            "n": len(tvs),
            "mean_tv_alone_vs_batched": float(np.mean(tvs)) if tvs else None,
            "mean_replicate_tv": float(np.mean(noise)) if noise else None,
        }
    return out


def mode_collapse(rows: Sequence[Row], seed: int = 0) -> dict[str, Any]:
    """Where does the mass go under chance?

    For each chance cell (and ``D0``) and profile: the mean probability on
    the most likely outcome(s) under p*, the share of items whose top choice
    is such a mode, and the share whose top choice is the outcome listed
    first in the stated chances.

    Args:
        rows: Scored rows.
        seed: Scenario seed (to regenerate p*).

    Returns:
        Cell -> profile -> summary.
    """
    out: dict[str, Any] = {}
    for cell in ("SFp", "SFc", "SP", "SFpast", "SFnudge", "US", "D0"):
        out[cell] = {}
        for prof in PROFILES:
            g = [
                r
                for r in rows
                if not r["alone"] and r["cell"] == cell and r["profile"] == prof
            ]
            if not g:
                continue
            p_mode, is_mode, first = [], [], []
            for r in g:
                sc = make_scenario(r["scenario"], seed)
                top_p = max(sc.p_star.values())
                modes = [o for o, p in sc.p_star.items() if p == top_p]
                listed = [
                    o for o, _ in sorted(sc.p_star.items(), key=lambda kv: -kv[1])
                ]
                p_mode.append(sum(r["dist"][o] for o in modes))
                is_mode.append(r["top"] in modes)
                first.append(r["top"] == listed[0])
            out[cell][prof] = {
                "n": len(g),
                "p_star_mode": max(PROFILES[prof]) / 20,
                "mean_mass_on_modes": float(np.mean(p_mode)),
                "mean_p_max": float(np.mean([r["p_max"] for r in g])),
                "share_top_is_mode": float(np.mean(is_mode)),
                "share_top_is_first_listed": float(np.mean(first)),
            }
    return out


def run(settings: Settings, config: Config) -> dict[str, Any]:
    """Collect (cache first), score, analyse and write the outputs.

    Args:
        settings: Run settings.
        config: Options.

    Returns:
        The summary (``results`` only when every unit is complete).
    """
    scenarios, tasks = scenario_tasks(
        config.scenarios, config.alone_scenarios, config.replicates, config.seed
    )
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
    expected = Counter(t.unit for t in tasks)
    rows = score_scenarios(scenarios, collected.complete(expected))
    out = settings.out(NAME)
    write_jsonl(out / "rows.jsonl", rows)
    summary: dict[str, Any] = {
        "config": as_dict(config),
        "units_expected": len(expected),
        "units_complete": len(rows),
        **run_summary(client, collected),
    }
    if len(rows) == len(expected):
        summary["results"] = analyse(rows, config)
    else:
        logger.warning(
            "%s: %d of %d units complete; no analysis", NAME, len(rows), len(expected)
        )
    write_json(out / "summary.json", summary)
    return summary


EXPERIMENT = Experiment(
    name=NAME,
    summary="Made-up facts with known ideal answers: knowledge versus chance",
    config_type=Config,
    requests=requests,
    run=run,
)
