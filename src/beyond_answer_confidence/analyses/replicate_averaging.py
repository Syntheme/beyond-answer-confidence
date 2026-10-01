"""Calibration of a single call versus the average of two or three.

Every experiment averages three identical calls into one distribution, but
a user normally receives one call. The request cache keeps every call with
its replicate index, so this analysis rebuilds each experiment's items,
answers them from the cache only (no backend; a missing call raises) and
scores them in several modes:

- ``mean3``: the average of all three calls (reproduces the experiment's
  rows; the number of mismatching rows is reported);
- ``single_r0`` / ``single_r1`` / ``single_r2``: each call on its own;
- ``single_random``: one randomly chosen call per item (what a user sees);
- ``mean2``: the average of two calls, averaged over the three pairs.

For each calibration set it reports accuracy, mean confidence, the gap and
SmoothECE per mode, SmoothECE bounds under four resampling schemes for
``single_random``, and the spread of SmoothECE and accuracy over repeated
random single-call draws. The knowledge-dial rows store one row per call,
so they are re-aggregated rather than rebuilt from the cache.

Output: ``<output_dir>/replicate_averaging/summary.json``.
"""

import itertools
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from beyond_answer_confidence.analyses.cutoff_calibration import CUTOFF_MONTH, NEWS_SET
from beyond_answer_confidence.analyses.knowledge_dial import aggregate
from beyond_answer_confidence.backends.cache import CachedClient, CallRecord
from beyond_answer_confidence.config import as_dict
from beyond_answer_confidence.experiments.base import Analysis, read_jsonl, write_json
from beyond_answer_confidence.metrics.calibration import smece
from beyond_answer_confidence.runner import run_collect
from beyond_answer_confidence.settings import Settings
from beyond_answer_confidence.stats.smece_bounds import smece_with_bounds
from beyond_answer_confidence.tasks.schema import Item, item_tasks
from beyond_answer_confidence.tasks.scoring import score_items

NAME = "replicate_averaging"
REPLICATES = 3
KNOWLEDGE_CONDITIONS = ("L1", "L2", "L4", "L8", "Lname", "Lname+8")
"""Knowledge-dial conditions with calibration sets."""
KNOWLEDGE_DIAL_KEYS = ("item", "condition", "code_style", "example_seed")
"""Columns identifying a knowledge-dial cell within the main arm."""

Groups = dict[str, tuple[np.ndarray, np.ndarray]]
"""Set name -> (confidence, correct)."""
ItemBuilder = Callable[[Settings], list[Item]]


@dataclass(frozen=True)
class Config:
    """Options.

    Attributes:
        sources: Knowledge-dial datasets (``knowledge_dial_<dataset>``) and
            experiments to analyse, in order (one generator for all).
        resamples: Resamples per scheme for the ``single_random`` bounds.
        draws: Repeated random single-call draws (0 skips them).
        seed: Seed of the generator and of each source's random call.
        smece_reference: Reference SmoothECE for the tail shares and the
            share of draws below it.
    """

    sources: tuple[str, ...] = (
        "knowledge_dial_banking77",
        "knowledge_dial_clinc150",
        "knowledge_boundary",
        "evidence_sufficiency",
        "benchmarks",
        "option_count",
        "open_benchmarks",
    )
    resamples: int = 1000
    draws: int = 500
    seed: int = 0
    smece_reference: float = 0.05


# --- modes ----------------------------------------------------------------------


def per_call_rows(
    items: Sequence[Item], cache: Path, model: str, label: str, seed: int = 0
) -> dict[str, list[list[dict[str, Any]]]]:
    """Score items under every mode, from the cache only.

    Args:
        items: The experiment's items, as built for its run.
        cache: The experiment's request cache.
        model: Model name (part of every cache key).
        label: Progress label.
        seed: Seed for the random single call (one draw per item, in item
            order).

    Returns:
        Mode -> list of row sets (three for ``mean2``, one otherwise).

    Raises:
        RuntimeError: If any item is not fully cached.
    """
    client = CachedClient(cache, model=model, max_input_tokens=0)
    collected = run_collect(
        item_tasks(items, REPLICATES), client, concurrency=1, label=label
    )
    complete = collected.complete(dict.fromkeys((it.unit for it in items), REPLICATES))
    if len(complete) != len(items):
        raise RuntimeError(f"{label}: {len(items) - len(complete)} items not cached")

    def score(picks: Sequence[int] | Mapping[str, int]) -> list[dict[str, Any]]:
        chosen: dict[str, list[CallRecord]] = {}
        for u, recs in complete.items():
            idx = [picks[u]] if isinstance(picks, Mapping) else list(picks)
            chosen[u] = [recs[i] for i in idx]
        return score_items(items, chosen)

    rng = np.random.default_rng(seed)
    choice = {u: int(rng.integers(0, REPLICATES)) for u in complete}
    modes: dict[str, list[list[dict[str, Any]]]] = {
        "mean3": [score(range(REPLICATES))],
        "single_random": [score(choice)],
        "mean2": [score(pair) for pair in itertools.combinations(range(REPLICATES), 2)],
    }
    for r in range(REPLICATES):
        modes[f"single_r{r}"] = [score([r])]
    return modes


def count_mismatches(rows: Sequence[Mapping[str, Any]], saved: Path) -> int:
    """Count rows whose ``p_max`` differs from the experiment's saved rows.

    Args:
        rows: ``mean3`` rows.
        saved: The experiment's ``rows.jsonl``.

    Returns:
        Number of mismatching or missing units (0 expected).
    """
    old = {r["unit"]: r.get("p_max") for r in read_jsonl(saved)}
    count = 0
    for r in rows:
        if r.get("p_max") is None:
            continue
        before = old.get(r["unit"])
        if before is None or abs(float(before) - float(r["p_max"])) > 1e-9:
            count += 1
    return count


def knowledge_dial_modes(
    rows: pd.DataFrame, seed: int = 0
) -> dict[str, list[pd.DataFrame]]:
    """Knowledge-dial main-arm cells under every mode (rows are one per call).

    Args:
        rows: Knowledge-dial rows (one per request, with ``replicate``).
        seed: Seed for the random single call (one draw per cell).

    Returns:
        Mode -> aggregated main-arm cells.
    """
    rows = rows[rows["arm"] == "main"]
    keys = list(KNOWLEDGE_DIAL_KEYS)
    rng = np.random.default_rng(seed)
    units = rows[keys].drop_duplicates().reset_index(drop=True)
    units["pick"] = rng.integers(0, REPLICATES, len(units))
    picked = rows.merge(units, on=keys)
    modes: dict[str, list[pd.DataFrame]] = {
        "mean3": [aggregate(rows)],
        "single_random": [aggregate(picked[picked["replicate"] == picked["pick"]])],
        "mean2": [
            aggregate(rows[rows["replicate"].isin(pair)])
            for pair in itertools.combinations(range(REPLICATES), 2)
        ],
    }
    for r in range(REPLICATES):
        modes[f"single_r{r}"] = [aggregate(rows[rows["replicate"] == r])]
    return modes


# --- calibration groups ------------------------------------------------------------


def _pc(g: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    return g["p_max"].to_numpy(float), g["correct"].to_numpy(float)


def groups_knowledge_dial(cells: pd.DataFrame) -> Groups:
    """Knowledge-dial calibration conditions.

    Args:
        cells: Aggregated main-arm cells.

    Returns:
        Calibration groups.
    """
    return {c: _pc(cells[cells["condition"] == c]) for c in KNOWLEDGE_CONDITIONS}


def groups_knowledge_boundary(df: pd.DataFrame) -> Groups:
    """PopQA and dated yes/no news before and after the cutoff.

    Args:
        df: Knowledge-boundary rows.

    Returns:
        Calibration groups.
    """
    tf = df[df["set"] == NEWS_SET]
    return {
        "popqa": _pc(df[df["set"] == "popqa"]),
        "oracle_tf_pre": _pc(tf[tf["month"] < CUTOFF_MONTH]),
        "oracle_tf_post": _pc(tf[tf["month"] >= CUTOFF_MONTH]),
    }


def groups_evidence_sufficiency(df: pd.DataFrame) -> Groups:
    """HotpotQA cells and Quizbowl first, middle and full clue.

    Args:
        df: Evidence-sufficiency rows.

    Returns:
        Calibration groups.
    """
    hp, qb = df[df["set"] == "hotpot"], df[df["set"] == "quizbowl"]
    out = {
        f"hotpot_{c}": _pc(hp[hp["cell"] == c])
        for c in ("closed", "dose0", "dose1", "dose2")
    }
    mid = qb[qb["k"] == qb["n_sentences"].map(lambda n: math.ceil(n / 2))]
    out |= {
        "quizbowl_first": _pc(qb[qb["k"] == 1]),
        "quizbowl_middle": _pc(mid),
        "quizbowl_full": _pc(qb[qb["k"] == qb["n_sentences"]]),
    }
    return out


def groups_benchmarks(df: pd.DataFrame) -> Groups:
    """Factual question-answering sets.

    Args:
        df: Benchmark rows.

    Returns:
        Calibration groups.
    """
    return {
        s: _pc(df[df["set"] == s])
        for s in ("triviaqa", "simpleqa", "truthfulqa_binary", "truthfulqa_mc1")
    }


def groups_option_count(df: pd.DataFrame) -> Groups:
    """MMLU-Redux (unflawed items), MMLU-CF, ANLI and CLINC150 per option count.

    Args:
        df: Option-count rows.

    Returns:
        Calibration groups.
    """
    red = df[(df["set"] == "mmlu_redux") & (df["error_type"] == "ok")]
    out = {"mmlu_redux": _pc(red), "mmlu_cf": _pc(df[df["set"] == "mmlu_cf"])}
    out["anli"] = _pc(df[df["set"] == "anli"])
    clinc = df[df["set"] == "clinc_k"]
    out |= {f"clinc_K{int(cast(int, k))}": _pc(g) for k, g in clinc.groupby("K")}
    return out


def groups_verification(df: pd.DataFrame) -> Groups:
    """TriviaQA verification: confidence in the yes/no verdict.

    Args:
        df: Verification rows (``p_correct``, ``truth``).

    Returns:
        Calibration groups.
    """
    tv = df[df["set"] == "trivia_verify"]
    p = tv["p_correct"].to_numpy(float)
    truth = tv["truth"].to_numpy(bool)
    return {
        "trivia_verify": (np.maximum(p, 1 - p), ((p >= 0.5) == truth).astype(float))
    }


# --- summaries ---------------------------------------------------------------------


def metrics(conf: np.ndarray, ok: np.ndarray) -> dict[str, float]:
    """Accuracy, mean confidence, gap and SmoothECE.

    Args:
        conf: Confidences.
        ok: Correctness.

    Returns:
        The metrics.
    """
    return {
        "n": len(conf),
        "accuracy": float(ok.mean()),
        "mean_confidence": float(conf.mean()),
        "gap": float(conf.mean() - ok.mean()),
        "smece": smece(conf, ok),
    }


def _spread(v: np.ndarray) -> dict[str, float]:
    lo, hi = np.percentile(v, [2.5, 97.5])
    return {
        "mean": float(v.mean()),
        "p2.5": float(lo),
        "p97.5": float(hi),
        "min": float(v.min()),
        "max": float(v.max()),
    }


def random_draws(
    singles: Sequence[pd.DataFrame],
    keys: Sequence[str],
    group_fn: Callable[[pd.DataFrame], Groups],
    draws: int,
    rng: np.random.Generator,
    reference: float = 0.05,
) -> dict[str, dict[str, Any]]:
    """Repeat the "one random call per item" draw many times.

    Each draw picks one of the three calls independently for every item, so
    the spread over draws shows how much single-call calibration varies with
    which call a user happened to receive.

    Args:
        singles: The three single-replicate frames (replicate 0, 1, 2).
        keys: Columns identifying an item (aligns the three frames).
        group_fn: Splits rows into calibration groups.
        draws: Number of random draws.
        rng: Random generator.
        reference: Reference SmoothECE.

    Returns:
        Set -> SmoothECE and accuracy over draws (mean, 2.5/97.5
        percentiles, min, max) and the share of draws with SmoothECE below
        ``reference``.

    Raises:
        ValueError: If the frames do not cover the same items.
    """
    aligned = [f.set_index(list(keys)).sort_index() for f in singles]
    index = aligned[0].index
    if any(not a.index.equals(index) for a in aligned[1:]):
        raise ValueError("replicate frames do not cover the same items")
    groups = [group_fn(a.reset_index()) for a in aligned]
    out: dict[str, dict[str, Any]] = {}
    for name in groups[0]:
        conf = np.stack([g[name][0] for g in groups])
        ok = np.stack([g[name][1] for g in groups])
        cols = np.arange(conf.shape[1])
        ece, acc = np.empty(draws), np.empty(draws)
        for d in range(draws):
            pick = rng.integers(0, len(groups), conf.shape[1])
            ece[d] = smece(conf[pick, cols], ok[pick, cols])
            acc[d] = ok[pick, cols].mean()
        out[name] = {
            "draws": draws,
            "smece": _spread(ece),
            "accuracy": _spread(acc),
            "share_smece_below_reference": float(np.mean(ece < reference)),
        }
    return out


def summarise_modes(
    modes: Mapping[str, Sequence[pd.DataFrame]],
    group_fn: Callable[[pd.DataFrame], Groups],
    b: int,
    rng: np.random.Generator,
    keys: Sequence[str] = ("unit",),
    draws: int = 0,
    reference: float = 0.05,
) -> dict[str, Any]:
    """Metrics per calibration set and mode.

    Args:
        modes: Mode -> row sets as data frames.
        group_fn: Splits rows into calibration groups.
        b: Resamples per scheme for the ``single_random`` SmoothECE bounds.
        rng: Random generator.
        keys: Columns identifying an item, for aligning replicates.
        draws: Repeated random single-call draws (0 skips them).
        reference: Reference SmoothECE.

    Returns:
        Set -> mode -> metrics (``mean2`` averaged over pairs), the
        ``single_random`` bounds, the range over single calls, the SmoothECE
        increase of a random single call over ``mean3`` and, with ``draws``,
        the spread over repeated draws.
    """
    out: dict[str, Any] = {}
    for mode, frames in modes.items():
        per_frame = [group_fn(f) for f in frames]
        for name in per_frame[0]:
            ms = [metrics(*g[name]) for g in per_frame]
            out.setdefault(name, {})[mode] = {
                k: float(np.mean([m[k] for m in ms])) for k in ms[0]
            }
            if mode == "single_random":
                conf, ok = per_frame[0][name]
                chk = smece_with_bounds(conf, ok, reference, b, rng)
                out[name]["single_random_bounds"] = {
                    k: chk[k] for k in ("smece", "bounds", "bootstrap_bias")
                }
    for res in out.values():
        singles = [res[f"single_r{r}"]["smece"] for r in range(REPLICATES)]
        res["single_call_smece_range"] = [min(singles), max(singles)]
        res["smece_increase_single_vs_mean3"] = float(
            res["single_random"]["smece"] - res["mean3"]["smece"]
        )
    if draws:
        singles_f = [modes[f"single_r{r}"][0] for r in range(REPLICATES)]
        spreads = random_draws(singles_f, keys, group_fn, draws, rng, reference)
        for name, spread in spreads.items():
            out[name]["random_draws"] = spread
    return out


# --- sources -----------------------------------------------------------------------


def _knowledge_boundary_items(settings: Settings) -> list[Item]:
    from beyond_answer_confidence.experiments import knowledge_boundary as exp

    return exp.build_items(settings, exp.Config())


def _evidence_items(settings: Settings) -> list[Item]:
    from beyond_answer_confidence.experiments import evidence_sufficiency as exp

    return exp.build_items(settings, exp.Config())


def _benchmark_items(settings: Settings) -> list[Item]:
    from beyond_answer_confidence.experiments import benchmarks as exp

    return exp.build_items(settings, exp.Config())


def _option_count_items(settings: Settings) -> list[Item]:
    from beyond_answer_confidence.experiments import option_count as exp

    return exp.build_items(settings, exp.Config())


def _verification_items(settings: Settings) -> list[Item]:
    from beyond_answer_confidence.data import loaders
    from beyond_answer_confidence.data.distractors import distractor_picker
    from beyond_answer_confidence.experiments import open_benchmarks as exp
    from beyond_answer_confidence.tasks.benchmarks import triviaqa_items

    config = exp.Config()
    d = settings.data_dir
    trivia = triviaqa_items(
        loaders.load_triviaqa_validation(d), distractor_picker(d), config.seed
    )
    return exp.verification_items(
        trivia, config.verification_questions, config.verification_seed
    )


EXPERIMENT_SOURCES: dict[str, tuple[ItemBuilder, Callable[[pd.DataFrame], Groups]]] = {
    "knowledge_boundary": (_knowledge_boundary_items, groups_knowledge_boundary),
    "evidence_sufficiency": (_evidence_items, groups_evidence_sufficiency),
    "benchmarks": (_benchmark_items, groups_benchmarks),
    "option_count": (_option_count_items, groups_option_count),
    "open_benchmarks": (_verification_items, groups_verification),
}
"""Experiment -> (item builder with default sizes, calibration groups)."""
KNOWLEDGE_DIAL_PREFIX = "knowledge_dial_"


def analyse_source(
    settings: Settings, source: str, config: Config, rng: np.random.Generator
) -> dict[str, Any]:
    """Every mode for one source.

    Args:
        settings: Run settings (caches and saved rows).
        source: ``knowledge_dial_<dataset>`` or a key of
            :data:`EXPERIMENT_SOURCES`.
        config: Options.
        rng: Shared generator.

    Returns:
        The source's summary.

    Raises:
        KeyError: For an unknown source.
    """
    if source.startswith(KNOWLEDGE_DIAL_PREFIX):
        from beyond_answer_confidence.experiments.knowledge_dial import rows_path

        ds = source.removeprefix(KNOWLEDGE_DIAL_PREFIX)
        rows = pd.DataFrame(read_jsonl(rows_path(settings, "test", ds)))
        return summarise_modes(
            knowledge_dial_modes(rows, config.seed),
            groups_knowledge_dial,
            config.resamples,
            rng,
            keys=KNOWLEDGE_DIAL_KEYS,
            draws=config.draws,
            reference=config.smece_reference,
        )
    if source not in EXPERIMENT_SOURCES:
        raise KeyError(f"unknown source {source!r}")
    build, group_fn = EXPERIMENT_SOURCES[source]
    raw = per_call_rows(
        build(settings),
        settings.cache_file(source),
        settings.model,
        source,
        config.seed,
    )
    mismatches = count_mismatches(raw["mean3"][0], settings.out(source) / "rows.jsonl")
    modes = {m: [pd.DataFrame(rs) for rs in sets] for m, sets in raw.items()}
    return {
        "mean3_mismatches_vs_rows": mismatches,
        **summarise_modes(
            modes,
            group_fn,
            config.resamples,
            rng,
            draws=config.draws,
            reference=config.smece_reference,
        ),
    }


def run(settings: Settings, config: Config) -> dict[str, Any]:
    """Run every source and write the summary.

    Args:
        settings: Run settings.
        config: Options.

    Returns:
        Source -> summary.
    """
    rng = np.random.default_rng(config.seed)
    res = {s: analyse_source(settings, s, config, rng) for s in config.sources}
    write_json(settings.out(NAME) / "summary.json", {"config": as_dict(config), **res})
    return res


ANALYSIS = Analysis(
    name=NAME,
    summary="Calibration of one call versus the mean of two or three",
    config_type=Config,
    run=run,
)
