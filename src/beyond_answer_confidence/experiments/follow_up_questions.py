"""Follow-up questions: wording, odds formats, the knowledge cutoff, warned paragraphs.

Four probes on the item sets of the core experiments (see
:mod:`beyond_answer_confidence.tasks.follow_up`):

- ``wording``: does every paraphrase of ``settled``, ``known`` and
  ``enough`` separate the cases the original separates (cluster-bootstrap
  AUROC per wording), including adversarial synthetic cells?
- ``odds_formats``: how close are stated chances read back as per-option
  yes/no probabilities, per-option scores, or an instructed choice?
- ``cutoff``: does giving today's date reduce overconfidence after the
  knowledge cutoff, and does an ``unknown`` option flag post-cutoff news?
- ``warned_paragraphs``: is HotpotQA calibrated when told the paragraphs may
  be irrelevant?

All four share one cache. The ``cutoff`` probe's ``base`` condition repeats
the knowledge-boundary requests unchanged. The warned-paragraph comparison
reads the evidence-sufficiency rows when they exist.
"""

import logging
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from beyond_answer_confidence.config import as_dict
from beyond_answer_confidence.data import loaders
from beyond_answer_confidence.experiments.base import (
    Experiment,
    backend_for,
    read_jsonl,
    write_json,
    write_jsonl,
)
from beyond_answer_confidence.metrics.ranking import auroc
from beyond_answer_confidence.runner import run_items
from beyond_answer_confidence.settings import Settings
from beyond_answer_confidence.stats.inference import (
    auroc_vs_threshold,
    holm_adjust_tails,
    mean_interval,
)
from beyond_answer_confidence.stats.resampling import smece_interval
from beyond_answer_confidence.tasks.entities import fabricated_items, popqa_items
from beyond_answer_confidence.tasks.evidence import hotpot_items
from beyond_answer_confidence.tasks.follow_up import (
    CUTOFF_MONTH,
    ENOUGH_WORDINGS,
    KNOWN_WORDINGS,
    SETTLED_WORDINGS,
    TODAY,
    cutoff_items,
    odds_format_items,
    odds_readings,
    warned_paragraph_items,
    wording_items,
)
from beyond_answer_confidence.tasks.news import yes_no_news_items
from beyond_answer_confidence.tasks.schema import Item, Task, item_tasks
from beyond_answer_confidence.tasks.synthetic import make_scenario

logger = logging.getLogger(__name__)

NAME = "follow_up_questions"

PROBES = ("wording", "odds_formats", "cutoff", "warned_paragraphs")
"""Probe names, in build order."""

_UNIT_PREFIX = {
    "wording": "wording",
    "odds_formats": "odds",
    "cutoff": "cutoff",
    "warned_paragraphs": "warned",
}
"""Probe -> unit prefix of its items."""

EVIDENCE_EXPERIMENT = "evidence_sufficiency"
"""Experiment whose rows the warned-paragraph probe compares against."""


@dataclass(frozen=True)
class Config:
    """Options of the follow-up-question probes.

    Attributes:
        replicates: Replicates per item.
        seed: Seed of the source item sets (scenarios, distractors, news
            sampling) and of the probes' own sampling.
        fabricated_per_relation: Fabricated entities per relation in the
            source set (to rebuild it exactly).
        yes_no_per_month: News questions per month in the source set.
        wording_scenarios: Synthetic scenarios in the wording probe.
        wording_per_quintile: Real PopQA items per popularity quintile.
        wording_fabricated_per_relation: Fabricated items per relation.
        wording_paragraph_questions: HotpotQA questions in the wording probe.
        odds_scenarios: Synthetic scenarios in the odds-format probe.
        cutoff_month: First month counted as after the knowledge cutoff.
        today: Date given in the dated cutoff conditions.
        bootstrap: Bootstrap resamples.
        bootstrap_seed: Bootstrap seed (a fresh generator per probe).
        auroc_reference: Reference AUROC for the wording and ``unknown``
            checks.
        abs_error_reference: Reference mean absolute error of odds readings.
        confident_wrong_reference: Reference share of confident errors.
        confident_level: Top probability counted as confident.
        smece_reference: Reference SmoothECE.
    """

    replicates: int = 3
    seed: int = 0
    fabricated_per_relation: int = 100
    yes_no_per_month: int = 80
    wording_scenarios: int = 120
    wording_per_quintile: int = 80
    wording_fabricated_per_relation: int = 25
    wording_paragraph_questions: int = 300
    odds_scenarios: int = 480
    cutoff_month: str = CUTOFF_MONTH
    today: str = TODAY
    bootstrap: int = 2000
    bootstrap_seed: int = 0
    auroc_reference: float = 0.80
    abs_error_reference: float = 0.10
    confident_wrong_reference: float = 0.05
    confident_level: float = 0.9
    smece_reference: float = 0.05


def build_items(settings: Settings, config: Config) -> dict[str, list[Item]]:
    """Load the source datasets and build every probe's items.

    Args:
        settings: Run settings (data directory).
        config: Options.

    Returns:
        Probe name -> items.
    """
    d = settings.data_dir
    popqa = loaders.load_popqa(d)
    entities = popqa_items(popqa, config.seed) + fabricated_items(
        popqa, config.fabricated_per_relation, config.seed
    )
    hotpot = hotpot_items(loaders.load_hotpot_validation(d), config.seed)
    news = yes_no_news_items(
        loaders.load_daily_oracle("tf", d), config.yes_no_per_month, config.seed
    )
    return {
        "wording": wording_items(
            entities,
            hotpot,
            n_scenarios=config.wording_scenarios,
            per_quintile=config.wording_per_quintile,
            fabricated_per_relation=config.wording_fabricated_per_relation,
            paragraph_questions=config.wording_paragraph_questions,
            seed=config.seed,
        ),
        "odds_formats": odds_format_items(config.odds_scenarios, config.seed),
        "cutoff": cutoff_items(
            news, cutoff_month=config.cutoff_month, today=config.today, seed=config.seed
        ),
        "warned_paragraphs": warned_paragraph_items(hotpot),
    }


def requests(settings: Settings, config: Config) -> list[Task]:
    """Every task of the experiment.

    Args:
        settings: Run settings.
        config: Options.

    Returns:
        The tasks.
    """
    items = build_items(settings, config)
    return item_tasks([it for p in PROBES for it in items[p]], config.replicates)


def probe_of(unit: str) -> str:
    """Probe an item belongs to, from its unit.

    Args:
        unit: Unit id.

    Returns:
        The probe name.
    """
    prefix = unit.split(":", 1)[0]
    return next(p for p, u in _UNIT_PREFIX.items() if u == prefix)


def _wording_auroc(
    g: pd.DataFrame,
    col: str,
    positive: np.ndarray,
    reverse: bool,
    clusters: np.ndarray,
    config: Config,
    rng: np.random.Generator,
) -> dict[str, Any]:
    score = g[col].to_numpy(float)
    if reverse:
        score = 1 - score
    return auroc_vs_threshold(
        score, positive, config.auroc_reference, config.bootstrap, rng, clusters
    )


def analyse_wording(
    rows: Sequence[Mapping[str, Any]], config: Config
) -> dict[str, Any]:
    """AUROC of every wording, plus per-cell means.

    Bootstrap draws (one generator, in this order): each ``settled``
    wording, ``settled`` on the adversarial cells, each ``known`` wording,
    each ``enough`` wording. Synthetic rows are clustered by scenario,
    HotpotQA rows by question.

    Args:
        rows: Scored rows of the wording probe.
        config: Options.

    Returns:
        AUROC per wording (tail shares below the reference, Holm-adjusted
        across the four families; ``exists`` is reported but not part of
        the ``known`` family's tail) and mean P(yes) per cell or group.
    """
    rng = np.random.default_rng(config.bootstrap_seed)
    df = pd.DataFrame(rows)
    st = df[df["part"] == "settled"]
    pos = st["positive"].to_numpy(bool)
    cl = st["scenario"].to_numpy()
    settled = {
        w: _wording_auroc(st, f"p_{w}", pos, rev, cl, config, rng)
        for w, (_, rev) in SETTLED_WORDINGS.items()
    }
    adv = st[st["cell"].str.startswith("ADV")]
    adversarial = _wording_auroc(
        adv,
        "p_settled",
        adv["positive"].to_numpy(bool),
        False,
        adv["scenario"].to_numpy(),
        config,
        rng,
    )
    kn = df[df["part"] == "known"]
    fab = kn["fabricated"].to_numpy(bool)
    known = {
        w: _wording_auroc(kn, f"p_{w}", fab, not rev, np.arange(len(kn)), config, rng)
        for w, (_, rev) in KNOWN_WORDINGS.items()
    }
    en = df[df["part"] == "enough"]
    both = (en["cell"] == "dose2").to_numpy()
    enough = {
        w: _wording_auroc(en, f"p_{w}", both, rev, en["qid"].to_numpy(), config, rng)
        for w, (_, rev) in ENOUGH_WORDINGS.items()
    }
    res: dict[str, Any] = holm_adjust_tails(
        {
            "settled_wordings": {
                "wordings": settled,
                "tail": max(v["p"] for v in settled.values()),
            },
            "settled_adversarial_cells": adversarial,
            "known_wordings": {
                "wordings": known,
                "tail": max(v["p"] for k, v in known.items() if k != "exists"),
            },
            "enough_wordings": {
                "wordings": enough,
                "tail": max(v["p"] for v in enough.values()),
            },
        }
    )
    res |= {
        "settled_by_cell": {
            str(c): {w: float(g[f"p_{w}"].mean()) for w in SETTLED_WORDINGS}
            for c, g in st.groupby("cell")
        },
        "known_by_group": {
            ("made_up" if f else "real"): {
                w: float(g[f"p_{w}"].mean()) for w in KNOWN_WORDINGS
            }
            for f, g in kn.groupby("fabricated")
        },
        "enough_by_cell": {
            str(c): {w: float(g[f"p_{w}"].mean()) for w in ENOUGH_WORDINGS}
            for c, g in en.groupby("cell")
        },
    }
    return res


def mean_by(
    rows: Sequence[Mapping[str, Any]],
    key: str,
    value: Callable[[Mapping[str, Any]], float],
) -> dict[str, float]:
    """Mean of ``value(row)`` grouped by ``row[key]`` (groups sorted by name).

    Args:
        rows: Rows.
        key: Grouping field.
        value: Value extractor.

    Returns:
        Group -> mean.
    """
    groups: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        groups[str(r[key])].append(value(r))
    return {k: float(np.mean(v)) for k, v in sorted(groups.items())}


def analyse_odds(rows: Sequence[Mapping[str, Any]], config: Config) -> dict[str, Any]:
    """Error of odds read back in three formats.

    For each scenario the per-option readings are compared with the stated
    chances: mean absolute error (unnormalised readings) and total variation
    after normalising. Bootstrap draws: yes/no error, score error,
    instructed-choice total variation.

    Args:
        rows: Scored rows of the odds-format probe.
        config: Options.

    Returns:
        Mean error per format with intervals (tail shares at or above the
        reference, Holm-adjusted) and, under ``by_format``, correlation
        with the stated chances, mean sum of readings and error by profile.
    """
    rng = np.random.default_rng(config.bootstrap_seed)
    b, ref = config.bootstrap, config.abs_error_reference
    per: dict[str, list[dict[str, Any]]] = {"noul": [], "score": [], "instructed": []}
    for r in rows:
        sc = make_scenario(r["scenario"], config.seed)
        star = [sc.p_star[o] for o in sc.options]
        read = odds_readings(r, sc)
        per[r["format"]].append(
            {
                "profile": sc.profile,
                "abs_err": float(np.mean(np.abs(np.array(read) - np.array(star)))),
                "tv": 0.5
                * float(
                    np.abs(np.array(read) / max(sum(read), 1e-9) - np.array(star)).sum()
                ),
                "sum": float(sum(read)),
                "read": read,
                "star": star,
            }
        )

    def errors(fmt: str, field: str) -> dict[str, Any]:
        return mean_interval(
            np.array([x[field] for x in per[fmt]]), b, rng, threshold=ref
        )

    res: dict[str, Any] = holm_adjust_tails(
        {
            "per_option_yes_no": errors("noul", "abs_err"),
            "per_option_score": errors("score", "abs_err"),
            "instructed_choice": errors("instructed", "tv"),
        }
    )
    by_format = {}
    for fmt, xs in per.items():
        reads = np.concatenate([x["read"] for x in xs])
        stars = np.concatenate([x["star"] for x in xs])
        by_format[fmt] = {
            "corr_with_p_star": float(np.corrcoef(reads, stars)[0, 1]),
            "mean_sum_of_readings": float(np.mean([x["sum"] for x in xs])),
            "mean_abs_err_by_profile": mean_by(xs, "profile", lambda x: x["abs_err"]),
            "mean_tv_by_profile": mean_by(xs, "profile", lambda x: x["tv"]),
            "mean_reading_by_p_star": {
                str(p): float(reads[stars == p].mean())
                for p in sorted(set(stars.tolist()))
            },
        }
    res["by_format"] = by_format
    return res


def analyse_cutoff(rows: Sequence[Mapping[str, Any]], config: Config) -> dict[str, Any]:
    """Overconfidence after the cutoff, with a date and with an ``unknown`` option.

    Bootstrap draws: change in post-cutoff gap (mean top probability minus
    accuracy) when the date is given; AUROC of P(unknown) for post-cutoff
    questions; share of confident yes/no errors with the ``unknown`` option.

    Args:
        rows: Scored rows of the cutoff probe.
        config: Options.

    Returns:
        The three effect sizes with intervals (Holm-adjusted tail shares)
        and per-condition summaries under ``by_condition``.
    """
    rng = np.random.default_rng(config.bootstrap_seed)
    b, level = config.bootstrap, config.confident_level
    df = pd.DataFrame(rows)
    df["key"] = df["unit"].str.replace(r"^cutoff:[a-z_]+:", "", regex=True)

    def yes_no_wrong(r: Any) -> bool:
        return r["top"] in ("yes", "no") and r["top"] != r["gold"]

    base = df[df["cond"] == "base"].set_index("key")
    date = df[df["cond"] == "date"].set_index("key")
    post_keys = base.index[base["post"]]
    gap_b = base.loc[post_keys, "p_max"] - base.loc[post_keys, "correct"].astype(float)
    gap_d = date.loc[post_keys, "p_max"] - date.loc[post_keys, "correct"].astype(float)
    date_effect = mean_interval((gap_d - gap_b).to_numpy(), b, rng, threshold=0.0)
    unk = df[df["cond"] == "unknown"]
    p_unk = np.array([d["unknown"] for d in unk["dist"]])
    flags = auroc_vs_threshold(
        p_unk, unk["post"].to_numpy(bool), config.auroc_reference, b, rng
    )
    post_unk = unk[unk["post"]]
    cw = np.array(
        [(r["p_max"] >= level) and yes_no_wrong(r) for _, r in post_unk.iterrows()],
        float,
    )
    confident = mean_interval(cw, b, rng, threshold=config.confident_wrong_reference)
    res: dict[str, Any] = holm_adjust_tails(
        {
            "date_changes_post_cutoff_gap": date_effect,
            "unknown_option_flags_post_cutoff": flags,
            "confident_errors_with_unknown_option": confident,
        }
    )
    summ: dict[str, Any] = {}
    for (cond, post), g in df.groupby(["cond", "post"]):
        answered = g[g["top"] != "unknown"]
        summ[f"{cond}:{'post' if post else 'pre'}"] = {
            "n": len(g),
            "share_no": float((g["top"] == "no").mean()),
            "share_unknown": float((g["top"] == "unknown").mean()),
            "accuracy_all": float((g["top"] == g["gold"]).mean()),
            "accuracy_answered": float((answered["top"] == answered["gold"]).mean())
            if len(answered)
            else None,
            "mean_p_max": float(g["p_max"].mean()),
            "confident_wrong": float(
                np.mean(
                    [(r["p_max"] >= level) and yes_no_wrong(r) for _, r in g.iterrows()]
                )
            ),
            "mean_p_known": float(g["p_known"].mean()),
        }
    du = df[df["cond"] == "date_unknown"]
    summ["date_unknown_auroc_p_unknown"] = auroc(
        np.array([d["unknown"] for d in du["dist"]]), du["post"].to_numpy(bool)
    )
    res["by_condition"] = summ
    return res


def analyse_warned(
    rows: Sequence[Mapping[str, Any]],
    config: Config,
    unwarned: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Calibration of HotpotQA answers with the paragraph warning.

    Bootstrap draws: SmoothECE for zero, then one supporting paragraph.

    Args:
        rows: Scored rows of the warned-paragraph probe.
        config: Options.
        unwarned: Evidence-sufficiency rows (the same questions without the
            warning), for a descriptive comparison; ``None`` to skip it.

    Returns:
        SmoothECE per cell with bias-recentred intervals (tail share at or
        above the reference) and, if given, a per-cell comparison with the
        unwarned answers under ``by_cell``.
    """
    rng = np.random.default_rng(config.bootstrap_seed)
    df = pd.DataFrame(rows)
    cells = {
        str(c): smece_interval(
            g["p_max"].to_numpy(),
            g["correct"].to_numpy(),
            config.bootstrap,
            rng,
            threshold=config.smece_reference,
        )
        for c, g in df.groupby("cell")
        if c in ("dose0", "dose1")
    }
    res: dict[str, Any] = holm_adjust_tails(
        {
            "calibration_with_warning": {
                "cells": cells,
                "tail": max(v["tail"] for v in cells.values()),
            }
        }
    )
    if unwarned is None:
        return res
    old = pd.DataFrame(list(unwarned))
    old = old[old["set"] == "hotpot"]
    comp = {}
    for c in ("closed", "dose0", "dose1", "dose2"):
        o = old[old["cell"] == c]
        n = df[df["cell"] == c]
        comp[c] = {
            "unwarned_accuracy": float(o["correct"].mean()),
            "unwarned_mean_p_max": float(o["p_max"].mean()),
            "warned_accuracy": float(n["correct"].mean()) if len(n) else None,
            "warned_mean_p_max": float(n["p_max"].mean()) if len(n) else None,
            "warned_p_enough": float(n["p_enough"].mean()) if len(n) else None,
        }
    res["by_cell"] = comp
    return res


def _unwarned_rows(settings: Settings) -> list[dict[str, Any]] | None:
    path: Path = settings.out(EVIDENCE_EXPERIMENT) / "rows.jsonl"
    if not path.exists():
        logger.warning("%s not found; skipping the unwarned comparison", path)
        return None
    return read_jsonl(path)


def run(settings: Settings, config: Config) -> dict[str, Any]:
    """Collect (cache first), score, analyse every complete probe, write outputs.

    Args:
        settings: Run settings.
        config: Options.

    Returns:
        The summary; ``results`` holds one entry per probe whose items are
        all complete.
    """
    items = build_items(settings, config)
    rows, book = run_items(
        [it for p in PROBES for it in items[p]],
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
    by_probe: dict[str, list[dict[str, Any]]] = {p: [] for p in PROBES}
    for r in rows:
        by_probe[probe_of(r["unit"])].append(r)
    summary: dict[str, Any] = {
        "config": as_dict(config),
        **book,
        "items_by_probe": {p: len(items[p]) for p in PROBES},
        "complete_by_probe": {p: len(by_probe[p]) for p in PROBES},
    }
    results: dict[str, Any] = {}
    for p in PROBES:
        if len(by_probe[p]) != len(items[p]):
            logger.warning("%s: %s incomplete; no analysis", NAME, p)
            continue
        if p == "wording":
            results[p] = analyse_wording(by_probe[p], config)
        elif p == "odds_formats":
            results[p] = analyse_odds(by_probe[p], config)
        elif p == "cutoff":
            results[p] = analyse_cutoff(by_probe[p], config)
        else:
            results[p] = analyse_warned(by_probe[p], config, _unwarned_rows(settings))
    if results:
        summary["results"] = results
    write_json(out / "summary.json", summary)
    return summary


EXPERIMENT = Experiment(
    name=NAME,
    summary="Re-worded meta-questions, odds formats, cutoff cues, warned paragraphs",
    config_type=Config,
    requests=requests,
    run=run,
)
