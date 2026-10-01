"""Shortcut baselines for the follow-up readouts (offline; reads saved rows).

High AUROCs for ``known``, ``enough`` or ``settled`` might reflect cues in
the input (unfamiliar-looking names, dates, text length) rather than what
the model knows. Everything here reuses scored rows and deterministic item
builders; no requests are made.

First group (one bootstrap generator, in this order):

1. ``entity_text_baselines``: fabricated versus real subjects. Character
   n-gram model of the question alone, ``known``, confidence, both
   combined, and ``known`` within text-score quintiles; ``known`` tracking
   correctness within popularity quintiles.
2. ``news_text_baselines``: news after versus before the knowledge cutoff.
   Date-only score, text model with and without dates, ``known`` and
   confidence (month-cluster intervals); correctness tracking within period
   and month.
3. ``second_look_baseline``: the second-look questions next to ``known``
   on the same items, with paired AUROC differences.
4. ``proper_scores``: multiclass Brier and log loss for every scored set,
   and expected scores against stated distributions for chance tasks.
5. ``length_checks``: text length by evidence condition, and ``enough``
   against a length-only score.
6. ``recalibration_groupings`` and ``forward_chaining``: merging maps and
   recalibration on earlier months only (see
   :mod:`beyond_answer_confidence.analyses.level_merging`).
7. ``label_balance``: news label balance by period, month and category.

Second group (a fresh generator with the same seed, in this order):

8. ``enough_length``: ``enough`` within narrow word-count strata and
   against cross-validated length-feature models (folds grouped by
   question).
9. ``paired_known_vs_confidence``: ``known`` minus confidence on the
   cue-controlled comparisons (look-alike names, dates removed).
10. ``grouped_text_baselines``: text-only baselines with folds grouped by
    subject or month.
11. ``date_routing``: answering in order of the question's own date against
    the "not known" option at matched coverage, over post-cutoff shares.
12. ``enough_decisions``: held-out answer acceptance by ``enough`` versus
    confidence at matched coverage.
13. ``coverage_table``: coverage and confident errors of the cutoff
    conditions.
14. ``floor_sensitivity``: log loss under several probability floors.
15. ``masked_label_safe``: the date-masked run on answer-safe subsets.
16. ``date_only_subset``: the date-only baseline on the masked items.
17. ``matched_confidence_baselines``: ``enough`` and ``settled`` against
    answer confidence on the same contrasts.

Inputs: rows of ``knowledge_boundary``, ``second_look``,
``evidence_sufficiency``, ``benchmarks``, ``option_count``,
``contrastive_facts``, ``synthetic_worlds``, ``stated_odds``,
``follow_up_questions`` and ``shortcut_controls`` under the output
directory, and the PopQA, Daily Oracle and HotpotQA datasets. Output:
``<output_dir>/shortcut_baselines/summary.json``.
"""

import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from beyond_answer_confidence.analyses.cutoff_calibration import (
    BLOCK_MONTHS,
    REPORT_SHARES,
    SHARES,
    at_share,
    paired_rows,
)
from beyond_answer_confidence.analyses.level_merging import (
    forward_chaining,
    recalibration_groupings,
)
from beyond_answer_confidence.config import as_dict
from beyond_answer_confidence.data import loaders
from beyond_answer_confidence.experiments.base import (
    Analysis,
    read_jsonl,
    source_rows,
    write_json,
)
from beyond_answer_confidence.experiments.shortcut_controls import (
    entity_items,
    obscure_real_units,
    split_rows,
)
from beyond_answer_confidence.metrics.distributions import brier, expected_scores, logit
from beyond_answer_confidence.metrics.ranking import (
    auroc,
    stratified_auroc,
    weighted_accuracy_at,
)
from beyond_answer_confidence.settings import Settings
from beyond_answer_confidence.stats.crossfit import (
    grouped_oof,
    oof_features,
    oof_text_scores,
)
from beyond_answer_confidence.stats.resampling import (
    auroc_interval,
    month_block_indices,
    paired_auroc_difference,
)
from beyond_answer_confidence.tasks.chance import DEVICES
from beyond_answer_confidence.tasks.contrastive import contrastive_items
from beyond_answer_confidence.tasks.evidence import hotpot_items
from beyond_answer_confidence.tasks.follow_up import CUTOFF_MONTH
from beyond_answer_confidence.tasks.news import yes_no_news_items
from beyond_answer_confidence.tasks.schema import Item
from beyond_answer_confidence.tasks.shortcuts import (
    latest_date,
    lookalike_items,
    mask_dates,
)

NAME = "shortcut_baselines"

EPS = 1e-4
"""Log-loss floor for the proper scores (exact zeros occur)."""

FLOORS = (1e-2, 1e-3, 1e-4, 1e-6)
"""Probability floors for the log-loss sensitivity table."""

COVERAGES = (0.25, 0.5, 0.75, 0.9)
"""Target coverages for the held-out acceptance comparison."""

PROPER_SETS: dict[str, str | None] = {
    "knowledge_boundary": None,
    "evidence_sufficiency": "cell",
    "benchmarks": None,
    "option_count": None,
    "contrastive_facts": "variant",
}
"""Experiments in the proper-score table, with an optional sub-grouping column."""

SOURCES = (
    *PROPER_SETS,
    "second_look",
    "synthetic_worlds",
    "stated_odds",
    "follow_up_questions",
    "shortcut_controls",
)
"""Experiments whose rows are read."""


@dataclass(frozen=True)
class Config:
    """Options.

    Attributes:
        bootstrap: Bootstrap resamples.
        seed: Seed of both bootstrap generators.
        item_seed: Seed of the source item sets.
        yes_no_per_month: News questions per month in the source set.
        fabricated_per_relation: Fabricated entities per relation.
        lookalike_per_relation: Look-alike names per person relation.
        cases_per_policy: Contrastive cases per policy.
        cutoff_month: First month counted as after the knowledge cutoff.
        fold_seed: Seed of the stratified cross-validation folds.
        recalibration_splits: Month half-splits for cross-fitted merging maps.
        null_months: Random-month nulls for the in-sample interval optimum.
        decision_splits: Question half-splits for held-out acceptance.
        min_train_months: Months before the first forward-chaining test month.
        date_edit_labels: Optional JSON file of hand labels for the masked
            news questions (``[{"unit": ..., "label": ...}]``; label ``P``
            marks an edit that preserved the answer). Empty = skip.
    """

    bootstrap: int = 1000
    seed: int = 0
    item_seed: int = 0
    yes_no_per_month: int = 80
    fabricated_per_relation: int = 100
    lookalike_per_relation: int = 100
    cases_per_policy: int = 100
    cutoff_month: str = CUTOFF_MONTH
    fold_seed: int = 0
    recalibration_splits: int = 50
    null_months: int = 200
    decision_splits: int = 100
    min_train_months: int = 12
    date_edit_labels: str = ""


@dataclass(frozen=True)
class Inputs:
    """Everything the analysis reads, already loaded.

    Attributes:
        rows: Experiment -> rows (``follow_up_questions`` holds only the
            cutoff rows).
        entities: PopQA items followed by the fabricated twins.
        subjects: PopQA unit -> subject.
        news: Yes/no news items of the source set.
        hotpot: HotpotQA items.
        contrastive: Contrastive-facts items.
        lookalike: Look-alike items.
        date_edit_labels: Masked-news unit -> hand label, if available.
    """

    rows: Mapping[str, Sequence[Mapping[str, Any]]]
    entities: Sequence[Item]
    subjects: Mapping[str, str]
    news: Sequence[Item]
    hotpot: Sequence[Item]
    contrastive: Sequence[Item]
    lookalike: Sequence[Item]
    date_edit_labels: Mapping[str, str] | None = None


# --- helpers -----------------------------------------------------------------


def _by_unit(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {str(r["unit"]): r for r in rows}


def word_count(state: Any) -> int:
    """Whitespace-separated words in every string of a (nested) state.

    Args:
        state: Request state (strings, mappings and sequences).

    Returns:
        The number of words.
    """
    if isinstance(state, str):
        return len(state.split())
    if isinstance(state, Mapping):
        return sum(word_count(x) for x in state.values())
    if isinstance(state, list | tuple):
        return sum(word_count(x) for x in state)
    return 0


def _paired(
    a: np.ndarray,
    c: np.ndarray,
    y: np.ndarray,
    b: int,
    rng: np.random.Generator,
    clusters: np.ndarray | None = None,
) -> dict[str, Any]:
    """AUROC(a) - AUROC(c) with 95 % and 90 % paired-bootstrap intervals."""
    return paired_auroc_difference(a, c, y, b, rng, clusters)


# --- 1-2. text-only baselines ----------------------------------------------------


def entity_baselines(
    entities: Sequence[Item],
    first: Mapping[str, Mapping[str, Any]],
    b: int,
    rng: np.random.Generator,
    fold_seed: int = 0,
) -> dict[str, Any]:
    """Fabricated versus real subjects: text-only baselines against ``known``.

    Args:
        entities: PopQA items followed by the fabricated twins.
        first: Knowledge-boundary rows by unit.
        b: Bootstrap resamples.
        rng: Random generator.
        fold_seed: Fold seed of the out-of-fold models.

    Returns:
        AUROCs (fabricated = positive) of the text model, ``1 - known``,
        ``1 - confidence``, text plus ``known``, and ``known`` within
        text-score quintiles, for all real and for the least popular fifth;
        and ``known`` tracking correctness among real items.
    """
    items = [it for it in entities if it.unit in first]
    df = pd.DataFrame(
        {
            "unit": [it.unit for it in items],
            "question": [str(it.state["question"]) for it in items],
            "fabricated": [bool(it.info["fabricated"]) for it in items],
            "s_pop": [it.info.get("s_pop", -1) for it in items],
        }
    )
    df["p_known"] = [first[u]["p_known"] for u in df["unit"]]
    df["p_max"] = [first[u]["p_max"] for u in df["unit"]]
    df["correct"] = [first[u]["correct"] for u in df["unit"]]
    real = df[~df["fabricated"]].copy()
    real["quintile"] = pd.qcut(real["s_pop"].rank(method="first"), 5, labels=False)
    df["quintile"] = -1
    df.loc[real.index, "quintile"] = real["quintile"]
    y = df["fabricated"].to_numpy(bool)
    text = oof_text_scores(df["question"].tolist(), y, fold_seed)
    df["text"] = text
    combo = oof_features(
        np.column_stack([text, logit(df["p_known"].to_numpy())]), y, fold_seed
    )
    out: dict[str, Any] = {
        "n_fabricated": int(y.sum()),
        "n_real": int((~y).sum()),
        "all_real": {
            "text_char_ngram": auroc_interval(text, y, b, rng),
            "known": auroc_interval(-df["p_known"].to_numpy(), y, b, rng),
            "confidence": auroc_interval(-df["p_max"].to_numpy(), y, b, rng),
            "text_plus_known": auroc_interval(combo, y, b, rng),
            "known_within_text_quintiles": stratified_auroc(
                -df["p_known"].to_numpy(), y, pd.qcut(text, 5, labels=False)
            ),
        },
    }
    least = df[(df["quintile"] == 0) | df["fabricated"]].reset_index(drop=True)
    yl = least["fabricated"].to_numpy(bool)
    text_l = oof_text_scores(least["question"].tolist(), yl, fold_seed)
    combo_l = oof_features(
        np.column_stack([text_l, logit(least["p_known"].to_numpy())]), yl, fold_seed
    )
    out["least_popular_real"] = {
        "text_char_ngram": auroc_interval(text_l, yl, b, rng),
        "known": auroc_interval(-least["p_known"].to_numpy(), yl, b, rng),
        "confidence": auroc_interval(-least["p_max"].to_numpy(), yl, b, rng),
        "text_plus_known": auroc_interval(combo_l, yl, b, rng),
        "known_within_text_quintiles": stratified_auroc(
            -least["p_known"].to_numpy(), yl, pd.qcut(text_l, 5, labels=False)
        ),
    }
    ok = real["correct"].to_numpy(bool)
    out["known_tracks_correctness_real"] = {
        "known_pooled": auroc_interval(real["p_known"].to_numpy(), ok, b, rng),
        "confidence_pooled": auroc_interval(real["p_max"].to_numpy(), ok, b, rng),
        "known_within_quintile": stratified_auroc(
            real["p_known"].to_numpy(), ok, real["quintile"].to_numpy()
        ),
        "confidence_within_quintile": stratified_auroc(
            real["p_max"].to_numpy(), ok, real["quintile"].to_numpy()
        ),
        "by_quintile": {
            int(cast(int, q)): {
                "n": len(g),
                "accuracy": float(g["correct"].mean()),
                "known": auroc(g["p_known"].to_numpy(), g["correct"].to_numpy(bool)),
                "confidence": auroc(g["p_max"].to_numpy(), g["correct"].to_numpy(bool)),
            }
            for q, g in real.groupby("quintile")
        },
    }
    return out


def news_frame(
    news: Sequence[Item],
    first: Mapping[str, Mapping[str, Any]],
    cutoff_month: str = CUTOFF_MONTH,
) -> pd.DataFrame:
    """Yes/no news rows joined with their question text.

    Args:
        news: Yes/no news items.
        first: Knowledge-boundary rows by unit.
        cutoff_month: First month counted as after the cutoff.

    Returns:
        One row per scored question: ``unit``, ``question``, ``month``,
        ``category``, ``gold``, ``p_known``, ``p_max``, ``correct``, ``top``,
        ``p_yes`` and ``post``.
    """
    items = [it for it in news if it.unit in first]
    df = pd.DataFrame(
        {
            "unit": [it.unit for it in items],
            "question": [str(it.state["question"]) for it in items],
            "month": [str(it.info["month"]) for it in items],
            "category": [str(it.info["category"]) for it in items],
            "gold": [str(it.gold) for it in items],
        }
    )
    for k in ("p_known", "p_max", "correct", "top"):
        df[k] = [first[u][k] for u in df["unit"]]
    df["p_yes"] = [first[u]["dist"]["yes"] for u in df["unit"]]
    df["post"] = df["month"] >= cutoff_month
    return df


def news_baselines(
    df: pd.DataFrame, b: int, rng: np.random.Generator, fold_seed: int = 0
) -> dict[str, Any]:
    """News after versus before the cutoff: date and text baselines against ``known``.

    Args:
        df: :func:`news_frame` output.
        b: Bootstrap resamples.
        rng: Random generator.
        fold_seed: Fold seed of the out-of-fold models.

    Returns:
        AUROCs with month-cluster intervals (after = positive), and
        within-period tracking of correctness.
    """
    y = df["post"].to_numpy(bool)
    months = df["month"].to_numpy()
    masked = [mask_dates(q) for q in df["question"]]
    date = np.array([latest_date(q) for q in df["question"]])
    has_date = ~np.isnan(date)
    date_filled = np.where(has_date, date, np.nanmedian(date))
    text = oof_text_scores(df["question"].tolist(), y, fold_seed)
    text_masked = oof_text_scores(masked, y, fold_seed)
    residual_dates = [q for q in masked if re.search(r"(?:19|20)\d\d", q)]
    combo = oof_features(
        np.column_stack([text_masked, logit(df["p_known"].to_numpy())]), y, fold_seed
    )
    out: dict[str, Any] = {
        "n_post": int(y.sum()),
        "n_pre": int((~y).sum()),
        "share_with_year": float(has_date.mean()),
        "masked_with_residual_year": len(residual_dates),
        "date_only": auroc_interval(date_filled, y, b, rng, months),
        "text_char_ngram": auroc_interval(text, y, b, rng, months),
        "text_date_masked": auroc_interval(text_masked, y, b, rng, months),
        "known": auroc_interval(-df["p_known"].to_numpy(), y, b, rng, months),
        "confidence": auroc_interval(-df["p_max"].to_numpy(), y, b, rng, months),
        "masked_text_plus_known": auroc_interval(combo, y, b, rng, months),
    }
    within: dict[str, Any] = {}
    for name, g in (("pre", df[~df["post"]]), ("post", df[df["post"]])):
        ok = g["correct"].to_numpy(bool)
        within[name] = {
            "n": len(g),
            "accuracy": float(ok.mean()),
            "known_vs_correct": auroc_interval(
                g["p_known"].to_numpy(), ok, b, rng, g["month"].to_numpy()
            ),
            "confidence_vs_correct": auroc_interval(
                g["p_max"].to_numpy(), ok, b, rng, g["month"].to_numpy()
            ),
            "known_vs_correct_within_month": stratified_auroc(
                g["p_known"].to_numpy(), ok, g["month"].to_numpy()
            ),
        }
    out["within_period"] = within
    pre = df[~df["post"]]
    out["known_vs_month_pre_spearman"] = float(
        pd.Series(pre["p_known"].to_numpy()).corr(
            pd.Series(pd.to_datetime(pre["month"]).map(pd.Timestamp.toordinal)),
            method="spearman",
        )
    )
    return out


# --- 3. second look as a baseline ------------------------------------------------


def second_look_baseline(
    second: Sequence[Mapping[str, Any]],
    first: Mapping[str, Mapping[str, Any]],
    b: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """The second-look questions next to ``known`` on the same items.

    Args:
        second: Second-look rows.
        first: Knowledge-boundary rows by unit.
        b: Bootstrap resamples.
        rng: Random generator.

    Returns:
        For fabricated versus real (item bootstrap) and after versus before
        the cutoff (month clusters): AUROCs of ``1 - known``, ``1 -
        P(correct)``, ``1 - chance score`` and ``1 - confidence``, and the
        paired difference ``known`` minus P(correct).
    """
    k = pd.DataFrame(list(second))
    k["base"] = k["unit"]
    k["p_known"] = [first.get(u, {}).get("p_known", np.nan) for u in k["base"]]
    k["month"] = [first.get(u, {}).get("month") for u in k["base"]]
    out: dict[str, Any] = {}
    for name, sub, label, clusters in (
        ("fabricated_vs_real", k[k["set"] == "popqa"], "made_up", None),
        ("post_vs_pre", k[k["set"] == "oracle_tf"], "post", "month"),
    ):
        sub = sub.dropna(subset=["p_known"])
        y = sub[label].to_numpy(bool)
        cl = sub[clusters].to_numpy() if clusters else None
        scores = {
            "known": -sub["p_known"].to_numpy(),
            "second_look_yes_no": -sub["p_correct"].to_numpy(),
            "second_look_score": -sub["s_chance"].to_numpy(),
            "confidence": -sub["first_p_max"].to_numpy(),
        }
        out[name] = {
            "n": len(sub),
            "n_positive": int(y.sum()),
            **{s: auroc_interval(v, y, b, rng, cl) for s, v in scores.items()},
            "known_minus_second_look": _paired(
                scores["known"], scores["second_look_yes_no"], y, b, rng, cl
            ),
        }
    return out


# --- 4. proper scoring rules -------------------------------------------------------


def _scores(dist: Mapping[str, float], gold: str) -> tuple[float, float, bool]:
    return (
        brier(dist, gold),
        -float(np.log(max(dist.get(gold, 0.0), EPS))),
        dist.get(gold, 0.0) == 0.0,
    )


def _set_key(exp: str, r: Mapping[str, Any]) -> str:
    return f"{exp}:{r['set']}" if r.get("set") else exp


def proper_scores(
    tables: Mapping[str, Sequence[Mapping[str, Any]]],
    synthetic: Sequence[Mapping[str, Any]],
    devices: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Multiclass Brier and log loss per scored set, and chance-task expected scores.

    Args:
        tables: Experiment -> rows, for the experiments in :data:`PROPER_SETS`.
        synthetic: Synthetic-worlds rows (expected scores against ``ideal``).
        devices: Stated-odds rows (devices against their true odds).

    Returns:
        Per group: n, accuracy, Brier, log loss (floor :data:`EPS`), share
        with zero probability on the answer, and uniform references; per
        chance cell or device: expected Brier and log loss, and those of the
        stated distribution itself.
    """
    out: dict[str, Any] = {"log_loss_floor": EPS, "sets": {}, "stated_odds": {}}
    for exp, sub in PROPER_SETS.items():
        rows = [r for r in tables[exp] if r.get("gold") is not None and r.get("dist")]
        groups: dict[str, list[Mapping[str, Any]]] = {}
        for r in rows:
            key = _set_key(exp, r)
            if sub and r.get(sub) is not None:
                key += f":{r[sub]}"
            groups.setdefault(key, []).append(r)
        for key, g in sorted(groups.items()):
            sc = np.array([_scores(r["dist"], r["gold"]) for r in g], float)
            k = np.array([len(r["dist"]) for r in g], float)
            out["sets"][key] = {
                "n": len(g),
                "accuracy": float(np.mean([bool(r["correct"]) for r in g])),
                "brier": float(sc[:, 0].mean()),
                "log_loss": float(sc[:, 1].mean()),
                "zero_on_answer": float(sc[:, 2].mean()),
                "uniform_brier": float(np.mean(1 - 1 / k)),
                "uniform_log_loss": float(np.mean(np.log(k))),
            }
    cells: dict[str, list[dict[str, float]]] = {}
    for r in synthetic:
        if r.get("ideal") and not r.get("alone"):
            cells.setdefault(f"synthetic_worlds:{r['cell']}", []).append(
                expected_scores(r["dist"], r["ideal"], EPS)
            )
    for r in devices:
        if r["set"] == "device":
            cells.setdefault(f"stated_odds:{r['device']}", []).append(
                expected_scores(r["dist"], DEVICES[r["device"]][1], EPS)
            )
    for key, v in sorted(cells.items()):
        out["stated_odds"][key] = {
            "n": len(v),
            "expected_brier": float(np.mean([e["brier"] for e in v])),
            "expected_log_loss": float(np.mean([e["log_loss"] for e in v])),
            "ideal_brier": float(np.mean([e["ideal_brier"] for e in v])),
            "ideal_log_loss": float(np.mean([e["ideal_log_loss"] for e in v])),
        }
    return out


def floor_sensitivity(
    tables: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Mean log loss under several probability floors (no renormalisation).

    Args:
        tables: Experiment -> rows, for the experiments in :data:`PROPER_SETS`.

    Returns:
        Per set: log loss at each floor in :data:`FLOORS` and the share with
        zero probability on the answer.
    """
    out: dict[str, Any] = {"floors": list(FLOORS), "sets": {}}
    for exp in PROPER_SETS:
        groups: dict[str, list[float]] = {}
        for r in tables[exp]:
            if r.get("gold") is None or not r.get("dist"):
                continue
            groups.setdefault(_set_key(exp, r), []).append(
                float(r["dist"].get(r["gold"], 0.0))
            )
        for key, pg in sorted(groups.items()):
            p = np.array(pg)
            out["sets"][key] = {
                "n": len(p),
                "zero_on_answer": float((p == 0).mean()),
                **{
                    f"log_loss_floor_{f:g}": float(-np.log(np.maximum(p, f)).mean())
                    for f in FLOORS
                },
            }
    return out


# --- 5. length checks for enough ---------------------------------------------------


def hotpot_frame(
    evidence: Sequence[Mapping[str, Any]],
    hotpot: Sequence[Item],
    cells: Sequence[str] | None = None,
) -> pd.DataFrame:
    """HotpotQA evidence rows with their word counts.

    Args:
        evidence: Evidence-sufficiency rows.
        hotpot: HotpotQA items.
        cells: Cells to keep (all if ``None``).

    Returns:
        Rows of set ``hotpot`` with ``words`` (whole state).
    """
    items = {it.unit: it for it in hotpot}
    df = pd.DataFrame(
        [
            r
            for r in evidence
            if r["set"] == "hotpot" and (cells is None or r["cell"] in cells)
        ]
    )
    df["words"] = [word_count(items[u].state) for u in df["unit"]]
    return df


def length_checks(
    evidence: Sequence[Mapping[str, Any]],
    hotpot: Sequence[Item],
    contrastive: Sequence[Item],
    policy: Sequence[Mapping[str, Any]],
    b: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Text length by evidence condition, and ``enough`` against length alone.

    Args:
        evidence: Evidence-sufficiency rows.
        hotpot: HotpotQA items.
        contrastive: Contrastive-facts items.
        policy: Contrastive-facts rows.
        b: Bootstrap resamples.
        rng: Random generator.

    Returns:
        Word counts per HotpotQA cell and policy variant; AUROCs of
        ``enough`` and of length alone for each contrast (question or case
        clusters).
    """
    items = {it.unit: it for it in hotpot}
    hot = hotpot_frame(evidence, hotpot)
    hot["paragraphs"] = [len(items[u].state.get("paragraphs", [])) for u in hot["unit"]]
    out: dict[str, Any] = {
        "hotpot_by_cell": {
            str(c): {
                "n": len(g),
                "paragraphs": sorted(set(g["paragraphs"])),
                "words_mean": float(g["words"].mean()),
                "words_sd": float(g["words"].std()),
                "p_enough": float(g["p_enough"].mean()),
            }
            for c, g in hot.groupby("cell")
        }
    }
    for name, cells in (
        ("dose2_vs_all_other_cells", ("closed", "dose0", "dose1")),
        ("dose2_vs_two_paragraph_cells", ("dose0", "dose1")),
    ):
        g = hot[hot["cell"].isin(("dose2", *cells))]
        y = (g["cell"] == "dose2").to_numpy()
        qid = g["qid"].to_numpy()
        out[name] = {
            "enough": auroc_interval(g["p_enough"].to_numpy(), y, b, rng, qid),
            "length_only_longer": auroc_interval(g["words"].to_numpy(), y, b, rng, qid),
        }
    two = hot[hot["cell"].isin(("dose0", "dose1", "dose2"))]
    within = two.groupby("qid")["words"].agg(lambda s: s.max() - s.min())
    out["two_paragraph_within_question_word_range"] = {
        "median": float(within.median()),
        "mean": float(within.mean()),
    }
    lens = {it.unit: word_count(it.state) for it in contrastive}
    pol = pd.DataFrame(list(policy))
    pol["words"] = [lens[u] for u in pol["unit"]]
    out["policy_by_variant"] = {
        str(v): {
            "n": len(g),
            "words_mean": float(g["words"].mean()),
            "words_sd": float(g["words"].std()),
            "p_enough": float(g["p_enough"].mean()),
        }
        for v, g in pol.groupby("variant")
    }
    pairs = pol[pol["variant"].isin(("base", "deleted"))]
    y = (pairs["variant"] == "deleted").to_numpy()
    case = (pairs["domain"] + ":" + pairs["case"].astype(str)).to_numpy()
    wide = pairs.pivot_table(
        index=["domain", "case"], columns="variant", values="words"
    )
    out["policy_deleted_vs_base"] = {
        "enough": auroc_interval(-pairs["p_enough"].to_numpy(), y, b, rng, case),
        "length_only_shorter": auroc_interval(
            -pairs["words"].to_numpy(), y, b, rng, case
        ),
        "deleted_minus_base_words": {
            "mean": float((wide["deleted"] - wide["base"]).mean()),
            "share_shorter": float((wide["deleted"] < wide["base"]).mean()),
            "share_equal": float((wide["deleted"] == wide["base"]).mean()),
            "share_longer": float((wide["deleted"] > wide["base"]).mean()),
        },
    }
    und = pol["variant"].isin(("deleted", "contradictory")).to_numpy()
    out["policy_undecidable_vs_rest"] = {
        "enough": auroc_interval(
            -pol["p_enough"].to_numpy(),
            und,
            b,
            rng,
            (pol["domain"] + ":" + pol["case"].astype(str)).to_numpy(),
        ),
        "length_only_longer": auroc(pol["words"].to_numpy(), und),
        "length_only_shorter": auroc(-pol["words"].to_numpy(), und),
    }
    return out


# --- 7. label balance --------------------------------------------------------------


def label_balance(df: pd.DataFrame, cutoff_month: str = CUTOFF_MONTH) -> dict[str, Any]:
    """Share of "no" answers (true and predicted) by period, month and category.

    Args:
        df: :func:`news_frame` output.
        cutoff_month: First month counted as after the cutoff.

    Returns:
        Per period: true and predicted "no" shares and accuracies by
        predicted and true answer; per category and period (at least 30
        questions); the monthly range of the true "no" share; the category
        mix change; and the within-category accuracy drop.
    """
    d = df.assign(
        gold_no=df["gold"] == "no",
        pred_no=df["top"] == "no",
        period=np.where(df["post"], "post", "pre"),
    )
    by_period = {
        str(p): {
            "n": len(g),
            "true_no": float(g["gold_no"].mean()),
            "predicted_no": float(g["pred_no"].mean()),
            "accuracy": float(g["correct"].mean()),
            "accuracy_when_predicting_no": float(g[g["pred_no"]]["correct"].mean()),
            "accuracy_when_predicting_yes": float(g[~g["pred_no"]]["correct"].mean())
            if (~g["pred_no"]).any()
            else None,
            "accuracy_on_true_no": float(g[g["gold_no"]]["correct"].mean()),
            "accuracy_on_true_yes": float(g[~g["gold_no"]]["correct"].mean()),
        }
        for p, g in d.groupby("period")
    }
    by_cat = {
        f"{c}|{p}": {
            "n": len(g),
            "true_no": float(g["gold_no"].mean()),
            "predicted_no": float(g["pred_no"].mean()),
            "accuracy": float(g["correct"].mean()),
        }
        for (c, p), g in d.groupby(["category", "period"])
        if len(g) >= 30
    }
    by_month = d.groupby("month").agg(
        true_no=("gold_no", "mean"),
        predicted_no=("pred_no", "mean"),
        n=("gold_no", "size"),
    )
    drop, weight = 0.0, 0
    for _, g in d.groupby("category"):
        pre, post = g[g["period"] == "pre"], g[g["period"] == "post"]
        if len(pre) and len(post):
            drop += (pre["correct"].mean() - post["correct"].mean()) * len(post)
            weight += len(post)
    comp = d.groupby(["period", "category"]).size().unstack(0).fillna(0)
    comp = comp / comp.sum()
    return {
        "by_period": by_period,
        "by_category_period": by_cat,
        "true_no_by_month_range": {
            ("post" if p else "pre"): [
                float(g["true_no"].min()),
                float(g["true_no"].max()),
            ]
            for p, g in by_month.groupby(by_month.index >= cutoff_month)
        },
        "category_share_tv_pre_post": float(
            0.5 * (comp["pre"] - comp["post"]).abs().sum()
        ),
        "within_category_accuracy_drop_post_weighted": float(drop / weight)
        if weight
        else None,
    }


# --- 8. enough with length controlled ------------------------------------------------


def enough_length(
    evidence: Sequence[Mapping[str, Any]],
    hotpot: Sequence[Item],
    contrastive: Sequence[Item],
    policy: Sequence[Mapping[str, Any]],
    b: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """``enough`` beyond text length, on the two-paragraph HotpotQA conditions.

    Args:
        evidence: Evidence-sufficiency rows.
        hotpot: HotpotQA items.
        contrastive: Contrastive-facts items.
        policy: Contrastive-facts rows.
        b: Bootstrap resamples.
        rng: Random generator.

    Returns:
        AUROCs within word-count strata; question-grouped out-of-fold AUROCs
        of gradient-boosted length-feature models with and without
        ``enough`` and their paired difference; closest-length pairs within
        a question; and the policy cases within exact word counts.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier

    items = {it.unit: it for it in hotpot}
    df = hotpot_frame(evidence, hotpot, ("dose0", "dose1", "dose2"))
    paras = [list(items[u].state.get("paragraphs", [])) for u in df["unit"]]
    df["p1"] = [len(p[0].split()) for p in paras]
    df["p2"] = [len(p[1].split()) for p in paras]
    df["q_words"] = [len(str(items[u].state["question"]).split()) for u in df["unit"]]
    y = (df["cell"] == "dose2").to_numpy()
    qid = df["qid"].to_numpy()
    out: dict[str, Any] = {
        "n": len(df),
        "enough": auroc_interval(df["p_enough"].to_numpy(), y, b, rng, qid),
        "shorter_is_complete": auroc_interval(-df["words"].to_numpy(), y, b, rng, qid),
    }
    for width in (10, 20):
        strata = (df["words"] // width).to_numpy()
        out[f"enough_within_{width}_word_strata"] = stratified_auroc(
            df["p_enough"].to_numpy(), y, strata
        )
        out[f"shorter_within_{width}_word_strata"] = stratified_auroc(
            -df["words"].to_numpy(), y, strata
        )
    feats = np.column_stack(
        [
            df["words"],
            np.log(df["words"]),
            np.minimum(df["p1"], df["p2"]),
            np.maximum(df["p1"], df["p2"]),
            df["q_words"],
        ]
    ).astype(float)
    with_enough = np.column_stack([feats, logit(df["p_enough"].to_numpy())])

    def gbm() -> Any:
        return HistGradientBoostingClassifier(max_iter=200, random_state=0)

    oof_len = grouped_oof(gbm, feats, y, qid)
    oof_both = grouped_oof(gbm, with_enough, y, qid)
    out["length_features_gbm"] = auroc_interval(oof_len, y, b, rng, qid)
    out["length_features_plus_enough_gbm"] = auroc_interval(oof_both, y, b, rng, qid)
    out["plus_enough_minus_length_only"] = _paired(oof_both, oof_len, y, b, rng, qid)
    # matched pairs: each complete set against the incomplete set of the same
    # question whose length is closest (both cells hold two paragraphs)
    gaps, wins = [], []
    for _, g in df.groupby("qid"):
        comp = g[g["cell"] == "dose2"]
        inc = g[g["cell"] != "dose2"]
        if comp.empty or inc.empty:
            continue
        c = comp.iloc[0]
        j = (inc["words"] - c["words"]).abs().idxmin()
        gaps.append(abs(int(inc.loc[j, "words"]) - int(c["words"])))
        wins.append(
            float(c["p_enough"] > inc.loc[j, "p_enough"])
            + 0.5 * float(c["p_enough"] == inc.loc[j, "p_enough"])
        )
    gaps_a, wins_a = np.array(gaps), np.array(wins)
    close = gaps_a <= 10
    out["within_question_closest_length_pairs"] = {
        "pairs": len(wins_a),
        "complete_ranked_higher": float(wins_a.mean()),
        "pairs_within_10_words": int(close.sum()),
        "complete_ranked_higher_within_10_words": float(wins_a[close].mean())
        if close.any()
        else None,
    }
    lens = {it.unit: word_count(it.state) for it in contrastive}
    pol = pd.DataFrame(list(policy))
    pol = pol[pol["variant"].isin(("base", "deleted"))].copy()
    pol["words"] = [lens[u] for u in pol["unit"]]
    yp = (pol["variant"] == "deleted").to_numpy()
    out["policy_deleted_vs_base_same_word_count"] = stratified_auroc(
        -pol["p_enough"].to_numpy(), yp, pol["words"].to_numpy()
    )
    out["policy_shorter_is_deleted"] = auroc(-pol["words"].to_numpy(), yp)
    return out


# --- 9. paired differences for the cue-controlled comparisons -----------------------


def paired_known_vs_confidence(
    entities: Sequence[Item],
    first: Mapping[str, Mapping[str, Any]],
    lookalike: Sequence[Mapping[str, Any]],
    masked: Sequence[Mapping[str, Any]],
    b: int,
    rng: np.random.Generator,
    cutoff_month: str = CUTOFF_MONTH,
) -> dict[str, Any]:
    """``known`` minus answer confidence on the same items, for each control.

    Args:
        entities: PopQA items followed by the fabricated twins.
        first: Knowledge-boundary rows by unit.
        lookalike: Look-alike rows of the shortcut controls.
        masked: Masked-news rows of the shortcut controls.
        b: Bootstrap resamples.
        rng: Random generator.
        cutoff_month: First month counted as after the cutoff.

    Returns:
        Paired AUROC differences with 95 % and 90 % intervals.
    """
    real = obscure_real_units(entities, first)
    known = np.r_[
        [first[u]["p_known"] for u in real["unit"]], [r["p_known"] for r in lookalike]
    ]
    conf = np.r_[
        [first[u]["p_max"] for u in real["unit"]], [r["p_max"] for r in lookalike]
    ]
    y = np.r_[np.zeros(len(real), bool), np.ones(len(lookalike), bool)]
    out: dict[str, Any] = {
        "lookalike_vs_obscure_real": {
            "n_made_up": int(y.sum()),
            "n_real": int((~y).sum()),
            "orientation": "higher score = made up (1 - P(known), 1 - p_max)",
            "known_minus_confidence": _paired(-known, -conf, y, b, rng),
        }
    }
    m = pd.DataFrame(list(masked))
    post = (m["month"] >= cutoff_month).to_numpy()
    orig = pd.DataFrame([first[u] for u in m["base"]])
    for name, d in (("dates_in_question", orig), ("dates_removed", m)):
        out[f"news_{name}"] = {
            "n_post": int(post.sum()),
            "n_pre": int((~post).sum()),
            "orientation": "higher score = after the cutoff",
            "known_minus_confidence": _paired(
                -d["p_known"].to_numpy(),
                -d["p_max"].to_numpy(),
                post,
                b,
                rng,
                m["month"].to_numpy(),
            ),
        }
    return out


# --- 10. grouped text baselines ------------------------------------------------------


def grouped_text_baselines(
    entities: Sequence[Item],
    subjects: Mapping[str, str],
    first: Mapping[str, Mapping[str, Any]],
    lookalike_items_: Sequence[Item],
    lookalike: Sequence[Mapping[str, Any]],
    news: Sequence[Item],
    b: int,
    rng: np.random.Generator,
    cutoff_month: str = CUTOFF_MONTH,
) -> dict[str, Any]:
    """Character n-gram baselines with subject- (or month-) grouped folds.

    Args:
        entities: PopQA items followed by the fabricated twins.
        subjects: PopQA unit -> subject.
        first: Knowledge-boundary rows by unit.
        lookalike_items_: Look-alike items (question text).
        lookalike: Look-alike rows of the shortcut controls.
        news: Yes/no news items.
        b: Bootstrap resamples.
        rng: Random generator.
        cutoff_month: First month counted as after the cutoff.

    Returns:
        AUROCs of grouped out-of-fold text models, and the grouping used.
    """
    from sklearn.linear_model import LogisticRegression

    items = [it for it in entities if it.unit in first]
    texts = [str(it.state["question"]) for it in items]
    y = np.array([bool(it.info["fabricated"]) for it in items])
    groups = np.array(
        [
            str(it.info.get("subject")) if it.info["fabricated"] else subjects[it.unit]
            for it in items
        ]
    )

    def lr() -> Any:
        return LogisticRegression(C=4.0, max_iter=3000)

    out: dict[str, Any] = {
        "made_up_vs_real": {
            "grouping": "subject",
            "n_groups": len(set(groups)),
            "text_char_ngram": auroc_interval(
                grouped_oof(lr, texts, y, groups, text=True), y, b, rng
            ),
        }
    }
    keep = set(obscure_real_units(entities, first)["unit"])
    real = [it for it in items if it.unit in keep]
    look_text = {it.unit: str(it.state["question"]) for it in lookalike_items_}
    lt = [str(it.state["question"]) for it in real] + [
        look_text[r["unit"]] for r in lookalike
    ]
    ly = np.r_[np.zeros(len(real), bool), np.ones(len(lookalike), bool)]
    lg = np.array(
        [subjects[it.unit] for it in real] + [str(r["subject"]) for r in lookalike]
    )
    out["lookalike_vs_obscure_real"] = {
        "grouping": "subject",
        "text_char_ngram": auroc_interval(
            grouped_oof(lr, lt, ly, lg, text=True), ly, b, rng
        ),
    }
    scored = [it for it in news if it.unit in first]
    nm = [mask_dates(str(it.state["question"])) for it in scored]
    npost = np.array([str(it.info["month"]) >= cutoff_month for it in scored])
    months = np.array([str(it.info["month"]) for it in scored])
    out["news_dates_masked"] = {
        "grouping": "month (no month in both train and test)",
        "text_char_ngram": auroc_interval(
            grouped_oof(lr, nm, npost, months, text=True), npost, b, rng, months
        ),
    }
    return out


# --- 11. date routing ------------------------------------------------------------------


def date_routing(
    cutoff: Sequence[Mapping[str, Any]],
    news: Sequence[Item],
    b: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """The "not known" option against routing on the question's own date.

    The date router answers the questions whose latest mentioned date is
    earliest first (questions without a year first), using the plain
    answers, and skips the rest; coverage is matched exactly to the
    option's at every post-cutoff share (as in the cutoff-calibration
    analysis). Intervals at the reported shares come from the same
    period-stratified month-block bootstrap.

    Args:
        cutoff: Cutoff rows of the follow-up questions experiment.
        news: Yes/no news items (question text).
        b: Bootstrap resamples at each reported share.
        rng: Random generator.

    Returns:
        A curve over shares and intervals for option minus date router.
    """
    e = pd.DataFrame(list(cutoff))
    x = paired_rows(e)
    keys = (
        e[e["cond"] == "base"]["unit"]
        .str.replace(r"^cutoff:[a-z_]+:", "", regex=True)
        .to_numpy()
    )
    q = {it.unit: str(it.state["question"]) for it in news}
    dates = np.array([latest_date(q[k]) for k in keys])
    x["date_score"] = -np.where(np.isnan(dates), -np.inf, dates)  # earlier first

    def at(share: float, idx: np.ndarray | None = None) -> dict[str, float]:
        i = np.arange(len(x["post"])) if idx is None else idx
        base = at_share(x, share, idx)
        post = x["post"][i]
        w = np.where(
            post, share / max(post.sum(), 1), (1 - share) / max((~post).sum(), 1)
        )
        acc, cov = weighted_accuracy_at(
            x["date_score"][i], x["base_ok"][i], w, base["coverage"]
        )
        return {**base, "date_router": acc, "achieved_coverage_date_router": cov}

    curve = {f"{s:.2f}": at(s) for s in SHARES}
    report = {}
    for s in REPORT_SHARES:
        pt = at(s)
        d = []
        for _ in range(b):
            r = at(s, month_block_indices(x["month"], x["post"], rng, BLOCK_MONTHS))
            d.append(r["option"] - r["date_router"])
        report[f"{s:.3f}"] = {
            "coverage": pt["coverage"],
            "option": pt["option"],
            "date_router": pt["date_router"],
            "p_max": pt["p_max"],
            "option_minus_date_router": pt["option"] - pt["date_router"],
            "option_minus_date_router_ci": [
                float(np.percentile(d, 2.5)),
                float(np.percentile(d, 97.5)),
            ],
        }
    return {
        "questions_without_a_year": int(np.isnan(dates).sum()),
        "curve": curve,
        "at_selected_shares": report,
        "shares_where_option_beats_date_router": [
            float(k) for k in curve if curve[k]["option"] > curve[k]["date_router"]
        ],
    }


# --- 12. decision value of enough ------------------------------------------------------


def enough_decisions(
    evidence: Sequence[Mapping[str, Any]], rng: np.random.Generator, splits: int = 100
) -> dict[str, Any]:
    """Held-out answer acceptance by ``enough`` versus confidence on HotpotQA.

    All four evidence conditions are pooled. For each split, half of the
    questions set the threshold that gives the target coverage; accuracy and
    coverage are measured on the other half.

    Args:
        evidence: Evidence-sufficiency rows.
        rng: Random generator (one half-split per split).
        splits: Random half-splits by question.

    Returns:
        Mean held-out accuracy and coverage per score and target coverage,
        and ``enough`` minus confidence.
    """
    df = pd.DataFrame([r for r in evidence if r["set"] == "hotpot"])
    scores = {
        "confidence": df["p_max"].to_numpy(),
        "enough": df["p_enough"].to_numpy(),
        "enough_times_confidence": (df["p_enough"] * df["p_max"]).to_numpy(),
    }
    ok = df["correct"].to_numpy(bool)
    qids = df["qid"].unique()
    res: dict[str, dict[str, list[tuple[float, float]]]] = {
        k: {f"{c:.2f}": [] for c in COVERAGES} for k in scores
    }
    for _ in range(splits):
        half = set(rng.choice(qids, len(qids) // 2, replace=False))
        fit = df["qid"].isin(half).to_numpy()
        for name, s in scores.items():
            for c in COVERAGES:
                thr = np.quantile(s[fit], 1 - c)
                acc_mask = (~fit) & (s >= thr)
                res[name][f"{c:.2f}"].append(
                    (float(ok[acc_mask].mean()), float(acc_mask.sum() / (~fit).sum()))
                )
    out: dict[str, Any] = {
        "n": len(df),
        "all_answered_accuracy": float(ok.mean()),
        "splits": splits,
    }
    for name, per in res.items():
        out[name] = {
            c: {
                "accuracy": float(np.mean([a for a, _ in v])),
                "accuracy_range": [
                    float(np.min([a for a, _ in v])),
                    float(np.max([a for a, _ in v])),
                ],
                "coverage": float(np.mean([cv for _, cv in v])),
            }
            for c, v in per.items()
        }
    diffs_by: dict[str, Any] = {}
    for c in COVERAGES:
        k = f"{c:.2f}"
        diffs = [
            a - a2
            for (a, _), (a2, _) in zip(
                res["enough"][k], res["confidence"][k], strict=True
            )
        ]
        diffs_by[k] = {
            "mean": float(np.mean(diffs)),
            "range_over_splits": [float(np.min(diffs)), float(np.max(diffs))],
        }
    out["enough_minus_confidence"] = diffs_by
    return out


# --- 13. coverage table ------------------------------------------------------------------


def coverage_table(cutoff: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Coverage and error rates of the cutoff conditions.

    Args:
        cutoff: Cutoff rows of the follow-up questions experiment (``cond``,
            ``post``, ``top``, ``correct``, ``p_max``).

    Returns:
        Per condition and period: coverage, accuracy among answered,
        confident errors (wrong with top probability >= 0.9) among all and
        among answered.
    """
    e = pd.DataFrame(list(cutoff))
    out: dict[str, Any] = {}
    for (cond, post), g in e.groupby(["cond", "post"]):
        answered = g["top"] != "unknown"
        conf_wrong = answered & ~g["correct"].astype(bool) & (g["p_max"] >= 0.9)
        out[f"{cond}|{'post' if post else 'pre'}"] = {
            "n": len(g),
            "coverage": float(answered.mean()),
            "accuracy_answered": float(g[answered]["correct"].mean()),
            "confident_errors_of_all": float(conf_wrong.mean()),
            "confident_errors_of_answered": float(conf_wrong[answered].mean()),
        }
    return out


# --- 15-16. the date-masked run on answer-safe subsets -------------------------------------


def masked_label_safe(
    first: Mapping[str, Mapping[str, Any]],
    masked: Sequence[Mapping[str, Any]],
    labels: Mapping[str, str] | None,
    b: int,
    rng: np.random.Generator,
    cutoff_month: str = CUTOFF_MONTH,
) -> dict[str, Any]:
    """The date-masked run restricted to answer-safe questions.

    Two subsets: events that happened (label "yes"; "it happened by the
    deadline" implies "it happened", so the label survives removing a
    deadline unless the date identified the event), and, when hand labels
    are given, the questions whose edit preserved the answer (label ``P``).

    Args:
        first: Knowledge-boundary rows by unit (the originals).
        masked: Masked-news rows of the shortcut controls.
        labels: News unit -> hand label, or ``None``.
        b: Bootstrap resamples.
        rng: Random generator.
        cutoff_month: First month counted as after the cutoff.

    Returns:
        Per subset and period: share answered "no" and accuracy, original
        versus masked; label counts; and the after-versus-before ``known``
        AUROC on all items (month clusters).
    """
    m = pd.DataFrame(list(masked))
    m["post"] = m["month"] >= cutoff_month
    m["orig_top"] = [first[u]["top"] for u in m["base"]]
    m["orig_correct"] = [first[u]["correct"] for u in m["base"]]

    def summarise(d: pd.DataFrame) -> dict[str, Any]:
        res: dict[str, Any] = {}
        for period, g in d.groupby(np.where(d["post"], "post", "pre")):
            res[str(period)] = {
                "n": len(g),
                "said_no_original": float((g["orig_top"] == "no").mean()),
                "said_no_masked": float((g["top"] == "no").mean()),
                "accuracy_original": float(g["orig_correct"].mean()),
                "accuracy_masked": float(g["correct"].mean()),
            }
        return res

    out: dict[str, Any] = {"events_that_happened": summarise(m[m["gold"] == "yes"])}
    if labels is not None:
        m["label"] = m["base"].map(labels)
        s = m[m["label"].notna()]
        out["hand_labelled_sample"] = {
            "n": len(s),
            "labels": {str(k): int(v) for k, v in s["label"].value_counts().items()},
            "labels_by_gold": {
                "|".join(map(str, cast(tuple[Any, Any], key))): int(n)
                for key, n in s.groupby(["gold", "label"]).size().items()
            },
            "answer_preserving": summarise(s[s["label"] == "P"]),
        }
    post = m["post"].to_numpy()
    out["known_post_vs_pre_all_items"] = {
        "note": "does not use answer labels, so it is valid whether or not an edit preserved the answer",
        "original": auroc_interval(
            -np.array([first[u]["p_known"] for u in m["base"]]),
            post,
            b,
            rng,
            m["month"].to_numpy(),
        ),
        "masked": auroc_interval(
            -m["p_known"].to_numpy(), post, b, rng, m["month"].to_numpy()
        ),
    }
    return out


def date_only_subset(
    masked: Sequence[Mapping[str, Any]],
    news: Sequence[Item],
    b: int,
    rng: np.random.Generator,
    cutoff_month: str = CUTOFF_MONTH,
) -> dict[str, Any]:
    """The date-only baseline on the same questions as the masked comparison.

    Args:
        masked: Masked-news rows of the shortcut controls.
        news: Yes/no news items (question text).
        b: Bootstrap resamples.
        rng: Random generator.
        cutoff_month: First month counted as after the cutoff.

    Returns:
        AUROC (later period positive, later latest mentioned date = higher)
        with a month-cluster interval.
    """
    m = pd.DataFrame(list(masked))
    q = {it.unit: str(it.state["question"]) for it in news}
    dates = np.array([latest_date(q[u]) for u in m["base"]])
    filled = np.where(np.isnan(dates), np.nanmedian(dates), dates)
    post = (m["month"] >= cutoff_month).to_numpy()
    return {
        "date_only": auroc_interval(filled, post, b, rng, m["month"].to_numpy()),
        "questions_without_a_year": int(np.isnan(dates).sum()),
    }


# --- 17. matched confidence baselines ------------------------------------------------------


def matched_confidence_baselines(
    evidence: Sequence[Mapping[str, Any]],
    hotpot: Sequence[Item],
    synthetic: Sequence[Mapping[str, Any]],
    b: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """The follow-up questions against answer confidence on the same contrast.

    Completeness (HotpotQA, the three two-paragraph conditions, complete =
    positive): ``enough``, confidence, answer uncertainty (1 - confidence)
    and "shorter = complete", with paired differences clustered by
    question. Settledness (synthetic worlds, a fixed unknown fact ``D0``
    against a future chance event ``SFp``, clustered by scenario; also the
    uniform-chance scenarios, where the ideal distributions are identical):
    ``settled`` against confidence and answer uncertainty.

    Args:
        evidence: Evidence-sufficiency rows.
        hotpot: HotpotQA items.
        synthetic: Synthetic-worlds rows.
        b: Bootstrap resamples.
        rng: Random generator.

    Returns:
        AUROCs with intervals and paired differences.
    """
    rows = hotpot_frame(evidence, hotpot, ("dose0", "dose1", "dose2"))
    y = (rows["cell"] == "dose2").to_numpy()
    q = rows["qid"].to_numpy()
    enough, conf = rows["p_enough"].to_numpy(), rows["p_max"].to_numpy()
    shorter = -rows["words"].to_numpy(float)
    out: dict[str, Any] = {
        "completeness": {
            "n": len(rows),
            "n_complete": int(y.sum()),
            "orientation": "positive = complete evidence (both supporting paragraphs)",
            "enough": auroc_interval(enough, y, b, rng, q),
            "confidence_higher_is_complete": auroc_interval(conf, y, b, rng, q),
            "answer_uncertainty": auroc_interval(-conf, y, b, rng, q),
            "shorter_is_complete": auroc_interval(shorter, y, b, rng, q),
            "enough_minus_confidence": _paired(enough, conf, y, b, rng, q),
            "enough_minus_length": _paired(enough, shorter, y, b, rng, q),
        }
    }
    syn = pd.DataFrame(list(synthetic))
    syn = syn[~syn["alone"].astype(bool)]
    sets = {
        "D0_vs_SFp": syn[syn["cell"].isin(["D0", "SFp"])],
        "D0_vs_SFp_uniform_chance": syn[
            syn["cell"].isin(["D0", "SFp"]) & (syn["profile"] == "uniform")
        ],
    }
    for name, d in sets.items():
        ys = (d["cell"] == "D0").to_numpy()
        sc = d["scenario"].to_numpy()
        settled, pm = d["p_settled"].to_numpy(), d["p_max"].to_numpy()
        out[f"settledness_{name}"] = {
            "n": len(d),
            "orientation": "positive = fixed but unknown fact (D0); future chance event (SFp) negative",
            "settled": auroc_interval(settled, ys, b, rng, sc),
            "confidence": auroc_interval(pm, ys, b, rng, sc),
            "answer_uncertainty": auroc_interval(-pm, ys, b, rng, sc),
            "settled_minus_answer_uncertainty": _paired(settled, -pm, ys, b, rng, sc),
            "mean_confidence": {
                "D0": float(d[d["cell"] == "D0"]["p_max"].mean()),
                "SFp": float(d[d["cell"] == "SFp"]["p_max"].mean()),
            },
        }
    return out


# --- runner ---------------------------------------------------------------------------------


def analyse(inputs: Inputs, config: Config) -> dict[str, Any]:
    """Run every step (two generators; see the module docstring for the order).

    Args:
        inputs: Loaded rows and items.
        config: Options.

    Returns:
        Step name -> results.
    """
    b = config.bootstrap
    rows = inputs.rows
    first = _by_unit(rows["knowledge_boundary"])
    controls = split_rows(rows["shortcut_controls"])
    cutoff = rows["follow_up_questions"]
    tables = {exp: rows[exp] for exp in PROPER_SETS}
    news = news_frame(inputs.news, first, config.cutoff_month)
    rng = np.random.default_rng(config.seed)
    steps: dict[str, Callable[[], dict[str, Any]]] = {
        "entity_text_baselines": lambda: entity_baselines(
            inputs.entities, first, b, rng, config.fold_seed
        ),
        "news_text_baselines": lambda: news_baselines(news, b, rng, config.fold_seed),
        "second_look_baseline": lambda: second_look_baseline(
            rows["second_look"], first, b, rng
        ),
        "proper_scores": lambda: proper_scores(
            tables, rows["synthetic_worlds"], rows["stated_odds"]
        ),
        "length_checks": lambda: length_checks(
            rows["evidence_sufficiency"],
            inputs.hotpot,
            inputs.contrastive,
            rows["contrastive_facts"],
            b,
            rng,
        ),
        "recalibration_groupings": lambda: recalibration_groupings(
            news, rng, config.recalibration_splits, config.null_months
        ),
        "forward_chaining": lambda: forward_chaining(
            news, config.min_train_months, config.cutoff_month
        ),
        "label_balance": lambda: label_balance(news, config.cutoff_month),
    }
    out: dict[str, Any] = {name: step() for name, step in steps.items()}
    rng = np.random.default_rng(config.seed)
    second: dict[str, Callable[[], dict[str, Any]]] = {
        "enough_length": lambda: enough_length(
            rows["evidence_sufficiency"],
            inputs.hotpot,
            inputs.contrastive,
            rows["contrastive_facts"],
            b,
            rng,
        ),
        "paired_known_vs_confidence": lambda: paired_known_vs_confidence(
            inputs.entities,
            first,
            controls["lookalike"],
            controls["masked"],
            b,
            rng,
            config.cutoff_month,
        ),
        "grouped_text_baselines": lambda: grouped_text_baselines(
            inputs.entities,
            inputs.subjects,
            first,
            inputs.lookalike,
            controls["lookalike"],
            inputs.news,
            b,
            rng,
            config.cutoff_month,
        ),
        "date_routing": lambda: date_routing(cutoff, inputs.news, b, rng),
        "enough_decisions": lambda: enough_decisions(
            rows["evidence_sufficiency"], rng, config.decision_splits
        ),
        "coverage_table": lambda: coverage_table(cutoff),
        "floor_sensitivity": lambda: floor_sensitivity(tables),
        "masked_label_safe": lambda: masked_label_safe(
            first,
            controls["masked"],
            inputs.date_edit_labels,
            b,
            rng,
            config.cutoff_month,
        ),
        "date_only_subset": lambda: date_only_subset(
            controls["masked"], inputs.news, b, rng, config.cutoff_month
        ),
        "matched_confidence_baselines": lambda: matched_confidence_baselines(
            rows["evidence_sufficiency"],
            inputs.hotpot,
            rows["synthetic_worlds"],
            b,
            rng,
        ),
    }
    out |= {name: step() for name, step in second.items()}
    return out


def read_labels(path: str) -> dict[str, str] | None:
    """Read optional hand labels for the masked news questions.

    Args:
        path: JSON file (``[{"unit": ..., "label": ...}]``), or ``""``.

    Returns:
        Unit -> label, or ``None`` when no file is given.

    Raises:
        FileNotFoundError: If a path is given but does not exist.
    """
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    return {str(r["unit"]): str(r["label"]) for r in json.loads(p.read_text("utf-8"))}


def load_inputs(settings: Settings, config: Config) -> Inputs:
    """Read the source rows and rebuild the items the analysis needs.

    Args:
        settings: Run settings.
        config: Options.

    Returns:
        The inputs.
    """
    rows = {name: read_jsonl(source_rows(settings, name)) for name in SOURCES}
    rows["follow_up_questions"] = [
        r for r in rows["follow_up_questions"] if str(r["unit"]).startswith("cutoff:")
    ]
    d = settings.data_dir
    popqa = loaders.load_popqa(d)
    return Inputs(
        rows=rows,
        entities=entity_items(popqa, config.fabricated_per_relation, config.item_seed),
        subjects={
            f"popqa:{i}": str(s)
            for i, s in zip(popqa["id"], popqa["subj"], strict=True)
        },
        news=yes_no_news_items(
            loaders.load_daily_oracle("tf", d),
            config.yes_no_per_month,
            config.item_seed,
        ),
        hotpot=hotpot_items(loaders.load_hotpot_validation(d), config.item_seed),
        contrastive=contrastive_items(config.cases_per_policy, config.item_seed)[1],
        lookalike=lookalike_items(popqa, config.lookalike_per_relation),
        date_edit_labels=read_labels(config.date_edit_labels),
    )


def run(settings: Settings, config: Config) -> dict[str, Any]:
    """Load the inputs, run every step and write the summary.

    Args:
        settings: Run settings.
        config: Options.

    Returns:
        The summary (``config`` and ``results``).
    """
    summary = {
        "config": as_dict(config),
        "results": analyse(load_inputs(settings, config), config),
    }
    write_json(settings.out(NAME) / "summary.json", summary)
    return summary


ANALYSIS = Analysis(
    name=NAME,
    summary="Text, date and length baselines for the follow-up readouts",
    config_type=Config,
    run=run,
)
