"""Shortcut controls: what do the follow-up readouts respond to?

Seven small runs, each holding one surface cue fixed (builders in
:mod:`beyond_answer_confidence.tasks.shortcuts`):

- ``masked``: yes/no news with every date removed. Does ``known`` still
  separate events after the knowledge cutoff, and does the "no" default
  remain?
- ``lookalike``: made-up people with realistic recombined names, against
  obscure real people and pseudo-word names.
- ``evidence``: made-up subjects with a sentence stating an answer, or an
  unrelated sentence. Does ``known`` rise with evidence?
- ``polarity``: the second look with the opposite answer proposed. A belief
  that an event did not happen accepts "no" and rejects "yes"; a reflex
  towards the word "no" rejects both.
- ``semantics``: random devices asked four ways (which outcome will occur,
  which is most likely, the probability of each, a bet), to see which
  target the choice distribution follows.
- ``negated``: the second look worded negatively ("is the proposed answer
  wrong?") with both answers proposed; a preference for the word "no" and
  a belief that the event did not happen predict different answers.
- ``near_day``: the unchanged news question with today's date just after
  the event, so the outcome is settled in every version; compared with no
  date and a fixed later date on the same items.

Needs the rows of ``knowledge_boundary`` (first answers; also the requests
of ``polarity``) and, for the analysis, ``second_look`` and
``follow_up_questions``.
"""

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from beyond_answer_confidence.backends.cache import CachedClient
from beyond_answer_confidence.config import as_dict
from beyond_answer_confidence.data import loaders
from beyond_answer_confidence.experiments.base import (
    Experiment,
    backend_for,
    read_jsonl,
    source_rows,
    write_json,
    write_jsonl,
)
from beyond_answer_confidence.metrics.ranking import auroc
from beyond_answer_confidence.runner import run_collect, run_summary
from beyond_answer_confidence.settings import Settings
from beyond_answer_confidence.stats.crossfit import oof_text_scores
from beyond_answer_confidence.stats.resampling import auroc_interval, bootstrap
from beyond_answer_confidence.tasks.chance import DEVICES
from beyond_answer_confidence.tasks.entities import (
    PERSON_RELATIONS,
    fabricated_items,
    popqa_items,
)
from beyond_answer_confidence.tasks.follow_up import CUTOFF_MONTH
from beyond_answer_confidence.tasks.news import yes_no_news_items
from beyond_answer_confidence.tasks.schema import Item, Task
from beyond_answer_confidence.tasks.scoring import score_items
from beyond_answer_confidence.tasks.shortcuts import (
    STUDY_DATE,
    evidence_items,
    lookalike_items,
    masked_items,
    near_day_items,
    negated_items,
    news_subset,
    polarity_items,
    semantics_items,
)

logger = logging.getLogger(__name__)

NAME = "shortcut_controls"

PARTS = (
    "masked",
    "lookalike",
    "evidence",
    "polarity",
    "semantics",
    "negated",
    "near_day",
)
"""Parts in run and analysis order."""

ORDERED_PARTS = frozenset({"semantics"})
"""Parts asked once per option order (the order is the replicate index)."""


@dataclass(frozen=True)
class Config:
    """Options of the shortcut controls.

    Attributes:
        replicates: Replicates per item (parts other than ``semantics``).
        seed: Seed of the source item sets (news sample, entities).
        yes_no_per_month: News questions per month in the source set.
        fabricated_per_relation: Fabricated entities per relation in the
            source set.
        cutoff_month: First month counted as after the knowledge cutoff.
        lookalike_per_relation: Look-alike names per person relation.
        evidence_per_relation: Fabricated items per relation in ``evidence``.
        semantic_orders: Option orders per device and wording.
        study_date: Latest ``today`` in ``near_day``.
        bootstrap: Bootstrap resamples.
        bootstrap_seed: Seed of the two bootstrap generators (one for the
            first five parts, a fresh one for ``negated`` and ``near_day``).
        fold_seed: Seed of the text-baseline folds.
    """

    replicates: int = 3
    seed: int = 0
    yes_no_per_month: int = 80
    fabricated_per_relation: int = 100
    cutoff_month: str = CUTOFF_MONTH
    lookalike_per_relation: int = 100
    evidence_per_relation: int = 25
    semantic_orders: int = 24
    study_date: str = STUDY_DATE
    bootstrap: int = 1000
    bootstrap_seed: int = 0
    fold_seed: int = 0


def rows_by_unit(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Index rows by unit.

    Args:
        rows: Rows.

    Returns:
        Unit -> row.
    """
    return {str(r["unit"]): dict(r) for r in rows}


def entity_items(
    popqa: pd.DataFrame, fabricated_per_relation: int = 100, seed: int = 0
) -> list[Item]:
    """PopQA items followed by the fabricated twins (as the source set).

    Args:
        popqa: PopQA table.
        fabricated_per_relation: Fabricated entities per relation.
        seed: Item seed.

    Returns:
        Items.
    """
    return popqa_items(popqa, seed) + fabricated_items(
        popqa, fabricated_per_relation, seed
    )


def build_parts(
    popqa: pd.DataFrame,
    news: Sequence[Item],
    first: Mapping[str, Mapping[str, Any]],
    config: Config,
) -> dict[str, list[Item]]:
    """Every part's items from loaded inputs.

    Args:
        popqa: PopQA table.
        news: Yes/no news items of the source set.
        first: Knowledge-boundary rows by unit (first answers).
        config: Options.

    Returns:
        Part -> items, in :data:`PARTS` order.
    """
    subset = news_subset(news, config.cutoff_month, config.seed)
    fabricated = fabricated_items(popqa, config.fabricated_per_relation, config.seed)
    return {
        "masked": masked_items(subset),
        "lookalike": lookalike_items(popqa, config.lookalike_per_relation),
        "evidence": evidence_items(fabricated, config.evidence_per_relation),
        "polarity": polarity_items(subset, first, config.cutoff_month),
        "semantics": semantics_items(config.semantic_orders),
        "negated": negated_items(subset, config.cutoff_month),
        "near_day": near_day_items(subset, config.study_date),
    }


def build_items(settings: Settings, config: Config) -> dict[str, list[Item]]:
    """Load the datasets and first answers, and build every part.

    Args:
        settings: Run settings.
        config: Options.

    Returns:
        Part -> items.
    """
    d = settings.data_dir
    news = yes_no_news_items(
        loaders.load_daily_oracle("tf", d), config.yes_no_per_month, config.seed
    )
    first = rows_by_unit(read_jsonl(source_rows(settings, "knowledge_boundary")))
    return build_parts(loaders.load_popqa(d), news, first, config)


def part_tasks(parts: Mapping[str, Sequence[Item]], replicates: int) -> list[Task]:
    """Tasks of every part (replicate-major; ordered parts once per order).

    Args:
        parts: Part -> items.
        replicates: Replicates of the replicated parts.

    Returns:
        Tasks.
    """
    tasks: list[Task] = []
    for part, items in parts.items():
        if part in ORDERED_PARTS:
            tasks.extend(
                Task(it.unit, it.request(int(it.unit.rsplit(":", 1)[-1])))
                for it in items
            )
        else:
            tasks.extend(
                Task(it.unit, it.request(r)) for r in range(replicates) for it in items
            )
    return tasks


def requests(settings: Settings, config: Config) -> list[Task]:
    """Every task of the experiment (needs the knowledge-boundary rows).

    Args:
        settings: Run settings.
        config: Options.

    Returns:
        The tasks.
    """
    return part_tasks(build_items(settings, config), config.replicates)


# --- analysis ------------------------------------------------------------------


def _boot_means(
    values: np.ndarray, b: int, rng: np.random.Generator, shift: float = 0.0
) -> np.ndarray:
    """Bootstrap means of ``values`` minus ``shift`` (``b`` separate item draws)."""
    return bootstrap(len(values), lambda i: values[i].mean() - shift, b, rng)


def _ci95(boot: np.ndarray) -> list[float]:
    # Percentiles at exactly 2.5 and 97.5: stats.resampling.percentile_ci
    # derives its levels from the coverage (2.5000000000000022), which moves
    # some bounds by one ulp.
    return [float(np.percentile(boot, q)) for q in (2.5, 97.5)]


def analyse_masked(
    rows: Sequence[Mapping[str, Any]],
    first: Mapping[str, Mapping[str, Any]],
    b: int,
    rng: np.random.Generator,
    cutoff_month: str = CUTOFF_MONTH,
) -> dict[str, Any]:
    """Masked versus original news on the same items.

    Bootstrap draws: for the original, then the masked version, the
    ``known`` and then the confidence AUROC (month clusters).

    Args:
        rows: Masked rows.
        first: Knowledge-boundary rows by unit (the originals).
        b: Bootstrap resamples.
        rng: Random generator.
        cutoff_month: First month counted as after the cutoff.

    Returns:
        Per version and period: accuracy, "no" share, confidence, gap,
        mean ``known`` and its AUROC for correctness; and the after-versus-
        before AUROCs of ``known`` and confidence.
    """
    df = pd.DataFrame(rows)
    df["post"] = df["month"] >= cutoff_month
    orig = pd.DataFrame([first[u] for u in df["base"]])
    y = df["post"].to_numpy(bool)
    months = df["month"].to_numpy()
    out: dict[str, Any] = {"n": len(df), "n_post": int(y.sum())}
    for name, d in (("original", orig), ("masked", df)):
        per: dict[str, Any] = {}
        for period, m in (("pre", ~y), ("post", y)):
            g = d[m]
            per[period] = {
                "accuracy": float(g["correct"].mean()),
                "predicted_no": float((g["top"] == "no").mean()),
                "mean_p_max": float(g["p_max"].mean()),
                "gap": float(g["p_max"].mean() - g["correct"].mean()),
                "p_known": float(g["p_known"].mean()),
                "known_vs_correct": auroc(
                    g["p_known"].to_numpy(), g["correct"].to_numpy(bool)
                ),
            }
        out[name] = {
            **per,
            "known_post_vs_pre": auroc_interval(
                -d["p_known"].to_numpy(), y, b, rng, months
            ),
            "confidence_post_vs_pre": auroc_interval(
                -d["p_max"].to_numpy(), y, b, rng, months
            ),
        }
    return out


def obscure_real_units(
    entities: Sequence[Item], first: Mapping[str, Any], share: float = 0.2
) -> pd.DataFrame:
    """Real people in the least popular ``share`` of each person relation.

    Args:
        entities: PopQA items (fabricated twins are skipped).
        first: Units with a knowledge-boundary row.
        share: Popularity quantile kept per relation.

    Returns:
        ``unit``, ``prop``, ``s_pop`` and ``question`` per kept item.
    """
    real = [
        it
        for it in entities
        if it.info["prop"] in PERSON_RELATIONS
        and it.unit in first
        and not it.info["fabricated"]
    ]
    df = pd.DataFrame(
        {
            "unit": [it.unit for it in real],
            "prop": [it.info["prop"] for it in real],
            "s_pop": [it.info["s_pop"] for it in real],
            "question": [str(it.state["question"]) for it in real],
        }
    )
    keep = df.groupby("prop")["s_pop"].transform(lambda s: s <= s.quantile(share))
    return df[keep].reset_index(drop=True)


def analyse_lookalike(
    rows: Sequence[Mapping[str, Any]],
    items: Sequence[Item],
    entities: Sequence[Item],
    first: Mapping[str, Mapping[str, Any]],
    b: int,
    rng: np.random.Generator,
    fold_seed: int = 0,
) -> dict[str, Any]:
    """Look-alike and pseudo-word made-up names against obscure real people.

    Bootstrap draws: for pseudo-word names, then look-alike names, the
    ``known``, confidence and text-only AUROCs.

    Args:
        rows: Look-alike rows.
        items: Look-alike items (for the question text).
        entities: PopQA items followed by the fabricated twins.
        first: Knowledge-boundary rows by unit.
        b: Bootstrap resamples.
        rng: Random generator.
        fold_seed: Seed of the text-baseline folds.

    Returns:
        Group means, and per made-up group the AUROCs (made up = positive)
        of ``1 - known``, ``1 - confidence`` and a character n-gram model
        of the question alone.
    """
    real = obscure_real_units(entities, first)
    pseudo = [
        it
        for it in entities
        if it.info["prop"] in PERSON_RELATIONS
        and it.unit in first
        and it.info["fabricated"]
    ]
    text_of = {it.unit: str(it.state["question"]) for it in items}
    look = pd.DataFrame(rows)
    groups = {
        "real_least_popular": (
            real["question"].tolist(),
            np.array([first[u]["p_known"] for u in real["unit"]]),
            np.array([first[u]["p_max"] for u in real["unit"]]),
        ),
        "pseudo_word": (
            [str(it.state["question"]) for it in pseudo],
            np.array([first[it.unit]["p_known"] for it in pseudo]),
            np.array([first[it.unit]["p_max"] for it in pseudo]),
        ),
        "lookalike": (
            [text_of[u] for u in look["unit"]],
            look["p_known"].to_numpy(),
            look["p_max"].to_numpy(),
        ),
    }
    out: dict[str, Any] = {
        k: {"n": len(v[1]), "p_known": float(v[1].mean()), "p_max": float(v[2].mean())}
        for k, v in groups.items()
    }
    r_text, r_known, r_pmax = groups["real_least_popular"]
    for made_up in ("pseudo_word", "lookalike"):
        m_text, m_known, m_pmax = groups[made_up]
        y = np.r_[np.zeros(len(r_known), bool), np.ones(len(m_known), bool)]
        out[f"{made_up}_vs_real"] = {
            "known": auroc_interval(-np.r_[r_known, m_known], y, b, rng),
            "confidence": auroc_interval(-np.r_[r_pmax, m_pmax], y, b, rng),
            "text_char_ngram": auroc_interval(
                oof_text_scores([*r_text, *m_text], y, fold_seed), y, b, rng
            ),
        }
    return out


def analyse_evidence(
    rows: Sequence[Mapping[str, Any]], first: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    """``known`` and answers with a stating versus an unrelated sentence.

    Args:
        rows: Evidence rows.
        first: Knowledge-boundary rows by unit (the no-sentence baseline).

    Returns:
        Means per arm, the paired difference, the share of items where the
        stated arm is higher, and how often the answer follows the sentence.
    """
    df = pd.DataFrame(rows)
    wide = df.pivot_table(index="base", columns="arm", values="p_known")
    base = np.array([first[u]["p_known"] for u in wide.index])
    st = df[df["arm"] == "stated"]
    return {
        "n_items": len(wide),
        "p_known": {
            "no_sentence": float(base.mean()),
            "unrelated": float(wide["unrelated"].mean()),
            "stated": float(wide["stated"].mean()),
        },
        "stated_minus_unrelated": float((wide["stated"] - wide["unrelated"]).mean()),
        "share_stated_higher": float((wide["stated"] > wide["unrelated"]).mean()),
        "known_auroc_stated_vs_unrelated": auroc(
            df["p_known"].to_numpy(), (df["arm"] == "stated").to_numpy()
        ),
        "follows_stated_answer": float(st["correct"].mean()),
        "p_on_stated_answer": float(st["p_gold"].mean()),
        "p_max": {str(a): float(g["p_max"].mean()) for a, g in df.groupby("arm")},
    }


def analyse_polarity(
    rows: Sequence[Mapping[str, Any]],
    own: Mapping[str, Mapping[str, Any]],
    b: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Coherence of the two second-look answers ("yes" and "no" proposed).

    Args:
        rows: Polarity rows (opposite answer proposed).
        own: Second-look news rows by news unit (own answer proposed).
        b: Bootstrap resamples per interval bound (item draws for the mean
            sum; the lower bound uses the first ``b``, the upper the next).
        rng: Random generator.

    Returns:
        Per period: mean P(correct) for each proposed answer, the shares
        rejecting or accepting both, and the implied answer and its
        accuracy; plus a 95 % interval for mean(sum) - 1 (each bound from
        its own ``b`` resamples).
    """
    df = pd.DataFrame(rows)
    df["p_other"] = df["p_correct"]
    df["p_own"] = [own[u]["p_correct"] for u in df["base"]]
    df["p_yes_proposed"] = np.where(df["own"] == "yes", df["p_own"], df["p_other"])
    df["p_no_proposed"] = np.where(df["own"] == "no", df["p_own"], df["p_other"])
    df["sum"] = df["p_yes_proposed"] + df["p_no_proposed"]
    df["gold_yes"] = df["gold_answer"] == "yes"
    out: dict[str, Any] = {"n": len(df)}
    for period, g in df.groupby(np.where(df["post"], "post", "pre")):
        implied_yes = g["p_yes_proposed"] > g["p_no_proposed"]
        out[str(period)] = {
            "n": len(g),
            "mean_p_correct_when_yes_proposed": float(g["p_yes_proposed"].mean()),
            "mean_p_correct_when_no_proposed": float(g["p_no_proposed"].mean()),
            "mean_sum": float(g["sum"].mean()),
            "share_rejecting_both": float(
                ((g["p_yes_proposed"] < 0.5) & (g["p_no_proposed"] < 0.5)).mean()
            ),
            "share_accepting_both": float(
                ((g["p_yes_proposed"] > 0.5) & (g["p_no_proposed"] > 0.5)).mean()
            ),
            "implied_answer_yes_share": float(implied_yes.mean()),
            "implied_answer_accuracy": float((implied_yes == g["gold_yes"]).mean()),
            "true_yes_share": float(g["gold_yes"].mean()),
        }
    d = df["sum"].to_numpy()
    # Each bound comes from its own ``b`` resamples (lower first).
    out["sum_minus_one_ci"] = [
        float(np.percentile(_boot_means(d, b, rng, shift=1.0), q)) for q in (2.5, 97.5)
    ]
    return out


def analyse_semantics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Mean distribution, distance to the stated odds and mass on the mode.

    Args:
        rows: Semantics rows.

    Returns:
        Device -> wording -> summary.
    """
    out: dict[str, Any] = {}
    df = pd.DataFrame(rows)
    for (dev, wording), g in df.groupby(["device", "wording"]):
        ideal = DEVICES[str(dev)][1]
        labels = list(ideal)
        mean = {k: float(np.mean([r[k] for r in g["dist"]])) for k in labels}
        mode = max(ideal, key=ideal.__getitem__)
        out.setdefault(str(dev), {})[str(wording)] = {
            "n": len(g),
            "tv_to_stated": 0.5 * sum(abs(mean[k] - ideal[k]) for k in labels),
            "p_on_mode": mean[mode],
            "stated_p_mode": ideal[mode],
            "top_label": max(mean, key=mean.__getitem__),
            "mean": mean,
        }
    return out


def analyse_negated(
    rows: Sequence[Mapping[str, Any]],
    own: Mapping[str, Mapping[str, Any]],
    opposite: Mapping[str, Mapping[str, Any]],
    b: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Positive versus negative verification wording, both answers per item.

    Args:
        rows: Negated rows.
        own: Second-look news rows by news unit (own answer, positive).
        opposite: Polarity rows by news unit (opposite answer, positive).
        b: Bootstrap resamples (item draws, after-cutoff items).
        rng: Random generator.

    Returns:
        Per period: mean P(correct) and P(wrong) for each proposed answer,
        the coherence P(correct) + P(wrong), the shares accepting or
        rejecting both, and the answer implied by each wording; plus the
        after-cutoff mean of P(wrong | yes) - P(wrong | no) with a 95 %
        interval.
    """
    neg = pd.DataFrame(rows).pivot_table(
        index="base", columns="proposed", values="p_wrong"
    )
    out: dict[str, Any] = {"n_items": len(neg)}
    recs = []
    for base in neg.index:
        o, p = own[base], opposite[base]
        pos = {o["proposed"]: o["p_correct"], p["proposed"]: p["p_correct"]}
        recs.append(
            {
                "base": base,
                "post": bool(p["post"]),
                "gold_yes": p["gold_answer"] == "yes",
                "pos_yes": pos["yes"],
                "pos_no": pos["no"],
                "neg_yes": float(neg.loc[base, "yes"]),
                "neg_no": float(neg.loc[base, "no"]),
            }
        )
    df = pd.DataFrame(recs)
    for period, g in df.groupby(np.where(df["post"], "post", "pre")):
        pos_implies_yes = g["pos_yes"] > g["pos_no"]
        neg_implies_yes = g["neg_no"] > g["neg_yes"]  # "no is wrong" => it happened
        out[str(period)] = {
            "n": len(g),
            "positive_p_correct_yes_proposed": float(g["pos_yes"].mean()),
            "positive_p_correct_no_proposed": float(g["pos_no"].mean()),
            "negative_p_wrong_yes_proposed": float(g["neg_yes"].mean()),
            "negative_p_wrong_no_proposed": float(g["neg_no"].mean()),
            "coherence_yes_proposed": float((g["pos_yes"] + g["neg_yes"]).mean()),
            "coherence_no_proposed": float((g["pos_no"] + g["neg_no"]).mean()),
            "positive_rejects_both": float(
                ((g["pos_yes"] < 0.5) & (g["pos_no"] < 0.5)).mean()
            ),
            "negative_accepts_both": float(
                ((g["neg_yes"] < 0.5) & (g["neg_no"] < 0.5)).mean()
            ),
            "negative_rejects_both": float(
                ((g["neg_yes"] > 0.5) & (g["neg_no"] > 0.5)).mean()
            ),
            "implied_yes_positive": float(pos_implies_yes.mean()),
            "implied_yes_negative": float(neg_implies_yes.mean()),
            "implied_answers_agree": float((pos_implies_yes == neg_implies_yes).mean()),
            "implied_accuracy_negative": float(
                (neg_implies_yes == g["gold_yes"]).mean()
            ),
            "true_yes": float(g["gold_yes"].mean()),
        }
    post = df[df["post"]]
    diff = (post["neg_yes"] - post["neg_no"]).to_numpy()
    out["post_wrong_yes_minus_no"] = {
        "mean": float(diff.mean()),
        "ci": _ci95(_boot_means(diff, b, rng)),
    }
    return out


def analyse_near_day(
    rows: Sequence[Mapping[str, Any]],
    cutoff_rows: Sequence[Mapping[str, Any]],
    b: int,
    rng: np.random.Generator,
    cutoff_month: str = CUTOFF_MONTH,
) -> dict[str, Any]:
    """No date versus a fixed later date versus today just after the event.

    Args:
        rows: Near-day rows.
        cutoff_rows: Follow-up cutoff rows (``cond`` ``base`` and ``date``;
            units ``cutoff:<cond>:<news unit>``).
        b: Bootstrap resamples.
        rng: Random generator (one AUROC interval per version, in order).
        cutoff_month: First month counted as after the cutoff.

    Returns:
        Per version and period: accuracy, "no" share, gap, mean ``known``
        and its AUROC for correctness; and the after-versus-before AUROC of
        ``1 - known`` (month clusters).
    """
    e = pd.DataFrame(cutoff_rows)
    e["base_unit"] = e["unit"].str.replace(r"^cutoff:[a-z_]+:", "", regex=True)
    versions = {
        "no_date": e[e["cond"] == "base"].set_index("base_unit"),
        "fixed_today": e[e["cond"] == "date"].set_index("base_unit"),
        "today_just_after": pd.DataFrame(rows).set_index("base"),
    }
    units = versions["today_just_after"].index
    out: dict[str, Any] = {}
    for name, frame in versions.items():
        d = frame.loc[units]
        post = (d["month"] >= cutoff_month).to_numpy()
        per: dict[str, Any] = {}
        for period, m in (("pre", ~post), ("post", post)):
            g = d[m]
            per[period] = {
                "accuracy": float(g["correct"].mean()),
                "predicted_no": float((g["top"] == "no").mean()),
                "gap": float(g["p_max"].mean() - g["correct"].mean()),
                "p_known": float(g["p_known"].mean()),
                "known_vs_correct": auroc(
                    g["p_known"].to_numpy(), g["correct"].to_numpy(bool)
                ),
            }
        out[name] = {
            **per,
            "known_post_vs_pre": auroc_interval(
                -d["p_known"].to_numpy(), post, b, rng, d["month"].to_numpy()
            ),
        }
    return out


@dataclass(frozen=True)
class Sources:
    """Rows and items of other experiments used by the analysis.

    Attributes:
        first: Knowledge-boundary rows by unit.
        second_look: Second-look news rows by news unit.
        cutoff_rows: Follow-up rows of the cutoff probe.
        entities: PopQA items followed by the fabricated twins.
    """

    first: Mapping[str, Mapping[str, Any]]
    second_look: Mapping[str, Mapping[str, Any]]
    cutoff_rows: Sequence[Mapping[str, Any]]
    entities: Sequence[Item]


def analyse(
    rows: Mapping[str, Sequence[Mapping[str, Any]]],
    parts: Mapping[str, Sequence[Item]],
    sources: Sources,
    config: Config,
) -> dict[str, Any]:
    """Analyse every part.

    Two generators seeded with ``bootstrap_seed``: the first is used by
    ``masked``, ``lookalike`` and ``polarity`` in that order; the second by
    ``negated`` and then ``near_day``.

    Args:
        rows: Part -> scored rows.
        parts: Part -> items.
        sources: Rows and items of other experiments.
        config: Options.

    Returns:
        Part -> results.
    """
    b = config.bootstrap
    rng = np.random.default_rng(config.bootstrap_seed)
    res: dict[str, Any] = {
        "masked": analyse_masked(
            rows["masked"], sources.first, b, rng, config.cutoff_month
        ),
        "lookalike": analyse_lookalike(
            rows["lookalike"],
            parts["lookalike"],
            sources.entities,
            sources.first,
            b,
            rng,
            config.fold_seed,
        ),
        "evidence": analyse_evidence(rows["evidence"], sources.first),
        "polarity": analyse_polarity(rows["polarity"], sources.second_look, b, rng),
        "semantics": analyse_semantics(rows["semantics"]),
    }
    rng = np.random.default_rng(config.bootstrap_seed)
    res["negated"] = analyse_negated(
        rows["negated"],
        sources.second_look,
        rows_by_unit([{**r, "unit": r["base"]} for r in rows["polarity"]]),
        b,
        rng,
    )
    res["near_day"] = analyse_near_day(
        rows["near_day"], sources.cutoff_rows, b, rng, config.cutoff_month
    )
    return res


def load_sources(settings: Settings, config: Config) -> Sources:
    """Read the other experiments' rows and rebuild the entity items.

    Args:
        settings: Run settings.
        config: Options.

    Returns:
        The sources.
    """
    second = {
        str(r["unit"]): r
        for r in read_jsonl(source_rows(settings, "second_look"))
        if r["set"] == "oracle_tf"
    }
    cutoff = [
        r
        for r in read_jsonl(source_rows(settings, "follow_up_questions"))
        if str(r["unit"]).startswith("cutoff:")
    ]
    return Sources(
        first=rows_by_unit(read_jsonl(source_rows(settings, "knowledge_boundary"))),
        second_look=second,
        cutoff_rows=cutoff,
        entities=entity_items(
            loaders.load_popqa(settings.data_dir),
            config.fabricated_per_relation,
            config.seed,
        ),
    )


def split_rows(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, list[Mapping[str, Any]]]:
    """Group scored rows by part.

    Args:
        rows: Rows with a ``part`` field.

    Returns:
        Part -> rows (every part in :data:`PARTS`, possibly empty).
    """
    out: dict[str, list[Mapping[str, Any]]] = {p: [] for p in PARTS}
    for r in rows:
        out[str(r["part"])].append(r)
    return out


def run(settings: Settings, config: Config) -> dict[str, Any]:
    """Collect (cache first), score, analyse and write the outputs.

    Args:
        settings: Run settings.
        config: Options.

    Returns:
        The summary (``results`` only when every item is complete).
    """
    parts = build_items(settings, config)
    client = CachedClient(
        settings.cache_file(NAME),
        model=settings.model,
        max_input_tokens=settings.max_input_tokens,
    )
    collected = run_collect(
        part_tasks(parts, config.replicates),
        client,
        backend=backend_for(settings),
        concurrency=settings.concurrency,
        label=NAME,
    )
    expected = {
        it.unit: 1 if part in ORDERED_PARTS else config.replicates
        for part, items in parts.items()
        for it in items
    }
    all_items = [it for items in parts.values() for it in items]
    rows = score_items(all_items, collected.complete(expected))
    by_part = split_rows(rows)
    out = settings.out(NAME)
    write_jsonl(out / "rows.jsonl", rows)
    summary: dict[str, Any] = {
        "config": as_dict(config),
        "items": len(expected),
        "complete_items": len(rows),
        "parts": {
            p: {"items": len(parts[p]), "complete_items": len(by_part[p])}
            for p in PARTS
        },
        **run_summary(client, collected),
    }
    if len(rows) == len(expected):
        summary["results"] = analyse(
            by_part, parts, load_sources(settings, config), config
        )
    else:
        logger.warning("%s: incomplete items; no analysis", NAME)
    write_json(out / "summary.json", summary)
    return summary


EXPERIMENT = Experiment(
    name=NAME,
    summary="Surface-cue controls: masked dates, look-alike names, evidence, polarity",
    config_type=Config,
    requests=requests,
    run=run,
)
