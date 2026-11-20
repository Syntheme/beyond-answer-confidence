"""Cue controls with dependence-aware intervals and stronger text-only models.

Two contrasts carry the claim that the ``known`` follow-up adds little over
answer confidence once surface cues are controlled: realistic made-up names
against obscure real people, and news questions with their dates removed.
This analysis revisits both.

- **Dependence.** Each made-up name joins the first name of one obscure
  person to the surname of another, so names share people with each other
  and with the real items; linking every shared person joins most items into
  one component, so the paired AUROC difference ``known`` minus ``1 - p_max``
  is bootstrapped under each one-way clustering instead: by first-name
  donor, by surname donor (a real item is its own subject in both) and by
  relation, with a leave-one-relation-out check as there are only six.
  News is clustered by month.
- **Text-only controls.** Besides character n-grams, word n-grams, sentence
  embeddings (all-MiniLM-L6-v2), name-structure features (token counts and
  lengths, initials, particles, how common the first name and surname are
  among PopQA people) and embeddings plus name features are fitted out of
  fold, with label-stratified folds grouped by subject or, more strictly,
  by relation (names) and by month (news). A text model
  near 0.5 shows the contrast is not separable by that model; it cannot show
  that no cue exists.
"""

import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from beyond_answer_confidence.analyses.shortcut_baselines import (
    Config as BaselineConfig,
)
from beyond_answer_confidence.analyses.shortcut_baselines import load_inputs
from beyond_answer_confidence.config import as_dict
from beyond_answer_confidence.experiments.base import Analysis, write_json
from beyond_answer_confidence.experiments.shortcut_controls import (
    obscure_real_units,
    split_rows,
)
from beyond_answer_confidence.settings import Settings
from beyond_answer_confidence.stats.crossfit import FOLDS, _char_ngrams
from beyond_answer_confidence.stats.resampling import (
    auroc_interval,
    paired_auroc_difference,
)
from beyond_answer_confidence.tasks.follow_up import CUTOFF_MONTH
from beyond_answer_confidence.tasks.shortcuts import (
    _SEED_NAMESPACE,
    lookalike_subjects,
    mask_dates,
)

NAME = "cue_controls"

PARTICLES = frozenset(
    {"de", "da", "di", "del", "della", "van", "von", "der", "den", "le", "la",
     "bin", "ibn", "al", "el", "y", "du", "dos", "das", "ter", "ten", "mac"}
)  # fmt: skip
"""Lower-case name particles counted as a name-structure feature."""


@dataclass(frozen=True)
class Config:
    """Options.

    Attributes:
        bootstrap: Bootstrap resamples.
        seed: Seed of the bootstrap generator.
        item_seed: Seed of the source item sets.
        yes_no_per_month: News questions per month in the source set.
        fabricated_per_relation: Fabricated entities per relation.
        lookalike_per_relation: Look-alike names per person relation.
        cutoff_month: First month counted as after the knowledge cutoff.
    """

    bootstrap: int = 1000
    seed: int = 0
    item_seed: int = 0
    yes_no_per_month: int = 80
    fabricated_per_relation: int = 100
    lookalike_per_relation: int = 100
    cutoff_month: str = CUTOFF_MONTH


def donor_map(popqa: Any, per_relation: int) -> dict[str, tuple[str, str]]:
    """Look-alike unit -> (first-name donor, surname donor).

    Rebuilds the names exactly as the shortcut controls drew them.

    Args:
        popqa: PopQA table.
        per_relation: Names per person relation.

    Returns:
        The donors of every look-alike unit.
    """
    import random

    rng = random.Random(f"{_SEED_NAMESPACE}:lookalike")  # noqa: S311  # nosec B311
    out = {}
    for prop, picks in lookalike_subjects(popqa, per_relation, rng).items():
        for j, (_name, first, last) in enumerate(picks):
            out[f"lookalike:{prop}:{j}"] = (first, last)
    return out


def one_way_clusters(
    real_subjects: Sequence[str], donors: Sequence[tuple[str, str]]
) -> dict[str, np.ndarray]:
    """Cluster ids per item under each one-way clustering of the people.

    Args:
        real_subjects: Subject of each real item (in order).
        donors: Donor pair of each made-up item (in order, after the real).

    Returns:
        ``first_name_donor`` and ``surname_donor`` cluster ids (a real item
        is its own subject in both), plus the size of the largest component
        when every shared person is linked (``linked_component``).
    """
    first = np.array(list(real_subjects) + [a for a, _ in donors])
    last = np.array(list(real_subjects) + [b for _, b in donors])
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in donors:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    linked = np.array([find(x) for x in first])
    return {"first_name_donor": first, "surname_donor": last, "linked": linked}


def _tokens(name: str) -> list[str]:
    return [t for t in re.split(r"\s+", name.strip()) if t]


def name_features(names: Sequence[str], reference: Sequence[str]) -> np.ndarray:
    """Name-structure features of each name.

    Args:
        names: Person names.
        reference: Names whose first and last tokens set the frequency
            features (PopQA person subjects).

    Returns:
        A ``len(names) x 10`` matrix.
    """
    firsts = Counter(_tokens(n)[0].lower() for n in reference if _tokens(n))
    lasts = Counter(_tokens(n)[-1].lower() for n in reference if _tokens(n))
    rows = []
    for n in names:
        t = _tokens(n) or [""]
        rows.append(
            [
                len(t),
                len(n),
                float(np.mean([len(x) for x in t])),
                sum(bool(re.fullmatch(r"[A-Z]\.?", x)) for x in t),
                sum(x.lower() in PARTICLES for x in t),
                int("-" in n),
                int(any(ord(c) > 127 for c in n)),
                np.log1p(firsts[t[0].lower()]),
                np.log1p(lasts[t[-1].lower()]),
                sum(x[:1].isupper() for x in t) / len(t),
            ]
        )
    return np.asarray(rows, float)


def _oof(
    fit_predict: Callable[[np.ndarray, np.ndarray], np.ndarray],
    n: int,
    y: np.ndarray,
    groups: np.ndarray,
) -> np.ndarray:
    """Out-of-fold scores with :data:`FOLDS` group-disjoint, label-stratified folds.

    Stratification keeps each fold's class balance close to the whole set's;
    without it, out-of-fold AUROC is biased below 0.5 when the text carries
    no signal (each test fold's balance mirrors its training fold's).
    """
    from sklearn.model_selection import StratifiedGroupKFold

    out = np.empty(n)
    labels = np.asarray(y, int)
    folds = StratifiedGroupKFold(FOLDS, shuffle=True, random_state=0)
    for train, test in folds.split(np.zeros(n), labels, groups):
        out[test] = fit_predict(train, test)
    return out


def text_models(
    texts: Sequence[str], features: np.ndarray | None, embeddings: np.ndarray
) -> dict[str, Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]]:
    """Fold-wise fit-and-predict functions of every text-only model.

    Args:
        texts: Question texts.
        features: Optional name-structure features (rows of ``texts``).
        embeddings: Sentence embeddings (rows of ``texts``).

    Returns:
        Model name -> function ``(train, test, y) -> test scores``.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def lr() -> Any:
        return LogisticRegression(C=4.0, max_iter=3000)

    def tfidf(analyzer: str, ngrams: tuple[int, int]) -> Any:
        def run(train: np.ndarray, test: np.ndarray, y: np.ndarray) -> np.ndarray:
            vec = TfidfVectorizer(
                analyzer=analyzer, ngram_range=ngrams, sublinear_tf=True
            )
            xtr = vec.fit_transform([texts[i] for i in train])
            model = lr().fit(xtr, y[train])
            return np.asarray(
                model.predict_proba(vec.transform([texts[i] for i in test]))[:, 1]
            )

        return run

    def dense(x: np.ndarray) -> Any:
        def run(train: np.ndarray, test: np.ndarray, y: np.ndarray) -> np.ndarray:
            model = make_pipeline(StandardScaler(), lr()).fit(x[train], y[train])
            return np.asarray(model.predict_proba(x[test])[:, 1])

        return run

    def char(train: np.ndarray, test: np.ndarray, y: np.ndarray) -> np.ndarray:
        vec = _char_ngrams()
        xtr = vec.fit_transform([texts[i] for i in train])
        model = lr().fit(xtr, y[train])
        return np.asarray(
            model.predict_proba(vec.transform([texts[i] for i in test]))[:, 1]
        )

    models = {
        "char_ngram": char,
        "word_ngram": tfidf("word", (1, 2)),
        "sentence_embedding": dense(embeddings),
    }
    if features is not None:
        models["name_structure"] = dense(features)
        models["embedding_and_name_structure"] = dense(np.c_[embeddings, features])
    return models


def fit_text_controls(
    texts: Sequence[str],
    y: np.ndarray,
    groups: np.ndarray,
    embeddings: np.ndarray,
    features: np.ndarray | None,
    b: int,
    rng: np.random.Generator,
    clusters: np.ndarray | None = None,
    permutations: int = 0,
) -> dict[str, Any]:
    """Grouped out-of-fold AUROC of every text-only model.

    Args:
        texts: Question texts.
        y: Labels (positive = made up, or after the cutoff).
        groups: Fold groups (no group in both train and test).
        embeddings: Sentence embeddings.
        features: Optional name-structure features.
        b: Bootstrap resamples.
        rng: Random generator.
        clusters: Bootstrap clusters of the intervals (default: ``groups``).
        permutations: Label shuffles for a permutation reference (0 = none):
            the mean shuffled AUROC and the share of shuffles at least as
            high as the observed AUROC (a one-sided test that the text
            identifies the positive class).

    Returns:
        Model -> AUROC with a cluster-bootstrap interval (and the
        permutation reference).
    """
    from beyond_answer_confidence.metrics.ranking import auroc

    labels = np.asarray(y, int)
    boot = groups if clusters is None else clusters
    out: dict[str, Any] = {}

    def scores(fit: Callable[..., np.ndarray], z: np.ndarray) -> np.ndarray:
        def fold(train: np.ndarray, test: np.ndarray) -> np.ndarray:
            return fit(train, test, z)

        return _oof(fold, len(texts), z, groups)

    for name, fit in text_models(texts, features, embeddings).items():
        res = auroc_interval(scores(fit, labels), np.asarray(y, bool), b, rng, boot)
        if permutations:
            null = []
            for _ in range(permutations):
                ys = rng.permutation(labels)
                null.append(auroc(scores(fit, ys), ys.astype(bool)))
            res["permutation"] = {
                "shuffles": permutations,
                "mean": float(np.mean(null)),
                "p_at_least_observed": float(np.mean(np.array(null) >= res["auroc"])),
            }
        out[name] = res
    return out


def lookalike_block(
    known: np.ndarray,
    conf: np.ndarray,
    y: np.ndarray,
    clusterings: Mapping[str, np.ndarray],
    relations: np.ndarray,
    b: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """The paired ``known`` - confidence difference under each resampling unit.

    Args:
        known: P(known) per item.
        conf: p_max per item.
        y: True for made-up items.
        clusterings: ``first_name_donor``, ``surname_donor`` and ``linked``
            cluster ids per item (see :func:`one_way_clusters`).
        relations: Relation per item.
        b: Bootstrap resamples.
        rng: Random generator.

    Returns:
        Differences with intervals by item, first-name donor, surname donor
        and relation, the leave-one-relation-out range and the cluster counts.
    """
    a, c = -known, -conf
    first = clusterings["first_name_donor"]
    loro = {}
    for rel in sorted(set(relations)):
        keep = relations != rel
        loro[str(rel)] = paired_auroc_difference(
            a[keep], c[keep], y[keep], b, rng, first[keep]
        )["diff"]
    sizes = {k: Counter(v) for k, v in clusterings.items()}
    return {
        "orientation": "higher score = made up (1 - P(known), 1 - p_max)",
        "n_items": len(y),
        "n_made_up": int(y.sum()),
        "n_clusters": {
            "first_name_donor": len(sizes["first_name_donor"]),
            "surname_donor": len(sizes["surname_donor"]),
            "relation": len(set(relations)),
        },
        "largest_linked_component_items": int(max(sizes["linked"].values())),
        "by_item": paired_auroc_difference(a, c, y, b, rng),
        "by_first_name_donor": paired_auroc_difference(a, c, y, b, rng, first),
        "by_surname_donor": paired_auroc_difference(
            a, c, y, b, rng, clusterings["surname_donor"]
        ),
        "by_relation": paired_auroc_difference(a, c, y, b, rng, relations),
        "leave_one_relation_out": {
            "diff": loro,
            "min": float(min(loro.values())),
            "max": float(max(loro.values())),
        },
    }


def analyse(
    inputs: Any,
    popqa: Any,
    embed: Callable[[Sequence[str]], np.ndarray],
    config: Config,
) -> dict[str, Any]:
    """Run both contrasts.

    Args:
        inputs: Shortcut-baseline inputs (rows and rebuilt items).
        popqa: PopQA table.
        embed: Texts -> sentence embeddings.
        config: Options.

    Returns:
        ``lookalike_vs_obscure_real`` and ``news_dates_removed`` blocks.
    """
    b = config.bootstrap
    rng = np.random.default_rng(config.seed)
    first = {str(r["unit"]): r for r in inputs.rows["knowledge_boundary"]}
    controls = split_rows(inputs.rows["shortcut_controls"])
    look = list(controls["lookalike"])
    donors = donor_map(popqa, config.lookalike_per_relation)
    real = obscure_real_units(inputs.entities, first)
    real_subjects = [inputs.subjects[u] for u in real["unit"]]
    pairs = [donors[str(r["unit"])] for r in look]
    clusterings = one_way_clusters(real_subjects, pairs)
    relations = np.array(list(real["prop"]) + [str(r["prop"]) for r in look])
    known = np.r_[
        [first[u]["p_known"] for u in real["unit"]], [r["p_known"] for r in look]
    ]
    conf = np.r_[[first[u]["p_max"] for u in real["unit"]], [r["p_max"] for r in look]]
    y = np.r_[np.zeros(len(real), bool), np.ones(len(look), bool)]
    look_text = {it.unit: str(it.state["question"]) for it in inputs.lookalike}
    texts = list(real["question"]) + [look_text[str(r["unit"])] for r in look]
    names = real_subjects + [str(r["subject"]) for r in look]
    persons = [
        str(s)
        for s, p in zip(popqa["subj"], popqa["prop"], strict=True)
        if p in set(real["prop"])
    ]
    vectors, feats = embed(texts), name_features(names, persons)
    subjects = np.array(names)
    out: dict[str, Any] = {
        "lookalike_vs_obscure_real": {
            "known_minus_confidence": lookalike_block(
                known, conf, y, clusterings, relations, b, rng
            ),
            "text_only_folds_by_subject": fit_text_controls(
                texts,
                y,
                subjects,
                vectors,
                feats,
                b,
                rng,
                clusterings["first_name_donor"],
            ),
            "text_only_folds_by_relation": fit_text_controls(
                texts,
                y,
                relations,
                vectors,
                feats,
                b,
                rng,
                clusterings["first_name_donor"],
            ),
        }
    }
    news = [it for it in inputs.news if it.unit in first]
    masked = [mask_dates(str(it.state["question"])) for it in news]
    months = np.array([str(it.info["month"]) for it in news])
    post = months >= config.cutoff_month
    out["news_dates_removed"] = {
        "n_items": len(news),
        "n_months": len(set(months)),
        "n_months_pre": len(set(months[~post])),
        "n_months_post": len(set(months[post])),
        "text_only_grouped_by_month": fit_text_controls(
            masked, post, months, embed(masked), None, b, rng
        ),
    }
    return out


def run(settings: Settings, config: Config) -> dict[str, Any]:
    """Load the inputs, run both contrasts and write the summary.

    Args:
        settings: Run settings.
        config: Options.

    Returns:
        The summary (``config`` and ``results``).
    """
    from beyond_answer_confidence.data import loaders
    from beyond_answer_confidence.data.distractors import embed

    base = BaselineConfig(
        item_seed=config.item_seed,
        yes_no_per_month=config.yes_no_per_month,
        fabricated_per_relation=config.fabricated_per_relation,
        lookalike_per_relation=config.lookalike_per_relation,
        cutoff_month=config.cutoff_month,
    )
    popqa = loaders.load_popqa(settings.data_dir)
    summary = {
        "config": as_dict(config),
        "results": analyse(
            load_inputs(settings, base),
            popqa,
            lambda texts: embed(texts, settings.data_dir),
            config,
        ),
    }
    write_json(settings.out(NAME) / "summary.json", summary)
    return summary


ANALYSIS = Analysis(
    name=NAME,
    summary="Cue controls with people- and month-clustered intervals and stronger "
    "text-only models",
    config_type=Config,
    run=run,
)
