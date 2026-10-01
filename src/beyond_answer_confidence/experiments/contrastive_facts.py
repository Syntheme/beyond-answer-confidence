"""Contrastive facts: do answers follow the facts that decide them?

Made-up policy cases in five variants (see
:mod:`beyond_answer_confidence.tasks.contrastive`): both facts, a label-flipping
counterfactual, a deciding fact deleted, an irrelevant fact added, and a
contradiction added. The analysis reports the probability on the rule's
answer for decidable variants, the "cannot tell" probability for
undecidable ones, the shift caused by an irrelevant fact, and how well the
``enough`` question separates decidable from undecidable variants.
"""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from beyond_answer_confidence.config import as_dict
from beyond_answer_confidence.experiments.base import (
    Experiment,
    backend_for,
    write_json,
    write_jsonl,
)
from beyond_answer_confidence.metrics.distributions import tv
from beyond_answer_confidence.metrics.ranking import auroc
from beyond_answer_confidence.runner import run_items
from beyond_answer_confidence.settings import Settings
from beyond_answer_confidence.stats.inference import (
    auroc_vs_threshold,
    holm_adjust_tails,
    mean_interval,
    mean_interval_lower,
)
from beyond_answer_confidence.tasks.contrastive import CANNOT_TELL, contrastive_items
from beyond_answer_confidence.tasks.schema import Item, Task, item_tasks

logger = logging.getLogger(__name__)

NAME = "contrastive_facts"


@dataclass(frozen=True)
class Config:
    """Options of the contrastive-facts experiment.

    Attributes:
        replicates: Replicates per item.
        cases_per_policy: Generated cases per policy.
        seed: Case-generation seed.
        bootstrap: Bootstrap resamples.
        bootstrap_seed: Bootstrap seed.
        follows_reference: Reference mean probability on the rule's answer.
        cannot_tell_reference: Reference mean "cannot tell" probability.
        irrelevant_reference: Reference mean total variation caused by an
            irrelevant fact.
        auroc_reference: Reference AUROC of ``enough`` for undecidable cases.
    """

    replicates: int = 3
    cases_per_policy: int = 100
    seed: int = 0
    bootstrap: int = 2000
    bootstrap_seed: int = 0
    follows_reference: float = 0.90
    cannot_tell_reference: float = 0.50
    irrelevant_reference: float = 0.05
    auroc_reference: float = 0.80


def build_items(config: Config) -> list[Item]:
    """Every item.

    Args:
        config: Options.

    Returns:
        Items (policy-major, then case, then variant).
    """
    return contrastive_items(config.cases_per_policy, config.seed)[1]


def requests(settings: Settings, config: Config) -> list[Task]:
    """Every task of the experiment.

    Args:
        settings: Run settings (unused; the items are generated).
        config: Options.

    Returns:
        The tasks.
    """
    del settings
    return item_tasks(build_items(config), config.replicates)


def analyse(rows: list[dict[str, Any]], config: Config) -> dict[str, Any]:
    """Evidence-following, "cannot tell", irrelevant-fact shift, ``enough`` AUROC.

    Bootstrap draws (one generator, in this order): P(rule's answer) for
    ``base`` and ``counterfactual``; P(cannot tell) for ``deleted`` and
    ``contradictory``; irrelevant-fact total variation; ``enough`` AUROC
    (cases resampled as clusters).

    Args:
        rows: Scored rows.
        config: Options.

    Returns:
        Effect sizes with intervals (tail shares Holm-adjusted across the
        five), per-variant and per-policy summaries.
    """
    b = config.bootstrap
    rng = np.random.default_rng(config.bootstrap_seed)
    df = pd.DataFrame(rows)
    df["p_cannot"] = [d[CANNOT_TELL] for d in df["dist"]]

    def at_least(values: np.ndarray, threshold: float) -> dict[str, Any]:
        return mean_interval_lower(values, b, rng, threshold)

    follows = {
        v: at_least(
            df[df["variant"] == v]["p_gold"].to_numpy(float), config.follows_reference
        )
        for v in ("base", "counterfactual")
    }
    base = df[df["variant"] == "base"].set_index(["domain", "case"])
    irr = df[df["variant"] == "irrelevant"].set_index(["domain", "case"])
    tv_irr = np.array([tv(irr.loc[k, "dist"], base.loc[k, "dist"]) for k in base.index])
    undecidable = df["variant"].isin(["deleted", "contradictory"])
    res: dict[str, Any] = holm_adjust_tails(
        {
            "follows_evidence": {
                "variants": follows,
                "tail": max(v["tail"] for v in follows.values()),
            },
            "deleted_gives_cannot_tell": at_least(
                df[df["variant"] == "deleted"]["p_cannot"].to_numpy(float),
                config.cannot_tell_reference,
            ),
            "contradiction_gives_cannot_tell": at_least(
                df[df["variant"] == "contradictory"]["p_cannot"].to_numpy(float),
                config.cannot_tell_reference,
            ),
            "irrelevant_fact_shift": mean_interval(
                tv_irr, b, rng, threshold=config.irrelevant_reference
            ),
            "enough_flags_undecidable": auroc_vs_threshold(
                1 - df["p_enough"].to_numpy(float),
                undecidable.to_numpy(),
                config.auroc_reference,
                b,
                rng,
                clusters=(df["domain"] + ":" + df["case"].astype(str)).to_numpy(),
            ),
        }
    )
    by_variant: dict[str, Any] = {}
    for var, g in df.groupby("variant"):
        yes = g["p_outcome"].to_numpy(float)
        by_variant[str(var)] = {
            "n": len(g),
            "accuracy": float((g["top"] == g["gold"]).mean()),
            "mean_p_gold": float(g["p_gold"].mean()),
            "mean_p_cannot": float(g["p_cannot"].mean()),
            "mean_p_enough": float(g["p_enough"].mean()),
            "share_outcome_noul_extreme": float(np.mean((yes <= 0.1) | (yes >= 0.9))),
            "share_outcome_noul_le_0.1": float(np.mean(yes <= 0.1)),
        }
    res["by_variant"] = by_variant
    res["by_policy"] = {
        str(p): {str(v): float(x["p_gold"].mean()) for v, x in g.groupby("variant")}
        for p, g in df.groupby("domain")
    }
    dele = df[df["variant"] == "deleted"]
    res["deleted_share_top_not_cannot_tell"] = float(
        np.mean([t != CANNOT_TELL for t in dele["top"]])
    )
    pair = df[df["variant"].isin(["base", "deleted"])]
    res["enough_auroc_deleted_only"] = auroc(
        1 - pair["p_enough"].to_numpy(float), (pair["variant"] == "deleted").to_numpy()
    )
    return res


def run(settings: Settings, config: Config) -> dict[str, Any]:
    """Collect (cache first), score, analyse and write the outputs.

    Args:
        settings: Run settings.
        config: Options.

    Returns:
        The summary (``results`` only when every item is complete).
    """
    rows, book = run_items(
        build_items(config),
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
    summary: dict[str, Any] = {"config": as_dict(config), **book}
    if book["complete_items"] == book["items"]:
        summary["results"] = analyse(rows, config)
    else:
        logger.warning("%s: incomplete items; no analysis", NAME)
    write_json(out / "summary.json", summary)
    return summary


EXPERIMENT = Experiment(
    name=NAME,
    summary="Policy cases: counterfactual, deleted, irrelevant and contradictory facts",
    config_type=Config,
    requests=requests,
    run=run,
)
