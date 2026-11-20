"""Does an explicit scoring rule turn Choice percentages into odds?

Dice, coins, cards and new spinner profiles with stated chances are asked as
Choice in two wordings, ``plain`` and ``log_score``, and the spinners also
as one yes/no question per outcome (see
:mod:`beyond_answer_confidence.tasks.odds_semantics`). The analysis reports,
per part and wording: the total-variation distance from the stated chances
of single calls and of the distribution averaged over option orders, the
mass on the first-listed option and its excess over that option's stated
chance, the mean absolute error and the mass on the most likely outcome;
the change from ``plain`` to ``log_score`` with a bootstrap over devices or
profiles; and for the yes/no form the mean absolute error and the sum of the
probabilities over a profile's outcomes (1 for coherent answers).
"""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from beyond_answer_confidence.backends.cache import CachedClient
from beyond_answer_confidence.config import as_dict
from beyond_answer_confidence.experiments.base import (
    Experiment,
    analysable,
    backend_for,
    write_json,
    write_jsonl,
)
from beyond_answer_confidence.metrics.distributions import tv
from beyond_answer_confidence.runner import run_collect, run_summary
from beyond_answer_confidence.settings import Settings
from beyond_answer_confidence.stats.resampling import percentile_ci
from beyond_answer_confidence.tasks.chance import device_tasks
from beyond_answer_confidence.tasks.odds_semantics import odds_items
from beyond_answer_confidence.tasks.schema import Item, Task
from beyond_answer_confidence.tasks.scoring import score_items

logger = logging.getLogger(__name__)

NAME = "odds_semantics"


@dataclass(frozen=True)
class Config:
    """Options.

    Attributes:
        device_orders: Option orders per device.
        spinners: Spinner profiles.
        spinner_orders: Option orders per spinner and wording.
        seed: Item seed.
        bootstrap: Bootstrap resamples.
        bootstrap_seed: Bootstrap seed.
    """

    device_orders: int = 24
    spinners: int = 40
    spinner_orders: int = 6
    seed: int = 0
    bootstrap: int = 2000
    bootstrap_seed: int = 0


def build(config: Config) -> tuple[list[Item], list[Task]]:
    """Items and their tasks (one per option order).

    Args:
        config: Options.

    Returns:
        ``(items, tasks)``.
    """
    items = odds_items(
        config.device_orders, config.spinners, config.spinner_orders, config.seed
    )
    return items, device_tasks([(it, 1) for it in items])


def requests(settings: Settings, config: Config) -> list[Task]:
    """Every task of the experiment.

    Args:
        settings: Run settings (unused; the items are generated).
        config: Options.

    Returns:
        The tasks.
    """
    del settings
    return build(config)[1]


def choice_metrics(g: pd.DataFrame) -> dict[str, Any]:
    """Distance and position metrics of one part and wording.

    Args:
        g: Choice rows (``dist``, ``ideal``, ``order``, ``key``).

    Returns:
        The metrics.
    """
    single = [tv(d, i) for d, i in zip(g["dist"], g["ideal"], strict=True)]
    order_mean = []
    for _key, h in g.groupby("key"):
        mean = {
            o: float(np.mean([d[o] for d in h["dist"]])) for o in h["ideal"].iloc[0]
        }
        order_mean.append(tv(mean, h["ideal"].iloc[0]))
    first = [d[o[0]] for d, o in zip(g["dist"], g["order"], strict=True)]
    first_ideal = [i[o[0]] for i, o in zip(g["ideal"], g["order"], strict=True)]
    mode = [
        d[max(i, key=i.__getitem__)] for d, i in zip(g["dist"], g["ideal"], strict=True)
    ]
    mode_ideal = [max(i.values()) for i in g["ideal"]]
    mae = [
        float(np.mean([abs(d[o] - i[o]) for o in i]))
        for d, i in zip(g["dist"], g["ideal"], strict=True)
    ]
    return {
        "n_calls": len(g),
        "single_call_tv": float(np.mean(single)),
        "order_mean_tv": float(np.mean(order_mean)),
        "first_listed_mass": float(np.mean(first)),
        "first_listed_excess": float(np.mean(np.array(first) - np.array(first_ideal))),
        "mean_abs_error": float(np.mean(mae)),
        "mode_mass": float(np.mean(mode)),
        "mode_stated": float(np.mean(mode_ideal)),
    }


def analyse(rows: list[dict[str, Any]], config: Config) -> dict[str, Any]:
    """Per-part metrics, the wording effect and the yes/no form.

    Args:
        rows: Scored rows.
        config: Options.

    Returns:
        The results.
    """
    rng = np.random.default_rng(config.bootstrap_seed)
    df = pd.DataFrame(rows)
    out: dict[str, Any] = {}
    choice = df[df["set"].isin(["device", "spinner"])].copy()
    choice["single_tv"] = [
        tv(d, i) for d, i in zip(choice["dist"], choice["ideal"], strict=True)
    ]
    for part, g in choice.groupby("set"):
        block: dict[str, Any] = {
            str(arm): choice_metrics(h) for arm, h in g.groupby("arm")
        }
        per_key = g.pivot_table(index="key", columns="arm", values="single_tv")
        diff = (per_key["log_score"] - per_key["plain"]).to_numpy()
        boot = [
            float(diff[rng.integers(0, len(diff), len(diff))].mean())
            for _ in range(config.bootstrap)
        ]
        block["log_score_minus_plain_single_call_tv"] = {
            "mean": float(diff.mean()),
            "ci95": percentile_ci(np.array(boot)),
            "n_keys": len(diff),
        }
        out[str(part)] = block
    yn = df[df["set"] == "spinner_yes_no"]
    maes, sums, norm = [], [], []
    for _, r in yn.iterrows():
        ideal = r["ideal"]
        p = {c: float(r[f"p_lands_{c}"]) for c in ideal}
        maes.append(float(np.mean([abs(p[c] - ideal[c]) for c in ideal])))
        total = sum(p.values())
        sums.append(total)
        norm.append(float(np.mean([abs(p[c] / total - ideal[c]) for c in ideal])))
    out["spinner_yes_no"] = {
        "n_profiles": len(yn),
        "mean_abs_error": float(np.mean(maes)),
        "mean_sum_of_p_yes": float(np.mean(sums)),
        "mean_abs_error_normalised": float(np.mean(norm)),
    }
    return out


def run(settings: Settings, config: Config) -> dict[str, Any]:
    """Collect (cache first), score, analyse and write the outputs.

    Args:
        settings: Run settings.
        config: Options.

    Returns:
        The summary.
    """
    items, tasks = build(config)
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
    rows = score_items(items, collected.complete({it.unit: 1 for it in items}))
    out = settings.out(NAME)
    write_jsonl(out / "rows.jsonl", rows)
    summary: dict[str, Any] = {
        "config": as_dict(config),
        "items": len(items),
        "complete_items": len(rows),
        **run_summary(client, collected),
    }
    if analysable(len(rows), len(items), summary.get("errors"), summary.get("fatal")):
        summary["results"] = analyse(rows, config)
    else:
        logger.warning("%s: incomplete items; no analysis", NAME)
    write_json(out / "summary.json", summary)
    return summary


EXPERIMENT = Experiment(
    name=NAME,
    summary="Stated odds under an explicit log-score rule, new spinner profiles "
    "and yes/no per outcome",
    config_type=Config,
    requests=requests,
    run=run,
)
