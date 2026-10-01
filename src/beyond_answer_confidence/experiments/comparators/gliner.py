"""GLiNER2.5-Decide, a local zero-shot classifier, on the knowledge-dial items.

The classifier sees the same test items and the same opaque codes as the
knowledge dial (:mod:`beyond_answer_confidence.tasks.intents`), under four settings:
no information (``L0``), intent names as codes (``Lname``), one example per
code (``L1``) and the real intent names as labels (``names``, its native
mode). Rows use the knowledge-dial row format, so the same aggregation
applies; the comparison (:data:`ANALYSIS`) sets them beside the API's
answers on the same items.

Running the classifier needs ``gliner2``, which requires ``transformers<5``
and ``huggingface-hub<1`` and so conflicts with this package's main
environment. Use a separate virtual environment, without API credentials::

    python -m venv .venv-gliner
    .venv-gliner/bin/pip install gliner2==2.0.0 "transformers<5" "huggingface-hub<1"
    .venv-gliner/bin/pip install datasets pandas
    .venv-gliner/bin/pip install -e . --no-deps

and call :func:`classify` (the :data:`INFERENCE` entry point) from it. Only
the standard library, numpy and pandas are imported at module import time;
torch and gliner2 are imported when classification runs. The comparison
runs in the main environment.
"""

import logging
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from beyond_answer_confidence.analyses.knowledge_dial import (
    aggregate,
    condition_table,
    no_knowledge,
)
from beyond_answer_confidence.config import as_dict
from beyond_answer_confidence.data.intents import (
    IntentDataset,
    load,
    nested_subset,
    select_items,
)
from beyond_answer_confidence.experiments.base import Analysis, read_jsonl, write_json
from beyond_answer_confidence.json_output import dumps
from beyond_answer_confidence.metrics.calibration import reliability_bins, smece
from beyond_answer_confidence.settings import Settings
from beyond_answer_confidence.stats.resampling import bootstrap_means, percentile_ci
from beyond_answer_confidence.tasks.intents import (
    Condition,
    build_choice,
    sample_examples,
)

logger = logging.getLogger(__name__)

NAME = "gliner"
MODEL = "fastino/GLiNER2.5-Decide"
MODEL_REVISION = "7ee5da4c2415e32259bcdc0b1a7367c32ce8d6f6"  # pragma: allowlist secret
NAMES = "names"
"""Setting with the real intent names as labels (the classifier's native mode)."""
RESERVED = (
    "[P]",
    "[L]",
    "[C]",
    "[E]",
    "[R]",
    "[DESCRIPTION]",
    "[EXAMPLE]",
    "[OUTPUT]",
    "(",
    ")",
)
"""Strings GLiNER rejects in label descriptions (it injects them verbatim)."""


@dataclass(frozen=True)
class InferenceConfig:
    """Classification options (defaults: the Banking77 run).

    The CLINC150 run used ``dataset="clinc150"``, ``subset_per_intent=3`` and
    ``conditions=("L0", "names", "Lname")``.

    Attributes:
        dataset: Dataset name.
        conditions: Settings to run, in order.
        per_intent: Knowledge-dial test items per intent.
        subset_per_intent: Keep only the knowledge dial's nested k-per-intent
            subset (0 = all items).
        threads: Torch CPU threads.
        limit: Only the first ``limit`` items (0 = all; for smoke tests).
    """

    dataset: str = "banking77"
    conditions: tuple[str, ...] = ("L0", NAMES, "Lname", "L1")
    per_intent: int = 10
    subset_per_intent: int = 0
    threads: int = 14
    limit: int = 0


def sanitise(text: str) -> str:
    """Replace GLiNER's reserved strings with spaces and collapse whitespace.

    Args:
        text: A label description.

    Returns:
        The cleaned description.
    """
    for tok in RESERVED:
        text = text.replace(tok, " ")
    return " ".join(text.split())


def describe(value: Any) -> str | None:
    """Turn a knowledge-dial code description into a GLiNER label description.

    Args:
        value: ``None`` or a dict with ``name`` and/or ``examples``.

    Returns:
        The description, or ``None`` for an undescribed code.
    """
    if value is None:
        return None
    parts = []
    if "name" in value:
        parts.append(f"name: {value['name']}")
    if "examples" in value:
        parts.append("examples: " + " | ".join(value["examples"]))
    return sanitise("; ".join(parts))


def labels_for(
    ds: IntentDataset, item: int, condition: str, examples: Mapping[str, list[str]]
) -> tuple[list[str] | dict[str, str], dict[str, str]]:
    """GLiNER labels for one item and setting, with the knowledge dial's codes.

    Args:
        ds: The dataset.
        item: Test index.
        condition: A knowledge-dial condition or :data:`NAMES`.
        examples: The knowledge dial's seed-0 example sets.

    Returns:
        ``(labels, label_to_intent)``: a list (no descriptions) or a
        ``{label: description}`` mapping, and each label's intent.
    """
    if condition == NAMES:
        return list(ds.labels), {lab: lab for lab in ds.labels}
    spec = build_choice(
        ds.test[item], f"{ds.name}:test:{item}", ds, Condition(condition), examples, []
    )
    descriptions = {code: describe(v) for code, v in spec.criteria.items()}
    if all(d is None for d in descriptions.values()):
        return list(descriptions), dict(spec.code_to_label)
    return {c: d or c for c, d in descriptions.items()}, dict(spec.code_to_label)


def row_for(
    item: int,
    condition: str,
    gold: str,
    probs: Mapping[str, float],
    first_listed: str,
    n_tokens: int,
) -> dict[str, Any]:
    """Build a row in the knowledge-dial format.

    Args:
        item: Test index.
        condition: Setting.
        gold: Gold intent.
        probs: Probability per intent.
        first_listed: Intent of the first listed label.
        n_tokens: Packed input length.

    Returns:
        The row.
    """
    k = len(probs)
    nz = [p for p in probs.values() if p > 0]
    return {
        "arm": "gliner",
        "item": item,
        "condition": condition,
        "code_style": "names" if condition == NAMES else "ordinal",
        "example_seed": 0,
        "replicate": 0,
        "K": k,
        "gold": gold,
        "swap_target": None,
        "choice": max(probs, key=probs.__getitem__),
        "first_listed": first_listed,
        "p_gold": probs[gold],
        "p_max": max(probs.values()),
        "norm_entropy": -sum(p * math.log(p) for p in nz) / math.log(k),
        "probs": dict(probs),
        "model": f"{MODEL}@{MODEL_REVISION[:12]}",
        "n_tokens": n_tokens,
    }


def done_keys(path: Path) -> set[tuple[int, str]]:
    """``(item, condition)`` pairs already written (the run is resumable).

    Args:
        path: Output JSONL.

    Returns:
        The pairs.
    """
    if not path.exists():
        return set()
    return {(r["item"], r["condition"]) for r in read_jsonl(path)}


def choose_items(ds: IntentDataset, config: InferenceConfig) -> list[int]:
    """The knowledge dial's main-arm test items, optionally its nested subset.

    Args:
        ds: The dataset.
        config: Options.

    Returns:
        Test indices.
    """
    items = select_items(ds.test, config.per_intent)
    if config.subset_per_intent:
        items = nested_subset(ds.test, items, config.subset_per_intent)
    return items[: config.limit] if config.limit else items


def rows_file(settings: Settings, dataset: str) -> Path:
    """Where the classifier's rows for a dataset live.

    Args:
        settings: Run settings.
        dataset: Dataset name.

    Returns:
        ``<output_dir>/gliner/<dataset>.jsonl``.
    """
    return settings.out(NAME) / f"{dataset}.jsonl"


def classify(
    settings: Settings, config: InferenceConfig
) -> dict[str, Any]:  # pragma: no cover - needs gliner2 and a model download
    """Score every item under every setting and append rows.

    Args:
        settings: Run settings.
        config: Options.

    Returns:
        Rows written.
    """
    import torch
    from gliner2.classification.engine import Classifier
    from gliner2.classification.schema import ClassificationSchema

    torch.set_num_threads(config.threads)
    # Pinned revision (satisfies bandit B615). Whether this revision ships
    # safetensors weights has not been verified, so the loader's own default
    # format is kept (no use_safetensors=True).
    clf = Classifier.from_pretrained(
        MODEL, revision=MODEL_REVISION, cache_dir=str(settings.hf_dir)
    )
    ds = load(config.dataset, settings.data_dir)
    examples = sample_examples(ds, seed=0)
    out = rows_file(settings, config.dataset)
    done = done_keys(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    start, n = time.time(), 0
    with out.open("a", encoding="utf-8") as f:
        for condition in config.conditions:
            for item in choose_items(ds, config):
                if (item, condition) in done:
                    continue
                labels, to_intent = labels_for(ds, item, condition, examples)
                schema = ClassificationSchema().task(
                    "intent", labels, min_labels=1, max_labels=1
                )
                compiled = clf.compile_schema(schema)
                text = ds.test[item].text
                batch = clf.scorer.processor.collate_fn_inference(
                    [(text, compiled.build())], max_len=None
                )
                scores = clf.score(text, schema)
                raw = {
                    to_intent[lab]: scores.probability("intent", lab)
                    for lab in scores.tasks["intent"]
                }
                total = sum(raw.values())
                probs = {k: v / total for k, v in raw.items()}
                row = row_for(
                    item,
                    condition,
                    ds.test[item].label,
                    probs,
                    to_intent[next(iter(labels))],
                    int(batch["input_ids"].shape[-1]),
                )
                f.write(dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                n += 1
                if n % 50 == 0:
                    logger.info(
                        "%s: %d rows, %.1f s/row",
                        condition,
                        n,
                        (time.time() - start) / n,
                    )
    return {"config": as_dict(config), "rows_written": n}


INFERENCE = Analysis(
    name=NAME,
    summary="Run GLiNER2.5-Decide on the knowledge-dial items (separate environment)",
    config_type=InferenceConfig,
    run=classify,
)
"""Classification entry point (local compute, no API calls)."""


# --- Comparison with the API ------------------------------------------------------

FIGURE_PANELS = ("L0", "Lname", "L1", NAMES)
ENTROPY_CONDITIONS = ("Lname", "L1")
"""Settings whose entropy is compared with no information."""


@dataclass(frozen=True)
class AnalysisConfig:
    """Comparison options.

    Attributes:
        datasets: Datasets with classifier rows (a fresh generator each).
        bootstrap: Bootstrap resamples.
        seed: Bootstrap seed.
    """

    datasets: tuple[str, ...] = ("banking77", "clinc150")
    bootstrap: int = 2000
    seed: int = 0


def paired(
    a: pd.DataFrame, b: pd.DataFrame, cond_a: str, cond_b: str, col: str
) -> np.ndarray:
    """Per-item ``a[cond_a][col] - b[cond_b][col]`` on items present in both.

    Args:
        a: First aggregated frame.
        b: Second aggregated frame.
        cond_a: Condition in ``a``.
        cond_b: Condition in ``b``.
        col: Column to difference.

    Returns:
        Differences, one per shared item.
    """
    x = a[a["condition"] == cond_a].set_index("item")[col]
    y = b[b["condition"] == cond_b].set_index("item")[col]
    shared = x.index.intersection(y.index)
    diff: np.ndarray = (x[shared] - y[shared]).to_numpy(float)
    return diff


def _bins(cells: pd.DataFrame, cond: str) -> list[dict[str, float]]:
    c = cells[cells["condition"] == cond]
    if not len(c):
        return []
    return reliability_bins(c["p_max"].to_numpy(float), c["correct"].to_numpy(bool))


def _smece_of(cells: pd.DataFrame, cond: str) -> float:
    c = cells[cells["condition"] == cond]
    return smece(c["p_max"].to_numpy(), c["correct"].to_numpy())


def _names_summary(names: pd.DataFrame) -> dict[str, Any]:
    return {
        "n": len(names),
        "accuracy": float(names["correct"].mean()),
        "p_max": float(names["p_max"].mean()),
        "norm_entropy": float(names["norm_entropy"].mean()),
        "smece": smece(names["p_max"].to_numpy(), names["correct"].to_numpy())
        if len(names)
        else float("nan"),
    }


def _api_table(api: pd.DataFrame) -> dict[str, Any]:
    return {
        str(c): {
            "accuracy": float(g["correct"].mean()),
            "p_max": float(g["p_max"].mean()),
            "norm_entropy": float(g["norm_entropy"].mean()),
            "chose_first_listed": float(g["chose_first_listed"].mean()),
        }
        for c, g in api.groupby("condition")
        if c in ("L0", "Lname", "L1")
    }


def compare_rows(
    gliner_rows: pd.DataFrame, api_rows: pd.DataFrame, b: int, seed: int = 0
) -> dict[str, Any]:
    """Compare the classifier with the API on the same items.

    Draw order: the classifier's condition table, its no-information
    summary, the paired ``L0`` top-probability difference, then the paired
    entropy differences.

    Args:
        gliner_rows: Classifier rows.
        api_rows: Knowledge-dial rows of the same dataset.
        b: Bootstrap resamples.
        seed: Bootstrap seed.

    Returns:
        JSON-serialisable results.
    """
    rng = np.random.default_rng(seed)
    gl = aggregate(gliner_rows)
    api_all = aggregate(api_rows)
    api = api_all[(api_all["arm"] == "main") & api_all["item"].isin(set(gl["item"]))]
    table = condition_table(gl, b, rng)
    ignorance = no_knowledge(gl, b, rng)
    d_l0 = paired(api, gl, "L0", "L0", "p_max")
    l0 = {
        "mean": float(d_l0.mean()),
        "ci95": list(percentile_ci(bootstrap_means(d_l0, b, rng))),
        "n": len(d_l0),
    }
    present = set(gl["condition"])
    entropy: dict[str, Any] = {}
    for cond in [c for c in ENTROPY_CONDITIONS if c in present]:
        d = paired(gl, gl, cond, "L0", "norm_entropy")
        entropy[cond] = {
            "mean": float(d.mean()),
            "ci95": list(percentile_ci(bootstrap_means(d, b, rng))),
        }
    return {
        "reliability_bins": {
            cond: {"jev": _bins(api, cond), "gliner": _bins(gl, cond)}
            for cond in FIGURE_PANELS
        },
        "gliner_conditions": table,
        "gliner_names": _names_summary(gl[gl["condition"] == NAMES]),
        "gliner_l0_chose_first_listed": float(
            gl[gl["condition"] == "L0"]["chose_first_listed"].mean()
        ),
        "jev_same_items": _api_table(api),
        "gliner_no_knowledge": ignorance,
        "l0_p_max_jev_minus_gliner": l0,
        "gliner_entropy_minus_l0": entropy,
        "smece": {
            "gliner": {
                c: table[c]["smece"]["value"] for c in ENTROPY_CONDITIONS if c in table
            },
            "jev_same_items": {
                c: _smece_of(api, c) for c in ENTROPY_CONDITIONS if c in present
            },
        },
        "n_tokens_median": {
            str(c): int(g)
            for c, g in gliner_rows.groupby("condition")["n_tokens"].median().items()
        },
    }


def compare(settings: Settings, config: AnalysisConfig) -> dict[str, Any]:
    """Compare every dataset with classifier rows and write the summary.

    Args:
        settings: Run settings.
        config: Options.

    Returns:
        Results keyed by dataset.
    """
    from beyond_answer_confidence.experiments.knowledge_dial import rows_path

    results: dict[str, Any] = {}
    for name in config.datasets:
        path = rows_file(settings, name)
        if not path.exists():
            logger.warning("no classifier rows for %s at %s", name, path)
            continue
        results[name] = compare_rows(
            pd.DataFrame(read_jsonl(path)),
            pd.DataFrame(read_jsonl(rows_path(settings, "test", name))),
            config.bootstrap,
            config.seed,
        )
    write_json(
        settings.out(NAME) / "summary.json", {"config": as_dict(config), **results}
    )
    return results


ANALYSIS = Analysis(
    name=f"{NAME}_comparison",
    summary="Compare the API with GLiNER2.5-Decide on the same items",
    config_type=AnalysisConfig,
    run=compare,
)
