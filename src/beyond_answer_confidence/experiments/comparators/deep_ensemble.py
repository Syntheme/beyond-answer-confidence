"""A fine-tuned deep ensemble as an epistemic reference.

Five fine-tuned copies of a small encoder
(``sentence-transformers/all-MiniLM-L6-v2``, pinned revision, Apache-2.0)
that differ only in their seed. Unlike a black box, an ensemble has a
standard per-item epistemic estimate: the entropy of the mean prediction
(total) splits into the mean member entropy (aleatoric) plus the mutual
information between prediction and member (MI, the disagreement: epistemic).

Setups per dataset:

- ``full``: train on the whole in-scope pool; evaluate on the
  knowledge-dial test items (and, for CLINC150, its out-of-scope test items);
- ``heldout`` (datasets without an out-of-scope split): train without the
  intents held out by the out-of-scope experiment; evaluate on its items;
- ``k1`` ... ``k8``: train on exactly the knowledge dial's seed-0 example
  sets (k per intent); evaluate on the knowledge-dial test items.

Training needs the optional ``models`` extra (``pip install
'beyond-answer-confidence[models]'``); torch and transformers are imported only when
training runs. Predictions go to ``<output_dir>/deep_ensemble/`` as
``<dataset>_<setup>.npz``; the comparison (:data:`ANALYSIS`) reads them with
the knowledge-dial and out-of-scope outputs and writes ``summary.json``.
"""

import logging
import math
import random
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from beyond_answer_confidence.analyses.knowledge_dial import aggregate
from beyond_answer_confidence.config import as_dict
from beyond_answer_confidence.data.intents import (
    DATASETS,
    Example,
    IntentDataset,
    load,
    select_items,
)
from beyond_answer_confidence.experiments.base import Analysis, read_jsonl, write_json
from beyond_answer_confidence.experiments.knowledge_dial import rows_path
from beyond_answer_confidence.experiments.out_of_scope import NAME as SCOPE
from beyond_answer_confidence.metrics.calibration import smece
from beyond_answer_confidence.metrics.ranking import auroc
from beyond_answer_confidence.settings import Settings
from beyond_answer_confidence.stats.inference import spearman
from beyond_answer_confidence.stats.resampling import bootstrap, on_resample
from beyond_answer_confidence.tasks.intents import sample_examples
from beyond_answer_confidence.tasks.out_of_scope import (
    build_items,
    heldout_intents,
    normalise_key,
)

logger = logging.getLogger(__name__)

NAME = "deep_ensemble"
ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
ENCODER_REVISION = (
    "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"  # pragma: allowlist secret
)
MAX_LEN = 64
BATCH = 32
LR = 5e-5
"""Encoder learning rate."""
HEAD_LR = 2e-3
"""Learning rate of the randomly initialised classification head (with the
encoder's rate the head barely moves and predictions stay near uniform)."""
FULL_EPOCHS = 5
MIN_STEPS = 400
"""Low-data setups train for at least this many optimiser steps."""
MAX_EPOCHS = 200
LOW_DATA_KS = (1, 2, 4, 8)


@dataclass(frozen=True)
class TrainConfig:
    """Ensemble training options.

    Attributes:
        datasets: Datasets to train on.
        setups: Setup names to train (empty = all).
        member_seeds: One ensemble member per seed.
        per_intent: Knowledge-dial test items per intent (evaluation set).
        heldout_intents: Intents held out of the ``heldout`` setup.
        threads: Torch CPU threads per process.
    """

    datasets: tuple[str, ...] = DATASETS
    setups: tuple[str, ...] = ()
    member_seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    per_intent: int = 10
    heldout_intents: int = 10
    threads: int = 7


@dataclass(frozen=True)
class Setup:
    """One ensemble training setup.

    Attributes:
        name: Setup name.
        labels: Output classes.
        train: Training examples.
        eval_ids: Evaluation item ids (``test:<i>``, ``oos:<i>`` or
            out-of-scope item keys).
        eval_texts: Evaluation utterances.
        epochs: Training epochs.
    """

    name: str
    labels: tuple[str, ...]
    train: tuple[Example, ...]
    eval_ids: tuple[str, ...]
    eval_texts: tuple[str, ...]
    epochs: int


def epochs_for(n_train: int, full: bool) -> int:
    """Number of epochs: fixed for full data, enough steps for tiny sets.

    Args:
        n_train: Training examples.
        full: Whether this is a full-data setup.

    Returns:
        Epoch count.
    """
    if full:
        return FULL_EPOCHS
    steps_per_epoch = math.ceil(n_train / BATCH)
    return int(
        min(MAX_EPOCHS, max(FULL_EPOCHS, math.ceil(MIN_STEPS / steps_per_epoch)))
    )


def setups_for(
    ds: IntentDataset, per_intent: int = 10, n_heldout: int = 10
) -> list[Setup]:
    """Every training setup of a dataset.

    Args:
        ds: The dataset.
        per_intent: Knowledge-dial test items per intent.
        n_heldout: Intents held out by the out-of-scope experiment (datasets
            without an out-of-scope split).

    Returns:
        Setups in run order.
    """
    main_idx = select_items(ds.test, per_intent)
    dial_ids = tuple(f"test:{i}" for i in main_idx)
    dial_texts = tuple(ds.test[i].text for i in main_idx)
    ids, texts = dial_ids, dial_texts
    if ds.oos_test:
        ids += tuple(f"oos:{i}" for i in range(len(ds.oos_test)))
        texts += tuple(ex.text for ex in ds.oos_test)
    out = [
        Setup("full", ds.labels, ds.pool, ids, texts, epochs_for(len(ds.pool), True))
    ]
    if not ds.oos_test:
        held = set(heldout_intents(ds.labels, n_heldout))
        _, items = build_items(ds, per_intent, n_heldout)
        train = tuple(ex for ex in ds.pool if ex.label not in held)
        out.append(
            Setup(
                "heldout",
                tuple(lab for lab in ds.labels if lab not in held),
                train,
                tuple(it.key for it in items),
                tuple(it.text for it in items),
                epochs_for(len(train), True),
            )
        )
    examples = sample_examples(ds, seed=0)
    for k in LOW_DATA_KS:
        train = tuple(Example(t, lab) for lab in ds.labels for t in examples[lab][:k])
        out.append(
            Setup(
                f"k{k}",
                ds.labels,
                train,
                dial_ids,
                dial_texts,
                epochs_for(len(train), False),
            )
        )
    return out


def decompose(member_probs: np.ndarray) -> dict[str, np.ndarray]:
    """Entropy decomposition of an ensemble's predictions.

    Args:
        member_probs: Shape ``(members, items, K)``.

    Returns:
        Per item: ``mean`` probabilities, ``total`` entropy, ``aleatoric``
        (mean member entropy) and ``mi`` (total - aleatoric), in nats.
    """
    eps = 1e-12
    mean = member_probs.mean(axis=0)
    total = -(mean * np.log(mean + eps)).sum(axis=-1)
    aleatoric = -(member_probs * np.log(member_probs + eps)).sum(axis=-1).mean(axis=0)
    return {
        "mean": mean,
        "total": total,
        "aleatoric": aleatoric,
        "mi": np.maximum(total - aleatoric, 0.0),
    }


def train_and_predict(
    setup: Setup, seed: int, hf_dir: Path, threads: int
) -> np.ndarray:  # pragma: no cover - needs torch and a model download
    """Fine-tune one ensemble member and predict on the setup's evaluation texts.

    Args:
        setup: The setup.
        seed: Member seed (head initialisation, regularisation noise, data order).
        hf_dir: Hugging Face download cache.
        threads: Torch CPU threads.

    Returns:
        Probabilities, shape ``(n_eval, K)``.
    """
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    cache = str(hf_dir)
    # Pinned revision (satisfies bandit B615).
    tok = AutoTokenizer.from_pretrained(
        ENCODER, revision=ENCODER_REVISION, cache_dir=cache
    )
    # Pinned revision (satisfies bandit B615); it ships model.safetensors, so
    # loading never falls back to pickle.
    model = AutoModelForSequenceClassification.from_pretrained(
        ENCODER,
        revision=ENCODER_REVISION,
        num_labels=len(setup.labels),
        cache_dir=cache,
        use_safetensors=True,
    )
    index = {lab: i for i, lab in enumerate(setup.labels)}
    texts = [ex.text for ex in setup.train]
    y = torch.tensor([index[ex.label] for ex in setup.train])
    # Tokenise once without padding; pad each batch only to its longest item.
    ids = tok(texts, truncation=True, max_length=MAX_LEN)["input_ids"]
    head = [p for n, p in model.named_parameters() if n.startswith("classifier")]
    body = [p for n, p in model.named_parameters() if not n.startswith("classifier")]
    opt = torch.optim.AdamW(
        [{"params": body, "lr": LR}, {"params": head, "lr": HEAD_LR}]
    )
    total_steps = setup.epochs * math.ceil(len(texts) / BATCH)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda step: max(0.0, 1 - step / total_steps)
    )
    gen = torch.Generator().manual_seed(seed)
    model.train()
    for _ in range(setup.epochs):
        order = torch.randperm(len(texts), generator=gen).tolist()
        for start in range(0, len(texts), BATCH):
            idx = order[start : start + BATCH]
            batch = tok.pad({"input_ids": [ids[i] for i in idx]}, return_tensors="pt")
            model(**batch, labels=y[idx]).loss.backward()
            opt.step()
            sched.step()
            opt.zero_grad()
    model.eval()
    probs = []
    with torch.no_grad():
        for start in range(0, len(setup.eval_texts), 256):
            batch = tok(
                list(setup.eval_texts[start : start + 256]),
                padding=True,
                truncation=True,
                max_length=MAX_LEN,
                return_tensors="pt",
            )
            probs.append(torch.softmax(model(**batch).logits, dim=-1).numpy())
    return np.concatenate(probs)


def predictions_path(settings: Settings, dataset: str, setup: str) -> Path:
    """Where one setup's member predictions are saved.

    Args:
        settings: Run settings.
        dataset: Dataset name.
        setup: Setup name.

    Returns:
        ``<output_dir>/deep_ensemble/<dataset>_<setup>.npz``.
    """
    return settings.out(NAME) / f"{dataset}_{setup}.npz"


def train(
    settings: Settings, config: TrainConfig
) -> dict[str, Any]:  # pragma: no cover - needs torch and a model download
    """Train every setup's ensemble and save predictions (existing files are kept).

    Args:
        settings: Run settings.
        config: Options.

    Returns:
        Training size, epochs and time per dataset and setup.
    """
    summary: dict[str, Any] = {"config": as_dict(config)}
    for name in config.datasets:
        ds = load(name, settings.data_dir)
        for setup in setups_for(ds, config.per_intent, config.heldout_intents):
            if config.setups and setup.name not in config.setups:
                continue
            path = predictions_path(settings, name, setup.name)
            if path.exists():
                logger.info("%s/%s: exists, skipping", name, setup.name)
                continue
            start = time.time()
            members = np.stack(
                [
                    train_and_predict(setup, s, settings.hf_dir, config.threads)
                    for s in config.member_seeds
                ]
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                path,
                probs=members,
                labels=np.array(setup.labels),
                eval_ids=np.array(setup.eval_ids),
            )
            summary[f"{name}/{setup.name}"] = {
                "n_train": len(setup.train),
                "epochs": setup.epochs,
                "seconds": round(time.time() - start),
            }
            logger.info("%s/%s: %s", name, setup.name, summary[f"{name}/{setup.name}"])
    write_json(settings.out(NAME) / "training.json", summary)
    return summary


TRAINING = Analysis(
    name=NAME,
    summary="Train the deep-ensemble comparator (needs the 'models' extra)",
    config_type=TrainConfig,
    run=train,
)
"""Training entry point (local compute, no API calls)."""


# --- Comparison with the API ------------------------------------------------------

API_CONDITIONS = ("Lname", "L8", "Lname+8")
"""Knowledge-dial conditions compared item by item with the ensemble."""


@dataclass(frozen=True)
class AnalysisConfig:
    """Comparison options.

    Attributes:
        datasets: Datasets to compare (one bootstrap generator, in order).
        bootstrap: Bootstrap resamples.
        seed: Bootstrap seed.
    """

    datasets: tuple[str, ...] = DATASETS
    bootstrap: int = 2000
    seed: int = 0


def boot_ci(
    stat: Callable[[np.ndarray], float], n: int, b: int, rng: np.random.Generator
) -> list[float]:
    """Item-bootstrap 95 % interval of ``stat(idx)``.

    The percentiles are taken at exactly 2.5 and 97.5 (``percentile_ci``
    computes the levels from the coverage, which can differ in the last
    bit); the fixed levels are kept for reproducibility of released results.

    Args:
        stat: Function of a resampled index array.
        n: Number of items.
        b: Resamples.
        rng: Random generator.

    Returns:
        ``[low, high]``.
    """
    vals = bootstrap(n, stat, b, rng)
    return [float(v) for v in np.nanpercentile(vals, [2.5, 97.5])]


def ensemble_frame(path: Path, gold: Mapping[str, str | None]) -> pd.DataFrame:
    """Per-item ensemble statistics from saved member predictions.

    Args:
        path: ``.npz`` with ``probs`` (members x items x K), ``labels`` and
            ``eval_ids``.
        gold: Gold label (``None`` for out of scope) per evaluation id.

    Returns:
        One row per evaluation id.
    """
    z = np.load(path)
    labels = list(z["labels"])
    ids = [normalise_key(str(i)) for i in z["eval_ids"]]
    d = decompose(z["probs"])
    top = d["mean"].argmax(axis=1)
    return pd.DataFrame(
        {
            "id": ids,
            "gold": [gold[i] for i in ids],
            "correct": [gold[i] == labels[t] for i, t in zip(ids, top, strict=True)],
            "p_max": d["mean"].max(axis=1),
            "norm_entropy": d["total"] / math.log(len(labels)),
            "mi": d["mi"],
            "aleatoric": d["aleatoric"],
        }
    )


def api_frame(rows: pd.DataFrame) -> pd.DataFrame:
    """Replicate-mean per-item statistics of the knowledge dial's main arm.

    Args:
        rows: Knowledge-dial test rows of one dataset.

    Returns:
        One row per (item, condition), with ``id = "test:<item>"``.
    """
    agg = aggregate(rows)
    main = agg[agg["arm"] == "main"].copy()
    main["id"] = "test:" + main["item"].astype(str)
    return main


def agreement(
    ens: pd.DataFrame, api: pd.DataFrame, b: int, rng: np.random.Generator
) -> dict[str, Any]:
    """Spearman agreement between ensemble and API uncertainty per item.

    Args:
        ens: Ensemble frame (in-scope items only).
        api: API frame.
        b: Bootstrap resamples.
        rng: Random generator (two bootstraps per condition).

    Returns:
        Correlations with intervals and error overlap per API condition.
    """
    out: dict[str, Any] = {}
    for cond in API_CONDITIONS:
        m = ens.merge(
            api[api["condition"] == cond][["id", "norm_entropy", "correct"]],
            on="id",
            suffixes=("_ens", "_jev"),
        )
        if m.empty:
            continue
        x_mi, x_h = m["mi"].to_numpy(), m["norm_entropy_ens"].to_numpy()
        y = m["norm_entropy_jev"].to_numpy()
        out[cond] = {
            "n": len(m),
            "spearman_mi_vs_jev_entropy": spearman(x_mi, y),
            "ci95_mi": boot_ci(on_resample(spearman, x_mi, y), len(m), b, rng),
            "spearman_total_entropy_vs_jev_entropy": spearman(x_h, y),
            "ci95_total": boot_ci(on_resample(spearman, x_h, y), len(m), b, rng),
            "error_agreement": float(np.mean(m["correct_ens"] == m["correct_jev"])),
            "both_wrong": int((~m["correct_ens"] & ~m["correct_jev"]).sum()),
        }
    return out


def _auroc_block(
    scores: Mapping[str, np.ndarray], pos: np.ndarray, b: int, rng: np.random.Generator
) -> dict[str, Any]:
    """AUROC with an item-bootstrap interval per score."""
    return {
        name: {
            "auroc": auroc(s, pos),
            "ci95": boot_ci(on_resample(auroc, s, pos), len(pos), b, rng),
        }
        for name, s in scores.items()
    }


def _auroc_difference(
    a: np.ndarray, c: np.ndarray, pos: np.ndarray
) -> Callable[[np.ndarray], float]:
    """``idx -> AUROC(a) - AUROC(c)`` on the resampled rows."""

    def at(idx: np.ndarray) -> float:
        return auroc(a[idx], pos[idx]) - auroc(c[idx], pos[idx])

    return at


def oos_comparison(
    ens: pd.DataFrame,
    scope_items: pd.DataFrame,
    id_map: Mapping[str, str],
    b: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Out-of-scope AUROC: ensemble scores vs the API's scores on the same items.

    Args:
        ens: Ensemble frame covering in-scope and out-of-scope ids.
        scope_items: Out-of-scope experiment item rows.
        id_map: Item key -> ensemble evaluation id.
        b: Bootstrap resamples.
        rng: Random generator.

    Returns:
        AUROC per score and the paired difference between the best API and
        the best ensemble score.
    """
    items = scope_items.copy()
    items["id"] = items["key"].map(id_map)
    m = items.merge(ens, on="id", suffixes=("_jev", "_ens"))
    pos = m["oos"].to_numpy(bool)
    scores = {
        "ens_mi": m["mi"].to_numpy(),
        "ens_total_entropy": m["norm_entropy"].to_numpy(),
        "ens_1_minus_p_max": 1 - m["p_max"].to_numpy(),
        "jev_closed_1_minus_p_max": 1 - m["closed_p_max"].to_numpy(),
        "jev_with_oos_p_oos": m["p_oos"].to_numpy(),
        "jev_fits_1_minus_p": 1 - m["p_fits"].to_numpy(),
    }
    out: dict[str, Any] = {"n_in_scope": int((~pos).sum()), "n_oos": int(pos.sum())}
    out |= _auroc_block(scores, pos, b, rng)
    best_api = max(
        (k for k in scores if k.startswith("jev")), key=lambda k: out[k]["auroc"]
    )
    best_ens = max(
        (k for k in scores if k.startswith("ens")), key=lambda k: out[k]["auroc"]
    )
    sa, se = scores[best_api], scores[best_ens]
    out["paired_best_jev_minus_best_ens"] = {
        "jev": best_api,
        "ens": best_ens,
        "diff": out[best_api]["auroc"] - out[best_ens]["auroc"],
        "ci95": boot_ci(_auroc_difference(sa, se, pos), len(m), b, rng),
    }
    return out


def ensemble_oos_only(
    ens: pd.DataFrame, b: int, rng: np.random.Generator
) -> dict[str, Any]:
    """Out-of-scope AUROC of the ensemble alone (when the API's scores are missing).

    Args:
        ens: Ensemble frame; out-of-scope ids contain ``"oos"``.
        b: Bootstrap resamples.
        rng: Random generator.

    Returns:
        AUROC with interval per ensemble score.
    """
    pos = ens["id"].str.contains("oos").to_numpy(bool)
    scores = {
        "ens_mi": ens["mi"].to_numpy(),
        "ens_total_entropy": ens["norm_entropy"].to_numpy(),
        "ens_1_minus_p_max": 1 - ens["p_max"].to_numpy(),
    }
    out: dict[str, Any] = {
        "n_in_scope": int((~pos).sum()),
        "n_oos": int(pos.sum()),
        "jev_scores_missing": True,
    }
    return out | _auroc_block(scores, pos, b, rng)


def reducibility(
    paths: Mapping[int, Path], gold: Mapping[str, str | None], api: pd.DataFrame
) -> dict[str, Any]:
    """Ensemble vs API as the number of examples per intent grows.

    Args:
        paths: k -> saved predictions of the ``k<k>`` setup (existing files).
        gold: Gold label per evaluation id.
        api: API frame.

    Returns:
        Per k: accuracy, normalised entropy and SmoothECE of both, ensemble MI.
    """
    out: dict[str, Any] = {}
    for k, path in paths.items():
        e = ensemble_frame(path, gold)
        j = api[api["condition"] == f"L{k}"]
        out[f"k{k}"] = {
            "ens_accuracy": float(e["correct"].mean()),
            "ens_norm_entropy": float(e["norm_entropy"].mean()),
            "ens_mi": float(e["mi"].mean()),
            "ens_smece": smece(e["p_max"].to_numpy(), e["correct"].to_numpy()),
            "jev_accuracy": float(j["correct"].mean()),
            "jev_norm_entropy": float(j["norm_entropy"].mean()),
            "jev_smece": smece(j["p_max"].to_numpy(), j["correct"].to_numpy()),
        }
    return out


def scope_id(key: str) -> str:
    """Map an out-of-scope item key to the ensemble's evaluation id.

    Args:
        key: ``<dataset>:in|oos:<index>``.

    Returns:
        ``test:<index>`` or ``oos:<index>``.
    """
    *_, kind, index = key.split(":")
    return f"{'test' if kind == 'in' else 'oos'}:{index}"


def _full_summary(in_scope: pd.DataFrame) -> dict[str, Any]:
    return {
        "n": len(in_scope),
        "accuracy": float(in_scope["correct"].mean()),
        "smece": smece(in_scope["p_max"].to_numpy(), in_scope["correct"].to_numpy()),
        "mean_mi": float(in_scope["mi"].mean()),
        "mean_norm_entropy": float(in_scope["norm_entropy"].mean()),
    }


def _oos_block(
    settings: Settings,
    ds: IntentDataset,
    full: pd.DataFrame,
    b: int,
    rng: np.random.Generator,
) -> dict[str, Any] | None:
    """Out-of-scope comparison for one dataset, with whatever outputs exist."""
    scope_path = settings.out(SCOPE) / f"{ds.name}_items.jsonl"
    heldout = predictions_path(settings, ds.name, "heldout")
    if not scope_path.exists():
        if ds.oos_test:
            return ensemble_oos_only(full, b, rng)
        if heldout.exists():
            _, items = build_items(ds)
            held_gold = {it.key: it.gold for it in items}
            return ensemble_oos_only(ensemble_frame(heldout, held_gold), b, rng)
        return None
    # pandas' own JSON reader (its float parsing is kept for reproducibility).
    scope_items = pd.read_json(scope_path, lines=True)
    if ds.oos_test:
        id_map = {k: scope_id(k) for k in scope_items["key"]}
        return oos_comparison(full, scope_items, id_map, b, rng)
    if heldout.exists():
        held_gold = dict(zip(scope_items["key"], scope_items["gold"], strict=True))
        held = ensemble_frame(heldout, held_gold)
        return oos_comparison(
            held, scope_items, {k: k for k in scope_items["key"]}, b, rng
        )
    return None


def compare_dataset(
    settings: Settings, name: str, b: int, rng: np.random.Generator
) -> dict[str, Any]:
    """All comparisons for one dataset.

    Args:
        settings: Run settings (locates the inputs).
        name: Dataset name.
        b: Bootstrap resamples.
        rng: Random generator.

    Returns:
        Ensemble summary, agreement, reducibility and (if possible) the
        out-of-scope comparison.
    """
    ds = load(name, settings.data_dir)
    gold: dict[str, str | None] = {
        f"test:{i}": ex.label for i, ex in enumerate(ds.test)
    }
    gold |= {f"oos:{i}": None for i in range(len(ds.oos_test))}
    api = api_frame(pd.DataFrame(read_jsonl(rows_path(settings, "test", name))))
    full = ensemble_frame(predictions_path(settings, name, "full"), gold)
    in_scope = full[full["id"].str.startswith("test:")]
    ks = {
        k: p
        for k in LOW_DATA_KS
        if (p := predictions_path(settings, name, f"k{k}")).exists()
    }
    res: dict[str, Any] = {
        "ensemble_full": _full_summary(in_scope),
        "agreement": agreement(in_scope, api, b, rng),
        "reducibility": reducibility(ks, gold, api),
    }
    if (oos := _oos_block(settings, ds, full, b, rng)) is not None:
        res["oos"] = oos
    return res


def compare(settings: Settings, config: AnalysisConfig) -> dict[str, Any]:
    """Compare the ensemble with the API and write the summary.

    Args:
        settings: Run settings.
        config: Options.

    Returns:
        Results keyed by dataset.
    """
    rng = np.random.default_rng(config.seed)
    results = {
        name: compare_dataset(settings, name, config.bootstrap, rng)
        for name in config.datasets
    }
    write_json(
        settings.out(NAME) / "summary.json", {"config": as_dict(config), **results}
    )
    return results


ANALYSIS = Analysis(
    name=f"{NAME}_comparison",
    summary="Compare the API's uncertainty with a deep ensemble's (saved predictions)",
    config_type=AnalysisConfig,
    run=compare,
)
