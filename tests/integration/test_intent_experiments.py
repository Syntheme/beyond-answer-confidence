"""Offline end-to-end runs of the intent experiments on synthetic data.

Answers come from a deterministic fake that favours codes whose description
mentions the message's (synthetic) intent token, so knowledge helps. The
cache is filled through the backend first; every experiment then runs
offline from it, like a reproduction of a finished live run.
"""

import asyncio
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from beyond_answer_confidence.analyses import error_detection
from beyond_answer_confidence.backends.base import RawAnswer, Request
from beyond_answer_confidence.backends.cache import CachedClient
from beyond_answer_confidence.data import intents as data
from beyond_answer_confidence.data.intents import Example, IntentDataset
from beyond_answer_confidence.experiments import (
    instrument_checks,
    knowledge_dial,
    out_of_scope,
    sampling_frequencies,
)
from beyond_answer_confidence.experiments.base import write_jsonl
from beyond_answer_confidence.experiments.comparators import deep_ensemble, gliner
from beyond_answer_confidence.settings import Settings
from beyond_answer_confidence.tasks.schema import Task


def synthetic(name: str, n_labels: int, with_oos: bool) -> IntentDataset:
    labels = tuple(f"intent_{i}" for i in range(n_labels))
    pool = tuple(Example(f"{lab} pool {j}", lab) for lab in labels for j in range(10))
    dev = tuple(Example(f"{lab} dev {j}", lab) for lab in labels for j in range(4))
    test = tuple(Example(f"{lab} test {j}", lab) for lab in labels for j in range(3))
    oos = tuple(Example(f"unrelated filler {j}", "oos") for j in range(80))
    if not with_oos:
        return IntentDataset(name, labels, pool, dev, test)
    return IntentDataset(name, labels, pool, dev, test, oos_test=oos[:6], oos_pool=oos)


BANKING = synthetic("banking77", 8, with_oos=False)
CLINC = synthetic("clinc150", 6, with_oos=True)


def _noise(*parts: object) -> float:
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()
    return int(digest[:4], 16) / 65536


class KnowingBackend:
    """Deterministic fake that uses whatever the codes' descriptions reveal."""

    def __init__(self) -> None:
        self.calls = 0

    async def __aenter__(self) -> "KnowingBackend":
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    def _choice(self, q: dict[str, Any], msg: str, rep: int) -> dict[str, Any]:
        token = msg.split(" ")[0]
        weights = {}
        for i, (code, desc) in enumerate(q["criteria"].items()):
            text = json.dumps(desc)
            hits = text.count(token + " ") + text.count(token + '"')
            first = 1.5 if i == 0 else 0.0
            weights[code] = 1.0 + 3.0 * hits + first + _noise(msg, code, rep)
        total = sum(weights.values())
        probs = {c: w / total for c, w in weights.items()}
        k, top = len(probs), max(probs.values())
        return {
            "type": "choice",
            "choice": max(probs, key=probs.__getitem__),
            "confidence": (k * top - 1) / (k - 1),
            "probabilities": probs,
        }

    async def answer(self, request: Request, model: str) -> RawAnswer:
        self.calls += 1
        msg = request.state["customer_message"]
        answers: dict[str, Any] = {}
        for name, q in request.questions.items():
            if q["type"] == "noul":
                p = 0.8 if msg.startswith("intent") else 0.3
                answers[name] = {"type": "noul", "noul": p}
            else:
                answers[name] = self._choice(q, msg, request.replicate)
        body = {"model": model, "usage": {"input_tokens": 50}, "answers": answers}
        return RawAnswer(body)


@pytest.fixture
def settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    monkeypatch.setattr(data, "load_banking77", lambda _d: BANKING)
    monkeypatch.setattr(data, "load_clinc150", lambda _d: CLINC)
    return Settings(data_dir=tmp_path / "data", output_dir=tmp_path / "out")


def prefill(settings: Settings, tasks: Sequence[Task]) -> int:
    """Answer tasks through the fake into the shared cache; return new calls."""
    client = CachedClient(
        settings.cache_file("intents"), model=settings.model, max_input_tokens=10**9
    )
    backend = KnowingBackend()
    results = asyncio.run(client.ask_many([t.request for t in tasks], backend))
    assert not [r for r in results if isinstance(r, BaseException)]
    return backend.calls


def run_offline(settings: Settings, exp: Any, config: Any) -> dict[str, Any]:
    prefill(settings, exp.requests(settings, config))
    summary: dict[str, Any] = dict(exp.run(settings, config))
    assert summary["new_api_calls"] == 0
    assert exp.run(settings, config) == summary  # offline reruns are identical
    return summary


def test_instrument_checks(settings: Settings) -> None:
    cfg = instrument_checks.Config(determinism_items=4, order_items=4, batch_items=2)
    s = run_offline(settings, instrument_checks.EXPERIMENT, cfg)
    assert s["determinism"]["deterministic"] is False  # replicates add noise
    assert s["confidence_formula"]["matches_documented_formula"] is True
    assert s["cost"]["banking77/Lname"]["accuracy"] == 1.0
    assert s["cost"]["clinc150/L0"]["K"] == 6
    assert (settings.out("instrument_checks") / "summary.json").exists()


def test_instrument_checks_fail_without_cache(settings: Settings) -> None:
    with pytest.raises(RuntimeError, match="unanswered"):
        instrument_checks.EXPERIMENT.run(settings, instrument_checks.Config())


def test_sampling_frequencies(settings: Settings) -> None:
    cfg = sampling_frequencies.Config(replicates=6)
    s = run_offline(settings, sampling_frequencies.EXPERIMENT, cfg)
    assert len(s["per_series"]) == 10
    assert all(v["replicates"] == 6 for v in s["per_series"].values())
    assert s["errors"] == 0


def dial_config() -> knowledge_dial.Config:
    return knowledge_dial.Config(
        per_intent=2, subset_per_intent=1, replicates=2, bootstrap=50, chunk_size=100
    )


def test_knowledge_dial_and_comparisons(settings: Settings, tmp_path: Path) -> None:
    dial = run_offline(settings, knowledge_dial.EXPERIMENT, dial_config())
    tasks = knowledge_dial.EXPERIMENT.requests(settings, dial_config())
    assert (
        dial["datasets"]["banking77"]["rows"]
        == dial["datasets"]["banking77"]["requests"]
    )
    assert len(tasks) == sum(d["requests"] for d in dial["datasets"].values())
    b77 = dial["results"]["banking77"]
    assert b77["conditions"]["L8"]["correct"]["mean"] == 1.0
    assert b77["conditions"]["L0"]["norm_entropy"]["mean"] > 0.9
    assert b77["dose_response"]["item_bootstrap"]["slope"] < 0
    assert b77["design_checks"]["swap_chose_swap_target"] == 1.0
    assert knowledge_dial.rows_path(settings, "test", "clinc150").exists()

    scope_cfg = out_of_scope.Config(per_intent=2, heldout_intents=2, bootstrap=50)
    scope = run_offline(settings, out_of_scope.EXPERIMENT, scope_cfg)
    assert scope["banking77"]["heldout_intents"] == ["intent_6", "intent_7"]
    assert scope["clinc150"]["metrics"]["n_oos"] == len(CLINC.oos_test)
    assert scope["clinc150"]["metrics"]["fits_1_minus_p"]["auroc"] == 1.0

    write_ensemble_predictions(settings)
    ens = deep_ensemble.ANALYSIS.run(
        settings, deep_ensemble.AnalysisConfig(bootstrap=30)
    )
    assert set(ens["banking77"]) == {
        "ensemble_full",
        "agreement",
        "reducibility",
        "oos",
    }
    assert ens["clinc150"]["oos"]["n_oos"] == len(CLINC.oos_test)
    assert set(ens["banking77"]["reducibility"]) == {"k1", "k8"}

    det = error_detection.ANALYSIS.run(settings, error_detection.Config(bootstrap=30))
    assert det["banking77"]["error_detection"]["L0"]["n"] == 2 * len(BANKING.labels)
    assert set(det["clinc150"]["reliability_bins"]) == set(error_detection.KNOWLEDGE)

    write_gliner_rows(settings)
    gl = gliner.ANALYSIS.run(settings, gliner.AnalysisConfig(bootstrap=30))
    assert set(gl) == {"banking77"}
    assert gl["banking77"]["gliner_names"]["n"] == 2 * len(BANKING.labels)
    assert (settings.out("gliner") / "summary.json").exists()


def write_ensemble_predictions(settings: Settings) -> None:
    rng = np.random.default_rng(0)
    for ds in (BANKING, CLINC):
        for setup in deep_ensemble.setups_for(ds, per_intent=2, n_heldout=2):
            if setup.name not in ("full", "heldout", "k1", "k8"):
                continue
            probs = rng.dirichlet(np.ones(len(setup.labels)), (3, len(setup.eval_ids)))
            path = deep_ensemble.predictions_path(settings, ds.name, setup.name)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                path,
                probs=probs,
                labels=np.array(setup.labels),
                eval_ids=np.array(setup.eval_ids),
            )


def write_gliner_rows(settings: Settings) -> None:
    cfg = gliner.InferenceConfig(per_intent=2)
    rows = []
    for cond in cfg.conditions:
        for item in gliner.choose_items(BANKING, cfg):
            gold = BANKING.test[item].label
            probs = {lab: (0.65 if lab == gold else 0.05) for lab in BANKING.labels}
            rows.append(gliner.row_for(item, cond, gold, probs, BANKING.labels[0], 90))
    write_jsonl(gliner.rows_file(settings, "banking77"), rows)
