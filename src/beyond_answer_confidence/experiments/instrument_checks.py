"""Instrument checks: is the API's probability output a usable measurement?

Runs on development items only, so the test split stays unseen. Checks:

- determinism: identical requests repeated;
- whether the reported ``confidence`` is exactly ``(K * p_max - 1) / (K - 1)``
  (i.e. carries nothing beyond the probabilities);
- invariance to which opaque code each intent gets (option order);
- batching: several questions in one request vs. one request each;
- token cost and latency per condition.

Output: ``<output_dir>/instrument_checks/summary.json``.
"""

import asyncio
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Any

from beyond_answer_confidence.backends.base import Request, Response
from beyond_answer_confidence.backends.cache import CachedClient, CallRecord
from beyond_answer_confidence.config import as_dict
from beyond_answer_confidence.data.intents import (
    DATASETS,
    IntentDataset,
    filler_texts,
    load,
    spread_indices,
)
from beyond_answer_confidence.experiments.base import (
    Experiment,
    backend_for,
    write_json,
)
from beyond_answer_confidence.metrics.distributions import (
    choice_confidence,
    js_divergence,
)
from beyond_answer_confidence.runner import collect
from beyond_answer_confidence.settings import Settings
from beyond_answer_confidence.tasks.intents import (
    QUESTION,
    ChoiceSpec,
    Condition,
    build_choice,
    label_probs,
    sample_examples,
    state_for,
)
from beyond_answer_confidence.tasks.schema import Task

NAME = "instrument_checks"
CACHE = "intents"


@dataclass(frozen=True)
class Config:
    """Instrument-check sizes.

    Attributes:
        datasets: Datasets loaded (cost is measured on each).
        check_dataset: Dataset of the determinism, order and batching checks.
        determinism_items: Items repeated identically.
        replicates: Repeats per item.
        order_items: Items asked under several code assignments.
        permutations: Code assignments per item.
        batch_items: Items of the batching check.
        batch_perm_seeds: Code-assignment seeds of the batched questions.
        cost_items: Items per dataset and condition for cost and latency.
        cost_conditions: Conditions whose cost is measured.
    """

    datasets: tuple[str, ...] = DATASETS
    check_dataset: str = "banking77"
    determinism_items: int = 10
    replicates: int = 5
    order_items: int = 10
    permutations: int = 5
    batch_items: int = 5
    batch_perm_seeds: tuple[int, ...] = (10, 11, 12)
    cost_items: int = 3
    cost_conditions: tuple[str, ...] = ("L0", "Lname", "L8", "Lname+8")


@dataclass(frozen=True)
class Probe:
    """One request of the checks.

    Attributes:
        check: ``determinism``, ``order``, ``batched``, ``single`` or ``cost``.
        dataset: Dataset name.
        index: Dev index.
        condition: Knowledge condition.
        specs: Question name -> spec.
        request: The request.
    """

    check: str
    dataset: str
    index: int
    condition: Condition
    specs: Mapping[str, ChoiceSpec]
    request: Request


def max_abs_diff(dists: Sequence[Mapping[str, float]]) -> float:
    """Largest absolute probability difference between any two distributions.

    Args:
        dists: Distributions over the same options.

    Returns:
        The maximum over pairs and options (0 for fewer than two).
    """
    return max(
        (abs(a[k] - b[k]) for a, b in combinations(dists, 2) for k in a),
        default=0.0,
    )


def summarise(values: Sequence[float]) -> dict[str, float]:
    """Count, min, median, max and mean (empty dict if empty).

    Args:
        values: Numbers.

    Returns:
        Summary statistics.
    """
    if not values:
        return {}
    return {
        "n": len(values),
        "min": min(values),
        "median": statistics.median(values),
        "max": max(values),
        "mean": statistics.fmean(values),
    }


class ProbeBuilder:
    """Builds probes for loaded datasets."""

    def __init__(
        self, data: Mapping[str, IntentDataset], filler: Sequence[str]
    ) -> None:
        """Sample the example sets.

        Args:
            data: Datasets by name.
            filler: Irrelevant utterances (unused by the default conditions).
        """
        self.data = data
        self.filler = filler
        self.examples = {name: sample_examples(ds, 0) for name, ds in data.items()}

    def spec(self, dataset: str, idx: int, cond: Condition, perm: int) -> ChoiceSpec:
        """Choice question for one dev item.

        Args:
            dataset: Dataset name.
            idx: Dev index.
            cond: Condition.
            perm: Code-assignment seed.

        Returns:
            The spec.
        """
        ds = self.data[dataset]
        return build_choice(
            ds.dev[idx],
            f"{dataset}:dev:{idx}",
            ds,
            cond,
            self.examples[dataset],
            self.filler,
            perm,
        )

    def single(
        self,
        check: str,
        dataset: str,
        idx: int,
        cond: Condition,
        perm: int = 0,
        rep: int = 0,
    ) -> Probe:
        """A one-question probe.

        Args:
            check: Check name.
            dataset: Dataset name.
            idx: Dev index.
            cond: Condition.
            perm: Code-assignment seed.
            rep: Replicate index.

        Returns:
            The probe.
        """
        spec = self.spec(dataset, idx, cond, perm)
        state = state_for(self.data[dataset].dev[idx].text)
        return Probe(
            check,
            dataset,
            idx,
            cond,
            {QUESTION: spec},
            Request(state, {QUESTION: spec.question()}, rep),
        )

    def batching(self, dataset: str, idx: int, perms: Sequence[int]) -> list[Probe]:
        """One batched probe with several questions, then one probe per question.

        Args:
            dataset: Dataset name.
            idx: Dev index.
            perms: Code-assignment seeds, one question each.

        Returns:
            The batched probe followed by the single ones.
        """
        specs = {f"q{p}": self.spec(dataset, idx, Condition.NAME, p) for p in perms}
        state = state_for(self.data[dataset].dev[idx].text)
        out = [
            Probe(
                "batched",
                dataset,
                idx,
                Condition.NAME,
                specs,
                Request(state, {n: s.question() for n, s in specs.items()}),
            )
        ]
        out += [
            Probe(
                "single",
                dataset,
                idx,
                Condition.NAME,
                {n: s},
                Request(state, {n: s.question()}),
            )
            for n, s in specs.items()
        ]
        return out


def build_probes(builder: ProbeBuilder, config: Config) -> list[Probe]:
    """Every probe, in the order the checks read them.

    Args:
        builder: Probe builder over the loaded datasets.
        config: Sizes.

    Returns:
        Determinism, order, batching and cost probes.
    """
    name = config.check_dataset
    size = len(builder.data[name].dev)
    probes = [
        builder.single("determinism", name, idx, Condition.NAME, rep=rep)
        for idx in spread_indices(size, config.determinism_items)
        for rep in range(config.replicates)
    ]
    probes += [
        builder.single("order", name, idx, Condition.NAME, perm=perm)
        for idx in spread_indices(size, config.order_items)
        for perm in range(config.permutations)
    ]
    for idx in spread_indices(size, config.batch_items):
        probes += builder.batching(name, idx, config.batch_perm_seeds)
    probes += [
        builder.single("cost", ds, idx, Condition(cond))
        for ds in builder.data
        for cond in config.cost_conditions
        for idx in spread_indices(len(builder.data[ds].dev), config.cost_items)
    ]
    return probes


def _dist(probe: Probe, response: Response, name: str = QUESTION) -> dict[str, float]:
    return label_probs(probe.specs[name], response.choices[name].probabilities)


def _grouped(
    pairs: Sequence[tuple[Probe, CallRecord]], check: str
) -> dict[int, list[dict[str, float]]]:
    """Distributions over intents of one check, grouped by item (input order)."""
    out: dict[int, list[dict[str, float]]] = {}
    for probe, rec in pairs:
        if probe.check == check:
            out.setdefault(probe.index, []).append(_dist(probe, rec.response))
    return out


def determinism(pairs: Sequence[tuple[Probe, CallRecord]]) -> dict[str, Any]:
    """Largest probability change between identical requests, per item.

    Args:
        pairs: Probes and their records.

    Returns:
        Summary of the per-item maximum absolute difference.
    """
    diffs = [max_abs_diff(d) for d in _grouped(pairs, "determinism").values()]
    return {
        "max_abs_prob_diff_per_item": summarise(diffs),
        "deterministic": max(diffs) == 0.0,
    }


def order_invariance(pairs: Sequence[tuple[Probe, CallRecord]]) -> dict[str, Any]:
    """Distributions over intents under different code assignments.

    Args:
        pairs: Probes and their records.

    Returns:
        Pairwise Jensen-Shannon divergences and the argmax agreement rate.
    """
    js: list[float] = []
    agree: list[bool] = []
    for dists in _grouped(pairs, "order").values():
        js.extend(js_divergence(a, b) for a, b in combinations(dists, 2))
        agree.append(len({max(d, key=d.__getitem__) for d in dists}) == 1)
    return {
        "pairwise_js_nats": summarise(js),
        "argmax_agreement_rate": sum(agree) / len(agree),
    }


def batching(pairs: Sequence[tuple[Probe, CallRecord]]) -> dict[str, Any]:
    """Several questions in one request against one request each.

    Args:
        pairs: Probes and their records.

    Returns:
        Summary of the per-question maximum absolute difference.
    """
    diffs = []
    batched: Response | None = None
    for probe, rec in pairs:
        if probe.check == "batched":
            batched = rec.response
        elif probe.check == "single" and batched is not None:
            (name,) = probe.specs
            a = _dist(probe, batched, name)
            b = _dist(probe, rec.response, name)
            diffs.append(max_abs_diff([a, b]))
    return {
        "max_abs_prob_diff": summarise(diffs),
        "batch_invariant": max(diffs) == 0.0,
    }


def cost(
    pairs: Sequence[tuple[Probe, CallRecord]], labels: Mapping[str, int]
) -> dict[str, Any]:
    """Input tokens, latency, accuracy and top probability per dataset and condition.

    Args:
        pairs: Probes and their records.
        labels: Number of intents per dataset.

    Returns:
        Statistics keyed by ``<dataset>/<condition>``.
    """
    groups: dict[str, list[tuple[Probe, CallRecord]]] = {}
    for probe, rec in pairs:
        if probe.check == "cost":
            groups.setdefault(f"{probe.dataset}/{probe.condition.value}", []).append(
                (probe, rec)
            )
    out: dict[str, Any] = {}
    for key, members in groups.items():
        answers = [
            (p.specs[QUESTION], r.response.choices[QUESTION]) for p, r in members
        ]
        out[key] = {
            "K": labels[members[0][0].dataset],
            "input_tokens": summarise(
                [float(r.response.input_tokens) for _, r in members]
            ),
            "latency_s": summarise(
                [r.latency_s for _, r in members if r.latency_s is not None]
            ),
            "accuracy": statistics.fmean(
                float(a.choice == s.gold_code) for s, a in answers
            ),
            "p_max": summarise([max(a.probabilities.values()) for _, a in answers]),
        }
    return out


def confidence_formula(pairs: Sequence[tuple[Probe, CallRecord]]) -> dict[str, Any]:
    """Reported confidence against ``(K * p_max - 1) / (K - 1)``.

    Uses the single-question answers of the determinism, order and cost
    checks (repeated requests count once per check).

    Args:
        pairs: Probes and their records.

    Returns:
        Absolute residual summary, whether it matches, and probability sums.
    """
    answers = [
        rec.response.choices[QUESTION]
        for probe, rec in pairs
        if probe.check in ("determinism", "order", "cost")
    ]
    resid = [
        abs(a.confidence - choice_confidence(list(a.probabilities.values())))
        for a in answers
        if a.confidence is not None
    ]
    return {
        "confidence_formula": {
            "abs_residual": summarise(resid),
            "matches_documented_formula": bool(resid) and max(resid) < 1e-6,
        },
        "probability_sums": summarise([sum(a.probabilities.values()) for a in answers]),
    }


def analyse(
    pairs: Sequence[tuple[Probe, CallRecord]], labels: Mapping[str, int]
) -> dict[str, Any]:
    """Run every check on answered probes.

    Args:
        pairs: Probes and their records, in build order.
        labels: Number of intents per dataset.

    Returns:
        The check results.
    """
    return {
        "determinism": determinism(pairs),
        "order_invariance": order_invariance(pairs),
        "batching": batching(pairs),
        "cost": cost(pairs, labels),
        **confidence_formula(pairs),
        "response_models": sorted({rec.response.model for _, rec in pairs}),
    }


def _builder(settings: Settings, config: Config) -> ProbeBuilder:
    data = {name: load(name, settings.data_dir) for name in config.datasets}
    filler = filler_texts(data, settings.data_dir) if "clinc150" in data else []
    return ProbeBuilder(data, filler)


def requests(settings: Settings, config: Config) -> list[Task]:
    """Every request of the checks.

    Args:
        settings: Run settings.
        config: Sizes.

    Returns:
        The tasks.
    """
    probes = build_probes(_builder(settings, config), config)
    return [Task(str(i), p.request) for i, p in enumerate(probes)]


def run(settings: Settings, config: Config) -> dict[str, Any]:
    """Collect, check and write the summary.

    Args:
        settings: Run settings.
        config: Sizes.

    Returns:
        The summary.

    Raises:
        RuntimeError: If any request could not be answered.
    """
    builder = _builder(settings, config)
    probes = build_probes(builder, config)
    client = CachedClient(
        settings.cache_file(CACHE),
        model=settings.model,
        max_input_tokens=settings.max_input_tokens,
    )
    tasks = [Task(str(i), p.request) for i, p in enumerate(probes)]
    got = asyncio.run(
        collect(tasks, client, backend_for(settings), concurrency=settings.concurrency)
    )
    if got.errors:
        raise RuntimeError(f"unanswered requests: {dict(got.errors)}")
    pairs = [(p, got.records[t.unit][0]) for p, t in zip(probes, tasks, strict=True)]
    labels = {name: len(ds.labels) for name, ds in builder.data.items()}
    summary = {
        "model": client.model,
        "config": as_dict(config),
        **analyse(pairs, labels),
        "new_api_calls": client.api_calls,
        "new_input_tokens": client.spent_input_tokens,
    }
    write_json(settings.out(NAME) / "summary.json", summary)
    return summary


EXPERIMENT = Experiment(
    name=NAME,
    summary="Determinism, order, batching, confidence-formula and cost checks",
    config_type=Config,
    requests=requests,
    run=run,
)
