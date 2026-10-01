"""Knowledge dial: does uncertainty fall as the model is told more?

Intent-classification items are asked with opaque intent codes that come
with a controlled dose of knowledge (nothing, 1-8 examples, the intent name,
name plus examples, irrelevant filler, or swapped examples); see
:mod:`beyond_answer_confidence.tasks.intents`. Three arms:

- **main**: all nine conditions, ordinal codes, example seed 0, on a
  stratified sample of ``per_intent`` items per intent, ``replicates``
  replicates each;
- **seeds**: the example-dependent conditions with other example sets on a
  nested ``subset_per_intent`` subset, one replicate (separates "which
  examples" from "how many");
- **letters**: no information, names and eight examples with random-letter
  codes on the same subset (separates first-listed bias from numeric-code
  bias).

Outputs under ``<output_dir>/knowledge_dial/<split>/``: one row per answered
request in ``<dataset>.jsonl`` and ``summary.json`` with the analysis of
:mod:`beyond_answer_confidence.analyses.knowledge_dial`.
"""

import asyncio
import logging
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, overload

import pandas as pd

from beyond_answer_confidence.analyses.knowledge_dial import analyse
from beyond_answer_confidence.backends.cache import CachedClient
from beyond_answer_confidence.config import as_dict
from beyond_answer_confidence.data.intents import (
    DATASETS,
    IntentDataset,
    filler_texts,
    load,
)
from beyond_answer_confidence.experiments.base import (
    Experiment,
    backend_for,
    read_jsonl,
    write_json,
)
from beyond_answer_confidence.json_output import dumps
from beyond_answer_confidence.runner import chunks, collect
from beyond_answer_confidence.settings import Settings
from beyond_answer_confidence.tasks.intents import Cell, Planner, dial_row, plan_cells
from beyond_answer_confidence.tasks.schema import Task

logger = logging.getLogger(__name__)

NAME = "knowledge_dial"
CACHE = "intents"
"""Request cache shared by the intent-classification experiments."""


@dataclass(frozen=True)
class Config:
    """Knowledge-dial options (defaults: the standard test-split run).

    Attributes:
        datasets: Datasets to run, in order (the bootstrap generator is
            shared across them in this order).
        split: ``"test"`` or ``"dev"``.
        per_intent: Main-arm items per intent.
        subset_per_intent: Seed- and letter-arm items per intent.
        replicates: Replicates in the main and letter arms.
        extra_seeds: Example-set seeds of the seed arm.
        bootstrap: Bootstrap resamples.
        seed: Bootstrap seed.
        chunk_size: Requests built and sent per chunk (bounds memory).
    """

    datasets: tuple[str, ...] = DATASETS
    split: str = "test"
    per_intent: int = 10
    subset_per_intent: int = 3
    replicates: int = 3
    extra_seeds: tuple[int, ...] = (1, 2)
    bootstrap: int = 2000
    seed: int = 0
    chunk_size: int = 2000


@dataclass
class _Plan:
    """Planner and cells of one dataset."""

    name: str
    planner: Planner
    cells: list[Cell] = field(default_factory=list)


def _plans(settings: Settings, config: Config) -> list[_Plan]:
    """Load the data and list every cell, dataset by dataset."""
    data: dict[str, IntentDataset] = {
        name: load(name, settings.data_dir) for name in config.datasets
    }
    filler = filler_texts(data, settings.data_dir)
    out = []
    for name, ds in data.items():
        planner = Planner(ds, config.split, filler, (0, *config.extra_seeds))
        cells = plan_cells(
            planner.examples,
            per_intent=config.per_intent,
            subset_per_intent=config.subset_per_intent,
            replicates=config.replicates,
            extra_seeds=config.extra_seeds,
        )
        out.append(_Plan(name, planner, cells))
    return out


class PlannedTasks(Sequence[Task]):
    """All tasks of a run, built lazily (a full plan does not fit in memory)."""

    def __init__(self, plans: Sequence[_Plan]) -> None:
        """Index the plans.

        Args:
            plans: Planner and cells per dataset.
        """
        self._flat = [(p, i) for p in plans for i in range(len(p.cells))]

    def __len__(self) -> int:
        """Return the number of tasks."""
        return len(self._flat)

    def _task(self, pos: int) -> Task:
        plan, i = self._flat[pos]
        request = plan.planner.planned(plan.cells[i]).request
        return Task(f"{plan.name}:{i}", request)

    @overload
    def __getitem__(self, index: int) -> Task: ...

    @overload
    def __getitem__(self, index: slice) -> list[Task]: ...

    def __getitem__(self, index: int | slice) -> Task | list[Task]:
        """Build one task or a slice of tasks.

        Args:
            index: Position or slice.

        Returns:
            The task(s).
        """
        if isinstance(index, slice):
            return [self._task(i) for i in range(*index.indices(len(self)))]
        return self._task(range(len(self))[index])

    def __iter__(self) -> Iterator[Task]:
        """Iterate over the tasks, building each on demand."""
        return (self._task(i) for i in range(len(self)))


def requests(settings: Settings, config: Config) -> Sequence[Task]:
    """Every request of the run (built lazily).

    Args:
        settings: Run settings.
        config: Options.

    Returns:
        The tasks.
    """
    return PlannedTasks(_plans(settings, config))


def rows_path(settings: Settings, split: str, dataset: str) -> Path:
    """Where the rows of one dataset and split are written.

    Args:
        settings: Run settings.
        split: Split name.
        dataset: Dataset name.

    Returns:
        ``<output_dir>/knowledge_dial/<split>/<dataset>.jsonl``.
    """
    return settings.out(NAME) / split / f"{dataset}.jsonl"


def _collect_plan(
    plan: _Plan, client: CachedClient, settings: Settings, config: Config, out: Path
) -> dict[str, Any]:
    """Answer one dataset's cells chunk by chunk, streaming rows to ``out``."""
    written, errors, fatal = 0, dict[str, int](), None
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for n, part in enumerate(chunks(range(len(plan.cells)), config.chunk_size)):
            planned = [plan.planner.planned(plan.cells[i]) for i in part]
            tasks = [
                Task(str(i), p.request) for i, p in zip(part, planned, strict=True)
            ]
            got = asyncio.run(
                collect(
                    tasks,
                    client,
                    backend_for(settings),
                    concurrency=settings.concurrency,
                    chunk_size=len(tasks),
                    label=f"{plan.name} part {n + 1}",
                )
            )
            for task, p in zip(tasks, planned, strict=True):
                recs = got.records.get(task.unit)
                if recs:
                    f.write(dumps(dial_row(p, recs[0]), ensure_ascii=False) + "\n")
                    written += 1
            for k, v in got.errors.items():
                errors[k] = errors.get(k, 0) + v
            if got.fatal is not None:
                fatal = got.fatal
                break
    return {
        "requests": len(plan.cells),
        "rows": written,
        "errors": errors,
        "fatal": fatal,
    }


def run(settings: Settings, config: Config) -> dict[str, Any]:
    """Collect, write rows, analyse and write the summary.

    Args:
        settings: Run settings.
        config: Options.

    Returns:
        The summary.
    """
    plans = _plans(settings, config)
    client = CachedClient(
        settings.cache_file(CACHE),
        model=settings.model,
        max_input_tokens=settings.max_input_tokens,
    )
    per_dataset: dict[str, Any] = {}
    for plan in plans:
        path = rows_path(settings, config.split, plan.name)
        per_dataset[plan.name] = _collect_plan(plan, client, settings, config, path)
        logger.info("%s: %s", plan.name, per_dataset[plan.name])
        if per_dataset[plan.name]["fatal"] is not None:
            logger.warning("stopping after a fatal API error")
            break
    bookkeeping = {
        "new_api_calls": client.api_calls,
        "new_input_tokens": client.spent_input_tokens,
    }
    del client  # the cache can take gigabytes; the analysis reads the rows
    rows = {
        name: pd.DataFrame(read_jsonl(rows_path(settings, config.split, name)))
        for name, info in per_dataset.items()
        if info["rows"]
    }
    summary = {
        "config": as_dict(config),
        "datasets": per_dataset,
        **bookkeeping,
        "results": analyse(rows, config.bootstrap, config.seed),
    }
    write_json(settings.out(NAME) / config.split / "summary.json", summary)
    return summary


EXPERIMENT = Experiment(
    name=NAME,
    summary="Uncertainty of intent classification as knowledge about opaque codes grows",
    config_type=Config,
    requests=requests,
    run=run,
)
