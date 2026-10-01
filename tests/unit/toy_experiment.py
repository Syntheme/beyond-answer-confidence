"""A tiny experiment and analysis for command-line and registry tests."""

from dataclasses import dataclass
from typing import Any

from beyond_answer_confidence.backends.fake import FakeBackend
from beyond_answer_confidence.experiments.base import (
    Analysis,
    Experiment,
    write_json,
    write_jsonl,
)
from beyond_answer_confidence.runner import run_items
from beyond_answer_confidence.settings import Settings
from beyond_answer_confidence.tasks.schema import Item, Task, item_tasks

NAME = "toy_experiment"
CACHE = "toy_cache"


@dataclass(frozen=True)
class Config:
    items: int = 4
    replicates: int = 2


def build_items(config: Config) -> list[Item]:
    return [
        Item(
            unit=f"toy:{i}",
            state={"case": i},
            options=("A", "B"),
            gold="AB"[i % 2],
            info={"set": "toy"},
        )
        for i in range(config.items)
    ]


def requests(settings: Settings, config: Config) -> list[Task]:
    return item_tasks(build_items(config), config.replicates)


def run(settings: Settings, config: Config) -> dict[str, Any]:
    rows, info = run_items(
        build_items(config),
        cache=settings.cache_file(CACHE),
        model=settings.model,
        replicates=config.replicates,
        backend=FakeBackend if settings.live else None,
        max_input_tokens=settings.max_input_tokens,
    )
    write_jsonl(settings.out(NAME) / "rows.jsonl", rows)
    summary = {"config": {"items": config.items}, **info}
    write_json(settings.out(NAME) / "summary.json", summary)
    return summary


def analyse(settings: Settings, config: Config) -> dict[str, Any]:
    summary = {"items": config.items}
    write_json(settings.out("toy_analysis") / "summary.json", summary)
    return summary


EXPERIMENT = Experiment(NAME, "A toy experiment", Config, requests, run)
ANALYSIS = Analysis("toy_analysis", "A toy analysis", Config, analyse)
NOT_REGISTRABLE = object()
