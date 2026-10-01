"""The interface every experiment and offline analysis implements.

An :class:`Experiment` can rebuild all of its requests (for ``verify`` and
cost estimates) and run end to end: collect answers through the cache,
score, analyse and write ``rows.jsonl`` and ``summary.json`` under
``<output_dir>/<name>/``. An :class:`Analysis` reads rows that experiments
wrote and makes no requests.
"""

import json
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from beyond_answer_confidence.backends.jev import JevBackend
from beyond_answer_confidence.json_output import dumps
from beyond_answer_confidence.runner import BackendFactory
from beyond_answer_confidence.settings import Settings
from beyond_answer_confidence.tasks.schema import Task

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Experiment:
    """A registered experiment.

    Attributes:
        name: Command-line name; also the cache and output directory name.
        summary: One-line description.
        config_type: Frozen dataclass of the experiment's options.
        requests: Rebuild every task the experiment sends.
        run: Run it (collect, score, analyse, write); returns the summary.
    """

    name: str
    summary: str
    config_type: type
    requests: Callable[[Settings, Any], Sequence[Task]]
    run: Callable[[Settings, Any], Mapping[str, Any]]


@dataclass(frozen=True)
class Analysis:
    """A registered offline analysis (no requests).

    Attributes:
        name: Command-line name; also the output directory name.
        summary: One-line description.
        config_type: Frozen dataclass of the analysis's options.
        run: Run it; returns the summary.
    """

    name: str
    summary: str
    config_type: type
    run: Callable[[Settings, Any], Mapping[str, Any]]


def backend_for(settings: Settings) -> BackendFactory | None:
    """Return the live backend factory for a run, or ``None`` offline.

    Args:
        settings: Run settings.

    Returns:
        A factory when ``settings.live`` is set, else ``None``.
    """
    return JevBackend if settings.live else None


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write rows as JSON lines (overwrites; rows are derived from the cache).

    Non-finite floats are written as ``null`` (see
    :mod:`beyond_answer_confidence.json_output`).

    Args:
        path: Output file.
        rows: Rows.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.writelines(dumps(r, ensure_ascii=False) + "\n" for r in rows)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSON lines.

    Args:
        path: Input file.

    Returns:
        The rows.
    """
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path: Path, obj: Any) -> None:
    """Write a JSON document (indented, UTF-8; non-finite floats as ``null``).

    Args:
        path: Output file.
        obj: JSON-serialisable object.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps(obj, indent=2, ensure_ascii=False) + "\n", "utf-8")
    logger.info("wrote %s", path)


def source_rows(settings: Settings, name: str) -> Path:
    """Rows file written by an earlier experiment, checked to exist.

    Args:
        settings: Run settings.
        name: Experiment name.

    Returns:
        ``<output_dir>/<name>/rows.jsonl``.

    Raises:
        FileNotFoundError: If the experiment has not been run.
    """
    path = settings.out(name) / "rows.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run {name} first")
    return path
