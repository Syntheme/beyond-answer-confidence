"""Check that every request of an experiment rebuilds exactly.

Rebuilding the tasks with the experiment's own builders and looking up each
request's cache key shows whether a cache still answers the whole
experiment: a missing key means the rebuilt request differs from the one
that was sent (or was never sent). The same lookup gives the number of new
calls and an input-token estimate for a live run. No API calls are made.

Pure functions (:func:`coverage`, :func:`estimate`) work on a set of cache
keys; :func:`read_cache_keys` and :func:`verify` do the file I/O. Each cache
file is read once, however many experiments share it.
"""

import json
import logging
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from beyond_answer_confidence.backends.base import estimate_tokens
from beyond_answer_confidence.backends.cache import request_key
from beyond_answer_confidence.tasks.schema import Task

logger = logging.getLogger(__name__)

_KEY_PREFIX = '{"key": "'
_KEY_LEN = 64


@dataclass(frozen=True)
class Coverage:
    """How many rebuilt requests a cache answers.

    Attributes:
        requests: Rebuilt requests.
        found: Requests whose key is in the cache.
        missing: Requests whose key is not.
        missing_examples: Units of the first few missing requests.
    """

    requests: int
    found: int
    missing: int
    missing_examples: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Estimate:
    """New calls a live run would make.

    Attributes:
        requests: Rebuilt requests.
        cached: Distinct requests already in the cache.
        new_requests: Distinct requests not in the cache.
        estimated_input_tokens: Conservative input-token estimate for them
            (see :func:`~beyond_answer_confidence.backends.base.estimate_tokens`).
    """

    requests: int
    cached: int
    new_requests: int
    estimated_input_tokens: int


@dataclass(frozen=True)
class Target:
    """One experiment to check.

    Attributes:
        name: Experiment name.
        cache: Its cache file.
        tasks: Rebuilds its tasks (called once, lazily).
    """

    name: str
    cache: Path
    tasks: Callable[[], Iterable[Task]]


def read_cache_keys(path: Path) -> set[str]:
    """Read the request keys of a JSONL cache (responses are not parsed).

    Malformed lines (e.g. a final line truncated by an interrupted run) are
    skipped with a warning that names the line number, never its content.

    Args:
        path: Cache file; a missing file has no keys.

    Returns:
        The keys.
    """
    keys: set[str] = set()
    if not path.exists():
        logger.warning("no cache at %s", path)
        return keys
    malformed = 0
    with path.open(encoding="utf-8", errors="replace") as f:
        for no, line in enumerate(f, 1):
            if line.startswith(_KEY_PREFIX):
                end = len(_KEY_PREFIX) + _KEY_LEN
                if line[end : end + 1] == '"' and line.endswith("\n"):
                    keys.add(line[len(_KEY_PREFIX) : end])
                    continue
            if not line.strip():
                continue
            try:
                key = json.loads(line)["key"]
                if not isinstance(key, str):
                    raise TypeError("key is not a string")
            except (ValueError, KeyError, TypeError) as err:
                malformed += 1
                tail = "" if line.endswith("\n") else " (truncated final line)"
                logger.warning(
                    "%s: skipping malformed line %d%s: %s",
                    path,
                    no,
                    tail,
                    type(err).__name__,
                )
                continue
            keys.add(key)
    if malformed:
        logger.warning("%s: %d malformed line(s) skipped", path, malformed)
    return keys


def task_key(task: Task, model: str) -> str:
    """Return the cache key of a task.

    Args:
        task: The task.
        model: Model name (part of the key).

    Returns:
        The key.
    """
    r = task.request
    return request_key(model, r.state, r.questions, r.replicate)


def coverage(
    tasks: Iterable[Task], model: str, keys: set[str], examples: int = 5
) -> Coverage:
    """Count rebuilt requests found in a set of cache keys.

    Args:
        tasks: Rebuilt tasks.
        model: Model name.
        keys: Cache keys.
        examples: Missing units to report.

    Returns:
        The counts.
    """
    n = 0
    missing: list[str] = []
    n_missing = 0
    for t in tasks:
        n += 1
        if task_key(t, model) not in keys:
            n_missing += 1
            if len(missing) < examples:
                missing.append(t.unit)
    return Coverage(n, n - n_missing, n_missing, missing)


def estimate(tasks: Iterable[Task], model: str, keys: set[str]) -> Estimate:
    """Count the distinct requests a live run would send and their tokens.

    Args:
        tasks: Rebuilt tasks.
        model: Model name.
        keys: Cache keys.

    Returns:
        The estimate.
    """
    n = 0
    seen: set[str] = set()
    cached = new = tokens = 0
    for t in tasks:
        n += 1
        k = task_key(t, model)
        if k in seen:
            continue
        seen.add(k)
        if k in keys:
            cached += 1
        else:
            new += 1
            tokens += estimate_tokens(t.request)
    return Estimate(n, cached, new, tokens)


def _grouped(targets: Sequence[Target]) -> list[tuple[Path, list[Target]]]:
    """Group targets by cache file, in order of first appearance."""
    groups: dict[Path, list[Target]] = {}
    for t in targets:
        groups.setdefault(t.cache, []).append(t)
    return list(groups.items())


def run_checks[R](
    targets: Sequence[Target],
    check: Callable[[Iterable[Task], set[str]], R],
    load_keys: Callable[[Path], set[str]] = read_cache_keys,
    *,
    catch: bool = False,
) -> dict[str, R | str]:
    """Apply a check to every target, reading each cache file once.

    Targets sharing a cache are checked together, and each key set is
    released before the next file is read.

    Args:
        targets: Experiments to check.
        check: ``(tasks, keys) -> result``.
        load_keys: Reads the keys of a cache file.
        catch: Record a target whose tasks cannot be rebuilt (e.g. missing
            data or upstream outputs) as an error message and go on, instead
            of raising.

    Returns:
        Result (or error message) per target name, in the order given.
    """
    results: dict[str, R | str] = {}
    for cache, group in _grouped(targets):
        logger.info("reading cache %s", cache.name)
        keys = load_keys(cache)
        for t in group:
            try:
                results[t.name] = check(t.tasks(), keys)
            except Exception as err:
                if not catch:
                    raise
                results[t.name] = f"{type(err).__name__}: {err}"
                logger.error("%s: cannot rebuild requests: %s", t.name, err)
                continue
            logger.info("%s: %s", t.name, results[t.name])
        del keys
    return {t.name: results[t.name] for t in targets}


def verify(
    targets: Sequence[Target],
    model: str,
    load_keys: Callable[[Path], set[str]] = read_cache_keys,
) -> dict[str, Any]:
    """Check cache coverage of every target.

    Args:
        targets: Experiments to check.
        model: Model name.
        load_keys: Reads the keys of a cache file.

    Returns:
        ``{"model", "experiments": {name: coverage or {"error": message}},
        "requests", "missing", "errors", "ok"}``.
    """
    per = run_checks(
        targets, lambda ts, keys: coverage(ts, model, keys), load_keys, catch=True
    )
    done = {n: c for n, c in per.items() if isinstance(c, Coverage)}
    errors = sorted(n for n, c in per.items() if not isinstance(c, Coverage))
    missing = sum(c.missing for c in done.values())
    return {
        "model": model,
        "experiments": {
            n: asdict(c) if isinstance(c, Coverage) else {"error": c}
            for n, c in per.items()
        },
        "requests": sum(c.requests for c in done.values()),
        "missing": missing,
        "errors": errors,
        "ok": missing == 0 and not errors,
    }


def estimate_all(
    targets: Sequence[Target],
    model: str,
    load_keys: Callable[[Path], set[str]] = read_cache_keys,
) -> dict[str, Any]:
    """Estimate the new calls and input tokens of every target.

    Args:
        targets: Experiments to estimate.
        model: Model name.
        load_keys: Reads the keys of a cache file.

    Returns:
        ``{"experiments": {name: estimate}, "new_requests",
        "estimated_input_tokens"}``.
    """
    raw = run_checks(targets, lambda ts, keys: estimate(ts, model, keys), load_keys)
    per = {n: e for n, e in raw.items() if isinstance(e, Estimate)}
    return {
        "model": model,
        "experiments": {name: asdict(e) for name, e in per.items()},
        "new_requests": sum(e.new_requests for e in per.values()),
        "estimated_input_tokens": sum(e.estimated_input_tokens for e in per.values()),
    }
