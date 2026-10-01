"""Answer tasks through the cache, chunk by chunk.

Responses are appended to the cache as they arrive, so an interrupted run
loses nothing: rerunning sends only the requests still missing. After a
fatal API error (HTTP 401/402/403, e.g. out of credits) the run stops
issuing calls and reports it.
"""

import asyncio
import logging
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from beyond_answer_confidence.backends.base import LiveBackend, is_fatal
from beyond_answer_confidence.backends.cache import (
    CachedClient,
    CallRecord,
    check_concurrency,
)
from beyond_answer_confidence.tasks.schema import (
    Item,
    Task,
    item_tasks,
    order_collisions,
)
from beyond_answer_confidence.tasks.scoring import score_items

logger = logging.getLogger(__name__)

BackendFactory = Callable[[], LiveBackend]
"""Creates a live backend; ``None`` in its place means offline."""


@dataclass
class Collected:
    """Answered tasks grouped by unit, plus run bookkeeping."""

    records: dict[str, list[CallRecord]] = field(
        default_factory=lambda: defaultdict(list)
    )
    errors: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    fatal: str | None = None

    def complete(self, replicates: Mapping[str, int]) -> dict[str, list[CallRecord]]:
        """Return the units whose every replicate was answered.

        Args:
            replicates: Expected replicate count per unit.

        Returns:
            Unit -> records, for complete units only.
        """
        return {
            u: recs
            for u, recs in self.records.items()
            if len(recs) == replicates.get(u, -1)
        }


def chunks[T](items: Sequence[T], size: int) -> list[Sequence[T]]:
    """Split a sequence into consecutive chunks.

    Args:
        items: Items to split.
        size: Chunk size.

    Returns:
        The chunks.
    """
    return [items[i : i + size] for i in range(0, len(items), size)]


async def collect(
    tasks: Sequence[Task],
    client: CachedClient,
    backend: BackendFactory | None = None,
    *,
    concurrency: int = 8,
    chunk_size: int = 2000,
    label: str = "",
) -> Collected:
    """Answer every task, cache first, chunk by chunk.

    Args:
        tasks: Tasks to answer.
        client: Cached client (holds the token budget).
        backend: Factory for a live backend, or ``None`` to answer only from
            the cache.
        concurrency: Maximum simultaneous API calls.
        chunk_size: Requests per chunk (bounds memory and progress granularity).
        label: Prefix for progress messages.

    Returns:
        Records per unit, error counts, and the fatal error (if any).

    Raises:
        ValueError: If two tasks differ only in key order (they would share a
            cache entry), or on an invalid ``concurrency``.
    """
    check_concurrency(concurrency)
    if (bad := order_collisions(tasks, client.model)) > 0:
        raise ValueError(
            f"{bad} tasks differ only in key order and would share cache entries"
        )
    out = Collected()
    live = backend() if backend is not None else None
    if live is not None:
        await live.__aenter__()
    try:
        parts = chunks(tasks, chunk_size)
        for n, chunk in enumerate(parts):
            results = await client.ask_many(
                [t.request for t in chunk], live, concurrency=concurrency
            )
            for task, rec in zip(chunk, results, strict=True):
                if isinstance(rec, CallRecord):
                    out.records[task.unit].append(rec)
                else:
                    out.errors[type(rec).__name__] += 1
                    if is_fatal(rec) and out.fatal is None:
                        out.fatal = repr(rec)[:300]
            logger.info(
                "%s chunk %d/%d: %s new tokens, %d new calls, errors %s",
                label,
                n + 1,
                len(parts),
                f"{client.spent_input_tokens:,}",
                client.api_calls,
                dict(out.errors),
            )
            if out.fatal is not None:
                logger.warning("%s: stopping after fatal API error", label)
                break
    finally:
        if live is not None:
            await live.__aexit__(None, None, None)
    return out


def run_collect(tasks: Sequence[Task], client: CachedClient, **kw: Any) -> Collected:
    """Synchronous wrapper around :func:`collect`.

    Args:
        tasks: Tasks to answer.
        client: Cached client.
        **kw: Passed to :func:`collect`.

    Returns:
        The collected records.
    """
    return asyncio.run(collect(tasks, client, **kw))


def run_summary(client: CachedClient, collected: Collected) -> dict[str, Any]:
    """Bookkeeping fields for a run's summary file.

    Args:
        client: The client used.
        collected: The run's results.

    Returns:
        Calls, tokens, errors, fatal error and the models seen.
    """
    models = sorted(
        {r.response.model for recs in collected.records.values() for r in recs}
    )
    return {
        "new_api_calls": client.api_calls,
        "new_input_tokens": client.spent_input_tokens,
        "errors": dict(collected.errors),
        "fatal": collected.fatal,
        "models": models,
    }


def run_items(
    items: Sequence[Item],
    *,
    cache: Path,
    model: str,
    replicates: int,
    backend: BackendFactory | None = None,
    max_input_tokens: int = 0,
    concurrency: int = 8,
    label: str = "",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Collect and score a list of items; the common body of most experiments.

    Args:
        items: Items.
        cache: Cache file for this experiment.
        model: Model name.
        replicates: Replicates per item.
        backend: Live backend factory, or ``None`` for offline.
        max_input_tokens: Budget of new input tokens.
        concurrency: Maximum simultaneous API calls.
        label: Progress label.

    Returns:
        ``(rows, bookkeeping)``; rows only for complete items.
    """
    client = CachedClient(cache, model=model, max_input_tokens=max_input_tokens)
    tasks = item_tasks(items, replicates)
    collected = run_collect(
        tasks, client, backend=backend, concurrency=concurrency, label=label
    )
    complete = collected.complete(dict.fromkeys((it.unit for it in items), replicates))
    rows = score_items(items, complete)
    return rows, {
        "items": len(items),
        "complete_items": len(rows),
        **run_summary(client, collected),
    }
