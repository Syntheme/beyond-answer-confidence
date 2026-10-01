"""Cache-first, budget-capped request client.

Every answered request is appended to a JSONL file keyed by a hash of the
request, so analyses re-read the cache instead of calling the API again, and
an interrupted run loses nothing. Only ``model``, ``state``, ``questions``,
``replicate`` and the response are written; credentials never pass through
this module.

The cache holds the prompts, which usually contain dataset text: keep it out
of version control.

A cache file has a single writer: run at most one process that appends to a
given file at a time. Each entry is appended with one ``write`` on an
``O_APPEND`` descriptor, so an interrupted run leaves at most one truncated
final line, which the loader skips with a warning (as it does any other
malformed line).

Budget: before each new call, the call's own token estimate plus the
estimates of calls already in flight are reserved against the cap. Spending
can therefore exceed the cap only by estimation error: at most one call's
error (actual minus estimated tokens) per concurrent slot. The estimate
(:func:`~beyond_answer_confidence.backends.base.estimate_tokens`) is deliberately
conservative. Only recorded responses are counted: a call billed by the
provider whose response never arrives (e.g. an SDK retry after a timeout) is
not.
"""

import asyncio
import hashlib
import json
import logging
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from beyond_answer_confidence.backends.base import (
    BudgetExceededError,
    FatalAPIError,
    LiveBackend,
    MalformedResponseError,
    OfflineError,
    Request,
    Response,
    estimate_tokens,
    is_fatal,
    parse_response,
)

logger = logging.getLogger(__name__)

MAX_CONCURRENCY = 64
"""Upper limit for simultaneous API calls."""


def check_concurrency(concurrency: int) -> int:
    """Validate a concurrency setting.

    Args:
        concurrency: Requested simultaneous calls.

    Returns:
        The value, unchanged.

    Raises:
        ValueError: If it is not between 1 and :data:`MAX_CONCURRENCY`.
    """
    if not 1 <= concurrency <= MAX_CONCURRENCY:
        raise ValueError(
            f"concurrency must be between 1 and {MAX_CONCURRENCY}, got {concurrency}"
        )
    return concurrency


def _parse(body: Mapping[str, Any]) -> Response:
    """Parse a body, wrapping any failure in :class:`MalformedResponseError`."""
    try:
        return parse_response(body)
    except (ValueError, KeyError, TypeError, AttributeError) as err:
        raise MalformedResponseError(f"unparsable response: {err!r}") from err


def _billed_input_tokens(body: Mapping[str, Any]) -> int:
    """Input tokens reported in a raw body, read without full parsing."""
    try:
        usage = body.get("usage") or {}
        return int(usage.get("input_tokens") or 0)
    except (AttributeError, TypeError, ValueError):
        logger.warning("response without a readable input-token count")
        return 0


def request_key(
    model: str, state: Any, questions: Mapping[str, Any], replicate: int = 0
) -> str:
    """Hash a request canonically.

    The format is fixed: changing it would orphan every existing cache.

    Args:
        model: Model name.
        state: Request state (JSON-serialisable).
        questions: Question dictionaries (JSON-serialisable).
        replicate: Replicate index.

    Returns:
        A hex SHA-256 digest.
    """
    payload = {
        "model": model,
        "state": state,
        "questions": questions,
        "replicate": replicate,
    }
    blob = json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )
    return hashlib.sha256(blob.encode()).hexdigest()


@dataclass(frozen=True)
class CallRecord:
    """One answered request.

    Attributes:
        key: Cache key.
        response: Parsed response.
        cached: Whether it came from the cache.
        latency_s: Call latency when it was made, if known.
        request_id: Provider request id, if known.
    """

    key: str
    response: Response
    cached: bool
    latency_s: float | None
    request_id: str | None


class CachedClient:
    """Cache-first client with an input-token spend cap."""

    def __init__(
        self,
        cache_path: Path,
        *,
        model: str,
        max_input_tokens: int = 0,
    ) -> None:
        """Load the cache.

        Args:
            cache_path: JSONL cache file.
            model: Model to pin (part of every cache key).
            max_input_tokens: Budget of *new* (uncached) input tokens for this
                client's lifetime; 0 allows no paid calls.
        """
        self._cache_path = cache_path
        self.model = model
        self.max_input_tokens = max_input_tokens
        self.spent_input_tokens = 0
        self.api_calls = 0
        self.malformed_lines = 0
        # Only the response (as compact JSON) and call metadata stay in
        # memory; request bodies stay on disk.
        self._cache: dict[str, tuple[str, float | None, str | None]] = {}
        if cache_path.exists():
            self._load(cache_path)

    def _load(self, path: Path) -> None:
        """Read a cache file, skipping (and counting) malformed lines."""
        with path.open(encoding="utf-8", errors="replace") as f:
            for no, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                    key, body = entry["key"], entry["response"]
                    if not isinstance(key, str) or not isinstance(body, dict):
                        raise TypeError("bad key or response type")
                except (ValueError, KeyError, TypeError) as err:
                    self.malformed_lines += 1
                    tail = "" if line.endswith("\n") else " (truncated final line)"
                    logger.warning(
                        "%s: skipping malformed line %d%s: %s",
                        path,
                        no,
                        tail,
                        type(err).__name__,
                    )
                    continue
                self._remember(
                    key, body, entry.get("latency_s"), entry.get("request_id")
                )
        if self.malformed_lines:
            logger.warning(
                "%s: %d malformed line(s) skipped", path, self.malformed_lines
            )

    def __len__(self) -> int:
        """Return the number of cached responses."""
        return len(self._cache)

    def _remember(
        self,
        key: str,
        body: Mapping[str, Any],
        latency: float | None,
        request_id: str | None,
    ) -> None:
        self._cache[key] = (
            json.dumps(body, separators=(",", ":")),
            latency,
            request_id,
        )

    def _cached(self, key: str) -> CallRecord | None:
        """Return the cached record, ``None`` if absent.

        Raises:
            MalformedResponseError: If the cached body cannot be parsed.
        """
        hit = self._cache.get(key)
        if hit is None:
            return None
        body, latency, request_id = hit
        return CallRecord(key, _parse(json.loads(body)), True, latency, request_id)

    def _check_budget(self, reserved: int) -> None:
        """Refuse a call if spent plus ``reserved`` tokens would exceed the cap.

        Args:
            reserved: Estimated tokens of the new call plus calls in flight.

        Raises:
            BudgetExceededError: If the cap would be exceeded.
        """
        if self.spent_input_tokens + reserved > self.max_input_tokens:
            raise BudgetExceededError(
                f"input-token budget of {self.max_input_tokens} reached "
                f"(spent {self.spent_input_tokens}, reserving ~{reserved})"
            )

    def _append(self, entry: Mapping[str, Any]) -> None:
        """Append one entry with a single write on an ``O_APPEND`` descriptor.

        A truncated final line left by an interrupted writer is terminated
        first, so the new entry always starts on its own line.
        """
        data = (json.dumps(entry, ensure_ascii=False) + "\n").encode("utf-8")
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self._cache_path, os.O_RDWR | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            size = os.fstat(fd).st_size
            if size and os.pread(fd, 1, size - 1) != b"\n":
                data = b"\n" + data
            written = os.write(fd, data)
            if written != len(data):
                raise OSError(f"short write to cache ({written} of {len(data)} bytes)")
        finally:
            os.close(fd)

    def _store(
        self,
        key: str,
        request: Request,
        body: Mapping[str, Any],
        request_id: str | None,
        latency: float,
    ) -> CallRecord:
        """Count, persist and parse one paid response.

        The call is counted and the raw body persisted *before* parsing, so a
        paid response that fails to parse is neither lost nor paid again.

        Raises:
            MalformedResponseError: If the body cannot be parsed (it is
                cached and counted all the same).
        """
        self.api_calls += 1
        self.spent_input_tokens += _billed_input_tokens(body)
        entry = {
            "key": key,
            "model": self.model,
            "replicate": request.replicate,
            "state": request.state,
            "questions": request.questions,
            "response": body,
            "request_id": request_id,
            "latency_s": latency,
            "timestamp": time.time(),
        }
        self._append(entry)
        self._remember(key, body, latency, request_id)
        try:
            response = _parse(body)
        except MalformedResponseError as err:
            logger.warning(
                "response %s cached but could not be parsed: %s", key[:12], err
            )
            raise
        return CallRecord(key, response, False, latency, request_id)

    def key_for(self, request: Request) -> str:
        """Return the cache key of a request under this client's model.

        Args:
            request: The request.

        Returns:
            Its cache key.
        """
        return request_key(
            self.model, request.state, request.questions, request.replicate
        )

    def has(self, request: Request) -> bool:
        """Return whether a request is answered in the cache.

        Args:
            request: The request.

        Returns:
            True if cached.
        """
        return self.key_for(request) in self._cache

    def get(self, request: Request) -> CallRecord:
        """Answer a request from the cache.

        Args:
            request: The request.

        Returns:
            The cached record.

        Raises:
            OfflineError: If the request is not cached.
            MalformedResponseError: If the cached body cannot be parsed.
        """
        hit = self._cached(self.key_for(request))
        if hit is None:
            raise OfflineError("request not in cache")
        return hit

    async def ask_many(
        self,
        requests: Sequence[Request],
        backend: LiveBackend | None,
        *,
        concurrency: int = 8,
    ) -> list[CallRecord | BaseException]:
        """Answer many requests concurrently, cache first.

        Duplicate requests are sent once. Before each new call, its own
        token estimate plus the estimates of calls in flight are reserved
        against the budget, so spending exceeds the cap by at most one call's
        estimation error per concurrent slot (see the module docstring).
        Failures (after the backend's own retries) are returned in place
        instead of aborting the batch; rerunning retries only them. A paid
        response that cannot be parsed is cached, counted and returned as a
        :class:`MalformedResponseError`. After a fatal error (HTTP
        401/402/403) no new calls start.

        Args:
            requests: Requests to answer.
            backend: An open live backend, or ``None`` for cache-only use.
            concurrency: Maximum simultaneous calls (1 to
                :data:`MAX_CONCURRENCY`).

        Returns:
            One record or exception per request, in input order.

        Raises:
            ValueError: On an invalid ``concurrency``.
        """
        check_concurrency(concurrency)
        results: dict[str, CallRecord | BaseException] = {}
        pending: dict[str, Request] = {}
        for req in requests:
            key = self.key_for(req)
            if key in results or key in pending:
                continue
            try:
                hit = self._cached(key)
            except MalformedResponseError as err:
                results[key] = err
                continue
            if hit is not None:
                results[key] = hit
            elif backend is None:
                results[key] = OfflineError("request not in cache")
            else:
                pending[key] = req

        semaphore = asyncio.Semaphore(concurrency)
        in_flight = 0
        fatal: list[BaseException] = []

        async def run(key: str, req: Request, live: LiveBackend) -> None:
            nonlocal in_flight
            estimate = estimate_tokens(req)
            async with semaphore:
                if fatal:
                    results[key] = FatalAPIError(f"skipped after: {fatal[0]!r}")
                    return
                try:
                    self._check_budget(in_flight + estimate)
                except BudgetExceededError as err:
                    results[key] = err
                    return
                in_flight += estimate
                start = time.perf_counter()
                try:
                    raw = await live.answer(req, self.model)
                    results[key] = self._store(
                        key, req, raw.body, raw.request_id, time.perf_counter() - start
                    )
                except Exception as err:  # returned in place, see docstring
                    results[key] = err
                    if is_fatal(err):
                        fatal.append(err)
                finally:
                    in_flight -= estimate

        if backend is not None and pending:
            await asyncio.gather(*(run(k, r, backend) for k, r in pending.items()))
        return [results[self.key_for(req)] for req in requests]
