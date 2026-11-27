"""OpenRouter Decisions API backend (many providers' decision models).

OpenRouter serves decision models from several providers behind one
endpoint (``POST https://openrouter.ai/api/alpha/decisions``) whose request
and response use the same ``noul`` / ``choice`` / ``score`` questions as
System One. Model names are OpenRouter ids such as ``typesafe/jev-1.13``,
``perplexity/pplx-decider-v1.3:free`` or ``baseten/groundhog-1``; any model
name containing ``/`` is sent here. A suffix ``@<provider>`` (e.g.
``cloudflare/clef-flash@deepinfra``) pins the request to that provider with
no fallback; because the suffix is part of the model name, it is also part of
every cache key, so answers from different providers never mix.

- The request *state* is sent as its JSON text (the endpoint takes text or
  media parts); questions are sent as they are.
- Answers already have the System One shape. A refused question is stored as
  ``{"type": "refusal", "question_type", "options"}`` and scored by the
  refusal mode (see :func:`beyond_answer_confidence.backends.base.set_refusal_mode`).
- The response's ``model`` (e.g. a dated version), ``provider`` and billed
  ``cost`` are kept in the cached body.
- Rate limits (HTTP 429, common for ``:free`` models) and server errors are
  retried with backoff, honouring ``Retry-After``; HTTP 401/402/403 stop the
  run.

The key is read from ``OPENROUTER_API_KEY`` when the client opens and kept
only inside the HTTP client's default headers; it is never logged or cached.
The ``httpx`` logger is pinned to WARNING so wire logging cannot expose it.

This project is not affiliated with or endorsed by OpenRouter or any model
provider.
"""

import asyncio
import json
import logging
import os
from collections.abc import Mapping
from typing import Any

from beyond_answer_confidence.backends.base import RawAnswer, Request

ENDPOINT = "https://openrouter.ai/api/alpha/decisions"
"""Decisions endpoint."""

API_KEY_ENV = "OPENROUTER_API_KEY"  # pragma: allowlist secret (a variable name)
"""Environment variable holding the OpenRouter API key."""

RETRY_STATUSES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
"""HTTP statuses retried with backoff."""

for _name in ("httpx", "httpcore"):
    logging.getLogger(_name).setLevel(logging.WARNING)


class OpenRouterAPIError(RuntimeError):
    """An API error with its integer HTTP status (for :func:`is_fatal`).

    Attributes:
        status: HTTP status, or ``None`` for connection errors.
    """

    def __init__(self, message: str, status: int | None) -> None:
        """Create the error.

        Args:
            message: Error description (no credentials).
            status: HTTP status, if any.
        """
        super().__init__(message)
        self.status = status


def is_openrouter_model(model: str) -> bool:
    """Return whether a model name is an OpenRouter id.

    Args:
        model: Model name.

    Returns:
        Whether it contains ``/`` (``provider/model``).
    """
    return "/" in model


def split_provider(model: str) -> tuple[str, str | None]:
    """Split an optional ``@<provider>`` suffix off a model name.

    Args:
        model: Model name, e.g. ``cloudflare/clef-flash@deepinfra``.

    Returns:
        ``(OpenRouter model id, provider or None)``.
    """
    base, sep, provider = model.partition("@")
    return base, (provider or None) if sep else None


def payload_for(request: Request, model: str) -> dict[str, Any]:
    """The request body for one request.

    Args:
        request: The request.
        model: Model name (with an optional ``@<provider>`` suffix).

    Returns:
        ``model``, ``state`` and ``questions``, plus ``provider`` routing
        (that provider only, no fallbacks) when the name pins one.
    """
    base, provider = split_provider(model)
    body: dict[str, Any] = {
        "model": base,
        "state": to_state(request.state),
        "questions": dict(request.questions),
    }
    if provider:
        body["provider"] = {"order": [provider], "allow_fallbacks": False}
    return body


def to_state(state: Any) -> str:
    """Serialise a request state as the endpoint's text state.

    Args:
        state: The request state.

    Returns:
        Strings as they are; anything else as compact JSON text.
    """
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False, separators=(",", ":"))


def _refusal(question: Mapping[str, Any]) -> dict[str, Any]:
    crit = question.get("criteria")
    options = list(crit) if isinstance(crit, Mapping | list) else []
    return {
        "type": "refusal",
        "question_type": question.get("type"),
        "options": [str(o) for o in options],
    }


def to_body(data: Mapping[str, Any], questions: Mapping[str, Any]) -> dict[str, Any]:
    """Convert an endpoint response to the cached System One shape.

    Args:
        data: Decoded response JSON.
        questions: The request's questions (for refusal options).

    Returns:
        ``{"model", "provider", "usage": {"input_tokens", "output_tokens",
        "cost"}, "answers"}``.
    """
    answers: dict[str, Any] = {}
    for name, ans in (data.get("answers") or {}).items():
        answers[name] = (
            _refusal(questions.get(name, {}))
            if ans.get("type") == "refusal"
            else dict(ans)
        )
    usage = data.get("usage") or {}
    return {
        "model": str(data.get("model", "")),
        "provider": data.get("provider"),
        "usage": {
            "input_tokens": int(usage.get("input_tokens") or 0),
            "output_tokens": int(usage.get("output_tokens") or 0),
            "cost": usage.get("cost"),
        },
        "answers": answers,
    }


def redact(text: str) -> str:
    """Remove the API key from a message, should an error ever echo it.

    Args:
        text: A message.

    Returns:
        The message with every occurrence of the key replaced.
    """
    key = os.environ.get(API_KEY_ENV, "")
    return text.replace(key, "[redacted]") if key else text


def _retry_after(headers: Mapping[str, str], attempt: int) -> float:
    try:
        return min(float(headers.get("retry-after", "")), 60.0)
    except ValueError:
        return min(2.0**attempt, 60.0)


class OpenRouterBackend:
    """Live OpenRouter Decisions backend (an async context manager)."""

    def __init__(self, *, timeout_s: float = 90.0, max_retries: int = 6) -> None:
        """Configure the backend; no connection is made until ``async with``.

        Args:
            timeout_s: Per-request timeout.
            max_retries: Retries on :data:`RETRY_STATUSES` and connection
                errors.
        """
        self._timeout_s = timeout_s
        self._max_retries = max_retries
        self._client: Any = None

    async def __aenter__(self) -> "OpenRouterBackend":
        """Open the HTTP client with the key from the environment.

        Returns:
            This backend.

        Raises:
            OpenRouterAPIError: If the key variable is unset or empty (status
                401, so the run stops at once).
        """
        import httpx

        if not os.environ.get(API_KEY_ENV):
            raise OpenRouterAPIError(f"{API_KEY_ENV} is not set", 401)
        self._client = httpx.AsyncClient(
            timeout=self._timeout_s,
            headers={
                "Authorization": f"Bearer {os.environ[API_KEY_ENV]}",
                "Content-Type": "application/json",
            },
        )
        return self

    async def __aexit__(self, *exc: object) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def answer(self, request: Request, model: str) -> RawAnswer:
        """Answer one request.

        Args:
            request: The request.
            model: OpenRouter model id.

        Returns:
            The response in the System One shape and its generation id.

        Raises:
            RuntimeError: If called outside ``async with``.
            OpenRouterAPIError: On an API error after retries (with status).
        """
        import httpx

        if self._client is None:
            raise RuntimeError("OpenRouterBackend used outside 'async with'")
        payload = payload_for(request, model)
        last: OpenRouterAPIError | None = None
        for attempt in range(self._max_retries + 1):
            try:
                resp = await self._client.post(ENDPOINT, json=payload)
            except httpx.HTTPError as err:
                last = OpenRouterAPIError(redact(f"{type(err).__name__}: {err}"), None)
                await asyncio.sleep(min(2.0**attempt, 60.0))
                continue
            if resp.status_code == 200:
                data = resp.json()
                gen = data.get("id")
                return RawAnswer(
                    to_body(data, request.questions),
                    gen if isinstance(gen, str) else None,
                )
            last = OpenRouterAPIError(
                redact(f"HTTP {resp.status_code}: {resp.text[:500]}"), resp.status_code
            )
            if resp.status_code not in RETRY_STATUSES:
                break
            await asyncio.sleep(_retry_after(resp.headers, attempt))
        # The loop runs at least once and always sets ``last`` before breaking.
        assert last is not None  # noqa: S101  # nosec B101
        raise last
