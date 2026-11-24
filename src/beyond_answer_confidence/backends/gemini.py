"""Google Gemini Decisions API backend (``gemini-decision-1``).

The Decisions API takes one shared input and named ``noul`` / ``choice`` /
``score`` questions in the same shape this package already uses, and returns
a probability per option. Two differences from System One matter:

- every field is text, so the request *state*, and any structured
  instructions or option descriptions, are sent as their JSON serialisation
  (see :func:`to_input` and :func:`to_questions`);
- yes/no probabilities are clipped by the API to [0.01, 0.99].

Responses are converted to the System One JSON shape (see
:func:`beyond_answer_confidence.backends.base.parse_response`) before they are
cached, so every analysis works unchanged. The API reports no model version,
so the requested model name is recorded.

The API key is read from ``GOOGLE_GEMINI_API_KEY`` when the client opens and
handed straight to the SDK; it is never stored on this object, logged or
written to the cache. The SDK and HTTP loggers are pinned to WARNING so wire
logging (which includes the key header) cannot be switched on by a stray
DEBUG setting.

This project is not affiliated with or endorsed by Google.
"""

import json
import logging
import os
from collections.abc import Mapping
from typing import Any

from beyond_answer_confidence.backends.base import RawAnswer, Request

DEFAULT_MODEL = "gemini-decision-1"
"""The Decisions API model (``v1beta`` only)."""

API_KEY_ENV = "GOOGLE_GEMINI_API_KEY"  # pragma: allowlist secret (a variable name)
"""Environment variable holding the Gemini API key."""

API_VERSION = "v1beta"
"""The Decisions API is only served on ``v1beta``."""

MODEL_PREFIX = "gemini-"
"""Model names starting with this prefix are sent to this backend."""

for _name in ("google_genai", "google.genai", "httpx", "httpcore"):
    logging.getLogger(_name).setLevel(logging.WARNING)


class GeminiAPIError(RuntimeError):
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


class GeminiBlockedError(RuntimeError):
    """The API declined to answer (a safety or policy block)."""


def is_gemini_model(model: str) -> bool:
    """Return whether a model name belongs to this backend.

    Args:
        model: Model name.

    Returns:
        Whether it starts with :data:`MODEL_PREFIX`.
    """
    return model.startswith(MODEL_PREFIX)


def to_input(state: Any) -> str:
    """Serialise a request state as the Decisions API text input.

    Strings are sent as they are; anything else as compact, key-order
    preserving JSON with non-ASCII characters kept (fewer tokens than
    escapes).

    Args:
        state: The request state.

    Returns:
        The input text.
    """
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False, separators=(",", ":"))


def _text(value: Any) -> str:
    """Render a question field as text: ``None`` -> ``""``, JSON otherwise."""
    if value is None:
        return ""
    return to_input(value)


def to_questions(questions: Mapping[str, Any]) -> dict[str, Any]:
    """Convert questions to the Decisions API's text-only fields.

    System One accepts any JSON as ``instructions`` and as choice-option
    descriptions (the knowledge dial sends ``{"name", "examples"}`` objects);
    the Decisions API accepts only strings. Non-string values are sent as
    their JSON text (as the state is, see :func:`to_input`) and a missing
    description (``None``) as ``""``, which the API documents as "the option
    id is self-explanatory". Strings, score levels and ``noul`` criteria are
    passed through.

    Args:
        questions: Question dictionaries keyed by name.

    Returns:
        New question dictionaries (the input is not modified).
    """
    out: dict[str, Any] = {}
    for name, q in questions.items():
        conv = dict(q)
        conv["instructions"] = _text(q.get("instructions"))
        crit = q.get("criteria")
        if q.get("type") == "choice" and isinstance(crit, Mapping):
            conv["criteria"] = {opt: _text(desc) for opt, desc in crit.items()}
        elif q.get("type") == "score" and isinstance(crit, list):
            conv["criteria"] = [_text(level) for level in crit]
        out[name] = conv
    return out


def redact(text: str) -> str:
    """Remove the API key from a message, should an SDK error ever echo it.

    Args:
        text: A message.

    Returns:
        The message with every occurrence of the key replaced.
    """
    key = os.environ.get(API_KEY_ENV, "")
    return text.replace(key, "[redacted]") if key else text


def _status_of(err: BaseException) -> int | None:
    """Read an integer HTTP status from an SDK exception, if it has one."""
    for attr in ("status_code", "code", "status"):
        value = getattr(err, attr, None)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def _answer_to_dict(answer: Any) -> dict[str, Any]:
    """Convert one SDK answer to the System One answer shape."""
    data: Mapping[str, Any] = (
        answer.model_dump(mode="json") if hasattr(answer, "model_dump") else answer
    )
    kind = data.get("type")
    out: dict[str, Any] = {"type": kind}
    if kind == "noul":
        out["noul"] = data["noul"]
    elif kind == "choice":
        out["choice"] = data.get("choice")
        out["probabilities"] = dict(data["probabilities"])
    elif kind == "score":
        out["score"] = data["score"]
        out["probabilities"] = dict(data.get("probabilities") or {})
    else:
        raise ValueError(f"unknown answer type {kind!r}")
    if data.get("confidence") is not None:
        out["confidence"] = data["confidence"]
    return out


def to_body(response: Any, model: str) -> dict[str, Any]:
    """Convert a Decisions API response to the System One JSON shape.

    Args:
        response: The SDK response (or its JSON dump).
        model: The requested model name (the API reports none).

    Returns:
        ``{"model", "usage": {"input_tokens", "output_tokens"}, "answers"}``.

    Raises:
        GeminiBlockedError: If the response carries a block reason and no
            answers.
    """
    data: Mapping[str, Any] = (
        response.model_dump(mode="json")
        if hasattr(response, "model_dump")
        else response
    )
    answers = data.get("answers") or {}
    feedback = data.get("prompt_feedback") or {}
    if not answers and feedback.get("block_reason"):
        raise GeminiBlockedError(
            f"blocked: {feedback.get('block_reason')}: "
            f"{feedback.get('block_reason_message') or ''}".rstrip(": ")
        )
    usage = data.get("usage") or {}
    return {
        "model": model,
        "usage": {
            "input_tokens": int(usage.get("total_input_tokens") or 0),
            "output_tokens": int(usage.get("total_output_tokens") or 0),
        },
        "answers": {name: _answer_to_dict(a) for name, a in answers.items()},
    }


class GeminiBackend:
    """Live Gemini Decisions API backend (an async context manager)."""

    def __init__(self, *, timeout_s: float = 90.0, max_retries: int = 4) -> None:
        """Configure the backend; no connection is made until ``async with``.

        Args:
            timeout_s: Per-request timeout.
            max_retries: SDK retries (with backoff, on 408/429/5xx) per
                request.
        """
        self._timeout_s = timeout_s
        self._max_retries = max_retries
        self._client: Any = None

    async def __aenter__(self) -> "GeminiBackend":
        """Open the SDK client with the key from the environment.

        Returns:
            This backend.

        Raises:
            GeminiAPIError: If the key variable is unset or empty (status 401,
                so the run stops at once).
        """
        from google import genai
        from google.genai import types

        if not os.environ.get(API_KEY_ENV):
            raise GeminiAPIError(f"{API_KEY_ENV} is not set", 401)
        client = genai.Client(
            api_key=os.environ[API_KEY_ENV],
            http_options=types.HttpOptions(
                api_version=API_VERSION,
                timeout=int(self._timeout_s * 1000),
                retry_options=types.HttpRetryOptions(attempts=self._max_retries + 1),
            ),
        )
        self._client = client.aio
        return self

    async def __aexit__(self, *exc: object) -> None:
        """Close the SDK client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def answer(self, request: Request, model: str) -> RawAnswer:
        """Answer one request.

        Args:
            request: The request.
            model: Model name.

        Returns:
            The response in the System One shape (no request id).

        Raises:
            RuntimeError: If called outside ``async with``.
            GeminiAPIError: On an API or connection error (with its status).
            GeminiBlockedError: If the API declined to answer.
        """
        if self._client is None:
            raise RuntimeError("GeminiBackend used outside 'async with'")
        try:
            response = await self._client.decisions.create(
                model=model,
                input=to_input(request.state),
                questions=to_questions(request.questions),
            )
        except Exception as err:  # SDK errors -> integer status for is_fatal
            raise GeminiAPIError(
                redact(f"{type(err).__name__}: {err}"), _status_of(err)
            ) from None
        return RawAnswer(to_body(response, model), None)
