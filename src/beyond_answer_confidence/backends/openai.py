"""OpenAI Decisions API backend (``gpt-6-luna``).

The Decisions API answers predicate (yes/no), choice and score questions about
one shared input and returns a probability per option. Requests in this
package's format are translated as follows:

- the request *state* is sent as one user message holding an ``input_json``
  part, so the model receives JSON as System One does (strings are sent as an
  ``input_text`` part);
- questions become a list, each carrying its name; ``noul`` becomes
  ``predicate``; choice criteria ``{option: description}`` become ``choices``
  with ``value`` and ``description``; score criteria become ``levels``
  (``label``, at most 10);
- instructions and option descriptions must be text, so structured ones are
  sent as their JSON text, and a missing description is omitted.

Responses are converted back to the System One JSON shape (see
:func:`beyond_answer_confidence.backends.base.parse_response`) before they are
cached, so every analysis works unchanged. The response's ``model`` field
records the model that answered.

The API key is read by the OpenAI SDK itself from ``OPENAI_API_KEY`` and never
passes through this package. The SDK and HTTP loggers are pinned to WARNING so
wire logging (which includes the key header) cannot be switched on by a stray
DEBUG setting.

This project is not affiliated with or endorsed by OpenAI.
"""

import json
import logging
import os
from collections.abc import Mapping
from typing import Any

from beyond_answer_confidence.backends.base import RawAnswer, Request

DEFAULT_MODEL = "gpt-6-luna"
"""The model the Decisions API serves."""

API_KEY_ENV = "OPENAI_API_KEY"  # pragma: allowlist secret (a variable name)
"""Environment variable the OpenAI SDK reads the key from (only its presence
is checked here; the value is never read into this package)."""

MODEL_PREFIXES = ("gpt-", "openai-")
"""Model names starting with one of these prefixes are sent to this backend."""

MAX_SCORE_LEVELS = 10
"""The API accepts 2 to 10 levels per score question."""

QUOTA_CODES = frozenset({"insufficient_quota", "billing_hard_limit_reached"})
"""Error codes OpenAI returns with HTTP 429 when the account has no credit;
they are mapped to status 402 so the run stops (see :func:`is_fatal`)."""

for _name in ("openai", "httpx", "httpcore"):
    logging.getLogger(_name).setLevel(logging.WARNING)


class OpenAIAPIError(RuntimeError):
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


def is_openai_model(model: str) -> bool:
    """Return whether a model name belongs to this backend.

    Args:
        model: Model name.

    Returns:
        Whether it starts with one of :data:`MODEL_PREFIXES`.
    """
    return model.startswith(MODEL_PREFIXES)


def _text(value: Any) -> str:
    """Render a field as text: strings as they are, anything else as JSON."""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def to_input(state: Any) -> list[dict[str, Any]]:
    """Convert a request state to the Decisions API input.

    Args:
        state: The request state.

    Returns:
        One user message with an ``input_json`` part (``input_text`` for a
        string state).
    """
    part: dict[str, Any] = (
        {"type": "input_text", "text": state}
        if isinstance(state, str)
        else {"type": "input_json", "data": state}
    )
    return [{"type": "message", "role": "user", "content": [part]}]


def to_questions(questions: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Convert questions to the Decisions API's named question list.

    Args:
        questions: Question dictionaries keyed by name (System One shape).

    Returns:
        Questions in the Decisions API shape, in the same order.

    Raises:
        ValueError: On an unknown question type or a score question with
            more than :data:`MAX_SCORE_LEVELS` levels.
    """
    out: list[dict[str, Any]] = []
    for name, q in questions.items():
        kind = q.get("type")
        conv: dict[str, Any] = {
            "name": name,
            "instructions": _text(q.get("instructions", "")),
        }
        if kind == "noul":
            conv["type"] = "predicate"
        elif kind == "choice":
            conv["type"] = "choice"
            conv["choices"] = [
                {"value": opt}
                if desc is None
                else {"value": opt, "description": _text(desc)}
                for opt, desc in q["criteria"].items()
            ]
        elif kind == "score":
            levels = list(q["criteria"])
            if len(levels) > MAX_SCORE_LEVELS:
                raise ValueError(
                    f"score question {name!r} has {len(levels)} levels; "
                    f"the API accepts at most {MAX_SCORE_LEVELS}"
                )
            conv["type"] = "score"
            conv["levels"] = [{"label": _text(level)} for level in levels]
        else:
            raise ValueError(f"unknown question type {kind!r} for {name!r}")
        out.append(conv)
    return out


def _answer_to_dict(data: Mapping[str, Any]) -> dict[str, Any]:
    """Convert one Decisions API answer to the System One answer shape."""
    kind = data.get("type")
    if kind == "predicate":
        return {"type": "noul", "noul": data["probability"]}
    if kind == "choice":
        out: dict[str, Any] = {
            "type": "choice",
            "choice": data.get("choice"),
            "probabilities": {
                str(p["value"]): p["probability"] for p in data["probabilities"]
            },
        }
    elif kind == "score":
        out = {
            "type": "score",
            "score": data["score"],
            "probabilities": {
                str(p["value"]): p["probability"] for p in data["probabilities"]
            },
        }
    else:
        raise ValueError(f"unknown answer type {kind!r}")
    if data.get("confidence") is not None:
        out["confidence"] = data["confidence"]
    return out


def _refusal(question: Mapping[str, Any]) -> dict[str, Any]:
    """Store a refused question with what a no-preference reading needs."""
    crit = question.get("criteria")
    options = list(crit) if isinstance(crit, (Mapping, list)) else []
    return {
        "type": "refusal",
        "question_type": question.get("type"),
        "options": [str(o) for o in options],
    }


def to_body(response: Any, questions: Mapping[str, Any]) -> dict[str, Any]:
    """Convert a Decisions API response to the System One JSON shape.

    Answers come back in question order; an unnamed answer is matched to its
    question by position. A refused question is kept as a ``refusal`` answer
    (with the question's type and options), so the paid response is cached
    once and scored by the refusal mode (see
    :func:`beyond_answer_confidence.backends.base.set_refusal_mode`).

    Args:
        response: The SDK response (or its JSON dump).
        questions: The request's questions, keyed by name.

    Returns:
        ``{"model", "usage": {"input_tokens", "output_tokens"}, "answers"}``.
    """
    data: Mapping[str, Any] = (
        response.model_dump(mode="json")
        if hasattr(response, "model_dump")
        else response
    )
    names = list(questions)
    answers: dict[str, Any] = {}
    for i, ans in enumerate(data.get("answers") or []):
        name = ans.get("name") or (names[i] if i < len(names) else str(i))
        if ans.get("type") == "refusal":
            answers[name] = _refusal(questions.get(name, {}))
        else:
            answers[name] = _answer_to_dict(ans)
    usage = data.get("usage") or {}
    return {
        "model": str(data.get("model", "")),
        "usage": {
            "input_tokens": int(usage.get("input_tokens") or 0),
            "output_tokens": int(usage.get("output_tokens") or 0),
        },
        "answers": answers,
    }


def _status_of(err: BaseException) -> int | None:
    """Read an integer HTTP status from an SDK error; no credit becomes 402."""
    status = getattr(err, "status_code", None)
    if not isinstance(status, int) or isinstance(status, bool):
        return None
    if status == 429 and getattr(err, "code", None) in QUOTA_CODES:
        return 402
    return status


def redact(text: str) -> str:
    """Remove the API key from a message, should an SDK error ever echo it.

    Args:
        text: A message.

    Returns:
        The message with every occurrence of the key replaced.
    """
    key = os.environ.get(API_KEY_ENV, "")
    return text.replace(key, "[redacted]") if key else text


class OpenAIBackend:
    """Live OpenAI Decisions API backend (an async context manager)."""

    def __init__(self, *, timeout_s: float = 90.0, max_retries: int = 4) -> None:
        """Configure the backend; no connection is made until ``async with``.

        Args:
            timeout_s: Per-request timeout.
            max_retries: SDK retries (with backoff, on 408/409/429/5xx) per
                request.
        """
        self._timeout_s = timeout_s
        self._max_retries = max_retries
        self._client: Any = None

    async def __aenter__(self) -> "OpenAIBackend":
        """Open the SDK client (the SDK reads the key from the environment).

        Returns:
            This backend.

        Raises:
            OpenAIAPIError: If the key variable is unset or empty (status 401,
                so the run stops at once).
        """
        from openai import AsyncOpenAI

        if not os.environ.get(API_KEY_ENV):
            raise OpenAIAPIError(f"{API_KEY_ENV} is not set", 401)
        self._client = AsyncOpenAI(
            timeout=self._timeout_s, max_retries=self._max_retries
        )
        return self

    async def __aexit__(self, *exc: object) -> None:
        """Close the SDK client."""
        if self._client is not None:
            await self._client.close()
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
            OpenAIAPIError: On an API or connection error (with its status).
        """
        if self._client is None:
            raise RuntimeError("OpenAIBackend used outside 'async with'")
        try:
            response = await self._client.decisions.create(
                model=model,
                input=to_input(request.state),
                questions=to_questions(request.questions),
            )
        except Exception as err:  # SDK errors -> integer status for is_fatal
            raise OpenAIAPIError(
                redact(f"{type(err).__name__}: {err}"), _status_of(err)
            ) from None
        request_id = getattr(response, "_request_id", None)
        return RawAnswer(
            to_body(response, request.questions),
            request_id if isinstance(request_id, str) else None,
        )
