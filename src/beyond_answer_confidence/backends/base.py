"""Backend-neutral request and response types.

A request is a *state* (any JSON object describing the case) plus named
*questions*. Three question types exist, in the wire format of the TypeSafe
System One API, which the other backends interpret the same way:

- ``{"type": "choice", "instructions": str, "criteria": {option: description}}``
  asks for a probability distribution over the options;
- ``{"type": "noul", "instructions": str}`` asks a yes/no question and gets
  P(yes);
- ``{"type": "score", "instructions": str, "criteria": [level, ...]}`` asks
  for a rubric score (and a distribution over the levels).

Responses are parsed into the frozen dataclasses below, so analyses never
depend on a particular SDK.
"""

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

FATAL_STATUSES = frozenset({401, 402, 403})
"""HTTP statuses after which no further call can succeed (bad key, no credits,
no permission)."""


class BudgetExceededError(RuntimeError):
    """Raised before a call once the input-token budget is spent."""


class FatalAPIError(RuntimeError):
    """Returned in place of calls skipped after a fatal API error."""


class OfflineError(RuntimeError):
    """Returned for a request that is not cached while running offline."""


class MalformedResponseError(ValueError):
    """Returned for a response body that :func:`parse_response` cannot parse.

    Paid responses are cached and counted before parsing, so a body that
    fails to parse is not paid for twice; the error is returned in place.
    """


def is_fatal(err: BaseException) -> bool:
    """Return whether an error means further API calls are pointless.

    Args:
        err: An exception returned by a call.

    Returns:
        ``True`` for HTTP 401/402/403 errors and skipped-after-fatal markers.
    """
    return isinstance(err, FatalAPIError) or (
        getattr(err, "status", None) in FATAL_STATUSES
    )


@dataclass(frozen=True)
class Request:
    """One request.

    Attributes:
        state: The case (JSON-serialisable).
        questions: Question dictionaries keyed by name (JSON-serialisable).
        replicate: Replicate index. Not sent to the backend; it gives
            deliberate repeats of the same request their own cache entries.
    """

    state: Any
    questions: Mapping[str, Any]
    replicate: int = 0


@dataclass(frozen=True)
class ChoiceAnswer:
    """Answer to a ``choice`` question.

    Attributes:
        probabilities: Option -> probability (sums to about 1; options the
            backend omits are missing).
        choice: The option the backend picked, if reported.
        confidence: The backend's own confidence score, if reported (kept
            only to check that it adds nothing beyond the probabilities).
    """

    probabilities: Mapping[str, float]
    choice: str | None = None
    confidence: float | None = None


@dataclass(frozen=True)
class YesNoAnswer:
    """Answer to a ``noul`` (yes/no) question.

    Attributes:
        p_yes: Probability of "yes".
    """

    p_yes: float


@dataclass(frozen=True)
class ScoreAnswer:
    """Answer to a ``score`` question.

    Attributes:
        score: The expected (or reported) score on the level scale.
        probabilities: Level index (as a string) -> probability, if reported.
    """

    score: float
    probabilities: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Response:
    """A parsed response.

    Attributes:
        model: The model that answered (may differ from the requested alias).
        input_tokens: Input tokens billed.
        output_tokens: Output tokens billed.
        choices: Choice answers by question name.
        yes_no: Yes/no answers by question name.
        scores: Score answers by question name.
    """

    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    choices: Mapping[str, ChoiceAnswer] = field(default_factory=dict)
    yes_no: Mapping[str, YesNoAnswer] = field(default_factory=dict)
    scores: Mapping[str, ScoreAnswer] = field(default_factory=dict)


def parse_response(raw: Mapping[str, Any]) -> Response:
    """Parse a response body in the System One JSON shape.

    The shape is ``{"model", "usage": {"input_tokens", "output_tokens"},
    "answers": {name: {"type": ..., ...}}}``. Every backend stores its raw
    responses in this shape, so one parser serves all of them.

    Args:
        raw: The decoded response body.

    Returns:
        The parsed response.

    Raises:
        ValueError: On an unknown answer type.
    """
    choices: dict[str, ChoiceAnswer] = {}
    yes_no: dict[str, YesNoAnswer] = {}
    scores: dict[str, ScoreAnswer] = {}
    for name, ans in raw.get("answers", {}).items():
        kind = ans.get("type")
        if kind == "choice":
            conf = ans.get("confidence")
            choices[name] = ChoiceAnswer(
                dict(ans["probabilities"]),
                ans.get("choice"),
                None if conf is None else float(conf),
            )
        elif kind == "noul":
            yes_no[name] = YesNoAnswer(float(ans["noul"]))
        elif kind == "score":
            scores[name] = ScoreAnswer(
                float(ans["score"]), dict(ans.get("probabilities") or {})
            )
        else:
            raise ValueError(f"unknown answer type {kind!r} for question {name!r}")
    usage = raw.get("usage") or {}
    return Response(
        model=str(raw.get("model", "")),
        input_tokens=int(usage.get("input_tokens") or 0),
        output_tokens=int(usage.get("output_tokens") or 0),
        choices=choices,
        yes_no=yes_no,
        scores=scores,
    )


@dataclass(frozen=True)
class RawAnswer:
    """What a live backend returns for one request.

    Attributes:
        body: The response in the System One JSON shape (see
            :func:`parse_response`); this is what the cache stores.
        request_id: The provider's request id, if any.
    """

    body: Mapping[str, Any]
    request_id: str | None = None


class LiveBackend(Protocol):
    """A connection to a paid API.

    Implementations are async context managers; :meth:`answer` may only be
    called inside ``async with``. They must read credentials themselves (from
    the environment) and never expose them.
    """

    async def __aenter__(self) -> "LiveBackend":
        """Open the connection."""
        ...

    async def __aexit__(self, *exc: object) -> None:
        """Close the connection."""
        ...

    async def answer(self, request: Request, model: str) -> RawAnswer:
        """Answer one request.

        Args:
            request: The request (``replicate`` is not sent).
            model: Model name.

        Returns:
            The raw answer.
        """
        ...


CHARS_PER_TOKEN = 1.5
"""Characters of request JSON per estimated token. Measured requests used 1.7
(bare opaque codes) to 3.4 (with examples) characters per token, so 1.5
over-estimates every measured request."""


def estimate_tokens(request: Request) -> int:
    """Conservative input-token estimate for budget reservation.

    ``ceil(len(request JSON) / 1.5)`` (see :data:`CHARS_PER_TOKEN`). It is an
    estimate, not a bound: a request with unusually short tokens can still
    cost more.

    Args:
        request: The request.

    Returns:
        Estimated input tokens.
    """
    chars = len(json.dumps({"s": request.state, "q": request.questions}))
    return math.ceil(chars / CHARS_PER_TOKEN)
