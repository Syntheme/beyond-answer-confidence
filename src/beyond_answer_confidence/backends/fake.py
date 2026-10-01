"""A deterministic offline backend for tests and the quick start.

It needs no network and no key. For a choice question it puts half the mass
on the first option and spreads the rest uniformly; yes/no answers are 0.25
or 0.75 depending on the question name; scores sit mid-scale.
"""

from collections.abc import Mapping
from typing import Any

from beyond_answer_confidence.backends.base import RawAnswer, Request


class StatusError(Exception):
    """A simulated API error carrying an HTTP status."""

    def __init__(self, status: int) -> None:
        """Create the error.

        Args:
            status: HTTP status.
        """
        super().__init__(f"HTTP {status}")
        self.status = status


def fake_body(
    questions: Mapping[str, Any], model: str, tokens: int = 100
) -> dict[str, Any]:
    """Build the fake response body for a set of questions.

    Args:
        questions: Question dictionaries.
        model: Model name to report.
        tokens: Input tokens to report.

    Returns:
        A body in the System One JSON shape.
    """
    answers: dict[str, Any] = {}
    for name, q in questions.items():
        if q["type"] == "noul":
            answers[name] = {"type": "noul", "noul": 0.25 + 0.5 * (len(name) % 2)}
        elif q["type"] == "score":
            n = len(q["criteria"])
            answers[name] = {
                "type": "score",
                "score": (n - 1) / 2,
                "probabilities": {str(i): 1 / n for i in range(n)},
            }
        else:
            codes = list(q["criteria"])
            k = len(codes)
            probs = {
                c: 0.5 + 0.5 / k if i == 0 else 0.5 / k for i, c in enumerate(codes)
            }
            answers[name] = {
                "type": "choice",
                "choice": codes[0],
                "probabilities": probs,
            }
    return {
        "model": model,
        "usage": {"input_tokens": tokens, "output_tokens": 0},
        "answers": answers,
    }


class FakeBackend:
    """Deterministic stand-in for a live backend.

    Attributes:
        calls: Number of answered calls.
        max_concurrent: Highest number of simultaneous calls seen.
    """

    def __init__(
        self,
        tokens_per_call: int = 100,
        *,
        fail_on: frozenset[str] = frozenset(),
        status_on: Mapping[str, int] | None = None,
        state_field: str = "text",
    ) -> None:
        """Configure the fake.

        Args:
            tokens_per_call: Input tokens reported per call.
            fail_on: Values of ``state[state_field]`` that raise a connection
                error.
            status_on: Values of ``state[state_field]`` that raise an HTTP
                error with the given status.
            state_field: State field inspected by ``fail_on``/``status_on``.
        """
        self.tokens_per_call = tokens_per_call
        self.fail_on = fail_on
        self.status_on = dict(status_on or {})
        self.state_field = state_field
        self.calls = 0
        self.max_concurrent = 0
        self._active = 0

    async def __aenter__(self) -> "FakeBackend":
        """Open (no-op).

        Returns:
            This backend.
        """
        return self

    async def __aexit__(self, *exc: object) -> None:
        """Close (no-op)."""

    async def answer(self, request: Request, model: str) -> RawAnswer:
        """Answer one request.

        Args:
            request: The request.
            model: Model name.

        Returns:
            The fake answer.

        Raises:
            ConnectionError: For states listed in ``fail_on``.
            StatusError: For states listed in ``status_on``.
        """
        import asyncio

        self._active += 1
        self.max_concurrent = max(self.max_concurrent, self._active)
        try:
            await asyncio.sleep(0)
            marker = (
                request.state.get(self.state_field)
                if isinstance(request.state, Mapping)
                else None
            )
            if marker in self.fail_on:
                raise ConnectionError("simulated failure")
            if (status := self.status_on.get(str(marker))) is not None:
                raise StatusError(status)
            self.calls += 1
            return RawAnswer(fake_body(request.questions, model, self.tokens_per_call))
        finally:
            self._active -= 1
