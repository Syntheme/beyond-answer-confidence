"""TypeSafe System One ("Jev") backend.

The API key is read by the SDK from ``TYPESAFE_API_KEY`` and never passes
through this package. The SDK logger is pinned to WARNING so wire logging
(which includes headers) cannot be switched on by a stray DEBUG setting.

This project is not affiliated with or endorsed by TypeSafe.
"""

import logging
from typing import Any

from beyond_answer_confidence.backends.base import RawAnswer, Request

DEFAULT_MODEL = "jev-latest"
"""The API exposes aliases (``jev-latest``, ``jev-preview``), not concrete
versions; each response's ``model`` field records what actually answered."""

logging.getLogger("typesafe_sdk").setLevel(logging.WARNING)


class JevBackend:
    """Live System One backend (an async context manager)."""

    def __init__(self, *, timeout_s: float = 90.0, max_retries: int = 4) -> None:
        """Configure the backend; no connection is made until ``async with``.

        Args:
            timeout_s: Per-request timeout.
            max_retries: SDK retries (with backoff) per request.
        """
        self._timeout_s = timeout_s
        self._max_retries = max_retries
        self._client: Any = None

    async def __aenter__(self) -> "JevBackend":
        """Open the SDK client (reads the key from the environment).

        Returns:
            This backend.
        """
        from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy

        client = AsyncTypeSafeClient(
            timeout=self._timeout_s, retry=RetryPolicy(max_retries=self._max_retries)
        )
        self._client = await client.__aenter__()
        return self

    async def __aexit__(self, *exc: object) -> None:
        """Close the SDK client."""
        if self._client is not None:
            await self._client.__aexit__(None, None, None)
            self._client = None

    async def answer(self, request: Request, model: str) -> RawAnswer:
        """Answer one request.

        Args:
            request: The request.
            model: Model alias.

        Returns:
            The response body and request id.

        Raises:
            RuntimeError: If called outside ``async with``.
        """
        from typesafe_sdk import TypeSafeError

        if self._client is None:
            raise RuntimeError("JevBackend used outside 'async with'")
        response = await self._client.system_one(
            request.state, request.questions, model=model
        )
        try:
            request_id: str | None = response.request_id
        except TypeSafeError:
            request_id = None
        return RawAnswer(response.model_dump(mode="json"), request_id)
