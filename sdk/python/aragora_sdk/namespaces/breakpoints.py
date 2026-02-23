"""
Breakpoints Namespace API

Provides methods for managing debug breakpoints in debate execution.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class BreakpointsAPI:
    """Synchronous Breakpoints API."""

    def __init__(self, client: AragoraClient):
        self._client = client

    def create(self, **kwargs: Any) -> dict[str, Any]:
        """Create a new breakpoint.

        Args:
            **kwargs: Breakpoint configuration (debate_id, round, condition, etc.).

        Returns:
            Dict with created breakpoint details.
        """
        return self._client.request("POST", "/api/v1/breakpoints", json=kwargs)


class AsyncBreakpointsAPI:
    """Asynchronous Breakpoints API."""

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def create(self, **kwargs: Any) -> dict[str, Any]:
        """Create a new breakpoint."""
        return await self._client.request("POST", "/api/v1/breakpoints", json=kwargs)
