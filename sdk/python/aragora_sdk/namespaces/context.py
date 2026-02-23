"""
Context Namespace API

Provides methods for context budget management and estimation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class ContextAPI:
    """Synchronous Context API."""

    def __init__(self, client: AragoraClient):
        self._client = client

    def get_budget(self) -> dict[str, Any]:
        """Get current context budget usage.

        Returns:
            Dict with context budget status and remaining tokens.
        """
        return self._client.request("GET", "/api/v1/context/budget")

    def estimate_budget(self, **params: Any) -> dict[str, Any]:
        """Estimate context budget for a planned operation.

        Args:
            **params: Estimation parameters (task, agents, rounds, etc.).

        Returns:
            Dict with estimated token usage and cost.
        """
        return self._client.request(
            "GET", "/api/v1/context/budget/estimate", params=params
        )


class AsyncContextAPI:
    """Asynchronous Context API."""

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def get_budget(self) -> dict[str, Any]:
        """Get current context budget usage."""
        return await self._client.request("GET", "/api/v1/context/budget")

    async def estimate_budget(self, **params: Any) -> dict[str, Any]:
        """Estimate context budget for a planned operation."""
        return await self._client.request(
            "GET", "/api/v1/context/budget/estimate", params=params
        )
