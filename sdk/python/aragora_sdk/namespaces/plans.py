"""
Plans Namespace API

Provides methods for plan management.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class PlansAPI:
    """Synchronous Plans API."""

    def __init__(self, client: AragoraClient):
        self._client = client

    def update(self, **kwargs: Any) -> dict[str, Any]:
        """Update a plan.

        Args:
            **kwargs: Plan update fields (plan_id, status, tasks, etc.).

        Returns:
            Dict with updated plan details.
        """
        return self._client.request("PUT", "/api/v1/plans", json=kwargs)


class AsyncPlansAPI:
    """Asynchronous Plans API."""

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def update(self, **kwargs: Any) -> dict[str, Any]:
        """Update a plan."""
        return await self._client.request("PUT", "/api/v1/plans", json=kwargs)
