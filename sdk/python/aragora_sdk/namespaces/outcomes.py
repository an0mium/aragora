"""
Outcomes Namespace API

Provides methods for querying decision outcomes:
- Impact analysis
- Outcome search
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class OutcomesAPI:
    """Synchronous Outcomes API."""

    def __init__(self, client: AragoraClient):
        self._client = client

    def get_impact(self, **params: Any) -> dict[str, Any]:
        """Get outcome impact analysis.

        Args:
            **params: Filter parameters (decision_id, date_range, etc.).

        Returns:
            Dict with impact metrics and analysis.
        """
        return self._client.request("GET", "/api/v1/outcomes/impact", params=params or None)

    def search(self, **params: Any) -> dict[str, Any]:
        """Search outcomes.

        Args:
            **params: Search parameters (query, status, limit, offset, etc.).

        Returns:
            Dict with matching outcomes and pagination.
        """
        return self._client.request("GET", "/api/v1/outcomes/search", params=params or None)


class AsyncOutcomesAPI:
    """Asynchronous Outcomes API."""

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def get_impact(self, **params: Any) -> dict[str, Any]:
        """Get outcome impact analysis."""
        return await self._client.request("GET", "/api/v1/outcomes/impact", params=params or None)

    async def search(self, **params: Any) -> dict[str, Any]:
        """Search outcomes."""
        return await self._client.request("GET", "/api/v1/outcomes/search", params=params or None)
