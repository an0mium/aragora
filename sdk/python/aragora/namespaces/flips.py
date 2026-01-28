"""
Flips Namespace API

Provides access to Trickster flip detection data.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class FlipsAPI:
    """Synchronous Flips API for consensus flip detection."""

    def __init__(self, client: AragoraClient):
        self._client = client

    def get_recent(
        self,
        limit: int = 20,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Get recent consensus flips.

        Args:
            limit: Maximum results.
            offset: Pagination offset.

        Returns:
            Recent flips with detection details.
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        return self._client.request("GET", "/api/v1/flips/recent", params=params)

    def get_summary(self, period: str | None = None) -> dict[str, Any]:
        """Get flip summary statistics.

        Args:
            period: Time period for summary (e.g., '7d', '30d').

        Returns:
            Summary statistics for flips.
        """
        params: dict[str, Any] = {}
        if period:
            params["period"] = period
        return self._client.request("GET", "/api/v1/flips/summary", params=params)


class AsyncFlipsAPI:
    """Asynchronous Flips API for consensus flip detection."""

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def get_recent(
        self,
        limit: int = 20,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Get recent consensus flips.

        Args:
            limit: Maximum results.
            offset: Pagination offset.

        Returns:
            Recent flips with detection details.
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        return await self._client.request("GET", "/api/v1/flips/recent", params=params)

    async def get_summary(self, period: str | None = None) -> dict[str, Any]:
        """Get flip summary statistics.

        Args:
            period: Time period for summary (e.g., '7d', '30d').

        Returns:
            Summary statistics for flips.
        """
        params: dict[str, Any] = {}
        if period:
            params["period"] = period
        return await self._client.request("GET", "/api/v1/flips/summary", params=params)
