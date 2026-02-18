"""
Flips Namespace API

Provides access to Trickster flip detection data:
- Recent consensus flips detected by the Trickster agent
- Summary statistics on flip frequency and patterns
- Individual flip details
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class FlipsAPI:
    """
    Synchronous Flips API for consensus flip detection.

    The Trickster agent monitors debates for "hollow consensus" -- cases
    where agents appear to agree but their reasoning is inconsistent.
    Flips occur when positions shift without genuine justification.

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai")
        >>> recent = client.flips.get_recent(limit=10)
        >>> summary = client.flips.get_summary(period="30d")
        >>> print(f"Flips in last 30 days: {summary['total_flips']}")
    """

    def __init__(self, client: AragoraClient):
        self._client = client

    def get_recent(
        self,
        limit: int = 20,
        offset: int = 0,
    ) -> dict[str, Any]:
        """
        Get recent consensus flips.

        Args:
            limit: Maximum results.
            offset: Pagination offset.

        Returns:
            Dict with recent flips including detection details,
            involved agents, and flip severity.
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        return self._client.request("GET", "/api/v1/flips/recent", params=params)

    def get_summary(self, period: str | None = None) -> dict[str, Any]:
        """
        Get flip summary statistics.

        Args:
            period: Time period for summary (e.g., '7d', '30d', '90d').

        Returns:
            Dict with summary statistics including total flips,
            flip rate, most affected agents, and trend data.
        """
        params: dict[str, Any] = {}
        if period:
            params["period"] = period
        return self._client.request("GET", "/api/v1/flips/summary", params=params or None)

    def get(self, flip_id: str) -> dict[str, Any]:
        """
        Get details for a specific flip event.

        Args:
            flip_id: Flip event identifier.

        Returns:
            Dict with full flip details including the debate context,
            position before/after, and trickster assessment.
        """
        return self._client.request("GET", f"/api/v1/flips/{flip_id}")


class AsyncFlipsAPI:
    """
    Asynchronous Flips API for consensus flip detection.

    Example:
        >>> async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
        ...     recent = await client.flips.get_recent(limit=10)
        ...     summary = await client.flips.get_summary(period="30d")
    """

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def get_recent(
        self,
        limit: int = 20,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Get recent consensus flips."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        return await self._client.request("GET", "/api/v1/flips/recent", params=params)

    async def get_summary(self, period: str | None = None) -> dict[str, Any]:
        """Get flip summary statistics."""
        params: dict[str, Any] = {}
        if period:
            params["period"] = period
        return await self._client.request(
            "GET", "/api/v1/flips/summary", params=params or None
        )

    async def get(self, flip_id: str) -> dict[str, Any]:
        """Get details for a specific flip event."""
        return await self._client.request("GET", f"/api/v1/flips/{flip_id}")
