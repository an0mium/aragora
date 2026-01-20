"""MemoryAPI resource for the Aragora client."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING, cast

from ..models import (
    MemoryAnalyticsResponse,
    MemorySnapshotResponse,
)

if TYPE_CHECKING:
    from ..client import AragoraClient


class MemoryAPI:
    """API interface for memory tier analytics."""

    def __init__(self, client: "AragoraClient"):
        self._client = client

    def analytics(self, days: int = 30) -> MemoryAnalyticsResponse:
        """
        Get comprehensive memory tier analytics.

        Args:
            days: Number of days to analyze (1-365).

        Returns:
            MemoryAnalyticsResponse with tier stats and recommendations.
        """
        response = self._client._get("/api/memory/analytics", params={"days": days})
        return MemoryAnalyticsResponse(**response)

    async def analytics_async(self, days: int = 30) -> MemoryAnalyticsResponse:
        """Async version of analytics()."""
        response = await self._client._get_async("/api/memory/analytics", params={"days": days})
        return MemoryAnalyticsResponse(**response)

    def tier_stats(self, tier_name: str, days: int = 30) -> dict[str, Any]:
        """
        Get statistics for a specific memory tier.

        Args:
            tier_name: Name of the tier (fast, medium, slow, glacial).
            days: Number of days to analyze.

        Returns:
            Dict with tier-specific statistics.
        """
        response = self._client._get(
            f"/api/memory/analytics/tier/{tier_name}", params={"days": days}
        )
        return cast(dict[str, Any], response)

    async def tier_stats_async(self, tier_name: str, days: int = 30) -> dict[str, Any]:
        """Async version of tier_stats()."""
        response = await self._client._get_async(
            f"/api/memory/analytics/tier/{tier_name}", params={"days": days}
        )
        return cast(dict[str, Any], response)

    def snapshot(self) -> MemorySnapshotResponse:
        """
        Take a manual memory analytics snapshot.

        Returns:
            MemorySnapshotResponse with snapshot details.
        """
        response = self._client._post("/api/memory/analytics/snapshot", {})
        return MemorySnapshotResponse(**response)

    async def snapshot_async(self) -> MemorySnapshotResponse:
        """Async version of snapshot()."""
        response = await self._client._post_async("/api/memory/analytics/snapshot", {})
        return MemorySnapshotResponse(**response)
