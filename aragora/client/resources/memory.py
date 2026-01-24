"""MemoryAPI resource for the Aragora client."""

from __future__ import annotations

from typing import Any, Optional, TYPE_CHECKING, cast

from ..models import (
    CritiqueEntry,
    MemoryAnalyticsResponse,
    MemoryEntry,
    MemorySnapshotResponse,
    MemoryStats,
    MemoryTierStats,
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

    def stats(self) -> MemoryStats:
        """
        Get memory system statistics.

        Returns:
            MemoryStats with storage and tier information.
        """
        response = self._client._get("/api/v1/memory/stats")
        return MemoryStats(**response)

    async def stats_async(self) -> MemoryStats:
        """Async version of stats()."""
        response = await self._client._get_async("/api/v1/memory/stats")
        return MemoryStats(**response)

    def search(
        self,
        query: str,
        tiers: Optional[list[str]] = None,
        agent: Optional[str] = None,
        limit: int = 20,
        min_importance: Optional[float] = None,
        include_expired: bool = False,
    ) -> list[MemoryEntry]:
        """
        Search memory entries.

        Args:
            query: Search query string.
            tiers: Optional list of tiers to search (fast, medium, slow, glacial).
            agent: Optional agent filter.
            limit: Maximum entries to return.
            min_importance: Minimum importance threshold.
            include_expired: Include expired entries.

        Returns:
            List of MemoryEntry matches.
        """
        params: dict[str, Any] = {
            "q": query,
            "limit": limit,
            "include_expired": include_expired,
        }
        if tiers:
            params["tiers"] = ",".join(tiers)
        if agent:
            params["agent"] = agent
        if min_importance is not None:
            params["min_importance"] = min_importance

        response = self._client._get("/api/v1/memory/search", params=params)
        entries = response.get("entries", response) if isinstance(response, dict) else response
        return [MemoryEntry(**e) for e in entries]

    async def search_async(
        self,
        query: str,
        tiers: Optional[list[str]] = None,
        agent: Optional[str] = None,
        limit: int = 20,
        min_importance: Optional[float] = None,
        include_expired: bool = False,
    ) -> list[MemoryEntry]:
        """Async version of search()."""
        params: dict[str, Any] = {
            "q": query,
            "limit": limit,
            "include_expired": include_expired,
        }
        if tiers:
            params["tiers"] = ",".join(tiers)
        if agent:
            params["agent"] = agent
        if min_importance is not None:
            params["min_importance"] = min_importance

        response = await self._client._get_async("/api/v1/memory/search", params=params)
        entries = response.get("entries", response) if isinstance(response, dict) else response
        return [MemoryEntry(**e) for e in entries]

    def get_tiers(self) -> list[MemoryTierStats]:
        """
        Get statistics for all memory tiers.

        Returns:
            List of MemoryTierStats for each tier.
        """
        response = self._client._get("/api/v1/memory/tiers")
        tiers = response.get("tiers", response) if isinstance(response, dict) else response
        return [MemoryTierStats(**t) for t in tiers]

    async def get_tiers_async(self) -> list[MemoryTierStats]:
        """Async version of get_tiers()."""
        response = await self._client._get_async("/api/v1/memory/tiers")
        tiers = response.get("tiers", response) if isinstance(response, dict) else response
        return [MemoryTierStats(**t) for t in tiers]

    def get_critiques(
        self,
        limit: int = 20,
        offset: int = 0,
    ) -> list[CritiqueEntry]:
        """
        Get critique entries from memory.

        Args:
            limit: Maximum entries to return.
            offset: Entries to skip.

        Returns:
            List of CritiqueEntry records.
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        response = self._client._get("/api/v1/memory/critiques", params=params)
        critiques = response.get("critiques", response) if isinstance(response, dict) else response
        return [CritiqueEntry(**c) for c in critiques]

    async def get_critiques_async(
        self,
        limit: int = 20,
        offset: int = 0,
    ) -> list[CritiqueEntry]:
        """Async version of get_critiques()."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        response = await self._client._get_async("/api/v1/memory/critiques", params=params)
        critiques = response.get("critiques", response) if isinstance(response, dict) else response
        return [CritiqueEntry(**c) for c in critiques]
