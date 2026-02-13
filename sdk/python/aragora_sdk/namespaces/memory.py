"""
Memory Namespace API

Provides methods for interacting with the memory system:
- Memory search across tiers
- Statistics and monitoring (stats, tier-stats, archive-stats, pressure, analytics)
- Tier listing
- Continuum retrieval
- Critique operations (list, store)

Note: Core CRUD (store/retrieve/update/delete by key), query, semantic-search,
tier operations (get_tier, move, promote, demote), context management,
cross-debate memory, export/import, snapshots, and maintenance operations
(prune, compact, sync, vacuum, rebuild-index, consolidate, continuum store/stats)
were removed as their handler routes no longer exist.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient

MemoryTier = Literal["fast", "medium", "slow", "glacial"]


class MemoryAPI:
    """Synchronous Memory API."""

    def __init__(self, client: AragoraClient):
        self._client = client

    # ===========================================================================
    # Search Operations
    # ===========================================================================

    def search(
        self,
        query: str,
        *,
        tier: list[MemoryTier] | None = None,
        limit: int = 20,
        min_importance: float = 0.0,
        sort: str = "relevance",
    ) -> dict[str, Any]:
        """Search memories across tiers."""
        params: dict[str, Any] = {
            "q": query,
            "limit": limit,
            "min_importance": min_importance,
            "sort": sort,
        }
        if tier:
            params["tier"] = ",".join(tier)
        return self._client.request("GET", "/api/v1/memory/search", params=params)

    # ===========================================================================
    # Statistics and Monitoring
    # ===========================================================================

    def stats(self) -> dict[str, Any]:
        """
        Get memory system statistics.

        Returns:
            Dict with overall memory statistics
        """
        return self._client.request("GET", "/api/v1/memory/stats")

    def get_tier_stats(self) -> dict[str, Any]:
        """Get tier statistics."""
        return self._client.request("GET", "/api/v1/memory/tier-stats")

    def get_archive_stats(self) -> dict[str, Any]:
        """Get archive statistics."""
        return self._client.request("GET", "/api/v1/memory/archive-stats")

    def get_pressure(self) -> dict[str, Any]:
        """
        Get memory pressure and utilization.

        Returns:
            Dict with utilization percentage, pressure level, and recommendations
        """
        return self._client.request("GET", "/api/v1/memory/pressure")

    def get_analytics(
        self,
        *,
        start_time: str | None = None,
        end_time: str | None = None,
        granularity: str = "hour",
    ) -> dict[str, Any]:
        """
        Get memory analytics over time.

        Args:
            start_time: Start of time range (ISO format)
            end_time: End of time range (ISO format)
            granularity: Time granularity (minute, hour, day)

        Returns:
            Dict with time-series analytics data
        """
        params: dict[str, Any] = {"granularity": granularity}
        if start_time:
            params["start_time"] = start_time
        if end_time:
            params["end_time"] = end_time
        return self._client.request("GET", "/api/v1/memory/analytics", params=params)

    # ===========================================================================
    # Tier Operations
    # ===========================================================================

    def list_tiers(self) -> dict[str, Any]:
        """List all memory tiers with detailed stats."""
        return self._client.request("GET", "/api/v1/memory/tiers")

    def tiers(self) -> dict[str, Any]:
        """
        Get information about memory tiers.

        Alias for list_tiers() for TypeScript SDK compatibility.
        """
        return self.list_tiers()

    # ===========================================================================
    # Continuum Operations
    # ===========================================================================

    def retrieve_continuum(
        self,
        query: str = "",
        tiers: list[MemoryTier] | None = None,
        limit: int = 10,
        min_importance: float = 0.0,
    ) -> dict[str, Any]:
        """Retrieve memories from the continuum."""
        params: dict[str, Any] = {
            "query": query,
            "limit": limit,
            "min_importance": min_importance,
        }
        if tiers:
            params["tiers"] = ",".join(tiers)
        return self._client.request("GET", "/api/v1/memory/continuum/retrieve", params=params)

    def retrieve_from_continuum(
        self,
        query: str,
        *,
        tiers: list[MemoryTier] | None = None,
        limit: int = 10,
        min_importance: float = 0.0,
    ) -> dict[str, Any]:
        """
        Retrieve content from the continuum memory system.

        Alias for retrieve_continuum() for TypeScript SDK compatibility.
        """
        return self.retrieve_continuum(
            query, tiers=tiers, limit=limit, min_importance=min_importance
        )

    # ===========================================================================
    # Critique Operations
    # ===========================================================================

    def list_critiques(
        self,
        agent: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Browse critique store entries."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if agent:
            params["agent"] = agent
        return self._client.request("GET", "/api/v1/memory/critiques", params=params)

    def critiques(
        self,
        *,
        agent: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> dict[str, Any]:
        """
        Get stored critiques from memory.

        Alias for list_critiques() for TypeScript SDK compatibility.
        """
        return self.list_critiques(agent=agent, limit=limit, offset=offset)

    def store_critique(
        self,
        critique: str,
        *,
        agent: str,
        debate_id: str | None = None,
        target_agent: str | None = None,
        score: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Store a critique in memory.

        Args:
            critique: The critique content
            agent: Agent that generated the critique
            debate_id: Associated debate ID
            target_agent: Agent being critiqued
            score: Critique quality score
            metadata: Additional metadata

        Returns:
            Dict with stored critique ID
        """
        body: dict[str, Any] = {"critique": critique, "agent": agent}
        if debate_id:
            body["debate_id"] = debate_id
        if target_agent:
            body["target_agent"] = target_agent
        if score is not None:
            body["score"] = score
        if metadata:
            body["metadata"] = metadata
        return self._client.request("POST", "/api/v1/memory/critiques", json=body)



class AsyncMemoryAPI:
    """Asynchronous Memory API."""

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    # ===========================================================================
    # Search Operations
    # ===========================================================================

    async def search(
        self,
        query: str,
        *,
        tier: list[MemoryTier] | None = None,
        limit: int = 20,
        min_importance: float = 0.0,
        sort: str = "relevance",
    ) -> dict[str, Any]:
        """Search memories across tiers."""
        params: dict[str, Any] = {
            "q": query,
            "limit": limit,
            "min_importance": min_importance,
            "sort": sort,
        }
        if tier:
            params["tier"] = ",".join(tier)
        return await self._client.request("GET", "/api/v1/memory/search", params=params)

    # ===========================================================================
    # Statistics and Monitoring
    # ===========================================================================

    async def stats(self) -> dict[str, Any]:
        """
        Get memory system statistics.

        Returns:
            Dict with overall memory statistics
        """
        return await self._client.request("GET", "/api/v1/memory/stats")

    async def get_tier_stats(self) -> dict[str, Any]:
        """Get tier statistics."""
        return await self._client.request("GET", "/api/v1/memory/tier-stats")

    async def get_archive_stats(self) -> dict[str, Any]:
        """Get archive statistics."""
        return await self._client.request("GET", "/api/v1/memory/archive-stats")

    async def get_pressure(self) -> dict[str, Any]:
        """
        Get memory pressure and utilization.

        Returns:
            Dict with utilization percentage, pressure level, and recommendations
        """
        return await self._client.request("GET", "/api/v1/memory/pressure")

    async def get_analytics(
        self,
        *,
        start_time: str | None = None,
        end_time: str | None = None,
        granularity: str = "hour",
    ) -> dict[str, Any]:
        """
        Get memory analytics over time.

        Args:
            start_time: Start of time range (ISO format)
            end_time: End of time range (ISO format)
            granularity: Time granularity (minute, hour, day)

        Returns:
            Dict with time-series analytics data
        """
        params: dict[str, Any] = {"granularity": granularity}
        if start_time:
            params["start_time"] = start_time
        if end_time:
            params["end_time"] = end_time
        return await self._client.request("GET", "/api/v1/memory/analytics", params=params)

    # ===========================================================================
    # Tier Operations
    # ===========================================================================

    async def list_tiers(self) -> dict[str, Any]:
        """List all memory tiers with detailed stats."""
        return await self._client.request("GET", "/api/v1/memory/tiers")

    async def tiers(self) -> dict[str, Any]:
        """
        Get information about memory tiers.

        Alias for list_tiers() for TypeScript SDK compatibility.
        """
        return await self.list_tiers()

    # ===========================================================================
    # Continuum Operations
    # ===========================================================================

    async def retrieve_continuum(
        self,
        query: str = "",
        tiers: list[MemoryTier] | None = None,
        limit: int = 10,
        min_importance: float = 0.0,
    ) -> dict[str, Any]:
        """Retrieve memories from the continuum."""
        params: dict[str, Any] = {
            "query": query,
            "limit": limit,
            "min_importance": min_importance,
        }
        if tiers:
            params["tiers"] = ",".join(tiers)
        return await self._client.request("GET", "/api/v1/memory/continuum/retrieve", params=params)

    async def retrieve_from_continuum(
        self,
        query: str,
        *,
        tiers: list[MemoryTier] | None = None,
        limit: int = 10,
        min_importance: float = 0.0,
    ) -> dict[str, Any]:
        """
        Retrieve content from the continuum memory system.

        Alias for retrieve_continuum() for TypeScript SDK compatibility.
        """
        return await self.retrieve_continuum(
            query, tiers=tiers, limit=limit, min_importance=min_importance
        )

    # ===========================================================================
    # Critique Operations
    # ===========================================================================

    async def list_critiques(
        self,
        agent: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Browse critique store entries."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if agent:
            params["agent"] = agent
        return await self._client.request("GET", "/api/v1/memory/critiques", params=params)

    async def critiques(
        self,
        *,
        agent: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> dict[str, Any]:
        """
        Get stored critiques from memory.

        Alias for list_critiques() for TypeScript SDK compatibility.
        """
        return await self.list_critiques(agent=agent, limit=limit, offset=offset)

    async def store_critique(
        self,
        critique: str,
        *,
        agent: str,
        debate_id: str | None = None,
        target_agent: str | None = None,
        score: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Store a critique in memory.

        Args:
            critique: The critique content
            agent: Agent that generated the critique
            debate_id: Associated debate ID
            target_agent: Agent being critiqued
            score: Critique quality score
            metadata: Additional metadata

        Returns:
            Dict with stored critique ID
        """
        body: dict[str, Any] = {"critique": critique, "agent": agent}
        if debate_id:
            body["debate_id"] = debate_id
        if target_agent:
            body["target_agent"] = target_agent
        if score is not None:
            body["score"] = score
        if metadata:
            body["metadata"] = metadata
        return await self._client.request("POST", "/api/v1/memory/critiques", json=body)

