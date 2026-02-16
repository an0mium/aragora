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

    # ===========================================================================
    # Progressive Retrieval & Viewer
    # ===========================================================================

    def search_index(
        self,
        query: str,
        *,
        limit: int = 20,
        min_importance: float = 0.0,
        tiers: list[MemoryTier] | None = None,
        use_hybrid: bool = False,
    ) -> dict[str, Any]:
        """
        Progressive retrieval stage 1: compact index entries.

        GET /api/v1/memory/search-index

        Args:
            query: Search query
            limit: Maximum entries to return (1-100)
            min_importance: Minimum importance threshold (0.0-1.0)
            tiers: Filter by memory tiers
            use_hybrid: Enable hybrid search mode

        Returns:
            Dict with compact index entries for progressive loading
        """
        params: dict[str, Any] = {
            "q": query,
            "limit": limit,
            "min_importance": min_importance,
            "use_hybrid": str(use_hybrid).lower(),
        }
        if tiers:
            params["tiers"] = ",".join(tiers)
        return self._client.request("GET", "/api/v1/memory/search-index", params=params)

    def search_timeline(
        self,
        query: str,
        *,
        limit: int = 20,
        min_importance: float = 0.0,
        tiers: list[MemoryTier] | None = None,
    ) -> dict[str, Any]:
        """
        Progressive retrieval: timeline-ordered search results.

        GET /api/v1/memory/search-timeline

        Args:
            query: Search query
            limit: Maximum entries to return
            min_importance: Minimum importance threshold
            tiers: Filter by memory tiers

        Returns:
            Dict with timeline-ordered memory entries
        """
        params: dict[str, Any] = {
            "q": query,
            "limit": limit,
            "min_importance": min_importance,
        }
        if tiers:
            params["tiers"] = ",".join(tiers)
        return self._client.request("GET", "/api/v1/memory/search-timeline", params=params)

    def list_entries(
        self,
        *,
        limit: int = 20,
        offset: int = 0,
        tier: MemoryTier | None = None,
    ) -> dict[str, Any]:
        """
        List memory entries.

        GET /api/v1/memory/entries

        Args:
            limit: Maximum entries to return
            offset: Pagination offset
            tier: Filter by memory tier

        Returns:
            Dict with memory entries and pagination
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if tier:
            params["tier"] = tier
        return self._client.request("GET", "/api/v1/memory/entries", params=params)

    def get_viewer(self) -> dict[str, Any]:
        """
        Get memory viewer HTML interface.

        GET /api/v1/memory/viewer

        Returns:
            Dict with viewer HTML content
        """
        return self._client.request("GET", "/api/v1/memory/viewer")

    # ===========================================================================
    # Advanced Operations
    # ===========================================================================

    def compact(
        self, workspace_id: str | None = None
    ) -> dict[str, Any]:
        """Trigger memory compaction."""
        body: dict[str, Any] = {}
        if workspace_id:
            body["workspace_id"] = workspace_id
        return self._client.request("POST", "/api/v1/memory/compact", json=body)

    def get_context(
        self, workspace_id: str | None = None
    ) -> dict[str, Any]:
        """Get memory context for current session."""
        params: dict[str, Any] = {}
        if workspace_id:
            params["workspace_id"] = workspace_id
        return self._client.request("GET", "/api/v1/memory/context", params=params)

    def get_cross_debate(
        self, workspace_id: str | None = None
    ) -> dict[str, Any]:
        """Get cross-debate memory entries."""
        params: dict[str, Any] = {}
        if workspace_id:
            params["workspace_id"] = workspace_id
        return self._client.request("GET", "/api/v1/memory/cross-debate", params=params)

    def inject_cross_debate(
        self,
        debate_id: str,
        entries: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Inject cross-debate memories into a debate."""
        body: dict[str, Any] = {"debate_id": debate_id}
        if entries:
            body["entries"] = entries
        return self._client.request("POST", "/api/v1/memory/cross-debate/inject", json=body)

    def export_memories(
        self,
        format: str = "json",
        workspace_id: str | None = None,
    ) -> dict[str, Any]:
        """Export memories."""
        body: dict[str, Any] = {"format": format}
        if workspace_id:
            body["workspace_id"] = workspace_id
        return self._client.request("POST", "/api/v1/memory/export", json=body)

    def import_memories(self, data: dict[str, Any]) -> dict[str, Any]:
        """Import memories from exported data."""
        return self._client.request("POST", "/api/v1/memory/import", json=data)

    def prune(
        self,
        workspace_id: str | None = None,
        max_age_days: int | None = None,
    ) -> dict[str, Any]:
        """Prune old or low-importance memories."""
        body: dict[str, Any] = {}
        if workspace_id:
            body["workspace_id"] = workspace_id
        if max_age_days is not None:
            body["max_age_days"] = max_age_days
        return self._client.request("POST", "/api/v1/memory/prune", json=body)

    def query(
        self,
        prompt: str,
        workspace_id: str | None = None,
    ) -> dict[str, Any]:
        """Query memories using natural language."""
        body: dict[str, Any] = {"prompt": prompt}
        if workspace_id:
            body["workspace_id"] = workspace_id
        return self._client.request("POST", "/api/v1/memory/query", json=body)

    def rebuild_index(
        self, workspace_id: str | None = None
    ) -> dict[str, Any]:
        """Rebuild the memory search index."""
        body: dict[str, Any] = {}
        if workspace_id:
            body["workspace_id"] = workspace_id
        return self._client.request("POST", "/api/v1/memory/rebuild-index", json=body)

    def semantic_search(
        self,
        query: str,
        *,
        limit: int = 20,
        min_similarity: float = 0.5,
    ) -> dict[str, Any]:
        """Perform semantic search across memories."""
        body: dict[str, Any] = {
            "query": query,
            "limit": limit,
            "min_similarity": min_similarity,
        }
        return self._client.request("POST", "/api/v1/memory/semantic-search", json=body)

    def list_snapshots(
        self, limit: int = 20, offset: int = 0
    ) -> dict[str, Any]:
        """List memory snapshots."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        return self._client.request("GET", "/api/v1/memory/snapshots", params=params)

    def get_snapshot(self, snapshot_id: str) -> dict[str, Any]:
        """Get a specific memory snapshot."""
        return self._client.request("GET", f"/api/v1/memory/snapshots/{snapshot_id}")

    def restore_snapshot(self, snapshot_id: str) -> dict[str, Any]:
        """Restore a memory snapshot."""
        return self._client.request("POST", f"/api/v1/memory/snapshots/{snapshot_id}/restore", json={})

    def sync_memories(
        self, workspace_id: str | None = None
    ) -> dict[str, Any]:
        """Sync memories across tiers."""
        body: dict[str, Any] = {}
        if workspace_id:
            body["workspace_id"] = workspace_id
        return self._client.request("POST", "/api/v1/memory/sync", json=body)

    def get_tier(self, tier: str) -> dict[str, Any]:
        """Get detailed information about a specific memory tier."""
        return self._client.request("GET", f"/api/v1/memory/tier/{tier}")

    def vacuum(self) -> dict[str, Any]:
        """Run vacuum to reclaim storage space."""
        return self._client.request("POST", "/api/v1/memory/vacuum", json={})

    def promote(
        self, key: str, target_tier: str | None = None
    ) -> dict[str, Any]:
        """Promote a memory entry to a higher tier."""
        body: dict[str, Any] = {}
        if target_tier:
            body["target_tier"] = target_tier
        return self._client.request("POST", f"/api/v1/memory/{key}/promote", json=body)

    def demote(
        self, key: str, target_tier: str | None = None
    ) -> dict[str, Any]:
        """Demote a memory entry to a lower tier."""
        body: dict[str, Any] = {}
        if target_tier:
            body["target_tier"] = target_tier
        return self._client.request("POST", f"/api/v1/memory/{key}/demote", json=body)

    def move(self, key: str, target_tier: str) -> dict[str, Any]:
        """Move a memory entry to a specific tier."""
        body: dict[str, Any] = {"target_tier": target_tier}
        return self._client.request("POST", f"/api/v1/memory/{key}/move", json=body)


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

    # ===========================================================================
    # Progressive Retrieval & Viewer
    # ===========================================================================

    async def search_index(
        self,
        query: str,
        *,
        limit: int = 20,
        min_importance: float = 0.0,
        tiers: list[MemoryTier] | None = None,
        use_hybrid: bool = False,
    ) -> dict[str, Any]:
        """Progressive retrieval stage 1: compact index entries. GET /api/v1/memory/search-index"""
        params: dict[str, Any] = {
            "q": query,
            "limit": limit,
            "min_importance": min_importance,
            "use_hybrid": str(use_hybrid).lower(),
        }
        if tiers:
            params["tiers"] = ",".join(tiers)
        return await self._client.request("GET", "/api/v1/memory/search-index", params=params)

    async def search_timeline(
        self,
        query: str,
        *,
        limit: int = 20,
        min_importance: float = 0.0,
        tiers: list[MemoryTier] | None = None,
    ) -> dict[str, Any]:
        """Progressive retrieval: timeline-ordered results. GET /api/v1/memory/search-timeline"""
        params: dict[str, Any] = {
            "q": query,
            "limit": limit,
            "min_importance": min_importance,
        }
        if tiers:
            params["tiers"] = ",".join(tiers)
        return await self._client.request(
            "GET", "/api/v1/memory/search-timeline", params=params
        )

    async def list_entries(
        self,
        *,
        limit: int = 20,
        offset: int = 0,
        tier: MemoryTier | None = None,
    ) -> dict[str, Any]:
        """List memory entries. GET /api/v1/memory/entries"""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if tier:
            params["tier"] = tier
        return await self._client.request("GET", "/api/v1/memory/entries", params=params)

    async def get_viewer(self) -> dict[str, Any]:
        """Get memory viewer HTML interface. GET /api/v1/memory/viewer"""
        return await self._client.request("GET", "/api/v1/memory/viewer")

    # ===========================================================================
    # Advanced Operations
    # ===========================================================================

    async def compact(
        self, workspace_id: str | None = None
    ) -> dict[str, Any]:
        """Trigger memory compaction."""
        body: dict[str, Any] = {}
        if workspace_id:
            body["workspace_id"] = workspace_id
        return await self._client.request("POST", "/api/v1/memory/compact", json=body)

    async def get_context(
        self, workspace_id: str | None = None
    ) -> dict[str, Any]:
        """Get memory context for current session."""
        params: dict[str, Any] = {}
        if workspace_id:
            params["workspace_id"] = workspace_id
        return await self._client.request("GET", "/api/v1/memory/context", params=params)

    async def get_cross_debate(
        self, workspace_id: str | None = None
    ) -> dict[str, Any]:
        """Get cross-debate memory entries."""
        params: dict[str, Any] = {}
        if workspace_id:
            params["workspace_id"] = workspace_id
        return await self._client.request("GET", "/api/v1/memory/cross-debate", params=params)

    async def inject_cross_debate(
        self,
        debate_id: str,
        entries: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Inject cross-debate memories into a debate."""
        body: dict[str, Any] = {"debate_id": debate_id}
        if entries:
            body["entries"] = entries
        return await self._client.request(
            "POST", "/api/v1/memory/cross-debate/inject", json=body
        )

    async def export_memories(
        self,
        format: str = "json",
        workspace_id: str | None = None,
    ) -> dict[str, Any]:
        """Export memories."""
        body: dict[str, Any] = {"format": format}
        if workspace_id:
            body["workspace_id"] = workspace_id
        return await self._client.request("POST", "/api/v1/memory/export", json=body)

    async def import_memories(self, data: dict[str, Any]) -> dict[str, Any]:
        """Import memories from exported data."""
        return await self._client.request("POST", "/api/v1/memory/import", json=data)

    async def prune(
        self,
        workspace_id: str | None = None,
        max_age_days: int | None = None,
    ) -> dict[str, Any]:
        """Prune old or low-importance memories."""
        body: dict[str, Any] = {}
        if workspace_id:
            body["workspace_id"] = workspace_id
        if max_age_days is not None:
            body["max_age_days"] = max_age_days
        return await self._client.request("POST", "/api/v1/memory/prune", json=body)

    async def query(
        self,
        prompt: str,
        workspace_id: str | None = None,
    ) -> dict[str, Any]:
        """Query memories using natural language."""
        body: dict[str, Any] = {"prompt": prompt}
        if workspace_id:
            body["workspace_id"] = workspace_id
        return await self._client.request("POST", "/api/v1/memory/query", json=body)

    async def rebuild_index(
        self, workspace_id: str | None = None
    ) -> dict[str, Any]:
        """Rebuild the memory search index."""
        body: dict[str, Any] = {}
        if workspace_id:
            body["workspace_id"] = workspace_id
        return await self._client.request("POST", "/api/v1/memory/rebuild-index", json=body)

    async def semantic_search(
        self,
        query: str,
        *,
        limit: int = 20,
        min_similarity: float = 0.5,
    ) -> dict[str, Any]:
        """Perform semantic search across memories."""
        body: dict[str, Any] = {
            "query": query,
            "limit": limit,
            "min_similarity": min_similarity,
        }
        return await self._client.request("POST", "/api/v1/memory/semantic-search", json=body)

    async def list_snapshots(
        self, limit: int = 20, offset: int = 0
    ) -> dict[str, Any]:
        """List memory snapshots."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        return await self._client.request("GET", "/api/v1/memory/snapshots", params=params)

    async def get_snapshot(self, snapshot_id: str) -> dict[str, Any]:
        """Get a specific memory snapshot."""
        return await self._client.request("GET", f"/api/v1/memory/snapshots/{snapshot_id}")

    async def restore_snapshot(self, snapshot_id: str) -> dict[str, Any]:
        """Restore a memory snapshot."""
        return await self._client.request(
            "POST", f"/api/v1/memory/snapshots/{snapshot_id}/restore", json={}
        )

    async def sync_memories(
        self, workspace_id: str | None = None
    ) -> dict[str, Any]:
        """Sync memories across tiers."""
        body: dict[str, Any] = {}
        if workspace_id:
            body["workspace_id"] = workspace_id
        return await self._client.request("POST", "/api/v1/memory/sync", json=body)

    async def get_tier(self, tier: str) -> dict[str, Any]:
        """Get detailed information about a specific memory tier."""
        return await self._client.request("GET", f"/api/v1/memory/tier/{tier}")

    async def vacuum(self) -> dict[str, Any]:
        """Run vacuum to reclaim storage space."""
        return await self._client.request("POST", "/api/v1/memory/vacuum", json={})

    async def promote(
        self, key: str, target_tier: str | None = None
    ) -> dict[str, Any]:
        """Promote a memory entry to a higher tier."""
        body: dict[str, Any] = {}
        if target_tier:
            body["target_tier"] = target_tier
        return await self._client.request("POST", f"/api/v1/memory/{key}/promote", json=body)

    async def demote(
        self, key: str, target_tier: str | None = None
    ) -> dict[str, Any]:
        """Demote a memory entry to a lower tier."""
        body: dict[str, Any] = {}
        if target_tier:
            body["target_tier"] = target_tier
        return await self._client.request("POST", f"/api/v1/memory/{key}/demote", json=body)

    async def move(self, key: str, target_tier: str) -> dict[str, Any]:
        """Move a memory entry to a specific tier."""
        body: dict[str, Any] = {"target_tier": target_tier}
        return await self._client.request("POST", f"/api/v1/memory/{key}/move", json=body)

