"""
Memory Namespace API

Provides methods for interacting with the memory system:
- Core CRUD operations (store, retrieve, update, delete)
- Continuum tier management (fast/medium/slow/glacial operations)
- Memory search/query (semantic search, filtered queries)
- Cross-debate memory (institutional knowledge)
- Memory export/import (backup/restore)
- Memory analytics/stats
- Context management
- Maintenance operations (prune, compact, sync)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import quote

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient

MemoryTier = Literal["fast", "medium", "slow", "glacial"]
ConflictResolution = Literal["latest_wins", "merge", "manual"]
SortOrder = Literal["asc", "desc"]
PressureLevel = Literal["low", "medium", "high", "critical"]


class MemoryAPI:
    """Synchronous Memory API."""

    def __init__(self, client: AragoraClient):
        self._client = client

    # ===========================================================================
    # Core CRUD Operations
    # ===========================================================================

    def store(
        self,
        key: str,
        value: Any,
        *,
        tier: MemoryTier | None = None,
        importance: float | None = None,
        tags: list[str] | None = None,
        ttl_seconds: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Store a value in memory.

        Args:
            key: Key for the memory entry
            value: Value to store
            tier: Target memory tier (fast, medium, slow, glacial)
            importance: Importance score (0.0-1.0)
            tags: Tags for categorization
            ttl_seconds: Time-to-live in seconds
            metadata: Additional metadata

        Returns:
            Dict with stored status and tier info
        """
        body: dict[str, Any] = {"key": key, "value": value}
        if tier:
            body["tier"] = tier
        if importance is not None:
            body["importance"] = importance
        if tags:
            body["tags"] = tags
        if ttl_seconds is not None:
            body["ttl_seconds"] = ttl_seconds
        if metadata:
            body["metadata"] = metadata
        return self._client.request("POST", "/api/v1/memory", json=body)

    def retrieve(
        self,
        key: str,
        *,
        tier: MemoryTier | None = None,
    ) -> dict[str, Any] | None:
        """
        Retrieve a value from memory by key.

        Args:
            key: The key to retrieve
            tier: Specific tier to look in (searches all if not specified)

        Returns:
            Dict with value, tier, and metadata, or None if not found
        """
        params: dict[str, Any] = {}
        if tier:
            params["tier"] = tier
        return self._client.request("GET", f"/api/v1/memory/{quote(key, safe='')}", params=params)

    def update(
        self,
        key: str,
        value: Any,
        *,
        tier: MemoryTier | None = None,
        merge: bool = False,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Update an existing memory entry.

        Args:
            key: The key of the entry to update
            value: The new value
            tier: Target tier (optional)
            merge: Whether to merge with existing value (default: replace)
            tags: Updated tags
            metadata: Updated metadata

        Returns:
            Dict with updated status and tier info
        """
        body: dict[str, Any] = {"value": value}
        if tier:
            body["tier"] = tier
        if merge:
            body["merge"] = merge
        if tags:
            body["tags"] = tags
        if metadata:
            body["metadata"] = metadata
        return self._client.request("PUT", f"/api/v1/memory/{quote(key, safe='')}", json=body)

    def delete(
        self,
        key: str,
        *,
        tier: MemoryTier | None = None,
    ) -> dict[str, Any]:
        """
        Delete a memory entry by key.

        Args:
            key: The key to delete
            tier: Specific tier to delete from

        Returns:
            Dict with deleted status
        """
        params: dict[str, Any] = {}
        if tier:
            params["tier"] = tier
        return self._client.request(
            "DELETE", f"/api/v1/memory/{quote(key, safe='')}", params=params
        )

    # ===========================================================================
    # Search and Query Operations
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

    def query(
        self,
        *,
        filter: dict[str, Any] | None = None,
        sort_by: str | None = None,
        sort_order: SortOrder = "desc",
        limit: int = 20,
        offset: int = 0,
        include_metadata: bool = True,
    ) -> dict[str, Any]:
        """
        Query memory entries with advanced filtering.

        Args:
            filter: Filter conditions (e.g., {"tags": ["important"], "tier": "slow"})
            sort_by: Field to sort by (e.g., "created_at", "importance")
            sort_order: Sort direction ("asc" or "desc")
            limit: Maximum number of results
            offset: Offset for pagination
            include_metadata: Whether to include metadata in results

        Returns:
            Dict with entries list and total count
        """
        body: dict[str, Any] = {
            "limit": limit,
            "offset": offset,
            "include_metadata": include_metadata,
        }
        if filter:
            body["filter"] = filter
        if sort_by:
            body["sort_by"] = sort_by
            body["sort_order"] = sort_order
        return self._client.request("POST", "/api/v1/memory/query", json=body)

    def semantic_search(
        self,
        query: str,
        *,
        tiers: list[MemoryTier] | None = None,
        limit: int = 10,
        min_similarity: float = 0.7,
        include_embeddings: bool = False,
    ) -> dict[str, Any]:
        """
        Perform semantic search across memory entries.

        Args:
            query: Natural language query
            tiers: Tiers to search (all if not specified)
            limit: Maximum results
            min_similarity: Minimum similarity threshold (0.0-1.0)
            include_embeddings: Whether to include embedding vectors

        Returns:
            Dict with entries and similarity scores
        """
        body: dict[str, Any] = {
            "query": query,
            "limit": limit,
            "min_similarity": min_similarity,
            "include_embeddings": include_embeddings,
        }
        if tiers:
            body["tiers"] = tiers
        return self._client.request("POST", "/api/v1/memory/semantic-search", json=body)

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

    def get_tier(
        self,
        tier: MemoryTier,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """
        Get entries from a specific tier.

        Args:
            tier: The memory tier (fast, medium, slow, glacial)
            limit: Maximum entries to return
            offset: Offset for pagination

        Returns:
            Dict with entries list and total count
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        return self._client.request("GET", f"/api/v1/memory/tier/{tier}", params=params)

    def move_tier(
        self,
        key: str,
        from_tier: MemoryTier,
        to_tier: MemoryTier,
    ) -> dict[str, Any]:
        """
        Move an entry between tiers.

        Args:
            key: The entry key
            from_tier: Source tier
            to_tier: Destination tier

        Returns:
            Dict with move status
        """
        body: dict[str, Any] = {"from_tier": from_tier, "to_tier": to_tier}
        return self._client.request("POST", f"/api/v1/memory/{quote(key, safe='')}/move", json=body)

    def promote(
        self,
        key: str,
        *,
        reason: str | None = None,
    ) -> dict[str, Any]:
        """
        Promote an entry to a faster tier.

        Args:
            key: The entry key
            reason: Optional reason for promotion

        Returns:
            Dict with promotion status and new tier
        """
        body: dict[str, Any] = {}
        if reason:
            body["reason"] = reason
        return self._client.request(
            "POST", f"/api/v1/memory/{quote(key, safe='')}/promote", json=body
        )

    def demote(
        self,
        key: str,
        *,
        reason: str | None = None,
    ) -> dict[str, Any]:
        """
        Demote an entry to a slower tier.

        Args:
            key: The entry key
            reason: Optional reason for demotion

        Returns:
            Dict with demotion status and new tier
        """
        body: dict[str, Any] = {}
        if reason:
            body["reason"] = reason
        return self._client.request(
            "POST", f"/api/v1/memory/{quote(key, safe='')}/demote", json=body
        )

    # ===========================================================================
    # Continuum Operations
    # ===========================================================================

    def store_to_continuum(
        self,
        content: str,
        *,
        importance: float | None = None,
        tags: list[str] | None = None,
        source: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Store content in the continuum memory system.

        Args:
            content: Content to store
            importance: Importance score (0.0-1.0)
            tags: Tags for categorization
            source: Source identifier
            metadata: Additional metadata

        Returns:
            Dict with stored entry ID and assigned tier
        """
        body: dict[str, Any] = {"content": content}
        if importance is not None:
            body["importance"] = importance
        if tags:
            body["tags"] = tags
        if source:
            body["source"] = source
        if metadata:
            body["metadata"] = metadata
        return self._client.request("POST", "/api/v1/memory/continuum", json=body)

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

    def continuum_stats(self) -> dict[str, Any]:
        """
        Get continuum memory statistics.

        Returns:
            Dict with continuum-specific statistics
        """
        return self._client.request("GET", "/api/v1/memory/continuum/stats")

    def consolidate(self) -> dict[str, Any]:
        """
        Consolidate memory by archiving old entries.

        Returns:
            Dict with consolidation status
        """
        return self._client.request("POST", "/api/v1/memory/consolidate", json={})

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
    # Context Management
    # ===========================================================================

    def get_context(
        self,
        context_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Get the current memory context.

        Args:
            context_id: Optional context ID (defaults to current session)

        Returns:
            Dict with context data, created_at, expires_at
        """
        params: dict[str, Any] = {}
        if context_id:
            params["context_id"] = context_id
        return self._client.request("GET", "/api/v1/memory/context", params=params)

    def set_context(
        self,
        data: dict[str, Any],
        *,
        context_id: str | None = None,
        ttl_seconds: int | None = None,
    ) -> dict[str, Any]:
        """
        Set or update the memory context.

        Args:
            data: Context data to set
            context_id: Optional context ID
            ttl_seconds: Time-to-live for the context

        Returns:
            Dict with context_id and updated data
        """
        body: dict[str, Any] = {"data": data}
        if context_id:
            body["context_id"] = context_id
        if ttl_seconds is not None:
            body["ttl_seconds"] = ttl_seconds
        return self._client.request("POST", "/api/v1/memory/context", json=body)

    def clear_context(
        self,
        context_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Clear the memory context.

        Args:
            context_id: Context ID to clear (current session if not specified)

        Returns:
            Dict with cleared status
        """
        params: dict[str, Any] = {}
        if context_id:
            params["context_id"] = context_id
        return self._client.request("DELETE", "/api/v1/memory/context", params=params)

    # ===========================================================================
    # Cross-Debate Memory (Institutional Knowledge)
    # ===========================================================================

    def get_cross_debate(
        self,
        *,
        topic: str | None = None,
        limit: int = 10,
        min_relevance: float = 0.5,
    ) -> dict[str, Any]:
        """
        Get cross-debate institutional knowledge.

        Args:
            topic: Topic to retrieve knowledge for
            limit: Maximum entries
            min_relevance: Minimum relevance threshold

        Returns:
            Dict with institutional knowledge entries
        """
        params: dict[str, Any] = {"limit": limit, "min_relevance": min_relevance}
        if topic:
            params["topic"] = topic
        return self._client.request("GET", "/api/v1/memory/cross-debate", params=params)

    def store_cross_debate(
        self,
        content: str,
        *,
        debate_id: str,
        topic: str | None = None,
        conclusion: str | None = None,
        confidence: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Store cross-debate knowledge from a debate outcome.

        Args:
            content: Knowledge content
            debate_id: Source debate ID
            topic: Topic classification
            conclusion: Debate conclusion
            confidence: Confidence level
            metadata: Additional metadata

        Returns:
            Dict with stored entry ID
        """
        body: dict[str, Any] = {"content": content, "debate_id": debate_id}
        if topic:
            body["topic"] = topic
        if conclusion:
            body["conclusion"] = conclusion
        if confidence is not None:
            body["confidence"] = confidence
        if metadata:
            body["metadata"] = metadata
        return self._client.request("POST", "/api/v1/memory/cross-debate", json=body)

    def inject_institutional(
        self,
        debate_id: str,
        *,
        topic: str | None = None,
        max_entries: int = 5,
    ) -> dict[str, Any]:
        """
        Inject institutional knowledge into a debate.

        Args:
            debate_id: Target debate ID
            topic: Topic to filter knowledge by
            max_entries: Maximum entries to inject

        Returns:
            Dict with injected entries count
        """
        body: dict[str, Any] = {"debate_id": debate_id, "max_entries": max_entries}
        if topic:
            body["topic"] = topic
        return self._client.request("POST", "/api/v1/memory/cross-debate/inject", json=body)

    # ===========================================================================
    # Export/Import (Backup/Restore)
    # ===========================================================================

    def export_memory(
        self,
        *,
        tiers: list[MemoryTier] | None = None,
        tags: list[str] | None = None,
        format: str = "json",
        include_metadata: bool = True,
    ) -> dict[str, Any]:
        """
        Export memory entries for backup.

        Args:
            tiers: Tiers to export (all if not specified)
            tags: Filter by tags
            format: Export format (json, msgpack)
            include_metadata: Include entry metadata

        Returns:
            Dict with export data or download URL
        """
        body: dict[str, Any] = {
            "format": format,
            "include_metadata": include_metadata,
        }
        if tiers:
            body["tiers"] = tiers
        if tags:
            body["tags"] = tags
        return self._client.request("POST", "/api/v1/memory/export", json=body)

    def import_memory(
        self,
        data: dict[str, Any] | list[dict[str, Any]],
        *,
        overwrite: bool = False,
        target_tier: MemoryTier | None = None,
    ) -> dict[str, Any]:
        """
        Import memory entries from backup.

        Args:
            data: Memory data to import
            overwrite: Whether to overwrite existing entries
            target_tier: Force all entries to specific tier

        Returns:
            Dict with import status and counts
        """
        body: dict[str, Any] = {"data": data, "overwrite": overwrite}
        if target_tier:
            body["target_tier"] = target_tier
        return self._client.request("POST", "/api/v1/memory/import", json=body)

    def create_snapshot(
        self,
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> dict[str, Any]:
        """
        Create a point-in-time snapshot of memory.

        Args:
            name: Snapshot name
            description: Snapshot description

        Returns:
            Dict with snapshot ID and metadata
        """
        body: dict[str, Any] = {}
        if name:
            body["name"] = name
        if description:
            body["description"] = description
        return self._client.request("POST", "/api/v1/memory/snapshots", json=body)

    def list_snapshots(
        self,
        *,
        limit: int = 20,
        offset: int = 0,
    ) -> dict[str, Any]:
        """
        List available memory snapshots.

        Args:
            limit: Maximum snapshots to return
            offset: Offset for pagination

        Returns:
            Dict with snapshots list
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        return self._client.request("GET", "/api/v1/memory/snapshots", params=params)

    def restore_snapshot(
        self,
        snapshot_id: str,
        *,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        """
        Restore memory from a snapshot.

        Args:
            snapshot_id: Snapshot ID to restore
            overwrite: Whether to overwrite current state

        Returns:
            Dict with restore status
        """
        body: dict[str, Any] = {"overwrite": overwrite}
        return self._client.request(
            "POST", f"/api/v1/memory/snapshots/{snapshot_id}/restore", json=body
        )

    def delete_snapshot(
        self,
        snapshot_id: str,
    ) -> dict[str, Any]:
        """
        Delete a memory snapshot.

        Args:
            snapshot_id: Snapshot ID to delete

        Returns:
            Dict with deletion status
        """
        return self._client.request("DELETE", f"/api/v1/memory/snapshots/{snapshot_id}")

    # ===========================================================================
    # Maintenance Operations
    # ===========================================================================

    def prune(
        self,
        *,
        older_than_days: int | None = None,
        min_importance: float | None = None,
        tiers: list[MemoryTier] | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """
        Prune old or low-importance memory entries.

        Args:
            older_than_days: Prune entries older than this
            min_importance: Prune entries below this importance
            tiers: Tiers to prune from
            dry_run: If True, only simulate pruning

        Returns:
            Dict with pruned count and freed bytes
        """
        body: dict[str, Any] = {"dry_run": dry_run}
        if older_than_days is not None:
            body["older_than_days"] = older_than_days
        if min_importance is not None:
            body["min_importance"] = min_importance
        if tiers:
            body["tiers"] = tiers
        return self._client.request("POST", "/api/v1/memory/prune", json=body)

    def compact(
        self,
        *,
        tier: MemoryTier | None = None,
        merge_threshold: float = 0.9,
    ) -> dict[str, Any]:
        """
        Compact memory storage by merging related entries.

        Args:
            tier: Specific tier to compact
            merge_threshold: Similarity threshold for merging

        Returns:
            Dict with compacted status and space saved
        """
        body: dict[str, Any] = {"merge_threshold": merge_threshold}
        if tier:
            body["tier"] = tier
        return self._client.request("POST", "/api/v1/memory/compact", json=body)

    def sync(
        self,
        *,
        target: str | None = None,
        conflict_resolution: ConflictResolution = "latest_wins",
        tiers: list[MemoryTier] | None = None,
    ) -> dict[str, Any]:
        """
        Synchronize memory across distributed systems.

        Args:
            target: Sync target (e.g., "all", specific node)
            conflict_resolution: How to resolve conflicts
            tiers: Tiers to sync

        Returns:
            Dict with sync status and conflict info
        """
        body: dict[str, Any] = {"conflict_resolution": conflict_resolution}
        if target:
            body["target"] = target
        if tiers:
            body["tiers"] = tiers
        return self._client.request("POST", "/api/v1/memory/sync", json=body)

    def vacuum(self) -> dict[str, Any]:
        """
        Run vacuum operation to reclaim storage space.

        Returns:
            Dict with vacuum status and space reclaimed
        """
        return self._client.request("POST", "/api/v1/memory/vacuum", json={})

    def rebuild_index(
        self,
        *,
        tier: MemoryTier | None = None,
    ) -> dict[str, Any]:
        """
        Rebuild memory search indices.

        Args:
            tier: Specific tier to reindex (all if not specified)

        Returns:
            Dict with reindex status
        """
        body: dict[str, Any] = {}
        if tier:
            body["tier"] = tier
        return self._client.request("POST", "/api/v1/memory/rebuild-index", json=body)


class AsyncMemoryAPI:
    """Asynchronous Memory API."""

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    # ===========================================================================
    # Core CRUD Operations
    # ===========================================================================

    async def store(
        self,
        key: str,
        value: Any,
        *,
        tier: MemoryTier | None = None,
        importance: float | None = None,
        tags: list[str] | None = None,
        ttl_seconds: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Store a value in memory.

        Args:
            key: Key for the memory entry
            value: Value to store
            tier: Target memory tier (fast, medium, slow, glacial)
            importance: Importance score (0.0-1.0)
            tags: Tags for categorization
            ttl_seconds: Time-to-live in seconds
            metadata: Additional metadata

        Returns:
            Dict with stored status and tier info
        """
        body: dict[str, Any] = {"key": key, "value": value}
        if tier:
            body["tier"] = tier
        if importance is not None:
            body["importance"] = importance
        if tags:
            body["tags"] = tags
        if ttl_seconds is not None:
            body["ttl_seconds"] = ttl_seconds
        if metadata:
            body["metadata"] = metadata
        return await self._client.request("POST", "/api/v1/memory", json=body)

    async def retrieve(
        self,
        key: str,
        *,
        tier: MemoryTier | None = None,
    ) -> dict[str, Any] | None:
        """
        Retrieve a value from memory by key.

        Args:
            key: The key to retrieve
            tier: Specific tier to look in (searches all if not specified)

        Returns:
            Dict with value, tier, and metadata, or None if not found
        """
        params: dict[str, Any] = {}
        if tier:
            params["tier"] = tier
        return await self._client.request(
            "GET", f"/api/v1/memory/{quote(key, safe='')}", params=params
        )

    async def update(
        self,
        key: str,
        value: Any,
        *,
        tier: MemoryTier | None = None,
        merge: bool = False,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Update an existing memory entry.

        Args:
            key: The key of the entry to update
            value: The new value
            tier: Target tier (optional)
            merge: Whether to merge with existing value (default: replace)
            tags: Updated tags
            metadata: Updated metadata

        Returns:
            Dict with updated status and tier info
        """
        body: dict[str, Any] = {"value": value}
        if tier:
            body["tier"] = tier
        if merge:
            body["merge"] = merge
        if tags:
            body["tags"] = tags
        if metadata:
            body["metadata"] = metadata
        return await self._client.request("PUT", f"/api/v1/memory/{quote(key, safe='')}", json=body)

    async def delete(
        self,
        key: str,
        *,
        tier: MemoryTier | None = None,
    ) -> dict[str, Any]:
        """
        Delete a memory entry by key.

        Args:
            key: The key to delete
            tier: Specific tier to delete from

        Returns:
            Dict with deleted status
        """
        params: dict[str, Any] = {}
        if tier:
            params["tier"] = tier
        return await self._client.request(
            "DELETE", f"/api/v1/memory/{quote(key, safe='')}", params=params
        )

    # ===========================================================================
    # Search and Query Operations
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

    async def query(
        self,
        *,
        filter: dict[str, Any] | None = None,
        sort_by: str | None = None,
        sort_order: SortOrder = "desc",
        limit: int = 20,
        offset: int = 0,
        include_metadata: bool = True,
    ) -> dict[str, Any]:
        """
        Query memory entries with advanced filtering.

        Args:
            filter: Filter conditions (e.g., {"tags": ["important"], "tier": "slow"})
            sort_by: Field to sort by (e.g., "created_at", "importance")
            sort_order: Sort direction ("asc" or "desc")
            limit: Maximum number of results
            offset: Offset for pagination
            include_metadata: Whether to include metadata in results

        Returns:
            Dict with entries list and total count
        """
        body: dict[str, Any] = {
            "limit": limit,
            "offset": offset,
            "include_metadata": include_metadata,
        }
        if filter:
            body["filter"] = filter
        if sort_by:
            body["sort_by"] = sort_by
            body["sort_order"] = sort_order
        return await self._client.request("POST", "/api/v1/memory/query", json=body)

    async def semantic_search(
        self,
        query: str,
        *,
        tiers: list[MemoryTier] | None = None,
        limit: int = 10,
        min_similarity: float = 0.7,
        include_embeddings: bool = False,
    ) -> dict[str, Any]:
        """
        Perform semantic search across memory entries.

        Args:
            query: Natural language query
            tiers: Tiers to search (all if not specified)
            limit: Maximum results
            min_similarity: Minimum similarity threshold (0.0-1.0)
            include_embeddings: Whether to include embedding vectors

        Returns:
            Dict with entries and similarity scores
        """
        body: dict[str, Any] = {
            "query": query,
            "limit": limit,
            "min_similarity": min_similarity,
            "include_embeddings": include_embeddings,
        }
        if tiers:
            body["tiers"] = tiers
        return await self._client.request("POST", "/api/v1/memory/semantic-search", json=body)

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

    async def get_tier(
        self,
        tier: MemoryTier,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """
        Get entries from a specific tier.

        Args:
            tier: The memory tier (fast, medium, slow, glacial)
            limit: Maximum entries to return
            offset: Offset for pagination

        Returns:
            Dict with entries list and total count
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        return await self._client.request("GET", f"/api/v1/memory/tier/{tier}", params=params)

    async def move_tier(
        self,
        key: str,
        from_tier: MemoryTier,
        to_tier: MemoryTier,
    ) -> dict[str, Any]:
        """
        Move an entry between tiers.

        Args:
            key: The entry key
            from_tier: Source tier
            to_tier: Destination tier

        Returns:
            Dict with move status
        """
        body: dict[str, Any] = {"from_tier": from_tier, "to_tier": to_tier}
        return await self._client.request(
            "POST", f"/api/v1/memory/{quote(key, safe='')}/move", json=body
        )

    async def promote(
        self,
        key: str,
        *,
        reason: str | None = None,
    ) -> dict[str, Any]:
        """
        Promote an entry to a faster tier.

        Args:
            key: The entry key
            reason: Optional reason for promotion

        Returns:
            Dict with promotion status and new tier
        """
        body: dict[str, Any] = {}
        if reason:
            body["reason"] = reason
        return await self._client.request(
            "POST", f"/api/v1/memory/{quote(key, safe='')}/promote", json=body
        )

    async def demote(
        self,
        key: str,
        *,
        reason: str | None = None,
    ) -> dict[str, Any]:
        """
        Demote an entry to a slower tier.

        Args:
            key: The entry key
            reason: Optional reason for demotion

        Returns:
            Dict with demotion status and new tier
        """
        body: dict[str, Any] = {}
        if reason:
            body["reason"] = reason
        return await self._client.request(
            "POST", f"/api/v1/memory/{quote(key, safe='')}/demote", json=body
        )

    # ===========================================================================
    # Continuum Operations
    # ===========================================================================

    async def store_to_continuum(
        self,
        content: str,
        *,
        importance: float | None = None,
        tags: list[str] | None = None,
        source: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Store content in the continuum memory system.

        Args:
            content: Content to store
            importance: Importance score (0.0-1.0)
            tags: Tags for categorization
            source: Source identifier
            metadata: Additional metadata

        Returns:
            Dict with stored entry ID and assigned tier
        """
        body: dict[str, Any] = {"content": content}
        if importance is not None:
            body["importance"] = importance
        if tags:
            body["tags"] = tags
        if source:
            body["source"] = source
        if metadata:
            body["metadata"] = metadata
        return await self._client.request("POST", "/api/v1/memory/continuum", json=body)

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

    async def continuum_stats(self) -> dict[str, Any]:
        """
        Get continuum memory statistics.

        Returns:
            Dict with continuum-specific statistics
        """
        return await self._client.request("GET", "/api/v1/memory/continuum/stats")

    async def consolidate(self) -> dict[str, Any]:
        """
        Consolidate memory by archiving old entries.

        Returns:
            Dict with consolidation status
        """
        return await self._client.request("POST", "/api/v1/memory/consolidate", json={})

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
    # Context Management
    # ===========================================================================

    async def get_context(
        self,
        context_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Get the current memory context.

        Args:
            context_id: Optional context ID (defaults to current session)

        Returns:
            Dict with context data, created_at, expires_at
        """
        params: dict[str, Any] = {}
        if context_id:
            params["context_id"] = context_id
        return await self._client.request("GET", "/api/v1/memory/context", params=params)

    async def set_context(
        self,
        data: dict[str, Any],
        *,
        context_id: str | None = None,
        ttl_seconds: int | None = None,
    ) -> dict[str, Any]:
        """
        Set or update the memory context.

        Args:
            data: Context data to set
            context_id: Optional context ID
            ttl_seconds: Time-to-live for the context

        Returns:
            Dict with context_id and updated data
        """
        body: dict[str, Any] = {"data": data}
        if context_id:
            body["context_id"] = context_id
        if ttl_seconds is not None:
            body["ttl_seconds"] = ttl_seconds
        return await self._client.request("POST", "/api/v1/memory/context", json=body)

    async def clear_context(
        self,
        context_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Clear the memory context.

        Args:
            context_id: Context ID to clear (current session if not specified)

        Returns:
            Dict with cleared status
        """
        params: dict[str, Any] = {}
        if context_id:
            params["context_id"] = context_id
        return await self._client.request("DELETE", "/api/v1/memory/context", params=params)

    # ===========================================================================
    # Cross-Debate Memory (Institutional Knowledge)
    # ===========================================================================

    async def get_cross_debate(
        self,
        *,
        topic: str | None = None,
        limit: int = 10,
        min_relevance: float = 0.5,
    ) -> dict[str, Any]:
        """
        Get cross-debate institutional knowledge.

        Args:
            topic: Topic to retrieve knowledge for
            limit: Maximum entries
            min_relevance: Minimum relevance threshold

        Returns:
            Dict with institutional knowledge entries
        """
        params: dict[str, Any] = {"limit": limit, "min_relevance": min_relevance}
        if topic:
            params["topic"] = topic
        return await self._client.request("GET", "/api/v1/memory/cross-debate", params=params)

    async def store_cross_debate(
        self,
        content: str,
        *,
        debate_id: str,
        topic: str | None = None,
        conclusion: str | None = None,
        confidence: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Store cross-debate knowledge from a debate outcome.

        Args:
            content: Knowledge content
            debate_id: Source debate ID
            topic: Topic classification
            conclusion: Debate conclusion
            confidence: Confidence level
            metadata: Additional metadata

        Returns:
            Dict with stored entry ID
        """
        body: dict[str, Any] = {"content": content, "debate_id": debate_id}
        if topic:
            body["topic"] = topic
        if conclusion:
            body["conclusion"] = conclusion
        if confidence is not None:
            body["confidence"] = confidence
        if metadata:
            body["metadata"] = metadata
        return await self._client.request("POST", "/api/v1/memory/cross-debate", json=body)

    async def inject_institutional(
        self,
        debate_id: str,
        *,
        topic: str | None = None,
        max_entries: int = 5,
    ) -> dict[str, Any]:
        """
        Inject institutional knowledge into a debate.

        Args:
            debate_id: Target debate ID
            topic: Topic to filter knowledge by
            max_entries: Maximum entries to inject

        Returns:
            Dict with injected entries count
        """
        body: dict[str, Any] = {"debate_id": debate_id, "max_entries": max_entries}
        if topic:
            body["topic"] = topic
        return await self._client.request("POST", "/api/v1/memory/cross-debate/inject", json=body)

    # ===========================================================================
    # Export/Import (Backup/Restore)
    # ===========================================================================

    async def export_memory(
        self,
        *,
        tiers: list[MemoryTier] | None = None,
        tags: list[str] | None = None,
        format: str = "json",
        include_metadata: bool = True,
    ) -> dict[str, Any]:
        """
        Export memory entries for backup.

        Args:
            tiers: Tiers to export (all if not specified)
            tags: Filter by tags
            format: Export format (json, msgpack)
            include_metadata: Include entry metadata

        Returns:
            Dict with export data or download URL
        """
        body: dict[str, Any] = {
            "format": format,
            "include_metadata": include_metadata,
        }
        if tiers:
            body["tiers"] = tiers
        if tags:
            body["tags"] = tags
        return await self._client.request("POST", "/api/v1/memory/export", json=body)

    async def import_memory(
        self,
        data: dict[str, Any] | list[dict[str, Any]],
        *,
        overwrite: bool = False,
        target_tier: MemoryTier | None = None,
    ) -> dict[str, Any]:
        """
        Import memory entries from backup.

        Args:
            data: Memory data to import
            overwrite: Whether to overwrite existing entries
            target_tier: Force all entries to specific tier

        Returns:
            Dict with import status and counts
        """
        body: dict[str, Any] = {"data": data, "overwrite": overwrite}
        if target_tier:
            body["target_tier"] = target_tier
        return await self._client.request("POST", "/api/v1/memory/import", json=body)

    async def create_snapshot(
        self,
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> dict[str, Any]:
        """
        Create a point-in-time snapshot of memory.

        Args:
            name: Snapshot name
            description: Snapshot description

        Returns:
            Dict with snapshot ID and metadata
        """
        body: dict[str, Any] = {}
        if name:
            body["name"] = name
        if description:
            body["description"] = description
        return await self._client.request("POST", "/api/v1/memory/snapshots", json=body)

    async def list_snapshots(
        self,
        *,
        limit: int = 20,
        offset: int = 0,
    ) -> dict[str, Any]:
        """
        List available memory snapshots.

        Args:
            limit: Maximum snapshots to return
            offset: Offset for pagination

        Returns:
            Dict with snapshots list
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        return await self._client.request("GET", "/api/v1/memory/snapshots", params=params)

    async def restore_snapshot(
        self,
        snapshot_id: str,
        *,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        """
        Restore memory from a snapshot.

        Args:
            snapshot_id: Snapshot ID to restore
            overwrite: Whether to overwrite current state

        Returns:
            Dict with restore status
        """
        body: dict[str, Any] = {"overwrite": overwrite}
        return await self._client.request(
            "POST", f"/api/v1/memory/snapshots/{snapshot_id}/restore", json=body
        )

    async def delete_snapshot(
        self,
        snapshot_id: str,
    ) -> dict[str, Any]:
        """
        Delete a memory snapshot.

        Args:
            snapshot_id: Snapshot ID to delete

        Returns:
            Dict with deletion status
        """
        return await self._client.request("DELETE", f"/api/v1/memory/snapshots/{snapshot_id}")

    # ===========================================================================
    # Maintenance Operations
    # ===========================================================================

    async def prune(
        self,
        *,
        older_than_days: int | None = None,
        min_importance: float | None = None,
        tiers: list[MemoryTier] | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """
        Prune old or low-importance memory entries.

        Args:
            older_than_days: Prune entries older than this
            min_importance: Prune entries below this importance
            tiers: Tiers to prune from
            dry_run: If True, only simulate pruning

        Returns:
            Dict with pruned count and freed bytes
        """
        body: dict[str, Any] = {"dry_run": dry_run}
        if older_than_days is not None:
            body["older_than_days"] = older_than_days
        if min_importance is not None:
            body["min_importance"] = min_importance
        if tiers:
            body["tiers"] = tiers
        return await self._client.request("POST", "/api/v1/memory/prune", json=body)

    async def compact(
        self,
        *,
        tier: MemoryTier | None = None,
        merge_threshold: float = 0.9,
    ) -> dict[str, Any]:
        """
        Compact memory storage by merging related entries.

        Args:
            tier: Specific tier to compact
            merge_threshold: Similarity threshold for merging

        Returns:
            Dict with compacted status and space saved
        """
        body: dict[str, Any] = {"merge_threshold": merge_threshold}
        if tier:
            body["tier"] = tier
        return await self._client.request("POST", "/api/v1/memory/compact", json=body)

    async def sync(
        self,
        *,
        target: str | None = None,
        conflict_resolution: ConflictResolution = "latest_wins",
        tiers: list[MemoryTier] | None = None,
    ) -> dict[str, Any]:
        """
        Synchronize memory across distributed systems.

        Args:
            target: Sync target (e.g., "all", specific node)
            conflict_resolution: How to resolve conflicts
            tiers: Tiers to sync

        Returns:
            Dict with sync status and conflict info
        """
        body: dict[str, Any] = {"conflict_resolution": conflict_resolution}
        if target:
            body["target"] = target
        if tiers:
            body["tiers"] = tiers
        return await self._client.request("POST", "/api/v1/memory/sync", json=body)

    async def vacuum(self) -> dict[str, Any]:
        """
        Run vacuum operation to reclaim storage space.

        Returns:
            Dict with vacuum status and space reclaimed
        """
        return await self._client.request("POST", "/api/v1/memory/vacuum", json={})

    async def rebuild_index(
        self,
        *,
        tier: MemoryTier | None = None,
    ) -> dict[str, Any]:
        """
        Rebuild memory search indices.

        Args:
            tier: Specific tier to reindex (all if not specified)

        Returns:
            Dict with reindex status
        """
        body: dict[str, Any] = {}
        if tier:
            body["tier"] = tier
        return await self._client.request("POST", "/api/v1/memory/rebuild-index", json=body)
