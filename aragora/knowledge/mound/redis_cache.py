"""
Redis caching layer for Knowledge Mound.

Provides high-performance caching for queries, nodes, and culture patterns
to reduce load on the primary storage backend.

Requires: redis (aioredis)
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from aragora.knowledge.mound.types import (
        CultureProfile,
        KnowledgeItem,
        QueryResult,
    )

logger = logging.getLogger(__name__)


class RedisCache:
    """
    Redis caching layer for Knowledge Mound.

    Cache structure:
    - aragora:km:{workspace}:node:{node_id} -> JSON(KnowledgeItem)
    - aragora:km:{workspace}:query:{hash} -> JSON(QueryResult)
    - aragora:km:{workspace}:culture -> JSON(CultureProfile)
    - aragora:km:staleness:pending -> ZSET(node_id, staleness_score)
    - aragora:km:_entry_tracker -> ZSET(cache_key, access_timestamp) [LRU tracking]
    """

    def __init__(
        self,
        url: str,
        default_ttl: int = 300,  # 5 minutes
        culture_ttl: int = 3600,  # 1 hour
        prefix: str = "aragora:km",
        max_entries: int = 10_000,
        event_emitter: Any | None = None,
    ):
        """
        Initialize Redis cache.

        Args:
            url: Redis connection URL (redis://host:port)
            default_ttl: Default TTL for cached items in seconds
            culture_ttl: TTL for culture patterns in seconds
            prefix: Key prefix for all cached items
            max_entries: Maximum number of cached entries before LRU eviction
        """
        self._url = url
        self._default_ttl = default_ttl
        self._culture_ttl = culture_ttl
        self._prefix = prefix
        self._max_entries = max_entries
        self._client: Any | None = None
        self._connected = False
        self._event_emitter = event_emitter

    async def connect(self) -> None:
        """Connect to Redis."""
        if self._connected:
            return

        try:
            import redis.asyncio as redis

            self._client = redis.from_url(
                self._url,
                encoding="utf-8",
                decode_responses=True,
            )

            # Test connection
            await self._client.ping()
            self._connected = True
            logger.info("Redis cache connected: %s", self._url)

        except ImportError:
            raise ImportError("redis required for caching. Install with: pip install redis")
        except (OSError, ConnectionError, RuntimeError) as e:
            logger.error("Redis connection failed: %s", e)
            raise

    async def close(self) -> None:
        """Close Redis connection."""
        if self._client:
            await self._client.close()
            self._client = None
        self._connected = False

    def _ensure_connected(self) -> None:
        """Ensure Redis is connected."""
        if not self._connected or not self._client:
            raise RuntimeError("Redis not connected. Call connect() first.")

    # =========================================================================
    # Node Caching
    # =========================================================================

    async def get_node(self, node_id: str) -> KnowledgeItem | None:
        """Get a cached node."""
        self._ensure_connected()

        key = f"{self._prefix}:node:{node_id}"
        data = await self._client.get(key)

        if data:
            try:
                from aragora.knowledge.mound.types import KnowledgeItem

                await self._touch_entry(key)
                return KnowledgeItem.from_dict(json.loads(data))
            except (ValueError, KeyError, json.JSONDecodeError, TypeError) as e:
                logger.warning("Failed to deserialize cached node: %s", e)
                await self._client.delete(key)
                await self._untrack_entry(key)

        return None

    async def set_node(
        self,
        node_id: str,
        node: KnowledgeItem,
        ttl: int | None = None,
    ) -> None:
        """Cache a node."""
        self._ensure_connected()

        key = f"{self._prefix}:node:{node_id}"
        data = json.dumps(node.to_dict())

        await self._enforce_max_entries()
        await self._client.setex(key, ttl or self._default_ttl, data)
        await self._track_entry(key)

    async def invalidate_node(self, node_id: str) -> None:
        """Invalidate a cached node."""
        self._ensure_connected()

        key = f"{self._prefix}:node:{node_id}"
        await self._client.delete(key)
        await self._untrack_entry(key)

    async def invalidate_nodes(self, node_ids: list[str]) -> None:
        """Invalidate multiple cached nodes."""
        if not node_ids:
            return

        self._ensure_connected()

        keys = [f"{self._prefix}:node:{nid}" for nid in node_ids]
        await self._client.delete(*keys)
        await self._untrack_entries(keys)

    # =========================================================================
    # Query Caching
    # =========================================================================

    async def get_query(self, cache_key: str) -> QueryResult | None:
        """Get a cached query result."""
        self._ensure_connected()

        key = f"{self._prefix}:query:{self._hash_key(cache_key)}"
        data = await self._client.get(key)

        if data:
            try:
                from aragora.knowledge.mound.types import QueryResult, KnowledgeItem

                parsed = json.loads(data)
                await self._touch_entry(key)
                return QueryResult(
                    items=[KnowledgeItem.from_dict(i) for i in parsed["items"]],
                    total_count=parsed["total_count"],
                    query=parsed["query"],
                    execution_time_ms=parsed.get("execution_time_ms", 0),
                    sources_queried=[],
                )
            except (ValueError, KeyError, json.JSONDecodeError, TypeError) as e:
                logger.warning("Failed to deserialize cached query: %s", e)
                await self._client.delete(key)
                await self._untrack_entry(key)

        return None

    async def set_query(
        self,
        cache_key: str,
        result: QueryResult,
        ttl: int | None = None,
    ) -> None:
        """Cache a query result."""
        self._ensure_connected()

        key = f"{self._prefix}:query:{self._hash_key(cache_key)}"
        data = json.dumps(result.to_dict())

        await self._enforce_max_entries()
        # Shorter TTL for queries (1 minute default)
        await self._client.setex(key, ttl or 60, data)
        await self._track_entry(key)

    async def invalidate_queries(self, workspace_id: str) -> None:
        """Invalidate all cached queries for a workspace."""
        self._ensure_connected()

        # Use pattern matching to find and delete query keys
        pattern = f"{self._prefix}:query:*"

        cursor = 0
        while True:
            cursor, keys = await self._client.scan(cursor, match=pattern, count=100)
            if keys:
                await self._client.delete(*keys)
                await self._untrack_entries(keys)
            if cursor == 0:
                break

    # =========================================================================
    # Culture Caching
    # =========================================================================

    async def get_culture(self, workspace_id: str) -> CultureProfile | None:
        """Get cached culture profile."""
        self._ensure_connected()

        key = f"{self._prefix}:{workspace_id}:culture"
        data = await self._client.get(key)

        if data:
            try:
                from aragora.knowledge.mound.types import (
                    CultureProfile,
                    CulturePattern,
                    CulturePatternType,
                )

                parsed = json.loads(data)

                # Reconstruct patterns dict
                patterns: dict[CulturePatternType, list[CulturePattern]] = {}
                for type_str, pattern_list in parsed.get("patterns", {}).items():
                    pattern_type = CulturePatternType(type_str)
                    patterns[pattern_type] = [
                        CulturePattern(
                            id=p["id"],
                            workspace_id=p["workspace_id"],
                            pattern_type=pattern_type,
                            pattern_key=p["pattern_key"],
                            pattern_value=p["pattern_value"],
                            observation_count=p["observation_count"],
                            confidence=p["confidence"],
                            first_observed_at=datetime.fromisoformat(p["first_observed_at"]),
                            last_observed_at=datetime.fromisoformat(p["last_observed_at"]),
                            contributing_debates=p.get("contributing_debates", []),
                        )
                        for p in pattern_list
                    ]

                await self._touch_entry(key)
                return CultureProfile(
                    workspace_id=parsed["workspace_id"],
                    patterns=patterns,
                    generated_at=datetime.fromisoformat(parsed["generated_at"]),
                    total_observations=parsed.get("total_observations", 0),
                    dominant_traits=parsed.get("dominant_traits", {}),
                )
            except (ValueError, KeyError, json.JSONDecodeError, TypeError) as e:
                logger.warning("Failed to deserialize cached culture: %s", e)
                await self._client.delete(key)
                await self._untrack_entry(key)

        return None

    async def set_culture(
        self,
        workspace_id: str,
        profile: CultureProfile,
        ttl: int | None = None,
    ) -> None:
        """Cache a culture profile."""
        self._ensure_connected()

        key = f"{self._prefix}:{workspace_id}:culture"

        # Serialize patterns
        patterns_dict = {}
        for pattern_type, pattern_list in profile.patterns.items():
            patterns_dict[pattern_type.value] = [
                {
                    "id": p.id,
                    "workspace_id": p.workspace_id,
                    "pattern_key": p.pattern_key,
                    "pattern_value": p.pattern_value,
                    "observation_count": p.observation_count,
                    "confidence": p.confidence,
                    "first_observed_at": p.first_observed_at.isoformat(),
                    "last_observed_at": p.last_observed_at.isoformat(),
                    "contributing_debates": p.contributing_debates,
                }
                for p in pattern_list
            ]

        data = json.dumps(
            {
                "workspace_id": profile.workspace_id,
                "patterns": patterns_dict,
                "generated_at": profile.generated_at.isoformat(),
                "total_observations": profile.total_observations,
                "dominant_traits": profile.dominant_traits,
            }
        )

        await self._enforce_max_entries()
        await self._client.setex(key, ttl or self._culture_ttl, data)
        await self._track_entry(key)

    async def invalidate_culture(self, workspace_id: str) -> None:
        """Invalidate cached culture profile."""
        self._ensure_connected()

        key = f"{self._prefix}:{workspace_id}:culture"
        await self._client.delete(key)
        await self._untrack_entry(key)

    # =========================================================================
    # Staleness Tracking
    # =========================================================================

    async def add_stale_node(self, node_id: str, staleness_score: float) -> None:
        """Add a node to the staleness tracking set."""
        self._ensure_connected()

        key = f"{self._prefix}:staleness:pending"
        await self._client.zadd(key, {node_id: staleness_score})

    async def get_stale_nodes(self, limit: int = 100) -> list[tuple]:
        """Get nodes pending revalidation, ordered by staleness."""
        self._ensure_connected()

        key = f"{self._prefix}:staleness:pending"
        # Get highest staleness scores first
        results = await self._client.zrevrange(key, 0, limit - 1, withscores=True)

        return [(node_id, score) for node_id, score in results]

    async def remove_stale_node(self, node_id: str) -> None:
        """Remove a node from staleness tracking."""
        self._ensure_connected()

        key = f"{self._prefix}:staleness:pending"
        await self._client.zrem(key, node_id)

    # =========================================================================
    # Statistics
    # =========================================================================

    async def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        self._ensure_connected()

        info = await self._client.info("memory")

        # Count keys by type
        node_count = 0
        query_count = 0
        culture_count = 0

        async for key in self._client.scan_iter(f"{self._prefix}:*"):
            if ":node:" in key:
                node_count += 1
            elif ":query:" in key:
                query_count += 1
            elif ":culture" in key:
                culture_count += 1

        return {
            "used_memory": info.get("used_memory_human", "unknown"),
            "connected_clients": info.get("connected_clients", 0),
            "cached_nodes": node_count,
            "cached_queries": query_count,
            "cached_cultures": culture_count,
        }

    async def clear_all(self, workspace_id: str | None = None) -> int:
        """Clear all cached items for a workspace or all."""
        self._ensure_connected()

        if workspace_id:
            pattern = f"{self._prefix}:{workspace_id}:*"
        else:
            pattern = f"{self._prefix}:*"

        deleted = 0
        cursor = 0
        while True:
            cursor, keys = await self._client.scan(cursor, match=pattern, count=100)
            if keys:
                deleted += await self._client.delete(*keys)
                await self._untrack_entries(keys)
            if cursor == 0:
                break

        return deleted

    # =========================================================================
    # Cache Invalidation Bus Integration
    # =========================================================================

    async def subscribe_to_invalidation_bus(self) -> None:
        """
        Subscribe to the CacheInvalidationBus for event-driven cache updates.

        This enables automatic cache invalidation when knowledge is updated
        through the ResilientPostgresStore or any other component that
        publishes to the invalidation bus.
        """
        from aragora.knowledge.mound.resilience import (
            CacheInvalidationEvent,
            get_invalidation_bus,
        )

        bus = get_invalidation_bus()

        async def handle_invalidation(event: CacheInvalidationEvent) -> None:
            """Handle cache invalidation events."""
            try:
                if event.event_type == "node_updated":
                    if event.item_id:
                        await self.invalidate_node(event.item_id)
                    # Also invalidate related queries
                    await self.invalidate_queries(event.workspace_id)
                    logger.debug("Cache invalidated: node %s in %s", event.item_id, event.workspace_id)

                elif event.event_type == "node_deleted":
                    if event.item_id:
                        await self.invalidate_node(event.item_id)
                        await self.remove_stale_node(event.item_id)
                    await self.invalidate_queries(event.workspace_id)
                    logger.debug(
                        "Cache invalidated: deleted node %s in %s", event.item_id, event.workspace_id
                    )

                elif event.event_type == "query_invalidated":
                    await self.invalidate_queries(event.workspace_id)
                    logger.debug("Cache invalidated: queries in %s", event.workspace_id)

                elif event.event_type == "culture_updated":
                    await self.invalidate_culture(event.workspace_id)
                    logger.debug("Cache invalidated: culture in %s", event.workspace_id)

                # Emit KM_CACHE_INVALIDATED event
                self._emit_cache_invalidated(event.event_type, event.workspace_id, event.item_id)

            except (OSError, ConnectionError, RuntimeError) as e:
                logger.warning("Cache invalidation failed for event %s: %s", event.event_type, e)

        self._unsubscribe = bus.subscribe(handle_invalidation)
        logger.info("Redis cache subscribed to invalidation bus")

    def unsubscribe_from_invalidation_bus(self) -> None:
        """Unsubscribe from the CacheInvalidationBus."""
        if hasattr(self, "_unsubscribe") and self._unsubscribe:
            self._unsubscribe()
            self._unsubscribe = None
            logger.info("Redis cache unsubscribed from invalidation bus")

    def _emit_cache_invalidated(
        self, event_type: str, workspace_id: str, item_id: str | None
    ) -> None:
        """Emit KM_CACHE_INVALIDATED event for real-time monitoring."""
        if not self._event_emitter:
            return
        try:
            from aragora.events.types import StreamEvent, StreamEventType

            self._event_emitter.emit(StreamEvent(
                type=StreamEventType.KM_CACHE_INVALIDATED,
                data={
                    "event_type": event_type,
                    "workspace_id": workspace_id,
                    "item_id": item_id,
                },
            ))
        except (ImportError, AttributeError, TypeError):
            pass

    # =========================================================================
    # LRU Entry Tracking
    # =========================================================================

    @property
    def _tracker_key(self) -> str:
        """Key for the sorted set tracking all cached entries."""
        return f"{self._prefix}:_entry_tracker"

    async def _track_entry(self, cache_key: str) -> None:
        """Register a cache key in the LRU tracker with current timestamp."""
        await self._client.zadd(self._tracker_key, {cache_key: time.time()})

    async def _touch_entry(self, cache_key: str) -> None:
        """Update access time for a cache key (LRU refresh)."""
        score = await self._client.zscore(self._tracker_key, cache_key)
        if score is not None:
            await self._client.zadd(self._tracker_key, {cache_key: time.time()})

    async def _untrack_entry(self, cache_key: str) -> None:
        """Remove a cache key from the LRU tracker."""
        await self._client.zrem(self._tracker_key, cache_key)

    async def _untrack_entries(self, cache_keys: list[str]) -> None:
        """Remove multiple cache keys from the LRU tracker."""
        if cache_keys:
            await self._client.zrem(self._tracker_key, *cache_keys)

    async def _enforce_max_entries(self) -> int:
        """Evict oldest entries if cache exceeds max_entries.

        Returns:
            Number of entries evicted.
        """
        count = await self._client.zcard(self._tracker_key)
        if count < self._max_entries:
            return 0

        overage = count - self._max_entries + 1  # +1 to make room for the new entry
        # Get the oldest entries (lowest scores = oldest access times)
        victims = await self._client.zrange(self._tracker_key, 0, overage - 1)

        if not victims:
            return 0

        # Delete the actual cache keys
        await self._client.delete(*victims)
        # Remove from tracker
        await self._client.zremrangebyrank(self._tracker_key, 0, overage - 1)

        logger.debug(
            "LRU eviction: removed %s entries (max_entries=%s)", len(victims), self._max_entries
        )
        return len(victims)

    async def get_entry_count(self) -> int:
        """Get the current number of tracked cache entries."""
        self._ensure_connected()
        return await self._client.zcard(self._tracker_key)

    async def get_memory_stats(self) -> dict[str, Any]:
        """Get cache memory statistics including entry count and limits.

        Returns:
            Dict with entry_count, max_entries, utilization, and memory info.
        """
        self._ensure_connected()

        entry_count = await self._client.zcard(self._tracker_key)
        info = await self._client.info("memory")

        return {
            "entry_count": entry_count,
            "max_entries": self._max_entries,
            "utilization": entry_count / self._max_entries if self._max_entries > 0 else 0,
            "used_memory": info.get("used_memory_human", "unknown"),
            "used_memory_bytes": info.get("used_memory", 0),
        }

    # =========================================================================
    # Helpers
    # =========================================================================

    def _hash_key(self, key: str) -> str:
        """Hash a key for consistent sizing."""
        return hashlib.sha256(key.encode()).hexdigest()[:16]
