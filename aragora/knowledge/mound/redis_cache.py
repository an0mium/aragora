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
from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING

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
    """

    def __init__(
        self,
        url: str,
        default_ttl: int = 300,  # 5 minutes
        culture_ttl: int = 3600,  # 1 hour
        prefix: str = "aragora:km",
    ):
        """
        Initialize Redis cache.

        Args:
            url: Redis connection URL (redis://host:port)
            default_ttl: Default TTL for cached items in seconds
            culture_ttl: TTL for culture patterns in seconds
            prefix: Key prefix for all cached items
        """
        self._url = url
        self._default_ttl = default_ttl
        self._culture_ttl = culture_ttl
        self._prefix = prefix
        self._client: Optional[Any] = None
        self._connected = False

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
            logger.info(f"Redis cache connected: {self._url}")

        except ImportError:
            raise ImportError("redis required for caching. Install with: pip install redis")
        except Exception as e:
            logger.error(f"Redis connection failed: {e}")
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

    async def get_node(self, node_id: str) -> Optional["KnowledgeItem"]:
        """Get a cached node."""
        self._ensure_connected()

        key = f"{self._prefix}:node:{node_id}"
        data = await self._client.get(key)

        if data:
            try:
                from aragora.knowledge.mound.types import KnowledgeItem

                return KnowledgeItem.from_dict(json.loads(data))
            except Exception as e:
                logger.warning(f"Failed to deserialize cached node: {e}")
                await self._client.delete(key)

        return None

    async def set_node(
        self,
        node_id: str,
        node: "KnowledgeItem",
        ttl: Optional[int] = None,
    ) -> None:
        """Cache a node."""
        self._ensure_connected()

        key = f"{self._prefix}:node:{node_id}"
        data = json.dumps(node.to_dict())

        await self._client.setex(key, ttl or self._default_ttl, data)

    async def invalidate_node(self, node_id: str) -> None:
        """Invalidate a cached node."""
        self._ensure_connected()

        key = f"{self._prefix}:node:{node_id}"
        await self._client.delete(key)

    async def invalidate_nodes(self, node_ids: List[str]) -> None:
        """Invalidate multiple cached nodes."""
        if not node_ids:
            return

        self._ensure_connected()

        keys = [f"{self._prefix}:node:{nid}" for nid in node_ids]
        await self._client.delete(*keys)

    # =========================================================================
    # Query Caching
    # =========================================================================

    async def get_query(self, cache_key: str) -> Optional["QueryResult"]:
        """Get a cached query result."""
        self._ensure_connected()

        key = f"{self._prefix}:query:{self._hash_key(cache_key)}"
        data = await self._client.get(key)

        if data:
            try:
                from aragora.knowledge.mound.types import QueryResult, KnowledgeItem

                parsed = json.loads(data)
                return QueryResult(
                    items=[KnowledgeItem.from_dict(i) for i in parsed["items"]],
                    total_count=parsed["total_count"],
                    query=parsed["query"],
                    execution_time_ms=parsed.get("execution_time_ms", 0),
                    sources_queried=[],
                )
            except Exception as e:
                logger.warning(f"Failed to deserialize cached query: {e}")
                await self._client.delete(key)

        return None

    async def set_query(
        self,
        cache_key: str,
        result: "QueryResult",
        ttl: Optional[int] = None,
    ) -> None:
        """Cache a query result."""
        self._ensure_connected()

        key = f"{self._prefix}:query:{self._hash_key(cache_key)}"
        data = json.dumps(result.to_dict())

        # Shorter TTL for queries (1 minute default)
        await self._client.setex(key, ttl or 60, data)

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
            if cursor == 0:
                break

    # =========================================================================
    # Culture Caching
    # =========================================================================

    async def get_culture(self, workspace_id: str) -> Optional["CultureProfile"]:
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
                patterns: Dict[CulturePatternType, List[CulturePattern]] = {}
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

                return CultureProfile(
                    workspace_id=parsed["workspace_id"],
                    patterns=patterns,
                    generated_at=datetime.fromisoformat(parsed["generated_at"]),
                    total_observations=parsed.get("total_observations", 0),
                    dominant_traits=parsed.get("dominant_traits", {}),
                )
            except Exception as e:
                logger.warning(f"Failed to deserialize cached culture: {e}")
                await self._client.delete(key)

        return None

    async def set_culture(
        self,
        workspace_id: str,
        profile: "CultureProfile",
        ttl: Optional[int] = None,
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

        await self._client.setex(key, ttl or self._culture_ttl, data)

    async def invalidate_culture(self, workspace_id: str) -> None:
        """Invalidate cached culture profile."""
        self._ensure_connected()

        key = f"{self._prefix}:{workspace_id}:culture"
        await self._client.delete(key)

    # =========================================================================
    # Staleness Tracking
    # =========================================================================

    async def add_stale_node(self, node_id: str, staleness_score: float) -> None:
        """Add a node to the staleness tracking set."""
        self._ensure_connected()

        key = f"{self._prefix}:staleness:pending"
        await self._client.zadd(key, {node_id: staleness_score})

    async def get_stale_nodes(self, limit: int = 100) -> List[tuple]:
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

    async def get_stats(self) -> Dict[str, Any]:
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

    async def clear_all(self, workspace_id: Optional[str] = None) -> int:
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
                    logger.debug(
                        f"Cache invalidated: node {event.item_id} in {event.workspace_id}"
                    )

                elif event.event_type == "node_deleted":
                    if event.item_id:
                        await self.invalidate_node(event.item_id)
                        await self.remove_stale_node(event.item_id)
                    await self.invalidate_queries(event.workspace_id)
                    logger.debug(
                        f"Cache invalidated: deleted node {event.item_id} in {event.workspace_id}"
                    )

                elif event.event_type == "query_invalidated":
                    await self.invalidate_queries(event.workspace_id)
                    logger.debug(
                        f"Cache invalidated: queries in {event.workspace_id}"
                    )

                elif event.event_type == "culture_updated":
                    await self.invalidate_culture(event.workspace_id)
                    logger.debug(
                        f"Cache invalidated: culture in {event.workspace_id}"
                    )

            except Exception as e:
                logger.warning(f"Cache invalidation failed for event {event.event_type}: {e}")

        self._unsubscribe = bus.subscribe(handle_invalidation)
        logger.info("Redis cache subscribed to invalidation bus")

    def unsubscribe_from_invalidation_bus(self) -> None:
        """Unsubscribe from the CacheInvalidationBus."""
        if hasattr(self, "_unsubscribe") and self._unsubscribe:
            self._unsubscribe()
            self._unsubscribe = None
            logger.info("Redis cache unsubscribed from invalidation bus")

    # =========================================================================
    # Helpers
    # =========================================================================

    def _hash_key(self, key: str) -> str:
        """Hash a key for consistent sizing."""
        return hashlib.sha256(key.encode()).hexdigest()[:16]
