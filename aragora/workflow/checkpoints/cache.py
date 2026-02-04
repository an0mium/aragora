"""
LRU cache and caching wrapper for checkpoint stores.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Any

from aragora.workflow.checkpoints._compat import MAX_CHECKPOINT_CACHE_SIZE
from aragora.workflow.checkpoints.protocol import CheckpointStore
from aragora.workflow.types import WorkflowCheckpoint

logger = logging.getLogger(__name__)


class LRUCheckpointCache:
    """
    LRU cache for workflow checkpoints with bounded size.

    Prevents unbounded memory growth in long-running workflows.
    Thread-safe for concurrent access patterns.
    """

    def __init__(self, max_size: int = MAX_CHECKPOINT_CACHE_SIZE):
        self._cache: OrderedDict[str, WorkflowCheckpoint] = OrderedDict()
        self._max_size = max_size
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> WorkflowCheckpoint | None:
        """Get checkpoint from cache, updating LRU order."""
        if key in self._cache:
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self._hits += 1
            return self._cache[key]
        self._misses += 1
        return None

    def put(self, key: str, checkpoint: WorkflowCheckpoint) -> None:
        """Put checkpoint in cache, evicting oldest if full."""
        if key in self._cache:
            self._cache[key] = checkpoint
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self._max_size:
                # Evict oldest (first item)
                evicted_key, _ = self._cache.popitem(last=False)
                logger.debug(f"LRU eviction: {evicted_key}")
            self._cache[key] = checkpoint

    def remove(self, key: str) -> bool:
        """Remove checkpoint from cache."""
        if key in self._cache:
            del self._cache[key]
            return True
        return False

    def clear(self) -> None:
        """Clear entire cache."""
        self._cache.clear()

    @property
    def size(self) -> int:
        """Current cache size."""
        return len(self._cache)

    @property
    def stats(self) -> dict[str, Any]:
        """Cache statistics."""
        total = self._hits + self._misses
        hit_rate = (self._hits / total) if total > 0 else 0.0
        return {
            "size": len(self._cache),
            "max_size": self._max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
        }


class CachingCheckpointStore:
    """
    Caching wrapper for any CheckpointStore implementation.

    Provides an LRU cache layer on top of any backend store, reducing
    redundant reads for frequently accessed checkpoints.

    Usage:
        base_store = RedisCheckpointStore()
        cached_store = CachingCheckpointStore(base_store)

        # First load hits backend, subsequent loads hit cache
        cp1 = await cached_store.load(checkpoint_id)  # Backend
        cp2 = await cached_store.load(checkpoint_id)  # Cache hit
    """

    def __init__(
        self,
        store: CheckpointStore,
        max_cache_size: int = MAX_CHECKPOINT_CACHE_SIZE,
    ):
        """
        Initialize caching checkpoint store.

        Args:
            store: The underlying CheckpointStore to wrap
            max_cache_size: Maximum number of checkpoints to cache (default 100)
        """
        self._store = store
        self._cache = LRUCheckpointCache(max_size=max_cache_size)

    async def save(self, checkpoint: WorkflowCheckpoint) -> str:
        """Save checkpoint to store and update cache."""
        checkpoint_id = await self._store.save(checkpoint)
        # Update cache with saved checkpoint
        self._cache.put(checkpoint_id, checkpoint)
        return checkpoint_id

    async def load(self, checkpoint_id: str) -> WorkflowCheckpoint | None:
        """Load checkpoint from cache or store."""
        # Check cache first
        cached = self._cache.get(checkpoint_id)
        if cached is not None:
            return cached

        # Miss - load from backend
        checkpoint = await self._store.load(checkpoint_id)
        if checkpoint is not None:
            self._cache.put(checkpoint_id, checkpoint)
        return checkpoint

    async def load_latest(self, workflow_id: str) -> WorkflowCheckpoint | None:
        """Load latest checkpoint for workflow (always hits backend)."""
        # Always go to backend for latest since we don't track recency
        checkpoint = await self._store.load_latest(workflow_id)
        if checkpoint is not None:
            self._cache.put(checkpoint.id, checkpoint)
        return checkpoint

    async def list_checkpoints(self, workflow_id: str) -> list[str]:
        """List checkpoint IDs for workflow (always hits backend)."""
        return await self._store.list_checkpoints(workflow_id)

    async def delete(self, checkpoint_id: str) -> bool:
        """Delete checkpoint from store and cache."""
        self._cache.remove(checkpoint_id)
        return await self._store.delete(checkpoint_id)

    def clear_cache(self) -> None:
        """Clear the cache without affecting the backend store."""
        self._cache.clear()

    @property
    def cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return self._cache.stats

    @property
    def backend_store(self) -> CheckpointStore:
        """Get the underlying backend store."""
        return self._store
