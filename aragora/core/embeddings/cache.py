"""Unified embedding cache with TTL and LRU eviction.

This module provides a single caching layer for all embedding operations,
replacing the fragmented caches across memory, debate, and knowledge modules.
"""

import hashlib
import logging
import threading
import time
from collections import OrderedDict
from typing import Optional

from aragora.core.embeddings.types import CacheStats

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """Thread-safe LRU cache for embeddings with TTL expiration.

    Uses OrderedDict for O(1) LRU eviction and get/set operations.
    Supports both list[float] and numpy arrays transparently.

    Example:
        cache = EmbeddingCache(ttl_seconds=3600, max_size=1000)

        # Check cache
        cached = cache.get("some text")
        if cached is None:
            embedding = await compute_embedding("some text")
            cache.set("some text", embedding)
    """

    def __init__(
        self,
        ttl_seconds: float = 3600.0,
        max_size: int = 1000,
    ):
        """Initialize embedding cache.

        Args:
            ttl_seconds: Time-to-live for cached entries (default: 1 hour)
            max_size: Maximum number of entries before LRU eviction
        """
        self._cache: OrderedDict[str, tuple[float, list[float]]] = OrderedDict()
        self._ttl = ttl_seconds
        self._max_size = max_size
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0

    def _make_key(self, text: str) -> str:
        """Generate cache key from text."""
        return hashlib.sha256(text.lower().strip().encode()).hexdigest()

    def get(self, text: str) -> Optional[list[float]]:
        """Get cached embedding if valid.

        Args:
            text: Text to look up

        Returns:
            Cached embedding if found and not expired, None otherwise
        """
        key = self._make_key(text)

        with self._lock:
            if key in self._cache:
                timestamp, embedding = self._cache[key]
                if time.time() - timestamp < self._ttl:
                    self._hits += 1
                    # Move to end to mark as recently used (LRU)
                    self._cache.move_to_end(key)
                    return embedding
                # Expired - remove
                del self._cache[key]

            self._misses += 1
        return None

    def set(self, text: str, embedding: list[float]) -> None:
        """Cache an embedding.

        Args:
            text: Original text
            embedding: Embedding vector to cache
        """
        key = self._make_key(text)

        with self._lock:
            # If key exists, update timestamp and move to end
            if key in self._cache:
                self._cache[key] = (time.time(), embedding)
                self._cache.move_to_end(key)
                return

            # Evict oldest entry if at capacity - O(1) with popitem(last=False)
            while len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)

            self._cache[key] = (time.time(), embedding)

    def get_stats(self) -> CacheStats:
        """Get cache statistics.

        Returns:
            CacheStats with current cache state
        """
        with self._lock:
            now = time.time()
            valid = sum(1 for ts, _ in self._cache.values() if now - ts < self._ttl)
            total = self._hits + self._misses

            return CacheStats(
                size=len(self._cache),
                valid=valid,
                hits=self._hits,
                misses=self._misses,
                hit_rate=self._hits / total if total > 0 else 0.0,
                ttl_seconds=self._ttl,
            )

    def clear(self) -> None:
        """Clear all cached entries and reset stats."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    def __len__(self) -> int:
        """Return number of cached entries."""
        return len(self._cache)


class ScopedCacheManager:
    """Manager for scope-isolated embedding caches.

    Prevents cross-contamination between different contexts (e.g., debates)
    by providing isolated caches per scope ID.

    Example:
        manager = ScopedCacheManager()
        cache = manager.get_cache("debate_123")
        # Use cache...
        manager.cleanup("debate_123")  # Free memory when done
    """

    def __init__(
        self,
        default_ttl: float = 3600.0,
        default_max_size: int = 1024,
    ):
        """Initialize cache manager.

        Args:
            default_ttl: Default TTL for new caches
            default_max_size: Default max size for new caches
        """
        self._caches: dict[str, EmbeddingCache] = {}
        self._lock = threading.Lock()
        self._default_ttl = default_ttl
        self._default_max_size = default_max_size

    def configure(
        self,
        ttl_seconds: float = 3600.0,
        max_size: int = 1024,
    ) -> None:
        """Configure defaults for new caches."""
        with self._lock:
            self._default_ttl = ttl_seconds
            self._default_max_size = max_size

    def get_cache(self, scope_id: str) -> EmbeddingCache:
        """Get or create cache for a specific scope.

        Args:
            scope_id: Unique identifier for the scope (e.g., debate_id)

        Returns:
            EmbeddingCache instance isolated to this scope
        """
        with self._lock:
            if scope_id not in self._caches:
                self._caches[scope_id] = EmbeddingCache(
                    ttl_seconds=self._default_ttl,
                    max_size=self._default_max_size,
                )
                logger.debug(f"Created new embedding cache for scope {scope_id}")
            return self._caches[scope_id]

    def cleanup(self, scope_id: str) -> None:
        """Remove and clear cache for a completed scope.

        Args:
            scope_id: Scope ID to cleanup
        """
        with self._lock:
            if scope_id in self._caches:
                self._caches[scope_id].clear()
                del self._caches[scope_id]
                logger.debug(f"Cleaned up embedding cache for scope {scope_id}")

    def get_stats(self) -> dict[str, CacheStats]:
        """Get statistics for all active caches."""
        with self._lock:
            return {scope_id: cache.get_stats() for scope_id, cache in self._caches.items()}

    def clear_all(self) -> None:
        """Clear all caches (for testing)."""
        with self._lock:
            for cache in self._caches.values():
                cache.clear()
            self._caches.clear()


# Global cache instances
_global_cache: Optional[EmbeddingCache] = None
_scoped_manager = ScopedCacheManager()


def get_global_cache(
    ttl_seconds: float = 3600.0,
    max_size: int = 1000,
) -> EmbeddingCache:
    """Get or create the global embedding cache.

    For most use cases, prefer the scoped cache via get_scoped_cache().
    """
    global _global_cache
    if _global_cache is None:
        _global_cache = EmbeddingCache(ttl_seconds=ttl_seconds, max_size=max_size)
    return _global_cache


def get_scoped_cache(scope_id: str) -> EmbeddingCache:
    """Get embedding cache scoped to a specific context.

    This is the preferred method for getting caches in isolated contexts
    (e.g., debates) to prevent cross-contamination.
    """
    return _scoped_manager.get_cache(scope_id)


def cleanup_scoped_cache(scope_id: str) -> None:
    """Cleanup cache for a completed scope."""
    _scoped_manager.cleanup(scope_id)


def reset_caches() -> None:
    """Reset all caches (for testing)."""
    global _global_cache
    if _global_cache:
        _global_cache.clear()
    _global_cache = None
    _scoped_manager.clear_all()
