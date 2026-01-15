"""
Unified caching infrastructure for Aragora.

This module provides a single entry point for all caching needs, with a common
protocol that all cache backends implement.

Usage:
    from aragora.cache import (
        # Protocol and types
        CacheBackend,
        CacheStats,
        # Implementations
        TTLCache,
        RedisTTLCache,
        HybridTTLCache,
        # Decorators
        cached,
        async_cached,
        lru_cache_with_ttl,
        cached_property_ttl,
        # Registry
        get_cache,
        register_cache,
        get_all_cache_stats,
        # Global caches
        get_method_cache,
        get_query_cache,
        # Utilities
        invalidate_cache,
        clear_all_caches,
    )

    # Using the cached decorator
    @cached(ttl_seconds=300, key_prefix="users")
    def get_user(user_id: str) -> dict:
        return expensive_lookup(user_id)

    # Using a specific cache instance
    cache = TTLCache(maxsize=1000, ttl_seconds=600)
    cache.set("key", {"data": "value"})
    result = cache.get("key")

    # Using the cache registry
    register_cache("users", cache)
    stats = get_all_cache_stats()

Cache Implementations:
    - TTLCache: In-memory LRU cache with TTL expiry (thread-safe)
    - RedisTTLCache: Redis-backed cache with in-memory fallback
    - HybridTTLCache: Factory that returns best available cache
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any, Optional, Protocol, TypeVar, runtime_checkable

logger = logging.getLogger(__name__)

T = TypeVar("T")


# =============================================================================
# Cache Protocol
# =============================================================================


@dataclass
class CacheStats:
    """Statistics for a cache instance."""

    size: int
    maxsize: int
    ttl_seconds: float
    hits: int
    misses: int
    hit_rate: float
    extra: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CacheStats":
        """Create CacheStats from a dictionary."""
        return cls(
            size=d.get("size", 0),
            maxsize=d.get("maxsize", 0),
            ttl_seconds=d.get("ttl_seconds", 0),
            hits=d.get("hits", 0),
            misses=d.get("misses", 0),
            hit_rate=d.get("hit_rate", 0.0),
            extra={k: v for k, v in d.items() if k not in {
                "size", "maxsize", "ttl_seconds", "hits", "misses", "hit_rate"
            }} or None,
        )


@runtime_checkable
class CacheBackend(Protocol[T]):
    """Protocol defining the interface for cache backends.

    All cache implementations should conform to this protocol to ensure
    consistent behavior across the codebase.
    """

    def get(self, key: str) -> Optional[T]:
        """Get a value from cache if not expired.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        ...

    def set(self, key: str, value: T) -> None:
        """Store a value in cache.

        Args:
            key: Cache key
            value: Value to store
        """
        ...

    def invalidate(self, key: str) -> bool:
        """Invalidate a specific key.

        Args:
            key: Cache key to invalidate

        Returns:
            True if key was found and removed
        """
        ...

    def clear(self) -> int:
        """Clear all entries.

        Returns:
            Number of entries cleared
        """
        ...

    def clear_prefix(self, prefix: str) -> int:
        """Clear entries with keys starting with prefix.

        Args:
            prefix: Key prefix to match

        Returns:
            Number of entries cleared
        """
        ...

    @property
    def stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with size, maxsize, ttl_seconds, hits, misses, hit_rate
        """
        ...


# =============================================================================
# Cache Registry
# =============================================================================

_cache_registry: dict[str, CacheBackend] = {}
_registry_lock = threading.Lock()


def register_cache(name: str, cache: CacheBackend) -> None:
    """Register a cache instance in the global registry.

    Args:
        name: Unique name for the cache
        cache: Cache instance implementing CacheBackend
    """
    with _registry_lock:
        _cache_registry[name] = cache
        logger.debug(f"Registered cache: {name}")


def get_cache(name: str) -> Optional[CacheBackend]:
    """Get a cache instance from the registry.

    Args:
        name: Name of the cache to retrieve

    Returns:
        Cache instance or None if not found
    """
    with _registry_lock:
        return _cache_registry.get(name)


def get_all_cache_stats() -> dict[str, CacheStats]:
    """Get statistics for all registered caches.

    Returns:
        Dictionary mapping cache names to their statistics
    """
    with _registry_lock:
        result = {}
        for name, cache in _cache_registry.items():
            try:
                stats_dict = cache.stats
                result[name] = CacheStats.from_dict(stats_dict)
            except Exception as e:
                logger.warning(f"Failed to get stats for cache {name}: {e}")
        return result


def list_caches() -> list[str]:
    """List all registered cache names.

    Returns:
        List of cache names
    """
    with _registry_lock:
        return list(_cache_registry.keys())


# =============================================================================
# Re-exports from implementations
# =============================================================================

# Core cache classes
from aragora.utils.cache import (
    TTLCache,
    # Decorators
    async_ttl_cache as async_cached,
    cached_property_ttl,
    # Global cache accessors
    clear_all_caches,
    get_cache_stats,
    get_method_cache,
    get_query_cache,
    # Cache invalidation
    invalidate_cache,
    invalidate_method_cache,
    lru_cache_with_ttl,
    ttl_cache as cached,
)

# Redis-backed cache
from aragora.utils.redis_cache import HybridTTLCache, RedisTTLCache

# =============================================================================
# Auto-register global caches
# =============================================================================


def _auto_register_global_caches() -> None:
    """Auto-register the global caches from utils.cache."""
    try:
        register_cache("method", get_method_cache())
        register_cache("query", get_query_cache())
    except Exception as e:
        logger.debug(f"Failed to auto-register global caches: {e}")


# Register on import
_auto_register_global_caches()


__all__ = [
    # Protocol and types
    "CacheBackend",
    "CacheStats",
    # Implementations
    "TTLCache",
    "RedisTTLCache",
    "HybridTTLCache",
    # Decorators
    "cached",
    "async_cached",
    "lru_cache_with_ttl",
    "cached_property_ttl",
    # Registry
    "register_cache",
    "get_cache",
    "get_all_cache_stats",
    "list_caches",
    # Global caches
    "get_method_cache",
    "get_query_cache",
    "get_cache_stats",
    # Utilities
    "invalidate_cache",
    "invalidate_method_cache",
    "clear_all_caches",
]
