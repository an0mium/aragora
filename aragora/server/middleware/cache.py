"""
Caching Middleware.

Re-exports cache functionality from handlers/base.py for a cleaner import path.

This module provides:
- ttl_cache: Decorator for caching function results with TTL expiry
- cache: Alias for ttl_cache for convenience
- clear_cache: Clear cached entries
- invalidate_cache: Invalidate caches related to data sources
- get_cache_stats: Get cache statistics
- CacheConfig: Configuration dataclass
- CACHE_INVALIDATION_MAP: Maps data sources to cache key prefixes

Usage:
    from aragora.server.middleware import cache, ttl_cache, clear_cache

    @ttl_cache(ttl_seconds=300, key_prefix="leaderboard")
    def get_leaderboard(self, limit: int):
        ...

    # Or use the shorter alias
    @cache(ttl_seconds=60)
    def get_agents(self):
        ...
"""

from dataclasses import dataclass
from typing import Optional

# Re-export from handlers/base.py for backward compatibility
from aragora.server.handlers.base import (
    ttl_cache,
    clear_cache,
    invalidate_cache,
    get_cache_stats,
    CACHE_INVALIDATION_MAP,
    BoundedTTLCache,
    _cache as _global_cache,
)


def get_cache() -> BoundedTTLCache:
    """Get the global cache instance."""
    return _global_cache


def reset_cache() -> None:
    """Reset the global cache. Primarily for testing."""
    _global_cache.clear()


@dataclass
class CacheConfig:
    """Configuration for cache behavior."""

    ttl_seconds: float = 60.0
    key_prefix: str = ""
    max_entries: int = 1000
    enabled: bool = True


# Alias for convenience
cache = ttl_cache


__all__ = [
    "cache",
    "ttl_cache",
    "clear_cache",
    "invalidate_cache",
    "get_cache_stats",
    "CacheConfig",
    "CACHE_INVALIDATION_MAP",
    "BoundedTTLCache",
]
