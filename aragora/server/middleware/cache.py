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

# Use lazy imports to avoid circular import with handlers/base.py
# The actual imports are done when the functions are first accessed
_base_module = None


def _get_base():
    """Lazy import of handlers/base.py to avoid circular imports."""
    global _base_module
    if _base_module is None:
        from aragora.server.handlers import base as _base_module
    return _base_module


def ttl_cache(*args, **kwargs):
    """Decorator for caching function results with TTL expiry."""
    return _get_base().ttl_cache(*args, **kwargs)


def clear_cache(*args, **kwargs):
    """Clear cached entries."""
    return _get_base().clear_cache(*args, **kwargs)


def invalidate_cache(*args, **kwargs):
    """Invalidate caches related to data sources."""
    return _get_base().invalidate_cache(*args, **kwargs)


def get_cache_stats():
    """Get cache statistics."""
    return _get_base().get_cache_stats()


def get_cache():
    """Get the global cache instance."""
    return _get_base()._cache


def reset_cache() -> None:
    """Reset the global cache. Primarily for testing."""
    _get_base()._cache.clear()


def get_cache_invalidation_map():
    """Get the CACHE_INVALIDATION_MAP from handlers/base.py.

    Returns the dict that maps event names to cache key prefixes.
    """
    return _get_base().CACHE_INVALIDATION_MAP


def get_bounded_ttl_cache_class():
    """Get the BoundedTTLCache class from handlers/base.py.

    Returns the BoundedTTLCache class for creating cache instances.
    """
    return _get_base().BoundedTTLCache


# For backward compatibility, provide CACHE_INVALIDATION_MAP as a callable
# Tests should use get_cache_invalidation_map() instead
CACHE_INVALIDATION_MAP = get_cache_invalidation_map
BoundedTTLCache = get_bounded_ttl_cache_class


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
    "get_cache",
    "reset_cache",
    "get_cache_invalidation_map",
    "get_bounded_ttl_cache_class",
    "CacheConfig",
    "CACHE_INVALIDATION_MAP",
    "BoundedTTLCache",
]
