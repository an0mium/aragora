"""Canonical caching access to the in-memory TTL cache primitives.

These TTL utilities physically live in the foundation-layer
:mod:`aragora.utils.cache` module. ``aragora.caching`` re-exports them here so
the consolidated caching substrate offers a single ``aragora.caching`` import
surface for callers while the foundation module remains the implementation.
"""

from __future__ import annotations

from aragora.utils.cache import (
    CachedCallable,
    CacheManager,
    CachePreset,
    TTLCache,
    async_ttl_cache,
    cached_property_ttl,
    clear_all_caches,
    get_cache_manager,
    get_cache_stats,
    get_handler_cache,
    get_method_cache,
    get_query_cache,
    invalidate_cache,
    invalidate_method_cache,
    lru_cache_with_ttl,
    ttl_cache,
)

__all__ = [
    "CachedCallable",
    "CacheManager",
    "CachePreset",
    "TTLCache",
    "async_ttl_cache",
    "cached_property_ttl",
    "clear_all_caches",
    "get_cache_manager",
    "get_cache_stats",
    "get_handler_cache",
    "get_method_cache",
    "get_query_cache",
    "invalidate_cache",
    "invalidate_method_cache",
    "lru_cache_with_ttl",
    "ttl_cache",
]
