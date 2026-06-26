"""Deprecated import location for Aragora's caching substrate.

The unified caching implementation now lives under :mod:`aragora.caching`.
Importing from ``aragora.cache`` still works but is deprecated; import from
``aragora.caching`` (and its submodules ``aragora.caching.registry`` /
``aragora.caching.redis``) instead.
"""

from __future__ import annotations

import warnings

from aragora.caching.redis import HybridTTLCache, RedisTTLCache
from aragora.caching.registry import (
    CacheBackend,
    CacheStats,
    get_all_cache_stats,
    get_cache,
    list_caches,
    make_cache_key,
    make_content_hash,
    register_cache,
)
from aragora.caching.ttl import (
    TTLCache,
    async_ttl_cache as async_cached,
    cached_property_ttl,
    clear_all_caches,
    get_cache_stats,
    get_method_cache,
    get_query_cache,
    invalidate_cache,
    invalidate_method_cache,
    lru_cache_with_ttl,
    ttl_cache as cached,
)

warnings.warn(
    "aragora.cache is deprecated; import from aragora.caching instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "CacheBackend",
    "CacheStats",
    "TTLCache",
    "RedisTTLCache",
    "HybridTTLCache",
    "cached",
    "async_cached",
    "lru_cache_with_ttl",
    "cached_property_ttl",
    "register_cache",
    "get_cache",
    "get_all_cache_stats",
    "list_caches",
    "get_method_cache",
    "get_query_cache",
    "get_cache_stats",
    "invalidate_cache",
    "invalidate_method_cache",
    "clear_all_caches",
    "make_cache_key",
    "make_content_hash",
]
