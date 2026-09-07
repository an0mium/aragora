"""
Aragora Caching Module

Provides decorators for caching function results with TTL-based expiration,
LRU eviction, and async compatibility.

Usage:
    from aragora.caching import cached, async_cached, memoize, cache_key

    @cached(ttl_seconds=300, maxsize=128)
    def expensive_computation(x: int) -> int:
        return x * x

    @async_cached(ttl_seconds=60)
    async def fetch_data(url: str) -> dict:
        ...

    @memoize
    def pure_function(n: int) -> int:
        return fibonacci(n)

    @cache_key("user_id", "action")
    @cached(ttl_seconds=600)
    def get_user_action(user_id: int, action: str, metadata: dict) -> dict:
        ...
"""

from aragora.caching.adaptive import AccessPattern, AdaptiveTTLCache, CacheOptimizer
from aragora.caching.decorators import (
    cached,
    async_cached,
    memoize,
    cache_key,
    CacheStats,
    CacheEntry,
    get_global_cache_stats,
    clear_all_caches,
)
from aragora.caching.redis import HybridTTLCache, RedisTTLCache
from aragora.caching.registry import (
    CacheBackend,
    get_all_cache_stats,
    get_cache,
    list_caches,
    make_cache_key,
    make_content_hash,
    register_cache,
)
from aragora.caching.ttl import (
    CacheManager,
    CachePreset,
    TTLCache,
    async_ttl_cache,
    cached_property_ttl,
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
    # Decorators (function-result caching)
    "cached",
    "async_cached",
    "memoize",
    "cache_key",
    "CacheStats",
    "CacheEntry",
    "get_global_cache_stats",
    "clear_all_caches",
    # Cache registry + backend protocol
    "CacheBackend",
    "register_cache",
    "get_cache",
    "get_all_cache_stats",
    "list_caches",
    "make_cache_key",
    "make_content_hash",
    # In-memory TTL cache primitives
    "TTLCache",
    "CacheManager",
    "CachePreset",
    "ttl_cache",
    "async_ttl_cache",
    "lru_cache_with_ttl",
    "cached_property_ttl",
    "get_cache_manager",
    "get_cache_stats",
    "get_handler_cache",
    "get_method_cache",
    "get_query_cache",
    "invalidate_cache",
    "invalidate_method_cache",
    # Redis-backed cache
    "RedisTTLCache",
    "HybridTTLCache",
    # Adaptive cache
    "AdaptiveTTLCache",
    "AccessPattern",
    "CacheOptimizer",
]
