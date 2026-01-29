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

__all__ = [
    "cached",
    "async_cached",
    "memoize",
    "cache_key",
    "CacheStats",
    "CacheEntry",
    "get_global_cache_stats",
    "clear_all_caches",
]
