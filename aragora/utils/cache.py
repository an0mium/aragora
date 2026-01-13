"""
General-purpose caching utilities for Aragora.

Provides LRU caching with TTL expiry that can be used throughout the codebase,
not just in HTTP handlers.

Usage:
    from aragora.utils.cache import lru_cache_with_ttl, cached_property_ttl

    class MyService:
        @lru_cache_with_ttl(ttl_seconds=300, maxsize=100)
        def get_expensive_data(self, key: str) -> dict:
            # Expensive operation
            ...

        @cached_property_ttl(ttl_seconds=600)
        def config(self) -> dict:
            # Computed once, cached for 10 minutes
            ...
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from functools import wraps
from typing import Any, Awaitable, Callable, Generic, Optional, TypeVar

from aragora.config import CACHE_TTL_METHOD, CACHE_TTL_QUERY

logger = logging.getLogger(__name__)

T = TypeVar("T")


class TTLCache(Generic[T]):
    """
    Thread-safe LRU cache with TTL expiry.

    Generic version that can store any type T.
    """

    def __init__(self, maxsize: int = 128, ttl_seconds: float = 300.0):
        self._cache: OrderedDict[str, tuple[float, T]] = OrderedDict()
        self._maxsize = maxsize
        self._ttl_seconds = ttl_seconds
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[T]:
        """Get a value from cache if not expired."""
        with self._lock:
            if key in self._cache:
                cached_time, value = self._cache[key]
                if time.time() - cached_time < self._ttl_seconds:
                    # Move to end (most recently used)
                    self._cache.move_to_end(key)
                    self._hits += 1
                    return value
                else:
                    # Expired
                    del self._cache[key]
            self._misses += 1
            return None

    def set(self, key: str, value: T) -> None:
        """Store a value in cache."""
        with self._lock:
            # Remove oldest entries if at capacity
            while len(self._cache) >= self._maxsize:
                self._cache.popitem(last=False)

            self._cache[key] = (time.time(), value)

    def invalidate(self, key: str) -> bool:
        """Invalidate a specific key."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def clear(self) -> int:
        """Clear all entries."""
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            return count

    def clear_prefix(self, prefix: str) -> int:
        """Clear entries with keys starting with prefix."""
        with self._lock:
            keys_to_remove = [k for k in self._cache if k.startswith(prefix)]
            for k in keys_to_remove:
                del self._cache[k]
            return len(keys_to_remove)

    @property
    def stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total = self._hits + self._misses
            return {
                "size": len(self._cache),
                "maxsize": self._maxsize,
                "ttl_seconds": self._ttl_seconds,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": self._hits / total if total > 0 else 0.0,
            }

    def __len__(self) -> int:
        return len(self._cache)


# Global cache instances for different purposes
_method_cache = TTLCache[Any](maxsize=1000, ttl_seconds=CACHE_TTL_METHOD)
_query_cache = TTLCache[Any](maxsize=500, ttl_seconds=CACHE_TTL_QUERY)


# Track registration status
_caches_registered = False


def _register_caches_with_service_registry() -> None:
    """Register caches with ServiceRegistry for observability.

    Called lazily on first access to avoid circular imports.
    Uses marker types from aragora.services for proper registration.
    """
    global _caches_registered
    if _caches_registered:
        return

    try:
        from aragora.services import (
            ServiceRegistry,
            MethodCacheService,
            QueryCacheService,
        )

        registry = ServiceRegistry.get()

        if not registry.has(MethodCacheService):
            registry.register(MethodCacheService, _method_cache)
        if not registry.has(QueryCacheService):
            registry.register(QueryCacheService, _query_cache)

        _caches_registered = True
        logger.debug("Caches registered with ServiceRegistry")
    except ImportError:
        pass  # Services module not available


def get_method_cache() -> TTLCache[Any]:
    """Get the global method cache, registering with ServiceRegistry if available."""
    _register_caches_with_service_registry()
    return _method_cache


def get_query_cache() -> TTLCache[Any]:
    """Get the global query cache, registering with ServiceRegistry if available."""
    _register_caches_with_service_registry()
    return _query_cache


def lru_cache_with_ttl(
    ttl_seconds: float = 300.0,
    maxsize: int = 128,
    key_prefix: str = "",
    cache: Optional[TTLCache] = None,
):
    """
    Decorator for caching function/method results with LRU eviction and TTL expiry.

    Args:
        ttl_seconds: How long to cache results (default 5 minutes)
        maxsize: Maximum number of cached results (default 128)
        key_prefix: Optional prefix for cache keys
        cache: Optional custom cache instance (uses global cache if not provided)

    Example:
        @lru_cache_with_ttl(ttl_seconds=300)
        def get_user(user_id: str) -> User:
            return db.query(User).filter_by(id=user_id).first()

        @lru_cache_with_ttl(ttl_seconds=60, key_prefix="leaderboard")
        def get_leaderboard(self, limit: int = 20) -> list[dict]:
            ...
    """
    # Create dedicated cache if maxsize differs from global
    if cache is None:
        if maxsize != 128:
            cache = TTLCache(maxsize=maxsize, ttl_seconds=ttl_seconds)
        else:
            cache = _method_cache

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            # Build cache key
            # Skip 'self' for methods by checking if first arg is an object
            skip_first = (
                args
                and hasattr(args[0], "__class__")
                and not isinstance(args[0], (str, int, float, bool, list, dict, tuple))
            )
            cache_args = args[1:] if skip_first else args

            # Create hashable key
            key_parts = [key_prefix or func.__name__]
            key_parts.append(str(cache_args))
            if kwargs:
                key_parts.append(str(sorted(kwargs.items())))
            cache_key = ":".join(key_parts)

            # Try to get from cache
            cached = cache.get(cache_key)
            if cached is not None:
                return cached

            # Compute and cache result
            result = func(*args, **kwargs)
            cache.set(cache_key, result)
            return result

        # Attach cache reference for manual invalidation
        wrapper.cache = cache  # type: ignore
        wrapper.cache_key_prefix = key_prefix or func.__name__  # type: ignore

        return wrapper

    return decorator


def cached_property_ttl(ttl_seconds: float = 300.0):
    """
    Decorator for cached properties with TTL expiry.

    Similar to @property but caches the result for the specified duration.
    Each instance gets its own cached value.

    Example:
        class MyClass:
            @cached_property_ttl(ttl_seconds=60)
            def expensive_computation(self) -> dict:
                return self._compute_expensive_thing()
    """

    def decorator(func: Callable[[Any], T]) -> property:
        attr_name = f"_cached_{func.__name__}"
        time_attr = f"_cached_{func.__name__}_time"

        @wraps(func)
        def getter(self) -> T:
            now = time.time()
            cached_time = getattr(self, time_attr, 0)

            if now - cached_time < ttl_seconds:
                cached_value = getattr(self, attr_name, None)
                if cached_value is not None:
                    return cached_value

            # Compute new value
            value = func(self)
            setattr(self, attr_name, value)
            setattr(self, time_attr, now)
            return value

        return property(getter)

    return decorator


def invalidate_method_cache(prefix: str) -> int:
    """Invalidate entries in the global method cache by prefix."""
    return _method_cache.clear_prefix(prefix)


def get_cache_stats() -> dict[str, Any]:
    """Get statistics for global caches."""
    # Register with service registry on first stats call
    _register_caches_with_service_registry()

    return {
        "method_cache": _method_cache.stats,
        "query_cache": _query_cache.stats,
    }


def clear_all_caches() -> dict[str, int]:
    """Clear all global caches."""
    return {
        "method_cache": _method_cache.clear(),
        "query_cache": _query_cache.clear(),
    }


# =============================================================================
# Handler-style cache utilities (for use by memory and other modules)
# =============================================================================

# Global bounded cache for handler-style caching
_handler_cache: TTLCache[Any] = TTLCache(maxsize=1000, ttl_seconds=60.0)


def ttl_cache(ttl_seconds: float = 60.0, key_prefix: str = "", skip_first: bool = True):
    """
    Decorator for caching function results with TTL expiry.

    This is the handler-style cache decorator that can be used by both
    server handlers and other modules like memory.

    Args:
        ttl_seconds: How long to cache results (default 60s)
        key_prefix: Prefix for cache key to namespace different functions
        skip_first: If True, skip first arg (self) when building cache key for methods.
                   Default is True since most usage is on class methods.
                   Set to False when decorating standalone functions.

    Usage:
        @ttl_cache(ttl_seconds=300, key_prefix="leaderboard")
        def _get_leaderboard(self, limit: int):
            ...
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            # Skip 'self' when building cache key for methods
            cache_args = args[1:] if skip_first and args else args
            # Build cache key from function name, args and kwargs
            cache_key = f"{key_prefix}:{func.__name__}:{cache_args}:{sorted(kwargs.items())}"

            # Check cache
            cached = _handler_cache.get(cache_key)
            if cached is not None:
                logger.debug(f"Cache hit for {cache_key}")
                return cached

            # Cache miss - compute and store
            result = func(*args, **kwargs)
            _handler_cache.set(cache_key, result)
            logger.debug(f"Cache miss, stored {cache_key}")
            return result

        return wrapper

    return decorator


def async_ttl_cache(ttl_seconds: float = 60.0, key_prefix: str = "", skip_first: bool = True):
    """
    Async decorator for caching coroutine results with TTL expiry.

    Same as ttl_cache but works with async functions.

    Args:
        ttl_seconds: How long to cache results (default 60s)
        key_prefix: Prefix for cache key to namespace different functions
        skip_first: If True, skip first arg (self) when building cache key for methods.
    """
    import asyncio

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            # Skip 'self' when building cache key for methods
            cache_args = args[1:] if skip_first and args else args
            cache_key = f"{key_prefix}:{func.__name__}:{cache_args}:{sorted(kwargs.items())}"

            # Check cache
            cached = _handler_cache.get(cache_key)
            if cached is not None:
                logger.debug(f"Cache hit for {cache_key}")
                return cached

            # Cache miss - compute and store
            result = await func(*args, **kwargs)
            _handler_cache.set(cache_key, result)
            logger.debug(f"Cache miss, stored {cache_key}")
            return result

        return wrapper  # type: ignore

    return decorator


# Maps data sources to cache prefixes that should be invalidated
CACHE_INVALIDATION_MAP: dict[str, list[str]] = {
    # Memory module events
    "memory": ["analytics_memory", "critique_patterns", "critique_stats"],
    # Debate events
    "debates": ["dashboard_debates", "analytics_debates", "replays_list", "consensus_stats"],
    # Consensus events
    "consensus": ["consensus_stats", "consensus_settled", "consensus_similar"],
    # ELO/ranking events
    "elo": ["leaderboard", "lb_rankings", "agents_list", "agent_profile"],
    # Agent events
    "agent": ["agent_profile", "agents_list"],
}


def invalidate_cache(data_source: str) -> int:
    """Invalidate cache entries associated with a data source.

    This function clears all cache prefixes registered for a given data source.
    Can be used by memory modules, server handlers, or any other code that
    needs to invalidate related caches.

    Args:
        data_source: Name of the data source (e.g., "memory", "debates", "consensus")

    Returns:
        Total number of cache entries invalidated
    """
    prefixes = CACHE_INVALIDATION_MAP.get(data_source, [data_source])
    total_cleared = 0

    for prefix in prefixes:
        cleared = _handler_cache.clear_prefix(prefix)
        total_cleared += cleared
        if cleared > 0:
            logger.debug(f"Cache invalidation: cleared {cleared} entries with prefix '{prefix}'")

    if total_cleared > 0:
        logger.info(f"Cache invalidated: source={data_source}, entries_cleared={total_cleared}")

    return total_cleared


def get_handler_cache() -> TTLCache[Any]:
    """Get the global handler cache instance."""
    return _handler_cache
