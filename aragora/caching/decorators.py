"""
Caching Decorators for Aragora

This module provides a comprehensive caching layer with:
- TTL-based expiration with LRU eviction
- Async-compatible caching
- Simple memoization
- Custom cache key generation
- Statistics tracking

Thread-safe implementation using locks for concurrent access.
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import inspect
import pickle
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Generic,
    Optional,
    ParamSpec,
    TypeVar,
    cast,
    overload,
)

# Type variables for generic typing
P = ParamSpec("P")
T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])


@dataclass
class CacheStats:
    """Statistics for cache performance tracking."""

    hits: int = 0
    misses: int = 0
    size: int = 0
    maxsize: int = 0
    evictions: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate the cache hit rate as a percentage."""
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return (self.hits / total) * 100

    def __repr__(self) -> str:
        return (
            f"CacheStats(hits={self.hits}, misses={self.misses}, "
            f"size={self.size}, maxsize={self.maxsize}, "
            f"evictions={self.evictions}, hit_rate={self.hit_rate:.1f}%)"
        )


@dataclass
class CacheEntry(Generic[T]):
    """A single cache entry with value and expiration tracking."""

    value: T
    created_at: float
    ttl_seconds: Optional[float] = None

    def is_expired(self) -> bool:
        """Check if this cache entry has expired."""
        if self.ttl_seconds is None:
            return False
        return time.time() - self.created_at > self.ttl_seconds


# Global registry of all caches for management
_cache_registry: list[dict[str, Any]] = []
_registry_lock = threading.Lock()


def _register_cache(cache_dict: OrderedDict, stats: CacheStats, lock: threading.Lock) -> None:
    """Register a cache for global management."""
    with _registry_lock:
        _cache_registry.append(
            {
                "cache": cache_dict,
                "stats": stats,
                "lock": lock,
            }
        )


def get_global_cache_stats() -> list[CacheStats]:
    """Get statistics for all registered caches."""
    with _registry_lock:
        return [entry["stats"] for entry in _cache_registry]


def clear_all_caches() -> int:
    """
    Clear all registered caches.

    Returns:
        Number of caches cleared.
    """
    with _registry_lock:
        count = 0
        for entry in _cache_registry:
            with entry["lock"]:
                entry["cache"].clear()
                entry["stats"].size = 0
                count += 1
        return count


def _make_cache_key(
    args: tuple,
    kwargs: dict,
    key_args: Optional[tuple[str, ...]] = None,
    param_names: Optional[tuple[str, ...]] = None,
) -> str:
    """
    Generate a cache key from function arguments.

    Args:
        args: Positional arguments
        kwargs: Keyword arguments
        key_args: Optional specific argument names to use for key generation
        param_names: Parameter names from function signature (for positional arg mapping)

    Returns:
        A string hash representing the cache key.
    """
    if key_args and param_names:
        # Build a complete mapping of argument names to values
        all_args: dict[str, Any] = {}

        # Map positional args to their parameter names
        for i, value in enumerate(args):
            if i < len(param_names):
                all_args[param_names[i]] = value

        # Add keyword arguments (these override positional if duplicated)
        all_args.update(kwargs)

        # Only use specified arguments for the key
        key_parts: list[tuple[str, Any]] = []
        for key in key_args:
            if key in all_args:
                key_parts.append((key, all_args[key]))
        key_data: Any = tuple(key_parts)
    else:
        # Use all arguments
        key_data = (args, tuple(sorted(kwargs.items())))

    try:
        # Try to pickle for hashable check
        serialized = pickle.dumps(key_data, protocol=pickle.HIGHEST_PROTOCOL)
        return hashlib.sha256(serialized).hexdigest()
    except (pickle.PicklingError, TypeError):
        # Fall back to repr for unhashable types
        return hashlib.sha256(repr(key_data).encode()).hexdigest()


class _TTLCache(Generic[T]):
    """
    Thread-safe TTL cache with LRU eviction.

    This cache stores values with time-to-live expiration and evicts
    least recently used entries when the cache reaches maxsize.
    """

    def __init__(self, maxsize: int = 128, ttl_seconds: float = 300.0):
        """
        Initialize the TTL cache.

        Args:
            maxsize: Maximum number of entries to store
            ttl_seconds: Time-to-live for entries in seconds
        """
        self._cache: OrderedDict[str, CacheEntry[T]] = OrderedDict()
        self._lock = threading.Lock()
        self._maxsize = maxsize
        self._ttl_seconds = ttl_seconds
        self._stats = CacheStats(maxsize=maxsize)

        # Register for global management
        _register_cache(self._cache, self._stats, self._lock)

    def get(self, key: str) -> tuple[bool, Optional[T]]:
        """
        Get a value from the cache.

        Args:
            key: The cache key

        Returns:
            Tuple of (found, value). If found is False, value is None.
        """
        with self._lock:
            if key not in self._cache:
                self._stats.misses += 1
                return False, None

            entry = self._cache[key]

            if entry.is_expired():
                del self._cache[key]
                self._stats.size = len(self._cache)
                self._stats.misses += 1
                return False, None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self._stats.hits += 1
            return True, entry.value

    def set(self, key: str, value: T) -> None:
        """
        Set a value in the cache.

        Args:
            key: The cache key
            value: The value to cache
        """
        with self._lock:
            # Remove expired entries first
            self._cleanup_expired()

            if key in self._cache:
                # Update existing entry
                self._cache[key] = CacheEntry(
                    value=value,
                    created_at=time.time(),
                    ttl_seconds=self._ttl_seconds,
                )
                self._cache.move_to_end(key)
            else:
                # Add new entry
                while len(self._cache) >= self._maxsize:
                    # Evict LRU (first item)
                    self._cache.popitem(last=False)
                    self._stats.evictions += 1

                self._cache[key] = CacheEntry(
                    value=value,
                    created_at=time.time(),
                    ttl_seconds=self._ttl_seconds,
                )

            self._stats.size = len(self._cache)

    def _cleanup_expired(self) -> None:
        """Remove expired entries from the cache. Must be called with lock held."""
        expired_keys = [key for key, entry in self._cache.items() if entry.is_expired()]
        for key in expired_keys:
            del self._cache[key]

    def clear(self) -> None:
        """Clear all entries from the cache."""
        with self._lock:
            self._cache.clear()
            self._stats.size = 0

    def cache_info(self) -> CacheStats:
        """Get current cache statistics."""
        with self._lock:
            self._stats.size = len(self._cache)
            return CacheStats(
                hits=self._stats.hits,
                misses=self._stats.misses,
                size=self._stats.size,
                maxsize=self._stats.maxsize,
                evictions=self._stats.evictions,
            )


@overload
def cached(func: F) -> F: ...


@overload
def cached(
    ttl_seconds: float = 300.0,
    maxsize: int = 128,
) -> Callable[[F], F]: ...


def cached(  # type: ignore[misc]  # overload implementation signature
    func: Optional[F] = None,
    ttl_seconds: float = 300.0,
    maxsize: int = 128,
) -> F | Callable[[F], F]:
    """
    Decorator for TTL-based caching with LRU eviction.

    Caches function return values with automatic expiration and
    least-recently-used eviction when the cache is full.

    Args:
        func: The function to decorate (when used without parentheses)
        ttl_seconds: Time-to-live for cached values in seconds (default: 300)
        maxsize: Maximum number of cached values (default: 128)

    Returns:
        Decorated function with caching enabled.

    Example:
        @cached(ttl_seconds=60, maxsize=100)
        def expensive_computation(x: int) -> int:
            return x * x

        # Can also be used without arguments
        @cached
        def another_function(x: int) -> int:
            return x + 1

    The decorated function has additional methods:
        - cache_info(): Returns CacheStats with hits, misses, size
        - cache_clear(): Clears all cached values
    """

    def decorator(fn: F) -> F:
        cache = _TTLCache[Any](maxsize=maxsize, ttl_seconds=ttl_seconds)

        # Get parameter names for positional argument mapping (captured at decoration time)
        try:
            sig = inspect.signature(fn)
            param_names = tuple(sig.parameters.keys())
        except (ValueError, TypeError):
            param_names = None

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Check for custom key args at call time (allows @cache_key to be applied after)
            key_args = getattr(wrapper, "_cache_key_args", None)
            effective_param_names = param_names if key_args else None
            cache_key = _make_cache_key(args, kwargs, key_args, effective_param_names)

            found, value = cache.get(cache_key)
            if found:
                return value

            result = fn(*args, **kwargs)
            cache.set(cache_key, result)
            return result

        # Attach cache management methods using setattr to avoid attr-defined errors
        setattr(wrapper, "cache_info", cache.cache_info)
        setattr(wrapper, "cache_clear", cache.clear)

        return cast(F, wrapper)

    # Handle both @cached and @cached() syntax
    if func is not None:
        return decorator(func)
    return decorator


@overload
def async_cached(func: F) -> F: ...


@overload
def async_cached(
    ttl_seconds: float = 300.0,
    maxsize: int = 128,
) -> Callable[[F], F]: ...


def async_cached(  # type: ignore[misc]  # overload implementation signature
    func: Optional[F] = None,
    ttl_seconds: float = 300.0,
    maxsize: int = 128,
) -> F | Callable[[F], F]:
    """
    Decorator for TTL-based caching of async functions.

    Same as @cached but works with async/await functions.
    Uses asyncio.Lock for async-safe concurrent access.

    Args:
        func: The async function to decorate (when used without parentheses)
        ttl_seconds: Time-to-live for cached values in seconds (default: 300)
        maxsize: Maximum number of cached values (default: 128)

    Returns:
        Decorated async function with caching enabled.

    Example:
        @async_cached(ttl_seconds=60)
        async def fetch_user_data(user_id: int) -> dict:
            async with aiohttp.ClientSession() as session:
                response = await session.get(f"/users/{user_id}")
                return await response.json()

    The decorated function has additional methods:
        - cache_info(): Returns CacheStats with hits, misses, size
        - cache_clear(): Clears all cached values
    """

    def decorator(fn: F) -> F:
        cache = _TTLCache[Any](maxsize=maxsize, ttl_seconds=ttl_seconds)
        async_lock = asyncio.Lock()

        # Get parameter names for positional argument mapping (captured at decoration time)
        try:
            sig = inspect.signature(fn)
            param_names = tuple(sig.parameters.keys())
        except (ValueError, TypeError):
            param_names = None

        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Check for custom key args at call time (allows @cache_key to be applied after)
            key_args = getattr(wrapper, "_cache_key_args", None)
            effective_param_names = param_names if key_args else None
            cache_key = _make_cache_key(args, kwargs, key_args, effective_param_names)

            # Check cache without lock first (fast path)
            found, value = cache.get(cache_key)
            if found:
                return value

            # Slow path: acquire lock and check again
            async with async_lock:
                # Double-check after acquiring lock
                found, value = cache.get(cache_key)
                if found:
                    return value

                result = await fn(*args, **kwargs)
                cache.set(cache_key, result)
                return result

        # Attach cache management methods using setattr to avoid attr-defined errors
        setattr(wrapper, "cache_info", cache.cache_info)
        setattr(wrapper, "cache_clear", cache.clear)

        return cast(F, wrapper)

    # Handle both @async_cached and @async_cached() syntax
    if func is not None:
        return decorator(func)
    return decorator


def memoize(func: F) -> F:
    """
    Simple memoization decorator without TTL.

    Caches function results indefinitely based on arguments.
    Uses an unbounded cache - suitable for pure functions with
    limited unique argument combinations.

    Args:
        func: The function to memoize

    Returns:
        Memoized function.

    Example:
        @memoize
        def fibonacci(n: int) -> int:
            if n < 2:
                return n
            return fibonacci(n - 1) + fibonacci(n - 2)

    The decorated function has additional methods:
        - cache_info(): Returns CacheStats with hits, misses, size
        - cache_clear(): Clears all cached values

    Note:
        This decorator creates an unbounded cache. For functions with
        many unique argument combinations, consider using @cached with
        appropriate maxsize instead.
    """
    cache: dict[str, Any] = {}
    stats = CacheStats(maxsize=-1)  # -1 indicates unbounded
    lock = threading.Lock()

    # Register for global management
    with _registry_lock:
        _cache_registry.append(
            {
                "cache": cache,
                "stats": stats,
                "lock": lock,
            }
        )

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        cache_key = _make_cache_key(args, kwargs)

        with lock:
            if cache_key in cache:
                stats.hits += 1
                return cache[cache_key]

        result = func(*args, **kwargs)

        with lock:
            cache[cache_key] = result
            stats.misses += 1
            stats.size = len(cache)

        return result

    def cache_info() -> CacheStats:
        with lock:
            return CacheStats(
                hits=stats.hits,
                misses=stats.misses,
                size=len(cache),
                maxsize=-1,
                evictions=0,
            )

    def cache_clear() -> None:
        with lock:
            cache.clear()
            stats.size = 0

    setattr(wrapper, "cache_info", cache_info)
    setattr(wrapper, "cache_clear", cache_clear)

    return cast(F, wrapper)


def cache_key(*key_args: str) -> Callable[[F], F]:
    """
    Decorator to specify which arguments to use for cache key generation.

    By default, all arguments are used to generate cache keys. This decorator
    allows you to specify only certain arguments, which is useful when some
    arguments don't affect the result or contain unhashable data.

    Args:
        *key_args: Names of arguments to use for cache key generation

    Returns:
        A decorator that marks the function with the specified key arguments.

    Example:
        @cache_key("user_id", "action")
        @cached(ttl_seconds=300)
        def get_user_action(user_id: int, action: str, request_metadata: dict) -> dict:
            # request_metadata is ignored for caching
            return perform_action(user_id, action)

    Note:
        This decorator must be applied BEFORE the caching decorator (i.e.,
        it should be the innermost decorator).
    """

    def decorator(func: F) -> F:
        setattr(func, "_cache_key_args", key_args)
        return func

    return decorator


class CacheContext:
    """
    Context manager for temporary cache configuration.

    Useful for testing or temporarily disabling caching.

    Example:
        with CacheContext(enabled=False):
            # Caching is disabled within this block
            result = expensive_computation(42)
    """

    _enabled: bool = True
    _lock = threading.Lock()

    def __init__(self, enabled: bool = True):
        """
        Initialize cache context.

        Args:
            enabled: Whether caching should be enabled in this context
        """
        self._new_enabled = enabled
        self._old_enabled: Optional[bool] = None

    def __enter__(self) -> "CacheContext":
        with self._lock:
            self._old_enabled = CacheContext._enabled
            CacheContext._enabled = self._new_enabled
        return self

    def __exit__(self, *args: Any) -> None:
        with self._lock:
            if self._old_enabled is not None:
                CacheContext._enabled = self._old_enabled

    @classmethod
    def is_enabled(cls) -> bool:
        """Check if caching is currently enabled."""
        with cls._lock:
            return cls._enabled
