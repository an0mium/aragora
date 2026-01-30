"""
Singleflight pattern for cache stampede prevention.

Ensures that only one execution of a function is in-flight for a given key
at any time. When multiple callers request the same key concurrently, only
the first caller computes the value; subsequent callers wait for and share
the result.

This prevents the "thundering herd" problem where a popular cache entry
expires and hundreds of concurrent requests all recompute the same value.

Usage:
    from aragora.utils.singleflight import SingleFlight, AsyncSingleFlight

    # Sync usage
    sf = SingleFlight()
    result = sf.do("user:123", lambda: expensive_db_query(123))

    # Async usage
    asf = AsyncSingleFlight()
    result = await asf.do("user:123", lambda: async_db_query(123))

    # Decorator usage
    from aragora.utils.singleflight import singleflight_cached, async_singleflight_cached

    @singleflight_cached(ttl_seconds=300)
    def get_leaderboard(limit: int) -> list:
        ...

    @async_singleflight_cached(ttl_seconds=60)
    async def get_dashboard(org_id: str) -> dict:
        ...
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Awaitable, Callable, Generic, TypeVar, cast

logger = logging.getLogger(__name__)

T = TypeVar("T")

# ---------------------------------------------------------------------------
# Sync SingleFlight
# ---------------------------------------------------------------------------


@dataclass
class _FlightResult:
    """Result of an in-flight computation."""

    value: Any = None
    error: Exception | None = None
    done: threading.Event = field(default_factory=threading.Event)


class SingleFlight:
    """
    Coalesces concurrent calls for the same key into a single execution.

    Thread-safe. When multiple threads call ``do()`` with the same key,
    only the first thread runs the function; the others block until
    the result is available and then return the shared result.

    Attributes:
        _flights: Map of in-flight keys to their result holders.
        _lock: Protects _flights map access.
        _total_calls: Total number of do() calls.
        _shared_calls: Calls that shared another caller's result.
    """

    def __init__(self) -> None:
        self._flights: dict[str, _FlightResult] = {}
        self._lock = threading.Lock()
        self._total_calls = 0
        self._shared_calls = 0

    def do(self, key: str, fn: Callable[[], T]) -> T:
        """Execute fn() for key, coalescing concurrent calls.

        Args:
            key: Deduplication key.
            fn: Callable to execute on cache miss.

        Returns:
            The result of fn().

        Raises:
            Exception: Re-raises any exception from fn().
        """
        self._total_calls += 1

        with self._lock:
            if key in self._flights:
                flight = self._flights[key]
                self._shared_calls += 1
                is_owner = False
            else:
                flight = _FlightResult()
                self._flights[key] = flight
                is_owner = True

        if not is_owner:
            # Wait for the owner to finish
            flight.done.wait()
            if flight.error is not None:
                raise flight.error
            return cast(T, flight.value)

        # Owner: execute the function
        try:
            result = fn()
            flight.value = result
            return result
        except Exception as e:
            flight.error = e
            raise
        finally:
            flight.done.set()
            with self._lock:
                self._flights.pop(key, None)

    @property
    def stats(self) -> dict[str, Any]:
        """Return call statistics."""
        return {
            "total_calls": self._total_calls,
            "shared_calls": self._shared_calls,
            "in_flight": len(self._flights),
            "dedup_rate": (
                (self._shared_calls / self._total_calls * 100)
                if self._total_calls > 0
                else 0.0
            ),
        }

    def reset_stats(self) -> None:
        """Reset call counters."""
        self._total_calls = 0
        self._shared_calls = 0


# ---------------------------------------------------------------------------
# Async SingleFlight
# ---------------------------------------------------------------------------


@dataclass
class _AsyncFlightResult:
    """Result of an in-flight async computation."""

    value: Any = None
    error: Exception | None = None
    done: asyncio.Event = field(default_factory=asyncio.Event)


class AsyncSingleFlight:
    """
    Async version of SingleFlight for use with asyncio.

    When multiple coroutines call ``do()`` with the same key, only the
    first coroutine runs the function; the others await the shared result.
    """

    def __init__(self) -> None:
        self._flights: dict[str, _AsyncFlightResult] = {}
        self._lock = asyncio.Lock()
        self._total_calls = 0
        self._shared_calls = 0

    async def do(self, key: str, fn: Callable[[], Awaitable[T]]) -> T:
        """Execute fn() for key, coalescing concurrent calls.

        Args:
            key: Deduplication key.
            fn: Async callable to execute on cache miss.

        Returns:
            The result of fn().

        Raises:
            Exception: Re-raises any exception from fn().
        """
        self._total_calls += 1

        async with self._lock:
            if key in self._flights:
                flight = self._flights[key]
                self._shared_calls += 1
                is_owner = False
            else:
                flight = _AsyncFlightResult()
                self._flights[key] = flight
                is_owner = True

        if not is_owner:
            await flight.done.wait()
            if flight.error is not None:
                raise flight.error
            return cast(T, flight.value)

        try:
            result = await fn()
            flight.value = result
            return result
        except Exception as e:
            flight.error = e
            raise
        finally:
            flight.done.set()
            async with self._lock:
                self._flights.pop(key, None)

    @property
    def stats(self) -> dict[str, Any]:
        """Return call statistics."""
        return {
            "total_calls": self._total_calls,
            "shared_calls": self._shared_calls,
            "in_flight": len(self._flights),
            "dedup_rate": (
                (self._shared_calls / self._total_calls * 100)
                if self._total_calls > 0
                else 0.0
            ),
        }

    def reset_stats(self) -> None:
        """Reset call counters."""
        self._total_calls = 0
        self._shared_calls = 0


# ---------------------------------------------------------------------------
# Cache + SingleFlight decorator combos
# ---------------------------------------------------------------------------

# Global instances for decorator use
_sync_sf = SingleFlight()
_async_sf = AsyncSingleFlight()


def singleflight_cached(
    ttl_seconds: float = 300.0,
    maxsize: int = 128,
    key_prefix: str = "",
    skip_first: bool = True,
):
    """
    Decorator combining TTL cache with singleflight stampede prevention.

    On a cache miss, only one caller computes the value while others wait.
    The computed value is then cached for future requests.

    Args:
        ttl_seconds: Cache TTL.
        maxsize: Maximum cache entries.
        key_prefix: Optional key prefix.
        skip_first: Skip first arg (self) when building cache key.
    """
    from aragora.utils.cache import TTLCache

    cache = TTLCache[Any](maxsize=maxsize, ttl_seconds=ttl_seconds)

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            cache_args = args[1:] if skip_first and args else args
            parts = [key_prefix or func.__name__, str(cache_args)]
            if kwargs:
                parts.append(str(sorted(kwargs.items())))
            cache_key = ":".join(parts)

            # Fast path: cache hit
            cached = cache.get(cache_key)
            if cached is not None:
                return cached

            # Slow path: singleflight ensures one computation
            def compute() -> T:
                # Double-check cache inside singleflight (another flight
                # may have populated it between our check and acquiring the flight)
                cached_inner = cache.get(cache_key)
                if cached_inner is not None:
                    return cached_inner
                result = func(*args, **kwargs)
                cache.set(cache_key, result)
                return result

            return _sync_sf.do(cache_key, compute)

        setattr(wrapper, "cache", cache)
        setattr(wrapper, "singleflight", _sync_sf)
        setattr(wrapper, "cache_key_prefix", key_prefix or func.__name__)
        return cast(Callable[..., T], wrapper)

    return decorator


def async_singleflight_cached(
    ttl_seconds: float = 300.0,
    maxsize: int = 128,
    key_prefix: str = "",
    skip_first: bool = True,
):
    """
    Async decorator combining TTL cache with singleflight stampede prevention.

    On a cache miss, only one coroutine computes the value while others await.
    The computed value is then cached for future requests.

    Args:
        ttl_seconds: Cache TTL.
        maxsize: Maximum cache entries.
        key_prefix: Optional key prefix.
        skip_first: Skip first arg (self) when building cache key.
    """
    from aragora.utils.cache import TTLCache

    cache = TTLCache[Any](maxsize=maxsize, ttl_seconds=ttl_seconds)

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            cache_args = args[1:] if skip_first and args else args
            parts = [key_prefix or func.__name__, str(cache_args)]
            if kwargs:
                parts.append(str(sorted(kwargs.items())))
            cache_key = ":".join(parts)

            # Fast path: cache hit
            cached = cache.get(cache_key)
            if cached is not None:
                return cached

            # Slow path: async singleflight ensures one computation
            async def compute() -> T:
                cached_inner = cache.get(cache_key)
                if cached_inner is not None:
                    return cached_inner
                result = await func(*args, **kwargs)
                cache.set(cache_key, result)
                return result

            return await _async_sf.do(cache_key, compute)

        setattr(wrapper, "cache", cache)
        setattr(wrapper, "singleflight", _async_sf)
        setattr(wrapper, "cache_key_prefix", key_prefix or func.__name__)
        return cast(Callable[..., Awaitable[T]], wrapper)

    return decorator


def get_singleflight_stats() -> dict[str, dict[str, Any]]:
    """Get statistics for global singleflight instances."""
    return {
        "sync": _sync_sf.stats,
        "async": _async_sf.stats,
    }


__all__ = [
    "SingleFlight",
    "AsyncSingleFlight",
    "singleflight_cached",
    "async_singleflight_cached",
    "get_singleflight_stats",
]
