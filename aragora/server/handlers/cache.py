"""
Handler caching utilities with TTL expiry and bounded size.

Provides:
- BoundedTTLCache: Thread-safe TTL cache with LRU eviction
- ttl_cache: Decorator for caching function results
- Event-driven cache invalidation
- Cache statistics for monitoring
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import OrderedDict
from functools import wraps
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Lazy import to avoid circular dependency
_record_cache_hit = None
_record_cache_miss = None


def _get_metrics() -> tuple:
    """Lazy load metrics functions."""
    global _record_cache_hit, _record_cache_miss
    if _record_cache_hit is None:
        try:
            from aragora.observability.metrics import record_cache_hit, record_cache_miss
            _record_cache_hit = record_cache_hit
            _record_cache_miss = record_cache_miss
        except ImportError:
            _record_cache_hit = lambda x: None
            _record_cache_miss = lambda x: None
    return _record_cache_hit, _record_cache_miss


# Cache configuration from environment
CACHE_MAX_ENTRIES = int(os.environ.get("ARAGORA_CACHE_MAX_ENTRIES", "1000"))
CACHE_EVICT_PERCENT = float(os.environ.get("ARAGORA_CACHE_EVICT_PERCENT", "0.1"))


class BoundedTTLCache:
    """
    Thread-safe TTL cache with bounded size and LRU eviction.

    Prevents memory leaks by limiting the number of entries and
    evicting oldest entries when the limit is reached.
    """

    def __init__(self, max_entries: int = CACHE_MAX_ENTRIES, evict_percent: float = CACHE_EVICT_PERCENT):
        self._cache: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._lock = threading.RLock()  # Thread safety lock
        self._max_entries = max_entries
        self._evict_count = max(1, int(max_entries * evict_percent))
        self._hits = 0
        self._misses = 0

    def get(self, key: str, ttl_seconds: float) -> tuple[bool, Any]:
        """
        Get a value from cache if not expired (thread-safe).

        Returns:
            Tuple of (hit, value). If hit is False, value is None.
        """
        now = time.time()

        with self._lock:
            if key in self._cache:
                cached_time, cached_value = self._cache[key]
                if now - cached_time < ttl_seconds:
                    # Move to end (most recently used)
                    self._cache.move_to_end(key)
                    self._hits += 1
                    return True, cached_value
                else:
                    # Expired - remove it
                    del self._cache[key]

            self._misses += 1
            return False, None

    def set(self, key: str, value: Any) -> None:
        """Store a value in cache, evicting old entries if necessary (thread-safe)."""
        now = time.time()

        with self._lock:
            # If key exists, update and move to end
            if key in self._cache:
                self._cache[key] = (now, value)
                self._cache.move_to_end(key)
                return

            # Check if we need to evict
            if len(self._cache) >= self._max_entries:
                self._evict_oldest_unlocked()

            # Add new entry
            self._cache[key] = (now, value)

    def _evict_oldest_unlocked(self) -> int:
        """Evict oldest entries to make room (must hold lock). Returns count evicted."""
        evicted = 0
        for _ in range(self._evict_count):
            if self._cache:
                self._cache.popitem(last=False)
                evicted += 1
        if evicted > 0:
            logger.debug(f"Cache evicted {evicted} entries (size: {len(self._cache)})")
        return evicted

    def clear(self, key_prefix: str | None = None) -> int:
        """Clear entries, optionally filtered by prefix (thread-safe)."""
        with self._lock:
            if key_prefix is None:
                count = len(self._cache)
                self._cache.clear()
                return count
            else:
                keys_to_remove = [k for k in self._cache if k.startswith(key_prefix)]
                for k in keys_to_remove:
                    del self._cache[k]
                return len(keys_to_remove)

    def invalidate_containing(self, substring: str) -> int:
        """Invalidate all keys containing substring (thread-safe)."""
        with self._lock:
            keys_to_remove = [k for k in self._cache if substring in k]
            for k in keys_to_remove:
                del self._cache[k]
            return len(keys_to_remove)

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)

    def __contains__(self, key: str) -> bool:
        with self._lock:
            return key in self._cache

    def items(self) -> list:
        """Return a copy of cache items (thread-safe snapshot)."""
        with self._lock:
            return list(self._cache.items())

    @property
    def stats(self) -> dict:
        """Get cache statistics (thread-safe)."""
        with self._lock:
            total = self._hits + self._misses
            return {
                "entries": len(self._cache),
                "max_entries": self._max_entries,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": self._hits / total if total > 0 else 0.0,
            }


# Global bounded cache instance
_cache = BoundedTTLCache()

# Track registration status
_handler_cache_registered = False


def _register_handler_cache() -> None:
    """Register handler cache with ServiceRegistry for observability."""
    global _handler_cache_registered
    if _handler_cache_registered:
        return

    try:
        from aragora.services import ServiceRegistry, HandlerCacheService

        registry = ServiceRegistry.get()
        if not registry.has(HandlerCacheService):
            registry.register(HandlerCacheService, _cache)
        _handler_cache_registered = True
        logger.debug("Handler cache registered with ServiceRegistry")
    except ImportError:
        pass  # Services module not available


def get_handler_cache() -> BoundedTTLCache:
    """Get the global handler cache, registering with ServiceRegistry if available."""
    _register_handler_cache()
    return _cache


def ttl_cache(ttl_seconds: float = 60.0, key_prefix: str = "", skip_first: bool = True):
    """
    Decorator for caching function results with TTL expiry.

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
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            record_hit, record_miss = _get_metrics()

            # Skip 'self' when building cache key for methods
            cache_args = args[1:] if skip_first and args else args
            # Build cache key from function name, args and kwargs
            cache_key = f"{key_prefix}:{func.__name__}:{cache_args}:{sorted(kwargs.items())}"

            hit, cached_value = _cache.get(cache_key, ttl_seconds)
            if hit:
                record_hit(key_prefix or func.__name__)
                logger.debug(f"Cache hit for {cache_key}")
                return cached_value

            # Cache miss or expired
            record_miss(key_prefix or func.__name__)
            result = func(*args, **kwargs)
            _cache.set(cache_key, result)
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

    Usage:
        @async_ttl_cache(ttl_seconds=120, key_prefix="graph_debate")
        async def _get_graph_debate(self, debate_id: str):
            ...
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            record_hit, record_miss = _get_metrics()

            # Skip 'self' when building cache key for methods
            cache_args = args[1:] if skip_first and args else args
            # Build cache key from function name, args and kwargs
            cache_key = f"{key_prefix}:{func.__name__}:{cache_args}:{sorted(kwargs.items())}"

            hit, cached_value = _cache.get(cache_key, ttl_seconds)
            if hit:
                record_hit(key_prefix or func.__name__)
                logger.debug(f"Cache hit for {cache_key}")
                return cached_value

            # Cache miss or expired
            record_miss(key_prefix or func.__name__)
            result = await func(*args, **kwargs)
            _cache.set(cache_key, result)
            logger.debug(f"Cache miss, stored {cache_key}")
            return result
        return wrapper
    return decorator


def clear_cache(key_prefix: str | None = None) -> int:
    """Clear cached entries, optionally filtered by prefix.

    Returns number of entries cleared.
    """
    return _cache.clear(key_prefix)


def invalidates_cache(*events: str):
    """
    Decorator that invalidates cache entries after a function completes.

    Use this decorator on methods that modify data to ensure related
    caches are automatically invalidated, preventing stale data.

    Args:
        *events: Event names from CACHE_INVALIDATION_MAP (e.g., "elo_updated",
                "debate_completed"). Each event invalidates its associated
                cache prefixes.

    Usage:
        @invalidates_cache("elo_updated", "match_recorded")
        def record_match(self, winner: str, loser: str):
            # Record match logic
            ...
            # Caches for leaderboard, rankings, etc. are auto-invalidated

        @invalidates_cache("debate_completed")
        async def complete_debate(self, debate_id: str):
            # Complete debate logic
            ...

    The decorator:
    1. Executes the wrapped function
    2. If successful, invalidates all caches for the specified events
    3. Logs the number of cache entries cleared
    4. Returns the function's result

    Note: For async functions, use with await; the decorator auto-detects.
    """
    import asyncio
    from functools import wraps

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            _invalidate_events(events, func.__name__)
            return result

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            result = await func(*args, **kwargs)
            _invalidate_events(events, func.__name__)
            return result

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


def _invalidate_events(events: tuple[str, ...], func_name: str) -> int:
    """Helper to invalidate cache for multiple events."""
    total_cleared = 0
    for event in events:
        cleared = invalidate_on_event(event)
        total_cleared += cleared
    if total_cleared > 0:
        logger.debug(f"@invalidates_cache({events}): {func_name} cleared {total_cleared} entries")
    return total_cleared


def get_cache_stats() -> dict:
    """Get cache statistics for monitoring."""
    return _cache.stats


# =============================================================================
# Event-Driven Cache Invalidation
# =============================================================================

# Maps event names to cache prefixes that should be invalidated
CACHE_INVALIDATION_MAP: dict[str, list[str]] = {
    # ELO/ranking events
    "elo_updated": [
        "leaderboard", "lb_rankings", "agents_list", "agent_profile",
        "calibration_lb", "recent_matches", "analytics_ranking",
    ],
    "match_recorded": [
        "leaderboard", "lb_rankings", "lb_matches", "recent_matches",
        "agent_h2h", "analytics_ranking",
    ],
    # Debate events
    "debate_completed": [
        "dashboard_debates", "analytics_debates", "replays_list",
        "consensus_stats", "consensus_similar",
    ],
    "debate_started": [
        "dashboard_debates",
    ],
    # Agent events
    "agent_updated": [
        "agent_profile", "agents_list", "lb_introspection",
    ],
    # Memory events
    "memory_updated": [
        "analytics_memory", "critique_patterns", "critique_stats",
    ],
    # Consensus events
    "consensus_reached": [
        "consensus_stats", "consensus_settled", "consensus_similar",
    ],
}


def invalidate_on_event(event_name: str) -> int:
    """Invalidate cache entries associated with an event.

    Args:
        event_name: Name of the event (e.g., "elo_updated", "debate_completed")

    Returns:
        Total number of cache entries invalidated
    """
    prefixes = CACHE_INVALIDATION_MAP.get(event_name, [])
    total_cleared = 0
    for prefix in prefixes:
        cleared = _cache.clear(prefix)
        total_cleared += cleared
        if cleared > 0:
            logger.debug(f"Cache invalidation: {event_name} cleared {cleared} entries with prefix '{prefix}'")
    if total_cleared > 0:
        logger.info(f"Cache invalidated: event={event_name}, entries_cleared={total_cleared}")
    return total_cleared


def invalidate_cache(data_source: str) -> int:
    """Invalidate cache entries associated with a data source.

    This function clears all cache prefixes registered for a given data source
    in the CACHE_INVALIDATION_MAP.

    Args:
        data_source: Name of the data source (e.g., "elo", "memory", "debates")

    Returns:
        Total number of cache entries invalidated
    """
    # Map data sources to event names
    source_to_event = {
        "elo": "elo_updated",
        "memory": "memory_updated",
        "debates": "debate_completed",
        "consensus": "consensus_reached",
        "agent": "agent_updated",
        "calibration": "elo_updated",  # Uses same prefixes as ELO
    }
    event_name = source_to_event.get(data_source)
    if event_name:
        return invalidate_on_event(event_name)

    # Fallback: try to clear by prefix directly
    cleared = _cache.clear(data_source)
    if cleared > 0:
        logger.debug(f"Cache invalidation: cleared {cleared} entries with prefix '{data_source}'")
    return cleared


def invalidate_leaderboard_cache() -> int:
    """Convenience function to invalidate all leaderboard-related caches."""
    return invalidate_on_event("elo_updated")


def invalidate_agent_cache(agent_name: str | None = None) -> int:
    """Invalidate agent-related cache entries.

    Args:
        agent_name: If provided, only invalidate entries for this agent.
                   If None, invalidate all agent caches.
    """
    if agent_name:
        # Invalidate entries containing the agent name (thread-safe)
        return _cache.invalidate_containing(agent_name)
    else:
        return invalidate_on_event("agent_updated")


def invalidate_debate_cache(debate_id: str | None = None) -> int:
    """Invalidate debate-related cache entries.

    Args:
        debate_id: If provided, only invalidate entries for this debate.
                  If None, invalidate all debate caches.
    """
    if debate_id:
        # Invalidate entries containing the debate ID (thread-safe)
        return _cache.invalidate_containing(debate_id)
    else:
        return invalidate_on_event("debate_completed")
