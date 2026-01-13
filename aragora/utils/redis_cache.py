"""
Redis-backed TTL cache with automatic fallback.

Provides a drop-in replacement for TTLCache that uses Redis for distributed
caching when available, with automatic fallback to in-memory caching.

Usage:
    from aragora.utils.redis_cache import RedisTTLCache

    cache = RedisTTLCache(
        prefix="leaderboard",
        ttl_seconds=300,
        maxsize=1000,  # For fallback in-memory cache
    )
    cache.set("key", {"data": "value"})
    result = cache.get("key")
"""

import json
import logging
import threading
import time
from collections import OrderedDict
from typing import Any, Generic, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class RedisTTLCache(Generic[T]):
    """
    TTL cache with Redis backend and in-memory fallback.

    When Redis is available, uses it for distributed caching across
    multiple server instances. Falls back to in-memory caching when
    Redis is unavailable.

    Features:
    - Automatic Redis detection and fallback
    - JSON serialization for complex values
    - TTL-based expiration
    - Prefix-based key namespacing
    - Statistics tracking
    """

    def __init__(
        self,
        prefix: str = "cache",
        ttl_seconds: float = 300.0,
        maxsize: int = 1000,
    ):
        """Initialize the cache.

        Args:
            prefix: Key prefix for Redis (namespace isolation)
            ttl_seconds: Time-to-live for entries
            maxsize: Max entries for in-memory fallback
        """
        self._prefix = prefix
        self._ttl_seconds = ttl_seconds
        self._maxsize = maxsize

        # In-memory fallback cache
        self._memory_cache: OrderedDict[str, tuple[float, T]] = OrderedDict()
        self._lock = threading.RLock()

        # Statistics
        self._hits = 0
        self._misses = 0
        self._redis_hits = 0
        self._redis_misses = 0

        # Redis client (lazy initialized)
        self._redis: Optional[Any] = None
        self._redis_checked = False

    def _get_redis(self) -> Optional[Any]:
        """Get Redis client (lazy initialization)."""
        if self._redis_checked:
            return self._redis

        try:
            from aragora.server.redis_config import get_redis_client

            self._redis = get_redis_client()
            self._redis_checked = True
            if self._redis:
                logger.debug(f"RedisTTLCache[{self._prefix}] using Redis backend")
            else:
                logger.debug(f"RedisTTLCache[{self._prefix}] using in-memory fallback")
        except ImportError:
            self._redis_checked = True

        return self._redis

    def _redis_key(self, key: str) -> str:
        """Build Redis key with prefix."""
        return f"aragora:{self._prefix}:{key}"

    def get(self, key: str) -> Optional[T]:
        """Get a value from cache.

        Tries Redis first, falls back to in-memory cache.
        """
        redis = self._get_redis()

        if redis is not None:
            try:
                redis_key = self._redis_key(key)
                data = redis.get(redis_key)
                if data is not None:
                    self._hits += 1
                    self._redis_hits += 1
                    return json.loads(data)
                self._misses += 1
                self._redis_misses += 1
                return None
            except Exception as e:
                logger.debug(f"Redis get failed, using fallback: {e}")
                # Fall through to memory cache

        # In-memory fallback
        with self._lock:
            if key in self._memory_cache:
                cached_time, value = self._memory_cache[key]
                if time.time() - cached_time < self._ttl_seconds:
                    self._memory_cache.move_to_end(key)
                    self._hits += 1
                    return value
                else:
                    del self._memory_cache[key]
            self._misses += 1
            return None

    def set(self, key: str, value: T) -> None:
        """Store a value in cache.

        Writes to Redis if available, always writes to in-memory as backup.
        """
        redis = self._get_redis()

        if redis is not None:
            try:
                redis_key = self._redis_key(key)
                redis.setex(
                    redis_key,
                    int(self._ttl_seconds),
                    json.dumps(value, default=str),
                )
            except Exception as e:
                logger.debug(f"Redis set failed: {e}")

        # Always write to memory cache (backup and for stats)
        with self._lock:
            while len(self._memory_cache) >= self._maxsize:
                self._memory_cache.popitem(last=False)
            self._memory_cache[key] = (time.time(), value)

    def invalidate(self, key: str) -> bool:
        """Invalidate a specific key.

        Returns True if key was found in either cache.
        """
        found = False
        redis = self._get_redis()

        if redis is not None:
            try:
                redis_key = self._redis_key(key)
                if redis.delete(redis_key) > 0:
                    found = True
            except Exception as e:
                logger.debug(f"Redis invalidate failed: {e}")

        with self._lock:
            if key in self._memory_cache:
                del self._memory_cache[key]
                found = True

        return found

    def clear(self) -> int:
        """Clear all entries with this prefix.

        Returns count of in-memory entries cleared.
        """
        redis = self._get_redis()

        if redis is not None:
            try:
                pattern = self._redis_key("*")
                keys = redis.keys(pattern)
                if keys:
                    redis.delete(*keys)
            except Exception as e:
                logger.debug(f"Redis clear failed: {e}")

        with self._lock:
            count = len(self._memory_cache)
            self._memory_cache.clear()
            return count

    def clear_prefix(self, prefix: str) -> int:
        """Clear entries with keys starting with prefix.

        Returns count of in-memory entries cleared.
        """
        redis = self._get_redis()

        if redis is not None:
            try:
                pattern = self._redis_key(f"{prefix}*")
                keys = redis.keys(pattern)
                if keys:
                    redis.delete(*keys)
            except Exception as e:
                logger.debug(f"Redis clear_prefix failed: {e}")

        with self._lock:
            keys_to_remove = [k for k in self._memory_cache if k.startswith(prefix)]
            for k in keys_to_remove:
                del self._memory_cache[k]
            return len(keys_to_remove)

    @property
    def stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        redis = self._get_redis()
        using_redis = redis is not None

        with self._lock:
            total = self._hits + self._misses
            return {
                "size": len(self._memory_cache),
                "maxsize": self._maxsize,
                "ttl_seconds": self._ttl_seconds,
                "prefix": self._prefix,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": self._hits / total if total > 0 else 0.0,
                "using_redis": using_redis,
                "redis_hits": self._redis_hits,
                "redis_misses": self._redis_misses,
            }

    def __len__(self) -> int:
        """Return count of in-memory entries."""
        return len(self._memory_cache)


def HybridTTLCache(
    prefix: str = "cache",
    ttl_seconds: float = 300.0,
    maxsize: int = 1000,
) -> "RedisTTLCache[Any]":
    """
    Factory function that returns the best available cache implementation.

    Uses RedisTTLCache (with Redis backend) if Redis is available,
    otherwise uses a standard in-memory TTLCache.

    Args:
        prefix: Key prefix for Redis (namespace isolation)
        ttl_seconds: Time-to-live for entries
        maxsize: Max entries for in-memory fallback

    Returns:
        RedisTTLCache instance which handles Redis availability internally.
    """
    return RedisTTLCache(
        prefix=prefix,
        ttl_seconds=ttl_seconds,
        maxsize=maxsize,
    )


__all__ = [
    "RedisTTLCache",
    "HybridTTLCache",
]
