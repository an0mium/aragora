"""
Adaptive TTL Cache with Access Pattern Analysis.

Dynamically adjusts cache TTL based on access patterns:
- Frequently accessed items get longer TTL
- Infrequently accessed items get shorter TTL
- Detects hot spots and optimizes accordingly

Features:
- Access frequency tracking
- Dynamic TTL adjustment
- Hot spot detection
- Cache size optimization
- Prometheus metrics integration

Usage:
    cache = AdaptiveTTLCache(
        name="user_cache",
        base_ttl=60,        # Base TTL in seconds
        min_ttl=10,         # Minimum TTL
        max_ttl=3600,       # Maximum TTL (1 hour)
    )

    await cache.set("user:123", user_data)
    user = await cache.get("user:123")
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Generic, List, Optional, TypeVar

logger = logging.getLogger(__name__)

K = TypeVar("K")
V = TypeVar("V")


@dataclass
class AccessPattern:
    """Tracks access patterns for a cache key."""

    key: str
    access_count: int = 0
    last_access: float = 0.0
    first_access: float = 0.0
    access_times: List[float] = field(default_factory=list)

    # Adaptive TTL
    current_ttl: float = 0.0

    # Hot spot detection
    is_hot: bool = False

    @property
    def access_frequency(self) -> float:
        """Calculate access frequency (accesses per second)."""
        if self.access_count < 2:
            return 0.0
        duration = self.last_access - self.first_access
        if duration <= 0:
            return 0.0
        return self.access_count / duration

    @property
    def avg_interval_ms(self) -> float:
        """Average time between accesses in milliseconds."""
        if len(self.access_times) < 2:
            return float("inf")
        intervals = [
            (self.access_times[i] - self.access_times[i - 1]) * 1000
            for i in range(1, len(self.access_times))
        ]
        return sum(intervals) / len(intervals)

    def record_access(self, max_history: int = 100) -> None:
        """Record an access event."""
        now = time.time()
        self.access_count += 1
        self.last_access = now

        if self.first_access == 0:
            self.first_access = now

        self.access_times.append(now)
        # Keep only recent history
        if len(self.access_times) > max_history:
            self.access_times = self.access_times[-max_history:]


@dataclass
class CacheEntry(Generic[V]):
    """A cached value with metadata."""

    value: V
    created_at: float
    expires_at: float
    access_pattern: AccessPattern


@dataclass
class CacheStats:
    """Statistics for the adaptive cache."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    ttl_adjustments: int = 0
    hot_spots: int = 0
    avg_ttl_seconds: float = 0.0

    @property
    def hit_rate(self) -> float:
        """Cache hit rate as percentage."""
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return (self.hits / total) * 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "hit_rate": f"{self.hit_rate:.1f}%",
            "ttl_adjustments": self.ttl_adjustments,
            "hot_spots": self.hot_spots,
            "avg_ttl_seconds": round(self.avg_ttl_seconds, 1),
        }


class AdaptiveTTLCache(Generic[K, V]):
    """
    Cache with dynamic TTL based on access patterns.

    Automatically adjusts TTL:
    - Frequently accessed items: TTL increases toward max_ttl
    - Infrequently accessed items: TTL decreases toward min_ttl
    - Hot spots (very frequent access): TTL set to max_ttl
    """

    # TTL adjustment thresholds
    HOT_SPOT_FREQUENCY = 10.0  # Accesses per second to be considered hot
    INCREASE_FREQUENCY = 1.0  # Frequency to increase TTL
    DECREASE_FREQUENCY = 0.1  # Frequency to decrease TTL

    def __init__(
        self,
        name: str = "adaptive_cache",
        base_ttl: float = 60.0,
        min_ttl: float = 10.0,
        max_ttl: float = 3600.0,
        max_size: int = 10000,
        ttl_adjustment_factor: float = 1.5,
        cleanup_interval: float = 60.0,
    ):
        """
        Initialize adaptive cache.

        Args:
            name: Cache name for logging/metrics
            base_ttl: Default TTL in seconds
            min_ttl: Minimum TTL in seconds
            max_ttl: Maximum TTL in seconds
            max_size: Maximum number of entries
            ttl_adjustment_factor: Factor for TTL adjustments
            cleanup_interval: Seconds between cleanup runs
        """
        self._name = name
        self._base_ttl = base_ttl
        self._min_ttl = min_ttl
        self._max_ttl = max_ttl
        self._max_size = max_size
        self._adjustment_factor = ttl_adjustment_factor
        self._cleanup_interval = cleanup_interval

        self._cache: Dict[str, CacheEntry[V]] = {}
        self._lock = threading.RLock()
        self._stats = CacheStats()

        # Background cleanup
        self._cleanup_task: Optional[asyncio.Task[None]] = None
        self._running = False

    async def start(self) -> None:
        """Start background cleanup task."""
        if self._running:
            return
        self._running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.debug(f"AdaptiveTTLCache '{self._name}' started")

    async def stop(self) -> None:
        """Stop background cleanup task."""
        self._running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        logger.debug(f"AdaptiveTTLCache '{self._name}' stopped")

    async def _cleanup_loop(self) -> None:
        """Background cleanup of expired entries."""
        while self._running:
            try:
                await asyncio.sleep(self._cleanup_interval)
                self._cleanup_expired()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Cleanup error in {self._name}: {e}")

    def _cleanup_expired(self) -> None:
        """Remove expired entries."""
        now = time.time()
        with self._lock:
            expired = [key for key, entry in self._cache.items() if entry.expires_at <= now]
            for key in expired:
                del self._cache[key]
                self._stats.evictions += 1

    def _calculate_ttl(self, pattern: AccessPattern) -> float:
        """Calculate TTL based on access pattern."""
        freq = pattern.access_frequency

        # Hot spot: max TTL
        if freq >= self.HOT_SPOT_FREQUENCY:
            pattern.is_hot = True
            self._stats.hot_spots += 1
            return self._max_ttl

        pattern.is_hot = False

        # Adjust based on frequency
        if freq >= self.INCREASE_FREQUENCY:
            # Increase TTL
            new_ttl = pattern.current_ttl * self._adjustment_factor
        elif freq <= self.DECREASE_FREQUENCY:
            # Decrease TTL
            new_ttl = pattern.current_ttl / self._adjustment_factor
        else:
            # Keep current
            new_ttl = pattern.current_ttl

        # Clamp to bounds
        return max(self._min_ttl, min(self._max_ttl, new_ttl))

    def _evict_if_needed(self) -> None:
        """Evict entries if cache is full."""
        if len(self._cache) < self._max_size:
            return

        # Evict least recently used, non-hot entries first
        with self._lock:
            entries = sorted(
                self._cache.items(),
                key=lambda x: (x[1].access_pattern.is_hot, x[1].access_pattern.last_access),
            )
            # Evict 10% of entries
            to_evict = max(1, len(entries) // 10)
            for key, _ in entries[:to_evict]:
                del self._cache[key]
                self._stats.evictions += 1

    async def get(self, key: K) -> Optional[V]:
        """
        Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        str_key = str(key)
        now = time.time()

        with self._lock:
            entry = self._cache.get(str_key)

            if entry is None:
                self._stats.misses += 1
                return None

            # Check expiration
            if entry.expires_at <= now:
                del self._cache[str_key]
                self._stats.misses += 1
                self._stats.evictions += 1
                return None

            # Record access and adjust TTL
            entry.access_pattern.record_access()
            old_ttl = entry.access_pattern.current_ttl
            new_ttl = self._calculate_ttl(entry.access_pattern)

            if new_ttl != old_ttl:
                entry.access_pattern.current_ttl = new_ttl
                # Extend expiration if TTL increased
                if new_ttl > old_ttl:
                    entry.expires_at = now + new_ttl
                self._stats.ttl_adjustments += 1

            self._stats.hits += 1
            return entry.value

    async def set(
        self,
        key: K,
        value: V,
        ttl: Optional[float] = None,
    ) -> None:
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Optional explicit TTL (overrides adaptive)
        """
        str_key = str(key)
        now = time.time()
        actual_ttl = ttl if ttl is not None else self._base_ttl

        with self._lock:
            self._evict_if_needed()

            # Get or create access pattern
            existing = self._cache.get(str_key)
            if existing:
                pattern = existing.access_pattern
                pattern.record_access()
            else:
                pattern = AccessPattern(key=str_key, current_ttl=actual_ttl)
                pattern.record_access()

            # Create entry
            self._cache[str_key] = CacheEntry(
                value=value,
                created_at=now,
                expires_at=now + actual_ttl,
                access_pattern=pattern,
            )

    async def delete(self, key: K) -> bool:
        """
        Delete value from cache.

        Args:
            key: Cache key

        Returns:
            True if key was present
        """
        str_key = str(key)
        with self._lock:
            if str_key in self._cache:
                del self._cache[str_key]
                return True
            return False

    def clear(self) -> None:
        """Clear all entries."""
        with self._lock:
            self._cache.clear()

    @property
    def stats(self) -> CacheStats:
        """Get cache statistics."""
        with self._lock:
            if self._cache:
                ttls = [e.access_pattern.current_ttl for e in self._cache.values()]
                self._stats.avg_ttl_seconds = sum(ttls) / len(ttls)
            else:
                self._stats.avg_ttl_seconds = 0.0
        return self._stats

    @property
    def size(self) -> int:
        """Current cache size."""
        return len(self._cache)

    def get_hot_spots(self) -> List[str]:
        """Get list of hot spot keys."""
        with self._lock:
            return [key for key, entry in self._cache.items() if entry.access_pattern.is_hot]


class CacheOptimizer:
    """
    Analyzes cache performance and provides optimization recommendations.
    """

    def __init__(self, cache: AdaptiveTTLCache[Any, Any]):
        """Initialize optimizer with a cache to monitor."""
        self._cache = cache
        self._history: List[Dict[str, Any]] = []

    def record_snapshot(self) -> Dict[str, Any]:
        """Record current cache state."""
        snapshot = {
            "timestamp": time.time(),
            "size": self._cache.size,
            "stats": self._cache.stats.to_dict(),
            "hot_spots_count": len(self._cache.get_hot_spots()),
        }
        self._history.append(snapshot)
        # Keep last 100 snapshots
        if len(self._history) > 100:
            self._history = self._history[-100:]
        return snapshot

    def get_recommendations(self) -> List[str]:
        """Get optimization recommendations based on cache behavior."""
        recommendations = []
        stats = self._cache.stats

        # Low hit rate
        if stats.hit_rate < 50:
            recommendations.append(
                f"Hit rate is low ({stats.hit_rate:.1f}%). "
                "Consider increasing base TTL or cache size."
            )

        # High eviction rate
        if stats.evictions > stats.hits:
            recommendations.append(
                "High eviction rate detected. " "Consider increasing max_size to reduce churn."
            )

        # Many hot spots
        hot_spots = len(self._cache.get_hot_spots())
        if hot_spots > self._cache.size * 0.5:
            recommendations.append(
                f"Many hot spots ({hot_spots}). " "Consider dedicated caching for hot keys."
            )

        # Low TTL adjustment
        if stats.ttl_adjustments == 0 and stats.hits > 100:
            recommendations.append(
                "No TTL adjustments made. " "Access patterns may be too uniform."
            )

        return recommendations


__all__ = [
    "AccessPattern",
    "CacheEntry",
    "CacheStats",
    "AdaptiveTTLCache",
    "CacheOptimizer",
]
