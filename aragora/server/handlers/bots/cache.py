"""
Bounded TTL cache for bot handler state.

Prevents unbounded memory growth from user-controlled input
by enforcing maximum size and time-to-live eviction.

Optimizations:
- Min-heap expiration queue for O(log n) cleanup instead of O(n)
- Lazy cleanup only triggers when heap size exceeds threshold
- Individual expired entries removed lazily on access
"""

from __future__ import annotations

import heapq
import logging
import threading
import time
from typing import Any, TypeVar
from collections.abc import Iterator

logger = logging.getLogger(__name__)

V = TypeVar("V")


class BoundedTTLDict:
    """Thread-safe dictionary with maximum size and per-entry TTL expiration.

    Entries are evicted in two ways:
    - **TTL expiration**: Entries older than ``ttl_seconds`` are lazily removed
      on access (get/set/contains) and eagerly removed via ``cleanup()``.
    - **LRU eviction**: When inserting a new key would exceed ``max_size``,
      the oldest entries (by insertion time) are evicted to make room.

    All public methods acquire an internal lock, making this class safe to
    use from multiple threads.

    Performance:
    - Uses a min-heap for O(log n) amortized cleanup of expired entries
    - Individual expired entries are removed lazily on access (O(1))
    - Full cleanup only processes entries at front of heap until non-expired

    Args:
        max_size: Maximum number of entries. Must be >= 1.
        ttl_seconds: Time-to-live for each entry in seconds. Must be > 0.
        name: Human-readable name used in log messages.
    """

    __slots__ = (
        "_data",
        "_expiry_heap",
        "_lock",
        "_max_size",
        "_name",
        "_ttl_seconds",
    )

    def __init__(self, max_size: int, ttl_seconds: int, name: str = "cache") -> None:
        if max_size < 1:
            raise ValueError(f"max_size must be >= 1, got {max_size}")
        if ttl_seconds <= 0:
            raise ValueError(f"ttl_seconds must be > 0, got {ttl_seconds}")
        self._max_size = max_size
        self._ttl_seconds = ttl_seconds
        self._name = name
        self._lock = threading.Lock()
        # _data maps key -> (value, insertion_timestamp)
        self._data: dict[str, tuple[Any, float]] = {}
        # Min-heap of (expiry_time, key) for efficient expired entry discovery
        self._expiry_heap: list[tuple[float, str]] = []

    # ------------------------------------------------------------------
    # Internal helpers (caller must hold self._lock)
    # ------------------------------------------------------------------

    def _is_expired(self, key: str) -> bool:
        """Check if a key has expired. Caller must hold the lock."""
        entry = self._data.get(key)
        if entry is None:
            return True
        _, ts = entry
        return (time.monotonic() - ts) >= self._ttl_seconds

    def _evict_expired(self) -> None:
        """Remove expired entries using min-heap for O(log n) amortized cleanup.

        Caller must hold the lock.

        Uses a min-heap of (expiry_time, key) to efficiently find expired entries.
        Entries are popped from the heap while expired. Stale heap entries (where
        the key was already deleted or updated) are skipped.

        Also performs a fallback scan for entries not tracked by the heap (e.g.,
        entries whose timestamps were modified directly for testing).
        """
        now = time.monotonic()
        evicted_count = 0

        # Phase 1: Fast path using heap
        while self._expiry_heap:
            expiry_time, key = self._expiry_heap[0]

            # If front of heap is not expired, nothing more to evict via heap
            if expiry_time > now:
                break

            # Pop this entry from the heap
            heapq.heappop(self._expiry_heap)

            # Check if this heap entry is still valid (key exists with same timestamp)
            entry = self._data.get(key)
            if entry is None:
                # Key was already deleted, skip
                continue

            _, ts = entry
            entry_expiry = ts + self._ttl_seconds

            # Check if this is a stale heap entry (key was updated with new timestamp)
            # Use small epsilon for float comparison
            if abs(entry_expiry - expiry_time) > 0.001:
                # Entry was updated, there's a newer heap entry for it
                continue

            # Entry is expired, remove it
            del self._data[key]
            evicted_count += 1

        # Phase 2: Fallback scan for entries not tracked by heap
        # This handles entries whose timestamps were modified directly
        # (e.g., for testing purposes) without going through __setitem__
        expired_keys = [k for k, (_, ts) in self._data.items() if (now - ts) >= self._ttl_seconds]
        for k in expired_keys:
            del self._data[k]
            evicted_count += 1

        if evicted_count > 0:
            logger.debug("[%s] evicted %d expired entries", self._name, evicted_count)

    def _evict_oldest(self, count: int) -> None:
        """Remove the *count* oldest entries by insertion time. Caller must hold the lock."""
        if count <= 0:
            return
        # dict preserves insertion order in Python 3.7+; oldest are first
        keys_to_remove = list(self._data.keys())[:count]
        for k in keys_to_remove:
            del self._data[k]
            logger.debug("[%s] LRU eviction", self._name)

    # ------------------------------------------------------------------
    # Public dict-like interface
    # ------------------------------------------------------------------

    def __setitem__(self, key: str, value: Any) -> None:
        with self._lock:
            # Lazy cleanup of expired entries on write
            self._evict_expired()
            # If key already exists, remove so re-insertion moves it to end
            self._data.pop(key, None)
            # Evict oldest if at capacity
            overflow = len(self._data) - self._max_size + 1
            if overflow > 0:
                self._evict_oldest(overflow)
            ts = time.monotonic()
            self._data[key] = (value, ts)
            # Push expiry time to heap for O(log n) cleanup
            heapq.heappush(self._expiry_heap, (ts + self._ttl_seconds, key))

    def __getitem__(self, key: str) -> Any:
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                raise KeyError(key)
            value, ts = entry
            if (time.monotonic() - ts) >= self._ttl_seconds:
                del self._data[key]
                logger.debug("[%s] expired key evicted on read", self._name)
                raise KeyError(key)
            return value

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return False
            _, ts = entry
            if (time.monotonic() - ts) >= self._ttl_seconds:
                del self._data[key]
                logger.debug("[%s] expired key evicted on contains check", self._name)
                return False
            return True

    def __delitem__(self, key: str) -> None:
        with self._lock:
            if key not in self._data:
                raise KeyError(key)
            del self._data[key]

    def __len__(self) -> int:
        with self._lock:
            self._evict_expired()
            return len(self._data)

    def __bool__(self) -> bool:
        with self._lock:
            self._evict_expired()
            return bool(self._data)

    def __iter__(self) -> Iterator[str]:
        with self._lock:
            self._evict_expired()
            return iter(list(self._data.keys()))

    def get(self, key: str, default: Any = None) -> Any:
        """Return the value for *key* if present and not expired, else *default*."""
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return default
            value, ts = entry
            if (time.monotonic() - ts) >= self._ttl_seconds:
                del self._data[key]
                logger.debug("[%s] expired key evicted on get", self._name)
                return default
            return value

    def pop(self, key: str, *args: Any) -> Any:
        """Remove and return the value for *key*.

        If *key* is not found or expired, return *default* if given,
        otherwise raise ``KeyError``.
        """
        with self._lock:
            entry = self._data.pop(key, None)
            if entry is None:
                if args:
                    return args[0]
                raise KeyError(key)
            value, ts = entry
            if (time.monotonic() - ts) >= self._ttl_seconds:
                # Already popped, so just return default / raise
                if args:
                    return args[0]
                raise KeyError(key)
            return value

    def values(self) -> list[Any]:
        """Return a list of all non-expired values."""
        with self._lock:
            self._evict_expired()
            return [v for v, _ in self._data.values()]

    def items(self) -> list[tuple[str, Any]]:
        """Return a list of (key, value) pairs for all non-expired entries."""
        with self._lock:
            self._evict_expired()
            return [(k, v) for k, (v, _) in self._data.items()]

    def keys(self) -> list[str]:
        """Return a list of all non-expired keys."""
        with self._lock:
            self._evict_expired()
            return list(self._data.keys())

    def clear(self) -> None:
        """Remove all entries."""
        with self._lock:
            self._data.clear()
            self._expiry_heap.clear()

    def cleanup(self) -> int:
        """Manually remove all expired entries.

        Returns:
            The number of entries removed.
        """
        with self._lock:
            before = len(self._data)
            self._evict_expired()
            removed = before - len(self._data)
            return removed

    def __repr__(self) -> str:
        with self._lock:
            return (
                f"BoundedTTLDict(name={self._name!r}, "
                f"size={len(self._data)}, "
                f"max_size={self._max_size}, "
                f"ttl_seconds={self._ttl_seconds})"
            )


__all__ = ["BoundedTTLDict"]
