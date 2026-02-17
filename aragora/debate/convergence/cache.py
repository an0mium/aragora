"""
Pairwise similarity cache for session-scoped caching.

Optimizes convergence detection by caching pairwise similarity results
within debate sessions. Includes automatic TTL-based eviction and
periodic background cleanup.
"""

from __future__ import annotations

import atexit
import hashlib
import logging
import sys
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# Pairwise Similarity Cache for Session-Scoped Caching
# =============================================================================


@dataclass
class CachedSimilarity:
    """Cached similarity result with metadata."""

    similarity: float
    computed_at: float  # timestamp


class PairwiseSimilarityCache:
    """
    Session-scoped cache for pairwise similarity computations.

    Optimizes the O(n^2) similarity checks in convergence detection by
    caching results within a debate session. Uses content hashing to
    handle text variations efficiently.

    Features:
    - Per-debate-session isolation
    - LRU eviction when cache is full
    - TTL-based expiry for long debates
    - Symmetric key normalization (A,B == B,A)

    Performance impact:
    - Without cache: O(n^2) similarity computations per round
    - With cache: O(1) for repeated pairs (amortized O(n^2) first time)
    - Expected speedup: 10-50x for multi-round debates with stable agents
    """

    def __init__(
        self,
        session_id: str,
        max_size: int = 1024,
        ttl_seconds: float = 600.0,  # 10 minutes default
    ):
        """
        Initialize pairwise similarity cache.

        Args:
            session_id: Unique identifier for this debate session
            max_size: Maximum cache entries (LRU eviction when exceeded)
            ttl_seconds: Time-to-live for cached entries
        """
        self.session_id = session_id
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, CachedSimilarity] = OrderedDict()
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0

    def _hash_text(self, text: str) -> str:
        """Generate hash for text content."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    def _make_key(self, text1: str, text2: str) -> str:
        """Generate symmetric cache key for text pair."""
        hash1 = self._hash_text(text1)
        hash2 = self._hash_text(text2)
        # Normalize order for symmetric lookup
        if hash1 <= hash2:
            return f"{hash1}:{hash2}"
        return f"{hash2}:{hash1}"

    def get(self, text1: str, text2: str) -> float | None:
        """
        Get cached similarity for a text pair.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Cached similarity score or None if not cached/expired
        """
        key = self._make_key(text1, text2)
        now = time.time()

        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None

            entry = self._cache[key]

            # Check TTL expiry
            if now - entry.computed_at > self.ttl_seconds:
                del self._cache[key]
                self._misses += 1
                return None

            # Move to end for LRU
            self._cache.move_to_end(key)
            self._hits += 1
            return entry.similarity

    def put(self, text1: str, text2: str, similarity: float) -> None:
        """
        Store similarity result in cache.

        Args:
            text1: First text
            text2: Second text
            similarity: Computed similarity score
        """
        key = self._make_key(text1, text2)
        now = time.time()

        with self._lock:
            # LRU eviction if at capacity
            while len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)

            self._cache[key] = CachedSimilarity(
                similarity=similarity,
                computed_at=now,
            )

    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    def evict_expired(self) -> int:
        """
        Proactively evict all expired entries from the cache.

        This provides eager TTL-based eviction rather than waiting for
        entries to be accessed via get(). Useful for memory management
        in long-running debates.

        Returns:
            Number of entries evicted
        """
        now = time.time()
        evicted = 0

        with self._lock:
            expired_keys = [
                key
                for key, entry in self._cache.items()
                if now - entry.computed_at > self.ttl_seconds
            ]
            for key in expired_keys:
                del self._cache[key]
                evicted += 1

        if evicted > 0:
            logger.debug(f"Evicted {evicted} expired entries from cache {self.session_id}")

        return evicted

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        now = time.time()
        with self._lock:
            total = self._hits + self._misses
            # Count expired entries (for diagnostics)
            expired_count = sum(
                1 for entry in self._cache.values() if now - entry.computed_at > self.ttl_seconds
            )
            return {
                "session_id": self.session_id,
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": self._hits / total if total > 0 else 0.0,
                "expired_entries": expired_count,
                "ttl_seconds": self.ttl_seconds,
            }


# Global cache manager for similarity caches
_similarity_cache_manager: dict[str, PairwiseSimilarityCache] = {}
_similarity_cache_timestamps: dict[str, float] = {}  # Track when each cache was created/accessed
_similarity_cache_lock = threading.Lock()

# Cache manager limits
DEFAULT_MAX_SIMILARITY_CACHES = 100
MAX_SIMILARITY_CACHES = DEFAULT_MAX_SIMILARITY_CACHES  # Maximum concurrent debate caches
CACHE_MANAGER_TTL_SECONDS = 3600  # 1 hour - cleanup idle caches

# Periodic cleanup configuration
PERIODIC_CLEANUP_INTERVAL_SECONDS = 600  # 10 minutes


def _effective_max_similarity_caches() -> int:
    """
    Resolve the active max-cache limit.

    Tests and callers sometimes patch ``aragora.debate.convergence.MAX_SIMILARITY_CACHES``
    (package re-export). This helper keeps cache-manager behavior aligned with that value.
    """
    module_value = MAX_SIMILARITY_CACHES

    # If tests patch the package re-export, prefer that package override.
    # If not, keep the module-level value (which may itself be patched).
    try:
        pkg = sys.modules.get("aragora.debate.convergence")
        if pkg is not None:
            patched = getattr(pkg, "MAX_SIMILARITY_CACHES", None)
            if (
                isinstance(patched, int)
                and patched > 0
                and patched != DEFAULT_MAX_SIMILARITY_CACHES
            ):
                return patched
    except (AttributeError, TypeError):
        pass
    return module_value


class _PeriodicCacheCleanup:
    """
    Background thread for periodic cleanup of stale similarity caches.

    Runs every PERIODIC_CLEANUP_INTERVAL_SECONDS to remove caches that
    haven't been accessed within CACHE_MANAGER_TTL_SECONDS.

    Thread-safety:
    - Uses daemon thread that terminates when main program exits
    - Uses Event for clean shutdown coordination
    - Uses the global _similarity_cache_lock for cache access
    - Registered with atexit for graceful shutdown
    """

    def __init__(self, interval_seconds: float = PERIODIC_CLEANUP_INTERVAL_SECONDS):
        """
        Initialize periodic cleanup manager.

        Args:
            interval_seconds: Seconds between cleanup runs (default 600 = 10 minutes)
        """
        self._interval = interval_seconds
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._started = False
        self._cleanup_count = 0
        self._entries_evicted_count = 0
        self._last_cleanup_time: float | None = None

    def start(self) -> None:
        """Start the periodic cleanup thread if not already running."""
        with self._lock:
            if self._started and self._thread and self._thread.is_alive():
                return  # Already running

            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._cleanup_loop,
                name="SimilarityCacheCleanup",
                daemon=True,  # Daemon thread - won't prevent program exit
            )
            self._thread.start()
            self._started = True
            logger.debug("Periodic similarity cache cleanup started")

    def stop(self, timeout: float = 5.0) -> None:
        """
        Stop the periodic cleanup thread.

        Args:
            timeout: Maximum seconds to wait for thread to stop
        """
        with self._lock:
            if not self._started:
                return

            self._stop_event.set()
            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=timeout)
            self._started = False
            self._thread = None
            logger.debug("Periodic similarity cache cleanup stopped")

    def _cleanup_loop(self) -> None:
        """Background loop that performs periodic cleanup."""
        while not self._stop_event.is_set():
            # Wait for interval or until stop is signaled
            if self._stop_event.wait(timeout=self._interval):
                break  # Stop event was set

            try:
                # Phase 1: Remove stale session caches (TTL-based at session level)
                cleaned = cleanup_stale_similarity_caches()
                self._cleanup_count += cleaned

                # Phase 2: Evict expired entries within active caches (TTL-based at entry level)
                entries_evicted = evict_expired_cache_entries()
                self._entries_evicted_count += entries_evicted

                self._last_cleanup_time = time.time()

                if cleaned > 0 or entries_evicted > 0:
                    logger.info(
                        f"Periodic cleanup: removed {cleaned} stale caches, "
                        f"evicted {entries_evicted} expired entries"
                    )
            except (RuntimeError, ValueError, TypeError, AttributeError, KeyError, OSError) as e:
                logger.warning(f"Error during periodic cache cleanup: {e}")

    def get_stats(self) -> dict[str, Any]:
        """Get cleanup thread statistics."""
        with self._lock:
            return {
                "running": self._started and self._thread is not None and self._thread.is_alive(),
                "interval_seconds": self._interval,
                "total_caches_cleaned": self._cleanup_count,
                "total_entries_evicted": self._entries_evicted_count,
                "last_cleanup_time": self._last_cleanup_time,
            }

    def is_running(self) -> bool:
        """Check if cleanup thread is running."""
        with self._lock:
            return self._started and self._thread is not None and self._thread.is_alive()


# Global periodic cleanup instance
_periodic_cleanup: _PeriodicCacheCleanup | None = None
_periodic_cleanup_init_lock = threading.Lock()


def _ensure_periodic_cleanup_started() -> None:
    """Ensure the periodic cleanup thread is started (lazy initialization)."""
    global _periodic_cleanup
    with _periodic_cleanup_init_lock:
        if _periodic_cleanup is None:
            _periodic_cleanup = _PeriodicCacheCleanup()
            # Register shutdown handler
            atexit.register(_shutdown_periodic_cleanup)
        if not _periodic_cleanup.is_running():
            _periodic_cleanup.start()


def _shutdown_periodic_cleanup() -> None:
    """Shutdown handler for periodic cleanup (registered with atexit)."""
    global _periodic_cleanup
    if _periodic_cleanup is not None:
        _periodic_cleanup.stop()


def get_periodic_cleanup_stats() -> dict[str, Any]:
    """
    Get statistics about the periodic cleanup thread.

    Returns:
        Dict with running status, interval, total cleaned, total entries evicted, last cleanup time
    """
    global _periodic_cleanup
    if _periodic_cleanup is None:
        return {
            "running": False,
            "interval_seconds": PERIODIC_CLEANUP_INTERVAL_SECONDS,
            "total_caches_cleaned": 0,
            "total_entries_evicted": 0,
            "last_cleanup_time": None,
        }
    return _periodic_cleanup.get_stats()


def stop_periodic_cleanup() -> None:
    """
    Stop the periodic cleanup thread.

    Useful for testing or controlled shutdown scenarios.
    The thread will be restarted automatically on the next cache access.
    """
    global _periodic_cleanup
    if _periodic_cleanup is not None:
        _periodic_cleanup.stop()


def get_pairwise_similarity_cache(
    session_id: str,
    max_size: int = 1024,
    ttl_seconds: float = 600.0,
) -> PairwiseSimilarityCache:
    """
    Get or create a pairwise similarity cache for a debate session.

    This function manages the global cache manager with automatic cleanup:
    - Tracks creation/access timestamps for TTL-based eviction
    - Enforces MAX_SIMILARITY_CACHES limit to prevent unbounded growth
    - Runs periodic cleanup when cache count reaches limit
    - Starts background cleanup thread on first access

    Args:
        session_id: Unique debate session identifier
        max_size: Maximum cache entries per session
        ttl_seconds: Cache TTL in seconds for individual entries

    Returns:
        PairwiseSimilarityCache instance for this session
    """
    # Ensure periodic cleanup is running (lazy start)
    _ensure_periodic_cleanup_started()

    now = time.time()

    with _similarity_cache_lock:
        # Check if cache already exists - update timestamp and return
        if session_id in _similarity_cache_manager:
            _similarity_cache_timestamps[session_id] = now
            return _similarity_cache_manager[session_id]

        # At capacity - run cleanup to remove stale caches
        max_caches = _effective_max_similarity_caches()
        if len(_similarity_cache_manager) >= max_caches:
            # Release lock temporarily for cleanup (avoid holding lock too long)
            # We'll re-acquire and re-check after cleanup
            pass

    # Run cleanup outside the lock to avoid blocking other threads
    if len(_similarity_cache_manager) >= _effective_max_similarity_caches():
        cleanup_stale_similarity_caches()

    with _similarity_cache_lock:
        # Re-check if session was created while we were cleaning up
        if session_id in _similarity_cache_manager:
            _similarity_cache_timestamps[session_id] = now
            return _similarity_cache_manager[session_id]

        # Still at limit after cleanup? Remove oldest cache
        max_caches = _effective_max_similarity_caches()
        if len(_similarity_cache_manager) >= max_caches:
            if _similarity_cache_timestamps:
                oldest_session = min(
                    _similarity_cache_timestamps.items(),
                    key=lambda x: x[1],
                )[0]
                if oldest_session in _similarity_cache_manager:
                    _similarity_cache_manager[oldest_session].clear()
                    del _similarity_cache_manager[oldest_session]
                if oldest_session in _similarity_cache_timestamps:
                    del _similarity_cache_timestamps[oldest_session]
                logger.debug(f"Evicted oldest similarity cache {oldest_session} to make room")

        # Create new cache
        _similarity_cache_manager[session_id] = PairwiseSimilarityCache(
            session_id=session_id,
            max_size=max_size,
            ttl_seconds=ttl_seconds,
        )
        _similarity_cache_timestamps[session_id] = now

        return _similarity_cache_manager[session_id]


def cleanup_similarity_cache(session_id: str) -> None:
    """
    Cleanup similarity cache for a completed debate.

    Args:
        session_id: Debate session ID to cleanup
    """
    with _similarity_cache_lock:
        if session_id in _similarity_cache_manager:
            _similarity_cache_manager[session_id].clear()
            del _similarity_cache_manager[session_id]
        if session_id in _similarity_cache_timestamps:
            del _similarity_cache_timestamps[session_id]
        logger.debug(f"Cleaned up similarity cache for session {session_id}")


def cleanup_stale_similarity_caches(
    max_age_seconds: float = CACHE_MANAGER_TTL_SECONDS,
) -> int:
    """
    Remove similarity caches that haven't been used recently.

    This function provides automatic TTL-based eviction for the global
    cache manager, preventing unbounded memory growth from abandoned
    debate sessions.

    Args:
        max_age_seconds: Maximum age in seconds before a cache is considered stale.
            Defaults to CACHE_MANAGER_TTL_SECONDS (1 hour).

    Returns:
        Number of caches cleaned up.
    """
    now = time.time()
    cleaned = 0

    with _similarity_cache_lock:
        stale_sessions = [
            session_id
            for session_id, timestamp in _similarity_cache_timestamps.items()
            if now - timestamp > max_age_seconds
        ]
        for session_id in stale_sessions:
            if session_id in _similarity_cache_manager:
                _similarity_cache_manager[session_id].clear()
                del _similarity_cache_manager[session_id]
            if session_id in _similarity_cache_timestamps:
                del _similarity_cache_timestamps[session_id]
            cleaned += 1

    if cleaned > 0:
        logger.debug(f"Cleaned up {cleaned} stale similarity caches")

    return cleaned


def evict_expired_cache_entries() -> int:
    """
    Evict expired entries from all active similarity caches.

    This performs TTL-based eviction at the entry level within each
    active PairwiseSimilarityCache. Unlike the lazy eviction in get(),
    this proactively removes all expired entries.

    Thread-safety: Acquires _similarity_cache_lock to get cache list,
    then calls evict_expired() on each cache (which has its own lock).

    Returns:
        Total number of entries evicted across all caches
    """
    total_evicted = 0

    # Get list of caches while holding the lock
    with _similarity_cache_lock:
        caches = list(_similarity_cache_manager.values())

    # Evict from each cache (each has its own internal lock)
    for cache in caches:
        try:
            evicted = cache.evict_expired()
            total_evicted += evicted
        except (RuntimeError, ValueError, TypeError, AttributeError, KeyError, OSError) as e:
            logger.warning(f"Error evicting expired entries from cache {cache.session_id}: {e}")

    return total_evicted


def cleanup_stale_caches(max_age_seconds: float | None = None) -> dict[str, Any]:
    """
    Public function to cleanup stale caches.

    This is the externally-callable cleanup function that can be invoked
    by external processes, monitoring systems, or scheduled tasks.

    Performs two-phase cleanup:
    1. Remove entire caches that haven't been accessed recently (session-level TTL)
    2. Evict expired entries within remaining active caches (entry-level TTL)

    Args:
        max_age_seconds: Maximum age for session caches. If None, uses
            CACHE_MANAGER_TTL_SECONDS (1 hour).

    Returns:
        Dict with cleanup statistics:
        - cleaned_count: Number of caches removed
        - entries_evicted: Number of expired entries evicted from active caches
        - remaining_count: Number of caches still active
        - cleanup_time: Timestamp of this cleanup
        - periodic_cleanup_running: Whether background cleanup is active
    """
    if max_age_seconds is None:
        max_age_seconds = CACHE_MANAGER_TTL_SECONDS

    # Phase 1: Remove stale session caches
    cleaned = cleanup_stale_similarity_caches(max_age_seconds)

    # Phase 2: Evict expired entries within active caches
    entries_evicted = evict_expired_cache_entries()

    with _similarity_cache_lock:
        remaining = len(_similarity_cache_manager)

    return {
        "cleaned_count": cleaned,
        "entries_evicted": entries_evicted,
        "remaining_count": remaining,
        "cleanup_time": time.time(),
        "periodic_cleanup_running": _periodic_cleanup.is_running() if _periodic_cleanup else False,
    }


def get_cache_manager_stats() -> dict[str, Any]:
    """
    Get statistics about the similarity cache manager.

    Returns:
        Dict with cache manager statistics:
        - active_caches: Number of active debate caches
        - max_caches: Maximum allowed caches
        - cache_ttl_seconds: TTL for stale cache eviction
        - cleanup_interval_seconds: Periodic cleanup interval
        - periodic_cleanup: Stats from periodic cleanup thread
        - caches: Dict of session_id -> {age_seconds, cache_stats}
    """
    now = time.time()

    with _similarity_cache_lock:
        cache_details = {}
        for session_id, cache in _similarity_cache_manager.items():
            timestamp = _similarity_cache_timestamps.get(session_id, now)
            cache_details[session_id] = {
                "age_seconds": now - timestamp,
                "stats": cache.get_stats(),
            }

        return {
            "active_caches": len(_similarity_cache_manager),
            "max_caches": _effective_max_similarity_caches(),
            "cache_ttl_seconds": CACHE_MANAGER_TTL_SECONDS,
            "cleanup_interval_seconds": PERIODIC_CLEANUP_INTERVAL_SECONDS,
            "periodic_cleanup": get_periodic_cleanup_stats(),
            "caches": cache_details,
        }
