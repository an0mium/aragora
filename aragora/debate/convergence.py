"""
Semantic convergence detection for multi-agent debates.

Detects when agents' positions have converged, allowing early termination
of debates when further rounds would provide diminishing returns.

Uses a 3-tier fallback for similarity computation:
1. SentenceTransformer (best accuracy, requires sentence-transformers)
2. TF-IDF (good accuracy, requires scikit-learn)
3. Jaccard (always available, zero dependencies)

Inspired by ai-counsel's convergence detection system.

Module Structure
----------------
This module has been split into submodules for maintainability:

- `aragora.debate.cache.embeddings_lru` - EmbeddingCache for text embeddings
- `aragora.debate.similarity.backends` - Similarity computation backends

This file contains:
- ConvergenceResult - Result of convergence check
- Advanced convergence metrics (G3)
- AdvancedConvergenceAnalyzer - Multi-metric analysis
- ConvergenceDetector - Main convergence detection

Performance optimizations:
- LRU cache for pairwise similarity results (256 pairs per backend)
- Session-scoped similarity cache for debate-specific computations
- Batch similarity computation for vectorizable backends
- Early termination in within-round convergence checks
"""

from __future__ import annotations

import atexit
import hashlib
import logging
import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
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
MAX_SIMILARITY_CACHES = 100  # Maximum concurrent debate caches
CACHE_MANAGER_TTL_SECONDS = 3600  # 1 hour - cleanup idle caches

# Periodic cleanup configuration
PERIODIC_CLEANUP_INTERVAL_SECONDS = 600  # 10 minutes


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
            except Exception as e:
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
        if len(_similarity_cache_manager) >= MAX_SIMILARITY_CACHES:
            # Release lock temporarily for cleanup (avoid holding lock too long)
            # We'll re-acquire and re-check after cleanup
            pass

    # Run cleanup outside the lock to avoid blocking other threads
    if len(_similarity_cache_manager) >= MAX_SIMILARITY_CACHES:
        cleanup_stale_similarity_caches()

    with _similarity_cache_lock:
        # Re-check if session was created while we were cleaning up
        if session_id in _similarity_cache_manager:
            _similarity_cache_timestamps[session_id] = now
            return _similarity_cache_manager[session_id]

        # Still at limit after cleanup? Remove oldest cache
        if len(_similarity_cache_manager) >= MAX_SIMILARITY_CACHES:
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
        except Exception as e:
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
            "max_caches": MAX_SIMILARITY_CACHES,
            "cache_ttl_seconds": CACHE_MANAGER_TTL_SECONDS,
            "cleanup_interval_seconds": PERIODIC_CLEANUP_INTERVAL_SECONDS,
            "periodic_cleanup": get_periodic_cleanup_stats(),
            "caches": cache_details,
        }


# Re-export cache utilities
from aragora.debate.cache.embeddings_lru import (
    EmbeddingCache,
    cleanup_embedding_cache,
    get_embedding_cache,
    get_scoped_embedding_cache,
    reset_embedding_cache,
)

# Re-export similarity backends
from aragora.debate.similarity.backends import (
    _ENV_CONVERGENCE_BACKEND,
    JaccardBackend,
    SentenceTransformerBackend,
    SimilarityBackend,
    TFIDFBackend,
    _normalize_backend_name,
    get_similarity_backend,
)

# =============================================================================
# Convergence Result
# =============================================================================


@dataclass
class ConvergenceResult:
    """Result of convergence detection check."""

    converged: bool
    status: str  # "converged", "diverging", "refining"
    min_similarity: float
    avg_similarity: float
    per_agent_similarity: dict[str, float] = field(default_factory=dict)
    consecutive_stable_rounds: int = 0


# =============================================================================
# Advanced Convergence Metrics (G3)
# =============================================================================


@dataclass
class ArgumentDiversityMetric:
    """
    Measures diversity of arguments across agents.

    High diversity = agents covering different points (good for exploration)
    Low diversity = agents focusing on same points (may indicate convergence)
    """

    unique_arguments: int
    total_arguments: int
    diversity_score: float  # 0-1, higher = more diverse

    @property
    def is_converging(self) -> bool:
        """Arguments becoming less diverse suggests convergence."""
        return self.diversity_score < 0.3


@dataclass
class EvidenceConvergenceMetric:
    """
    Measures overlap in cited evidence/sources.

    High overlap = agents citing same sources (strong agreement)
    Low overlap = agents using different evidence (disagreement or complementary)
    """

    shared_citations: int
    total_citations: int
    overlap_score: float  # 0-1, higher = more overlap

    @property
    def is_converging(self) -> bool:
        """High citation overlap suggests convergence."""
        return self.overlap_score > 0.6


@dataclass
class StanceVolatilityMetric:
    """
    Measures how often agents change their positions.

    High volatility = agents frequently changing stances (unstable)
    Low volatility = agents maintaining consistent positions (stable)
    """

    stance_changes: int
    total_responses: int
    volatility_score: float  # 0-1, higher = more volatile

    @property
    def is_stable(self) -> bool:
        """Low volatility indicates stable positions."""
        return self.volatility_score < 0.2


@dataclass
class AdvancedConvergenceMetrics:
    """
    Comprehensive convergence metrics for debate analysis.

    Combines multiple signals to provide a nuanced view of
    debate convergence beyond simple text similarity.
    """

    # Core similarity (from ConvergenceDetector)
    semantic_similarity: float

    # Advanced metrics
    argument_diversity: ArgumentDiversityMetric | None = None
    evidence_convergence: EvidenceConvergenceMetric | None = None
    stance_volatility: StanceVolatilityMetric | None = None

    # Aggregate score
    overall_convergence: float = 0.0  # 0-1, higher = more converged

    # Domain context
    domain: str = "general"

    def compute_overall_score(self) -> float:
        """Compute weighted overall convergence score."""
        weights = {
            "semantic": 0.4,
            "diversity": 0.2,
            "evidence": 0.2,
            "stability": 0.2,
        }

        score = self.semantic_similarity * weights["semantic"]

        if self.argument_diversity:
            # Lower diversity = higher convergence
            score += (1 - self.argument_diversity.diversity_score) * weights["diversity"]

        if self.evidence_convergence:
            score += self.evidence_convergence.overlap_score * weights["evidence"]

        if self.stance_volatility:
            # Lower volatility = higher convergence
            score += (1 - self.stance_volatility.volatility_score) * weights["stability"]

        self.overall_convergence = min(1.0, max(0.0, score))
        return self.overall_convergence

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        result = {
            "semantic_similarity": self.semantic_similarity,
            "overall_convergence": self.overall_convergence,
            "domain": self.domain,
        }

        if self.argument_diversity:
            result["argument_diversity"] = {
                "unique_arguments": self.argument_diversity.unique_arguments,
                "total_arguments": self.argument_diversity.total_arguments,
                "diversity_score": self.argument_diversity.diversity_score,
            }

        if self.evidence_convergence:
            result["evidence_convergence"] = {
                "shared_citations": self.evidence_convergence.shared_citations,
                "total_citations": self.evidence_convergence.total_citations,
                "overlap_score": self.evidence_convergence.overlap_score,
            }

        if self.stance_volatility:
            result["stance_volatility"] = {
                "stance_changes": self.stance_volatility.stance_changes,
                "total_responses": self.stance_volatility.total_responses,
                "volatility_score": self.stance_volatility.volatility_score,
            }

        return result


# =============================================================================
# Advanced Convergence Analyzer
# =============================================================================


class AdvancedConvergenceAnalyzer:
    """
    Analyzes debate convergence using multiple metrics.

    Provides a more nuanced view than simple text similarity by
    considering argument diversity, evidence overlap, and stance stability.

    Performance optimizations:
    - Session-scoped pairwise similarity cache
    - Batch computation for vectorizable backends
    - Early termination for non-converged states
    """

    def __init__(
        self,
        similarity_backend: SimilarityBackend | None = None,
        debate_id: str | None = None,
        enable_cache: bool = True,
    ):
        """
        Initialize analyzer.

        Args:
            similarity_backend: Backend for text similarity (auto-selects if None)
            debate_id: Unique debate ID for session-scoped caching
            enable_cache: Whether to enable pairwise similarity caching
        """
        if similarity_backend is None:
            # Use factory function for consistent backend selection
            self.backend: SimilarityBackend = get_similarity_backend("auto")
        else:
            self.backend = similarity_backend

        # Session-scoped pairwise similarity cache
        self._debate_id = debate_id
        self._enable_cache = enable_cache and debate_id is not None
        self._similarity_cache: PairwiseSimilarityCache | None = None

        if self._enable_cache and debate_id:
            self._similarity_cache = get_pairwise_similarity_cache(debate_id)
            logger.debug(f"AdvancedConvergenceAnalyzer caching enabled: debate={debate_id}")

    def _compute_similarity_cached(self, text1: str, text2: str) -> float:
        """
        Compute similarity with session-scoped caching.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score (0.0 to 1.0)
        """
        # Check cache first
        if self._similarity_cache:
            cached = self._similarity_cache.get(text1, text2)
            if cached is not None:
                return cached

        # Compute similarity
        similarity = self.backend.compute_similarity(text1, text2)

        # Cache the result
        if self._similarity_cache:
            self._similarity_cache.put(text1, text2, similarity)

        return similarity

    def cleanup(self) -> None:
        """Cleanup resources when debate ends."""
        if self._debate_id:
            cleanup_similarity_cache(self._debate_id)
            logger.debug(f"AdvancedConvergenceAnalyzer cleanup: debate={self._debate_id}")

    def get_cache_stats(self) -> dict | None:
        """Get cache statistics."""
        if self._similarity_cache:
            return self._similarity_cache.get_stats()
        return None

    def extract_arguments(self, text: str) -> list[str]:
        """
        Extract distinct arguments/claims from text.

        Simple heuristic: split by sentence and filter.
        """
        import re

        # Split into sentences
        sentences = re.split(r"[.!?]+", text)

        # Filter to substantive sentences (> 5 words)
        arguments = []
        for s in sentences:
            s = s.strip()
            if len(s.split()) > 5:
                arguments.append(s)

        return arguments

    def extract_citations(self, text: str) -> set[str]:
        """
        Extract citations/sources from text.

        Looks for URLs, academic-style citations, and quoted sources.
        """
        import re

        citations = set()

        # URLs
        urls = re.findall(r'https?://[^\s<>"]+', text)
        citations.update(urls)

        # Academic citations like (Author, 2024) or [1]
        academic = re.findall(r"\([A-Z][a-z]+(?:\s+et\s+al\.?)?,?\s*\d{4}\)", text)
        citations.update(academic)

        # Numbered citations [1], [2], etc.
        numbered = re.findall(r"\[\d+\]", text)
        citations.update(numbered)

        # Quoted sources "According to X"
        quoted = re.findall(
            r"(?:according to|per|from)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", text, re.I
        )
        citations.update(quoted)

        return citations

    def detect_stance(self, text: str) -> str:
        """
        Detect the stance/position in text.

        Returns: "support", "oppose", "neutral", or "mixed"
        """
        import re

        text_lower = text.lower()

        # Strong support indicators
        support_patterns = (
            r"\b(agree|support|favor|endorse|recommend|should|must|definitely|certainly)\b"
        )
        support_count = len(re.findall(support_patterns, text_lower))

        # Strong oppose indicators
        oppose_patterns = (
            r"\b(disagree|oppose|against|reject|shouldn\'t|must not|definitely not|certainly not)\b"
        )
        oppose_count = len(re.findall(oppose_patterns, text_lower))

        # Neutral indicators
        neutral_patterns = r"\b(depends|unclear|both|however|on the other hand|alternatively)\b"
        neutral_count = len(re.findall(neutral_patterns, text_lower))

        # Determine stance
        if support_count > oppose_count and support_count > neutral_count:
            return "support"
        elif oppose_count > support_count and oppose_count > neutral_count:
            return "oppose"
        elif neutral_count > 0 and support_count > 0 and oppose_count > 0:
            return "mixed"
        else:
            return "neutral"

    def compute_argument_diversity(
        self,
        agent_responses: dict[str, str],
        use_optimized: bool = True,
    ) -> ArgumentDiversityMetric:
        """
        Compute argument diversity across agents.

        High diversity = agents making different points.

        Args:
            agent_responses: Dict mapping agent names to their response texts
            use_optimized: Use O(n log n) ANN-based algorithm when possible
                          (falls back to O(n²) if embeddings unavailable)

        Returns:
            ArgumentDiversityMetric with unique/total counts and diversity score
        """
        all_arguments: list[str] = []
        for text in agent_responses.values():
            all_arguments.extend(self.extract_arguments(text))

        if not all_arguments:
            return ArgumentDiversityMetric(
                unique_arguments=0,
                total_arguments=0,
                diversity_score=0.0,
            )

        # Try optimized path using vectorized similarity computation
        if use_optimized and len(all_arguments) >= 5:
            try:
                unique_count, total, diversity_score = self._compute_diversity_optimized(
                    all_arguments
                )
                return ArgumentDiversityMetric(
                    unique_arguments=unique_count,
                    total_arguments=total,
                    diversity_score=diversity_score,
                )
            except Exception as e:
                logger.debug(f"Optimized diversity computation failed, using fallback: {e}")

        # Fallback: O(n²) pairwise comparison with caching
        # Arguments with < 0.7 similarity to all others are "unique"
        # Uses session-scoped cache to avoid redundant computations
        unique_count = 0
        for i, arg in enumerate(all_arguments):
            is_unique = True
            for j, other in enumerate(all_arguments):
                if i != j:
                    # Use cached similarity to avoid redundant computations
                    sim = self._compute_similarity_cached(arg, other)
                    if sim > 0.7:
                        is_unique = False
                        break
            if is_unique:
                unique_count += 1

        diversity_score = unique_count / len(all_arguments) if all_arguments else 0.0

        return ArgumentDiversityMetric(
            unique_arguments=unique_count,
            total_arguments=len(all_arguments),
            diversity_score=diversity_score,
        )

    def _compute_diversity_optimized(
        self, arguments: list[str], threshold: float = 0.7
    ) -> tuple[int, int, float]:
        """Compute diversity using optimized vectorized operations.

        Uses SentenceTransformer embeddings + vectorized numpy/FAISS operations
        for O(n log n) complexity instead of O(n²).

        Args:
            arguments: List of argument texts
            threshold: Similarity threshold for considering arguments as duplicates

        Returns:
            Tuple of (unique_count, total_count, diversity_score)
        """
        import numpy as np

        from aragora.debate.similarity.ann import count_unique_fast

        # Get embeddings from backend if it supports it
        if hasattr(self.backend, "_get_embedding"):
            # SentenceTransformerBackend has _get_embedding
            embeddings = []
            for arg in arguments:
                emb = self.backend._get_embedding(arg)
                embeddings.append(emb)
            embeddings_array = np.vstack(embeddings)
        elif hasattr(self.backend, "vectorizer"):
            # TFIDFBackend has vectorizer
            from scipy.sparse import issparse

            tfidf_matrix = self.backend.vectorizer.fit_transform(arguments)
            if issparse(tfidf_matrix):
                embeddings_array = tfidf_matrix.toarray().astype(np.float32)
            else:
                embeddings_array = tfidf_matrix.astype(np.float32)
        else:
            # JaccardBackend or unknown - can't use optimized path
            raise ValueError("Backend doesn't support embedding extraction")

        return count_unique_fast(embeddings_array, threshold=threshold)

    def compute_evidence_convergence(
        self,
        agent_responses: dict[str, str],
    ) -> EvidenceConvergenceMetric:
        """
        Compute evidence/citation overlap across agents.

        High overlap = agents citing same sources.
        """
        all_citations: list[set[str]] = []
        for text in agent_responses.values():
            all_citations.append(self.extract_citations(text))

        # Flatten for total count
        all_unique = set().union(*all_citations) if all_citations else set()
        total = len(all_unique)

        if total == 0 or len(all_citations) < 2:
            return EvidenceConvergenceMetric(
                shared_citations=0,
                total_citations=0,
                overlap_score=0.0,
            )

        # Find citations shared by at least 2 agents
        shared = set()
        for citation in all_unique:
            count = sum(1 for agent_cites in all_citations if citation in agent_cites)
            if count >= 2:
                shared.add(citation)

        overlap_score = len(shared) / total if total > 0 else 0.0

        return EvidenceConvergenceMetric(
            shared_citations=len(shared),
            total_citations=total,
            overlap_score=overlap_score,
        )

    def compute_stance_volatility(
        self,
        response_history: list[dict[str, str]],
    ) -> StanceVolatilityMetric:
        """
        Compute stance volatility across rounds.

        Args:
            response_history: List of {agent: response} dicts per round

        Returns:
            StanceVolatilityMetric
        """
        if len(response_history) < 2:
            return StanceVolatilityMetric(
                stance_changes=0,
                total_responses=0,
                volatility_score=0.0,
            )

        # Track stance per agent per round
        agent_stances: dict[str, list[str]] = {}
        for round_responses in response_history:
            for agent, text in round_responses.items():
                if agent not in agent_stances:
                    agent_stances[agent] = []
                agent_stances[agent].append(self.detect_stance(text))

        # Count stance changes
        total_changes = 0
        total_responses = 0
        for agent, stances in agent_stances.items():
            total_responses += len(stances)
            for i in range(1, len(stances)):
                if stances[i] != stances[i - 1]:
                    total_changes += 1

        volatility_score = total_changes / max(1, total_responses - len(agent_stances))

        return StanceVolatilityMetric(
            stance_changes=total_changes,
            total_responses=total_responses,
            volatility_score=min(1.0, volatility_score),
        )

    def analyze(
        self,
        current_responses: dict[str, str],
        previous_responses: dict[str, str] | None = None,
        response_history: list[dict[str, str] | None] = None,
        domain: str = "general",
    ) -> AdvancedConvergenceMetrics:
        """
        Perform comprehensive convergence analysis.

        Args:
            current_responses: {agent: response} for current round
            previous_responses: {agent: response} for previous round (optional)
            response_history: Full history of responses (optional)
            domain: Debate domain for context

        Returns:
            AdvancedConvergenceMetrics with all computed metrics
        """
        # Compute semantic similarity with caching
        if previous_responses:
            common_agents = set(current_responses.keys()) & set(previous_responses.keys())
            if common_agents:
                similarities = []
                for agent in common_agents:
                    # Use cached similarity to avoid redundant computations
                    sim = self._compute_similarity_cached(
                        current_responses[agent],
                        previous_responses[agent],
                    )
                    similarities.append(sim)
                semantic_sim = sum(similarities) / len(similarities)
            else:
                semantic_sim = 0.0
        else:
            semantic_sim = 0.0

        # Compute advanced metrics
        arg_diversity = self.compute_argument_diversity(current_responses)
        evidence_conv = self.compute_evidence_convergence(current_responses)

        stance_vol = None
        if response_history:
            stance_vol = self.compute_stance_volatility(response_history)

        # Build result
        metrics = AdvancedConvergenceMetrics(
            semantic_similarity=semantic_sim,
            argument_diversity=arg_diversity,
            evidence_convergence=evidence_conv,
            stance_volatility=stance_vol,
            domain=domain,
        )

        # Compute overall score
        metrics.compute_overall_score()

        return metrics


# =============================================================================
# Convergence Detector
# =============================================================================


class ConvergenceDetector:
    """
    Detects when debate has converged semantically.

    Uses semantic similarity between consecutive rounds to determine
    if agents have reached consensus or are still refining positions.

    Thresholds:
        - converged: ≥85% similarity (agents agree)
        - refining: 40-85% similarity (still improving)
        - diverging: <40% similarity (positions splitting)
    """

    def __init__(
        self,
        convergence_threshold: float = 0.85,
        divergence_threshold: float = 0.40,
        min_rounds_before_check: int = 1,
        consecutive_rounds_needed: int = 1,
        debate_id: str | None = None,
    ):
        """
        Initialize convergence detector.

        Args:
            convergence_threshold: Similarity threshold for convergence (default 0.85)
            divergence_threshold: Below this is diverging (default 0.40)
            min_rounds_before_check: Minimum rounds before checking (default 1)
            consecutive_rounds_needed: Stable rounds needed for convergence (default 1)
            debate_id: Debate ID for scoped caching (prevents cross-debate contamination)
        """
        self.convergence_threshold = convergence_threshold
        self.divergence_threshold = divergence_threshold
        self.min_rounds_before_check = min_rounds_before_check
        self.consecutive_rounds_needed = consecutive_rounds_needed
        self.consecutive_stable_count = 0
        self.debate_id = debate_id
        self.backend = self._select_backend()

        logger.info(f"ConvergenceDetector initialized with {self.backend.__class__.__name__}")

    def _select_backend(self) -> SimilarityBackend:
        """
        Select best available similarity backend using SimilarityFactory.

        Uses the unified SimilarityFactory for backend selection, which:
        - Respects ARAGORA_SIMILARITY_BACKEND environment variable
        - Auto-selects best available backend based on input size
        - Handles debate_id for scoped caching
        """
        from aragora.debate.similarity.factory import get_backend

        # Check for legacy env override
        env_override = _normalize_backend_name(os.getenv(_ENV_CONVERGENCE_BACKEND, ""))
        if env_override:
            try:
                backend = get_similarity_backend(env_override, debate_id=self.debate_id)
                logger.info(f"Using {env_override} backend via {_ENV_CONVERGENCE_BACKEND}")
                return backend
            except (ImportError, RuntimeError, OSError) as e:
                logger.warning(
                    f"{_ENV_CONVERGENCE_BACKEND}={env_override} failed: {e}. Falling back to factory."
                )
            except Exception as e:
                logger.exception(
                    f"{_ENV_CONVERGENCE_BACKEND}={env_override} unexpected error: {e}. Falling back to factory."
                )

        # Use SimilarityFactory for unified backend selection
        try:
            backend = get_backend(
                preferred="auto",
                input_size=10,  # Default for typical debate sizes
                debate_id=self.debate_id,
            )
            logger.info(f"Using {backend.__class__.__name__} via SimilarityFactory")
            return backend
        except Exception as e:
            logger.warning(f"SimilarityFactory failed: {e}. Using JaccardBackend fallback.")
            return JaccardBackend()

    def check_convergence(
        self,
        current_responses: dict[str, str],
        previous_responses: dict[str, str],
        round_number: int,
    ) -> ConvergenceResult | None:
        """
        Check if debate has converged.

        Args:
            current_responses: Agent name -> response text for current round
            previous_responses: Agent name -> response text for previous round
            round_number: Current round number (1-indexed)

        Returns:
            ConvergenceResult or None if too early to check
        """
        # Don't check before minimum rounds
        if round_number <= self.min_rounds_before_check:
            return None

        # Match agents between rounds
        common_agents = set(current_responses.keys()) & set(previous_responses.keys())
        if not common_agents:
            logger.warning("No matching agents between rounds")
            return None

        # Compute similarity for each agent
        per_agent = {}
        agent_list = list(common_agents)

        # Use batch method if available (SentenceTransformerBackend)
        if hasattr(self.backend, "compute_pairwise_similarities"):
            texts_current = [current_responses[a] for a in agent_list]
            texts_previous = [previous_responses[a] for a in agent_list]
            similarities = self.backend.compute_pairwise_similarities(texts_current, texts_previous)
            per_agent = dict(zip(agent_list, similarities))
        else:
            # Fallback to individual comparisons
            for agent in agent_list:
                similarity = self.backend.compute_similarity(
                    current_responses[agent], previous_responses[agent]
                )
                per_agent[agent] = similarity

        # Compute aggregate metrics
        similarities = list(per_agent.values())
        min_similarity = min(similarities)
        avg_similarity = sum(similarities) / len(similarities)

        # Determine status
        if min_similarity >= self.convergence_threshold:
            self.consecutive_stable_count += 1
            if self.consecutive_stable_count >= self.consecutive_rounds_needed:
                status = "converged"
                converged = True
            else:
                status = "refining"
                converged = False
        elif min_similarity < self.divergence_threshold:
            status = "diverging"
            converged = False
            self.consecutive_stable_count = 0
        else:
            status = "refining"
            converged = False
            self.consecutive_stable_count = 0

        return ConvergenceResult(
            converged=converged,
            status=status,
            min_similarity=min_similarity,
            avg_similarity=avg_similarity,
            per_agent_similarity=per_agent,
            consecutive_stable_rounds=self.consecutive_stable_count,
        )

    def reset(self) -> None:
        """Reset the consecutive stable count."""
        self.consecutive_stable_count = 0

    def check_within_round_convergence(
        self,
        responses: dict[str, str],
        threshold: float | None = None,
    ) -> tuple[bool, float, float]:
        """
        Check if all agents' responses within a single round have converged.

        Uses ANN-optimized vectorized operations with early termination for O(n log n)
        complexity instead of O(n²) pairwise comparison.

        This is useful for detecting when agents agree with each other within a round,
        which can indicate premature consensus or echo chamber effects.

        Args:
            responses: Agent name -> response text for current round
            threshold: Convergence threshold (defaults to self.convergence_threshold)

        Returns:
            Tuple of (converged: bool, min_similarity: float, avg_similarity: float)
        """
        import numpy as np

        from aragora.debate.similarity.ann import (
            compute_batch_similarity_fast,
            find_convergence_threshold,
        )

        if threshold is None:
            threshold = self.convergence_threshold

        texts = list(responses.values())
        if len(texts) < 2:
            return True, 1.0, 1.0

        # Get embeddings using backend
        embeddings = None
        if hasattr(self.backend, "_get_embedding"):
            # SentenceTransformerBackend
            embeddings_list = [self.backend._get_embedding(t) for t in texts]
            embeddings = np.vstack(embeddings_list).astype(np.float32)
        elif hasattr(self.backend, "vectorizer"):
            # TFIDFBackend
            from scipy.sparse import issparse

            tfidf_matrix = self.backend.vectorizer.fit_transform(texts)
            if issparse(tfidf_matrix):
                embeddings = tfidf_matrix.toarray().astype(np.float32)
            else:
                embeddings = np.array(tfidf_matrix).astype(np.float32)

        if embeddings is not None:
            # Use optimized ANN functions with early termination
            converged, min_sim = find_convergence_threshold(embeddings, threshold=threshold)
            avg_sim = compute_batch_similarity_fast(embeddings)
            return converged, min_sim, avg_sim

        # Fallback to individual comparisons for JaccardBackend
        similarities = []
        for i, t1 in enumerate(texts):
            for t2 in texts[i + 1 :]:
                sim = self.backend.compute_similarity(t1, t2)
                similarities.append(sim)
                # Early termination
                if sim < threshold:
                    return False, sim, sum(similarities) / len(similarities)

        min_sim = min(similarities) if similarities else 1.0
        avg_sim = sum(similarities) / len(similarities) if similarities else 1.0
        return min_sim >= threshold, min_sim, avg_sim

    def check_convergence_fast(
        self,
        current_responses: dict[str, str],
        previous_responses: dict[str, str],
        round_number: int,
    ) -> ConvergenceResult | None:
        """
        Fast convergence check with ANN optimizations and early termination.

        Same interface as check_convergence but uses vectorized operations
        for better performance with many agents.

        Args:
            current_responses: Agent name -> response text for current round
            previous_responses: Agent name -> response text for previous round
            round_number: Current round number (1-indexed)

        Returns:
            ConvergenceResult or None if too early to check
        """
        import numpy as np

        # Don't check before minimum rounds
        if round_number <= self.min_rounds_before_check:
            return None

        # Match agents between rounds
        common_agents = set(current_responses.keys()) & set(previous_responses.keys())
        if not common_agents:
            logger.warning("No matching agents between rounds")
            return None

        agent_list = list(common_agents)

        # Try optimized path with embeddings
        embeddings_curr = None
        embeddings_prev = None

        if hasattr(self.backend, "_get_embedding"):
            embeddings_curr = np.vstack(
                [self.backend._get_embedding(current_responses[a]) for a in agent_list]
            ).astype(np.float32)
            embeddings_prev = np.vstack(
                [self.backend._get_embedding(previous_responses[a]) for a in agent_list]
            ).astype(np.float32)

        if embeddings_curr is not None and embeddings_prev is not None:
            # Compute pairwise similarities between current and previous
            # Normalize embeddings
            norms_curr = np.linalg.norm(embeddings_curr, axis=1, keepdims=True)
            norms_prev = np.linalg.norm(embeddings_prev, axis=1, keepdims=True)
            norms_curr = np.where(norms_curr == 0, 1, norms_curr)
            norms_prev = np.where(norms_prev == 0, 1, norms_prev)
            norm_curr = embeddings_curr / norms_curr
            norm_prev = embeddings_prev / norms_prev

            # Diagonal of matrix product gives per-agent similarity
            per_agent_sims = np.sum(norm_curr * norm_prev, axis=1)
            per_agent = dict(zip(agent_list, per_agent_sims.tolist()))

            min_similarity = float(np.min(per_agent_sims))
            avg_similarity = float(np.mean(per_agent_sims))
        else:
            # Fallback to standard computation
            return self.check_convergence(current_responses, previous_responses, round_number)

        # Determine status
        if min_similarity >= self.convergence_threshold:
            self.consecutive_stable_count += 1
            if self.consecutive_stable_count >= self.consecutive_rounds_needed:
                status = "converged"
                converged = True
            else:
                status = "refining"
                converged = False
        elif min_similarity < self.divergence_threshold:
            status = "diverging"
            converged = False
            self.consecutive_stable_count = 0
        else:
            status = "refining"
            converged = False
            self.consecutive_stable_count = 0

        return ConvergenceResult(
            converged=converged,
            status=status,
            min_similarity=min_similarity,
            avg_similarity=avg_similarity,
            per_agent_similarity=per_agent,
            consecutive_stable_rounds=self.consecutive_stable_count,
        )

    def cleanup(self) -> None:
        """Cleanup resources when debate session ends.

        Should be called when the debate completes to free memory.
        Cleans up embedding caches associated with this debate.
        """
        if self.debate_id:
            cleanup_embedding_cache(self.debate_id)
            cleanup_similarity_cache(self.debate_id)
            logger.debug(f"ConvergenceDetector cleanup complete: debate={self.debate_id}")


__all__ = [
    # Cache (re-exported)
    "EmbeddingCache",
    "cleanup_embedding_cache",
    "get_embedding_cache",
    "get_scoped_embedding_cache",
    "reset_embedding_cache",
    # Pairwise similarity cache
    "PairwiseSimilarityCache",
    "get_pairwise_similarity_cache",
    "cleanup_similarity_cache",
    "cleanup_stale_similarity_caches",
    "evict_expired_cache_entries",
    "cleanup_stale_caches",
    "get_cache_manager_stats",
    "MAX_SIMILARITY_CACHES",
    "CACHE_MANAGER_TTL_SECONDS",
    "PERIODIC_CLEANUP_INTERVAL_SECONDS",
    # Periodic cleanup management
    "get_periodic_cleanup_stats",
    "stop_periodic_cleanup",
    # Backends (re-exported)
    "SimilarityBackend",
    "JaccardBackend",
    "TFIDFBackend",
    "SentenceTransformerBackend",
    "get_similarity_backend",
    # Convergence
    "ConvergenceResult",
    "ArgumentDiversityMetric",
    "EvidenceConvergenceMetric",
    "StanceVolatilityMetric",
    "AdvancedConvergenceMetrics",
    "AdvancedConvergenceAnalyzer",
    "ConvergenceDetector",
]
