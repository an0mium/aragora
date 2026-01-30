"""
Cache metrics.

Provides Prometheus metrics for tracking cache hit/miss rates across
different cache subsystems including general caches, knowledge caches,
and RLM caches.
"""

from __future__ import annotations

import logging
from typing import Any

from aragora.observability.metrics.base import NoOpMetric, get_metrics_enabled

logger = logging.getLogger(__name__)

# General cache metrics
CACHE_HITS: Any = None
CACHE_MISSES: Any = None

# Knowledge cache metrics
KNOWLEDGE_CACHE_HITS: Any = None
KNOWLEDGE_CACHE_MISSES: Any = None

# RLM cache metrics
RLM_CACHE_HITS: Any = None
RLM_CACHE_MISSES: Any = None

_initialized = False


def init_cache_metrics() -> None:
    """Initialize cache metrics."""
    global _initialized
    global CACHE_HITS, CACHE_MISSES
    global KNOWLEDGE_CACHE_HITS, KNOWLEDGE_CACHE_MISSES
    global RLM_CACHE_HITS, RLM_CACHE_MISSES

    if _initialized:
        return

    if not get_metrics_enabled():
        _init_noop_metrics()
        _initialized = True
        return

    try:
        from prometheus_client import Counter

        # General cache metrics
        CACHE_HITS = Counter(
            "aragora_cache_hits_total",
            "Cache hit count",
            ["cache_name"],
        )

        CACHE_MISSES = Counter(
            "aragora_cache_misses_total",
            "Cache miss count",
            ["cache_name"],
        )

        # Knowledge cache metrics
        KNOWLEDGE_CACHE_HITS = Counter(
            "aragora_knowledge_cache_hits_total",
            "Knowledge query cache hits",
        )

        KNOWLEDGE_CACHE_MISSES = Counter(
            "aragora_knowledge_cache_misses_total",
            "Knowledge query cache misses",
        )

        # RLM cache metrics
        RLM_CACHE_HITS = Counter(
            "aragora_rlm_cache_hits_total",
            "RLM compression cache hits",
        )

        RLM_CACHE_MISSES = Counter(
            "aragora_rlm_cache_misses_total",
            "RLM compression cache misses",
        )

        _initialized = True
        logger.debug("Cache metrics initialized")

    except ImportError:
        _init_noop_metrics()
        _initialized = True


def _init_noop_metrics() -> None:
    """Initialize no-op metrics when Prometheus is disabled."""
    global CACHE_HITS, CACHE_MISSES
    global KNOWLEDGE_CACHE_HITS, KNOWLEDGE_CACHE_MISSES
    global RLM_CACHE_HITS, RLM_CACHE_MISSES

    noop = NoOpMetric()
    CACHE_HITS = noop
    CACHE_MISSES = noop
    KNOWLEDGE_CACHE_HITS = noop
    KNOWLEDGE_CACHE_MISSES = noop
    RLM_CACHE_HITS = noop
    RLM_CACHE_MISSES = noop


def _ensure_init() -> None:
    """Ensure metrics are initialized."""
    if not _initialized:
        init_cache_metrics()


# =============================================================================
# General Cache Recording Functions
# =============================================================================


def record_cache_hit(cache_name: str) -> None:
    """Record a cache hit.

    Args:
        cache_name: Name of the cache that was hit
    """
    _ensure_init()
    CACHE_HITS.labels(cache_name=cache_name).inc()


def record_cache_miss(cache_name: str) -> None:
    """Record a cache miss.

    Args:
        cache_name: Name of the cache that was missed
    """
    _ensure_init()
    CACHE_MISSES.labels(cache_name=cache_name).inc()


# =============================================================================
# Knowledge Cache Recording Functions
# =============================================================================


def record_knowledge_cache_hit() -> None:
    """Record a knowledge query cache hit."""
    _ensure_init()
    KNOWLEDGE_CACHE_HITS.inc()


def record_knowledge_cache_miss() -> None:
    """Record a knowledge query cache miss."""
    _ensure_init()
    KNOWLEDGE_CACHE_MISSES.inc()


# =============================================================================
# RLM Cache Recording Functions
# =============================================================================


def record_rlm_cache_hit() -> None:
    """Record an RLM compression cache hit."""
    _ensure_init()
    RLM_CACHE_HITS.inc()


def record_rlm_cache_miss() -> None:
    """Record an RLM compression cache miss."""
    _ensure_init()
    RLM_CACHE_MISSES.inc()


__all__ = [
    # Metrics
    "CACHE_HITS",
    "CACHE_MISSES",
    "KNOWLEDGE_CACHE_HITS",
    "KNOWLEDGE_CACHE_MISSES",
    "RLM_CACHE_HITS",
    "RLM_CACHE_MISSES",
    # Functions
    "init_cache_metrics",
    "record_cache_hit",
    "record_cache_miss",
    "record_knowledge_cache_hit",
    "record_knowledge_cache_miss",
    "record_rlm_cache_hit",
    "record_rlm_cache_miss",
]
