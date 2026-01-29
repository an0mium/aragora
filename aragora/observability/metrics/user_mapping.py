"""
User ID mapping metrics.

Provides Prometheus metrics for tracking user ID mapping operations
across different platforms (Slack, Discord, Teams).
"""

from __future__ import annotations

import logging
from typing import Any

from aragora.observability.metrics.base import NoOpMetric, get_metrics_enabled

logger = logging.getLogger(__name__)

# Global metric variables
USER_MAPPING_OPERATIONS_TOTAL: Any = None
USER_MAPPING_CACHE_HITS_TOTAL: Any = None
USER_MAPPING_CACHE_MISSES_TOTAL: Any = None
USER_MAPPINGS_ACTIVE: Any = None

_initialized = False


def init_user_mapping_metrics() -> None:
    """Initialize user ID mapping metrics."""
    global _initialized
    global USER_MAPPING_OPERATIONS_TOTAL, USER_MAPPING_CACHE_HITS_TOTAL
    global USER_MAPPING_CACHE_MISSES_TOTAL, USER_MAPPINGS_ACTIVE

    if _initialized:
        return

    if not get_metrics_enabled():
        _init_noop_metrics()
        _initialized = True
        return

    try:
        from prometheus_client import Counter, Gauge

        USER_MAPPING_OPERATIONS_TOTAL = Counter(
            "aragora_user_mapping_operations_total",
            "Total user ID mapping operations",
            ["operation", "platform", "status"],
        )

        USER_MAPPING_CACHE_HITS_TOTAL = Counter(
            "aragora_user_mapping_cache_hits_total",
            "Total user ID mapping cache hits",
            ["platform"],
        )

        USER_MAPPING_CACHE_MISSES_TOTAL = Counter(
            "aragora_user_mapping_cache_misses_total",
            "Total user ID mapping cache misses",
            ["platform"],
        )

        USER_MAPPINGS_ACTIVE = Gauge(
            "aragora_user_mappings_active",
            "Current number of active user ID mappings",
            ["platform"],
        )

        _initialized = True
        logger.debug("User mapping metrics initialized")

    except ImportError:
        _init_noop_metrics()
        _initialized = True


def _init_noop_metrics() -> None:
    """Initialize no-op metrics when Prometheus is disabled."""
    global USER_MAPPING_OPERATIONS_TOTAL, USER_MAPPING_CACHE_HITS_TOTAL
    global USER_MAPPING_CACHE_MISSES_TOTAL, USER_MAPPINGS_ACTIVE

    USER_MAPPING_OPERATIONS_TOTAL = NoOpMetric()
    USER_MAPPING_CACHE_HITS_TOTAL = NoOpMetric()
    USER_MAPPING_CACHE_MISSES_TOTAL = NoOpMetric()
    USER_MAPPINGS_ACTIVE = NoOpMetric()


def _ensure_init() -> None:
    """Ensure metrics are initialized."""
    if not _initialized:
        init_user_mapping_metrics()


# =============================================================================
# Recording Functions
# =============================================================================


def record_user_mapping_operation(operation: str, platform: str, found: bool) -> None:
    """Record a user ID mapping operation.

    Args:
        operation: Operation type (save, get, delete)
        platform: Platform name (slack, discord, teams)
        found: Whether the mapping was found (for get operations)
    """
    _ensure_init()
    status = "success" if found else "not_found"
    USER_MAPPING_OPERATIONS_TOTAL.labels(
        operation=operation, platform=platform, status=status
    ).inc()


def record_user_mapping_cache_hit(platform: str) -> None:
    """Record a user ID mapping cache hit.

    Args:
        platform: Platform name (slack, discord, teams)
    """
    _ensure_init()
    USER_MAPPING_CACHE_HITS_TOTAL.labels(platform=platform).inc()


def record_user_mapping_cache_miss(platform: str) -> None:
    """Record a user ID mapping cache miss.

    Args:
        platform: Platform name (slack, discord, teams)
    """
    _ensure_init()
    USER_MAPPING_CACHE_MISSES_TOTAL.labels(platform=platform).inc()


def set_user_mappings_active(platform: str, count: int) -> None:
    """Set the number of active user ID mappings for a platform.

    Args:
        platform: Platform name (slack, discord, teams)
        count: Number of active mappings
    """
    _ensure_init()
    USER_MAPPINGS_ACTIVE.labels(platform=platform).set(count)


__all__ = [
    # Metrics
    "USER_MAPPING_OPERATIONS_TOTAL",
    "USER_MAPPING_CACHE_HITS_TOTAL",
    "USER_MAPPING_CACHE_MISSES_TOTAL",
    "USER_MAPPINGS_ACTIVE",
    # Functions
    "init_user_mapping_metrics",
    "record_user_mapping_operation",
    "record_user_mapping_cache_hit",
    "record_user_mapping_cache_miss",
    "set_user_mappings_active",
]
