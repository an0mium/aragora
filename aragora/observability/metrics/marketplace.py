"""
Marketplace metrics for Aragora.

Provides metrics for:
- Template counts and visibility
- Download tracking
- Ratings and reviews
- Operation latency
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Any
from collections.abc import Generator

from aragora.observability.config import get_metrics_config
from aragora.observability.metrics.base import NoOpMetric

logger = logging.getLogger(__name__)

# Module-level initialization state
_initialized = False

# Marketplace metrics
MARKETPLACE_TEMPLATES_TOTAL: Any = None
MARKETPLACE_DOWNLOADS_TOTAL: Any = None
MARKETPLACE_RATINGS_TOTAL: Any = None
MARKETPLACE_RATINGS_DISTRIBUTION: Any = None
MARKETPLACE_REVIEWS_TOTAL: Any = None
MARKETPLACE_OPERATION_LATENCY: Any = None


def init_marketplace_metrics() -> bool:
    """Initialize marketplace Prometheus metrics."""
    global _initialized
    global MARKETPLACE_TEMPLATES_TOTAL, MARKETPLACE_DOWNLOADS_TOTAL
    global MARKETPLACE_RATINGS_TOTAL, MARKETPLACE_RATINGS_DISTRIBUTION
    global MARKETPLACE_REVIEWS_TOTAL, MARKETPLACE_OPERATION_LATENCY

    if _initialized:
        return True

    config = get_metrics_config()
    if not config.enabled:
        _init_noop_metrics()
        _initialized = True
        return False

    try:
        from prometheus_client import Counter, Gauge, Histogram

        MARKETPLACE_TEMPLATES_TOTAL = Gauge(
            "aragora_marketplace_templates_total",
            "Total templates in marketplace",
            ["category", "visibility"],
        )
        MARKETPLACE_DOWNLOADS_TOTAL = Counter(
            "aragora_marketplace_downloads_total",
            "Total template downloads",
            ["template_id", "category"],
        )
        MARKETPLACE_RATINGS_TOTAL = Counter(
            "aragora_marketplace_ratings_total",
            "Total template ratings",
            ["template_id"],
        )
        MARKETPLACE_RATINGS_DISTRIBUTION = Histogram(
            "aragora_marketplace_ratings_distribution",
            "Distribution of template ratings",
            ["category"],
            buckets=[1, 2, 3, 4, 5],
        )
        MARKETPLACE_REVIEWS_TOTAL = Counter(
            "aragora_marketplace_reviews_total",
            "Total template reviews",
            ["template_id", "status"],
        )
        MARKETPLACE_OPERATION_LATENCY = Histogram(
            "aragora_marketplace_operation_latency_seconds",
            "Marketplace operation latency",
            ["operation"],
            buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
        )

        _initialized = True
        return True

    except (ImportError, ValueError):
        logger.warning("prometheus_client not available, using no-op metrics")
        _init_noop_metrics()
        _initialized = True
        return False


def _init_noop_metrics() -> None:
    """Initialize no-op metrics when Prometheus is disabled."""
    global MARKETPLACE_TEMPLATES_TOTAL, MARKETPLACE_DOWNLOADS_TOTAL
    global MARKETPLACE_RATINGS_TOTAL, MARKETPLACE_RATINGS_DISTRIBUTION
    global MARKETPLACE_REVIEWS_TOTAL, MARKETPLACE_OPERATION_LATENCY

    MARKETPLACE_TEMPLATES_TOTAL = NoOpMetric()
    MARKETPLACE_DOWNLOADS_TOTAL = NoOpMetric()
    MARKETPLACE_RATINGS_TOTAL = NoOpMetric()
    MARKETPLACE_RATINGS_DISTRIBUTION = NoOpMetric()
    MARKETPLACE_REVIEWS_TOTAL = NoOpMetric()
    MARKETPLACE_OPERATION_LATENCY = NoOpMetric()


def _ensure_init() -> None:
    """Ensure metrics are initialized before use."""
    if not _initialized:
        init_marketplace_metrics()


def set_marketplace_templates_count(
    category: str,
    visibility: str,
    count: int,
) -> None:
    """Set the total number of templates in marketplace.

    Args:
        category: Template category (workflow, debate, analysis)
        visibility: Visibility level (public, private, team)
        count: Number of templates
    """
    _ensure_init()
    MARKETPLACE_TEMPLATES_TOTAL.labels(category=category, visibility=visibility).set(count)


def record_marketplace_download(
    template_id: str,
    category: str,
) -> None:
    """Record a template download.

    Args:
        template_id: ID of the downloaded template
        category: Template category
    """
    _ensure_init()
    MARKETPLACE_DOWNLOADS_TOTAL.labels(template_id=template_id, category=category).inc()


def record_marketplace_rating(
    template_id: str,
    category: str,
    rating: int,
) -> None:
    """Record a template rating.

    Args:
        template_id: ID of the rated template
        category: Template category
        rating: Rating value (1-5)
    """
    _ensure_init()
    MARKETPLACE_RATINGS_TOTAL.labels(template_id=template_id).inc()
    MARKETPLACE_RATINGS_DISTRIBUTION.labels(category=category).observe(rating)


def record_marketplace_review(
    template_id: str,
    status: str,
) -> None:
    """Record a template review.

    Args:
        template_id: ID of the reviewed template
        status: Review status (submitted, approved, rejected)
    """
    _ensure_init()
    MARKETPLACE_REVIEWS_TOTAL.labels(template_id=template_id, status=status).inc()


def record_marketplace_operation_latency(
    operation: str,
    latency_seconds: float,
) -> None:
    """Record marketplace operation latency.

    Args:
        operation: Operation type (list, search, publish, download, rate)
        latency_seconds: Operation latency in seconds
    """
    _ensure_init()
    MARKETPLACE_OPERATION_LATENCY.labels(operation=operation).observe(latency_seconds)


@contextmanager
def track_marketplace_operation(operation: str) -> Generator[None, None, None]:
    """Context manager to track marketplace operations.

    Args:
        operation: Operation type (list, search, publish, download, rate)

    Example:
        with track_marketplace_operation("search"):
            results = await search_templates(query)
    """
    _ensure_init()
    start = time.perf_counter()
    try:
        yield
    finally:
        latency = time.perf_counter() - start
        record_marketplace_operation_latency(operation, latency)


__all__ = [
    "init_marketplace_metrics",
    "set_marketplace_templates_count",
    "record_marketplace_download",
    "record_marketplace_rating",
    "record_marketplace_review",
    "record_marketplace_operation_latency",
    "track_marketplace_operation",
    # Metrics (for direct access if needed)
    "MARKETPLACE_TEMPLATES_TOTAL",
    "MARKETPLACE_DOWNLOADS_TOTAL",
    "MARKETPLACE_RATINGS_TOTAL",
    "MARKETPLACE_RATINGS_DISTRIBUTION",
    "MARKETPLACE_REVIEWS_TOTAL",
    "MARKETPLACE_OPERATION_LATENCY",
]
