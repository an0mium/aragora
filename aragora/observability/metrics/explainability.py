"""
Batch explainability metrics for Aragora.

Provides metrics for:
- Active explainability jobs
- Job completion tracking
- Debate processing latency
- Error tracking
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

# Batch explainability metrics
BATCH_EXPLAINABILITY_JOBS_ACTIVE: Any = None
BATCH_EXPLAINABILITY_JOBS_TOTAL: Any = None
BATCH_EXPLAINABILITY_DEBATES_PROCESSED: Any = None
BATCH_EXPLAINABILITY_PROCESSING_LATENCY: Any = None
BATCH_EXPLAINABILITY_ERRORS_TOTAL: Any = None


def init_explainability_metrics() -> bool:
    """Initialize batch explainability Prometheus metrics."""
    global _initialized
    global BATCH_EXPLAINABILITY_JOBS_ACTIVE, BATCH_EXPLAINABILITY_JOBS_TOTAL
    global BATCH_EXPLAINABILITY_DEBATES_PROCESSED, BATCH_EXPLAINABILITY_PROCESSING_LATENCY
    global BATCH_EXPLAINABILITY_ERRORS_TOTAL

    if _initialized:
        return True

    config = get_metrics_config()
    if not config.enabled:
        _init_noop_metrics()
        _initialized = True
        return False

    try:
        from prometheus_client import Counter, Gauge, Histogram

        BATCH_EXPLAINABILITY_JOBS_ACTIVE = Gauge(
            "aragora_batch_explainability_jobs_active",
            "Number of active batch explainability jobs",
        )
        BATCH_EXPLAINABILITY_JOBS_TOTAL = Counter(
            "aragora_batch_explainability_jobs_total",
            "Total batch explainability jobs",
            ["status"],
        )
        BATCH_EXPLAINABILITY_DEBATES_PROCESSED = Counter(
            "aragora_batch_explainability_debates_processed_total",
            "Total debates processed in batch jobs",
            ["status"],
        )
        BATCH_EXPLAINABILITY_PROCESSING_LATENCY = Histogram(
            "aragora_batch_explainability_processing_latency_seconds",
            "Batch explainability processing latency",
            buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0],
        )
        BATCH_EXPLAINABILITY_ERRORS_TOTAL = Counter(
            "aragora_batch_explainability_errors_total",
            "Total batch explainability errors",
            ["error_type"],
        )

        _initialized = True
        return True

    except ImportError:
        logger.warning("prometheus_client not available, using no-op metrics")
        _init_noop_metrics()
        _initialized = True
        return False


def _init_noop_metrics() -> None:
    """Initialize no-op metrics when Prometheus is disabled."""
    global BATCH_EXPLAINABILITY_JOBS_ACTIVE, BATCH_EXPLAINABILITY_JOBS_TOTAL
    global BATCH_EXPLAINABILITY_DEBATES_PROCESSED, BATCH_EXPLAINABILITY_PROCESSING_LATENCY
    global BATCH_EXPLAINABILITY_ERRORS_TOTAL

    BATCH_EXPLAINABILITY_JOBS_ACTIVE = NoOpMetric()
    BATCH_EXPLAINABILITY_JOBS_TOTAL = NoOpMetric()
    BATCH_EXPLAINABILITY_DEBATES_PROCESSED = NoOpMetric()
    BATCH_EXPLAINABILITY_PROCESSING_LATENCY = NoOpMetric()
    BATCH_EXPLAINABILITY_ERRORS_TOTAL = NoOpMetric()


def _ensure_init() -> None:
    """Ensure metrics are initialized before use."""
    if not _initialized:
        init_explainability_metrics()


def set_batch_explainability_jobs_active(count: int) -> None:
    """Set the number of active batch explainability jobs.

    Args:
        count: Number of active jobs
    """
    _ensure_init()
    BATCH_EXPLAINABILITY_JOBS_ACTIVE.set(count)


def record_batch_explainability_job(status: str) -> None:
    """Record a batch explainability job.

    Args:
        status: Job status (started, completed, failed, cancelled)
    """
    _ensure_init()
    BATCH_EXPLAINABILITY_JOBS_TOTAL.labels(status=status).inc()


def record_batch_explainability_debate(
    status: str,
    latency_seconds: float,
) -> None:
    """Record a debate processed in a batch job.

    Args:
        status: Processing status (success, error)
        latency_seconds: Processing latency in seconds
    """
    _ensure_init()
    BATCH_EXPLAINABILITY_DEBATES_PROCESSED.labels(status=status).inc()
    BATCH_EXPLAINABILITY_PROCESSING_LATENCY.observe(latency_seconds)


def record_batch_explainability_error(error_type: str) -> None:
    """Record a batch explainability error.

    Args:
        error_type: Type of error (timeout, invalid_debate, generation_failed)
    """
    _ensure_init()
    BATCH_EXPLAINABILITY_ERRORS_TOTAL.labels(error_type=error_type).inc()


@contextmanager
def track_batch_explainability_debate() -> Generator[None, None, None]:
    """Context manager to track debate processing in batch jobs.

    Example:
        with track_batch_explainability_debate():
            explanation = await generate_explanation(debate_id)
    """
    _ensure_init()
    start = time.perf_counter()
    success = True
    try:
        yield
    except BaseException:
        success = False
        raise
    finally:
        latency = time.perf_counter() - start
        status = "success" if success else "error"
        record_batch_explainability_debate(status, latency)


__all__ = [
    "init_explainability_metrics",
    "set_batch_explainability_jobs_active",
    "record_batch_explainability_job",
    "record_batch_explainability_debate",
    "record_batch_explainability_error",
    "track_batch_explainability_debate",
    # Metrics (for direct access if needed)
    "BATCH_EXPLAINABILITY_JOBS_ACTIVE",
    "BATCH_EXPLAINABILITY_JOBS_TOTAL",
    "BATCH_EXPLAINABILITY_DEBATES_PROCESSED",
    "BATCH_EXPLAINABILITY_PROCESSING_LATENCY",
    "BATCH_EXPLAINABILITY_ERRORS_TOTAL",
]
