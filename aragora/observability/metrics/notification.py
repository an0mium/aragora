"""
Notification delivery metrics.

Provides Prometheus metrics for tracking notification delivery
across channels (Slack, Email, Webhook), including latency and errors.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Any, Generator

from aragora.observability.metrics.base import NoOpMetric, get_metrics_enabled

logger = logging.getLogger(__name__)

# Global metric variables
NOTIFICATION_SENT_TOTAL: Any = None
NOTIFICATION_LATENCY: Any = None
NOTIFICATION_ERRORS_TOTAL: Any = None
NOTIFICATION_QUEUE_SIZE: Any = None

_initialized = False


def init_notification_metrics() -> None:
    """Initialize notification delivery metrics."""
    global _initialized
    global NOTIFICATION_SENT_TOTAL, NOTIFICATION_LATENCY
    global NOTIFICATION_ERRORS_TOTAL, NOTIFICATION_QUEUE_SIZE

    if _initialized:
        return

    if not get_metrics_enabled():
        _init_noop_metrics()
        _initialized = True
        return

    try:
        from prometheus_client import Counter, Gauge, Histogram

        NOTIFICATION_SENT_TOTAL = Counter(
            "aragora_notification_sent_total",
            "Total notifications sent",
            ["channel", "severity", "priority", "status"],
        )

        NOTIFICATION_LATENCY = Histogram(
            "aragora_notification_latency_seconds",
            "Notification delivery latency in seconds",
            ["channel"],
            buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
        )

        NOTIFICATION_ERRORS_TOTAL = Counter(
            "aragora_notification_errors_total",
            "Total notification delivery errors",
            ["channel", "error_type"],
        )

        NOTIFICATION_QUEUE_SIZE = Gauge(
            "aragora_notification_queue_size",
            "Current notification queue size",
            ["channel"],
        )

        _initialized = True
        logger.debug("Notification metrics initialized")

    except ImportError:
        _init_noop_metrics()
        _initialized = True


def _init_noop_metrics() -> None:
    """Initialize no-op metrics when Prometheus is disabled."""
    global NOTIFICATION_SENT_TOTAL, NOTIFICATION_LATENCY
    global NOTIFICATION_ERRORS_TOTAL, NOTIFICATION_QUEUE_SIZE

    NOTIFICATION_SENT_TOTAL = NoOpMetric()
    NOTIFICATION_LATENCY = NoOpMetric()
    NOTIFICATION_ERRORS_TOTAL = NoOpMetric()
    NOTIFICATION_QUEUE_SIZE = NoOpMetric()


def _ensure_init() -> None:
    """Ensure metrics are initialized."""
    if not _initialized:
        init_notification_metrics()


# =============================================================================
# Recording Functions
# =============================================================================


def record_notification_sent(
    channel: str,
    severity: str,
    priority: str,
    success: bool,
    latency_seconds: float,
) -> None:
    """Record a notification delivery attempt.

    Args:
        channel: Notification channel (slack, email, webhook, in_app)
        severity: Notification severity (info, warning, error, critical)
        priority: Notification priority (low, normal, high, urgent)
        success: Whether the delivery succeeded
        latency_seconds: Delivery latency in seconds
    """
    _ensure_init()
    status = "success" if success else "failed"
    NOTIFICATION_SENT_TOTAL.labels(
        channel=channel, severity=severity, priority=priority, status=status
    ).inc()
    NOTIFICATION_LATENCY.labels(channel=channel).observe(latency_seconds)


def record_notification_error(channel: str, error_type: str) -> None:
    """Record a notification delivery error.

    Args:
        channel: Notification channel (slack, email, webhook)
        error_type: Error category (timeout, auth_failed, rate_limited, connection_error, etc.)
    """
    _ensure_init()
    NOTIFICATION_ERRORS_TOTAL.labels(channel=channel, error_type=error_type).inc()


def set_notification_queue_size(channel: str, size: int) -> None:
    """Set the current notification queue size.

    Args:
        channel: Notification channel
        size: Current queue size
    """
    _ensure_init()
    NOTIFICATION_QUEUE_SIZE.labels(channel=channel).set(size)


@contextmanager
def track_notification_delivery(
    channel: str,
    severity: str = "info",
    priority: str = "normal",
) -> Generator[None, None, None]:
    """Context manager to track notification delivery.

    Automatically records latency and success/failure.

    Args:
        channel: Notification channel
        severity: Notification severity
        priority: Notification priority

    Example:
        with track_notification_delivery("slack", "warning", "high"):
            await send_slack_message(...)
    """
    _ensure_init()
    start = time.perf_counter()
    success = True
    try:
        yield
    except Exception:
        success = False
        raise
    finally:
        latency = time.perf_counter() - start
        record_notification_sent(channel, severity, priority, success, latency)


__all__ = [
    # Metrics
    "NOTIFICATION_SENT_TOTAL",
    "NOTIFICATION_LATENCY",
    "NOTIFICATION_ERRORS_TOTAL",
    "NOTIFICATION_QUEUE_SIZE",
    # Functions
    "record_notification_sent",
    "record_notification_error",
    "set_notification_queue_size",
    "track_notification_delivery",
    "init_notification_metrics",
]
