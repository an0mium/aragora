"""
Webhook Delivery Metrics.

Prometheus metrics for tracking webhook delivery performance and reliability.

Metrics exposed:
- aragora_webhook_deliveries_total: Counter of webhook delivery attempts
- aragora_webhook_delivery_duration_seconds: Histogram of delivery duration
- aragora_webhook_delivery_retries_total: Counter of retry attempts
- aragora_webhook_queue_size: Gauge of pending deliveries
- aragora_webhook_active_endpoints: Gauge of registered webhook endpoints

Usage:
    from aragora.observability.metrics.webhook import (
        record_webhook_delivery,
        record_webhook_retry,
        set_queue_size,
    )

    # Record a successful delivery
    record_webhook_delivery(
        event_type="debate_end",
        success=True,
        duration_seconds=0.25,
    )

    # Record a retry attempt
    record_webhook_retry(event_type="slo_violation", attempt=2)
"""

import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

# Lazy-loaded Prometheus metrics
_metrics_initialized = False
_DELIVERIES_TOTAL = None
_DELIVERY_DURATION = None
_RETRIES_TOTAL = None
_QUEUE_SIZE = None
_ACTIVE_ENDPOINTS = None
_FAILURES_BY_STATUS = None


def _get_or_create_metric(metric_class, name: str, description: str, labelnames=None, **kwargs):
    """Get existing metric from registry or create new one.

    If a metric with the same name exists (even with different labels),
    we return the existing one since prometheus doesn't allow duplicate names.
    """
    from prometheus_client import REGISTRY

    # Check if metric already exists by name in the default registry
    # _names_to_collectors is keyed by metric name
    if name in REGISTRY._names_to_collectors:
        return REGISTRY._names_to_collectors[name]

    # Also check for metrics with suffixes (Counter has _total, _created)
    # and Histogram has _bucket, _count, _sum
    base_name = name.replace("_total", "").replace("_created", "")
    if base_name in REGISTRY._names_to_collectors:
        return REGISTRY._names_to_collectors[base_name]

    # Create new metric
    try:
        if labelnames:
            return metric_class(name, description, labelnames, **kwargs)
        return metric_class(name, description, **kwargs)
    except ValueError:
        # Metric was registered by another thread/module - try to get it
        if name in REGISTRY._names_to_collectors:
            return REGISTRY._names_to_collectors[name]
        raise


def _init_metrics():
    """Initialize Prometheus metrics lazily."""
    global _metrics_initialized
    global _DELIVERIES_TOTAL, _DELIVERY_DURATION, _RETRIES_TOTAL
    global _QUEUE_SIZE, _ACTIVE_ENDPOINTS, _FAILURES_BY_STATUS

    if _metrics_initialized:
        return True

    try:
        from prometheus_client import Counter, Gauge, Histogram

        # Set flag BEFORE registering metrics (optimistic locking)
        # This prevents duplicate registration if called concurrently
        _metrics_initialized = True

        _DELIVERIES_TOTAL = _get_or_create_metric(
            Counter,
            "aragora_webhook_deliveries_total",
            "Total webhook delivery attempts",
            ["event_type", "success"],
        )

        _DELIVERY_DURATION = _get_or_create_metric(
            Histogram,
            "aragora_webhook_delivery_duration_seconds",
            "Webhook delivery duration in seconds",
            ["event_type"],
            buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
        )

        _RETRIES_TOTAL = _get_or_create_metric(
            Counter,
            "aragora_webhook_delivery_retries_total",
            "Total webhook delivery retry attempts",
            ["event_type", "attempt"],
        )

        _QUEUE_SIZE = _get_or_create_metric(
            Gauge,
            "aragora_webhook_queue_size",
            "Current number of pending webhook deliveries",
        )

        _ACTIVE_ENDPOINTS = _get_or_create_metric(
            Gauge,
            "aragora_webhook_active_endpoints",
            "Number of active webhook endpoints",
            ["event_type"],
        )

        _FAILURES_BY_STATUS = _get_or_create_metric(
            Counter,
            "aragora_webhook_failures_by_status_total",
            "Webhook failures by HTTP status code",
            ["event_type", "status_code"],
        )

        logger.debug("Webhook Prometheus metrics initialized")
        return True

    except ImportError:
        logger.debug("prometheus_client not installed, webhook metrics disabled")
        return False


def record_webhook_delivery(
    event_type: str,
    success: bool,
    duration_seconds: float,
    status_code: Optional[int] = None,
) -> None:
    """
    Record a webhook delivery attempt.

    Args:
        event_type: Type of event being delivered (e.g., "debate_end")
        success: Whether delivery was successful
        duration_seconds: Time taken for delivery
        status_code: HTTP status code (if applicable)
    """
    if not _init_metrics():
        return

    # Record delivery counter
    if _DELIVERIES_TOTAL is not None:
        _DELIVERIES_TOTAL.labels(
            event_type=event_type,
            success=str(success).lower(),
        ).inc()

    # Record duration histogram
    if _DELIVERY_DURATION is not None:
        _DELIVERY_DURATION.labels(event_type=event_type).observe(duration_seconds)

    # Record failure by status code
    if not success and status_code is not None and _FAILURES_BY_STATUS:
        _FAILURES_BY_STATUS.labels(
            event_type=event_type,
            status_code=str(status_code),
        ).inc()


def record_webhook_retry(event_type: str, attempt: int) -> None:
    """
    Record a webhook delivery retry attempt.

    Args:
        event_type: Type of event being retried
        attempt: Retry attempt number (1, 2, 3, etc.)
    """
    if not _init_metrics():
        return

    if _RETRIES_TOTAL is not None:
        _RETRIES_TOTAL.labels(
            event_type=event_type,
            attempt=str(min(attempt, 5)),  # Cap at 5 to limit cardinality
        ).inc()


def set_queue_size(size: int) -> None:
    """
    Set the current webhook delivery queue size.

    Args:
        size: Number of pending deliveries
    """
    if not _init_metrics():
        return

    if _QUEUE_SIZE is not None:
        _QUEUE_SIZE.set(size)


def set_active_endpoints(event_type: str, count: int) -> None:
    """
    Set the number of active webhook endpoints for an event type.

    Args:
        event_type: Event type
        count: Number of active endpoints
    """
    if not _init_metrics():
        return

    if _ACTIVE_ENDPOINTS is not None:
        _ACTIVE_ENDPOINTS.labels(event_type=event_type).set(count)


def increment_queue() -> None:
    """Increment the queue size by 1."""
    if not _init_metrics():
        return
    if _QUEUE_SIZE is not None:
        _QUEUE_SIZE.inc()


def decrement_queue() -> None:
    """Decrement the queue size by 1."""
    if not _init_metrics():
        return
    if _QUEUE_SIZE is not None:
        _QUEUE_SIZE.dec()


class WebhookDeliveryTimer:
    """
    Context manager for timing webhook deliveries.

    Usage:
        with WebhookDeliveryTimer("debate_end") as timer:
            # Perform delivery
            success = deliver_webhook(...)
            timer.set_success(success)
    """

    def __init__(self, event_type: str):
        self.event_type = event_type
        self.success = True
        self.status_code: int | None = None
        self._start_time: float | None = None

    def __enter__(self):
        self._start_time = time.time()
        increment_queue()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self._start_time
        decrement_queue()

        if exc_type is not None:
            self.success = False

        record_webhook_delivery(
            event_type=self.event_type,
            success=self.success,
            duration_seconds=duration,
            status_code=self.status_code,
        )

        return False  # Don't suppress exceptions

    def set_success(self, success: bool, status_code: Optional[int] = None) -> None:
        """Set the delivery success status."""
        self.success = success
        self.status_code = status_code


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "record_webhook_delivery",
    "record_webhook_retry",
    "set_queue_size",
    "set_active_endpoints",
    "increment_queue",
    "decrement_queue",
    "WebhookDeliveryTimer",
]
