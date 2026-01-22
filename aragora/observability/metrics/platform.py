"""
Platform-specific metrics for chat integrations.

Tracks delivery success rates, latencies, circuit breaker states,
and dead letter queue statistics across all chat platforms.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from aragora.observability.config import get_metrics_config

logger = logging.getLogger(__name__)

# Metric instances - initialized lazily
_initialized = False

# Platform delivery metrics
PLATFORM_REQUESTS_TOTAL: Any = None
PLATFORM_REQUEST_LATENCY: Any = None
PLATFORM_ERRORS_TOTAL: Any = None
PLATFORM_CIRCUIT_STATE: Any = None
PLATFORM_RATE_LIMIT_REMAINING: Any = None
PLATFORM_RATE_LIMIT_EXCEEDED: Any = None

# Dead letter queue metrics
DLQ_ENQUEUED_TOTAL: Any = None
DLQ_PROCESSED_TOTAL: Any = None
DLQ_FAILED_TOTAL: Any = None
DLQ_PENDING_GAUGE: Any = None
DLQ_RETRY_LATENCY: Any = None

# Webhook metrics (per-platform)
WEBHOOK_DELIVERY_TOTAL: Any = None
WEBHOOK_DELIVERY_LATENCY: Any = None
WEBHOOK_RETRY_TOTAL: Any = None

# Bot handler metrics
BOT_COMMAND_TOTAL: Any = None
BOT_COMMAND_LATENCY: Any = None
BOT_COMMAND_TIMEOUT_TOTAL: Any = None


def _initialize_platform_metrics() -> None:
    """Initialize Prometheus metrics for platforms."""
    global _initialized
    global PLATFORM_REQUESTS_TOTAL, PLATFORM_REQUEST_LATENCY, PLATFORM_ERRORS_TOTAL
    global PLATFORM_CIRCUIT_STATE, PLATFORM_RATE_LIMIT_REMAINING, PLATFORM_RATE_LIMIT_EXCEEDED
    global DLQ_ENQUEUED_TOTAL, DLQ_PROCESSED_TOTAL, DLQ_FAILED_TOTAL
    global DLQ_PENDING_GAUGE, DLQ_RETRY_LATENCY
    global WEBHOOK_DELIVERY_TOTAL, WEBHOOK_DELIVERY_LATENCY, WEBHOOK_RETRY_TOTAL
    global BOT_COMMAND_TOTAL, BOT_COMMAND_LATENCY, BOT_COMMAND_TIMEOUT_TOTAL

    if _initialized:
        return

    config = get_metrics_config()
    if not config.enabled:
        _initialized = True
        return

    try:
        from prometheus_client import Counter, Gauge, Histogram

        # Platform request metrics
        PLATFORM_REQUESTS_TOTAL = Counter(
            "aragora_platform_requests_total",
            "Total platform delivery requests",
            ["platform", "operation", "status"],
        )

        PLATFORM_REQUEST_LATENCY = Histogram(
            "aragora_platform_request_latency_ms",
            "Platform request latency in milliseconds",
            ["platform", "operation"],
            buckets=[10, 50, 100, 250, 500, 1000, 2500, 5000, 10000],
        )

        PLATFORM_ERRORS_TOTAL = Counter(
            "aragora_platform_errors_total",
            "Total platform errors by type",
            ["platform", "error_type"],
        )

        PLATFORM_CIRCUIT_STATE = Gauge(
            "aragora_platform_circuit_state",
            "Platform circuit breaker state (0=closed, 1=half-open, 2=open)",
            ["platform"],
        )

        PLATFORM_RATE_LIMIT_REMAINING = Gauge(
            "aragora_platform_rate_limit_remaining",
            "Remaining rate limit capacity",
            ["platform", "key"],
        )

        PLATFORM_RATE_LIMIT_EXCEEDED = Counter(
            "aragora_platform_rate_limit_exceeded_total",
            "Rate limit exceeded events",
            ["platform", "key"],
        )

        # Dead letter queue metrics
        DLQ_ENQUEUED_TOTAL = Counter(
            "aragora_dlq_enqueued_total",
            "Messages added to dead letter queue",
            ["platform"],
        )

        DLQ_PROCESSED_TOTAL = Counter(
            "aragora_dlq_processed_total",
            "Messages successfully processed from DLQ",
            ["platform"],
        )

        DLQ_FAILED_TOTAL = Counter(
            "aragora_dlq_failed_total",
            "Messages that exceeded max retries",
            ["platform"],
        )

        DLQ_PENDING_GAUGE = Gauge(
            "aragora_dlq_pending",
            "Messages pending in dead letter queue",
            ["platform"],
        )

        DLQ_RETRY_LATENCY = Histogram(
            "aragora_dlq_retry_latency_seconds",
            "Time from enqueue to successful delivery",
            ["platform"],
            buckets=[60, 300, 900, 1800, 3600, 7200, 14400],
        )

        # Webhook delivery metrics
        WEBHOOK_DELIVERY_TOTAL = Counter(
            "aragora_webhook_platform_delivery_total",
            "Webhook deliveries by platform",
            ["platform", "status"],
        )

        WEBHOOK_DELIVERY_LATENCY = Histogram(
            "aragora_webhook_platform_latency_ms",
            "Webhook delivery latency by platform",
            ["platform"],
            buckets=[50, 100, 250, 500, 1000, 2500, 5000],
        )

        WEBHOOK_RETRY_TOTAL = Counter(
            "aragora_webhook_platform_retry_total",
            "Webhook retry attempts by platform",
            ["platform", "attempt"],
        )

        # Bot handler metrics
        BOT_COMMAND_TOTAL = Counter(
            "aragora_bot_command_total",
            "Bot commands processed",
            ["platform", "command", "status"],
        )

        BOT_COMMAND_LATENCY = Histogram(
            "aragora_bot_command_latency_ms",
            "Bot command processing latency",
            ["platform", "command"],
            buckets=[100, 500, 1000, 2500, 5000, 10000, 25000],
        )

        BOT_COMMAND_TIMEOUT_TOTAL = Counter(
            "aragora_bot_command_timeout_total",
            "Bot command timeouts",
            ["platform", "command"],
        )

        _initialized = True
        logger.info("Platform metrics initialized")

    except ImportError:
        logger.warning("prometheus_client not installed, platform metrics disabled")
        _initialized = True


def _ensure_initialized() -> None:
    """Ensure metrics are initialized."""
    if not _initialized:
        _initialize_platform_metrics()


# =============================================================================
# Recording Functions
# =============================================================================


def record_platform_request(
    platform: str,
    operation: str,
    success: bool,
    latency_ms: float,
    error_type: Optional[str] = None,
) -> None:
    """Record a platform delivery request."""
    _ensure_initialized()

    if PLATFORM_REQUESTS_TOTAL is not None:
        status = "success" if success else "error"
        PLATFORM_REQUESTS_TOTAL.labels(platform=platform, operation=operation, status=status).inc()

    if PLATFORM_REQUEST_LATENCY is not None:
        PLATFORM_REQUEST_LATENCY.labels(platform=platform, operation=operation).observe(latency_ms)

    if not success and PLATFORM_ERRORS_TOTAL is not None and error_type:
        PLATFORM_ERRORS_TOTAL.labels(platform=platform, error_type=error_type).inc()


def record_circuit_state(platform: str, state: str) -> None:
    """Record circuit breaker state change."""
    _ensure_initialized()

    if PLATFORM_CIRCUIT_STATE is not None:
        state_value = {"closed": 0, "half-open": 1, "open": 2}.get(state, 0)
        PLATFORM_CIRCUIT_STATE.labels(platform=platform).set(state_value)


def record_rate_limit(platform: str, key: str, remaining: int, exceeded: bool = False) -> None:
    """Record rate limit status."""
    _ensure_initialized()

    if PLATFORM_RATE_LIMIT_REMAINING is not None:
        PLATFORM_RATE_LIMIT_REMAINING.labels(platform=platform, key=key).set(remaining)

    if exceeded and PLATFORM_RATE_LIMIT_EXCEEDED is not None:
        PLATFORM_RATE_LIMIT_EXCEEDED.labels(platform=platform, key=key).inc()


def record_dlq_enqueue(platform: str) -> None:
    """Record message added to DLQ."""
    _ensure_initialized()

    if DLQ_ENQUEUED_TOTAL is not None:
        DLQ_ENQUEUED_TOTAL.labels(platform=platform).inc()


def record_dlq_processed(platform: str) -> None:
    """Record message successfully processed from DLQ."""
    _ensure_initialized()

    if DLQ_PROCESSED_TOTAL is not None:
        DLQ_PROCESSED_TOTAL.labels(platform=platform).inc()


def record_dlq_failed(platform: str) -> None:
    """Record message that exceeded max retries."""
    _ensure_initialized()

    if DLQ_FAILED_TOTAL is not None:
        DLQ_FAILED_TOTAL.labels(platform=platform).inc()


def update_dlq_pending(platform: str, count: int) -> None:
    """Update pending DLQ message count."""
    _ensure_initialized()

    if DLQ_PENDING_GAUGE is not None:
        DLQ_PENDING_GAUGE.labels(platform=platform).set(count)


def record_dlq_retry_latency(platform: str, latency_seconds: float) -> None:
    """Record time from enqueue to successful delivery."""
    _ensure_initialized()

    if DLQ_RETRY_LATENCY is not None:
        DLQ_RETRY_LATENCY.labels(platform=platform).observe(latency_seconds)


def record_webhook_delivery(platform: str, success: bool, latency_ms: float) -> None:
    """Record webhook delivery attempt."""
    _ensure_initialized()

    if WEBHOOK_DELIVERY_TOTAL is not None:
        status = "success" if success else "error"
        WEBHOOK_DELIVERY_TOTAL.labels(platform=platform, status=status).inc()

    if WEBHOOK_DELIVERY_LATENCY is not None:
        WEBHOOK_DELIVERY_LATENCY.labels(platform=platform).observe(latency_ms)


def record_webhook_retry(platform: str, attempt: int) -> None:
    """Record webhook retry attempt."""
    _ensure_initialized()

    if WEBHOOK_RETRY_TOTAL is not None:
        WEBHOOK_RETRY_TOTAL.labels(platform=platform, attempt=str(attempt)).inc()


def record_bot_command(
    platform: str,
    command: str,
    success: bool,
    latency_ms: float,
    timeout: bool = False,
) -> None:
    """Record bot command processing."""
    _ensure_initialized()

    if BOT_COMMAND_TOTAL is not None:
        status = "success" if success else ("timeout" if timeout else "error")
        BOT_COMMAND_TOTAL.labels(platform=platform, command=command, status=status).inc()

    if BOT_COMMAND_LATENCY is not None:
        BOT_COMMAND_LATENCY.labels(platform=platform, command=command).observe(latency_ms)

    if timeout and BOT_COMMAND_TIMEOUT_TOTAL is not None:
        BOT_COMMAND_TIMEOUT_TOTAL.labels(platform=platform, command=command).inc()


# =============================================================================
# Helper Functions (for use with existing metrics infrastructure)
# =============================================================================


def counter_inc(name: str, labels: dict[str, str], value: float = 1.0) -> None:
    """Generic counter increment (for backward compatibility)."""
    _ensure_initialized()

    # Map to specific counters based on name
    if name == "aragora_platform_requests_total" and PLATFORM_REQUESTS_TOTAL is not None:
        PLATFORM_REQUESTS_TOTAL.labels(**labels).inc(value)
    elif name == "aragora_platform_errors_total" and PLATFORM_ERRORS_TOTAL is not None:
        PLATFORM_ERRORS_TOTAL.labels(**labels).inc(value)
    # Add more mappings as needed


def histogram_observe(name: str, value: float, labels: dict[str, str]) -> None:
    """Generic histogram observation (for backward compatibility)."""
    _ensure_initialized()

    if name == "aragora_platform_latency_ms" and PLATFORM_REQUEST_LATENCY is not None:
        PLATFORM_REQUEST_LATENCY.labels(**labels).observe(value)


__all__ = [
    # Initialization
    "_initialize_platform_metrics",
    # Recording functions
    "record_platform_request",
    "record_circuit_state",
    "record_rate_limit",
    "record_dlq_enqueue",
    "record_dlq_processed",
    "record_dlq_failed",
    "update_dlq_pending",
    "record_dlq_retry_latency",
    "record_webhook_delivery",
    "record_webhook_retry",
    "record_bot_command",
    # Generic helpers
    "counter_inc",
    "histogram_observe",
]
