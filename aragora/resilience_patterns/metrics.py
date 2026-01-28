"""
Prometheus metrics for resilience patterns.

This module provides Prometheus-compatible metrics for monitoring:
- Circuit breaker state changes
- Retry attempts and delays
- Timeout occurrences
- Health status changes

Metrics follow the lazy initialization pattern used in the codebase to avoid
startup overhead when metrics aren't needed.

Usage:
    from aragora.resilience_patterns.metrics import (
        circuit_breaker_state_changed,
        retry_attempt,
        timeout_occurred,
        health_status_changed,
    )

    # Use as callbacks in resilience pattern configs
    config = CircuitBreakerConfig(
        on_state_change=circuit_breaker_state_changed
    )
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

# Lazy metric storage
_metrics: Dict[str, Any] = {}
_prometheus_available: Optional[bool] = None


def _check_prometheus() -> bool:
    """Check if prometheus_client is available."""
    global _prometheus_available
    if _prometheus_available is None:
        try:
            import prometheus_client  # noqa: F401

            _prometheus_available = True
        except ImportError:
            _prometheus_available = False
            logger.debug("prometheus_client not available, metrics will be no-ops")
    return _prometheus_available


def _get_or_create_metric(
    name: str,
    metric_type: type,
    description: str,
    labels: Optional[list[str]] = None,
) -> Any:
    """Get or create a metric with lazy initialization.

    Args:
        name: Metric name
        metric_type: Counter, Gauge, or Histogram
        description: Metric description
        labels: Optional list of label names

    Returns:
        The metric instance, or None if prometheus_client unavailable
    """
    if not _check_prometheus():
        return None

    if name not in _metrics:
        if labels:
            _metrics[name] = metric_type(name, description, labels)
        else:
            _metrics[name] = metric_type(name, description)
    return _metrics[name]


def circuit_breaker_state_changed(
    name: str,
    old_state: Any,
    new_state: Any,
) -> None:
    """Record a circuit breaker state transition.

    This callback signature matches CircuitBreakerConfig.on_state_change.

    Args:
        name: Circuit breaker name
        old_state: Previous state (CircuitState or str)
        new_state: New state (CircuitState or str)
    """
    if not _check_prometheus():
        return

    from prometheus_client import Counter

    counter = _get_or_create_metric(
        "resilience_circuit_breaker_state_changes_total",
        Counter,
        "Circuit breaker state transitions",
        ["breaker_name", "from_state", "to_state"],
    )
    if counter:
        old_val = old_state.value if hasattr(old_state, "value") else str(old_state)
        new_val = new_state.value if hasattr(new_state, "value") else str(new_state)
        counter.labels(breaker_name=name, from_state=old_val, to_state=new_val).inc()


def retry_attempt(
    operation_name: str,
    attempt: int,
    delay: float,
    exception: Optional[Exception] = None,
) -> None:
    """Record a retry attempt.

    Args:
        operation_name: Name of the operation being retried
        attempt: Attempt number (1-indexed)
        delay: Delay before this retry in seconds
        exception: The exception that triggered the retry (optional)
    """
    if not _check_prometheus():
        return

    from prometheus_client import Counter, Histogram

    # Count retry attempts
    counter = _get_or_create_metric(
        "resilience_retry_attempts_total",
        Counter,
        "Total retry attempts",
        ["operation_name"],
    )
    if counter:
        counter.labels(operation_name=operation_name).inc()

    # Track retry delay distribution
    histogram = _get_or_create_metric(
        "resilience_retry_delay_seconds",
        Histogram,
        "Retry delay distribution in seconds",
        ["operation_name"],
    )
    if histogram:
        histogram.labels(operation_name=operation_name).observe(delay)


def retry_exhausted(
    operation_name: str,
    total_attempts: int,
    last_exception: Optional[Exception] = None,
) -> None:
    """Record when all retry attempts are exhausted.

    Args:
        operation_name: Name of the operation that failed
        total_attempts: Total number of attempts made
        last_exception: The final exception (optional)
    """
    if not _check_prometheus():
        return

    from prometheus_client import Counter

    counter = _get_or_create_metric(
        "resilience_retry_exhausted_total",
        Counter,
        "Operations that exhausted all retry attempts",
        ["operation_name"],
    )
    if counter:
        counter.labels(operation_name=operation_name).inc()


def timeout_occurred(
    operation_name: str,
    timeout_seconds: float,
) -> None:
    """Record a timeout occurrence.

    Args:
        operation_name: Name of the operation that timed out
        timeout_seconds: The timeout value that was exceeded
    """
    if not _check_prometheus():
        return

    from prometheus_client import Counter, Histogram

    # Count timeouts
    counter = _get_or_create_metric(
        "resilience_timeouts_total",
        Counter,
        "Total timeout occurrences",
        ["operation_name"],
    )
    if counter:
        counter.labels(operation_name=operation_name).inc()

    # Track timeout values
    histogram = _get_or_create_metric(
        "resilience_timeout_seconds",
        Histogram,
        "Timeout value distribution in seconds",
        ["operation_name"],
    )
    if histogram:
        histogram.labels(operation_name=operation_name).observe(timeout_seconds)


def health_status_changed(
    component: str,
    healthy: bool,
    consecutive_failures: int = 0,
) -> None:
    """Record a health status change.

    Args:
        component: Component name
        healthy: Whether the component is now healthy
        consecutive_failures: Number of consecutive failures
    """
    if not _check_prometheus():
        return

    from prometheus_client import Gauge

    # Health status gauge (1 = healthy, 0 = unhealthy)
    gauge = _get_or_create_metric(
        "resilience_health_status",
        Gauge,
        "Component health status (1=healthy, 0=unhealthy)",
        ["component_name"],
    )
    if gauge:
        gauge.labels(component_name=component).set(1 if healthy else 0)

    # Consecutive failures gauge
    failures_gauge = _get_or_create_metric(
        "resilience_health_consecutive_failures",
        Gauge,
        "Consecutive failures for a component",
        ["component_name"],
    )
    if failures_gauge:
        failures_gauge.labels(component_name=component).set(consecutive_failures)


def operation_duration(
    operation_name: str,
    duration_seconds: float,
    success: bool = True,
) -> None:
    """Record operation duration for tracking latency.

    Args:
        operation_name: Name of the operation
        duration_seconds: How long the operation took
        success: Whether the operation succeeded
    """
    if not _check_prometheus():
        return

    from prometheus_client import Histogram

    histogram = _get_or_create_metric(
        "resilience_operation_duration_seconds",
        Histogram,
        "Operation duration in seconds",
        ["operation_name", "success"],
    )
    if histogram:
        histogram.labels(
            operation_name=operation_name,
            success=str(success).lower(),
        ).observe(duration_seconds)


def create_metrics_callbacks() -> Dict[str, Callable]:
    """Create a dictionary of metrics callbacks for decorator integration.

    Returns:
        Dictionary mapping callback names to functions
    """
    return {
        "on_circuit_state_change": circuit_breaker_state_changed,
        "on_retry": retry_attempt,
        "on_retry_exhausted": retry_exhausted,
        "on_timeout": timeout_occurred,
        "on_health_change": health_status_changed,
        "on_operation_complete": operation_duration,
    }


def reset_metrics() -> None:
    """Reset all metrics (useful for testing)."""
    global _metrics
    _metrics.clear()


__all__ = [
    "circuit_breaker_state_changed",
    "retry_attempt",
    "retry_exhausted",
    "timeout_occurred",
    "health_status_changed",
    "operation_duration",
    "create_metrics_callbacks",
    "reset_metrics",
]
