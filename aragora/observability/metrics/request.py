"""
HTTP request metrics.

Provides Prometheus metrics for tracking HTTP request counts,
latency, and endpoint performance.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import time
from typing import Any, Callable, TypeVar

from aragora.observability.metrics.base import (
    NoOpMetric,
    get_metrics_enabled,
    normalize_endpoint,
)

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

# Global metric variables
REQUEST_COUNT: Any = None
REQUEST_LATENCY: Any = None

_initialized = False


def init_request_metrics() -> None:
    """Initialize request metrics."""
    global _initialized
    global REQUEST_COUNT, REQUEST_LATENCY

    if _initialized:
        return

    if not get_metrics_enabled():
        _init_noop_metrics()
        _initialized = True
        return

    try:
        from prometheus_client import Counter, Histogram

        REQUEST_COUNT = Counter(
            "aragora_http_requests_total",
            "Total HTTP requests",
            ["method", "endpoint", "status"],
        )

        REQUEST_LATENCY = Histogram(
            "aragora_http_request_latency_seconds",
            "HTTP request latency",
            ["endpoint"],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
        )

        _initialized = True
        logger.debug("Request metrics initialized")

    except ImportError:
        _init_noop_metrics()
        _initialized = True


def _init_noop_metrics() -> None:
    """Initialize no-op metrics when Prometheus is disabled."""
    global REQUEST_COUNT, REQUEST_LATENCY

    REQUEST_COUNT = NoOpMetric()
    REQUEST_LATENCY = NoOpMetric()


def _ensure_init() -> None:
    """Ensure metrics are initialized."""
    if not _initialized:
        init_request_metrics()


# =============================================================================
# Recording Functions
# =============================================================================


def record_request(
    method: str,
    endpoint: str,
    status: int,
    latency: float,
) -> None:
    """Record an HTTP request.

    Args:
        method: HTTP method (GET, POST, etc.)
        endpoint: Request endpoint path
        status: HTTP status code
        latency: Request latency in seconds
    """
    _ensure_init()
    normalized_endpoint = normalize_endpoint(endpoint)
    REQUEST_COUNT.labels(
        method=method,
        endpoint=normalized_endpoint,
        status=str(status),
    ).inc()
    REQUEST_LATENCY.labels(endpoint=normalized_endpoint).observe(latency)


def record_latency(endpoint: str, latency_seconds: float) -> None:
    """Record endpoint latency.

    Args:
        endpoint: Endpoint name or path
        latency_seconds: Latency in seconds
    """
    _ensure_init()
    REQUEST_LATENCY.labels(endpoint=endpoint).observe(latency_seconds)


# =============================================================================
# Decorators
# =============================================================================


def measure_latency(metric_name: str = "request") -> Callable[[F], F]:
    """Decorator to measure function execution latency.

    Args:
        metric_name: Label for the metric

    Example:
        @measure_latency("health_check")
        def health_check():
            return {"status": "ok"}
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            _ensure_init()
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                latency = time.perf_counter() - start
                REQUEST_LATENCY.labels(endpoint=metric_name).observe(latency)

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            _ensure_init()
            start = time.perf_counter()
            try:
                return await func(*args, **kwargs)
            finally:
                latency = time.perf_counter() - start
                REQUEST_LATENCY.labels(endpoint=metric_name).observe(latency)

        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore[return-value]
        return sync_wrapper  # type: ignore[return-value]

    return decorator


def measure_async_latency(metric_name: str = "request") -> Callable[[F], F]:
    """Decorator to measure async function execution latency.

    Alias for measure_latency that makes async intent explicit.

    Args:
        metric_name: Label for the metric

    Example:
        @measure_async_latency("async_operation")
        async def async_operation():
            await asyncio.sleep(0.1)
    """
    return measure_latency(metric_name)


__all__ = [
    # Metrics
    "REQUEST_COUNT",
    "REQUEST_LATENCY",
    # Functions
    "init_request_metrics",
    "record_request",
    "record_latency",
    "measure_latency",
    "measure_async_latency",
]
