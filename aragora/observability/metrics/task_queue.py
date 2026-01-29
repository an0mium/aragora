"""
Persistent task queue metrics.

Provides Prometheus metrics for tracking task queue operations,
including enqueue/dequeue latency, queue sizes, and recovery.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Any, Generator

from aragora.observability.metrics.base import NoOpMetric, get_metrics_enabled

logger = logging.getLogger(__name__)

# Global metric variables
TASK_QUEUE_OPERATIONS_TOTAL: Any = None
TASK_QUEUE_OPERATION_LATENCY: Any = None
TASK_QUEUE_SIZE: Any = None
TASK_QUEUE_RECOVERED_TOTAL: Any = None
TASK_QUEUE_CLEANUP_TOTAL: Any = None

_initialized = False


def init_task_queue_metrics() -> None:
    """Initialize task queue metrics."""
    global _initialized
    global TASK_QUEUE_OPERATIONS_TOTAL, TASK_QUEUE_OPERATION_LATENCY
    global TASK_QUEUE_SIZE, TASK_QUEUE_RECOVERED_TOTAL, TASK_QUEUE_CLEANUP_TOTAL

    if _initialized:
        return

    if not get_metrics_enabled():
        _init_noop_metrics()
        _initialized = True
        return

    try:
        from prometheus_client import Counter, Gauge, Histogram

        TASK_QUEUE_OPERATIONS_TOTAL = Counter(
            "aragora_task_queue_operations_total",
            "Total task queue operations",
            ["operation", "status"],
        )

        TASK_QUEUE_OPERATION_LATENCY = Histogram(
            "aragora_task_queue_operation_latency_seconds",
            "Task queue operation latency in seconds",
            ["operation"],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
        )

        TASK_QUEUE_SIZE = Gauge(
            "aragora_task_queue_size",
            "Current task queue size by status",
            ["status"],
        )

        TASK_QUEUE_RECOVERED_TOTAL = Counter(
            "aragora_task_queue_recovered_total",
            "Total tasks recovered on startup",
            ["original_status"],
        )

        TASK_QUEUE_CLEANUP_TOTAL = Counter(
            "aragora_task_queue_cleanup_total",
            "Total completed tasks cleaned up",
        )

        _initialized = True
        logger.debug("Task queue metrics initialized")

    except ImportError:
        _init_noop_metrics()
        _initialized = True


def _init_noop_metrics() -> None:
    """Initialize no-op metrics when Prometheus is disabled."""
    global TASK_QUEUE_OPERATIONS_TOTAL, TASK_QUEUE_OPERATION_LATENCY
    global TASK_QUEUE_SIZE, TASK_QUEUE_RECOVERED_TOTAL, TASK_QUEUE_CLEANUP_TOTAL

    TASK_QUEUE_OPERATIONS_TOTAL = NoOpMetric()
    TASK_QUEUE_OPERATION_LATENCY = NoOpMetric()
    TASK_QUEUE_SIZE = NoOpMetric()
    TASK_QUEUE_RECOVERED_TOTAL = NoOpMetric()
    TASK_QUEUE_CLEANUP_TOTAL = NoOpMetric()


def _ensure_init() -> None:
    """Ensure metrics are initialized."""
    if not _initialized:
        init_task_queue_metrics()


# =============================================================================
# Recording Functions
# =============================================================================


def record_task_queue_operation(
    operation: str,
    success: bool,
    latency_seconds: float,
) -> None:
    """Record a task queue operation.

    Args:
        operation: Operation type (enqueue, dequeue, complete, fail, cancel)
        success: Whether the operation succeeded
        latency_seconds: Operation latency in seconds
    """
    _ensure_init()
    status = "success" if success else "error"
    TASK_QUEUE_OPERATIONS_TOTAL.labels(operation=operation, status=status).inc()
    TASK_QUEUE_OPERATION_LATENCY.labels(operation=operation).observe(latency_seconds)


def set_task_queue_size(pending: int, ready: int, running: int) -> None:
    """Set the current task queue sizes by status.

    Args:
        pending: Number of pending tasks
        ready: Number of ready tasks
        running: Number of running tasks
    """
    _ensure_init()
    TASK_QUEUE_SIZE.labels(status="pending").set(pending)
    TASK_QUEUE_SIZE.labels(status="ready").set(ready)
    TASK_QUEUE_SIZE.labels(status="running").set(running)


def record_task_queue_recovery(original_status: str, count: int = 1) -> None:
    """Record recovered tasks on startup.

    Args:
        original_status: Original status of recovered task (pending, ready, running)
        count: Number of tasks recovered
    """
    _ensure_init()
    TASK_QUEUE_RECOVERED_TOTAL.labels(original_status=original_status).inc(count)


def record_task_queue_cleanup(count: int) -> None:
    """Record completed tasks cleaned up.

    Args:
        count: Number of tasks cleaned up
    """
    _ensure_init()
    TASK_QUEUE_CLEANUP_TOTAL.inc(count)


@contextmanager
def track_task_queue_operation(operation: str) -> Generator[None, None, None]:
    """Context manager to track task queue operations.

    Automatically records latency and success/failure.

    Args:
        operation: Operation type (enqueue, dequeue, complete, fail)

    Example:
        with track_task_queue_operation("enqueue"):
            await queue.enqueue(task)
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
        record_task_queue_operation(operation, success, latency)


__all__ = [
    # Metrics
    "TASK_QUEUE_OPERATIONS_TOTAL",
    "TASK_QUEUE_OPERATION_LATENCY",
    "TASK_QUEUE_SIZE",
    "TASK_QUEUE_RECOVERED_TOTAL",
    "TASK_QUEUE_CLEANUP_TOTAL",
    # Functions
    "init_task_queue_metrics",
    "record_task_queue_operation",
    "set_task_queue_size",
    "record_task_queue_recovery",
    "record_task_queue_cleanup",
    "track_task_queue_operation",
]
