"""
Checkpoint store metrics.

Provides Prometheus metrics for tracking checkpoint operations
including create, restore, delete, and comparison operations.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Any
from collections.abc import Generator

from aragora.observability.metrics.base import NoOpMetric, get_metrics_enabled

logger = logging.getLogger(__name__)

# Global metric variables
CHECKPOINT_OPERATIONS: Any = None
CHECKPOINT_OPERATION_LATENCY: Any = None
CHECKPOINT_SIZE: Any = None
CHECKPOINT_RESTORE_RESULTS: Any = None

_initialized = False


def init_checkpoint_metrics() -> None:
    """Initialize checkpoint store metrics."""
    global _initialized
    global CHECKPOINT_OPERATIONS, CHECKPOINT_OPERATION_LATENCY
    global CHECKPOINT_SIZE, CHECKPOINT_RESTORE_RESULTS

    if _initialized:
        return

    if not get_metrics_enabled():
        _init_noop_metrics()
        _initialized = True
        return

    try:
        from prometheus_client import Counter, Histogram

        CHECKPOINT_OPERATIONS = Counter(
            "aragora_checkpoint_operations_total",
            "Total checkpoint operations",
            ["operation", "status"],
        )

        CHECKPOINT_OPERATION_LATENCY = Histogram(
            "aragora_checkpoint_operation_latency_seconds",
            "Checkpoint operation latency",
            ["operation"],
            buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0],
        )

        CHECKPOINT_SIZE = Histogram(
            "aragora_checkpoint_size_bytes",
            "Checkpoint file size in bytes",
            buckets=[1000, 10000, 100000, 1000000, 10000000, 100000000],
        )

        CHECKPOINT_RESTORE_RESULTS = Counter(
            "aragora_checkpoint_restore_results_total",
            "Checkpoint restore results",
            ["result"],
        )

        _initialized = True
        logger.debug("Checkpoint metrics initialized")

    except (ImportError, ValueError):
        _init_noop_metrics()
        _initialized = True


def _init_noop_metrics() -> None:
    """Initialize no-op metrics when Prometheus is disabled."""
    global CHECKPOINT_OPERATIONS, CHECKPOINT_OPERATION_LATENCY
    global CHECKPOINT_SIZE, CHECKPOINT_RESTORE_RESULTS

    CHECKPOINT_OPERATIONS = NoOpMetric()
    CHECKPOINT_OPERATION_LATENCY = NoOpMetric()
    CHECKPOINT_SIZE = NoOpMetric()
    CHECKPOINT_RESTORE_RESULTS = NoOpMetric()


def _ensure_init() -> None:
    """Ensure metrics are initialized."""
    if not _initialized:
        init_checkpoint_metrics()


# =============================================================================
# Recording Functions
# =============================================================================


def record_checkpoint_operation(
    operation: str,
    success: bool,
    latency_seconds: float,
    size_bytes: int = 0,
) -> None:
    """Record a checkpoint operation.

    Args:
        operation: Operation type (create, restore, delete, list, compare)
        success: Whether the operation succeeded
        latency_seconds: Operation latency in seconds
        size_bytes: Checkpoint size in bytes (for create operations)
    """
    _ensure_init()
    status = "success" if success else "error"
    CHECKPOINT_OPERATIONS.labels(operation=operation, status=status).inc()
    CHECKPOINT_OPERATION_LATENCY.labels(operation=operation).observe(latency_seconds)
    if size_bytes > 0:
        CHECKPOINT_SIZE.observe(size_bytes)


def record_checkpoint_restore_result(
    nodes_restored: int,
    nodes_skipped: int,
    errors: int,
) -> None:
    """Record checkpoint restore results.

    Args:
        nodes_restored: Number of nodes successfully restored
        nodes_skipped: Number of nodes skipped (duplicates)
        errors: Number of errors during restore
    """
    _ensure_init()
    if nodes_restored > 0:
        CHECKPOINT_RESTORE_RESULTS.labels(result="nodes_restored").inc(nodes_restored)
    if nodes_skipped > 0:
        CHECKPOINT_RESTORE_RESULTS.labels(result="nodes_skipped").inc(nodes_skipped)
    if errors > 0:
        CHECKPOINT_RESTORE_RESULTS.labels(result="errors").inc(errors)


@contextmanager
def track_checkpoint_operation(operation: str) -> Generator[dict, None, None]:
    """Context manager to track checkpoint operations.

    Args:
        operation: Operation type (create, restore, delete, list, compare)

    Example:
        with track_checkpoint_operation("create") as ctx:
            checkpoint = store.create_checkpoint("my_checkpoint")
            ctx["size_bytes"] = checkpoint.size_bytes
    """
    _ensure_init()
    start = time.perf_counter()
    ctx: dict[str, Any] = {"size_bytes": 0}
    success = True
    try:
        yield ctx
    except BaseException:
        success = False
        raise
    finally:
        latency = time.perf_counter() - start
        record_checkpoint_operation(operation, success, latency, ctx.get("size_bytes", 0))


__all__ = [
    # Metrics
    "CHECKPOINT_OPERATIONS",
    "CHECKPOINT_OPERATION_LATENCY",
    "CHECKPOINT_SIZE",
    "CHECKPOINT_RESTORE_RESULTS",
    # Functions
    "init_checkpoint_metrics",
    "record_checkpoint_operation",
    "record_checkpoint_restore_result",
    "track_checkpoint_operation",
]
