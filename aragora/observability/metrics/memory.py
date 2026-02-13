"""
Memory operation metrics.

Provides Prometheus metrics for tracking memory operations including:
- Multi-tier memory operations (fast/medium/slow/glacial)
- Memory coordinator atomic writes
- Adaptive round changes from memory strategy
"""

from __future__ import annotations

import logging
from typing import Any

from aragora.observability.metrics.base import NoOpMetric, get_metrics_enabled

logger = logging.getLogger(__name__)

# Memory operation metrics
MEMORY_OPERATIONS: Any = None
MEMORY_COORDINATOR_WRITES: Any = None
ADAPTIVE_ROUND_CHANGES: Any = None

_initialized = False


def init_memory_metrics() -> None:
    """Initialize memory operation metrics."""
    global _initialized
    global MEMORY_OPERATIONS, MEMORY_COORDINATOR_WRITES, ADAPTIVE_ROUND_CHANGES

    if _initialized:
        return

    if not get_metrics_enabled():
        _init_noop_metrics()
        _initialized = True
        return

    try:
        from prometheus_client import Counter

        MEMORY_OPERATIONS = Counter(
            "aragora_memory_operations_total",
            "Total memory operations",
            ["operation", "tier"],
        )

        MEMORY_COORDINATOR_WRITES = Counter(
            "aragora_memory_coordinator_writes_total",
            "Atomic memory coordinator writes",
            ["status"],
        )

        ADAPTIVE_ROUND_CHANGES = Counter(
            "aragora_adaptive_round_changes_total",
            "Debate round count adjustments from memory strategy",
            ["direction"],
        )

        _initialized = True
        logger.debug("Memory metrics initialized")

    except (ImportError, ValueError):
        _init_noop_metrics()
        _initialized = True
    except Exception as e:
        logger.warning(f"Failed to initialize memory metrics: {e}")
        _init_noop_metrics()
        _initialized = True


def _init_noop_metrics() -> None:
    """Initialize no-op metrics when Prometheus is disabled."""
    global MEMORY_OPERATIONS, MEMORY_COORDINATOR_WRITES, ADAPTIVE_ROUND_CHANGES

    noop = NoOpMetric()
    MEMORY_OPERATIONS = noop
    MEMORY_COORDINATOR_WRITES = noop
    ADAPTIVE_ROUND_CHANGES = noop


def _ensure_init() -> None:
    """Ensure metrics are initialized."""
    if not _initialized:
        init_memory_metrics()


# =============================================================================
# Recording Functions
# =============================================================================


def record_memory_operation(operation: str, tier: str) -> None:
    """Record a memory operation.

    Args:
        operation: Type of operation (read, write, delete, flush)
        tier: Memory tier (fast, medium, slow, glacial)
    """
    _ensure_init()
    MEMORY_OPERATIONS.labels(operation=operation, tier=tier).inc()


def record_memory_coordinator_write(success: bool) -> None:
    """Record a memory coordinator atomic write.

    The memory coordinator ensures atomic writes across multiple
    memory subsystems (CritiqueStore, ContinuumMemory, etc.)

    Args:
        success: Whether the atomic write succeeded
    """
    _ensure_init()
    status = "success" if success else "error"
    MEMORY_COORDINATOR_WRITES.labels(status=status).inc()


def record_adaptive_round_change(direction: str) -> None:
    """Record an adaptive round change.

    Adaptive rounds adjust debate round counts based on memory
    strategy feedback (e.g., convergence patterns from previous debates).

    Args:
        direction: Direction of change (increase, decrease, unchanged)
    """
    _ensure_init()
    ADAPTIVE_ROUND_CHANGES.labels(direction=direction).inc()


__all__ = [
    # Metrics
    "MEMORY_OPERATIONS",
    "MEMORY_COORDINATOR_WRITES",
    "ADAPTIVE_ROUND_CHANGES",
    # Functions
    "init_memory_metrics",
    "record_memory_operation",
    "record_memory_coordinator_write",
    "record_adaptive_round_change",
]
