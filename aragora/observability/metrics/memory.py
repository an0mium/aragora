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

# Titans sweep metrics
SWEEP_ITEMS_PROCESSED: Any = None
SWEEP_ACTIONS: Any = None
SWEEP_ERRORS: Any = None
SWEEP_DURATION: Any = None
TRIGGER_FIRES: Any = None

_initialized = False


def init_memory_metrics() -> None:
    """Initialize memory operation metrics."""
    global _initialized
    global MEMORY_OPERATIONS, MEMORY_COORDINATOR_WRITES, ADAPTIVE_ROUND_CHANGES
    global SWEEP_ITEMS_PROCESSED, SWEEP_ACTIONS, SWEEP_ERRORS, SWEEP_DURATION, TRIGGER_FIRES

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

        SWEEP_ITEMS_PROCESSED = Counter(
            "aragora_memory_sweep_items_total",
            "Total items processed by Titans memory sweep",
        )

        SWEEP_ACTIONS = Counter(
            "aragora_memory_sweep_actions_total",
            "Sweep actions executed by RetentionGate",
            ["action"],
        )

        SWEEP_ERRORS = Counter(
            "aragora_memory_sweep_errors_total",
            "Errors during memory sweep processing",
        )

        from prometheus_client import Histogram

        SWEEP_DURATION = Histogram(
            "aragora_memory_sweep_duration_seconds",
            "Duration of a single sweep pass",
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0],
        )

        TRIGGER_FIRES = Counter(
            "aragora_memory_trigger_fires_total",
            "Memory trigger firings",
            ["trigger_name", "success"],
        )

        _initialized = True
        logger.debug("Memory metrics initialized")

    except (ImportError, ValueError):
        _init_noop_metrics()
        _initialized = True
    except (RuntimeError, TypeError) as e:
        logger.warning("Failed to initialize memory metrics: %s", e)
        _init_noop_metrics()
        _initialized = True


def _init_noop_metrics() -> None:
    """Initialize no-op metrics when Prometheus is disabled."""
    global MEMORY_OPERATIONS, MEMORY_COORDINATOR_WRITES, ADAPTIVE_ROUND_CHANGES
    global SWEEP_ITEMS_PROCESSED, SWEEP_ACTIONS, SWEEP_ERRORS, SWEEP_DURATION, TRIGGER_FIRES

    noop = NoOpMetric()
    MEMORY_OPERATIONS = noop
    MEMORY_COORDINATOR_WRITES = noop
    ADAPTIVE_ROUND_CHANGES = noop
    SWEEP_ITEMS_PROCESSED = noop
    SWEEP_ACTIONS = noop
    SWEEP_ERRORS = noop
    SWEEP_DURATION = noop
    TRIGGER_FIRES = noop


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


def record_sweep_result(
    items_processed: int,
    actions: dict[str, int],
    errors: int,
    duration_seconds: float,
) -> None:
    """Record results from a Titans memory sweep pass.

    Args:
        items_processed: Number of items evaluated by RetentionGate.
        actions: Map of action name to count (retain, forget, demote, consolidate).
        errors: Number of errors during sweep.
        duration_seconds: Wall-clock duration of the sweep.
    """
    _ensure_init()
    SWEEP_ITEMS_PROCESSED.inc(items_processed)
    for action, count in actions.items():
        SWEEP_ACTIONS.labels(action=action).inc(count)
    if errors:
        SWEEP_ERRORS.inc(errors)
    SWEEP_DURATION.observe(duration_seconds)


def record_trigger_fire(trigger_name: str, success: bool) -> None:
    """Record a memory trigger firing.

    Args:
        trigger_name: Name of the trigger that fired.
        success: Whether the trigger action succeeded.
    """
    _ensure_init()
    TRIGGER_FIRES.labels(trigger_name=trigger_name, success=str(success)).inc()


__all__ = [
    # Metrics
    "MEMORY_OPERATIONS",
    "MEMORY_COORDINATOR_WRITES",
    "ADAPTIVE_ROUND_CHANGES",
    "SWEEP_ITEMS_PROCESSED",
    "SWEEP_ACTIONS",
    "SWEEP_ERRORS",
    "SWEEP_DURATION",
    "TRIGGER_FIRES",
    # Functions
    "init_memory_metrics",
    "record_memory_operation",
    "record_memory_coordinator_write",
    "record_adaptive_round_change",
    "record_sweep_result",
    "record_trigger_fire",
]
