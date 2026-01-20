"""
Store metrics for Aragora.

Provides metrics for:
- Persistent Task Queue
- Governance Store
- User ID Mapping
- Checkpoint Store
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Any, Generator

from aragora.observability.config import get_metrics_config
from aragora.observability.metrics.base import NoOpMetric

logger = logging.getLogger(__name__)

# Module-level initialization state
_initialized = False

# Task Queue metrics
TASK_QUEUE_OPERATIONS_TOTAL: Any = None
TASK_QUEUE_OPERATION_LATENCY: Any = None
TASK_QUEUE_SIZE: Any = None
TASK_QUEUE_RECOVERED_TOTAL: Any = None
TASK_QUEUE_CLEANUP_TOTAL: Any = None

# Governance metrics
GOVERNANCE_DECISIONS_TOTAL: Any = None
GOVERNANCE_VERIFICATIONS_TOTAL: Any = None
GOVERNANCE_APPROVALS_TOTAL: Any = None
GOVERNANCE_STORE_LATENCY: Any = None
GOVERNANCE_ARTIFACTS_ACTIVE: Any = None

# User mapping metrics
USER_MAPPING_OPERATIONS_TOTAL: Any = None
USER_MAPPING_CACHE_HITS_TOTAL: Any = None
USER_MAPPING_CACHE_MISSES_TOTAL: Any = None
USER_MAPPINGS_ACTIVE: Any = None

# Checkpoint metrics
CHECKPOINT_OPERATIONS: Any = None
CHECKPOINT_OPERATION_LATENCY: Any = None
CHECKPOINT_SIZE: Any = None
CHECKPOINT_RESTORE_RESULTS: Any = None


def init_store_metrics() -> bool:
    """Initialize store Prometheus metrics."""
    global _initialized
    global TASK_QUEUE_OPERATIONS_TOTAL, TASK_QUEUE_OPERATION_LATENCY
    global TASK_QUEUE_SIZE, TASK_QUEUE_RECOVERED_TOTAL, TASK_QUEUE_CLEANUP_TOTAL
    global GOVERNANCE_DECISIONS_TOTAL, GOVERNANCE_VERIFICATIONS_TOTAL
    global GOVERNANCE_APPROVALS_TOTAL, GOVERNANCE_STORE_LATENCY
    global GOVERNANCE_ARTIFACTS_ACTIVE
    global USER_MAPPING_OPERATIONS_TOTAL, USER_MAPPING_CACHE_HITS_TOTAL
    global USER_MAPPING_CACHE_MISSES_TOTAL, USER_MAPPINGS_ACTIVE
    global CHECKPOINT_OPERATIONS, CHECKPOINT_OPERATION_LATENCY
    global CHECKPOINT_SIZE, CHECKPOINT_RESTORE_RESULTS

    if _initialized:
        return True

    config = get_metrics_config()
    if not config.enabled:
        _init_noop_metrics()
        _initialized = True
        return False

    try:
        from prometheus_client import Counter, Gauge, Histogram

        # Task Queue metrics
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
            "Current number of tasks in the queue",
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

        # Governance metrics
        GOVERNANCE_DECISIONS_TOTAL = Counter(
            "aragora_governance_decisions_total",
            "Total governance decisions stored",
            ["decision_type", "outcome"],
        )
        GOVERNANCE_VERIFICATIONS_TOTAL = Counter(
            "aragora_governance_verifications_total",
            "Total verifications stored",
            ["verification_type", "result"],
        )
        GOVERNANCE_APPROVALS_TOTAL = Counter(
            "aragora_governance_approvals_total",
            "Total approvals stored",
            ["approval_type", "status"],
        )
        GOVERNANCE_STORE_LATENCY = Histogram(
            "aragora_governance_store_latency_seconds",
            "Governance store operation latency in seconds",
            ["operation"],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5],
        )
        GOVERNANCE_ARTIFACTS_ACTIVE = Gauge(
            "aragora_governance_artifacts_active",
            "Current number of active governance artifacts",
            ["artifact_type"],
        )

        # User mapping metrics
        USER_MAPPING_OPERATIONS_TOTAL = Counter(
            "aragora_user_mapping_operations_total",
            "Total user ID mapping operations",
            ["operation", "platform", "status"],
        )
        USER_MAPPING_CACHE_HITS_TOTAL = Counter(
            "aragora_user_mapping_cache_hits_total",
            "User ID mapping cache hits",
            ["platform"],
        )
        USER_MAPPING_CACHE_MISSES_TOTAL = Counter(
            "aragora_user_mapping_cache_misses_total",
            "User ID mapping cache misses",
            ["platform"],
        )
        USER_MAPPINGS_ACTIVE = Gauge(
            "aragora_user_mappings_active",
            "Number of active user ID mappings",
            ["platform"],
        )

        # Checkpoint metrics
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
        logger.debug("Store metrics initialized")
        return True

    except ImportError:
        _init_noop_metrics()
        _initialized = True
        return False


def _init_noop_metrics() -> None:
    """Initialize no-op metrics when Prometheus is disabled."""
    global TASK_QUEUE_OPERATIONS_TOTAL, TASK_QUEUE_OPERATION_LATENCY
    global TASK_QUEUE_SIZE, TASK_QUEUE_RECOVERED_TOTAL, TASK_QUEUE_CLEANUP_TOTAL
    global GOVERNANCE_DECISIONS_TOTAL, GOVERNANCE_VERIFICATIONS_TOTAL
    global GOVERNANCE_APPROVALS_TOTAL, GOVERNANCE_STORE_LATENCY
    global GOVERNANCE_ARTIFACTS_ACTIVE
    global USER_MAPPING_OPERATIONS_TOTAL, USER_MAPPING_CACHE_HITS_TOTAL
    global USER_MAPPING_CACHE_MISSES_TOTAL, USER_MAPPINGS_ACTIVE
    global CHECKPOINT_OPERATIONS, CHECKPOINT_OPERATION_LATENCY
    global CHECKPOINT_SIZE, CHECKPOINT_RESTORE_RESULTS

    noop = NoOpMetric()
    TASK_QUEUE_OPERATIONS_TOTAL = noop
    TASK_QUEUE_OPERATION_LATENCY = noop
    TASK_QUEUE_SIZE = noop
    TASK_QUEUE_RECOVERED_TOTAL = noop
    TASK_QUEUE_CLEANUP_TOTAL = noop
    GOVERNANCE_DECISIONS_TOTAL = noop
    GOVERNANCE_VERIFICATIONS_TOTAL = noop
    GOVERNANCE_APPROVALS_TOTAL = noop
    GOVERNANCE_STORE_LATENCY = noop
    GOVERNANCE_ARTIFACTS_ACTIVE = noop
    USER_MAPPING_OPERATIONS_TOTAL = noop
    USER_MAPPING_CACHE_HITS_TOTAL = noop
    USER_MAPPING_CACHE_MISSES_TOTAL = noop
    USER_MAPPINGS_ACTIVE = noop
    CHECKPOINT_OPERATIONS = noop
    CHECKPOINT_OPERATION_LATENCY = noop
    CHECKPOINT_SIZE = noop
    CHECKPOINT_RESTORE_RESULTS = noop


def _ensure_init() -> None:
    """Ensure metrics are initialized."""
    if not _initialized:
        init_store_metrics()


# =============================================================================
# Task Queue Functions
# =============================================================================


def record_task_queue_operation(operation: str, success: bool, latency_seconds: float) -> None:
    """Record a task queue operation."""
    _ensure_init()
    status = "success" if success else "error"
    TASK_QUEUE_OPERATIONS_TOTAL.labels(operation=operation, status=status).inc()
    TASK_QUEUE_OPERATION_LATENCY.labels(operation=operation).observe(latency_seconds)


def set_task_queue_size(pending: int, ready: int, running: int) -> None:
    """Set the current task queue sizes by status."""
    _ensure_init()
    TASK_QUEUE_SIZE.labels(status="pending").set(pending)
    TASK_QUEUE_SIZE.labels(status="ready").set(ready)
    TASK_QUEUE_SIZE.labels(status="running").set(running)


def record_task_queue_recovery(original_status: str, count: int = 1) -> None:
    """Record recovered tasks on startup."""
    _ensure_init()
    TASK_QUEUE_RECOVERED_TOTAL.labels(original_status=original_status).inc(count)


def record_task_queue_cleanup(count: int) -> None:
    """Record completed tasks cleaned up."""
    _ensure_init()
    TASK_QUEUE_CLEANUP_TOTAL.inc(count)


@contextmanager
def track_task_queue_operation(operation: str) -> Generator[None, None, None]:
    """Context manager to track task queue operations."""
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


# =============================================================================
# Governance Functions
# =============================================================================


def record_governance_decision(decision_type: str, outcome: str) -> None:
    """Record a governance decision stored."""
    _ensure_init()
    GOVERNANCE_DECISIONS_TOTAL.labels(decision_type=decision_type, outcome=outcome).inc()


def record_governance_verification(verification_type: str, result: str) -> None:
    """Record a verification stored."""
    _ensure_init()
    GOVERNANCE_VERIFICATIONS_TOTAL.labels(verification_type=verification_type, result=result).inc()


def record_governance_approval(approval_type: str, status: str) -> None:
    """Record an approval stored."""
    _ensure_init()
    GOVERNANCE_APPROVALS_TOTAL.labels(approval_type=approval_type, status=status).inc()


def record_governance_store_latency(operation: str, latency_seconds: float) -> None:
    """Record governance store operation latency."""
    _ensure_init()
    GOVERNANCE_STORE_LATENCY.labels(operation=operation).observe(latency_seconds)


def set_governance_artifacts_active(decisions: int, verifications: int, approvals: int) -> None:
    """Set the current number of active governance artifacts."""
    _ensure_init()
    GOVERNANCE_ARTIFACTS_ACTIVE.labels(artifact_type="decision").set(decisions)
    GOVERNANCE_ARTIFACTS_ACTIVE.labels(artifact_type="verification").set(verifications)
    GOVERNANCE_ARTIFACTS_ACTIVE.labels(artifact_type="approval").set(approvals)


@contextmanager
def track_governance_store_operation(operation: str) -> Generator[None, None, None]:
    """Context manager to track governance store operations."""
    _ensure_init()
    start = time.perf_counter()
    try:
        yield
    finally:
        latency = time.perf_counter() - start
        record_governance_store_latency(operation, latency)


# =============================================================================
# User Mapping Functions
# =============================================================================


def record_user_mapping_operation(operation: str, platform: str, found: bool) -> None:
    """Record a user ID mapping operation."""
    _ensure_init()
    status = "success" if found else "not_found"
    USER_MAPPING_OPERATIONS_TOTAL.labels(operation=operation, platform=platform, status=status).inc()


def record_user_mapping_cache_hit(platform: str) -> None:
    """Record a user ID mapping cache hit."""
    _ensure_init()
    USER_MAPPING_CACHE_HITS_TOTAL.labels(platform=platform).inc()


def record_user_mapping_cache_miss(platform: str) -> None:
    """Record a user ID mapping cache miss."""
    _ensure_init()
    USER_MAPPING_CACHE_MISSES_TOTAL.labels(platform=platform).inc()


def set_user_mappings_active(platform: str, count: int) -> None:
    """Set the number of active user ID mappings for a platform."""
    _ensure_init()
    USER_MAPPINGS_ACTIVE.labels(platform=platform).set(count)


# =============================================================================
# Checkpoint Functions
# =============================================================================


def record_checkpoint_operation(
    operation: str,
    success: bool,
    latency_seconds: float,
    size_bytes: int = 0,
) -> None:
    """Record a checkpoint operation."""
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
    """Record checkpoint restore results."""
    _ensure_init()
    if nodes_restored > 0:
        CHECKPOINT_RESTORE_RESULTS.labels(result="nodes_restored").inc(nodes_restored)
    if nodes_skipped > 0:
        CHECKPOINT_RESTORE_RESULTS.labels(result="nodes_skipped").inc(nodes_skipped)
    if errors > 0:
        CHECKPOINT_RESTORE_RESULTS.labels(result="errors").inc(errors)


@contextmanager
def track_checkpoint_operation(operation: str) -> Generator[dict, None, None]:
    """Context manager to track checkpoint operations."""
    _ensure_init()
    start = time.perf_counter()
    ctx: dict[str, Any] = {"size_bytes": 0}
    success = True
    try:
        yield ctx
    except Exception:
        success = False
        raise
    finally:
        latency = time.perf_counter() - start
        record_checkpoint_operation(operation, success, latency, ctx.get("size_bytes", 0))
