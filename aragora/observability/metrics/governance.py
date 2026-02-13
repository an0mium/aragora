"""
Governance store metrics.

Provides Prometheus metrics for tracking governance artifacts
including decisions, verifications, and approvals.
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
GOVERNANCE_DECISIONS_TOTAL: Any = None
GOVERNANCE_VERIFICATIONS_TOTAL: Any = None
GOVERNANCE_APPROVALS_TOTAL: Any = None
GOVERNANCE_STORE_LATENCY: Any = None
GOVERNANCE_ARTIFACTS_ACTIVE: Any = None

_initialized = False


def init_governance_metrics() -> None:
    """Initialize governance store metrics."""
    global _initialized
    global GOVERNANCE_DECISIONS_TOTAL, GOVERNANCE_VERIFICATIONS_TOTAL
    global GOVERNANCE_APPROVALS_TOTAL, GOVERNANCE_STORE_LATENCY
    global GOVERNANCE_ARTIFACTS_ACTIVE

    if _initialized:
        return

    if not get_metrics_enabled():
        _init_noop_metrics()
        _initialized = True
        return

    try:
        from prometheus_client import Counter, Gauge, Histogram

        GOVERNANCE_DECISIONS_TOTAL = Counter(
            "aragora_governance_decisions_total",
            "Total governance decisions stored",
            ["decision_type", "outcome"],
        )

        GOVERNANCE_VERIFICATIONS_TOTAL = Counter(
            "aragora_governance_verifications_total",
            "Total governance verifications stored",
            ["verification_type", "result"],
        )

        GOVERNANCE_APPROVALS_TOTAL = Counter(
            "aragora_governance_approvals_total",
            "Total governance approvals stored",
            ["approval_type", "status"],
        )

        GOVERNANCE_STORE_LATENCY = Histogram(
            "aragora_governance_store_latency_seconds",
            "Governance store operation latency in seconds",
            ["operation"],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
        )

        GOVERNANCE_ARTIFACTS_ACTIVE = Gauge(
            "aragora_governance_artifacts_active",
            "Current number of active governance artifacts",
            ["artifact_type"],
        )

        _initialized = True
        logger.debug("Governance metrics initialized")

    except (ImportError, ValueError):
        _init_noop_metrics()
        _initialized = True


def _init_noop_metrics() -> None:
    """Initialize no-op metrics when Prometheus is disabled."""
    global GOVERNANCE_DECISIONS_TOTAL, GOVERNANCE_VERIFICATIONS_TOTAL
    global GOVERNANCE_APPROVALS_TOTAL, GOVERNANCE_STORE_LATENCY
    global GOVERNANCE_ARTIFACTS_ACTIVE

    GOVERNANCE_DECISIONS_TOTAL = NoOpMetric()
    GOVERNANCE_VERIFICATIONS_TOTAL = NoOpMetric()
    GOVERNANCE_APPROVALS_TOTAL = NoOpMetric()
    GOVERNANCE_STORE_LATENCY = NoOpMetric()
    GOVERNANCE_ARTIFACTS_ACTIVE = NoOpMetric()


def _ensure_init() -> None:
    """Ensure metrics are initialized."""
    if not _initialized:
        init_governance_metrics()


# =============================================================================
# Recording Functions
# =============================================================================


def record_governance_decision(decision_type: str, outcome: str) -> None:
    """Record a governance decision stored.

    Args:
        decision_type: Type of decision (manual, auto)
        outcome: Decision outcome (approved, rejected)
    """
    _ensure_init()
    GOVERNANCE_DECISIONS_TOTAL.labels(decision_type=decision_type, outcome=outcome).inc()


def record_governance_verification(verification_type: str, result: str) -> None:
    """Record a verification stored.

    Args:
        verification_type: Type of verification (formal, runtime)
        result: Verification result (valid, invalid)
    """
    _ensure_init()
    GOVERNANCE_VERIFICATIONS_TOTAL.labels(verification_type=verification_type, result=result).inc()


def record_governance_approval(approval_type: str, status: str) -> None:
    """Record an approval stored.

    Args:
        approval_type: Type of approval (nomic, deploy, change)
        status: Approval status (granted, revoked)
    """
    _ensure_init()
    GOVERNANCE_APPROVALS_TOTAL.labels(approval_type=approval_type, status=status).inc()


def record_governance_store_latency(operation: str, latency_seconds: float) -> None:
    """Record governance store operation latency.

    Args:
        operation: Operation type (save, get, list, delete)
        latency_seconds: Operation latency in seconds
    """
    _ensure_init()
    GOVERNANCE_STORE_LATENCY.labels(operation=operation).observe(latency_seconds)


def set_governance_artifacts_active(decisions: int, verifications: int, approvals: int) -> None:
    """Set the current number of active governance artifacts.

    Args:
        decisions: Number of active decisions
        verifications: Number of active verifications
        approvals: Number of active approvals
    """
    _ensure_init()
    GOVERNANCE_ARTIFACTS_ACTIVE.labels(artifact_type="decision").set(decisions)
    GOVERNANCE_ARTIFACTS_ACTIVE.labels(artifact_type="verification").set(verifications)
    GOVERNANCE_ARTIFACTS_ACTIVE.labels(artifact_type="approval").set(approvals)


@contextmanager
def track_governance_store_operation(operation: str) -> Generator[None, None, None]:
    """Context manager to track governance store operations.

    Automatically records latency.

    Args:
        operation: Operation type (save, get, list, delete)

    Example:
        with track_governance_store_operation("save"):
            await store.save_verification(...)
    """
    _ensure_init()
    start = time.perf_counter()
    try:
        yield
    finally:
        latency = time.perf_counter() - start
        record_governance_store_latency(operation, latency)


__all__ = [
    # Metrics
    "GOVERNANCE_DECISIONS_TOTAL",
    "GOVERNANCE_VERIFICATIONS_TOTAL",
    "GOVERNANCE_APPROVALS_TOTAL",
    "GOVERNANCE_STORE_LATENCY",
    "GOVERNANCE_ARTIFACTS_ACTIVE",
    # Functions
    "init_governance_metrics",
    "record_governance_decision",
    "record_governance_verification",
    "record_governance_approval",
    "record_governance_store_latency",
    "set_governance_artifacts_active",
    "track_governance_store_operation",
]
