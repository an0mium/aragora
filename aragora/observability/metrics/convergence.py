"""
Convergence detection metrics.

Provides Prometheus metrics for tracking semantic convergence detection
in debates, including convergence checks, process evaluation bonuses,
and RLM ready quorum events.
"""

from __future__ import annotations

import logging
from typing import Any

from aragora.observability.metrics.base import NoOpMetric, get_metrics_enabled

logger = logging.getLogger(__name__)

# Convergence detection metrics
CONVERGENCE_CHECKS_TOTAL: Any = None
PROCESS_EVALUATION_BONUSES: Any = None
RLM_READY_QUORUM_EVENTS: Any = None

_initialized = False


def init_convergence_metrics() -> None:
    """Initialize convergence detection metrics."""
    global _initialized
    global CONVERGENCE_CHECKS_TOTAL, PROCESS_EVALUATION_BONUSES
    global RLM_READY_QUORUM_EVENTS

    if _initialized:
        return

    if not get_metrics_enabled():
        _init_noop_metrics()
        _initialized = True
        return

    try:
        from prometheus_client import Counter

        CONVERGENCE_CHECKS_TOTAL = Counter(
            "aragora_convergence_checks_total",
            "Total convergence check events",
            ["status", "blocked"],
        )

        PROCESS_EVALUATION_BONUSES = Counter(
            "aragora_process_evaluation_bonuses_total",
            "Process evaluation vote bonuses applied",
            ["agent"],
        )

        RLM_READY_QUORUM_EVENTS = Counter(
            "aragora_rlm_ready_quorum_total",
            "RLM ready signal quorum events",
        )

        _initialized = True
        logger.debug("Convergence metrics initialized")

    except (ImportError, ValueError):
        _init_noop_metrics()
        _initialized = True
    except Exception as e:
        logger.warning(f"Failed to initialize convergence metrics: {e}")
        _init_noop_metrics()
        _initialized = True


def _init_noop_metrics() -> None:
    """Initialize no-op metrics when Prometheus is disabled."""
    global CONVERGENCE_CHECKS_TOTAL, PROCESS_EVALUATION_BONUSES
    global RLM_READY_QUORUM_EVENTS

    noop = NoOpMetric()
    CONVERGENCE_CHECKS_TOTAL = noop
    PROCESS_EVALUATION_BONUSES = noop
    RLM_READY_QUORUM_EVENTS = noop


def _ensure_init() -> None:
    """Ensure metrics are initialized."""
    if not _initialized:
        init_convergence_metrics()


# =============================================================================
# Recording Functions
# =============================================================================


def record_convergence_check(status: str, blocked: bool = False) -> None:
    """Record a convergence check event.

    Args:
        status: Convergence check status (e.g., converged, diverged, partial)
        blocked: Whether convergence was blocked by external factors
    """
    _ensure_init()
    CONVERGENCE_CHECKS_TOTAL.labels(status=status, blocked=str(blocked)).inc()


def record_process_evaluation_bonus(agent: str) -> None:
    """Record a process evaluation vote bonus.

    Applied when an agent demonstrates good reasoning process
    during debate convergence evaluation.

    Args:
        agent: Agent identifier that received the bonus
    """
    _ensure_init()
    PROCESS_EVALUATION_BONUSES.labels(agent=agent).inc()


def record_rlm_ready_quorum() -> None:
    """Record an RLM ready signal quorum event.

    Called when enough agents signal RLM readiness to
    trigger context consolidation.
    """
    _ensure_init()
    RLM_READY_QUORUM_EVENTS.inc()


__all__ = [
    # Metrics
    "CONVERGENCE_CHECKS_TOTAL",
    "PROCESS_EVALUATION_BONUSES",
    "RLM_READY_QUORUM_EVENTS",
    # Functions
    "init_convergence_metrics",
    "record_convergence_check",
    "record_process_evaluation_bonus",
    "record_rlm_ready_quorum",
]
