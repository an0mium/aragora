"""
Settlement and calibration observability metrics.

Provides Prometheus metrics for tracking epistemic settlement lifecycle,
calibration accuracy, and operator intervention events.
"""

from __future__ import annotations

import logging
from typing import Any

from aragora.observability.metrics.base import NoOpMetric, get_metrics_enabled

logger = logging.getLogger(__name__)

# Settlement metrics
SETTLEMENT_CAPTURED_TOTAL: Any = None
SETTLEMENT_REVIEW_DUE_TOTAL: Any = None
SETTLEMENT_STATUS_TRANSITIONS: Any = None
SETTLEMENT_CONFIDENCE: Any = None
SETTLEMENT_FALSIFIER_COUNT: Any = None

# Calibration metrics
CALIBRATION_BRIER_SCORE: Any = None
CALIBRATION_OUTCOMES_TOTAL: Any = None

# Intervention metrics
INTERVENTION_TOTAL: Any = None
INTERVENTION_PAUSE_DURATION: Any = None

_initialized = False
_CALIBRATION_OUTCOME_COUNTS: dict[str, int] = {}


def init_settlement_metrics() -> None:
    """Initialize settlement, calibration, and intervention metrics."""
    global _initialized
    global SETTLEMENT_CAPTURED_TOTAL, SETTLEMENT_REVIEW_DUE_TOTAL
    global SETTLEMENT_STATUS_TRANSITIONS, SETTLEMENT_CONFIDENCE
    global SETTLEMENT_FALSIFIER_COUNT
    global CALIBRATION_BRIER_SCORE, CALIBRATION_OUTCOMES_TOTAL
    global INTERVENTION_TOTAL, INTERVENTION_PAUSE_DURATION

    if _initialized:
        return

    if not get_metrics_enabled():
        _init_noop_metrics()
        _initialized = True
        return

    try:
        from prometheus_client import Counter, Gauge, Histogram

        SETTLEMENT_CAPTURED_TOTAL = Counter(
            "aragora_settlement_captured_total",
            "Total settlements captured after debates",
            ["status"],
        )

        SETTLEMENT_REVIEW_DUE_TOTAL = Gauge(
            "aragora_settlement_review_due",
            "Number of settlements currently due for review",
        )

        SETTLEMENT_STATUS_TRANSITIONS = Counter(
            "aragora_settlement_status_transitions_total",
            "Settlement status transitions",
            ["from_status", "to_status"],
        )

        SETTLEMENT_CONFIDENCE = Histogram(
            "aragora_settlement_confidence",
            "Distribution of settlement confidence scores",
            buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        )

        SETTLEMENT_FALSIFIER_COUNT = Histogram(
            "aragora_settlement_falsifier_count",
            "Number of falsifiers per settlement",
            buckets=[0, 1, 2, 3, 5, 10, 20],
        )

        CALIBRATION_BRIER_SCORE = Histogram(
            "aragora_calibration_brier_score",
            "Brier score distribution per agent",
            ["agent_name"],
            buckets=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 1.0],
        )

        CALIBRATION_OUTCOMES_TOTAL = Counter(
            "aragora_calibration_outcomes_total",
            "Total calibration outcome recordings",
            ["outcome"],
        )

        INTERVENTION_TOTAL = Counter(
            "aragora_intervention_total",
            "Total operator intervention actions",
            ["action"],
        )

        INTERVENTION_PAUSE_DURATION = Histogram(
            "aragora_intervention_pause_duration_seconds",
            "Duration of debate pauses from operator intervention",
            buckets=[1, 5, 10, 30, 60, 120, 300, 600],
        )

        _initialized = True
        logger.debug("Settlement/calibration/intervention metrics initialized")

    except (ImportError, ValueError):
        _init_noop_metrics()
        _initialized = True


def _init_noop_metrics() -> None:
    """Initialize no-op metrics when prometheus is unavailable."""
    global SETTLEMENT_CAPTURED_TOTAL, SETTLEMENT_REVIEW_DUE_TOTAL
    global SETTLEMENT_STATUS_TRANSITIONS, SETTLEMENT_CONFIDENCE
    global SETTLEMENT_FALSIFIER_COUNT
    global CALIBRATION_BRIER_SCORE, CALIBRATION_OUTCOMES_TOTAL
    global INTERVENTION_TOTAL, INTERVENTION_PAUSE_DURATION

    noop = NoOpMetric()
    SETTLEMENT_CAPTURED_TOTAL = noop
    SETTLEMENT_REVIEW_DUE_TOTAL = noop
    SETTLEMENT_STATUS_TRANSITIONS = noop
    SETTLEMENT_CONFIDENCE = noop
    SETTLEMENT_FALSIFIER_COUNT = noop
    CALIBRATION_BRIER_SCORE = noop
    CALIBRATION_OUTCOMES_TOTAL = noop
    INTERVENTION_TOTAL = noop
    INTERVENTION_PAUSE_DURATION = noop


def reset_settlement_metric_state() -> None:
    """Reset module-level counters and init state for tests."""
    global _initialized
    global _CALIBRATION_OUTCOME_COUNTS
    _initialized = False
    _CALIBRATION_OUTCOME_COUNTS = {}


def get_calibration_outcomes_summary() -> dict[str, Any]:
    """Return normalized calibration outcome counters for dashboards.

    Buckets:
    - correct: explicit correct outcomes
    - incorrect: explicit incorrect outcomes
    - skipped: outcomes prefixed with "skipped"
    - deferred: outcomes prefixed with "pending" or explicit deferred labels
    """
    raw = dict(_CALIBRATION_OUTCOME_COUNTS)
    correct = int(raw.get("correct", 0))
    incorrect = int(raw.get("incorrect", 0))
    skipped = sum(v for k, v in raw.items() if k.startswith("skipped"))
    deferred = sum(v for k, v in raw.items() if k.startswith("pending") or k == "deferred")
    total = sum(int(v) for v in raw.values())
    return {
        "correct": correct,
        "incorrect": incorrect,
        "skipped": skipped,
        "deferred": deferred,
        "total": total,
        "raw": raw,
        "available": True,
    }

def _ensure_init() -> None:
    """Ensure metrics are initialized."""
    if not _initialized:
        init_settlement_metrics()


# =============================================================================
# Settlement Recording Functions
# =============================================================================


def record_settlement_captured(status: str = "settled") -> None:
    """Record a settlement captured after debate completion.

    Args:
        status: Initial settlement status (settled, pending, etc.)
    """
    _ensure_init()
    SETTLEMENT_CAPTURED_TOTAL.labels(status=status).inc()


def record_settlement_review_due(count: int) -> None:
    """Set the gauge for settlements currently due for review.

    Args:
        count: Number of settlements due for review.
    """
    _ensure_init()
    SETTLEMENT_REVIEW_DUE_TOTAL.set(count)


def record_settlement_transition(from_status: str, to_status: str) -> None:
    """Record a settlement status transition.

    Args:
        from_status: Previous status.
        to_status: New status.
    """
    _ensure_init()
    SETTLEMENT_STATUS_TRANSITIONS.labels(from_status=from_status, to_status=to_status).inc()


def record_settlement_confidence(confidence: float) -> None:
    """Record a settlement confidence score.

    Args:
        confidence: Confidence score (0.0-1.0).
    """
    _ensure_init()
    SETTLEMENT_CONFIDENCE.observe(confidence)


def record_settlement_falsifiers(count: int) -> None:
    """Record the number of falsifiers in a settlement.

    Args:
        count: Number of falsifiers identified.
    """
    _ensure_init()
    SETTLEMENT_FALSIFIER_COUNT.observe(count)


# =============================================================================
# Calibration Recording Functions
# =============================================================================


def record_calibration_brier(agent_name: str, brier_score: float) -> None:
    """Record a Brier score for an agent's calibration.

    Args:
        agent_name: Name of the agent.
        brier_score: Brier score (0.0 = perfect, 1.0 = worst).
    """
    _ensure_init()
    CALIBRATION_BRIER_SCORE.labels(agent_name=agent_name).observe(brier_score)


def record_calibration_outcome(outcome: str) -> None:
    """Record a calibration outcome event.

    Args:
        outcome: Outcome type (correct, incorrect, partial).
    """
    _ensure_init()
    outcome_label = str(outcome).strip().lower() or "unknown"
    _CALIBRATION_OUTCOME_COUNTS[outcome_label] = (
        _CALIBRATION_OUTCOME_COUNTS.get(outcome_label, 0) + 1
    )
    CALIBRATION_OUTCOMES_TOTAL.labels(outcome=outcome_label).inc()


# =============================================================================
# Intervention Recording Functions
# =============================================================================


def record_intervention(action: str) -> None:
    """Record an operator intervention action.

    Args:
        action: Action type (pause, resume, restart, inject_context).
    """
    _ensure_init()
    INTERVENTION_TOTAL.labels(action=action).inc()


def record_intervention_pause_duration(duration_seconds: float) -> None:
    """Record the duration of a debate pause.

    Args:
        duration_seconds: How long the debate was paused.
    """
    _ensure_init()
    INTERVENTION_PAUSE_DURATION.observe(duration_seconds)


__all__ = [
    # Settlement Metrics
    "SETTLEMENT_CAPTURED_TOTAL",
    "SETTLEMENT_REVIEW_DUE_TOTAL",
    "SETTLEMENT_STATUS_TRANSITIONS",
    "SETTLEMENT_CONFIDENCE",
    "SETTLEMENT_FALSIFIER_COUNT",
    # Calibration Metrics
    "CALIBRATION_BRIER_SCORE",
    "CALIBRATION_OUTCOMES_TOTAL",
    # Intervention Metrics
    "INTERVENTION_TOTAL",
    "INTERVENTION_PAUSE_DURATION",
    # Init
    "init_settlement_metrics",
    # Settlement Recording
    "record_settlement_captured",
    "record_settlement_review_due",
    "record_settlement_transition",
    "record_settlement_confidence",
    "record_settlement_falsifiers",
    # Calibration Recording
    "record_calibration_brier",
    "record_calibration_outcome",
    "get_calibration_outcomes_summary",
    "reset_settlement_metric_state",
    # Intervention Recording
    "record_intervention",
    "record_intervention_pause_duration",
]
