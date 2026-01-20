"""
Cross-pollination bridge metrics.

Provides Prometheus metrics for tracking bridge sync operations,
performance routing, novelty scoring, and echo chamber detection.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Any, Generator

from aragora.observability.metrics.base import NoOpMetric, get_metrics_enabled

logger = logging.getLogger(__name__)

# Global metric variables
BRIDGE_SYNCS: Any = None
BRIDGE_SYNC_LATENCY: Any = None
BRIDGE_ERRORS: Any = None
PERFORMANCE_ROUTING_DECISIONS: Any = None
PERFORMANCE_ROUTING_LATENCY: Any = None
OUTCOME_COMPLEXITY_ADJUSTMENTS: Any = None
ANALYTICS_SELECTION_RECOMMENDATIONS: Any = None
NOVELTY_SCORE_CALCULATIONS: Any = None
NOVELTY_PENALTIES: Any = None
ECHO_CHAMBER_DETECTIONS: Any = None
RELATIONSHIP_BIAS_ADJUSTMENTS: Any = None
RLM_SELECTION_RECOMMENDATIONS: Any = None
CALIBRATION_COST_CALCULATIONS: Any = None
BUDGET_FILTERING_EVENTS: Any = None

_initialized = False


def init_bridge_metrics() -> None:
    """Initialize bridge metrics."""
    global _initialized
    global BRIDGE_SYNCS, BRIDGE_SYNC_LATENCY, BRIDGE_ERRORS
    global PERFORMANCE_ROUTING_DECISIONS, PERFORMANCE_ROUTING_LATENCY
    global OUTCOME_COMPLEXITY_ADJUSTMENTS, ANALYTICS_SELECTION_RECOMMENDATIONS
    global NOVELTY_SCORE_CALCULATIONS, NOVELTY_PENALTIES
    global ECHO_CHAMBER_DETECTIONS, RELATIONSHIP_BIAS_ADJUSTMENTS
    global RLM_SELECTION_RECOMMENDATIONS, CALIBRATION_COST_CALCULATIONS
    global BUDGET_FILTERING_EVENTS

    if _initialized:
        return

    if not get_metrics_enabled():
        _init_noop_metrics()
        _initialized = True
        return

    try:
        from prometheus_client import Counter, Histogram

        BRIDGE_SYNCS = Counter(
            "aragora_bridge_syncs_total",
            "Cross-pollination bridge sync operations",
            ["bridge", "status"],
        )

        BRIDGE_SYNC_LATENCY = Histogram(
            "aragora_bridge_sync_latency_seconds",
            "Bridge sync operation latency",
            ["bridge"],
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
        )

        BRIDGE_ERRORS = Counter(
            "aragora_bridge_errors_total",
            "Cross-pollination bridge errors",
            ["bridge", "error_type"],
        )

        PERFORMANCE_ROUTING_DECISIONS = Counter(
            "aragora_performance_routing_decisions_total",
            "Performance-based routing decisions",
            ["task_type", "selected_agent"],
        )

        PERFORMANCE_ROUTING_LATENCY = Histogram(
            "aragora_performance_routing_latency_seconds",
            "Time to compute routing decision",
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1],
        )

        OUTCOME_COMPLEXITY_ADJUSTMENTS = Counter(
            "aragora_outcome_complexity_adjustments_total",
            "Complexity budget adjustments from outcome patterns",
            ["direction"],
        )

        ANALYTICS_SELECTION_RECOMMENDATIONS = Counter(
            "aragora_analytics_selection_recommendations_total",
            "Analytics-driven team selection recommendations",
            ["recommendation_type"],
        )

        NOVELTY_SCORE_CALCULATIONS = Counter(
            "aragora_novelty_score_calculations_total",
            "Novelty score calculations performed",
            ["agent"],
        )

        NOVELTY_PENALTIES = Counter(
            "aragora_novelty_penalties_total",
            "Selection penalties for low novelty",
            ["agent"],
        )

        ECHO_CHAMBER_DETECTIONS = Counter(
            "aragora_echo_chamber_detections_total",
            "Echo chamber risk detections in team composition",
            ["risk_level"],
        )

        RELATIONSHIP_BIAS_ADJUSTMENTS = Counter(
            "aragora_relationship_bias_adjustments_total",
            "Voting weight adjustments for alliance bias",
            ["agent", "direction"],
        )

        RLM_SELECTION_RECOMMENDATIONS = Counter(
            "aragora_rlm_selection_recommendations_total",
            "RLM-efficient agent selection recommendations",
            ["agent"],
        )

        CALIBRATION_COST_CALCULATIONS = Counter(
            "aragora_calibration_cost_calculations_total",
            "Cost efficiency calculations with calibration",
            ["agent", "efficiency"],
        )

        BUDGET_FILTERING_EVENTS = Counter(
            "aragora_budget_filtering_events_total",
            "Agent filtering events due to budget constraints",
            ["outcome"],
        )

        _initialized = True
        logger.debug("Bridge metrics initialized")

    except ImportError:
        _init_noop_metrics()
        _initialized = True


def _init_noop_metrics() -> None:
    """Initialize no-op metrics when Prometheus is disabled."""
    global BRIDGE_SYNCS, BRIDGE_SYNC_LATENCY, BRIDGE_ERRORS
    global PERFORMANCE_ROUTING_DECISIONS, PERFORMANCE_ROUTING_LATENCY
    global OUTCOME_COMPLEXITY_ADJUSTMENTS, ANALYTICS_SELECTION_RECOMMENDATIONS
    global NOVELTY_SCORE_CALCULATIONS, NOVELTY_PENALTIES
    global ECHO_CHAMBER_DETECTIONS, RELATIONSHIP_BIAS_ADJUSTMENTS
    global RLM_SELECTION_RECOMMENDATIONS, CALIBRATION_COST_CALCULATIONS
    global BUDGET_FILTERING_EVENTS

    BRIDGE_SYNCS = NoOpMetric()
    BRIDGE_SYNC_LATENCY = NoOpMetric()
    BRIDGE_ERRORS = NoOpMetric()
    PERFORMANCE_ROUTING_DECISIONS = NoOpMetric()
    PERFORMANCE_ROUTING_LATENCY = NoOpMetric()
    OUTCOME_COMPLEXITY_ADJUSTMENTS = NoOpMetric()
    ANALYTICS_SELECTION_RECOMMENDATIONS = NoOpMetric()
    NOVELTY_SCORE_CALCULATIONS = NoOpMetric()
    NOVELTY_PENALTIES = NoOpMetric()
    ECHO_CHAMBER_DETECTIONS = NoOpMetric()
    RELATIONSHIP_BIAS_ADJUSTMENTS = NoOpMetric()
    RLM_SELECTION_RECOMMENDATIONS = NoOpMetric()
    CALIBRATION_COST_CALCULATIONS = NoOpMetric()
    BUDGET_FILTERING_EVENTS = NoOpMetric()


def _ensure_init() -> None:
    """Ensure metrics are initialized."""
    if not _initialized:
        init_bridge_metrics()


# =============================================================================
# Recording Functions
# =============================================================================


def record_bridge_sync(bridge: str, success: bool) -> None:
    """Record a bridge sync operation.

    Args:
        bridge: Bridge name (performance_router, relationship_bias, etc.)
        success: Whether the sync succeeded
    """
    _ensure_init()
    status = "success" if success else "error"
    BRIDGE_SYNCS.labels(bridge=bridge, status=status).inc()


def record_bridge_sync_latency(bridge: str, latency_seconds: float) -> None:
    """Record bridge sync operation latency.

    Args:
        bridge: Bridge name
        latency_seconds: Time taken for sync operation
    """
    _ensure_init()
    BRIDGE_SYNC_LATENCY.labels(bridge=bridge).observe(latency_seconds)


def record_bridge_error(bridge: str, error_type: str) -> None:
    """Record a bridge error.

    Args:
        bridge: Bridge name
        error_type: Type of error (e.g., "initialization", "sync", "compute")
    """
    _ensure_init()
    BRIDGE_ERRORS.labels(bridge=bridge, error_type=error_type).inc()


def record_performance_routing_decision(task_type: str, selected_agent: str) -> None:
    """Record a performance-based routing decision.

    Args:
        task_type: Task type (speed, precision, balanced)
        selected_agent: Agent selected for the task
    """
    _ensure_init()
    PERFORMANCE_ROUTING_DECISIONS.labels(task_type=task_type, selected_agent=selected_agent).inc()


def record_performance_routing_latency(latency_seconds: float) -> None:
    """Record time to compute routing decision.

    Args:
        latency_seconds: Time taken to compute routing
    """
    _ensure_init()
    PERFORMANCE_ROUTING_LATENCY.observe(latency_seconds)


def record_outcome_complexity_adjustment(direction: str) -> None:
    """Record a complexity budget adjustment.

    Args:
        direction: Adjustment direction (increased, decreased)
    """
    _ensure_init()
    OUTCOME_COMPLEXITY_ADJUSTMENTS.labels(direction=direction).inc()


def record_analytics_selection_recommendation(recommendation_type: str) -> None:
    """Record an analytics-driven selection recommendation.

    Args:
        recommendation_type: Type of recommendation (boost, penalty, neutral)
    """
    _ensure_init()
    ANALYTICS_SELECTION_RECOMMENDATIONS.labels(recommendation_type=recommendation_type).inc()


def record_novelty_score_calculation(agent: str) -> None:
    """Record a novelty score calculation.

    Args:
        agent: Agent name
    """
    _ensure_init()
    NOVELTY_SCORE_CALCULATIONS.labels(agent=agent).inc()


def record_novelty_penalty(agent: str) -> None:
    """Record a selection penalty for low novelty.

    Args:
        agent: Agent name
    """
    _ensure_init()
    NOVELTY_PENALTIES.labels(agent=agent).inc()


def record_echo_chamber_detection(risk_level: str) -> None:
    """Record an echo chamber risk detection.

    Args:
        risk_level: Risk level (low, medium, high)
    """
    _ensure_init()
    ECHO_CHAMBER_DETECTIONS.labels(risk_level=risk_level).inc()


def record_relationship_bias_adjustment(agent: str, direction: str) -> None:
    """Record a voting weight adjustment for alliance bias.

    Args:
        agent: Agent name
        direction: Adjustment direction (up, down)
    """
    _ensure_init()
    RELATIONSHIP_BIAS_ADJUSTMENTS.labels(agent=agent, direction=direction).inc()


def record_rlm_selection_recommendation(agent: str) -> None:
    """Record an RLM-efficient agent selection recommendation.

    Args:
        agent: Agent name recommended for RLM efficiency
    """
    _ensure_init()
    RLM_SELECTION_RECOMMENDATIONS.labels(agent=agent).inc()


def record_calibration_cost_calculation(agent: str, efficiency: str) -> None:
    """Record a cost efficiency calculation.

    Args:
        agent: Agent name
        efficiency: Efficiency category (efficient, moderate, inefficient)
    """
    _ensure_init()
    CALIBRATION_COST_CALCULATIONS.labels(agent=agent, efficiency=efficiency).inc()


def record_budget_filtering_event(outcome: str) -> None:
    """Record an agent filtering event due to budget constraints.

    Args:
        outcome: Filtering outcome (included, excluded)
    """
    _ensure_init()
    BUDGET_FILTERING_EVENTS.labels(outcome=outcome).inc()


@contextmanager
def track_bridge_sync(bridge: str) -> Generator[None, None, None]:
    """Context manager to track bridge sync operations.

    Automatically records sync success/failure and latency.

    Args:
        bridge: Bridge name

    Example:
        with track_bridge_sync("performance_router"):
            bridge.sync_to_router()
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
        record_bridge_sync(bridge, success)
        record_bridge_sync_latency(bridge, latency)


__all__ = [
    # Metrics
    "BRIDGE_SYNCS",
    "BRIDGE_SYNC_LATENCY",
    "BRIDGE_ERRORS",
    "PERFORMANCE_ROUTING_DECISIONS",
    "PERFORMANCE_ROUTING_LATENCY",
    "OUTCOME_COMPLEXITY_ADJUSTMENTS",
    "ANALYTICS_SELECTION_RECOMMENDATIONS",
    "NOVELTY_SCORE_CALCULATIONS",
    "NOVELTY_PENALTIES",
    "ECHO_CHAMBER_DETECTIONS",
    "RELATIONSHIP_BIAS_ADJUSTMENTS",
    "RLM_SELECTION_RECOMMENDATIONS",
    "CALIBRATION_COST_CALCULATIONS",
    "BUDGET_FILTERING_EVENTS",
    # Functions
    "record_bridge_sync",
    "record_bridge_sync_latency",
    "record_bridge_error",
    "record_performance_routing_decision",
    "record_performance_routing_latency",
    "record_outcome_complexity_adjustment",
    "record_analytics_selection_recommendation",
    "record_novelty_score_calculation",
    "record_novelty_penalty",
    "record_echo_chamber_detection",
    "record_relationship_bias_adjustment",
    "record_rlm_selection_recommendation",
    "record_calibration_cost_calculation",
    "record_budget_filtering_event",
    "track_bridge_sync",
    "init_bridge_metrics",
]
