"""
Ranking and agent selection metrics.

Provides Prometheus metrics for tracking ELO rankings, calibration adjustments,
learning bonuses, voting accuracy, and agent selection feedback. These metrics
support monitoring of:

- Calibration adjustments applied to proposal confidence
- Learning efficiency ELO bonuses
- Voting accuracy updates
- Agent selection weight adjustments
- Performance-based routing decisions
- Novelty scoring and penalties
- Echo chamber detection
- Relationship bias adjustments
- RLM-efficient selection recommendations
- Budget filtering events
- Outcome complexity adjustments
- Analytics-driven selection recommendations
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Any
from collections.abc import Generator

from aragora.observability.metrics.base import NoOpMetric, get_metrics_enabled

logger = logging.getLogger(__name__)

# Core ELO/calibration metrics
CALIBRATION_ADJUSTMENTS: Any = None
LEARNING_BONUSES: Any = None
VOTING_ACCURACY_UPDATES: Any = None
SELECTION_FEEDBACK_ADJUSTMENTS: Any = None

# Performance routing metrics
PERFORMANCE_ROUTING_DECISIONS: Any = None
PERFORMANCE_ROUTING_LATENCY: Any = None

# Novelty and diversity metrics
NOVELTY_SCORE_CALCULATIONS: Any = None
NOVELTY_PENALTIES: Any = None
ECHO_CHAMBER_DETECTIONS: Any = None
RELATIONSHIP_BIAS_ADJUSTMENTS: Any = None

# RLM and cost efficiency metrics
RLM_SELECTION_RECOMMENDATIONS: Any = None
CALIBRATION_COST_CALCULATIONS: Any = None
BUDGET_FILTERING_EVENTS: Any = None

# Analytics-driven selection metrics
OUTCOME_COMPLEXITY_ADJUSTMENTS: Any = None
ANALYTICS_SELECTION_RECOMMENDATIONS: Any = None

_initialized = False


def init_ranking_metrics() -> None:
    """Initialize ranking and agent selection metrics."""
    global _initialized
    global CALIBRATION_ADJUSTMENTS, LEARNING_BONUSES, VOTING_ACCURACY_UPDATES
    global SELECTION_FEEDBACK_ADJUSTMENTS
    global PERFORMANCE_ROUTING_DECISIONS, PERFORMANCE_ROUTING_LATENCY
    global NOVELTY_SCORE_CALCULATIONS, NOVELTY_PENALTIES
    global ECHO_CHAMBER_DETECTIONS, RELATIONSHIP_BIAS_ADJUSTMENTS
    global RLM_SELECTION_RECOMMENDATIONS, CALIBRATION_COST_CALCULATIONS
    global BUDGET_FILTERING_EVENTS
    global OUTCOME_COMPLEXITY_ADJUSTMENTS, ANALYTICS_SELECTION_RECOMMENDATIONS

    if _initialized:
        return

    if not get_metrics_enabled():
        _init_noop_metrics()
        _initialized = True
        return

    try:
        from prometheus_client import Counter, Histogram

        # Core ELO/calibration metrics
        CALIBRATION_ADJUSTMENTS = Counter(
            "aragora_calibration_adjustments_total",
            "Proposal confidence calibrations applied",
            ["agent"],
        )

        LEARNING_BONUSES = Counter(
            "aragora_learning_bonuses_total",
            "Learning efficiency ELO bonuses applied",
            ["agent", "category"],
        )

        VOTING_ACCURACY_UPDATES = Counter(
            "aragora_voting_accuracy_updates_total",
            "Voting accuracy records updated",
            ["result"],
        )

        SELECTION_FEEDBACK_ADJUSTMENTS = Counter(
            "aragora_selection_feedback_adjustments_total",
            "Agent selection weight adjustments",
            ["agent", "direction"],
        )

        # Performance routing metrics
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

        # Novelty and diversity metrics
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

        # RLM and cost efficiency metrics
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

        # Analytics-driven selection metrics
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

        _initialized = True
        logger.debug("Ranking metrics initialized")

    except (ImportError, ValueError):
        _init_noop_metrics()
        _initialized = True
    except (RuntimeError, TypeError) as e:
        logger.warning(f"Failed to initialize ranking metrics: {e}")
        _init_noop_metrics()
        _initialized = True


def _init_noop_metrics() -> None:
    """Initialize no-op metrics when Prometheus is disabled."""
    global CALIBRATION_ADJUSTMENTS, LEARNING_BONUSES, VOTING_ACCURACY_UPDATES
    global SELECTION_FEEDBACK_ADJUSTMENTS
    global PERFORMANCE_ROUTING_DECISIONS, PERFORMANCE_ROUTING_LATENCY
    global NOVELTY_SCORE_CALCULATIONS, NOVELTY_PENALTIES
    global ECHO_CHAMBER_DETECTIONS, RELATIONSHIP_BIAS_ADJUSTMENTS
    global RLM_SELECTION_RECOMMENDATIONS, CALIBRATION_COST_CALCULATIONS
    global BUDGET_FILTERING_EVENTS
    global OUTCOME_COMPLEXITY_ADJUSTMENTS, ANALYTICS_SELECTION_RECOMMENDATIONS

    noop = NoOpMetric()
    CALIBRATION_ADJUSTMENTS = noop
    LEARNING_BONUSES = noop
    VOTING_ACCURACY_UPDATES = noop
    SELECTION_FEEDBACK_ADJUSTMENTS = noop
    PERFORMANCE_ROUTING_DECISIONS = noop
    PERFORMANCE_ROUTING_LATENCY = noop
    NOVELTY_SCORE_CALCULATIONS = noop
    NOVELTY_PENALTIES = noop
    ECHO_CHAMBER_DETECTIONS = noop
    RELATIONSHIP_BIAS_ADJUSTMENTS = noop
    RLM_SELECTION_RECOMMENDATIONS = noop
    CALIBRATION_COST_CALCULATIONS = noop
    BUDGET_FILTERING_EVENTS = noop
    OUTCOME_COMPLEXITY_ADJUSTMENTS = noop
    ANALYTICS_SELECTION_RECOMMENDATIONS = noop


def _ensure_init() -> None:
    """Ensure metrics are initialized."""
    if not _initialized:
        init_ranking_metrics()


# =============================================================================
# Core ELO/Calibration Recording Functions
# =============================================================================


def record_calibration_adjustment(agent: str) -> None:
    """Record a calibration adjustment applied to proposal confidence.

    Args:
        agent: Name of the agent receiving the calibration adjustment
    """
    _ensure_init()
    CALIBRATION_ADJUSTMENTS.labels(agent=agent).inc()


def record_learning_bonus(agent: str, category: str) -> None:
    """Record a learning efficiency ELO bonus.

    Args:
        agent: Name of the agent receiving the bonus
        category: Category of learning (e.g., "improvement", "consistency")
    """
    _ensure_init()
    LEARNING_BONUSES.labels(agent=agent, category=category).inc()


def record_voting_accuracy_update(result: str) -> None:
    """Record a voting accuracy update.

    Args:
        result: Result of the accuracy update (e.g., "correct", "incorrect")
    """
    _ensure_init()
    VOTING_ACCURACY_UPDATES.labels(result=result).inc()


def record_selection_feedback_adjustment(agent: str, direction: str) -> None:
    """Record an agent selection weight adjustment.

    Args:
        agent: Name of the agent being adjusted
        direction: Direction of adjustment ("increase" or "decrease")
    """
    _ensure_init()
    SELECTION_FEEDBACK_ADJUSTMENTS.labels(agent=agent, direction=direction).inc()


# =============================================================================
# Performance Routing Recording Functions
# =============================================================================


def record_performance_routing_decision(task_type: str, selected_agent: str) -> None:
    """Record a performance-based routing decision.

    Args:
        task_type: Type of task being routed
        selected_agent: Agent selected for the task
    """
    _ensure_init()
    PERFORMANCE_ROUTING_DECISIONS.labels(task_type=task_type, selected_agent=selected_agent).inc()


def record_performance_routing_latency(latency_seconds: float) -> None:
    """Record the latency of a routing decision computation.

    Args:
        latency_seconds: Time taken to compute the routing decision
    """
    _ensure_init()
    PERFORMANCE_ROUTING_LATENCY.observe(latency_seconds)


@contextmanager
def track_performance_routing(task_type: str) -> Generator[dict[str, Any], None, None]:
    """Context manager to track performance routing operations.

    Records both the routing decision and latency automatically.

    Args:
        task_type: Type of task being routed

    Yields:
        A dict that should be populated with 'selected_agent' key

    Example:
        with track_performance_routing("critique") as ctx:
            agent = await select_best_agent(candidates)
            ctx["selected_agent"] = agent.name
    """
    _ensure_init()
    start = time.perf_counter()
    context: dict[str, Any] = {"selected_agent": "unknown"}
    try:
        yield context
    finally:
        latency = time.perf_counter() - start
        record_performance_routing_decision(task_type, context.get("selected_agent", "unknown"))
        record_performance_routing_latency(latency)


# =============================================================================
# Novelty and Diversity Recording Functions
# =============================================================================


def record_novelty_score_calculation(agent: str) -> None:
    """Record a novelty score calculation.

    Args:
        agent: Name of the agent whose novelty was calculated
    """
    _ensure_init()
    NOVELTY_SCORE_CALCULATIONS.labels(agent=agent).inc()


def record_novelty_penalty(agent: str) -> None:
    """Record a selection penalty for low novelty.

    Args:
        agent: Name of the agent penalized
    """
    _ensure_init()
    NOVELTY_PENALTIES.labels(agent=agent).inc()


def record_echo_chamber_detection(risk_level: str) -> None:
    """Record an echo chamber risk detection.

    Args:
        risk_level: Level of risk detected (e.g., "low", "medium", "high")
    """
    _ensure_init()
    ECHO_CHAMBER_DETECTIONS.labels(risk_level=risk_level).inc()


def record_relationship_bias_adjustment(agent: str, direction: str) -> None:
    """Record a voting weight adjustment for alliance bias.

    Args:
        agent: Name of the agent being adjusted
        direction: Direction of adjustment ("increase" or "decrease")
    """
    _ensure_init()
    RELATIONSHIP_BIAS_ADJUSTMENTS.labels(agent=agent, direction=direction).inc()


# =============================================================================
# RLM and Cost Efficiency Recording Functions
# =============================================================================


def record_rlm_selection_recommendation(agent: str) -> None:
    """Record an RLM-efficient agent selection recommendation.

    Args:
        agent: Name of the recommended agent
    """
    _ensure_init()
    RLM_SELECTION_RECOMMENDATIONS.labels(agent=agent).inc()


def record_calibration_cost_calculation(agent: str, efficiency: str) -> None:
    """Record a cost efficiency calculation with calibration.

    Args:
        agent: Name of the agent
        efficiency: Efficiency classification (e.g., "high", "medium", "low")
    """
    _ensure_init()
    CALIBRATION_COST_CALCULATIONS.labels(agent=agent, efficiency=efficiency).inc()


def record_budget_filtering_event(outcome: str) -> None:
    """Record an agent filtering event due to budget constraints.

    Args:
        outcome: Outcome of the filtering (e.g., "filtered", "passed", "bypassed")
    """
    _ensure_init()
    BUDGET_FILTERING_EVENTS.labels(outcome=outcome).inc()


# =============================================================================
# Analytics-Driven Selection Recording Functions
# =============================================================================


def record_outcome_complexity_adjustment(direction: str) -> None:
    """Record a complexity budget adjustment from outcome patterns.

    Args:
        direction: Direction of adjustment ("increase" or "decrease")
    """
    _ensure_init()
    OUTCOME_COMPLEXITY_ADJUSTMENTS.labels(direction=direction).inc()


def record_analytics_selection_recommendation(recommendation_type: str) -> None:
    """Record an analytics-driven team selection recommendation.

    Args:
        recommendation_type: Type of recommendation made
    """
    _ensure_init()
    ANALYTICS_SELECTION_RECOMMENDATIONS.labels(recommendation_type=recommendation_type).inc()


__all__ = [
    # Metrics
    "CALIBRATION_ADJUSTMENTS",
    "LEARNING_BONUSES",
    "VOTING_ACCURACY_UPDATES",
    "SELECTION_FEEDBACK_ADJUSTMENTS",
    "PERFORMANCE_ROUTING_DECISIONS",
    "PERFORMANCE_ROUTING_LATENCY",
    "NOVELTY_SCORE_CALCULATIONS",
    "NOVELTY_PENALTIES",
    "ECHO_CHAMBER_DETECTIONS",
    "RELATIONSHIP_BIAS_ADJUSTMENTS",
    "RLM_SELECTION_RECOMMENDATIONS",
    "CALIBRATION_COST_CALCULATIONS",
    "BUDGET_FILTERING_EVENTS",
    "OUTCOME_COMPLEXITY_ADJUSTMENTS",
    "ANALYTICS_SELECTION_RECOMMENDATIONS",
    # Functions
    "init_ranking_metrics",
    "record_calibration_adjustment",
    "record_learning_bonus",
    "record_voting_accuracy_update",
    "record_selection_feedback_adjustment",
    "record_performance_routing_decision",
    "record_performance_routing_latency",
    "track_performance_routing",
    "record_novelty_score_calculation",
    "record_novelty_penalty",
    "record_echo_chamber_detection",
    "record_relationship_bias_adjustment",
    "record_rlm_selection_recommendation",
    "record_calibration_cost_calculation",
    "record_budget_filtering_event",
    "record_outcome_complexity_adjustment",
    "record_analytics_selection_recommendation",
]
