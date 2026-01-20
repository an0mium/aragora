"""
Performance-ELO Integration for adaptive agent ranking.

This module connects AgentPerformanceMonitor telemetry to ELO calculations,
enabling the ranking system to use actual performance metrics beyond just
win/loss outcomes.

The integration enables:
1. K-factor modulation based on agent consistency
2. Performance-weighted ELO adjustments
3. Automatic detection of degraded agents

Usage:
    from aragora.ranking.performance_integrator import PerformanceEloIntegrator

    integrator = PerformanceEloIntegrator(elo_system, performance_monitor)

    # Get K-factor multipliers for a match
    multipliers = integrator.compute_k_multipliers(["agent1", "agent2"])

    # Record match with performance-aware ELO
    elo_changes = elo_system.record_match(
        debate_id="debate-123",
        participants=["agent1", "agent2"],
        scores={"agent1": 1.0, "agent2": 0.0},
        calibration_tracker=integrator,  # Uses same interface
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from aragora.agents.performance_monitor import AgentPerformanceMonitor
    from aragora.ranking.elo import EloSystem

logger = logging.getLogger(__name__)


@dataclass
class PerformanceScore:
    """Performance score breakdown for an agent."""

    agent_name: str
    response_quality_score: float = 0.5  # 0.0-1.0, based on success rate
    latency_score: float = 0.5  # 0.0-1.0, faster = higher
    consistency_score: float = 0.5  # 0.0-1.0, low variance = higher
    participation_score: float = 0.5  # 0.0-1.0, more calls = higher (up to saturation)
    composite_score: float = 0.5  # Weighted combination

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "agent_name": self.agent_name,
            "response_quality_score": round(self.response_quality_score, 3),
            "latency_score": round(self.latency_score, 3),
            "consistency_score": round(self.consistency_score, 3),
            "participation_score": round(self.participation_score, 3),
            "composite_score": round(self.composite_score, 3),
        }


@dataclass
class PerformanceEloIntegrator:
    """Integrates AgentPerformanceMonitor metrics into ELO calculations.

    This class provides K-factor multipliers based on performance metrics,
    making ELO ratings more responsive for inconsistent agents and more
    stable for consistent performers.

    The integrator follows these principles:
    1. **Quality matters**: High success rate → lower K-factor (stable)
    2. **Speed is secondary**: Latency contributes less to score
    3. **Consistency is key**: Low variance → more stable ELO
    4. **Engagement counts**: Active agents get meaningful adjustments

    Attributes:
        elo_system: The ELO system for rating lookups
        performance_monitor: The performance monitor with telemetry
        response_quality_weight: Weight for success rate (0.0-1.0)
        latency_weight: Weight for response speed (0.0-1.0)
        consistency_weight: Weight for low variance (0.0-1.0)
        participation_weight: Weight for engagement (0.0-1.0)
        min_calls_for_adjustment: Minimum calls before adjusting K-factor
        k_factor_range: (min, max) multiplier range for K-factor
    """

    elo_system: Optional["EloSystem"] = None
    performance_monitor: Optional["AgentPerformanceMonitor"] = None

    # Weights for performance components (should sum to 1.0)
    response_quality_weight: float = 0.4  # Success rate impact
    latency_weight: float = 0.1  # Speed impact (minor)
    consistency_weight: float = 0.2  # Variance impact
    participation_weight: float = 0.3  # Engagement impact

    # Thresholds
    min_calls_for_adjustment: int = 5  # Minimum calls before adjusting
    latency_baseline_ms: float = 30000.0  # 30 seconds baseline
    max_latency_ms: float = 120000.0  # 2 minutes cap

    # K-factor modulation
    k_factor_range: tuple = (0.7, 1.5)  # Min/max K-factor multiplier

    # Cached scores
    _score_cache: Dict[str, PerformanceScore] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        """Normalize weights to sum to 1.0."""
        total = (
            self.response_quality_weight
            + self.latency_weight
            + self.consistency_weight
            + self.participation_weight
        )
        if total > 0 and abs(total - 1.0) > 0.001:
            # Normalize
            self.response_quality_weight /= total
            self.latency_weight /= total
            self.consistency_weight /= total
            self.participation_weight /= total

    def compute_performance_score(self, agent_name: str) -> PerformanceScore:
        """Compute a composite performance score for an agent.

        Args:
            agent_name: Name of the agent

        Returns:
            PerformanceScore with component and composite scores
        """
        if self.performance_monitor is None:
            return PerformanceScore(agent_name=agent_name)

        stats = self.performance_monitor.agent_stats.get(agent_name)
        if stats is None or stats.total_calls < self.min_calls_for_adjustment:
            # Not enough data, return neutral score
            return PerformanceScore(agent_name=agent_name)

        # 1. Response quality score (success rate)
        # Higher success rate → higher score
        quality_score = stats.success_rate / 100.0  # Convert percentage to 0-1

        # 2. Latency score (faster is better)
        # Normalize to 0-1 range, faster = higher
        avg_ms = stats.avg_duration_ms
        if avg_ms <= 0:
            latency_score = 0.5
        elif avg_ms >= self.max_latency_ms:
            latency_score = 0.0
        else:
            # Inverse relationship: faster = higher score
            # At baseline, score is 0.5
            # At 0ms, score is 1.0
            # At max, score is 0.0
            latency_score = max(0.0, 1.0 - (avg_ms / self.max_latency_ms))

        # 3. Consistency score (low variance is better)
        # Use the range between min and max as a proxy for variance
        if stats.min_duration_ms == float("inf") or stats.max_duration_ms == 0:
            consistency_score = 0.5
        else:
            range_ms = stats.max_duration_ms - stats.min_duration_ms
            # Normalize: smaller range = higher score
            # If range is 0 (perfect consistency), score is 1.0
            # If range >= baseline, score approaches 0
            normalized_range = min(range_ms / self.latency_baseline_ms, 2.0)
            consistency_score = max(0.0, 1.0 - (normalized_range / 2.0))

        # 4. Participation score (more calls = more engaged, with saturation)
        # Saturates at 20 calls for full score
        saturation_point = 20
        participation_score = min(1.0, stats.total_calls / saturation_point)

        # Compute weighted composite
        composite = (
            quality_score * self.response_quality_weight
            + latency_score * self.latency_weight
            + consistency_score * self.consistency_weight
            + participation_score * self.participation_weight
        )

        score = PerformanceScore(
            agent_name=agent_name,
            response_quality_score=quality_score,
            latency_score=latency_score,
            consistency_score=consistency_score,
            participation_score=participation_score,
            composite_score=composite,
        )

        # Cache for later use
        self._score_cache[agent_name] = score

        logger.debug(
            "performance_score agent=%s quality=%.2f latency=%.2f "
            "consistency=%.2f participation=%.2f composite=%.2f",
            agent_name,
            quality_score,
            latency_score,
            consistency_score,
            participation_score,
            composite,
        )

        return score

    def compute_k_multipliers(self, participants: List[str]) -> Dict[str, float]:
        """Compute K-factor multipliers for each participant.

        Higher multipliers mean more volatile ELO (bigger changes per match).
        This is applied to agents with inconsistent performance, while
        stable performers get lower multipliers.

        The logic:
        - High performance score → lower K-factor (stable ELO)
        - Low performance score → higher K-factor (volatile ELO)

        Args:
            participants: List of agent names

        Returns:
            Dict mapping agent_name → K-factor multiplier
        """
        multipliers: Dict[str, float] = {}
        min_k, max_k = self.k_factor_range

        for agent_name in participants:
            score = self.compute_performance_score(agent_name)

            # Map composite score to K-factor multiplier
            # High score → low K (stable), Low score → high K (volatile)
            # Composite of 1.0 → min_k, Composite of 0.0 → max_k
            multiplier = max_k - (score.composite_score * (max_k - min_k))

            # Clamp to range
            multiplier = max(min_k, min(max_k, multiplier))
            multipliers[agent_name] = multiplier

            logger.debug(
                "k_multiplier agent=%s score=%.2f multiplier=%.2f",
                agent_name,
                score.composite_score,
                multiplier,
            )

        return multipliers

    def get_cached_score(self, agent_name: str) -> Optional[PerformanceScore]:
        """Get cached performance score for an agent.

        Args:
            agent_name: Name of the agent

        Returns:
            Cached PerformanceScore or None if not computed
        """
        return self._score_cache.get(agent_name)

    def get_all_scores(self) -> Dict[str, PerformanceScore]:
        """Get all cached performance scores.

        Returns:
            Dict mapping agent_name → PerformanceScore
        """
        return dict(self._score_cache)

    def clear_cache(self) -> None:
        """Clear the score cache."""
        self._score_cache.clear()

    def get_degraded_agents(self, threshold: float = 0.3) -> List[str]:
        """Get list of agents with low performance scores.

        Args:
            threshold: Score threshold below which agents are flagged

        Returns:
            List of agent names with composite score below threshold
        """
        degraded = []

        if self.performance_monitor is None:
            return degraded

        for agent_name in self.performance_monitor.agent_stats:
            score = self.compute_performance_score(agent_name)
            if score.composite_score < threshold:
                degraded.append(agent_name)

        return degraded

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get a summary of performance metrics for all agents.

        Returns:
            Dictionary with agent performance summaries and recommendations
        """
        if self.performance_monitor is None:
            return {"message": "No performance monitor configured"}

        summary: Dict[str, Any] = {
            "agents": {},
            "degraded": [],
            "top_performers": [],
            "recommendations": [],
        }

        scores = []
        for agent_name in self.performance_monitor.agent_stats:
            score = self.compute_performance_score(agent_name)
            scores.append((agent_name, score))
            summary["agents"][agent_name] = score.to_dict()

        # Sort by composite score
        scores.sort(key=lambda x: x[1].composite_score, reverse=True)

        # Top performers
        summary["top_performers"] = [
            {"agent": name, "score": round(score.composite_score, 3)}
            for name, score in scores[:3]
        ]

        # Degraded agents
        summary["degraded"] = [
            {"agent": name, "score": round(score.composite_score, 3)}
            for name, score in scores
            if score.composite_score < 0.3
        ]

        # Recommendations
        for name, score in scores:
            if score.response_quality_score < 0.5:
                summary["recommendations"].append(
                    f"Agent '{name}' has low success rate ({score.response_quality_score:.0%}). "
                    f"Check API availability or prompt issues."
                )
            if score.consistency_score < 0.3:
                summary["recommendations"].append(
                    f"Agent '{name}' has high latency variance. "
                    f"Consider rate limiting or timeout tuning."
                )

        return summary

    # =========================================================================
    # CalibrationTracker interface compatibility
    # Allows PerformanceEloIntegrator to be passed to EloSystem.record_match()
    # =========================================================================

    def get_calibration_summary(self, agent_name: str) -> Any:
        """Get calibration summary for an agent (CalibrationTracker interface).

        This method allows the integrator to be used wherever a CalibrationTracker
        is expected, enabling seamless integration with existing ELO code.

        Args:
            agent_name: Name of the agent

        Returns:
            Object with calibration_score and total_predictions attributes
        """
        score = self.compute_performance_score(agent_name)

        # Return a duck-typed object that matches CalibrationTracker expectations
        @dataclass
        class CalibrationSummaryProxy:
            calibration_score: float
            total_predictions: int
            brier_score: float

        # Map composite score to calibration-style metrics
        # High performance → low Brier (good calibration)
        brier = 1.0 - score.composite_score  # Invert: high performance = low Brier

        stats = None
        if self.performance_monitor:
            stats = self.performance_monitor.agent_stats.get(agent_name)

        return CalibrationSummaryProxy(
            calibration_score=score.composite_score,
            total_predictions=stats.total_calls if stats else 0,
            brier_score=brier,
        )


def create_performance_integrator(
    elo_system: Optional["EloSystem"] = None,
    performance_monitor: Optional["AgentPerformanceMonitor"] = None,
    **kwargs: Any,
) -> PerformanceEloIntegrator:
    """Create a PerformanceEloIntegrator with optional configuration.

    Args:
        elo_system: EloSystem for rating lookups
        performance_monitor: AgentPerformanceMonitor for telemetry
        **kwargs: Additional configuration (weights, thresholds, etc.)

    Returns:
        Configured PerformanceEloIntegrator instance
    """
    return PerformanceEloIntegrator(
        elo_system=elo_system,
        performance_monitor=performance_monitor,
        **kwargs,
    )


__all__ = [
    "PerformanceEloIntegrator",
    "PerformanceScore",
    "create_performance_integrator",
]
