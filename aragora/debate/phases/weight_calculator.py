"""
Vote weight calculation for consensus phase.

This module extracts the vote weighting logic from ConsensusPhase,
providing a clean interface for calculating agent voting weights
based on multiple factors:
- Reputation (from memory)
- Reliability (from capability probing)
- Consistency (from FlipDetector)
- Calibration (from ELO or CalibrationTracker)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    from aragora.core import Agent

logger = logging.getLogger(__name__)


@dataclass
class WeightFactors:
    """Individual weight factors for an agent.

    Stores the component weights that are multiplied together
    to produce the final vote weight.
    """

    reputation: float = 1.0  # 0.5-1.5 from memory vote history
    reliability: float = 1.0  # 0.0-1.0 from capability probing
    consistency: float = 1.0  # 0.5-1.0 from FlipDetector
    calibration: float = 1.0  # 0.5-1.5 from ELO calibration score

    @property
    def total(self) -> float:
        """Calculate combined weight from all factors."""
        return self.reputation * self.reliability * self.consistency * self.calibration


@dataclass
class WeightCalculatorConfig:
    """Configuration for weight calculation.

    Controls which weight factors are enabled and their bounds.
    """

    enable_reputation: bool = True
    enable_reliability: bool = True
    enable_consistency: bool = True
    enable_calibration: bool = True

    # Bounds for weight factors
    min_weight: float = 0.1
    max_weight: float = 5.0


class WeightCalculator:
    """Calculate agent voting weights from multiple sources.

    Usage:
        calculator = WeightCalculator(
            memory=memory_system,
            elo_system=elo_system,
            flip_detector=flip_detector,
            agent_weights=probe_weights,
        )

        # Calculate weights for all agents
        weights = calculator.compute_weights(agents)

        # Get individual weight with breakdown
        weight, factors = calculator.get_weight_with_factors(agent_name)
    """

    def __init__(
        self,
        memory: Any = None,
        elo_system: Any = None,
        flip_detector: Any = None,
        agent_weights: Optional[dict[str, float]] = None,
        calibration_tracker: Any = None,
        get_calibration_weight: Optional[Callable[[str], float]] = None,
        config: Optional[WeightCalculatorConfig] = None,
    ):
        """Initialize the weight calculator.

        Args:
            memory: Memory system with get_vote_weight method
            elo_system: ELO system for calibration scores
            flip_detector: FlipDetector for consistency scores
            agent_weights: Pre-computed reliability weights from probing
            calibration_tracker: CalibrationTracker for calibration scores
            get_calibration_weight: Fallback callback for calibration
            config: Configuration for weight calculation
        """
        self.memory = memory
        self.elo_system = elo_system
        self.flip_detector = flip_detector
        self.agent_weights = agent_weights or {}
        self.calibration_tracker = calibration_tracker
        self._get_calibration_weight = get_calibration_weight
        self.config = config or WeightCalculatorConfig()

        # Cache for batch operations
        self._ratings_cache: dict[str, Any] = {}

    def compute_weights(self, agents: list["Agent"]) -> dict[str, float]:
        """Compute vote weights for all agents.

        Args:
            agents: List of agents to compute weights for

        Returns:
            Dict mapping agent names to their weights
        """
        # Batch fetch ELO ratings for efficiency
        self._prefetch_ratings([a.name for a in agents])

        weights = {}
        for agent in agents:
            weights[agent.name] = self.get_weight(agent.name)

        return weights

    def get_weight(self, agent_name: str) -> float:
        """Get the combined vote weight for an agent.

        Args:
            agent_name: Name of the agent

        Returns:
            Combined weight (product of all factors)
        """
        factors = self._compute_factors(agent_name)
        weight = factors.total

        # Apply bounds
        weight = max(self.config.min_weight, min(self.config.max_weight, weight))

        return weight

    def get_weight_with_factors(self, agent_name: str) -> tuple[float, WeightFactors]:
        """Get weight with breakdown of individual factors.

        Useful for debugging and understanding weight contributions.

        Args:
            agent_name: Name of the agent

        Returns:
            Tuple of (final_weight, WeightFactors)
        """
        factors = self._compute_factors(agent_name)
        weight = max(self.config.min_weight, min(self.config.max_weight, factors.total))
        return weight, factors

    def _prefetch_ratings(self, agent_names: list[str]) -> None:
        """Batch fetch ELO ratings for efficiency."""
        if not self.elo_system:
            return

        try:
            self._ratings_cache = self.elo_system.get_ratings_batch(agent_names)
        except Exception as e:
            logger.debug(f"Batch ratings fetch failed: {e}")
            self._ratings_cache = {}

    def _compute_factors(self, agent_name: str) -> WeightFactors:
        """Compute individual weight factors for an agent."""
        factors = WeightFactors()

        # Reputation weight (0.5-1.5)
        if self.config.enable_reputation:
            factors.reputation = self._get_reputation_weight(agent_name)

        # Reliability weight (0.0-1.0)
        if self.config.enable_reliability:
            factors.reliability = self._get_reliability_weight(agent_name)

        # Consistency weight (0.5-1.0)
        if self.config.enable_consistency:
            factors.consistency = self._get_consistency_weight(agent_name)

        # Calibration weight (0.5-1.5)
        if self.config.enable_calibration:
            factors.calibration = self._get_calibration_weight_for_agent(agent_name)

        return factors

    def _get_reputation_weight(self, agent_name: str) -> float:
        """Get reputation weight from memory system."""
        if not self.memory or not hasattr(self.memory, "get_vote_weight"):
            return 1.0

        try:
            return self.memory.get_vote_weight(agent_name)
        except Exception as e:
            logger.debug(f"Reputation weight error for {agent_name}: {e}")
            return 1.0

    def _get_reliability_weight(self, agent_name: str) -> float:
        """Get reliability weight from capability probing."""
        if not self.agent_weights:
            return 1.0

        return self.agent_weights.get(agent_name, 1.0)

    def _get_consistency_weight(self, agent_name: str) -> float:
        """Get consistency weight from FlipDetector."""
        if not self.flip_detector:
            return 1.0

        try:
            consistency = self.flip_detector.get_agent_consistency(agent_name)
            # Map 0.0-1.0 consistency score to 0.5-1.0 weight
            return 0.5 + (consistency.consistency_score * 0.5)
        except Exception as e:
            logger.debug(f"Consistency weight error for {agent_name}: {e}")
            return 1.0

    def _get_calibration_weight_for_agent(self, agent_name: str) -> float:
        """Get calibration weight from ELO or CalibrationTracker."""
        # Try cached ELO ratings first
        if agent_name in self._ratings_cache:
            cal_score = self._ratings_cache[agent_name].calibration_score
            return 0.5 + cal_score

        # Fallback to callback
        if self._get_calibration_weight:
            try:
                return self._get_calibration_weight(agent_name)
            except Exception as e:
                logger.debug(f"Calibration weight callback error for {agent_name}: {e}")

        return 1.0

    def clear_cache(self) -> None:
        """Clear the ratings cache."""
        self._ratings_cache.clear()
