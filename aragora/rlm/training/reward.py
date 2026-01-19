"""
Reward models for RLM training.

Provides reward signals for RL training of context management strategies,
based on debate outcomes, efficiency, and answer quality.

Usage:
    from aragora.rlm.training.reward import DebateOutcomeReward

    reward_model = DebateOutcomeReward()
    reward = reward_model.compute(trajectory)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from .buffer import Trajectory

logger = logging.getLogger(__name__)


@dataclass
class RewardConfig:
    """Configuration for reward computation."""

    # Component weights (should sum to 1.0)
    consensus_weight: float = 0.4
    efficiency_weight: float = 0.2
    confidence_weight: float = 0.2
    iteration_penalty_weight: float = 0.1
    quality_weight: float = 0.1

    # Scaling factors
    max_sub_calls: int = 10  # Sub-calls above this get penalized
    max_iterations: int = 5  # Iterations above this get penalized
    target_tokens: int = 4000  # Target context size for efficiency


class RewardModel(ABC):
    """Abstract base class for reward models."""

    @abstractmethod
    def compute(self, trajectory: "Trajectory") -> float:
        """
        Compute reward for a trajectory.

        Args:
            trajectory: The trajectory to evaluate

        Returns:
            Reward value (typically in range [-1, 1])
        """
        pass

    @abstractmethod
    def compute_components(self, trajectory: "Trajectory") -> dict[str, float]:
        """
        Compute individual reward components.

        Args:
            trajectory: The trajectory to evaluate

        Returns:
            Dictionary of component names to values
        """
        pass


@dataclass
class DebateOutcomeReward(RewardModel):
    """
    Reward model based on debate outcomes.

    Rewards:
    - Consensus reached -> positive
    - Fewer sub-LM calls -> positive (efficiency)
    - Higher confidence -> positive
    - Fewer refinement iterations -> positive
    - High quality synthesis -> positive

    Penalties:
    - No consensus -> negative
    - Excessive sub-calls -> negative
    - Low confidence -> negative
    - Max iterations hit -> negative
    """

    config: RewardConfig = field(default_factory=RewardConfig)

    def compute(self, trajectory: "Trajectory") -> float:
        """Compute total reward for trajectory."""
        components = self.compute_components(trajectory)
        return sum(components.values())

    def compute_components(self, trajectory: "Trajectory") -> dict[str, float]:
        """Compute individual reward components."""
        components = {}

        # Consensus component
        consensus_reward = self._compute_consensus_reward(trajectory)
        components["consensus"] = consensus_reward * self.config.consensus_weight

        # Efficiency component (sub-calls)
        efficiency_reward = self._compute_efficiency_reward(trajectory)
        components["efficiency"] = efficiency_reward * self.config.efficiency_weight

        # Confidence component
        confidence_reward = self._compute_confidence_reward(trajectory)
        components["confidence"] = confidence_reward * self.config.confidence_weight

        # Iteration penalty
        iteration_reward = self._compute_iteration_reward(trajectory)
        components["iteration"] = iteration_reward * self.config.iteration_penalty_weight

        # Quality component (if available)
        quality_reward = self._compute_quality_reward(trajectory)
        components["quality"] = quality_reward * self.config.quality_weight

        return components

    def _compute_consensus_reward(self, trajectory: "Trajectory") -> float:
        """Reward for reaching consensus."""
        if not trajectory.outcome:
            return -1.0

        consensus_reached = trajectory.outcome.get("consensus_reached", False)
        if consensus_reached:
            # Bonus for strong consensus
            agreement_score = trajectory.outcome.get("agreement_score", 0.5)
            return 0.5 + (agreement_score * 0.5)  # Range [0.5, 1.0]
        else:
            # Partial credit for near-consensus
            agreement_score = trajectory.outcome.get("agreement_score", 0.0)
            if agreement_score > 0.7:
                return agreement_score - 0.5  # Range [0.2, 0.5]
            return -0.5

    def _compute_efficiency_reward(self, trajectory: "Trajectory") -> float:
        """Reward for efficient sub-call usage."""
        sub_calls = trajectory.stats.get("sub_calls_made", 0)

        if sub_calls == 0:
            # No sub-calls might mean we missed useful decomposition
            return 0.5

        if sub_calls <= self.config.max_sub_calls:
            # Fewer is better, but some are expected
            ratio = sub_calls / self.config.max_sub_calls
            return 1.0 - (ratio * 0.5)  # Range [0.5, 1.0]
        else:
            # Penalize excessive sub-calls
            excess = sub_calls - self.config.max_sub_calls
            penalty = min(excess * 0.1, 0.5)
            return 0.0 - penalty

    def _compute_confidence_reward(self, trajectory: "Trajectory") -> float:
        """Reward for high confidence."""
        confidence = trajectory.stats.get("confidence", 0.5)
        # Map confidence [0, 1] to reward [-0.5, 1.0]
        return (confidence * 1.5) - 0.5

    def _compute_iteration_reward(self, trajectory: "Trajectory") -> float:
        """Penalize excessive refinement iterations."""
        iterations = trajectory.stats.get("iterations", 1)
        ready = trajectory.stats.get("ready", True)

        if iterations == 1:
            return 1.0  # Perfect - no refinement needed

        if ready:
            # Refinement succeeded
            if iterations <= 3:
                return 0.5
            elif iterations <= self.config.max_iterations:
                return 0.0
            else:
                return -0.5
        else:
            # Hit max iterations without ready=True
            return -1.0

    def _compute_quality_reward(self, trajectory: "Trajectory") -> float:
        """Reward for answer quality (if evaluated)."""
        quality_score = trajectory.outcome.get("quality_score", None)
        if quality_score is None:
            return 0.0  # No quality evaluation available

        # Quality score is typically 0-1
        return (quality_score * 2.0) - 1.0  # Map to [-1, 1]


@dataclass
class CompositeReward(RewardModel):
    """
    Composite reward combining multiple reward models.

    Useful for combining different aspects of performance
    (e.g., debate outcomes + code quality + efficiency).
    """

    models: list[tuple[RewardModel, float]] = field(default_factory=list)

    def add_model(self, model: RewardModel, weight: float = 1.0) -> None:
        """Add a reward model with weight."""
        self.models.append((model, weight))

    def compute(self, trajectory: "Trajectory") -> float:
        """Compute weighted sum of all model rewards."""
        if not self.models:
            return 0.0

        total_weight = sum(w for _, w in self.models)
        if total_weight == 0:
            return 0.0

        weighted_sum = sum(
            model.compute(trajectory) * weight
            for model, weight in self.models
        )
        return weighted_sum / total_weight

    def compute_components(self, trajectory: "Trajectory") -> dict[str, float]:
        """Compute components from all models."""
        components = {}
        for i, (model, weight) in enumerate(self.models):
            model_components = model.compute_components(trajectory)
            for key, value in model_components.items():
                components[f"{type(model).__name__}_{i}_{key}"] = value * weight
        return components


@dataclass
class SparseReward(RewardModel):
    """
    Sparse reward model that only provides reward at episode end.

    Useful for training with outcome-based rewards without
    intermediate shaping.
    """

    success_reward: float = 1.0
    failure_penalty: float = -1.0
    partial_reward: float = 0.0

    def compute(self, trajectory: "Trajectory") -> float:
        """Compute sparse reward based on final outcome."""
        if not trajectory.is_terminal:
            return 0.0

        if not trajectory.outcome:
            return self.failure_penalty

        success = trajectory.outcome.get("success", False)
        if success:
            return self.success_reward

        # Check for partial success
        partial = trajectory.outcome.get("partial_success", False)
        if partial:
            return self.partial_reward

        return self.failure_penalty

    def compute_components(self, trajectory: "Trajectory") -> dict[str, float]:
        """Return single component for sparse reward."""
        return {"sparse": self.compute(trajectory)}


__all__ = [
    "RewardConfig",
    "RewardModel",
    "DebateOutcomeReward",
    "CompositeReward",
    "SparseReward",
]
