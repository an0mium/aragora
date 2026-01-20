"""
Novelty Tracker to Selection Feedback Bridge.

Bridges novelty metrics from NoveltyTracker into the SelectionFeedbackLoop,
enabling selection weight adjustments based on proposal originality.

This closes the loop between:
1. NoveltyTracker: Measures semantic novelty of proposals (vs low novelty)
2. SelectionFeedbackLoop: Adjusts selection weights based on performance metrics

By connecting them, we enable:
- Penalizing agents who consistently produce low-novelty responses
- Rewarding agents who consistently bring fresh perspectives
- Breaking "groupthink" by selecting for diversity

Usage:
    from aragora.debate.novelty_selection_bridge import NoveltySelectionBridge

    bridge = NoveltySelectionBridge(
        novelty_tracker=tracker,
        selection_feedback=feedback_loop,
        low_novelty_penalty=0.15,
    )

    # After each debate round
    bridge.record_round_novelty(round_result)

    # Get selection penalty for an agent
    penalty = bridge.get_novelty_penalty("claude")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from aragora.debate.novelty import NoveltyTracker, NoveltyResult
    from aragora.debate.selection_feedback import SelectionFeedbackLoop

logger = logging.getLogger(__name__)


@dataclass
class AgentNoveltyStats:
    """Novelty statistics for a single agent."""

    agent_name: str
    total_rounds: int = 0
    low_novelty_rounds: int = 0
    total_novelty_score: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)

    @property
    def avg_novelty(self) -> float:
        """Average novelty score across rounds."""
        if self.total_rounds == 0:
            return 1.0  # Default to max novelty
        return self.total_novelty_score / self.total_rounds

    @property
    def low_novelty_rate(self) -> float:
        """Proportion of rounds with low novelty."""
        if self.total_rounds == 0:
            return 0.0
        return self.low_novelty_rounds / self.total_rounds


@dataclass
class NoveltySelectionBridgeConfig:
    """Configuration for the novelty-selection bridge."""

    # Minimum rounds before applying penalty
    min_rounds_for_penalty: int = 5

    # Penalty weight for low novelty rate
    low_novelty_penalty_weight: float = 0.2

    # Bonus weight for consistently high novelty
    high_novelty_bonus_weight: float = 0.1

    # Low novelty threshold (sync with NoveltyTracker default)
    low_novelty_threshold: float = 0.15

    # High novelty threshold for bonus
    high_novelty_threshold: float = 0.6

    # Maximum penalty (prevents extreme deselection)
    max_penalty: float = 0.3

    # Maximum bonus
    max_bonus: float = 0.2

    # Decay factor for old novelty data (applied daily)
    decay_factor: float = 0.95


@dataclass
class NoveltySelectionBridge:
    """Bridges NoveltyTracker metrics into SelectionFeedbackLoop decisions.

    Key integration points:
    1. Tracks low_novelty_agents frequency per agent
    2. Computes selection penalties for consistently low-novelty agents
    3. Optionally rewards consistently novel agents
    4. Feeds into SelectionFeedbackLoop adjustments
    """

    novelty_tracker: Optional["NoveltyTracker"] = None
    selection_feedback: Optional["SelectionFeedbackLoop"] = None
    config: NoveltySelectionBridgeConfig = field(default_factory=NoveltySelectionBridgeConfig)

    # Internal state
    _agent_stats: Dict[str, AgentNoveltyStats] = field(default_factory=dict, repr=False)
    _novelty_adjustments: Dict[str, float] = field(default_factory=dict, repr=False)

    def record_round_novelty(
        self,
        novelty_result: "NoveltyResult",
        debate_id: Optional[str] = None,
    ) -> Dict[str, float]:
        """Record novelty results from a debate round.

        Args:
            novelty_result: NoveltyResult from NoveltyTracker
            debate_id: Optional debate identifier for logging

        Returns:
            Dict of agent_name -> novelty adjustment for this round
        """
        adjustments: Dict[str, float] = {}

        for agent_name, novelty_score in novelty_result.per_agent_novelty.items():
            stats = self._get_or_create_stats(agent_name)

            stats.total_rounds += 1
            stats.total_novelty_score += novelty_score

            if agent_name in novelty_result.low_novelty_agents:
                stats.low_novelty_rounds += 1

            stats.last_updated = datetime.now()

            # Compute adjustment
            adjustment = self._compute_adjustment(stats)
            adjustments[agent_name] = adjustment
            self._novelty_adjustments[agent_name] = adjustment

        logger.debug(
            f"novelty_selection_recorded round={novelty_result.round_num} "
            f"agents={len(adjustments)} "
            f"low_novelty={novelty_result.low_novelty_agents}"
        )

        return adjustments

    def record_from_tracker(self, debate_id: Optional[str] = None) -> Dict[str, float]:
        """Record all rounds from the attached NoveltyTracker.

        Useful when processing a completed debate.

        Args:
            debate_id: Optional debate identifier for logging

        Returns:
            Dict of final adjustments per agent
        """
        if self.novelty_tracker is None:
            logger.debug("No novelty tracker attached")
            return {}

        all_adjustments: Dict[str, float] = {}

        for novelty_result in self.novelty_tracker.scores:
            round_adjustments = self.record_round_novelty(novelty_result, debate_id)
            all_adjustments.update(round_adjustments)

        return all_adjustments

    def _get_or_create_stats(self, agent_name: str) -> AgentNoveltyStats:
        """Get or create novelty stats for an agent."""
        if agent_name not in self._agent_stats:
            self._agent_stats[agent_name] = AgentNoveltyStats(agent_name=agent_name)
        return self._agent_stats[agent_name]

    def _compute_adjustment(self, stats: AgentNoveltyStats) -> float:
        """Compute selection adjustment based on novelty stats.

        Args:
            stats: Agent's novelty statistics

        Returns:
            Adjustment factor (negative = penalty, positive = bonus)
        """
        if stats.total_rounds < self.config.min_rounds_for_penalty:
            return 0.0

        adjustment = 0.0

        # Penalty for high low-novelty rate
        if stats.low_novelty_rate > 0.3:  # >30% low novelty rounds
            penalty = stats.low_novelty_rate * self.config.low_novelty_penalty_weight
            penalty = min(penalty, self.config.max_penalty)
            adjustment -= penalty

        # Bonus for consistently high novelty
        if stats.avg_novelty > self.config.high_novelty_threshold:
            bonus = (
                stats.avg_novelty - self.config.high_novelty_threshold
            ) * self.config.high_novelty_bonus_weight
            bonus = min(bonus, self.config.max_bonus)
            adjustment += bonus

        # Additional penalty for very low average novelty
        if stats.avg_novelty < self.config.low_novelty_threshold * 2:
            # Below 0.3 average novelty
            severity = 1 - (stats.avg_novelty / (self.config.low_novelty_threshold * 2))
            adjustment -= severity * self.config.low_novelty_penalty_weight * 0.5

        return adjustment

    def get_novelty_penalty(self, agent_name: str) -> float:
        """Get the current novelty-based selection penalty for an agent.

        Args:
            agent_name: Name of the agent

        Returns:
            Penalty value (negative adjustment)
        """
        adjustment = self._novelty_adjustments.get(agent_name, 0.0)

        # Only return penalties (negative values)
        return min(0.0, adjustment)

    def get_novelty_bonus(self, agent_name: str) -> float:
        """Get the current novelty-based selection bonus for an agent.

        Args:
            agent_name: Name of the agent

        Returns:
            Bonus value (positive adjustment)
        """
        adjustment = self._novelty_adjustments.get(agent_name, 0.0)

        # Only return bonuses (positive values)
        return max(0.0, adjustment)

    def get_combined_adjustment(self, agent_name: str) -> float:
        """Get the combined novelty adjustment for an agent.

        Args:
            agent_name: Name of the agent

        Returns:
            Combined adjustment (penalty + bonus)
        """
        return self._novelty_adjustments.get(agent_name, 0.0)

    def get_all_adjustments(self) -> Dict[str, float]:
        """Get novelty adjustments for all tracked agents.

        Returns:
            Dict mapping agent names to adjustments
        """
        return dict(self._novelty_adjustments)

    def get_low_novelty_agents(self, threshold: float = 0.3) -> List[str]:
        """Get agents with consistently low novelty.

        Args:
            threshold: Low novelty rate threshold (default 30%)

        Returns:
            List of agent names with high low-novelty rates
        """
        return [
            agent_name
            for agent_name, stats in self._agent_stats.items()
            if stats.total_rounds >= self.config.min_rounds_for_penalty
            and stats.low_novelty_rate > threshold
        ]

    def get_high_novelty_agents(self, threshold: float = 0.6) -> List[str]:
        """Get agents with consistently high novelty.

        Args:
            threshold: High average novelty threshold (default 0.6)

        Returns:
            List of agent names with high average novelty
        """
        return [
            agent_name
            for agent_name, stats in self._agent_stats.items()
            if stats.total_rounds >= self.config.min_rounds_for_penalty
            and stats.avg_novelty > threshold
        ]

    def sync_to_selection_feedback(self) -> int:
        """Sync novelty adjustments to the SelectionFeedbackLoop.

        Applies novelty-based adjustments to the feedback loop's
        selection weights.

        Returns:
            Number of agents updated
        """
        if self.selection_feedback is None:
            logger.debug("No selection feedback loop attached")
            return 0

        updated = 0
        for agent_name, adjustment in self._novelty_adjustments.items():
            # Get current adjustment from feedback loop
            current = self.selection_feedback.get_selection_adjustment(agent_name)

            # Add novelty adjustment (it's already bounded)
            # Note: This modifies the feedback loop's internal state
            state = self.selection_feedback.get_agent_state(agent_name)
            if state:
                # Store in feedback loop's adjustment dict
                self.selection_feedback._selection_adjustments[agent_name] = current + adjustment
                updated += 1

        logger.info(f"novelty_selection_synced agents_updated={updated}")
        return updated

    def get_agent_stats(self, agent_name: str) -> Optional[AgentNoveltyStats]:
        """Get novelty statistics for an agent.

        Args:
            agent_name: Name of the agent

        Returns:
            AgentNoveltyStats if available
        """
        return self._agent_stats.get(agent_name)

    def get_all_stats(self) -> Dict[str, AgentNoveltyStats]:
        """Get novelty statistics for all agents.

        Returns:
            Dict mapping agent names to stats
        """
        return dict(self._agent_stats)

    def apply_decay(self) -> None:
        """Apply decay to all novelty stats.

        Call periodically (e.g., daily) to reduce influence of old data.
        """
        for stats in self._agent_stats.values():
            stats.total_novelty_score *= self.config.decay_factor
            stats.low_novelty_rounds = int(stats.low_novelty_rounds * self.config.decay_factor)
            stats.total_rounds = int(stats.total_rounds * self.config.decay_factor)

            # Recompute adjustment
            if stats.total_rounds > 0:
                adjustment = self._compute_adjustment(stats)
                self._novelty_adjustments[stats.agent_name] = adjustment

        logger.debug(f"novelty_decay_applied agents={len(self._agent_stats)}")

    def get_stats(self) -> Dict[str, Any]:
        """Get bridge statistics.

        Returns:
            Dict with bridge metrics
        """
        low_novelty_agents = self.get_low_novelty_agents()
        high_novelty_agents = self.get_high_novelty_agents()

        return {
            "agents_tracked": len(self._agent_stats),
            "low_novelty_agents": len(low_novelty_agents),
            "high_novelty_agents": len(high_novelty_agents),
            "total_rounds_recorded": sum(s.total_rounds for s in self._agent_stats.values()),
            "avg_adjustment": (
                sum(self._novelty_adjustments.values()) / len(self._novelty_adjustments)
                if self._novelty_adjustments
                else 0.0
            ),
        }

    def reset(self) -> None:
        """Reset all novelty statistics."""
        self._agent_stats.clear()
        self._novelty_adjustments.clear()
        logger.debug("NoveltySelectionBridge reset")


def create_novelty_selection_bridge(
    novelty_tracker: Optional["NoveltyTracker"] = None,
    selection_feedback: Optional["SelectionFeedbackLoop"] = None,
    **config_kwargs: Any,
) -> NoveltySelectionBridge:
    """Create and configure a NoveltySelectionBridge.

    Args:
        novelty_tracker: NoveltyTracker instance
        selection_feedback: SelectionFeedbackLoop instance
        **config_kwargs: Additional configuration options

    Returns:
        Configured NoveltySelectionBridge instance
    """
    config = NoveltySelectionBridgeConfig(**config_kwargs)
    return NoveltySelectionBridge(
        novelty_tracker=novelty_tracker,
        selection_feedback=selection_feedback,
        config=config,
    )


__all__ = [
    "NoveltySelectionBridge",
    "NoveltySelectionBridgeConfig",
    "AgentNoveltyStats",
    "create_novelty_selection_bridge",
]
