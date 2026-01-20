"""
Selection Feedback Loop - Performance metrics to agent selection weights.

Provides a feedback mechanism that adjusts agent selection weights based on
their actual debate performance. This creates a self-improving system where
agents that perform well get selected more often for future debates.

Features:
- Tracks win rates, timeout rates, calibration quality per agent
- Computes selection weight adjustments
- Decays old feedback over time
- Bounded adjustments to prevent runaway effects
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class FeedbackLoopConfig:
    """Configuration for the selection feedback loop."""

    enabled: bool = True
    performance_to_selection_weight: float = 0.15  # How much performance affects selection
    calibration_to_elo_weight: float = 0.1  # How much calibration affects ELO contribution
    min_debates_for_feedback: int = 3  # Min debates before applying feedback
    feedback_decay_factor: float = 0.9  # Decay multiplier per day
    max_adjustment: float = 0.5  # Maximum selection weight adjustment (+/-)
    recency_window_days: int = 30  # Only consider debates within this window


@dataclass
class AgentFeedbackState:
    """Tracks performance metrics for a single agent."""

    agent_name: str
    total_debates: int = 0
    wins: int = 0
    losses: int = 0
    timeouts: int = 0
    correct_predictions: int = 0
    total_predictions: int = 0
    last_updated: datetime = field(default_factory=datetime.now)

    # Domain-specific tracking
    domain_wins: Dict[str, int] = field(default_factory=dict)
    domain_losses: Dict[str, int] = field(default_factory=dict)

    # Running averages
    avg_confidence: float = 0.5
    avg_response_time_ms: float = 0.0

    @property
    def win_rate(self) -> float:
        """Calculate overall win rate."""
        total = self.wins + self.losses
        return self.wins / total if total > 0 else 0.5

    @property
    def timeout_rate(self) -> float:
        """Calculate timeout rate."""
        return self.timeouts / self.total_debates if self.total_debates > 0 else 0.0

    @property
    def calibration_score(self) -> float:
        """Calculate prediction accuracy (calibration)."""
        if self.total_predictions == 0:
            return 0.5
        return self.correct_predictions / self.total_predictions

    def domain_win_rate(self, domain: str) -> float:
        """Calculate win rate for a specific domain."""
        wins = self.domain_wins.get(domain, 0)
        losses = self.domain_losses.get(domain, 0)
        total = wins + losses
        return wins / total if total > 0 else 0.5


@dataclass
class FeedbackLoopMetrics:
    """Metrics for the feedback loop system."""

    debates_processed: int = 0
    adjustments_computed: int = 0
    agents_tracked: int = 0
    total_adjustment_sum: float = 0.0
    last_processed: Optional[datetime] = None


class SelectionFeedbackLoop:
    """
    Connects agent performance metrics to selection weights.

    Tracks how agents perform in debates and computes adjustments to
    their selection weights. Agents that win more often, respond faster,
    and make accurate predictions get higher selection weights.

    Example:
        feedback = SelectionFeedbackLoop(config=FeedbackLoopConfig(
            performance_to_selection_weight=0.2,
            min_debates_for_feedback=5,
        ))

        # After each debate
        adjustments = feedback.process_debate_outcome(
            debate_id="debate-123",
            participants=["claude", "gpt-4", "gemini"],
            winner="claude",
            domain="security",
        )

        # When selecting agents
        for agent in candidates:
            adjustment = feedback.get_selection_adjustment(agent.name)
            agent.selection_weight *= (1.0 + adjustment)
    """

    def __init__(
        self,
        config: Optional[FeedbackLoopConfig] = None,
        elo_system: Optional[Any] = None,
        calibration_tracker: Optional[Any] = None,
    ):
        """
        Initialize the selection feedback loop.

        Args:
            config: Configuration for feedback behavior
            elo_system: Optional EloSystem for ELO-based adjustments
            calibration_tracker: Optional CalibrationTracker for prediction accuracy
        """
        self.config = config or FeedbackLoopConfig()
        self.elo_system = elo_system
        self.calibration_tracker = calibration_tracker

        self._agent_states: Dict[str, AgentFeedbackState] = {}
        self._selection_adjustments: Dict[str, float] = {}
        self.metrics = FeedbackLoopMetrics()

    def process_debate_outcome(
        self,
        debate_id: str,
        participants: List[str],
        winner: Optional[str],
        domain: str = "general",
        confidence: float = 0.0,
        response_times: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        Process a debate outcome and update agent feedback states.

        Args:
            debate_id: Unique identifier for the debate
            participants: List of agent names that participated
            winner: Name of the winning agent (None if no winner)
            domain: Domain/topic category of the debate
            confidence: Debate confidence score
            response_times: Optional dict of agent_name -> response_time_ms

        Returns:
            Dict of agent_name -> selection adjustment for this debate
        """
        if not self.config.enabled:
            return {}

        adjustments: Dict[str, float] = {}
        response_times = response_times or {}

        for agent_name in participants:
            state = self._get_or_create_state(agent_name)

            # Update debate count
            state.total_debates += 1

            # Update wins/losses
            if winner:
                if agent_name == winner:
                    state.wins += 1
                    state.domain_wins[domain] = state.domain_wins.get(domain, 0) + 1
                else:
                    state.losses += 1
                    state.domain_losses[domain] = state.domain_losses.get(domain, 0) + 1

            # Update response time average
            if agent_name in response_times:
                rt = response_times[agent_name]
                # Exponential moving average
                state.avg_response_time_ms = (
                    0.9 * state.avg_response_time_ms + 0.1 * rt
                )

            # Update confidence average
            if confidence > 0:
                state.avg_confidence = 0.9 * state.avg_confidence + 0.1 * confidence

            state.last_updated = datetime.now()

            # Compute adjustment
            adjustment = self._compute_adjustment(state, domain)
            adjustments[agent_name] = adjustment
            self._selection_adjustments[agent_name] = adjustment

        self.metrics.debates_processed += 1
        self.metrics.adjustments_computed += len(adjustments)
        self.metrics.last_processed = datetime.now()

        return adjustments

    def get_selection_adjustment(self, agent_name: str) -> float:
        """
        Get the current selection weight adjustment for an agent.

        Args:
            agent_name: Name of the agent

        Returns:
            Adjustment factor in range [-max_adjustment, +max_adjustment]
        """
        if not self.config.enabled:
            return 0.0

        # Check if we have enough data
        state = self._agent_states.get(agent_name)
        if not state or state.total_debates < self.config.min_debates_for_feedback:
            return 0.0

        # Apply decay based on recency
        adjustment = self._selection_adjustments.get(agent_name, 0.0)
        adjustment = self._apply_decay(adjustment, state.last_updated)

        return adjustment

    def get_domain_adjustment(self, agent_name: str, domain: str) -> float:
        """
        Get domain-specific selection adjustment.

        Args:
            agent_name: Name of the agent
            domain: Domain to check

        Returns:
            Domain-specific adjustment factor
        """
        state = self._agent_states.get(agent_name)
        if not state:
            return 0.0

        domain_wins = state.domain_wins.get(domain, 0)
        domain_losses = state.domain_losses.get(domain, 0)
        total = domain_wins + domain_losses

        if total < self.config.min_debates_for_feedback:
            return 0.0

        win_rate = domain_wins / total
        # Centered around 0.5 (50% win rate = no adjustment)
        return (win_rate - 0.5) * self.config.max_adjustment * 2

    def record_timeout(self, agent_name: str) -> None:
        """Record a timeout for an agent."""
        state = self._get_or_create_state(agent_name)
        state.timeouts += 1
        state.last_updated = datetime.now()

        # Timeouts negatively impact selection
        current = self._selection_adjustments.get(agent_name, 0.0)
        penalty = 0.1 * self.config.performance_to_selection_weight
        self._selection_adjustments[agent_name] = max(
            -self.config.max_adjustment,
            current - penalty,
        )

    def record_prediction(
        self,
        agent_name: str,
        predicted_winner: str,
        actual_winner: str,
        confidence: float,
    ) -> None:
        """
        Record a prediction outcome for calibration tracking.

        Args:
            agent_name: Agent that made the prediction
            predicted_winner: Who the agent predicted would win
            actual_winner: Who actually won
            confidence: Agent's confidence in prediction
        """
        state = self._get_or_create_state(agent_name)
        state.total_predictions += 1

        if predicted_winner == actual_winner:
            state.correct_predictions += 1

        state.last_updated = datetime.now()

    def _get_or_create_state(self, agent_name: str) -> AgentFeedbackState:
        """Get or create feedback state for an agent."""
        if agent_name not in self._agent_states:
            self._agent_states[agent_name] = AgentFeedbackState(agent_name=agent_name)
            self.metrics.agents_tracked += 1
        return self._agent_states[agent_name]

    def _compute_adjustment(
        self,
        state: AgentFeedbackState,
        domain: str,
    ) -> float:
        """Compute selection weight adjustment for an agent."""
        if state.total_debates < self.config.min_debates_for_feedback:
            return 0.0

        adjustment = 0.0
        weight = self.config.performance_to_selection_weight

        # Win rate contribution (centered around 0.5)
        win_contribution = (state.win_rate - 0.5) * weight * 2
        adjustment += win_contribution

        # Domain specialization bonus
        domain_rate = state.domain_win_rate(domain)
        if state.domain_wins.get(domain, 0) + state.domain_losses.get(domain, 0) >= 2:
            domain_bonus = (domain_rate - 0.5) * weight
            adjustment += domain_bonus

        # Timeout penalty
        timeout_penalty = state.timeout_rate * weight * 2
        adjustment -= timeout_penalty

        # Calibration bonus (accurate predictions)
        if state.total_predictions >= 3:
            cal_bonus = (state.calibration_score - 0.5) * self.config.calibration_to_elo_weight
            adjustment += cal_bonus

        # Optional: ELO contribution
        if self.elo_system:
            try:
                rating = self.elo_system.get_rating(state.agent_name)
                if rating:
                    # Normalize ELO to adjustment range
                    # Assume 1000-2000 range, center at 1500
                    elo_factor = (rating.elo - 1500) / 500 * weight * 0.5
                    adjustment += elo_factor
            except Exception:
                pass

        # Clamp to max adjustment
        adjustment = max(-self.config.max_adjustment, min(self.config.max_adjustment, adjustment))

        self.metrics.total_adjustment_sum += abs(adjustment)

        return adjustment

    def _apply_decay(self, adjustment: float, last_updated: datetime) -> float:
        """Apply decay to adjustment based on recency."""
        now = datetime.now()
        days_since = (now - last_updated).days

        if days_since > self.config.recency_window_days:
            return 0.0

        decay = self.config.feedback_decay_factor ** days_since
        return adjustment * decay

    def get_agent_state(self, agent_name: str) -> Optional[AgentFeedbackState]:
        """Get feedback state for an agent."""
        return self._agent_states.get(agent_name)

    def get_all_states(self) -> Dict[str, AgentFeedbackState]:
        """Get all agent feedback states."""
        return dict(self._agent_states)

    def get_metrics(self) -> Dict[str, Any]:
        """Get feedback loop metrics."""
        return {
            "debates_processed": self.metrics.debates_processed,
            "adjustments_computed": self.metrics.adjustments_computed,
            "agents_tracked": self.metrics.agents_tracked,
            "average_adjustment": (
                self.metrics.total_adjustment_sum / self.metrics.adjustments_computed
                if self.metrics.adjustments_computed > 0
                else 0.0
            ),
            "last_processed": (
                self.metrics.last_processed.isoformat()
                if self.metrics.last_processed
                else None
            ),
        }

    def reset(self) -> None:
        """Reset all feedback states and metrics."""
        self._agent_states.clear()
        self._selection_adjustments.clear()
        self.metrics = FeedbackLoopMetrics()


__all__ = [
    "SelectionFeedbackLoop",
    "FeedbackLoopConfig",
    "AgentFeedbackState",
    "FeedbackLoopMetrics",
]
