"""
Outcome Tracker to Complexity Governor Bridge.

Bridges historical outcome data from OutcomeTracker into the ComplexityGovernor's
timeout and complexity calculations, enabling adaptive resource allocation.

This closes the loop between:
1. OutcomeTracker: Records actual implementation outcomes (success, failure, time_to_failure)
2. ComplexityGovernor: Sets timeouts based on static keyword heuristics

By connecting them, we enable:
- Dynamic timeout factors based on actual failure history
- Task complexity reassessment based on historical success rates
- Adaptive resource allocation that learns from outcomes

Usage:
    from aragora.debate.outcome_complexity_bridge import OutcomeComplexityBridge

    bridge = OutcomeComplexityBridge(
        outcome_tracker=tracker,
        learning_rate=0.1,
    )

    # Get adaptive timeout factor for a task
    factor = bridge.get_adaptive_timeout_factor(task, TaskComplexity.COMPLEX)

    # Update after debate
    bridge.record_complexity_outcome(debate_id, complexity, succeeded)
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from aragora.debate.outcome_tracker import OutcomeTracker, ConsensusOutcome

logger = logging.getLogger(__name__)


@dataclass
class ComplexityStats:
    """Statistics for a complexity level."""

    total_debates: int = 0
    successful_debates: int = 0
    timeout_debates: int = 0
    total_time_to_failure: float = 0.0
    avg_rounds: float = 0.0

    @property
    def success_rate(self) -> float:
        """Success rate for this complexity level."""
        if self.total_debates == 0:
            return 0.5  # Neutral
        return self.successful_debates / self.total_debates

    @property
    def timeout_rate(self) -> float:
        """Timeout rate for this complexity level."""
        if self.total_debates == 0:
            return 0.0
        return self.timeout_debates / self.total_debates

    @property
    def avg_time_to_failure(self) -> float:
        """Average time to failure in seconds."""
        failed = self.total_debates - self.successful_debates
        if failed == 0:
            return float("inf")
        return self.total_time_to_failure / failed


@dataclass
class TimeoutAdjustment:
    """Result of timeout factor adjustment."""

    original_factor: float
    adjusted_factor: float
    complexity: str
    reason: str
    confidence: float = 0.0


@dataclass
class OutcomeComplexityBridgeConfig:
    """Configuration for the outcome-complexity bridge."""

    # Minimum debates before adjusting timeouts
    min_debates_for_adjustment: int = 10

    # Learning rate for factor adjustments (0-1)
    learning_rate: float = 0.1

    # Maximum factor increase (prevent runaway)
    max_factor_increase: float = 0.5

    # Maximum factor decrease
    max_factor_decrease: float = 0.3

    # Timeout rate threshold to trigger increase
    timeout_threshold_increase: float = 0.15  # >15% timeouts = increase

    # Success rate threshold to allow decrease
    success_threshold_decrease: float = 0.85  # >85% success = can decrease

    # Weight for time_to_failure in adjustment
    time_to_failure_weight: float = 0.3


@dataclass
class OutcomeComplexityBridge:
    """Bridges OutcomeTracker outcomes to ComplexityGovernor timeout factors.

    Key integration points:
    1. Tracks success/failure rates per complexity level
    2. Adjusts timeout factors based on actual outcome history
    3. Identifies task signals that predict timeouts
    4. Feeds time_to_failure data into timeout calibration
    """

    outcome_tracker: Optional["OutcomeTracker"] = None
    config: OutcomeComplexityBridgeConfig = field(
        default_factory=OutcomeComplexityBridgeConfig
    )

    # Internal state - stats per complexity level
    _complexity_stats: Dict[str, ComplexityStats] = field(
        default_factory=lambda: defaultdict(ComplexityStats), repr=False
    )
    # Current timeout factor adjustments
    _factor_adjustments: Dict[str, float] = field(default_factory=dict, repr=False)
    # Task signal patterns that predict failure
    _failure_signals: Dict[str, int] = field(
        default_factory=lambda: defaultdict(int), repr=False
    )

    def __post_init__(self) -> None:
        """Initialize from outcome tracker history if available."""
        if self.outcome_tracker is not None:
            self._load_from_tracker()

    def _load_from_tracker(self) -> None:
        """Load historical outcomes from tracker to bootstrap statistics."""
        try:
            # Get recent outcomes from tracker
            outcomes = self.outcome_tracker.get_recent_outcomes(limit=100)
            for outcome in outcomes:
                self._process_outcome(outcome)
            logger.info(f"Loaded {len(outcomes)} historical outcomes")
        except Exception as e:
            logger.debug(f"Could not load historical outcomes: {e}")

    def _process_outcome(self, outcome: "ConsensusOutcome") -> None:
        """Process an outcome to update statistics.

        Args:
            outcome: ConsensusOutcome record
        """
        # Infer complexity from outcome metadata or default
        complexity = self._infer_complexity(outcome)
        stats = self._complexity_stats[complexity]

        stats.total_debates += 1
        stats.avg_rounds = (
            (stats.avg_rounds * (stats.total_debates - 1) + outcome.rounds_completed)
            / stats.total_debates
        )

        if outcome.implementation_succeeded:
            stats.successful_debates += 1
        else:
            if outcome.time_to_failure is not None:
                stats.total_time_to_failure += outcome.time_to_failure

            # Check if it was a timeout
            if outcome.failure_reason and "timeout" in outcome.failure_reason.lower():
                stats.timeout_debates += 1

            # Track failure signals from consensus text
            self._extract_failure_signals(outcome.consensus_text)

    def _infer_complexity(self, outcome: "ConsensusOutcome") -> str:
        """Infer complexity level from outcome data.

        Args:
            outcome: ConsensusOutcome record

        Returns:
            Complexity level string
        """
        from aragora.core import TaskComplexity
        from aragora.debate.complexity_governor import classify_task_complexity

        # Use consensus text to classify complexity
        complexity = classify_task_complexity(outcome.consensus_text)
        return complexity.value

    def _extract_failure_signals(self, text: str) -> None:
        """Extract task signals that correlate with failures.

        Args:
            text: Task or consensus text
        """
        if not text:
            return

        text_lower = text.lower()

        # Track signals that appear in failed tasks
        signals = [
            "distributed",
            "concurrent",
            "real-time",
            "optimize",
            "scale",
            "security",
            "formal",
            "prove",
            "complex",
        ]

        for signal in signals:
            if signal in text_lower:
                self._failure_signals[signal] += 1

    def record_complexity_outcome(
        self,
        debate_id: str,
        complexity: str,
        succeeded: bool,
        timeout: bool = False,
        time_to_failure: Optional[float] = None,
        task_text: Optional[str] = None,
    ) -> None:
        """Record an outcome for a complexity level.

        Args:
            debate_id: ID of the debate
            complexity: Complexity level (simple, moderate, complex, unknown)
            succeeded: Whether the debate/implementation succeeded
            timeout: Whether the failure was due to timeout
            time_to_failure: Seconds until first failure (if any)
            task_text: Original task text for signal extraction
        """
        stats = self._complexity_stats[complexity]
        stats.total_debates += 1

        if succeeded:
            stats.successful_debates += 1
        else:
            if timeout:
                stats.timeout_debates += 1
            if time_to_failure is not None:
                stats.total_time_to_failure += time_to_failure
            if task_text:
                self._extract_failure_signals(task_text)

        # Check if we should recalibrate
        if stats.total_debates % 10 == 0:
            self._maybe_recalibrate(complexity)

        logger.debug(
            f"outcome_complexity_recorded complexity={complexity} succeeded={succeeded} "
            f"timeout={timeout}"
        )

    def _maybe_recalibrate(self, complexity: str) -> Optional[TimeoutAdjustment]:
        """Check if timeout factor should be adjusted.

        Args:
            complexity: Complexity level to check

        Returns:
            TimeoutAdjustment if adjustment was made
        """
        stats = self._complexity_stats[complexity]

        if stats.total_debates < self.config.min_debates_for_adjustment:
            return None

        from aragora.debate.complexity_governor import COMPLEXITY_TIMEOUT_FACTORS
        from aragora.core import TaskComplexity

        try:
            complexity_enum = TaskComplexity(complexity)
        except ValueError:
            return None

        original_factor = COMPLEXITY_TIMEOUT_FACTORS.get(complexity_enum, 1.0)
        current_adjustment = self._factor_adjustments.get(complexity, 0.0)
        current_factor = original_factor + current_adjustment

        adjustment = 0.0
        reason = ""

        # High timeout rate → increase timeout
        if stats.timeout_rate > self.config.timeout_threshold_increase:
            adjustment = self.config.learning_rate * (
                stats.timeout_rate - self.config.timeout_threshold_increase
            )
            adjustment = min(adjustment, self.config.max_factor_increase)
            reason = f"high_timeout_rate ({stats.timeout_rate:.1%})"

        # High success rate → can decrease timeout
        elif stats.success_rate > self.config.success_threshold_decrease:
            adjustment = -self.config.learning_rate * (
                stats.success_rate - self.config.success_threshold_decrease
            )
            adjustment = max(adjustment, -self.config.max_factor_decrease)
            reason = f"high_success_rate ({stats.success_rate:.1%})"

        if abs(adjustment) > 0.01:
            self._factor_adjustments[complexity] = current_adjustment + adjustment

            result = TimeoutAdjustment(
                original_factor=current_factor,
                adjusted_factor=current_factor + adjustment,
                complexity=complexity,
                reason=reason,
                confidence=min(1.0, stats.total_debates / 50),
            )

            logger.info(
                f"timeout_factor_adjusted complexity={complexity} "
                f"factor={current_factor:.2f}→{result.adjusted_factor:.2f} "
                f"reason={reason}"
            )

            return result

        return None

    def get_adaptive_timeout_factor(
        self, task: str, base_complexity: Optional[str] = None
    ) -> float:
        """Get adaptive timeout factor for a task.

        Args:
            task: Task description text
            base_complexity: Pre-classified complexity (optional)

        Returns:
            Timeout factor to apply (1.0 = default)
        """
        from aragora.debate.complexity_governor import (
            COMPLEXITY_TIMEOUT_FACTORS,
            classify_task_complexity,
        )
        from aragora.core import TaskComplexity

        # Classify complexity if not provided
        if base_complexity is None:
            complexity = classify_task_complexity(task)
        else:
            try:
                complexity = TaskComplexity(base_complexity)
            except ValueError:
                complexity = TaskComplexity.UNKNOWN

        # Get base factor
        base_factor = COMPLEXITY_TIMEOUT_FACTORS.get(complexity, 1.0)

        # Apply learned adjustment
        adjustment = self._factor_adjustments.get(complexity.value, 0.0)
        adjusted_factor = base_factor + adjustment

        # Apply failure signal boost
        signal_boost = self._compute_signal_boost(task)
        final_factor = adjusted_factor + signal_boost

        # Clamp to reasonable range
        final_factor = max(0.3, min(3.0, final_factor))

        logger.debug(
            f"adaptive_timeout_factor complexity={complexity.value} "
            f"base={base_factor:.2f} adjusted={final_factor:.2f}"
        )

        return final_factor

    def _compute_signal_boost(self, task: str) -> float:
        """Compute timeout boost based on failure-correlated signals.

        Args:
            task: Task description text

        Returns:
            Additional timeout factor (0.0-0.5)
        """
        if not task or not self._failure_signals:
            return 0.0

        task_lower = task.lower()
        total_signal_weight = sum(self._failure_signals.values())

        if total_signal_weight == 0:
            return 0.0

        boost = 0.0
        for signal, count in self._failure_signals.items():
            if signal in task_lower:
                # Weight by how often this signal predicted failure
                signal_weight = count / total_signal_weight
                boost += 0.1 * signal_weight  # Max 0.1 per signal

        return min(0.5, boost)  # Cap total signal boost

    def get_complexity_stats(self, complexity: str) -> Optional[ComplexityStats]:
        """Get statistics for a complexity level.

        Args:
            complexity: Complexity level string

        Returns:
            ComplexityStats if available
        """
        if complexity not in self._complexity_stats:
            return None
        return self._complexity_stats[complexity]

    def get_all_stats(self) -> Dict[str, ComplexityStats]:
        """Get statistics for all complexity levels.

        Returns:
            Dict mapping complexity to stats
        """
        return dict(self._complexity_stats)

    def get_factor_adjustments(self) -> Dict[str, float]:
        """Get current timeout factor adjustments.

        Returns:
            Dict mapping complexity to adjustment delta
        """
        return dict(self._factor_adjustments)

    def get_failure_signals(self) -> Dict[str, int]:
        """Get failure-correlated signals and their counts.

        Returns:
            Dict mapping signal to occurrence count in failures
        """
        return dict(self._failure_signals)

    def get_stats(self) -> Dict[str, Any]:
        """Get bridge statistics.

        Returns:
            Dict with bridge metrics
        """
        return {
            "complexity_levels_tracked": len(self._complexity_stats),
            "total_outcomes_processed": sum(
                s.total_debates for s in self._complexity_stats.values()
            ),
            "factor_adjustments_active": len(self._factor_adjustments),
            "failure_signals_tracked": len(self._failure_signals),
        }


def create_outcome_complexity_bridge(
    outcome_tracker: Optional["OutcomeTracker"] = None,
    **config_kwargs: Any,
) -> OutcomeComplexityBridge:
    """Create and configure an OutcomeComplexityBridge.

    Args:
        outcome_tracker: OutcomeTracker instance
        **config_kwargs: Additional configuration options

    Returns:
        Configured OutcomeComplexityBridge instance
    """
    config = OutcomeComplexityBridgeConfig(**config_kwargs)
    return OutcomeComplexityBridge(
        outcome_tracker=outcome_tracker,
        config=config,
    )


__all__ = [
    "OutcomeComplexityBridge",
    "OutcomeComplexityBridgeConfig",
    "ComplexityStats",
    "TimeoutAdjustment",
    "create_outcome_complexity_bridge",
]
