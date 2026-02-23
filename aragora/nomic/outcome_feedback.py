"""Outcome Feedback Bridge — empirically-driven Nomic Loop goals.

Connects OutcomeVerifier systematic error patterns to the Nomic Loop's
MetaPlanner, generating targeted improvement goals based on real-world
decision outcomes rather than heuristic guesses.

When the OutcomeVerifier identifies that agents are systematically
miscalibrated in a domain (e.g., "security assessments are 20%
overconfident"), this bridge:
1. Converts the pattern into an ImprovementSuggestion
2. Queues it in the ImprovementQueue for MetaPlanner consumption
3. Provides priority scoring based on pattern severity

This closes the ultimate feedback loop:
    Debate → Decision → Real-world outcome → Calibration update
    → Systematic error detection → Nomic Loop improvement goal
    → Code/config change → Better future debates

Usage:
    bridge = OutcomeFeedbackBridge()
    goals = bridge.generate_improvement_goals()
    # → [ImprovementSuggestion(category="calibration", ...)]

    # Or as a scheduled job:
    bridge.run_feedback_cycle()
"""

from __future__ import annotations

__all__ = [
    "OutcomeFeedbackBridge",
    "FeedbackGoal",
]

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class FeedbackGoal:
    """An empirically-derived improvement goal."""

    domain: str
    agent: str
    goal_type: str  # "reduce_overconfidence", "increase_accuracy", "domain_training"
    severity: float  # 0.0-1.0, how urgent
    description: str
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def priority(self) -> int:
        """Convert severity to integer priority (1-10)."""
        return max(1, min(10, int(self.severity * 10)))


class OutcomeFeedbackBridge:
    """Bridges outcome verification patterns to Nomic Loop improvement goals.

    Reads systematic error data from OutcomeVerifier and converts it
    into actionable ImprovementSuggestion objects for the MetaPlanner.
    """

    def __init__(
        self,
        min_verifications: int = 5,
        overconfidence_threshold: float = 0.1,
        low_accuracy_threshold: float = 0.6,
    ):
        self.min_verifications = min_verifications
        self.overconfidence_threshold = overconfidence_threshold
        self.low_accuracy_threshold = low_accuracy_threshold

    def generate_improvement_goals(self) -> list[FeedbackGoal]:
        """Analyze outcome patterns and generate improvement goals.

        Returns:
            List of FeedbackGoal objects, sorted by severity (most urgent first)
        """
        try:
            from aragora.debate.outcome_verifier import OutcomeVerifier

            verifier = OutcomeVerifier()
        except (ImportError, OSError) as e:
            logger.debug("OutcomeVerifier not available: %s", e)
            return []

        errors = verifier.get_systematic_errors(
            min_count=self.min_verifications,
            min_overconfidence=self.overconfidence_threshold,
        )

        goals = []
        for error in errors:
            goals.extend(self._error_to_goals(error))

        # Sort by severity (most urgent first)
        goals.sort(key=lambda g: g.severity, reverse=True)
        return goals

    def queue_improvement_suggestions(self) -> int:
        """Generate goals and queue them as ImprovementSuggestions.

        Returns:
            Number of suggestions queued
        """
        goals = self.generate_improvement_goals()
        if not goals:
            return 0

        try:
            from aragora.nomic.improvement_queue import (
                ImprovementSuggestion,
                get_improvement_queue,
            )

            queue = get_improvement_queue()
            queued = 0

            for goal in goals:
                suggestion = ImprovementSuggestion(
                    debate_id=f"outcome-feedback-{goal.domain}-{goal.agent}",
                    task=goal.description,
                    suggestion=self._goal_to_action(goal),
                    category="calibration",
                    confidence=goal.severity,
                )
                queue.enqueue(suggestion)
                queued += 1

            logger.info("Queued %d outcome-driven improvement suggestions", queued)
            return queued
        except (ImportError, OSError) as e:
            logger.debug("ImprovementQueue not available: %s", e)
            return 0

    def run_feedback_cycle(self) -> dict[str, Any]:
        """Run a full feedback cycle: analyze → generate → queue.

        Returns:
            Summary of the cycle results
        """
        goals = self.generate_improvement_goals()
        queued = self.queue_improvement_suggestions()

        # Also compute calibration adjustment for Trickster
        adjustment = self._get_trickster_adjustment()

        return {
            "goals_generated": len(goals),
            "suggestions_queued": queued,
            "trickster_adjustment": adjustment,
            "domains_flagged": list({g.domain for g in goals}),
            "agents_flagged": list({g.agent for g in goals}),
        }

    def _error_to_goals(self, error: dict[str, Any]) -> list[FeedbackGoal]:
        """Convert a systematic error pattern into improvement goals."""
        goals = []
        domain = error["domain"]
        agent = error["agent"]
        overconfidence = error.get("overconfidence", 0.0)
        success_rate = error.get("success_rate", 0.0)
        avg_brier = error.get("avg_brier_score", 0.0)
        total = error.get("total_verifications", 0)

        # Goal 1: Reduce overconfidence
        if overconfidence > self.overconfidence_threshold:
            severity = min(1.0, overconfidence * 3)  # 0.33 overconfidence → max severity
            goals.append(
                FeedbackGoal(
                    domain=domain,
                    agent=agent,
                    goal_type="reduce_overconfidence",
                    severity=severity,
                    description=(
                        f"Agent '{agent}' is {overconfidence:.0%} overconfident in "
                        f"'{domain}' domain ({total} decisions, {success_rate:.0%} success "
                        f"vs {error.get('avg_confidence', 0):.0%} avg confidence)"
                    ),
                    metrics={
                        "overconfidence": overconfidence,
                        "success_rate": success_rate,
                        "avg_confidence": error.get("avg_confidence", 0),
                        "total_verifications": total,
                    },
                )
            )

        # Goal 2: Increase accuracy in low-performing domains
        if success_rate < self.low_accuracy_threshold:
            severity = min(1.0, (self.low_accuracy_threshold - success_rate) * 5)
            goals.append(
                FeedbackGoal(
                    domain=domain,
                    agent=agent,
                    goal_type="increase_accuracy",
                    severity=severity,
                    description=(
                        f"Agent '{agent}' has low accuracy ({success_rate:.0%}) in "
                        f"'{domain}' domain — consider domain-specific training or "
                        f"reducing agent's selection weight for this domain"
                    ),
                    metrics={
                        "success_rate": success_rate,
                        "avg_brier": avg_brier,
                        "total_verifications": total,
                    },
                )
            )

        # Goal 3: Domain training for consistently poor Brier scores
        if avg_brier > 0.3 and total >= self.min_verifications * 2:
            severity = min(1.0, (avg_brier - 0.3) * 5)
            goals.append(
                FeedbackGoal(
                    domain=domain,
                    agent=agent,
                    goal_type="domain_training",
                    severity=severity,
                    description=(
                        f"Agent '{agent}' has poor calibration (Brier={avg_brier:.2f}) in "
                        f"'{domain}' domain — needs domain-specific prompt tuning or "
                        f"temperature scaling adjustment"
                    ),
                    metrics={
                        "avg_brier": avg_brier,
                        "total_verifications": total,
                    },
                )
            )

        return goals

    def _goal_to_action(self, goal: FeedbackGoal) -> str:
        """Convert a FeedbackGoal into an actionable suggestion string."""
        if goal.goal_type == "reduce_overconfidence":
            return (
                f"Apply temperature scaling for '{goal.agent}' in '{goal.domain}' domain. "
                f"Current overconfidence: {goal.metrics.get('overconfidence', 0):.0%}. "
                f"Consider: CalibrationTracker.auto_tune_agent('{goal.agent}') or "
                f"increasing Trickster sensitivity for this domain."
            )
        elif goal.goal_type == "increase_accuracy":
            return (
                f"Reduce selection weight for '{goal.agent}' in '{goal.domain}' domain "
                f"(current success rate: {goal.metrics.get('success_rate', 0):.0%}). "
                f"Consider adding domain-specific knowledge to KM or switching "
                f"to a more capable agent for this domain."
            )
        elif goal.goal_type == "domain_training":
            return (
                f"Tune '{goal.agent}' calibration for '{goal.domain}' domain. "
                f"Run: CalibrationTracker.auto_tune_agent('{goal.agent}') with "
                f"domain='{goal.domain}' to compute optimal temperature scaling."
            )
        return goal.description

    def _get_trickster_adjustment(self) -> float:
        """Get Trickster sensitivity adjustment from outcome patterns."""
        try:
            from aragora.debate.outcome_tracker import OutcomeTracker

            tracker = OutcomeTracker()
            return tracker.get_calibration_adjustment()
        except (ImportError, OSError):
            return 1.0
