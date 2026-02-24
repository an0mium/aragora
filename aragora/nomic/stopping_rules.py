"""Stopping Rules Engine for the Nomic Loop.

Evaluates a set of configurable rules to determine whether the
autonomous self-improvement pipeline should halt. Each rule is
independent and provides a reason string when triggered.

Usage:
    from aragora.nomic.stopping_rules import StoppingRuleEngine, StoppingConfig

    engine = StoppingRuleEngine()
    should_stop, reason = engine.should_stop(
        telemetry=collector,
        budget=10.0,
        config=StoppingConfig(budget_limit=10.0, max_cycles=20),
    )
    if should_stop:
        print(f"Stopping: {reason}")
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class StoppingConfig:
    """Configuration for the stopping-rule engine."""

    budget_limit: float = 10.0  # Maximum cumulative cost in USD
    min_quality_delta: float = 0.001  # Minimum improvement per cycle
    consecutive_low_delta: int = 3  # Consecutive low-delta cycles to trigger diminishing returns
    max_cycles: int = 50  # Absolute cycle-count cap
    max_duration_hours: float = 8.0  # Wall-clock time limit
    min_goal_confidence: float = 0.7  # Goal proposer confidence floor


class StoppingRuleEngine:
    """Evaluate stopping rules against telemetry and configuration.

    Rules:
        BudgetExhausted       - cumulative cost >= budget_limit
        DiminishingReturns     - quality_delta < threshold for N consecutive cycles
        CycleLimit            - total cycles >= max_cycles
        TimeLimit             - elapsed wall-clock time >= max_duration
        NoViableGoals         - goal proposer returned 0 goals above threshold
    """

    def should_stop(
        self,
        telemetry: Any | None = None,
        budget: float | None = None,
        config: StoppingConfig | None = None,
        goal_proposer: Any | None = None,
        start_time: float | None = None,
    ) -> tuple[bool, str]:
        """Evaluate all stopping rules.

        Args:
            telemetry: CycleTelemetryCollector (optional).
            budget: Explicit remaining budget override; if None, uses
                    telemetry.get_total_cost() against config.budget_limit.
            config: StoppingConfig (uses defaults if None).
            goal_proposer: GoalProposer instance for NoViableGoals rule.
            start_time: Epoch timestamp when the run started (for TimeLimit).

        Returns:
            Tuple of (should_stop: bool, reason: str). ``reason`` is empty
            when should_stop is False.
        """
        config = config or StoppingConfig()

        # Collect per-rule results
        checks: list[tuple[bool, str]] = [
            self._check_budget(telemetry, budget, config),
            self._check_diminishing_returns(telemetry, config),
            self._check_cycle_limit(telemetry, config),
            self._check_time_limit(start_time, config),
            self._check_no_viable_goals(goal_proposer, config),
        ]

        for stop, reason in checks:
            if stop:
                logger.info("stopping_rule_triggered reason=%s", reason)
                return True, reason

        return False, ""

    # ------------------------------------------------------------------
    # Individual rules
    # ------------------------------------------------------------------

    @staticmethod
    def _check_budget(
        telemetry: Any | None,
        budget_override: float | None,
        config: StoppingConfig,
    ) -> tuple[bool, str]:
        """BudgetExhausted: cumulative cost >= budget_limit."""
        if config.budget_limit <= 0:
            return False, ""

        total_cost = 0.0
        if budget_override is not None:
            # budget_override represents remaining budget
            total_cost = config.budget_limit - budget_override
        elif telemetry is not None:
            try:
                total_cost = telemetry.get_total_cost()
            except (AttributeError, RuntimeError, TypeError):
                return False, ""

        if total_cost >= config.budget_limit:
            return (
                True,
                f"BudgetExhausted: cumulative cost ${total_cost:.4f} >= limit ${config.budget_limit:.4f}",
            )
        return False, ""

    @staticmethod
    def _check_diminishing_returns(
        telemetry: Any | None,
        config: StoppingConfig,
    ) -> tuple[bool, str]:
        """DiminishingReturns: quality_delta < threshold for N consecutive cycles."""
        if telemetry is None:
            return False, ""

        try:
            recent = telemetry.get_recent_cycles(n=config.consecutive_low_delta)
        except (AttributeError, RuntimeError, TypeError):
            return False, ""

        if len(recent) < config.consecutive_low_delta:
            return False, ""

        # recent is ordered most-recent-first
        tail = recent[: config.consecutive_low_delta]
        all_low = all(abs(r.quality_delta) < config.min_quality_delta for r in tail)

        if all_low:
            deltas = [r.quality_delta for r in tail]
            return (
                True,
                f"DiminishingReturns: quality_delta < {config.min_quality_delta} "
                f"for {config.consecutive_low_delta} consecutive cycles (deltas={deltas})",
            )
        return False, ""

    @staticmethod
    def _check_cycle_limit(
        telemetry: Any | None,
        config: StoppingConfig,
    ) -> tuple[bool, str]:
        """CycleLimit: total cycle count >= max_cycles."""
        if telemetry is None:
            return False, ""

        try:
            count = telemetry.get_cycle_count()
        except (AttributeError, RuntimeError, TypeError):
            return False, ""

        if count >= config.max_cycles:
            return True, f"CycleLimit: {count} cycles >= max {config.max_cycles}"
        return False, ""

    @staticmethod
    def _check_time_limit(
        start_time: float | None,
        config: StoppingConfig,
    ) -> tuple[bool, str]:
        """TimeLimit: elapsed wall-clock time >= max_duration_hours."""
        if start_time is None or config.max_duration_hours <= 0:
            return False, ""

        elapsed_hours = (time.time() - start_time) / 3600.0
        if elapsed_hours >= config.max_duration_hours:
            return (
                True,
                f"TimeLimit: {elapsed_hours:.2f}h >= max {config.max_duration_hours:.2f}h",
            )
        return False, ""

    @staticmethod
    def _check_no_viable_goals(
        goal_proposer: Any | None,
        config: StoppingConfig,
    ) -> tuple[bool, str]:
        """NoViableGoals: goal proposer returns 0 goals above confidence threshold."""
        if goal_proposer is None:
            return False, ""

        try:
            goals = goal_proposer.propose_goals(
                max_goals=1,
                min_confidence=config.min_goal_confidence,
            )
        except (RuntimeError, TypeError, AttributeError, ValueError) as e:
            logger.debug("no_viable_goals_check_error: %s", e)
            return False, ""

        if len(goals) == 0:
            return (
                True,
                f"NoViableGoals: no goals above confidence {config.min_goal_confidence}",
            )
        return False, ""


__all__ = [
    "StoppingConfig",
    "StoppingRuleEngine",
]
