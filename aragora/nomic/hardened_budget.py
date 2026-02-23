"""
Budget enforcement mixin for HardenedOrchestrator.

Extracted from hardened_orchestrator.py for maintainability.
Handles budget tracking, rate limiting, circuit breakers, and
agent outcome recording.
"""

from __future__ import annotations

import asyncio
import collections
import logging
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aragora.nomic.autonomous_orchestrator import AgentAssignment
    from aragora.nomic.hardened_orchestrator import BudgetEnforcementConfig

logger = logging.getLogger(__name__)


class BudgetMixin:
    """Mixin providing budget enforcement methods for HardenedOrchestrator."""

    # These attributes are expected to be set by the host class __init__
    hardened_config: Any
    _budget_spent_usd: float
    _budget_manager: Any
    _budget_id: str | None
    _completed_assignments: list
    _active_assignments: list
    _call_timestamps: collections.deque
    _agent_failure_counts: dict[str, int]
    _agent_success_counts: dict[str, int]
    _agent_open_until: dict[str, float]

    def _init_budget_manager(self, config: BudgetEnforcementConfig) -> None:
        """Initialize BudgetManager integration."""
        try:
            from aragora.billing.budget_manager import get_budget_manager

            self._budget_manager = get_budget_manager()

            if config.budget_id:
                # Use existing budget
                self._budget_id = config.budget_id
            else:
                # Auto-create a budget for this orchestration run
                budget = self._budget_manager.create_budget(
                    org_id=config.org_id,
                    name=f"orchestration-{id(self)}",
                    amount_usd=self.hardened_config.budget_limit_usd or 10.0,
                    description="Auto-created by HardenedOrchestrator",
                )
                self._budget_id = budget.budget_id

            logger.info(
                "budget_manager_initialized budget_id=%s org_id=%s",
                self._budget_id,
                config.org_id,
            )
        except ImportError:
            logger.debug("BudgetManager unavailable, using simple float counter")

    def _check_budget_allows(self, assignment: AgentAssignment) -> bool:
        """Check if the budget allows executing this assignment.

        Uses BudgetManager.can_spend() when configured, falls back to
        simple float counter when budget_limit_usd is set without
        BudgetEnforcementConfig.

        Returns:
            True if assignment may proceed, False if skipped due to budget.
        """
        be_config = self.hardened_config.budget_enforcement
        estimate = be_config.cost_per_subtask_estimate if be_config else 0.10

        # Path 1: BudgetManager integration
        if self._budget_manager is not None and self._budget_id is not None:
            budget = self._budget_manager.get_budget(self._budget_id)
            if budget is None:
                logger.warning("budget_not_found id=%s, allowing", self._budget_id)
                return True

            # Check hard_stop_percent
            hard_stop = be_config.hard_stop_percent if be_config else 1.0
            if budget.usage_percentage >= hard_stop:
                self._skip_assignment(assignment, "budget_exceeded_hard_stop")
                return False

            result = budget.can_spend_extended(estimate)
            if not result.allowed:
                logger.warning(
                    "budget_blocked subtask=%s reason=%s spent=%.2f limit=%.2f",
                    assignment.subtask.id,
                    result.message,
                    budget.spent_usd,
                    budget.amount_usd,
                )
                self._skip_assignment(assignment, "budget_exceeded")
                return False

            return True

        # Path 2: Simple float counter (legacy)
        if self.hardened_config.budget_limit_usd is not None:
            if self._budget_spent_usd >= self.hardened_config.budget_limit_usd:
                logger.warning(
                    "budget_exceeded subtask=%s spent=%.2f limit=%.2f",
                    assignment.subtask.id,
                    self._budget_spent_usd,
                    self.hardened_config.budget_limit_usd,
                )
                self._skip_assignment(assignment, "budget_exceeded")
                return False

        return True

    def _record_budget_spend(
        self,
        assignment: AgentAssignment,
        amount_usd: float | None = None,
    ) -> None:
        """Record spending after a subtask completes.

        Uses BudgetManager.record_spend() when configured, otherwise
        increments the simple float counter.
        """
        be_config = self.hardened_config.budget_enforcement
        cost = amount_usd or (be_config.cost_per_subtask_estimate if be_config else 0.10)

        # Path 1: BudgetManager
        if self._budget_manager is not None and self._budget_id is not None:
            budget = self._budget_manager.get_budget(self._budget_id)
            if budget is not None:
                self._budget_manager.record_spend(
                    org_id=budget.org_id,
                    amount_usd=cost,
                    description=f"subtask:{assignment.subtask.id} ({assignment.subtask.title})",
                )
                logger.info(
                    "budget_spend_recorded subtask=%s cost=%.4f",
                    assignment.subtask.id,
                    cost,
                )
            return

        # Path 2: Simple counter
        self._budget_spent_usd += cost

    def _skip_assignment(self, assignment: AgentAssignment, reason: str) -> None:
        """Mark an assignment as skipped and move to completed list."""
        assignment.status = "skipped"
        assignment.result = {"reason": reason}
        assignment.completed_at = datetime.now(timezone.utc)
        self._completed_assignments.append(assignment)
        if assignment in self._active_assignments:
            self._active_assignments.remove(assignment)

    async def _enforce_rate_limit(self) -> None:
        """Enforce sliding window rate limiting on agent API calls.

        Uses a deque of timestamps to track calls within the current window.
        When the window is full, waits until the oldest call expires.
        """
        config = self.hardened_config
        now = time.monotonic()
        window = config.rate_limit_window_seconds

        # Evict expired timestamps
        while self._call_timestamps and (now - self._call_timestamps[0]) > window:
            self._call_timestamps.popleft()

        # If at capacity, wait for the oldest call to expire
        if len(self._call_timestamps) >= config.rate_limit_max_calls:
            wait_time = window - (now - self._call_timestamps[0])
            if wait_time > 0:
                logger.info(
                    "rate_limit_wait seconds=%.2f calls=%d",
                    wait_time,
                    len(self._call_timestamps),
                )
                await asyncio.sleep(wait_time)
                # Re-evict after waiting
                now = time.monotonic()
                while self._call_timestamps and (now - self._call_timestamps[0]) > window:
                    self._call_timestamps.popleft()

        self._call_timestamps.append(time.monotonic())

    def _check_agent_circuit_breaker(self, agent_type: str) -> bool:
        """Check if the circuit breaker is open for an agent type.

        Uses a simple failure counter with timeout per agent type.
        Circuit opens after ``circuit_breaker_threshold`` consecutive failures
        and stays open for ``circuit_breaker_timeout`` seconds.

        Returns:
            True if the agent is allowed to execute, False if circuit is open.
        """
        open_until = self._agent_open_until.get(agent_type, 0)
        if open_until > 0:
            if time.monotonic() < open_until:
                logger.warning(
                    "circuit_breaker_open agent_type=%s failures=%d",
                    agent_type,
                    self._agent_failure_counts[agent_type],
                )
                return False
            # Timeout expired, reset to half-open (allow one attempt)
            self._agent_failure_counts[agent_type] = 0
            self._agent_open_until.pop(agent_type, None)

        return True

    def _record_agent_outcome(self, agent_type: str, success: bool) -> None:
        """Record success/failure for agent circuit breaker and pool tracking."""
        config = self.hardened_config

        if success:
            self._agent_success_counts[agent_type] += 1
            self._agent_failure_counts[agent_type] = 0
            self._agent_open_until.pop(agent_type, None)
        else:
            self._agent_failure_counts[agent_type] += 1
            if self._agent_failure_counts[agent_type] >= config.circuit_breaker_threshold:
                self._agent_open_until[agent_type] = (
                    time.monotonic() + config.circuit_breaker_timeout
                )
                logger.warning(
                    "circuit_breaker_opened agent_type=%s threshold=%d",
                    agent_type,
                    config.circuit_breaker_threshold,
                )


__all__ = ["BudgetMixin"]
