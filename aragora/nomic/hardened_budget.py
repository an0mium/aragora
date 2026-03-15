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

logger = logging.getLogger("aragora.nomic.hardened_orchestrator")


def _safe_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


class BudgetMixin:
    """Mixin providing budget enforcement methods for HardenedOrchestrator."""

    # These attributes are expected to be set by the host class __init__
    hardened_config: Any
    _budget_spent_usd: float
    _budget_reserved_usd: float
    _budget_reservations: dict[str, float]
    _budget_lock: Any
    _budget_manager: Any
    _budget_id: str | None
    _total_cost_usd: float
    _completed_assignments: list
    _active_assignments: list
    _call_timestamps: collections.deque
    _agent_failure_counts: dict[str, int]
    _agent_success_counts: dict[str, int]
    _agent_open_until: dict[str, float]

    @staticmethod
    def _reservation_key(assignment: AgentAssignment) -> str:
        return assignment.subtask.id

    def _estimated_assignment_cost(self, assignment: AgentAssignment) -> float:
        be_config = self.hardened_config.budget_enforcement
        return float(be_config.cost_per_subtask_estimate if be_config else 0.10)

    def _reset_budget_tracking(self) -> None:
        """Reset committed and reserved budget for a new orchestrator run."""
        with self._budget_lock:
            self._budget_spent_usd = 0.0
            self._budget_reserved_usd = 0.0
            self._budget_reservations.clear()
            self._total_cost_usd = 0.0

    def _budget_snapshot(
        self,
        *,
        spent_usd: float | None = None,
        limit_usd: float | None = None,
    ) -> dict[str, float | bool | None]:
        with self._budget_lock:
            spent = max(float(spent_usd or 0.0), float(self._budget_spent_usd or 0.0))
            reserved = float(self._budget_reserved_usd or 0.0)
        limit = self.hardened_config.budget_limit_usd if limit_usd is None else limit_usd
        committed = spent + reserved
        remaining = None if limit is None else max(0.0, float(limit) - committed)
        return {
            "spent_usd": round(spent, 4),
            "reserved_usd": round(reserved, 4),
            "committed_usd": round(committed, 4),
            "remaining_usd": None if remaining is None else round(remaining, 4),
            "limit_usd": None if limit is None else round(float(limit), 4),
            "exhausted": bool(limit is not None and committed >= float(limit)),
        }

    def _reserve_budget(self, assignment: AgentAssignment, amount_usd: float) -> None:
        key = self._reservation_key(assignment)
        with self._budget_lock:
            if key in self._budget_reservations:
                return
            self._budget_reservations[key] = amount_usd
            self._budget_reserved_usd += amount_usd
        snapshot = self._budget_snapshot()
        self._emit_event(  # type: ignore[attr-defined]
            "budget_reserved",
            subtask=assignment.subtask.id,
            reserved=round(amount_usd, 4),
            committed=snapshot["committed_usd"],
            remaining=snapshot["remaining_usd"],
            limit=snapshot["limit_usd"],
        )

    def _cancel_budget_reservation(self, assignment: AgentAssignment, *, reason: str) -> None:
        key = self._reservation_key(assignment)
        with self._budget_lock:
            reserved = self._budget_reservations.pop(key, 0.0)
            if reserved > 0:
                self._budget_reserved_usd = max(0.0, self._budget_reserved_usd - reserved)
            else:
                reserved = 0.0
        if reserved <= 0:
            return
        snapshot = self._budget_snapshot()
        self._emit_event(  # type: ignore[attr-defined]
            "budget_reservation_released",
            subtask=assignment.subtask.id,
            released=round(reserved, 4),
            committed=snapshot["committed_usd"],
            remaining=snapshot["remaining_usd"],
            limit=snapshot["limit_usd"],
            reason=reason,
        )

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
        estimate = self._estimated_assignment_cost(assignment)
        reservation_key = self._reservation_key(assignment)

        with self._budget_lock:
            existing_reservation = self._budget_reservations.get(reservation_key)
        if existing_reservation is not None:
            return True

        # Path 1: BudgetManager integration
        if self._budget_manager is not None and self._budget_id is not None:
            budget = self._budget_manager.get_budget(self._budget_id)
            if budget is None:
                logger.warning("budget_not_found id=%s, allowing", self._budget_id)
                self._reserve_budget(assignment, estimate)
                return True

            # Check hard_stop_percent
            hard_stop = be_config.hard_stop_percent if be_config else 1.0
            usage_percentage = _safe_float(getattr(budget, "usage_percentage", 0.0))
            if usage_percentage >= hard_stop:
                self._skip_assignment(assignment, "budget_exceeded_hard_stop")
                return False

            limit_usd = _safe_float(
                getattr(budget, "amount_usd", self.hardened_config.budget_limit_usd or 0.0),
                _safe_float(self.hardened_config.budget_limit_usd, 0.0),
            )
            spent_usd = max(
                _safe_float(getattr(budget, "spent_usd", 0.0)),
                _safe_float(self._budget_spent_usd, 0.0),
            )
            projected_total = spent_usd + float(self._budget_reserved_usd or 0.0) + estimate
            if limit_usd > 0 and projected_total > limit_usd + 1e-9:
                logger.warning(
                    "budget_blocked_projected subtask=%s projected=%.2f spent=%.2f reserved=%.2f limit=%.2f",
                    assignment.subtask.id,
                    projected_total,
                    spent_usd,
                    self._budget_reserved_usd,
                    limit_usd,
                )
                self._skip_assignment(assignment, "budget_exceeded")
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

            self._reserve_budget(assignment, estimate)
            return True

        # Path 2: Simple float counter (legacy)
        if self.hardened_config.budget_limit_usd is not None:
            projected_total = self._budget_spent_usd + self._budget_reserved_usd + estimate
            if projected_total > self.hardened_config.budget_limit_usd + 1e-9:
                logger.warning(
                    "budget_exceeded subtask=%s projected=%.2f spent=%.2f reserved=%.2f limit=%.2f",
                    assignment.subtask.id,
                    projected_total,
                    self._budget_spent_usd,
                    self._budget_reserved_usd,
                    self.hardened_config.budget_limit_usd,
                )
                self._skip_assignment(assignment, "budget_exceeded")
                return False

        self._reserve_budget(assignment, estimate)
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
        cost = float(amount_usd or (be_config.cost_per_subtask_estimate if be_config else 0.10))
        reservation_key = self._reservation_key(assignment)
        with self._budget_lock:
            reserved = self._budget_reservations.pop(reservation_key, 0.0)
            if reserved > 0:
                self._budget_reserved_usd = max(0.0, self._budget_reserved_usd - reserved)
            self._budget_spent_usd += cost
            self._total_cost_usd = self._budget_spent_usd

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
        snapshot = self._budget_snapshot()

        # Emit budget tracking event
        self._emit_event(  # type: ignore[attr-defined]
            "budget_update",
            subtask=assignment.subtask.id,
            cost=round(cost, 4),
            released_reservation=round(reserved, 4),
            total_spent=snapshot["spent_usd"],
            reserved=snapshot["reserved_usd"],
            remaining=snapshot["remaining_usd"],
            limit=snapshot["limit_usd"],
        )

    def _skip_assignment(self, assignment: AgentAssignment, reason: str) -> None:
        """Mark an assignment as skipped and move to completed list."""
        self._cancel_budget_reservation(assignment, reason=reason)
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
