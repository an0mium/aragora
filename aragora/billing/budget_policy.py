"""
Workspace Budget Policy Engine.

Enforces spending limits per workspace/tenant with configurable policies.
Provides:
- Per-workspace budget policies (monthly, daily, per-debate limits)
- Soft threshold warnings and hard limit enforcement
- Usage summary aggregation
- Integration with CostTracker for real-time cost data

Note: Policies are stored in-memory for now. For production use,
persist policies to the database via BudgetManager or a dedicated
policy table.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class BudgetPolicy:
    """Budget policy configuration for a workspace.

    Attributes:
        monthly_limit: Monthly spending limit in USD. 0 = unlimited.
        daily_limit: Daily spending limit in USD. 0 = unlimited.
        per_debate_limit: Per-debate spending limit in USD. 0 = unlimited.
        alert_threshold_pct: Percentage of budget at which to emit warnings.
        hard_limit: If True, deny operations over limit. If False, warn only.
    """

    monthly_limit: float = 0.0
    daily_limit: float = 0.0
    per_debate_limit: float = 0.0
    alert_threshold_pct: float = 80.0
    hard_limit: bool = False


@dataclass
class BudgetDecision:
    """Result of a budget check.

    Attributes:
        allowed: Whether the operation is permitted.
        reason: Human-readable explanation of the decision.
        usage_pct: Current usage as a percentage of the applicable limit.
        remaining: Remaining budget in USD.
    """

    allowed: bool
    reason: str
    usage_pct: float
    remaining: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "allowed": self.allowed,
            "reason": self.reason,
            "usage_pct": self.usage_pct,
            "remaining": self.remaining,
        }


@dataclass
class UsageSummary:
    """Usage summary for a workspace within a period.

    Attributes:
        workspace_id: The workspace identifier.
        period: The period label (e.g. "monthly", "daily").
        total_cost: Total cost incurred in the period.
        limit: The applicable spending limit.
        usage_pct: Usage as a percentage of the limit.
        debates_count: Number of debates in the period.
    """

    workspace_id: str
    period: str
    total_cost: float
    limit: float
    usage_pct: float
    debates_count: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "workspace_id": self.workspace_id,
            "period": self.period,
            "total_cost": self.total_cost,
            "limit": self.limit,
            "usage_pct": self.usage_pct,
            "debates_count": self.debates_count,
        }


class BudgetPolicyEngine:
    """Enforces budget policies per workspace/tenant.

    Stores policies in-memory and checks estimated costs against
    configured limits. Integrates with CostTracker for real-time
    usage aggregation when available.

    Usage:
        engine = BudgetPolicyEngine()
        await engine.set_policy("ws-1", BudgetPolicy(monthly_limit=100.0, hard_limit=True))
        decision = await engine.check_budget("ws-1", estimated_cost=5.0)
        if not decision.allowed:
            raise Exception(decision.reason)
    """

    def __init__(self) -> None:
        # workspace_id -> BudgetPolicy
        # In-memory store; for production, persist to DB via BudgetManager
        # or a dedicated policy table.
        self._policies: dict[str, BudgetPolicy] = {}

        # workspace_id -> accumulated cost for current period
        # This is a lightweight in-memory tracker. When a CostTracker
        # instance is available, prefer its workspace stats instead.
        self._usage: dict[str, float] = {}

        # workspace_id -> debate count for current period
        self._debate_counts: dict[str, int] = {}

    async def set_policy(self, workspace_id: str, policy: BudgetPolicy) -> None:
        """Set or update the budget policy for a workspace.

        Args:
            workspace_id: Workspace identifier.
            policy: Budget policy to apply.
        """
        self._policies[workspace_id] = policy
        logger.info(
            "Budget policy set for workspace %s: monthly=$%.2f daily=$%.2f per_debate=$%.2f hard=%s",
            workspace_id,
            policy.monthly_limit,
            policy.daily_limit,
            policy.per_debate_limit,
            policy.hard_limit,
        )

    def get_policy(self, workspace_id: str) -> BudgetPolicy | None:
        """Get the budget policy for a workspace.

        Args:
            workspace_id: Workspace identifier.

        Returns:
            The policy if set, or None.
        """
        return self._policies.get(workspace_id)

    async def record_cost(
        self,
        workspace_id: str,
        cost: float,
        debate_id: str | None = None,
    ) -> None:
        """Record a cost against a workspace's usage.

        Args:
            workspace_id: Workspace identifier.
            cost: Cost in USD to record.
            debate_id: Optional debate identifier (increments debate count).
        """
        self._usage[workspace_id] = self._usage.get(workspace_id, 0.0) + cost
        if debate_id is not None:
            self._debate_counts[workspace_id] = self._debate_counts.get(workspace_id, 0) + 1

    async def check_budget(
        self,
        workspace_id: str,
        estimated_cost: float,
    ) -> BudgetDecision:
        """Check whether an estimated cost is allowed under the workspace policy.

        Evaluates the estimated cost against the workspace's configured
        monthly limit. If no policy is set, or the limit is 0 (unlimited),
        the operation is always allowed.

        Args:
            workspace_id: Workspace identifier.
            estimated_cost: Estimated cost in USD for the upcoming operation.

        Returns:
            BudgetDecision indicating whether the operation is allowed.
        """
        policy = self._policies.get(workspace_id)

        # No policy configured -- allow everything
        if policy is None:
            return BudgetDecision(
                allowed=True,
                reason="No budget policy configured",
                usage_pct=0.0,
                remaining=0.0,
            )

        current_usage = self._usage.get(workspace_id, 0.0)

        # Check monthly limit
        if policy.monthly_limit > 0:
            projected = current_usage + estimated_cost
            remaining = max(0.0, policy.monthly_limit - current_usage)
            usage_pct = (current_usage / policy.monthly_limit) * 100.0

            # Hard limit exceeded
            if projected > policy.monthly_limit and policy.hard_limit:
                logger.warning(
                    "Budget DENIED for workspace %s: projected $%.2f exceeds monthly limit $%.2f",
                    workspace_id,
                    projected,
                    policy.monthly_limit,
                )
                return BudgetDecision(
                    allowed=False,
                    reason=f"Monthly budget exceeded: ${current_usage:.2f} used of ${policy.monthly_limit:.2f} limit",
                    usage_pct=usage_pct,
                    remaining=remaining,
                )

            # Soft threshold warning
            if usage_pct >= policy.alert_threshold_pct:
                logger.warning(
                    "Budget WARNING for workspace %s: %.1f%% of monthly limit used ($%.2f / $%.2f)",
                    workspace_id,
                    usage_pct,
                    current_usage,
                    policy.monthly_limit,
                )
                return BudgetDecision(
                    allowed=True,
                    reason=f"Warning: {usage_pct:.1f}% of monthly budget used",
                    usage_pct=usage_pct,
                    remaining=remaining,
                )

            # Within budget
            return BudgetDecision(
                allowed=True,
                reason="Within budget",
                usage_pct=usage_pct,
                remaining=remaining,
            )

        # No monthly limit set (0 = unlimited) -- allow
        return BudgetDecision(
            allowed=True,
            reason="No monthly limit configured",
            usage_pct=0.0,
            remaining=0.0,
        )

    async def get_usage_summary(self, workspace_id: str) -> UsageSummary:
        """Get a usage summary for a workspace.

        Aggregates current usage against the configured monthly policy.

        Args:
            workspace_id: Workspace identifier.

        Returns:
            UsageSummary with current cost, limit, and usage percentage.
        """
        policy = self._policies.get(workspace_id)
        current_usage = self._usage.get(workspace_id, 0.0)
        debate_count = self._debate_counts.get(workspace_id, 0)

        limit = policy.monthly_limit if policy else 0.0
        usage_pct = (current_usage / limit * 100.0) if limit > 0 else 0.0

        return UsageSummary(
            workspace_id=workspace_id,
            period="monthly",
            total_cost=current_usage,
            limit=limit,
            usage_pct=usage_pct,
            debates_count=debate_count,
        )

    def reset_usage(self, workspace_id: str) -> None:
        """Reset usage counters for a workspace (e.g. at period rollover).

        Args:
            workspace_id: Workspace identifier.
        """
        self._usage.pop(workspace_id, None)
        self._debate_counts.pop(workspace_id, None)
        logger.info("Usage reset for workspace %s", workspace_id)

    def reset_all_usage(self) -> None:
        """Reset all usage counters (e.g. at monthly rollover)."""
        self._usage.clear()
        self._debate_counts.clear()
        logger.info("All workspace usage counters reset")


# Module-level singleton
_budget_policy_engine: BudgetPolicyEngine | None = None


def get_budget_policy_engine() -> BudgetPolicyEngine:
    """Get or create the global BudgetPolicyEngine singleton."""
    global _budget_policy_engine
    if _budget_policy_engine is None:
        _budget_policy_engine = BudgetPolicyEngine()
    return _budget_policy_engine


__all__ = [
    "BudgetDecision",
    "BudgetPolicy",
    "BudgetPolicyEngine",
    "UsageSummary",
    "get_budget_policy_engine",
]
