"""
Channel budget enforcement for Slack/Teams initiated debates.

Checks workspace and per-channel budgets before starting debates,
warns when approaching limits, and blocks when exceeded.

Usage:
    enforcer = ChannelBudgetEnforcer()

    # Check before starting a debate
    result = await enforcer.check_budget("slack", "C01ABC", "T01XYZ")
    if result.blocked:
        # Post explanation to channel
        ...
    elif result.warning:
        # Post warning but allow
        ...

    # Record spend after debate completes
    await enforcer.record_spend("slack", "C01ABC", "T01XYZ", cost_usd=0.15)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# Default budget limits
DEFAULT_WORKSPACE_BUDGET_USD = 50.0
DEFAULT_CHANNEL_BUDGET_USD = 10.0
WARNING_THRESHOLD = 0.80  # 80% of budget


@dataclass
class BudgetCheckResult:
    """Result of a budget check."""

    allowed: bool = True
    blocked: bool = False
    warning: bool = False
    remaining_usd: float = 0.0
    budget_usd: float = 0.0
    spent_usd: float = 0.0
    utilization: float = 0.0
    message: str = ""

    @property
    def utilization_pct(self) -> str:
        """Human-readable utilization percentage."""
        return f"{self.utilization:.0%}"


@dataclass
class ChannelSpendRecord:
    """Tracks spending for a channel within a workspace."""

    channel_key: str  # "platform:workspace:channel"
    total_usd: float = 0.0
    debate_count: int = 0
    last_spend_at: str = ""
    budget_limit_usd: float = DEFAULT_CHANNEL_BUDGET_USD


class ChannelBudgetEnforcer:
    """Enforces budget limits for channel-initiated debates.

    Tracks per-channel and per-workspace spending, warns when approaching
    limits (80% default), and blocks debates when budget is exceeded.

    Args:
        workspace_budget_usd: Default workspace budget in USD.
        channel_budget_usd: Default per-channel budget in USD.
        warning_threshold: Fraction of budget triggering warning (default 0.80).
        budget_manager: Optional BudgetManager instance for persistence.
    """

    def __init__(
        self,
        workspace_budget_usd: float = DEFAULT_WORKSPACE_BUDGET_USD,
        channel_budget_usd: float = DEFAULT_CHANNEL_BUDGET_USD,
        warning_threshold: float = WARNING_THRESHOLD,
        budget_manager: Any | None = None,
    ) -> None:
        self._workspace_budget = workspace_budget_usd
        self._channel_budget = channel_budget_usd
        self._warning_threshold = warning_threshold
        self._budget_manager = budget_manager
        self._channel_spend: dict[str, ChannelSpendRecord] = {}
        self._workspace_spend: dict[str, float] = {}  # workspace_key -> total_usd

    def _channel_key(self, platform: str, channel_id: str, workspace_id: str) -> str:
        """Build a unique key for a channel."""
        return f"{platform}:{workspace_id}:{channel_id}"

    def _workspace_key(self, platform: str, workspace_id: str) -> str:
        """Build a unique key for a workspace."""
        return f"{platform}:{workspace_id}"

    async def check_budget(
        self,
        platform: str,
        channel_id: str,
        workspace_id: str,
        estimated_cost_usd: float = 0.0,
    ) -> BudgetCheckResult:
        """Check if a debate can proceed given budget constraints.

        Checks both per-channel and per-workspace limits.

        Args:
            platform: "slack" or "teams".
            channel_id: Channel or conversation ID.
            workspace_id: Workspace/team ID.
            estimated_cost_usd: Estimated cost of the debate (optional).

        Returns:
            BudgetCheckResult indicating whether the debate can proceed.
        """
        ch_key = self._channel_key(platform, channel_id, workspace_id)
        ws_key = self._workspace_key(platform, workspace_id)

        # Get current spend
        channel_record = self._channel_spend.get(ch_key)
        channel_spent = channel_record.total_usd if channel_record else 0.0
        channel_budget = (
            channel_record.budget_limit_usd if channel_record else self._channel_budget
        )

        workspace_spent = self._workspace_spend.get(ws_key, 0.0)

        # Try external budget manager
        if self._budget_manager is not None:
            try:
                external_spent = self._get_external_spend(workspace_id)
                if external_spent is not None:
                    workspace_spent = max(workspace_spent, external_spent)
            except (ImportError, RuntimeError, OSError, TypeError, ValueError) as exc:
                logger.debug("External budget check failed: %s", exc)

        # Check workspace budget first (more restrictive)
        ws_utilization = workspace_spent / self._workspace_budget if self._workspace_budget > 0 else 0.0
        projected_ws = workspace_spent + estimated_cost_usd

        if projected_ws > self._workspace_budget and self._workspace_budget > 0:
            return BudgetCheckResult(
                allowed=False,
                blocked=True,
                remaining_usd=max(0.0, self._workspace_budget - workspace_spent),
                budget_usd=self._workspace_budget,
                spent_usd=workspace_spent,
                utilization=ws_utilization,
                message=(
                    f"Workspace budget exceeded. "
                    f"Spent ${workspace_spent:.2f} of ${self._workspace_budget:.2f} limit. "
                    f"Please contact your administrator to increase the budget."
                ),
            )

        # Check channel budget
        ch_utilization = channel_spent / channel_budget if channel_budget > 0 else 0.0
        projected_ch = channel_spent + estimated_cost_usd

        if projected_ch > channel_budget and channel_budget > 0:
            return BudgetCheckResult(
                allowed=False,
                blocked=True,
                remaining_usd=max(0.0, channel_budget - channel_spent),
                budget_usd=channel_budget,
                spent_usd=channel_spent,
                utilization=ch_utilization,
                message=(
                    f"Channel budget exceeded. "
                    f"Spent ${channel_spent:.2f} of ${channel_budget:.2f} limit. "
                    f"Try another channel or contact your administrator."
                ),
            )

        # Check for warning threshold
        effective_utilization = max(ws_utilization, ch_utilization)
        warning = effective_utilization >= self._warning_threshold

        remaining = min(
            self._workspace_budget - workspace_spent if self._workspace_budget > 0 else float("inf"),
            channel_budget - channel_spent if channel_budget > 0 else float("inf"),
        )
        if remaining == float("inf"):
            remaining = 0.0

        result = BudgetCheckResult(
            allowed=True,
            blocked=False,
            warning=warning,
            remaining_usd=max(0.0, remaining),
            budget_usd=channel_budget,
            spent_usd=channel_spent,
            utilization=effective_utilization,
        )

        if warning:
            result.message = (
                f"Approaching budget limit: {effective_utilization:.0%} utilized. "
                f"${remaining:.2f} remaining."
            )

        return result

    async def record_spend(
        self,
        platform: str,
        channel_id: str,
        workspace_id: str,
        cost_usd: float,
        debate_id: str = "",
    ) -> None:
        """Record spending for a completed debate.

        Args:
            platform: "slack" or "teams".
            channel_id: Channel or conversation ID.
            workspace_id: Workspace/team ID.
            cost_usd: Actual cost of the debate in USD.
            debate_id: Optional debate ID for tracking.
        """
        if cost_usd <= 0:
            return

        ch_key = self._channel_key(platform, channel_id, workspace_id)
        ws_key = self._workspace_key(platform, workspace_id)

        # Update channel spend
        if ch_key not in self._channel_spend:
            self._channel_spend[ch_key] = ChannelSpendRecord(
                channel_key=ch_key,
                budget_limit_usd=self._channel_budget,
            )

        record = self._channel_spend[ch_key]
        record.total_usd += cost_usd
        record.debate_count += 1
        record.last_spend_at = datetime.now(timezone.utc).isoformat()

        # Update workspace spend
        self._workspace_spend[ws_key] = self._workspace_spend.get(ws_key, 0.0) + cost_usd

        logger.info(
            "Recorded spend $%.4f for %s (channel total: $%.2f, workspace total: $%.2f)",
            cost_usd,
            ch_key,
            record.total_usd,
            self._workspace_spend[ws_key],
        )

    def set_channel_budget(
        self,
        platform: str,
        channel_id: str,
        workspace_id: str,
        budget_usd: float,
    ) -> None:
        """Set a custom budget for a specific channel.

        Args:
            platform: "slack" or "teams".
            channel_id: Channel or conversation ID.
            workspace_id: Workspace/team ID.
            budget_usd: Budget limit in USD.
        """
        ch_key = self._channel_key(platform, channel_id, workspace_id)
        if ch_key not in self._channel_spend:
            self._channel_spend[ch_key] = ChannelSpendRecord(
                channel_key=ch_key,
                budget_limit_usd=budget_usd,
            )
        else:
            self._channel_spend[ch_key].budget_limit_usd = budget_usd

        logger.info("Set channel budget for %s to $%.2f", ch_key, budget_usd)

    def get_channel_spend(
        self,
        platform: str,
        channel_id: str,
        workspace_id: str,
    ) -> ChannelSpendRecord | None:
        """Get spend tracking data for a channel.

        Args:
            platform: "slack" or "teams".
            channel_id: Channel or conversation ID.
            workspace_id: Workspace/team ID.

        Returns:
            ChannelSpendRecord or None if no spending recorded.
        """
        ch_key = self._channel_key(platform, channel_id, workspace_id)
        return self._channel_spend.get(ch_key)

    def get_workspace_spend(
        self,
        platform: str,
        workspace_id: str,
    ) -> float:
        """Get total workspace spending.

        Args:
            platform: "slack" or "teams".
            workspace_id: Workspace/team ID.

        Returns:
            Total spend in USD.
        """
        ws_key = self._workspace_key(platform, workspace_id)
        return self._workspace_spend.get(ws_key, 0.0)

    def _get_external_spend(self, workspace_id: str) -> float | None:
        """Query external budget manager for workspace spend."""
        if self._budget_manager is None:
            return None
        try:
            if hasattr(self._budget_manager, "get_usage"):
                usage = self._budget_manager.get_usage(workspace_id)
                if isinstance(usage, dict):
                    return usage.get("total_cost_usd", 0.0)
                return float(usage) if usage is not None else None
        except (TypeError, ValueError, AttributeError):
            pass
        return None

    def reset_channel(
        self,
        platform: str,
        channel_id: str,
        workspace_id: str,
    ) -> None:
        """Reset spending for a channel (e.g., at period boundary).

        Args:
            platform: "slack" or "teams".
            channel_id: Channel or conversation ID.
            workspace_id: Workspace/team ID.
        """
        ch_key = self._channel_key(platform, channel_id, workspace_id)
        if ch_key in self._channel_spend:
            budget = self._channel_spend[ch_key].budget_limit_usd
            self._channel_spend[ch_key] = ChannelSpendRecord(
                channel_key=ch_key,
                budget_limit_usd=budget,
            )
            logger.info("Reset spend for channel %s", ch_key)

    def reset_workspace(self, platform: str, workspace_id: str) -> None:
        """Reset spending for a workspace.

        Args:
            platform: "slack" or "teams".
            workspace_id: Workspace/team ID.
        """
        ws_key = self._workspace_key(platform, workspace_id)
        self._workspace_spend.pop(ws_key, None)
        logger.info("Reset spend for workspace %s", ws_key)


__all__ = [
    "BudgetCheckResult",
    "ChannelBudgetEnforcer",
    "ChannelSpendRecord",
]
