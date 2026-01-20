"""
Cost Tracking for Workspace and Agent-Level Attribution.

Provides granular cost tracking with:
- Per-workspace cost attribution
- Per-agent cost breakdown
- Budget management with alerts
- Cost projections and anomaly detection
- Real-time cost streaming
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
from uuid import uuid4

from aragora.billing.usage import (
    UsageEvent,
    UsageEventType,
    UsageTracker,
    calculate_token_cost,
)

logger = logging.getLogger(__name__)


class BudgetAlertLevel(str, Enum):
    """Budget alert severity levels."""

    INFO = "info"  # 50% of budget
    WARNING = "warning"  # 75% of budget
    CRITICAL = "critical"  # 90% of budget
    EXCEEDED = "exceeded"  # Over budget


class CostGranularity(str, Enum):
    """Cost aggregation granularity."""

    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


@dataclass
class TokenUsage:
    """Token usage for a single API call."""

    id: str = field(default_factory=lambda: str(uuid4()))
    workspace_id: str = ""
    agent_id: str = ""
    agent_name: str = ""
    debate_id: Optional[str] = None
    session_id: Optional[str] = None

    # Provider info
    provider: str = ""
    model: str = ""

    # Token counts
    tokens_in: int = 0
    tokens_out: int = 0
    tokens_cached: int = 0  # Cached/prompt caching tokens (if supported)

    # Computed cost
    cost_usd: Decimal = Decimal("0")

    # Timing
    latency_ms: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Additional metadata
    operation: str = ""  # e.g., "debate_round", "analysis", "summarization"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def calculate_cost(self) -> Decimal:
        """Calculate cost based on token usage."""
        self.cost_usd = calculate_token_cost(
            self.provider, self.model, self.tokens_in, self.tokens_out
        )
        return self.cost_usd

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "workspace_id": self.workspace_id,
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "debate_id": self.debate_id,
            "session_id": self.session_id,
            "provider": self.provider,
            "model": self.model,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "tokens_cached": self.tokens_cached,
            "cost_usd": str(self.cost_usd),
            "latency_ms": self.latency_ms,
            "timestamp": self.timestamp.isoformat(),
            "operation": self.operation,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TokenUsage:
        """Create from dictionary."""
        usage = cls(
            id=data.get("id", str(uuid4())),
            workspace_id=data.get("workspace_id", ""),
            agent_id=data.get("agent_id", ""),
            agent_name=data.get("agent_name", ""),
            debate_id=data.get("debate_id"),
            session_id=data.get("session_id"),
            provider=data.get("provider", ""),
            model=data.get("model", ""),
            tokens_in=data.get("tokens_in", 0),
            tokens_out=data.get("tokens_out", 0),
            tokens_cached=data.get("tokens_cached", 0),
            cost_usd=Decimal(data.get("cost_usd", "0")),
            latency_ms=data.get("latency_ms", 0.0),
            operation=data.get("operation", ""),
            metadata=data.get("metadata", {}),
        )
        if "timestamp" in data and data["timestamp"]:
            if isinstance(data["timestamp"], str):
                usage.timestamp = datetime.fromisoformat(data["timestamp"])
            else:
                usage.timestamp = data["timestamp"]
        return usage


@dataclass
class Budget:
    """Budget configuration for a workspace or organization."""

    id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    workspace_id: Optional[str] = None
    org_id: Optional[str] = None

    # Budget limits
    monthly_limit_usd: Optional[Decimal] = None
    daily_limit_usd: Optional[Decimal] = None
    per_debate_limit_usd: Optional[Decimal] = None
    per_agent_limit_usd: Optional[Decimal] = None

    # Alert thresholds (as percentages)
    alert_threshold_50: bool = True
    alert_threshold_75: bool = True
    alert_threshold_90: bool = True

    # Current spend (for quick access)
    current_monthly_spend: Decimal = Decimal("0")
    current_daily_spend: Decimal = Decimal("0")

    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def check_alert_level(self) -> Optional[BudgetAlertLevel]:
        """Check if budget threshold is exceeded."""
        if self.monthly_limit_usd is None or self.monthly_limit_usd <= 0:
            return None

        percentage = (self.current_monthly_spend / self.monthly_limit_usd) * 100

        if percentage >= 100:
            return BudgetAlertLevel.EXCEEDED
        elif percentage >= 90 and self.alert_threshold_90:
            return BudgetAlertLevel.CRITICAL
        elif percentage >= 75 and self.alert_threshold_75:
            return BudgetAlertLevel.WARNING
        elif percentage >= 50 and self.alert_threshold_50:
            return BudgetAlertLevel.INFO

        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "workspace_id": self.workspace_id,
            "org_id": self.org_id,
            "monthly_limit_usd": str(self.monthly_limit_usd) if self.monthly_limit_usd else None,
            "daily_limit_usd": str(self.daily_limit_usd) if self.daily_limit_usd else None,
            "per_debate_limit_usd": str(self.per_debate_limit_usd) if self.per_debate_limit_usd else None,
            "per_agent_limit_usd": str(self.per_agent_limit_usd) if self.per_agent_limit_usd else None,
            "current_monthly_spend": str(self.current_monthly_spend),
            "current_daily_spend": str(self.current_daily_spend),
            "alert_level": self.check_alert_level().value if self.check_alert_level() else None,
        }


@dataclass
class BudgetAlert:
    """A budget alert event."""

    id: str = field(default_factory=lambda: str(uuid4()))
    budget_id: str = ""
    workspace_id: Optional[str] = None
    org_id: Optional[str] = None
    level: BudgetAlertLevel = BudgetAlertLevel.INFO
    message: str = ""
    current_spend: Decimal = Decimal("0")
    limit: Decimal = Decimal("0")
    percentage: float = 0.0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    acknowledged: bool = False
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None


@dataclass
class CostReport:
    """Aggregated cost report."""

    workspace_id: Optional[str] = None
    org_id: Optional[str] = None
    period_start: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    period_end: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    granularity: CostGranularity = CostGranularity.DAILY

    # Total costs
    total_cost_usd: Decimal = Decimal("0")
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    total_api_calls: int = 0

    # Breakdowns
    cost_by_agent: Dict[str, Decimal] = field(default_factory=dict)
    cost_by_model: Dict[str, Decimal] = field(default_factory=dict)
    cost_by_provider: Dict[str, Decimal] = field(default_factory=dict)
    cost_by_operation: Dict[str, Decimal] = field(default_factory=dict)

    # Time series
    cost_over_time: List[Dict[str, Any]] = field(default_factory=list)

    # Efficiency metrics
    avg_cost_per_call: Decimal = Decimal("0")
    avg_tokens_per_call: float = 0.0
    avg_latency_ms: float = 0.0

    # Top consumers
    top_debates_by_cost: List[Dict[str, Any]] = field(default_factory=list)
    top_agents_by_cost: List[Dict[str, Any]] = field(default_factory=list)

    # Projections
    projected_monthly_cost: Optional[Decimal] = None
    projected_daily_rate: Optional[Decimal] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "workspace_id": self.workspace_id,
            "org_id": self.org_id,
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "granularity": self.granularity.value,
            "total_cost_usd": str(self.total_cost_usd),
            "total_tokens_in": self.total_tokens_in,
            "total_tokens_out": self.total_tokens_out,
            "total_api_calls": self.total_api_calls,
            "cost_by_agent": {k: str(v) for k, v in self.cost_by_agent.items()},
            "cost_by_model": {k: str(v) for k, v in self.cost_by_model.items()},
            "cost_by_provider": {k: str(v) for k, v in self.cost_by_provider.items()},
            "cost_by_operation": {k: str(v) for k, v in self.cost_by_operation.items()},
            "cost_over_time": self.cost_over_time,
            "avg_cost_per_call": str(self.avg_cost_per_call),
            "avg_tokens_per_call": self.avg_tokens_per_call,
            "avg_latency_ms": self.avg_latency_ms,
            "top_debates_by_cost": self.top_debates_by_cost,
            "top_agents_by_cost": self.top_agents_by_cost,
            "projected_monthly_cost": str(self.projected_monthly_cost) if self.projected_monthly_cost else None,
            "projected_daily_rate": str(self.projected_daily_rate) if self.projected_daily_rate else None,
        }


AlertCallback = Callable[[BudgetAlert], None]


class CostTracker:
    """
    Real-time cost tracking with workspace/agent attribution.

    Provides comprehensive cost monitoring with budget management,
    alerting, and detailed reporting.
    """

    def __init__(
        self,
        usage_tracker: Optional[UsageTracker] = None,
    ):
        """
        Initialize cost tracker.

        Args:
            usage_tracker: Optional UsageTracker for persistence
        """
        self._usage_tracker = usage_tracker

        # In-memory tracking for real-time updates
        self._usage_buffer: List[TokenUsage] = []
        self._buffer_lock = asyncio.Lock()
        self._buffer_max_size = 1000

        # Budget management
        self._budgets: Dict[str, Budget] = {}  # budget_id -> Budget
        self._workspace_budgets: Dict[str, str] = {}  # workspace_id -> budget_id
        self._org_budgets: Dict[str, str] = {}  # org_id -> budget_id

        # Alert management
        self._alert_callbacks: List[AlertCallback] = []
        self._sent_alerts: Set[str] = set()  # Deduplicate alerts

        # Aggregated stats (refreshed periodically)
        self._workspace_stats: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                "total_cost": Decimal("0"),
                "tokens_in": 0,
                "tokens_out": 0,
                "api_calls": 0,
                "by_agent": defaultdict(lambda: Decimal("0")),
                "by_model": defaultdict(lambda: Decimal("0")),
            }
        )

    async def record(self, usage: TokenUsage) -> None:
        """
        Record a token usage event.

        Args:
            usage: Token usage to record
        """
        # Calculate cost if not already done
        if usage.cost_usd == Decimal("0"):
            usage.calculate_cost()

        # Update in-memory stats
        async with self._buffer_lock:
            self._usage_buffer.append(usage)

            # Flush buffer if too large
            if len(self._usage_buffer) >= self._buffer_max_size:
                await self._flush_buffer()

        # Update workspace stats
        stats = self._workspace_stats[usage.workspace_id]
        stats["total_cost"] += usage.cost_usd
        stats["tokens_in"] += usage.tokens_in
        stats["tokens_out"] += usage.tokens_out
        stats["api_calls"] += 1
        stats["by_agent"][usage.agent_name] += usage.cost_usd
        stats["by_model"][usage.model] += usage.cost_usd

        # Update budget tracking
        await self._update_budget(usage)

        # Persist to usage tracker
        if self._usage_tracker:
            event = UsageEvent(
                user_id=usage.metadata.get("user_id", ""),
                org_id=usage.metadata.get("org_id", ""),
                event_type=UsageEventType.AGENT_CALL,
                debate_id=usage.debate_id,
                tokens_in=usage.tokens_in,
                tokens_out=usage.tokens_out,
                provider=usage.provider,
                model=usage.model,
                cost_usd=usage.cost_usd,
                metadata={
                    "workspace_id": usage.workspace_id,
                    "agent_id": usage.agent_id,
                    "agent_name": usage.agent_name,
                    "session_id": usage.session_id,
                    "operation": usage.operation,
                    "latency_ms": usage.latency_ms,
                },
            )
            self._usage_tracker.record(event)

        logger.debug(
            f"cost_recorded workspace={usage.workspace_id} agent={usage.agent_name} "
            f"cost=${usage.cost_usd:.6f} tokens={usage.tokens_in + usage.tokens_out}"
        )

    async def record_batch(self, usages: List[TokenUsage]) -> None:
        """Record multiple usage events."""
        for usage in usages:
            await self.record(usage)

    async def _flush_buffer(self) -> None:
        """Flush usage buffer (called when buffer is full)."""
        if not self._usage_buffer:
            return

        # For now, just clear the buffer
        # In production, this would persist to a time-series database
        buffer_size = len(self._usage_buffer)
        self._usage_buffer = []
        logger.debug(f"Flushed {buffer_size} usage records from buffer")

    async def _update_budget(self, usage: TokenUsage) -> None:
        """Update budget tracking and check for alerts."""
        # Find applicable budget
        budget = None

        if usage.workspace_id and usage.workspace_id in self._workspace_budgets:
            budget_id = self._workspace_budgets[usage.workspace_id]
            budget = self._budgets.get(budget_id)

        org_id = usage.metadata.get("org_id", "")
        if not budget and org_id and org_id in self._org_budgets:
            budget_id = self._org_budgets[org_id]
            budget = self._budgets.get(budget_id)

        if not budget:
            return

        # Update spend
        budget.current_daily_spend += usage.cost_usd
        budget.current_monthly_spend += usage.cost_usd
        budget.updated_at = datetime.now(timezone.utc)

        # Check for alerts
        await self._check_budget_alerts(budget)

    async def _check_budget_alerts(self, budget: Budget) -> None:
        """Check if budget thresholds are exceeded and send alerts."""
        alert_level = budget.check_alert_level()
        if not alert_level:
            return

        # Create unique key to avoid duplicate alerts
        alert_key = f"{budget.id}:{alert_level.value}:{datetime.now(timezone.utc).strftime('%Y-%m-%d')}"
        if alert_key in self._sent_alerts:
            return

        self._sent_alerts.add(alert_key)

        # Calculate percentage
        percentage = float(
            (budget.current_monthly_spend / budget.monthly_limit_usd) * 100
        ) if budget.monthly_limit_usd else 0

        alert = BudgetAlert(
            budget_id=budget.id,
            workspace_id=budget.workspace_id,
            org_id=budget.org_id,
            level=alert_level,
            message=f"Budget {budget.name}: {percentage:.1f}% used (${budget.current_monthly_spend:.2f} of ${budget.monthly_limit_usd:.2f})",
            current_spend=budget.current_monthly_spend,
            limit=budget.monthly_limit_usd or Decimal("0"),
            percentage=percentage,
        )

        # Notify callbacks
        for callback in self._alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")

        logger.warning(f"budget_alert {alert.message}")

    def set_budget(self, budget: Budget) -> None:
        """
        Set a budget for a workspace or organization.

        Args:
            budget: Budget configuration
        """
        self._budgets[budget.id] = budget

        if budget.workspace_id:
            self._workspace_budgets[budget.workspace_id] = budget.id
        if budget.org_id:
            self._org_budgets[budget.org_id] = budget.id

    def get_budget(
        self,
        workspace_id: Optional[str] = None,
        org_id: Optional[str] = None,
    ) -> Optional[Budget]:
        """Get budget for workspace or organization."""
        if workspace_id and workspace_id in self._workspace_budgets:
            return self._budgets.get(self._workspace_budgets[workspace_id])
        if org_id and org_id in self._org_budgets:
            return self._budgets.get(self._org_budgets[org_id])
        return None

    def add_alert_callback(self, callback: AlertCallback) -> None:
        """Register a callback for budget alerts."""
        self._alert_callbacks.append(callback)

    def remove_alert_callback(self, callback: AlertCallback) -> None:
        """Remove an alert callback."""
        if callback in self._alert_callbacks:
            self._alert_callbacks.remove(callback)

    def get_workspace_stats(self, workspace_id: str) -> Dict[str, Any]:
        """
        Get real-time cost stats for a workspace.

        Args:
            workspace_id: Workspace ID

        Returns:
            Cost statistics
        """
        stats = self._workspace_stats.get(workspace_id, {})
        return {
            "workspace_id": workspace_id,
            "total_cost_usd": str(stats.get("total_cost", Decimal("0"))),
            "total_tokens_in": stats.get("tokens_in", 0),
            "total_tokens_out": stats.get("tokens_out", 0),
            "total_api_calls": stats.get("api_calls", 0),
            "cost_by_agent": {
                k: str(v) for k, v in stats.get("by_agent", {}).items()
            },
            "cost_by_model": {
                k: str(v) for k, v in stats.get("by_model", {}).items()
            },
        }

    async def generate_report(
        self,
        workspace_id: Optional[str] = None,
        org_id: Optional[str] = None,
        period_start: Optional[datetime] = None,
        period_end: Optional[datetime] = None,
        granularity: CostGranularity = CostGranularity.DAILY,
    ) -> CostReport:
        """
        Generate a comprehensive cost report.

        Args:
            workspace_id: Filter by workspace
            org_id: Filter by organization
            period_start: Report start time
            period_end: Report end time
            granularity: Time aggregation granularity

        Returns:
            Cost report
        """
        if period_end is None:
            period_end = datetime.now(timezone.utc)
        if period_start is None:
            period_start = period_end - timedelta(days=30)

        report = CostReport(
            workspace_id=workspace_id,
            org_id=org_id,
            period_start=period_start,
            period_end=period_end,
            granularity=granularity,
        )

        # Aggregate from in-memory stats if workspace specified
        if workspace_id and workspace_id in self._workspace_stats:
            stats = self._workspace_stats[workspace_id]
            report.total_cost_usd = stats.get("total_cost", Decimal("0"))
            report.total_tokens_in = stats.get("tokens_in", 0)
            report.total_tokens_out = stats.get("tokens_out", 0)
            report.total_api_calls = stats.get("api_calls", 0)
            report.cost_by_agent = dict(stats.get("by_agent", {}))
            report.cost_by_model = dict(stats.get("by_model", {}))

            # Calculate averages
            if report.total_api_calls > 0:
                report.avg_cost_per_call = report.total_cost_usd / report.total_api_calls
                report.avg_tokens_per_call = (
                    report.total_tokens_in + report.total_tokens_out
                ) / report.total_api_calls

        # If we have a usage tracker, get historical data
        if self._usage_tracker and org_id:
            summary = self._usage_tracker.get_summary(
                org_id=org_id,
                period_start=period_start,
                period_end=period_end,
            )

            # Merge with in-memory if not already populated
            if report.total_cost_usd == Decimal("0"):
                report.total_cost_usd = summary.total_cost_usd
                report.total_tokens_in = summary.total_tokens_in
                report.total_tokens_out = summary.total_tokens_out
                report.total_api_calls = summary.total_agent_calls + summary.total_debates

            report.cost_by_provider = summary.cost_by_provider

        # Calculate projections
        if report.total_api_calls > 0:
            days_in_period = max(1, (period_end - period_start).days)
            report.projected_daily_rate = report.total_cost_usd / days_in_period

            # Project to full month
            days_in_month = 30
            report.projected_monthly_cost = report.projected_daily_rate * days_in_month

        # Get top agents
        if report.cost_by_agent:
            sorted_agents = sorted(
                report.cost_by_agent.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:10]
            report.top_agents_by_cost = [
                {"agent": k, "cost_usd": str(v)} for k, v in sorted_agents
            ]

        return report

    async def get_agent_costs(
        self,
        workspace_id: str,
        period_start: Optional[datetime] = None,
        period_end: Optional[datetime] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get cost breakdown by agent.

        Args:
            workspace_id: Workspace ID
            period_start: Start of period
            period_end: End of period

        Returns:
            Cost breakdown by agent name
        """
        stats = self._workspace_stats.get(workspace_id, {})
        by_agent = stats.get("by_agent", {})

        result: Dict[str, Dict[str, Any]] = {}
        total_cost = stats.get("total_cost", Decimal("0"))

        for agent_name, cost in by_agent.items():
            percentage = (cost / total_cost * 100) if total_cost > 0 else 0
            result[agent_name] = {
                "cost_usd": str(cost),
                "percentage": float(percentage),
            }

        return result

    async def get_debate_cost(self, debate_id: str) -> Dict[str, Any]:
        """
        Get total cost for a debate.

        Args:
            debate_id: Debate ID

        Returns:
            Debate cost breakdown
        """
        total_cost = Decimal("0")
        total_tokens_in = 0
        total_tokens_out = 0
        by_agent: Dict[str, Decimal] = defaultdict(lambda: Decimal("0"))

        # Check buffer for recent debate data
        async with self._buffer_lock:
            for usage in self._usage_buffer:
                if usage.debate_id == debate_id:
                    total_cost += usage.cost_usd
                    total_tokens_in += usage.tokens_in
                    total_tokens_out += usage.tokens_out
                    by_agent[usage.agent_name] += usage.cost_usd

        # Also check persistent storage
        if self._usage_tracker:
            db_cost = self._usage_tracker.get_debate_cost(debate_id)
            if db_cost > total_cost:
                total_cost = db_cost

        return {
            "debate_id": debate_id,
            "total_cost_usd": str(total_cost),
            "total_tokens_in": total_tokens_in,
            "total_tokens_out": total_tokens_out,
            "cost_by_agent": {k: str(v) for k, v in by_agent.items()},
        }

    def reset_daily_budgets(self) -> None:
        """Reset daily budget counters (called at midnight)."""
        for budget in self._budgets.values():
            budget.current_daily_spend = Decimal("0")

        # Clear daily alert dedup keys
        keys_to_remove = [k for k in self._sent_alerts if "daily" in k]
        for key in keys_to_remove:
            self._sent_alerts.discard(key)

    def reset_monthly_budgets(self) -> None:
        """Reset monthly budget counters (called at month start)."""
        for budget in self._budgets.values():
            budget.current_monthly_spend = Decimal("0")
            budget.current_daily_spend = Decimal("0")

        # Clear all alert dedup keys
        self._sent_alerts.clear()

        # Reset workspace stats
        self._workspace_stats.clear()


# Global cost tracker instance
_cost_tracker: Optional[CostTracker] = None


def get_cost_tracker() -> CostTracker:
    """Get or create the global cost tracker."""
    global _cost_tracker
    if _cost_tracker is None:
        try:
            usage_tracker = UsageTracker()
        except Exception:  # noqa: BLE001 - Optional dependency, graceful degradation
            usage_tracker = None
        _cost_tracker = CostTracker(usage_tracker=usage_tracker)
    return _cost_tracker


async def record_usage(
    workspace_id: str,
    agent_name: str,
    provider: str,
    model: str,
    tokens_in: int,
    tokens_out: int,
    debate_id: Optional[str] = None,
    operation: str = "",
    latency_ms: float = 0.0,
    metadata: Optional[Dict[str, Any]] = None,
) -> TokenUsage:
    """
    Convenience function to record token usage.

    Args:
        workspace_id: Workspace ID
        agent_name: Name of the agent
        provider: LLM provider
        model: Model used
        tokens_in: Input tokens
        tokens_out: Output tokens
        debate_id: Optional debate ID
        operation: Operation type
        latency_ms: Request latency
        metadata: Additional metadata

    Returns:
        Created TokenUsage record
    """
    usage = TokenUsage(
        workspace_id=workspace_id,
        agent_name=agent_name,
        provider=provider,
        model=model,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        debate_id=debate_id,
        operation=operation,
        latency_ms=latency_ms,
        metadata=metadata or {},
    )

    tracker = get_cost_tracker()
    await tracker.record(usage)

    return usage


__all__ = [
    "CostTracker",
    "TokenUsage",
    "Budget",
    "BudgetAlert",
    "BudgetAlertLevel",
    "CostReport",
    "CostGranularity",
    "get_cost_tracker",
    "record_usage",
]
