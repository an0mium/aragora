"""
Cost visibility data models and CostTracker integration.

Provides:
- CostEntry, BudgetAlert, CostSummary dataclasses
- CostTracker singleton access (_get_cost_tracker)
- record_cost() for recording cost entries
- get_cost_summary() for fetching cost summaries
- _get_active_alerts() for budget alert retrieval
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

logger = logging.getLogger(__name__)


def _is_demo_mode() -> bool:
    """Check if demo mode is enabled.

    When demo mode is enabled, mock data is returned instead of real data.
    This is useful for frontend development and demos without full backend setup.
    """
    try:
        from aragora.config.settings import get_settings

        return get_settings().features.demo_mode
    except Exception:
        return False


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class CostEntry:
    """A single cost entry."""

    timestamp: datetime
    provider: str
    feature: str
    tokens_input: int
    tokens_output: int
    cost: float
    model: str
    workspace_id: str
    user_id: str | None = None


@dataclass
class BudgetAlert:
    """A budget alert."""

    id: str
    type: str  # budget_warning, spike_detected, limit_reached
    message: str
    severity: str  # critical, warning, info
    timestamp: datetime
    acknowledged: bool = False


@dataclass
class CostSummary:
    """Cost summary data."""

    total_cost: float
    budget: float
    tokens_used: int
    api_calls: int
    last_updated: datetime
    cost_by_provider: list[dict[str, Any]] = field(default_factory=list)
    cost_by_feature: list[dict[str, Any]] = field(default_factory=list)
    daily_costs: list[dict[str, Any]] = field(default_factory=list)
    alerts: list[dict[str, Any]] = field(default_factory=list)


# =============================================================================
# CostTracker Integration (replaces in-memory storage)
# =============================================================================

_cost_tracker = None


def _get_cost_tracker():
    """Get or create the cost tracker instance."""
    global _cost_tracker
    if _cost_tracker is None:
        try:
            from aragora.billing.cost_tracker import get_cost_tracker

            _cost_tracker = get_cost_tracker()
            logger.info("[CostHandler] Connected to CostTracker with persistence")
        except Exception as e:
            logger.warning(f"[CostHandler] CostTracker unavailable, using fallback: {e}")
            _cost_tracker = None
    return _cost_tracker


def record_cost(
    provider: str,
    feature: str,
    tokens_input: int,
    tokens_output: int,
    cost: float,
    model: str,
    workspace_id: str = "default",
    user_id: str | None = None,
) -> None:
    """Record a cost entry via CostTracker."""
    tracker = _get_cost_tracker()

    if tracker:
        try:
            import asyncio
            from aragora.billing.cost_tracker import TokenUsage

            usage = TokenUsage(
                workspace_id=workspace_id,
                provider=provider,
                model=model,
                tokens_in=tokens_input,
                tokens_out=tokens_output,
                cost_usd=Decimal(str(cost)),
                operation=feature,
                metadata={"user_id": user_id} if user_id else {},
            )

            # Record asynchronously if in async context, otherwise sync
            try:
                asyncio.get_running_loop()  # Check if loop exists
                asyncio.create_task(tracker.record(usage))
            except RuntimeError:
                # No running loop, run synchronously
                asyncio.run(tracker.record(usage))

            logger.debug(f"[CostHandler] Recorded cost: ${cost:.6f} for {feature}")
        except Exception as e:
            logger.error(f"[CostHandler] Failed to record cost: {e}")
    else:
        logger.debug("[CostHandler] CostTracker not available, cost not persisted")


def _get_active_alerts(tracker, workspace_id: str) -> list[dict[str, Any]]:
    """Get active budget alerts from tracker."""
    alerts = []
    try:
        # Check if budget is approaching limits
        budget = tracker.get_budget(workspace_id=workspace_id)
        if budget:
            alert_level = budget.check_alert_level()
            if alert_level:
                from aragora.billing.cost_tracker import BudgetAlertLevel

                severity_map = {
                    BudgetAlertLevel.INFO: "info",
                    BudgetAlertLevel.WARNING: "warning",
                    BudgetAlertLevel.CRITICAL: "critical",
                    BudgetAlertLevel.EXCEEDED: "critical",
                }
                percentage = (
                    float(budget.current_monthly_spend / budget.monthly_limit_usd * 100)
                    if budget.monthly_limit_usd
                    else 0
                )
                alerts.append(
                    {
                        "id": f"budget_{workspace_id}",
                        "type": "budget_warning",
                        "message": f"Budget usage at {percentage:.1f}% (${float(budget.current_monthly_spend):.2f} of ${float(budget.monthly_limit_usd):.2f})",
                        "severity": severity_map.get(alert_level, "warning"),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                )
    except Exception as e:
        logger.debug(f"[CostHandler] Could not get alerts: {e}")
    return alerts


def _empty_cost_summary() -> CostSummary:
    """Return an empty cost summary when no data is available."""
    now = datetime.now(timezone.utc)
    return CostSummary(
        total_cost=0.0,
        budget=0.0,
        tokens_used=0,
        api_calls=0,
        last_updated=now,
        cost_by_provider=[],
        cost_by_feature=[],
        daily_costs=[],
        alerts=[],
    )


async def get_cost_summary(
    workspace_id: str = "default",
    time_range: str = "7d",
) -> CostSummary:
    """Get cost summary data from CostTracker.

    Behavior depends on ARAGORA_DEMO_MODE:
    - When demo mode is enabled: Returns mock data for development/demos
    - When demo mode is disabled: Returns real data or empty data if unavailable
    """
    from .helpers import _generate_mock_summary

    demo_mode = _is_demo_mode()

    now = datetime.now(timezone.utc)
    range_days = {"24h": 1, "7d": 7, "30d": 30, "90d": 90}.get(time_range, 7)
    start_date = now - timedelta(days=range_days)

    tracker = _get_cost_tracker()

    if tracker:
        try:
            # Use CostTracker for real data
            from aragora.billing.cost_tracker import CostGranularity

            # Get report from tracker
            report = await tracker.generate_report(
                workspace_id=workspace_id,
                period_start=start_date,
                period_end=now,
                granularity=CostGranularity.DAILY,
            )

            # Get budget
            budget_obj = tracker.get_budget(workspace_id=workspace_id)
            budget = (
                float(budget_obj.monthly_limit_usd)
                if budget_obj and budget_obj.monthly_limit_usd
                else 500.0
            )

            # Convert cost_by_provider to list format
            total_cost = float(report.total_cost_usd)
            cost_by_provider = (
                [
                    {
                        "name": name,
                        "cost": float(cost),
                        "percentage": (float(cost) / total_cost * 100) if total_cost > 0 else 0,
                    }
                    for name, cost in sorted(
                        report.cost_by_provider.items(),
                        key=lambda x: float(x[1]),
                        reverse=True,
                    )
                ]
                if report.cost_by_provider
                else []
            )

            # Convert cost_by_operation (feature) to list format
            cost_by_feature = (
                [
                    {
                        "name": name,
                        "cost": float(cost),
                        "percentage": (float(cost) / total_cost * 100) if total_cost > 0 else 0,
                    }
                    for name, cost in sorted(
                        report.cost_by_operation.items(),
                        key=lambda x: float(x[1]),
                        reverse=True,
                    )
                ]
                if report.cost_by_operation
                else []
            )

            # If no real data yet
            if total_cost == 0 and not report.cost_over_time:
                if demo_mode:
                    logger.info(
                        "[CostHandler] No cost data, returning mock data (ARAGORA_DEMO_MODE=true)"
                    )
                    return _generate_mock_summary(time_range)
                else:
                    logger.debug("[CostHandler] No cost data, returning empty summary")
                    return _empty_cost_summary()

            return CostSummary(
                total_cost=total_cost,
                budget=budget,
                tokens_used=report.total_tokens_in + report.total_tokens_out,
                api_calls=report.total_api_calls,
                last_updated=now,
                cost_by_provider=cost_by_provider if cost_by_provider else [],
                cost_by_feature=cost_by_feature if cost_by_feature else [],
                daily_costs=report.cost_over_time if report.cost_over_time else [],
                alerts=_get_active_alerts(tracker, workspace_id),
            )

        except Exception as e:
            if demo_mode:
                logger.warning(
                    f"[CostHandler] CostTracker query failed, using mock "
                    f"(ARAGORA_DEMO_MODE=true): {e}"
                )
                return _generate_mock_summary(time_range)
            else:
                logger.warning(f"[CostHandler] CostTracker query failed, returning empty: {e}")
                return _empty_cost_summary()

    # No tracker available
    if demo_mode:
        logger.info(
            "[CostHandler] CostTracker unavailable, returning mock data (ARAGORA_DEMO_MODE=true)"
        )
        return _generate_mock_summary(time_range)
    else:
        logger.debug("[CostHandler] CostTracker unavailable, returning empty summary")
        return _empty_cost_summary()
