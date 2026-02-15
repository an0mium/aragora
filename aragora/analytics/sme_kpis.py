"""
SME KPI computation module.

Computes key performance indicators for SME customers:
- Cost per decision
- Decision velocity (debates per week)
- Time saved (hours)
- ROI percentage

Aggregates data from DebateAnalytics and CostTracker with graceful
fallbacks when data sources are unavailable.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

logger = logging.getLogger(__name__)

# Industry benchmarks
DEFAULT_MANUAL_DECISION_HOURS = 0.75  # 45 minutes
DEFAULT_HOURLY_RATE = 75.0  # USD

PERIOD_DAYS = {
    "week": 7,
    "month": 30,
    "quarter": 90,
    "year": 365,
}


@dataclass
class SMEKPIs:
    """Key performance indicators for SME customers."""

    cost_per_decision: float = 0.0
    decision_velocity: float = 0.0  # debates per week
    time_saved_hours: float = 0.0
    roi_percentage: float = 0.0

    # Supporting details
    total_debates: int = 0
    total_cost_usd: float = 0.0
    avg_debate_duration_minutes: float = 5.0
    period_days: int = 30

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "cost_per_decision": round(self.cost_per_decision, 2),
            "decision_velocity": round(self.decision_velocity, 2),
            "time_saved_hours": round(self.time_saved_hours, 1),
            "roi_percentage": round(self.roi_percentage, 1),
            "total_debates": self.total_debates,
            "total_cost_usd": round(self.total_cost_usd, 2),
            "avg_debate_duration_minutes": round(self.avg_debate_duration_minutes, 1),
            "period_days": self.period_days,
        }


def _get_debate_analytics() -> Any | None:
    """Lazily import and return DebateAnalytics instance."""
    try:
        from aragora.analytics.debate_analytics import get_debate_analytics

        return get_debate_analytics()
    except ImportError:
        logger.debug("DebateAnalytics not available")
        return None


def _get_cost_tracker() -> Any | None:
    """Lazily import and return CostTracker instance."""
    try:
        from aragora.billing.cost_tracker import get_cost_tracker

        return get_cost_tracker()
    except ImportError:
        logger.debug("CostTracker not available")
        return None


def _run_async(coro: Any) -> Any:
    """Run an async coroutine from sync context."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Already in an async context, create a task
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(asyncio.run, coro).result(timeout=10)
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


def get_sme_kpis(
    org_id: str,
    period: str = "month",
    manual_decision_hours: float = DEFAULT_MANUAL_DECISION_HOURS,
    hourly_rate: float = DEFAULT_HOURLY_RATE,
) -> SMEKPIs:
    """
    Compute SME key performance indicators for an organization.

    Args:
        org_id: Organization ID
        period: Time period - "week", "month", "quarter", or "year"
        manual_decision_hours: Benchmark hours for manual decision-making
        hourly_rate: Hourly rate for time-value calculations (USD)

    Returns:
        SMEKPIs with computed metrics, defaults when data unavailable
    """
    days = PERIOD_DAYS.get(period, 30)
    weeks = max(1, days / 7)

    kpis = SMEKPIs(period_days=days)

    # Get debate stats from analytics
    analytics = _get_debate_analytics()
    debate_stats = None
    if analytics is not None:
        try:
            debate_stats = _run_async(analytics.get_debate_stats(org_id=org_id, days_back=days))
        except Exception as e:
            logger.debug(f"Failed to get debate stats: {e}")

    # Get cost data from cost tracker
    cost_tracker = _get_cost_tracker()
    total_cost = Decimal("0")
    if cost_tracker is not None:
        try:
            workspace_stats = cost_tracker.get_workspace_stats(org_id)
            total_cost = Decimal(workspace_stats.get("total_cost_usd", "0"))
        except Exception as e:
            logger.debug(f"Failed to get cost data: {e}")

    kpis.total_cost_usd = float(total_cost)

    if debate_stats is not None and debate_stats.total_debates > 0:
        kpis.total_debates = debate_stats.total_debates
        kpis.avg_debate_duration_minutes = (
            debate_stats.avg_duration_seconds / 60 if debate_stats.avg_duration_seconds > 0 else 5.0
        )
    else:
        # Estimate from cost data if no analytics
        api_calls = 0
        if cost_tracker is not None:
            try:
                ws = cost_tracker.get_workspace_stats(org_id)
                api_calls = ws.get("total_api_calls", 0)
            except Exception as e:
                logger.warning("Failed to get workspace API call stats for %s: %s", org_id, e)
        kpis.total_debates = max(1, api_calls // 10) if api_calls > 0 else 0

    # Cost per decision
    if kpis.total_debates > 0:
        kpis.cost_per_decision = kpis.total_cost_usd / kpis.total_debates

    # Decision velocity (debates per week)
    kpis.decision_velocity = kpis.total_debates / weeks

    # Time saved
    ai_hours = kpis.total_debates * kpis.avg_debate_duration_minutes / 60
    manual_hours = kpis.total_debates * manual_decision_hours
    kpis.time_saved_hours = max(0.0, manual_hours - ai_hours)

    # ROI percentage
    time_value_saved = kpis.time_saved_hours * hourly_rate
    if kpis.total_cost_usd > 0:
        kpis.roi_percentage = (time_value_saved - kpis.total_cost_usd) / kpis.total_cost_usd * 100
    else:
        kpis.roi_percentage = 0.0

    return kpis
