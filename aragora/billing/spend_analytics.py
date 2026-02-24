"""
Spend Analytics for Actionable Cost Visibility.

Provides workspace-level spend analysis including:
- Spend trends over configurable periods
- Provider and agent cost breakdowns
- Linear cost forecasting
- Anomaly detection via z-score analysis

Designed for the spend analytics dashboard (GitHub issue #264).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from statistics import mean, stdev
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aragora.billing.cost_tracker import CostTracker

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class DailySpend:
    """A single day's spend data point."""

    date: str
    cost_usd: float


@dataclass
class SpendTrend:
    """Daily spend over a period with summary statistics."""

    workspace_id: str
    period: str
    points: list[DailySpend] = field(default_factory=list)
    total_usd: float = 0.0
    avg_daily_usd: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "workspace_id": self.workspace_id,
            "period": self.period,
            "points": [{"date": p.date, "cost_usd": p.cost_usd} for p in self.points],
            "total_usd": round(self.total_usd, 4),
            "avg_daily_usd": round(self.avg_daily_usd, 4),
        }


@dataclass
class CostForecast:
    """Linear cost projection over a future period."""

    workspace_id: str
    forecast_days: int
    projected_total_usd: float = 0.0
    projected_daily_avg_usd: float = 0.0
    trend: str = "stable"  # "increasing" | "stable" | "decreasing"
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "workspace_id": self.workspace_id,
            "forecast_days": self.forecast_days,
            "projected_total_usd": round(self.projected_total_usd, 4),
            "projected_daily_avg_usd": round(self.projected_daily_avg_usd, 4),
            "trend": self.trend,
            "confidence": round(self.confidence, 3),
        }


@dataclass
class SpendAnomaly:
    """An unusual spend event detected via statistical analysis."""

    date: str
    actual_usd: float
    expected_usd: float
    z_score: float
    severity: str  # "warning" | "critical"
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "date": self.date,
            "actual_usd": round(self.actual_usd, 4),
            "expected_usd": round(self.expected_usd, 4),
            "z_score": round(self.z_score, 2),
            "severity": self.severity,
            "description": self.description,
        }


# ---------------------------------------------------------------------------
# Period parsing helper
# ---------------------------------------------------------------------------

_PERIOD_DAYS = {
    "7d": 7,
    "14d": 14,
    "30d": 30,
    "60d": 60,
    "90d": 90,
}


def _parse_period_days(period: str) -> int:
    """Convert a period string like '30d' to an integer number of days."""
    if period in _PERIOD_DAYS:
        return _PERIOD_DAYS[period]
    # Attempt to parse custom format (e.g. "45d")
    if period.endswith("d") and period[:-1].isdigit():
        return int(period[:-1])
    return 30  # safe default


# ---------------------------------------------------------------------------
# SpendAnalytics
# ---------------------------------------------------------------------------


class SpendAnalytics:
    """Workspace-level spend analytics built on CostTracker data.

    Reads from CostTracker's in-memory buffer to provide trend,
    breakdown, forecast, and anomaly-detection capabilities without
    requiring any additional persistence layer.

    Usage:
        from aragora.billing.spend_analytics import SpendAnalytics
        from aragora.billing.cost_tracker import get_cost_tracker

        analytics = SpendAnalytics(cost_tracker=get_cost_tracker())
        trend = await analytics.get_spend_trend("ws-1", period="30d")
    """

    def __init__(self, cost_tracker: CostTracker | None = None) -> None:
        self._cost_tracker = cost_tracker

    def set_cost_tracker(self, tracker: CostTracker) -> None:
        """Set the cost tracker instance."""
        self._cost_tracker = tracker

    # -- internal helpers ---------------------------------------------------

    async def _daily_costs(
        self,
        workspace_id: str,
        days: int,
    ) -> list[tuple[str, float]]:
        """Aggregate cost tracker buffer into (date_str, total_cost) pairs."""
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(days=days)

        daily: dict[str, float] = {}

        if self._cost_tracker:
            async with self._cost_tracker._buffer_lock:
                for usage in self._cost_tracker._usage_buffer:
                    if usage.workspace_id != workspace_id:
                        continue
                    if usage.timestamp < cutoff:
                        continue
                    key = usage.timestamp.strftime("%Y-%m-%d")
                    daily[key] = daily.get(key, 0.0) + float(usage.cost_usd)

        # Fill missing days with zero for a continuous series
        result: list[tuple[str, float]] = []
        for i in range(days):
            d = (cutoff + timedelta(days=i + 1)).strftime("%Y-%m-%d")
            result.append((d, daily.get(d, 0.0)))
        return result

    # -- public API ---------------------------------------------------------

    async def get_spend_trend(
        self,
        workspace_id: str,
        period: str = "30d",
    ) -> SpendTrend:
        """Return daily spend over the given period.

        Args:
            workspace_id: Workspace identifier.
            period: Period string, e.g. "7d", "30d", "90d".

        Returns:
            SpendTrend with daily data points and summary stats.
        """
        days = _parse_period_days(period)
        raw = await self._daily_costs(workspace_id, days)

        points = [DailySpend(date=d, cost_usd=c) for d, c in raw]
        total = sum(c for _, c in raw)
        avg = total / max(1, len(raw))

        return SpendTrend(
            workspace_id=workspace_id,
            period=period,
            points=points,
            total_usd=total,
            avg_daily_usd=avg,
        )

    async def get_spend_by_provider(self, workspace_id: str) -> dict[str, float]:
        """Return cost breakdown by API provider.

        Args:
            workspace_id: Workspace identifier.

        Returns:
            Dict mapping provider name to total cost in USD.
        """
        by_provider: dict[str, float] = {}

        if self._cost_tracker:
            async with self._cost_tracker._buffer_lock:
                for usage in self._cost_tracker._usage_buffer:
                    if usage.workspace_id != workspace_id:
                        continue
                    by_provider[usage.provider] = (
                        by_provider.get(usage.provider, 0.0) + float(usage.cost_usd)
                    )

        return by_provider

    async def get_spend_by_agent(self, workspace_id: str) -> dict[str, float]:
        """Return cost breakdown by agent type.

        Args:
            workspace_id: Workspace identifier.

        Returns:
            Dict mapping agent name to total cost in USD.
        """
        by_agent: dict[str, float] = {}

        if self._cost_tracker:
            async with self._cost_tracker._buffer_lock:
                for usage in self._cost_tracker._usage_buffer:
                    if usage.workspace_id != workspace_id:
                        continue
                    name = usage.agent_name or usage.agent_id or "unknown"
                    by_agent[name] = by_agent.get(name, 0.0) + float(usage.cost_usd)

        return by_agent

    async def get_cost_forecast(
        self,
        workspace_id: str,
        days: int = 30,
    ) -> CostForecast:
        """Project future spend using linear regression on recent history.

        Args:
            workspace_id: Workspace identifier.
            days: Number of days to forecast.

        Returns:
            CostForecast with projected totals and trend direction.
        """
        # Use last 30 days as history baseline
        raw = await self._daily_costs(workspace_id, 30)
        costs = [c for _, c in raw]

        if not costs or all(c == 0.0 for c in costs):
            return CostForecast(workspace_id=workspace_id, forecast_days=days)

        n = len(costs)
        x_mean = (n - 1) / 2.0
        y_mean = mean(costs)

        # Least-squares slope
        numerator = sum((i - x_mean) * (costs[i] - y_mean) for i in range(n))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        slope = numerator / denominator if denominator != 0 else 0.0

        # R-squared for confidence
        y_pred = [y_mean + slope * (i - x_mean) for i in range(n)]
        ss_res = sum((costs[i] - y_pred[i]) ** 2 for i in range(n))
        ss_tot = sum((costs[i] - y_mean) ** 2 for i in range(n))
        r_squared = max(0.0, 1 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

        # Project forward from last predicted value
        last_pred = y_mean + slope * (n - 1 - x_mean)
        projected_daily = [last_pred + slope * (i + 1) for i in range(days)]
        projected_daily = [max(0.0, v) for v in projected_daily]
        projected_total = sum(projected_daily)
        projected_avg = projected_total / max(1, days)

        # Determine trend direction
        if y_mean > 0:
            daily_rate = (slope / y_mean) * 100
        else:
            daily_rate = 0.0

        if abs(daily_rate) < 1:
            trend = "stable"
        elif daily_rate > 0:
            trend = "increasing"
        else:
            trend = "decreasing"

        return CostForecast(
            workspace_id=workspace_id,
            forecast_days=days,
            projected_total_usd=projected_total,
            projected_daily_avg_usd=projected_avg,
            trend=trend,
            confidence=round(r_squared, 3),
        )

    async def get_anomalies(
        self,
        workspace_id: str,
        period: str = "30d",
        z_threshold: float = 2.0,
    ) -> list[SpendAnomaly]:
        """Detect days with anomalous spend using z-score analysis.

        Only days with non-zero spend are considered. The statistics
        (mean, stdev) are computed over non-zero days, and anomalies
        are flagged among those days. At least 3 non-zero days are
        required to produce meaningful results.

        Args:
            workspace_id: Workspace identifier.
            period: Period to scan.
            z_threshold: Z-score threshold for flagging anomalies.

        Returns:
            List of SpendAnomaly instances ordered by z-score descending.
        """
        days = _parse_period_days(period)
        raw = await self._daily_costs(workspace_id, days)

        # Only consider days that had actual spend
        active_days = [(d, c) for d, c in raw if c > 0]

        if len(active_days) < 3:
            return []

        active_costs = [c for _, c in active_days]
        avg = mean(active_costs)
        sd = stdev(active_costs) if len(active_costs) > 1 else 0.0

        if sd == 0:
            return []

        anomalies: list[SpendAnomaly] = []
        for date_str, cost in active_days:
            z = (cost - avg) / sd
            if abs(z) >= z_threshold:
                severity = "critical" if abs(z) >= 3.0 else "warning"
                pct = ((cost - avg) / avg * 100) if avg > 0 else 0
                anomalies.append(
                    SpendAnomaly(
                        date=date_str,
                        actual_usd=cost,
                        expected_usd=avg,
                        z_score=z,
                        severity=severity,
                        description=(
                            f"Spend ${cost:.2f} was {abs(pct):.0f}% "
                            f"{'above' if z > 0 else 'below'} average"
                        ),
                    )
                )

        anomalies.sort(key=lambda a: abs(a.z_score), reverse=True)
        return anomalies


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_spend_analytics: SpendAnalytics | None = None


def get_spend_analytics() -> SpendAnalytics:
    """Get or create the global SpendAnalytics instance."""
    global _spend_analytics
    if _spend_analytics is None:
        try:
            from aragora.billing.cost_tracker import get_cost_tracker

            tracker = get_cost_tracker()
            _spend_analytics = SpendAnalytics(cost_tracker=tracker)
        except (ImportError, RuntimeError, OSError) as e:
            logger.debug("SpendAnalytics created without cost tracker: %s", e)
            _spend_analytics = SpendAnalytics()
    return _spend_analytics


__all__ = [
    "CostForecast",
    "DailySpend",
    "SpendAnalytics",
    "SpendAnomaly",
    "SpendTrend",
    "get_spend_analytics",
]
