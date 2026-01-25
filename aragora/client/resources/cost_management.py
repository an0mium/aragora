"""
Cost Management API resource for the Aragora client.

Provides methods for tracking and managing AI costs:
- Cost dashboard and summaries
- Cost breakdown by provider, feature, model
- Budget management and alerts
- Usage timeline and forecasting
- Optimization recommendations
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from ..client import AragoraClient

logger = logging.getLogger(__name__)


@dataclass
class CostBreakdownItem:
    """A cost breakdown item."""

    name: str
    cost: float
    percentage: float


@dataclass
class DailyCost:
    """Daily cost data point."""

    date: str
    cost: float
    tokens: int = 0


@dataclass
class CostAlert:
    """A budget alert."""

    id: str
    type: str  # budget_warning, spike_detected, limit_reached
    message: str
    severity: str  # critical, warning, info
    timestamp: str
    acknowledged: bool = False


@dataclass
class CostSummary:
    """Cost summary data."""

    total_cost: float
    budget: float
    tokens_used: int
    api_calls: int
    last_updated: str
    cost_by_provider: List[CostBreakdownItem] = field(default_factory=list)
    cost_by_feature: List[CostBreakdownItem] = field(default_factory=list)
    daily_costs: List[DailyCost] = field(default_factory=list)
    alerts: List[CostAlert] = field(default_factory=list)


@dataclass
class Budget:
    """A budget configuration."""

    workspace_id: str
    monthly_limit: float
    daily_limit: Optional[float] = None
    current_spend: float = 0.0
    alert_thresholds: List[int] = field(default_factory=lambda: [50, 80, 100])


@dataclass
class CostRecommendation:
    """A cost optimization recommendation."""

    id: str
    type: str  # model_downgrade, caching, batching, etc.
    title: str
    description: str
    estimated_savings: float
    effort: str  # low, medium, high
    status: str  # pending, applied, dismissed
    created_at: Optional[datetime] = None


@dataclass
class EfficiencyMetrics:
    """Efficiency metrics."""

    cost_per_1k_tokens: float
    tokens_per_call: float
    cost_per_call: float
    total_tokens: int
    total_calls: int
    total_cost: float
    model_utilization: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class CostForecast:
    """Cost forecast data."""

    workspace_id: str
    forecast_days: int
    projected_cost: float
    confidence_interval: tuple[float, float] = (0.0, 0.0)
    trend: str = "stable"  # increasing, decreasing, stable
    daily_projections: List[Dict[str, Any]] = field(default_factory=list)


class CostManagementAPI:
    """API interface for cost management."""

    def __init__(self, client: "AragoraClient"):
        self._client = client

    # =========================================================================
    # Cost Dashboard
    # =========================================================================

    def get_summary(
        self,
        workspace_id: str = "default",
        time_range: str = "7d",
    ) -> CostSummary:
        """
        Get cost dashboard summary.

        Args:
            workspace_id: The workspace ID.
            time_range: Time range (24h, 7d, 30d, 90d).

        Returns:
            CostSummary object.
        """
        params = {"workspace_id": workspace_id, "range": time_range}
        response = self._client._get("/api/costs", params=params)
        return self._parse_summary(response)

    async def get_summary_async(
        self,
        workspace_id: str = "default",
        time_range: str = "7d",
    ) -> CostSummary:
        """Async version of get_summary()."""
        params = {"workspace_id": workspace_id, "range": time_range}
        response = await self._client._get_async("/api/costs", params=params)
        return self._parse_summary(response)

    def get_breakdown(
        self,
        workspace_id: str = "default",
        time_range: str = "7d",
        group_by: str = "provider",
    ) -> tuple[List[CostBreakdownItem], float]:
        """
        Get detailed cost breakdown.

        Args:
            workspace_id: The workspace ID.
            time_range: Time range (24h, 7d, 30d, 90d).
            group_by: Grouping (provider, feature, model).

        Returns:
            Tuple of (breakdown items, total cost).
        """
        params = {
            "workspace_id": workspace_id,
            "range": time_range,
            "group_by": group_by,
        }
        response = self._client._get("/api/costs/breakdown", params=params)
        items = [self._parse_breakdown_item(b) for b in response.get("breakdown", [])]
        return items, response.get("total", 0.0)

    async def get_breakdown_async(
        self,
        workspace_id: str = "default",
        time_range: str = "7d",
        group_by: str = "provider",
    ) -> tuple[List[CostBreakdownItem], float]:
        """Async version of get_breakdown()."""
        params = {
            "workspace_id": workspace_id,
            "range": time_range,
            "group_by": group_by,
        }
        response = await self._client._get_async("/api/costs/breakdown", params=params)
        items = [self._parse_breakdown_item(b) for b in response.get("breakdown", [])]
        return items, response.get("total", 0.0)

    def get_timeline(
        self,
        workspace_id: str = "default",
        time_range: str = "7d",
    ) -> tuple[List[DailyCost], float, float]:
        """
        Get usage timeline data.

        Args:
            workspace_id: The workspace ID.
            time_range: Time range (24h, 7d, 30d, 90d).

        Returns:
            Tuple of (daily costs, total, average).
        """
        params = {"workspace_id": workspace_id, "range": time_range}
        response = self._client._get("/api/costs/timeline", params=params)
        daily = [self._parse_daily_cost(d) for d in response.get("timeline", [])]
        return daily, response.get("total", 0.0), response.get("average", 0.0)

    async def get_timeline_async(
        self,
        workspace_id: str = "default",
        time_range: str = "7d",
    ) -> tuple[List[DailyCost], float, float]:
        """Async version of get_timeline()."""
        params = {"workspace_id": workspace_id, "range": time_range}
        response = await self._client._get_async("/api/costs/timeline", params=params)
        daily = [self._parse_daily_cost(d) for d in response.get("timeline", [])]
        return daily, response.get("total", 0.0), response.get("average", 0.0)

    # =========================================================================
    # Alerts
    # =========================================================================

    def get_alerts(self, workspace_id: str = "default") -> List[CostAlert]:
        """
        Get budget alerts.

        Args:
            workspace_id: The workspace ID.

        Returns:
            List of CostAlert objects.
        """
        params = {"workspace_id": workspace_id}
        response = self._client._get("/api/costs/alerts", params=params)
        return [self._parse_alert(a) for a in response.get("alerts", [])]

    async def get_alerts_async(self, workspace_id: str = "default") -> List[CostAlert]:
        """Async version of get_alerts()."""
        params = {"workspace_id": workspace_id}
        response = await self._client._get_async("/api/costs/alerts", params=params)
        return [self._parse_alert(a) for a in response.get("alerts", [])]

    def dismiss_alert(self, alert_id: str, workspace_id: str = "default") -> bool:
        """
        Dismiss a budget alert.

        Args:
            alert_id: The alert ID.
            workspace_id: The workspace ID.

        Returns:
            True if successful.
        """
        self._client._post(
            f"/api/costs/alerts/{alert_id}/dismiss",
            {"workspace_id": workspace_id},
        )
        return True

    async def dismiss_alert_async(self, alert_id: str, workspace_id: str = "default") -> bool:
        """Async version of dismiss_alert()."""
        await self._client._post_async(
            f"/api/costs/alerts/{alert_id}/dismiss",
            {"workspace_id": workspace_id},
        )
        return True

    # =========================================================================
    # Budget Management
    # =========================================================================

    def set_budget(
        self,
        budget: float,
        workspace_id: str = "default",
        daily_limit: Optional[float] = None,
        name: Optional[str] = None,
    ) -> Budget:
        """
        Set budget limits.

        Args:
            budget: Monthly budget in USD.
            workspace_id: The workspace ID.
            daily_limit: Optional daily limit.
            name: Optional budget name.

        Returns:
            Budget object.
        """
        body: Dict[str, Any] = {
            "budget": budget,
            "workspace_id": workspace_id,
        }
        if daily_limit is not None:
            body["daily_limit"] = daily_limit
        if name:
            body["name"] = name

        response = self._client._post("/api/costs/budget", body)
        return Budget(
            workspace_id=response.get("workspace_id", workspace_id),
            monthly_limit=response.get("budget", budget),
            daily_limit=response.get("daily_limit"),
        )

    async def set_budget_async(
        self,
        budget: float,
        workspace_id: str = "default",
        daily_limit: Optional[float] = None,
        name: Optional[str] = None,
    ) -> Budget:
        """Async version of set_budget()."""
        body: Dict[str, Any] = {
            "budget": budget,
            "workspace_id": workspace_id,
        }
        if daily_limit is not None:
            body["daily_limit"] = daily_limit
        if name:
            body["name"] = name

        response = await self._client._post_async("/api/costs/budget", body)
        return Budget(
            workspace_id=response.get("workspace_id", workspace_id),
            monthly_limit=response.get("budget", budget),
            daily_limit=response.get("daily_limit"),
        )

    # =========================================================================
    # Recommendations
    # =========================================================================

    def get_recommendations(
        self,
        workspace_id: str = "default",
        status: Optional[str] = None,
        type_filter: Optional[str] = None,
    ) -> List[CostRecommendation]:
        """
        Get cost optimization recommendations.

        Args:
            workspace_id: The workspace ID.
            status: Filter by status (pending, applied, dismissed).
            type_filter: Filter by type (model_downgrade, caching, batching).

        Returns:
            List of CostRecommendation objects.
        """
        params: Dict[str, Any] = {"workspace_id": workspace_id}
        if status:
            params["status"] = status
        if type_filter:
            params["type"] = type_filter

        response = self._client._get("/api/costs/recommendations", params=params)
        return [self._parse_recommendation(r) for r in response.get("recommendations", [])]

    async def get_recommendations_async(
        self,
        workspace_id: str = "default",
        status: Optional[str] = None,
        type_filter: Optional[str] = None,
    ) -> List[CostRecommendation]:
        """Async version of get_recommendations()."""
        params: Dict[str, Any] = {"workspace_id": workspace_id}
        if status:
            params["status"] = status
        if type_filter:
            params["type"] = type_filter

        response = await self._client._get_async("/api/costs/recommendations", params=params)
        return [self._parse_recommendation(r) for r in response.get("recommendations", [])]

    def apply_recommendation(
        self, recommendation_id: str, user_id: Optional[str] = None
    ) -> CostRecommendation:
        """
        Apply a recommendation.

        Args:
            recommendation_id: The recommendation ID.
            user_id: User applying the recommendation.

        Returns:
            Updated CostRecommendation object.
        """
        body: Dict[str, Any] = {}
        if user_id:
            body["user_id"] = user_id

        response = self._client._post(f"/api/costs/recommendations/{recommendation_id}/apply", body)
        return self._parse_recommendation(response.get("recommendation", response))

    async def apply_recommendation_async(
        self, recommendation_id: str, user_id: Optional[str] = None
    ) -> CostRecommendation:
        """Async version of apply_recommendation()."""
        body: Dict[str, Any] = {}
        if user_id:
            body["user_id"] = user_id

        response = await self._client._post_async(
            f"/api/costs/recommendations/{recommendation_id}/apply", body
        )
        return self._parse_recommendation(response.get("recommendation", response))

    def dismiss_recommendation(self, recommendation_id: str) -> bool:
        """
        Dismiss a recommendation.

        Args:
            recommendation_id: The recommendation ID.

        Returns:
            True if successful.
        """
        self._client._post(f"/api/costs/recommendations/{recommendation_id}/dismiss", {})
        return True

    async def dismiss_recommendation_async(self, recommendation_id: str) -> bool:
        """Async version of dismiss_recommendation()."""
        await self._client._post_async(
            f"/api/costs/recommendations/{recommendation_id}/dismiss", {}
        )
        return True

    # =========================================================================
    # Efficiency Metrics
    # =========================================================================

    def get_efficiency(
        self,
        workspace_id: str = "default",
        time_range: str = "7d",
    ) -> EfficiencyMetrics:
        """
        Get efficiency metrics.

        Args:
            workspace_id: The workspace ID.
            time_range: Time range (24h, 7d, 30d).

        Returns:
            EfficiencyMetrics object.
        """
        params = {"workspace_id": workspace_id, "range": time_range}
        response = self._client._get("/api/costs/efficiency", params=params)
        return self._parse_efficiency(response)

    async def get_efficiency_async(
        self,
        workspace_id: str = "default",
        time_range: str = "7d",
    ) -> EfficiencyMetrics:
        """Async version of get_efficiency()."""
        params = {"workspace_id": workspace_id, "range": time_range}
        response = await self._client._get_async("/api/costs/efficiency", params=params)
        return self._parse_efficiency(response)

    # =========================================================================
    # Forecasting
    # =========================================================================

    def get_forecast(
        self,
        workspace_id: str = "default",
        days: int = 30,
    ) -> CostForecast:
        """
        Get cost forecast.

        Args:
            workspace_id: The workspace ID.
            days: Forecast days.

        Returns:
            CostForecast object.
        """
        params = {"workspace_id": workspace_id, "days": days}
        response = self._client._get("/api/costs/forecast", params=params)
        return self._parse_forecast(response)

    async def get_forecast_async(
        self,
        workspace_id: str = "default",
        days: int = 30,
    ) -> CostForecast:
        """Async version of get_forecast()."""
        params = {"workspace_id": workspace_id, "days": days}
        response = await self._client._get_async("/api/costs/forecast", params=params)
        return self._parse_forecast(response)

    def simulate_scenario(
        self,
        workspace_id: str,
        scenario: Dict[str, Any],
        days: int = 30,
    ) -> CostForecast:
        """
        Simulate a cost scenario.

        Args:
            workspace_id: The workspace ID.
            scenario: Scenario object with name, description, changes.
            days: Days to simulate.

        Returns:
            CostForecast object.
        """
        body = {
            "workspace_id": workspace_id,
            "scenario": scenario,
            "days": days,
        }
        response = self._client._post("/api/costs/forecast/simulate", body)
        return self._parse_forecast(response)

    async def simulate_scenario_async(
        self,
        workspace_id: str,
        scenario: Dict[str, Any],
        days: int = 30,
    ) -> CostForecast:
        """Async version of simulate_scenario()."""
        body = {
            "workspace_id": workspace_id,
            "scenario": scenario,
            "days": days,
        }
        response = await self._client._post_async("/api/costs/forecast/simulate", body)
        return self._parse_forecast(response)

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _parse_summary(self, data: Dict[str, Any]) -> CostSummary:
        """Parse summary data into CostSummary object."""
        cost_by_provider = [self._parse_breakdown_item(p) for p in data.get("costByProvider", [])]
        cost_by_feature = [self._parse_breakdown_item(f) for f in data.get("costByFeature", [])]
        daily_costs = [self._parse_daily_cost(d) for d in data.get("dailyCosts", [])]
        alerts = [self._parse_alert(a) for a in data.get("alerts", [])]

        return CostSummary(
            total_cost=data.get("totalCost", 0.0),
            budget=data.get("budget", 0.0),
            tokens_used=data.get("tokensUsed", 0),
            api_calls=data.get("apiCalls", 0),
            last_updated=data.get("lastUpdated", ""),
            cost_by_provider=cost_by_provider,
            cost_by_feature=cost_by_feature,
            daily_costs=daily_costs,
            alerts=alerts,
        )

    def _parse_breakdown_item(self, data: Dict[str, Any]) -> CostBreakdownItem:
        """Parse breakdown item data."""
        return CostBreakdownItem(
            name=data.get("name", ""),
            cost=data.get("cost", 0.0),
            percentage=data.get("percentage", 0.0),
        )

    def _parse_daily_cost(self, data: Dict[str, Any]) -> DailyCost:
        """Parse daily cost data."""
        return DailyCost(
            date=data.get("date", ""),
            cost=data.get("cost", 0.0),
            tokens=data.get("tokens", 0),
        )

    def _parse_alert(self, data: Dict[str, Any]) -> CostAlert:
        """Parse alert data."""
        return CostAlert(
            id=data.get("id", ""),
            type=data.get("type", ""),
            message=data.get("message", ""),
            severity=data.get("severity", "info"),
            timestamp=data.get("timestamp", ""),
            acknowledged=data.get("acknowledged", False),
        )

    def _parse_recommendation(self, data: Dict[str, Any]) -> CostRecommendation:
        """Parse recommendation data."""
        created_at = None
        if data.get("created_at"):
            try:
                created_at = datetime.fromisoformat(data["created_at"].replace("Z", "+00:00"))
            except (ValueError, TypeError):
                pass

        return CostRecommendation(
            id=data.get("id", ""),
            type=data.get("type", ""),
            title=data.get("title", ""),
            description=data.get("description", ""),
            estimated_savings=data.get("estimated_savings", 0.0),
            effort=data.get("effort", "medium"),
            status=data.get("status", "pending"),
            created_at=created_at,
        )

    def _parse_efficiency(self, data: Dict[str, Any]) -> EfficiencyMetrics:
        """Parse efficiency metrics data."""
        metrics = data.get("metrics", data)
        return EfficiencyMetrics(
            cost_per_1k_tokens=metrics.get("cost_per_1k_tokens", 0.0),
            tokens_per_call=metrics.get("tokens_per_call", 0.0),
            cost_per_call=metrics.get("cost_per_call", 0.0),
            total_tokens=metrics.get("total_tokens", 0),
            total_calls=metrics.get("total_calls", 0),
            total_cost=metrics.get("total_cost", 0.0),
            model_utilization=data.get("model_utilization", []),
        )

    def _parse_forecast(self, data: Dict[str, Any]) -> CostForecast:
        """Parse forecast data."""
        ci = data.get("confidence_interval", [0.0, 0.0])
        return CostForecast(
            workspace_id=data.get("workspace_id", "default"),
            forecast_days=data.get("forecast_days", 30),
            projected_cost=data.get("projected_cost", 0.0),
            confidence_interval=(ci[0] if len(ci) > 0 else 0.0, ci[1] if len(ci) > 1 else 0.0),
            trend=data.get("trend", "stable"),
            daily_projections=data.get("daily_projections", []),
        )


__all__ = [
    "CostManagementAPI",
    "CostSummary",
    "CostBreakdownItem",
    "DailyCost",
    "CostAlert",
    "Budget",
    "CostRecommendation",
    "EfficiencyMetrics",
    "CostForecast",
]
