"""
Cost Management Namespace API.

Provides methods for tracking and managing AI costs:
- Cost dashboard and summaries
- Cost breakdown by provider, feature, model
- Budget management and alerts
- Usage timeline and forecasting
- Optimization recommendations
- Pre-execution cost estimation
- What-if scenario simulation
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient

TimeRange = Literal["24h", "7d", "30d", "90d"]
GroupBy = Literal["provider", "feature", "model"]
TrendDirection = Literal["increasing", "decreasing", "stable"]
AlertSeverity = Literal["info", "warning", "critical"]
RecommendationStatus = Literal["pending", "applied", "dismissed", "expired", "partial"]
RecommendationPriority = Literal["critical", "high", "medium", "low"]
RecommendationType = Literal[
    "model_downgrade",
    "caching",
    "batching",
    "rate_limiting",
    "prompt_optimization",
    "provider_switch",
    "time_shifting",
    "quota_adjustment",
]
ThrottleLevel = Literal["none", "light", "medium", "heavy", "blocked"]
CostEnforcementMode = Literal["hard", "soft", "throttle", "estimate"]


class CostManagementAPI:
    """
    Synchronous Cost Management API.

    Provides methods for:
    - Cost dashboard and summaries
    - Budget management and alerts
    - Optimization recommendations
    - Cost forecasting and simulation
    - Pre-execution cost estimation
    """

    def __init__(self, client: AragoraClient) -> None:
        self._client = client

    # =========================================================================
    # Cost Dashboard
    # =========================================================================

    def get_summary(
        self,
        workspace_id: str | None = None,
        range: TimeRange | None = None,
    ) -> dict[str, Any]:
        """
        Get cost dashboard summary.

        Args:
            workspace_id: Optional workspace filter.
            range: Time range (24h, 7d, 30d, 90d).

        Returns:
            Cost summary with breakdowns and alerts.
        """
        params: dict[str, Any] = {}
        if workspace_id:
            params["workspace_id"] = workspace_id
        if range:
            params["range"] = range

        return self._client._request("GET", "/api/costs", params=params)

    def get_breakdown(
        self,
        workspace_id: str | None = None,
        range: TimeRange | None = None,
        group_by: GroupBy | None = None,
    ) -> dict[str, Any]:
        """
        Get detailed cost breakdown.

        Args:
            workspace_id: Optional workspace filter.
            range: Time range.
            group_by: Group by dimension (provider, feature, model).

        Returns:
            Cost breakdown with total.
        """
        params: dict[str, Any] = {}
        if workspace_id:
            params["workspace_id"] = workspace_id
        if range:
            params["range"] = range
        if group_by:
            params["group_by"] = group_by

        return self._client._request("GET", "/api/costs/breakdown", params=params)

    def get_timeline(
        self,
        workspace_id: str | None = None,
        range: TimeRange | None = None,
    ) -> dict[str, Any]:
        """
        Get usage timeline data.

        Args:
            workspace_id: Optional workspace filter.
            range: Time range.

        Returns:
            Timeline with daily costs and averages.
        """
        params: dict[str, Any] = {}
        if workspace_id:
            params["workspace_id"] = workspace_id
        if range:
            params["range"] = range

        return self._client._request("GET", "/api/costs/timeline", params=params)

    # =========================================================================
    # Alerts
    # =========================================================================

    def get_alerts(
        self,
        workspace_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Get budget alerts.

        Args:
            workspace_id: Optional workspace filter.

        Returns:
            List of cost alerts.
        """
        params: dict[str, Any] = {}
        if workspace_id:
            params["workspace_id"] = workspace_id

        return self._client._request("GET", "/api/costs/alerts", params=params)

    def dismiss_alert(
        self,
        alert_id: str,
        workspace_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Dismiss a budget alert.

        Args:
            alert_id: Alert identifier.
            workspace_id: Optional workspace.

        Returns:
            Success status.
        """
        data: dict[str, Any] = {}
        if workspace_id:
            data["workspace_id"] = workspace_id

        return self._client._request("POST", f"/api/costs/alerts/{alert_id}/dismiss", json=data)

    # =========================================================================
    # Budget Management
    # =========================================================================

    def set_budget(
        self,
        budget: float,
        workspace_id: str | None = None,
        daily_limit: float | None = None,
        name: str | None = None,
    ) -> dict[str, Any]:
        """
        Set budget limits.

        Args:
            budget: Monthly budget limit.
            workspace_id: Optional workspace.
            daily_limit: Optional daily limit.
            name: Budget name.

        Returns:
            Created budget configuration.
        """
        data: dict[str, Any] = {"budget": budget}
        if workspace_id:
            data["workspace_id"] = workspace_id
        if daily_limit is not None:
            data["daily_limit"] = daily_limit
        if name:
            data["name"] = name

        return self._client._request("POST", "/api/costs/budget", json=data)

    # =========================================================================
    # Recommendations
    # =========================================================================

    def get_recommendations(
        self,
        workspace_id: str | None = None,
        status: RecommendationStatus | None = None,
        type: str | None = None,
    ) -> dict[str, Any]:
        """
        Get cost optimization recommendations.

        Args:
            workspace_id: Optional workspace filter.
            status: Filter by status.
            type: Filter by recommendation type.

        Returns:
            List of recommendations.
        """
        params: dict[str, Any] = {}
        if workspace_id:
            params["workspace_id"] = workspace_id
        if status:
            params["status"] = status
        if type:
            params["type"] = type

        return self._client._request("GET", "/api/costs/recommendations", params=params)

    def get_detailed_recommendations(
        self,
        workspace_id: str | None = None,
        status: RecommendationStatus | None = None,
        type: RecommendationType | None = None,
        priority: RecommendationPriority | None = None,
    ) -> dict[str, Any]:
        """
        Get detailed recommendations with full analysis.

        Args:
            workspace_id: Optional workspace filter.
            status: Filter by status.
            type: Filter by type.
            priority: Filter by priority.

        Returns:
            Detailed recommendations with summary.
        """
        params: dict[str, Any] = {}
        if workspace_id:
            params["workspace_id"] = workspace_id
        if status:
            params["status"] = status
        if type:
            params["type"] = type
        if priority:
            params["priority"] = priority

        return self._client._request("GET", "/api/costs/recommendations/detailed", params=params)

    def get_recommendation(self, recommendation_id: str) -> dict[str, Any]:
        """
        Get a specific recommendation by ID.

        Args:
            recommendation_id: Recommendation identifier.

        Returns:
            Detailed recommendation.
        """
        return self._client._request("GET", f"/api/costs/recommendations/{recommendation_id}")

    def apply_recommendation(
        self,
        recommendation_id: str,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Apply a recommendation.

        Args:
            recommendation_id: Recommendation identifier.
            user_id: Optional user applying the recommendation.

        Returns:
            Applied recommendation.
        """
        data: dict[str, Any] = {}
        if user_id:
            data["user_id"] = user_id

        return self._client._request(
            "POST", f"/api/costs/recommendations/{recommendation_id}/apply", json=data
        )

    def dismiss_recommendation(self, recommendation_id: str) -> dict[str, Any]:
        """
        Dismiss a recommendation.

        Args:
            recommendation_id: Recommendation identifier.

        Returns:
            Success status.
        """
        return self._client._request(
            "POST", f"/api/costs/recommendations/{recommendation_id}/dismiss"
        )

    # =========================================================================
    # Efficiency Metrics
    # =========================================================================

    def get_efficiency(
        self,
        workspace_id: str | None = None,
        range: TimeRange | None = None,
    ) -> dict[str, Any]:
        """
        Get efficiency metrics.

        Args:
            workspace_id: Optional workspace filter.
            range: Time range.

        Returns:
            Efficiency metrics including cost per token.
        """
        params: dict[str, Any] = {}
        if workspace_id:
            params["workspace_id"] = workspace_id
        if range:
            params["range"] = range

        return self._client._request("GET", "/api/costs/efficiency", params=params)

    # =========================================================================
    # Cost Estimation
    # =========================================================================

    def estimate_cost(
        self,
        task: str,
        model: str | None = None,
        agents: list[str] | None = None,
        rounds: int | None = None,
        workspace_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Estimate cost for a task before execution.

        Args:
            task: Task description.
            model: Target model.
            agents: List of agents to use.
            rounds: Number of debate rounds.
            workspace_id: Optional workspace.

        Returns:
            Cost estimate with confidence.
        """
        data: dict[str, Any] = {"task": task}
        if model:
            data["model"] = model
        if agents:
            data["agents"] = agents
        if rounds is not None:
            data["rounds"] = rounds
        if workspace_id:
            data["workspace_id"] = workspace_id

        return self._client._request("POST", "/api/costs/estimate", json=data)

    def check_constraints(
        self,
        task: str,
        model: str | None = None,
        workspace_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Check if a task is allowed under current budget constraints.

        Args:
            task: Task description.
            model: Target model.
            workspace_id: Optional workspace.

        Returns:
            Constraint check result with throttle level.
        """
        data: dict[str, Any] = {"task": task}
        if model:
            data["model"] = model
        if workspace_id:
            data["workspace_id"] = workspace_id

        return self._client._request("POST", "/api/costs/constraints/check", json=data)

    # =========================================================================
    # Forecasting
    # =========================================================================

    def get_forecast(
        self,
        workspace_id: str | None = None,
        days: int | None = None,
    ) -> dict[str, Any]:
        """
        Get basic cost forecast.

        Args:
            workspace_id: Optional workspace filter.
            days: Forecast period in days.

        Returns:
            Cost forecast with projections.
        """
        params: dict[str, Any] = {}
        if workspace_id:
            params["workspace_id"] = workspace_id
        if days is not None:
            params["days"] = days

        return self._client._request("GET", "/api/costs/forecast", params=params)

    def get_detailed_forecast(
        self,
        workspace_id: str | None = None,
        days: int | None = None,
        include_alerts: bool = True,
    ) -> dict[str, Any]:
        """
        Get detailed forecast report with trend analysis.

        Args:
            workspace_id: Optional workspace filter.
            days: Forecast period in days.
            include_alerts: Include forecast alerts.

        Returns:
            Detailed forecast report.
        """
        params: dict[str, Any] = {"include_alerts": include_alerts}
        if workspace_id:
            params["workspace_id"] = workspace_id
        if days is not None:
            params["days"] = days

        return self._client._request("GET", "/api/costs/forecast/detailed", params=params)

    def simulate_scenario(
        self,
        workspace_id: str,
        scenario: dict[str, Any],
        days: int | None = None,
    ) -> dict[str, Any]:
        """
        Simulate a cost scenario.

        Args:
            workspace_id: Workspace identifier.
            scenario: Scenario definition with name and changes.
            days: Simulation period in days.

        Returns:
            Simulation result with cost comparison.
        """
        data: dict[str, Any] = {
            "workspace_id": workspace_id,
            "scenario": scenario,
        }
        if days is not None:
            data["days"] = days

        return self._client._request("POST", "/api/costs/forecast/simulate", json=data)


class AsyncCostManagementAPI:
    """Asynchronous Cost Management API."""

    def __init__(self, client: AragoraAsyncClient) -> None:
        self._client = client

    # =========================================================================
    # Cost Dashboard
    # =========================================================================

    async def get_summary(
        self,
        workspace_id: str | None = None,
        range: TimeRange | None = None,
    ) -> dict[str, Any]:
        """Get cost dashboard summary."""
        params: dict[str, Any] = {}
        if workspace_id:
            params["workspace_id"] = workspace_id
        if range:
            params["range"] = range

        return await self._client._request("GET", "/api/costs", params=params)

    async def get_breakdown(
        self,
        workspace_id: str | None = None,
        range: TimeRange | None = None,
        group_by: GroupBy | None = None,
    ) -> dict[str, Any]:
        """Get detailed cost breakdown."""
        params: dict[str, Any] = {}
        if workspace_id:
            params["workspace_id"] = workspace_id
        if range:
            params["range"] = range
        if group_by:
            params["group_by"] = group_by

        return await self._client._request("GET", "/api/costs/breakdown", params=params)

    async def get_timeline(
        self,
        workspace_id: str | None = None,
        range: TimeRange | None = None,
    ) -> dict[str, Any]:
        """Get usage timeline data."""
        params: dict[str, Any] = {}
        if workspace_id:
            params["workspace_id"] = workspace_id
        if range:
            params["range"] = range

        return await self._client._request("GET", "/api/costs/timeline", params=params)

    # =========================================================================
    # Alerts
    # =========================================================================

    async def get_alerts(
        self,
        workspace_id: str | None = None,
    ) -> dict[str, Any]:
        """Get budget alerts."""
        params: dict[str, Any] = {}
        if workspace_id:
            params["workspace_id"] = workspace_id

        return await self._client._request("GET", "/api/costs/alerts", params=params)

    async def dismiss_alert(
        self,
        alert_id: str,
        workspace_id: str | None = None,
    ) -> dict[str, Any]:
        """Dismiss a budget alert."""
        data: dict[str, Any] = {}
        if workspace_id:
            data["workspace_id"] = workspace_id

        return await self._client._request(
            "POST", f"/api/costs/alerts/{alert_id}/dismiss", json=data
        )

    # =========================================================================
    # Budget Management
    # =========================================================================

    async def set_budget(
        self,
        budget: float,
        workspace_id: str | None = None,
        daily_limit: float | None = None,
        name: str | None = None,
    ) -> dict[str, Any]:
        """Set budget limits."""
        data: dict[str, Any] = {"budget": budget}
        if workspace_id:
            data["workspace_id"] = workspace_id
        if daily_limit is not None:
            data["daily_limit"] = daily_limit
        if name:
            data["name"] = name

        return await self._client._request("POST", "/api/costs/budget", json=data)

    # =========================================================================
    # Recommendations
    # =========================================================================

    async def get_recommendations(
        self,
        workspace_id: str | None = None,
        status: RecommendationStatus | None = None,
        type: str | None = None,
    ) -> dict[str, Any]:
        """Get cost optimization recommendations."""
        params: dict[str, Any] = {}
        if workspace_id:
            params["workspace_id"] = workspace_id
        if status:
            params["status"] = status
        if type:
            params["type"] = type

        return await self._client._request("GET", "/api/costs/recommendations", params=params)

    async def get_detailed_recommendations(
        self,
        workspace_id: str | None = None,
        status: RecommendationStatus | None = None,
        type: RecommendationType | None = None,
        priority: RecommendationPriority | None = None,
    ) -> dict[str, Any]:
        """Get detailed recommendations with full analysis."""
        params: dict[str, Any] = {}
        if workspace_id:
            params["workspace_id"] = workspace_id
        if status:
            params["status"] = status
        if type:
            params["type"] = type
        if priority:
            params["priority"] = priority

        return await self._client._request(
            "GET", "/api/costs/recommendations/detailed", params=params
        )

    async def get_recommendation(self, recommendation_id: str) -> dict[str, Any]:
        """Get a specific recommendation by ID."""
        return await self._client._request("GET", f"/api/costs/recommendations/{recommendation_id}")

    async def apply_recommendation(
        self,
        recommendation_id: str,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        """Apply a recommendation."""
        data: dict[str, Any] = {}
        if user_id:
            data["user_id"] = user_id

        return await self._client._request(
            "POST", f"/api/costs/recommendations/{recommendation_id}/apply", json=data
        )

    async def dismiss_recommendation(self, recommendation_id: str) -> dict[str, Any]:
        """Dismiss a recommendation."""
        return await self._client._request(
            "POST", f"/api/costs/recommendations/{recommendation_id}/dismiss"
        )

    # =========================================================================
    # Efficiency Metrics
    # =========================================================================

    async def get_efficiency(
        self,
        workspace_id: str | None = None,
        range: TimeRange | None = None,
    ) -> dict[str, Any]:
        """Get efficiency metrics."""
        params: dict[str, Any] = {}
        if workspace_id:
            params["workspace_id"] = workspace_id
        if range:
            params["range"] = range

        return await self._client._request("GET", "/api/costs/efficiency", params=params)

    # =========================================================================
    # Cost Estimation
    # =========================================================================

    async def estimate_cost(
        self,
        task: str,
        model: str | None = None,
        agents: list[str] | None = None,
        rounds: int | None = None,
        workspace_id: str | None = None,
    ) -> dict[str, Any]:
        """Estimate cost for a task before execution."""
        data: dict[str, Any] = {"task": task}
        if model:
            data["model"] = model
        if agents:
            data["agents"] = agents
        if rounds is not None:
            data["rounds"] = rounds
        if workspace_id:
            data["workspace_id"] = workspace_id

        return await self._client._request("POST", "/api/costs/estimate", json=data)

    async def check_constraints(
        self,
        task: str,
        model: str | None = None,
        workspace_id: str | None = None,
    ) -> dict[str, Any]:
        """Check if a task is allowed under current budget constraints."""
        data: dict[str, Any] = {"task": task}
        if model:
            data["model"] = model
        if workspace_id:
            data["workspace_id"] = workspace_id

        return await self._client._request("POST", "/api/costs/constraints/check", json=data)

    # =========================================================================
    # Forecasting
    # =========================================================================

    async def get_forecast(
        self,
        workspace_id: str | None = None,
        days: int | None = None,
    ) -> dict[str, Any]:
        """Get basic cost forecast."""
        params: dict[str, Any] = {}
        if workspace_id:
            params["workspace_id"] = workspace_id
        if days is not None:
            params["days"] = days

        return await self._client._request("GET", "/api/costs/forecast", params=params)

    async def get_detailed_forecast(
        self,
        workspace_id: str | None = None,
        days: int | None = None,
        include_alerts: bool = True,
    ) -> dict[str, Any]:
        """Get detailed forecast report with trend analysis."""
        params: dict[str, Any] = {"include_alerts": include_alerts}
        if workspace_id:
            params["workspace_id"] = workspace_id
        if days is not None:
            params["days"] = days

        return await self._client._request("GET", "/api/costs/forecast/detailed", params=params)

    async def simulate_scenario(
        self,
        workspace_id: str,
        scenario: dict[str, Any],
        days: int | None = None,
    ) -> dict[str, Any]:
        """Simulate a cost scenario."""
        data: dict[str, Any] = {
            "workspace_id": workspace_id,
            "scenario": scenario,
        }
        if days is not None:
            data["days"] = days

        return await self._client._request("POST", "/api/costs/forecast/simulate", json=data)
