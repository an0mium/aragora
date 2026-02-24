"""
Costs Namespace API

Provides methods for cost tracking and management:
- View usage costs and breakdowns
- Manage budgets and alerts
- Get cost optimization recommendations
- Cost forecasting and simulation
- Export cost data
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class CostsAPI:
    """
    Synchronous Costs API.

    Provides methods for AI cost tracking, budget management,
    optimization recommendations, and forecasting.

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai")
        >>> summary = client.costs.get_summary()
        >>> alerts = client.costs.get_alerts()
    """

    def __init__(self, client: AragoraClient):
        self._client = client

    # ===========================================================================
    # Core Cost Data
    # ===========================================================================

    def get_summary(
        self, period: str | None = None, workspace_id: str | None = None
    ) -> dict[str, Any]:
        """
        Get cost summary dashboard data.

        Args:
            period: Time range (24h, 7d, 30d, 90d)
            workspace_id: Workspace ID

        Returns:
            Dict with cost summary including totals, budgets, and breakdowns
        """
        params: dict[str, Any] = {}
        if period:
            params["range"] = period
        if workspace_id:
            params["workspace_id"] = workspace_id
        return self._client.request("GET", "/api/v1/costs", params=params)

    def get_breakdown(
        self, group_by: str = "provider", period: str | None = None, workspace_id: str | None = None
    ) -> dict[str, Any]:
        """
        Get cost breakdown by provider, feature, or model.

        Args:
            group_by: Grouping dimension (provider, feature, model)
            period: Time range (24h, 7d, 30d, 90d)
            workspace_id: Workspace ID

        Returns:
            Dict with grouped cost breakdown
        """
        params: dict[str, Any] = {"group_by": group_by}
        if period:
            params["range"] = period
        if workspace_id:
            params["workspace_id"] = workspace_id
        return self._client.request("GET", "/api/v1/costs/breakdown", params=params)

    def get_timeline(
        self, period: str | None = None, workspace_id: str | None = None
    ) -> dict[str, Any]:
        """
        Get cost timeline data.

        Args:
            period: Time range (24h, 7d, 30d, 90d)
            workspace_id: Workspace ID

        Returns:
            Dict with timeline cost data points
        """
        params: dict[str, Any] = {}
        if period:
            params["range"] = period
        if workspace_id:
            params["workspace_id"] = workspace_id
        return self._client.request("GET", "/api/v1/costs/timeline", params=params)

    def get_usage(
        self, period: str | None = None, group_by: str = "provider", workspace_id: str | None = None
    ) -> dict[str, Any]:
        """
        Get detailed usage tracking data.

        Args:
            period: Time range (24h, 7d, 30d, 90d)
            group_by: Grouping (provider, model, operation)
            workspace_id: Workspace ID

        Returns:
            Dict with usage breakdown and totals
        """
        params: dict[str, Any] = {"group_by": group_by}
        if period:
            params["range"] = period
        if workspace_id:
            params["workspace_id"] = workspace_id
        return self._client.request("GET", "/api/v1/costs/usage", params=params)

    def get_efficiency(
        self, period: str | None = None, workspace_id: str | None = None
    ) -> dict[str, Any]:
        """
        Get cost efficiency metrics.

        Args:
            period: Time range (24h, 7d, 30d)
            workspace_id: Workspace ID

        Returns:
            Dict with efficiency metrics (cost per token, utilization)
        """
        params: dict[str, Any] = {}
        if period:
            params["range"] = period
        if workspace_id:
            params["workspace_id"] = workspace_id
        return self._client.request("GET", "/api/v1/costs/efficiency", params=params)

    def export(
        self,
        fmt: str = "json",
        period: str | None = None,
        group_by: str = "daily",
        workspace_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Export cost data as CSV or JSON.

        Args:
            fmt: Export format (csv, json)
            period: Time range (24h, 7d, 30d, 90d)
            group_by: Grouping (daily, provider, feature)
            workspace_id: Workspace ID

        Returns:
            Exported cost data
        """
        params: dict[str, Any] = {"format": fmt, "group_by": group_by}
        if period:
            params["range"] = period
        if workspace_id:
            params["workspace_id"] = workspace_id
        return self._client.request("GET", "/api/v1/costs/export", params=params)

    # ===========================================================================
    # Alerts
    # ===========================================================================

    def get_alerts(self, workspace_id: str | None = None) -> dict[str, Any]:
        """
        Get budget alerts.

        Args:
            workspace_id: Workspace ID

        Returns:
            Dict with active alerts list
        """
        params: dict[str, Any] = {}
        if workspace_id:
            params["workspace_id"] = workspace_id
        return self._client.request("GET", "/api/v1/costs/alerts", params=params)

    def create_alert(
        self,
        name: str,
        alert_type: str = "budget_threshold",
        threshold: float = 80,
        notification_channels: list[str] | None = None,
        workspace_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Create a cost alert.

        Args:
            name: Alert name
            alert_type: Type (budget_threshold, spike_detection, daily_limit)
            threshold: Threshold value
            notification_channels: Notification channels (email, slack, webhook)
            workspace_id: Workspace ID

        Returns:
            Dict with created alert details
        """
        body: dict[str, Any] = {"name": name, "type": alert_type, "threshold": threshold}
        if notification_channels:
            body["notification_channels"] = notification_channels
        if workspace_id:
            body["workspace_id"] = workspace_id
        return self._client.request("POST", "/api/v1/costs/alerts", json=body)

    def dismiss_alert(self, alert_id: str, workspace_id: str | None = None) -> dict[str, Any]:
        """
        Dismiss a budget alert.

        Args:
            alert_id: Alert ID to dismiss
            workspace_id: Workspace ID

        Returns:
            Dict confirming dismissal
        """
        params: dict[str, Any] = {}
        if workspace_id:
            params["workspace_id"] = workspace_id
        return self._client.request(
            "POST", f"/api/v1/costs/alerts/{alert_id}/dismiss", params=params
        )

    # ===========================================================================
    # Budgets
    # ===========================================================================

    def set_budget(
        self,
        budget: float,
        workspace_id: str | None = None,
        daily_limit: float | None = None,
        name: str | None = None,
    ) -> dict[str, Any]:
        """
        Set budget limits (legacy endpoint).

        Args:
            budget: Monthly budget in USD
            workspace_id: Workspace ID
            daily_limit: Optional daily spending limit
            name: Optional budget name

        Returns:
            Dict confirming budget was set
        """
        body: dict[str, Any] = {"budget": budget}
        if workspace_id:
            body["workspace_id"] = workspace_id
        if daily_limit is not None:
            body["daily_limit"] = daily_limit
        if name:
            body["name"] = name
        return self._client.request("POST", "/api/v1/costs/budget", json=body)

    def list_budgets(self, workspace_id: str | None = None) -> dict[str, Any]:
        """
        List all budgets for the workspace.

        Args:
            workspace_id: Workspace ID

        Returns:
            Dict with budgets list and count
        """
        params: dict[str, Any] = {}
        if workspace_id:
            params["workspace_id"] = workspace_id
        return self._client.request("GET", "/api/v1/costs/budgets", params=params)

    def create_budget(
        self,
        monthly_limit_usd: float,
        workspace_id: str | None = None,
        name: str | None = None,
        daily_limit_usd: float | None = None,
        alert_thresholds: list[int] | None = None,
    ) -> dict[str, Any]:
        """
        Create a new budget.

        Args:
            monthly_limit_usd: Monthly spending limit in USD
            workspace_id: Workspace ID
            name: Budget name
            daily_limit_usd: Optional daily spending limit
            alert_thresholds: Alert thresholds as percentages

        Returns:
            Dict with created budget details
        """
        body: dict[str, Any] = {"monthly_limit_usd": monthly_limit_usd}
        if workspace_id:
            body["workspace_id"] = workspace_id
        if name:
            body["name"] = name
        if daily_limit_usd is not None:
            body["daily_limit_usd"] = daily_limit_usd
        if alert_thresholds:
            body["alert_thresholds"] = alert_thresholds
        return self._client.request("POST", "/api/v1/costs/budgets", json=body)

    # ===========================================================================
    # Recommendations
    # ===========================================================================

    def get_recommendations(
        self,
        workspace_id: str | None = None,
        status: str | None = None,
        rec_type: str | None = None,
    ) -> dict[str, Any]:
        """
        Get cost optimization recommendations.

        Args:
            workspace_id: Workspace ID
            status: Filter by status (pending, applied, dismissed)
            rec_type: Filter by type (model_downgrade, caching, batching)

        Returns:
            Dict with recommendations list and summary
        """
        params: dict[str, Any] = {}
        if workspace_id:
            params["workspace_id"] = workspace_id
        if status:
            params["status"] = status
        if rec_type:
            params["type"] = rec_type
        return self._client.request("GET", "/api/v1/costs/recommendations", params=params)

    def get_recommendations_detailed(
        self,
        workspace_id: str | None = None,
        include_implementation: bool = True,
        min_savings: float = 0,
    ) -> dict[str, Any]:
        """
        Get detailed cost optimization recommendations with implementation steps.

        Args:
            workspace_id: Workspace ID
            include_implementation: Include implementation steps
            min_savings: Minimum savings threshold in USD

        Returns:
            Dict with detailed recommendations and total potential savings
        """
        params: dict[str, Any] = {
            "include_implementation": str(include_implementation).lower(),
            "min_savings": str(min_savings),
        }
        if workspace_id:
            params["workspace_id"] = workspace_id
        return self._client.request("GET", "/api/v1/costs/recommendations/detailed", params=params)

    def get_recommendation(self, recommendation_id: str) -> dict[str, Any]:
        """
        Get a specific cost optimization recommendation.

        Args:
            recommendation_id: Recommendation ID

        Returns:
            Dict with recommendation details
        """
        return self._client.request("GET", f"/api/v1/costs/recommendations/{recommendation_id}")

    def apply_recommendation(
        self, recommendation_id: str, user_id: str | None = None
    ) -> dict[str, Any]:
        """
        Apply a cost optimization recommendation.

        Args:
            recommendation_id: Recommendation ID
            user_id: User applying the recommendation

        Returns:
            Dict confirming recommendation was applied
        """
        body: dict[str, Any] = {}
        if user_id:
            body["user_id"] = user_id
        return self._client.request(
            "POST", f"/api/v1/costs/recommendations/{recommendation_id}/apply", json=body
        )

    def dismiss_recommendation(self, recommendation_id: str) -> dict[str, Any]:
        """
        Dismiss a cost optimization recommendation.

        Args:
            recommendation_id: Recommendation ID

        Returns:
            Dict confirming recommendation was dismissed
        """
        return self._client.request(
            "POST", f"/api/v1/costs/recommendations/{recommendation_id}/dismiss"
        )

    # ===========================================================================
    # Forecasting
    # ===========================================================================

    def get_forecast(
        self, workspace_id: str | None = None, days: int | None = None
    ) -> dict[str, Any]:
        """
        Get cost forecast.

        Args:
            workspace_id: Workspace ID
            days: Number of forecast days (default: 30)

        Returns:
            Dict with cost forecast report
        """
        params: dict[str, Any] = {}
        if workspace_id:
            params["workspace_id"] = workspace_id
        if days is not None:
            params["days"] = days
        return self._client.request("GET", "/api/v1/costs/forecast", params=params)

    def get_forecast_detailed(
        self,
        workspace_id: str | None = None,
        days: int | None = None,
        include_confidence: bool = True,
    ) -> dict[str, Any]:
        """
        Get detailed cost forecast with daily breakdowns and confidence intervals.

        Args:
            workspace_id: Workspace ID
            days: Number of forecast days (default: 30, max: 90)
            include_confidence: Include confidence intervals

        Returns:
            Dict with detailed forecast including daily projections
        """
        params: dict[str, Any] = {"include_confidence": str(include_confidence).lower()}
        if workspace_id:
            params["workspace_id"] = workspace_id
        if days is not None:
            params["days"] = days
        return self._client.request("GET", "/api/v1/costs/forecast/detailed", params=params)

    def simulate_forecast(
        self, scenario: dict[str, Any], workspace_id: str | None = None, days: int | None = None
    ) -> dict[str, Any]:
        """
        Simulate a cost scenario.

        Args:
            scenario: Scenario object with name, description, changes
            workspace_id: Workspace ID
            days: Days to simulate (default: 30)

        Returns:
            Dict with simulation results
        """
        body: dict[str, Any] = {"scenario": scenario}
        if workspace_id:
            body["workspace_id"] = workspace_id
        if days is not None:
            body["days"] = days
        return self._client.request("POST", "/api/v1/costs/forecast/simulate", json=body)

    # ===========================================================================
    # Constraints and Estimates
    # ===========================================================================

    def check_constraints(
        self,
        estimated_cost_usd: float,
        workspace_id: str | None = None,
        operation: str | None = None,
    ) -> dict[str, Any]:
        """
        Pre-flight check if an operation would exceed budget constraints.

        Args:
            estimated_cost_usd: Estimated cost of the operation in USD
            workspace_id: Workspace ID
            operation: Operation type

        Returns:
            Dict with constraint check result (allowed, reason, remaining budget)
        """
        body: dict[str, Any] = {"estimated_cost_usd": estimated_cost_usd}
        if workspace_id:
            body["workspace_id"] = workspace_id
        if operation:
            body["operation"] = operation
        return self._client.request("POST", "/api/v1/costs/constraints/check", json=body)

    def estimate(
        self,
        operation: str,
        tokens_input: int = 0,
        tokens_output: int = 0,
        model: str | None = None,
        provider: str | None = None,
    ) -> dict[str, Any]:
        """
        Estimate the cost of an operation.

        Args:
            operation: Operation type (debate, analysis, etc.)
            tokens_input: Estimated input tokens
            tokens_output: Estimated output tokens
            model: Model to use
            provider: Provider name

        Returns:
            Dict with estimated cost and pricing breakdown
        """
        body: dict[str, Any] = {
            "operation": operation,
            "tokens_input": tokens_input,
            "tokens_output": tokens_output,
        }
        if model:
            body["model"] = model
        if provider:
            body["provider"] = provider
        return self._client.request("POST", "/api/v1/costs/estimate", json=body)


class AsyncCostsAPI:
    """
    Asynchronous Costs API.

    Provides async methods for AI cost tracking, budget management,
    optimization recommendations, and forecasting.

    Example:
        >>> async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
        ...     summary = await client.costs.get_summary()
        ...     alerts = await client.costs.get_alerts()
    """

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    # ===========================================================================
    # Core Cost Data
    # ===========================================================================

    async def get_summary(
        self, period: str | None = None, workspace_id: str | None = None
    ) -> dict[str, Any]:
        """Get cost summary dashboard data."""
        params: dict[str, Any] = {}
        if period:
            params["range"] = period
        if workspace_id:
            params["workspace_id"] = workspace_id
        return await self._client.request("GET", "/api/v1/costs", params=params)

    async def get_breakdown(
        self, group_by: str = "provider", period: str | None = None, workspace_id: str | None = None
    ) -> dict[str, Any]:
        """Get cost breakdown by provider, feature, or model."""
        params: dict[str, Any] = {"group_by": group_by}
        if period:
            params["range"] = period
        if workspace_id:
            params["workspace_id"] = workspace_id
        return await self._client.request("GET", "/api/v1/costs/breakdown", params=params)

    async def get_timeline(
        self, period: str | None = None, workspace_id: str | None = None
    ) -> dict[str, Any]:
        """Get cost timeline data."""
        params: dict[str, Any] = {}
        if period:
            params["range"] = period
        if workspace_id:
            params["workspace_id"] = workspace_id
        return await self._client.request("GET", "/api/v1/costs/timeline", params=params)

    async def get_usage(
        self, period: str | None = None, group_by: str = "provider", workspace_id: str | None = None
    ) -> dict[str, Any]:
        """Get detailed usage tracking data."""
        params: dict[str, Any] = {"group_by": group_by}
        if period:
            params["range"] = period
        if workspace_id:
            params["workspace_id"] = workspace_id
        return await self._client.request("GET", "/api/v1/costs/usage", params=params)

    async def get_efficiency(
        self, period: str | None = None, workspace_id: str | None = None
    ) -> dict[str, Any]:
        """Get cost efficiency metrics."""
        params: dict[str, Any] = {}
        if period:
            params["range"] = period
        if workspace_id:
            params["workspace_id"] = workspace_id
        return await self._client.request("GET", "/api/v1/costs/efficiency", params=params)

    async def export(
        self,
        fmt: str = "json",
        period: str | None = None,
        group_by: str = "daily",
        workspace_id: str | None = None,
    ) -> dict[str, Any]:
        """Export cost data as CSV or JSON."""
        params: dict[str, Any] = {"format": fmt, "group_by": group_by}
        if period:
            params["range"] = period
        if workspace_id:
            params["workspace_id"] = workspace_id
        return await self._client.request("GET", "/api/v1/costs/export", params=params)

    # ===========================================================================
    # Alerts
    # ===========================================================================

    async def get_alerts(self, workspace_id: str | None = None) -> dict[str, Any]:
        """Get budget alerts."""
        params: dict[str, Any] = {}
        if workspace_id:
            params["workspace_id"] = workspace_id
        return await self._client.request("GET", "/api/v1/costs/alerts", params=params)

    async def create_alert(
        self,
        name: str,
        alert_type: str = "budget_threshold",
        threshold: float = 80,
        notification_channels: list[str] | None = None,
        workspace_id: str | None = None,
    ) -> dict[str, Any]:
        """Create a cost alert."""
        body: dict[str, Any] = {"name": name, "type": alert_type, "threshold": threshold}
        if notification_channels:
            body["notification_channels"] = notification_channels
        if workspace_id:
            body["workspace_id"] = workspace_id
        return await self._client.request("POST", "/api/v1/costs/alerts", json=body)

    async def dismiss_alert(self, alert_id: str, workspace_id: str | None = None) -> dict[str, Any]:
        """Dismiss a budget alert."""
        params: dict[str, Any] = {}
        if workspace_id:
            params["workspace_id"] = workspace_id
        return await self._client.request(
            "POST", f"/api/v1/costs/alerts/{alert_id}/dismiss", params=params
        )

    # ===========================================================================
    # Budgets
    # ===========================================================================

    async def set_budget(
        self,
        budget: float,
        workspace_id: str | None = None,
        daily_limit: float | None = None,
        name: str | None = None,
    ) -> dict[str, Any]:
        """Set budget limits (legacy endpoint)."""
        body: dict[str, Any] = {"budget": budget}
        if workspace_id:
            body["workspace_id"] = workspace_id
        if daily_limit is not None:
            body["daily_limit"] = daily_limit
        if name:
            body["name"] = name
        return await self._client.request("POST", "/api/v1/costs/budget", json=body)

    async def list_budgets(self, workspace_id: str | None = None) -> dict[str, Any]:
        """List all budgets for the workspace."""
        params: dict[str, Any] = {}
        if workspace_id:
            params["workspace_id"] = workspace_id
        return await self._client.request("GET", "/api/v1/costs/budgets", params=params)

    async def create_budget(
        self,
        monthly_limit_usd: float,
        workspace_id: str | None = None,
        name: str | None = None,
        daily_limit_usd: float | None = None,
        alert_thresholds: list[int] | None = None,
    ) -> dict[str, Any]:
        """Create a new budget."""
        body: dict[str, Any] = {"monthly_limit_usd": monthly_limit_usd}
        if workspace_id:
            body["workspace_id"] = workspace_id
        if name:
            body["name"] = name
        if daily_limit_usd is not None:
            body["daily_limit_usd"] = daily_limit_usd
        if alert_thresholds:
            body["alert_thresholds"] = alert_thresholds
        return await self._client.request("POST", "/api/v1/costs/budgets", json=body)

    # ===========================================================================
    # Recommendations
    # ===========================================================================

    async def get_recommendations(
        self,
        workspace_id: str | None = None,
        status: str | None = None,
        rec_type: str | None = None,
    ) -> dict[str, Any]:
        """Get cost optimization recommendations."""
        params: dict[str, Any] = {}
        if workspace_id:
            params["workspace_id"] = workspace_id
        if status:
            params["status"] = status
        if rec_type:
            params["type"] = rec_type
        return await self._client.request("GET", "/api/v1/costs/recommendations", params=params)

    async def get_recommendations_detailed(
        self,
        workspace_id: str | None = None,
        include_implementation: bool = True,
        min_savings: float = 0,
    ) -> dict[str, Any]:
        """Get detailed cost optimization recommendations with implementation steps."""
        params: dict[str, Any] = {
            "include_implementation": str(include_implementation).lower(),
            "min_savings": str(min_savings),
        }
        if workspace_id:
            params["workspace_id"] = workspace_id
        return await self._client.request(
            "GET", "/api/v1/costs/recommendations/detailed", params=params
        )

    async def get_recommendation(self, recommendation_id: str) -> dict[str, Any]:
        """Get a specific cost optimization recommendation."""
        return await self._client.request(
            "GET", f"/api/v1/costs/recommendations/{recommendation_id}"
        )

    async def apply_recommendation(
        self, recommendation_id: str, user_id: str | None = None
    ) -> dict[str, Any]:
        """Apply a cost optimization recommendation."""
        body: dict[str, Any] = {}
        if user_id:
            body["user_id"] = user_id
        return await self._client.request(
            "POST", f"/api/v1/costs/recommendations/{recommendation_id}/apply", json=body
        )

    async def dismiss_recommendation(self, recommendation_id: str) -> dict[str, Any]:
        """Dismiss a cost optimization recommendation."""
        return await self._client.request(
            "POST", f"/api/v1/costs/recommendations/{recommendation_id}/dismiss"
        )

    # ===========================================================================
    # Forecasting
    # ===========================================================================

    async def get_forecast(
        self, workspace_id: str | None = None, days: int | None = None
    ) -> dict[str, Any]:
        """Get cost forecast."""
        params: dict[str, Any] = {}
        if workspace_id:
            params["workspace_id"] = workspace_id
        if days is not None:
            params["days"] = days
        return await self._client.request("GET", "/api/v1/costs/forecast", params=params)

    async def get_forecast_detailed(
        self,
        workspace_id: str | None = None,
        days: int | None = None,
        include_confidence: bool = True,
    ) -> dict[str, Any]:
        """Get detailed cost forecast with daily breakdowns and confidence intervals."""
        params: dict[str, Any] = {"include_confidence": str(include_confidence).lower()}
        if workspace_id:
            params["workspace_id"] = workspace_id
        if days is not None:
            params["days"] = days
        return await self._client.request("GET", "/api/v1/costs/forecast/detailed", params=params)

    async def simulate_forecast(
        self, scenario: dict[str, Any], workspace_id: str | None = None, days: int | None = None
    ) -> dict[str, Any]:
        """Simulate a cost scenario."""
        body: dict[str, Any] = {"scenario": scenario}
        if workspace_id:
            body["workspace_id"] = workspace_id
        if days is not None:
            body["days"] = days
        return await self._client.request("POST", "/api/v1/costs/forecast/simulate", json=body)

    # ===========================================================================
    # Constraints and Estimates
    # ===========================================================================

    async def check_constraints(
        self,
        estimated_cost_usd: float,
        workspace_id: str | None = None,
        operation: str | None = None,
    ) -> dict[str, Any]:
        """Pre-flight check if an operation would exceed budget constraints."""
        body: dict[str, Any] = {"estimated_cost_usd": estimated_cost_usd}
        if workspace_id:
            body["workspace_id"] = workspace_id
        if operation:
            body["operation"] = operation
        return await self._client.request("POST", "/api/v1/costs/constraints/check", json=body)

    async def estimate(
        self,
        operation: str,
        tokens_input: int = 0,
        tokens_output: int = 0,
        model: str | None = None,
        provider: str | None = None,
    ) -> dict[str, Any]:
        """Estimate the cost of an operation."""
        body: dict[str, Any] = {
            "operation": operation,
            "tokens_input": tokens_input,
            "tokens_output": tokens_output,
        }
        if model:
            body["model"] = model
        if provider:
            body["provider"] = provider
        return await self._client.request("POST", "/api/v1/costs/estimate", json=body)
