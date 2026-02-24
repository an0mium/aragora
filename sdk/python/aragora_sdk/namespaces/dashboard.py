"""
Dashboard Namespace API.

Provides REST APIs for the main dashboard:
- Overview stats and metrics
- Quick actions
- Recent activity
- Inbox summary
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient

PeriodType = Literal["day", "week", "month"]
PriorityType = Literal["critical", "high", "medium", "low"]
ChangeType = Literal["increase", "decrease", "neutral"]

class DashboardAPI:
    """
    Synchronous Dashboard API.

    Provides methods for dashboard functionality:
    - Overview with key metrics
    - Detailed statistics
    - Recent activity
    - Quick actions

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai")
        >>> # Get dashboard overview
        >>> overview = client.dashboard.get_overview()
        >>> print(f"Unread: {overview['inbox']['total_unread']}")
        >>> # Get detailed stats
        >>> stats = client.dashboard.get_stats(period="week")
        >>> # Execute a quick action
        >>> result = client.dashboard.execute_quick_action("archive_old")
    """

    def __init__(self, client: AragoraClient) -> None:
        self._client = client

    def get_overview(self, refresh: bool = False) -> dict[str, Any]:
        """
        Get dashboard overview.

        Args:
            refresh: Force refresh cache.

        Returns:
            Dashboard overview with inbox, today, team, and AI stats.
        """
        params: dict[str, Any] = {}
        if refresh:
            params["refresh"] = True
        return self._client.request("GET", "/api/v1/dashboard", params=params if params else None)

    def get_overview_page(self, **kwargs: Any) -> dict[str, Any]:
        """
        Get dashboard overview page data.

        Returns:
            Dict with overview metrics and summary.
        """
        return self._client.request("GET", "/api/v1/dashboard/overview", params=kwargs or None)

    def get_stats(self, period: PeriodType = "week") -> dict[str, Any]:
        """
        Get detailed statistics.

        Args:
            period: Time period (day, week, month).

        Returns:
            Dashboard stats with charts and summaries.
        """
        return self._client.request("GET", "/api/v1/dashboard/stats", params={"period": period})

    def get_activity(
        self,
        limit: int | None = None,
        offset: int | None = None,
        activity_type: str | None = None,
    ) -> dict[str, Any]:
        """
        Get recent activity.

        Args:
            limit: Maximum number of results.
            offset: Pagination offset.
            activity_type: Filter by activity type.

        Returns:
            Paginated list of activities.
        """
        params: dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        if offset is not None:
            params["offset"] = offset
        if activity_type:
            params["type"] = activity_type
        return self._client.request(
            "GET", "/api/v1/dashboard/activity", params=params if params else None
        )

    def get_inbox_summary(self) -> dict[str, Any]:
        """
        Get inbox summary for dashboard.

        Returns:
            Inbox counts, priority breakdown, and urgent items.
        """
        return self._client.request("GET", "/api/v1/dashboard/inbox-summary")

    def get_quick_actions(self) -> dict[str, Any]:
        """
        Get available quick actions.

        Returns:
            List of available quick actions with metadata.
        """
        return self._client.request("GET", "/api/v1/dashboard/quick-actions")

    def list_debates(
        self,
        status: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        limit: int | None = None,
        offset: int | None = None,
    ) -> dict[str, Any]:
        """List debates on the dashboard."""
        params: dict[str, Any] = {}
        if status:
            params["status"] = status
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        if limit is not None:
            params["limit"] = limit
        if offset is not None:
            params["offset"] = offset
        return self._client.request(
            "GET", "/api/v1/dashboard/debates", params=params if params else None
        )

    def get_stat_cards(self) -> dict[str, Any]:
        """Get dashboard stat cards."""
        return self._client.request("GET", "/api/v1/dashboard/stat-cards")

    # --- Team Performance ---

    def get_team_performance(
        self,
        sort_by: str | None = None,
        sort_order: str | None = None,
        min_debates: int | None = None,
        limit: int | None = None,
        offset: int | None = None,
    ) -> dict[str, Any]:
        """Get team performance metrics."""
        params: dict[str, Any] = {}
        if sort_by:
            params["sort_by"] = sort_by
        if sort_order:
            params["sort_order"] = sort_order
        if min_debates is not None:
            params["min_debates"] = min_debates
        if limit is not None:
            params["limit"] = limit
        if offset is not None:
            params["offset"] = offset
        return self._client.request(
            "GET", "/api/v1/dashboard/team-performance", params=params if params else None
        )

    def get_top_senders(
        self,
        domain: str | None = None,
        min_messages: int | None = None,
        sort_by: str | None = None,
        limit: int | None = None,
        offset: int | None = None,
    ) -> dict[str, Any]:
        """Get top email senders."""
        params: dict[str, Any] = {}
        if domain:
            params["domain"] = domain
        if min_messages is not None:
            params["min_messages"] = min_messages
        if sort_by:
            params["sort_by"] = sort_by
        if limit is not None:
            params["limit"] = limit
        if offset is not None:
            params["offset"] = offset
        return self._client.request(
            "GET", "/api/v1/dashboard/top-senders", params=params if params else None
        )

    def get_labels(self) -> dict[str, Any]:
        """Get dashboard labels."""
        return self._client.request("GET", "/api/v1/dashboard/labels")

    # --- Urgent Items & Actions ---

    def get_urgent_items(
        self,
        action_type: str | None = None,
        min_importance: int | None = None,
        include_deadline_passed: bool | None = None,
        limit: int | None = None,
        offset: int | None = None,
    ) -> dict[str, Any]:
        """Get urgent items."""
        params: dict[str, Any] = {}
        if action_type:
            params["action_type"] = action_type
        if min_importance is not None:
            params["min_importance"] = min_importance
        if include_deadline_passed is not None:
            params["include_deadline_passed"] = include_deadline_passed
        if limit is not None:
            params["limit"] = limit
        if offset is not None:
            params["offset"] = offset
        return self._client.request(
            "GET", "/api/v1/dashboard/urgent", params=params if params else None
        )

    def get_pending_actions(
        self, limit: int | None = None, offset: int | None = None
    ) -> dict[str, Any]:
        """Get pending actions."""
        params: dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        if offset is not None:
            params["offset"] = offset
        return self._client.request(
            "GET", "/api/v1/dashboard/pending-actions", params=params if params else None
        )

    def search(
        self,
        query: str,
        types: list[str] | None = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """Search the dashboard."""
        params: dict[str, Any] = {"query": query}
        if types:
            params["types"] = ",".join(types)
        if limit is not None:
            params["limit"] = limit
        return self._client.request("GET", "/api/v1/dashboard/search", params=params)

    def export_data(
        self,
        format: str,
        include: list[str] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict[str, Any]:
        """Export dashboard data."""
        data: dict[str, Any] = {"format": format}
        if include:
            data["include"] = include
        if start_date:
            data["start_date"] = start_date
        if end_date:
            data["end_date"] = end_date
        return self._client.request("POST", "/api/v1/dashboard/export", json=data)

    # --- Convenience ---

    def get_recent_activity(self, limit: int = 20) -> dict[str, Any]:
        """Get recent activity (convenience wrapper)."""
        return self.get_activity(limit=limit)

    # --- Gastown Dashboard ---

    def get_gastown_overview(self) -> dict[str, Any]:
        """Get Gastown dashboard overview.

        Returns:
            Dict with Gastown overview metrics.
        """
        return self._client.request("GET", "/api/v1/dashboard/gastown/overview")

    def get_gastown_agents(self) -> dict[str, Any]:
        """Get Gastown agent metrics.

        Returns:
            Dict with agent performance data.
        """
        return self._client.request("GET", "/api/v1/dashboard/gastown/agents")

    def get_gastown_beads(self) -> dict[str, Any]:
        """Get Gastown bead metrics.

        Returns:
            Dict with bead statistics.
        """
        return self._client.request("GET", "/api/v1/dashboard/gastown/beads")

    def get_gastown_convoys(self) -> dict[str, Any]:
        """Get Gastown convoy metrics.

        Returns:
            Dict with convoy statistics.
        """
        return self._client.request("GET", "/api/v1/dashboard/gastown/convoys")

    def get_gastown_metrics(self) -> dict[str, Any]:
        """Get Gastown detailed metrics.

        Returns:
            Dict with detailed Gastown metrics.
        """
        return self._client.request("GET", "/api/v1/dashboard/gastown/metrics")

    # --- Write Operations ---

    def execute_quick_action(
        self, action_id: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Execute a quick action.

        Args:
            action_id: The action to execute (e.g. 'archive_read', 'snooze_low').
            params: Optional action-specific parameters.

        Returns:
            Dict with execution result including affected count and message.
        """
        return self._client.request(
            "POST", f"/api/v1/dashboard/quick-actions/{action_id}", json=params
        )

    def dismiss_urgent_item(self, item_id: str) -> dict[str, Any]:
        """Dismiss an urgent item.

        Args:
            item_id: ID of the urgent item to dismiss.

        Returns:
            Dict with success status.
        """
        return self._client.request("POST", f"/api/v1/dashboard/urgent/{item_id}/dismiss")

    def complete_action(
        self, action_id: str, result: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Complete a pending action.

        Args:
            action_id: ID of the action to complete.
            result: Optional result data for the completed action.

        Returns:
            Dict with success status.
        """
        return self._client.request(
            "POST", f"/api/v1/dashboard/pending-actions/{action_id}/complete", json=result
        )

    def get_debate(self, debate_id: str) -> dict[str, Any]:
        """Get a specific debate summary from the dashboard.

        Args:
            debate_id: The debate ID.

        Returns:
            Dict with debate summary data.
        """
        return self._client.request("GET", f"/api/v1/dashboard/debates/{debate_id}")

    def get_team_by_id(self, team_id: str) -> dict[str, Any]:
        """Get performance metrics for a specific team.

        Args:
            team_id: The team ID.

        Returns:
            Dict with team performance data.
        """
        return self._client.request("GET", f"/api/v1/dashboard/team-performance/{team_id}")

    # --- Outcome Dashboard ---

    def get_outcome_dashboard(self, period: str = "30d") -> dict[str, Any]:
        """Get full outcome dashboard data.

        Combines decision quality scores, agent performance, consensus metrics,
        and calibration curve data into a single payload.

        Args:
            period: Time period (e.g. '7d', '30d', '90d').

        Returns:
            Dict with quality, agents, history, and calibration data.
        """
        return self._client.request(
            "GET", "/api/v1/outcome-dashboard", params={"period": period}
        )

    def get_outcome_quality(self, period: str = "30d") -> dict[str, Any]:
        """Get decision quality score and trend.

        Args:
            period: Time period (e.g. '7d', '30d', '90d').

        Returns:
            Dict with quality score, consensus rate, and trend data.
        """
        return self._client.request(
            "GET", "/api/v1/outcome-dashboard/quality", params={"period": period}
        )

    def get_outcome_agents(self, period: str = "30d") -> dict[str, Any]:
        """Get agent leaderboard with ELO and calibration scores.

        Args:
            period: Time period (e.g. '7d', '30d', '90d').

        Returns:
            Dict with agent performance rankings.
        """
        return self._client.request(
            "GET", "/api/v1/outcome-dashboard/agents", params={"period": period}
        )

    def get_outcome_history(
        self,
        period: str = "30d",
        limit: int | None = None,
        offset: int | None = None,
    ) -> dict[str, Any]:
        """Get paginated decision history with quality scores.

        Args:
            period: Time period (e.g. '7d', '30d', '90d').
            limit: Maximum number of results.
            offset: Pagination offset.

        Returns:
            Dict with decisions list and pagination info.
        """
        params: dict[str, Any] = {"period": period}
        if limit is not None:
            params["limit"] = limit
        if offset is not None:
            params["offset"] = offset
        return self._client.request(
            "GET", "/api/v1/outcome-dashboard/history", params=params
        )

    def get_outcome_calibration(self, period: str = "30d") -> dict[str, Any]:
        """Get calibration curve data (predicted vs actual confidence).

        Args:
            period: Time period (e.g. '7d', '30d', '90d').

        Returns:
            Dict with calibration points and total observations.
        """
        return self._client.request(
            "GET", "/api/v1/outcome-dashboard/calibration", params={"period": period}
        )

    # --- Usage Dashboard ---

    def get_usage_summary(self, period: str = "30d") -> dict[str, Any]:
        """Get unified usage metrics summary.

        Args:
            period: Time period (e.g. '7d', '30d', '90d').

        Returns:
            Dict with usage metrics including debates, costs, and consensus rate.
        """
        return self._client.request(
            "GET", "/api/v1/usage/summary", params={"period": period}
        )

    def get_usage_breakdown(
        self, dimension: str = "agent", period: str = "30d"
    ) -> dict[str, Any]:
        """Get detailed usage breakdown by dimension.

        Args:
            dimension: Breakdown dimension (e.g. 'agent', 'model', 'team').
            period: Time period.

        Returns:
            Dict with breakdown data by the specified dimension.
        """
        return self._client.request(
            "GET", "/api/v1/usage/breakdown",
            params={"dimension": dimension, "period": period},
        )

    def get_usage_roi(self, period: str = "30d") -> dict[str, Any]:
        """Get ROI analysis for usage.

        Args:
            period: Time period.

        Returns:
            Dict with ROI metrics, time savings, and cost per decision.
        """
        return self._client.request(
            "GET", "/api/v1/usage/roi", params={"period": period}
        )

    def export_usage(
        self, format: str = "json", period: str = "30d"
    ) -> dict[str, Any]:
        """Export usage data.

        Args:
            format: Export format ('json', 'csv', 'pdf').
            period: Time period.

        Returns:
            Dict with exported data or download URL.
        """
        return self._client.request(
            "GET", "/api/v1/usage/export",
            params={"format": format, "period": period},
        )

    def get_budget_status(self) -> dict[str, Any]:
        """Get budget utilization status.

        Returns:
            Dict with budget limits, spent amount, remaining, and forecast.
        """
        return self._client.request("GET", "/api/v1/usage/budget-status")

    # --- Spend Analytics Dashboard ---

    def get_spend_summary(self) -> dict[str, Any]:
        """Get spend analytics summary.

        Returns:
            Dict with total spend, budget utilization, and trend direction.
        """
        return self._client.request("GET", "/api/v1/analytics/spend/summary")

    def get_spend_trends(
        self, period: str = "30d", granularity: str = "daily"
    ) -> dict[str, Any]:
        """Get spend trends over time.

        Args:
            period: Time period (e.g. '7d', '30d', '90d').
            granularity: Data granularity ('daily', 'weekly', 'monthly').

        Returns:
            Dict with spend data points over time.
        """
        return self._client.request(
            "GET", "/api/v1/analytics/spend/trends",
            params={"period": period, "granularity": granularity},
        )

    def get_spend_by_agent(self) -> dict[str, Any]:
        """Get cost breakdown per agent type.

        Returns:
            Dict with per-agent cost data.
        """
        return self._client.request("GET", "/api/v1/analytics/spend/by-agent")

    def get_spend_by_decision(self) -> dict[str, Any]:
        """Get cost per debate/decision.

        Returns:
            Dict with per-decision cost data.
        """
        return self._client.request("GET", "/api/v1/analytics/spend/by-decision")

    def get_spend_budget(self) -> dict[str, Any]:
        """Get budget limits, remaining, and forecast to exhaustion.

        Returns:
            Dict with budget data and exhaustion forecast.
        """
        return self._client.request("GET", "/api/v1/analytics/spend/budget")


class AsyncDashboardAPI:
    """Asynchronous Dashboard API."""

    def __init__(self, client: AragoraAsyncClient) -> None:
        self._client = client

    async def get_overview(self, refresh: bool = False) -> dict[str, Any]:
        """Get dashboard overview."""
        params: dict[str, Any] = {}
        if refresh:
            params["refresh"] = True
        return await self._client.request(
            "GET", "/api/v1/dashboard", params=params if params else None
        )

    async def get_overview_page(self, **kwargs: Any) -> dict[str, Any]:
        """Get dashboard overview page data."""
        return await self._client.request("GET", "/api/v1/dashboard/overview", params=kwargs or None)

    async def get_stats(self, period: PeriodType = "week") -> dict[str, Any]:
        """Get detailed statistics."""
        return await self._client.request(
            "GET", "/api/v1/dashboard/stats", params={"period": period}
        )

    async def get_activity(
        self,
        limit: int | None = None,
        offset: int | None = None,
        activity_type: str | None = None,
    ) -> dict[str, Any]:
        """Get recent activity."""
        params: dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        if offset is not None:
            params["offset"] = offset
        if activity_type:
            params["type"] = activity_type
        return await self._client.request(
            "GET", "/api/v1/dashboard/activity", params=params if params else None
        )

    async def get_inbox_summary(self) -> dict[str, Any]:
        """Get inbox summary for dashboard."""
        return await self._client.request("GET", "/api/v1/dashboard/inbox-summary")

    async def get_quick_actions(self) -> dict[str, Any]:
        """Get available quick actions."""
        return await self._client.request("GET", "/api/v1/dashboard/quick-actions")

    async def list_debates(
        self,
        status: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        limit: int | None = None,
        offset: int | None = None,
    ) -> dict[str, Any]:
        """List debates on the dashboard."""
        params: dict[str, Any] = {}
        if status:
            params["status"] = status
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        if limit is not None:
            params["limit"] = limit
        if offset is not None:
            params["offset"] = offset
        return await self._client.request(
            "GET", "/api/v1/dashboard/debates", params=params if params else None
        )

    async def get_stat_cards(self) -> dict[str, Any]:
        """Get dashboard stat cards."""
        return await self._client.request("GET", "/api/v1/dashboard/stat-cards")

    async def get_team_performance(
        self,
        sort_by: str | None = None,
        sort_order: str | None = None,
        min_debates: int | None = None,
        limit: int | None = None,
        offset: int | None = None,
    ) -> dict[str, Any]:
        """Get team performance metrics."""
        params: dict[str, Any] = {}
        if sort_by:
            params["sort_by"] = sort_by
        if sort_order:
            params["sort_order"] = sort_order
        if min_debates is not None:
            params["min_debates"] = min_debates
        if limit is not None:
            params["limit"] = limit
        if offset is not None:
            params["offset"] = offset
        return await self._client.request(
            "GET", "/api/v1/dashboard/team-performance", params=params if params else None
        )

    async def get_top_senders(
        self,
        domain: str | None = None,
        min_messages: int | None = None,
        sort_by: str | None = None,
        limit: int | None = None,
        offset: int | None = None,
    ) -> dict[str, Any]:
        """Get top email senders."""
        params: dict[str, Any] = {}
        if domain:
            params["domain"] = domain
        if min_messages is not None:
            params["min_messages"] = min_messages
        if sort_by:
            params["sort_by"] = sort_by
        if limit is not None:
            params["limit"] = limit
        if offset is not None:
            params["offset"] = offset
        return await self._client.request(
            "GET", "/api/v1/dashboard/top-senders", params=params if params else None
        )

    async def get_labels(self) -> dict[str, Any]:
        """Get dashboard labels."""
        return await self._client.request("GET", "/api/v1/dashboard/labels")

    async def get_urgent_items(
        self,
        action_type: str | None = None,
        min_importance: int | None = None,
        include_deadline_passed: bool | None = None,
        limit: int | None = None,
        offset: int | None = None,
    ) -> dict[str, Any]:
        """Get urgent items."""
        params: dict[str, Any] = {}
        if action_type:
            params["action_type"] = action_type
        if min_importance is not None:
            params["min_importance"] = min_importance
        if include_deadline_passed is not None:
            params["include_deadline_passed"] = include_deadline_passed
        if limit is not None:
            params["limit"] = limit
        if offset is not None:
            params["offset"] = offset
        return await self._client.request(
            "GET", "/api/v1/dashboard/urgent", params=params if params else None
        )

    async def get_pending_actions(
        self, limit: int | None = None, offset: int | None = None
    ) -> dict[str, Any]:
        """Get pending actions."""
        params: dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        if offset is not None:
            params["offset"] = offset
        return await self._client.request(
            "GET", "/api/v1/dashboard/pending-actions", params=params if params else None
        )

    async def search(
        self,
        query: str,
        types: list[str] | None = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """Search the dashboard."""
        params: dict[str, Any] = {"query": query}
        if types:
            params["types"] = ",".join(types)
        if limit is not None:
            params["limit"] = limit
        return await self._client.request("GET", "/api/v1/dashboard/search", params=params)

    async def export_data(
        self,
        format: str,
        include: list[str] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict[str, Any]:
        """Export dashboard data."""
        data: dict[str, Any] = {"format": format}
        if include:
            data["include"] = include
        if start_date:
            data["start_date"] = start_date
        if end_date:
            data["end_date"] = end_date
        return await self._client.request("POST", "/api/v1/dashboard/export", json=data)

    async def get_recent_activity(self, limit: int = 20) -> dict[str, Any]:
        """Get recent activity (convenience wrapper)."""
        return await self.get_activity(limit=limit)

    # --- Gastown Dashboard ---

    async def get_gastown_overview(self) -> dict[str, Any]:
        """Get Gastown dashboard overview."""
        return await self._client.request("GET", "/api/v1/dashboard/gastown/overview")

    async def get_gastown_agents(self) -> dict[str, Any]:
        """Get Gastown agent metrics."""
        return await self._client.request("GET", "/api/v1/dashboard/gastown/agents")

    async def get_gastown_beads(self) -> dict[str, Any]:
        """Get Gastown bead metrics."""
        return await self._client.request("GET", "/api/v1/dashboard/gastown/beads")

    async def get_gastown_convoys(self) -> dict[str, Any]:
        """Get Gastown convoy metrics."""
        return await self._client.request("GET", "/api/v1/dashboard/gastown/convoys")

    async def get_gastown_metrics(self) -> dict[str, Any]:
        """Get Gastown detailed metrics."""
        return await self._client.request("GET", "/api/v1/dashboard/gastown/metrics")

    # --- Write Operations ---

    async def execute_quick_action(
        self, action_id: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Execute a quick action."""
        return await self._client.request(
            "POST", f"/api/v1/dashboard/quick-actions/{action_id}", json=params
        )

    async def dismiss_urgent_item(self, item_id: str) -> dict[str, Any]:
        """Dismiss an urgent item."""
        return await self._client.request("POST", f"/api/v1/dashboard/urgent/{item_id}/dismiss")

    async def complete_action(
        self, action_id: str, result: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Complete a pending action."""
        return await self._client.request(
            "POST", f"/api/v1/dashboard/pending-actions/{action_id}/complete", json=result
        )

    async def get_debate(self, debate_id: str) -> dict[str, Any]:
        """Get a specific debate summary from the dashboard."""
        return await self._client.request("GET", f"/api/v1/dashboard/debates/{debate_id}")

    async def get_team_by_id(self, team_id: str) -> dict[str, Any]:
        """Get performance metrics for a specific team."""
        return await self._client.request("GET", f"/api/v1/dashboard/team-performance/{team_id}")

    # --- Outcome Dashboard ---

    async def get_outcome_dashboard(self, period: str = "30d") -> dict[str, Any]:
        """Get full outcome dashboard data."""
        return await self._client.request(
            "GET", "/api/v1/outcome-dashboard", params={"period": period}
        )

    async def get_outcome_quality(self, period: str = "30d") -> dict[str, Any]:
        """Get decision quality score and trend."""
        return await self._client.request(
            "GET", "/api/v1/outcome-dashboard/quality", params={"period": period}
        )

    async def get_outcome_agents(self, period: str = "30d") -> dict[str, Any]:
        """Get agent leaderboard with ELO and calibration scores."""
        return await self._client.request(
            "GET", "/api/v1/outcome-dashboard/agents", params={"period": period}
        )

    async def get_outcome_history(
        self,
        period: str = "30d",
        limit: int | None = None,
        offset: int | None = None,
    ) -> dict[str, Any]:
        """Get paginated decision history with quality scores."""
        params: dict[str, Any] = {"period": period}
        if limit is not None:
            params["limit"] = limit
        if offset is not None:
            params["offset"] = offset
        return await self._client.request(
            "GET", "/api/v1/outcome-dashboard/history", params=params
        )

    async def get_outcome_calibration(self, period: str = "30d") -> dict[str, Any]:
        """Get calibration curve data (predicted vs actual confidence)."""
        return await self._client.request(
            "GET", "/api/v1/outcome-dashboard/calibration", params={"period": period}
        )

    # --- Usage Dashboard ---

    async def get_usage_summary(self, period: str = "30d") -> dict[str, Any]:
        """Get unified usage metrics summary."""
        return await self._client.request(
            "GET", "/api/v1/usage/summary", params={"period": period}
        )

    async def get_usage_breakdown(
        self, dimension: str = "agent", period: str = "30d"
    ) -> dict[str, Any]:
        """Get detailed usage breakdown by dimension."""
        return await self._client.request(
            "GET", "/api/v1/usage/breakdown",
            params={"dimension": dimension, "period": period},
        )

    async def get_usage_roi(self, period: str = "30d") -> dict[str, Any]:
        """Get ROI analysis for usage."""
        return await self._client.request(
            "GET", "/api/v1/usage/roi", params={"period": period}
        )

    async def export_usage(
        self, format: str = "json", period: str = "30d"
    ) -> dict[str, Any]:
        """Export usage data."""
        return await self._client.request(
            "GET", "/api/v1/usage/export",
            params={"format": format, "period": period},
        )

    async def get_budget_status(self) -> dict[str, Any]:
        """Get budget utilization status."""
        return await self._client.request("GET", "/api/v1/usage/budget-status")

    # --- Spend Analytics Dashboard ---

    async def get_spend_summary(self) -> dict[str, Any]:
        """Get spend analytics summary."""
        return await self._client.request("GET", "/api/v1/analytics/spend/summary")

    async def get_spend_trends(
        self, period: str = "30d", granularity: str = "daily"
    ) -> dict[str, Any]:
        """Get spend trends over time."""
        return await self._client.request(
            "GET", "/api/v1/analytics/spend/trends",
            params={"period": period, "granularity": granularity},
        )

    async def get_spend_by_agent(self) -> dict[str, Any]:
        """Get cost breakdown per agent type."""
        return await self._client.request("GET", "/api/v1/analytics/spend/by-agent")

    async def get_spend_by_decision(self) -> dict[str, Any]:
        """Get cost per debate/decision."""
        return await self._client.request("GET", "/api/v1/analytics/spend/by-decision")

    async def get_spend_budget(self) -> dict[str, Any]:
        """Get budget limits, remaining, and forecast to exhaustion."""
        return await self._client.request("GET", "/api/v1/analytics/spend/budget")
