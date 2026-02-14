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
