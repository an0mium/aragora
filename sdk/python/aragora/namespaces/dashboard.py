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

    def execute_quick_action(
        self,
        action_id: str,
        confirm: bool = True,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Execute a quick action.

        Args:
            action_id: Action ID to execute.
            confirm: Confirm execution.
            options: Action-specific options.

        Returns:
            Quick action execution result.
        """
        data: dict[str, Any] = {"confirm": confirm}
        if options:
            data["options"] = options
        return self._client.request(
            "POST", f"/api/v1/dashboard/quick-actions/{action_id}", json=data
        )


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

    async def execute_quick_action(
        self,
        action_id: str,
        confirm: bool = True,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute a quick action."""
        data: dict[str, Any] = {"confirm": confirm}
        if options:
            data["options"] = options
        return await self._client.request(
            "POST", f"/api/v1/dashboard/quick-actions/{action_id}", json=data
        )
