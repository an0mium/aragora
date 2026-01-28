"""
Budgets namespace for cost control and spending management.

Provides API access to manage budgets, alerts, and spending limits.
Critical for SME cost control and preventing unexpected charges.
"""

from __future__ import annotations

from typing import Any, Literal


class BudgetsAPI:
    """Synchronous budgets API."""

    def __init__(self, client: Any) -> None:
        self._client = client

    # =========================================================================
    # Budget CRUD
    # =========================================================================

    def list(
        self,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """
        List all budgets for the organization.

        Args:
            limit: Maximum number of budgets to return
            offset: Number of budgets to skip

        Returns:
            List of budgets with pagination
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        return self._client._request("GET", "/api/v1/budgets", params=params)

    def create(
        self,
        name: str,
        limit_amount: float,
        period: Literal["daily", "weekly", "monthly", "quarterly", "yearly"],
        alert_threshold: int = 80,
        scope: str | None = None,
    ) -> dict[str, Any]:
        """
        Create a new budget.

        Args:
            name: Budget name
            limit_amount: Spending limit in USD
            period: Budget period
            alert_threshold: Percentage threshold for alerts (0-100)
            scope: Optional scope (workspace_id, team_id, etc.)

        Returns:
            Created budget record
        """
        data: dict[str, Any] = {
            "name": name,
            "limit_amount": limit_amount,
            "period": period,
            "alert_threshold": alert_threshold,
        }
        if scope:
            data["scope"] = scope

        return self._client._request("POST", "/api/v1/budgets", json=data)

    def get(self, budget_id: str) -> dict[str, Any]:
        """
        Get a budget by ID.

        Args:
            budget_id: Budget identifier

        Returns:
            Budget details
        """
        return self._client._request("GET", f"/api/v1/budgets/{budget_id}")

    def update(
        self,
        budget_id: str,
        name: str | None = None,
        limit_amount: float | None = None,
        alert_threshold: int | None = None,
        enabled: bool | None = None,
    ) -> dict[str, Any]:
        """
        Update a budget.

        Args:
            budget_id: Budget identifier
            name: New budget name
            limit_amount: New spending limit
            alert_threshold: New alert threshold
            enabled: Enable/disable the budget

        Returns:
            Updated budget record
        """
        data: dict[str, Any] = {}
        if name is not None:
            data["name"] = name
        if limit_amount is not None:
            data["limit_amount"] = limit_amount
        if alert_threshold is not None:
            data["alert_threshold"] = alert_threshold
        if enabled is not None:
            data["enabled"] = enabled

        return self._client._request("PATCH", f"/api/v1/budgets/{budget_id}", json=data)

    def delete(self, budget_id: str) -> dict[str, Any]:
        """
        Delete a budget.

        Args:
            budget_id: Budget identifier

        Returns:
            Deletion confirmation
        """
        return self._client._request("DELETE", f"/api/v1/budgets/{budget_id}")

    # =========================================================================
    # Budget Checks
    # =========================================================================

    def check(
        self,
        operation: str,
        estimated_cost: float,
        budget_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Check if an operation is allowed within budget.

        Call this before expensive operations to ensure budget compliance.

        Args:
            operation: Operation type (debate, gauntlet, etc.)
            estimated_cost: Estimated cost in USD
            budget_id: Optional specific budget to check

        Returns:
            Check result with allowed status and remaining budget
        """
        data: dict[str, Any] = {
            "operation": operation,
            "estimated_cost": estimated_cost,
        }
        if budget_id:
            data["budget_id"] = budget_id

        return self._client._request("POST", "/api/v1/budgets/check", json=data)

    def get_summary(self) -> dict[str, Any]:
        """
        Get organization-wide budget summary.

        Returns:
            Summary including total budgets, exceeded count, etc.
        """
        return self._client._request("GET", "/api/v1/budgets/summary")

    # =========================================================================
    # Alerts
    # =========================================================================

    def get_alerts(
        self,
        budget_id: str,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """
        Get alerts for a budget.

        Args:
            budget_id: Budget identifier
            limit: Maximum alerts to return
            offset: Number of alerts to skip

        Returns:
            List of alerts
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        return self._client._request("GET", f"/api/v1/budgets/{budget_id}/alerts", params=params)

    def acknowledge_alert(self, budget_id: str, alert_id: str) -> dict[str, Any]:
        """
        Acknowledge a budget alert.

        Args:
            budget_id: Budget identifier
            alert_id: Alert identifier

        Returns:
            Acknowledgment confirmation
        """
        return self._client._request(
            "POST", f"/api/v1/budgets/{budget_id}/alerts/{alert_id}/acknowledge"
        )

    # =========================================================================
    # Overrides
    # =========================================================================

    def add_override(
        self,
        budget_id: str,
        user_id: str,
        limit: float,
        reason: str | None = None,
    ) -> dict[str, Any]:
        """
        Add a user-specific budget override.

        Useful for giving specific users higher limits for their work.

        Args:
            budget_id: Budget identifier
            user_id: User to override
            limit: Override limit in USD
            reason: Reason for the override

        Returns:
            Override confirmation
        """
        data: dict[str, Any] = {"user_id": user_id, "limit": limit}
        if reason:
            data["reason"] = reason

        return self._client._request("POST", f"/api/v1/budgets/{budget_id}/overrides", json=data)

    def remove_override(self, budget_id: str, user_id: str) -> dict[str, Any]:
        """
        Remove a user-specific budget override.

        Args:
            budget_id: Budget identifier
            user_id: User to remove override for

        Returns:
            Removal confirmation
        """
        return self._client._request("DELETE", f"/api/v1/budgets/{budget_id}/overrides/{user_id}")

    # =========================================================================
    # Period Management
    # =========================================================================

    def reset(self, budget_id: str) -> dict[str, Any]:
        """
        Reset a budget period.

        Useful for manual resets or when changing budget parameters.

        Args:
            budget_id: Budget identifier

        Returns:
            Reset confirmation with new period start
        """
        return self._client._request("POST", f"/api/v1/budgets/{budget_id}/reset")

    # =========================================================================
    # Transaction History
    # =========================================================================

    def get_transactions(
        self,
        budget_id: str,
        date_from: int | None = None,
        date_to: int | None = None,
        user_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """
        Get transaction history for a budget.

        Args:
            budget_id: Budget identifier
            date_from: Start timestamp (Unix seconds)
            date_to: End timestamp (Unix seconds)
            user_id: Filter by user
            limit: Maximum transactions to return
            offset: Number of transactions to skip

        Returns:
            List of transactions
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if date_from is not None:
            params["date_from"] = date_from
        if date_to is not None:
            params["date_to"] = date_to
        if user_id:
            params["user_id"] = user_id

        return self._client._request(
            "GET", f"/api/v1/budgets/{budget_id}/transactions", params=params
        )

    # =========================================================================
    # Spending Trends
    # =========================================================================

    def get_trends(
        self,
        budget_id: str,
        period: Literal["hour", "day", "week", "month"] = "day",
        limit: int = 30,
    ) -> dict[str, Any]:
        """
        Get spending trends for a budget.

        Args:
            budget_id: Budget identifier
            period: Aggregation period
            limit: Number of periods to return

        Returns:
            Trend data for charts and analysis
        """
        params: dict[str, Any] = {"period": period, "limit": limit}
        return self._client._request("GET", f"/api/v1/budgets/{budget_id}/trends", params=params)

    def get_org_trends(
        self,
        period: Literal["hour", "day", "week", "month"] = "day",
        limit: int = 30,
    ) -> dict[str, Any]:
        """
        Get organization-wide spending trends across all budgets.

        Useful for executive dashboards and org-level reporting.

        Args:
            period: Aggregation period
            limit: Number of periods to return

        Returns:
            Organization-wide trend data
        """
        params: dict[str, Any] = {"period": period, "limit": limit}
        return self._client._request("GET", "/api/v1/budgets/trends", params=params)


class AsyncBudgetsAPI:
    """Asynchronous budgets API."""

    def __init__(self, client: Any) -> None:
        self._client = client

    # =========================================================================
    # Budget CRUD
    # =========================================================================

    async def list(
        self,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """List all budgets for the organization."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        return await self._client._request("GET", "/api/v1/budgets", params=params)

    async def create(
        self,
        name: str,
        limit_amount: float,
        period: Literal["daily", "weekly", "monthly", "quarterly", "yearly"],
        alert_threshold: int = 80,
        scope: str | None = None,
    ) -> dict[str, Any]:
        """Create a new budget."""
        data: dict[str, Any] = {
            "name": name,
            "limit_amount": limit_amount,
            "period": period,
            "alert_threshold": alert_threshold,
        }
        if scope:
            data["scope"] = scope

        return await self._client._request("POST", "/api/v1/budgets", json=data)

    async def get(self, budget_id: str) -> dict[str, Any]:
        """Get a budget by ID."""
        return await self._client._request("GET", f"/api/v1/budgets/{budget_id}")

    async def update(
        self,
        budget_id: str,
        name: str | None = None,
        limit_amount: float | None = None,
        alert_threshold: int | None = None,
        enabled: bool | None = None,
    ) -> dict[str, Any]:
        """Update a budget."""
        data: dict[str, Any] = {}
        if name is not None:
            data["name"] = name
        if limit_amount is not None:
            data["limit_amount"] = limit_amount
        if alert_threshold is not None:
            data["alert_threshold"] = alert_threshold
        if enabled is not None:
            data["enabled"] = enabled

        return await self._client._request("PATCH", f"/api/v1/budgets/{budget_id}", json=data)

    async def delete(self, budget_id: str) -> dict[str, Any]:
        """Delete a budget."""
        return await self._client._request("DELETE", f"/api/v1/budgets/{budget_id}")

    # =========================================================================
    # Budget Checks
    # =========================================================================

    async def check(
        self,
        operation: str,
        estimated_cost: float,
        budget_id: str | None = None,
    ) -> dict[str, Any]:
        """Check if an operation is allowed within budget."""
        data: dict[str, Any] = {
            "operation": operation,
            "estimated_cost": estimated_cost,
        }
        if budget_id:
            data["budget_id"] = budget_id

        return await self._client._request("POST", "/api/v1/budgets/check", json=data)

    async def get_summary(self) -> dict[str, Any]:
        """Get organization-wide budget summary."""
        return await self._client._request("GET", "/api/v1/budgets/summary")

    # =========================================================================
    # Alerts
    # =========================================================================

    async def get_alerts(
        self,
        budget_id: str,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Get alerts for a budget."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        return await self._client._request(
            "GET", f"/api/v1/budgets/{budget_id}/alerts", params=params
        )

    async def acknowledge_alert(self, budget_id: str, alert_id: str) -> dict[str, Any]:
        """Acknowledge a budget alert."""
        return await self._client._request(
            "POST", f"/api/v1/budgets/{budget_id}/alerts/{alert_id}/acknowledge"
        )

    # =========================================================================
    # Overrides
    # =========================================================================

    async def add_override(
        self,
        budget_id: str,
        user_id: str,
        limit: float,
        reason: str | None = None,
    ) -> dict[str, Any]:
        """Add a user-specific budget override."""
        data: dict[str, Any] = {"user_id": user_id, "limit": limit}
        if reason:
            data["reason"] = reason

        return await self._client._request(
            "POST", f"/api/v1/budgets/{budget_id}/overrides", json=data
        )

    async def remove_override(self, budget_id: str, user_id: str) -> dict[str, Any]:
        """Remove a user-specific budget override."""
        return await self._client._request(
            "DELETE", f"/api/v1/budgets/{budget_id}/overrides/{user_id}"
        )

    # =========================================================================
    # Period Management
    # =========================================================================

    async def reset(self, budget_id: str) -> dict[str, Any]:
        """Reset a budget period."""
        return await self._client._request("POST", f"/api/v1/budgets/{budget_id}/reset")

    # =========================================================================
    # Transaction History
    # =========================================================================

    async def get_transactions(
        self,
        budget_id: str,
        date_from: int | None = None,
        date_to: int | None = None,
        user_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Get transaction history for a budget."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if date_from is not None:
            params["date_from"] = date_from
        if date_to is not None:
            params["date_to"] = date_to
        if user_id:
            params["user_id"] = user_id

        return await self._client._request(
            "GET", f"/api/v1/budgets/{budget_id}/transactions", params=params
        )

    # =========================================================================
    # Spending Trends
    # =========================================================================

    async def get_trends(
        self,
        budget_id: str,
        period: Literal["hour", "day", "week", "month"] = "day",
        limit: int = 30,
    ) -> dict[str, Any]:
        """Get spending trends for a budget."""
        params: dict[str, Any] = {"period": period, "limit": limit}
        return await self._client._request(
            "GET", f"/api/v1/budgets/{budget_id}/trends", params=params
        )

    async def get_org_trends(
        self,
        period: Literal["hour", "day", "week", "month"] = "day",
        limit: int = 30,
    ) -> dict[str, Any]:
        """Get organization-wide spending trends across all budgets."""
        params: dict[str, Any] = {"period": period, "limit": limit}
        return await self._client._request("GET", "/api/v1/budgets/trends", params=params)
