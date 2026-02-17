"""
SME Namespace API

Provides methods for SME (Small/Medium Enterprise) features including
pre-built workflow templates, onboarding, and quick-start helpers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient

ReportType = Literal["sales", "inventory", "customers", "financial"]
ReportPeriod = Literal["daily", "weekly", "monthly", "quarterly"]
ReportFormat = Literal["pdf", "excel", "html", "json"]
FollowupType = Literal["post_sale", "check_in", "renewal", "feedback"]


class SMEAPI:
    """
    Synchronous SME API.

    Provides methods for SME (Small/Medium Enterprise) features:
    - Budget management
    - Slack integration (workspaces, subscriptions)
    - Pre-built SME workflow templates

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai")
        >>> budgets = client.sme.list_budgets()
        >>> workspaces = client.sme.list_slack_workspaces()
    """

    def __init__(self, client: AragoraClient):
        self._client = client

    # ===========================================================================
    # Budgets
    # ===========================================================================

    def list_budgets(self) -> dict[str, Any]:
        """List all budgets."""
        return self._client.request("GET", "/api/v1/sme/budgets")

    def create_budget(self, **kwargs: Any) -> dict[str, Any]:
        """Create a new budget."""
        return self._client.request("POST", "/api/v1/sme/budgets", json=kwargs)

    def get_budget(self, budget_id: str) -> dict[str, Any]:
        """Get a budget by ID."""
        return self._client.request("GET", f"/api/v1/sme/budgets/{budget_id}")

    def update_budget(self, budget_id: str, **kwargs: Any) -> dict[str, Any]:
        """Update a budget."""
        return self._client.request("PATCH", f"/api/v1/sme/budgets/{budget_id}", json=kwargs)

    def delete_budget(self, budget_id: str) -> dict[str, Any]:
        """Delete a budget."""
        return self._client.request("DELETE", f"/api/v1/sme/budgets/{budget_id}")

    def get_budget_alerts(self, budget_id: str) -> dict[str, Any]:
        """Get alerts for a budget."""
        return self._client.request("GET", f"/api/v1/sme/budgets/{budget_id}/alerts")

    def get_budget_transactions(self, budget_id: str) -> dict[str, Any]:
        """Get transactions for a budget."""
        return self._client.request("GET", f"/api/v1/sme/budgets/{budget_id}/transactions")

    # ===========================================================================
    # Slack Integration
    # ===========================================================================

    def get_slack_oauth_callback(self) -> dict[str, Any]:
        """Handle Slack OAuth callback."""
        return self._client.request("GET", "/api/v1/sme/slack/oauth/callback")

    def get_slack_oauth_start(self) -> dict[str, Any]:
        """Start Slack OAuth flow."""
        return self._client.request("GET", "/api/v1/sme/slack/oauth/start")

    def slack_subscribe(self, **kwargs: Any) -> dict[str, Any]:
        """Create a Slack subscription."""
        return self._client.request("POST", "/api/v1/sme/slack/subscribe", json=kwargs)

    def list_slack_subscriptions(self) -> dict[str, Any]:
        """List Slack subscriptions."""
        return self._client.request("GET", "/api/v1/sme/slack/subscriptions")

    def delete_slack_subscription(self, subscription_id: str) -> dict[str, Any]:
        """Delete a Slack subscription."""
        return self._client.request("DELETE", f"/api/v1/sme/slack/subscriptions/{subscription_id}")

    def list_slack_workspaces(self) -> dict[str, Any]:
        """List Slack workspaces."""
        return self._client.request("GET", "/api/v1/sme/slack/workspaces")

    def create_slack_workspace(self, **kwargs: Any) -> dict[str, Any]:
        """Create a Slack workspace integration."""
        return self._client.request("POST", "/api/v1/sme/slack/workspaces", json=kwargs)

    def get_slack_workspace(self, workspace_id: str) -> dict[str, Any]:
        """Get a Slack workspace by ID."""
        return self._client.request("GET", f"/api/v1/sme/slack/workspaces/{workspace_id}")

    def update_slack_workspace(self, workspace_id: str, **kwargs: Any) -> dict[str, Any]:
        """Update a Slack workspace."""
        return self._client.request("PATCH", f"/api/v1/sme/slack/workspaces/{workspace_id}", json=kwargs)

    def delete_slack_workspace(self, workspace_id: str) -> dict[str, Any]:
        """Delete a Slack workspace integration."""
        return self._client.request("DELETE", f"/api/v1/sme/slack/workspaces/{workspace_id}")

    def list_slack_channels(self, workspace_id: str) -> dict[str, Any]:
        """List channels in a Slack workspace."""
        return self._client.request("GET", f"/api/v1/sme/slack/workspaces/{workspace_id}/channels")

    def test_slack_workspace(self, workspace_id: str) -> dict[str, Any]:
        """Test a Slack workspace connection."""
        return self._client.request("POST", f"/api/v1/sme/slack/workspaces/{workspace_id}/test")

    # ===========================================================================
    # Quick Helpers
    # ===========================================================================

    def quick_invoice(
        self,
        customer_email: str,
        customer_name: str,
        items: list[dict[str, Any]],
        due_date: str | None = None,
    ) -> dict[str, Any]:
        """Quick invoice generation helper."""
        formatted_items = [
            {
                "name": item["name"],
                "unit_price": item["price"],
                "quantity": item.get("quantity", 1),
            }
            for item in items
        ]
        inputs: dict[str, Any] = {
            "customer_email": customer_email,
            "customer_name": customer_name,
            "items": formatted_items,
        }
        if due_date:
            inputs["due_date"] = due_date
        return self.execute_workflow("invoice", inputs=inputs)

    def quick_inventory_check(
        self,
        product_id: str,
        min_threshold: int,
        notification_email: str,
    ) -> dict[str, Any]:
        """Quick inventory check helper."""
        return self.execute_workflow(
            "inventory",
            inputs={
                "product_id": product_id,
                "min_threshold": min_threshold,
                "notification_email": notification_email,
            },
        )

    def quick_report(
        self,
        report_type: ReportType,
        period: ReportPeriod,
        format: ReportFormat = "pdf",
        email: str | None = None,
    ) -> dict[str, Any]:
        """Quick report generation helper."""
        inputs: dict[str, Any] = {
            "report_type": report_type,
            "period": period,
            "format": format,
        }
        if email:
            inputs["delivery_email"] = email
        return self.execute_workflow("report", inputs=inputs)

    def quick_followup(
        self,
        customer_id: str,
        followup_type: FollowupType,
        message: str | None = None,
        delay_days: int = 0,
    ) -> dict[str, Any]:
        """Quick customer follow-up helper."""
        inputs: dict[str, Any] = {
            "customer_id": customer_id,
            "followup_type": followup_type,
            "delay_days": delay_days,
        }
        if message:
            inputs["custom_message"] = message
        return self.execute_workflow("followup", inputs=inputs)


class AsyncSMEAPI:
    """
    Asynchronous SME API.

    Example:
        >>> async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
        ...     budgets = await client.sme.list_budgets()
    """

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    # ===========================================================================
    # Budgets
    # ===========================================================================

    async def list_budgets(self) -> dict[str, Any]:
        """List all budgets."""
        return await self._client.request("GET", "/api/v1/sme/budgets")

    async def create_budget(self, **kwargs: Any) -> dict[str, Any]:
        """Create a new budget."""
        return await self._client.request("POST", "/api/v1/sme/budgets", json=kwargs)

    async def get_budget(self, budget_id: str) -> dict[str, Any]:
        """Get a budget by ID."""
        return await self._client.request("GET", f"/api/v1/sme/budgets/{budget_id}")

    async def update_budget(self, budget_id: str, **kwargs: Any) -> dict[str, Any]:
        """Update a budget."""
        return await self._client.request("PATCH", f"/api/v1/sme/budgets/{budget_id}", json=kwargs)

    async def delete_budget(self, budget_id: str) -> dict[str, Any]:
        """Delete a budget."""
        return await self._client.request("DELETE", f"/api/v1/sme/budgets/{budget_id}")

    async def get_budget_alerts(self, budget_id: str) -> dict[str, Any]:
        """Get alerts for a budget."""
        return await self._client.request("GET", f"/api/v1/sme/budgets/{budget_id}/alerts")

    async def get_budget_transactions(self, budget_id: str) -> dict[str, Any]:
        """Get transactions for a budget."""
        return await self._client.request("GET", f"/api/v1/sme/budgets/{budget_id}/transactions")

    # ===========================================================================
    # Slack Integration
    # ===========================================================================

    async def get_slack_oauth_callback(self) -> dict[str, Any]:
        """Handle Slack OAuth callback."""
        return await self._client.request("GET", "/api/v1/sme/slack/oauth/callback")

    async def get_slack_oauth_start(self) -> dict[str, Any]:
        """Start Slack OAuth flow."""
        return await self._client.request("GET", "/api/v1/sme/slack/oauth/start")

    async def slack_subscribe(self, **kwargs: Any) -> dict[str, Any]:
        """Create a Slack subscription."""
        return await self._client.request("POST", "/api/v1/sme/slack/subscribe", json=kwargs)

    async def list_slack_subscriptions(self) -> dict[str, Any]:
        """List Slack subscriptions."""
        return await self._client.request("GET", "/api/v1/sme/slack/subscriptions")

    async def delete_slack_subscription(self, subscription_id: str) -> dict[str, Any]:
        """Delete a Slack subscription."""
        return await self._client.request("DELETE", f"/api/v1/sme/slack/subscriptions/{subscription_id}")

    async def list_slack_workspaces(self) -> dict[str, Any]:
        """List Slack workspaces."""
        return await self._client.request("GET", "/api/v1/sme/slack/workspaces")

    async def create_slack_workspace(self, **kwargs: Any) -> dict[str, Any]:
        """Create a Slack workspace integration."""
        return await self._client.request("POST", "/api/v1/sme/slack/workspaces", json=kwargs)

    async def get_slack_workspace(self, workspace_id: str) -> dict[str, Any]:
        """Get a Slack workspace by ID."""
        return await self._client.request("GET", f"/api/v1/sme/slack/workspaces/{workspace_id}")

    async def update_slack_workspace(self, workspace_id: str, **kwargs: Any) -> dict[str, Any]:
        """Update a Slack workspace."""
        return await self._client.request("PATCH", f"/api/v1/sme/slack/workspaces/{workspace_id}", json=kwargs)

    async def delete_slack_workspace(self, workspace_id: str) -> dict[str, Any]:
        """Delete a Slack workspace integration."""
        return await self._client.request("DELETE", f"/api/v1/sme/slack/workspaces/{workspace_id}")

    async def list_slack_channels(self, workspace_id: str) -> dict[str, Any]:
        """List channels in a Slack workspace."""
        return await self._client.request("GET", f"/api/v1/sme/slack/workspaces/{workspace_id}/channels")

    async def test_slack_workspace(self, workspace_id: str) -> dict[str, Any]:
        """Test a Slack workspace connection."""
        return await self._client.request("POST", f"/api/v1/sme/slack/workspaces/{workspace_id}/test")

    # ===========================================================================
    # Quick Helpers
    # ===========================================================================

    async def quick_invoice(
        self,
        customer_email: str,
        customer_name: str,
        items: list[dict[str, Any]],
        due_date: str | None = None,
    ) -> dict[str, Any]:
        """Quick invoice generation helper."""
        formatted_items = [
            {
                "name": item["name"],
                "unit_price": item["price"],
                "quantity": item.get("quantity", 1),
            }
            for item in items
        ]
        inputs: dict[str, Any] = {
            "customer_email": customer_email,
            "customer_name": customer_name,
            "items": formatted_items,
        }
        if due_date:
            inputs["due_date"] = due_date
        return await self.execute_workflow("invoice", inputs=inputs)

    async def quick_inventory_check(
        self,
        product_id: str,
        min_threshold: int,
        notification_email: str,
    ) -> dict[str, Any]:
        """Quick inventory check helper."""
        return await self.execute_workflow(
            "inventory",
            inputs={
                "product_id": product_id,
                "min_threshold": min_threshold,
                "notification_email": notification_email,
            },
        )

    async def quick_report(
        self,
        report_type: ReportType,
        period: ReportPeriod,
        format: ReportFormat = "pdf",
        email: str | None = None,
    ) -> dict[str, Any]:
        """Quick report generation helper."""
        inputs: dict[str, Any] = {
            "report_type": report_type,
            "period": period,
            "format": format,
        }
        if email:
            inputs["delivery_email"] = email
        return await self.execute_workflow("report", inputs=inputs)

    async def quick_followup(
        self,
        customer_id: str,
        followup_type: FollowupType,
        message: str | None = None,
        delay_days: int = 0,
    ) -> dict[str, Any]:
        """Quick customer follow-up helper."""
        inputs: dict[str, Any] = {
            "customer_id": customer_id,
            "followup_type": followup_type,
            "delay_days": delay_days,
        }
        if message:
            inputs["custom_message"] = message
        return await self.execute_workflow("followup", inputs=inputs)
