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
    - Pre-built SME workflow templates (invoice, followup, inventory, reports)
    - Onboarding status and completion
    - Quick-start helpers for common tasks

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai")
        >>> workflows = client.sme.list_workflows()
        >>> result = client.sme.execute_workflow("invoice", inputs={"customer_email": "..."})
    """

    def __init__(self, client: AragoraClient):
        self._client = client

    # ===========================================================================
    # SME Workflows
    # ===========================================================================

    def list_workflows(
        self,
        category: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """
        List available SME workflow templates.

        SME workflows are pre-built templates for common small business tasks:
        - invoice: Generate and send invoices
        - followup: Customer follow-up campaigns
        - inventory: Stock level monitoring and alerts
        - report: Automated business reports

        Args:
            category: Filter by category
            limit: Maximum results
            offset: Pagination offset

        Returns:
            List of workflow templates
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if category:
            params["category"] = category

        return self._client.request("GET", "/api/v1/sme/workflows", params=params)

    def get_workflow(self, workflow_id: str) -> dict[str, Any]:
        """
        Get details of a specific SME workflow template.

        Args:
            workflow_id: Workflow identifier (e.g., "invoice", "inventory")

        Returns:
            Workflow template details
        """
        return self._client.request("GET", f"/api/v1/sme/workflows/{workflow_id}")

    def execute_workflow(
        self,
        workflow_id: str,
        inputs: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
        execute: bool = True,
        tenant_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Execute an SME workflow template with inputs.

        Args:
            workflow_id: Workflow identifier
            inputs: Workflow input parameters
            context: Additional context
            execute: Whether to execute immediately
            tenant_id: Tenant ID for multi-tenant setups

        Returns:
            Execution result with execution_id
        """
        data: dict[str, Any] = {"execute": execute}
        if inputs:
            data["inputs"] = inputs
        if context:
            data["context"] = context
        if tenant_id:
            data["tenant_id"] = tenant_id

        return self._client.request(
            "POST", f"/api/v1/sme/workflows/{workflow_id}/execute", json=data
        )

    # ===========================================================================
    # Onboarding
    # ===========================================================================

    def get_onboarding_status(self) -> dict[str, Any]:
        """
        Get current onboarding status.

        Returns:
            Onboarding progress and status
        """
        return self._client.request("GET", "/api/v1/sme/onboarding/status")

    def complete_onboarding(
        self,
        first_debate_id: str | None = None,
        template_used: str | None = None,
    ) -> dict[str, Any]:
        """
        Mark onboarding as complete.

        Args:
            first_debate_id: ID of the first debate created during onboarding
            template_used: Name of the template used

        Returns:
            Completion confirmation
        """
        data: dict[str, Any] = {}
        if first_debate_id:
            data["first_debate_id"] = first_debate_id
        if template_used:
            data["template_used"] = template_used

        return self._client.request("POST", "/api/v1/sme/onboarding/complete", json=data)

    # ===========================================================================
    # Quick Start Helpers
    # ===========================================================================

    def quick_invoice(
        self,
        customer_email: str,
        customer_name: str,
        items: list[dict[str, Any]],
        due_date: str | None = None,
    ) -> dict[str, Any]:
        """
        Quick invoice generation helper.

        Args:
            customer_email: Customer email address
            customer_name: Customer name
            items: Invoice items with name, price, and optional quantity
            due_date: Invoice due date (ISO format)

        Returns:
            Execution result with execution_id

        Example:
            >>> result = client.sme.quick_invoice(
            ...     customer_email="billing@client.com",
            ...     customer_name="Client Corp",
            ...     items=[{"name": "Service", "price": 1000}]
            ... )
        """
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
        """
        Quick inventory check helper.

        Sets up inventory monitoring for a product.

        Args:
            product_id: Product SKU or ID
            min_threshold: Minimum stock threshold
            notification_email: Email for alerts

        Returns:
            Execution result with execution_id
        """
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
        """
        Quick report generation helper.

        Args:
            report_type: Type of report (sales, inventory, customers, financial)
            period: Report period (daily, weekly, monthly, quarterly)
            format: Output format (pdf, excel, html, json)
            email: Delivery email address

        Returns:
            Execution result with execution_id
        """
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
        """
        Quick customer follow-up helper.

        Creates a follow-up campaign for a customer.

        Args:
            customer_id: Customer identifier
            followup_type: Type of follow-up
            message: Custom message
            delay_days: Delay before sending

        Returns:
            Execution result with execution_id
        """
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
        ...     workflows = await client.sme.list_workflows()
    """

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    # ===========================================================================
    # SME Workflows
    # ===========================================================================

    async def list_workflows(
        self,
        category: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """List available SME workflow templates."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if category:
            params["category"] = category

        return await self._client.request("GET", "/api/v1/sme/workflows", params=params)

    async def get_workflow(self, workflow_id: str) -> dict[str, Any]:
        """Get details of a specific SME workflow template."""
        return await self._client.request("GET", f"/api/v1/sme/workflows/{workflow_id}")

    async def execute_workflow(
        self,
        workflow_id: str,
        inputs: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
        execute: bool = True,
        tenant_id: str | None = None,
    ) -> dict[str, Any]:
        """Execute an SME workflow template with inputs."""
        data: dict[str, Any] = {"execute": execute}
        if inputs:
            data["inputs"] = inputs
        if context:
            data["context"] = context
        if tenant_id:
            data["tenant_id"] = tenant_id

        return await self._client.request(
            "POST", f"/api/v1/sme/workflows/{workflow_id}/execute", json=data
        )

    # ===========================================================================
    # Onboarding
    # ===========================================================================

    async def get_onboarding_status(self) -> dict[str, Any]:
        """Get current onboarding status."""
        return await self._client.request("GET", "/api/v1/sme/onboarding/status")

    async def complete_onboarding(
        self,
        first_debate_id: str | None = None,
        template_used: str | None = None,
    ) -> dict[str, Any]:
        """Mark onboarding as complete."""
        data: dict[str, Any] = {}
        if first_debate_id:
            data["first_debate_id"] = first_debate_id
        if template_used:
            data["template_used"] = template_used

        return await self._client.request("POST", "/api/v1/sme/onboarding/complete", json=data)

    # ===========================================================================
    # Quick Start Helpers
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
