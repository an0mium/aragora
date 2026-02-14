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
