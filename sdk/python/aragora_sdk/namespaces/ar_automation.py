"""
AR Automation Namespace API.

Provides Accounts Receivable automation:
- Invoice creation and sending
- Payment reminders
- Aging reports
- Collection suggestions
- Customer balance tracking
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient

ARInvoiceStatus = Literal["draft", "sent", "viewed", "paid", "partial", "overdue"]
ReminderLevel = Literal[1, 2, 3, 4]
CollectionActionType = Literal[
    "send_reminder", "phone_call", "escalate_to_collections", "offer_payment_plan", "final_notice"
]


class ARAutomationAPI:
    """
    Synchronous Accounts Receivable Automation API.

    Provides methods for:
    - Invoice creation and sending
    - Payment reminders and escalation
    - Aging reports and analytics
    - Collection suggestions
    - Customer balance tracking
    """

    def __init__(self, client: AragoraClient) -> None:
        self._client = client

    # =========================================================================
    # Invoice Management
    # =========================================================================

    def create_invoice(
        self,
        customer_id: str,
        customer_name: str,
        due_date: str,
        line_items: list[dict[str, Any]],
        customer_email: str | None = None,
        invoice_date: str | None = None,
        payment_terms: str | None = None,
        notes: str | None = None,
        send_immediately: bool = False,
    ) -> dict[str, Any]:
        """
        Create a new AR invoice.

        Args:
            customer_id: Customer identifier.
            customer_name: Customer display name.
            due_date: Payment due date (ISO format).
            line_items: List of line items with description and amount.
            customer_email: Customer email for sending invoice.
            invoice_date: Invoice date (ISO format).
            payment_terms: Payment terms (e.g., "Net 30").
            notes: Additional notes.
            send_immediately: Send invoice immediately after creation.

        Returns:
            Created invoice details.
        """
        data: dict[str, Any] = {
            "customer_id": customer_id,
            "customer_name": customer_name,
            "due_date": due_date,
            "line_items": line_items,
        }
        if customer_email:
            data["customer_email"] = customer_email
        if invoice_date:
            data["invoice_date"] = invoice_date
        if payment_terms:
            data["payment_terms"] = payment_terms
        if notes:
            data["notes"] = notes
        if send_immediately:
            data["send_immediately"] = send_immediately

        return self._client._request("POST", "/api/v1/accounting/ar/invoices", json=data)

    def list_invoices(
        self,
        status: ARInvoiceStatus | None = None,
        customer_id: str | None = None,
        overdue_only: bool = False,
        due_before: str | None = None,
        due_after: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """
        List AR invoices with filtering.

        Args:
            status: Filter by invoice status.
            customer_id: Filter by customer.
            overdue_only: Only show overdue invoices.
            due_before: Filter by due date (before).
            due_after: Filter by due date (after).
            limit: Maximum results.
            offset: Pagination offset.

        Returns:
            List of invoices with total count.
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status
        if customer_id:
            params["customer_id"] = customer_id
        if overdue_only:
            params["overdue_only"] = overdue_only
        if due_before:
            params["due_before"] = due_before
        if due_after:
            params["due_after"] = due_after

        return self._client._request("GET", "/api/v1/accounting/ar/invoices", params=params)

    def get_invoice(self, invoice_id: str) -> dict[str, Any]:
        """
        Get AR invoice details.

        Args:
            invoice_id: Invoice identifier.

        Returns:
            Invoice details.
        """
        return self._client._request("GET", f"/api/v1/accounting/ar/invoices/{invoice_id}")

    def send_invoice(self, invoice_id: str) -> dict[str, Any]:
        """
        Send an invoice to the customer.

        Args:
            invoice_id: Invoice identifier.

        Returns:
            Send confirmation.
        """
        return self._client._request("POST", f"/api/v1/accounting/ar/invoices/{invoice_id}/send")

    def send_reminder(
        self,
        invoice_id: str,
        escalation_level: ReminderLevel | None = None,
    ) -> dict[str, Any]:
        """
        Send a payment reminder for an invoice.

        Args:
            invoice_id: Invoice identifier.
            escalation_level: Reminder escalation level (1-4).

        Returns:
            Reminder confirmation with next escalation level.
        """
        data: dict[str, Any] = {}
        if escalation_level:
            data["escalation_level"] = escalation_level

        return self._client._request(
            "POST", f"/api/v1/accounting/ar/invoices/{invoice_id}/reminder", json=data
        )

    def record_payment(
        self,
        invoice_id: str,
        amount: float,
        payment_date: str | None = None,
        payment_method: str | None = None,
        reference_number: str | None = None,
        notes: str | None = None,
    ) -> dict[str, Any]:
        """
        Record a payment against an AR invoice.

        Args:
            invoice_id: Invoice identifier.
            amount: Payment amount.
            payment_date: Payment date (ISO format).
            payment_method: Payment method used.
            reference_number: Check/reference number.
            notes: Payment notes.

        Returns:
            Payment confirmation with updated invoice.
        """
        data: dict[str, Any] = {"amount": amount}
        if payment_date:
            data["payment_date"] = payment_date
        if payment_method:
            data["payment_method"] = payment_method
        if reference_number:
            data["reference_number"] = reference_number
        if notes:
            data["notes"] = notes

        return self._client._request(
            "POST", f"/api/v1/accounting/ar/invoices/{invoice_id}/payment", json=data
        )

    # =========================================================================
    # Aging and Collections
    # =========================================================================

    def get_aging_report(self) -> dict[str, Any]:
        """
        Get AR aging report.

        Returns:
            Aging report with buckets (current, 1-30, 31-60, 61-90, 90+).
        """
        return self._client._request("GET", "/api/v1/accounting/ar/aging")

    def get_collection_suggestions(self) -> dict[str, Any]:
        """
        Get collection suggestions for overdue invoices.

        Returns:
            List of suggested collection actions.
        """
        return self._client._request("GET", "/api/v1/accounting/ar/collections")

    # =========================================================================
    # Customer Management
    # =========================================================================

    def add_customer(
        self,
        name: str,
        email: str | None = None,
        phone: str | None = None,
        billing_address: dict[str, str] | None = None,
        payment_terms: str | None = None,
        credit_limit: float | None = None,
        notes: str | None = None,
    ) -> dict[str, Any]:
        """
        Add a new customer.

        Args:
            name: Customer name.
            email: Customer email.
            phone: Customer phone.
            billing_address: Billing address dict (street, city, state, zip, country).
            payment_terms: Default payment terms.
            credit_limit: Credit limit.
            notes: Additional notes.

        Returns:
            Created customer details.
        """
        data: dict[str, Any] = {"name": name}
        if email:
            data["email"] = email
        if phone:
            data["phone"] = phone
        if billing_address:
            data["billing_address"] = billing_address
        if payment_terms:
            data["payment_terms"] = payment_terms
        if credit_limit is not None:
            data["credit_limit"] = credit_limit
        if notes:
            data["notes"] = notes

        return self._client._request("POST", "/api/v1/accounting/ar/customers", json=data)

    def get_customer_balance(self, customer_id: str) -> dict[str, Any]:
        """
        Get customer balance summary.

        Args:
            customer_id: Customer identifier.

        Returns:
            Customer balance with outstanding and overdue amounts.
        """
        return self._client._request(
            "GET", f"/api/v1/accounting/ar/customers/{customer_id}/balance"
        )


class AsyncARAutomationAPI:
    """Asynchronous Accounts Receivable Automation API."""

    def __init__(self, client: AragoraAsyncClient) -> None:
        self._client = client

    # =========================================================================
    # Invoice Management
    # =========================================================================

    async def create_invoice(
        self,
        customer_id: str,
        customer_name: str,
        due_date: str,
        line_items: list[dict[str, Any]],
        customer_email: str | None = None,
        invoice_date: str | None = None,
        payment_terms: str | None = None,
        notes: str | None = None,
        send_immediately: bool = False,
    ) -> dict[str, Any]:
        """Create a new AR invoice."""
        data: dict[str, Any] = {
            "customer_id": customer_id,
            "customer_name": customer_name,
            "due_date": due_date,
            "line_items": line_items,
        }
        if customer_email:
            data["customer_email"] = customer_email
        if invoice_date:
            data["invoice_date"] = invoice_date
        if payment_terms:
            data["payment_terms"] = payment_terms
        if notes:
            data["notes"] = notes
        if send_immediately:
            data["send_immediately"] = send_immediately

        return await self._client._request("POST", "/api/v1/accounting/ar/invoices", json=data)

    async def list_invoices(
        self,
        status: ARInvoiceStatus | None = None,
        customer_id: str | None = None,
        overdue_only: bool = False,
        due_before: str | None = None,
        due_after: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """List AR invoices with filtering."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status
        if customer_id:
            params["customer_id"] = customer_id
        if overdue_only:
            params["overdue_only"] = overdue_only
        if due_before:
            params["due_before"] = due_before
        if due_after:
            params["due_after"] = due_after

        return await self._client._request("GET", "/api/v1/accounting/ar/invoices", params=params)

    async def get_invoice(self, invoice_id: str) -> dict[str, Any]:
        """Get AR invoice details."""
        return await self._client._request("GET", f"/api/v1/accounting/ar/invoices/{invoice_id}")

    async def send_invoice(self, invoice_id: str) -> dict[str, Any]:
        """Send an invoice to the customer."""
        return await self._client._request(
            "POST", f"/api/v1/accounting/ar/invoices/{invoice_id}/send"
        )

    async def send_reminder(
        self,
        invoice_id: str,
        escalation_level: ReminderLevel | None = None,
    ) -> dict[str, Any]:
        """Send a payment reminder for an invoice."""
        data: dict[str, Any] = {}
        if escalation_level:
            data["escalation_level"] = escalation_level

        return await self._client._request(
            "POST", f"/api/v1/accounting/ar/invoices/{invoice_id}/reminder", json=data
        )

    async def record_payment(
        self,
        invoice_id: str,
        amount: float,
        payment_date: str | None = None,
        payment_method: str | None = None,
        reference_number: str | None = None,
        notes: str | None = None,
    ) -> dict[str, Any]:
        """Record a payment against an AR invoice."""
        data: dict[str, Any] = {"amount": amount}
        if payment_date:
            data["payment_date"] = payment_date
        if payment_method:
            data["payment_method"] = payment_method
        if reference_number:
            data["reference_number"] = reference_number
        if notes:
            data["notes"] = notes

        return await self._client._request(
            "POST", f"/api/v1/accounting/ar/invoices/{invoice_id}/payment", json=data
        )

    # =========================================================================
    # Aging and Collections
    # =========================================================================

    async def get_aging_report(self) -> dict[str, Any]:
        """Get AR aging report."""
        return await self._client._request("GET", "/api/v1/accounting/ar/aging")

    async def get_collection_suggestions(self) -> dict[str, Any]:
        """Get collection suggestions for overdue invoices."""
        return await self._client._request("GET", "/api/v1/accounting/ar/collections")

    # =========================================================================
    # Customer Management
    # =========================================================================

    async def add_customer(
        self,
        name: str,
        email: str | None = None,
        phone: str | None = None,
        billing_address: dict[str, str] | None = None,
        payment_terms: str | None = None,
        credit_limit: float | None = None,
        notes: str | None = None,
    ) -> dict[str, Any]:
        """Add a new customer."""
        data: dict[str, Any] = {"name": name}
        if email:
            data["email"] = email
        if phone:
            data["phone"] = phone
        if billing_address:
            data["billing_address"] = billing_address
        if payment_terms:
            data["payment_terms"] = payment_terms
        if credit_limit is not None:
            data["credit_limit"] = credit_limit
        if notes:
            data["notes"] = notes

        return await self._client._request("POST", "/api/v1/accounting/ar/customers", json=data)

    async def get_customer_balance(self, customer_id: str) -> dict[str, Any]:
        """Get customer balance summary."""
        return await self._client._request(
            "GET", f"/api/v1/accounting/ar/customers/{customer_id}/balance"
        )
