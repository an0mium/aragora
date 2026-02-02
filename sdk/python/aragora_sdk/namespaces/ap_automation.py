"""
AP Automation Namespace API.

Provides Accounts Payable automation:
- Invoice management
- Payment optimization
- Cash flow forecasting
- Early payment discount detection
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient

PaymentPriority = Literal["critical", "high", "normal", "low", "hold"]
APPaymentMethod = Literal["ach", "wire", "check", "credit_card"]
APInvoiceStatus = Literal["unpaid", "partial", "paid", "overdue"]


class APAutomationAPI:
    """
    Synchronous Accounts Payable Automation API.

    Provides methods for:
    - Invoice management and tracking
    - Payment optimization and scheduling
    - Cash flow forecasting
    - Early payment discount capture
    """

    def __init__(self, client: AragoraClient) -> None:
        self._client = client

    # =========================================================================
    # Invoice Management
    # =========================================================================

    def add_invoice(
        self,
        vendor_id: str,
        vendor_name: str,
        due_date: str,
        total_amount: float,
        invoice_number: str | None = None,
        invoice_date: str | None = None,
        payment_terms: str | None = None,
        early_pay_discount: float | None = None,
        discount_deadline: str | None = None,
        priority: PaymentPriority | None = None,
        preferred_payment_method: APPaymentMethod | None = None,
        notes: str | None = None,
    ) -> dict[str, Any]:
        """
        Add a new AP invoice.

        Args:
            vendor_id: Vendor identifier.
            vendor_name: Vendor display name.
            due_date: Payment due date (ISO format).
            total_amount: Invoice total amount.
            invoice_number: Optional invoice number.
            invoice_date: Invoice date (ISO format).
            payment_terms: Payment terms (e.g., "Net 30").
            early_pay_discount: Early payment discount percentage.
            discount_deadline: Discount deadline (ISO format).
            priority: Payment priority level.
            preferred_payment_method: Preferred payment method.
            notes: Additional notes.

        Returns:
            Created invoice details.
        """
        data: dict[str, Any] = {
            "vendor_id": vendor_id,
            "vendor_name": vendor_name,
            "due_date": due_date,
            "total_amount": total_amount,
        }
        if invoice_number:
            data["invoice_number"] = invoice_number
        if invoice_date:
            data["invoice_date"] = invoice_date
        if payment_terms:
            data["payment_terms"] = payment_terms
        if early_pay_discount is not None:
            data["early_pay_discount"] = early_pay_discount
        if discount_deadline:
            data["discount_deadline"] = discount_deadline
        if priority:
            data["priority"] = priority
        if preferred_payment_method:
            data["preferred_payment_method"] = preferred_payment_method
        if notes:
            data["notes"] = notes

        return self._client._request("POST", "/api/v2/ap/invoices", json=data)

    def list_invoices(
        self,
        status: APInvoiceStatus | None = None,
        vendor_id: str | None = None,
        priority: PaymentPriority | None = None,
        due_before: str | None = None,
        due_after: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """
        List AP invoices with filtering.

        Args:
            status: Filter by invoice status.
            vendor_id: Filter by vendor.
            priority: Filter by priority.
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
        if vendor_id:
            params["vendor_id"] = vendor_id
        if priority:
            params["priority"] = priority
        if due_before:
            params["due_before"] = due_before
        if due_after:
            params["due_after"] = due_after

        return self._client._request("GET", "/api/v2/ap/invoices", params=params)

    def get_invoice(self, invoice_id: str) -> dict[str, Any]:
        """
        Get AP invoice details.

        Args:
            invoice_id: Invoice identifier.

        Returns:
            Invoice details.
        """
        return self._client._request("GET", f"/api/v2/ap/invoices/{invoice_id}")

    def record_payment(
        self,
        invoice_id: str,
        amount: float,
        payment_date: str | None = None,
        payment_method: APPaymentMethod | None = None,
        reference_number: str | None = None,
        notes: str | None = None,
    ) -> dict[str, Any]:
        """
        Record a payment against an AP invoice.

        Args:
            invoice_id: Invoice identifier.
            amount: Payment amount.
            payment_date: Payment date (ISO format).
            payment_method: Payment method used.
            reference_number: Check/reference number.
            notes: Payment notes.

        Returns:
            Payment confirmation with updated invoice status.
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
            "POST", f"/api/v2/ap/invoices/{invoice_id}/payments", json=data
        )

    # =========================================================================
    # Payment Optimization
    # =========================================================================

    def optimize_payments(
        self,
        available_cash: float | None = None,
        target_date: str | None = None,
        prioritize_discounts: bool = True,
        include_vendors: list[str] | None = None,
        exclude_vendors: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Optimize payment schedule to maximize discounts and manage cash flow.

        Args:
            available_cash: Available cash for payments.
            target_date: Target date for payments (ISO format).
            prioritize_discounts: Prioritize early payment discounts.
            include_vendors: Only include these vendors.
            exclude_vendors: Exclude these vendors.

        Returns:
            Optimized payment schedule with discount opportunities.
        """
        data: dict[str, Any] = {"prioritize_discounts": prioritize_discounts}
        if available_cash is not None:
            data["available_cash"] = available_cash
        if target_date:
            data["target_date"] = target_date
        if include_vendors:
            data["include_vendors"] = include_vendors
        if exclude_vendors:
            data["exclude_vendors"] = exclude_vendors

        return self._client._request("POST", "/api/v2/ap/optimize", json=data)

    def create_batch_payment(
        self,
        invoice_ids: list[str],
        payment_date: str | None = None,
        payment_method: APPaymentMethod | None = None,
        notes: str | None = None,
    ) -> dict[str, Any]:
        """
        Create a batch payment for multiple invoices.

        Args:
            invoice_ids: List of invoice IDs to pay.
            payment_date: Payment date (ISO format).
            payment_method: Payment method.
            notes: Batch notes.

        Returns:
            Batch payment details with status.
        """
        data: dict[str, Any] = {"invoice_ids": invoice_ids}
        if payment_date:
            data["payment_date"] = payment_date
        if payment_method:
            data["payment_method"] = payment_method
        if notes:
            data["notes"] = notes

        return self._client._request("POST", "/api/v2/ap/batch-payments", json=data)

    # =========================================================================
    # Cash Flow
    # =========================================================================

    def get_cash_flow_forecast(
        self,
        days: int = 30,
        include_pending: bool = True,
    ) -> dict[str, Any]:
        """
        Get cash flow forecast for AP.

        Args:
            days: Forecast period in days.
            include_pending: Include pending invoices.

        Returns:
            Cash flow forecast with projections.
        """
        params = {"days": days, "include_pending": include_pending}
        return self._client._request("GET", "/api/v2/ap/cash-flow", params=params)

    def get_discount_opportunities(self) -> dict[str, Any]:
        """
        Get available early payment discount opportunities.

        Returns:
            List of invoices with discount opportunities and potential savings.
        """
        return self._client._request("GET", "/api/v2/ap/discount-opportunities")


class AsyncAPAutomationAPI:
    """Asynchronous Accounts Payable Automation API."""

    def __init__(self, client: AragoraAsyncClient) -> None:
        self._client = client

    # =========================================================================
    # Invoice Management
    # =========================================================================

    async def add_invoice(
        self,
        vendor_id: str,
        vendor_name: str,
        due_date: str,
        total_amount: float,
        invoice_number: str | None = None,
        invoice_date: str | None = None,
        payment_terms: str | None = None,
        early_pay_discount: float | None = None,
        discount_deadline: str | None = None,
        priority: PaymentPriority | None = None,
        preferred_payment_method: APPaymentMethod | None = None,
        notes: str | None = None,
    ) -> dict[str, Any]:
        """Add a new AP invoice."""
        data: dict[str, Any] = {
            "vendor_id": vendor_id,
            "vendor_name": vendor_name,
            "due_date": due_date,
            "total_amount": total_amount,
        }
        if invoice_number:
            data["invoice_number"] = invoice_number
        if invoice_date:
            data["invoice_date"] = invoice_date
        if payment_terms:
            data["payment_terms"] = payment_terms
        if early_pay_discount is not None:
            data["early_pay_discount"] = early_pay_discount
        if discount_deadline:
            data["discount_deadline"] = discount_deadline
        if priority:
            data["priority"] = priority
        if preferred_payment_method:
            data["preferred_payment_method"] = preferred_payment_method
        if notes:
            data["notes"] = notes

        return await self._client._request("POST", "/api/v2/ap/invoices", json=data)

    async def list_invoices(
        self,
        status: APInvoiceStatus | None = None,
        vendor_id: str | None = None,
        priority: PaymentPriority | None = None,
        due_before: str | None = None,
        due_after: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """List AP invoices with filtering."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status
        if vendor_id:
            params["vendor_id"] = vendor_id
        if priority:
            params["priority"] = priority
        if due_before:
            params["due_before"] = due_before
        if due_after:
            params["due_after"] = due_after

        return await self._client._request("GET", "/api/v2/ap/invoices", params=params)

    async def get_invoice(self, invoice_id: str) -> dict[str, Any]:
        """Get AP invoice details."""
        return await self._client._request("GET", f"/api/v2/ap/invoices/{invoice_id}")

    async def record_payment(
        self,
        invoice_id: str,
        amount: float,
        payment_date: str | None = None,
        payment_method: APPaymentMethod | None = None,
        reference_number: str | None = None,
        notes: str | None = None,
    ) -> dict[str, Any]:
        """Record a payment against an AP invoice."""
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
            "POST", f"/api/v2/ap/invoices/{invoice_id}/payments", json=data
        )

    # =========================================================================
    # Payment Optimization
    # =========================================================================

    async def optimize_payments(
        self,
        available_cash: float | None = None,
        target_date: str | None = None,
        prioritize_discounts: bool = True,
        include_vendors: list[str] | None = None,
        exclude_vendors: list[str] | None = None,
    ) -> dict[str, Any]:
        """Optimize payment schedule."""
        data: dict[str, Any] = {"prioritize_discounts": prioritize_discounts}
        if available_cash is not None:
            data["available_cash"] = available_cash
        if target_date:
            data["target_date"] = target_date
        if include_vendors:
            data["include_vendors"] = include_vendors
        if exclude_vendors:
            data["exclude_vendors"] = exclude_vendors

        return await self._client._request("POST", "/api/v2/ap/optimize", json=data)

    async def create_batch_payment(
        self,
        invoice_ids: list[str],
        payment_date: str | None = None,
        payment_method: APPaymentMethod | None = None,
        notes: str | None = None,
    ) -> dict[str, Any]:
        """Create a batch payment for multiple invoices."""
        data: dict[str, Any] = {"invoice_ids": invoice_ids}
        if payment_date:
            data["payment_date"] = payment_date
        if payment_method:
            data["payment_method"] = payment_method
        if notes:
            data["notes"] = notes

        return await self._client._request("POST", "/api/v2/ap/batch-payments", json=data)

    # =========================================================================
    # Cash Flow
    # =========================================================================

    async def get_cash_flow_forecast(
        self,
        days: int = 30,
        include_pending: bool = True,
    ) -> dict[str, Any]:
        """Get cash flow forecast for AP."""
        params = {"days": days, "include_pending": include_pending}
        return await self._client._request("GET", "/api/v2/ap/cash-flow", params=params)

    async def get_discount_opportunities(self) -> dict[str, Any]:
        """Get available early payment discount opportunities."""
        return await self._client._request("GET", "/api/v2/ap/discount-opportunities")
