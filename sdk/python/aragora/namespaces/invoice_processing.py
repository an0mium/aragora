"""
Invoice Processing Namespace API.

Provides invoice processing and approval workflows:
- Document upload and OCR extraction
- Invoice approval/rejection
- PO matching
- Anomaly detection
- Payment scheduling
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient

InvoiceProcessingStatus = Literal["pending", "approved", "rejected", "paid", "processing"]
AnomalySeverity = Literal["critical", "high", "medium", "low"]


class InvoiceProcessingAPI:
    """
    Synchronous Invoice Processing API.

    Provides methods for:
    - Document upload and OCR extraction
    - Invoice approval/rejection
    - PO matching
    - Anomaly detection
    - Payment scheduling
    """

    def __init__(self, client: AragoraClient) -> None:
        self._client = client

    # =========================================================================
    # Document Processing
    # =========================================================================

    def upload(
        self,
        document_data: str,
        content_type: str | None = None,
        vendor_hint: str | None = None,
    ) -> dict[str, Any]:
        """
        Upload and process an invoice document.

        Args:
            document_data: Base64 encoded document data.
            content_type: MIME type (e.g., application/pdf).
            vendor_hint: Optional vendor name hint for matching.

        Returns:
            Processed invoice with detected anomalies.
        """
        data: dict[str, Any] = {"document_data": document_data}
        if content_type:
            data["content_type"] = content_type
        if vendor_hint:
            data["vendor_hint"] = vendor_hint

        return self._client._request("POST", "/api/v1/accounting/invoices/upload", json=data)

    def create(
        self,
        vendor_name: str,
        due_date: str,
        total_amount: float,
        vendor_id: str | None = None,
        invoice_number: str | None = None,
        invoice_date: str | None = None,
        tax_amount: float | None = None,
        line_items: list[dict[str, Any]] | None = None,
        po_number: str | None = None,
        notes: str | None = None,
    ) -> dict[str, Any]:
        """
        Create an invoice manually.

        Args:
            vendor_name: Vendor display name.
            due_date: Payment due date (ISO format).
            total_amount: Invoice total amount.
            vendor_id: Vendor identifier.
            invoice_number: Invoice number.
            invoice_date: Invoice date (ISO format).
            tax_amount: Tax amount.
            line_items: List of line items.
            po_number: Purchase order number.
            notes: Additional notes.

        Returns:
            Created invoice details.
        """
        data: dict[str, Any] = {
            "vendor_name": vendor_name,
            "due_date": due_date,
            "total_amount": total_amount,
        }
        if vendor_id:
            data["vendor_id"] = vendor_id
        if invoice_number:
            data["invoice_number"] = invoice_number
        if invoice_date:
            data["invoice_date"] = invoice_date
        if tax_amount is not None:
            data["tax_amount"] = tax_amount
        if line_items:
            data["line_items"] = line_items
        if po_number:
            data["po_number"] = po_number
        if notes:
            data["notes"] = notes

        return self._client._request("POST", "/api/v1/accounting/invoices", json=data)

    def list(
        self,
        status: InvoiceProcessingStatus | None = None,
        vendor_id: str | None = None,
        approver_id: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """
        List invoices with filtering.

        Args:
            status: Filter by status.
            vendor_id: Filter by vendor.
            approver_id: Filter by approver.
            date_from: Filter from date.
            date_to: Filter to date.
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
        if approver_id:
            params["approver_id"] = approver_id
        if date_from:
            params["date_from"] = date_from
        if date_to:
            params["date_to"] = date_to

        return self._client._request("GET", "/api/v1/accounting/invoices", params=params)

    def get(self, invoice_id: str) -> dict[str, Any]:
        """
        Get a specific invoice.

        Args:
            invoice_id: Invoice identifier.

        Returns:
            Invoice details.
        """
        return self._client._request("GET", f"/api/v1/accounting/invoices/{invoice_id}")

    # =========================================================================
    # Approval Workflow
    # =========================================================================

    def approve(
        self,
        invoice_id: str,
        approver_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Approve an invoice.

        Args:
            invoice_id: Invoice identifier.
            approver_id: Optional approver ID.

        Returns:
            Approved invoice details.
        """
        data: dict[str, Any] = {}
        if approver_id:
            data["approver_id"] = approver_id

        return self._client._request(
            "POST", f"/api/v1/accounting/invoices/{invoice_id}/approve", json=data
        )

    def reject(
        self,
        invoice_id: str,
        reason: str | None = None,
    ) -> dict[str, Any]:
        """
        Reject an invoice.

        Args:
            invoice_id: Invoice identifier.
            reason: Rejection reason.

        Returns:
            Rejected invoice details.
        """
        data: dict[str, Any] = {}
        if reason:
            data["reason"] = reason

        return self._client._request(
            "POST", f"/api/v1/accounting/invoices/{invoice_id}/reject", json=data
        )

    def get_pending_approvals(self) -> dict[str, Any]:
        """
        Get invoices pending approval.

        Returns:
            List of pending invoices with count.
        """
        return self._client._request("GET", "/api/v1/accounting/invoices/pending")

    # =========================================================================
    # PO Matching
    # =========================================================================

    def match_to_po(self, invoice_id: str) -> dict[str, Any]:
        """
        Match an invoice to a purchase order.

        Args:
            invoice_id: Invoice identifier.

        Returns:
            Match result with variance details.
        """
        return self._client._request("POST", f"/api/v1/accounting/invoices/{invoice_id}/match")

    def create_purchase_order(
        self,
        vendor_id: str,
        vendor_name: str,
        line_items: list[dict[str, Any]],
        notes: str | None = None,
    ) -> dict[str, Any]:
        """
        Create a purchase order.

        Args:
            vendor_id: Vendor identifier.
            vendor_name: Vendor display name.
            line_items: List of line items with description, quantity, unit_price.
            notes: Additional notes.

        Returns:
            Created purchase order.
        """
        data: dict[str, Any] = {
            "vendor_id": vendor_id,
            "vendor_name": vendor_name,
            "line_items": line_items,
        }
        if notes:
            data["notes"] = notes

        return self._client._request("POST", "/api/v1/accounting/purchase-orders", json=data)

    # =========================================================================
    # Anomaly Detection
    # =========================================================================

    def get_anomalies(self, invoice_id: str) -> dict[str, Any]:
        """
        Get anomalies detected in an invoice.

        Args:
            invoice_id: Invoice identifier.

        Returns:
            List of anomalies with severity.
        """
        return self._client._request("GET", f"/api/v1/accounting/invoices/{invoice_id}/anomalies")

    # =========================================================================
    # Payment Scheduling
    # =========================================================================

    def schedule_payment(
        self,
        invoice_id: str,
        payment_date: str | None = None,
        payment_method: str | None = None,
        notes: str | None = None,
    ) -> dict[str, Any]:
        """
        Schedule payment for an invoice.

        Args:
            invoice_id: Invoice identifier.
            payment_date: Scheduled payment date (ISO format).
            payment_method: Payment method.
            notes: Payment notes.

        Returns:
            Scheduled payment details.
        """
        data: dict[str, Any] = {}
        if payment_date:
            data["payment_date"] = payment_date
        if payment_method:
            data["payment_method"] = payment_method
        if notes:
            data["notes"] = notes

        return self._client._request(
            "POST", f"/api/v1/accounting/invoices/{invoice_id}/schedule", json=data
        )

    def get_scheduled_payments(
        self,
        status: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """
        Get scheduled payments.

        Args:
            status: Filter by status.
            date_from: Filter from date.
            date_to: Filter to date.
            limit: Maximum results.
            offset: Pagination offset.

        Returns:
            List of scheduled payments.
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status
        if date_from:
            params["date_from"] = date_from
        if date_to:
            params["date_to"] = date_to

        return self._client._request("GET", "/api/v1/accounting/payments/scheduled", params=params)

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_overdue(self) -> dict[str, Any]:
        """
        Get overdue invoices.

        Returns:
            List of overdue invoices with total amount.
        """
        return self._client._request("GET", "/api/v1/accounting/invoices/overdue")

    def get_stats(self, period: str | None = None) -> dict[str, Any]:
        """
        Get invoice processing statistics.

        Args:
            period: Statistics period (e.g., "30d", "90d").

        Returns:
            Invoice processing statistics.
        """
        params: dict[str, Any] = {}
        if period:
            params["period"] = period

        return self._client._request("GET", "/api/v1/accounting/invoices/stats", params=params)


class AsyncInvoiceProcessingAPI:
    """Asynchronous Invoice Processing API."""

    def __init__(self, client: AragoraAsyncClient) -> None:
        self._client = client

    # =========================================================================
    # Document Processing
    # =========================================================================

    async def upload(
        self,
        document_data: str,
        content_type: str | None = None,
        vendor_hint: str | None = None,
    ) -> dict[str, Any]:
        """Upload and process an invoice document."""
        data: dict[str, Any] = {"document_data": document_data}
        if content_type:
            data["content_type"] = content_type
        if vendor_hint:
            data["vendor_hint"] = vendor_hint

        return await self._client._request("POST", "/api/v1/accounting/invoices/upload", json=data)

    async def create(
        self,
        vendor_name: str,
        due_date: str,
        total_amount: float,
        vendor_id: str | None = None,
        invoice_number: str | None = None,
        invoice_date: str | None = None,
        tax_amount: float | None = None,
        line_items: list[dict[str, Any]] | None = None,
        po_number: str | None = None,
        notes: str | None = None,
    ) -> dict[str, Any]:
        """Create an invoice manually."""
        data: dict[str, Any] = {
            "vendor_name": vendor_name,
            "due_date": due_date,
            "total_amount": total_amount,
        }
        if vendor_id:
            data["vendor_id"] = vendor_id
        if invoice_number:
            data["invoice_number"] = invoice_number
        if invoice_date:
            data["invoice_date"] = invoice_date
        if tax_amount is not None:
            data["tax_amount"] = tax_amount
        if line_items:
            data["line_items"] = line_items
        if po_number:
            data["po_number"] = po_number
        if notes:
            data["notes"] = notes

        return await self._client._request("POST", "/api/v1/accounting/invoices", json=data)

    async def list(
        self,
        status: InvoiceProcessingStatus | None = None,
        vendor_id: str | None = None,
        approver_id: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """List invoices with filtering."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status
        if vendor_id:
            params["vendor_id"] = vendor_id
        if approver_id:
            params["approver_id"] = approver_id
        if date_from:
            params["date_from"] = date_from
        if date_to:
            params["date_to"] = date_to

        return await self._client._request("GET", "/api/v1/accounting/invoices", params=params)

    async def get(self, invoice_id: str) -> dict[str, Any]:
        """Get a specific invoice."""
        return await self._client._request("GET", f"/api/v1/accounting/invoices/{invoice_id}")

    # =========================================================================
    # Approval Workflow
    # =========================================================================

    async def approve(
        self,
        invoice_id: str,
        approver_id: str | None = None,
    ) -> dict[str, Any]:
        """Approve an invoice."""
        data: dict[str, Any] = {}
        if approver_id:
            data["approver_id"] = approver_id

        return await self._client._request(
            "POST", f"/api/v1/accounting/invoices/{invoice_id}/approve", json=data
        )

    async def reject(
        self,
        invoice_id: str,
        reason: str | None = None,
    ) -> dict[str, Any]:
        """Reject an invoice."""
        data: dict[str, Any] = {}
        if reason:
            data["reason"] = reason

        return await self._client._request(
            "POST", f"/api/v1/accounting/invoices/{invoice_id}/reject", json=data
        )

    async def get_pending_approvals(self) -> dict[str, Any]:
        """Get invoices pending approval."""
        return await self._client._request("GET", "/api/v1/accounting/invoices/pending")

    # =========================================================================
    # PO Matching
    # =========================================================================

    async def match_to_po(self, invoice_id: str) -> dict[str, Any]:
        """Match an invoice to a purchase order."""
        return await self._client._request(
            "POST", f"/api/v1/accounting/invoices/{invoice_id}/match"
        )

    async def create_purchase_order(
        self,
        vendor_id: str,
        vendor_name: str,
        line_items: list[dict[str, Any]],
        notes: str | None = None,
    ) -> dict[str, Any]:
        """Create a purchase order."""
        data: dict[str, Any] = {
            "vendor_id": vendor_id,
            "vendor_name": vendor_name,
            "line_items": line_items,
        }
        if notes:
            data["notes"] = notes

        return await self._client._request("POST", "/api/v1/accounting/purchase-orders", json=data)

    # =========================================================================
    # Anomaly Detection
    # =========================================================================

    async def get_anomalies(self, invoice_id: str) -> dict[str, Any]:
        """Get anomalies detected in an invoice."""
        return await self._client._request(
            "GET", f"/api/v1/accounting/invoices/{invoice_id}/anomalies"
        )

    # =========================================================================
    # Payment Scheduling
    # =========================================================================

    async def schedule_payment(
        self,
        invoice_id: str,
        payment_date: str | None = None,
        payment_method: str | None = None,
        notes: str | None = None,
    ) -> dict[str, Any]:
        """Schedule payment for an invoice."""
        data: dict[str, Any] = {}
        if payment_date:
            data["payment_date"] = payment_date
        if payment_method:
            data["payment_method"] = payment_method
        if notes:
            data["notes"] = notes

        return await self._client._request(
            "POST", f"/api/v1/accounting/invoices/{invoice_id}/schedule", json=data
        )

    async def get_scheduled_payments(
        self,
        status: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Get scheduled payments."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status
        if date_from:
            params["date_from"] = date_from
        if date_to:
            params["date_to"] = date_to

        return await self._client._request(
            "GET", "/api/v1/accounting/payments/scheduled", params=params
        )

    # =========================================================================
    # Statistics
    # =========================================================================

    async def get_overdue(self) -> dict[str, Any]:
        """Get overdue invoices."""
        return await self._client._request("GET", "/api/v1/accounting/invoices/overdue")

    async def get_stats(self, period: str | None = None) -> dict[str, Any]:
        """Get invoice processing statistics."""
        params: dict[str, Any] = {}
        if period:
            params["period"] = period

        return await self._client._request(
            "GET", "/api/v1/accounting/invoices/stats", params=params
        )
