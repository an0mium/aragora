"""
Invoice Processing Namespace API.

Provides invoice processing and approval workflows.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class InvoiceProcessingAPI:
    """Synchronous Invoice Processing API."""

    def __init__(self, client: AragoraClient):
        self._client = client

    def list_invoices(self) -> dict[str, Any]:
        """List invoices."""
        return self._client.request("GET", "/api/v1/accounting/invoices")

    def create_invoice(self, **kwargs: Any) -> dict[str, Any]:
        """Create an invoice."""
        return self._client.request("POST", "/api/v1/accounting/invoices", json=kwargs)

    def get_invoice(self, invoice_id: str) -> dict[str, Any]:
        """Get an invoice by ID."""
        return self._client.request("GET", f"/api/v1/accounting/invoices/{invoice_id}")

    def get_overdue(self) -> dict[str, Any]:
        """Get overdue invoices."""
        return self._client.request("GET", "/api/v1/accounting/invoices/overdue")

    def get_pending(self) -> dict[str, Any]:
        """Get pending invoices."""
        return self._client.request("GET", "/api/v1/accounting/invoices/pending")

    def get_stats(self) -> dict[str, Any]:
        """Get invoice statistics."""
        return self._client.request("GET", "/api/v1/accounting/invoices/stats")

    def get_status(self) -> dict[str, Any]:
        """Get invoice status summary."""
        return self._client.request("GET", "/api/v1/accounting/invoices/status")

    def upload(self, **kwargs: Any) -> dict[str, Any]:
        """Upload an invoice document."""
        return self._client.request("POST", "/api/v1/accounting/invoices/upload", json=kwargs)

    def get_anomalies(self, invoice_id: str) -> dict[str, Any]:
        """Get anomalies detected for an invoice."""
        return self._client.request("GET", f"/api/v1/accounting/invoices/{invoice_id}/anomalies")

    def approve(self, invoice_id: str) -> dict[str, Any]:
        """Approve an invoice."""
        return self._client.request("POST", f"/api/v1/accounting/invoices/{invoice_id}/approve")

    def reject(self, invoice_id: str, **kwargs: Any) -> dict[str, Any]:
        """Reject an invoice."""
        return self._client.request("POST", f"/api/v1/accounting/invoices/{invoice_id}/reject", json=kwargs)

    def match(self, invoice_id: str, **kwargs: Any) -> dict[str, Any]:
        """Match an invoice to a purchase order."""
        return self._client.request("POST", f"/api/v1/accounting/invoices/{invoice_id}/match", json=kwargs)

    def schedule(self, invoice_id: str, **kwargs: Any) -> dict[str, Any]:
        """Schedule an invoice for payment."""
        return self._client.request("POST", f"/api/v1/accounting/invoices/{invoice_id}/schedule", json=kwargs)


class AsyncInvoiceProcessingAPI:
    """Asynchronous Invoice Processing API."""

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def list_invoices(self) -> dict[str, Any]:
        """List invoices."""
        return await self._client.request("GET", "/api/v1/accounting/invoices")

    async def create_invoice(self, **kwargs: Any) -> dict[str, Any]:
        """Create an invoice."""
        return await self._client.request("POST", "/api/v1/accounting/invoices", json=kwargs)

    async def get_invoice(self, invoice_id: str) -> dict[str, Any]:
        """Get an invoice by ID."""
        return await self._client.request("GET", f"/api/v1/accounting/invoices/{invoice_id}")

    async def get_overdue(self) -> dict[str, Any]:
        """Get overdue invoices."""
        return await self._client.request("GET", "/api/v1/accounting/invoices/overdue")

    async def get_pending(self) -> dict[str, Any]:
        """Get pending invoices."""
        return await self._client.request("GET", "/api/v1/accounting/invoices/pending")

    async def get_stats(self) -> dict[str, Any]:
        """Get invoice statistics."""
        return await self._client.request("GET", "/api/v1/accounting/invoices/stats")

    async def get_status(self) -> dict[str, Any]:
        """Get invoice status summary."""
        return await self._client.request("GET", "/api/v1/accounting/invoices/status")

    async def upload(self, **kwargs: Any) -> dict[str, Any]:
        """Upload an invoice document."""
        return await self._client.request("POST", "/api/v1/accounting/invoices/upload", json=kwargs)

    async def get_anomalies(self, invoice_id: str) -> dict[str, Any]:
        """Get anomalies detected for an invoice."""
        return await self._client.request("GET", f"/api/v1/accounting/invoices/{invoice_id}/anomalies")

    async def approve(self, invoice_id: str) -> dict[str, Any]:
        """Approve an invoice."""
        return await self._client.request("POST", f"/api/v1/accounting/invoices/{invoice_id}/approve")

    async def reject(self, invoice_id: str, **kwargs: Any) -> dict[str, Any]:
        """Reject an invoice."""
        return await self._client.request("POST", f"/api/v1/accounting/invoices/{invoice_id}/reject", json=kwargs)

    async def match(self, invoice_id: str, **kwargs: Any) -> dict[str, Any]:
        """Match an invoice to a purchase order."""
        return await self._client.request("POST", f"/api/v1/accounting/invoices/{invoice_id}/match", json=kwargs)

    async def schedule(self, invoice_id: str, **kwargs: Any) -> dict[str, Any]:
        """Schedule an invoice for payment."""
        return await self._client.request("POST", f"/api/v1/accounting/invoices/{invoice_id}/schedule", json=kwargs)
