"""
Accounting Namespace API.

Provides a namespaced interface for QuickBooks Online and Gusto payroll integration.

Note: The accounting backend uses direct route registration (app.router.add_*)
rather than the ROUTES class-variable pattern. SDK methods will be re-added once
the handler is migrated to the standard ROUTES pattern for parity tracking.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class AccountingAPI:
    """Synchronous Accounting API."""

    def __init__(self, client: AragoraClient) -> None:
        self._client = client

    # =========================================================================
    # Gusto Payroll Integration
    # =========================================================================

    def list_gusto_employees(
        self,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """
        List Gusto employees.

        Args:
            limit: Maximum number of employees to return.
            offset: Pagination offset.

        Returns:
            Employee list with pagination.
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        return self._client.request("GET", "/api/v1/accounting/gusto/employees", params=params)

    def list_gusto_payrolls(
        self,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """
        List Gusto payroll runs.

        Args:
            limit: Maximum number of payrolls to return.
            offset: Pagination offset.

        Returns:
            Payroll list with pagination.
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        return self._client.request("GET", "/api/v1/accounting/gusto/payrolls", params=params)

    def get_gusto_status(self) -> dict[str, Any]:
        """
        Get Gusto integration status.

        Returns:
            Gusto connection status and company information.
        """
        return self._client.request("GET", "/api/v1/accounting/gusto/status")

    # =========================================================================
    # Invoice Status
    # =========================================================================

    def get_invoice_status(self) -> dict[str, Any]:
        """
        Get invoice processing status summary.

        Returns:
            Invoice status overview.
        """
        return self._client.request("GET", "/api/v1/accounting/invoices/status")

    def update_invoice_status(
        self,
        invoice_id: str | None = None,
        status: str | None = None,
    ) -> dict[str, Any]:
        """
        Update invoice processing status.

        Args:
            invoice_id: Invoice to update.
            status: New status value.

        Returns:
            Updated invoice status.
        """
        data: dict[str, Any] = {}
        if invoice_id is not None:
            data["invoice_id"] = invoice_id
        if status is not None:
            data["status"] = status
        return self._client.request("POST", "/api/v1/accounting/invoices/status", json=data)


class AsyncAccountingAPI:
    """Asynchronous Accounting API."""

    def __init__(self, client: AragoraAsyncClient) -> None:
        self._client = client

    # =========================================================================
    # Gusto Payroll Integration
    # =========================================================================

    async def list_gusto_employees(
        self,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """List Gusto employees."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        return await self._client.request(
            "GET", "/api/v1/accounting/gusto/employees", params=params
        )

    async def list_gusto_payrolls(
        self,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """List Gusto payroll runs."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        return await self._client.request(
            "GET", "/api/v1/accounting/gusto/payrolls", params=params
        )

    async def get_gusto_status(self) -> dict[str, Any]:
        """Get Gusto integration status."""
        return await self._client.request("GET", "/api/v1/accounting/gusto/status")

    # =========================================================================
    # Invoice Status
    # =========================================================================

    async def get_invoice_status(self) -> dict[str, Any]:
        """Get invoice processing status summary."""
        return await self._client.request("GET", "/api/v1/accounting/invoices/status")

    async def update_invoice_status(
        self,
        invoice_id: str | None = None,
        status: str | None = None,
    ) -> dict[str, Any]:
        """Update invoice processing status."""
        data: dict[str, Any] = {}
        if invoice_id is not None:
            data["invoice_id"] = invoice_id
        if status is not None:
            data["status"] = status
        return await self._client.request(
            "POST", "/api/v1/accounting/invoices/status", json=data
        )
