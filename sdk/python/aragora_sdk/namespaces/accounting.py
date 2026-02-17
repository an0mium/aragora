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
    # Connection Management
    # =========================================================================

    def get_status(self) -> dict[str, Any]:
        """Get accounting integration status."""
        return self._client.request("GET", "/api/v1/accounting/status")

    def connect(self) -> dict[str, Any]:
        """Connect accounting integration."""
        return self._client.request("POST", "/api/v1/accounting/connect")

    def disconnect(self) -> dict[str, Any]:
        """Disconnect accounting integration."""
        return self._client.request("POST", "/api/v1/accounting/disconnect")

    # =========================================================================
    # Customers
    # =========================================================================

    def list_customers(
        self,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """List accounting customers."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        return self._client.request("GET", "/api/v1/accounting/customers", params=params)

    # =========================================================================
    # Transactions
    # =========================================================================

    def list_transactions(
        self,
        limit: int = 50,
        offset: int = 0,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict[str, Any]:
        """List accounting transactions."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        return self._client.request("GET", "/api/v1/accounting/transactions", params=params)

    # =========================================================================
    # Reports
    # =========================================================================

    def generate_report(
        self,
        report_type: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict[str, Any]:
        """Generate an accounting report."""
        data: dict[str, Any] = {"report_type": report_type}
        if start_date:
            data["start_date"] = start_date
        if end_date:
            data["end_date"] = end_date
        return self._client.request("POST", "/api/v1/accounting/reports", json=data)

    # =========================================================================
    # Invoices
    # =========================================================================

    def list_invoices(
        self,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """List accounting invoices."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        return self._client.request("GET", "/api/v1/accounting/invoices", params=params)

    def upload_invoice(self, invoice_data: dict[str, Any]) -> dict[str, Any]:
        """Upload an invoice."""
        return self._client.request("POST", "/api/v1/accounting/invoices/upload", json=invoice_data)

    def get_pending_invoices(self) -> dict[str, Any]:
        """Get pending invoices."""
        return self._client.request("GET", "/api/v1/accounting/invoices/pending")

    def get_overdue_invoices(self) -> dict[str, Any]:
        """Get overdue invoices."""
        return self._client.request("GET", "/api/v1/accounting/invoices/overdue")

    def get_invoice_stats(self) -> dict[str, Any]:
        """Get invoice statistics."""
        return self._client.request("GET", "/api/v1/accounting/invoices/stats")

    # =========================================================================
    # Purchase Orders
    # =========================================================================

    def create_purchase_order(self, order: dict[str, Any]) -> dict[str, Any]:
        """Create a purchase order."""
        return self._client.request("POST", "/api/v1/accounting/purchase-orders", json=order)

    # =========================================================================
    # Payments
    # =========================================================================

    def get_scheduled_payments(
        self,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Get scheduled payments."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        return self._client.request("GET", "/api/v1/accounting/payments/scheduled", params=params)

    # =========================================================================
    # Expenses
    # =========================================================================

    def list_expenses(
        self,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """List expenses."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        return self._client.request("GET", "/api/v1/accounting/expenses", params=params)

    def upload_expense(self, expense_data: dict[str, Any]) -> dict[str, Any]:
        """Upload an expense receipt."""
        return self._client.request("POST", "/api/v1/accounting/expenses/upload", json=expense_data)

    def get_pending_expenses(self) -> dict[str, Any]:
        """Get pending expenses."""
        return self._client.request("GET", "/api/v1/accounting/expenses/pending")

    def categorize_expenses(self, data: dict[str, Any]) -> dict[str, Any]:
        """Categorize expenses."""
        return self._client.request("POST", "/api/v1/accounting/expenses/categorize", json=data)

    def sync_expenses(self, data: dict[str, Any] | None = None) -> dict[str, Any]:
        """Sync expenses with accounting system."""
        return self._client.request("POST", "/api/v1/accounting/expenses/sync", json=data or {})

    def get_expense_stats(self) -> dict[str, Any]:
        """Get expense statistics."""
        return self._client.request("GET", "/api/v1/accounting/expenses/stats")

    def export_expenses(self, format: str = "csv") -> dict[str, Any]:
        """Export expenses."""
        return self._client.request("GET", "/api/v1/accounting/expenses/export", params={"format": format})

    # =========================================================================
    # AP Automation
    # =========================================================================

    def list_ap_invoices(
        self,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """List AP invoices."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        return self._client.request("GET", "/api/v1/accounting/ap/invoices", params=params)

    def optimize_ap(self, data: dict[str, Any] | None = None) -> dict[str, Any]:
        """Optimize AP payments."""
        return self._client.request("POST", "/api/v1/accounting/ap/optimize", json=data or {})

    def create_ap_batch(self, data: dict[str, Any]) -> dict[str, Any]:
        """Create AP batch payment."""
        return self._client.request("POST", "/api/v1/accounting/ap/batch", json=data)

    def get_ap_forecast(self, days: int = 30) -> dict[str, Any]:
        """Get AP cash flow forecast."""
        return self._client.request("GET", "/api/v1/accounting/ap/forecast", params={"days": days})

    def get_ap_discounts(self) -> dict[str, Any]:
        """Get AP early payment discounts."""
        return self._client.request("GET", "/api/v1/accounting/ap/discounts")

    # =========================================================================
    # Gusto Payroll Integration (direct routes)
    # =========================================================================

    def get_gusto_connect(self) -> dict[str, Any]:
        """Connect Gusto integration."""
        return self._client.request("POST", "/api/v1/gusto/connect")

    def get_gusto_disconnect(self) -> dict[str, Any]:
        """Disconnect Gusto integration."""
        return self._client.request("POST", "/api/v1/gusto/disconnect")

    def get_direct_gusto_status(self) -> dict[str, Any]:
        """Get Gusto status via direct route."""
        return self._client.request("GET", "/api/v1/gusto/status")

    def list_direct_gusto_employees(
        self,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """List Gusto employees via direct route."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        return self._client.request("GET", "/api/v1/gusto/employees", params=params)

    def list_direct_gusto_payrolls(
        self,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """List Gusto payrolls via direct route."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        return self._client.request("GET", "/api/v1/gusto/payrolls", params=params)

    # =========================================================================
    # Gusto Payroll Integration (namespaced routes)
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
    # Connection Management
    # =========================================================================

    async def get_status(self) -> dict[str, Any]:
        """Get accounting integration status."""
        return await self._client.request("GET", "/api/v1/accounting/status")

    async def connect(self) -> dict[str, Any]:
        """Connect accounting integration."""
        return await self._client.request("POST", "/api/v1/accounting/connect")

    async def disconnect(self) -> dict[str, Any]:
        """Disconnect accounting integration."""
        return await self._client.request("POST", "/api/v1/accounting/disconnect")

    # =========================================================================
    # Customers
    # =========================================================================

    async def list_customers(self, limit: int = 50, offset: int = 0) -> dict[str, Any]:
        """List accounting customers."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        return await self._client.request("GET", "/api/v1/accounting/customers", params=params)

    # =========================================================================
    # Transactions
    # =========================================================================

    async def list_transactions(
        self, limit: int = 50, offset: int = 0,
        start_date: str | None = None, end_date: str | None = None,
    ) -> dict[str, Any]:
        """List accounting transactions."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        return await self._client.request("GET", "/api/v1/accounting/transactions", params=params)

    # =========================================================================
    # Reports
    # =========================================================================

    async def generate_report(
        self, report_type: str, start_date: str | None = None, end_date: str | None = None,
    ) -> dict[str, Any]:
        """Generate an accounting report."""
        data: dict[str, Any] = {"report_type": report_type}
        if start_date:
            data["start_date"] = start_date
        if end_date:
            data["end_date"] = end_date
        return await self._client.request("POST", "/api/v1/accounting/reports", json=data)

    # =========================================================================
    # Invoices
    # =========================================================================

    async def list_invoices(self, limit: int = 50, offset: int = 0) -> dict[str, Any]:
        """List accounting invoices."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        return await self._client.request("GET", "/api/v1/accounting/invoices", params=params)

    async def upload_invoice(self, invoice_data: dict[str, Any]) -> dict[str, Any]:
        """Upload an invoice."""
        return await self._client.request("POST", "/api/v1/accounting/invoices/upload", json=invoice_data)

    async def get_pending_invoices(self) -> dict[str, Any]:
        """Get pending invoices."""
        return await self._client.request("GET", "/api/v1/accounting/invoices/pending")

    async def get_overdue_invoices(self) -> dict[str, Any]:
        """Get overdue invoices."""
        return await self._client.request("GET", "/api/v1/accounting/invoices/overdue")

    async def get_invoice_stats(self) -> dict[str, Any]:
        """Get invoice statistics."""
        return await self._client.request("GET", "/api/v1/accounting/invoices/stats")

    async def create_purchase_order(self, order: dict[str, Any]) -> dict[str, Any]:
        """Create a purchase order."""
        return await self._client.request("POST", "/api/v1/accounting/purchase-orders", json=order)

    async def get_scheduled_payments(self, limit: int = 50, offset: int = 0) -> dict[str, Any]:
        """Get scheduled payments."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        return await self._client.request("GET", "/api/v1/accounting/payments/scheduled", params=params)

    # =========================================================================
    # Expenses
    # =========================================================================

    async def list_expenses(self, limit: int = 50, offset: int = 0) -> dict[str, Any]:
        """List expenses."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        return await self._client.request("GET", "/api/v1/accounting/expenses", params=params)

    async def upload_expense(self, expense_data: dict[str, Any]) -> dict[str, Any]:
        """Upload an expense receipt."""
        return await self._client.request("POST", "/api/v1/accounting/expenses/upload", json=expense_data)

    async def get_pending_expenses(self) -> dict[str, Any]:
        """Get pending expenses."""
        return await self._client.request("GET", "/api/v1/accounting/expenses/pending")

    async def categorize_expenses(self, data: dict[str, Any]) -> dict[str, Any]:
        """Categorize expenses."""
        return await self._client.request("POST", "/api/v1/accounting/expenses/categorize", json=data)

    async def sync_expenses(self, data: dict[str, Any] | None = None) -> dict[str, Any]:
        """Sync expenses with accounting system."""
        return await self._client.request("POST", "/api/v1/accounting/expenses/sync", json=data or {})

    async def get_expense_stats(self) -> dict[str, Any]:
        """Get expense statistics."""
        return await self._client.request("GET", "/api/v1/accounting/expenses/stats")

    async def export_expenses(self, format: str = "csv") -> dict[str, Any]:
        """Export expenses."""
        return await self._client.request("GET", "/api/v1/accounting/expenses/export", params={"format": format})

    # =========================================================================
    # AP Automation
    # =========================================================================

    async def list_ap_invoices(self, limit: int = 50, offset: int = 0) -> dict[str, Any]:
        """List AP invoices."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        return await self._client.request("GET", "/api/v1/accounting/ap/invoices", params=params)

    async def optimize_ap(self, data: dict[str, Any] | None = None) -> dict[str, Any]:
        """Optimize AP payments."""
        return await self._client.request("POST", "/api/v1/accounting/ap/optimize", json=data or {})

    async def create_ap_batch(self, data: dict[str, Any]) -> dict[str, Any]:
        """Create AP batch payment."""
        return await self._client.request("POST", "/api/v1/accounting/ap/batch", json=data)

    async def get_ap_forecast(self, days: int = 30) -> dict[str, Any]:
        """Get AP cash flow forecast."""
        return await self._client.request("GET", "/api/v1/accounting/ap/forecast", params={"days": days})

    async def get_ap_discounts(self) -> dict[str, Any]:
        """Get AP early payment discounts."""
        return await self._client.request("GET", "/api/v1/accounting/ap/discounts")

    # =========================================================================
    # Gusto Payroll (direct routes)
    # =========================================================================

    async def get_gusto_connect(self) -> dict[str, Any]:
        """Connect Gusto integration."""
        return await self._client.request("POST", "/api/v1/gusto/connect")

    async def get_gusto_disconnect(self) -> dict[str, Any]:
        """Disconnect Gusto integration."""
        return await self._client.request("POST", "/api/v1/gusto/disconnect")

    async def get_direct_gusto_status(self) -> dict[str, Any]:
        """Get Gusto status via direct route."""
        return await self._client.request("GET", "/api/v1/gusto/status")

    async def list_direct_gusto_employees(self, limit: int = 50, offset: int = 0) -> dict[str, Any]:
        """List Gusto employees via direct route."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        return await self._client.request("GET", "/api/v1/gusto/employees", params=params)

    async def list_direct_gusto_payrolls(self, limit: int = 50, offset: int = 0) -> dict[str, Any]:
        """List Gusto payrolls via direct route."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        return await self._client.request("GET", "/api/v1/gusto/payrolls", params=params)

    # =========================================================================
    # Gusto Payroll Integration (namespaced routes)
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
