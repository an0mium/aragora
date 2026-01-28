"""
Accounting Namespace API.

Provides a namespaced interface for QuickBooks Online and Gusto payroll integration.
Enables transaction sync, customer management, and financial reporting.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient

ReportType = Literal["profit_loss", "balance_sheet", "ar_aging", "ap_aging"]
TransactionType = Literal["all", "invoice", "expense"]


class AccountingAPI:
    """
    Synchronous Accounting API.

    Provides methods for QuickBooks Online and Gusto payroll integration:
    - OAuth connection flows
    - Customer and transaction management
    - Financial report generation
    - Employee and payroll data
    """

    def __init__(self, client: AragoraClient) -> None:
        self._client = client

    # =========================================================================
    # QuickBooks Connection
    # =========================================================================

    def get_status(self) -> dict[str, Any]:
        """
        Get QuickBooks connection status and dashboard data.

        Returns:
            Connection status with company info, financial stats,
            and recent customers/transactions if connected.
        """
        return self._client._request("GET", "/api/v2/accounting/status")

    def connect(self) -> dict[str, Any]:
        """
        Initiate QuickBooks OAuth connection.

        Returns:
            URL to redirect user for OAuth authorization.
        """
        return self._client._request("POST", "/api/v2/accounting/connect")

    def disconnect(self) -> dict[str, Any]:
        """
        Disconnect QuickBooks integration.

        Returns:
            Success status and message.
        """
        return self._client._request("POST", "/api/v2/accounting/disconnect")

    # =========================================================================
    # Customers
    # =========================================================================

    def list_customers(
        self,
        active: bool | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """
        List QuickBooks customers.

        Args:
            active: Filter by active status.
            limit: Maximum results (default 50).
            offset: Pagination offset.

        Returns:
            List of customers with total count.
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if active is not None:
            params["active"] = active
        return self._client._request("GET", "/api/v2/accounting/customers", params=params)

    # =========================================================================
    # Transactions
    # =========================================================================

    def list_transactions(
        self,
        type: TransactionType | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """
        List transactions (invoices, expenses).

        Args:
            type: Filter by type (all, invoice, expense).
            start_date: Filter from date (ISO format).
            end_date: Filter to date (ISO format).
            limit: Maximum results.
            offset: Pagination offset.

        Returns:
            List of transactions with total count.
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if type:
            params["type"] = type
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        return self._client._request("GET", "/api/v2/accounting/transactions", params=params)

    # =========================================================================
    # Reports
    # =========================================================================

    def generate_report(
        self,
        type: ReportType,
        start_date: str,
        end_date: str,
    ) -> dict[str, Any]:
        """
        Generate a financial report.

        Args:
            type: Report type (profit_loss, balance_sheet, ar_aging, ap_aging).
            start_date: Report start date (ISO format).
            end_date: Report end date (ISO format).

        Returns:
            Financial report with sections and totals.
        """
        data = {
            "type": type,
            "start_date": start_date,
            "end_date": end_date,
        }
        return self._client._request("POST", "/api/v2/accounting/reports", json=data)

    # =========================================================================
    # Gusto Payroll
    # =========================================================================

    def get_gusto_status(self) -> dict[str, Any]:
        """
        Get Gusto connection status.

        Returns:
            Connection status and company name if connected.
        """
        return self._client._request("GET", "/api/v2/gusto/status")

    def connect_gusto(self) -> dict[str, Any]:
        """
        Initiate Gusto OAuth connection.

        Returns:
            URL to redirect user for OAuth authorization.
        """
        return self._client._request("POST", "/api/v2/gusto/connect")

    def disconnect_gusto(self) -> dict[str, Any]:
        """
        Disconnect Gusto integration.

        Returns:
            Success status and message.
        """
        return self._client._request("POST", "/api/v2/gusto/disconnect")

    def list_employees(
        self,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """
        List Gusto employees.

        Args:
            limit: Maximum results.
            offset: Pagination offset.

        Returns:
            List of employees with total count.
        """
        params = {"limit": limit, "offset": offset}
        return self._client._request("GET", "/api/v2/gusto/employees", params=params)

    def list_payrolls(
        self,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """
        List payroll runs.

        Args:
            limit: Maximum results.
            offset: Pagination offset.

        Returns:
            List of payroll runs with total count.
        """
        params = {"limit": limit, "offset": offset}
        return self._client._request("GET", "/api/v2/gusto/payrolls", params=params)

    def get_payroll(self, payroll_id: str) -> dict[str, Any]:
        """
        Get payroll run details.

        Args:
            payroll_id: Payroll run ID.

        Returns:
            Detailed payroll information including employee compensations.
        """
        return self._client._request("GET", f"/api/v2/gusto/payrolls/{payroll_id}")

    def generate_journal_entry(self, payroll_id: str) -> dict[str, Any]:
        """
        Generate journal entry for a payroll run.

        Creates a journal entry that can be imported into QuickBooks
        or other accounting software.

        Args:
            payroll_id: Payroll run ID.

        Returns:
            Journal entry with debit/credit lines.
        """
        return self._client._request("POST", f"/api/v2/gusto/payrolls/{payroll_id}/journal-entry")


class AsyncAccountingAPI:
    """
    Asynchronous Accounting API.

    Provides async methods for QuickBooks Online and Gusto payroll integration.
    """

    def __init__(self, client: AragoraAsyncClient) -> None:
        self._client = client

    # =========================================================================
    # QuickBooks Connection
    # =========================================================================

    async def get_status(self) -> dict[str, Any]:
        """Get QuickBooks connection status and dashboard data."""
        return await self._client._request("GET", "/api/v2/accounting/status")

    async def connect(self) -> dict[str, Any]:
        """Initiate QuickBooks OAuth connection."""
        return await self._client._request("POST", "/api/v2/accounting/connect")

    async def disconnect(self) -> dict[str, Any]:
        """Disconnect QuickBooks integration."""
        return await self._client._request("POST", "/api/v2/accounting/disconnect")

    # =========================================================================
    # Customers
    # =========================================================================

    async def list_customers(
        self,
        active: bool | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """List QuickBooks customers."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if active is not None:
            params["active"] = active
        return await self._client._request("GET", "/api/v2/accounting/customers", params=params)

    # =========================================================================
    # Transactions
    # =========================================================================

    async def list_transactions(
        self,
        type: TransactionType | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """List transactions (invoices, expenses)."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if type:
            params["type"] = type
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        return await self._client._request("GET", "/api/v2/accounting/transactions", params=params)

    # =========================================================================
    # Reports
    # =========================================================================

    async def generate_report(
        self,
        type: ReportType,
        start_date: str,
        end_date: str,
    ) -> dict[str, Any]:
        """Generate a financial report."""
        data = {
            "type": type,
            "start_date": start_date,
            "end_date": end_date,
        }
        return await self._client._request("POST", "/api/v2/accounting/reports", json=data)

    # =========================================================================
    # Gusto Payroll
    # =========================================================================

    async def get_gusto_status(self) -> dict[str, Any]:
        """Get Gusto connection status."""
        return await self._client._request("GET", "/api/v2/gusto/status")

    async def connect_gusto(self) -> dict[str, Any]:
        """Initiate Gusto OAuth connection."""
        return await self._client._request("POST", "/api/v2/gusto/connect")

    async def disconnect_gusto(self) -> dict[str, Any]:
        """Disconnect Gusto integration."""
        return await self._client._request("POST", "/api/v2/gusto/disconnect")

    async def list_employees(
        self,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """List Gusto employees."""
        params = {"limit": limit, "offset": offset}
        return await self._client._request("GET", "/api/v2/gusto/employees", params=params)

    async def list_payrolls(
        self,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """List payroll runs."""
        params = {"limit": limit, "offset": offset}
        return await self._client._request("GET", "/api/v2/gusto/payrolls", params=params)

    async def get_payroll(self, payroll_id: str) -> dict[str, Any]:
        """Get payroll run details."""
        return await self._client._request("GET", f"/api/v2/gusto/payrolls/{payroll_id}")

    async def generate_journal_entry(self, payroll_id: str) -> dict[str, Any]:
        """Generate journal entry for a payroll run."""
        return await self._client._request(
            "POST", f"/api/v2/gusto/payrolls/{payroll_id}/journal-entry"
        )
