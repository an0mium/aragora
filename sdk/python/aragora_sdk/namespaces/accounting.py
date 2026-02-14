"""
Accounting Namespace API.

Provides a namespaced interface for QuickBooks Online and Gusto payroll integration.

Note: The accounting backend uses direct route registration (app.router.add_*)
rather than the ROUTES class-variable pattern. SDK methods will be re-added once
the handler is migrated to the standard ROUTES pattern for parity tracking.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class AccountingAPI:
    """Synchronous Accounting API."""

    def __init__(self, client: AragoraClient) -> None:
        self._client = client

    def delete_expenses(self) -> dict[str, Any]:
        """"""
        return self._client.request("DELETE", "/api/v1/accounting/expenses")

    def get_expenses(self) -> dict[str, Any]:
        """"""
        return self._client.request("GET", "/api/v1/accounting/expenses")

    def post_expenses(self) -> dict[str, Any]:
        """"""
        return self._client.request("POST", "/api/v1/accounting/expenses")

    def put_expenses(self) -> dict[str, Any]:
        """"""
        return self._client.request("PUT", "/api/v1/accounting/expenses")

    def delete_expenses_categorize(self) -> dict[str, Any]:
        """"""
        return self._client.request("DELETE", "/api/v1/accounting/expenses/categorize")

    def get_expenses_categorize(self) -> dict[str, Any]:
        """"""
        return self._client.request("GET", "/api/v1/accounting/expenses/categorize")

    def post_expenses_categorize(self) -> dict[str, Any]:
        """"""
        return self._client.request("POST", "/api/v1/accounting/expenses/categorize")

    def put_expenses_categorize(self) -> dict[str, Any]:
        """"""
        return self._client.request("PUT", "/api/v1/accounting/expenses/categorize")

    def delete_expenses_export(self) -> dict[str, Any]:
        """"""
        return self._client.request("DELETE", "/api/v1/accounting/expenses/export")

    def get_expenses_export(self) -> dict[str, Any]:
        """"""
        return self._client.request("GET", "/api/v1/accounting/expenses/export")

    def post_expenses_export(self) -> dict[str, Any]:
        """"""
        return self._client.request("POST", "/api/v1/accounting/expenses/export")

    def put_expenses_export(self) -> dict[str, Any]:
        """"""
        return self._client.request("PUT", "/api/v1/accounting/expenses/export")

    def delete_expenses_pending(self) -> dict[str, Any]:
        """"""
        return self._client.request("DELETE", "/api/v1/accounting/expenses/pending")

    def get_expenses_pending(self) -> dict[str, Any]:
        """"""
        return self._client.request("GET", "/api/v1/accounting/expenses/pending")

    def post_expenses_pending(self) -> dict[str, Any]:
        """"""
        return self._client.request("POST", "/api/v1/accounting/expenses/pending")

    def put_expenses_pending(self) -> dict[str, Any]:
        """"""
        return self._client.request("PUT", "/api/v1/accounting/expenses/pending")

    def delete_expenses_stats(self) -> dict[str, Any]:
        """"""
        return self._client.request("DELETE", "/api/v1/accounting/expenses/stats")

    def get_expenses_stats(self) -> dict[str, Any]:
        """"""
        return self._client.request("GET", "/api/v1/accounting/expenses/stats")

    def post_expenses_stats(self) -> dict[str, Any]:
        """"""
        return self._client.request("POST", "/api/v1/accounting/expenses/stats")

    def put_expenses_stats(self) -> dict[str, Any]:
        """"""
        return self._client.request("PUT", "/api/v1/accounting/expenses/stats")

    def delete_expenses_sync(self) -> dict[str, Any]:
        """"""
        return self._client.request("DELETE", "/api/v1/accounting/expenses/sync")

    def get_expenses_sync(self) -> dict[str, Any]:
        """"""
        return self._client.request("GET", "/api/v1/accounting/expenses/sync")

    def post_expenses_sync(self) -> dict[str, Any]:
        """"""
        return self._client.request("POST", "/api/v1/accounting/expenses/sync")

    def put_expenses_sync(self) -> dict[str, Any]:
        """"""
        return self._client.request("PUT", "/api/v1/accounting/expenses/sync")

    def delete_expenses_upload(self) -> dict[str, Any]:
        """"""
        return self._client.request("DELETE", "/api/v1/accounting/expenses/upload")

    def get_expenses_upload(self) -> dict[str, Any]:
        """"""
        return self._client.request("GET", "/api/v1/accounting/expenses/upload")

    def post_expenses_upload(self) -> dict[str, Any]:
        """"""
        return self._client.request("POST", "/api/v1/accounting/expenses/upload")

    def put_expenses_upload(self) -> dict[str, Any]:
        """"""
        return self._client.request("PUT", "/api/v1/accounting/expenses/upload")

    def get_invoices(self) -> dict[str, Any]:
        """"""
        return self._client.request("GET", "/api/v1/accounting/invoices")

    def post_invoices(self) -> dict[str, Any]:
        """"""
        return self._client.request("POST", "/api/v1/accounting/invoices")

    def get_invoices_overdue(self) -> dict[str, Any]:
        """"""
        return self._client.request("GET", "/api/v1/accounting/invoices/overdue")

    def post_invoices_overdue(self) -> dict[str, Any]:
        """"""
        return self._client.request("POST", "/api/v1/accounting/invoices/overdue")

    def get_invoices_pending(self) -> dict[str, Any]:
        """"""
        return self._client.request("GET", "/api/v1/accounting/invoices/pending")

    def post_invoices_pending(self) -> dict[str, Any]:
        """"""
        return self._client.request("POST", "/api/v1/accounting/invoices/pending")

    def get_invoices_stats(self) -> dict[str, Any]:
        """"""
        return self._client.request("GET", "/api/v1/accounting/invoices/stats")

    def post_invoices_stats(self) -> dict[str, Any]:
        """"""
        return self._client.request("POST", "/api/v1/accounting/invoices/stats")

    def get_invoices_upload(self) -> dict[str, Any]:
        """"""
        return self._client.request("GET", "/api/v1/accounting/invoices/upload")

    def post_invoices_upload(self) -> dict[str, Any]:
        """"""
        return self._client.request("POST", "/api/v1/accounting/invoices/upload")

    def get_payments_scheduled(self) -> dict[str, Any]:
        """"""
        return self._client.request("GET", "/api/v1/accounting/payments/scheduled")

    def get_purchase_orders(self) -> dict[str, Any]:
        """"""
        return self._client.request("GET", "/api/v1/accounting/purchase-orders")

    def post_payments_scheduled(self) -> dict[str, Any]:
        """"""
        return self._client.request("POST", "/api/v1/accounting/payments/scheduled")

    def post_purchase_orders(self) -> dict[str, Any]:
        """"""
        return self._client.request("POST", "/api/v1/accounting/purchase-orders")


class AsyncAccountingAPI:
    """Asynchronous Accounting API."""

    def __init__(self, client: AragoraAsyncClient) -> None:
        self._client = client

    async def delete_expenses(self) -> dict[str, Any]:
        """"""
        return await self._client.request("DELETE", "/api/v1/accounting/expenses")

    async def get_expenses(self) -> dict[str, Any]:
        """"""
        return await self._client.request("GET", "/api/v1/accounting/expenses")

    async def post_expenses(self) -> dict[str, Any]:
        """"""
        return await self._client.request("POST", "/api/v1/accounting/expenses")

    async def put_expenses(self) -> dict[str, Any]:
        """"""
        return await self._client.request("PUT", "/api/v1/accounting/expenses")

    async def delete_expenses_categorize(self) -> dict[str, Any]:
        """"""
        return await self._client.request("DELETE", "/api/v1/accounting/expenses/categorize")

    async def get_expenses_categorize(self) -> dict[str, Any]:
        """"""
        return await self._client.request("GET", "/api/v1/accounting/expenses/categorize")

    async def post_expenses_categorize(self) -> dict[str, Any]:
        """"""
        return await self._client.request("POST", "/api/v1/accounting/expenses/categorize")

    async def put_expenses_categorize(self) -> dict[str, Any]:
        """"""
        return await self._client.request("PUT", "/api/v1/accounting/expenses/categorize")

    async def delete_expenses_export(self) -> dict[str, Any]:
        """"""
        return await self._client.request("DELETE", "/api/v1/accounting/expenses/export")

    async def get_expenses_export(self) -> dict[str, Any]:
        """"""
        return await self._client.request("GET", "/api/v1/accounting/expenses/export")

    async def post_expenses_export(self) -> dict[str, Any]:
        """"""
        return await self._client.request("POST", "/api/v1/accounting/expenses/export")

    async def put_expenses_export(self) -> dict[str, Any]:
        """"""
        return await self._client.request("PUT", "/api/v1/accounting/expenses/export")

    async def delete_expenses_pending(self) -> dict[str, Any]:
        """"""
        return await self._client.request("DELETE", "/api/v1/accounting/expenses/pending")

    async def get_expenses_pending(self) -> dict[str, Any]:
        """"""
        return await self._client.request("GET", "/api/v1/accounting/expenses/pending")

    async def post_expenses_pending(self) -> dict[str, Any]:
        """"""
        return await self._client.request("POST", "/api/v1/accounting/expenses/pending")

    async def put_expenses_pending(self) -> dict[str, Any]:
        """"""
        return await self._client.request("PUT", "/api/v1/accounting/expenses/pending")

    async def delete_expenses_stats(self) -> dict[str, Any]:
        """"""
        return await self._client.request("DELETE", "/api/v1/accounting/expenses/stats")

    async def get_expenses_stats(self) -> dict[str, Any]:
        """"""
        return await self._client.request("GET", "/api/v1/accounting/expenses/stats")

    async def post_expenses_stats(self) -> dict[str, Any]:
        """"""
        return await self._client.request("POST", "/api/v1/accounting/expenses/stats")

    async def put_expenses_stats(self) -> dict[str, Any]:
        """"""
        return await self._client.request("PUT", "/api/v1/accounting/expenses/stats")

    async def delete_expenses_sync(self) -> dict[str, Any]:
        """"""
        return await self._client.request("DELETE", "/api/v1/accounting/expenses/sync")

    async def get_expenses_sync(self) -> dict[str, Any]:
        """"""
        return await self._client.request("GET", "/api/v1/accounting/expenses/sync")

    async def post_expenses_sync(self) -> dict[str, Any]:
        """"""
        return await self._client.request("POST", "/api/v1/accounting/expenses/sync")

    async def put_expenses_sync(self) -> dict[str, Any]:
        """"""
        return await self._client.request("PUT", "/api/v1/accounting/expenses/sync")

    async def delete_expenses_upload(self) -> dict[str, Any]:
        """"""
        return await self._client.request("DELETE", "/api/v1/accounting/expenses/upload")

    async def get_expenses_upload(self) -> dict[str, Any]:
        """"""
        return await self._client.request("GET", "/api/v1/accounting/expenses/upload")

    async def post_expenses_upload(self) -> dict[str, Any]:
        """"""
        return await self._client.request("POST", "/api/v1/accounting/expenses/upload")

    async def put_expenses_upload(self) -> dict[str, Any]:
        """"""
        return await self._client.request("PUT", "/api/v1/accounting/expenses/upload")

    async def get_invoices(self) -> dict[str, Any]:
        """"""
        return await self._client.request("GET", "/api/v1/accounting/invoices")

    async def post_invoices(self) -> dict[str, Any]:
        """"""
        return await self._client.request("POST", "/api/v1/accounting/invoices")

    async def get_invoices_overdue(self) -> dict[str, Any]:
        """"""
        return await self._client.request("GET", "/api/v1/accounting/invoices/overdue")

    async def post_invoices_overdue(self) -> dict[str, Any]:
        """"""
        return await self._client.request("POST", "/api/v1/accounting/invoices/overdue")

    async def get_invoices_pending(self) -> dict[str, Any]:
        """"""
        return await self._client.request("GET", "/api/v1/accounting/invoices/pending")

    async def post_invoices_pending(self) -> dict[str, Any]:
        """"""
        return await self._client.request("POST", "/api/v1/accounting/invoices/pending")

    async def get_invoices_stats(self) -> dict[str, Any]:
        """"""
        return await self._client.request("GET", "/api/v1/accounting/invoices/stats")

    async def post_invoices_stats(self) -> dict[str, Any]:
        """"""
        return await self._client.request("POST", "/api/v1/accounting/invoices/stats")

    async def get_invoices_upload(self) -> dict[str, Any]:
        """"""
        return await self._client.request("GET", "/api/v1/accounting/invoices/upload")

    async def post_invoices_upload(self) -> dict[str, Any]:
        """"""
        return await self._client.request("POST", "/api/v1/accounting/invoices/upload")

    async def get_payments_scheduled(self) -> dict[str, Any]:
        """"""
        return await self._client.request("GET", "/api/v1/accounting/payments/scheduled")

    async def get_purchase_orders(self) -> dict[str, Any]:
        """"""
        return await self._client.request("GET", "/api/v1/accounting/purchase-orders")

    async def post_payments_scheduled(self) -> dict[str, Any]:
        """"""
        return await self._client.request("POST", "/api/v1/accounting/payments/scheduled")

    async def post_purchase_orders(self) -> dict[str, Any]:
        """"""
        return await self._client.request("POST", "/api/v1/accounting/purchase-orders")
