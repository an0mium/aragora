"""
Expenses Namespace API.

Provides a namespaced interface for expense tracking and management.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class ExpensesAPI:
    """Synchronous Expenses API."""

    def __init__(self, client: AragoraClient):
        self._client = client

    def list_expenses(self) -> dict[str, Any]:
        """List expenses."""
        return self._client.request("GET", "/api/v1/accounting/expenses")

    def create_expense(self, **kwargs: Any) -> dict[str, Any]:
        """Create an expense."""
        return self._client.request("POST", "/api/v1/accounting/expenses", json=kwargs)

    def get_expense(self, expense_id: str) -> dict[str, Any]:
        """Get an expense by ID."""
        return self._client.request("GET", f"/api/v1/accounting/expenses/{expense_id}")

    def update_expense(self, expense_id: str, **kwargs: Any) -> dict[str, Any]:
        """Update an expense."""
        return self._client.request("PUT", f"/api/v1/accounting/expenses/{expense_id}", json=kwargs)

    def delete_expense(self, expense_id: str) -> dict[str, Any]:
        """Delete an expense."""
        return self._client.request("DELETE", f"/api/v1/accounting/expenses/{expense_id}")

    def approve(self, expense_id: str) -> dict[str, Any]:
        """Approve an expense."""
        return self._client.request("POST", f"/api/v1/accounting/expenses/{expense_id}/approve")

    def reject(self, expense_id: str, **kwargs: Any) -> dict[str, Any]:
        """Reject an expense."""
        return self._client.request(
            "POST", f"/api/v1/accounting/expenses/{expense_id}/reject", json=kwargs
        )

    def get_pending(self) -> dict[str, Any]:
        """Get pending expenses."""
        return self._client.request("GET", "/api/v1/accounting/expenses/pending")

    def get_stats(self) -> dict[str, Any]:
        """Get expense statistics."""
        return self._client.request("GET", "/api/v1/accounting/expenses/stats")

    def categorize(self, **kwargs: Any) -> dict[str, Any]:
        """Categorize expenses."""
        return self._client.request("POST", "/api/v1/accounting/expenses/categorize", json=kwargs)

    def export(self, **kwargs: Any) -> dict[str, Any]:
        """Export expenses."""
        return self._client.request("GET", "/api/v1/accounting/expenses/export")

    def sync(self, **kwargs: Any) -> dict[str, Any]:
        """Sync expenses with external system."""
        return self._client.request("POST", "/api/v1/accounting/expenses/sync", json=kwargs)

    def upload(self, **kwargs: Any) -> dict[str, Any]:
        """Upload expense receipts."""
        return self._client.request("POST", "/api/v1/accounting/expenses/upload", json=kwargs)


class AsyncExpensesAPI:
    """Asynchronous Expenses API."""

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def list_expenses(self) -> dict[str, Any]:
        """List expenses."""
        return await self._client.request("GET", "/api/v1/accounting/expenses")

    async def create_expense(self, **kwargs: Any) -> dict[str, Any]:
        """Create an expense."""
        return await self._client.request("POST", "/api/v1/accounting/expenses", json=kwargs)

    async def get_expense(self, expense_id: str) -> dict[str, Any]:
        """Get an expense by ID."""
        return await self._client.request("GET", f"/api/v1/accounting/expenses/{expense_id}")

    async def update_expense(self, expense_id: str, **kwargs: Any) -> dict[str, Any]:
        """Update an expense."""
        return await self._client.request(
            "PUT", f"/api/v1/accounting/expenses/{expense_id}", json=kwargs
        )

    async def delete_expense(self, expense_id: str) -> dict[str, Any]:
        """Delete an expense."""
        return await self._client.request("DELETE", f"/api/v1/accounting/expenses/{expense_id}")

    async def approve(self, expense_id: str) -> dict[str, Any]:
        """Approve an expense."""
        return await self._client.request(
            "POST", f"/api/v1/accounting/expenses/{expense_id}/approve"
        )

    async def reject(self, expense_id: str, **kwargs: Any) -> dict[str, Any]:
        """Reject an expense."""
        return await self._client.request(
            "POST", f"/api/v1/accounting/expenses/{expense_id}/reject", json=kwargs
        )

    async def get_pending(self) -> dict[str, Any]:
        """Get pending expenses."""
        return await self._client.request("GET", "/api/v1/accounting/expenses/pending")

    async def get_stats(self) -> dict[str, Any]:
        """Get expense statistics."""
        return await self._client.request("GET", "/api/v1/accounting/expenses/stats")

    async def categorize(self, **kwargs: Any) -> dict[str, Any]:
        """Categorize expenses."""
        return await self._client.request(
            "POST", "/api/v1/accounting/expenses/categorize", json=kwargs
        )

    async def export(self, **kwargs: Any) -> dict[str, Any]:
        """Export expenses."""
        return await self._client.request("GET", "/api/v1/accounting/expenses/export")

    async def sync(self, **kwargs: Any) -> dict[str, Any]:
        """Sync expenses with external system."""
        return await self._client.request("POST", "/api/v1/accounting/expenses/sync", json=kwargs)

    async def upload(self, **kwargs: Any) -> dict[str, Any]:
        """Upload expense receipts."""
        return await self._client.request("POST", "/api/v1/accounting/expenses/upload", json=kwargs)
