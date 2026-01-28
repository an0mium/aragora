"""
Expenses Namespace API.

Provides a namespaced interface for expense tracking and management.
Supports receipt processing, auto-categorization, approval workflows, and QBO sync.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient

ExpenseCategory = Literal[
    "travel",
    "meals",
    "office_supplies",
    "software",
    "equipment",
    "professional_services",
    "marketing",
    "utilities",
    "rent",
    "insurance",
    "other",
]
ExpenseStatus = Literal["pending", "approved", "rejected", "synced"]
PaymentMethod = Literal["credit_card", "debit_card", "cash", "check", "bank_transfer", "other"]


class ExpensesAPI:
    """
    Synchronous Expenses API.

    Provides methods for:
    - Receipt upload and OCR processing
    - Expense creation and management
    - Approval workflows
    - Auto-categorization
    - QuickBooks Online sync
    """

    def __init__(self, client: AragoraClient) -> None:
        self._client = client

    # =========================================================================
    # Receipt Processing
    # =========================================================================

    def upload_receipt(
        self,
        receipt_data: str,
        content_type: str | None = None,
        employee_id: str | None = None,
        payment_method: PaymentMethod | None = None,
    ) -> dict[str, Any]:
        """
        Upload and process a receipt image.

        Extracts vendor, amount, date, and category from the receipt
        using OCR and AI analysis.

        Args:
            receipt_data: Base64 encoded receipt image.
            content_type: MIME type (image/png, image/jpeg, application/pdf).
            employee_id: Employee identifier.
            payment_method: Payment method used.

        Returns:
            Created expense from receipt.
        """
        data: dict[str, Any] = {"receipt_data": receipt_data}
        if content_type:
            data["content_type"] = content_type
        if employee_id:
            data["employee_id"] = employee_id
        if payment_method:
            data["payment_method"] = payment_method

        return self._client.request("POST", "/api/v1/accounting/expenses/upload", json=data)

    # =========================================================================
    # CRUD Operations
    # =========================================================================

    def create(
        self,
        vendor_name: str,
        amount: float,
        date: str | None = None,
        category: ExpenseCategory | None = None,
        payment_method: PaymentMethod | None = None,
        description: str | None = None,
        employee_id: str | None = None,
        is_reimbursable: bool = False,
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Create an expense manually.

        Args:
            vendor_name: Vendor name.
            amount: Expense amount.
            date: Expense date (ISO format).
            category: Expense category.
            payment_method: Payment method used.
            description: Expense description.
            employee_id: Employee identifier.
            is_reimbursable: Whether expense is reimbursable.
            tags: Optional tags.

        Returns:
            Created expense.
        """
        data: dict[str, Any] = {"vendor_name": vendor_name, "amount": amount}
        if date:
            data["date"] = date
        if category:
            data["category"] = category
        if payment_method:
            data["payment_method"] = payment_method
        if description:
            data["description"] = description
        if employee_id:
            data["employee_id"] = employee_id
        if is_reimbursable:
            data["is_reimbursable"] = is_reimbursable
        if tags:
            data["tags"] = tags

        return self._client.request("POST", "/api/v1/accounting/expenses", json=data)

    def list(
        self,
        category: ExpenseCategory | None = None,
        vendor: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        status: ExpenseStatus | None = None,
        employee_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """
        List expenses with filters.

        Args:
            category: Filter by category.
            vendor: Filter by vendor.
            start_date: Filter from date.
            end_date: Filter to date.
            status: Filter by status.
            employee_id: Filter by employee.
            limit: Maximum results.
            offset: Pagination offset.

        Returns:
            Paginated expense list.
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if category:
            params["category"] = category
        if vendor:
            params["vendor"] = vendor
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        if status:
            params["status"] = status
        if employee_id:
            params["employee_id"] = employee_id

        return self._client.request("GET", "/api/v1/accounting/expenses", params=params)

    def get(self, expense_id: str) -> dict[str, Any]:
        """
        Get expense by ID.

        Args:
            expense_id: Expense identifier.

        Returns:
            Expense details.
        """
        return self._client.request("GET", f"/api/v1/accounting/expenses/{expense_id}")

    def update(
        self,
        expense_id: str,
        vendor_name: str | None = None,
        amount: float | None = None,
        category: ExpenseCategory | None = None,
        description: str | None = None,
        status: ExpenseStatus | None = None,
        is_reimbursable: bool | None = None,
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Update an expense.

        Args:
            expense_id: Expense identifier.
            vendor_name: New vendor name.
            amount: New amount.
            category: New category.
            description: New description.
            status: New status.
            is_reimbursable: Update reimbursable flag.
            tags: New tags.

        Returns:
            Updated expense.
        """
        data: dict[str, Any] = {}
        if vendor_name:
            data["vendor_name"] = vendor_name
        if amount is not None:
            data["amount"] = amount
        if category:
            data["category"] = category
        if description:
            data["description"] = description
        if status:
            data["status"] = status
        if is_reimbursable is not None:
            data["is_reimbursable"] = is_reimbursable
        if tags is not None:
            data["tags"] = tags

        return self._client.request("PUT", f"/api/v1/accounting/expenses/{expense_id}", json=data)

    def delete(self, expense_id: str) -> dict[str, Any]:
        """
        Delete an expense.

        Args:
            expense_id: Expense identifier.

        Returns:
            Confirmation message.
        """
        return self._client.request("DELETE", f"/api/v1/accounting/expenses/{expense_id}")

    # =========================================================================
    # Approval Workflow
    # =========================================================================

    def approve(self, expense_id: str) -> dict[str, Any]:
        """
        Approve an expense for sync.

        Args:
            expense_id: Expense identifier.

        Returns:
            Approved expense.
        """
        return self._client.request("POST", f"/api/v1/accounting/expenses/{expense_id}/approve")

    def reject(
        self,
        expense_id: str,
        reason: str | None = None,
    ) -> dict[str, Any]:
        """
        Reject an expense.

        Args:
            expense_id: Expense identifier.
            reason: Optional rejection reason.

        Returns:
            Rejected expense.
        """
        data: dict[str, Any] = {}
        if reason:
            data["reason"] = reason

        return self._client.request(
            "POST", f"/api/v1/accounting/expenses/{expense_id}/reject", json=data
        )

    def get_pending(self) -> dict[str, Any]:
        """
        Get expenses pending approval.

        Returns:
            Pending expenses with count.
        """
        return self._client.request("GET", "/api/v1/accounting/expenses/pending")

    # =========================================================================
    # Categorization
    # =========================================================================

    def categorize(
        self,
        expense_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Auto-categorize expenses using AI.

        Args:
            expense_ids: Optional list of expense IDs (categorizes all uncategorized if empty).

        Returns:
            Categorization results.
        """
        data: dict[str, Any] = {}
        if expense_ids:
            data["expense_ids"] = expense_ids

        return self._client.request("POST", "/api/v1/accounting/expenses/categorize", json=data)

    # =========================================================================
    # QBO Sync
    # =========================================================================

    def sync_to_qbo(
        self,
        expense_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Sync expenses to QuickBooks Online.

        Args:
            expense_ids: Optional list of expense IDs (syncs all approved if empty).

        Returns:
            Sync results.
        """
        data: dict[str, Any] = {}
        if expense_ids:
            data["expense_ids"] = expense_ids

        return self._client.request("POST", "/api/v1/accounting/expenses/sync", json=data)

    # =========================================================================
    # Statistics and Export
    # =========================================================================

    def get_stats(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict[str, Any]:
        """
        Get expense statistics.

        Args:
            start_date: Optional start date filter.
            end_date: Optional end date filter.

        Returns:
            Expense statistics.
        """
        params: dict[str, Any] = {}
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date

        return self._client.request("GET", "/api/v1/accounting/expenses/stats", params=params)

    def export(
        self,
        format: str = "csv",
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict[str, Any]:
        """
        Export expenses to CSV or JSON.

        Args:
            format: Export format ('csv' or 'json').
            start_date: Optional start date filter.
            end_date: Optional end date filter.

        Returns:
            Exported data.
        """
        params: dict[str, Any] = {"format": format}
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date

        return self._client.request("GET", "/api/v1/accounting/expenses/export", params=params)


class AsyncExpensesAPI:
    """Asynchronous Expenses API."""

    def __init__(self, client: AragoraAsyncClient) -> None:
        self._client = client

    # =========================================================================
    # Receipt Processing
    # =========================================================================

    async def upload_receipt(
        self,
        receipt_data: str,
        content_type: str | None = None,
        employee_id: str | None = None,
        payment_method: PaymentMethod | None = None,
    ) -> dict[str, Any]:
        """Upload and process a receipt image."""
        data: dict[str, Any] = {"receipt_data": receipt_data}
        if content_type:
            data["content_type"] = content_type
        if employee_id:
            data["employee_id"] = employee_id
        if payment_method:
            data["payment_method"] = payment_method

        return await self._client.request("POST", "/api/v1/accounting/expenses/upload", json=data)

    # =========================================================================
    # CRUD Operations
    # =========================================================================

    async def create(
        self,
        vendor_name: str,
        amount: float,
        date: str | None = None,
        category: ExpenseCategory | None = None,
        payment_method: PaymentMethod | None = None,
        description: str | None = None,
        employee_id: str | None = None,
        is_reimbursable: bool = False,
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """Create an expense manually."""
        data: dict[str, Any] = {"vendor_name": vendor_name, "amount": amount}
        if date:
            data["date"] = date
        if category:
            data["category"] = category
        if payment_method:
            data["payment_method"] = payment_method
        if description:
            data["description"] = description
        if employee_id:
            data["employee_id"] = employee_id
        if is_reimbursable:
            data["is_reimbursable"] = is_reimbursable
        if tags:
            data["tags"] = tags

        return await self._client.request("POST", "/api/v1/accounting/expenses", json=data)

    async def list(
        self,
        category: ExpenseCategory | None = None,
        vendor: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        status: ExpenseStatus | None = None,
        employee_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """List expenses with filters."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if category:
            params["category"] = category
        if vendor:
            params["vendor"] = vendor
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        if status:
            params["status"] = status
        if employee_id:
            params["employee_id"] = employee_id

        return await self._client.request("GET", "/api/v1/accounting/expenses", params=params)

    async def get(self, expense_id: str) -> dict[str, Any]:
        """Get expense by ID."""
        return await self._client.request("GET", f"/api/v1/accounting/expenses/{expense_id}")

    async def update(
        self,
        expense_id: str,
        vendor_name: str | None = None,
        amount: float | None = None,
        category: ExpenseCategory | None = None,
        description: str | None = None,
        status: ExpenseStatus | None = None,
        is_reimbursable: bool | None = None,
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """Update an expense."""
        data: dict[str, Any] = {}
        if vendor_name:
            data["vendor_name"] = vendor_name
        if amount is not None:
            data["amount"] = amount
        if category:
            data["category"] = category
        if description:
            data["description"] = description
        if status:
            data["status"] = status
        if is_reimbursable is not None:
            data["is_reimbursable"] = is_reimbursable
        if tags is not None:
            data["tags"] = tags

        return await self._client.request(
            "PUT", f"/api/v1/accounting/expenses/{expense_id}", json=data
        )

    async def delete(self, expense_id: str) -> dict[str, Any]:
        """Delete an expense."""
        return await self._client.request("DELETE", f"/api/v1/accounting/expenses/{expense_id}")

    # =========================================================================
    # Approval Workflow
    # =========================================================================

    async def approve(self, expense_id: str) -> dict[str, Any]:
        """Approve an expense for sync."""
        return await self._client.request(
            "POST", f"/api/v1/accounting/expenses/{expense_id}/approve"
        )

    async def reject(
        self,
        expense_id: str,
        reason: str | None = None,
    ) -> dict[str, Any]:
        """Reject an expense."""
        data: dict[str, Any] = {}
        if reason:
            data["reason"] = reason

        return await self._client.request(
            "POST", f"/api/v1/accounting/expenses/{expense_id}/reject", json=data
        )

    async def get_pending(self) -> dict[str, Any]:
        """Get expenses pending approval."""
        return await self._client.request("GET", "/api/v1/accounting/expenses/pending")

    # =========================================================================
    # Categorization
    # =========================================================================

    async def categorize(
        self,
        expense_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """Auto-categorize expenses using AI."""
        data: dict[str, Any] = {}
        if expense_ids:
            data["expense_ids"] = expense_ids

        return await self._client.request(
            "POST", "/api/v1/accounting/expenses/categorize", json=data
        )

    # =========================================================================
    # QBO Sync
    # =========================================================================

    async def sync_to_qbo(
        self,
        expense_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """Sync expenses to QuickBooks Online."""
        data: dict[str, Any] = {}
        if expense_ids:
            data["expense_ids"] = expense_ids

        return await self._client.request("POST", "/api/v1/accounting/expenses/sync", json=data)

    # =========================================================================
    # Statistics and Export
    # =========================================================================

    async def get_stats(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict[str, Any]:
        """Get expense statistics."""
        params: dict[str, Any] = {}
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date

        return await self._client.request("GET", "/api/v1/accounting/expenses/stats", params=params)

    async def export(
        self,
        format: str = "csv",
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict[str, Any]:
        """Export expenses to CSV or JSON."""
        params: dict[str, Any] = {"format": format}
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date

        return await self._client.request(
            "GET", "/api/v1/accounting/expenses/export", params=params
        )
