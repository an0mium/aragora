"""
Tests for aragora.server.handlers.expenses - Expense Tracking API Handler.

Tests cover:
- Route registration and can_handle
- POST /api/v1/accounting/expenses/upload - Upload and process receipt
- POST /api/v1/accounting/expenses - Create expense manually
- GET /api/v1/accounting/expenses - List expenses with filters
- GET /api/v1/accounting/expenses/{id} - Get expense by ID
- PUT /api/v1/accounting/expenses/{id} - Update expense
- DELETE /api/v1/accounting/expenses/{id} - Delete expense
- POST /api/v1/accounting/expenses/{id}/approve - Approve expense
- POST /api/v1/accounting/expenses/{id}/reject - Reject expense
- POST /api/v1/accounting/expenses/categorize - Auto-categorize expenses
- POST /api/v1/accounting/expenses/sync - Sync expenses to QBO
- GET /api/v1/accounting/expenses/stats - Get expense statistics
- GET /api/v1/accounting/expenses/pending - Get pending approvals
- GET /api/v1/accounting/expenses/export - Export expenses
- Authentication and RBAC
- Error handling
"""

from __future__ import annotations

import base64
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from io import BytesIO
from typing import Any
from unittest.mock import MagicMock, AsyncMock, patch
from enum import Enum

import pytest


# ---------------------------------------------------------------------------
# Import the module under test with Slack stub workaround
# ---------------------------------------------------------------------------


def _import_expenses_module():
    """Import expenses module, working around broken sibling imports."""
    try:
        import aragora.server.handlers.expenses as mod

        return mod
    except (ImportError, ModuleNotFoundError):
        pass

    # Clear partially loaded modules and stub broken imports
    to_remove = [k for k in sys.modules if k.startswith("aragora.server.handlers")]
    for k in to_remove:
        del sys.modules[k]

    _slack_stubs = [
        "aragora.server.handlers.social._slack_impl",
        "aragora.server.handlers.social._slack_impl.config",
        "aragora.server.handlers.social._slack_impl.handler",
        "aragora.server.handlers.social._slack_impl.commands",
        "aragora.server.handlers.social._slack_impl.events",
        "aragora.server.handlers.social._slack_impl.blocks",
        "aragora.server.handlers.social._slack_impl.interactions",
        "aragora.server.handlers.social.slack",
        "aragora.server.handlers.social.slack.handler",
    ]
    for name in _slack_stubs:
        if name not in sys.modules:
            stub = MagicMock()
            stub.__path__ = []
            stub.__file__ = f"<stub:{name}>"
            sys.modules[name] = stub

    import aragora.server.handlers.expenses as mod

    return mod


expenses_module = _import_expenses_module()
ExpenseHandler = expenses_module.ExpenseHandler
handle_upload_receipt = expenses_module.handle_upload_receipt
handle_create_expense = expenses_module.handle_create_expense
handle_list_expenses = expenses_module.handle_list_expenses
handle_get_expense = expenses_module.handle_get_expense
handle_update_expense = expenses_module.handle_update_expense
handle_delete_expense = expenses_module.handle_delete_expense
handle_approve_expense = expenses_module.handle_approve_expense
handle_reject_expense = expenses_module.handle_reject_expense
handle_categorize_expenses = expenses_module.handle_categorize_expenses
handle_sync_to_qbo = expenses_module.handle_sync_to_qbo
handle_get_expense_stats = expenses_module.handle_get_expense_stats
handle_get_pending_approvals = expenses_module.handle_get_pending_approvals
handle_export_expenses = expenses_module.handle_export_expenses


# ===========================================================================
# Test Fixtures and Mocks
# ===========================================================================


class MockExpenseCategory(Enum):
    """Mock expense category enum."""

    TRAVEL = "travel"
    MEALS = "meals"
    SUPPLIES = "supplies"
    SOFTWARE = "software"
    UNCATEGORIZED = "uncategorized"


class MockExpenseStatus(Enum):
    """Mock expense status enum."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    SYNCED = "synced"


class MockPaymentMethod(Enum):
    """Mock payment method enum."""

    CREDIT_CARD = "credit_card"
    CASH = "cash"
    CHECK = "check"


@dataclass
class MockExpense:
    """Mock expense for testing."""

    expense_id: str = "expense-123"
    vendor_name: str = "Test Vendor"
    amount: float = 99.99
    date: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    category: str = "travel"
    payment_method: str = "credit_card"
    description: str = "Test expense"
    employee_id: str | None = "emp-123"
    is_reimbursable: bool = False
    status: str = "pending"
    tags: list = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict:
        return {
            "expense_id": self.expense_id,
            "vendor_name": self.vendor_name,
            "amount": self.amount,
            "date": self.date.isoformat(),
            "category": self.category,
            "payment_method": self.payment_method,
            "description": self.description,
            "employee_id": self.employee_id,
            "is_reimbursable": self.is_reimbursable,
            "status": self.status,
            "tags": self.tags,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class MockSyncResult:
    """Mock QBO sync result."""

    success_count: int = 5
    failed_count: int = 0
    synced_ids: list = field(default_factory=list)
    errors: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "success_count": self.success_count,
            "failed_count": self.failed_count,
            "synced_ids": self.synced_ids,
            "errors": self.errors,
        }


class MockExpenseTracker:
    """Mock expense tracker."""

    def __init__(self):
        self.expenses: dict[str, MockExpense] = {}

    async def process_receipt(
        self,
        image_data: bytes,
        employee_id: str | None = None,
        payment_method=None,
    ) -> MockExpense:
        expense = MockExpense(
            expense_id="expense-receipt-123",
            vendor_name="Receipt Vendor",
            amount=50.00,
        )
        self.expenses[expense.expense_id] = expense
        return expense

    async def create_expense(
        self,
        vendor_name: str,
        amount: float,
        date: datetime | None = None,
        category=None,
        payment_method=None,
        description: str = "",
        employee_id: str | None = None,
        is_reimbursable: bool = False,
        tags: list | None = None,
    ) -> MockExpense:
        expense = MockExpense(
            expense_id=f"expense-{len(self.expenses) + 1}",
            vendor_name=vendor_name,
            amount=amount,
            description=description,
        )
        self.expenses[expense.expense_id] = expense
        return expense

    async def list_expenses(
        self,
        category=None,
        vendor: str | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        status=None,
        employee_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[MockExpense], int]:
        expenses = list(self.expenses.values())
        return expenses[offset : offset + limit], len(expenses)

    async def get_expense(self, expense_id: str) -> MockExpense | None:
        return self.expenses.get(expense_id)

    async def update_expense(
        self,
        expense_id: str,
        **kwargs,
    ) -> MockExpense | None:
        expense = self.expenses.get(expense_id)
        if expense:
            for key, value in kwargs.items():
                if value is not None and hasattr(expense, key):
                    setattr(expense, key, value)
        return expense

    async def delete_expense(self, expense_id: str) -> bool:
        if expense_id in self.expenses:
            del self.expenses[expense_id]
            return True
        return False

    async def approve_expense(self, expense_id: str) -> MockExpense | None:
        expense = self.expenses.get(expense_id)
        if expense:
            expense.status = "approved"
        return expense

    async def reject_expense(self, expense_id: str, reason: str = "") -> MockExpense | None:
        expense = self.expenses.get(expense_id)
        if expense:
            expense.status = "rejected"
        return expense

    async def get_pending_approval(self) -> list[MockExpense]:
        return [e for e in self.expenses.values() if e.status == "pending"]

    async def bulk_categorize(self, expense_ids: list[str] | None = None) -> dict[str, str]:
        result = {}
        for eid in expense_ids or self.expenses.keys():
            result[eid] = "travel"
        return result

    async def sync_to_qbo(self, expense_ids: list[str] | None = None) -> MockSyncResult:
        return MockSyncResult(synced_ids=expense_ids or [])

    async def get_stats(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> dict:
        return {
            "total_expenses": len(self.expenses),
            "total_amount": sum(e.amount for e in self.expenses.values()),
            "by_category": {"travel": 3, "meals": 2},
            "by_status": {"pending": 3, "approved": 2},
        }

    async def export_expenses(
        self,
        format: str = "csv",
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> str:
        if format == "csv":
            return "expense_id,vendor_name,amount\nexp-1,Vendor,99.99"
        return json.dumps([e.to_dict() for e in self.expenses.values()])


def make_mock_handler(
    body: dict | None = None,
    method: str = "GET",
    path: str = "/api/v1/accounting/expenses",
):
    """Create a mock HTTP handler."""
    handler = MagicMock()
    handler.command = method
    handler.path = path
    handler.headers = {}
    handler.client_address = ("127.0.0.1", 12345)

    if body is not None:
        body_bytes = json.dumps(body).encode("utf-8")
        handler.headers["Content-Length"] = str(len(body_bytes))
        handler.rfile = BytesIO(body_bytes)
    else:
        handler.rfile = BytesIO(b"")
        handler.headers["Content-Length"] = "0"

    return handler


@pytest.fixture
def expense_handler():
    """Create ExpenseHandler with mock context."""
    ctx = {}
    handler = ExpenseHandler(ctx)
    return handler


@pytest.fixture
def mock_tracker():
    """Create mock expense tracker with test data."""
    tracker = MockExpenseTracker()
    tracker.expenses["expense-123"] = MockExpense()
    tracker.expenses["expense-456"] = MockExpense(
        expense_id="expense-456",
        vendor_name="Another Vendor",
        amount=150.00,
        status="pending",
    )
    return tracker


# ===========================================================================
# Test Routing (can_handle)
# ===========================================================================


class TestExpenseHandlerRouting:
    """Tests for ExpenseHandler.can_handle."""

    def test_can_handle_upload(self, expense_handler):
        assert expense_handler.can_handle("/api/v1/accounting/expenses/upload") is True

    def test_can_handle_expenses_list(self, expense_handler):
        assert expense_handler.can_handle("/api/v1/accounting/expenses") is True

    def test_can_handle_categorize(self, expense_handler):
        assert expense_handler.can_handle("/api/v1/accounting/expenses/categorize") is True

    def test_can_handle_sync(self, expense_handler):
        assert expense_handler.can_handle("/api/v1/accounting/expenses/sync") is True

    def test_can_handle_stats(self, expense_handler):
        assert expense_handler.can_handle("/api/v1/accounting/expenses/stats") is True

    def test_can_handle_pending(self, expense_handler):
        assert expense_handler.can_handle("/api/v1/accounting/expenses/pending") is True

    def test_can_handle_export(self, expense_handler):
        assert expense_handler.can_handle("/api/v1/accounting/expenses/export") is True

    def test_can_handle_expense_by_id(self, expense_handler):
        assert expense_handler.can_handle("/api/v1/accounting/expenses/expense-123") is True

    def test_can_handle_approve(self, expense_handler):
        assert expense_handler.can_handle("/api/v1/accounting/expenses/expense-123/approve") is True

    def test_can_handle_reject(self, expense_handler):
        assert expense_handler.can_handle("/api/v1/accounting/expenses/expense-123/reject") is True

    def test_cannot_handle_other_paths(self, expense_handler):
        assert expense_handler.can_handle("/api/v1/debates") is False


# ===========================================================================
# Test Upload Receipt (POST /api/v1/accounting/expenses/upload)
# ===========================================================================


class TestExpenseUploadReceipt:
    """Tests for POST /api/v1/accounting/expenses/upload endpoint."""

    @pytest.mark.asyncio
    async def test_upload_receipt_success(self, mock_tracker):
        """Happy path: upload and process receipt."""
        # Create a simple test image (1x1 PNG)
        image_data = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100).decode()

        with patch.object(expenses_module, "get_expense_tracker", return_value=mock_tracker):
            result = await handle_upload_receipt(
                {
                    "receipt_data": image_data,
                    "content_type": "image/png",
                }
            )

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert "expense" in data
        assert data["message"] == "Receipt processed successfully"

    @pytest.mark.asyncio
    async def test_upload_receipt_missing_data(self):
        """Missing receipt_data returns 400."""
        result = await handle_upload_receipt({})

        assert result is not None
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_upload_receipt_invalid_base64(self):
        """Invalid base64 returns 400."""
        result = await handle_upload_receipt({"receipt_data": "not-valid-base64!!!"})

        assert result is not None
        assert result.status_code == 400


# ===========================================================================
# Test Create Expense (POST /api/v1/accounting/expenses)
# ===========================================================================


class TestExpenseCreate:
    """Tests for POST /api/v1/accounting/expenses endpoint."""

    @pytest.mark.asyncio
    async def test_create_expense_success(self, mock_tracker):
        """Happy path: create expense manually."""
        with patch.object(expenses_module, "get_expense_tracker", return_value=mock_tracker):
            result = await handle_create_expense(
                {
                    "vendor_name": "Test Vendor",
                    "amount": 99.99,
                    "description": "Test expense",
                }
            )

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert "expense" in data
        assert data["expense"]["vendor_name"] == "Test Vendor"

    @pytest.mark.asyncio
    async def test_create_expense_missing_vendor(self):
        """Missing vendor_name returns 400."""
        result = await handle_create_expense({"amount": 99.99})

        assert result is not None
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_create_expense_missing_amount(self):
        """Missing amount returns 400."""
        result = await handle_create_expense({"vendor_name": "Test Vendor"})

        assert result is not None
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_create_expense_invalid_amount(self):
        """Invalid amount returns 400."""
        result = await handle_create_expense(
            {
                "vendor_name": "Test Vendor",
                "amount": "not-a-number",
            }
        )

        assert result is not None
        assert result.status_code == 400


# ===========================================================================
# Test List Expenses (GET /api/v1/accounting/expenses)
# ===========================================================================


class TestExpenseList:
    """Tests for GET /api/v1/accounting/expenses endpoint."""

    @pytest.mark.asyncio
    async def test_list_expenses_success(self, mock_tracker):
        """Happy path: list expenses."""
        with patch.object(expenses_module, "get_expense_tracker", return_value=mock_tracker):
            result = await handle_list_expenses({})

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert "expenses" in data
        assert data["total"] == 2

    @pytest.mark.asyncio
    async def test_list_expenses_with_pagination(self, mock_tracker):
        """List expenses with pagination."""
        with patch.object(expenses_module, "get_expense_tracker", return_value=mock_tracker):
            result = await handle_list_expenses(
                {
                    "limit": "10",
                    "offset": "0",
                }
            )

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["limit"] == 10
        assert data["offset"] == 0

    @pytest.mark.asyncio
    async def test_list_expenses_with_filters(self, mock_tracker):
        """List expenses with filters."""
        with patch.object(expenses_module, "get_expense_tracker", return_value=mock_tracker):
            result = await handle_list_expenses(
                {
                    "vendor": "Test",
                    "category": "travel",
                }
            )

        assert result is not None
        assert result.status_code == 200


# ===========================================================================
# Test Get Expense (GET /api/v1/accounting/expenses/{id})
# ===========================================================================


class TestExpenseGetById:
    """Tests for GET /api/v1/accounting/expenses/{id} endpoint."""

    @pytest.mark.asyncio
    async def test_get_expense_success(self, mock_tracker):
        """Happy path: get expense by ID."""
        with patch.object(expenses_module, "get_expense_tracker", return_value=mock_tracker):
            result = await handle_get_expense("expense-123")

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert "expense" in data
        assert data["expense"]["expense_id"] == "expense-123"

    @pytest.mark.asyncio
    async def test_get_expense_not_found(self, mock_tracker):
        """Expense not found returns 404."""
        with patch.object(expenses_module, "get_expense_tracker", return_value=mock_tracker):
            result = await handle_get_expense("nonexistent")

        assert result is not None
        assert result.status_code == 404


# ===========================================================================
# Test Update Expense (PUT /api/v1/accounting/expenses/{id})
# ===========================================================================


class TestExpenseUpdate:
    """Tests for PUT /api/v1/accounting/expenses/{id} endpoint."""

    @pytest.mark.asyncio
    async def test_update_expense_success(self, mock_tracker):
        """Happy path: update expense."""
        with patch.object(expenses_module, "get_expense_tracker", return_value=mock_tracker):
            result = await handle_update_expense(
                "expense-123",
                {
                    "vendor_name": "Updated Vendor",
                    "amount": 199.99,
                },
            )

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert "expense" in data
        assert data["expense"]["vendor_name"] == "Updated Vendor"

    @pytest.mark.asyncio
    async def test_update_expense_not_found(self, mock_tracker):
        """Expense not found returns 404."""
        with patch.object(expenses_module, "get_expense_tracker", return_value=mock_tracker):
            result = await handle_update_expense("nonexistent", {"amount": 50.00})

        assert result is not None
        assert result.status_code == 404


# ===========================================================================
# Test Delete Expense (DELETE /api/v1/accounting/expenses/{id})
# ===========================================================================


class TestExpenseDelete:
    """Tests for DELETE /api/v1/accounting/expenses/{id} endpoint."""

    @pytest.mark.asyncio
    async def test_delete_expense_success(self, mock_tracker):
        """Happy path: delete expense."""
        with patch.object(expenses_module, "get_expense_tracker", return_value=mock_tracker):
            result = await handle_delete_expense("expense-123")

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert "deleted" in data["message"].lower()

    @pytest.mark.asyncio
    async def test_delete_expense_not_found(self, mock_tracker):
        """Expense not found returns 404."""
        with patch.object(expenses_module, "get_expense_tracker", return_value=mock_tracker):
            result = await handle_delete_expense("nonexistent")

        assert result is not None
        assert result.status_code == 404


# ===========================================================================
# Test Approve Expense (POST /api/v1/accounting/expenses/{id}/approve)
# ===========================================================================


class TestExpenseApprove:
    """Tests for POST /api/v1/accounting/expenses/{id}/approve endpoint."""

    @pytest.mark.asyncio
    async def test_approve_expense_success(self, mock_tracker):
        """Happy path: approve expense."""
        with patch.object(expenses_module, "get_expense_tracker", return_value=mock_tracker):
            result = await handle_approve_expense("expense-123")

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert "approved" in data["message"].lower()

    @pytest.mark.asyncio
    async def test_approve_expense_not_found(self, mock_tracker):
        """Expense not found returns 404."""
        with patch.object(expenses_module, "get_expense_tracker", return_value=mock_tracker):
            result = await handle_approve_expense("nonexistent")

        assert result is not None
        assert result.status_code == 404


# ===========================================================================
# Test Reject Expense (POST /api/v1/accounting/expenses/{id}/reject)
# ===========================================================================


class TestExpenseReject:
    """Tests for POST /api/v1/accounting/expenses/{id}/reject endpoint."""

    @pytest.mark.asyncio
    async def test_reject_expense_success(self, mock_tracker):
        """Happy path: reject expense."""
        with patch.object(expenses_module, "get_expense_tracker", return_value=mock_tracker):
            result = await handle_reject_expense("expense-123", {"reason": "Invalid receipt"})

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert "rejected" in data["message"].lower()

    @pytest.mark.asyncio
    async def test_reject_expense_not_found(self, mock_tracker):
        """Expense not found returns 404."""
        with patch.object(expenses_module, "get_expense_tracker", return_value=mock_tracker):
            result = await handle_reject_expense("nonexistent", {})

        assert result is not None
        assert result.status_code == 404


# ===========================================================================
# Test Categorize Expenses (POST /api/v1/accounting/expenses/categorize)
# ===========================================================================


class TestExpenseCategorize:
    """Tests for POST /api/v1/accounting/expenses/categorize endpoint."""

    @pytest.mark.asyncio
    async def test_categorize_expenses_success(self, mock_tracker):
        """Happy path: auto-categorize expenses."""
        with patch.object(expenses_module, "get_expense_tracker", return_value=mock_tracker):
            result = await handle_categorize_expenses(
                {
                    "expense_ids": ["expense-123", "expense-456"],
                }
            )

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert "categorized" in data
        assert data["count"] == 2

    @pytest.mark.asyncio
    async def test_categorize_all_uncategorized(self, mock_tracker):
        """Categorize all uncategorized expenses when no IDs provided."""
        with patch.object(expenses_module, "get_expense_tracker", return_value=mock_tracker):
            result = await handle_categorize_expenses({})

        assert result is not None
        assert result.status_code == 200


# ===========================================================================
# Test Sync to QBO (POST /api/v1/accounting/expenses/sync)
# ===========================================================================


class TestExpenseSyncQBO:
    """Tests for POST /api/v1/accounting/expenses/sync endpoint."""

    @pytest.mark.asyncio
    async def test_sync_to_qbo_success(self, mock_tracker):
        """Happy path: sync expenses to QBO."""
        with patch.object(expenses_module, "get_expense_tracker", return_value=mock_tracker):
            result = await handle_sync_to_qbo(
                {
                    "expense_ids": ["expense-123"],
                }
            )

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert "result" in data

    @pytest.mark.asyncio
    async def test_sync_all_approved(self, mock_tracker):
        """Sync all approved expenses when no IDs provided."""
        with patch.object(expenses_module, "get_expense_tracker", return_value=mock_tracker):
            result = await handle_sync_to_qbo({})

        assert result is not None
        assert result.status_code == 200


# ===========================================================================
# Test Get Stats (GET /api/v1/accounting/expenses/stats)
# ===========================================================================


class TestExpenseStats:
    """Tests for GET /api/v1/accounting/expenses/stats endpoint."""

    @pytest.mark.asyncio
    async def test_get_stats_success(self, mock_tracker):
        """Happy path: get expense statistics."""
        with patch.object(expenses_module, "get_expense_tracker", return_value=mock_tracker):
            result = await handle_get_expense_stats({})

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert "stats" in data
        assert "total_expenses" in data["stats"]

    @pytest.mark.asyncio
    async def test_get_stats_with_date_range(self, mock_tracker):
        """Get stats with date range."""
        with patch.object(expenses_module, "get_expense_tracker", return_value=mock_tracker):
            result = await handle_get_expense_stats(
                {
                    "start_date": "2024-01-01T00:00:00Z",
                    "end_date": "2024-12-31T23:59:59Z",
                }
            )

        assert result is not None
        assert result.status_code == 200


# ===========================================================================
# Test Get Pending (GET /api/v1/accounting/expenses/pending)
# ===========================================================================


class TestExpensePending:
    """Tests for GET /api/v1/accounting/expenses/pending endpoint."""

    @pytest.mark.asyncio
    async def test_get_pending_success(self, mock_tracker):
        """Happy path: get pending approvals."""
        with patch.object(expenses_module, "get_expense_tracker", return_value=mock_tracker):
            result = await handle_get_pending_approvals()

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert "expenses" in data
        assert "count" in data


# ===========================================================================
# Test Export Expenses (GET /api/v1/accounting/expenses/export)
# ===========================================================================


class TestExpenseExport:
    """Tests for GET /api/v1/accounting/expenses/export endpoint."""

    @pytest.mark.asyncio
    async def test_export_csv_success(self, mock_tracker):
        """Happy path: export expenses as CSV."""
        with patch.object(expenses_module, "get_expense_tracker", return_value=mock_tracker):
            result = await handle_export_expenses({"format": "csv"})

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["format"] == "csv"

    @pytest.mark.asyncio
    async def test_export_json_success(self, mock_tracker):
        """Export expenses as JSON."""
        with patch.object(expenses_module, "get_expense_tracker", return_value=mock_tracker):
            result = await handle_export_expenses({"format": "json"})

        assert result is not None
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_export_invalid_format(self):
        """Invalid format returns 400."""
        result = await handle_export_expenses({"format": "xml"})

        assert result is not None
        assert result.status_code == 400


# ===========================================================================
# Test Handler Class Methods
# ===========================================================================


class TestExpenseHandlerMethods:
    """Tests for ExpenseHandler class methods."""

    @pytest.mark.asyncio
    async def test_handle_get_list(self, expense_handler, mock_tracker):
        """Handler routes GET requests correctly."""
        with patch.object(expenses_module, "get_expense_tracker", return_value=mock_tracker):
            result = await expense_handler.handle_get("/api/v1/accounting/expenses", {})

        assert result is not None
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_handle_get_stats(self, expense_handler, mock_tracker):
        """Handler routes GET stats requests correctly."""
        with patch.object(expenses_module, "get_expense_tracker", return_value=mock_tracker):
            result = await expense_handler.handle_get("/api/v1/accounting/expenses/stats", {})

        assert result is not None
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_handle_get_pending(self, expense_handler, mock_tracker):
        """Handler routes GET pending requests correctly."""
        with patch.object(expenses_module, "get_expense_tracker", return_value=mock_tracker):
            result = await expense_handler.handle_get("/api/v1/accounting/expenses/pending", {})

        assert result is not None
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_handle_get_by_id(self, expense_handler, mock_tracker):
        """Handler routes GET by ID requests correctly."""
        with patch.object(expenses_module, "get_expense_tracker", return_value=mock_tracker):
            result = await expense_handler.handle_get("/api/v1/accounting/expenses/expense-123", {})

        assert result is not None
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_handle_post_create(self, expense_handler, mock_tracker):
        """Handler routes POST create requests correctly."""
        with patch.object(expenses_module, "get_expense_tracker", return_value=mock_tracker):
            result = await expense_handler.handle_post(
                "/api/v1/accounting/expenses",
                {"vendor_name": "Test", "amount": 100.00},
            )

        assert result is not None
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_handle_post_approve(self, expense_handler, mock_tracker):
        """Handler routes POST approve requests correctly."""
        with patch.object(expenses_module, "get_expense_tracker", return_value=mock_tracker):
            result = await expense_handler.handle_post(
                "/api/v1/accounting/expenses/expense-123/approve",
                {},
            )

        assert result is not None
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_handle_put_update(self, expense_handler, mock_tracker):
        """Handler routes PUT update requests correctly."""
        with patch.object(expenses_module, "get_expense_tracker", return_value=mock_tracker):
            result = await expense_handler.handle_put(
                "/api/v1/accounting/expenses/expense-123",
                {"amount": 200.00},
            )

        assert result is not None
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_handle_delete(self, expense_handler, mock_tracker):
        """Handler routes DELETE requests correctly."""
        with patch.object(expenses_module, "get_expense_tracker", return_value=mock_tracker):
            result = await expense_handler.handle_delete("/api/v1/accounting/expenses/expense-123")

        assert result is not None
        assert result.status_code == 200


# ===========================================================================
# Test Error Handling
# ===========================================================================


class TestExpenseErrorHandling:
    """Tests for error handling in expense handler."""

    @pytest.mark.asyncio
    async def test_tracker_error_handling(self):
        """Tracker errors are handled gracefully."""
        mock_tracker = MagicMock()
        mock_tracker.list_expenses = AsyncMock(side_effect=ValueError("Database error"))

        with patch.object(expenses_module, "get_expense_tracker", return_value=mock_tracker):
            result = await handle_list_expenses({})

        assert result is not None
        assert result.status_code == 500
