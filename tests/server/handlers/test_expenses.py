"""
Tests for the expense tracking API handler.

Tests cover:
- Receipt upload and processing
- Expense CRUD operations (create, list, get, update, delete)
- Approval workflow (approve, reject, pending)
- Auto-categorization
- QBO sync integration
- Statistics and export
- ExpenseHandler routing
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch
from enum import Enum

import pytest

from aragora.server.handlers.expenses import (
    ExpenseHandler,
    handle_upload_receipt,
    handle_create_expense,
    handle_list_expenses,
    handle_get_expense,
    handle_update_expense,
    handle_delete_expense,
    handle_approve_expense,
    handle_reject_expense,
    handle_get_pending_approvals,
    handle_categorize_expenses,
    handle_sync_to_qbo,
    handle_get_expense_stats,
    handle_export_expenses,
)


def parse_result(result):
    """Parse HandlerResult into (data, status_code) tuple."""
    data = json.loads(result.body.decode("utf-8"))
    return data, result.status_code


# =============================================================================
# Mock Enums and Data
# =============================================================================


class MockExpenseCategory(Enum):
    TRAVEL = "travel"
    MEALS = "meals"
    SUPPLIES = "supplies"
    SOFTWARE = "software"
    OTHER = "other"


class MockExpenseStatus(Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    SYNCED = "synced"


class MockPaymentMethod(Enum):
    CREDIT_CARD = "credit_card"
    DEBIT_CARD = "debit_card"
    CASH = "cash"
    CHECK = "check"


@dataclass
class MockExpense:
    """Mock expense object."""

    id: str = "exp_123"
    vendor_name: str = "Test Vendor"
    amount: float = 100.00
    category: MockExpenseCategory = MockExpenseCategory.OTHER
    status: MockExpenseStatus = MockExpenseStatus.PENDING
    payment_method: MockPaymentMethod = MockPaymentMethod.CREDIT_CARD
    description: str = ""
    employee_id: Optional[str] = None
    is_reimbursable: bool = False
    tags: List[str] = field(default_factory=list)
    date: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "vendor_name": self.vendor_name,
            "amount": self.amount,
            "category": self.category.value,
            "status": self.status.value,
            "payment_method": self.payment_method.value,
            "description": self.description,
            "employee_id": self.employee_id,
            "is_reimbursable": self.is_reimbursable,
            "tags": self.tags,
            "date": self.date.isoformat(),
        }


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_tracker():
    """Create mock expense tracker."""
    tracker = MagicMock()
    tracker.process_receipt = AsyncMock(return_value=MockExpense())
    tracker.create_expense = AsyncMock(return_value=MockExpense())
    tracker.list_expenses = AsyncMock(return_value=([MockExpense()], 1))
    tracker.get_expense = AsyncMock(return_value=MockExpense())
    tracker.update_expense = AsyncMock(return_value=MockExpense())
    tracker.delete_expense = AsyncMock(return_value=True)
    tracker.approve_expense = AsyncMock(return_value=MockExpense())
    tracker.reject_expense = AsyncMock(return_value=MockExpense())
    tracker.get_pending_approval = AsyncMock(return_value=[MockExpense()])
    tracker.bulk_categorize = AsyncMock(return_value={"exp_1": MockExpenseCategory.TRAVEL})
    tracker.sync_to_qbo = AsyncMock(return_value={"synced": 5})
    tracker.get_stats = AsyncMock(return_value={"total": 5000.00, "count": 50})
    tracker.export_expenses = AsyncMock(return_value="csv,data,here")
    return tracker


@pytest.fixture
def handler():
    """Create ExpenseHandler instance."""
    mock_context = {"storage": None}
    return ExpenseHandler(mock_context)


# =============================================================================
# Receipt Upload Tests
# =============================================================================


class TestReceiptUpload:
    """Tests for receipt upload endpoint."""

    @pytest.mark.asyncio
    async def test_upload_receipt_success(self, mock_tracker):
        """Successfully upload and process receipt."""
        with patch(
            "aragora.server.handlers.expenses.get_expense_tracker",
            return_value=mock_tracker,
        ):
            # Valid base64 encoded image
            image_data = b"fake image data"
            b64_data = base64.b64encode(image_data).decode()

            result = await handle_upload_receipt(
                {"receipt_data": b64_data, "employee_id": "emp_123"}
            )
            data, status = parse_result(result)

            assert status == 200
            assert "expense" in data
            mock_tracker.process_receipt.assert_called_once()

    @pytest.mark.asyncio
    async def test_upload_receipt_missing_data(self, mock_tracker):
        """Reject upload without receipt data."""
        with patch(
            "aragora.server.handlers.expenses.get_expense_tracker",
            return_value=mock_tracker,
        ):
            result = await handle_upload_receipt({})
            _, status = parse_result(result)

            assert status == 400

    @pytest.mark.asyncio
    async def test_upload_receipt_invalid_base64(self, mock_tracker):
        """Reject invalid base64 data."""
        with patch(
            "aragora.server.handlers.expenses.get_expense_tracker",
            return_value=mock_tracker,
        ):
            result = await handle_upload_receipt({"receipt_data": "not-valid-base64!!!"})
            _, status = parse_result(result)

            assert status == 400


# =============================================================================
# Expense CRUD Tests
# =============================================================================


class TestExpenseCreate:
    """Tests for expense creation."""

    @pytest.mark.asyncio
    async def test_create_expense_success(self, mock_tracker):
        """Successfully create expense."""
        with patch(
            "aragora.server.handlers.expenses.get_expense_tracker",
            return_value=mock_tracker,
        ):
            result = await handle_create_expense(
                {
                    "vendor_name": "Office Depot",
                    "amount": 150.00,
                    "category": "supplies",
                    "description": "Office supplies",
                }
            )
            data, status = parse_result(result)

            assert status == 200
            assert "expense" in data
            mock_tracker.create_expense.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_expense_missing_vendor(self, mock_tracker):
        """Reject creation without vendor."""
        with patch(
            "aragora.server.handlers.expenses.get_expense_tracker",
            return_value=mock_tracker,
        ):
            result = await handle_create_expense({"amount": 100.00})
            data, status = parse_result(result)

            assert status == 400
            assert "vendor_name" in data.get("error", "")

    @pytest.mark.asyncio
    async def test_create_expense_missing_amount(self, mock_tracker):
        """Reject creation without amount."""
        with patch(
            "aragora.server.handlers.expenses.get_expense_tracker",
            return_value=mock_tracker,
        ):
            result = await handle_create_expense({"vendor_name": "Test"})
            data, status = parse_result(result)

            assert status == 400
            assert "amount" in data.get("error", "")

    @pytest.mark.asyncio
    async def test_create_expense_invalid_amount(self, mock_tracker):
        """Reject invalid amount."""
        with patch(
            "aragora.server.handlers.expenses.get_expense_tracker",
            return_value=mock_tracker,
        ):
            result = await handle_create_expense({"vendor_name": "Test", "amount": "not-a-number"})
            _, status = parse_result(result)

            assert status == 400

    @pytest.mark.asyncio
    async def test_create_expense_invalid_date(self, mock_tracker):
        """Reject invalid date format."""
        with patch(
            "aragora.server.handlers.expenses.get_expense_tracker",
            return_value=mock_tracker,
        ):
            result = await handle_create_expense(
                {"vendor_name": "Test", "amount": 100.00, "date": "not-a-date"}
            )
            _, status = parse_result(result)

            assert status == 400


class TestExpenseList:
    """Tests for expense listing."""

    @pytest.mark.asyncio
    async def test_list_expenses_success(self, mock_tracker):
        """Successfully list expenses."""
        with patch(
            "aragora.server.handlers.expenses.get_expense_tracker",
            return_value=mock_tracker,
        ):
            result = await handle_list_expenses({})

            assert result[1] == 200
            data = result[0]
            assert "expenses" in data
            assert "total" in data
            assert len(data["expenses"]) == 1

    @pytest.mark.asyncio
    async def test_list_expenses_with_filters(self, mock_tracker):
        """List expenses with filters."""
        with patch(
            "aragora.server.handlers.expenses.get_expense_tracker",
            return_value=mock_tracker,
        ):
            result = await handle_list_expenses(
                {
                    "category": "travel",
                    "vendor": "Airlines",
                    "start_date": "2026-01-01",
                    "end_date": "2026-01-31",
                    "limit": "50",
                    "offset": "10",
                }
            )

            assert result[1] == 200


class TestExpenseGet:
    """Tests for getting single expense."""

    @pytest.mark.asyncio
    async def test_get_expense_success(self, mock_tracker):
        """Successfully get expense by ID."""
        with patch(
            "aragora.server.handlers.expenses.get_expense_tracker",
            return_value=mock_tracker,
        ):
            result = await handle_get_expense("exp_123")

            assert result[1] == 200
            data = result[0]
            assert "expense" in data

    @pytest.mark.asyncio
    async def test_get_expense_not_found(self, mock_tracker):
        """Return 404 for missing expense."""
        mock_tracker.get_expense = AsyncMock(return_value=None)

        with patch(
            "aragora.server.handlers.expenses.get_expense_tracker",
            return_value=mock_tracker,
        ):
            result = await handle_get_expense("nonexistent")

            assert result[1] == 404


class TestExpenseUpdate:
    """Tests for expense updates."""

    @pytest.mark.asyncio
    async def test_update_expense_success(self, mock_tracker):
        """Successfully update expense."""
        with patch(
            "aragora.server.handlers.expenses.get_expense_tracker",
            return_value=mock_tracker,
        ):
            result = await handle_update_expense(
                "exp_123",
                {"amount": 200.00, "description": "Updated"},
            )

            assert result[1] == 200
            mock_tracker.update_expense.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_expense_not_found(self, mock_tracker):
        """Return 404 for missing expense."""
        mock_tracker.update_expense = AsyncMock(return_value=None)

        with patch(
            "aragora.server.handlers.expenses.get_expense_tracker",
            return_value=mock_tracker,
        ):
            result = await handle_update_expense("nonexistent", {"amount": 200.00})

            assert result[1] == 404


class TestExpenseDelete:
    """Tests for expense deletion."""

    @pytest.mark.asyncio
    async def test_delete_expense_success(self, mock_tracker):
        """Successfully delete expense."""
        with patch(
            "aragora.server.handlers.expenses.get_expense_tracker",
            return_value=mock_tracker,
        ):
            result = await handle_delete_expense("exp_123")

            assert result[1] == 200

    @pytest.mark.asyncio
    async def test_delete_expense_not_found(self, mock_tracker):
        """Return 404 for missing expense."""
        mock_tracker.delete_expense = AsyncMock(return_value=False)

        with patch(
            "aragora.server.handlers.expenses.get_expense_tracker",
            return_value=mock_tracker,
        ):
            result = await handle_delete_expense("nonexistent")

            assert result[1] == 404


# =============================================================================
# Approval Workflow Tests
# =============================================================================


class TestApprovalWorkflow:
    """Tests for approval workflow."""

    @pytest.mark.asyncio
    async def test_approve_expense_success(self, mock_tracker):
        """Successfully approve expense."""
        with patch(
            "aragora.server.handlers.expenses.get_expense_tracker",
            return_value=mock_tracker,
        ):
            result = await handle_approve_expense("exp_123")

            assert result[1] == 200
            data = result[0]
            assert "expense" in data
            mock_tracker.approve_expense.assert_called_once_with("exp_123")

    @pytest.mark.asyncio
    async def test_approve_expense_not_found(self, mock_tracker):
        """Return 404 for missing expense."""
        mock_tracker.approve_expense = AsyncMock(return_value=None)

        with patch(
            "aragora.server.handlers.expenses.get_expense_tracker",
            return_value=mock_tracker,
        ):
            result = await handle_approve_expense("nonexistent")

            assert result[1] == 404

    @pytest.mark.asyncio
    async def test_reject_expense_success(self, mock_tracker):
        """Successfully reject expense."""
        with patch(
            "aragora.server.handlers.expenses.get_expense_tracker",
            return_value=mock_tracker,
        ):
            result = await handle_reject_expense("exp_123", {"reason": "Invalid receipt"})

            assert result[1] == 200
            mock_tracker.reject_expense.assert_called_once_with("exp_123", "Invalid receipt")

    @pytest.mark.asyncio
    async def test_reject_expense_not_found(self, mock_tracker):
        """Return 404 for missing expense."""
        mock_tracker.reject_expense = AsyncMock(return_value=None)

        with patch(
            "aragora.server.handlers.expenses.get_expense_tracker",
            return_value=mock_tracker,
        ):
            result = await handle_reject_expense("nonexistent", {})

            assert result[1] == 404

    @pytest.mark.asyncio
    async def test_get_pending_approvals(self, mock_tracker):
        """Get pending approval list."""
        with patch(
            "aragora.server.handlers.expenses.get_expense_tracker",
            return_value=mock_tracker,
        ):
            result = await handle_get_pending_approvals()

            assert result[1] == 200
            data = result[0]
            assert "expenses" in data
            assert "count" in data


# =============================================================================
# Categorization Tests
# =============================================================================


class TestCategorization:
    """Tests for expense categorization."""

    @pytest.mark.asyncio
    async def test_categorize_expenses_success(self, mock_tracker):
        """Successfully categorize expenses."""
        with patch(
            "aragora.server.handlers.expenses.get_expense_tracker",
            return_value=mock_tracker,
        ):
            result = await handle_categorize_expenses({"expense_ids": ["exp_1", "exp_2"]})

            assert result[1] == 200
            data = result[0]
            assert "categorized" in data
            assert "count" in data

    @pytest.mark.asyncio
    async def test_categorize_all_uncategorized(self, mock_tracker):
        """Categorize all uncategorized when no IDs provided."""
        with patch(
            "aragora.server.handlers.expenses.get_expense_tracker",
            return_value=mock_tracker,
        ):
            result = await handle_categorize_expenses({})

            assert result[1] == 200
            mock_tracker.bulk_categorize.assert_called_once_with(None)


# =============================================================================
# QBO Sync Tests
# =============================================================================


class TestQBOSync:
    """Tests for QBO sync."""

    @pytest.mark.asyncio
    async def test_sync_to_qbo_success(self, mock_tracker):
        """Successfully sync to QBO."""
        with patch(
            "aragora.server.handlers.expenses.get_expense_tracker",
            return_value=mock_tracker,
        ):
            result = await handle_sync_to_qbo({"expense_ids": ["exp_1", "exp_2"]})

            assert result[1] == 200
            data = result[0]
            assert "result" in data

    @pytest.mark.asyncio
    async def test_sync_all_approved(self, mock_tracker):
        """Sync all approved when no IDs provided."""
        with patch(
            "aragora.server.handlers.expenses.get_expense_tracker",
            return_value=mock_tracker,
        ):
            result = await handle_sync_to_qbo({})

            assert result[1] == 200


# =============================================================================
# Statistics and Export Tests
# =============================================================================


class TestStatsAndExport:
    """Tests for statistics and export."""

    @pytest.mark.asyncio
    async def test_get_expense_stats(self, mock_tracker):
        """Get expense statistics."""
        with patch(
            "aragora.server.handlers.expenses.get_expense_tracker",
            return_value=mock_tracker,
        ):
            result = await handle_get_expense_stats(
                {"start_date": "2026-01-01", "end_date": "2026-01-31"}
            )

            assert result[1] == 200
            data = result[0]
            assert "stats" in data

    @pytest.mark.asyncio
    async def test_export_expenses_csv(self, mock_tracker):
        """Export expenses as CSV."""
        with patch(
            "aragora.server.handlers.expenses.get_expense_tracker",
            return_value=mock_tracker,
        ):
            result = await handle_export_expenses({"format": "csv"})

            assert result[1] == 200
            data = result[0]
            assert data["format"] == "csv"

    @pytest.mark.asyncio
    async def test_export_expenses_json(self, mock_tracker):
        """Export expenses as JSON."""
        with patch(
            "aragora.server.handlers.expenses.get_expense_tracker",
            return_value=mock_tracker,
        ):
            result = await handle_export_expenses({"format": "json"})

            assert result[1] == 200
            data = result[0]
            assert data["format"] == "json"

    @pytest.mark.asyncio
    async def test_export_expenses_invalid_format(self, mock_tracker):
        """Reject invalid export format."""
        with patch(
            "aragora.server.handlers.expenses.get_expense_tracker",
            return_value=mock_tracker,
        ):
            result = await handle_export_expenses({"format": "xml"})

            assert result[1] == 400


# =============================================================================
# ExpenseHandler Routing Tests
# =============================================================================


class TestExpenseHandlerRouting:
    """Tests for ExpenseHandler route handling."""

    def test_can_handle_static_routes(self, handler):
        """Can handle static routes."""
        assert handler.can_handle("/api/v1/accounting/expenses") is True
        assert handler.can_handle("/api/v1/accounting/expenses/upload") is True
        assert handler.can_handle("/api/v1/accounting/expenses/categorize") is True
        assert handler.can_handle("/api/v1/accounting/expenses/sync") is True
        assert handler.can_handle("/api/v1/accounting/expenses/stats") is True
        assert handler.can_handle("/api/v1/accounting/expenses/pending") is True
        assert handler.can_handle("/api/v1/accounting/expenses/export") is True

    def test_can_handle_dynamic_routes(self, handler):
        """Can handle dynamic routes."""
        assert handler.can_handle("/api/v1/accounting/expenses/exp_123") is True
        assert handler.can_handle("/api/v1/accounting/expenses/exp_456/approve") is True
        assert handler.can_handle("/api/v1/accounting/expenses/exp_789/reject") is True

    def test_cannot_handle_unknown_routes(self, handler):
        """Cannot handle unknown routes."""
        assert handler.can_handle("/api/v1/unknown") is False
        assert handler.can_handle("/api/v1/accounting/invoices") is False

    def test_extract_expense_id(self, handler):
        """Correctly extract expense ID from path."""
        assert handler._extract_expense_id("/api/v1/accounting/expenses/exp_123") == "exp_123"
        assert (
            handler._extract_expense_id("/api/v1/accounting/expenses/exp_456/approve") == "exp_456"
        )
        assert handler._extract_expense_id("/api/v1/accounting/expenses") is None

    @pytest.mark.asyncio
    async def test_handle_get_list(self, handler, mock_tracker):
        """Handle GET list request."""
        with patch(
            "aragora.server.handlers.expenses.get_expense_tracker",
            return_value=mock_tracker,
        ):
            result = await handler.handle_get("/api/v1/accounting/expenses", {})

            assert result[1] == 200

    @pytest.mark.asyncio
    async def test_handle_get_single(self, handler, mock_tracker):
        """Handle GET single expense request."""
        with patch(
            "aragora.server.handlers.expenses.get_expense_tracker",
            return_value=mock_tracker,
        ):
            result = await handler.handle_get("/api/v1/accounting/expenses/exp_123", {})

            assert result[1] == 200

    @pytest.mark.asyncio
    async def test_handle_post_create(self, handler, mock_tracker):
        """Handle POST create request."""
        with patch(
            "aragora.server.handlers.expenses.get_expense_tracker",
            return_value=mock_tracker,
        ):
            result = await handler.handle_post(
                "/api/v1/accounting/expenses",
                {"vendor_name": "Test", "amount": 100.00},
            )

            assert result[1] == 200

    @pytest.mark.asyncio
    async def test_handle_put_update(self, handler, mock_tracker):
        """Handle PUT update request."""
        with patch(
            "aragora.server.handlers.expenses.get_expense_tracker",
            return_value=mock_tracker,
        ):
            result = await handler.handle_put(
                "/api/v1/accounting/expenses/exp_123",
                {"amount": 200.00},
            )

            assert result[1] == 200

    @pytest.mark.asyncio
    async def test_handle_delete(self, handler, mock_tracker):
        """Handle DELETE request."""
        with patch(
            "aragora.server.handlers.expenses.get_expense_tracker",
            return_value=mock_tracker,
        ):
            result = await handler.handle_delete("/api/v1/accounting/expenses/exp_123")

            assert result[1] == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
