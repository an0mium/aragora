"""Tests for expense handler."""

import base64
import json
from datetime import date, datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.expenses import (
    ExpenseHandler,
    handle_approve_expense,
    handle_categorize_expenses,
    handle_create_expense,
    handle_delete_expense,
    handle_export_expenses,
    handle_get_expense,
    handle_get_expense_stats,
    handle_get_pending_approvals,
    handle_list_expenses,
    handle_reject_expense,
    handle_sync_to_qbo,
    handle_update_expense,
    handle_upload_receipt,
)


@pytest.fixture
def mock_expense():
    """Create a mock expense object."""
    expense = MagicMock()
    expense.expense_id = "exp-123"
    expense.vendor_name = "Test Vendor"
    expense.amount = Decimal("99.99")
    expense.currency = "USD"
    expense.expense_date = date(2024, 1, 15)
    expense.category = "Office Supplies"
    expense.status = MagicMock(value="pending")
    expense.payment_method = MagicMock(value="credit_card")
    expense.created_at = datetime(2024, 1, 15, 10, 0, 0)
    expense.to_dict.return_value = {
        "expense_id": "exp-123",
        "vendor_name": "Test Vendor",
        "amount": "99.99",
        "category": "Office Supplies",
        "status": "pending",
    }
    return expense


@pytest.fixture
def mock_tracker(mock_expense):
    """Create a mock expense tracker."""
    tracker = MagicMock()
    tracker.process_receipt = AsyncMock(return_value=mock_expense)
    tracker.create_expense = AsyncMock(return_value=mock_expense)
    tracker.get_expense = AsyncMock(return_value=mock_expense)
    tracker.update_expense = AsyncMock(return_value=mock_expense)
    tracker.delete_expense = AsyncMock(return_value=True)
    tracker.list_expenses = AsyncMock(return_value=[mock_expense])
    tracker.approve_expense = AsyncMock(return_value=mock_expense)
    tracker.reject_expense = AsyncMock(return_value=mock_expense)
    tracker.categorize_expenses = AsyncMock(return_value={"exp-123": "Office Supplies"})
    tracker.sync_to_qbo = AsyncMock(return_value={"synced": 1, "failed": 0})
    tracker.get_stats = AsyncMock(
        return_value={
            "total_expenses": 100,
            "total_amount": "9999.99",
            "by_category": {},
        }
    )
    tracker.get_pending_approvals = AsyncMock(return_value=[mock_expense])
    tracker.detect_duplicates = AsyncMock(return_value=[])
    tracker.export_expenses = AsyncMock(
        return_value=b"expense_id,vendor,amount\nexp-123,Test,99.99"
    )
    tracker.categorize_expense = AsyncMock(return_value="Office Supplies")
    return tracker


@pytest.fixture
def handler():
    """Create expense handler with mock context."""
    return ExpenseHandler(server_context={})


def parse_body(result):
    """Parse JSON body from HandlerResult."""
    return json.loads(result.body.decode())


class TestExpenseHandler:
    """Tests for ExpenseHandler class."""

    def test_can_handle_static_routes(self, handler):
        """Test can_handle for static routes."""
        assert handler.can_handle("/api/v1/accounting/expenses") is True
        assert handler.can_handle("/api/v1/accounting/expenses/upload") is True
        assert handler.can_handle("/api/v1/accounting/expenses/stats") is True
        assert handler.can_handle("/api/v1/accounting/expenses/sync") is True
        assert handler.can_handle("/api/v1/accounting/expenses/pending") is True
        assert handler.can_handle("/api/v1/accounting/expenses/export") is True
        assert handler.can_handle("/api/v1/accounting/expenses/categorize") is True
        assert handler.can_handle("/api/v1/other") is False

    def test_can_handle_dynamic_routes(self, handler):
        """Test can_handle for dynamic routes with path params."""
        assert handler.can_handle("/api/v1/accounting/expenses/exp-123") is True
        assert handler.can_handle("/api/v1/accounting/expenses/exp-456/approve") is True
        assert handler.can_handle("/api/v1/accounting/expenses/exp-789/reject") is True

    def test_matches_pattern(self, handler):
        """Test _matches_pattern helper."""
        assert (
            handler._matches_pattern(
                "/api/v1/accounting/expenses/exp-123",
                "/api/v1/accounting/expenses/{expense_id}",
            )
            is True
        )
        assert (
            handler._matches_pattern(
                "/api/v1/accounting/expenses/exp-123/approve",
                "/api/v1/accounting/expenses/{expense_id}/approve",
            )
            is True
        )
        assert (
            handler._matches_pattern(
                "/api/v1/accounting/expenses",
                "/api/v1/accounting/expenses/{expense_id}",
            )
            is False
        )

    def test_extract_expense_id(self, handler):
        """Test _extract_expense_id helper."""
        assert handler._extract_expense_id("/api/v1/accounting/expenses/exp-123") == "exp-123"
        assert (
            handler._extract_expense_id("/api/v1/accounting/expenses/exp-123/approve") == "exp-123"
        )
        assert handler._extract_expense_id("/api/v1/accounting/expenses") is None

    @pytest.mark.asyncio
    async def test_handle_get_list(self, handler, mock_tracker):
        """Test GET request for listing expenses."""
        with patch(
            "aragora.server.handlers.expenses.get_expense_tracker",
            return_value=mock_tracker,
        ):
            result = await handler.handle_get("/api/v1/accounting/expenses")
            assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_handle_get_single(self, handler, mock_tracker):
        """Test GET request for single expense."""
        with patch(
            "aragora.server.handlers.expenses.get_expense_tracker",
            return_value=mock_tracker,
        ):
            result = await handler.handle_get("/api/v1/accounting/expenses/exp-123")
            assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_handle_get_stats(self, handler, mock_tracker):
        """Test GET request for expense stats."""
        with patch(
            "aragora.server.handlers.expenses.get_expense_tracker",
            return_value=mock_tracker,
        ):
            result = await handler.handle_get("/api/v1/accounting/expenses/stats")
            assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_handle_post_create(self, handler, mock_tracker):
        """Test POST request for creating expense."""
        data = {
            "vendor_name": "Test Vendor",
            "amount": 99.99,
            "category": "Office Supplies",
        }

        with patch(
            "aragora.server.handlers.expenses.get_expense_tracker",
            return_value=mock_tracker,
        ):
            result = await handler.handle_post("/api/v1/accounting/expenses", data)
            assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_handle_post_upload(self, handler, mock_tracker):
        """Test POST request for uploading receipt."""
        data = {
            "receipt_data": base64.b64encode(b"fake image data").decode(),
            "content_type": "image/png",
        }

        with patch(
            "aragora.server.handlers.expenses.get_expense_tracker",
            return_value=mock_tracker,
        ):
            result = await handler.handle_post("/api/v1/accounting/expenses/upload", data)
            assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_handle_post_approve(self, handler, mock_tracker):
        """Test POST request for approving expense."""
        with patch(
            "aragora.server.handlers.expenses.get_expense_tracker",
            return_value=mock_tracker,
        ):
            result = await handler.handle_post("/api/v1/accounting/expenses/exp-123/approve")
            assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_handle_put_update(self, handler, mock_tracker):
        """Test PUT request for updating expense."""
        data = {"category": "Updated Category"}

        with patch(
            "aragora.server.handlers.expenses.get_expense_tracker",
            return_value=mock_tracker,
        ):
            result = await handler.handle_put("/api/v1/accounting/expenses/exp-123", data)
            assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_handle_delete(self, handler, mock_tracker):
        """Test DELETE request."""
        with patch(
            "aragora.server.handlers.expenses.get_expense_tracker",
            return_value=mock_tracker,
        ):
            result = await handler.handle_delete("/api/v1/accounting/expenses/exp-123")
            assert result.status_code == 200


class TestUploadReceipt:
    """Tests for handle_upload_receipt."""

    @pytest.mark.asyncio
    async def test_upload_success(self, mock_tracker, mock_expense):
        """Test successful receipt upload."""
        data = {
            "receipt_data": base64.b64encode(b"image data").decode(),
            "content_type": "image/png",
        }

        with patch(
            "aragora.server.handlers.expenses.get_expense_tracker",
            return_value=mock_tracker,
        ):
            result = await handle_upload_receipt(data)
            assert result.status_code == 200
            body = parse_body(result)
            assert "expense" in body

    @pytest.mark.asyncio
    async def test_upload_missing_receipt_data(self):
        """Test upload with missing receipt data."""
        result = await handle_upload_receipt({})
        assert result.status_code == 400
        body = parse_body(result)
        assert "receipt_data is required" in body.get("error", "")

    @pytest.mark.asyncio
    async def test_upload_invalid_base64(self):
        """Test upload with invalid base64."""
        result = await handle_upload_receipt({"receipt_data": "not-valid-base64!!!"})
        assert result.status_code == 400
        body = parse_body(result)
        assert "Invalid base64" in body.get("error", "")


class TestCreateExpense:
    """Tests for handle_create_expense."""

    @pytest.mark.asyncio
    async def test_create_success(self, mock_tracker, mock_expense):
        """Test successful expense creation."""
        data = {
            "vendor_name": "Test Vendor",
            "amount": 99.99,
            "category": "Office Supplies",
        }

        with patch(
            "aragora.server.handlers.expenses.get_expense_tracker",
            return_value=mock_tracker,
        ):
            result = await handle_create_expense(data)
            assert result.status_code == 200
            body = parse_body(result)
            assert "expense" in body

    @pytest.mark.asyncio
    async def test_create_missing_vendor(self, mock_tracker):
        """Test create with missing vendor name."""
        data = {"amount": 99.99}

        with patch(
            "aragora.server.handlers.expenses.get_expense_tracker",
            return_value=mock_tracker,
        ):
            result = await handle_create_expense(data)
            assert result.status_code == 400
            body = parse_body(result)
            assert "vendor_name" in body.get("error", "")

    @pytest.mark.asyncio
    async def test_create_missing_amount(self, mock_tracker):
        """Test create with missing amount."""
        data = {"vendor_name": "Test"}

        with patch(
            "aragora.server.handlers.expenses.get_expense_tracker",
            return_value=mock_tracker,
        ):
            result = await handle_create_expense(data)
            assert result.status_code == 400
            body = parse_body(result)
            assert "amount" in body.get("error", "")


class TestListExpenses:
    """Tests for handle_list_expenses."""

    @pytest.mark.asyncio
    async def test_list_success(self, mock_tracker, mock_expense):
        """Test successful expense listing."""
        with patch(
            "aragora.server.handlers.expenses.get_expense_tracker",
            return_value=mock_tracker,
        ):
            result = await handle_list_expenses({})
            assert result.status_code == 200
            body = parse_body(result)
            assert "expenses" in body

    @pytest.mark.asyncio
    async def test_list_with_filters(self, mock_tracker, mock_expense):
        """Test expense listing with filters."""
        query_params = {
            "category": "Office Supplies",
            "status": "pending",
            "limit": "10",
        }

        with patch(
            "aragora.server.handlers.expenses.get_expense_tracker",
            return_value=mock_tracker,
        ):
            result = await handle_list_expenses(query_params)
            assert result.status_code == 200


class TestGetExpense:
    """Tests for handle_get_expense."""

    @pytest.mark.asyncio
    async def test_get_success(self, mock_tracker, mock_expense):
        """Test successful expense retrieval."""
        with patch(
            "aragora.server.handlers.expenses.get_expense_tracker",
            return_value=mock_tracker,
        ):
            result = await handle_get_expense("exp-123")
            assert result.status_code == 200
            body = parse_body(result)
            assert "expense" in body

    @pytest.mark.asyncio
    async def test_get_not_found(self, mock_tracker):
        """Test expense not found."""
        mock_tracker.get_expense = AsyncMock(return_value=None)

        with patch(
            "aragora.server.handlers.expenses.get_expense_tracker",
            return_value=mock_tracker,
        ):
            result = await handle_get_expense("exp-nonexistent")
            assert result.status_code == 404


class TestUpdateExpense:
    """Tests for handle_update_expense."""

    @pytest.mark.asyncio
    async def test_update_success(self, mock_tracker, mock_expense):
        """Test successful expense update."""
        data = {"category": "New Category"}

        with patch(
            "aragora.server.handlers.expenses.get_expense_tracker",
            return_value=mock_tracker,
        ):
            result = await handle_update_expense("exp-123", data)
            assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_update_not_found(self, mock_tracker):
        """Test update expense not found."""
        mock_tracker.update_expense = AsyncMock(return_value=None)

        with patch(
            "aragora.server.handlers.expenses.get_expense_tracker",
            return_value=mock_tracker,
        ):
            result = await handle_update_expense("exp-nonexistent", {})
            assert result.status_code == 404


class TestDeleteExpense:
    """Tests for handle_delete_expense."""

    @pytest.mark.asyncio
    async def test_delete_success(self, mock_tracker):
        """Test successful expense deletion."""
        with patch(
            "aragora.server.handlers.expenses.get_expense_tracker",
            return_value=mock_tracker,
        ):
            result = await handle_delete_expense("exp-123")
            assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_delete_not_found(self, mock_tracker):
        """Test delete expense not found."""
        mock_tracker.delete_expense = AsyncMock(return_value=False)

        with patch(
            "aragora.server.handlers.expenses.get_expense_tracker",
            return_value=mock_tracker,
        ):
            result = await handle_delete_expense("exp-nonexistent")
            assert result.status_code == 404


class TestApproveExpense:
    """Tests for handle_approve_expense."""

    @pytest.mark.asyncio
    async def test_approve_success(self, mock_tracker, mock_expense):
        """Test successful expense approval."""
        with patch(
            "aragora.server.handlers.expenses.get_expense_tracker",
            return_value=mock_tracker,
        ):
            result = await handle_approve_expense("exp-123")
            assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_approve_not_found(self, mock_tracker):
        """Test approve expense not found."""
        mock_tracker.approve_expense = AsyncMock(return_value=None)

        with patch(
            "aragora.server.handlers.expenses.get_expense_tracker",
            return_value=mock_tracker,
        ):
            result = await handle_approve_expense("exp-nonexistent")
            assert result.status_code == 404


class TestRejectExpense:
    """Tests for handle_reject_expense."""

    @pytest.mark.asyncio
    async def test_reject_success(self, mock_tracker, mock_expense):
        """Test successful expense rejection."""
        data = {"reason": "Invalid receipt"}

        with patch(
            "aragora.server.handlers.expenses.get_expense_tracker",
            return_value=mock_tracker,
        ):
            result = await handle_reject_expense("exp-123", data)
            assert result.status_code == 200


class TestCategorizeExpenses:
    """Tests for handle_categorize_expenses."""

    @pytest.mark.asyncio
    async def test_categorize_success(self, mock_tracker):
        """Test successful expense categorization."""
        data = {"expense_ids": ["exp-123", "exp-456"]}

        with patch(
            "aragora.server.handlers.expenses.get_expense_tracker",
            return_value=mock_tracker,
        ):
            result = await handle_categorize_expenses(data)
            assert result.status_code == 200
            body = parse_body(result)
            assert "categorized" in body


class TestSyncToQBO:
    """Tests for handle_sync_to_qbo."""

    @pytest.mark.asyncio
    async def test_sync_success(self, mock_tracker):
        """Test successful QBO sync."""
        data = {"expense_ids": ["exp-123"]}

        with patch(
            "aragora.server.handlers.expenses.get_expense_tracker",
            return_value=mock_tracker,
        ):
            result = await handle_sync_to_qbo(data)
            assert result.status_code == 200


class TestGetExpenseStats:
    """Tests for handle_get_expense_stats."""

    @pytest.mark.asyncio
    async def test_stats_success(self, mock_tracker):
        """Test successful stats retrieval."""
        with patch(
            "aragora.server.handlers.expenses.get_expense_tracker",
            return_value=mock_tracker,
        ):
            result = await handle_get_expense_stats({})
            assert result.status_code == 200
            body = parse_body(result)
            assert "total_expenses" in body


class TestGetPendingApprovals:
    """Tests for handle_get_pending_approvals."""

    @pytest.mark.asyncio
    async def test_pending_success(self, mock_tracker, mock_expense):
        """Test successful pending approvals retrieval."""
        with patch(
            "aragora.server.handlers.expenses.get_expense_tracker",
            return_value=mock_tracker,
        ):
            result = await handle_get_pending_approvals()
            assert result.status_code == 200
            body = parse_body(result)
            assert "expenses" in body


class TestExportExpenses:
    """Tests for handle_export_expenses."""

    @pytest.mark.asyncio
    async def test_export_success(self, mock_tracker, mock_expense):
        """Test successful expense export."""
        query_params = {"format": "csv"}

        with patch(
            "aragora.server.handlers.expenses.get_expense_tracker",
            return_value=mock_tracker,
        ):
            result = await handle_export_expenses(query_params)
            assert result.status_code == 200
