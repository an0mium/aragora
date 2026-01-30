"""Tests for the Expense Tracker service."""

from __future__ import annotations

import json
import pytest
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.services.expense_tracker import (
    ExpenseCategory,
    ExpenseRecord,
    ExpenseStats,
    ExpenseStatus,
    ExpenseTracker,
    LineItem,
    PaymentMethod,
    SyncResult,
    VENDOR_CATEGORY_PATTERNS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tracker(**kw) -> ExpenseTracker:
    """Create tracker with circuit breakers disabled by default."""
    kw.setdefault("enable_circuit_breakers", False)
    kw.setdefault("enable_llm_categorization", False)
    return ExpenseTracker(**kw)


def _make_expense(tracker: ExpenseTracker, **overrides) -> ExpenseRecord:
    """Directly insert an expense into the tracker for test setup."""
    defaults = dict(
        id="exp_test123",
        vendor_name="Acme Corp",
        amount=Decimal("100.00"),
        currency="USD",
        date=datetime(2025, 6, 1),
        category=ExpenseCategory.SOFTWARE,
        status=ExpenseStatus.PROCESSED,
        payment_method=PaymentMethod.CREDIT_CARD,
    )
    defaults.update(overrides)
    expense = ExpenseRecord(**defaults)
    tracker._store_expense(expense)
    return expense


# ---------------------------------------------------------------------------
# Dataclass / enum tests
# ---------------------------------------------------------------------------


class TestExpenseEnumsAndDataclasses:
    def test_expense_category_values(self):
        assert ExpenseCategory.TRAVEL.value == "travel"
        assert ExpenseCategory.SOFTWARE.value == "software"
        assert ExpenseCategory.OTHER.value == "other"

    def test_expense_status_values(self):
        assert ExpenseStatus.PENDING.value == "pending"
        assert ExpenseStatus.SYNCED.value == "synced"
        assert ExpenseStatus.DUPLICATE.value == "duplicate"

    def test_payment_method_values(self):
        assert PaymentMethod.CREDIT_CARD.value == "credit_card"
        assert PaymentMethod.ACH.value == "ach"

    def test_line_item_to_dict(self):
        li = LineItem(
            description="Widget",
            quantity=2.0,
            unit_price=Decimal("10.50"),
            amount=Decimal("21.00"),
            category=ExpenseCategory.OFFICE_SUPPLIES,
        )
        d = li.to_dict()
        assert d["description"] == "Widget"
        assert d["quantity"] == 2.0
        assert d["amount"] == 21.0
        assert d["category"] == "office_supplies"

    def test_line_item_to_dict_no_category(self):
        li = LineItem(description="Thing")
        assert li.to_dict()["category"] is None

    def test_expense_record_total_amount(self):
        exp = ExpenseRecord(
            id="x",
            vendor_name="V",
            amount=Decimal("100"),
            tax_amount=Decimal("8.25"),
            tip_amount=Decimal("15.00"),
        )
        assert exp.total_amount == Decimal("123.25")

    def test_expense_record_hash_key_deterministic(self):
        exp = ExpenseRecord(
            id="x",
            vendor_name="Starbucks",
            amount=Decimal("5.00"),
            date=datetime(2025, 3, 15),
        )
        h1 = exp.hash_key
        h2 = exp.hash_key
        assert h1 == h2
        assert len(h1) == 32  # md5 hex digest

    def test_expense_record_to_dict(self):
        exp = ExpenseRecord(
            id="exp_abc",
            vendor_name="Test Vendor",
            amount=Decimal("50.00"),
            tax_amount=Decimal("4.13"),
        )
        d = exp.to_dict()
        assert d["id"] == "exp_abc"
        assert d["vendorName"] == "Test Vendor"
        assert d["amount"] == 50.0
        assert d["taxAmount"] == 4.13
        assert d["totalAmount"] == 54.13

    def test_sync_result_to_dict(self):
        sr = SyncResult(success_count=3, failed_count=1, duplicate_count=2)
        d = sr.to_dict()
        assert d["successCount"] == 3
        assert d["failedCount"] == 1

    def test_expense_stats_to_dict(self):
        s = ExpenseStats(total_expenses=10, total_amount=1500.0, avg_expense=150.0)
        d = s.to_dict()
        assert d["totalExpenses"] == 10
        assert d["avgExpense"] == 150.0


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestExpenseTrackerInit:
    def test_init_defaults(self):
        tracker = _make_tracker()
        assert tracker.enable_ocr is True
        assert tracker.enable_llm_categorization is False
        assert tracker._expenses == {}

    def test_init_with_qbo_connector(self):
        mock_qbo = MagicMock()
        tracker = _make_tracker(qbo_connector=mock_qbo)
        assert tracker.qbo is mock_qbo


# ---------------------------------------------------------------------------
# create_expense
# ---------------------------------------------------------------------------


class TestCreateExpense:
    @pytest.mark.asyncio
    async def test_create_basic_expense(self):
        tracker = _make_tracker()
        expense = await tracker.create_expense(
            vendor_name="Starbucks",
            amount=5.75,
            category=ExpenseCategory.MEALS,
        )
        assert expense.vendor_name == "Starbucks"
        assert expense.amount == Decimal("5.75")
        assert expense.category == ExpenseCategory.MEALS
        assert expense.id.startswith("exp_")

    @pytest.mark.asyncio
    async def test_create_expense_auto_categorize(self):
        tracker = _make_tracker()
        expense = await tracker.create_expense(
            vendor_name="Starbucks Coffee",
            amount=5.75,
        )
        # Should match pattern for meals
        assert expense.category == ExpenseCategory.MEALS

    @pytest.mark.asyncio
    async def test_create_expense_stored_in_memory(self):
        tracker = _make_tracker()
        expense = await tracker.create_expense(
            vendor_name="Test",
            amount=10.0,
            category=ExpenseCategory.OTHER,
        )
        assert expense.id in tracker._expenses

    @pytest.mark.asyncio
    async def test_create_expense_with_tags(self):
        tracker = _make_tracker()
        expense = await tracker.create_expense(
            vendor_name="Test",
            amount=10.0,
            category=ExpenseCategory.OTHER,
            tags=["team-lunch", "q1"],
        )
        assert expense.tags == ["team-lunch", "q1"]

    @pytest.mark.asyncio
    async def test_create_expense_duplicate_detection(self):
        tracker = _make_tracker()
        exp1 = await tracker.create_expense(
            vendor_name="Acme",
            amount=100.0,
            date=datetime(2025, 6, 1),
            category=ExpenseCategory.SOFTWARE,
        )
        exp2 = await tracker.create_expense(
            vendor_name="Acme",
            amount=100.0,
            date=datetime(2025, 6, 1),
        )
        assert exp2.status == ExpenseStatus.DUPLICATE
        assert exp2.duplicate_of == exp1.id


# ---------------------------------------------------------------------------
# categorize_expense
# ---------------------------------------------------------------------------


class TestCategorizeExpense:
    @pytest.mark.asyncio
    async def test_categorize_meals(self):
        tracker = _make_tracker()
        expense = ExpenseRecord(id="x", vendor_name="Chipotle", amount=Decimal("12"))
        cat = await tracker.categorize_expense(expense)
        assert cat == ExpenseCategory.MEALS

    @pytest.mark.asyncio
    async def test_categorize_travel(self):
        tracker = _make_tracker()
        expense = ExpenseRecord(id="x", vendor_name="Delta Airlines", amount=Decimal("300"))
        cat = await tracker.categorize_expense(expense)
        assert cat == ExpenseCategory.TRAVEL

    @pytest.mark.asyncio
    async def test_categorize_software(self):
        tracker = _make_tracker()
        expense = ExpenseRecord(id="x", vendor_name="GitHub Enterprise", amount=Decimal("50"))
        cat = await tracker.categorize_expense(expense)
        assert cat == ExpenseCategory.SOFTWARE

    @pytest.mark.asyncio
    async def test_categorize_unknown_vendor(self):
        tracker = _make_tracker()
        expense = ExpenseRecord(id="x", vendor_name="ZZZ Unknown Co", amount=Decimal("50"))
        cat = await tracker.categorize_expense(expense)
        assert cat == ExpenseCategory.OTHER


# ---------------------------------------------------------------------------
# detect_duplicates
# ---------------------------------------------------------------------------


class TestDetectDuplicates:
    @pytest.mark.asyncio
    async def test_exact_hash_match(self):
        tracker = _make_tracker()
        exp1 = _make_expense(
            tracker,
            id="exp_1",
            vendor_name="Acme",
            amount=Decimal("100"),
            date=datetime(2025, 6, 1),
        )
        exp2 = ExpenseRecord(
            id="exp_2", vendor_name="Acme", amount=Decimal("100"), date=datetime(2025, 6, 1)
        )
        dups = await tracker.detect_duplicates(exp2)
        assert len(dups) == 1
        assert dups[0].id == "exp_1"

    @pytest.mark.asyncio
    async def test_fuzzy_match_close_date(self):
        tracker = _make_tracker()
        _make_expense(
            tracker,
            id="exp_1",
            vendor_name="Acme",
            amount=Decimal("100.00"),
            date=datetime(2025, 6, 1),
        )
        exp2 = ExpenseRecord(
            id="exp_2", vendor_name="Acme", amount=Decimal("100.50"), date=datetime(2025, 6, 2)
        )
        dups = await tracker.detect_duplicates(exp2)
        assert len(dups) == 1

    @pytest.mark.asyncio
    async def test_no_duplicate_different_vendor(self):
        tracker = _make_tracker()
        _make_expense(
            tracker,
            id="exp_1",
            vendor_name="Acme",
            amount=Decimal("100"),
            date=datetime(2025, 6, 1),
        )
        exp2 = ExpenseRecord(
            id="exp_2", vendor_name="Other", amount=Decimal("100"), date=datetime(2025, 6, 1)
        )
        dups = await tracker.detect_duplicates(exp2)
        assert len(dups) == 0


# ---------------------------------------------------------------------------
# sync_to_qbo
# ---------------------------------------------------------------------------


class TestSyncToQbo:
    @pytest.mark.asyncio
    async def test_sync_no_qbo_connector(self):
        tracker = _make_tracker()
        result = await tracker.sync_to_qbo()
        assert result.errors[0]["error"] == "QBO connector not configured"

    @pytest.mark.asyncio
    async def test_sync_with_expenses(self):
        mock_qbo = MagicMock()
        tracker = _make_tracker(qbo_connector=mock_qbo)
        exp1 = _make_expense(tracker, id="exp_1", status=ExpenseStatus.APPROVED)
        result = await tracker.sync_to_qbo(expenses=[exp1])
        assert result.success_count == 1

    @pytest.mark.asyncio
    async def test_sync_skips_duplicates(self):
        mock_qbo = MagicMock()
        tracker = _make_tracker(qbo_connector=mock_qbo)
        exp1 = _make_expense(tracker, id="exp_1", status=ExpenseStatus.DUPLICATE)
        result = await tracker.sync_to_qbo(expenses=[exp1])
        assert result.duplicate_count == 1
        assert result.success_count == 0


# ---------------------------------------------------------------------------
# CRUD operations
# ---------------------------------------------------------------------------


class TestExpenseCRUD:
    @pytest.mark.asyncio
    async def test_get_expense(self):
        tracker = _make_tracker()
        exp = _make_expense(tracker)
        fetched = await tracker.get_expense(exp.id)
        assert fetched is not None
        assert fetched.id == exp.id

    @pytest.mark.asyncio
    async def test_get_expense_not_found(self):
        tracker = _make_tracker()
        result = await tracker.get_expense("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_update_expense(self):
        tracker = _make_tracker()
        exp = _make_expense(tracker)
        updated = await tracker.update_expense(exp.id, vendor_name="New Vendor", amount=200.0)
        assert updated is not None
        assert updated.vendor_name == "New Vendor"
        assert updated.amount == Decimal("200.0")

    @pytest.mark.asyncio
    async def test_update_expense_not_found(self):
        tracker = _make_tracker()
        result = await tracker.update_expense("nonexistent", vendor_name="X")
        assert result is None

    @pytest.mark.asyncio
    async def test_approve_expense(self):
        tracker = _make_tracker()
        exp = _make_expense(tracker)
        approved = await tracker.approve_expense(exp.id)
        assert approved.status == ExpenseStatus.APPROVED

    @pytest.mark.asyncio
    async def test_reject_expense(self):
        tracker = _make_tracker()
        exp = _make_expense(tracker)
        rejected = await tracker.reject_expense(exp.id, reason="Duplicate")
        assert rejected.status == ExpenseStatus.REJECTED
        assert rejected.notes == "Duplicate"

    @pytest.mark.asyncio
    async def test_delete_expense(self):
        tracker = _make_tracker()
        exp = _make_expense(tracker)
        result = await tracker.delete_expense(exp.id)
        assert result is True
        assert exp.id not in tracker._expenses

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self):
        tracker = _make_tracker()
        result = await tracker.delete_expense("nonexistent")
        assert result is False


# ---------------------------------------------------------------------------
# list_expenses & filtering
# ---------------------------------------------------------------------------


class TestListExpenses:
    @pytest.mark.asyncio
    async def test_list_all(self):
        tracker = _make_tracker()
        _make_expense(tracker, id="exp_1")
        _make_expense(tracker, id="exp_2", vendor_name="Other Corp")
        expenses, total = await tracker.list_expenses()
        assert total == 2

    @pytest.mark.asyncio
    async def test_filter_by_category(self):
        tracker = _make_tracker()
        _make_expense(tracker, id="exp_1", category=ExpenseCategory.SOFTWARE)
        _make_expense(tracker, id="exp_2", category=ExpenseCategory.MEALS)
        expenses, total = await tracker.list_expenses(category=ExpenseCategory.SOFTWARE)
        assert total == 1
        assert expenses[0].category == ExpenseCategory.SOFTWARE

    @pytest.mark.asyncio
    async def test_filter_by_vendor(self):
        tracker = _make_tracker()
        _make_expense(tracker, id="exp_1", vendor_name="Acme Corp")
        _make_expense(tracker, id="exp_2", vendor_name="Other Corp")
        expenses, total = await tracker.list_expenses(vendor="acme")
        assert total == 1

    @pytest.mark.asyncio
    async def test_filter_by_status(self):
        tracker = _make_tracker()
        _make_expense(tracker, id="exp_1", status=ExpenseStatus.APPROVED)
        _make_expense(tracker, id="exp_2", status=ExpenseStatus.PENDING)
        expenses, total = await tracker.list_expenses(status=ExpenseStatus.APPROVED)
        assert total == 1

    @pytest.mark.asyncio
    async def test_pagination(self):
        tracker = _make_tracker()
        for i in range(5):
            _make_expense(tracker, id=f"exp_{i}", vendor_name=f"V{i}")
        expenses, total = await tracker.list_expenses(limit=2, offset=0)
        assert total == 5
        assert len(expenses) == 2


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


class TestExpenseStats:
    def test_empty_stats(self):
        tracker = _make_tracker()
        stats = tracker.get_stats()
        assert stats.total_expenses == 0

    def test_stats_with_expenses(self):
        tracker = _make_tracker()
        _make_expense(tracker, id="exp_1", amount=Decimal("100"), vendor_name="V1")
        _make_expense(tracker, id="exp_2", amount=Decimal("200"), vendor_name="V2")
        stats = tracker.get_stats()
        assert stats.total_expenses == 2
        assert stats.total_amount == 300.0

    def test_stats_excludes_duplicates_and_rejected(self):
        tracker = _make_tracker()
        _make_expense(tracker, id="exp_1", amount=Decimal("100"), status=ExpenseStatus.PROCESSED)
        _make_expense(tracker, id="exp_2", amount=Decimal("200"), status=ExpenseStatus.DUPLICATE)
        _make_expense(tracker, id="exp_3", amount=Decimal("300"), status=ExpenseStatus.REJECTED)
        stats = tracker.get_stats()
        assert stats.total_expenses == 1
        assert stats.total_amount == 100.0


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------


class TestQueryHelpers:
    @pytest.mark.asyncio
    async def test_get_expenses_by_vendor(self):
        tracker = _make_tracker()
        _make_expense(tracker, id="exp_1", vendor_name="Acme Corp")
        _make_expense(tracker, id="exp_2", vendor_name="Other Corp")
        results = await tracker.get_expenses_by_vendor("Acme Corp")
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_get_expenses_by_category(self):
        tracker = _make_tracker()
        _make_expense(tracker, id="exp_1", category=ExpenseCategory.MEALS)
        _make_expense(tracker, id="exp_2", category=ExpenseCategory.SOFTWARE)
        results = await tracker.get_expenses_by_category(ExpenseCategory.MEALS)
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_get_pending_approval(self):
        tracker = _make_tracker()
        _make_expense(tracker, id="exp_1", status=ExpenseStatus.PROCESSED)
        _make_expense(tracker, id="exp_2", status=ExpenseStatus.APPROVED)
        _make_expense(tracker, id="exp_3", status=ExpenseStatus.CATEGORIZED)
        pending = await tracker.get_pending_approval()
        assert len(pending) == 2

    @pytest.mark.asyncio
    async def test_get_reimbursable(self):
        tracker = _make_tracker()
        _make_expense(tracker, id="exp_1", is_reimbursable=True, employee_id="emp1")
        _make_expense(tracker, id="exp_2", is_reimbursable=False)
        results = await tracker.get_reimbursable_expenses()
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_get_reimbursable_by_employee(self):
        tracker = _make_tracker()
        _make_expense(tracker, id="exp_1", is_reimbursable=True, employee_id="emp1")
        _make_expense(tracker, id="exp_2", is_reimbursable=True, employee_id="emp2")
        results = await tracker.get_reimbursable_expenses(employee_id="emp1")
        assert len(results) == 1


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


class TestExportExpenses:
    @pytest.mark.asyncio
    async def test_export_csv(self):
        tracker = _make_tracker()
        _make_expense(tracker, id="exp_1", vendor_name="Acme", description="Test")
        csv_data = await tracker.export_expenses(format="csv")
        assert "date,vendor,amount" in csv_data
        assert "Acme" in csv_data

    @pytest.mark.asyncio
    async def test_export_json(self):
        tracker = _make_tracker()
        _make_expense(tracker, id="exp_1", vendor_name="Acme")
        json_data = await tracker.export_expenses(format="json")
        parsed = json.loads(json_data)
        assert isinstance(parsed, list)
        assert len(parsed) == 1
        assert parsed[0]["vendorName"] == "Acme"


# ---------------------------------------------------------------------------
# Receipt text parsing
# ---------------------------------------------------------------------------


class TestReceiptParsing:
    def test_parse_empty_text(self):
        tracker = _make_tracker()
        result = tracker._parse_receipt_text("")
        assert result["vendor"] == ""
        assert result["amount"] == 0.0

    def test_parse_total_amount(self):
        tracker = _make_tracker()
        text = "Acme Store\nItem 1 $5.00\nTotal: $15.50"
        result = tracker._parse_receipt_text(text)
        assert result["amount"] == 15.50

    def test_parse_tax(self):
        tracker = _make_tracker()
        text = "Store\nSubtotal $10.00\nTax: $0.83\nTotal: $10.83"
        result = tracker._parse_receipt_text(text)
        assert result["tax"] == 0.83

    def test_parse_tip(self):
        tracker = _make_tracker()
        text = "Restaurant\nFood $20.00\nTip: $4.00\nTotal: $24.00"
        result = tracker._parse_receipt_text(text)
        assert result["tip"] == 4.00

    def test_parse_vendor_name(self):
        tracker = _make_tracker()
        text = "Fancy Restaurant\n123 Main St\nTotal: $50.00"
        result = tracker._parse_receipt_text(text)
        assert result["vendor"] == "Fancy Restaurant"
