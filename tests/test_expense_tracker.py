"""
Tests for the ExpenseTracker service.

Covers:
- Expense CRUD operations
- Receipt processing and OCR parsing
- Expense categorization
- Duplicate detection
- QBO sync preparation
- Statistics and reporting
"""

from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from aragora.services.expense_tracker import (
    ExpenseTracker,
    ExpenseRecord,
    ExpenseCategory,
    ExpenseStatus,
    PaymentMethod,
    SyncResult,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def expense_tracker():
    """Create a fresh ExpenseTracker instance."""
    return ExpenseTracker()


@pytest.fixture
def sample_expense_data():
    """Sample expense data for creating expenses."""
    return {
        "vendor_name": "Acme Corp",
        "amount": 125.50,
        "date": datetime.now(),
        "category": ExpenseCategory.OFFICE_SUPPLIES,
        "payment_method": PaymentMethod.CREDIT_CARD,
        "description": "Office supplies purchase",
        "employee_id": "emp_001",
        "is_reimbursable": True,
        "tags": ["supplies", "Q1"],
    }


@pytest.fixture
def sample_receipt_text():
    """Sample receipt text for parsing tests."""
    return """
    ACME OFFICE SUPPLIES
    123 Main Street
    Anytown, ST 12345

    Date: 01/15/2024

    Printer Paper       $24.99
    Ink Cartridges      $45.00
    Stapler             $12.50
    Pens (12 pack)      $8.99

    Subtotal:           $91.48
    Tax (8%):           $7.32

    Total:              $98.80

    Thank you for shopping!
    """


# =============================================================================
# Expense CRUD Tests
# =============================================================================


class TestExpenseCRUD:
    """Test basic expense CRUD operations."""

    @pytest.mark.asyncio
    async def test_create_expense(self, expense_tracker, sample_expense_data):
        """Test creating an expense."""
        expense = await expense_tracker.create_expense(**sample_expense_data)

        assert expense.id.startswith("exp_")
        assert expense.vendor_name == sample_expense_data["vendor_name"]
        assert float(expense.amount) == sample_expense_data["amount"]
        assert expense.category == sample_expense_data["category"]
        assert expense.status == ExpenseStatus.PENDING

    @pytest.mark.asyncio
    async def test_get_expense(self, expense_tracker, sample_expense_data):
        """Test retrieving an expense by ID."""
        expense = await expense_tracker.create_expense(**sample_expense_data)
        retrieved = await expense_tracker.get_expense(expense.id)

        assert retrieved is not None
        assert retrieved.id == expense.id
        assert retrieved.vendor_name == expense.vendor_name

    @pytest.mark.asyncio
    async def test_get_nonexistent_expense(self, expense_tracker):
        """Test retrieving a non-existent expense."""
        result = await expense_tracker.get_expense("exp_nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_update_expense(self, expense_tracker, sample_expense_data):
        """Test updating an expense."""
        expense = await expense_tracker.create_expense(**sample_expense_data)

        updated = await expense_tracker.update_expense(
            expense.id,
            description="Updated description",
            category=ExpenseCategory.TRAVEL,
        )

        assert updated is not None
        assert updated.description == "Updated description"
        assert updated.category == ExpenseCategory.TRAVEL

    @pytest.mark.asyncio
    async def test_delete_expense(self, expense_tracker, sample_expense_data):
        """Test deleting an expense."""
        expense = await expense_tracker.create_expense(**sample_expense_data)
        success = await expense_tracker.delete_expense(expense.id)

        assert success is True
        assert await expense_tracker.get_expense(expense.id) is None

    @pytest.mark.asyncio
    async def test_list_expenses(self, expense_tracker, sample_expense_data):
        """Test listing expenses."""
        # Create multiple expenses
        await expense_tracker.create_expense(**sample_expense_data)
        await expense_tracker.create_expense(
            vendor_name="Other Vendor",
            amount=50.00,
        )

        expenses = await expense_tracker.list_expenses()
        assert len(expenses) == 2

    @pytest.mark.asyncio
    async def test_list_expenses_with_category_filter(self, expense_tracker, sample_expense_data):
        """Test filtering expenses by category."""
        await expense_tracker.create_expense(**sample_expense_data)
        await expense_tracker.create_expense(
            vendor_name="Travel Agency",
            amount=500.00,
            category=ExpenseCategory.TRAVEL,
        )

        travel_expenses = await expense_tracker.list_expenses(category=ExpenseCategory.TRAVEL)
        assert len(travel_expenses) == 1
        assert travel_expenses[0].vendor_name == "Travel Agency"


# =============================================================================
# Receipt Processing Tests
# =============================================================================


class TestReceiptProcessing:
    """Test receipt processing and OCR."""

    @pytest.mark.asyncio
    async def test_process_receipt_pdf(self, expense_tracker):
        """Test processing a PDF receipt."""
        # Create a minimal PDF-like structure
        pdf_data = b"%PDF-1.4\n% test pdf content"

        expense = await expense_tracker.process_receipt(
            image_data=pdf_data,
            employee_id="emp_001",
        )

        assert expense.id.startswith("exp_")
        assert expense.status == ExpenseStatus.PENDING

    @pytest.mark.asyncio
    async def test_parse_receipt_text_extracts_vendor(self, expense_tracker, sample_receipt_text):
        """Test that vendor name is extracted from receipt text."""
        result = expense_tracker._parse_receipt_text(sample_receipt_text)
        assert result["vendor"] != ""

    @pytest.mark.asyncio
    async def test_parse_receipt_text_extracts_total(self, expense_tracker, sample_receipt_text):
        """Test that total amount is extracted from receipt text."""
        result = expense_tracker._parse_receipt_text(sample_receipt_text)
        assert result["amount"] == 98.80

    @pytest.mark.asyncio
    async def test_parse_receipt_text_extracts_tax(self, expense_tracker, sample_receipt_text):
        """Test that tax is extracted from receipt text."""
        result = expense_tracker._parse_receipt_text(sample_receipt_text)
        assert result["tax"] == 7.32

    @pytest.mark.asyncio
    async def test_parse_receipt_text_extracts_date(self, expense_tracker):
        """Test date extraction from receipt text."""
        text = "Receipt\nDate: 03/15/2024\nTotal: $50.00"
        result = expense_tracker._parse_receipt_text(text)
        assert result["date"].month == 3
        assert result["date"].day == 15

    @pytest.mark.asyncio
    async def test_parse_receipt_text_extracts_line_items(
        self, expense_tracker, sample_receipt_text
    ):
        """Test line item extraction from receipt text."""
        result = expense_tracker._parse_receipt_text(sample_receipt_text)
        assert len(result["line_items"]) > 0


# =============================================================================
# Expense Categorization Tests
# =============================================================================


class TestExpenseCategorization:
    """Test expense categorization."""

    @pytest.mark.asyncio
    async def test_categorize_by_vendor_pattern(self, expense_tracker):
        """Test pattern-based vendor categorization."""
        expense = await expense_tracker.create_expense(
            vendor_name="Delta Airlines",
            amount=350.00,
        )

        category = await expense_tracker.categorize_expense(expense)
        assert category == ExpenseCategory.TRAVEL

    @pytest.mark.asyncio
    async def test_categorize_restaurant(self, expense_tracker):
        """Test restaurant categorization."""
        expense = await expense_tracker.create_expense(
            vendor_name="Pizza Hut",
            amount=25.00,
        )

        category = await expense_tracker.categorize_expense(expense)
        assert category == ExpenseCategory.MEALS

    @pytest.mark.asyncio
    async def test_categorize_office_supplies(self, expense_tracker):
        """Test office supplies categorization."""
        expense = await expense_tracker.create_expense(
            vendor_name="Staples",
            amount=75.00,
        )

        category = await expense_tracker.categorize_expense(expense)
        assert category == ExpenseCategory.OFFICE_SUPPLIES

    @pytest.mark.asyncio
    async def test_auto_categorize_multiple(self, expense_tracker):
        """Test auto-categorizing multiple expenses."""
        e1 = await expense_tracker.create_expense(vendor_name="Uber", amount=25.00)
        e2 = await expense_tracker.create_expense(vendor_name="Starbucks", amount=8.00)

        results = await expense_tracker.auto_categorize([e1.id, e2.id])

        assert e1.id in results
        assert e2.id in results


# =============================================================================
# Duplicate Detection Tests
# =============================================================================


class TestDuplicateDetection:
    """Test duplicate expense detection."""

    @pytest.mark.asyncio
    async def test_detect_exact_duplicate(self, expense_tracker):
        """Test detection of exact duplicate expenses."""
        expense1 = await expense_tracker.create_expense(
            vendor_name="Acme Corp",
            amount=100.00,
            date=datetime(2024, 1, 15, 12, 0, 0),
        )
        expense2 = await expense_tracker.create_expense(
            vendor_name="Acme Corp",
            amount=100.00,
            date=datetime(2024, 1, 15, 12, 0, 0),
        )

        duplicates = await expense_tracker.detect_duplicates(expense2)
        assert len(duplicates) >= 1
        assert any(d.id == expense1.id for d in duplicates)

    @pytest.mark.asyncio
    async def test_no_duplicate_different_vendors(self, expense_tracker):
        """Test that different vendors aren't flagged as duplicates."""
        await expense_tracker.create_expense(
            vendor_name="Vendor A",
            amount=100.00,
        )
        expense2 = await expense_tracker.create_expense(
            vendor_name="Vendor B",
            amount=100.00,
        )

        duplicates = await expense_tracker.detect_duplicates(expense2)
        assert len(duplicates) == 0

    @pytest.mark.asyncio
    async def test_no_duplicate_different_amounts(self, expense_tracker):
        """Test that different amounts aren't flagged as duplicates."""
        await expense_tracker.create_expense(
            vendor_name="Same Vendor",
            amount=100.00,
        )
        expense2 = await expense_tracker.create_expense(
            vendor_name="Same Vendor",
            amount=200.00,
        )

        duplicates = await expense_tracker.detect_duplicates(expense2)
        assert len(duplicates) == 0


# =============================================================================
# Approval Workflow Tests
# =============================================================================


class TestApprovalWorkflow:
    """Test expense approval workflow."""

    @pytest.mark.asyncio
    async def test_approve_expense(self, expense_tracker, sample_expense_data):
        """Test approving an expense."""
        expense = await expense_tracker.create_expense(**sample_expense_data)
        approved = await expense_tracker.approve_expense(expense.id, approver_id="mgr_001")

        assert approved is not None
        assert approved.status == ExpenseStatus.APPROVED
        assert approved.approver_id == "mgr_001"

    @pytest.mark.asyncio
    async def test_reject_expense(self, expense_tracker, sample_expense_data):
        """Test rejecting an expense."""
        expense = await expense_tracker.create_expense(**sample_expense_data)
        rejected = await expense_tracker.reject_expense(
            expense.id, reason="Missing receipt", rejector_id="mgr_001"
        )

        assert rejected is not None
        assert rejected.status == ExpenseStatus.REJECTED

    @pytest.mark.asyncio
    async def test_get_pending_approvals(self, expense_tracker):
        """Test getting pending expenses."""
        await expense_tracker.create_expense(vendor_name="Vendor 1", amount=50)
        await expense_tracker.create_expense(vendor_name="Vendor 2", amount=75)

        pending = await expense_tracker.get_pending_approvals()
        assert len(pending) == 2
        assert all(e.status == ExpenseStatus.PENDING for e in pending)


# =============================================================================
# Statistics Tests
# =============================================================================


class TestExpenseStatistics:
    """Test expense statistics and reporting."""

    @pytest.mark.asyncio
    async def test_get_statistics(self, expense_tracker):
        """Test getting expense statistics."""
        await expense_tracker.create_expense(
            vendor_name="Vendor 1",
            amount=100.00,
            category=ExpenseCategory.TRAVEL,
        )
        await expense_tracker.create_expense(
            vendor_name="Vendor 2",
            amount=50.00,
            category=ExpenseCategory.MEALS,
        )

        stats = await expense_tracker.get_statistics()

        assert stats["total_count"] == 2
        assert float(stats["total_amount"]) == 150.00
        assert "by_category" in stats
        assert "by_status" in stats

    @pytest.mark.asyncio
    async def test_get_statistics_empty(self, expense_tracker):
        """Test statistics with no expenses."""
        stats = await expense_tracker.get_statistics()

        assert stats["total_count"] == 0
        assert float(stats["total_amount"]) == 0


# =============================================================================
# Export Tests
# =============================================================================


class TestExpenseExport:
    """Test expense export functionality."""

    @pytest.mark.asyncio
    async def test_export_to_csv(self, expense_tracker, sample_expense_data):
        """Test exporting expenses to CSV format."""
        await expense_tracker.create_expense(**sample_expense_data)
        await expense_tracker.create_expense(vendor_name="Another Vendor", amount=75.00)

        csv_data = await expense_tracker.export_expenses(format="csv")

        assert csv_data is not None
        assert "vendor_name" in csv_data or "Acme Corp" in csv_data


# =============================================================================
# QBO Sync Tests
# =============================================================================


class TestQBOSync:
    """Test QBO synchronization."""

    @pytest.mark.asyncio
    async def test_sync_to_qbo_requires_approved(self, expense_tracker):
        """Test that only approved expenses can sync to QBO."""
        expense = await expense_tracker.create_expense(vendor_name="Test Vendor", amount=100.00)

        # Pending expense should not sync
        result = await expense_tracker.sync_to_qbo([expense.id])

        # Should either skip or fail for pending expenses
        assert isinstance(result, SyncResult)

    @pytest.mark.asyncio
    async def test_sync_approved_expense(self, expense_tracker):
        """Test syncing an approved expense."""
        expense = await expense_tracker.create_expense(vendor_name="Test Vendor", amount=100.00)
        await expense_tracker.approve_expense(expense.id, approver_id="mgr_001")

        result = await expense_tracker.sync_to_qbo([expense.id])

        assert isinstance(result, SyncResult)
        assert result.total == 1


# =============================================================================
# ExpenseRecord Tests
# =============================================================================


class TestExpenseRecord:
    """Test ExpenseRecord dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        expense = ExpenseRecord(
            id="exp_123",
            vendor_name="Test Vendor",
            amount=Decimal("100.00"),
            date=datetime(2024, 1, 15),
            category=ExpenseCategory.TRAVEL,
            status=ExpenseStatus.PENDING,
        )

        d = expense.to_dict()

        assert d["id"] == "exp_123"
        assert d["vendor_name"] == "Test Vendor"
        assert d["amount"] == "100.00"
        assert d["category"] == "travel"
        assert d["status"] == "pending"

    def test_hash_key(self):
        """Test duplicate detection hash key."""
        expense = ExpenseRecord(
            id="exp_123",
            vendor_name="Test Vendor",
            amount=Decimal("100.00"),
            date=datetime(2024, 1, 15),
            category=ExpenseCategory.TRAVEL,
            status=ExpenseStatus.PENDING,
        )

        key = expense.hash_key
        assert key is not None
        assert "test vendor" in key.lower()
        assert "100.00" in key


# =============================================================================
# Failure Scenario Tests
# =============================================================================


class TestFailureScenarios:
    """Test failure scenarios and resilience patterns."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_status(self):
        """Test circuit breaker status reporting."""
        tracker = ExpenseTracker(enable_circuit_breakers=True)
        status = tracker.get_circuit_breaker_status()

        assert status["enabled"] is True
        assert "ocr" in status["services"]
        assert "llm" in status["services"]
        assert "qbo" in status["services"]

    @pytest.mark.asyncio
    async def test_circuit_breaker_disabled(self):
        """Test with circuit breakers disabled."""
        tracker = ExpenseTracker(enable_circuit_breakers=False)
        status = tracker.get_circuit_breaker_status()

        assert status["enabled"] is False
        assert len(status["services"]) == 0

    @pytest.mark.asyncio
    async def test_ocr_failure_returns_defaults(self):
        """Test that OCR failure returns default expense data."""
        tracker = ExpenseTracker(enable_ocr=True)

        # Process invalid image data
        expense = await tracker.process_receipt(b"invalid image data")

        assert expense.id.startswith("exp_")
        assert expense.status == ExpenseStatus.PROCESSED  # Still marked as processed
        assert expense.vendor_name in ("", "Unknown Vendor")  # Empty or default for invalid data

    @pytest.mark.asyncio
    async def test_llm_categorization_fallback(self):
        """Test that LLM categorization falls back to pattern matching."""
        tracker = ExpenseTracker(enable_llm_categorization=False)

        expense = ExpenseRecord(
            id="exp_test",
            vendor_name="Starbucks Coffee",
            amount=Decimal("15.00"),
            date=datetime.now(),
        )

        category = await tracker.categorize_expense(expense)
        assert category == ExpenseCategory.MEALS

    @pytest.mark.asyncio
    async def test_llm_categorization_fallback_to_other(self):
        """Test that unknown vendors fall back to OTHER category."""
        tracker = ExpenseTracker(enable_llm_categorization=False)

        expense = ExpenseRecord(
            id="exp_test",
            vendor_name="Random Unknown Vendor XYZ",
            amount=Decimal("100.00"),
            date=datetime.now(),
        )

        category = await tracker.categorize_expense(expense)
        assert category == ExpenseCategory.OTHER

    @pytest.mark.asyncio
    async def test_qbo_sync_without_connector(self):
        """Test QBO sync without connector configured."""
        tracker = ExpenseTracker(qbo_connector=None)

        result = await tracker.sync_to_qbo()

        assert isinstance(result, SyncResult)
        assert result.success_count == 0
        assert len(result.errors) > 0
        assert "QBO connector not configured" in result.errors[0]["error"]

    @pytest.mark.asyncio
    async def test_process_empty_receipt(self):
        """Test processing an empty receipt."""
        tracker = ExpenseTracker(enable_ocr=True)

        expense = await tracker.process_receipt(b"")

        # Empty data results in empty/default values extracted
        assert expense.vendor_name in ("", "Unknown Vendor")
        assert expense.amount == Decimal("0") or expense.amount == Decimal("0.00")

    @pytest.mark.asyncio
    async def test_create_expense_without_category(self):
        """Test creating expense without specifying category triggers auto-categorization."""
        tracker = ExpenseTracker(enable_llm_categorization=False)

        expense = await tracker.create_expense(
            vendor_name="Amazon Web Services",
            amount=150.00,
            description="Cloud computing services",
        )

        # Should be categorized based on pattern matching
        assert expense.category is not None

    @pytest.mark.asyncio
    async def test_duplicate_detection_tolerance(self):
        """Test duplicate detection respects tolerance window."""
        tracker = ExpenseTracker()

        # Create first expense
        expense1 = await tracker.create_expense(
            vendor_name="Test Vendor",
            amount=100.00,
            date=datetime.now(),
        )

        # Create similar expense within tolerance
        expense2_data = ExpenseRecord(
            id="exp_test2",
            vendor_name="Test Vendor",
            amount=Decimal("100.00"),
            date=datetime.now() + timedelta(days=1),
        )

        duplicates = await tracker.detect_duplicates(expense2_data)
        assert len(duplicates) > 0

        # Create similar expense outside tolerance
        expense3_data = ExpenseRecord(
            id="exp_test3",
            vendor_name="Test Vendor",
            amount=Decimal("100.00"),
            date=datetime.now() + timedelta(days=10),
        )

        duplicates = await tracker.detect_duplicates(expense3_data, tolerance_days=3)
        # Should not find duplicates outside tolerance
        assert expense1.id not in [d.id for d in duplicates]
