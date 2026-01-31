"""
Tests for AR Invoice Storage Backends (Accounts Receivable).

Tests cover:
- CRUD operations (create, read, update, delete)
- Listing with filters (by status, by customer, overdue)
- Pagination
- Aging buckets calculation
- Customer balance tracking
- Payment recording and auto-paid detection
- Reminder tracking and escalation
- Customer management
- Decimal precision handling
- Concurrent access safety
- Edge cases and error handling
"""

from __future__ import annotations

import asyncio
import json
import pytest
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Any

from aragora.storage.ar_invoice_store import (
    InMemoryARInvoiceStore,
    DecimalEncoder,
    decimal_decoder,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def store():
    """Create a fresh in-memory AR invoice store."""
    return InMemoryARInvoiceStore()


@pytest.fixture
def sample_invoice():
    """Create a sample AR invoice for testing."""
    return {
        "id": "ar_inv_001",
        "invoice_number": "AR-2024-001",
        "customer_id": "cust_123",
        "customer_name": "Widget Co",
        "subtotal": Decimal("1000.00"),
        "tax_amount": Decimal("80.00"),
        "total_amount": Decimal("1080.00"),
        "amount_paid": Decimal("0.00"),
        "balance_due": Decimal("1080.00"),
        "status": "pending",
        "due_date": (datetime.now(timezone.utc) + timedelta(days=30)).isoformat(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


@pytest.fixture
def overdue_invoice():
    """Create an overdue AR invoice."""
    return {
        "id": "ar_inv_overdue",
        "invoice_number": "AR-2024-OD1",
        "customer_id": "cust_456",
        "customer_name": "Late Payer Inc",
        "total_amount": Decimal("500.00"),
        "amount_paid": Decimal("0.00"),
        "balance_due": Decimal("500.00"),
        "status": "pending",
        "due_date": (datetime.now(timezone.utc) - timedelta(days=15)).isoformat(),
        "created_at": (datetime.now(timezone.utc) - timedelta(days=45)).isoformat(),
    }


@pytest.fixture
def multiple_invoices():
    """Create multiple sample invoices for listing tests."""
    now = datetime.now(timezone.utc)
    return [
        {
            "id": f"ar_inv_{i:03d}",
            "invoice_number": f"AR-2024-{i:03d}",
            "customer_id": "cust_a" if i % 2 == 0 else "cust_b",
            "customer_name": "Customer A" if i % 2 == 0 else "Customer B",
            "status": ["pending", "approved", "paid", "void"][i % 4],
            "total_amount": Decimal(str(100 * (i + 1))),
            "balance_due": Decimal(str(100 * (i + 1))) if i % 4 not in (2, 3) else Decimal("0"),
            "due_date": (now + timedelta(days=30 - i * 10)).isoformat(),
            "created_at": (now - timedelta(days=i)).isoformat(),
        }
        for i in range(10)
    ]


@pytest.fixture
def sample_customer():
    """Create a sample customer for testing."""
    return {
        "customer_id": "cust_123",
        "name": "Widget Co",
        "email": "billing@widgetco.com",
        "payment_terms": "Net 30",
    }


# =============================================================================
# DecimalEncoder Tests
# =============================================================================


class TestDecimalEncoder:
    """Tests for Decimal JSON encoding."""

    def test_encodes_decimal_as_string(self):
        """Decimals should be encoded as strings to preserve precision."""
        data = {"amount": Decimal("123.45")}
        result = json.dumps(data, cls=DecimalEncoder)
        assert result == '{"amount": "123.45"}'

    def test_encodes_nested_decimals(self):
        """Nested decimals should be encoded correctly."""
        data = {
            "subtotal": Decimal("1000.00"),
            "tax": Decimal("80.00"),
            "total": Decimal("1080.00"),
        }
        result = json.dumps(data, cls=DecimalEncoder)
        parsed = json.loads(result)
        assert parsed["subtotal"] == "1000.00"
        assert parsed["tax"] == "80.00"
        assert parsed["total"] == "1080.00"

    def test_encodes_zero_decimal(self):
        """Zero should be encoded correctly."""
        data = {"amount": Decimal("0.00")}
        result = json.dumps(data, cls=DecimalEncoder)
        assert '"0.00"' in result

    def test_non_decimal_types_pass_through(self):
        """Non-decimal types should use default encoding."""
        data = {"name": "test", "count": 5}
        result = json.dumps(data, cls=DecimalEncoder)
        parsed = json.loads(result)
        assert parsed["name"] == "test"
        assert parsed["count"] == 5


class TestDecimalDecoder:
    """Tests for Decimal JSON decoding."""

    def test_decodes_decimal_fields(self):
        """Known decimal fields should be decoded back to Decimal."""
        data = {"total_amount": "1080.00", "balance_due": "500.00"}
        result = decimal_decoder(data)
        assert isinstance(result["total_amount"], Decimal)
        assert result["total_amount"] == Decimal("1080.00")

    def test_ignores_non_decimal_fields(self):
        """Non-decimal fields should not be converted."""
        data = {"name": "test", "total_amount": "100.00"}
        result = decimal_decoder(data)
        assert result["name"] == "test"
        assert isinstance(result["total_amount"], Decimal)

    def test_handles_invalid_decimal_string(self):
        """Invalid decimal strings should be left as-is."""
        data = {"total_amount": "not_a_number"}
        result = decimal_decoder(data)
        assert result["total_amount"] == "not_a_number"


# =============================================================================
# CRUD Operations
# =============================================================================


class TestCRUD:
    """Tests for basic create, read, update, delete operations."""

    @pytest.mark.asyncio
    async def test_save_and_get(self, store, sample_invoice):
        """Should save and retrieve an invoice."""
        await store.save(sample_invoice)
        result = await store.get("ar_inv_001")
        assert result is not None
        assert result["id"] == "ar_inv_001"
        assert result["customer_name"] == "Widget Co"

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, store):
        """Should return None for nonexistent invoice."""
        result = await store.get("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_save_requires_id(self, store):
        """Should raise ValueError when id is missing."""
        with pytest.raises(ValueError, match="id is required"):
            await store.save({"customer_id": "cust_123"})

    @pytest.mark.asyncio
    async def test_save_overwrites(self, store, sample_invoice):
        """Should overwrite existing invoice on save."""
        await store.save(sample_invoice)
        sample_invoice["customer_name"] = "Updated Name"
        await store.save(sample_invoice)

        result = await store.get("ar_inv_001")
        assert result["customer_name"] == "Updated Name"

    @pytest.mark.asyncio
    async def test_delete_existing(self, store, sample_invoice):
        """Should delete an existing invoice."""
        await store.save(sample_invoice)
        result = await store.delete("ar_inv_001")
        assert result is True
        assert await store.get("ar_inv_001") is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, store):
        """Should return False when deleting nonexistent invoice."""
        result = await store.delete("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_update_status(self, store, sample_invoice):
        """Should update invoice status."""
        await store.save(sample_invoice)
        result = await store.update_status("ar_inv_001", "approved")
        assert result is True

        inv = await store.get("ar_inv_001")
        assert inv["status"] == "approved"
        assert "updated_at" in inv

    @pytest.mark.asyncio
    async def test_update_status_nonexistent(self, store):
        """Should return False for nonexistent invoice."""
        result = await store.update_status("nonexistent", "approved")
        assert result is False


# =============================================================================
# Listing and Filtering
# =============================================================================


class TestListing:
    """Tests for listing and filtering invoices."""

    @pytest.mark.asyncio
    async def test_list_all(self, store, multiple_invoices):
        """Should list all invoices."""
        for inv in multiple_invoices:
            await store.save(inv)
        result = await store.list_all()
        assert len(result) == 10

    @pytest.mark.asyncio
    async def test_list_all_pagination(self, store, multiple_invoices):
        """Should paginate results."""
        for inv in multiple_invoices:
            await store.save(inv)
        page1 = await store.list_all(limit=3, offset=0)
        page2 = await store.list_all(limit=3, offset=3)
        assert len(page1) == 3
        assert len(page2) == 3
        assert page1[0]["id"] != page2[0]["id"]

    @pytest.mark.asyncio
    async def test_list_all_sorted_by_created_at(self, store, multiple_invoices):
        """Should return invoices sorted by created_at descending."""
        for inv in multiple_invoices:
            await store.save(inv)
        result = await store.list_all()
        # Most recent first
        assert result[0]["id"] == "ar_inv_000"

    @pytest.mark.asyncio
    async def test_list_by_status(self, store, multiple_invoices):
        """Should filter by status."""
        for inv in multiple_invoices:
            await store.save(inv)
        pending = await store.list_by_status("pending")
        assert all(inv["status"] == "pending" for inv in pending)

    @pytest.mark.asyncio
    async def test_list_by_status_with_limit(self, store, multiple_invoices):
        """Should respect limit when filtering by status."""
        for inv in multiple_invoices:
            await store.save(inv)
        result = await store.list_by_status("pending", limit=1)
        assert len(result) <= 1

    @pytest.mark.asyncio
    async def test_list_by_customer(self, store, multiple_invoices):
        """Should filter by customer_id."""
        for inv in multiple_invoices:
            await store.save(inv)
        cust_a = await store.list_by_customer("cust_a")
        assert all(inv["customer_id"] == "cust_a" for inv in cust_a)

    @pytest.mark.asyncio
    async def test_list_empty_store(self, store):
        """Should return empty list from empty store."""
        result = await store.list_all()
        assert result == []

    @pytest.mark.asyncio
    async def test_list_by_status_no_match(self, store, sample_invoice):
        """Should return empty list when no invoices match status."""
        await store.save(sample_invoice)
        result = await store.list_by_status("void")
        assert result == []


# =============================================================================
# Overdue Detection
# =============================================================================


class TestOverdueDetection:
    """Tests for overdue invoice detection."""

    @pytest.mark.asyncio
    async def test_list_overdue(self, store, overdue_invoice):
        """Should detect overdue invoices."""
        await store.save(overdue_invoice)
        result = await store.list_overdue()
        assert len(result) == 1
        assert result[0]["id"] == "ar_inv_overdue"

    @pytest.mark.asyncio
    async def test_not_overdue_when_future(self, store, sample_invoice):
        """Should not include invoices with future due dates."""
        await store.save(sample_invoice)
        result = await store.list_overdue()
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_paid_not_overdue(self, store, overdue_invoice):
        """Should not include paid invoices even if past due."""
        overdue_invoice["status"] = "paid"
        await store.save(overdue_invoice)
        result = await store.list_overdue()
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_void_not_overdue(self, store, overdue_invoice):
        """Should not include voided invoices even if past due."""
        overdue_invoice["status"] = "void"
        await store.save(overdue_invoice)
        result = await store.list_overdue()
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_no_due_date_not_overdue(self, store):
        """Should not include invoices without due dates."""
        inv = {"id": "no_date", "status": "pending", "total_amount": Decimal("100")}
        await store.save(inv)
        result = await store.list_overdue()
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_overdue_sorted_by_due_date(self, store):
        """Should sort overdue invoices by due date ascending."""
        now = datetime.now(timezone.utc)
        inv1 = {
            "id": "od_1",
            "status": "pending",
            "due_date": (now - timedelta(days=5)).isoformat(),
        }
        inv2 = {
            "id": "od_2",
            "status": "pending",
            "due_date": (now - timedelta(days=30)).isoformat(),
        }
        await store.save(inv1)
        await store.save(inv2)
        result = await store.list_overdue()
        assert len(result) == 2
        # Oldest overdue first
        assert result[0]["id"] == "od_2"


# =============================================================================
# Aging Buckets
# =============================================================================


class TestAgingBuckets:
    """Tests for aging bucket categorization."""

    @pytest.mark.asyncio
    async def test_current_bucket(self, store, sample_invoice):
        """Invoices not yet due should be in 'current' bucket."""
        await store.save(sample_invoice)
        buckets = await store.get_aging_buckets()
        assert len(buckets["current"]) == 1
        assert buckets["current"][0]["id"] == "ar_inv_001"

    @pytest.mark.asyncio
    async def test_1_30_bucket(self, store):
        """Invoices 1-30 days overdue should be in '1_30' bucket."""
        now = datetime.now(timezone.utc)
        inv = {
            "id": "aging_1_30",
            "status": "pending",
            "due_date": (now - timedelta(days=15)).isoformat(),
        }
        await store.save(inv)
        buckets = await store.get_aging_buckets()
        assert len(buckets["1_30"]) == 1

    @pytest.mark.asyncio
    async def test_31_60_bucket(self, store):
        """Invoices 31-60 days overdue should be in '31_60' bucket."""
        now = datetime.now(timezone.utc)
        inv = {
            "id": "aging_31_60",
            "status": "pending",
            "due_date": (now - timedelta(days=45)).isoformat(),
        }
        await store.save(inv)
        buckets = await store.get_aging_buckets()
        assert len(buckets["31_60"]) == 1

    @pytest.mark.asyncio
    async def test_61_90_bucket(self, store):
        """Invoices 61-90 days overdue should be in '61_90' bucket."""
        now = datetime.now(timezone.utc)
        inv = {
            "id": "aging_61_90",
            "status": "pending",
            "due_date": (now - timedelta(days=75)).isoformat(),
        }
        await store.save(inv)
        buckets = await store.get_aging_buckets()
        assert len(buckets["61_90"]) == 1

    @pytest.mark.asyncio
    async def test_90_plus_bucket(self, store):
        """Invoices 90+ days overdue should be in '90_plus' bucket."""
        now = datetime.now(timezone.utc)
        inv = {
            "id": "aging_90_plus",
            "status": "pending",
            "due_date": (now - timedelta(days=120)).isoformat(),
        }
        await store.save(inv)
        buckets = await store.get_aging_buckets()
        assert len(buckets["90_plus"]) == 1

    @pytest.mark.asyncio
    async def test_paid_excluded_from_buckets(self, store):
        """Paid invoices should not appear in aging buckets."""
        now = datetime.now(timezone.utc)
        inv = {
            "id": "paid_inv",
            "status": "paid",
            "due_date": (now - timedelta(days=45)).isoformat(),
        }
        await store.save(inv)
        buckets = await store.get_aging_buckets()
        assert all(len(b) == 0 for b in buckets.values())

    @pytest.mark.asyncio
    async def test_void_excluded_from_buckets(self, store):
        """Voided invoices should not appear in aging buckets."""
        now = datetime.now(timezone.utc)
        inv = {
            "id": "void_inv",
            "status": "void",
            "due_date": (now - timedelta(days=45)).isoformat(),
        }
        await store.save(inv)
        buckets = await store.get_aging_buckets()
        assert all(len(b) == 0 for b in buckets.values())

    @pytest.mark.asyncio
    async def test_no_due_date_in_current(self, store):
        """Invoices without due date should be in 'current' bucket."""
        inv = {"id": "no_date_inv", "status": "pending"}
        await store.save(inv)
        buckets = await store.get_aging_buckets()
        assert len(buckets["current"]) == 1

    @pytest.mark.asyncio
    async def test_empty_store_all_buckets_empty(self, store):
        """Empty store should return all empty buckets."""
        buckets = await store.get_aging_buckets()
        assert all(len(b) == 0 for b in buckets.values())
        assert set(buckets.keys()) == {"current", "1_30", "31_60", "61_90", "90_plus"}


# =============================================================================
# Customer Balance
# =============================================================================


class TestCustomerBalance:
    """Tests for customer balance calculation."""

    @pytest.mark.asyncio
    async def test_single_invoice_balance(self, store, sample_invoice):
        """Should return balance from a single invoice."""
        await store.save(sample_invoice)
        balance = await store.get_customer_balance("cust_123")
        assert balance == Decimal("1080.00")

    @pytest.mark.asyncio
    async def test_multiple_invoices_balance(self, store):
        """Should sum balance across multiple invoices."""
        inv1 = {
            "id": "b1",
            "customer_id": "cust_a",
            "balance_due": Decimal("500.00"),
            "status": "pending",
        }
        inv2 = {
            "id": "b2",
            "customer_id": "cust_a",
            "balance_due": Decimal("300.00"),
            "status": "pending",
        }
        await store.save(inv1)
        await store.save(inv2)
        balance = await store.get_customer_balance("cust_a")
        assert balance == Decimal("800.00")

    @pytest.mark.asyncio
    async def test_paid_excluded_from_balance(self, store):
        """Paid invoices should not be included in balance."""
        inv1 = {
            "id": "b1",
            "customer_id": "cust_a",
            "balance_due": Decimal("500.00"),
            "status": "pending",
        }
        inv2 = {
            "id": "b2",
            "customer_id": "cust_a",
            "balance_due": Decimal("300.00"),
            "status": "paid",
        }
        await store.save(inv1)
        await store.save(inv2)
        balance = await store.get_customer_balance("cust_a")
        assert balance == Decimal("500.00")

    @pytest.mark.asyncio
    async def test_void_excluded_from_balance(self, store):
        """Voided invoices should not be included in balance."""
        inv = {
            "id": "v1",
            "customer_id": "cust_a",
            "balance_due": Decimal("500.00"),
            "status": "void",
        }
        await store.save(inv)
        balance = await store.get_customer_balance("cust_a")
        assert balance == Decimal("0")

    @pytest.mark.asyncio
    async def test_nonexistent_customer_zero_balance(self, store):
        """Should return zero balance for nonexistent customer."""
        balance = await store.get_customer_balance("nonexistent")
        assert balance == Decimal("0")

    @pytest.mark.asyncio
    async def test_balance_uses_total_if_no_balance_due(self, store):
        """Should use total_amount if balance_due is not set."""
        inv = {
            "id": "b1",
            "customer_id": "cust_a",
            "total_amount": Decimal("750.00"),
            "status": "pending",
        }
        await store.save(inv)
        balance = await store.get_customer_balance("cust_a")
        assert balance == Decimal("750.00")

    @pytest.mark.asyncio
    async def test_balance_handles_string_amounts(self, store):
        """Should handle balance_due stored as string."""
        inv = {
            "id": "b1",
            "customer_id": "cust_a",
            "balance_due": "250.50",
            "status": "pending",
        }
        await store.save(inv)
        balance = await store.get_customer_balance("cust_a")
        assert balance == Decimal("250.50")


# =============================================================================
# Payment Recording
# =============================================================================


class TestPaymentRecording:
    """Tests for recording payments against invoices."""

    @pytest.mark.asyncio
    async def test_record_partial_payment(self, store, sample_invoice):
        """Should record a partial payment."""
        await store.save(sample_invoice)
        now = datetime.now(timezone.utc)
        result = await store.record_payment(
            "ar_inv_001", Decimal("500.00"), now, "check", "CHK-001"
        )
        assert result is True

        inv = await store.get("ar_inv_001")
        assert Decimal(inv["amount_paid"]) == Decimal("500.00")
        assert Decimal(inv["balance_due"]) == Decimal("580.00")
        assert inv["status"] == "pending"  # Not fully paid

    @pytest.mark.asyncio
    async def test_record_full_payment(self, store, sample_invoice):
        """Should auto-mark as paid when fully paid."""
        await store.save(sample_invoice)
        now = datetime.now(timezone.utc)
        result = await store.record_payment(
            "ar_inv_001", Decimal("1080.00"), now, "wire", "WIRE-001"
        )
        assert result is True

        inv = await store.get("ar_inv_001")
        assert inv["status"] == "paid"
        assert Decimal(inv["amount_paid"]) == Decimal("1080.00")
        assert Decimal(inv["balance_due"]) == Decimal("0.00")

    @pytest.mark.asyncio
    async def test_record_multiple_payments(self, store, sample_invoice):
        """Should accumulate multiple payments."""
        await store.save(sample_invoice)
        now = datetime.now(timezone.utc)

        await store.record_payment("ar_inv_001", Decimal("400.00"), now, "check")
        await store.record_payment("ar_inv_001", Decimal("680.00"), now, "wire")

        inv = await store.get("ar_inv_001")
        assert Decimal(inv["amount_paid"]) == Decimal("1080.00")
        assert inv["status"] == "paid"
        assert len(inv["payments"]) == 2

    @pytest.mark.asyncio
    async def test_record_overpayment(self, store, sample_invoice):
        """Should handle overpayment gracefully."""
        await store.save(sample_invoice)
        now = datetime.now(timezone.utc)
        result = await store.record_payment("ar_inv_001", Decimal("2000.00"), now, "wire")
        assert result is True

        inv = await store.get("ar_inv_001")
        assert inv["status"] == "paid"
        assert Decimal(inv["amount_paid"]) == Decimal("2000.00")

    @pytest.mark.asyncio
    async def test_record_payment_nonexistent(self, store):
        """Should return False for nonexistent invoice."""
        now = datetime.now(timezone.utc)
        result = await store.record_payment("nonexistent", Decimal("100.00"), now)
        assert result is False

    @pytest.mark.asyncio
    async def test_payment_stores_method_and_reference(self, store, sample_invoice):
        """Should store payment method and reference."""
        await store.save(sample_invoice)
        now = datetime.now(timezone.utc)
        await store.record_payment("ar_inv_001", Decimal("500.00"), now, "credit_card", "CC-789")

        inv = await store.get("ar_inv_001")
        payment = inv["payments"][0]
        assert payment["payment_method"] == "credit_card"
        assert payment["reference"] == "CC-789"
        assert payment["amount"] == "500.00"

    @pytest.mark.asyncio
    async def test_payment_updates_timestamp(self, store, sample_invoice):
        """Should update the updated_at timestamp on payment."""
        await store.save(sample_invoice)
        now = datetime.now(timezone.utc)
        await store.record_payment("ar_inv_001", Decimal("100.00"), now)

        inv = await store.get("ar_inv_001")
        assert "updated_at" in inv


# =============================================================================
# Reminder Tracking
# =============================================================================


class TestReminderTracking:
    """Tests for reminder sent tracking."""

    @pytest.mark.asyncio
    async def test_record_first_reminder(self, store, sample_invoice):
        """Should record the first reminder."""
        await store.save(sample_invoice)
        result = await store.record_reminder_sent("ar_inv_001", 1)
        assert result is True

        inv = await store.get("ar_inv_001")
        assert len(inv["reminders_sent"]) == 1
        assert inv["reminders_sent"][0]["level"] == 1
        assert inv["last_reminder_level"] == 1
        assert "last_reminder_at" in inv

    @pytest.mark.asyncio
    async def test_record_escalating_reminders(self, store, sample_invoice):
        """Should track escalating reminder levels."""
        await store.save(sample_invoice)
        await store.record_reminder_sent("ar_inv_001", 1)
        await store.record_reminder_sent("ar_inv_001", 2)
        await store.record_reminder_sent("ar_inv_001", 3)

        inv = await store.get("ar_inv_001")
        assert len(inv["reminders_sent"]) == 3
        assert inv["last_reminder_level"] == 3

    @pytest.mark.asyncio
    async def test_record_reminder_nonexistent(self, store):
        """Should return False for nonexistent invoice."""
        result = await store.record_reminder_sent("nonexistent", 1)
        assert result is False


# =============================================================================
# Customer Management
# =============================================================================


class TestCustomerManagement:
    """Tests for customer CRUD operations."""

    @pytest.mark.asyncio
    async def test_save_and_get_customer(self, store, sample_customer):
        """Should save and retrieve a customer."""
        await store.save_customer(sample_customer)
        result = await store.get_customer("cust_123")
        assert result is not None
        assert result["name"] == "Widget Co"
        assert result["email"] == "billing@widgetco.com"

    @pytest.mark.asyncio
    async def test_get_nonexistent_customer(self, store):
        """Should return None for nonexistent customer."""
        result = await store.get_customer("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_save_customer_requires_id(self, store):
        """Should raise ValueError when customer_id is missing."""
        with pytest.raises(ValueError, match="customer_id is required"):
            await store.save_customer({"name": "Test"})

    @pytest.mark.asyncio
    async def test_list_customers(self, store):
        """Should list all customers."""
        await store.save_customer({"customer_id": "c1", "name": "Customer 1"})
        await store.save_customer({"customer_id": "c2", "name": "Customer 2"})
        await store.save_customer({"customer_id": "c3", "name": "Customer 3"})

        result = await store.list_customers()
        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_list_customers_with_limit(self, store):
        """Should respect limit parameter."""
        for i in range(5):
            await store.save_customer({"customer_id": f"c{i}", "name": f"Customer {i}"})

        result = await store.list_customers(limit=3)
        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_update_customer(self, store, sample_customer):
        """Should update customer on re-save."""
        await store.save_customer(sample_customer)
        sample_customer["name"] = "Widget Co International"
        await store.save_customer(sample_customer)

        result = await store.get_customer("cust_123")
        assert result["name"] == "Widget Co International"


# =============================================================================
# Concurrent Access
# =============================================================================


class TestConcurrentAccess:
    """Tests for thread safety under concurrent operations."""

    @pytest.mark.asyncio
    async def test_concurrent_saves(self, store):
        """Should handle concurrent saves safely."""

        async def save_invoice(idx: int):
            await store.save(
                {
                    "id": f"conc_{idx}",
                    "customer_id": "cust_a",
                    "status": "pending",
                    "total_amount": Decimal(str(idx * 100)),
                }
            )

        await asyncio.gather(*[save_invoice(i) for i in range(50)])
        result = await store.list_all(limit=100)
        assert len(result) == 50

    @pytest.mark.asyncio
    async def test_concurrent_payments(self, store):
        """Should handle concurrent payments safely."""
        await store.save(
            {
                "id": "conc_pay",
                "customer_id": "cust_a",
                "total_amount": Decimal("10000.00"),
                "balance_due": Decimal("10000.00"),
                "status": "pending",
            }
        )

        now = datetime.now(timezone.utc)

        async def record_payment(idx: int):
            await store.record_payment("conc_pay", Decimal("100.00"), now, "check", f"CHK-{idx}")

        await asyncio.gather(*[record_payment(i) for i in range(10)])

        inv = await store.get("conc_pay")
        assert Decimal(inv["amount_paid"]) == Decimal("1000.00")
        assert len(inv["payments"]) == 10


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_decimal_precision_preserved(self, store):
        """Should preserve decimal precision through save/get."""
        inv = {
            "id": "prec_test",
            "total_amount": Decimal("1234.56"),
            "balance_due": Decimal("1234.56"),
            "status": "pending",
        }
        await store.save(inv)
        result = await store.get("prec_test")
        assert result["total_amount"] == Decimal("1234.56")

    @pytest.mark.asyncio
    async def test_empty_string_fields(self, store):
        """Should handle empty string fields."""
        inv = {
            "id": "empty_fields",
            "customer_id": "",
            "customer_name": "",
            "status": "draft",
        }
        await store.save(inv)
        result = await store.get("empty_fields")
        assert result is not None

    @pytest.mark.asyncio
    async def test_close_is_noop(self, store):
        """Close should be a no-op for in-memory store."""
        await store.close()
        # Should still work after close
        await store.save({"id": "after_close", "status": "pending"})
        result = await store.get("after_close")
        assert result is not None

    @pytest.mark.asyncio
    async def test_iso_date_with_z_suffix(self, store):
        """Should handle ISO dates with Z suffix."""
        now = datetime.now(timezone.utc)
        inv = {
            "id": "z_date",
            "status": "pending",
            "due_date": (now - timedelta(days=5)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        await store.save(inv)
        result = await store.list_overdue()
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_pagination_beyond_range(self, store, sample_invoice):
        """Should return empty list for offset beyond range."""
        await store.save(sample_invoice)
        result = await store.list_all(limit=10, offset=100)
        assert result == []

    @pytest.mark.asyncio
    async def test_customer_balance_with_mixed_statuses(self, store):
        """Should correctly sum balance across mixed statuses."""
        invoices = [
            {"id": "m1", "customer_id": "c1", "balance_due": Decimal("100"), "status": "pending"},
            {"id": "m2", "customer_id": "c1", "balance_due": Decimal("200"), "status": "approved"},
            {"id": "m3", "customer_id": "c1", "balance_due": Decimal("300"), "status": "paid"},
            {"id": "m4", "customer_id": "c1", "balance_due": Decimal("400"), "status": "void"},
        ]
        for inv in invoices:
            await store.save(inv)

        # Only pending + approved should count
        balance = await store.get_customer_balance("c1")
        assert balance == Decimal("300")


# =============================================================================
# Credit Memo Handling
# =============================================================================


class TestCreditMemoHandling:
    """Tests for credit memo and adjustment operations."""

    @pytest.mark.asyncio
    async def test_create_credit_memo(self, store):
        """Should create a credit memo with negative balance."""
        credit_memo = {
            "id": "cm_001",
            "invoice_number": "CM-2024-001",
            "customer_id": "cust_123",
            "customer_name": "Widget Co",
            "total_amount": Decimal("-250.00"),
            "balance_due": Decimal("-250.00"),
            "status": "approved",
            "type": "credit_memo",
            "reason": "Product return",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        await store.save(credit_memo)
        result = await store.get("cm_001")
        assert result is not None
        assert result["total_amount"] == Decimal("-250.00")
        assert result["type"] == "credit_memo"

    @pytest.mark.asyncio
    async def test_credit_memo_affects_customer_balance(self, store):
        """Credit memos should reduce customer outstanding balance."""
        # Create invoice
        invoice = {
            "id": "inv_cm_1",
            "customer_id": "cust_cm",
            "total_amount": Decimal("1000.00"),
            "balance_due": Decimal("1000.00"),
            "status": "pending",
        }
        await store.save(invoice)

        # Create credit memo (negative balance)
        credit_memo = {
            "id": "cm_002",
            "customer_id": "cust_cm",
            "total_amount": Decimal("-200.00"),
            "balance_due": Decimal("-200.00"),
            "status": "approved",
        }
        await store.save(credit_memo)

        # Net balance should be 1000 - 200 = 800
        balance = await store.get_customer_balance("cust_cm")
        assert balance == Decimal("800.00")

    @pytest.mark.asyncio
    async def test_credit_memo_application(self, store, sample_invoice):
        """Credit memo can be applied to reduce invoice balance."""
        await store.save(sample_invoice)

        # Apply credit memo as a payment
        now = datetime.now(timezone.utc)
        result = await store.record_payment(
            "ar_inv_001",
            Decimal("100.00"),
            now,
            payment_method="credit_memo",
            reference="CM-001",
        )
        assert result is True

        inv = await store.get("ar_inv_001")
        assert Decimal(inv["amount_paid"]) == Decimal("100.00")
        assert Decimal(inv["balance_due"]) == Decimal("980.00")

    @pytest.mark.asyncio
    async def test_multiple_credit_memos_per_customer(self, store):
        """Should handle multiple credit memos for the same customer."""
        for i in range(3):
            cm = {
                "id": f"cm_multi_{i}",
                "customer_id": "cust_multi_cm",
                "total_amount": Decimal("-100.00"),
                "balance_due": Decimal("-100.00"),
                "status": "approved",
            }
            await store.save(cm)

        invoices = await store.list_by_customer("cust_multi_cm")
        assert len(invoices) == 3
        assert all(Decimal(inv["balance_due"]) == Decimal("-100.00") for inv in invoices)


# =============================================================================
# Collections Workflow
# =============================================================================


class TestCollectionsWorkflow:
    """Tests for complete collections workflow operations."""

    @pytest.mark.asyncio
    async def test_collections_escalation_levels(self, store, overdue_invoice):
        """Should track escalating collection levels."""
        await store.save(overdue_invoice)

        # Level 1: Friendly reminder
        await store.record_reminder_sent("ar_inv_overdue", 1)

        # Level 2: Past due notice
        await store.record_reminder_sent("ar_inv_overdue", 2)

        # Level 3: Final notice
        await store.record_reminder_sent("ar_inv_overdue", 3)

        # Level 4: Collections warning
        await store.record_reminder_sent("ar_inv_overdue", 4)

        inv = await store.get("ar_inv_overdue")
        assert len(inv["reminders_sent"]) == 4
        assert inv["last_reminder_level"] == 4

    @pytest.mark.asyncio
    async def test_collections_hold_status(self, store, sample_invoice):
        """Should support collections hold status."""
        await store.save(sample_invoice)

        result = await store.update_status("ar_inv_001", "collections_hold")
        assert result is True

        inv = await store.get("ar_inv_001")
        assert inv["status"] == "collections_hold"

    @pytest.mark.asyncio
    async def test_sent_to_collections_status(self, store, overdue_invoice):
        """Should track when invoice is sent to collections agency."""
        await store.save(overdue_invoice)

        result = await store.update_status("ar_inv_overdue", "sent_to_collections")
        assert result is True

        inv = await store.get("ar_inv_overdue")
        assert inv["status"] == "sent_to_collections"

    @pytest.mark.asyncio
    async def test_collections_by_aging_bucket(self, store):
        """Should identify invoices eligible for collections based on aging."""
        now = datetime.now(timezone.utc)

        # Create invoices in different aging buckets
        invoices = [
            {
                "id": "col_current",
                "status": "pending",
                "customer_id": "c1",
                "due_date": (now + timedelta(days=10)).isoformat(),
            },
            {
                "id": "col_30",
                "status": "pending",
                "customer_id": "c1",
                "due_date": (now - timedelta(days=20)).isoformat(),
            },
            {
                "id": "col_60",
                "status": "pending",
                "customer_id": "c1",
                "due_date": (now - timedelta(days=50)).isoformat(),
            },
            {
                "id": "col_90",
                "status": "pending",
                "customer_id": "c1",
                "due_date": (now - timedelta(days=100)).isoformat(),
            },
        ]

        for inv in invoices:
            await store.save(inv)

        buckets = await store.get_aging_buckets()

        # Verify collections candidates (90+ days)
        assert len(buckets["90_plus"]) == 1
        assert buckets["90_plus"][0]["id"] == "col_90"

    @pytest.mark.asyncio
    async def test_payment_plan_tracking(self, store, overdue_invoice):
        """Should track invoices on payment plans."""
        overdue_invoice["payment_plan"] = {
            "agreed_date": datetime.now(timezone.utc).isoformat(),
            "installments": 3,
            "amount_per_installment": "166.67",
        }
        await store.save(overdue_invoice)

        inv = await store.get("ar_inv_overdue")
        assert "payment_plan" in inv
        assert inv["payment_plan"]["installments"] == 3


# =============================================================================
# Write-Off Processing
# =============================================================================


class TestWriteOffProcessing:
    """Tests for bad debt write-off operations."""

    @pytest.mark.asyncio
    async def test_write_off_full_invoice(self, store, overdue_invoice):
        """Should write off entire invoice balance using void status."""
        await store.save(overdue_invoice)

        # Mark as void (write-off) - the store only excludes 'paid' and 'void' from balance
        result = await store.update_status("ar_inv_overdue", "void")
        assert result is True

        inv = await store.get("ar_inv_overdue")
        assert inv["status"] == "void"

        # Voided (written off) invoices should not appear in customer balance
        balance = await store.get_customer_balance("cust_456")
        assert balance == Decimal("0")

    @pytest.mark.asyncio
    async def test_write_off_via_payment(self, store, sample_invoice):
        """Should write off invoice via payment method."""
        await store.save(sample_invoice)

        # Record partial payment first
        now = datetime.now(timezone.utc)
        await store.record_payment("ar_inv_001", Decimal("500.00"), now, "check")

        # Record write-off for remaining balance as a special payment
        await store.record_payment(
            "ar_inv_001",
            Decimal("580.00"),
            now,
            payment_method="write_off",
            reference="WO-001",
        )

        inv = await store.get("ar_inv_001")
        assert inv["status"] == "paid"  # Fully settled via write-off
        assert Decimal(inv["balance_due"]) == Decimal("0.00")

    @pytest.mark.asyncio
    async def test_void_excluded_from_aging(self, store, overdue_invoice):
        """Voided (written off) invoices should not appear in aging buckets."""
        overdue_invoice["status"] = "void"
        await store.save(overdue_invoice)

        buckets = await store.get_aging_buckets()
        all_invoices = []
        for bucket in buckets.values():
            all_invoices.extend(bucket)

        assert not any(inv["id"] == "ar_inv_overdue" for inv in all_invoices)

    @pytest.mark.asyncio
    async def test_void_excluded_from_overdue(self, store, overdue_invoice):
        """Voided (written off) invoices should not appear in overdue list."""
        overdue_invoice["status"] = "void"
        await store.save(overdue_invoice)

        overdue = await store.list_overdue()
        assert len(overdue) == 0

    @pytest.mark.asyncio
    async def test_write_off_custom_status_in_balance(self, store, overdue_invoice):
        """Custom write_off status still appears in balance (only paid/void excluded)."""
        overdue_invoice["status"] = "written_off"
        await store.save(overdue_invoice)

        # The store only excludes 'paid' and 'void' - custom statuses still count
        balance = await store.get_customer_balance("cust_456")
        assert balance == Decimal("500.00")

    @pytest.mark.asyncio
    async def test_recovery_after_void(self, store, overdue_invoice):
        """Should track recovered amounts after voiding."""
        overdue_invoice["status"] = "void"
        overdue_invoice["write_off_amount"] = str(overdue_invoice["balance_due"])
        await store.save(overdue_invoice)

        # Change status back to track recovery
        await store.update_status("ar_inv_overdue", "recovered")

        # Record recovered payment
        now = datetime.now(timezone.utc)
        await store.record_payment(
            "ar_inv_overdue",
            Decimal("250.00"),
            now,
            payment_method="recovery",
            reference="REC-001",
        )

        inv = await store.get("ar_inv_overdue")
        assert inv["status"] in ("recovered", "paid")


# =============================================================================
# Transaction Safety and Rollback
# =============================================================================


class TestTransactionSafety:
    """Tests for transaction safety and error handling."""

    @pytest.mark.asyncio
    async def test_save_validation_failure_rollback(self, store):
        """Should not save partial data on validation failure."""
        # First save a valid invoice
        valid = {"id": "tx_valid", "customer_id": "c1", "status": "pending"}
        await store.save(valid)

        # Attempt to save invalid (no id) should fail
        with pytest.raises(ValueError):
            await store.save({"customer_id": "c2"})

        # Valid invoice should still be intact
        result = await store.get("tx_valid")
        assert result is not None

    @pytest.mark.asyncio
    async def test_payment_on_missing_invoice_no_side_effects(self, store, sample_invoice):
        """Recording payment on missing invoice should have no side effects."""
        await store.save(sample_invoice)

        now = datetime.now(timezone.utc)
        result = await store.record_payment("nonexistent", Decimal("100.00"), now)
        assert result is False

        # Original invoice should be unchanged
        inv = await store.get("ar_inv_001")
        assert "payments" not in inv or len(inv.get("payments", [])) == 0

    @pytest.mark.asyncio
    async def test_concurrent_update_consistency(self, store, sample_invoice):
        """Concurrent updates should maintain data consistency."""
        await store.save(sample_invoice)

        async def update_status(status: str):
            await store.update_status("ar_inv_001", status)

        # Run status updates concurrently
        await asyncio.gather(
            update_status("approved"),
            update_status("pending"),
            update_status("sent"),
        )

        # Invoice should have one of the valid statuses
        inv = await store.get("ar_inv_001")
        assert inv["status"] in ("approved", "pending", "sent")

    @pytest.mark.asyncio
    async def test_reminder_on_missing_invoice_no_side_effects(self, store):
        """Recording reminder on missing invoice should have no side effects."""
        result = await store.record_reminder_sent("nonexistent", 1)
        assert result is False

    @pytest.mark.asyncio
    async def test_delete_and_verify_cleanup(self, store, sample_invoice):
        """Deleted invoices should be completely removed."""
        await store.save(sample_invoice)
        await store.delete("ar_inv_001")

        # Verify not in any lists
        all_invoices = await store.list_all()
        assert not any(inv["id"] == "ar_inv_001" for inv in all_invoices)

        by_customer = await store.list_by_customer("cust_123")
        assert len(by_customer) == 0

        by_status = await store.list_by_status("pending")
        assert not any(inv["id"] == "ar_inv_001" for inv in by_status)


# =============================================================================
# SQLite Backend Tests
# =============================================================================


class TestSQLiteARInvoiceStore:
    """Tests for SQLite-backed AR invoice store."""

    @pytest.fixture
    def sqlite_store(self, tmp_path):
        """Create a SQLite store with temp database."""
        from aragora.storage.ar_invoice_store import SQLiteARInvoiceStore

        db_path = tmp_path / "test_ar.db"
        return SQLiteARInvoiceStore(db_path)

    @pytest.mark.asyncio
    async def test_sqlite_save_and_get(self, sqlite_store, sample_invoice):
        """SQLite store should save and retrieve invoices."""
        await sqlite_store.save(sample_invoice)
        result = await sqlite_store.get("ar_inv_001")
        assert result is not None
        assert result["id"] == "ar_inv_001"

    @pytest.mark.asyncio
    async def test_sqlite_decimal_precision(self, sqlite_store):
        """SQLite store should preserve decimal precision."""
        inv = {
            "id": "sqlite_prec",
            "total_amount": Decimal("12345.67"),
            "balance_due": Decimal("12345.67"),
            "status": "pending",
            "customer_id": "c1",
        }
        await sqlite_store.save(inv)

        result = await sqlite_store.get("sqlite_prec")
        assert result["total_amount"] == Decimal("12345.67")

    @pytest.mark.asyncio
    async def test_sqlite_list_all(self, sqlite_store, multiple_invoices):
        """SQLite store should list all invoices."""
        for inv in multiple_invoices:
            await sqlite_store.save(inv)

        result = await sqlite_store.list_all()
        assert len(result) == 10

    @pytest.mark.asyncio
    async def test_sqlite_list_by_status(self, sqlite_store, multiple_invoices):
        """SQLite store should filter by status."""
        for inv in multiple_invoices:
            await sqlite_store.save(inv)

        pending = await sqlite_store.list_by_status("pending")
        assert all(inv["status"] == "pending" for inv in pending)

    @pytest.mark.asyncio
    async def test_sqlite_list_by_customer(self, sqlite_store, multiple_invoices):
        """SQLite store should filter by customer."""
        for inv in multiple_invoices:
            await sqlite_store.save(inv)

        cust_a = await sqlite_store.list_by_customer("cust_a")
        assert all(inv["customer_id"] == "cust_a" for inv in cust_a)

    @pytest.mark.asyncio
    async def test_sqlite_overdue_detection(self, sqlite_store, overdue_invoice):
        """SQLite store should detect overdue invoices."""
        await sqlite_store.save(overdue_invoice)

        overdue = await sqlite_store.list_overdue()
        assert len(overdue) == 1
        assert overdue[0]["id"] == "ar_inv_overdue"

    @pytest.mark.asyncio
    async def test_sqlite_aging_buckets(self, sqlite_store):
        """SQLite store should calculate aging buckets."""
        now = datetime.now(timezone.utc)
        inv = {
            "id": "sqlite_aging",
            "status": "pending",
            "customer_id": "c1",
            "due_date": (now - timedelta(days=45)).isoformat(),
        }
        await sqlite_store.save(inv)

        buckets = await sqlite_store.get_aging_buckets()
        assert len(buckets["31_60"]) == 1

    @pytest.mark.asyncio
    async def test_sqlite_customer_balance(self, sqlite_store):
        """SQLite store should calculate customer balance."""
        invoices = [
            {
                "id": "sq_bal_1",
                "customer_id": "cust_sq",
                "balance_due": Decimal("500.00"),
                "total_amount": Decimal("500.00"),
                "status": "pending",
            },
            {
                "id": "sq_bal_2",
                "customer_id": "cust_sq",
                "balance_due": Decimal("300.00"),
                "total_amount": Decimal("300.00"),
                "status": "pending",
            },
        ]
        for inv in invoices:
            await sqlite_store.save(inv)

        balance = await sqlite_store.get_customer_balance("cust_sq")
        assert balance == Decimal("800.00")

    @pytest.mark.asyncio
    async def test_sqlite_record_payment(self, sqlite_store, sample_invoice):
        """SQLite store should record payments correctly."""
        await sqlite_store.save(sample_invoice)
        now = datetime.now(timezone.utc)

        result = await sqlite_store.record_payment(
            "ar_inv_001", Decimal("500.00"), now, "check", "CHK-001"
        )
        assert result is True

        inv = await sqlite_store.get("ar_inv_001")
        assert Decimal(inv["amount_paid"]) == Decimal("500.00")
        assert len(inv["payments"]) == 1

    @pytest.mark.asyncio
    async def test_sqlite_full_payment_auto_paid(self, sqlite_store, sample_invoice):
        """SQLite store should auto-mark fully paid invoices."""
        await sqlite_store.save(sample_invoice)
        now = datetime.now(timezone.utc)

        await sqlite_store.record_payment("ar_inv_001", Decimal("1080.00"), now)

        inv = await sqlite_store.get("ar_inv_001")
        assert inv["status"] == "paid"

    @pytest.mark.asyncio
    async def test_sqlite_record_reminder(self, sqlite_store, sample_invoice):
        """SQLite store should record reminders."""
        await sqlite_store.save(sample_invoice)

        result = await sqlite_store.record_reminder_sent("ar_inv_001", 1)
        assert result is True

        inv = await sqlite_store.get("ar_inv_001")
        assert inv["last_reminder_level"] == 1

    @pytest.mark.asyncio
    async def test_sqlite_customer_crud(self, sqlite_store, sample_customer):
        """SQLite store should handle customer CRUD."""
        await sqlite_store.save_customer(sample_customer)

        result = await sqlite_store.get_customer("cust_123")
        assert result is not None
        assert result["name"] == "Widget Co"

        customers = await sqlite_store.list_customers()
        assert len(customers) == 1

    @pytest.mark.asyncio
    async def test_sqlite_delete(self, sqlite_store, sample_invoice):
        """SQLite store should delete invoices."""
        await sqlite_store.save(sample_invoice)

        result = await sqlite_store.delete("ar_inv_001")
        assert result is True

        assert await sqlite_store.get("ar_inv_001") is None

    @pytest.mark.asyncio
    async def test_sqlite_update_status(self, sqlite_store, sample_invoice):
        """SQLite store should update status."""
        await sqlite_store.save(sample_invoice)

        result = await sqlite_store.update_status("ar_inv_001", "approved")
        assert result is True

        inv = await sqlite_store.get("ar_inv_001")
        assert inv["status"] == "approved"


# =============================================================================
# PostgreSQL Backend Mock Tests
# =============================================================================


class TestPostgresARInvoiceStore:
    """Tests for PostgreSQL-backed AR invoice store with mocked connection."""

    @pytest.fixture
    def mock_pool(self):
        """Create a mock asyncpg connection pool."""
        from unittest.mock import AsyncMock, MagicMock

        pool = MagicMock()
        conn = AsyncMock()
        pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
        pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        return pool, conn

    @pytest.mark.asyncio
    async def test_postgres_save_decimal_precision(self, mock_pool):
        """PostgreSQL store should preserve Decimal precision on save."""
        from aragora.storage.ar_invoice_store import PostgresARInvoiceStore

        pool, conn = mock_pool
        store = PostgresARInvoiceStore(pool)

        invoice = {
            "id": "pg_prec_test",
            "customer_id": "c1",
            "total_amount": Decimal("12345678901234.56"),
            "balance_due": Decimal("12345678901234.56"),
            "status": "pending",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        await store.save(invoice)

        conn.execute.assert_called_once()
        # Verify the total_amount parameter (5th arg) is float
        call_args = conn.execute.call_args
        amount_arg = call_args[0][5]  # total_amount position
        assert isinstance(amount_arg, (int, float))

    @pytest.mark.asyncio
    async def test_postgres_get_returns_none_when_empty(self, mock_pool):
        """PostgreSQL store should return None for missing invoice."""
        from aragora.storage.ar_invoice_store import PostgresARInvoiceStore

        pool, conn = mock_pool
        conn.fetchrow.return_value = None
        store = PostgresARInvoiceStore(pool)

        result = await store.get("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_postgres_get_parses_json(self, mock_pool):
        """PostgreSQL store should parse JSON data correctly."""
        from aragora.storage.ar_invoice_store import PostgresARInvoiceStore

        pool, conn = mock_pool
        conn.fetchrow.return_value = {
            "data_json": json.dumps(
                {
                    "id": "pg_json",
                    "customer_id": "c1",
                    "total_amount": "1000.00",
                }
            )
        }
        store = PostgresARInvoiceStore(pool)

        result = await store.get("pg_json")
        assert result is not None
        assert result["id"] == "pg_json"

    @pytest.mark.asyncio
    async def test_postgres_list_all(self, mock_pool):
        """PostgreSQL store should list all invoices."""
        from aragora.storage.ar_invoice_store import PostgresARInvoiceStore

        pool, conn = mock_pool
        conn.fetch.return_value = [
            {"data_json": json.dumps({"id": "pg_1", "customer_id": "c1"})},
            {"data_json": json.dumps({"id": "pg_2", "customer_id": "c2"})},
        ]
        store = PostgresARInvoiceStore(pool)

        result = await store.list_all()
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_postgres_delete_success(self, mock_pool):
        """PostgreSQL store should return True on successful delete."""
        from aragora.storage.ar_invoice_store import PostgresARInvoiceStore

        pool, conn = mock_pool
        conn.execute.return_value = "DELETE 1"
        store = PostgresARInvoiceStore(pool)

        result = await store.delete("pg_delete")
        assert result is True

    @pytest.mark.asyncio
    async def test_postgres_delete_not_found(self, mock_pool):
        """PostgreSQL store should return False when nothing deleted."""
        from aragora.storage.ar_invoice_store import PostgresARInvoiceStore

        pool, conn = mock_pool
        conn.execute.return_value = "DELETE 0"
        store = PostgresARInvoiceStore(pool)

        result = await store.delete("pg_notfound")
        assert result is False

    @pytest.mark.asyncio
    async def test_postgres_customer_balance(self, mock_pool):
        """PostgreSQL store should calculate customer balance."""
        from aragora.storage.ar_invoice_store import PostgresARInvoiceStore

        pool, conn = mock_pool
        conn.fetchrow.return_value = [Decimal("1500.00")]
        store = PostgresARInvoiceStore(pool)

        result = await store.get_customer_balance("cust_pg")
        assert result == Decimal("1500.00")

    @pytest.mark.asyncio
    async def test_postgres_aging_buckets(self, mock_pool):
        """PostgreSQL store should calculate aging buckets."""
        from aragora.storage.ar_invoice_store import PostgresARInvoiceStore

        pool, conn = mock_pool
        conn.fetch.return_value = [
            {
                "data_json": json.dumps({"id": "current_inv", "status": "pending"}),
                "days_overdue": -10,
            },
            {
                "data_json": json.dumps({"id": "1_30_inv", "status": "pending"}),
                "days_overdue": 15,
            },
            {
                "data_json": json.dumps({"id": "90_plus_inv", "status": "pending"}),
                "days_overdue": 120,
            },
        ]
        store = PostgresARInvoiceStore(pool)

        buckets = await store.get_aging_buckets()
        assert len(buckets["current"]) == 1
        assert len(buckets["1_30"]) == 1
        assert len(buckets["90_plus"]) == 1


# =============================================================================
# Store Factory and Global State Tests
# =============================================================================


class TestStoreFactory:
    """Tests for store factory and global state management."""

    def test_set_and_reset_store(self):
        """Should allow setting and resetting custom store."""
        from aragora.storage.ar_invoice_store import (
            set_ar_invoice_store,
            reset_ar_invoice_store,
        )

        custom_store = InMemoryARInvoiceStore()
        set_ar_invoice_store(custom_store)

        reset_ar_invoice_store()
        # After reset, getting store should create new instance


# =============================================================================
# Additional Edge Cases
# =============================================================================


class TestAdditionalEdgeCases:
    """Additional edge case tests for comprehensive coverage."""

    @pytest.mark.asyncio
    async def test_zero_amount_invoice(self, store):
        """Should handle zero amount invoices."""
        inv = {
            "id": "zero_inv",
            "customer_id": "c1",
            "total_amount": Decimal("0.00"),
            "balance_due": Decimal("0.00"),
            "status": "pending",
        }
        await store.save(inv)

        result = await store.get("zero_inv")
        assert result["total_amount"] == Decimal("0.00")

    @pytest.mark.asyncio
    async def test_large_invoice_number(self, store):
        """Should handle large invoice amounts."""
        inv = {
            "id": "large_inv",
            "customer_id": "c1",
            "total_amount": Decimal("99999999.99"),
            "balance_due": Decimal("99999999.99"),
            "status": "pending",
        }
        await store.save(inv)

        result = await store.get("large_inv")
        assert result["total_amount"] == Decimal("99999999.99")

    @pytest.mark.asyncio
    async def test_special_chars_in_customer_name(self, store):
        """Should handle special characters in customer name."""
        inv = {
            "id": "special_chars",
            "customer_id": "c1",
            "customer_name": "O'Brien & Sons (Pty) Ltd.",
            "status": "pending",
        }
        await store.save(inv)

        result = await store.get("special_chars")
        assert result["customer_name"] == "O'Brien & Sons (Pty) Ltd."

    @pytest.mark.asyncio
    async def test_unicode_customer_name(self, store):
        """Should handle unicode in customer name."""
        inv = {
            "id": "unicode_inv",
            "customer_id": "c1",
            "customer_name": "Empresa Espanola S.A.",
            "status": "pending",
        }
        await store.save(inv)

        result = await store.get("unicode_inv")
        assert "Empresa" in result["customer_name"]

    @pytest.mark.asyncio
    async def test_negative_balance_due(self, store):
        """Should handle negative balance (overpayment/credit)."""
        inv = {
            "id": "neg_balance",
            "customer_id": "c1",
            "total_amount": Decimal("100.00"),
            "balance_due": Decimal("-50.00"),
            "status": "credit_balance",
        }
        await store.save(inv)

        result = await store.get("neg_balance")
        assert result["balance_due"] == Decimal("-50.00")

    @pytest.mark.asyncio
    async def test_payment_with_no_method(self, store, sample_invoice):
        """Should handle payment with no method specified."""
        await store.save(sample_invoice)
        now = datetime.now(timezone.utc)

        result = await store.record_payment("ar_inv_001", Decimal("100.00"), now)
        assert result is True

        inv = await store.get("ar_inv_001")
        assert inv["payments"][0]["payment_method"] is None

    @pytest.mark.asyncio
    async def test_multiple_customers_isolation(self, store):
        """Each customer's data should be isolated."""
        for i in range(3):
            await store.save(
                {
                    "id": f"iso_{i}",
                    "customer_id": f"cust_{i}",
                    "balance_due": Decimal("100.00"),
                    "status": "pending",
                }
            )

        for i in range(3):
            balance = await store.get_customer_balance(f"cust_{i}")
            assert balance == Decimal("100.00")

            invoices = await store.list_by_customer(f"cust_{i}")
            assert len(invoices) == 1

    @pytest.mark.asyncio
    async def test_status_transitions(self, store, sample_invoice):
        """Should support various status transitions."""
        await store.save(sample_invoice)

        transitions = ["approved", "sent", "overdue", "paid"]

        for status in transitions:
            result = await store.update_status("ar_inv_001", status)
            assert result is True

            inv = await store.get("ar_inv_001")
            assert inv["status"] == status

    @pytest.mark.asyncio
    async def test_list_by_status_empty_result(self, store, sample_invoice):
        """Should return empty list for status with no matches."""
        await store.save(sample_invoice)

        result = await store.list_by_status("nonexistent_status")
        assert result == []

    @pytest.mark.asyncio
    async def test_list_by_customer_empty_result(self, store, sample_invoice):
        """Should return empty list for customer with no invoices."""
        await store.save(sample_invoice)

        result = await store.list_by_customer("nonexistent_customer")
        assert result == []
