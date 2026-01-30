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
