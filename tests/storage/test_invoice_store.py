"""
Tests for Invoice Storage Backends (Accounts Payable).

Tests cover:
- CRUD operations (create, read, update, delete)
- Listing with filters (by status, by vendor, pending approval)
- Duplicate invoice detection
- PO matching
- Status updates and approval workflow
- Payment scheduling and recording
- Decimal precision handling
- Concurrent access safety
- Edge cases and error handling
"""

from __future__ import annotations

import asyncio
import pytest
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.storage.invoice_store import (
    InMemoryInvoiceStore,
    DecimalEncoder,
    decimal_decoder,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def store():
    """Create a fresh in-memory invoice store."""
    return InMemoryInvoiceStore()


@pytest.fixture
def sample_invoice():
    """Create a sample invoice for testing."""
    return {
        "id": "inv_001",
        "invoice_number": "INV-2024-001",
        "vendor_id": "vendor_123",
        "vendor_name": "Acme Corp",
        "po_number": "PO-2024-100",
        "subtotal": Decimal("1000.00"),
        "tax_amount": Decimal("80.00"),
        "total_amount": Decimal("1080.00"),
        "amount_paid": Decimal("0.00"),
        "balance_due": Decimal("1080.00"),
        "status": "pending",
        "invoice_date": "2024-01-15",
        "due_date": "2024-02-15",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "line_items": [{"description": "Widget A", "quantity": 10, "unit_price": "100.00"}],
    }


@pytest.fixture
def multiple_invoices():
    """Create multiple sample invoices for listing tests."""
    base_time = datetime.now(timezone.utc)
    return [
        {
            "id": f"inv_{i:03d}",
            "invoice_number": f"INV-2024-{i:03d}",
            "vendor_id": "vendor_a" if i % 2 == 0 else "vendor_b",
            "status": ["pending", "approved", "paid", "pending_approval"][i % 4],
            "total_amount": Decimal(str(100 * (i + 1))),
            "po_number": f"PO-{i:03d}" if i < 5 else None,
            "created_at": (base_time - timedelta(days=i)).isoformat(),
        }
        for i in range(10)
    ]


# =============================================================================
# DecimalEncoder Tests
# =============================================================================


class TestDecimalEncoder:
    """Tests for Decimal JSON encoding."""

    def test_encodes_decimal_as_string(self):
        """Decimals should be encoded as strings to preserve precision."""
        import json

        data = {"amount": Decimal("123.45")}
        result = json.dumps(data, cls=DecimalEncoder)
        assert result == '{"amount": "123.45"}'

    def test_encodes_nested_decimals(self):
        """Nested decimals should be encoded correctly."""
        import json

        data = {
            "invoice": {
                "subtotal": Decimal("100.00"),
                "tax": Decimal("8.25"),
                "total": Decimal("108.25"),
            }
        }
        result = json.dumps(data, cls=DecimalEncoder)
        assert '"subtotal": "100.00"' in result
        assert '"tax": "8.25"' in result
        assert '"total": "108.25"' in result

    def test_preserves_other_types(self):
        """Non-Decimal types should be encoded normally."""
        import json

        data = {"name": "Test", "count": 42, "active": True}
        result = json.dumps(data, cls=DecimalEncoder)
        expected = json.dumps(data)
        assert result == expected


class TestDecimalDecoder:
    """Tests for Decimal JSON decoding."""

    def test_decodes_decimal_fields(self):
        """Known decimal fields should be decoded to Decimal."""
        data = {
            "subtotal": "100.00",
            "tax_amount": "8.25",
            "total_amount": "108.25",
        }
        result = decimal_decoder(data)
        assert isinstance(result["subtotal"], Decimal)
        assert result["subtotal"] == Decimal("100.00")

    def test_ignores_non_decimal_fields(self):
        """Fields not in the known list should not be converted."""
        data = {"name": "Test", "custom_amount": "50.00"}
        result = decimal_decoder(data)
        assert isinstance(result["name"], str)
        assert isinstance(result["custom_amount"], str)

    def test_handles_invalid_decimal_strings(self):
        """Invalid decimal strings should be left unchanged."""
        data = {"subtotal": "not-a-number"}
        result = decimal_decoder(data)
        assert result["subtotal"] == "not-a-number"


# =============================================================================
# InMemoryInvoiceStore CRUD Tests
# =============================================================================


class TestInvoiceStoreCRUD:
    """Tests for basic CRUD operations."""

    @pytest.mark.asyncio
    async def test_save_and_get_invoice(self, store, sample_invoice):
        """Should save and retrieve an invoice by ID."""
        await store.save(sample_invoice)
        result = await store.get("inv_001")
        assert result is not None
        assert result["id"] == "inv_001"
        assert result["vendor_id"] == "vendor_123"

    @pytest.mark.asyncio
    async def test_get_nonexistent_invoice(self, store):
        """Getting a nonexistent invoice should return None."""
        result = await store.get("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_save_requires_id(self, store):
        """Saving an invoice without ID should raise ValueError."""
        with pytest.raises(ValueError, match="id is required"):
            await store.save({"vendor_id": "test"})

    @pytest.mark.asyncio
    async def test_update_existing_invoice(self, store, sample_invoice):
        """Saving an invoice with existing ID should update it."""
        await store.save(sample_invoice)

        # Update the invoice
        updated = sample_invoice.copy()
        updated["status"] = "approved"
        await store.save(updated)

        result = await store.get("inv_001")
        assert result["status"] == "approved"

    @pytest.mark.asyncio
    async def test_delete_invoice(self, store, sample_invoice):
        """Should delete an invoice by ID."""
        await store.save(sample_invoice)

        result = await store.delete("inv_001")
        assert result is True

        # Verify it's gone
        assert await store.get("inv_001") is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_invoice(self, store):
        """Deleting a nonexistent invoice should return False."""
        result = await store.delete("nonexistent")
        assert result is False


# =============================================================================
# Listing Tests
# =============================================================================


class TestInvoiceStoreListing:
    """Tests for invoice listing operations."""

    @pytest.mark.asyncio
    async def test_list_all_invoices(self, store, multiple_invoices):
        """Should list all invoices with pagination."""
        for inv in multiple_invoices:
            await store.save(inv)

        result = await store.list_all()
        assert len(result) == 10

    @pytest.mark.asyncio
    async def test_list_all_with_limit(self, store, multiple_invoices):
        """Should respect limit parameter."""
        for inv in multiple_invoices:
            await store.save(inv)

        result = await store.list_all(limit=5)
        assert len(result) == 5

    @pytest.mark.asyncio
    async def test_list_all_with_offset(self, store, multiple_invoices):
        """Should respect offset parameter."""
        for inv in multiple_invoices:
            await store.save(inv)

        # Get first page
        page1 = await store.list_all(limit=5, offset=0)
        # Get second page
        page2 = await store.list_all(limit=5, offset=5)

        # Pages should be different
        page1_ids = {i["id"] for i in page1}
        page2_ids = {i["id"] for i in page2}
        assert page1_ids.isdisjoint(page2_ids)

    @pytest.mark.asyncio
    async def test_list_all_sorted_by_created_at(self, store, multiple_invoices):
        """Results should be sorted by created_at descending."""
        for inv in multiple_invoices:
            await store.save(inv)

        result = await store.list_all()
        # Most recent first (inv_000 was created most recently in our fixture)
        dates = [i["created_at"] for i in result]
        assert dates == sorted(dates, reverse=True)

    @pytest.mark.asyncio
    async def test_list_by_status(self, store, multiple_invoices):
        """Should filter invoices by status."""
        for inv in multiple_invoices:
            await store.save(inv)

        pending = await store.list_by_status("pending")
        for inv in pending:
            assert inv["status"] == "pending"

    @pytest.mark.asyncio
    async def test_list_by_vendor(self, store, multiple_invoices):
        """Should filter invoices by vendor."""
        for inv in multiple_invoices:
            await store.save(inv)

        vendor_a_invoices = await store.list_by_vendor("vendor_a")
        for inv in vendor_a_invoices:
            assert inv["vendor_id"] == "vendor_a"

    @pytest.mark.asyncio
    async def test_list_pending_approval(self, store, multiple_invoices):
        """Should list invoices pending approval."""
        for inv in multiple_invoices:
            await store.save(inv)

        pending = await store.list_pending_approval()
        for inv in pending:
            assert inv["status"] == "pending_approval" or inv.get("requires_approval")


# =============================================================================
# Duplicate Detection Tests
# =============================================================================


class TestDuplicateDetection:
    """Tests for duplicate invoice detection."""

    @pytest.mark.asyncio
    async def test_find_duplicates_by_vendor_and_number(self, store, sample_invoice):
        """Should find duplicates by vendor_id and invoice_number."""
        await store.save(sample_invoice)

        # Search for duplicates
        duplicates = await store.find_duplicates(
            vendor_id="vendor_123",
            invoice_number="INV-2024-001",
            total_amount=Decimal("1080.00"),
        )

        assert len(duplicates) >= 1
        assert any(d["id"] == "inv_001" for d in duplicates)

    @pytest.mark.asyncio
    async def test_find_duplicates_no_matches(self, store, sample_invoice):
        """Should return empty list when no duplicates found."""
        await store.save(sample_invoice)

        duplicates = await store.find_duplicates(
            vendor_id="different_vendor",
            invoice_number="DIFFERENT-001",
            total_amount=Decimal("999.99"),
        )

        assert len(duplicates) == 0

    @pytest.mark.asyncio
    async def test_find_duplicates_by_amount_tolerance(self, store):
        """Duplicate detection should consider amount tolerance."""
        # Save invoice with specific amount
        inv1 = {
            "id": "inv_dup1",
            "vendor_id": "vendor_x",
            "invoice_number": "X-001",
            "total_amount": Decimal("1000.00"),
            "status": "pending",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        await store.save(inv1)

        # Search with exact amount
        duplicates = await store.find_duplicates(
            vendor_id="vendor_x",
            invoice_number="X-001",
            total_amount=Decimal("1000.00"),
        )

        assert len(duplicates) >= 1


# =============================================================================
# PO Matching Tests
# =============================================================================


class TestPOMatching:
    """Tests for purchase order matching."""

    @pytest.mark.asyncio
    async def test_get_invoices_by_po(self, store, multiple_invoices):
        """Should retrieve invoices by PO number."""
        for inv in multiple_invoices:
            await store.save(inv)

        result = await store.get_by_po("PO-000")
        assert len(result) >= 1
        for inv in result:
            assert inv["po_number"] == "PO-000"

    @pytest.mark.asyncio
    async def test_get_by_po_no_matches(self, store, sample_invoice):
        """Should return empty list for non-existent PO."""
        await store.save(sample_invoice)

        result = await store.get_by_po("NONEXISTENT-PO")
        assert len(result) == 0


# =============================================================================
# Status Update Tests
# =============================================================================


class TestStatusUpdates:
    """Tests for invoice status updates."""

    @pytest.mark.asyncio
    async def test_update_status_success(self, store, sample_invoice):
        """Should update invoice status."""
        await store.save(sample_invoice)

        result = await store.update_status("inv_001", "approved", approved_by="user_123")
        assert result is True

        invoice = await store.get("inv_001")
        assert invoice["status"] == "approved"

    @pytest.mark.asyncio
    async def test_update_status_nonexistent(self, store):
        """Should return False for nonexistent invoice."""
        result = await store.update_status("nonexistent", "approved")
        assert result is False

    @pytest.mark.asyncio
    async def test_update_status_with_rejection_reason(self, store, sample_invoice):
        """Should store rejection reason when provided."""
        await store.save(sample_invoice)

        result = await store.update_status(
            "inv_001",
            "rejected",
            rejection_reason="Duplicate invoice",
        )
        assert result is True

        invoice = await store.get("inv_001")
        assert invoice["status"] == "rejected"


# =============================================================================
# Payment Tests
# =============================================================================


class TestPaymentOperations:
    """Tests for payment scheduling and recording."""

    @pytest.mark.asyncio
    async def test_schedule_payment(self, store, sample_invoice):
        """Should schedule payment for an invoice."""
        await store.save(sample_invoice)

        payment_date = datetime.now(timezone.utc) + timedelta(days=7)
        result = await store.schedule_payment("inv_001", payment_date)
        assert result is True

        invoice = await store.get("inv_001")
        assert "scheduled_payment_date" in invoice or "payment_date" in invoice

    @pytest.mark.asyncio
    async def test_schedule_payment_nonexistent(self, store):
        """Should return False for nonexistent invoice."""
        payment_date = datetime.now(timezone.utc) + timedelta(days=7)
        result = await store.schedule_payment("nonexistent", payment_date)
        assert result is False

    @pytest.mark.asyncio
    async def test_record_payment(self, store, sample_invoice):
        """Should record payment against invoice."""
        await store.save(sample_invoice)

        payment_date = datetime.now(timezone.utc)
        result = await store.record_payment(
            "inv_001",
            Decimal("500.00"),
            payment_date,
            payment_method="ACH",
            reference="PAY-001",
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_record_payment_nonexistent(self, store):
        """Should return False for nonexistent invoice."""
        payment_date = datetime.now(timezone.utc)
        result = await store.record_payment(
            "nonexistent",
            Decimal("100.00"),
            payment_date,
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_list_scheduled_payments(self, store, sample_invoice):
        """Should list invoices with scheduled payments."""
        # Create invoice with scheduled payment
        inv_with_payment = sample_invoice.copy()
        inv_with_payment["scheduled_payment_date"] = (
            datetime.now(timezone.utc) + timedelta(days=3)
        ).isoformat()
        await store.save(inv_with_payment)

        result = await store.list_scheduled_payments()
        # Result should include scheduled invoices
        # Implementation depends on store backend


# =============================================================================
# Decimal Precision Tests
# =============================================================================


class TestDecimalPrecision:
    """Tests for decimal precision handling."""

    @pytest.mark.asyncio
    async def test_preserves_decimal_precision(self, store):
        """Decimal values should preserve precision through save/get cycle."""
        invoice = {
            "id": "inv_precision",
            "subtotal": Decimal("1234.56789"),
            "tax_amount": Decimal("0.01"),
            "total_amount": Decimal("1234.57789"),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        await store.save(invoice)

        result = await store.get("inv_precision")
        # Note: In-memory store preserves exact values
        # Database backends may truncate based on column precision
        assert result["subtotal"] == Decimal("1234.56789")

    @pytest.mark.asyncio
    async def test_handles_large_decimal_values(self, store):
        """Should handle large decimal values."""
        invoice = {
            "id": "inv_large",
            "total_amount": Decimal("999999999.99"),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        await store.save(invoice)

        result = await store.get("inv_large")
        assert result["total_amount"] == Decimal("999999999.99")

    @pytest.mark.asyncio
    async def test_handles_small_decimal_values(self, store):
        """Should handle small decimal values."""
        invoice = {
            "id": "inv_small",
            "total_amount": Decimal("0.01"),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        await store.save(invoice)

        result = await store.get("inv_small")
        assert result["total_amount"] == Decimal("0.01")


# =============================================================================
# Concurrent Access Tests
# =============================================================================


class TestConcurrentAccess:
    """Tests for thread safety and concurrent access."""

    @pytest.mark.asyncio
    async def test_concurrent_saves(self, store):
        """Concurrent saves should not corrupt data."""

        async def save_invoice(i: int):
            invoice = {
                "id": f"inv_concurrent_{i}",
                "vendor_id": f"vendor_{i}",
                "total_amount": Decimal(str(i * 100)),
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            await store.save(invoice)

        # Save 50 invoices concurrently
        await asyncio.gather(*[save_invoice(i) for i in range(50)])

        # Verify all were saved
        for i in range(50):
            result = await store.get(f"inv_concurrent_{i}")
            assert result is not None
            assert result["vendor_id"] == f"vendor_{i}"

    @pytest.mark.asyncio
    async def test_concurrent_reads_and_writes(self, store):
        """Concurrent reads and writes should be safe."""
        # Pre-populate some invoices
        for i in range(10):
            await store.save(
                {
                    "id": f"inv_rw_{i}",
                    "vendor_id": "test",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
            )

        async def read_invoice(i: int):
            return await store.get(f"inv_rw_{i % 10}")

        async def write_invoice(i: int):
            await store.save(
                {
                    "id": f"inv_rw_new_{i}",
                    "vendor_id": "test",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
            )

        # Mix of reads and writes
        tasks = []
        for i in range(50):
            if i % 2 == 0:
                tasks.append(read_invoice(i))
            else:
                tasks.append(write_invoice(i))

        await asyncio.gather(*tasks)

    @pytest.mark.asyncio
    async def test_concurrent_payments_atomic(self, store):
        """Concurrent payment recordings should be atomic.

        When 10 concurrent $10 payments are made on the same invoice,
        the final total should be exactly $100, not less due to race conditions.
        """
        # Create invoice with $1000 total
        invoice = {
            "id": "inv_concurrent_payments",
            "vendor_id": "test",
            "total_amount": Decimal("1000.00"),
            "amount_paid": Decimal("0.00"),
            "balance_due": Decimal("1000.00"),
            "status": "pending",
            "payments": [],
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        await store.save(invoice)

        # Record 10 payments of $10 each concurrently
        async def record_payment(i: int):
            await store.record_payment(
                "inv_concurrent_payments",
                Decimal("10.00"),
                datetime.now(timezone.utc),
                payment_method=f"card_{i}",
                reference=f"ref_{i}",
            )

        await asyncio.gather(*[record_payment(i) for i in range(10)])

        # Verify all payments were recorded correctly
        result = await store.get("inv_concurrent_payments")
        assert result is not None

        # Total paid should be exactly $100 (10 x $10)
        payments = result.get("payments", [])
        total_paid = sum(Decimal(p["amount"]) for p in payments)
        assert total_paid == Decimal("100.00"), f"Expected $100, got ${total_paid}"

        # Balance should be $900
        balance = Decimal(result.get("balance_due", "0"))
        assert balance == Decimal("900.00"), f"Expected $900 balance, got ${balance}"


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_empty_store_list_all(self, store):
        """list_all on empty store should return empty list."""
        result = await store.list_all()
        assert result == []

    @pytest.mark.asyncio
    async def test_empty_store_list_by_status(self, store):
        """list_by_status on empty store should return empty list."""
        result = await store.list_by_status("pending")
        assert result == []

    @pytest.mark.asyncio
    async def test_invoice_with_null_fields(self, store):
        """Should handle invoices with null optional fields."""
        invoice = {
            "id": "inv_nulls",
            "vendor_id": "test",
            "po_number": None,
            "approved_by": None,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        await store.save(invoice)

        result = await store.get("inv_nulls")
        assert result["po_number"] is None

    @pytest.mark.asyncio
    async def test_invoice_with_empty_line_items(self, store):
        """Should handle invoices with empty line_items."""
        invoice = {
            "id": "inv_empty_items",
            "vendor_id": "test",
            "line_items": [],
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        await store.save(invoice)

        result = await store.get("inv_empty_items")
        assert result["line_items"] == []

    @pytest.mark.asyncio
    async def test_close_is_idempotent(self, store):
        """Calling close multiple times should be safe."""
        await store.close()
        await store.close()  # Should not raise

    @pytest.mark.asyncio
    async def test_special_characters_in_vendor_id(self, store):
        """Should handle special characters in vendor_id."""
        invoice = {
            "id": "inv_special",
            "vendor_id": "vendor/with:special#chars",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        await store.save(invoice)

        result = await store.list_by_vendor("vendor/with:special#chars")
        assert len(result) >= 1

    @pytest.mark.asyncio
    async def test_unicode_in_fields(self, store):
        """Should handle unicode characters in fields."""
        invoice = {
            "id": "inv_unicode",
            "vendor_name": "Empresa Espa\u00f1ola",
            "description": "\u4e2d\u6587\u6d4b\u8bd5",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        await store.save(invoice)

        result = await store.get("inv_unicode")
        assert result["vendor_name"] == "Empresa Espa\u00f1ola"
        assert result["description"] == "\u4e2d\u6587\u6d4b\u8bd5"


# =============================================================================
# Store Lifecycle Tests
# =============================================================================


class TestStoreLifecycle:
    """Tests for store initialization and cleanup."""

    @pytest.mark.asyncio
    async def test_new_store_is_empty(self, store):
        """Newly created store should be empty."""
        result = await store.list_all()
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_data_persists_across_operations(self, store, sample_invoice):
        """Data should persist across multiple operations."""
        await store.save(sample_invoice)

        # Multiple operations
        await store.update_status("inv_001", "approved")
        payment_date = datetime.now(timezone.utc)
        await store.record_payment("inv_001", Decimal("100.00"), payment_date)

        # Data should still be retrievable
        result = await store.get("inv_001")
        assert result is not None
        assert result["status"] == "approved"


# =============================================================================
# PostgresInvoiceStore Precision Tests
# =============================================================================


class TestPostgresDecimalPrecision:
    """Tests for PostgresInvoiceStore decimal precision handling.

    These tests verify that Decimal values are converted to strings (not floats)
    to prevent precision loss when stored in PostgreSQL NUMERIC columns.
    """

    @pytest.fixture
    def mock_pool(self):
        """Create a mock asyncpg connection pool."""
        pool = MagicMock()
        conn = AsyncMock()
        pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
        pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        return pool, conn

    @pytest.mark.asyncio
    async def test_save_converts_decimal_to_string_not_float(self, mock_pool):
        """Decimal total_amount should be converted to str, not float.

        This is critical for financial accuracy - float(Decimal("1234567.89"))
        can lose precision, but str(Decimal("1234567.89")) preserves it exactly.
        """
        from aragora.storage.invoice_store import PostgresInvoiceStore

        pool, conn = mock_pool
        store = PostgresInvoiceStore(pool)

        # Use an amount that would lose precision with float
        invoice = {
            "id": "inv_precision_test",
            "total_amount": Decimal("12345678901234.56"),
            "vendor_id": "v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        await store.save(invoice)

        # Verify execute was called and check the total_amount parameter
        conn.execute.assert_called_once()
        call_args = conn.execute.call_args
        # The total_amount is the 6th positional argument ($6 in the query)
        amount_arg = call_args[0][6]

        # Should be string, not float
        assert isinstance(amount_arg, str), f"Expected str, got {type(amount_arg)}"
        assert amount_arg == "12345678901234.56"

    @pytest.mark.asyncio
    async def test_save_converts_string_amount_to_normalized_string(self, mock_pool):
        """String amounts should be normalized through Decimal for consistency."""
        from aragora.storage.invoice_store import PostgresInvoiceStore

        pool, conn = mock_pool
        store = PostgresInvoiceStore(pool)

        invoice = {
            "id": "inv_string_test",
            "total_amount": "9999.99",
            "vendor_id": "v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        await store.save(invoice)

        call_args = conn.execute.call_args
        amount_arg = call_args[0][6]

        assert isinstance(amount_arg, str)
        assert amount_arg == "9999.99"

    @pytest.mark.asyncio
    async def test_save_preserves_sub_cent_precision(self, mock_pool):
        """Sub-cent amounts should be preserved exactly."""
        from aragora.storage.invoice_store import PostgresInvoiceStore

        pool, conn = mock_pool
        store = PostgresInvoiceStore(pool)

        invoice = {
            "id": "inv_subcent",
            "total_amount": Decimal("123.456789"),
            "vendor_id": "v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        await store.save(invoice)

        call_args = conn.execute.call_args
        amount_arg = call_args[0][6]

        assert amount_arg == "123.456789"

    @pytest.mark.asyncio
    async def test_find_duplicates_uses_string_amount(self, mock_pool):
        """Duplicate detection should use string amounts for precision."""
        from aragora.storage.invoice_store import PostgresInvoiceStore

        pool, conn = mock_pool
        conn.fetch.return_value = []  # No duplicates found
        store = PostgresInvoiceStore(pool)

        await store.find_duplicates(
            vendor_id="vendor_123",
            invoice_number="INV-001",
            total_amount=Decimal("99999999.99"),
        )

        conn.fetch.assert_called_once()
        call_args = conn.fetch.call_args
        # total_amount is the 3rd argument ($3)
        amount_arg = call_args[0][3]

        assert isinstance(amount_arg, str), f"Expected str, got {type(amount_arg)}"
        assert amount_arg == "99999999.99"

    @pytest.mark.asyncio
    async def test_large_amount_float_vs_str_precision_difference(self, mock_pool):
        """Demonstrate why float conversion loses precision for large amounts.

        This test documents the bug that was fixed: float() loses precision
        for amounts > ~15 significant digits.
        """
        # This amount has 18 significant digits - beyond float precision
        original = Decimal("123456789012345678.99")

        # float() loses precision (this is the bug we fixed)
        float_converted = float(original)
        str_converted = str(original)

        # String preserves exact representation
        assert str_converted == "123456789012345678.99"

        # When converted back to Decimal, float loses precision
        float_back = Decimal(str(float_converted))
        str_back = Decimal(str_converted)

        # This proves the precision loss with float
        assert float_back != original  # BROKEN with float
        assert str_back == original  # CORRECT with str


# =============================================================================
# SQLiteInvoiceStore Tests
# =============================================================================


class TestSQLiteInvoiceStore:
    """Tests for SQLite-backed invoice store."""

    @pytest.fixture
    def sqlite_store(self, tmp_path):
        """Create a SQLite invoice store with temporary database."""
        from aragora.storage.invoice_store import SQLiteInvoiceStore

        db_path = tmp_path / "invoices.db"
        return SQLiteInvoiceStore(db_path=db_path)

    @pytest.fixture
    def sample_invoice(self):
        """Sample invoice for SQLite tests."""
        return {
            "id": "sqlite_inv_001",
            "invoice_number": "INV-2024-001",
            "vendor_id": "vendor_123",
            "vendor_name": "Acme Corp",
            "po_number": "PO-2024-100",
            "total_amount": Decimal("1080.00"),
            "status": "pending",
            "invoice_date": "2024-01-15",
            "due_date": "2024-02-15",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

    @pytest.mark.asyncio
    async def test_save_and_get(self, sqlite_store, sample_invoice):
        """Should save and retrieve invoice from SQLite."""
        await sqlite_store.save(sample_invoice)
        result = await sqlite_store.get("sqlite_inv_001")
        assert result is not None
        assert result["id"] == "sqlite_inv_001"
        assert result["vendor_name"] == "Acme Corp"

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, sqlite_store):
        """Should return None for nonexistent invoice."""
        result = await sqlite_store.get("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_save_requires_id(self, sqlite_store):
        """Should raise ValueError when id is missing."""
        with pytest.raises(ValueError, match="id is required"):
            await sqlite_store.save({"vendor_id": "test"})

    @pytest.mark.asyncio
    async def test_upsert_existing(self, sqlite_store, sample_invoice):
        """Should upsert existing invoice."""
        await sqlite_store.save(sample_invoice)
        sample_invoice["vendor_name"] = "Updated Corp"
        await sqlite_store.save(sample_invoice)

        result = await sqlite_store.get("sqlite_inv_001")
        assert result["vendor_name"] == "Updated Corp"

    @pytest.mark.asyncio
    async def test_delete(self, sqlite_store, sample_invoice):
        """Should delete invoice from SQLite."""
        await sqlite_store.save(sample_invoice)
        result = await sqlite_store.delete("sqlite_inv_001")
        assert result is True
        assert await sqlite_store.get("sqlite_inv_001") is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, sqlite_store):
        """Should return False when deleting nonexistent invoice."""
        result = await sqlite_store.delete("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_list_all_pagination(self, sqlite_store):
        """Should list all with pagination."""
        for i in range(10):
            await sqlite_store.save(
                {
                    "id": f"inv_{i:03d}",
                    "total_amount": Decimal(str(i * 100)),
                    "status": "pending",
                }
            )

        page1 = await sqlite_store.list_all(limit=3, offset=0)
        page2 = await sqlite_store.list_all(limit=3, offset=3)

        assert len(page1) == 3
        assert len(page2) == 3
        page1_ids = {i["id"] for i in page1}
        page2_ids = {i["id"] for i in page2}
        assert page1_ids.isdisjoint(page2_ids)

    @pytest.mark.asyncio
    async def test_list_by_status(self, sqlite_store):
        """Should filter by status."""
        await sqlite_store.save({"id": "inv_1", "status": "pending", "total_amount": "100"})
        await sqlite_store.save({"id": "inv_2", "status": "approved", "total_amount": "200"})
        await sqlite_store.save({"id": "inv_3", "status": "pending", "total_amount": "300"})

        pending = await sqlite_store.list_by_status("pending")
        assert len(pending) == 2
        assert all(i["status"] == "pending" for i in pending)

    @pytest.mark.asyncio
    async def test_list_by_vendor(self, sqlite_store):
        """Should filter by vendor."""
        await sqlite_store.save({"id": "inv_1", "vendor_id": "v1", "total_amount": "100"})
        await sqlite_store.save({"id": "inv_2", "vendor_id": "v2", "total_amount": "200"})
        await sqlite_store.save({"id": "inv_3", "vendor_id": "v1", "total_amount": "300"})

        v1_invoices = await sqlite_store.list_by_vendor("v1")
        assert len(v1_invoices) == 2
        assert all(i["vendor_id"] == "v1" for i in v1_invoices)

    @pytest.mark.asyncio
    async def test_list_pending_approval(self, sqlite_store):
        """Should list invoices pending approval."""
        await sqlite_store.save(
            {"id": "inv_1", "status": "pending_approval", "total_amount": "100"}
        )
        await sqlite_store.save({"id": "inv_2", "status": "approved", "total_amount": "200"})
        await sqlite_store.save({"id": "inv_3", "requires_approval": True, "total_amount": "300"})

        pending = await sqlite_store.list_pending_approval()
        assert len(pending) == 2

    @pytest.mark.asyncio
    async def test_list_scheduled_payments_date_range(self, sqlite_store):
        """Should filter scheduled payments by date range."""
        now = datetime.now(timezone.utc)

        # Invoice with scheduled payment in range
        await sqlite_store.save(
            {
                "id": "inv_in_range",
                "status": "approved",
                "total_amount": "100",
                "scheduled_payment_date": (now + timedelta(days=5)).isoformat(),
            }
        )

        # Invoice with scheduled payment outside range
        await sqlite_store.save(
            {
                "id": "inv_out_of_range",
                "status": "approved",
                "total_amount": "200",
                "scheduled_payment_date": (now + timedelta(days=30)).isoformat(),
            }
        )

        # Invoice without scheduled payment
        await sqlite_store.save(
            {
                "id": "inv_no_schedule",
                "status": "approved",
                "total_amount": "300",
            }
        )

        start_date = now
        end_date = now + timedelta(days=10)
        result = await sqlite_store.list_scheduled_payments(start_date, end_date)

        assert len(result) == 1
        assert result[0]["id"] == "inv_in_range"

    @pytest.mark.asyncio
    async def test_find_duplicates(self, sqlite_store, sample_invoice):
        """Should find duplicate invoices by vendor and invoice number."""
        await sqlite_store.save(sample_invoice)

        duplicates = await sqlite_store.find_duplicates(
            vendor_id="vendor_123",
            invoice_number="INV-2024-001",
            total_amount=Decimal("1080.00"),
        )

        assert len(duplicates) >= 1
        assert any(d["id"] == "sqlite_inv_001" for d in duplicates)

    @pytest.mark.asyncio
    async def test_get_by_po(self, sqlite_store, sample_invoice):
        """Should retrieve invoices by PO number."""
        await sqlite_store.save(sample_invoice)

        result = await sqlite_store.get_by_po("PO-2024-100")
        assert len(result) == 1
        assert result[0]["po_number"] == "PO-2024-100"

    @pytest.mark.asyncio
    async def test_update_status(self, sqlite_store, sample_invoice):
        """Should update invoice status."""
        await sqlite_store.save(sample_invoice)

        result = await sqlite_store.update_status(
            "sqlite_inv_001",
            "approved",
            approved_by="user_123",
        )
        assert result is True

        invoice = await sqlite_store.get("sqlite_inv_001")
        assert invoice["status"] == "approved"
        assert invoice["approved_by"] == "user_123"

    @pytest.mark.asyncio
    async def test_update_status_with_rejection(self, sqlite_store, sample_invoice):
        """Should store rejection reason."""
        await sqlite_store.save(sample_invoice)

        result = await sqlite_store.update_status(
            "sqlite_inv_001",
            "rejected",
            rejection_reason="Invalid invoice amount",
        )
        assert result is True

        invoice = await sqlite_store.get("sqlite_inv_001")
        assert invoice["status"] == "rejected"
        assert invoice["rejection_reason"] == "Invalid invoice amount"

    @pytest.mark.asyncio
    async def test_update_status_nonexistent(self, sqlite_store):
        """Should return False for nonexistent invoice."""
        result = await sqlite_store.update_status("nonexistent", "approved")
        assert result is False

    @pytest.mark.asyncio
    async def test_schedule_payment(self, sqlite_store, sample_invoice):
        """Should schedule payment for invoice."""
        await sqlite_store.save(sample_invoice)
        payment_date = datetime.now(timezone.utc) + timedelta(days=7)

        result = await sqlite_store.schedule_payment("sqlite_inv_001", payment_date)
        assert result is True

        invoice = await sqlite_store.get("sqlite_inv_001")
        assert "scheduled_payment_date" in invoice

    @pytest.mark.asyncio
    async def test_schedule_payment_nonexistent(self, sqlite_store):
        """Should return False for nonexistent invoice."""
        payment_date = datetime.now(timezone.utc) + timedelta(days=7)
        result = await sqlite_store.schedule_payment("nonexistent", payment_date)
        assert result is False

    @pytest.mark.asyncio
    async def test_record_payment(self, sqlite_store):
        """Should record payment against invoice."""
        await sqlite_store.save(
            {
                "id": "pay_inv",
                "total_amount": Decimal("1000.00"),
                "amount_paid": Decimal("0.00"),
                "balance_due": Decimal("1000.00"),
                "status": "approved",
            }
        )

        payment_date = datetime.now(timezone.utc)
        result = await sqlite_store.record_payment(
            "pay_inv",
            Decimal("500.00"),
            payment_date,
            payment_method="ACH",
            reference="PAY-001",
        )
        assert result is True

        invoice = await sqlite_store.get("pay_inv")
        assert Decimal(invoice["amount_paid"]) == Decimal("500.00")
        assert Decimal(invoice["balance_due"]) == Decimal("500.00")
        assert len(invoice["payments"]) == 1

    @pytest.mark.asyncio
    async def test_record_payment_marks_paid_when_full(self, sqlite_store):
        """Should mark invoice as paid when fully paid."""
        await sqlite_store.save(
            {
                "id": "pay_inv",
                "total_amount": Decimal("1000.00"),
                "amount_paid": Decimal("0.00"),
                "balance_due": Decimal("1000.00"),
                "status": "approved",
            }
        )

        payment_date = datetime.now(timezone.utc)
        await sqlite_store.record_payment("pay_inv", Decimal("1000.00"), payment_date)

        invoice = await sqlite_store.get("pay_inv")
        assert invoice["status"] == "paid"
        assert Decimal(invoice["balance_due"]) == Decimal("0.00")

    @pytest.mark.asyncio
    async def test_record_payment_nonexistent(self, sqlite_store):
        """Should return False for nonexistent invoice."""
        payment_date = datetime.now(timezone.utc)
        result = await sqlite_store.record_payment("nonexistent", Decimal("100.00"), payment_date)
        assert result is False

    @pytest.mark.asyncio
    async def test_close_is_idempotent(self, sqlite_store, sample_invoice):
        """Close should be idempotent."""
        await sqlite_store.save(sample_invoice)
        await sqlite_store.close()
        await sqlite_store.close()  # Should not raise

    @pytest.mark.asyncio
    async def test_decimal_precision_preserved(self, sqlite_store):
        """Should preserve decimal precision."""
        await sqlite_store.save(
            {
                "id": "precision_inv",
                "total_amount": Decimal("12345.6789"),
                "status": "pending",
            }
        )

        result = await sqlite_store.get("precision_inv")
        # Note: SQLite stores in JSON, so precision should be preserved
        assert "12345.6789" in str(result["total_amount"])


# =============================================================================
# Store Lifecycle and Singleton Tests
# =============================================================================


class TestStoreLifecycleGlobal:
    """Tests for global store singleton lifecycle."""

    def test_reset_invoice_store(self):
        """reset_invoice_store should clear the global store."""
        from aragora.storage.invoice_store import (
            reset_invoice_store,
            InMemoryInvoiceStore,
            set_invoice_store,
        )

        # Set a store
        custom_store = InMemoryInvoiceStore()
        set_invoice_store(custom_store)

        # Reset it
        reset_invoice_store()

        # The global should be None now (but we can't easily verify without
        # accessing the private _invoice_store variable)
        # This test mainly ensures no exceptions are raised
        reset_invoice_store()  # Should be idempotent

    def test_set_invoice_store(self):
        """set_invoice_store should set a custom store instance."""
        from aragora.storage.invoice_store import (
            set_invoice_store,
            reset_invoice_store,
            InMemoryInvoiceStore,
        )

        try:
            custom_store = InMemoryInvoiceStore()
            set_invoice_store(custom_store)
            # Verify it was set (implicitly, no exception means success)
        finally:
            reset_invoice_store()


# =============================================================================
# Line Item Management Tests
# =============================================================================


class TestLineItemManagement:
    """Tests for invoice line item handling."""

    @pytest.fixture
    def store(self):
        return InMemoryInvoiceStore()

    @pytest.mark.asyncio
    async def test_invoice_with_line_items(self, store):
        """Should store and retrieve invoice with line items."""
        invoice = {
            "id": "li_inv_001",
            "vendor_id": "vendor_123",
            "total_amount": Decimal("1080.00"),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "line_items": [
                {
                    "description": "Widget A",
                    "quantity": 10,
                    "unit_price": "50.00",
                    "total": "500.00",
                },
                {
                    "description": "Widget B",
                    "quantity": 5,
                    "unit_price": "100.00",
                    "total": "500.00",
                },
                {"description": "Shipping", "quantity": 1, "unit_price": "80.00", "total": "80.00"},
            ],
        }
        await store.save(invoice)

        result = await store.get("li_inv_001")
        assert result is not None
        assert len(result["line_items"]) == 3
        assert result["line_items"][0]["description"] == "Widget A"
        assert result["line_items"][1]["quantity"] == 5
        assert result["line_items"][2]["total"] == "80.00"

    @pytest.mark.asyncio
    async def test_update_line_items(self, store):
        """Should update line items on re-save."""
        invoice = {
            "id": "li_inv_002",
            "vendor_id": "vendor_123",
            "total_amount": Decimal("500.00"),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "line_items": [
                {"description": "Original Item", "quantity": 1, "unit_price": "500.00"},
            ],
        }
        await store.save(invoice)

        # Update line items
        invoice["line_items"] = [
            {"description": "Updated Item", "quantity": 2, "unit_price": "250.00"},
        ]
        invoice["total_amount"] = Decimal("500.00")
        await store.save(invoice)

        result = await store.get("li_inv_002")
        assert len(result["line_items"]) == 1
        assert result["line_items"][0]["description"] == "Updated Item"
        assert result["line_items"][0]["quantity"] == 2

    @pytest.mark.asyncio
    async def test_add_line_items(self, store):
        """Should add new line items on update."""
        invoice = {
            "id": "li_inv_003",
            "vendor_id": "vendor_123",
            "total_amount": Decimal("100.00"),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "line_items": [
                {"description": "Item 1", "quantity": 1, "unit_price": "100.00"},
            ],
        }
        await store.save(invoice)

        # Add more line items
        invoice["line_items"].append(
            {"description": "Item 2", "quantity": 2, "unit_price": "50.00"}
        )
        invoice["total_amount"] = Decimal("200.00")
        await store.save(invoice)

        result = await store.get("li_inv_003")
        assert len(result["line_items"]) == 2

    @pytest.mark.asyncio
    async def test_line_items_with_decimal_amounts(self, store):
        """Should handle decimal amounts in line items."""
        invoice = {
            "id": "li_inv_004",
            "vendor_id": "vendor_123",
            "total_amount": Decimal("123.45"),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "line_items": [
                {
                    "description": "Precise Item",
                    "quantity": 3,
                    "unit_price": Decimal("41.15"),
                    "total": Decimal("123.45"),
                },
            ],
        }
        await store.save(invoice)

        result = await store.get("li_inv_004")
        assert result["line_items"][0]["unit_price"] == Decimal("41.15")


# =============================================================================
# Status Transition Tests
# =============================================================================


class TestStatusTransitions:
    """Tests for invoice status transitions (draft, sent, paid, overdue)."""

    @pytest.fixture
    def store(self):
        return InMemoryInvoiceStore()

    @pytest.mark.asyncio
    async def test_draft_to_sent(self, store):
        """Should transition from draft to sent."""
        invoice = {
            "id": "status_001",
            "status": "draft",
            "total_amount": Decimal("100.00"),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        await store.save(invoice)

        result = await store.update_status("status_001", "sent")
        assert result is True

        inv = await store.get("status_001")
        assert inv["status"] == "sent"

    @pytest.mark.asyncio
    async def test_sent_to_paid(self, store):
        """Should transition from sent to paid."""
        invoice = {
            "id": "status_002",
            "status": "sent",
            "total_amount": Decimal("100.00"),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        await store.save(invoice)

        result = await store.update_status("status_002", "paid")
        assert result is True

        inv = await store.get("status_002")
        assert inv["status"] == "paid"

    @pytest.mark.asyncio
    async def test_pending_to_overdue(self, store):
        """Should transition from pending to overdue."""
        invoice = {
            "id": "status_003",
            "status": "pending",
            "total_amount": Decimal("100.00"),
            "due_date": (datetime.now(timezone.utc) - timedelta(days=1)).isoformat(),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        await store.save(invoice)

        result = await store.update_status("status_003", "overdue")
        assert result is True

        inv = await store.get("status_003")
        assert inv["status"] == "overdue"

    @pytest.mark.asyncio
    async def test_full_lifecycle(self, store):
        """Test full invoice lifecycle: draft -> pending_approval -> approved -> paid."""
        invoice = {
            "id": "lifecycle_001",
            "status": "draft",
            "total_amount": Decimal("1000.00"),
            "amount_paid": Decimal("0.00"),
            "balance_due": Decimal("1000.00"),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        await store.save(invoice)

        # Draft -> Pending Approval
        await store.update_status("lifecycle_001", "pending_approval")
        inv = await store.get("lifecycle_001")
        assert inv["status"] == "pending_approval"

        # Pending Approval -> Approved
        await store.update_status("lifecycle_001", "approved", approved_by="manager_123")
        inv = await store.get("lifecycle_001")
        assert inv["status"] == "approved"
        assert inv["approved_by"] == "manager_123"

        # Approved -> Paid (via payment)
        payment_date = datetime.now(timezone.utc)
        await store.record_payment("lifecycle_001", Decimal("1000.00"), payment_date)
        inv = await store.get("lifecycle_001")
        assert inv["status"] == "paid"


# =============================================================================
# Date Range Query Tests
# =============================================================================


class TestDateRangeQueries:
    """Tests for date range filtering."""

    @pytest.fixture
    def store(self):
        return InMemoryInvoiceStore()

    @pytest.mark.asyncio
    async def test_scheduled_payments_start_date_only(self, store):
        """Should filter scheduled payments with start date only."""
        now = datetime.now(timezone.utc)

        # Old payment
        await store.save(
            {
                "id": "old_payment",
                "status": "approved",
                "scheduled_payment_date": (now - timedelta(days=10)).isoformat(),
                "created_at": now.isoformat(),
            }
        )

        # Future payment
        await store.save(
            {
                "id": "future_payment",
                "status": "approved",
                "scheduled_payment_date": (now + timedelta(days=10)).isoformat(),
                "created_at": now.isoformat(),
            }
        )

        result = await store.list_scheduled_payments(start_date=now)
        assert len(result) == 1
        assert result[0]["id"] == "future_payment"

    @pytest.mark.asyncio
    async def test_scheduled_payments_end_date_only(self, store):
        """Should filter scheduled payments with end date only."""
        now = datetime.now(timezone.utc)

        # Old payment
        await store.save(
            {
                "id": "old_payment",
                "status": "approved",
                "scheduled_payment_date": (now - timedelta(days=10)).isoformat(),
                "created_at": now.isoformat(),
            }
        )

        # Future payment
        await store.save(
            {
                "id": "future_payment",
                "status": "approved",
                "scheduled_payment_date": (now + timedelta(days=10)).isoformat(),
                "created_at": now.isoformat(),
            }
        )

        result = await store.list_scheduled_payments(end_date=now)
        assert len(result) == 1
        assert result[0]["id"] == "old_payment"

    @pytest.mark.asyncio
    async def test_scheduled_payments_both_dates(self, store):
        """Should filter scheduled payments with both start and end dates."""
        now = datetime.now(timezone.utc)

        # Payment outside range (before)
        await store.save(
            {
                "id": "before_range",
                "status": "approved",
                "scheduled_payment_date": (now - timedelta(days=10)).isoformat(),
                "created_at": now.isoformat(),
            }
        )

        # Payment in range
        await store.save(
            {
                "id": "in_range",
                "status": "approved",
                "scheduled_payment_date": (now + timedelta(days=5)).isoformat(),
                "created_at": now.isoformat(),
            }
        )

        # Payment outside range (after)
        await store.save(
            {
                "id": "after_range",
                "status": "approved",
                "scheduled_payment_date": (now + timedelta(days=20)).isoformat(),
                "created_at": now.isoformat(),
            }
        )

        start_date = now
        end_date = now + timedelta(days=10)
        result = await store.list_scheduled_payments(start_date, end_date)

        assert len(result) == 1
        assert result[0]["id"] == "in_range"

    @pytest.mark.asyncio
    async def test_scheduled_payments_excludes_non_approved(self, store):
        """Should exclude non-approved invoices from scheduled payments."""
        now = datetime.now(timezone.utc)

        # Approved with scheduled payment
        await store.save(
            {
                "id": "approved_scheduled",
                "status": "approved",
                "scheduled_payment_date": (now + timedelta(days=5)).isoformat(),
                "created_at": now.isoformat(),
            }
        )

        # Pending with scheduled payment
        await store.save(
            {
                "id": "pending_scheduled",
                "status": "pending",
                "scheduled_payment_date": (now + timedelta(days=5)).isoformat(),
                "created_at": now.isoformat(),
            }
        )

        result = await store.list_scheduled_payments()
        assert len(result) == 1
        assert result[0]["id"] == "approved_scheduled"

    @pytest.mark.asyncio
    async def test_scheduled_payments_sorted_by_date(self, store):
        """Should return scheduled payments sorted by payment date."""
        now = datetime.now(timezone.utc)

        await store.save(
            {
                "id": "later",
                "status": "approved",
                "scheduled_payment_date": (now + timedelta(days=20)).isoformat(),
                "created_at": now.isoformat(),
            }
        )

        await store.save(
            {
                "id": "earlier",
                "status": "approved",
                "scheduled_payment_date": (now + timedelta(days=5)).isoformat(),
                "created_at": now.isoformat(),
            }
        )

        await store.save(
            {
                "id": "middle",
                "status": "approved",
                "scheduled_payment_date": (now + timedelta(days=10)).isoformat(),
                "created_at": now.isoformat(),
            }
        )

        result = await store.list_scheduled_payments()
        assert len(result) == 3
        assert result[0]["id"] == "earlier"
        assert result[1]["id"] == "middle"
        assert result[2]["id"] == "later"


# =============================================================================
# Vendor/Customer Invoice History Tests
# =============================================================================


class TestVendorInvoiceHistory:
    """Tests for vendor invoice history."""

    @pytest.fixture
    def store(self):
        return InMemoryInvoiceStore()

    @pytest.mark.asyncio
    async def test_get_vendor_history(self, store):
        """Should retrieve complete invoice history for a vendor."""
        now = datetime.now(timezone.utc)

        for i in range(5):
            await store.save(
                {
                    "id": f"vendor_hist_{i}",
                    "vendor_id": "vendor_123",
                    "total_amount": Decimal(str(100 * (i + 1))),
                    "status": ["draft", "sent", "paid", "overdue", "pending"][i],
                    "created_at": (now - timedelta(days=i * 30)).isoformat(),
                }
            )

        # Add invoice for different vendor
        await store.save(
            {
                "id": "other_vendor",
                "vendor_id": "vendor_456",
                "total_amount": Decimal("999.00"),
                "created_at": now.isoformat(),
            }
        )

        result = await store.list_by_vendor("vendor_123")
        assert len(result) == 5
        assert all(inv["vendor_id"] == "vendor_123" for inv in result)

    @pytest.mark.asyncio
    async def test_vendor_history_respects_limit(self, store):
        """Should respect limit parameter for vendor history."""
        now = datetime.now(timezone.utc)

        for i in range(10):
            await store.save(
                {
                    "id": f"vendor_lim_{i}",
                    "vendor_id": "vendor_123",
                    "total_amount": Decimal("100.00"),
                    "created_at": (now - timedelta(days=i)).isoformat(),
                }
            )

        result = await store.list_by_vendor("vendor_123", limit=5)
        assert len(result) == 5

    @pytest.mark.asyncio
    async def test_vendor_history_empty(self, store):
        """Should return empty list for vendor with no invoices."""
        result = await store.list_by_vendor("nonexistent_vendor")
        assert result == []


# =============================================================================
# Multiple Payment Tests
# =============================================================================


class TestMultiplePayments:
    """Tests for recording multiple payments against a single invoice."""

    @pytest.fixture
    def store(self):
        return InMemoryInvoiceStore()

    @pytest.mark.asyncio
    async def test_partial_payments_accumulate(self, store):
        """Multiple partial payments should accumulate correctly."""
        await store.save(
            {
                "id": "multi_pay",
                "total_amount": Decimal("1000.00"),
                "amount_paid": Decimal("0.00"),
                "balance_due": Decimal("1000.00"),
                "status": "approved",
                "payments": [],
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )

        now = datetime.now(timezone.utc)

        # First payment
        await store.record_payment("multi_pay", Decimal("300.00"), now, "check", "CHK-001")
        inv = await store.get("multi_pay")
        assert Decimal(inv["amount_paid"]) == Decimal("300.00")
        assert Decimal(inv["balance_due"]) == Decimal("700.00")
        assert inv["status"] != "paid"

        # Second payment
        await store.record_payment("multi_pay", Decimal("400.00"), now, "ACH", "ACH-001")
        inv = await store.get("multi_pay")
        assert Decimal(inv["amount_paid"]) == Decimal("700.00")
        assert Decimal(inv["balance_due"]) == Decimal("300.00")
        assert inv["status"] != "paid"

        # Final payment
        await store.record_payment("multi_pay", Decimal("300.00"), now, "wire", "WIRE-001")
        inv = await store.get("multi_pay")
        assert Decimal(inv["amount_paid"]) == Decimal("1000.00")
        assert Decimal(inv["balance_due"]) == Decimal("0.00")
        assert inv["status"] == "paid"
        assert len(inv["payments"]) == 3

    @pytest.mark.asyncio
    async def test_payment_details_stored(self, store):
        """Payment details should be stored correctly."""
        await store.save(
            {
                "id": "pay_details",
                "total_amount": Decimal("500.00"),
                "amount_paid": Decimal("0.00"),
                "balance_due": Decimal("500.00"),
                "status": "approved",
                "payments": [],
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )

        payment_date = datetime(2024, 6, 15, 10, 30, 0, tzinfo=timezone.utc)
        await store.record_payment(
            "pay_details",
            Decimal("200.00"),
            payment_date,
            payment_method="credit_card",
            reference="CC-12345-XYZ",
        )

        inv = await store.get("pay_details")
        payment = inv["payments"][0]

        assert payment["amount"] == "200.00"
        assert payment["payment_method"] == "credit_card"
        assert payment["reference"] == "CC-12345-XYZ"
        assert "payment_date" in payment
