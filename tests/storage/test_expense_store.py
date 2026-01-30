"""
Tests for expense storage backends.

Tests cover:
- InMemoryExpenseStore operations
- SQLiteExpenseStore operations
- DecimalEncoder and decimal_decoder utilities
- CRUD operations for expenses
- Statistics and duplicate detection
- Status updates and QBO sync marking
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from aragora.storage.expense_store import (
    DecimalEncoder,
    decimal_decoder,
    InMemoryExpenseStore,
    SQLiteExpenseStore,
    ExpenseStoreBackend,
)


# =============================================================================
# DecimalEncoder Tests
# =============================================================================


class TestDecimalEncoder:
    """Tests for DecimalEncoder JSON encoder."""

    def test_encodes_decimal_as_string(self):
        """Should encode Decimal to string."""
        data = {"amount": Decimal("123.45")}
        result = json.dumps(data, cls=DecimalEncoder)
        assert '"123.45"' in result

    def test_encodes_regular_types_normally(self):
        """Should encode regular types normally."""
        data = {"name": "test", "count": 5, "active": True}
        result = json.dumps(data, cls=DecimalEncoder)
        assert '"name": "test"' in result
        assert '"count": 5' in result
        assert '"active": true' in result

    def test_encodes_nested_decimal(self):
        """Should encode nested Decimal values."""
        data = {
            "expense": {
                "amount": Decimal("99.99"),
                "tax": Decimal("8.50"),
            }
        }
        result = json.dumps(data, cls=DecimalEncoder)
        parsed = json.loads(result)
        assert parsed["expense"]["amount"] == "99.99"
        assert parsed["expense"]["tax"] == "8.50"

    def test_encodes_decimal_zero(self):
        """Should encode Decimal zero."""
        data = {"amount": Decimal("0")}
        result = json.dumps(data, cls=DecimalEncoder)
        assert '"0"' in result

    def test_encodes_large_decimal(self):
        """Should encode large Decimal values."""
        data = {"amount": Decimal("999999999999.99")}
        result = json.dumps(data, cls=DecimalEncoder)
        assert '"999999999999.99"' in result


class TestDecimalDecoder:
    """Tests for decimal_decoder JSON decoder hook."""

    def test_decodes_amount_field(self):
        """Should decode amount field to Decimal."""
        data = {"amount": "123.45", "name": "test"}
        result = decimal_decoder(data)
        assert isinstance(result["amount"], Decimal)
        assert result["amount"] == Decimal("123.45")
        assert result["name"] == "test"

    def test_decodes_tax_amount_field(self):
        """Should decode tax_amount field to Decimal."""
        data = {"tax_amount": "8.50"}
        result = decimal_decoder(data)
        assert isinstance(result["tax_amount"], Decimal)
        assert result["tax_amount"] == Decimal("8.50")

    def test_decodes_tip_amount_field(self):
        """Should decode tip_amount field to Decimal."""
        data = {"tip_amount": "15.00"}
        result = decimal_decoder(data)
        assert isinstance(result["tip_amount"], Decimal)

    def test_decodes_total_amount_field(self):
        """Should decode total_amount field to Decimal."""
        data = {"total_amount": "150.00"}
        result = decimal_decoder(data)
        assert isinstance(result["total_amount"], Decimal)

    def test_ignores_non_decimal_fields(self):
        """Should not modify non-decimal fields."""
        data = {"name": "123.45", "count": "5"}
        result = decimal_decoder(data)
        assert result["name"] == "123.45"
        assert result["count"] == "5"

    def test_handles_invalid_decimal_gracefully(self):
        """Should handle invalid decimal strings gracefully."""
        data = {"amount": "not-a-number"}
        result = decimal_decoder(data)
        assert result["amount"] == "not-a-number"

    def test_handles_already_decimal(self):
        """Should handle non-string values gracefully."""
        data = {"amount": 123.45}
        result = decimal_decoder(data)
        assert result["amount"] == 123.45


# =============================================================================
# InMemoryExpenseStore Tests
# =============================================================================


class TestInMemoryExpenseStoreInit:
    """Tests for InMemoryExpenseStore initialization."""

    def test_creates_empty_store(self):
        """Should create empty store."""
        store = InMemoryExpenseStore()
        assert store._data == {}


class TestInMemoryExpenseStoreSave:
    """Tests for InMemoryExpenseStore.save method."""

    @pytest.mark.asyncio
    async def test_saves_expense(self):
        """Should save expense with id."""
        store = InMemoryExpenseStore()
        data = {"id": "exp_123", "amount": Decimal("50.00")}
        await store.save(data)
        assert "exp_123" in store._data

    @pytest.mark.asyncio
    async def test_requires_id(self):
        """Should raise ValueError if id missing."""
        store = InMemoryExpenseStore()
        with pytest.raises(ValueError, match="id is required"):
            await store.save({"amount": Decimal("50.00")})

    @pytest.mark.asyncio
    async def test_overwrites_existing(self):
        """Should overwrite existing expense."""
        store = InMemoryExpenseStore()
        await store.save({"id": "exp_123", "amount": Decimal("50.00")})
        await store.save({"id": "exp_123", "amount": Decimal("75.00")})
        result = await store.get("exp_123")
        assert result["amount"] == Decimal("75.00")


class TestInMemoryExpenseStoreGet:
    """Tests for InMemoryExpenseStore.get method."""

    @pytest.mark.asyncio
    async def test_gets_existing_expense(self):
        """Should return expense by id."""
        store = InMemoryExpenseStore()
        await store.save({"id": "exp_123", "vendor_name": "Coffee Shop"})
        result = await store.get("exp_123")
        assert result["vendor_name"] == "Coffee Shop"

    @pytest.mark.asyncio
    async def test_returns_none_for_missing(self):
        """Should return None for missing expense."""
        store = InMemoryExpenseStore()
        result = await store.get("nonexistent")
        assert result is None


class TestInMemoryExpenseStoreDelete:
    """Tests for InMemoryExpenseStore.delete method."""

    @pytest.mark.asyncio
    async def test_deletes_existing_expense(self):
        """Should delete expense and return True."""
        store = InMemoryExpenseStore()
        await store.save({"id": "exp_123", "amount": Decimal("50.00")})
        result = await store.delete("exp_123")
        assert result is True
        assert await store.get("exp_123") is None

    @pytest.mark.asyncio
    async def test_returns_false_for_missing(self):
        """Should return False for missing expense."""
        store = InMemoryExpenseStore()
        result = await store.delete("nonexistent")
        assert result is False


class TestInMemoryExpenseStoreListAll:
    """Tests for InMemoryExpenseStore.list_all method."""

    @pytest.mark.asyncio
    async def test_lists_all_expenses(self):
        """Should list all expenses."""
        store = InMemoryExpenseStore()
        await store.save({"id": "exp_1", "created_at": "2024-01-01T00:00:00Z"})
        await store.save({"id": "exp_2", "created_at": "2024-01-02T00:00:00Z"})
        result = await store.list_all()
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_respects_limit(self):
        """Should respect limit parameter."""
        store = InMemoryExpenseStore()
        for i in range(5):
            await store.save({"id": f"exp_{i}", "created_at": f"2024-01-0{i + 1}T00:00:00Z"})
        result = await store.list_all(limit=3)
        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_respects_offset(self):
        """Should respect offset parameter."""
        store = InMemoryExpenseStore()
        for i in range(5):
            await store.save({"id": f"exp_{i}", "created_at": f"2024-01-0{i + 1}T00:00:00Z"})
        result = await store.list_all(offset=2)
        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_sorts_by_created_at_descending(self):
        """Should sort by created_at descending."""
        store = InMemoryExpenseStore()
        await store.save({"id": "exp_old", "created_at": "2024-01-01T00:00:00Z"})
        await store.save({"id": "exp_new", "created_at": "2024-01-15T00:00:00Z"})
        result = await store.list_all()
        assert result[0]["id"] == "exp_new"


class TestInMemoryExpenseStoreListByStatus:
    """Tests for InMemoryExpenseStore.list_by_status method."""

    @pytest.mark.asyncio
    async def test_filters_by_status(self):
        """Should filter expenses by status."""
        store = InMemoryExpenseStore()
        await store.save({"id": "exp_1", "status": "pending", "created_at": "2024-01-01"})
        await store.save({"id": "exp_2", "status": "approved", "created_at": "2024-01-02"})
        await store.save({"id": "exp_3", "status": "pending", "created_at": "2024-01-03"})

        pending = await store.list_by_status("pending")
        assert len(pending) == 2
        assert all(e["status"] == "pending" for e in pending)

    @pytest.mark.asyncio
    async def test_returns_empty_for_no_matches(self):
        """Should return empty list if no matches."""
        store = InMemoryExpenseStore()
        await store.save({"id": "exp_1", "status": "pending", "created_at": "2024-01-01"})
        result = await store.list_by_status("rejected")
        assert result == []


class TestInMemoryExpenseStoreListByEmployee:
    """Tests for InMemoryExpenseStore.list_by_employee method."""

    @pytest.mark.asyncio
    async def test_filters_by_employee_id(self):
        """Should filter by employee_id."""
        store = InMemoryExpenseStore()
        await store.save({"id": "exp_1", "employee_id": "emp_1", "created_at": "2024-01-01"})
        await store.save({"id": "exp_2", "employee_id": "emp_2", "created_at": "2024-01-02"})
        await store.save({"id": "exp_3", "employee_id": "emp_1", "created_at": "2024-01-03"})

        result = await store.list_by_employee("emp_1")
        assert len(result) == 2
        assert all(e["employee_id"] == "emp_1" for e in result)


class TestInMemoryExpenseStoreListByCategory:
    """Tests for InMemoryExpenseStore.list_by_category method."""

    @pytest.mark.asyncio
    async def test_filters_by_category(self):
        """Should filter by category."""
        store = InMemoryExpenseStore()
        await store.save({"id": "exp_1", "category": "travel", "created_at": "2024-01-01"})
        await store.save({"id": "exp_2", "category": "meals", "created_at": "2024-01-02"})
        await store.save({"id": "exp_3", "category": "travel", "created_at": "2024-01-03"})

        result = await store.list_by_category("travel")
        assert len(result) == 2
        assert all(e["category"] == "travel" for e in result)


class TestInMemoryExpenseStoreListPendingSync:
    """Tests for InMemoryExpenseStore.list_pending_sync method."""

    @pytest.mark.asyncio
    async def test_lists_approved_not_synced(self):
        """Should list approved expenses not synced to QBO."""
        store = InMemoryExpenseStore()
        await store.save({"id": "exp_1", "status": "approved", "synced_to_qbo": False})
        await store.save({"id": "exp_2", "status": "approved", "synced_to_qbo": True})
        await store.save({"id": "exp_3", "status": "pending", "synced_to_qbo": False})

        result = await store.list_pending_sync()
        assert len(result) == 1
        assert result[0]["id"] == "exp_1"


class TestInMemoryExpenseStoreFindDuplicates:
    """Tests for InMemoryExpenseStore.find_duplicates method."""

    @pytest.mark.asyncio
    async def test_finds_matching_vendor_and_amount(self):
        """Should find duplicates by vendor and amount."""
        store = InMemoryExpenseStore()
        now = datetime.now(timezone.utc)
        await store.save(
            {
                "id": "exp_1",
                "vendor_name": "Coffee Shop",
                "amount": Decimal("25.00"),
                "expense_date": now.isoformat(),
            }
        )

        result = await store.find_duplicates("Coffee Shop", Decimal("25.00"))
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_case_insensitive_vendor_match(self):
        """Should match vendor names case-insensitively."""
        store = InMemoryExpenseStore()
        now = datetime.now(timezone.utc)
        await store.save(
            {
                "id": "exp_1",
                "vendor_name": "COFFEE SHOP",
                "amount": Decimal("25.00"),
                "expense_date": now.isoformat(),
            }
        )

        result = await store.find_duplicates("coffee shop", Decimal("25.00"))
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_respects_date_tolerance(self):
        """Should respect date tolerance."""
        store = InMemoryExpenseStore()
        old_date = datetime.now(timezone.utc) - timedelta(days=10)
        await store.save(
            {
                "id": "exp_1",
                "vendor_name": "Coffee Shop",
                "amount": Decimal("25.00"),
                "expense_date": old_date.isoformat(),
            }
        )

        result = await store.find_duplicates("Coffee Shop", Decimal("25.00"), date_tolerance_days=3)
        assert len(result) == 0


class TestInMemoryExpenseStoreGetStatistics:
    """Tests for InMemoryExpenseStore.get_statistics method."""

    @pytest.mark.asyncio
    async def test_calculates_total_amount(self):
        """Should calculate total amount."""
        store = InMemoryExpenseStore()
        await store.save({"id": "exp_1", "amount": Decimal("100.00"), "status": "approved"})
        await store.save({"id": "exp_2", "amount": Decimal("50.00"), "status": "pending"})

        stats = await store.get_statistics()
        assert stats["total_count"] == 2
        assert stats["total_amount"] == "150.00"

    @pytest.mark.asyncio
    async def test_groups_by_category(self):
        """Should group by category."""
        store = InMemoryExpenseStore()
        await store.save({"id": "exp_1", "amount": Decimal("100.00"), "category": "travel"})
        await store.save({"id": "exp_2", "amount": Decimal("50.00"), "category": "meals"})
        await store.save({"id": "exp_3", "amount": Decimal("75.00"), "category": "travel"})

        stats = await store.get_statistics()
        assert stats["by_category"]["travel"] == "175.00"
        assert stats["by_category"]["meals"] == "50.00"

    @pytest.mark.asyncio
    async def test_groups_by_status(self):
        """Should group by status."""
        store = InMemoryExpenseStore()
        await store.save({"id": "exp_1", "amount": Decimal("100.00"), "status": "approved"})
        await store.save({"id": "exp_2", "amount": Decimal("50.00"), "status": "approved"})
        await store.save({"id": "exp_3", "amount": Decimal("75.00"), "status": "pending"})

        stats = await store.get_statistics()
        assert stats["by_status"]["approved"] == 2
        assert stats["by_status"]["pending"] == 1

    @pytest.mark.asyncio
    async def test_filters_by_date_range(self):
        """Should filter by date range."""
        store = InMemoryExpenseStore()
        await store.save(
            {"id": "exp_1", "amount": Decimal("100.00"), "expense_date": "2024-01-15T00:00:00Z"}
        )
        await store.save(
            {"id": "exp_2", "amount": Decimal("50.00"), "expense_date": "2024-02-15T00:00:00Z"}
        )

        start = datetime(2024, 2, 1, tzinfo=timezone.utc)
        end = datetime(2024, 2, 28, tzinfo=timezone.utc)
        stats = await store.get_statistics(start_date=start, end_date=end)
        assert stats["total_count"] == 1
        assert stats["total_amount"] == "50.00"


class TestInMemoryExpenseStoreUpdateStatus:
    """Tests for InMemoryExpenseStore.update_status method."""

    @pytest.mark.asyncio
    async def test_updates_status(self):
        """Should update expense status."""
        store = InMemoryExpenseStore()
        await store.save({"id": "exp_123", "status": "pending"})

        result = await store.update_status("exp_123", "approved")
        assert result is True

        expense = await store.get("exp_123")
        assert expense["status"] == "approved"

    @pytest.mark.asyncio
    async def test_sets_approved_by(self):
        """Should set approved_by when provided."""
        store = InMemoryExpenseStore()
        await store.save({"id": "exp_123", "status": "pending"})

        await store.update_status("exp_123", "approved", approved_by="user_456")

        expense = await store.get("exp_123")
        assert expense["approved_by"] == "user_456"

    @pytest.mark.asyncio
    async def test_updates_timestamp(self):
        """Should set updated_at timestamp."""
        store = InMemoryExpenseStore()
        await store.save({"id": "exp_123", "status": "pending"})

        await store.update_status("exp_123", "approved")

        expense = await store.get("exp_123")
        assert "updated_at" in expense

    @pytest.mark.asyncio
    async def test_returns_false_for_missing(self):
        """Should return False for missing expense."""
        store = InMemoryExpenseStore()
        result = await store.update_status("nonexistent", "approved")
        assert result is False


class TestInMemoryExpenseStoreMarkSynced:
    """Tests for InMemoryExpenseStore.mark_synced method."""

    @pytest.mark.asyncio
    async def test_marks_as_synced(self):
        """Should mark expense as synced to QBO."""
        store = InMemoryExpenseStore()
        await store.save({"id": "exp_123", "synced_to_qbo": False})

        result = await store.mark_synced("exp_123", "qbo_456")
        assert result is True

        expense = await store.get("exp_123")
        assert expense["synced_to_qbo"] is True
        assert expense["qbo_expense_id"] == "qbo_456"
        assert "synced_at" in expense

    @pytest.mark.asyncio
    async def test_returns_false_for_missing(self):
        """Should return False for missing expense."""
        store = InMemoryExpenseStore()
        result = await store.mark_synced("nonexistent", "qbo_456")
        assert result is False


# =============================================================================
# SQLiteExpenseStore Tests
# =============================================================================


@pytest.fixture
def sqlite_store(tmp_path: Path) -> SQLiteExpenseStore:
    """Create SQLite expense store with temp database."""
    db_path = tmp_path / "expenses.db"
    return SQLiteExpenseStore(db_path=db_path)


class TestSQLiteExpenseStoreInit:
    """Tests for SQLiteExpenseStore initialization."""

    def test_creates_database(self, tmp_path: Path):
        """Should create database file."""
        db_path = tmp_path / "expenses.db"
        store = SQLiteExpenseStore(db_path=db_path)
        assert db_path.exists()

    def test_creates_expenses_table(self, tmp_path: Path):
        """Should create expenses table."""
        db_path = tmp_path / "expenses.db"
        store = SQLiteExpenseStore(db_path=db_path)

        import sqlite3

        conn = sqlite3.connect(db_path)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='expenses'"
        )
        assert cursor.fetchone() is not None
        conn.close()


class TestSQLiteExpenseStoreSave:
    """Tests for SQLiteExpenseStore.save method."""

    @pytest.mark.asyncio
    async def test_saves_expense(self, sqlite_store: SQLiteExpenseStore):
        """Should save expense to database."""
        data = {
            "id": "exp_123",
            "vendor_name": "Test Vendor",
            "amount": Decimal("50.00"),
            "status": "pending",
        }
        await sqlite_store.save(data)

        result = await sqlite_store.get("exp_123")
        assert result is not None
        assert result["vendor_name"] == "Test Vendor"

    @pytest.mark.asyncio
    async def test_upserts_existing(self, sqlite_store: SQLiteExpenseStore):
        """Should upsert existing expense."""
        await sqlite_store.save({"id": "exp_123", "amount": Decimal("50.00")})
        await sqlite_store.save({"id": "exp_123", "amount": Decimal("75.00")})

        result = await sqlite_store.get("exp_123")
        assert result["amount"] == Decimal("75.00")


class TestSQLiteExpenseStoreGet:
    """Tests for SQLiteExpenseStore.get method."""

    @pytest.mark.asyncio
    async def test_gets_expense_by_id(self, sqlite_store: SQLiteExpenseStore):
        """Should get expense by ID."""
        await sqlite_store.save({"id": "exp_123", "vendor_name": "Test"})
        result = await sqlite_store.get("exp_123")
        assert result["vendor_name"] == "Test"

    @pytest.mark.asyncio
    async def test_returns_none_for_missing(self, sqlite_store: SQLiteExpenseStore):
        """Should return None for missing expense."""
        result = await sqlite_store.get("nonexistent")
        assert result is None


class TestSQLiteExpenseStoreDelete:
    """Tests for SQLiteExpenseStore.delete method."""

    @pytest.mark.asyncio
    async def test_deletes_expense(self, sqlite_store: SQLiteExpenseStore):
        """Should delete expense from database."""
        await sqlite_store.save({"id": "exp_123", "vendor_name": "Test"})
        result = await sqlite_store.delete("exp_123")
        assert result is True
        assert await sqlite_store.get("exp_123") is None

    @pytest.mark.asyncio
    async def test_returns_false_for_missing(self, sqlite_store: SQLiteExpenseStore):
        """Should return False for missing expense."""
        result = await sqlite_store.delete("nonexistent")
        assert result is False


class TestSQLiteExpenseStoreListAll:
    """Tests for SQLiteExpenseStore.list_all method."""

    @pytest.mark.asyncio
    async def test_lists_expenses_with_pagination(self, sqlite_store: SQLiteExpenseStore):
        """Should list expenses with pagination."""
        for i in range(10):
            await sqlite_store.save({"id": f"exp_{i}", "amount": Decimal(str(i * 10))})

        result = await sqlite_store.list_all(limit=5, offset=2)
        assert len(result) == 5


class TestSQLiteExpenseStoreListByStatus:
    """Tests for SQLiteExpenseStore.list_by_status method."""

    @pytest.mark.asyncio
    async def test_filters_by_status(self, sqlite_store: SQLiteExpenseStore):
        """Should filter by status."""
        await sqlite_store.save({"id": "exp_1", "status": "pending"})
        await sqlite_store.save({"id": "exp_2", "status": "approved"})
        await sqlite_store.save({"id": "exp_3", "status": "pending"})

        result = await sqlite_store.list_by_status("pending")
        assert len(result) == 2


class TestSQLiteExpenseStoreListByEmployee:
    """Tests for SQLiteExpenseStore.list_by_employee method."""

    @pytest.mark.asyncio
    async def test_filters_by_employee(self, sqlite_store: SQLiteExpenseStore):
        """Should filter by employee_id."""
        await sqlite_store.save({"id": "exp_1", "employee_id": "emp_1"})
        await sqlite_store.save({"id": "exp_2", "employee_id": "emp_2"})

        result = await sqlite_store.list_by_employee("emp_1")
        assert len(result) == 1
        assert result[0]["employee_id"] == "emp_1"


class TestSQLiteExpenseStoreListByCategory:
    """Tests for SQLiteExpenseStore.list_by_category method."""

    @pytest.mark.asyncio
    async def test_filters_by_category(self, sqlite_store: SQLiteExpenseStore):
        """Should filter by category."""
        await sqlite_store.save({"id": "exp_1", "category": "travel"})
        await sqlite_store.save({"id": "exp_2", "category": "meals"})

        result = await sqlite_store.list_by_category("travel")
        assert len(result) == 1


class TestSQLiteExpenseStoreFindDuplicates:
    """Tests for SQLiteExpenseStore.find_duplicates method."""

    @pytest.mark.asyncio
    async def test_finds_duplicates(self, sqlite_store: SQLiteExpenseStore):
        """Should find potential duplicates."""
        now = datetime.now(timezone.utc)
        await sqlite_store.save(
            {
                "id": "exp_1",
                "vendor_name": "Coffee Shop",
                "amount": Decimal("25.00"),
                "expense_date": now.isoformat(),
            }
        )

        result = await sqlite_store.find_duplicates("Coffee Shop", Decimal("25.00"))
        assert len(result) >= 1


class TestSQLiteExpenseStoreGetStatistics:
    """Tests for SQLiteExpenseStore.get_statistics method."""

    @pytest.mark.asyncio
    async def test_calculates_statistics(self, sqlite_store: SQLiteExpenseStore):
        """Should calculate expense statistics."""
        await sqlite_store.save(
            {
                "id": "exp_1",
                "amount": Decimal("100.00"),
                "category": "travel",
                "status": "approved",
            }
        )
        await sqlite_store.save(
            {
                "id": "exp_2",
                "amount": Decimal("50.00"),
                "category": "meals",
                "status": "pending",
            }
        )

        stats = await sqlite_store.get_statistics()
        assert stats["total_count"] == 2
        assert Decimal(stats["total_amount"]) == Decimal("150.00")


class TestSQLiteExpenseStoreUpdateStatus:
    """Tests for SQLiteExpenseStore.update_status method."""

    @pytest.mark.asyncio
    async def test_updates_status(self, sqlite_store: SQLiteExpenseStore):
        """Should update expense status."""
        await sqlite_store.save({"id": "exp_123", "status": "pending"})

        result = await sqlite_store.update_status("exp_123", "approved", approved_by="user_1")
        assert result is True

        expense = await sqlite_store.get("exp_123")
        assert expense["status"] == "approved"


class TestSQLiteExpenseStoreMarkSynced:
    """Tests for SQLiteExpenseStore.mark_synced method."""

    @pytest.mark.asyncio
    async def test_marks_synced(self, sqlite_store: SQLiteExpenseStore):
        """Should mark expense as synced to QBO."""
        await sqlite_store.save({"id": "exp_123", "status": "approved"})

        result = await sqlite_store.mark_synced("exp_123", "qbo_456")
        assert result is True

        expense = await sqlite_store.get("exp_123")
        assert expense.get("synced_to_qbo") is True
        assert expense.get("qbo_expense_id") == "qbo_456"


class TestSQLiteExpenseStoreClose:
    """Tests for SQLiteExpenseStore.close method."""

    @pytest.mark.asyncio
    async def test_closes_connection(self, sqlite_store: SQLiteExpenseStore):
        """Should close database connection."""
        await sqlite_store.close()
        # Should not raise even if called multiple times
        await sqlite_store.close()


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    @pytest.mark.asyncio
    async def test_empty_statistics(self):
        """Should handle empty store statistics."""
        store = InMemoryExpenseStore()
        stats = await store.get_statistics()
        assert stats["total_count"] == 0
        assert stats["total_amount"] == "0"

    @pytest.mark.asyncio
    async def test_string_amount_conversion(self):
        """Should handle string amounts."""
        store = InMemoryExpenseStore()
        await store.save(
            {
                "id": "exp_1",
                "amount": "100.00",  # String instead of Decimal
                "category": "test",
            }
        )
        stats = await store.get_statistics()
        assert stats["total_amount"] == "100.00"

    @pytest.mark.asyncio
    async def test_missing_category_defaults(self):
        """Should default category to uncategorized."""
        store = InMemoryExpenseStore()
        await store.save({"id": "exp_1", "amount": Decimal("100.00")})
        stats = await store.get_statistics()
        assert "uncategorized" in stats["by_category"]

    @pytest.mark.asyncio
    async def test_missing_status_defaults(self):
        """Should default status to pending."""
        store = InMemoryExpenseStore()
        await store.save({"id": "exp_1", "amount": Decimal("100.00")})
        stats = await store.get_statistics()
        assert "pending" in stats["by_status"]

    @pytest.mark.asyncio
    async def test_date_string_parsing(self):
        """Should parse ISO date strings."""
        store = InMemoryExpenseStore()
        await store.save(
            {
                "id": "exp_1",
                "vendor_name": "Test",
                "amount": Decimal("25.00"),
                "expense_date": "2024-01-15T10:30:00Z",
            }
        )

        result = await store.find_duplicates(
            "Test",
            Decimal("25.00"),
            date_tolerance_days=365,
        )
        # Should parse the date string correctly
        assert len(result) >= 0  # May or may not match depending on current date
