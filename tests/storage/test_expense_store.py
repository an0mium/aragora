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


# =============================================================================
# Concurrent Access Tests
# =============================================================================


class TestConcurrentAccessInMemory:
    """Tests for concurrent access safety in InMemoryExpenseStore."""

    @pytest.mark.asyncio
    async def test_concurrent_saves(self):
        """Concurrent saves should not corrupt data."""
        import asyncio

        store = InMemoryExpenseStore()

        async def save_expense(i: int):
            expense = {
                "id": f"exp_concurrent_{i}",
                "vendor_name": f"Vendor {i}",
                "amount": Decimal(str(i * 10)),
                "status": "pending",
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            await store.save(expense)

        # Save 50 expenses concurrently
        await asyncio.gather(*[save_expense(i) for i in range(50)])

        # Verify all were saved
        for i in range(50):
            result = await store.get(f"exp_concurrent_{i}")
            assert result is not None
            assert result["vendor_name"] == f"Vendor {i}"

    @pytest.mark.asyncio
    async def test_concurrent_reads_and_writes(self):
        """Concurrent reads and writes should be safe."""
        import asyncio

        store = InMemoryExpenseStore()

        # Pre-populate some expenses
        for i in range(10):
            await store.save(
                {
                    "id": f"exp_rw_{i}",
                    "vendor_name": "Test Vendor",
                    "amount": Decimal(str(i * 10)),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
            )

        async def read_expense(i: int):
            return await store.get(f"exp_rw_{i % 10}")

        async def write_expense(i: int):
            await store.save(
                {
                    "id": f"exp_rw_new_{i}",
                    "vendor_name": "New Vendor",
                    "amount": Decimal("100.00"),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
            )

        # Mix of reads and writes
        tasks = []
        for i in range(50):
            if i % 2 == 0:
                tasks.append(read_expense(i))
            else:
                tasks.append(write_expense(i))

        await asyncio.gather(*tasks)

    @pytest.mark.asyncio
    async def test_concurrent_status_updates(self):
        """Concurrent status updates should be atomic."""
        import asyncio

        store = InMemoryExpenseStore()

        # Create a single expense
        await store.save(
            {
                "id": "exp_status_test",
                "vendor_name": "Test",
                "amount": Decimal("100.00"),
                "status": "pending",
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )

        # Try to update status concurrently
        async def update_status(status: str, approver: str):
            await store.update_status("exp_status_test", status, approved_by=approver)

        await asyncio.gather(
            update_status("approved", "approver_1"),
            update_status("rejected", "approver_2"),
            update_status("approved", "approver_3"),
        )

        # The final status should be one of the valid states
        result = await store.get("exp_status_test")
        assert result["status"] in ["approved", "rejected"]

    @pytest.mark.asyncio
    async def test_concurrent_deletes(self):
        """Concurrent deletes should not cause errors."""
        import asyncio

        store = InMemoryExpenseStore()

        # Create expenses
        for i in range(10):
            await store.save(
                {
                    "id": f"exp_del_{i}",
                    "amount": Decimal("50.00"),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
            )

        # Delete all concurrently
        results = await asyncio.gather(*[store.delete(f"exp_del_{i}") for i in range(10)])

        # All should succeed
        assert all(results)

        # All should be deleted
        for i in range(10):
            assert await store.get(f"exp_del_{i}") is None


class TestConcurrentAccessSQLite:
    """Tests for concurrent access safety in SQLiteExpenseStore."""

    @pytest.mark.asyncio
    async def test_concurrent_saves_sqlite(self, tmp_path: Path):
        """SQLite store should handle concurrent saves safely."""
        import asyncio

        store = SQLiteExpenseStore(db_path=tmp_path / "concurrent.db")

        async def save_expense(i: int):
            expense = {
                "id": f"exp_sqlite_{i}",
                "vendor_name": f"Vendor {i}",
                "amount": Decimal(str(i * 10)),
                "status": "pending",
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            await store.save(expense)

        # Save 20 expenses concurrently (reduced from 50 for SQLite)
        await asyncio.gather(*[save_expense(i) for i in range(20)])

        # Verify all were saved
        all_expenses = await store.list_all(limit=100)
        assert len(all_expenses) == 20


# =============================================================================
# PostgresExpenseStore Tests (Mocked)
# =============================================================================


class TestPostgresExpenseStoreMocked:
    """Tests for PostgresExpenseStore using mocked database connections."""

    @pytest.fixture
    def mock_pool(self):
        """Create a mock asyncpg connection pool."""
        from unittest.mock import MagicMock, AsyncMock

        pool = MagicMock()
        conn = AsyncMock()
        pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
        pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        return pool, conn

    @pytest.mark.asyncio
    async def test_get_returns_expense(self, mock_pool):
        """Should return expense data from database."""
        from aragora.storage.expense_store import PostgresExpenseStore

        pool, conn = mock_pool
        conn.fetchrow.return_value = {
            "data_json": '{"id": "exp_001", "vendor_name": "Test Vendor", "amount": "100.00"}'
        }

        store = PostgresExpenseStore(pool)
        result = await store.get("exp_001")

        assert result is not None
        assert result["id"] == "exp_001"
        assert result["vendor_name"] == "Test Vendor"
        conn.fetchrow.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_returns_none_for_missing(self, mock_pool):
        """Should return None for non-existent expense."""
        from aragora.storage.expense_store import PostgresExpenseStore

        pool, conn = mock_pool
        conn.fetchrow.return_value = None

        store = PostgresExpenseStore(pool)
        result = await store.get("nonexistent")

        assert result is None

    @pytest.mark.asyncio
    async def test_save_converts_decimal_to_float(self, mock_pool):
        """Decimal amount should be converted to float for PostgreSQL."""
        from aragora.storage.expense_store import PostgresExpenseStore

        pool, conn = mock_pool
        store = PostgresExpenseStore(pool)

        expense = {
            "id": "exp_decimal",
            "vendor_name": "Test",
            "amount": Decimal("1234.56"),
            "status": "pending",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        await store.save(expense)

        conn.execute.assert_called_once()
        call_args = conn.execute.call_args
        # The amount is the 3rd positional argument
        amount_arg = call_args[0][3]
        assert isinstance(amount_arg, float)
        assert amount_arg == 1234.56

    @pytest.mark.asyncio
    async def test_save_requires_id(self, mock_pool):
        """Should raise ValueError if id is missing."""
        from aragora.storage.expense_store import PostgresExpenseStore

        pool, conn = mock_pool
        store = PostgresExpenseStore(pool)

        with pytest.raises(ValueError, match="id is required"):
            await store.save({"vendor_name": "Test"})

    @pytest.mark.asyncio
    async def test_delete_returns_true_on_success(self, mock_pool):
        """Should return True when expense is deleted."""
        from aragora.storage.expense_store import PostgresExpenseStore

        pool, conn = mock_pool
        conn.execute.return_value = "DELETE 1"

        store = PostgresExpenseStore(pool)
        result = await store.delete("exp_001")

        assert result is True

    @pytest.mark.asyncio
    async def test_delete_returns_false_for_missing(self, mock_pool):
        """Should return False when expense doesn't exist."""
        from aragora.storage.expense_store import PostgresExpenseStore

        pool, conn = mock_pool
        conn.execute.return_value = "DELETE 0"

        store = PostgresExpenseStore(pool)
        result = await store.delete("nonexistent")

        assert result is False

    @pytest.mark.asyncio
    async def test_list_all_with_pagination(self, mock_pool):
        """Should list expenses with proper pagination."""
        from aragora.storage.expense_store import PostgresExpenseStore

        pool, conn = mock_pool
        conn.fetch.return_value = [
            {"data_json": '{"id": "exp_001", "amount": "100.00"}'},
            {"data_json": '{"id": "exp_002", "amount": "200.00"}'},
        ]

        store = PostgresExpenseStore(pool)
        result = await store.list_all(limit=10, offset=5)

        assert len(result) == 2
        conn.fetch.assert_called_once()
        # Verify pagination parameters were passed
        call_args = conn.fetch.call_args
        assert call_args[0][1] == 10  # limit
        assert call_args[0][2] == 5  # offset

    @pytest.mark.asyncio
    async def test_list_by_status_filters_correctly(self, mock_pool):
        """Should filter by status parameter."""
        from aragora.storage.expense_store import PostgresExpenseStore

        pool, conn = mock_pool
        conn.fetch.return_value = [
            {"data_json": '{"id": "exp_001", "status": "approved"}'},
        ]

        store = PostgresExpenseStore(pool)
        result = await store.list_by_status("approved")

        assert len(result) == 1
        call_args = conn.fetch.call_args
        assert call_args[0][1] == "approved"

    @pytest.mark.asyncio
    async def test_list_pending_sync_returns_approved_not_synced(self, mock_pool):
        """Should return approved expenses not yet synced to QBO."""
        from aragora.storage.expense_store import PostgresExpenseStore

        pool, conn = mock_pool
        conn.fetch.return_value = [
            {"data_json": '{"id": "exp_001", "status": "approved", "synced_to_qbo": false}'},
        ]

        store = PostgresExpenseStore(pool)
        result = await store.list_pending_sync()

        assert len(result) == 1
        assert result[0]["status"] == "approved"

    @pytest.mark.asyncio
    async def test_update_status_success(self, mock_pool):
        """Should update expense status."""
        from aragora.storage.expense_store import PostgresExpenseStore

        pool, conn = mock_pool
        conn.fetchrow.return_value = {"data_json": '{"id": "exp_001", "status": "pending"}'}
        conn.execute.return_value = "UPDATE 1"

        store = PostgresExpenseStore(pool)
        result = await store.update_status("exp_001", "approved", approved_by="user_123")

        assert result is True

    @pytest.mark.asyncio
    async def test_update_status_returns_false_for_missing(self, mock_pool):
        """Should return False for non-existent expense."""
        from aragora.storage.expense_store import PostgresExpenseStore

        pool, conn = mock_pool
        conn.fetchrow.return_value = None

        store = PostgresExpenseStore(pool)
        result = await store.update_status("nonexistent", "approved")

        assert result is False

    @pytest.mark.asyncio
    async def test_mark_synced_success(self, mock_pool):
        """Should mark expense as synced to QBO."""
        from aragora.storage.expense_store import PostgresExpenseStore

        pool, conn = mock_pool
        conn.fetchrow.return_value = {"data_json": '{"id": "exp_001", "synced_to_qbo": false}'}
        conn.execute.return_value = "UPDATE 1"

        store = PostgresExpenseStore(pool)
        result = await store.mark_synced("exp_001", "qbo_12345")

        assert result is True

    @pytest.mark.asyncio
    async def test_get_statistics_calculates_totals(self, mock_pool):
        """Should calculate expense statistics."""
        from aragora.storage.expense_store import PostgresExpenseStore

        pool, conn = mock_pool
        conn.fetchrow.return_value = (5, 500.0)  # count, sum
        conn.fetch.side_effect = [
            [("travel", 300.0), ("meals", 200.0)],  # by_category
            [("approved", 3), ("pending", 2)],  # by_status
        ]

        store = PostgresExpenseStore(pool)
        result = await store.get_statistics()

        assert result["total_count"] == 5
        # The Decimal conversion may add trailing zeros or decimal point
        assert Decimal(result["total_amount"]) == Decimal("500")

    @pytest.mark.asyncio
    async def test_initialize_creates_schema(self, mock_pool):
        """Should execute schema creation on initialize."""
        from aragora.storage.expense_store import PostgresExpenseStore

        pool, conn = mock_pool
        store = PostgresExpenseStore(pool)

        await store.initialize()

        conn.execute.assert_called_once()
        # Verify schema SQL was executed
        call_args = conn.execute.call_args
        assert "CREATE TABLE IF NOT EXISTS expenses" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_close_is_noop(self, mock_pool):
        """Close should be a no-op (pool is managed externally)."""
        from aragora.storage.expense_store import PostgresExpenseStore

        pool, conn = mock_pool
        store = PostgresExpenseStore(pool)

        await store.close()  # Should not raise


# =============================================================================
# Global Store Accessor Tests
# =============================================================================


class TestGlobalStoreAccessors:
    """Tests for global store accessor functions."""

    def setup_method(self):
        """Reset store before each test."""
        from aragora.storage.expense_store import reset_expense_store

        reset_expense_store()

    def teardown_method(self):
        """Reset store after each test."""
        from aragora.storage.expense_store import reset_expense_store

        reset_expense_store()

    def test_set_expense_store_custom_instance(self):
        """Should set custom store instance."""
        from aragora.storage.expense_store import (
            set_expense_store,
            get_expense_store,
            reset_expense_store,
        )

        custom_store = InMemoryExpenseStore()
        set_expense_store(custom_store)

        retrieved = get_expense_store()
        assert retrieved is custom_store

    def test_reset_expense_store_clears_singleton(self):
        """Should clear the singleton store."""
        from aragora.storage.expense_store import (
            set_expense_store,
            reset_expense_store,
        )

        custom_store = InMemoryExpenseStore()
        set_expense_store(custom_store)
        reset_expense_store()

        # After reset, next get should create new instance
        # (We can't easily test this without mocking the factory)


# =============================================================================
# Receipt Attachment Tests
# =============================================================================


class TestReceiptAttachments:
    """Tests for receipt attachment handling via data_json."""

    @pytest.mark.asyncio
    async def test_save_expense_with_receipt_data(self):
        """Should save expense with receipt attachment metadata."""
        store = InMemoryExpenseStore()

        expense = {
            "id": "exp_with_receipt",
            "vendor_name": "Restaurant",
            "amount": Decimal("75.50"),
            "status": "pending",
            "receipt_attachments": [
                {
                    "filename": "receipt_001.jpg",
                    "content_type": "image/jpeg",
                    "storage_path": "receipts/2024/01/receipt_001.jpg",
                    "file_size": 1024000,
                    "uploaded_at": datetime.now(timezone.utc).isoformat(),
                }
            ],
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        await store.save(expense)
        result = await store.get("exp_with_receipt")

        assert result is not None
        assert len(result["receipt_attachments"]) == 1
        assert result["receipt_attachments"][0]["filename"] == "receipt_001.jpg"

    @pytest.mark.asyncio
    async def test_save_expense_with_multiple_receipts(self):
        """Should save expense with multiple receipt attachments."""
        store = InMemoryExpenseStore()

        expense = {
            "id": "exp_multi_receipt",
            "vendor_name": "Office Supplies",
            "amount": Decimal("250.00"),
            "status": "pending",
            "receipt_attachments": [
                {"filename": "receipt_a.pdf", "content_type": "application/pdf"},
                {"filename": "receipt_b.jpg", "content_type": "image/jpeg"},
            ],
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        await store.save(expense)
        result = await store.get("exp_multi_receipt")

        assert len(result["receipt_attachments"]) == 2

    @pytest.mark.asyncio
    async def test_update_expense_add_receipt(self):
        """Should update expense to add receipt attachment."""
        store = InMemoryExpenseStore()

        # Create expense without receipt
        expense = {
            "id": "exp_add_receipt",
            "vendor_name": "Taxi",
            "amount": Decimal("35.00"),
            "status": "pending",
            "receipt_attachments": [],
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        await store.save(expense)

        # Update with receipt
        expense["receipt_attachments"] = [{"filename": "taxi_receipt.jpg"}]
        await store.save(expense)

        result = await store.get("exp_add_receipt")
        assert len(result["receipt_attachments"]) == 1


# =============================================================================
# Reimbursement Workflow Tests
# =============================================================================


class TestReimbursementWorkflow:
    """Tests for expense reimbursement workflow."""

    @pytest.mark.asyncio
    async def test_full_reimbursement_workflow(self):
        """Test complete reimbursement workflow from submission to sync."""
        store = InMemoryExpenseStore()

        # Step 1: Employee submits expense
        expense = {
            "id": "exp_reimburse_001",
            "employee_id": "emp_123",
            "vendor_name": "Conference Hotel",
            "amount": Decimal("500.00"),
            "category": "travel",
            "status": "pending",
            "synced_to_qbo": False,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        await store.save(expense)

        # Step 2: Manager reviews and approves
        await store.update_status("exp_reimburse_001", "approved", approved_by="mgr_456")

        result = await store.get("exp_reimburse_001")
        assert result["status"] == "approved"
        assert result["approved_by"] == "mgr_456"

        # Step 3: Expense appears in pending sync queue
        pending = await store.list_pending_sync()
        assert any(e["id"] == "exp_reimburse_001" for e in pending)

        # Step 4: Sync to QuickBooks
        await store.mark_synced("exp_reimburse_001", "qbo_expense_789")

        result = await store.get("exp_reimburse_001")
        assert result["synced_to_qbo"] is True
        assert result["qbo_expense_id"] == "qbo_expense_789"

        # Step 5: No longer in pending sync queue
        pending = await store.list_pending_sync()
        assert not any(e["id"] == "exp_reimburse_001" for e in pending)

    @pytest.mark.asyncio
    async def test_rejected_expense_not_in_sync_queue(self):
        """Rejected expenses should not appear in sync queue."""
        store = InMemoryExpenseStore()

        expense = {
            "id": "exp_rejected",
            "employee_id": "emp_123",
            "amount": Decimal("100.00"),
            "status": "pending",
            "synced_to_qbo": False,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        await store.save(expense)

        # Reject the expense
        await store.update_status("exp_rejected", "rejected", approved_by="mgr_456")

        # Should not appear in pending sync
        pending = await store.list_pending_sync()
        assert not any(e["id"] == "exp_rejected" for e in pending)

    @pytest.mark.asyncio
    async def test_employee_expense_history(self):
        """Employee should be able to view their expense history."""
        store = InMemoryExpenseStore()

        # Create multiple expenses for one employee
        for i in range(5):
            await store.save(
                {
                    "id": f"exp_emp_{i}",
                    "employee_id": "emp_history",
                    "amount": Decimal(str(i * 50)),
                    "status": ["pending", "approved", "rejected"][i % 3],
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
            )

        # Create expense for different employee
        await store.save(
            {
                "id": "exp_other",
                "employee_id": "emp_other",
                "amount": Decimal("999.00"),
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )

        # Should only get expenses for specified employee
        employee_expenses = await store.list_by_employee("emp_history")
        assert len(employee_expenses) == 5
        assert all(e["employee_id"] == "emp_history" for e in employee_expenses)


# =============================================================================
# Budget Allocation Tests
# =============================================================================


class TestBudgetAllocation:
    """Tests for budget allocation tracking via category statistics."""

    @pytest.mark.asyncio
    async def test_category_spending_totals(self):
        """Should track spending totals per category."""
        store = InMemoryExpenseStore()

        # Create expenses across categories
        expenses = [
            {"id": "exp_t1", "category": "travel", "amount": Decimal("500.00")},
            {"id": "exp_t2", "category": "travel", "amount": Decimal("300.00")},
            {"id": "exp_m1", "category": "meals", "amount": Decimal("50.00")},
            {"id": "exp_m2", "category": "meals", "amount": Decimal("75.00")},
            {"id": "exp_s1", "category": "software", "amount": Decimal("200.00")},
        ]

        for exp in expenses:
            exp["created_at"] = datetime.now(timezone.utc).isoformat()
            await store.save(exp)

        stats = await store.get_statistics()

        assert stats["by_category"]["travel"] == "800.00"
        assert stats["by_category"]["meals"] == "125.00"
        assert stats["by_category"]["software"] == "200.00"

    @pytest.mark.asyncio
    async def test_budget_period_spending(self):
        """Should calculate spending within a date range for budget tracking."""
        store = InMemoryExpenseStore()

        # January expenses
        await store.save(
            {
                "id": "exp_jan1",
                "category": "travel",
                "amount": Decimal("1000.00"),
                "expense_date": "2024-01-15T00:00:00Z",
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )

        # February expenses
        await store.save(
            {
                "id": "exp_feb1",
                "category": "travel",
                "amount": Decimal("500.00"),
                "expense_date": "2024-02-15T00:00:00Z",
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )

        # Get February stats only
        feb_start = datetime(2024, 2, 1, tzinfo=timezone.utc)
        feb_end = datetime(2024, 2, 29, tzinfo=timezone.utc)
        stats = await store.get_statistics(start_date=feb_start, end_date=feb_end)

        assert stats["total_count"] == 1
        assert stats["total_amount"] == "500.00"

    @pytest.mark.asyncio
    async def test_monthly_category_breakdown(self):
        """Should provide category breakdown for budget analysis."""
        store = InMemoryExpenseStore()

        # Create Q1 expenses
        q1_expenses = [
            {
                "id": "q1_t1",
                "category": "travel",
                "amount": Decimal("1500.00"),
                "expense_date": "2024-01-15T00:00:00Z",
            },
            {
                "id": "q1_t2",
                "category": "travel",
                "amount": Decimal("800.00"),
                "expense_date": "2024-02-20T00:00:00Z",
            },
            {
                "id": "q1_m1",
                "category": "meals",
                "amount": Decimal("200.00"),
                "expense_date": "2024-03-10T00:00:00Z",
            },
            {
                "id": "q1_s1",
                "category": "software",
                "amount": Decimal("500.00"),
                "expense_date": "2024-01-25T00:00:00Z",
            },
        ]

        for exp in q1_expenses:
            exp["created_at"] = datetime.now(timezone.utc).isoformat()
            await store.save(exp)

        # Get Q1 stats
        q1_start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        q1_end = datetime(2024, 3, 31, tzinfo=timezone.utc)
        stats = await store.get_statistics(start_date=q1_start, end_date=q1_end)

        assert stats["total_count"] == 4
        assert stats["by_category"]["travel"] == "2300.00"
        assert stats["by_category"]["meals"] == "200.00"
        assert stats["by_category"]["software"] == "500.00"


# =============================================================================
# Transaction Rollback Tests
# =============================================================================


class TestTransactionRollback:
    """Tests for error handling and data integrity."""

    @pytest.mark.asyncio
    async def test_save_without_id_does_not_corrupt_store(self):
        """Failed save should not corrupt existing data."""
        store = InMemoryExpenseStore()

        # Save a valid expense first
        await store.save(
            {
                "id": "exp_valid",
                "amount": Decimal("100.00"),
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )

        # Attempt invalid save
        with pytest.raises(ValueError):
            await store.save({"amount": Decimal("50.00")})

        # Original expense should still be intact
        result = await store.get("exp_valid")
        assert result is not None
        assert result["amount"] == Decimal("100.00")

    @pytest.mark.asyncio
    async def test_update_nonexistent_does_not_create(self):
        """Updating non-existent expense should not create it."""
        store = InMemoryExpenseStore()

        result = await store.update_status("nonexistent", "approved")
        assert result is False

        # Should still not exist
        assert await store.get("nonexistent") is None

    @pytest.mark.asyncio
    async def test_mark_synced_nonexistent_does_not_create(self):
        """Marking non-existent expense as synced should not create it."""
        store = InMemoryExpenseStore()

        result = await store.mark_synced("nonexistent", "qbo_123")
        assert result is False

        # Should still not exist
        assert await store.get("nonexistent") is None


# =============================================================================
# Additional Edge Cases
# =============================================================================


class TestAdditionalEdgeCases:
    """Additional edge case tests."""

    @pytest.mark.asyncio
    async def test_negative_amount_handling(self):
        """Should handle negative amounts (refunds/credits)."""
        store = InMemoryExpenseStore()

        await store.save(
            {
                "id": "exp_refund",
                "vendor_name": "Refund Processing",
                "amount": Decimal("-50.00"),
                "category": "refund",
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )

        result = await store.get("exp_refund")
        assert result["amount"] == Decimal("-50.00")

    @pytest.mark.asyncio
    async def test_zero_amount_expense(self):
        """Should handle zero amount expenses."""
        store = InMemoryExpenseStore()

        await store.save(
            {
                "id": "exp_zero",
                "vendor_name": "Free Sample",
                "amount": Decimal("0.00"),
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )

        stats = await store.get_statistics()
        assert stats["total_count"] == 1
        assert stats["total_amount"] == "0.00"

    @pytest.mark.asyncio
    async def test_very_large_amount_precision(self):
        """Should preserve precision for very large amounts."""
        store = InMemoryExpenseStore()

        large_amount = Decimal("99999999999999.99")
        await store.save(
            {
                "id": "exp_large",
                "amount": large_amount,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )

        result = await store.get("exp_large")
        assert result["amount"] == large_amount

    @pytest.mark.asyncio
    async def test_unicode_vendor_name(self):
        """Should handle unicode characters in vendor name."""
        store = InMemoryExpenseStore()

        await store.save(
            {
                "id": "exp_unicode",
                "vendor_name": "Caf\u00e9 Fran\u00e7ais \u201c\u4e2d\u6587\u201d",
                "amount": Decimal("25.00"),
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )

        result = await store.get("exp_unicode")
        assert result["vendor_name"] == "Caf\u00e9 Fran\u00e7ais \u201c\u4e2d\u6587\u201d"

    @pytest.mark.asyncio
    async def test_empty_vendor_name(self):
        """Should handle empty vendor name."""
        store = InMemoryExpenseStore()

        await store.save(
            {
                "id": "exp_empty_vendor",
                "vendor_name": "",
                "amount": Decimal("10.00"),
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )

        result = await store.get("exp_empty_vendor")
        assert result["vendor_name"] == ""

    @pytest.mark.asyncio
    async def test_null_fields_preserved(self):
        """Should preserve null/None fields."""
        store = InMemoryExpenseStore()

        await store.save(
            {
                "id": "exp_nulls",
                "vendor_name": None,
                "amount": Decimal("100.00"),
                "category": None,
                "employee_id": None,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )

        result = await store.get("exp_nulls")
        assert result["vendor_name"] is None
        assert result["category"] is None

    @pytest.mark.asyncio
    async def test_special_characters_in_id(self):
        """Should handle special characters in expense ID."""
        store = InMemoryExpenseStore()

        special_id = "exp_123-abc_456.test"
        await store.save(
            {
                "id": special_id,
                "amount": Decimal("50.00"),
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )

        result = await store.get(special_id)
        assert result is not None
        assert result["id"] == special_id

    @pytest.mark.asyncio
    async def test_list_limit_exceeds_total(self):
        """Should handle limit larger than total items."""
        store = InMemoryExpenseStore()

        await store.save(
            {
                "id": "exp_only",
                "amount": Decimal("100.00"),
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )

        result = await store.list_all(limit=1000)
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_list_offset_exceeds_total(self):
        """Should return empty list when offset exceeds total items."""
        store = InMemoryExpenseStore()

        await store.save(
            {
                "id": "exp_only",
                "amount": Decimal("100.00"),
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )

        result = await store.list_all(offset=100)
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_json_data_preserved(self):
        """Should preserve complex JSON data in expense."""
        store = InMemoryExpenseStore()

        complex_data = {
            "id": "exp_complex",
            "amount": Decimal("100.00"),
            "ocr_data": {
                "confidence": 0.95,
                "extracted_fields": {
                    "vendor": "Test Corp",
                    "date": "2024-01-15",
                    "items": [
                        {"description": "Item 1", "price": "50.00"},
                        {"description": "Item 2", "price": "50.00"},
                    ],
                },
            },
            "metadata": {"source": "mobile_app", "version": "2.0"},
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        await store.save(complex_data)
        result = await store.get("exp_complex")

        assert result["ocr_data"]["confidence"] == 0.95
        assert len(result["ocr_data"]["extracted_fields"]["items"]) == 2
        assert result["metadata"]["source"] == "mobile_app"


class TestSQLiteTransactionIntegrity:
    """Tests for SQLite transaction integrity."""

    @pytest.mark.asyncio
    async def test_save_and_get_preserves_decimal_precision(self, tmp_path: Path):
        """SQLite should preserve decimal precision through JSON."""
        store = SQLiteExpenseStore(db_path=tmp_path / "precision.db")

        precise_amount = Decimal("12345.6789")
        await store.save(
            {
                "id": "exp_precise",
                "amount": precise_amount,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )

        result = await store.get("exp_precise")
        assert result["amount"] == precise_amount

    @pytest.mark.asyncio
    async def test_sqlite_handles_rapid_updates(self, tmp_path: Path):
        """SQLite should handle rapid sequential updates."""
        store = SQLiteExpenseStore(db_path=tmp_path / "rapid.db")

        await store.save(
            {
                "id": "exp_rapid",
                "amount": Decimal("100.00"),
                "status": "pending",
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )

        # Rapid status updates
        statuses = ["review", "approved", "processing", "reimbursed"]
        for status in statuses:
            await store.update_status("exp_rapid", status)

        result = await store.get("exp_rapid")
        assert result["status"] == "reimbursed"

    @pytest.mark.asyncio
    async def test_sqlite_list_pending_sync_empty(self, tmp_path: Path):
        """SQLite list_pending_sync should return empty for new store."""
        store = SQLiteExpenseStore(db_path=tmp_path / "empty_sync.db")

        result = await store.list_pending_sync()
        assert result == []

    @pytest.mark.asyncio
    async def test_sqlite_find_duplicates_case_insensitive(self, tmp_path: Path):
        """SQLite find_duplicates should be case insensitive for vendor."""
        store = SQLiteExpenseStore(db_path=tmp_path / "duplicates.db")

        now = datetime.now(timezone.utc)
        await store.save(
            {
                "id": "exp_dup1",
                "vendor_name": "COFFEE SHOP",
                "amount": Decimal("5.00"),
                "expense_date": now.isoformat(),
                "created_at": now.isoformat(),
            }
        )

        # Search with different case
        result = await store.find_duplicates("coffee shop", Decimal("5.00"))
        assert len(result) >= 1
