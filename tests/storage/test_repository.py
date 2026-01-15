"""
Tests for aragora.storage.repository - Base repository pattern.

Tests cover:
- Column name validation
- DatabaseRepository base class
- CRUD operations (exists, count, get_by_id, get_all, batch_get, delete)
- Change notification callbacks
- SQL injection prevention
"""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aragora.storage.repository import (
    DatabaseRepository,
    _validate_column_name,
    _SQL_IDENTIFIER_PATTERN,
)


# ===========================================================================
# Test Column Name Validation
# ===========================================================================


class TestValidateColumnName:
    """Tests for _validate_column_name function."""

    def test_valid_simple_name(self):
        assert _validate_column_name("id") == "id"
        assert _validate_column_name("name") == "name"
        assert _validate_column_name("user_id") == "user_id"

    def test_valid_underscore_prefix(self):
        assert _validate_column_name("_id") == "_id"
        assert _validate_column_name("_private") == "_private"

    def test_valid_mixed_case(self):
        assert _validate_column_name("firstName") == "firstName"
        assert _validate_column_name("LastName") == "LastName"
        assert _validate_column_name("createdAt") == "createdAt"

    def test_valid_with_numbers(self):
        assert _validate_column_name("field1") == "field1"
        assert _validate_column_name("col_123") == "col_123"

    def test_invalid_empty(self):
        with pytest.raises(ValueError, match="Invalid column name length"):
            _validate_column_name("")

    def test_invalid_too_long(self):
        long_name = "a" * 65
        with pytest.raises(ValueError, match="Invalid column name length"):
            _validate_column_name(long_name)

    def test_invalid_starts_with_number(self):
        with pytest.raises(ValueError, match="Invalid column name"):
            _validate_column_name("1column")

    def test_invalid_special_chars(self):
        with pytest.raises(ValueError, match="Invalid column name"):
            _validate_column_name("user-id")
        with pytest.raises(ValueError, match="Invalid column name"):
            _validate_column_name("table.column")
        with pytest.raises(ValueError, match="Invalid column name"):
            _validate_column_name("col;umn")

    def test_sql_injection_attempts(self):
        with pytest.raises(ValueError, match="Invalid column name"):
            _validate_column_name("id; DROP TABLE users;--")
        with pytest.raises(ValueError, match="Invalid column name"):
            _validate_column_name("id OR 1=1")
        with pytest.raises(ValueError, match="Invalid column name"):
            _validate_column_name("id' OR '1'='1")


class TestSqlIdentifierPattern:
    """Tests for _SQL_IDENTIFIER_PATTERN regex."""

    def test_matches_valid_identifiers(self):
        assert _SQL_IDENTIFIER_PATTERN.match("id")
        assert _SQL_IDENTIFIER_PATTERN.match("_id")
        assert _SQL_IDENTIFIER_PATTERN.match("user_name")
        assert _SQL_IDENTIFIER_PATTERN.match("Column1")
        assert _SQL_IDENTIFIER_PATTERN.match("ABC123")

    def test_rejects_invalid_identifiers(self):
        assert not _SQL_IDENTIFIER_PATTERN.match("123abc")
        assert not _SQL_IDENTIFIER_PATTERN.match("user-name")
        assert not _SQL_IDENTIFIER_PATTERN.match("table.column")
        assert not _SQL_IDENTIFIER_PATTERN.match("")
        assert not _SQL_IDENTIFIER_PATTERN.match(" name")


# ===========================================================================
# Test Repository Base Class
# ===========================================================================


class ConcreteRepository(DatabaseRepository):
    """Concrete implementation for testing."""

    TABLE_NAME = "test_items"

    def _init_schema(self):
        with self.connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS test_items (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    value INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )
            conn.commit()

    def insert(self, id: str, name: str, value: int = 0) -> None:
        with self.connection() as conn:
            conn.execute(
                "INSERT INTO test_items (id, name, value) VALUES (?, ?, ?)", (id, name, value)
            )
            conn.commit()
        self._notify_change("insert")


@pytest.fixture
def db_path(tmp_path):
    """Create temp database path."""
    return tmp_path / "test.db"


@pytest.fixture
def repo(db_path):
    """Create repository with test data."""
    r = ConcreteRepository(db_path)
    r.insert("item-1", "First Item", 100)
    r.insert("item-2", "Second Item", 200)
    r.insert("item-3", "Third Item", 300)
    return r


class TestDatabaseRepositoryInit:
    """Tests for DatabaseRepository initialization."""

    def test_init_creates_db_path(self, db_path):
        repo = ConcreteRepository(db_path)
        assert repo.db_path == str(db_path)

    def test_init_with_auto_init(self, db_path):
        repo = ConcreteRepository(db_path, auto_init=True)
        # Table should exist
        with repo.connection() as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='test_items'"
            )
            assert cursor.fetchone() is not None

    def test_init_without_auto_init(self, db_path):
        repo = ConcreteRepository(db_path, auto_init=False)
        # Table should not exist
        with repo.connection() as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='test_items'"
            )
            assert cursor.fetchone() is None


class TestDatabaseRepositoryExists:
    """Tests for exists() method."""

    def test_exists_returns_true(self, repo):
        assert repo.exists("item-1") is True

    def test_exists_returns_false(self, repo):
        assert repo.exists("nonexistent") is False

    def test_exists_with_custom_column(self, repo):
        assert repo.exists("First Item", id_column="name") is True

    def test_exists_validates_column_name(self, repo):
        with pytest.raises(ValueError, match="Invalid column name"):
            repo.exists("value", id_column="invalid;column")


class TestDatabaseRepositoryCount:
    """Tests for count() method."""

    def test_count_all(self, repo):
        assert repo.count() == 3

    def test_count_with_where(self, repo):
        count = repo.count(where="value > ?", params=(150,))
        assert count == 2

    def test_count_empty_result(self, repo):
        count = repo.count(where="value > ?", params=(1000,))
        assert count == 0


class TestDatabaseRepositoryGetById:
    """Tests for get_by_id() method."""

    def test_get_existing(self, repo):
        result = repo.get_by_id("item-1")
        assert result is not None
        assert result["id"] == "item-1"
        assert result["name"] == "First Item"
        assert result["value"] == 100

    def test_get_nonexistent(self, repo):
        result = repo.get_by_id("nonexistent")
        assert result is None

    def test_get_with_custom_column(self, repo):
        result = repo.get_by_id("Second Item", id_column="name")
        assert result is not None
        assert result["id"] == "item-2"


class TestDatabaseRepositoryGetAll:
    """Tests for get_all() method."""

    def test_get_all_default(self, repo):
        results = repo.get_all()
        assert len(results) == 3

    def test_get_all_with_limit(self, repo):
        results = repo.get_all(limit=2)
        assert len(results) == 2

    def test_get_all_with_offset(self, repo):
        results = repo.get_all(limit=2, offset=1)
        assert len(results) == 2

    def test_get_all_with_where(self, repo):
        results = repo.get_all(where="value >= ?", params=(200,))
        assert len(results) == 2

    def test_get_all_with_order_by_asc(self, repo):
        results = repo.get_all(order_by="value ASC")
        assert results[0]["value"] == 100
        assert results[-1]["value"] == 300

    def test_get_all_with_order_by_desc(self, repo):
        results = repo.get_all(order_by="value DESC")
        assert results[0]["value"] == 300
        assert results[-1]["value"] == 100

    def test_get_all_order_by_multiple(self, repo):
        results = repo.get_all(order_by="name ASC, value DESC")
        assert len(results) == 3

    def test_get_all_order_by_invalid_direction(self, repo):
        with pytest.raises(ValueError, match="Invalid sort direction"):
            repo.get_all(order_by="value SIDEWAYS")

    def test_get_all_order_by_invalid_column(self, repo):
        with pytest.raises(ValueError, match="Invalid column name"):
            repo.get_all(order_by="bad;column")


class TestDatabaseRepositoryBatchGet:
    """Tests for batch_get() method."""

    def test_batch_get_multiple(self, repo):
        results = repo.batch_get(["item-1", "item-3"])
        assert len(results) == 2
        ids = {r["id"] for r in results}
        assert ids == {"item-1", "item-3"}

    def test_batch_get_empty_list(self, repo):
        results = repo.batch_get([])
        assert results == []

    def test_batch_get_with_nonexistent(self, repo):
        results = repo.batch_get(["item-1", "nonexistent"])
        assert len(results) == 1
        assert results[0]["id"] == "item-1"

    def test_batch_get_custom_column(self, repo):
        results = repo.batch_get(["First Item", "Third Item"], id_column="name")
        assert len(results) == 2


class TestDatabaseRepositoryDelete:
    """Tests for delete methods."""

    def test_delete_by_id_existing(self, repo):
        assert repo.exists("item-1") is True
        result = repo.delete_by_id("item-1")
        assert result is True
        assert repo.exists("item-1") is False

    def test_delete_by_id_nonexistent(self, repo):
        result = repo.delete_by_id("nonexistent")
        assert result is False

    def test_delete_where(self, repo):
        count = repo.delete_where("value < ?", (150,))
        assert count == 1
        assert repo.count() == 2

    def test_delete_where_multiple(self, repo):
        count = repo.delete_where("value >= ?", (200,))
        assert count == 2
        assert repo.count() == 1


class TestDatabaseRepositoryCallbacks:
    """Tests for change notification callbacks."""

    def test_on_change_callback_called(self, db_path):
        repo = ConcreteRepository(db_path)
        callback = MagicMock()
        repo.on_change(callback)

        repo.insert("test-id", "Test Item")

        callback.assert_called_once_with("insert")

    def test_multiple_callbacks(self, db_path):
        repo = ConcreteRepository(db_path)
        callback1 = MagicMock()
        callback2 = MagicMock()
        repo.on_change(callback1)
        repo.on_change(callback2)

        repo.insert("test-id", "Test Item")

        callback1.assert_called_once()
        callback2.assert_called_once()

    def test_callback_on_delete(self, repo):
        callback = MagicMock()
        repo.on_change(callback)

        repo.delete_by_id("item-1")

        callback.assert_called_with("delete")

    def test_callback_error_handled(self, db_path):
        repo = ConcreteRepository(db_path)

        def bad_callback(op):
            raise RuntimeError("Callback failed")

        repo.on_change(bad_callback)

        # Should not raise, just log warning
        repo.insert("test-id", "Test Item")
        assert repo.exists("test-id")
