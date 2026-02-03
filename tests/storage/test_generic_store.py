"""
Tests for GenericStoreBackend and implementations.

Tests cover:
- GenericInMemoryStore CRUD operations
- GenericSQLiteStore CRUD operations
- Thread safety
- Helper methods (_filter_by, _query_by_column, _update_json_field)
- Factory function (create_store_factory)
- Error handling (missing primary key, invalid config)
"""

import asyncio
import json
import tempfile
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aragora.storage.generic_store import (
    GenericInMemoryStore,
    GenericSQLiteStore,
    GenericStoreBackend,
    create_store_factory,
)


# =============================================================================
# Test Subclasses
# =============================================================================


class TestInMemoryStore(GenericInMemoryStore):
    """Concrete in-memory store for testing."""

    PRIMARY_KEY = "item_id"


class TestSQLiteStore(GenericSQLiteStore):
    """Concrete SQLite store for testing."""

    TABLE_NAME = "test_items"
    PRIMARY_KEY = "item_id"
    SCHEMA_SQL = """
        CREATE TABLE IF NOT EXISTS test_items (
            item_id TEXT PRIMARY KEY,
            status TEXT,
            category TEXT,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            data_json TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_test_status ON test_items(status);
        CREATE INDEX IF NOT EXISTS idx_test_category ON test_items(category);
    """
    INDEX_COLUMNS = {"status", "category"}


# =============================================================================
# GenericInMemoryStore Tests
# =============================================================================


class TestGenericInMemoryStore:
    """Tests for GenericInMemoryStore."""

    @pytest.fixture
    def store(self):
        return TestInMemoryStore()

    @pytest.mark.asyncio
    async def test_save_and_get(self, store):
        """Should save and retrieve items."""
        await store.save({"item_id": "1", "name": "Test"})
        result = await store.get("1")
        assert result == {"item_id": "1", "name": "Test"}

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, store):
        """Should return None for missing items."""
        result = await store.get("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_save_requires_primary_key(self, store):
        """Should raise ValueError when primary key is missing."""
        with pytest.raises(ValueError, match="item_id is required"):
            await store.save({"name": "No ID"})

    @pytest.mark.asyncio
    async def test_save_upsert(self, store):
        """Should update existing items."""
        await store.save({"item_id": "1", "name": "Original"})
        await store.save({"item_id": "1", "name": "Updated"})
        result = await store.get("1")
        assert result["name"] == "Updated"

    @pytest.mark.asyncio
    async def test_delete_existing(self, store):
        """Should delete existing items and return True."""
        await store.save({"item_id": "1", "name": "Test"})
        result = await store.delete("1")
        assert result is True
        assert await store.get("1") is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, store):
        """Should return False for missing items."""
        result = await store.delete("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_list_all(self, store):
        """Should list all items."""
        await store.save({"item_id": "1", "name": "A"})
        await store.save({"item_id": "2", "name": "B"})
        result = await store.list_all()
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_list_all_empty(self, store):
        """Should return empty list when no items."""
        result = await store.list_all()
        assert result == []

    @pytest.mark.asyncio
    async def test_exists(self, store):
        """Should check item existence."""
        await store.save({"item_id": "1", "name": "Test"})
        assert await store.exists("1") is True
        assert await store.exists("nonexistent") is False

    @pytest.mark.asyncio
    async def test_count(self, store):
        """Should count items."""
        assert await store.count() == 0
        await store.save({"item_id": "1", "name": "A"})
        await store.save({"item_id": "2", "name": "B"})
        assert await store.count() == 2

    @pytest.mark.asyncio
    async def test_close_noop(self, store):
        """Close should be a no-op."""
        await store.close()  # Should not raise

    def test_filter_by(self, store):
        """Should filter items by field value."""
        asyncio.get_event_loop().run_until_complete(
            store.save({"item_id": "1", "status": "active"})
        )
        asyncio.get_event_loop().run_until_complete(
            store.save({"item_id": "2", "status": "inactive"})
        )
        asyncio.get_event_loop().run_until_complete(
            store.save({"item_id": "3", "status": "active"})
        )
        result = store._filter_by("status", "active")
        assert len(result) == 2


# =============================================================================
# GenericSQLiteStore Tests
# =============================================================================


class TestGenericSQLiteStore:
    """Tests for GenericSQLiteStore."""

    @pytest.fixture
    def store(self, tmp_path):
        return TestSQLiteStore(db_path=tmp_path / "test.db")

    @pytest.mark.asyncio
    async def test_save_and_get(self, store):
        """Should save and retrieve items."""
        await store.save({"item_id": "1", "name": "Test", "status": "active"})
        result = await store.get("1")
        assert result is not None
        assert result["item_id"] == "1"
        assert result["name"] == "Test"
        assert result["status"] == "active"

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, store):
        """Should return None for missing items."""
        result = await store.get("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_save_requires_primary_key(self, store):
        """Should raise ValueError when primary key is missing."""
        with pytest.raises(ValueError, match="item_id is required"):
            await store.save({"name": "No ID"})

    @pytest.mark.asyncio
    async def test_save_upsert(self, store):
        """Should update existing items via INSERT OR REPLACE."""
        await store.save({"item_id": "1", "name": "Original", "status": "active"})
        await store.save({"item_id": "1", "name": "Updated", "status": "active"})
        result = await store.get("1")
        assert result["name"] == "Updated"

    @pytest.mark.asyncio
    async def test_delete_existing(self, store):
        """Should delete existing items and return True."""
        await store.save({"item_id": "1", "name": "Test", "status": "active"})
        result = await store.delete("1")
        assert result is True
        assert await store.get("1") is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, store):
        """Should return False for missing items."""
        result = await store.delete("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_list_all(self, store):
        """Should list all items ordered by created_at DESC."""
        await store.save({"item_id": "1", "name": "A", "status": "active"})
        await store.save({"item_id": "2", "name": "B", "status": "inactive"})
        result = await store.list_all()
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_list_all_empty(self, store):
        """Should return empty list when no items."""
        result = await store.list_all()
        assert result == []

    @pytest.mark.asyncio
    async def test_exists(self, store):
        """Should check item existence."""
        await store.save({"item_id": "1", "name": "Test", "status": "active"})
        assert await store.exists("1") is True
        assert await store.exists("nonexistent") is False

    @pytest.mark.asyncio
    async def test_count(self, store):
        """Should count items."""
        assert await store.count() == 0
        await store.save({"item_id": "1", "name": "A", "status": "active"})
        await store.save({"item_id": "2", "name": "B", "status": "active"})
        assert await store.count() == 2

    @pytest.mark.asyncio
    async def test_query_by_column(self, store):
        """Should query items by indexed column."""
        await store.save({"item_id": "1", "status": "active", "category": "A"})
        await store.save({"item_id": "2", "status": "inactive", "category": "A"})
        await store.save({"item_id": "3", "status": "active", "category": "B"})

        active = store._query_by_column("status", "active")
        assert len(active) == 2

        cat_a = store._query_by_column("category", "A")
        assert len(cat_a) == 2

    @pytest.mark.asyncio
    async def test_query_with_sql(self, store):
        """Should query items with custom WHERE clause."""
        await store.save({"item_id": "1", "status": "active", "category": "A"})
        await store.save({"item_id": "2", "status": "inactive", "category": "A"})
        await store.save({"item_id": "3", "status": "active", "category": "B"})

        result = store._query_with_sql("status = ? AND category = ?", ("active", "A"))
        assert len(result) == 1
        assert result[0]["item_id"] == "1"

    @pytest.mark.asyncio
    async def test_update_json_field(self, store):
        """Should update specific fields in data_json."""
        await store.save({"item_id": "1", "name": "Original", "status": "pending"})

        result = store._update_json_field(
            "1",
            {"name": "Updated", "status": "approved"},
            extra_column_updates={"status": "approved"},
        )
        assert result is True

        item = await store.get("1")
        assert item["name"] == "Updated"
        assert item["status"] == "approved"

    @pytest.mark.asyncio
    async def test_update_json_field_nonexistent(self, store):
        """Should return False for nonexistent item."""
        result = store._update_json_field("nonexistent", {"name": "Updated"})
        assert result is False

    @pytest.mark.asyncio
    async def test_close_noop(self, store):
        """Close should be a no-op for SQLite."""
        await store.close()  # Should not raise


# =============================================================================
# Configuration Validation Tests
# =============================================================================


class TestGenericStoreValidation:
    """Tests for store configuration validation."""

    def test_sqlite_requires_table_name(self, tmp_path):
        """Should raise ValueError when TABLE_NAME is empty."""

        class BadStore(GenericSQLiteStore):
            TABLE_NAME = ""
            SCHEMA_SQL = "CREATE TABLE foo (id TEXT PRIMARY KEY);"

        with pytest.raises(ValueError, match="TABLE_NAME must be set"):
            BadStore(db_path=tmp_path / "test.db")

    def test_sqlite_requires_schema_sql(self, tmp_path):
        """Should raise ValueError when SCHEMA_SQL is empty."""

        class BadStore(GenericSQLiteStore):
            TABLE_NAME = "test"
            SCHEMA_SQL = ""

        with pytest.raises(ValueError, match="SCHEMA_SQL must be set"):
            BadStore(db_path=tmp_path / "test.db")


# =============================================================================
# Thread Safety Tests
# =============================================================================


class TestGenericStoreThreadSafety:
    """Tests for thread safety."""

    @pytest.mark.asyncio
    async def test_concurrent_in_memory_writes(self):
        """Should handle concurrent writes safely."""
        store = TestInMemoryStore()
        errors = []

        def writer(start, count):
            loop = asyncio.new_event_loop()
            try:
                for i in range(start, start + count):
                    loop.run_until_complete(store.save({"item_id": str(i), "value": i}))
            except Exception as e:
                errors.append(e)
            finally:
                loop.close()

        threads = [threading.Thread(target=writer, args=(i * 100, 100)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert await store.count() == 500

    @pytest.mark.asyncio
    async def test_concurrent_sqlite_writes(self, tmp_path):
        """Should handle concurrent SQLite writes safely."""
        store = TestSQLiteStore(db_path=tmp_path / "concurrent.db")
        errors = []

        def writer(start, count):
            loop = asyncio.new_event_loop()
            try:
                for i in range(start, start + count):
                    loop.run_until_complete(
                        store.save({"item_id": str(i), "status": "active", "value": i})
                    )
            except Exception as e:
                errors.append(e)
            finally:
                loop.close()

        threads = [threading.Thread(target=writer, args=(i * 50, 50)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert await store.count() == 200


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestCreateStoreFactory:
    """Tests for create_store_factory."""

    def test_factory_creates_sqlite_by_default(self, tmp_path):
        """Should create SQLite store by default."""
        get_store, set_store = create_store_factory(
            store_name="test",
            sqlite_class=TestSQLiteStore,
            default_db_name=str(tmp_path / "factory_test.db"),
        )

        with patch.dict("os.environ", {}, clear=False):
            store = get_store()
            assert isinstance(store, TestSQLiteStore)

    def test_factory_set_store(self):
        """Should allow injecting custom store."""
        get_store, set_store = create_store_factory(
            store_name="test",
            sqlite_class=TestSQLiteStore,
        )

        custom_store = TestInMemoryStore()
        set_store(custom_store)
        assert get_store() is custom_store

    def test_factory_returns_singleton(self, tmp_path):
        """Should return same instance on repeated calls."""
        get_store, set_store = create_store_factory(
            store_name="test",
            sqlite_class=TestSQLiteStore,
            default_db_name=str(tmp_path / "singleton_test.db"),
        )

        store1 = get_store()
        store2 = get_store()
        assert store1 is store2


# =============================================================================
# Abstract Base Tests
# =============================================================================


class TestGenericStoreBackendInterface:
    """Tests for abstract base interface."""

    def test_cannot_instantiate_abstract(self):
        """Should not allow direct instantiation."""
        with pytest.raises(TypeError):
            GenericStoreBackend()

    @pytest.mark.asyncio
    async def test_default_exists_uses_get(self):
        """Default exists() should delegate to get()."""
        store = TestInMemoryStore()
        await store.save({"item_id": "1", "name": "Test"})
        assert await store.exists("1") is True
        assert await store.exists("2") is False

    @pytest.mark.asyncio
    async def test_default_count_uses_list_all(self):
        """Default count() should delegate to list_all()."""
        store = TestInMemoryStore()
        await store.save({"item_id": "1", "name": "A"})
        await store.save({"item_id": "2", "name": "B"})
        assert await store.count() == 2
