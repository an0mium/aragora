"""Tests for GauntletRunStore backends.

Comprehensive tests covering:
- GauntletRunItem dataclass (serialization, defaults)
- InMemoryGauntletRunStore (CRUD, status transitions)
- SQLiteGauntletRunStore (persistence, pagination, analytics)
- RedisGauntletRunStore (with mocked Redis, fallback behavior)
- PostgresGauntletRunStore (with mocked asyncpg pool)
- Global store accessor functions
- Error handling paths
- Edge cases and boundary conditions
"""

import asyncio
import json
import pytest
import tempfile
import time
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

from aragora.storage.gauntlet_run_store import (
    GauntletRunItem,
    GauntletRunStoreBackend,
    InMemoryGauntletRunStore,
    SQLiteGauntletRunStore,
    RedisGauntletRunStore,
    PostgresGauntletRunStore,
    get_gauntlet_run_store,
    set_gauntlet_run_store,
    reset_gauntlet_run_store,
    _batch_deserialize_json,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_run_data():
    """Sample gauntlet run data for testing."""
    return {
        "run_id": "gauntlet-test-001",
        "template_id": "security-audit",
        "status": "pending",
        "config_data": {"persona": "sec_analyst", "max_duration": 300},
        "triggered_by": "user-123",
        "workspace_id": "ws-456",
        "tags": ["security", "audit"],
    }


@pytest.fixture
def sample_run_data_2():
    """Second sample for listing tests."""
    return {
        "run_id": "gauntlet-test-002",
        "template_id": "compliance-check",
        "status": "running",
        "config_data": {"persona": "compliance_officer"},
        "triggered_by": "user-789",
        "workspace_id": "ws-456",
        "tags": ["compliance"],
    }


@pytest.fixture
def sample_run_data_3():
    """Third sample for multi-run tests."""
    return {
        "run_id": "gauntlet-test-003",
        "template_id": "security-audit",
        "status": "completed",
        "config_data": {"persona": "sec_analyst"},
        "result_data": {"verdict": "pass", "score": 95},
        "triggered_by": "user-123",
        "workspace_id": "ws-789",
        "tags": ["security"],
    }


# =============================================================================
# Helper Function Tests
# =============================================================================


class TestBatchDeserializeJson:
    """Tests for _batch_deserialize_json helper function."""

    def test_empty_rows(self):
        """Test deserialization of empty list."""
        result = _batch_deserialize_json([])
        assert result == []

    def test_json_string_rows(self):
        """Test deserialization of JSON string rows."""
        rows = [
            ('{"key": "value1"}',),
            ('{"key": "value2"}',),
        ]
        result = _batch_deserialize_json(rows)
        assert len(result) == 2
        assert result[0]["key"] == "value1"
        assert result[1]["key"] == "value2"

    def test_dict_rows_already_parsed(self):
        """Test handling of already-parsed dict rows (asyncpg JSONB)."""
        rows = [
            ({"key": "value1"},),
            ({"key": "value2"},),
        ]
        result = _batch_deserialize_json(rows)
        assert len(result) == 2
        assert result[0]["key"] == "value1"
        assert result[1]["key"] == "value2"

    def test_mixed_rows(self):
        """Test mixed string and dict rows."""
        rows = [
            ('{"key": "string"}',),
            ({"key": "dict"},),
        ]
        result = _batch_deserialize_json(rows)
        assert result[0]["key"] == "string"
        assert result[1]["key"] == "dict"

    def test_custom_index(self):
        """Test deserialization with custom column index."""
        rows = [
            ("ignored", '{"key": "value"}'),
            ("ignored", '{"key": "other"}'),
        ]
        result = _batch_deserialize_json(rows, idx=1)
        assert len(result) == 2
        assert result[0]["key"] == "value"


# =============================================================================
# GauntletRunItem Tests
# =============================================================================


class TestGauntletRunItem:
    """Tests for GauntletRunItem dataclass."""

    def test_default_timestamps(self):
        """Test that timestamps are set by default."""
        item = GauntletRunItem(run_id="test-1", template_id="test-template")
        assert item.created_at != ""
        assert item.updated_at != ""

    def test_default_values(self):
        """Test default values for optional fields."""
        item = GauntletRunItem(run_id="test-1", template_id="test-template")
        assert item.status == "pending"
        assert item.config_data == {}
        assert item.result_data is None
        assert item.started_at is None
        assert item.completed_at is None
        assert item.triggered_by is None
        assert item.workspace_id is None
        assert item.tags == []

    def test_to_dict(self):
        """Test conversion to dictionary."""
        item = GauntletRunItem(
            run_id="test-1",
            template_id="test-template",
            status="running",
            config_data={"key": "value"},
        )
        d = item.to_dict()
        assert d["run_id"] == "test-1"
        assert d["template_id"] == "test-template"
        assert d["status"] == "running"
        assert d["config_data"] == {"key": "value"}

    def test_to_dict_completeness(self):
        """Test that to_dict includes all fields."""
        item = GauntletRunItem(
            run_id="test-1",
            template_id="test-template",
            status="completed",
            config_data={"key": "value"},
            result_data={"score": 100},
            started_at="2024-01-01T00:00:00Z",
            completed_at="2024-01-01T01:00:00Z",
            triggered_by="user-1",
            workspace_id="ws-1",
            tags=["tag1", "tag2"],
        )
        d = item.to_dict()
        expected_keys = {
            "run_id",
            "template_id",
            "status",
            "config_data",
            "result_data",
            "started_at",
            "completed_at",
            "triggered_by",
            "workspace_id",
            "tags",
            "created_at",
            "updated_at",
        }
        assert set(d.keys()) == expected_keys

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "run_id": "test-1",
            "template_id": "test-template",
            "status": "completed",
            "config_data": {"key": "value"},
            "result_data": {"verdict": "pass"},
        }
        item = GauntletRunItem.from_dict(data)
        assert item.run_id == "test-1"
        assert item.template_id == "test-template"
        assert item.status == "completed"
        assert item.result_data == {"verdict": "pass"}

    def test_from_dict_missing_fields(self):
        """Test from_dict handles missing optional fields."""
        data = {"run_id": "test-1", "template_id": "test-template"}
        item = GauntletRunItem.from_dict(data)
        assert item.run_id == "test-1"
        assert item.status == "pending"
        assert item.config_data == {}
        assert item.tags == []

    def test_from_dict_empty_run_id(self):
        """Test from_dict with empty run_id."""
        data = {"template_id": "test-template"}
        item = GauntletRunItem.from_dict(data)
        assert item.run_id == ""

    def test_json_roundtrip(self):
        """Test JSON serialization roundtrip."""
        item = GauntletRunItem(
            run_id="test-1",
            template_id="test-template",
            status="pending",
            config_data={"nested": {"data": 123}},
        )
        json_str = item.to_json()
        restored = GauntletRunItem.from_json(json_str)
        assert restored.run_id == item.run_id
        assert restored.config_data == item.config_data

    def test_json_with_special_characters(self):
        """Test JSON handling of special characters."""
        item = GauntletRunItem(
            run_id="test-1",
            template_id="test-template",
            config_data={"message": 'Hello "World"\nNew line'},
        )
        json_str = item.to_json()
        restored = GauntletRunItem.from_json(json_str)
        assert restored.config_data["message"] == 'Hello "World"\nNew line'


# =============================================================================
# InMemoryGauntletRunStore Tests
# =============================================================================


class TestInMemoryGauntletRunStore:
    """Tests for InMemoryGauntletRunStore."""

    @pytest.fixture
    def store(self):
        """Create a fresh in-memory store."""
        return InMemoryGauntletRunStore()

    @pytest.mark.asyncio
    async def test_save_and_get(self, store, sample_run_data):
        """Test saving and retrieving a run."""
        await store.save(sample_run_data)
        result = await store.get(sample_run_data["run_id"])
        assert result is not None
        assert result["run_id"] == sample_run_data["run_id"]
        assert result["template_id"] == sample_run_data["template_id"]

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, store):
        """Test getting a non-existent run."""
        result = await store.get("nonexistent-id")
        assert result is None

    @pytest.mark.asyncio
    async def test_save_requires_run_id(self, store):
        """Test that save requires run_id."""
        with pytest.raises(ValueError, match="run_id is required"):
            await store.save({"template_id": "test"})

    @pytest.mark.asyncio
    async def test_save_overwrites_existing(self, store, sample_run_data):
        """Test that saving with same run_id overwrites."""
        await store.save(sample_run_data)
        updated_data = {**sample_run_data, "status": "running"}
        await store.save(updated_data)
        result = await store.get(sample_run_data["run_id"])
        assert result["status"] == "running"

    @pytest.mark.asyncio
    async def test_delete(self, store, sample_run_data):
        """Test deleting a run."""
        await store.save(sample_run_data)
        deleted = await store.delete(sample_run_data["run_id"])
        assert deleted is True
        result = await store.get(sample_run_data["run_id"])
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, store):
        """Test deleting a non-existent run."""
        deleted = await store.delete("nonexistent-id")
        assert deleted is False

    @pytest.mark.asyncio
    async def test_list_all(self, store, sample_run_data, sample_run_data_2):
        """Test listing all runs."""
        await store.save(sample_run_data)
        await store.save(sample_run_data_2)
        all_runs = await store.list_all()
        assert len(all_runs) == 2

    @pytest.mark.asyncio
    async def test_list_all_empty(self, store):
        """Test listing all runs when empty."""
        all_runs = await store.list_all()
        assert all_runs == []

    @pytest.mark.asyncio
    async def test_list_by_status(self, store, sample_run_data, sample_run_data_2):
        """Test listing runs by status."""
        await store.save(sample_run_data)  # pending
        await store.save(sample_run_data_2)  # running
        pending = await store.list_by_status("pending")
        assert len(pending) == 1
        assert pending[0]["run_id"] == sample_run_data["run_id"]

    @pytest.mark.asyncio
    async def test_list_by_status_no_matches(self, store, sample_run_data):
        """Test listing by status with no matches."""
        await store.save(sample_run_data)  # pending
        failed = await store.list_by_status("failed")
        assert failed == []

    @pytest.mark.asyncio
    async def test_list_by_template(self, store, sample_run_data, sample_run_data_2):
        """Test listing runs by template."""
        await store.save(sample_run_data)
        await store.save(sample_run_data_2)
        security = await store.list_by_template("security-audit")
        assert len(security) == 1
        assert security[0]["run_id"] == sample_run_data["run_id"]

    @pytest.mark.asyncio
    async def test_list_by_template_multiple(self, store, sample_run_data, sample_run_data_3):
        """Test listing multiple runs by same template."""
        await store.save(sample_run_data)  # security-audit
        await store.save(sample_run_data_3)  # security-audit
        security = await store.list_by_template("security-audit")
        assert len(security) == 2

    @pytest.mark.asyncio
    async def test_list_active(self, store, sample_run_data, sample_run_data_2, sample_run_data_3):
        """Test listing active runs."""
        await store.save(sample_run_data)  # pending
        await store.save(sample_run_data_2)  # running
        await store.save(sample_run_data_3)  # completed
        active = await store.list_active()
        assert len(active) == 2  # pending + running

    @pytest.mark.asyncio
    async def test_list_active_empty(self, store, sample_run_data_3):
        """Test list_active when no active runs."""
        await store.save(sample_run_data_3)  # completed
        active = await store.list_active()
        assert active == []

    @pytest.mark.asyncio
    async def test_update_status(self, store, sample_run_data):
        """Test updating run status."""
        await store.save(sample_run_data)
        updated = await store.update_status(
            sample_run_data["run_id"],
            "running",
        )
        assert updated is True
        result = await store.get(sample_run_data["run_id"])
        assert result["status"] == "running"
        assert result["started_at"] is not None

    @pytest.mark.asyncio
    async def test_update_status_with_result(self, store, sample_run_data):
        """Test updating status with result data."""
        await store.save(sample_run_data)
        result_data = {"verdict": "pass", "score": 95}
        updated = await store.update_status(
            sample_run_data["run_id"],
            "completed",
            result_data=result_data,
        )
        assert updated is True
        result = await store.get(sample_run_data["run_id"])
        assert result["status"] == "completed"
        assert result["result_data"] == result_data
        assert result["completed_at"] is not None

    @pytest.mark.asyncio
    async def test_update_status_nonexistent(self, store):
        """Test updating non-existent run."""
        updated = await store.update_status("nonexistent-id", "running")
        assert updated is False

    @pytest.mark.asyncio
    async def test_update_status_failed_sets_completed_at(self, store, sample_run_data):
        """Test that failed status sets completed_at."""
        await store.save(sample_run_data)
        await store.update_status(sample_run_data["run_id"], "failed")
        result = await store.get(sample_run_data["run_id"])
        assert result["completed_at"] is not None

    @pytest.mark.asyncio
    async def test_update_status_cancelled_sets_completed_at(self, store, sample_run_data):
        """Test that cancelled status sets completed_at."""
        await store.save(sample_run_data)
        await store.update_status(sample_run_data["run_id"], "cancelled")
        result = await store.get(sample_run_data["run_id"])
        assert result["completed_at"] is not None

    @pytest.mark.asyncio
    async def test_update_status_running_twice_keeps_started_at(self, store, sample_run_data):
        """Test that started_at isn't overwritten on second running update."""
        await store.save(sample_run_data)
        await store.update_status(sample_run_data["run_id"], "running")
        result1 = await store.get(sample_run_data["run_id"])
        original_started_at = result1["started_at"]

        # Update to running again
        await store.update_status(sample_run_data["run_id"], "running")
        result2 = await store.get(sample_run_data["run_id"])
        assert result2["started_at"] == original_started_at

    @pytest.mark.asyncio
    async def test_close_is_noop(self, store):
        """Test that close is a no-op for in-memory store."""
        await store.close()  # Should not raise


# =============================================================================
# SQLiteGauntletRunStore Tests
# =============================================================================


class TestSQLiteGauntletRunStore:
    """Tests for SQLiteGauntletRunStore."""

    @pytest.fixture
    def store(self, tmp_path):
        """Create a SQLite store with temp database."""
        db_path = tmp_path / "test_gauntlet_runs.db"
        return SQLiteGauntletRunStore(db_path=db_path)

    @pytest.mark.asyncio
    async def test_save_and_get(self, store, sample_run_data):
        """Test saving and retrieving a run."""
        await store.save(sample_run_data)
        result = await store.get(sample_run_data["run_id"])
        assert result is not None
        assert result["run_id"] == sample_run_data["run_id"]

    @pytest.mark.asyncio
    async def test_save_requires_run_id(self, store):
        """Test that save requires run_id."""
        with pytest.raises(ValueError, match="run_id is required"):
            await store.save({"template_id": "test"})

    @pytest.mark.asyncio
    async def test_persistence(self, tmp_path, sample_run_data):
        """Test that data persists across store instances."""
        db_path = tmp_path / "persistence_test.db"

        # Save with first instance
        store1 = SQLiteGauntletRunStore(db_path=db_path)
        await store1.save(sample_run_data)

        # Retrieve with second instance
        store2 = SQLiteGauntletRunStore(db_path=db_path)
        result = await store2.get(sample_run_data["run_id"])
        assert result is not None
        assert result["run_id"] == sample_run_data["run_id"]

    @pytest.mark.asyncio
    async def test_list_all_ordered(self, store, sample_run_data, sample_run_data_2):
        """Test that list_all returns results ordered by created_at DESC."""
        await store.save(sample_run_data)
        await store.save(sample_run_data_2)
        all_runs = await store.list_all()
        assert len(all_runs) == 2
        # Second one was saved later, should be first
        assert all_runs[0]["run_id"] == sample_run_data_2["run_id"]

    @pytest.mark.asyncio
    async def test_list_all_pagination(self, store):
        """Test list_all pagination with limit and offset."""
        # Create 5 runs
        for i in range(5):
            await store.save(
                {
                    "run_id": f"run-{i}",
                    "template_id": "template-1",
                    "status": "pending",
                }
            )

        # Get first 2
        page1 = await store.list_all(limit=2, offset=0)
        assert len(page1) == 2

        # Get next 2
        page2 = await store.list_all(limit=2, offset=2)
        assert len(page2) == 2

        # Verify no overlap
        page1_ids = {r["run_id"] for r in page1}
        page2_ids = {r["run_id"] for r in page2}
        assert page1_ids.isdisjoint(page2_ids)

    @pytest.mark.asyncio
    async def test_list_by_status_pagination(self, store):
        """Test list_by_status pagination."""
        # Create 5 pending runs
        for i in range(5):
            await store.save(
                {
                    "run_id": f"pending-{i}",
                    "template_id": "template-1",
                    "status": "pending",
                }
            )

        page1 = await store.list_by_status("pending", limit=3, offset=0)
        assert len(page1) == 3

        page2 = await store.list_by_status("pending", limit=3, offset=3)
        assert len(page2) == 2  # Only 2 remaining

    @pytest.mark.asyncio
    async def test_list_by_template_pagination(self, store):
        """Test list_by_template pagination."""
        # Create 4 runs with same template
        for i in range(4):
            await store.save(
                {
                    "run_id": f"run-{i}",
                    "template_id": "shared-template",
                    "status": "pending",
                }
            )

        page1 = await store.list_by_template("shared-template", limit=2, offset=0)
        assert len(page1) == 2

        page2 = await store.list_by_template("shared-template", limit=2, offset=2)
        assert len(page2) == 2

    @pytest.mark.asyncio
    async def test_list_active_limit(self, store):
        """Test list_active respects limit."""
        # Create 10 pending runs
        for i in range(10):
            await store.save(
                {
                    "run_id": f"pending-{i}",
                    "template_id": "template-1",
                    "status": "pending",
                }
            )

        active = await store.list_active(limit=5)
        assert len(active) == 5

    @pytest.mark.asyncio
    async def test_update_status_full_cycle(self, store, sample_run_data):
        """Test full status update cycle."""
        await store.save(sample_run_data)

        # Start running
        await store.update_status(sample_run_data["run_id"], "running")
        result = await store.get(sample_run_data["run_id"])
        assert result["status"] == "running"
        assert result["started_at"] is not None

        # Complete
        await store.update_status(
            sample_run_data["run_id"],
            "completed",
            result_data={"verdict": "pass"},
        )
        result = await store.get(sample_run_data["run_id"])
        assert result["status"] == "completed"
        assert result["completed_at"] is not None
        assert result["result_data"]["verdict"] == "pass"

    @pytest.mark.asyncio
    async def test_delete(self, store, sample_run_data):
        """Test deleting a run."""
        await store.save(sample_run_data)
        deleted = await store.delete(sample_run_data["run_id"])
        assert deleted is True
        result = await store.get(sample_run_data["run_id"])
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, store):
        """Test deleting non-existent run."""
        deleted = await store.delete("nonexistent-id")
        assert deleted is False

    @pytest.mark.asyncio
    async def test_close_is_noop(self, store):
        """Test that close is a no-op for SQLite store."""
        await store.close()  # Should not raise

    @pytest.mark.asyncio
    async def test_default_db_path(self, monkeypatch, tmp_path):
        """Test default database path from environment."""
        monkeypatch.setenv("ARAGORA_DATA_DIR", str(tmp_path))
        store = SQLiteGauntletRunStore()
        assert store._db_path == tmp_path / "gauntlet_runs.db"

    @pytest.mark.asyncio
    async def test_creates_parent_directories(self, tmp_path):
        """Test that store creates parent directories if needed."""
        db_path = tmp_path / "nested" / "dir" / "gauntlet_runs.db"
        store = SQLiteGauntletRunStore(db_path=db_path)
        await store.save({"run_id": "test", "template_id": "t"})
        assert db_path.exists()


# =============================================================================
# Queue Analytics Tests
# =============================================================================


class TestQueueAnalytics:
    """Tests for queue analytics using window functions."""

    @pytest.fixture
    def store(self, tmp_path):
        """Create a SQLite store with temp database."""
        db_path = tmp_path / "test_analytics.db"
        return SQLiteGauntletRunStore(db_path=db_path)

    @pytest.mark.asyncio
    async def test_empty_queue_analytics(self, store):
        """Test analytics on empty queue."""
        analytics = await store.get_queue_analytics()
        assert analytics["total_active"] == 0
        assert analytics["pending_count"] == 0
        assert analytics["running_count"] == 0
        assert analytics["queue"] == []

    @pytest.mark.asyncio
    async def test_pending_runs_in_analytics(self, store):
        """Test that pending runs are included in analytics."""
        await store.save(
            {
                "run_id": "run-1",
                "template_id": "template-1",
                "status": "pending",
            }
        )
        await store.save(
            {
                "run_id": "run-2",
                "template_id": "template-2",
                "status": "pending",
            }
        )

        analytics = await store.get_queue_analytics()
        assert analytics["total_active"] == 2
        assert analytics["pending_count"] == 2
        assert analytics["running_count"] == 0
        assert len(analytics["queue"]) == 2

    @pytest.mark.asyncio
    async def test_running_runs_in_analytics(self, store):
        """Test that running runs are included in analytics."""
        await store.save(
            {
                "run_id": "run-1",
                "template_id": "template-1",
                "status": "running",
            }
        )

        analytics = await store.get_queue_analytics()
        assert analytics["total_active"] == 1
        assert analytics["pending_count"] == 0
        assert analytics["running_count"] == 1
        assert len(analytics["queue"]) == 1

    @pytest.mark.asyncio
    async def test_completed_runs_not_in_analytics(self, store):
        """Test that completed runs are not in queue analytics."""
        await store.save(
            {
                "run_id": "run-1",
                "template_id": "template-1",
                "status": "completed",
            }
        )
        await store.save(
            {
                "run_id": "run-2",
                "template_id": "template-2",
                "status": "failed",
            }
        )

        analytics = await store.get_queue_analytics()
        assert analytics["total_active"] == 0
        assert analytics["queue"] == []

    @pytest.mark.asyncio
    async def test_position_in_status(self, store):
        """Test that position_in_status is calculated correctly."""
        # Create 3 pending runs
        for i in range(1, 4):
            await store.save(
                {
                    "run_id": f"run-{i}",
                    "template_id": "template-1",
                    "status": "pending",
                }
            )

        analytics = await store.get_queue_analytics()
        positions = [item["position_in_status"] for item in analytics["queue"]]
        assert sorted(positions) == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_mixed_status_analytics(self, store):
        """Test analytics with mixed pending and running."""
        await store.save(
            {
                "run_id": "run-pending-1",
                "template_id": "template-1",
                "status": "pending",
            }
        )
        await store.save(
            {
                "run_id": "run-pending-2",
                "template_id": "template-2",
                "status": "pending",
            }
        )
        await store.save(
            {
                "run_id": "run-running-1",
                "template_id": "template-3",
                "status": "running",
            }
        )
        await store.save(
            {
                "run_id": "run-completed",
                "template_id": "template-4",
                "status": "completed",
            }
        )

        analytics = await store.get_queue_analytics()
        assert analytics["total_active"] == 3
        assert analytics["pending_count"] == 2
        assert analytics["running_count"] == 1
        assert len(analytics["queue"]) == 3

    @pytest.mark.asyncio
    async def test_queue_item_fields(self, store):
        """Test that queue items have required fields."""
        await store.save(
            {
                "run_id": "run-1",
                "template_id": "template-1",
                "status": "pending",
            }
        )

        analytics = await store.get_queue_analytics()
        item = analytics["queue"][0]
        assert "run_id" in item
        assert "template_id" in item
        assert "status" in item
        assert "created_at" in item
        assert "position_in_status" in item


# =============================================================================
# RedisGauntletRunStore Tests
# =============================================================================


class TestRedisGauntletRunStore:
    """Tests for RedisGauntletRunStore with mocked Redis."""

    @pytest.fixture
    def mock_redis(self):
        """Create a mock Redis client."""
        mock = MagicMock()
        mock.ping.return_value = True
        mock.get.return_value = None
        mock.set.return_value = True
        mock.delete.return_value = True
        mock.sadd.return_value = True
        mock.srem.return_value = True
        mock.smembers.return_value = set()
        mock.scan.return_value = (0, [])
        mock.mget.return_value = []
        mock.pipeline.return_value = MagicMock()
        mock.pipeline.return_value.execute.return_value = []
        return mock

    @pytest.fixture
    def store_with_redis(self, mock_redis, tmp_path):
        """Create store with mocked Redis connection."""
        with patch("redis.from_url", return_value=mock_redis):
            store = RedisGauntletRunStore(
                fallback_db_path=tmp_path / "fallback.db", redis_url="redis://localhost:6379"
            )
            store._redis_client = mock_redis
            store._using_fallback = False
            return store

    @pytest.fixture
    def store_fallback(self, tmp_path):
        """Create store in fallback mode (no Redis)."""
        store = RedisGauntletRunStore(fallback_db_path=tmp_path / "fallback.db", redis_url="")
        return store

    @pytest.mark.asyncio
    async def test_fallback_when_no_redis_url(self, store_fallback):
        """Test that store falls back to SQLite when no Redis URL."""
        assert store_fallback._using_fallback is True

    @pytest.mark.asyncio
    async def test_save_to_redis_and_fallback(self, store_with_redis, mock_redis, sample_run_data):
        """Test that save writes to both Redis and fallback."""
        pipe_mock = MagicMock()
        mock_redis.pipeline.return_value = pipe_mock

        await store_with_redis.save(sample_run_data)

        # Verify Redis pipeline was used
        pipe_mock.set.assert_called()
        pipe_mock.sadd.assert_called()
        pipe_mock.execute.assert_called()

        # Verify fallback also has the data
        result = await store_with_redis._fallback.get(sample_run_data["run_id"])
        assert result is not None

    @pytest.mark.asyncio
    async def test_get_from_redis(self, store_with_redis, mock_redis, sample_run_data):
        """Test getting data from Redis."""
        mock_redis.get.return_value = json.dumps(sample_run_data)

        result = await store_with_redis.get(sample_run_data["run_id"])
        assert result is not None
        assert result["run_id"] == sample_run_data["run_id"]

    @pytest.mark.asyncio
    async def test_get_falls_back_on_redis_error(
        self, store_with_redis, mock_redis, sample_run_data
    ):
        """Test that get falls back to SQLite on Redis error."""
        mock_redis.get.side_effect = Exception("Redis connection error")

        # First save to fallback
        await store_with_redis._fallback.save(sample_run_data)

        # Should fall back to SQLite
        result = await store_with_redis.get(sample_run_data["run_id"])
        assert result is not None
        assert result["run_id"] == sample_run_data["run_id"]

    @pytest.mark.asyncio
    async def test_delete_cleans_indexes(self, store_with_redis, mock_redis, sample_run_data):
        """Test that delete removes from indexes."""
        mock_redis.get.return_value = json.dumps(sample_run_data)
        pipe_mock = MagicMock()
        mock_redis.pipeline.return_value = pipe_mock

        await store_with_redis.delete(sample_run_data["run_id"])

        # Verify index cleanup
        pipe_mock.srem.assert_called()
        pipe_mock.delete.assert_called()

    @pytest.mark.asyncio
    async def test_list_by_status_from_redis(self, store_with_redis, mock_redis, sample_run_data):
        """Test listing by status from Redis indexes."""
        mock_redis.smembers.return_value = {b"gauntlet-test-001"}
        mock_redis.mget.return_value = [json.dumps(sample_run_data)]

        result = await store_with_redis.list_by_status("pending")
        assert len(result) == 1
        assert result[0]["run_id"] == sample_run_data["run_id"]

    @pytest.mark.asyncio
    async def test_list_by_status_empty(self, store_with_redis, mock_redis):
        """Test listing by status with no results."""
        mock_redis.smembers.return_value = set()

        result = await store_with_redis.list_by_status("pending")
        assert result == []

    @pytest.mark.asyncio
    async def test_list_by_template_from_redis(self, store_with_redis, mock_redis, sample_run_data):
        """Test listing by template from Redis indexes."""
        mock_redis.smembers.return_value = {b"gauntlet-test-001"}
        mock_redis.mget.return_value = [json.dumps(sample_run_data)]

        result = await store_with_redis.list_by_template("security-audit")
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_list_active_combines_pending_and_running(self, store_with_redis, mock_redis):
        """Test that list_active combines pending and running."""
        pending_data = {"run_id": "pending-1", "template_id": "t1", "status": "pending"}
        running_data = {"run_id": "running-1", "template_id": "t2", "status": "running"}

        # First call for pending, second for running
        mock_redis.smembers.side_effect = [
            {b"pending-1"},  # pending status
            {b"running-1"},  # running status
        ]
        mock_redis.mget.side_effect = [
            [json.dumps(pending_data)],
            [json.dumps(running_data)],
        ]

        result = await store_with_redis.list_active()
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_update_status_updates_indexes(
        self, store_with_redis, mock_redis, sample_run_data
    ):
        """Test that update_status updates Redis indexes."""
        mock_redis.get.return_value = json.dumps(sample_run_data)
        pipe_mock = MagicMock()
        mock_redis.pipeline.return_value = pipe_mock

        await store_with_redis.update_status(sample_run_data["run_id"], "running")

        # Should remove from old index and add to new
        pipe_mock.srem.assert_called()
        pipe_mock.sadd.assert_called()

    @pytest.mark.asyncio
    async def test_close_closes_redis_and_fallback(self, store_with_redis, mock_redis):
        """Test that close closes both Redis and fallback."""
        await store_with_redis.close()
        mock_redis.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_handles_already_closed_redis(self, store_with_redis, mock_redis):
        """Test that close handles already closed Redis connection."""
        mock_redis.close.side_effect = ConnectionError("Already closed")
        await store_with_redis.close()  # Should not raise


# =============================================================================
# PostgresGauntletRunStore Tests
# =============================================================================


class TestPostgresGauntletRunStore:
    """Tests for PostgresGauntletRunStore with mocked asyncpg pool."""

    @pytest.fixture
    def mock_pool(self):
        """Create a mock asyncpg pool."""
        pool = AsyncMock()
        conn = AsyncMock()
        pool.acquire.return_value.__aenter__.return_value = conn
        pool.acquire.return_value.__aexit__.return_value = None
        return pool, conn

    @pytest.fixture
    def store(self, mock_pool):
        """Create store with mocked pool."""
        pool, _ = mock_pool
        return PostgresGauntletRunStore(pool)

    @pytest.mark.asyncio
    async def test_initialize_creates_schema(self, store, mock_pool):
        """Test that initialize creates database schema."""
        _, conn = mock_pool
        await store.initialize()
        conn.execute.assert_called_once()
        assert store._initialized is True

    @pytest.mark.asyncio
    async def test_initialize_only_once(self, store, mock_pool):
        """Test that initialize is idempotent."""
        _, conn = mock_pool
        await store.initialize()
        await store.initialize()
        # Should only be called once
        assert conn.execute.call_count == 1

    @pytest.mark.asyncio
    async def test_get_returns_data(self, store, mock_pool, sample_run_data):
        """Test getting data from Postgres."""
        _, conn = mock_pool
        conn.fetchrow.return_value = {"data_json": json.dumps(sample_run_data)}

        result = await store.get(sample_run_data["run_id"])
        assert result["run_id"] == sample_run_data["run_id"]

    @pytest.mark.asyncio
    async def test_get_returns_none_for_missing(self, store, mock_pool):
        """Test get returns None for missing run."""
        _, conn = mock_pool
        conn.fetchrow.return_value = None

        result = await store.get("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_handles_jsonb_dict(self, store, mock_pool, sample_run_data):
        """Test get handles JSONB returned as dict (asyncpg behavior)."""
        _, conn = mock_pool
        # asyncpg returns JSONB as dict, not string
        conn.fetchrow.return_value = {"data_json": sample_run_data}

        result = await store.get(sample_run_data["run_id"])
        assert result["run_id"] == sample_run_data["run_id"]

    @pytest.mark.asyncio
    async def test_save_inserts_data(self, store, mock_pool, sample_run_data):
        """Test saving data to Postgres."""
        _, conn = mock_pool
        await store.save(sample_run_data)
        conn.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_save_requires_run_id(self, store):
        """Test that save requires run_id."""
        with pytest.raises(ValueError, match="run_id is required"):
            await store.save({"template_id": "test"})

    @pytest.mark.asyncio
    async def test_delete_returns_true_on_success(self, store, mock_pool):
        """Test delete returns True when row deleted."""
        _, conn = mock_pool
        conn.execute.return_value = "DELETE 1"

        result = await store.delete("run-123")
        assert result is True

    @pytest.mark.asyncio
    async def test_delete_returns_false_when_not_found(self, store, mock_pool):
        """Test delete returns False when run not found."""
        _, conn = mock_pool
        conn.execute.return_value = "DELETE 0"

        result = await store.delete("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_list_all_returns_deserialized_data(self, store, mock_pool, sample_run_data):
        """Test list_all deserializes data correctly."""
        _, conn = mock_pool
        conn.fetch.return_value = [
            {"data_json": json.dumps(sample_run_data)},
        ]

        result = await store.list_all()
        assert len(result) == 1
        assert result[0]["run_id"] == sample_run_data["run_id"]

    @pytest.mark.asyncio
    async def test_list_by_status(self, store, mock_pool, sample_run_data):
        """Test list_by_status filters correctly."""
        _, conn = mock_pool
        conn.fetch.return_value = [{"data_json": sample_run_data}]

        result = await store.list_by_status("pending")
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_list_by_template(self, store, mock_pool, sample_run_data):
        """Test list_by_template filters correctly."""
        _, conn = mock_pool
        conn.fetch.return_value = [{"data_json": sample_run_data}]

        result = await store.list_by_template("security-audit")
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_list_active(self, store, mock_pool, sample_run_data):
        """Test list_active returns pending and running."""
        _, conn = mock_pool
        conn.fetch.return_value = [{"data_json": sample_run_data}]

        result = await store.list_active()
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_update_status_success(self, store, mock_pool, sample_run_data):
        """Test update_status updates data correctly."""
        _, conn = mock_pool
        conn.fetchrow.return_value = {"data_json": json.dumps(sample_run_data)}

        result = await store.update_status(sample_run_data["run_id"], "running")
        assert result is True
        conn.execute.assert_called()

    @pytest.mark.asyncio
    async def test_update_status_not_found(self, store, mock_pool):
        """Test update_status returns False when not found."""
        _, conn = mock_pool
        conn.fetchrow.return_value = None

        result = await store.update_status("nonexistent", "running")
        assert result is False

    @pytest.mark.asyncio
    async def test_update_status_with_result_data(self, store, mock_pool, sample_run_data):
        """Test update_status with result_data."""
        _, conn = mock_pool
        conn.fetchrow.return_value = {"data_json": sample_run_data}

        result_data = {"verdict": "pass"}
        result = await store.update_status(
            sample_run_data["run_id"], "completed", result_data=result_data
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_close_is_noop(self, store):
        """Test that close is a no-op for pool-based store."""
        await store.close()  # Should not raise


# =============================================================================
# Global Store Accessor Tests
# =============================================================================


class TestGlobalStoreAccessor:
    """Tests for global store accessor functions."""

    def setup_method(self):
        """Reset store before each test."""
        reset_gauntlet_run_store()

    def teardown_method(self):
        """Reset store after each test."""
        reset_gauntlet_run_store()

    def test_get_default_store(self, monkeypatch, tmp_path):
        """Test getting default store uses SQLite."""
        monkeypatch.setenv("ARAGORA_DATA_DIR", str(tmp_path))
        monkeypatch.setenv("ARAGORA_DB_BACKEND", "sqlite")
        store = get_gauntlet_run_store()
        assert isinstance(store, SQLiteGauntletRunStore)

    def test_get_memory_store(self, monkeypatch):
        """Test getting memory store via env var."""
        monkeypatch.setenv("ARAGORA_GAUNTLET_STORE_BACKEND", "memory")
        store = get_gauntlet_run_store()
        assert isinstance(store, InMemoryGauntletRunStore)

    def test_singleton_behavior(self, monkeypatch):
        """Test that get_gauntlet_run_store returns singleton."""
        monkeypatch.setenv("ARAGORA_GAUNTLET_STORE_BACKEND", "memory")
        store1 = get_gauntlet_run_store()
        store2 = get_gauntlet_run_store()
        assert store1 is store2

    def test_set_custom_store(self, monkeypatch):
        """Test setting a custom store."""
        custom_store = InMemoryGauntletRunStore()
        set_gauntlet_run_store(custom_store)
        store = get_gauntlet_run_store()
        assert store is custom_store

    def test_reset_allows_new_store(self, monkeypatch):
        """Test that reset allows creating new store."""
        monkeypatch.setenv("ARAGORA_GAUNTLET_STORE_BACKEND", "memory")
        store1 = get_gauntlet_run_store()
        reset_gauntlet_run_store()
        store2 = get_gauntlet_run_store()
        assert store1 is not store2

    def test_redis_backend_selection(self, monkeypatch, tmp_path):
        """Test Redis backend selection."""
        monkeypatch.setenv("ARAGORA_GAUNTLET_STORE_BACKEND", "redis")
        monkeypatch.setenv("ARAGORA_DATA_DIR", str(tmp_path))
        # Redis will fail to connect and use fallback
        store = get_gauntlet_run_store()
        assert isinstance(store, RedisGauntletRunStore)

    def test_store_specific_backend_overrides_global(self, monkeypatch):
        """Test that ARAGORA_GAUNTLET_STORE_BACKEND overrides ARAGORA_DB_BACKEND."""
        monkeypatch.setenv("ARAGORA_DB_BACKEND", "sqlite")
        monkeypatch.setenv("ARAGORA_GAUNTLET_STORE_BACKEND", "memory")
        store = get_gauntlet_run_store()
        assert isinstance(store, InMemoryGauntletRunStore)


# =============================================================================
# Concurrency Tests
# =============================================================================


class TestConcurrency:
    """Tests for thread-safety and concurrent operations."""

    @pytest.fixture
    def store(self):
        """Create an in-memory store for concurrency testing."""
        return InMemoryGauntletRunStore()

    @pytest.mark.asyncio
    async def test_concurrent_saves(self, store):
        """Test that concurrent saves don't cause data corruption."""

        async def save_run(i: int):
            await store.save(
                {
                    "run_id": f"run-{i}",
                    "template_id": "template-1",
                    "status": "pending",
                }
            )

        # Save 50 runs concurrently
        await asyncio.gather(*[save_run(i) for i in range(50)])

        # Verify all were saved
        all_runs = await store.list_all()
        assert len(all_runs) == 50

    @pytest.mark.asyncio
    async def test_concurrent_reads_and_writes(self, store, sample_run_data):
        """Test concurrent reads and writes."""
        await store.save(sample_run_data)

        async def read_run():
            return await store.get(sample_run_data["run_id"])

        async def update_run():
            await store.update_status(sample_run_data["run_id"], "running")

        # Mix reads and writes
        tasks = [read_run() for _ in range(10)] + [update_run() for _ in range(5)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Verify no exceptions
        exceptions = [r for r in results if isinstance(r, Exception)]
        assert len(exceptions) == 0
