"""Tests for GauntletRunStore backends."""

import asyncio
import pytest
import tempfile
from pathlib import Path
from datetime import datetime

from aragora.storage.gauntlet_run_store import (
    GauntletRunItem,
    InMemoryGauntletRunStore,
    SQLiteGauntletRunStore,
    get_gauntlet_run_store,
    reset_gauntlet_run_store,
)


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


class TestGauntletRunItem:
    """Tests for GauntletRunItem dataclass."""

    def test_default_timestamps(self):
        """Test that timestamps are set by default."""
        item = GauntletRunItem(run_id="test-1", template_id="test-template")
        assert item.created_at != ""
        assert item.updated_at != ""

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
    async def test_list_by_status(self, store, sample_run_data, sample_run_data_2):
        """Test listing runs by status."""
        await store.save(sample_run_data)  # pending
        await store.save(sample_run_data_2)  # running
        pending = await store.list_by_status("pending")
        assert len(pending) == 1
        assert pending[0]["run_id"] == sample_run_data["run_id"]

    @pytest.mark.asyncio
    async def test_list_by_template(self, store, sample_run_data, sample_run_data_2):
        """Test listing runs by template."""
        await store.save(sample_run_data)
        await store.save(sample_run_data_2)
        security = await store.list_by_template("security-audit")
        assert len(security) == 1
        assert security[0]["run_id"] == sample_run_data["run_id"]

    @pytest.mark.asyncio
    async def test_list_active(self, store, sample_run_data, sample_run_data_2):
        """Test listing active runs."""
        await store.save(sample_run_data)  # pending
        await store.save(sample_run_data_2)  # running
        completed_data = {
            "run_id": "gauntlet-test-003",
            "template_id": "test",
            "status": "completed",
        }
        await store.save(completed_data)
        active = await store.list_active()
        assert len(active) == 2  # pending + running

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
