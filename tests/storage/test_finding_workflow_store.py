"""
Tests for FindingWorkflowStore backends.

Tests all three backends: InMemory, SQLite, and Redis (with fallback).
"""

import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from aragora.storage.finding_workflow_store import (
    WorkflowDataItem,
    InMemoryFindingWorkflowStore,
    SQLiteFindingWorkflowStore,
    RedisFindingWorkflowStore,
    get_finding_workflow_store,
    reset_finding_workflow_store,
    set_finding_workflow_store,
)


@pytest.fixture
def temp_db_path():
    """Create a temporary database path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test_finding_workflows.db"


@pytest.fixture
def memory_store():
    """Create an in-memory store for testing."""
    return InMemoryFindingWorkflowStore()


@pytest.fixture
def sqlite_store(temp_db_path):
    """Create a SQLite store for testing."""
    return SQLiteFindingWorkflowStore(temp_db_path)


@pytest.fixture
def sample_workflow():
    """Create a sample workflow data dict."""
    return {
        "finding_id": "finding-123",
        "current_state": "open",
        "history": [],
        "assigned_to": None,
        "assigned_by": None,
        "assigned_at": None,
        "priority": 3,
        "due_date": None,
        "linked_findings": [],
        "parent_finding_id": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


@pytest.fixture
def sample_assigned_workflow():
    """Create a workflow assigned to a user."""
    return {
        "finding_id": "finding-456",
        "current_state": "investigating",
        "history": [
            {
                "id": "event-1",
                "event_type": "state_change",
                "from_state": "open",
                "to_state": "investigating",
                "user_id": "user-1",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        ],
        "assigned_to": "user-1",
        "assigned_by": "admin",
        "assigned_at": datetime.now(timezone.utc).isoformat(),
        "priority": 1,
        "due_date": (datetime.now(timezone.utc) + timedelta(days=7)).isoformat(),
        "linked_findings": [],
        "parent_finding_id": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


@pytest.fixture
def sample_overdue_workflow():
    """Create an overdue workflow."""
    return {
        "finding_id": "finding-overdue",
        "current_state": "investigating",
        "history": [],
        "assigned_to": "user-1",
        "assigned_by": "admin",
        "assigned_at": datetime.now(timezone.utc).isoformat(),
        "priority": 1,
        "due_date": (datetime.now(timezone.utc) - timedelta(days=1)).isoformat(),
        "linked_findings": [],
        "parent_finding_id": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


class TestWorkflowDataItem:
    """Tests for WorkflowDataItem dataclass."""

    def test_to_dict(self, sample_workflow):
        """Should serialize to dict."""
        item = WorkflowDataItem.from_dict(sample_workflow)
        result = item.to_dict()
        assert result["finding_id"] == "finding-123"
        assert result["current_state"] == "open"
        assert result["priority"] == 3

    def test_to_json_roundtrip(self, sample_workflow):
        """JSON serialization should preserve data."""
        item = WorkflowDataItem.from_dict(sample_workflow)
        json_str = item.to_json()
        restored = WorkflowDataItem.from_json(json_str)
        assert restored.finding_id == item.finding_id
        assert restored.current_state == item.current_state
        assert restored.priority == item.priority

    def test_default_timestamps(self):
        """Should set default timestamps if not provided."""
        item = WorkflowDataItem(finding_id="test-123")
        assert item.created_at != ""
        assert item.updated_at != ""


class TestInMemoryFindingWorkflowStore:
    """Tests for InMemoryFindingWorkflowStore."""

    @pytest.mark.asyncio
    async def test_save_and_get(self, memory_store, sample_workflow):
        """Should save and retrieve workflow."""
        await memory_store.save(sample_workflow)
        retrieved = await memory_store.get("finding-123")
        assert retrieved is not None
        assert retrieved["current_state"] == "open"

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, memory_store):
        """Should return None for nonexistent finding."""
        result = await memory_store.get("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete(self, memory_store, sample_workflow):
        """Should delete workflow."""
        await memory_store.save(sample_workflow)
        deleted = await memory_store.delete("finding-123")
        assert deleted is True
        result = await memory_store.get("finding-123")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, memory_store):
        """Should return False when deleting nonexistent."""
        deleted = await memory_store.delete("nonexistent")
        assert deleted is False

    @pytest.mark.asyncio
    async def test_list_all(self, memory_store, sample_workflow, sample_assigned_workflow):
        """Should list all workflows."""
        await memory_store.save(sample_workflow)
        await memory_store.save(sample_assigned_workflow)

        all_workflows = await memory_store.list_all()
        assert len(all_workflows) == 2

    @pytest.mark.asyncio
    async def test_list_by_assignee(self, memory_store, sample_workflow, sample_assigned_workflow):
        """Should list workflows by assignee."""
        await memory_store.save(sample_workflow)
        await memory_store.save(sample_assigned_workflow)

        assignments = await memory_store.list_by_assignee("user-1")
        assert len(assignments) == 1
        assert assignments[0]["finding_id"] == "finding-456"

    @pytest.mark.asyncio
    async def test_list_overdue(
        self, memory_store, sample_workflow, sample_assigned_workflow, sample_overdue_workflow
    ):
        """Should list overdue findings."""
        await memory_store.save(sample_workflow)
        await memory_store.save(sample_assigned_workflow)
        await memory_store.save(sample_overdue_workflow)

        overdue = await memory_store.list_overdue()
        assert len(overdue) == 1
        assert overdue[0]["finding_id"] == "finding-overdue"

    @pytest.mark.asyncio
    async def test_list_overdue_excludes_terminal(self, memory_store):
        """Should exclude terminal states from overdue list."""
        resolved_workflow = {
            "finding_id": "finding-resolved",
            "current_state": "resolved",
            "history": [],
            "assigned_to": None,
            "priority": 1,
            "due_date": (datetime.now(timezone.utc) - timedelta(days=1)).isoformat(),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        await memory_store.save(resolved_workflow)

        overdue = await memory_store.list_overdue()
        assert len(overdue) == 0

    @pytest.mark.asyncio
    async def test_list_by_state(self, memory_store, sample_workflow, sample_assigned_workflow):
        """Should list workflows by state."""
        await memory_store.save(sample_workflow)
        await memory_store.save(sample_assigned_workflow)

        open_findings = await memory_store.list_by_state("open")
        assert len(open_findings) == 1
        assert open_findings[0]["finding_id"] == "finding-123"

        investigating = await memory_store.list_by_state("investigating")
        assert len(investigating) == 1
        assert investigating[0]["finding_id"] == "finding-456"


class TestSQLiteFindingWorkflowStore:
    """Tests for SQLiteFindingWorkflowStore."""

    @pytest.mark.asyncio
    async def test_save_and_get(self, sqlite_store, sample_workflow):
        """Should save and retrieve workflow."""
        await sqlite_store.save(sample_workflow)
        retrieved = await sqlite_store.get("finding-123")
        assert retrieved is not None
        assert retrieved["current_state"] == "open"

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, sqlite_store):
        """Should return None for nonexistent finding."""
        result = await sqlite_store.get("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete(self, sqlite_store, sample_workflow):
        """Should delete workflow."""
        await sqlite_store.save(sample_workflow)
        deleted = await sqlite_store.delete("finding-123")
        assert deleted is True
        result = await sqlite_store.get("finding-123")
        assert result is None

    @pytest.mark.asyncio
    async def test_persistence(self, temp_db_path, sample_workflow):
        """Data should persist across store instances."""
        store1 = SQLiteFindingWorkflowStore(temp_db_path)
        await store1.save(sample_workflow)
        await store1.close()

        store2 = SQLiteFindingWorkflowStore(temp_db_path)
        retrieved = await store2.get("finding-123")
        assert retrieved is not None
        assert retrieved["current_state"] == "open"
        await store2.close()

    @pytest.mark.asyncio
    async def test_update_existing(self, sqlite_store, sample_workflow):
        """Should update existing workflow."""
        await sqlite_store.save(sample_workflow)

        sample_workflow["current_state"] = "investigating"
        sample_workflow["assigned_to"] = "user-1"
        await sqlite_store.save(sample_workflow)

        retrieved = await sqlite_store.get("finding-123")
        assert retrieved is not None
        assert retrieved["current_state"] == "investigating"
        assert retrieved["assigned_to"] == "user-1"

    @pytest.mark.asyncio
    async def test_list_by_assignee(self, sqlite_store, sample_workflow, sample_assigned_workflow):
        """Should list workflows by assignee."""
        await sqlite_store.save(sample_workflow)
        await sqlite_store.save(sample_assigned_workflow)

        assignments = await sqlite_store.list_by_assignee("user-1")
        assert len(assignments) == 1
        assert assignments[0]["finding_id"] == "finding-456"

    @pytest.mark.asyncio
    async def test_list_overdue(self, sqlite_store, sample_workflow, sample_overdue_workflow):
        """Should list overdue findings."""
        await sqlite_store.save(sample_workflow)
        await sqlite_store.save(sample_overdue_workflow)

        overdue = await sqlite_store.list_overdue()
        assert len(overdue) == 1
        assert overdue[0]["finding_id"] == "finding-overdue"

    @pytest.mark.asyncio
    async def test_list_by_state(self, sqlite_store, sample_workflow, sample_assigned_workflow):
        """Should list workflows by state."""
        await sqlite_store.save(sample_workflow)
        await sqlite_store.save(sample_assigned_workflow)

        open_findings = await sqlite_store.list_by_state("open")
        assert len(open_findings) == 1

        investigating = await sqlite_store.list_by_state("investigating")
        assert len(investigating) == 1


class TestRedisFindingWorkflowStore:
    """Tests for RedisFindingWorkflowStore (with SQLite fallback)."""

    @pytest.fixture
    def redis_store(self, temp_db_path):
        """Create a Redis store (will use SQLite fallback if Redis unavailable)."""
        return RedisFindingWorkflowStore(temp_db_path, redis_url="redis://localhost:6379")

    @pytest.mark.asyncio
    async def test_save_and_get(self, redis_store, sample_workflow):
        """Should save and retrieve workflow."""
        await redis_store.save(sample_workflow)
        retrieved = await redis_store.get("finding-123")
        assert retrieved is not None
        assert retrieved["current_state"] == "open"

    @pytest.mark.asyncio
    async def test_delete(self, redis_store, sample_workflow):
        """Should delete workflow."""
        await redis_store.save(sample_workflow)
        deleted = await redis_store.delete("finding-123")
        assert deleted is True

    @pytest.mark.asyncio
    async def test_fallback_persistence(self, temp_db_path, sample_workflow):
        """SQLite fallback should persist data."""
        store1 = RedisFindingWorkflowStore(temp_db_path)
        await store1.save(sample_workflow)
        await store1.close()

        store2 = RedisFindingWorkflowStore(temp_db_path)
        retrieved = await store2.get("finding-123")
        assert retrieved is not None
        await store2.close()

    @pytest.mark.asyncio
    async def test_list_by_assignee(self, redis_store, sample_workflow, sample_assigned_workflow):
        """Should list workflows by assignee."""
        await redis_store.save(sample_workflow)
        await redis_store.save(sample_assigned_workflow)

        assignments = await redis_store.list_by_assignee("user-1")
        assert len(assignments) == 1
        assert assignments[0]["finding_id"] == "finding-456"


class TestGlobalStore:
    """Tests for global store factory functions."""

    def setup_method(self):
        """Reset global store before each test."""
        reset_finding_workflow_store()

    def teardown_method(self):
        """Reset global store after each test."""
        reset_finding_workflow_store()

    def test_get_default_store(self, monkeypatch, temp_db_path):
        """Should create default SQLite store."""
        monkeypatch.setenv("ARAGORA_DATA_DIR", str(temp_db_path.parent))
        store = get_finding_workflow_store()
        assert isinstance(store, SQLiteFindingWorkflowStore)

    def test_get_memory_store(self, monkeypatch):
        """Should create in-memory store when configured."""
        monkeypatch.setenv("ARAGORA_WORKFLOW_STORE_BACKEND", "memory")
        store = get_finding_workflow_store()
        assert isinstance(store, InMemoryFindingWorkflowStore)

    def test_set_custom_store(self):
        """Should allow setting custom store."""
        custom_store = InMemoryFindingWorkflowStore()
        set_finding_workflow_store(custom_store)
        store = get_finding_workflow_store()
        assert store is custom_store

    def test_singleton_pattern(self, monkeypatch):
        """Should return same instance on multiple calls."""
        monkeypatch.setenv("ARAGORA_WORKFLOW_STORE_BACKEND", "memory")
        store1 = get_finding_workflow_store()
        store2 = get_finding_workflow_store()
        assert store1 is store2


class TestHistoryPreservation:
    """Tests for history/audit trail preservation."""

    @pytest.mark.asyncio
    async def test_history_persisted(self, memory_store, sample_workflow):
        """Should preserve workflow history."""
        sample_workflow["history"] = [
            {
                "id": "event-1",
                "event_type": "state_change",
                "from_state": "open",
                "to_state": "triaging",
                "user_id": "user-1",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            {
                "id": "event-2",
                "event_type": "comment",
                "user_id": "user-2",
                "comment": "Test comment",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        ]

        await memory_store.save(sample_workflow)
        retrieved = await memory_store.get("finding-123")

        assert retrieved is not None
        assert len(retrieved["history"]) == 2
        assert retrieved["history"][0]["event_type"] == "state_change"
        assert retrieved["history"][1]["comment"] == "Test comment"

    @pytest.mark.asyncio
    async def test_linked_findings_persisted(self, sqlite_store, sample_workflow):
        """Should preserve linked findings."""
        sample_workflow["linked_findings"] = ["finding-789", "finding-999"]
        sample_workflow["parent_finding_id"] = "finding-parent"

        await sqlite_store.save(sample_workflow)
        retrieved = await sqlite_store.get("finding-123")

        assert retrieved is not None
        assert len(retrieved["linked_findings"]) == 2
        assert "finding-789" in retrieved["linked_findings"]
        assert retrieved["parent_finding_id"] == "finding-parent"
