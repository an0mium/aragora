"""Tests for the Nomic Beads module - state management across improvement cycles.

This module tests the Bead, BeadEvent, and BeadStore classes which provide
git-backed atomic work unit tracking for the Nomic Loop.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from aragora.nomic.beads import (
    Bead,
    BeadEvent,
    BeadPriority,
    BeadStatus,
    BeadStore,
    BeadType,
    create_bead_store,
    get_bead_store,
    reset_bead_store,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def tmp_bead_dir(tmp_path: Path) -> Path:
    """Create a temporary directory for bead storage."""
    bead_dir = tmp_path / "beads"
    bead_dir.mkdir()
    return bead_dir


@pytest.fixture
async def bead_store(tmp_bead_dir: Path) -> BeadStore:
    """Create and initialize a bead store for testing (git disabled)."""
    store = BeadStore(tmp_bead_dir, git_enabled=False)
    await store.initialize()
    return store


@pytest.fixture
async def git_bead_store(tmp_bead_dir: Path) -> BeadStore:
    """Create and initialize a bead store with git enabled for testing."""
    store = BeadStore(tmp_bead_dir, git_enabled=True, auto_commit=False)
    await store.initialize()
    return store


@pytest.fixture
def sample_bead() -> Bead:
    """Create a sample bead for testing."""
    return Bead.create(
        bead_type=BeadType.TASK,
        title="Sample Task",
        description="A sample task for testing",
        priority=BeadPriority.NORMAL,
        tags=["test", "sample"],
        metadata={"key": "value"},
    )


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the singleton store between tests."""
    reset_bead_store()
    yield
    reset_bead_store()


# =============================================================================
# Bead Creation and Lifecycle Tests
# =============================================================================


class TestBeadCreation:
    """Tests for Bead creation and basic operations."""

    def test_bead_create_with_defaults(self):
        """Test creating a bead with default values."""
        bead = Bead.create(
            bead_type=BeadType.TASK,
            title="Test Task",
        )

        assert bead.id is not None
        assert len(bead.id) == 36  # UUID format
        assert bead.bead_type == BeadType.TASK
        assert bead.title == "Test Task"
        assert bead.description == ""
        assert bead.status == BeadStatus.PENDING
        assert bead.priority == BeadPriority.NORMAL
        assert bead.claimed_by is None
        assert bead.claimed_at is None
        assert bead.completed_at is None
        assert bead.parent_id is None
        assert bead.dependencies == []
        assert bead.tags == []
        assert bead.metadata == {}
        assert bead.error_message is None
        assert bead.attempt_count == 0
        assert bead.max_attempts == 3
        assert bead.created_at is not None
        assert bead.updated_at is not None

    def test_bead_create_with_all_fields(self):
        """Test creating a bead with all fields specified."""
        bead = Bead.create(
            bead_type=BeadType.EPIC,
            title="Epic Task",
            description="A detailed description",
            parent_id="parent-123",
            dependencies=["dep-1", "dep-2"],
            priority=BeadPriority.URGENT,
            tags=["urgent", "important"],
            metadata={"source": "test", "version": 1},
        )

        assert bead.bead_type == BeadType.EPIC
        assert bead.title == "Epic Task"
        assert bead.description == "A detailed description"
        assert bead.parent_id == "parent-123"
        assert bead.dependencies == ["dep-1", "dep-2"]
        assert bead.priority == BeadPriority.URGENT
        assert bead.tags == ["urgent", "important"]
        assert bead.metadata == {"source": "test", "version": 1}

    def test_bead_create_all_types(self):
        """Test creating beads of all types."""
        for bead_type in BeadType:
            bead = Bead.create(bead_type=bead_type, title=f"Test {bead_type.value}")
            assert bead.bead_type == bead_type

    def test_bead_create_all_priorities(self):
        """Test creating beads with all priority levels."""
        for priority in BeadPriority:
            bead = Bead.create(
                bead_type=BeadType.TASK,
                title="Test Task",
                priority=priority,
            )
            assert bead.priority == priority


class TestBeadSerialization:
    """Tests for Bead serialization and deserialization."""

    def test_bead_to_dict(self, sample_bead):
        """Test serializing a bead to dictionary."""
        data = sample_bead.to_dict()

        assert data["id"] == sample_bead.id
        assert data["bead_type"] == "task"
        assert data["status"] == "pending"
        assert data["title"] == "Sample Task"
        assert data["description"] == "A sample task for testing"
        assert data["priority"] == BeadPriority.NORMAL.value
        assert data["tags"] == ["test", "sample"]
        assert data["metadata"] == {"key": "value"}
        assert "created_at" in data
        assert "updated_at" in data

    def test_bead_to_dict_with_timestamps(self):
        """Test serializing a bead with claim and completion timestamps."""
        bead = Bead.create(bead_type=BeadType.TASK, title="Test")
        bead.claimed_at = datetime.now(timezone.utc)
        bead.completed_at = datetime.now(timezone.utc)

        data = bead.to_dict()

        assert "claimed_at" in data
        assert "completed_at" in data
        assert isinstance(data["claimed_at"], str)
        assert isinstance(data["completed_at"], str)

    def test_bead_from_dict(self, sample_bead):
        """Test deserializing a bead from dictionary."""
        data = sample_bead.to_dict()
        restored = Bead.from_dict(data)

        assert restored.id == sample_bead.id
        assert restored.bead_type == sample_bead.bead_type
        assert restored.status == sample_bead.status
        assert restored.title == sample_bead.title
        assert restored.description == sample_bead.description
        assert restored.priority == sample_bead.priority
        assert restored.tags == sample_bead.tags
        assert restored.metadata == sample_bead.metadata

    def test_bead_roundtrip_serialization(self):
        """Test that bead survives roundtrip serialization."""
        original = Bead.create(
            bead_type=BeadType.ISSUE,
            title="Issue Task",
            description="Issue description",
            parent_id="parent-456",
            dependencies=["dep-a", "dep-b"],
            priority=BeadPriority.HIGH,
            tags=["bug", "critical"],
            metadata={"severity": "high"},
        )
        original.claimed_by = "agent-123"
        original.claimed_at = datetime.now(timezone.utc)
        original.error_message = "Test error"
        original.attempt_count = 2

        data = original.to_dict()
        restored = Bead.from_dict(data)

        assert restored.id == original.id
        assert restored.parent_id == original.parent_id
        assert restored.dependencies == original.dependencies
        assert restored.claimed_by == original.claimed_by
        assert restored.error_message == original.error_message
        assert restored.attempt_count == original.attempt_count

    def test_bead_from_dict_minimal(self):
        """Test deserializing a bead with minimal data."""
        data = {
            "id": "test-id-123",
            "bead_type": "task",
            "status": "pending",
            "title": "Minimal Task",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        bead = Bead.from_dict(data)

        assert bead.id == "test-id-123"
        assert bead.description == ""
        assert bead.dependencies == []
        assert bead.tags == []
        assert bead.metadata == {}


class TestBeadLifecycleMethods:
    """Tests for Bead lifecycle helper methods."""

    def test_can_start_no_dependencies(self):
        """Test can_start returns True when no dependencies."""
        bead = Bead.create(bead_type=BeadType.TASK, title="Test")
        assert bead.can_start(set()) is True

    def test_can_start_dependencies_met(self):
        """Test can_start returns True when dependencies are met."""
        bead = Bead.create(
            bead_type=BeadType.TASK,
            title="Test",
            dependencies=["dep-1", "dep-2"],
        )
        completed_ids = {"dep-1", "dep-2", "dep-3"}
        assert bead.can_start(completed_ids) is True

    def test_can_start_dependencies_not_met(self):
        """Test can_start returns False when dependencies are not met."""
        bead = Bead.create(
            bead_type=BeadType.TASK,
            title="Test",
            dependencies=["dep-1", "dep-2"],
        )
        completed_ids = {"dep-1"}  # Missing dep-2
        assert bead.can_start(completed_ids) is False

    def test_is_terminal_completed(self):
        """Test is_terminal returns True for COMPLETED status."""
        bead = Bead.create(bead_type=BeadType.TASK, title="Test")
        bead.status = BeadStatus.COMPLETED
        assert bead.is_terminal() is True

    def test_is_terminal_failed(self):
        """Test is_terminal returns True for FAILED status."""
        bead = Bead.create(bead_type=BeadType.TASK, title="Test")
        bead.status = BeadStatus.FAILED
        assert bead.is_terminal() is True

    def test_is_terminal_cancelled(self):
        """Test is_terminal returns True for CANCELLED status."""
        bead = Bead.create(bead_type=BeadType.TASK, title="Test")
        bead.status = BeadStatus.CANCELLED
        assert bead.is_terminal() is True

    def test_is_terminal_non_terminal_statuses(self):
        """Test is_terminal returns False for non-terminal statuses."""
        non_terminal = [
            BeadStatus.PENDING,
            BeadStatus.CLAIMED,
            BeadStatus.RUNNING,
            BeadStatus.BLOCKED,
        ]
        for status in non_terminal:
            bead = Bead.create(bead_type=BeadType.TASK, title="Test")
            bead.status = status
            assert bead.is_terminal() is False, f"Expected False for {status}"

    def test_can_retry_failed_under_max(self):
        """Test can_retry returns True when failed and under max attempts."""
        bead = Bead.create(bead_type=BeadType.TASK, title="Test")
        bead.status = BeadStatus.FAILED
        bead.attempt_count = 1
        bead.max_attempts = 3
        assert bead.can_retry() is True

    def test_can_retry_failed_at_max(self):
        """Test can_retry returns False when at max attempts."""
        bead = Bead.create(bead_type=BeadType.TASK, title="Test")
        bead.status = BeadStatus.FAILED
        bead.attempt_count = 3
        bead.max_attempts = 3
        assert bead.can_retry() is False

    def test_can_retry_not_failed(self):
        """Test can_retry returns False when not failed."""
        bead = Bead.create(bead_type=BeadType.TASK, title="Test")
        bead.status = BeadStatus.PENDING
        bead.attempt_count = 0
        assert bead.can_retry() is False


class TestBeadProtocolProperties:
    """Tests for BeadRecord protocol properties."""

    def test_bead_id_property(self, sample_bead):
        """Test bead_id protocol property."""
        assert sample_bead.bead_id == sample_bead.id

    def test_bead_convoy_id_property(self, sample_bead):
        """Test bead_convoy_id returns None (inverse relationship)."""
        assert sample_bead.bead_convoy_id is None

    def test_bead_status_value_property(self, sample_bead):
        """Test bead_status_value protocol property."""
        assert sample_bead.bead_status_value == "pending"

    def test_bead_content_property(self, sample_bead):
        """Test bead_content maps to description."""
        assert sample_bead.bead_content == sample_bead.description

    def test_bead_created_at_property(self, sample_bead):
        """Test bead_created_at protocol property."""
        assert sample_bead.bead_created_at == sample_bead.created_at

    def test_bead_metadata_property(self, sample_bead):
        """Test bead_metadata protocol property."""
        assert sample_bead.bead_metadata == sample_bead.metadata


# =============================================================================
# BeadEvent Tests
# =============================================================================


class TestBeadEvent:
    """Tests for BeadEvent dataclass."""

    def test_event_creation(self):
        """Test creating a bead event."""
        event = BeadEvent(
            event_id="evt-123",
            bead_id="bead-456",
            event_type="created",
            timestamp=datetime.now(timezone.utc),
            agent_id="agent-789",
            old_status=None,
            new_status=BeadStatus.PENDING,
            data={"title": "Test"},
        )

        assert event.event_id == "evt-123"
        assert event.bead_id == "bead-456"
        assert event.event_type == "created"
        assert event.agent_id == "agent-789"
        assert event.new_status == BeadStatus.PENDING
        assert event.data == {"title": "Test"}

    def test_event_to_dict(self):
        """Test serializing event to dictionary."""
        timestamp = datetime.now(timezone.utc)
        event = BeadEvent(
            event_id="evt-123",
            bead_id="bead-456",
            event_type="status_changed",
            timestamp=timestamp,
            agent_id="agent-789",
            old_status=BeadStatus.PENDING,
            new_status=BeadStatus.RUNNING,
            data={"reason": "starting work"},
        )

        data = event.to_dict()

        assert data["event_id"] == "evt-123"
        assert data["bead_id"] == "bead-456"
        assert data["event_type"] == "status_changed"
        assert data["timestamp"] == timestamp.isoformat()
        assert data["agent_id"] == "agent-789"
        assert data["old_status"] == "pending"
        assert data["new_status"] == "running"
        assert data["data"] == {"reason": "starting work"}

    def test_event_to_dict_none_statuses(self):
        """Test serializing event with None statuses."""
        event = BeadEvent(
            event_id="evt-123",
            bead_id="bead-456",
            event_type="comment",
            timestamp=datetime.now(timezone.utc),
        )

        data = event.to_dict()

        assert data["old_status"] is None
        assert data["new_status"] is None
        assert data["agent_id"] is None


# =============================================================================
# BeadStore Multi-Cycle State Management Tests
# =============================================================================


class TestBeadStoreInitialization:
    """Tests for BeadStore initialization."""

    @pytest.mark.asyncio
    async def test_store_initialization(self, tmp_bead_dir: Path):
        """Test basic store initialization."""
        store = BeadStore(tmp_bead_dir, git_enabled=False)
        await store.initialize()

        assert store._initialized is True
        assert store.bead_dir == tmp_bead_dir
        assert store.bead_file == tmp_bead_dir / "beads.jsonl"
        assert store.events_file == tmp_bead_dir / "events.jsonl"

    @pytest.mark.asyncio
    async def test_store_double_initialization(self, tmp_bead_dir: Path):
        """Test that double initialization is safe."""
        store = BeadStore(tmp_bead_dir, git_enabled=False)
        await store.initialize()
        await store.initialize()  # Should not raise

        assert store._initialized is True

    @pytest.mark.asyncio
    async def test_store_creates_directory(self, tmp_path: Path):
        """Test store creates directory if not exists."""
        new_dir = tmp_path / "new_bead_dir"
        store = BeadStore(new_dir, git_enabled=False)
        await store.initialize()

        assert new_dir.exists()


class TestBeadStoreCRUD:
    """Tests for BeadStore CRUD operations."""

    @pytest.mark.asyncio
    async def test_create_bead(self, bead_store: BeadStore, sample_bead: Bead):
        """Test creating a bead in the store."""
        bead_id = await bead_store.create(sample_bead)

        assert bead_id == sample_bead.id
        assert sample_bead.id in bead_store._beads_cache

    @pytest.mark.asyncio
    async def test_create_duplicate_bead_raises(self, bead_store: BeadStore, sample_bead: Bead):
        """Test creating a duplicate bead raises ValueError."""
        await bead_store.create(sample_bead)

        with pytest.raises(ValueError, match="already exists"):
            await bead_store.create(sample_bead)

    @pytest.mark.asyncio
    async def test_get_bead(self, bead_store: BeadStore, sample_bead: Bead):
        """Test getting a bead by ID."""
        await bead_store.create(sample_bead)

        retrieved = await bead_store.get(sample_bead.id)

        assert retrieved is not None
        assert retrieved.id == sample_bead.id
        assert retrieved.title == sample_bead.title

    @pytest.mark.asyncio
    async def test_get_nonexistent_bead(self, bead_store: BeadStore):
        """Test getting a nonexistent bead returns None."""
        result = await bead_store.get("nonexistent-id")
        assert result is None

    @pytest.mark.asyncio
    async def test_update_bead(self, bead_store: BeadStore, sample_bead: Bead):
        """Test updating a bead."""
        await bead_store.create(sample_bead)

        sample_bead.title = "Updated Title"
        sample_bead.description = "Updated description"
        await bead_store.update(sample_bead)

        retrieved = await bead_store.get(sample_bead.id)
        assert retrieved.title == "Updated Title"
        assert retrieved.description == "Updated description"

    @pytest.mark.asyncio
    async def test_update_nonexistent_bead_raises(self, bead_store: BeadStore, sample_bead: Bead):
        """Test updating a nonexistent bead raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            await bead_store.update(sample_bead)

    @pytest.mark.asyncio
    async def test_update_records_status_change_event(
        self, bead_store: BeadStore, sample_bead: Bead
    ):
        """Test that status changes are recorded as events."""
        await bead_store.create(sample_bead)

        sample_bead.status = BeadStatus.RUNNING
        await bead_store.update(sample_bead)

        # Verify event file contains the status change
        assert bead_store.events_file.exists()
        with open(bead_store.events_file) as f:
            lines = f.readlines()
            # Should have created event and status_changed event
            assert len(lines) >= 1


class TestBeadStoreClaim:
    """Tests for BeadStore claim operations."""

    @pytest.mark.asyncio
    async def test_claim_bead(self, bead_store: BeadStore, sample_bead: Bead):
        """Test claiming a pending bead."""
        await bead_store.create(sample_bead)

        result = await bead_store.claim(sample_bead.id, "agent-123")

        assert result is True
        bead = await bead_store.get(sample_bead.id)
        assert bead.status == BeadStatus.CLAIMED
        assert bead.claimed_by == "agent-123"
        assert bead.claimed_at is not None

    @pytest.mark.asyncio
    async def test_claim_already_claimed_bead(self, bead_store: BeadStore, sample_bead: Bead):
        """Test claiming an already claimed bead returns False."""
        await bead_store.create(sample_bead)
        await bead_store.claim(sample_bead.id, "agent-123")

        result = await bead_store.claim(sample_bead.id, "agent-456")

        assert result is False

    @pytest.mark.asyncio
    async def test_claim_nonexistent_bead_raises(self, bead_store: BeadStore):
        """Test claiming a nonexistent bead raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            await bead_store.claim("nonexistent-id", "agent-123")


class TestBeadStoreStatusUpdates:
    """Tests for BeadStore status update operations."""

    @pytest.mark.asyncio
    async def test_update_status_to_running(self, bead_store: BeadStore, sample_bead: Bead):
        """Test updating status to RUNNING increments attempt count."""
        await bead_store.create(sample_bead)

        await bead_store.update_status(sample_bead.id, BeadStatus.RUNNING)

        bead = await bead_store.get(sample_bead.id)
        assert bead.status == BeadStatus.RUNNING
        assert bead.attempt_count == 1

    @pytest.mark.asyncio
    async def test_update_status_to_completed(self, bead_store: BeadStore, sample_bead: Bead):
        """Test updating status to COMPLETED sets completed_at."""
        await bead_store.create(sample_bead)

        await bead_store.update_status(sample_bead.id, BeadStatus.COMPLETED)

        bead = await bead_store.get(sample_bead.id)
        assert bead.status == BeadStatus.COMPLETED
        assert bead.completed_at is not None

    @pytest.mark.asyncio
    async def test_update_status_to_failed_with_error(
        self, bead_store: BeadStore, sample_bead: Bead
    ):
        """Test updating status to FAILED records error message."""
        await bead_store.create(sample_bead)

        await bead_store.update_status(
            sample_bead.id,
            BeadStatus.FAILED,
            error_message="Connection timeout",
        )

        bead = await bead_store.get(sample_bead.id)
        assert bead.status == BeadStatus.FAILED
        assert bead.error_message == "Connection timeout"
        assert bead.attempt_count == 1

    @pytest.mark.asyncio
    async def test_update_status_nonexistent_bead_raises(self, bead_store: BeadStore):
        """Test updating status of nonexistent bead raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            await bead_store.update_status("nonexistent-id", BeadStatus.RUNNING)


class TestBeadStoreQueries:
    """Tests for BeadStore query operations."""

    @pytest.mark.asyncio
    async def test_list_by_status(self, bead_store: BeadStore):
        """Test listing beads by status."""
        bead1 = Bead.create(bead_type=BeadType.TASK, title="Task 1")
        bead2 = Bead.create(bead_type=BeadType.TASK, title="Task 2")
        bead3 = Bead.create(bead_type=BeadType.TASK, title="Task 3")

        await bead_store.create(bead1)
        await bead_store.create(bead2)
        await bead_store.create(bead3)

        await bead_store.update_status(bead2.id, BeadStatus.RUNNING)
        await bead_store.update_status(bead3.id, BeadStatus.COMPLETED)

        pending = await bead_store.list_by_status(BeadStatus.PENDING)
        running = await bead_store.list_by_status(BeadStatus.RUNNING)
        completed = await bead_store.list_by_status(BeadStatus.COMPLETED)

        assert len(pending) == 1
        assert len(running) == 1
        assert len(completed) == 1

    @pytest.mark.asyncio
    async def test_list_by_agent(self, bead_store: BeadStore):
        """Test listing beads by agent."""
        bead1 = Bead.create(bead_type=BeadType.TASK, title="Task 1")
        bead2 = Bead.create(bead_type=BeadType.TASK, title="Task 2")

        await bead_store.create(bead1)
        await bead_store.create(bead2)

        await bead_store.claim(bead1.id, "agent-1")
        await bead_store.claim(bead2.id, "agent-2")

        agent1_beads = await bead_store.list_by_agent("agent-1")
        agent2_beads = await bead_store.list_by_agent("agent-2")

        assert len(agent1_beads) == 1
        assert agent1_beads[0].id == bead1.id
        assert len(agent2_beads) == 1
        assert agent2_beads[0].id == bead2.id

    @pytest.mark.asyncio
    async def test_list_by_type(self, bead_store: BeadStore):
        """Test listing beads by type."""
        task = Bead.create(bead_type=BeadType.TASK, title="Task")
        issue = Bead.create(bead_type=BeadType.ISSUE, title="Issue")
        epic = Bead.create(bead_type=BeadType.EPIC, title="Epic")

        await bead_store.create(task)
        await bead_store.create(issue)
        await bead_store.create(epic)

        tasks = await bead_store.list_by_type(BeadType.TASK)
        issues = await bead_store.list_by_type(BeadType.ISSUE)
        epics = await bead_store.list_by_type(BeadType.EPIC)

        assert len(tasks) == 1
        assert len(issues) == 1
        assert len(epics) == 1

    @pytest.mark.asyncio
    async def test_list_beads_with_filters(self, bead_store: BeadStore):
        """Test list_beads with status, priority, and limit filters."""
        for i in range(5):
            bead = Bead.create(
                bead_type=BeadType.TASK,
                title=f"Task {i}",
                priority=BeadPriority.HIGH if i % 2 == 0 else BeadPriority.LOW,
            )
            await bead_store.create(bead)

        # Test limit
        limited = await bead_store.list_beads(limit=3)
        assert len(limited) == 3

        # Test priority filter
        high_priority = await bead_store.list_beads(priority=BeadPriority.HIGH)
        assert len(high_priority) == 3

        # Test status filter
        pending = await bead_store.list_beads(status=BeadStatus.PENDING)
        assert len(pending) == 5

    @pytest.mark.asyncio
    async def test_list_pending_runnable(self, bead_store: BeadStore):
        """Test listing pending runnable beads (dependencies met)."""
        bead1 = Bead.create(bead_type=BeadType.TASK, title="Task 1")
        bead2 = Bead.create(
            bead_type=BeadType.TASK,
            title="Task 2",
            dependencies=[bead1.id],
        )
        bead3 = Bead.create(bead_type=BeadType.TASK, title="Task 3")

        await bead_store.create(bead1)
        await bead_store.create(bead2)
        await bead_store.create(bead3)

        # Initially, bead1 and bead3 should be runnable
        runnable = await bead_store.list_pending_runnable()
        assert len(runnable) == 2

        # Complete bead1
        await bead_store.update_status(bead1.id, BeadStatus.COMPLETED)

        # Now bead2 and bead3 should be runnable
        runnable = await bead_store.list_pending_runnable()
        assert len(runnable) == 2

    @pytest.mark.asyncio
    async def test_list_retryable(self, bead_store: BeadStore):
        """Test listing retryable beads."""
        bead1 = Bead.create(bead_type=BeadType.TASK, title="Task 1")
        bead2 = Bead.create(bead_type=BeadType.TASK, title="Task 2")
        bead3 = Bead.create(bead_type=BeadType.TASK, title="Task 3")

        await bead_store.create(bead1)
        await bead_store.create(bead2)
        await bead_store.create(bead3)

        # Fail bead1 once (retryable)
        await bead_store.update_status(bead1.id, BeadStatus.FAILED)

        # Fail bead2 three times (not retryable)
        for _ in range(3):
            b2 = await bead_store.get(bead2.id)
            b2.status = BeadStatus.PENDING  # Reset for re-fail
            await bead_store.update(b2)
            await bead_store.update_status(bead2.id, BeadStatus.FAILED)

        retryable = await bead_store.list_retryable()
        assert len(retryable) == 1
        assert retryable[0].id == bead1.id

    @pytest.mark.asyncio
    async def test_list_all(self, bead_store: BeadStore):
        """Test listing all beads."""
        for i in range(3):
            bead = Bead.create(bead_type=BeadType.TASK, title=f"Task {i}")
            await bead_store.create(bead)

        all_beads = await bead_store.list_all()
        assert len(all_beads) == 3

    @pytest.mark.asyncio
    async def test_get_children(self, bead_store: BeadStore):
        """Test getting child beads of a parent."""
        parent = Bead.create(bead_type=BeadType.EPIC, title="Parent Epic")
        child1 = Bead.create(
            bead_type=BeadType.TASK,
            title="Child 1",
            parent_id=parent.id,
        )
        child2 = Bead.create(
            bead_type=BeadType.TASK,
            title="Child 2",
            parent_id=parent.id,
        )
        other = Bead.create(bead_type=BeadType.TASK, title="Other")

        await bead_store.create(parent)
        await bead_store.create(child1)
        await bead_store.create(child2)
        await bead_store.create(other)

        children = await bead_store.get_children(parent.id)
        assert len(children) == 2
        assert all(c.parent_id == parent.id for c in children)


class TestBeadStoreStatistics:
    """Tests for BeadStore statistics."""

    @pytest.mark.asyncio
    async def test_get_statistics(self, bead_store: BeadStore):
        """Test getting store statistics."""
        task1 = Bead.create(bead_type=BeadType.TASK, title="Task 1")
        task2 = Bead.create(bead_type=BeadType.TASK, title="Task 2")
        issue = Bead.create(bead_type=BeadType.ISSUE, title="Issue")

        await bead_store.create(task1)
        await bead_store.create(task2)
        await bead_store.create(issue)

        await bead_store.claim(task1.id, "agent-1")
        await bead_store.update_status(task2.id, BeadStatus.COMPLETED)

        stats = await bead_store.get_statistics()

        assert stats["total"] == 3
        assert stats["by_type"]["task"] == 2
        assert stats["by_type"]["issue"] == 1
        assert stats["by_status"]["claimed"] == 1
        assert stats["by_status"]["completed"] == 1
        assert stats["by_status"]["pending"] == 1
        assert stats["agents_active"] == 1

    @pytest.mark.asyncio
    async def test_get_statistics_empty_store(self, bead_store: BeadStore):
        """Test getting statistics from empty store."""
        stats = await bead_store.get_statistics()

        assert stats["total"] == 0
        assert stats["by_status"] == {}
        assert stats["by_type"] == {}
        assert stats["agents_active"] == 0


# =============================================================================
# Bead Persistence Tests
# =============================================================================


class TestBeadStorePersistence:
    """Tests for BeadStore persistence across restarts."""

    @pytest.mark.asyncio
    async def test_beads_persisted_to_jsonl(self, tmp_bead_dir: Path, sample_bead: Bead):
        """Test that beads are persisted to JSONL file."""
        store = BeadStore(tmp_bead_dir, git_enabled=False)
        await store.initialize()

        await store.create(sample_bead)

        # Check file exists and contains data
        assert store.bead_file.exists()
        with open(store.bead_file) as f:
            lines = f.readlines()
            assert len(lines) == 1
            data = json.loads(lines[0])
            assert data["id"] == sample_bead.id

    @pytest.mark.asyncio
    async def test_beads_loaded_on_initialization(self, tmp_bead_dir: Path, sample_bead: Bead):
        """Test that beads are loaded when store initializes."""
        # Create first store and add bead
        store1 = BeadStore(tmp_bead_dir, git_enabled=False)
        await store1.initialize()
        await store1.create(sample_bead)

        # Create second store (simulates restart)
        store2 = BeadStore(tmp_bead_dir, git_enabled=False)
        await store2.initialize()

        # Bead should be loaded
        loaded = await store2.get(sample_bead.id)
        assert loaded is not None
        assert loaded.title == sample_bead.title

    @pytest.mark.asyncio
    async def test_events_persisted_to_jsonl(self, tmp_bead_dir: Path, sample_bead: Bead):
        """Test that events are persisted to events JSONL file."""
        store = BeadStore(tmp_bead_dir, git_enabled=False)
        await store.initialize()

        await store.create(sample_bead)
        await store.claim(sample_bead.id, "agent-123")

        # Check events file
        assert store.events_file.exists()
        with open(store.events_file) as f:
            lines = f.readlines()
            assert len(lines) >= 2  # created + claimed events

    @pytest.mark.asyncio
    async def test_store_handles_corrupted_lines(self, tmp_bead_dir: Path):
        """Test store gracefully handles corrupted JSONL lines."""
        bead_file = tmp_bead_dir / "beads.jsonl"

        # Write valid and invalid lines
        bead = Bead.create(bead_type=BeadType.TASK, title="Valid Bead")
        with open(bead_file, "w") as f:
            f.write(json.dumps(bead.to_dict()) + "\n")
            f.write("invalid json line\n")
            f.write("{}\n")  # Missing required fields

        store = BeadStore(tmp_bead_dir, git_enabled=False)
        await store.initialize()

        # Should load valid bead despite corrupted lines
        all_beads = await store.list_all()
        assert len(all_beads) == 1
        assert all_beads[0].id == bead.id

    @pytest.mark.asyncio
    async def test_store_handles_empty_file(self, tmp_bead_dir: Path):
        """Test store handles empty JSONL file."""
        bead_file = tmp_bead_dir / "beads.jsonl"
        bead_file.touch()

        store = BeadStore(tmp_bead_dir, git_enabled=False)
        await store.initialize()

        all_beads = await store.list_all()
        assert len(all_beads) == 0


# =============================================================================
# Git Integration Tests
# =============================================================================


class TestBeadStoreGitIntegration:
    """Tests for BeadStore git integration."""

    @pytest.mark.asyncio
    async def test_git_init_on_initialization(self, tmp_bead_dir: Path):
        """Test git repository is initialized."""
        store = BeadStore(tmp_bead_dir, git_enabled=True)
        await store.initialize()

        git_dir = tmp_bead_dir / ".git"
        assert git_dir.exists()

    @pytest.mark.asyncio
    async def test_commit_to_git(self, git_bead_store: BeadStore, sample_bead: Bead):
        """Test committing changes to git."""
        await git_bead_store.create(sample_bead)

        commit_hash = await git_bead_store.commit_to_git("Test commit")

        assert commit_hash is not None
        assert len(commit_hash) == 8  # Short hash

    @pytest.mark.asyncio
    async def test_commit_no_changes(self, git_bead_store: BeadStore, sample_bead: Bead):
        """Test commit returns None when no changes."""
        await git_bead_store.create(sample_bead)
        await git_bead_store.commit_to_git("First commit")

        # No changes since last commit
        commit_hash = await git_bead_store.commit_to_git("Empty commit")

        assert commit_hash is None

    @pytest.mark.asyncio
    async def test_git_disabled(self, tmp_bead_dir: Path, sample_bead: Bead):
        """Test store works with git disabled."""
        store = BeadStore(tmp_bead_dir, git_enabled=False)
        await store.initialize()

        await store.create(sample_bead)
        commit_hash = await store.commit_to_git("Test")

        assert commit_hash is None
        assert not (tmp_bead_dir / ".git").exists()

    @pytest.mark.asyncio
    async def test_auto_commit(self, tmp_bead_dir: Path, sample_bead: Bead):
        """Test auto-commit feature."""
        store = BeadStore(tmp_bead_dir, git_enabled=True, auto_commit=True)
        await store.initialize()

        # Auto-commit should happen on create
        await store.create(sample_bead)

        # Verify commit was made
        git_dir = tmp_bead_dir / ".git"
        assert git_dir.exists()


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestBeadStoreErrorHandling:
    """Tests for BeadStore error handling."""

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, bead_store: BeadStore):
        """Test concurrent operations are handled safely via locking."""
        beads = [Bead.create(bead_type=BeadType.TASK, title=f"Task {i}") for i in range(10)]

        # Create all beads concurrently
        await asyncio.gather(*[bead_store.create(bead) for bead in beads])

        all_beads = await bead_store.list_all()
        assert len(all_beads) == 10

    @pytest.mark.asyncio
    async def test_claim_race_condition(self, bead_store: BeadStore):
        """Test that only one agent can claim a bead."""
        bead = Bead.create(bead_type=BeadType.TASK, title="Contested Task")
        await bead_store.create(bead)

        # Simulate race condition
        results = await asyncio.gather(
            bead_store.claim(bead.id, "agent-1"),
            bead_store.claim(bead.id, "agent-2"),
            bead_store.claim(bead.id, "agent-3"),
        )

        # Only one should succeed
        assert sum(results) == 1

        bead = await bead_store.get(bead.id)
        assert bead.claimed_by in ["agent-1", "agent-2", "agent-3"]


# =============================================================================
# Factory Functions Tests
# =============================================================================


class TestFactoryFunctions:
    """Tests for factory and convenience functions."""

    @pytest.mark.asyncio
    async def test_create_bead_store(self, tmp_bead_dir: Path):
        """Test create_bead_store factory function."""
        store = await create_bead_store(
            bead_dir=str(tmp_bead_dir),
            git_enabled=False,
            auto_commit=False,
        )

        assert store._initialized is True
        assert store.bead_dir == tmp_bead_dir

    @pytest.mark.asyncio
    async def test_get_bead_store_singleton(self, tmp_bead_dir: Path):
        """Test get_bead_store returns singleton."""
        store1 = await get_bead_store(str(tmp_bead_dir))
        store2 = await get_bead_store(str(tmp_bead_dir))

        assert store1 is store2

    def test_reset_bead_store(self, tmp_bead_dir: Path):
        """Test reset_bead_store clears singleton."""
        reset_bead_store()
        # Should not raise
        reset_bead_store()


# =============================================================================
# Edge Cases Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_bead_with_empty_title(self, bead_store: BeadStore):
        """Test bead with empty title."""
        bead = Bead.create(bead_type=BeadType.TASK, title="")
        await bead_store.create(bead)

        retrieved = await bead_store.get(bead.id)
        assert retrieved.title == ""

    @pytest.mark.asyncio
    async def test_bead_with_unicode_content(self, bead_store: BeadStore):
        """Test bead with unicode characters."""
        bead = Bead.create(
            bead_type=BeadType.TASK,
            title="Unicode Task: \u4e2d\u6587 \u0420\u0443\u0441\u0441\u043a\u0438\u0439 \ud83d\ude80",
            description="Description with emojis \ud83c\udf89 and symbols \u00a9\u00ae\u2122",
            tags=["\u0442\u0435\u0433", "\u6807\u7b7e"],
            metadata={"key": "\u5024"},
        )
        await bead_store.create(bead)

        retrieved = await bead_store.get(bead.id)
        assert "\u4e2d\u6587" in retrieved.title
        assert "\ud83c\udf89" in retrieved.description
        assert "\u0442\u0435\u0433" in retrieved.tags

    @pytest.mark.asyncio
    async def test_bead_with_large_metadata(self, bead_store: BeadStore):
        """Test bead with large metadata."""
        large_data = {"key_" + str(i): "value_" * 100 for i in range(100)}
        bead = Bead.create(
            bead_type=BeadType.TASK,
            title="Large Metadata Task",
            metadata=large_data,
        )
        await bead_store.create(bead)

        retrieved = await bead_store.get(bead.id)
        assert len(retrieved.metadata) == 100

    @pytest.mark.asyncio
    async def test_bead_with_many_dependencies(self, bead_store: BeadStore):
        """Test bead with many dependencies."""
        dependencies = [f"dep-{i}" for i in range(50)]
        bead = Bead.create(
            bead_type=BeadType.TASK,
            title="Many Dependencies Task",
            dependencies=dependencies,
        )
        await bead_store.create(bead)

        retrieved = await bead_store.get(bead.id)
        assert len(retrieved.dependencies) == 50

    @pytest.mark.asyncio
    async def test_bead_with_nested_metadata(self, bead_store: BeadStore):
        """Test bead with deeply nested metadata."""
        nested_data = {
            "level1": {
                "level2": {
                    "level3": {
                        "level4": ["a", "b", "c"],
                    },
                },
            },
        }
        bead = Bead.create(
            bead_type=BeadType.TASK,
            title="Nested Metadata Task",
            metadata=nested_data,
        )
        await bead_store.create(bead)

        retrieved = await bead_store.get(bead.id)
        assert retrieved.metadata["level1"]["level2"]["level3"]["level4"] == [
            "a",
            "b",
            "c",
        ]

    def test_bead_type_enum_values(self):
        """Test all BeadType enum values."""
        assert BeadType.ISSUE.value == "issue"
        assert BeadType.TASK.value == "task"
        assert BeadType.EPIC.value == "epic"
        assert BeadType.HOOK.value == "hook"
        assert BeadType.DEBATE_DECISION.value == "debate_decision"

    def test_bead_status_enum_values(self):
        """Test all BeadStatus enum values."""
        assert BeadStatus.PENDING.value == "pending"
        assert BeadStatus.CLAIMED.value == "claimed"
        assert BeadStatus.RUNNING.value == "running"
        assert BeadStatus.COMPLETED.value == "completed"
        assert BeadStatus.FAILED.value == "failed"
        assert BeadStatus.CANCELLED.value == "cancelled"
        assert BeadStatus.BLOCKED.value == "blocked"

    def test_bead_priority_enum_values(self):
        """Test all BeadPriority enum values."""
        assert BeadPriority.LOW.value == 0
        assert BeadPriority.NORMAL.value == 50
        assert BeadPriority.HIGH.value == 75
        assert BeadPriority.URGENT.value == 100

    @pytest.mark.asyncio
    async def test_multiple_status_transitions(self, bead_store: BeadStore, sample_bead: Bead):
        """Test bead through multiple status transitions."""
        await bead_store.create(sample_bead)

        # PENDING -> CLAIMED -> RUNNING -> FAILED -> RUNNING -> COMPLETED
        await bead_store.claim(sample_bead.id, "agent-1")
        await bead_store.update_status(sample_bead.id, BeadStatus.RUNNING)
        await bead_store.update_status(
            sample_bead.id, BeadStatus.FAILED, error_message="First failure"
        )

        bead = await bead_store.get(sample_bead.id)
        assert bead.attempt_count == 2  # RUNNING + FAILED
        assert bead.can_retry() is True

        # Reset status and retry
        bead.status = BeadStatus.PENDING
        await bead_store.update(bead)
        await bead_store.update_status(sample_bead.id, BeadStatus.RUNNING)
        await bead_store.update_status(sample_bead.id, BeadStatus.COMPLETED)

        final_bead = await bead_store.get(sample_bead.id)
        assert final_bead.status == BeadStatus.COMPLETED
        assert final_bead.completed_at is not None
