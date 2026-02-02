"""Tests for the Hook Queue module - GUPP Recovery Pattern.

This module tests the HookEntry, HookQueue, and HookQueueRegistry classes which
implement the GUPP (Guaranteed Unconditional Processing Priority) recovery pattern
for per-agent work queues backed by beads.

The GUPP principle states: "If there is work on your Hook, YOU MUST RUN IT."
This ensures that assigned work is never lost and agents always resume
incomplete work before accepting new tasks.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from aragora.nomic.beads import (
    Bead,
    BeadPriority,
    BeadStatus,
    BeadStore,
    BeadType,
)
from aragora.nomic.hook_queue import (
    HookEntry,
    HookEntryStatus,
    HookQueue,
    HookQueueRegistry,
    get_hook_queue_registry,
    reset_hook_queue_registry,
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
async def hook_queue(bead_store: BeadStore) -> HookQueue:
    """Create and initialize a hook queue for testing."""
    queue = HookQueue(agent_id="test-agent", bead_store=bead_store)
    await queue.initialize()
    return queue


@pytest.fixture
async def sample_bead(bead_store: BeadStore) -> Bead:
    """Create a sample bead and store it."""
    bead = Bead.create(
        bead_type=BeadType.TASK,
        title="Sample Task",
        description="A sample task for testing",
        priority=BeadPriority.NORMAL,
    )
    await bead_store.create(bead)
    return bead


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the singleton registry between tests."""
    reset_hook_queue_registry()
    yield
    reset_hook_queue_registry()


# =============================================================================
# HookEntry Tests
# =============================================================================


class TestHookEntryCreation:
    """Tests for HookEntry creation and basic operations."""

    def test_hook_entry_create_with_defaults(self):
        """Test creating a hook entry with default values."""
        entry = HookEntry.create(
            bead_id="bead-123",
            agent_id="agent-456",
        )

        assert entry.id is not None
        assert len(entry.id) == 12
        assert entry.bead_id == "bead-123"
        assert entry.agent_id == "agent-456"
        assert entry.priority == 50
        assert entry.status == HookEntryStatus.QUEUED
        assert entry.created_at is not None
        assert entry.updated_at is not None
        assert entry.started_at is None
        assert entry.completed_at is None
        assert entry.attempt_count == 0
        assert entry.max_attempts == 3
        assert entry.error_message is None
        assert entry.metadata == {}

    def test_hook_entry_create_with_priority(self):
        """Test creating a hook entry with custom priority."""
        entry = HookEntry.create(
            bead_id="bead-123",
            agent_id="agent-456",
            priority=100,
        )

        assert entry.priority == 100

    def test_hook_entry_create_with_max_attempts(self):
        """Test creating a hook entry with custom max attempts."""
        entry = HookEntry.create(
            bead_id="bead-123",
            agent_id="agent-456",
            max_attempts=5,
        )

        assert entry.max_attempts == 5


class TestHookEntrySerialization:
    """Tests for HookEntry serialization and deserialization."""

    def test_hook_entry_to_dict(self):
        """Test serializing a hook entry to dictionary."""
        entry = HookEntry.create(
            bead_id="bead-123",
            agent_id="agent-456",
            priority=75,
        )
        entry.metadata = {"key": "value"}

        data = entry.to_dict()

        assert data["id"] == entry.id
        assert data["bead_id"] == "bead-123"
        assert data["agent_id"] == "agent-456"
        assert data["priority"] == 75
        assert data["status"] == "queued"
        assert "created_at" in data
        assert "updated_at" in data
        assert data["started_at"] is None
        assert data["completed_at"] is None
        assert data["attempt_count"] == 0
        assert data["max_attempts"] == 3
        assert data["error_message"] is None
        assert data["metadata"] == {"key": "value"}

    def test_hook_entry_to_dict_with_timestamps(self):
        """Test serializing a hook entry with all timestamps."""
        entry = HookEntry.create(
            bead_id="bead-123",
            agent_id="agent-456",
        )
        entry.started_at = datetime.now(timezone.utc)
        entry.completed_at = datetime.now(timezone.utc)

        data = entry.to_dict()

        assert data["started_at"] is not None
        assert data["completed_at"] is not None
        assert isinstance(data["started_at"], str)
        assert isinstance(data["completed_at"], str)

    def test_hook_entry_from_dict(self):
        """Test deserializing a hook entry from dictionary."""
        entry = HookEntry.create(
            bead_id="bead-123",
            agent_id="agent-456",
            priority=75,
        )
        entry.metadata = {"key": "value"}

        data = entry.to_dict()
        restored = HookEntry.from_dict(data)

        assert restored.id == entry.id
        assert restored.bead_id == entry.bead_id
        assert restored.agent_id == entry.agent_id
        assert restored.priority == entry.priority
        assert restored.status == entry.status
        assert restored.metadata == entry.metadata

    def test_hook_entry_roundtrip_serialization(self):
        """Test that hook entry survives roundtrip serialization."""
        entry = HookEntry.create(
            bead_id="bead-123",
            agent_id="agent-456",
            priority=100,
            max_attempts=5,
        )
        entry.status = HookEntryStatus.PROCESSING
        entry.started_at = datetime.now(timezone.utc)
        entry.attempt_count = 2
        entry.error_message = "Test error"
        entry.metadata = {"source": "test"}

        data = entry.to_dict()
        restored = HookEntry.from_dict(data)

        assert restored.id == entry.id
        assert restored.bead_id == entry.bead_id
        assert restored.agent_id == entry.agent_id
        assert restored.priority == entry.priority
        assert restored.status == entry.status
        assert restored.attempt_count == entry.attempt_count
        assert restored.max_attempts == entry.max_attempts
        assert restored.error_message == entry.error_message
        assert restored.metadata == entry.metadata

    def test_hook_entry_from_dict_with_optional_fields(self):
        """Test deserializing a hook entry with minimal required fields."""
        now = datetime.now(timezone.utc)
        data = {
            "id": "entry-123",
            "bead_id": "bead-456",
            "agent_id": "agent-789",
            "priority": 50,
            "status": "queued",
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
        }

        entry = HookEntry.from_dict(data)

        assert entry.id == "entry-123"
        assert entry.attempt_count == 0
        assert entry.max_attempts == 3
        assert entry.metadata == {}


class TestHookEntryRetry:
    """Tests for HookEntry retry logic."""

    def test_can_retry_under_max(self):
        """Test can_retry returns True when under max attempts."""
        entry = HookEntry.create(
            bead_id="bead-123",
            agent_id="agent-456",
            max_attempts=3,
        )
        entry.attempt_count = 1

        assert entry.can_retry() is True

    def test_can_retry_at_max(self):
        """Test can_retry returns False when at max attempts."""
        entry = HookEntry.create(
            bead_id="bead-123",
            agent_id="agent-456",
            max_attempts=3,
        )
        entry.attempt_count = 3

        assert entry.can_retry() is False

    def test_can_retry_above_max(self):
        """Test can_retry returns False when above max attempts."""
        entry = HookEntry.create(
            bead_id="bead-123",
            agent_id="agent-456",
            max_attempts=3,
        )
        entry.attempt_count = 5

        assert entry.can_retry() is False

    def test_can_retry_zero_max_attempts(self):
        """Test can_retry with zero max attempts."""
        entry = HookEntry.create(
            bead_id="bead-123",
            agent_id="agent-456",
            max_attempts=0,
        )
        entry.attempt_count = 0

        assert entry.can_retry() is False


class TestHookEntryStatusEnum:
    """Tests for HookEntryStatus enum."""

    def test_hook_entry_status_values(self):
        """Test all HookEntryStatus enum values."""
        assert HookEntryStatus.QUEUED.value == "queued"
        assert HookEntryStatus.PROCESSING.value == "processing"
        assert HookEntryStatus.COMPLETED.value == "completed"
        assert HookEntryStatus.FAILED.value == "failed"
        assert HookEntryStatus.SKIPPED.value == "skipped"


# =============================================================================
# HookQueue Initialization Tests
# =============================================================================


class TestHookQueueInitialization:
    """Tests for HookQueue initialization."""

    @pytest.mark.asyncio
    async def test_queue_initialization(self, bead_store: BeadStore):
        """Test basic queue initialization."""
        queue = HookQueue(agent_id="test-agent", bead_store=bead_store)
        await queue.initialize()

        assert queue._initialized is True
        assert queue.agent_id == "test-agent"
        assert queue.hooks_dir.exists()

    @pytest.mark.asyncio
    async def test_queue_double_initialization(self, bead_store: BeadStore):
        """Test that double initialization is safe."""
        queue = HookQueue(agent_id="test-agent", bead_store=bead_store)
        await queue.initialize()
        await queue.initialize()  # Should not raise

        assert queue._initialized is True

    @pytest.mark.asyncio
    async def test_queue_creates_hooks_directory(self, bead_store: BeadStore):
        """Test queue creates hooks directory if not exists."""
        queue = HookQueue(agent_id="test-agent", bead_store=bead_store)
        await queue.initialize()

        assert queue.hooks_dir.exists()
        assert queue.hooks_dir == bead_store.bead_dir / "hooks"

    @pytest.mark.asyncio
    async def test_queue_custom_hooks_dir(self, bead_store: BeadStore, tmp_path: Path):
        """Test queue with custom hooks directory."""
        custom_dir = tmp_path / "custom_hooks"
        queue = HookQueue(
            agent_id="test-agent",
            bead_store=bead_store,
            hooks_dir=custom_dir,
        )
        await queue.initialize()

        assert queue.hooks_dir == custom_dir
        assert custom_dir.exists()


# =============================================================================
# HookQueue Enqueue Operations Tests
# =============================================================================


class TestHookQueueEnqueue:
    """Tests for HookQueue push (enqueue) operations."""

    @pytest.mark.asyncio
    async def test_push_bead(self, hook_queue: HookQueue, sample_bead: Bead):
        """Test pushing a bead to the queue."""
        entry = await hook_queue.push(sample_bead.id)

        assert entry is not None
        assert entry.bead_id == sample_bead.id
        assert entry.agent_id == "test-agent"
        assert entry.status == HookEntryStatus.QUEUED
        assert entry.priority == 50

    @pytest.mark.asyncio
    async def test_push_with_priority(self, hook_queue: HookQueue, sample_bead: Bead):
        """Test pushing a bead with custom priority."""
        entry = await hook_queue.push(sample_bead.id, priority=100)

        assert entry.priority == 100

    @pytest.mark.asyncio
    async def test_push_with_max_attempts(self, hook_queue: HookQueue, sample_bead: Bead):
        """Test pushing a bead with custom max attempts."""
        entry = await hook_queue.push(sample_bead.id, max_attempts=5)

        assert entry.max_attempts == 5

    @pytest.mark.asyncio
    async def test_push_nonexistent_bead_raises(self, hook_queue: HookQueue):
        """Test pushing a nonexistent bead raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            await hook_queue.push("nonexistent-bead-id")

    @pytest.mark.asyncio
    async def test_push_duplicate_bead_returns_existing(
        self, hook_queue: HookQueue, sample_bead: Bead
    ):
        """Test pushing a duplicate bead returns the existing entry."""
        entry1 = await hook_queue.push(sample_bead.id, priority=50)
        entry2 = await hook_queue.push(sample_bead.id, priority=100)

        assert entry1.id == entry2.id
        # Priority should be unchanged
        assert entry2.priority == 50

    @pytest.mark.asyncio
    async def test_push_multiple_beads(self, hook_queue: HookQueue, bead_store: BeadStore):
        """Test pushing multiple beads to the queue."""
        beads = []
        for i in range(5):
            bead = Bead.create(bead_type=BeadType.TASK, title=f"Task {i}")
            await bead_store.create(bead)
            beads.append(bead)

        for bead in beads:
            await hook_queue.push(bead.id)

        stats = await hook_queue.get_statistics()
        assert stats["total_entries"] == 5
        assert stats["pending"] == 5


# =============================================================================
# HookQueue Dequeue Operations Tests
# =============================================================================


class TestHookQueueDequeue:
    """Tests for HookQueue pop (dequeue) operations."""

    @pytest.mark.asyncio
    async def test_pop_bead(self, hook_queue: HookQueue, sample_bead: Bead):
        """Test popping a bead from the queue."""
        await hook_queue.push(sample_bead.id)

        bead = await hook_queue.pop()

        assert bead is not None
        assert bead.id == sample_bead.id

    @pytest.mark.asyncio
    async def test_pop_empty_queue(self, hook_queue: HookQueue):
        """Test popping from an empty queue returns None."""
        bead = await hook_queue.pop()

        assert bead is None

    @pytest.mark.asyncio
    async def test_pop_marks_entry_as_processing(self, hook_queue: HookQueue, sample_bead: Bead):
        """Test pop marks the entry as processing."""
        entry = await hook_queue.push(sample_bead.id)

        await hook_queue.pop()

        # Check entry status in internal state
        assert hook_queue._entries[entry.id].status == HookEntryStatus.PROCESSING

    @pytest.mark.asyncio
    async def test_pop_increments_attempt_count(self, hook_queue: HookQueue, sample_bead: Bead):
        """Test pop increments the attempt count."""
        entry = await hook_queue.push(sample_bead.id)

        await hook_queue.pop()

        assert hook_queue._entries[entry.id].attempt_count == 1

    @pytest.mark.asyncio
    async def test_pop_sets_started_at(self, hook_queue: HookQueue, sample_bead: Bead):
        """Test pop sets started_at timestamp."""
        entry = await hook_queue.push(sample_bead.id)

        await hook_queue.pop()

        assert hook_queue._entries[entry.id].started_at is not None

    @pytest.mark.asyncio
    async def test_pop_claims_pending_bead(
        self, hook_queue: HookQueue, sample_bead: Bead, bead_store: BeadStore
    ):
        """Test pop claims the bead in the bead store."""
        await hook_queue.push(sample_bead.id)

        await hook_queue.pop()

        bead = await bead_store.get(sample_bead.id)
        assert bead.status == BeadStatus.RUNNING
        assert bead.claimed_by == "test-agent"


# =============================================================================
# HookQueue Priority Handling Tests
# =============================================================================


class TestHookQueuePriority:
    """Tests for HookQueue priority handling."""

    @pytest.mark.asyncio
    async def test_pop_returns_highest_priority(self, hook_queue: HookQueue, bead_store: BeadStore):
        """Test pop returns the highest priority bead."""
        low_priority = Bead.create(bead_type=BeadType.TASK, title="Low Priority")
        high_priority = Bead.create(bead_type=BeadType.TASK, title="High Priority")
        medium_priority = Bead.create(bead_type=BeadType.TASK, title="Medium Priority")

        await bead_store.create(low_priority)
        await bead_store.create(high_priority)
        await bead_store.create(medium_priority)

        await hook_queue.push(low_priority.id, priority=25)
        await hook_queue.push(high_priority.id, priority=100)
        await hook_queue.push(medium_priority.id, priority=50)

        bead = await hook_queue.pop()
        assert bead.id == high_priority.id

    @pytest.mark.asyncio
    async def test_pop_order_by_priority(self, hook_queue: HookQueue, bead_store: BeadStore):
        """Test beads are popped in priority order."""
        beads = []
        priorities = [25, 100, 50, 75, 10]
        for i, priority in enumerate(priorities):
            bead = Bead.create(bead_type=BeadType.TASK, title=f"Task {priority}")
            await bead_store.create(bead)
            beads.append((bead, priority))
            await hook_queue.push(bead.id, priority=priority)

        popped_priorities = []
        for _ in range(5):
            bead = await hook_queue.pop()
            # Find the priority for this bead
            for b, p in beads:
                if b.id == bead.id:
                    popped_priorities.append(p)
                    break
            # Complete the entry to allow next pop
            await hook_queue.complete(bead.id)

        assert popped_priorities == [100, 75, 50, 25, 10]

    @pytest.mark.asyncio
    async def test_peek_returns_highest_priority(
        self, hook_queue: HookQueue, bead_store: BeadStore
    ):
        """Test peek returns the highest priority bead without removing it."""
        low_priority = Bead.create(bead_type=BeadType.TASK, title="Low Priority")
        high_priority = Bead.create(bead_type=BeadType.TASK, title="High Priority")

        await bead_store.create(low_priority)
        await bead_store.create(high_priority)

        await hook_queue.push(low_priority.id, priority=25)
        await hook_queue.push(high_priority.id, priority=100)

        bead = await hook_queue.peek()
        assert bead.id == high_priority.id

        # Peek again - should return same bead
        bead2 = await hook_queue.peek()
        assert bead2.id == high_priority.id

    @pytest.mark.asyncio
    async def test_peek_empty_queue(self, hook_queue: HookQueue):
        """Test peek on empty queue returns None."""
        bead = await hook_queue.peek()

        assert bead is None


# =============================================================================
# HookQueue Complete and Fail Tests
# =============================================================================


class TestHookQueueComplete:
    """Tests for HookQueue complete operation."""

    @pytest.mark.asyncio
    async def test_complete_bead(self, hook_queue: HookQueue, sample_bead: Bead):
        """Test completing a bead in the queue."""
        entry = await hook_queue.push(sample_bead.id)
        await hook_queue.pop()

        await hook_queue.complete(sample_bead.id)

        assert hook_queue._entries[entry.id].status == HookEntryStatus.COMPLETED
        assert hook_queue._entries[entry.id].completed_at is not None

    @pytest.mark.asyncio
    async def test_complete_updates_bead_status(
        self, hook_queue: HookQueue, sample_bead: Bead, bead_store: BeadStore
    ):
        """Test complete updates the bead status in the store."""
        await hook_queue.push(sample_bead.id)
        await hook_queue.pop()

        await hook_queue.complete(sample_bead.id)

        bead = await bead_store.get(sample_bead.id)
        assert bead.status == BeadStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_complete_nonexistent_bead(self, hook_queue: HookQueue):
        """Test completing a nonexistent bead is a no-op."""
        # Should not raise
        await hook_queue.complete("nonexistent-bead-id")


class TestHookQueueFail:
    """Tests for HookQueue fail operation."""

    @pytest.mark.asyncio
    async def test_fail_bead_can_retry(self, hook_queue: HookQueue, sample_bead: Bead):
        """Test failing a bead that can be retried."""
        await hook_queue.push(sample_bead.id, max_attempts=3)
        await hook_queue.pop()

        can_retry = await hook_queue.fail(sample_bead.id, "Test error")

        assert can_retry is True

    @pytest.mark.asyncio
    async def test_fail_bead_requeued(self, hook_queue: HookQueue, sample_bead: Bead):
        """Test failed bead is requeued for retry."""
        entry = await hook_queue.push(sample_bead.id, max_attempts=3)
        await hook_queue.pop()

        await hook_queue.fail(sample_bead.id, "Test error")

        assert hook_queue._entries[entry.id].status == HookEntryStatus.QUEUED
        assert hook_queue._entries[entry.id].error_message == "Test error"

    @pytest.mark.asyncio
    async def test_fail_bead_max_attempts_reached(self, hook_queue: HookQueue, sample_bead: Bead):
        """Test failing a bead at max attempts."""
        entry = await hook_queue.push(sample_bead.id, max_attempts=1)
        await hook_queue.pop()

        can_retry = await hook_queue.fail(sample_bead.id, "Final error")

        assert can_retry is False
        assert hook_queue._entries[entry.id].status == HookEntryStatus.FAILED
        assert hook_queue._entries[entry.id].completed_at is not None

    @pytest.mark.asyncio
    async def test_fail_updates_bead_status_on_max_attempts(
        self, hook_queue: HookQueue, sample_bead: Bead, bead_store: BeadStore
    ):
        """Test fail updates bead status when max attempts reached."""
        await hook_queue.push(sample_bead.id, max_attempts=1)
        await hook_queue.pop()

        await hook_queue.fail(sample_bead.id, "Final error")

        bead = await bead_store.get(sample_bead.id)
        assert bead.status == BeadStatus.FAILED
        assert bead.error_message == "Final error"

    @pytest.mark.asyncio
    async def test_fail_nonexistent_bead(self, hook_queue: HookQueue):
        """Test failing a nonexistent bead returns False."""
        can_retry = await hook_queue.fail("nonexistent-bead-id", "Error")

        assert can_retry is False


# =============================================================================
# HookQueue Has Work and Statistics Tests
# =============================================================================


class TestHookQueueHasWork:
    """Tests for HookQueue has_work method."""

    @pytest.mark.asyncio
    async def test_has_work_empty_queue(self, hook_queue: HookQueue):
        """Test has_work returns False for empty queue."""
        result = await hook_queue.has_work()

        assert result is False

    @pytest.mark.asyncio
    async def test_has_work_with_queued_entries(self, hook_queue: HookQueue, sample_bead: Bead):
        """Test has_work returns True when there are queued entries."""
        await hook_queue.push(sample_bead.id)

        result = await hook_queue.has_work()

        assert result is True

    @pytest.mark.asyncio
    async def test_has_work_no_queued_only_processing(
        self, hook_queue: HookQueue, sample_bead: Bead
    ):
        """Test has_work returns False when only processing entries exist."""
        await hook_queue.push(sample_bead.id)
        await hook_queue.pop()

        result = await hook_queue.has_work()

        assert result is False


class TestHookQueueStatistics:
    """Tests for HookQueue statistics."""

    @pytest.mark.asyncio
    async def test_get_statistics_empty(self, hook_queue: HookQueue):
        """Test statistics for empty queue."""
        stats = await hook_queue.get_statistics()

        assert stats["agent_id"] == "test-agent"
        assert stats["total_entries"] == 0
        assert stats["pending"] == 0
        assert stats["processing"] == 0

    @pytest.mark.asyncio
    async def test_get_statistics_with_entries(self, hook_queue: HookQueue, bead_store: BeadStore):
        """Test statistics with various entry states."""
        beads = []
        for i in range(4):
            bead = Bead.create(bead_type=BeadType.TASK, title=f"Task {i}")
            await bead_store.create(bead)
            beads.append(bead)
            await hook_queue.push(bead.id)

        # Pop one to make it processing
        await hook_queue.pop()

        # Complete one
        await hook_queue.complete(beads[1].id)

        stats = await hook_queue.get_statistics()

        assert stats["total_entries"] == 4
        assert stats["pending"] == 2
        assert stats["processing"] == 1
        assert stats["by_status"]["completed"] == 1


# =============================================================================
# HookQueue Recovery Tests (GUPP)
# =============================================================================


class TestHookQueueRecovery:
    """Tests for GUPP recovery pattern."""

    @pytest.mark.asyncio
    async def test_recover_on_startup_empty(self, hook_queue: HookQueue):
        """Test recovery on startup with empty queue."""
        beads = await hook_queue.recover_on_startup()

        assert beads == []

    @pytest.mark.asyncio
    async def test_recover_on_startup_with_queued_entries(
        self, hook_queue: HookQueue, bead_store: BeadStore
    ):
        """Test recovery returns queued beads."""
        beads_to_queue = []
        for i in range(3):
            bead = Bead.create(bead_type=BeadType.TASK, title=f"Task {i}")
            await bead_store.create(bead)
            beads_to_queue.append(bead)
            await hook_queue.push(bead.id)

        recovered = await hook_queue.recover_on_startup()

        assert len(recovered) == 3

    @pytest.mark.asyncio
    async def test_recover_resets_processing_to_queued(
        self, hook_queue: HookQueue, sample_bead: Bead
    ):
        """Test recovery resets PROCESSING entries to QUEUED."""
        entry = await hook_queue.push(sample_bead.id)
        await hook_queue.pop()  # Marks as PROCESSING

        # Simulate restart - create new queue pointing to same files
        new_queue = HookQueue(
            agent_id="test-agent",
            bead_store=hook_queue.bead_store,
            hooks_dir=hook_queue.hooks_dir,
        )

        recovered = await new_queue.recover_on_startup()

        assert len(recovered) == 1
        assert new_queue._entries[entry.id].status == HookEntryStatus.QUEUED
        assert new_queue._entries[entry.id].started_at is None

    @pytest.mark.asyncio
    async def test_recover_respects_priority_order(
        self, hook_queue: HookQueue, bead_store: BeadStore
    ):
        """Test recovery returns beads in priority order."""
        low = Bead.create(bead_type=BeadType.TASK, title="Low")
        high = Bead.create(bead_type=BeadType.TASK, title="High")
        medium = Bead.create(bead_type=BeadType.TASK, title="Medium")

        await bead_store.create(low)
        await bead_store.create(high)
        await bead_store.create(medium)

        await hook_queue.push(low.id, priority=25)
        await hook_queue.push(high.id, priority=100)
        await hook_queue.push(medium.id, priority=50)

        recovered = await hook_queue.recover_on_startup()

        assert recovered[0].id == high.id
        assert recovered[1].id == medium.id
        assert recovered[2].id == low.id

    @pytest.mark.asyncio
    async def test_recover_excludes_terminal_beads(
        self, hook_queue: HookQueue, bead_store: BeadStore
    ):
        """Test recovery excludes beads in terminal state."""
        active = Bead.create(bead_type=BeadType.TASK, title="Active")
        completed = Bead.create(bead_type=BeadType.TASK, title="Completed")

        await bead_store.create(active)
        await bead_store.create(completed)

        await hook_queue.push(active.id)
        await hook_queue.push(completed.id)

        # Mark one bead as completed in the bead store
        await bead_store.update_status(completed.id, BeadStatus.COMPLETED)

        recovered = await hook_queue.recover_on_startup()

        assert len(recovered) == 1
        assert recovered[0].id == active.id

    @pytest.mark.asyncio
    async def test_recover_handles_missing_beads(
        self, hook_queue: HookQueue, bead_store: BeadStore
    ):
        """Test recovery handles beads that no longer exist."""
        bead = Bead.create(bead_type=BeadType.TASK, title="Existing")
        await bead_store.create(bead)
        await hook_queue.push(bead.id)

        # Add an entry for a non-existent bead (simulating deletion)
        fake_entry = HookEntry.create(
            bead_id="deleted-bead-id",
            agent_id="test-agent",
        )
        hook_queue._entries[fake_entry.id] = fake_entry
        await hook_queue._save_entries()

        recovered = await hook_queue.recover_on_startup()

        # Only the existing bead should be recovered
        assert len(recovered) == 1
        assert recovered[0].id == bead.id


# =============================================================================
# HookQueue Persistence Tests
# =============================================================================


class TestHookQueuePersistence:
    """Tests for HookQueue persistence across restarts."""

    @pytest.mark.asyncio
    async def test_entries_persisted_to_file(self, hook_queue: HookQueue, sample_bead: Bead):
        """Test entries are persisted to JSONL file."""
        await hook_queue.push(sample_bead.id)

        assert hook_queue.hook_file.exists()
        with open(hook_queue.hook_file) as f:
            lines = f.readlines()
            assert len(lines) == 1
            data = json.loads(lines[0])
            assert data["bead_id"] == sample_bead.id

    @pytest.mark.asyncio
    async def test_entries_loaded_on_initialization(self, bead_store: BeadStore, sample_bead: Bead):
        """Test entries are loaded when queue initializes."""
        # Create first queue and add entry
        queue1 = HookQueue(agent_id="test-agent", bead_store=bead_store)
        await queue1.initialize()
        await queue1.push(sample_bead.id)

        # Create second queue (simulates restart)
        queue2 = HookQueue(agent_id="test-agent", bead_store=bead_store)
        await queue2.initialize()

        assert len(queue2._entries) == 1
        assert any(e.bead_id == sample_bead.id for e in queue2._entries.values())

    @pytest.mark.asyncio
    async def test_handles_corrupted_entries(self, bead_store: BeadStore, tmp_path: Path):
        """Test queue gracefully handles corrupted entries."""
        hooks_dir = tmp_path / "hooks"
        hooks_dir.mkdir()
        hook_file = hooks_dir / "test-agent.jsonl"

        # Write valid and invalid entries
        entry = HookEntry.create(bead_id="bead-123", agent_id="test-agent")
        with open(hook_file, "w") as f:
            f.write(json.dumps(entry.to_dict()) + "\n")
            f.write("invalid json line\n")
            f.write("{}\n")  # Missing required fields

        queue = HookQueue(
            agent_id="test-agent",
            bead_store=bead_store,
            hooks_dir=hooks_dir,
        )
        await queue.initialize()

        # Should load valid entry despite corrupted lines
        assert len(queue._entries) == 1


# =============================================================================
# HookQueue Clear Completed Tests
# =============================================================================


class TestHookQueueClearCompleted:
    """Tests for HookQueue clear_completed method."""

    @pytest.mark.asyncio
    async def test_clear_completed_entries(self, hook_queue: HookQueue, bead_store: BeadStore):
        """Test clearing completed entries."""
        beads = []
        for i in range(3):
            bead = Bead.create(bead_type=BeadType.TASK, title=f"Task {i}")
            await bead_store.create(bead)
            beads.append(bead)
            await hook_queue.push(bead.id)

        # Complete one, fail one
        await hook_queue.pop()
        await hook_queue.complete(beads[0].id)

        await hook_queue.pop()
        entry_id = [e.id for e in hook_queue._entries.values() if e.bead_id == beads[1].id][0]
        hook_queue._entries[entry_id].max_attempts = 1
        await hook_queue.fail(beads[1].id, "Error")

        cleared = await hook_queue.clear_completed()

        assert cleared == 2  # Completed + Failed
        assert len(hook_queue._entries) == 1

    @pytest.mark.asyncio
    async def test_clear_completed_clears_skipped(self, hook_queue: HookQueue, sample_bead: Bead):
        """Test clear_completed also clears skipped entries."""
        entry = await hook_queue.push(sample_bead.id)
        hook_queue._entries[entry.id].status = HookEntryStatus.SKIPPED
        await hook_queue._save_entries()

        cleared = await hook_queue.clear_completed()

        assert cleared == 1
        assert len(hook_queue._entries) == 0

    @pytest.mark.asyncio
    async def test_clear_completed_empty_queue(self, hook_queue: HookQueue):
        """Test clear_completed on empty queue."""
        cleared = await hook_queue.clear_completed()

        assert cleared == 0


# =============================================================================
# HookQueue Pop Skips Missing Beads Tests
# =============================================================================


class TestHookQueuePopSkipsMissing:
    """Tests for HookQueue pop skipping missing beads.

    Note: The production code has a known issue where pop() recursively calls
    itself within the lock when a bead is missing. Since asyncio.Lock is not
    re-entrant, this can cause deadlock. These tests verify the skip logic
    works by manually simulating the skipped state.
    """

    @pytest.mark.asyncio
    async def test_pop_skips_already_skipped_entries(self, bead_store: BeadStore):
        """Test pop ignores entries that are already marked as SKIPPED."""
        # Create a fresh queue
        queue = HookQueue(agent_id="skip-test-agent", bead_store=bead_store)
        await queue.initialize()

        # Create beads
        bead1 = Bead.create(bead_type=BeadType.TASK, title="First - already skipped")
        bead2 = Bead.create(bead_type=BeadType.TASK, title="Second - valid")
        await bead_store.create(bead1)
        await bead_store.create(bead2)

        # Push both with bead1 at higher priority
        entry1 = await queue.push(bead1.id, priority=100)
        await queue.push(bead2.id, priority=50)

        # Manually mark entry1 as skipped (simulating what would happen
        # if the bead was deleted and pop tried to process it)
        queue._entries[entry1.id].status = HookEntryStatus.SKIPPED
        await queue._save_entries()

        # Now pop should return bead2 since bead1's entry is skipped
        popped = await queue.pop()

        assert popped is not None
        assert popped.id == bead2.id

    @pytest.mark.asyncio
    async def test_skipped_entries_not_counted_as_pending(self, bead_store: BeadStore):
        """Test that skipped entries are not counted as pending work."""
        queue = HookQueue(agent_id="skip-count-agent", bead_store=bead_store)
        await queue.initialize()

        bead = Bead.create(bead_type=BeadType.TASK, title="Skipped")
        await bead_store.create(bead)

        entry = await queue.push(bead.id)

        # Mark as skipped
        queue._entries[entry.id].status = HookEntryStatus.SKIPPED
        await queue._save_entries()

        # has_work should return False
        assert await queue.has_work() is False

        stats = await queue.get_statistics()
        assert stats["pending"] == 0


# =============================================================================
# HookQueue Concurrent Operations Tests
# =============================================================================


class TestHookQueueConcurrency:
    """Tests for HookQueue concurrent operations."""

    @pytest.mark.asyncio
    async def test_concurrent_push_operations(self, hook_queue: HookQueue, bead_store: BeadStore):
        """Test concurrent push operations are handled safely."""
        beads = []
        for i in range(10):
            bead = Bead.create(bead_type=BeadType.TASK, title=f"Task {i}")
            await bead_store.create(bead)
            beads.append(bead)

        # Push all beads concurrently
        await asyncio.gather(*[hook_queue.push(bead.id) for bead in beads])

        stats = await hook_queue.get_statistics()
        assert stats["total_entries"] == 10

    @pytest.mark.asyncio
    async def test_concurrent_pop_operations(self, hook_queue: HookQueue, bead_store: BeadStore):
        """Test concurrent pop operations don't return same bead twice."""
        beads = []
        for i in range(5):
            bead = Bead.create(bead_type=BeadType.TASK, title=f"Task {i}")
            await bead_store.create(bead)
            beads.append(bead)
            await hook_queue.push(bead.id)

        # Pop all beads concurrently
        results = await asyncio.gather(*[hook_queue.pop() for _ in range(5)])

        # Filter out Nones and check unique IDs
        popped_ids = [b.id for b in results if b is not None]
        assert len(popped_ids) == len(set(popped_ids))


# =============================================================================
# HookQueueRegistry Tests
# =============================================================================


class TestHookQueueRegistry:
    """Tests for HookQueueRegistry."""

    @pytest.mark.asyncio
    async def test_registry_get_queue(self, bead_store: BeadStore):
        """Test getting a queue from the registry."""
        registry = HookQueueRegistry(bead_store)

        queue = await registry.get_queue("agent-1")

        assert queue is not None
        assert queue.agent_id == "agent-1"
        assert queue._initialized is True

    @pytest.mark.asyncio
    async def test_registry_get_same_queue(self, bead_store: BeadStore):
        """Test getting the same queue returns same instance."""
        registry = HookQueueRegistry(bead_store)

        queue1 = await registry.get_queue("agent-1")
        queue2 = await registry.get_queue("agent-1")

        assert queue1 is queue2

    @pytest.mark.asyncio
    async def test_registry_get_different_queues(self, bead_store: BeadStore):
        """Test getting different queues returns different instances."""
        registry = HookQueueRegistry(bead_store)

        queue1 = await registry.get_queue("agent-1")
        queue2 = await registry.get_queue("agent-2")

        assert queue1 is not queue2
        assert queue1.agent_id == "agent-1"
        assert queue2.agent_id == "agent-2"


class TestHookQueueRegistryRecovery:
    """Tests for HookQueueRegistry recovery."""

    @pytest.mark.asyncio
    async def test_recover_all_empty(self, bead_store: BeadStore):
        """Test recover_all with no agents."""
        registry = HookQueueRegistry(bead_store)

        results = await registry.recover_all()

        assert results == {}

    @pytest.mark.asyncio
    async def test_recover_all_with_agents(self, bead_store: BeadStore):
        """Test recover_all returns beads for all agents."""
        registry = HookQueueRegistry(bead_store)

        # Create beads and queues for multiple agents
        for agent_id in ["agent-1", "agent-2"]:
            queue = await registry.get_queue(agent_id)
            for i in range(2):
                bead = Bead.create(bead_type=BeadType.TASK, title=f"{agent_id}-task-{i}")
                await bead_store.create(bead)
                await queue.push(bead.id)

        # Create new registry (simulates restart)
        new_registry = HookQueueRegistry(bead_store)
        results = await new_registry.recover_all()

        assert len(results) == 2
        assert len(results["agent-1"]) == 2
        assert len(results["agent-2"]) == 2

    @pytest.mark.asyncio
    async def test_recover_all_skips_empty_agents(self, bead_store: BeadStore):
        """Test recover_all skips agents with no pending work."""
        registry = HookQueueRegistry(bead_store)

        # Agent 1 has work
        queue1 = await registry.get_queue("agent-1")
        bead1 = Bead.create(bead_type=BeadType.TASK, title="Task 1")
        await bead_store.create(bead1)
        await queue1.push(bead1.id)

        # Agent 2 has completed work only
        queue2 = await registry.get_queue("agent-2")
        bead2 = Bead.create(bead_type=BeadType.TASK, title="Task 2")
        await bead_store.create(bead2)
        await queue2.push(bead2.id)
        await queue2.pop()
        await queue2.complete(bead2.id)

        # Create new registry (simulates restart)
        new_registry = HookQueueRegistry(bead_store)
        results = await new_registry.recover_all()

        assert "agent-1" in results
        assert "agent-2" not in results  # No pending work


class TestHookQueueRegistryStatistics:
    """Tests for HookQueueRegistry statistics."""

    @pytest.mark.asyncio
    async def test_get_all_statistics_empty(self, bead_store: BeadStore):
        """Test get_all_statistics with no queues."""
        registry = HookQueueRegistry(bead_store)

        stats = await registry.get_all_statistics()

        assert stats["total_agents"] == 0
        assert stats["queues"] == {}

    @pytest.mark.asyncio
    async def test_get_all_statistics_with_agents(self, bead_store: BeadStore):
        """Test get_all_statistics with multiple agents."""
        registry = HookQueueRegistry(bead_store)

        for agent_id in ["agent-1", "agent-2"]:
            queue = await registry.get_queue(agent_id)
            bead = Bead.create(bead_type=BeadType.TASK, title=f"{agent_id}-task")
            await bead_store.create(bead)
            await queue.push(bead.id)

        stats = await registry.get_all_statistics()

        assert stats["total_agents"] == 2
        assert "agent-1" in stats["queues"]
        assert "agent-2" in stats["queues"]


# =============================================================================
# Singleton Functions Tests
# =============================================================================


class TestSingletonFunctions:
    """Tests for singleton functions."""

    @pytest.mark.asyncio
    async def test_get_hook_queue_registry(self, bead_store: BeadStore):
        """Test get_hook_queue_registry returns singleton."""
        registry1 = await get_hook_queue_registry(bead_store)
        registry2 = await get_hook_queue_registry(bead_store)

        assert registry1 is registry2

    def test_reset_hook_queue_registry(self):
        """Test reset_hook_queue_registry clears singleton."""
        reset_hook_queue_registry()
        # Should not raise
        reset_hook_queue_registry()


# =============================================================================
# Edge Cases Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_push_already_completed_bead(self, hook_queue: HookQueue, sample_bead: Bead):
        """Test pushing a bead that was already completed creates a new entry."""
        first_entry = await hook_queue.push(sample_bead.id)
        await hook_queue.pop()
        await hook_queue.complete(sample_bead.id)

        # Push the same bead again - should create a NEW entry since
        # the old one is in terminal state (COMPLETED)
        second_entry = await hook_queue.push(sample_bead.id)

        assert second_entry is not None
        # New entry should be QUEUED (fresh entry)
        assert second_entry.status == HookEntryStatus.QUEUED
        # Should be a different entry ID than the original
        assert second_entry.id != first_entry.id
        # Now there should be 2 entries for the same bead
        assert len(hook_queue._entries) == 2

    @pytest.mark.asyncio
    async def test_queue_with_zero_priority_beads(
        self, hook_queue: HookQueue, bead_store: BeadStore
    ):
        """Test queue handles zero priority beads."""
        bead1 = Bead.create(bead_type=BeadType.TASK, title="Zero Priority")
        bead2 = Bead.create(bead_type=BeadType.TASK, title="Normal Priority")

        await bead_store.create(bead1)
        await bead_store.create(bead2)

        await hook_queue.push(bead1.id, priority=0)
        await hook_queue.push(bead2.id, priority=50)

        popped = await hook_queue.pop()
        assert popped.id == bead2.id  # Higher priority first

    @pytest.mark.asyncio
    async def test_queue_with_negative_priority(self, hook_queue: HookQueue, sample_bead: Bead):
        """Test queue handles negative priority."""
        entry = await hook_queue.push(sample_bead.id, priority=-10)

        assert entry.priority == -10

    @pytest.mark.asyncio
    async def test_queue_with_very_high_priority(self, hook_queue: HookQueue, sample_bead: Bead):
        """Test queue handles very high priority."""
        entry = await hook_queue.push(sample_bead.id, priority=10000)

        assert entry.priority == 10000

    @pytest.mark.asyncio
    async def test_multiple_failures_before_final(self, hook_queue: HookQueue, sample_bead: Bead):
        """Test multiple failures before final failure."""
        await hook_queue.push(sample_bead.id, max_attempts=3)

        for i in range(3):
            await hook_queue.pop()
            can_retry = await hook_queue.fail(sample_bead.id, f"Error {i + 1}")

            if i < 2:
                assert can_retry is True
            else:
                assert can_retry is False

        entry = list(hook_queue._entries.values())[0]
        assert entry.status == HookEntryStatus.FAILED
        assert entry.attempt_count == 3

    @pytest.mark.asyncio
    async def test_complete_without_pop(self, hook_queue: HookQueue, sample_bead: Bead):
        """Test completing a bead that was never popped."""
        await hook_queue.push(sample_bead.id)

        # Complete without popping first
        await hook_queue.complete(sample_bead.id)

        entry = list(hook_queue._entries.values())[0]
        assert entry.status == HookEntryStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_queue_agent_id_with_special_chars(self, bead_store: BeadStore):
        """Test queue with special characters in agent ID."""
        queue = HookQueue(
            agent_id="agent-test_123",
            bead_store=bead_store,
        )
        await queue.initialize()

        bead = Bead.create(bead_type=BeadType.TASK, title="Test")
        await bead_store.create(bead)
        await queue.push(bead.id)

        assert queue.hook_file.name == "agent-test_123.jsonl"

    @pytest.mark.asyncio
    async def test_pop_claims_already_claimed_bead(
        self, hook_queue: HookQueue, sample_bead: Bead, bead_store: BeadStore
    ):
        """Test pop handles bead already claimed by same agent."""
        await hook_queue.push(sample_bead.id)

        # Claim the bead first
        await bead_store.claim(sample_bead.id, "test-agent")

        # Pop should still work
        popped = await hook_queue.pop()

        assert popped is not None
        assert popped.status == BeadStatus.RUNNING


# =============================================================================
# Multi-Agent Scenario Tests
# =============================================================================


class TestMultiAgentScenarios:
    """Tests for multi-agent scenarios."""

    @pytest.mark.asyncio
    async def test_different_agents_independent_queues(self, bead_store: BeadStore):
        """Test different agents have independent queues."""
        registry = HookQueueRegistry(bead_store)

        queue1 = await registry.get_queue("agent-1")
        queue2 = await registry.get_queue("agent-2")

        bead1 = Bead.create(bead_type=BeadType.TASK, title="Task 1")
        bead2 = Bead.create(bead_type=BeadType.TASK, title="Task 2")
        await bead_store.create(bead1)
        await bead_store.create(bead2)

        await queue1.push(bead1.id)
        await queue2.push(bead2.id)

        stats1 = await queue1.get_statistics()
        stats2 = await queue2.get_statistics()

        assert stats1["total_entries"] == 1
        assert stats2["total_entries"] == 1
        assert stats1["agent_id"] == "agent-1"
        assert stats2["agent_id"] == "agent-2"

    @pytest.mark.asyncio
    async def test_same_bead_different_agents(self, bead_store: BeadStore):
        """Test same bead can be in different agent queues."""
        registry = HookQueueRegistry(bead_store)

        queue1 = await registry.get_queue("agent-1")
        queue2 = await registry.get_queue("agent-2")

        bead = Bead.create(bead_type=BeadType.TASK, title="Shared Task")
        await bead_store.create(bead)

        # Both agents can have the same bead in their queue
        entry1 = await queue1.push(bead.id)
        entry2 = await queue2.push(bead.id)

        assert entry1.agent_id == "agent-1"
        assert entry2.agent_id == "agent-2"
        assert entry1.bead_id == entry2.bead_id

    @pytest.mark.asyncio
    async def test_agent_recovery_does_not_affect_others(self, bead_store: BeadStore):
        """Test one agent's recovery doesn't affect other agents."""
        registry = HookQueueRegistry(bead_store)

        queue1 = await registry.get_queue("agent-1")
        queue2 = await registry.get_queue("agent-2")

        bead1 = Bead.create(bead_type=BeadType.TASK, title="Task 1")
        bead2 = Bead.create(bead_type=BeadType.TASK, title="Task 2")
        await bead_store.create(bead1)
        await bead_store.create(bead2)

        await queue1.push(bead1.id)
        await queue2.push(bead2.id)

        # Put agent-1's task in processing state
        await queue1.pop()

        # Simulate agent-1 restart
        new_queue1 = HookQueue(
            agent_id="agent-1",
            bead_store=bead_store,
            hooks_dir=queue1.hooks_dir,
        )
        await new_queue1.recover_on_startup()

        # Agent-2's queue should be unchanged
        stats2 = await queue2.get_statistics()
        assert stats2["pending"] == 1

    @pytest.mark.asyncio
    async def test_concurrent_multi_agent_operations(self, bead_store: BeadStore):
        """Test concurrent operations across multiple agents."""
        registry = HookQueueRegistry(bead_store)

        # Create beads
        beads = []
        for i in range(20):
            bead = Bead.create(bead_type=BeadType.TASK, title=f"Task {i}")
            await bead_store.create(bead)
            beads.append(bead)

        # Create queues for multiple agents
        queues = [await registry.get_queue(f"agent-{i}") for i in range(4)]

        # Push beads to different queues concurrently
        async def push_beads(queue, bead_subset):
            for bead in bead_subset:
                await queue.push(bead.id)

        await asyncio.gather(
            push_beads(queues[0], beads[0:5]),
            push_beads(queues[1], beads[5:10]),
            push_beads(queues[2], beads[10:15]),
            push_beads(queues[3], beads[15:20]),
        )

        # Verify each queue has correct count
        for queue in queues:
            stats = await queue.get_statistics()
            assert stats["total_entries"] == 5
