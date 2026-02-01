"""Tests for Convoy module - grouped work orders for beads."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from aragora.nomic.beads import Bead, BeadPriority, BeadStatus, BeadStore, BeadType
from aragora.nomic.convoys import (
    Convoy,
    ConvoyManager,
    ConvoyPriority,
    ConvoyProgress,
    ConvoyStatus,
    reset_convoy_manager,
)


@pytest.fixture
def tmp_bead_dir(tmp_path: Path) -> Path:
    """Create a temporary directory for bead storage."""
    bead_dir = tmp_path / "beads"
    bead_dir.mkdir()
    return bead_dir


@pytest.fixture
async def bead_store(tmp_bead_dir: Path) -> BeadStore:
    """Create and initialize a bead store for testing."""
    store = BeadStore(tmp_bead_dir, git_enabled=False)
    await store.initialize()
    return store


@pytest.fixture
async def convoy_manager(bead_store: BeadStore, tmp_bead_dir: Path) -> ConvoyManager:
    """Create and initialize a convoy manager for testing."""
    manager = ConvoyManager(bead_store, convoy_dir=tmp_bead_dir)
    await manager.initialize()
    yield manager
    reset_convoy_manager()


class TestConvoyDataclass:
    """Tests for Convoy dataclass."""

    def test_convoy_create(self):
        """Test creating a convoy with default values."""
        convoy = Convoy.create(
            title="Test Convoy",
            bead_ids=["bead-1", "bead-2"],
            description="Test description",
        )
        assert convoy.title == "Test Convoy"
        assert convoy.bead_ids == ["bead-1", "bead-2"]
        assert convoy.description == "Test description"
        assert convoy.status == ConvoyStatus.PENDING
        assert convoy.priority == ConvoyPriority.NORMAL
        assert convoy.assigned_to == []
        assert convoy.dependencies == []
        assert isinstance(convoy.id, str)
        assert convoy.created_at is not None
        assert convoy.updated_at is not None

    def test_convoy_create_with_custom_id(self):
        """Test creating a convoy with a custom ID."""
        convoy = Convoy.create(
            title="Custom ID Convoy",
            bead_ids=["bead-1"],
            convoy_id="custom-convoy-123",
        )
        assert convoy.id == "custom-convoy-123"

    def test_convoy_create_with_priority(self):
        """Test creating a convoy with priority."""
        convoy = Convoy.create(
            title="Urgent Convoy",
            bead_ids=["bead-1"],
            priority=ConvoyPriority.URGENT,
        )
        assert convoy.priority == ConvoyPriority.URGENT

    def test_convoy_create_with_dependencies(self):
        """Test creating a convoy with dependencies."""
        convoy = Convoy.create(
            title="Dependent Convoy",
            bead_ids=["bead-1"],
            dependencies=["convoy-a", "convoy-b"],
        )
        assert convoy.dependencies == ["convoy-a", "convoy-b"]

    def test_convoy_to_dict_and_from_dict(self):
        """Test convoy serialization and deserialization."""
        original = Convoy.create(
            title="Serialization Test",
            bead_ids=["bead-1", "bead-2"],
            description="Test desc",
            priority=ConvoyPriority.HIGH,
            tags=["tag1", "tag2"],
            metadata={"key": "value"},
        )
        data = original.to_dict()
        restored = Convoy.from_dict(data)

        assert restored.id == original.id
        assert restored.title == original.title
        assert restored.bead_ids == original.bead_ids
        assert restored.description == original.description
        assert restored.priority == original.priority
        assert restored.tags == original.tags
        assert restored.metadata == original.metadata
        assert restored.status == original.status

    def test_convoy_can_start_no_dependencies(self):
        """Test convoy can start without dependencies."""
        convoy = Convoy.create(title="No Deps", bead_ids=["bead-1"])
        assert convoy.can_start(set()) is True

    def test_convoy_can_start_with_met_dependencies(self):
        """Test convoy can start when dependencies are met."""
        convoy = Convoy.create(
            title="Has Deps",
            bead_ids=["bead-1"],
            dependencies=["dep-1", "dep-2"],
        )
        assert convoy.can_start({"dep-1", "dep-2", "other"}) is True

    def test_convoy_cannot_start_with_unmet_dependencies(self):
        """Test convoy cannot start when dependencies are not met."""
        convoy = Convoy.create(
            title="Has Deps",
            bead_ids=["bead-1"],
            dependencies=["dep-1", "dep-2"],
        )
        assert convoy.can_start({"dep-1"}) is False

    def test_convoy_protocol_properties(self):
        """Test ConvoyRecord protocol properties."""
        convoy = Convoy.create(
            title="Protocol Test",
            bead_ids=["bead-1"],
            description="Protocol desc",
            metadata={"foo": "bar"},
        )
        convoy.assigned_to = ["agent-1"]
        convoy.error_message = "Test error"

        assert convoy.convoy_id == convoy.id
        assert convoy.convoy_title == "Protocol Test"
        assert convoy.convoy_description == "Protocol desc"
        assert convoy.convoy_bead_ids == ["bead-1"]
        assert convoy.convoy_status_value == "pending"
        assert convoy.convoy_assigned_agents == ["agent-1"]
        assert convoy.convoy_error == "Test error"
        assert convoy.convoy_metadata == {"foo": "bar"}


class TestConvoyProgress:
    """Tests for ConvoyProgress dataclass."""

    def test_progress_is_complete_all_done(self):
        """Test is_complete when all beads are done."""
        progress = ConvoyProgress(
            total_beads=3,
            pending_beads=0,
            running_beads=0,
            completed_beads=3,
            failed_beads=0,
            completion_percentage=100.0,
        )
        assert progress.is_complete is True

    def test_progress_is_complete_some_running(self):
        """Test is_complete when some beads are running."""
        progress = ConvoyProgress(
            total_beads=3,
            pending_beads=0,
            running_beads=1,
            completed_beads=2,
            failed_beads=0,
            completion_percentage=66.7,
        )
        assert progress.is_complete is False

    def test_progress_is_complete_some_pending(self):
        """Test is_complete when some beads are pending."""
        progress = ConvoyProgress(
            total_beads=3,
            pending_beads=1,
            running_beads=0,
            completed_beads=2,
            failed_beads=0,
            completion_percentage=66.7,
        )
        assert progress.is_complete is False


class TestConvoyManager:
    """Tests for ConvoyManager."""

    @pytest.mark.asyncio
    async def test_create_convoy(self, convoy_manager: ConvoyManager, bead_store: BeadStore):
        """Test creating a convoy with existing beads."""
        # Create beads first
        bead1 = Bead.create(BeadType.TASK, title="Task 1", description="First task")
        bead2 = Bead.create(BeadType.TASK, title="Task 2", description="Second task")
        await bead_store.create(bead1)
        await bead_store.create(bead2)

        # Create convoy
        convoy = await convoy_manager.create_convoy(
            title="Test Convoy",
            bead_ids=[bead1.id, bead2.id],
            description="Test description",
            priority=ConvoyPriority.HIGH,
        )

        assert convoy.title == "Test Convoy"
        assert convoy.bead_ids == [bead1.id, bead2.id]
        assert convoy.priority == ConvoyPriority.HIGH
        assert convoy.status == ConvoyStatus.PENDING

    @pytest.mark.asyncio
    async def test_create_convoy_nonexistent_bead_fails(self, convoy_manager: ConvoyManager):
        """Test creating convoy with non-existent bead raises error."""
        with pytest.raises(ValueError, match="Bead .* not found"):
            await convoy_manager.create_convoy(
                title="Bad Convoy",
                bead_ids=["nonexistent-bead"],
            )

    @pytest.mark.asyncio
    async def test_create_convoy_from_subtasks(self, convoy_manager: ConvoyManager):
        """Test creating convoy from subtask definitions."""
        subtasks = [
            {"title": "Subtask 1", "description": "First subtask"},
            {"title": "Subtask 2", "description": "Second subtask", "dependencies": []},
        ]

        convoy = await convoy_manager.create_convoy_from_subtasks(
            title="Subtask Convoy",
            subtasks=subtasks,
            priority=ConvoyPriority.NORMAL,
        )

        assert convoy.title == "Subtask Convoy"
        assert len(convoy.bead_ids) == 2

    @pytest.mark.asyncio
    async def test_get_convoy(self, convoy_manager: ConvoyManager, bead_store: BeadStore):
        """Test retrieving a convoy by ID."""
        bead = Bead.create(BeadType.TASK, title="Task", description="Desc")
        await bead_store.create(bead)

        convoy = await convoy_manager.create_convoy(
            title="Get Test",
            bead_ids=[bead.id],
        )

        retrieved = await convoy_manager.get_convoy(convoy.id)
        assert retrieved is not None
        assert retrieved.id == convoy.id
        assert retrieved.title == "Get Test"

    @pytest.mark.asyncio
    async def test_get_convoy_not_found(self, convoy_manager: ConvoyManager):
        """Test retrieving non-existent convoy returns None."""
        result = await convoy_manager.get_convoy("nonexistent-id")
        assert result is None

    @pytest.mark.asyncio
    async def test_assign_convoy(self, convoy_manager: ConvoyManager, bead_store: BeadStore):
        """Test assigning a convoy to agents."""
        bead = Bead.create(BeadType.TASK, title="Task", description="Desc")
        await bead_store.create(bead)

        convoy = await convoy_manager.create_convoy(
            title="Assignment Test",
            bead_ids=[bead.id],
        )

        result = await convoy_manager.assign_convoy(convoy.id, ["agent-1", "agent-2"])
        assert result is True

        updated = await convoy_manager.get_convoy(convoy.id)
        assert updated.assigned_to == ["agent-1", "agent-2"]
        assert updated.status == ConvoyStatus.ACTIVE
        assert updated.started_at is not None

    @pytest.mark.asyncio
    async def test_assign_convoy_already_active(
        self, convoy_manager: ConvoyManager, bead_store: BeadStore
    ):
        """Test assigning already active convoy returns False."""
        bead = Bead.create(BeadType.TASK, title="Task", description="Desc")
        await bead_store.create(bead)

        convoy = await convoy_manager.create_convoy(
            title="Already Active",
            bead_ids=[bead.id],
        )

        await convoy_manager.assign_convoy(convoy.id, ["agent-1"])
        result = await convoy_manager.assign_convoy(convoy.id, ["agent-2"])
        assert result is False

    @pytest.mark.asyncio
    async def test_assign_convoy_not_found(self, convoy_manager: ConvoyManager):
        """Test assigning non-existent convoy raises error."""
        with pytest.raises(ValueError, match="Convoy .* not found"):
            await convoy_manager.assign_convoy("nonexistent", ["agent-1"])

    @pytest.mark.asyncio
    async def test_get_convoy_progress(
        self, convoy_manager: ConvoyManager, bead_store: BeadStore
    ):
        """Test getting convoy progress."""
        # Create beads with different statuses
        bead1 = Bead.create(BeadType.TASK, title="Task 1", description="Desc")
        bead2 = Bead.create(BeadType.TASK, title="Task 2", description="Desc")
        bead3 = Bead.create(BeadType.TASK, title="Task 3", description="Desc")
        await bead_store.create(bead1)
        await bead_store.create(bead2)
        await bead_store.create(bead3)

        # Update statuses
        bead1.status = BeadStatus.COMPLETED
        bead2.status = BeadStatus.RUNNING
        await bead_store.update(bead1)
        await bead_store.update(bead2)

        convoy = await convoy_manager.create_convoy(
            title="Progress Test",
            bead_ids=[bead1.id, bead2.id, bead3.id],
        )

        progress = await convoy_manager.get_convoy_progress(convoy.id)
        assert progress.total_beads == 3
        assert progress.completed_beads == 1
        assert progress.running_beads == 1
        assert progress.pending_beads == 1
        assert progress.failed_beads == 0
        assert progress.completion_percentage == pytest.approx(33.33, rel=0.1)

    @pytest.mark.asyncio
    async def test_get_convoy_progress_not_found(self, convoy_manager: ConvoyManager):
        """Test getting progress for non-existent convoy raises error."""
        with pytest.raises(ValueError, match="Convoy .* not found"):
            await convoy_manager.get_convoy_progress("nonexistent")

    @pytest.mark.asyncio
    async def test_update_convoy_status_completed(
        self, convoy_manager: ConvoyManager, bead_store: BeadStore
    ):
        """Test updating convoy status when all beads complete."""
        bead = Bead.create(BeadType.TASK, title="Task", description="Desc")
        await bead_store.create(bead)

        convoy = await convoy_manager.create_convoy(
            title="Status Update Test",
            bead_ids=[bead.id],
        )

        # Complete the bead
        bead.status = BeadStatus.COMPLETED
        await bead_store.update(bead)

        new_status = await convoy_manager.update_convoy_status(convoy.id)
        assert new_status == ConvoyStatus.COMPLETED

        updated = await convoy_manager.get_convoy(convoy.id)
        assert updated.completed_at is not None

    @pytest.mark.asyncio
    async def test_update_convoy_status_failed(
        self, convoy_manager: ConvoyManager, bead_store: BeadStore
    ):
        """Test updating convoy status when all beads fail."""
        bead = Bead.create(BeadType.TASK, title="Task", description="Desc")
        await bead_store.create(bead)

        convoy = await convoy_manager.create_convoy(
            title="Failure Test",
            bead_ids=[bead.id],
        )

        # Fail the bead
        bead.status = BeadStatus.FAILED
        await bead_store.update(bead)

        new_status = await convoy_manager.update_convoy_status(convoy.id)
        assert new_status == ConvoyStatus.FAILED

    @pytest.mark.asyncio
    async def test_update_convoy_status_partial(
        self, convoy_manager: ConvoyManager, bead_store: BeadStore
    ):
        """Test updating convoy status when some beads fail."""
        bead1 = Bead.create(BeadType.TASK, title="Task 1", description="Desc")
        bead2 = Bead.create(BeadType.TASK, title="Task 2", description="Desc")
        await bead_store.create(bead1)
        await bead_store.create(bead2)

        convoy = await convoy_manager.create_convoy(
            title="Partial Test",
            bead_ids=[bead1.id, bead2.id],
        )

        # Complete one, fail one
        bead1.status = BeadStatus.COMPLETED
        bead2.status = BeadStatus.FAILED
        await bead_store.update(bead1)
        await bead_store.update(bead2)

        new_status = await convoy_manager.update_convoy_status(convoy.id)
        assert new_status == ConvoyStatus.PARTIAL

    @pytest.mark.asyncio
    async def test_list_convoys(self, convoy_manager: ConvoyManager, bead_store: BeadStore):
        """Test listing convoys."""
        bead = Bead.create(BeadType.TASK, title="Task", description="Desc")
        await bead_store.create(bead)

        await convoy_manager.create_convoy(title="Convoy 1", bead_ids=[bead.id])
        await convoy_manager.create_convoy(title="Convoy 2", bead_ids=[bead.id])

        convoys = await convoy_manager.list_convoys()
        assert len(convoys) == 2

    @pytest.mark.asyncio
    async def test_list_convoys_by_status(
        self, convoy_manager: ConvoyManager, bead_store: BeadStore
    ):
        """Test listing convoys filtered by status."""
        bead = Bead.create(BeadType.TASK, title="Task", description="Desc")
        await bead_store.create(bead)

        convoy1 = await convoy_manager.create_convoy(title="Pending Convoy", bead_ids=[bead.id])
        convoy2 = await convoy_manager.create_convoy(title="Active Convoy", bead_ids=[bead.id])
        await convoy_manager.assign_convoy(convoy2.id, ["agent-1"])

        pending = await convoy_manager.list_convoys(status=ConvoyStatus.PENDING)
        active = await convoy_manager.list_convoys(status=ConvoyStatus.ACTIVE)

        assert len(pending) == 1
        assert pending[0].id == convoy1.id
        assert len(active) == 1
        assert active[0].id == convoy2.id

    @pytest.mark.asyncio
    async def test_list_convoys_by_agent(
        self, convoy_manager: ConvoyManager, bead_store: BeadStore
    ):
        """Test listing convoys filtered by agent."""
        bead = Bead.create(BeadType.TASK, title="Task", description="Desc")
        await bead_store.create(bead)

        convoy1 = await convoy_manager.create_convoy(title="Convoy 1", bead_ids=[bead.id])
        convoy2 = await convoy_manager.create_convoy(title="Convoy 2", bead_ids=[bead.id])
        await convoy_manager.assign_convoy(convoy1.id, ["agent-1"])
        await convoy_manager.assign_convoy(convoy2.id, ["agent-2"])

        agent1_convoys = await convoy_manager.list_convoys(agent_id="agent-1")
        assert len(agent1_convoys) == 1
        assert agent1_convoys[0].id == convoy1.id

    @pytest.mark.asyncio
    async def test_list_pending_runnable(
        self, convoy_manager: ConvoyManager, bead_store: BeadStore
    ):
        """Test listing pending runnable convoys (dependencies met)."""
        bead = Bead.create(BeadType.TASK, title="Task", description="Desc")
        await bead_store.create(bead)

        # Create first convoy (no deps)
        convoy1 = await convoy_manager.create_convoy(
            title="No Deps",
            bead_ids=[bead.id],
            convoy_id="convoy-1",
        )

        # Create second convoy (depends on first)
        convoy2 = await convoy_manager.create_convoy(
            title="Has Deps",
            bead_ids=[bead.id],
            dependencies=["convoy-1"],
        )

        # Only convoy1 should be runnable
        runnable = await convoy_manager.list_pending_runnable()
        assert len(runnable) == 1
        assert runnable[0].id == convoy1.id

        # Complete convoy1 by updating status
        await convoy_manager.update_convoy(convoy1.id, status=ConvoyStatus.COMPLETED)

        # Now convoy2 should also be runnable
        runnable = await convoy_manager.list_pending_runnable()
        assert len(runnable) == 1
        assert runnable[0].id == convoy2.id

    @pytest.mark.asyncio
    async def test_update_convoy(self, convoy_manager: ConvoyManager, bead_store: BeadStore):
        """Test updating convoy fields."""
        bead = Bead.create(BeadType.TASK, title="Task", description="Desc")
        await bead_store.create(bead)

        convoy = await convoy_manager.create_convoy(
            title="Update Test",
            bead_ids=[bead.id],
        )
        original_updated_at = convoy.updated_at

        updated = await convoy_manager.update_convoy(
            convoy.id,
            title="Updated Title",
            description="Updated description",
            metadata_updates={"key": "value"},
            error_message="Some error",
        )

        assert updated.title == "Updated Title"
        assert updated.description == "Updated description"
        assert updated.metadata["key"] == "value"
        assert updated.error_message == "Some error"
        # Updated timestamp should be same or later (can be same on fast systems)
        assert updated.updated_at >= original_updated_at

    @pytest.mark.asyncio
    async def test_get_statistics(self, convoy_manager: ConvoyManager, bead_store: BeadStore):
        """Test getting convoy statistics."""
        bead1 = Bead.create(BeadType.TASK, title="Task 1", description="Desc")
        bead2 = Bead.create(BeadType.TASK, title="Task 2", description="Desc")
        await bead_store.create(bead1)
        await bead_store.create(bead2)

        convoy1 = await convoy_manager.create_convoy(
            title="Convoy 1", bead_ids=[bead1.id, bead2.id]
        )
        await convoy_manager.create_convoy(title="Convoy 2", bead_ids=[bead1.id])
        await convoy_manager.assign_convoy(convoy1.id, ["agent-1"])

        stats = await convoy_manager.get_statistics()
        assert stats["total_convoys"] == 2
        assert stats["total_beads"] == 3
        assert stats["avg_beads_per_convoy"] == 1.5
        assert "pending" in stats["by_status"]
        assert "active" in stats["by_status"]


class TestConvoyPersistence:
    """Tests for convoy persistence across restarts."""

    @pytest.mark.asyncio
    async def test_convoy_persisted_and_loaded(self, bead_store: BeadStore, tmp_bead_dir: Path):
        """Test convoys are persisted and can be loaded."""
        bead = Bead.create(BeadType.TASK, title="Task", description="Desc")
        await bead_store.create(bead)

        # Create convoy
        manager1 = ConvoyManager(bead_store, convoy_dir=tmp_bead_dir)
        await manager1.initialize()

        convoy = await manager1.create_convoy(
            title="Persistent Convoy",
            bead_ids=[bead.id],
            priority=ConvoyPriority.HIGH,
            metadata={"key": "value"},
        )

        # Create new manager (simulates restart)
        manager2 = ConvoyManager(bead_store, convoy_dir=tmp_bead_dir)
        await manager2.initialize()

        loaded = await manager2.get_convoy(convoy.id)
        assert loaded is not None
        assert loaded.title == "Persistent Convoy"
        assert loaded.priority == ConvoyPriority.HIGH
        assert loaded.metadata["key"] == "value"

    @pytest.mark.asyncio
    async def test_empty_store_initialization(self, bead_store: BeadStore, tmp_path: Path):
        """Test initializing with empty storage."""
        manager = ConvoyManager(bead_store, convoy_dir=tmp_path / "empty")
        await manager.initialize()

        convoys = await manager.list_convoys()
        assert len(convoys) == 0
