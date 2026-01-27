"""Tests for GUPP Hook Recovery integration with debates.

Tests the Gastown-style GUPP (Guaranteed Unconditional Processing Priority)
integration that provides crash recovery for in-progress debates.
"""

import pytest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.core_types import DebateResult
from aragora.debate.protocol import DebateProtocol


class TestDebateProtocolHookConfig:
    """Tests for hook tracking configuration in DebateProtocol."""

    def test_hook_tracking_disabled_by_default(self):
        """Hook tracking should be disabled by default."""
        protocol = DebateProtocol()
        assert protocol.enable_hook_tracking is False

    def test_hook_tracking_can_be_enabled(self):
        """Hook tracking can be enabled."""
        protocol = DebateProtocol(enable_hook_tracking=True)
        assert protocol.enable_hook_tracking is True

    def test_hook_max_recovery_age_default(self):
        """Default max recovery age should be 24 hours."""
        protocol = DebateProtocol()
        assert protocol.hook_max_recovery_age_hours == 24

    def test_hook_tracking_requires_bead_tracking(self):
        """Hook tracking works best with bead tracking enabled."""
        # When both are enabled, full GUPP recovery is possible
        protocol = DebateProtocol(
            enable_bead_tracking=True,
            enable_hook_tracking=True,
        )
        assert protocol.enable_bead_tracking is True
        assert protocol.enable_hook_tracking is True


class TestHookQueueIntegration:
    """Tests for HookQueue integration with debates."""

    @pytest.fixture
    def temp_bead_dir(self, tmp_path):
        """Create a temporary bead directory."""
        return tmp_path / ".beads"

    @pytest.mark.asyncio
    async def test_hook_queue_push_and_pop(self, temp_bead_dir):
        """Should push and pop beads from hook queue."""
        from aragora.nomic.beads import Bead, BeadStore, BeadType
        from aragora.nomic.hook_queue import HookQueue

        # Create bead store and hook queue
        store = BeadStore(bead_dir=temp_bead_dir, git_enabled=False)
        await store.initialize()

        # Create a debate bead
        bead = Bead.create(
            bead_type=BeadType.DEBATE_DECISION,
            title="Test debate",
            description="Test description",
        )
        bead_id = await store.create(bead)

        # Create hook queue for an agent
        hook = HookQueue(agent_id="claude-001", bead_store=store)
        await hook.initialize()

        # Push bead to hook
        entry = await hook.push(bead_id=bead_id, priority=75)
        assert entry is not None
        assert entry.bead_id == bead_id

        # Pop from hook
        popped = await hook.pop()
        assert popped is not None
        assert popped.id == bead_id

    @pytest.mark.asyncio
    async def test_hook_queue_recover_on_startup(self, temp_bead_dir):
        """Should recover pending work on startup."""
        from aragora.nomic.beads import Bead, BeadStore, BeadType
        from aragora.nomic.hook_queue import HookQueue, HookEntryStatus

        # Create bead store and hook queue
        store = BeadStore(bead_dir=temp_bead_dir, git_enabled=False)
        await store.initialize()

        # Create a debate bead
        bead = Bead.create(
            bead_type=BeadType.DEBATE_DECISION,
            title="Pending debate",
            description="Simulating crash recovery",
        )
        bead_id = await store.create(bead)

        # Create hook queue and push work
        hook = HookQueue(agent_id="claude-002", bead_store=store)
        await hook.initialize()
        entry = await hook.push(bead_id=bead_id, priority=75)

        # Simulate crash by marking entry as PROCESSING
        entry.status = HookEntryStatus.PROCESSING
        await hook._save_entries()

        # Create new hook queue (simulating restart)
        hook2 = HookQueue(agent_id="claude-002", bead_store=store)

        # Recover on startup - should reset PROCESSING to QUEUED
        pending = await hook2.recover_on_startup()
        assert len(pending) == 1
        assert pending[0].id == bead_id

    @pytest.mark.asyncio
    async def test_hook_queue_registry_recover_all(self, temp_bead_dir):
        """Should recover pending work for all agents."""
        from aragora.nomic.beads import Bead, BeadStore, BeadType
        from aragora.nomic.hook_queue import HookQueue, HookQueueRegistry

        # Create bead store
        store = BeadStore(bead_dir=temp_bead_dir, git_enabled=False)
        await store.initialize()

        # Create beads for multiple agents
        for agent_id in ["agent-1", "agent-2"]:
            bead = Bead.create(
                bead_type=BeadType.DEBATE_DECISION,
                title=f"Debate for {agent_id}",
                description="Test",
            )
            bead_id = await store.create(bead)

            hook = HookQueue(agent_id=agent_id, bead_store=store)
            await hook.initialize()
            await hook.push(bead_id=bead_id, priority=50)

        # Create registry and recover all
        registry = HookQueueRegistry(store)
        recovered = await registry.recover_all()

        assert len(recovered) == 2
        assert "agent-1" in recovered
        assert "agent-2" in recovered


class TestArenaHookTracking:
    """Tests for Arena hook tracking methods."""

    @pytest.fixture
    def mock_arena(self):
        """Create a mock Arena with hook tracking enabled."""
        from aragora.core_types import Agent, Environment

        arena = MagicMock()
        arena.protocol = DebateProtocol(
            enable_hook_tracking=True,
            enable_bead_tracking=True,
        )
        arena.env = Environment(task="Test debate task")
        arena.env.context = {}
        arena.agents = [
            MagicMock(name="claude"),
            MagicMock(name="gpt4"),
        ]
        arena._bead_store = None
        arena._hook_registry = None
        return arena

    @pytest.mark.asyncio
    async def test_create_pending_debate_bead(self, mock_arena, tmp_path):
        """Should create a pending bead for GUPP tracking."""
        from aragora.debate.orchestrator import Arena
        from aragora.core_types import Environment

        # Create proper agent objects with name attribute
        class SimpleAgent:
            def __init__(self, name: str):
                self.name = name

        # Import the actual method
        arena = Arena.__new__(Arena)
        arena.protocol = mock_arena.protocol
        arena.env = Environment(task="Test debate task")
        arena.env.context = {"bead_dir": str(tmp_path / ".beads")}
        arena.agents = [SimpleAgent("claude"), SimpleAgent("gpt4")]
        arena._bead_store = None

        bead_id = await arena._create_pending_debate_bead(
            debate_id="test-debate-123",
            task="Should we use microservices?",
        )

        assert bead_id is not None

        # Verify bead was created
        from aragora.nomic.beads import BeadStore

        store = arena._bead_store
        bead = await store.get(bead_id)
        assert bead is not None
        assert "[Pending]" in bead.title
        assert "gupp-tracked" in bead.tags

    @pytest.mark.asyncio
    async def test_init_hook_tracking(self, tmp_path):
        """Should push debate work to agent hooks."""
        from aragora.debate.orchestrator import Arena
        from aragora.nomic.beads import Bead, BeadStore, BeadType

        # Create proper agent objects
        class SimpleAgent:
            def __init__(self, name: str):
                self.name = name

        # Create bead store
        store = BeadStore(bead_dir=tmp_path / ".beads", git_enabled=False)
        await store.initialize()

        # Create a debate bead
        bead = Bead.create(
            bead_type=BeadType.DEBATE_DECISION,
            title="Test debate",
            description="Test",
            metadata={"debate_id": "debate-456"},
        )
        bead_id = await store.create(bead)

        # Create arena with minimal setup
        arena = Arena.__new__(Arena)
        arena.protocol = DebateProtocol(enable_hook_tracking=True)
        arena.agents = [SimpleAgent("agent-a"), SimpleAgent("agent-b")]
        arena._bead_store = store
        arena._hook_registry = None

        # Initialize hook tracking
        hook_entries = await arena._init_hook_tracking(
            debate_id="debate-456",
            bead_id=bead_id,
        )

        assert len(hook_entries) == 2
        assert "agent-a" in hook_entries
        assert "agent-b" in hook_entries

    @pytest.mark.asyncio
    async def test_complete_hook_tracking(self, tmp_path):
        """Should complete hooks on debate success."""
        from aragora.debate.orchestrator import Arena
        from aragora.nomic.beads import Bead, BeadStore, BeadType
        from aragora.nomic.hook_queue import HookEntryStatus, HookQueueRegistry

        # Create proper agent objects
        class SimpleAgent:
            def __init__(self, name: str):
                self.name = name

        # Create bead store
        store = BeadStore(bead_dir=tmp_path / ".beads", git_enabled=False)
        await store.initialize()

        # Create a debate bead
        bead = Bead.create(
            bead_type=BeadType.DEBATE_DECISION,
            title="Test debate",
            description="Test",
        )
        bead_id = await store.create(bead)

        # Create arena and push hooks
        arena = Arena.__new__(Arena)
        arena.protocol = DebateProtocol(enable_hook_tracking=True)
        arena.agents = [SimpleAgent("agent-x")]
        arena._bead_store = store
        arena._hook_registry = None

        # Initialize hook tracking
        hook_entries = await arena._init_hook_tracking(
            debate_id="debate-789",
            bead_id=bead_id,
        )

        # Complete hooks
        await arena._complete_hook_tracking(
            bead_id=bead_id,
            hook_entries=hook_entries,
            success=True,
        )

        # Verify hook is completed
        registry = arena._hook_registry
        queue = await registry.get_queue("agent-x")
        stats = await queue.get_statistics()
        assert stats["by_status"].get(HookEntryStatus.COMPLETED.value, 0) == 1


class TestRecoverPendingDebates:
    """Tests for Arena.recover_pending_debates class method."""

    @pytest.mark.asyncio
    async def test_recover_no_pending_debates(self, tmp_path):
        """Should return empty list when no pending debates."""
        from aragora.debate.orchestrator import Arena
        from aragora.nomic.beads import BeadStore

        store = BeadStore(bead_dir=tmp_path / ".beads", git_enabled=False)
        await store.initialize()

        recovered = await Arena.recover_pending_debates(bead_store=store)
        assert recovered == []

    @pytest.mark.asyncio
    async def test_recover_pending_debates(self, tmp_path):
        """Should recover debates with pending hooks."""
        from aragora.debate.orchestrator import Arena
        from aragora.nomic.beads import Bead, BeadStore, BeadType
        from aragora.nomic.hook_queue import HookQueue

        # Create bead store
        store = BeadStore(bead_dir=tmp_path / ".beads", git_enabled=False)
        await store.initialize()

        # Create pending debate bead
        bead = Bead.create(
            bead_type=BeadType.DEBATE_DECISION,
            title="[Pending] Decision: Migration strategy",
            description="In progress",
            tags=["debate", "pending", "gupp-tracked"],
            metadata={
                "debate_id": "recover-test-001",
                "status": "in_progress",
            },
        )
        bead_id = await store.create(bead)

        # Push to agent hooks
        for agent_id in ["recover-agent-1", "recover-agent-2"]:
            hook = HookQueue(agent_id=agent_id, bead_store=store)
            await hook.initialize()
            await hook.push(bead_id=bead_id, priority=75)

        # Recover pending debates
        recovered = await Arena.recover_pending_debates(
            bead_store=store,
            max_age_hours=24,
        )

        assert len(recovered) == 1
        assert recovered[0]["debate_id"] == "recover-test-001"
        assert len(recovered[0]["agents"]) == 2

    @pytest.mark.asyncio
    async def test_recover_skips_completed_beads(self, tmp_path):
        """Should skip completed beads during recovery."""
        from aragora.debate.orchestrator import Arena
        from aragora.nomic.beads import Bead, BeadStatus, BeadStore, BeadType
        from aragora.nomic.hook_queue import HookQueue

        # Create bead store
        store = BeadStore(bead_dir=tmp_path / ".beads", git_enabled=False)
        await store.initialize()

        # Create completed debate bead
        bead = Bead.create(
            bead_type=BeadType.DEBATE_DECISION,
            title="Completed debate",
            description="Done",
            metadata={"debate_id": "completed-001"},
        )
        bead_id = await store.create(bead)
        await store.update_status(bead_id, BeadStatus.COMPLETED)

        # Push to agent hook (simulating incomplete cleanup)
        hook = HookQueue(agent_id="leftover-agent", bead_store=store)
        await hook.initialize()
        await hook.push(bead_id=bead_id, priority=50)

        # Recover - should skip completed beads
        recovered = await Arena.recover_pending_debates(
            bead_store=store,
            max_age_hours=24,
        )

        assert len(recovered) == 0


class TestGUPPPrinciple:
    """Tests verifying the GUPP principle: 'If there is work on your Hook, YOU MUST RUN IT.'"""

    @pytest.mark.asyncio
    async def test_gupp_work_not_lost_on_crash(self, tmp_path):
        """Verify that work is not lost when system crashes."""
        from aragora.nomic.beads import Bead, BeadStore, BeadType
        from aragora.nomic.hook_queue import HookEntryStatus, HookQueue

        # Setup
        store = BeadStore(bead_dir=tmp_path / ".beads", git_enabled=False)
        await store.initialize()

        bead = Bead.create(
            bead_type=BeadType.DEBATE_DECISION,
            title="Critical debate",
            description="Must not be lost",
        )
        bead_id = await store.create(bead)

        # Agent starts processing (then crashes)
        hook = HookQueue(agent_id="crash-agent", bead_store=store)
        await hook.initialize()
        await hook.push(bead_id=bead_id, priority=100)

        # Simulate starting work
        await hook.pop()  # Marks as PROCESSING

        # Simulate crash - just create new hook (no explicit cleanup)
        hook2 = HookQueue(agent_id="crash-agent", bead_store=store)

        # GUPP recovery
        pending = await hook2.recover_on_startup()

        # Work MUST be recovered
        assert len(pending) == 1
        assert pending[0].id == bead_id

    @pytest.mark.asyncio
    async def test_gupp_processing_reset_to_queued(self, tmp_path):
        """Verify PROCESSING entries are reset to QUEUED on recovery."""
        from aragora.nomic.beads import Bead, BeadStore, BeadType
        from aragora.nomic.hook_queue import HookEntryStatus, HookQueue

        store = BeadStore(bead_dir=tmp_path / ".beads", git_enabled=False)
        await store.initialize()

        bead = Bead.create(
            bead_type=BeadType.DEBATE_DECISION,
            title="Processing debate",
            description="Interrupted",
        )
        bead_id = await store.create(bead)

        hook = HookQueue(agent_id="processing-agent", bead_store=store)
        await hook.initialize()
        entry = await hook.push(bead_id=bead_id)

        # Mark as processing manually
        entry.status = HookEntryStatus.PROCESSING
        await hook._save_entries()

        # Recover
        hook2 = HookQueue(agent_id="processing-agent", bead_store=store)
        await hook2.recover_on_startup()

        # Verify reset to QUEUED
        stats = await hook2.get_statistics()
        assert stats["pending"] == 1
        assert stats["processing"] == 0
