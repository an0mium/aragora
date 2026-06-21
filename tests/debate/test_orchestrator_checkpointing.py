"""
Tests for debate checkpointing and resume functionality.

Tests checkpoint save/resume features:
- Checkpoint creation at configurable intervals
- State serialization and deserialization
- Resume from checkpoint with state consistency
- Crash recovery scenarios
"""

import pytest
import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch
from typing import Optional
from datetime import datetime

from aragora.core import (
    Agent,
    Environment,
    Vote,
    Message,
    Critique,
    DebateResult,
)
from aragora.debate.orchestrator import Arena, DebateProtocol
from aragora.debate.checkpoint import (
    DebateCheckpoint,
    CheckpointStatus,
    AgentState,
    CheckpointStore,
    FileCheckpointStore,
    CheckpointManager,
    CheckpointConfig,
)


class CheckpointMockAgent(Agent):
    """Mock agent for checkpoint testing."""

    def __init__(self, name: str, proposals: list = None):
        super().__init__(name=name, model="mock-model", role="proposer")
        self.agent_type = "mock"
        self._proposals = proposals or [f"Proposal {i} from {name}" for i in range(10)]
        self._idx = 0
        self.call_history = []

    async def generate(self, prompt: str, context: list = None) -> str:
        self.call_history.append(("generate", self._idx))
        proposal = self._proposals[self._idx % len(self._proposals)]
        self._idx += 1
        return proposal

    async def generate_stream(self, prompt: str, context: list = None):
        yield await self.generate(prompt, context)

    async def critique(self, proposal: str, task: str, context: list = None) -> Critique:
        self.call_history.append(("critique", proposal[:20]))
        return Critique(
            agent=self.name,
            target_agent="unknown",
            target_content=proposal[:100],
            issues=["Issue"],
            suggestions=["Suggestion"],
            severity=0.3,
            reasoning="Test critique",
        )

    async def vote(self, proposals: dict, task: str) -> Vote:
        choice = self.name if self.name in proposals else list(proposals.keys())[0]
        self.call_history.append(("vote", choice))
        return Vote(
            agent=self.name,
            choice=choice,
            reasoning="Test vote",
            confidence=0.8,
            continue_debate=False,
        )

    def get_state(self) -> dict:
        """Get agent state for checkpointing."""
        return {
            "name": self.name,
            "idx": self._idx,
            "call_history": self.call_history,
        }

    def restore_state(self, state: dict):
        """Restore agent state from checkpoint."""
        self._idx = state.get("idx", 0)
        self.call_history = state.get("call_history", [])


class TestCheckpointDataclass:
    """Tests for DebateCheckpoint dataclass."""

    def test_checkpoint_has_required_fields(self):
        """DebateCheckpoint has all required fields."""
        checkpoint = DebateCheckpoint(
            checkpoint_id="test-123",
            debate_id="debate-456",
            task="Test task",
            current_round=2,
            total_rounds=5,
            phase="proposal",
            messages=[],
            critiques=[],
            votes=[],
            agent_states=[],
        )

        assert checkpoint.checkpoint_id == "test-123"
        assert checkpoint.debate_id == "debate-456"
        assert checkpoint.current_round == 2
        assert checkpoint.total_rounds == 5
        assert checkpoint.phase == "proposal"
        assert checkpoint.status == CheckpointStatus.COMPLETE

    def test_checkpoint_with_messages(self):
        """Checkpoint stores message history."""
        messages = [
            {"role": "assistant", "agent": "alice", "content": "test"},
        ]
        checkpoint = DebateCheckpoint(
            checkpoint_id="msg-test",
            debate_id="d1",
            task="Task",
            current_round=1,
            total_rounds=3,
            phase="proposal",
            messages=messages,
            critiques=[],
            votes=[],
            agent_states=[],
        )

        assert len(checkpoint.messages) == 1
        assert checkpoint.messages[0]["agent"] == "alice"

    def test_checkpoint_status_enum(self):
        """CheckpointStatus enum has expected values."""
        assert CheckpointStatus.CREATING.value == "creating"
        assert CheckpointStatus.COMPLETE.value == "complete"
        assert CheckpointStatus.RESUMING.value == "resuming"
        assert CheckpointStatus.CORRUPTED.value == "corrupted"


class TestAgentStateDataclass:
    """Tests for AgentState dataclass."""

    def test_agent_state_creation(self):
        """AgentState can be created with required fields."""
        state = AgentState(
            agent_name="alice",
            agent_model="gpt-4",
            agent_role="proposer",
            system_prompt="You are helpful",
            stance="neutral",
        )

        assert state.agent_name == "alice"
        assert state.agent_model == "gpt-4"
        assert state.agent_role == "proposer"

    def test_agent_state_with_memory(self):
        """AgentState can include memory snapshot."""
        state = AgentState(
            agent_name="bob",
            agent_model="claude",
            agent_role="critic",
            system_prompt="",
            stance="skeptical",
            memory_snapshot={"key": "value"},
        )

        assert state.memory_snapshot == {"key": "value"}


class TestMockAgentState:
    """Tests for mock agent state operations."""

    def test_agent_state_serialization(self):
        """Agent state can be serialized."""
        agent = CheckpointMockAgent("alice")
        agent._idx = 5
        agent.call_history = [("generate", 0), ("vote", "alice")]

        state = agent.get_state()

        assert state["name"] == "alice"
        assert state["idx"] == 5
        assert len(state["call_history"]) == 2

    def test_agent_state_restoration(self):
        """Agent state can be restored."""
        agent = CheckpointMockAgent("alice")
        state = {
            "name": "alice",
            "idx": 3,
            "call_history": [("generate", 0), ("generate", 1)],
        }

        agent.restore_state(state)

        assert agent._idx == 3
        assert len(agent.call_history) == 2


class TestFileCheckpointStore:
    """Tests for file-based checkpoint storage."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_file_store_initialization(self, temp_dir):
        """File store initializes correctly."""
        store = FileCheckpointStore(str(temp_dir))
        # Use resolve() for comparison as macOS /var -> /private/var
        assert store.base_dir.resolve() == temp_dir.resolve()
        assert store.compress is True

    def test_file_store_no_compression(self, temp_dir):
        """File store can disable compression."""
        store = FileCheckpointStore(str(temp_dir), compress=False)
        assert store.compress is False

    @pytest.mark.asyncio
    async def test_file_store_save_and_load(self, temp_dir):
        """File store saves and loads checkpoints."""
        store = FileCheckpointStore(str(temp_dir))
        checkpoint = DebateCheckpoint(
            checkpoint_id="save-test-001",
            debate_id="debate-1",
            task="Test task",
            current_round=1,
            total_rounds=3,
            phase="proposal",
            messages=[{"content": "test"}],
            critiques=[],
            votes=[],
            agent_states=[],
        )

        # Save
        path = await store.save(checkpoint)
        assert path is not None

        # Load
        loaded = await store.load("save-test-001")
        assert loaded is not None
        assert loaded.checkpoint_id == "save-test-001"
        assert loaded.current_round == 1

    @pytest.mark.asyncio
    async def test_file_store_handles_missing(self, temp_dir):
        """File store handles missing checkpoint gracefully."""
        store = FileCheckpointStore(str(temp_dir))

        loaded = await store.load("nonexistent")

        assert loaded is None

    @pytest.mark.asyncio
    async def test_file_store_list_checkpoints(self, temp_dir):
        """File store lists checkpoints."""
        store = FileCheckpointStore(str(temp_dir))

        # Save a checkpoint
        checkpoint = DebateCheckpoint(
            checkpoint_id="list-test-001",
            debate_id="debate-list",
            task="List Test",
            current_round=1,
            total_rounds=5,
            phase="proposal",
            messages=[],
            critiques=[],
            votes=[],
            agent_states=[],
        )
        await store.save(checkpoint)

        # List should find it
        checkpoints = await store.list_checkpoints()
        assert len(checkpoints) >= 1


class TestCheckpointConfig:
    """Tests for CheckpointConfig."""

    def test_default_config(self):
        """CheckpointConfig has sensible defaults."""
        config = CheckpointConfig()
        assert config.interval_rounds >= 1
        assert config.interval_seconds >= 0

    def test_custom_config(self):
        """CheckpointConfig accepts custom values."""
        config = CheckpointConfig(
            interval_rounds=2,
            interval_seconds=120,
        )
        assert config.interval_rounds == 2
        assert config.interval_seconds == 120


class TestCheckpointManager:
    """Tests for CheckpointManager."""

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def manager(self, temp_dir):
        store = FileCheckpointStore(str(temp_dir))
        config = CheckpointConfig(interval_rounds=1)
        return CheckpointManager(store=store, config=config)

    def test_manager_creation(self, manager):
        """Manager initializes correctly."""
        assert manager.store is not None
        assert manager.config is not None

    def test_should_checkpoint_on_interval(self, manager):
        """Manager triggers checkpoint on round interval."""
        # Round 1 should checkpoint (interval=1)
        assert manager.should_checkpoint("debate-1", 1) is True
        # Round 2 should also checkpoint
        assert manager.should_checkpoint("debate-1", 2) is True

    @pytest.mark.asyncio
    async def test_manager_creates_checkpoint(self, manager):
        """Manager creates checkpoint with all required data."""
        messages = [
            Message(role="assistant", agent="alice", content="test", round=1),
        ]
        critiques = []
        votes = []
        agents = [CheckpointMockAgent("alice")]

        checkpoint = await manager.create_checkpoint(
            debate_id="mgr-test",
            task="Manager test",
            current_round=2,
            total_rounds=5,
            phase="voting",
            messages=messages,
            critiques=critiques,
            votes=votes,
            agents=agents,
        )

        assert checkpoint is not None
        assert checkpoint.debate_id == "mgr-test"
        assert checkpoint.current_round == 2
        assert len(checkpoint.messages) == 1


class TestCheckpointIntegrity:
    """Tests for checkpoint data integrity."""

    def test_checkpoint_with_unicode(self):
        """Checkpoint handles unicode content."""
        checkpoint = DebateCheckpoint(
            checkpoint_id="unicode-test",
            debate_id="edge-2",
            task="Unicode test: émojis 🎉 and symbols ∞",
            current_round=1,
            total_rounds=2,
            phase="proposal",
            messages=[{"content": "Unicode: 你好世界 🌍"}],
            critiques=[],
            votes=[],
            agent_states=[],
        )

        assert "émojis" in checkpoint.task
        assert "🎉" in checkpoint.task

    def test_checkpoint_status_transitions(self):
        """Checkpoint status can be changed."""
        checkpoint = DebateCheckpoint(
            checkpoint_id="status-test",
            debate_id="edge-3",
            task="Status test",
            current_round=1,
            total_rounds=2,
            phase="proposal",
            messages=[],
            critiques=[],
            votes=[],
            agent_states=[],
            status=CheckpointStatus.CREATING,
        )

        assert checkpoint.status == CheckpointStatus.CREATING

        # Status can be changed
        checkpoint.status = CheckpointStatus.COMPLETE
        assert checkpoint.status == CheckpointStatus.COMPLETE

    def test_empty_checkpoint_valid(self):
        """Empty checkpoint is valid."""
        checkpoint = DebateCheckpoint(
            checkpoint_id="empty",
            debate_id="edge-1",
            task="Empty test",
            current_round=0,
            total_rounds=1,
            phase="init",
            messages=[],
            critiques=[],
            votes=[],
            agent_states=[],
            status=CheckpointStatus.CREATING,
        )

        assert checkpoint.current_round == 0
        assert len(checkpoint.messages) == 0


class TestCheckpointResume:
    """Tests for resuming from checkpoints."""

    @pytest.fixture
    def env(self):
        return Environment(task="Resume test")

    @pytest.fixture
    def agents(self):
        return [
            CheckpointMockAgent("alice"),
            CheckpointMockAgent("bob"),
        ]

    @pytest.mark.asyncio
    @pytest.mark.timeout(60)
    async def test_debate_completes_normally(self, env, agents):
        """Normal debate completes without needing resume."""
        protocol = DebateProtocol(rounds=2, consensus="majority", enable_calibration=False)
        arena = Arena(environment=env, agents=agents, protocol=protocol)

        result = await arena.run()

        assert result is not None
        assert result.final_answer is not None or result.messages

    @pytest.mark.asyncio
    async def test_agents_can_restore_state(self, agents):
        """Agents can save and restore their state."""
        alice = agents[0]

        # Simulate some calls
        await alice.generate("test", [])
        await alice.generate("test2", [])
        original_state = alice.get_state()

        # Create new agent and restore
        new_alice = CheckpointMockAgent("alice")
        new_alice.restore_state(original_state)

        assert new_alice._idx == alice._idx
        assert len(new_alice.call_history) == len(alice.call_history)
