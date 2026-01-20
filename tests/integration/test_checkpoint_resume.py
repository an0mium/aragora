"""
Integration tests for checkpoint/resume functionality.

Tests the complete checkpoint/resume flow:
- Debate creates checkpoints during execution
- Checkpoints can be loaded and verified
- Debates can be resumed from checkpoints with correct state
- Memory state is included in checkpoints when enabled
"""

from __future__ import annotations

import asyncio
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.core import (
    Agent,
    Critique,
    DebateResult,
    Environment,
    Message,
    Vote,
)
from aragora.debate.checkpoint import (
    AgentState,
    CheckpointConfig,
    CheckpointManager,
    CheckpointStatus,
    DebateCheckpoint,
    FileCheckpointStore,
    ResumedDebate,
)
from aragora.debate.checkpoint_ops import CheckpointOperations
from aragora.debate.orchestrator import Arena
from aragora.debate.protocol import DebateProtocol


class MockCheckpointAgent(Agent):
    """Mock agent for checkpoint integration tests."""

    def __init__(self, name: str, proposals: Optional[list[str]] = None):
        super().__init__(name=name, model="mock-checkpoint", role="proposer")
        self.agent_type = "mock"
        self._proposals = proposals or [f"Proposal {i} from {name}" for i in range(10)]
        self._idx = 0
        self.call_count = 0

    async def generate(self, prompt: str, context: list = None) -> str:
        self.call_count += 1
        proposal = self._proposals[self._idx % len(self._proposals)]
        self._idx += 1
        return proposal

    async def generate_stream(self, prompt: str, context: list = None):
        yield await self.generate(prompt, context)

    async def critique(self, proposal: str, task: str, context: list = None) -> Critique:
        return Critique(
            agent=self.name,
            target_agent="unknown",
            target_content=proposal[:100],
            issues=["Minor issue"],
            suggestions=["Consider refinement"],
            severity=0.2,
            reasoning="Good overall approach",
        )

    async def vote(self, proposals: dict, task: str) -> Vote:
        choice = list(proposals.keys())[0]
        return Vote(
            agent=self.name,
            choice=choice,
            reasoning="Best proposal",
            confidence=0.85,
            continue_debate=False,
        )


class TestCheckpointCreation:
    """Tests for checkpoint creation during debate."""

    @pytest.fixture
    def temp_checkpoint_dir(self, tmp_path):
        """Create temporary directory for checkpoints."""
        return tmp_path / "checkpoints"

    @pytest.fixture
    def checkpoint_manager(self, temp_checkpoint_dir):
        """Create a CheckpointManager with file store."""
        store = FileCheckpointStore(str(temp_checkpoint_dir), compress=False)
        config = CheckpointConfig(
            interval_rounds=1,
            max_checkpoints=10,
            expiry_hours=24,
            compress=False,
        )
        return CheckpointManager(store=store, config=config)

    @pytest.fixture
    def mock_agents(self):
        """Create mock agents for testing."""
        return [
            MockCheckpointAgent("agent1", ["Proposal A from agent1"]),
            MockCheckpointAgent("agent2", ["Proposal B from agent2"]),
        ]

    @pytest.mark.asyncio
    async def test_checkpoint_created_after_round(
        self, checkpoint_manager, mock_agents, temp_checkpoint_dir
    ):
        """Checkpoint should be created after each round when enabled."""
        # Create mock debate context
        debate_id = "test-debate-001"
        task = "Test checkpoint creation"

        # Manually create a checkpoint
        checkpoint = await checkpoint_manager.create_checkpoint(
            debate_id=debate_id,
            task=task,
            current_round=1,
            total_rounds=3,
            phase="proposal",
            messages=[
                Message(
                    role="assistant",
                    agent="agent1",
                    content="Test proposal",
                    timestamp=datetime.now(),
                    round=1,
                )
            ],
            critiques=[],
            votes=[],
            agents=mock_agents,
        )

        assert checkpoint is not None
        assert checkpoint.debate_id == debate_id
        assert checkpoint.current_round == 1
        assert checkpoint.total_rounds == 3
        assert len(checkpoint.messages) == 1
        assert checkpoint.status == CheckpointStatus.COMPLETE

    @pytest.mark.asyncio
    async def test_checkpoint_store_persistence(self, checkpoint_manager, mock_agents):
        """Checkpoints should be persisted and loadable."""
        debate_id = "test-persist-001"
        task = "Test persistence"

        # Create checkpoint
        checkpoint = await checkpoint_manager.create_checkpoint(
            debate_id=debate_id,
            task=task,
            current_round=2,
            total_rounds=5,
            phase="critique",
            messages=[
                Message(
                    role="assistant",
                    agent="agent1",
                    content="Proposal content",
                    timestamp=datetime.now(),
                    round=1,
                ),
                Message(
                    role="assistant",
                    agent="agent2",
                    content="Response content",
                    timestamp=datetime.now(),
                    round=2,
                ),
            ],
            critiques=[],
            votes=[],
            agents=mock_agents,
        )

        # Load checkpoint back
        loaded = await checkpoint_manager.store.load(checkpoint.checkpoint_id)

        assert loaded is not None
        assert loaded.debate_id == debate_id
        assert loaded.current_round == 2
        assert len(loaded.messages) == 2
        assert loaded.verify_integrity()

    @pytest.mark.asyncio
    async def test_multiple_checkpoints_per_debate(self, checkpoint_manager, mock_agents):
        """Multiple checkpoints should be created for different rounds."""
        debate_id = "test-multi-001"
        task = "Multi-round debate"

        checkpoints = []
        for round_num in range(1, 4):
            cp = await checkpoint_manager.create_checkpoint(
                debate_id=debate_id,
                task=task,
                current_round=round_num,
                total_rounds=5,
                phase="proposal",
                messages=[
                    Message(
                        role="assistant",
                        agent=f"agent{round_num}",
                        content=f"Round {round_num} content",
                        timestamp=datetime.now(),
                        round=round_num,
                    )
                ],
                critiques=[],
                votes=[],
                agents=mock_agents,
            )
            checkpoints.append(cp)

        # List checkpoints for debate
        listed = await checkpoint_manager.store.list_checkpoints(debate_id=debate_id)

        assert len(listed) == 3
        assert all(cp["debate_id"] == debate_id for cp in listed)


class TestCheckpointResume:
    """Tests for resuming debates from checkpoints."""

    @pytest.fixture
    def temp_checkpoint_dir(self, tmp_path):
        """Create temporary directory for checkpoints."""
        return tmp_path / "checkpoints"

    @pytest.fixture
    def checkpoint_manager(self, temp_checkpoint_dir):
        """Create a CheckpointManager with file store."""
        store = FileCheckpointStore(str(temp_checkpoint_dir), compress=False)
        config = CheckpointConfig(
            interval_rounds=1,
            max_checkpoints=10,
            expiry_hours=24,
            compress=False,
        )
        return CheckpointManager(store=store, config=config)

    @pytest.fixture
    def mock_agents(self):
        """Create mock agents for testing."""
        return [
            MockCheckpointAgent("agent1"),
            MockCheckpointAgent("agent2"),
        ]

    @pytest.mark.asyncio
    async def test_resume_restores_messages(self, checkpoint_manager, mock_agents):
        """Resuming should restore all messages from checkpoint."""
        debate_id = "test-resume-001"
        messages = [
            Message(
                role="assistant",
                agent="agent1",
                content="First message",
                timestamp=datetime.now(),
                round=1,
            ),
            Message(
                role="assistant",
                agent="agent2",
                content="Second message",
                timestamp=datetime.now(),
                round=1,
            ),
            Message(
                role="assistant",
                agent="agent1",
                content="Third message",
                timestamp=datetime.now(),
                round=2,
            ),
        ]

        # Create checkpoint
        checkpoint = await checkpoint_manager.create_checkpoint(
            debate_id=debate_id,
            task="Resume test",
            current_round=2,
            total_rounds=5,
            phase="proposal",
            messages=messages,
            critiques=[],
            votes=[],
            agents=mock_agents,
        )

        # Resume from checkpoint
        resumed = await checkpoint_manager.resume_from_checkpoint(
            checkpoint.checkpoint_id, resumed_by="test"
        )

        assert resumed is not None
        assert len(resumed.messages) == 3
        assert resumed.messages[0].content == "First message"
        assert resumed.messages[2].content == "Third message"
        assert resumed.checkpoint.resume_count == 1

    @pytest.mark.asyncio
    async def test_resume_restores_votes(self, checkpoint_manager, mock_agents):
        """Resuming should restore all votes from checkpoint."""
        debate_id = "test-resume-votes-001"

        votes = [
            Vote(
                agent="agent1",
                choice="agent2",
                reasoning="Better proposal",
                confidence=0.9,
                continue_debate=False,
            ),
            Vote(
                agent="agent2",
                choice="agent1",
                reasoning="More detailed",
                confidence=0.8,
                continue_debate=True,
            ),
        ]

        # Create checkpoint with votes
        checkpoint = await checkpoint_manager.create_checkpoint(
            debate_id=debate_id,
            task="Vote resume test",
            current_round=3,
            total_rounds=5,
            phase="vote",
            messages=[],
            critiques=[],
            votes=votes,
            agents=mock_agents,
        )

        # Resume
        resumed = await checkpoint_manager.resume_from_checkpoint(checkpoint.checkpoint_id)

        assert resumed is not None
        assert len(resumed.votes) == 2
        assert resumed.votes[0].choice == "agent2"
        assert resumed.votes[1].confidence == 0.8

    @pytest.mark.asyncio
    async def test_resume_updates_metadata(self, checkpoint_manager, mock_agents):
        """Resume should update checkpoint metadata."""
        debate_id = "test-resume-meta-001"

        checkpoint = await checkpoint_manager.create_checkpoint(
            debate_id=debate_id,
            task="Metadata test",
            current_round=1,
            total_rounds=3,
            phase="proposal",
            messages=[],
            critiques=[],
            votes=[],
            agents=mock_agents,
        )

        # Resume multiple times
        await checkpoint_manager.resume_from_checkpoint(
            checkpoint.checkpoint_id, resumed_by="user1"
        )
        await checkpoint_manager.resume_from_checkpoint(
            checkpoint.checkpoint_id, resumed_by="user2"
        )

        # Load checkpoint to check updated metadata
        loaded = await checkpoint_manager.store.load(checkpoint.checkpoint_id)

        assert loaded.resume_count == 2
        assert loaded.resumed_by == "user2"
        assert loaded.last_resumed_at is not None

    @pytest.mark.asyncio
    async def test_resume_detects_corruption(self, checkpoint_manager):
        """Resume should detect corrupted checkpoints."""
        # Create a minimal checkpoint directly
        checkpoint = DebateCheckpoint(
            checkpoint_id="corrupt-test-001",
            debate_id="test-corrupt",
            task="Corruption test",
            current_round=1,
            total_rounds=3,
            phase="proposal",
            messages=[],
            critiques=[],
            votes=[],
            agent_states=[],
        )

        # Save it
        await checkpoint_manager.store.save(checkpoint)

        # Manually corrupt by changing the checksum source data
        checkpoint.current_round = 99  # This changes expected checksum

        # Save corrupted version (checksum no longer matches)
        await checkpoint_manager.store.save(checkpoint)

        # Attempt resume should fail integrity check
        resumed = await checkpoint_manager.resume_from_checkpoint("corrupt-test-001")

        # Should return None due to integrity failure
        assert resumed is None


class TestCheckpointIntegration:
    """Integration tests for checkpoint with debate flow."""

    @pytest.fixture
    def temp_checkpoint_dir(self, tmp_path):
        """Create temporary directory for checkpoints."""
        return tmp_path / "checkpoints"

    @pytest.fixture
    def checkpoint_manager(self, temp_checkpoint_dir):
        """Create a CheckpointManager with file store."""
        store = FileCheckpointStore(str(temp_checkpoint_dir), compress=False)
        config = CheckpointConfig(
            interval_rounds=1,
            max_checkpoints=10,
            expiry_hours=24,
            compress=False,
        )
        return CheckpointManager(store=store, config=config)

    @pytest.mark.asyncio
    async def test_checkpoint_ops_integration(self, checkpoint_manager):
        """CheckpointOperations should integrate with CheckpointManager."""
        ops = CheckpointOperations(checkpoint_manager=checkpoint_manager)

        # Create mock context
        mock_ctx = MagicMock()
        mock_ctx.debate_id = "ops-test-001"
        mock_ctx.result = MagicMock()
        mock_ctx.result.messages = [
            Message(
                role="assistant",
                agent="agent1",
                content="Test",
                timestamp=datetime.now(),
                round=1,
            )
        ]
        mock_ctx.result.critiques = []
        mock_ctx.result.votes = []
        mock_ctx.result.final_answer = "Test consensus"

        mock_env = MagicMock()
        mock_env.task = "Test task"

        mock_protocol = MagicMock()
        mock_protocol.rounds = 3

        mock_agents = [
            MockCheckpointAgent("agent1"),
            MockCheckpointAgent("agent2"),
        ]

        # Create checkpoint through ops
        await ops.create_checkpoint(
            ctx=mock_ctx,
            round_num=1,
            env=mock_env,
            agents=mock_agents,
            protocol=mock_protocol,
        )

        # Verify checkpoint was created
        checkpoints = await checkpoint_manager.store.list_checkpoints(debate_id="ops-test-001")
        assert len(checkpoints) == 1

    @pytest.mark.asyncio
    async def test_checkpoint_includes_agent_states(self, checkpoint_manager):
        """Checkpoint should include serialized agent states."""
        agents = [
            MockCheckpointAgent("proposer1"),
            MockCheckpointAgent("critic1"),
        ]
        agents[1].role = "critic"

        checkpoint = await checkpoint_manager.create_checkpoint(
            debate_id="agent-state-001",
            task="Agent state test",
            current_round=1,
            total_rounds=3,
            phase="proposal",
            messages=[],
            critiques=[],
            votes=[],
            agents=agents,
        )

        assert len(checkpoint.agent_states) == 2
        assert checkpoint.agent_states[0].agent_name == "proposer1"
        assert checkpoint.agent_states[0].agent_role == "proposer"
        assert checkpoint.agent_states[1].agent_name == "critic1"

    @pytest.mark.asyncio
    async def test_checkpoint_with_memory_state(self, checkpoint_manager):
        """Checkpoint should include memory state when provided."""
        memory_state = {
            "fast": [{"id": "mem1", "content": "Fast memory"}],
            "medium": [{"id": "mem2", "content": "Medium memory"}],
            "slow": [],
        }

        checkpoint = await checkpoint_manager.create_checkpoint(
            debate_id="memory-state-001",
            task="Memory state test",
            current_round=2,
            total_rounds=5,
            phase="critique",
            messages=[],
            critiques=[],
            votes=[],
            agents=[MockCheckpointAgent("agent1")],
            continuum_memory_state=memory_state,
        )

        assert checkpoint.continuum_memory_state is not None
        assert len(checkpoint.continuum_memory_state["fast"]) == 1
        assert checkpoint.continuum_memory_state["fast"][0]["id"] == "mem1"

        # Load and verify persistence
        loaded = await checkpoint_manager.store.load(checkpoint.checkpoint_id)
        assert loaded.continuum_memory_state is not None
        assert loaded.continuum_memory_state["fast"][0]["content"] == "Fast memory"


class TestCheckpointCleanup:
    """Tests for checkpoint cleanup and expiry."""

    @pytest.fixture
    def temp_checkpoint_dir(self, tmp_path):
        """Create temporary directory for checkpoints."""
        return tmp_path / "checkpoints"

    @pytest.fixture
    def checkpoint_manager(self, temp_checkpoint_dir):
        """Create a CheckpointManager with low limits for testing."""
        store = FileCheckpointStore(str(temp_checkpoint_dir), compress=False)
        config = CheckpointConfig(
            interval_rounds=1,
            max_checkpoints=3,  # Low limit for testing
            expiry_hours=24,
            compress=False,
            auto_cleanup=True,
        )
        return CheckpointManager(store=store, config=config)

    @pytest.mark.asyncio
    async def test_old_checkpoints_cleaned_up(self, checkpoint_manager):
        """Old checkpoints beyond limit should be cleaned up."""
        debate_id = "cleanup-test-001"
        agents = [MockCheckpointAgent("agent1")]

        # Create more checkpoints than the limit
        for round_num in range(1, 6):
            await checkpoint_manager.create_checkpoint(
                debate_id=debate_id,
                task="Cleanup test",
                current_round=round_num,
                total_rounds=10,
                phase="proposal",
                messages=[],
                critiques=[],
                votes=[],
                agents=agents,
            )

        # Should only have max_checkpoints remaining
        checkpoints = await checkpoint_manager.store.list_checkpoints(debate_id=debate_id)
        assert len(checkpoints) <= 3  # max_checkpoints

    @pytest.mark.asyncio
    async def test_delete_checkpoint(self, checkpoint_manager):
        """Checkpoints should be deletable."""
        agents = [MockCheckpointAgent("agent1")]

        checkpoint = await checkpoint_manager.create_checkpoint(
            debate_id="delete-test-001",
            task="Delete test",
            current_round=1,
            total_rounds=3,
            phase="proposal",
            messages=[],
            critiques=[],
            votes=[],
            agents=agents,
        )

        # Delete
        deleted = await checkpoint_manager.store.delete(checkpoint.checkpoint_id)
        assert deleted

        # Verify gone
        loaded = await checkpoint_manager.store.load(checkpoint.checkpoint_id)
        assert loaded is None


class TestCheckpointIntervention:
    """Tests for human intervention on checkpoints."""

    @pytest.fixture
    def temp_checkpoint_dir(self, tmp_path):
        """Create temporary directory for checkpoints."""
        return tmp_path / "checkpoints"

    @pytest.fixture
    def checkpoint_manager(self, temp_checkpoint_dir):
        """Create a CheckpointManager with file store."""
        store = FileCheckpointStore(str(temp_checkpoint_dir), compress=False)
        return CheckpointManager(store=store)

    @pytest.mark.asyncio
    async def test_add_intervention_note(self, checkpoint_manager):
        """Should be able to add intervention notes to checkpoint."""
        agents = [MockCheckpointAgent("agent1")]

        checkpoint = await checkpoint_manager.create_checkpoint(
            debate_id="intervention-test-001",
            task="Intervention test",
            current_round=2,
            total_rounds=5,
            phase="critique",
            messages=[],
            critiques=[],
            votes=[],
            agents=agents,
        )

        # Add intervention
        success = await checkpoint_manager.add_intervention(
            checkpoint.checkpoint_id,
            note="Need to review agent responses",
            by="human-reviewer",
        )

        assert success

        # Load and verify
        loaded = await checkpoint_manager.store.load(checkpoint.checkpoint_id)
        assert loaded.pending_intervention
        assert len(loaded.intervention_notes) == 1
        assert "human-reviewer" in loaded.intervention_notes[0]

    @pytest.mark.asyncio
    async def test_multiple_interventions(self, checkpoint_manager):
        """Multiple intervention notes should accumulate."""
        agents = [MockCheckpointAgent("agent1")]

        checkpoint = await checkpoint_manager.create_checkpoint(
            debate_id="multi-intervention-001",
            task="Multi intervention test",
            current_round=1,
            total_rounds=3,
            phase="proposal",
            messages=[],
            critiques=[],
            votes=[],
            agents=agents,
        )

        # Add multiple interventions
        await checkpoint_manager.add_intervention(
            checkpoint.checkpoint_id, "First note", by="reviewer1"
        )
        await checkpoint_manager.add_intervention(
            checkpoint.checkpoint_id, "Second note", by="reviewer2"
        )
        await checkpoint_manager.add_intervention(
            checkpoint.checkpoint_id, "Third note", by="reviewer1"
        )

        loaded = await checkpoint_manager.store.load(checkpoint.checkpoint_id)
        assert len(loaded.intervention_notes) == 3


class TestListDebatesWithCheckpoints:
    """Tests for listing debates that have checkpoints."""

    @pytest.fixture
    def temp_checkpoint_dir(self, tmp_path):
        """Create temporary directory for checkpoints."""
        return tmp_path / "checkpoints"

    @pytest.fixture
    def checkpoint_manager(self, temp_checkpoint_dir):
        """Create a CheckpointManager with file store."""
        store = FileCheckpointStore(str(temp_checkpoint_dir), compress=False)
        return CheckpointManager(store=store)

    @pytest.mark.asyncio
    async def test_list_debates_with_checkpoints(self, checkpoint_manager):
        """Should list all debates that have checkpoints."""
        agents = [MockCheckpointAgent("agent1")]

        # Create checkpoints for multiple debates
        for debate_num in range(1, 4):
            for round_num in range(1, 3):
                await checkpoint_manager.create_checkpoint(
                    debate_id=f"debate-{debate_num:03d}",
                    task=f"Task {debate_num}",
                    current_round=round_num,
                    total_rounds=5,
                    phase="proposal",
                    messages=[],
                    critiques=[],
                    votes=[],
                    agents=agents,
                )

        debates = await checkpoint_manager.list_debates_with_checkpoints()

        assert len(debates) == 3
        for debate in debates:
            assert "debate_id" in debate
            assert "checkpoint_count" in debate
            assert debate["checkpoint_count"] == 2
            assert debate["latest_round"] == 2
