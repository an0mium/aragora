"""
Tests for Debate Checkpoint Module.

Tests the checkpoint functionality including:
- CheckpointStatus enum
- AgentState dataclass
- DebateCheckpoint dataclass
- FileCheckpointStore class
- Checkpoint integrity verification
- Serialization/deserialization
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aragora.debate.checkpoint import (
    AgentState,
    CheckpointStatus,
    DebateCheckpoint,
    FileCheckpointStore,
    ResumedDebate,
    SAFE_CHECKPOINT_ID,
)


# =============================================================================
# CheckpointStatus Enum Tests
# =============================================================================


class TestCheckpointStatus:
    """Test CheckpointStatus enum."""

    def test_creating_status(self):
        """Test creating status value."""
        assert CheckpointStatus.CREATING.value == "creating"

    def test_complete_status(self):
        """Test complete status value."""
        assert CheckpointStatus.COMPLETE.value == "complete"

    def test_resuming_status(self):
        """Test resuming status value."""
        assert CheckpointStatus.RESUMING.value == "resuming"

    def test_corrupted_status(self):
        """Test corrupted status value."""
        assert CheckpointStatus.CORRUPTED.value == "corrupted"

    def test_expired_status(self):
        """Test expired status value."""
        assert CheckpointStatus.EXPIRED.value == "expired"

    def test_status_from_string(self):
        """Test creating status from string."""
        assert CheckpointStatus("complete") == CheckpointStatus.COMPLETE


# =============================================================================
# AgentState Dataclass Tests
# =============================================================================


class TestAgentState:
    """Test AgentState dataclass."""

    def test_create_agent_state(self):
        """Test creating agent state."""
        state = AgentState(
            agent_name="claude",
            agent_model="claude-3",
            agent_role="proposer",
            system_prompt="You are a helpful assistant.",
            stance="neutral",
        )

        assert state.agent_name == "claude"
        assert state.agent_model == "claude-3"
        assert state.agent_role == "proposer"
        assert state.stance == "neutral"

    def test_agent_state_with_memory(self):
        """Test agent state with memory snapshot."""
        memory = {"recent_messages": ["msg1", "msg2"]}
        state = AgentState(
            agent_name="gpt",
            agent_model="gpt-4",
            agent_role="critic",
            system_prompt="Analyze critically.",
            stance="skeptical",
            memory_snapshot=memory,
        )

        assert state.memory_snapshot == memory

    def test_agent_state_optional_memory(self):
        """Test agent state without memory."""
        state = AgentState(
            agent_name="agent",
            agent_model="model",
            agent_role="role",
            system_prompt="prompt",
            stance="stance",
        )

        assert state.memory_snapshot is None


# =============================================================================
# DebateCheckpoint Dataclass Tests
# =============================================================================


class TestDebateCheckpoint:
    """Test DebateCheckpoint dataclass."""

    @pytest.fixture
    def sample_agent_state(self):
        """Create sample agent state."""
        return AgentState(
            agent_name="claude",
            agent_model="claude-3",
            agent_role="proposer",
            system_prompt="Help debate.",
            stance="pro",
        )

    @pytest.fixture
    def sample_checkpoint(self, sample_agent_state):
        """Create sample checkpoint."""
        return DebateCheckpoint(
            checkpoint_id="cp-123",
            debate_id="debate-456",
            task="Should we implement feature X?",
            current_round=2,
            total_rounds=5,
            phase="critique",
            messages=[{"role": "assistant", "content": "Proposal text"}],
            critiques=[{"agent": "critic", "text": "Good point but..."}],
            votes=[],
            agent_states=[sample_agent_state],
        )

    def test_create_checkpoint(self, sample_checkpoint):
        """Test creating a checkpoint."""
        assert sample_checkpoint.checkpoint_id == "cp-123"
        assert sample_checkpoint.debate_id == "debate-456"
        assert sample_checkpoint.current_round == 2
        assert sample_checkpoint.total_rounds == 5
        assert sample_checkpoint.phase == "critique"

    def test_checkpoint_computes_checksum(self, sample_checkpoint):
        """Test checkpoint computes checksum on creation."""
        assert sample_checkpoint.checksum != ""
        assert len(sample_checkpoint.checksum) == 16

    def test_checkpoint_verify_integrity_pass(self, sample_checkpoint):
        """Test integrity verification passes for valid checkpoint."""
        assert sample_checkpoint.verify_integrity() is True

    def test_checkpoint_verify_integrity_fail(self, sample_checkpoint):
        """Test integrity verification fails after modification."""
        original_checksum = sample_checkpoint.checksum
        sample_checkpoint.current_round = 999
        # Checksum doesn't change, but verify_integrity should detect mismatch
        assert sample_checkpoint.checksum == original_checksum
        assert sample_checkpoint.verify_integrity() is False

    def test_checkpoint_to_dict(self, sample_checkpoint):
        """Test checkpoint serialization to dict."""
        d = sample_checkpoint.to_dict()

        assert d["checkpoint_id"] == "cp-123"
        assert d["debate_id"] == "debate-456"
        assert d["current_round"] == 2
        assert d["status"] == "complete"
        assert len(d["agent_states"]) == 1
        assert d["agent_states"][0]["agent_name"] == "claude"

    def test_checkpoint_from_dict(self, sample_checkpoint):
        """Test checkpoint deserialization from dict."""
        d = sample_checkpoint.to_dict()
        restored = DebateCheckpoint.from_dict(d)

        assert restored.checkpoint_id == sample_checkpoint.checkpoint_id
        assert restored.debate_id == sample_checkpoint.debate_id
        assert restored.current_round == sample_checkpoint.current_round
        assert len(restored.agent_states) == 1

    def test_checkpoint_round_trip(self, sample_checkpoint):
        """Test checkpoint survives serialization round-trip."""
        d = sample_checkpoint.to_dict()
        json_str = json.dumps(d)
        restored_dict = json.loads(json_str)
        restored = DebateCheckpoint.from_dict(restored_dict)

        assert restored.checkpoint_id == sample_checkpoint.checkpoint_id
        assert restored.verify_integrity() is True

    def test_checkpoint_with_consensus(self, sample_agent_state):
        """Test checkpoint with consensus data."""
        cp = DebateCheckpoint(
            checkpoint_id="cp-with-consensus",
            debate_id="debate-1",
            task="Test task",
            current_round=3,
            total_rounds=3,
            phase="synthesis",
            messages=[],
            critiques=[],
            votes=[],
            agent_states=[sample_agent_state],
            current_consensus="The proposal is valid.",
            consensus_confidence=0.85,
            convergence_status="converged",
        )

        assert cp.current_consensus == "The proposal is valid."
        assert cp.consensus_confidence == 0.85

    def test_checkpoint_with_intervention(self, sample_agent_state):
        """Test checkpoint with pending intervention."""
        cp = DebateCheckpoint(
            checkpoint_id="cp-intervention",
            debate_id="debate-2",
            task="Test task",
            current_round=2,
            total_rounds=5,
            phase="proposal",
            messages=[],
            critiques=[],
            votes=[],
            agent_states=[sample_agent_state],
            pending_intervention=True,
            intervention_notes=["Need human review of proposal"],
        )

        assert cp.pending_intervention is True
        assert "Need human review" in cp.intervention_notes[0]

    def test_checkpoint_default_status(self, sample_checkpoint):
        """Test checkpoint default status is COMPLETE."""
        assert sample_checkpoint.status == CheckpointStatus.COMPLETE

    def test_checkpoint_timestamps(self, sample_checkpoint):
        """Test checkpoint has created_at timestamp."""
        assert sample_checkpoint.created_at is not None
        # Should be a valid ISO format timestamp
        datetime.fromisoformat(sample_checkpoint.created_at)


# =============================================================================
# ResumedDebate Tests
# =============================================================================


class TestResumedDebate:
    """Test ResumedDebate dataclass."""

    def test_create_resumed_debate(self):
        """Test creating resumed debate context."""
        agent_state = AgentState(
            agent_name="test",
            agent_model="model",
            agent_role="role",
            system_prompt="prompt",
            stance="neutral",
        )
        checkpoint = DebateCheckpoint(
            checkpoint_id="cp-1",
            debate_id="debate-1",
            task="Test",
            current_round=1,
            total_rounds=3,
            phase="proposal",
            messages=[],
            critiques=[],
            votes=[],
            agent_states=[agent_state],
        )

        resumed = ResumedDebate(
            checkpoint=checkpoint,
            original_debate_id="debate-1",
            resumed_at=datetime.now().isoformat(),
            resumed_by="user-123",
            messages=[],
            votes=[],
        )

        assert resumed.original_debate_id == "debate-1"
        assert resumed.resumed_by == "user-123"
        assert resumed.context_drift_detected is False


# =============================================================================
# FileCheckpointStore Tests
# =============================================================================


class TestFileCheckpointStore:
    """Test FileCheckpointStore class."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def store(self, temp_dir):
        """Create file checkpoint store."""
        return FileCheckpointStore(base_dir=temp_dir, compress=False)

    @pytest.fixture
    def compressed_store(self, temp_dir):
        """Create compressed file checkpoint store."""
        return FileCheckpointStore(base_dir=temp_dir, compress=True)

    @pytest.fixture
    def sample_checkpoint(self):
        """Create sample checkpoint for testing."""
        agent_state = AgentState(
            agent_name="test-agent",
            agent_model="test-model",
            agent_role="proposer",
            system_prompt="Test prompt",
            stance="neutral",
        )
        return DebateCheckpoint(
            checkpoint_id="test-cp-123",
            debate_id="debate-test-1",
            task="Test task for checkpointing",
            current_round=1,
            total_rounds=3,
            phase="proposal",
            messages=[{"role": "user", "content": "Hello"}],
            critiques=[],
            votes=[],
            agent_states=[agent_state],
        )

    @pytest.mark.asyncio
    async def test_save_checkpoint(self, store, sample_checkpoint):
        """Test saving checkpoint to file."""
        path = await store.save(sample_checkpoint)

        assert Path(path).exists()
        assert "test-cp-123" in path

    @pytest.mark.asyncio
    async def test_load_checkpoint(self, store, sample_checkpoint):
        """Test loading checkpoint from file."""
        await store.save(sample_checkpoint)

        loaded = await store.load(sample_checkpoint.checkpoint_id)

        assert loaded is not None
        assert loaded.checkpoint_id == sample_checkpoint.checkpoint_id
        assert loaded.debate_id == sample_checkpoint.debate_id

    @pytest.mark.asyncio
    async def test_load_nonexistent_checkpoint(self, store):
        """Test loading non-existent checkpoint returns None."""
        loaded = await store.load("nonexistent-id")

        assert loaded is None

    @pytest.mark.asyncio
    async def test_delete_checkpoint(self, store, sample_checkpoint):
        """Test deleting checkpoint."""
        path = await store.save(sample_checkpoint)
        assert Path(path).exists()

        result = await store.delete(sample_checkpoint.checkpoint_id)

        assert result is True
        assert not Path(path).exists()

    @pytest.mark.asyncio
    async def test_delete_nonexistent_checkpoint(self, store):
        """Test deleting non-existent checkpoint returns False."""
        result = await store.delete("nonexistent-id")

        assert result is False

    @pytest.mark.asyncio
    async def test_list_checkpoints(self, store, sample_checkpoint):
        """Test listing checkpoints."""
        await store.save(sample_checkpoint)

        # Create another checkpoint
        cp2 = DebateCheckpoint(
            checkpoint_id="test-cp-456",
            debate_id="debate-test-1",
            task="Another test",
            current_round=2,
            total_rounds=3,
            phase="critique",
            messages=[],
            critiques=[],
            votes=[],
            agent_states=[],
        )
        await store.save(cp2)

        checkpoints = await store.list_checkpoints()

        assert len(checkpoints) >= 2

    @pytest.mark.asyncio
    async def test_list_checkpoints_by_debate_id(self, store, sample_checkpoint):
        """Test listing checkpoints filtered by debate_id."""
        await store.save(sample_checkpoint)

        # Create checkpoint for different debate
        cp_other = DebateCheckpoint(
            checkpoint_id="other-cp",
            debate_id="different-debate",
            task="Other task",
            current_round=1,
            total_rounds=3,
            phase="proposal",
            messages=[],
            critiques=[],
            votes=[],
            agent_states=[],
        )
        await store.save(cp_other)

        checkpoints = await store.list_checkpoints(debate_id="debate-test-1")

        assert all(cp["debate_id"] == "debate-test-1" for cp in checkpoints)

    @pytest.mark.asyncio
    async def test_compressed_save_load(self, compressed_store, sample_checkpoint):
        """Test save/load with compression."""
        path = await compressed_store.save(sample_checkpoint)

        assert path.endswith(".json.gz")

        loaded = await compressed_store.load(sample_checkpoint.checkpoint_id)

        assert loaded is not None
        assert loaded.checkpoint_id == sample_checkpoint.checkpoint_id

    @pytest.mark.asyncio
    async def test_sanitize_checkpoint_id_path_traversal(self, store):
        """Test checkpoint ID sanitization prevents path traversal."""
        # Attempt path traversal
        sanitized = store._sanitize_checkpoint_id("../../../etc/passwd")

        # Should not contain path separators
        assert "/" not in sanitized
        assert "\\" not in sanitized
        assert ".." not in sanitized

    @pytest.mark.asyncio
    async def test_sanitize_checkpoint_id_special_chars(self, store):
        """Test checkpoint ID sanitization removes special characters."""
        sanitized = store._sanitize_checkpoint_id("test<>id|with*special")

        # Should only contain safe characters
        assert all(c.isalnum() or c in "_-" for c in sanitized)


# =============================================================================
# Safe Checkpoint ID Pattern Tests
# =============================================================================


class TestSafeCheckpointIdPattern:
    """Test the SAFE_CHECKPOINT_ID regex pattern."""

    def test_valid_checkpoint_ids(self):
        """Test valid checkpoint IDs match pattern."""
        valid_ids = [
            "checkpoint-123",
            "cp_456",
            "a1b2c3",
            "my-checkpoint-id",
            "UPPERCASE",
            "MixedCase123",
        ]

        for id_ in valid_ids:
            assert SAFE_CHECKPOINT_ID.match(id_) is not None, f"{id_} should be valid"

    def test_invalid_checkpoint_ids(self):
        """Test invalid checkpoint IDs don't match pattern."""
        invalid_ids = [
            "",  # Empty
            "-starts-with-dash",  # Starts with dash
            "_starts-with-underscore",  # Starts with underscore
            "../path/traversal",  # Path traversal
            "has spaces",  # Spaces
            "has<special>chars",  # Special characters
        ]

        for id_ in invalid_ids:
            assert SAFE_CHECKPOINT_ID.match(id_) is None, f"{id_} should be invalid"

    def test_max_length(self):
        """Test maximum length enforcement."""
        # 128 characters should be valid
        long_valid = "a" * 128
        assert SAFE_CHECKPOINT_ID.match(long_valid) is not None

        # 129 characters should be invalid (pattern limits to 128)
        too_long = "a" * 129
        assert SAFE_CHECKPOINT_ID.match(too_long) is None
