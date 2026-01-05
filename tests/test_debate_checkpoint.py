"""
Tests for the Incremental Consensus Checkpointing system.

Tests cover:
- CheckpointStatus enum
- AgentState dataclass
- DebateCheckpoint (creation, serialization, integrity)
- FileCheckpointStore (save, load, list, delete, path security)
- CheckpointConfig
- CheckpointManager (creation, resumption, intervention, cleanup)
- CheckpointWebhook
- Security tests (path traversal prevention)
"""

import asyncio
import json
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch

import pytest

from aragora.debate.checkpoint import (
    CheckpointStatus,
    AgentState,
    DebateCheckpoint,
    ResumedDebate,
    CheckpointStore,
    FileCheckpointStore,
    GitCheckpointStore,
    CheckpointConfig,
    CheckpointManager,
    CheckpointWebhook,
    SAFE_CHECKPOINT_ID,
)
from aragora.core import Message, Vote


class TestCheckpointStatusEnum:
    """Tests for CheckpointStatus enumeration."""

    def test_all_status_values_defined(self):
        """Verify all expected status values exist."""
        expected = ["creating", "complete", "resuming", "corrupted", "expired"]
        actual = [s.value for s in CheckpointStatus]
        assert sorted(expected) == sorted(actual)

    def test_status_values(self):
        """Test specific status values."""
        assert CheckpointStatus.CREATING.value == "creating"
        assert CheckpointStatus.COMPLETE.value == "complete"
        assert CheckpointStatus.RESUMING.value == "resuming"
        assert CheckpointStatus.CORRUPTED.value == "corrupted"
        assert CheckpointStatus.EXPIRED.value == "expired"


class TestSafeCheckpointIdPattern:
    """Tests for the safe checkpoint ID regex pattern."""

    def test_valid_checkpoint_ids(self):
        """Valid checkpoint IDs should match pattern."""
        valid_ids = [
            "cp-abc-001-def4",
            "checkpoint123",
            "a",
            "test_checkpoint",
            "test-checkpoint-2026",
        ]
        for id in valid_ids:
            assert SAFE_CHECKPOINT_ID.match(id), f"{id} should be valid"

    def test_invalid_checkpoint_ids(self):
        """Invalid checkpoint IDs should not match pattern."""
        invalid_ids = [
            "../../../etc/passwd",
            "path/to/file",
            "checkpoint;rm -rf /",
            "-starts-with-dash",
            "_starts_with_underscore",
            "",
            "a" * 200,  # Too long
        ]
        for id in invalid_ids:
            assert not SAFE_CHECKPOINT_ID.match(id), f"{id} should be invalid"


class TestAgentState:
    """Tests for AgentState dataclass."""

    def test_agent_state_creation(self):
        """Agent state should be created with required fields."""
        state = AgentState(
            agent_name="claude",
            agent_model="claude-3-opus",
            agent_role="proposer",
            system_prompt="Be helpful",
            stance="neutral",
        )
        assert state.agent_name == "claude"
        assert state.agent_model == "claude-3-opus"
        assert state.memory_snapshot is None

    def test_agent_state_with_memory(self):
        """Agent state should accept memory snapshot."""
        state = AgentState(
            agent_name="claude",
            agent_model="claude-3-opus",
            agent_role="proposer",
            system_prompt="Be helpful",
            stance="neutral",
            memory_snapshot={"key": "value"},
        )
        assert state.memory_snapshot == {"key": "value"}


class TestDebateCheckpoint:
    """Tests for DebateCheckpoint dataclass."""

    @pytest.fixture
    def sample_checkpoint(self):
        """Create a sample checkpoint for testing."""
        return DebateCheckpoint(
            checkpoint_id="cp-test-001-abc",
            debate_id="debate-123",
            task="Test debate task",
            current_round=3,
            total_rounds=5,
            phase="critique",
            messages=[{"agent": "claude", "content": "Hello"}],
            critiques=[],
            votes=[],
            agent_states=[
                AgentState(
                    agent_name="claude",
                    agent_model="claude-3-opus",
                    agent_role="proposer",
                    system_prompt="Be helpful",
                    stance="pro",
                )
            ],
        )

    def test_checkpoint_creation(self, sample_checkpoint):
        """Checkpoint should be created with correct fields."""
        assert sample_checkpoint.checkpoint_id == "cp-test-001-abc"
        assert sample_checkpoint.debate_id == "debate-123"
        assert sample_checkpoint.current_round == 3
        assert sample_checkpoint.status == CheckpointStatus.COMPLETE

    def test_checksum_computed(self, sample_checkpoint):
        """Checksum should be computed on creation."""
        assert sample_checkpoint.checksum != ""
        assert len(sample_checkpoint.checksum) == 16

    def test_verify_integrity_valid(self, sample_checkpoint):
        """verify_integrity should return True for valid checkpoint."""
        assert sample_checkpoint.verify_integrity() is True

    def test_verify_integrity_corrupted(self, sample_checkpoint):
        """verify_integrity should return False for modified checkpoint."""
        sample_checkpoint.current_round = 99  # Modify without updating checksum
        assert sample_checkpoint.verify_integrity() is False

    def test_to_dict_serialization(self, sample_checkpoint):
        """to_dict should serialize all important fields."""
        d = sample_checkpoint.to_dict()

        assert d["checkpoint_id"] == "cp-test-001-abc"
        assert d["debate_id"] == "debate-123"
        assert d["current_round"] == 3
        assert d["total_rounds"] == 5
        assert d["phase"] == "critique"
        assert len(d["agent_states"]) == 1
        assert d["status"] == "complete"
        assert "checksum" in d

    def test_from_dict_deserialization(self, sample_checkpoint):
        """from_dict should restore checkpoint from dictionary."""
        d = sample_checkpoint.to_dict()
        restored = DebateCheckpoint.from_dict(d)

        assert restored.checkpoint_id == sample_checkpoint.checkpoint_id
        assert restored.debate_id == sample_checkpoint.debate_id
        assert restored.current_round == sample_checkpoint.current_round
        assert len(restored.agent_states) == 1
        assert restored.agent_states[0].agent_name == "claude"

    def test_to_dict_from_dict_roundtrip(self, sample_checkpoint):
        """Checkpoint should survive to_dict/from_dict roundtrip."""
        d = sample_checkpoint.to_dict()
        restored = DebateCheckpoint.from_dict(d)

        assert restored.verify_integrity()
        assert restored.checksum == sample_checkpoint.checksum

    def test_intervention_notes(self, sample_checkpoint):
        """Checkpoint should support intervention notes."""
        sample_checkpoint.pending_intervention = True
        sample_checkpoint.intervention_notes.append("[human] Please review")

        d = sample_checkpoint.to_dict()
        restored = DebateCheckpoint.from_dict(d)

        assert restored.pending_intervention is True
        assert "[human] Please review" in restored.intervention_notes


class TestFileCheckpointStore:
    """Tests for FileCheckpointStore."""

    @pytest.fixture
    def temp_store(self, temp_dir):
        """Create a temporary checkpoint store."""
        return FileCheckpointStore(base_dir=str(temp_dir), compress=False)

    @pytest.fixture
    def sample_checkpoint(self):
        """Create a sample checkpoint for testing."""
        return DebateCheckpoint(
            checkpoint_id="cp-test-001-abc",
            debate_id="debate-123",
            task="Test task",
            current_round=1,
            total_rounds=5,
            phase="proposal",
            messages=[],
            critiques=[],
            votes=[],
            agent_states=[],
        )

    @pytest.mark.asyncio
    async def test_save_and_load(self, temp_store, sample_checkpoint):
        """Save and load should roundtrip correctly."""
        path = await temp_store.save(sample_checkpoint)
        assert path.endswith(".json")

        loaded = await temp_store.load(sample_checkpoint.checkpoint_id)
        assert loaded is not None
        assert loaded.checkpoint_id == sample_checkpoint.checkpoint_id
        assert loaded.debate_id == sample_checkpoint.debate_id

    @pytest.mark.asyncio
    async def test_load_nonexistent(self, temp_store):
        """Load of nonexistent checkpoint should return None."""
        loaded = await temp_store.load("nonexistent-checkpoint")
        assert loaded is None

    @pytest.mark.asyncio
    async def test_list_checkpoints(self, temp_store, sample_checkpoint):
        """list_checkpoints should return saved checkpoints."""
        await temp_store.save(sample_checkpoint)

        checkpoints = await temp_store.list_checkpoints()

        assert len(checkpoints) >= 1
        assert any(cp["checkpoint_id"] == sample_checkpoint.checkpoint_id for cp in checkpoints)

    @pytest.mark.asyncio
    async def test_list_checkpoints_by_debate(self, temp_store):
        """list_checkpoints should filter by debate_id."""
        cp1 = DebateCheckpoint(
            checkpoint_id="cp-d1-001",
            debate_id="debate-1",
            task="Task 1",
            current_round=1,
            total_rounds=5,
            phase="proposal",
            messages=[],
            critiques=[],
            votes=[],
            agent_states=[],
        )
        cp2 = DebateCheckpoint(
            checkpoint_id="cp-d2-001",
            debate_id="debate-2",
            task="Task 2",
            current_round=1,
            total_rounds=5,
            phase="proposal",
            messages=[],
            critiques=[],
            votes=[],
            agent_states=[],
        )
        await temp_store.save(cp1)
        await temp_store.save(cp2)

        debate_1_checkpoints = await temp_store.list_checkpoints(debate_id="debate-1")

        assert len(debate_1_checkpoints) == 1
        assert debate_1_checkpoints[0]["debate_id"] == "debate-1"

    @pytest.mark.asyncio
    async def test_delete(self, temp_store, sample_checkpoint):
        """delete should remove checkpoint."""
        await temp_store.save(sample_checkpoint)

        result = await temp_store.delete(sample_checkpoint.checkpoint_id)
        assert result is True

        loaded = await temp_store.load(sample_checkpoint.checkpoint_id)
        assert loaded is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, temp_store):
        """delete of nonexistent checkpoint should return False."""
        result = await temp_store.delete("nonexistent")
        assert result is False

    def test_path_traversal_prevention(self, temp_store):
        """Path traversal attacks should be blocked."""
        malicious_ids = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32",
            "checkpoint/../../../secret",
        ]
        for malicious_id in malicious_ids:
            # The sanitization should prevent any path traversal
            sanitized = temp_store._sanitize_checkpoint_id(malicious_id)
            path = temp_store._get_path(sanitized)
            # Path should be within base_dir
            assert str(temp_store.base_dir) in str(path.resolve())

    def test_sanitize_removes_special_chars(self, temp_store):
        """Sanitization should remove special characters."""
        dangerous = "checkpoint;rm -rf /"
        sanitized = temp_store._sanitize_checkpoint_id(dangerous)
        assert ";" not in sanitized
        assert "/" not in sanitized
        assert " " not in sanitized

    def test_empty_id_raises(self, temp_store):
        """Empty checkpoint ID should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid checkpoint ID"):
            temp_store._sanitize_checkpoint_id("")


class TestFileCheckpointStoreCompressed:
    """Tests for compressed FileCheckpointStore."""

    @pytest.fixture
    def compressed_store(self, temp_dir):
        """Create a compressed checkpoint store."""
        return FileCheckpointStore(base_dir=str(temp_dir), compress=True)

    @pytest.mark.asyncio
    async def test_compressed_save_load(self, compressed_store):
        """Compressed checkpoint should save and load correctly."""
        checkpoint = DebateCheckpoint(
            checkpoint_id="cp-compressed",
            debate_id="debate-1",
            task="Test",
            current_round=1,
            total_rounds=5,
            phase="proposal",
            messages=[{"agent": "test", "content": "A" * 1000}],  # Large content
            critiques=[],
            votes=[],
            agent_states=[],
        )

        path = await compressed_store.save(checkpoint)
        assert path.endswith(".json.gz")

        loaded = await compressed_store.load("cp-compressed")
        assert loaded is not None
        assert loaded.messages[0]["content"] == "A" * 1000


class TestCheckpointConfig:
    """Tests for CheckpointConfig dataclass."""

    def test_default_config(self):
        """Default config should have sensible values."""
        config = CheckpointConfig()
        assert config.interval_rounds == 1
        assert config.interval_seconds == 300.0
        assert config.max_checkpoints == 10
        assert config.expiry_hours == 72.0
        assert config.compress is True
        assert config.auto_cleanup is True

    def test_custom_config(self):
        """Custom config should override defaults."""
        config = CheckpointConfig(
            interval_rounds=5,
            max_checkpoints=3,
            expiry_hours=24.0,
        )
        assert config.interval_rounds == 5
        assert config.max_checkpoints == 3
        assert config.expiry_hours == 24.0


class TestCheckpointManager:
    """Tests for CheckpointManager."""

    @pytest.fixture
    def temp_store(self, temp_dir):
        """Create a temporary checkpoint store."""
        return FileCheckpointStore(base_dir=str(temp_dir), compress=False)

    @pytest.fixture
    def manager(self, temp_store):
        """Create a checkpoint manager."""
        config = CheckpointConfig(interval_rounds=2, max_checkpoints=3)
        return CheckpointManager(store=temp_store, config=config)

    def test_should_checkpoint_by_round(self, manager):
        """should_checkpoint should trigger on round interval."""
        assert manager.should_checkpoint("debate-1", current_round=0) is True
        assert manager.should_checkpoint("debate-1", current_round=1) is False
        assert manager.should_checkpoint("debate-1", current_round=2) is True
        assert manager.should_checkpoint("debate-1", current_round=3) is False
        assert manager.should_checkpoint("debate-1", current_round=4) is True

    @pytest.mark.asyncio
    async def test_create_checkpoint(self, manager):
        """create_checkpoint should save checkpoint."""
        # Create mock messages
        messages = [
            Message(
                role="assistant",
                agent="claude",
                content="Test message",
                round=1,
            )
        ]

        # Create mock agents
        mock_agent = Mock()
        mock_agent.name = "claude"
        mock_agent.model = "claude-3-opus"
        mock_agent.role = "proposer"
        mock_agent.system_prompt = "Be helpful"
        mock_agent.stance = "neutral"

        checkpoint = await manager.create_checkpoint(
            debate_id="debate-1",
            task="Test task",
            current_round=1,
            total_rounds=5,
            phase="proposal",
            messages=messages,
            critiques=[],
            votes=[],
            agents=[mock_agent],
        )

        assert checkpoint is not None
        assert checkpoint.debate_id == "debate-1"
        assert len(checkpoint.messages) == 1
        assert len(checkpoint.agent_states) == 1

    @pytest.mark.asyncio
    async def test_resume_from_checkpoint(self, manager):
        """resume_from_checkpoint should restore state."""
        # Create a checkpoint first
        messages = [
            Message(
                role="assistant",
                agent="claude",
                content="Test",
                round=1,
            )
        ]
        mock_agent = Mock()
        mock_agent.name = "claude"
        mock_agent.model = "claude-3-opus"
        mock_agent.role = "proposer"
        mock_agent.system_prompt = "Be helpful"
        mock_agent.stance = "neutral"

        checkpoint = await manager.create_checkpoint(
            debate_id="debate-1",
            task="Test",
            current_round=1,
            total_rounds=5,
            phase="proposal",
            messages=messages,
            critiques=[],
            votes=[],
            agents=[mock_agent],
        )

        # Resume from checkpoint
        resumed = await manager.resume_from_checkpoint(
            checkpoint.checkpoint_id,
            resumed_by="test-user",
        )

        assert resumed is not None
        assert resumed.original_debate_id == "debate-1"
        assert len(resumed.messages) == 1
        assert resumed.resumed_by == "test-user"

    @pytest.mark.asyncio
    async def test_resume_nonexistent(self, manager):
        """resume_from_checkpoint should return None for nonexistent."""
        resumed = await manager.resume_from_checkpoint("nonexistent")
        assert resumed is None

    @pytest.mark.asyncio
    async def test_add_intervention(self, manager):
        """add_intervention should add note to checkpoint."""
        # Create checkpoint
        mock_agent = Mock()
        mock_agent.name = "claude"
        mock_agent.model = "claude-3-opus"
        mock_agent.role = "proposer"
        mock_agent.system_prompt = "Be helpful"
        mock_agent.stance = "neutral"

        checkpoint = await manager.create_checkpoint(
            debate_id="debate-1",
            task="Test",
            current_round=1,
            total_rounds=5,
            phase="proposal",
            messages=[],
            critiques=[],
            votes=[],
            agents=[mock_agent],
        )

        # Add intervention
        result = await manager.add_intervention(
            checkpoint.checkpoint_id,
            "Please review this debate",
            by="reviewer",
        )

        assert result is True

        # Verify intervention was added
        loaded = await manager.store.load(checkpoint.checkpoint_id)
        assert loaded.pending_intervention is True
        assert "[reviewer] Please review this debate" in loaded.intervention_notes

    @pytest.mark.asyncio
    async def test_list_debates_with_checkpoints(self, manager):
        """list_debates_with_checkpoints should group by debate."""
        mock_agent = Mock()
        mock_agent.name = "claude"
        mock_agent.model = "claude-3-opus"
        mock_agent.role = "proposer"
        mock_agent.system_prompt = "Be helpful"
        mock_agent.stance = "neutral"

        # Create checkpoints for two debates
        await manager.create_checkpoint(
            debate_id="debate-1", task="Task 1",
            current_round=1, total_rounds=5, phase="proposal",
            messages=[], critiques=[], votes=[], agents=[mock_agent],
        )
        await manager.create_checkpoint(
            debate_id="debate-1", task="Task 1",
            current_round=2, total_rounds=5, phase="critique",
            messages=[], critiques=[], votes=[], agents=[mock_agent],
        )
        await manager.create_checkpoint(
            debate_id="debate-2", task="Task 2",
            current_round=1, total_rounds=3, phase="proposal",
            messages=[], critiques=[], votes=[], agents=[mock_agent],
        )

        debates = await manager.list_debates_with_checkpoints()

        assert len(debates) == 2
        debate_1 = next(d for d in debates if d["debate_id"] == "debate-1")
        assert debate_1["checkpoint_count"] == 2
        assert debate_1["latest_round"] == 2


class TestCheckpointWebhook:
    """Tests for CheckpointWebhook."""

    def test_register_handler(self):
        """Handlers should be registered correctly."""
        webhook = CheckpointWebhook()

        @webhook.on_checkpoint
        def handler1(data):
            pass

        @webhook.on_resume
        def handler2(data):
            pass

        assert len(webhook.handlers["on_checkpoint"]) == 1
        assert len(webhook.handlers["on_resume"]) == 1

    @pytest.mark.asyncio
    async def test_emit_calls_handlers(self):
        """emit should call registered handlers."""
        webhook = CheckpointWebhook()
        called = []

        @webhook.on_checkpoint
        def handler(data):
            called.append(data)

        await webhook.emit("on_checkpoint", {"test": "data"})

        assert len(called) == 1
        assert called[0] == {"test": "data"}

    @pytest.mark.asyncio
    async def test_emit_calls_async_handlers(self):
        """emit should call async handlers."""
        webhook = CheckpointWebhook()
        called = []

        @webhook.on_checkpoint
        async def async_handler(data):
            called.append(data)

        await webhook.emit("on_checkpoint", {"async": True})

        assert len(called) == 1

    @pytest.mark.asyncio
    async def test_emit_handles_errors(self):
        """emit should not fail if handler raises."""
        webhook = CheckpointWebhook()

        @webhook.on_checkpoint
        def bad_handler(data):
            raise RuntimeError("Handler error")

        # Should not raise
        await webhook.emit("on_checkpoint", {"test": "data"})


class TestGitCheckpointStore:
    """Tests for GitCheckpointStore (without actual git)."""

    def test_safe_checkpoint_id_validation(self):
        """GitCheckpointStore should validate checkpoint IDs."""
        # Test the pattern directly since we can't easily mock git
        valid_ids = ["cp-abc-001", "test_checkpoint", "checkpoint123"]
        invalid_ids = ["../etc/passwd", "path/to/file", "checkpoint;rm"]

        for id in valid_ids:
            assert SAFE_CHECKPOINT_ID.match(id), f"{id} should be valid"

        for id in invalid_ids:
            assert not SAFE_CHECKPOINT_ID.match(id), f"{id} should be invalid"


class TestResumedDebate:
    """Tests for ResumedDebate dataclass."""

    def test_resumed_debate_creation(self):
        """ResumedDebate should be created with checkpoint context."""
        checkpoint = DebateCheckpoint(
            checkpoint_id="cp-test",
            debate_id="debate-1",
            task="Test",
            current_round=2,
            total_rounds=5,
            phase="critique",
            messages=[],
            critiques=[],
            votes=[],
            agent_states=[],
        )

        resumed = ResumedDebate(
            checkpoint=checkpoint,
            original_debate_id="debate-1",
            resumed_at=datetime.now().isoformat(),
            resumed_by="user",
            messages=[],
            votes=[],
        )

        assert resumed.original_debate_id == "debate-1"
        assert resumed.resumed_by == "user"
        assert resumed.context_drift_detected is False

    def test_resumed_debate_drift_detection(self):
        """ResumedDebate should track context drift."""
        checkpoint = DebateCheckpoint(
            checkpoint_id="cp-test",
            debate_id="debate-1",
            task="Test",
            current_round=2,
            total_rounds=5,
            phase="critique",
            messages=[],
            critiques=[],
            votes=[],
            agent_states=[],
        )

        resumed = ResumedDebate(
            checkpoint=checkpoint,
            original_debate_id="debate-1",
            resumed_at=datetime.now().isoformat(),
            resumed_by="user",
            messages=[],
            votes=[],
            context_drift_detected=True,
            drift_notes=["Agent behavior has changed"],
        )

        assert resumed.context_drift_detected is True
        assert "Agent behavior has changed" in resumed.drift_notes
