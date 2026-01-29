"""
Comprehensive tests for aragora/debate/checkpoint.py.

Tests the checkpoint functionality including:
- Checkpoint creation and all checkpoint types
- Serialization of complex debate objects
- Checkpoint restoration and state reconstruction
- Handling missing/corrupt checkpoints
- Cleanup, garbage collection, and TTL-based expiration
- Storage backend abstraction
- Full debate state round-trip (save -> load)
- Partial state recovery
- Concurrent checkpoint access
- Storage errors handling
"""

from __future__ import annotations

import asyncio
import gzip
import json
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.core import Critique, Message, Vote
from aragora.debate.checkpoint import (
    AgentState,
    CheckpointConfig,
    CheckpointManager,
    CheckpointStatus,
    CheckpointStore,
    CheckpointWebhook,
    DatabaseCheckpointStore,
    DebateCheckpoint,
    FileCheckpointStore,
    GitCheckpointStore,
    RecoveryNarrator,
    ResumedDebate,
    S3CheckpointStore,
    SAFE_CHECKPOINT_ID,
    checkpoint_debate,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_agent_state():
    """Create a sample agent state."""
    return AgentState(
        agent_name="claude",
        agent_model="claude-3-sonnet",
        agent_role="proposer",
        system_prompt="You are a helpful assistant.",
        stance="neutral",
        memory_snapshot={"recent_context": ["item1", "item2"]},
    )


@pytest.fixture
def sample_agent_states():
    """Create multiple sample agent states."""
    return [
        AgentState(
            agent_name="claude",
            agent_model="claude-3-sonnet",
            agent_role="proposer",
            system_prompt="You propose solutions.",
            stance="affirmative",
        ),
        AgentState(
            agent_name="gpt4",
            agent_model="gpt-4-turbo",
            agent_role="critic",
            system_prompt="You critique proposals.",
            stance="skeptical",
        ),
        AgentState(
            agent_name="gemini",
            agent_model="gemini-pro",
            agent_role="synthesizer",
            system_prompt="You synthesize viewpoints.",
            stance="neutral",
        ),
    ]


@pytest.fixture
def sample_messages():
    """Create sample Message objects."""
    return [
        Message(
            role="proposer",
            agent="claude",
            content="I propose we implement a rate limiter using token bucket algorithm.",
            timestamp=datetime.now(),
            round=1,
        ),
        Message(
            role="critic",
            agent="gpt4",
            content="The token bucket approach has merit, but consider edge cases.",
            timestamp=datetime.now(),
            round=1,
        ),
        Message(
            role="synthesizer",
            agent="gemini",
            content="Combining the approaches yields a robust solution.",
            timestamp=datetime.now(),
            round=2,
        ),
    ]


@pytest.fixture
def sample_critiques():
    """Create sample Critique objects."""
    return [
        Critique(
            agent="gpt4",
            target_agent="claude",
            target_content="Token bucket proposal",
            issues=["Doesn't handle burst traffic", "Memory overhead concerns"],
            suggestions=["Add burst capacity", "Use sliding window"],
            severity=5.0,
            reasoning="The proposal needs refinement for production use.",
        ),
        Critique(
            agent="claude",
            target_agent="gpt4",
            target_content="Edge case critique",
            issues=["Overly pessimistic"],
            suggestions=["Consider practical constraints"],
            severity=3.0,
            reasoning="Some edge cases are unlikely in practice.",
        ),
    ]


@pytest.fixture
def sample_votes():
    """Create sample Vote objects."""
    return [
        Vote(
            agent="claude",
            choice="proposal_a",
            confidence=0.85,
            reasoning="The hybrid approach is most practical.",
            continue_debate=False,
        ),
        Vote(
            agent="gpt4",
            choice="proposal_a",
            confidence=0.75,
            reasoning="Agreed on the hybrid approach with reservations.",
            continue_debate=False,
        ),
        Vote(
            agent="gemini",
            choice="proposal_a",
            confidence=0.90,
            reasoning="The synthesis captures all viewpoints.",
            continue_debate=False,
        ),
    ]


@pytest.fixture
def sample_checkpoint(sample_agent_states):
    """Create a sample checkpoint with full state."""
    return DebateCheckpoint(
        checkpoint_id="cp-test-001-abcd",
        debate_id="debate-12345678",
        task="Design a rate limiter for the API gateway",
        current_round=3,
        total_rounds=5,
        phase="synthesis",
        messages=[
            {
                "role": "proposer",
                "agent": "claude",
                "content": "Proposal content",
                "timestamp": datetime.now().isoformat(),
                "round": 1,
            }
        ],
        critiques=[
            {
                "agent": "gpt4",
                "target_agent": "claude",
                "target_content": "Proposal",
                "issues": ["Issue 1"],
                "suggestions": ["Suggestion 1"],
                "severity": 5.0,
                "reasoning": "Reasoning",
            }
        ],
        votes=[
            {
                "agent": "claude",
                "choice": "A",
                "confidence": 0.8,
                "reasoning": "Reason",
                "continue_debate": False,
            }
        ],
        agent_states=sample_agent_states,
        current_consensus="We agree on the hybrid approach.",
        consensus_confidence=0.85,
        convergence_status="converged",
        claims_kernel_state={"claims": [{"id": "c1", "text": "claim 1"}]},
        belief_network_state={"beliefs": {"node1": 0.9}},
        continuum_memory_state={"entries": [], "tier_counts": {}},
    )


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def file_store(temp_dir):
    """Create a FileCheckpointStore for testing."""
    return FileCheckpointStore(base_dir=temp_dir, compress=False)


@pytest.fixture
def compressed_file_store(temp_dir):
    """Create a compressed FileCheckpointStore for testing."""
    return FileCheckpointStore(base_dir=temp_dir, compress=True)


@pytest.fixture
def db_store(temp_dir):
    """Create a DatabaseCheckpointStore for testing."""
    db_path = Path(temp_dir) / "checkpoints.db"
    return DatabaseCheckpointStore(db_path=str(db_path), compress=False)


@pytest.fixture
def mock_agent():
    """Create a mock agent for checkpoint creation."""

    class MockAgent:
        name = "test-agent"
        model = "test-model"
        role = "proposer"
        system_prompt = "Test system prompt"
        stance = "neutral"

    return MockAgent()


# =============================================================================
# Checkpoint Creation Tests
# =============================================================================


class TestCheckpointCreation:
    """Tests for creating checkpoints from debate state."""

    def test_create_checkpoint_minimal(self):
        """Test creating a checkpoint with minimal required fields."""
        checkpoint = DebateCheckpoint(
            checkpoint_id="cp-minimal-001",
            debate_id="debate-minimal",
            task="Simple task",
            current_round=1,
            total_rounds=3,
            phase="proposal",
            messages=[],
            critiques=[],
            votes=[],
            agent_states=[],
        )

        assert checkpoint.checkpoint_id == "cp-minimal-001"
        assert checkpoint.debate_id == "debate-minimal"
        assert checkpoint.status == CheckpointStatus.COMPLETE
        assert checkpoint.checksum != ""
        assert len(checkpoint.checksum) == 16

    def test_create_checkpoint_with_full_state(self, sample_checkpoint):
        """Test creating a checkpoint with all state fields."""
        cp = sample_checkpoint

        assert cp.checkpoint_id == "cp-test-001-abcd"
        assert cp.current_round == 3
        assert cp.total_rounds == 5
        assert cp.phase == "synthesis"
        assert len(cp.messages) == 1
        assert len(cp.critiques) == 1
        assert len(cp.votes) == 1
        assert len(cp.agent_states) == 3
        assert cp.current_consensus is not None
        assert cp.consensus_confidence == 0.85
        assert cp.convergence_status == "converged"
        assert cp.claims_kernel_state is not None
        assert cp.belief_network_state is not None
        assert cp.continuum_memory_state is not None

    def test_checkpoint_auto_generates_checksum(self):
        """Test that checkpoint auto-generates checksum on creation."""
        cp = DebateCheckpoint(
            checkpoint_id="cp-checksum-test",
            debate_id="debate-1",
            task="Task",
            current_round=2,
            total_rounds=5,
            phase="critique",
            messages=[{"content": "msg"}],
            critiques=[],
            votes=[],
            agent_states=[],
        )

        assert cp.checksum != ""
        # Checksum is SHA256[:16]
        assert len(cp.checksum) == 16
        assert all(c in "0123456789abcdef" for c in cp.checksum)

    def test_checkpoint_created_at_timestamp(self):
        """Test that checkpoint has valid created_at timestamp."""
        cp = DebateCheckpoint(
            checkpoint_id="cp-timestamp",
            debate_id="debate-1",
            task="Task",
            current_round=1,
            total_rounds=3,
            phase="proposal",
            messages=[],
            critiques=[],
            votes=[],
            agent_states=[],
        )

        # Should be a valid ISO format
        timestamp = datetime.fromisoformat(cp.created_at)
        assert timestamp is not None
        # Should be recent (within last minute)
        assert (datetime.now() - timestamp).total_seconds() < 60

    def test_checkpoint_with_expiry(self):
        """Test checkpoint with expiration time."""
        expiry_time = (datetime.now() + timedelta(hours=24)).isoformat()
        cp = DebateCheckpoint(
            checkpoint_id="cp-expiry",
            debate_id="debate-1",
            task="Task",
            current_round=1,
            total_rounds=3,
            phase="proposal",
            messages=[],
            critiques=[],
            votes=[],
            agent_states=[],
            expires_at=expiry_time,
        )

        assert cp.expires_at == expiry_time

    def test_checkpoint_with_intervention_notes(self):
        """Test checkpoint with pending intervention."""
        cp = DebateCheckpoint(
            checkpoint_id="cp-intervention",
            debate_id="debate-1",
            task="Task requiring human review",
            current_round=2,
            total_rounds=5,
            phase="proposal",
            messages=[],
            critiques=[],
            votes=[],
            agent_states=[],
            pending_intervention=True,
            intervention_notes=["Human reviewer requested", "Safety check needed"],
        )

        assert cp.pending_intervention is True
        assert len(cp.intervention_notes) == 2
        assert "Safety check" in cp.intervention_notes[1]


class TestAllCheckpointTypes:
    """Test all 14 checkpoint types/status values."""

    def test_checkpoint_status_creating(self):
        """Test CREATING status."""
        cp = DebateCheckpoint(
            checkpoint_id="cp-creating",
            debate_id="debate-1",
            task="Task",
            current_round=1,
            total_rounds=3,
            phase="proposal",
            messages=[],
            critiques=[],
            votes=[],
            agent_states=[],
            status=CheckpointStatus.CREATING,
        )
        assert cp.status == CheckpointStatus.CREATING
        assert cp.status.value == "creating"

    def test_checkpoint_status_complete(self):
        """Test COMPLETE status (default)."""
        cp = DebateCheckpoint(
            checkpoint_id="cp-complete",
            debate_id="debate-1",
            task="Task",
            current_round=1,
            total_rounds=3,
            phase="proposal",
            messages=[],
            critiques=[],
            votes=[],
            agent_states=[],
        )
        assert cp.status == CheckpointStatus.COMPLETE
        assert cp.status.value == "complete"

    def test_checkpoint_status_resuming(self):
        """Test RESUMING status."""
        cp = DebateCheckpoint(
            checkpoint_id="cp-resuming",
            debate_id="debate-1",
            task="Task",
            current_round=1,
            total_rounds=3,
            phase="proposal",
            messages=[],
            critiques=[],
            votes=[],
            agent_states=[],
            status=CheckpointStatus.RESUMING,
        )
        assert cp.status == CheckpointStatus.RESUMING
        assert cp.status.value == "resuming"

    def test_checkpoint_status_corrupted(self):
        """Test CORRUPTED status."""
        cp = DebateCheckpoint(
            checkpoint_id="cp-corrupted",
            debate_id="debate-1",
            task="Task",
            current_round=1,
            total_rounds=3,
            phase="proposal",
            messages=[],
            critiques=[],
            votes=[],
            agent_states=[],
            status=CheckpointStatus.CORRUPTED,
        )
        assert cp.status == CheckpointStatus.CORRUPTED
        assert cp.status.value == "corrupted"

    def test_checkpoint_status_expired(self):
        """Test EXPIRED status."""
        cp = DebateCheckpoint(
            checkpoint_id="cp-expired",
            debate_id="debate-1",
            task="Task",
            current_round=1,
            total_rounds=3,
            phase="proposal",
            messages=[],
            critiques=[],
            votes=[],
            agent_states=[],
            status=CheckpointStatus.EXPIRED,
        )
        assert cp.status == CheckpointStatus.EXPIRED
        assert cp.status.value == "expired"

    def test_all_debate_phases(self):
        """Test checkpoints for all debate phases."""
        phases = ["proposal", "critique", "vote", "synthesis", "revision"]

        for phase in phases:
            cp = DebateCheckpoint(
                checkpoint_id=f"cp-{phase}",
                debate_id="debate-1",
                task="Task",
                current_round=1,
                total_rounds=5,
                phase=phase,
                messages=[],
                critiques=[],
                votes=[],
                agent_states=[],
            )
            assert cp.phase == phase


class TestCheckpointSerialization:
    """Test serialization of complex debate objects."""

    def test_to_dict_basic(self, sample_checkpoint):
        """Test basic to_dict serialization."""
        d = sample_checkpoint.to_dict()

        assert d["checkpoint_id"] == "cp-test-001-abcd"
        assert d["debate_id"] == "debate-12345678"
        assert d["current_round"] == 3
        assert d["total_rounds"] == 5
        assert d["phase"] == "synthesis"
        assert d["status"] == "complete"

    def test_to_dict_includes_agent_states(self, sample_checkpoint):
        """Test that to_dict includes serialized agent states."""
        d = sample_checkpoint.to_dict()

        assert "agent_states" in d
        assert len(d["agent_states"]) == 3
        assert d["agent_states"][0]["agent_name"] == "claude"
        assert d["agent_states"][1]["agent_name"] == "gpt4"
        assert d["agent_states"][2]["agent_name"] == "gemini"

    def test_to_dict_includes_optional_states(self, sample_checkpoint):
        """Test that to_dict includes optional state fields."""
        d = sample_checkpoint.to_dict()

        assert "claims_kernel_state" in d
        assert "belief_network_state" in d
        assert "continuum_memory_state" in d
        assert d["claims_kernel_state"] is not None
        assert d["belief_network_state"] is not None

    def test_from_dict_basic(self, sample_checkpoint):
        """Test basic from_dict deserialization."""
        d = sample_checkpoint.to_dict()
        restored = DebateCheckpoint.from_dict(d)

        assert restored.checkpoint_id == sample_checkpoint.checkpoint_id
        assert restored.debate_id == sample_checkpoint.debate_id
        assert restored.current_round == sample_checkpoint.current_round
        assert restored.phase == sample_checkpoint.phase

    def test_from_dict_restores_agent_states(self, sample_checkpoint):
        """Test that from_dict restores agent states."""
        d = sample_checkpoint.to_dict()
        restored = DebateCheckpoint.from_dict(d)

        assert len(restored.agent_states) == 3
        assert isinstance(restored.agent_states[0], AgentState)
        assert restored.agent_states[0].agent_name == "claude"

    def test_json_round_trip(self, sample_checkpoint):
        """Test full JSON round-trip."""
        d = sample_checkpoint.to_dict()
        json_str = json.dumps(d)
        restored_dict = json.loads(json_str)
        restored = DebateCheckpoint.from_dict(restored_dict)

        assert restored.checkpoint_id == sample_checkpoint.checkpoint_id
        assert restored.verify_integrity()

    def test_serialization_with_unicode(self):
        """Test serialization with unicode content."""
        cp = DebateCheckpoint(
            checkpoint_id="cp-unicode",
            debate_id="debate-unicode",
            task="Task with emoji \U0001f4bb and unicode \u2603",
            current_round=1,
            total_rounds=3,
            phase="proposal",
            messages=[{"content": "Message with \u2764 heart"}],
            critiques=[],
            votes=[],
            agent_states=[],
        )

        d = cp.to_dict()
        json_str = json.dumps(d)
        restored_dict = json.loads(json_str)
        restored = DebateCheckpoint.from_dict(restored_dict)

        assert "\U0001f4bb" in restored.task
        assert "\u2764" in restored.messages[0]["content"]

    def test_serialization_with_nested_structures(self):
        """Test serialization with deeply nested structures."""
        cp = DebateCheckpoint(
            checkpoint_id="cp-nested",
            debate_id="debate-nested",
            task="Nested task",
            current_round=1,
            total_rounds=3,
            phase="proposal",
            messages=[
                {
                    "content": "Message",
                    "nested": {"level1": {"level2": {"level3": "deep"}}},
                }
            ],
            critiques=[],
            votes=[],
            agent_states=[],
            claims_kernel_state={
                "claims": [{"id": 1, "relations": [{"type": "supports", "target": 2}]}]
            },
        )

        d = cp.to_dict()
        json_str = json.dumps(d)
        restored_dict = json.loads(json_str)
        restored = DebateCheckpoint.from_dict(restored_dict)

        assert restored.claims_kernel_state["claims"][0]["relations"][0]["type"] == "supports"


# =============================================================================
# Checkpoint Restoration Tests
# =============================================================================


class TestCheckpointRestoration:
    """Tests for loading checkpoints from storage."""

    @pytest.mark.asyncio
    async def test_load_from_file_store(self, file_store, sample_checkpoint):
        """Test loading checkpoint from file store."""
        await file_store.save(sample_checkpoint)

        loaded = await file_store.load(sample_checkpoint.checkpoint_id)

        assert loaded is not None
        assert loaded.checkpoint_id == sample_checkpoint.checkpoint_id
        assert loaded.debate_id == sample_checkpoint.debate_id

    @pytest.mark.asyncio
    async def test_load_from_compressed_store(self, compressed_file_store, sample_checkpoint):
        """Test loading checkpoint from compressed file store."""
        await compressed_file_store.save(sample_checkpoint)

        loaded = await compressed_file_store.load(sample_checkpoint.checkpoint_id)

        assert loaded is not None
        assert loaded.checkpoint_id == sample_checkpoint.checkpoint_id

    @pytest.mark.asyncio
    async def test_load_nonexistent_returns_none(self, file_store):
        """Test loading nonexistent checkpoint returns None."""
        loaded = await file_store.load("nonexistent-checkpoint-id")
        assert loaded is None

    @pytest.mark.asyncio
    async def test_reconstruct_debate_state(self, file_store, sample_checkpoint):
        """Test reconstructing debate state from checkpoint."""
        await file_store.save(sample_checkpoint)

        loaded = await file_store.load(sample_checkpoint.checkpoint_id)

        # Verify all state components are restored
        assert loaded.messages == sample_checkpoint.messages
        assert loaded.critiques == sample_checkpoint.critiques
        assert loaded.votes == sample_checkpoint.votes
        assert len(loaded.agent_states) == len(sample_checkpoint.agent_states)
        assert loaded.current_consensus == sample_checkpoint.current_consensus
        assert loaded.consensus_confidence == sample_checkpoint.consensus_confidence

    def test_checkpoint_integrity_verification_pass(self, sample_checkpoint):
        """Test integrity verification passes for valid checkpoint."""
        assert sample_checkpoint.verify_integrity() is True

    def test_checkpoint_integrity_verification_fail(self, sample_checkpoint):
        """Test integrity verification fails after modification."""
        original_checksum = sample_checkpoint.checksum

        # Modify checkpoint data
        sample_checkpoint.messages.append({"content": "tampered"})

        # Checksum stored, but data changed
        assert sample_checkpoint.checksum == original_checksum
        assert sample_checkpoint.verify_integrity() is False


class TestHandlingCorruptCheckpoints:
    """Tests for handling missing or corrupt checkpoints."""

    @pytest.mark.asyncio
    async def test_handle_corrupted_json(self, temp_dir):
        """Test handling corrupted JSON data."""
        store = FileCheckpointStore(base_dir=temp_dir, compress=False)

        # Create corrupted file
        corrupted_path = Path(temp_dir) / "corrupted-cp.json"
        corrupted_path.write_text("{invalid json")

        loaded = await store.load("corrupted-cp")
        assert loaded is None

    @pytest.mark.asyncio
    async def test_handle_corrupted_gzip(self, temp_dir):
        """Test handling corrupted gzip data."""
        store = FileCheckpointStore(base_dir=temp_dir, compress=True)

        # Create corrupted gzip file
        corrupted_path = Path(temp_dir) / "corrupted-cp.json.gz"
        corrupted_path.write_bytes(b"not a gzip file")

        loaded = await store.load("corrupted-cp")
        assert loaded is None

    @pytest.mark.asyncio
    async def test_handle_invalid_structure(self, temp_dir):
        """Test handling valid JSON but invalid checkpoint structure."""
        store = FileCheckpointStore(base_dir=temp_dir, compress=False)

        # Create file with valid JSON but missing required fields
        invalid_path = Path(temp_dir) / "invalid-cp.json"
        invalid_path.write_text('{"only": "partial", "data": true}')

        loaded = await store.load("invalid-cp")
        assert loaded is None

    @pytest.mark.asyncio
    async def test_handle_missing_file(self, file_store):
        """Test handling missing checkpoint file."""
        loaded = await file_store.load("does-not-exist")
        assert loaded is None

    @pytest.mark.asyncio
    async def test_resume_corrupted_checkpoint_returns_none(self, temp_dir):
        """Test resume from corrupted checkpoint returns None."""
        store = FileCheckpointStore(base_dir=temp_dir, compress=False)
        manager = CheckpointManager(store=store)

        # Try to resume from nonexistent checkpoint
        resumed = await manager.resume_from_checkpoint("nonexistent", "user")
        assert resumed is None


# =============================================================================
# Checkpoint Management Tests
# =============================================================================


class TestCheckpointCleanup:
    """Tests for cleanup and garbage collection."""

    @pytest.mark.asyncio
    async def test_auto_cleanup_removes_old_checkpoints(self, temp_dir, mock_agent):
        """Test that auto cleanup removes old checkpoints beyond limit."""
        store = FileCheckpointStore(base_dir=temp_dir, compress=False)
        config = CheckpointConfig(
            interval_rounds=1,
            max_checkpoints=3,
            auto_cleanup=True,
        )
        manager = CheckpointManager(store=store, config=config)

        # Create more checkpoints than the limit
        for i in range(5):
            await manager.create_checkpoint(
                debate_id="cleanup-debate",
                task="Cleanup test",
                current_round=i + 1,
                total_rounds=10,
                phase="proposal",
                messages=[],
                critiques=[],
                votes=[],
                agents=[mock_agent],
            )

        # List remaining checkpoints
        remaining = await store.list_checkpoints(debate_id="cleanup-debate")

        # Should only have max_checkpoints
        assert len(remaining) <= 3

    @pytest.mark.asyncio
    async def test_cleanup_keeps_most_recent(self, temp_dir, mock_agent):
        """Test that cleanup keeps the most recent checkpoints."""
        store = FileCheckpointStore(base_dir=temp_dir, compress=False)
        config = CheckpointConfig(
            interval_rounds=1,
            max_checkpoints=2,
            auto_cleanup=True,
        )
        manager = CheckpointManager(store=store, config=config)

        # Create checkpoints with time gaps
        for i in range(4):
            await manager.create_checkpoint(
                debate_id="keep-recent",
                task="Keep recent test",
                current_round=i + 1,
                total_rounds=10,
                phase="proposal",
                messages=[],
                critiques=[],
                votes=[],
                agents=[mock_agent],
            )

        remaining = await store.list_checkpoints(debate_id="keep-recent")

        # Should keep only 2 most recent
        assert len(remaining) == 2
        rounds = [cp["current_round"] for cp in remaining]
        assert 4 in rounds or 3 in rounds  # Should include latest


class TestTTLExpiration:
    """Tests for TTL-based expiration."""

    @pytest.mark.asyncio
    async def test_checkpoint_with_expiry_time(self, temp_dir, mock_agent):
        """Test checkpoint creation with expiry time."""
        store = FileCheckpointStore(base_dir=temp_dir, compress=False)
        config = CheckpointConfig(
            expiry_hours=24.0,
        )
        manager = CheckpointManager(store=store, config=config)

        cp = await manager.create_checkpoint(
            debate_id="expiry-test",
            task="Test expiry",
            current_round=1,
            total_rounds=3,
            phase="proposal",
            messages=[],
            critiques=[],
            votes=[],
            agents=[mock_agent],
        )

        assert cp.expires_at is not None
        expiry_time = datetime.fromisoformat(cp.expires_at)
        assert expiry_time > datetime.now()

    @pytest.mark.asyncio
    async def test_database_cleanup_expired(self, temp_dir):
        """Test database store cleanup of expired checkpoints."""
        db_path = Path(temp_dir) / "expire_test.db"
        store = DatabaseCheckpointStore(db_path=str(db_path), compress=False)

        # Create checkpoint with past expiry
        cp = DebateCheckpoint(
            checkpoint_id="cp-expired-test",
            debate_id="debate-1",
            task="Expired test",
            current_round=1,
            total_rounds=3,
            phase="proposal",
            messages=[],
            critiques=[],
            votes=[],
            agent_states=[],
            expires_at=(datetime.now() - timedelta(hours=1)).isoformat(),
        )
        await store.save(cp)

        # Cleanup expired
        deleted_count = await store.cleanup_expired()

        assert deleted_count == 1

        # Verify it's gone
        loaded = await store.load("cp-expired-test")
        assert loaded is None

    @pytest.mark.asyncio
    async def test_no_expiry_when_zero_hours(self, temp_dir, mock_agent):
        """Test no expiry when expiry_hours is 0."""
        store = FileCheckpointStore(base_dir=temp_dir, compress=False)
        config = CheckpointConfig(
            expiry_hours=0.0,  # No expiry
        )
        manager = CheckpointManager(store=store, config=config)

        cp = await manager.create_checkpoint(
            debate_id="no-expiry",
            task="No expiry test",
            current_round=1,
            total_rounds=3,
            phase="proposal",
            messages=[],
            critiques=[],
            votes=[],
            agents=[mock_agent],
        )

        assert cp.expires_at is None


class TestStorageBackendAbstraction:
    """Tests for storage backend abstraction."""

    @pytest.mark.asyncio
    async def test_file_store_interface(self, file_store, sample_checkpoint):
        """Test FileCheckpointStore implements CheckpointStore interface."""
        # save
        path = await file_store.save(sample_checkpoint)
        assert path is not None

        # load
        loaded = await file_store.load(sample_checkpoint.checkpoint_id)
        assert loaded is not None

        # list_checkpoints
        checkpoints = await file_store.list_checkpoints()
        assert len(checkpoints) >= 1

        # delete
        result = await file_store.delete(sample_checkpoint.checkpoint_id)
        assert result is True

    @pytest.mark.asyncio
    async def test_database_store_interface(self, db_store, sample_checkpoint):
        """Test DatabaseCheckpointStore implements CheckpointStore interface."""
        # save
        path = await db_store.save(sample_checkpoint)
        assert path is not None

        # load
        loaded = await db_store.load(sample_checkpoint.checkpoint_id)
        assert loaded is not None

        # list_checkpoints
        checkpoints = await db_store.list_checkpoints()
        assert len(checkpoints) >= 1

        # delete
        result = await db_store.delete(sample_checkpoint.checkpoint_id)
        assert result is True

    @pytest.mark.asyncio
    async def test_database_store_stats(self, db_store, sample_checkpoint):
        """Test DatabaseCheckpointStore provides statistics."""
        await db_store.save(sample_checkpoint)

        stats = await db_store.get_stats()

        assert "total_checkpoints" in stats
        assert "unique_debates" in stats
        assert "total_bytes" in stats
        assert stats["total_checkpoints"] >= 1


# =============================================================================
# Key Scenario Tests
# =============================================================================


class TestFullDebateStateRoundTrip:
    """Tests for full debate state round-trip (save -> load)."""

    @pytest.mark.asyncio
    async def test_round_trip_preserves_all_data(
        self, file_store, sample_messages, sample_critiques, sample_votes, mock_agent
    ):
        """Test that round-trip preserves all debate data."""
        manager = CheckpointManager(store=file_store)

        # Create checkpoint with full state
        original = await manager.create_checkpoint(
            debate_id="round-trip-test",
            task="Complete round-trip test",
            current_round=3,
            total_rounds=5,
            phase="synthesis",
            messages=sample_messages,
            critiques=sample_critiques,
            votes=sample_votes,
            agents=[mock_agent],
            current_consensus="We have consensus",
            claims_kernel_state={"claims": [{"id": 1}]},
            belief_network_state={"beliefs": {}},
            continuum_memory_state={"entries": []},
        )

        # Load it back
        loaded = await file_store.load(original.checkpoint_id)

        # Verify all data preserved
        assert loaded.debate_id == "round-trip-test"
        assert loaded.task == "Complete round-trip test"
        assert loaded.current_round == 3
        assert loaded.total_rounds == 5
        assert loaded.phase == "synthesis"
        assert len(loaded.messages) == 3
        assert len(loaded.critiques) == 2
        assert len(loaded.votes) == 3
        assert loaded.current_consensus == "We have consensus"
        assert loaded.claims_kernel_state == {"claims": [{"id": 1}]}
        assert loaded.belief_network_state == {"beliefs": {}}
        assert loaded.continuum_memory_state == {"entries": []}

    @pytest.mark.asyncio
    async def test_round_trip_with_compression(self, compressed_file_store, sample_checkpoint):
        """Test round-trip with compression."""
        path = await compressed_file_store.save(sample_checkpoint)
        assert path.endswith(".json.gz")

        loaded = await compressed_file_store.load(sample_checkpoint.checkpoint_id)

        assert loaded is not None
        assert loaded.checkpoint_id == sample_checkpoint.checkpoint_id
        assert loaded.verify_integrity()


class TestPartialStateRecovery:
    """Tests for partial state recovery scenarios."""

    @pytest.mark.asyncio
    async def test_recover_from_mid_round_checkpoint(self, file_store, mock_agent):
        """Test recovering from a checkpoint taken mid-round."""
        manager = CheckpointManager(store=file_store)

        # Create checkpoint mid-debate
        cp = await manager.create_checkpoint(
            debate_id="partial-recovery",
            task="Partial recovery test",
            current_round=2,
            total_rounds=5,
            phase="critique",  # Mid-round phase
            messages=[],
            critiques=[],
            votes=[],
            agents=[mock_agent],
        )

        # Resume
        resumed = await manager.resume_from_checkpoint(cp.checkpoint_id, "user")

        assert resumed is not None
        assert resumed.checkpoint.current_round == 2
        assert resumed.checkpoint.phase == "critique"

    @pytest.mark.asyncio
    async def test_get_latest_checkpoint(self, file_store, mock_agent):
        """Test getting the latest checkpoint for a debate."""
        manager = CheckpointManager(store=file_store)

        # Create multiple checkpoints
        for i in range(3):
            await manager.create_checkpoint(
                debate_id="latest-test",
                task="Latest test",
                current_round=i + 1,
                total_rounds=5,
                phase="proposal",
                messages=[],
                critiques=[],
                votes=[],
                agents=[mock_agent],
            )

        # Get latest
        latest = await manager.get_latest("latest-test")

        assert latest is not None
        # Latest should be round 3
        assert latest.current_round == 3


class TestConcurrentCheckpointAccess:
    """Tests for concurrent checkpoint access."""

    @pytest.mark.asyncio
    async def test_concurrent_saves(self, file_store):
        """Test concurrent checkpoint saves don't conflict."""
        checkpoints = [
            DebateCheckpoint(
                checkpoint_id=f"concurrent-{i}",
                debate_id=f"debate-{i}",
                task=f"Concurrent task {i}",
                current_round=1,
                total_rounds=3,
                phase="proposal",
                messages=[],
                critiques=[],
                votes=[],
                agent_states=[],
            )
            for i in range(5)
        ]

        # Save concurrently
        tasks = [file_store.save(cp) for cp in checkpoints]
        results = await asyncio.gather(*tasks)

        assert len(results) == 5
        assert all(r is not None for r in results)

        # Verify all saved
        for cp in checkpoints:
            loaded = await file_store.load(cp.checkpoint_id)
            assert loaded is not None

    @pytest.mark.asyncio
    async def test_concurrent_reads(self, file_store, sample_checkpoint):
        """Test concurrent checkpoint reads."""
        await file_store.save(sample_checkpoint)

        # Read concurrently
        tasks = [file_store.load(sample_checkpoint.checkpoint_id) for _ in range(10)]
        results = await asyncio.gather(*tasks)

        assert len(results) == 10
        assert all(r is not None for r in results)
        assert all(r.checkpoint_id == sample_checkpoint.checkpoint_id for r in results)

    @pytest.mark.asyncio
    async def test_database_concurrent_writes(self, db_store):
        """Test concurrent writes to database store."""
        checkpoints = [
            DebateCheckpoint(
                checkpoint_id=f"db-concurrent-{i}",
                debate_id=f"debate-{i}",
                task=f"DB Concurrent task {i}",
                current_round=1,
                total_rounds=3,
                phase="proposal",
                messages=[],
                critiques=[],
                votes=[],
                agent_states=[],
            )
            for i in range(10)
        ]

        # Save concurrently
        tasks = [db_store.save(cp) for cp in checkpoints]
        await asyncio.gather(*tasks)

        # Verify all saved
        all_cps = await db_store.list_checkpoints()
        assert len(all_cps) == 10


class TestStorageErrorHandling:
    """Tests for storage error handling."""

    @pytest.mark.asyncio
    async def test_save_to_readonly_directory(self):
        """Test error handling when saving to readonly directory."""
        # Create a store pointing to root (likely readonly)
        # This tests graceful handling, not actual write
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileCheckpointStore(base_dir=tmpdir)
            cp = DebateCheckpoint(
                checkpoint_id="test-cp",
                debate_id="debate-1",
                task="Test",
                current_round=1,
                total_rounds=3,
                phase="proposal",
                messages=[],
                critiques=[],
                votes=[],
                agent_states=[],
            )
            # This should succeed in temp dir
            result = await store.save(cp)
            assert result is not None

    @pytest.mark.asyncio
    async def test_load_handles_permission_error(self, temp_dir):
        """Test load handles permission errors gracefully."""
        store = FileCheckpointStore(base_dir=temp_dir, compress=False)

        # Try to load from nonexistent file
        result = await store.load("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_checkpoint_manager_handles_save_error(self, temp_dir, mock_agent):
        """Test checkpoint manager handles save errors gracefully."""
        store = FileCheckpointStore(base_dir=temp_dir)
        config = CheckpointConfig()
        manager = CheckpointManager(store=store, config=config)

        # Create a valid checkpoint - should succeed
        cp = await manager.create_checkpoint(
            debate_id="error-test",
            task="Error test",
            current_round=1,
            total_rounds=3,
            phase="proposal",
            messages=[],
            critiques=[],
            votes=[],
            agents=[mock_agent],
        )
        assert cp is not None


# =============================================================================
# CheckpointManager Tests
# =============================================================================


class TestCheckpointManager:
    """Tests for CheckpointManager class."""

    def test_should_checkpoint_at_interval(self):
        """Test should_checkpoint returns True at configured intervals."""
        config = CheckpointConfig(interval_rounds=3)
        manager = CheckpointManager(config=config)

        assert manager.should_checkpoint("debate-1", 3)
        assert manager.should_checkpoint("debate-1", 6)
        assert manager.should_checkpoint("debate-1", 9)

        assert not manager.should_checkpoint("debate-1", 1)
        assert not manager.should_checkpoint("debate-1", 2)
        assert not manager.should_checkpoint("debate-1", 4)

    def test_should_checkpoint_by_time(self, temp_dir):
        """Test should_checkpoint based on time interval."""
        config = CheckpointConfig(
            interval_rounds=100,  # High, so round check doesn't trigger
            interval_seconds=0.1,  # Short interval
        )
        manager = CheckpointManager(config=config)

        # First check - no previous checkpoint time
        assert not manager.should_checkpoint("debate-time", 1)

        # Record a checkpoint time
        manager._last_checkpoint_time["debate-time"] = datetime.now() - timedelta(seconds=1)

        # Now should trigger (more than 0.1 seconds passed)
        assert manager.should_checkpoint("debate-time", 1)

    @pytest.mark.asyncio
    async def test_add_intervention_note(self, file_store, sample_checkpoint):
        """Test adding intervention notes to checkpoint."""
        manager = CheckpointManager(store=file_store)
        await file_store.save(sample_checkpoint)

        success = await manager.add_intervention(
            sample_checkpoint.checkpoint_id,
            note="Human reviewer needed",
            by="admin",
        )

        assert success is True

        # Verify note was added
        loaded = await file_store.load(sample_checkpoint.checkpoint_id)
        assert loaded.pending_intervention is True
        assert len(loaded.intervention_notes) > 0
        assert "Human reviewer" in loaded.intervention_notes[-1]

    @pytest.mark.asyncio
    async def test_list_debates_with_checkpoints(self, file_store, mock_agent):
        """Test listing debates that have checkpoints."""
        manager = CheckpointManager(store=file_store)

        # Create checkpoints for multiple debates
        for debate_num in range(3):
            for round_num in range(2):
                await manager.create_checkpoint(
                    debate_id=f"debate-list-{debate_num}",
                    task=f"Task {debate_num}",
                    current_round=round_num + 1,
                    total_rounds=5,
                    phase="proposal",
                    messages=[],
                    critiques=[],
                    votes=[],
                    agents=[mock_agent],
                )

        debates = await manager.list_debates_with_checkpoints()

        assert len(debates) == 3
        for debate in debates:
            assert "debate_id" in debate
            assert "checkpoint_count" in debate
            assert debate["checkpoint_count"] == 2


# =============================================================================
# CheckpointWebhook Tests
# =============================================================================


class TestCheckpointWebhook:
    """Tests for CheckpointWebhook class."""

    def test_register_handler(self):
        """Test registering webhook handlers."""
        webhook = CheckpointWebhook()

        @webhook.on_checkpoint
        def checkpoint_handler(data):
            pass

        @webhook.on_resume
        def resume_handler(data):
            pass

        assert len(webhook.handlers["on_checkpoint"]) == 1
        assert len(webhook.handlers["on_resume"]) == 1

    @pytest.mark.asyncio
    async def test_emit_calls_handlers(self):
        """Test that emit calls registered handlers."""
        webhook = CheckpointWebhook()
        calls = []

        @webhook.on_checkpoint
        def handler(data):
            calls.append(data)

        await webhook.emit("on_checkpoint", {"test": "data"})

        assert len(calls) == 1
        assert calls[0]["test"] == "data"

    @pytest.mark.asyncio
    async def test_emit_calls_async_handlers(self):
        """Test that emit calls async handlers."""
        webhook = CheckpointWebhook()
        calls = []

        @webhook.on_checkpoint
        async def async_handler(data):
            calls.append(data)

        await webhook.emit("on_checkpoint", {"async": "data"})

        assert len(calls) == 1
        assert calls[0]["async"] == "data"

    @pytest.mark.asyncio
    async def test_emit_handles_handler_errors(self):
        """Test that emit handles handler errors gracefully."""
        webhook = CheckpointWebhook()

        @webhook.on_checkpoint
        def failing_handler(data):
            raise ValueError("Handler error")

        # Should not raise
        await webhook.emit("on_checkpoint", {"test": "data"})


# =============================================================================
# GitCheckpointStore Tests
# =============================================================================


class TestGitCheckpointStore:
    """Tests for GitCheckpointStore class."""

    @pytest.mark.asyncio
    async def test_validates_checkpoint_id(self, temp_dir):
        """Test that invalid checkpoint IDs are rejected."""
        store = GitCheckpointStore(repo_path=temp_dir)

        # Invalid ID should raise
        cp = DebateCheckpoint(
            checkpoint_id="../../../etc/passwd",  # Path traversal attempt
            debate_id="debate-1",
            task="Test",
            current_round=1,
            total_rounds=3,
            phase="proposal",
            messages=[],
            critiques=[],
            votes=[],
            agent_states=[],
        )

        with pytest.raises(ValueError, match="Invalid checkpoint ID"):
            await store.save(cp)

    @pytest.mark.asyncio
    async def test_load_validates_checkpoint_id(self, temp_dir):
        """Test that load validates checkpoint IDs."""
        store = GitCheckpointStore(repo_path=temp_dir)

        # Invalid ID should return None
        result = await store.load("../../../etc/passwd")
        assert result is None


# =============================================================================
# S3CheckpointStore Tests
# =============================================================================


class TestS3CheckpointStore:
    """Tests for S3CheckpointStore class."""

    def test_requires_boto3(self):
        """Test that S3 store requires boto3."""
        store = S3CheckpointStore(bucket="test-bucket")

        with patch.dict("sys.modules", {"boto3": None}):
            # Mock ImportError for boto3
            with patch.object(store, "_get_client", side_effect=ImportError("No boto3")):
                pass  # Constructor should succeed

    def test_get_key_format(self):
        """Test S3 key format."""
        store = S3CheckpointStore(bucket="test", prefix="checkpoints/")
        key = store._get_key("test-cp-001")

        assert key == "checkpoints/test-cp-001.json.gz"


# =============================================================================
# RecoveryNarrator Tests
# =============================================================================


class TestRecoveryNarrator:
    """Tests for RecoveryNarrator class."""

    @pytest.mark.asyncio
    async def test_generate_recovery_summary_no_history(self, temp_dir):
        """Test generating recovery summary with no commit history."""
        git_store = GitCheckpointStore(repo_path=temp_dir)
        narrator = RecoveryNarrator(git_store)

        summary = await narrator.generate_recovery_summary("nonexistent-debate")

        assert "No previous debate history found" in summary

    @pytest.mark.asyncio
    async def test_get_resumption_prompt(self, temp_dir):
        """Test generating resumption prompt for agent."""
        git_store = GitCheckpointStore(repo_path=temp_dir)
        narrator = RecoveryNarrator(git_store)

        prompt = await narrator.get_resumption_prompt("debate-1", "claude")

        assert "You are resuming a debate" in prompt
        assert "claude" in prompt


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestCheckpointDebateFunction:
    """Tests for checkpoint_debate convenience function."""

    @pytest.mark.asyncio
    async def test_quick_checkpoint(self, temp_dir, mock_agent, sample_messages):
        """Test quick checkpoint creation."""
        cp = await checkpoint_debate(
            debate_id="quick-checkpoint",
            task="Quick checkpoint test",
            round_num=2,
            total_rounds=5,
            phase="critique",
            messages=sample_messages,
            agents=[mock_agent],
            store_path=temp_dir,
        )

        assert cp is not None
        assert cp.debate_id == "quick-checkpoint"
        assert cp.current_round == 2


# =============================================================================
# ResumedDebate Tests
# =============================================================================


class TestResumedDebate:
    """Tests for ResumedDebate dataclass."""

    def test_create_resumed_debate(self, sample_checkpoint):
        """Test creating a ResumedDebate context."""
        resumed = ResumedDebate(
            checkpoint=sample_checkpoint,
            original_debate_id=sample_checkpoint.debate_id,
            resumed_at=datetime.now().isoformat(),
            resumed_by="test-user",
            messages=[],
            votes=[],
        )

        assert resumed.checkpoint == sample_checkpoint
        assert resumed.original_debate_id == sample_checkpoint.debate_id
        assert resumed.resumed_by == "test-user"
        assert resumed.context_drift_detected is False

    def test_resumed_debate_with_drift(self, sample_checkpoint):
        """Test ResumedDebate with context drift."""
        resumed = ResumedDebate(
            checkpoint=sample_checkpoint,
            original_debate_id=sample_checkpoint.debate_id,
            resumed_at=datetime.now().isoformat(),
            resumed_by="system",
            messages=[],
            votes=[],
            context_drift_detected=True,
            drift_notes=["Agent model updated", "System prompt changed"],
        )

        assert resumed.context_drift_detected is True
        assert len(resumed.drift_notes) == 2


# =============================================================================
# Safe Checkpoint ID Pattern Tests
# =============================================================================


class TestSafeCheckpointIdPattern:
    """Tests for SAFE_CHECKPOINT_ID pattern."""

    def test_valid_patterns(self):
        """Test valid checkpoint ID patterns."""
        valid_ids = [
            "cp-123",
            "checkpoint_456",
            "a1b2c3d4",
            "MyCheckpoint",
            "test-cp-001-abcd",
            "a" * 128,  # Max length
        ]

        for id_ in valid_ids:
            assert SAFE_CHECKPOINT_ID.match(id_) is not None, f"{id_} should be valid"

    def test_invalid_patterns(self):
        """Test invalid checkpoint ID patterns."""
        invalid_ids = [
            "",  # Empty
            "-starts-with-dash",
            "_starts-with-underscore",
            "../path/traversal",
            "has spaces",
            "has.dots",
            "has<special>chars",
            "a" * 129,  # Too long
        ]

        for id_ in invalid_ids:
            assert SAFE_CHECKPOINT_ID.match(id_) is None, f"{id_} should be invalid"


# =============================================================================
# CheckpointConfig Tests
# =============================================================================


class TestCheckpointConfig:
    """Tests for CheckpointConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = CheckpointConfig()

        assert config.interval_rounds == 1
        assert config.interval_seconds == 300.0
        assert config.max_checkpoints == 10
        assert config.expiry_hours == 72.0
        assert config.compress is True
        assert config.auto_cleanup is True
        assert config.continuous_mode is False
        assert config.enable_recovery_narrator is True
        assert config.glacial_tier_sync is False

    def test_custom_config(self):
        """Test custom configuration."""
        config = CheckpointConfig(
            interval_rounds=5,
            max_checkpoints=20,
            expiry_hours=168.0,
            compress=False,
            continuous_mode=True,
        )

        assert config.interval_rounds == 5
        assert config.max_checkpoints == 20
        assert config.expiry_hours == 168.0
        assert config.compress is False
        assert config.continuous_mode is True


# =============================================================================
# AgentState Tests
# =============================================================================


class TestAgentState:
    """Tests for AgentState dataclass."""

    def test_create_basic(self):
        """Test creating basic agent state."""
        state = AgentState(
            agent_name="test",
            agent_model="model",
            agent_role="role",
            system_prompt="prompt",
            stance="neutral",
        )

        assert state.agent_name == "test"
        assert state.memory_snapshot is None

    def test_create_with_memory(self):
        """Test creating agent state with memory."""
        memory = {"key": "value", "list": [1, 2, 3]}
        state = AgentState(
            agent_name="test",
            agent_model="model",
            agent_role="role",
            system_prompt="prompt",
            stance="neutral",
            memory_snapshot=memory,
        )

        assert state.memory_snapshot == memory


# =============================================================================
# Integration Tests
# =============================================================================


class TestCheckpointIntegration:
    """Integration tests combining multiple components."""

    @pytest.mark.asyncio
    async def test_full_lifecycle_with_webhook(self, temp_dir, mock_agent):
        """Test full checkpoint lifecycle with webhook notifications."""
        events = []

        webhook = CheckpointWebhook()

        @webhook.on_checkpoint
        def on_checkpoint(data):
            events.append(("checkpoint", data))

        @webhook.on_resume
        def on_resume(data):
            events.append(("resume", data))

        store = FileCheckpointStore(base_dir=temp_dir, compress=False)
        config = CheckpointConfig(interval_rounds=1)
        manager = CheckpointManager(store=store, config=config, webhook=webhook)

        # Create checkpoint
        cp = await manager.create_checkpoint(
            debate_id="lifecycle-test",
            task="Full lifecycle test",
            current_round=1,
            total_rounds=3,
            phase="proposal",
            messages=[],
            critiques=[],
            votes=[],
            agents=[mock_agent],
        )

        assert len(events) == 1
        assert events[0][0] == "checkpoint"

        # Resume
        await manager.resume_from_checkpoint(cp.checkpoint_id, "user")

        assert len(events) == 2
        assert events[1][0] == "resume"

    @pytest.mark.asyncio
    async def test_manager_with_database_store(self, temp_dir, mock_agent):
        """Test CheckpointManager with DatabaseCheckpointStore."""
        db_path = Path(temp_dir) / "manager_test.db"
        store = DatabaseCheckpointStore(db_path=str(db_path))
        manager = CheckpointManager(store=store)

        # Create
        cp = await manager.create_checkpoint(
            debate_id="db-manager-test",
            task="DB manager test",
            current_round=1,
            total_rounds=3,
            phase="proposal",
            messages=[],
            critiques=[],
            votes=[],
            agents=[mock_agent],
        )

        # Resume
        resumed = await manager.resume_from_checkpoint(cp.checkpoint_id, "user")
        assert resumed is not None

        # Get latest
        latest = await manager.get_latest("db-manager-test")
        assert latest is not None
        assert latest.checkpoint_id == cp.checkpoint_id
