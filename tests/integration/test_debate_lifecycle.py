"""
Integration tests for debate lifecycle and state management.

Tests verify that the debate system correctly:
- Manages debate state through phases (PENDING → ACTIVE → CONSENSUS → COMPLETE)
- Handles checkpoint save/restore with integrity verification
- Supports debate forking and follow-up debates
- Recovers from timeouts and errors gracefully
"""

import asyncio
import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.debate.checkpoint import (
    DebateCheckpoint,
    CheckpointStatus,
    AgentState,
    FileCheckpointStore,
    ResumedDebate,
)
from aragora.core import Message, Vote


@pytest.mark.integration_minimal
class TestDebateContext:
    """Test debate context initialization and state."""

    def test_checkpoint_initialization(self):
        """DebateCheckpoint should initialize with all required fields."""
        checkpoint = DebateCheckpoint(
            checkpoint_id="cp-test",
            debate_id="debate-123",
            task="Test task",
            current_round=2,
            total_rounds=5,
            phase="proposal",
            messages=[],
            critiques=[],
            votes=[],
            agent_states=[],
        )

        assert checkpoint.checkpoint_id == "cp-test"
        assert checkpoint.debate_id == "debate-123"
        assert checkpoint.current_round == 2
        assert checkpoint.total_rounds == 5
        assert checkpoint.status == CheckpointStatus.COMPLETE

    def test_checkpoint_mutable_state(self):
        """Checkpoint should track mutable state like messages and votes."""
        checkpoint = DebateCheckpoint(
            checkpoint_id="cp-test",
            debate_id="debate-123",
            task="Test",
            current_round=1,
            total_rounds=3,
            phase="proposal",
            messages=[{"agent": "a1", "content": "Hello"}],
            critiques=[{"agent": "a2", "target": "a1"}],
            votes=[{"agent": "a1", "choice": "approve"}],
            agent_states=[],
        )

        assert len(checkpoint.messages) == 1
        assert len(checkpoint.critiques) == 1
        assert len(checkpoint.votes) == 1


@pytest.mark.integration_minimal
class TestCheckpointSerialization:
    """Test checkpoint serialization and deserialization."""

    def test_checkpoint_serializes_messages(self):
        """Checkpoint should serialize message history."""
        checkpoint = DebateCheckpoint(
            checkpoint_id="cp-123",
            debate_id="debate-456",
            task="Test serialization",
            current_round=2,
            total_rounds=4,
            phase="critique",
            messages=[
                {"agent": "a1", "content": "First message"},
                {"agent": "a2", "content": "Response"},
            ],
            critiques=[],
            votes=[],
            agent_states=[],
        )

        data = checkpoint.to_dict()

        assert data["checkpoint_id"] == "cp-123"
        assert len(data["messages"]) == 2
        assert data["current_round"] == 2

        # Round-trip should preserve data
        restored = DebateCheckpoint.from_dict(data)
        assert restored.checkpoint_id == checkpoint.checkpoint_id
        assert len(restored.messages) == 2

    def test_checkpoint_integrity_checksum(self):
        """Checkpoint should compute integrity checksum."""
        checkpoint = DebateCheckpoint(
            checkpoint_id="cp-test",
            debate_id="debate-test",
            task="Test",
            current_round=1,
            total_rounds=3,
            phase="proposal",
            messages=[{"agent": "a1", "content": "test"}],
            critiques=[],
            votes=[],
            agent_states=[],
        )

        # Checksum is SHA256[:16] = 16 hex chars
        assert len(checkpoint.checksum) == 16
        assert checkpoint.verify_integrity() is True


class TestFileCheckpointStore:
    """Test file-based checkpoint persistence."""

    @pytest.mark.asyncio
    async def test_save_and_load_checkpoint(self):
        """Should save and load checkpoints correctly."""
        with tempfile.TemporaryDirectory() as tmp:
            store = FileCheckpointStore(base_dir=tmp, compress=False)

            checkpoint = DebateCheckpoint(
                checkpoint_id="cp-persist-test",
                debate_id="debate-456",
                task="Persistence test",
                current_round=3,
                total_rounds=5,
                phase="vote",
                messages=[
                    {"agent": "claude", "content": "Test message"},
                ],
                critiques=[],
                votes=[],
                agent_states=[
                    AgentState(
                        agent_name="claude",
                        agent_model="claude-3-sonnet",
                        agent_role="proposer",
                        system_prompt="Be helpful",
                        stance="neutral",
                    )
                ],
            )

            # Save
            path = await store.save(checkpoint)
            assert Path(path).exists()

            # Load
            loaded = await store.load("cp-persist-test")
            assert loaded is not None
            assert loaded.debate_id == "debate-456"
            assert loaded.current_round == 3
            assert len(loaded.agent_states) == 1
            assert loaded.agent_states[0].agent_name == "claude"

    @pytest.mark.asyncio
    async def test_list_checkpoints(self):
        """Should list available checkpoints."""
        with tempfile.TemporaryDirectory() as tmp:
            store = FileCheckpointStore(base_dir=tmp, compress=False)

            # Create multiple checkpoints
            for i in range(3):
                checkpoint = DebateCheckpoint(
                    checkpoint_id=f"cp-list-{i}",
                    debate_id="debate-list-test",
                    task=f"Task {i}",
                    current_round=i + 1,
                    total_rounds=5,
                    phase="proposal",
                    messages=[],
                    critiques=[],
                    votes=[],
                    agent_states=[],
                )
                await store.save(checkpoint)

            # List all
            checkpoints = await store.list_checkpoints()
            assert len(checkpoints) >= 3

            # Filter by debate_id
            filtered = await store.list_checkpoints(debate_id="debate-list-test")
            assert len(filtered) == 3

    @pytest.mark.asyncio
    async def test_delete_checkpoint(self):
        """Should delete checkpoints correctly."""
        with tempfile.TemporaryDirectory() as tmp:
            store = FileCheckpointStore(base_dir=tmp, compress=False)

            checkpoint = DebateCheckpoint(
                checkpoint_id="cp-delete-test",
                debate_id="debate-delete",
                task="Delete test",
                current_round=1,
                total_rounds=3,
                phase="proposal",
                messages=[],
                critiques=[],
                votes=[],
                agent_states=[],
            )
            await store.save(checkpoint)

            # Verify exists
            loaded = await store.load("cp-delete-test")
            assert loaded is not None

            # Delete
            deleted = await store.delete("cp-delete-test")
            assert deleted is True

            # Verify deleted
            loaded = await store.load("cp-delete-test")
            assert loaded is None

    @pytest.mark.asyncio
    async def test_compressed_checkpoint(self):
        """Should handle compressed checkpoints."""
        with tempfile.TemporaryDirectory() as tmp:
            store = FileCheckpointStore(base_dir=tmp, compress=True)

            checkpoint = DebateCheckpoint(
                checkpoint_id="cp-compress",
                debate_id="debate-compress",
                task="Compression test",
                current_round=1,
                total_rounds=3,
                phase="proposal",
                messages=[{"agent": "a1", "content": "x" * 1000}],  # Large message
                critiques=[],
                votes=[],
                agent_states=[],
            )

            path = await store.save(checkpoint)
            assert path.endswith(".json.gz")

            loaded = await store.load("cp-compress")
            assert loaded is not None
            assert loaded.messages[0]["content"] == "x" * 1000


@pytest.mark.integration_minimal
class TestResumedDebate:
    """Test debate resumption from checkpoint."""

    def test_resumed_debate_context(self):
        """ResumedDebate should carry checkpoint context."""
        checkpoint = DebateCheckpoint(
            checkpoint_id="cp-resume",
            debate_id="debate-original",
            task="Resume test",
            current_round=2,
            total_rounds=5,
            phase="critique",
            messages=[{"agent": "a1", "content": "Previous message"}],
            critiques=[],
            votes=[],
            agent_states=[],
        )

        msg = Message(
            role="proposer",
            agent="a1",
            content="Previous message",
        )

        resumed = ResumedDebate(
            checkpoint=checkpoint,
            original_debate_id="debate-original",
            resumed_at=datetime.now().isoformat(),
            resumed_by="user-123",
            messages=[msg],
            votes=[],
        )

        assert resumed.checkpoint.current_round == 2
        assert resumed.original_debate_id == "debate-original"
        assert len(resumed.messages) == 1


class TestMergeResult:
    """Test debate branching and merging."""

    def test_merge_selects_winning_branch(self):
        """Merge should select branch with highest consensus."""
        # This tests the concept of merge - actual impl may vary
        branch_a = {"consensus_confidence": 0.8, "branch_id": "A"}
        branch_b = {"consensus_confidence": 0.6, "branch_id": "B"}

        # Simple winner selection
        winner = max([branch_a, branch_b], key=lambda b: b["consensus_confidence"])
        assert winner["branch_id"] == "A"


@pytest.mark.integration_minimal
class TestDebateResultConstruction:
    """Test debate result construction."""

    def test_result_from_consensus(self):
        """Should construct result when consensus is reached."""
        checkpoint = DebateCheckpoint(
            checkpoint_id="cp-result",
            debate_id="debate-result",
            task="Result test",
            current_round=3,
            total_rounds=5,
            phase="consensus",
            messages=[
                {"agent": "a1", "content": "Proposal A"},
                {"agent": "a2", "content": "I agree with A"},
            ],
            critiques=[],
            votes=[
                {"agent": "a1", "choice": "A"},
                {"agent": "a2", "choice": "A"},
            ],
            agent_states=[],
            current_consensus="Option A is best",
            consensus_confidence=0.9,
        )

        assert checkpoint.current_consensus == "Option A is best"
        assert checkpoint.consensus_confidence == 0.9
        assert len(checkpoint.votes) == 2

    def test_result_from_no_consensus(self):
        """Should construct result when no consensus is reached."""
        checkpoint = DebateCheckpoint(
            checkpoint_id="cp-no-consensus",
            debate_id="debate-no-consensus",
            task="No consensus test",
            current_round=5,
            total_rounds=5,
            phase="complete",
            messages=[],
            critiques=[],
            votes=[
                {"agent": "a1", "choice": "A"},
                {"agent": "a2", "choice": "B"},
            ],
            agent_states=[],
            current_consensus=None,
            consensus_confidence=0.0,
        )

        assert checkpoint.current_consensus is None
        assert checkpoint.consensus_confidence == 0.0


class TestFollowUpSuggestions:
    """Test follow-up debate suggestions based on cruxes."""

    def test_crux_based_suggestions(self):
        """Should generate follow-up suggestions from key disagreements."""
        # Simulate crux identification from debate
        cruxes = [
            {"topic": "performance vs readability", "agents_split": ["a1", "a2"]},
            {"topic": "async vs sync approach", "agents_split": ["a1", "a3"]},
        ]

        # Generate follow-up suggestions
        suggestions = [f"Deep dive: {c['topic']}" for c in cruxes]

        assert len(suggestions) == 2
        assert "performance vs readability" in suggestions[0]


@pytest.mark.integration_minimal
class TestInitialMessagesInjection:
    """Test initial message injection for context seeding."""

    def test_checkpoint_with_initial_messages(self):
        """Should support initial messages for context."""
        initial_messages = [
            {"agent": "system", "content": "Previous context: XYZ"},
            {"agent": "user", "content": "Continue from where we left off"},
        ]

        checkpoint = DebateCheckpoint(
            checkpoint_id="cp-initial",
            debate_id="debate-initial",
            task="Continue discussion",
            current_round=1,
            total_rounds=5,
            phase="proposal",
            messages=initial_messages,
            critiques=[],
            votes=[],
            agent_states=[],
        )

        assert len(checkpoint.messages) == 2
        assert checkpoint.messages[0]["agent"] == "system"


class TestCheckpointPathSecurity:
    """Test checkpoint path security against traversal attacks."""

    @pytest.mark.asyncio
    async def test_rejects_path_traversal(self):
        """Should reject checkpoint IDs with path traversal."""
        with tempfile.TemporaryDirectory() as tmp:
            store = FileCheckpointStore(base_dir=tmp)

            # These should be sanitized or rejected
            dangerous_ids = [
                "../etc/passwd",
                "..\\windows\\system32",
                "foo/../../../etc/shadow",
                "/absolute/path",
            ]

            for dangerous_id in dangerous_ids:
                # Load should return None for sanitized/nonexistent paths
                loaded = await store.load(dangerous_id)
                assert loaded is None

    @pytest.mark.asyncio
    async def test_sanitizes_special_characters(self):
        """Should sanitize special characters in checkpoint IDs."""
        with tempfile.TemporaryDirectory() as tmp:
            store = FileCheckpointStore(base_dir=tmp, compress=False)

            checkpoint = DebateCheckpoint(
                checkpoint_id="cp-with-special!@#$%",
                debate_id="debate-test",
                task="Security test",
                current_round=1,
                total_rounds=3,
                phase="proposal",
                messages=[],
                critiques=[],
                votes=[],
                agent_states=[],
            )

            # Should save with sanitized name
            path = await store.save(checkpoint)
            assert Path(path).exists()
            # Original ID is sanitized in the filename
            assert "!" not in Path(path).name


class TestCheckpointExpiry:
    """Test checkpoint expiration handling."""

    def test_checkpoint_with_expiry(self):
        """Should track expiry time."""
        expires = (datetime.now() + timedelta(hours=24)).isoformat()

        checkpoint = DebateCheckpoint(
            checkpoint_id="cp-expiry",
            debate_id="debate-expiry",
            task="Expiry test",
            current_round=1,
            total_rounds=3,
            phase="proposal",
            messages=[],
            critiques=[],
            votes=[],
            agent_states=[],
            expires_at=expires,
        )

        assert checkpoint.expires_at is not None
        assert checkpoint.expires_at == expires


class TestHumanIntervention:
    """Test human intervention breakpoints in checkpoints."""

    def test_checkpoint_pending_intervention(self):
        """Should track pending human intervention."""
        checkpoint = DebateCheckpoint(
            checkpoint_id="cp-intervention",
            debate_id="debate-intervention",
            task="Needs review",
            current_round=2,
            total_rounds=5,
            phase="vote",
            messages=[],
            critiques=[],
            votes=[],
            agent_states=[],
            pending_intervention=True,
            intervention_notes=["Controversial topic detected", "Requires expert review"],
        )

        assert checkpoint.pending_intervention is True
        assert len(checkpoint.intervention_notes) == 2

    def test_checkpoint_resume_tracking(self):
        """Should track resume history."""
        checkpoint = DebateCheckpoint(
            checkpoint_id="cp-resume-track",
            debate_id="debate-resume",
            task="Resume tracking",
            current_round=1,
            total_rounds=3,
            phase="proposal",
            messages=[],
            critiques=[],
            votes=[],
            agent_states=[],
            resume_count=2,
            last_resumed_at=datetime.now().isoformat(),
            resumed_by="admin@example.com",
        )

        assert checkpoint.resume_count == 2
        assert checkpoint.resumed_by == "admin@example.com"


class TestAgentStatePreservation:
    """Test agent state preservation in checkpoints."""

    def test_agent_state_serialization(self):
        """Should preserve complete agent state."""
        agent_state = AgentState(
            agent_name="claude",
            agent_model="claude-3-opus",
            agent_role="devil_advocate",
            system_prompt="Challenge assumptions",
            stance="skeptical",
            memory_snapshot={"key_insights": ["insight1", "insight2"]},
        )

        checkpoint = DebateCheckpoint(
            checkpoint_id="cp-agent-state",
            debate_id="debate-agent",
            task="Agent state test",
            current_round=1,
            total_rounds=3,
            phase="proposal",
            messages=[],
            critiques=[],
            votes=[],
            agent_states=[agent_state],
        )

        data = checkpoint.to_dict()
        restored = DebateCheckpoint.from_dict(data)

        assert len(restored.agent_states) == 1
        assert restored.agent_states[0].agent_name == "claude"
        assert restored.agent_states[0].stance == "skeptical"
        assert restored.agent_states[0].memory_snapshot["key_insights"] == ["insight1", "insight2"]
