"""
Integration tests for checkpoint/resume functionality.

Tests full debate checkpoint/resume scenarios including:
- Arena checkpoint creation during debates
- Cross-session debate resumption
- Checkpoint handler API integration
- Error recovery and edge cases
"""

from __future__ import annotations

import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.debate.checkpoint import (
    AgentState,
    CheckpointConfig,
    CheckpointManager,
    CheckpointStatus,
    DebateCheckpoint,
    FileCheckpointStore,
)


class TestCheckpointLifecycle:
    """Test complete checkpoint lifecycle."""

    @pytest.fixture
    def temp_store(self):
        """Create a temporary checkpoint store."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileCheckpointStore(base_dir=tmpdir, compress=False)
            yield store

    @pytest.fixture
    def checkpoint_manager(self, temp_store):
        """Create a checkpoint manager for testing."""
        config = CheckpointConfig(
            interval_rounds=1,
            max_checkpoints=10,
            expiry_hours=24.0,
            compress=False,
        )
        return CheckpointManager(store=temp_store, config=config)

    @pytest.mark.asyncio
    async def test_checkpoint_creation_and_retrieval(self, checkpoint_manager):
        """Test creating and retrieving a checkpoint."""

        class MockAgent:
            name = "test-agent"
            model = "mock-model"
            role = "proposer"
            system_prompt = "Test prompt"
            stance = "neutral"

        checkpoint = await checkpoint_manager.create_checkpoint(
            debate_id="debate-001",
            task="Test debate task",
            current_round=2,
            total_rounds=5,
            phase="critique",
            messages=[],  # Empty - actual Message objects required for real data
            critiques=[],
            votes=[],
            agents=[MockAgent()],
        )

        assert checkpoint is not None
        assert checkpoint.debate_id == "debate-001"
        assert checkpoint.current_round == 2
        assert checkpoint.total_rounds == 5

        # Retrieve checkpoint
        loaded = await checkpoint_manager.store.load(checkpoint.checkpoint_id)
        assert loaded is not None
        assert loaded.debate_id == "debate-001"
        assert loaded.verify_integrity()

    @pytest.mark.asyncio
    async def test_multiple_checkpoints_same_debate(self, checkpoint_manager):
        """Test creating multiple checkpoints for the same debate."""

        class MockAgent:
            name = "test-agent"
            model = "mock-model"
            role = "proposer"
            system_prompt = "Test prompt"
            stance = "neutral"

        checkpoints = []
        for round_num in range(1, 4):
            cp = await checkpoint_manager.create_checkpoint(
                debate_id="debate-002",
                task="Multi-checkpoint debate",
                current_round=round_num,
                total_rounds=5,
                phase="proposal",
                messages=[],
                critiques=[],
                votes=[],
                agents=[MockAgent()],
            )
            checkpoints.append(cp)

        # List checkpoints for debate
        debate_checkpoints = await checkpoint_manager.store.list_checkpoints(debate_id="debate-002")

        assert len(debate_checkpoints) == 3
        rounds = [cp["current_round"] for cp in debate_checkpoints]
        assert sorted(rounds) == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_checkpoint_expiration(self, temp_store):
        """Test checkpoint expiration handling."""
        config = CheckpointConfig(
            interval_rounds=1,
            max_checkpoints=10,
            expiry_hours=0.001,  # Very short expiry
        )
        manager = CheckpointManager(store=temp_store, config=config)

        class MockAgent:
            name = "test-agent"
            model = "mock-model"
            role = "proposer"
            system_prompt = "Test prompt"
            stance = "neutral"

        checkpoint = await manager.create_checkpoint(
            debate_id="debate-expire",
            task="Expiration test",
            current_round=1,
            total_rounds=3,
            phase="proposal",
            messages=[],
            critiques=[],
            votes=[],
            agents=[MockAgent()],
        )

        # Manually expire the checkpoint
        checkpoint.expires_at = (datetime.now() - timedelta(hours=1)).isoformat()
        checkpoint.status = CheckpointStatus.EXPIRED
        await temp_store.save(checkpoint)

        # Verify status is expired
        loaded = await temp_store.load(checkpoint.checkpoint_id)
        assert loaded.status == CheckpointStatus.EXPIRED


class TestCheckpointResume:
    """Test checkpoint resumption scenarios."""

    @pytest.fixture
    def temp_store(self):
        """Create a temporary checkpoint store."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileCheckpointStore(base_dir=tmpdir, compress=False)
            yield store

    @pytest.fixture
    def checkpoint_manager(self, temp_store):
        """Create a checkpoint manager for testing."""
        return CheckpointManager(store=temp_store)

    @pytest.mark.asyncio
    async def test_resume_from_checkpoint(self, checkpoint_manager):
        """Test resuming a debate from checkpoint."""

        class MockAgent:
            name = "proposer-1"
            model = "mock-model"
            role = "proposer"
            system_prompt = "You are a proposer"
            stance = "in_favor"

        # Create checkpoint with empty messages (actual Message objects would be needed for real data)
        checkpoint = await checkpoint_manager.create_checkpoint(
            debate_id="debate-resume",
            task="Resume test debate",
            current_round=3,
            total_rounds=5,
            phase="synthesis",
            messages=[],
            critiques=[],
            votes=[],
            agents=[MockAgent()],
            current_consensus="Partial agreement",
        )

        # Resume from checkpoint
        resumed = await checkpoint_manager.resume_from_checkpoint(
            checkpoint.checkpoint_id,
            resumed_by="test-user",
        )

        assert resumed is not None
        assert resumed.original_debate_id == "debate-resume"
        assert resumed.resumed_by == "test-user"
        assert resumed.checkpoint.current_round == 3

    @pytest.mark.asyncio
    async def test_resume_increments_count(self, checkpoint_manager):
        """Test that resume increments the resume count."""

        class MockAgent:
            name = "test-agent"
            model = "mock-model"
            role = "proposer"
            system_prompt = "Test prompt"
            stance = "neutral"

        checkpoint = await checkpoint_manager.create_checkpoint(
            debate_id="debate-count",
            task="Count test",
            current_round=1,
            total_rounds=3,
            phase="proposal",
            messages=[],
            critiques=[],
            votes=[],
            agents=[MockAgent()],
        )

        assert checkpoint.resume_count == 0

        # Resume multiple times
        for i in range(3):
            resumed = await checkpoint_manager.resume_from_checkpoint(
                checkpoint.checkpoint_id,
                resumed_by=f"user-{i}",
            )
            assert resumed is not None

        # Check final resume count
        loaded = await checkpoint_manager.store.load(checkpoint.checkpoint_id)
        assert loaded.resume_count == 3

    @pytest.mark.asyncio
    async def test_resume_nonexistent_checkpoint(self, checkpoint_manager):
        """Test resuming from nonexistent checkpoint returns None."""
        resumed = await checkpoint_manager.resume_from_checkpoint(
            "nonexistent-checkpoint-id",
            resumed_by="test-user",
        )

        assert resumed is None


class TestCheckpointIntegrity:
    """Test checkpoint integrity verification."""

    def test_checkpoint_integrity_valid(self):
        """Test that valid checkpoints pass integrity check."""
        checkpoint = DebateCheckpoint(
            checkpoint_id="cp-valid",
            debate_id="debate-integrity",
            task="Integrity test",
            current_round=2,
            total_rounds=5,
            phase="critique",
            messages=[{"content": "test1"}, {"content": "test2"}],
            critiques=[],
            votes=[],
            agent_states=[],
        )

        assert checkpoint.verify_integrity()

    def test_checkpoint_integrity_invalid_after_modification(self):
        """Test that modified checkpoints fail integrity check."""
        checkpoint = DebateCheckpoint(
            checkpoint_id="cp-modified",
            debate_id="debate-modified",
            task="Modified test",
            current_round=2,
            total_rounds=5,
            phase="critique",
            messages=[{"content": "test"}],
            critiques=[],
            votes=[],
            agent_states=[],
        )

        original_checksum = checkpoint.checksum

        # Modify checkpoint without updating checksum
        checkpoint.messages.append({"content": "tampered"})

        # Checksum should no longer match
        assert checkpoint.checksum == original_checksum
        assert not checkpoint.verify_integrity()


class TestCheckpointStoreOperations:
    """Test checkpoint store operations."""

    @pytest.fixture
    def temp_store(self):
        """Create a temporary checkpoint store."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileCheckpointStore(base_dir=tmpdir, compress=False)
            yield store

    @pytest.mark.asyncio
    async def test_delete_checkpoint(self, temp_store):
        """Test deleting a checkpoint."""
        checkpoint = DebateCheckpoint(
            checkpoint_id="cp-delete",
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

        # Save checkpoint
        await temp_store.save(checkpoint)

        # Verify it exists
        loaded = await temp_store.load("cp-delete")
        assert loaded is not None

        # Delete checkpoint
        deleted = await temp_store.delete("cp-delete")
        assert deleted is True

        # Verify it's gone
        loaded = await temp_store.load("cp-delete")
        assert loaded is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_checkpoint(self, temp_store):
        """Test deleting nonexistent checkpoint returns False."""
        deleted = await temp_store.delete("nonexistent-id")
        assert deleted is False

    @pytest.mark.asyncio
    async def test_list_checkpoints_empty(self, temp_store):
        """Test listing checkpoints when none exist."""
        checkpoints = await temp_store.list_checkpoints()
        assert checkpoints == []

    @pytest.mark.asyncio
    async def test_list_checkpoints_filtered(self, temp_store):
        """Test listing checkpoints with debate_id filter."""
        # Create checkpoints for different debates
        for debate_id in ["debate-a", "debate-a", "debate-b"]:
            checkpoint = DebateCheckpoint(
                checkpoint_id=f"cp-{debate_id}-{datetime.now().timestamp()}",
                debate_id=debate_id,
                task=f"Task for {debate_id}",
                current_round=1,
                total_rounds=3,
                phase="proposal",
                messages=[],
                critiques=[],
                votes=[],
                agent_states=[],
            )
            await temp_store.save(checkpoint)

        # List all
        all_checkpoints = await temp_store.list_checkpoints()
        assert len(all_checkpoints) == 3

        # List filtered
        debate_a_checkpoints = await temp_store.list_checkpoints(debate_id="debate-a")
        assert len(debate_a_checkpoints) == 2

        debate_b_checkpoints = await temp_store.list_checkpoints(debate_id="debate-b")
        assert len(debate_b_checkpoints) == 1


class TestCheckpointCompression:
    """Test checkpoint compression functionality."""

    @pytest.mark.asyncio
    async def test_compressed_checkpoint_save_load(self):
        """Test saving and loading compressed checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileCheckpointStore(base_dir=tmpdir, compress=True)

            checkpoint = DebateCheckpoint(
                checkpoint_id="cp-compressed",
                debate_id="debate-compress",
                task="Compression test with lots of repeated content " * 100,
                current_round=1,
                total_rounds=3,
                phase="proposal",
                messages=[{"content": "test " * 1000}],
                critiques=[],
                votes=[],
                agent_states=[],
            )

            # Save compressed
            path = await store.save(checkpoint)
            assert path.endswith(".json.gz")

            # Load and verify
            loaded = await store.load("cp-compressed")
            assert loaded is not None
            assert loaded.debate_id == "debate-compress"
            assert loaded.verify_integrity()


class TestCheckpointWithAgentState:
    """Test checkpoints with agent state preservation."""

    @pytest.mark.asyncio
    async def test_agent_state_preservation(self):
        """Test that agent states are preserved in checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileCheckpointStore(base_dir=tmpdir, compress=False)
            manager = CheckpointManager(store=store)

            class DetailedAgent:
                name = "expert-agent"
                model = "gpt-4"
                role = "domain_expert"
                system_prompt = "You are an expert in AI safety"
                stance = "cautious"

            checkpoint = await manager.create_checkpoint(
                debate_id="debate-agents",
                task="Agent state test",
                current_round=2,
                total_rounds=4,
                phase="synthesis",
                messages=[],
                critiques=[],
                votes=[],
                agents=[DetailedAgent()],
            )

            # Verify agent state
            assert len(checkpoint.agent_states) == 1
            agent_state = checkpoint.agent_states[0]
            assert agent_state.agent_name == "expert-agent"
            assert agent_state.agent_model == "gpt-4"
            assert agent_state.agent_role == "domain_expert"
            assert agent_state.stance == "cautious"

            # Load and verify
            loaded = await store.load(checkpoint.checkpoint_id)
            assert len(loaded.agent_states) == 1
            assert loaded.agent_states[0].agent_name == "expert-agent"


class TestCheckpointInterventions:
    """Test human intervention notes on checkpoints."""

    @pytest.fixture
    def temp_store(self):
        """Create a temporary checkpoint store."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileCheckpointStore(base_dir=tmpdir, compress=False)
            yield store

    @pytest.fixture
    def checkpoint_manager(self, temp_store):
        """Create a checkpoint manager for testing."""
        return CheckpointManager(store=temp_store)

    @pytest.mark.asyncio
    async def test_add_intervention_note(self, checkpoint_manager):
        """Test adding intervention notes to checkpoints."""

        class MockAgent:
            name = "test-agent"
            model = "mock-model"
            role = "proposer"
            system_prompt = "Test prompt"
            stance = "neutral"

        checkpoint = await checkpoint_manager.create_checkpoint(
            debate_id="debate-intervention",
            task="Intervention test",
            current_round=2,
            total_rounds=5,
            phase="critique",
            messages=[],
            critiques=[],
            votes=[],
            agents=[MockAgent()],
        )

        # Add intervention
        success = await checkpoint_manager.add_intervention(
            checkpoint.checkpoint_id,
            note="Human review: The debate needs more evidence",
            by="reviewer-1",
        )

        assert success is True

        # Load and verify
        loaded = await checkpoint_manager.store.load(checkpoint.checkpoint_id)
        assert loaded.pending_intervention is True
        assert len(loaded.intervention_notes) == 1
        assert "more evidence" in loaded.intervention_notes[0]

    @pytest.mark.asyncio
    async def test_multiple_intervention_notes(self, checkpoint_manager):
        """Test adding multiple intervention notes."""

        class MockAgent:
            name = "test-agent"
            model = "mock-model"
            role = "proposer"
            system_prompt = "Test prompt"
            stance = "neutral"

        checkpoint = await checkpoint_manager.create_checkpoint(
            debate_id="debate-multi-intervention",
            task="Multi intervention test",
            current_round=1,
            total_rounds=3,
            phase="proposal",
            messages=[],
            critiques=[],
            votes=[],
            agents=[MockAgent()],
        )

        # Add multiple interventions
        await checkpoint_manager.add_intervention(
            checkpoint.checkpoint_id, note="First review", by="reviewer-1"
        )
        await checkpoint_manager.add_intervention(
            checkpoint.checkpoint_id, note="Second review", by="reviewer-2"
        )

        loaded = await checkpoint_manager.store.load(checkpoint.checkpoint_id)
        assert len(loaded.intervention_notes) == 2


class TestCheckpointEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_save_load_empty_checkpoint(self):
        """Test saving and loading checkpoint with minimal data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileCheckpointStore(base_dir=tmpdir, compress=False)

            checkpoint = DebateCheckpoint(
                checkpoint_id="cp-minimal",
                debate_id="debate-minimal",
                task="",
                current_round=0,
                total_rounds=0,
                phase="",
                messages=[],
                critiques=[],
                votes=[],
                agent_states=[],
            )

            await store.save(checkpoint)
            loaded = await store.load("cp-minimal")

            assert loaded is not None
            assert loaded.messages == []
            assert loaded.agent_states == []

    @pytest.mark.asyncio
    async def test_checkpoint_with_special_characters(self):
        """Test checkpoint with special characters in content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileCheckpointStore(base_dir=tmpdir, compress=False)

            special_content = "Task with \"quotes\" and 'apostrophes' and\nnewlines"
            checkpoint = DebateCheckpoint(
                checkpoint_id="cp-special",
                debate_id="debate-special",
                task=special_content,
                current_round=1,
                total_rounds=3,
                phase="proposal",
                messages=[{"content": "Message with unicode: \u2603 \u2764"}],
                critiques=[],
                votes=[],
                agent_states=[],
            )

            await store.save(checkpoint)
            loaded = await store.load("cp-special")

            assert loaded is not None
            assert loaded.task == special_content
            assert "\u2603" in loaded.messages[0]["content"]

    def test_checkpoint_id_sanitization(self):
        """Test that checkpoint IDs are sanitized for file safety."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileCheckpointStore(base_dir=tmpdir, compress=False)

            # Path traversal attempts get sanitized (characters replaced)
            path = store._get_path("../../../etc/passwd")
            # Should be sanitized to remove dangerous characters
            assert ".." not in str(path)
            assert "/" not in Path(path).name  # No path separators in filename

            # Test truly empty ID raises ValueError
            with pytest.raises(ValueError):
                store._get_path("")

            # Valid ID should work
            path = store._get_path("valid-checkpoint-123")
            assert "valid-checkpoint-123" in str(path)


class TestCheckpointShouldCheckpoint:
    """Test checkpoint interval logic."""

    def test_should_checkpoint_at_interval(self):
        """Test checkpoint interval checking."""
        config = CheckpointConfig(interval_rounds=3)
        manager = CheckpointManager(config=config)

        # Should checkpoint at round 3, 6, 9, etc.
        assert manager.should_checkpoint("debate-1", 3)
        assert manager.should_checkpoint("debate-1", 6)
        assert manager.should_checkpoint("debate-1", 9)

        # Should not checkpoint at other rounds
        assert not manager.should_checkpoint("debate-1", 1)
        assert not manager.should_checkpoint("debate-1", 2)
        assert not manager.should_checkpoint("debate-1", 4)
        assert not manager.should_checkpoint("debate-1", 5)

    def test_should_checkpoint_every_round(self):
        """Test checkpointing every round."""
        config = CheckpointConfig(interval_rounds=1)
        manager = CheckpointManager(config=config)

        for round_num in range(1, 10):
            assert manager.should_checkpoint("debate-1", round_num)

    def test_should_checkpoint_large_interval(self):
        """Test checkpointing with large interval (effectively disabled)."""
        config = CheckpointConfig(interval_rounds=1000)  # Very large interval
        manager = CheckpointManager(config=config)

        # Should not checkpoint at normal round numbers
        for round_num in range(1, 100):
            assert not manager.should_checkpoint("debate-1", round_num)

        # Would checkpoint at 1000 if debate lasted that long
        assert manager.should_checkpoint("debate-1", 1000)
