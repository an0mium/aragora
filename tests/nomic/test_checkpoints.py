"""
Tests for Nomic Loop Checkpoints.

Checkpoint functionality:
- State persistence between phases
- Recovery from failures
- Resumable execution
"""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestCheckpointCreation:
    """Tests for checkpoint creation."""

    def test_creates_checkpoint_with_state(self, mock_nomic_state, tmp_path):
        """Should create checkpoint with current state."""
        from aragora.nomic.checkpoints import CheckpointManager

        manager = CheckpointManager(checkpoint_dir=tmp_path)

        checkpoint = manager.create_checkpoint(
            cycle=1,
            phase="debate",
            state=mock_nomic_state.__dict__,
        )

        assert checkpoint is not None
        assert "id" in checkpoint
        assert checkpoint["cycle"] == 1
        assert checkpoint["phase"] == "debate"

    def test_checkpoint_includes_timestamp(self, mock_nomic_state, tmp_path):
        """Should include timestamp in checkpoint."""
        from aragora.nomic.checkpoints import CheckpointManager

        manager = CheckpointManager(checkpoint_dir=tmp_path)

        checkpoint = manager.create_checkpoint(
            cycle=1,
            phase="context",
            state={},
        )

        assert "timestamp" in checkpoint
        assert checkpoint["timestamp"] is not None

    def test_checkpoint_persisted_to_disk(self, mock_nomic_state, tmp_path):
        """Should persist checkpoint to disk."""
        from aragora.nomic.checkpoints import CheckpointManager

        manager = CheckpointManager(checkpoint_dir=tmp_path)

        checkpoint = manager.create_checkpoint(
            cycle=1,
            phase="debate",
            state={"key": "value"},
        )

        # Should have created a file
        checkpoint_files = list(tmp_path.glob("*.json"))
        assert len(checkpoint_files) >= 1


class TestCheckpointRecovery:
    """Tests for checkpoint recovery."""

    def test_loads_latest_checkpoint(self, tmp_path):
        """Should load the most recent checkpoint."""
        from aragora.nomic.checkpoints import CheckpointManager

        manager = CheckpointManager(checkpoint_dir=tmp_path)

        # Create multiple checkpoints
        manager.create_checkpoint(cycle=1, phase="context", state={"step": 1})
        manager.create_checkpoint(cycle=1, phase="debate", state={"step": 2})
        latest = manager.create_checkpoint(cycle=1, phase="design", state={"step": 3})

        recovered = manager.load_latest_checkpoint()

        assert recovered is not None
        assert recovered["phase"] == "design"
        assert recovered["state"]["step"] == 3

    def test_loads_checkpoint_by_id(self, tmp_path):
        """Should load specific checkpoint by ID."""
        from aragora.nomic.checkpoints import CheckpointManager

        manager = CheckpointManager(checkpoint_dir=tmp_path)

        checkpoint = manager.create_checkpoint(
            cycle=2,
            phase="implement",
            state={"data": "test"},
        )

        recovered = manager.load_checkpoint(checkpoint["id"])

        assert recovered is not None
        assert recovered["id"] == checkpoint["id"]
        assert recovered["state"]["data"] == "test"

    def test_returns_none_for_missing_checkpoint(self, tmp_path):
        """Should return None for non-existent checkpoint."""
        from aragora.nomic.checkpoints import CheckpointManager

        manager = CheckpointManager(checkpoint_dir=tmp_path)

        recovered = manager.load_checkpoint("nonexistent-id")

        assert recovered is None


class TestCheckpointResumption:
    """Tests for resuming from checkpoints."""

    @pytest.mark.asyncio
    async def test_resumes_from_correct_phase(self, tmp_path):
        """Should resume execution from checkpointed phase."""
        from aragora.nomic.checkpoints import CheckpointManager

        manager = CheckpointManager(checkpoint_dir=tmp_path)

        # Create checkpoint at debate phase
        manager.create_checkpoint(
            cycle=1,
            phase="debate",
            state={"proposals": [{"id": "p1"}]},
        )

        resume_info = manager.get_resume_point()

        assert resume_info is not None
        assert resume_info["phase"] == "debate"
        assert resume_info["cycle"] == 1

    @pytest.mark.asyncio
    async def test_resumes_with_preserved_state(self, tmp_path):
        """Should preserve state when resuming."""
        from aragora.nomic.checkpoints import CheckpointManager

        manager = CheckpointManager(checkpoint_dir=tmp_path)

        original_state = {
            "context": "gathered context",
            "proposals": [{"id": "p1", "text": "Add feature"}],
        }

        manager.create_checkpoint(
            cycle=1,
            phase="design",
            state=original_state,
        )

        resume_info = manager.get_resume_point()

        assert resume_info["state"]["context"] == "gathered context"
        assert len(resume_info["state"]["proposals"]) == 1


class TestCheckpointCleanup:
    """Tests for checkpoint cleanup."""

    def test_removes_old_checkpoints(self, tmp_path):
        """Should remove checkpoints older than retention period."""
        from aragora.nomic.checkpoints import CheckpointManager

        manager = CheckpointManager(
            checkpoint_dir=tmp_path,
            max_checkpoints=3,
        )

        # Create more than max checkpoints
        for i in range(5):
            manager.create_checkpoint(
                cycle=i,
                phase="context",
                state={"iteration": i},
            )

        manager.cleanup_old_checkpoints()

        checkpoint_files = list(tmp_path.glob("*.json"))
        assert len(checkpoint_files) <= 3

    def test_keeps_most_recent_checkpoints(self, tmp_path):
        """Should keep most recent checkpoints during cleanup."""
        from aragora.nomic.checkpoints import CheckpointManager

        manager = CheckpointManager(
            checkpoint_dir=tmp_path,
            max_checkpoints=2,
        )

        # Create checkpoints
        manager.create_checkpoint(cycle=1, phase="context", state={"v": 1})
        manager.create_checkpoint(cycle=2, phase="context", state={"v": 2})
        manager.create_checkpoint(cycle=3, phase="context", state={"v": 3})

        manager.cleanup_old_checkpoints()

        latest = manager.load_latest_checkpoint()
        assert latest["state"]["v"] == 3


class TestCheckpointIntegration:
    """Integration tests for checkpoint system."""

    @pytest.mark.asyncio
    async def test_full_checkpoint_cycle(self, tmp_path):
        """Should handle full checkpoint create/recover cycle."""
        from aragora.nomic.checkpoints import CheckpointManager

        manager = CheckpointManager(checkpoint_dir=tmp_path)

        # Simulate nomic loop with checkpoints
        phases = ["context", "debate", "design", "implement", "verify"]
        state = {"data": []}

        for i, phase in enumerate(phases):
            state["data"].append(phase)
            manager.create_checkpoint(
                cycle=1,
                phase=phase,
                state=state.copy(),
            )

        # Recover and verify
        recovered = manager.load_latest_checkpoint()

        assert recovered["phase"] == "verify"
        assert len(recovered["state"]["data"]) == 5
        assert "context" in recovered["state"]["data"]
        assert "verify" in recovered["state"]["data"]

    @pytest.mark.asyncio
    async def test_checkpoint_survives_crash_simulation(self, tmp_path):
        """Should recover state after simulated crash."""
        from aragora.nomic.checkpoints import CheckpointManager

        # First manager creates checkpoint
        manager1 = CheckpointManager(checkpoint_dir=tmp_path)
        manager1.create_checkpoint(
            cycle=5,
            phase="implement",
            state={
                "design": {"approved": True},
                "code_changes": {"file.py": "content"},
            },
        )

        # Simulate crash by creating new manager
        manager2 = CheckpointManager(checkpoint_dir=tmp_path)
        recovered = manager2.load_latest_checkpoint()

        assert recovered is not None
        assert recovered["cycle"] == 5
        assert recovered["phase"] == "implement"
        assert recovered["state"]["design"]["approved"] is True
