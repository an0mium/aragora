"""Tests for Checkpoint Memory State integration (Phase 8.5).

Tests the integration between ContinuumMemory snapshots and DebateCheckpoint
for complete debate state restoration.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import pytest

from aragora.memory.continuum import ContinuumMemory, ContinuumMemoryEntry
from aragora.memory.tier_manager import MemoryTier
from aragora.debate.checkpoint import (
    DebateCheckpoint,
    CheckpointManager,
    FileCheckpointStore,
    CheckpointConfig,
    AgentState,
    CheckpointStatus,
)


class TestContinuumMemorySnapshot:
    """Test ContinuumMemory snapshot export and restore."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_continuum.db"
            yield str(db_path)

    @pytest.fixture
    def memory(self, temp_db):
        """Create a ContinuumMemory instance for testing."""
        return ContinuumMemory(db_path=temp_db)

    def test_export_empty_snapshot(self, memory: ContinuumMemory) -> None:
        """Export snapshot from empty memory returns valid structure."""
        snapshot = memory.export_snapshot()

        assert "entries" in snapshot
        assert "tier_counts" in snapshot
        assert "hyperparams" in snapshot
        assert "snapshot_time" in snapshot
        assert "total_entries" in snapshot
        assert "version" in snapshot

        assert len(snapshot["entries"]) == 0
        assert snapshot["total_entries"] == 0

    def test_export_snapshot_with_entries(self, memory: ContinuumMemory) -> None:
        """Export snapshot includes all memory entries."""
        # Add entries to different tiers
        memory.add("fast-1", "Fast tier content", tier=MemoryTier.FAST, importance=0.9)
        memory.add("medium-1", "Medium tier content", tier=MemoryTier.MEDIUM, importance=0.7)
        memory.add("slow-1", "Slow tier content", tier=MemoryTier.SLOW, importance=0.5)

        snapshot = memory.export_snapshot()

        assert snapshot["total_entries"] == 3
        assert snapshot["tier_counts"]["fast"] == 1
        assert snapshot["tier_counts"]["medium"] == 1
        assert snapshot["tier_counts"]["slow"] == 1

        # Check entries contain expected data
        entry_ids = [e["id"] for e in snapshot["entries"]]
        assert "fast-1" in entry_ids
        assert "medium-1" in entry_ids
        assert "slow-1" in entry_ids

    def test_export_snapshot_specific_tiers(self, memory: ContinuumMemory) -> None:
        """Export snapshot can filter to specific tiers."""
        memory.add("fast-1", "Fast content", tier=MemoryTier.FAST)
        memory.add("slow-1", "Slow content", tier=MemoryTier.SLOW)

        snapshot = memory.export_snapshot(tiers=[MemoryTier.FAST])

        assert snapshot["total_entries"] == 1
        assert snapshot["tier_counts"]["fast"] == 1
        assert "slow" not in snapshot["tier_counts"] or snapshot["tier_counts"]["slow"] == 0

    def test_export_snapshot_max_entries(self, memory: ContinuumMemory) -> None:
        """Export snapshot respects max_entries_per_tier limit."""
        # Add many entries to fast tier
        for i in range(10):
            memory.add(f"fast-{i}", f"Content {i}", tier=MemoryTier.FAST, importance=0.5 + i * 0.01)

        snapshot = memory.export_snapshot(max_entries_per_tier=5)

        assert snapshot["tier_counts"]["fast"] == 5
        # Should include highest importance entries
        entry_ids = [e["id"] for e in snapshot["entries"]]
        assert "fast-9" in entry_ids  # Highest importance

    def test_export_snapshot_excludes_metadata(self, memory: ContinuumMemory) -> None:
        """Export snapshot can exclude metadata."""
        memory.add("mem-1", "Content", metadata={"key": "value"})

        snapshot_with_meta = memory.export_snapshot(include_metadata=True)
        snapshot_without_meta = memory.export_snapshot(include_metadata=False)

        assert "metadata" in snapshot_with_meta["entries"][0]
        assert "metadata" not in snapshot_without_meta["entries"][0]

    def test_export_snapshot_preserves_red_line(self, memory: ContinuumMemory) -> None:
        """Export snapshot includes red line status."""
        memory.add("critical-1", "Critical content")
        memory.mark_red_line("critical-1", "Safety critical", promote_to_glacial=False)

        snapshot = memory.export_snapshot()

        entry = next(e for e in snapshot["entries"] if e["id"] == "critical-1")
        assert entry["red_line"] is True
        assert entry["red_line_reason"] == "Safety critical"


class TestContinuumMemoryRestore:
    """Test ContinuumMemory snapshot restoration."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_continuum.db"
            yield str(db_path)

    @pytest.fixture
    def memory(self, temp_db):
        """Create a ContinuumMemory instance for testing."""
        return ContinuumMemory(db_path=temp_db)

    def test_restore_empty_snapshot(self, memory: ContinuumMemory) -> None:
        """Restore empty snapshot does nothing."""
        snapshot = {"entries": []}
        result = memory.restore_snapshot(snapshot)

        assert result["restored"] == 0
        assert result["skipped"] == 0
        assert result["updated"] == 0

    def test_restore_invalid_snapshot(self, memory: ContinuumMemory) -> None:
        """Restore invalid snapshot returns zero counts."""
        result = memory.restore_snapshot({})

        assert result["restored"] == 0
        assert result["skipped"] == 0
        assert result["updated"] == 0

    def test_restore_snapshot_basic(self, memory: ContinuumMemory) -> None:
        """Restore snapshot creates entries."""
        snapshot = {
            "entries": [
                {
                    "id": "mem-1",
                    "tier": "fast",
                    "content": "Test content",
                    "importance": 0.8,
                    "surprise_score": 0.3,
                    "consolidation_score": 0.5,
                    "update_count": 5,
                    "success_count": 3,
                    "failure_count": 2,
                    "created_at": "2026-01-01T00:00:00",
                    "updated_at": "2026-01-15T00:00:00",
                    "metadata": {"key": "value"},
                }
            ]
        }

        result = memory.restore_snapshot(snapshot)

        assert result["restored"] == 1
        assert result["skipped"] == 0
        assert result["updated"] == 0

        # Verify entry was created
        entry = memory.get("mem-1")
        assert entry is not None
        assert entry.tier == MemoryTier.FAST
        assert entry.content == "Test content"
        assert entry.importance == 0.8

    def test_restore_snapshot_replace_mode(self, memory: ContinuumMemory) -> None:
        """Replace mode overwrites existing entries."""
        # Add existing entry
        memory.add("mem-1", "Original content", importance=0.5)

        snapshot = {
            "entries": [
                {
                    "id": "mem-1",
                    "tier": "fast",
                    "content": "Updated content",
                    "importance": 0.9,
                }
            ]
        }

        result = memory.restore_snapshot(snapshot, merge_mode="replace")

        assert result["restored"] == 1

        entry = memory.get("mem-1")
        assert entry.content == "Updated content"
        assert entry.importance == 0.9

    def test_restore_snapshot_keep_mode(self, memory: ContinuumMemory) -> None:
        """Keep mode preserves existing entries."""
        # Add existing entry
        memory.add("mem-1", "Original content", importance=0.5)

        snapshot = {
            "entries": [
                {
                    "id": "mem-1",
                    "tier": "fast",
                    "content": "Updated content",
                    "importance": 0.9,
                },
                {
                    "id": "mem-2",
                    "tier": "medium",
                    "content": "New content",
                    "importance": 0.7,
                },
            ]
        }

        result = memory.restore_snapshot(snapshot, merge_mode="keep")

        assert result["restored"] == 1  # Only mem-2
        assert result["skipped"] == 1  # mem-1 skipped

        entry = memory.get("mem-1")
        assert entry.content == "Original content"

        entry2 = memory.get("mem-2")
        assert entry2.content == "New content"

    def test_restore_snapshot_merge_mode(self, memory: ContinuumMemory) -> None:
        """Merge mode keeps higher importance entries."""
        # Add existing entry with high importance
        memory.add("mem-1", "High importance content", importance=0.9)
        # Add existing entry with low importance
        memory.add("mem-2", "Low importance content", importance=0.3)

        snapshot = {
            "entries": [
                {
                    "id": "mem-1",
                    "tier": "fast",
                    "content": "Lower importance",
                    "importance": 0.5,
                },
                {
                    "id": "mem-2",
                    "tier": "fast",
                    "content": "Higher importance",
                    "importance": 0.8,
                },
            ]
        }

        result = memory.restore_snapshot(snapshot, merge_mode="merge")

        assert result["skipped"] == 1  # mem-1 kept existing
        assert result["updated"] == 1  # mem-2 updated

        entry1 = memory.get("mem-1")
        assert entry1.importance == 0.9  # Kept higher

        entry2 = memory.get("mem-2")
        assert entry2.importance == 0.8  # Updated to higher

    def test_restore_snapshot_preserves_red_line(self, memory: ContinuumMemory) -> None:
        """Restore snapshot preserves red line status."""
        snapshot = {
            "entries": [
                {
                    "id": "critical-1",
                    "tier": "glacial",
                    "content": "Critical content",
                    "importance": 1.0,
                    "red_line": True,
                    "red_line_reason": "Safety critical",
                }
            ]
        }

        result = memory.restore_snapshot(snapshot)

        assert result["restored"] == 1

        entry = memory.get("critical-1")
        assert entry.red_line is True
        assert entry.red_line_reason == "Safety critical"


class TestRoundTripSnapshotRestore:
    """Test complete export/restore round-trip."""

    @pytest.fixture
    def temp_dbs(self):
        """Create temporary databases for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db1 = Path(tmpdir) / "source.db"
            db2 = Path(tmpdir) / "target.db"
            yield str(db1), str(db2)

    def test_full_round_trip(self, temp_dbs) -> None:
        """Full round-trip preserves all data."""
        source_db, target_db = temp_dbs
        source = ContinuumMemory(db_path=source_db)
        target = ContinuumMemory(db_path=target_db)

        # Add various entries to source
        source.add("entry-1", "Fast content", tier=MemoryTier.FAST, importance=0.9)
        source.add("entry-2", "Medium content", tier=MemoryTier.MEDIUM, importance=0.7)
        source.add("entry-3", "Slow content", tier=MemoryTier.SLOW, importance=0.5)
        source.mark_red_line("entry-1", "Critical")

        # Export and restore
        snapshot = source.export_snapshot()
        result = target.restore_snapshot(snapshot)

        assert result["restored"] == 3

        # Verify all entries restored correctly
        for entry_id in ["entry-1", "entry-2", "entry-3"]:
            source_entry = source.get(entry_id)
            target_entry = target.get(entry_id)

            assert target_entry is not None
            assert target_entry.tier == source_entry.tier
            assert target_entry.content == source_entry.content
            assert target_entry.importance == source_entry.importance

        # Verify red line preserved
        target_entry_1 = target.get("entry-1")
        assert target_entry_1.red_line is True


class TestDebateCheckpointMemoryState:
    """Test DebateCheckpoint with continuum_memory_state."""

    def test_checkpoint_includes_memory_state(self) -> None:
        """DebateCheckpoint can include continuum_memory_state."""
        memory_state = {
            "entries": [{"id": "mem-1", "content": "test"}],
            "tier_counts": {"fast": 1},
            "total_entries": 1,
        }

        checkpoint = DebateCheckpoint(
            checkpoint_id="cp-test-001",
            debate_id="debate-1",
            task="Test task",
            current_round=1,
            total_rounds=3,
            phase="proposal",
            messages=[],
            critiques=[],
            votes=[],
            agent_states=[],
            continuum_memory_state=memory_state,
        )

        assert checkpoint.continuum_memory_state == memory_state

    def test_checkpoint_serialization_with_memory(self) -> None:
        """Checkpoint serialization includes memory state."""
        memory_state = {
            "entries": [{"id": "mem-1", "content": "test", "importance": 0.8}],
            "tier_counts": {"fast": 1},
        }

        checkpoint = DebateCheckpoint(
            checkpoint_id="cp-test-001",
            debate_id="debate-1",
            task="Test task",
            current_round=1,
            total_rounds=3,
            phase="proposal",
            messages=[],
            critiques=[],
            votes=[],
            agent_states=[],
            continuum_memory_state=memory_state,
        )

        # Serialize
        checkpoint_dict = checkpoint.to_dict()
        assert "continuum_memory_state" in checkpoint_dict
        assert checkpoint_dict["continuum_memory_state"] == memory_state

        # Deserialize
        restored = DebateCheckpoint.from_dict(checkpoint_dict)
        assert restored.continuum_memory_state == memory_state

    def test_checkpoint_without_memory_state(self) -> None:
        """Checkpoint works without memory state."""
        checkpoint = DebateCheckpoint(
            checkpoint_id="cp-test-001",
            debate_id="debate-1",
            task="Test task",
            current_round=1,
            total_rounds=3,
            phase="proposal",
            messages=[],
            critiques=[],
            votes=[],
            agent_states=[],
        )

        assert checkpoint.continuum_memory_state is None

        checkpoint_dict = checkpoint.to_dict()
        assert checkpoint_dict["continuum_memory_state"] is None

        restored = DebateCheckpoint.from_dict(checkpoint_dict)
        assert restored.continuum_memory_state is None


class TestCheckpointManagerMemoryState:
    """Test CheckpointManager with memory state."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def manager(self, temp_dir):
        """Create a CheckpointManager for testing."""
        store = FileCheckpointStore(base_dir=temp_dir, compress=False)
        config = CheckpointConfig(
            interval_rounds=1,
            max_checkpoints=5,
            expiry_hours=1.0,
            compress=False,
        )
        return CheckpointManager(store=store, config=config)

    @pytest.mark.asyncio
    async def test_create_checkpoint_with_memory_state(self, manager) -> None:
        """CheckpointManager can create checkpoint with memory state."""
        from aragora.core import Message
        from datetime import datetime

        memory_state = {
            "entries": [{"id": "mem-1", "content": "test"}],
            "total_entries": 1,
        }

        # Create mock agent
        class MockAgent:
            name = "test-agent"
            model = "test-model"
            role = "proposer"
            system_prompt = "test prompt"
            stance = "neutral"

        checkpoint = await manager.create_checkpoint(
            debate_id="debate-123",
            task="Test task",
            current_round=1,
            total_rounds=3,
            phase="proposal",
            messages=[],
            critiques=[],
            votes=[],
            agents=[MockAgent()],
            continuum_memory_state=memory_state,
        )

        assert checkpoint.continuum_memory_state == memory_state

    @pytest.mark.asyncio
    async def test_resume_checkpoint_with_memory_state(self, manager) -> None:
        """CheckpointManager can resume checkpoint with memory state."""
        from aragora.core import Message
        from datetime import datetime

        memory_state = {
            "entries": [{"id": "mem-1", "content": "test", "importance": 0.9}],
            "total_entries": 1,
        }

        class MockAgent:
            name = "test-agent"
            model = "test-model"
            role = "proposer"
            system_prompt = "test prompt"
            stance = "neutral"

        # Create checkpoint
        checkpoint = await manager.create_checkpoint(
            debate_id="debate-123",
            task="Test task",
            current_round=1,
            total_rounds=3,
            phase="proposal",
            messages=[],
            critiques=[],
            votes=[],
            agents=[MockAgent()],
            continuum_memory_state=memory_state,
        )

        # Resume checkpoint
        resumed = await manager.resume_from_checkpoint(
            checkpoint.checkpoint_id,
            resumed_by="test-user",
        )

        assert resumed is not None
        assert resumed.checkpoint.continuum_memory_state == memory_state


class TestEndToEndCheckpointMemory:
    """End-to-end tests for checkpoint with memory restoration."""

    @pytest.fixture
    def temp_dirs(self):
        """Create temporary directories for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "continuum.db"
            checkpoint_dir = Path(tmpdir) / "checkpoints"
            checkpoint_dir.mkdir()
            yield str(db_path), str(checkpoint_dir)

    @pytest.mark.asyncio
    async def test_full_checkpoint_restore_cycle(self, temp_dirs) -> None:
        """Full cycle: create memory, checkpoint, restore to new memory."""
        db_path, checkpoint_dir = temp_dirs

        # Create source memory with entries
        source_memory = ContinuumMemory(db_path=db_path)
        source_memory.add("insight-1", "Important insight", tier=MemoryTier.FAST, importance=0.9)
        source_memory.add("pattern-1", "Recognized pattern", tier=MemoryTier.MEDIUM, importance=0.7)

        # Export memory state
        memory_snapshot = source_memory.export_snapshot()

        # Create checkpoint with memory state
        class MockAgent:
            name = "test-agent"
            model = "test-model"
            role = "proposer"
            system_prompt = "test prompt"
            stance = "neutral"

        store = FileCheckpointStore(base_dir=checkpoint_dir, compress=False)
        manager = CheckpointManager(store=store)

        checkpoint = await manager.create_checkpoint(
            debate_id="debate-end-to-end",
            task="End to end test",
            current_round=2,
            total_rounds=5,
            phase="critique",
            messages=[],
            critiques=[],
            votes=[],
            agents=[MockAgent()],
            continuum_memory_state=memory_snapshot,
        )

        # Simulate loading checkpoint (like resuming debate)
        resumed = await manager.resume_from_checkpoint(checkpoint.checkpoint_id)
        assert resumed is not None

        # Create new memory and restore from checkpoint
        with tempfile.TemporaryDirectory() as new_tmpdir:
            new_db_path = Path(new_tmpdir) / "restored.db"
            target_memory = ContinuumMemory(db_path=str(new_db_path))

            # Restore from checkpoint memory state
            restore_result = target_memory.restore_snapshot(
                resumed.checkpoint.continuum_memory_state
            )

            assert restore_result["restored"] == 2

            # Verify entries are restored
            insight = target_memory.get("insight-1")
            assert insight is not None
            assert insight.content == "Important insight"
            assert insight.importance == 0.9

            pattern = target_memory.get("pattern-1")
            assert pattern is not None
            assert pattern.content == "Recognized pattern"
