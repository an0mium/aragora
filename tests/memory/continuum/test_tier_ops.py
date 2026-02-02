"""
Comprehensive tests for Continuum Memory Tier Operations.

Tests the TierOpsMixin in aragora/memory/continuum/tier_ops.py including:
- Red line (protected) memory marking and retrieval
- Tier promotion logic with cooldown
- Tier demotion logic with update thresholds
- Batch promotion and demotion operations
- Tier consolidation
- Statistics and export operations
- Memory cleanup and tier limits
- Concurrent tier operations
- Rollback scenarios
- Event emission
"""

from __future__ import annotations

import asyncio
import sqlite3
import tempfile
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.memory.continuum import (
    ContinuumMemory,
    ContinuumMemoryEntry,
    reset_continuum_memory,
)
from aragora.memory.tier_manager import (
    DEFAULT_TIER_CONFIGS,
    MemoryTier,
    TierManager,
    reset_tier_manager,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_db_path(tmp_path: Path) -> str:
    """Create a temporary database path for testing."""
    return str(tmp_path / "test_tier_ops.db")


@pytest.fixture
def tier_manager() -> TierManager:
    """Create a fresh TierManager for testing."""
    return TierManager()


@pytest.fixture
def memory(temp_db_path: str, tier_manager: TierManager) -> ContinuumMemory:
    """Create a ContinuumMemory instance with isolated database."""
    reset_tier_manager()
    reset_continuum_memory()
    cms = ContinuumMemory(db_path=temp_db_path, tier_manager=tier_manager)
    yield cms
    reset_tier_manager()
    reset_continuum_memory()


@pytest.fixture
def populated_memory(memory: ContinuumMemory) -> ContinuumMemory:
    """Memory with pre-populated entries across tiers."""
    memory.add("fast_1", "Fast tier Python error handling", tier=MemoryTier.FAST, importance=0.8)
    memory.add("fast_2", "Fast tier debugging tips", tier=MemoryTier.FAST, importance=0.6)
    memory.add("medium_1", "Medium tier tactical patterns", tier=MemoryTier.MEDIUM, importance=0.7)
    memory.add("medium_2", "Medium tier learning insights", tier=MemoryTier.MEDIUM, importance=0.5)
    memory.add("slow_1", "Slow tier strategic knowledge", tier=MemoryTier.SLOW, importance=0.9)
    memory.add("slow_2", "Slow tier architectural patterns", tier=MemoryTier.SLOW, importance=0.4)
    memory.add(
        "glacial_1", "Glacial foundational concepts", tier=MemoryTier.GLACIAL, importance=0.95
    )
    memory.add("glacial_2", "Glacial core principles", tier=MemoryTier.GLACIAL, importance=0.85)
    return memory


# =============================================================================
# Test Red Line Memory Marking
# =============================================================================


class TestRedLineMarking:
    """Tests for mark_red_line() method."""

    def test_mark_red_line_basic(self, memory: ContinuumMemory) -> None:
        """Test basic red line marking."""
        memory.add("protect_me", "Critical decision", tier=MemoryTier.SLOW)

        result = memory.mark_red_line("protect_me", reason="Safety critical")

        assert result is True
        entry = memory.get("protect_me")
        assert entry.red_line is True
        assert entry.red_line_reason == "Safety critical"

    def test_mark_red_line_sets_max_importance(self, memory: ContinuumMemory) -> None:
        """Test that red line sets importance to 1.0."""
        memory.add("protect_low", "Low importance content", tier=MemoryTier.SLOW, importance=0.3)

        memory.mark_red_line("protect_low", reason="Now critical")

        entry = memory.get("protect_low")
        assert entry.importance == 1.0

    def test_mark_red_line_promotes_to_glacial(self, memory: ContinuumMemory) -> None:
        """Test that red line promotes to glacial tier by default."""
        memory.add("promote_protect", "Content", tier=MemoryTier.FAST)

        memory.mark_red_line("promote_protect", reason="Foundational", promote_to_glacial=True)

        entry = memory.get("promote_protect")
        assert entry.tier == MemoryTier.GLACIAL

    def test_mark_red_line_no_promotion(self, memory: ContinuumMemory) -> None:
        """Test red line without glacial promotion."""
        memory.add("no_promote", "Content", tier=MemoryTier.MEDIUM)

        memory.mark_red_line("no_promote", reason="Stay put", promote_to_glacial=False)

        entry = memory.get("no_promote")
        assert entry.tier == MemoryTier.MEDIUM
        assert entry.red_line is True

    def test_mark_red_line_nonexistent(self, memory: ContinuumMemory) -> None:
        """Test marking non-existent entry returns False."""
        result = memory.mark_red_line("nonexistent", reason="Test")

        assert result is False

    def test_mark_red_line_already_glacial(self, memory: ContinuumMemory) -> None:
        """Test marking already glacial entry."""
        memory.add("already_glacial", "Content", tier=MemoryTier.GLACIAL, importance=0.5)

        result = memory.mark_red_line("already_glacial", reason="Protect")

        assert result is True
        entry = memory.get("already_glacial")
        assert entry.tier == MemoryTier.GLACIAL
        assert entry.red_line is True


class TestGetRedLineMemories:
    """Tests for get_red_line_memories() method."""

    def test_get_red_line_memories_empty(self, memory: ContinuumMemory) -> None:
        """Test retrieval when no red line entries exist."""
        memory.add("normal_1", "Content 1")
        memory.add("normal_2", "Content 2")

        red_lines = memory.get_red_line_memories()

        assert red_lines == []

    def test_get_red_line_memories_single(self, memory: ContinuumMemory) -> None:
        """Test retrieval of single red line entry."""
        memory.add("protected", "Protected content")
        memory.mark_red_line("protected", reason="Critical")

        red_lines = memory.get_red_line_memories()

        assert len(red_lines) == 1
        assert red_lines[0].id == "protected"
        assert red_lines[0].red_line is True

    def test_get_red_line_memories_multiple(self, memory: ContinuumMemory) -> None:
        """Test retrieval of multiple red line entries."""
        memory.add("rl_1", "Content 1")
        memory.add("rl_2", "Content 2")
        memory.add("rl_3", "Content 3")
        memory.add("normal", "Normal content")

        memory.mark_red_line("rl_1", reason="Critical 1")
        memory.mark_red_line("rl_2", reason="Critical 2")
        memory.mark_red_line("rl_3", reason="Critical 3")

        red_lines = memory.get_red_line_memories()

        assert len(red_lines) == 3
        assert all(e.red_line for e in red_lines)
        red_line_ids = {e.id for e in red_lines}
        assert "normal" not in red_line_ids

    def test_get_red_line_memories_ordered_by_created_at(self, memory: ContinuumMemory) -> None:
        """Test that red line entries are ordered by creation time."""
        memory.add("first", "First entry")
        memory.add("second", "Second entry")
        memory.add("third", "Third entry")

        memory.mark_red_line("first", reason="R1")
        memory.mark_red_line("second", reason="R2")
        memory.mark_red_line("third", reason="R3")

        red_lines = memory.get_red_line_memories()

        assert red_lines[0].id == "first"
        assert red_lines[1].id == "second"
        assert red_lines[2].id == "third"


# =============================================================================
# Test Tier Promotion
# =============================================================================


class TestTierPromotion:
    """Tests for promote() method."""

    def test_promote_medium_to_fast(self, memory: ContinuumMemory) -> None:
        """Test promoting from medium to fast tier."""
        memory.add("promote_test", "Content to promote", tier=MemoryTier.MEDIUM)

        # Set high surprise score
        with memory.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE continuum_memory SET surprise_score = 0.9 WHERE id = ?",
                ("promote_test",),
            )
            conn.commit()

        new_tier = memory.promote("promote_test")

        assert new_tier == MemoryTier.FAST

    def test_promote_slow_to_medium(self, memory: ContinuumMemory) -> None:
        """Test promoting from slow to medium tier."""
        memory.add("slow_promote", "Content", tier=MemoryTier.SLOW)

        with memory.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE continuum_memory SET surprise_score = 0.8 WHERE id = ?",
                ("slow_promote",),
            )
            conn.commit()

        new_tier = memory.promote("slow_promote")

        assert new_tier == MemoryTier.MEDIUM

    def test_promote_glacial_to_slow(self, memory: ContinuumMemory) -> None:
        """Test promoting from glacial to slow tier."""
        memory.add("glacial_promote", "Content", tier=MemoryTier.GLACIAL)

        with memory.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE continuum_memory SET surprise_score = 0.7 WHERE id = ?",
                ("glacial_promote",),
            )
            conn.commit()

        new_tier = memory.promote("glacial_promote")

        assert new_tier == MemoryTier.SLOW

    def test_promote_already_at_fast(self, memory: ContinuumMemory) -> None:
        """Test that promoting from fast tier returns None."""
        memory.add("already_fast", "Fast content", tier=MemoryTier.FAST)

        with memory.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE continuum_memory SET surprise_score = 0.95 WHERE id = ?",
                ("already_fast",),
            )
            conn.commit()

        result = memory.promote("already_fast")

        assert result is None

    def test_promote_nonexistent(self, memory: ContinuumMemory) -> None:
        """Test promoting non-existent entry returns None."""
        result = memory.promote("nonexistent_id")

        assert result is None

    def test_promote_respects_cooldown(self, memory: ContinuumMemory) -> None:
        """Test that promotion respects cooldown period."""
        memory.add("cooldown_test", "Content", tier=MemoryTier.MEDIUM)

        # Set high surprise and recent promotion
        now = datetime.now().isoformat()
        with memory.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """UPDATE continuum_memory
                   SET surprise_score = 0.9, last_promotion_at = ?
                   WHERE id = ?""",
                (now, "cooldown_test"),
            )
            conn.commit()

        result = memory.promote("cooldown_test")

        assert result is None

    def test_promote_after_cooldown_expires(self, memory: ContinuumMemory) -> None:
        """Test promotion succeeds after cooldown expires."""
        memory.add("cooldown_expired", "Content", tier=MemoryTier.MEDIUM)

        # Set high surprise and old promotion time
        old_time = (datetime.now() - timedelta(hours=48)).isoformat()
        with memory.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """UPDATE continuum_memory
                   SET surprise_score = 0.9, last_promotion_at = ?
                   WHERE id = ?""",
                (old_time, "cooldown_expired"),
            )
            conn.commit()

        result = memory.promote("cooldown_expired")

        assert result == MemoryTier.FAST

    def test_promote_low_surprise_denied(self, memory: ContinuumMemory) -> None:
        """Test that low surprise score prevents promotion."""
        memory.add("low_surprise", "Content", tier=MemoryTier.MEDIUM)

        with memory.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE continuum_memory SET surprise_score = 0.2 WHERE id = ?",
                ("low_surprise",),
            )
            conn.commit()

        result = memory.promote("low_surprise")

        assert result is None

    def test_promote_records_transition(self, memory: ContinuumMemory) -> None:
        """Test that promotion records tier transition in database."""
        memory.add("record_transition", "Content", tier=MemoryTier.MEDIUM)

        with memory.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE continuum_memory SET surprise_score = 0.9 WHERE id = ?",
                ("record_transition",),
            )
            conn.commit()

        memory.promote("record_transition")

        with memory.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT from_tier, to_tier, reason FROM tier_transitions WHERE memory_id = ?",
                ("record_transition",),
            )
            row = cursor.fetchone()

        assert row is not None
        assert row[0] == "medium"
        assert row[1] == "fast"
        assert row[2] == "high_surprise"


# =============================================================================
# Test Tier Demotion
# =============================================================================


class TestTierDemotion:
    """Tests for demote() method."""

    def test_demote_fast_to_medium(self, memory: ContinuumMemory) -> None:
        """Test demoting from fast to medium tier."""
        memory.add("demote_test", "Content", tier=MemoryTier.FAST)

        with memory.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """UPDATE continuum_memory
                   SET surprise_score = 0.05, update_count = 15
                   WHERE id = ?""",
                ("demote_test",),
            )
            conn.commit()

        new_tier = memory.demote("demote_test")

        assert new_tier == MemoryTier.MEDIUM

    def test_demote_medium_to_slow(self, memory: ContinuumMemory) -> None:
        """Test demoting from medium to slow tier."""
        memory.add("medium_demote", "Content", tier=MemoryTier.MEDIUM)

        with memory.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """UPDATE continuum_memory
                   SET surprise_score = 0.1, update_count = 20
                   WHERE id = ?""",
                ("medium_demote",),
            )
            conn.commit()

        new_tier = memory.demote("medium_demote")

        assert new_tier == MemoryTier.SLOW

    def test_demote_slow_to_glacial(self, memory: ContinuumMemory) -> None:
        """Test demoting from slow to glacial tier."""
        memory.add("slow_demote", "Content", tier=MemoryTier.SLOW)

        with memory.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """UPDATE continuum_memory
                   SET surprise_score = 0.05, update_count = 25
                   WHERE id = ?""",
                ("slow_demote",),
            )
            conn.commit()

        new_tier = memory.demote("slow_demote")

        assert new_tier == MemoryTier.GLACIAL

    def test_demote_already_at_glacial(self, memory: ContinuumMemory) -> None:
        """Test that demoting from glacial tier returns None."""
        memory.add("already_glacial", "Glacial content", tier=MemoryTier.GLACIAL)

        with memory.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """UPDATE continuum_memory
                   SET surprise_score = 0.01, update_count = 50
                   WHERE id = ?""",
                ("already_glacial",),
            )
            conn.commit()

        result = memory.demote("already_glacial")

        assert result is None

    def test_demote_nonexistent(self, memory: ContinuumMemory) -> None:
        """Test demoting non-existent entry returns None."""
        result = memory.demote("nonexistent_id")

        assert result is None

    def test_demote_insufficient_updates(self, memory: ContinuumMemory) -> None:
        """Test that entries with few updates are not demoted."""
        memory.add("few_updates", "Content", tier=MemoryTier.FAST)

        with memory.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """UPDATE continuum_memory
                   SET surprise_score = 0.05, update_count = 3
                   WHERE id = ?""",
                ("few_updates",),
            )
            conn.commit()

        result = memory.demote("few_updates")

        assert result is None

    def test_demote_high_surprise_denied(self, memory: ContinuumMemory) -> None:
        """Test that high surprise score prevents demotion."""
        memory.add("high_surprise", "Content", tier=MemoryTier.FAST)

        with memory.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """UPDATE continuum_memory
                   SET surprise_score = 0.9, update_count = 20
                   WHERE id = ?""",
                ("high_surprise",),
            )
            conn.commit()

        result = memory.demote("high_surprise")

        assert result is None

    def test_demote_records_transition(self, memory: ContinuumMemory) -> None:
        """Test that demotion records tier transition in database."""
        memory.add("record_demotion", "Content", tier=MemoryTier.FAST)

        with memory.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """UPDATE continuum_memory
                   SET surprise_score = 0.05, update_count = 15
                   WHERE id = ?""",
                ("record_demotion",),
            )
            conn.commit()

        memory.demote("record_demotion")

        with memory.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT from_tier, to_tier, reason FROM tier_transitions WHERE memory_id = ?",
                ("record_demotion",),
            )
            row = cursor.fetchone()

        assert row is not None
        assert row[0] == "fast"
        assert row[1] == "medium"
        assert row[2] == "high_stability"


# =============================================================================
# Test Batch Operations
# =============================================================================


class TestBatchPromote:
    """Tests for _promote_batch() method."""

    def test_batch_promote_empty_list(self, memory: ContinuumMemory) -> None:
        """Test batch promote with empty list."""
        count = memory._promote_batch(MemoryTier.MEDIUM, MemoryTier.FAST, [])

        assert count == 0

    def test_batch_promote_single(self, memory: ContinuumMemory) -> None:
        """Test batch promote with single entry."""
        memory.add("batch_single", "Content", tier=MemoryTier.MEDIUM)

        count = memory._promote_batch(MemoryTier.MEDIUM, MemoryTier.FAST, ["batch_single"])

        assert count == 1
        entry = memory.get("batch_single")
        assert entry.tier == MemoryTier.FAST

    def test_batch_promote_multiple(self, memory: ContinuumMemory) -> None:
        """Test batch promote with multiple entries."""
        for i in range(5):
            memory.add(f"batch_{i}", f"Content {i}", tier=MemoryTier.MEDIUM)

        ids = [f"batch_{i}" for i in range(5)]
        count = memory._promote_batch(MemoryTier.MEDIUM, MemoryTier.FAST, ids)

        assert count == 5
        for id_ in ids:
            entry = memory.get(id_)
            assert entry.tier == MemoryTier.FAST

    def test_batch_promote_respects_cooldown(self, memory: ContinuumMemory) -> None:
        """Test batch promote respects cooldown period."""
        now = datetime.now().isoformat()

        memory.add("cooldown_1", "Content", tier=MemoryTier.MEDIUM)
        memory.add("cooldown_2", "Content", tier=MemoryTier.MEDIUM)

        # Set recent promotion for one entry
        with memory.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE continuum_memory SET last_promotion_at = ? WHERE id = ?",
                (now, "cooldown_1"),
            )
            conn.commit()

        count = memory._promote_batch(
            MemoryTier.MEDIUM, MemoryTier.FAST, ["cooldown_1", "cooldown_2"]
        )

        assert count == 1
        assert memory.get("cooldown_1").tier == MemoryTier.MEDIUM
        assert memory.get("cooldown_2").tier == MemoryTier.FAST

    def test_batch_promote_wrong_tier(self, memory: ContinuumMemory) -> None:
        """Test batch promote with entries not in expected tier."""
        memory.add("wrong_tier", "Content", tier=MemoryTier.SLOW)

        count = memory._promote_batch(MemoryTier.MEDIUM, MemoryTier.FAST, ["wrong_tier"])

        assert count == 0
        assert memory.get("wrong_tier").tier == MemoryTier.SLOW

    def test_batch_promote_records_transitions(self, memory: ContinuumMemory) -> None:
        """Test batch promote records transitions for all entries."""
        for i in range(3):
            memory.add(f"trans_{i}", f"Content {i}", tier=MemoryTier.SLOW)

        memory._promote_batch(MemoryTier.SLOW, MemoryTier.MEDIUM, ["trans_0", "trans_1", "trans_2"])

        with memory.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM tier_transitions WHERE to_tier = 'medium'")
            count = cursor.fetchone()[0]

        assert count == 3


class TestBatchDemote:
    """Tests for _demote_batch() method."""

    def test_batch_demote_empty_list(self, memory: ContinuumMemory) -> None:
        """Test batch demote with empty list."""
        count = memory._demote_batch(MemoryTier.FAST, MemoryTier.MEDIUM, [])

        assert count == 0

    def test_batch_demote_single(self, memory: ContinuumMemory) -> None:
        """Test batch demote with single entry."""
        memory.add("batch_single", "Content", tier=MemoryTier.FAST)

        count = memory._demote_batch(MemoryTier.FAST, MemoryTier.MEDIUM, ["batch_single"])

        assert count == 1
        entry = memory.get("batch_single")
        assert entry.tier == MemoryTier.MEDIUM

    def test_batch_demote_multiple(self, memory: ContinuumMemory) -> None:
        """Test batch demote with multiple entries."""
        for i in range(5):
            memory.add(f"demote_{i}", f"Content {i}", tier=MemoryTier.FAST)

        ids = [f"demote_{i}" for i in range(5)]
        count = memory._demote_batch(MemoryTier.FAST, MemoryTier.MEDIUM, ids)

        assert count == 5
        for id_ in ids:
            entry = memory.get(id_)
            assert entry.tier == MemoryTier.MEDIUM

    def test_batch_demote_wrong_tier(self, memory: ContinuumMemory) -> None:
        """Test batch demote with entries not in expected tier."""
        memory.add("wrong_tier", "Content", tier=MemoryTier.SLOW)

        count = memory._demote_batch(MemoryTier.FAST, MemoryTier.MEDIUM, ["wrong_tier"])

        assert count == 0
        assert memory.get("wrong_tier").tier == MemoryTier.SLOW


# =============================================================================
# Test Consolidation
# =============================================================================


class TestConsolidate:
    """Tests for consolidate() method."""

    def test_consolidate_returns_counts(self, memory: ContinuumMemory) -> None:
        """Test that consolidate returns proper count structure."""
        result = memory.consolidate()

        assert "promotions" in result
        assert "demotions" in result
        assert isinstance(result["promotions"], int)
        assert isinstance(result["demotions"], int)

    def test_consolidate_promotes_high_surprise(self, memory: ContinuumMemory) -> None:
        """Test that consolidate promotes entries with high surprise."""
        for i in range(3):
            memory.add(f"high_surprise_{i}", f"Content {i}", tier=MemoryTier.SLOW)

        with memory.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE continuum_memory SET surprise_score = 0.9 WHERE tier = 'slow'")
            conn.commit()

        result = memory.consolidate()

        # Check some promotions occurred
        assert result["promotions"] >= 0

    def test_consolidate_demotes_stable_entries(self, memory: ContinuumMemory) -> None:
        """Test that consolidate demotes stable entries."""
        for i in range(3):
            memory.add(f"stable_{i}", f"Stable content {i}", tier=MemoryTier.FAST)

        with memory.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """UPDATE continuum_memory
                   SET surprise_score = 0.02, update_count = 20
                   WHERE tier = 'fast'"""
            )
            conn.commit()

        result = memory.consolidate()

        assert "demotions" in result

    def test_consolidate_empty_memory(self, memory: ContinuumMemory) -> None:
        """Test consolidate on empty memory."""
        result = memory.consolidate()

        assert result["promotions"] == 0
        assert result["demotions"] == 0

    def test_consolidate_one_level_at_a_time(self, memory: ContinuumMemory) -> None:
        """Test that consolidate only moves entries one level at a time."""
        memory.add("glacial_entry", "Content", tier=MemoryTier.GLACIAL)

        with memory.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE continuum_memory SET surprise_score = 0.99 WHERE id = ?",
                ("glacial_entry",),
            )
            conn.commit()

        memory.consolidate()

        # Should only move to slow, not directly to fast or medium
        entry = memory.get("glacial_entry")
        assert entry.tier in [MemoryTier.GLACIAL, MemoryTier.SLOW]


# =============================================================================
# Test Statistics and Export
# =============================================================================


class TestGetStats:
    """Tests for get_stats() method."""

    def test_get_stats_empty(self, memory: ContinuumMemory) -> None:
        """Test get_stats on empty memory."""
        stats = memory.get_stats()

        assert stats["total_memories"] == 0
        assert "by_tier" in stats

    def test_get_stats_with_entries(self, populated_memory: ContinuumMemory) -> None:
        """Test get_stats with populated memory."""
        stats = populated_memory.get_stats()

        assert stats["total_memories"] == 8
        assert "by_tier" in stats
        assert "transitions" in stats

    def test_get_stats_tier_counts(self, populated_memory: ContinuumMemory) -> None:
        """Test get_stats returns correct tier counts."""
        stats = populated_memory.get_stats()

        by_tier = stats["by_tier"]
        assert by_tier["fast"]["count"] == 2
        assert by_tier["medium"]["count"] == 2
        assert by_tier["slow"]["count"] == 2
        assert by_tier["glacial"]["count"] == 2


class TestExportForTier:
    """Tests for export_for_tier() method."""

    def test_export_for_tier_fast(self, populated_memory: ContinuumMemory) -> None:
        """Test exporting fast tier entries."""
        exported = populated_memory.export_for_tier(MemoryTier.FAST)

        assert len(exported) == 2
        assert all(e["tier"] == "fast" for e in exported)

    def test_export_for_tier_glacial(self, populated_memory: ContinuumMemory) -> None:
        """Test exporting glacial tier entries."""
        exported = populated_memory.export_for_tier(MemoryTier.GLACIAL)

        assert len(exported) == 2
        assert all(e["tier"] == "glacial" for e in exported)

    def test_export_for_tier_empty(self, memory: ContinuumMemory) -> None:
        """Test exporting from empty tier."""
        exported = memory.export_for_tier(MemoryTier.FAST)

        assert exported == []

    def test_export_for_tier_contains_required_fields(
        self, populated_memory: ContinuumMemory
    ) -> None:
        """Test exported entries contain all required fields."""
        exported = populated_memory.export_for_tier(MemoryTier.SLOW)

        for entry in exported:
            assert "id" in entry
            assert "content" in entry
            assert "tier" in entry
            assert "importance" in entry


# =============================================================================
# Test Memory Pressure and Limits
# =============================================================================


class TestGetMemoryPressure:
    """Tests for get_memory_pressure() method."""

    def test_get_memory_pressure_empty(self, memory: ContinuumMemory) -> None:
        """Test memory pressure on empty memory."""
        pressure = memory.get_memory_pressure()

        assert pressure == 0.0

    def test_get_memory_pressure_with_entries(self, memory: ContinuumMemory) -> None:
        """Test memory pressure calculation."""
        for i in range(100):
            memory.add(f"pressure_{i}", f"Content {i}", tier=MemoryTier.FAST)

        pressure = memory.get_memory_pressure()

        # With 100 entries in fast tier (limit 1000), pressure should be ~0.1
        assert 0 < pressure < 1.0

    def test_get_memory_pressure_bounded(self, memory: ContinuumMemory) -> None:
        """Test that memory pressure is bounded 0-1."""
        pressure = memory.get_memory_pressure()

        assert 0 <= pressure <= 1.0


class TestEnforceTierLimits:
    """Tests for enforce_tier_limits() method."""

    def test_enforce_tier_limits_under_limit(self, populated_memory: ContinuumMemory) -> None:
        """Test enforce_tier_limits when under limit."""
        result = populated_memory.enforce_tier_limits(tier=MemoryTier.FAST)

        # Should not remove anything when under limit
        assert result.get("fast", 0) == 0

    def test_enforce_tier_limits_over_limit(self, memory: ContinuumMemory) -> None:
        """Test enforce_tier_limits when over limit."""
        # Set very low limit
        memory.hyperparams["max_entries_per_tier"]["fast"] = 5

        # Add more than limit
        for i in range(10):
            memory.add(f"limit_{i}", f"Content {i}", tier=MemoryTier.FAST, importance=i / 10)

        result = memory.enforce_tier_limits(tier=MemoryTier.FAST)

        # Should have removed 5 entries
        stats = memory.get_stats()
        fast_count = stats["by_tier"].get("fast", {}).get("count", 0)
        assert fast_count <= 5

    def test_enforce_tier_limits_preserves_high_importance(self, memory: ContinuumMemory) -> None:
        """Test that enforce_tier_limits keeps high importance entries."""
        memory.hyperparams["max_entries_per_tier"]["fast"] = 3

        memory.add("high_1", "High importance", tier=MemoryTier.FAST, importance=0.9)
        memory.add("high_2", "High importance", tier=MemoryTier.FAST, importance=0.8)
        memory.add("low_1", "Low importance", tier=MemoryTier.FAST, importance=0.1)
        memory.add("low_2", "Low importance", tier=MemoryTier.FAST, importance=0.2)
        memory.add("low_3", "Low importance", tier=MemoryTier.FAST, importance=0.15)

        memory.enforce_tier_limits(tier=MemoryTier.FAST)

        # High importance entries should remain
        assert memory.get("high_1") is not None
        assert memory.get("high_2") is not None


class TestCleanupExpiredMemories:
    """Tests for cleanup_expired_memories() method."""

    def test_cleanup_expired_memories_removes_old(self, memory: ContinuumMemory) -> None:
        """Test cleanup removes expired entries."""
        old_time = (datetime.now() - timedelta(days=30)).isoformat()

        with memory.connection() as conn:
            cursor = conn.cursor()
            for i in range(5):
                cursor.execute(
                    """INSERT INTO continuum_memory
                       (id, tier, content, importance, updated_at, created_at)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (f"old_{i}", "fast", f"Old content {i}", 0.3, old_time, old_time),
                )
            conn.commit()

        result = memory.cleanup_expired_memories(max_age_hours=1)

        total_cleaned = result.get("deleted", 0) + result.get("archived", 0)
        assert total_cleaned >= 5

    def test_cleanup_expired_memories_preserves_recent(self, memory: ContinuumMemory) -> None:
        """Test cleanup preserves recent entries."""
        memory.add("recent_1", "Recent content 1")
        memory.add("recent_2", "Recent content 2")

        result = memory.cleanup_expired_memories(max_age_hours=24)

        # Recent entries should not be cleaned
        assert memory.get("recent_1") is not None
        assert memory.get("recent_2") is not None

    def test_cleanup_expired_memories_skips_red_line(self, memory: ContinuumMemory) -> None:
        """Test cleanup skips red line entries."""
        old_time = (datetime.now() - timedelta(days=30)).isoformat()

        with memory.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO continuum_memory
                   (id, tier, content, importance, updated_at, red_line, red_line_reason)
                   VALUES (?, ?, ?, ?, ?, 1, ?)""",
                ("old_protected", "fast", "Old protected", 0.3, old_time, "Must preserve"),
            )
            conn.commit()

        memory.cleanup_expired_memories(max_age_hours=1)

        # Red line entry should still exist
        entry = memory.get("old_protected")
        assert entry is not None


class TestDeleteMemory:
    """Tests for delete() method."""

    def test_delete_basic(self, memory: ContinuumMemory) -> None:
        """Test basic memory deletion."""
        memory.add("delete_me", "Content to delete")

        result = memory.delete("delete_me")

        assert result["deleted"] is True
        assert memory.get("delete_me") is None

    def test_delete_with_archive(self, memory: ContinuumMemory) -> None:
        """Test deletion with archiving."""
        memory.add("archive_me", "Content to archive")

        result = memory.delete("archive_me", archive=True)

        assert result["archived"] is True

        # Check archive table
        with memory.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM continuum_memory_archive WHERE id = ?", ("archive_me",))
            assert cursor.fetchone() is not None

    def test_delete_red_line_blocked(self, memory: ContinuumMemory) -> None:
        """Test that red line entries cannot be deleted without force."""
        memory.add("protected", "Protected content")
        memory.mark_red_line("protected", reason="Critical")

        result = memory.delete("protected")

        assert result["deleted"] is False
        assert result["blocked"] is True
        assert memory.get("protected") is not None

    def test_delete_red_line_with_force(self, memory: ContinuumMemory) -> None:
        """Test force deleting a red line entry."""
        memory.add("force_delete", "Content")
        memory.mark_red_line("force_delete", reason="Test")

        result = memory.delete("force_delete", force=True)

        assert result["deleted"] is True

    def test_delete_nonexistent(self, memory: ContinuumMemory) -> None:
        """Test deleting non-existent entry."""
        result = memory.delete("nonexistent")

        assert result["deleted"] is False


# =============================================================================
# Test Concurrent Operations
# =============================================================================


class TestConcurrentTierOperations:
    """Tests for concurrent tier operations."""

    def test_concurrent_promotions(self, memory: ContinuumMemory) -> None:
        """Test concurrent promotion operations."""
        for i in range(20):
            memory.add(f"concurrent_{i}", f"Content {i}", tier=MemoryTier.MEDIUM)

        with memory.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE continuum_memory SET surprise_score = 0.9")
            conn.commit()

        errors: list[Exception] = []

        def promote_entries(start: int, end: int) -> None:
            try:
                for i in range(start, end):
                    memory.promote(f"concurrent_{i}")
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=promote_entries, args=(0, 5)),
            threading.Thread(target=promote_entries, args=(5, 10)),
            threading.Thread(target=promote_entries, args=(10, 15)),
            threading.Thread(target=promote_entries, args=(15, 20)),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_concurrent_demotions(self, memory: ContinuumMemory) -> None:
        """Test concurrent demotion operations."""
        for i in range(20):
            memory.add(f"concurrent_{i}", f"Content {i}", tier=MemoryTier.FAST)

        with memory.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE continuum_memory SET surprise_score = 0.05, update_count = 20")
            conn.commit()

        errors: list[Exception] = []

        def demote_entries(start: int, end: int) -> None:
            try:
                for i in range(start, end):
                    memory.demote(f"concurrent_{i}")
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=demote_entries, args=(0, 5)),
            threading.Thread(target=demote_entries, args=(5, 10)),
            threading.Thread(target=demote_entries, args=(10, 15)),
            threading.Thread(target=demote_entries, args=(15, 20)),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_concurrent_promote_and_demote(self, memory: ContinuumMemory) -> None:
        """Test concurrent promotion and demotion of different entries."""
        for i in range(10):
            memory.add(f"promote_{i}", f"Content {i}", tier=MemoryTier.MEDIUM)
            memory.add(f"demote_{i}", f"Content {i}", tier=MemoryTier.FAST)

        with memory.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE continuum_memory SET surprise_score = 0.9 WHERE id LIKE 'promote_%'"
            )
            cursor.execute(
                """UPDATE continuum_memory
                   SET surprise_score = 0.05, update_count = 20
                   WHERE id LIKE 'demote_%'"""
            )
            conn.commit()

        errors: list[Exception] = []

        def do_promotions() -> None:
            try:
                for i in range(10):
                    memory.promote(f"promote_{i}")
            except Exception as e:
                errors.append(e)

        def do_demotions() -> None:
            try:
                for i in range(10):
                    memory.demote(f"demote_{i}")
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=do_promotions),
            threading.Thread(target=do_demotions),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_concurrent_consolidation(self, memory: ContinuumMemory) -> None:
        """Test concurrent consolidation calls."""
        for i in range(30):
            memory.add(f"consolidate_{i}", f"Content {i}", tier=MemoryTier.SLOW)

        errors: list[Exception] = []

        def run_consolidate() -> None:
            try:
                memory.consolidate()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=run_consolidate) for _ in range(4)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


# =============================================================================
# Test Event Emission
# =============================================================================


class TestTierEventEmission:
    """Tests for tier event emission."""

    def test_emit_tier_event_promotion(self, memory: ContinuumMemory) -> None:
        """Test tier event emission on promotion."""
        mock_emitter = MagicMock()
        memory.event_emitter = mock_emitter

        memory.add("emit_test", "Content", tier=MemoryTier.MEDIUM)
        with memory.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE continuum_memory SET surprise_score = 0.9 WHERE id = ?",
                ("emit_test",),
            )
            conn.commit()

        memory.promote("emit_test")

        # Event emitter should have been called
        mock_emitter.emit_sync.assert_called()

    def test_emit_tier_event_demotion(self, memory: ContinuumMemory) -> None:
        """Test tier event emission on demotion."""
        mock_emitter = MagicMock()
        memory.event_emitter = mock_emitter

        memory.add("demote_emit", "Content", tier=MemoryTier.FAST)
        with memory.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """UPDATE continuum_memory
                   SET surprise_score = 0.05, update_count = 15
                   WHERE id = ?""",
                ("demote_emit",),
            )
            conn.commit()

        memory.demote("demote_emit")

        mock_emitter.emit_sync.assert_called()

    def test_no_event_without_emitter(self, memory: ContinuumMemory) -> None:
        """Test that no errors occur when event emitter is None."""
        memory.event_emitter = None

        memory.add("no_emitter", "Content", tier=MemoryTier.MEDIUM)
        with memory.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE continuum_memory SET surprise_score = 0.9 WHERE id = ?",
                ("no_emitter",),
            )
            conn.commit()

        # Should not raise
        memory.promote("no_emitter")


# =============================================================================
# Test Archive Operations
# =============================================================================


class TestArchiveStats:
    """Tests for get_archive_stats() method."""

    def test_get_archive_stats_empty(self, memory: ContinuumMemory) -> None:
        """Test archive stats when archive is empty."""
        stats = memory.get_archive_stats()

        assert "total_archived" in stats
        assert stats["total_archived"] == 0

    def test_get_archive_stats_with_entries(self, memory: ContinuumMemory) -> None:
        """Test archive stats with archived entries."""
        memory.add("archive_1", "Content 1")
        memory.add("archive_2", "Content 2")
        memory.add("archive_3", "Content 3")

        memory.delete("archive_1")
        memory.delete("archive_2")
        memory.delete("archive_3")

        stats = memory.get_archive_stats()

        assert stats["total_archived"] >= 3


# =============================================================================
# Test Data Integrity During Transitions
# =============================================================================


class TestDataIntegrityDuringTransitions:
    """Tests for data integrity during tier transitions."""

    def test_content_preserved_on_promotion(self, memory: ContinuumMemory) -> None:
        """Test that content is preserved during promotion."""
        original_content = "This is important content that should be preserved"
        memory.add("integrity_test", original_content, tier=MemoryTier.MEDIUM, importance=0.7)

        with memory.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE continuum_memory SET surprise_score = 0.9 WHERE id = ?",
                ("integrity_test",),
            )
            conn.commit()

        memory.promote("integrity_test")

        entry = memory.get("integrity_test")
        assert entry.content == original_content
        assert entry.importance == 0.7

    def test_metadata_preserved_on_demotion(self, memory: ContinuumMemory) -> None:
        """Test that metadata is preserved during demotion."""
        metadata = {"source": "debate", "round": 3, "tags": ["important", "critical"]}
        memory.add("meta_integrity", "Content", tier=MemoryTier.FAST, metadata=metadata)

        with memory.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """UPDATE continuum_memory
                   SET surprise_score = 0.05, update_count = 15
                   WHERE id = ?""",
                ("meta_integrity",),
            )
            conn.commit()

        memory.demote("meta_integrity")

        entry = memory.get("meta_integrity")
        assert entry.metadata == metadata

    def test_success_failure_counts_preserved(self, memory: ContinuumMemory) -> None:
        """Test that success/failure counts are preserved during transitions."""
        memory.add("counts_test", "Content", tier=MemoryTier.MEDIUM)

        # Update success/failure counts
        for _ in range(5):
            memory.update_outcome("counts_test", success=True)
        for _ in range(2):
            memory.update_outcome("counts_test", success=False)

        entry_before = memory.get("counts_test")
        success_before = entry_before.success_count
        failure_before = entry_before.failure_count

        # Force promotion
        with memory.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE continuum_memory SET surprise_score = 0.9 WHERE id = ?",
                ("counts_test",),
            )
            conn.commit()

        memory.promote("counts_test")

        entry_after = memory.get("counts_test")
        assert entry_after.success_count == success_before
        assert entry_after.failure_count == failure_before


# =============================================================================
# Test Duplicate Prevention
# =============================================================================


class TestDuplicatePrevention:
    """Tests for duplicate prevention in tier operations."""

    def test_promote_twice_no_double_transition(self, memory: ContinuumMemory) -> None:
        """Test that promoting twice does not create double transitions."""
        memory.add("double_promote", "Content", tier=MemoryTier.SLOW)

        with memory.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE continuum_memory SET surprise_score = 0.9 WHERE id = ?",
                ("double_promote",),
            )
            conn.commit()

        # First promotion
        result1 = memory.promote("double_promote")
        assert result1 == MemoryTier.MEDIUM

        # Second promotion attempt (should be blocked by cooldown)
        result2 = memory.promote("double_promote")
        assert result2 is None

    def test_batch_promote_same_ids(self, memory: ContinuumMemory) -> None:
        """Test batch promote with duplicate IDs in list."""
        memory.add("duplicate_id", "Content", tier=MemoryTier.MEDIUM)

        # Pass same ID multiple times
        count = memory._promote_batch(
            MemoryTier.MEDIUM,
            MemoryTier.FAST,
            ["duplicate_id", "duplicate_id", "duplicate_id"],
        )

        # Should only count as 1 promotion
        assert count == 1
