"""
Comprehensive tests for Continuum Memory Statistics and Cleanup Module.

Tests the extracted statistics and cleanup functions from continuum_stats.py including:
- Memory statistics calculation (get_stats)
- Tier export functionality (export_for_tier)
- Memory pressure calculation (get_memory_pressure)
- Expired memory cleanup (cleanup_expired_memories)
- Individual memory deletion (delete_memory)
- Tier limit enforcement (enforce_tier_limits)
- Archive statistics (get_archive_stats)
"""

import sqlite3
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from aragora.memory.continuum import (
    ContinuumMemory,
    ContinuumMemoryEntry,
    reset_continuum_memory,
)
from aragora.memory.continuum_stats import (
    cleanup_expired_memories,
    delete_memory,
    enforce_tier_limits,
    export_for_tier,
    get_archive_stats,
    get_memory_pressure,
    get_stats,
)
from aragora.memory.tier_manager import (
    MemoryTier,
    TierManager,
    reset_tier_manager,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_db_path(tmp_path):
    """Create a temporary database path for testing."""
    return str(tmp_path / "test_continuum_stats.db")


@pytest.fixture
def tier_manager():
    """Create a fresh TierManager for testing."""
    return TierManager()


@pytest.fixture
def memory(temp_db_path, tier_manager):
    """Create a ContinuumMemory instance with isolated database."""
    reset_tier_manager()
    reset_continuum_memory()
    cms = ContinuumMemory(db_path=temp_db_path, tier_manager=tier_manager)
    yield cms
    reset_tier_manager()
    reset_continuum_memory()


@pytest.fixture
def populated_memory(memory):
    """Memory with pre-populated entries across all tiers."""
    # Add entries to each tier with varying importance
    memory.add("fast_1", "Fast tier entry 1", tier=MemoryTier.FAST, importance=0.8)
    memory.add("fast_2", "Fast tier entry 2", tier=MemoryTier.FAST, importance=0.6)
    memory.add("fast_3", "Fast tier entry 3", tier=MemoryTier.FAST, importance=0.4)
    memory.add("medium_1", "Medium tier entry 1", tier=MemoryTier.MEDIUM, importance=0.7)
    memory.add("medium_2", "Medium tier entry 2", tier=MemoryTier.MEDIUM, importance=0.5)
    memory.add("slow_1", "Slow tier entry 1", tier=MemoryTier.SLOW, importance=0.9)
    memory.add("glacial_1", "Glacial tier entry 1", tier=MemoryTier.GLACIAL, importance=0.95)
    return memory


@pytest.fixture
def memory_with_transitions(populated_memory):
    """Memory with tier transitions recorded."""
    # Manually insert tier transitions for testing
    with populated_memory.connection() as conn:
        cursor = conn.cursor()
        transitions = [
            ("fast_1", "medium", "fast", "high_surprise", 0.8),
            ("medium_1", "slow", "medium", "high_surprise", 0.7),
            ("slow_1", "medium", "slow", "consolidated", 0.3),
            ("fast_2", "medium", "fast", "high_surprise", 0.75),
        ]
        for memory_id, from_tier, to_tier, reason, surprise in transitions:
            cursor.execute(
                """INSERT INTO tier_transitions
                   (memory_id, from_tier, to_tier, reason, surprise_score)
                   VALUES (?, ?, ?, ?, ?)""",
                (memory_id, from_tier, to_tier, reason, surprise),
            )
        conn.commit()
    return populated_memory


# =============================================================================
# Test get_stats Function
# =============================================================================


class TestGetStats:
    """Test get_stats function for memory statistics."""

    def test_get_stats_empty_memory(self, memory):
        """Test get_stats on empty memory returns zero counts."""
        stats = get_stats(memory)

        assert stats["total_memories"] == 0
        assert stats["by_tier"] == {}
        assert stats["transitions"] == []

    def test_get_stats_populated_memory(self, populated_memory):
        """Test get_stats on populated memory returns correct counts."""
        stats = get_stats(populated_memory)

        assert stats["total_memories"] == 7
        assert "fast" in stats["by_tier"]
        assert "medium" in stats["by_tier"]
        assert "slow" in stats["by_tier"]
        assert "glacial" in stats["by_tier"]
        assert stats["by_tier"]["fast"]["count"] == 3
        assert stats["by_tier"]["medium"]["count"] == 2
        assert stats["by_tier"]["slow"]["count"] == 1
        assert stats["by_tier"]["glacial"]["count"] == 1

    def test_get_stats_average_importance(self, populated_memory):
        """Test that get_stats calculates average importance correctly."""
        stats = get_stats(populated_memory)

        # Fast tier: (0.8 + 0.6 + 0.4) / 3 = 0.6
        fast_avg = stats["by_tier"]["fast"]["avg_importance"]
        assert abs(fast_avg - 0.6) < 0.01

        # Glacial tier: 0.95
        glacial_avg = stats["by_tier"]["glacial"]["avg_importance"]
        assert abs(glacial_avg - 0.95) < 0.01

    def test_get_stats_with_transitions(self, memory_with_transitions):
        """Test get_stats includes transition history."""
        stats = get_stats(memory_with_transitions)

        assert len(stats["transitions"]) > 0
        # Should have transitions from medium to fast (2 entries)
        medium_to_fast = [
            t for t in stats["transitions"] if t["from"] == "medium" and t["to"] == "fast"
        ]
        assert len(medium_to_fast) == 1
        assert medium_to_fast[0]["count"] == 2

    def test_get_stats_surprise_and_consolidation(self, memory):
        """Test get_stats calculates average surprise and consolidation scores."""
        memory.add("test_1", "Content 1", tier=MemoryTier.FAST)
        memory.add("test_2", "Content 2", tier=MemoryTier.FAST)

        # Set specific scores
        with memory.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE continuum_memory SET surprise_score = 0.6, consolidation_score = 0.4 WHERE id = ?",
                ("test_1",),
            )
            cursor.execute(
                "UPDATE continuum_memory SET surprise_score = 0.4, consolidation_score = 0.6 WHERE id = ?",
                ("test_2",),
            )
            conn.commit()

        stats = get_stats(memory)

        # Averages should be 0.5 for both
        assert abs(stats["by_tier"]["fast"]["avg_surprise"] - 0.5) < 0.01
        assert abs(stats["by_tier"]["fast"]["avg_consolidation"] - 0.5) < 0.01


# =============================================================================
# Test export_for_tier Function
# =============================================================================


class TestExportForTier:
    """Test export_for_tier function for tier-specific exports."""

    def test_export_empty_tier(self, memory):
        """Test exporting from empty tier returns empty list."""
        exported = export_for_tier(memory, MemoryTier.FAST)

        assert exported == []

    def test_export_fast_tier(self, populated_memory):
        """Test exporting fast tier entries."""
        exported = export_for_tier(populated_memory, MemoryTier.FAST)

        assert len(exported) == 3
        assert all("id" in e for e in exported)
        assert all("content" in e for e in exported)
        assert all("importance" in e for e in exported)

    def test_export_includes_all_fields(self, populated_memory):
        """Test export includes all required fields."""
        exported = export_for_tier(populated_memory, MemoryTier.GLACIAL)

        assert len(exported) == 1
        entry = exported[0]
        assert entry["id"] == "glacial_1"
        assert entry["content"] == "Glacial tier entry 1"
        assert entry["importance"] == 0.95
        assert "surprise_score" in entry
        assert "consolidation_score" in entry
        assert "success_rate" in entry
        assert "update_count" in entry

    def test_export_multiple_tiers_separately(self, populated_memory):
        """Test exporting multiple tiers gives different results."""
        fast_export = export_for_tier(populated_memory, MemoryTier.FAST)
        medium_export = export_for_tier(populated_memory, MemoryTier.MEDIUM)

        assert len(fast_export) == 3
        assert len(medium_export) == 2
        # IDs should not overlap
        fast_ids = {e["id"] for e in fast_export}
        medium_ids = {e["id"] for e in medium_export}
        assert fast_ids.isdisjoint(medium_ids)


# =============================================================================
# Test get_memory_pressure Function
# =============================================================================


class TestGetMemoryPressure:
    """Test get_memory_pressure function for utilization calculation."""

    def test_pressure_empty_memory(self, memory):
        """Test pressure is 0 for empty memory."""
        pressure = get_memory_pressure(memory)

        assert pressure == 0.0

    def test_pressure_with_entries(self, memory):
        """Test pressure calculation with entries."""
        # Add entries to fast tier
        for i in range(50):
            memory.add(f"pressure_{i}", f"Content {i}", tier=MemoryTier.FAST)

        pressure = get_memory_pressure(memory)

        # Default fast limit is 1000, so 50/1000 = 0.05
        assert 0.04 <= pressure <= 0.06

    def test_pressure_at_limit(self, memory):
        """Test pressure equals 1.0 when at tier limit."""
        # Set a very low limit
        memory.hyperparams["max_entries_per_tier"]["fast"] = 10

        for i in range(10):
            memory.add(f"limit_{i}", f"Content {i}", tier=MemoryTier.FAST)

        pressure = get_memory_pressure(memory)

        assert pressure == 1.0

    def test_pressure_over_limit(self, memory):
        """Test pressure capped at 1.0 when over limit."""
        memory.hyperparams["max_entries_per_tier"]["fast"] = 5

        for i in range(20):
            memory.add(f"over_{i}", f"Content {i}", tier=MemoryTier.FAST)

        pressure = get_memory_pressure(memory)

        # Capped at 1.0
        assert pressure == 1.0

    def test_pressure_highest_tier(self, memory):
        """Test pressure returns highest tier utilization."""
        memory.hyperparams["max_entries_per_tier"]["fast"] = 100
        memory.hyperparams["max_entries_per_tier"]["medium"] = 10

        # Add 50 to fast (50% utilization)
        for i in range(50):
            memory.add(f"fast_{i}", f"Content {i}", tier=MemoryTier.FAST)

        # Add 8 to medium (80% utilization)
        for i in range(8):
            memory.add(f"medium_{i}", f"Content {i}", tier=MemoryTier.MEDIUM)

        pressure = get_memory_pressure(memory)

        # Should return medium's 80% as it's higher
        assert 0.75 <= pressure <= 0.85

    def test_pressure_with_zero_limit(self, memory):
        """Test pressure handles zero limit gracefully."""
        memory.hyperparams["max_entries_per_tier"]["fast"] = 0

        memory.add("test", "Content", tier=MemoryTier.FAST)

        pressure = get_memory_pressure(memory)

        # Zero limit should be skipped, not cause division by zero
        assert 0.0 <= pressure <= 1.0

    def test_pressure_with_no_max_entries(self, memory):
        """Test pressure returns 0 when max_entries is None."""
        memory.hyperparams["max_entries_per_tier"] = None

        pressure = get_memory_pressure(memory)

        assert pressure == 0.0


# =============================================================================
# Test cleanup_expired_memories Function
# =============================================================================


class TestCleanupExpiredMemories:
    """Test cleanup_expired_memories function for retention management."""

    def test_cleanup_no_expired(self, populated_memory):
        """Test cleanup with no expired memories."""
        # All entries are new, none should expire
        result = cleanup_expired_memories(populated_memory)

        assert result["deleted"] == 0
        assert result["archived"] == 0

    def test_cleanup_expired_entries(self, memory):
        """Test cleanup removes expired entries."""
        # Insert old entries directly
        old_time = (datetime.now() - timedelta(days=30)).isoformat()
        with memory.connection() as conn:
            cursor = conn.cursor()
            for i in range(5):
                cursor.execute(
                    """INSERT INTO continuum_memory
                       (id, tier, content, importance, updated_at)
                       VALUES (?, ?, ?, ?, ?)""",
                    (f"old_{i}", "fast", f"Old content {i}", 0.3, old_time),
                )
            conn.commit()

        result = cleanup_expired_memories(memory, max_age_hours=1)

        # Should have deleted or archived all 5
        assert result["deleted"] >= 5 or result["archived"] >= 5

    def test_cleanup_specific_tier(self, memory):
        """Test cleanup only affects specified tier."""
        old_time = (datetime.now() - timedelta(days=30)).isoformat()
        with memory.connection() as conn:
            cursor = conn.cursor()
            # Add old entries to both fast and medium
            cursor.execute(
                """INSERT INTO continuum_memory
                   (id, tier, content, importance, updated_at)
                   VALUES (?, ?, ?, ?, ?)""",
                ("old_fast", "fast", "Old fast content", 0.3, old_time),
            )
            cursor.execute(
                """INSERT INTO continuum_memory
                   (id, tier, content, importance, updated_at)
                   VALUES (?, ?, ?, ?, ?)""",
                ("old_medium", "medium", "Old medium content", 0.3, old_time),
            )
            conn.commit()

        result = cleanup_expired_memories(memory, tier=MemoryTier.FAST, max_age_hours=1)

        # Should only cleanup fast tier
        assert "fast" in result["by_tier"]
        assert result["by_tier"]["fast"]["deleted"] >= 1

        # Medium entry should still exist
        with memory.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM continuum_memory WHERE id = ?", ("old_medium",))
            assert cursor.fetchone() is not None

    def test_cleanup_with_archive(self, memory):
        """Test cleanup archives entries when archive=True."""
        old_time = (datetime.now() - timedelta(days=30)).isoformat()
        with memory.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO continuum_memory
                   (id, tier, content, importance, updated_at)
                   VALUES (?, ?, ?, ?, ?)""",
                ("archive_me", "fast", "Content to archive", 0.3, old_time),
            )
            conn.commit()

        result = cleanup_expired_memories(memory, archive=True, max_age_hours=1)

        assert result["archived"] >= 1

        # Check archive table
        with memory.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, archive_reason FROM continuum_memory_archive WHERE id = ?",
                ("archive_me",),
            )
            row = cursor.fetchone()
            assert row is not None
            assert row[1] == "expired"

    def test_cleanup_without_archive(self, memory):
        """Test cleanup skips archiving when archive=False."""
        old_time = (datetime.now() - timedelta(days=30)).isoformat()
        with memory.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO continuum_memory
                   (id, tier, content, importance, updated_at)
                   VALUES (?, ?, ?, ?, ?)""",
                ("delete_me", "fast", "Content to delete", 0.3, old_time),
            )
            conn.commit()

        result = cleanup_expired_memories(memory, archive=False, max_age_hours=1)

        assert result["deleted"] >= 1
        assert result["archived"] == 0

    def test_cleanup_skips_red_line(self, memory):
        """Test cleanup skips red-lined entries."""
        old_time = (datetime.now() - timedelta(days=30)).isoformat()
        with memory.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO continuum_memory
                   (id, tier, content, importance, updated_at, red_line, red_line_reason)
                   VALUES (?, ?, ?, ?, ?, 1, ?)""",
                ("protected", "fast", "Protected content", 0.3, old_time, "Safety critical"),
            )
            conn.commit()

        result = cleanup_expired_memories(memory, max_age_hours=1)

        # Red-lined entry should not be deleted
        with memory.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM continuum_memory WHERE id = ?", ("protected",))
            assert cursor.fetchone() is not None

    def test_cleanup_returns_by_tier_breakdown(self, memory):
        """Test cleanup returns breakdown by tier."""
        result = cleanup_expired_memories(memory)

        # Should have entries for all tiers
        assert "by_tier" in result
        assert "fast" in result["by_tier"]
        assert "medium" in result["by_tier"]
        assert "slow" in result["by_tier"]
        assert "glacial" in result["by_tier"]

        # Each tier should have cutoff_hours
        for tier_name, tier_data in result["by_tier"].items():
            assert "cutoff_hours" in tier_data


# =============================================================================
# Test delete_memory Function
# =============================================================================


class TestDeleteMemory:
    """Test delete_memory function for individual entry deletion."""

    def test_delete_existing_memory(self, memory):
        """Test deleting an existing memory entry."""
        memory.add("to_delete", "Content to delete")

        result = delete_memory(memory, "to_delete")

        assert result["deleted"] is True
        assert result["id"] == "to_delete"
        assert memory.get("to_delete") is None

    def test_delete_nonexistent_memory(self, memory):
        """Test deleting non-existent memory returns not deleted."""
        result = delete_memory(memory, "nonexistent")

        assert result["deleted"] is False
        assert result["archived"] is False

    def test_delete_with_archive(self, memory):
        """Test deleting with archive=True archives the entry."""
        memory.add("archive_delete", "Content to archive")

        result = delete_memory(memory, "archive_delete", archive=True)

        assert result["deleted"] is True
        assert result["archived"] is True

        # Check archive
        with memory.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, archive_reason FROM continuum_memory_archive WHERE id = ?",
                ("archive_delete",),
            )
            row = cursor.fetchone()
            assert row is not None

    def test_delete_without_archive(self, memory):
        """Test deleting with archive=False skips archiving."""
        memory.add("no_archive", "Content")

        result = delete_memory(memory, "no_archive", archive=False)

        assert result["deleted"] is True
        assert result["archived"] is False

    def test_delete_with_custom_reason(self, memory):
        """Test deleting with custom archive reason."""
        memory.add("custom_reason", "Content")

        result = delete_memory(memory, "custom_reason", reason="admin_cleanup")

        with memory.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT archive_reason FROM continuum_memory_archive WHERE id = ?",
                ("custom_reason",),
            )
            row = cursor.fetchone()
            assert row[0] == "admin_cleanup"

    def test_delete_blocks_red_line(self, memory):
        """Test deleting red-lined memory is blocked."""
        memory.add("protected", "Protected content")
        memory.mark_red_line("protected", reason="Critical")

        result = delete_memory(memory, "protected")

        assert result["deleted"] is False
        assert result["blocked"] is True
        assert result["red_line_reason"] == "Critical"
        assert memory.get("protected") is not None

    def test_delete_force_red_line(self, memory):
        """Test force deleting red-lined memory."""
        memory.add("force_delete", "Content")
        memory.mark_red_line("force_delete", reason="Test")

        result = delete_memory(memory, "force_delete", force=True)

        assert result["deleted"] is True
        assert result["blocked"] is False


# =============================================================================
# Test enforce_tier_limits Function
# =============================================================================


class TestEnforceTierLimits:
    """Test enforce_tier_limits function for tier capacity management."""

    def test_enforce_no_excess(self, memory):
        """Test enforce with no excess entries."""
        memory.hyperparams["max_entries_per_tier"]["fast"] = 100

        for i in range(10):
            memory.add(f"within_limit_{i}", f"Content {i}", tier=MemoryTier.FAST)

        result = enforce_tier_limits(memory)

        assert result["fast"] == 0

    def test_enforce_removes_excess(self, memory):
        """Test enforce removes excess entries."""
        memory.hyperparams["max_entries_per_tier"]["fast"] = 10

        for i in range(20):
            memory.add(f"excess_{i}", f"Content {i}", tier=MemoryTier.FAST, importance=i / 20)

        result = enforce_tier_limits(memory, tier=MemoryTier.FAST)

        # Should have removed 10 excess entries
        assert result["fast"] == 10

        # Check remaining count
        stats = get_stats(memory)
        assert stats["by_tier"]["fast"]["count"] == 10

    def test_enforce_removes_lowest_importance(self, memory):
        """Test enforce removes lowest importance entries first."""
        memory.hyperparams["max_entries_per_tier"]["fast"] = 3

        memory.add("high", "High importance", tier=MemoryTier.FAST, importance=0.9)
        memory.add("medium", "Medium importance", tier=MemoryTier.FAST, importance=0.5)
        memory.add("low", "Low importance", tier=MemoryTier.FAST, importance=0.1)
        memory.add("very_low", "Very low importance", tier=MemoryTier.FAST, importance=0.05)
        memory.add("lowest", "Lowest importance", tier=MemoryTier.FAST, importance=0.01)

        enforce_tier_limits(memory, tier=MemoryTier.FAST)

        # Should keep high, medium, low (highest 3)
        assert memory.get("high") is not None
        assert memory.get("medium") is not None
        assert memory.get("low") is not None
        assert memory.get("very_low") is None
        assert memory.get("lowest") is None

    def test_enforce_specific_tier(self, memory):
        """Test enforce only affects specified tier."""
        memory.hyperparams["max_entries_per_tier"]["fast"] = 5
        memory.hyperparams["max_entries_per_tier"]["medium"] = 5

        for i in range(10):
            memory.add(f"fast_{i}", f"Content {i}", tier=MemoryTier.FAST)
        for i in range(10):
            memory.add(f"medium_{i}", f"Content {i}", tier=MemoryTier.MEDIUM)

        result = enforce_tier_limits(memory, tier=MemoryTier.FAST)

        # Only fast should be enforced
        assert result["fast"] == 5

        # Medium should still have 10
        stats = get_stats(memory)
        assert stats["by_tier"]["medium"]["count"] == 10

    def test_enforce_all_tiers(self, memory):
        """Test enforce on all tiers at once."""
        memory.hyperparams["max_entries_per_tier"]["fast"] = 3
        memory.hyperparams["max_entries_per_tier"]["medium"] = 2

        for i in range(5):
            memory.add(f"fast_{i}", f"Content {i}", tier=MemoryTier.FAST)
        for i in range(4):
            memory.add(f"medium_{i}", f"Content {i}", tier=MemoryTier.MEDIUM)

        result = enforce_tier_limits(memory)

        assert result["fast"] == 2  # 5 - 3 = 2 removed
        assert result["medium"] == 2  # 4 - 2 = 2 removed

    def test_enforce_with_archive(self, memory):
        """Test enforce archives excess entries when archive=True."""
        memory.hyperparams["max_entries_per_tier"]["fast"] = 1

        memory.add("keep", "Keep this", tier=MemoryTier.FAST, importance=0.9)
        memory.add("archive", "Archive this", tier=MemoryTier.FAST, importance=0.1)

        enforce_tier_limits(memory, tier=MemoryTier.FAST, archive=True)

        # Check archive
        with memory.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT archive_reason FROM continuum_memory_archive WHERE id = ?",
                ("archive",),
            )
            row = cursor.fetchone()
            assert row is not None
            assert row[0] == "tier_limit"

    def test_enforce_skips_red_line(self, memory):
        """Test enforce skips red-lined entries even if lowest importance."""
        memory.hyperparams["max_entries_per_tier"]["fast"] = 2

        memory.add("normal", "Normal content", tier=MemoryTier.FAST, importance=0.5)
        memory.add("protected", "Protected content", tier=MemoryTier.FAST, importance=0.01)
        memory.add("another", "Another content", tier=MemoryTier.FAST, importance=0.3)

        memory.mark_red_line("protected", reason="Critical")

        enforce_tier_limits(memory, tier=MemoryTier.FAST)

        # Protected should still exist despite lowest importance
        assert memory.get("protected") is not None


# =============================================================================
# Test get_archive_stats Function
# =============================================================================


class TestGetArchiveStats:
    """Test get_archive_stats function for archive statistics."""

    def test_archive_stats_empty(self, memory):
        """Test archive stats on empty archive."""
        stats = get_archive_stats(memory)

        assert stats["total_archived"] == 0
        assert stats["by_tier_reason"] == {}

    def test_archive_stats_with_entries(self, memory):
        """Test archive stats with archived entries."""
        memory.add("entry_1", "Content 1")
        memory.add("entry_2", "Content 2")
        memory.add("entry_3", "Content 3")

        # Delete with different reasons
        delete_memory(memory, "entry_1", reason="user_deleted")
        delete_memory(memory, "entry_2", reason="admin_cleanup")
        delete_memory(memory, "entry_3", reason="user_deleted")

        stats = get_archive_stats(memory)

        assert stats["total_archived"] == 3

    def test_archive_stats_by_tier_reason(self, memory):
        """Test archive stats breakdown by tier and reason."""
        memory.add("fast_1", "Content", tier=MemoryTier.FAST)
        memory.add("fast_2", "Content", tier=MemoryTier.FAST)
        memory.add("medium_1", "Content", tier=MemoryTier.MEDIUM)

        delete_memory(memory, "fast_1", reason="expired")
        delete_memory(memory, "fast_2", reason="tier_limit")
        delete_memory(memory, "medium_1", reason="expired")

        stats = get_archive_stats(memory)

        assert "fast" in stats["by_tier_reason"]
        assert stats["by_tier_reason"]["fast"]["expired"] == 1
        assert stats["by_tier_reason"]["fast"]["tier_limit"] == 1
        assert stats["by_tier_reason"]["medium"]["expired"] == 1

    def test_archive_stats_timestamps(self, memory):
        """Test archive stats includes timestamp range."""
        memory.add("entry_1", "Content 1")
        memory.add("entry_2", "Content 2")

        delete_memory(memory, "entry_1")
        delete_memory(memory, "entry_2")

        stats = get_archive_stats(memory)

        assert stats["oldest_archived"] is not None
        assert stats["newest_archived"] is not None


# =============================================================================
# Test Edge Cases and Boundary Conditions
# =============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_stats_single_entry(self, memory):
        """Test stats with single entry."""
        memory.add("single", "Single entry", tier=MemoryTier.SLOW, importance=0.7)

        stats = get_stats(memory)

        assert stats["total_memories"] == 1
        assert stats["by_tier"]["slow"]["count"] == 1
        assert stats["by_tier"]["slow"]["avg_importance"] == 0.7

    def test_pressure_with_partial_limits(self, memory):
        """Test pressure with limits only on some tiers."""
        # Only set limit for fast tier
        memory.hyperparams["max_entries_per_tier"] = {"fast": 10}

        for i in range(8):
            memory.add(f"fast_{i}", f"Content {i}", tier=MemoryTier.FAST)

        pressure = get_memory_pressure(memory)

        assert pressure == 0.8

    def test_enforce_default_limit(self, memory):
        """Test enforce uses default limit when tier not in config."""
        # Remove fast from max_entries
        original = memory.hyperparams["max_entries_per_tier"].copy()
        memory.hyperparams["max_entries_per_tier"] = {"medium": 5}

        for i in range(5):
            memory.add(f"fast_{i}", f"Content {i}", tier=MemoryTier.FAST)

        result = enforce_tier_limits(memory, tier=MemoryTier.FAST)

        # Should use default of 10000
        assert result["fast"] == 0

        memory.hyperparams["max_entries_per_tier"] = original

    def test_cleanup_all_tiers_uses_retention_multiplier(self, memory):
        """Test cleanup uses retention_multiplier from hyperparams."""
        memory.hyperparams["retention_multiplier"] = 1.5

        result = cleanup_expired_memories(memory)

        # Check cutoff hours match expected
        # Fast tier has 1 hour half-life, so cutoff = 1 * 1.5 = 1.5 hours
        assert result["by_tier"]["fast"]["cutoff_hours"] == 1.5

    def test_delete_memory_logs_correctly(self, memory, caplog):
        """Test delete_memory logs appropriate messages."""
        import logging

        memory.add("log_test", "Content")

        with caplog.at_level(logging.INFO, logger="aragora.memory.continuum_stats"):
            delete_memory(memory, "log_test", reason="test_reason")

        assert "log_test" in caplog.text
        assert "test_reason" in caplog.text


# =============================================================================
# Test Integration with ContinuumMemory Methods
# =============================================================================


class TestContinuumMemoryIntegration:
    """Test that stats functions work through ContinuumMemory delegation."""

    def test_cms_get_stats_delegates(self, populated_memory):
        """Test ContinuumMemory.get_stats() delegates to get_stats."""
        stats = populated_memory.get_stats()

        assert stats["total_memories"] == 7
        assert "by_tier" in stats

    def test_cms_export_for_tier_delegates(self, populated_memory):
        """Test ContinuumMemory.export_for_tier() delegates correctly."""
        exported = populated_memory.export_for_tier(MemoryTier.FAST)

        assert len(exported) == 3

    def test_cms_get_memory_pressure_delegates(self, memory):
        """Test ContinuumMemory.get_memory_pressure() delegates correctly."""
        for i in range(50):
            memory.add(f"test_{i}", f"Content {i}", tier=MemoryTier.FAST)

        pressure = memory.get_memory_pressure()

        assert 0.04 <= pressure <= 0.06

    def test_cms_cleanup_expired_delegates(self, memory):
        """Test ContinuumMemory.cleanup_expired_memories() delegates correctly."""
        result = memory.cleanup_expired_memories()

        assert "deleted" in result
        assert "archived" in result
        assert "by_tier" in result

    def test_cms_enforce_tier_limits_delegates(self, memory):
        """Test ContinuumMemory.enforce_tier_limits() delegates correctly."""
        result = memory.enforce_tier_limits()

        assert isinstance(result, dict)

    def test_cms_get_archive_stats_delegates(self, memory):
        """Test ContinuumMemory.get_archive_stats() delegates correctly."""
        stats = memory.get_archive_stats()

        assert "total_archived" in stats
        assert "by_tier_reason" in stats
