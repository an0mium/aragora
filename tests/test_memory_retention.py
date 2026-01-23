"""
Tests for ContinuumMemory retention policies.

Tests cleanup_expired_memories() and enforce_tier_limits() methods
that implement memory retention and archival.
"""

import json
import pytest
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

from aragora.memory.continuum import (
    ContinuumMemory,
    MemoryTier,
    TIER_CONFIGS,
    DEFAULT_RETENTION_MULTIPLIER,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_db(tmp_path):
    """Create a temporary database path."""
    return str(tmp_path / "test_continuum.db")


@pytest.fixture
def continuum(temp_db):
    """Create a ContinuumMemory instance with temp database."""
    return ContinuumMemory(db_path=temp_db)


@pytest.fixture
def populated_continuum(continuum):
    """Create a ContinuumMemory with test entries in each tier."""
    # Add entries to each tier
    for tier in MemoryTier:
        for i in range(5):
            continuum.add(
                id=f"{tier.value}-entry-{i}",
                content=f"Test content for {tier.value} tier entry {i}",
                tier=tier,
                importance=0.5 + (i * 0.1),
            )
    return continuum


# =============================================================================
# Schema Migration Tests
# =============================================================================


class TestSchemaV2Migration:
    """Tests for v2 schema migration."""

    def test_archive_table_created(self, continuum):
        """Archive table should be created on init."""
        with sqlite3.connect(continuum.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='continuum_memory_archive'
            """
            )
            result = cursor.fetchone()
        assert result is not None

    def test_expires_at_column_exists(self, continuum):
        """expires_at column should be added to main table."""
        with sqlite3.connect(continuum.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(continuum_memory)")
            columns = [row[1] for row in cursor.fetchall()]
        assert "expires_at" in columns


# =============================================================================
# Cleanup Expired Memories Tests
# =============================================================================


class TestCleanupExpiredMemories:
    """Tests for cleanup_expired_memories method."""

    def test_cleanup_removes_old_entries(self, continuum):
        """Old entries should be removed during cleanup."""
        # Add entry
        continuum.add("old-entry", "Old content", tier=MemoryTier.FAST)

        # Manually set updated_at to 5 hours ago (FAST tier half-life is 1h)
        old_time = (datetime.now() - timedelta(hours=5)).isoformat()
        with sqlite3.connect(continuum.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE continuum_memory SET updated_at = ? WHERE id = ?",
                (old_time, "old-entry"),
            )
            conn.commit()

        # Add recent entry
        continuum.add("new-entry", "New content", tier=MemoryTier.FAST)

        # Cleanup
        result = continuum.cleanup_expired_memories(tier=MemoryTier.FAST)

        # Verify old entry removed
        assert continuum.get("old-entry") is None
        assert continuum.get("new-entry") is not None
        assert result["deleted"] >= 1

    def test_cleanup_archives_by_default(self, continuum):
        """Entries should be archived by default."""
        continuum.add("to-archive", "Archive me", tier=MemoryTier.FAST)

        # Set old timestamp
        old_time = (datetime.now() - timedelta(hours=5)).isoformat()
        with sqlite3.connect(continuum.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE continuum_memory SET updated_at = ? WHERE id = ?",
                (old_time, "to-archive"),
            )
            conn.commit()

        # Cleanup with archive=True (default)
        result = continuum.cleanup_expired_memories(tier=MemoryTier.FAST, archive=True)

        # Check archive table
        with sqlite3.connect(continuum.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM continuum_memory_archive WHERE id = ?",
                ("to-archive",),
            )
            row = cursor.fetchone()

        assert row is not None
        assert result["archived"] >= 1

    def test_cleanup_all_tiers(self, populated_continuum):
        """Cleanup without tier should process all tiers."""
        result = populated_continuum.cleanup_expired_memories()

        # Should have by_tier keys for all tiers
        assert "by_tier" in result
        assert set(result["by_tier"].keys()) == {"fast", "medium", "slow", "glacial"}

    def test_cleanup_respects_tier_half_life(self, continuum):
        """Cleanup should use tier-specific retention times."""
        # FAST tier: 1h half-life * 2 = 2h retention
        # GLACIAL tier: 720h half-life * 2 = 1440h retention

        fast_config = TIER_CONFIGS[MemoryTier.FAST]
        glacial_config = TIER_CONFIGS[MemoryTier.GLACIAL]

        # Fast entry 3h old should be cleaned up
        continuum.add("fast-old", "Fast old", tier=MemoryTier.FAST)
        fast_time = (datetime.now() - timedelta(hours=3)).isoformat()

        # Glacial entry 3h old should NOT be cleaned up
        continuum.add("glacial-recent", "Glacial recent", tier=MemoryTier.GLACIAL)
        glacial_time = (datetime.now() - timedelta(hours=3)).isoformat()

        with sqlite3.connect(continuum.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE continuum_memory SET updated_at = ? WHERE id = ?",
                (fast_time, "fast-old"),
            )
            cursor.execute(
                "UPDATE continuum_memory SET updated_at = ? WHERE id = ?",
                (glacial_time, "glacial-recent"),
            )
            conn.commit()

        result = continuum.cleanup_expired_memories()

        assert continuum.get("fast-old") is None  # Removed
        assert continuum.get("glacial-recent") is not None  # Kept

    def test_cleanup_with_custom_max_age(self, continuum):
        """Custom max_age_hours should override tier settings."""
        continuum.add("entry", "Content", tier=MemoryTier.GLACIAL)

        # Set 2h old
        old_time = (datetime.now() - timedelta(hours=2)).isoformat()
        with sqlite3.connect(continuum.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE continuum_memory SET updated_at = ? WHERE id = ?",
                (old_time, "entry"),
            )
            conn.commit()

        # Cleanup with 1h max age should remove the 2h old entry
        result = continuum.cleanup_expired_memories(max_age_hours=1.0)

        assert continuum.get("entry") is None
        assert result["deleted"] >= 1

    def test_cleanup_no_archive(self, continuum):
        """archive=False should delete without archiving."""
        continuum.add("to-delete", "Delete me", tier=MemoryTier.FAST)

        # Set old timestamp
        old_time = (datetime.now() - timedelta(hours=5)).isoformat()
        with sqlite3.connect(continuum.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE continuum_memory SET updated_at = ? WHERE id = ?",
                (old_time, "to-delete"),
            )
            conn.commit()

        # Cleanup with archive=False
        result = continuum.cleanup_expired_memories(archive=False)

        # Entry should be deleted but not archived
        with sqlite3.connect(continuum.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM continuum_memory_archive WHERE id = ?",
                ("to-delete",),
            )
            row = cursor.fetchone()

        assert row is None
        assert result["archived"] == 0


# =============================================================================
# Enforce Tier Limits Tests
# =============================================================================


class TestEnforceTierLimits:
    """Tests for enforce_tier_limits method."""

    def test_removes_excess_entries(self, continuum):
        """Should remove entries when over limit."""
        # Set low limit for testing
        continuum.hyperparams["max_entries_per_tier"]["fast"] = 3

        # Add 5 entries
        for i in range(5):
            continuum.add(
                f"entry-{i}",
                f"Content {i}",
                tier=MemoryTier.FAST,
                importance=i * 0.2,  # 0.0, 0.2, 0.4, 0.6, 0.8
            )

        result = continuum.enforce_tier_limits(tier=MemoryTier.FAST)

        # Should have removed 2 entries (5 - 3)
        assert result["fast"] == 2

        # Verify lowest importance entries removed
        assert continuum.get("entry-0") is None  # importance 0.0
        assert continuum.get("entry-1") is None  # importance 0.2
        assert continuum.get("entry-2") is not None  # importance 0.4
        assert continuum.get("entry-3") is not None  # importance 0.6
        assert continuum.get("entry-4") is not None  # importance 0.8

    def test_keeps_highest_importance(self, continuum):
        """Should keep highest importance entries."""
        continuum.hyperparams["max_entries_per_tier"]["fast"] = 2

        # Add entries with different importance
        continuum.add("low", "Low", tier=MemoryTier.FAST, importance=0.1)
        continuum.add("high", "High", tier=MemoryTier.FAST, importance=0.9)
        continuum.add("medium", "Medium", tier=MemoryTier.FAST, importance=0.5)

        continuum.enforce_tier_limits(tier=MemoryTier.FAST)

        # Low importance should be removed
        assert continuum.get("low") is None
        # High and medium should remain
        assert continuum.get("high") is not None
        assert continuum.get("medium") is not None

    def test_no_removal_under_limit(self, continuum):
        """Should not remove entries when under limit."""
        continuum.hyperparams["max_entries_per_tier"]["fast"] = 10

        # Add only 3 entries
        for i in range(3):
            continuum.add(f"entry-{i}", f"Content {i}", tier=MemoryTier.FAST)

        result = continuum.enforce_tier_limits(tier=MemoryTier.FAST)

        assert result["fast"] == 0

    def test_archives_excess_by_default(self, continuum):
        """Excess entries should be archived by default."""
        continuum.hyperparams["max_entries_per_tier"]["fast"] = 1

        continuum.add("keep", "Keep", tier=MemoryTier.FAST, importance=0.9)
        continuum.add("archive", "Archive", tier=MemoryTier.FAST, importance=0.1)

        continuum.enforce_tier_limits(tier=MemoryTier.FAST, archive=True)

        # Check archive table
        with sqlite3.connect(continuum.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT archive_reason FROM continuum_memory_archive WHERE id = ?",
                ("archive",),
            )
            row = cursor.fetchone()

        assert row is not None
        assert row[0] == "tier_limit"


# =============================================================================
# Archive Stats Tests
# =============================================================================


class TestArchiveStats:
    """Tests for get_archive_stats method."""

    def test_empty_archive_stats(self, continuum):
        """Should return empty stats when no archived entries."""
        stats = continuum.get_archive_stats()
        assert stats["total_archived"] == 0

    def test_archive_stats_after_cleanup(self, continuum):
        """Should show correct stats after archiving."""
        # Add and archive entries
        continuum.add("entry", "Content", tier=MemoryTier.FAST)
        old_time = (datetime.now() - timedelta(hours=5)).isoformat()
        with sqlite3.connect(continuum.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE continuum_memory SET updated_at = ? WHERE id = ?",
                (old_time, "entry"),
            )
            conn.commit()

        continuum.cleanup_expired_memories(tier=MemoryTier.FAST)
        stats = continuum.get_archive_stats()

        assert stats["total_archived"] >= 1
        assert "by_tier_reason" in stats


# =============================================================================
# Handler Tests
# =============================================================================


class TestMemoryHandler:
    """Tests for memory handler API endpoints."""

    def test_handler_routes_exist(self):
        """Memory handler should have cleanup routes."""
        from aragora.server.handlers.memory import MemoryHandler

        handler = MemoryHandler({})
        assert handler.can_handle("/api/v1/memory/continuum/cleanup")
        assert handler.can_handle("/api/v1/memory/tier-stats")
        assert handler.can_handle("/api/v1/memory/archive-stats")
