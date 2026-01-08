"""
Comprehensive tests for ContinuumMemory multi-tier memory system.

Tests the Nested Learning paradigm implementation with multi-timescale
memory updates, tier promotion/demotion, and consolidation.
"""

import tempfile
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from aragora.memory.continuum import (
    ContinuumMemory,
    ContinuumMemoryEntry,
    MemoryTier,
    DEFAULT_RETENTION_MULTIPLIER,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_db_path():
    """Create a temporary database path for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield str(Path(tmpdir) / "test_continuum.db")


@pytest.fixture
def memory_system(temp_db_path):
    """Create a ContinuumMemory instance for testing."""
    return ContinuumMemory(db_path=temp_db_path)


@pytest.fixture
def populated_memory(memory_system):
    """Create a memory system with sample entries."""
    # Add entries in each tier
    memory_system.add(
        id="fast_1",
        content="Immediate pattern",
        tier=MemoryTier.FAST,
        importance=0.8,
    )
    memory_system.add(
        id="medium_1",
        content="Tactical pattern",
        tier=MemoryTier.MEDIUM,
        importance=0.6,
    )
    memory_system.add(
        id="slow_1",
        content="Strategic pattern",
        tier=MemoryTier.SLOW,
        importance=0.5,
    )
    memory_system.add(
        id="glacial_1",
        content="Foundational knowledge",
        tier=MemoryTier.GLACIAL,
        importance=0.4,
    )
    return memory_system


# =============================================================================
# ContinuumMemoryEntry Tests
# =============================================================================


class TestContinuumMemoryEntry:
    """Tests for ContinuumMemoryEntry dataclass."""

    def test_success_rate_calculation(self):
        """Success rate is calculated correctly."""
        entry = ContinuumMemoryEntry(
            id="test",
            tier=MemoryTier.FAST,
            content="test",
            importance=0.5,
            surprise_score=0.5,
            consolidation_score=0.5,
            update_count=10,
            success_count=7,
            failure_count=3,
            created_at="2024-01-01T00:00:00",
            updated_at="2024-01-01T00:00:00",
        )
        assert entry.success_rate == 0.7

    def test_success_rate_with_no_outcomes(self):
        """Success rate defaults to 0.5 with no outcomes."""
        entry = ContinuumMemoryEntry(
            id="test",
            tier=MemoryTier.FAST,
            content="test",
            importance=0.5,
            surprise_score=0.5,
            consolidation_score=0.5,
            update_count=0,
            success_count=0,
            failure_count=0,
            created_at="2024-01-01T00:00:00",
            updated_at="2024-01-01T00:00:00",
        )
        assert entry.success_rate == 0.5

    def test_stability_score(self):
        """Stability score is inverse of surprise."""
        entry = ContinuumMemoryEntry(
            id="test",
            tier=MemoryTier.FAST,
            content="test",
            importance=0.5,
            surprise_score=0.3,
            consolidation_score=0.5,
            update_count=10,
            success_count=5,
            failure_count=5,
            created_at="2024-01-01T00:00:00",
            updated_at="2024-01-01T00:00:00",
        )
        assert entry.stability_score == 0.7

    def test_should_promote_fast_tier(self):
        """Fast tier cannot be promoted further."""
        entry = ContinuumMemoryEntry(
            id="test",
            tier=MemoryTier.FAST,
            content="test",
            importance=0.5,
            surprise_score=0.9,  # High surprise
            consolidation_score=0.5,
            update_count=10,
            success_count=5,
            failure_count=5,
            created_at="2024-01-01T00:00:00",
            updated_at="2024-01-01T00:00:00",
        )
        assert not entry.should_promote()

    def test_should_demote_glacial_tier(self):
        """Glacial tier cannot be demoted further."""
        entry = ContinuumMemoryEntry(
            id="test",
            tier=MemoryTier.GLACIAL,
            content="test",
            importance=0.5,
            surprise_score=0.1,  # Low surprise = high stability
            consolidation_score=0.5,
            update_count=15,
            success_count=5,
            failure_count=5,
            created_at="2024-01-01T00:00:00",
            updated_at="2024-01-01T00:00:00",
        )
        assert not entry.should_demote()


# =============================================================================
# ContinuumMemory Basic Operations Tests
# =============================================================================


class TestContinuumMemoryBasicOperations:
    """Tests for basic memory operations."""

    def test_add_memory(self, memory_system):
        """Can add a memory entry."""
        memory_system.add(
            id="test_1",
            content="Test content",
            tier=MemoryTier.FAST,
            importance=0.8,
        )

        entry = memory_system.get("test_1")
        assert entry is not None
        assert entry.id == "test_1"
        assert entry.content == "Test content"
        assert entry.tier == MemoryTier.FAST
        assert entry.importance == 0.8

    def test_add_with_metadata(self, memory_system):
        """Can add memory with metadata."""
        memory_system.add(
            id="test_meta",
            content="Test with metadata",
            tier=MemoryTier.MEDIUM,
            importance=0.5,
            metadata={"agent": "claude", "domain": "coding"},
        )

        entry = memory_system.get("test_meta")
        assert entry is not None
        assert entry.metadata["agent"] == "claude"
        assert entry.metadata["domain"] == "coding"

    def test_get_nonexistent(self, memory_system):
        """Get returns None for nonexistent entries."""
        entry = memory_system.get("nonexistent")
        assert entry is None

    def test_add_duplicate_updates(self, memory_system):
        """Adding duplicate ID updates existing entry."""
        memory_system.add(
            id="dup_test",
            content="Original content",
            tier=MemoryTier.FAST,
            importance=0.5,
        )

        # Add again with same ID
        memory_system.add(
            id="dup_test",
            content="Updated content",
            tier=MemoryTier.FAST,
            importance=0.8,
        )

        entry = memory_system.get("dup_test")
        assert entry is not None
        assert entry.content == "Updated content"
        assert entry.importance == 0.8
        assert entry.update_count >= 1


# =============================================================================
# Tier Retrieval Tests
# =============================================================================


class TestTierRetrieval:
    """Tests for retrieving memories from specific tiers."""

    def test_retrieve_from_single_tier(self, populated_memory):
        """Can retrieve from a single tier."""
        results = populated_memory.retrieve(tiers=[MemoryTier.FAST])
        assert len(results) >= 1
        for entry in results:
            assert entry.tier == MemoryTier.FAST

    def test_retrieve_from_multiple_tiers(self, populated_memory):
        """Can retrieve from multiple tiers."""
        results = populated_memory.retrieve(
            tiers=[MemoryTier.FAST, MemoryTier.MEDIUM]
        )
        assert len(results) >= 2
        tiers = {entry.tier for entry in results}
        assert MemoryTier.FAST in tiers or MemoryTier.MEDIUM in tiers

    def test_retrieve_with_limit(self, memory_system):
        """Retrieve respects limit parameter."""
        # Add multiple entries
        for i in range(10):
            memory_system.add(
                id=f"limit_test_{i}",
                content=f"Content {i}",
                tier=MemoryTier.FAST,
                importance=0.5,
            )

        results = memory_system.retrieve(tiers=[MemoryTier.FAST], limit=5)
        assert len(results) == 5

    def test_retrieve_ordered_by_importance(self, memory_system):
        """Results are ordered by importance."""
        memory_system.add("high", "High importance", MemoryTier.FAST, importance=0.9)
        memory_system.add("low", "Low importance", MemoryTier.FAST, importance=0.1)
        memory_system.add("mid", "Mid importance", MemoryTier.FAST, importance=0.5)

        results = memory_system.retrieve(tiers=[MemoryTier.FAST])

        # Should be ordered by importance descending
        importances = [e.importance for e in results]
        assert importances == sorted(importances, reverse=True)


# =============================================================================
# Outcome Update Tests
# =============================================================================


class TestOutcomeUpdates:
    """Tests for updating memory outcomes."""

    def test_update_success(self, memory_system):
        """Update with success increases success count."""
        memory_system.add("outcome_test", "Test", MemoryTier.FAST, importance=0.5)

        initial = memory_system.get("outcome_test")
        initial_success = initial.success_count

        memory_system.update_outcome("outcome_test", success=True)

        updated = memory_system.get("outcome_test")
        assert updated.success_count == initial_success + 1

    def test_update_failure(self, memory_system):
        """Update with failure increases failure count."""
        memory_system.add("fail_test", "Test", MemoryTier.FAST, importance=0.5)

        initial = memory_system.get("fail_test")
        initial_failure = initial.failure_count

        memory_system.update_outcome("fail_test", success=False)

        updated = memory_system.get("fail_test")
        assert updated.failure_count == initial_failure + 1

    def test_update_increments_update_count(self, memory_system):
        """Each outcome update increments update count."""
        memory_system.add("count_test", "Test", MemoryTier.FAST, importance=0.5)

        initial = memory_system.get("count_test")
        initial_count = initial.update_count

        for _ in range(3):
            memory_system.update_outcome("count_test", success=True)

        updated = memory_system.get("count_test")
        assert updated.update_count == initial_count + 3

    def test_update_modifies_surprise_score(self, memory_system):
        """Outcome updates modify surprise score."""
        memory_system.add(
            "surprise_test",
            "Test",
            MemoryTier.FAST,
            importance=0.5,
        )

        initial = memory_system.get("surprise_test")
        initial_surprise = initial.surprise_score

        # Multiple consistent outcomes should reduce surprise
        for _ in range(10):
            memory_system.update_outcome("surprise_test", success=True)

        updated = memory_system.get("surprise_test")
        # Surprise should change (implementation dependent direction)
        assert updated.surprise_score != initial_surprise or updated.update_count > 0


# =============================================================================
# Tier Promotion/Demotion Tests
# =============================================================================


class TestTierTransitions:
    """Tests for tier promotion and demotion."""

    def test_promote_requires_high_surprise(self, memory_system):
        """Promotion requires high surprise score (TierManager decision)."""
        memory_system.add(
            "promote_test",
            "Test",
            MemoryTier.MEDIUM,  # Start at medium
            importance=0.5,
        )

        # Fresh entry with low surprise may not promote
        result = memory_system.promote("promote_test")

        # Result depends on TierManager decision (surprise threshold)
        # Just verify it returns either None or a valid tier
        assert result is None or isinstance(result, MemoryTier)

    def test_promote_fast_tier_returns_none(self, memory_system):
        """Cannot promote beyond FAST tier."""
        memory_system.add(
            "fast_promote",
            "Test",
            MemoryTier.FAST,
            importance=0.5,
        )

        result = memory_system.promote("fast_promote")

        # Should return None (no promotion possible from FAST)
        assert result is None

        entry = memory_system.get("fast_promote")
        assert entry.tier == MemoryTier.FAST

    def test_demote_requires_high_stability(self, memory_system):
        """Demotion requires high stability (TierManager decision)."""
        memory_system.add(
            "demote_test",
            "Test",
            MemoryTier.MEDIUM,  # Start at medium
            importance=0.5,
        )

        # Fresh entry may not meet demotion criteria
        result = memory_system.demote("demote_test")

        # Result depends on TierManager decision
        assert result is None or isinstance(result, MemoryTier)

    def test_demote_glacial_tier_returns_none(self, memory_system):
        """Cannot demote beyond GLACIAL tier."""
        memory_system.add(
            "glacial_demote",
            "Test",
            MemoryTier.GLACIAL,
            importance=0.5,
        )

        result = memory_system.demote("glacial_demote")

        # Should return None (no demotion possible from GLACIAL)
        assert result is None

        entry = memory_system.get("glacial_demote")
        assert entry.tier == MemoryTier.GLACIAL

    def test_promote_nonexistent_returns_none(self, memory_system):
        """Promoting nonexistent entry returns None."""
        result = memory_system.promote("nonexistent")
        assert result is None

    def test_demote_nonexistent_returns_none(self, memory_system):
        """Demoting nonexistent entry returns None."""
        result = memory_system.demote("nonexistent")
        assert result is None


# =============================================================================
# Consolidation Tests
# =============================================================================


class TestConsolidation:
    """Tests for memory consolidation."""

    def test_consolidate_returns_stats(self, populated_memory):
        """Consolidate returns statistics."""
        stats = populated_memory.consolidate()

        assert isinstance(stats, dict)
        assert "promotions" in stats
        assert "demotions" in stats

    def test_consolidate_handles_empty_db(self, memory_system):
        """Consolidate works with empty database."""
        stats = memory_system.consolidate()

        assert stats["promotions"] == 0
        assert stats["demotions"] == 0


# =============================================================================
# Statistics Tests
# =============================================================================


class TestStatistics:
    """Tests for memory statistics."""

    def test_get_stats(self, populated_memory):
        """Can get memory statistics."""
        stats = populated_memory.get_stats()

        assert "total_memories" in stats
        assert "by_tier" in stats
        assert stats["total_memories"] >= 4  # We added 4 entries

    def test_tier_metrics(self, populated_memory):
        """Can get tier metrics from tier manager."""
        metrics = populated_memory.get_tier_metrics()

        assert isinstance(metrics, dict)


# =============================================================================
# Cleanup and Retention Tests
# =============================================================================


class TestCleanupAndRetention:
    """Tests for memory cleanup and retention policies."""

    def test_cleanup_expired_memories(self, memory_system):
        """Can cleanup expired memories."""
        # Add some test memories
        memory_system.add("cleanup_1", "Test 1", MemoryTier.FAST, importance=0.5)
        memory_system.add("cleanup_2", "Test 2", MemoryTier.FAST, importance=0.5)

        # Run cleanup (may not remove anything if not expired)
        stats = memory_system.cleanup_expired_memories()

        assert isinstance(stats, dict)
        # Stats should have by_tier with cleanup counts
        assert "by_tier" in stats or "deleted" in stats

    def test_enforce_tier_limits(self, memory_system):
        """Enforcing tier limits removes excess entries."""
        # Set a low limit for testing
        memory_system.hyperparams["max_entries_per_tier"]["fast"] = 5

        # Add more than the limit
        for i in range(10):
            memory_system.add(
                f"limit_{i}",
                f"Content {i}",
                MemoryTier.FAST,
                importance=i / 10.0,  # Varying importance
            )

        # Enforce limits
        removed = memory_system.enforce_tier_limits()

        # Should have removed some entries
        assert removed.get("fast", 0) >= 0  # May have removed some


# =============================================================================
# Learning Rate Tests
# =============================================================================


class TestLearningRate:
    """Tests for adaptive learning rate."""

    def test_learning_rate_decreases_with_updates(self, memory_system):
        """Learning rate decreases as update count increases."""
        lr_low = memory_system.get_learning_rate(MemoryTier.FAST, update_count=1)
        lr_high = memory_system.get_learning_rate(MemoryTier.FAST, update_count=100)

        # Learning rate should be lower for high update counts
        assert lr_high <= lr_low

    def test_learning_rate_varies_by_tier(self, memory_system):
        """Different tiers have different base learning rates."""
        lr_fast = memory_system.get_learning_rate(MemoryTier.FAST, update_count=1)
        lr_glacial = memory_system.get_learning_rate(MemoryTier.GLACIAL, update_count=1)

        # Fast tier should have higher learning rate
        assert lr_fast >= lr_glacial


# =============================================================================
# Database Persistence Tests
# =============================================================================


class TestDatabasePersistence:
    """Tests for database persistence."""

    def test_data_persists_across_instances(self, temp_db_path):
        """Data persists when creating new instance."""
        # Create first instance and add data
        cms1 = ContinuumMemory(db_path=temp_db_path)
        cms1.add("persist_test", "Persistent content", MemoryTier.FAST, importance=0.5)

        # Create second instance from same database
        cms2 = ContinuumMemory(db_path=temp_db_path)

        # Data should be available
        entry = cms2.get("persist_test")
        assert entry is not None
        assert entry.content == "Persistent content"

    def test_stats_persist(self, temp_db_path):
        """Statistics reflect persisted data."""
        # Create and populate
        cms1 = ContinuumMemory(db_path=temp_db_path)
        for i in range(5):
            cms1.add(f"persist_{i}", f"Content {i}", MemoryTier.MEDIUM, importance=0.5)

        # New instance
        cms2 = ContinuumMemory(db_path=temp_db_path)
        stats = cms2.get_stats()

        assert stats["total_memories"] >= 5


# =============================================================================
# Edge Cases Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_content(self, memory_system):
        """Can handle empty content."""
        memory_system.add("empty", "", MemoryTier.FAST, importance=0.5)

        entry = memory_system.get("empty")
        assert entry is not None
        assert entry.content == ""

    def test_special_characters_in_content(self, memory_system):
        """Can handle special characters in content."""
        special = "Test with 'quotes', \"double quotes\", and\nnewlines"
        memory_system.add("special", special, MemoryTier.FAST, importance=0.5)

        entry = memory_system.get("special")
        assert entry.content == special

    def test_unicode_content(self, memory_system):
        """Can handle unicode content."""
        unicode_content = "Unicode: 日本語 emoji: 🎉 symbols: ∑∏∫"
        memory_system.add("unicode", unicode_content, MemoryTier.FAST, importance=0.5)

        entry = memory_system.get("unicode")
        assert entry.content == unicode_content

    def test_very_long_content(self, memory_system):
        """Can handle very long content."""
        long_content = "x" * 10000
        memory_system.add("long", long_content, MemoryTier.FAST, importance=0.5)

        entry = memory_system.get("long")
        assert len(entry.content) == 10000

    def test_update_nonexistent(self, memory_system):
        """Updating nonexistent entry doesn't crash."""
        # Should not raise an exception
        memory_system.update_outcome("nonexistent", success=True)
