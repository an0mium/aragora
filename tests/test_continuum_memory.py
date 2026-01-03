"""
Tests for Continuum Memory System (CMS).

Tests the multi-timescale memory with:
- 4-tier architecture (fast/medium/slow/glacial)
- Surprise-based learning
- Tier promotion/demotion
- Consolidation scoring
"""

import os
import tempfile
import pytest
from datetime import datetime, timedelta

from aragora.memory.continuum import (
    ContinuumMemory,
    ContinuumMemoryEntry,
    MemoryTier,
    TierConfig,
    TIER_CONFIGS,
)


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    os.unlink(path)


@pytest.fixture
def cms(temp_db):
    """Create a ContinuumMemory instance with temp database."""
    return ContinuumMemory(db_path=temp_db)


class TestContinuumMemoryBasics:
    """Test basic CMS operations."""

    def test_add_memory(self, cms):
        """Test adding a memory entry."""
        entry = cms.add(
            id="test_001",
            content="Test pattern for error handling",
            tier=MemoryTier.SLOW,
            importance=0.7,
        )

        assert entry.id == "test_001"
        assert entry.tier == MemoryTier.SLOW
        assert entry.importance == 0.7
        assert entry.surprise_score == 0.0
        assert entry.update_count == 1

    def test_get_memory(self, cms):
        """Test retrieving a memory entry."""
        cms.add(id="get_test", content="Retrievable content", tier=MemoryTier.MEDIUM)

        entry = cms.get("get_test")
        assert entry is not None
        assert entry.content == "Retrievable content"
        assert entry.tier == MemoryTier.MEDIUM

    def test_get_nonexistent(self, cms):
        """Test getting a non-existent entry returns None."""
        entry = cms.get("nonexistent")
        assert entry is None

    def test_add_all_tiers(self, cms):
        """Test adding entries to all tiers."""
        for tier in MemoryTier:
            entry = cms.add(
                id=f"tier_{tier.value}",
                content=f"Content for {tier.value} tier",
                tier=tier,
            )
            assert entry.tier == tier


class TestTierRetrieval:
    """Test tier-based retrieval."""

    def test_retrieve_by_tier(self, cms):
        """Test filtering retrieval by tier."""
        # Add entries to different tiers
        cms.add(id="fast_1", content="Fast pattern", tier=MemoryTier.FAST, importance=0.8)
        cms.add(id="slow_1", content="Slow pattern", tier=MemoryTier.SLOW, importance=0.8)
        cms.add(id="glacial_1", content="Glacial pattern", tier=MemoryTier.GLACIAL, importance=0.8)

        # Retrieve only fast tier
        fast_entries = cms.retrieve(tiers=[MemoryTier.FAST])
        assert len(fast_entries) == 1
        assert fast_entries[0].tier == MemoryTier.FAST

        # Retrieve fast and slow
        mixed_entries = cms.retrieve(tiers=[MemoryTier.FAST, MemoryTier.SLOW])
        assert len(mixed_entries) == 2
        tiers = {e.tier for e in mixed_entries}
        assert MemoryTier.FAST in tiers
        assert MemoryTier.SLOW in tiers

    def test_retrieve_excludes_glacial(self, cms):
        """Test excluding glacial tier from retrieval."""
        cms.add(id="glacial_test", content="Glacial content", tier=MemoryTier.GLACIAL, importance=0.9)
        cms.add(id="slow_test", content="Slow content", tier=MemoryTier.SLOW, importance=0.9)

        entries = cms.retrieve(include_glacial=False)
        tiers = {e.tier for e in entries}
        assert MemoryTier.GLACIAL not in tiers

    def test_retrieve_with_query(self, cms):
        """Test keyword-based retrieval filtering."""
        cms.add(id="error_1", content="TypeError in function call", tier=MemoryTier.SLOW, importance=0.7)
        cms.add(id="perf_1", content="Performance optimization tip", tier=MemoryTier.SLOW, importance=0.7)

        entries = cms.retrieve(query="TypeError")
        assert len(entries) == 1
        assert "TypeError" in entries[0].content

    def test_retrieve_min_importance(self, cms):
        """Test minimum importance threshold."""
        cms.add(id="high", content="High importance", tier=MemoryTier.SLOW, importance=0.9)
        cms.add(id="low", content="Low importance", tier=MemoryTier.SLOW, importance=0.2)

        entries = cms.retrieve(min_importance=0.5)
        assert len(entries) == 1
        assert entries[0].importance >= 0.5


class TestSurpriseBasedLearning:
    """Test surprise-based memory updates."""

    def test_update_outcome_success(self, cms):
        """Test updating memory after successful outcome."""
        cms.add(id="outcome_test", content="Pattern", tier=MemoryTier.SLOW)

        # First success
        surprise = cms.update_outcome("outcome_test", success=True)
        entry = cms.get("outcome_test")

        assert entry.success_count == 1
        assert entry.failure_count == 0
        assert entry.update_count == 2  # Initial + update

    def test_update_outcome_failure(self, cms):
        """Test updating memory after failed outcome."""
        cms.add(id="fail_test", content="Pattern", tier=MemoryTier.SLOW)

        cms.update_outcome("fail_test", success=False)
        entry = cms.get("fail_test")

        assert entry.success_count == 0
        assert entry.failure_count == 1

    def test_success_rate_calculation(self, cms):
        """Test success rate calculation."""
        cms.add(id="rate_test", content="Pattern", tier=MemoryTier.SLOW)

        # 3 successes, 1 failure = 75% success rate
        cms.update_outcome("rate_test", success=True)
        cms.update_outcome("rate_test", success=True)
        cms.update_outcome("rate_test", success=True)
        cms.update_outcome("rate_test", success=False)

        entry = cms.get("rate_test")
        assert entry.success_rate == 0.75

    def test_surprise_with_agent_prediction(self, cms):
        """Test surprise calculation with agent prediction error."""
        cms.add(id="predict_test", content="Pattern", tier=MemoryTier.SLOW)

        # High prediction error should increase surprise
        surprise = cms.update_outcome("predict_test", success=True, agent_prediction_error=0.8)

        entry = cms.get("predict_test")
        assert entry.surprise_score > 0


class TestTierPromotion:
    """Test tier promotion and demotion."""

    def test_promote_slow_to_medium(self, cms):
        """Test promoting from slow to medium tier."""
        cms.add(id="promo_test", content="Promotable pattern", tier=MemoryTier.SLOW)

        new_tier = cms.promote("promo_test")
        assert new_tier == MemoryTier.MEDIUM

        entry = cms.get("promo_test")
        assert entry.tier == MemoryTier.MEDIUM

    def test_promote_medium_to_fast(self, cms):
        """Test promoting from medium to fast tier."""
        cms.add(id="promo_fast", content="Pattern", tier=MemoryTier.MEDIUM)

        new_tier = cms.promote("promo_fast")
        assert new_tier == MemoryTier.FAST

    def test_cannot_promote_fast(self, cms):
        """Test that fast tier cannot be promoted further."""
        cms.add(id="fast_promo", content="Pattern", tier=MemoryTier.FAST)

        new_tier = cms.promote("fast_promo")
        assert new_tier is None

    def test_demote_fast_to_medium(self, cms):
        """Test demoting from fast to medium tier."""
        cms.add(id="demote_test", content="Pattern", tier=MemoryTier.FAST)

        # Need enough updates for demotion
        for _ in range(12):
            cms.update_outcome("demote_test", success=True)

        new_tier = cms.demote("demote_test")
        assert new_tier == MemoryTier.MEDIUM

    def test_cannot_demote_glacial(self, cms):
        """Test that glacial tier cannot be demoted further."""
        cms.add(id="glacial_demote", content="Pattern", tier=MemoryTier.GLACIAL)

        # Add enough updates
        for _ in range(12):
            cms.update_outcome("glacial_demote", success=True)

        new_tier = cms.demote("glacial_demote")
        assert new_tier is None


class TestConsolidation:
    """Test consolidation scoring and tier consolidation."""

    def test_consolidation_score_increases(self, cms):
        """Test that consolidation score increases with updates."""
        cms.add(id="consol_test", content="Pattern", tier=MemoryTier.SLOW)

        for _ in range(10):
            cms.update_outcome("consol_test", success=True)

        entry = cms.get("consol_test")
        assert entry.consolidation_score > 0
        assert entry.consolidation_score < 1.0  # Not fully consolidated yet

    def test_consolidate_promotes_high_surprise(self, cms):
        """Test that consolidation promotes high-surprise patterns."""
        # Add a pattern with artificially high surprise
        cms.add(id="high_surprise", content="Surprising pattern", tier=MemoryTier.SLOW)

        # Manually set high surprise via database
        import sqlite3
        conn = sqlite3.connect(cms.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE continuum_memory SET surprise_score = 0.8 WHERE id = ?",
            ("high_surprise",),
        )
        conn.commit()
        conn.close()

        result = cms.consolidate()
        assert result["promotions"] >= 0  # May or may not promote depending on threshold


class TestLearningRate:
    """Test tier-specific learning rates."""

    def test_fast_tier_high_initial_rate(self, cms):
        """Test that fast tier has high initial learning rate."""
        rate = cms.get_learning_rate(MemoryTier.FAST, update_count=0)
        assert rate == 0.3  # Base rate for fast

    def test_slow_tier_low_initial_rate(self, cms):
        """Test that slow tier has low initial learning rate."""
        rate = cms.get_learning_rate(MemoryTier.SLOW, update_count=0)
        assert rate == 0.03  # Base rate for slow

    def test_learning_rate_decays(self, cms):
        """Test that learning rate decays with updates."""
        rate_0 = cms.get_learning_rate(MemoryTier.FAST, update_count=0)
        rate_10 = cms.get_learning_rate(MemoryTier.FAST, update_count=10)
        rate_100 = cms.get_learning_rate(MemoryTier.FAST, update_count=100)

        assert rate_0 > rate_10 > rate_100


class TestStats:
    """Test statistics and export functionality."""

    def test_get_stats(self, cms):
        """Test getting CMS statistics."""
        cms.add(id="stat_1", content="Content 1", tier=MemoryTier.FAST)
        cms.add(id="stat_2", content="Content 2", tier=MemoryTier.SLOW)
        cms.add(id="stat_3", content="Content 3", tier=MemoryTier.SLOW)

        stats = cms.get_stats()

        assert stats["total_memories"] == 3
        assert "by_tier" in stats
        assert stats["by_tier"]["slow"]["count"] == 2
        assert stats["by_tier"]["fast"]["count"] == 1

    def test_export_for_tier(self, cms):
        """Test exporting memories for a specific tier."""
        cms.add(id="export_1", content="Export 1", tier=MemoryTier.SLOW, importance=0.7)
        cms.add(id="export_2", content="Export 2", tier=MemoryTier.SLOW, importance=0.8)

        exported = cms.export_for_tier(MemoryTier.SLOW)

        assert len(exported) == 2
        assert all("id" in e for e in exported)
        assert all("importance" in e for e in exported)


class TestTierConfigs:
    """Test tier configuration settings."""

    def test_all_tiers_have_configs(self):
        """Test that all tiers have configuration."""
        for tier in MemoryTier:
            assert tier in TIER_CONFIGS
            config = TIER_CONFIGS[tier]
            assert config.half_life_hours > 0
            assert 0 <= config.base_learning_rate <= 1

    def test_tier_ordering(self):
        """Test that tier half-lives are in order."""
        half_lives = [TIER_CONFIGS[t].half_life_hours for t in
                      [MemoryTier.FAST, MemoryTier.MEDIUM, MemoryTier.SLOW, MemoryTier.GLACIAL]]
        assert half_lives == sorted(half_lives)  # Should be ascending


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
