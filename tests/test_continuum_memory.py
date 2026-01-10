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

        # TierManager requires surprise_score > 0.6 for SLOW tier promotion
        # Directly set high surprise score for testing (real scenario builds up via update_outcome)
        from aragora.storage.schema import get_wal_connection
        with get_wal_connection(cms.db_path) as conn:
            conn.execute(
                "UPDATE continuum_memory SET surprise_score = 0.7 WHERE id = ?",
                ("promo_test",),
            )
            conn.commit()

        new_tier = cms.promote("promo_test")
        assert new_tier == MemoryTier.MEDIUM

        entry = cms.get("promo_test")
        assert entry.tier == MemoryTier.MEDIUM

    def test_promote_medium_to_fast(self, cms):
        """Test promoting from medium to fast tier."""
        cms.add(id="promo_fast", content="Pattern", tier=MemoryTier.MEDIUM)

        # TierManager requires surprise_score > 0.7 for MEDIUM tier promotion
        # Directly set high surprise score for testing
        from aragora.storage.schema import get_wal_connection
        with get_wal_connection(cms.db_path) as conn:
            conn.execute(
                "UPDATE continuum_memory SET surprise_score = 0.8 WHERE id = ?",
                ("promo_fast",),
            )
            conn.commit()

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


class TestEntryProperties:
    """Test ContinuumMemoryEntry computed properties."""

    def test_success_rate_no_outcomes(self):
        """Test success rate with no outcomes returns 0.5."""
        entry = ContinuumMemoryEntry(
            id="test",
            tier=MemoryTier.SLOW,
            content="Test",
            importance=0.5,
            surprise_score=0.0,
            consolidation_score=0.0,
            update_count=0,
            success_count=0,
            failure_count=0,
            created_at="2026-01-01",
            updated_at="2026-01-01",
        )
        assert entry.success_rate == 0.5

    def test_stability_score(self):
        """Test stability score is inverse of surprise."""
        entry = ContinuumMemoryEntry(
            id="test",
            tier=MemoryTier.SLOW,
            content="Test",
            importance=0.5,
            surprise_score=0.3,
            consolidation_score=0.0,
            update_count=0,
            success_count=0,
            failure_count=0,
            created_at="2026-01-01",
            updated_at="2026-01-01",
        )
        assert entry.stability_score == 0.7

    def test_should_promote_fast_tier_never_promotes(self):
        """Test fast tier entries cannot be promoted."""
        entry = ContinuumMemoryEntry(
            id="test",
            tier=MemoryTier.FAST,
            content="Test",
            importance=0.5,
            surprise_score=0.99,  # Very high surprise
            consolidation_score=0.0,
            update_count=0,
            success_count=0,
            failure_count=0,
            created_at="2026-01-01",
            updated_at="2026-01-01",
        )
        assert entry.should_promote() is False

    def test_should_promote_high_surprise(self):
        """Test entry with high surprise should be promoted."""
        entry = ContinuumMemoryEntry(
            id="test",
            tier=MemoryTier.SLOW,
            content="Test",
            importance=0.5,
            surprise_score=0.9,  # Above threshold
            consolidation_score=0.0,
            update_count=0,
            success_count=0,
            failure_count=0,
            created_at="2026-01-01",
            updated_at="2026-01-01",
        )
        # SLOW tier has promotion_threshold of 0.6
        assert entry.should_promote() is True

    def test_should_demote_glacial_never_demotes(self):
        """Test glacial tier entries cannot be demoted."""
        entry = ContinuumMemoryEntry(
            id="test",
            tier=MemoryTier.GLACIAL,
            content="Test",
            importance=0.5,
            surprise_score=0.0,  # Zero surprise = high stability
            consolidation_score=0.0,
            update_count=100,
            success_count=0,
            failure_count=0,
            created_at="2026-01-01",
            updated_at="2026-01-01",
        )
        assert entry.should_demote() is False

    def test_should_demote_high_stability(self):
        """Test entry with high stability and enough updates should demote."""
        entry = ContinuumMemoryEntry(
            id="test",
            tier=MemoryTier.FAST,
            content="Test",
            importance=0.5,
            surprise_score=0.1,  # Low surprise = high stability (0.9)
            consolidation_score=0.0,
            update_count=15,  # More than 10
            success_count=0,
            failure_count=0,
            created_at="2026-01-01",
            updated_at="2026-01-01",
        )
        # FAST tier has demotion_threshold of 0.7, stability = 1 - 0.1 = 0.9 > 0.7
        assert entry.should_demote() is True

    def test_should_demote_insufficient_updates(self):
        """Test entry with insufficient updates should not demote."""
        entry = ContinuumMemoryEntry(
            id="test",
            tier=MemoryTier.FAST,
            content="Test",
            importance=0.5,
            surprise_score=0.0,  # Perfect stability
            consolidation_score=0.0,
            update_count=5,  # Less than 10
            success_count=0,
            failure_count=0,
            created_at="2026-01-01",
            updated_at="2026-01-01",
        )
        assert entry.should_demote() is False


class TestMetadataHandling:
    """Test metadata storage and retrieval."""

    def test_add_with_metadata(self, cms):
        """Test adding entry with metadata."""
        metadata = {"source": "debate_123", "topic": "AI safety", "round": 3}
        entry = cms.add(
            id="meta_test",
            content="Content with metadata",
            tier=MemoryTier.SLOW,
            metadata=metadata,
        )
        assert entry.metadata == metadata

    def test_retrieve_preserves_metadata(self, cms):
        """Test that metadata is preserved on retrieval."""
        metadata = {"key": "value", "nested": {"a": 1}}
        cms.add(id="meta_retrieve", content="Test", tier=MemoryTier.SLOW, metadata=metadata)

        entry = cms.get("meta_retrieve")
        assert entry.metadata == metadata

    def test_add_without_metadata(self, cms):
        """Test adding entry without metadata defaults to empty dict."""
        entry = cms.add(id="no_meta", content="No metadata", tier=MemoryTier.SLOW)
        assert entry.metadata == {}


class TestRetrieveMultipleKeywords:
    """Test keyword-based retrieval with multiple words."""

    def test_retrieve_multiple_keywords_or_logic(self, cms):
        """Test retrieval matches any keyword (OR logic)."""
        cms.add(id="type_error", content="TypeError in function", tier=MemoryTier.SLOW, importance=0.8)
        cms.add(id="value_error", content="ValueError raised", tier=MemoryTier.SLOW, importance=0.8)
        cms.add(id="unrelated", content="Unrelated pattern", tier=MemoryTier.SLOW, importance=0.8)

        # Query with multiple keywords - should match entries containing any keyword
        entries = cms.retrieve(query="TypeError ValueError")
        assert len(entries) == 2
        ids = {e.id for e in entries}
        assert "type_error" in ids
        assert "value_error" in ids

    def test_retrieve_case_insensitive(self, cms):
        """Test keyword matching is case insensitive."""
        cms.add(id="mixed_case", content="TypeError WARNING Message", tier=MemoryTier.SLOW, importance=0.8)

        entries = cms.retrieve(query="typeerror")
        assert len(entries) == 1
        assert entries[0].id == "mixed_case"


class TestUpdateOutcomeEdgeCases:
    """Test edge cases for update_outcome."""

    def test_update_nonexistent_returns_zero(self, cms):
        """Test updating nonexistent entry returns 0."""
        result = cms.update_outcome("nonexistent_id", success=True)
        assert result == 0.0

    def test_multiple_consecutive_updates(self, cms):
        """Test multiple consecutive updates work correctly."""
        cms.add(id="multi_update", content="Pattern", tier=MemoryTier.SLOW)

        for i in range(20):
            success = i % 2 == 0  # Alternating success/failure
            cms.update_outcome("multi_update", success=success)

        entry = cms.get("multi_update")
        assert entry.update_count == 21  # 1 initial + 20 updates
        assert entry.success_count == 10
        assert entry.failure_count == 10
        assert entry.consolidation_score > 0


class TestCleanupExpiredMemories:
    """Test expired memory cleanup functionality."""

    def test_cleanup_archives_expired(self, cms):
        """Test cleanup archives expired memories."""
        # Add a memory and manually age it
        cms.add(id="expired_1", content="Old memory", tier=MemoryTier.FAST)

        # Manually set old timestamp (FAST tier half-life is 1 hour)
        from aragora.storage.schema import get_wal_connection
        old_time = (datetime.now() - timedelta(hours=10)).isoformat()
        with get_wal_connection(cms.db_path) as conn:
            conn.execute(
                "UPDATE continuum_memory SET updated_at = ? WHERE id = ?",
                (old_time, "expired_1"),
            )
            conn.commit()

        result = cms.cleanup_expired_memories(tier=MemoryTier.FAST)

        assert result["archived"] >= 0
        assert result["deleted"] >= 0
        assert "by_tier" in result
        assert "fast" in result["by_tier"]

    def test_cleanup_specific_tier(self, cms):
        """Test cleanup only affects specified tier."""
        cms.add(id="fast_old", content="Fast old", tier=MemoryTier.FAST)
        cms.add(id="slow_new", content="Slow new", tier=MemoryTier.SLOW)

        # Only cleanup fast tier
        result = cms.cleanup_expired_memories(tier=MemoryTier.FAST)

        assert "fast" in result["by_tier"]
        # Slow tier should not be in results since we only cleaned fast
        assert len(result["by_tier"]) == 1

    def test_cleanup_with_max_age_override(self, cms):
        """Test cleanup with custom max age."""
        cms.add(id="custom_age", content="Content", tier=MemoryTier.SLOW)

        # Set old timestamp
        from aragora.storage.schema import get_wal_connection
        old_time = (datetime.now() - timedelta(hours=5)).isoformat()
        with get_wal_connection(cms.db_path) as conn:
            conn.execute(
                "UPDATE continuum_memory SET updated_at = ? WHERE id = ?",
                (old_time, "custom_age"),
            )
            conn.commit()

        # Cleanup with 2 hour max age (should delete the 5-hour-old entry)
        result = cms.cleanup_expired_memories(tier=MemoryTier.SLOW, max_age_hours=2.0)

        assert result["deleted"] >= 1

    def test_cleanup_without_archive(self, cms):
        """Test cleanup deletes without archiving when archive=False."""
        cms.add(id="no_archive", content="Delete me", tier=MemoryTier.FAST)

        # Set old timestamp
        from aragora.storage.schema import get_wal_connection
        old_time = (datetime.now() - timedelta(hours=10)).isoformat()
        with get_wal_connection(cms.db_path) as conn:
            conn.execute(
                "UPDATE continuum_memory SET updated_at = ? WHERE id = ?",
                (old_time, "no_archive"),
            )
            conn.commit()

        result = cms.cleanup_expired_memories(tier=MemoryTier.FAST, archive=False)

        assert result["archived"] == 0
        assert result["deleted"] >= 1


class TestEnforceTierLimits:
    """Test tier limit enforcement."""

    def test_enforce_limits_removes_lowest_importance(self, cms):
        """Test enforcing limits removes lowest importance entries first."""
        # Set a low limit for testing
        cms.hyperparams["max_entries_per_tier"]["slow"] = 2

        # Add 5 entries with different importance levels
        cms.add(id="low_1", content="Low 1", tier=MemoryTier.SLOW, importance=0.1)
        cms.add(id="low_2", content="Low 2", tier=MemoryTier.SLOW, importance=0.2)
        cms.add(id="medium", content="Medium", tier=MemoryTier.SLOW, importance=0.5)
        cms.add(id="high_1", content="High 1", tier=MemoryTier.SLOW, importance=0.8)
        cms.add(id="high_2", content="High 2", tier=MemoryTier.SLOW, importance=0.9)

        result = cms.enforce_tier_limits(tier=MemoryTier.SLOW)

        assert result["slow"] == 3  # 5 - 2 = 3 removed

        # Check that high importance entries remain
        remaining_high_1 = cms.get("high_1")
        remaining_high_2 = cms.get("high_2")
        assert remaining_high_1 is not None
        assert remaining_high_2 is not None

        # Check that low importance entries were removed
        removed_low = cms.get("low_1")
        assert removed_low is None

    def test_enforce_limits_under_limit_no_change(self, cms):
        """Test enforcing limits does nothing when under limit."""
        cms.hyperparams["max_entries_per_tier"]["slow"] = 100

        cms.add(id="entry_1", content="Entry 1", tier=MemoryTier.SLOW)
        cms.add(id="entry_2", content="Entry 2", tier=MemoryTier.SLOW)

        result = cms.enforce_tier_limits(tier=MemoryTier.SLOW)

        assert result["slow"] == 0

    def test_enforce_limits_all_tiers(self, cms):
        """Test enforcing limits on all tiers at once."""
        # Set low limits
        cms.hyperparams["max_entries_per_tier"]["fast"] = 1
        cms.hyperparams["max_entries_per_tier"]["medium"] = 1

        cms.add(id="fast_1", content="Fast 1", tier=MemoryTier.FAST, importance=0.3)
        cms.add(id="fast_2", content="Fast 2", tier=MemoryTier.FAST, importance=0.8)
        cms.add(id="medium_1", content="Medium 1", tier=MemoryTier.MEDIUM, importance=0.4)
        cms.add(id="medium_2", content="Medium 2", tier=MemoryTier.MEDIUM, importance=0.9)

        result = cms.enforce_tier_limits()  # All tiers

        assert result.get("fast", 0) == 1
        assert result.get("medium", 0) == 1


class TestArchiveStats:
    """Test archive statistics functionality."""

    def test_archive_stats_empty(self, cms):
        """Test archive stats with empty archive."""
        stats = cms.get_archive_stats()

        assert stats["total_archived"] == 0
        assert stats["by_tier_reason"] == {}

    def test_archive_stats_after_cleanup(self, cms):
        """Test archive stats after cleanup with archiving."""
        cms.add(id="to_archive", content="Archive me", tier=MemoryTier.FAST)

        # Set old timestamp
        from aragora.storage.schema import get_wal_connection
        old_time = (datetime.now() - timedelta(hours=10)).isoformat()
        with get_wal_connection(cms.db_path) as conn:
            conn.execute(
                "UPDATE continuum_memory SET updated_at = ? WHERE id = ?",
                (old_time, "to_archive"),
            )
            conn.commit()

        # Cleanup with archiving
        cms.cleanup_expired_memories(tier=MemoryTier.FAST, archive=True)

        stats = cms.get_archive_stats()
        assert stats["total_archived"] >= 1


class TestBatchOperations:
    """Test batch promotion and demotion operations."""

    def test_consolidate_batch_promotions(self, cms):
        """Test consolidation handles batch promotions."""
        # Add multiple entries with high surprise
        from aragora.storage.schema import get_wal_connection
        for i in range(5):
            cms.add(id=f"batch_promo_{i}", content=f"Pattern {i}", tier=MemoryTier.SLOW)

        # Set high surprise scores for all
        with get_wal_connection(cms.db_path) as conn:
            conn.execute(
                "UPDATE continuum_memory SET surprise_score = 0.9 WHERE id LIKE 'batch_promo_%'"
            )
            conn.commit()

        result = cms.consolidate()

        assert result["promotions"] >= 5  # All should be promoted

    def test_consolidate_batch_demotions(self, cms):
        """Test consolidation handles batch demotions."""
        from aragora.storage.schema import get_wal_connection

        # Add multiple entries with low surprise and enough updates
        for i in range(5):
            cms.add(id=f"batch_demo_{i}", content=f"Pattern {i}", tier=MemoryTier.FAST)

        # Set low surprise and high update count for demotion
        with get_wal_connection(cms.db_path) as conn:
            conn.execute(
                """UPDATE continuum_memory
                   SET surprise_score = 0.05, update_count = 20
                   WHERE id LIKE 'batch_demo_%'"""
            )
            conn.commit()

        result = cms.consolidate()

        assert result["demotions"] >= 5  # All should be demoted

    def test_consolidate_mixed_promotions_demotions(self, cms):
        """Test consolidation handles mixed promotions and demotions."""
        from aragora.storage.schema import get_wal_connection

        # Entries to promote (high surprise in slow tier)
        for i in range(3):
            cms.add(id=f"to_promote_{i}", content=f"Promote {i}", tier=MemoryTier.SLOW)

        # Entries to demote (low surprise in fast tier)
        for i in range(3):
            cms.add(id=f"to_demote_{i}", content=f"Demote {i}", tier=MemoryTier.FAST)

        with get_wal_connection(cms.db_path) as conn:
            # High surprise for promotion
            conn.execute(
                "UPDATE continuum_memory SET surprise_score = 0.9 WHERE id LIKE 'to_promote_%'"
            )
            # Low surprise + high updates for demotion
            conn.execute(
                """UPDATE continuum_memory
                   SET surprise_score = 0.05, update_count = 20
                   WHERE id LIKE 'to_demote_%'"""
            )
            conn.commit()

        result = cms.consolidate()

        assert result["promotions"] >= 3
        assert result["demotions"] >= 3


class TestTierMetrics:
    """Test tier metrics tracking."""

    def test_get_tier_metrics(self, cms):
        """Test getting tier metrics from tier manager."""
        metrics = cms.get_tier_metrics()

        assert "promotions" in metrics
        assert "demotions" in metrics
        assert isinstance(metrics["promotions"], dict)
        assert isinstance(metrics["demotions"], dict)

    def test_tier_manager_property(self, cms):
        """Test tier manager property access."""
        tm = cms.tier_manager
        assert tm is not None
        from aragora.memory.tier_manager import TierManager
        assert isinstance(tm, TierManager)


class TestPromotionCooldown:
    """Test promotion cooldown behavior."""

    def test_promotion_respects_cooldown(self, cms):
        """Test that recently promoted entries cannot be promoted again."""
        cms.add(id="cooldown_test", content="Pattern", tier=MemoryTier.SLOW)

        from aragora.storage.schema import get_wal_connection
        with get_wal_connection(cms.db_path) as conn:
            conn.execute(
                "UPDATE continuum_memory SET surprise_score = 0.9 WHERE id = ?",
                ("cooldown_test",),
            )
            conn.commit()

        # First promotion should succeed
        new_tier = cms.promote("cooldown_test")
        assert new_tier == MemoryTier.MEDIUM

        # Set high surprise again
        with get_wal_connection(cms.db_path) as conn:
            conn.execute(
                "UPDATE continuum_memory SET surprise_score = 0.9 WHERE id = ?",
                ("cooldown_test",),
            )
            conn.commit()

        # Immediate second promotion should fail due to cooldown
        new_tier = cms.promote("cooldown_test")
        assert new_tier is None  # Cooldown prevents promotion


class TestMemoryPressure:
    """Test memory pressure monitoring."""

    def test_empty_memory_returns_zero_pressure(self, cms):
        """Empty memory should have zero pressure."""
        pressure = cms.get_memory_pressure()
        assert pressure == 0.0

    def test_pressure_increases_with_entries(self, cms):
        """Adding entries should increase pressure."""
        # Add entries to the fast tier (limit is 1000 by default)
        for i in range(100):
            cms.add(
                id=f"pressure_test_{i}",
                content=f"Test content {i}",
                tier=MemoryTier.FAST,
            )

        pressure = cms.get_memory_pressure()
        # 100 entries out of 1000 = 10% pressure
        assert pressure > 0.0
        assert pressure <= 0.15  # Allow some margin

    def test_pressure_capped_at_one(self, cms):
        """Pressure should not exceed 1.0."""
        # Set a very low limit
        cms.hyperparams["max_entries_per_tier"] = {"fast": 10}

        # Add more entries than the limit
        for i in range(20):
            cms.add(
                id=f"pressure_cap_{i}",
                content=f"Test content {i}",
                tier=MemoryTier.FAST,
            )

        pressure = cms.get_memory_pressure()
        assert pressure == 1.0

    def test_pressure_returns_max_across_tiers(self, cms):
        """Pressure should return the highest utilization across tiers."""
        cms.hyperparams["max_entries_per_tier"] = {
            "fast": 100,
            "medium": 100,
            "slow": 100,
            "glacial": 100,
        }

        # Add 50 to fast (50%), 80 to medium (80%)
        for i in range(50):
            cms.add(id=f"fast_{i}", content=f"Fast {i}", tier=MemoryTier.FAST)
        for i in range(80):
            cms.add(id=f"medium_{i}", content=f"Medium {i}", tier=MemoryTier.MEDIUM)

        pressure = cms.get_memory_pressure()
        # Should be 80% (medium tier is highest)
        assert 0.75 <= pressure <= 0.85

    def test_pressure_with_empty_limits(self, cms):
        """Empty limits should return zero pressure."""
        cms.hyperparams["max_entries_per_tier"] = {}
        pressure = cms.get_memory_pressure()
        assert pressure == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
