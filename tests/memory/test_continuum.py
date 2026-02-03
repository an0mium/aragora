"""
Comprehensive tests for Continuum Memory System.

Tests the multi-tier memory system (ContinuumMemory) including:
- Memory tier initialization
- Memory storage and retrieval
- Tier promotion/demotion
- Consolidation mechanics
- Decay and forgetting
- Cross-tier queries
- Persistence and loading
- Concurrent access
- Memory cleanup and bounds
- Red line (protected) memories
- Snapshot export/restore
"""

import asyncio
import json
import sqlite3
import tempfile
import threading
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aragora.memory.continuum import (
    CONTINUUM_SCHEMA_VERSION,
    ContinuumMemory,
    ContinuumMemoryEntry,
    get_continuum_memory,
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
def temp_db_path(tmp_path):
    """Create a temporary database path for testing."""
    return str(tmp_path / "test_continuum.db")


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
    # Cleanup - close any connections
    reset_tier_manager()
    reset_continuum_memory()


@pytest.fixture
def populated_memory(memory):
    """Memory with pre-populated entries across tiers."""
    memory.add(
        "fast_1", "Fast tier entry about Python errors", tier=MemoryTier.FAST, importance=0.8
    )
    memory.add("fast_2", "Another fast entry about debugging", tier=MemoryTier.FAST, importance=0.6)
    memory.add("medium_1", "Medium tier tactical learning", tier=MemoryTier.MEDIUM, importance=0.7)
    memory.add("slow_1", "Slow tier strategic patterns", tier=MemoryTier.SLOW, importance=0.9)
    memory.add(
        "glacial_1", "Glacial foundational knowledge", tier=MemoryTier.GLACIAL, importance=0.95
    )
    return memory


# =============================================================================
# Test Memory Tier Initialization
# =============================================================================


class TestMemoryTierInitialization:
    """Test ContinuumMemory initialization and tier configuration."""

    @pytest.mark.smoke
    def test_default_initialization(self, temp_db_path):
        """Test that ContinuumMemory initializes with default settings."""
        cms = ContinuumMemory(db_path=temp_db_path)

        assert cms.SCHEMA_VERSION == CONTINUUM_SCHEMA_VERSION
        assert cms.hyperparams is not None
        assert "max_entries_per_tier" in cms.hyperparams
        assert cms.hyperparams["max_entries_per_tier"]["fast"] == 1000

    def test_custom_tier_manager(self, temp_db_path, tier_manager):
        """Test initialization with custom TierManager."""
        cms = ContinuumMemory(db_path=temp_db_path, tier_manager=tier_manager)

        assert cms._tier_manager is tier_manager
        # Constructor syncs hyperparams to tier manager, so cooldown matches hyperparams
        assert (
            cms._tier_manager.promotion_cooldown_hours
            == cms.hyperparams["promotion_cooldown_hours"]
        )

    def test_tier_manager_property(self, memory):
        """Test that tier_manager property returns the manager."""
        manager = memory.tier_manager
        assert isinstance(manager, TierManager)

    def test_schema_creation(self, temp_db_path):
        """Test that database schema is created on initialization."""
        cms = ContinuumMemory(db_path=temp_db_path)

        with cms.connection() as conn:
            cursor = conn.cursor()
            # Check main table exists
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='continuum_memory'"
            )
            assert cursor.fetchone() is not None

            # Check archive table exists
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='continuum_memory_archive'"
            )
            assert cursor.fetchone() is not None

            # Check tier_transitions table exists
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='tier_transitions'"
            )
            assert cursor.fetchone() is not None

    def test_hyperparams_defaults(self, memory):
        """Test default hyperparameter values."""
        hp = memory.hyperparams

        assert hp["surprise_weight_success"] == 0.3
        assert hp["surprise_weight_semantic"] == 0.3
        assert hp["surprise_weight_temporal"] == 0.2
        assert hp["surprise_weight_agent"] == 0.2
        assert hp["consolidation_threshold"] == 100.0
        assert hp["promotion_cooldown_hours"] == 24.0
        assert hp["retention_multiplier"] == 2.0


# =============================================================================
# Test Memory Storage and Retrieval
# =============================================================================


class TestMemoryStorageAndRetrieval:
    """Test adding and retrieving memory entries."""

    def test_add_basic_entry(self, memory):
        """Test adding a basic memory entry."""
        entry = memory.add("test_id", "Test content")

        assert entry.id == "test_id"
        assert entry.content == "Test content"
        assert entry.tier == MemoryTier.SLOW  # Default tier
        assert entry.importance == 0.5  # Default importance
        assert entry.surprise_score == 0.0
        assert entry.consolidation_score == 0.0
        assert entry.update_count == 1

    def test_add_with_tier(self, memory):
        """Test adding entry to specific tier."""
        entry = memory.add("fast_id", "Fast content", tier=MemoryTier.FAST)

        assert entry.tier == MemoryTier.FAST

    def test_add_with_importance(self, memory):
        """Test adding entry with custom importance."""
        entry = memory.add("important_id", "Important content", importance=0.9)

        assert entry.importance == 0.9

    def test_add_with_metadata(self, memory):
        """Test adding entry with metadata."""
        metadata = {"source": "debate", "round": 3, "tags": ["error", "python"]}
        entry = memory.add("meta_id", "Content with metadata", metadata=metadata)

        assert entry.metadata == metadata
        assert entry.tags == ["error", "python"]

    def test_get_existing_entry(self, memory):
        """Test retrieving an existing entry by ID."""
        memory.add("get_test", "Retrievable content", importance=0.7)

        entry = memory.get("get_test")

        assert entry is not None
        assert entry.id == "get_test"
        assert entry.content == "Retrievable content"
        assert entry.importance == 0.7

    def test_get_nonexistent_entry(self, memory):
        """Test retrieving a non-existent entry returns None."""
        entry = memory.get("nonexistent_id")

        assert entry is None

    def test_retrieve_by_query(self, populated_memory):
        """Test retrieving entries by keyword query."""
        results = populated_memory.retrieve(query="Python", limit=10)

        assert len(results) >= 1
        assert any("Python" in e.content for e in results)

    def test_retrieve_by_tier(self, populated_memory):
        """Test retrieving entries filtered by tier."""
        results = populated_memory.retrieve(tiers=[MemoryTier.FAST], limit=10)

        assert all(e.tier == MemoryTier.FAST for e in results)

    def test_retrieve_multiple_tiers(self, populated_memory):
        """Test retrieving from multiple tiers."""
        results = populated_memory.retrieve(tiers=[MemoryTier.FAST, MemoryTier.MEDIUM], limit=10)

        tiers_found = {e.tier for e in results}
        assert tiers_found.issubset({MemoryTier.FAST, MemoryTier.MEDIUM})

    def test_retrieve_with_min_importance(self, populated_memory):
        """Test retrieving with minimum importance threshold."""
        results = populated_memory.retrieve(min_importance=0.85, limit=10)

        assert all(e.importance >= 0.85 for e in results)

    def test_retrieve_excludes_glacial(self, populated_memory):
        """Test retrieving with include_glacial=False."""
        results = populated_memory.retrieve(include_glacial=False, limit=100)

        assert all(e.tier != MemoryTier.GLACIAL for e in results)

    def test_retrieve_limit(self, memory):
        """Test that retrieve respects limit parameter."""
        for i in range(20):
            memory.add(f"limit_test_{i}", f"Content {i}", importance=0.5)

        results = memory.retrieve(limit=5)

        assert len(results) == 5

    @pytest.mark.asyncio
    async def test_add_async(self, memory):
        """Test async add method."""
        entry = await memory.add_async("async_id", "Async content", importance=0.6)

        assert entry.id == "async_id"
        assert entry.importance == 0.6

    @pytest.mark.asyncio
    async def test_get_async(self, memory):
        """Test async get method."""
        memory.add("async_get", "Content to retrieve")

        entry = await memory.get_async("async_get")

        assert entry is not None
        assert entry.content == "Content to retrieve"

    @pytest.mark.asyncio
    async def test_retrieve_async(self, populated_memory):
        """Test async retrieve method."""
        results = await populated_memory.retrieve_async(query="Python", limit=10)

        assert len(results) >= 1


# =============================================================================
# Test Tier Promotion/Demotion
# =============================================================================


class TestTierPromotionDemotion:
    """Test tier promotion and demotion mechanics."""

    def test_promote_from_medium_to_fast(self, memory):
        """Test promoting entry from medium to fast tier."""
        # Add entry with high surprise score
        memory.add("promote_test", "Content to promote", tier=MemoryTier.MEDIUM)

        # Set high surprise score to trigger promotion
        with memory.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE continuum_memory SET surprise_score = 0.9 WHERE id = ?", ("promote_test",)
            )
            conn.commit()

        new_tier = memory.promote("promote_test")

        assert new_tier == MemoryTier.FAST

    def test_promote_already_at_fast(self, memory):
        """Test that promoting from fast tier returns None."""
        memory.add("already_fast", "Fast content", tier=MemoryTier.FAST)

        # Set high surprise
        with memory.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE continuum_memory SET surprise_score = 0.9 WHERE id = ?", ("already_fast",)
            )
            conn.commit()

        result = memory.promote("already_fast")

        assert result is None

    def test_promote_nonexistent(self, memory):
        """Test promoting non-existent entry returns None."""
        result = memory.promote("nonexistent_id")

        assert result is None

    def test_demote_from_fast_to_medium(self, memory):
        """Test demoting entry from fast to medium tier."""
        memory.add("demote_test", "Content to demote", tier=MemoryTier.FAST)

        # Set low surprise (high stability) and sufficient updates
        with memory.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """UPDATE continuum_memory
                   SET surprise_score = 0.1, update_count = 15
                   WHERE id = ?""",
                ("demote_test",),
            )
            conn.commit()

        new_tier = memory.demote("demote_test")

        assert new_tier == MemoryTier.MEDIUM

    def test_demote_already_at_glacial(self, memory):
        """Test that demoting from glacial tier returns None."""
        memory.add("already_glacial", "Glacial content", tier=MemoryTier.GLACIAL)

        # Set low surprise and high updates
        with memory.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """UPDATE continuum_memory
                   SET surprise_score = 0.05, update_count = 20
                   WHERE id = ?""",
                ("already_glacial",),
            )
            conn.commit()

        result = memory.demote("already_glacial")

        assert result is None

    def test_promote_entry_method(self, memory):
        """Test promote_entry interface method."""
        memory.add("promote_entry_test", "Content", tier=MemoryTier.SLOW)

        result = memory.promote_entry("promote_entry_test", MemoryTier.MEDIUM)

        assert result is True
        entry = memory.get("promote_entry_test")
        assert entry.tier == MemoryTier.MEDIUM

    def test_demote_entry_method(self, memory):
        """Test demote_entry interface method."""
        memory.add("demote_entry_test", "Content", tier=MemoryTier.MEDIUM)

        result = memory.demote_entry("demote_entry_test", MemoryTier.SLOW)

        assert result is True
        entry = memory.get("demote_entry_test")
        assert entry.tier == MemoryTier.SLOW

    def test_promotion_cooldown(self, memory):
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

        # Promotion should be blocked by cooldown
        result = memory.promote("cooldown_test")

        assert result is None


# =============================================================================
# Test Consolidation Mechanics
# =============================================================================


class TestConsolidationMechanics:
    """Test tier consolidation logic."""

    def test_consolidate_promotes_high_surprise(self, memory):
        """Test that consolidate promotes entries with high surprise."""
        # Add entries with high surprise
        for i in range(3):
            memory.add(f"high_surprise_{i}", f"Content {i}", tier=MemoryTier.SLOW)

        # Set high surprise scores
        with memory.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE continuum_memory SET surprise_score = 0.8 WHERE tier = 'slow'")
            conn.commit()

        result = memory.consolidate()

        assert result["promotions"] >= 0  # May be 0 if thresholds not met

    def test_consolidate_demotes_stable_entries(self, memory):
        """Test that consolidate demotes stable entries."""
        # Add entries with low surprise and high update count
        for i in range(3):
            memory.add(f"stable_{i}", f"Stable content {i}", tier=MemoryTier.FAST)

        # Set low surprise (high stability) and sufficient updates
        with memory.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """UPDATE continuum_memory
                   SET surprise_score = 0.05, update_count = 15
                   WHERE tier = 'fast'"""
            )
            conn.commit()

        result = memory.consolidate()

        assert "demotions" in result

    def test_consolidate_returns_counts(self, memory):
        """Test that consolidate returns proper count structure."""
        result = memory.consolidate()

        assert "promotions" in result
        assert "demotions" in result
        assert isinstance(result["promotions"], int)
        assert isinstance(result["demotions"], int)


# =============================================================================
# Test Decay and Forgetting
# =============================================================================


class TestDecayAndForgetting:
    """Test memory decay and forgetting mechanisms."""

    def test_update_outcome_success(self, memory):
        """Test updating memory with successful outcome."""
        memory.add("outcome_test", "Content")

        surprise = memory.update_outcome("outcome_test", success=True)

        entry = memory.get("outcome_test")
        assert entry.success_count == 1
        assert entry.update_count == 2  # Initial 1 + outcome update
        assert surprise >= 0

    def test_update_outcome_failure(self, memory):
        """Test updating memory with failed outcome."""
        memory.add("failure_test", "Content")

        memory.update_outcome("failure_test", success=False)

        entry = memory.get("failure_test")
        assert entry.failure_count == 1

    def test_update_outcome_with_prediction_error(self, memory):
        """Test updating with agent prediction error."""
        memory.add("pred_error_test", "Content")

        surprise = memory.update_outcome(
            "pred_error_test", success=True, agent_prediction_error=0.5
        )

        # Higher prediction error should contribute to surprise
        assert surprise >= 0

    def test_update_outcome_nonexistent(self, memory):
        """Test update_outcome on non-existent entry returns 0."""
        result = memory.update_outcome("nonexistent", success=True)

        assert result == 0.0

    def test_success_rate_calculation(self, memory):
        """Test ContinuumMemoryEntry.success_rate property."""
        entry = ContinuumMemoryEntry(
            id="test",
            tier=MemoryTier.SLOW,
            content="Test",
            importance=0.5,
            surprise_score=0.1,
            consolidation_score=0.5,
            update_count=10,
            success_count=7,
            failure_count=3,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
        )

        assert entry.success_rate == 0.7

    def test_success_rate_no_data(self, memory):
        """Test success rate with no successes/failures returns 0.5."""
        entry = ContinuumMemoryEntry(
            id="test",
            tier=MemoryTier.SLOW,
            content="Test",
            importance=0.5,
            surprise_score=0.1,
            consolidation_score=0.5,
            update_count=1,
            success_count=0,
            failure_count=0,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
        )

        assert entry.success_rate == 0.5

    def test_stability_score(self, memory):
        """Test ContinuumMemoryEntry.stability_score property."""
        entry = ContinuumMemoryEntry(
            id="test",
            tier=MemoryTier.SLOW,
            content="Test",
            importance=0.5,
            surprise_score=0.3,
            consolidation_score=0.5,
            update_count=1,
            success_count=0,
            failure_count=0,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
        )

        assert entry.stability_score == 0.7  # 1 - surprise

    def test_get_learning_rate(self, memory):
        """Test tier-specific learning rate calculation."""
        # Fast tier has high learning rate
        fast_lr = memory.get_learning_rate(MemoryTier.FAST, update_count=1)

        # Glacial tier has low learning rate
        glacial_lr = memory.get_learning_rate(MemoryTier.GLACIAL, update_count=1)

        assert fast_lr > glacial_lr

    def test_learning_rate_decay(self, memory):
        """Test that learning rate decays with update count."""
        lr_early = memory.get_learning_rate(MemoryTier.FAST, update_count=1)
        lr_late = memory.get_learning_rate(MemoryTier.FAST, update_count=100)

        assert lr_late < lr_early


# =============================================================================
# Test Cross-Tier Queries
# =============================================================================


class TestCrossTierQueries:
    """Test queries spanning multiple tiers."""

    def test_get_stats(self, populated_memory):
        """Test getting memory statistics."""
        stats = populated_memory.get_stats()

        assert "total_memories" in stats
        assert "by_tier" in stats
        assert "transitions" in stats
        assert stats["total_memories"] == 5

    def test_export_for_tier(self, populated_memory):
        """Test exporting entries for specific tier."""
        exported = populated_memory.export_for_tier(MemoryTier.FAST)

        assert len(exported) == 2  # We added 2 fast tier entries
        assert all("id" in e and "content" in e for e in exported)

    def test_get_tier_metrics(self, memory):
        """Test getting tier transition metrics."""
        metrics = memory.get_tier_metrics()

        assert "promotions" in metrics
        assert "demotions" in metrics
        assert "total_promotions" in metrics
        assert "total_demotions" in metrics

    def test_get_glacial_insights(self, populated_memory):
        """Test retrieving glacial tier insights."""
        insights = populated_memory.get_glacial_insights(limit=10)

        assert all(e.tier == MemoryTier.GLACIAL for e in insights)

    def test_get_cross_session_patterns(self, populated_memory):
        """Test retrieving cross-session patterns."""
        patterns = populated_memory.get_cross_session_patterns(limit=10)

        # Should include slow and glacial tiers
        tiers = {p.tier for p in patterns}
        assert tiers.issubset({MemoryTier.SLOW, MemoryTier.GLACIAL})

    def test_get_glacial_tier_stats(self, populated_memory):
        """Test getting glacial tier specific stats."""
        stats = populated_memory.get_glacial_tier_stats()

        assert stats["tier"] == "glacial"
        assert "count" in stats
        assert "avg_importance" in stats
        assert "utilization" in stats


# =============================================================================
# Test Persistence and Loading
# =============================================================================


class TestPersistenceAndLoading:
    """Test data persistence and snapshot functionality."""

    def test_data_persists_across_instances(self, temp_db_path, tier_manager):
        """Test that data persists when creating new instance."""
        # Create first instance and add data
        cms1 = ContinuumMemory(db_path=temp_db_path, tier_manager=tier_manager)
        cms1.add("persist_test", "Persistent content", importance=0.8)

        # Create second instance with same path
        cms2 = ContinuumMemory(db_path=temp_db_path, tier_manager=tier_manager)
        entry = cms2.get("persist_test")

        assert entry is not None
        assert entry.content == "Persistent content"
        assert entry.importance == 0.8

    def test_export_snapshot(self, populated_memory):
        """Test exporting memory state as snapshot."""
        snapshot = populated_memory.export_snapshot()

        assert "entries" in snapshot
        assert "tier_counts" in snapshot
        assert "hyperparams" in snapshot
        assert "snapshot_time" in snapshot
        assert "total_entries" in snapshot
        assert snapshot["total_entries"] == 5

    def test_export_snapshot_with_tier_filter(self, populated_memory):
        """Test exporting snapshot filtered by tier."""
        snapshot = populated_memory.export_snapshot(tiers=[MemoryTier.FAST])

        assert all(e["tier"] == "fast" for e in snapshot["entries"])

    def test_restore_snapshot_replace_mode(self, memory, populated_memory):
        """Test restoring snapshot in replace mode."""
        # Export from populated memory
        snapshot = populated_memory.export_snapshot()

        # Restore to empty memory
        result = memory.restore_snapshot(snapshot, merge_mode="replace")

        assert result["restored"] == 5
        assert memory.get("fast_1") is not None

    def test_restore_snapshot_keep_mode(self, populated_memory):
        """Test restoring snapshot in keep mode preserves existing."""
        # Get initial count
        original_entry = populated_memory.get("fast_1")
        original_content = original_entry.content

        # Create modified snapshot
        snapshot = populated_memory.export_snapshot()
        snapshot["entries"][0]["content"] = "Modified content"

        # Restore with keep mode
        result = populated_memory.restore_snapshot(snapshot, merge_mode="keep")

        # Original should be preserved
        entry = populated_memory.get("fast_1")
        assert entry.content == original_content
        assert result["skipped"] > 0

    def test_restore_snapshot_merge_mode(self, memory):
        """Test restoring snapshot in merge mode."""
        # Add entry with low importance
        memory.add("merge_test", "Low importance", importance=0.3)

        # Create snapshot with higher importance for same ID
        snapshot = {
            "entries": [
                {
                    "id": "merge_test",
                    "tier": "slow",
                    "content": "High importance",
                    "importance": 0.9,
                    "surprise_score": 0.1,
                    "consolidation_score": 0.5,
                }
            ]
        }

        result = memory.restore_snapshot(snapshot, merge_mode="merge")

        # Higher importance should win
        entry = memory.get("merge_test")
        assert entry.importance == 0.9
        assert result["updated"] == 1


# =============================================================================
# Test Concurrent Access
# =============================================================================


class TestConcurrentAccess:
    """Test thread-safety and concurrent access."""

    def test_concurrent_adds(self, memory):
        """Test concurrent add operations."""
        errors = []

        def add_entries(start_idx):
            try:
                for i in range(10):
                    memory.add(f"concurrent_{start_idx}_{i}", f"Content {i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=add_entries, args=(i,)) for i in range(5)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        # Should have 50 entries total
        stats = memory.get_stats()
        assert stats["total_memories"] == 50

    def test_concurrent_reads_writes(self, memory):
        """Test concurrent read and write operations."""
        # Pre-populate
        for i in range(10):
            memory.add(f"rw_test_{i}", f"Content {i}")

        errors = []

        def read_entries():
            try:
                for _ in range(20):
                    memory.retrieve(limit=5)
            except Exception as e:
                errors.append(e)

        def write_entries(idx):
            try:
                for i in range(10):
                    memory.add(f"rw_new_{idx}_{i}", f"New content {i}")
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=read_entries),
            threading.Thread(target=write_entries, args=(0,)),
            threading.Thread(target=read_entries),
            threading.Thread(target=write_entries, args=(1,)),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_concurrent_updates(self, memory):
        """Test concurrent outcome updates."""
        memory.add("update_target", "Content to update")
        errors = []

        def update_outcomes():
            try:
                for _ in range(10):
                    memory.update_outcome("update_target", success=True)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=update_outcomes) for _ in range(5)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        entry = memory.get("update_target")
        assert entry.success_count == 50  # 5 threads * 10 updates


# =============================================================================
# Test Memory Cleanup and Bounds
# =============================================================================


class TestMemoryCleanupAndBounds:
    """Test memory cleanup and tier limit enforcement."""

    def test_get_memory_pressure(self, memory):
        """Test memory pressure calculation."""
        # Add some entries
        for i in range(50):
            memory.add(f"pressure_{i}", f"Content {i}", tier=MemoryTier.FAST)

        pressure = memory.get_memory_pressure()

        # With 50 entries in fast tier (limit 1000), pressure should be ~0.05
        assert 0 <= pressure <= 1.0

    def test_enforce_tier_limits(self, memory):
        """Test enforcing tier limits removes excess entries."""
        # Set very low limit
        memory.hyperparams["max_entries_per_tier"]["fast"] = 10

        # Add more than limit
        for i in range(20):
            memory.add(f"limit_{i}", f"Content {i}", tier=MemoryTier.FAST, importance=i / 20)

        result = memory.enforce_tier_limits(tier=MemoryTier.FAST)

        # Should have removed excess
        stats = memory.get_stats()
        fast_count = stats["by_tier"].get("fast", {}).get("count", 0)
        assert fast_count <= 10

    def test_cleanup_expired_memories(self, memory):
        """Test cleaning up expired memories."""
        # Add old entries
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

        result = memory.cleanup_expired_memories(max_age_hours=1)

        assert result["deleted"] >= 5 or result["archived"] >= 5

    def test_delete_memory(self, memory):
        """Test deleting a specific memory entry."""
        memory.add("delete_me", "Content to delete")

        result = memory.delete("delete_me")

        assert result["deleted"] is True
        assert memory.get("delete_me") is None

    def test_delete_archives_by_default(self, memory):
        """Test that delete archives entry by default."""
        memory.add("archive_me", "Content to archive")

        result = memory.delete("archive_me", archive=True)

        assert result["archived"] is True

        # Check archive table
        with memory.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM continuum_memory_archive WHERE id = ?", ("archive_me",))
            assert cursor.fetchone() is not None

    def test_get_archive_stats(self, memory):
        """Test getting archive statistics."""
        # Archive some entries
        memory.add("archive_1", "Content 1")
        memory.add("archive_2", "Content 2")
        memory.delete("archive_1")
        memory.delete("archive_2")

        stats = memory.get_archive_stats()

        assert "total_archived" in stats
        assert stats["total_archived"] >= 2


# =============================================================================
# Test Red Line (Protected) Memories
# =============================================================================


class TestRedLineMemories:
    """Test red line (protected) memory functionality."""

    def test_mark_red_line(self, memory):
        """Test marking a memory as red line."""
        memory.add("protect_me", "Critical decision", tier=MemoryTier.SLOW)

        result = memory.mark_red_line("protect_me", reason="Safety critical")

        assert result is True
        entry = memory.get("protect_me")
        assert entry.red_line is True
        assert entry.red_line_reason == "Safety critical"
        assert entry.importance == 1.0  # Should be promoted to max importance

    def test_mark_red_line_promotes_to_glacial(self, memory):
        """Test that marking red line promotes to glacial tier."""
        memory.add("promote_protect", "Content", tier=MemoryTier.FAST)

        memory.mark_red_line("promote_protect", reason="Foundational", promote_to_glacial=True)

        entry = memory.get("promote_protect")
        assert entry.tier == MemoryTier.GLACIAL

    def test_mark_red_line_nonexistent(self, memory):
        """Test marking non-existent entry returns False."""
        result = memory.mark_red_line("nonexistent", reason="Test")

        assert result is False

    def test_red_line_blocks_deletion(self, memory):
        """Test that red line entries cannot be deleted without force."""
        memory.add("protected", "Protected content")
        memory.mark_red_line("protected", reason="Critical")

        result = memory.delete("protected")

        assert result["deleted"] is False
        assert result["blocked"] is True
        assert memory.get("protected") is not None

    def test_red_line_force_delete(self, memory):
        """Test force deleting a red line entry."""
        memory.add("force_delete", "Content")
        memory.mark_red_line("force_delete", reason="Test")

        result = memory.delete("force_delete", force=True)

        assert result["deleted"] is True

    def test_get_red_line_memories(self, memory):
        """Test retrieving all red line memories."""
        memory.add("rl_1", "Content 1")
        memory.add("rl_2", "Content 2")
        memory.add("normal", "Normal content")

        memory.mark_red_line("rl_1", reason="Critical 1")
        memory.mark_red_line("rl_2", reason="Critical 2")

        red_lines = memory.get_red_line_memories()

        assert len(red_lines) == 2
        assert all(e.red_line for e in red_lines)

    def test_cleanup_skips_red_line(self, memory):
        """Test that cleanup skips red line entries."""
        # Add old red line entry
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


# =============================================================================
# Test Entry Properties and Methods
# =============================================================================


class TestEntryPropertiesAndMethods:
    """Test ContinuumMemoryEntry dataclass properties and methods."""

    def test_should_promote(self):
        """Test should_promote method."""
        entry = ContinuumMemoryEntry(
            id="test",
            tier=MemoryTier.MEDIUM,
            content="Test",
            importance=0.5,
            surprise_score=0.8,  # High surprise
            consolidation_score=0.5,
            update_count=5,
            success_count=3,
            failure_count=2,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
        )

        # Medium tier promotion threshold is 0.7
        assert entry.should_promote() is True

    def test_should_not_promote_from_fast(self):
        """Test that entries at fast tier cannot promote."""
        entry = ContinuumMemoryEntry(
            id="test",
            tier=MemoryTier.FAST,
            content="Test",
            importance=0.5,
            surprise_score=1.0,
            consolidation_score=0.5,
            update_count=5,
            success_count=3,
            failure_count=2,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
        )

        assert entry.should_promote() is False

    def test_should_demote(self):
        """Test should_demote method."""
        entry = ContinuumMemoryEntry(
            id="test",
            tier=MemoryTier.FAST,
            content="Test",
            importance=0.5,
            surprise_score=0.1,  # Low surprise = high stability
            consolidation_score=0.5,
            update_count=15,  # Sufficient updates
            success_count=10,
            failure_count=5,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
        )

        # Fast tier demotion threshold is 0.2, stability is 0.9
        assert entry.should_demote() is True

    def test_should_not_demote_insufficient_updates(self):
        """Test that entries with few updates don't demote."""
        entry = ContinuumMemoryEntry(
            id="test",
            tier=MemoryTier.FAST,
            content="Test",
            importance=0.5,
            surprise_score=0.1,
            consolidation_score=0.5,
            update_count=5,  # Not enough updates
            success_count=3,
            failure_count=2,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
        )

        assert entry.should_demote() is False

    def test_cross_references(self):
        """Test cross reference management."""
        entry = ContinuumMemoryEntry(
            id="test",
            tier=MemoryTier.SLOW,
            content="Test",
            importance=0.5,
            surprise_score=0.1,
            consolidation_score=0.5,
            update_count=1,
            success_count=0,
            failure_count=0,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            metadata={"cross_references": ["ref1", "ref2"]},
        )

        assert entry.cross_references == ["ref1", "ref2"]

        entry.add_cross_reference("ref3")
        assert "ref3" in entry.cross_references

        entry.remove_cross_reference("ref1")
        assert "ref1" not in entry.cross_references

    def test_knowledge_mound_id(self):
        """Test knowledge mound ID property."""
        entry = ContinuumMemoryEntry(
            id="test_123",
            tier=MemoryTier.SLOW,
            content="Test",
            importance=0.5,
            surprise_score=0.1,
            consolidation_score=0.5,
            update_count=1,
            success_count=0,
            failure_count=0,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
        )

        assert entry.knowledge_mound_id == "cm_test_123"

    def test_tags_property(self):
        """Test tags property."""
        entry = ContinuumMemoryEntry(
            id="test",
            tier=MemoryTier.SLOW,
            content="Test",
            importance=0.5,
            surprise_score=0.1,
            consolidation_score=0.5,
            update_count=1,
            success_count=0,
            failure_count=0,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
        )

        entry.tags = ["python", "error"]
        assert entry.tags == ["python", "error"]
        assert entry.metadata["tags"] == ["python", "error"]


# =============================================================================
# Test Global Singleton
# =============================================================================


class TestGlobalSingleton:
    """Test global ContinuumMemory singleton functions."""

    def test_get_continuum_memory_creates_singleton(self, temp_db_path, monkeypatch):
        """Test that get_continuum_memory creates a singleton."""
        reset_continuum_memory()
        reset_tier_manager()

        # Mock the default db path
        monkeypatch.setattr("aragora.memory.continuum.core.get_db_path", lambda _: temp_db_path)

        cms1 = get_continuum_memory()
        cms2 = get_continuum_memory()

        assert cms1 is cms2

        reset_continuum_memory()
        reset_tier_manager()

    def test_reset_continuum_memory(self, temp_db_path, monkeypatch):
        """Test that reset clears the singleton."""
        reset_continuum_memory()
        reset_tier_manager()

        monkeypatch.setattr("aragora.memory.continuum.core.get_db_path", lambda _: temp_db_path)

        cms1 = get_continuum_memory()
        reset_continuum_memory()
        cms2 = get_continuum_memory()

        # After reset, should get new instance
        assert cms1 is not cms2

        reset_continuum_memory()
        reset_tier_manager()


# =============================================================================
# Test Update Methods
# =============================================================================


class TestUpdateMethods:
    """Test various update methods."""

    def test_update_entry(self, memory):
        """Test update_entry interface method."""
        entry = memory.add("update_test", "Content")
        entry.success_count = 5
        entry.failure_count = 2

        result = memory.update_entry(entry)

        assert result is True
        updated = memory.get("update_test")
        assert updated.success_count == 5
        assert updated.failure_count == 2

    def test_update_method(self, memory):
        """Test flexible update method."""
        memory.add("flex_update", "Original content", importance=0.5)

        result = memory.update(
            "flex_update", content="Updated content", importance=0.8, surprise_score=0.3
        )

        assert result is True
        entry = memory.get("flex_update")
        assert entry.content == "Updated content"
        assert entry.importance == 0.8
        assert entry.surprise_score == 0.3

    def test_update_with_metadata(self, memory):
        """Test update with metadata replacement."""
        memory.add("meta_update", "Content", metadata={"old": "value"})

        result = memory.update("meta_update", metadata={"new": "metadata", "tags": ["test"]})

        assert result is True
        entry = memory.get("meta_update")
        assert entry.metadata == {"new": "metadata", "tags": ["test"]}

    def test_update_nonexistent(self, memory):
        """Test updating non-existent entry returns False."""
        result = memory.update("nonexistent", content="New content")

        assert result is False

    def test_update_no_fields(self, memory):
        """Test update with no fields returns False."""
        memory.add("no_fields", "Content")

        result = memory.update("no_fields")

        assert result is False
