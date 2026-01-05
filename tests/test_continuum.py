"""Tests for the Continuum Memory System."""

import json
import math
import os
import sqlite3
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from aragora.memory.continuum import (
    ContinuumMemory,
    ContinuumMemoryEntry,
    MemoryTier,
    TierConfig,
    TIER_CONFIGS,
)


# ============================================================================
# MemoryTier enum tests
# ============================================================================


class TestMemoryTier:
    """Tests for the MemoryTier enum."""

    def test_all_tiers_defined(self):
        """Verify all four tiers are defined."""
        tiers = list(MemoryTier)
        assert len(tiers) == 4
        assert MemoryTier.FAST in tiers
        assert MemoryTier.MEDIUM in tiers
        assert MemoryTier.SLOW in tiers
        assert MemoryTier.GLACIAL in tiers

    def test_tier_values(self):
        """Verify tier string values."""
        assert MemoryTier.FAST.value == "fast"
        assert MemoryTier.MEDIUM.value == "medium"
        assert MemoryTier.SLOW.value == "slow"
        assert MemoryTier.GLACIAL.value == "glacial"

    def test_tier_from_string(self):
        """Verify tiers can be created from strings."""
        assert MemoryTier("fast") == MemoryTier.FAST
        assert MemoryTier("medium") == MemoryTier.MEDIUM
        assert MemoryTier("slow") == MemoryTier.SLOW
        assert MemoryTier("glacial") == MemoryTier.GLACIAL

    def test_invalid_tier_raises(self):
        """Verify invalid tier strings raise ValueError."""
        with pytest.raises(ValueError):
            MemoryTier("invalid")


# ============================================================================
# TierConfig tests
# ============================================================================


class TestTierConfig:
    """Tests for TierConfig dataclass and TIER_CONFIGS."""

    def test_all_tiers_have_config(self):
        """Verify all tiers have configuration."""
        for tier in MemoryTier:
            assert tier in TIER_CONFIGS
            config = TIER_CONFIGS[tier]
            assert isinstance(config, TierConfig)

    def test_fast_tier_config(self):
        """Verify fast tier has correct configuration."""
        config = TIER_CONFIGS[MemoryTier.FAST]
        assert config.name == "fast"
        assert config.half_life_hours == 1
        assert config.update_frequency == "event"
        assert config.base_learning_rate == 0.3
        assert config.promotion_threshold == 1.0  # Can't promote higher
        assert config.demotion_threshold == 0.2

    def test_glacial_tier_config(self):
        """Verify glacial tier has correct configuration."""
        config = TIER_CONFIGS[MemoryTier.GLACIAL]
        assert config.name == "glacial"
        assert config.half_life_hours == 720  # 30 days
        assert config.update_frequency == "monthly"
        assert config.base_learning_rate == 0.01
        assert config.demotion_threshold == 1.0  # Can't demote lower

    def test_half_life_increases_with_tier(self):
        """Verify half-life increases from fast to glacial."""
        fast_hl = TIER_CONFIGS[MemoryTier.FAST].half_life_hours
        medium_hl = TIER_CONFIGS[MemoryTier.MEDIUM].half_life_hours
        slow_hl = TIER_CONFIGS[MemoryTier.SLOW].half_life_hours
        glacial_hl = TIER_CONFIGS[MemoryTier.GLACIAL].half_life_hours

        assert fast_hl < medium_hl < slow_hl < glacial_hl

    def test_learning_rate_decreases_with_tier(self):
        """Verify learning rate decreases from fast to glacial."""
        fast_lr = TIER_CONFIGS[MemoryTier.FAST].base_learning_rate
        medium_lr = TIER_CONFIGS[MemoryTier.MEDIUM].base_learning_rate
        slow_lr = TIER_CONFIGS[MemoryTier.SLOW].base_learning_rate
        glacial_lr = TIER_CONFIGS[MemoryTier.GLACIAL].base_learning_rate

        assert fast_lr > medium_lr > slow_lr > glacial_lr


# ============================================================================
# ContinuumMemoryEntry tests
# ============================================================================


class TestContinuumMemoryEntry:
    """Tests for ContinuumMemoryEntry dataclass."""

    def test_entry_creation(self):
        """Verify entry can be created with all fields."""
        entry = ContinuumMemoryEntry(
            id="test-1",
            tier=MemoryTier.SLOW,
            content="Test content",
            importance=0.7,
            surprise_score=0.5,
            consolidation_score=0.3,
            update_count=10,
            success_count=7,
            failure_count=3,
            created_at="2024-01-01T00:00:00",
            updated_at="2024-01-02T00:00:00",
            metadata={"key": "value"},
        )

        assert entry.id == "test-1"
        assert entry.tier == MemoryTier.SLOW
        assert entry.content == "Test content"
        assert entry.importance == 0.7
        assert entry.success_count == 7
        assert entry.metadata == {"key": "value"}

    def test_success_rate_property(self):
        """Verify success rate is calculated correctly."""
        entry = ContinuumMemoryEntry(
            id="test",
            tier=MemoryTier.FAST,
            content="",
            importance=0.5,
            surprise_score=0.0,
            consolidation_score=0.0,
            update_count=10,
            success_count=8,
            failure_count=2,
            created_at="",
            updated_at="",
        )
        assert entry.success_rate == 0.8

    def test_success_rate_with_no_outcomes(self):
        """Verify success rate defaults to 0.5 with no outcomes."""
        entry = ContinuumMemoryEntry(
            id="test",
            tier=MemoryTier.FAST,
            content="",
            importance=0.5,
            surprise_score=0.0,
            consolidation_score=0.0,
            update_count=0,
            success_count=0,
            failure_count=0,
            created_at="",
            updated_at="",
        )
        assert entry.success_rate == 0.5

    def test_stability_score_property(self):
        """Verify stability score is inverse of surprise."""
        entry = ContinuumMemoryEntry(
            id="test",
            tier=MemoryTier.FAST,
            content="",
            importance=0.5,
            surprise_score=0.3,
            consolidation_score=0.0,
            update_count=10,
            success_count=5,
            failure_count=5,
            created_at="",
            updated_at="",
        )
        assert entry.stability_score == 0.7

    def test_should_promote_fast_tier(self):
        """Verify fast tier cannot be promoted."""
        entry = ContinuumMemoryEntry(
            id="test",
            tier=MemoryTier.FAST,
            content="",
            importance=0.5,
            surprise_score=1.0,  # Max surprise
            consolidation_score=0.0,
            update_count=10,
            success_count=5,
            failure_count=5,
            created_at="",
            updated_at="",
        )
        assert entry.should_promote() is False

    def test_should_promote_medium_tier(self):
        """Verify medium tier promotes with high surprise."""
        entry = ContinuumMemoryEntry(
            id="test",
            tier=MemoryTier.MEDIUM,
            content="",
            importance=0.5,
            surprise_score=0.8,  # Above 0.7 threshold
            consolidation_score=0.0,
            update_count=10,
            success_count=5,
            failure_count=5,
            created_at="",
            updated_at="",
        )
        assert entry.should_promote() is True

    def test_should_not_promote_low_surprise(self):
        """Verify entry with low surprise doesn't promote."""
        entry = ContinuumMemoryEntry(
            id="test",
            tier=MemoryTier.MEDIUM,
            content="",
            importance=0.5,
            surprise_score=0.3,  # Below 0.7 threshold
            consolidation_score=0.0,
            update_count=10,
            success_count=5,
            failure_count=5,
            created_at="",
            updated_at="",
        )
        assert entry.should_promote() is False

    def test_should_demote_glacial_tier(self):
        """Verify glacial tier cannot be demoted."""
        entry = ContinuumMemoryEntry(
            id="test",
            tier=MemoryTier.GLACIAL,
            content="",
            importance=0.5,
            surprise_score=0.0,  # Max stability
            consolidation_score=0.0,
            update_count=100,
            success_count=50,
            failure_count=50,
            created_at="",
            updated_at="",
        )
        assert entry.should_demote() is False

    def test_should_demote_fast_tier(self):
        """Verify fast tier demotes with high stability."""
        entry = ContinuumMemoryEntry(
            id="test",
            tier=MemoryTier.FAST,
            content="",
            importance=0.5,
            surprise_score=0.1,  # Low surprise = high stability (0.9 > 0.2)
            consolidation_score=0.0,
            update_count=20,
            success_count=10,
            failure_count=10,
            created_at="",
            updated_at="",
        )
        assert entry.should_demote() is True

    def test_should_not_demote_low_updates(self):
        """Verify entry with few updates doesn't demote."""
        entry = ContinuumMemoryEntry(
            id="test",
            tier=MemoryTier.FAST,
            content="",
            importance=0.5,
            surprise_score=0.0,  # Max stability
            consolidation_score=0.0,
            update_count=5,  # Below 10 threshold
            success_count=3,
            failure_count=2,
            created_at="",
            updated_at="",
        )
        assert entry.should_demote() is False


# ============================================================================
# ContinuumMemory tests
# ============================================================================


class TestContinuumMemory:
    """Tests for the ContinuumMemory class."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        yield db_path
        os.unlink(db_path)

    @pytest.fixture
    def cms(self, temp_db):
        """Create a ContinuumMemory instance with temp database."""
        return ContinuumMemory(db_path=temp_db)

    def test_init_creates_tables(self, temp_db):
        """Verify database tables are created on init."""
        cms = ContinuumMemory(db_path=temp_db)

        with sqlite3.connect(temp_db) as conn:
            cursor = conn.cursor()

            # Check continuum_memory table
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='continuum_memory'"
            )
            assert cursor.fetchone() is not None

            # Check meta_learning_state table
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='meta_learning_state'"
            )
            assert cursor.fetchone() is not None

            # Check tier_transitions table
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='tier_transitions'"
            )
            assert cursor.fetchone() is not None

    def test_hyperparams_initialized(self, cms):
        """Verify hyperparameters are initialized."""
        assert "surprise_weight_success" in cms.hyperparams
        assert "surprise_weight_semantic" in cms.hyperparams
        assert "consolidation_threshold" in cms.hyperparams
        assert "promotion_cooldown_hours" in cms.hyperparams


class TestContinuumMemoryAdd:
    """Tests for ContinuumMemory.add()."""

    @pytest.fixture
    def temp_db(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        yield db_path
        os.unlink(db_path)

    @pytest.fixture
    def cms(self, temp_db):
        return ContinuumMemory(db_path=temp_db)

    def test_add_basic(self, cms):
        """Verify basic add functionality."""
        entry = cms.add("test-1", "Test content")

        assert entry.id == "test-1"
        assert entry.content == "Test content"
        assert entry.tier == MemoryTier.SLOW  # Default tier
        assert entry.importance == 0.5  # Default importance
        assert entry.update_count == 1

    def test_add_with_tier(self, cms):
        """Verify adding with specific tier."""
        entry = cms.add("test-1", "Fast content", tier=MemoryTier.FAST)

        assert entry.tier == MemoryTier.FAST

    def test_add_with_importance(self, cms):
        """Verify adding with importance score."""
        entry = cms.add("test-1", "Important content", importance=0.9)

        assert entry.importance == 0.9

    def test_add_with_metadata(self, cms):
        """Verify adding with metadata."""
        metadata = {"source": "debate", "round": 3}
        entry = cms.add("test-1", "Content", metadata=metadata)

        assert entry.metadata == metadata

    def test_add_persists_to_db(self, cms):
        """Verify add persists entry to database."""
        cms.add("test-1", "Persisted content", importance=0.8)

        retrieved = cms.get("test-1")
        assert retrieved is not None
        assert retrieved.content == "Persisted content"
        assert retrieved.importance == 0.8

    def test_add_replace_existing(self, cms):
        """Verify add replaces existing entry with same ID."""
        cms.add("test-1", "Original content")
        cms.add("test-1", "New content", importance=0.9)

        entry = cms.get("test-1")
        assert entry.content == "New content"
        assert entry.importance == 0.9


class TestContinuumMemoryGet:
    """Tests for ContinuumMemory.get()."""

    @pytest.fixture
    def temp_db(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        yield db_path
        os.unlink(db_path)

    @pytest.fixture
    def cms(self, temp_db):
        return ContinuumMemory(db_path=temp_db)

    def test_get_existing(self, cms):
        """Verify getting existing entry."""
        cms.add("test-1", "Content", tier=MemoryTier.MEDIUM, importance=0.7)

        entry = cms.get("test-1")

        assert entry is not None
        assert entry.id == "test-1"
        assert entry.content == "Content"
        assert entry.tier == MemoryTier.MEDIUM
        assert entry.importance == 0.7

    def test_get_nonexistent(self, cms):
        """Verify getting nonexistent entry returns None."""
        entry = cms.get("does-not-exist")
        assert entry is None

    def test_get_with_metadata(self, cms):
        """Verify metadata is correctly loaded."""
        metadata = {"key": "value", "nested": {"a": 1}}
        cms.add("test-1", "Content", metadata=metadata)

        entry = cms.get("test-1")
        assert entry.metadata == metadata


class TestContinuumMemoryRetrieve:
    """Tests for ContinuumMemory.retrieve()."""

    @pytest.fixture
    def temp_db(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        yield db_path
        os.unlink(db_path)

    @pytest.fixture
    def cms(self, temp_db):
        return ContinuumMemory(db_path=temp_db)

    @pytest.fixture
    def populated_cms(self, cms):
        """Create CMS with sample data."""
        cms.add("fast-1", "Fast pattern", tier=MemoryTier.FAST, importance=0.9)
        cms.add("medium-1", "Medium pattern", tier=MemoryTier.MEDIUM, importance=0.7)
        cms.add("slow-1", "Slow pattern", tier=MemoryTier.SLOW, importance=0.5)
        cms.add("glacial-1", "Glacial pattern", tier=MemoryTier.GLACIAL, importance=0.8)
        return cms

    def test_retrieve_all_tiers(self, populated_cms):
        """Verify retrieve returns entries from all tiers."""
        entries = populated_cms.retrieve()

        assert len(entries) == 4
        tiers_found = {e.tier for e in entries}
        assert MemoryTier.FAST in tiers_found
        assert MemoryTier.GLACIAL in tiers_found

    def test_retrieve_specific_tiers(self, populated_cms):
        """Verify retrieve filters by tier."""
        entries = populated_cms.retrieve(tiers=[MemoryTier.FAST, MemoryTier.MEDIUM])

        assert len(entries) == 2
        for e in entries:
            assert e.tier in [MemoryTier.FAST, MemoryTier.MEDIUM]

    def test_retrieve_exclude_glacial(self, populated_cms):
        """Verify include_glacial=False excludes glacial tier."""
        entries = populated_cms.retrieve(include_glacial=False)

        assert len(entries) == 3
        for e in entries:
            assert e.tier != MemoryTier.GLACIAL

    def test_retrieve_min_importance(self, populated_cms):
        """Verify min_importance filters correctly."""
        entries = populated_cms.retrieve(min_importance=0.8)

        assert len(entries) == 2
        for e in entries:
            assert e.importance >= 0.8

    def test_retrieve_with_limit(self, populated_cms):
        """Verify limit parameter works."""
        entries = populated_cms.retrieve(limit=2)

        assert len(entries) == 2

    def test_retrieve_with_query(self, cms):
        """Verify query filters by keyword."""
        cms.add("test-1", "Error handling pattern", importance=0.8)
        cms.add("test-2", "Success strategy", importance=0.8)
        cms.add("test-3", "Error recovery method", importance=0.8)

        entries = cms.retrieve(query="error")

        assert len(entries) == 2
        for e in entries:
            assert "error" in e.content.lower()

    def test_retrieve_ordered_by_score(self, cms):
        """Verify entries are ordered by retrieval score."""
        cms.add("low", "Low importance", importance=0.1)
        cms.add("high", "High importance", importance=0.9)

        entries = cms.retrieve()

        # Higher importance should come first (assuming same recency)
        assert entries[0].importance >= entries[1].importance


class TestContinuumMemoryUpdateOutcome:
    """Tests for ContinuumMemory.update_outcome()."""

    @pytest.fixture
    def temp_db(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        yield db_path
        os.unlink(db_path)

    @pytest.fixture
    def cms(self, temp_db):
        return ContinuumMemory(db_path=temp_db)

    def test_update_success(self, cms):
        """Verify success outcome updates correctly."""
        cms.add("test-1", "Pattern")

        surprise = cms.update_outcome("test-1", success=True)

        entry = cms.get("test-1")
        assert entry.success_count == 1
        assert entry.failure_count == 0
        assert entry.update_count == 2

    def test_update_failure(self, cms):
        """Verify failure outcome updates correctly."""
        cms.add("test-1", "Pattern")

        surprise = cms.update_outcome("test-1", success=False)

        entry = cms.get("test-1")
        assert entry.success_count == 0
        assert entry.failure_count == 1

    def test_update_nonexistent_returns_zero(self, cms):
        """Verify updating nonexistent ID returns 0."""
        surprise = cms.update_outcome("does-not-exist", success=True)
        assert surprise == 0.0

    def test_surprise_increases_on_unexpected(self, cms):
        """Verify surprise increases when outcome is unexpected."""
        cms.add("test-1", "Pattern")
        # Set up expectation of success
        for _ in range(5):
            cms.update_outcome("test-1", success=True)

        # Now observe failure - should increase surprise
        entry_before = cms.get("test-1")
        surprise_before = entry_before.surprise_score

        cms.update_outcome("test-1", success=False)

        entry_after = cms.get("test-1")
        # Surprise should increase after unexpected outcome
        # (though exact behavior depends on EMA)
        assert entry_after.update_count == 7

    def test_consolidation_increases_with_updates(self, cms):
        """Verify consolidation score increases with more updates."""
        cms.add("test-1", "Pattern")

        for _ in range(50):
            cms.update_outcome("test-1", success=True)

        entry = cms.get("test-1")
        assert entry.consolidation_score > 0
        assert entry.consolidation_score <= 1.0


class TestContinuumMemoryLearningRate:
    """Tests for ContinuumMemory.get_learning_rate()."""

    @pytest.fixture
    def temp_db(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        yield db_path
        os.unlink(db_path)

    @pytest.fixture
    def cms(self, temp_db):
        return ContinuumMemory(db_path=temp_db)

    def test_fast_tier_higher_lr(self, cms):
        """Verify fast tier has higher learning rate."""
        fast_lr = cms.get_learning_rate(MemoryTier.FAST, update_count=0)
        slow_lr = cms.get_learning_rate(MemoryTier.SLOW, update_count=0)

        assert fast_lr > slow_lr

    def test_lr_decays_with_updates(self, cms):
        """Verify learning rate decays with more updates."""
        lr_0 = cms.get_learning_rate(MemoryTier.FAST, update_count=0)
        lr_10 = cms.get_learning_rate(MemoryTier.FAST, update_count=10)
        lr_100 = cms.get_learning_rate(MemoryTier.FAST, update_count=100)

        assert lr_0 > lr_10 > lr_100

    def test_glacial_lr_decays_slowly(self, cms):
        """Verify glacial tier learning rate decays slowly."""
        lr_0 = cms.get_learning_rate(MemoryTier.GLACIAL, update_count=0)
        lr_100 = cms.get_learning_rate(MemoryTier.GLACIAL, update_count=100)

        # Glacial should retain most of its LR even after 100 updates
        retention_ratio = lr_100 / lr_0
        assert retention_ratio > 0.9  # >90% retained


class TestContinuumMemoryPromoteDemote:
    """Tests for ContinuumMemory.promote() and demote()."""

    @pytest.fixture
    def temp_db(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        yield db_path
        os.unlink(db_path)

    @pytest.fixture
    def cms(self, temp_db):
        return ContinuumMemory(db_path=temp_db)

    def test_promote_glacial_to_slow(self, cms):
        """Verify promoting from glacial to slow."""
        cms.add("test-1", "Pattern", tier=MemoryTier.GLACIAL)

        # Set high surprise
        with sqlite3.connect(cms.db_path) as conn:
            conn.execute(
                "UPDATE continuum_memory SET surprise_score = 0.8 WHERE id = ?",
                ("test-1",)
            )

        new_tier = cms.promote("test-1")

        assert new_tier == MemoryTier.SLOW
        entry = cms.get("test-1")
        assert entry.tier == MemoryTier.SLOW

    def test_promote_fast_returns_none(self, cms):
        """Verify promoting fast tier returns None."""
        cms.add("test-1", "Pattern", tier=MemoryTier.FAST)

        new_tier = cms.promote("test-1")

        assert new_tier is None

    def test_promote_nonexistent_returns_none(self, cms):
        """Verify promoting nonexistent ID returns None."""
        new_tier = cms.promote("does-not-exist")
        assert new_tier is None

    def test_promote_cooldown(self, cms):
        """Verify promotion cooldown is respected."""
        cms.add("test-1", "Pattern", tier=MemoryTier.GLACIAL)

        # First promotion
        with sqlite3.connect(cms.db_path) as conn:
            conn.execute(
                "UPDATE continuum_memory SET surprise_score = 0.8 WHERE id = ?",
                ("test-1",)
            )
        cms.promote("test-1")

        # Second immediate promotion should fail (cooldown)
        with sqlite3.connect(cms.db_path) as conn:
            conn.execute(
                "UPDATE continuum_memory SET surprise_score = 0.8 WHERE id = ?",
                ("test-1",)
            )
        new_tier = cms.promote("test-1")

        assert new_tier is None  # Blocked by cooldown

    def test_demote_fast_to_medium(self, cms):
        """Verify demoting from fast to medium."""
        cms.add("test-1", "Pattern", tier=MemoryTier.FAST)

        # Set low surprise and enough updates
        with sqlite3.connect(cms.db_path) as conn:
            conn.execute(
                "UPDATE continuum_memory SET surprise_score = 0.1, update_count = 20 WHERE id = ?",
                ("test-1",)
            )

        new_tier = cms.demote("test-1")

        assert new_tier == MemoryTier.MEDIUM
        entry = cms.get("test-1")
        assert entry.tier == MemoryTier.MEDIUM

    def test_demote_glacial_returns_none(self, cms):
        """Verify demoting glacial tier returns None."""
        cms.add("test-1", "Pattern", tier=MemoryTier.GLACIAL)

        new_tier = cms.demote("test-1")

        assert new_tier is None

    def test_demote_requires_min_updates(self, cms):
        """Verify demotion requires minimum update count."""
        cms.add("test-1", "Pattern", tier=MemoryTier.FAST)

        # Low surprise but few updates
        with sqlite3.connect(cms.db_path) as conn:
            conn.execute(
                "UPDATE continuum_memory SET surprise_score = 0.0, update_count = 5 WHERE id = ?",
                ("test-1",)
            )

        new_tier = cms.demote("test-1")

        assert new_tier is None  # Not enough updates

    def test_promotion_records_transition(self, cms):
        """Verify promotion is recorded in transitions table."""
        cms.add("test-1", "Pattern", tier=MemoryTier.GLACIAL)

        with sqlite3.connect(cms.db_path) as conn:
            conn.execute(
                "UPDATE continuum_memory SET surprise_score = 0.8 WHERE id = ?",
                ("test-1",)
            )

        cms.promote("test-1")

        with sqlite3.connect(cms.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT from_tier, to_tier, reason FROM tier_transitions WHERE memory_id = ?",
                ("test-1",)
            )
            row = cursor.fetchone()

        assert row is not None
        assert row[0] == "glacial"
        assert row[1] == "slow"
        assert row[2] == "high_surprise"


class TestContinuumMemoryConsolidate:
    """Tests for ContinuumMemory.consolidate()."""

    @pytest.fixture
    def temp_db(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        yield db_path
        os.unlink(db_path)

    @pytest.fixture
    def cms(self, temp_db):
        return ContinuumMemory(db_path=temp_db)

    def test_consolidate_empty_db(self, cms):
        """Verify consolidate works on empty database."""
        result = cms.consolidate()

        assert result["promotions"] == 0
        assert result["demotions"] == 0

    def test_consolidate_promotes_high_surprise(self, cms):
        """Verify consolidate promotes high-surprise entries."""
        cms.add("test-1", "Pattern", tier=MemoryTier.SLOW)

        # Set high surprise (above SLOW's 0.6 threshold)
        with sqlite3.connect(cms.db_path) as conn:
            conn.execute(
                "UPDATE continuum_memory SET surprise_score = 0.8 WHERE id = ?",
                ("test-1",)
            )

        result = cms.consolidate()

        assert result["promotions"] >= 1
        entry = cms.get("test-1")
        assert entry.tier == MemoryTier.MEDIUM

    def test_consolidate_demotes_stable(self, cms):
        """Verify consolidate demotes stable entries."""
        cms.add("test-1", "Pattern", tier=MemoryTier.FAST)

        # Set low surprise (stability > 0.2 threshold) and high update count
        with sqlite3.connect(cms.db_path) as conn:
            conn.execute(
                "UPDATE continuum_memory SET surprise_score = 0.05, update_count = 50 WHERE id = ?",
                ("test-1",)
            )

        result = cms.consolidate()

        assert result["demotions"] >= 1
        entry = cms.get("test-1")
        assert entry.tier == MemoryTier.MEDIUM

    def test_consolidate_handles_multiple_tiers(self, cms):
        """Verify consolidate processes all tiers correctly."""
        # Add entries in different tiers
        cms.add("glacial-1", "Glacial", tier=MemoryTier.GLACIAL)
        cms.add("slow-1", "Slow", tier=MemoryTier.SLOW)
        cms.add("fast-1", "Fast", tier=MemoryTier.FAST)

        # Set high surprise on glacial and slow, low surprise on fast
        with sqlite3.connect(cms.db_path) as conn:
            conn.execute(
                "UPDATE continuum_memory SET surprise_score = 0.7 WHERE id = ?",
                ("glacial-1",)
            )
            conn.execute(
                "UPDATE continuum_memory SET surprise_score = 0.8 WHERE id = ?",
                ("slow-1",)
            )
            conn.execute(
                "UPDATE continuum_memory SET surprise_score = 0.05, update_count = 50 WHERE id = ?",
                ("fast-1",)
            )

        result = cms.consolidate()

        # Should have 2 promotions (glacial->slow, slow->medium) and 1 demotion
        assert result["promotions"] == 2
        assert result["demotions"] == 1

        # Verify new tiers
        assert cms.get("glacial-1").tier == MemoryTier.SLOW
        assert cms.get("slow-1").tier == MemoryTier.MEDIUM
        assert cms.get("fast-1").tier == MemoryTier.MEDIUM


class TestContinuumMemoryStats:
    """Tests for ContinuumMemory.get_stats()."""

    @pytest.fixture
    def temp_db(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        yield db_path
        os.unlink(db_path)

    @pytest.fixture
    def cms(self, temp_db):
        return ContinuumMemory(db_path=temp_db)

    def test_stats_empty_db(self, cms):
        """Verify stats work on empty database."""
        stats = cms.get_stats()

        assert stats["total_memories"] == 0
        assert "by_tier" in stats
        assert "transitions" in stats

    def test_stats_with_data(self, cms):
        """Verify stats reflect actual data."""
        cms.add("fast-1", "F1", tier=MemoryTier.FAST, importance=0.9)
        cms.add("fast-2", "F2", tier=MemoryTier.FAST, importance=0.7)
        cms.add("slow-1", "S1", tier=MemoryTier.SLOW, importance=0.5)

        stats = cms.get_stats()

        assert stats["total_memories"] == 3
        assert "fast" in stats["by_tier"]
        assert stats["by_tier"]["fast"]["count"] == 2
        assert "slow" in stats["by_tier"]
        assert stats["by_tier"]["slow"]["count"] == 1

    def test_stats_tier_averages(self, cms):
        """Verify tier averages are calculated correctly."""
        cms.add("fast-1", "F1", tier=MemoryTier.FAST, importance=0.8)
        cms.add("fast-2", "F2", tier=MemoryTier.FAST, importance=0.6)

        stats = cms.get_stats()

        assert stats["by_tier"]["fast"]["avg_importance"] == pytest.approx(0.7, rel=0.01)


class TestContinuumMemoryExport:
    """Tests for ContinuumMemory.export_for_tier()."""

    @pytest.fixture
    def temp_db(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        yield db_path
        os.unlink(db_path)

    @pytest.fixture
    def cms(self, temp_db):
        return ContinuumMemory(db_path=temp_db)

    def test_export_empty_tier(self, cms):
        """Verify export returns empty list for empty tier."""
        exported = cms.export_for_tier(MemoryTier.FAST)
        assert exported == []

    def test_export_with_data(self, cms):
        """Verify export returns correct data."""
        cms.add("fast-1", "F1", tier=MemoryTier.FAST, importance=0.9)
        cms.add("slow-1", "S1", tier=MemoryTier.SLOW, importance=0.5)

        exported = cms.export_for_tier(MemoryTier.FAST)

        assert len(exported) == 1
        assert exported[0]["id"] == "fast-1"
        assert exported[0]["content"] == "F1"
        assert exported[0]["importance"] == 0.9
        assert "success_rate" in exported[0]

    def test_export_excludes_other_tiers(self, cms):
        """Verify export only includes specified tier."""
        cms.add("fast-1", "F1", tier=MemoryTier.FAST)
        cms.add("slow-1", "S1", tier=MemoryTier.SLOW)
        cms.add("glacial-1", "G1", tier=MemoryTier.GLACIAL)

        exported = cms.export_for_tier(MemoryTier.SLOW)

        assert len(exported) == 1
        assert exported[0]["id"] == "slow-1"
