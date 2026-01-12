"""
Integration tests for ContinuumMemory tier system.

Tests verify that the memory tier system correctly:
- Manages tier configurations and transitions (GLACIAL → SLOW → MEDIUM → FAST)
- Promotes/demotes memories based on surprise and stability scores
- Handles tier-specific TTLs and retention policies
- Provides tier statistics and metrics
"""

import tempfile
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aragora.memory.tier_manager import (
    TierManager,
    TierConfig,
    MemoryTier,
    TIER_ORDER,
    DEFAULT_TIER_CONFIGS,
    TierTransitionMetrics,
)
from aragora.memory.continuum import ContinuumMemory, ContinuumMemoryEntry


class TestMemoryTierEnum:
    """Test MemoryTier enum values and ordering."""

    def test_tier_ordering(self):
        """Tiers should be ordered from slowest to fastest."""
        assert TIER_ORDER == [
            MemoryTier.GLACIAL,
            MemoryTier.SLOW,
            MemoryTier.MEDIUM,
            MemoryTier.FAST,
        ]

    def test_tier_values(self):
        """Tier enum values should match expected strings."""
        assert MemoryTier.FAST.value == "fast"
        assert MemoryTier.MEDIUM.value == "medium"
        assert MemoryTier.SLOW.value == "slow"
        assert MemoryTier.GLACIAL.value == "glacial"


class TestContinuumMemoryEntry:
    """Test ContinuumMemoryEntry dataclass."""

    def test_entry_initialization(self):
        """Entry should initialize with all required fields."""
        now = datetime.now().isoformat()
        entry = ContinuumMemoryEntry(
            id="mem-123",
            tier=MemoryTier.MEDIUM,
            content="Test memory content",
            importance=0.75,
            surprise_score=0.5,
            consolidation_score=0.3,
            update_count=5,
            success_count=3,
            failure_count=2,
            created_at=now,
            updated_at=now,
        )

        assert entry.id == "mem-123"
        assert entry.tier == MemoryTier.MEDIUM
        assert entry.importance == 0.75
        assert entry.surprise_score == 0.5

    def test_success_rate_calculation(self):
        """Success rate should be calculated correctly."""
        now = datetime.now().isoformat()
        entry = ContinuumMemoryEntry(
            id="mem-test",
            tier=MemoryTier.SLOW,
            content="Test",
            importance=0.5,
            surprise_score=0.3,
            consolidation_score=0.5,
            update_count=10,
            success_count=7,
            failure_count=3,
            created_at=now,
            updated_at=now,
        )

        assert entry.success_rate == 0.7

    def test_success_rate_zero_division(self):
        """Success rate should handle zero total gracefully."""
        now = datetime.now().isoformat()
        entry = ContinuumMemoryEntry(
            id="mem-test",
            tier=MemoryTier.SLOW,
            content="Test",
            importance=0.5,
            surprise_score=0.3,
            consolidation_score=0.5,
            update_count=0,
            success_count=0,
            failure_count=0,
            created_at=now,
            updated_at=now,
        )

        # Should return 0.5 as default
        assert entry.success_rate == 0.5

    def test_stability_score(self):
        """Stability score should be inverse of surprise."""
        now = datetime.now().isoformat()
        entry = ContinuumMemoryEntry(
            id="mem-test",
            tier=MemoryTier.MEDIUM,
            content="Test",
            importance=0.5,
            surprise_score=0.3,
            consolidation_score=0.5,
            update_count=5,
            success_count=3,
            failure_count=2,
            created_at=now,
            updated_at=now,
        )

        assert entry.stability_score == 0.7


class TestTierManager:
    """Test TierManager configuration and decisions."""

    def test_tier_config_defaults(self):
        """TierManager should have sensible defaults."""
        manager = TierManager()

        fast_config = manager.get_config(MemoryTier.FAST)
        assert fast_config.half_life_hours == 1
        assert fast_config.update_frequency == "event"

        glacial_config = manager.get_config(MemoryTier.GLACIAL)
        assert glacial_config.half_life_hours == 720  # 30 days
        assert glacial_config.update_frequency == "monthly"

    def test_should_promote_high_surprise(self):
        """High surprise should trigger promotion."""
        manager = TierManager()

        # High surprise in SLOW tier should promote
        should_promote = manager.should_promote(
            tier=MemoryTier.SLOW,
            surprise_score=0.7,  # Above SLOW's promotion_threshold of 0.6
        )
        assert should_promote is True

    def test_should_not_promote_fast_tier(self):
        """FAST tier cannot be promoted further."""
        manager = TierManager()

        should_promote = manager.should_promote(
            tier=MemoryTier.FAST,
            surprise_score=0.99,
        )
        assert should_promote is False

    def test_should_demote_stable_pattern(self):
        """Stable patterns should be demoted."""
        manager = TierManager()

        # Stability = 1 - surprise, so surprise=0.2 means stability=0.8
        # FAST tier demotion_threshold is 0.2, so stability > 0.8 should demote
        should_demote = manager.should_demote(
            tier=MemoryTier.FAST,
            surprise_score=0.1,  # Stability = 0.9 > 0.8
            update_count=15,
        )
        assert should_demote is True

    def test_should_not_demote_glacial_tier(self):
        """GLACIAL tier cannot be demoted further."""
        manager = TierManager()

        should_demote = manager.should_demote(
            tier=MemoryTier.GLACIAL,
            surprise_score=0.0,
            update_count=100,
        )
        assert should_demote is False

    def test_get_next_tier_faster(self):
        """Should get correct next tier for promotion."""
        manager = TierManager()

        assert manager.get_next_tier(MemoryTier.GLACIAL, "faster") == MemoryTier.SLOW
        assert manager.get_next_tier(MemoryTier.SLOW, "faster") == MemoryTier.MEDIUM
        assert manager.get_next_tier(MemoryTier.MEDIUM, "faster") == MemoryTier.FAST
        assert manager.get_next_tier(MemoryTier.FAST, "faster") is None

    def test_get_next_tier_slower(self):
        """Should get correct next tier for demotion."""
        manager = TierManager()

        assert manager.get_next_tier(MemoryTier.FAST, "slower") == MemoryTier.MEDIUM
        assert manager.get_next_tier(MemoryTier.MEDIUM, "slower") == MemoryTier.SLOW
        assert manager.get_next_tier(MemoryTier.SLOW, "slower") == MemoryTier.GLACIAL
        assert manager.get_next_tier(MemoryTier.GLACIAL, "slower") is None

    def test_promotion_cooldown(self):
        """Should respect promotion cooldown."""
        manager = TierManager(promotion_cooldown_hours=24.0)

        # Recent promotion should block
        recent_promotion = datetime.now().isoformat()
        should_promote = manager.should_promote(
            tier=MemoryTier.SLOW,
            surprise_score=0.9,
            last_promotion_at=recent_promotion,
        )
        assert should_promote is False

        # Old promotion should allow
        old_promotion = (datetime.now() - timedelta(hours=48)).isoformat()
        should_promote = manager.should_promote(
            tier=MemoryTier.SLOW,
            surprise_score=0.9,
            last_promotion_at=old_promotion,
        )
        assert should_promote is True


class TestTierTransitionMetrics:
    """Test tier transition metrics tracking."""

    def test_record_promotion(self):
        """Should record promotion events."""
        metrics = TierTransitionMetrics()

        metrics.record_promotion(MemoryTier.SLOW, MemoryTier.MEDIUM)
        metrics.record_promotion(MemoryTier.SLOW, MemoryTier.MEDIUM)
        metrics.record_promotion(MemoryTier.MEDIUM, MemoryTier.FAST)

        data = metrics.to_dict()
        assert data["total_promotions"] == 3
        assert data["promotions"]["slow->medium"] == 2
        assert data["promotions"]["medium->fast"] == 1

    def test_record_demotion(self):
        """Should record demotion events."""
        metrics = TierTransitionMetrics()

        metrics.record_demotion(MemoryTier.FAST, MemoryTier.MEDIUM)
        metrics.record_demotion(MemoryTier.MEDIUM, MemoryTier.SLOW)

        data = metrics.to_dict()
        assert data["total_demotions"] == 2
        assert data["demotions"]["fast->medium"] == 1
        assert data["demotions"]["medium->slow"] == 1

    def test_reset_metrics(self):
        """Should reset metrics correctly."""
        metrics = TierTransitionMetrics()

        metrics.record_promotion(MemoryTier.SLOW, MemoryTier.MEDIUM)
        metrics.record_demotion(MemoryTier.FAST, MemoryTier.MEDIUM)

        metrics.reset()

        data = metrics.to_dict()
        assert data["total_promotions"] == 0
        assert data["total_demotions"] == 0
        assert len(data["promotions"]) == 0


class TestTierConfig:
    """Test TierConfig dataclass."""

    def test_half_life_seconds(self):
        """Should convert half-life hours to seconds."""
        config = TierConfig(
            name="test",
            half_life_hours=24,
            update_frequency="daily",
            base_learning_rate=0.1,
            decay_rate=0.99,
            promotion_threshold=0.7,
            demotion_threshold=0.3,
        )

        assert config.half_life_seconds == 24 * 3600

    def test_default_tier_configs(self):
        """Default configs should have sensible values."""
        for tier, config in DEFAULT_TIER_CONFIGS.items():
            assert config.half_life_hours > 0
            assert 0 <= config.promotion_threshold <= 1
            assert 0 <= config.demotion_threshold <= 1
            assert config.base_learning_rate > 0


class TestContinuumMemoryIntegration:
    """Integration tests for ContinuumMemory with TierManager."""

    def test_memory_initialization(self):
        """ContinuumMemory should initialize with tier manager."""
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "test_memory.db")
            memory = ContinuumMemory(db_path=db_path)

            # Should have tier manager
            assert memory.tier_manager is not None

    def test_tier_manager_injection(self):
        """Should accept custom tier manager."""
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "test_memory.db")
            custom_manager = TierManager(promotion_cooldown_hours=1.0)

            memory = ContinuumMemory(db_path=db_path, tier_manager=custom_manager)

            assert memory.tier_manager is custom_manager

    def test_get_tier_metrics(self):
        """Should return tier metrics from manager."""
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "test_memory.db")
            memory = ContinuumMemory(db_path=db_path)

            metrics = memory.get_tier_metrics()
            assert "total_promotions" in metrics
            assert "total_demotions" in metrics


class TestConcurrentTierAccess:
    """Test thread-safety of tier operations."""

    def test_concurrent_metrics_update(self):
        """Metrics should be thread-safe."""
        metrics = TierTransitionMetrics()
        num_threads = 10
        updates_per_thread = 100

        def record_updates():
            for _ in range(updates_per_thread):
                metrics.record_promotion(MemoryTier.SLOW, MemoryTier.MEDIUM)
                metrics.record_demotion(MemoryTier.FAST, MemoryTier.MEDIUM)

        threads = [threading.Thread(target=record_updates) for _ in range(num_threads)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        data = metrics.to_dict()
        assert data["total_promotions"] == num_threads * updates_per_thread
        assert data["total_demotions"] == num_threads * updates_per_thread


class TestEntryPromotionDemotion:
    """Test entry-level promotion/demotion decisions."""

    def test_entry_should_promote(self):
        """Entry with high surprise should want promotion."""
        now = datetime.now().isoformat()
        entry = ContinuumMemoryEntry(
            id="mem-promote",
            tier=MemoryTier.SLOW,
            content="Surprising pattern",
            importance=0.7,
            surprise_score=0.8,  # High surprise
            consolidation_score=0.2,
            update_count=5,
            success_count=4,
            failure_count=1,
            created_at=now,
            updated_at=now,
        )

        assert entry.should_promote() is True

    def test_entry_should_demote(self):
        """Entry with low surprise (high stability) should want demotion."""
        now = datetime.now().isoformat()
        entry = ContinuumMemoryEntry(
            id="mem-demote",
            tier=MemoryTier.FAST,
            content="Stable pattern",
            importance=0.5,
            surprise_score=0.1,  # Low surprise = high stability
            consolidation_score=0.9,
            update_count=50,  # Many updates
            success_count=45,
            failure_count=5,
            created_at=now,
            updated_at=now,
        )

        assert entry.should_demote() is True

    def test_entry_at_boundary_no_promote(self):
        """FAST tier entry should not promote."""
        now = datetime.now().isoformat()
        entry = ContinuumMemoryEntry(
            id="mem-fast",
            tier=MemoryTier.FAST,
            content="Already fast",
            importance=0.9,
            surprise_score=0.99,
            consolidation_score=0.1,
            update_count=1,
            success_count=1,
            failure_count=0,
            created_at=now,
            updated_at=now,
        )

        assert entry.should_promote() is False

    def test_entry_at_boundary_no_demote(self):
        """GLACIAL tier entry should not demote."""
        now = datetime.now().isoformat()
        entry = ContinuumMemoryEntry(
            id="mem-glacial",
            tier=MemoryTier.GLACIAL,
            content="Already glacial",
            importance=0.1,
            surprise_score=0.01,
            consolidation_score=0.99,
            update_count=1000,
            success_count=999,
            failure_count=1,
            created_at=now,
            updated_at=now,
        )

        assert entry.should_demote() is False


class TestCustomTierConfigs:
    """Test custom tier configuration."""

    def test_custom_config_application(self):
        """Should apply custom tier configs."""
        custom_configs = {
            MemoryTier.FAST: TierConfig(
                name="custom_fast",
                half_life_hours=0.5,  # 30 minutes
                update_frequency="event",
                base_learning_rate=0.5,
                decay_rate=0.9,
                promotion_threshold=1.0,
                demotion_threshold=0.1,
            ),
            MemoryTier.MEDIUM: DEFAULT_TIER_CONFIGS[MemoryTier.MEDIUM],
            MemoryTier.SLOW: DEFAULT_TIER_CONFIGS[MemoryTier.SLOW],
            MemoryTier.GLACIAL: DEFAULT_TIER_CONFIGS[MemoryTier.GLACIAL],
        }

        manager = TierManager(configs=custom_configs)

        fast_config = manager.get_config(MemoryTier.FAST)
        assert fast_config.half_life_hours == 0.5
        assert fast_config.base_learning_rate == 0.5

    def test_update_config_at_runtime(self):
        """Should update tier config at runtime."""
        manager = TierManager()

        new_config = TierConfig(
            name="updated_slow",
            half_life_hours=96,  # 4 days
            update_frequency="weekly",
            base_learning_rate=0.05,
            decay_rate=0.995,
            promotion_threshold=0.5,
            demotion_threshold=0.5,
        )

        manager.update_config(MemoryTier.SLOW, new_config)

        updated = manager.get_config(MemoryTier.SLOW)
        assert updated.half_life_hours == 96
