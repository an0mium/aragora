"""Tests for memory tier management."""

import pytest
from datetime import datetime
from unittest.mock import MagicMock

from aragora.memory.tier_manager import (
    MemoryTier,
    TierConfig,
    TierTransitionMetrics,
    TIER_ORDER,
    DEFAULT_TIER_CONFIGS,
)


class TestMemoryTier:
    """Test MemoryTier enum."""

    def test_all_tiers_defined(self):
        """Test all expected tiers exist."""
        expected = ["FAST", "MEDIUM", "SLOW", "GLACIAL"]
        for tier in expected:
            assert hasattr(MemoryTier, tier)

    def test_tier_values(self):
        """Test tier values."""
        assert MemoryTier.FAST.value == "fast"
        assert MemoryTier.MEDIUM.value == "medium"
        assert MemoryTier.SLOW.value == "slow"
        assert MemoryTier.GLACIAL.value == "glacial"


class TestTierOrder:
    """Test tier ordering."""

    def test_tier_order_length(self):
        """Test all tiers are in order list."""
        assert len(TIER_ORDER) == 4

    def test_tier_order_slowest_first(self):
        """Test glacial is first (slowest)."""
        assert TIER_ORDER[0] == MemoryTier.GLACIAL

    def test_tier_order_fastest_last(self):
        """Test fast is last (fastest)."""
        assert TIER_ORDER[-1] == MemoryTier.FAST

    def test_tier_order_sequence(self):
        """Test correct sequence."""
        expected = [
            MemoryTier.GLACIAL,
            MemoryTier.SLOW,
            MemoryTier.MEDIUM,
            MemoryTier.FAST,
        ]
        assert TIER_ORDER == expected


class TestTierConfig:
    """Test TierConfig dataclass."""

    def test_create_config(self):
        """Test creating a tier config."""
        config = TierConfig(
            name="fast",
            half_life_hours=1.0,
            update_frequency="event",
            base_learning_rate=0.3,
            decay_rate=0.95,
            promotion_threshold=1.0,
            demotion_threshold=0.2,
        )

        assert config.name == "fast"
        assert config.half_life_hours == 1.0
        assert config.base_learning_rate == 0.3

    def test_half_life_seconds(self):
        """Test half_life_seconds property."""
        config = TierConfig(
            name="test",
            half_life_hours=24.0,
            update_frequency="round",
            base_learning_rate=0.1,
            decay_rate=0.99,
            promotion_threshold=0.7,
            demotion_threshold=0.3,
        )

        # 24 hours = 86400 seconds
        assert config.half_life_seconds == 86400.0

    def test_half_life_seconds_fractional(self):
        """Test half_life_seconds with fractional hours."""
        config = TierConfig(
            name="test",
            half_life_hours=0.5,  # 30 minutes
            update_frequency="event",
            base_learning_rate=0.3,
            decay_rate=0.95,
            promotion_threshold=1.0,
            demotion_threshold=0.2,
        )

        assert config.half_life_seconds == 1800.0


class TestDefaultTierConfigs:
    """Test default tier configurations."""

    def test_all_tiers_have_config(self):
        """Test all tiers have a configuration."""
        for tier in MemoryTier:
            assert tier in DEFAULT_TIER_CONFIGS

    def test_fast_config(self):
        """Test fast tier configuration."""
        config = DEFAULT_TIER_CONFIGS[MemoryTier.FAST]

        assert config.name == "fast"
        assert config.half_life_hours == 1
        assert config.update_frequency == "event"
        assert config.base_learning_rate == 0.3
        assert config.promotion_threshold == 1.0  # Can't promote higher

    def test_medium_config(self):
        """Test medium tier configuration."""
        config = DEFAULT_TIER_CONFIGS[MemoryTier.MEDIUM]

        assert config.name == "medium"
        assert config.half_life_hours == 24
        assert config.update_frequency == "round"

    def test_slow_config(self):
        """Test slow tier configuration."""
        config = DEFAULT_TIER_CONFIGS[MemoryTier.SLOW]

        assert config.name == "slow"
        assert config.half_life_hours == 168  # 7 days
        assert config.update_frequency == "cycle"

    def test_glacial_config(self):
        """Test glacial tier configuration."""
        config = DEFAULT_TIER_CONFIGS[MemoryTier.GLACIAL]

        assert config.name == "glacial"
        assert config.half_life_hours == 720  # 30 days
        assert config.update_frequency == "monthly"
        assert config.demotion_threshold == 1.0  # Can't demote lower

    def test_learning_rates_decrease_by_tier(self):
        """Test learning rates decrease for slower tiers."""
        fast_lr = DEFAULT_TIER_CONFIGS[MemoryTier.FAST].base_learning_rate
        medium_lr = DEFAULT_TIER_CONFIGS[MemoryTier.MEDIUM].base_learning_rate
        slow_lr = DEFAULT_TIER_CONFIGS[MemoryTier.SLOW].base_learning_rate
        glacial_lr = DEFAULT_TIER_CONFIGS[MemoryTier.GLACIAL].base_learning_rate

        assert fast_lr > medium_lr > slow_lr > glacial_lr

    def test_decay_rates_increase_for_slower_tiers(self):
        """Test decay rates are closer to 1.0 for slower tiers."""
        fast_decay = DEFAULT_TIER_CONFIGS[MemoryTier.FAST].decay_rate
        medium_decay = DEFAULT_TIER_CONFIGS[MemoryTier.MEDIUM].decay_rate
        slow_decay = DEFAULT_TIER_CONFIGS[MemoryTier.SLOW].decay_rate
        glacial_decay = DEFAULT_TIER_CONFIGS[MemoryTier.GLACIAL].decay_rate

        assert fast_decay < medium_decay < slow_decay < glacial_decay


class TestTierTransitionMetrics:
    """Test TierTransitionMetrics dataclass."""

    def test_create_metrics(self):
        """Test creating transition metrics."""
        metrics = TierTransitionMetrics()

        assert metrics.total_promotions == 0
        assert metrics.total_demotions == 0
        assert metrics.promotions == {}
        assert metrics.demotions == {}

    def test_record_promotion(self):
        """Test recording a promotion."""
        metrics = TierTransitionMetrics()

        metrics.record_promotion(MemoryTier.SLOW, MemoryTier.MEDIUM)

        assert metrics.total_promotions == 1
        assert "slow->medium" in metrics.promotions
        assert metrics.promotions["slow->medium"] == 1

    def test_record_multiple_promotions(self):
        """Test recording multiple promotions."""
        metrics = TierTransitionMetrics()

        metrics.record_promotion(MemoryTier.SLOW, MemoryTier.MEDIUM)
        metrics.record_promotion(MemoryTier.SLOW, MemoryTier.MEDIUM)
        metrics.record_promotion(MemoryTier.MEDIUM, MemoryTier.FAST)

        assert metrics.total_promotions == 3
        assert metrics.promotions["slow->medium"] == 2
        assert metrics.promotions["medium->fast"] == 1

    def test_record_demotion(self):
        """Test recording a demotion."""
        metrics = TierTransitionMetrics()

        metrics.record_demotion(MemoryTier.FAST, MemoryTier.MEDIUM)

        assert metrics.total_demotions == 1
        assert "fast->medium" in metrics.demotions
        assert metrics.demotions["fast->medium"] == 1

    def test_record_multiple_demotions(self):
        """Test recording multiple demotions."""
        metrics = TierTransitionMetrics()

        metrics.record_demotion(MemoryTier.FAST, MemoryTier.MEDIUM)
        metrics.record_demotion(MemoryTier.FAST, MemoryTier.MEDIUM)
        metrics.record_demotion(MemoryTier.MEDIUM, MemoryTier.SLOW)

        assert metrics.total_demotions == 3
        assert metrics.demotions["fast->medium"] == 2
        assert metrics.demotions["medium->slow"] == 1

    def test_to_dict(self):
        """Test conversion to dictionary."""
        metrics = TierTransitionMetrics()
        metrics.record_promotion(MemoryTier.SLOW, MemoryTier.MEDIUM)
        metrics.record_demotion(MemoryTier.FAST, MemoryTier.MEDIUM)

        data = metrics.to_dict()

        assert data["total_promotions"] == 1
        assert data["total_demotions"] == 1
        assert "promotions" in data
        assert "demotions" in data
        assert "last_reset" in data

    def test_reset(self):
        """Test resetting metrics."""
        metrics = TierTransitionMetrics()
        metrics.record_promotion(MemoryTier.SLOW, MemoryTier.MEDIUM)
        metrics.record_demotion(MemoryTier.FAST, MemoryTier.MEDIUM)

        metrics.reset()

        assert metrics.total_promotions == 0
        assert metrics.total_demotions == 0
        assert metrics.promotions == {}
        assert metrics.demotions == {}

    def test_thread_safety(self):
        """Test that operations are thread-safe."""
        import threading

        metrics = TierTransitionMetrics()
        errors = []

        def promote_many():
            try:
                for _ in range(100):
                    metrics.record_promotion(MemoryTier.SLOW, MemoryTier.MEDIUM)
            except Exception as e:
                errors.append(e)

        def demote_many():
            try:
                for _ in range(100):
                    metrics.record_demotion(MemoryTier.FAST, MemoryTier.MEDIUM)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=promote_many),
            threading.Thread(target=demote_many),
            threading.Thread(target=promote_many),
            threading.Thread(target=demote_many),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert metrics.total_promotions == 200
        assert metrics.total_demotions == 200


class TestTierTransitions:
    """Test tier transition logic."""

    def test_promotion_thresholds(self):
        """Test promotion thresholds make sense."""
        # Fast can't promote (threshold is 1.0)
        assert DEFAULT_TIER_CONFIGS[MemoryTier.FAST].promotion_threshold == 1.0

        # Others have lower thresholds
        for tier in [MemoryTier.MEDIUM, MemoryTier.SLOW, MemoryTier.GLACIAL]:
            config = DEFAULT_TIER_CONFIGS[tier]
            assert config.promotion_threshold < 1.0

    def test_demotion_thresholds(self):
        """Test demotion thresholds make sense."""
        # Glacial can't demote (threshold is 1.0)
        assert DEFAULT_TIER_CONFIGS[MemoryTier.GLACIAL].demotion_threshold == 1.0

        # Others have lower thresholds
        for tier in [MemoryTier.FAST, MemoryTier.MEDIUM, MemoryTier.SLOW]:
            config = DEFAULT_TIER_CONFIGS[tier]
            assert config.demotion_threshold < 1.0

    def test_get_next_tier_faster(self):
        """Test getting next faster tier."""
        for i, tier in enumerate(TIER_ORDER[:-1]):  # All except FAST
            next_tier = TIER_ORDER[i + 1]
            assert TIER_ORDER.index(next_tier) > TIER_ORDER.index(tier)

    def test_get_next_tier_slower(self):
        """Test getting next slower tier."""
        for i, tier in enumerate(TIER_ORDER[1:], 1):  # All except GLACIAL
            prev_tier = TIER_ORDER[i - 1]
            assert TIER_ORDER.index(prev_tier) < TIER_ORDER.index(tier)


class TestTierConfigValidation:
    """Test tier configuration validation."""

    def test_valid_config_ranges(self):
        """Test all configs have valid value ranges."""
        for tier, config in DEFAULT_TIER_CONFIGS.items():
            # Half-life should be positive
            assert config.half_life_hours > 0

            # Learning rate should be between 0 and 1
            assert 0 < config.base_learning_rate <= 1

            # Decay rate should be between 0 and 1
            assert 0 < config.decay_rate < 1

            # Thresholds should be between 0 and 1
            assert 0 <= config.promotion_threshold <= 1
            assert 0 <= config.demotion_threshold <= 1

    def test_config_names_match_tiers(self):
        """Test config names match tier values."""
        for tier, config in DEFAULT_TIER_CONFIGS.items():
            assert config.name == tier.value
