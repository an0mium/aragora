"""Tests for the TierManager component of Continuum Memory System."""

import pytest
from datetime import datetime, timedelta

from aragora.memory.tier_manager import (
    TierManager,
    TierConfig,
    TierTransitionMetrics,
    MemoryTier,
    TIER_ORDER,
    DEFAULT_TIER_CONFIGS,
    get_tier_manager,
    reset_tier_manager,
)
from aragora.services import ServiceRegistry


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def fresh_registry():
    """Reset ServiceRegistry before and after each test."""
    ServiceRegistry.reset()
    yield
    ServiceRegistry.reset()


@pytest.fixture
def tier_manager(fresh_registry):
    """Create a fresh TierManager instance for testing."""
    return TierManager()


# ============================================================================
# MemoryTier Tests
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

    def test_tier_order_slowest_to_fastest(self):
        """Verify TIER_ORDER is from slowest to fastest."""
        assert TIER_ORDER[0] == MemoryTier.GLACIAL
        assert TIER_ORDER[-1] == MemoryTier.FAST
        assert len(TIER_ORDER) == 4


# ============================================================================
# TierConfig Tests
# ============================================================================


class TestTierConfig:
    """Tests for the TierConfig dataclass."""

    def test_all_required_fields(self):
        """Test TierConfig requires all fields."""
        config = TierConfig(
            name="test",
            half_life_hours=12.0,
            update_frequency="hourly",
            base_learning_rate=0.2,
            decay_rate=0.95,
            promotion_threshold=0.8,
            demotion_threshold=0.3,
        )
        assert config.name == "test"
        assert config.half_life_hours == 12.0
        assert config.update_frequency == "hourly"
        assert config.base_learning_rate == 0.2
        assert config.decay_rate == 0.95
        assert config.promotion_threshold == 0.8
        assert config.demotion_threshold == 0.3

    def test_half_life_seconds_property(self):
        """Test half_life_seconds property conversion."""
        config = TierConfig(
            name="test",
            half_life_hours=2.0,
            update_frequency="event",
            base_learning_rate=0.1,
            decay_rate=0.9,
            promotion_threshold=0.5,
            demotion_threshold=0.5,
        )
        assert config.half_life_seconds == 7200  # 2 hours * 3600 seconds


class TestDefaultTierConfigs:
    """Tests for DEFAULT_TIER_CONFIGS."""

    def test_all_tiers_have_config(self):
        """Verify all tiers have configuration."""
        for tier in MemoryTier:
            assert tier in DEFAULT_TIER_CONFIGS
            config = DEFAULT_TIER_CONFIGS[tier]
            assert isinstance(config, TierConfig)

    def test_fast_tier_config(self):
        """Verify fast tier has correct configuration."""
        config = DEFAULT_TIER_CONFIGS[MemoryTier.FAST]
        assert config.name == "fast"
        assert config.half_life_hours == 1
        assert config.update_frequency == "event"
        assert config.promotion_threshold == 1.0  # Can't promote higher
        assert config.demotion_threshold == 0.2

    def test_glacial_tier_config(self):
        """Verify glacial tier has correct configuration."""
        config = DEFAULT_TIER_CONFIGS[MemoryTier.GLACIAL]
        assert config.name == "glacial"
        assert config.half_life_hours == 720  # 30 days
        assert config.demotion_threshold == 1.0  # Can't demote lower


# ============================================================================
# TierTransitionMetrics Tests
# ============================================================================


class TestTierTransitionMetrics:
    """Tests for the TierTransitionMetrics dataclass."""

    def test_default_values(self):
        """Test TierTransitionMetrics default values."""
        metrics = TierTransitionMetrics()
        assert metrics.promotions == {}
        assert metrics.demotions == {}
        assert metrics.total_promotions == 0
        assert metrics.total_demotions == 0
        assert metrics.last_reset is not None

    def test_record_promotion(self):
        """Test recording a promotion."""
        metrics = TierTransitionMetrics()
        metrics.record_promotion(MemoryTier.MEDIUM, MemoryTier.FAST)

        assert metrics.total_promotions == 1
        assert "medium->fast" in metrics.promotions
        assert metrics.promotions["medium->fast"] == 1

    def test_record_demotion(self):
        """Test recording a demotion."""
        metrics = TierTransitionMetrics()
        metrics.record_demotion(MemoryTier.FAST, MemoryTier.MEDIUM)

        assert metrics.total_demotions == 1
        assert "fast->medium" in metrics.demotions
        assert metrics.demotions["fast->medium"] == 1

    def test_record_multiple_transitions(self):
        """Test recording multiple transitions."""
        metrics = TierTransitionMetrics()

        # Record several promotions
        metrics.record_promotion(MemoryTier.GLACIAL, MemoryTier.SLOW)
        metrics.record_promotion(MemoryTier.GLACIAL, MemoryTier.SLOW)
        metrics.record_promotion(MemoryTier.SLOW, MemoryTier.MEDIUM)

        # Record some demotions
        metrics.record_demotion(MemoryTier.FAST, MemoryTier.MEDIUM)

        assert metrics.total_promotions == 3
        assert metrics.total_demotions == 1
        assert metrics.promotions["glacial->slow"] == 2
        assert metrics.promotions["slow->medium"] == 1

    def test_to_dict(self):
        """Test to_dict serialization."""
        metrics = TierTransitionMetrics()
        metrics.record_promotion(MemoryTier.SLOW, MemoryTier.MEDIUM)

        d = metrics.to_dict()
        assert "promotions" in d
        assert "demotions" in d
        assert "total_promotions" in d
        assert "total_demotions" in d
        assert "last_reset" in d
        assert d["total_promotions"] == 1

    def test_reset(self):
        """Test reset clears all metrics."""
        metrics = TierTransitionMetrics()
        metrics.record_promotion(MemoryTier.SLOW, MemoryTier.MEDIUM)
        metrics.record_demotion(MemoryTier.FAST, MemoryTier.MEDIUM)

        old_reset = metrics.last_reset
        metrics.reset()

        assert metrics.promotions == {}
        assert metrics.demotions == {}
        assert metrics.total_promotions == 0
        assert metrics.total_demotions == 0
        assert metrics.last_reset != old_reset


# ============================================================================
# TierManager Initialization Tests
# ============================================================================


class TestTierManagerInit:
    """Tests for TierManager initialization."""

    def test_default_initialization(self, tier_manager):
        """Test default TierManager initialization."""
        assert tier_manager is not None
        assert tier_manager.promotion_cooldown_hours == 24.0
        assert tier_manager.min_updates_for_demotion == 10

    def test_custom_cooldown(self, fresh_registry):
        """Test TierManager with custom cooldown."""
        tm = TierManager(promotion_cooldown_hours=12.0)
        assert tm.promotion_cooldown_hours == 12.0

    def test_custom_min_updates(self, fresh_registry):
        """Test TierManager with custom min updates."""
        tm = TierManager(min_updates_for_demotion=5)
        assert tm.min_updates_for_demotion == 5

    def test_custom_configs(self, fresh_registry):
        """Test TierManager with custom tier configs."""
        custom_config = TierConfig(
            name="custom_fast",
            half_life_hours=0.5,
            update_frequency="event",
            base_learning_rate=0.5,
            decay_rate=0.9,
            promotion_threshold=0.9,
            demotion_threshold=0.1,
        )
        custom_configs = {**DEFAULT_TIER_CONFIGS, MemoryTier.FAST: custom_config}
        tm = TierManager(configs=custom_configs)

        config = tm.get_config(MemoryTier.FAST)
        assert config.half_life_hours == 0.5
        assert config.promotion_threshold == 0.9


# ============================================================================
# TierManager Config Methods Tests
# ============================================================================


class TestTierManagerConfigs:
    """Tests for TierManager configuration methods."""

    def test_get_config(self, tier_manager):
        """Test get_config returns correct config."""
        for tier in MemoryTier:
            config = tier_manager.get_config(tier)
            assert isinstance(config, TierConfig)
            assert config.name == tier.value

    def test_get_all_configs(self, tier_manager):
        """Test get_all_configs returns all configs."""
        configs = tier_manager.get_all_configs()
        assert len(configs) == 4
        for tier in MemoryTier:
            assert tier in configs

    def test_get_all_configs_returns_copy(self, tier_manager):
        """Test get_all_configs returns a copy, not the original."""
        configs1 = tier_manager.get_all_configs()
        configs2 = tier_manager.get_all_configs()
        assert configs1 is not configs2

    def test_update_config(self, tier_manager):
        """Test update_config updates the configuration."""
        new_config = TierConfig(
            name="fast",
            half_life_hours=0.5,
            update_frequency="event",
            base_learning_rate=0.4,
            decay_rate=0.92,
            promotion_threshold=1.0,
            demotion_threshold=0.15,
        )
        tier_manager.update_config(MemoryTier.FAST, new_config)

        config = tier_manager.get_config(MemoryTier.FAST)
        assert config.half_life_hours == 0.5
        assert config.demotion_threshold == 0.15


# ============================================================================
# TierManager get_tier_index Tests
# ============================================================================


class TestGetTierIndex:
    """Tests for get_tier_index method."""

    def test_glacial_is_index_0(self, tier_manager):
        """Test GLACIAL has index 0 (slowest)."""
        assert tier_manager.get_tier_index(MemoryTier.GLACIAL) == 0

    def test_slow_is_index_1(self, tier_manager):
        """Test SLOW has index 1."""
        assert tier_manager.get_tier_index(MemoryTier.SLOW) == 1

    def test_medium_is_index_2(self, tier_manager):
        """Test MEDIUM has index 2."""
        assert tier_manager.get_tier_index(MemoryTier.MEDIUM) == 2

    def test_fast_is_index_3(self, tier_manager):
        """Test FAST has index 3 (fastest)."""
        assert tier_manager.get_tier_index(MemoryTier.FAST) == 3


# ============================================================================
# TierManager get_next_tier Tests
# ============================================================================


class TestGetNextTier:
    """Tests for get_next_tier method."""

    def test_promote_from_glacial(self, tier_manager):
        """Test promotion from GLACIAL to SLOW."""
        next_tier = tier_manager.get_next_tier(MemoryTier.GLACIAL, "faster")
        assert next_tier == MemoryTier.SLOW

    def test_promote_from_slow(self, tier_manager):
        """Test promotion from SLOW to MEDIUM."""
        next_tier = tier_manager.get_next_tier(MemoryTier.SLOW, "faster")
        assert next_tier == MemoryTier.MEDIUM

    def test_promote_from_medium(self, tier_manager):
        """Test promotion from MEDIUM to FAST."""
        next_tier = tier_manager.get_next_tier(MemoryTier.MEDIUM, "faster")
        assert next_tier == MemoryTier.FAST

    def test_promote_from_fast_returns_none(self, tier_manager):
        """Test that FAST cannot be promoted further."""
        next_tier = tier_manager.get_next_tier(MemoryTier.FAST, "faster")
        assert next_tier is None

    def test_demote_from_fast(self, tier_manager):
        """Test demotion from FAST to MEDIUM."""
        next_tier = tier_manager.get_next_tier(MemoryTier.FAST, "slower")
        assert next_tier == MemoryTier.MEDIUM

    def test_demote_from_medium(self, tier_manager):
        """Test demotion from MEDIUM to SLOW."""
        next_tier = tier_manager.get_next_tier(MemoryTier.MEDIUM, "slower")
        assert next_tier == MemoryTier.SLOW

    def test_demote_from_slow(self, tier_manager):
        """Test demotion from SLOW to GLACIAL."""
        next_tier = tier_manager.get_next_tier(MemoryTier.SLOW, "slower")
        assert next_tier == MemoryTier.GLACIAL

    def test_demote_from_glacial_returns_none(self, tier_manager):
        """Test that GLACIAL cannot be demoted further."""
        next_tier = tier_manager.get_next_tier(MemoryTier.GLACIAL, "slower")
        assert next_tier is None

    def test_invalid_direction_raises(self, tier_manager):
        """Test invalid direction raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            tier_manager.get_next_tier(MemoryTier.FAST, "invalid")
        assert "Invalid direction" in str(exc_info.value)


# ============================================================================
# TierManager should_promote Tests
# ============================================================================


class TestShouldPromote:
    """Tests for should_promote method."""

    def test_high_surprise_promotes(self, tier_manager):
        """Test high surprise score triggers promotion."""
        # MEDIUM tier has promotion_threshold of 0.7
        result = tier_manager.should_promote(
            tier=MemoryTier.MEDIUM,
            surprise_score=0.9,  # Above 0.7
        )
        assert result is True

    def test_low_surprise_does_not_promote(self, tier_manager):
        """Test low surprise score does not trigger promotion."""
        result = tier_manager.should_promote(
            tier=MemoryTier.MEDIUM,
            surprise_score=0.5,  # Below 0.7
        )
        assert result is False

    def test_fast_tier_never_promotes(self, tier_manager):
        """Test FAST tier never promotes regardless of surprise."""
        result = tier_manager.should_promote(
            tier=MemoryTier.FAST,
            surprise_score=1.0,  # Maximum surprise
        )
        assert result is False

    def test_cooldown_prevents_promotion(self, tier_manager):
        """Test recent promotion prevents new promotion."""
        recent_promotion = datetime.now().isoformat()
        result = tier_manager.should_promote(
            tier=MemoryTier.MEDIUM,
            surprise_score=0.9,
            last_promotion_at=recent_promotion,
        )
        assert result is False

    def test_old_promotion_allows_new_promotion(self, tier_manager):
        """Test promotion after cooldown period is allowed."""
        old_promotion = (datetime.now() - timedelta(hours=48)).isoformat()
        result = tier_manager.should_promote(
            tier=MemoryTier.MEDIUM,
            surprise_score=0.9,
            last_promotion_at=old_promotion,
        )
        assert result is True

    def test_no_last_promotion_allows_promotion(self, tier_manager):
        """Test promotion without previous promotion history."""
        result = tier_manager.should_promote(
            tier=MemoryTier.MEDIUM,
            surprise_score=0.9,
            last_promotion_at=None,
        )
        assert result is True


# ============================================================================
# TierManager should_demote Tests
# ============================================================================


class TestShouldDemote:
    """Tests for should_demote method."""

    def test_low_surprise_with_enough_updates_demotes(self, tier_manager):
        """Test low surprise with sufficient updates triggers demotion."""
        # FAST tier has demotion_threshold of 0.2
        # Stability = 1 - surprise, so surprise of 0.1 gives stability of 0.9
        result = tier_manager.should_demote(
            tier=MemoryTier.FAST,
            surprise_score=0.1,  # Stability = 0.9 > 0.2
            update_count=20,  # > min_updates_for_demotion (10)
        )
        assert result is True

    def test_high_surprise_does_not_demote(self, tier_manager):
        """Test high surprise (low stability) does not demote."""
        result = tier_manager.should_demote(
            tier=MemoryTier.FAST,
            surprise_score=0.9,  # Stability = 0.1 < 0.2
            update_count=20,
        )
        assert result is False

    def test_insufficient_updates_does_not_demote(self, tier_manager):
        """Test insufficient updates prevents demotion."""
        result = tier_manager.should_demote(
            tier=MemoryTier.FAST,
            surprise_score=0.1,  # Would normally demote
            update_count=5,  # < min_updates_for_demotion (10)
        )
        assert result is False

    def test_glacial_tier_never_demotes(self, tier_manager):
        """Test GLACIAL tier never demotes."""
        result = tier_manager.should_demote(
            tier=MemoryTier.GLACIAL,
            surprise_score=0.0,  # Maximum stability
            update_count=1000,
        )
        assert result is False


# ============================================================================
# TierManager calculate_decay_factor Tests
# ============================================================================


class TestCalculateDecayFactor:
    """Tests for calculate_decay_factor method."""

    def test_zero_hours_returns_one(self, tier_manager):
        """Test zero elapsed time gives no decay."""
        factor = tier_manager.calculate_decay_factor(MemoryTier.FAST, 0)
        assert factor == 1.0

    def test_negative_hours_returns_one(self, tier_manager):
        """Test negative elapsed time gives no decay."""
        factor = tier_manager.calculate_decay_factor(MemoryTier.FAST, -1)
        assert factor == 1.0

    def test_half_life_returns_half(self, tier_manager):
        """Test one half-life gives 0.5 decay."""
        # FAST tier has half_life_hours of 1
        factor = tier_manager.calculate_decay_factor(MemoryTier.FAST, 1.0)
        assert abs(factor - 0.5) < 0.001

    def test_two_half_lives_returns_quarter(self, tier_manager):
        """Test two half-lives gives 0.25 decay."""
        factor = tier_manager.calculate_decay_factor(MemoryTier.FAST, 2.0)
        assert abs(factor - 0.25) < 0.001

    def test_different_tier_half_lives(self, tier_manager):
        """Test different tiers have different decay rates."""
        fast_factor = tier_manager.calculate_decay_factor(MemoryTier.FAST, 24)
        slow_factor = tier_manager.calculate_decay_factor(MemoryTier.SLOW, 24)

        # FAST decays much faster than SLOW
        assert fast_factor < slow_factor


# ============================================================================
# TierManager Metrics Tests
# ============================================================================


class TestTierManagerMetrics:
    """Tests for TierManager metrics functionality."""

    def test_record_promotion(self, tier_manager):
        """Test recording a promotion."""
        tier_manager.record_promotion(MemoryTier.MEDIUM, MemoryTier.FAST)

        metrics = tier_manager.get_metrics()
        assert metrics.total_promotions == 1
        assert "medium->fast" in metrics.promotions

    def test_record_demotion(self, tier_manager):
        """Test recording a demotion."""
        tier_manager.record_demotion(MemoryTier.FAST, MemoryTier.MEDIUM)

        metrics = tier_manager.get_metrics()
        assert metrics.total_demotions == 1
        assert "fast->medium" in metrics.demotions

    def test_get_metrics_dict(self, tier_manager):
        """Test get_metrics_dict returns serializable dict."""
        tier_manager.record_promotion(MemoryTier.SLOW, MemoryTier.MEDIUM)

        d = tier_manager.get_metrics_dict()
        assert isinstance(d, dict)
        assert d["total_promotions"] == 1

    def test_reset_metrics(self, tier_manager):
        """Test reset_metrics clears all metrics."""
        tier_manager.record_promotion(MemoryTier.SLOW, MemoryTier.MEDIUM)
        tier_manager.record_demotion(MemoryTier.FAST, MemoryTier.MEDIUM)

        tier_manager.reset_metrics()

        metrics = tier_manager.get_metrics()
        assert metrics.total_promotions == 0
        assert metrics.total_demotions == 0


# ============================================================================
# TierManager Properties Tests
# ============================================================================


class TestTierManagerProperties:
    """Tests for TierManager property getters and setters."""

    def test_promotion_cooldown_getter(self, tier_manager):
        """Test promotion_cooldown_hours getter."""
        assert tier_manager.promotion_cooldown_hours == 24.0

    def test_promotion_cooldown_setter(self, tier_manager):
        """Test promotion_cooldown_hours setter."""
        tier_manager.promotion_cooldown_hours = 12.0
        assert tier_manager.promotion_cooldown_hours == 12.0

    def test_promotion_cooldown_minimum_zero(self, tier_manager):
        """Test promotion_cooldown_hours enforces minimum of 0."""
        tier_manager.promotion_cooldown_hours = -5.0
        assert tier_manager.promotion_cooldown_hours == 0.0

    def test_min_updates_getter(self, tier_manager):
        """Test min_updates_for_demotion getter."""
        assert tier_manager.min_updates_for_demotion == 10

    def test_min_updates_setter(self, tier_manager):
        """Test min_updates_for_demotion setter."""
        tier_manager.min_updates_for_demotion = 5
        assert tier_manager.min_updates_for_demotion == 5

    def test_min_updates_minimum_one(self, tier_manager):
        """Test min_updates_for_demotion enforces minimum of 1."""
        tier_manager.min_updates_for_demotion = 0
        assert tier_manager.min_updates_for_demotion == 1


# ============================================================================
# ServiceRegistry Integration Tests
# ============================================================================


class TestServiceRegistryIntegration:
    """Tests for TierManager ServiceRegistry integration."""

    def test_get_tier_manager_creates_instance(self, fresh_registry):
        """Test get_tier_manager creates new instance."""
        tm = get_tier_manager()
        assert tm is not None
        assert isinstance(tm, TierManager)

    def test_get_tier_manager_returns_singleton(self, fresh_registry):
        """Test get_tier_manager returns same instance."""
        tm1 = get_tier_manager()
        tm2 = get_tier_manager()
        assert tm1 is tm2

    def test_reset_tier_manager_clears_instance(self, fresh_registry):
        """Test reset_tier_manager removes from registry."""
        tm1 = get_tier_manager()
        reset_tier_manager()
        tm2 = get_tier_manager()
        assert tm1 is not tm2

    def test_reset_preserves_other_services(self, fresh_registry):
        """Test reset_tier_manager doesn't affect other services."""

        class OtherService:
            pass

        from aragora.services import register_service, has_service

        other = OtherService()
        register_service(OtherService, other)

        get_tier_manager()
        reset_tier_manager()

        # Other service should still be registered
        assert has_service(OtherService)


# ============================================================================
# Edge Cases and Performance
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_boundary_surprise_scores(self, tier_manager):
        """Test boundary surprise score values."""
        # Exactly at threshold should not promote (> not >=)
        result = tier_manager.should_promote(
            tier=MemoryTier.MEDIUM,
            surprise_score=0.7,  # Exactly at threshold
        )
        assert result is False

        # Just above threshold should promote
        result = tier_manager.should_promote(
            tier=MemoryTier.MEDIUM,
            surprise_score=0.71,
        )
        assert result is True


class TestPerformance:
    """Tests for performance characteristics."""

    def test_many_transition_records(self, tier_manager):
        """Test recording many transitions doesn't degrade."""
        for i in range(1000):
            tier_manager.record_promotion(MemoryTier.SLOW, MemoryTier.MEDIUM)

        metrics = tier_manager.get_metrics()
        assert metrics.total_promotions == 1000
        assert metrics.promotions["slow->medium"] == 1000

    def test_metrics_retrieval_fast(self, tier_manager):
        """Test metrics retrieval is fast even with many records."""
        import time

        # Record many transitions
        for i in range(1000):
            tier_manager.record_promotion(MemoryTier.SLOW, MemoryTier.MEDIUM)

        start = time.time()
        for _ in range(100):
            tier_manager.get_metrics_dict()
        elapsed = time.time() - start

        # Should complete 100 metrics retrievals in under 0.1 seconds
        assert elapsed < 0.1
