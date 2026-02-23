"""Tests for surprise-modulated dynamic decay rates.

Titans-inspired: high surprise → longer half-life (preserve novel knowledge),
low surprise + tier pressure → shorter half-life (forget faster).
"""

from __future__ import annotations

import pytest

from aragora.knowledge.mound.ops.confidence_decay import (
    ConfidenceDecayManager,
    DecayConfig,
    DecayModel,
)


class TestCalculateDynamicHalfLife:
    """Tests for calculate_dynamic_half_life method."""

    def _manager(self, **kwargs) -> ConfidenceDecayManager:
        return ConfidenceDecayManager(
            DecayConfig(
                enable_surprise_modulated_decay=True,
                **kwargs,
            )
        )

    def test_neutral_surprise_no_change(self):
        """Surprise=0.5 should yield factor=1.0, preserving base half-life."""
        mgr = self._manager()
        result = mgr.calculate_dynamic_half_life(90.0, item_surprise=0.5)
        assert result == pytest.approx(90.0)

    def test_high_surprise_increases_half_life(self):
        """High surprise (=1.0) should extend half-life (slower decay)."""
        mgr = self._manager()
        result = mgr.calculate_dynamic_half_life(90.0, item_surprise=1.0)
        assert result > 90.0

    def test_low_surprise_decreases_half_life(self):
        """Low surprise (=0.0) should shorten half-life (faster decay)."""
        mgr = self._manager()
        result = mgr.calculate_dynamic_half_life(90.0, item_surprise=0.0)
        assert result < 90.0

    def test_clamped_to_floor(self):
        """Half-life never goes below base * min_half_life_ratio."""
        mgr = self._manager(min_half_life_ratio=0.25, surprise_decay_strength=10.0)
        result = mgr.calculate_dynamic_half_life(90.0, item_surprise=0.0)
        assert result >= 90.0 * 0.25

    def test_clamped_to_ceiling(self):
        """Half-life never exceeds base * max_half_life_ratio."""
        mgr = self._manager(max_half_life_ratio=3.0, surprise_decay_strength=10.0)
        result = mgr.calculate_dynamic_half_life(90.0, item_surprise=1.0)
        assert result <= 90.0 * 3.0

    def test_tier_pressure_accelerates_forgetting_for_low_surprise(self):
        """High tier pressure + low surprise → even shorter half-life."""
        mgr = self._manager()
        no_pressure = mgr.calculate_dynamic_half_life(90.0, item_surprise=0.2, tier_pressure=0.0)
        full_pressure = mgr.calculate_dynamic_half_life(90.0, item_surprise=0.2, tier_pressure=1.0)
        assert full_pressure < no_pressure

    def test_tier_pressure_does_not_affect_high_surprise(self):
        """High surprise items resist tier pressure."""
        mgr = self._manager()
        no_pressure = mgr.calculate_dynamic_half_life(90.0, item_surprise=1.0, tier_pressure=0.0)
        full_pressure = mgr.calculate_dynamic_half_life(90.0, item_surprise=1.0, tier_pressure=1.0)
        # surprise=1.0 → (1.0 - 1.0) = 0 → pressure_factor = 1.0
        assert full_pressure == pytest.approx(no_pressure)

    def test_custom_strength(self):
        """Strength parameter controls how strongly surprise affects half-life."""
        weak = ConfidenceDecayManager(
            DecayConfig(
                enable_surprise_modulated_decay=True,
                surprise_decay_strength=1.0,
            )
        )
        strong = ConfidenceDecayManager(
            DecayConfig(
                enable_surprise_modulated_decay=True,
                surprise_decay_strength=4.0,
            )
        )

        weak_hl = weak.calculate_dynamic_half_life(90.0, item_surprise=0.8)
        strong_hl = strong.calculate_dynamic_half_life(90.0, item_surprise=0.8)
        # Stronger strength → more dramatic effect
        assert strong_hl > weak_hl

    def test_zero_base_half_life(self):
        """Zero base half-life stays zero regardless of surprise."""
        mgr = self._manager()
        result = mgr.calculate_dynamic_half_life(0.0, item_surprise=0.8)
        assert result == 0.0


class TestCalculateDecayWithSurprise:
    """Tests for calculate_decay integration with surprise modulation."""

    def test_disabled_by_default(self):
        """Without enable_surprise_modulated_decay, surprise_score is ignored."""
        mgr = ConfidenceDecayManager(DecayConfig(half_life_days=90.0))
        without_surprise = mgr.calculate_decay(1.0, age_days=90.0)
        with_surprise = mgr.calculate_decay(1.0, age_days=90.0, surprise_score=1.0)
        assert without_surprise == pytest.approx(with_surprise)

    def test_enabled_high_surprise_slower_decay(self):
        """With surprise modulation, high-surprise items decay slower."""
        config = DecayConfig(
            half_life_days=90.0,
            enable_surprise_modulated_decay=True,
        )
        mgr = ConfidenceDecayManager(config)

        normal = mgr.calculate_decay(1.0, age_days=90.0, surprise_score=0.5)
        slow = mgr.calculate_decay(1.0, age_days=90.0, surprise_score=1.0)

        # High surprise → longer half-life → higher confidence after same time
        assert slow > normal

    def test_enabled_low_surprise_faster_decay(self):
        """With surprise modulation, low-surprise items decay faster."""
        config = DecayConfig(
            half_life_days=90.0,
            enable_surprise_modulated_decay=True,
        )
        mgr = ConfidenceDecayManager(config)

        normal = mgr.calculate_decay(1.0, age_days=90.0, surprise_score=0.5)
        fast = mgr.calculate_decay(1.0, age_days=90.0, surprise_score=0.0)

        assert fast < normal

    def test_no_surprise_score_no_modulation(self):
        """When surprise_score is None, even if enabled, no modulation applied."""
        config = DecayConfig(
            half_life_days=90.0,
            enable_surprise_modulated_decay=True,
        )
        mgr = ConfidenceDecayManager(config)

        without = mgr.calculate_decay(1.0, age_days=90.0, surprise_score=None)
        baseline = ConfidenceDecayManager(DecayConfig(half_life_days=90.0)).calculate_decay(
            1.0, age_days=90.0
        )
        assert without == pytest.approx(baseline)

    def test_domain_half_life_used_as_base(self):
        """Domain-specific half-life is the base for dynamic modulation."""
        config = DecayConfig(
            half_life_days=90.0,
            domain_half_lives={"news": 7.0},
            enable_surprise_modulated_decay=True,
        )
        mgr = ConfidenceDecayManager(config)

        # News domain with neutral surprise
        result = mgr.calculate_decay(1.0, age_days=7.0, domain="news", surprise_score=0.5)
        # Should be approximately half (neutral surprise → base half-life)
        assert result == pytest.approx(0.5, abs=0.01)

    def test_linear_model_with_surprise(self):
        """Linear decay model also works with surprise modulation."""
        config = DecayConfig(
            model=DecayModel.LINEAR,
            half_life_days=90.0,
            enable_surprise_modulated_decay=True,
        )
        mgr = ConfidenceDecayManager(config)

        normal = mgr.calculate_decay(1.0, age_days=45.0, surprise_score=0.5)
        slow = mgr.calculate_decay(1.0, age_days=45.0, surprise_score=1.0)

        assert slow > normal

    def test_tier_pressure_in_calculate_decay(self):
        """tier_pressure parameter flows through to dynamic half-life."""
        config = DecayConfig(
            half_life_days=90.0,
            enable_surprise_modulated_decay=True,
        )
        mgr = ConfidenceDecayManager(config)

        relaxed = mgr.calculate_decay(1.0, age_days=90.0, surprise_score=0.2, tier_pressure=0.0)
        pressured = mgr.calculate_decay(1.0, age_days=90.0, surprise_score=0.2, tier_pressure=1.0)
        assert pressured < relaxed

    def test_min_confidence_floor_still_applies(self):
        """min_confidence floor still respected with surprise modulation."""
        config = DecayConfig(
            half_life_days=10.0,
            min_confidence=0.1,
            enable_surprise_modulated_decay=True,
        )
        mgr = ConfidenceDecayManager(config)
        result = mgr.calculate_decay(1.0, age_days=1000.0, surprise_score=0.0, tier_pressure=1.0)
        assert result >= 0.1


class TestDecayConfigDefaults:
    """Verify DecayConfig default values for new fields."""

    def test_surprise_modulated_decay_off_by_default(self):
        config = DecayConfig()
        assert config.enable_surprise_modulated_decay is False

    def test_default_strength(self):
        config = DecayConfig()
        assert config.surprise_decay_strength == 2.0

    def test_default_ratios(self):
        config = DecayConfig()
        assert config.min_half_life_ratio == 0.25
        assert config.max_half_life_ratio == 3.0
