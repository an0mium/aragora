"""
Tests for weight calculation module.

Tests cover:
- WeightFactors dataclass
- WeightCalculatorConfig dataclass
- WeightCalculator class
"""

from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

from aragora.debate.phases.weight_calculator import (
    WeightCalculator,
    WeightCalculatorConfig,
    WeightFactors,
)


@dataclass
class MockAgent:
    """Mock agent for testing."""

    name: str
    role: str = "debater"


@dataclass
class MockRating:
    """Mock rating for testing."""

    elo: float = 1000.0
    calibration_score: float = 0.5


@dataclass
class MockConsistency:
    """Mock consistency result for testing."""

    consistency_score: float = 0.8


class TestWeightFactors:
    """Tests for WeightFactors dataclass."""

    def test_default_factors(self):
        """Default factors are all 1.0."""
        factors = WeightFactors()

        assert factors.reputation == 1.0
        assert factors.reliability == 1.0
        assert factors.consistency == 1.0
        assert factors.calibration == 1.0
        assert factors.total == 1.0

    def test_custom_factors(self):
        """Custom factors are stored correctly."""
        factors = WeightFactors(
            reputation=1.2,
            reliability=0.9,
            consistency=0.8,
            calibration=1.1,
        )

        assert factors.reputation == 1.2
        assert factors.reliability == 0.9
        assert factors.consistency == 0.8
        assert factors.calibration == 1.1

    def test_total_calculation(self):
        """Total is product of all factors."""
        factors = WeightFactors(
            reputation=1.5,
            reliability=0.8,
            consistency=1.0,
            calibration=1.2,
        )

        expected = 1.5 * 0.8 * 1.0 * 1.2
        assert factors.total == pytest.approx(expected)


class TestWeightCalculatorConfig:
    """Tests for WeightCalculatorConfig dataclass."""

    def test_default_config(self):
        """Default config enables all factors."""
        config = WeightCalculatorConfig()

        assert config.enable_reputation is True
        assert config.enable_reliability is True
        assert config.enable_consistency is True
        assert config.enable_calibration is True
        assert config.min_weight == 0.1
        assert config.max_weight == 5.0

    def test_custom_config(self):
        """Custom config overrides defaults."""
        config = WeightCalculatorConfig(
            enable_reputation=False,
            enable_reliability=False,
            min_weight=0.5,
            max_weight=2.0,
        )

        assert config.enable_reputation is False
        assert config.enable_reliability is False
        assert config.min_weight == 0.5
        assert config.max_weight == 2.0


class TestWeightCalculator:
    """Tests for WeightCalculator class."""

    def test_default_weight(self):
        """Default weight is 1.0 when no systems configured."""
        calculator = WeightCalculator()

        weight = calculator.get_weight("agent1")

        assert weight == 1.0

    def test_weight_with_reputation(self):
        """Reputation weight is used from memory."""
        memory = MagicMock()
        memory.get_vote_weight.return_value = 1.3

        calculator = WeightCalculator(memory=memory)
        weight = calculator.get_weight("agent1")

        memory.get_vote_weight.assert_called_with("agent1")
        assert weight == 1.3

    def test_weight_with_reliability(self):
        """Reliability weight is used from agent_weights."""
        calculator = WeightCalculator(
            agent_weights={"agent1": 0.9, "agent2": 0.8}
        )

        weight1 = calculator.get_weight("agent1")
        weight2 = calculator.get_weight("agent2")

        assert weight1 == 0.9
        assert weight2 == 0.8

    def test_weight_with_consistency(self):
        """Consistency weight is calculated from FlipDetector."""
        flip_detector = MagicMock()
        flip_detector.get_agent_consistency.return_value = MockConsistency(
            consistency_score=1.0
        )

        calculator = WeightCalculator(flip_detector=flip_detector)
        weight = calculator.get_weight("agent1")

        flip_detector.get_agent_consistency.assert_called_with("agent1")
        # Consistency maps 0-1 to 0.5-1.0, so 1.0 -> 1.0
        assert weight == 1.0

    def test_weight_with_calibration_callback(self):
        """Calibration weight uses callback when provided."""
        get_cal_weight = MagicMock(return_value=1.2)

        calculator = WeightCalculator(get_calibration_weight=get_cal_weight)
        weight = calculator.get_weight("agent1")

        get_cal_weight.assert_called_with("agent1")
        assert weight == 1.2

    def test_weight_respects_bounds(self):
        """Weight is clamped to configured bounds."""
        config = WeightCalculatorConfig(min_weight=0.5, max_weight=2.0)
        # All factors multiply to create very high weight
        memory = MagicMock()
        memory.get_vote_weight.return_value = 10.0

        calculator = WeightCalculator(memory=memory, config=config)
        weight = calculator.get_weight("agent1")

        assert weight == 2.0  # Clamped to max

    def test_weight_minimum_bound(self):
        """Weight respects minimum bound."""
        config = WeightCalculatorConfig(min_weight=0.5, max_weight=2.0)
        memory = MagicMock()
        memory.get_vote_weight.return_value = 0.1

        calculator = WeightCalculator(memory=memory, config=config)
        weight = calculator.get_weight("agent1")

        assert weight == 0.5  # Clamped to min

    def test_compute_weights_batch(self):
        """compute_weights calculates for all agents."""
        calculator = WeightCalculator(
            agent_weights={"agent1": 1.5, "agent2": 1.2, "agent3": 0.9}
        )
        agents = [MockAgent("agent1"), MockAgent("agent2"), MockAgent("agent3")]

        weights = calculator.compute_weights(agents)

        assert weights["agent1"] == 1.5
        assert weights["agent2"] == 1.2
        assert weights["agent3"] == 0.9

    def test_get_weight_with_factors(self):
        """get_weight_with_factors returns breakdown."""
        memory = MagicMock()
        memory.get_vote_weight.return_value = 1.2

        calculator = WeightCalculator(
            memory=memory,
            agent_weights={"agent1": 0.9},
        )

        weight, factors = calculator.get_weight_with_factors("agent1")

        assert factors.reputation == 1.2
        assert factors.reliability == 0.9
        assert weight == pytest.approx(1.08)  # 1.2 * 0.9

    def test_disabled_factors(self):
        """Disabled factors are not used."""
        config = WeightCalculatorConfig(
            enable_reputation=False,
            enable_reliability=False,
            enable_consistency=False,
            enable_calibration=False,
        )
        memory = MagicMock()
        memory.get_vote_weight.return_value = 2.0

        calculator = WeightCalculator(memory=memory, config=config)
        weight, factors = calculator.get_weight_with_factors("agent1")

        # All factors should be 1.0 since disabled
        assert factors.reputation == 1.0
        assert factors.reliability == 1.0
        assert factors.consistency == 1.0
        assert factors.calibration == 1.0
        assert weight == 1.0

    def test_prefetch_ratings(self):
        """Ratings are prefetched for batch operations."""
        elo_system = MagicMock()
        elo_system.get_ratings_batch.return_value = {
            "agent1": MockRating(calibration_score=0.8),
        }

        calculator = WeightCalculator(elo_system=elo_system)
        agents = [MockAgent("agent1")]

        weights = calculator.compute_weights(agents)

        elo_system.get_ratings_batch.assert_called_once_with(["agent1"])

    def test_clear_cache(self):
        """clear_cache clears the ratings cache."""
        elo_system = MagicMock()
        elo_system.get_ratings_batch.return_value = {"agent1": MockRating()}

        calculator = WeightCalculator(elo_system=elo_system)
        calculator._prefetch_ratings(["agent1"])

        assert "agent1" in calculator._ratings_cache

        calculator.clear_cache()

        assert calculator._ratings_cache == {}

    def test_memory_error_handling(self):
        """Memory errors are handled gracefully."""
        memory = MagicMock()
        memory.get_vote_weight.side_effect = Exception("Memory error")

        calculator = WeightCalculator(memory=memory)
        weight = calculator.get_weight("agent1")

        # Should fall back to 1.0
        assert weight == 1.0

    def test_flip_detector_error_handling(self):
        """FlipDetector errors are handled gracefully."""
        flip_detector = MagicMock()
        flip_detector.get_agent_consistency.side_effect = Exception("FD error")

        calculator = WeightCalculator(flip_detector=flip_detector)
        weight = calculator.get_weight("agent1")

        # Should fall back to 1.0
        assert weight == 1.0

    def test_calibration_callback_error_handling(self):
        """Calibration callback errors are handled gracefully."""
        get_cal = MagicMock(side_effect=Exception("Cal error"))

        calculator = WeightCalculator(get_calibration_weight=get_cal)
        weight = calculator.get_weight("agent1")

        # Should fall back to 1.0
        assert weight == 1.0
