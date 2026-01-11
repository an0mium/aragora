"""Tests for WeightCalculator class."""

import pytest
from unittest.mock import MagicMock, patch

from aragora.debate.phases.weight_calculator import (
    WeightCalculator,
    WeightFactors,
    WeightCalculatorConfig,
)


class MockAgent:
    """Mock agent for testing."""

    def __init__(self, name: str):
        self.name = name


class TestWeightFactors:
    """Tests for WeightFactors dataclass."""

    def test_default_values(self):
        """Test default weight factor values."""
        factors = WeightFactors()
        assert factors.reputation == 1.0
        assert factors.reliability == 1.0
        assert factors.consistency == 1.0
        assert factors.calibration == 1.0

    def test_total_with_defaults(self):
        """Test total weight with default factors."""
        factors = WeightFactors()
        assert factors.total == 1.0

    def test_total_with_custom_values(self):
        """Test total weight calculation with custom values."""
        factors = WeightFactors(
            reputation=1.2,
            reliability=0.8,
            consistency=0.9,
            calibration=1.1,
        )
        expected = 1.2 * 0.8 * 0.9 * 1.1
        assert abs(factors.total - expected) < 0.001

    def test_total_with_zero_factor(self):
        """Test total weight when one factor is zero."""
        factors = WeightFactors(reliability=0.0)
        assert factors.total == 0.0


class TestWeightCalculatorConfig:
    """Tests for WeightCalculatorConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = WeightCalculatorConfig()
        assert config.enable_reputation is True
        assert config.enable_reliability is True
        assert config.enable_consistency is True
        assert config.enable_calibration is True
        assert config.min_weight == 0.1
        assert config.max_weight == 5.0

    def test_custom_config(self):
        """Test custom configuration."""
        config = WeightCalculatorConfig(
            enable_reputation=False,
            min_weight=0.5,
            max_weight=2.0,
        )
        assert config.enable_reputation is False
        assert config.min_weight == 0.5
        assert config.max_weight == 2.0


class TestWeightCalculator:
    """Tests for WeightCalculator class."""

    def test_init_with_no_dependencies(self):
        """Test initialization with no dependencies."""
        calculator = WeightCalculator()
        assert calculator.memory is None
        assert calculator.elo_system is None
        assert calculator.flip_detector is None
        assert calculator.agent_weights == {}

    def test_init_with_dependencies(self):
        """Test initialization with dependencies."""
        memory = MagicMock()
        elo = MagicMock()
        calculator = WeightCalculator(memory=memory, elo_system=elo)
        assert calculator.memory is memory
        assert calculator.elo_system is elo

    def test_compute_weights_empty_agents(self):
        """Test compute_weights with empty agent list."""
        calculator = WeightCalculator()
        weights = calculator.compute_weights([])
        assert weights == {}

    def test_compute_weights_single_agent(self):
        """Test compute_weights with single agent."""
        calculator = WeightCalculator()
        agents = [MockAgent("agent1")]
        weights = calculator.compute_weights(agents)
        assert "agent1" in weights
        assert weights["agent1"] == 1.0

    def test_compute_weights_multiple_agents(self):
        """Test compute_weights with multiple agents."""
        calculator = WeightCalculator()
        agents = [MockAgent("a"), MockAgent("b"), MockAgent("c")]
        weights = calculator.compute_weights(agents)
        assert len(weights) == 3
        assert all(w == 1.0 for w in weights.values())

    def test_get_weight_unknown_agent(self):
        """Test get_weight for agent not in cache."""
        calculator = WeightCalculator()
        weight = calculator.get_weight("unknown_agent")
        assert weight == 1.0

    def test_get_weight_with_factors(self):
        """Test get_weight_with_factors returns breakdown."""
        calculator = WeightCalculator()
        weight, factors = calculator.get_weight_with_factors("agent1")
        assert weight == 1.0
        assert isinstance(factors, WeightFactors)
        assert factors.reputation == 1.0

    def test_reputation_weight_from_memory(self):
        """Test reputation weight extraction from memory."""
        memory = MagicMock()
        memory.get_vote_weight.return_value = 1.3

        calculator = WeightCalculator(memory=memory)
        weight = calculator.get_weight("agent1")

        memory.get_vote_weight.assert_called_with("agent1")
        assert weight == 1.3

    def test_reliability_weight_from_agent_weights(self):
        """Test reliability weight from pre-computed weights."""
        agent_weights = {"agent1": 0.8, "agent2": 0.9}
        calculator = WeightCalculator(agent_weights=agent_weights)

        weight1 = calculator.get_weight("agent1")
        weight2 = calculator.get_weight("agent2")
        weight3 = calculator.get_weight("agent3")

        assert weight1 == 0.8
        assert weight2 == 0.9
        assert weight3 == 1.0  # Not in weights, default

    def test_consistency_weight_from_flip_detector(self):
        """Test consistency weight from FlipDetector."""
        flip_detector = MagicMock()
        consistency = MagicMock()
        consistency.consistency_score = 0.8
        flip_detector.get_agent_consistency.return_value = consistency

        calculator = WeightCalculator(flip_detector=flip_detector)
        weight = calculator.get_weight("agent1")

        flip_detector.get_agent_consistency.assert_called_with("agent1")
        # Expected: 0.5 + (0.8 * 0.5) = 0.9
        assert abs(weight - 0.9) < 0.001

    def test_calibration_weight_from_callback(self):
        """Test calibration weight from callback."""
        get_cal_weight = MagicMock(return_value=1.2)
        calculator = WeightCalculator(get_calibration_weight=get_cal_weight)

        weight = calculator.get_weight("agent1")

        get_cal_weight.assert_called_with("agent1")
        assert weight == 1.2

    def test_calibration_weight_from_elo_cache(self):
        """Test calibration weight from ELO ratings cache."""
        elo_system = MagicMock()
        rating = MagicMock()
        rating.calibration_score = 0.7
        elo_system.get_ratings_batch.return_value = {"agent1": rating}

        calculator = WeightCalculator(elo_system=elo_system)
        agents = [MockAgent("agent1")]
        weights = calculator.compute_weights(agents)

        # Expected: 0.5 + 0.7 = 1.2
        assert abs(weights["agent1"] - 1.2) < 0.001

    def test_combined_weights(self):
        """Test weight calculation with all factors."""
        memory = MagicMock()
        memory.get_vote_weight.return_value = 1.2  # reputation

        agent_weights = {"agent1": 0.8}  # reliability

        flip_detector = MagicMock()
        consistency = MagicMock()
        consistency.consistency_score = 0.6  # -> 0.5 + 0.3 = 0.8
        flip_detector.get_agent_consistency.return_value = consistency

        get_cal_weight = MagicMock(return_value=1.1)  # calibration

        calculator = WeightCalculator(
            memory=memory,
            agent_weights=agent_weights,
            flip_detector=flip_detector,
            get_calibration_weight=get_cal_weight,
        )

        weight, factors = calculator.get_weight_with_factors("agent1")

        assert factors.reputation == 1.2
        assert factors.reliability == 0.8
        assert abs(factors.consistency - 0.8) < 0.001
        assert factors.calibration == 1.1

        expected = 1.2 * 0.8 * 0.8 * 1.1
        assert abs(weight - expected) < 0.001

    def test_weight_bounds_min(self):
        """Test weight is clamped to minimum."""
        agent_weights = {"agent1": 0.01}  # Very low
        calculator = WeightCalculator(agent_weights=agent_weights)

        weight = calculator.get_weight("agent1")
        assert weight == 0.1  # min_weight

    def test_weight_bounds_max(self):
        """Test weight is clamped to maximum."""
        memory = MagicMock()
        memory.get_vote_weight.return_value = 3.0

        flip_detector = MagicMock()
        consistency = MagicMock()
        consistency.consistency_score = 1.0
        flip_detector.get_agent_consistency.return_value = consistency

        get_cal_weight = MagicMock(return_value = 2.0)

        calculator = WeightCalculator(
            memory=memory,
            flip_detector=flip_detector,
            get_calibration_weight=get_cal_weight,
        )

        weight = calculator.get_weight("agent1")
        assert weight == 5.0  # max_weight

    def test_disabled_factors(self):
        """Test that disabled factors return 1.0."""
        memory = MagicMock()
        memory.get_vote_weight.return_value = 1.5

        config = WeightCalculatorConfig(enable_reputation=False)
        calculator = WeightCalculator(memory=memory, config=config)

        weight, factors = calculator.get_weight_with_factors("agent1")

        # Reputation should be default 1.0 since disabled
        assert factors.reputation == 1.0
        assert weight == 1.0

    def test_clear_cache(self):
        """Test cache clearing."""
        elo_system = MagicMock()
        rating = MagicMock()
        rating.calibration_score = 0.5
        elo_system.get_ratings_batch.return_value = {"agent1": rating}

        calculator = WeightCalculator(elo_system=elo_system)
        agents = [MockAgent("agent1")]
        calculator.compute_weights(agents)

        assert len(calculator._ratings_cache) == 1

        calculator.clear_cache()

        assert len(calculator._ratings_cache) == 0

    def test_error_handling_memory(self):
        """Test graceful handling of memory errors."""
        memory = MagicMock()
        memory.get_vote_weight.side_effect = Exception("Memory error")

        calculator = WeightCalculator(memory=memory)
        weight = calculator.get_weight("agent1")

        # Should return default weight on error
        assert weight == 1.0

    def test_error_handling_flip_detector(self):
        """Test graceful handling of flip detector errors."""
        flip_detector = MagicMock()
        flip_detector.get_agent_consistency.side_effect = Exception("Flip error")

        calculator = WeightCalculator(flip_detector=flip_detector)
        weight = calculator.get_weight("agent1")

        assert weight == 1.0

    def test_error_handling_calibration_callback(self):
        """Test graceful handling of calibration callback errors."""
        get_cal_weight = MagicMock(side_effect=Exception("Cal error"))
        calculator = WeightCalculator(get_calibration_weight=get_cal_weight)

        weight = calculator.get_weight("agent1")

        assert weight == 1.0

    def test_batch_prefetch_error(self):
        """Test graceful handling of batch prefetch errors."""
        elo_system = MagicMock()
        elo_system.get_ratings_batch.side_effect = Exception("Batch error")

        calculator = WeightCalculator(elo_system=elo_system)
        agents = [MockAgent("agent1")]
        weights = calculator.compute_weights(agents)

        # Should still return weights with defaults
        assert weights["agent1"] == 1.0
