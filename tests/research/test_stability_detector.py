"""Tests for BetaBinomialStabilityDetector."""

import pytest
from aragora.debate.stability_detector import (
    BetaBinomialStabilityDetector,
    StabilityConfig,
    StabilityResult,
    create_stability_detector,
)


class TestStabilityDetector:
    """Test suite for BetaBinomialStabilityDetector."""

    def test_init_default_config(self) -> None:
        """Test initialization with default config."""
        detector = BetaBinomialStabilityDetector()
        assert detector.config.stability_threshold == 0.85
        assert detector.config.ks_threshold == 0.1
        assert detector.config.min_stable_rounds == 1

    def test_init_custom_config(self) -> None:
        """Test initialization with custom config."""
        config = StabilityConfig(
            stability_threshold=0.9,
            ks_threshold=0.05,
            min_stable_rounds=2,
        )
        detector = BetaBinomialStabilityDetector(config)
        assert detector.config.stability_threshold == 0.9
        assert detector.config.ks_threshold == 0.05

    def test_first_round_not_stable(self) -> None:
        """Test that first round always returns not stable."""
        detector = BetaBinomialStabilityDetector()
        votes = {"agent1": 0.8, "agent2": 0.2}

        result = detector.update(votes, round_num=1)

        assert not result.is_stable
        assert result.recommendation == "continue"
        assert result.rounds_since_stable == 0

    def test_stable_when_votes_consistent(self) -> None:
        """Test detection of stability when votes are consistent."""
        config = StabilityConfig(
            min_rounds_before_check=2,
            ks_threshold=0.15,
            stability_threshold=0.7,  # Lower for small sample sizes
            min_stable_rounds=1,
        )
        detector = BetaBinomialStabilityDetector(config)

        # Use enough agents so ks_2samp produces meaningful results
        # (with only 2 agents, ks_2samp always returns 0.5)
        votes = {f"agent{i}": 0.1 + 0.05 * i for i in range(10)}
        result1 = detector.update(votes, round_num=1)
        assert result1.recommendation == "continue"

        # Round 2 - identical votes
        result2 = detector.update(votes, round_num=2)

        # Should detect stability
        assert result2.ks_distance < 0.15
        assert result2.is_stable

    def test_not_stable_when_votes_change(self) -> None:
        """Test that changing votes are not stable."""
        config = StabilityConfig(min_rounds_before_check=2)
        detector = BetaBinomialStabilityDetector(config)

        # Round 1
        votes1 = {"agent1": 0.9, "agent2": 0.1}
        detector.update(votes1, round_num=1)

        # Round 2 - drastically different
        votes2 = {"agent1": 0.3, "agent2": 0.7}
        result2 = detector.update(votes2, round_num=2)

        assert result2.ks_distance > 0.1
        assert not result2.is_stable
        assert result2.recommendation == "continue"

    def test_muse_gating(self) -> None:
        """Test that high MUSE divergence gates stability."""
        config = StabilityConfig(
            min_rounds_before_check=2,
            muse_disagreement_gate=0.3,
            ks_threshold=0.15,
            stability_threshold=0.7,
        )
        detector = BetaBinomialStabilityDetector(config)

        # Use enough agents so ks_2samp works properly
        votes = {f"agent{i}": 0.1 + 0.05 * i for i in range(10)}
        detector.update(votes, round_num=1)

        # Round 2 - stable votes but high MUSE divergence
        result = detector.update(
            votes,
            round_num=2,
            muse_divergence=0.5,  # Above gate
        )

        assert result.muse_gated
        assert result.recommendation == "continue"

    def test_ascot_gating(self) -> None:
        """Test that high ASCoT fragility returns one_more_round."""
        config = StabilityConfig(
            min_rounds_before_check=2,
            ascot_fragility_gate=0.6,
            ks_threshold=0.15,
            stability_threshold=0.7,  # Lower for small sample sizes
        )
        detector = BetaBinomialStabilityDetector(config)

        # Use enough agents so ks_2samp produces meaningful KS distance
        votes = {f"agent{i}": 0.1 + 0.05 * i for i in range(10)}
        detector.update(votes, round_num=1)

        # Round 2 - stable (identical votes) but high fragility
        result = detector.update(
            votes,
            round_num=2,
            ascot_fragility=0.8,  # Above gate
        )

        assert result.ascot_gated
        assert result.recommendation == "one_more_round"

    def test_stop_after_min_stable_rounds(self) -> None:
        """Test stop recommendation after minimum stable rounds."""
        config = StabilityConfig(
            min_rounds_before_check=2,
            min_stable_rounds=2,
            ks_threshold=0.2,
        )
        detector = BetaBinomialStabilityDetector(config)

        # Use enough agents so ks_2samp produces meaningful KS distance
        votes = {f"agent{i}": 0.1 + 0.05 * i for i in range(10)}

        # Simulate consistent voting across rounds (identical each time)
        for round_num in range(1, 7):
            result = detector.update(votes, round_num=round_num)

        # After sufficient stable rounds, should recommend stop
        assert result.rounds_since_stable >= 2
        assert result.is_stable
        assert result.recommendation == "stop"

    def test_reset_clears_state(self) -> None:
        """Test that reset clears all internal state."""
        detector = BetaBinomialStabilityDetector()

        # Add some history
        detector.update({"agent1": 0.7}, round_num=1)
        detector.update({"agent1": 0.7}, round_num=2)

        # Reset
        detector.reset()

        # First update after reset should behave like first round
        result = detector.update({"agent1": 0.8}, round_num=1)
        assert not result.is_stable
        assert result.rounds_since_stable == 0

    def test_get_metrics(self) -> None:
        """Test metrics retrieval."""
        detector = BetaBinomialStabilityDetector()

        detector.update({"agent1": 0.7}, round_num=1)
        detector.update({"agent1": 0.7}, round_num=2)

        metrics = detector.get_metrics()

        assert metrics["total_rounds"] == 2
        assert "stability_scores" in metrics
        assert "avg_stability" in metrics

    def test_create_stability_detector_helper(self) -> None:
        """Test the convenience factory function."""
        detector = create_stability_detector(
            early_termination_threshold=0.9,
            min_rounds=3,
        )

        assert detector.config.stability_threshold == 0.9
        assert detector.config.min_rounds_before_check == 3

    def test_handles_empty_votes(self) -> None:
        """Test handling of empty vote dictionaries."""
        detector = BetaBinomialStabilityDetector()

        result1 = detector.update({}, round_num=1)
        assert result1.recommendation == "continue"

        result2 = detector.update({}, round_num=2)
        assert result2.ks_distance == 0.0

    def test_handles_new_agents(self) -> None:
        """Test handling when new agents appear in later rounds."""
        config = StabilityConfig(min_rounds_before_check=2)
        detector = BetaBinomialStabilityDetector(config)

        # Round 1 with 2 agents
        votes1 = {"agent1": 0.6, "agent2": 0.4}
        detector.update(votes1, round_num=1)

        # Round 2 with 3 agents
        votes2 = {"agent1": 0.5, "agent2": 0.3, "agent3": 0.2}
        result = detector.update(votes2, round_num=2)

        # Should handle gracefully
        assert result.ks_distance >= 0
        assert isinstance(result.recommendation, str)


class TestStabilityResult:
    """Test StabilityResult dataclass."""

    def test_default_values(self) -> None:
        """Test default values in StabilityResult."""
        result = StabilityResult(
            is_stable=True,
            stability_score=0.9,
            ks_distance=0.05,
            rounds_since_stable=2,
            recommendation="stop",
        )

        assert result.muse_gated is False
        assert result.ascot_gated is False
        assert result.beta_binomial_prob == 0.0
