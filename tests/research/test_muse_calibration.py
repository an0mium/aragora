"""Tests for MUSECalculator."""

import pytest
from aragora.ranking.muse_calibration import (
    MUSECalculator,
    MUSEConfig,
    MUSEResult,
    apply_muse_to_votes,
)


class TestMUSECalculator:
    """Test suite for MUSECalculator."""

    def test_init_default_config(self) -> None:
        """Test initialization with default config."""
        calculator = MUSECalculator()
        assert calculator.config.min_subset_size == 2
        assert calculator.config.max_subset_size == 5
        assert calculator.config.default_brier == 0.5

    def test_init_custom_config(self) -> None:
        """Test initialization with custom config."""
        config = MUSEConfig(
            min_subset_size=3,
            max_subset_size=7,
            default_brier=0.3,
        )
        calculator = MUSECalculator(config)
        assert calculator.config.min_subset_size == 3
        assert calculator.config.max_subset_size == 7

    def test_empty_responses(self) -> None:
        """Test handling of empty responses."""
        calculator = MUSECalculator()
        result = calculator.calculate_ensemble_uncertainty({}, {})

        assert result.consensus_confidence == 0.0
        assert result.divergence_score == 1.0
        assert len(result.best_subset) == 0

    def test_single_agent(self) -> None:
        """Test handling of single agent response."""
        calculator = MUSECalculator()
        responses = {
            "agent1": {"answer": "A", "confidence": 0.8},
        }

        result = calculator.calculate_ensemble_uncertainty(responses, {})

        assert result.consensus_confidence == 0.8
        assert result.divergence_score == 0.0
        assert "agent1" in result.best_subset

    def test_agreeing_agents(self) -> None:
        """Test high confidence when agents agree."""
        calculator = MUSECalculator()
        responses = {
            "agent1": {"answer": "A", "confidence": 0.9, "distribution": [0.9, 0.1]},
            "agent2": {"answer": "A", "confidence": 0.85, "distribution": [0.85, 0.15]},
            "agent3": {"answer": "A", "confidence": 0.88, "distribution": [0.88, 0.12]},
        }

        result = calculator.calculate_ensemble_uncertainty(responses, {})

        # High agreement should result in high confidence
        assert result.consensus_confidence > 0.7
        assert result.divergence_score < 0.3

    def test_disagreeing_agents(self) -> None:
        """Test low confidence when agents disagree."""
        calculator = MUSECalculator()
        responses = {
            "agent1": {"answer": "A", "confidence": 0.9, "distribution": [0.9, 0.1]},
            "agent2": {"answer": "B", "confidence": 0.9, "distribution": [0.1, 0.9]},
        }

        result = calculator.calculate_ensemble_uncertainty(responses, {})

        # High disagreement should result in lower confidence
        assert result.divergence_score > 0.5

    def test_calibration_affects_subset(self) -> None:
        """Test that historical calibration affects subset selection."""
        calculator = MUSECalculator()
        responses = {
            "agent1": {"answer": "A", "confidence": 0.9, "distribution": [0.9, 0.1]},
            "agent2": {"answer": "A", "confidence": 0.85, "distribution": [0.85, 0.15]},
            "agent3": {"answer": "A", "confidence": 0.88, "distribution": [0.88, 0.12]},
        }

        # agent1 has best calibration (lowest Brier)
        calibration = {
            "agent1": 0.1,  # Best calibrated
            "agent2": 0.5,  # Poorly calibrated
            "agent3": 0.2,  # Well calibrated
        }

        result = calculator.calculate_ensemble_uncertainty(responses, calibration)

        # Best calibrated agents should be in subset
        assert "agent1" in result.best_subset
        assert result.subset_brier_score < 0.5

    def test_distribution_normalization(self) -> None:
        """Test that distributions are properly normalized."""
        calculator = MUSECalculator()
        responses = {
            "agent1": {"answer": "A", "confidence": 0.9, "distribution": [0.9, 0.1]},
            "agent2": {"answer": "A", "confidence": 0.8, "distribution": [0.8]},  # Short
        }

        # Should not raise, should handle normalization
        result = calculator.calculate_ensemble_uncertainty(responses, {})
        assert isinstance(result.divergence_score, float)

    def test_update_calibration(self) -> None:
        """Test calibration history updates."""
        calculator = MUSECalculator()

        # Agent makes correct prediction
        calculator.update_calibration("agent1", predicted_confidence=0.8, actual_outcome=1.0)

        # Agent makes incorrect prediction
        calculator.update_calibration("agent1", predicted_confidence=0.9, actual_outcome=0.0)

        scores = calculator.get_calibration_scores()
        assert "agent1" in scores
        # Brier score should reflect both predictions
        assert 0 < scores["agent1"] < 1

    def test_reset_history(self) -> None:
        """Test calibration history reset."""
        calculator = MUSECalculator()

        calculator.update_calibration("agent1", 0.8, 1.0)
        assert len(calculator.get_calibration_scores()) > 0

        calculator.reset_history()
        assert len(calculator.get_calibration_scores()) == 0

    def test_individual_divergences(self) -> None:
        """Test that individual divergences are calculated."""
        calculator = MUSECalculator()
        responses = {
            "agent1": {"answer": "A", "confidence": 0.9, "distribution": [0.9, 0.1]},
            "agent2": {"answer": "A", "confidence": 0.6, "distribution": [0.6, 0.4]},
            "agent3": {"answer": "B", "confidence": 0.8, "distribution": [0.2, 0.8]},
        }

        result = calculator.calculate_ensemble_uncertainty(responses, {})

        # Should have individual divergences for subset members
        assert len(result.individual_divergences) > 0

    def test_subset_agreement(self) -> None:
        """Test subset agreement calculation."""
        calculator = MUSECalculator()
        responses = {
            "agent1": {"answer": "A", "confidence": 0.9},
            "agent2": {"answer": "A", "confidence": 0.91},
        }

        result = calculator.calculate_ensemble_uncertainty(responses, {})

        # Very similar confidences should give high agreement
        assert result.subset_agreement > 0.9


class TestApplyMuseToVotes:
    """Test apply_muse_to_votes function."""

    def test_boosts_subset_members(self) -> None:
        """Test that subset members get weight boost."""
        votes = [
            {"agent_id": "agent1", "weight": 1.0},
            {"agent_id": "agent2", "weight": 1.0},
            {"agent_id": "agent3", "weight": 1.0},
        ]

        muse_result = MUSEResult(
            consensus_confidence=0.9,
            divergence_score=0.1,
            best_subset={"agent1", "agent3"},
            subset_agreement=0.95,
            subset_brier_score=0.15,
        )

        adjusted = apply_muse_to_votes(votes, muse_result, muse_weight=0.15)

        # Subset members should be boosted
        agent1_vote = next(v for v in adjusted if v["agent_id"] == "agent1")
        agent2_vote = next(v for v in adjusted if v["agent_id"] == "agent2")
        agent3_vote = next(v for v in adjusted if v["agent_id"] == "agent3")

        assert agent1_vote["weight"] > 1.0
        assert agent2_vote["weight"] == 1.0  # Not in subset
        assert agent3_vote["weight"] > 1.0
        assert agent1_vote["muse_boosted"] is True
        assert agent2_vote["muse_boosted"] is False

    def test_preserves_original_votes(self) -> None:
        """Test that original vote list is not modified."""
        votes = [{"agent_id": "agent1", "weight": 1.0}]
        original_weight = votes[0]["weight"]

        muse_result = MUSEResult(
            consensus_confidence=0.9,
            divergence_score=0.1,
            best_subset={"agent1"},
            subset_agreement=0.95,
            subset_brier_score=0.15,
        )

        apply_muse_to_votes(votes, muse_result, muse_weight=0.15)

        # Original should be unchanged
        assert votes[0]["weight"] == original_weight
        assert "muse_boosted" not in votes[0]


class TestMUSEResult:
    """Test MUSEResult dataclass."""

    def test_default_individual_divergences(self) -> None:
        """Test default value for individual_divergences."""
        result = MUSEResult(
            consensus_confidence=0.9,
            divergence_score=0.1,
            best_subset={"agent1"},
            subset_agreement=0.95,
            subset_brier_score=0.15,
        )

        assert result.individual_divergences == {}
