"""
Integration tests for cross-pollination features.

Tests the integrations between:
- ELO skill weighting → vote weights
- Evidence quality → consensus weights
- Memory tiers → debate strategy
- Verification → vote confidence
- Voting accuracy → ELO scoring
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, AsyncMock
from dataclasses import dataclass


class TestDebateStrategy:
    """Test memory-based debate strategy."""

    def test_strategy_estimates_rounds_without_memory(self):
        """Strategy returns default when no memory available."""
        from aragora.debate.strategy import DebateStrategy, StrategyRecommendation

        strategy = DebateStrategy(continuum_memory=None)
        result = strategy.estimate_rounds("test task")

        assert isinstance(result, StrategyRecommendation)
        assert result.estimated_rounds == strategy.exploration_rounds
        assert result.confidence == 0.0
        assert "No memory system available" in result.reasoning

    def test_strategy_recommendation_dataclass(self):
        """StrategyRecommendation has required fields."""
        from aragora.debate.strategy import StrategyRecommendation

        rec = StrategyRecommendation(
            estimated_rounds=3,
            confidence=0.85,
            reasoning="Test reasoning",
            relevant_memories=["mem1", "mem2"],
        )

        assert rec.estimated_rounds == 3
        assert rec.confidence == 0.85
        assert rec.reasoning == "Test reasoning"
        assert rec.relevant_memories == ["mem1", "mem2"]


class TestEloSkillWeighting:
    """Test ELO skill weighting in vote weights."""

    def test_weight_factors_includes_elo_skill(self):
        """WeightFactors dataclass includes elo_skill field."""
        from aragora.debate.phases.weight_calculator import WeightFactors

        factors = WeightFactors()
        assert hasattr(factors, "elo_skill")
        assert factors.elo_skill == 1.0

    def test_weight_calculator_config_has_elo_settings(self):
        """WeightCalculatorConfig has ELO configuration."""
        from aragora.debate.phases.weight_calculator import WeightCalculatorConfig

        config = WeightCalculatorConfig()
        assert hasattr(config, "enable_elo_skill")
        assert hasattr(config, "elo_baseline")
        assert hasattr(config, "elo_scale")
        assert config.enable_elo_skill is True
        assert config.elo_baseline == 1500.0


class TestRLMHierarchyCache:
    """Test RLM compression caching."""

    def test_cache_initialization(self):
        """RLMHierarchyCache initializes correctly."""
        from aragora.rlm.bridge import RLMHierarchyCache

        cache = RLMHierarchyCache(knowledge_mound=None)
        assert cache._local_cache == {}
        assert cache._cache_hits == 0
        assert cache._cache_misses == 0

    def test_cache_stats(self):
        """Cache provides statistics."""
        from aragora.rlm.bridge import RLMHierarchyCache

        cache = RLMHierarchyCache()
        stats = cache.stats

        assert "hits" in stats
        assert "misses" in stats
        assert "hit_rate" in stats
        assert "local_cache_size" in stats

    def test_cache_hash_computation(self):
        """Cache computes consistent hashes."""
        from aragora.rlm.bridge import RLMHierarchyCache

        cache = RLMHierarchyCache()
        hash1 = cache._compute_task_hash("test content", "text")
        hash2 = cache._compute_task_hash("test content", "text")
        hash3 = cache._compute_task_hash("different content", "text")

        assert hash1 == hash2  # Same content = same hash
        assert hash1 != hash3  # Different content = different hash


class TestVerificationConfidenceAdjustment:
    """Test verification → vote confidence adjustment."""

    def test_consensus_verifier_has_adjustment_method(self):
        """ConsensusVerifier has adjust_vote_confidence_from_verification method."""
        from aragora.debate.phases.consensus_verification import ConsensusVerifier

        verifier = ConsensusVerifier()
        assert hasattr(verifier, "adjust_vote_confidence_from_verification")
        assert callable(verifier.adjust_vote_confidence_from_verification)

    def test_adjustment_boosts_verified_vote_confidence(self):
        """Verified proposals boost vote confidence."""
        from aragora.debate.phases.consensus_verification import ConsensusVerifier

        @dataclass
        class MockVote:
            choice: str
            confidence: float

        verifier = ConsensusVerifier()
        votes = [MockVote(choice="agent_a", confidence=0.7)]
        verification_results = {"agent_a": {"verified": 1, "disproven": 0}}
        proposals = {"agent_a": "proposal text"}

        verifier.adjust_vote_confidence_from_verification(votes, verification_results, proposals)

        # Confidence should be boosted (0.7 * 1.3 = 0.91)
        assert votes[0].confidence > 0.7
        assert votes[0].confidence <= 0.99

    def test_adjustment_reduces_disproven_vote_confidence(self):
        """Disproven proposals reduce vote confidence."""
        from aragora.debate.phases.consensus_verification import ConsensusVerifier

        @dataclass
        class MockVote:
            choice: str
            confidence: float

        verifier = ConsensusVerifier()
        votes = [MockVote(choice="agent_b", confidence=0.8)]
        verification_results = {"agent_b": {"verified": 0, "disproven": 1}}
        proposals = {"agent_b": "proposal text"}

        verifier.adjust_vote_confidence_from_verification(votes, verification_results, proposals)

        # Confidence should be reduced (0.8 * 0.3 = 0.24)
        assert votes[0].confidence < 0.8
        assert votes[0].confidence >= 0.01


class TestVotingAccuracyTracking:
    """Test voting accuracy → ELO scoring."""

    def test_elo_system_has_voting_accuracy_methods(self):
        """EloSystem has voting accuracy tracking methods."""
        import tempfile
        from aragora.ranking.elo import EloSystem

        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            elo = EloSystem(db_path=f.name)
            assert hasattr(elo, "update_voting_accuracy")
            assert hasattr(elo, "get_voting_accuracy")
            assert callable(elo.update_voting_accuracy)
            assert callable(elo.get_voting_accuracy)

    def test_voting_accuracy_updates(self):
        """Voting accuracy updates agent stats."""
        import tempfile
        from aragora.ranking.elo import EloSystem

        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            elo = EloSystem(db_path=f.name)

            # Record some voting accuracy
            elo.update_voting_accuracy(
                agent_name="test_agent",
                voted_for_consensus=True,
                domain="general",
                apply_elo_bonus=False,  # Disable bonus to test pure tracking
            )

            stats = elo.get_voting_accuracy("test_agent")
            assert stats["total_votes"] == 1
            assert stats["correct_votes"] == 1
            assert stats["accuracy"] == 1.0


class TestCalibrationToProposals:
    """Test calibration → proposal confidence scaling."""

    def test_proposal_phase_has_calibration_tracker(self):
        """ProposalPhase accepts calibration_tracker parameter."""
        from aragora.debate.phases.proposal_phase import ProposalPhase

        tracker = MagicMock()
        phase = ProposalPhase(calibration_tracker=tracker)
        assert phase.calibration_tracker is tracker

    def test_get_calibrated_confidence_without_tracker(self):
        """Returns raw confidence when no tracker available."""
        from aragora.debate.phases.proposal_phase import ProposalPhase

        phase = ProposalPhase()
        ctx = MagicMock()

        result = phase._get_calibrated_confidence("test_agent", 0.7, ctx)
        assert result == 0.7

    def test_get_calibrated_confidence_with_tracker(self):
        """Applies calibration when tracker available."""
        from aragora.debate.phases.proposal_phase import ProposalPhase

        # Create mock calibration summary with enough predictions
        summary = MagicMock()
        summary.total_predictions = 50
        summary.adjust_confidence.return_value = 0.55  # Scaled down (overconfident agent)

        tracker = MagicMock()
        tracker.get_calibration_summary.return_value = summary

        phase = ProposalPhase(calibration_tracker=tracker)
        ctx = MagicMock()
        ctx.domain = "testing"

        result = phase._get_calibrated_confidence("test_agent", 0.7, ctx)
        assert result == 0.55
        summary.adjust_confidence.assert_called_once_with(0.7, domain="testing")

    def test_calibration_skipped_with_insufficient_data(self):
        """Skips calibration when insufficient prediction history."""
        from aragora.debate.phases.proposal_phase import ProposalPhase

        # Summary with too few predictions
        summary = MagicMock()
        summary.total_predictions = 5  # Less than 10 threshold
        summary.adjust_confidence.return_value = 0.55

        tracker = MagicMock()
        tracker.get_calibration_summary.return_value = summary

        phase = ProposalPhase(calibration_tracker=tracker)
        ctx = MagicMock()

        result = phase._get_calibrated_confidence("test_agent", 0.7, ctx)
        assert result == 0.7  # Returns raw, doesn't call adjust
        summary.adjust_confidence.assert_not_called()


class TestLearningEfficiency:
    """Test debate outcomes → learning efficiency tracking."""

    def test_elo_system_has_learning_efficiency_methods(self):
        """EloSystem has learning efficiency tracking methods."""
        import tempfile
        from aragora.ranking.elo import EloSystem

        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            elo = EloSystem(db_path=f.name)
            assert hasattr(elo, "get_learning_efficiency")
            assert hasattr(elo, "apply_learning_bonus")
            assert callable(elo.get_learning_efficiency)
            assert callable(elo.apply_learning_bonus)

    def test_learning_efficiency_with_insufficient_data(self):
        """Learning efficiency returns insufficient_data with few debates."""
        import tempfile
        from aragora.ranking.elo import EloSystem

        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            elo = EloSystem(db_path=f.name)
            efficiency = elo.get_learning_efficiency("new_agent")

            assert efficiency["learning_category"] == "insufficient_data"
            assert efficiency["has_meaningful_data"] is False
            assert efficiency["elo_gain_rate"] == 0.0

    def test_learning_efficiency_computation(self):
        """Learning efficiency computes metrics from ELO history."""
        import tempfile
        from aragora.ranking.elo import EloSystem

        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            elo = EloSystem(db_path=f.name)

            # Simulate improving agent by recording multiple matches
            for i in range(6):
                participants = ["improving_agent", f"opponent_{i}"]
                elo.record_match(
                    debate_id=f"debate_{i}",
                    winner="improving_agent",
                    participants=participants,
                    domain="testing",
                    scores={"improving_agent": 0.8, f"opponent_{i}": 0.2},
                )

            efficiency = elo.get_learning_efficiency("improving_agent", domain="testing")

            assert "elo_gain_rate" in efficiency
            assert "consistency_score" in efficiency
            assert "learning_category" in efficiency
            # Should have positive gain rate after consistent wins
            assert efficiency["elo_gain_rate"] >= 0

    def test_feedback_phase_applies_learning_bonuses(self):
        """FeedbackPhase has learning bonus method."""
        from aragora.debate.phases.feedback_phase import FeedbackPhase

        phase = FeedbackPhase(elo_system=MagicMock())
        assert hasattr(phase, "_apply_learning_bonuses")
        assert callable(phase._apply_learning_bonuses)


class TestKnowledgeMoundThreshold:
    """Test knowledge mound confidence threshold."""

    def test_knowledge_ops_uses_high_threshold(self):
        """KnowledgeMoundOperations uses 0.85 confidence threshold."""
        # Verify the threshold by checking the source
        import inspect
        from aragora.debate.knowledge_mound_ops import KnowledgeMoundOperations

        source = inspect.getsource(KnowledgeMoundOperations.ingest_debate_outcome)
        assert "0.85" in source or "0.85" in source
