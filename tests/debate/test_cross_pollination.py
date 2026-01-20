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

        verifier.adjust_vote_confidence_from_verification(
            votes, verification_results, proposals
        )

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

        verifier.adjust_vote_confidence_from_verification(
            votes, verification_results, proposals
        )

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


class TestKnowledgeMoundThreshold:
    """Test knowledge mound confidence threshold."""

    def test_knowledge_ops_uses_high_threshold(self):
        """KnowledgeMoundOperations uses 0.85 confidence threshold."""
        # Verify the threshold by checking the source
        import inspect
        from aragora.debate.knowledge_mound_ops import KnowledgeMoundOperations

        source = inspect.getsource(KnowledgeMoundOperations.ingest_debate_outcome)
        assert "0.85" in source or "0.85" in source
