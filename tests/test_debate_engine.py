"""
Unit tests for debate engine components.

Tests core debate logic in isolation:
- Consensus detection (VoteType, Evidence, Claim, ConsensusProof)
- Convergence detection (Jaccard, TF-IDF backends)
- Vote aggregation (AggregatedVotes, VoteAggregator)
- Confidence calculation and early termination
"""

from __future__ import annotations

import pytest
from collections import Counter
from datetime import datetime
from unittest.mock import patch, MagicMock


# =============================================================================
# Consensus Module Tests
# =============================================================================

class TestVoteType:
    """Tests for VoteType enum."""

    def test_vote_type_values(self):
        """Test all vote type values exist."""
        from aragora.debate.consensus import VoteType

        assert VoteType.AGREE.value == "agree"
        assert VoteType.DISAGREE.value == "disagree"
        assert VoteType.ABSTAIN.value == "abstain"
        assert VoteType.CONDITIONAL.value == "conditional"

    def test_vote_type_from_string(self):
        """Test creating VoteType from string."""
        from aragora.debate.consensus import VoteType

        assert VoteType("agree") == VoteType.AGREE
        assert VoteType("disagree") == VoteType.DISAGREE


class TestEvidence:
    """Tests for Evidence dataclass."""

    def test_evidence_creation(self):
        """Test creating Evidence instance."""
        from aragora.debate.consensus import Evidence

        evidence = Evidence(
            evidence_id="ev-1",
            source="agent-1",
            content="Supporting argument",
            evidence_type="argument",
            supports_claim=True,
            strength=0.8,
        )

        assert evidence.evidence_id == "ev-1"
        assert evidence.source == "agent-1"
        assert evidence.supports_claim is True
        assert evidence.strength == 0.8

    def test_evidence_default_timestamp(self):
        """Test Evidence gets default timestamp."""
        from aragora.debate.consensus import Evidence

        evidence = Evidence(
            evidence_id="ev-1",
            source="agent",
            content="Test",
            evidence_type="argument",
            supports_claim=True,
            strength=0.5,
        )

        assert evidence.timestamp is not None
        assert isinstance(evidence.timestamp, str)

    def test_evidence_default_metadata(self):
        """Test Evidence gets empty default metadata."""
        from aragora.debate.consensus import Evidence

        evidence = Evidence(
            evidence_id="ev-1",
            source="agent",
            content="Test",
            evidence_type="argument",
            supports_claim=True,
            strength=0.5,
        )

        assert evidence.metadata == {}


class TestClaim:
    """Tests for Claim dataclass."""

    def test_claim_creation(self):
        """Test creating Claim instance."""
        from aragora.debate.consensus import Claim

        claim = Claim(
            claim_id="c-1",
            statement="Test claim",
            author="agent-1",
            confidence=0.9,
        )

        assert claim.claim_id == "c-1"
        assert claim.statement == "Test claim"
        assert claim.author == "agent-1"
        assert claim.confidence == 0.9

    def test_claim_default_lists(self):
        """Test Claim gets empty default lists."""
        from aragora.debate.consensus import Claim

        claim = Claim(
            claim_id="c-1",
            statement="Test",
            author="agent",
            confidence=0.5,
        )

        assert claim.supporting_evidence == []
        assert claim.refuting_evidence == []

    def test_claim_net_evidence_strength_no_evidence(self):
        """Test net evidence strength with no evidence."""
        from aragora.debate.consensus import Claim

        claim = Claim(
            claim_id="c-1",
            statement="Test",
            author="agent",
            confidence=0.5,
        )

        assert claim.net_evidence_strength == 0.0

    def test_claim_net_evidence_strength_only_supporting(self):
        """Test net evidence strength with only supporting evidence."""
        from aragora.debate.consensus import Claim, Evidence

        claim = Claim(
            claim_id="c-1",
            statement="Test",
            author="agent",
            confidence=0.5,
            supporting_evidence=[
                Evidence(
                    evidence_id="e-1",
                    source="agent",
                    content="Support",
                    evidence_type="argument",
                    supports_claim=True,
                    strength=0.8,
                )
            ],
        )

        assert claim.net_evidence_strength == 1.0  # All supporting

    def test_claim_net_evidence_strength_mixed(self):
        """Test net evidence strength with mixed evidence."""
        from aragora.debate.consensus import Claim, Evidence

        claim = Claim(
            claim_id="c-1",
            statement="Test",
            author="agent",
            confidence=0.5,
            supporting_evidence=[
                Evidence(
                    evidence_id="e-1",
                    source="agent",
                    content="Support",
                    evidence_type="argument",
                    supports_claim=True,
                    strength=0.6,
                )
            ],
            refuting_evidence=[
                Evidence(
                    evidence_id="e-2",
                    source="agent",
                    content="Refute",
                    evidence_type="argument",
                    supports_claim=False,
                    strength=0.4,
                )
            ],
        )

        # (0.6 - 0.4) / 1.0 = 0.2
        assert claim.net_evidence_strength == pytest.approx(0.2, rel=0.01)


class TestConsensusVote:
    """Tests for ConsensusVote dataclass."""

    def test_consensus_vote_creation(self):
        """Test creating ConsensusVote instance."""
        from aragora.debate.consensus import ConsensusVote, VoteType

        vote = ConsensusVote(
            agent="agent-1",
            vote=VoteType.AGREE,
            confidence=0.9,
            reasoning="Strong argument",
        )

        assert vote.agent == "agent-1"
        assert vote.vote == VoteType.AGREE
        assert vote.confidence == 0.9
        assert vote.reasoning == "Strong argument"

    def test_consensus_vote_default_conditions(self):
        """Test ConsensusVote gets empty default conditions."""
        from aragora.debate.consensus import ConsensusVote, VoteType

        vote = ConsensusVote(
            agent="agent-1",
            vote=VoteType.CONDITIONAL,
            confidence=0.7,
            reasoning="With reservations",
        )

        assert vote.conditions == []

    def test_consensus_vote_with_conditions(self):
        """Test ConsensusVote with conditions."""
        from aragora.debate.consensus import ConsensusVote, VoteType

        vote = ConsensusVote(
            agent="agent-1",
            vote=VoteType.CONDITIONAL,
            confidence=0.7,
            reasoning="With reservations",
            conditions=["Must include tests", "Must have documentation"],
        )

        assert len(vote.conditions) == 2
        assert "Must include tests" in vote.conditions


class TestDissentRecord:
    """Tests for DissentRecord dataclass."""

    def test_dissent_record_creation(self):
        """Test creating DissentRecord instance."""
        from aragora.debate.consensus import DissentRecord

        dissent = DissentRecord(
            agent="agent-1",
            claim_id="c-1",
            dissent_type="partial",
            reasons=["Missing edge case handling"],
        )

        assert dissent.agent == "agent-1"
        assert dissent.dissent_type == "partial"
        assert len(dissent.reasons) == 1

    def test_dissent_record_severity_default(self):
        """Test DissentRecord default severity."""
        from aragora.debate.consensus import DissentRecord

        dissent = DissentRecord(
            agent="agent-1",
            claim_id="c-1",
            dissent_type="full",
            reasons=["Fundamental disagreement"],
        )

        assert dissent.severity == 0.5


class TestConsensusProof:
    """Tests for ConsensusProof dataclass."""

    def test_consensus_proof_creation(self):
        """Test creating ConsensusProof instance."""
        from aragora.debate.consensus import ConsensusProof

        proof = ConsensusProof(
            proof_id="p-1",
            debate_id="d-1",
            task="Test task",
            final_claim="Agreed solution",
            confidence=0.85,
            consensus_reached=True,
            votes=[],
            supporting_agents=["agent-1", "agent-2"],
            dissenting_agents=[],
            claims=[],
            dissents=[],
            unresolved_tensions=[],
            evidence_chain=[],
            reasoning_summary="All agents agreed",
        )

        assert proof.proof_id == "p-1"
        assert proof.consensus_reached is True
        assert proof.confidence == 0.85

    def test_consensus_proof_checksum(self):
        """Test ConsensusProof generates checksum."""
        from aragora.debate.consensus import ConsensusProof

        proof = ConsensusProof(
            proof_id="p-1",
            debate_id="d-1",
            task="Test task",
            final_claim="Agreed solution",
            confidence=0.85,
            consensus_reached=True,
            votes=[],
            supporting_agents=["agent-1"],
            dissenting_agents=[],
            claims=[],
            dissents=[],
            unresolved_tensions=[],
            evidence_chain=[],
            reasoning_summary="All agents agreed",
        )

        checksum = proof.checksum
        assert checksum is not None
        assert isinstance(checksum, str)
        assert len(checksum) == 16  # SHA-256 hex digest truncated to 16

    def test_consensus_proof_checksum_deterministic(self):
        """Test ConsensusProof checksum is deterministic."""
        from aragora.debate.consensus import ConsensusProof

        proof1 = ConsensusProof(
            proof_id="p-1",
            debate_id="d-1",
            task="Test task",
            final_claim="Same claim",
            confidence=0.85,
            consensus_reached=True,
            votes=[],
            supporting_agents=["agent-1"],
            dissenting_agents=[],
            claims=[],
            dissents=[],
            unresolved_tensions=[],
            evidence_chain=[],
            reasoning_summary="Summary",
        )

        proof2 = ConsensusProof(
            proof_id="p-2",  # Different ID
            debate_id="d-1",
            task="Test task",
            final_claim="Same claim",  # Same claim
            confidence=0.85,
            consensus_reached=True,
            votes=[],
            supporting_agents=["agent-1"],
            dissenting_agents=[],
            claims=[],
            dissents=[],
            unresolved_tensions=[],
            evidence_chain=[],
            reasoning_summary="Summary",
        )

        # Checksums should be the same since they're based on claim content
        assert proof1.checksum == proof2.checksum


# =============================================================================
# Convergence Module Tests
# =============================================================================

class TestJaccardBackend:
    """Tests for Jaccard similarity backend."""

    def test_jaccard_identical_texts(self):
        """Test Jaccard similarity with identical texts."""
        from aragora.debate.convergence import JaccardBackend

        backend = JaccardBackend()
        similarity = backend.compute_similarity("hello world", "hello world")
        assert similarity == 1.0

    def test_jaccard_completely_different(self):
        """Test Jaccard similarity with completely different texts."""
        from aragora.debate.convergence import JaccardBackend

        backend = JaccardBackend()
        similarity = backend.compute_similarity("apple orange", "car boat")
        assert similarity == 0.0

    def test_jaccard_partial_overlap(self):
        """Test Jaccard similarity with partial overlap."""
        from aragora.debate.convergence import JaccardBackend

        backend = JaccardBackend()
        # "hello" overlaps, "world" and "universe" don't
        similarity = backend.compute_similarity("hello world", "hello universe")
        # |{hello}| / |{hello, world, universe}| = 1/3
        assert similarity == pytest.approx(1/3, rel=0.01)

    def test_jaccard_case_insensitive(self):
        """Test Jaccard similarity is case insensitive."""
        from aragora.debate.convergence import JaccardBackend

        backend = JaccardBackend()
        similarity = backend.compute_similarity("Hello World", "hello world")
        assert similarity == 1.0

    def test_jaccard_empty_text(self):
        """Test Jaccard similarity with empty text."""
        from aragora.debate.convergence import JaccardBackend

        backend = JaccardBackend()
        assert backend.compute_similarity("", "test") == 0.0
        assert backend.compute_similarity("test", "") == 0.0
        assert backend.compute_similarity("", "") == 0.0

    def test_jaccard_batch_similarity_single_text(self):
        """Test batch similarity with single text returns 1.0."""
        from aragora.debate.convergence import JaccardBackend

        backend = JaccardBackend()
        similarity = backend.compute_batch_similarity(["just one"])
        assert similarity == 1.0

    def test_jaccard_batch_similarity_identical_texts(self):
        """Test batch similarity with identical texts."""
        from aragora.debate.convergence import JaccardBackend

        backend = JaccardBackend()
        texts = ["hello world", "hello world", "hello world"]
        similarity = backend.compute_batch_similarity(texts)
        assert similarity == 1.0

    def test_jaccard_batch_similarity_mixed(self):
        """Test batch similarity with mixed texts."""
        from aragora.debate.convergence import JaccardBackend

        backend = JaccardBackend()
        texts = [
            "the solution is caching",
            "caching is the solution",
            "we need caching",
        ]
        similarity = backend.compute_batch_similarity(texts)
        # Should have some overlap but not perfect
        assert 0.0 < similarity < 1.0


class TestConvergenceBackendSelection:
    """Tests for convergence backend selection."""

    def test_normalize_backend_name_valid(self):
        """Test normalizing valid backend names."""
        from aragora.debate.convergence import _normalize_backend_name

        assert _normalize_backend_name("auto") == "auto"
        assert _normalize_backend_name("jaccard") == "jaccard"
        assert _normalize_backend_name("tfidf") == "tfidf"

    def test_normalize_backend_name_aliases(self):
        """Test normalizing backend name aliases."""
        from aragora.debate.convergence import _normalize_backend_name

        assert _normalize_backend_name("sentence-transformers") == "sentence-transformer"
        assert _normalize_backend_name("sentence_transformers") == "sentence-transformer"
        assert _normalize_backend_name("tf-idf") == "tfidf"
        assert _normalize_backend_name("tf_idf") == "tfidf"

    def test_normalize_backend_name_invalid(self):
        """Test normalizing invalid backend names."""
        from aragora.debate.convergence import _normalize_backend_name

        assert _normalize_backend_name("invalid") is None
        assert _normalize_backend_name("") is None


# =============================================================================
# Vote Aggregation Tests
# =============================================================================

class TestAggregatedVotes:
    """Tests for AggregatedVotes dataclass."""

    def test_aggregated_votes_default_values(self):
        """Test AggregatedVotes default values."""
        from aragora.debate.phases.vote_aggregator import AggregatedVotes

        result = AggregatedVotes()
        assert result.vote_counts == Counter()
        assert result.total_weighted == 0.0
        assert result.total_votes == 0

    def test_get_winner_no_votes(self):
        """Test get_winner with no votes."""
        from aragora.debate.phases.vote_aggregator import AggregatedVotes

        result = AggregatedVotes()
        assert result.get_winner() is None

    def test_get_winner_with_votes(self):
        """Test get_winner with votes."""
        from aragora.debate.phases.vote_aggregator import AggregatedVotes

        result = AggregatedVotes(
            vote_counts=Counter({"option_a": 3, "option_b": 2}),
            total_weighted=5.0,
        )
        winner = result.get_winner()
        assert winner is not None
        assert winner[0] == "option_a"
        assert winner[1] == 3

    def test_get_confidence_no_votes(self):
        """Test get_confidence with no votes returns default."""
        from aragora.debate.phases.vote_aggregator import AggregatedVotes

        result = AggregatedVotes()
        assert result.get_confidence() == 0.5

    def test_get_confidence_with_votes(self):
        """Test get_confidence calculation."""
        from aragora.debate.phases.vote_aggregator import AggregatedVotes

        result = AggregatedVotes(
            vote_counts=Counter({"option_a": 4, "option_b": 1}),
            total_weighted=5.0,
        )
        # 4/5 = 0.8
        assert result.get_confidence() == pytest.approx(0.8)

    def test_get_vote_distribution(self):
        """Test get_vote_distribution calculation."""
        from aragora.debate.phases.vote_aggregator import AggregatedVotes

        result = AggregatedVotes(
            vote_counts=Counter({"option_a": 6, "option_b": 4}),
            total_weighted=10.0,
        )
        dist = result.get_vote_distribution()
        assert dist["option_a"] == pytest.approx(0.6)
        assert dist["option_b"] == pytest.approx(0.4)

    def test_get_vote_distribution_empty(self):
        """Test get_vote_distribution with no votes."""
        from aragora.debate.phases.vote_aggregator import AggregatedVotes

        result = AggregatedVotes()
        assert result.get_vote_distribution() == {}


class TestVoteAggregator:
    """Tests for VoteAggregator class."""

    def test_aggregator_initialization(self):
        """Test VoteAggregator initialization."""
        from aragora.debate.phases.vote_aggregator import VoteAggregator

        aggregator = VoteAggregator(user_vote_weight=0.7)
        assert aggregator._base_user_weight == 0.7

    def test_aggregator_default_weights(self):
        """Test VoteAggregator default weights."""
        from aragora.debate.phases.vote_aggregator import VoteAggregator

        aggregator = VoteAggregator()
        assert aggregator._base_user_weight == 0.5

    def test_aggregate_empty_votes(self):
        """Test aggregating empty votes list."""
        from aragora.debate.phases.vote_aggregator import VoteAggregator

        aggregator = VoteAggregator()
        result = aggregator.aggregate(votes=[])

        assert result.total_votes == 0
        assert result.get_winner() is None

    def test_aggregate_with_agent_votes(self):
        """Test aggregating agent votes."""
        from aragora.debate.phases.vote_aggregator import VoteAggregator
        from aragora.core import Vote

        votes = [
            Vote(agent="agent-1", choice="option_a", reasoning="Good option", confidence=0.9),
            Vote(agent="agent-2", choice="option_a", reasoning="I agree", confidence=0.8),
            Vote(agent="agent-3", choice="option_b", reasoning="Different view", confidence=0.7),
        ]

        aggregator = VoteAggregator()
        result = aggregator.aggregate(votes=votes)

        assert result.agent_votes_count == 3
        winner = result.get_winner()
        assert winner is not None
        assert winner[0] == "option_a"

    def test_aggregate_with_weights(self):
        """Test aggregating votes with custom weights."""
        from aragora.debate.phases.vote_aggregator import VoteAggregator
        from aragora.core import Vote

        votes = [
            Vote(agent="expert", choice="option_a", reasoning="Expert analysis", confidence=0.9),
            Vote(agent="novice", choice="option_b", reasoning="My guess", confidence=0.9),
        ]
        weights = {
            "expert": 2.0,  # Expert weight is doubled
            "novice": 1.0,
        }

        aggregator = VoteAggregator()
        result = aggregator.aggregate(votes=votes, weights=weights)

        # Expert's vote should outweigh novice
        winner = result.get_winner()
        assert winner is not None
        assert winner[0] == "option_a"


class TestTieBreaking:
    """Tests for vote tie breaking scenarios."""

    def test_tie_with_confidence(self):
        """Test tie breaking uses confidence."""
        from aragora.debate.phases.vote_aggregator import AggregatedVotes

        # With equal counts, confidence matters
        result = AggregatedVotes(
            vote_counts=Counter({"option_a": 2, "option_b": 2}),
            total_weighted=4.0,
        )

        # Should return the first most common (alphabetical in Counter)
        winner = result.get_winner()
        assert winner is not None
        assert winner[1] == 2  # Tied count


class TestUnresolvedTension:
    """Tests for UnresolvedTension dataclass."""

    def test_unresolved_tension_creation(self):
        """Test creating UnresolvedTension instance."""
        from aragora.debate.consensus import UnresolvedTension

        tension = UnresolvedTension(
            tension_id="t-1",
            description="Performance vs readability",
            agents_involved=["agent-1", "agent-2"],
            options=["Optimize for speed", "Optimize for clarity"],
            impact="Affects code review process",
        )

        assert tension.tension_id == "t-1"
        assert len(tension.agents_involved) == 2
        assert len(tension.options) == 2


# =============================================================================
# Convergence Detection Integration Tests
# =============================================================================

class TestConvergenceDetection:
    """Tests for convergence detection scenarios."""

    def test_detect_high_convergence(self):
        """Test detecting high convergence in similar positions."""
        from aragora.debate.convergence import JaccardBackend

        backend = JaccardBackend()
        positions = [
            "We should implement caching to improve performance",
            "Implementing a cache will improve the performance",
            "A caching layer would improve our performance",
        ]

        similarity = backend.compute_batch_similarity(positions)
        # Jaccard has moderate overlap with different word forms
        assert similarity > 0.2  # Lower threshold for Jaccard

    def test_detect_low_convergence(self):
        """Test detecting low convergence in different positions."""
        from aragora.debate.convergence import JaccardBackend

        backend = JaccardBackend()
        positions = [
            "We should use a SQL database",
            "NoSQL would be better for this use case",
            "A file-based approach is simpler",
        ]

        similarity = backend.compute_batch_similarity(positions)
        # Low overlap - different approaches
        assert similarity < 0.5


# =============================================================================
# Consensus Building Tests
# =============================================================================

class TestConsensusBuilding:
    """Tests for consensus building scenarios."""

    def test_majority_consensus(self):
        """Test majority consensus detection."""
        from aragora.debate.phases.vote_aggregator import VoteAggregator, AggregatedVotes
        from aragora.core import Vote

        votes = [
            Vote(agent="agent-1", choice="option_a", reasoning="Strong support", confidence=0.9),
            Vote(agent="agent-2", choice="option_a", reasoning="Agree", confidence=0.8),
            Vote(agent="agent-3", choice="option_a", reasoning="Makes sense", confidence=0.85),
            Vote(agent="agent-4", choice="option_b", reasoning="Different view", confidence=0.7),
        ]

        aggregator = VoteAggregator()
        result = aggregator.aggregate(votes=votes)

        winner = result.get_winner()
        confidence = result.get_confidence()

        assert winner[0] == "option_a"
        assert winner[1] == 3
        assert confidence >= 0.5  # Majority achieved

    def test_unanimous_consensus(self):
        """Test unanimous consensus detection."""
        from aragora.debate.phases.vote_aggregator import VoteAggregator
        from aragora.core import Vote

        votes = [
            Vote(agent="agent-1", choice="option_a", reasoning="Fully agree", confidence=0.95),
            Vote(agent="agent-2", choice="option_a", reasoning="Best option", confidence=0.9),
            Vote(agent="agent-3", choice="option_a", reasoning="Clear choice", confidence=0.85),
        ]

        aggregator = VoteAggregator()
        result = aggregator.aggregate(votes=votes)

        confidence = result.get_confidence()
        assert confidence == 1.0  # All votes for same option

    def test_no_consensus(self):
        """Test when no consensus is reached."""
        from aragora.debate.phases.vote_aggregator import VoteAggregator
        from aragora.core import Vote

        votes = [
            Vote(agent="agent-1", choice="option_a", reasoning="A is best", confidence=0.9),
            Vote(agent="agent-2", choice="option_b", reasoning="B is best", confidence=0.9),
            Vote(agent="agent-3", choice="option_c", reasoning="C is best", confidence=0.9),
        ]

        aggregator = VoteAggregator()
        result = aggregator.aggregate(votes=votes)

        confidence = result.get_confidence()
        # With 3 different choices, winner gets only 1/3
        assert confidence < 0.5


# =============================================================================
# Evidence Chain Tests
# =============================================================================

class TestEvidenceChain:
    """Tests for evidence chain tracking."""

    def test_evidence_chain_construction(self):
        """Test building evidence chain for consensus proof."""
        from aragora.debate.consensus import Evidence, ConsensusProof

        evidence_chain = [
            Evidence(
                evidence_id="e-1",
                source="agent-1",
                content="Initial analysis shows X",
                evidence_type="argument",
                supports_claim=True,
                strength=0.7,
            ),
            Evidence(
                evidence_id="e-2",
                source="agent-2",
                content="Data supports analysis X",
                evidence_type="data",
                supports_claim=True,
                strength=0.9,
            ),
        ]

        proof = ConsensusProof(
            proof_id="p-1",
            debate_id="d-1",
            task="Test task",
            final_claim="Conclusion based on evidence",
            confidence=0.85,
            consensus_reached=True,
            votes=[],
            supporting_agents=["agent-1", "agent-2"],
            dissenting_agents=[],
            claims=[],
            dissents=[],
            unresolved_tensions=[],
            evidence_chain=evidence_chain,
            reasoning_summary="Based on evidence chain",
        )

        assert len(proof.evidence_chain) == 2
        assert all(e.supports_claim for e in proof.evidence_chain)


# =============================================================================
# Confidence Calculation Tests
# =============================================================================

class TestConfidenceCalculation:
    """Tests for confidence score calculations."""

    def test_claim_confidence_from_evidence(self):
        """Test claim confidence based on evidence."""
        from aragora.debate.consensus import Claim, Evidence

        claim = Claim(
            claim_id="c-1",
            statement="Test claim",
            author="agent",
            confidence=0.0,  # Will be calculated
            supporting_evidence=[
                Evidence(
                    evidence_id="e-1",
                    source="agent",
                    content="Strong support",
                    evidence_type="argument",
                    supports_claim=True,
                    strength=0.9,
                ),
                Evidence(
                    evidence_id="e-2",
                    source="agent",
                    content="Weak support",
                    evidence_type="argument",
                    supports_claim=True,
                    strength=0.3,
                ),
            ],
        )

        # Net strength = (0.9 + 0.3) / (0.9 + 0.3) = 1.0 (all supporting)
        assert claim.net_evidence_strength == 1.0

    def test_vote_aggregation_confidence(self):
        """Test confidence from vote aggregation."""
        from aragora.debate.phases.vote_aggregator import AggregatedVotes

        # 80% vote for option A
        result = AggregatedVotes(
            vote_counts=Counter({"option_a": 8, "option_b": 2}),
            total_weighted=10.0,
        )

        assert result.get_confidence() == 0.8


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_vote(self):
        """Test aggregation with single vote."""
        from aragora.debate.phases.vote_aggregator import VoteAggregator
        from aragora.core import Vote

        votes = [Vote(agent="agent-1", choice="option_a", reasoning="Only vote", confidence=0.9)]

        aggregator = VoteAggregator()
        result = aggregator.aggregate(votes=votes)

        assert result.get_confidence() == 1.0
        assert result.get_winner() == ("option_a", 1)

    def test_zero_confidence_vote(self):
        """Test vote with zero confidence."""
        from aragora.debate.phases.vote_aggregator import VoteAggregator
        from aragora.core import Vote

        votes = [
            Vote(agent="agent-1", choice="option_a", reasoning="Unsure", confidence=0.0),
            Vote(agent="agent-2", choice="option_b", reasoning="Confident", confidence=0.9),
        ]

        aggregator = VoteAggregator()
        result = aggregator.aggregate(votes=votes)

        # Both votes count equally (confidence doesn't affect vote weight in base aggregator)
        assert result.agent_votes_count == 2

    def test_very_long_claim_statement(self):
        """Test claim with very long statement."""
        from aragora.debate.consensus import Claim

        long_statement = "A" * 10000

        claim = Claim(
            claim_id="c-1",
            statement=long_statement,
            author="agent",
            confidence=0.5,
        )

        assert len(claim.statement) == 10000

    def test_unicode_in_vote_choices(self):
        """Test Unicode characters in vote choices."""
        from aragora.debate.phases.vote_aggregator import VoteAggregator
        from aragora.core import Vote

        votes = [
            Vote(agent="agent-1", choice="选项一", reasoning="中文理由", confidence=0.9),
            Vote(agent="agent-2", choice="選項二", reasoning="繁體理由", confidence=0.8),
        ]

        aggregator = VoteAggregator()
        result = aggregator.aggregate(votes=votes)

        assert "选项一" in result.vote_counts
        assert "選項二" in result.vote_counts

    def test_empty_evidence_chain(self):
        """Test consensus proof with empty evidence chain."""
        from aragora.debate.consensus import ConsensusProof

        proof = ConsensusProof(
            proof_id="p-1",
            debate_id="d-1",
            task="Test",
            final_claim="Claim",
            confidence=0.5,
            consensus_reached=True,
            votes=[],
            supporting_agents=[],
            dissenting_agents=[],
            claims=[],
            dissents=[],
            unresolved_tensions=[],
            evidence_chain=[],
            reasoning_summary="No evidence",
        )

        assert proof.evidence_chain == []
        # Should still generate checksum
        assert proof.checksum is not None
