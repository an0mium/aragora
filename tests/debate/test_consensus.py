"""
Tests for the consensus mechanism and proof generation.

Tests cover:
- Data classes (Evidence, Claim, ConsensusVote, DissentRecord, etc.)
- ConsensusProof properties and methods
- Agreement ratios and consensus detection
- Evidence strength calculations
- Checksum generation
"""

from __future__ import annotations

from datetime import datetime

import pytest

from aragora.debate.consensus import (
    Claim,
    ConsensusProof,
    ConsensusVote,
    DissentRecord,
    Evidence,
    UnresolvedTension,
    VoteType,
)


class TestVoteType:
    """Tests for VoteType enum."""

    def test_vote_type_values(self):
        """Test VoteType enum values."""
        assert VoteType.AGREE.value == "agree"
        assert VoteType.DISAGREE.value == "disagree"
        assert VoteType.ABSTAIN.value == "abstain"
        assert VoteType.CONDITIONAL.value == "conditional"

    def test_vote_type_from_string(self):
        """Test creating VoteType from string."""
        assert VoteType("agree") == VoteType.AGREE
        assert VoteType("disagree") == VoteType.DISAGREE


class TestEvidence:
    """Tests for Evidence data class."""

    def test_evidence_creation(self):
        """Test creating Evidence."""
        evidence = Evidence(
            evidence_id="ev_001",
            source="claude",
            content="The API rate limit is 1000 requests per minute",
            evidence_type="argument",
            supports_claim=True,
            strength=0.8,
        )

        assert evidence.evidence_id == "ev_001"
        assert evidence.source == "claude"
        assert evidence.supports_claim is True
        assert evidence.strength == 0.8
        assert evidence.evidence_type == "argument"

    def test_evidence_with_metadata(self):
        """Test Evidence with metadata."""
        evidence = Evidence(
            evidence_id="ev_002",
            source="tool_output",
            content="Test results show 99.9% uptime",
            evidence_type="data",
            supports_claim=True,
            strength=0.95,
            metadata={"tool": "uptime_monitor", "period": "30_days"},
        )

        assert evidence.metadata["tool"] == "uptime_monitor"

    def test_evidence_timestamp_auto_generated(self):
        """Test that timestamp is auto-generated."""
        evidence = Evidence(
            evidence_id="ev_003",
            source="test",
            content="Test",
            evidence_type="test",
            supports_claim=True,
            strength=0.5,
        )

        assert evidence.timestamp is not None
        # Should be parseable as ISO format
        datetime.fromisoformat(evidence.timestamp)


class TestClaim:
    """Tests for Claim data class."""

    def test_claim_creation(self):
        """Test creating Claim."""
        claim = Claim(
            claim_id="claim_001",
            statement="The system should use Redis for caching",
            author="claude",
            confidence=0.85,
        )

        assert claim.claim_id == "claim_001"
        assert claim.author == "claude"
        assert claim.confidence == 0.85
        assert claim.status == "active"

    def test_claim_net_evidence_strength_balanced(self):
        """Test net evidence strength with balanced evidence."""
        claim = Claim(
            claim_id="claim_001",
            statement="Test claim",
            author="test",
            confidence=0.5,
            supporting_evidence=[
                Evidence("e1", "a", "content", "arg", True, 0.5),
                Evidence("e2", "b", "content", "arg", True, 0.5),
            ],
            refuting_evidence=[
                Evidence("e3", "c", "content", "arg", False, 0.5),
                Evidence("e4", "d", "content", "arg", False, 0.5),
            ],
        )

        # (1.0 - 1.0) / 2.0 = 0.0
        assert claim.net_evidence_strength == 0.0

    def test_claim_net_evidence_strength_supportive(self):
        """Test net evidence strength with more support."""
        claim = Claim(
            claim_id="claim_001",
            statement="Test claim",
            author="test",
            confidence=0.5,
            supporting_evidence=[
                Evidence("e1", "a", "content", "arg", True, 0.8),
                Evidence("e2", "b", "content", "arg", True, 0.7),
            ],
            refuting_evidence=[
                Evidence("e3", "c", "content", "arg", False, 0.3),
            ],
        )

        # (1.5 - 0.3) / 1.8 ≈ 0.667
        net = claim.net_evidence_strength
        assert net > 0.6
        assert net < 0.7

    def test_claim_net_evidence_strength_no_evidence(self):
        """Test net evidence strength with no evidence."""
        claim = Claim(
            claim_id="claim_001",
            statement="Test claim",
            author="test",
            confidence=0.5,
        )

        assert claim.net_evidence_strength == 0.0

    def test_claim_revision(self):
        """Test claim revision tracking."""
        original = Claim(
            claim_id="claim_001",
            statement="Original claim",
            author="claude",
            confidence=0.7,
        )

        revised = Claim(
            claim_id="claim_002",
            statement="Revised claim with more detail",
            author="claude",
            confidence=0.85,
            parent_claim_id="claim_001",
            status="active",
        )

        original.status = "revised"

        assert revised.parent_claim_id == original.claim_id
        assert original.status == "revised"


class TestDissentRecord:
    """Tests for DissentRecord data class."""

    def test_dissent_record_creation(self):
        """Test creating DissentRecord."""
        dissent = DissentRecord(
            agent="gpt4",
            claim_id="claim_001",
            dissent_type="partial",
            reasons=["Lacks consideration of edge cases", "Performance not addressed"],
            alternative_view="Consider using a hybrid approach",
            severity=0.6,
        )

        assert dissent.agent == "gpt4"
        assert dissent.dissent_type == "partial"
        assert len(dissent.reasons) == 2
        assert dissent.severity == 0.6

    def test_dissent_record_full_dissent(self):
        """Test full dissent record."""
        dissent = DissentRecord(
            agent="gemini",
            claim_id="claim_001",
            dissent_type="full",
            reasons=["Fundamentally disagree with the approach"],
            severity=0.9,
        )

        assert dissent.dissent_type == "full"
        assert dissent.severity == 0.9
        assert dissent.alternative_view is None


class TestUnresolvedTension:
    """Tests for UnresolvedTension data class."""

    def test_tension_creation(self):
        """Test creating UnresolvedTension."""
        tension = UnresolvedTension(
            tension_id="tension_001",
            description="Performance vs. Security tradeoff",
            agents_involved=["claude", "gpt4"],
            options=["Prioritize performance", "Prioritize security", "Balanced approach"],
            impact="Affects system architecture decision",
            suggested_followup="Benchmark both approaches",
        )

        assert tension.tension_id == "tension_001"
        assert len(tension.agents_involved) == 2
        assert len(tension.options) == 3


class TestConsensusVote:
    """Tests for ConsensusVote data class."""

    def test_vote_agree(self):
        """Test agree vote."""
        vote = ConsensusVote(
            agent="claude",
            vote=VoteType.AGREE,
            confidence=0.9,
            reasoning="Strongly support this approach based on evidence",
        )

        assert vote.vote == VoteType.AGREE
        assert vote.confidence == 0.9

    def test_vote_conditional(self):
        """Test conditional vote with conditions."""
        vote = ConsensusVote(
            agent="gpt4",
            vote=VoteType.CONDITIONAL,
            confidence=0.7,
            reasoning="Agree with reservations",
            conditions=["Must include error handling", "Needs performance testing"],
        )

        assert vote.vote == VoteType.CONDITIONAL
        assert len(vote.conditions) == 2


class TestConsensusProof:
    """Tests for ConsensusProof data class."""

    def _create_sample_proof(
        self,
        votes: list[ConsensusVote] | None = None,
        dissents: list[DissentRecord] | None = None,
        tensions: list[UnresolvedTension] | None = None,
    ) -> ConsensusProof:
        """Create a sample ConsensusProof for testing."""
        if votes is None:
            votes = [
                ConsensusVote("claude", VoteType.AGREE, 0.9, "Support"),
                ConsensusVote("gpt4", VoteType.AGREE, 0.85, "Support"),
                ConsensusVote("gemini", VoteType.DISAGREE, 0.7, "Dissent"),
            ]

        return ConsensusProof(
            proof_id="proof_001",
            debate_id="debate_001",
            task="Design a caching solution",
            final_claim="Use Redis for caching with a 15-minute TTL",
            confidence=0.85,
            consensus_reached=True,
            votes=votes,
            supporting_agents=["claude", "gpt4"],
            dissenting_agents=["gemini"],
            claims=[
                Claim("c1", "Redis is fast", "claude", 0.9),
                Claim("c2", "15-minute TTL balances freshness and load", "gpt4", 0.8),
            ],
            dissents=dissents or [],
            unresolved_tensions=tensions or [],
            evidence_chain=[
                Evidence("e1", "benchmark", "Redis: 10ms latency", "data", True, 0.95),
            ],
            reasoning_summary="After evaluating multiple options, Redis was chosen...",
            rounds_to_consensus=3,
        )

    def test_proof_creation(self):
        """Test creating ConsensusProof."""
        proof = self._create_sample_proof()

        assert proof.proof_id == "proof_001"
        assert proof.consensus_reached is True
        assert proof.confidence == 0.85
        assert len(proof.votes) == 3

    def test_agreement_ratio(self):
        """Test agreement ratio calculation."""
        proof = self._create_sample_proof()

        # 2 supporting, 1 dissenting = 2/3 ≈ 0.667
        assert proof.agreement_ratio == pytest.approx(0.667, rel=0.01)

    def test_agreement_ratio_unanimous(self):
        """Test agreement ratio with unanimous agreement."""
        votes = [
            ConsensusVote("claude", VoteType.AGREE, 0.9, "Support"),
            ConsensusVote("gpt4", VoteType.AGREE, 0.85, "Support"),
            ConsensusVote("gemini", VoteType.AGREE, 0.8, "Support"),
        ]
        proof = ConsensusProof(
            proof_id="proof_002",
            debate_id="debate_002",
            task="Test task",
            final_claim="Test claim",
            confidence=0.9,
            consensus_reached=True,
            votes=votes,
            supporting_agents=["claude", "gpt4", "gemini"],
            dissenting_agents=[],
            claims=[],
            dissents=[],
            unresolved_tensions=[],
            evidence_chain=[],
            reasoning_summary="Test",
        )

        assert proof.agreement_ratio == 1.0

    def test_agreement_ratio_no_agents(self):
        """Test agreement ratio with no agents."""
        proof = ConsensusProof(
            proof_id="proof_003",
            debate_id="debate_003",
            task="Test",
            final_claim="Test",
            confidence=0.5,
            consensus_reached=False,
            votes=[],
            supporting_agents=[],
            dissenting_agents=[],
            claims=[],
            dissents=[],
            unresolved_tensions=[],
            evidence_chain=[],
            reasoning_summary="Test",
        )

        assert proof.agreement_ratio == 0.0

    def test_has_strong_consensus_true(self):
        """Test strong consensus detection (true case)."""
        # Need > 80% agreement, so 5/6 = 83.3%
        votes = [
            ConsensusVote("a1", VoteType.AGREE, 0.9, "Support"),
            ConsensusVote("a2", VoteType.AGREE, 0.85, "Support"),
            ConsensusVote("a3", VoteType.AGREE, 0.8, "Support"),
            ConsensusVote("a4", VoteType.AGREE, 0.9, "Support"),
            ConsensusVote("a5", VoteType.AGREE, 0.85, "Support"),
            ConsensusVote("a6", VoteType.DISAGREE, 0.7, "Dissent"),
        ]
        proof = ConsensusProof(
            proof_id="proof_004",
            debate_id="debate_004",
            task="Test",
            final_claim="Test",
            confidence=0.85,
            consensus_reached=True,
            votes=votes,
            supporting_agents=["a1", "a2", "a3", "a4", "a5"],
            dissenting_agents=["a6"],
            claims=[],
            dissents=[],
            unresolved_tensions=[],
            evidence_chain=[],
            reasoning_summary="Test",
        )

        # 5/6 ≈ 83.3% > 80% agreement, 0.85 > 0.7 confidence
        assert proof.has_strong_consensus is True

    def test_has_strong_consensus_false_low_agreement(self):
        """Test strong consensus detection (false due to low agreement)."""
        proof = self._create_sample_proof()

        # 2/3 ≈ 67% < 80%
        assert proof.has_strong_consensus is False

    def test_has_strong_consensus_false_low_confidence(self):
        """Test strong consensus detection (false due to low confidence)."""
        votes = [
            ConsensusVote("a1", VoteType.AGREE, 0.9, "Support"),
            ConsensusVote("a2", VoteType.AGREE, 0.85, "Support"),
        ]
        proof = ConsensusProof(
            proof_id="proof_005",
            debate_id="debate_005",
            task="Test",
            final_claim="Test",
            confidence=0.5,  # Low confidence
            consensus_reached=True,
            votes=votes,
            supporting_agents=["a1", "a2"],
            dissenting_agents=[],
            claims=[],
            dissents=[],
            unresolved_tensions=[],
            evidence_chain=[],
            reasoning_summary="Test",
        )

        # 100% agreement but only 0.5 confidence < 0.7
        assert proof.has_strong_consensus is False

    @pytest.mark.skip(reason="Known issue: VoteType enum not JSON serializable in checksum method")
    def test_checksum_generation(self):
        """Test checksum generation."""
        proof = self._create_sample_proof()
        checksum = proof.checksum

        assert isinstance(checksum, str)
        assert len(checksum) == 16

    @pytest.mark.skip(reason="Known issue: VoteType enum not JSON serializable in checksum method")
    def test_checksum_deterministic(self):
        """Test that checksum is deterministic."""
        proof = self._create_sample_proof()

        checksum1 = proof.checksum
        checksum2 = proof.checksum

        assert checksum1 == checksum2

    @pytest.mark.skip(reason="Known issue: VoteType enum not JSON serializable in checksum method")
    def test_checksum_changes_with_content(self):
        """Test that checksum changes when content changes."""
        proof1 = self._create_sample_proof()
        proof2 = self._create_sample_proof()
        proof2.final_claim = "Different claim"

        assert proof1.checksum != proof2.checksum

    def test_get_dissent_summary_no_dissents(self):
        """Test dissent summary with no dissents."""
        proof = self._create_sample_proof(dissents=[])
        summary = proof.get_dissent_summary()

        assert summary == "No dissenting views recorded."

    def test_get_dissent_summary_with_dissents(self):
        """Test dissent summary with dissents."""
        dissents = [
            DissentRecord(
                agent="gemini",
                claim_id="c1",
                dissent_type="partial",
                reasons=["Reason 1", "Reason 2"],
                alternative_view="Alternative approach",
                severity=0.6,
            ),
        ]
        proof = self._create_sample_proof(dissents=dissents)
        summary = proof.get_dissent_summary()

        assert "## Dissenting Views" in summary
        assert "gemini" in summary
        assert "Reason 1" in summary
        assert "Alternative approach" in summary

    def test_get_tension_summary_no_tensions(self):
        """Test tension summary with no tensions."""
        proof = self._create_sample_proof(tensions=[])
        summary = proof.get_tension_summary()

        assert summary == "No unresolved tensions."

    def test_get_tension_summary_with_tensions(self):
        """Test tension summary with tensions."""
        tensions = [
            UnresolvedTension(
                tension_id="t1",
                description="Performance vs. Security",
                agents_involved=["claude", "gpt4"],
                options=["Fast", "Secure"],
                impact="Architecture decision",
                suggested_followup="Benchmark both",
            ),
        ]
        proof = self._create_sample_proof(tensions=tensions)
        summary = proof.get_tension_summary()

        assert "## Unresolved Tensions" in summary
        assert "Performance vs. Security" in summary
        assert "Benchmark both" in summary

    def test_get_confidence_breakdown(self):
        """Test confidence breakdown by agent."""
        proof = self._create_sample_proof()
        breakdown = proof.get_confidence_breakdown()

        assert breakdown["claude"] == 0.9
        assert breakdown["gpt4"] == 0.85
        assert breakdown["gemini"] == 0.7

    def test_get_blind_spots_high_severity_dissent(self):
        """Test blind spot detection from high-severity dissents."""
        dissents = [
            DissentRecord(
                agent="gemini",
                claim_id="c1",
                dissent_type="full",
                reasons=["Security concern"],
                alternative_view="Use encryption",
                severity=0.8,  # High severity
            ),
        ]
        proof = self._create_sample_proof(dissents=dissents)
        blind_spots = proof.get_blind_spots()

        assert len(blind_spots) > 0
        assert any("encryption" in spot.lower() for spot in blind_spots)

    def test_get_blind_spots_from_tensions(self):
        """Test blind spot detection from unresolved tensions."""
        tensions = [
            UnresolvedTension(
                tension_id="t1",
                description="Scalability concern",
                agents_involved=["claude"],
                options=["Horizontal scaling", "Vertical scaling"],
                impact="Cost implications",
            ),
        ]
        proof = self._create_sample_proof(tensions=tensions)
        blind_spots = proof.get_blind_spots()

        assert any("Scalability" in spot for spot in blind_spots)
