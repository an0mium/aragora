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
    ConsensusBuilder,
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

    def test_checksum_generation(self):
        """Test checksum generation."""
        proof = self._create_sample_proof()
        checksum = proof.checksum

        assert isinstance(checksum, str)
        assert len(checksum) == 16

    def test_checksum_deterministic(self):
        """Test that checksum is deterministic."""
        proof = self._create_sample_proof()

        checksum1 = proof.checksum
        checksum2 = proof.checksum

        assert checksum1 == checksum2

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

    def test_get_blind_spots_low_agreement(self):
        """Test blind spot detection from low agreement ratio."""
        # 1 supporting, 2 dissenting = 33% agreement
        votes = [
            ConsensusVote("a1", VoteType.AGREE, 0.9, "Support"),
            ConsensusVote("a2", VoteType.DISAGREE, 0.8, "Dissent"),
            ConsensusVote("a3", VoteType.DISAGREE, 0.7, "Dissent"),
        ]
        proof = ConsensusProof(
            proof_id="proof_006",
            debate_id="debate_006",
            task="Test",
            final_claim="Test",
            confidence=0.7,
            consensus_reached=False,
            votes=votes,
            supporting_agents=["a1"],
            dissenting_agents=["a2", "a3"],
            claims=[],
            dissents=[],
            unresolved_tensions=[],
            evidence_chain=[],
            reasoning_summary="Test",
        )

        blind_spots = proof.get_blind_spots()
        assert any("Low consensus" in spot for spot in blind_spots)

    def test_get_risk_correlation_empty(self):
        """Test risk correlation with no claims."""
        proof = self._create_sample_proof()
        proof.claims = []
        proof.unresolved_tensions = []
        correlation = proof.get_risk_correlation()

        assert correlation["unanimous"] == []
        assert correlation["majority"] == []
        assert correlation["contested"] == []

    def test_get_risk_correlation_unanimous(self):
        """Test risk correlation with unanimous support."""
        claim = Claim(
            claim_id="c1",
            statement="All agents agree on this claim",
            author="claude",
            confidence=0.9,
            supporting_evidence=[
                Evidence("e1", "a", "content", "arg", True, 0.8),
                Evidence("e2", "b", "content", "arg", True, 0.7),
            ],
            refuting_evidence=[],  # No refuting evidence
        )
        proof = self._create_sample_proof()
        proof.claims = [claim]
        proof.unresolved_tensions = []

        correlation = proof.get_risk_correlation()
        assert len(correlation["unanimous"]) == 1
        assert "All agents agree" in correlation["unanimous"][0]

    def test_get_risk_correlation_contested(self):
        """Test risk correlation with contested claims."""
        claim = Claim(
            claim_id="c1",
            statement="Controversial claim",
            author="claude",
            confidence=0.5,
            supporting_evidence=[
                Evidence("e1", "a", "content", "arg", True, 0.5),
            ],
            refuting_evidence=[
                Evidence("e2", "b", "content", "arg", False, 0.5),
            ],
        )
        tensions = [
            UnresolvedTension(
                tension_id="t1",
                description="Big disagreement",
                agents_involved=["a", "b"],
                options=["Option A", "Option B"],
                impact="Major",
            ),
        ]
        proof = self._create_sample_proof()
        proof.claims = [claim]
        proof.unresolved_tensions = tensions

        correlation = proof.get_risk_correlation()
        assert len(correlation["contested"]) >= 1
        assert any("Big disagreement" in item for item in correlation["contested"])

    def test_to_dict(self):
        """Test conversion to dictionary."""
        proof = self._create_sample_proof()
        data = proof.to_dict()

        assert data["proof_id"] == "proof_001"
        assert data["debate_id"] == "debate_001"
        assert data["final_claim"] == "Use Redis for caching with a 15-minute TTL"
        assert data["confidence"] == 0.85
        assert data["consensus_reached"] is True
        assert len(data["votes"]) == 3
        assert "checksum" in data

    def test_to_json(self):
        """Test JSON serialization."""
        proof = self._create_sample_proof()
        json_str = proof.to_json()

        import json

        data = json.loads(json_str)
        assert data["proof_id"] == "proof_001"
        assert isinstance(json_str, str)

    def test_to_json_with_indent(self):
        """Test JSON serialization with custom indent."""
        proof = self._create_sample_proof()
        json_str = proof.to_json(indent=4)

        # With indent=4, the string should contain indented lines
        assert "    " in json_str

    def test_to_markdown(self):
        """Test Markdown report generation."""
        dissents = [
            DissentRecord(
                agent="gemini",
                claim_id="c1",
                dissent_type="partial",
                reasons=["Reason 1"],
                alternative_view="Alternative",
                severity=0.5,
            ),
        ]
        tensions = [
            UnresolvedTension(
                tension_id="t1",
                description="Test tension",
                agents_involved=["claude"],
                options=["A", "B"],
                impact="Test impact",
            ),
        ]
        proof = self._create_sample_proof(dissents=dissents, tensions=tensions)
        markdown = proof.to_markdown()

        assert "# Consensus Proof" in markdown
        assert "**Proof ID:** `proof_001`" in markdown
        assert "## Task" in markdown
        assert "## Consensus" in markdown
        assert "Design a caching solution" in markdown
        assert "Supporting Agents" in markdown
        assert "Dissenting Agents" in markdown
        assert "## Evidence Chain" in markdown

    def test_to_markdown_no_dissenting_agents(self):
        """Test Markdown with no dissenting agents."""
        votes = [
            ConsensusVote("claude", VoteType.AGREE, 0.9, "Support"),
            ConsensusVote("gpt4", VoteType.AGREE, 0.85, "Support"),
        ]
        proof = ConsensusProof(
            proof_id="proof_007",
            debate_id="debate_007",
            task="Test task",
            final_claim="Test claim",
            confidence=0.9,
            consensus_reached=True,
            votes=votes,
            supporting_agents=["claude", "gpt4"],
            dissenting_agents=[],
            claims=[],
            dissents=[],
            unresolved_tensions=[],
            evidence_chain=[],
            reasoning_summary="All agreed",
        )

        markdown = proof.to_markdown()
        assert "*None*" in markdown  # No dissenting agents section


class TestConsensusBuilder:
    """Tests for ConsensusBuilder class."""

    def test_builder_initialization(self):
        """Test ConsensusBuilder initialization."""
        builder = ConsensusBuilder("debate_001", "Design a cache system")

        assert builder.debate_id == "debate_001"
        assert builder.task == "Design a cache system"
        assert builder.claims == []
        assert builder.evidence == []
        assert builder.votes == []
        assert builder.dissents == []
        assert builder.tensions == []

    def test_add_claim(self):
        """Test adding a claim."""
        builder = ConsensusBuilder("debate_001", "Test task")
        claim = builder.add_claim(
            statement="Use Redis for caching",
            author="claude",
            confidence=0.85,
            round_num=1,
        )

        assert claim.claim_id == "debate_001-claim-1"
        assert claim.statement == "Use Redis for caching"
        assert claim.author == "claude"
        assert claim.confidence == 0.85
        assert claim.round_introduced == 1
        assert len(builder.claims) == 1

    def test_add_claim_with_parent(self):
        """Test adding a claim that revises another."""
        builder = ConsensusBuilder("debate_001", "Test task")
        original = builder.add_claim("Original claim", "claude", 0.7, 1)
        revised = builder.add_claim(
            statement="Revised claim",
            author="claude",
            confidence=0.9,
            round_num=2,
            parent_claim_id=original.claim_id,
        )

        assert revised.parent_claim_id == original.claim_id
        assert len(builder.claims) == 2

    def test_add_multiple_claims_increments_counter(self):
        """Test that claim IDs increment."""
        builder = ConsensusBuilder("debate_001", "Test task")
        claim1 = builder.add_claim("Claim 1", "a", 0.5)
        claim2 = builder.add_claim("Claim 2", "b", 0.5)
        claim3 = builder.add_claim("Claim 3", "c", 0.5)

        assert claim1.claim_id == "debate_001-claim-1"
        assert claim2.claim_id == "debate_001-claim-2"
        assert claim3.claim_id == "debate_001-claim-3"

    def test_add_evidence_supporting(self):
        """Test adding supporting evidence."""
        builder = ConsensusBuilder("debate_001", "Test task")
        claim = builder.add_claim("Redis is fast", "claude", 0.8)
        evidence = builder.add_evidence(
            claim_id=claim.claim_id,
            source="benchmark",
            content="Redis: 10ms latency",
            evidence_type="data",
            supports=True,
            strength=0.95,
        )

        assert evidence.evidence_id == "debate_001-ev-1"
        assert evidence.supports_claim is True
        assert evidence.strength == 0.95
        assert len(claim.supporting_evidence) == 1
        assert len(claim.refuting_evidence) == 0

    def test_add_evidence_refuting(self):
        """Test adding refuting evidence."""
        builder = ConsensusBuilder("debate_001", "Test task")
        claim = builder.add_claim("Redis is always best", "claude", 0.8)
        evidence = builder.add_evidence(
            claim_id=claim.claim_id,
            source="gpt4",
            content="Memcached is faster for simple key-value",
            evidence_type="argument",
            supports=False,
            strength=0.7,
        )

        assert evidence.supports_claim is False
        assert len(claim.supporting_evidence) == 0
        assert len(claim.refuting_evidence) == 1

    def test_add_evidence_nonexistent_claim(self):
        """Test adding evidence for non-existent claim."""
        builder = ConsensusBuilder("debate_001", "Test task")
        evidence = builder.add_evidence(
            claim_id="nonexistent",
            source="test",
            content="Content",
            evidence_type="argument",
            supports=True,
            strength=0.5,
        )

        # Evidence is still created and tracked
        assert evidence.evidence_id == "debate_001-ev-1"
        assert len(builder.evidence) == 1

    def test_record_vote(self):
        """Test recording a vote."""
        builder = ConsensusBuilder("debate_001", "Test task")
        vote = builder.record_vote(
            agent="claude",
            vote=VoteType.AGREE,
            confidence=0.9,
            reasoning="Strongly support this approach",
        )

        assert vote.agent == "claude"
        assert vote.vote == VoteType.AGREE
        assert vote.confidence == 0.9
        assert len(builder.votes) == 1

    def test_record_vote_conditional(self):
        """Test recording a conditional vote."""
        builder = ConsensusBuilder("debate_001", "Test task")
        vote = builder.record_vote(
            agent="gpt4",
            vote=VoteType.CONDITIONAL,
            confidence=0.7,
            reasoning="Agree with reservations",
            conditions=["Must handle errors", "Need tests"],
        )

        assert vote.vote == VoteType.CONDITIONAL
        assert len(vote.conditions) == 2

    def test_record_dissent(self):
        """Test recording dissent."""
        builder = ConsensusBuilder("debate_001", "Test task")
        claim = builder.add_claim("Disputed claim", "claude", 0.6)
        dissent = builder.record_dissent(
            agent="gemini",
            claim_id=claim.claim_id,
            reasons=["Security concerns", "Performance issues"],
            dissent_type="partial",
            alternative="Use a different approach",
            severity=0.7,
        )

        assert dissent.agent == "gemini"
        assert dissent.claim_id == claim.claim_id
        assert dissent.dissent_type == "partial"
        assert len(dissent.reasons) == 2
        assert dissent.alternative_view == "Use a different approach"
        assert dissent.severity == 0.7
        assert len(builder.dissents) == 1

    def test_record_tension(self):
        """Test recording unresolved tension."""
        builder = ConsensusBuilder("debate_001", "Test task")
        tension = builder.record_tension(
            description="Performance vs Security tradeoff",
            agents=["claude", "gpt4"],
            options=["Prioritize performance", "Prioritize security"],
            impact="Affects architecture decision",
            followup="Run benchmarks",
        )

        assert tension.tension_id == "debate_001-tension-1"
        assert tension.description == "Performance vs Security tradeoff"
        assert len(tension.agents_involved) == 2
        assert len(tension.options) == 2
        assert tension.suggested_followup == "Run benchmarks"
        assert len(builder.tensions) == 1

    def test_record_multiple_tensions(self):
        """Test recording multiple tensions increments ID."""
        builder = ConsensusBuilder("debate_001", "Test task")
        t1 = builder.record_tension("Tension 1", ["a"], ["opt"], "impact")
        t2 = builder.record_tension("Tension 2", ["b"], ["opt"], "impact")

        assert t1.tension_id == "debate_001-tension-1"
        assert t2.tension_id == "debate_001-tension-2"

    def test_build_consensus_proof(self):
        """Test building a ConsensusProof."""
        builder = ConsensusBuilder("debate_001", "Design a cache")

        # Add claims
        claim = builder.add_claim("Use Redis", "claude", 0.85, 1)
        builder.add_evidence(claim.claim_id, "benchmark", "Fast", "data", True, 0.9)

        # Record votes
        builder.record_vote("claude", VoteType.AGREE, 0.9, "Support")
        builder.record_vote("gpt4", VoteType.AGREE, 0.85, "Support")
        builder.record_vote("gemini", VoteType.DISAGREE, 0.6, "Concerns")

        # Build proof
        proof = builder.build(
            final_claim="Use Redis with 15min TTL",
            confidence=0.85,
            consensus_reached=True,
            reasoning_summary="Agents agreed on Redis",
            rounds=3,
        )

        assert proof.proof_id == "proof-debate_001"
        assert proof.debate_id == "debate_001"
        assert proof.task == "Design a cache"
        assert proof.final_claim == "Use Redis with 15min TTL"
        assert proof.confidence == 0.85
        assert proof.consensus_reached is True
        assert proof.rounds_to_consensus == 3
        assert len(proof.votes) == 3
        assert "claude" in proof.supporting_agents
        assert "gpt4" in proof.supporting_agents
        assert "gemini" in proof.dissenting_agents

    def test_build_categorizes_conditional_as_supporting(self):
        """Test that conditional votes are counted as supporting."""
        builder = ConsensusBuilder("debate_001", "Test task")
        builder.record_vote("claude", VoteType.CONDITIONAL, 0.8, "Agree with conditions")

        proof = builder.build(
            final_claim="Final",
            confidence=0.8,
            consensus_reached=True,
            reasoning_summary="Summary",
        )

        assert "claude" in proof.supporting_agents
        assert "claude" not in proof.dissenting_agents

    def test_build_includes_dissents_and_tensions(self):
        """Test that build includes recorded dissents and tensions."""
        builder = ConsensusBuilder("debate_001", "Test task")
        claim = builder.add_claim("Claim", "claude", 0.7)
        builder.record_dissent("gemini", claim.claim_id, ["Reason"], severity=0.8)
        builder.record_tension("Tension", ["a", "b"], ["opt1", "opt2"], "Impact")

        proof = builder.build(
            final_claim="Final",
            confidence=0.7,
            consensus_reached=False,
            reasoning_summary="No consensus",
        )

        assert len(proof.dissents) == 1
        assert len(proof.unresolved_tensions) == 1
