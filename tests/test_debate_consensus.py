"""Tests for the consensus module - consensus detection, voting, and proof generation."""

import json

import pytest
from unittest.mock import MagicMock

from aragora.debate.consensus import (
    VoteType,
    Evidence,
    Claim,
    DissentRecord,
    UnresolvedTension,
    ConsensusVote,
    ConsensusProof,
    ConsensusBuilder,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_evidence():
    """Create sample evidence for testing."""
    return Evidence(
        evidence_id="ev-1",
        source="agent-1",
        content="Test evidence content",
        evidence_type="argument",
        supports_claim=True,
        strength=0.7,
    )


@pytest.fixture
def sample_claim(sample_evidence):
    """Create sample claim with evidence."""
    return Claim(
        claim_id="claim-1",
        statement="Test claim statement",
        author="agent-1",
        confidence=0.8,
        supporting_evidence=[sample_evidence],
    )


@pytest.fixture
def sample_vote():
    """Create sample vote."""
    return ConsensusVote(
        agent="agent-1",
        vote=VoteType.AGREE,
        confidence=0.9,
        reasoning="Sounds good to me",
    )


@pytest.fixture
def sample_dissent():
    """Create sample dissent record."""
    return DissentRecord(
        agent="agent-2",
        claim_id="claim-1",
        dissent_type="partial",
        reasons=["Missing edge case", "Performance concern"],
        alternative_view="Should use async approach",
        severity=0.6,
    )


@pytest.fixture
def sample_tension():
    """Create sample unresolved tension."""
    return UnresolvedTension(
        tension_id="tension-1",
        description="Performance vs readability tradeoff",
        agents_involved=["agent-1", "agent-2"],
        options=["Optimize for speed", "Keep code simple"],
        impact="Affects maintenance cost",
        suggested_followup="Benchmark both approaches",
    )


@pytest.fixture
def builder():
    """Create ConsensusBuilder for testing."""
    return ConsensusBuilder(debate_id="test-debate", task="Test task")


@pytest.fixture
def minimal_proof():
    """Create minimal ConsensusProof for testing."""
    return ConsensusProof(
        proof_id="proof-1",
        debate_id="debate-1",
        task="Test task",
        final_claim="Final answer",
        confidence=0.85,
        consensus_reached=True,
        votes=[],
        supporting_agents=["agent-1", "agent-2"],
        dissenting_agents=[],
        claims=[],
        dissents=[],
        unresolved_tensions=[],
        evidence_chain=[],
        reasoning_summary="Test reasoning",
    )


@pytest.fixture
def full_proof(sample_vote, sample_claim, sample_dissent, sample_tension, sample_evidence):
    """Create ConsensusProof with all fields populated."""
    return ConsensusProof(
        proof_id="proof-full",
        debate_id="debate-full",
        task="Complex task",
        final_claim="Complex final answer",
        confidence=0.75,
        consensus_reached=True,
        votes=[sample_vote],
        supporting_agents=["agent-1", "agent-3"],
        dissenting_agents=["agent-2"],
        claims=[sample_claim],
        dissents=[sample_dissent],
        unresolved_tensions=[sample_tension],
        evidence_chain=[sample_evidence],
        reasoning_summary="Detailed reasoning summary",
        rounds_to_consensus=3,
    )


# =============================================================================
# Test VoteType Enum
# =============================================================================


class TestVoteType:
    """Tests for VoteType enum."""

    def test_vote_values_count(self):
        """Should have 4 vote types."""
        assert len(VoteType) == 4

    def test_agree_value(self):
        """Should have correct agree value."""
        assert VoteType.AGREE.value == "agree"

    def test_disagree_value(self):
        """Should have correct disagree value."""
        assert VoteType.DISAGREE.value == "disagree"

    def test_abstain_value(self):
        """Should have correct abstain value."""
        assert VoteType.ABSTAIN.value == "abstain"

    def test_conditional_value(self):
        """Should have correct conditional value."""
        assert VoteType.CONDITIONAL.value == "conditional"


# =============================================================================
# Test Evidence Dataclass
# =============================================================================


class TestEvidence:
    """Tests for Evidence dataclass."""

    def test_creation_with_required_fields(self):
        """Should create evidence with required fields."""
        evidence = Evidence(
            evidence_id="ev-1",
            source="agent-1",
            content="Test content",
            evidence_type="argument",
            supports_claim=True,
            strength=0.5,
        )
        assert evidence.evidence_id == "ev-1"
        assert evidence.source == "agent-1"
        assert evidence.content == "Test content"
        assert evidence.evidence_type == "argument"
        assert evidence.supports_claim is True
        assert evidence.strength == 0.5

    def test_timestamp_auto_generated(self):
        """Should auto-generate timestamp."""
        evidence = Evidence(
            evidence_id="ev-1",
            source="agent-1",
            content="Test",
            evidence_type="argument",
            supports_claim=True,
            strength=0.5,
        )
        assert evidence.timestamp is not None
        assert len(evidence.timestamp) > 0

    def test_metadata_defaults_to_empty_dict(self):
        """Should default metadata to empty dict."""
        evidence = Evidence(
            evidence_id="ev-1",
            source="agent-1",
            content="Test",
            evidence_type="argument",
            supports_claim=True,
            strength=0.5,
        )
        assert evidence.metadata == {}

    def test_creation_with_metadata(self):
        """Should accept custom metadata."""
        evidence = Evidence(
            evidence_id="ev-1",
            source="agent-1",
            content="Test",
            evidence_type="data",
            supports_claim=False,
            strength=0.8,
            metadata={"source_url": "http://example.com"},
        )
        assert evidence.metadata == {"source_url": "http://example.com"}


# =============================================================================
# Test Claim Dataclass
# =============================================================================


class TestClaim:
    """Tests for Claim dataclass."""

    def test_creation_with_defaults(self):
        """Should create claim with default values."""
        claim = Claim(
            claim_id="claim-1",
            statement="Test statement",
            author="agent-1",
            confidence=0.5,
        )
        assert claim.claim_id == "claim-1"
        assert claim.supporting_evidence == []
        assert claim.refuting_evidence == []
        assert claim.round_introduced == 0
        assert claim.status == "active"
        assert claim.parent_claim_id is None

    def test_net_evidence_strength_no_evidence(self):
        """Should return 0.0 with no evidence."""
        claim = Claim(
            claim_id="claim-1",
            statement="Test",
            author="agent-1",
            confidence=0.5,
        )
        assert claim.net_evidence_strength == 0.0

    def test_net_evidence_strength_only_support(self):
        """Should return positive value with only supporting evidence."""
        evidence = Evidence(
            evidence_id="ev-1",
            source="agent-1",
            content="Support",
            evidence_type="argument",
            supports_claim=True,
            strength=0.8,
        )
        claim = Claim(
            claim_id="claim-1",
            statement="Test",
            author="agent-1",
            confidence=0.5,
            supporting_evidence=[evidence],
        )
        assert claim.net_evidence_strength == 1.0  # 0.8 / 0.8

    def test_net_evidence_strength_only_refute(self):
        """Should return negative value with only refuting evidence."""
        evidence = Evidence(
            evidence_id="ev-1",
            source="agent-2",
            content="Refute",
            evidence_type="argument",
            supports_claim=False,
            strength=0.6,
        )
        claim = Claim(
            claim_id="claim-1",
            statement="Test",
            author="agent-1",
            confidence=0.5,
            refuting_evidence=[evidence],
        )
        assert claim.net_evidence_strength == -1.0  # -0.6 / 0.6

    def test_net_evidence_strength_balanced(self):
        """Should return 0.0 when evidence is balanced."""
        support = Evidence(
            evidence_id="ev-1",
            source="agent-1",
            content="Support",
            evidence_type="argument",
            supports_claim=True,
            strength=0.5,
        )
        refute = Evidence(
            evidence_id="ev-2",
            source="agent-2",
            content="Refute",
            evidence_type="argument",
            supports_claim=False,
            strength=0.5,
        )
        claim = Claim(
            claim_id="claim-1",
            statement="Test",
            author="agent-1",
            confidence=0.5,
            supporting_evidence=[support],
            refuting_evidence=[refute],
        )
        assert claim.net_evidence_strength == 0.0

    def test_net_evidence_strength_mixed(self):
        """Should calculate net strength with mixed evidence."""
        support = Evidence(
            evidence_id="ev-1",
            source="agent-1",
            content="Support",
            evidence_type="argument",
            supports_claim=True,
            strength=0.8,
        )
        refute = Evidence(
            evidence_id="ev-2",
            source="agent-2",
            content="Refute",
            evidence_type="argument",
            supports_claim=False,
            strength=0.2,
        )
        claim = Claim(
            claim_id="claim-1",
            statement="Test",
            author="agent-1",
            confidence=0.5,
            supporting_evidence=[support],
            refuting_evidence=[refute],
        )
        # (0.8 - 0.2) / (0.8 + 0.2) = 0.6
        assert claim.net_evidence_strength == pytest.approx(0.6)

    def test_status_default_active(self):
        """Should default to active status."""
        claim = Claim(
            claim_id="claim-1",
            statement="Test",
            author="agent-1",
            confidence=0.5,
        )
        assert claim.status == "active"


# =============================================================================
# Test DissentRecord Dataclass
# =============================================================================


class TestDissentRecord:
    """Tests for DissentRecord dataclass."""

    def test_creation_with_required_fields(self):
        """Should create dissent with required fields."""
        dissent = DissentRecord(
            agent="agent-1",
            claim_id="claim-1",
            dissent_type="full",
            reasons=["Reason 1", "Reason 2"],
        )
        assert dissent.agent == "agent-1"
        assert dissent.claim_id == "claim-1"
        assert dissent.dissent_type == "full"
        assert dissent.reasons == ["Reason 1", "Reason 2"]

    def test_optional_fields_default_none(self):
        """Should default optional fields to None."""
        dissent = DissentRecord(
            agent="agent-1",
            claim_id="claim-1",
            dissent_type="partial",
            reasons=["Reason"],
        )
        assert dissent.alternative_view is None
        assert dissent.suggested_resolution is None

    def test_severity_default(self):
        """Should default severity to 0.5."""
        dissent = DissentRecord(
            agent="agent-1",
            claim_id="claim-1",
            dissent_type="partial",
            reasons=["Reason"],
        )
        assert dissent.severity == 0.5

    def test_timestamp_auto_generated(self):
        """Should auto-generate timestamp."""
        dissent = DissentRecord(
            agent="agent-1",
            claim_id="claim-1",
            dissent_type="partial",
            reasons=["Reason"],
        )
        assert dissent.timestamp is not None
        assert len(dissent.timestamp) > 0


# =============================================================================
# Test UnresolvedTension Dataclass
# =============================================================================


class TestUnresolvedTension:
    """Tests for UnresolvedTension dataclass."""

    def test_creation(self):
        """Should create tension with required fields."""
        tension = UnresolvedTension(
            tension_id="tension-1",
            description="Test tension",
            agents_involved=["agent-1", "agent-2"],
            options=["Option A", "Option B"],
            impact="High impact",
        )
        assert tension.tension_id == "tension-1"
        assert tension.description == "Test tension"
        assert tension.agents_involved == ["agent-1", "agent-2"]
        assert tension.options == ["Option A", "Option B"]
        assert tension.impact == "High impact"

    def test_optional_followup(self):
        """Should default followup to None."""
        tension = UnresolvedTension(
            tension_id="tension-1",
            description="Test",
            agents_involved=["agent-1"],
            options=["A"],
            impact="Low",
        )
        assert tension.suggested_followup is None


# =============================================================================
# Test ConsensusVote Dataclass
# =============================================================================


class TestConsensusVote:
    """Tests for ConsensusVote dataclass."""

    def test_creation(self):
        """Should create vote with required fields."""
        vote = ConsensusVote(
            agent="agent-1",
            vote=VoteType.AGREE,
            confidence=0.9,
            reasoning="Good solution",
        )
        assert vote.agent == "agent-1"
        assert vote.vote == VoteType.AGREE
        assert vote.confidence == 0.9
        assert vote.reasoning == "Good solution"

    def test_conditions_default_empty(self):
        """Should default conditions to empty list."""
        vote = ConsensusVote(
            agent="agent-1",
            vote=VoteType.CONDITIONAL,
            confidence=0.7,
            reasoning="With reservations",
        )
        assert vote.conditions == []

    def test_timestamp_auto_generated(self):
        """Should auto-generate timestamp."""
        vote = ConsensusVote(
            agent="agent-1",
            vote=VoteType.AGREE,
            confidence=0.9,
            reasoning="Good",
        )
        assert vote.timestamp is not None

    def test_creation_with_conditions(self):
        """Should accept conditions list."""
        vote = ConsensusVote(
            agent="agent-1",
            vote=VoteType.CONDITIONAL,
            confidence=0.7,
            reasoning="If conditions met",
            conditions=["Add tests", "Fix edge case"],
        )
        assert vote.conditions == ["Add tests", "Fix edge case"]


# =============================================================================
# Test ConsensusProof Properties
# =============================================================================


class TestConsensusProofProperties:
    """Tests for ConsensusProof properties."""

    def test_checksum_deterministic(self, minimal_proof):
        """Checksum should be deterministic for same content."""
        checksum1 = minimal_proof.checksum
        checksum2 = minimal_proof.checksum
        assert checksum1 == checksum2

    def test_checksum_changes_with_content(self, minimal_proof):
        """Checksum should change when content changes."""
        checksum1 = minimal_proof.checksum
        minimal_proof.final_claim = "Different claim"
        checksum2 = minimal_proof.checksum
        assert checksum1 != checksum2

    def test_agreement_ratio_all_agree(self):
        """Should return 1.0 when all agree."""
        proof = ConsensusProof(
            proof_id="p1",
            debate_id="d1",
            task="Task",
            final_claim="Claim",
            confidence=0.9,
            consensus_reached=True,
            votes=[],
            supporting_agents=["a1", "a2", "a3"],
            dissenting_agents=[],
            claims=[],
            dissents=[],
            unresolved_tensions=[],
            evidence_chain=[],
            reasoning_summary="Summary",
        )
        assert proof.agreement_ratio == 1.0

    def test_agreement_ratio_all_dissent(self):
        """Should return 0.0 when all dissent."""
        proof = ConsensusProof(
            proof_id="p1",
            debate_id="d1",
            task="Task",
            final_claim="Claim",
            confidence=0.5,
            consensus_reached=False,
            votes=[],
            supporting_agents=[],
            dissenting_agents=["a1", "a2"],
            claims=[],
            dissents=[],
            unresolved_tensions=[],
            evidence_chain=[],
            reasoning_summary="Summary",
        )
        assert proof.agreement_ratio == 0.0

    def test_agreement_ratio_mixed(self):
        """Should calculate correct ratio with mixed votes."""
        proof = ConsensusProof(
            proof_id="p1",
            debate_id="d1",
            task="Task",
            final_claim="Claim",
            confidence=0.7,
            consensus_reached=True,
            votes=[],
            supporting_agents=["a1", "a2", "a3"],
            dissenting_agents=["a4"],
            claims=[],
            dissents=[],
            unresolved_tensions=[],
            evidence_chain=[],
            reasoning_summary="Summary",
        )
        assert proof.agreement_ratio == 0.75

    def test_agreement_ratio_empty(self):
        """Should return 0.0 when no agents."""
        proof = ConsensusProof(
            proof_id="p1",
            debate_id="d1",
            task="Task",
            final_claim="Claim",
            confidence=0.5,
            consensus_reached=False,
            votes=[],
            supporting_agents=[],
            dissenting_agents=[],
            claims=[],
            dissents=[],
            unresolved_tensions=[],
            evidence_chain=[],
            reasoning_summary="Summary",
        )
        assert proof.agreement_ratio == 0.0

    def test_has_strong_consensus_true(self):
        """Should return True for strong consensus."""
        proof = ConsensusProof(
            proof_id="p1",
            debate_id="d1",
            task="Task",
            final_claim="Claim",
            confidence=0.85,
            consensus_reached=True,
            votes=[],
            supporting_agents=["a1", "a2", "a3", "a4", "a5"],
            dissenting_agents=[],
            claims=[],
            dissents=[],
            unresolved_tensions=[],
            evidence_chain=[],
            reasoning_summary="Summary",
        )
        assert proof.has_strong_consensus is True

    def test_has_strong_consensus_low_agreement(self):
        """Should return False when agreement < 80%."""
        proof = ConsensusProof(
            proof_id="p1",
            debate_id="d1",
            task="Task",
            final_claim="Claim",
            confidence=0.85,
            consensus_reached=True,
            votes=[],
            supporting_agents=["a1", "a2"],
            dissenting_agents=["a3", "a4", "a5"],
            claims=[],
            dissents=[],
            unresolved_tensions=[],
            evidence_chain=[],
            reasoning_summary="Summary",
        )
        assert proof.has_strong_consensus is False

    def test_has_strong_consensus_low_confidence(self):
        """Should return False when confidence < 0.7."""
        proof = ConsensusProof(
            proof_id="p1",
            debate_id="d1",
            task="Task",
            final_claim="Claim",
            confidence=0.6,
            consensus_reached=True,
            votes=[],
            supporting_agents=["a1", "a2", "a3", "a4", "a5"],
            dissenting_agents=[],
            claims=[],
            dissents=[],
            unresolved_tensions=[],
            evidence_chain=[],
            reasoning_summary="Summary",
        )
        assert proof.has_strong_consensus is False

    def test_has_strong_consensus_not_reached(self):
        """Should return False when consensus not reached."""
        proof = ConsensusProof(
            proof_id="p1",
            debate_id="d1",
            task="Task",
            final_claim="Claim",
            confidence=0.85,
            consensus_reached=False,
            votes=[],
            supporting_agents=["a1", "a2", "a3", "a4", "a5"],
            dissenting_agents=[],
            claims=[],
            dissents=[],
            unresolved_tensions=[],
            evidence_chain=[],
            reasoning_summary="Summary",
        )
        assert proof.has_strong_consensus is False


# =============================================================================
# Test ConsensusProof Summary Methods
# =============================================================================


class TestConsensusProofSummaries:
    """Tests for ConsensusProof summary methods."""

    def test_get_dissent_summary_empty(self, minimal_proof):
        """Should return message when no dissents."""
        summary = minimal_proof.get_dissent_summary()
        assert summary == "No dissenting views recorded."

    def test_get_dissent_summary_with_dissents(self, full_proof):
        """Should generate summary with dissents."""
        summary = full_proof.get_dissent_summary()
        assert "## Dissenting Views" in summary
        assert "agent-2" in summary
        assert "partial" in summary
        assert "Missing edge case" in summary

    def test_get_dissent_summary_with_alternative(self, full_proof):
        """Should include alternative view in summary."""
        summary = full_proof.get_dissent_summary()
        assert "**Alternative:**" in summary
        assert "async approach" in summary

    def test_get_tension_summary_empty(self, minimal_proof):
        """Should return message when no tensions."""
        summary = minimal_proof.get_tension_summary()
        assert summary == "No unresolved tensions."

    def test_get_tension_summary_with_tensions(self, full_proof):
        """Should generate summary with tensions."""
        summary = full_proof.get_tension_summary()
        assert "## Unresolved Tensions" in summary
        assert "Performance vs readability" in summary
        assert "agent-1" in summary
        assert "agent-2" in summary

    def test_get_tension_summary_with_followup(self, full_proof):
        """Should include suggested followup."""
        summary = full_proof.get_tension_summary()
        assert "**Suggested followup:**" in summary
        assert "Benchmark" in summary


# =============================================================================
# Test ConsensusProof Serialization
# =============================================================================


class TestConsensusProofSerialization:
    """Tests for ConsensusProof serialization methods."""

    def test_to_dict_contains_all_fields(self, minimal_proof):
        """Should include all fields in dict."""
        d = minimal_proof.to_dict()
        assert "proof_id" in d
        assert "debate_id" in d
        assert "task" in d
        assert "final_claim" in d
        assert "confidence" in d
        assert "consensus_reached" in d
        assert "votes" in d
        assert "supporting_agents" in d
        assert "dissenting_agents" in d
        assert "claims" in d
        assert "dissents" in d
        assert "unresolved_tensions" in d
        assert "evidence_chain" in d
        assert "reasoning_summary" in d
        assert "created_at" in d
        assert "rounds_to_consensus" in d
        assert "metadata" in d

    def test_to_dict_includes_checksum(self, minimal_proof):
        """Should include checksum in dict."""
        d = minimal_proof.to_dict()
        assert "checksum" in d
        assert d["checksum"] == minimal_proof.checksum

    def test_to_json_valid_json(self, minimal_proof):
        """Should produce valid JSON."""
        json_str = minimal_proof.to_json()
        parsed = json.loads(json_str)
        assert parsed["proof_id"] == "proof-1"

    def test_to_json_roundtrip(self, minimal_proof):
        """Should roundtrip through JSON."""
        json_str = minimal_proof.to_json()
        parsed = json.loads(json_str)
        assert parsed["final_claim"] == minimal_proof.final_claim
        assert parsed["confidence"] == minimal_proof.confidence

    def test_to_markdown_contains_header(self, minimal_proof):
        """Should contain markdown header."""
        md = minimal_proof.to_markdown()
        assert "# Consensus Proof" in md

    def test_to_markdown_contains_task(self, minimal_proof):
        """Should contain task in markdown."""
        md = minimal_proof.to_markdown()
        assert "## Task" in md
        assert "Test task" in md

    def test_to_markdown_contains_agents(self, minimal_proof):
        """Should contain agent information."""
        md = minimal_proof.to_markdown()
        assert "Supporting Agents" in md
        assert "agent-1" in md
        assert "agent-2" in md

    def test_to_markdown_contains_evidence(self, sample_evidence):
        """Should contain evidence chain."""
        # Use a proof without VoteType enums (which aren't JSON serializable in checksum)
        proof = ConsensusProof(
            proof_id="proof-ev",
            debate_id="debate-ev",
            task="Test task",
            final_claim="Final",
            confidence=0.8,
            consensus_reached=True,
            votes=[],
            supporting_agents=["agent-1"],
            dissenting_agents=[],
            claims=[],
            dissents=[],
            unresolved_tensions=[],
            evidence_chain=[sample_evidence],
            reasoning_summary="Summary",
        )
        md = proof.to_markdown()
        assert "## Evidence Chain" in md
        assert "agent-1" in md


# =============================================================================
# Test ConsensusBuilder Initialization
# =============================================================================


class TestConsensusBuilderInit:
    """Tests for ConsensusBuilder initialization."""

    def test_initialization(self):
        """Should initialize with debate_id and task."""
        builder = ConsensusBuilder(debate_id="d1", task="Test task")
        assert builder.debate_id == "d1"
        assert builder.task == "Test task"

    def test_empty_lists_on_init(self, builder):
        """Should initialize with empty lists."""
        assert builder.claims == []
        assert builder.evidence == []
        assert builder.votes == []
        assert builder.dissents == []
        assert builder.tensions == []

    def test_counters_start_at_zero(self, builder):
        """Should initialize counters at zero."""
        assert builder._claim_counter == 0
        assert builder._evidence_counter == 0


# =============================================================================
# Test ConsensusBuilder Claim Operations
# =============================================================================


class TestConsensusBuilderClaims:
    """Tests for ConsensusBuilder claim operations."""

    def test_add_claim_basic(self, builder):
        """Should add basic claim."""
        claim = builder.add_claim(
            statement="Test statement",
            author="agent-1",
        )
        assert claim.statement == "Test statement"
        assert claim.author == "agent-1"
        assert len(builder.claims) == 1

    def test_add_claim_with_confidence(self, builder):
        """Should add claim with custom confidence."""
        claim = builder.add_claim(
            statement="Test",
            author="agent-1",
            confidence=0.9,
        )
        assert claim.confidence == 0.9

    def test_add_claim_with_round(self, builder):
        """Should add claim with round number."""
        claim = builder.add_claim(
            statement="Test",
            author="agent-1",
            round_num=2,
        )
        assert claim.round_introduced == 2

    def test_add_claim_with_parent(self, builder):
        """Should add claim with parent reference."""
        parent = builder.add_claim(statement="Parent", author="agent-1")
        child = builder.add_claim(
            statement="Child",
            author="agent-2",
            parent_claim_id=parent.claim_id,
        )
        assert child.parent_claim_id == parent.claim_id

    def test_add_claim_generates_unique_ids(self, builder):
        """Should generate unique claim IDs."""
        claim1 = builder.add_claim(statement="Claim 1", author="agent-1")
        claim2 = builder.add_claim(statement="Claim 2", author="agent-2")
        assert claim1.claim_id != claim2.claim_id

    def test_add_claim_increments_counter(self, builder):
        """Should increment counter after each claim."""
        assert builder._claim_counter == 0
        builder.add_claim(statement="Claim 1", author="agent-1")
        assert builder._claim_counter == 1
        builder.add_claim(statement="Claim 2", author="agent-2")
        assert builder._claim_counter == 2


# =============================================================================
# Test ConsensusBuilder Evidence Operations
# =============================================================================


class TestConsensusBuilderEvidence:
    """Tests for ConsensusBuilder evidence operations."""

    def test_add_evidence_supporting(self, builder):
        """Should add supporting evidence."""
        claim = builder.add_claim(statement="Test", author="agent-1")
        evidence = builder.add_evidence(
            claim_id=claim.claim_id,
            source="agent-2",
            content="Supporting point",
            supports=True,
        )
        assert evidence.supports_claim is True
        assert len(builder.evidence) == 1

    def test_add_evidence_refuting(self, builder):
        """Should add refuting evidence."""
        claim = builder.add_claim(statement="Test", author="agent-1")
        evidence = builder.add_evidence(
            claim_id=claim.claim_id,
            source="agent-2",
            content="Counter point",
            supports=False,
        )
        assert evidence.supports_claim is False

    def test_add_evidence_attaches_to_claim(self, builder):
        """Should attach evidence to the correct claim."""
        claim = builder.add_claim(statement="Test", author="agent-1")
        builder.add_evidence(
            claim_id=claim.claim_id,
            source="agent-2",
            content="Support",
            supports=True,
        )
        builder.add_evidence(
            claim_id=claim.claim_id,
            source="agent-3",
            content="Refute",
            supports=False,
        )
        assert len(claim.supporting_evidence) == 1
        assert len(claim.refuting_evidence) == 1

    def test_add_evidence_nonexistent_claim(self, builder):
        """Should handle non-existent claim gracefully."""
        evidence = builder.add_evidence(
            claim_id="nonexistent",
            source="agent-1",
            content="Test",
            supports=True,
        )
        # Should still create evidence, just not attach to any claim
        assert evidence is not None
        assert len(builder.evidence) == 1

    def test_add_evidence_generates_unique_ids(self, builder):
        """Should generate unique evidence IDs."""
        claim = builder.add_claim(statement="Test", author="agent-1")
        ev1 = builder.add_evidence(
            claim_id=claim.claim_id,
            source="agent-2",
            content="Evidence 1",
        )
        ev2 = builder.add_evidence(
            claim_id=claim.claim_id,
            source="agent-3",
            content="Evidence 2",
        )
        assert ev1.evidence_id != ev2.evidence_id


# =============================================================================
# Test ConsensusBuilder Vote Operations
# =============================================================================


class TestConsensusBuilderVotes:
    """Tests for ConsensusBuilder vote operations."""

    def test_record_vote_agree(self, builder):
        """Should record agree vote."""
        vote = builder.record_vote(
            agent="agent-1",
            vote=VoteType.AGREE,
            confidence=0.9,
            reasoning="Good approach",
        )
        assert vote.vote == VoteType.AGREE
        assert len(builder.votes) == 1

    def test_record_vote_disagree(self, builder):
        """Should record disagree vote."""
        vote = builder.record_vote(
            agent="agent-1",
            vote=VoteType.DISAGREE,
            confidence=0.8,
            reasoning="Flawed logic",
        )
        assert vote.vote == VoteType.DISAGREE

    def test_record_vote_conditional(self, builder):
        """Should record conditional vote."""
        vote = builder.record_vote(
            agent="agent-1",
            vote=VoteType.CONDITIONAL,
            confidence=0.7,
            reasoning="With conditions",
        )
        assert vote.vote == VoteType.CONDITIONAL

    def test_record_vote_with_conditions(self, builder):
        """Should record vote with conditions list."""
        vote = builder.record_vote(
            agent="agent-1",
            vote=VoteType.CONDITIONAL,
            confidence=0.7,
            reasoning="If conditions met",
            conditions=["Add tests", "Fix edge case"],
        )
        assert vote.conditions == ["Add tests", "Fix edge case"]


# =============================================================================
# Test ConsensusBuilder Dissent Operations
# =============================================================================


class TestConsensusBuilderDissent:
    """Tests for ConsensusBuilder dissent operations."""

    def test_record_dissent_basic(self, builder):
        """Should record basic dissent."""
        dissent = builder.record_dissent(
            agent="agent-1",
            claim_id="claim-1",
            reasons=["Reason 1"],
        )
        assert dissent.agent == "agent-1"
        assert dissent.claim_id == "claim-1"
        assert len(builder.dissents) == 1

    def test_record_dissent_with_alternative(self, builder):
        """Should record dissent with alternative view."""
        dissent = builder.record_dissent(
            agent="agent-1",
            claim_id="claim-1",
            reasons=["Issue"],
            alternative="Better approach",
        )
        assert dissent.alternative_view == "Better approach"

    def test_record_dissent_severity(self, builder):
        """Should record dissent with custom severity."""
        dissent = builder.record_dissent(
            agent="agent-1",
            claim_id="claim-1",
            reasons=["Critical issue"],
            severity=0.9,
        )
        assert dissent.severity == 0.9


# =============================================================================
# Test ConsensusBuilder Tension Operations
# =============================================================================


class TestConsensusBuilderTension:
    """Tests for ConsensusBuilder tension operations."""

    def test_record_tension_basic(self, builder):
        """Should record basic tension."""
        tension = builder.record_tension(
            description="Test tension",
            agents=["agent-1", "agent-2"],
            options=["Option A", "Option B"],
            impact="Affects X",
        )
        assert tension.description == "Test tension"
        assert len(builder.tensions) == 1

    def test_record_tension_with_followup(self, builder):
        """Should record tension with followup suggestion."""
        tension = builder.record_tension(
            description="Test tension",
            agents=["agent-1"],
            options=["A", "B"],
            impact="Low",
            followup="Do further analysis",
        )
        assert tension.suggested_followup == "Do further analysis"

    def test_record_tension_generates_id(self, builder):
        """Should generate tension ID."""
        tension = builder.record_tension(
            description="Test",
            agents=["agent-1"],
            options=["A"],
            impact="Low",
        )
        assert tension.tension_id.startswith("test-debate-tension-")


# =============================================================================
# Test ConsensusBuilder Build
# =============================================================================


class TestConsensusBuilderBuild:
    """Tests for ConsensusBuilder build method."""

    def test_build_basic(self, builder):
        """Should build basic proof."""
        proof = builder.build(
            final_claim="Final answer",
            confidence=0.8,
            consensus_reached=True,
            reasoning_summary="Summary",
        )
        assert proof.final_claim == "Final answer"
        assert proof.confidence == 0.8
        assert proof.consensus_reached is True

    def test_build_categorizes_supporting_agents(self, builder):
        """Should categorize agree votes as supporting."""
        builder.record_vote("agent-1", VoteType.AGREE, 0.9, "Good")
        builder.record_vote("agent-2", VoteType.AGREE, 0.8, "Agreed")
        proof = builder.build(
            final_claim="Claim",
            confidence=0.9,
            consensus_reached=True,
            reasoning_summary="Summary",
        )
        assert "agent-1" in proof.supporting_agents
        assert "agent-2" in proof.supporting_agents

    def test_build_categorizes_dissenting_agents(self, builder):
        """Should categorize disagree votes as dissenting."""
        builder.record_vote("agent-1", VoteType.AGREE, 0.9, "Good")
        builder.record_vote("agent-2", VoteType.DISAGREE, 0.8, "Bad")
        proof = builder.build(
            final_claim="Claim",
            confidence=0.7,
            consensus_reached=False,
            reasoning_summary="Summary",
        )
        assert "agent-1" in proof.supporting_agents
        assert "agent-2" in proof.dissenting_agents

    def test_build_conditional_counted_as_supporting(self, builder):
        """Should count conditional votes as supporting."""
        builder.record_vote("agent-1", VoteType.CONDITIONAL, 0.7, "With conditions")
        proof = builder.build(
            final_claim="Claim",
            confidence=0.7,
            consensus_reached=True,
            reasoning_summary="Summary",
        )
        assert "agent-1" in proof.supporting_agents

    def test_build_includes_all_claims(self, builder):
        """Should include all claims in proof."""
        builder.add_claim("Claim 1", "agent-1")
        builder.add_claim("Claim 2", "agent-2")
        proof = builder.build(
            final_claim="Final",
            confidence=0.8,
            consensus_reached=True,
            reasoning_summary="Summary",
        )
        assert len(proof.claims) == 2

    def test_build_includes_all_evidence(self, builder):
        """Should include all evidence in proof."""
        claim = builder.add_claim("Claim", "agent-1")
        builder.add_evidence(claim.claim_id, "agent-2", "Evidence 1")
        builder.add_evidence(claim.claim_id, "agent-3", "Evidence 2")
        proof = builder.build(
            final_claim="Final",
            confidence=0.8,
            consensus_reached=True,
            reasoning_summary="Summary",
        )
        assert len(proof.evidence_chain) == 2

    def test_build_sets_rounds(self, builder):
        """Should set rounds to consensus."""
        proof = builder.build(
            final_claim="Final",
            confidence=0.8,
            consensus_reached=True,
            reasoning_summary="Summary",
            rounds=5,
        )
        assert proof.rounds_to_consensus == 5


# =============================================================================
# Test from_debate_result Integration
# =============================================================================


class TestFromDebateResult:
    """Tests for ConsensusBuilder.from_debate_result method."""

    def _make_mock_message(self, agent, role, content, round_num):
        """Create mock message."""
        msg = MagicMock()
        msg.agent = agent
        msg.role = role
        msg.content = content
        msg.round = round_num
        return msg

    def _make_mock_critique(self, agent, target, issues, severity, suggestions=None):
        """Create mock critique."""
        critique = MagicMock()
        critique.agent = agent
        critique.target_agent = target
        critique.issues = issues
        critique.severity = severity
        critique.suggestions = suggestions or []
        return critique

    def test_extracts_claims_from_messages(self):
        """Should extract claims from proposer messages."""
        result = MagicMock()
        result.id = "debate-1"
        result.task = "Test task"
        result.messages = [
            self._make_mock_message("agent-1", "proposer", "Proposal 1", 1),
            self._make_mock_message("agent-2", "proposer", "Proposal 2", 2),
        ]
        result.critiques = []
        result.consensus_reached = True
        result.confidence = 0.8

        builder = ConsensusBuilder.from_debate_result(result)
        assert len(builder.claims) == 2

    def test_extracts_evidence_from_critiques(self):
        """Should extract evidence from critiques."""
        result = MagicMock()
        result.id = "debate-1"
        result.task = "Test task"
        result.messages = [
            self._make_mock_message("agent-1", "proposer", "Proposal", 1),
        ]
        result.critiques = [
            self._make_mock_critique("agent-2", "agent-1", ["Issue 1", "Issue 2"], 0.5),
        ]
        result.consensus_reached = True
        result.confidence = 0.8

        builder = ConsensusBuilder.from_debate_result(result)
        # 1 proposal evidence + 2 critique evidences
        assert len(builder.evidence) == 3

    def test_records_tensions_for_high_severity(self):
        """Should record tensions for high severity critiques."""
        result = MagicMock()
        result.id = "debate-1"
        result.task = "Test task"
        result.messages = [
            self._make_mock_message("agent-1", "proposer", "Proposal", 1),
        ]
        result.critiques = [
            self._make_mock_critique("agent-2", "agent-1", ["Critical issue"], 0.8),
        ]
        result.consensus_reached = False
        result.confidence = 0.5

        builder = ConsensusBuilder.from_debate_result(result)
        assert len(builder.tensions) == 1

    def test_infers_votes_from_critiques(self):
        """Should infer votes based on critique severity."""
        result = MagicMock()
        result.id = "debate-1"
        result.task = "Test task"
        result.messages = [
            self._make_mock_message("agent-1", "proposer", "Proposal", 1),
            self._make_mock_message("agent-2", "critic", "Critique", 1),
        ]
        result.critiques = [
            self._make_mock_critique("agent-2", "agent-1", ["Minor issue"], 0.3),
        ]
        result.consensus_reached = True
        result.confidence = 0.8

        builder = ConsensusBuilder.from_debate_result(result)
        assert len(builder.votes) == 2

    def test_handles_empty_debate(self):
        """Should handle debate with no messages."""
        result = MagicMock()
        result.id = "debate-1"
        result.task = "Empty task"
        result.messages = []
        result.critiques = []
        result.consensus_reached = False
        result.confidence = 0.0

        builder = ConsensusBuilder.from_debate_result(result)
        assert len(builder.claims) == 0
        assert len(builder.votes) == 0
