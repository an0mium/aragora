"""
Tests for the Claims Kernel - structured reasoning primitives.

Tests cover:
- Claim types, relation types, evidence types (enums)
- TypedClaim creation and properties
- TypedEvidence creation and serialization
- ClaimRelation creation
- ClaimsKernel operations (add_claim, add_evidence, add_relation)
- Fast claim extraction patterns
- Confidence propagation
- Claim status management
"""

import pytest

from aragora.reasoning.claims import (
    ClaimType,
    RelationType,
    EvidenceType,
    SourceReference,
    TypedEvidence,
    TypedClaim,
    ClaimRelation,
    ClaimsKernel,
    fast_extract_claims,
)


class TestClaimTypeEnum:
    """Tests for ClaimType enumeration."""

    def test_all_claim_types_defined(self):
        """Verify all expected claim types exist."""
        expected = [
            "assertion", "proposal", "objection", "concession",
            "rebuttal", "synthesis", "assumption", "question"
        ]
        actual = [ct.value for ct in ClaimType]
        assert sorted(expected) == sorted(actual)

    def test_claim_type_values(self):
        """Test specific claim type values."""
        assert ClaimType.ASSERTION.value == "assertion"
        assert ClaimType.PROPOSAL.value == "proposal"
        assert ClaimType.OBJECTION.value == "objection"


class TestRelationTypeEnum:
    """Tests for RelationType enumeration."""

    def test_all_relation_types_defined(self):
        """Verify all expected relation types exist."""
        expected = [
            "supports", "contradicts", "refines", "depends_on",
            "answers", "supersedes", "elaborates", "qualifies"
        ]
        actual = [rt.value for rt in RelationType]
        assert sorted(expected) == sorted(actual)

    def test_relation_type_values(self):
        """Test specific relation type values."""
        assert RelationType.SUPPORTS.value == "supports"
        assert RelationType.CONTRADICTS.value == "contradicts"


class TestEvidenceTypeEnum:
    """Tests for EvidenceType enumeration."""

    def test_all_evidence_types_defined(self):
        """Verify all expected evidence types exist."""
        expected = [
            "argument", "data", "citation", "example",
            "tool_output", "code", "test_result", "expert_opinion"
        ]
        actual = [et.value for et in EvidenceType]
        assert sorted(expected) == sorted(actual)


class TestSourceReference:
    """Tests for SourceReference dataclass."""

    def test_source_reference_creation(self):
        """Test basic source reference creation."""
        ref = SourceReference(
            source_type="agent",
            identifier="claude-3-opus"
        )
        assert ref.source_type == "agent"
        assert ref.identifier == "claude-3-opus"
        assert ref.timestamp is not None

    def test_source_reference_with_metadata(self):
        """Test source reference with metadata."""
        ref = SourceReference(
            source_type="url",
            identifier="https://example.com",
            metadata={"fetched_at": "2026-01-05"}
        )
        assert ref.metadata["fetched_at"] == "2026-01-05"


class TestTypedEvidence:
    """Tests for TypedEvidence dataclass."""

    def test_typed_evidence_creation(self):
        """Test basic evidence creation."""
        source = SourceReference(source_type="agent", identifier="test")
        evidence = TypedEvidence(
            evidence_id="ev-001",
            evidence_type=EvidenceType.ARGUMENT,
            content="This is a logical argument",
            source=source,
            strength=0.8,
        )
        assert evidence.evidence_id == "ev-001"
        assert evidence.evidence_type == EvidenceType.ARGUMENT
        assert evidence.strength == 0.8
        assert evidence.verified is False

    def test_typed_evidence_to_dict(self):
        """Test evidence serialization."""
        source = SourceReference(source_type="agent", identifier="test")
        evidence = TypedEvidence(
            evidence_id="ev-001",
            evidence_type=EvidenceType.DATA,
            content="Some data",
            source=source,
            strength=0.5,
        )
        d = evidence.to_dict()
        assert d["evidence_id"] == "ev-001"
        assert d["evidence_type"] == "data"
        assert d["strength"] == 0.5


class TestTypedClaim:
    """Tests for TypedClaim dataclass."""

    def test_typed_claim_creation(self):
        """Test basic claim creation."""
        claim = TypedClaim(
            claim_id="claim-001",
            claim_type=ClaimType.ASSERTION,
            statement="Python is a programming language",
            author="claude",
            confidence=0.9,
        )
        assert claim.claim_id == "claim-001"
        assert claim.claim_type == ClaimType.ASSERTION
        assert claim.statement == "Python is a programming language"
        assert claim.confidence == 0.9
        assert claim.status == "active"

    def test_evidence_strength_empty(self):
        """Test evidence strength with no evidence."""
        claim = TypedClaim(
            claim_id="claim-001",
            claim_type=ClaimType.ASSERTION,
            statement="Test",
            author="test",
            confidence=0.5,
        )
        assert claim.evidence_strength == 0.0

    def test_evidence_strength_calculated(self):
        """Test evidence strength calculation."""
        source = SourceReference(source_type="agent", identifier="test")
        evidence = [
            TypedEvidence(
                evidence_id="e1", evidence_type=EvidenceType.ARGUMENT,
                content="arg", source=source, strength=0.6
            ),
            TypedEvidence(
                evidence_id="e2", evidence_type=EvidenceType.DATA,
                content="data", source=source, strength=0.8
            ),
        ]
        claim = TypedClaim(
            claim_id="claim-001",
            claim_type=ClaimType.ASSERTION,
            statement="Test",
            author="test",
            confidence=0.5,
            evidence=evidence,
        )
        # Average of 0.6 and 0.8
        assert claim.evidence_strength == 0.7

    def test_adjusted_confidence_with_evidence(self):
        """Test adjusted confidence increases with evidence."""
        source = SourceReference(source_type="agent", identifier="test")
        evidence = [
            TypedEvidence(
                evidence_id="e1", evidence_type=EvidenceType.ARGUMENT,
                content="strong", source=source, strength=0.9
            ),
        ]
        claim = TypedClaim(
            claim_id="claim-001",
            claim_type=ClaimType.ASSERTION,
            statement="Test",
            author="test",
            confidence=0.8,
            evidence=evidence,
        )
        # Base adjusted = 0.8 * (0.5 + 0.5 * 0.9) = 0.8 * 0.95 = 0.76
        assert 0.75 < claim.adjusted_confidence < 0.78

    def test_adjusted_confidence_with_challenges(self):
        """Test adjusted confidence decreases with challenges."""
        claim = TypedClaim(
            claim_id="claim-001",
            claim_type=ClaimType.ASSERTION,
            statement="Test",
            author="test",
            confidence=0.8,
            challenges=["c1", "c2"],  # Two challenges
        )
        # Penalty = 2 * 0.1 = 0.2
        # With no evidence, base = 0.8 * 0.5 = 0.4
        # Adjusted = 0.4 - 0.2 = 0.2
        assert claim.adjusted_confidence == pytest.approx(0.2, abs=0.01)

    def test_typed_claim_to_dict(self):
        """Test claim serialization."""
        claim = TypedClaim(
            claim_id="claim-001",
            claim_type=ClaimType.PROPOSAL,
            statement="We should use Python",
            author="claude",
            confidence=0.7,
        )
        d = claim.to_dict()
        assert d["claim_id"] == "claim-001"
        assert d["claim_type"] == "proposal"
        assert d["statement"] == "We should use Python"


class TestClaimRelation:
    """Tests for ClaimRelation dataclass."""

    def test_claim_relation_creation(self):
        """Test basic relation creation."""
        relation = ClaimRelation(
            relation_id="rel-001",
            source_claim_id="c1",
            target_claim_id="c2",
            relation_type=RelationType.SUPPORTS,
            strength=0.9,
        )
        assert relation.relation_id == "rel-001"
        assert relation.source_claim_id == "c1"
        assert relation.target_claim_id == "c2"
        assert relation.relation_type == RelationType.SUPPORTS

    def test_claim_relation_with_explanation(self):
        """Test relation with explanation."""
        relation = ClaimRelation(
            relation_id="rel-001",
            source_claim_id="c1",
            target_claim_id="c2",
            relation_type=RelationType.CONTRADICTS,
            explanation="These claims are mutually exclusive",
        )
        assert relation.explanation == "These claims are mutually exclusive"


class TestClaimsKernel:
    """Tests for the ClaimsKernel reasoning engine."""

    def test_kernel_initialization(self):
        """Test kernel initialization."""
        kernel = ClaimsKernel("debate-001")
        assert kernel.debate_id == "debate-001"
        assert len(kernel.claims) == 0
        assert len(kernel.relations) == 0

    def test_add_claim(self):
        """Test adding a claim."""
        kernel = ClaimsKernel("debate-001")
        claim = kernel.add_claim(
            statement="Test assertion",
            author="claude",
            claim_type=ClaimType.ASSERTION,
            confidence=0.8,
        )
        assert claim.claim_id.startswith("debate-001-c")
        assert claim.statement == "Test assertion"
        assert claim.author == "claude"
        assert kernel.claims[claim.claim_id] == claim

    def test_add_multiple_claims(self):
        """Test adding multiple claims with unique IDs."""
        kernel = ClaimsKernel("debate-001")
        c1 = kernel.add_claim("First", "agent1", ClaimType.ASSERTION)
        c2 = kernel.add_claim("Second", "agent2", ClaimType.PROPOSAL)
        c3 = kernel.add_claim("Third", "agent1", ClaimType.OBJECTION)

        assert c1.claim_id != c2.claim_id != c3.claim_id
        assert len(kernel.claims) == 3

    def test_add_claim_with_premises(self):
        """Test adding claim with premises."""
        kernel = ClaimsKernel("debate-001")
        premise = kernel.add_claim("Premise claim", "claude", ClaimType.ASSERTION)
        conclusion = kernel.add_claim(
            "Conclusion claim",
            "claude",
            ClaimType.SYNTHESIS,
            premises=[premise.claim_id],
        )
        assert premise.claim_id in conclusion.premises

    def test_add_evidence_to_claim(self):
        """Test adding evidence to a claim."""
        kernel = ClaimsKernel("debate-001")
        claim = kernel.add_claim("Test claim", "claude", ClaimType.ASSERTION)
        evidence = kernel.add_evidence(
            claim_id=claim.claim_id,
            content="Supporting argument",
            evidence_type=EvidenceType.ARGUMENT,
            source_type="agent",
            source_id="claude",
            strength=0.7,
        )
        assert evidence.evidence_id.startswith("debate-001-e")
        assert len(claim.evidence) == 1
        assert claim.evidence[0] == evidence

    def test_add_relation(self):
        """Test adding relation between claims."""
        kernel = ClaimsKernel("debate-001")
        c1 = kernel.add_claim("First", "agent1", ClaimType.ASSERTION)
        c2 = kernel.add_claim("Second", "agent2", ClaimType.ASSERTION)

        relation = kernel.add_relation(
            source_claim_id=c1.claim_id,
            target_claim_id=c2.claim_id,
            relation_type=RelationType.SUPPORTS,
            strength=0.8,
        )
        assert relation.relation_id.startswith("debate-001-r")
        assert kernel.relations[relation.relation_id] == relation

    def test_add_contradiction_updates_status(self):
        """Test that contradicting claims update status."""
        kernel = ClaimsKernel("debate-001")
        c1 = kernel.add_claim("Original claim", "agent1", ClaimType.ASSERTION)
        c2 = kernel.add_claim("Contradicting claim", "agent2", ClaimType.OBJECTION)

        kernel.add_relation(
            source_claim_id=c2.claim_id,
            target_claim_id=c1.claim_id,
            relation_type=RelationType.CONTRADICTS,
        )

        assert c1.status == "challenged"
        assert c2.claim_id in c1.challenges


class TestFastClaimExtraction:
    """Tests for fast claim extraction using regex patterns."""

    def test_empty_text_returns_empty(self):
        """Test that empty text returns no claims."""
        assert fast_extract_claims("") == []
        assert fast_extract_claims("short") == []

    def test_extract_proposal_claim(self):
        """Test extracting proposal claims."""
        text = "We should use Python for this project."
        claims = fast_extract_claims(text, "claude")
        assert len(claims) == 1
        assert claims[0]["type"] == "proposal"
        assert claims[0]["author"] == "claude"

    def test_extract_objection_claim(self):
        """Test extracting objection claims."""
        text = "However, there are some problems with this approach."
        claims = fast_extract_claims(text, "gemini")
        assert len(claims) == 1
        assert claims[0]["type"] == "objection"

    def test_extract_question_claim(self):
        """Test extracting question claims."""
        text = "What are the performance implications?"
        claims = fast_extract_claims(text, "agent")
        assert len(claims) == 1
        assert claims[0]["type"] == "question"

    def test_extract_concession_claim(self):
        """Test extracting concession claims."""
        # Use "agree" without "concern" (which would match objection first)
        text = "I agree that you make a good point here."
        claims = fast_extract_claims(text, "agent")
        assert len(claims) == 1
        assert claims[0]["type"] == "concession"

    def test_extract_rebuttal_claim(self):
        """Test extracting rebuttal claims."""
        text = "Actually, that's not how it works in practice."
        claims = fast_extract_claims(text, "agent")
        assert len(claims) == 1
        assert claims[0]["type"] == "rebuttal"

    def test_extract_synthesis_claim(self):
        """Test extracting synthesis claims."""
        # Avoid "valid" which matches concession before "therefore" matches synthesis
        text = "Therefore, we can conclude that both approaches work well."
        claims = fast_extract_claims(text, "agent")
        assert len(claims) == 1
        assert claims[0]["type"] == "synthesis"

    def test_default_to_assertion(self):
        """Test that unmatched text defaults to assertion."""
        text = "Python is a programming language."
        claims = fast_extract_claims(text, "agent")
        assert len(claims) == 1
        assert claims[0]["type"] == "assertion"

    def test_confidence_boost_for_strong_indicators(self):
        """Test confidence boost for strong language."""
        text = "This must definitely be the solution."
        claims = fast_extract_claims(text, "agent")
        assert len(claims) == 1
        assert claims[0]["confidence"] >= 0.5

    def test_confidence_reduction_for_hedging(self):
        """Test confidence reduction for hedging language."""
        text = "This might possibly be a good approach."
        claims = fast_extract_claims(text, "agent")
        assert len(claims) == 1
        assert claims[0]["confidence"] <= 0.4

    def test_multiple_sentences_extracted(self):
        """Test extraction from multiple sentences."""
        text = "We should use Python. However, JavaScript is also good. What do you think?"
        claims = fast_extract_claims(text, "agent")
        assert len(claims) == 3
        types = [c["type"] for c in claims]
        assert "proposal" in types
        assert "objection" in types
        assert "question" in types

    def test_truncation_of_long_sentences(self):
        """Test that long sentences are truncated."""
        long_sentence = "This is a very " + "long " * 50 + "sentence."
        claims = fast_extract_claims(long_sentence, "agent")
        assert len(claims) == 1
        assert len(claims[0]["text"]) <= 200
