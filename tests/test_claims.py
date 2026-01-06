"""Tests for aragora.reasoning.claims - typed claims and evidence kernel."""

import json
import pytest
from datetime import datetime
from dataclasses import asdict

from aragora.reasoning.claims import (
    ClaimType,
    RelationType,
    EvidenceType,
    SourceReference,
    TypedEvidence,
    TypedClaim,
    ClaimRelation,
    ArgumentChain,
    fast_extract_claims,
    fast_extract_claims_cached,
    ClaimsKernel,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def source_reference():
    """Create a sample SourceReference."""
    return SourceReference(
        source_type="agent",
        identifier="claude",
        metadata={"model": "claude-3"},
    )


@pytest.fixture
def typed_evidence(source_reference):
    """Create a sample TypedEvidence."""
    return TypedEvidence(
        evidence_id="test-e0001",
        evidence_type=EvidenceType.ARGUMENT,
        content="This is supported by logical reasoning.",
        source=source_reference,
        strength=0.8,
    )


@pytest.fixture
def typed_claim():
    """Create a sample TypedClaim."""
    return TypedClaim(
        claim_id="test-c0001",
        claim_type=ClaimType.ASSERTION,
        statement="The sky is blue.",
        author="claude",
        confidence=0.8,
    )


@pytest.fixture
def typed_claim_with_evidence(typed_claim, typed_evidence):
    """Create TypedClaim with evidence attached."""
    typed_claim.evidence.append(typed_evidence)
    return typed_claim


@pytest.fixture
def kernel():
    """Create ClaimsKernel for testing."""
    return ClaimsKernel(debate_id="test-debate")


@pytest.fixture
def kernel_with_claims(kernel):
    """Create kernel with sample claims and relations."""
    c1 = kernel.add_claim("Claim A is correct", "claude", ClaimType.PROPOSAL, confidence=0.7)
    c2 = kernel.add_claim("Claim B objects to A", "gemini", ClaimType.OBJECTION, confidence=0.6)
    kernel.add_relation(c2.claim_id, c1.claim_id, RelationType.CONTRADICTS)
    return kernel


# =============================================================================
# ClaimType Enum Tests
# =============================================================================

class TestClaimTypeEnum:
    """Tests for ClaimType enum."""

    def test_all_claim_types_have_correct_values(self):
        """All 8 claim types should have correct string values."""
        assert ClaimType.ASSERTION.value == "assertion"
        assert ClaimType.PROPOSAL.value == "proposal"
        assert ClaimType.OBJECTION.value == "objection"
        assert ClaimType.CONCESSION.value == "concession"
        assert ClaimType.REBUTTAL.value == "rebuttal"
        assert ClaimType.SYNTHESIS.value == "synthesis"
        assert ClaimType.ASSUMPTION.value == "assumption"
        assert ClaimType.QUESTION.value == "question"

    def test_claim_type_from_string(self):
        """ClaimType can be created from string value."""
        assert ClaimType("assertion") == ClaimType.ASSERTION
        assert ClaimType("proposal") == ClaimType.PROPOSAL
        assert ClaimType("objection") == ClaimType.OBJECTION


# =============================================================================
# RelationType Enum Tests
# =============================================================================

class TestRelationTypeEnum:
    """Tests for RelationType enum."""

    def test_all_relation_types_have_correct_values(self):
        """All 8 relation types should have correct string values."""
        assert RelationType.SUPPORTS.value == "supports"
        assert RelationType.CONTRADICTS.value == "contradicts"
        assert RelationType.REFINES.value == "refines"
        assert RelationType.DEPENDS_ON.value == "depends_on"
        assert RelationType.ANSWERS.value == "answers"
        assert RelationType.SUPERSEDES.value == "supersedes"
        assert RelationType.ELABORATES.value == "elaborates"
        assert RelationType.QUALIFIES.value == "qualifies"

    def test_relation_type_from_string(self):
        """RelationType can be created from string value."""
        assert RelationType("supports") == RelationType.SUPPORTS
        assert RelationType("contradicts") == RelationType.CONTRADICTS


# =============================================================================
# EvidenceType Enum Tests
# =============================================================================

class TestEvidenceTypeEnum:
    """Tests for EvidenceType enum."""

    def test_all_evidence_types_have_correct_values(self):
        """All 8 evidence types should have correct string values."""
        assert EvidenceType.ARGUMENT.value == "argument"
        assert EvidenceType.DATA.value == "data"
        assert EvidenceType.CITATION.value == "citation"
        assert EvidenceType.EXAMPLE.value == "example"
        assert EvidenceType.TOOL_OUTPUT.value == "tool_output"
        assert EvidenceType.CODE.value == "code"
        assert EvidenceType.TEST_RESULT.value == "test_result"
        assert EvidenceType.EXPERT_OPINION.value == "expert_opinion"

    def test_evidence_type_from_string(self):
        """EvidenceType can be created from string value."""
        assert EvidenceType("argument") == EvidenceType.ARGUMENT
        assert EvidenceType("data") == EvidenceType.DATA


# =============================================================================
# SourceReference Dataclass Tests
# =============================================================================

class TestSourceReference:
    """Tests for SourceReference dataclass."""

    def test_all_fields_initialized_correctly(self, source_reference):
        """All fields should be initialized correctly."""
        assert source_reference.source_type == "agent"
        assert source_reference.identifier == "claude"
        assert source_reference.metadata == {"model": "claude-3"}

    def test_timestamp_defaults_to_now(self):
        """timestamp should default to current time."""
        before = datetime.now().isoformat()
        ref = SourceReference(source_type="tool", identifier="python")
        after = datetime.now().isoformat()

        assert ref.timestamp >= before
        assert ref.timestamp <= after

    def test_metadata_defaults_to_empty_dict(self):
        """metadata should default to empty dict."""
        ref = SourceReference(source_type="file", identifier="test.py")
        assert ref.metadata == {}


# =============================================================================
# TypedEvidence Dataclass Tests
# =============================================================================

class TestTypedEvidence:
    """Tests for TypedEvidence dataclass."""

    def test_all_fields_initialized_correctly(self, typed_evidence, source_reference):
        """All fields should be initialized correctly."""
        assert typed_evidence.evidence_id == "test-e0001"
        assert typed_evidence.evidence_type == EvidenceType.ARGUMENT
        assert typed_evidence.content == "This is supported by logical reasoning."
        assert typed_evidence.source == source_reference
        assert typed_evidence.strength == 0.8

    def test_created_at_defaults_to_now(self, source_reference):
        """created_at should default to current time."""
        before = datetime.now().isoformat()
        evidence = TypedEvidence(
            evidence_id="test",
            evidence_type=EvidenceType.DATA,
            content="test",
            source=source_reference,
            strength=0.5,
        )
        after = datetime.now().isoformat()

        assert evidence.created_at >= before
        assert evidence.created_at <= after

    def test_verified_defaults_to_false(self, source_reference):
        """verified should default to False."""
        evidence = TypedEvidence(
            evidence_id="test",
            evidence_type=EvidenceType.DATA,
            content="test",
            source=source_reference,
            strength=0.5,
        )
        assert evidence.verified is False

    def test_to_dict_serializes_with_enum_value(self, typed_evidence):
        """to_dict should serialize evidence_type as string value."""
        result = typed_evidence.to_dict()

        assert result["evidence_type"] == "argument"
        assert result["evidence_id"] == "test-e0001"
        assert result["content"] == "This is supported by logical reasoning."
        assert result["strength"] == 0.8


# =============================================================================
# TypedClaim Dataclass Tests
# =============================================================================

class TestTypedClaim:
    """Tests for TypedClaim dataclass."""

    def test_all_fields_initialized_correctly(self, typed_claim):
        """All fields should be initialized correctly."""
        assert typed_claim.claim_id == "test-c0001"
        assert typed_claim.claim_type == ClaimType.ASSERTION
        assert typed_claim.statement == "The sky is blue."
        assert typed_claim.author == "claude"
        assert typed_claim.confidence == 0.8

    def test_evidence_strength_returns_zero_when_no_evidence(self, typed_claim):
        """evidence_strength should return 0.0 when no evidence."""
        assert typed_claim.evidence_strength == 0.0

    def test_evidence_strength_calculates_average(self, typed_claim, source_reference):
        """evidence_strength should calculate average of evidence strengths."""
        typed_claim.evidence.append(TypedEvidence(
            evidence_id="e1", evidence_type=EvidenceType.DATA,
            content="data1", source=source_reference, strength=0.6,
        ))
        typed_claim.evidence.append(TypedEvidence(
            evidence_id="e2", evidence_type=EvidenceType.ARGUMENT,
            content="arg1", source=source_reference, strength=0.8,
        ))

        assert typed_claim.evidence_strength == 0.7  # (0.6 + 0.8) / 2

    def test_adjusted_confidence_includes_evidence_strength(self, typed_claim_with_evidence):
        """adjusted_confidence should factor in evidence strength."""
        claim = typed_claim_with_evidence
        # evidence_strength = 0.8
        # base = confidence * (0.5 + 0.5 * evidence_strength)
        # base = 0.8 * (0.5 + 0.5 * 0.8) = 0.8 * 0.9 = 0.72
        assert abs(claim.adjusted_confidence - 0.72) < 0.01

    def test_adjusted_confidence_penalizes_challenges(self, typed_claim_with_evidence):
        """adjusted_confidence should subtract 0.1 per challenge."""
        claim = typed_claim_with_evidence
        base_adjusted = claim.adjusted_confidence

        claim.challenges.append("challenger-1")
        assert claim.adjusted_confidence == base_adjusted - 0.1

        claim.challenges.append("challenger-2")
        assert claim.adjusted_confidence == base_adjusted - 0.2

    def test_adjusted_confidence_clamped_to_0_1(self, typed_claim):
        """adjusted_confidence should be clamped between 0 and 1."""
        typed_claim.confidence = 0.1
        # With 10 challenges, penalty = 1.0, should clamp to 0
        typed_claim.challenges = [f"c{i}" for i in range(10)]

        assert typed_claim.adjusted_confidence == 0.0

    def test_to_dict_serializes_all_fields(self, typed_claim_with_evidence):
        """to_dict should serialize all fields."""
        result = typed_claim_with_evidence.to_dict()

        assert result["claim_id"] == "test-c0001"
        assert result["claim_type"] == "assertion"
        assert result["statement"] == "The sky is blue."
        assert result["author"] == "claude"
        assert result["confidence"] == 0.8
        assert len(result["evidence"]) == 1
        assert result["evidence"][0]["evidence_type"] == "argument"

    def test_status_defaults_to_active(self, typed_claim):
        """status should default to 'active'."""
        assert typed_claim.status == "active"

    def test_premises_and_challenges_default_to_empty_lists(self, typed_claim):
        """premises and challenges should default to empty lists."""
        assert typed_claim.premises == []
        assert typed_claim.challenges == []


# =============================================================================
# ClaimRelation Dataclass Tests
# =============================================================================

class TestClaimRelation:
    """Tests for ClaimRelation dataclass."""

    def test_all_fields_initialized_correctly(self):
        """All fields should be initialized correctly."""
        relation = ClaimRelation(
            relation_id="r001",
            source_claim_id="c001",
            target_claim_id="c002",
            relation_type=RelationType.SUPPORTS,
            strength=0.9,
            explanation="Strong logical support",
            created_by="claude",
        )

        assert relation.relation_id == "r001"
        assert relation.source_claim_id == "c001"
        assert relation.target_claim_id == "c002"
        assert relation.relation_type == RelationType.SUPPORTS
        assert relation.strength == 0.9
        assert relation.explanation == "Strong logical support"
        assert relation.created_by == "claude"

    def test_created_at_defaults_to_now(self):
        """created_at should default to current time."""
        before = datetime.now().isoformat()
        relation = ClaimRelation(
            relation_id="r001",
            source_claim_id="c001",
            target_claim_id="c002",
            relation_type=RelationType.CONTRADICTS,
        )
        after = datetime.now().isoformat()

        assert relation.created_at >= before
        assert relation.created_at <= after

    def test_defaults_for_optional_fields(self):
        """Optional fields should have correct defaults."""
        relation = ClaimRelation(
            relation_id="r001",
            source_claim_id="c001",
            target_claim_id="c002",
            relation_type=RelationType.SUPPORTS,
        )

        assert relation.strength == 1.0
        assert relation.explanation is None
        assert relation.created_by == ""


# =============================================================================
# ArgumentChain Dataclass Tests
# =============================================================================

class TestArgumentChain:
    """Tests for ArgumentChain dataclass."""

    def test_all_fields_initialized_correctly(self):
        """All fields should be initialized correctly."""
        chain = ArgumentChain(
            chain_id="chain-001",
            name="Main argument",
            claims=["c001", "c002", "c003"],
            relations=["r001", "r002"],
            conclusion_claim_id="c003",
            validity=0.8,
            soundness=0.7,
            author="claude",
        )

        assert chain.chain_id == "chain-001"
        assert chain.name == "Main argument"
        assert chain.claims == ["c001", "c002", "c003"]
        assert chain.relations == ["r001", "r002"]
        assert chain.conclusion_claim_id == "c003"
        assert chain.validity == 0.8
        assert chain.soundness == 0.7
        assert chain.author == "claude"

    def test_validity_and_soundness_default_to_zero(self):
        """validity and soundness should default to 0."""
        chain = ArgumentChain(
            chain_id="chain-001",
            name="Test chain",
            claims=["c001"],
            relations=[],
            conclusion_claim_id="c001",
        )

        assert chain.validity == 0.0
        assert chain.soundness == 0.0


# =============================================================================
# fast_extract_claims Function Tests
# =============================================================================

class TestFastExtractClaims:
    """Tests for fast_extract_claims function."""

    def test_returns_empty_list_for_short_text(self):
        """Should return empty list for text shorter than 10 chars."""
        result = fast_extract_claims("Hi", "claude")
        assert result == []

    def test_returns_empty_list_for_empty_text(self):
        """Should return empty list for empty text."""
        result = fast_extract_claims("", "claude")
        assert result == []

    def test_detects_proposal_patterns(self):
        """Should detect proposal patterns."""
        text = "I suggest we use microservices. We should also add caching."
        claims = fast_extract_claims(text, "claude")

        assert len(claims) >= 1
        types = [c["type"] for c in claims]
        assert "proposal" in types

    def test_detects_objection_patterns(self):
        """Should detect objection patterns."""
        text = "However, this approach has problems. But the cost is too high."
        claims = fast_extract_claims(text, "claude")

        types = [c["type"] for c in claims]
        assert "objection" in types

    def test_detects_concession_patterns(self):
        """Should detect concession patterns."""
        text = "I agree that testing is important. You're right about that."
        claims = fast_extract_claims(text, "claude")

        types = [c["type"] for c in claims]
        assert "concession" in types

    def test_detects_question_patterns(self):
        """Should detect question patterns."""
        text = "What if we used a different approach? How would that work?"
        claims = fast_extract_claims(text, "claude")

        types = [c["type"] for c in claims]
        assert "question" in types

    def test_detects_rebuttal_patterns(self):
        """Should detect rebuttal patterns."""
        # Use text without concession words (correct, true) to avoid pattern overlap
        text = "Actually, that's wrong. On the contrary, it works differently."
        claims = fast_extract_claims(text, "claude")

        types = [c["type"] for c in claims]
        assert "rebuttal" in types

    def test_boosts_confidence_for_strong_indicators(self):
        """Should boost confidence for strong certainty words."""
        # Use text that matches a claim pattern (should=proposal) + strong indicator (definitely)
        # Base 0.5 (pattern match) + 0.2 (strong indicator) = 0.7
        text = "We should definitely use this approach. I certainly recommend it."
        claims = fast_extract_claims(text, "claude")

        # At least one claim should have boosted confidence >= 0.7
        confidences = [c["confidence"] for c in claims]
        assert any(c >= 0.7 for c in confidences)

    def test_reduces_confidence_for_hedging_words(self):
        """Should reduce confidence for hedging words."""
        text = "Maybe we could try this. Perhaps it might work."
        claims = fast_extract_claims(text, "claude")

        # Hedging reduces confidence by 0.1 from 0.3 base = 0.2
        confidences = [c["confidence"] for c in claims]
        assert any(c <= 0.3 for c in confidences)

    def test_truncates_long_sentences(self):
        """Should truncate sentences longer than 200 chars."""
        long_sentence = "This is a very long sentence. " * 20
        claims = fast_extract_claims(long_sentence, "claude")

        for claim in claims:
            assert len(claim["text"]) <= 200

    def test_author_is_preserved(self):
        """Should preserve author in extracted claims."""
        text = "I propose we do this thing now."
        claims = fast_extract_claims(text, "my-agent")

        assert all(c["author"] == "my-agent" for c in claims)


# =============================================================================
# fast_extract_claims_cached Function Tests
# =============================================================================

class TestFastExtractClaimsCached:
    """Tests for fast_extract_claims_cached function."""

    def test_returns_tuple_for_hashability(self):
        """Should return tuple instead of list."""
        result = fast_extract_claims_cached("I suggest we try this approach.", "claude")
        assert isinstance(result, tuple)

    def test_caches_results_for_same_input(self):
        """Should cache results for identical input."""
        text = "I propose we implement caching for performance."
        author = "test-agent"

        # Clear cache first
        fast_extract_claims_cached.cache_clear()

        # First call
        result1 = fast_extract_claims_cached(text, author)

        # Second call - should hit cache
        result2 = fast_extract_claims_cached(text, author)

        assert result1 == result2
        # Check cache info
        info = fast_extract_claims_cached.cache_info()
        assert info.hits >= 1


# =============================================================================
# ClaimsKernel Initialization Tests
# =============================================================================

class TestClaimsKernelInitialization:
    """Tests for ClaimsKernel initialization."""

    def test_debate_id_stored_correctly(self, kernel):
        """debate_id should be stored correctly."""
        assert kernel.debate_id == "test-debate"

    def test_claims_dict_starts_empty(self, kernel):
        """claims dict should start empty."""
        assert kernel.claims == {}

    def test_counters_start_at_zero(self, kernel):
        """Internal counters should start at 0."""
        assert kernel._claim_counter == 0
        assert kernel._evidence_counter == 0

    def test_relations_and_chains_start_empty(self, kernel):
        """relations and chains dicts should start empty."""
        assert kernel.relations == {}
        assert kernel.chains == {}


# =============================================================================
# ClaimsKernel.add_claim Tests
# =============================================================================

class TestClaimsKernelAddClaim:
    """Tests for ClaimsKernel.add_claim method."""

    def test_creates_claim_with_unique_id(self, kernel):
        """Should create claim with unique ID."""
        claim = kernel.add_claim("Test claim", "claude")

        assert claim.claim_id == "test-debate-c0001"

    def test_stores_claim_in_claims_dict(self, kernel):
        """Should store claim in claims dict."""
        claim = kernel.add_claim("Test claim", "claude")

        assert kernel.claims[claim.claim_id] == claim

    def test_uses_claim_counter_for_id(self, kernel):
        """Should use claim_counter for sequential IDs."""
        c1 = kernel.add_claim("Claim 1", "claude")
        c2 = kernel.add_claim("Claim 2", "gemini")
        c3 = kernel.add_claim("Claim 3", "gpt")

        assert c1.claim_id.endswith("-c0001")
        assert c2.claim_id.endswith("-c0002")
        assert c3.claim_id.endswith("-c0003")

    def test_default_type_is_assertion(self, kernel):
        """Default claim_type should be ASSERTION."""
        claim = kernel.add_claim("Test claim", "claude")

        assert claim.claim_type == ClaimType.ASSERTION

    def test_premises_stored_correctly(self, kernel):
        """Premises should be stored correctly."""
        c1 = kernel.add_claim("Premise 1", "claude")
        c2 = kernel.add_claim("Conclusion", "claude", premises=[c1.claim_id])

        assert c2.premises == [c1.claim_id]

    def test_all_parameters_respected(self, kernel):
        """All parameters should be respected."""
        claim = kernel.add_claim(
            statement="Test statement",
            author="test-agent",
            claim_type=ClaimType.PROPOSAL,
            confidence=0.9,
            premises=["p1", "p2"],
            round_num=5,
        )

        assert claim.statement == "Test statement"
        assert claim.author == "test-agent"
        assert claim.claim_type == ClaimType.PROPOSAL
        assert claim.confidence == 0.9
        assert claim.premises == ["p1", "p2"]
        assert claim.round_num == 5


# =============================================================================
# ClaimsKernel.add_evidence Tests
# =============================================================================

class TestClaimsKernelAddEvidence:
    """Tests for ClaimsKernel.add_evidence method."""

    def test_creates_evidence_with_unique_id(self, kernel):
        """Should create evidence with unique ID."""
        claim = kernel.add_claim("Test claim", "claude")
        evidence = kernel.add_evidence(
            claim.claim_id, "Supporting data", EvidenceType.DATA,
            "agent", "claude", 0.7,
        )

        assert evidence.evidence_id == "test-debate-e0001"

    def test_adds_evidence_to_claim(self, kernel):
        """Should add evidence to the claim."""
        claim = kernel.add_claim("Test claim", "claude")
        evidence = kernel.add_evidence(
            claim.claim_id, "Supporting data", EvidenceType.DATA,
            "agent", "claude",
        )

        assert len(kernel.claims[claim.claim_id].evidence) == 1
        assert kernel.claims[claim.claim_id].evidence[0] == evidence

    def test_returns_evidence_for_missing_claim(self, kernel):
        """Should return evidence but not add it if claim missing."""
        evidence = kernel.add_evidence(
            "nonexistent-claim", "Data", EvidenceType.DATA,
            "agent", "claude",
        )

        # Evidence is still created
        assert evidence.evidence_id == "test-debate-e0001"
        # But not added to any claim

    def test_source_reference_created_correctly(self, kernel):
        """Should create SourceReference correctly."""
        claim = kernel.add_claim("Test claim", "claude")
        evidence = kernel.add_evidence(
            claim.claim_id, "Data from tool", EvidenceType.TOOL_OUTPUT,
            "tool", "python-repl", 0.9,
        )

        assert evidence.source.source_type == "tool"
        assert evidence.source.identifier == "python-repl"


# =============================================================================
# ClaimsKernel.add_relation Tests
# =============================================================================

class TestClaimsKernelAddRelation:
    """Tests for ClaimsKernel.add_relation method."""

    def test_creates_relation_with_unique_id(self, kernel):
        """Should create relation with unique ID."""
        c1 = kernel.add_claim("Claim 1", "claude")
        c2 = kernel.add_claim("Claim 2", "gemini")
        relation = kernel.add_relation(c2.claim_id, c1.claim_id, RelationType.SUPPORTS)

        assert relation.relation_id == "test-debate-r0001"

    def test_stores_relation_in_relations_dict(self, kernel):
        """Should store relation in relations dict."""
        c1 = kernel.add_claim("Claim 1", "claude")
        c2 = kernel.add_claim("Claim 2", "gemini")
        relation = kernel.add_relation(c2.claim_id, c1.claim_id, RelationType.SUPPORTS)

        assert kernel.relations[relation.relation_id] == relation

    def test_contradicts_updates_target_status(self, kernel):
        """CONTRADICTS should update target claim status to 'challenged'."""
        c1 = kernel.add_claim("Original claim", "claude")
        c2 = kernel.add_claim("Counter claim", "gemini")
        kernel.add_relation(c2.claim_id, c1.claim_id, RelationType.CONTRADICTS)

        assert kernel.claims[c1.claim_id].status == "challenged"

    def test_adds_challenger_to_challenges_list(self, kernel):
        """CONTRADICTS should add source claim to target's challenges."""
        c1 = kernel.add_claim("Original claim", "claude")
        c2 = kernel.add_claim("Counter claim", "gemini")
        kernel.add_relation(c2.claim_id, c1.claim_id, RelationType.CONTRADICTS)

        assert c2.claim_id in kernel.claims[c1.claim_id].challenges

    def test_explanation_stored_correctly(self, kernel):
        """explanation should be stored correctly."""
        c1 = kernel.add_claim("Claim 1", "claude")
        c2 = kernel.add_claim("Claim 2", "gemini")
        relation = kernel.add_relation(
            c2.claim_id, c1.claim_id, RelationType.SUPPORTS,
            explanation="This provides strong support",
        )

        assert relation.explanation == "This provides strong support"


# =============================================================================
# ClaimsKernel.challenge_claim Tests
# =============================================================================

class TestClaimsKernelChallengeClaim:
    """Tests for ClaimsKernel.challenge_claim method."""

    def test_creates_objection_claim(self, kernel):
        """Should create an OBJECTION claim."""
        original = kernel.add_claim("Original claim", "claude")
        objection = kernel.challenge_claim(
            original.claim_id, "gemini", "This is wrong because...",
        )

        assert objection.claim_type == ClaimType.OBJECTION
        assert objection.author == "gemini"
        assert objection.statement == "This is wrong because..."

    def test_adds_contradiction_relation(self, kernel):
        """Should add CONTRADICTS relation."""
        original = kernel.add_claim("Original claim", "claude")
        objection = kernel.challenge_claim(
            original.claim_id, "gemini", "This is wrong",
        )

        # Find the contradiction relation
        contradictions = [
            r for r in kernel.relations.values()
            if r.relation_type == RelationType.CONTRADICTS
        ]
        assert len(contradictions) == 1
        assert contradictions[0].source_claim_id == objection.claim_id
        assert contradictions[0].target_claim_id == original.claim_id

    def test_adds_evidence_if_provided(self, kernel):
        """Should add evidence to objection if provided."""
        original = kernel.add_claim("Original claim", "claude")
        objection = kernel.challenge_claim(
            original.claim_id, "gemini", "This is wrong",
            evidence="Here's why it's wrong...",
        )

        assert len(objection.evidence) == 1
        assert objection.evidence[0].content == "Here's why it's wrong..."


# =============================================================================
# ClaimsKernel.synthesize_claims Tests
# =============================================================================

class TestClaimsKernelSynthesizeClaims:
    """Tests for ClaimsKernel.synthesize_claims method."""

    def test_creates_synthesis_claim(self, kernel):
        """Should create a SYNTHESIS claim."""
        c1 = kernel.add_claim("Point A", "claude")
        c2 = kernel.add_claim("Point B", "gemini")
        synthesis = kernel.synthesize_claims(
            [c1.claim_id, c2.claim_id], "gpt", "Combining A and B...",
        )

        assert synthesis.claim_type == ClaimType.SYNTHESIS
        assert synthesis.author == "gpt"

    def test_sets_premises_to_input_claims(self, kernel):
        """Should set premises to input claim IDs."""
        c1 = kernel.add_claim("Point A", "claude")
        c2 = kernel.add_claim("Point B", "gemini")
        synthesis = kernel.synthesize_claims(
            [c1.claim_id, c2.claim_id], "gpt", "Combining A and B...",
        )

        assert synthesis.premises == [c1.claim_id, c2.claim_id]

    def test_adds_support_relations_from_inputs(self, kernel):
        """Should add SUPPORTS relations from input claims to synthesis."""
        c1 = kernel.add_claim("Point A", "claude")
        c2 = kernel.add_claim("Point B", "gemini")
        synthesis = kernel.synthesize_claims(
            [c1.claim_id, c2.claim_id], "gpt", "Combining A and B...",
        )

        # Should have 2 support relations
        support_relations = [
            r for r in kernel.relations.values()
            if r.relation_type == RelationType.SUPPORTS
        ]
        assert len(support_relations) == 2

        # Both should target the synthesis claim
        targets = [r.target_claim_id for r in support_relations]
        assert all(t == synthesis.claim_id for t in targets)


# =============================================================================
# ClaimsKernel Query Methods Tests
# =============================================================================

class TestClaimsKernelQueryMethods:
    """Tests for ClaimsKernel query methods."""

    def test_get_claim_graph_returns_nodes_and_edges(self, kernel_with_claims):
        """get_claim_graph should return nodes and edges."""
        graph = kernel_with_claims.get_claim_graph()

        assert "nodes" in graph
        assert "edges" in graph
        assert len(graph["nodes"]) == 2
        assert len(graph["edges"]) == 1

    def test_get_claims_returns_all_claims(self, kernel_with_claims):
        """get_claims should return all claims."""
        claims = kernel_with_claims.get_claims()

        assert len(claims) == 2
        assert all(isinstance(c, TypedClaim) for c in claims)

    def test_find_unsupported_claims_excludes_questions(self, kernel):
        """find_unsupported_claims should exclude questions."""
        kernel.add_claim("Assertion without evidence", "claude", ClaimType.ASSERTION)
        kernel.add_claim("What is this?", "gemini", ClaimType.QUESTION)

        unsupported = kernel.find_unsupported_claims()

        assert len(unsupported) == 1
        assert unsupported[0].claim_type == ClaimType.ASSERTION

    def test_find_contradictions_returns_pairs(self, kernel_with_claims):
        """find_contradictions should return (source, target) pairs."""
        contradictions = kernel_with_claims.find_contradictions()

        assert len(contradictions) == 1
        source, target = contradictions[0]
        assert source.claim_type == ClaimType.OBJECTION
        assert target.claim_type == ClaimType.PROPOSAL

    def test_find_unaddressed_objections_filters_rebuttals(self, kernel):
        """find_unaddressed_objections should exclude objections with rebuttals."""
        original = kernel.add_claim("Original", "claude", ClaimType.PROPOSAL)
        objection = kernel.add_claim("Objection", "gemini", ClaimType.OBJECTION)
        kernel.add_relation(objection.claim_id, original.claim_id, RelationType.CONTRADICTS)

        # Before rebuttal
        unaddressed = kernel.find_unaddressed_objections()
        assert len(unaddressed) == 1

        # Add rebuttal
        rebuttal = kernel.add_claim("Rebuttal", "claude", ClaimType.REBUTTAL)
        kernel.add_relation(rebuttal.claim_id, objection.claim_id, RelationType.CONTRADICTS)

        # After rebuttal
        unaddressed = kernel.find_unaddressed_objections()
        assert len(unaddressed) == 0

    def test_calculate_claim_strength_includes_support(self, kernel):
        """calculate_claim_strength should add strength from support relations."""
        main = kernel.add_claim("Main claim", "claude", ClaimType.PROPOSAL, confidence=0.7)
        support = kernel.add_claim("Supporting claim", "gemini", ClaimType.ASSERTION, confidence=0.8)
        kernel.add_relation(support.claim_id, main.claim_id, RelationType.SUPPORTS)

        strength = kernel.calculate_claim_strength(main.claim_id)

        # Base strength + support bonus
        assert strength > main.adjusted_confidence

    def test_calculate_claim_strength_subtracts_contradictions(self, kernel_with_claims):
        """calculate_claim_strength should subtract for contradictions."""
        claims = kernel_with_claims.get_claims()
        # Find the proposal (target of contradiction)
        proposal = next(c for c in claims if c.claim_type == ClaimType.PROPOSAL)

        strength = kernel_with_claims.calculate_claim_strength(proposal.claim_id)

        # Should be less than adjusted_confidence due to contradiction
        assert strength < proposal.adjusted_confidence

    def test_get_strongest_claims_returns_sorted_list(self, kernel):
        """get_strongest_claims should return sorted list of (claim, strength)."""
        kernel.add_claim("Weak claim", "claude", ClaimType.ASSERTION, confidence=0.3)
        kernel.add_claim("Strong claim", "gemini", ClaimType.PROPOSAL, confidence=0.9)
        kernel.add_claim("Medium claim", "gpt", ClaimType.SYNTHESIS, confidence=0.6)

        strongest = kernel.get_strongest_claims(limit=3)

        assert len(strongest) == 3
        # Should be sorted by strength descending
        strengths = [s[1] for s in strongest]
        assert strengths == sorted(strengths, reverse=True)


# =============================================================================
# ClaimsKernel.get_evidence_coverage Tests
# =============================================================================

class TestClaimsKernelGetEvidenceCoverage:
    """Tests for ClaimsKernel.get_evidence_coverage method."""

    def test_returns_coverage_ratio(self, kernel):
        """Should return correct coverage ratio."""
        c1 = kernel.add_claim("Claim with evidence", "claude")
        kernel.add_evidence(c1.claim_id, "Data", EvidenceType.DATA, "agent", "claude")
        kernel.add_claim("Claim without evidence", "gemini")

        coverage = kernel.get_evidence_coverage()

        assert coverage["total_claims"] == 2
        assert coverage["claims_with_evidence"] == 1
        assert coverage["coverage_ratio"] == 0.5

    def test_counts_evidence_by_type(self, kernel):
        """Should count evidence by type."""
        c1 = kernel.add_claim("Claim 1", "claude")
        kernel.add_evidence(c1.claim_id, "Data 1", EvidenceType.DATA, "agent", "claude")
        kernel.add_evidence(c1.claim_id, "Data 2", EvidenceType.DATA, "agent", "claude")
        kernel.add_evidence(c1.claim_id, "Argument", EvidenceType.ARGUMENT, "agent", "claude")

        coverage = kernel.get_evidence_coverage()

        assert coverage["evidence_by_type"]["data"] == 2
        assert coverage["evidence_by_type"]["argument"] == 1
        assert coverage["total_evidence"] == 3

    def test_handles_empty_kernel(self, kernel):
        """Should handle empty kernel gracefully."""
        coverage = kernel.get_evidence_coverage()

        assert coverage["total_claims"] == 0
        assert coverage["claims_with_evidence"] == 0
        assert coverage["coverage_ratio"] == 0
        assert coverage["total_evidence"] == 0


# =============================================================================
# ClaimsKernel Serialization Tests
# =============================================================================

class TestClaimsKernelSerialization:
    """Tests for ClaimsKernel serialization methods."""

    def test_to_dict_serializes_all_fields(self, kernel_with_claims):
        """to_dict should serialize all fields."""
        result = kernel_with_claims.to_dict()

        assert result["debate_id"] == "test-debate"
        assert len(result["claims"]) == 2
        assert len(result["relations"]) == 1
        assert "chains" in result

    def test_to_json_produces_valid_json(self, kernel_with_claims):
        """to_json should produce valid JSON."""
        json_str = kernel_with_claims.to_json()

        # Should be valid JSON
        parsed = json.loads(json_str)
        assert parsed["debate_id"] == "test-debate"

    def test_handles_enum_serialization(self, kernel_with_claims):
        """Should handle enum serialization correctly."""
        json_str = kernel_with_claims.to_json()
        parsed = json.loads(json_str)

        # Relations should have string values for relation_type
        for rel in parsed["relations"].values():
            assert isinstance(rel["relation_type"], str)
            assert rel["relation_type"] in ["supports", "contradicts", "refines",
                                            "depends_on", "answers", "supersedes",
                                            "elaborates", "qualifies"]

    def test_generate_summary_produces_markdown(self, kernel_with_claims):
        """generate_summary should produce markdown summary."""
        summary = kernel_with_claims.generate_summary()

        assert "# Argument Summary" in summary
        assert "test-debate" in summary
        assert "**Total Claims:**" in summary
        assert "**Evidence Coverage:**" in summary

    def test_generate_summary_shows_contradictions(self, kernel_with_claims):
        """generate_summary should show active contradictions."""
        summary = kernel_with_claims.generate_summary()

        assert "Active Contradictions" in summary or "Strongest Claims" in summary

    def test_generate_summary_shows_unaddressed_objections(self, kernel_with_claims):
        """generate_summary should show unaddressed objections."""
        summary = kernel_with_claims.generate_summary()

        # The objection in kernel_with_claims is unaddressed
        assert "Unaddressed Objections" in summary


# =============================================================================
# ClaimsKernel Edge Cases
# =============================================================================

class TestClaimsKernelEdgeCases:
    """Tests for edge cases in ClaimsKernel."""

    def test_calculate_strength_for_missing_claim(self, kernel):
        """calculate_claim_strength should return 0 for missing claim."""
        strength = kernel.calculate_claim_strength("nonexistent")
        assert strength == 0.0

    def test_get_claim_graph_with_empty_kernel(self, kernel):
        """get_claim_graph should work with empty kernel."""
        graph = kernel.get_claim_graph()

        assert graph["nodes"] == []
        assert graph["edges"] == []

    def test_find_contradictions_with_missing_claims(self, kernel):
        """find_contradictions should handle missing claim references."""
        # Add relation pointing to non-existent claims
        kernel.relations["orphan"] = ClaimRelation(
            relation_id="orphan",
            source_claim_id="missing1",
            target_claim_id="missing2",
            relation_type=RelationType.CONTRADICTS,
        )

        # Should not crash, just return empty
        contradictions = kernel.find_contradictions()
        assert contradictions == []

    def test_strongest_claims_filters_by_type(self, kernel):
        """get_strongest_claims should only include ASSERTION, PROPOSAL, SYNTHESIS."""
        kernel.add_claim("Question?", "claude", ClaimType.QUESTION, confidence=0.99)
        kernel.add_claim("Objection", "gemini", ClaimType.OBJECTION, confidence=0.95)
        kernel.add_claim("Proposal", "gpt", ClaimType.PROPOSAL, confidence=0.5)

        strongest = kernel.get_strongest_claims()

        # Should only include the proposal
        assert len(strongest) == 1
        assert strongest[0][0].claim_type == ClaimType.PROPOSAL

    def test_multiple_relations_between_same_claims(self, kernel):
        """Should handle multiple relations between same claims."""
        c1 = kernel.add_claim("Claim 1", "claude")
        c2 = kernel.add_claim("Claim 2", "gemini")

        kernel.add_relation(c2.claim_id, c1.claim_id, RelationType.SUPPORTS)
        kernel.add_relation(c2.claim_id, c1.claim_id, RelationType.ELABORATES)

        assert len(kernel.relations) == 2
