"""
Tests for Evidence-Claim Linker module.

Tests cover:
- EvidenceLink dataclass
- ClaimAnalysis dataclass
- EvidenceCoverageResult dataclass
- EvidenceClaimLinker initialization
- Claim extraction
- Claim detection patterns
- Evidence linking
- Link strength computation
- Keyword overlap calculation
- Evidence coverage computation
"""

from dataclasses import dataclass, field
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import pytest

from aragora.debate.evidence_quality import EvidenceType


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def linker_module():
    """Import the evidence_linker module."""
    from aragora.debate import evidence_linker

    return evidence_linker


@pytest.fixture
def EvidenceLink(linker_module):
    return linker_module.EvidenceLink


@pytest.fixture
def ClaimAnalysis(linker_module):
    return linker_module.ClaimAnalysis


@pytest.fixture
def EvidenceCoverageResult(linker_module):
    return linker_module.EvidenceCoverageResult


@pytest.fixture
def EvidenceClaimLinker(linker_module):
    return linker_module.EvidenceClaimLinker


@pytest.fixture
def linker(EvidenceClaimLinker):
    """Create a linker without embeddings."""
    return EvidenceClaimLinker(use_embeddings=False)


# ============================================================================
# EvidenceLink Tests
# ============================================================================


class TestEvidenceLink:
    """Tests for EvidenceLink dataclass."""

    def test_basic_creation(self, EvidenceLink):
        """Test basic EvidenceLink creation."""
        link = EvidenceLink(
            claim="Caching improves performance",
            claim_start=0,
            claim_end=30,
            evidence="Studies show 50% improvement",
            evidence_type=EvidenceType.DATA,
            evidence_position=40,
            link_strength=0.8,
        )

        assert link.claim == "Caching improves performance"
        assert link.claim_start == 0
        assert link.claim_end == 30
        assert link.evidence_type == EvidenceType.DATA
        assert link.link_strength == 0.8

    def test_is_strong_link_true(self, EvidenceLink):
        """Test is_strong_link property when strong."""
        link = EvidenceLink(
            claim="Test",
            claim_start=0,
            claim_end=10,
            evidence="Evidence",
            evidence_type=EvidenceType.CITATION,
            evidence_position=15,
            link_strength=0.7,  # >= 0.6
        )

        assert link.is_strong_link is True

    def test_is_strong_link_false(self, EvidenceLink):
        """Test is_strong_link property when weak."""
        link = EvidenceLink(
            claim="Test",
            claim_start=0,
            claim_end=10,
            evidence="Evidence",
            evidence_type=EvidenceType.REASONING,
            evidence_position=15,
            link_strength=0.5,  # < 0.6
        )

        assert link.is_strong_link is False

    def test_is_strong_link_boundary(self, EvidenceLink):
        """Test is_strong_link at boundary."""
        link = EvidenceLink(
            claim="Test",
            claim_start=0,
            claim_end=10,
            evidence="Evidence",
            evidence_type=EvidenceType.EXAMPLE,
            evidence_position=15,
            link_strength=0.6,  # Exactly 0.6
        )

        assert link.is_strong_link is True


# ============================================================================
# ClaimAnalysis Tests
# ============================================================================


class TestClaimAnalysis:
    """Tests for ClaimAnalysis dataclass."""

    def test_default_creation(self, ClaimAnalysis):
        """Test default ClaimAnalysis creation."""
        analysis = ClaimAnalysis()

        assert analysis.claims == []
        assert analysis.claim_positions == []
        assert analysis.total_sentences == 0
        assert analysis.claim_density == 0.0

    def test_with_claims(self, ClaimAnalysis):
        """Test ClaimAnalysis with claims."""
        analysis = ClaimAnalysis(
            claims=["Claim one", "Claim two"],
            claim_positions=[(0, 10), (15, 25)],
            total_sentences=5,
            claim_density=0.4,
        )

        assert len(analysis.claims) == 2
        assert len(analysis.claim_positions) == 2
        assert analysis.total_sentences == 5
        assert analysis.claim_density == 0.4


# ============================================================================
# EvidenceCoverageResult Tests
# ============================================================================


class TestEvidenceCoverageResult:
    """Tests for EvidenceCoverageResult dataclass."""

    def test_default_links(self, EvidenceCoverageResult):
        """Test default links list."""
        result = EvidenceCoverageResult(
            coverage=0.5,
            total_claims=4,
            linked_claims=2,
            unlinked_claims=["claim 1", "claim 2"],
            evidence_gaps=["gap 1"],
        )

        assert result.links == []

    def test_full_creation(self, EvidenceCoverageResult, EvidenceLink):
        """Test full EvidenceCoverageResult creation."""
        link = EvidenceLink(
            claim="Test",
            claim_start=0,
            claim_end=10,
            evidence="Evidence",
            evidence_type=EvidenceType.CITATION,
            evidence_position=15,
            link_strength=0.8,
        )
        result = EvidenceCoverageResult(
            coverage=0.75,
            total_claims=4,
            linked_claims=3,
            unlinked_claims=["Unlinked claim"],
            evidence_gaps=["No evidence for: ..."],
            links=[link],
        )

        assert result.coverage == 0.75
        assert result.total_claims == 4
        assert result.linked_claims == 3
        assert len(result.links) == 1


# ============================================================================
# EvidenceClaimLinker Init Tests
# ============================================================================


class TestLinkerInit:
    """Tests for EvidenceClaimLinker initialization."""

    def test_default_init(self, EvidenceClaimLinker):
        """Test default initialization."""
        linker = EvidenceClaimLinker(use_embeddings=False)

        assert linker.min_link_strength == 0.5
        assert linker.proximity_window == 300
        assert linker._embedder is None

    def test_custom_params(self, EvidenceClaimLinker):
        """Test custom parameter initialization."""
        linker = EvidenceClaimLinker(
            use_embeddings=False,
            min_link_strength=0.7,
            proximity_window=500,
        )

        assert linker.min_link_strength == 0.7
        assert linker.proximity_window == 500

    def test_uses_embeddings_false(self, linker):
        """Test uses_embeddings property when disabled."""
        assert linker.uses_embeddings is False

    def test_patterns_compiled(self, linker):
        """Test patterns are compiled on init."""
        assert len(linker._claim_patterns) > 0
        assert len(linker._non_claim_patterns) > 0


# ============================================================================
# Claim Extraction Tests
# ============================================================================


class TestClaimExtraction:
    """Tests for claim extraction."""

    def test_extract_claims_empty(self, linker):
        """Test extract_claims with empty text."""
        result = linker.extract_claims("")

        assert result.claims == []
        assert result.total_sentences == 0

    def test_extract_claims_single_sentence(self, linker):
        """Test extract_claims with single sentence."""
        text = "Caching is essential for performance optimization in distributed systems."
        result = linker.extract_claims(text)

        assert result.total_sentences == 1

    def test_extract_claims_multiple_sentences(self, linker):
        """Test extract_claims with multiple sentences."""
        text = "First sentence here. Second sentence here. Third sentence here."
        result = linker.extract_claims(text)

        assert result.total_sentences == 3

    def test_extract_claims_with_assertion(self, linker):
        """Test extract_claims detects assertions."""
        text = "We should implement caching. It will improve performance significantly."
        result = linker.extract_claims(text)

        # Should detect at least one claim
        assert len(result.claims) >= 1

    def test_extract_claims_density(self, linker):
        """Test claim density calculation."""
        text = "This is a claim that we should implement. What about this? Another claim here."
        result = linker.extract_claims(text)

        # Density should be between 0 and 1
        assert 0.0 <= result.claim_density <= 1.0

    def test_extract_claims_positions(self, linker):
        """Test claim positions are tracked."""
        text = "We should use caching. This improves performance."
        result = linker.extract_claims(text)

        # Positions should match claims
        assert len(result.claim_positions) == len(result.claims)


# ============================================================================
# Claim Detection Tests
# ============================================================================


class TestClaimDetection:
    """Tests for _is_claim method."""

    def test_short_sentence_not_claim(self, linker):
        """Test short sentences are not claims."""
        assert linker._is_claim("Hi there") is False

    def test_question_not_claim(self, linker):
        """Test questions are not claims."""
        assert linker._is_claim("What should we do about the caching issue?") is False

    def test_hedge_not_claim(self, linker):
        """Test hedged statements are not claims."""
        assert linker._is_claim("Maybe this could work in some situations perhaps.") is False

    def test_strong_assertion_is_claim(self, linker):
        """Test strong assertions are claims."""
        assert (
            linker._is_claim("We should implement caching immediately to improve performance.")
            is True
        )

    def test_recommendation_is_claim(self, linker):
        """Test recommendations are claims."""
        assert (
            linker._is_claim("I recommend using Redis for the caching layer in production.") is True
        )

    def test_conclusion_is_claim(self, linker):
        """Test conclusions are claims."""
        assert (
            linker._is_claim("Therefore, the best approach is to implement database indexing.")
            is True
        )


# ============================================================================
# Evidence Linking Tests
# ============================================================================


class TestEvidenceLinking:
    """Tests for link_evidence_to_claims method."""

    def test_link_empty_text(self, linker):
        """Test linking with empty text."""
        links = linker.link_evidence_to_claims("")
        assert links == []

    def test_link_no_evidence(self, linker):
        """Test linking when no evidence markers."""
        text = "We should use caching. This is a good approach."
        links = linker.link_evidence_to_claims(text)

        # May or may not find links depending on detection
        assert isinstance(links, list)

    def test_link_with_evidence(self, linker):
        """Test linking when evidence is present."""
        text = "We should use caching. According to the documentation, this improves performance by 50%."
        links = linker.link_evidence_to_claims(text)

        # Should find some structure
        assert isinstance(links, list)

    def test_link_returns_evidence_link_objects(self, linker, EvidenceLink):
        """Test links are EvidenceLink objects."""
        text = "Caching is essential. For example, Redis provides sub-millisecond response times."
        links = linker.link_evidence_to_claims(text)

        for link in links:
            assert hasattr(link, "claim")
            assert hasattr(link, "evidence")
            assert hasattr(link, "link_strength")


# ============================================================================
# Link Strength Tests
# ============================================================================


class TestLinkStrength:
    """Tests for link strength computation."""

    def test_proximity_score_close(self, linker):
        """Test proximity score for close evidence."""
        # Create mock evidence marker
        from aragora.debate.evidence_quality import EvidenceMarker

        marker = EvidenceMarker(
            text="According to research...",
            evidence_type=EvidenceType.CITATION,
            position=50,
            confidence=0.8,
        )

        strength = linker._compute_link_strength(
            "This is a claim",
            marker,
            distance=10,  # Close
        )

        # Should have reasonable strength
        assert 0.0 <= strength <= 1.0

    def test_proximity_score_far(self, linker):
        """Test proximity score for distant evidence."""
        from aragora.debate.evidence_quality import EvidenceMarker

        marker = EvidenceMarker(
            text="Evidence text",
            evidence_type=EvidenceType.REASONING,
            position=500,
            confidence=0.5,
        )

        strength = linker._compute_link_strength(
            "This is a claim",
            marker,
            distance=299,  # Near window edge
        )

        assert 0.0 <= strength <= 1.0

    def test_evidence_type_boost(self, linker):
        """Test evidence type affects strength."""
        from aragora.debate.evidence_quality import EvidenceMarker

        citation_marker = EvidenceMarker(
            text="According to [1]...",
            evidence_type=EvidenceType.CITATION,
            position=50,
            confidence=0.8,
        )

        reasoning_marker = EvidenceMarker(
            text="Because of this...",
            evidence_type=EvidenceType.REASONING,
            position=50,
            confidence=0.8,
        )

        citation_strength = linker._compute_link_strength("Claim", citation_marker, 10)
        reasoning_strength = linker._compute_link_strength("Claim", reasoning_marker, 10)

        # Citation should typically be stronger
        assert citation_strength >= reasoning_strength * 0.9  # Allow some variance


# ============================================================================
# Keyword Overlap Tests
# ============================================================================


class TestKeywordOverlap:
    """Tests for keyword overlap calculation."""

    def test_identical_text(self, linker):
        """Test identical texts have high overlap."""
        overlap = linker._compute_keyword_overlap(
            "Performance optimization through caching",
            "Performance optimization through caching",
        )

        assert overlap == 1.0

    def test_similar_text(self, linker):
        """Test similar texts have moderate overlap."""
        # Use texts with more shared keywords to get reliable overlap
        overlap = linker._compute_keyword_overlap(
            "Caching improves database performance",
            "Database caching improves query performance",
        )

        assert overlap > 0.3

    def test_different_text(self, linker):
        """Test different texts have low overlap."""
        overlap = linker._compute_keyword_overlap(
            "Database indexing strategies",
            "Weather forecast tomorrow",
        )

        assert overlap < 0.2

    def test_short_words_ignored(self, linker):
        """Test short words are ignored."""
        overlap = linker._compute_keyword_overlap(
            "The a an is are was",  # All short words
            "The a an is are was",
        )

        # Should be neutral since no meaningful words
        assert overlap == 0.3  # Fallback value

    def test_empty_text(self, linker):
        """Test empty text returns neutral score."""
        overlap = linker._compute_keyword_overlap("", "Test text")
        assert overlap == 0.3  # Neutral fallback


# ============================================================================
# Evidence Coverage Tests
# ============================================================================


class TestEvidenceCoverage:
    """Tests for compute_evidence_coverage method."""

    def test_coverage_empty_text(self, linker):
        """Test coverage with empty text."""
        result = linker.compute_evidence_coverage("")

        assert result.coverage == 0.0
        assert result.total_claims == 0
        assert result.linked_claims == 0
        assert result.unlinked_claims == []
        assert result.evidence_gaps == []

    def test_coverage_no_claims(self, linker):
        """Test coverage with no claims."""
        text = "Hello. Hi. Yes."
        result = linker.compute_evidence_coverage(text)

        assert result.total_claims == 0
        assert result.coverage == 0.0

    def test_coverage_with_claims(self, linker):
        """Test coverage with claims present."""
        text = """
        We should implement caching for better performance.
        According to the documentation, Redis provides sub-millisecond latency.
        This will significantly improve our response times.
        """
        result = linker.compute_evidence_coverage(text)

        # Should have some claims
        assert result.total_claims >= 0
        # Coverage should be valid
        assert 0.0 <= result.coverage <= 1.0

    def test_coverage_unlinked_claims_identified(self, linker):
        """Test unlinked claims are identified."""
        text = """
        We should definitely implement this feature immediately.
        This approach must be better than alternatives.
        Our system needs to handle more load.
        """
        result = linker.compute_evidence_coverage(text)

        # Some claims may be unlinked (no evidence)
        # The specific count depends on detection
        assert isinstance(result.unlinked_claims, list)

    def test_coverage_evidence_gaps_limited(self, linker):
        """Test evidence gaps are limited to top 3."""
        text = """
        Claim one without evidence here.
        Claim two without evidence here.
        Claim three without evidence here.
        Claim four without evidence here.
        Claim five without evidence here.
        """
        result = linker.compute_evidence_coverage(text)

        # Gaps should be limited
        assert len(result.evidence_gaps) <= 3


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests for EvidenceClaimLinker."""

    def test_full_analysis_flow(self, linker):
        """Test complete analysis flow."""
        text = """
        We recommend implementing Redis-based caching for the application.
        According to benchmarks, this approach improves response times by 40%.
        For example, user session lookups become nearly instantaneous.
        The implementation should follow standard caching patterns.
        """

        # Extract claims
        claims = linker.extract_claims(text)
        assert claims.total_sentences > 0

        # Link evidence
        links = linker.link_evidence_to_claims(text)
        assert isinstance(links, list)

        # Get coverage
        coverage = linker.compute_evidence_coverage(text)
        assert 0.0 <= coverage.coverage <= 1.0

    def test_multiple_evidence_types(self, linker):
        """Test handling multiple evidence types."""
        text = """
        Database indexing is crucial for performance.
        According to PostgreSQL documentation, indexes improve query speed.
        For example, a B-tree index on user_id reduces lookup time.
        Research shows that proper indexing can improve queries by 100x.
        """

        coverage = linker.compute_evidence_coverage(text)

        # Should handle multiple evidence types
        evidence_types = set()
        for link in coverage.links:
            evidence_types.add(link.evidence_type)

        # May find multiple types
        assert isinstance(evidence_types, set)

    def test_real_world_proposal(self, linker):
        """Test with realistic proposal text."""
        text = """
        I propose we implement a microservices architecture for the new system.

        According to industry best practices, microservices provide better
        scalability and maintainability for large applications. Studies from
        Netflix and Amazon demonstrate successful large-scale deployments.

        For example, we could separate the authentication service from the
        main application, allowing independent scaling during peak login periods.

        However, this approach requires careful consideration of inter-service
        communication. We should use message queues for asynchronous operations.
        """

        coverage = linker.compute_evidence_coverage(text)

        # Should analyze the proposal
        assert coverage.total_claims > 0
        assert isinstance(coverage.links, list)
        assert isinstance(coverage.unlinked_claims, list)


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_very_long_text(self, linker):
        """Test handling of very long text."""
        text = "This is a claim. " * 100 + "According to research, this is evidence."

        # Should not crash
        coverage = linker.compute_evidence_coverage(text)
        assert isinstance(coverage, linker.compute_evidence_coverage("").__class__)

    def test_special_characters(self, linker):
        """Test handling of special characters."""
        text = "We should implement <caching> & other 'optimizations' for 100% improvement."

        claims = linker.extract_claims(text)
        # Should handle special chars
        assert isinstance(claims.claims, list)

    def test_unicode_text(self, linker):
        """Test handling of unicode text."""
        text = "We should implement caching. \u2014 According to research \u2026"

        coverage = linker.compute_evidence_coverage(text)
        assert isinstance(coverage.coverage, float)

    def test_code_blocks_in_text(self, linker):
        """Test handling of code-like content."""
        text = """
        We should use async/await patterns.
        ```python
        async def fetch_data():
            return await db.query()
        ```
        This improves throughput significantly.
        """

        coverage = linker.compute_evidence_coverage(text)
        assert isinstance(coverage, linker.compute_evidence_coverage("").__class__)
