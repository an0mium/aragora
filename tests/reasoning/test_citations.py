"""
Tests for Scholarly Citation Grounding.

Tests the citation module including:
- CitationType and CitationQuality enums
- ScholarlyEvidence dataclass and formatting
- CitedClaim with grounding scores
- GroundedVerdict with bibliography
- CitationExtractor for claim detection
- CitationStore for citation management
- URL-based citation creation
"""

from __future__ import annotations

from datetime import datetime

import pytest

from aragora.reasoning.citations import (
    CitationType,
    CitationQuality,
    ScholarlyEvidence,
    CitedClaim,
    GroundedVerdict,
    CitationExtractor,
    CitationStore,
    create_citation_from_url,
)


# =============================================================================
# CitationType Enum Tests
# =============================================================================


class TestCitationType:
    """Test CitationType enum."""

    def test_academic_types(self):
        """Test academic citation types."""
        assert CitationType.ACADEMIC_PAPER.value == "academic_paper"
        assert CitationType.BOOK.value == "book"
        assert CitationType.CONFERENCE.value == "conference"
        assert CitationType.PREPRINT.value == "preprint"

    def test_official_types(self):
        """Test official/documentation citation types."""
        assert CitationType.DOCUMENTATION.value == "documentation"
        assert CitationType.OFFICIAL_SOURCE.value == "official_source"

    def test_informal_types(self):
        """Test informal citation types."""
        assert CitationType.NEWS_ARTICLE.value == "news_article"
        assert CitationType.BLOG_POST.value == "blog_post"
        assert CitationType.WEB_PAGE.value == "web_page"

    def test_special_types(self):
        """Test special citation types."""
        assert CitationType.CODE_REPOSITORY.value == "code_repository"
        assert CitationType.DATASET.value == "dataset"
        assert CitationType.INTERNAL_DEBATE.value == "internal_debate"
        assert CitationType.UNKNOWN.value == "unknown"


# =============================================================================
# CitationQuality Enum Tests
# =============================================================================


class TestCitationQuality:
    """Test CitationQuality enum."""

    def test_quality_levels(self):
        """Test quality level values."""
        assert CitationQuality.PEER_REVIEWED.value == "peer_reviewed"
        assert CitationQuality.AUTHORITATIVE.value == "authoritative"
        assert CitationQuality.REPUTABLE.value == "reputable"
        assert CitationQuality.MIXED.value == "mixed"
        assert CitationQuality.UNVERIFIED.value == "unverified"
        assert CitationQuality.QUESTIONABLE.value == "questionable"


# =============================================================================
# ScholarlyEvidence Tests
# =============================================================================


class TestScholarlyEvidence:
    """Test ScholarlyEvidence dataclass."""

    def test_basic_creation(self):
        """Test basic ScholarlyEvidence creation."""
        evidence = ScholarlyEvidence(
            title="Test Paper",
            authors=["John Doe", "Jane Smith"],
            publication="Nature",
            year=2023,
        )
        assert evidence.title == "Test Paper"
        assert evidence.authors == ["John Doe", "Jane Smith"]
        assert evidence.publication == "Nature"
        assert evidence.year == 2023

    def test_auto_id_generation(self):
        """Test automatic ID generation from content."""
        evidence = ScholarlyEvidence(
            title="Test Paper",
            authors=["John Doe"],
            year=2023,
        )
        assert evidence.id  # Should be non-empty
        assert len(evidence.id) == 12  # SHA256 truncated to 12 chars

    def test_deterministic_id(self):
        """Test ID is deterministic for same content."""
        evidence1 = ScholarlyEvidence(
            title="Test Paper",
            authors=["John Doe"],
            year=2023,
        )
        evidence2 = ScholarlyEvidence(
            title="Test Paper",
            authors=["John Doe"],
            year=2023,
        )
        assert evidence1.id == evidence2.id

    def test_different_content_different_id(self):
        """Test different content produces different ID."""
        evidence1 = ScholarlyEvidence(
            title="Paper One",
            authors=["John Doe"],
            year=2023,
        )
        evidence2 = ScholarlyEvidence(
            title="Paper Two",
            authors=["John Doe"],
            year=2023,
        )
        assert evidence1.id != evidence2.id

    def test_format_apa_academic_paper(self):
        """Test APA formatting for academic paper."""
        evidence = ScholarlyEvidence(
            citation_type=CitationType.ACADEMIC_PAPER,
            title="Deep Learning",
            authors=["Smith, J.", "Jones, K."],
            publication="Nature",
            year=2023,
        )
        apa = evidence.format_apa()
        assert "Smith, J., Jones, K." in apa
        assert "(2023)" in apa
        assert "Deep Learning" in apa
        assert "Nature" in apa

    def test_format_apa_with_et_al(self):
        """Test APA formatting with more than 3 authors."""
        evidence = ScholarlyEvidence(
            citation_type=CitationType.ACADEMIC_PAPER,
            title="Many Authors Paper",
            authors=["Author1", "Author2", "Author3", "Author4", "Author5"],
            publication="Journal",
            year=2022,
        )
        apa = evidence.format_apa()
        assert "et al." in apa
        # First 3 authors should be listed
        assert "Author1" in apa

    def test_format_apa_no_year(self):
        """Test APA formatting without year."""
        evidence = ScholarlyEvidence(
            citation_type=CitationType.BOOK,
            title="Timeless Book",
            authors=["Ancient Author"],
            publication="Publisher",
        )
        apa = evidence.format_apa()
        assert "(n.d.)" in apa

    def test_format_apa_web_page(self):
        """Test APA formatting for web page."""
        evidence = ScholarlyEvidence(
            citation_type=CitationType.WEB_PAGE,
            title="Web Article",
            authors=["Web Author"],
            url="https://example.com/article",
            year=2024,
        )
        apa = evidence.format_apa()
        assert "Retrieved from" in apa
        assert "https://example.com/article" in apa

    def test_format_inline_single_author(self):
        """Test inline citation with single author."""
        evidence = ScholarlyEvidence(
            title="Paper",
            authors=["John Smith"],
            year=2023,
        )
        inline = evidence.format_inline()
        assert "(Smith, 2023)" == inline

    def test_format_inline_two_authors(self):
        """Test inline citation with two authors."""
        evidence = ScholarlyEvidence(
            title="Paper",
            authors=["John Smith", "Jane Doe"],
            year=2023,
        )
        inline = evidence.format_inline()
        assert "(Smith & Doe, 2023)" == inline

    def test_format_inline_many_authors(self):
        """Test inline citation with many authors."""
        evidence = ScholarlyEvidence(
            title="Paper",
            authors=["John Smith", "Jane Doe", "Bob Wilson"],
            year=2023,
        )
        inline = evidence.format_inline()
        assert "et al." in inline
        assert "2023" in inline

    def test_format_inline_no_authors(self):
        """Test inline citation without authors."""
        evidence = ScholarlyEvidence(
            title="Anonymous Paper with Long Title",
            year=2023,
        )
        inline = evidence.format_inline()
        assert "2023" in inline
        assert "..." in inline  # Title is truncated

    def test_quality_score_academic_paper(self):
        """Test quality score for academic paper."""
        evidence = ScholarlyEvidence(
            citation_type=CitationType.ACADEMIC_PAPER,
            title="High Quality Paper",
            peer_reviewed=True,
            doi="10.1000/example",
            verified=True,
        )
        score = evidence.quality_score()
        assert score > 0.9  # Academic paper + peer reviewed + DOI + verified

    def test_quality_score_blog_post(self):
        """Test quality score for blog post."""
        evidence = ScholarlyEvidence(
            citation_type=CitationType.BLOG_POST,
            title="Blog Post",
        )
        score = evidence.quality_score()
        assert score < 0.5  # Blog posts have low base score

    def test_quality_score_high_citation_count(self):
        """Test quality score boost from citation count."""
        evidence = ScholarlyEvidence(
            citation_type=CitationType.PREPRINT,
            title="Popular Preprint",
            citation_count=500,
        )
        score = evidence.quality_score()
        # Preprint (0.6) + citation boost (0.05)
        assert score >= 0.6

    def test_to_dict(self):
        """Test serialization to dictionary."""
        evidence = ScholarlyEvidence(
            citation_type=CitationType.ACADEMIC_PAPER,
            title="Test Paper",
            authors=["Author One"],
            publication="Journal",
            year=2023,
            url="https://example.com",
            doi="10.1000/test",
            excerpt="Important finding",
            relevance_score=0.8,
            quality=CitationQuality.PEER_REVIEWED,
            peer_reviewed=True,
        )
        data = evidence.to_dict()

        assert data["citation_type"] == "academic_paper"
        assert data["title"] == "Test Paper"
        assert data["authors"] == ["Author One"]
        assert data["year"] == 2023
        assert data["peer_reviewed"] is True
        assert "quality_score" in data


# =============================================================================
# CitedClaim Tests
# =============================================================================


class TestCitedClaim:
    """Test CitedClaim dataclass."""

    def test_basic_creation(self):
        """Test basic CitedClaim creation."""
        claim = CitedClaim(claim_text="Test claim about something.")
        assert claim.claim_text == "Test claim about something."
        assert claim.claim_id  # Auto-generated
        assert len(claim.claim_id) == 12

    def test_deterministic_claim_id(self):
        """Test claim ID is deterministic."""
        claim1 = CitedClaim(claim_text="Same claim text.")
        claim2 = CitedClaim(claim_text="Same claim text.")
        assert claim1.claim_id == claim2.claim_id

    def test_grounding_score_no_citations(self):
        """Test grounding score with no citations."""
        claim = CitedClaim(claim_text="Ungrounded claim.")
        assert claim.grounding_score == 0.0

    def test_grounding_score_with_citations(self):
        """Test grounding score with citations."""
        citation = ScholarlyEvidence(
            citation_type=CitationType.ACADEMIC_PAPER,
            title="Supporting Paper",
            relevance_score=0.9,
            peer_reviewed=True,
        )
        claim = CitedClaim(
            claim_text="Well-grounded claim.",
            citations=[citation],
        )
        assert claim.grounding_score > 0.0

    def test_to_dict(self):
        """Test serialization to dictionary."""
        citation = ScholarlyEvidence(
            title="Paper",
            authors=["Author"],
            year=2023,
        )
        claim = CitedClaim(
            claim_text="Test claim.",
            citations=[citation],
            confidence=0.85,
        )
        data = claim.to_dict()

        assert data["claim_text"] == "Test claim."
        assert data["confidence"] == 0.85
        assert len(data["citations"]) == 1


# =============================================================================
# GroundedVerdict Tests
# =============================================================================


class TestGroundedVerdict:
    """Test GroundedVerdict dataclass."""

    def test_basic_creation(self):
        """Test basic GroundedVerdict creation."""
        verdict = GroundedVerdict(
            verdict="The hypothesis is supported.",
            confidence=0.85,
        )
        assert verdict.verdict == "The hypothesis is supported."
        assert verdict.confidence == 0.85
        assert verdict.grounding_score == 0.0  # No claims

    def test_citation_collection(self):
        """Test citations are collected from claims."""
        citation1 = ScholarlyEvidence(
            title="Paper 1",
            authors=["Author 1"],
            year=2022,
        )
        citation2 = ScholarlyEvidence(
            title="Paper 2",
            authors=["Author 2"],
            year=2023,
        )
        claim1 = CitedClaim(claim_text="Claim 1", citations=[citation1])
        claim2 = CitedClaim(claim_text="Claim 2", citations=[citation2])

        verdict = GroundedVerdict(
            verdict="Supported",
            confidence=0.9,
            claims=[claim1, claim2],
        )

        assert len(verdict.all_citations) == 2

    def test_citation_deduplication(self):
        """Test duplicate citations are removed."""
        citation = ScholarlyEvidence(
            title="Shared Paper",
            authors=["Author"],
            year=2023,
        )
        claim1 = CitedClaim(claim_text="Claim 1", citations=[citation])
        claim2 = CitedClaim(claim_text="Claim 2", citations=[citation])

        verdict = GroundedVerdict(
            verdict="Supported",
            confidence=0.9,
            claims=[claim1, claim2],
        )

        # Same citation used twice, but should only appear once
        assert len(verdict.all_citations) == 1

    def test_format_bibliography(self):
        """Test bibliography formatting."""
        citation = ScholarlyEvidence(
            citation_type=CitationType.ACADEMIC_PAPER,
            title="Important Paper",
            authors=["Researcher, A."],
            publication="Science",
            year=2023,
        )
        claim = CitedClaim(claim_text="Claim", citations=[citation])
        verdict = GroundedVerdict(
            verdict="Conclusion",
            confidence=0.9,
            claims=[claim],
        )

        bibliography = verdict.format_bibliography()
        assert "References:" in bibliography
        assert "[1]" in bibliography
        assert "Important Paper" in bibliography

    def test_format_bibliography_empty(self):
        """Test bibliography with no citations."""
        verdict = GroundedVerdict(
            verdict="Unsupported conclusion",
            confidence=0.5,
        )
        bibliography = verdict.format_bibliography()
        assert "No citations" in bibliography

    def test_summary(self):
        """Test summary generation."""
        citation = ScholarlyEvidence(
            title="Paper",
            authors=["Smith, J."],
            year=2023,
        )
        claim = CitedClaim(claim_text="Claim", citations=[citation])
        verdict = GroundedVerdict(
            verdict="The analysis shows positive results.",
            confidence=0.85,
            claims=[claim],
        )

        summary = verdict.summary()
        assert "GROUNDED VERDICT" in summary
        assert "85%" in summary  # Confidence formatted as percentage
        assert "1 citations" in summary

    def test_to_dict(self):
        """Test serialization to dictionary."""
        verdict = GroundedVerdict(
            verdict="Conclusion",
            confidence=0.9,
            claims=[],
        )
        data = verdict.to_dict()

        assert data["verdict"] == "Conclusion"
        assert data["confidence"] == 0.9
        assert "claims" in data
        assert "all_citations" in data


# =============================================================================
# CitationExtractor Tests
# =============================================================================


class TestCitationExtractor:
    """Test CitationExtractor class."""

    @pytest.fixture
    def extractor(self):
        """Create a CitationExtractor instance."""
        return CitationExtractor()

    def test_extract_research_claim(self, extractor):
        """Test extraction of research claims."""
        text = "Research shows that exercise improves mental health. Walking is good."
        claims = extractor.extract_claims(text)
        assert len(claims) == 1
        assert "Research shows" in claims[0]

    def test_extract_study_claim(self, extractor):
        """Test extraction of study claims."""
        text = "Studies have found that sleep affects memory consolidation."
        claims = extractor.extract_claims(text)
        assert len(claims) == 1
        assert "Studies have found" in claims[0]

    def test_extract_percentage_claim(self, extractor):
        """Test extraction of percentage claims."""
        text = "About 70% of users prefer the new design."
        claims = extractor.extract_claims(text)
        assert len(claims) == 1
        assert "70%" in claims[0]

    def test_extract_multiple_claims(self, extractor):
        """Test extraction of multiple claims."""
        text = """
        Research shows that A is true.
        According to experts, B is also valid.
        Evidence suggests that C follows from A and B.
        """
        claims = extractor.extract_claims(text)
        assert len(claims) == 3

    def test_no_claims_in_text(self, extractor):
        """Test text without claims."""
        text = "The sky is blue. Water is wet. This is a simple statement."
        claims = extractor.extract_claims(text)
        assert len(claims) == 0

    def test_case_insensitive(self, extractor):
        """Test case-insensitive pattern matching."""
        text = "RESEARCH SHOWS that this works. Evidence Suggests it's true."
        claims = extractor.extract_claims(text)
        assert len(claims) == 2

    def test_identify_citation_needs(self, extractor):
        """Test citation needs identification."""
        text = "Research shows that data indicates 50% improvement."
        needs = extractor.identify_citation_needs(text)
        assert len(needs) == 1
        assert needs[0]["priority"] == "high"  # Contains "data" and "%"

    def test_citation_needs_priority_levels(self, extractor):
        """Test priority levels for citation needs."""
        text_high = "Data indicates this is true."
        text_medium = "According to sources, this is true."
        text_low = "Evidence suggests this may be true."

        needs_high = extractor.identify_citation_needs(text_high)
        needs_medium = extractor.identify_citation_needs(text_medium)
        needs_low = extractor.identify_citation_needs(text_low)

        assert needs_high[0]["priority"] == "high"
        assert needs_low[0]["priority"] == "low"

    def test_suggest_source_types_research(self, extractor):
        """Test source type suggestions for research claims."""
        text = "Research shows the scientific evidence."
        needs = extractor.identify_citation_needs(text)
        suggested = needs[0]["suggested_source_types"]
        assert CitationType.ACADEMIC_PAPER in suggested

    def test_suggest_source_types_code(self, extractor):
        """Test source type suggestions for code claims."""
        text = "Best practice in software programming."
        needs = extractor.identify_citation_needs(text)
        suggested = needs[0]["suggested_source_types"]
        assert CitationType.DOCUMENTATION in suggested or CitationType.CODE_REPOSITORY in suggested

    def test_suggest_source_types_legal(self, extractor):
        """Test source type suggestions for legal claims."""
        text = "According to the regulation standards."
        needs = extractor.identify_citation_needs(text)
        suggested = needs[0]["suggested_source_types"]
        assert CitationType.OFFICIAL_SOURCE in suggested


# =============================================================================
# CitationStore Tests
# =============================================================================


class TestCitationStore:
    """Test CitationStore class."""

    @pytest.fixture
    def store(self):
        """Create a CitationStore instance."""
        return CitationStore()

    @pytest.fixture
    def sample_citation(self):
        """Create a sample citation."""
        return ScholarlyEvidence(
            title="Machine Learning Fundamentals",
            authors=["Smith, J."],
            publication="Nature ML",
            year=2023,
            excerpt="Deep learning has revolutionized AI.",
        )

    def test_add_citation(self, store, sample_citation):
        """Test adding a citation."""
        citation_id = store.add(sample_citation)
        assert citation_id == sample_citation.id
        assert sample_citation.id in store.citations

    def test_get_citation(self, store, sample_citation):
        """Test retrieving a citation."""
        store.add(sample_citation)
        retrieved = store.get(sample_citation.id)
        assert retrieved is not None
        assert retrieved.title == sample_citation.title

    def test_get_nonexistent_citation(self, store):
        """Test retrieving non-existent citation."""
        retrieved = store.get("nonexistent-id")
        assert retrieved is None

    def test_find_for_claim_keyword_match(self, store, sample_citation):
        """Test finding citations by keyword match."""
        store.add(sample_citation)
        results = store.find_for_claim("machine learning AI")
        assert len(results) > 0
        assert results[0].id == sample_citation.id

    def test_find_for_claim_no_match(self, store, sample_citation):
        """Test finding citations with no match."""
        store.add(sample_citation)
        results = store.find_for_claim("quantum physics chemistry")
        assert len(results) == 0

    def test_find_for_claim_limit(self, store):
        """Test finding citations with limit."""
        for i in range(10):
            citation = ScholarlyEvidence(
                title=f"Paper about topic {i}",
                authors=["Author"],
                year=2023,
                excerpt="topic is important",
            )
            store.add(citation)

        results = store.find_for_claim("topic important", limit=3)
        assert len(results) == 3

    def test_link_claim_to_citation(self, store, sample_citation):
        """Test linking claim to citation."""
        store.add(sample_citation)
        store.link_claim_to_citation("claim-123", sample_citation.id)

        assert "claim-123" in store.claim_to_citations
        assert sample_citation.id in store.claim_to_citations["claim-123"]

    def test_link_claim_multiple_citations(self, store):
        """Test linking claim to multiple citations."""
        citation1 = ScholarlyEvidence(title="Paper 1", authors=["A"], year=2022)
        citation2 = ScholarlyEvidence(title="Paper 2", authors=["B"], year=2023)
        store.add(citation1)
        store.add(citation2)

        store.link_claim_to_citation("claim-123", citation1.id)
        store.link_claim_to_citation("claim-123", citation2.id)

        assert len(store.claim_to_citations["claim-123"]) == 2

    def test_link_claim_no_duplicates(self, store, sample_citation):
        """Test linking same citation twice doesn't duplicate."""
        store.add(sample_citation)
        store.link_claim_to_citation("claim-123", sample_citation.id)
        store.link_claim_to_citation("claim-123", sample_citation.id)

        assert len(store.claim_to_citations["claim-123"]) == 1

    def test_get_citations_for_claim(self, store, sample_citation):
        """Test getting citations for a claim."""
        store.add(sample_citation)
        store.link_claim_to_citation("claim-123", sample_citation.id)

        citations = store.get_citations_for_claim("claim-123")
        assert len(citations) == 1
        assert citations[0].id == sample_citation.id

    def test_get_citations_for_nonexistent_claim(self, store):
        """Test getting citations for non-existent claim."""
        citations = store.get_citations_for_claim("nonexistent")
        assert len(citations) == 0


# =============================================================================
# create_citation_from_url Tests
# =============================================================================


class TestCreateCitationFromUrl:
    """Test create_citation_from_url helper function."""

    def test_arxiv_url(self):
        """Test citation creation from arXiv URL."""
        citation = create_citation_from_url(
            "https://arxiv.org/abs/2301.12345",
            title="ArXiv Paper",
        )
        assert citation.citation_type == CitationType.PREPRINT
        assert citation.quality == CitationQuality.REPUTABLE
        assert citation.title == "ArXiv Paper"

    def test_doi_url(self):
        """Test citation creation from DOI URL."""
        citation = create_citation_from_url(
            "https://doi.org/10.1000/example",
            title="DOI Paper",
        )
        assert citation.citation_type == CitationType.ACADEMIC_PAPER
        assert citation.quality == CitationQuality.PEER_REVIEWED

    def test_pubmed_url(self):
        """Test citation creation from PubMed URL."""
        citation = create_citation_from_url(
            "https://pubmed.ncbi.nlm.nih.gov/12345678",
            title="PubMed Article",
        )
        assert citation.citation_type == CitationType.ACADEMIC_PAPER
        assert citation.quality == CitationQuality.PEER_REVIEWED

    def test_github_url(self):
        """Test citation creation from GitHub URL."""
        citation = create_citation_from_url(
            "https://github.com/user/repo",
            title="Code Repository",
        )
        assert citation.citation_type == CitationType.CODE_REPOSITORY
        assert citation.quality == CitationQuality.REPUTABLE

    def test_gov_url(self):
        """Test citation creation from .gov URL."""
        citation = create_citation_from_url(
            "https://www.example.gov/report",
            title="Government Report",
        )
        assert citation.citation_type == CitationType.OFFICIAL_SOURCE
        assert citation.quality == CitationQuality.AUTHORITATIVE

    def test_edu_url(self):
        """Test citation creation from .edu URL."""
        citation = create_citation_from_url(
            "https://www.university.edu/research",
            title="University Research",
        )
        assert citation.citation_type == CitationType.OFFICIAL_SOURCE
        assert citation.quality == CitationQuality.AUTHORITATIVE

    def test_docs_url(self):
        """Test citation creation from documentation URL."""
        citation = create_citation_from_url(
            "https://docs.python.org/3/library/",
            title="Python Documentation",
        )
        assert citation.citation_type == CitationType.DOCUMENTATION
        assert citation.quality == CitationQuality.REPUTABLE

    def test_generic_url(self):
        """Test citation creation from generic URL."""
        citation = create_citation_from_url(
            "https://random-site.com/article",
            title="Generic Article",
            excerpt="Some content",
        )
        assert citation.citation_type == CitationType.WEB_PAGE
        assert citation.quality == CitationQuality.UNVERIFIED
        assert citation.excerpt == "Some content"

    def test_url_stored(self):
        """Test URL is stored in citation."""
        url = "https://example.com/article"
        citation = create_citation_from_url(url, title="Article")
        assert citation.url == url
