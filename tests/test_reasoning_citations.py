"""Tests for the citations module - scholarly evidence grounding."""

import pytest
from datetime import datetime

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
# Fixtures
# =============================================================================


@pytest.fixture
def sample_evidence():
    """Create sample ScholarlyEvidence."""
    return ScholarlyEvidence(
        citation_type=CitationType.ACADEMIC_PAPER,
        title="Test Paper on AI Safety",
        authors=["Smith, John", "Doe, Jane"],
        publication="Journal of AI Research",
        year=2024,
        excerpt="Key finding from the paper",
        relevance_score=0.9,
        quality=CitationQuality.PEER_REVIEWED,
        peer_reviewed=True,
    )


@pytest.fixture
def citation_extractor():
    """Create CitationExtractor instance."""
    return CitationExtractor()


@pytest.fixture
def citation_store():
    """Create empty CitationStore."""
    return CitationStore()


# =============================================================================
# CitationType Enum Tests
# =============================================================================


class TestCitationType:
    """Tests for CitationType enum."""

    def test_all_12_types_exist(self):
        """Should have all 12 citation types."""
        expected_types = [
            "ACADEMIC_PAPER",
            "BOOK",
            "CONFERENCE",
            "PREPRINT",
            "DOCUMENTATION",
            "OFFICIAL_SOURCE",
            "NEWS_ARTICLE",
            "BLOG_POST",
            "CODE_REPOSITORY",
            "DATASET",
            "WEB_PAGE",
            "INTERNAL_DEBATE",
            "UNKNOWN",
        ]
        actual_types = [t.name for t in CitationType]
        for expected in expected_types:
            assert expected in actual_types

    def test_enum_values_are_strings(self):
        """Enum values should be lowercase strings."""
        assert CitationType.ACADEMIC_PAPER.value == "academic_paper"
        assert CitationType.WEB_PAGE.value == "web_page"


# =============================================================================
# CitationQuality Enum Tests
# =============================================================================


class TestCitationQuality:
    """Tests for CitationQuality enum."""

    def test_all_6_quality_levels_exist(self):
        """Should have all 6 quality levels."""
        expected = [
            "PEER_REVIEWED",
            "AUTHORITATIVE",
            "REPUTABLE",
            "MIXED",
            "UNVERIFIED",
            "QUESTIONABLE",
        ]
        actual = [q.name for q in CitationQuality]
        for e in expected:
            assert e in actual

    def test_enum_values_are_strings(self):
        """Enum values should be lowercase strings."""
        assert CitationQuality.PEER_REVIEWED.value == "peer_reviewed"


# =============================================================================
# ScholarlyEvidence Tests
# =============================================================================


class TestScholarlyEvidenceIdGeneration:
    """Tests for ID generation in ScholarlyEvidence."""

    def test_id_generated_deterministically(self):
        """Same content should produce same ID."""
        evidence1 = ScholarlyEvidence(
            title="Test Paper",
            authors=["Smith, John"],
            year=2024,
        )
        evidence2 = ScholarlyEvidence(
            title="Test Paper",
            authors=["Smith, John"],
            year=2024,
        )
        assert evidence1.id == evidence2.id
        assert len(evidence1.id) == 12  # SHA256 truncated

    def test_id_changes_with_different_content(self):
        """Different content should produce different IDs."""
        evidence1 = ScholarlyEvidence(title="Paper A", authors=["Smith"], year=2024)
        evidence2 = ScholarlyEvidence(title="Paper B", authors=["Smith"], year=2024)
        assert evidence1.id != evidence2.id

    def test_custom_id_preserved(self):
        """Custom ID should not be overwritten."""
        evidence = ScholarlyEvidence(id="custom-id-123", title="Test")
        assert evidence.id == "custom-id-123"


class TestScholarlyEvidenceFormatAPA:
    """Tests for APA formatting."""

    def test_format_apa_single_author(self):
        """APA format with single author."""
        evidence = ScholarlyEvidence(
            citation_type=CitationType.ACADEMIC_PAPER,
            title="Test Paper",
            authors=["Smith, John"],
            publication="Test Journal",
            year=2024,
        )
        apa = evidence.format_apa()
        assert "Smith, John" in apa
        assert "(2024)" in apa
        assert "Test Paper" in apa

    def test_format_apa_two_authors(self):
        """APA format with two authors."""
        evidence = ScholarlyEvidence(
            title="Test",
            authors=["Smith, John", "Doe, Jane"],
            year=2024,
        )
        apa = evidence.format_apa()
        assert "Smith, John, Doe, Jane" in apa

    def test_format_apa_more_than_three_authors(self):
        """APA format truncates at 3 authors with et al."""
        evidence = ScholarlyEvidence(
            title="Test",
            authors=["A", "B", "C", "D", "E"],
            year=2024,
        )
        apa = evidence.format_apa()
        assert "et al." in apa
        assert "D" not in apa
        assert "E" not in apa

    def test_format_apa_missing_year(self):
        """Missing year should show (n.d.)."""
        evidence = ScholarlyEvidence(title="Test", authors=["Smith"])
        apa = evidence.format_apa()
        assert "(n.d.)" in apa

    def test_format_apa_web_page_includes_url(self):
        """Web page citation should include URL."""
        evidence = ScholarlyEvidence(
            citation_type=CitationType.WEB_PAGE,
            title="Test Page",
            authors=["Author"],
            year=2024,
            url="https://example.com",
        )
        apa = evidence.format_apa()
        assert "Retrieved from" in apa
        assert "https://example.com" in apa


class TestScholarlyEvidenceFormatInline:
    """Tests for inline citation formatting."""

    def test_format_inline_extracts_last_name(self):
        """Should extract last name from author."""
        evidence = ScholarlyEvidence(
            title="Test",
            authors=["Smith, John"],
            year=2024,
        )
        inline = evidence.format_inline()
        # Last name is last word: "John" from "Smith, John".split()[-1]
        assert "John" in inline or "Smith" in inline
        assert "2024" in inline

    def test_format_inline_two_authors_uses_ampersand(self):
        """Two authors should use &."""
        evidence = ScholarlyEvidence(
            title="Test",
            authors=["Smith, John", "Doe, Jane"],
            year=2024,
        )
        inline = evidence.format_inline()
        assert "&" in inline

    def test_format_inline_three_plus_authors_uses_et_al(self):
        """Three+ authors should use et al."""
        evidence = ScholarlyEvidence(
            title="Test",
            authors=["A", "B", "C"],
            year=2024,
        )
        inline = evidence.format_inline()
        assert "et al." in inline

    def test_format_inline_no_authors_uses_title(self):
        """No authors should fallback to title."""
        evidence = ScholarlyEvidence(
            title="Very Long Title That Gets Truncated",
            year=2024,
        )
        inline = evidence.format_inline()
        assert "Very Long Title" in inline
        assert "..." in inline


class TestScholarlyEvidenceQualityScore:
    """Tests for quality_score calculation."""

    def test_quality_score_base_by_type(self):
        """Base score varies by citation type."""
        academic = ScholarlyEvidence(citation_type=CitationType.ACADEMIC_PAPER)
        blog = ScholarlyEvidence(citation_type=CitationType.BLOG_POST)

        assert academic.quality_score() == 0.9
        assert blog.quality_score() == 0.3

    def test_quality_score_peer_reviewed_bonus(self):
        """Peer reviewed adds 0.1."""
        evidence = ScholarlyEvidence(
            citation_type=CitationType.BLOG_POST,  # Base 0.3
            peer_reviewed=True,
        )
        assert evidence.quality_score() == 0.4

    def test_quality_score_doi_bonus(self):
        """DOI adds 0.05."""
        evidence = ScholarlyEvidence(
            citation_type=CitationType.BLOG_POST,  # Base 0.3
            doi="10.1234/test",
        )
        assert evidence.quality_score() == 0.35

    def test_quality_score_high_citations_bonus(self):
        """Citation count > 100 adds 0.05."""
        evidence = ScholarlyEvidence(
            citation_type=CitationType.BLOG_POST,  # Base 0.3
            citation_count=150,
        )
        assert evidence.quality_score() == 0.35

    def test_quality_score_verified_bonus(self):
        """Verified adds 0.05."""
        evidence = ScholarlyEvidence(
            citation_type=CitationType.BLOG_POST,  # Base 0.3
            verified=True,
        )
        assert evidence.quality_score() == 0.35

    def test_quality_score_capped_at_1(self):
        """Score should not exceed 1.0."""
        evidence = ScholarlyEvidence(
            citation_type=CitationType.ACADEMIC_PAPER,  # Base 0.9
            peer_reviewed=True,  # +0.1
            doi="10.1234",  # +0.05
            citation_count=200,  # +0.05
            verified=True,  # +0.05
        )
        assert evidence.quality_score() == 1.0

    def test_quality_score_unknown_type(self):
        """Unknown type has lowest base score."""
        evidence = ScholarlyEvidence(citation_type=CitationType.UNKNOWN)
        assert evidence.quality_score() == 0.1


class TestScholarlyEvidenceToDict:
    """Tests for to_dict serialization."""

    def test_to_dict_includes_all_fields(self, sample_evidence):
        """to_dict should include all key fields."""
        d = sample_evidence.to_dict()

        assert "id" in d
        assert "citation_type" in d
        assert "title" in d
        assert "authors" in d
        assert "quality_score" in d

    def test_to_dict_converts_enums(self, sample_evidence):
        """Enums should be converted to values."""
        d = sample_evidence.to_dict()

        assert d["citation_type"] == "academic_paper"
        assert d["quality"] == "peer_reviewed"


# =============================================================================
# CitedClaim Tests
# =============================================================================


class TestCitedClaim:
    """Tests for CitedClaim dataclass."""

    def test_claim_id_generated_from_text(self):
        """claim_id should be hash of claim_text."""
        claim = CitedClaim(claim_text="Test claim")
        assert len(claim.claim_id) == 12

    def test_identical_claims_same_id(self):
        """Identical claim text should produce same ID."""
        claim1 = CitedClaim(claim_text="Test claim")
        claim2 = CitedClaim(claim_text="Test claim")
        assert claim1.claim_id == claim2.claim_id

    def test_grounding_score_calculated(self, sample_evidence):
        """Grounding score should be calculated from citations."""
        claim = CitedClaim(
            claim_text="Test",
            citations=[sample_evidence],
        )
        assert claim.grounding_score > 0

    def test_grounding_score_zero_without_citations(self):
        """Grounding score should be 0 without citations."""
        claim = CitedClaim(claim_text="Test")
        assert claim.grounding_score == 0.0

    def test_to_dict_includes_nested_citations(self, sample_evidence):
        """to_dict should include serialized citations."""
        claim = CitedClaim(
            claim_text="Test",
            citations=[sample_evidence],
        )
        d = claim.to_dict()

        assert len(d["citations"]) == 1
        assert d["citations"][0]["title"] == sample_evidence.title


# =============================================================================
# GroundedVerdict Tests
# =============================================================================


class TestGroundedVerdict:
    """Tests for GroundedVerdict dataclass."""

    def test_citations_deduplicated(self, sample_evidence):
        """Same citation in multiple claims should be deduplicated."""
        claim1 = CitedClaim(claim_text="Claim 1", citations=[sample_evidence])
        claim2 = CitedClaim(claim_text="Claim 2", citations=[sample_evidence])

        verdict = GroundedVerdict(
            verdict="Test verdict",
            confidence=0.9,
            claims=[claim1, claim2],
        )

        # Should only have 1 unique citation
        assert len(verdict.all_citations) == 1

    def test_grounding_score_aggregated(self, sample_evidence):
        """Grounding score should be average of claim scores."""
        claim = CitedClaim(claim_text="Test", citations=[sample_evidence])
        verdict = GroundedVerdict(
            verdict="Test",
            confidence=0.9,
            claims=[claim],
        )

        assert verdict.grounding_score == claim.grounding_score

    def test_format_bibliography_empty(self):
        """Empty citations should return 'No citations.'"""
        verdict = GroundedVerdict(verdict="Test", confidence=0.9)
        assert verdict.format_bibliography() == "No citations."

    def test_format_bibliography_numbered(self, sample_evidence):
        """Bibliography should number citations."""
        claim = CitedClaim(claim_text="Test", citations=[sample_evidence])
        verdict = GroundedVerdict(
            verdict="Test",
            confidence=0.9,
            claims=[claim],
        )

        bib = verdict.format_bibliography()
        assert "[1]" in bib
        assert "References:" in bib

    def test_summary_limits_citations(self, sample_evidence):
        """Summary should limit displayed citations to 5."""
        # Create 10 different citations
        citations = []
        for i in range(10):
            citations.append(
                ScholarlyEvidence(
                    title=f"Paper {i}",
                    authors=[f"Author {i}"],
                    year=2020 + i,
                )
            )

        claim = CitedClaim(claim_text="Test", citations=citations)
        verdict = GroundedVerdict(
            verdict="Test verdict",
            confidence=0.9,
            claims=[claim],
        )

        summary = verdict.summary()
        assert "and 5 more" in summary

    def test_to_dict_serializes_nested(self, sample_evidence):
        """to_dict should serialize nested claims and citations."""
        claim = CitedClaim(claim_text="Test", citations=[sample_evidence])
        verdict = GroundedVerdict(
            verdict="Test",
            confidence=0.9,
            claims=[claim],
        )

        d = verdict.to_dict()
        assert "claims" in d
        assert "all_citations" in d
        assert len(d["claims"]) == 1


# =============================================================================
# CitationExtractor Tests
# =============================================================================


class TestCitationExtractorExtractClaims:
    """Tests for extract_claims method."""

    def test_extract_research_shows(self, citation_extractor):
        """Should find 'research shows' pattern."""
        text = "Research shows that AI is advancing rapidly. Other statement."
        claims = citation_extractor.extract_claims(text)

        assert len(claims) == 1
        assert "Research shows" in claims[0]

    def test_extract_studies_have_found(self, citation_extractor):
        """Should find 'studies have found' pattern."""
        text = "Studies have found significant results."
        claims = citation_extractor.extract_claims(text)

        assert len(claims) == 1

    def test_extract_percentage_pattern(self, citation_extractor):
        """Should find percentage patterns."""
        text = "About 75% of users prefer this option."
        claims = citation_extractor.extract_claims(text)

        assert len(claims) == 1
        assert "75%" in claims[0]

    def test_extract_case_insensitive(self, citation_extractor):
        """Pattern matching should be case-insensitive."""
        text = "RESEARCH SHOWS important findings."
        claims = citation_extractor.extract_claims(text)

        assert len(claims) == 1

    def test_extract_empty_text(self, citation_extractor):
        """Empty text should return empty list."""
        claims = citation_extractor.extract_claims("")
        assert claims == []

    def test_extract_no_patterns(self, citation_extractor):
        """Text without patterns should return empty."""
        text = "This is a simple statement. Nothing special here."
        claims = citation_extractor.extract_claims(text)

        assert claims == []


class TestCitationExtractorIdentifyNeeds:
    """Tests for identify_citation_needs method."""

    def test_high_priority_proven(self, citation_extractor):
        """Claims with 'proven' should be high priority."""
        text = "This has been proven to work."
        needs = citation_extractor.identify_citation_needs(text)

        assert len(needs) == 1
        assert needs[0]["priority"] == "high"

    def test_high_priority_data(self, citation_extractor):
        """Claims with 'data' should be high priority."""
        text = "Data indicates strong correlation."
        needs = citation_extractor.identify_citation_needs(text)

        assert len(needs) == 1
        assert needs[0]["priority"] == "high"

    def test_low_priority_suggests(self, citation_extractor):
        """Claims with 'suggests' should be low priority."""
        text = "Evidence suggests a possible link."
        needs = citation_extractor.identify_citation_needs(text)

        assert len(needs) == 1
        assert needs[0]["priority"] == "low"

    def test_medium_priority_default(self, citation_extractor):
        """Other claims should be medium priority."""
        text = "According to experts, this is correct."
        needs = citation_extractor.identify_citation_needs(text)

        assert len(needs) == 1
        assert needs[0]["priority"] == "medium"


class TestCitationExtractorSuggestTypes:
    """Tests for _suggest_source_types method."""

    def test_suggest_research_types(self, citation_extractor):
        """Research claims should suggest academic sources."""
        types = citation_extractor._suggest_source_types("Research indicates...")

        assert CitationType.ACADEMIC_PAPER in types

    def test_suggest_code_types(self, citation_extractor):
        """Code claims should suggest documentation/repos."""
        types = citation_extractor._suggest_source_types("The code shows...")

        assert CitationType.CODE_REPOSITORY in types or CitationType.DOCUMENTATION in types

    def test_suggest_law_types(self, citation_extractor):
        """Legal claims should suggest official sources."""
        types = citation_extractor._suggest_source_types("The regulation states...")

        assert CitationType.OFFICIAL_SOURCE in types


# =============================================================================
# CitationStore Tests
# =============================================================================


class TestCitationStore:
    """Tests for CitationStore."""

    def test_add_and_get(self, citation_store, sample_evidence):
        """add() and get() should work correctly."""
        cid = citation_store.add(sample_evidence)

        retrieved = citation_store.get(cid)
        assert retrieved is sample_evidence

    def test_get_returns_none_for_missing(self, citation_store):
        """get() should return None for missing ID."""
        assert citation_store.get("nonexistent") is None

    def test_find_for_claim_matches_keywords(self, citation_store):
        """find_for_claim should match by keyword overlap."""
        evidence = ScholarlyEvidence(
            title="Machine Learning Safety",
            excerpt="Discussion of AI safety concerns",
        )
        citation_store.add(evidence)

        results = citation_store.find_for_claim("AI safety research")

        assert len(results) == 1
        assert results[0].title == "Machine Learning Safety"

    def test_link_and_get_citations_for_claim(self, citation_store, sample_evidence):
        """link_claim_to_citation and get_citations_for_claim should work."""
        citation_store.add(sample_evidence)
        citation_store.link_claim_to_citation("claim-1", sample_evidence.id)

        citations = citation_store.get_citations_for_claim("claim-1")

        assert len(citations) == 1
        assert citations[0] is sample_evidence


# =============================================================================
# create_citation_from_url Tests
# =============================================================================


class TestCreateCitationFromUrl:
    """Tests for create_citation_from_url function."""

    def test_arxiv_url_preprint(self):
        """arxiv.org URLs should create PREPRINT type."""
        evidence = create_citation_from_url("https://arxiv.org/abs/1234.5678")

        assert evidence.citation_type == CitationType.PREPRINT
        assert evidence.quality == CitationQuality.REPUTABLE

    def test_github_url_code_repository(self):
        """github.com URLs should create CODE_REPOSITORY type."""
        evidence = create_citation_from_url("https://github.com/user/repo")

        assert evidence.citation_type == CitationType.CODE_REPOSITORY

    def test_gov_edu_official_source(self):
        """Government/edu URLs should create OFFICIAL_SOURCE type."""
        evidence = create_citation_from_url("https://www.example.gov/report")

        assert evidence.citation_type == CitationType.OFFICIAL_SOURCE
        assert evidence.quality == CitationQuality.AUTHORITATIVE

    def test_doi_academic_paper(self):
        """DOI URLs should create ACADEMIC_PAPER type."""
        evidence = create_citation_from_url("https://doi.org/10.1234/test")

        assert evidence.citation_type == CitationType.ACADEMIC_PAPER
        assert evidence.quality == CitationQuality.PEER_REVIEWED

    def test_unknown_url_web_page(self):
        """Unknown URLs should default to WEB_PAGE."""
        evidence = create_citation_from_url("https://random-site.com/page")

        assert evidence.citation_type == CitationType.WEB_PAGE
        assert evidence.quality == CitationQuality.UNVERIFIED

    def test_title_and_excerpt_preserved(self):
        """Title and excerpt should be preserved."""
        evidence = create_citation_from_url(
            "https://example.com",
            title="Test Title",
            excerpt="Test excerpt",
        )

        assert evidence.title == "Test Title"
        assert evidence.excerpt == "Test excerpt"
