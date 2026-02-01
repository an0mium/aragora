"""
Tests for Audit-Evidence Adapter.

Tests the FindingEvidenceCollector which connects the document audit system
to the evidence collection system, enabling automatic evidence gathering
for audit findings.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import pytest

from aragora.audit.evidence_adapter import (
    EvidenceConfig,
    EvidenceEnrichment,
    EvidenceSource,
    FindingEvidenceCollector,
    enrich_finding_with_evidence,
)


# ===========================================================================
# Test Data
# ===========================================================================


@dataclass
class MockFinding:
    """Mock audit finding for testing."""

    id: str = "finding_001"
    title: str = "Missing Error Handling"
    description: str = "Function lacks proper error handling for edge cases"
    category: str = "code_quality"
    confidence: float = 0.75
    document_id: str = "doc_001"
    evidence_text: str = ""
    evidence_location: str = ""


SAMPLE_DOCUMENT = """
This is a sample document for testing evidence extraction.

The function process_data() does not handle None values properly.
When a None value is passed, the function crashes with an AttributeError.

Error handling should be improved across the codebase.
Missing error handling in several critical paths has been identified.

The following functions need attention:
- process_data() - No null check
- validate_input() - No type check
- save_results() - No IO error handling
"""

RELATED_DOCUMENT = """
Code review findings from last sprint:

The error handling patterns need improvement.
Several functions were identified as missing proper error handling.
This includes process_data and validate_input functions.

Recommended: Add try-except blocks and proper error propagation.
"""


# ===========================================================================
# Tests: EvidenceSource
# ===========================================================================


class TestEvidenceSource:
    """Tests for EvidenceSource dataclass."""

    def test_create_evidence_source(self):
        """Test creating an evidence source."""
        source = EvidenceSource(
            source_id="src_001",
            source_type="document",
            title="Test Document",
            snippet="This is relevant text",
            location="Page 5, Line 10",
            relevance_score=0.85,
            reliability_score=0.90,
        )

        assert source.source_id == "src_001"
        assert source.source_type == "document"
        assert source.relevance_score == 0.85
        assert source.url is None

    def test_evidence_source_with_url(self):
        """Test evidence source with URL."""
        source = EvidenceSource(
            source_id="src_002",
            source_type="external",
            title="External Reference",
            snippet="Reference text",
            location="https://example.com/doc",
            relevance_score=0.70,
            reliability_score=0.80,
            url="https://example.com/doc",
        )

        assert source.url == "https://example.com/doc"

    def test_evidence_source_to_dict(self):
        """Test serialization to dictionary."""
        source = EvidenceSource(
            source_id="src_001",
            source_type="document",
            title="Test",
            snippet="text",
            location="line 1",
            relevance_score=0.9,
            reliability_score=0.85,
            metadata={"key": "value"},
        )

        data = source.to_dict()
        assert data["source_id"] == "src_001"
        assert data["source_type"] == "document"
        assert data["relevance_score"] == 0.9
        assert data["metadata"] == {"key": "value"}

    def test_evidence_source_default_metadata(self):
        """Test default metadata is empty dict."""
        source = EvidenceSource(
            source_id="src_001",
            source_type="document",
            title="Test",
            snippet="text",
            location="line 1",
            relevance_score=0.9,
            reliability_score=0.85,
        )

        assert source.metadata == {}


# ===========================================================================
# Tests: EvidenceEnrichment
# ===========================================================================


class TestEvidenceEnrichment:
    """Tests for EvidenceEnrichment dataclass."""

    def test_create_enrichment(self):
        """Test creating an enrichment."""
        enrichment = EvidenceEnrichment(
            finding_id="f_001",
            sources=[],
            original_confidence=0.75,
            adjusted_confidence=0.75,
            evidence_summary="No evidence",
        )

        assert enrichment.finding_id == "f_001"
        assert enrichment.original_confidence == 0.75
        assert enrichment.collection_time_ms == 0

    def test_has_strong_evidence_true(self):
        """Test has_strong_evidence with multiple relevant sources."""
        sources = [
            EvidenceSource(
                source_id="s1",
                source_type="document",
                title="S1",
                snippet="text1",
                location="loc1",
                relevance_score=0.85,
                reliability_score=0.90,
            ),
            EvidenceSource(
                source_id="s2",
                source_type="document",
                title="S2",
                snippet="text2",
                location="loc2",
                relevance_score=0.80,
                reliability_score=0.85,
            ),
        ]

        enrichment = EvidenceEnrichment(
            finding_id="f_001",
            sources=sources,
            original_confidence=0.7,
            adjusted_confidence=0.85,
            evidence_summary="Strong evidence",
        )

        assert enrichment.has_strong_evidence is True

    def test_has_strong_evidence_false_low_relevance(self):
        """Test has_strong_evidence with low relevance sources."""
        sources = [
            EvidenceSource(
                source_id="s1",
                source_type="document",
                title="S1",
                snippet="text1",
                location="loc1",
                relevance_score=0.3,
                reliability_score=0.90,
            ),
            EvidenceSource(
                source_id="s2",
                source_type="document",
                title="S2",
                snippet="text2",
                location="loc2",
                relevance_score=0.4,
                reliability_score=0.85,
            ),
        ]

        enrichment = EvidenceEnrichment(
            finding_id="f_001",
            sources=sources,
            original_confidence=0.7,
            adjusted_confidence=0.65,
            evidence_summary="Weak evidence",
        )

        assert enrichment.has_strong_evidence is False

    def test_has_strong_evidence_false_single_source(self):
        """Test has_strong_evidence with only one source."""
        sources = [
            EvidenceSource(
                source_id="s1",
                source_type="document",
                title="S1",
                snippet="text1",
                location="loc1",
                relevance_score=0.95,
                reliability_score=0.90,
            ),
        ]

        enrichment = EvidenceEnrichment(
            finding_id="f_001",
            sources=sources,
            original_confidence=0.7,
            adjusted_confidence=0.85,
            evidence_summary="Single source",
        )

        assert enrichment.has_strong_evidence is False

    def test_has_strong_evidence_false_no_sources(self):
        """Test has_strong_evidence with no sources."""
        enrichment = EvidenceEnrichment(
            finding_id="f_001",
            sources=[],
            original_confidence=0.7,
            adjusted_confidence=0.7,
            evidence_summary="No evidence",
        )

        assert enrichment.has_strong_evidence is False

    def test_confidence_boost(self):
        """Test confidence boost calculation."""
        enrichment = EvidenceEnrichment(
            finding_id="f_001",
            sources=[],
            original_confidence=0.70,
            adjusted_confidence=0.85,
            evidence_summary="Boosted",
        )

        assert enrichment.confidence_boost == pytest.approx(0.15, abs=0.001)

    def test_confidence_boost_negative(self):
        """Test negative confidence boost (penalty)."""
        enrichment = EvidenceEnrichment(
            finding_id="f_001",
            sources=[],
            original_confidence=0.70,
            adjusted_confidence=0.60,
            evidence_summary="Penalized",
        )

        assert enrichment.confidence_boost == pytest.approx(-0.10, abs=0.001)

    def test_to_dict(self):
        """Test serialization to dictionary."""
        enrichment = EvidenceEnrichment(
            finding_id="f_001",
            sources=[
                EvidenceSource(
                    source_id="s1",
                    source_type="document",
                    title="S1",
                    snippet="text",
                    location="loc",
                    relevance_score=0.9,
                    reliability_score=0.8,
                ),
            ],
            original_confidence=0.7,
            adjusted_confidence=0.85,
            evidence_summary="Good evidence",
        )

        data = enrichment.to_dict()
        assert data["finding_id"] == "f_001"
        assert len(data["sources"]) == 1
        assert data["original_confidence"] == 0.7
        assert data["adjusted_confidence"] == 0.85
        assert "has_strong_evidence" in data
        assert "collected_at" in data


# ===========================================================================
# Tests: EvidenceConfig
# ===========================================================================


class TestEvidenceConfig:
    """Tests for EvidenceConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = EvidenceConfig()

        assert config.max_sources_per_finding == 5
        assert config.min_relevance_threshold == 0.3
        assert config.enable_external_sources is True
        assert config.enable_cross_reference is True
        assert config.evidence_weight == 0.3
        assert config.strong_evidence_boost == 0.15
        assert config.weak_evidence_penalty == 0.1
        assert config.search_window == 500
        assert config.max_parallel_searches == 5

    def test_custom_config(self):
        """Test custom configuration values."""
        config = EvidenceConfig(
            max_sources_per_finding=3,
            min_relevance_threshold=0.5,
            enable_external_sources=False,
            enable_cross_reference=False,
            evidence_weight=0.5,
        )

        assert config.max_sources_per_finding == 3
        assert config.min_relevance_threshold == 0.5
        assert config.enable_external_sources is False
        assert config.enable_cross_reference is False
        assert config.evidence_weight == 0.5


# ===========================================================================
# Tests: FindingEvidenceCollector
# ===========================================================================


class TestFindingEvidenceCollector:
    """Tests for FindingEvidenceCollector class."""

    @pytest.fixture
    def collector(self):
        """Create a collector with external sources disabled."""
        config = EvidenceConfig(
            enable_external_sources=False,
            enable_cross_reference=True,
        )
        return FindingEvidenceCollector(config=config)

    @pytest.fixture
    def collector_no_cross_ref(self):
        """Create a collector with both external and cross-ref disabled."""
        config = EvidenceConfig(
            enable_external_sources=False,
            enable_cross_reference=False,
        )
        return FindingEvidenceCollector(config=config)

    @pytest.fixture
    def finding_with_evidence(self):
        """Create a finding that has exact match evidence in SAMPLE_DOCUMENT."""
        return MockFinding(
            id="finding_001",
            title="Missing Error Handling",
            description="Function lacks proper error handling",
            evidence_text="does not handle None values properly",
            evidence_location="Line 4",
        )

    @pytest.fixture
    def finding_without_evidence(self):
        """Create a finding with no matching evidence text."""
        return MockFinding(
            id="finding_002",
            title="Memory Leak",
            description="Potential memory leak in allocator",
            evidence_text="xyz_does_not_exist_in_document",
        )

    @pytest.mark.asyncio
    async def test_enrich_finding_with_document_evidence(self, collector, finding_with_evidence):
        """Test enriching a finding with document evidence."""
        enrichment = await collector.enrich_finding(
            finding=finding_with_evidence,
            document_content=SAMPLE_DOCUMENT,
        )

        assert isinstance(enrichment, EvidenceEnrichment)
        assert enrichment.finding_id == "finding_001"
        assert len(enrichment.sources) > 0
        assert enrichment.collection_time_ms >= 0

    @pytest.mark.asyncio
    async def test_enrich_finding_exact_match(self, collector, finding_with_evidence):
        """Test that exact match evidence gets high relevance."""
        enrichment = await collector.enrich_finding(
            finding=finding_with_evidence,
            document_content=SAMPLE_DOCUMENT,
        )

        doc_sources = [s for s in enrichment.sources if s.source_type == "document"]
        assert len(doc_sources) > 0

        # The exact match should have high relevance
        exact_matches = [s for s in doc_sources if s.metadata.get("exact_match") is True]
        assert len(exact_matches) >= 1
        assert exact_matches[0].relevance_score >= 0.9

    @pytest.mark.asyncio
    async def test_enrich_finding_no_document(self, collector_no_cross_ref):
        """Test enriching without document content."""
        finding = MockFinding()

        enrichment = await collector_no_cross_ref.enrich_finding(
            finding=finding,
            document_content=None,
        )

        assert isinstance(enrichment, EvidenceEnrichment)
        # Without document and with external sources disabled, should have few/no sources
        assert enrichment.original_confidence == 0.75

    @pytest.mark.asyncio
    async def test_enrich_finding_no_matching_evidence(
        self, collector_no_cross_ref, finding_without_evidence
    ):
        """Test enriching when evidence text doesn't match document."""
        enrichment = await collector_no_cross_ref.enrich_finding(
            finding=finding_without_evidence,
            document_content=SAMPLE_DOCUMENT,
        )

        # Should still find keyword-based evidence
        assert isinstance(enrichment, EvidenceEnrichment)

    @pytest.mark.asyncio
    async def test_enrich_finding_with_cross_references(self, collector):
        """Test enriching with cross-reference documents."""
        finding = MockFinding(
            title="Error Handling Issues",
            description="Missing error handling in process_data function",
        )

        related = {"doc_002": RELATED_DOCUMENT}

        enrichment = await collector.enrich_finding(
            finding=finding,
            document_content=SAMPLE_DOCUMENT,
            related_documents=related,
        )

        # Should find cross-references
        xref_sources = [s for s in enrichment.sources if s.source_type == "cross_reference"]
        assert len(xref_sources) >= 1

    @pytest.mark.asyncio
    async def test_enrich_finding_confidence_adjustment(self, collector):
        """Test confidence is adjusted based on evidence."""
        finding = MockFinding(
            title="Missing Error Handling",
            description="Function lacks proper error handling",
            confidence=0.60,
            evidence_text="does not handle None values properly",
        )

        enrichment = await collector.enrich_finding(
            finding=finding,
            document_content=SAMPLE_DOCUMENT,
        )

        # With evidence found, confidence should be adjusted
        assert enrichment.original_confidence == 0.60
        assert enrichment.adjusted_confidence != enrichment.original_confidence

    @pytest.mark.asyncio
    async def test_sources_limited_by_config(self, collector):
        """Test that sources are limited to max_sources_per_finding."""
        config = EvidenceConfig(
            max_sources_per_finding=2,
            enable_external_sources=False,
        )
        limited_collector = FindingEvidenceCollector(config=config)

        finding = MockFinding(
            title="Error Handling",
            description="Missing error handling across codebase",
            evidence_text="does not handle None values properly",
        )

        enrichment = await limited_collector.enrich_finding(
            finding=finding,
            document_content=SAMPLE_DOCUMENT,
        )

        assert len(enrichment.sources) <= 2

    @pytest.mark.asyncio
    async def test_enrich_findings_batch(self, collector):
        """Test batch enrichment of multiple findings."""
        findings = [
            MockFinding(
                id="f1",
                title="Missing Error Handling",
                description="No error handling",
                document_id="doc_001",
            ),
            MockFinding(
                id="f2",
                title="Code Quality Issue",
                description="Poor code quality",
                document_id="doc_001",
            ),
        ]

        documents = {"doc_001": SAMPLE_DOCUMENT}

        results = await collector.enrich_findings_batch(
            findings=findings,
            documents=documents,
            max_concurrent=2,
        )

        assert len(results) == 2
        assert all(isinstance(v, EvidenceEnrichment) for v in results.values())


# ===========================================================================
# Tests: Internal Methods
# ===========================================================================


class TestCollectorInternalMethods:
    """Tests for internal helper methods."""

    @pytest.fixture
    def collector(self):
        return FindingEvidenceCollector(
            config=EvidenceConfig(
                enable_external_sources=False,
                enable_cross_reference=False,
            )
        )

    def test_extract_keywords(self, collector):
        """Test keyword extraction."""
        keywords = collector._extract_keywords(
            "Missing Error Handling in the process_data function"
        )

        assert len(keywords) > 0
        assert "missing" in keywords
        assert "error" in keywords
        assert "handling" in keywords
        # Stop words should be filtered out
        assert "the" not in keywords
        assert "in" not in keywords

    def test_extract_keywords_removes_short_words(self, collector):
        """Test that short words (<=3 chars) are removed."""
        keywords = collector._extract_keywords("a an or the big bad wolf")

        assert "big" not in keywords  # 3 chars, not > 3
        assert "wolf" in keywords

    def test_extract_keywords_limit(self, collector):
        """Test keyword extraction limited to 10."""
        long_text = " ".join(f"keyword{i}" for i in range(20))
        keywords = collector._extract_keywords(long_text)

        assert len(keywords) <= 10

    def test_find_keyword_occurrences(self, collector):
        """Test finding keyword occurrences in content."""
        occurrences = collector._find_keyword_occurrences(
            SAMPLE_DOCUMENT,
            "error",
        )

        assert len(occurrences) > 0
        assert all("pos" in occ for occ in occurrences)
        assert all("text" in occ for occ in occurrences)

    def test_find_keyword_occurrences_limit(self, collector):
        """Test occurrence limit."""
        occurrences = collector._find_keyword_occurrences(
            SAMPLE_DOCUMENT,
            "the",
            max_occurrences=2,
        )

        assert len(occurrences) <= 2

    def test_find_keyword_occurrences_not_found(self, collector):
        """Test when keyword is not found."""
        occurrences = collector._find_keyword_occurrences(
            SAMPLE_DOCUMENT,
            "xyznonexistent",
        )

        assert len(occurrences) == 0

    def test_rank_sources(self, collector):
        """Test source ranking."""
        sources = [
            EvidenceSource(
                source_id="s1",
                source_type="external",
                title="External",
                snippet="text",
                location="loc",
                relevance_score=0.9,
                reliability_score=0.9,
            ),
            EvidenceSource(
                source_id="s2",
                source_type="document",
                title="Document",
                snippet="text",
                location="loc",
                relevance_score=0.9,
                reliability_score=0.9,
            ),
        ]

        finding = MockFinding()
        ranked = collector._rank_sources(sources, finding)

        # Document source should rank higher than external with same scores
        assert ranked[0].source_type == "document"
        assert ranked[1].source_type == "external"

    def test_calculate_adjusted_confidence_no_sources(self, collector):
        """Test confidence with no sources returns original."""
        result = collector._calculate_adjusted_confidence(0.75, [])
        assert result == 0.75

    def test_calculate_adjusted_confidence_strong_evidence(self, collector):
        """Test confidence boosted with strong evidence."""
        sources = [
            EvidenceSource(
                source_id="s1",
                source_type="document",
                title="S1",
                snippet="text",
                location="loc",
                relevance_score=0.9,
                reliability_score=0.9,
            ),
            EvidenceSource(
                source_id="s2",
                source_type="document",
                title="S2",
                snippet="text",
                location="loc",
                relevance_score=0.85,
                reliability_score=0.85,
            ),
        ]

        result = collector._calculate_adjusted_confidence(0.70, sources)
        assert result > 0.70  # Should be boosted

    def test_calculate_adjusted_confidence_clamped(self, collector):
        """Test confidence is clamped to [0, 1]."""
        sources = [
            EvidenceSource(
                source_id="s1",
                source_type="document",
                title="S1",
                snippet="text",
                location="loc",
                relevance_score=0.99,
                reliability_score=0.99,
            ),
            EvidenceSource(
                source_id="s2",
                source_type="document",
                title="S2",
                snippet="text",
                location="loc",
                relevance_score=0.99,
                reliability_score=0.99,
            ),
        ]

        result = collector._calculate_adjusted_confidence(0.98, sources)
        assert result <= 1.0

    def test_generate_evidence_summary_no_sources(self, collector):
        """Test summary with no sources."""
        finding = MockFinding()
        summary = collector._generate_evidence_summary(finding, [])
        assert summary == "No supporting evidence collected."

    def test_generate_evidence_summary_mixed_sources(self, collector):
        """Test summary with mixed source types."""
        sources = [
            EvidenceSource(
                source_id="s1",
                source_type="document",
                title="Doc",
                snippet="Some relevant text from the document",
                location="loc",
                relevance_score=0.9,
                reliability_score=0.9,
            ),
            EvidenceSource(
                source_id="s2",
                source_type="cross_reference",
                title="XRef",
                snippet="text",
                location="loc",
                relevance_score=0.7,
                reliability_score=0.8,
            ),
        ]

        finding = MockFinding()
        summary = collector._generate_evidence_summary(finding, sources)

        assert "1 source(s) from the document" in summary
        assert "1 cross-reference(s)" in summary
        assert "Top evidence:" in summary


# ===========================================================================
# Tests: Convenience Function
# ===========================================================================


class TestConvenienceFunction:
    """Tests for enrich_finding_with_evidence function."""

    @pytest.mark.asyncio
    async def test_enrich_finding_with_evidence(self):
        """Test the convenience function."""
        finding = MockFinding(
            evidence_text="does not handle None values properly",
        )

        config = EvidenceConfig(
            enable_external_sources=False,
            enable_cross_reference=False,
        )

        enrichment = await enrich_finding_with_evidence(
            finding=finding,
            document_content=SAMPLE_DOCUMENT,
            config=config,
        )

        assert isinstance(enrichment, EvidenceEnrichment)
        assert enrichment.finding_id == "finding_001"
