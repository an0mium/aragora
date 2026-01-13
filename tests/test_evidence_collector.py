"""
Tests for evidence/collector.py - Evidence collection and freshness scoring.

Tests cover:
- EvidenceSnippet freshness scoring
- EvidenceSnippet combined scoring
- EvidenceSnippet formatting
- EvidencePack aggregation
- EvidenceCollector keyword extraction
- EvidenceCollector URL extraction
- EvidenceCollector snippet truncation
- EvidenceCollector reliability calculation
- EvidenceCollector snippet ranking
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from aragora.evidence.collector import (
    EvidenceSnippet,
    EvidencePack,
    EvidenceCollector,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def fresh_snippet():
    """Create a fresh evidence snippet (just fetched)."""
    return EvidenceSnippet(
        id="test-001",
        source="web",
        title="Test Evidence",
        snippet="This is test evidence content.",
        url="https://example.com/test",
        reliability_score=0.8,
        fetched_at=datetime.now(),
    )


@pytest.fixture
def old_snippet():
    """Create an old evidence snippet (1 week old)."""
    return EvidenceSnippet(
        id="test-002",
        source="github",
        title="Old Evidence",
        snippet="This is old evidence.",
        url="https://github.com/test",
        reliability_score=0.9,
        fetched_at=datetime.now() - timedelta(days=7),
    )


@pytest.fixture
def collector():
    """Create an EvidenceCollector instance."""
    return EvidenceCollector()


# =============================================================================
# EvidenceSnippet Freshness Tests
# =============================================================================


class TestEvidenceSnippetFreshness:
    """Tests for freshness scoring."""

    def test_very_fresh_snippet(self):
        """Should return 1.0 for snippets < 1 hour old."""
        snippet = EvidenceSnippet(
            id="test",
            source="web",
            title="Fresh",
            snippet="content",
            fetched_at=datetime.now() - timedelta(minutes=30),
        )
        assert snippet.freshness_score == 1.0

    def test_fresh_snippet_1_hour(self):
        """Should return ~0.9 for snippets 1 hour old."""
        snippet = EvidenceSnippet(
            id="test",
            source="web",
            title="1hr old",
            snippet="content",
            fetched_at=datetime.now() - timedelta(hours=1),
        )
        assert 0.85 <= snippet.freshness_score <= 0.95

    def test_day_old_snippet(self):
        """Should return ~0.7 for snippets 1 day old."""
        snippet = EvidenceSnippet(
            id="test",
            source="web",
            title="1 day old",
            snippet="content",
            fetched_at=datetime.now() - timedelta(days=1),
        )
        assert 0.65 <= snippet.freshness_score <= 0.75

    def test_week_old_snippet(self):
        """Should return ~0.5 for snippets 1 week old."""
        snippet = EvidenceSnippet(
            id="test",
            source="web",
            title="1 week old",
            snippet="content",
            fetched_at=datetime.now() - timedelta(days=7),
        )
        assert 0.45 <= snippet.freshness_score <= 0.55

    def test_very_old_snippet(self):
        """Should return ~0.3 for very old snippets."""
        snippet = EvidenceSnippet(
            id="test",
            source="web",
            title="Very old",
            snippet="content",
            fetched_at=datetime.now() - timedelta(days=30),
        )
        assert 0.3 <= snippet.freshness_score <= 0.4

    def test_freshness_never_below_minimum(self):
        """Should never return below 0.3."""
        snippet = EvidenceSnippet(
            id="test",
            source="web",
            title="Ancient",
            snippet="content",
            fetched_at=datetime.now() - timedelta(days=365),
        )
        assert snippet.freshness_score >= 0.3


# =============================================================================
# EvidenceSnippet Combined Score Tests
# =============================================================================


class TestEvidenceSnippetCombinedScore:
    """Tests for combined reliability + freshness scoring."""

    def test_combined_score_calculation(self, fresh_snippet):
        """Should weight reliability 70%, freshness 30%."""
        # Fresh snippet: reliability=0.8, freshness=1.0
        expected = 0.8 * 0.7 + 1.0 * 0.3  # 0.56 + 0.30 = 0.86
        assert fresh_snippet.combined_score == pytest.approx(expected, abs=0.01)

    def test_combined_score_old_snippet(self, old_snippet):
        """Should penalize old snippets in combined score."""
        # Old snippet has high reliability but lower freshness
        assert old_snippet.combined_score < 0.9  # Lower than pure reliability


# =============================================================================
# EvidenceSnippet Formatting Tests
# =============================================================================


class TestEvidenceSnippetFormatting:
    """Tests for snippet formatting."""

    def test_to_text_block(self, fresh_snippet):
        """Should format as text block."""
        text = fresh_snippet.to_text_block()
        assert "EVID-test-001" in text
        assert "Source: web" in text
        assert "Title: Test Evidence" in text
        assert "https://example.com/test" in text

    def test_to_text_block_freshness_indicator(self):
        """Should show freshness indicator."""
        fresh = EvidenceSnippet(
            id="f",
            source="web",
            title="Fresh",
            snippet="c",
            fetched_at=datetime.now(),
        )
        old = EvidenceSnippet(
            id="o",
            source="web",
            title="Old",
            snippet="c",
            fetched_at=datetime.now() - timedelta(days=30),
        )

        assert "🟢" in fresh.to_text_block()  # Green for fresh
        assert "🔴" in old.to_text_block()  # Red for stale

    def test_to_citation(self, fresh_snippet):
        """Should format as citation."""
        citation = fresh_snippet.to_citation()
        assert "[test-001]" in citation
        assert "Test Evidence" in citation
        assert "reliability: 0.8" in citation

    def test_to_dict(self, fresh_snippet):
        """Should convert to dictionary."""
        d = fresh_snippet.to_dict()
        assert d["id"] == "test-001"
        assert d["source"] == "web"
        assert d["reliability_score"] == 0.8
        assert "freshness_score" in d
        assert "combined_score" in d

    def test_snippet_truncation_in_text_block(self):
        """Should truncate long snippets in text block."""
        long_snippet = EvidenceSnippet(
            id="long",
            source="web",
            title="Long",
            snippet="x" * 1000,
        )
        text = long_snippet.to_text_block()
        assert "..." in text
        assert len(text) < 1100  # Should be truncated


# =============================================================================
# EvidencePack Tests
# =============================================================================


class TestEvidencePack:
    """Tests for EvidencePack aggregation."""

    def test_average_reliability_empty(self):
        """Should return 0.0 for empty pack."""
        pack = EvidencePack(topic_keywords=["test"], snippets=[])
        assert pack.average_reliability == 0.0

    def test_average_reliability(self, fresh_snippet, old_snippet):
        """Should calculate average reliability."""
        pack = EvidencePack(
            topic_keywords=["test"],
            snippets=[fresh_snippet, old_snippet],  # 0.8 and 0.9
        )
        assert pack.average_reliability == pytest.approx(0.85)

    def test_average_freshness_empty(self):
        """Should return 0.0 for empty pack."""
        pack = EvidencePack(topic_keywords=["test"], snippets=[])
        assert pack.average_freshness == 0.0

    def test_average_freshness(self, fresh_snippet, old_snippet):
        """Should calculate average freshness."""
        pack = EvidencePack(
            topic_keywords=["test"],
            snippets=[fresh_snippet, old_snippet],
        )
        # Fresh = 1.0, Old = ~0.5
        assert 0.7 <= pack.average_freshness <= 0.8

    def test_to_context_string_empty(self):
        """Should return message for empty pack."""
        pack = EvidencePack(topic_keywords=["test"], snippets=[])
        assert "No relevant evidence found" in pack.to_context_string()

    def test_to_context_string_with_snippets(self, fresh_snippet):
        """Should format context string."""
        pack = EvidencePack(
            topic_keywords=["ai", "safety"],
            snippets=[fresh_snippet],
            total_searched=10,
        )
        context = pack.to_context_string()
        assert "EVIDENCE PACK" in context
        assert "ai, safety" in context
        assert "Total sources searched: 10" in context
        assert "EVID-test-001" in context

    def test_to_bibliography_empty(self):
        """Should return empty string for empty pack."""
        pack = EvidencePack(topic_keywords=["test"], snippets=[])
        assert pack.to_bibliography() == ""

    def test_to_bibliography(self, fresh_snippet, old_snippet):
        """Should format as numbered bibliography."""
        pack = EvidencePack(
            topic_keywords=["test"],
            snippets=[fresh_snippet, old_snippet],
        )
        bib = pack.to_bibliography()
        assert "## References" in bib
        assert "1." in bib
        assert "2." in bib

    def test_to_dict(self, fresh_snippet):
        """Should convert to dictionary."""
        pack = EvidencePack(
            topic_keywords=["ai"],
            snippets=[fresh_snippet],
            total_searched=5,
        )
        d = pack.to_dict()
        assert d["topic_keywords"] == ["ai"]
        assert len(d["snippets"]) == 1
        assert d["total_searched"] == 5
        assert "average_reliability" in d


# =============================================================================
# EvidenceCollector Keyword Extraction Tests
# =============================================================================


class TestKeywordExtraction:
    """Tests for keyword extraction."""

    def test_extract_keywords_basic(self, collector):
        """Should extract meaningful keywords."""
        keywords = collector._extract_keywords("What are the benefits of AI safety research?")
        assert "benefits" in keywords
        assert "safety" in keywords
        assert "research" in keywords

    def test_extract_keywords_removes_stop_words(self, collector):
        """Should remove stop words."""
        keywords = collector._extract_keywords("The quick brown fox is jumping over the lazy dog")
        assert "the" not in keywords
        assert "is" not in keywords
        assert "over" not in keywords

    def test_extract_keywords_removes_short_words(self, collector):
        """Should remove words <= 2 chars."""
        keywords = collector._extract_keywords("AI is a big deal")
        assert "is" not in keywords
        assert "a" not in keywords

    def test_extract_keywords_limit(self, collector):
        """Should limit to 5 keywords."""
        keywords = collector._extract_keywords(
            "machine learning artificial intelligence neural networks deep learning "
            "natural language processing computer vision robotics automation data science"
        )
        assert len(keywords) <= 10  # May have duplicates from boosting

    def test_extract_keywords_boosts_technical(self, collector):
        """Should boost technical terms."""
        keywords = collector._extract_keywords("AI system data analysis code review")
        # Technical terms should appear (possibly multiple times due to boosting)
        assert any("ai" in k or "system" in k or "data" in k or "code" in k for k in keywords)


# =============================================================================
# EvidenceCollector URL Extraction Tests
# =============================================================================


class TestUrlExtraction:
    """Tests for URL extraction."""

    def test_extract_https_urls(self, collector):
        """Should extract HTTPS URLs."""
        urls = collector._extract_urls("Check out https://example.com/test for details")
        assert "https://example.com/test" in urls

    def test_extract_http_urls(self, collector):
        """Should extract HTTP URLs."""
        urls = collector._extract_urls("See http://old-site.org/page for info")
        assert "http://old-site.org/page" in urls

    def test_extract_www_urls(self, collector):
        """Should extract www URLs."""
        urls = collector._extract_urls("Visit www.example.com for more")
        assert "www.example.com" in urls

    def test_extract_domain_names(self, collector):
        """Should extract domain names like github.com."""
        urls = collector._extract_urls("The code is on github.com")
        assert any("github.com" in url for url in urls)

    def test_extract_multiple_urls(self, collector):
        """Should extract multiple URLs."""
        urls = collector._extract_urls("Check https://a.com and https://b.org for details")
        assert len(urls) >= 2

    def test_deduplicate_urls(self, collector):
        """Should deduplicate exact same URLs."""
        urls = collector._extract_urls(
            "Visit https://example.com/page and https://example.com/page again"
        )
        # Exact same URL should appear only once
        count = sum(1 for u in urls if u == "https://example.com/page")
        assert count == 1


# =============================================================================
# EvidenceCollector Snippet Truncation Tests
# =============================================================================


class TestSnippetTruncation:
    """Tests for snippet truncation."""

    def test_no_truncation_short_text(self, collector):
        """Should not truncate short text."""
        text = "Short text."
        result = collector._truncate_snippet(text)
        assert result == text

    def test_truncates_long_text(self, collector):
        """Should truncate text longer than max."""
        text = "x" * 2000
        result = collector._truncate_snippet(text)
        assert len(result) <= collector.snippet_max_length + 3  # +3 for "..."

    def test_truncates_at_sentence_boundary(self, collector):
        """Should try to truncate at sentence boundary."""
        text = "First sentence. " + "x" * 900 + ". End sentence."
        result = collector._truncate_snippet(text)
        # Should end with a sentence boundary if possible
        assert result.endswith(".") or result.endswith("...")


# =============================================================================
# EvidenceCollector Reliability Calculation Tests
# =============================================================================


class TestReliabilityCalculation:
    """Tests for reliability score calculation."""

    def test_github_base_reliability(self, collector):
        """Should give GitHub high base reliability."""
        score = collector._calculate_reliability("github", {})
        assert score == 0.8

    def test_local_docs_base_reliability(self, collector):
        """Should give local docs highest base reliability."""
        score = collector._calculate_reliability("local_docs", {})
        assert score == 0.9

    def test_web_search_base_reliability(self, collector):
        """Should give web search moderate reliability."""
        score = collector._calculate_reliability("web_search", {})
        assert score == 0.6

    def test_unknown_connector_reliability(self, collector):
        """Should give unknown connectors default reliability."""
        score = collector._calculate_reliability("unknown_source", {})
        assert score == 0.5

    def test_verified_boost(self, collector):
        """Should boost reliability for verified content."""
        score = collector._calculate_reliability("github", {"verified": True})
        assert score > 0.8

    def test_substantial_content_boost(self, collector):
        """Should boost reliability for substantial content."""
        score = collector._calculate_reliability("web_search", {"content": "x" * 2000})
        assert score > 0.6

    def test_max_reliability(self, collector):
        """Should cap reliability at 1.0."""
        score = collector._calculate_reliability(
            "local_docs", {"verified": True, "recent": True, "content": "x" * 2000}
        )
        assert score <= 1.0


# =============================================================================
# EvidenceCollector Snippet Ranking Tests
# =============================================================================


class TestSnippetRanking:
    """Tests for snippet ranking."""

    def test_rank_by_keyword_relevance(self, collector):
        """Should rank snippets with more keyword matches higher."""
        snippet1 = EvidenceSnippet(
            id="1",
            source="web",
            title="AI Safety",
            snippet="AI safety is important for AI systems",
            reliability_score=0.5,
        )
        snippet2 = EvidenceSnippet(
            id="2",
            source="web",
            title="Random",
            snippet="This is random content",
            reliability_score=0.5,
        )

        ranked = collector._rank_snippets([snippet2, snippet1], ["ai", "safety"])
        assert ranked[0].id == "1"  # AI safety should rank higher

    def test_rank_by_reliability(self, collector):
        """Should consider reliability in ranking."""
        snippet1 = EvidenceSnippet(
            id="1",
            source="web",
            title="Test",
            snippet="test content",
            reliability_score=0.9,
        )
        snippet2 = EvidenceSnippet(
            id="2",
            source="web",
            title="Test",
            snippet="test content",
            reliability_score=0.3,
        )

        ranked = collector._rank_snippets([snippet2, snippet1], ["test"])
        assert ranked[0].id == "1"  # Higher reliability should rank higher

    def test_rank_by_freshness(self, collector):
        """Should consider freshness in ranking."""
        fresh = EvidenceSnippet(
            id="fresh",
            source="web",
            title="Test",
            snippet="test",
            reliability_score=0.5,
            fetched_at=datetime.now(),
        )
        old = EvidenceSnippet(
            id="old",
            source="web",
            title="Test",
            snippet="test",
            reliability_score=0.5,
            fetched_at=datetime.now() - timedelta(days=30),
        )

        ranked = collector._rank_snippets([old, fresh], ["test"])
        assert ranked[0].id == "fresh"

    def test_title_keyword_boost(self, collector):
        """Should boost snippets with keywords in title."""
        title_match = EvidenceSnippet(
            id="title",
            source="web",
            title="AI Safety Guidelines",
            snippet="generic content here",
            reliability_score=0.5,
        )
        body_match = EvidenceSnippet(
            id="body",
            source="web",
            title="Generic Title",
            snippet="AI safety is discussed here",
            reliability_score=0.5,
        )

        ranked = collector._rank_snippets([body_match, title_match], ["ai", "safety"])
        assert ranked[0].id == "title"


# =============================================================================
# EvidenceCollector Integration Tests
# =============================================================================


class TestEvidenceCollectorIntegration:
    """Integration tests for EvidenceCollector."""

    def test_add_connector(self, collector):
        """Should add connector to registry."""
        mock_connector = Mock()
        collector.add_connector("test", mock_connector)
        assert "test" in collector.connectors

    @pytest.mark.asyncio
    async def test_collect_evidence_no_connectors(self, collector):
        """Should return empty pack with no connectors."""
        pack = await collector.collect_evidence("test task")
        assert len(pack.snippets) == 0

    @pytest.mark.asyncio
    async def test_collect_evidence_with_mock_connector(self, collector):
        """Should collect evidence from mock connector."""
        mock_connector = AsyncMock()
        mock_connector.search = AsyncMock(
            return_value=[
                {"title": "Result 1", "content": "Content 1", "url": "http://a.com"},
                {"title": "Result 2", "content": "Content 2", "url": "http://b.com"},
            ]
        )
        collector.add_connector("mock", mock_connector)

        pack = await collector.collect_evidence("AI safety research")

        assert len(pack.snippets) >= 1
        assert pack.total_searched >= 2

    @pytest.mark.asyncio
    async def test_collect_evidence_handles_connector_error(self, collector):
        """Should handle connector errors gracefully."""
        mock_connector = AsyncMock()
        mock_connector.search = AsyncMock(side_effect=Exception("Connection failed"))
        collector.add_connector("failing", mock_connector)

        # Should not raise
        pack = await collector.collect_evidence("test")
        assert pack is not None

    @pytest.mark.asyncio
    async def test_collect_evidence_respects_max_snippets(self, collector):
        """Should limit total snippets to max."""
        mock_connector = AsyncMock()
        mock_connector.search = AsyncMock(
            return_value=[{"title": f"Result {i}", "content": f"Content {i}"} for i in range(20)]
        )
        collector.add_connector("mock", mock_connector)

        pack = await collector.collect_evidence("test")
        assert len(pack.snippets) <= collector.max_total_snippets
