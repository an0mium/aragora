"""
Tests for evidence collection system.

Tests:
- Evidence snippet creation
- Evidence pack formatting
- Keyword extraction
- Connector integration
- Snippet ranking and filtering
"""

import asyncio
from unittest.mock import Mock, AsyncMock, patch

import pytest

from aragora.evidence.collector import (
    EvidenceSnippet,
    EvidencePack,
    EvidenceCollector,
)


class TestEvidenceSnippet:
    """Test evidence snippet dataclass."""

    def test_snippet_creation(self):
        """Snippet should be created with defaults."""
        snippet = EvidenceSnippet(
            id="test-1",
            source="github",
            title="Test Document",
            snippet="This is test content.",
        )

        assert snippet.id == "test-1"
        assert snippet.reliability_score == 0.5
        assert snippet.url == ""
        assert snippet.metadata == {}

    def test_snippet_with_metadata(self):
        """Snippet should accept metadata."""
        snippet = EvidenceSnippet(
            id="test-1",
            source="local_docs",
            title="API Reference",
            snippet="Content here",
            url="https://example.com/docs",
            reliability_score=0.9,
            metadata={"language": "python", "version": "3.11"},
        )

        assert snippet.reliability_score == 0.9
        assert snippet.url == "https://example.com/docs"
        assert snippet.metadata["language"] == "python"

    def test_to_text_block(self):
        """Snippet should format as text block."""
        snippet = EvidenceSnippet(
            id="abc123",
            source="github",
            title="Example Code",
            snippet="def hello(): print('world')",
            reliability_score=0.8,
        )

        block = snippet.to_text_block()

        assert "EVID-abc123" in block
        assert "github" in block
        assert "0.8 reliability" in block
        assert "Example Code" in block
        assert "def hello()" in block

    def test_long_snippet_truncated(self):
        """Long snippets should be truncated in text block."""
        long_content = "x" * 1000
        snippet = EvidenceSnippet(
            id="long",
            source="web",
            title="Long Document",
            snippet=long_content,
        )

        block = snippet.to_text_block()

        assert "..." in block
        assert len(block) < 1000 + 200  # Content truncated plus overhead


class TestEvidencePack:
    """Test evidence pack collection."""

    def test_pack_creation(self):
        """Pack should be created with snippets."""
        snippets = [
            EvidenceSnippet("1", "github", "Doc 1", "Content 1"),
            EvidenceSnippet("2", "web", "Doc 2", "Content 2"),
        ]

        pack = EvidencePack(
            topic_keywords=["python", "testing"],
            snippets=snippets,
            total_searched=10,
        )

        assert len(pack.snippets) == 2
        assert pack.total_searched == 10
        assert "python" in pack.topic_keywords

    def test_empty_pack_context_string(self):
        """Empty pack should return no evidence message."""
        pack = EvidencePack(
            topic_keywords=["test"],
            snippets=[],
            total_searched=5,
        )

        context = pack.to_context_string()

        assert "No relevant evidence found" in context

    def test_pack_context_string_format(self):
        """Pack should format as context string."""
        snippets = [
            EvidenceSnippet("1", "github", "Doc 1", "Content 1"),
        ]

        pack = EvidencePack(
            topic_keywords=["api", "design"],
            snippets=snippets,
            total_searched=3,
        )

        context = pack.to_context_string()

        assert "EVIDENCE PACK" in context
        assert "api" in context
        assert "Total sources searched: 3" in context
        assert "END EVIDENCE PACK" in context


class TestEvidenceCollector:
    """Test evidence collector."""

    @pytest.fixture
    def collector(self):
        return EvidenceCollector()

    def test_collector_init(self, collector):
        """Collector should initialize with empty connectors."""
        assert collector.connectors == {}
        assert collector.max_total_snippets == 8
        assert collector.snippet_max_length == 1000

    def test_add_connector(self, collector):
        """Should add connectors."""
        mock_connector = Mock()
        collector.add_connector("test", mock_connector)

        assert "test" in collector.connectors
        assert collector.connectors["test"] is mock_connector


class TestKeywordExtraction:
    """Test keyword extraction from tasks."""

    @pytest.fixture
    def collector(self):
        return EvidenceCollector()

    def test_basic_extraction(self, collector):
        """Should extract keywords from task."""
        keywords = collector._extract_keywords("Design a rate limiting system for APIs")

        assert len(keywords) > 0
        assert any("rate" in k or "limit" in k or "system" in k for k in keywords)

    def test_stop_words_removed(self, collector):
        """Stop words should be filtered out."""
        keywords = collector._extract_keywords("The quick brown fox jumps over the lazy dog")

        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on"}
        for kw in keywords:
            assert kw.lower() not in stop_words

    def test_short_words_removed(self, collector):
        """Words shorter than 3 chars should be filtered."""
        keywords = collector._extract_keywords("I am a test of the keyword extraction")

        for kw in keywords:
            assert len(kw) >= 3

    def test_max_keywords(self, collector):
        """Should return at most 5 keywords."""
        keywords = collector._extract_keywords(
            "Design implement test verify deploy monitor maintain update secure optimize performance scalability reliability"
        )

        assert len(keywords) <= 5


class TestSnippetTruncation:
    """Test snippet truncation."""

    @pytest.fixture
    def collector(self):
        return EvidenceCollector()

    def test_short_text_unchanged(self, collector):
        """Short text should not be truncated."""
        text = "Short text."
        result = collector._truncate_snippet(text)
        assert result == text

    def test_long_text_truncated(self, collector):
        """Long text should be truncated."""
        text = "x" * 2000
        result = collector._truncate_snippet(text)
        assert len(result) <= collector.snippet_max_length + 3  # +3 for "..."

    def test_truncation_at_sentence_boundary(self, collector):
        """Should try to truncate at sentence boundary."""
        text = "First sentence. " + "x" * 800 + ". More text here that goes beyond limit."
        result = collector._truncate_snippet(text)

        # Should end at a sentence if possible
        assert result.endswith(".") or result.endswith("...")


class TestReliabilityScoring:
    """Test reliability score calculation."""

    @pytest.fixture
    def collector(self):
        return EvidenceCollector()

    def test_github_base_score(self, collector):
        """GitHub should have high base reliability."""
        score = collector._calculate_reliability("github", {})
        assert score >= 0.7

    def test_local_docs_highest(self, collector):
        """Local docs should have highest reliability."""
        score = collector._calculate_reliability("local_docs", {})
        assert score >= 0.8

    def test_web_search_lower(self, collector):
        """Web search should have lower reliability."""
        score = collector._calculate_reliability("web_search", {})
        assert score < 0.8

    def test_verified_boost(self, collector):
        """Verified content should get boost."""
        base = collector._calculate_reliability("web_search", {})
        verified = collector._calculate_reliability("web_search", {"verified": True})
        assert verified > base

    def test_max_score_capped(self, collector):
        """Score should never exceed 1.0."""
        score = collector._calculate_reliability(
            "local_docs",
            {"verified": True, "recent": True, "content": "x" * 2000},
        )
        assert score <= 1.0


class TestSnippetRanking:
    """Test snippet ranking algorithm."""

    @pytest.fixture
    def collector(self):
        return EvidenceCollector()

    def test_keyword_match_ranking(self, collector):
        """Snippets with more keyword matches should rank higher."""
        snippets = [
            EvidenceSnippet("1", "web", "No matches", "Unrelated content"),
            EvidenceSnippet("2", "web", "API rate limiting", "Rate limiting for APIs"),
        ]

        ranked = collector._rank_snippets(snippets, ["rate", "limiting", "api"])

        assert ranked[0].id == "2"  # Better keyword match

    def test_title_match_boost(self, collector):
        """Keywords in title should provide extra boost."""
        snippets = [
            EvidenceSnippet("1", "web", "Generic title", "Rate limiting content"),
            EvidenceSnippet("2", "web", "Rate Limiting Guide", "Some content"),
        ]

        ranked = collector._rank_snippets(snippets, ["rate", "limiting"])

        assert ranked[0].id == "2"  # Title match

    def test_reliability_considered(self, collector):
        """Reliability score should factor into ranking."""
        snippets = [
            EvidenceSnippet("1", "web", "Keyword match", "keyword", reliability_score=0.3),
            EvidenceSnippet("2", "docs", "Keyword match", "keyword", reliability_score=0.9),
        ]

        ranked = collector._rank_snippets(snippets, ["keyword"])

        assert ranked[0].id == "2"  # Higher reliability


@pytest.mark.asyncio
class TestAsyncCollection:
    """Test async evidence collection."""

    async def test_collect_empty_connectors(self):
        """Should return empty pack with no connectors."""
        collector = EvidenceCollector()

        pack = await collector.collect_evidence("test task")

        assert len(pack.snippets) == 0
        assert pack.total_searched == 0

    async def test_collect_with_mock_connector(self):
        """Should collect from mock connector."""
        collector = EvidenceCollector()

        # Create mock connector
        mock_connector = Mock()
        mock_connector.search = AsyncMock(
            return_value=[
                {
                    "title": "Test Result",
                    "content": "Test content here",
                    "url": "https://example.com",
                }
            ]
        )

        collector.add_connector("test", mock_connector)

        pack = await collector.collect_evidence("test query")

        assert len(pack.snippets) > 0
        mock_connector.search.assert_called_once()

    async def test_connector_error_handled(self):
        """Should handle connector errors gracefully."""
        collector = EvidenceCollector()

        # Create failing mock connector
        mock_connector = Mock()
        mock_connector.search = AsyncMock(side_effect=Exception("Connection failed"))

        collector.add_connector("failing", mock_connector)

        # Should not raise, just return empty results
        pack = await collector.collect_evidence("test query")

        assert pack.total_searched == 0

    async def test_enabled_connectors_filter(self):
        """Should only use enabled connectors."""
        collector = EvidenceCollector()

        mock1 = Mock()
        mock1.search = AsyncMock(return_value=[{"title": "1", "content": "c1"}])
        mock2 = Mock()
        mock2.search = AsyncMock(return_value=[{"title": "2", "content": "c2"}])

        collector.add_connector("conn1", mock1)
        collector.add_connector("conn2", mock2)

        # Only enable conn1
        pack = await collector.collect_evidence("test", enabled_connectors=["conn1"])

        mock1.search.assert_called_once()
        mock2.search.assert_not_called()

    async def test_max_snippets_enforced(self):
        """Should limit total snippets."""
        collector = EvidenceCollector()
        collector.max_total_snippets = 3

        # Create connector returning many results
        mock_connector = Mock()
        mock_connector.search = AsyncMock(
            return_value=[{"title": f"Result {i}", "content": f"Content {i}"} for i in range(10)]
        )

        collector.add_connector("test", mock_connector)

        pack = await collector.collect_evidence("test query")

        assert len(pack.snippets) <= 3
