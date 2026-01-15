"""Tests for NewsAPI connector."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from aragora.connectors.newsapi import (
    NewsAPIConnector,
    HIGH_CREDIBILITY_SOURCES,
    MEDIUM_CREDIBILITY_SOURCES,
)
from aragora.connectors.base import Evidence
from aragora.reasoning.provenance import SourceType


# Sample NewsAPI responses
SAMPLE_EVERYTHING_RESPONSE = {
    "status": "ok",
    "totalResults": 2,
    "articles": [
        {
            "source": {"id": "bbc-news", "name": "BBC News"},
            "author": "John Smith",
            "title": "AI Breakthrough: New Model Shows Promise",
            "description": "Researchers announce major advancement in AI capabilities.",
            "url": "https://bbc.com/news/ai-breakthrough",
            "urlToImage": "https://bbc.com/image.jpg",
            "publishedAt": "2026-01-14T10:00:00Z",
            "content": "Researchers at a major lab have announced a significant breakthrough in artificial intelligence capabilities...",
        },
        {
            "source": {"id": "techcrunch", "name": "TechCrunch"},
            "author": "Jane Doe",
            "title": "Startup Raises $50M for AI Platform",
            "description": "A new AI startup secures funding.",
            "url": "https://techcrunch.com/startup-funding",
            "publishedAt": "2026-01-13T15:30:00Z",
            "content": "A Silicon Valley startup focused on enterprise AI solutions has raised $50 million in Series B funding...",
        },
    ],
}

SAMPLE_HEADLINES_RESPONSE = {
    "status": "ok",
    "totalResults": 1,
    "articles": [
        {
            "source": {"id": "reuters", "name": "Reuters"},
            "author": None,
            "title": "Markets Rally on Economic Data",
            "description": "Stock markets rise following positive economic indicators.",
            "url": "https://reuters.com/markets-rally",
            "publishedAt": "2026-01-14T09:00:00Z",
            "content": "Global stock markets rallied today...",
        },
    ],
}

SAMPLE_SOURCES_RESPONSE = {
    "status": "ok",
    "sources": [
        {
            "id": "bbc-news",
            "name": "BBC News",
            "description": "British Broadcasting Corporation",
            "category": "general",
            "language": "en",
            "country": "gb",
        },
        {
            "id": "techcrunch",
            "name": "TechCrunch",
            "description": "Tech news and analysis",
            "category": "technology",
            "language": "en",
            "country": "us",
        },
    ],
}


class TestNewsAPIConnector:
    """Tests for NewsAPIConnector."""

    @pytest.fixture
    def connector(self):
        """Create a NewsAPI connector for testing."""
        return NewsAPIConnector(
            api_key="test-api-key",
            rate_limit_delay=0.0,
        )

    @pytest.fixture
    def unconfigured_connector(self):
        """Create an unconfigured connector."""
        return NewsAPIConnector(api_key="", rate_limit_delay=0.0)

    def test_connector_properties(self, connector):
        """Test connector property methods."""
        assert connector.source_type == SourceType.EXTERNAL_API
        assert connector.name == "NewsAPI"
        assert connector.is_available
        assert connector.is_configured

    def test_unconfigured_connector(self, unconfigured_connector):
        """Test unconfigured connector properties."""
        assert not unconfigured_connector.is_configured

    @pytest.mark.asyncio
    async def test_search_unconfigured(self, unconfigured_connector):
        """Test that unconfigured connector returns empty results."""
        results = await unconfigured_connector.search("test query")
        assert results == []

    @pytest.mark.asyncio
    async def test_search_mocked(self, connector):
        """Test search with mocked HTTP response."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.json.return_value = SAMPLE_EVERYTHING_RESPONSE
            mock_response.raise_for_status = MagicMock()
            mock_response.status_code = 200

            mock_client_instance = AsyncMock()
            mock_client_instance.get = AsyncMock(return_value=mock_response)
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_client_instance

            results = await connector.search("artificial intelligence", limit=5)

            assert len(results) == 2
            assert all(isinstance(r, Evidence) for r in results)

            # Check first result (BBC - high credibility)
            bbc_result = results[0]
            assert "AI Breakthrough" in bbc_result.title
            assert bbc_result.author == "John Smith"
            assert bbc_result.confidence >= 0.75  # High credibility source

            # Check second result (TechCrunch - medium credibility)
            tc_result = results[1]
            assert "Startup" in tc_result.title
            assert tc_result.confidence >= 0.65

    @pytest.mark.asyncio
    async def test_get_headlines_mocked(self, connector):
        """Test headlines with mocked response."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.json.return_value = SAMPLE_HEADLINES_RESPONSE
            mock_response.raise_for_status = MagicMock()
            mock_response.status_code = 200

            mock_client_instance = AsyncMock()
            mock_client_instance.get = AsyncMock(return_value=mock_response)
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_client_instance

            results = await connector.get_headlines(country="us", category="business")

            assert len(results) == 1
            assert "Markets Rally" in results[0].title

    @pytest.mark.asyncio
    async def test_get_sources_mocked(self, connector):
        """Test sources endpoint."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.json.return_value = SAMPLE_SOURCES_RESPONSE
            mock_response.raise_for_status = MagicMock()

            mock_client_instance = AsyncMock()
            mock_client_instance.get = AsyncMock(return_value=mock_response)
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_client_instance

            sources = await connector.get_sources(category="technology")

            assert len(sources) == 2
            assert sources[0]["id"] == "bbc-news"

    @pytest.mark.asyncio
    async def test_fetch_returns_cached(self, connector):
        """Test fetch returns cached evidence."""
        # Pre-populate cache
        cached = Evidence(
            id="newsapi:cached123",
            source_type=SourceType.EXTERNAL_API,
            source_id="https://example.com/article",
            content="Cached article content",
            title="Cached Article",
        )
        connector._cache_put("newsapi:cached123", cached)

        result = await connector.fetch("newsapi:cached123")

        assert result is not None
        assert result.title == "Cached Article"

    @pytest.mark.asyncio
    async def test_fetch_uncached_returns_none(self, connector):
        """Test fetch returns None for uncached evidence."""
        result = await connector.fetch("newsapi:nonexistent")
        assert result is None

    def test_source_credibility_high(self, connector):
        """Test high credibility source scoring."""
        confidence, authority = connector._calculate_source_credibility("bbc-news", "BBC News")
        assert confidence == 0.80
        assert authority == 0.85

    def test_source_credibility_medium(self, connector):
        """Test medium credibility source scoring."""
        confidence, authority = connector._calculate_source_credibility("techcrunch", "TechCrunch")
        assert confidence == 0.70
        assert authority == 0.70

    def test_source_credibility_unknown(self, connector):
        """Test unknown source scoring."""
        confidence, authority = connector._calculate_source_credibility(
            "random-blog", "Random Blog"
        )
        assert confidence == connector.default_confidence
        assert authority == 0.55

    @pytest.mark.asyncio
    async def test_search_auth_error(self, connector):
        """Test handling of authentication errors."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 401

            mock_client_instance = AsyncMock()
            mock_client_instance.get = AsyncMock(return_value=mock_response)
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_client_instance

            results = await connector.search("test")
            assert results == []

    @pytest.mark.asyncio
    async def test_search_rate_limit_error(self, connector):
        """Test handling of rate limit errors."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 429

            mock_client_instance = AsyncMock()
            mock_client_instance.get = AsyncMock(return_value=mock_response)
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_client_instance

            results = await connector.search("test")
            assert results == []

    def test_parse_article_removed(self, connector):
        """Test that removed articles are skipped."""
        removed_article = {
            "title": "[Removed]",
            "content": "",
            "source": {"id": "test"},
        }
        result = connector._parse_article(removed_article)
        assert result is None

    def test_parse_article_no_content(self, connector):
        """Test that articles without content are skipped."""
        no_content_article = {
            "title": "Valid Title",
            "content": "",
            "description": "",
            "source": {"id": "test"},
        }
        result = connector._parse_article(no_content_article)
        assert result is None


class TestCredibilitySources:
    """Tests for credibility source sets."""

    def test_high_credibility_sources_present(self):
        """Test high credibility sources are defined."""
        assert "reuters" in HIGH_CREDIBILITY_SOURCES
        assert "bbc-news" in HIGH_CREDIBILITY_SOURCES
        assert "the-new-york-times" in HIGH_CREDIBILITY_SOURCES
        assert "associated-press" in HIGH_CREDIBILITY_SOURCES

    def test_medium_credibility_sources_present(self):
        """Test medium credibility sources are defined."""
        assert "cnn" in MEDIUM_CREDIBILITY_SOURCES
        assert "techcrunch" in MEDIUM_CREDIBILITY_SOURCES
        assert "wired" in MEDIUM_CREDIBILITY_SOURCES

    def test_no_overlap(self):
        """Test no overlap between high and medium credibility."""
        overlap = HIGH_CREDIBILITY_SOURCES & MEDIUM_CREDIBILITY_SOURCES
        assert len(overlap) == 0
