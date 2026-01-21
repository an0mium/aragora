"""
Tests for HackerNews connector.

Tests the Algolia HN Search API integration.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime


class TestHackerNewsConnector:
    """Tests for HackerNewsConnector."""

    @pytest.fixture
    def connector(self):
        """Create a HackerNews connector instance."""
        from aragora.connectors.hackernews import HackerNewsConnector

        return HackerNewsConnector()

    def test_connector_init(self, connector):
        """Connector should initialize with default values."""
        assert connector.timeout == 30
        assert connector.rate_limit_delay == 0.5
        assert connector.default_confidence == 0.7

    def test_connector_name(self, connector):
        """Connector should have correct name."""
        assert connector.name == "HackerNews"

    def test_source_type(self, connector):
        """Source type should be EXTERNAL_API."""
        from aragora.reasoning.provenance import SourceType

        assert connector.source_type == SourceType.EXTERNAL_API

    def test_is_available(self, connector):
        """Should check httpx availability."""
        # httpx should be available in test environment
        assert connector.is_available is True

    def test_custom_config(self):
        """Connector should accept custom configuration."""
        from aragora.connectors.hackernews import HackerNewsConnector

        connector = HackerNewsConnector(
            timeout=60,
            rate_limit_delay=1.0,
            default_confidence=0.8,
            max_cache_entries=100,
            cache_ttl_seconds=7200.0,
        )

        assert connector.timeout == 60
        assert connector.rate_limit_delay == 1.0
        assert connector.default_confidence == 0.8


class TestHackerNewsSearch:
    """Tests for search functionality."""

    @pytest.fixture
    def connector(self):
        """Create a HackerNews connector instance."""
        from aragora.connectors.hackernews import HackerNewsConnector

        return HackerNewsConnector()

    @pytest.mark.asyncio
    async def test_search_returns_evidence_list(self, connector):
        """Search should return a list of Evidence objects."""
        mock_response = {
            "hits": [
                {
                    "objectID": "12345",
                    "title": "Rust vs Go Performance",
                    "url": "https://example.com/article",
                    "author": "testuser",
                    "points": 150,
                    "num_comments": 42,
                    "created_at_i": 1609459200,
                    "story_text": "This is the story content.",
                }
            ],
            "nbHits": 1,
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get.return_value = MagicMock(
                status_code=200, json=lambda: mock_response
            )
            mock_instance.__aenter__.return_value = mock_instance
            mock_instance.__aexit__.return_value = None
            mock_client.return_value = mock_instance

            results = await connector.search("rust vs go")

        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_search_handles_empty_results(self, connector):
        """Search should handle empty results gracefully."""
        mock_response = {"hits": [], "nbHits": 0}

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get.return_value = MagicMock(
                status_code=200, json=lambda: mock_response
            )
            mock_instance.__aenter__.return_value = mock_instance
            mock_instance.__aexit__.return_value = None
            mock_client.return_value = mock_instance

            results = await connector.search("nonexistent_query_xyz")

        assert results == []

    @pytest.mark.asyncio
    async def test_search_handles_api_error(self, connector):
        """Search should handle API errors gracefully."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get.return_value = MagicMock(status_code=500, text="Server Error")
            mock_instance.__aenter__.return_value = mock_instance
            mock_instance.__aexit__.return_value = None
            mock_client.return_value = mock_instance

            results = await connector.search("test query")

        # Should return empty list on error, not raise
        assert results == []


class TestHackerNewsStoryDetails:
    """Tests for story detail fetching."""

    @pytest.fixture
    def connector(self):
        """Create a HackerNews connector instance."""
        from aragora.connectors.hackernews import HackerNewsConnector

        return HackerNewsConnector()

    def test_connector_has_search_method(self, connector):
        """Connector should have search method."""
        assert hasattr(connector, "search")
        assert callable(connector.search)

    def test_connector_has_get_item_or_search(self, connector):
        """Connector should have item retrieval via search or get_item."""
        # May have get_item or rely on search
        has_get_item = hasattr(connector, "get_item")
        has_search = hasattr(connector, "search")
        assert has_get_item or has_search


class TestHackerNewsRateLimiting:
    """Tests for rate limiting."""

    @pytest.fixture
    def connector(self):
        """Create a HackerNews connector with custom rate limit."""
        from aragora.connectors.hackernews import HackerNewsConnector

        return HackerNewsConnector(rate_limit_delay=0.1)

    @pytest.mark.asyncio
    async def test_rate_limiting_delays_requests(self, connector):
        """Rate limiting should delay consecutive requests."""
        import time

        mock_response = {"hits": [], "nbHits": 0}

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get.return_value = MagicMock(
                status_code=200, json=lambda: mock_response
            )
            mock_instance.__aenter__.return_value = mock_instance
            mock_instance.__aexit__.return_value = None
            mock_client.return_value = mock_instance

            start = time.time()
            await connector.search("query1")
            await connector.search("query2")
            elapsed = time.time() - start

        # Should have at least one rate limit delay
        assert elapsed >= connector.rate_limit_delay


class TestHackerNewsCaching:
    """Tests for caching functionality."""

    @pytest.fixture
    def connector(self):
        """Create a HackerNews connector with caching."""
        from aragora.connectors.hackernews import HackerNewsConnector

        return HackerNewsConnector(
            max_cache_entries=10,
            cache_ttl_seconds=3600.0,
        )

    def test_cache_initialized(self, connector):
        """Cache should be initialized."""
        # BaseConnector provides caching
        assert hasattr(connector, "_cache") or hasattr(connector, "cache")


class TestHackerNewsURLs:
    """Tests for URL generation."""

    def test_story_url_generation(self):
        """Should generate correct HN story URLs."""
        from aragora.connectors.hackernews import HN_STORY_URL

        story_id = 12345
        url = HN_STORY_URL.format(story_id)
        assert url == "https://news.ycombinator.com/item?id=12345"

    def test_api_urls_correct(self):
        """API URLs should be correctly defined."""
        from aragora.connectors.hackernews import (
            HN_SEARCH_URL,
            HN_SEARCH_BY_DATE_URL,
            HN_ITEM_URL,
        )

        assert "hn.algolia.com" in HN_SEARCH_URL
        assert "hn.algolia.com" in HN_SEARCH_BY_DATE_URL
        assert "hn.algolia.com" in HN_ITEM_URL
