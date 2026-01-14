"""Tests for ArXiv connector."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from aragora.connectors.arxiv import ArXivConnector, ARXIV_CATEGORIES
from aragora.connectors.base import Evidence
from aragora.reasoning.provenance import SourceType


# Sample ArXiv XML response
SAMPLE_ARXIV_XML = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2309.12345v1</id>
    <title>Attention Is All You Need</title>
    <summary>The dominant sequence transduction models are based on complex recurrent or convolutional neural networks.</summary>
    <author><name>Ashish Vaswani</name></author>
    <author><name>Noam Shazeer</name></author>
    <author><name>Niki Parmar</name></author>
    <published>2023-09-15T00:00:00Z</published>
    <arxiv:primary_category term="cs.CL"/>
    <category term="cs.LG"/>
    <link title="pdf" href="http://arxiv.org/pdf/2309.12345v1"/>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2308.54321v2</id>
    <title>BERT: Pre-training of Deep Bidirectional Transformers</title>
    <summary>We introduce a new language representation model called BERT.</summary>
    <author><name>Jacob Devlin</name></author>
    <published>2023-08-10T00:00:00Z</published>
    <arxiv:primary_category term="cs.CL"/>
  </entry>
</feed>
"""


class TestArXivConnector:
    """Tests for ArXivConnector."""

    @pytest.fixture
    def connector(self):
        """Create an ArXiv connector for testing."""
        return ArXivConnector(rate_limit_delay=0.0)  # Disable rate limiting for tests

    def test_connector_properties(self, connector):
        """Test connector property methods."""
        assert connector.source_type == SourceType.EXTERNAL_API
        assert connector.name == "ArXiv"
        assert connector.is_available  # Assumes httpx is installed

    def test_categories(self, connector):
        """Test available categories."""
        categories = connector.get_categories()

        assert "cs.AI" in categories
        assert "cs.CL" in categories
        assert "cs.LG" in categories
        assert categories["cs.AI"] == "Artificial Intelligence"

    def test_parse_arxiv_response(self, connector):
        """Test XML response parsing."""
        results = connector._parse_arxiv_response(SAMPLE_ARXIV_XML)

        assert len(results) == 2

        # Check first result
        paper1 = results[0]
        assert paper1.id == "arxiv:2309.12345v1"
        assert "Attention Is All You Need" in paper1.title
        assert "sequence transduction" in paper1.content
        assert "Ashish Vaswani" in paper1.author
        assert paper1.source_type == SourceType.EXTERNAL_API
        assert paper1.authority == 0.9  # Academic papers have high authority
        assert "cs.CL" in paper1.metadata["categories"]

        # Check second result
        paper2 = results[1]
        assert paper2.id == "arxiv:2308.54321v2"
        assert "BERT" in paper2.title

    @pytest.mark.asyncio
    async def test_search_mocked(self, connector):
        """Test search with mocked HTTP response."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.text = SAMPLE_ARXIV_XML
            mock_response.raise_for_status = MagicMock()

            mock_client_instance = AsyncMock()
            mock_client_instance.get = AsyncMock(return_value=mock_response)
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_client_instance

            results = await connector.search("transformer attention", limit=5)

            assert len(results) == 2
            assert all(isinstance(r, Evidence) for r in results)

    @pytest.mark.asyncio
    async def test_search_with_category(self, connector):
        """Test search with category filter."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.text = SAMPLE_ARXIV_XML
            mock_response.raise_for_status = MagicMock()

            mock_client_instance = AsyncMock()
            mock_client_instance.get = AsyncMock(return_value=mock_response)
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_client_instance

            results = await connector.search("neural networks", category="cs.AI", limit=5)

            # Verify category was included in query
            call_args = mock_client_instance.get.call_args
            params = call_args[1]["params"]
            assert "cat:cs.AI" in params["search_query"]

    @pytest.mark.asyncio
    async def test_fetch_mocked(self, connector):
        """Test fetch with mocked HTTP response."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.text = SAMPLE_ARXIV_XML
            mock_response.raise_for_status = MagicMock()

            mock_client_instance = AsyncMock()
            mock_client_instance.get = AsyncMock(return_value=mock_response)
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_client_instance

            result = await connector.fetch("2309.12345")

            assert result is not None
            assert "Attention" in result.title

    @pytest.mark.asyncio
    async def test_fetch_cached(self, connector):
        """Test that cached results are returned."""
        # Pre-populate cache
        cached_evidence = Evidence(
            id="arxiv:cached-123",
            source_type=SourceType.EXTERNAL_API,
            source_id="cached-123",
            content="Cached paper content",
            title="Cached Paper",
        )
        connector._cache_put("cached-123", cached_evidence)

        # Fetch should return cached result without HTTP call
        result = await connector.fetch("cached-123")

        assert result is not None
        assert result.title == "Cached Paper"
        assert result.content == "Cached paper content"

    @pytest.mark.asyncio
    async def test_search_by_author_mocked(self, connector):
        """Test author search."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.text = SAMPLE_ARXIV_XML
            mock_response.raise_for_status = MagicMock()

            mock_client_instance = AsyncMock()
            mock_client_instance.get = AsyncMock(return_value=mock_response)
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_client_instance

            results = await connector.search_by_author("Vaswani")

            call_args = mock_client_instance.get.call_args
            params = call_args[1]["params"]
            assert "au:" in params["search_query"]

    @pytest.mark.asyncio
    async def test_search_recent_mocked(self, connector):
        """Test recent papers search."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.text = SAMPLE_ARXIV_XML
            mock_response.raise_for_status = MagicMock()

            mock_client_instance = AsyncMock()
            mock_client_instance.get = AsyncMock(return_value=mock_response)
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_client_instance

            results = await connector.search_recent(category="cs.AI")

            call_args = mock_client_instance.get.call_args
            params = call_args[1]["params"]
            assert params["sortBy"] == "submittedDate"

    @pytest.mark.asyncio
    async def test_search_timeout_handling(self, connector):
        """Test handling of timeout errors."""
        import httpx

        with patch("httpx.AsyncClient") as mock_client:
            mock_client_instance = AsyncMock()
            mock_client_instance.get = AsyncMock(side_effect=httpx.TimeoutException("Timeout"))
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_client_instance

            results = await connector.search("test query")

            assert results == []  # Should return empty list on timeout

    @pytest.mark.asyncio
    async def test_search_http_error_handling(self, connector):
        """Test handling of HTTP errors."""
        import httpx

        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 500
            mock_response.raise_for_status = MagicMock(
                side_effect=httpx.HTTPStatusError("Server Error", request=MagicMock(), response=mock_response)
            )

            mock_client_instance = AsyncMock()
            mock_client_instance.get = AsyncMock(return_value=mock_response)
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_client_instance

            results = await connector.search("test query")

            assert results == []


class TestArXivCategories:
    """Tests for ArXiv category constants."""

    def test_cs_categories_present(self):
        """Test that CS categories are present."""
        assert "cs" in ARXIV_CATEGORIES
        assert "cs.AI" in ARXIV_CATEGORIES
        assert "cs.CL" in ARXIV_CATEGORIES
        assert "cs.CV" in ARXIV_CATEGORIES
        assert "cs.LG" in ARXIV_CATEGORIES

    def test_category_descriptions(self):
        """Test category descriptions are strings."""
        for code, description in ARXIV_CATEGORIES.items():
            assert isinstance(code, str)
            assert isinstance(description, str)
            assert len(description) > 0
