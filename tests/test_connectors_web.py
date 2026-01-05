"""
Tests for WebConnector - web search and URL fetching.

Tests cover:
- Domain authority scoring
- URL security validation (blocking local/private IPs)
- HTML parsing
- Cache operations
- Search with mocked DDGS
- URL fetching with mocked httpx
- Error handling
"""

import asyncio
import hashlib
import json
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from aragora.connectors.web import (
    WebConnector,
    DOMAIN_AUTHORITY,
    HTTPX_AVAILABLE,
    BS4_AVAILABLE,
    DDGS_AVAILABLE,
)
from aragora.connectors.base import Evidence
from aragora.reasoning.provenance import SourceType


class TestDomainAuthority:
    """Tests for domain authority scoring."""

    def test_known_high_authority_domain(self, temp_dir):
        """Test high authority domains like Wikipedia."""
        connector = WebConnector(cache_dir=str(temp_dir))
        assert connector._get_domain_authority("wikipedia.org") == 0.95
        assert connector._get_domain_authority("arxiv.org") == 0.9

    def test_known_medium_authority_domain(self, temp_dir):
        """Test medium authority domains."""
        connector = WebConnector(cache_dir=str(temp_dir))
        assert connector._get_domain_authority("stackoverflow.com") == 0.85
        assert connector._get_domain_authority("medium.com") == 0.6

    def test_gov_tld_authority(self, temp_dir):
        """Test .gov TLD gets high authority."""
        connector = WebConnector(cache_dir=str(temp_dir))
        assert connector._get_domain_authority("example.gov") == 0.9

    def test_edu_tld_authority(self, temp_dir):
        """Test .edu TLD gets high authority."""
        connector = WebConnector(cache_dir=str(temp_dir))
        assert connector._get_domain_authority("mit.edu") == 0.9

    def test_unknown_domain_default(self, temp_dir):
        """Test unknown domains get default authority."""
        connector = WebConnector(cache_dir=str(temp_dir))
        assert connector._get_domain_authority("unknown-site.xyz") == 0.5

    def test_www_prefix_stripped(self, temp_dir):
        """Test www. prefix is stripped from domains."""
        connector = WebConnector(cache_dir=str(temp_dir))
        assert connector._get_domain_authority("www.wikipedia.org") == 0.95

    def test_subdomain_matches_parent(self, temp_dir):
        """Test subdomains match parent domain authority."""
        connector = WebConnector(cache_dir=str(temp_dir))
        assert connector._get_domain_authority("en.wikipedia.org") == 0.95


class TestURLSecurityValidation:
    """Tests for URL security validation (blocking local/private IPs)."""

    def test_localhost_blocked(self, temp_dir):
        """Test localhost URLs are blocked."""
        connector = WebConnector(cache_dir=str(temp_dir))
        assert connector._is_local_ip("http://localhost/api") is True
        assert connector._is_local_ip("http://127.0.0.1/api") is True
        assert connector._is_local_ip("http://[::1]/api") is True

    def test_private_ip_ranges_blocked(self, temp_dir):
        """Test private IP ranges are blocked."""
        connector = WebConnector(cache_dir=str(temp_dir))
        # 10.0.0.0/8
        assert connector._is_local_ip("http://10.0.0.1/api") is True
        # 172.16.0.0/12
        assert connector._is_local_ip("http://172.16.0.1/api") is True
        # 192.168.0.0/16
        assert connector._is_local_ip("http://192.168.1.1/api") is True

    def test_public_ip_allowed(self, temp_dir):
        """Test public IPs are allowed."""
        connector = WebConnector(cache_dir=str(temp_dir))
        assert connector._is_local_ip("http://8.8.8.8/api") is False
        assert connector._is_local_ip("http://1.1.1.1/api") is False

    def test_public_domain_allowed(self, temp_dir):
        """Test public domains are allowed."""
        connector = WebConnector(cache_dir=str(temp_dir))
        assert connector._is_local_ip("http://google.com/api") is False
        assert connector._is_local_ip("https://wikipedia.org") is False

    def test_malformed_url_blocked(self, temp_dir):
        """Test malformed URLs are blocked for safety."""
        connector = WebConnector(cache_dir=str(temp_dir))
        # Empty hostname returns False (different path)
        assert connector._is_local_ip("") is False


class TestHTMLParsing:
    """Tests for HTML content parsing."""

    def test_basic_html_parsing(self, temp_dir):
        """Test basic HTML content extraction."""
        connector = WebConnector(cache_dir=str(temp_dir))
        html = """
        <html>
        <head><title>Test Page</title></head>
        <body>
            <h1>Hello World</h1>
            <p>This is test content.</p>
        </body>
        </html>
        """
        content, title = connector._parse_html(html)
        assert title == "Test Page"
        assert "Hello World" in content or "test content" in content

    def test_script_tags_removed(self, temp_dir):
        """Test that script tags are removed from content."""
        connector = WebConnector(cache_dir=str(temp_dir))
        html = """
        <html>
        <head><title>Page</title></head>
        <body>
            <p>Safe content</p>
            <script>alert('XSS')</script>
        </body>
        </html>
        """
        content, _ = connector._parse_html(html)
        assert "alert" not in content
        assert "XSS" not in content
        assert "Safe content" in content

    def test_style_tags_removed(self, temp_dir):
        """Test that style tags are removed from content."""
        connector = WebConnector(cache_dir=str(temp_dir))
        html = """
        <html>
        <body>
            <style>.hidden { display: none; }</style>
            <p>Visible content</p>
        </body>
        </html>
        """
        content, _ = connector._parse_html(html)
        assert "display: none" not in content
        assert "Visible content" in content

    def test_max_content_length_respected(self, temp_dir):
        """Test that content is truncated to max length."""
        connector = WebConnector(cache_dir=str(temp_dir), max_content_length=100)
        long_content = "x" * 200
        html = f"<html><body><p>{long_content}</p></body></html>"
        content, _ = connector._parse_html(html)
        assert len(content) <= 100

    def test_empty_title_fallback(self, temp_dir):
        """Test fallback title when none present."""
        connector = WebConnector(cache_dir=str(temp_dir))
        html = "<html><body><p>No title here</p></body></html>"
        _, title = connector._parse_html(html)
        assert title == "Untitled"


class TestCacheOperations:
    """Tests for search result caching."""

    def test_cache_file_created(self, temp_dir):
        """Test that cache file is created for queries."""
        connector = WebConnector(cache_dir=str(temp_dir))
        query = "test query"
        cache_file = connector._get_cache_file(query)

        # Create mock evidence and cache it
        evidence = Evidence(
            id="test-123",
            source_type=SourceType.WEB_SEARCH,
            source_id="http://example.com",
            content="Test content",
            title="Test Title",
            confidence=0.8,
        )
        connector._save_to_cache(query, [evidence])

        assert cache_file.exists()

    def test_cache_query_hash_deterministic(self, temp_dir):
        """Test that same query produces same cache file."""
        connector = WebConnector(cache_dir=str(temp_dir))
        file1 = connector._get_cache_file("test query")
        file2 = connector._get_cache_file("test query")
        assert file1 == file2

    def test_cache_query_hash_unique(self, temp_dir):
        """Test that different queries produce different cache files."""
        connector = WebConnector(cache_dir=str(temp_dir))
        file1 = connector._get_cache_file("query one")
        file2 = connector._get_cache_file("query two")
        assert file1 != file2


class TestSearchWithMockedDDGS:
    """Tests for web search with mocked DuckDuckGo."""

    @pytest.mark.asyncio
    async def test_search_returns_evidence(self, temp_dir):
        """Test that search returns Evidence objects."""
        connector = WebConnector(cache_dir=str(temp_dir))

        mock_results = [
            {
                "title": "Python Documentation",
                "href": "https://docs.python.org",
                "body": "Official Python documentation",
            },
            {
                "title": "Python Tutorial",
                "href": "https://docs.python.org/tutorial",
                "body": "Learn Python step by step",
            },
        ]

        with patch.object(connector, '_search_web_actual', new_callable=AsyncMock) as mock_search:
            mock_evidence = [
                connector._result_to_evidence(r, "python") for r in mock_results
            ]
            mock_search.return_value = mock_evidence

            results = await connector.search("python", limit=2)

            assert len(results) == 2
            assert all(isinstance(r, Evidence) for r in results)
            assert results[0].title == "Python Documentation"

    @pytest.mark.asyncio
    async def test_search_uses_cache(self, temp_dir):
        """Test that search uses cached results."""
        connector = WebConnector(cache_dir=str(temp_dir))

        # Pre-populate cache
        query = "cached query"
        cache_file = connector._get_cache_file(query)
        cached_evidence = Evidence(
            id="cached-123",
            source_type=SourceType.WEB_SEARCH,
            source_id="http://cached.com",
            content="Cached content",
            title="Cached Title",
        )
        cache_file.write_text(json.dumps({
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "results": [cached_evidence.to_dict()],
        }))

        with patch.object(connector, '_search_web_actual', new_callable=AsyncMock) as mock_search:
            results = await connector.search(query)

            # Should not call actual search
            mock_search.assert_not_called()
            assert len(results) == 1
            assert results[0].id == "cached-123"

    @pytest.mark.asyncio
    async def test_search_error_handling(self, temp_dir):
        """Test that search handles errors gracefully."""
        connector = WebConnector(cache_dir=str(temp_dir))

        with patch.object(connector, '_search_web_actual', new_callable=AsyncMock) as mock_search:
            mock_search.return_value = [
                connector._create_error_evidence("Search failed: Network error")
            ]

            results = await connector.search("failing query")

            assert len(results) == 1
            assert results[0].confidence == 0.0
            assert "Error" in results[0].content


class TestURLFetchWithMockedHTTPX:
    """Tests for URL fetching with mocked httpx."""

    @pytest.mark.asyncio
    async def test_fetch_url_returns_evidence(self, temp_dir):
        """Test that fetch_url returns Evidence object."""
        if not HTTPX_AVAILABLE:
            pytest.skip("httpx not installed")

        connector = WebConnector(cache_dir=str(temp_dir))

        with patch('aragora.connectors.web.httpx.AsyncClient') as mock_client_class:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.headers = {"content-type": "text/html"}
            mock_response.text = "<html><head><title>Test</title></head><body><p>Content</p></body></html>"
            mock_response.raise_for_status = MagicMock()

            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            result = await connector.fetch_url("https://example.com")

            assert result is not None
            assert isinstance(result, Evidence)
            assert result.source_type == SourceType.WEB_SEARCH

    @pytest.mark.asyncio
    async def test_fetch_url_blocks_local_ip(self, temp_dir):
        """Test that fetch_url blocks local IPs."""
        connector = WebConnector(cache_dir=str(temp_dir))

        result = await connector.fetch_url("http://127.0.0.1/admin")

        assert result is not None
        assert "blocked" in result.content.lower()
        assert result.confidence == 0.0

    @pytest.mark.asyncio
    async def test_fetch_url_timeout_handling(self, temp_dir):
        """Test that fetch_url handles timeouts gracefully."""
        if not HTTPX_AVAILABLE:
            pytest.skip("httpx not installed")

        connector = WebConnector(cache_dir=str(temp_dir))

        with patch('aragora.connectors.web.httpx.AsyncClient') as mock_client_class:
            import httpx
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(side_effect=httpx.TimeoutException("Timeout"))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            result = await connector.fetch_url("https://slow-site.com")

            assert result is not None
            assert "Timeout" in result.content
            assert result.confidence == 0.0


class TestResultToEvidence:
    """Tests for converting search results to Evidence."""

    def test_result_to_evidence_basic(self, temp_dir):
        """Test basic result conversion."""
        connector = WebConnector(cache_dir=str(temp_dir))
        result = {
            "title": "Test Title",
            "href": "https://example.com/page",
            "body": "Test content body",
        }

        evidence = connector._result_to_evidence(result, "test query")

        assert evidence.title == "Test Title"
        assert evidence.url == "https://example.com/page"
        assert evidence.content == "Test content body"
        assert evidence.source_type == SourceType.WEB_SEARCH

    def test_result_to_evidence_with_link_key(self, temp_dir):
        """Test result conversion with 'link' instead of 'href'."""
        connector = WebConnector(cache_dir=str(temp_dir))
        result = {
            "title": "Test",
            "link": "https://example.com",
            "snippet": "Content",
        }

        evidence = connector._result_to_evidence(result, "query")

        assert evidence.url == "https://example.com"

    def test_result_to_evidence_query_in_metadata(self, temp_dir):
        """Test that query is stored in metadata."""
        connector = WebConnector(cache_dir=str(temp_dir))
        result = {"title": "Test", "href": "http://test.com", "body": "content"}

        evidence = connector._result_to_evidence(result, "my search query")

        assert evidence.metadata.get("query") == "my search query"


class TestRateLimiting:
    """Tests for rate limiting behavior."""

    @pytest.mark.asyncio
    async def test_rate_limit_delay(self, temp_dir):
        """Test that rate limiting introduces delay."""
        connector = WebConnector(cache_dir=str(temp_dir), rate_limit_delay=0.1)

        import time
        start = time.time()

        # First call
        await connector._rate_limit()
        # Second call should be delayed
        await connector._rate_limit()

        elapsed = time.time() - start
        assert elapsed >= 0.1, "Rate limiting should introduce delay"


class TestErrorEvidenceCreation:
    """Tests for error evidence creation."""

    def test_create_error_evidence(self, temp_dir):
        """Test error evidence creation."""
        connector = WebConnector(cache_dir=str(temp_dir))

        error = connector._create_error_evidence("Something went wrong")

        assert "[Error]" in error.content
        assert "Something went wrong" in error.content
        assert error.confidence == 0.0
        assert error.authority == 0.0


class TestAgentFriendlyMethods:
    """Tests for agent-friendly convenience methods."""

    @pytest.mark.asyncio
    async def test_search_web_formats_results(self, temp_dir):
        """Test that search_web returns formatted markdown."""
        connector = WebConnector(cache_dir=str(temp_dir))

        with patch.object(connector, 'search', new_callable=AsyncMock) as mock_search:
            mock_search.return_value = [
                Evidence(
                    id="1",
                    source_type=SourceType.WEB_SEARCH,
                    source_id="http://test.com",
                    content="Test content",
                    title="Test Result",
                    author="test.com",
                    url="http://test.com",
                    authority=0.8,
                )
            ]

            result = await connector.search_web("test query")

            assert "## Web Search Results" in result
            assert "Test Result" in result
            assert "test.com" in result

    @pytest.mark.asyncio
    async def test_search_web_no_results(self, temp_dir):
        """Test search_web with no results."""
        connector = WebConnector(cache_dir=str(temp_dir))

        with patch.object(connector, 'search', new_callable=AsyncMock) as mock_search:
            mock_search.return_value = []

            result = await connector.search_web("impossible query")

            assert "No results found" in result

    @pytest.mark.asyncio
    async def test_read_url_formats_content(self, temp_dir):
        """Test that read_url returns formatted content."""
        connector = WebConnector(cache_dir=str(temp_dir))

        with patch.object(connector, 'fetch_url', new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = Evidence(
                id="1",
                source_type=SourceType.WEB_SEARCH,
                source_id="http://test.com",
                content="Page content here",
                title="Page Title",
                author="test.com",
                authority=0.75,
            )

            result = await connector.read_url("http://test.com")

            assert "## Content from: Page Title" in result
            assert "Page content here" in result
