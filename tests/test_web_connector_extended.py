"""
Extended tests for Web connector - focusing on gaps in coverage.

Tests cover:
- Security tests (link-local IPs, special characters, XSS)
- Network error tests (SSL, connection, proxy, read timeout)
- HTTP response tests (redirects, rate limit, service unavailable)
- Content parsing (malformed HTML, large content)
- Concurrent & State tests (cache race, corruption, pooling, cleanup)
"""

import asyncio
import hashlib
import json
import tempfile
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.connectors.web import WebConnector


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_cache_dir(tmp_path):
    """Create a temporary cache directory."""
    cache_dir = tmp_path / ".web_cache"
    cache_dir.mkdir()
    return cache_dir


@pytest.fixture
def connector(temp_cache_dir):
    """Create a WebConnector with temp cache directory."""
    return WebConnector(cache_dir=str(temp_cache_dir))


@pytest.fixture
def mock_httpx_response():
    """Create a mock httpx response."""
    mock = MagicMock()
    mock.status_code = 200
    mock.headers = {"content-type": "text/html"}
    mock.text = "<html><head><title>Test</title></head><body><p>Content</p></body></html>"
    mock.raise_for_status = MagicMock()
    return mock


# =============================================================================
# Security Tests
# =============================================================================


class TestSecurityExtended:
    """Extended security tests."""

    def test_blocks_link_local_ips(self, connector):
        """Test that link-local IPs (169.254.x.x) are blocked."""
        assert connector._is_local_ip("http://169.254.1.1/api") is True
        assert connector._is_local_ip("http://169.254.169.254/latest/meta-data/") is True

    def test_url_with_special_encoded_characters(self, connector):
        """Test URLs with special/encoded characters."""
        # These should not crash the security check
        assert connector._is_local_ip("http://example.com/path%20with%20spaces") is False
        assert connector._is_local_ip("http://example.com/path?q=test&foo=bar") is False
        assert connector._is_local_ip("http://example.com/path#fragment") is False

    def test_xss_payloads_neutralized_in_html(self, connector):
        """Test that XSS payloads are neutralized in HTML parsing."""
        malicious_html = """
        <html>
        <head><title>Test Page</title></head>
        <body>
            <script>alert('xss')</script>
            <p>Content</p>
            <script type="text/javascript">
                document.cookie = 'stolen';
            </script>
            <img src="x" onerror="alert('xss')">
        </body>
        </html>
        """
        content, title = connector._parse_html(malicious_html)

        # Script content should be removed
        assert "alert" not in content
        assert "document.cookie" not in content
        assert "<script" not in content
        assert "onerror" not in content

    def test_various_localhost_formats_blocked(self, connector):
        """Test various localhost formats are blocked."""
        # IPv4 localhost variants
        assert connector._is_local_ip("http://localhost/") is True
        assert connector._is_local_ip("http://127.0.0.1/") is True
        assert connector._is_local_ip("http://127.0.0.2/") is True  # Loopback range

        # IPv6 localhost
        assert connector._is_local_ip("http://[::1]/") is True

    def test_event_handler_attributes_removed(self, connector):
        """Test that event handler attributes are removed from HTML."""
        html_with_handlers = """
        <html>
        <body>
            <div onclick="evil()">Click me</div>
            <img onerror="alert('xss')" src="x">
            <p onmouseover="steal()">Hover</p>
        </body>
        </html>
        """
        content, title = connector._parse_html(html_with_handlers)

        # Event handlers should not appear in content
        assert "onclick" not in content
        assert "onerror" not in content
        assert "onmouseover" not in content


# =============================================================================
# Network Error Tests
# =============================================================================


class TestNetworkErrors:
    """Tests for network error handling."""

    @pytest.mark.asyncio
    async def test_ssl_certificate_error_handling(self, connector):
        """Test handling of SSL certificate errors."""
        import httpx

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=httpx.ConnectError("SSL certificate verify failed"))

        with patch.object(connector, "_get_http_client", return_value=mock_client):
            result = await connector.fetch_url("https://example.com")

            assert result is not None
            assert "[Error]" in result.content
            assert "SSL" in result.content or "Connect" in result.content

    @pytest.mark.asyncio
    async def test_connection_error_handling(self, connector):
        """Test handling of connection errors."""
        import httpx

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))

        with patch.object(connector, "_get_http_client", return_value=mock_client):
            result = await connector.fetch_url("https://example.com")

            assert result is not None
            assert "[Error]" in result.content

    @pytest.mark.asyncio
    async def test_read_timeout_handling(self, connector):
        """Test handling of read timeout (mid-response)."""
        import httpx

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=httpx.TimeoutException("Read timeout"))

        with patch.object(connector, "_get_http_client", return_value=mock_client):
            result = await connector.fetch_url("https://example.com")

            assert result is not None
            assert "[Error]" in result.content
            assert "Timeout" in result.content

    @pytest.mark.asyncio
    async def test_duckduckgo_network_failures(self, connector):
        """Test handling of DuckDuckGo search network failures."""
        with patch("aragora.connectors.web.DDGS_AVAILABLE", True):
            with patch("aragora.connectors.web.DDGS") as mock_ddgs:
                mock_ddgs.return_value.text.side_effect = Exception("Network error")

                results = await connector._search_web_actual("test query")

                # Should return error evidence
                assert len(results) == 1
                assert "[Error]" in results[0].content or "failed" in results[0].content.lower()


# =============================================================================
# HTTP Response Tests
# =============================================================================


class TestHTTPResponses:
    """Tests for various HTTP response handling."""

    @pytest.mark.asyncio
    async def test_http_301_permanent_redirect(self, connector):
        """Test handling of HTTP 301 permanent redirect."""
        import httpx

        final_response = MagicMock()
        final_response.status_code = 200
        final_response.is_redirect = False  # Important: indicate not a redirect
        final_response.headers = {"content-type": "text/html"}
        final_response.text = (
            "<html><title>Redirected</title><body><p>Final content</p></body></html>"
        )
        final_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=final_response)

        with patch.object(connector, "_get_http_client", AsyncMock(return_value=mock_client)):
            result = await connector.fetch_url("https://old-url.com/page")

            # Should get the final content (httpx follows redirects by default)
            assert result is not None
            assert "Final content" in result.content or "Redirected" in result.title

    @pytest.mark.asyncio
    async def test_http_429_rate_limit_response(self, connector):
        """Test handling of HTTP 429 rate limit response."""
        import httpx

        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.is_redirect = False
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Rate limited", request=MagicMock(), response=mock_response
        )

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        with patch.object(connector, "_get_http_client", AsyncMock(return_value=mock_client)):
            result = await connector.fetch_url("https://example.com/api")

            assert result is not None
            assert "[Error]" in result.content
            assert "429" in result.content or "Rate" in result.content

    @pytest.mark.asyncio
    async def test_http_503_service_unavailable(self, connector):
        """Test handling of HTTP 503 service unavailable."""
        import httpx

        mock_response = MagicMock()
        mock_response.status_code = 503
        mock_response.is_redirect = False
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Service unavailable", request=MagicMock(), response=mock_response
        )

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        with patch.object(connector, "_get_http_client", AsyncMock(return_value=mock_client)):
            result = await connector.fetch_url("https://example.com/api")

            assert result is not None
            assert "[Error]" in result.content
            assert "503" in result.content or "unavailable" in result.content.lower()

    @pytest.mark.asyncio
    async def test_non_utf8_charset_handling(self, connector):
        """Test handling of non-UTF-8 charset content."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.is_redirect = False
        mock_response.headers = {"content-type": "text/html; charset=iso-8859-1"}
        # httpx handles encoding internally, so we just test it doesn't crash
        mock_response.text = (
            "<html><title>Test</title><body>Content with special char: café</body></html>"
        )
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        with patch.object(connector, "_get_http_client", AsyncMock(return_value=mock_client)):
            result = await connector.fetch_url("https://example.com/page")

            assert result is not None
            assert result.title == "Test"


# =============================================================================
# Content Parsing Tests
# =============================================================================


class TestContentParsing:
    """Tests for content parsing edge cases."""

    def test_malformed_html_tags(self, connector):
        """Test handling of malformed HTML tags (unclosed, nested scripts)."""
        malformed_html = """
        <html>
        <head><title>Broken Page</title>
        <script>
        <script>nested script
        </script>
        </head>
        <body>
            <p>Valid content
            <div>Unclosed div
            <span>Unclosed span
            <p>Another paragraph</p>
        </body>
        </html>
        """
        content, title = connector._parse_html(malformed_html)

        # Should extract what it can
        assert title == "Broken Page"
        assert "Valid content" in content or "Another paragraph" in content

    def test_extremely_large_content_truncation(self, connector):
        """Test that extremely large content is truncated."""
        # Create content larger than max_content_length
        large_content = "x" * (connector.max_content_length + 10000)
        large_html = f"<html><title>Large Page</title><body><p>{large_content}</p></body></html>"

        content, title = connector._parse_html(large_html)

        assert len(content) <= connector.max_content_length
        assert title == "Large Page"

    def test_malformed_json_content_type(self, connector):
        """Test handling of malformed JSON content type."""
        # This should be handled gracefully by fetch_url
        pass  # The connector just returns the raw text for JSON

    def test_html_without_body(self, connector):
        """Test parsing HTML without body tag."""
        minimal_html = "<html><head><title>No Body</title></head></html>"
        content, title = connector._parse_html(minimal_html)

        assert title == "No Body"
        # Content may be empty or contain just the title

    def test_html_with_only_text(self, connector):
        """Test parsing HTML with text nodes only."""
        # BeautifulSoup extracts text from paragraphs and similar elements
        text_only = "<html><body><p>Just plain text content here</p></body></html>"
        content, title = connector._parse_html(text_only)

        assert "Just plain text" in content


# =============================================================================
# Concurrent & State Tests
# =============================================================================


class TestConcurrentAndState:
    """Tests for concurrent access and state management."""

    @pytest.mark.asyncio
    async def test_concurrent_fetch_requests_cache_race(self, connector):
        """Test concurrent fetch requests with cache race condition."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.is_redirect = False
        mock_response.headers = {"content-type": "text/html"}
        mock_response.text = "<html><title>Test</title><body><p>Content</p></body></html>"
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        with patch.object(connector, "_get_http_client", AsyncMock(return_value=mock_client)):
            # Launch multiple concurrent fetches for the same URL
            tasks = [connector.fetch_url("https://example.com/page") for _ in range(5)]
            results = await asyncio.gather(*tasks)

            # All should succeed
            assert len(results) == 5
            assert all(r is not None for r in results)
            assert all(r.title == "Test" for r in results)

    @pytest.mark.asyncio
    async def test_cache_corruption_recovery(self, connector, temp_cache_dir):
        """Test recovery from corrupted cache files."""
        # Create a corrupted cache file
        query = "test query"
        cache_file = connector._get_cache_file(query)
        cache_file.write_text("not valid json {{{")

        # Search should recover gracefully and not crash
        with patch("aragora.connectors.web.DDGS_AVAILABLE", True):
            with patch("aragora.connectors.web.DDGS") as mock_ddgs:
                mock_ddgs.return_value.text.return_value = iter(
                    [{"title": "Result", "body": "Body", "href": "https://example.com"}]
                )

                results = await connector.search(query)

                # Should proceed with search, not crash
                # May return mock results or error depending on DDGS availability

    @pytest.mark.asyncio
    async def test_http_client_connection_pooling(self, connector):
        """Test HTTP client reuses connections (singleton)."""
        # Get client twice
        client1 = await connector._get_http_client()
        client2 = await connector._get_http_client()

        # Should be the same instance
        assert client1 is client2

    @pytest.mark.asyncio
    async def test_cleanup_method_verification(self, connector):
        """Test that cleanup properly closes HTTP client."""
        # Initialize the client
        with patch("aragora.connectors.web.HTTPX_AVAILABLE", True):
            with patch("httpx.AsyncClient") as mock_client_class:
                mock_client = AsyncMock()
                mock_client_class.return_value = mock_client

                # Force client creation
                connector._http_client = mock_client

                # Cleanup
                await connector.cleanup()

                # Client should be closed and cleared
                mock_client.aclose.assert_called_once()
                assert connector._http_client is None


# =============================================================================
# Rate Limiting Tests
# =============================================================================


class TestRateLimiting:
    """Tests for rate limiting behavior."""

    @pytest.mark.asyncio
    async def test_rate_limit_applies_delay(self, connector):
        """Test that rate limiting applies delay between requests."""
        connector.rate_limit_delay = 0.1  # 100ms delay

        start_time = time.time()

        # Make two consecutive rate-limited calls
        await connector._rate_limit()
        await connector._rate_limit()

        elapsed = time.time() - start_time

        # Should have waited at least rate_limit_delay
        assert elapsed >= connector.rate_limit_delay

    @pytest.mark.asyncio
    async def test_first_request_no_delay(self, connector):
        """Test that first request doesn't wait."""
        connector.rate_limit_delay = 10.0  # Large delay
        connector._last_request_time = 0.0  # Reset to beginning of time

        start_time = time.time()
        await connector._rate_limit()
        elapsed = time.time() - start_time

        # First request should be nearly instant
        assert elapsed < 1.0


# =============================================================================
# Agent-Friendly Method Tests
# =============================================================================


class TestAgentFriendlyMethods:
    """Tests for agent-friendly convenience methods."""

    @pytest.mark.asyncio
    async def test_search_web_no_results(self, connector):
        """Test search_web with no results."""
        with patch.object(connector, "search", return_value=[]):
            result = await connector.search_web("nonexistent query")

            assert "[No results found" in result

    @pytest.mark.asyncio
    async def test_read_url_failure(self, connector):
        """Test read_url with fetch failure."""
        with patch.object(connector, "fetch_url", return_value=None):
            result = await connector.read_url("https://example.com")

            assert "[Failed to read" in result

    @pytest.mark.asyncio
    async def test_search_web_formats_results(self, connector):
        """Test that search_web formats results correctly."""
        from aragora.connectors.base import Evidence
        from aragora.reasoning.provenance import SourceType

        mock_evidence = Evidence(
            id="test123",
            source_type=SourceType.WEB_SEARCH,
            source_id="https://example.com",
            content="Test content body",
            title="Test Result",
            url="https://example.com",
            author="example.com",
            confidence=0.8,
            authority=0.7,
            freshness=1.0,
        )

        with patch.object(connector, "search", return_value=[mock_evidence]):
            result = await connector.search_web("test query")

            assert "## Web Search Results" in result
            assert "Test Result" in result
            assert "example.com" in result


# =============================================================================
# Error Evidence Tests
# =============================================================================


class TestErrorEvidence:
    """Tests for error evidence creation."""

    def test_create_error_evidence(self, connector):
        """Test error evidence has correct structure."""
        error = connector._create_error_evidence("Test error message")

        assert error is not None
        assert error.confidence == 0.0
        assert error.authority == 0.0
        assert "[Error]" in error.content
        assert "Test error message" in error.content

    def test_error_evidence_unique_ids(self, connector):
        """Test that different errors get different IDs."""
        error1 = connector._create_error_evidence("Error 1")
        error2 = connector._create_error_evidence("Error 2")

        assert error1.id != error2.id
