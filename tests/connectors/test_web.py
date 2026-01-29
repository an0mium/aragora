"""Tests for WebConnector - live web access for aragora agents."""

import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.connectors.base import Evidence
from aragora.connectors.web import DOMAIN_AUTHORITY, WebConnector
from aragora.reasoning.provenance import SourceType
from aragora.resilience import CircuitBreaker


# Sample DuckDuckGo search results
SAMPLE_DDGS_RESULTS = [
    {
        "title": "Python 3.12 Release Notes",
        "href": "https://docs.python.org/3/whatsnew/3.12.html",
        "body": "Python 3.12 introduces new syntax features and performance improvements.",
    },
    {
        "title": "What's New in Python 3.12",
        "href": "https://realpython.com/python312-new-features/",
        "body": "Learn about the exciting new features in Python 3.12 including pattern matching.",
    },
]

# Sample HTML content
SAMPLE_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Test Page Title</title>
</head>
<body>
    <header>Navigation header</header>
    <main>
        <article>
            <h1>Main Article Heading</h1>
            <p>This is the main content of the page with important information.</p>
            <p>Another paragraph with more details about the topic.</p>
        </article>
    </main>
    <footer>Footer content</footer>
</body>
</html>
"""

SAMPLE_HTML_NO_MAIN = """
<!DOCTYPE html>
<html>
<head><title>Simple Page</title></head>
<body>
    <h1>Simple heading</h1>
    <p>Body content without main/article tags.</p>
</body>
</html>
"""


class TestWebConnectorProperties:
    """Tests for WebConnector property methods."""

    @pytest.fixture
    def connector(self, tmp_path):
        """Create a WebConnector for testing with temp cache directory."""
        return WebConnector(
            rate_limit_delay=0.0,  # Disable rate limiting for tests
            cache_dir=str(tmp_path / ".web_cache"),
            enable_circuit_breaker=False,
        )

    def test_source_type(self, connector):
        """Test source_type property returns WEB_SEARCH."""
        assert connector.source_type == SourceType.WEB_SEARCH

    def test_name(self, connector):
        """Test name property returns 'Web Search'."""
        assert connector.name == "Web Search"

    def test_initialization_defaults(self, tmp_path):
        """Test default initialization values."""
        connector = WebConnector(
            cache_dir=str(tmp_path / ".web_cache"),
            enable_circuit_breaker=False,
        )
        assert connector.timeout == 30
        assert connector.max_content_length == 10000
        assert connector.rate_limit_delay == 1.0
        assert connector.default_confidence == 0.6

    def test_initialization_custom_values(self, tmp_path):
        """Test custom initialization values."""
        connector = WebConnector(
            default_confidence=0.8,
            timeout=60,
            max_content_length=5000,
            rate_limit_delay=2.0,
            cache_dir=str(tmp_path / ".web_cache"),
            enable_circuit_breaker=False,
        )
        assert connector.timeout == 60
        assert connector.max_content_length == 5000
        assert connector.rate_limit_delay == 2.0
        assert connector.default_confidence == 0.8

    def test_cache_directory_created(self, tmp_path):
        """Test that cache directory is created on initialization."""
        cache_path = tmp_path / ".test_web_cache"
        assert not cache_path.exists()

        WebConnector(
            cache_dir=str(cache_path),
            enable_circuit_breaker=False,
        )
        assert cache_path.exists()

    def test_circuit_breaker_initialization(self, tmp_path):
        """Test circuit breaker initialization options."""
        # With circuit breaker enabled (default)
        connector = WebConnector(
            cache_dir=str(tmp_path / ".cache1"),
            enable_circuit_breaker=True,
        )
        assert connector._circuit_breaker is not None

        # With circuit breaker disabled
        connector = WebConnector(
            cache_dir=str(tmp_path / ".cache2"),
            enable_circuit_breaker=False,
        )
        assert connector._circuit_breaker is None

        # With custom circuit breaker
        custom_cb = CircuitBreaker(failure_threshold=10, cooldown_seconds=60.0)
        connector = WebConnector(
            cache_dir=str(tmp_path / ".cache3"),
            circuit_breaker=custom_cb,
        )
        assert connector._circuit_breaker is custom_cb


class TestDomainAuthority:
    """Tests for domain authority scoring."""

    @pytest.fixture
    def connector(self, tmp_path):
        """Create a WebConnector for testing."""
        return WebConnector(
            rate_limit_delay=0.0,
            cache_dir=str(tmp_path / ".web_cache"),
            enable_circuit_breaker=False,
        )

    def test_high_authority_domains(self, connector):
        """Test high authority domain scores."""
        assert connector._get_domain_authority("wikipedia.org") == 0.95
        assert connector._get_domain_authority("arxiv.org") == 0.9
        assert connector._get_domain_authority("nature.com") == 0.95

    def test_tld_authority(self, connector):
        """Test TLD-based authority scores."""
        assert connector._get_domain_authority("mit.edu") == 0.9
        assert connector._get_domain_authority("nasa.gov") == 0.9

    def test_medium_authority_domains(self, connector):
        """Test medium authority domain scores."""
        assert connector._get_domain_authority("stackoverflow.com") == 0.85
        assert connector._get_domain_authority("medium.com") == 0.6
        assert connector._get_domain_authority("reddit.com") == 0.5

    def test_www_prefix_stripped(self, connector):
        """Test that www prefix is stripped from domains."""
        assert connector._get_domain_authority("www.wikipedia.org") == 0.95
        assert connector._get_domain_authority("www.github.com") == 0.85

    def test_unknown_domain_default(self, connector):
        """Test unknown domains get default authority score."""
        assert connector._get_domain_authority("unknown-site.xyz") == 0.5

    def test_suffix_matching(self, connector):
        """Test authority scoring for domain suffixes."""
        # docs.python.org is in the authority map
        assert connector._get_domain_authority("docs.python.org") == 0.9


class TestSearch:
    """Tests for search functionality."""

    @pytest.fixture
    def connector(self, tmp_path):
        """Create a WebConnector for testing."""
        return WebConnector(
            rate_limit_delay=0.0,
            cache_dir=str(tmp_path / ".web_cache"),
            enable_circuit_breaker=False,
        )

    @pytest.mark.asyncio
    async def test_search_mocked(self, connector):
        """Test search with mocked DDGS results."""
        with patch.object(connector, "_search_web_actual") as mock_search:
            # Create mock Evidence objects
            mock_evidence = [
                Evidence(
                    id="test-1",
                    source_type=SourceType.WEB_SEARCH,
                    source_id="https://docs.python.org/3/whatsnew/3.12.html",
                    content="Python 3.12 introduces new syntax features.",
                    title="Python 3.12 Release Notes",
                    url="https://docs.python.org/3/whatsnew/3.12.html",
                    confidence=0.6,
                    authority=0.9,
                ),
            ]
            mock_search.return_value = mock_evidence

            results = await connector.search("Python 3.12 features", limit=5)

            assert len(results) == 1
            assert all(isinstance(r, Evidence) for r in results)
            assert results[0].title == "Python 3.12 Release Notes"
            mock_search.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_cached_results(self, connector):
        """Test that cached results are returned without calling actual search."""
        # Pre-populate the cache file
        query = "cached search query"
        cache_file = connector._get_cache_file(query)

        cached_evidence = Evidence(
            id="cached-1",
            source_type=SourceType.WEB_SEARCH,
            source_id="https://example.com/cached",
            content="Cached content",
            title="Cached Title",
            url="https://example.com/cached",
        )

        cache_data = {
            "query": query,
            "timestamp": "2024-01-15T10:00:00",
            "results": [cached_evidence.to_dict()],
        }
        cache_file.write_text(json.dumps(cache_data))

        # Search should return cached results
        with patch.object(connector, "_search_web_actual") as mock_search:
            results = await connector.search(query)

            # Should not call actual search
            mock_search.assert_not_called()
            assert len(results) == 1
            assert results[0].title == "Cached Title"

    @pytest.mark.asyncio
    async def test_search_ddgs_unavailable(self, connector):
        """Test search when DDGS is not available."""
        with patch("aragora.connectors.web.DDGS_AVAILABLE", False):
            results = await connector._search_web_actual("test query")

            assert len(results) == 1
            assert "[Error]" in results[0].content
            assert "duckduckgo-search not installed" in results[0].content

    @pytest.mark.asyncio
    async def test_search_timeout_handling(self, connector):
        """Test handling of search timeout."""
        import aragora.connectors.web as web_module

        # Create a mock DDGS class that blocks
        mock_ddgs = MagicMock()
        mock_ddgs.text.side_effect = lambda *args, **kwargs: [
            item for item in iter(lambda: asyncio.sleep(100), None)
        ]
        mock_ddgs_class = MagicMock(return_value=mock_ddgs)

        # Use patch.dict to inject DDGS into the module namespace
        with patch.dict(web_module.__dict__, {"DDGS": mock_ddgs_class, "DDGS_AVAILABLE": True}):
            # Patch the timeout to be very short
            with patch.object(web_module, "DB_TIMEOUT_SECONDS", 0.01):
                results = await connector._search_web_actual("test query")

                assert len(results) == 1
                assert "[Error]" in results[0].content
                assert "timed out" in results[0].content.lower()

    @pytest.mark.asyncio
    async def test_search_network_error_handling(self, connector):
        """Test handling of network errors during search."""
        import aragora.connectors.web as web_module

        mock_ddgs = MagicMock()
        mock_ddgs.text.side_effect = ConnectionError("Network unreachable")
        mock_ddgs_class = MagicMock(return_value=mock_ddgs)

        with patch.dict(web_module.__dict__, {"DDGS": mock_ddgs_class, "DDGS_AVAILABLE": True}):
            results = await connector._search_web_actual("test query")

            assert len(results) == 1
            assert "[Error]" in results[0].content
            assert "Network error" in results[0].content

    @pytest.mark.asyncio
    async def test_search_runtime_error_handling(self, connector):
        """Test handling of runtime errors from DDGS library."""
        import aragora.connectors.web as web_module

        mock_ddgs = MagicMock()
        mock_ddgs.text.side_effect = RuntimeError("DDGS service error")
        mock_ddgs_class = MagicMock(return_value=mock_ddgs)

        with patch.dict(web_module.__dict__, {"DDGS": mock_ddgs_class, "DDGS_AVAILABLE": True}):
            results = await connector._search_web_actual("test query")

            assert len(results) == 1
            assert "[Error]" in results[0].content
            assert "Search service error" in results[0].content


class TestFetchUrl:
    """Tests for URL fetching functionality."""

    @pytest.fixture
    def connector(self, tmp_path):
        """Create a WebConnector for testing."""
        return WebConnector(
            rate_limit_delay=0.0,
            cache_dir=str(tmp_path / ".web_cache"),
            enable_circuit_breaker=False,
        )

    @pytest.mark.asyncio
    async def test_fetch_url_success(self, connector):
        """Test successful URL fetch."""
        import httpx

        mock_response = MagicMock()
        mock_response.text = SAMPLE_HTML
        mock_response.headers = {"content-type": "text/html"}
        mock_response.is_redirect = False
        mock_response.raise_for_status = MagicMock()

        with patch.object(connector, "_get_http_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            with patch.object(connector, "_resolve_and_validate_ip", return_value=(True, "")):
                result = await connector.fetch_url("https://example.com/page")

                assert result is not None
                assert result.title == "Test Page Title"
                assert "Main Article Heading" in result.content
                assert result.source_type == SourceType.WEB_SEARCH

    @pytest.mark.asyncio
    async def test_fetch_url_json_content(self, connector):
        """Test fetching JSON content."""
        import httpx

        mock_response = MagicMock()
        mock_response.text = '{"key": "value", "data": [1, 2, 3]}'
        mock_response.headers = {"content-type": "application/json"}
        mock_response.is_redirect = False
        mock_response.raise_for_status = MagicMock()

        with patch.object(connector, "_get_http_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            with patch.object(connector, "_resolve_and_validate_ip", return_value=(True, "")):
                result = await connector.fetch_url("https://api.example.com/data.json")

                assert result is not None
                assert result.title == "JSON Response"
                assert '"key": "value"' in result.content

    @pytest.mark.asyncio
    async def test_fetch_url_text_content(self, connector):
        """Test fetching plain text content."""
        mock_response = MagicMock()
        mock_response.text = "Plain text content here"
        mock_response.headers = {"content-type": "text/plain"}
        mock_response.is_redirect = False
        mock_response.raise_for_status = MagicMock()

        with patch.object(connector, "_get_http_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            with patch.object(connector, "_resolve_and_validate_ip", return_value=(True, "")):
                result = await connector.fetch_url("https://example.com/file.txt")

                assert result is not None
                assert result.title == "Text Content"
                assert result.content == "Plain text content here"

    @pytest.mark.asyncio
    async def test_fetch_url_unsupported_content_type(self, connector):
        """Test handling of unsupported content types."""
        mock_response = MagicMock()
        mock_response.headers = {"content-type": "application/octet-stream"}
        mock_response.is_redirect = False
        mock_response.raise_for_status = MagicMock()

        with patch.object(connector, "_get_http_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            with patch.object(connector, "_resolve_and_validate_ip", return_value=(True, "")):
                result = await connector.fetch_url("https://example.com/file.bin")

                assert result is not None
                assert "[Error]" in result.content
                assert "Unsupported content type" in result.content

    @pytest.mark.asyncio
    async def test_fetch_url_httpx_unavailable(self, connector):
        """Test fetch when httpx is not available."""
        with patch("aragora.connectors.web.HTTPX_AVAILABLE", False):
            result = await connector.fetch_url("https://example.com")

            assert result is not None
            assert "[Error]" in result.content
            assert "httpx not installed" in result.content

    @pytest.mark.asyncio
    async def test_fetch_url_local_ip_blocked(self, connector):
        """Test that local/private IPs are blocked."""
        result = await connector.fetch_url("http://127.0.0.1/admin")

        assert result is not None
        assert "[Error]" in result.content
        assert "local/private IPs blocked" in result.content

    @pytest.mark.asyncio
    async def test_fetch_url_ssrf_protection(self, connector):
        """Test SSRF protection blocks private IP resolution."""
        with patch.object(
            connector,
            "_resolve_and_validate_ip",
            return_value=(False, "Resolved to private IP: 192.168.1.1"),
        ):
            result = await connector.fetch_url("https://evil.example.com")

            assert result is not None
            assert "[Error]" in result.content
            assert "SSRF protection" in result.content

    @pytest.mark.asyncio
    async def test_fetch_url_circuit_breaker_open(self, tmp_path):
        """Test that circuit breaker blocks requests when open."""
        # Create a circuit breaker that's already open
        cb = CircuitBreaker(failure_threshold=1, cooldown_seconds=60.0)
        cb.record_failure()  # Trip the circuit breaker

        connector = WebConnector(
            rate_limit_delay=0.0,
            cache_dir=str(tmp_path / ".web_cache"),
            circuit_breaker=cb,
        )

        result = await connector.fetch_url("https://example.com")

        assert result is not None
        assert "[Error]" in result.content
        assert "Circuit breaker open" in result.content

    @pytest.mark.asyncio
    async def test_fetch_url_redirect_handling(self, connector):
        """Test that redirects are followed safely."""
        import httpx

        # First response is a redirect
        redirect_response = MagicMock()
        redirect_response.is_redirect = True
        redirect_response.headers = {"location": "https://example.com/final-page"}

        # Second response is the final content
        final_response = MagicMock()
        final_response.is_redirect = False
        final_response.text = SAMPLE_HTML
        final_response.headers = {"content-type": "text/html"}
        final_response.raise_for_status = MagicMock()

        with patch.object(connector, "_get_http_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(side_effect=[redirect_response, final_response])
            mock_get_client.return_value = mock_client

            with patch.object(connector, "_resolve_and_validate_ip", return_value=(True, "")):
                with patch.object(
                    connector, "_validate_redirect_target", return_value=(True, "")
                ):
                    result = await connector.fetch_url("https://example.com/old-page")

                    assert result is not None
                    assert result.title == "Test Page Title"

    @pytest.mark.asyncio
    async def test_fetch_url_too_many_redirects(self, connector):
        """Test that excessive redirects are blocked."""
        import httpx

        # Always return a redirect
        redirect_response = MagicMock()
        redirect_response.is_redirect = True
        redirect_response.headers = {"location": "https://example.com/redirect"}

        with patch.object(connector, "_get_http_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=redirect_response)
            mock_get_client.return_value = mock_client

            with patch.object(connector, "_resolve_and_validate_ip", return_value=(True, "")):
                with patch.object(
                    connector, "_validate_redirect_target", return_value=(True, "")
                ):
                    result = await connector.fetch_url(
                        "https://example.com/loop", max_redirects=3
                    )

                    assert result is not None
                    assert "[Error]" in result.content
                    assert "Too many redirects" in result.content

    @pytest.mark.asyncio
    async def test_fetch_url_timeout_with_retry(self, connector):
        """Test timeout handling with retries."""
        import httpx

        with patch.object(connector, "_get_http_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(side_effect=httpx.TimeoutException("Timeout"))
            mock_get_client.return_value = mock_client

            with patch.object(connector, "_resolve_and_validate_ip", return_value=(True, "")):
                result = await connector.fetch_url(
                    "https://example.com", max_retries=2, base_delay=0.01
                )

                assert result is not None
                assert "[Error]" in result.content
                assert "Failed after" in result.content
                # Should have tried 2 times
                assert mock_client.get.call_count == 2

    @pytest.mark.asyncio
    async def test_fetch_url_http_4xx_no_retry(self, connector):
        """Test that 4xx errors are not retried."""
        import httpx

        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.is_redirect = False

        with patch.object(connector, "_get_http_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_response.raise_for_status = MagicMock(
                side_effect=httpx.HTTPStatusError(
                    "Not Found", request=MagicMock(), response=mock_response
                )
            )
            mock_get_client.return_value = mock_client

            with patch.object(connector, "_resolve_and_validate_ip", return_value=(True, "")):
                result = await connector.fetch_url(
                    "https://example.com/missing", max_retries=3
                )

                assert result is not None
                assert "[Error]" in result.content
                assert "HTTP 404" in result.content
                # Should only try once - no retries for 4xx
                assert mock_client.get.call_count == 1


class TestFetch:
    """Tests for the fetch method (evidence ID lookup)."""

    @pytest.fixture
    def connector(self, tmp_path):
        """Create a WebConnector for testing."""
        return WebConnector(
            rate_limit_delay=0.0,
            cache_dir=str(tmp_path / ".web_cache"),
            enable_circuit_breaker=False,
        )

    @pytest.mark.asyncio
    async def test_fetch_from_cache(self, connector):
        """Test fetching cached evidence by ID."""
        cached_evidence = Evidence(
            id="cached-123",
            source_type=SourceType.WEB_SEARCH,
            source_id="https://example.com/cached",
            content="This is cached content that is long enough to not trigger refetch.",
            title="Cached Page",
        )
        # Make content long enough (> 500 chars)
        cached_evidence = Evidence(
            id="cached-123",
            source_type=SourceType.WEB_SEARCH,
            source_id="https://example.com/cached",
            content="A" * 600,  # Long content
            title="Cached Page",
        )
        connector._cache_put("cached-123", cached_evidence)

        result = await connector.fetch("cached-123")

        assert result is not None
        assert result.id == "cached-123"
        assert result.title == "Cached Page"

    @pytest.mark.asyncio
    async def test_fetch_url_directly(self, connector):
        """Test fetching when evidence_id is a URL."""
        mock_response = MagicMock()
        mock_response.text = SAMPLE_HTML
        mock_response.headers = {"content-type": "text/html"}
        mock_response.is_redirect = False
        mock_response.raise_for_status = MagicMock()

        with patch.object(connector, "_get_http_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            with patch.object(connector, "_resolve_and_validate_ip", return_value=(True, "")):
                result = await connector.fetch("https://example.com/page")

                assert result is not None
                assert result.title == "Test Page Title"

    @pytest.mark.asyncio
    async def test_fetch_unknown_id(self, connector):
        """Test fetching with unknown non-URL ID returns None."""
        result = await connector.fetch("unknown-evidence-id")

        assert result is None


class TestHtmlParsing:
    """Tests for HTML parsing functionality."""

    @pytest.fixture
    def connector(self, tmp_path):
        """Create a WebConnector for testing."""
        return WebConnector(
            rate_limit_delay=0.0,
            cache_dir=str(tmp_path / ".web_cache"),
            enable_circuit_breaker=False,
        )

    def test_parse_html_with_bs4(self, connector):
        """Test HTML parsing with BeautifulSoup (or fallback if not installed)."""
        import aragora.connectors.web as web_module

        content, title = connector._parse_html(SAMPLE_HTML)

        assert title == "Test Page Title"
        assert "Main Article Heading" in content
        assert "main content of the page" in content

        # Only check header/footer exclusion when BS4 is available
        # The fallback parser doesn't strip header/footer elements
        if web_module.BS4_AVAILABLE:
            assert "Navigation header" not in content
            assert "Footer content" not in content

    def test_parse_html_without_main(self, connector):
        """Test HTML parsing when no main/article tags exist."""
        content, title = connector._parse_html(SAMPLE_HTML_NO_MAIN)

        assert title == "Simple Page"
        assert "Simple heading" in content
        assert "Body content" in content

    def test_parse_html_removes_scripts(self, connector):
        """Test that script tags are removed."""
        html_with_script = """
        <html>
        <head><title>Page</title></head>
        <body>
            <p>Safe content</p>
            <script>alert('XSS');</script>
            <p>More safe content</p>
        </body>
        </html>
        """
        content, title = connector._parse_html(html_with_script)

        assert "Safe content" in content
        assert "alert" not in content
        assert "XSS" not in content

    def test_parse_html_removes_styles(self, connector):
        """Test that style tags are removed."""
        html_with_style = """
        <html>
        <head>
            <title>Styled Page</title>
            <style>body { color: red; }</style>
        </head>
        <body>
            <p>Content here</p>
            <style>.hidden { display: none; }</style>
        </body>
        </html>
        """
        content, title = connector._parse_html(html_with_style)

        assert "Content here" in content
        assert "color:" not in content
        assert "display:" not in content

    def test_parse_html_content_length_limit(self, tmp_path):
        """Test that content is truncated at max_content_length."""
        connector = WebConnector(
            max_content_length=100,
            rate_limit_delay=0.0,
            cache_dir=str(tmp_path / ".web_cache"),
            enable_circuit_breaker=False,
        )

        long_html = f"""
        <html>
        <head><title>Long Page</title></head>
        <body><p>{"A" * 500}</p></body>
        </html>
        """
        content, title = connector._parse_html(long_html)

        assert len(content) <= 100

    def test_parse_html_without_bs4(self, connector):
        """Test HTML parsing fallback without BeautifulSoup."""
        with patch("aragora.connectors.web.BS4_AVAILABLE", False):
            content, title = connector._parse_html(SAMPLE_HTML)

            assert title == "Test Page Title"
            # Basic parsing should still extract some content
            assert len(content) > 0


class TestSecurityValidation:
    """Tests for security validation methods."""

    @pytest.fixture
    def connector(self, tmp_path):
        """Create a WebConnector for testing."""
        return WebConnector(
            rate_limit_delay=0.0,
            cache_dir=str(tmp_path / ".web_cache"),
            enable_circuit_breaker=False,
        )

    def test_is_local_ip_localhost(self, connector):
        """Test detection of localhost URLs."""
        assert connector._is_local_ip("http://localhost/admin")
        assert connector._is_local_ip("http://127.0.0.1/api")
        assert connector._is_local_ip("http://[::1]/test")

    def test_is_local_ip_private_ranges(self, connector):
        """Test detection of private IP ranges."""
        assert connector._is_local_ip("http://192.168.1.1/admin")
        assert connector._is_local_ip("http://10.0.0.1/internal")
        assert connector._is_local_ip("http://172.16.0.1/app")

    def test_is_local_ip_public(self, connector):
        """Test that public IPs are not flagged."""
        assert not connector._is_local_ip("http://8.8.8.8/dns")
        assert not connector._is_local_ip("https://example.com/page")

    def test_validate_redirect_empty_url(self, connector):
        """Test validation of empty redirect URL."""
        is_safe, error = connector._validate_redirect_target("")
        assert not is_safe
        assert "Empty redirect URL" in error

    def test_validate_redirect_invalid_scheme(self, connector):
        """Test validation rejects non-http/https schemes."""
        is_safe, error = connector._validate_redirect_target("file:///etc/passwd")
        assert not is_safe
        assert "Invalid redirect scheme" in error

        is_safe, error = connector._validate_redirect_target("ftp://example.com/file")
        assert not is_safe
        assert "Invalid redirect scheme" in error

    def test_validate_redirect_to_localhost(self, connector):
        """Test validation rejects redirects to localhost."""
        with patch.object(
            connector,
            "_resolve_and_validate_ip",
            return_value=(False, "Localhost access blocked"),
        ):
            is_safe, error = connector._validate_redirect_target(
                "https://redirect-to-local.com"
            )
            assert not is_safe


class TestAgentConvenienceMethods:
    """Tests for agent-friendly convenience methods."""

    @pytest.fixture
    def connector(self, tmp_path):
        """Create a WebConnector for testing."""
        return WebConnector(
            rate_limit_delay=0.0,
            cache_dir=str(tmp_path / ".web_cache"),
            enable_circuit_breaker=False,
        )

    @pytest.mark.asyncio
    async def test_search_web_formatted(self, connector):
        """Test search_web returns formatted string results."""
        mock_evidence = [
            Evidence(
                id="test-1",
                source_type=SourceType.WEB_SEARCH,
                source_id="https://example.com/page1",
                content="Content about the topic with details.",
                title="Result Title",
                url="https://example.com/page1",
                author="example.com",
                authority=0.8,
                confidence=0.6,
            ),
        ]

        with patch.object(connector, "search", return_value=mock_evidence):
            result = await connector.search_web("test query", limit=5)

            assert "## Web Search Results for: test query" in result
            assert "Result Title" in result
            assert "authority: 80%" in result
            assert "https://example.com/page1" in result

    @pytest.mark.asyncio
    async def test_search_web_no_results(self, connector):
        """Test search_web with no results."""
        # Return evidence with zero confidence (error evidence)
        error_evidence = Evidence(
            id="error-1",
            source_type=SourceType.WEB_SEARCH,
            source_id="error",
            content="[Error]: No results",
            title="Search Error",
            confidence=0.0,
        )

        with patch.object(connector, "search", return_value=[error_evidence]):
            result = await connector.search_web("nonexistent topic")

            assert "[No results found for:" in result

    @pytest.mark.asyncio
    async def test_search_web_empty_results(self, connector):
        """Test search_web with empty results list."""
        with patch.object(connector, "search", return_value=[]):
            result = await connector.search_web("empty query")

            assert "[No results found for:" in result

    @pytest.mark.asyncio
    async def test_read_url_formatted(self, connector):
        """Test read_url returns formatted content."""
        mock_evidence = Evidence(
            id="page-1",
            source_type=SourceType.WEB_SEARCH,
            source_id="https://docs.example.com/guide",
            content="This is the documentation content.",
            title="Documentation Guide",
            author="docs.example.com",
            authority=0.85,
            confidence=0.6,
        )

        with patch.object(connector, "fetch_url", return_value=mock_evidence):
            result = await connector.read_url("https://docs.example.com/guide")

            assert "## Content from: Documentation Guide" in result
            assert "docs.example.com" in result
            assert "85%" in result  # Authority percentage
            assert "documentation content" in result

    @pytest.mark.asyncio
    async def test_read_url_failure(self, connector):
        """Test read_url with fetch failure."""
        error_evidence = Evidence(
            id="error-1",
            source_type=SourceType.WEB_SEARCH,
            source_id="error",
            content="[Error]: Connection failed",
            title="Fetch Error",
            confidence=0.0,
        )

        with patch.object(connector, "fetch_url", return_value=error_evidence):
            result = await connector.read_url("https://unreachable.example.com")

            assert "[Failed to read:" in result


class TestResultToEvidence:
    """Tests for converting search results to Evidence objects."""

    @pytest.fixture
    def connector(self, tmp_path):
        """Create a WebConnector for testing."""
        return WebConnector(
            rate_limit_delay=0.0,
            cache_dir=str(tmp_path / ".web_cache"),
            enable_circuit_breaker=False,
        )

    def test_result_to_evidence_basic(self, connector):
        """Test basic conversion of DDGS result to Evidence."""
        result = {
            "title": "Test Article",
            "href": "https://example.com/article",
            "body": "Article body content.",
        }

        evidence = connector._result_to_evidence(result, "test query")

        assert evidence.title == "Test Article"
        assert evidence.url == "https://example.com/article"
        assert evidence.content == "Article body content."
        assert evidence.author == "example.com"
        assert evidence.source_type == SourceType.WEB_SEARCH
        assert evidence.metadata["query"] == "test query"

    def test_result_to_evidence_alternative_keys(self, connector):
        """Test conversion with alternative result keys."""
        result = {
            "title": "Alternative Article",
            "link": "https://alt.example.com/page",  # 'link' instead of 'href'
            "snippet": "Snippet text.",  # 'snippet' instead of 'body'
        }

        evidence = connector._result_to_evidence(result, "alt query")

        assert evidence.title == "Alternative Article"
        assert evidence.url == "https://alt.example.com/page"
        assert evidence.content == "Snippet text."

    def test_result_to_evidence_authority(self, connector):
        """Test that authority is set based on domain."""
        result = {
            "title": "Wikipedia Article",
            "href": "https://en.wikipedia.org/wiki/Test",
            "body": "Wikipedia content.",
        }

        evidence = connector._result_to_evidence(result, "wiki query")

        assert evidence.authority == 0.95  # Wikipedia has high authority


class TestCleanup:
    """Tests for cleanup functionality."""

    @pytest.fixture
    def connector(self, tmp_path):
        """Create a WebConnector for testing."""
        return WebConnector(
            rate_limit_delay=0.0,
            cache_dir=str(tmp_path / ".web_cache"),
            enable_circuit_breaker=False,
        )

    @pytest.mark.asyncio
    async def test_cleanup_closes_client(self, connector):
        """Test that cleanup closes the HTTP client."""
        # Create a mock client
        mock_client = AsyncMock()
        mock_client.aclose = AsyncMock()
        connector._http_client = mock_client

        await connector.cleanup()

        mock_client.aclose.assert_called_once()
        assert connector._http_client is None

    @pytest.mark.asyncio
    async def test_cleanup_handles_error(self, connector):
        """Test that cleanup handles client close errors gracefully."""
        mock_client = AsyncMock()
        mock_client.aclose = AsyncMock(side_effect=Exception("Close error"))
        connector._http_client = mock_client

        # Should not raise
        await connector.cleanup()

        assert connector._http_client is None

    @pytest.mark.asyncio
    async def test_cleanup_no_client(self, connector):
        """Test that cleanup works when no client exists."""
        assert connector._http_client is None

        # Should not raise
        await connector.cleanup()


class TestCaching:
    """Tests for caching functionality."""

    @pytest.fixture
    def connector(self, tmp_path):
        """Create a WebConnector for testing."""
        return WebConnector(
            rate_limit_delay=0.0,
            cache_dir=str(tmp_path / ".web_cache"),
            enable_circuit_breaker=False,
        )

    def test_cache_file_path(self, connector):
        """Test cache file path generation."""
        query = "test search query"
        cache_file = connector._get_cache_file(query)

        assert cache_file.parent == connector.cache_dir
        assert cache_file.suffix == ".json"
        # Same query should produce same file
        assert cache_file == connector._get_cache_file(query)

    def test_save_and_load_cache(self, connector):
        """Test saving and loading from cache."""
        query = "cached query"
        evidence_list = [
            Evidence(
                id="cache-test-1",
                source_type=SourceType.WEB_SEARCH,
                source_id="https://example.com/cached",
                content="Cached content",
                title="Cached Title",
            )
        ]

        connector._save_to_cache(query, evidence_list)

        cache_file = connector._get_cache_file(query)
        assert cache_file.exists()

        # Load and verify
        cached_data = json.loads(cache_file.read_text())
        assert cached_data["query"] == query
        assert len(cached_data["results"]) == 1
        assert cached_data["results"][0]["title"] == "Cached Title"

    @pytest.mark.asyncio
    async def test_corrupted_cache_ignored(self, connector):
        """Test that corrupted cache files are ignored."""
        query = "corrupted query"
        cache_file = connector._get_cache_file(query)
        cache_file.write_text("not valid json{{{")

        with patch.object(connector, "_search_web_actual") as mock_search:
            mock_search.return_value = []
            results = await connector.search(query)

            # Should have called actual search due to corrupted cache
            mock_search.assert_called_once()


class TestRateLimiting:
    """Tests for rate limiting functionality."""

    @pytest.mark.asyncio
    async def test_rate_limiting_delay(self, tmp_path):
        """Test that rate limiting adds delay between requests."""
        import time

        connector = WebConnector(
            rate_limit_delay=0.1,  # 100ms delay
            cache_dir=str(tmp_path / ".web_cache"),
            enable_circuit_breaker=False,
        )

        start = time.time()

        # First call should not delay (no previous request)
        await connector._rate_limit()
        first_call = time.time()

        # Second call should delay
        await connector._rate_limit()
        second_call = time.time()

        # The second call should have taken at least rate_limit_delay
        assert (second_call - first_call) >= 0.09  # Allow small margin

    @pytest.mark.asyncio
    async def test_rate_limiting_disabled(self, tmp_path):
        """Test that rate limiting can be disabled."""
        import time

        connector = WebConnector(
            rate_limit_delay=0.0,  # Disabled
            cache_dir=str(tmp_path / ".web_cache"),
            enable_circuit_breaker=False,
        )

        start = time.time()
        await connector._rate_limit()
        await connector._rate_limit()
        elapsed = time.time() - start

        # Should be nearly instant
        assert elapsed < 0.05
