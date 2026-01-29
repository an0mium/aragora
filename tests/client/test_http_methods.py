"""
Tests for AragoraClient HTTP methods.

Tests cover:
- _get, _post, _put, _patch, _delete (sync)
- _get_async, _post_async, _put_async, _patch_async, _delete_async (async)
- Retry logic with exponential backoff
- Rate limiting integration
- HTTP error mapping to specific exceptions
- Header construction
- URL construction
"""

from __future__ import annotations

import json
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.client import AragoraClient
from aragora.client.errors import (
    AragoraAPIError,
    AuthenticationError,
    NotFoundError,
    QuotaExceededError,
    RateLimitError,
    ValidationError,
)
from aragora.client.transport import RetryConfig


class AsyncContextManagerMock:
    """Helper to create async context managers for mocking aiohttp responses."""

    def __init__(self, return_value):
        self.return_value = return_value

    async def __aenter__(self):
        return self.return_value

    async def __aexit__(self, *args):
        return False


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def client():
    """Create a basic client for testing."""
    return AragoraClient(base_url="http://test.example.com")


@pytest.fixture
def client_with_api_key():
    """Create a client with API key for testing."""
    return AragoraClient(base_url="http://test.example.com", api_key="test-api-key")


@pytest.fixture
def client_with_retry():
    """Create a client with retry configuration."""
    return AragoraClient(
        base_url="http://test.example.com",
        retry_config=RetryConfig(max_retries=2, backoff_factor=0.01, jitter=False),
    )


@pytest.fixture
def client_with_rate_limit():
    """Create a client with rate limiting."""
    return AragoraClient(base_url="http://test.example.com", rate_limit_rps=100)


# ============================================================================
# Header Construction Tests
# ============================================================================


class TestGetHeaders:
    """Tests for _get_headers method."""

    def test_content_type_header(self, client):
        """Test Content-Type header is set."""
        headers = client._get_headers()
        assert headers["Content-Type"] == "application/json"

    def test_no_auth_header_without_api_key(self, client):
        """Test no Authorization header when api_key not set."""
        headers = client._get_headers()
        assert "Authorization" not in headers

    def test_auth_header_with_api_key(self, client_with_api_key):
        """Test Authorization header when api_key is set."""
        headers = client_with_api_key._get_headers()
        assert headers["Authorization"] == "Bearer test-api-key"


# ============================================================================
# Sync HTTP Method Tests
# ============================================================================


class TestSyncGet:
    """Tests for _get method."""

    def test_get_success(self, client):
        """Test successful GET request."""
        response_data = {"key": "value"}

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = MagicMock()
            mock_response.read.return_value = json.dumps(response_data).encode()
            mock_response.__enter__ = MagicMock(return_value=mock_response)
            mock_response.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_response

            result = client._get("/api/test")

            assert result == response_data

    def test_get_with_params(self, client):
        """Test GET request with query parameters."""
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = MagicMock()
            mock_response.read.return_value = b'{"result": "ok"}'
            mock_response.__enter__ = MagicMock(return_value=mock_response)
            mock_response.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_response

            client._get("/api/test", params={"limit": 10, "offset": 0})

            # Verify URL includes query params
            call_args = mock_urlopen.call_args
            request = call_args[0][0]
            assert "limit=10" in request.full_url
            assert "offset=0" in request.full_url

    def test_get_constructs_correct_url(self, client):
        """Test GET constructs correct URL with base URL."""
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = MagicMock()
            mock_response.read.return_value = b"{}"
            mock_response.__enter__ = MagicMock(return_value=mock_response)
            mock_response.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_response

            client._get("/api/debates")

            call_args = mock_urlopen.call_args
            request = call_args[0][0]
            assert "http://test.example.com/api/debates" in request.full_url


class TestSyncPost:
    """Tests for _post method."""

    def test_post_success(self, client):
        """Test successful POST request."""
        response_data = {"id": "123"}
        request_data = {"task": "test task"}

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = MagicMock()
            mock_response.read.return_value = json.dumps(response_data).encode()
            mock_response.__enter__ = MagicMock(return_value=mock_response)
            mock_response.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_response

            result = client._post("/api/test", request_data)

            assert result == response_data

    def test_post_sends_json_data(self, client):
        """Test POST sends JSON-encoded data."""
        request_data = {"key": "value"}

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = MagicMock()
            mock_response.read.return_value = b"{}"
            mock_response.__enter__ = MagicMock(return_value=mock_response)
            mock_response.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_response

            client._post("/api/test", request_data)

            call_args = mock_urlopen.call_args
            request = call_args[0][0]
            assert request.data == json.dumps(request_data).encode()

    def test_post_with_custom_headers(self, client):
        """Test POST with custom headers."""
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = MagicMock()
            mock_response.read.return_value = b"{}"
            mock_response.__enter__ = MagicMock(return_value=mock_response)
            mock_response.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_response

            client._post("/api/test", {}, headers={"X-Custom": "value"})

            call_args = mock_urlopen.call_args
            request = call_args[0][0]
            assert request.get_header("X-custom") == "value"


class TestSyncPut:
    """Tests for _put method."""

    def test_put_success(self, client):
        """Test successful PUT request."""
        response_data = {"updated": True}

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = MagicMock()
            mock_response.read.return_value = json.dumps(response_data).encode()
            mock_response.__enter__ = MagicMock(return_value=mock_response)
            mock_response.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_response

            result = client._put("/api/test", {"data": "new"})

            assert result == response_data


class TestSyncPatch:
    """Tests for _patch method."""

    def test_patch_success(self, client):
        """Test successful PATCH request."""
        response_data = {"patched": True}

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = MagicMock()
            mock_response.read.return_value = json.dumps(response_data).encode()
            mock_response.__enter__ = MagicMock(return_value=mock_response)
            mock_response.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_response

            result = client._patch("/api/test", {"field": "value"})

            assert result == response_data


class TestSyncDelete:
    """Tests for _delete method."""

    def test_delete_success(self, client):
        """Test successful DELETE request."""
        response_data = {"deleted": True}

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = MagicMock()
            mock_response.read.return_value = json.dumps(response_data).encode()
            mock_response.__enter__ = MagicMock(return_value=mock_response)
            mock_response.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_response

            result = client._delete("/api/test/123")

            assert result == response_data

    def test_delete_with_params(self, client):
        """Test DELETE with query parameters."""
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = MagicMock()
            mock_response.read.return_value = b'{"ok": true}'
            mock_response.__enter__ = MagicMock(return_value=mock_response)
            mock_response.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_response

            client._delete("/api/test", params={"force": "true"})

            call_args = mock_urlopen.call_args
            request = call_args[0][0]
            assert "force=true" in request.full_url


# ============================================================================
# HTTP Error Mapping Tests
# ============================================================================


class TestHttpErrorMapping:
    """Tests for _handle_http_error mapping."""

    def _create_http_error(self, status_code: int, body: dict | None = None):
        """Create a mock HTTPError."""
        import urllib.error

        error = urllib.error.HTTPError(
            url="http://test.example.com/api/test",
            code=status_code,
            msg="Error",
            hdrs={},
            fp=BytesIO(json.dumps(body or {"error": "Test error"}).encode()),
        )
        return error

    def test_401_raises_authentication_error(self, client):
        """Test HTTP 401 raises AuthenticationError."""
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = self._create_http_error(401)

            with pytest.raises(AuthenticationError) as exc:
                client._get("/api/test")

            assert exc.value.status_code == 401

    def test_402_raises_quota_exceeded_error(self, client):
        """Test HTTP 402 raises QuotaExceededError."""
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = self._create_http_error(402)

            with pytest.raises(QuotaExceededError) as exc:
                client._get("/api/test")

            assert exc.value.status_code == 402

    def test_404_raises_not_found_error(self, client):
        """Test HTTP 404 raises NotFoundError."""
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = self._create_http_error(404)

            with pytest.raises(NotFoundError) as exc:
                client._get("/api/test")

            assert exc.value.status_code == 404

    def test_429_raises_rate_limit_error(self, client):
        """Test HTTP 429 raises RateLimitError."""
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = self._create_http_error(429)

            with pytest.raises(RateLimitError) as exc:
                client._get("/api/test")

            assert exc.value.status_code == 429

    def test_400_raises_validation_error(self, client):
        """Test HTTP 400 raises ValidationError."""
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = self._create_http_error(400)

            with pytest.raises(ValidationError) as exc:
                client._get("/api/test")

            assert exc.value.status_code == 400

    def test_500_raises_generic_api_error(self, client):
        """Test HTTP 500 raises generic AragoraAPIError."""
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = self._create_http_error(500)

            with pytest.raises(AragoraAPIError) as exc:
                client._get("/api/test")

            assert exc.value.status_code == 500

    def test_error_message_from_body(self, client):
        """Test error message is extracted from response body."""
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = self._create_http_error(
                400, {"error": "Field 'task' is required"}
            )

            with pytest.raises(ValidationError) as exc:
                client._get("/api/test")

            assert "task" in str(exc.value)


# ============================================================================
# Retry Logic Tests
# ============================================================================


class TestRetryLogic:
    """Tests for retry logic with exponential backoff."""

    def _create_http_error(self, status_code: int):
        """Create a mock HTTPError."""
        import urllib.error

        return urllib.error.HTTPError(
            url="http://test.example.com/api/test",
            code=status_code,
            msg="Error",
            hdrs={},
            fp=BytesIO(b'{"error": "Server error"}'),
        )

    def test_retry_on_500(self, client_with_retry):
        """Test request is retried on HTTP 500."""
        with patch("urllib.request.urlopen") as mock_urlopen:
            # First two calls fail with 500, third succeeds
            mock_response = MagicMock()
            mock_response.read.return_value = b'{"success": true}'
            mock_response.__enter__ = MagicMock(return_value=mock_response)
            mock_response.__exit__ = MagicMock(return_value=False)

            mock_urlopen.side_effect = [
                self._create_http_error(500),
                self._create_http_error(500),
                mock_response,
            ]

            result = client_with_retry._get("/api/test")

            assert result == {"success": True}
            assert mock_urlopen.call_count == 3

    def test_retry_on_429(self, client_with_retry):
        """Test request is retried on HTTP 429."""
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = MagicMock()
            mock_response.read.return_value = b'{"ok": true}'
            mock_response.__enter__ = MagicMock(return_value=mock_response)
            mock_response.__exit__ = MagicMock(return_value=False)

            mock_urlopen.side_effect = [
                self._create_http_error(429),
                mock_response,
            ]

            result = client_with_retry._get("/api/test")

            assert result == {"ok": True}
            assert mock_urlopen.call_count == 2

    def test_no_retry_on_400(self, client_with_retry):
        """Test no retry on HTTP 400 (not retryable)."""
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = self._create_http_error(400)

            with pytest.raises(ValidationError):
                client_with_retry._get("/api/test")

            # Should only attempt once (no retry)
            assert mock_urlopen.call_count == 1

    def test_no_retry_on_401(self, client_with_retry):
        """Test no retry on HTTP 401 (not retryable)."""
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = self._create_http_error(401)

            with pytest.raises(AuthenticationError):
                client_with_retry._get("/api/test")

            assert mock_urlopen.call_count == 1

    def test_max_retries_exhausted(self, client_with_retry):
        """Test error is raised after max retries exhausted."""
        with patch("urllib.request.urlopen") as mock_urlopen:
            # All attempts fail
            mock_urlopen.side_effect = self._create_http_error(500)

            with pytest.raises(AragoraAPIError):
                client_with_retry._get("/api/test")

            # max_retries=2 means 3 total attempts
            assert mock_urlopen.call_count == 3

    def test_no_retry_without_config(self, client):
        """Test no retry when retry_config is None."""
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = self._create_http_error(500)

            with pytest.raises(AragoraAPIError):
                client._get("/api/test")

            # Only one attempt
            assert mock_urlopen.call_count == 1


# ============================================================================
# Rate Limiting Integration Tests
# ============================================================================


class TestRateLimitingIntegration:
    """Tests for rate limiting integration with HTTP methods."""

    def test_rate_limiter_called_before_request(self, client_with_rate_limit):
        """Test rate limiter is called before each request."""
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = MagicMock()
            mock_response.read.return_value = b"{}"
            mock_response.__enter__ = MagicMock(return_value=mock_response)
            mock_response.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_response

            with patch.object(client_with_rate_limit._rate_limiter, "wait") as mock_wait:
                client_with_rate_limit._get("/api/test")

                mock_wait.assert_called_once()

    def test_no_rate_limiting_without_limiter(self, client):
        """Test no rate limiting when rate_limit_rps=0."""
        assert client._rate_limiter is None

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = MagicMock()
            mock_response.read.return_value = b"{}"
            mock_response.__enter__ = MagicMock(return_value=mock_response)
            mock_response.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_response

            # Should work without rate limiter
            result = client._get("/api/test")
            assert result == {}


# ============================================================================
# Async HTTP Method Tests
# ============================================================================


class TestAsyncGet:
    """Tests for _get_async method."""

    @pytest.mark.asyncio
    async def test_get_async_success(self, client):
        """Test successful async GET request."""
        response_data = {"key": "value"}

        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=response_data)

        # Create async context manager that returns mock_response
        mock_session.get.return_value = AsyncContextManagerMock(mock_response)

        client._session = mock_session
        result = await client._get_async("/api/test")

        assert result == response_data

    @pytest.mark.asyncio
    async def test_get_async_with_params(self, client):
        """Test async GET with query parameters."""
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={})

        mock_session.get.return_value = AsyncContextManagerMock(mock_response)

        client._session = mock_session
        await client._get_async("/api/test", params={"limit": 10})

        # Verify params were passed
        call_kwargs = mock_session.get.call_args[1]
        assert call_kwargs["params"] == {"limit": 10}


class TestAsyncPost:
    """Tests for _post_async method."""

    @pytest.mark.asyncio
    async def test_post_async_success(self, client):
        """Test successful async POST request."""
        response_data = {"id": "123"}

        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=response_data)

        mock_session.post.return_value = AsyncContextManagerMock(mock_response)

        client._session = mock_session
        result = await client._post_async("/api/test", {"task": "test"})

        assert result == response_data


class TestAsyncErrorHandling:
    """Tests for async error handling."""

    @pytest.mark.asyncio
    async def test_async_http_error_raises_api_error(self, client):
        """Test async HTTP error raises AragoraAPIError."""
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status = 500
        mock_response.json = AsyncMock(return_value={"error": "Server error"})

        mock_session.get.return_value = AsyncContextManagerMock(mock_response)

        client._session = mock_session

        with pytest.raises(AragoraAPIError) as exc:
            await client._get_async("/api/test")

        assert exc.value.status_code == 500


class TestAsyncRetryLogic:
    """Tests for async retry logic."""

    @pytest.mark.asyncio
    async def test_async_retry_on_500(self, client_with_retry):
        """Test async request is retried on HTTP 500."""
        mock_session = MagicMock()

        # First call fails, second succeeds
        mock_error_response = MagicMock()
        mock_error_response.status = 500
        mock_error_response.json = AsyncMock(return_value={"error": "Error"})

        mock_success_response = MagicMock()
        mock_success_response.status = 200
        mock_success_response.json = AsyncMock(return_value={"ok": True})

        mock_session.get.side_effect = [
            AsyncContextManagerMock(mock_error_response),
            AsyncContextManagerMock(mock_success_response),
        ]

        client_with_retry._session = mock_session
        result = await client_with_retry._get_async("/api/test")

        assert result == {"ok": True}
        assert mock_session.get.call_count == 2


# ============================================================================
# Context Manager Tests
# ============================================================================


class TestContextManager:
    """Tests for context manager support."""

    def test_sync_context_manager(self):
        """Test synchronous context manager."""
        with AragoraClient(base_url="http://test.example.com") as client:
            assert client is not None
            assert client.base_url == "http://test.example.com"

    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        """Test asynchronous context manager."""
        async with AragoraClient(base_url="http://test.example.com") as client:
            assert client is not None
            assert client.base_url == "http://test.example.com"


# ============================================================================
# Client Initialization Tests
# ============================================================================


class TestClientInitialization:
    """Tests for AragoraClient initialization."""

    def test_default_base_url(self):
        """Test default base URL."""
        client = AragoraClient()
        assert client.base_url == "http://localhost:8080"

    def test_custom_base_url(self):
        """Test custom base URL."""
        client = AragoraClient(base_url="https://api.example.com")
        assert client.base_url == "https://api.example.com"

    def test_base_url_trailing_slash_stripped(self):
        """Test trailing slash is stripped from base URL."""
        client = AragoraClient(base_url="http://example.com/")
        assert client.base_url == "http://example.com"

    def test_default_timeout(self):
        """Test default timeout is 60 seconds."""
        client = AragoraClient()
        assert client.timeout == 60

    def test_custom_timeout(self):
        """Test custom timeout."""
        client = AragoraClient(timeout=120)
        assert client.timeout == 120

    def test_api_key_stored(self):
        """Test API key is stored."""
        client = AragoraClient(api_key="secret-key")
        assert client.api_key == "secret-key"

    def test_retry_config_stored(self):
        """Test retry config is stored."""
        config = RetryConfig(max_retries=5)
        client = AragoraClient(retry_config=config)
        assert client.retry_config == config
        assert client.retry_config.max_retries == 5

    def test_rate_limiter_created(self):
        """Test rate limiter is created when rps > 0."""
        client = AragoraClient(rate_limit_rps=10)
        assert client._rate_limiter is not None
        assert client._rate_limiter.rps == 10

    def test_no_rate_limiter_when_disabled(self):
        """Test no rate limiter when rps=0."""
        client = AragoraClient(rate_limit_rps=0)
        assert client._rate_limiter is None

    def test_api_interfaces_initialized(self):
        """Test all API interfaces are initialized."""
        client = AragoraClient()

        # Check key interfaces exist
        assert hasattr(client, "debates")
        assert hasattr(client, "agents")
        assert hasattr(client, "auth")
        assert hasattr(client, "rbac")
        assert hasattr(client, "audit")
        assert hasattr(client, "documents")
        assert hasattr(client, "gauntlet")
        assert hasattr(client, "knowledge")
        assert hasattr(client, "memory")
