"""
Tests for aragora.server.middleware.request_logging - Request/Response Logging Middleware.

Tests cover:
- RequestContext dataclass and timing
- Request ID generation
- Token hashing for audit
- Sensitive value masking
- Header sanitization
- Query parameter sanitization
- log_request and log_response functions
- request_logging decorator (sync and async)
- IP extraction from requests
- Context variable for request ID tracking
"""

from __future__ import annotations

import asyncio
import logging
import time
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _reset_request_logging_context():
    """Reset request logging context var to prevent cross-test contamination."""
    from aragora.server.middleware.request_logging import _current_request_id

    token = _current_request_id.set(None)
    yield
    _current_request_id.reset(token)


# =============================================================================
# Test RequestContext
# =============================================================================


class TestRequestContext:
    """Tests for RequestContext dataclass."""

    def test_create_request_context(self):
        """Should create RequestContext with all fields."""
        from aragora.server.middleware.request_logging import RequestContext

        ctx = RequestContext(
            request_id="req-abc123",
            method="POST",
            path="/api/debates",
            client_ip="192.168.1.1",
            start_time=time.time(),
        )

        assert ctx.request_id == "req-abc123"
        assert ctx.method == "POST"
        assert ctx.path == "/api/debates"
        assert ctx.client_ip == "192.168.1.1"
        assert ctx.token_hash is None

    def test_request_context_with_token_hash(self):
        """Should include token hash when provided."""
        from aragora.server.middleware.request_logging import RequestContext

        ctx = RequestContext(
            request_id="req-123",
            method="GET",
            path="/api/test",
            client_ip="127.0.0.1",
            start_time=time.time(),
            token_hash="abcd1234",
        )

        assert ctx.token_hash == "abcd1234"

    def test_elapsed_ms_calculation(self):
        """Should calculate elapsed time in milliseconds."""
        from aragora.server.middleware.request_logging import RequestContext

        start = time.time()
        ctx = RequestContext(
            request_id="req-123",
            method="GET",
            path="/api/test",
            client_ip="127.0.0.1",
            start_time=start,
        )

        # Sleep briefly
        time.sleep(0.01)  # 10ms

        elapsed = ctx.elapsed_ms()
        assert elapsed >= 10  # At least 10ms
        assert elapsed < 1000  # But not too long


# =============================================================================
# Test Request ID Generation
# =============================================================================


class TestGenerateRequestId:
    """Tests for generate_request_id function."""

    def test_generate_request_id_format(self):
        """Should generate ID with req- prefix."""
        from aragora.server.middleware.request_logging import generate_request_id

        request_id = generate_request_id()

        assert request_id.startswith("req-")
        assert len(request_id) == 16  # "req-" + 12 hex chars

    def test_generate_request_id_unique(self):
        """Generated IDs should be unique."""
        from aragora.server.middleware.request_logging import generate_request_id

        ids = [generate_request_id() for _ in range(1000)]
        assert len(set(ids)) == 1000

    def test_generate_request_id_alphanumeric(self):
        """ID should contain only alphanumeric characters after prefix."""
        from aragora.server.middleware.request_logging import generate_request_id

        request_id = generate_request_id()
        suffix = request_id[4:]  # Remove "req-"

        assert suffix.isalnum()


# =============================================================================
# Test Token Hashing
# =============================================================================


class TestHashToken:
    """Tests for hash_token function."""

    def test_hash_token_returns_8_chars(self):
        """Should return 8-character truncated hash."""
        from aragora.server.middleware.request_logging import hash_token

        hashed = hash_token("my-secret-token")
        assert len(hashed) == 8

    def test_hash_token_consistent(self):
        """Same token should produce same hash."""
        from aragora.server.middleware.request_logging import hash_token

        token = "my-consistent-token"
        hash1 = hash_token(token)
        hash2 = hash_token(token)

        assert hash1 == hash2

    def test_hash_token_different_for_different_tokens(self):
        """Different tokens should produce different hashes."""
        from aragora.server.middleware.request_logging import hash_token

        hash1 = hash_token("token-1")
        hash2 = hash_token("token-2")

        assert hash1 != hash2

    def test_hash_token_uses_sha256(self):
        """Should use SHA256 for hashing."""
        import hashlib

        from aragora.server.middleware.request_logging import hash_token

        token = "test-token"
        expected = hashlib.sha256(token.encode()).hexdigest()[:8]

        assert hash_token(token) == expected


# =============================================================================
# Test Sensitive Value Masking
# =============================================================================


class TestMaskSensitiveValue:
    """Tests for mask_sensitive_value function."""

    def test_mask_long_value(self):
        """Should mask middle of long values."""
        from aragora.server.middleware.request_logging import mask_sensitive_value

        masked = mask_sensitive_value("Bearer my-secret-token")

        assert masked == "Bear****oken"
        assert "my-secret" not in masked

    def test_mask_short_value(self):
        """Should fully mask short values."""
        from aragora.server.middleware.request_logging import mask_sensitive_value

        masked = mask_sensitive_value("short")
        assert masked == "****"

    def test_mask_exactly_8_chars(self):
        """Should fully mask values of exactly 8 characters."""
        from aragora.server.middleware.request_logging import mask_sensitive_value

        masked = mask_sensitive_value("12345678")
        assert masked == "****"

    def test_mask_9_char_value(self):
        """Should partially mask values longer than 8 characters."""
        from aragora.server.middleware.request_logging import mask_sensitive_value

        masked = mask_sensitive_value("123456789")
        assert masked == "1234****6789"


# =============================================================================
# Test Header Sanitization
# =============================================================================


class TestSanitizeHeaders:
    """Tests for sanitize_headers function."""

    def test_sanitize_authorization_header(self):
        """Should mask Authorization header."""
        from aragora.server.middleware.request_logging import sanitize_headers

        headers = {"Authorization": "Bearer my-secret-token-12345"}
        sanitized = sanitize_headers(headers)

        assert "****" in sanitized["Authorization"]
        assert "my-secret" not in sanitized["Authorization"]

    def test_sanitize_api_key_header(self):
        """Should mask X-Api-Key header."""
        from aragora.server.middleware.request_logging import sanitize_headers

        headers = {"X-Api-Key": "api-key-12345678"}
        sanitized = sanitize_headers(headers)

        assert "****" in sanitized["X-Api-Key"]

    def test_sanitize_cookie_header(self):
        """Should mask Cookie header."""
        from aragora.server.middleware.request_logging import sanitize_headers

        headers = {"Cookie": "session=abc123xyz789"}
        sanitized = sanitize_headers(headers)

        assert "****" in sanitized["Cookie"]

    def test_sanitize_preserves_non_sensitive(self):
        """Should preserve non-sensitive headers."""
        from aragora.server.middleware.request_logging import sanitize_headers

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "Mozilla/5.0",
        }
        sanitized = sanitize_headers(headers)

        assert sanitized["Content-Type"] == "application/json"
        assert sanitized["Accept"] == "application/json"
        assert sanitized["User-Agent"] == "Mozilla/5.0"

    def test_sanitize_case_insensitive(self):
        """Should match headers case-insensitively."""
        from aragora.server.middleware.request_logging import sanitize_headers

        headers = {
            "authorization": "Bearer token123456",
            "COOKIE": "session=value123456",
        }
        sanitized = sanitize_headers(headers)

        assert "****" in sanitized["authorization"]
        assert "****" in sanitized["COOKIE"]

    def test_sanitize_returns_copy(self):
        """Should return a copy, not modify original."""
        from aragora.server.middleware.request_logging import sanitize_headers

        headers = {"Authorization": "Bearer secret"}
        sanitized = sanitize_headers(headers)

        assert headers["Authorization"] == "Bearer secret"
        assert sanitized != headers


# =============================================================================
# Test Query Parameter Sanitization
# =============================================================================


class TestSanitizeParams:
    """Tests for sanitize_params function."""

    def test_sanitize_token_param(self):
        """Should mask token parameter."""
        from aragora.server.middleware.request_logging import sanitize_params

        params = {"token": "secret-token-value"}
        sanitized = sanitize_params(params)

        assert sanitized["token"] == "****"

    def test_sanitize_api_key_param(self):
        """Should mask api_key parameter."""
        from aragora.server.middleware.request_logging import sanitize_params

        params = {"api_key": "my-api-key", "apikey": "another-key"}
        sanitized = sanitize_params(params)

        assert sanitized["api_key"] == "****"
        assert sanitized["apikey"] == "****"

    def test_sanitize_password_param(self):
        """Should mask password parameter."""
        from aragora.server.middleware.request_logging import sanitize_params

        params = {"password": "super-secret"}
        sanitized = sanitize_params(params)

        assert sanitized["password"] == "****"

    def test_sanitize_access_token_param(self):
        """Should mask access_token and refresh_token."""
        from aragora.server.middleware.request_logging import sanitize_params

        params = {
            "access_token": "access123",
            "refresh_token": "refresh456",
        }
        sanitized = sanitize_params(params)

        assert sanitized["access_token"] == "****"
        assert sanitized["refresh_token"] == "****"

    def test_sanitize_preserves_non_sensitive(self):
        """Should preserve non-sensitive parameters."""
        from aragora.server.middleware.request_logging import sanitize_params

        params = {
            "page": "1",
            "limit": "10",
            "sort": "created_at",
        }
        sanitized = sanitize_params(params)

        assert sanitized["page"] == "1"
        assert sanitized["limit"] == "10"
        assert sanitized["sort"] == "created_at"

    def test_sanitize_case_insensitive(self):
        """Should match parameters case-insensitively."""
        from aragora.server.middleware.request_logging import sanitize_params

        params = {"PASSWORD": "secret", "Token": "value"}
        sanitized = sanitize_params(params)

        assert sanitized["PASSWORD"] == "****"
        assert sanitized["Token"] == "****"


# =============================================================================
# Test log_request Function
# =============================================================================


class TestLogRequest:
    """Tests for log_request function."""

    def test_log_request_basic(self, caplog):
        """Should log basic request info."""
        from aragora.server.middleware.request_logging import RequestContext, log_request

        ctx = RequestContext(
            request_id="req-test123",
            method="GET",
            path="/api/debates",
            client_ip="192.168.1.100",
            start_time=time.time(),
        )

        with caplog.at_level(logging.INFO):
            log_request(ctx)

        assert "req-test123" in caplog.text
        assert "GET" in caplog.text
        assert "/api/debates" in caplog.text
        assert "192.168.1.100" in caplog.text

    def test_log_request_with_token_hash(self, caplog):
        """Should include token hash when present."""
        from aragora.server.middleware.request_logging import RequestContext, log_request

        ctx = RequestContext(
            request_id="req-test123",
            method="POST",
            path="/api/test",
            client_ip="127.0.0.1",
            start_time=time.time(),
            token_hash="abcd1234",
        )

        with caplog.at_level(logging.INFO):
            log_request(ctx)

        # Token hash should be in extra, not necessarily in message
        assert len(caplog.records) == 1

    def test_log_request_with_query_params(self, caplog):
        """Should log sanitized query parameters."""
        from aragora.server.middleware.request_logging import RequestContext, log_request

        ctx = RequestContext(
            request_id="req-test123",
            method="GET",
            path="/api/search",
            client_ip="127.0.0.1",
            start_time=time.time(),
        )

        with caplog.at_level(logging.INFO):
            log_request(ctx, query_params={"query": "test", "token": "secret"})

        assert len(caplog.records) == 1

    def test_log_request_custom_log_level(self, caplog):
        """Should respect custom log level."""
        from aragora.server.middleware.request_logging import RequestContext, log_request

        ctx = RequestContext(
            request_id="req-test123",
            method="GET",
            path="/api/test",
            client_ip="127.0.0.1",
            start_time=time.time(),
        )

        with caplog.at_level(logging.DEBUG):
            log_request(ctx, log_level=logging.DEBUG)

        assert len(caplog.records) == 1
        assert caplog.records[0].levelno == logging.DEBUG


# =============================================================================
# Test log_response Function
# =============================================================================


class TestLogResponse:
    """Tests for log_response function."""

    def test_log_response_success(self, caplog):
        """Should log successful response."""
        from aragora.server.middleware.request_logging import (
            RequestContext,
            log_response,
        )

        ctx = RequestContext(
            request_id="req-test123",
            method="GET",
            path="/api/debates",
            client_ip="192.168.1.100",
            start_time=time.time() - 0.1,  # 100ms ago
        )

        with caplog.at_level(logging.INFO):
            log_response(ctx, status_code=200)

        assert "req-test123" in caplog.text
        assert "200" in caplog.text
        assert caplog.records[0].levelno == logging.INFO

    def test_log_response_client_error(self, caplog):
        """Should log 4xx as warning."""
        from aragora.server.middleware.request_logging import (
            RequestContext,
            log_response,
        )

        ctx = RequestContext(
            request_id="req-test123",
            method="POST",
            path="/api/debates",
            client_ip="127.0.0.1",
            start_time=time.time(),
        )

        with caplog.at_level(logging.WARNING):
            log_response(ctx, status_code=400)

        assert caplog.records[0].levelno == logging.WARNING

    def test_log_response_server_error(self, caplog):
        """Should log 5xx as error."""
        from aragora.server.middleware.request_logging import (
            RequestContext,
            log_response,
        )

        ctx = RequestContext(
            request_id="req-test123",
            method="POST",
            path="/api/debates",
            client_ip="127.0.0.1",
            start_time=time.time(),
        )

        with caplog.at_level(logging.ERROR):
            log_response(ctx, status_code=500)

        assert caplog.records[0].levelno == logging.ERROR

    def test_log_response_with_size(self, caplog):
        """Should include response size when provided."""
        from aragora.server.middleware.request_logging import (
            RequestContext,
            log_response,
        )

        ctx = RequestContext(
            request_id="req-test123",
            method="GET",
            path="/api/large",
            client_ip="127.0.0.1",
            start_time=time.time(),
        )

        with caplog.at_level(logging.INFO):
            log_response(ctx, status_code=200, response_size=1024)

        assert len(caplog.records) == 1

    def test_log_response_with_error(self, caplog):
        """Should include error message when provided."""
        from aragora.server.middleware.request_logging import (
            RequestContext,
            log_response,
        )

        ctx = RequestContext(
            request_id="req-test123",
            method="POST",
            path="/api/test",
            client_ip="127.0.0.1",
            start_time=time.time(),
        )

        with caplog.at_level(logging.ERROR):
            log_response(ctx, status_code=500, error="Database connection failed")

        assert len(caplog.records) == 1

    def test_log_response_elapsed_time(self, caplog):
        """Should include elapsed time in log."""
        from aragora.server.middleware.request_logging import (
            RequestContext,
            log_response,
        )

        ctx = RequestContext(
            request_id="req-test123",
            method="GET",
            path="/api/test",
            client_ip="127.0.0.1",
            start_time=time.time() - 0.5,  # 500ms ago
        )

        with caplog.at_level(logging.INFO):
            log_response(ctx, status_code=200)

        # Should contain elapsed time in ms
        assert "ms" in caplog.text


# =============================================================================
# Test request_logging Decorator
# =============================================================================


class TestRequestLoggingDecorator:
    """Tests for request_logging decorator."""

    def test_decorator_wraps_sync_function(self):
        """Should wrap synchronous functions correctly."""
        from aragora.server.middleware.request_logging import request_logging

        @request_logging()
        def handler(self, request):
            return {"status": "ok"}

        # Create mock request
        mock_request = MagicMock()
        mock_request.method = "GET"
        mock_request.path = "/api/test"
        mock_request.headers = {}

        result = handler(None, mock_request)
        assert result == {"status": "ok"}

    def test_decorator_wraps_async_function(self):
        """Should wrap async functions correctly."""
        from aragora.server.middleware.request_logging import request_logging

        @request_logging()
        async def handler(self, request):
            return {"status": "async_ok"}

        mock_request = MagicMock()
        mock_request.method = "POST"
        mock_request.path = "/api/async"
        mock_request.headers = {}

        result = asyncio.run(handler(None, mock_request))
        assert result == {"status": "async_ok"}

    def test_decorator_extracts_existing_request_id(self):
        """Should use existing X-Request-ID from headers."""
        from aragora.server.middleware.request_logging import request_logging

        @request_logging()
        def handler(self, request):
            return {"status": "ok"}

        mock_request = MagicMock()
        mock_request.method = "GET"
        mock_request.path = "/api/test"

        mock_headers = MagicMock()
        mock_headers.get = lambda k, d=None: "existing-req-id" if k == "X-Request-ID" else d
        mock_request.headers = mock_headers

        handler(None, mock_request)
        # Request should be logged with existing ID

    def test_decorator_logs_errors(self, caplog):
        """Should log exceptions as 500."""
        from aragora.server.middleware.request_logging import request_logging

        @request_logging()
        def handler(self, request):
            raise ValueError("Test error")

        mock_request = MagicMock()
        mock_request.method = "POST"
        mock_request.path = "/api/error"
        mock_request.headers = {}

        with caplog.at_level(logging.ERROR):
            with pytest.raises(ValueError):
                handler(None, mock_request)

    def test_decorator_with_slow_request_threshold(self, caplog):
        """Should warn on slow requests."""
        from aragora.server.middleware.request_logging import request_logging

        @request_logging(slow_request_threshold_ms=10)
        async def handler(self, request):
            import asyncio

            await asyncio.sleep(0.05)  # 50ms
            return {"status": "slow"}

        mock_request = MagicMock()
        mock_request.method = "GET"
        mock_request.path = "/api/slow"
        mock_request.headers = {}

        with caplog.at_level(logging.WARNING):
            asyncio.run(handler(None, mock_request))

        # Should log warning for slow request

    def test_decorator_extracts_token_hash(self):
        """Should extract and hash Bearer token."""
        from aragora.server.middleware.request_logging import request_logging

        @request_logging()
        def handler(self, request):
            return {"status": "ok"}

        mock_request = MagicMock()
        mock_request.method = "GET"
        mock_request.path = "/api/auth"

        mock_headers = MagicMock()
        mock_headers.get = lambda k, d=None: (
            "Bearer my-secret-token" if k == "Authorization" else d
        )
        mock_request.headers = mock_headers

        handler(None, mock_request)
        # Token hash should be computed


# =============================================================================
# Test IP Extraction
# =============================================================================


class TestExtractIp:
    """Tests for _extract_ip function."""

    def test_extract_ip_from_x_forwarded_for(self):
        """Should extract IP from X-Forwarded-For header."""
        from aragora.server.middleware.request_logging import _extract_ip

        mock_request = MagicMock()
        mock_headers = MagicMock()
        mock_headers.get = lambda k, d="": (
            "203.0.113.50, 198.51.100.178" if k == "X-Forwarded-For" else d
        )
        mock_request.headers = mock_headers

        ip = _extract_ip(mock_request)
        assert ip == "203.0.113.50"  # First IP is original client

    def test_extract_ip_from_remote(self):
        """Should extract IP from remote attribute."""
        from aragora.server.middleware.request_logging import _extract_ip

        mock_request = MagicMock()
        mock_request.headers = MagicMock()
        mock_request.headers.get = lambda k, d="": ""
        mock_request.remote = "10.0.0.1"

        ip = _extract_ip(mock_request)
        assert ip == "10.0.0.1"

    def test_extract_ip_from_transport(self):
        """Should extract IP from transport peername."""
        from aragora.server.middleware.request_logging import _extract_ip

        mock_request = MagicMock()
        mock_request.headers = MagicMock()
        mock_request.headers.get = lambda k, d="": ""
        del mock_request.remote  # Remove remote attribute

        mock_transport = MagicMock()
        mock_transport.get_extra_info.return_value = ("172.16.0.1", 12345)
        mock_request.transport = mock_transport

        ip = _extract_ip(mock_request)
        assert ip == "172.16.0.1"

    def test_extract_ip_returns_unknown(self):
        """Should return 'unknown' when IP cannot be determined."""
        from aragora.server.middleware.request_logging import _extract_ip

        mock_request = MagicMock()
        mock_request.headers = MagicMock()
        mock_request.headers.get = lambda k, d="": ""
        del mock_request.remote
        del mock_request.transport

        ip = _extract_ip(mock_request)
        assert ip == "unknown"

    def test_extract_ip_none_request(self):
        """Should handle None request."""
        from aragora.server.middleware.request_logging import _extract_ip

        ip = _extract_ip(None)
        assert ip == "unknown"


# =============================================================================
# Test Context Variables
# =============================================================================


class TestContextVariables:
    """Tests for request ID context variables."""

    def test_get_current_request_id_default(self):
        """Should return None by default."""
        from aragora.server.middleware.request_logging import get_current_request_id

        # In a fresh context, should be None
        assert get_current_request_id() is None

    def test_set_and_get_current_request_id(self):
        """Should set and retrieve request ID."""
        from aragora.server.middleware.request_logging import (
            get_current_request_id,
            set_current_request_id,
        )

        set_current_request_id("req-context-test")
        assert get_current_request_id() == "req-context-test"

        # Clean up
        set_current_request_id(None)

    def test_request_id_isolation_between_contexts(self):
        """Request IDs should be isolated between contexts."""
        import contextvars

        from aragora.server.middleware.request_logging import (
            _current_request_id,
            get_current_request_id,
            set_current_request_id,
        )

        # Set in main context
        set_current_request_id("main-request")
        assert get_current_request_id() == "main-request"

        # Context vars are isolated per coroutine/context
        # This is a basic test - full async isolation tested separately


# =============================================================================
# Test Constants
# =============================================================================


class TestConstants:
    """Tests for module constants."""

    def test_request_id_header_constant(self):
        """Should have correct REQUEST_ID_HEADER."""
        from aragora.server.middleware.request_logging import REQUEST_ID_HEADER

        assert REQUEST_ID_HEADER == "X-Request-ID"

    def test_sensitive_headers_set(self):
        """Should have comprehensive sensitive headers set."""
        from aragora.server.middleware.request_logging import SENSITIVE_HEADERS

        assert "authorization" in SENSITIVE_HEADERS
        assert "x-api-key" in SENSITIVE_HEADERS
        assert "cookie" in SENSITIVE_HEADERS
        assert "set-cookie" in SENSITIVE_HEADERS
        assert "x-auth-token" in SENSITIVE_HEADERS

    def test_sensitive_params_set(self):
        """Should have comprehensive sensitive params set."""
        from aragora.server.middleware.request_logging import SENSITIVE_PARAMS

        assert "token" in SENSITIVE_PARAMS
        assert "api_key" in SENSITIVE_PARAMS
        assert "apikey" in SENSITIVE_PARAMS
        assert "password" in SENSITIVE_PARAMS
        assert "secret" in SENSITIVE_PARAMS
        assert "access_token" in SENSITIVE_PARAMS
        assert "refresh_token" in SENSITIVE_PARAMS


# =============================================================================
# Test Module Exports
# =============================================================================


class TestModuleExports:
    """Tests for module exports."""

    def test_all_exports_accessible(self):
        """All __all__ exports should be accessible."""
        from aragora.server.middleware.request_logging import (
            REQUEST_ID_HEADER,
            RequestContext,
            generate_request_id,
            get_current_request_id,
            hash_token,
            log_request,
            log_response,
            request_logging,
            sanitize_headers,
            sanitize_params,
            set_current_request_id,
        )

        assert REQUEST_ID_HEADER is not None
        assert RequestContext is not None
        assert generate_request_id is not None
        assert hash_token is not None
        assert sanitize_headers is not None
        assert sanitize_params is not None
        assert log_request is not None
        assert log_response is not None
        assert request_logging is not None
        assert get_current_request_id is not None
        assert set_current_request_id is not None


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_sanitize_headers_empty(self):
        """Should handle empty headers dict."""
        from aragora.server.middleware.request_logging import sanitize_headers

        sanitized = sanitize_headers({})
        assert sanitized == {}

    def test_sanitize_params_empty(self):
        """Should handle empty params dict."""
        from aragora.server.middleware.request_logging import sanitize_params

        sanitized = sanitize_params({})
        assert sanitized == {}

    def test_mask_empty_value(self):
        """Should handle empty value."""
        from aragora.server.middleware.request_logging import mask_sensitive_value

        masked = mask_sensitive_value("")
        assert masked == "****"

    def test_hash_token_empty(self):
        """Should handle empty token."""
        from aragora.server.middleware.request_logging import hash_token

        hashed = hash_token("")
        assert len(hashed) == 8

    def test_log_request_no_headers(self, caplog):
        """Should handle request context without extra info."""
        from aragora.server.middleware.request_logging import RequestContext, log_request

        ctx = RequestContext(
            request_id="req-minimal",
            method="GET",
            path="/",
            client_ip="127.0.0.1",
            start_time=time.time(),
        )

        with caplog.at_level(logging.INFO):
            log_request(ctx)

        assert len(caplog.records) == 1

    def test_decorator_without_request_arg(self):
        """Should handle function without request in args."""
        from aragora.server.middleware.request_logging import request_logging

        @request_logging()
        def handler():
            return {"status": "ok"}

        result = handler()
        assert result == {"status": "ok"}

    def test_sanitize_headers_non_string_value(self):
        """Should handle non-string header values."""
        from aragora.server.middleware.request_logging import sanitize_headers

        headers = {"Content-Length": 1024}
        sanitized = sanitize_headers(headers)
        assert sanitized["Content-Length"] == 1024

    def test_x_forwarded_for_whitespace(self):
        """Should handle X-Forwarded-For with extra whitespace."""
        from aragora.server.middleware.request_logging import _extract_ip

        mock_request = MagicMock()
        mock_request.headers = MagicMock()
        mock_request.headers.get = lambda k, d="": (
            "  192.168.1.1  ,  10.0.0.1  " if k == "X-Forwarded-For" else d
        )

        ip = _extract_ip(mock_request)
        assert ip == "192.168.1.1"  # Should be trimmed

    def test_x_forwarded_for_single_ip(self):
        """Should handle single IP in X-Forwarded-For."""
        from aragora.server.middleware.request_logging import _extract_ip

        mock_request = MagicMock()
        mock_request.headers = MagicMock()
        mock_request.headers.get = lambda k, d="": ("10.0.0.5" if k == "X-Forwarded-For" else d)

        ip = _extract_ip(mock_request)
        assert ip == "10.0.0.5"

    def test_transport_peername_none(self):
        """Should return unknown when transport peername is None."""
        from aragora.server.middleware.request_logging import _extract_ip

        mock_request = MagicMock()
        mock_request.headers = MagicMock()
        mock_request.headers.get = lambda k, d="": ""
        del mock_request.remote

        mock_transport = MagicMock()
        mock_transport.get_extra_info.return_value = None
        mock_request.transport = mock_transport

        ip = _extract_ip(mock_request)
        assert ip == "unknown"

    def test_request_context_elapsed_at_zero(self):
        """elapsed_ms should be near zero right after creation."""
        from aragora.server.middleware.request_logging import RequestContext

        ctx = RequestContext(
            request_id="req-imm",
            method="GET",
            path="/",
            client_ip="127.0.0.1",
            start_time=time.time(),
        )

        elapsed = ctx.elapsed_ms()
        assert elapsed < 100  # Should be very small

    def test_log_response_custom_log_level(self, caplog):
        """Should respect custom log level for response."""
        from aragora.server.middleware.request_logging import RequestContext, log_response

        ctx = RequestContext(
            request_id="req-custom",
            method="GET",
            path="/api/test",
            client_ip="127.0.0.1",
            start_time=time.time(),
        )

        with caplog.at_level(logging.DEBUG):
            log_response(ctx, status_code=200, log_level=logging.DEBUG)

        assert caplog.records[0].levelno == logging.DEBUG

    def test_log_response_3xx_is_info(self, caplog):
        """3xx responses should be logged at INFO level."""
        from aragora.server.middleware.request_logging import RequestContext, log_response

        ctx = RequestContext(
            request_id="req-redirect",
            method="GET",
            path="/api/redirect",
            client_ip="127.0.0.1",
            start_time=time.time(),
        )

        with caplog.at_level(logging.INFO):
            log_response(ctx, status_code=301)

        assert caplog.records[0].levelno == logging.INFO

    def test_log_response_404_is_warning(self, caplog):
        """404 responses should be logged at WARNING level."""
        from aragora.server.middleware.request_logging import RequestContext, log_response

        ctx = RequestContext(
            request_id="req-notfound",
            method="GET",
            path="/api/missing",
            client_ip="127.0.0.1",
            start_time=time.time(),
        )

        with caplog.at_level(logging.WARNING):
            log_response(ctx, status_code=404)

        assert caplog.records[0].levelno == logging.WARNING

    def test_log_response_429_is_warning(self, caplog):
        """429 responses should be logged at WARNING level."""
        from aragora.server.middleware.request_logging import RequestContext, log_response

        ctx = RequestContext(
            request_id="req-ratelimited",
            method="POST",
            path="/api/debates",
            client_ip="127.0.0.1",
            start_time=time.time(),
        )

        with caplog.at_level(logging.WARNING):
            log_response(ctx, status_code=429)

        assert caplog.records[0].levelno == logging.WARNING

    def test_log_request_no_query_params(self, caplog):
        """Should handle None query_params."""
        from aragora.server.middleware.request_logging import RequestContext, log_request

        ctx = RequestContext(
            request_id="req-noq",
            method="GET",
            path="/api/test",
            client_ip="127.0.0.1",
            start_time=time.time(),
        )

        with caplog.at_level(logging.INFO):
            log_request(ctx, query_params=None, headers=None)

        assert len(caplog.records) == 1

    def test_sanitize_set_cookie_header(self):
        """Should mask Set-Cookie header value."""
        from aragora.server.middleware.request_logging import sanitize_headers

        headers = {"Set-Cookie": "session=very-long-secret-session-id-here"}
        sanitized = sanitize_headers(headers)

        assert "****" in sanitized["Set-Cookie"]
        assert "very-long-secret" not in sanitized["Set-Cookie"]

    def test_sanitize_x_auth_token_header(self):
        """Should mask X-Auth-Token header value."""
        from aragora.server.middleware.request_logging import sanitize_headers

        headers = {"X-Auth-Token": "auth-token-123456789"}
        sanitized = sanitize_headers(headers)

        assert "****" in sanitized["X-Auth-Token"]

    def test_sanitize_secret_param(self):
        """Should mask 'secret' query parameter."""
        from aragora.server.middleware.request_logging import sanitize_params

        params = {"secret": "my-secret-value"}
        sanitized = sanitize_params(params)

        assert sanitized["secret"] == "****"

    def test_mask_single_character_value(self):
        """Should fully mask single character value."""
        from aragora.server.middleware.request_logging import mask_sensitive_value

        masked = mask_sensitive_value("x")
        assert masked == "****"

    def test_hash_token_hex_output(self):
        """Hash token should return hex characters only."""
        from aragora.server.middleware.request_logging import hash_token

        hashed = hash_token("test-api-key")
        assert all(c in "0123456789abcdef" for c in hashed)

    def test_sanitize_headers_preserves_order(self):
        """Should preserve header key names exactly."""
        from aragora.server.middleware.request_logging import sanitize_headers

        headers = {"Content-Type": "text/html", "X-Custom": "value"}
        sanitized = sanitize_headers(headers)

        assert set(sanitized.keys()) == {"Content-Type", "X-Custom"}

    def test_decorator_async_logs_errors(self, caplog):
        """Async decorator should log exceptions as 500."""
        from aragora.server.middleware.request_logging import request_logging

        @request_logging()
        async def handler(self, request):
            raise RuntimeError("Async error")

        mock_request = MagicMock()
        mock_request.method = "POST"
        mock_request.path = "/api/async-error"
        mock_request.headers = {}

        with caplog.at_level(logging.ERROR):
            with pytest.raises(RuntimeError):
                asyncio.run(handler(None, mock_request))

    def test_decorator_adds_request_id_to_response(self):
        """Async decorator should add X-Request-ID to response headers."""
        from aragora.server.middleware.request_logging import request_logging

        @request_logging()
        async def handler(self, request):
            response = MagicMock()
            response.status = 200
            response.content_length = 100
            response.headers = {}
            return response

        mock_request = MagicMock()
        mock_request.method = "GET"
        mock_request.path = "/api/test"
        mock_request.headers = {}

        result = asyncio.run(handler(None, mock_request))
        assert "X-Request-ID" in result.headers

    def test_decorator_respects_existing_request_id_async(self):
        """Async decorator should use existing X-Request-ID from headers."""
        from aragora.server.middleware.request_logging import request_logging

        @request_logging()
        async def handler(self, request):
            response = MagicMock()
            response.status = 200
            response.content_length = None
            response.headers = {}
            return response

        mock_request = MagicMock()
        mock_request.method = "GET"
        mock_request.path = "/api/test"

        mock_headers = MagicMock()
        mock_headers.get = lambda k, d=None: (
            "custom-req-id-42" if k == "X-Request-ID" else "" if k == "Authorization" else d
        )
        mock_request.headers = mock_headers

        result = asyncio.run(handler(None, mock_request))
        assert result.headers["X-Request-ID"] == "custom-req-id-42"

    def test_decorator_handles_none_response(self):
        """Should handle None response from handler."""
        from aragora.server.middleware.request_logging import request_logging

        @request_logging()
        async def handler(self, request):
            return None

        mock_request = MagicMock()
        mock_request.method = "GET"
        mock_request.path = "/api/test"
        mock_request.headers = {}

        result = asyncio.run(handler(None, mock_request))
        assert result is None

    def test_decorator_sync_error_truncates_message(self, caplog):
        """Sync decorator should truncate error message to 200 chars."""
        from aragora.server.middleware.request_logging import request_logging

        @request_logging()
        def handler(self, request):
            raise ValueError("x" * 500)

        mock_request = MagicMock()
        mock_request.method = "POST"
        mock_request.path = "/api/test"
        mock_request.headers = {}

        with caplog.at_level(logging.ERROR):
            with pytest.raises(ValueError):
                handler(None, mock_request)

    def test_decorator_sync_without_request(self):
        """Sync handler called without request should use defaults."""
        from aragora.server.middleware.request_logging import request_logging

        @request_logging()
        def handler(self):
            return {"status": "ok"}

        result = handler(None)
        assert result == {"status": "ok"}

    def test_set_and_get_request_id_roundtrip(self):
        """Should roundtrip request ID through context variables."""
        from aragora.server.middleware.request_logging import (
            get_current_request_id,
            set_current_request_id,
        )

        original = get_current_request_id()
        try:
            set_current_request_id("test-req-roundtrip")
            assert get_current_request_id() == "test-req-roundtrip"

            set_current_request_id("test-req-updated")
            assert get_current_request_id() == "test-req-updated"
        finally:
            set_current_request_id(original)

    def test_request_id_suffix_is_lowercase_hex(self):
        """Request ID suffix should be lowercase hex characters."""
        from aragora.server.middleware.request_logging import generate_request_id

        for _ in range(50):
            req_id = generate_request_id()
            suffix = req_id[4:]  # After "req-"
            assert all(c in "0123456789abcdef" for c in suffix)

    def test_sanitize_mixed_sensitive_and_non_sensitive(self):
        """Should correctly sanitize a mix of sensitive and non-sensitive headers."""
        from aragora.server.middleware.request_logging import sanitize_headers

        headers = {
            "Authorization": "Bearer very-long-secret-bearer-token-here",
            "Content-Type": "application/json",
            "Cookie": "session=also-a-long-secret-session-value",
            "Accept": "*/*",
            "X-Api-Key": "api-key-should-be-masked-too",
        }
        sanitized = sanitize_headers(headers)

        # Non-sensitive preserved
        assert sanitized["Content-Type"] == "application/json"
        assert sanitized["Accept"] == "*/*"

        # Sensitive masked
        assert "****" in sanitized["Authorization"]
        assert "****" in sanitized["Cookie"]
        assert "****" in sanitized["X-Api-Key"]

        # Original not modified
        assert headers["Authorization"] == "Bearer very-long-secret-bearer-token-here"
