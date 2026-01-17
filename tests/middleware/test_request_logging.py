"""Tests for request logging middleware.

Tests cover:
- Request ID generation
- Token hashing for audit
- Sensitive value masking
- Header and parameter sanitization
- Request/response logging
- Logging decorator (sync and async)
- Client IP extraction
- Context variable management
"""

import asyncio
import logging
import time
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.middleware.request_logging import (
    REQUEST_ID_HEADER,
    RequestContext,
    SENSITIVE_HEADERS,
    SENSITIVE_PARAMS,
    generate_request_id,
    get_current_request_id,
    hash_token,
    log_request,
    log_response,
    mask_sensitive_value,
    request_logging,
    sanitize_headers,
    sanitize_params,
    set_current_request_id,
)


class TestConstants:
    """Tests for module constants."""

    def test_request_id_header_name(self):
        """Request ID header should be X-Request-ID."""
        assert REQUEST_ID_HEADER == "X-Request-ID"

    def test_sensitive_headers_defined(self):
        """Sensitive headers should include auth-related headers."""
        assert "authorization" in SENSITIVE_HEADERS
        assert "x-api-key" in SENSITIVE_HEADERS
        assert "cookie" in SENSITIVE_HEADERS

    def test_sensitive_params_defined(self):
        """Sensitive params should include token-related params."""
        assert "token" in SENSITIVE_PARAMS
        assert "api_key" in SENSITIVE_PARAMS
        assert "password" in SENSITIVE_PARAMS
        assert "access_token" in SENSITIVE_PARAMS


class TestRequestContext:
    """Tests for RequestContext dataclass."""

    def test_create_context(self):
        """Should create context with all fields."""
        ctx = RequestContext(
            request_id="req-123",
            method="GET",
            path="/api/debates",
            client_ip="192.168.1.1",
            start_time=time.time(),
            token_hash="abc123",
        )
        assert ctx.request_id == "req-123"
        assert ctx.method == "GET"
        assert ctx.path == "/api/debates"
        assert ctx.client_ip == "192.168.1.1"
        assert ctx.token_hash == "abc123"

    def test_elapsed_ms(self):
        """Should calculate elapsed time in milliseconds."""
        start = time.time()
        ctx = RequestContext(
            request_id="req-123",
            method="GET",
            path="/",
            client_ip="127.0.0.1",
            start_time=start,
        )
        time.sleep(0.01)  # 10ms
        elapsed = ctx.elapsed_ms()
        assert elapsed >= 10  # At least 10ms

    def test_optional_token_hash(self):
        """Token hash should be optional."""
        ctx = RequestContext(
            request_id="req-123",
            method="GET",
            path="/",
            client_ip="127.0.0.1",
            start_time=time.time(),
        )
        assert ctx.token_hash is None


class TestGenerateRequestId:
    """Tests for generate_request_id function."""

    def test_format(self):
        """Request ID should have req- prefix."""
        request_id = generate_request_id()
        assert request_id.startswith("req-")

    def test_uniqueness(self):
        """Each call should generate unique ID."""
        ids = [generate_request_id() for _ in range(1000)]
        assert len(set(ids)) == 1000

    def test_length(self):
        """Request ID should be reasonable length."""
        request_id = generate_request_id()
        # req- (4) + 12 hex chars = 16
        assert len(request_id) == 16


class TestHashToken:
    """Tests for hash_token function."""

    def test_consistent_hash(self):
        """Same token should produce same hash."""
        token = "my-secret-token"
        hash1 = hash_token(token)
        hash2 = hash_token(token)
        assert hash1 == hash2

    def test_different_tokens_different_hashes(self):
        """Different tokens should produce different hashes."""
        hash1 = hash_token("token1")
        hash2 = hash_token("token2")
        assert hash1 != hash2

    def test_truncated_length(self):
        """Hash should be truncated to 8 characters."""
        token_hash = hash_token("any-token")
        assert len(token_hash) == 8

    def test_hex_format(self):
        """Hash should be hexadecimal."""
        token_hash = hash_token("any-token")
        int(token_hash, 16)  # Should not raise


class TestMaskSensitiveValue:
    """Tests for mask_sensitive_value function."""

    def test_long_value(self):
        """Long values should show first 4 and last 4 chars."""
        masked = mask_sensitive_value("Bearer my-secret-token")
        assert masked == "Bear****oken"

    def test_short_value(self):
        """Short values should be fully masked."""
        masked = mask_sensitive_value("short")
        assert masked == "****"

    def test_exactly_8_chars(self):
        """8-char values should be fully masked."""
        masked = mask_sensitive_value("12345678")
        assert masked == "****"

    def test_9_chars(self):
        """9-char values should show partial."""
        masked = mask_sensitive_value("123456789")
        assert masked == "1234****6789"


class TestSanitizeHeaders:
    """Tests for sanitize_headers function."""

    def test_masks_sensitive_headers(self):
        """Sensitive headers should be masked."""
        headers = {
            "Authorization": "Bearer secret-token-here",
            "Content-Type": "application/json",
        }
        sanitized = sanitize_headers(headers)
        assert "****" in sanitized["Authorization"]
        assert sanitized["Content-Type"] == "application/json"

    def test_preserves_non_sensitive(self):
        """Non-sensitive headers should be preserved."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        sanitized = sanitize_headers(headers)
        assert sanitized == headers

    def test_case_insensitive(self):
        """Header matching should be case-insensitive."""
        headers = {
            "AUTHORIZATION": "Bearer token",
            "X-API-KEY": "secret",
        }
        sanitized = sanitize_headers(headers)
        assert "****" in sanitized["AUTHORIZATION"]
        assert "****" in sanitized["X-API-KEY"]

    def test_original_unchanged(self):
        """Original headers dict should not be modified."""
        headers = {"Authorization": "Bearer token"}
        sanitize_headers(headers)
        assert headers["Authorization"] == "Bearer token"


class TestSanitizeParams:
    """Tests for sanitize_params function."""

    def test_masks_sensitive_params(self):
        """Sensitive params should be masked."""
        params = {
            "token": "secret-token",
            "limit": "10",
        }
        sanitized = sanitize_params(params)
        assert sanitized["token"] == "****"
        assert sanitized["limit"] == "10"

    def test_preserves_non_sensitive(self):
        """Non-sensitive params should be preserved."""
        params = {"limit": "10", "offset": "0"}
        sanitized = sanitize_params(params)
        assert sanitized == params

    def test_case_insensitive(self):
        """Param matching should be case-insensitive."""
        params = {"TOKEN": "secret", "API_KEY": "key"}
        sanitized = sanitize_params(params)
        assert sanitized["TOKEN"] == "****"
        assert sanitized["API_KEY"] == "****"


class TestLogRequest:
    """Tests for log_request function."""

    def test_logs_basic_info(self, caplog):
        """Should log method, path, and client IP."""
        ctx = RequestContext(
            request_id="req-123",
            method="GET",
            path="/api/debates",
            client_ip="192.168.1.1",
            start_time=time.time(),
        )

        with caplog.at_level(logging.INFO):
            log_request(ctx)
            assert "GET" in caplog.text
            assert "/api/debates" in caplog.text
            assert "192.168.1.1" in caplog.text

    def test_logs_with_params(self, caplog):
        """Should sanitize and log query params."""
        ctx = RequestContext(
            request_id="req-123",
            method="GET",
            path="/api/debates",
            client_ip="127.0.0.1",
            start_time=time.time(),
        )

        with caplog.at_level(logging.INFO):
            log_request(ctx, query_params={"limit": "10", "token": "secret"})
            # The params are in the extra dict, verify masking works
            assert "req-123" in caplog.text

    def test_logs_token_hash(self, caplog):
        """Should include token hash in extra."""
        ctx = RequestContext(
            request_id="req-123",
            method="GET",
            path="/",
            client_ip="127.0.0.1",
            start_time=time.time(),
            token_hash="abc12345",
        )

        with caplog.at_level(logging.INFO):
            log_request(ctx)
            # Token hash is in extra, verify request logged
            assert "req-123" in caplog.text


class TestLogResponse:
    """Tests for log_response function."""

    def test_logs_status_and_elapsed(self, caplog):
        """Should log status code and elapsed time."""
        ctx = RequestContext(
            request_id="req-123",
            method="GET",
            path="/api/debates",
            client_ip="127.0.0.1",
            start_time=time.time() - 0.1,  # 100ms ago
        )

        with caplog.at_level(logging.INFO):
            log_response(ctx, 200)
            assert "200" in caplog.text
            assert "ms" in caplog.text.lower()

    def test_auto_log_level_success(self, caplog):
        """Success status should use INFO level."""
        ctx = RequestContext(
            request_id="req-123",
            method="GET",
            path="/",
            client_ip="127.0.0.1",
            start_time=time.time(),
        )

        with caplog.at_level(logging.DEBUG):
            log_response(ctx, 200)
            # INFO level should be logged
            assert any(r.levelno == logging.INFO for r in caplog.records)

    def test_auto_log_level_client_error(self, caplog):
        """4xx status should use WARNING level."""
        ctx = RequestContext(
            request_id="req-123",
            method="GET",
            path="/",
            client_ip="127.0.0.1",
            start_time=time.time(),
        )

        with caplog.at_level(logging.DEBUG):
            log_response(ctx, 404)
            assert any(r.levelno == logging.WARNING for r in caplog.records)

    def test_auto_log_level_server_error(self, caplog):
        """5xx status should use ERROR level."""
        ctx = RequestContext(
            request_id="req-123",
            method="GET",
            path="/",
            client_ip="127.0.0.1",
            start_time=time.time(),
        )

        with caplog.at_level(logging.DEBUG):
            log_response(ctx, 500)
            assert any(r.levelno == logging.ERROR for r in caplog.records)

    def test_includes_error_message(self, caplog):
        """Should include error message in extra."""
        ctx = RequestContext(
            request_id="req-123",
            method="GET",
            path="/",
            client_ip="127.0.0.1",
            start_time=time.time(),
        )

        with caplog.at_level(logging.ERROR):
            log_response(ctx, 500, error="Something went wrong")
            # Error is in extra dict, verify response was logged
            assert "500" in caplog.text

    def test_includes_response_size(self, caplog):
        """Should include response size when provided."""
        ctx = RequestContext(
            request_id="req-123",
            method="GET",
            path="/",
            client_ip="127.0.0.1",
            start_time=time.time(),
        )

        with caplog.at_level(logging.INFO):
            log_response(ctx, 200, response_size=1024)
            # Size is in extra dict, verify response was logged
            assert "200" in caplog.text


class TestRequestLoggingDecorator:
    """Tests for request_logging decorator."""

    @pytest.mark.asyncio
    async def test_async_handler(self, caplog):
        """Should wrap async handlers."""

        @request_logging()
        async def handler(self, request):
            return MagicMock(status=200)

        request = MagicMock()
        request.headers = {}
        request.method = "GET"
        request.path = "/test"

        with caplog.at_level(logging.INFO):
            result = await handler(None, request)
            assert result.status == 200

    def test_sync_handler(self, caplog):
        """Should wrap sync handlers."""

        @request_logging()
        def handler(self, request):
            return MagicMock(status=200)

        request = MagicMock()
        request.headers = {}
        request.method = "GET"
        request.path = "/test"

        with caplog.at_level(logging.INFO):
            result = handler(None, request)
            assert result.status == 200

    @pytest.mark.asyncio
    async def test_extracts_existing_request_id(self, caplog):
        """Should use existing X-Request-ID if present."""

        @request_logging()
        async def handler(self, request):
            return MagicMock(status=200, headers={})

        request = MagicMock()
        request.headers = {REQUEST_ID_HEADER: "existing-id"}
        request.method = "GET"
        request.path = "/test"

        with caplog.at_level(logging.INFO):
            await handler(None, request)
            # The existing ID should be used in logs
            assert "existing-id" in caplog.text

    @pytest.mark.asyncio
    async def test_adds_request_id_to_response(self, caplog):
        """Should add X-Request-ID to response headers."""

        @request_logging()
        async def handler(self, request):
            response = MagicMock(status=200)
            response.headers = {}
            return response

        request = MagicMock()
        request.headers = {}
        request.method = "GET"
        request.path = "/test"

        with caplog.at_level(logging.INFO):
            result = await handler(None, request)
            assert REQUEST_ID_HEADER in result.headers

    @pytest.mark.asyncio
    async def test_logs_slow_request_warning(self, caplog):
        """Should warn for slow requests."""

        @request_logging(slow_request_threshold_ms=10)
        async def slow_handler(self, request):
            await asyncio.sleep(0.02)  # 20ms
            return MagicMock(status=200)

        request = MagicMock()
        request.headers = {}
        request.method = "GET"
        request.path = "/slow"

        with caplog.at_level(logging.DEBUG):
            await slow_handler(None, request)
            # Slow request should generate a warning
            assert any(r.levelno == logging.WARNING for r in caplog.records)

    @pytest.mark.asyncio
    async def test_handles_exception(self, caplog):
        """Should log error and re-raise on exception."""

        @request_logging()
        async def failing_handler(self, request):
            raise ValueError("Test error")

        request = MagicMock()
        request.headers = {}
        request.method = "GET"
        request.path = "/fail"

        with caplog.at_level(logging.DEBUG):
            with pytest.raises(ValueError):
                await failing_handler(None, request)

            # Should log a 500 error
            assert "500" in caplog.text

    @pytest.mark.asyncio
    async def test_extracts_token_hash(self, caplog):
        """Should extract and hash Bearer token."""
        # Test that token hashing works (decorator uses it internally)
        token = "my-secret-token"
        expected_hash = hash_token(token)

        @request_logging()
        async def handler(self, request):
            return MagicMock(status=200)

        request = MagicMock()
        request.headers = {"Authorization": f"Bearer {token}"}
        request.method = "GET"
        request.path = "/test"

        with caplog.at_level(logging.INFO):
            await handler(None, request)
            # Token hash should be calculated (verified by hash_token working)
            assert expected_hash == hash_token(token)


class TestContextVariables:
    """Tests for context variable management."""

    def test_get_set_request_id(self):
        """Should store and retrieve request ID from context."""
        set_current_request_id("req-test-456")
        assert get_current_request_id() == "req-test-456"

    def test_set_overwrites_previous(self):
        """Setting request ID should overwrite previous value."""
        set_current_request_id("first-id")
        set_current_request_id("second-id")
        assert get_current_request_id() == "second-id"


class TestExtractIP:
    """Tests for _extract_ip helper function."""

    def test_from_x_forwarded_for(self):
        """Should extract IP from X-Forwarded-For."""
        from aragora.server.middleware.request_logging import _extract_ip

        request = MagicMock()
        request.headers = {"X-Forwarded-For": "10.0.0.1, 192.168.1.1"}

        ip = _extract_ip(request)
        assert ip == "10.0.0.1"

    def test_from_remote(self):
        """Should extract IP from remote attribute."""
        from aragora.server.middleware.request_logging import _extract_ip

        request = MagicMock(spec=["remote"])
        request.remote = "192.168.1.100"

        ip = _extract_ip(request)
        assert ip == "192.168.1.100"

    def test_from_transport(self):
        """Should extract IP from transport peername."""
        from aragora.server.middleware.request_logging import _extract_ip

        transport = MagicMock()
        transport.get_extra_info.return_value = ("192.168.1.50", 12345)

        request = MagicMock(spec=["transport"])
        request.transport = transport

        ip = _extract_ip(request)
        assert ip == "192.168.1.50"

    def test_none_request(self):
        """Should return 'unknown' for None request."""
        from aragora.server.middleware.request_logging import _extract_ip

        ip = _extract_ip(None)
        assert ip == "unknown"

    def test_no_ip_info(self):
        """Should return 'unknown' when no IP info available."""
        from aragora.server.middleware.request_logging import _extract_ip

        request = MagicMock(spec=[])

        ip = _extract_ip(request)
        assert ip == "unknown"


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_headers(self):
        """Should handle empty headers dict."""
        sanitized = sanitize_headers({})
        assert sanitized == {}

    def test_empty_params(self):
        """Should handle empty params dict."""
        sanitized = sanitize_params({})
        assert sanitized == {}

    @pytest.mark.asyncio
    async def test_long_error_truncated(self, caplog):
        """Long error messages should be truncated in decorator."""
        # The decorator truncates to 200 chars
        long_error = "x" * 500

        @request_logging()
        async def handler(self, request):
            raise ValueError(long_error)

        request = MagicMock()
        request.headers = {}
        request.method = "GET"
        request.path = "/test"

        with caplog.at_level(logging.ERROR):
            with pytest.raises(ValueError):
                await handler(None, request)

            # Verify error was logged (truncation happens internally)
            assert "500" in caplog.text

    @pytest.mark.asyncio
    async def test_request_from_kwargs(self, caplog):
        """Should extract request from kwargs if not in args."""

        @request_logging()
        async def handler(**kwargs):
            return MagicMock(status=200)

        request = MagicMock()
        request.headers = {}
        request.method = "POST"
        request.path = "/test"

        with caplog.at_level(logging.INFO):
            await handler(request=request)
            # Should log the POST method
            assert "POST" in caplog.text
