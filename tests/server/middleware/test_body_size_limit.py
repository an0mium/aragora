"""
Tests for aragora.server.middleware.body_size_limit - Request body size limit middleware.

Tests cover:
- BodySizeLimitConfig dataclass and configuration
- BodySizeCheckResult dataclass
- BodySizeLimitMiddleware class (content length validation)
- LimitedBodyReader for chunked transfers
- with_body_size_limit decorator
- Convenience functions (check_body_size, get_body_size_stats)
"""

from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import pytest


# ===========================================================================
# Test Constants
# ===========================================================================


class TestConstants:
    """Tests for module constants."""

    def test_default_max_request_size(self):
        """DEFAULT_MAX_REQUEST_SIZE should be 10MB."""
        from aragora.server.middleware.body_size_limit import DEFAULT_MAX_REQUEST_SIZE

        assert DEFAULT_MAX_REQUEST_SIZE == 10 * 1024 * 1024  # 10MB

    def test_http_payload_too_large(self):
        """HTTP_PAYLOAD_TOO_LARGE should be 413."""
        from aragora.server.middleware.body_size_limit import HTTP_PAYLOAD_TOO_LARGE

        assert HTTP_PAYLOAD_TOO_LARGE == 413

    def test_default_large_endpoints_defined(self):
        """DEFAULT_LARGE_ENDPOINTS should contain file upload paths."""
        from aragora.server.middleware.body_size_limit import DEFAULT_LARGE_ENDPOINTS

        assert "/api/documents/upload" in DEFAULT_LARGE_ENDPOINTS
        assert "/api/files/upload" in DEFAULT_LARGE_ENDPOINTS

    def test_default_small_endpoints_defined(self):
        """DEFAULT_SMALL_ENDPOINTS should contain auth paths."""
        from aragora.server.middleware.body_size_limit import DEFAULT_SMALL_ENDPOINTS

        assert "/api/auth/" in DEFAULT_SMALL_ENDPOINTS
        assert "/api/login" in DEFAULT_SMALL_ENDPOINTS


# ===========================================================================
# Test BodySizeLimitConfig
# ===========================================================================


class TestBodySizeLimitConfig:
    """Tests for BodySizeLimitConfig dataclass."""

    def test_default_max_request_size(self):
        """Config should have default max request size."""
        from aragora.server.middleware.body_size_limit import (
            BodySizeLimitConfig,
            DEFAULT_MAX_REQUEST_SIZE,
        )

        config = BodySizeLimitConfig()
        assert config.max_request_size == DEFAULT_MAX_REQUEST_SIZE

    def test_custom_max_request_size(self):
        """Config should accept custom max request size."""
        from aragora.server.middleware.body_size_limit import BodySizeLimitConfig

        config = BodySizeLimitConfig(max_request_size=5 * 1024 * 1024)
        assert config.max_request_size == 5 * 1024 * 1024

    def test_endpoint_limits_initialized(self):
        """Config should have default endpoint limits."""
        from aragora.server.middleware.body_size_limit import BodySizeLimitConfig

        config = BodySizeLimitConfig()
        assert len(config.endpoint_limits) > 0

    def test_get_limit_for_endpoint_default(self):
        """get_limit_for_endpoint should return default for unknown paths."""
        from aragora.server.middleware.body_size_limit import BodySizeLimitConfig

        config = BodySizeLimitConfig()
        limit = config.get_limit_for_endpoint("/api/some/random/path")
        assert limit == config.max_request_size

    def test_get_limit_for_endpoint_large(self):
        """get_limit_for_endpoint should return larger limit for upload paths."""
        from aragora.server.middleware.body_size_limit import BodySizeLimitConfig

        config = BodySizeLimitConfig()
        limit = config.get_limit_for_endpoint("/api/documents/upload")
        assert limit == 100 * 1024 * 1024  # 100MB

    def test_get_limit_for_endpoint_small(self):
        """get_limit_for_endpoint should return smaller limit for auth paths."""
        from aragora.server.middleware.body_size_limit import BodySizeLimitConfig

        config = BodySizeLimitConfig()
        limit = config.get_limit_for_endpoint("/api/auth/token")
        assert limit == 1 * 1024 * 1024  # 1MB

    def test_get_limit_for_endpoint_longest_match(self):
        """get_limit_for_endpoint should use longest matching pattern."""
        from aragora.server.middleware.body_size_limit import BodySizeLimitConfig

        config = BodySizeLimitConfig(
            endpoint_limits={
                "/api": 5 * 1024 * 1024,
                "/api/special": 20 * 1024 * 1024,
            }
        )
        limit = config.get_limit_for_endpoint("/api/special/endpoint")
        assert limit == 20 * 1024 * 1024

    def test_config_from_environment(self):
        """Config should read from ARAGORA_MAX_REQUEST_SIZE environment variable."""
        from aragora.server.middleware.body_size_limit import BodySizeLimitConfig

        with patch.dict("os.environ", {"ARAGORA_MAX_REQUEST_SIZE": "20971520"}):
            config = BodySizeLimitConfig()
            assert config.max_request_size == 20 * 1024 * 1024


# ===========================================================================
# Test BodySizeCheckResult
# ===========================================================================


class TestBodySizeCheckResult:
    """Tests for BodySizeCheckResult dataclass."""

    def test_ok_result(self):
        """ok() should return allowed result."""
        from aragora.server.middleware.body_size_limit import BodySizeCheckResult

        result = BodySizeCheckResult.ok()
        assert result.allowed is True
        assert result.message == ""

    def test_too_large_result_with_size(self):
        """too_large() should include size details when requested."""
        from aragora.server.middleware.body_size_limit import BodySizeCheckResult

        result = BodySizeCheckResult.too_large(
            content_length=20 * 1024 * 1024,
            max_allowed=10 * 1024 * 1024,
            include_size=True,
        )
        assert result.allowed is False
        assert result.status_code == 413
        assert "20.00MB" in result.message
        assert "10.00MB" in result.message
        assert result.content_length == 20 * 1024 * 1024
        assert result.max_allowed == 10 * 1024 * 1024

    def test_too_large_result_without_size(self):
        """too_large() should hide size details when requested."""
        from aragora.server.middleware.body_size_limit import BodySizeCheckResult

        result = BodySizeCheckResult.too_large(
            content_length=20 * 1024 * 1024,
            max_allowed=10 * 1024 * 1024,
            include_size=False,
        )
        assert result.allowed is False
        assert result.message == "Request body too large"

    def test_invalid_content_length_result(self):
        """invalid_content_length() should return 400 status."""
        from aragora.server.middleware.body_size_limit import BodySizeCheckResult

        result = BodySizeCheckResult.invalid_content_length("not-a-number")
        assert result.allowed is False
        assert result.status_code == 400
        assert "not-a-number" in result.message

    def test_negative_content_length_result(self):
        """negative_content_length() should return 400 status."""
        from aragora.server.middleware.body_size_limit import BodySizeCheckResult

        result = BodySizeCheckResult.negative_content_length()
        assert result.allowed is False
        assert result.status_code == 400
        assert "negative" in result.message.lower()


# ===========================================================================
# Test BodySizeLimitMiddleware
# ===========================================================================


class TestBodySizeLimitMiddleware:
    """Tests for BodySizeLimitMiddleware class."""

    def test_check_content_length_valid(self):
        """check_content_length should allow valid requests."""
        from aragora.server.middleware.body_size_limit import BodySizeLimitMiddleware

        middleware = BodySizeLimitMiddleware()
        result = middleware.check_content_length(
            {"Content-Length": "1000"},
            path="/api/test",
        )
        assert result.allowed is True

    def test_check_content_length_too_large(self):
        """check_content_length should reject too large requests."""
        from aragora.server.middleware.body_size_limit import (
            BodySizeLimitConfig,
            BodySizeLimitMiddleware,
        )

        config = BodySizeLimitConfig(max_request_size=1000)
        middleware = BodySizeLimitMiddleware(config)
        result = middleware.check_content_length(
            {"Content-Length": "2000"},
            path="/api/test",
        )
        assert result.allowed is False
        assert result.status_code == 413

    def test_check_content_length_no_header(self):
        """check_content_length should allow requests without Content-Length."""
        from aragora.server.middleware.body_size_limit import BodySizeLimitMiddleware

        middleware = BodySizeLimitMiddleware()
        result = middleware.check_content_length({}, path="/api/test")
        assert result.allowed is True

    def test_check_content_length_invalid_header(self):
        """check_content_length should reject invalid Content-Length."""
        from aragora.server.middleware.body_size_limit import BodySizeLimitMiddleware

        middleware = BodySizeLimitMiddleware()
        result = middleware.check_content_length(
            {"Content-Length": "not-a-number"},
            path="/api/test",
        )
        assert result.allowed is False
        assert result.status_code == 400

    def test_check_content_length_negative(self):
        """check_content_length should reject negative Content-Length."""
        from aragora.server.middleware.body_size_limit import BodySizeLimitMiddleware

        middleware = BodySizeLimitMiddleware()
        result = middleware.check_content_length(
            {"Content-Length": "-100"},
            path="/api/test",
        )
        assert result.allowed is False
        assert result.status_code == 400

    def test_check_content_length_case_insensitive(self):
        """check_content_length should handle lowercase header."""
        from aragora.server.middleware.body_size_limit import BodySizeLimitMiddleware

        middleware = BodySizeLimitMiddleware()
        result = middleware.check_content_length(
            {"content-length": "1000"},
            path="/api/test",
        )
        assert result.allowed is True

    def test_check_content_length_with_override(self):
        """check_content_length should respect max_size_override."""
        from aragora.server.middleware.body_size_limit import BodySizeLimitMiddleware

        middleware = BodySizeLimitMiddleware()
        result = middleware.check_content_length(
            {"Content-Length": "5000"},
            path="/api/test",
            max_size_override=1000,
        )
        assert result.allowed is False
        assert result.status_code == 413

    def test_check_content_length_endpoint_specific(self):
        """check_content_length should use endpoint-specific limits."""
        from aragora.server.middleware.body_size_limit import BodySizeLimitMiddleware

        middleware = BodySizeLimitMiddleware()
        # Auth endpoints have 1MB limit
        result = middleware.check_content_length(
            {"Content-Length": str(2 * 1024 * 1024)},  # 2MB
            path="/api/auth/token",
        )
        assert result.allowed is False

    def test_violation_count_tracking(self):
        """Middleware should track violation counts."""
        from aragora.server.middleware.body_size_limit import (
            BodySizeLimitConfig,
            BodySizeLimitMiddleware,
        )

        config = BodySizeLimitConfig(max_request_size=1000, log_violations=False)
        middleware = BodySizeLimitMiddleware(config)
        middleware.reset_violation_count()

        # Trigger violations
        middleware.check_content_length({"Content-Length": "2000"})
        middleware.check_content_length({"Content-Length": "3000"})

        assert middleware.get_violation_count() == 2

    def test_wrap_body_reader(self):
        """wrap_body_reader should return LimitedBodyReader."""
        from aragora.server.middleware.body_size_limit import (
            BodySizeLimitMiddleware,
            LimitedBodyReader,
        )

        middleware = BodySizeLimitMiddleware()
        body = io.BytesIO(b"test data")
        reader = middleware.wrap_body_reader(body, path="/api/test")
        assert isinstance(reader, LimitedBodyReader)


# ===========================================================================
# Test LimitedBodyReader
# ===========================================================================


class TestLimitedBodyReader:
    """Tests for LimitedBodyReader class."""

    def test_read_within_limit(self):
        """read() should work when within limit."""
        from aragora.server.middleware.body_size_limit import LimitedBodyReader

        body = io.BytesIO(b"hello world")
        reader = LimitedBodyReader(body, max_size=100)
        data = reader.read()
        assert data == b"hello world"

    def test_read_exceeds_limit(self):
        """read() should raise when limit exceeded."""
        from aragora.server.middleware.body_size_limit import (
            BodySizeLimitExceeded,
            LimitedBodyReader,
        )

        body = io.BytesIO(b"hello world")
        reader = LimitedBodyReader(body, max_size=5)
        with pytest.raises(BodySizeLimitExceeded) as exc_info:
            reader.read()
        assert exc_info.value.bytes_read > 5
        assert exc_info.value.max_size == 5

    def test_read_with_size_argument(self):
        """read(size) should work when within limit."""
        from aragora.server.middleware.body_size_limit import LimitedBodyReader

        body = io.BytesIO(b"hello world")
        reader = LimitedBodyReader(body, max_size=100)
        data = reader.read(5)
        assert data == b"hello"

    def test_read_chunked_exceeds_limit(self):
        """Chunked reads should raise when cumulative size exceeds limit."""
        from aragora.server.middleware.body_size_limit import (
            BodySizeLimitExceeded,
            LimitedBodyReader,
        )

        body = io.BytesIO(b"hello world")
        reader = LimitedBodyReader(body, max_size=8)
        reader.read(5)  # 5 bytes - ok
        reader.read(3)  # 8 bytes - ok
        with pytest.raises(BodySizeLimitExceeded):
            reader.read(1)  # 9 bytes - exceeds

    def test_readline_within_limit(self):
        """readline() should work when within limit."""
        from aragora.server.middleware.body_size_limit import LimitedBodyReader

        body = io.BytesIO(b"line1\nline2\n")
        reader = LimitedBodyReader(body, max_size=100)
        line = reader.readline()
        assert line == b"line1\n"

    def test_readline_exceeds_limit(self):
        """readline() should raise when limit exceeded."""
        from aragora.server.middleware.body_size_limit import (
            BodySizeLimitExceeded,
            LimitedBodyReader,
        )

        body = io.BytesIO(b"this is a very long line\n")
        reader = LimitedBodyReader(body, max_size=5)
        with pytest.raises(BodySizeLimitExceeded):
            reader.readline()

    def test_bytes_read_property(self):
        """bytes_read should track total bytes read."""
        from aragora.server.middleware.body_size_limit import LimitedBodyReader

        body = io.BytesIO(b"hello world")
        reader = LimitedBodyReader(body, max_size=100)
        reader.read(5)
        assert reader.bytes_read == 5
        reader.read(3)
        assert reader.bytes_read == 8

    def test_on_exceeded_callback(self):
        """on_exceeded callback should be called when limit exceeded."""
        from aragora.server.middleware.body_size_limit import (
            BodySizeLimitExceeded,
            LimitedBodyReader,
        )

        callback_called = []

        def on_exceeded(bytes_read: int):
            callback_called.append(bytes_read)

        body = io.BytesIO(b"hello world")
        reader = LimitedBodyReader(body, max_size=5, on_exceeded=on_exceeded)
        with pytest.raises(BodySizeLimitExceeded):
            reader.read()
        assert len(callback_called) == 1
        assert callback_called[0] > 5


# ===========================================================================
# Test BodySizeLimitExceeded Exception
# ===========================================================================


class TestBodySizeLimitExceeded:
    """Tests for BodySizeLimitExceeded exception."""

    def test_exception_attributes(self):
        """Exception should have bytes_read and max_size attributes."""
        from aragora.server.middleware.body_size_limit import BodySizeLimitExceeded

        exc = BodySizeLimitExceeded(bytes_read=2000, max_size=1000)
        assert exc.bytes_read == 2000
        assert exc.max_size == 1000

    def test_exception_message(self):
        """Exception message should include size details."""
        from aragora.server.middleware.body_size_limit import BodySizeLimitExceeded

        exc = BodySizeLimitExceeded(bytes_read=2000, max_size=1000)
        assert "2000" in str(exc)
        assert "limit exceeded" in str(exc).lower()


# ===========================================================================
# Test with_body_size_limit Decorator
# ===========================================================================


class TestWithBodySizeLimitDecorator:
    """Tests for with_body_size_limit decorator."""

    def test_decorator_allows_valid_request(self):
        """Decorator should allow requests within limit."""
        from aragora.server.middleware.body_size_limit import with_body_size_limit

        @with_body_size_limit(max_bytes=10000)
        def handler(self, request):
            return "success"

        mock_handler = MagicMock()
        mock_handler.headers = {"Content-Length": "1000"}
        mock_handler.path = "/api/test"

        result = handler(None, mock_handler)
        assert result == "success"

    def test_decorator_rejects_large_request(self):
        """Decorator should reject requests exceeding limit."""
        from aragora.server.middleware.body_size_limit import with_body_size_limit

        @with_body_size_limit(max_bytes=1000)
        def handler(self, request):
            return "success"

        mock_handler = MagicMock()
        mock_handler.headers = {"Content-Length": "2000"}
        mock_handler.path = "/api/test"
        mock_handler.wfile = io.BytesIO()

        result = handler(None, mock_handler)
        assert result is None
        mock_handler.send_response.assert_called_with(413)


# ===========================================================================
# Test Convenience Functions
# ===========================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_check_body_size(self):
        """check_body_size should validate content length."""
        from aragora.server.middleware.body_size_limit import check_body_size

        result = check_body_size(
            {"Content-Length": "1000"},
            path="/api/test",
        )
        assert result.allowed is True

    def test_check_body_size_too_large(self):
        """check_body_size should reject too large content."""
        from aragora.server.middleware.body_size_limit import check_body_size

        result = check_body_size(
            {"Content-Length": "1000000000"},  # 1GB
            path="/api/test",
        )
        assert result.allowed is False

    def test_get_body_size_stats(self):
        """get_body_size_stats should return configuration info."""
        from aragora.server.middleware.body_size_limit import get_body_size_stats

        stats = get_body_size_stats()
        assert "max_request_size" in stats
        assert "max_request_size_mb" in stats
        assert "endpoint_limits_count" in stats
        assert "log_violations" in stats


# ===========================================================================
# Test Configuration Functions
# ===========================================================================


class TestConfigurationFunctions:
    """Tests for configuration functions."""

    def test_get_body_size_config(self):
        """get_body_size_config should return global config."""
        from aragora.server.middleware.body_size_limit import (
            BodySizeLimitConfig,
            get_body_size_config,
        )

        config = get_body_size_config()
        assert isinstance(config, BodySizeLimitConfig)

    def test_configure_body_size_limit(self):
        """configure_body_size_limit should update global config."""
        from aragora.server.middleware.body_size_limit import (
            configure_body_size_limit,
            get_body_size_config,
            reset_body_size_config,
        )

        # Reset first to get clean state
        reset_body_size_config()

        configure_body_size_limit(max_request_size=5 * 1024 * 1024)
        config = get_body_size_config()
        assert config.max_request_size == 5 * 1024 * 1024

        # Cleanup
        reset_body_size_config()

    def test_configure_endpoint_limits(self):
        """configure_body_size_limit should update endpoint limits."""
        from aragora.server.middleware.body_size_limit import (
            configure_body_size_limit,
            get_body_size_config,
            reset_body_size_config,
        )

        reset_body_size_config()

        configure_body_size_limit(endpoint_limits={"/api/custom": 50 * 1024 * 1024})
        config = get_body_size_config()
        assert "/api/custom" in config.endpoint_limits
        assert config.endpoint_limits["/api/custom"] == 50 * 1024 * 1024

        reset_body_size_config()

    def test_reset_body_size_config(self):
        """reset_body_size_config should clear global config."""
        from aragora.server.middleware.body_size_limit import (
            configure_body_size_limit,
            get_body_size_config,
            reset_body_size_config,
        )

        configure_body_size_limit(max_request_size=1000)
        reset_body_size_config()

        # Getting config again should create fresh default
        config = get_body_size_config()
        assert config.max_request_size == 10 * 1024 * 1024  # Default

        reset_body_size_config()


# ===========================================================================
# Test Logging
# ===========================================================================


class TestViolationLogging:
    """Tests for violation logging."""

    def test_violation_logged_when_enabled(self):
        """Violations should be logged when log_violations is True."""
        from aragora.server.middleware.body_size_limit import (
            BodySizeLimitConfig,
            BodySizeLimitMiddleware,
        )

        config = BodySizeLimitConfig(max_request_size=1000, log_violations=True)
        middleware = BodySizeLimitMiddleware(config)

        with patch("aragora.server.middleware.body_size_limit.logger") as mock_logger:
            middleware.check_content_length(
                {"Content-Length": "2000"},
                path="/api/test",
            )
            mock_logger.warning.assert_called_once()

    def test_violation_not_logged_when_disabled(self):
        """Violations should not be logged when log_violations is False."""
        from aragora.server.middleware.body_size_limit import (
            BodySizeLimitConfig,
            BodySizeLimitMiddleware,
        )

        config = BodySizeLimitConfig(max_request_size=1000, log_violations=False)
        middleware = BodySizeLimitMiddleware(config)

        with patch("aragora.server.middleware.body_size_limit.logger") as mock_logger:
            middleware.check_content_length(
                {"Content-Length": "2000"},
                path="/api/test",
            )
            mock_logger.warning.assert_not_called()


# ===========================================================================
# Integration Tests
# ===========================================================================


class TestIntegration:
    """Integration tests for body size limit middleware."""

    def test_middleware_import_from_package(self):
        """Middleware should be importable from package."""
        from aragora.server.middleware import (
            BodySizeCheckResult,
            BodySizeLimitConfig,
            BodySizeLimitExceeded,
            BodySizeLimitMiddleware,
            check_body_size,
            configure_body_size_limit,
            get_body_size_config,
            with_body_size_limit,
        )

        assert BodySizeCheckResult is not None
        assert BodySizeLimitConfig is not None
        assert BodySizeLimitExceeded is not None
        assert BodySizeLimitMiddleware is not None
        assert check_body_size is not None
        assert configure_body_size_limit is not None
        assert get_body_size_config is not None
        assert with_body_size_limit is not None

    def test_end_to_end_chunked_read(self):
        """Test complete chunked read flow."""
        from aragora.server.middleware.body_size_limit import (
            BodySizeLimitExceeded,
            BodySizeLimitMiddleware,
        )

        # Create middleware
        middleware = BodySizeLimitMiddleware()

        # Simulate chunked transfer
        body_data = b"A" * 1000
        body = io.BytesIO(body_data)

        reader = middleware.wrap_body_reader(body, max_size_override=500)

        # Read until limit exceeded
        with pytest.raises(BodySizeLimitExceeded):
            reader.read()

    def test_file_upload_has_larger_limit(self):
        """File upload endpoints should have larger limits."""
        from aragora.server.middleware.body_size_limit import BodySizeLimitMiddleware

        middleware = BodySizeLimitMiddleware()

        # Regular endpoint - 50MB should fail with default 10MB limit
        result = middleware.check_content_length(
            {"Content-Length": str(50 * 1024 * 1024)},
            path="/api/debates",
        )
        assert result.allowed is False

        # Upload endpoint - 50MB should pass with 100MB limit
        result = middleware.check_content_length(
            {"Content-Length": str(50 * 1024 * 1024)},
            path="/api/documents/upload",
        )
        assert result.allowed is True
