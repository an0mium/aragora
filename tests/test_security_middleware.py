"""Tests for security middleware."""

import pytest

from aragora.server.middleware.security import (
    SecurityMiddleware,
    SecurityConfig,
    ValidationResult,
)


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_ok_result(self):
        """ok() returns valid result."""
        result = ValidationResult.ok()
        assert result.valid is True
        assert result.error_message == ""

    def test_error_result(self):
        """error() returns invalid result with message."""
        result = ValidationResult.error("Test error", code=403)
        assert result.valid is False
        assert result.error_message == "Test error"
        assert result.error_code == 403

    def test_error_default_code(self):
        """error() uses 400 as default code."""
        result = ValidationResult.error("Bad request")
        assert result.error_code == 400


class TestSecurityConfig:
    """Tests for SecurityConfig."""

    def test_default_values(self):
        """Default config has sensible values."""
        config = SecurityConfig()
        assert config.max_content_length == 100 * 1024 * 1024  # 100MB
        assert config.max_json_length == 10 * 1024 * 1024  # 10MB
        assert config.max_multipart_parts == 10
        assert "limit" in config.allowed_query_params

    def test_custom_values(self):
        """Config accepts custom values."""
        config = SecurityConfig(
            max_content_length=50 * 1024 * 1024,
            max_json_length=5 * 1024 * 1024,
        )
        assert config.max_content_length == 50 * 1024 * 1024
        assert config.max_json_length == 5 * 1024 * 1024


class TestSecurityMiddleware:
    """Tests for SecurityMiddleware."""

    @pytest.fixture
    def middleware(self):
        """Create middleware with default config."""
        return SecurityMiddleware()

    @pytest.fixture
    def small_limit_middleware(self):
        """Create middleware with small limits for testing."""
        config = SecurityConfig(
            max_content_length=1000,
            max_json_length=500,
        )
        return SecurityMiddleware(config)

    # Content length validation tests

    def test_validate_content_length_ok(self, middleware):
        """Valid content length passes."""
        headers = {"Content-Length": "1024"}
        result = middleware.validate_content_length(headers)
        assert result.valid is True

    def test_validate_content_length_no_header(self, middleware):
        """Missing Content-Length passes (could be chunked)."""
        result = middleware.validate_content_length({})
        assert result.valid is True

    def test_validate_content_length_too_large(self, small_limit_middleware):
        """Content exceeding limit fails."""
        headers = {"Content-Length": "10000"}
        result = small_limit_middleware.validate_content_length(headers, is_json=True)
        assert result.valid is False
        assert result.error_code == 413
        assert "too large" in result.error_message.lower()

    def test_validate_content_length_invalid(self, middleware):
        """Invalid Content-Length fails."""
        headers = {"Content-Length": "not-a-number"}
        result = middleware.validate_content_length(headers)
        assert result.valid is False
        assert result.error_code == 400

    def test_validate_content_length_negative(self, middleware):
        """Negative Content-Length fails."""
        headers = {"Content-Length": "-100"}
        result = middleware.validate_content_length(headers)
        assert result.valid is False
        assert result.error_code == 400

    def test_validate_content_length_uses_json_limit(self, small_limit_middleware):
        """Uses JSON limit when is_json=True."""
        headers = {"Content-Length": "600"}  # > 500 JSON limit, < 1000 content limit
        result = small_limit_middleware.validate_content_length(headers, is_json=True)
        assert result.valid is False  # Exceeds JSON limit

    def test_validate_content_length_uses_content_limit(self, small_limit_middleware):
        """Uses content limit when is_json=False."""
        headers = {"Content-Length": "600"}
        result = small_limit_middleware.validate_content_length(headers, is_json=False)
        assert result.valid is True  # Under content limit

    # Query parameter validation tests

    def test_validate_query_params_ok(self, middleware):
        """Valid query params pass."""
        params = {"limit": ["10"], "offset": ["0"]}
        result = middleware.validate_query_params(params)
        assert result.valid is True

    def test_validate_query_params_unknown(self, middleware):
        """Unknown params fail."""
        params = {"unknown_param": ["value"]}
        result = middleware.validate_query_params(params)
        assert result.valid is False
        assert "unknown_param" in result.error_message.lower()

    def test_validate_query_params_invalid_value(self, middleware):
        """Invalid restricted values fail."""
        params = {"table": ["invalid_table"]}
        result = middleware.validate_query_params(params)
        assert result.valid is False
        assert "invalid value" in result.error_message.lower()

    def test_validate_query_params_valid_restricted(self, middleware):
        """Valid restricted values pass."""
        params = {"table": ["summary"]}
        result = middleware.validate_query_params(params)
        assert result.valid is True

    # Client IP tests

    def test_get_client_ip_direct(self, middleware):
        """Returns direct IP when not from trusted proxy."""
        ip = middleware.get_client_ip("1.2.3.4", {"X-Forwarded-For": "5.6.7.8"})
        assert ip == "1.2.3.4"  # Ignores X-Forwarded-For

    def test_get_client_ip_trusted_proxy(self, middleware):
        """Returns X-Forwarded-For IP from trusted proxy."""
        ip = middleware.get_client_ip(
            "127.0.0.1", {"X-Forwarded-For": "5.6.7.8, 9.10.11.12"}  # Trusted proxy
        )
        assert ip == "5.6.7.8"  # First IP in chain

    def test_get_client_ip_no_forwarded(self, middleware):
        """Returns remote address when no X-Forwarded-For."""
        ip = middleware.get_client_ip("127.0.0.1", {})
        assert ip == "127.0.0.1"

    # Multipart validation tests

    def test_validate_multipart_ok(self, middleware):
        """Valid part count passes."""
        result = middleware.validate_multipart_parts(5)
        assert result.valid is True

    def test_validate_multipart_too_many(self, middleware):
        """Too many parts fails."""
        result = middleware.validate_multipart_parts(100)
        assert result.valid is False
        assert "too many" in result.error_message.lower()

    # JSON body validation tests

    def test_validate_json_body_ok(self, small_limit_middleware):
        """Valid JSON size passes."""
        body = b'{"key": "value"}'
        result = small_limit_middleware.validate_json_body_size(body)
        assert result.valid is True

    def test_validate_json_body_too_large(self, small_limit_middleware):
        """Large JSON fails."""
        body = b"x" * 1000  # > 500 limit
        result = small_limit_middleware.validate_json_body_size(body)
        assert result.valid is False
        assert result.error_code == 413

    # Dynamic param addition tests

    def test_add_allowed_param(self, middleware):
        """Can add new allowed parameters."""
        middleware.add_allowed_param("custom_param")
        params = {"custom_param": ["value"]}
        result = middleware.validate_query_params(params)
        assert result.valid is True

    def test_add_allowed_param_with_values(self, middleware):
        """Can add new restricted parameters."""
        middleware.add_allowed_param("status", {"active", "inactive"})

        # Valid value passes
        result = middleware.validate_query_params({"status": ["active"]})
        assert result.valid is True

        # Invalid value fails
        result = middleware.validate_query_params({"status": ["invalid"]})
        assert result.valid is False
