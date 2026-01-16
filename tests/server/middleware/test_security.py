"""
Tests for aragora.server.middleware.security - Security middleware.

Tests cover:
- Security headers (SECURITY_HEADERS, CSP, HSTS)
- get_security_headers() function
- apply_security_headers() function
- generate_nonce() function
- SecurityConfig dataclass
- ValidationResult dataclass
- SecurityMiddleware class (content length, query params)
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


# ===========================================================================
# Test Security Headers Constants
# ===========================================================================


class TestSecurityHeadersConstants:
    """Tests for security header constants."""

    def test_security_headers_contains_required_headers(self):
        """SECURITY_HEADERS should contain essential security headers."""
        from aragora.server.middleware.security import SECURITY_HEADERS

        assert "X-Content-Type-Options" in SECURITY_HEADERS
        assert "X-Frame-Options" in SECURITY_HEADERS
        assert "X-XSS-Protection" in SECURITY_HEADERS
        assert "Referrer-Policy" in SECURITY_HEADERS
        assert "Permissions-Policy" in SECURITY_HEADERS

    def test_security_headers_values(self):
        """SECURITY_HEADERS should have correct values."""
        from aragora.server.middleware.security import SECURITY_HEADERS

        assert SECURITY_HEADERS["X-Content-Type-Options"] == "nosniff"
        assert SECURITY_HEADERS["X-Frame-Options"] == "DENY"
        assert SECURITY_HEADERS["X-XSS-Protection"] == "1; mode=block"

    def test_csp_constants_defined(self):
        """CSP constants should be defined."""
        from aragora.server.middleware.security import (
            CSP_API_STRICT,
            CSP_DEVELOPMENT,
            CSP_WEB_UI,
        )

        assert "default-src" in CSP_API_STRICT
        assert "default-src" in CSP_DEVELOPMENT
        assert "default-src" in CSP_WEB_UI

    def test_csp_api_strict_is_restrictive(self):
        """CSP_API_STRICT should be very restrictive."""
        from aragora.server.middleware.security import CSP_API_STRICT

        assert "default-src 'none'" in CSP_API_STRICT
        assert "frame-ancestors 'none'" in CSP_API_STRICT

    def test_csp_development_is_permissive(self):
        """CSP_DEVELOPMENT should allow unsafe inline/eval for dev tools."""
        from aragora.server.middleware.security import CSP_DEVELOPMENT

        assert "'unsafe-inline'" in CSP_DEVELOPMENT
        assert "'unsafe-eval'" in CSP_DEVELOPMENT


# ===========================================================================
# Test get_security_headers Function
# ===========================================================================


class TestGetSecurityHeaders:
    """Tests for get_security_headers function."""

    def test_returns_base_headers(self):
        """Should return base security headers by default."""
        from aragora.server.middleware.security import (
            SECURITY_HEADERS,
            get_security_headers,
        )

        headers = get_security_headers()

        for key in SECURITY_HEADERS:
            assert key in headers
            assert headers[key] == SECURITY_HEADERS[key]

    def test_production_enables_hsts(self):
        """Should enable HSTS in production mode."""
        from aragora.server.middleware.security import get_security_headers

        headers = get_security_headers(production=True, enable_hsts=True)

        assert "Strict-Transport-Security" in headers
        assert "max-age=" in headers["Strict-Transport-Security"]

    def test_non_production_no_hsts(self):
        """Should not include HSTS when not in production."""
        from aragora.server.middleware.security import get_security_headers

        headers = get_security_headers(production=False, enable_hsts=True)

        assert "Strict-Transport-Security" not in headers

    def test_enable_csp_api_mode(self):
        """Should enable strict CSP in API mode."""
        from aragora.server.middleware.security import get_security_headers

        headers = get_security_headers(enable_csp=True, csp_mode="api")

        assert "Content-Security-Policy" in headers
        assert "default-src 'none'" in headers["Content-Security-Policy"]

    def test_enable_csp_standard_mode(self):
        """Should enable standard CSP for web UI."""
        from aragora.server.middleware.security import get_security_headers

        headers = get_security_headers(enable_csp=True, csp_mode="standard")

        assert "Content-Security-Policy" in headers
        assert "default-src 'self'" in headers["Content-Security-Policy"]

    def test_enable_csp_development_mode(self):
        """Should enable permissive CSP in development mode."""
        from aragora.server.middleware.security import get_security_headers

        headers = get_security_headers(enable_csp=True, csp_mode="development")

        assert "Content-Security-Policy" in headers
        assert "'unsafe-inline'" in headers["Content-Security-Policy"]

    def test_custom_csp(self):
        """Should use custom CSP when provided."""
        from aragora.server.middleware.security import get_security_headers

        custom_csp = "default-src 'self'; script-src 'self'"
        headers = get_security_headers(enable_csp=True, custom_csp=custom_csp)

        assert headers["Content-Security-Policy"] == custom_csp

    def test_csp_with_nonce(self):
        """Should add nonce to script-src when provided."""
        from aragora.server.middleware.security import get_security_headers

        nonce = "abc123"
        headers = get_security_headers(
            enable_csp=True, csp_mode="standard", nonce=nonce
        )

        assert f"'nonce-{nonce}'" in headers["Content-Security-Policy"]

    def test_csp_with_report_uri(self):
        """Should add report-uri directive when provided."""
        from aragora.server.middleware.security import get_security_headers

        report_uri = "https://example.com/csp-report"
        headers = get_security_headers(
            enable_csp=True, csp_report_uri=report_uri
        )

        assert f"report-uri {report_uri}" in headers["Content-Security-Policy"]

    def test_csp_report_only_mode(self):
        """Should use report-only header when enabled."""
        from aragora.server.middleware.security import get_security_headers

        headers = get_security_headers(enable_csp=True, report_only=True)

        assert "Content-Security-Policy-Report-Only" in headers
        assert "Content-Security-Policy" not in headers


# ===========================================================================
# Test apply_security_headers Function
# ===========================================================================


class TestApplySecurityHeaders:
    """Tests for apply_security_headers function."""

    def test_applies_headers_to_handler(self):
        """Should call send_header for each security header."""
        from aragora.server.middleware.security import (
            SECURITY_HEADERS,
            apply_security_headers,
        )

        mock_handler = MagicMock()
        apply_security_headers(mock_handler)

        # Verify send_header was called for each base header
        for name, value in SECURITY_HEADERS.items():
            mock_handler.send_header.assert_any_call(name, value)

    def test_applies_csp_when_enabled(self):
        """Should apply CSP header when enabled."""
        from aragora.server.middleware.security import apply_security_headers

        mock_handler = MagicMock()
        apply_security_headers(mock_handler, enable_csp=True)

        # Check CSP header was sent
        call_args = [call[0] for call in mock_handler.send_header.call_args_list]
        header_names = [arg[0] for arg in call_args]
        assert "Content-Security-Policy" in header_names


# ===========================================================================
# Test generate_nonce Function
# ===========================================================================


class TestGenerateNonce:
    """Tests for generate_nonce function."""

    def test_returns_string(self):
        """Should return a string nonce."""
        from aragora.server.middleware.security import generate_nonce

        nonce = generate_nonce()
        assert isinstance(nonce, str)

    def test_returns_base64_encoded(self):
        """Should return valid base64 string."""
        import base64

        from aragora.server.middleware.security import generate_nonce

        nonce = generate_nonce()
        # Should not raise
        base64.b64decode(nonce)

    def test_returns_unique_values(self):
        """Should return unique nonces on each call."""
        from aragora.server.middleware.security import generate_nonce

        nonces = [generate_nonce() for _ in range(100)]
        assert len(set(nonces)) == 100  # All unique

    def test_sufficient_length(self):
        """Nonce should be of sufficient length for security."""
        from aragora.server.middleware.security import generate_nonce

        nonce = generate_nonce()
        # Base64 of 16 bytes should be ~22 chars
        assert len(nonce) >= 20


# ===========================================================================
# Test SecurityConfig Dataclass
# ===========================================================================


class TestSecurityConfig:
    """Tests for SecurityConfig dataclass."""

    def test_default_values(self):
        """SecurityConfig should have sensible defaults."""
        from aragora.server.middleware.security import SecurityConfig

        config = SecurityConfig()

        assert config.max_content_length == 100 * 1024 * 1024  # 100MB
        assert config.max_json_length == 10 * 1024 * 1024  # 10MB
        assert config.max_multipart_parts == 10

    def test_trusted_proxies_default(self):
        """Should have localhost in trusted proxies by default."""
        from aragora.server.middleware.security import SecurityConfig

        config = SecurityConfig()

        assert "127.0.0.1" in config.trusted_proxies

    def test_allowed_query_params_defined(self):
        """Should have common query params defined."""
        from aragora.server.middleware.security import SecurityConfig

        config = SecurityConfig()

        assert "limit" in config.allowed_query_params
        assert "offset" in config.allowed_query_params
        assert "domain" in config.allowed_query_params


# ===========================================================================
# Test ValidationResult Dataclass
# ===========================================================================


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_ok_returns_valid(self):
        """ok() should return a valid result."""
        from aragora.server.middleware.security import ValidationResult

        result = ValidationResult.ok()

        assert result.valid is True
        assert result.error_message == ""

    def test_error_returns_invalid(self):
        """error() should return an invalid result."""
        from aragora.server.middleware.security import ValidationResult

        result = ValidationResult.error("Test error", code=413)

        assert result.valid is False
        assert result.error_message == "Test error"
        assert result.error_code == 413

    def test_error_default_code(self):
        """error() should default to 400 status code."""
        from aragora.server.middleware.security import ValidationResult

        result = ValidationResult.error("Bad request")

        assert result.error_code == 400


# ===========================================================================
# Test SecurityMiddleware Class
# ===========================================================================


class TestSecurityMiddlewareContentLength:
    """Tests for SecurityMiddleware.validate_content_length."""

    def test_valid_content_length(self):
        """Should accept valid content length."""
        from aragora.server.middleware.security import SecurityMiddleware

        middleware = SecurityMiddleware()
        result = middleware.validate_content_length(
            headers={"Content-Length": "1024"}, is_json=True
        )

        assert result.valid is True

    def test_no_content_length_header(self):
        """Should accept requests without Content-Length (chunked encoding)."""
        from aragora.server.middleware.security import SecurityMiddleware

        middleware = SecurityMiddleware()
        result = middleware.validate_content_length(headers={}, is_json=True)

        assert result.valid is True

    def test_content_length_too_large(self):
        """Should reject content length exceeding limit."""
        from aragora.server.middleware.security import SecurityMiddleware

        middleware = SecurityMiddleware()
        # 10MB + 1 (default JSON limit is 10MB)
        result = middleware.validate_content_length(
            headers={"Content-Length": str(10 * 1024 * 1024 + 1)}, is_json=True
        )

        assert result.valid is False
        assert result.error_code == 413
        assert "too large" in result.error_message.lower()

    def test_invalid_content_length_format(self):
        """Should reject non-numeric content length."""
        from aragora.server.middleware.security import SecurityMiddleware

        middleware = SecurityMiddleware()
        result = middleware.validate_content_length(
            headers={"Content-Length": "not-a-number"}, is_json=True
        )

        assert result.valid is False
        assert result.error_code == 400

    def test_negative_content_length(self):
        """Should reject negative content length."""
        from aragora.server.middleware.security import SecurityMiddleware

        middleware = SecurityMiddleware()
        result = middleware.validate_content_length(
            headers={"Content-Length": "-100"}, is_json=True
        )

        assert result.valid is False
        assert result.error_code == 400

    def test_custom_max_size(self):
        """Should use custom max size when provided."""
        from aragora.server.middleware.security import SecurityMiddleware

        middleware = SecurityMiddleware()
        result = middleware.validate_content_length(
            headers={"Content-Length": "1000"}, max_size=500
        )

        assert result.valid is False
        assert result.error_code == 413

    def test_lowercase_content_length_header(self):
        """Should handle lowercase content-length header."""
        from aragora.server.middleware.security import SecurityMiddleware

        middleware = SecurityMiddleware()
        result = middleware.validate_content_length(
            headers={"content-length": "1024"}, is_json=True
        )

        assert result.valid is True


class TestSecurityMiddlewareQueryParams:
    """Tests for SecurityMiddleware.validate_query_params."""

    def test_valid_query_params(self):
        """Should accept allowed query parameters."""
        from aragora.server.middleware.security import SecurityMiddleware

        middleware = SecurityMiddleware()
        result = middleware.validate_query_params(
            params={"limit": ["10"], "offset": ["0"]}
        )

        assert result.valid is True

    def test_unknown_query_param(self):
        """Should reject unknown query parameters."""
        from aragora.server.middleware.security import SecurityMiddleware

        middleware = SecurityMiddleware()
        result = middleware.validate_query_params(
            params={"unknown_param": ["value"]}
        )

        assert result.valid is False
        assert result.error_code == 400
        assert "Unknown query parameter" in result.error_message

    def test_restricted_value_valid(self):
        """Should accept valid restricted values."""
        from aragora.server.middleware.security import SecurityMiddleware

        middleware = SecurityMiddleware()
        result = middleware.validate_query_params(
            params={"table": ["debates"]}
        )

        assert result.valid is True

    def test_restricted_value_invalid(self):
        """Should reject invalid restricted values."""
        from aragora.server.middleware.security import SecurityMiddleware

        middleware = SecurityMiddleware()
        result = middleware.validate_query_params(
            params={"table": ["invalid_table"]}
        )

        assert result.valid is False
        assert result.error_code == 400

    def test_empty_params(self):
        """Should accept empty query params."""
        from aragora.server.middleware.security import SecurityMiddleware

        middleware = SecurityMiddleware()
        result = middleware.validate_query_params(params={})

        assert result.valid is True


class TestSecurityMiddlewareWithCustomConfig:
    """Tests for SecurityMiddleware with custom configuration."""

    def test_custom_content_limits(self):
        """Should respect custom content limits."""
        from aragora.server.middleware.security import (
            SecurityConfig,
            SecurityMiddleware,
        )

        config = SecurityConfig(max_json_length=1000)
        middleware = SecurityMiddleware(config)

        result = middleware.validate_content_length(
            headers={"Content-Length": "2000"}, is_json=True
        )

        assert result.valid is False

    def test_custom_allowed_params(self):
        """Should respect custom allowed query params."""
        from aragora.server.middleware.security import (
            SecurityConfig,
            SecurityMiddleware,
        )

        config = SecurityConfig(allowed_query_params={"custom_param": None})
        middleware = SecurityMiddleware(config)

        # Should accept custom param
        result = middleware.validate_query_params(
            params={"custom_param": ["value"]}
        )
        assert result.valid is True

        # Should reject non-allowed param
        result = middleware.validate_query_params(
            params={"limit": ["10"]}
        )
        assert result.valid is False
