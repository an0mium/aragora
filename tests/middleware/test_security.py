"""Tests for security middleware.

Tests cover:
- Security headers generation
- CSP policies (api, standard, development)
- HSTS configuration
- Content length validation
- Query parameter whitelisting
- Client IP extraction with proxy support
- Multipart validation
- Nonce generation
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.middleware.security import (
    CSP_API_STRICT,
    CSP_DEFAULT,
    CSP_DEVELOPMENT,
    CSP_HEADER,
    CSP_REPORT_ONLY_HEADER,
    CSP_WEB_UI,
    HSTS_HEADER,
    HSTS_VALUE,
    SECURITY_HEADERS,
    SecurityConfig,
    SecurityMiddleware,
    ValidationResult,
    apply_security_headers,
    generate_nonce,
    get_security_headers,
)


class TestSecurityHeaders:
    """Tests for security header constants and generation."""

    def test_security_headers_contains_required_headers(self):
        """Verify all required security headers are defined."""
        assert "X-Content-Type-Options" in SECURITY_HEADERS
        assert "X-Frame-Options" in SECURITY_HEADERS
        assert "X-XSS-Protection" in SECURITY_HEADERS
        assert "Referrer-Policy" in SECURITY_HEADERS
        assert "Permissions-Policy" in SECURITY_HEADERS

    def test_security_headers_correct_values(self):
        """Verify security headers have secure values."""
        assert SECURITY_HEADERS["X-Content-Type-Options"] == "nosniff"
        assert SECURITY_HEADERS["X-Frame-Options"] == "DENY"
        assert SECURITY_HEADERS["X-XSS-Protection"] == "1; mode=block"

    def test_hsts_header_defined(self):
        """Verify HSTS header constants are defined."""
        assert HSTS_HEADER == "Strict-Transport-Security"
        assert "max-age=" in HSTS_VALUE
        assert "includeSubDomains" in HSTS_VALUE


class TestCSPPolicies:
    """Tests for Content Security Policy configurations."""

    def test_csp_api_strict_blocks_scripts(self):
        """API strict CSP should block all scripts."""
        assert "default-src 'none'" in CSP_API_STRICT
        assert "frame-ancestors 'none'" in CSP_API_STRICT

    def test_csp_web_ui_allows_self(self):
        """Web UI CSP should allow self-hosted resources."""
        assert "default-src 'self'" in CSP_WEB_UI
        assert "script-src 'self'" in CSP_WEB_UI

    def test_csp_development_permissive(self):
        """Development CSP should allow unsafe-inline for debugging."""
        assert "'unsafe-inline'" in CSP_DEVELOPMENT
        assert "'unsafe-eval'" in CSP_DEVELOPMENT

    def test_csp_default_is_development(self):
        """Default CSP should be development for backwards compatibility."""
        assert CSP_DEFAULT == CSP_DEVELOPMENT


class TestGetSecurityHeaders:
    """Tests for get_security_headers function."""

    def test_returns_base_headers_by_default(self):
        """Default call returns base security headers."""
        headers = get_security_headers()
        for key in SECURITY_HEADERS:
            assert key in headers
            assert headers[key] == SECURITY_HEADERS[key]

    def test_no_hsts_in_development(self):
        """HSTS should not be included in development."""
        headers = get_security_headers(production=False)
        assert HSTS_HEADER not in headers

    def test_hsts_in_production(self):
        """HSTS should be included in production."""
        headers = get_security_headers(production=True, enable_hsts=True)
        assert HSTS_HEADER in headers
        assert headers[HSTS_HEADER] == HSTS_VALUE

    def test_hsts_disabled_in_production(self):
        """HSTS can be disabled even in production."""
        headers = get_security_headers(production=True, enable_hsts=False)
        assert HSTS_HEADER not in headers

    def test_csp_disabled_by_default(self):
        """CSP should not be included by default."""
        headers = get_security_headers()
        assert CSP_HEADER not in headers
        assert CSP_REPORT_ONLY_HEADER not in headers

    def test_csp_api_mode(self):
        """CSP API mode should use strict policy."""
        headers = get_security_headers(enable_csp=True, csp_mode="api")
        assert CSP_HEADER in headers
        assert headers[CSP_HEADER] == CSP_API_STRICT

    def test_csp_standard_mode(self):
        """CSP standard mode should use web UI policy."""
        headers = get_security_headers(enable_csp=True, csp_mode="standard")
        assert CSP_HEADER in headers
        assert headers[CSP_HEADER] == CSP_WEB_UI

    def test_csp_development_mode(self):
        """CSP development mode should use permissive policy."""
        headers = get_security_headers(enable_csp=True, csp_mode="development")
        assert CSP_HEADER in headers
        assert headers[CSP_HEADER] == CSP_DEVELOPMENT

    def test_csp_custom_policy(self):
        """Custom CSP should override mode selection."""
        custom = "default-src 'self'; script-src 'none'"
        headers = get_security_headers(enable_csp=True, custom_csp=custom)
        assert headers[CSP_HEADER] == custom

    def test_csp_with_nonce(self):
        """Nonce should be added to script-src."""
        nonce = "abc123"
        headers = get_security_headers(enable_csp=True, csp_mode="standard", nonce=nonce)
        assert f"'nonce-{nonce}'" in headers[CSP_HEADER]

    def test_csp_with_report_uri(self):
        """Report URI should be added to CSP."""
        report_uri = "https://example.com/csp-report"
        headers = get_security_headers(enable_csp=True, csp_report_uri=report_uri)
        assert f"report-uri {report_uri}" in headers[CSP_HEADER]

    def test_csp_report_only_mode(self):
        """Report-only mode should use different header."""
        headers = get_security_headers(enable_csp=True, report_only=True)
        assert CSP_REPORT_ONLY_HEADER in headers
        assert CSP_HEADER not in headers


class TestApplySecurityHeaders:
    """Tests for apply_security_headers function."""

    def test_applies_headers_to_handler(self):
        """Headers should be applied via send_header method."""
        handler = MagicMock()
        apply_security_headers(handler)

        # Verify send_header was called for each base header
        calls = handler.send_header.call_args_list
        header_names = [call[0][0] for call in calls]

        for key in SECURITY_HEADERS:
            assert key in header_names

    def test_applies_hsts_in_production(self):
        """HSTS header should be applied in production."""
        handler = MagicMock()
        apply_security_headers(handler, production=True)

        calls = handler.send_header.call_args_list
        header_names = [call[0][0] for call in calls]
        assert HSTS_HEADER in header_names


class TestGenerateNonce:
    """Tests for nonce generation."""

    def test_generates_unique_nonces(self):
        """Each call should generate a unique nonce."""
        nonces = [generate_nonce() for _ in range(100)]
        assert len(set(nonces)) == 100

    def test_nonce_is_base64(self):
        """Nonce should be valid base64."""
        import base64

        nonce = generate_nonce()
        # Should not raise
        base64.b64decode(nonce)

    def test_nonce_sufficient_entropy(self):
        """Nonce should have sufficient length for security."""
        nonce = generate_nonce()
        # 16 bytes = 128 bits of entropy, base64 encoded is ~22 chars
        assert len(nonce) >= 20


class TestSecurityConfig:
    """Tests for SecurityConfig dataclass."""

    def test_default_content_limits(self):
        """Default content limits should be reasonable."""
        config = SecurityConfig()
        assert config.max_content_length == 100 * 1024 * 1024  # 100MB
        assert config.max_json_length == 10 * 1024 * 1024  # 10MB
        assert config.max_multipart_parts == 10

    def test_default_trusted_proxies(self):
        """Default trusted proxies should include localhost."""
        config = SecurityConfig()
        assert "127.0.0.1" in config.trusted_proxies
        assert "localhost" in config.trusted_proxies

    @patch.dict(os.environ, {"ARAGORA_TRUSTED_PROXIES": "10.0.0.1,192.168.1.1"})
    def test_trusted_proxies_from_env(self):
        """Trusted proxies should be configurable via environment."""
        config = SecurityConfig()
        assert "10.0.0.1" in config.trusted_proxies
        assert "192.168.1.1" in config.trusted_proxies

    def test_default_query_params(self):
        """Default allowed query params should include common ones."""
        config = SecurityConfig()
        assert "limit" in config.allowed_query_params
        assert "offset" in config.allowed_query_params
        assert "domain" in config.allowed_query_params

    def test_restricted_query_param_values(self):
        """Some params should have restricted allowed values."""
        config = SecurityConfig()
        table_values = config.allowed_query_params.get("table")
        assert table_values is not None
        assert "debates" in table_values
        assert "summary" in table_values


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_ok_result(self):
        """ok() should return valid result."""
        result = ValidationResult.ok()
        assert result.valid is True
        assert result.error_message == ""

    def test_error_result(self):
        """error() should return invalid result with message."""
        result = ValidationResult.error("Something went wrong", 400)
        assert result.valid is False
        assert result.error_message == "Something went wrong"
        assert result.error_code == 400

    def test_error_default_code(self):
        """error() should default to 400 status code."""
        result = ValidationResult.error("Bad request")
        assert result.error_code == 400


class TestSecurityMiddleware:
    """Tests for SecurityMiddleware class."""

    def test_init_with_default_config(self):
        """Middleware should use default config if none provided."""
        middleware = SecurityMiddleware()
        assert middleware.config is not None
        assert isinstance(middleware.config, SecurityConfig)

    def test_init_with_custom_config(self):
        """Middleware should accept custom config."""
        config = SecurityConfig(max_json_length=1024)
        middleware = SecurityMiddleware(config=config)
        assert middleware.config.max_json_length == 1024


class TestValidateContentLength:
    """Tests for content length validation."""

    def test_valid_content_length(self):
        """Valid content length should pass."""
        middleware = SecurityMiddleware()
        headers = {"Content-Length": "1024"}
        result = middleware.validate_content_length(headers)
        assert result.valid is True

    def test_missing_content_length_allowed(self):
        """Missing Content-Length should be allowed (chunked encoding)."""
        middleware = SecurityMiddleware()
        result = middleware.validate_content_length({})
        assert result.valid is True

    def test_invalid_content_length_format(self):
        """Non-numeric Content-Length should fail."""
        middleware = SecurityMiddleware()
        headers = {"Content-Length": "abc"}
        result = middleware.validate_content_length(headers)
        assert result.valid is False
        assert result.error_code == 400

    def test_content_length_exceeds_json_limit(self):
        """Content exceeding JSON limit should fail."""
        middleware = SecurityMiddleware()
        # Default JSON limit is 10MB
        headers = {"Content-Length": str(20 * 1024 * 1024)}
        result = middleware.validate_content_length(headers, is_json=True)
        assert result.valid is False
        assert result.error_code == 413

    def test_content_length_exceeds_upload_limit(self):
        """Content exceeding upload limit should fail."""
        middleware = SecurityMiddleware()
        # Default upload limit is 100MB
        headers = {"Content-Length": str(200 * 1024 * 1024)}
        result = middleware.validate_content_length(headers, is_json=False)
        assert result.valid is False
        assert result.error_code == 413

    def test_custom_max_size_override(self):
        """Custom max_size should override defaults."""
        middleware = SecurityMiddleware()
        headers = {"Content-Length": "2000"}
        result = middleware.validate_content_length(headers, max_size=1000)
        assert result.valid is False

    def test_negative_content_length(self):
        """Negative Content-Length should fail."""
        middleware = SecurityMiddleware()
        headers = {"Content-Length": "-100"}
        result = middleware.validate_content_length(headers)
        assert result.valid is False
        assert result.error_code == 400

    def test_lowercase_header_name(self):
        """Lowercase content-length header should work."""
        middleware = SecurityMiddleware()
        headers = {"content-length": "1024"}
        result = middleware.validate_content_length(headers)
        assert result.valid is True


class TestValidateQueryParams:
    """Tests for query parameter validation."""

    def test_valid_params(self):
        """Valid params should pass."""
        middleware = SecurityMiddleware()
        params = {"limit": ["10"], "offset": ["0"]}
        result = middleware.validate_query_params(params)
        assert result.valid is True

    def test_unknown_param_rejected(self):
        """Unknown params should be rejected."""
        middleware = SecurityMiddleware()
        params = {"unknown_param": ["value"]}
        result = middleware.validate_query_params(params)
        assert result.valid is False
        assert "unknown_param" in result.error_message.lower()

    def test_restricted_param_valid_value(self):
        """Restricted params with valid values should pass."""
        middleware = SecurityMiddleware()
        params = {"table": ["debates"]}
        result = middleware.validate_query_params(params)
        assert result.valid is True

    def test_restricted_param_invalid_value(self):
        """Restricted params with invalid values should fail."""
        middleware = SecurityMiddleware()
        params = {"table": ["invalid_table"]}
        result = middleware.validate_query_params(params)
        assert result.valid is False

    def test_unrestricted_param_any_value(self):
        """Unrestricted params should accept any value."""
        middleware = SecurityMiddleware()
        params = {"limit": ["999999"], "query": ["anything goes here"]}
        result = middleware.validate_query_params(params)
        assert result.valid is True

    def test_empty_params(self):
        """Empty params should pass."""
        middleware = SecurityMiddleware()
        result = middleware.validate_query_params({})
        assert result.valid is True


class TestGetClientIP:
    """Tests for client IP extraction."""

    def test_direct_connection(self):
        """Direct connection should return remote address."""
        middleware = SecurityMiddleware()
        result = middleware.get_client_ip("192.168.1.100", {})
        assert result == "192.168.1.100"

    def test_trusted_proxy_with_xff(self):
        """Trusted proxy should respect X-Forwarded-For."""
        middleware = SecurityMiddleware()
        headers = {"X-Forwarded-For": "10.0.0.1, 192.168.1.1"}
        result = middleware.get_client_ip("127.0.0.1", headers)
        assert result == "10.0.0.1"

    def test_untrusted_proxy_ignores_xff(self):
        """Untrusted proxy should ignore X-Forwarded-For."""
        middleware = SecurityMiddleware()
        headers = {"X-Forwarded-For": "10.0.0.1"}
        result = middleware.get_client_ip("192.168.1.100", headers)
        assert result == "192.168.1.100"

    def test_lowercase_xff_header(self):
        """Lowercase x-forwarded-for should work."""
        middleware = SecurityMiddleware()
        headers = {"x-forwarded-for": "10.0.0.1"}
        result = middleware.get_client_ip("127.0.0.1", headers)
        assert result == "10.0.0.1"

    def test_empty_xff(self):
        """Empty X-Forwarded-For should fall back to remote address."""
        middleware = SecurityMiddleware()
        headers = {"X-Forwarded-For": ""}
        result = middleware.get_client_ip("127.0.0.1", headers)
        assert result == "127.0.0.1"


class TestValidateMultipartParts:
    """Tests for multipart parts validation."""

    def test_valid_part_count(self):
        """Valid part count should pass."""
        middleware = SecurityMiddleware()
        result = middleware.validate_multipart_parts(5)
        assert result.valid is True

    def test_exceeds_max_parts(self):
        """Exceeding max parts should fail."""
        middleware = SecurityMiddleware()
        result = middleware.validate_multipart_parts(100)
        assert result.valid is False
        assert result.error_code == 400

    def test_exact_max_parts(self):
        """Exact max parts should pass."""
        middleware = SecurityMiddleware()
        result = middleware.validate_multipart_parts(10)
        assert result.valid is True


class TestAddAllowedParam:
    """Tests for adding allowed parameters."""

    def test_add_unrestricted_param(self):
        """Adding unrestricted param should allow any value."""
        middleware = SecurityMiddleware()
        middleware.add_allowed_param("custom_param")

        result = middleware.validate_query_params({"custom_param": ["anything"]})
        assert result.valid is True

    def test_add_restricted_param(self):
        """Adding restricted param should validate values."""
        middleware = SecurityMiddleware()
        middleware.add_allowed_param("status", {"active", "inactive"})

        result = middleware.validate_query_params({"status": ["active"]})
        assert result.valid is True

        result = middleware.validate_query_params({"status": ["unknown"]})
        assert result.valid is False


class TestValidateJsonBodySize:
    """Tests for JSON body size validation."""

    def test_valid_body_size(self):
        """Valid body size should pass."""
        middleware = SecurityMiddleware()
        body = b'{"key": "value"}'
        result = middleware.validate_json_body_size(body)
        assert result.valid is True

    def test_exceeds_max_size(self):
        """Exceeding max size should fail."""
        config = SecurityConfig(max_json_length=100)
        middleware = SecurityMiddleware(config=config)
        body = b"x" * 200
        result = middleware.validate_json_body_size(body)
        assert result.valid is False
        assert result.error_code == 413

    def test_empty_body(self):
        """Empty body should pass."""
        middleware = SecurityMiddleware()
        result = middleware.validate_json_body_size(b"")
        assert result.valid is True
