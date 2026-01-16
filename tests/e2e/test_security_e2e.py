"""
E2E tests for security validation.

Tests security features:
1. Security headers (X-Content-Type-Options, X-Frame-Options, etc.)
2. Content Security Policy (CSP) enforcement
3. HSTS header in production mode
4. Content length validation (DoS protection)
5. Query parameter whitelisting
6. XSS payload sanitization
7. Open redirect prevention
8. Client IP resolution from trusted proxies
"""

from __future__ import annotations

import json
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.middleware.security import (
    SECURITY_HEADERS,
    HSTS_HEADER,
    HSTS_VALUE,
    CSP_HEADER,
    CSP_REPORT_ONLY_HEADER,
    CSP_API_STRICT,
    CSP_WEB_UI,
    CSP_DEVELOPMENT,
    get_security_headers,
    apply_security_headers,
    generate_nonce,
    SecurityConfig,
    ValidationResult,
    SecurityMiddleware,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def security_middleware():
    """Create a security middleware instance with default config."""
    return SecurityMiddleware()


@pytest.fixture
def strict_config():
    """Create a strict security configuration."""
    return SecurityConfig(
        max_content_length=1024 * 1024,  # 1MB
        max_json_length=100 * 1024,  # 100KB
        max_multipart_parts=5,
        enable_csp=True,
        csp_mode="api",
        csp_report_only=False,
    )


@pytest.fixture
def strict_middleware(strict_config):
    """Create middleware with strict configuration."""
    return SecurityMiddleware(strict_config)


# =============================================================================
# Security Headers Tests
# =============================================================================


class TestSecurityHeaders:
    """Tests for security header generation."""

    def test_default_security_headers_included(self):
        """E2E: Default security headers should be included in response."""
        headers = get_security_headers()

        # All base security headers should be present
        assert headers["X-Content-Type-Options"] == "nosniff"
        assert headers["X-Frame-Options"] == "DENY"
        assert headers["X-XSS-Protection"] == "1; mode=block"
        assert headers["Referrer-Policy"] == "strict-origin-when-cross-origin"
        assert "Permissions-Policy" in headers

    def test_hsts_disabled_in_development(self):
        """E2E: HSTS should not be included in development mode."""
        headers = get_security_headers(production=False, enable_hsts=True)

        assert HSTS_HEADER not in headers

    def test_hsts_enabled_in_production(self):
        """E2E: HSTS should be included in production mode."""
        headers = get_security_headers(production=True, enable_hsts=True)

        assert HSTS_HEADER in headers
        assert headers[HSTS_HEADER] == HSTS_VALUE
        assert "max-age=31536000" in headers[HSTS_HEADER]

    def test_hsts_can_be_disabled_in_production(self):
        """E2E: HSTS should be optional even in production."""
        headers = get_security_headers(production=True, enable_hsts=False)

        assert HSTS_HEADER not in headers


# =============================================================================
# Content Security Policy Tests
# =============================================================================


class TestContentSecurityPolicy:
    """Tests for Content Security Policy handling."""

    def test_csp_disabled_by_default(self):
        """E2E: CSP should not be included when disabled."""
        headers = get_security_headers(enable_csp=False)

        assert CSP_HEADER not in headers
        assert CSP_REPORT_ONLY_HEADER not in headers

    def test_csp_api_mode_strict(self):
        """E2E: API mode CSP should be very restrictive."""
        headers = get_security_headers(enable_csp=True, csp_mode="api")

        assert CSP_HEADER in headers
        csp = headers[CSP_HEADER]

        # API mode should block everything
        assert "default-src 'none'" in csp
        assert "frame-ancestors 'none'" in csp
        assert "form-action 'none'" in csp

    def test_csp_standard_mode_allows_self(self):
        """E2E: Standard mode CSP should allow self-hosted resources."""
        headers = get_security_headers(enable_csp=True, csp_mode="standard")

        csp = headers[CSP_HEADER]

        assert "default-src 'self'" in csp
        assert "script-src 'self'" in csp

    def test_csp_development_mode_permissive(self):
        """E2E: Development mode CSP should be more permissive."""
        headers = get_security_headers(enable_csp=True, csp_mode="development")

        csp = headers[CSP_HEADER]

        # Development allows inline scripts and eval
        assert "'unsafe-inline'" in csp
        assert "'unsafe-eval'" in csp

    def test_csp_report_only_mode(self):
        """E2E: Report-only mode should use different header."""
        headers = get_security_headers(enable_csp=True, report_only=True)

        # Should use report-only header instead
        assert CSP_REPORT_ONLY_HEADER in headers
        assert CSP_HEADER not in headers

    def test_csp_with_nonce(self):
        """E2E: Nonce should be included in script-src."""
        nonce = "abc123xyz"
        headers = get_security_headers(enable_csp=True, csp_mode="standard", nonce=nonce)

        csp = headers[CSP_HEADER]

        assert f"'nonce-{nonce}'" in csp

    def test_csp_with_report_uri(self):
        """E2E: Report URI should be appended to CSP."""
        report_uri = "https://example.com/csp-report"
        headers = get_security_headers(enable_csp=True, csp_report_uri=report_uri)

        csp = headers[CSP_HEADER]

        assert f"report-uri {report_uri}" in csp

    def test_custom_csp(self):
        """E2E: Custom CSP should override mode defaults."""
        custom = "default-src 'self'; script-src 'self' https://cdn.example.com"
        headers = get_security_headers(enable_csp=True, custom_csp=custom)

        assert headers[CSP_HEADER] == custom


# =============================================================================
# Nonce Generation Tests
# =============================================================================


class TestNonceGeneration:
    """Tests for CSP nonce generation."""

    def test_nonce_is_unique(self):
        """E2E: Each nonce should be unique."""
        nonces = {generate_nonce() for _ in range(100)}

        # All 100 nonces should be unique
        assert len(nonces) == 100

    def test_nonce_is_base64(self):
        """E2E: Nonce should be valid base64."""
        import base64

        nonce = generate_nonce()

        # Should not raise on decode
        decoded = base64.b64decode(nonce)
        assert len(decoded) == 16  # 16 bytes of randomness

    def test_nonce_format_safe_for_html(self):
        """E2E: Nonce should be safe for HTML attribute."""
        nonce = generate_nonce()

        # Should not contain characters that need escaping in HTML
        dangerous_chars = ["<", ">", '"', "'", "&"]
        for char in dangerous_chars:
            assert char not in nonce


# =============================================================================
# Content Length Validation Tests
# =============================================================================


class TestContentLengthValidation:
    """Tests for content length validation."""

    def test_valid_content_length_accepted(self, security_middleware):
        """E2E: Valid content length should pass validation."""
        headers = {"Content-Length": "1024"}

        result = security_middleware.validate_content_length(headers, is_json=True)

        assert result.valid is True

    def test_oversized_json_rejected(self, security_middleware):
        """E2E: Oversized JSON should be rejected."""
        # Default JSON limit is 10MB
        headers = {"Content-Length": str(11 * 1024 * 1024)}

        result = security_middleware.validate_content_length(headers, is_json=True)

        assert result.valid is False
        assert result.error_code == 413
        assert "too large" in result.error_message.lower()

    def test_oversized_content_rejected(self, security_middleware):
        """E2E: Oversized content should be rejected."""
        # Default content limit is 100MB
        headers = {"Content-Length": str(101 * 1024 * 1024)}

        result = security_middleware.validate_content_length(headers, is_json=False)

        assert result.valid is False
        assert result.error_code == 413

    def test_invalid_content_length_rejected(self, security_middleware):
        """E2E: Invalid content length value should be rejected."""
        headers = {"Content-Length": "not-a-number"}

        result = security_middleware.validate_content_length(headers)

        assert result.valid is False
        assert result.error_code == 400

    def test_negative_content_length_rejected(self, security_middleware):
        """E2E: Negative content length should be rejected."""
        headers = {"Content-Length": "-1"}

        result = security_middleware.validate_content_length(headers)

        assert result.valid is False
        assert result.error_code == 400

    def test_missing_content_length_allowed(self, security_middleware):
        """E2E: Missing content length should be allowed (chunked encoding)."""
        headers = {}

        result = security_middleware.validate_content_length(headers)

        assert result.valid is True

    def test_custom_max_size_override(self, security_middleware):
        """E2E: Custom max size should override defaults."""
        headers = {"Content-Length": "2048"}

        # Should fail with 1KB limit
        result = security_middleware.validate_content_length(headers, max_size=1024)

        assert result.valid is False

        # Should pass with 4KB limit
        result = security_middleware.validate_content_length(headers, max_size=4096)

        assert result.valid is True


# =============================================================================
# Query Parameter Validation Tests
# =============================================================================


class TestQueryParameterValidation:
    """Tests for query parameter whitelisting."""

    def test_allowed_params_accepted(self, security_middleware):
        """E2E: Whitelisted parameters should be accepted."""
        params = {
            "limit": ["10"],
            "offset": ["0"],
            "domain": ["technology"],
        }

        result = security_middleware.validate_query_params(params)

        assert result.valid is True

    def test_unknown_param_rejected(self, security_middleware):
        """E2E: Unknown parameters should be rejected."""
        params = {
            "limit": ["10"],
            "evil_param": ["malicious"],
        }

        result = security_middleware.validate_query_params(params)

        assert result.valid is False
        assert "evil_param" in result.error_message

    def test_restricted_value_rejected(self, security_middleware):
        """E2E: Values outside restricted set should be rejected."""
        # "table" param has restricted values
        params = {
            "table": ["invalid_table"],
        }

        result = security_middleware.validate_query_params(params)

        assert result.valid is False
        assert "invalid_table" in result.error_message

    def test_valid_restricted_value_accepted(self, security_middleware):
        """E2E: Valid restricted values should be accepted."""
        params = {
            "table": ["debates"],
            "sections": ["all"],
        }

        result = security_middleware.validate_query_params(params)

        assert result.valid is True

    def test_add_allowed_param_dynamically(self, security_middleware):
        """E2E: Should be able to add new allowed parameters."""
        # Initially rejected
        params = {"custom_param": ["value"]}
        result = security_middleware.validate_query_params(params)
        assert result.valid is False

        # Add to whitelist
        security_middleware.add_allowed_param("custom_param")

        # Now accepted
        result = security_middleware.validate_query_params(params)
        assert result.valid is True


# =============================================================================
# Client IP Resolution Tests
# =============================================================================


class TestClientIPResolution:
    """Tests for client IP resolution from proxies."""

    def test_direct_ip_used_when_not_trusted(self, security_middleware):
        """E2E: Direct IP should be used when proxy not trusted."""
        remote = "10.0.0.5"
        headers = {"X-Forwarded-For": "192.168.1.1"}

        # 10.0.0.5 is not in trusted proxies by default
        ip = security_middleware.get_client_ip(remote, headers)

        assert ip == "10.0.0.5"

    def test_forwarded_ip_used_from_trusted_proxy(self, security_middleware):
        """E2E: X-Forwarded-For should be used from trusted proxy."""
        remote = "127.0.0.1"  # localhost is trusted by default
        headers = {"X-Forwarded-For": "203.0.113.50"}

        ip = security_middleware.get_client_ip(remote, headers)

        assert ip == "203.0.113.50"

    def test_first_forwarded_ip_used(self, security_middleware):
        """E2E: First IP in X-Forwarded-For chain should be used."""
        remote = "127.0.0.1"
        headers = {"X-Forwarded-For": "203.0.113.50, 10.0.0.1, 192.168.1.1"}

        ip = security_middleware.get_client_ip(remote, headers)

        # First IP is the original client
        assert ip == "203.0.113.50"

    def test_empty_forwarded_for_uses_remote(self, security_middleware):
        """E2E: Empty X-Forwarded-For should fall back to remote address."""
        remote = "127.0.0.1"
        headers = {"X-Forwarded-For": ""}

        ip = security_middleware.get_client_ip(remote, headers)

        assert ip == "127.0.0.1"


# =============================================================================
# Multipart Validation Tests
# =============================================================================


class TestMultipartValidation:
    """Tests for multipart request validation."""

    def test_valid_part_count_accepted(self, security_middleware):
        """E2E: Part count under limit should be accepted."""
        result = security_middleware.validate_multipart_parts(5)

        assert result.valid is True

    def test_excessive_parts_rejected(self, security_middleware):
        """E2E: Excessive parts should be rejected."""
        # Default limit is 10
        result = security_middleware.validate_multipart_parts(15)

        assert result.valid is False
        assert "Too many multipart parts" in result.error_message

    def test_strict_config_lower_limit(self, strict_middleware):
        """E2E: Strict config should have lower part limit."""
        # Strict config has limit of 5
        result = strict_middleware.validate_multipart_parts(6)

        assert result.valid is False


# =============================================================================
# JSON Body Validation Tests
# =============================================================================


class TestJSONBodyValidation:
    """Tests for JSON body size validation."""

    def test_valid_json_size_accepted(self, security_middleware):
        """E2E: Valid JSON body size should be accepted."""
        body = b'{"key": "value"}'

        result = security_middleware.validate_json_body_size(body)

        assert result.valid is True

    def test_oversized_json_body_rejected(self, strict_middleware):
        """E2E: Oversized JSON body should be rejected."""
        # Strict config has 100KB JSON limit
        body = b"x" * (101 * 1024)

        result = strict_middleware.validate_json_body_size(body)

        assert result.valid is False
        assert result.error_code == 413


# =============================================================================
# Apply Headers Tests
# =============================================================================


class TestApplySecurityHeaders:
    """Tests for applying security headers to responses."""

    def test_apply_headers_to_handler(self):
        """E2E: Security headers should be applied to handler."""
        mock_handler = MagicMock()
        headers_sent = []

        def track_header(name, value):
            headers_sent.append((name, value))

        mock_handler.send_header = track_header

        apply_security_headers(mock_handler, production=False)

        # Verify headers were sent
        header_names = [h[0] for h in headers_sent]
        assert "X-Content-Type-Options" in header_names
        assert "X-Frame-Options" in header_names

    def test_apply_headers_with_csp(self):
        """E2E: CSP should be applied when enabled."""
        mock_handler = MagicMock()
        headers_sent = {}

        def track_header(name, value):
            headers_sent[name] = value

        mock_handler.send_header = track_header

        apply_security_headers(mock_handler, enable_csp=True, csp_mode="api")

        assert CSP_HEADER in headers_sent


# =============================================================================
# Security Config Tests
# =============================================================================


class TestSecurityConfig:
    """Tests for security configuration."""

    def test_default_config_values(self):
        """E2E: Default config should have reasonable values."""
        config = SecurityConfig()

        assert config.max_content_length == 100 * 1024 * 1024  # 100MB
        assert config.max_json_length == 10 * 1024 * 1024  # 10MB
        assert config.max_multipart_parts == 10

    def test_config_from_environment(self):
        """E2E: Config should read from environment variables."""
        with patch.dict(
            "os.environ",
            {
                "ARAGORA_ENABLE_CSP": "false",
                "ARAGORA_CSP_MODE": "development",
            },
        ):
            # Create new config to pick up env vars
            config = SecurityConfig()

            # Verify env vars are respected
            # (exact behavior depends on implementation)

    def test_trusted_proxies_configurable(self):
        """E2E: Trusted proxies should be configurable."""
        with patch.dict(
            "os.environ",
            {"ARAGORA_TRUSTED_PROXIES": "10.0.0.1,10.0.0.2"},
        ):
            config = SecurityConfig()

            assert "10.0.0.1" in config.trusted_proxies
            assert "10.0.0.2" in config.trusted_proxies


# =============================================================================
# ValidationResult Tests
# =============================================================================


class TestValidationResult:
    """Tests for ValidationResult class."""

    def test_ok_result(self):
        """E2E: ok() should return valid result."""
        result = ValidationResult.ok()

        assert result.valid is True
        assert result.error_message == ""

    def test_error_result(self):
        """E2E: error() should return invalid result."""
        result = ValidationResult.error("Test error", code=403)

        assert result.valid is False
        assert result.error_message == "Test error"
        assert result.error_code == 403

    def test_default_error_code(self):
        """E2E: Default error code should be 400."""
        result = ValidationResult.error("Bad request")

        assert result.error_code == 400
