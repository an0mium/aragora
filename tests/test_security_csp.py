"""
Tests for CSP (Content Security Policy) hardening.

Covers:
- CSP policy modes (api, standard, development)
- Nonce generation and integration
- Report URI configuration
- Report-only mode
"""

import pytest

from aragora.server.middleware.security import (
    get_security_headers,
    generate_nonce,
    CSP_API_STRICT,
    CSP_WEB_UI,
    CSP_DEVELOPMENT,
    CSP_HEADER,
    CSP_REPORT_ONLY_HEADER,
)


class TestCSPModes:
    """Tests for different CSP modes."""

    def test_api_strict_mode_disallows_scripts(self):
        """API strict mode should have default-src 'none'."""
        headers = get_security_headers(enable_csp=True, csp_mode="api")

        assert CSP_HEADER in headers
        csp = headers[CSP_HEADER]
        assert "default-src 'none'" in csp
        assert "script-src" not in csp  # No scripts allowed
        assert "frame-ancestors 'none'" in csp

    def test_standard_mode_allows_self_scripts(self):
        """Standard mode should allow self-hosted scripts."""
        headers = get_security_headers(enable_csp=True, csp_mode="standard")

        csp = headers[CSP_HEADER]
        assert "script-src 'self'" in csp
        assert "'strict-dynamic'" in csp
        assert "upgrade-insecure-requests" in csp

    def test_development_mode_is_permissive(self):
        """Development mode should allow unsafe-inline and unsafe-eval."""
        headers = get_security_headers(enable_csp=True, csp_mode="development")

        csp = headers[CSP_HEADER]
        assert "'unsafe-inline'" in csp
        assert "'unsafe-eval'" in csp

    def test_csp_disabled_by_default(self):
        """CSP should be disabled by default."""
        headers = get_security_headers()

        assert CSP_HEADER not in headers

    def test_custom_csp_overrides_mode(self):
        """Custom CSP should override mode selection."""
        custom = "default-src 'self'; script-src 'self';"
        headers = get_security_headers(
            enable_csp=True,
            csp_mode="api",  # Would normally be strict
            custom_csp=custom,
        )

        assert headers[CSP_HEADER] == custom


class TestCSPNonce:
    """Tests for CSP nonce support."""

    def test_generate_nonce_returns_base64(self):
        """generate_nonce should return valid base64 string."""
        nonce = generate_nonce()

        assert nonce is not None
        assert len(nonce) > 0
        # Base64 chars plus padding
        import re
        assert re.match(r'^[A-Za-z0-9+/]+=*$', nonce)

    def test_generate_nonce_is_unique(self):
        """Each nonce should be unique."""
        nonces = [generate_nonce() for _ in range(100)]

        assert len(set(nonces)) == 100  # All unique

    def test_nonce_added_to_script_src(self):
        """Nonce should be added to script-src directive."""
        nonce = "test-nonce-123"
        headers = get_security_headers(
            enable_csp=True,
            csp_mode="standard",
            nonce=nonce,
        )

        csp = headers[CSP_HEADER]
        assert f"'nonce-{nonce}'" in csp
        assert "script-src 'self'" in csp

    def test_nonce_works_with_api_mode(self):
        """Nonce should not break API mode (no script-src)."""
        headers = get_security_headers(
            enable_csp=True,
            csp_mode="api",
            nonce="test-nonce",
        )

        csp = headers[CSP_HEADER]
        # API mode has no script-src, so nonce shouldn't appear
        assert "nonce" not in csp


class TestCSPReporting:
    """Tests for CSP violation reporting."""

    def test_report_uri_appended_to_csp(self):
        """Report URI should be appended to CSP."""
        report_uri = "https://example.com/csp-report"
        headers = get_security_headers(
            enable_csp=True,
            csp_report_uri=report_uri,
        )

        csp = headers[CSP_HEADER]
        assert f"report-uri {report_uri}" in csp

    def test_report_only_mode_uses_different_header(self):
        """Report-only mode should use CSP-Report-Only header."""
        headers = get_security_headers(
            enable_csp=True,
            report_only=True,
        )

        assert CSP_REPORT_ONLY_HEADER in headers
        assert CSP_HEADER not in headers

    def test_report_only_with_uri(self):
        """Report-only mode should support report URI."""
        report_uri = "https://example.com/csp-report"
        headers = get_security_headers(
            enable_csp=True,
            report_only=True,
            csp_report_uri=report_uri,
        )

        csp = headers[CSP_REPORT_ONLY_HEADER]
        assert f"report-uri {report_uri}" in csp


class TestCSPDirectives:
    """Tests for specific CSP directives."""

    def test_api_mode_blocks_forms(self):
        """API mode should block form submissions."""
        headers = get_security_headers(enable_csp=True, csp_mode="api")

        csp = headers[CSP_HEADER]
        assert "form-action 'none'" in csp

    def test_standard_mode_allows_self_forms(self):
        """Standard mode should allow forms to self."""
        headers = get_security_headers(enable_csp=True, csp_mode="standard")

        csp = headers[CSP_HEADER]
        assert "form-action 'self'" in csp

    def test_frame_ancestors_blocked(self):
        """All modes should block framing."""
        for mode in ["api", "standard", "development"]:
            headers = get_security_headers(enable_csp=True, csp_mode=mode)
            csp = headers[CSP_HEADER]
            assert "frame-ancestors 'none'" in csp

    def test_base_uri_restricted(self):
        """API and standard modes should restrict base-uri."""
        for mode in ["api", "standard"]:
            headers = get_security_headers(enable_csp=True, csp_mode=mode)
            csp = headers[CSP_HEADER]
            assert "base-uri" in csp

    def test_connect_src_allows_websocket(self):
        """Standard mode should allow WebSocket connections."""
        headers = get_security_headers(enable_csp=True, csp_mode="standard")

        csp = headers[CSP_HEADER]
        assert "connect-src 'self' wss:" in csp


class TestOtherSecurityHeaders:
    """Tests for non-CSP security headers."""

    def test_x_content_type_options_always_set(self):
        """X-Content-Type-Options should always be set."""
        headers = get_security_headers()

        assert headers["X-Content-Type-Options"] == "nosniff"

    def test_x_frame_options_always_set(self):
        """X-Frame-Options should always be set."""
        headers = get_security_headers()

        assert headers["X-Frame-Options"] == "DENY"

    def test_hsts_only_in_production(self):
        """HSTS should only be enabled in production."""
        dev_headers = get_security_headers(production=False)
        prod_headers = get_security_headers(production=True)

        assert "Strict-Transport-Security" not in dev_headers
        assert "Strict-Transport-Security" in prod_headers

    def test_hsts_can_be_disabled(self):
        """HSTS should be disableable even in production."""
        headers = get_security_headers(production=True, enable_hsts=False)

        assert "Strict-Transport-Security" not in headers

    def test_permissions_policy_set(self):
        """Permissions-Policy should restrict dangerous features."""
        headers = get_security_headers()

        assert "Permissions-Policy" in headers
        policy = headers["Permissions-Policy"]
        assert "geolocation=()" in policy
        assert "microphone=()" in policy
        assert "camera=()" in policy

    def test_referrer_policy_set(self):
        """Referrer-Policy should be set."""
        headers = get_security_headers()

        assert headers["Referrer-Policy"] == "strict-origin-when-cross-origin"
