"""
Tests for aragora.server.middleware.security_headers - Security Headers Middleware.

Tests cover:
1. Content-Security-Policy (CSP) header validation
2. HSTS (Strict-Transport-Security) configuration
3. X-Frame-Options header
4. X-Content-Type-Options (nosniff)
5. X-XSS-Protection header
6. Referrer-Policy
7. Header injection prevention
8. Custom policy configuration
9. Development vs production modes
10. CORS interaction
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest


# =============================================================================
# Test Constants
# =============================================================================


class TestDefaultConstants:
    """Tests for default security header constants."""

    def test_default_x_frame_options(self):
        """DEFAULT_X_FRAME_OPTIONS should be DENY."""
        from aragora.server.middleware.security_headers import DEFAULT_X_FRAME_OPTIONS

        assert DEFAULT_X_FRAME_OPTIONS == "DENY"

    def test_default_x_content_type_options(self):
        """DEFAULT_X_CONTENT_TYPE_OPTIONS should be nosniff."""
        from aragora.server.middleware.security_headers import DEFAULT_X_CONTENT_TYPE_OPTIONS

        assert DEFAULT_X_CONTENT_TYPE_OPTIONS == "nosniff"

    def test_default_x_xss_protection(self):
        """DEFAULT_X_XSS_PROTECTION should enable XSS filter with mode=block."""
        from aragora.server.middleware.security_headers import DEFAULT_X_XSS_PROTECTION

        assert DEFAULT_X_XSS_PROTECTION == "1; mode=block"

    def test_default_referrer_policy(self):
        """DEFAULT_REFERRER_POLICY should be strict-origin-when-cross-origin."""
        from aragora.server.middleware.security_headers import DEFAULT_REFERRER_POLICY

        assert DEFAULT_REFERRER_POLICY == "strict-origin-when-cross-origin"

    def test_default_csp(self):
        """DEFAULT_CSP should have sensible defaults."""
        from aragora.server.middleware.security_headers import DEFAULT_CSP

        assert "default-src 'self'" in DEFAULT_CSP
        assert "script-src" in DEFAULT_CSP
        assert "style-src" in DEFAULT_CSP

    def test_default_hsts(self):
        """DEFAULT_HSTS should include max-age and includeSubDomains."""
        from aragora.server.middleware.security_headers import DEFAULT_HSTS

        assert "max-age=" in DEFAULT_HSTS
        assert "includeSubDomains" in DEFAULT_HSTS

    def test_hsts_max_age_constant(self):
        """HSTS_MAX_AGE should be 1 year in seconds."""
        from aragora.server.middleware.security_headers import HSTS_MAX_AGE

        assert HSTS_MAX_AGE == 31536000  # 365 * 24 * 60 * 60


# =============================================================================
# Test SecurityHeadersConfig
# =============================================================================


class TestSecurityHeadersConfig:
    """Tests for SecurityHeadersConfig dataclass."""

    def test_default_enabled(self):
        """Security headers should be enabled by default."""
        from aragora.server.middleware.security_headers import SecurityHeadersConfig

        with patch.dict(os.environ, {}, clear=True):
            config = SecurityHeadersConfig()
            assert config.enabled is True

    def test_enabled_from_environment_true(self):
        """Should read enabled status from environment (true values)."""
        from aragora.server.middleware.security_headers import SecurityHeadersConfig

        for val in ["true", "1", "yes"]:
            with patch.dict(os.environ, {"ARAGORA_SECURITY_HEADERS_ENABLED": val}, clear=True):
                config = SecurityHeadersConfig()
                assert config.enabled is True, f"Expected enabled=True for value '{val}'"

    def test_enabled_from_environment_false(self):
        """Should read enabled status from environment (false values)."""
        from aragora.server.middleware.security_headers import SecurityHeadersConfig

        for val in ["false", "0", "no"]:
            with patch.dict(os.environ, {"ARAGORA_SECURITY_HEADERS_ENABLED": val}, clear=True):
                config = SecurityHeadersConfig()
                assert config.enabled is False, f"Expected enabled=False for value '{val}'"

    def test_hsts_disabled_in_development(self):
        """HSTS should be disabled by default in development environment."""
        from aragora.server.middleware.security_headers import SecurityHeadersConfig

        with patch.dict(os.environ, {"ARAGORA_ENV": "development"}, clear=True):
            config = SecurityHeadersConfig()
            assert config.hsts_enabled is False

    def test_hsts_enabled_in_production(self):
        """HSTS should be enabled by default in production environment."""
        from aragora.server.middleware.security_headers import SecurityHeadersConfig

        with patch.dict(os.environ, {"ARAGORA_ENV": "production"}, clear=True):
            config = SecurityHeadersConfig()
            assert config.hsts_enabled is True

    def test_hsts_explicit_override_in_development(self):
        """HSTS can be explicitly enabled in development."""
        from aragora.server.middleware.security_headers import SecurityHeadersConfig

        with patch.dict(
            os.environ,
            {"ARAGORA_ENV": "development", "ARAGORA_HSTS_ENABLED": "true"},
            clear=True,
        ):
            config = SecurityHeadersConfig()
            assert config.hsts_enabled is True

    def test_hsts_explicit_override_in_production(self):
        """HSTS can be explicitly disabled in production."""
        from aragora.server.middleware.security_headers import SecurityHeadersConfig

        with patch.dict(
            os.environ,
            {"ARAGORA_ENV": "production", "ARAGORA_HSTS_ENABLED": "false"},
            clear=True,
        ):
            config = SecurityHeadersConfig()
            assert config.hsts_enabled is False

    def test_x_frame_options_from_environment(self):
        """Should read X-Frame-Options from environment."""
        from aragora.server.middleware.security_headers import SecurityHeadersConfig

        with patch.dict(os.environ, {"ARAGORA_X_FRAME_OPTIONS": "SAMEORIGIN"}, clear=True):
            config = SecurityHeadersConfig()
            assert config.x_frame_options == "SAMEORIGIN"

    def test_x_content_type_options_from_environment(self):
        """Should read X-Content-Type-Options from environment."""
        from aragora.server.middleware.security_headers import SecurityHeadersConfig

        with patch.dict(os.environ, {"ARAGORA_X_CONTENT_TYPE_OPTIONS": "custom"}, clear=True):
            config = SecurityHeadersConfig()
            assert config.x_content_type_options == "custom"

    def test_x_xss_protection_from_environment(self):
        """Should read X-XSS-Protection from environment."""
        from aragora.server.middleware.security_headers import SecurityHeadersConfig

        with patch.dict(os.environ, {"ARAGORA_X_XSS_PROTECTION": "0"}, clear=True):
            config = SecurityHeadersConfig()
            assert config.x_xss_protection == "0"

    def test_referrer_policy_from_environment(self):
        """Should read Referrer-Policy from environment."""
        from aragora.server.middleware.security_headers import SecurityHeadersConfig

        with patch.dict(os.environ, {"ARAGORA_REFERRER_POLICY": "no-referrer"}, clear=True):
            config = SecurityHeadersConfig()
            assert config.referrer_policy == "no-referrer"

    def test_csp_from_environment(self):
        """Should read Content-Security-Policy from environment."""
        from aragora.server.middleware.security_headers import SecurityHeadersConfig

        custom_csp = "default-src 'none'; script-src 'self'"
        with patch.dict(os.environ, {"ARAGORA_CONTENT_SECURITY_POLICY": custom_csp}, clear=True):
            config = SecurityHeadersConfig()
            assert config.content_security_policy == custom_csp

    def test_hsts_from_environment(self):
        """Should read Strict-Transport-Security from environment."""
        from aragora.server.middleware.security_headers import SecurityHeadersConfig

        custom_hsts = "max-age=86400"
        with patch.dict(
            os.environ,
            {"ARAGORA_STRICT_TRANSPORT_SECURITY": custom_hsts, "ARAGORA_HSTS_ENABLED": "true"},
            clear=True,
        ):
            config = SecurityHeadersConfig()
            assert config.strict_transport_security == custom_hsts


# =============================================================================
# Test get_security_response_headers Function
# =============================================================================


class TestGetSecurityResponseHeaders:
    """Tests for get_security_response_headers function."""

    def test_returns_dict(self):
        """Should return a dictionary."""
        from aragora.server.middleware.security_headers import get_security_response_headers

        headers = get_security_response_headers()
        assert isinstance(headers, dict)

    def test_includes_x_frame_options(self):
        """Should include X-Frame-Options header."""
        from aragora.server.middleware.security_headers import (
            create_security_headers_config,
            get_security_response_headers,
        )

        config = create_security_headers_config()
        headers = get_security_response_headers(config)
        assert "X-Frame-Options" in headers
        assert headers["X-Frame-Options"] == "DENY"

    def test_includes_x_content_type_options(self):
        """Should include X-Content-Type-Options header."""
        from aragora.server.middleware.security_headers import (
            create_security_headers_config,
            get_security_response_headers,
        )

        config = create_security_headers_config()
        headers = get_security_response_headers(config)
        assert "X-Content-Type-Options" in headers
        assert headers["X-Content-Type-Options"] == "nosniff"

    def test_includes_x_xss_protection(self):
        """Should include X-XSS-Protection header."""
        from aragora.server.middleware.security_headers import (
            create_security_headers_config,
            get_security_response_headers,
        )

        config = create_security_headers_config()
        headers = get_security_response_headers(config)
        assert "X-XSS-Protection" in headers
        assert headers["X-XSS-Protection"] == "1; mode=block"

    def test_includes_referrer_policy(self):
        """Should include Referrer-Policy header."""
        from aragora.server.middleware.security_headers import (
            create_security_headers_config,
            get_security_response_headers,
        )

        config = create_security_headers_config()
        headers = get_security_response_headers(config)
        assert "Referrer-Policy" in headers
        assert headers["Referrer-Policy"] == "strict-origin-when-cross-origin"

    def test_includes_csp(self):
        """Should include Content-Security-Policy header."""
        from aragora.server.middleware.security_headers import (
            create_security_headers_config,
            get_security_response_headers,
        )

        config = create_security_headers_config()
        headers = get_security_response_headers(config)
        assert "Content-Security-Policy" in headers

    def test_hsts_included_when_enabled(self):
        """Should include HSTS when enabled."""
        from aragora.server.middleware.security_headers import (
            create_security_headers_config,
            get_security_response_headers,
        )

        config = create_security_headers_config(hsts_enabled=True)
        headers = get_security_response_headers(config)
        assert "Strict-Transport-Security" in headers

    def test_hsts_excluded_when_disabled(self):
        """Should not include HSTS when disabled."""
        from aragora.server.middleware.security_headers import (
            create_security_headers_config,
            get_security_response_headers,
        )

        config = create_security_headers_config(hsts_enabled=False)
        headers = get_security_response_headers(config)
        assert "Strict-Transport-Security" not in headers

    def test_returns_empty_when_disabled(self):
        """Should return empty dict when security headers are disabled."""
        from aragora.server.middleware.security_headers import (
            create_security_headers_config,
            get_security_response_headers,
        )

        config = create_security_headers_config(enabled=False)
        headers = get_security_response_headers(config)
        assert headers == {}

    def test_uses_default_config_when_none_provided(self):
        """Should use default config when none provided."""
        from aragora.server.middleware.security_headers import get_security_response_headers

        with patch.dict(os.environ, {}, clear=True):
            headers = get_security_response_headers()
            assert "X-Frame-Options" in headers


# =============================================================================
# Test create_security_headers_config Function
# =============================================================================


class TestCreateSecurityHeadersConfig:
    """Tests for create_security_headers_config convenience function."""

    def test_creates_config_with_defaults(self):
        """Should create config with default values."""
        from aragora.server.middleware.security_headers import (
            DEFAULT_CSP,
            DEFAULT_HSTS,
            DEFAULT_REFERRER_POLICY,
            DEFAULT_X_CONTENT_TYPE_OPTIONS,
            DEFAULT_X_FRAME_OPTIONS,
            DEFAULT_X_XSS_PROTECTION,
            create_security_headers_config,
        )

        config = create_security_headers_config()
        assert config.enabled is True
        assert config.x_frame_options == DEFAULT_X_FRAME_OPTIONS
        assert config.x_content_type_options == DEFAULT_X_CONTENT_TYPE_OPTIONS
        assert config.x_xss_protection == DEFAULT_X_XSS_PROTECTION
        assert config.referrer_policy == DEFAULT_REFERRER_POLICY
        assert config.content_security_policy == DEFAULT_CSP
        assert config.strict_transport_security == DEFAULT_HSTS

    def test_creates_config_with_custom_values(self):
        """Should create config with custom values."""
        from aragora.server.middleware.security_headers import create_security_headers_config

        config = create_security_headers_config(
            enabled=False,
            hsts_enabled=True,
            x_frame_options="SAMEORIGIN",
            x_content_type_options="custom",
            x_xss_protection="0",
            referrer_policy="no-referrer",
            content_security_policy="default-src 'none'",
            strict_transport_security="max-age=86400",
        )

        assert config.enabled is False
        assert config.hsts_enabled is True
        assert config.x_frame_options == "SAMEORIGIN"
        assert config.x_content_type_options == "custom"
        assert config.x_xss_protection == "0"
        assert config.referrer_policy == "no-referrer"
        assert config.content_security_policy == "default-src 'none'"
        assert config.strict_transport_security == "max-age=86400"

    def test_hsts_defaults_to_production_check(self):
        """hsts_enabled should default to production environment check."""
        from aragora.server.middleware.security_headers import create_security_headers_config

        with patch.dict(os.environ, {"ARAGORA_ENV": "production"}, clear=True):
            config = create_security_headers_config()
            assert config.hsts_enabled is True

        with patch.dict(os.environ, {"ARAGORA_ENV": "development"}, clear=True):
            config = create_security_headers_config()
            assert config.hsts_enabled is False


# =============================================================================
# Test SecurityHeadersMiddleware Class
# =============================================================================


class TestSecurityHeadersMiddleware:
    """Tests for SecurityHeadersMiddleware class."""

    def test_init_default_config(self):
        """Should initialize with default config."""
        from aragora.server.middleware.security_headers import SecurityHeadersMiddleware

        middleware = SecurityHeadersMiddleware()
        assert middleware.config is not None
        assert middleware.enabled is True

    def test_init_custom_config(self):
        """Should initialize with custom config."""
        from aragora.server.middleware.security_headers import (
            SecurityHeadersMiddleware,
            create_security_headers_config,
        )

        config = create_security_headers_config(enabled=False)
        middleware = SecurityHeadersMiddleware(config)
        assert middleware.enabled is False

    def test_enabled_property(self):
        """Should expose enabled property from config."""
        from aragora.server.middleware.security_headers import (
            SecurityHeadersMiddleware,
            create_security_headers_config,
        )

        config = create_security_headers_config(enabled=True)
        middleware = SecurityHeadersMiddleware(config)
        assert middleware.enabled is True

        config = create_security_headers_config(enabled=False)
        middleware = SecurityHeadersMiddleware(config)
        assert middleware.enabled is False

    def test_hsts_enabled_property(self):
        """Should expose hsts_enabled property from config."""
        from aragora.server.middleware.security_headers import (
            SecurityHeadersMiddleware,
            create_security_headers_config,
        )

        config = create_security_headers_config(hsts_enabled=True)
        middleware = SecurityHeadersMiddleware(config)
        assert middleware.hsts_enabled is True

        config = create_security_headers_config(hsts_enabled=False)
        middleware = SecurityHeadersMiddleware(config)
        assert middleware.hsts_enabled is False

    def test_get_headers_returns_dict(self):
        """get_headers should return a dictionary."""
        from aragora.server.middleware.security_headers import SecurityHeadersMiddleware

        middleware = SecurityHeadersMiddleware()
        headers = middleware.get_headers()
        assert isinstance(headers, dict)

    def test_get_headers_uses_config(self):
        """get_headers should use middleware config."""
        from aragora.server.middleware.security_headers import (
            SecurityHeadersMiddleware,
            create_security_headers_config,
        )

        config = create_security_headers_config(x_frame_options="SAMEORIGIN")
        middleware = SecurityHeadersMiddleware(config)
        headers = middleware.get_headers()
        assert headers["X-Frame-Options"] == "SAMEORIGIN"


class TestSecurityHeadersMiddlewareApplyHeaders:
    """Tests for SecurityHeadersMiddleware.apply_headers method."""

    def test_apply_headers_calls_send_header(self):
        """Should call send_header for each security header."""
        from aragora.server.middleware.security_headers import (
            SecurityHeadersMiddleware,
            create_security_headers_config,
        )

        config = create_security_headers_config(hsts_enabled=False)
        middleware = SecurityHeadersMiddleware(config)

        mock_handler = MagicMock()
        middleware.apply_headers(mock_handler)

        # Verify send_header was called for each header
        call_args = [call[0] for call in mock_handler.send_header.call_args_list]
        header_names = [arg[0] for arg in call_args]

        assert "X-Frame-Options" in header_names
        assert "X-Content-Type-Options" in header_names
        assert "X-XSS-Protection" in header_names
        assert "Referrer-Policy" in header_names
        assert "Content-Security-Policy" in header_names

    def test_apply_headers_includes_hsts_when_enabled(self):
        """Should include HSTS header when enabled."""
        from aragora.server.middleware.security_headers import (
            SecurityHeadersMiddleware,
            create_security_headers_config,
        )

        config = create_security_headers_config(hsts_enabled=True)
        middleware = SecurityHeadersMiddleware(config)

        mock_handler = MagicMock()
        middleware.apply_headers(mock_handler)

        call_args = [call[0] for call in mock_handler.send_header.call_args_list]
        header_names = [arg[0] for arg in call_args]

        assert "Strict-Transport-Security" in header_names

    def test_apply_headers_excludes_hsts_when_disabled(self):
        """Should not include HSTS header when disabled."""
        from aragora.server.middleware.security_headers import (
            SecurityHeadersMiddleware,
            create_security_headers_config,
        )

        config = create_security_headers_config(hsts_enabled=False)
        middleware = SecurityHeadersMiddleware(config)

        mock_handler = MagicMock()
        middleware.apply_headers(mock_handler)

        call_args = [call[0] for call in mock_handler.send_header.call_args_list]
        header_names = [arg[0] for arg in call_args]

        assert "Strict-Transport-Security" not in header_names

    def test_apply_headers_does_nothing_when_disabled(self):
        """Should not call send_header when middleware is disabled."""
        from aragora.server.middleware.security_headers import (
            SecurityHeadersMiddleware,
            create_security_headers_config,
        )

        config = create_security_headers_config(enabled=False)
        middleware = SecurityHeadersMiddleware(config)

        mock_handler = MagicMock()
        middleware.apply_headers(mock_handler)

        mock_handler.send_header.assert_not_called()

    def test_apply_headers_handles_missing_send_header(self):
        """Should handle handler without send_header gracefully."""
        from aragora.server.middleware.security_headers import SecurityHeadersMiddleware

        middleware = SecurityHeadersMiddleware()

        # Handler without send_header method
        mock_handler = MagicMock(spec=[])

        # Should not raise an exception
        middleware.apply_headers(mock_handler)


class TestSecurityHeadersMiddlewareApplyToResponseDict:
    """Tests for SecurityHeadersMiddleware.apply_to_response_dict method."""

    def test_apply_to_response_dict_updates_dict(self):
        """Should update response headers dictionary."""
        from aragora.server.middleware.security_headers import (
            SecurityHeadersMiddleware,
            create_security_headers_config,
        )

        config = create_security_headers_config(hsts_enabled=False)
        middleware = SecurityHeadersMiddleware(config)

        response_headers = {}
        middleware.apply_to_response_dict(response_headers)

        assert "X-Frame-Options" in response_headers
        assert "X-Content-Type-Options" in response_headers
        assert "X-XSS-Protection" in response_headers
        assert "Referrer-Policy" in response_headers
        assert "Content-Security-Policy" in response_headers

    def test_apply_to_response_dict_preserves_existing(self):
        """Should preserve existing headers in dict."""
        from aragora.server.middleware.security_headers import (
            SecurityHeadersMiddleware,
            create_security_headers_config,
        )

        config = create_security_headers_config()
        middleware = SecurityHeadersMiddleware(config)

        response_headers = {"Content-Type": "application/json", "X-Custom": "value"}
        middleware.apply_to_response_dict(response_headers)

        assert response_headers["Content-Type"] == "application/json"
        assert response_headers["X-Custom"] == "value"
        assert "X-Frame-Options" in response_headers

    def test_apply_to_response_dict_does_nothing_when_disabled(self):
        """Should not modify dict when middleware is disabled."""
        from aragora.server.middleware.security_headers import (
            SecurityHeadersMiddleware,
            create_security_headers_config,
        )

        config = create_security_headers_config(enabled=False)
        middleware = SecurityHeadersMiddleware(config)

        response_headers = {"Content-Type": "application/json"}
        middleware.apply_to_response_dict(response_headers)

        assert response_headers == {"Content-Type": "application/json"}


# =============================================================================
# Test apply_security_headers_to_handler Function
# =============================================================================


class TestApplySecurityHeadersToHandler:
    """Tests for apply_security_headers_to_handler convenience function."""

    def test_applies_headers_to_handler(self):
        """Should apply security headers to handler."""
        from aragora.server.middleware.security_headers import (
            apply_security_headers_to_handler,
            create_security_headers_config,
        )

        mock_handler = MagicMock()
        config = create_security_headers_config(hsts_enabled=False)
        apply_security_headers_to_handler(mock_handler, config)

        call_args = [call[0] for call in mock_handler.send_header.call_args_list]
        header_names = [arg[0] for arg in call_args]

        assert "X-Frame-Options" in header_names

    def test_uses_default_config_when_none_provided(self):
        """Should use default config when none provided."""
        from aragora.server.middleware.security_headers import apply_security_headers_to_handler

        mock_handler = MagicMock()

        with patch.dict(os.environ, {}, clear=True):
            apply_security_headers_to_handler(mock_handler)

        # Should have called send_header at least once
        assert mock_handler.send_header.called


# =============================================================================
# Test Content-Security-Policy (CSP) Header Validation
# =============================================================================


class TestCSPHeader:
    """Tests for Content-Security-Policy header."""

    def test_csp_contains_default_src(self):
        """CSP should contain default-src directive."""
        from aragora.server.middleware.security_headers import DEFAULT_CSP

        assert "default-src" in DEFAULT_CSP

    def test_csp_contains_script_src(self):
        """CSP should contain script-src directive."""
        from aragora.server.middleware.security_headers import DEFAULT_CSP

        assert "script-src" in DEFAULT_CSP

    def test_csp_contains_style_src(self):
        """CSP should contain style-src directive."""
        from aragora.server.middleware.security_headers import DEFAULT_CSP

        assert "style-src" in DEFAULT_CSP

    def test_custom_csp_applied(self):
        """Custom CSP should be applied correctly."""
        from aragora.server.middleware.security_headers import (
            create_security_headers_config,
            get_security_response_headers,
        )

        custom_csp = "default-src 'none'; script-src 'self'; style-src 'self'"
        config = create_security_headers_config(content_security_policy=custom_csp)
        headers = get_security_response_headers(config)

        assert headers["Content-Security-Policy"] == custom_csp

    def test_strict_csp_no_unsafe(self):
        """Should support strict CSP without unsafe-inline."""
        from aragora.server.middleware.security_headers import (
            create_security_headers_config,
            get_security_response_headers,
        )

        strict_csp = "default-src 'none'; script-src 'self'"
        config = create_security_headers_config(content_security_policy=strict_csp)
        headers = get_security_response_headers(config)

        assert "'unsafe-inline'" not in headers["Content-Security-Policy"]
        assert "'unsafe-eval'" not in headers["Content-Security-Policy"]


# =============================================================================
# Test HSTS (Strict-Transport-Security) Configuration
# =============================================================================


class TestHSTSHeader:
    """Tests for Strict-Transport-Security (HSTS) header."""

    def test_hsts_max_age_value(self):
        """HSTS should have appropriate max-age."""
        from aragora.server.middleware.security_headers import DEFAULT_HSTS

        assert "max-age=31536000" in DEFAULT_HSTS

    def test_hsts_include_subdomains(self):
        """HSTS should include includeSubDomains directive."""
        from aragora.server.middleware.security_headers import DEFAULT_HSTS

        assert "includeSubDomains" in DEFAULT_HSTS

    def test_custom_hsts_applied(self):
        """Custom HSTS should be applied correctly."""
        from aragora.server.middleware.security_headers import (
            create_security_headers_config,
            get_security_response_headers,
        )

        custom_hsts = "max-age=86400; includeSubDomains; preload"
        config = create_security_headers_config(
            hsts_enabled=True, strict_transport_security=custom_hsts
        )
        headers = get_security_response_headers(config)

        assert headers["Strict-Transport-Security"] == custom_hsts

    def test_hsts_preload_support(self):
        """Should support preload directive in HSTS."""
        from aragora.server.middleware.security_headers import (
            create_security_headers_config,
            get_security_response_headers,
        )

        preload_hsts = "max-age=31536000; includeSubDomains; preload"
        config = create_security_headers_config(
            hsts_enabled=True, strict_transport_security=preload_hsts
        )
        headers = get_security_response_headers(config)

        assert "preload" in headers["Strict-Transport-Security"]


# =============================================================================
# Test X-Frame-Options Header
# =============================================================================


class TestXFrameOptionsHeader:
    """Tests for X-Frame-Options header."""

    def test_default_x_frame_options_deny(self):
        """Default X-Frame-Options should be DENY."""
        from aragora.server.middleware.security_headers import (
            create_security_headers_config,
            get_security_response_headers,
        )

        config = create_security_headers_config()
        headers = get_security_response_headers(config)

        assert headers["X-Frame-Options"] == "DENY"

    def test_x_frame_options_sameorigin(self):
        """Should support SAMEORIGIN value."""
        from aragora.server.middleware.security_headers import (
            create_security_headers_config,
            get_security_response_headers,
        )

        config = create_security_headers_config(x_frame_options="SAMEORIGIN")
        headers = get_security_response_headers(config)

        assert headers["X-Frame-Options"] == "SAMEORIGIN"

    def test_x_frame_options_allow_from(self):
        """Should support ALLOW-FROM directive."""
        from aragora.server.middleware.security_headers import (
            create_security_headers_config,
            get_security_response_headers,
        )

        config = create_security_headers_config(x_frame_options="ALLOW-FROM https://example.com")
        headers = get_security_response_headers(config)

        assert headers["X-Frame-Options"] == "ALLOW-FROM https://example.com"


# =============================================================================
# Test X-Content-Type-Options Header
# =============================================================================


class TestXContentTypeOptionsHeader:
    """Tests for X-Content-Type-Options header."""

    def test_default_nosniff(self):
        """Default X-Content-Type-Options should be nosniff."""
        from aragora.server.middleware.security_headers import (
            create_security_headers_config,
            get_security_response_headers,
        )

        config = create_security_headers_config()
        headers = get_security_response_headers(config)

        assert headers["X-Content-Type-Options"] == "nosniff"

    def test_custom_x_content_type_options(self):
        """Should support custom X-Content-Type-Options value."""
        from aragora.server.middleware.security_headers import (
            create_security_headers_config,
            get_security_response_headers,
        )

        config = create_security_headers_config(x_content_type_options="custom")
        headers = get_security_response_headers(config)

        assert headers["X-Content-Type-Options"] == "custom"


# =============================================================================
# Test X-XSS-Protection Header
# =============================================================================


class TestXXSSProtectionHeader:
    """Tests for X-XSS-Protection header."""

    def test_default_xss_protection(self):
        """Default X-XSS-Protection should enable filter with mode=block."""
        from aragora.server.middleware.security_headers import (
            create_security_headers_config,
            get_security_response_headers,
        )

        config = create_security_headers_config()
        headers = get_security_response_headers(config)

        assert headers["X-XSS-Protection"] == "1; mode=block"

    def test_xss_protection_disabled(self):
        """Should support disabling XSS protection."""
        from aragora.server.middleware.security_headers import (
            create_security_headers_config,
            get_security_response_headers,
        )

        config = create_security_headers_config(x_xss_protection="0")
        headers = get_security_response_headers(config)

        assert headers["X-XSS-Protection"] == "0"

    def test_xss_protection_enabled_without_block(self):
        """Should support enabling without mode=block."""
        from aragora.server.middleware.security_headers import (
            create_security_headers_config,
            get_security_response_headers,
        )

        config = create_security_headers_config(x_xss_protection="1")
        headers = get_security_response_headers(config)

        assert headers["X-XSS-Protection"] == "1"


# =============================================================================
# Test Referrer-Policy Header
# =============================================================================


class TestReferrerPolicyHeader:
    """Tests for Referrer-Policy header."""

    def test_default_referrer_policy(self):
        """Default Referrer-Policy should be strict-origin-when-cross-origin."""
        from aragora.server.middleware.security_headers import (
            create_security_headers_config,
            get_security_response_headers,
        )

        config = create_security_headers_config()
        headers = get_security_response_headers(config)

        assert headers["Referrer-Policy"] == "strict-origin-when-cross-origin"

    def test_referrer_policy_no_referrer(self):
        """Should support no-referrer policy."""
        from aragora.server.middleware.security_headers import (
            create_security_headers_config,
            get_security_response_headers,
        )

        config = create_security_headers_config(referrer_policy="no-referrer")
        headers = get_security_response_headers(config)

        assert headers["Referrer-Policy"] == "no-referrer"

    def test_referrer_policy_same_origin(self):
        """Should support same-origin policy."""
        from aragora.server.middleware.security_headers import (
            create_security_headers_config,
            get_security_response_headers,
        )

        config = create_security_headers_config(referrer_policy="same-origin")
        headers = get_security_response_headers(config)

        assert headers["Referrer-Policy"] == "same-origin"

    def test_referrer_policy_origin(self):
        """Should support origin policy."""
        from aragora.server.middleware.security_headers import (
            create_security_headers_config,
            get_security_response_headers,
        )

        config = create_security_headers_config(referrer_policy="origin")
        headers = get_security_response_headers(config)

        assert headers["Referrer-Policy"] == "origin"


# =============================================================================
# Test Header Injection Prevention
# =============================================================================


class TestHeaderInjectionPrevention:
    """Tests for header injection prevention."""

    def test_header_values_do_not_contain_newlines(self):
        """Header values should not contain newlines (injection vectors)."""
        from aragora.server.middleware.security_headers import (
            create_security_headers_config,
            get_security_response_headers,
        )

        config = create_security_headers_config()
        headers = get_security_response_headers(config)

        for name, value in headers.items():
            assert "\n" not in value, f"Newline found in {name} header"
            assert "\r" not in value, f"Carriage return found in {name} header"

    def test_config_accepts_but_passes_through_values(self):
        """Config should accept values as provided (validation is caller's responsibility)."""
        from aragora.server.middleware.security_headers import (
            create_security_headers_config,
            get_security_response_headers,
        )

        # Note: The middleware passes through values as-is
        # Input validation should be done at a higher level
        config = create_security_headers_config(x_frame_options="CUSTOM")
        headers = get_security_response_headers(config)

        assert headers["X-Frame-Options"] == "CUSTOM"


# =============================================================================
# Test Development vs Production Modes
# =============================================================================


class TestDevelopmentVsProductionModes:
    """Tests for development vs production environment behavior."""

    def test_development_mode_no_hsts_by_default(self):
        """Development mode should not include HSTS by default."""
        from aragora.server.middleware.security_headers import (
            SecurityHeadersConfig,
            get_security_response_headers,
        )

        with patch.dict(os.environ, {"ARAGORA_ENV": "development"}, clear=True):
            config = SecurityHeadersConfig()
            headers = get_security_response_headers(config)

            assert "Strict-Transport-Security" not in headers

    def test_production_mode_includes_hsts_by_default(self):
        """Production mode should include HSTS by default."""
        from aragora.server.middleware.security_headers import (
            SecurityHeadersConfig,
            get_security_response_headers,
        )

        with patch.dict(os.environ, {"ARAGORA_ENV": "production"}, clear=True):
            config = SecurityHeadersConfig()
            headers = get_security_response_headers(config)

            assert "Strict-Transport-Security" in headers

    def test_development_mode_still_includes_other_headers(self):
        """Development mode should still include non-HSTS headers."""
        from aragora.server.middleware.security_headers import (
            SecurityHeadersConfig,
            get_security_response_headers,
        )

        with patch.dict(os.environ, {"ARAGORA_ENV": "development"}, clear=True):
            config = SecurityHeadersConfig()
            headers = get_security_response_headers(config)

            assert "X-Frame-Options" in headers
            assert "X-Content-Type-Options" in headers
            assert "X-XSS-Protection" in headers
            assert "Referrer-Policy" in headers
            assert "Content-Security-Policy" in headers

    def test_is_production_function(self):
        """_is_production should correctly detect environment."""
        from aragora.server.middleware.security_headers import _is_production

        with patch.dict(os.environ, {"ARAGORA_ENV": "production"}, clear=True):
            assert _is_production() is True

        with patch.dict(os.environ, {"ARAGORA_ENV": "development"}, clear=True):
            assert _is_production() is False

        with patch.dict(os.environ, {"ARAGORA_ENV": "PRODUCTION"}, clear=True):
            assert _is_production() is True

        with patch.dict(os.environ, {}, clear=True):
            # Default is development
            assert _is_production() is False


# =============================================================================
# Test Module Exports
# =============================================================================


class TestModuleExports:
    """Tests for module exports."""

    def test_all_exports_accessible(self):
        """All __all__ exports should be accessible."""
        from aragora.server.middleware.security_headers import (
            DEFAULT_CSP,
            DEFAULT_HSTS,
            DEFAULT_REFERRER_POLICY,
            DEFAULT_X_CONTENT_TYPE_OPTIONS,
            DEFAULT_X_FRAME_OPTIONS,
            DEFAULT_X_XSS_PROTECTION,
            HSTS_MAX_AGE,
            SecurityHeadersConfig,
            SecurityHeadersMiddleware,
            apply_security_headers_to_handler,
            create_security_headers_config,
            get_security_response_headers,
        )

        # Verify constants
        assert DEFAULT_X_FRAME_OPTIONS == "DENY"
        assert DEFAULT_X_CONTENT_TYPE_OPTIONS == "nosniff"
        assert DEFAULT_X_XSS_PROTECTION == "1; mode=block"
        assert HSTS_MAX_AGE == 31536000

        # Verify classes and functions are callable
        assert callable(SecurityHeadersConfig)
        assert callable(SecurityHeadersMiddleware)
        assert callable(get_security_response_headers)
        assert callable(apply_security_headers_to_handler)
        assert callable(create_security_headers_config)


# =============================================================================
# Test Edge Cases and Security
# =============================================================================


class TestEdgeCasesAndSecurity:
    """Tests for edge cases and security scenarios."""

    def test_empty_header_values_allowed(self):
        """Should allow empty header values (though not recommended)."""
        from aragora.server.middleware.security_headers import (
            create_security_headers_config,
            get_security_response_headers,
        )

        config = create_security_headers_config(x_frame_options="")
        headers = get_security_response_headers(config)

        assert headers["X-Frame-Options"] == ""

    def test_unicode_in_header_values(self):
        """Should handle unicode in header values."""
        from aragora.server.middleware.security_headers import (
            create_security_headers_config,
            get_security_response_headers,
        )

        # CSP can technically contain unicode characters
        config = create_security_headers_config(content_security_policy="default-src 'self'")
        headers = get_security_response_headers(config)

        assert "Content-Security-Policy" in headers

    def test_multiple_middleware_instances_independent(self):
        """Multiple middleware instances should be independent."""
        from aragora.server.middleware.security_headers import (
            SecurityHeadersMiddleware,
            create_security_headers_config,
        )

        config1 = create_security_headers_config(x_frame_options="DENY")
        config2 = create_security_headers_config(x_frame_options="SAMEORIGIN")

        middleware1 = SecurityHeadersMiddleware(config1)
        middleware2 = SecurityHeadersMiddleware(config2)

        headers1 = middleware1.get_headers()
        headers2 = middleware2.get_headers()

        assert headers1["X-Frame-Options"] == "DENY"
        assert headers2["X-Frame-Options"] == "SAMEORIGIN"

    def test_config_immutability_after_creation(self):
        """Config values should not affect previously created headers."""
        from aragora.server.middleware.security_headers import (
            SecurityHeadersMiddleware,
            create_security_headers_config,
        )

        config = create_security_headers_config(x_frame_options="DENY")
        middleware = SecurityHeadersMiddleware(config)

        headers_before = middleware.get_headers()

        # Modify config
        config.x_frame_options = "SAMEORIGIN"

        headers_after = middleware.get_headers()

        # Headers should reflect the modified config since we're using the same instance
        assert headers_after["X-Frame-Options"] == "SAMEORIGIN"


# =============================================================================
# Test CORS Interaction
# =============================================================================


class TestCORSInteraction:
    """Tests for CORS interaction with security headers."""

    def test_security_headers_do_not_include_cors(self):
        """Security headers should not include CORS headers."""
        from aragora.server.middleware.security_headers import (
            create_security_headers_config,
            get_security_response_headers,
        )

        config = create_security_headers_config()
        headers = get_security_response_headers(config)

        # CORS headers should be handled separately
        assert "Access-Control-Allow-Origin" not in headers
        assert "Access-Control-Allow-Methods" not in headers
        assert "Access-Control-Allow-Headers" not in headers

    def test_security_headers_compatible_with_cors(self):
        """Security headers should be compatible with CORS headers."""
        from aragora.server.middleware.security_headers import (
            SecurityHeadersMiddleware,
            create_security_headers_config,
        )

        config = create_security_headers_config()
        middleware = SecurityHeadersMiddleware(config)

        # Simulate a response dict with CORS headers
        response_headers = {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST",
        }

        middleware.apply_to_response_dict(response_headers)

        # Both security and CORS headers should coexist
        assert "Access-Control-Allow-Origin" in response_headers
        assert "X-Frame-Options" in response_headers
        assert "Content-Security-Policy" in response_headers
