"""
Tests for the security headers middleware.

Validates that all standard security headers are correctly generated,
configurable, and properly applied to response handlers.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from aragora.server.security.headers import (
    DEFAULT_CSP,
    DEFAULT_HSTS,
    DEFAULT_PERMISSIONS_POLICY,
    DEFAULT_REFERRER_POLICY,
    DEFAULT_X_CONTENT_TYPE_OPTIONS,
    DEFAULT_X_FRAME_OPTIONS,
    DEFAULT_X_XSS_PROTECTION,
    SecurityHeadersConfig,
    SecurityHeadersMiddleware,
    get_default_security_headers,
)


# ============================================================================
# SecurityHeadersConfig
# ============================================================================


class TestSecurityHeadersConfig:
    """Tests for the SecurityHeadersConfig dataclass."""

    def test_default_config_enabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ARAGORA_SECURITY_HEADERS", raising=False)
        config = SecurityHeadersConfig()
        assert config.enabled is True

    def test_config_disabled_via_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ARAGORA_SECURITY_HEADERS", "false")
        config = SecurityHeadersConfig()
        assert config.enabled is False

    def test_config_disabled_via_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ARAGORA_SECURITY_HEADERS", "0")
        config = SecurityHeadersConfig()
        assert config.enabled is False

    def test_hsts_disabled_in_development(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ARAGORA_ENV", "development")
        monkeypatch.delenv("ARAGORA_HSTS_ENABLED", raising=False)
        config = SecurityHeadersConfig()
        assert config.hsts_enabled is False

    def test_hsts_enabled_in_production(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ARAGORA_ENV", "production")
        monkeypatch.delenv("ARAGORA_HSTS_ENABLED", raising=False)
        config = SecurityHeadersConfig()
        assert config.hsts_enabled is True

    def test_hsts_explicit_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ARAGORA_ENV", "development")
        monkeypatch.setenv("ARAGORA_HSTS_ENABLED", "true")
        config = SecurityHeadersConfig()
        assert config.hsts_enabled is True

    def test_default_header_values(self) -> None:
        config = SecurityHeadersConfig()
        assert config.hsts == DEFAULT_HSTS
        assert config.x_content_type_options == DEFAULT_X_CONTENT_TYPE_OPTIONS
        assert config.x_frame_options == DEFAULT_X_FRAME_OPTIONS
        assert config.x_xss_protection == DEFAULT_X_XSS_PROTECTION
        assert config.content_security_policy == DEFAULT_CSP
        assert config.referrer_policy == DEFAULT_REFERRER_POLICY
        assert config.permissions_policy == DEFAULT_PERMISSIONS_POLICY

    def test_custom_csp(self) -> None:
        config = SecurityHeadersConfig(content_security_policy="default-src 'none'")
        assert config.content_security_policy == "default-src 'none'"


# ============================================================================
# get_default_security_headers
# ============================================================================


class TestGetDefaultSecurityHeaders:
    """Tests for the get_default_security_headers function."""

    def test_returns_all_standard_headers(self) -> None:
        config = SecurityHeadersConfig(enabled=True, hsts_enabled=True)
        headers = get_default_security_headers(config)

        assert "X-Content-Type-Options" in headers
        assert "X-Frame-Options" in headers
        assert "X-XSS-Protection" in headers
        assert "Content-Security-Policy" in headers
        assert "Referrer-Policy" in headers
        assert "Permissions-Policy" in headers
        assert "Strict-Transport-Security" in headers

    def test_hsts_omitted_when_disabled(self) -> None:
        config = SecurityHeadersConfig(enabled=True, hsts_enabled=False)
        headers = get_default_security_headers(config)

        assert "Strict-Transport-Security" not in headers
        # Other headers should still be present
        assert "X-Frame-Options" in headers

    def test_returns_empty_when_disabled(self) -> None:
        config = SecurityHeadersConfig(enabled=False)
        headers = get_default_security_headers(config)
        assert headers == {}

    def test_nosniff_value(self) -> None:
        config = SecurityHeadersConfig(enabled=True)
        headers = get_default_security_headers(config)
        assert headers["X-Content-Type-Options"] == "nosniff"

    def test_frame_deny_value(self) -> None:
        config = SecurityHeadersConfig(enabled=True)
        headers = get_default_security_headers(config)
        assert headers["X-Frame-Options"] == "DENY"

    def test_permissions_policy_content(self) -> None:
        config = SecurityHeadersConfig(enabled=True)
        headers = get_default_security_headers(config)
        pp = headers["Permissions-Policy"]
        assert "camera=()" in pp
        assert "microphone=()" in pp
        assert "geolocation=()" in pp

    def test_csp_frame_ancestors_none(self) -> None:
        config = SecurityHeadersConfig(enabled=True)
        headers = get_default_security_headers(config)
        assert "frame-ancestors 'none'" in headers["Content-Security-Policy"]

    def test_default_config_uses_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ARAGORA_SECURITY_HEADERS", "true")
        monkeypatch.setenv("ARAGORA_ENV", "development")
        monkeypatch.delenv("ARAGORA_HSTS_ENABLED", raising=False)
        headers = get_default_security_headers()
        assert "X-Frame-Options" in headers
        # HSTS should not be present in development by default
        assert "Strict-Transport-Security" not in headers


# ============================================================================
# SecurityHeadersMiddleware
# ============================================================================


class TestSecurityHeadersMiddleware:
    """Tests for the SecurityHeadersMiddleware class."""

    def test_apply_headers_to_handler(self) -> None:
        handler = MagicMock()
        config = SecurityHeadersConfig(enabled=True, hsts_enabled=True)
        mw = SecurityHeadersMiddleware(config)
        mw.apply_headers(handler)

        sent = {call.args[0]: call.args[1] for call in handler.send_header.call_args_list}
        assert sent["X-Frame-Options"] == "DENY"
        assert sent["X-Content-Type-Options"] == "nosniff"
        assert "Strict-Transport-Security" in sent

    def test_apply_headers_skips_disabled(self) -> None:
        handler = MagicMock()
        config = SecurityHeadersConfig(enabled=False)
        mw = SecurityHeadersMiddleware(config)
        mw.apply_headers(handler)
        handler.send_header.assert_not_called()

    def test_apply_to_dict(self) -> None:
        config = SecurityHeadersConfig(enabled=True, hsts_enabled=False)
        mw = SecurityHeadersMiddleware(config)
        resp: dict[str, str] = {}
        mw.apply_to_dict(resp)
        assert resp["X-Frame-Options"] == "DENY"
        assert "Strict-Transport-Security" not in resp

    def test_exclude_paths(self) -> None:
        config = SecurityHeadersConfig(
            enabled=True,
            exclude_paths=["/healthz", "/readyz"],
        )
        mw = SecurityHeadersMiddleware(config)

        assert mw.should_apply("/healthz") is False
        assert mw.should_apply("/readyz") is False
        assert mw.should_apply("/api/v1/debate") is True

    def test_exclude_path_prefix_match(self) -> None:
        config = SecurityHeadersConfig(
            enabled=True,
            exclude_paths=["/internal/"],
        )
        mw = SecurityHeadersMiddleware(config)
        assert mw.should_apply("/internal/metrics") is False
        assert mw.should_apply("/api/internal") is True  # not a prefix match

    def test_handler_without_send_header(self) -> None:
        """Middleware should warn but not crash if handler lacks send_header."""
        handler = SimpleNamespace()  # no send_header attribute
        config = SecurityHeadersConfig(enabled=True)
        mw = SecurityHeadersMiddleware(config)
        # Should not raise
        mw.apply_headers(handler)

    def test_cache_invalidation(self) -> None:
        config = SecurityHeadersConfig(enabled=True, hsts_enabled=False)
        mw = SecurityHeadersMiddleware(config)

        headers1 = mw.get_headers()
        assert "Strict-Transport-Security" not in headers1

        # Change config and invalidate
        mw.config.hsts_enabled = True
        mw.config.hsts = "max-age=3600"
        mw.invalidate_cache()

        headers2 = mw.get_headers()
        assert headers2["Strict-Transport-Security"] == "max-age=3600"

    def test_enabled_property(self) -> None:
        mw = SecurityHeadersMiddleware(SecurityHeadersConfig(enabled=True))
        assert mw.enabled is True

        mw2 = SecurityHeadersMiddleware(SecurityHeadersConfig(enabled=False))
        assert mw2.enabled is False

    def test_get_headers_returns_copy(self) -> None:
        """Mutating the returned dict must not affect the middleware cache."""
        config = SecurityHeadersConfig(enabled=True)
        mw = SecurityHeadersMiddleware(config)

        h1 = mw.get_headers()
        h1["X-Custom"] = "injected"

        h2 = mw.get_headers()
        assert "X-Custom" not in h2
