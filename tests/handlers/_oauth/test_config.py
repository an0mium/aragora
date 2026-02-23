"""Tests for aragora/server/handlers/oauth/config.py.

Covers all exported functions and module-level constants:

- _get_secret: AWS Secrets Manager fallback to os.environ
- _is_production: ARAGORA_ENV check
- Provider credential getters (Google, GitHub, Microsoft, Apple, OIDC)
- Redirect URI getters with dev fallbacks
- Frontend URL getters (success, error) with dev fallbacks
- _get_allowed_redirect_hosts: frozenset construction from env
- validate_oauth_config: startup configuration validation
- get_oauth_config_status: diagnostic status report
- Provider endpoint constants (Google, GitHub, Microsoft, Apple)
- Legacy module-level variables
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.oauth import config as cfg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result: object) -> dict:
    """Extract JSON body dict from a result (dict or tuple)."""
    if isinstance(result, dict):
        return result
    if isinstance(result, tuple):
        return result[0] if isinstance(result[0], dict) else {}
    return {}


def _status(result: object) -> int:
    """Extract HTTP status code from a result (dict or tuple)."""
    if isinstance(result, dict):
        return result.get("status_code", 200)
    if isinstance(result, tuple):
        return result[1] if len(result) > 1 else 200
    return 200


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


# All env vars that config.py reads or module-level vars snapshot
_ALL_ENV_VARS = [
    "ARAGORA_ENV",
    "ARAGORA_JWT_SECRET",
    "GOOGLE_OAUTH_CLIENT_ID",
    "GOOGLE_OAUTH_CLIENT_SECRET",
    "GOOGLE_OAUTH_REDIRECT_URI",
    "GITHUB_OAUTH_CLIENT_ID",
    "GITHUB_OAUTH_CLIENT_SECRET",
    "GITHUB_OAUTH_REDIRECT_URI",
    "MICROSOFT_OAUTH_CLIENT_ID",
    "MICROSOFT_OAUTH_CLIENT_SECRET",
    "MICROSOFT_OAUTH_TENANT",
    "MICROSOFT_OAUTH_REDIRECT_URI",
    "APPLE_OAUTH_CLIENT_ID",
    "APPLE_TEAM_ID",
    "APPLE_KEY_ID",
    "APPLE_PRIVATE_KEY",
    "APPLE_OAUTH_REDIRECT_URI",
    "OIDC_ISSUER",
    "OIDC_CLIENT_ID",
    "OIDC_CLIENT_SECRET",
    "OIDC_REDIRECT_URI",
    "OAUTH_SUCCESS_URL",
    "OAUTH_ERROR_URL",
    "OAUTH_ALLOWED_REDIRECT_HOSTS",
]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Remove all OAuth env vars before each test for isolation."""
    for var in _ALL_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    yield


# ===========================================================================
# _get_secret
# ===========================================================================


class TestGetSecret:
    """Tests for the _get_secret helper."""

    def test_fallback_to_env_when_secrets_module_unavailable(self, monkeypatch):
        """When aragora.config.secrets is not importable, use os.environ."""
        monkeypatch.setenv("MY_TEST_SECRET", "env-value")
        with patch.dict(sys.modules, {"aragora.config.secrets": None}):
            result = cfg._get_secret("MY_TEST_SECRET", "default")
        # ImportError path -> os.environ.get
        assert result == "env-value"

    def test_default_when_env_var_missing(self, monkeypatch):
        """Return default when env var not set and secrets module unavailable."""
        monkeypatch.delenv("NONEXISTENT_VAR", raising=False)
        with patch.dict(sys.modules, {"aragora.config.secrets": None}):
            result = cfg._get_secret("NONEXISTENT_VAR", "my-default")
        assert result == "my-default"

    def test_empty_default(self, monkeypatch):
        """Default default is empty string."""
        monkeypatch.delenv("NONEXISTENT_VAR", raising=False)
        with patch.dict(sys.modules, {"aragora.config.secrets": None}):
            result = cfg._get_secret("NONEXISTENT_VAR")
        assert result == ""

    def test_uses_secrets_module_when_available(self, monkeypatch):
        """When aragora.config.secrets is importable, use get_secret."""
        mock_mod = MagicMock()
        mock_mod.get_secret.return_value = "secret-value"
        with patch.dict(sys.modules, {"aragora.config.secrets": mock_mod}):
            result = cfg._get_secret("MY_KEY", "default")
        mock_mod.get_secret.assert_called_once_with("MY_KEY", "default")
        assert result == "secret-value"

    def test_secrets_module_returns_none_uses_default(self, monkeypatch):
        """When secrets module returns None/empty, fall back to default."""
        mock_mod = MagicMock()
        mock_mod.get_secret.return_value = None
        with patch.dict(sys.modules, {"aragora.config.secrets": mock_mod}):
            result = cfg._get_secret("MY_KEY", "fallback")
        assert result == "fallback"

    def test_secrets_module_returns_empty_uses_default(self, monkeypatch):
        """When secrets module returns empty string, fall back to default."""
        mock_mod = MagicMock()
        mock_mod.get_secret.return_value = ""
        with patch.dict(sys.modules, {"aragora.config.secrets": mock_mod}):
            result = cfg._get_secret("MY_KEY", "fallback")
        assert result == "fallback"

    def test_secrets_module_import_error_fallback(self, monkeypatch):
        """ImportError in secrets module triggers env fallback."""
        monkeypatch.setenv("FALLBACK_VAR", "env-val")
        # Patch the import to raise ImportError
        with patch(
            "aragora.server.handlers.oauth.config._get_secret",
            wraps=cfg._get_secret,
        ):
            with patch.dict(sys.modules, {"aragora.config.secrets": None}):
                result = cfg._get_secret("FALLBACK_VAR", "default")
        assert result == "env-val"


# ===========================================================================
# _is_production
# ===========================================================================


class TestIsProduction:
    """Tests for the _is_production helper."""

    def test_not_production_by_default(self, monkeypatch):
        """Default environment is not production."""
        monkeypatch.delenv("ARAGORA_ENV", raising=False)
        assert cfg._is_production() is False

    def test_production_when_set(self, monkeypatch):
        """Returns True when ARAGORA_ENV=production."""
        monkeypatch.setenv("ARAGORA_ENV", "production")
        assert cfg._is_production() is True

    def test_production_case_insensitive(self, monkeypatch):
        """Production check is case-insensitive."""
        monkeypatch.setenv("ARAGORA_ENV", "Production")
        assert cfg._is_production() is True

    def test_production_uppercase(self, monkeypatch):
        """PRODUCTION (uppercase) is recognized."""
        monkeypatch.setenv("ARAGORA_ENV", "PRODUCTION")
        assert cfg._is_production() is True

    def test_not_production_development(self, monkeypatch):
        """development is not production."""
        monkeypatch.setenv("ARAGORA_ENV", "development")
        assert cfg._is_production() is False

    def test_not_production_staging(self, monkeypatch):
        """staging is not production."""
        monkeypatch.setenv("ARAGORA_ENV", "staging")
        assert cfg._is_production() is False

    def test_not_production_empty_string(self, monkeypatch):
        """Empty string is not production."""
        monkeypatch.setenv("ARAGORA_ENV", "")
        assert cfg._is_production() is False


# ===========================================================================
# Provider credential getters
# ===========================================================================


class TestGoogleCredentialGetters:
    """Tests for Google OAuth credential getters."""

    def test_google_client_id_from_env(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_OAUTH_CLIENT_ID", "goog-id-123")
        with patch.dict(sys.modules, {"aragora.config.secrets": None}):
            assert cfg._get_google_client_id() == "goog-id-123"

    def test_google_client_id_default_empty(self, monkeypatch):
        with patch.dict(sys.modules, {"aragora.config.secrets": None}):
            assert cfg._get_google_client_id() == ""

    def test_google_client_secret_from_env(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_OAUTH_CLIENT_SECRET", "goog-secret")
        with patch.dict(sys.modules, {"aragora.config.secrets": None}):
            assert cfg._get_google_client_secret() == "goog-secret"

    def test_google_client_secret_default_empty(self, monkeypatch):
        with patch.dict(sys.modules, {"aragora.config.secrets": None}):
            assert cfg._get_google_client_secret() == ""


class TestGitHubCredentialGetters:
    """Tests for GitHub OAuth credential getters."""

    def test_github_client_id_from_env(self, monkeypatch):
        monkeypatch.setenv("GITHUB_OAUTH_CLIENT_ID", "gh-id-456")
        with patch.dict(sys.modules, {"aragora.config.secrets": None}):
            assert cfg._get_github_client_id() == "gh-id-456"

    def test_github_client_id_default_empty(self, monkeypatch):
        with patch.dict(sys.modules, {"aragora.config.secrets": None}):
            assert cfg._get_github_client_id() == ""

    def test_github_client_secret_from_env(self, monkeypatch):
        monkeypatch.setenv("GITHUB_OAUTH_CLIENT_SECRET", "gh-secret")
        with patch.dict(sys.modules, {"aragora.config.secrets": None}):
            assert cfg._get_github_client_secret() == "gh-secret"

    def test_github_client_secret_default_empty(self, monkeypatch):
        with patch.dict(sys.modules, {"aragora.config.secrets": None}):
            assert cfg._get_github_client_secret() == ""


class TestMicrosoftCredentialGetters:
    """Tests for Microsoft OAuth credential getters."""

    def test_microsoft_client_id_from_env(self, monkeypatch):
        monkeypatch.setenv("MICROSOFT_OAUTH_CLIENT_ID", "ms-id-789")
        with patch.dict(sys.modules, {"aragora.config.secrets": None}):
            assert cfg._get_microsoft_client_id() == "ms-id-789"

    def test_microsoft_client_secret_from_env(self, monkeypatch):
        monkeypatch.setenv("MICROSOFT_OAUTH_CLIENT_SECRET", "ms-secret")
        with patch.dict(sys.modules, {"aragora.config.secrets": None}):
            assert cfg._get_microsoft_client_secret() == "ms-secret"

    def test_microsoft_tenant_default_common(self, monkeypatch):
        """Default tenant is 'common' for multi-tenant."""
        with patch.dict(sys.modules, {"aragora.config.secrets": None}):
            assert cfg._get_microsoft_tenant() == "common"

    def test_microsoft_tenant_custom(self, monkeypatch):
        monkeypatch.setenv("MICROSOFT_OAUTH_TENANT", "my-tenant-id")
        with patch.dict(sys.modules, {"aragora.config.secrets": None}):
            assert cfg._get_microsoft_tenant() == "my-tenant-id"


class TestAppleCredentialGetters:
    """Tests for Apple Sign-In credential getters."""

    def test_apple_client_id_from_env(self, monkeypatch):
        monkeypatch.setenv("APPLE_OAUTH_CLIENT_ID", "apple-id")
        with patch.dict(sys.modules, {"aragora.config.secrets": None}):
            assert cfg._get_apple_client_id() == "apple-id"

    def test_apple_team_id_from_env(self, monkeypatch):
        monkeypatch.setenv("APPLE_TEAM_ID", "TEAM123")
        with patch.dict(sys.modules, {"aragora.config.secrets": None}):
            assert cfg._get_apple_team_id() == "TEAM123"

    def test_apple_key_id_from_env(self, monkeypatch):
        monkeypatch.setenv("APPLE_KEY_ID", "KEY456")
        with patch.dict(sys.modules, {"aragora.config.secrets": None}):
            assert cfg._get_apple_key_id() == "KEY456"

    def test_apple_private_key_from_env(self, monkeypatch):
        monkeypatch.setenv("APPLE_PRIVATE_KEY", "-----BEGIN PRIVATE KEY-----\nXYZ")
        with patch.dict(sys.modules, {"aragora.config.secrets": None}):
            assert cfg._get_apple_private_key() == "-----BEGIN PRIVATE KEY-----\nXYZ"

    def test_apple_defaults_empty(self, monkeypatch):
        with patch.dict(sys.modules, {"aragora.config.secrets": None}):
            assert cfg._get_apple_client_id() == ""
            assert cfg._get_apple_team_id() == ""
            assert cfg._get_apple_key_id() == ""
            assert cfg._get_apple_private_key() == ""


class TestOIDCCredentialGetters:
    """Tests for generic OIDC credential getters."""

    def test_oidc_issuer_from_env(self, monkeypatch):
        monkeypatch.setenv("OIDC_ISSUER", "https://idp.example.com")
        with patch.dict(sys.modules, {"aragora.config.secrets": None}):
            assert cfg._get_oidc_issuer() == "https://idp.example.com"

    def test_oidc_client_id_from_env(self, monkeypatch):
        monkeypatch.setenv("OIDC_CLIENT_ID", "oidc-client-id")
        with patch.dict(sys.modules, {"aragora.config.secrets": None}):
            assert cfg._get_oidc_client_id() == "oidc-client-id"

    def test_oidc_client_secret_from_env(self, monkeypatch):
        monkeypatch.setenv("OIDC_CLIENT_SECRET", "oidc-secret")
        with patch.dict(sys.modules, {"aragora.config.secrets": None}):
            assert cfg._get_oidc_client_secret() == "oidc-secret"

    def test_oidc_defaults_empty(self, monkeypatch):
        with patch.dict(sys.modules, {"aragora.config.secrets": None}):
            assert cfg._get_oidc_issuer() == ""
            assert cfg._get_oidc_client_id() == ""
            assert cfg._get_oidc_client_secret() == ""


# ===========================================================================
# Redirect URI getters (with dev fallbacks)
# ===========================================================================


class TestGoogleRedirectURI:
    """Tests for _get_google_redirect_uri."""

    def test_custom_uri_from_env(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_OAUTH_REDIRECT_URI", "https://myapp.com/callback")
        with patch.dict(sys.modules, {"aragora.config.secrets": None}):
            assert cfg._get_google_redirect_uri() == "https://myapp.com/callback"

    def test_dev_fallback(self, monkeypatch):
        """In non-production, returns localhost fallback."""
        monkeypatch.delenv("ARAGORA_ENV", raising=False)
        with patch.dict(sys.modules, {"aragora.config.secrets": None}):
            result = cfg._get_google_redirect_uri()
        assert result == "http://localhost:8080/api/auth/oauth/google/callback"

    def test_production_no_env_returns_empty(self, monkeypatch):
        """In production with no env var, returns empty string."""
        monkeypatch.setenv("ARAGORA_ENV", "production")
        with patch.dict(sys.modules, {"aragora.config.secrets": None}):
            # Must also patch _is_production since the function calls it at runtime
            with patch.object(cfg, "_is_production", return_value=True):
                result = cfg._get_google_redirect_uri()
        assert result == ""

    def test_explicit_uri_overrides_production(self, monkeypatch):
        """Explicit URI takes priority even in production."""
        monkeypatch.setenv("ARAGORA_ENV", "production")
        monkeypatch.setenv("GOOGLE_OAUTH_REDIRECT_URI", "https://prod.example.com/cb")
        with patch.dict(sys.modules, {"aragora.config.secrets": None}):
            assert cfg._get_google_redirect_uri() == "https://prod.example.com/cb"


class TestGitHubRedirectURI:
    """Tests for _get_github_redirect_uri."""

    def test_custom_uri_from_env(self, monkeypatch):
        monkeypatch.setenv("GITHUB_OAUTH_REDIRECT_URI", "https://myapp.com/gh-cb")
        with patch.dict(sys.modules, {"aragora.config.secrets": None}):
            assert cfg._get_github_redirect_uri() == "https://myapp.com/gh-cb"

    def test_dev_fallback(self, monkeypatch):
        with patch.dict(sys.modules, {"aragora.config.secrets": None}):
            result = cfg._get_github_redirect_uri()
        assert result == "http://localhost:8080/api/auth/oauth/github/callback"

    def test_production_no_env_returns_empty(self, monkeypatch):
        monkeypatch.setenv("ARAGORA_ENV", "production")
        with patch.dict(sys.modules, {"aragora.config.secrets": None}):
            with patch.object(cfg, "_is_production", return_value=True):
                result = cfg._get_github_redirect_uri()
        assert result == ""


class TestMicrosoftRedirectURI:
    """Tests for _get_microsoft_redirect_uri."""

    def test_custom_uri_from_env(self, monkeypatch):
        monkeypatch.setenv("MICROSOFT_OAUTH_REDIRECT_URI", "https://myapp.com/ms-cb")
        with patch.dict(sys.modules, {"aragora.config.secrets": None}):
            assert cfg._get_microsoft_redirect_uri() == "https://myapp.com/ms-cb"

    def test_dev_fallback(self, monkeypatch):
        with patch.dict(sys.modules, {"aragora.config.secrets": None}):
            result = cfg._get_microsoft_redirect_uri()
        assert result == "http://localhost:8080/api/auth/oauth/microsoft/callback"

    def test_production_no_env_returns_empty(self, monkeypatch):
        monkeypatch.setenv("ARAGORA_ENV", "production")
        with patch.dict(sys.modules, {"aragora.config.secrets": None}):
            with patch.object(cfg, "_is_production", return_value=True):
                result = cfg._get_microsoft_redirect_uri()
        assert result == ""


class TestAppleRedirectURI:
    """Tests for _get_apple_redirect_uri."""

    def test_custom_uri_from_env(self, monkeypatch):
        monkeypatch.setenv("APPLE_OAUTH_REDIRECT_URI", "https://myapp.com/apple-cb")
        with patch.dict(sys.modules, {"aragora.config.secrets": None}):
            assert cfg._get_apple_redirect_uri() == "https://myapp.com/apple-cb"

    def test_dev_fallback(self, monkeypatch):
        with patch.dict(sys.modules, {"aragora.config.secrets": None}):
            result = cfg._get_apple_redirect_uri()
        assert result == "http://localhost:8080/api/auth/oauth/apple/callback"

    def test_production_no_env_returns_empty(self, monkeypatch):
        monkeypatch.setenv("ARAGORA_ENV", "production")
        with patch.dict(sys.modules, {"aragora.config.secrets": None}):
            with patch.object(cfg, "_is_production", return_value=True):
                result = cfg._get_apple_redirect_uri()
        assert result == ""


class TestOIDCRedirectURI:
    """Tests for _get_oidc_redirect_uri."""

    def test_custom_uri_from_env(self, monkeypatch):
        monkeypatch.setenv("OIDC_REDIRECT_URI", "https://myapp.com/oidc-cb")
        with patch.dict(sys.modules, {"aragora.config.secrets": None}):
            assert cfg._get_oidc_redirect_uri() == "https://myapp.com/oidc-cb"

    def test_dev_fallback(self, monkeypatch):
        with patch.dict(sys.modules, {"aragora.config.secrets": None}):
            result = cfg._get_oidc_redirect_uri()
        assert result == "http://localhost:8080/api/auth/oauth/oidc/callback"

    def test_production_no_env_returns_empty(self, monkeypatch):
        monkeypatch.setenv("ARAGORA_ENV", "production")
        with patch.dict(sys.modules, {"aragora.config.secrets": None}):
            with patch.object(cfg, "_is_production", return_value=True):
                result = cfg._get_oidc_redirect_uri()
        assert result == ""


# ===========================================================================
# Frontend URL getters
# ===========================================================================


class TestOAuthSuccessURL:
    """Tests for _get_oauth_success_url."""

    def test_custom_url_from_env(self, monkeypatch):
        monkeypatch.setenv("OAUTH_SUCCESS_URL", "https://myapp.com/success")
        with patch.dict(sys.modules, {"aragora.config.secrets": None}):
            assert cfg._get_oauth_success_url() == "https://myapp.com/success"

    def test_dev_fallback(self, monkeypatch):
        with patch.dict(sys.modules, {"aragora.config.secrets": None}):
            result = cfg._get_oauth_success_url()
        assert result == "http://localhost:3000/auth/callback"

    def test_production_no_env_returns_empty(self, monkeypatch):
        monkeypatch.setenv("ARAGORA_ENV", "production")
        with patch.dict(sys.modules, {"aragora.config.secrets": None}):
            with patch.object(cfg, "_is_production", return_value=True):
                result = cfg._get_oauth_success_url()
        assert result == ""


class TestOAuthErrorURL:
    """Tests for _get_oauth_error_url."""

    def test_custom_url_from_env(self, monkeypatch):
        monkeypatch.setenv("OAUTH_ERROR_URL", "https://myapp.com/error")
        with patch.dict(sys.modules, {"aragora.config.secrets": None}):
            assert cfg._get_oauth_error_url() == "https://myapp.com/error"

    def test_dev_fallback(self, monkeypatch):
        with patch.dict(sys.modules, {"aragora.config.secrets": None}):
            result = cfg._get_oauth_error_url()
        assert result == "http://localhost:3000/auth/error"

    def test_production_no_env_returns_empty(self, monkeypatch):
        monkeypatch.setenv("ARAGORA_ENV", "production")
        with patch.dict(sys.modules, {"aragora.config.secrets": None}):
            with patch.object(cfg, "_is_production", return_value=True):
                result = cfg._get_oauth_error_url()
        assert result == ""


# ===========================================================================
# _get_allowed_redirect_hosts
# ===========================================================================


class TestGetAllowedRedirectHosts:
    """Tests for _get_allowed_redirect_hosts."""

    def test_dev_defaults(self, monkeypatch):
        """In dev mode, defaults to localhost and 127.0.0.1."""
        with patch.dict(sys.modules, {"aragora.config.secrets": None}):
            result = cfg._get_allowed_redirect_hosts()
        assert isinstance(result, frozenset)
        assert "localhost" in result
        assert "127.0.0.1" in result

    def test_custom_hosts_from_env(self, monkeypatch):
        monkeypatch.setenv("OAUTH_ALLOWED_REDIRECT_HOSTS", "example.com,myapp.com")
        with patch.dict(sys.modules, {"aragora.config.secrets": None}):
            result = cfg._get_allowed_redirect_hosts()
        assert result == frozenset({"example.com", "myapp.com"})

    def test_whitespace_trimmed(self, monkeypatch):
        monkeypatch.setenv("OAUTH_ALLOWED_REDIRECT_HOSTS", "  example.com , myapp.com  ")
        with patch.dict(sys.modules, {"aragora.config.secrets": None}):
            result = cfg._get_allowed_redirect_hosts()
        assert result == frozenset({"example.com", "myapp.com"})

    def test_hosts_lowercased(self, monkeypatch):
        monkeypatch.setenv("OAUTH_ALLOWED_REDIRECT_HOSTS", "Example.COM,MyApp.Com")
        with patch.dict(sys.modules, {"aragora.config.secrets": None}):
            result = cfg._get_allowed_redirect_hosts()
        assert result == frozenset({"example.com", "myapp.com"})

    def test_empty_entries_filtered(self, monkeypatch):
        monkeypatch.setenv("OAUTH_ALLOWED_REDIRECT_HOSTS", "example.com,,,,myapp.com,")
        with patch.dict(sys.modules, {"aragora.config.secrets": None}):
            result = cfg._get_allowed_redirect_hosts()
        assert result == frozenset({"example.com", "myapp.com"})

    def test_single_host(self, monkeypatch):
        monkeypatch.setenv("OAUTH_ALLOWED_REDIRECT_HOSTS", "single.example.com")
        with patch.dict(sys.modules, {"aragora.config.secrets": None}):
            result = cfg._get_allowed_redirect_hosts()
        assert result == frozenset({"single.example.com"})

    def test_production_no_env_returns_empty(self, monkeypatch):
        """In production with no env var, returns empty frozenset."""
        monkeypatch.setenv("ARAGORA_ENV", "production")
        with patch.dict(sys.modules, {"aragora.config.secrets": None}):
            with patch.object(cfg, "_is_production", return_value=True):
                result = cfg._get_allowed_redirect_hosts()
        assert result == frozenset()

    def test_returns_frozenset(self, monkeypatch):
        monkeypatch.setenv("OAUTH_ALLOWED_REDIRECT_HOSTS", "a.com,b.com")
        with patch.dict(sys.modules, {"aragora.config.secrets": None}):
            result = cfg._get_allowed_redirect_hosts()
        assert isinstance(result, frozenset)

    def test_whitespace_only_entries_filtered(self, monkeypatch):
        monkeypatch.setenv("OAUTH_ALLOWED_REDIRECT_HOSTS", " , , example.com")
        with patch.dict(sys.modules, {"aragora.config.secrets": None}):
            result = cfg._get_allowed_redirect_hosts()
        assert result == frozenset({"example.com"})


# ===========================================================================
# validate_oauth_config
# ===========================================================================


class TestValidateOAuthConfig:
    """Tests for validate_oauth_config."""

    def test_dev_mode_no_jwt_returns_missing(self, monkeypatch):
        """In dev mode without JWT secret, should report it missing (if not under pytest)."""
        # validate_oauth_config checks `"pytest" in sys.modules`, and since
        # we ARE running under pytest, it skips the JWT check.
        # So in dev mode, it should return empty (since it skips provider checks too).
        result = cfg.validate_oauth_config(log_warnings=False)
        assert isinstance(result, list)
        # Under pytest in dev mode: JWT skipped, non-production skips provider checks
        assert result == []

    def test_dev_mode_returns_early(self, monkeypatch):
        """In non-production mode, provider checks are skipped."""
        # _IS_PRODUCTION is module-level, so we need to patch it
        with patch.object(cfg, "_IS_PRODUCTION", False):
            result = cfg.validate_oauth_config(log_warnings=False)
        assert isinstance(result, list)
        # Only JWT is checked in dev (skipped under pytest), so empty
        assert result == []

    def test_production_google_enabled_missing_secret(self, monkeypatch):
        """In production with Google enabled but missing client secret."""
        with patch.object(cfg, "_IS_PRODUCTION", True), \
             patch.object(cfg, "GOOGLE_CLIENT_ID", "goog-id"), \
             patch.object(cfg, "GITHUB_CLIENT_ID", ""), \
             patch.object(cfg, "ALLOWED_OAUTH_REDIRECT_HOSTS", frozenset({"example.com"})):
            with patch.object(cfg, "_get_google_client_secret", return_value=""), \
                 patch.object(cfg, "_get_google_redirect_uri", return_value="https://example.com/cb"), \
                 patch.object(cfg, "_get_oauth_success_url", return_value="https://example.com/ok"), \
                 patch.object(cfg, "_get_oauth_error_url", return_value="https://example.com/err"):
                result = cfg.validate_oauth_config(log_warnings=False)
        assert "GOOGLE_OAUTH_CLIENT_SECRET" in result

    def test_production_google_enabled_missing_redirect(self, monkeypatch):
        """In production with Google enabled but missing redirect URI."""
        with patch.object(cfg, "_IS_PRODUCTION", True), \
             patch.object(cfg, "GOOGLE_CLIENT_ID", "goog-id"), \
             patch.object(cfg, "GITHUB_CLIENT_ID", ""), \
             patch.object(cfg, "ALLOWED_OAUTH_REDIRECT_HOSTS", frozenset({"example.com"})):
            with patch.object(cfg, "_get_google_client_secret", return_value="secret"), \
                 patch.object(cfg, "_get_google_redirect_uri", return_value=""), \
                 patch.object(cfg, "_get_oauth_success_url", return_value="https://example.com/ok"), \
                 patch.object(cfg, "_get_oauth_error_url", return_value="https://example.com/err"):
                result = cfg.validate_oauth_config(log_warnings=False)
        assert "GOOGLE_OAUTH_REDIRECT_URI" in result

    def test_production_google_enabled_missing_success_url(self, monkeypatch):
        """In production with Google enabled but missing success URL."""
        with patch.object(cfg, "_IS_PRODUCTION", True), \
             patch.object(cfg, "GOOGLE_CLIENT_ID", "goog-id"), \
             patch.object(cfg, "GITHUB_CLIENT_ID", ""), \
             patch.object(cfg, "ALLOWED_OAUTH_REDIRECT_HOSTS", frozenset({"example.com"})):
            with patch.object(cfg, "_get_google_client_secret", return_value="secret"), \
                 patch.object(cfg, "_get_google_redirect_uri", return_value="https://example.com/cb"), \
                 patch.object(cfg, "_get_oauth_success_url", return_value=""), \
                 patch.object(cfg, "_get_oauth_error_url", return_value="https://example.com/err"):
                result = cfg.validate_oauth_config(log_warnings=False)
        assert "OAUTH_SUCCESS_URL" in result

    def test_production_google_enabled_missing_error_url(self, monkeypatch):
        """In production with Google enabled but missing error URL."""
        with patch.object(cfg, "_IS_PRODUCTION", True), \
             patch.object(cfg, "GOOGLE_CLIENT_ID", "goog-id"), \
             patch.object(cfg, "GITHUB_CLIENT_ID", ""), \
             patch.object(cfg, "ALLOWED_OAUTH_REDIRECT_HOSTS", frozenset({"example.com"})):
            with patch.object(cfg, "_get_google_client_secret", return_value="secret"), \
                 patch.object(cfg, "_get_google_redirect_uri", return_value="https://example.com/cb"), \
                 patch.object(cfg, "_get_oauth_success_url", return_value="https://example.com/ok"), \
                 patch.object(cfg, "_get_oauth_error_url", return_value=""):
                result = cfg.validate_oauth_config(log_warnings=False)
        assert "OAUTH_ERROR_URL" in result

    def test_production_google_enabled_missing_allowed_hosts(self, monkeypatch):
        """In production with Google enabled but no allowed redirect hosts."""
        with patch.object(cfg, "_IS_PRODUCTION", True), \
             patch.object(cfg, "GOOGLE_CLIENT_ID", "goog-id"), \
             patch.object(cfg, "GITHUB_CLIENT_ID", ""), \
             patch.object(cfg, "ALLOWED_OAUTH_REDIRECT_HOSTS", frozenset()):
            with patch.object(cfg, "_get_google_client_secret", return_value="secret"), \
                 patch.object(cfg, "_get_google_redirect_uri", return_value="https://example.com/cb"), \
                 patch.object(cfg, "_get_oauth_success_url", return_value="https://example.com/ok"), \
                 patch.object(cfg, "_get_oauth_error_url", return_value="https://example.com/err"):
                result = cfg.validate_oauth_config(log_warnings=False)
        assert "OAUTH_ALLOWED_REDIRECT_HOSTS" in result

    def test_production_google_all_configured(self, monkeypatch):
        """In production with Google fully configured, no missing vars."""
        with patch.object(cfg, "_IS_PRODUCTION", True), \
             patch.object(cfg, "GOOGLE_CLIENT_ID", "goog-id"), \
             patch.object(cfg, "GITHUB_CLIENT_ID", ""), \
             patch.object(cfg, "ALLOWED_OAUTH_REDIRECT_HOSTS", frozenset({"example.com"})):
            with patch.object(cfg, "_get_google_client_secret", return_value="secret"), \
                 patch.object(cfg, "_get_google_redirect_uri", return_value="https://example.com/cb"), \
                 patch.object(cfg, "_get_oauth_success_url", return_value="https://example.com/ok"), \
                 patch.object(cfg, "_get_oauth_error_url", return_value="https://example.com/err"):
                result = cfg.validate_oauth_config(log_warnings=False)
        assert result == []

    def test_production_github_enabled_missing_secret(self, monkeypatch):
        """In production with GitHub enabled but missing client secret."""
        with patch.object(cfg, "_IS_PRODUCTION", True), \
             patch.object(cfg, "GOOGLE_CLIENT_ID", ""), \
             patch.object(cfg, "GITHUB_CLIENT_ID", "gh-id"), \
             patch.object(cfg, "ALLOWED_OAUTH_REDIRECT_HOSTS", frozenset({"example.com"})):
            with patch.object(cfg, "_get_github_client_secret", return_value=""), \
                 patch.object(cfg, "_get_github_redirect_uri", return_value="https://example.com/cb"), \
                 patch.object(cfg, "_get_oauth_success_url", return_value="https://example.com/ok"), \
                 patch.object(cfg, "_get_oauth_error_url", return_value="https://example.com/err"):
                result = cfg.validate_oauth_config(log_warnings=False)
        assert "GITHUB_OAUTH_CLIENT_SECRET" in result

    def test_production_github_enabled_missing_redirect(self, monkeypatch):
        """In production with GitHub enabled but missing redirect URI."""
        with patch.object(cfg, "_IS_PRODUCTION", True), \
             patch.object(cfg, "GOOGLE_CLIENT_ID", ""), \
             patch.object(cfg, "GITHUB_CLIENT_ID", "gh-id"), \
             patch.object(cfg, "ALLOWED_OAUTH_REDIRECT_HOSTS", frozenset({"example.com"})):
            with patch.object(cfg, "_get_github_client_secret", return_value="secret"), \
                 patch.object(cfg, "_get_github_redirect_uri", return_value=""), \
                 patch.object(cfg, "_get_oauth_success_url", return_value="https://example.com/ok"), \
                 patch.object(cfg, "_get_oauth_error_url", return_value="https://example.com/err"):
                result = cfg.validate_oauth_config(log_warnings=False)
        assert "GITHUB_OAUTH_REDIRECT_URI" in result

    def test_production_github_missing_shared_urls_no_duplicate(self, monkeypatch):
        """GitHub checks don't duplicate shared URL entries already from Google."""
        with patch.object(cfg, "_IS_PRODUCTION", True), \
             patch.object(cfg, "GOOGLE_CLIENT_ID", "goog-id"), \
             patch.object(cfg, "GITHUB_CLIENT_ID", "gh-id"), \
             patch.object(cfg, "ALLOWED_OAUTH_REDIRECT_HOSTS", frozenset({"example.com"})):
            with patch.object(cfg, "_get_google_client_secret", return_value="secret"), \
                 patch.object(cfg, "_get_google_redirect_uri", return_value="https://example.com/cb"), \
                 patch.object(cfg, "_get_github_client_secret", return_value="secret"), \
                 patch.object(cfg, "_get_github_redirect_uri", return_value="https://example.com/cb"), \
                 patch.object(cfg, "_get_oauth_success_url", return_value=""), \
                 patch.object(cfg, "_get_oauth_error_url", return_value=""):
                result = cfg.validate_oauth_config(log_warnings=False)
        # OAUTH_SUCCESS_URL and OAUTH_ERROR_URL should appear only once
        assert result.count("OAUTH_SUCCESS_URL") == 1
        assert result.count("OAUTH_ERROR_URL") == 1

    def test_production_no_providers_enabled(self, monkeypatch):
        """In production with no providers enabled, only JWT checked."""
        with patch.object(cfg, "_IS_PRODUCTION", True), \
             patch.object(cfg, "GOOGLE_CLIENT_ID", ""), \
             patch.object(cfg, "GITHUB_CLIENT_ID", ""):
            result = cfg.validate_oauth_config(log_warnings=False)
        # Under pytest, JWT check is skipped
        assert result == []

    def test_log_warnings_true(self, monkeypatch, caplog):
        """With log_warnings=True, warnings are logged for missing vars."""
        with patch.object(cfg, "_IS_PRODUCTION", True), \
             patch.object(cfg, "GOOGLE_CLIENT_ID", "goog-id"), \
             patch.object(cfg, "GITHUB_CLIENT_ID", ""), \
             patch.object(cfg, "ALLOWED_OAUTH_REDIRECT_HOSTS", frozenset({"example.com"})):
            with patch.object(cfg, "_get_google_client_secret", return_value=""), \
                 patch.object(cfg, "_get_google_redirect_uri", return_value="https://example.com/cb"), \
                 patch.object(cfg, "_get_oauth_success_url", return_value="https://example.com/ok"), \
                 patch.object(cfg, "_get_oauth_error_url", return_value="https://example.com/err"):
                with caplog.at_level(logging.WARNING, logger="aragora.server.handlers.oauth.config"):
                    result = cfg.validate_oauth_config(log_warnings=True)
        assert "GOOGLE_OAUTH_CLIENT_SECRET" in result
        assert any("Missing required variables" in r.message for r in caplog.records)

    def test_log_warnings_false_no_log(self, monkeypatch, caplog):
        """With log_warnings=False, no warnings are logged."""
        with patch.object(cfg, "_IS_PRODUCTION", True), \
             patch.object(cfg, "GOOGLE_CLIENT_ID", "goog-id"), \
             patch.object(cfg, "GITHUB_CLIENT_ID", ""), \
             patch.object(cfg, "ALLOWED_OAUTH_REDIRECT_HOSTS", frozenset({"example.com"})):
            with patch.object(cfg, "_get_google_client_secret", return_value=""), \
                 patch.object(cfg, "_get_google_redirect_uri", return_value="https://example.com/cb"), \
                 patch.object(cfg, "_get_oauth_success_url", return_value="https://example.com/ok"), \
                 patch.object(cfg, "_get_oauth_error_url", return_value="https://example.com/err"):
                with caplog.at_level(logging.WARNING, logger="aragora.server.handlers.oauth.config"):
                    result = cfg.validate_oauth_config(log_warnings=False)
        assert "GOOGLE_OAUTH_CLIENT_SECRET" in result
        # "Missing required variables" should NOT be in logs
        assert not any("Missing required variables" in r.message for r in caplog.records)

    def test_jwt_check_not_under_pytest(self, monkeypatch):
        """When NOT under pytest, missing JWT secret is reported."""
        monkeypatch.delenv("ARAGORA_JWT_SECRET", raising=False)
        # Temporarily remove pytest from sys.modules to simulate non-test env
        with patch.object(cfg, "_IS_PRODUCTION", False):
            saved = sys.modules.pop("pytest", None)
            try:
                result = cfg.validate_oauth_config(log_warnings=False)
            finally:
                if saved is not None:
                    sys.modules["pytest"] = saved
        assert "ARAGORA_JWT_SECRET" in result

    def test_jwt_short_secret_not_under_pytest(self, monkeypatch):
        """When NOT under pytest, short JWT secret is reported."""
        monkeypatch.setenv("ARAGORA_JWT_SECRET", "short")
        with patch.object(cfg, "_IS_PRODUCTION", False):
            saved = sys.modules.pop("pytest", None)
            try:
                result = cfg.validate_oauth_config(log_warnings=False)
            finally:
                if saved is not None:
                    sys.modules["pytest"] = saved
        assert any("too short" in item for item in result)

    def test_jwt_valid_secret_not_under_pytest(self, monkeypatch):
        """When NOT under pytest, valid JWT secret passes check."""
        monkeypatch.setenv("ARAGORA_JWT_SECRET", "a" * 32)
        with patch.object(cfg, "_IS_PRODUCTION", False):
            saved = sys.modules.pop("pytest", None)
            try:
                result = cfg.validate_oauth_config(log_warnings=False)
            finally:
                if saved is not None:
                    sys.modules["pytest"] = saved
        assert result == []

    def test_jwt_missing_logs_warning(self, monkeypatch, caplog):
        """When NOT under pytest, missing JWT logs a warning."""
        monkeypatch.delenv("ARAGORA_JWT_SECRET", raising=False)
        with patch.object(cfg, "_IS_PRODUCTION", False):
            saved = sys.modules.pop("pytest", None)
            try:
                with caplog.at_level(logging.WARNING, logger="aragora.server.handlers.oauth.config"):
                    cfg.validate_oauth_config(log_warnings=True)
            finally:
                if saved is not None:
                    sys.modules["pytest"] = saved
        assert any("ARAGORA_JWT_SECRET" in r.message for r in caplog.records)

    def test_jwt_short_logs_warning(self, monkeypatch, caplog):
        """When NOT under pytest, short JWT logs a warning."""
        monkeypatch.setenv("ARAGORA_JWT_SECRET", "short")
        with patch.object(cfg, "_IS_PRODUCTION", False):
            saved = sys.modules.pop("pytest", None)
            try:
                with caplog.at_level(logging.WARNING, logger="aragora.server.handlers.oauth.config"):
                    cfg.validate_oauth_config(log_warnings=True)
            finally:
                if saved is not None:
                    sys.modules["pytest"] = saved
        assert any("too short" in r.message for r in caplog.records)

    def test_production_github_missing_allowed_hosts_not_duplicated(self, monkeypatch):
        """GitHub missing hosts entry is not duplicated if already from Google."""
        with patch.object(cfg, "_IS_PRODUCTION", True), \
             patch.object(cfg, "GOOGLE_CLIENT_ID", "goog-id"), \
             patch.object(cfg, "GITHUB_CLIENT_ID", "gh-id"), \
             patch.object(cfg, "ALLOWED_OAUTH_REDIRECT_HOSTS", frozenset()):
            with patch.object(cfg, "_get_google_client_secret", return_value="s"), \
                 patch.object(cfg, "_get_google_redirect_uri", return_value="u"), \
                 patch.object(cfg, "_get_github_client_secret", return_value="s"), \
                 patch.object(cfg, "_get_github_redirect_uri", return_value="u"), \
                 patch.object(cfg, "_get_oauth_success_url", return_value="u"), \
                 patch.object(cfg, "_get_oauth_error_url", return_value="u"):
                result = cfg.validate_oauth_config(log_warnings=False)
        assert result.count("OAUTH_ALLOWED_REDIRECT_HOSTS") == 1


# ===========================================================================
# get_oauth_config_status
# ===========================================================================


class TestGetOAuthConfigStatus:
    """Tests for get_oauth_config_status."""

    def test_returns_dict(self):
        """Should return a dict with expected top-level keys."""
        result = cfg.get_oauth_config_status()
        assert isinstance(result, dict)
        assert "environment" in result
        assert "jwt" in result
        assert "google" in result
        assert "github" in result
        assert "urls" in result
        assert "validation" in result

    def test_environment_section(self, monkeypatch):
        """Environment section reflects current env."""
        monkeypatch.setenv("ARAGORA_ENV", "staging")
        result = cfg.get_oauth_config_status()
        assert result["environment"]["aragora_env"] == "staging"

    def test_environment_not_set(self, monkeypatch):
        """When ARAGORA_ENV not set, shows '(not set)'."""
        monkeypatch.delenv("ARAGORA_ENV", raising=False)
        result = cfg.get_oauth_config_status()
        assert result["environment"]["aragora_env"] == "(not set)"

    def test_jwt_not_configured(self, monkeypatch):
        """When JWT secret not set, jwt section reflects that."""
        monkeypatch.delenv("ARAGORA_JWT_SECRET", raising=False)
        result = cfg.get_oauth_config_status()
        assert result["jwt"]["secret_configured"] is False
        assert result["jwt"]["secret_length"] == 0
        assert result["jwt"]["secret_valid"] is False

    def test_jwt_configured_short(self, monkeypatch):
        """Short JWT secret is detected."""
        monkeypatch.setenv("ARAGORA_JWT_SECRET", "short")
        result = cfg.get_oauth_config_status()
        assert result["jwt"]["secret_configured"] is True
        assert result["jwt"]["secret_length"] == 5
        assert result["jwt"]["secret_valid"] is False

    def test_jwt_configured_valid(self, monkeypatch):
        """Valid JWT secret (32+ chars) is detected."""
        monkeypatch.setenv("ARAGORA_JWT_SECRET", "a" * 64)
        result = cfg.get_oauth_config_status()
        assert result["jwt"]["secret_configured"] is True
        assert result["jwt"]["secret_length"] == 64
        assert result["jwt"]["secret_valid"] is True

    def test_google_section_not_set(self):
        """When Google not configured, shows not enabled."""
        with patch.object(cfg, "GOOGLE_CLIENT_ID", ""):
            with patch.object(cfg, "_get_google_client_secret", return_value=""):
                result = cfg.get_oauth_config_status()
        assert result["google"]["enabled"] is False
        assert result["google"]["client_id"] == "(not set)"

    def test_google_section_configured(self):
        """When Google is configured, shows enabled with masked ID."""
        with patch.object(cfg, "GOOGLE_CLIENT_ID", "1234567890abcdef"):
            with patch.object(cfg, "_get_google_client_secret", return_value="secret"):
                result = cfg.get_oauth_config_status()
        assert result["google"]["enabled"] is True
        assert result["google"]["client_secret_set"] is True
        # Client ID should be masked: first 4 + ... + last 4
        assert result["google"]["client_id"] == "1234...cdef"

    def test_github_section_not_set(self):
        """When GitHub not configured, shows not enabled."""
        with patch.object(cfg, "GITHUB_CLIENT_ID", ""):
            with patch.object(cfg, "_get_github_client_secret", return_value=""):
                result = cfg.get_oauth_config_status()
        assert result["github"]["enabled"] is False

    def test_github_section_configured(self):
        """When GitHub is configured, shows enabled with masked ID."""
        with patch.object(cfg, "GITHUB_CLIENT_ID", "abcdefghijklmnop"):
            with patch.object(cfg, "_get_github_client_secret", return_value="secret"):
                result = cfg.get_oauth_config_status()
        assert result["github"]["enabled"] is True
        assert result["github"]["client_secret_set"] is True
        assert result["github"]["client_id"] == "abcd...mnop"

    def test_urls_section(self):
        """URLs section includes success, error, and allowed hosts."""
        result = cfg.get_oauth_config_status()
        assert "success_url" in result["urls"]
        assert "error_url" in result["urls"]
        assert "allowed_redirect_hosts" in result["urls"]

    def test_validation_section(self):
        """Validation section includes missing_vars and is_valid."""
        result = cfg.get_oauth_config_status()
        assert "missing_vars" in result["validation"]
        assert "is_valid" in result["validation"]
        assert isinstance(result["validation"]["missing_vars"], list)
        assert isinstance(result["validation"]["is_valid"], bool)

    def test_validation_is_valid_when_no_missing(self):
        """is_valid is True when no vars are missing."""
        with patch.object(cfg, "_IS_PRODUCTION", False):
            result = cfg.get_oauth_config_status()
        # In dev mode under pytest, validation passes
        assert result["validation"]["is_valid"] is True

    def test_urls_section_shows_values(self):
        """URLs show actual values or '(not set)'."""
        with patch.object(cfg, "_get_oauth_success_url", return_value=""):
            with patch.object(cfg, "_get_oauth_error_url", return_value=""):
                result = cfg.get_oauth_config_status()
        assert result["urls"]["success_url"] == "(not set)"
        assert result["urls"]["error_url"] == "(not set)"

    def test_allowed_hosts_empty_shows_none(self):
        """When no hosts configured, shows ['(none)']."""
        with patch.object(cfg, "ALLOWED_OAUTH_REDIRECT_HOSTS", frozenset()):
            result = cfg.get_oauth_config_status()
        assert result["urls"]["allowed_redirect_hosts"] == ["(none)"]


# ===========================================================================
# Masking helper (_mask inside get_oauth_config_status)
# ===========================================================================


class TestMaskFunction:
    """Tests for the _mask function embedded in get_oauth_config_status."""

    def test_empty_string_masked(self):
        """Empty string shows '(not set)'."""
        with patch.object(cfg, "GOOGLE_CLIENT_ID", ""):
            result = cfg.get_oauth_config_status()
        assert result["google"]["client_id"] == "(not set)"

    def test_short_string_masked(self):
        """Strings <= 8 chars show '****'."""
        with patch.object(cfg, "GOOGLE_CLIENT_ID", "abc"):
            result = cfg.get_oauth_config_status()
        assert result["google"]["client_id"] == "****"

    def test_exactly_8_chars_masked(self):
        """Exactly 8 chars shows '****'."""
        with patch.object(cfg, "GOOGLE_CLIENT_ID", "12345678"):
            result = cfg.get_oauth_config_status()
        assert result["google"]["client_id"] == "****"

    def test_9_chars_shows_partial(self):
        """9 chars shows first 4 ... last 4."""
        with patch.object(cfg, "GOOGLE_CLIENT_ID", "123456789"):
            result = cfg.get_oauth_config_status()
        assert result["google"]["client_id"] == "1234...6789"

    def test_long_string_masked(self):
        """Long strings show first 4 ... last 4."""
        with patch.object(cfg, "GOOGLE_CLIENT_ID", "abcdefghijklmnop"):
            result = cfg.get_oauth_config_status()
        assert result["google"]["client_id"] == "abcd...mnop"


# ===========================================================================
# Provider endpoint constants
# ===========================================================================


class TestProviderEndpointConstants:
    """Tests for provider endpoint URL constants."""

    # Google
    def test_google_auth_url(self):
        assert cfg.GOOGLE_AUTH_URL == "https://accounts.google.com/o/oauth2/v2/auth"

    def test_google_token_url(self):
        assert cfg.GOOGLE_TOKEN_URL == "https://oauth2.googleapis.com/token"

    def test_google_userinfo_url(self):
        assert cfg.GOOGLE_USERINFO_URL == "https://www.googleapis.com/oauth2/v2/userinfo"

    # GitHub
    def test_github_auth_url(self):
        assert cfg.GITHUB_AUTH_URL == "https://github.com/login/oauth/authorize"

    def test_github_token_url(self):
        assert cfg.GITHUB_TOKEN_URL == "https://github.com/login/oauth/access_token"

    def test_github_userinfo_url(self):
        assert cfg.GITHUB_USERINFO_URL == "https://api.github.com/user"

    def test_github_emails_url(self):
        assert cfg.GITHUB_EMAILS_URL == "https://api.github.com/user/emails"

    # Microsoft
    def test_microsoft_auth_url_template(self):
        assert "{tenant}" in cfg.MICROSOFT_AUTH_URL_TEMPLATE
        assert "authorize" in cfg.MICROSOFT_AUTH_URL_TEMPLATE

    def test_microsoft_token_url_template(self):
        assert "{tenant}" in cfg.MICROSOFT_TOKEN_URL_TEMPLATE
        assert "token" in cfg.MICROSOFT_TOKEN_URL_TEMPLATE

    def test_microsoft_auth_url_format(self):
        """Microsoft auth URL template can be formatted with tenant."""
        url = cfg.MICROSOFT_AUTH_URL_TEMPLATE.format(tenant="my-tenant")
        assert "my-tenant" in url
        assert "authorize" in url

    def test_microsoft_token_url_format(self):
        """Microsoft token URL template can be formatted with tenant."""
        url = cfg.MICROSOFT_TOKEN_URL_TEMPLATE.format(tenant="my-tenant")
        assert "my-tenant" in url
        assert "token" in url

    def test_microsoft_userinfo_url(self):
        assert cfg.MICROSOFT_USERINFO_URL == "https://graph.microsoft.com/v1.0/me"

    # Apple
    def test_apple_auth_url(self):
        assert cfg.APPLE_AUTH_URL == "https://appleid.apple.com/auth/authorize"

    def test_apple_token_url(self):
        assert cfg.APPLE_TOKEN_URL == "https://appleid.apple.com/auth/token"

    def test_apple_keys_url(self):
        assert cfg.APPLE_KEYS_URL == "https://appleid.apple.com/auth/keys"

    # URL scheme checks
    def test_all_endpoints_use_https(self):
        """All endpoint constants use HTTPS."""
        endpoints = [
            cfg.GOOGLE_AUTH_URL,
            cfg.GOOGLE_TOKEN_URL,
            cfg.GOOGLE_USERINFO_URL,
            cfg.GITHUB_AUTH_URL,
            cfg.GITHUB_TOKEN_URL,
            cfg.GITHUB_USERINFO_URL,
            cfg.GITHUB_EMAILS_URL,
            cfg.MICROSOFT_USERINFO_URL,
            cfg.APPLE_AUTH_URL,
            cfg.APPLE_TOKEN_URL,
            cfg.APPLE_KEYS_URL,
        ]
        for url in endpoints:
            assert url.startswith("https://"), f"Endpoint does not use HTTPS: {url}"


# ===========================================================================
# Legacy module-level variables
# ===========================================================================


class TestLegacyModuleLevelVars:
    """Tests for backward-compatible module-level variables."""

    def test_is_production_exists(self):
        """_IS_PRODUCTION module-level var exists."""
        assert hasattr(cfg, "_IS_PRODUCTION")
        assert isinstance(cfg._IS_PRODUCTION, bool)

    def test_google_client_id_exists(self):
        """GOOGLE_CLIENT_ID module-level var exists and is a string."""
        assert hasattr(cfg, "GOOGLE_CLIENT_ID")
        assert isinstance(cfg.GOOGLE_CLIENT_ID, str)

    def test_google_client_secret_exists(self):
        """GOOGLE_CLIENT_SECRET module-level var exists and is a string."""
        assert hasattr(cfg, "GOOGLE_CLIENT_SECRET")
        assert isinstance(cfg.GOOGLE_CLIENT_SECRET, str)

    def test_github_client_id_exists(self):
        """GITHUB_CLIENT_ID module-level var exists and is a string."""
        assert hasattr(cfg, "GITHUB_CLIENT_ID")
        assert isinstance(cfg.GITHUB_CLIENT_ID, str)

    def test_github_client_secret_exists(self):
        """GITHUB_CLIENT_SECRET module-level var exists and is a string."""
        assert hasattr(cfg, "GITHUB_CLIENT_SECRET")
        assert isinstance(cfg.GITHUB_CLIENT_SECRET, str)

    def test_google_redirect_uri_exists(self):
        """GOOGLE_REDIRECT_URI module-level var exists and is a string."""
        assert hasattr(cfg, "GOOGLE_REDIRECT_URI")
        assert isinstance(cfg.GOOGLE_REDIRECT_URI, str)

    def test_github_redirect_uri_exists(self):
        """GITHUB_REDIRECT_URI module-level var exists and is a string."""
        assert hasattr(cfg, "GITHUB_REDIRECT_URI")
        assert isinstance(cfg.GITHUB_REDIRECT_URI, str)

    def test_oauth_success_url_exists(self):
        """OAUTH_SUCCESS_URL module-level var exists and is a string."""
        assert hasattr(cfg, "OAUTH_SUCCESS_URL")
        assert isinstance(cfg.OAUTH_SUCCESS_URL, str)

    def test_oauth_error_url_exists(self):
        """OAUTH_ERROR_URL module-level var exists and is a string."""
        assert hasattr(cfg, "OAUTH_ERROR_URL")
        assert isinstance(cfg.OAUTH_ERROR_URL, str)

    def test_allowed_oauth_redirect_hosts_exists(self):
        """ALLOWED_OAUTH_REDIRECT_HOSTS exists and is a frozenset."""
        assert hasattr(cfg, "ALLOWED_OAUTH_REDIRECT_HOSTS")
        assert isinstance(cfg.ALLOWED_OAUTH_REDIRECT_HOSTS, frozenset)


# ===========================================================================
# Security edge cases
# ===========================================================================


class TestSecurityEdgeCases:
    """Security-related edge case tests."""

    def test_secrets_not_leaked_in_config_status(self):
        """Config status never includes full secret values."""
        with patch.object(cfg, "GOOGLE_CLIENT_ID", "supersecretclientid123456"):
            with patch.object(cfg, "GITHUB_CLIENT_ID", "anothersecretid987654321"):
                result = cfg.get_oauth_config_status()
        # Full values should not appear
        assert "supersecretclientid123456" not in str(result)
        assert "anothersecretid987654321" not in str(result)

    def test_jwt_secret_not_in_config_status(self, monkeypatch):
        """JWT secret value is never included in config status output."""
        monkeypatch.setenv("ARAGORA_JWT_SECRET", "my-ultra-secret-jwt-key-123456789")
        result = cfg.get_oauth_config_status()
        result_str = str(result)
        assert "my-ultra-secret-jwt-key-123456789" not in result_str

    def test_client_secrets_not_in_status(self):
        """Client secret values are not returned, only boolean flags."""
        with patch.object(cfg, "_get_google_client_secret", return_value="google-secret-value"):
            with patch.object(cfg, "_get_github_client_secret", return_value="github-secret-value"):
                result = cfg.get_oauth_config_status()
        result_str = str(result)
        assert "google-secret-value" not in result_str
        assert "github-secret-value" not in result_str
        # Only boolean flags
        assert result["google"]["client_secret_set"] is True
        assert result["github"]["client_secret_set"] is True

    def test_path_traversal_in_env_var_name(self, monkeypatch):
        """_get_secret handles unusual env var names safely."""
        # This should just return default since the env var with special chars
        # won't exist
        with patch.dict(sys.modules, {"aragora.config.secrets": None}):
            result = cfg._get_secret("../../etc/passwd", "safe-default")
        assert result == "safe-default"

    def test_null_bytes_in_secret_name(self, monkeypatch):
        """_get_secret handles null bytes in name gracefully."""
        with patch.dict(sys.modules, {"aragora.config.secrets": None}):
            # os.environ.get raises ValueError on null bytes, which is fine
            # because the function catches it or returns the default
            try:
                result = cfg._get_secret("VALID_KEY\x00INJECTION", "default")
                # If it doesn't raise, we get the default
                assert result == "default"
            except ValueError:
                # Expected: null bytes cause ValueError in os.environ
                pass

    def test_production_env_injection(self, monkeypatch):
        """Setting ARAGORA_ENV with padding doesn't fool production check."""
        monkeypatch.setenv("ARAGORA_ENV", " production ")
        # .lower() is called but not .strip(), so this should NOT be production
        assert cfg._is_production() is False

    def test_allowed_hosts_with_unusual_chars(self, monkeypatch):
        """Hosts with unusual but valid characters are included."""
        monkeypatch.setenv("OAUTH_ALLOWED_REDIRECT_HOSTS", "evil.com,normal.com,sub-domain.test.com")
        with patch.dict(sys.modules, {"aragora.config.secrets": None}):
            result = cfg._get_allowed_redirect_hosts()
        # The function strips and lowercases, but doesn't validate hostnames
        assert "normal.com" in result
        assert "evil.com" in result
        assert "sub-domain.test.com" in result


# ===========================================================================
# Integration-like tests
# ===========================================================================


class TestIntegration:
    """Integration-style tests combining multiple functions."""

    def test_config_status_validation_consistent(self):
        """Config status validation section matches direct validate call."""
        with patch.object(cfg, "_IS_PRODUCTION", False):
            status = cfg.get_oauth_config_status()
            direct = cfg.validate_oauth_config(log_warnings=False)
        assert status["validation"]["missing_vars"] == direct
        assert status["validation"]["is_valid"] == (len(direct) == 0)

    def test_dev_mode_full_flow(self, monkeypatch):
        """In dev mode, all URLs have sensible defaults."""
        monkeypatch.delenv("ARAGORA_ENV", raising=False)
        with patch.dict(sys.modules, {"aragora.config.secrets": None}):
            google_redirect = cfg._get_google_redirect_uri()
            github_redirect = cfg._get_github_redirect_uri()
            ms_redirect = cfg._get_microsoft_redirect_uri()
            apple_redirect = cfg._get_apple_redirect_uri()
            oidc_redirect = cfg._get_oidc_redirect_uri()
            success_url = cfg._get_oauth_success_url()
            error_url = cfg._get_oauth_error_url()
            hosts = cfg._get_allowed_redirect_hosts()

        # All dev defaults should be non-empty
        assert google_redirect
        assert github_redirect
        assert ms_redirect
        assert apple_redirect
        assert oidc_redirect
        assert success_url
        assert error_url
        assert len(hosts) >= 2  # localhost + 127.0.0.1

        # Redirects should point to localhost
        for url in [google_redirect, github_redirect, ms_redirect, apple_redirect, oidc_redirect]:
            assert "localhost" in url

    def test_production_mode_requires_explicit_config(self, monkeypatch):
        """In production mode, nothing defaults - all empty without env vars."""
        with patch.dict(sys.modules, {"aragora.config.secrets": None}):
            with patch.object(cfg, "_is_production", return_value=True):
                google_redirect = cfg._get_google_redirect_uri()
                github_redirect = cfg._get_github_redirect_uri()
                ms_redirect = cfg._get_microsoft_redirect_uri()
                apple_redirect = cfg._get_apple_redirect_uri()
                oidc_redirect = cfg._get_oidc_redirect_uri()
                success_url = cfg._get_oauth_success_url()
                error_url = cfg._get_oauth_error_url()
                hosts = cfg._get_allowed_redirect_hosts()

        assert google_redirect == ""
        assert github_redirect == ""
        assert ms_redirect == ""
        assert apple_redirect == ""
        assert oidc_redirect == ""
        assert success_url == ""
        assert error_url == ""
        assert hosts == frozenset()

    def test_mixed_providers_validation(self, monkeypatch):
        """Validate with one provider fully configured and another partially."""
        with patch.object(cfg, "_IS_PRODUCTION", True), \
             patch.object(cfg, "GOOGLE_CLIENT_ID", "goog-id"), \
             patch.object(cfg, "GITHUB_CLIENT_ID", "gh-id"), \
             patch.object(cfg, "ALLOWED_OAUTH_REDIRECT_HOSTS", frozenset({"example.com"})):
            # Google fully configured
            with patch.object(cfg, "_get_google_client_secret", return_value="gs"), \
                 patch.object(cfg, "_get_google_redirect_uri", return_value="https://example.com/g"), \
                 patch.object(cfg, "_get_oauth_success_url", return_value="https://example.com/ok"), \
                 patch.object(cfg, "_get_oauth_error_url", return_value="https://example.com/err"), \
                 patch.object(cfg, "_get_github_client_secret", return_value=""), \
                 patch.object(cfg, "_get_github_redirect_uri", return_value="https://example.com/gh"):
                result = cfg.validate_oauth_config(log_warnings=False)
        # Google should pass, GitHub should fail on client_secret
        assert "GOOGLE_OAUTH_CLIENT_SECRET" not in result
        assert "GITHUB_OAUTH_CLIENT_SECRET" in result
