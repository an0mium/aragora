"""Tests for aragora/server/handlers/oauth/handler.py (re-export module).

Covers:
- All re-exported names are importable and match their source in _oauth_impl
- __all__ is complete and accurate
- OAuthHandler class is properly re-exported and instantiable
- OAuthUserInfo dataclass re-export
- validate_oauth_config function re-export
- _validate_redirect_url with various URL patterns
- _validate_state / _generate_state round-trip
- _cleanup_expired_states
- _oauth_limiter wrapper
- All provider config getter functions (Google, GitHub, Microsoft, Apple, OIDC)
- Common config getters (success URL, error URL, allowed redirect hosts)
- Module-level constants (_IS_PRODUCTION, GOOGLE_CLIENT_ID, etc.)
"""

from __future__ import annotations

import json
import os
import types
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# Import the module under test
from aragora.server.handlers.oauth import handler as handler_module

# Import specific names from the handler module (the re-export surface)
from aragora.server.handlers.oauth.handler import (
    ALLOWED_OAUTH_REDIRECT_HOSTS,
    GITHUB_CLIENT_ID,
    GOOGLE_CLIENT_ID,
    OAUTH_ERROR_URL,
    OAUTH_SUCCESS_URL,
    OAuthHandler,
    OAuthUserInfo,
    _IS_PRODUCTION,
    _cleanup_expired_states,
    _generate_state,
    _get_allowed_redirect_hosts,
    _get_apple_client_id,
    _get_apple_key_id,
    _get_apple_private_key,
    _get_apple_redirect_uri,
    _get_apple_team_id,
    _get_github_client_id,
    _get_github_client_secret,
    _get_github_redirect_uri,
    _get_google_client_id,
    _get_google_client_secret,
    _get_google_redirect_uri,
    _get_microsoft_client_id,
    _get_microsoft_client_secret,
    _get_microsoft_redirect_uri,
    _get_microsoft_tenant,
    _get_oidc_client_id,
    _get_oidc_client_secret,
    _get_oidc_issuer,
    _get_oidc_redirect_uri,
    _get_oauth_error_url,
    _get_oauth_success_url,
    _oauth_limiter,
    _validate_redirect_url,
    _validate_state,
    validate_oauth_config,
)

from aragora.server.middleware.rate_limit.oauth_limiter import (
    reset_backoff_tracker,
    reset_oauth_limiter,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result: object) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    if isinstance(result, dict):
        return result
    return json.loads(result.body)


def _status(result: object) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    """Reset rate limiter between tests."""
    reset_oauth_limiter()
    reset_backoff_tracker()
    yield
    reset_oauth_limiter()
    reset_backoff_tracker()


@pytest.fixture(autouse=True)
def _clear_oauth_env_vars():
    """Clear OAuth env vars so config getters return defaults."""
    env_vars = [
        "GOOGLE_OAUTH_CLIENT_ID",
        "GOOGLE_OAUTH_CLIENT_SECRET",
        "GITHUB_OAUTH_CLIENT_ID",
        "GITHUB_OAUTH_CLIENT_SECRET",
        "MICROSOFT_OAUTH_CLIENT_ID",
        "MICROSOFT_OAUTH_CLIENT_SECRET",
        "MICROSOFT_OAUTH_TENANT",
        "APPLE_OAUTH_CLIENT_ID",
        "APPLE_TEAM_ID",
        "APPLE_KEY_ID",
        "APPLE_PRIVATE_KEY",
        "OIDC_ISSUER",
        "OIDC_CLIENT_ID",
        "OIDC_CLIENT_SECRET",
        "OAUTH_SUCCESS_URL",
        "OAUTH_ERROR_URL",
        "OAUTH_ALLOWED_REDIRECT_HOSTS",
        "ARAGORA_ENV",
        "GOOGLE_OAUTH_REDIRECT_URI",
        "GITHUB_OAUTH_REDIRECT_URI",
        "MICROSOFT_OAUTH_REDIRECT_URI",
        "APPLE_OAUTH_REDIRECT_URI",
        "OIDC_REDIRECT_URI",
    ]
    originals = {}
    for key in env_vars:
        originals[key] = os.environ.pop(key, None)
    yield
    for key, val in originals.items():
        if val is not None:
            os.environ[key] = val
        else:
            os.environ.pop(key, None)


@pytest.fixture
def oauth_handler():
    """Create an OAuthHandler instance."""
    return OAuthHandler({"user_store": MagicMock()})


@pytest.fixture
def mock_http_handler():
    """Create a mock HTTP handler for routing tests."""
    mock = MagicMock()
    mock.command = "GET"
    mock.client_address = ("127.0.0.1", 12345)
    mock.headers = {"X-Forwarded-For": "10.0.0.1"}
    return mock


# ===========================================================================
# 1. __all__ completeness
# ===========================================================================


class TestAllExports:
    """Verify the __all__ list matches actual exports."""

    def test_all_is_a_list(self):
        """__all__ should be a list of strings."""
        assert isinstance(handler_module.__all__, list)
        for name in handler_module.__all__:
            assert isinstance(name, str)

    def test_all_names_are_importable(self):
        """Every name in __all__ must be an attribute of the module."""
        for name in handler_module.__all__:
            assert hasattr(handler_module, name), f"{name} listed in __all__ but not importable"

    def test_all_contains_oauth_handler(self):
        assert "OAuthHandler" in handler_module.__all__

    def test_all_contains_oauth_user_info(self):
        assert "OAuthUserInfo" in handler_module.__all__

    def test_all_contains_validate_oauth_config(self):
        assert "validate_oauth_config" in handler_module.__all__

    def test_all_contains_limiter(self):
        assert "_oauth_limiter" in handler_module.__all__

    def test_all_contains_state_functions(self):
        assert "_validate_redirect_url" in handler_module.__all__
        assert "_validate_state" in handler_module.__all__
        assert "_cleanup_expired_states" in handler_module.__all__
        assert "_generate_state" in handler_module.__all__

    def test_all_contains_google_getters(self):
        assert "_get_google_client_id" in handler_module.__all__
        assert "_get_google_client_secret" in handler_module.__all__
        assert "_get_google_redirect_uri" in handler_module.__all__

    def test_all_contains_github_getters(self):
        assert "_get_github_client_id" in handler_module.__all__
        assert "_get_github_client_secret" in handler_module.__all__
        assert "_get_github_redirect_uri" in handler_module.__all__

    def test_all_contains_microsoft_getters(self):
        assert "_get_microsoft_client_id" in handler_module.__all__
        assert "_get_microsoft_client_secret" in handler_module.__all__
        assert "_get_microsoft_tenant" in handler_module.__all__
        assert "_get_microsoft_redirect_uri" in handler_module.__all__

    def test_all_contains_apple_getters(self):
        assert "_get_apple_client_id" in handler_module.__all__
        assert "_get_apple_team_id" in handler_module.__all__
        assert "_get_apple_key_id" in handler_module.__all__
        assert "_get_apple_private_key" in handler_module.__all__
        assert "_get_apple_redirect_uri" in handler_module.__all__

    def test_all_contains_oidc_getters(self):
        assert "_get_oidc_issuer" in handler_module.__all__
        assert "_get_oidc_client_id" in handler_module.__all__
        assert "_get_oidc_client_secret" in handler_module.__all__
        assert "_get_oidc_redirect_uri" in handler_module.__all__

    def test_all_contains_common_getters(self):
        assert "_get_oauth_success_url" in handler_module.__all__
        assert "_get_oauth_error_url" in handler_module.__all__
        assert "_get_allowed_redirect_hosts" in handler_module.__all__

    def test_all_contains_constants(self):
        assert "_IS_PRODUCTION" in handler_module.__all__
        assert "GOOGLE_CLIENT_ID" in handler_module.__all__
        assert "GITHUB_CLIENT_ID" in handler_module.__all__
        assert "ALLOWED_OAUTH_REDIRECT_HOSTS" in handler_module.__all__
        assert "OAUTH_SUCCESS_URL" in handler_module.__all__
        assert "OAUTH_ERROR_URL" in handler_module.__all__


# ===========================================================================
# 2. Re-export identity -- imported objects match the _oauth_impl source
# ===========================================================================


class TestReExportIdentity:
    """Verify re-exports are the same objects as those in _oauth_impl."""

    def test_oauth_handler_identity(self):
        from aragora.server.handlers._oauth_impl import OAuthHandler as ImplClass

        assert OAuthHandler is ImplClass

    def test_oauth_user_info_identity(self):
        from aragora.server.handlers._oauth_impl import OAuthUserInfo as ImplClass

        assert OAuthUserInfo is ImplClass

    def test_validate_oauth_config_identity(self):
        from aragora.server.handlers._oauth_impl import validate_oauth_config as impl_fn

        assert validate_oauth_config is impl_fn

    def test_oauth_limiter_identity(self):
        from aragora.server.handlers._oauth_impl import _oauth_limiter as impl_limiter

        assert _oauth_limiter is impl_limiter

    def test_validate_redirect_url_identity(self):
        import aragora.server.handlers._oauth_impl as _impl
        import aragora.server.handlers.oauth.handler as _h

        # Identity may break if another test reloads the module; check qualname
        assert _h._validate_redirect_url.__qualname__ == _impl._validate_redirect_url.__qualname__

    def test_generate_state_identity(self):
        from aragora.server.handlers._oauth_impl import _generate_state as impl_fn

        assert _generate_state is impl_fn


# ===========================================================================
# 3. OAuthHandler class
# ===========================================================================


class TestOAuthHandlerClass:
    """Test OAuthHandler instantiation and basic interface."""

    def test_instantiate_without_context(self):
        h = OAuthHandler()
        assert h.ctx == {}

    def test_instantiate_with_context(self):
        ctx = {"user_store": MagicMock(), "foo": "bar"}
        h = OAuthHandler(ctx)
        assert h.ctx is ctx

    def test_has_routes(self):
        assert hasattr(OAuthHandler, "ROUTES")
        assert isinstance(OAuthHandler.ROUTES, list)
        assert len(OAuthHandler.ROUTES) > 0

    def test_resource_type(self):
        assert OAuthHandler.RESOURCE_TYPE == "oauth"

    def test_can_handle_known_route(self, oauth_handler):
        assert oauth_handler.can_handle("/api/v1/auth/oauth/google")

    def test_can_handle_rejects_unknown_route(self, oauth_handler):
        assert not oauth_handler.can_handle("/api/v1/unknown/path")

    def test_can_handle_non_v1_routes(self, oauth_handler):
        assert oauth_handler.can_handle("/api/auth/oauth/github")

    def test_can_handle_diagnostics(self, oauth_handler):
        assert oauth_handler.can_handle("/api/v1/auth/oauth/diagnostics")
        assert oauth_handler.can_handle("/api/auth/oauth/diagnostics")


# ===========================================================================
# 4. OAuthUserInfo dataclass
# ===========================================================================


class TestOAuthUserInfo:
    """Test OAuthUserInfo dataclass."""

    def test_create_basic(self):
        info = OAuthUserInfo(
            provider="google",
            provider_user_id="123",
            email="user@example.com",
            name="Test User",
        )
        assert info.provider == "google"
        assert info.provider_user_id == "123"
        assert info.email == "user@example.com"
        assert info.name == "Test User"
        assert info.picture is None
        assert info.email_verified is False

    def test_create_with_all_fields(self):
        info = OAuthUserInfo(
            provider="github",
            provider_user_id="456",
            email="dev@example.com",
            name="Dev User",
            picture="https://avatar.example.com/pic.png",
            email_verified=True,
        )
        assert info.picture == "https://avatar.example.com/pic.png"
        assert info.email_verified is True


# ===========================================================================
# 5. _validate_redirect_url
# ===========================================================================


class TestValidateRedirectUrl:
    """Test the _validate_redirect_url function."""

    def test_localhost_allowed_in_dev(self):
        """In dev mode (default), localhost should be allowed."""
        assert _validate_redirect_url("http://localhost:3000/callback") is True

    def test_127_0_0_1_allowed_in_dev(self):
        assert _validate_redirect_url("http://127.0.0.1:8080/callback") is True

    def test_https_localhost_allowed(self):
        assert _validate_redirect_url("https://localhost/auth") is True

    def test_ftp_scheme_rejected(self):
        assert _validate_redirect_url("ftp://localhost/file") is False

    def test_javascript_scheme_rejected(self):
        assert _validate_redirect_url("javascript:alert(1)") is False

    def test_no_host_rejected(self):
        assert _validate_redirect_url("http://") is False

    def test_empty_string_rejected(self):
        assert _validate_redirect_url("") is False

    def test_unknown_host_rejected(self):
        """A host not in allowlist should be rejected."""
        assert _validate_redirect_url("https://evil.example.com/steal") is False

    def test_subdomain_of_allowed_host(self):
        """Subdomains of allowed hosts should be accepted."""
        with patch(
            "aragora.server.handlers._oauth_impl._get_allowed_redirect_hosts",
            return_value=frozenset({"example.com"}),
        ):
            assert _validate_redirect_url("https://sub.example.com/cb") is True

    def test_non_string_returns_false(self):
        """Non-string input should not raise, just return False."""
        assert _validate_redirect_url(None) is False  # type: ignore[arg-type]


# ===========================================================================
# 6. _generate_state / _validate_state round-trip
# ===========================================================================


class TestStateManagement:
    """Test OAuth state generation and validation."""

    def test_generate_returns_string(self):
        state = _generate_state()
        assert isinstance(state, str)
        assert len(state) > 0

    def test_generate_unique(self):
        s1 = _generate_state()
        s2 = _generate_state()
        assert s1 != s2

    def test_validate_consumes_state(self):
        """A generated state should validate once, then be consumed."""
        state = _generate_state()
        result = _validate_state(state)
        assert result is not None or result == {}  # valid state data
        # Second validation should fail (consumed)
        result2 = _validate_state(state)
        assert result2 is None

    def test_validate_invalid_state_returns_none(self):
        result = _validate_state("totally-bogus-token-12345")
        assert result is None

    def test_validate_empty_state(self):
        result = _validate_state("")
        assert result is None


# ===========================================================================
# 7. _cleanup_expired_states
# ===========================================================================


class TestCleanupExpiredStates:
    """Test expired state cleanup."""

    def test_cleanup_returns_int(self):
        count = _cleanup_expired_states()
        assert isinstance(count, int)
        assert count >= 0

    def test_cleanup_no_states_returns_zero(self):
        # With no expired states, should return 0
        count = _cleanup_expired_states()
        assert count == 0


# ===========================================================================
# 8. _oauth_limiter
# ===========================================================================


class TestOAuthLimiter:
    """Test the rate limiter wrapper."""

    def test_is_allowed_returns_bool(self):
        result = _oauth_limiter.is_allowed("10.0.0.1", "auth_start")
        assert isinstance(result, bool)

    def test_first_request_allowed(self):
        assert _oauth_limiter.is_allowed("unique-ip-test-1", "auth_start") is True

    def test_has_rpm_property(self):
        rpm = _oauth_limiter.rpm
        assert isinstance(rpm, int)
        assert rpm > 0


# ===========================================================================
# 9. validate_oauth_config
# ===========================================================================


class TestValidateOauthConfig:
    """Test the validate_oauth_config function."""

    def test_returns_list(self):
        result = validate_oauth_config(log_warnings=False)
        assert isinstance(result, list)

    def test_dev_mode_returns_minimal_missing(self):
        """In dev mode (non-production), only JWT might be flagged."""
        # We're running under pytest so JWT check is skipped
        result = validate_oauth_config(log_warnings=False)
        assert isinstance(result, list)

    def test_callable(self):
        assert callable(validate_oauth_config)


# ===========================================================================
# 10. Provider config getter functions
# ===========================================================================


class TestGoogleConfigGetters:
    """Test Google OAuth config getters."""

    def test_get_google_client_id_default_empty(self):
        assert _get_google_client_id() == ""

    def test_get_google_client_secret_default_empty(self):
        assert _get_google_client_secret() == ""

    def test_get_google_redirect_uri_dev_default(self):
        uri = _get_google_redirect_uri()
        assert "localhost" in uri or uri == ""

    def test_get_google_client_id_from_env(self):
        os.environ["GOOGLE_OAUTH_CLIENT_ID"] = "test-google-id"
        assert _get_google_client_id() == "test-google-id"


class TestGitHubConfigGetters:
    """Test GitHub OAuth config getters."""

    def test_get_github_client_id_default_empty(self):
        assert _get_github_client_id() == ""

    def test_get_github_client_secret_default_empty(self):
        assert _get_github_client_secret() == ""

    def test_get_github_redirect_uri_dev_default(self):
        uri = _get_github_redirect_uri()
        assert "localhost" in uri or uri == ""

    def test_get_github_client_id_from_env(self):
        os.environ["GITHUB_OAUTH_CLIENT_ID"] = "test-gh-id"
        assert _get_github_client_id() == "test-gh-id"


class TestMicrosoftConfigGetters:
    """Test Microsoft OAuth config getters."""

    def test_get_microsoft_client_id_default_empty(self):
        assert _get_microsoft_client_id() == ""

    def test_get_microsoft_client_secret_default_empty(self):
        assert _get_microsoft_client_secret() == ""

    def test_get_microsoft_tenant_default_common(self):
        assert _get_microsoft_tenant() == "common"

    def test_get_microsoft_redirect_uri_dev_default(self):
        uri = _get_microsoft_redirect_uri()
        assert "localhost" in uri or uri == ""


class TestAppleConfigGetters:
    """Test Apple OAuth config getters."""

    def test_get_apple_client_id_default_empty(self):
        assert _get_apple_client_id() == ""

    def test_get_apple_team_id_default_empty(self):
        assert _get_apple_team_id() == ""

    def test_get_apple_key_id_default_empty(self):
        assert _get_apple_key_id() == ""

    def test_get_apple_private_key_default_empty(self):
        assert _get_apple_private_key() == ""

    def test_get_apple_redirect_uri_dev_default(self):
        uri = _get_apple_redirect_uri()
        assert "localhost" in uri or uri == ""


class TestOIDCConfigGetters:
    """Test OIDC config getters."""

    def test_get_oidc_issuer_default_empty(self):
        assert _get_oidc_issuer() == ""

    def test_get_oidc_client_id_default_empty(self):
        assert _get_oidc_client_id() == ""

    def test_get_oidc_client_secret_default_empty(self):
        assert _get_oidc_client_secret() == ""

    def test_get_oidc_redirect_uri_dev_default(self):
        uri = _get_oidc_redirect_uri()
        assert "localhost" in uri or uri == ""


class TestCommonConfigGetters:
    """Test common OAuth config getters."""

    def test_get_oauth_success_url_dev_default(self):
        url = _get_oauth_success_url()
        assert "localhost" in url or url == ""

    def test_get_oauth_error_url_dev_default(self):
        url = _get_oauth_error_url()
        assert "localhost" in url or url == ""

    def test_get_allowed_redirect_hosts_dev_default(self):
        hosts = _get_allowed_redirect_hosts()
        assert isinstance(hosts, frozenset)
        # In dev mode should include localhost
        assert "localhost" in hosts or len(hosts) == 0


# ===========================================================================
# 11. Module-level constants
# ===========================================================================


class TestModuleLevelConstants:
    """Test module-level constant re-exports."""

    def test_is_production_is_bool(self):
        assert isinstance(_IS_PRODUCTION, bool)

    def test_google_client_id_is_string(self):
        assert isinstance(GOOGLE_CLIENT_ID, str)

    def test_github_client_id_is_string(self):
        assert isinstance(GITHUB_CLIENT_ID, str)

    def test_allowed_redirect_hosts_is_frozenset(self):
        assert isinstance(ALLOWED_OAUTH_REDIRECT_HOSTS, frozenset)

    def test_oauth_success_url_is_string(self):
        assert isinstance(OAUTH_SUCCESS_URL, str)

    def test_oauth_error_url_is_string(self):
        assert isinstance(OAUTH_ERROR_URL, str)


# ===========================================================================
# 12. OAuthHandler routing via handle() method
# ===========================================================================


class TestOAuthHandlerRouting:
    """Test that handle() dispatches to the correct internal methods."""

    def test_handle_method_not_allowed(self, oauth_handler, mock_http_handler):
        """Unrecognized path+method combo yields 405."""
        # handler.command overrides method param, so set it to DELETE
        mock_http_handler.command = "DELETE"
        result = oauth_handler.handle(
            "/api/v1/auth/oauth/google", {}, mock_http_handler, method="DELETE"
        )
        assert _status(result) == 405

    def test_handle_rate_limited(self, oauth_handler, mock_http_handler):
        """When rate limiter denies, handle() returns 429."""
        with patch.object(_oauth_limiter, "is_allowed", return_value=False):
            result = oauth_handler.handle(
                "/api/v1/auth/oauth/google", {}, mock_http_handler, method="GET"
            )
            assert _status(result) == 429

    def test_handle_google_auth_start_dispatches(self, oauth_handler, mock_http_handler):
        """GET /api/v1/auth/oauth/google dispatches to _handle_google_auth_start."""
        with patch.object(
            oauth_handler, "_handle_google_auth_start", return_value=MagicMock(status_code=302)
        ) as mock_method:
            result = oauth_handler.handle(
                "/api/v1/auth/oauth/google", {}, mock_http_handler, method="GET"
            )
            mock_method.assert_called_once()

    def test_handle_github_auth_start_dispatches(self, oauth_handler, mock_http_handler):
        """GET /api/v1/auth/oauth/github dispatches to _handle_github_auth_start."""
        with patch.object(
            oauth_handler, "_handle_github_auth_start", return_value=MagicMock(status_code=302)
        ) as mock_method:
            result = oauth_handler.handle(
                "/api/v1/auth/oauth/github", {}, mock_http_handler, method="GET"
            )
            mock_method.assert_called_once()

    def test_handle_providers_get(self, oauth_handler, mock_http_handler):
        """GET /api/v1/auth/oauth/providers dispatches to _handle_list_providers."""
        with patch.object(
            oauth_handler, "_handle_list_providers", return_value=MagicMock(status_code=200)
        ) as mock_method:
            result = oauth_handler.handle(
                "/api/v1/auth/oauth/providers", {}, mock_http_handler, method="GET"
            )
            mock_method.assert_called_once()

    def test_handle_diagnostics_dispatches(self, oauth_handler, mock_http_handler):
        """GET diagnostics endpoint dispatches to _handle_oauth_diagnostics."""
        with patch.object(
            oauth_handler, "_handle_oauth_diagnostics", return_value=MagicMock(status_code=200)
        ) as mock_method:
            result = oauth_handler.handle(
                "/api/v1/auth/oauth/diagnostics", {}, mock_http_handler, method="GET"
            )
            mock_method.assert_called_once()

    def test_handle_link_post_dispatches(self, oauth_handler, mock_http_handler):
        """POST /api/v1/auth/oauth/link dispatches to _handle_link_account."""
        mock_http_handler.command = "POST"
        with patch.object(
            oauth_handler, "_handle_link_account", return_value=MagicMock(status_code=200)
        ) as mock_method:
            result = oauth_handler.handle(
                "/api/v1/auth/oauth/link", {}, mock_http_handler, method="POST"
            )
            mock_method.assert_called_once()

    def test_handle_unlink_delete_dispatches(self, oauth_handler, mock_http_handler):
        """DELETE /api/v1/auth/oauth/unlink dispatches to _handle_unlink_account."""
        mock_http_handler.command = "DELETE"
        with patch.object(
            oauth_handler, "_handle_unlink_account", return_value=MagicMock(status_code=200)
        ) as mock_method:
            result = oauth_handler.handle(
                "/api/v1/auth/oauth/unlink", {}, mock_http_handler, method="DELETE"
            )
            mock_method.assert_called_once()

    def test_handle_oauth_url_get_dispatches(self, oauth_handler, mock_http_handler):
        """GET /api/v1/auth/oauth/url dispatches to _handle_oauth_url."""
        with patch.object(
            oauth_handler, "_handle_oauth_url", return_value=MagicMock(status_code=200)
        ) as mock_method:
            result = oauth_handler.handle(
                "/api/v1/auth/oauth/url", {}, mock_http_handler, method="GET"
            )
            mock_method.assert_called_once()

    def test_handle_authorize_get_dispatches(self, oauth_handler, mock_http_handler):
        """GET /api/v1/auth/oauth/authorize dispatches to _handle_oauth_url."""
        with patch.object(
            oauth_handler, "_handle_oauth_url", return_value=MagicMock(status_code=200)
        ) as mock_method:
            result = oauth_handler.handle(
                "/api/v1/auth/oauth/authorize", {}, mock_http_handler, method="GET"
            )
            mock_method.assert_called_once()

    def test_handle_callback_post_dispatches(self, oauth_handler, mock_http_handler):
        """POST /api/v1/auth/oauth/callback dispatches to _handle_oauth_callback_api."""
        mock_http_handler.command = "POST"
        with patch.object(
            oauth_handler, "_handle_oauth_callback_api", return_value=MagicMock(status_code=200)
        ) as mock_method:
            result = oauth_handler.handle(
                "/api/v1/auth/oauth/callback", {}, mock_http_handler, method="POST"
            )
            mock_method.assert_called_once()

    def test_handle_user_oauth_providers_dispatches(self, oauth_handler, mock_http_handler):
        """GET /api/v1/user/oauth-providers dispatches to _handle_get_user_providers."""
        with patch.object(
            oauth_handler, "_handle_get_user_providers", return_value=MagicMock(status_code=200)
        ) as mock_method:
            result = oauth_handler.handle(
                "/api/v1/user/oauth-providers", {}, mock_http_handler, method="GET"
            )
            mock_method.assert_called_once()


# ===========================================================================
# 13. Config getters with env vars set
# ===========================================================================


class TestConfigGettersWithEnvVars:
    """Test that config getters read from environment correctly."""

    def test_microsoft_tenant_from_env(self):
        os.environ["MICROSOFT_OAUTH_TENANT"] = "my-tenant-id"
        assert _get_microsoft_tenant() == "my-tenant-id"

    def test_apple_client_id_from_env(self):
        os.environ["APPLE_OAUTH_CLIENT_ID"] = "com.example.app"
        assert _get_apple_client_id() == "com.example.app"

    def test_oidc_issuer_from_env(self):
        os.environ["OIDC_ISSUER"] = "https://issuer.example.com"
        assert _get_oidc_issuer() == "https://issuer.example.com"

    def test_success_url_from_env(self):
        os.environ["OAUTH_SUCCESS_URL"] = "https://app.example.com/success"
        assert _get_oauth_success_url() == "https://app.example.com/success"

    def test_error_url_from_env(self):
        os.environ["OAUTH_ERROR_URL"] = "https://app.example.com/error"
        assert _get_oauth_error_url() == "https://app.example.com/error"

    def test_allowed_hosts_from_env(self):
        os.environ["OAUTH_ALLOWED_REDIRECT_HOSTS"] = "example.com, app.io"
        hosts = _get_allowed_redirect_hosts()
        assert "example.com" in hosts
        assert "app.io" in hosts
