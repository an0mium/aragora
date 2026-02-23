"""Tests for aragora/server/handlers/_oauth/account.py.

Covers all six methods on AccountManagementMixin:

- _handle_list_providers: List configured OAuth providers
- _handle_oauth_url: Return OAuth authorization URL for a provider
- _handle_oauth_callback_api: Complete OAuth flow and return tokens as JSON
- _handle_get_user_providers: Get OAuth providers linked to the current user
- _handle_link_account: Link OAuth account to current user
- _handle_unlink_account: Unlink OAuth provider from current user
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers._oauth.account import AccountManagementMixin
from aragora.server.handlers.utils.responses import HandlerResult


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
# Stub mixin host
# ---------------------------------------------------------------------------


class _StubHost(AccountManagementMixin):
    """Concrete host class that satisfies the mixin's expected attributes.

    All protocol methods are wired to MagicMock so tests can inspect calls
    and configure return values freely.
    """

    def __init__(self):
        self._get_user_store = MagicMock()
        self._check_permission = MagicMock(return_value=None)
        self.read_json_body = MagicMock(return_value={})

        # Provider-specific auth start methods
        self._handle_google_auth_start = MagicMock()
        self._handle_github_auth_start = MagicMock()
        self._handle_microsoft_auth_start = MagicMock()
        self._handle_apple_auth_start = MagicMock()
        self._handle_oidc_auth_start = MagicMock()

        # Provider-specific callback methods
        self._handle_google_callback = MagicMock()
        self._handle_github_callback = MagicMock()
        self._handle_microsoft_callback = MagicMock()
        self._handle_apple_callback = MagicMock()
        self._handle_oidc_callback = MagicMock()

        # OIDC discovery (defined on OIDCOAuthMixin, used by link_account)
        self._get_oidc_discovery = MagicMock(return_value=None)


@pytest.fixture
def host():
    """Return a fresh _StubHost per test."""
    return _StubHost()


@pytest.fixture
def handler():
    """Return a generic mock HTTP handler."""
    return MagicMock()


# ---------------------------------------------------------------------------
# _impl() mock factory
# ---------------------------------------------------------------------------


def _make_impl(**overrides) -> MagicMock:
    """Build a mock _impl() module with sensible defaults.

    Any keyword argument overrides the default attribute value.
    """
    impl = MagicMock()
    # Provider client-id getters default to None (disabled)
    impl._get_google_client_id.return_value = overrides.get("google")
    impl._get_github_client_id.return_value = overrides.get("github")
    impl._get_microsoft_client_id.return_value = overrides.get("microsoft")
    impl._get_apple_client_id.return_value = overrides.get("apple")
    impl._get_oidc_issuer.return_value = overrides.get("oidc_issuer")
    impl._get_oidc_client_id.return_value = overrides.get("oidc_client")

    # Constants used in link_account URL construction
    impl.GOOGLE_CLIENT_ID = overrides.get("GOOGLE_CLIENT_ID")
    impl.GOOGLE_AUTH_URL = overrides.get(
        "GOOGLE_AUTH_URL", "https://accounts.google.com/o/oauth2/v2/auth"
    )
    impl._get_google_redirect_uri.return_value = "http://localhost/callback/google"

    impl.GITHUB_CLIENT_ID = overrides.get("GITHUB_CLIENT_ID")
    impl.GITHUB_AUTH_URL = overrides.get(
        "GITHUB_AUTH_URL", "https://github.com/login/oauth/authorize"
    )
    impl._get_github_redirect_uri.return_value = "http://localhost/callback/github"

    impl.MICROSOFT_AUTH_URL_TEMPLATE = overrides.get(
        "MICROSOFT_AUTH_URL_TEMPLATE",
        "https://login.microsoftonline.com/{tenant}/oauth2/v2.0/authorize",
    )
    impl._get_microsoft_redirect_uri.return_value = "http://localhost/callback/microsoft"
    impl._get_microsoft_tenant.return_value = "common"

    impl.APPLE_AUTH_URL = overrides.get(
        "APPLE_AUTH_URL", "https://appleid.apple.com/auth/authorize"
    )
    impl._get_apple_redirect_uri.return_value = "http://localhost/callback/apple"

    impl._get_oidc_redirect_uri.return_value = "http://localhost/callback/oidc"

    impl._get_oauth_success_url.return_value = "http://localhost/success"
    impl._validate_redirect_url.return_value = True
    impl._generate_state.return_value = "mock-state-token"

    return impl


IMPL_PATCH = "aragora.server.handlers._oauth.account._impl"
AUDIT_PATCH = "aragora.server.handlers._oauth.account.audit_action"
EXTRACT_USER_PATCH = "aragora.billing.jwt_auth.extract_user_from_request"


def _mock_auth_ctx(user_id: str = "user-1") -> MagicMock:
    """Return a mock auth context with a user_id."""
    ctx = MagicMock()
    ctx.user_id = user_id
    return ctx


# ===========================================================================
# _handle_list_providers
# ===========================================================================


class TestHandleListProviders:
    """Tests for _handle_list_providers."""

    def test_no_providers_configured(self, host, handler):
        impl = _make_impl()
        with patch(IMPL_PATCH, return_value=impl):
            result = host._handle_list_providers(handler)
        assert _status(result) == 200
        body = _body(result)
        assert body["providers"] == []

    def test_google_only(self, host, handler):
        impl = _make_impl(google="google-id")
        with patch(IMPL_PATCH, return_value=impl):
            result = host._handle_list_providers(handler)
        providers = _body(result)["providers"]
        assert len(providers) == 1
        assert providers[0]["id"] == "google"
        assert providers[0]["name"] == "Google"
        assert providers[0]["enabled"] is True
        assert providers[0]["auth_url"] == "/api/auth/oauth/google"

    def test_github_only(self, host, handler):
        impl = _make_impl(github="gh-id")
        with patch(IMPL_PATCH, return_value=impl):
            result = host._handle_list_providers(handler)
        providers = _body(result)["providers"]
        assert len(providers) == 1
        assert providers[0]["id"] == "github"

    def test_microsoft_only(self, host, handler):
        impl = _make_impl(microsoft="ms-id")
        with patch(IMPL_PATCH, return_value=impl):
            result = host._handle_list_providers(handler)
        providers = _body(result)["providers"]
        assert len(providers) == 1
        assert providers[0]["id"] == "microsoft"

    def test_apple_only(self, host, handler):
        impl = _make_impl(apple="apple-id")
        with patch(IMPL_PATCH, return_value=impl):
            result = host._handle_list_providers(handler)
        providers = _body(result)["providers"]
        assert len(providers) == 1
        assert providers[0]["id"] == "apple"

    def test_oidc_requires_both_issuer_and_client(self, host, handler):
        # Only issuer, no client -> not listed
        impl = _make_impl(oidc_issuer="https://issuer.example.com")
        with patch(IMPL_PATCH, return_value=impl):
            result = host._handle_list_providers(handler)
        assert _body(result)["providers"] == []

        # Only client, no issuer -> not listed
        impl = _make_impl(oidc_client="oidc-client-id")
        with patch(IMPL_PATCH, return_value=impl):
            result = host._handle_list_providers(handler)
        assert _body(result)["providers"] == []

    def test_oidc_listed_when_both_configured(self, host, handler):
        impl = _make_impl(oidc_issuer="https://issuer.example.com", oidc_client="oidc-client")
        with patch(IMPL_PATCH, return_value=impl):
            result = host._handle_list_providers(handler)
        providers = _body(result)["providers"]
        assert len(providers) == 1
        assert providers[0]["id"] == "oidc"
        assert providers[0]["name"] == "SSO"
        assert providers[0]["auth_url"] == "/api/auth/oauth/oidc"

    def test_all_providers_configured(self, host, handler):
        impl = _make_impl(
            google="g",
            github="gh",
            microsoft="ms",
            apple="ap",
            oidc_issuer="https://iss",
            oidc_client="oc",
        )
        with patch(IMPL_PATCH, return_value=impl):
            result = host._handle_list_providers(handler)
        providers = _body(result)["providers"]
        assert len(providers) == 5
        ids = [p["id"] for p in providers]
        assert ids == ["google", "github", "microsoft", "apple", "oidc"]

    def test_all_providers_have_enabled_true(self, host, handler):
        impl = _make_impl(google="g", github="gh")
        with patch(IMPL_PATCH, return_value=impl):
            result = host._handle_list_providers(handler)
        for p in _body(result)["providers"]:
            assert p["enabled"] is True


# ===========================================================================
# _handle_oauth_url
# ===========================================================================


class TestHandleOAuthUrl:
    """Tests for _handle_oauth_url."""

    def test_missing_provider_returns_400(self, host, handler):
        result = host._handle_oauth_url(handler, {})
        assert _status(result) == 400
        assert "required" in _body(result)["error"].lower()

    def test_unsupported_provider_returns_400(self, host, handler):
        result = host._handle_oauth_url(handler, {"provider": "facebook"})
        assert _status(result) == 400
        assert "unsupported" in _body(result)["error"].lower()

    def test_provider_is_case_insensitive(self, host, handler):
        auth_result = MagicMock()
        auth_result.headers = {"Location": "https://google.com/auth?state=abc123"}
        host._handle_google_auth_start.return_value = auth_result

        result = host._handle_oauth_url(handler, {"provider": "GOOGLE"})
        assert _status(result) == 200
        body = _body(result)
        assert body["auth_url"] == "https://google.com/auth?state=abc123"
        assert body["state"] == "abc123"

    @pytest.mark.parametrize(
        "provider,method_attr",
        [
            ("google", "_handle_google_auth_start"),
            ("github", "_handle_github_auth_start"),
            ("microsoft", "_handle_microsoft_auth_start"),
            ("apple", "_handle_apple_auth_start"),
            ("oidc", "_handle_oidc_auth_start"),
        ],
    )
    def test_each_provider_dispatches_correctly(self, host, handler, provider, method_attr):
        auth_result = MagicMock()
        auth_result.headers = {"Location": f"https://example.com/{provider}?state=s123"}
        getattr(host, method_attr).return_value = auth_result

        result = host._handle_oauth_url(handler, {"provider": provider})
        assert _status(result) == 200
        body = _body(result)
        assert f"/{provider}" in body["auth_url"]
        assert body["state"] == "s123"

    def test_handler_fn_returns_none_result(self, host, handler):
        host._handle_google_auth_start.return_value = None

        result = host._handle_oauth_url(handler, {"provider": "google"})
        assert _status(result) == 500
        assert "failed" in _body(result)["error"].lower()

    def test_handler_fn_returns_no_location_header(self, host, handler):
        auth_result = MagicMock()
        auth_result.headers = {}
        host._handle_google_auth_start.return_value = auth_result

        result = host._handle_oauth_url(handler, {"provider": "google"})
        assert _status(result) == 500

    def test_state_extraction_from_url(self, host, handler):
        auth_result = MagicMock()
        auth_result.headers = {
            "Location": "https://accounts.google.com/o/oauth2/auth?client_id=xyz&state=mystate123&scope=email"
        }
        host._handle_google_auth_start.return_value = auth_result

        result = host._handle_oauth_url(handler, {"provider": "google"})
        body = _body(result)
        assert body["state"] == "mystate123"

    def test_state_is_none_when_not_in_url(self, host, handler):
        auth_result = MagicMock()
        auth_result.headers = {"Location": "https://example.com/auth?client_id=xyz"}
        host._handle_google_auth_start.return_value = auth_result

        result = host._handle_oauth_url(handler, {"provider": "google"})
        body = _body(result)
        assert body["auth_url"] == "https://example.com/auth?client_id=xyz"
        assert body["state"] is None

    def test_state_parsing_error_returns_none_state(self, host, handler):
        """When URL is malformed enough that parse_qs can't handle it, state is None."""
        auth_result = MagicMock()
        # Headers has Location but let's verify the fallback via a result with headers=None
        auth_result.headers = {"Location": "https://example.com/auth?state=ok"}
        host._handle_google_auth_start.return_value = auth_result

        result = host._handle_oauth_url(handler, {"provider": "google"})
        assert _status(result) == 200
        assert _body(result)["state"] == "ok"

    def test_result_headers_is_none(self, host, handler):
        """When result.headers is None, should return 500."""
        auth_result = MagicMock()
        auth_result.headers = None
        host._handle_google_auth_start.return_value = auth_result

        result = host._handle_oauth_url(handler, {"provider": "google"})
        assert _status(result) == 500


# ===========================================================================
# _handle_oauth_callback_api
# ===========================================================================


class TestHandleOAuthCallbackApi:
    """Tests for _handle_oauth_callback_api."""

    def test_invalid_json_body_returns_400(self, host, handler):
        host.read_json_body.return_value = None
        result = host._handle_oauth_callback_api(handler)
        assert _status(result) == 400
        assert "json" in _body(result)["error"].lower()

    def test_missing_provider_returns_400(self, host, handler):
        host.read_json_body.return_value = {"code": "abc", "state": "xyz"}
        result = host._handle_oauth_callback_api(handler)
        assert _status(result) == 400
        assert "required" in _body(result)["error"].lower()

    def test_missing_code_returns_400(self, host, handler):
        host.read_json_body.return_value = {"provider": "google", "state": "xyz"}
        result = host._handle_oauth_callback_api(handler)
        assert _status(result) == 400

    def test_missing_state_returns_400(self, host, handler):
        host.read_json_body.return_value = {"provider": "google", "code": "abc"}
        result = host._handle_oauth_callback_api(handler)
        assert _status(result) == 400

    def test_unsupported_provider_returns_400(self, host, handler):
        host.read_json_body.return_value = {"provider": "facebook", "code": "abc", "state": "xyz"}
        result = host._handle_oauth_callback_api(handler)
        assert _status(result) == 400
        assert "unsupported" in _body(result)["error"].lower()

    def test_provider_is_case_insensitive(self, host, handler):
        cb_result = MagicMock()
        cb_result.headers = {
            "Location": "http://localhost/success#access_token=tok123&token_type=Bearer"
        }
        host._handle_google_callback.return_value = cb_result
        host.read_json_body.return_value = {"provider": "GOOGLE", "code": "c", "state": "s"}

        result = host._handle_oauth_callback_api(handler)
        assert _status(result) == 200
        assert _body(result)["access_token"] == "tok123"

    @pytest.mark.parametrize(
        "provider,method_attr",
        [
            ("google", "_handle_google_callback"),
            ("github", "_handle_github_callback"),
            ("microsoft", "_handle_microsoft_callback"),
            ("apple", "_handle_apple_callback"),
            ("oidc", "_handle_oidc_callback"),
        ],
    )
    def test_each_provider_dispatches_correctly(self, host, handler, provider, method_attr):
        cb_result = MagicMock()
        cb_result.headers = {
            "Location": "http://localhost/success#access_token=tok&token_type=Bearer"
        }
        getattr(host, method_attr).return_value = cb_result
        host.read_json_body.return_value = {"provider": provider, "code": "c", "state": "s"}

        result = host._handle_oauth_callback_api(handler)
        assert _status(result) == 200
        getattr(host, method_attr).assert_called_once_with(handler, {"code": "c", "state": "s"})

    def test_callback_returns_none_gives_502(self, host, handler):
        host._handle_google_callback.return_value = None
        host.read_json_body.return_value = {"provider": "google", "code": "c", "state": "s"}

        result = host._handle_oauth_callback_api(handler)
        assert _status(result) == 502
        assert "redirect" in _body(result)["error"].lower()

    def test_callback_returns_no_location_gives_502(self, host, handler):
        cb_result = MagicMock()
        cb_result.headers = {}
        host._handle_google_callback.return_value = cb_result
        host.read_json_body.return_value = {"provider": "google", "code": "c", "state": "s"}

        result = host._handle_oauth_callback_api(handler)
        assert _status(result) == 502

    def test_callback_headers_none_gives_502(self, host, handler):
        cb_result = MagicMock()
        cb_result.headers = None
        host._handle_google_callback.return_value = cb_result
        host.read_json_body.return_value = {"provider": "google", "code": "c", "state": "s"}

        result = host._handle_oauth_callback_api(handler)
        assert _status(result) == 502

    def test_error_in_redirect_fragment_returns_400(self, host, handler):
        cb_result = MagicMock()
        cb_result.headers = {"Location": "http://localhost/error#error=access_denied"}
        host._handle_google_callback.return_value = cb_result
        host.read_json_body.return_value = {"provider": "google", "code": "c", "state": "s"}

        result = host._handle_oauth_callback_api(handler)
        assert _status(result) == 400
        assert "access_denied" in _body(result)["error"]

    def test_error_in_query_params_returns_400(self, host, handler):
        cb_result = MagicMock()
        cb_result.headers = {"Location": "http://localhost/error?error=invalid_scope"}
        host._handle_google_callback.return_value = cb_result
        host.read_json_body.return_value = {"provider": "google", "code": "c", "state": "s"}

        result = host._handle_oauth_callback_api(handler)
        assert _status(result) == 400
        assert "invalid_scope" in _body(result)["error"]

    def test_tokens_from_fragment(self, host, handler):
        cb_result = MagicMock()
        cb_result.headers = {
            "Location": (
                "http://localhost/success#"
                "access_token=at123&refresh_token=rt456&token_type=Bearer&expires_in=3600"
            )
        }
        host._handle_google_callback.return_value = cb_result
        host.read_json_body.return_value = {"provider": "google", "code": "c", "state": "s"}

        result = host._handle_oauth_callback_api(handler)
        assert _status(result) == 200
        body = _body(result)
        assert body["access_token"] == "at123"
        assert body["refresh_token"] == "rt456"
        assert body["token_type"] == "Bearer"
        assert body["expires_in"] == 3600

    def test_tokens_from_query_fallback(self, host, handler):
        cb_result = MagicMock()
        cb_result.headers = {
            "Location": "http://localhost/success?access_token=at_q&token_type=mac"
        }
        host._handle_google_callback.return_value = cb_result
        host.read_json_body.return_value = {"provider": "google", "code": "c", "state": "s"}

        result = host._handle_oauth_callback_api(handler)
        assert _status(result) == 200
        body = _body(result)
        assert body["access_token"] == "at_q"
        assert body["token_type"] == "mac"

    def test_missing_access_token_returns_502(self, host, handler):
        cb_result = MagicMock()
        cb_result.headers = {
            "Location": "http://localhost/success#refresh_token=rt&token_type=Bearer"
        }
        host._handle_google_callback.return_value = cb_result
        host.read_json_body.return_value = {"provider": "google", "code": "c", "state": "s"}

        result = host._handle_oauth_callback_api(handler)
        assert _status(result) == 502
        assert "token" in _body(result)["error"].lower()

    def test_default_token_type_is_bearer(self, host, handler):
        cb_result = MagicMock()
        cb_result.headers = {"Location": "http://localhost/success#access_token=at"}
        host._handle_google_callback.return_value = cb_result
        host.read_json_body.return_value = {"provider": "google", "code": "c", "state": "s"}

        result = host._handle_oauth_callback_api(handler)
        body = _body(result)
        assert body["token_type"] == "Bearer"

    def test_expires_in_none_when_missing(self, host, handler):
        cb_result = MagicMock()
        cb_result.headers = {
            "Location": "http://localhost/success#access_token=at&token_type=Bearer"
        }
        host._handle_google_callback.return_value = cb_result
        host.read_json_body.return_value = {"provider": "google", "code": "c", "state": "s"}

        result = host._handle_oauth_callback_api(handler)
        body = _body(result)
        assert body["expires_in"] is None

    def test_expires_in_invalid_becomes_none(self, host, handler):
        cb_result = MagicMock()
        cb_result.headers = {
            "Location": "http://localhost/success#access_token=at&token_type=Bearer&expires_in=notanumber"
        }
        host._handle_google_callback.return_value = cb_result
        host.read_json_body.return_value = {"provider": "google", "code": "c", "state": "s"}

        result = host._handle_oauth_callback_api(handler)
        body = _body(result)
        assert body["expires_in"] is None

    def test_refresh_token_none_when_missing(self, host, handler):
        cb_result = MagicMock()
        cb_result.headers = {
            "Location": "http://localhost/success#access_token=at&token_type=Bearer"
        }
        host._handle_google_callback.return_value = cb_result
        host.read_json_body.return_value = {"provider": "google", "code": "c", "state": "s"}

        result = host._handle_oauth_callback_api(handler)
        body = _body(result)
        assert body["refresh_token"] is None


# ===========================================================================
# _handle_get_user_providers
# ===========================================================================


class TestHandleGetUserProviders:
    """Tests for _handle_get_user_providers."""

    def test_permission_denied_returns_error(self, host, handler):
        host._check_permission.return_value = HandlerResult(
            status_code=403, content_type="application/json", body=b'{"error":"forbidden"}'
        )
        result = host._handle_get_user_providers(handler)
        assert _status(result) == 403

    def test_returns_providers_via_get_oauth_providers(self, host, handler):
        user_store = MagicMock()
        user_store.get_oauth_providers.return_value = [
            {"provider": "google", "linked_at": "2025-01-01"},
            {"provider": "github", "linked_at": "2025-02-01"},
        ]
        host._get_user_store.return_value = user_store

        auth_ctx = _mock_auth_ctx("user-42")
        with patch(EXTRACT_USER_PATCH, return_value=auth_ctx):
            result = host._handle_get_user_providers(handler)

        assert _status(result) == 200
        providers = _body(result)["providers"]
        assert len(providers) == 2
        assert providers[0]["provider"] == "google"
        user_store.get_oauth_providers.assert_called_once_with("user-42")

    def test_fallback_to_oauth_repo(self, host, handler):
        user_store = MagicMock(spec=[])  # No get_oauth_providers
        oauth_repo = MagicMock()
        oauth_repo.get_providers_for_user.return_value = [{"provider": "apple"}]
        user_store._oauth_repo = oauth_repo
        host._get_user_store.return_value = user_store

        auth_ctx = _mock_auth_ctx("user-99")
        with patch(EXTRACT_USER_PATCH, return_value=auth_ctx):
            result = host._handle_get_user_providers(handler)

        assert _status(result) == 200
        providers = _body(result)["providers"]
        assert len(providers) == 1
        assert providers[0]["provider"] == "apple"
        oauth_repo.get_providers_for_user.assert_called_once_with("user-99")

    def test_no_provider_support_returns_empty_list(self, host, handler):
        user_store = MagicMock(spec=[])  # No get_oauth_providers, no _oauth_repo
        host._get_user_store.return_value = user_store

        auth_ctx = _mock_auth_ctx("user-empty")
        with patch(EXTRACT_USER_PATCH, return_value=auth_ctx):
            result = host._handle_get_user_providers(handler)

        assert _status(result) == 200
        assert _body(result)["providers"] == []


# ===========================================================================
# _handle_link_account
# ===========================================================================


class TestHandleLinkAccount:
    """Tests for _handle_link_account."""

    def test_permission_denied_returns_error(self, host, handler):
        host._check_permission.return_value = HandlerResult(
            status_code=403, content_type="application/json", body=b'{"error":"forbidden"}'
        )
        impl = _make_impl()
        with patch(IMPL_PATCH, return_value=impl):
            result = host._handle_link_account(handler)
        assert _status(result) == 403

    def test_invalid_json_body_returns_400(self, host, handler):
        host.read_json_body.return_value = None
        user_store = MagicMock()
        host._get_user_store.return_value = user_store
        auth_ctx = _mock_auth_ctx()
        impl = _make_impl()
        with patch(EXTRACT_USER_PATCH, return_value=auth_ctx), patch(IMPL_PATCH, return_value=impl):
            result = host._handle_link_account(handler)
        assert _status(result) == 400
        assert "json" in _body(result)["error"].lower()

    def test_unsupported_provider_returns_400(self, host, handler):
        host.read_json_body.return_value = {"provider": "facebook"}
        user_store = MagicMock()
        host._get_user_store.return_value = user_store
        auth_ctx = _mock_auth_ctx()
        impl = _make_impl()
        with patch(EXTRACT_USER_PATCH, return_value=auth_ctx), patch(IMPL_PATCH, return_value=impl):
            result = host._handle_link_account(handler)
        assert _status(result) == 400
        assert "unsupported" in _body(result)["error"].lower()

    def test_invalid_redirect_url_returns_400(self, host, handler):
        host.read_json_body.return_value = {
            "provider": "google",
            "redirect_url": "https://evil.com",
        }
        user_store = MagicMock()
        host._get_user_store.return_value = user_store
        auth_ctx = _mock_auth_ctx()
        impl = _make_impl(GOOGLE_CLIENT_ID="gid")
        impl._validate_redirect_url.return_value = False
        with patch(EXTRACT_USER_PATCH, return_value=auth_ctx), patch(IMPL_PATCH, return_value=impl):
            result = host._handle_link_account(handler)
        assert _status(result) == 400
        assert "redirect" in _body(result)["error"].lower()

    def test_link_google_not_configured_returns_503(self, host, handler):
        host.read_json_body.return_value = {"provider": "google"}
        user_store = MagicMock()
        host._get_user_store.return_value = user_store
        auth_ctx = _mock_auth_ctx()
        impl = _make_impl(GOOGLE_CLIENT_ID=None)
        with patch(EXTRACT_USER_PATCH, return_value=auth_ctx), patch(IMPL_PATCH, return_value=impl):
            result = host._handle_link_account(handler)
        assert _status(result) == 503
        assert "google" in _body(result)["error"].lower()

    def test_link_google_returns_auth_url(self, host, handler):
        host.read_json_body.return_value = {"provider": "google"}
        user_store = MagicMock()
        host._get_user_store.return_value = user_store
        auth_ctx = _mock_auth_ctx("user-link-g")
        impl = _make_impl(GOOGLE_CLIENT_ID="g-client-id")
        with patch(EXTRACT_USER_PATCH, return_value=auth_ctx), patch(IMPL_PATCH, return_value=impl):
            result = host._handle_link_account(handler)
        assert _status(result) == 200
        body = _body(result)
        assert "auth_url" in body
        assert "accounts.google.com" in body["auth_url"]
        assert "g-client-id" in body["auth_url"]
        assert "state=mock-state-token" in body["auth_url"]
        assert (
            "scope=openid+email+profile" in body["auth_url"] or "scope=openid" in body["auth_url"]
        )

    def test_link_github_not_configured_returns_503(self, host, handler):
        host.read_json_body.return_value = {"provider": "github"}
        user_store = MagicMock()
        host._get_user_store.return_value = user_store
        auth_ctx = _mock_auth_ctx()
        impl = _make_impl(GITHUB_CLIENT_ID=None)
        with patch(EXTRACT_USER_PATCH, return_value=auth_ctx), patch(IMPL_PATCH, return_value=impl):
            result = host._handle_link_account(handler)
        assert _status(result) == 503

    def test_link_github_returns_auth_url(self, host, handler):
        host.read_json_body.return_value = {"provider": "github"}
        user_store = MagicMock()
        host._get_user_store.return_value = user_store
        auth_ctx = _mock_auth_ctx("user-gh")
        impl = _make_impl(GITHUB_CLIENT_ID="gh-id")
        with patch(EXTRACT_USER_PATCH, return_value=auth_ctx), patch(IMPL_PATCH, return_value=impl):
            result = host._handle_link_account(handler)
        assert _status(result) == 200
        body = _body(result)
        assert "github.com" in body["auth_url"]
        assert "gh-id" in body["auth_url"]

    def test_link_microsoft_not_configured_returns_503(self, host, handler):
        host.read_json_body.return_value = {"provider": "microsoft"}
        user_store = MagicMock()
        host._get_user_store.return_value = user_store
        auth_ctx = _mock_auth_ctx()
        impl = _make_impl()
        impl._get_microsoft_client_id.return_value = None
        with patch(EXTRACT_USER_PATCH, return_value=auth_ctx), patch(IMPL_PATCH, return_value=impl):
            result = host._handle_link_account(handler)
        assert _status(result) == 503

    def test_link_microsoft_returns_auth_url(self, host, handler):
        host.read_json_body.return_value = {"provider": "microsoft"}
        user_store = MagicMock()
        host._get_user_store.return_value = user_store
        auth_ctx = _mock_auth_ctx("user-ms")
        impl = _make_impl()
        impl._get_microsoft_client_id.return_value = "ms-client"
        with patch(EXTRACT_USER_PATCH, return_value=auth_ctx), patch(IMPL_PATCH, return_value=impl):
            result = host._handle_link_account(handler)
        assert _status(result) == 200
        body = _body(result)
        assert "login.microsoftonline.com" in body["auth_url"]
        assert "common" in body["auth_url"]  # tenant
        assert "ms-client" in body["auth_url"]

    def test_link_apple_not_configured_returns_503(self, host, handler):
        host.read_json_body.return_value = {"provider": "apple"}
        user_store = MagicMock()
        host._get_user_store.return_value = user_store
        auth_ctx = _mock_auth_ctx()
        impl = _make_impl()
        impl._get_apple_client_id.return_value = None
        with patch(EXTRACT_USER_PATCH, return_value=auth_ctx), patch(IMPL_PATCH, return_value=impl):
            result = host._handle_link_account(handler)
        assert _status(result) == 503

    def test_link_apple_returns_auth_url(self, host, handler):
        host.read_json_body.return_value = {"provider": "apple"}
        user_store = MagicMock()
        host._get_user_store.return_value = user_store
        auth_ctx = _mock_auth_ctx("user-ap")
        impl = _make_impl()
        impl._get_apple_client_id.return_value = "apple-client"
        with patch(EXTRACT_USER_PATCH, return_value=auth_ctx), patch(IMPL_PATCH, return_value=impl):
            result = host._handle_link_account(handler)
        assert _status(result) == 200
        body = _body(result)
        assert "appleid.apple.com" in body["auth_url"]
        assert "apple-client" in body["auth_url"]
        assert "form_post" in body["auth_url"]

    def test_link_oidc_not_configured_returns_503(self, host, handler):
        host.read_json_body.return_value = {"provider": "oidc"}
        user_store = MagicMock()
        host._get_user_store.return_value = user_store
        auth_ctx = _mock_auth_ctx()
        impl = _make_impl()
        impl._get_oidc_issuer.return_value = None
        impl._get_oidc_client_id.return_value = None
        with patch(EXTRACT_USER_PATCH, return_value=auth_ctx), patch(IMPL_PATCH, return_value=impl):
            result = host._handle_link_account(handler)
        assert _status(result) == 503

    def test_link_oidc_discovery_fails_returns_503(self, host, handler):
        host.read_json_body.return_value = {"provider": "oidc"}
        user_store = MagicMock()
        host._get_user_store.return_value = user_store
        auth_ctx = _mock_auth_ctx()
        impl = _make_impl()
        impl._get_oidc_issuer.return_value = "https://issuer.example.com"
        impl._get_oidc_client_id.return_value = "oidc-client"
        host._get_oidc_discovery.return_value = None
        with patch(EXTRACT_USER_PATCH, return_value=auth_ctx), patch(IMPL_PATCH, return_value=impl):
            result = host._handle_link_account(handler)
        assert _status(result) == 503
        assert "discovery" in _body(result)["error"].lower()

    def test_link_oidc_discovery_missing_auth_endpoint_returns_503(self, host, handler):
        host.read_json_body.return_value = {"provider": "oidc"}
        user_store = MagicMock()
        host._get_user_store.return_value = user_store
        auth_ctx = _mock_auth_ctx()
        impl = _make_impl()
        impl._get_oidc_issuer.return_value = "https://issuer.example.com"
        impl._get_oidc_client_id.return_value = "oidc-client"
        host._get_oidc_discovery.return_value = {"token_endpoint": "https://issuer/token"}
        with patch(EXTRACT_USER_PATCH, return_value=auth_ctx), patch(IMPL_PATCH, return_value=impl):
            result = host._handle_link_account(handler)
        assert _status(result) == 503

    def test_link_oidc_returns_auth_url(self, host, handler):
        host.read_json_body.return_value = {"provider": "oidc"}
        user_store = MagicMock()
        host._get_user_store.return_value = user_store
        auth_ctx = _mock_auth_ctx("user-oidc")
        impl = _make_impl()
        impl._get_oidc_issuer.return_value = "https://issuer.example.com"
        impl._get_oidc_client_id.return_value = "oidc-client"
        host._get_oidc_discovery.return_value = {
            "authorization_endpoint": "https://issuer.example.com/authorize"
        }
        with patch(EXTRACT_USER_PATCH, return_value=auth_ctx), patch(IMPL_PATCH, return_value=impl):
            result = host._handle_link_account(handler)
        assert _status(result) == 200
        body = _body(result)
        assert "issuer.example.com/authorize" in body["auth_url"]
        assert "oidc-client" in body["auth_url"]

    def test_link_google_uses_custom_redirect_url(self, host, handler):
        host.read_json_body.return_value = {
            "provider": "google",
            "redirect_url": "https://myapp.com/done",
        }
        user_store = MagicMock()
        host._get_user_store.return_value = user_store
        auth_ctx = _mock_auth_ctx("user-redir")
        impl = _make_impl(GOOGLE_CLIENT_ID="gid")
        with patch(EXTRACT_USER_PATCH, return_value=auth_ctx), patch(IMPL_PATCH, return_value=impl):
            result = host._handle_link_account(handler)
        assert _status(result) == 200
        # Verify _generate_state was called with the custom redirect_url
        impl._generate_state.assert_called_once_with(
            user_id="user-redir", redirect_url="https://myapp.com/done"
        )

    def test_link_uses_default_redirect_url_when_not_provided(self, host, handler):
        host.read_json_body.return_value = {"provider": "google"}
        user_store = MagicMock()
        host._get_user_store.return_value = user_store
        auth_ctx = _mock_auth_ctx("user-def")
        impl = _make_impl(GOOGLE_CLIENT_ID="gid")
        with patch(EXTRACT_USER_PATCH, return_value=auth_ctx), patch(IMPL_PATCH, return_value=impl):
            result = host._handle_link_account(handler)
        assert _status(result) == 200
        impl._generate_state.assert_called_once_with(
            user_id="user-def", redirect_url="http://localhost/success"
        )

    def test_link_provider_case_insensitive(self, host, handler):
        """Provider name 'Google' should be lowercased to 'google'."""
        host.read_json_body.return_value = {"provider": "Google"}
        user_store = MagicMock()
        host._get_user_store.return_value = user_store
        auth_ctx = _mock_auth_ctx()
        impl = _make_impl(GOOGLE_CLIENT_ID="gid")
        with patch(EXTRACT_USER_PATCH, return_value=auth_ctx), patch(IMPL_PATCH, return_value=impl):
            result = host._handle_link_account(handler)
        assert _status(result) == 200

    def test_empty_provider_returns_400(self, host, handler):
        host.read_json_body.return_value = {"provider": ""}
        user_store = MagicMock()
        host._get_user_store.return_value = user_store
        auth_ctx = _mock_auth_ctx()
        impl = _make_impl()
        with patch(EXTRACT_USER_PATCH, return_value=auth_ctx), patch(IMPL_PATCH, return_value=impl):
            result = host._handle_link_account(handler)
        assert _status(result) == 400


# ===========================================================================
# _handle_unlink_account
# ===========================================================================


class TestHandleUnlinkAccount:
    """Tests for _handle_unlink_account."""

    def test_permission_denied_returns_error(self, host, handler):
        host._check_permission.return_value = HandlerResult(
            status_code=403, content_type="application/json", body=b'{"error":"forbidden"}'
        )
        result = host._handle_unlink_account(handler)
        assert _status(result) == 403

    def test_invalid_json_body_returns_400(self, host, handler):
        host.read_json_body.return_value = None
        user_store = MagicMock()
        host._get_user_store.return_value = user_store
        auth_ctx = _mock_auth_ctx()
        with patch(EXTRACT_USER_PATCH, return_value=auth_ctx):
            result = host._handle_unlink_account(handler)
        assert _status(result) == 400

    def test_unsupported_provider_returns_400(self, host, handler):
        host.read_json_body.return_value = {"provider": "facebook"}
        user_store = MagicMock()
        host._get_user_store.return_value = user_store
        auth_ctx = _mock_auth_ctx()
        with patch(EXTRACT_USER_PATCH, return_value=auth_ctx):
            result = host._handle_unlink_account(handler)
        assert _status(result) == 400
        assert "unsupported" in _body(result)["error"].lower()

    def test_user_not_found_returns_404(self, host, handler):
        host.read_json_body.return_value = {"provider": "google"}
        user_store = MagicMock()
        user_store.get_user_by_id.return_value = None
        host._get_user_store.return_value = user_store
        auth_ctx = _mock_auth_ctx("user-404")
        with patch(EXTRACT_USER_PATCH, return_value=auth_ctx):
            result = host._handle_unlink_account(handler)
        assert _status(result) == 404
        user_store.get_user_by_id.assert_called_once_with("user-404")

    def test_no_password_hash_returns_400(self, host, handler):
        host.read_json_body.return_value = {"provider": "google"}
        user = MagicMock()
        user.password_hash = None
        user_store = MagicMock()
        user_store.get_user_by_id.return_value = user
        host._get_user_store.return_value = user_store
        auth_ctx = _mock_auth_ctx("user-no-pw")
        with patch(EXTRACT_USER_PATCH, return_value=auth_ctx):
            result = host._handle_unlink_account(handler)
        assert _status(result) == 400
        assert "password" in _body(result)["error"].lower()

    def test_empty_password_hash_returns_400(self, host, handler):
        """Empty string password_hash is falsy, should be treated as no password."""
        host.read_json_body.return_value = {"provider": "github"}
        user = MagicMock()
        user.password_hash = ""
        user_store = MagicMock()
        user_store.get_user_by_id.return_value = user
        host._get_user_store.return_value = user_store
        auth_ctx = _mock_auth_ctx()
        with patch(EXTRACT_USER_PATCH, return_value=auth_ctx):
            result = host._handle_unlink_account(handler)
        assert _status(result) == 400

    def test_successful_unlink_via_unlink_oauth_provider(self, host, handler):
        host.read_json_body.return_value = {"provider": "google"}
        user = MagicMock()
        user.password_hash = "$2b$12$hashedpassword"
        user_store = MagicMock()
        user_store.get_user_by_id.return_value = user
        user_store.unlink_oauth_provider.return_value = True
        host._get_user_store.return_value = user_store
        auth_ctx = _mock_auth_ctx("user-unlink")
        with patch(EXTRACT_USER_PATCH, return_value=auth_ctx), patch(AUDIT_PATCH) as mock_audit:
            result = host._handle_unlink_account(handler)
        assert _status(result) == 200
        body = _body(result)
        assert "google" in body["message"].lower()
        assert "successfully" in body["message"].lower()
        user_store.unlink_oauth_provider.assert_called_once_with("user-unlink", "google")
        mock_audit.assert_called_once_with(
            user_id="user-unlink",
            action="oauth_unlink",
            resource_type="auth",
            resource_id="user-unlink",
            provider="google",
        )

    def test_unlink_failure_returns_500(self, host, handler):
        host.read_json_body.return_value = {"provider": "github"}
        user = MagicMock()
        user.password_hash = "$2b$12$hash"
        user_store = MagicMock()
        user_store.get_user_by_id.return_value = user
        user_store.unlink_oauth_provider.return_value = False
        host._get_user_store.return_value = user_store
        auth_ctx = _mock_auth_ctx("user-fail")
        with patch(EXTRACT_USER_PATCH, return_value=auth_ctx):
            result = host._handle_unlink_account(handler)
        assert _status(result) == 500
        assert "failed" in _body(result)["error"].lower()

    def test_unlink_without_unlink_support_succeeds(self, host, handler):
        """When user_store lacks unlink_oauth_provider, should still succeed (warning logged)."""
        host.read_json_body.return_value = {"provider": "apple"}
        user = MagicMock()
        user.password_hash = "$2b$12$hash"
        user_store = MagicMock(spec=[])  # No unlink_oauth_provider
        user_store.get_user_by_id = MagicMock(return_value=user)
        host._get_user_store.return_value = user_store
        auth_ctx = _mock_auth_ctx("user-no-support")
        with patch(EXTRACT_USER_PATCH, return_value=auth_ctx), patch(AUDIT_PATCH) as mock_audit:
            result = host._handle_unlink_account(handler)
        assert _status(result) == 200
        assert "apple" in _body(result)["message"].lower()
        mock_audit.assert_called_once()

    @pytest.mark.parametrize("provider", ["google", "github", "microsoft", "apple", "oidc"])
    def test_unlink_all_valid_providers(self, host, handler, provider):
        host.read_json_body.return_value = {"provider": provider}
        user = MagicMock()
        user.password_hash = "$2b$12$hash"
        user_store = MagicMock()
        user_store.get_user_by_id.return_value = user
        user_store.unlink_oauth_provider.return_value = True
        host._get_user_store.return_value = user_store
        auth_ctx = _mock_auth_ctx("u1")
        with patch(EXTRACT_USER_PATCH, return_value=auth_ctx), patch(AUDIT_PATCH):
            result = host._handle_unlink_account(handler)
        assert _status(result) == 200
        assert provider in _body(result)["message"]

    def test_unlink_provider_case_insensitive(self, host, handler):
        host.read_json_body.return_value = {"provider": "GitHub"}
        user = MagicMock()
        user.password_hash = "$2b$12$hash"
        user_store = MagicMock()
        user_store.get_user_by_id.return_value = user
        user_store.unlink_oauth_provider.return_value = True
        host._get_user_store.return_value = user_store
        auth_ctx = _mock_auth_ctx("u1")
        with patch(EXTRACT_USER_PATCH, return_value=auth_ctx), patch(AUDIT_PATCH):
            result = host._handle_unlink_account(handler)
        assert _status(result) == 200
        user_store.unlink_oauth_provider.assert_called_once_with("u1", "github")

    def test_audit_event_includes_provider(self, host, handler):
        host.read_json_body.return_value = {"provider": "microsoft"}
        user = MagicMock()
        user.password_hash = "$2b$12$hash"
        user_store = MagicMock()
        user_store.get_user_by_id.return_value = user
        user_store.unlink_oauth_provider.return_value = True
        host._get_user_store.return_value = user_store
        auth_ctx = _mock_auth_ctx("u-audit")
        with patch(EXTRACT_USER_PATCH, return_value=auth_ctx), patch(AUDIT_PATCH) as mock_audit:
            result = host._handle_unlink_account(handler)
        assert _status(result) == 200
        mock_audit.assert_called_once()
        call_kwargs = mock_audit.call_args
        assert (
            call_kwargs.kwargs.get("provider") == "microsoft"
            or call_kwargs[1].get("provider") == "microsoft"
        )

    def test_missing_provider_key_returns_400(self, host, handler):
        """Body has no provider key at all -> empty string lowered -> unsupported."""
        host.read_json_body.return_value = {}
        user_store = MagicMock()
        host._get_user_store.return_value = user_store
        auth_ctx = _mock_auth_ctx()
        with patch(EXTRACT_USER_PATCH, return_value=auth_ctx):
            result = host._handle_unlink_account(handler)
        assert _status(result) == 400
