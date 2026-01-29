"""
Comprehensive tests for the OAuth handler at aragora/server/handlers/_oauth_impl.py.

Covers: token exchange, CSRF state, PKCE, redirect validation, rate limiting,
error paths, provider listing, account linking/unlinking, and more.

All imports are inside test functions/classes to avoid conftest issues.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Pre-stub the Slack handler modules to avoid circular ImportError when
# importing aragora.server.handlers (its __init__.py tries to import Slack).
# This must happen before any aragora.server.handlers imports.
# ---------------------------------------------------------------------------
import sys
import types as _types_mod

_SLACK_ATTRS = [
    "SlackHandler",
    "get_slack_handler",
    "get_slack_integration",
    "get_workspace_store",
    "resolve_workspace",
    "create_tracked_task",
    "_validate_slack_url",
    "SLACK_SIGNING_SECRET",
    "SLACK_BOT_TOKEN",
    "SLACK_WEBHOOK_URL",
    "SLACK_ALLOWED_DOMAINS",
    "SignatureVerifierMixin",
    "CommandsMixin",
    "EventsMixin",
    "init_slack_handler",
]

for _mod_name in (
    "aragora.server.handlers.social.slack.handler",
    "aragora.server.handlers.social.slack",
):
    if _mod_name not in sys.modules:
        _m = _types_mod.ModuleType(_mod_name)
        for _a in _SLACK_ATTRS:
            setattr(_m, _a, None)
        sys.modules[_mod_name] = _m

# ---------------------------------------------------------------------------

import json
import io
import os
import time
import types
import unittest.mock as mock
from dataclasses import dataclass
from typing import Any, Optional
from urllib.parse import urlencode, urlparse, parse_qs

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_rate_limiter():
    """Clear rate limiter state before each test."""
    from aragora.server.handlers._oauth_impl import _oauth_limiter

    # Clear the internal window tracking
    for attr_name in dir(_oauth_limiter):
        obj = getattr(_oauth_limiter, attr_name, None)
        if isinstance(obj, dict):
            try:
                obj.clear()
            except Exception:
                pass
    yield


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    """Ensure no production env leaks."""
    for var in [
        "ARAGORA_ENV",
        "GOOGLE_OAUTH_CLIENT_ID",
        "GOOGLE_OAUTH_CLIENT_SECRET",
        "GITHUB_OAUTH_CLIENT_ID",
        "GITHUB_OAUTH_CLIENT_SECRET",
        "MICROSOFT_OAUTH_CLIENT_ID",
        "APPLE_OAUTH_CLIENT_ID",
        "OIDC_ISSUER",
        "OIDC_CLIENT_ID",
    ]:
        monkeypatch.delenv(var, raising=False)
    yield


@pytest.fixture()
def _patch_google_secrets():
    """Provide a fully configured Google OAuth env."""
    with (
        mock.patch(
            "aragora.server.handlers._oauth_impl._get_google_client_id",
            return_value="goog-client-id",
        ),
        mock.patch(
            "aragora.server.handlers._oauth_impl._get_google_client_secret",
            return_value="goog-secret",
        ),
        mock.patch(
            "aragora.server.handlers._oauth_impl._get_google_redirect_uri",
            return_value="http://localhost:8080/api/auth/oauth/google/callback",
        ),
        mock.patch(
            "aragora.server.handlers._oauth_impl._get_oauth_success_url",
            return_value="http://localhost:3000/auth/callback",
        ),
        mock.patch(
            "aragora.server.handlers._oauth_impl._get_oauth_error_url",
            return_value="http://localhost:3000/auth/error",
        ),
        mock.patch(
            "aragora.server.handlers._oauth_impl._get_allowed_redirect_hosts",
            return_value=frozenset({"localhost", "127.0.0.1"}),
        ),
    ):
        yield


@pytest.fixture()
def _patch_github_secrets():
    """Provide a fully configured GitHub OAuth env."""
    with (
        mock.patch(
            "aragora.server.handlers._oauth_impl._get_github_client_id", return_value="gh-client-id"
        ),
        mock.patch(
            "aragora.server.handlers._oauth_impl._get_github_client_secret",
            return_value="gh-secret",
        ),
        mock.patch(
            "aragora.server.handlers._oauth_impl._get_github_redirect_uri",
            return_value="http://localhost:8080/api/auth/oauth/github/callback",
        ),
        mock.patch(
            "aragora.server.handlers._oauth_impl._get_oauth_success_url",
            return_value="http://localhost:3000/auth/callback",
        ),
        mock.patch(
            "aragora.server.handlers._oauth_impl._get_oauth_error_url",
            return_value="http://localhost:3000/auth/error",
        ),
        mock.patch(
            "aragora.server.handlers._oauth_impl._get_allowed_redirect_hosts",
            return_value=frozenset({"localhost", "127.0.0.1"}),
        ),
    ):
        yield


@pytest.fixture()
def _patch_microsoft_secrets():
    """Provide a fully configured Microsoft OAuth env."""
    with (
        mock.patch(
            "aragora.server.handlers._oauth_impl._get_microsoft_client_id",
            return_value="ms-client-id",
        ),
        mock.patch(
            "aragora.server.handlers._oauth_impl._get_microsoft_client_secret",
            return_value="ms-secret",
        ),
        mock.patch(
            "aragora.server.handlers._oauth_impl._get_microsoft_tenant", return_value="common"
        ),
        mock.patch(
            "aragora.server.handlers._oauth_impl._get_microsoft_redirect_uri",
            return_value="http://localhost:8080/api/auth/oauth/microsoft/callback",
        ),
        mock.patch(
            "aragora.server.handlers._oauth_impl._get_oauth_success_url",
            return_value="http://localhost:3000/auth/callback",
        ),
        mock.patch(
            "aragora.server.handlers._oauth_impl._get_oauth_error_url",
            return_value="http://localhost:3000/auth/error",
        ),
        mock.patch(
            "aragora.server.handlers._oauth_impl._get_allowed_redirect_hosts",
            return_value=frozenset({"localhost", "127.0.0.1"}),
        ),
    ):
        yield


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_handler_obj(
    client_ip: str = "127.0.0.1",
    headers: dict | None = None,
    body: bytes = b"",
    command: str = "GET",
):
    """Create a mock HTTP handler object."""
    h = mock.MagicMock()
    h.client_address = (client_ip, 12345)
    h.command = command
    hdrs = headers or {}
    # Use a MagicMock for headers to allow .get override (plain dict's .get is read-only)
    hdr_mock = mock.MagicMock()
    hdr_mock.get = mock.MagicMock(side_effect=lambda k, d=None: hdrs.get(k, d))
    hdr_mock.__getitem__ = mock.MagicMock(side_effect=lambda k: hdrs[k])
    hdr_mock.__contains__ = mock.MagicMock(side_effect=lambda k: k in hdrs)
    hdr_mock.__iter__ = mock.MagicMock(side_effect=lambda: iter(hdrs))
    h.headers = hdr_mock
    h.rfile = io.BytesIO(body)
    return h


def _make_oauth_handler(ctx: dict | None = None):
    """Instantiate OAuthHandler with a minimal server context."""
    from aragora.server.handlers._oauth_impl import OAuthHandler

    return OAuthHandler(ctx or {})


def _make_auth_ctx(
    is_authenticated=False, user_id=None, role="member", org_id=None, client_ip="127.0.0.1"
):
    """Create a minimal auth context object."""
    ac = mock.MagicMock()
    ac.is_authenticated = is_authenticated
    ac.user_id = user_id
    ac.role = role
    ac.org_id = org_id
    ac.client_ip = client_ip
    return ac


def _make_user(
    user_id="user-1",
    email="test@example.com",
    org_id="org-1",
    role="member",
    password_hash="hash123",
):
    """Create a minimal user object."""
    u = mock.MagicMock()
    u.id = user_id
    u.email = email
    u.org_id = org_id
    u.role = role
    u.password_hash = password_hash
    return u


def _make_token_pair():
    """Create a mock token pair."""
    tp = mock.MagicMock()
    tp.access_token = "test-access-token"
    tp.refresh_token = "test-refresh-token"
    tp.expires_in = 3600
    return tp


# ===========================================================================
# Tests: _get_param
# ===========================================================================


class TestGetParam:
    def test_get_param_string_value(self):
        from aragora.server.handlers._oauth_impl import _get_param

        assert _get_param({"key": "value"}, "key") == "value"

    def test_get_param_list_value(self):
        from aragora.server.handlers._oauth_impl import _get_param

        assert _get_param({"key": ["first", "second"]}, "key") == "first"

    def test_get_param_empty_list(self):
        from aragora.server.handlers._oauth_impl import _get_param

        assert _get_param({"key": []}, "key", "default") == "default"

    def test_get_param_missing_key(self):
        from aragora.server.handlers._oauth_impl import _get_param

        assert _get_param({}, "key", "fallback") == "fallback"

    def test_get_param_none_default(self):
        from aragora.server.handlers._oauth_impl import _get_param

        assert _get_param({}, "key") is None

    def test_get_param_integer_value(self):
        from aragora.server.handlers._oauth_impl import _get_param

        assert _get_param({"n": 42}, "n") == 42

    def test_get_param_none_value_in_dict(self):
        from aragora.server.handlers._oauth_impl import _get_param

        assert _get_param({"key": None}, "key") is None


# ===========================================================================
# Tests: _validate_redirect_url
# ===========================================================================


class TestValidateRedirectUrl:
    def test_valid_localhost_http(self):
        from aragora.server.handlers._oauth_impl import _validate_redirect_url

        with mock.patch(
            "aragora.server.handlers._oauth_impl._get_allowed_redirect_hosts",
            return_value=frozenset({"localhost"}),
        ):
            assert _validate_redirect_url("http://localhost:3000/callback") is True

    def test_valid_localhost_https(self):
        from aragora.server.handlers._oauth_impl import _validate_redirect_url

        with mock.patch(
            "aragora.server.handlers._oauth_impl._get_allowed_redirect_hosts",
            return_value=frozenset({"localhost"}),
        ):
            assert _validate_redirect_url("https://localhost/callback") is True

    def test_blocked_external_host(self):
        from aragora.server.handlers._oauth_impl import _validate_redirect_url

        with mock.patch(
            "aragora.server.handlers._oauth_impl._get_allowed_redirect_hosts",
            return_value=frozenset({"localhost"}),
        ):
            assert _validate_redirect_url("https://evil.com/steal") is False

    def test_javascript_scheme_blocked(self):
        from aragora.server.handlers._oauth_impl import _validate_redirect_url

        with mock.patch(
            "aragora.server.handlers._oauth_impl._get_allowed_redirect_hosts",
            return_value=frozenset({"localhost"}),
        ):
            assert _validate_redirect_url("javascript:alert(1)") is False

    def test_data_scheme_blocked(self):
        from aragora.server.handlers._oauth_impl import _validate_redirect_url

        with mock.patch(
            "aragora.server.handlers._oauth_impl._get_allowed_redirect_hosts",
            return_value=frozenset({"localhost"}),
        ):
            assert _validate_redirect_url("data:text/html,<h1>bad</h1>") is False

    def test_ftp_scheme_blocked(self):
        from aragora.server.handlers._oauth_impl import _validate_redirect_url

        with mock.patch(
            "aragora.server.handlers._oauth_impl._get_allowed_redirect_hosts",
            return_value=frozenset({"localhost"}),
        ):
            assert _validate_redirect_url("ftp://localhost/file") is False

    def test_empty_host_blocked(self):
        from aragora.server.handlers._oauth_impl import _validate_redirect_url

        with mock.patch(
            "aragora.server.handlers._oauth_impl._get_allowed_redirect_hosts",
            return_value=frozenset({"localhost"}),
        ):
            assert _validate_redirect_url("http://") is False

    def test_subdomain_allowed(self):
        from aragora.server.handlers._oauth_impl import _validate_redirect_url

        with mock.patch(
            "aragora.server.handlers._oauth_impl._get_allowed_redirect_hosts",
            return_value=frozenset({"example.com"}),
        ):
            assert _validate_redirect_url("https://app.example.com/callback") is True

    def test_empty_allowlist_blocks_all(self):
        from aragora.server.handlers._oauth_impl import _validate_redirect_url

        with mock.patch(
            "aragora.server.handlers._oauth_impl._get_allowed_redirect_hosts",
            return_value=frozenset(),
        ):
            assert _validate_redirect_url("https://example.com") is False

    def test_invalid_url_returns_false(self):
        from aragora.server.handlers._oauth_impl import _validate_redirect_url

        with mock.patch(
            "aragora.server.handlers._oauth_impl._get_allowed_redirect_hosts",
            return_value=frozenset({"localhost"}),
        ):
            assert _validate_redirect_url("") is False

    def test_case_insensitive_host(self):
        from aragora.server.handlers._oauth_impl import _validate_redirect_url

        with mock.patch(
            "aragora.server.handlers._oauth_impl._get_allowed_redirect_hosts",
            return_value=frozenset({"example.com"}),
        ):
            assert _validate_redirect_url("https://EXAMPLE.COM/callback") is True


# ===========================================================================
# Tests: OAuthHandler.can_handle
# ===========================================================================


class TestCanHandle:
    def test_google_route(self):
        h = _make_oauth_handler()
        assert h.can_handle("/api/v1/auth/oauth/google") is True

    def test_github_callback_route(self):
        h = _make_oauth_handler()
        assert h.can_handle("/api/auth/oauth/github/callback") is True

    def test_unknown_route(self):
        h = _make_oauth_handler()
        assert h.can_handle("/api/random/path") is False

    def test_providers_route(self):
        h = _make_oauth_handler()
        assert h.can_handle("/api/v1/auth/oauth/providers") is True

    def test_link_route(self):
        h = _make_oauth_handler()
        assert h.can_handle("/api/auth/oauth/link") is True

    def test_unlink_route(self):
        h = _make_oauth_handler()
        assert h.can_handle("/api/v1/auth/oauth/unlink") is True

    def test_microsoft_route(self):
        h = _make_oauth_handler()
        assert h.can_handle("/api/v1/auth/oauth/microsoft") is True

    def test_apple_route(self):
        h = _make_oauth_handler()
        assert h.can_handle("/api/auth/oauth/apple/callback") is True

    def test_oidc_route(self):
        h = _make_oauth_handler()
        assert h.can_handle("/api/v1/auth/oauth/oidc") is True

    def test_user_providers_route(self):
        h = _make_oauth_handler()
        assert h.can_handle("/api/user/oauth-providers") is True


# ===========================================================================
# Tests: Rate limiting
# ===========================================================================


class TestRateLimiting:
    def test_rate_limit_exceeded_returns_429(self):
        from aragora.server.handlers._oauth_impl import _oauth_limiter

        oh = _make_oauth_handler()
        handler = _make_handler_obj(client_ip="10.0.0.1")

        with mock.patch.object(_oauth_limiter, "is_allowed", return_value=False):
            result = oh.handle("/api/auth/oauth/google", {}, handler, method="GET")
            assert result is not None
            assert result.status_code == 429

    def test_rate_limit_allowed_proceeds(self, _patch_google_secrets):
        from aragora.server.handlers._oauth_impl import _oauth_limiter

        oh = _make_oauth_handler()
        handler = _make_handler_obj()

        with (
            mock.patch(
                "aragora.billing.jwt_auth.extract_user_from_request", return_value=_make_auth_ctx()
            ),
            mock.patch.object(_oauth_limiter, "is_allowed", return_value=True),
            mock.patch(
                "aragora.server.handlers._oauth_impl._generate_state", return_value="state123"
            ),
        ):
            result = oh.handle("/api/auth/oauth/google", {}, handler, method="GET")
            assert result is not None
            assert result.status_code == 302


# ===========================================================================
# Tests: Method not allowed
# ===========================================================================


class TestMethodNotAllowed:
    def test_post_to_google_start_returns_405(self):
        from aragora.server.handlers._oauth_impl import _oauth_limiter

        oh = _make_oauth_handler()
        handler = _make_handler_obj(command="POST")
        with mock.patch.object(_oauth_limiter, "is_allowed", return_value=True):
            result = oh.handle("/api/auth/oauth/google", {}, handler, method="POST")
            assert result is not None
            assert result.status_code == 405

    def test_delete_to_google_callback_returns_405(self):
        from aragora.server.handlers._oauth_impl import _oauth_limiter

        oh = _make_oauth_handler()
        handler = _make_handler_obj(command="DELETE")
        with mock.patch.object(_oauth_limiter, "is_allowed", return_value=True):
            result = oh.handle("/api/auth/oauth/google/callback", {}, handler, method="DELETE")
            assert result is not None
            assert result.status_code == 405


# ===========================================================================
# Tests: Google OAuth start
# ===========================================================================


class TestGoogleAuthStart:
    def test_not_configured_returns_503(self):
        oh = _make_oauth_handler()
        handler = _make_handler_obj()
        with (
            mock.patch(
                "aragora.server.handlers._oauth_impl._get_google_client_id", return_value=""
            ),
            mock.patch("aragora.server.handlers._oauth_impl._oauth_limiter") as lim,
        ):
            lim.is_allowed.return_value = True
            result = oh.handle("/api/auth/oauth/google", {}, handler, method="GET")
            assert result is not None
            assert result.status_code == 503

    def test_invalid_redirect_url_returns_400(self, _patch_google_secrets):
        oh = _make_oauth_handler()
        handler = _make_handler_obj()
        with (
            mock.patch("aragora.server.handlers._oauth_impl._oauth_limiter") as lim,
            mock.patch(
                "aragora.server.handlers._oauth_impl._validate_redirect_url", return_value=False
            ),
            mock.patch(
                "aragora.billing.jwt_auth.extract_user_from_request", return_value=_make_auth_ctx()
            ),
        ):
            lim.is_allowed.return_value = True
            result = oh.handle(
                "/api/auth/oauth/google", {"redirect_url": "http://evil.com"}, handler, method="GET"
            )
            assert result is not None
            assert result.status_code == 400

    def test_successful_redirect(self, _patch_google_secrets):
        oh = _make_oauth_handler()
        handler = _make_handler_obj()
        with (
            mock.patch("aragora.server.handlers._oauth_impl._oauth_limiter") as lim,
            mock.patch(
                "aragora.billing.jwt_auth.extract_user_from_request", return_value=_make_auth_ctx()
            ),
            mock.patch(
                "aragora.server.handlers._oauth_impl._generate_state", return_value="state-tok"
            ),
        ):
            lim.is_allowed.return_value = True
            result = oh.handle("/api/auth/oauth/google", {}, handler, method="GET")
            assert result is not None
            assert result.status_code == 302
            assert "Location" in result.headers
            loc = result.headers["Location"]
            assert "accounts.google.com" in loc
            assert "state=state-tok" in loc
            assert "client_id=goog-client-id" in loc

    def test_v1_route_also_works(self, _patch_google_secrets):
        oh = _make_oauth_handler()
        handler = _make_handler_obj()
        with (
            mock.patch("aragora.server.handlers._oauth_impl._oauth_limiter") as lim,
            mock.patch(
                "aragora.billing.jwt_auth.extract_user_from_request", return_value=_make_auth_ctx()
            ),
            mock.patch("aragora.server.handlers._oauth_impl._generate_state", return_value="st"),
        ):
            lim.is_allowed.return_value = True
            result = oh.handle("/api/v1/auth/oauth/google", {}, handler, method="GET")
            assert result is not None
            assert result.status_code == 302

    def test_authenticated_user_passes_user_id_to_state(self, _patch_google_secrets):
        oh = _make_oauth_handler()
        handler = _make_handler_obj()
        auth = _make_auth_ctx(is_authenticated=True, user_id="uid-42")
        with (
            mock.patch("aragora.server.handlers._oauth_impl._oauth_limiter") as lim,
            mock.patch("aragora.billing.jwt_auth.extract_user_from_request", return_value=auth),
            mock.patch(
                "aragora.server.handlers._oauth_impl._generate_state", return_value="st"
            ) as gen,
        ):
            lim.is_allowed.return_value = True
            oh.handle("/api/auth/oauth/google", {}, handler, method="GET")
            gen.assert_called_once()
            call_kwargs = gen.call_args
            # _generate_state is called with keyword args user_id and redirect_url
            assert call_kwargs[1].get("user_id") == "uid-42" or (
                call_kwargs[0] and call_kwargs[0][0] == "uid-42"
            )


# ===========================================================================
# Tests: Google OAuth callback
# ===========================================================================


class TestGoogleCallback:
    def test_error_from_google_redirects(self):
        oh = _make_oauth_handler()
        handler = _make_handler_obj()
        with (
            mock.patch("aragora.server.handlers._oauth_impl._oauth_limiter") as lim,
            mock.patch(
                "aragora.server.handlers._oauth_impl._get_oauth_error_url",
                return_value="http://localhost:3000/auth/error",
            ),
        ):
            lim.is_allowed.return_value = True
            result = oh.handle(
                "/api/auth/oauth/google/callback",
                {"error": "access_denied", "error_description": "User denied"},
                handler,
                method="GET",
            )
            assert result is not None
            assert result.status_code == 302
            assert "error" in result.headers.get("Location", "").lower()

    def test_missing_state_redirects_with_error(self):
        oh = _make_oauth_handler()
        handler = _make_handler_obj()
        with (
            mock.patch("aragora.server.handlers._oauth_impl._oauth_limiter") as lim,
            mock.patch(
                "aragora.server.handlers._oauth_impl._get_oauth_error_url",
                return_value="http://localhost:3000/auth/error",
            ),
        ):
            lim.is_allowed.return_value = True
            result = oh.handle("/api/auth/oauth/google/callback", {}, handler, method="GET")
            assert result is not None
            assert result.status_code == 302
            loc = result.headers.get("Location", "")
            assert "state" in loc.lower()

    def test_invalid_state_redirects_with_error(self):
        oh = _make_oauth_handler()
        handler = _make_handler_obj()
        with (
            mock.patch("aragora.server.handlers._oauth_impl._oauth_limiter") as lim,
            mock.patch("aragora.server.handlers._oauth_impl._validate_state", return_value=None),
            mock.patch(
                "aragora.server.handlers._oauth_impl._get_oauth_error_url",
                return_value="http://localhost:3000/auth/error",
            ),
        ):
            lim.is_allowed.return_value = True
            result = oh.handle(
                "/api/auth/oauth/google/callback",
                {"state": "bad-state-token"},
                handler,
                method="GET",
            )
            assert result is not None
            assert result.status_code == 302
            loc = result.headers.get("Location", "")
            assert "invalid" in loc.lower() or "expired" in loc.lower()

    def test_missing_code_redirects_with_error(self):
        oh = _make_oauth_handler()
        handler = _make_handler_obj()
        state_data = {"redirect_url": "http://localhost:3000/auth/callback"}
        with (
            mock.patch("aragora.server.handlers._oauth_impl._oauth_limiter") as lim,
            mock.patch(
                "aragora.server.handlers._oauth_impl._validate_state", return_value=state_data
            ),
            mock.patch(
                "aragora.server.handlers._oauth_impl._get_oauth_error_url",
                return_value="http://localhost:3000/auth/error",
            ),
        ):
            lim.is_allowed.return_value = True
            result = oh.handle(
                "/api/auth/oauth/google/callback", {"state": "valid-state"}, handler, method="GET"
            )
            assert result is not None
            assert result.status_code == 302
            loc = result.headers.get("Location", "")
            assert "authorization" in loc.lower() or "code" in loc.lower() or "Missing" in loc

    def test_token_exchange_failure_redirects_error(self):
        oh = _make_oauth_handler()
        handler = _make_handler_obj()
        state_data = {"redirect_url": "http://localhost:3000/auth/callback"}
        with (
            mock.patch("aragora.server.handlers._oauth_impl._oauth_limiter") as lim,
            mock.patch(
                "aragora.server.handlers._oauth_impl._validate_state", return_value=state_data
            ),
            mock.patch.object(
                oh, "_exchange_code_for_tokens", side_effect=Exception("network error")
            ),
            mock.patch(
                "aragora.server.handlers._oauth_impl._get_oauth_error_url",
                return_value="http://localhost:3000/auth/error",
            ),
        ):
            lim.is_allowed.return_value = True
            result = oh.handle(
                "/api/auth/oauth/google/callback",
                {"state": "valid", "code": "authcode123"},
                handler,
                method="GET",
            )
            assert result is not None
            assert result.status_code == 302
            loc = result.headers.get("Location", "")
            assert "Failed" in loc or "error" in loc.lower()

    def test_no_access_token_redirects_error(self):
        oh = _make_oauth_handler()
        handler = _make_handler_obj()
        state_data = {"redirect_url": "http://localhost:3000/auth/callback"}
        with (
            mock.patch("aragora.server.handlers._oauth_impl._oauth_limiter") as lim,
            mock.patch(
                "aragora.server.handlers._oauth_impl._validate_state", return_value=state_data
            ),
            mock.patch.object(oh, "_exchange_code_for_tokens", return_value={}),
            mock.patch(
                "aragora.server.handlers._oauth_impl._get_oauth_error_url",
                return_value="http://localhost:3000/auth/error",
            ),
        ):
            lim.is_allowed.return_value = True
            result = oh.handle(
                "/api/auth/oauth/google/callback",
                {"state": "valid", "code": "authcode123"},
                handler,
                method="GET",
            )
            assert result is not None
            assert result.status_code == 302
            assert "access" in result.headers.get("Location", "").lower()

    def test_successful_callback_new_user(self):
        from aragora.server.handlers._oauth_impl import OAuthUserInfo

        oh = _make_oauth_handler({"user_store": mock.MagicMock()})
        handler = _make_handler_obj()
        state_data = {"redirect_url": "http://localhost:3000/auth/callback"}
        user = _make_user()
        user_info = OAuthUserInfo(
            provider="google",
            provider_user_id="g123",
            email="test@gmail.com",
            name="Test User",
            email_verified=True,
        )
        tokens = _make_token_pair()

        with (
            mock.patch("aragora.server.handlers._oauth_impl._oauth_limiter") as lim,
            mock.patch(
                "aragora.server.handlers._oauth_impl._validate_state", return_value=state_data
            ),
            mock.patch.object(
                oh, "_exchange_code_for_tokens", return_value={"access_token": "at123"}
            ),
            mock.patch.object(oh, "_get_google_user_info", return_value=user_info),
            mock.patch.object(oh, "_find_user_by_oauth", return_value=None),
            mock.patch.object(oh, "_create_oauth_user", return_value=user),
            mock.patch(
                "aragora.server.handlers._oauth_impl._get_oauth_error_url",
                return_value="http://localhost:3000/auth/error",
            ),
            mock.patch(
                "aragora.server.handlers._oauth_impl._get_oauth_success_url",
                return_value="http://localhost:3000/auth/callback",
            ),
            mock.patch("aragora.billing.jwt_auth.create_token_pair", return_value=tokens),
        ):
            lim.is_allowed.return_value = True
            us = oh._get_user_store()
            us.get_user_by_email.return_value = None

            result = oh.handle(
                "/api/auth/oauth/google/callback",
                {"state": "valid", "code": "authcode123"},
                handler,
                method="GET",
            )
            assert result is not None
            assert result.status_code == 302
            loc = result.headers.get("Location", "")
            assert "access_token=test-access-token" in loc
            assert "refresh_token=test-refresh-token" in loc

    def test_callback_existing_user_by_oauth(self):
        from aragora.server.handlers._oauth_impl import OAuthUserInfo

        oh = _make_oauth_handler({"user_store": mock.MagicMock()})
        handler = _make_handler_obj()
        state_data = {"redirect_url": "http://localhost:3000/auth/callback"}
        user = _make_user()
        user_info = OAuthUserInfo(
            provider="google",
            provider_user_id="g123",
            email="test@gmail.com",
            name="Test User",
            email_verified=True,
        )
        tokens = _make_token_pair()

        with (
            mock.patch("aragora.server.handlers._oauth_impl._oauth_limiter") as lim,
            mock.patch(
                "aragora.server.handlers._oauth_impl._validate_state", return_value=state_data
            ),
            mock.patch.object(oh, "_exchange_code_for_tokens", return_value={"access_token": "at"}),
            mock.patch.object(oh, "_get_google_user_info", return_value=user_info),
            mock.patch.object(oh, "_find_user_by_oauth", return_value=user),
            mock.patch(
                "aragora.server.handlers._oauth_impl._get_oauth_error_url",
                return_value="http://localhost:3000/auth/error",
            ),
            mock.patch(
                "aragora.server.handlers._oauth_impl._get_oauth_success_url",
                return_value="http://localhost:3000/auth/callback",
            ),
            mock.patch("aragora.billing.jwt_auth.create_token_pair", return_value=tokens),
        ):
            lim.is_allowed.return_value = True
            result = oh.handle(
                "/api/auth/oauth/google/callback",
                {"state": "valid", "code": "authcode"},
                handler,
                method="GET",
            )
            assert result is not None
            assert result.status_code == 302

    def test_callback_links_existing_email_user(self):
        from aragora.server.handlers._oauth_impl import OAuthUserInfo

        oh = _make_oauth_handler({"user_store": mock.MagicMock()})
        handler = _make_handler_obj()
        state_data = {"redirect_url": "http://localhost:3000/auth/callback"}
        user = _make_user()
        user_info = OAuthUserInfo(
            provider="google",
            provider_user_id="g123",
            email="existing@gmail.com",
            name="Existing",
            email_verified=True,
        )
        tokens = _make_token_pair()

        with (
            mock.patch("aragora.server.handlers._oauth_impl._oauth_limiter") as lim,
            mock.patch(
                "aragora.server.handlers._oauth_impl._validate_state", return_value=state_data
            ),
            mock.patch.object(oh, "_exchange_code_for_tokens", return_value={"access_token": "at"}),
            mock.patch.object(oh, "_get_google_user_info", return_value=user_info),
            mock.patch.object(oh, "_find_user_by_oauth", return_value=None),
            mock.patch.object(oh, "_link_oauth_to_user") as link_mock,
            mock.patch(
                "aragora.server.handlers._oauth_impl._get_oauth_error_url",
                return_value="http://localhost:3000/auth/error",
            ),
            mock.patch(
                "aragora.server.handlers._oauth_impl._get_oauth_success_url",
                return_value="http://localhost:3000/auth/callback",
            ),
            mock.patch("aragora.billing.jwt_auth.create_token_pair", return_value=tokens),
        ):
            lim.is_allowed.return_value = True
            us = oh._get_user_store()
            us.get_user_by_email.return_value = user

            result = oh.handle(
                "/api/auth/oauth/google/callback",
                {"state": "valid", "code": "authcode"},
                handler,
                method="GET",
            )
            assert result.status_code == 302
            link_mock.assert_called_once()

    def test_callback_user_store_unavailable(self):
        oh = _make_oauth_handler({})
        handler = _make_handler_obj()
        from aragora.server.handlers._oauth_impl import OAuthUserInfo

        state_data = {"redirect_url": "http://localhost:3000/auth/callback"}
        user_info = OAuthUserInfo(
            provider="google",
            provider_user_id="g123",
            email="t@g.com",
            name="T",
            email_verified=True,
        )
        with (
            mock.patch("aragora.server.handlers._oauth_impl._oauth_limiter") as lim,
            mock.patch(
                "aragora.server.handlers._oauth_impl._validate_state", return_value=state_data
            ),
            mock.patch.object(oh, "_exchange_code_for_tokens", return_value={"access_token": "at"}),
            mock.patch.object(oh, "_get_google_user_info", return_value=user_info),
            mock.patch(
                "aragora.server.handlers._oauth_impl._get_oauth_error_url",
                return_value="http://localhost:3000/auth/error",
            ),
        ):
            lim.is_allowed.return_value = True
            result = oh.handle(
                "/api/auth/oauth/google/callback",
                {"state": "valid", "code": "authcode"},
                handler,
                method="GET",
            )
            assert result.status_code == 302
            loc = result.headers.get("Location", "").lower()
            assert "unavailable" in loc or "error" in loc

    def test_callback_account_linking_flow(self):
        from aragora.server.handlers._oauth_impl import OAuthUserInfo

        user_store = mock.MagicMock()
        oh = _make_oauth_handler({"user_store": user_store})
        handler = _make_handler_obj()
        state_data = {"redirect_url": "http://localhost:3000/auth/callback", "user_id": "link-uid"}
        user_info = OAuthUserInfo(
            provider="google",
            provider_user_id="g123",
            email="t@g.com",
            name="T",
            email_verified=True,
        )
        user = _make_user(user_id="link-uid")
        user_store.get_user_by_id.return_value = user

        with (
            mock.patch("aragora.server.handlers._oauth_impl._oauth_limiter") as lim,
            mock.patch(
                "aragora.server.handlers._oauth_impl._validate_state", return_value=state_data
            ),
            mock.patch.object(oh, "_exchange_code_for_tokens", return_value={"access_token": "at"}),
            mock.patch.object(oh, "_get_google_user_info", return_value=user_info),
            mock.patch.object(oh, "_find_user_by_oauth", return_value=None),
            mock.patch.object(oh, "_link_oauth_to_user", return_value=True),
            mock.patch(
                "aragora.server.handlers._oauth_impl._get_oauth_error_url",
                return_value="http://localhost:3000/auth/error",
            ),
            mock.patch(
                "aragora.server.handlers._oauth_impl._get_oauth_success_url",
                return_value="http://localhost:3000/auth/callback",
            ),
        ):
            lim.is_allowed.return_value = True
            result = oh.handle(
                "/api/auth/oauth/google/callback",
                {"state": "valid", "code": "authcode"},
                handler,
                method="GET",
            )
            assert result.status_code == 302
            assert "linked=google" in result.headers.get("Location", "")

    def test_get_user_info_failure_redirects_error(self):
        oh = _make_oauth_handler()
        handler = _make_handler_obj()
        state_data = {"redirect_url": "http://localhost:3000/auth/callback"}
        with (
            mock.patch("aragora.server.handlers._oauth_impl._oauth_limiter") as lim,
            mock.patch(
                "aragora.server.handlers._oauth_impl._validate_state", return_value=state_data
            ),
            mock.patch.object(oh, "_exchange_code_for_tokens", return_value={"access_token": "at"}),
            mock.patch.object(oh, "_get_google_user_info", side_effect=Exception("API error")),
            mock.patch(
                "aragora.server.handlers._oauth_impl._get_oauth_error_url",
                return_value="http://localhost:3000/auth/error",
            ),
        ):
            lim.is_allowed.return_value = True
            result = oh.handle(
                "/api/auth/oauth/google/callback",
                {"state": "valid", "code": "authcode"},
                handler,
                method="GET",
            )
            assert result.status_code == 302
            loc = result.headers.get("Location", "").lower()
            assert "failed" in loc or "error" in loc

    def test_failed_user_creation_redirects_error(self):
        from aragora.server.handlers._oauth_impl import OAuthUserInfo

        user_store = mock.MagicMock()
        user_store.get_user_by_email.return_value = None
        oh = _make_oauth_handler({"user_store": user_store})
        handler = _make_handler_obj()
        state_data = {"redirect_url": "http://localhost:3000/auth/callback"}
        user_info = OAuthUserInfo(
            provider="google",
            provider_user_id="g123",
            email="t@g.com",
            name="T",
            email_verified=True,
        )
        with (
            mock.patch("aragora.server.handlers._oauth_impl._oauth_limiter") as lim,
            mock.patch(
                "aragora.server.handlers._oauth_impl._validate_state", return_value=state_data
            ),
            mock.patch.object(oh, "_exchange_code_for_tokens", return_value={"access_token": "at"}),
            mock.patch.object(oh, "_get_google_user_info", return_value=user_info),
            mock.patch.object(oh, "_find_user_by_oauth", return_value=None),
            mock.patch.object(oh, "_create_oauth_user", return_value=None),
            mock.patch(
                "aragora.server.handlers._oauth_impl._get_oauth_error_url",
                return_value="http://localhost:3000/auth/error",
            ),
        ):
            lim.is_allowed.return_value = True
            result = oh.handle(
                "/api/auth/oauth/google/callback",
                {"state": "valid", "code": "authcode"},
                handler,
                method="GET",
            )
            assert result.status_code == 302
            loc = result.headers.get("Location", "").lower()
            assert "failed" in loc or "error" in loc


# ===========================================================================
# Tests: GitHub OAuth start
# ===========================================================================


class TestGitHubAuthStart:
    def test_not_configured_returns_503(self):
        oh = _make_oauth_handler()
        handler = _make_handler_obj()
        with (
            mock.patch(
                "aragora.server.handlers._oauth_impl._get_github_client_id", return_value=""
            ),
            mock.patch("aragora.server.handlers._oauth_impl._oauth_limiter") as lim,
        ):
            lim.is_allowed.return_value = True
            result = oh.handle("/api/auth/oauth/github", {}, handler, method="GET")
            assert result is not None
            assert result.status_code == 503

    def test_successful_redirect(self, _patch_github_secrets):
        oh = _make_oauth_handler()
        handler = _make_handler_obj()
        with (
            mock.patch("aragora.server.handlers._oauth_impl._oauth_limiter") as lim,
            mock.patch(
                "aragora.billing.jwt_auth.extract_user_from_request", return_value=_make_auth_ctx()
            ),
            mock.patch(
                "aragora.server.handlers._oauth_impl._generate_state", return_value="gh-state"
            ),
        ):
            lim.is_allowed.return_value = True
            result = oh.handle("/api/auth/oauth/github", {}, handler, method="GET")
            assert result.status_code == 302
            loc = result.headers["Location"]
            assert "github.com" in loc
            assert "state=gh-state" in loc


# ===========================================================================
# Tests: GitHub callback
# ===========================================================================


class TestGitHubCallback:
    def test_error_from_github(self):
        oh = _make_oauth_handler()
        handler = _make_handler_obj()
        with (
            mock.patch("aragora.server.handlers._oauth_impl._oauth_limiter") as lim,
            mock.patch(
                "aragora.server.handlers._oauth_impl._get_oauth_error_url",
                return_value="http://localhost:3000/auth/error",
            ),
        ):
            lim.is_allowed.return_value = True
            result = oh.handle(
                "/api/auth/oauth/github/callback", {"error": "access_denied"}, handler, method="GET"
            )
            assert result.status_code == 302

    def test_missing_state(self):
        oh = _make_oauth_handler()
        handler = _make_handler_obj()
        with (
            mock.patch("aragora.server.handlers._oauth_impl._oauth_limiter") as lim,
            mock.patch(
                "aragora.server.handlers._oauth_impl._get_oauth_error_url",
                return_value="http://localhost:3000/auth/error",
            ),
        ):
            lim.is_allowed.return_value = True
            result = oh.handle("/api/auth/oauth/github/callback", {}, handler, method="GET")
            assert result.status_code == 302


# ===========================================================================
# Tests: Microsoft OAuth start
# ===========================================================================


class TestMicrosoftAuthStart:
    def test_not_configured_returns_503(self):
        oh = _make_oauth_handler()
        handler = _make_handler_obj()
        with (
            mock.patch(
                "aragora.server.handlers._oauth_impl._get_microsoft_client_id", return_value=""
            ),
            mock.patch("aragora.server.handlers._oauth_impl._oauth_limiter") as lim,
        ):
            lim.is_allowed.return_value = True
            result = oh.handle("/api/auth/oauth/microsoft", {}, handler, method="GET")
            assert result.status_code == 503

    def test_successful_redirect(self, _patch_microsoft_secrets):
        oh = _make_oauth_handler()
        handler = _make_handler_obj()
        with (
            mock.patch("aragora.server.handlers._oauth_impl._oauth_limiter") as lim,
            mock.patch(
                "aragora.billing.jwt_auth.extract_user_from_request", return_value=_make_auth_ctx()
            ),
            mock.patch(
                "aragora.server.handlers._oauth_impl._generate_state", return_value="ms-state"
            ),
        ):
            lim.is_allowed.return_value = True
            result = oh.handle("/api/auth/oauth/microsoft", {}, handler, method="GET")
            assert result.status_code == 302
            loc = result.headers["Location"]
            assert "login.microsoftonline.com" in loc
            assert "state=ms-state" in loc


# ===========================================================================
# Tests: Apple OAuth start
# ===========================================================================


class TestAppleAuthStart:
    def test_not_configured_returns_503(self):
        oh = _make_oauth_handler()
        handler = _make_handler_obj()
        with (
            mock.patch("aragora.server.handlers._oauth_impl._get_apple_client_id", return_value=""),
            mock.patch("aragora.server.handlers._oauth_impl._oauth_limiter") as lim,
        ):
            lim.is_allowed.return_value = True
            result = oh.handle("/api/auth/oauth/apple", {}, handler, method="GET")
            assert result.status_code == 503

    def test_successful_redirect(self):
        oh = _make_oauth_handler()
        handler = _make_handler_obj()
        with (
            mock.patch(
                "aragora.server.handlers._oauth_impl._get_apple_client_id", return_value="apple-id"
            ),
            mock.patch(
                "aragora.server.handlers._oauth_impl._get_apple_redirect_uri",
                return_value="http://localhost:8080/api/auth/oauth/apple/callback",
            ),
            mock.patch(
                "aragora.server.handlers._oauth_impl._get_oauth_success_url",
                return_value="http://localhost:3000/auth/callback",
            ),
            mock.patch(
                "aragora.server.handlers._oauth_impl._get_allowed_redirect_hosts",
                return_value=frozenset({"localhost"}),
            ),
            mock.patch("aragora.server.handlers._oauth_impl._oauth_limiter") as lim,
            mock.patch(
                "aragora.billing.jwt_auth.extract_user_from_request", return_value=_make_auth_ctx()
            ),
            mock.patch(
                "aragora.server.handlers._oauth_impl._generate_state", return_value="apple-state"
            ),
        ):
            lim.is_allowed.return_value = True
            result = oh.handle("/api/auth/oauth/apple", {}, handler, method="GET")
            assert result.status_code == 302
            loc = result.headers["Location"]
            assert "appleid.apple.com" in loc
            assert "response_mode=form_post" in loc


# ===========================================================================
# Tests: OIDC OAuth start
# ===========================================================================


class TestOIDCAuthStart:
    def test_not_configured_returns_503(self):
        oh = _make_oauth_handler()
        handler = _make_handler_obj()
        with (
            mock.patch("aragora.server.handlers._oauth_impl._get_oidc_issuer", return_value=""),
            mock.patch("aragora.server.handlers._oauth_impl._get_oidc_client_id", return_value=""),
            mock.patch("aragora.server.handlers._oauth_impl._oauth_limiter") as lim,
        ):
            lim.is_allowed.return_value = True
            result = oh.handle("/api/auth/oauth/oidc", {}, handler, method="GET")
            assert result.status_code == 503


# ===========================================================================
# Tests: List providers
# ===========================================================================


class TestListProviders:
    def _patch_all_providers(
        self, google="", github="", microsoft="", apple="", oidc_issuer="", oidc_client=""
    ):
        """Context manager to patch all provider getters."""
        return [
            mock.patch(
                "aragora.server.handlers._oauth_impl._get_google_client_id", return_value=google
            ),
            mock.patch(
                "aragora.server.handlers._oauth_impl._get_github_client_id", return_value=github
            ),
            mock.patch(
                "aragora.server.handlers._oauth_impl._get_microsoft_client_id",
                return_value=microsoft,
            ),
            mock.patch(
                "aragora.server.handlers._oauth_impl._get_apple_client_id", return_value=apple
            ),
            mock.patch(
                "aragora.server.handlers._oauth_impl._get_oidc_issuer", return_value=oidc_issuer
            ),
            mock.patch(
                "aragora.server.handlers._oauth_impl._get_oidc_client_id", return_value=oidc_client
            ),
            mock.patch("aragora.server.handlers._oauth_impl._oauth_limiter"),
        ]

    def test_no_providers_configured(self):
        oh = _make_oauth_handler()
        handler = _make_handler_obj()
        patches = self._patch_all_providers()
        for p in patches:
            p.start()
        try:
            patches[-1].start().is_allowed.return_value = True  # already started
            result = oh.handle("/api/auth/oauth/providers", {}, handler, method="GET")
            assert result.status_code == 200
            data = json.loads(result.body)
            assert data["providers"] == []
        finally:
            for p in patches:
                p.stop()

    def test_google_provider_listed(self):
        oh = _make_oauth_handler()
        handler = _make_handler_obj()
        with (
            mock.patch(
                "aragora.server.handlers._oauth_impl._get_google_client_id", return_value="gid"
            ),
            mock.patch(
                "aragora.server.handlers._oauth_impl._get_github_client_id", return_value=""
            ),
            mock.patch(
                "aragora.server.handlers._oauth_impl._get_microsoft_client_id", return_value=""
            ),
            mock.patch("aragora.server.handlers._oauth_impl._get_apple_client_id", return_value=""),
            mock.patch("aragora.server.handlers._oauth_impl._get_oidc_issuer", return_value=""),
            mock.patch("aragora.server.handlers._oauth_impl._get_oidc_client_id", return_value=""),
            mock.patch("aragora.server.handlers._oauth_impl._oauth_limiter") as lim,
        ):
            lim.is_allowed.return_value = True
            result = oh.handle("/api/auth/oauth/providers", {}, handler, method="GET")
            data = json.loads(result.body)
            assert len(data["providers"]) == 1
            assert data["providers"][0]["id"] == "google"

    def test_all_providers_listed(self):
        oh = _make_oauth_handler()
        handler = _make_handler_obj()
        with (
            mock.patch(
                "aragora.server.handlers._oauth_impl._get_google_client_id", return_value="gid"
            ),
            mock.patch(
                "aragora.server.handlers._oauth_impl._get_github_client_id", return_value="ghid"
            ),
            mock.patch(
                "aragora.server.handlers._oauth_impl._get_microsoft_client_id", return_value="msid"
            ),
            mock.patch(
                "aragora.server.handlers._oauth_impl._get_apple_client_id", return_value="apid"
            ),
            mock.patch(
                "aragora.server.handlers._oauth_impl._get_oidc_issuer",
                return_value="https://oidc.example.com",
            ),
            mock.patch(
                "aragora.server.handlers._oauth_impl._get_oidc_client_id", return_value="oid"
            ),
            mock.patch("aragora.server.handlers._oauth_impl._oauth_limiter") as lim,
        ):
            lim.is_allowed.return_value = True
            result = oh.handle("/api/auth/oauth/providers", {}, handler, method="GET")
            data = json.loads(result.body)
            ids = {p["id"] for p in data["providers"]}
            assert ids == {"google", "github", "microsoft", "apple", "oidc"}


# ===========================================================================
# Tests: OAuth URL endpoint
# ===========================================================================


class TestOAuthUrl:
    def test_missing_provider(self):
        oh = _make_oauth_handler()
        handler = _make_handler_obj()
        with mock.patch("aragora.server.handlers._oauth_impl._oauth_limiter") as lim:
            lim.is_allowed.return_value = True
            result = oh.handle("/api/auth/oauth/url", {}, handler, method="GET")
            assert result.status_code == 400

    def test_unsupported_provider(self):
        oh = _make_oauth_handler()
        handler = _make_handler_obj()
        with mock.patch("aragora.server.handlers._oauth_impl._oauth_limiter") as lim:
            lim.is_allowed.return_value = True
            result = oh.handle(
                "/api/auth/oauth/url", {"provider": "linkedin"}, handler, method="GET"
            )
            assert result.status_code == 400

    def test_returns_auth_url_json(self, _patch_google_secrets):
        oh = _make_oauth_handler()
        handler = _make_handler_obj()
        with (
            mock.patch("aragora.server.handlers._oauth_impl._oauth_limiter") as lim,
            mock.patch(
                "aragora.billing.jwt_auth.extract_user_from_request", return_value=_make_auth_ctx()
            ),
            mock.patch(
                "aragora.server.handlers._oauth_impl._generate_state", return_value="url-state"
            ),
        ):
            lim.is_allowed.return_value = True
            result = oh.handle("/api/auth/oauth/url", {"provider": "google"}, handler, method="GET")
            assert result.status_code == 200
            data = json.loads(result.body)
            assert "auth_url" in data
            assert "accounts.google.com" in data["auth_url"]
            assert data["state"] == "url-state"

    def test_authorize_alias_works(self, _patch_google_secrets):
        oh = _make_oauth_handler()
        handler = _make_handler_obj()
        with (
            mock.patch("aragora.server.handlers._oauth_impl._oauth_limiter") as lim,
            mock.patch(
                "aragora.billing.jwt_auth.extract_user_from_request", return_value=_make_auth_ctx()
            ),
            mock.patch("aragora.server.handlers._oauth_impl._generate_state", return_value="st"),
        ):
            lim.is_allowed.return_value = True
            result = oh.handle(
                "/api/auth/oauth/authorize", {"provider": "google"}, handler, method="GET"
            )
            assert result.status_code == 200


# ===========================================================================
# Tests: OAuthUserInfo dataclass
# ===========================================================================


class TestOAuthUserInfo:
    def test_creation(self):
        from aragora.server.handlers._oauth_impl import OAuthUserInfo

        info = OAuthUserInfo(
            provider="google", provider_user_id="123", email="test@test.com", name="Test"
        )
        assert info.provider == "google"
        assert info.email_verified is False
        assert info.picture is None

    def test_with_all_fields(self):
        from aragora.server.handlers._oauth_impl import OAuthUserInfo

        info = OAuthUserInfo(
            provider="github",
            provider_user_id="456",
            email="dev@gh.com",
            name="Dev",
            picture="https://avatar.url",
            email_verified=True,
        )
        assert info.picture == "https://avatar.url"
        assert info.email_verified is True


# ===========================================================================
# Tests: _complete_oauth_flow
# ===========================================================================


class TestCompleteOAuthFlow:
    def test_no_user_store(self):
        oh = _make_oauth_handler({})
        from aragora.server.handlers._oauth_impl import OAuthUserInfo

        info = OAuthUserInfo(
            provider="microsoft", provider_user_id="ms1", email="t@t.com", name="T"
        )
        with mock.patch(
            "aragora.server.handlers._oauth_impl._get_oauth_error_url",
            return_value="http://localhost:3000/auth/error",
        ):
            result = oh._complete_oauth_flow(info, {"redirect_url": "http://localhost:3000"})
            assert result.status_code == 302
            loc = result.headers.get("Location", "").lower()
            assert "unavailable" in loc or "error" in loc

    def test_user_creation_failure(self):
        from aragora.server.handlers._oauth_impl import OAuthUserInfo

        user_store = mock.MagicMock()
        user_store.get_user_by_email.return_value = None
        oh = _make_oauth_handler({"user_store": user_store})
        info = OAuthUserInfo(
            provider="microsoft", provider_user_id="ms1", email="t@t.com", name="T"
        )
        with (
            mock.patch.object(oh, "_find_user_by_oauth", return_value=None),
            mock.patch.object(oh, "_create_oauth_user", return_value=None),
            mock.patch(
                "aragora.server.handlers._oauth_impl._get_oauth_error_url",
                return_value="http://localhost:3000/auth/error",
            ),
        ):
            result = oh._complete_oauth_flow(info, {"redirect_url": "http://localhost:3000"})
            assert result.status_code == 302
            loc = result.headers.get("Location", "").lower()
            assert "failed" in loc or "error" in loc

    def test_successful_flow_creates_tokens(self):
        from aragora.server.handlers._oauth_impl import OAuthUserInfo

        user_store = mock.MagicMock()
        user = _make_user()
        user_store.get_user_by_email.return_value = None
        oh = _make_oauth_handler({"user_store": user_store})
        info = OAuthUserInfo(
            provider="microsoft", provider_user_id="ms1", email="t@t.com", name="T"
        )
        tokens = _make_token_pair()
        with (
            mock.patch.object(oh, "_find_user_by_oauth", return_value=None),
            mock.patch.object(oh, "_create_oauth_user", return_value=user),
            mock.patch("aragora.billing.jwt_auth.create_token_pair", return_value=tokens),
            mock.patch(
                "aragora.server.handlers._oauth_impl._get_oauth_success_url",
                return_value="http://localhost:3000/auth/callback",
            ),
        ):
            result = oh._complete_oauth_flow(
                info, {"redirect_url": "http://localhost:3000/auth/callback"}
            )
            assert result.status_code == 302
            assert "access_token" in result.headers.get("Location", "")


# ===========================================================================
# Tests: _find_user_by_oauth
# ===========================================================================


class TestFindUserByOAuth:
    def test_store_supports_oauth_lookup(self):
        oh = _make_oauth_handler()
        from aragora.server.handlers._oauth_impl import OAuthUserInfo

        store = mock.MagicMock()
        store.get_user_by_oauth.return_value = "found_user"
        info = OAuthUserInfo(provider="google", provider_user_id="g1", email="t@t.com", name="T")
        result = oh._find_user_by_oauth(store, info)
        assert result == "found_user"
        store.get_user_by_oauth.assert_called_with("google", "g1")

    def test_store_without_oauth_lookup(self):
        oh = _make_oauth_handler()
        from aragora.server.handlers._oauth_impl import OAuthUserInfo

        store = mock.MagicMock(spec=[])
        info = OAuthUserInfo(provider="google", provider_user_id="g1", email="t@t.com", name="T")
        result = oh._find_user_by_oauth(store, info)
        assert result is None


# ===========================================================================
# Tests: _link_oauth_to_user
# ===========================================================================


class TestLinkOAuthToUser:
    def test_store_supports_linking(self):
        oh = _make_oauth_handler()
        from aragora.server.handlers._oauth_impl import OAuthUserInfo

        store = mock.MagicMock()
        store.link_oauth_provider.return_value = True
        info = OAuthUserInfo(provider="github", provider_user_id="gh1", email="t@t.com", name="T")
        result = oh._link_oauth_to_user(store, "uid-1", info)
        assert result is True

    def test_store_without_linking(self):
        oh = _make_oauth_handler()
        from aragora.server.handlers._oauth_impl import OAuthUserInfo

        store = mock.MagicMock(spec=[])
        info = OAuthUserInfo(provider="github", provider_user_id="gh1", email="t@t.com", name="T")
        result = oh._link_oauth_to_user(store, "uid-1", info)
        assert result is False


# ===========================================================================
# Tests: _handle_account_linking
# ===========================================================================


class TestHandleAccountLinking:
    def test_user_not_found(self):
        user_store = mock.MagicMock()
        user_store.get_user_by_id.return_value = None
        oh = _make_oauth_handler({"user_store": user_store})
        from aragora.server.handlers._oauth_impl import OAuthUserInfo

        info = OAuthUserInfo(provider="google", provider_user_id="g1", email="t@t.com", name="T")
        with mock.patch(
            "aragora.server.handlers._oauth_impl._get_oauth_error_url",
            return_value="http://localhost:3000/auth/error",
        ):
            result = oh._handle_account_linking(user_store, "uid-missing", info, {})
            assert result.status_code == 302
            loc = result.headers.get("Location", "").lower()
            assert "not" in loc and "found" in loc

    def test_oauth_already_linked_to_other_user(self):
        user_store = mock.MagicMock()
        user_store.get_user_by_id.return_value = _make_user(user_id="uid-1")
        oh = _make_oauth_handler({"user_store": user_store})
        from aragora.server.handlers._oauth_impl import OAuthUserInfo

        info = OAuthUserInfo(provider="google", provider_user_id="g1", email="t@t.com", name="T")
        other_user = _make_user(user_id="uid-other")
        with (
            mock.patch.object(oh, "_find_user_by_oauth", return_value=other_user),
            mock.patch(
                "aragora.server.handlers._oauth_impl._get_oauth_error_url",
                return_value="http://localhost:3000/auth/error",
            ),
        ):
            result = oh._handle_account_linking(user_store, "uid-1", info, {})
            assert result.status_code == 302
            assert "already" in result.headers.get("Location", "").lower()

    def test_successful_linking(self):
        user_store = mock.MagicMock()
        user_store.get_user_by_id.return_value = _make_user(user_id="uid-1")
        oh = _make_oauth_handler({"user_store": user_store})
        from aragora.server.handlers._oauth_impl import OAuthUserInfo

        info = OAuthUserInfo(provider="google", provider_user_id="g1", email="t@t.com", name="T")
        with (
            mock.patch.object(oh, "_find_user_by_oauth", return_value=None),
            mock.patch.object(oh, "_link_oauth_to_user", return_value=True),
            mock.patch(
                "aragora.server.handlers._oauth_impl._get_oauth_success_url",
                return_value="http://localhost:3000/auth/callback",
            ),
        ):
            result = oh._handle_account_linking(
                user_store, "uid-1", info, {"redirect_url": "http://localhost:3000/auth/callback"}
            )
            assert result.status_code == 302
            assert "linked=google" in result.headers.get("Location", "")


# ===========================================================================
# Tests: _redirect_with_tokens
# ===========================================================================


class TestRedirectWithTokens:
    def test_redirect_includes_tokens(self):
        oh = _make_oauth_handler()
        tokens = _make_token_pair()
        result = oh._redirect_with_tokens("http://localhost:3000/callback", tokens)
        assert result.status_code == 302
        loc = result.headers["Location"]
        assert "access_token=test-access-token" in loc
        assert "refresh_token=test-refresh-token" in loc
        assert "token_type=Bearer" in loc
        assert "expires_in=3600" in loc

    def test_no_cache_headers(self):
        oh = _make_oauth_handler()
        tokens = _make_token_pair()
        result = oh._redirect_with_tokens("http://localhost:3000/callback", tokens)
        assert "no-store" in result.headers.get("Cache-Control", "")
        assert result.headers.get("Pragma") == "no-cache"
        assert result.headers.get("Expires") == "0"


# ===========================================================================
# Tests: _redirect_with_error
# ===========================================================================


class TestRedirectWithError:
    def test_error_redirect(self):
        oh = _make_oauth_handler()
        with mock.patch(
            "aragora.server.handlers._oauth_impl._get_oauth_error_url",
            return_value="http://localhost:3000/auth/error",
        ):
            result = oh._redirect_with_error("Something went wrong")
            assert result.status_code == 302
            loc = result.headers["Location"]
            assert "error=" in loc
            assert "Something" in loc

    def test_no_cache_headers_on_error(self):
        oh = _make_oauth_handler()
        with mock.patch(
            "aragora.server.handlers._oauth_impl._get_oauth_error_url",
            return_value="http://localhost:3000/auth/error",
        ):
            result = oh._redirect_with_error("err")
            assert "no-store" in result.headers.get("Cache-Control", "")

    def test_url_encodes_error_message(self):
        oh = _make_oauth_handler()
        with mock.patch(
            "aragora.server.handlers._oauth_impl._get_oauth_error_url",
            return_value="http://localhost:3000/auth/error",
        ):
            result = oh._redirect_with_error("has spaces & special=chars")
            loc = result.headers["Location"]
            # Should be URL-encoded
            assert "has%20spaces" in loc or "has+spaces" in loc


# ===========================================================================
# Tests: CSRF state validation
# ===========================================================================


class TestCsrfState:
    def test_validate_state_delegates(self):
        from aragora.server.handlers._oauth_impl import _validate_state

        with mock.patch(
            "aragora.server.handlers._oauth_impl._validate_state_internal",
            return_value={"user_id": None},
        ) as vi:
            result = _validate_state("tok123")
            vi.assert_called_once_with("tok123")
            assert result == {"user_id": None}

    def test_validate_state_returns_none_for_invalid(self):
        from aragora.server.handlers._oauth_impl import _validate_state

        with mock.patch(
            "aragora.server.handlers._oauth_impl._validate_state_internal", return_value=None
        ):
            assert _validate_state("bad") is None


# ===========================================================================
# Tests: _cleanup_expired_states
# ===========================================================================


class TestCleanupExpiredStates:
    def test_cleanup_works(self):
        from aragora.server.handlers._oauth_impl import _cleanup_expired_states

        result = _cleanup_expired_states()
        assert isinstance(result, int)


# ===========================================================================
# Tests: Provider detection in handle()
# ===========================================================================


class TestProviderDetection:
    def test_google_provider_detected(self):
        oh = _make_oauth_handler()
        handler = _make_handler_obj()
        with (
            mock.patch("aragora.server.handlers._oauth_impl._oauth_limiter") as lim,
            mock.patch(
                "aragora.server.handlers._oauth_impl._get_google_client_id", return_value=""
            ),
            mock.patch("aragora.server.handlers._oauth_impl.create_span") as cs,
        ):
            lim.is_allowed.return_value = True
            cs.return_value.__enter__ = mock.MagicMock(return_value=mock.MagicMock())
            cs.return_value.__exit__ = mock.MagicMock(return_value=False)
            oh.handle("/api/auth/oauth/google", {}, handler, method="GET")
            cs.assert_called_once()
            args = cs.call_args
            assert args[0][0] == "oauth.google"

    def test_github_provider_detected(self):
        oh = _make_oauth_handler()
        handler = _make_handler_obj()
        with (
            mock.patch("aragora.server.handlers._oauth_impl._oauth_limiter") as lim,
            mock.patch(
                "aragora.server.handlers._oauth_impl._get_github_client_id", return_value=""
            ),
            mock.patch("aragora.server.handlers._oauth_impl.create_span") as cs,
        ):
            lim.is_allowed.return_value = True
            cs.return_value.__enter__ = mock.MagicMock(return_value=mock.MagicMock())
            cs.return_value.__exit__ = mock.MagicMock(return_value=False)
            oh.handle("/api/auth/oauth/github", {}, handler, method="GET")
            assert cs.call_args[0][0] == "oauth.github"


# ===========================================================================
# Tests: _is_production
# ===========================================================================


class TestIsProduction:
    def test_not_production_by_default(self, monkeypatch):
        monkeypatch.delenv("ARAGORA_ENV", raising=False)
        from aragora.server.handlers._oauth_impl import _is_production

        assert _is_production() is False

    def test_production_when_set(self, monkeypatch):
        monkeypatch.setenv("ARAGORA_ENV", "production")
        from aragora.server.handlers._oauth_impl import _is_production

        assert _is_production() is True

    def test_production_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("ARAGORA_ENV", "PRODUCTION")
        from aragora.server.handlers._oauth_impl import _is_production

        assert _is_production() is True


# ===========================================================================
# Tests: validate_oauth_config
# ===========================================================================


class TestValidateOAuthConfig:
    def test_dev_mode_no_validation(self):
        from aragora.server.handlers._oauth_impl import validate_oauth_config

        result = validate_oauth_config()
        assert result == []


# ===========================================================================
# Tests: _OAuthStatesView
# ===========================================================================


class TestOAuthStatesView:
    def _make_store_with_memory(self):
        """Create a store object that has _memory_store._states as _OAuthStatesView expects."""
        from aragora.server.oauth_state_store import InMemoryOAuthStateStore

        memory_store = InMemoryOAuthStateStore()
        store = types.SimpleNamespace(_memory_store=memory_store)
        return store

    def test_view_set_and_get(self):
        from aragora.server.handlers._oauth_impl import _OAuthStatesView
        from aragora.server.oauth_state_store import OAuthState

        store = self._make_store_with_memory()
        view = _OAuthStatesView(store)
        state = OAuthState(
            user_id="u1", redirect_url="http://localhost", expires_at=time.time() + 600
        )
        view["key1"] = state
        result = view["key1"]
        assert isinstance(result, dict)
        assert result["user_id"] == "u1"

    def test_view_len(self):
        from aragora.server.handlers._oauth_impl import _OAuthStatesView

        store = self._make_store_with_memory()
        view = _OAuthStatesView(store)
        assert len(view) == 0

    def test_view_delete(self):
        from aragora.server.handlers._oauth_impl import _OAuthStatesView
        from aragora.server.oauth_state_store import OAuthState

        store = self._make_store_with_memory()
        view = _OAuthStatesView(store)
        state = OAuthState(user_id=None, redirect_url=None, expires_at=time.time() + 600)
        view["k"] = state
        del view["k"]
        assert len(view) == 0

    def test_view_iter(self):
        from aragora.server.handlers._oauth_impl import _OAuthStatesView
        from aragora.server.oauth_state_store import OAuthState

        store = self._make_store_with_memory()
        view = _OAuthStatesView(store)
        state = OAuthState(user_id=None, redirect_url=None, expires_at=time.time() + 600)
        view["a"] = state
        view["b"] = state
        keys = list(view)
        assert set(keys) == {"a", "b"}

    def test_view_get_default(self):
        from aragora.server.handlers._oauth_impl import _OAuthStatesView

        store = self._make_store_with_memory()
        view = _OAuthStatesView(store)
        assert view.get("missing", "default") == "default"

    def test_view_values(self):
        from aragora.server.handlers._oauth_impl import _OAuthStatesView
        from aragora.server.oauth_state_store import OAuthState

        store = self._make_store_with_memory()
        view = _OAuthStatesView(store)
        state = OAuthState(user_id="u1", redirect_url=None, expires_at=time.time() + 600)
        view["k1"] = state
        vals = view.values()
        assert len(vals) == 1
        assert vals[0]["user_id"] == "u1"

    def test_view_items(self):
        from aragora.server.handlers._oauth_impl import _OAuthStatesView
        from aragora.server.oauth_state_store import OAuthState

        store = self._make_store_with_memory()
        view = _OAuthStatesView(store)
        state = OAuthState(user_id="u2", redirect_url=None, expires_at=time.time() + 600)
        view["k2"] = state
        items = view.items()
        assert len(items) == 1
        assert items[0][0] == "k2"

    def test_view_set_dict(self):
        from aragora.server.handlers._oauth_impl import _OAuthStatesView

        store = self._make_store_with_memory()
        view = _OAuthStatesView(store)
        view["k3"] = {"user_id": "u3", "redirect_url": None, "expires_at": time.time() + 600}
        result = view["k3"]
        assert result["user_id"] == "u3"

    def test_view_set_plain_value(self):
        from aragora.server.handlers._oauth_impl import _OAuthStatesView

        store = self._make_store_with_memory()
        view = _OAuthStatesView(store)
        view["k4"] = "plain_value"
        result = view["k4"]
        assert result == {"value": "plain_value"}


# ===========================================================================
# Tests: Redirect URL helper functions
# ===========================================================================


class TestRedirectUrlHelpers:
    def test_google_redirect_uri_default(self, monkeypatch):
        monkeypatch.delenv("ARAGORA_ENV", raising=False)
        with mock.patch("aragora.server.handlers._oauth_impl._get_secret", return_value=""):
            from aragora.server.handlers._oauth_impl import _get_google_redirect_uri

            result = _get_google_redirect_uri()
            assert "localhost" in result

    def test_github_redirect_uri_default(self, monkeypatch):
        monkeypatch.delenv("ARAGORA_ENV", raising=False)
        with mock.patch("aragora.server.handlers._oauth_impl._get_secret", return_value=""):
            from aragora.server.handlers._oauth_impl import _get_github_redirect_uri

            result = _get_github_redirect_uri()
            assert "github" in result

    def test_redirect_uri_from_secret(self):
        with mock.patch(
            "aragora.server.handlers._oauth_impl._get_secret",
            return_value="https://custom.example.com/callback",
        ):
            from aragora.server.handlers._oauth_impl import _get_google_redirect_uri

            assert _get_google_redirect_uri() == "https://custom.example.com/callback"

    def test_production_empty_when_no_secret(self, monkeypatch):
        monkeypatch.setenv("ARAGORA_ENV", "production")
        with mock.patch("aragora.server.handlers._oauth_impl._get_secret", return_value=""):
            from aragora.server.handlers._oauth_impl import _get_google_redirect_uri

            assert _get_google_redirect_uri() == ""

    def test_allowed_redirect_hosts_default(self, monkeypatch):
        monkeypatch.delenv("ARAGORA_ENV", raising=False)
        with mock.patch("aragora.server.handlers._oauth_impl._get_secret", return_value=""):
            from aragora.server.handlers._oauth_impl import _get_allowed_redirect_hosts

            hosts = _get_allowed_redirect_hosts()
            assert "localhost" in hosts
            assert "127.0.0.1" in hosts

    def test_allowed_redirect_hosts_custom(self):
        with mock.patch(
            "aragora.server.handlers._oauth_impl._get_secret",
            return_value="example.com,app.example.com",
        ):
            from aragora.server.handlers._oauth_impl import _get_allowed_redirect_hosts

            hosts = _get_allowed_redirect_hosts()
            assert "example.com" in hosts
            assert "app.example.com" in hosts

    def test_microsoft_redirect_uri_default(self, monkeypatch):
        monkeypatch.delenv("ARAGORA_ENV", raising=False)
        with mock.patch("aragora.server.handlers._oauth_impl._get_secret", return_value=""):
            from aragora.server.handlers._oauth_impl import _get_microsoft_redirect_uri

            result = _get_microsoft_redirect_uri()
            assert "microsoft" in result

    def test_apple_redirect_uri_default(self, monkeypatch):
        monkeypatch.delenv("ARAGORA_ENV", raising=False)
        with mock.patch("aragora.server.handlers._oauth_impl._get_secret", return_value=""):
            from aragora.server.handlers._oauth_impl import _get_apple_redirect_uri

            result = _get_apple_redirect_uri()
            assert "apple" in result

    def test_oidc_redirect_uri_default(self, monkeypatch):
        monkeypatch.delenv("ARAGORA_ENV", raising=False)
        with mock.patch("aragora.server.handlers._oauth_impl._get_secret", return_value=""):
            from aragora.server.handlers._oauth_impl import _get_oidc_redirect_uri

            result = _get_oidc_redirect_uri()
            assert "oidc" in result


# ===========================================================================
# Tests: _parse_apple_id_token
# ===========================================================================


class TestParseAppleIdToken:
    def _make_id_token(self, payload: dict) -> str:
        import base64

        header = (
            base64.urlsafe_b64encode(json.dumps({"alg": "ES256"}).encode()).decode().rstrip("=")
        )
        body = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
        sig = base64.urlsafe_b64encode(b"fakesig").decode().rstrip("=")
        return f"{header}.{body}.{sig}"

    def test_valid_token(self):
        oh = _make_oauth_handler()
        token = self._make_id_token(
            {"sub": "apple-user-1", "email": "user@icloud.com", "email_verified": True}
        )
        result = oh._parse_apple_id_token(token, {"name": {"firstName": "John", "lastName": "Doe"}})
        assert result.provider == "apple"
        assert result.email == "user@icloud.com"
        assert result.name == "John Doe"
        assert result.email_verified is True

    def test_no_email_raises(self):
        oh = _make_oauth_handler()
        token = self._make_id_token({"sub": "apple-user-1"})
        with pytest.raises(ValueError, match="No email"):
            oh._parse_apple_id_token(token, {})

    def test_invalid_format_raises(self):
        oh = _make_oauth_handler()
        with pytest.raises(ValueError, match="Invalid Apple ID token"):
            oh._parse_apple_id_token("not.a.valid.token.with.too.many.parts", {})

    def test_fallback_name_from_email(self):
        oh = _make_oauth_handler()
        token = self._make_id_token(
            {"sub": "a1", "email": "jane@apple.com", "email_verified": "true"}
        )
        result = oh._parse_apple_id_token(token, {})
        assert result.name == "jane"
        assert result.email_verified is True


# ===========================================================================
# Tests: Handler routing (method/path combinations)
# ===========================================================================


class TestHandlerRouting:
    """Test various method/path combinations route correctly."""

    def test_link_post(self):
        oh = _make_oauth_handler()
        handler = _make_handler_obj(command="POST")
        with (
            mock.patch("aragora.server.handlers._oauth_impl._oauth_limiter") as lim,
            mock.patch.object(
                oh, "_handle_link_account", return_value=mock.MagicMock(status_code=200)
            ) as m,
        ):
            lim.is_allowed.return_value = True
            oh.handle("/api/auth/oauth/link", {}, handler, method="POST")
            m.assert_called_once()

    def test_unlink_delete(self):
        oh = _make_oauth_handler()
        handler = _make_handler_obj(command="DELETE")
        with (
            mock.patch("aragora.server.handlers._oauth_impl._oauth_limiter") as lim,
            mock.patch.object(
                oh, "_handle_unlink_account", return_value=mock.MagicMock(status_code=200)
            ) as m,
        ):
            lim.is_allowed.return_value = True
            oh.handle("/api/auth/oauth/unlink", {}, handler, method="DELETE")
            m.assert_called_once()

    def test_callback_api_post(self):
        oh = _make_oauth_handler()
        handler = _make_handler_obj(command="POST")
        with (
            mock.patch("aragora.server.handlers._oauth_impl._oauth_limiter") as lim,
            mock.patch.object(
                oh, "_handle_oauth_callback_api", return_value=mock.MagicMock(status_code=200)
            ) as m,
        ):
            lim.is_allowed.return_value = True
            oh.handle("/api/auth/oauth/callback", {}, handler, method="POST")
            m.assert_called_once()

    def test_user_providers_get(self):
        oh = _make_oauth_handler()
        handler = _make_handler_obj()
        with (
            mock.patch("aragora.server.handlers._oauth_impl._oauth_limiter") as lim,
            mock.patch.object(
                oh, "_handle_get_user_providers", return_value=mock.MagicMock(status_code=200)
            ) as m,
        ):
            lim.is_allowed.return_value = True
            oh.handle("/api/user/oauth-providers", {}, handler, method="GET")
            m.assert_called_once()


# ===========================================================================
# Tests: Apple form_post (PKCE-adjacent)
# ===========================================================================


class TestAppleFormPost:
    def test_apple_callback_accepts_post(self):
        oh = _make_oauth_handler()
        handler = _make_handler_obj()
        with (
            mock.patch("aragora.server.handlers._oauth_impl._oauth_limiter") as lim,
            mock.patch.object(
                oh, "_handle_apple_callback", return_value=mock.MagicMock(status_code=302)
            ) as m,
        ):
            lim.is_allowed.return_value = True
            oh.handle("/api/auth/oauth/apple/callback", {}, handler, method="POST")
            m.assert_called_once()

    def test_apple_callback_accepts_get(self):
        oh = _make_oauth_handler()
        handler = _make_handler_obj()
        with (
            mock.patch("aragora.server.handlers._oauth_impl._oauth_limiter") as lim,
            mock.patch.object(
                oh, "_handle_apple_callback", return_value=mock.MagicMock(status_code=302)
            ) as m,
        ):
            lim.is_allowed.return_value = True
            oh.handle("/api/auth/oauth/apple/callback", {}, handler, method="GET")
            m.assert_called_once()


# ===========================================================================
# Tests: Google OAuth URL constants
# ===========================================================================


class TestOAuthConstants:
    def test_google_auth_url(self):
        from aragora.server.handlers._oauth_impl import GOOGLE_AUTH_URL

        assert "accounts.google.com" in GOOGLE_AUTH_URL

    def test_google_token_url(self):
        from aragora.server.handlers._oauth_impl import GOOGLE_TOKEN_URL

        assert "oauth2.googleapis.com" in GOOGLE_TOKEN_URL

    def test_github_auth_url(self):
        from aragora.server.handlers._oauth_impl import GITHUB_AUTH_URL

        assert "github.com" in GITHUB_AUTH_URL

    def test_microsoft_auth_url_template(self):
        from aragora.server.handlers._oauth_impl import MICROSOFT_AUTH_URL_TEMPLATE

        assert "{tenant}" in MICROSOFT_AUTH_URL_TEMPLATE

    def test_apple_auth_url(self):
        from aragora.server.handlers._oauth_impl import APPLE_AUTH_URL

        assert "appleid.apple.com" in APPLE_AUTH_URL
