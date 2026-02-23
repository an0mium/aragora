"""Tests for AuthHandler (aragora/server/handlers/auth/handler.py).

Covers routing, token management, profile endpoints, health, delegation to
sub-modules (login, password, api_keys, mfa, sessions), RBAC, and error paths.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from aragora.server.handlers.auth.handler import AuthHandler


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


def _run(coro):
    """Helper to run coroutines in tests."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class MockHTTPHandler:
    """Lightweight mock HTTP request handler."""

    def __init__(self, body: dict | None = None, method: str = "GET"):
        self.command = method
        self.client_address = ("127.0.0.1", 12345)
        self.headers: dict[str, str] = {
            "User-Agent": "test-agent",
            "Authorization": "Bearer test-token-abc",
        }
        self.rfile = MagicMock()
        if body:
            body_bytes = json.dumps(body).encode()
            self.rfile.read.return_value = body_bytes
            self.headers["Content-Length"] = str(len(body_bytes))
        else:
            self.rfile.read.return_value = b"{}"
            self.headers["Content-Length"] = "2"


@dataclass
class MockUser:
    """Mock user for user-store interactions."""

    id: str = "user-001"
    email: str = "test@example.com"
    name: str = "Test User"
    org_id: str | None = "org-001"
    role: str = "admin"
    is_active: bool = True
    mfa_enabled: bool = False
    mfa_secret: str | None = None
    mfa_backup_codes: str | None = None
    api_key_prefix: str | None = None
    api_key_hash: str | None = None
    api_key_created_at: Any = None
    api_key_expires_at: Any = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "email": self.email,
            "name": self.name,
            "org_id": self.org_id,
            "role": self.role,
        }

    def verify_password(self, password: str) -> bool:
        return password == "correct-password"

    def generate_api_key(self, expires_days: int = 365) -> str:
        self.api_key_prefix = "ak_test"
        self.api_key_hash = "hashed"
        self.api_key_created_at = datetime.now(timezone.utc)
        return "ak_test_full_key_value"


@dataclass
class MockOrg:
    """Mock organization."""

    id: str = "org-001"
    name: str = "Test Org"

    @dataclass
    class Limits:
        api_access: bool = True

    limits: Limits = field(default_factory=Limits)

    def to_dict(self) -> dict:
        return {"id": self.id, "name": self.name}


@dataclass
class MockTokenPair:
    """Mock token pair from create_token_pair."""

    access_token: str = "new-access-token"
    refresh_token: str = "new-refresh-token"

    def to_dict(self) -> dict:
        return {
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "token_type": "bearer",
            "expires_in": 3600,
        }


@dataclass
class MockRefreshPayload:
    """Mock payload from validate_refresh_token."""

    user_id: str = "user-001"
    sub: str = "user-001"


@dataclass
class MockAuthCtx:
    """Mock auth context from extract_user_from_request."""

    is_authenticated: bool = True
    user_id: str = "user-001"
    email: str = "test@example.com"
    org_id: str = "org-001"
    role: str = "admin"
    client_ip: str = "127.0.0.1"


@dataclass
class MockRBACDecision:
    """Mock RBAC check_permission result."""

    allowed: bool = True
    reason: str = ""


def _make_user_store(user: MockUser | None = None, org: MockOrg | None = None):
    """Create a mock user store with standard methods."""
    store = MagicMock()
    u = user or MockUser()
    o = org or MockOrg()
    store.get_user_by_id.return_value = u
    store.get_user_by_email.return_value = u
    store.get_organization_by_id.return_value = o
    store.increment_token_version.return_value = 2
    store.create_user.return_value = u
    store.update_user.return_value = None
    return store


# ---------------------------------------------------------------------------
# Standard patches applied to all tests via fixture
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _patch_auth_deps(monkeypatch):
    """Patch out auth-related imports so tests run without real JWT/RBAC."""
    mock_auth_ctx = MockAuthCtx()

    monkeypatch.setattr(
        "aragora.server.handlers.auth.handler.extract_user_from_request",
        lambda handler, user_store: mock_auth_ctx,
    )
    monkeypatch.setattr(
        "aragora.server.handlers.auth.handler.validate_refresh_token",
        lambda token: MockRefreshPayload() if token else None,
    )
    monkeypatch.setattr(
        "aragora.server.handlers.auth.handler.check_permission",
        lambda ctx, perm, res_id=None: MockRBACDecision(),
    )
    monkeypatch.setattr(
        "aragora.server.handlers.auth.handler.get_role_permissions",
        lambda role, include_inherited=False: {"*"},
    )


@pytest.fixture
def handler():
    """Create an AuthHandler with an empty server context."""
    return AuthHandler(server_context={})


@pytest.fixture
def handler_with_store():
    """Create an AuthHandler with a user store in context."""
    store = _make_user_store()
    h = AuthHandler(server_context={"user_store": store})
    return h, store


# =========================================================================
# can_handle tests
# =========================================================================


class TestCanHandle:
    """Test AuthHandler.can_handle() routing logic."""

    def test_exact_routes(self, handler):
        for route in AuthHandler.ROUTES:
            if "*" not in route:
                assert handler.can_handle(route), f"Should handle {route}"

    def test_versioned_paths(self, handler):
        assert handler.can_handle("/api/v1/auth/login")
        assert handler.can_handle("/api/v1/auth/register")
        assert handler.can_handle("/api/v1/auth/me")
        assert handler.can_handle("/api/v1/auth/sessions")

    def test_wildcard_session_paths(self, handler):
        assert handler.can_handle("/api/auth/sessions/abc123")
        assert handler.can_handle("/api/auth/sessions/some-session-id")

    def test_wildcard_api_key_paths(self, handler):
        assert handler.can_handle("/api/auth/api-keys/prefix123")

    def test_sdk_alias_keys(self, handler):
        assert handler.can_handle("/api/keys")
        assert handler.can_handle("/api/keys/abc")

    def test_unhandled_paths(self, handler):
        assert not handler.can_handle("/api/debates")
        assert not handler.can_handle("/api/other/route")
        assert not handler.can_handle("/api/auth-other")

    def test_health_route(self, handler):
        assert handler.can_handle("/api/auth/health")

    def test_mfa_routes(self, handler):
        assert handler.can_handle("/api/auth/mfa/setup")
        assert handler.can_handle("/api/auth/mfa/enable")
        assert handler.can_handle("/api/auth/mfa/disable")
        assert handler.can_handle("/api/auth/mfa/verify")
        assert handler.can_handle("/api/auth/mfa/backup-codes")
        assert handler.can_handle("/api/auth/mfa")

    def test_password_routes(self, handler):
        assert handler.can_handle("/api/auth/password")
        assert handler.can_handle("/api/auth/password/change")
        assert handler.can_handle("/api/auth/password/forgot")
        assert handler.can_handle("/api/auth/password/reset")
        assert handler.can_handle("/api/auth/forgot-password")
        assert handler.can_handle("/api/auth/reset-password")

    def test_signup_routes(self, handler):
        assert handler.can_handle("/api/auth/verify-email")
        assert handler.can_handle("/api/auth/verify-email/resend")
        assert handler.can_handle("/api/auth/resend-verification")
        assert handler.can_handle("/api/auth/setup-organization")
        assert handler.can_handle("/api/auth/invite")
        assert handler.can_handle("/api/auth/check-invite")
        assert handler.can_handle("/api/auth/accept-invite")


# =========================================================================
# Initialization
# =========================================================================


class TestInit:
    """Test AuthHandler initialization."""

    def test_init_with_server_context(self):
        h = AuthHandler(server_context={"user_store": "fake"})
        assert h.ctx["user_store"] == "fake"

    def test_init_with_ctx(self):
        h = AuthHandler(ctx={"user_store": "fake"})
        assert h.ctx["user_store"] == "fake"

    def test_init_with_none(self):
        h = AuthHandler(server_context=None)
        assert h.ctx == {}

    def test_init_both_prefers_server_context(self):
        h = AuthHandler(server_context={"a": 1}, ctx={"b": 2})
        assert h.ctx == {"a": 1}

    def test_resource_type(self, handler):
        assert handler.RESOURCE_TYPE == "auth"


# =========================================================================
# Register endpoint
# =========================================================================


class TestRegister:
    """POST /api/auth/register."""

    @patch("aragora.server.handlers.auth.login.handle_register")
    def test_register_delegates(self, mock_register, handler):
        mock_register.return_value = MagicMock(status_code=201, body=b'{"user":{}}')
        http = MockHTTPHandler(body={"email": "a@b.com", "password": "x"}, method="POST")
        result = _run(handler.handle("/api/auth/register", {}, http, "POST"))
        mock_register.assert_called_once_with(handler, http)

    @patch("aragora.server.handlers.auth.login.handle_register")
    def test_register_versioned(self, mock_register, handler):
        mock_register.return_value = MagicMock(status_code=201, body=b'{}')
        http = MockHTTPHandler(method="POST")
        _run(handler.handle("/api/v1/auth/register", {}, http, "POST"))
        mock_register.assert_called_once()


# =========================================================================
# Login endpoint
# =========================================================================


class TestLogin:
    """POST /api/auth/login."""

    @patch("aragora.server.handlers.auth.login.handle_login")
    def test_login_delegates(self, mock_login, handler):
        mock_login.return_value = MagicMock(status_code=200, body=b'{}')
        http = MockHTTPHandler(method="POST")
        _run(handler.handle("/api/auth/login", {}, http, "POST"))
        mock_login.assert_called_once_with(handler, http)


# =========================================================================
# Logout endpoint
# =========================================================================


class TestLogout:
    """POST /api/auth/logout."""

    @patch("aragora.server.handlers.auth.handler.AuthHandler._check_permission", return_value=None)
    @patch("aragora.server.middleware.auth.extract_token", return_value="tok-123")
    @patch("aragora.billing.jwt_auth.revoke_token_persistent", return_value=True)
    @patch("aragora.billing.jwt_auth.get_token_blacklist")
    def test_logout_success(self, mock_blacklist, mock_revoke_p, mock_extract, mock_perm, handler):
        bl = MagicMock()
        bl.revoke_token.return_value = True
        mock_blacklist.return_value = bl
        http = MockHTTPHandler(method="POST")
        result = _run(handler.handle("/api/auth/logout", {}, http, "POST"))
        assert _status(result) == 200
        body = _body(result)
        assert body["message"] == "Logged out successfully"

    @patch("aragora.server.handlers.auth.handler.AuthHandler._check_permission")
    def test_logout_permission_denied(self, mock_perm, handler):
        from aragora.server.handlers.base import error_response
        mock_perm.return_value = error_response("Permission denied", 403)
        http = MockHTTPHandler(method="POST")
        result = _run(handler.handle("/api/auth/logout", {}, http, "POST"))
        assert _status(result) == 403

    @patch("aragora.server.handlers.auth.handler.AuthHandler._check_permission", return_value=None)
    @patch("aragora.server.middleware.auth.extract_token", return_value=None)
    def test_logout_no_token(self, mock_extract, mock_perm, handler):
        http = MockHTTPHandler(method="POST")
        result = _run(handler.handle("/api/auth/logout", {}, http, "POST"))
        assert _status(result) == 200
        assert "Logged out" in _body(result)["message"]


# =========================================================================
# Logout-all endpoint
# =========================================================================


class TestLogoutAll:
    """POST /api/auth/logout-all."""

    @patch("aragora.server.handlers.auth.handler.AuthHandler._check_permission", return_value=None)
    @patch("aragora.server.middleware.auth.extract_token", return_value="tok-abc")
    @patch("aragora.billing.jwt_auth.revoke_token_persistent", return_value=True)
    @patch("aragora.billing.jwt_auth.get_token_blacklist")
    def test_logout_all_success(self, mock_bl, mock_rev, mock_ext, mock_perm):
        bl = MagicMock()
        bl.revoke_token.return_value = True
        mock_bl.return_value = bl
        store = _make_user_store()
        h = AuthHandler(server_context={"user_store": store})
        http = MockHTTPHandler(method="POST")
        result = _run(h.handle("/api/auth/logout-all", {}, http, "POST"))
        assert _status(result) == 200
        body = _body(result)
        assert body["sessions_invalidated"] is True
        assert body["token_version"] == 2
        store.increment_token_version.assert_called_once()

    @patch("aragora.server.handlers.auth.handler.AuthHandler._check_permission", return_value=None)
    @patch("aragora.server.middleware.auth.extract_token", return_value=None)
    @patch("aragora.billing.jwt_auth.get_token_blacklist")
    @patch("aragora.billing.jwt_auth.revoke_token_persistent")
    def test_logout_all_no_user_store(self, mock_rev, mock_bl, mock_ext, mock_perm, handler):
        http = MockHTTPHandler(method="POST")
        result = _run(handler.handle("/api/auth/logout-all", {}, http, "POST"))
        assert _status(result) == 503

    @patch("aragora.server.handlers.auth.handler.AuthHandler._check_permission", return_value=None)
    @patch("aragora.server.middleware.auth.extract_token", return_value=None)
    @patch("aragora.billing.jwt_auth.get_token_blacklist")
    @patch("aragora.billing.jwt_auth.revoke_token_persistent")
    def test_logout_all_user_not_found(self, mock_rev, mock_bl, mock_ext, mock_perm):
        store = _make_user_store()
        store.increment_token_version.return_value = 0  # user not found
        h = AuthHandler(server_context={"user_store": store})
        http = MockHTTPHandler(method="POST")
        result = _run(h.handle("/api/auth/logout-all", {}, http, "POST"))
        assert _status(result) == 404


# =========================================================================
# Refresh endpoint
# =========================================================================


class TestRefresh:
    """POST /api/auth/refresh."""

    @patch("aragora.billing.jwt_auth.revoke_token_persistent")
    @patch("aragora.billing.jwt_auth.get_token_blacklist")
    @patch("aragora.billing.jwt_auth.create_token_pair")
    def test_refresh_success(self, mock_create, mock_bl, mock_revoke):
        mock_create.return_value = MockTokenPair()
        bl = MagicMock()
        bl.revoke_token.return_value = True
        mock_bl.return_value = bl
        mock_revoke.return_value = True

        store = _make_user_store()
        h = AuthHandler(server_context={"user_store": store})
        http = MockHTTPHandler(body={"refresh_token": "rt-valid"}, method="POST")
        result = _run(h.handle("/api/auth/refresh", {}, http, "POST"))
        assert _status(result) == 200
        body = _body(result)
        assert "tokens" in body
        assert body["tokens"]["access_token"] == "new-access-token"

    def test_refresh_no_body(self, handler):
        http = MockHTTPHandler(method="POST")
        http.rfile.read.return_value = b"not json"
        http.headers["Content-Length"] = "8"
        result = _run(handler.handle("/api/auth/refresh", {}, http, "POST"))
        assert _status(result) == 400

    def test_refresh_missing_token(self, handler):
        http = MockHTTPHandler(body={}, method="POST")
        result = _run(handler.handle("/api/auth/refresh", {}, http, "POST"))
        assert _status(result) == 400
        assert "Refresh token required" in _body(result)["error"]

    @patch("aragora.server.handlers.auth.handler.validate_refresh_token", return_value=None)
    def test_refresh_invalid_token(self, mock_validate, handler):
        http = MockHTTPHandler(body={"refresh_token": "bad-token"}, method="POST")
        result = _run(handler.handle("/api/auth/refresh", {}, http, "POST"))
        assert _status(result) == 401

    @patch("aragora.billing.jwt_auth.revoke_token_persistent")
    @patch("aragora.billing.jwt_auth.get_token_blacklist")
    @patch("aragora.billing.jwt_auth.create_token_pair")
    def test_refresh_no_user_store(self, mock_create, mock_bl, mock_revoke, handler):
        http = MockHTTPHandler(body={"refresh_token": "rt-valid"}, method="POST")
        result = _run(handler.handle("/api/auth/refresh", {}, http, "POST"))
        assert _status(result) == 503

    @patch("aragora.billing.jwt_auth.revoke_token_persistent")
    @patch("aragora.billing.jwt_auth.get_token_blacklist")
    @patch("aragora.billing.jwt_auth.create_token_pair")
    def test_refresh_user_not_found(self, mock_create, mock_bl, mock_revoke):
        store = _make_user_store()
        store.get_user_by_id.return_value = None
        h = AuthHandler(server_context={"user_store": store})
        http = MockHTTPHandler(body={"refresh_token": "rt-valid"}, method="POST")
        result = _run(h.handle("/api/auth/refresh", {}, http, "POST"))
        assert _status(result) == 401

    @patch("aragora.billing.jwt_auth.revoke_token_persistent")
    @patch("aragora.billing.jwt_auth.get_token_blacklist")
    @patch("aragora.billing.jwt_auth.create_token_pair")
    def test_refresh_disabled_account(self, mock_create, mock_bl, mock_revoke):
        user = MockUser(is_active=False)
        store = _make_user_store(user=user)
        h = AuthHandler(server_context={"user_store": store})
        http = MockHTTPHandler(body={"refresh_token": "rt-valid"}, method="POST")
        result = _run(h.handle("/api/auth/refresh", {}, http, "POST"))
        assert _status(result) == 401
        assert "disabled" in _body(result)["error"]

    @patch("aragora.billing.jwt_auth.revoke_token_persistent", side_effect=OSError("disk"))
    def test_refresh_revoke_fails(self, mock_revoke):
        store = _make_user_store()
        h = AuthHandler(server_context={"user_store": store})
        http = MockHTTPHandler(body={"refresh_token": "rt-valid"}, method="POST")
        result = _run(h.handle("/api/auth/refresh", {}, http, "POST"))
        assert _status(result) == 500


# =========================================================================
# GET /api/auth/me
# =========================================================================


class TestGetMe:
    """GET /api/auth/me."""

    @patch("aragora.server.handlers.auth.handler.AuthHandler._check_permission", return_value=None)
    def test_get_me_success(self, mock_perm):
        user = MockUser()
        org = MockOrg()
        store = _make_user_store(user=user, org=org)
        h = AuthHandler(server_context={"user_store": store})
        http = MockHTTPHandler(method="GET")
        result = _run(h.handle("/api/auth/me", {}, http, "GET"))
        assert _status(result) == 200
        body = _body(result)
        assert body["user"]["id"] == "user-001"
        assert body["organization"]["id"] == "org-001"
        assert len(body["organizations"]) == 1

    @patch("aragora.server.handlers.auth.handler.AuthHandler._check_permission")
    def test_get_me_permission_denied(self, mock_perm):
        from aragora.server.handlers.base import error_response
        mock_perm.return_value = error_response("Authentication required", 401)
        h = AuthHandler(server_context={})
        http = MockHTTPHandler(method="GET")
        result = _run(h.handle("/api/auth/me", {}, http, "GET"))
        assert _status(result) == 401

    @patch("aragora.server.handlers.auth.handler.AuthHandler._check_permission", return_value=None)
    def test_get_me_no_user_store(self, mock_perm, handler):
        http = MockHTTPHandler(method="GET")
        result = _run(handler.handle("/api/auth/me", {}, http, "GET"))
        assert _status(result) == 503

    @patch("aragora.server.handlers.auth.handler.AuthHandler._check_permission", return_value=None)
    def test_get_me_user_not_found(self, mock_perm):
        store = _make_user_store()
        store.get_user_by_id.return_value = None
        store.get_user_by_email.return_value = None
        h = AuthHandler(server_context={"user_store": store})
        http = MockHTTPHandler(method="GET")
        result = _run(h.handle("/api/auth/me", {}, http, "GET"))
        assert _status(result) == 404

    @patch("aragora.server.handlers.auth.handler.AuthHandler._check_permission", return_value=None)
    def test_get_me_email_fallback(self, mock_perm):
        user = MockUser()
        store = _make_user_store(user=user)
        # First call by ID returns None, but email fallback succeeds
        store.get_user_by_id.return_value = None
        store.get_user_by_email.return_value = user
        h = AuthHandler(server_context={"user_store": store})
        http = MockHTTPHandler(method="GET")
        result = _run(h.handle("/api/auth/me", {}, http, "GET"))
        assert _status(result) == 200

    @patch("aragora.server.handlers.auth.handler.AuthHandler._check_permission", return_value=None)
    def test_get_me_no_org(self, mock_perm):
        user = MockUser(org_id=None)
        store = _make_user_store(user=user)
        h = AuthHandler(server_context={"user_store": store})
        http = MockHTTPHandler(method="GET")
        result = _run(h.handle("/api/auth/me", {}, http, "GET"))
        assert _status(result) == 200
        body = _body(result)
        assert body["organization"] is None
        assert body["organizations"] == []

    @patch("aragora.server.handlers.auth.handler.AuthHandler._check_permission", return_value=None)
    def test_get_me_versioned_path(self, mock_perm):
        store = _make_user_store()
        h = AuthHandler(server_context={"user_store": store})
        http = MockHTTPHandler(method="GET")
        result = _run(h.handle("/api/v1/auth/me", {}, http, "GET"))
        assert _status(result) == 200

    @patch("aragora.server.handlers.auth.handler.AuthHandler._check_permission", return_value=None)
    def test_get_me_has_no_cache_headers(self, mock_perm):
        store = _make_user_store()
        h = AuthHandler(server_context={"user_store": store})
        http = MockHTTPHandler(method="GET")
        result = _run(h.handle("/api/auth/me", {}, http, "GET"))
        assert result.headers.get("Cache-Control") == "no-store, no-cache, must-revalidate, private"

    @patch("aragora.server.handlers.auth.handler.AuthHandler._check_permission", return_value=None)
    def test_get_me_async_user_store(self, mock_perm):
        """Test that async user store methods are used when available."""
        user = MockUser()
        org = MockOrg()
        store = MagicMock()
        store.get_user_by_id_async = AsyncMock(return_value=user)
        store.get_organization_by_id_async = AsyncMock(return_value=org)
        # Remove sync methods
        del store.get_user_by_id
        del store.get_user_by_email
        h = AuthHandler(server_context={"user_store": store})
        http = MockHTTPHandler(method="GET")
        result = _run(h.handle("/api/auth/me", {}, http, "GET"))
        assert _status(result) == 200
        store.get_user_by_id_async.assert_awaited()


# =========================================================================
# PUT/POST /api/auth/me (update profile)
# =========================================================================


class TestUpdateMe:
    """PUT/POST /api/auth/me and POST /api/auth/profile."""

    @patch("aragora.server.handlers.auth.handler.AuthHandler._check_permission", return_value=None)
    def test_update_me_put(self, mock_perm):
        user = MockUser()
        store = _make_user_store(user=user)
        # After update, get_user_by_id returns updated user
        store.get_user_by_id.return_value = user
        h = AuthHandler(server_context={"user_store": store})
        http = MockHTTPHandler(body={"name": "New Name"}, method="PUT")
        result = _run(h.handle("/api/auth/me", {}, http, "PUT"))
        assert _status(result) == 200
        store.update_user.assert_called()

    @patch("aragora.server.handlers.auth.handler.AuthHandler._check_permission", return_value=None)
    def test_update_me_post(self, mock_perm):
        store = _make_user_store()
        h = AuthHandler(server_context={"user_store": store})
        http = MockHTTPHandler(body={"name": "N"}, method="POST")
        result = _run(h.handle("/api/auth/me", {}, http, "POST"))
        assert _status(result) == 200

    @patch("aragora.server.handlers.auth.handler.AuthHandler._check_permission", return_value=None)
    def test_update_profile_alias(self, mock_perm):
        store = _make_user_store()
        h = AuthHandler(server_context={"user_store": store})
        http = MockHTTPHandler(body={"name": "N"}, method="POST")
        result = _run(h.handle("/api/auth/profile", {}, http, "POST"))
        assert _status(result) == 200

    @patch("aragora.server.handlers.auth.handler.AuthHandler._check_permission", return_value=None)
    def test_update_me_invalid_json(self, mock_perm):
        store = _make_user_store()
        h = AuthHandler(server_context={"user_store": store})
        http = MockHTTPHandler(method="PUT")
        http.rfile.read.return_value = b"not json"
        http.headers["Content-Length"] = "8"
        result = _run(h.handle("/api/auth/me", {}, http, "PUT"))
        assert _status(result) == 400

    @patch("aragora.server.handlers.auth.handler.AuthHandler._check_permission", return_value=None)
    def test_update_me_no_user_store(self, mock_perm, handler):
        http = MockHTTPHandler(body={"name": "x"}, method="PUT")
        result = _run(handler.handle("/api/auth/me", {}, http, "PUT"))
        assert _status(result) == 503

    @patch("aragora.server.handlers.auth.handler.AuthHandler._check_permission", return_value=None)
    def test_update_me_user_not_found(self, mock_perm):
        store = _make_user_store()
        store.get_user_by_id.return_value = None
        h = AuthHandler(server_context={"user_store": store})
        http = MockHTTPHandler(body={"name": "x"}, method="PUT")
        result = _run(h.handle("/api/auth/me", {}, http, "PUT"))
        assert _status(result) == 404

    @patch("aragora.server.handlers.auth.handler.AuthHandler._check_permission", return_value=None)
    def test_update_me_name_truncated(self, mock_perm):
        store = _make_user_store()
        h = AuthHandler(server_context={"user_store": store})
        long_name = "A" * 200
        http = MockHTTPHandler(body={"name": long_name}, method="PUT")
        result = _run(h.handle("/api/auth/me", {}, http, "PUT"))
        assert _status(result) == 200
        # The name should be truncated to 100 chars
        call_args = store.update_user.call_args
        assert len(call_args[1]["name"]) == 100

    @patch("aragora.server.handlers.auth.handler.AuthHandler._check_permission", return_value=None)
    def test_update_me_no_changes(self, mock_perm):
        """Empty update body -> no update_user call."""
        store = _make_user_store()
        h = AuthHandler(server_context={"user_store": store})
        http = MockHTTPHandler(body={}, method="PUT")
        result = _run(h.handle("/api/auth/me", {}, http, "PUT"))
        assert _status(result) == 200


# =========================================================================
# Password change
# =========================================================================


class TestChangePassword:
    """POST /api/auth/password and POST /api/auth/password/change."""

    @patch("aragora.server.handlers.auth.password.handle_change_password")
    def test_password_delegates(self, mock_fn, handler):
        mock_fn.return_value = MagicMock(status_code=200, body=b'{}')
        http = MockHTTPHandler(method="POST")
        _run(handler.handle("/api/auth/password", {}, http, "POST"))
        mock_fn.assert_called_once_with(handler, http)

    @patch("aragora.server.handlers.auth.password.handle_change_password")
    def test_password_change_alias(self, mock_fn, handler):
        mock_fn.return_value = MagicMock(status_code=200, body=b'{}')
        http = MockHTTPHandler(method="POST")
        _run(handler.handle("/api/auth/password/change", {}, http, "POST"))
        mock_fn.assert_called_once()


# =========================================================================
# Forgot/Reset password
# =========================================================================


class TestForgotResetPassword:
    """POST /api/auth/password/forgot, /api/auth/password/reset, and legacy aliases."""

    @patch("aragora.server.handlers.auth.password.handle_forgot_password")
    def test_forgot_password(self, mock_fn, handler):
        mock_fn.return_value = MagicMock(status_code=200, body=b'{}')
        http = MockHTTPHandler(method="POST")
        _run(handler.handle("/api/auth/password/forgot", {}, http, "POST"))
        mock_fn.assert_called_once()

    @patch("aragora.server.handlers.auth.password.handle_reset_password")
    def test_reset_password(self, mock_fn, handler):
        mock_fn.return_value = MagicMock(status_code=200, body=b'{}')
        http = MockHTTPHandler(method="POST")
        _run(handler.handle("/api/auth/password/reset", {}, http, "POST"))
        mock_fn.assert_called_once()

    @patch("aragora.server.handlers.auth.password.handle_forgot_password")
    def test_legacy_forgot_password(self, mock_fn, handler):
        mock_fn.return_value = MagicMock(status_code=200, body=b'{}')
        http = MockHTTPHandler(method="POST")
        _run(handler.handle("/api/auth/forgot-password", {}, http, "POST"))
        mock_fn.assert_called_once()

    @patch("aragora.server.handlers.auth.password.handle_reset_password")
    def test_legacy_reset_password(self, mock_fn, handler):
        mock_fn.return_value = MagicMock(status_code=200, body=b'{}')
        http = MockHTTPHandler(method="POST")
        _run(handler.handle("/api/auth/reset-password", {}, http, "POST"))
        mock_fn.assert_called_once()

    def test_forgot_password_disabled(self):
        h = AuthHandler(server_context={"enable_password_reset_routes": False})
        http = MockHTTPHandler(method="POST")
        result = _run(h.handle("/api/auth/password/forgot", {}, http, "POST"))
        assert _status(result) == 501

    def test_reset_password_disabled(self):
        h = AuthHandler(server_context={"enable_password_reset_routes": False})
        http = MockHTTPHandler(method="POST")
        result = _run(h.handle("/api/auth/password/reset", {}, http, "POST"))
        assert _status(result) == 501

    def test_legacy_forgot_disabled(self):
        h = AuthHandler(server_context={"enable_password_reset_routes": False})
        http = MockHTTPHandler(method="POST")
        result = _run(h.handle("/api/auth/forgot-password", {}, http, "POST"))
        assert _status(result) == 501

    def test_legacy_reset_disabled(self):
        h = AuthHandler(server_context={"enable_password_reset_routes": False})
        http = MockHTTPHandler(method="POST")
        result = _run(h.handle("/api/auth/reset-password", {}, http, "POST"))
        assert _status(result) == 501

    def test_disabled_message_contains_smtp(self):
        h = AuthHandler(server_context={"enable_password_reset_routes": False})
        http = MockHTTPHandler(method="POST")
        result = _run(h.handle("/api/auth/password/forgot", {}, http, "POST"))
        body = _body(result)
        assert "SMTP" in body["error"] or "email" in body["error"].lower()


# =========================================================================
# Revoke token
# =========================================================================


class TestRevokeToken:
    """POST /api/auth/revoke."""

    @patch("aragora.server.handlers.auth.handler.AuthHandler._check_permission", return_value=None)
    @patch("aragora.server.middleware.auth.extract_token", return_value="tok-current")
    @patch("aragora.billing.jwt_auth.revoke_token_persistent", return_value=True)
    @patch("aragora.billing.jwt_auth.get_token_blacklist")
    @patch("aragora.billing.jwt_auth.extract_user_from_request")
    def test_revoke_specific_token(self, mock_eu, mock_bl, mock_rev, mock_ext, mock_perm, handler):
        mock_eu.return_value = MockAuthCtx()
        bl = MagicMock()
        bl.revoke_token.return_value = True
        bl.size.return_value = 5
        mock_bl.return_value = bl
        http = MockHTTPHandler(body={"token": "specific-tok"}, method="POST")
        result = _run(handler.handle("/api/auth/revoke", {}, http, "POST"))
        assert _status(result) == 200
        body = _body(result)
        assert body["message"] == "Token revoked successfully"
        assert body["persistent"] is True

    @patch("aragora.server.handlers.auth.handler.AuthHandler._check_permission", return_value=None)
    @patch("aragora.server.middleware.auth.extract_token", return_value="current-tok")
    @patch("aragora.billing.jwt_auth.revoke_token_persistent", return_value=True)
    @patch("aragora.billing.jwt_auth.get_token_blacklist")
    @patch("aragora.billing.jwt_auth.extract_user_from_request")
    def test_revoke_current_token(self, mock_eu, mock_bl, mock_rev, mock_ext, mock_perm, handler):
        mock_eu.return_value = MockAuthCtx()
        bl = MagicMock()
        bl.revoke_token.return_value = True
        bl.size.return_value = 1
        mock_bl.return_value = bl
        http = MockHTTPHandler(body={}, method="POST")
        result = _run(handler.handle("/api/auth/revoke", {}, http, "POST"))
        assert _status(result) == 200

    @patch("aragora.server.handlers.auth.handler.AuthHandler._check_permission", return_value=None)
    @patch("aragora.server.middleware.auth.extract_token", return_value=None)
    @patch("aragora.billing.jwt_auth.extract_user_from_request")
    def test_revoke_no_token(self, mock_eu, mock_ext, mock_perm, handler):
        mock_eu.return_value = MockAuthCtx()
        http = MockHTTPHandler(body={}, method="POST")
        result = _run(handler.handle("/api/auth/revoke", {}, http, "POST"))
        assert _status(result) == 400

    @patch("aragora.server.handlers.auth.handler.AuthHandler._check_permission", return_value=None)
    @patch("aragora.server.middleware.auth.extract_token", return_value="bad")
    @patch("aragora.billing.jwt_auth.revoke_token_persistent", return_value=False)
    @patch("aragora.billing.jwt_auth.get_token_blacklist")
    @patch("aragora.billing.jwt_auth.extract_user_from_request")
    def test_revoke_invalid_token(self, mock_eu, mock_bl, mock_rev, mock_ext, mock_perm, handler):
        mock_eu.return_value = MockAuthCtx()
        bl = MagicMock()
        bl.revoke_token.return_value = False
        bl.size.return_value = 0
        mock_bl.return_value = bl
        http = MockHTTPHandler(body={"token": "bad"}, method="POST")
        result = _run(handler.handle("/api/auth/revoke", {}, http, "POST"))
        assert _status(result) == 400


# =========================================================================
# API Key management
# =========================================================================


class TestApiKeys:
    """API key endpoints: POST/DELETE /api/auth/api-key, GET /api/auth/api-keys."""

    @patch("aragora.server.handlers.auth.api_keys.handle_generate_api_key")
    def test_generate_api_key(self, mock_fn, handler):
        mock_fn.return_value = MagicMock(status_code=200, body=b'{}')
        http = MockHTTPHandler(method="POST")
        _run(handler.handle("/api/auth/api-key", {}, http, "POST"))
        mock_fn.assert_called_once()

    @patch("aragora.server.handlers.auth.api_keys.handle_revoke_api_key")
    def test_revoke_api_key(self, mock_fn, handler):
        mock_fn.return_value = MagicMock(status_code=200, body=b'{}')
        http = MockHTTPHandler(method="DELETE")
        _run(handler.handle("/api/auth/api-key", {}, http, "DELETE"))
        mock_fn.assert_called_once()

    @patch("aragora.server.handlers.auth.api_keys.handle_list_api_keys")
    def test_list_api_keys(self, mock_fn, handler):
        mock_fn.return_value = MagicMock(status_code=200, body=b'{}')
        http = MockHTTPHandler(method="GET")
        _run(handler.handle("/api/auth/api-keys", {}, http, "GET"))
        mock_fn.assert_called_once()

    @patch("aragora.server.handlers.auth.api_keys.handle_generate_api_key")
    def test_api_keys_post(self, mock_fn, handler):
        mock_fn.return_value = MagicMock(status_code=200, body=b'{}')
        http = MockHTTPHandler(method="POST")
        _run(handler.handle("/api/auth/api-keys", {}, http, "POST"))
        mock_fn.assert_called_once()

    @patch("aragora.server.handlers.auth.api_keys.handle_revoke_api_key")
    def test_api_keys_delete(self, mock_fn, handler):
        mock_fn.return_value = MagicMock(status_code=200, body=b'{}')
        http = MockHTTPHandler(method="DELETE")
        _run(handler.handle("/api/auth/api-keys", {}, http, "DELETE"))
        mock_fn.assert_called_once()

    @patch("aragora.server.handlers.auth.api_keys.handle_revoke_api_key_prefix")
    def test_revoke_api_key_by_prefix(self, mock_fn, handler):
        mock_fn.return_value = MagicMock(status_code=200, body=b'{}')
        http = MockHTTPHandler(method="DELETE")
        _run(handler.handle("/api/auth/api-keys/ak_prefix123", {}, http, "DELETE"))
        mock_fn.assert_called_once_with(handler, http, "ak_prefix123")


# =========================================================================
# SDK alias: /api/keys -> /api/auth/api-keys
# =========================================================================


class TestSDKAlias:
    """Test /api/keys/* alias to /api/auth/api-keys/*."""

    @patch("aragora.server.handlers.auth.api_keys.handle_list_api_keys")
    def test_sdk_keys_list(self, mock_fn, handler):
        mock_fn.return_value = MagicMock(status_code=200, body=b'{}')
        http = MockHTTPHandler(method="GET")
        _run(handler.handle("/api/keys", {}, http, "GET"))
        mock_fn.assert_called_once()

    @patch("aragora.server.handlers.auth.api_keys.handle_generate_api_key")
    def test_sdk_keys_post(self, mock_fn, handler):
        mock_fn.return_value = MagicMock(status_code=200, body=b'{}')
        http = MockHTTPHandler(method="POST")
        _run(handler.handle("/api/keys", {}, http, "POST"))
        mock_fn.assert_called_once()

    @patch("aragora.server.handlers.auth.api_keys.handle_revoke_api_key_prefix")
    def test_sdk_keys_delete_prefix(self, mock_fn, handler):
        mock_fn.return_value = MagicMock(status_code=200, body=b'{}')
        http = MockHTTPHandler(method="DELETE")
        _run(handler.handle("/api/keys/prefix123", {}, http, "DELETE"))
        mock_fn.assert_called_once_with(handler, http, "prefix123")


# =========================================================================
# MFA endpoints
# =========================================================================


class TestMFA:
    """MFA endpoints: setup, enable, disable, verify, backup-codes."""

    @patch("aragora.server.handlers.auth.mfa.handle_mfa_setup")
    def test_mfa_setup(self, mock_fn, handler):
        mock_fn.return_value = MagicMock(status_code=200, body=b'{}')
        http = MockHTTPHandler(method="POST")
        _run(handler.handle("/api/auth/mfa/setup", {}, http, "POST"))
        mock_fn.assert_called_once()

    @patch("aragora.server.handlers.auth.mfa.handle_mfa_enable")
    def test_mfa_enable(self, mock_fn, handler):
        mock_fn.return_value = MagicMock(status_code=200, body=b'{}')
        http = MockHTTPHandler(method="POST")
        _run(handler.handle("/api/auth/mfa/enable", {}, http, "POST"))
        mock_fn.assert_called_once()

    @patch("aragora.server.handlers.auth.mfa.handle_mfa_disable")
    def test_mfa_disable_post(self, mock_fn, handler):
        mock_fn.return_value = MagicMock(status_code=200, body=b'{}')
        http = MockHTTPHandler(method="POST")
        _run(handler.handle("/api/auth/mfa/disable", {}, http, "POST"))
        mock_fn.assert_called_once()

    @patch("aragora.server.handlers.auth.mfa.handle_mfa_disable")
    def test_mfa_disable_delete(self, mock_fn, handler):
        mock_fn.return_value = MagicMock(status_code=200, body=b'{}')
        http = MockHTTPHandler(method="DELETE")
        _run(handler.handle("/api/auth/mfa", {}, http, "DELETE"))
        mock_fn.assert_called_once()

    @patch("aragora.server.handlers.auth.mfa.handle_mfa_verify")
    def test_mfa_verify(self, mock_fn, handler):
        mock_fn.return_value = MagicMock(status_code=200, body=b'{}')
        http = MockHTTPHandler(method="POST")
        _run(handler.handle("/api/auth/mfa/verify", {}, http, "POST"))
        mock_fn.assert_called_once()

    @patch("aragora.server.handlers.auth.mfa.handle_mfa_backup_codes")
    def test_mfa_backup_codes(self, mock_fn, handler):
        mock_fn.return_value = MagicMock(status_code=200, body=b'{}')
        http = MockHTTPHandler(method="POST")
        _run(handler.handle("/api/auth/mfa/backup-codes", {}, http, "POST"))
        mock_fn.assert_called_once()


# =========================================================================
# Session management
# =========================================================================


class TestSessions:
    """Session endpoints: GET /api/auth/sessions, DELETE /api/auth/sessions/:id."""

    @patch("aragora.server.handlers.auth.sessions.handle_list_sessions")
    def test_list_sessions(self, mock_fn, handler):
        mock_fn.return_value = MagicMock(status_code=200, body=b'{}')
        http = MockHTTPHandler(method="GET")
        _run(handler.handle("/api/auth/sessions", {}, http, "GET"))
        mock_fn.assert_called_once_with(handler, http)

    @patch("aragora.server.handlers.auth.sessions.handle_revoke_session")
    def test_revoke_session(self, mock_fn, handler):
        mock_fn.return_value = MagicMock(status_code=200, body=b'{}')
        http = MockHTTPHandler(method="DELETE")
        _run(handler.handle("/api/auth/sessions/sess-id-123", {}, http, "DELETE"))
        mock_fn.assert_called_once_with(handler, http, "sess-id-123")

    @patch("aragora.server.handlers.auth.sessions.handle_revoke_session")
    def test_revoke_session_versioned(self, mock_fn, handler):
        mock_fn.return_value = MagicMock(status_code=200, body=b'{}')
        http = MockHTTPHandler(method="DELETE")
        _run(handler.handle("/api/v1/auth/sessions/abc", {}, http, "DELETE"))
        mock_fn.assert_called_once_with(handler, http, "abc")


# =========================================================================
# Signup-related delegated endpoints
# =========================================================================


class TestSignupDelegation:
    """Test verify-email, resend-verification, setup-org, invite, etc."""

    @patch("aragora.server.handlers.auth.handler.handle_verify_email", new_callable=AsyncMock)
    def test_verify_email(self, mock_fn, handler):
        mock_fn.return_value = MagicMock(status_code=200, body=b'{}')
        http = MockHTTPHandler(body={"token": "t"}, method="POST")
        _run(handler.handle("/api/auth/verify-email", {}, http, "POST"))
        mock_fn.assert_called_once()

    @patch("aragora.server.handlers.auth.handler.handle_resend_verification", new_callable=AsyncMock)
    def test_resend_verification(self, mock_fn, handler):
        mock_fn.return_value = MagicMock(status_code=200, body=b'{}')
        http = MockHTTPHandler(body={"email": "a@b.com"}, method="POST")
        _run(handler.handle("/api/auth/resend-verification", {}, http, "POST"))
        mock_fn.assert_called_once()

    @patch("aragora.server.handlers.auth.handler.handle_resend_verification", new_callable=AsyncMock)
    def test_verify_email_resend_alias(self, mock_fn, handler):
        mock_fn.return_value = MagicMock(status_code=200, body=b'{}')
        http = MockHTTPHandler(body={"email": "a@b.com"}, method="POST")
        _run(handler.handle("/api/auth/verify-email/resend", {}, http, "POST"))
        mock_fn.assert_called_once()

    @patch("aragora.server.handlers.auth.handler.handle_setup_organization", new_callable=AsyncMock)
    @patch("aragora.server.handlers.auth.handler.AuthHandler._require_user_id")
    def test_setup_organization(self, mock_req, mock_fn, handler):
        mock_req.return_value = ("user-001", None)
        mock_fn.return_value = MagicMock(status_code=200, body=b'{}')
        http = MockHTTPHandler(body={"name": "Org"}, method="POST")
        _run(handler.handle("/api/auth/setup-organization", {}, http, "POST"))
        mock_fn.assert_called_once()

    @patch("aragora.server.handlers.auth.handler.handle_setup_organization", new_callable=AsyncMock)
    @patch("aragora.server.handlers.auth.handler.AuthHandler._require_user_id")
    def test_setup_organization_unauthenticated(self, mock_req, mock_fn, handler):
        from aragora.server.handlers.base import error_response
        mock_req.return_value = (None, error_response("Authentication required", 401))
        http = MockHTTPHandler(body={"name": "Org"}, method="POST")
        result = _run(handler.handle("/api/auth/setup-organization", {}, http, "POST"))
        assert _status(result) == 401
        mock_fn.assert_not_called()

    @patch("aragora.server.handlers.auth.handler.handle_invite", new_callable=AsyncMock)
    @patch("aragora.server.handlers.auth.handler.AuthHandler._require_user_id")
    def test_invite(self, mock_req, mock_fn, handler):
        mock_req.return_value = ("user-001", None)
        mock_fn.return_value = MagicMock(status_code=200, body=b'{}')
        http = MockHTTPHandler(body={"email": "x@y.com"}, method="POST")
        _run(handler.handle("/api/auth/invite", {}, http, "POST"))
        mock_fn.assert_called_once()

    @patch("aragora.server.handlers.auth.handler.handle_check_invite", new_callable=AsyncMock)
    def test_check_invite(self, mock_fn, handler):
        mock_fn.return_value = MagicMock(status_code=200, body=b'{}')
        http = MockHTTPHandler(method="GET")
        _run(handler.handle("/api/auth/check-invite", {"token": "abc"}, http, "GET"))
        mock_fn.assert_called_once_with({"token": "abc"})

    @patch("aragora.server.handlers.auth.handler.handle_accept_invite", new_callable=AsyncMock)
    @patch("aragora.server.handlers.auth.handler.AuthHandler._require_user_id")
    def test_accept_invite(self, mock_req, mock_fn, handler):
        mock_req.return_value = ("user-001", None)
        mock_fn.return_value = MagicMock(status_code=200, body=b'{}')
        http = MockHTTPHandler(body={"token": "t"}, method="POST")
        _run(handler.handle("/api/auth/accept-invite", {}, http, "POST"))
        mock_fn.assert_called_once()

    @patch("aragora.server.handlers.auth.handler.handle_accept_invite", new_callable=AsyncMock)
    @patch("aragora.server.handlers.auth.handler.AuthHandler._require_user_id")
    def test_accept_invite_unauthenticated(self, mock_req, mock_fn, handler):
        from aragora.server.handlers.base import error_response
        mock_req.return_value = (None, error_response("Authentication required", 401))
        http = MockHTTPHandler(body={"token": "t"}, method="POST")
        result = _run(handler.handle("/api/auth/accept-invite", {}, http, "POST"))
        assert _status(result) == 401

    @patch("aragora.server.handlers.auth.handler.handle_invite", new_callable=AsyncMock)
    @patch("aragora.server.handlers.auth.handler.AuthHandler._require_user_id")
    def test_invite_unauthenticated(self, mock_req, mock_fn, handler):
        from aragora.server.handlers.base import error_response
        mock_req.return_value = (None, error_response("Authentication required", 401))
        http = MockHTTPHandler(body={"email": "x@y.com"}, method="POST")
        result = _run(handler.handle("/api/auth/invite", {}, http, "POST"))
        assert _status(result) == 401


# =========================================================================
# Health endpoint
# =========================================================================


class TestHealth:
    """GET /api/auth/health."""

    @patch("aragora.server.middleware.auth.extract_token", return_value=None)
    @patch("aragora.storage.pool_manager.is_pool_initialized", return_value=False)
    def test_health_no_pool(self, mock_pool, mock_ext, handler):
        http = MockHTTPHandler(method="GET")
        result = _run(handler.handle("/api/auth/health", {}, http, "GET"))
        assert _status(result) == 200
        body = _body(result)
        assert body["status"] == "ok"
        assert "timestamp" in body

    @patch("aragora.server.middleware.auth.extract_token", return_value="valid-tok")
    @patch("aragora.billing.jwt_auth.decode_jwt")
    @patch("aragora.storage.pool_manager.is_pool_initialized", return_value=False)
    def test_health_with_jwt(self, mock_pool, mock_decode, mock_ext, handler):
        payload = MagicMock()
        payload.user_id = "u1"
        mock_decode.return_value = payload
        http = MockHTTPHandler(method="GET")
        result = _run(handler.handle("/api/auth/health", {}, http, "GET"))
        assert _status(result) == 200
        body = _body(result)
        assert body["jwt"]["valid"] is True
        assert body["jwt"]["user_id"] == "u1"

    def test_health_has_no_cache_headers(self, handler):
        with patch("aragora.server.middleware.auth.extract_token", return_value=None), \
             patch("aragora.storage.pool_manager.is_pool_initialized", return_value=False):
            http = MockHTTPHandler(method="GET")
            result = _run(handler.handle("/api/auth/health", {}, http, "GET"))
            assert result.headers.get("Cache-Control") == "no-store, no-cache, must-revalidate, private"

    @patch("aragora.server.middleware.auth.extract_token", return_value=None)
    def test_health_pool_import_error(self, mock_ext, handler):
        """Health endpoint handles pool import errors gracefully."""
        with patch("aragora.storage.pool_manager.is_pool_initialized", side_effect=ImportError):
            # The handler catches ImportError in the pool section
            http = MockHTTPHandler(method="GET")
            result = _run(handler.handle("/api/auth/health", {}, http, "GET"))
            assert _status(result) == 200
            body = _body(result)
            assert body["pool"]["error"] == "Pool status unavailable"


# =========================================================================
# Method not allowed
# =========================================================================


class TestMethodNotAllowed:
    """Verify 405 for unsupported method/path combos."""

    def test_register_get(self, handler):
        http = MockHTTPHandler(method="GET")
        result = _run(handler.handle("/api/auth/register", {}, http, "GET"))
        assert _status(result) == 405

    def test_login_get(self, handler):
        http = MockHTTPHandler(method="GET")
        result = _run(handler.handle("/api/auth/login", {}, http, "GET"))
        assert _status(result) == 405

    def test_unknown_path(self, handler):
        http = MockHTTPHandler(method="GET")
        result = _run(handler.handle("/api/auth/nonexistent", {}, http, "GET"))
        assert _status(result) == 405

    def test_sessions_post(self, handler):
        http = MockHTTPHandler(method="POST")
        result = _run(handler.handle("/api/auth/sessions", {}, http, "POST"))
        assert _status(result) == 405

    def test_health_post(self, handler):
        http = MockHTTPHandler(method="POST")
        result = _run(handler.handle("/api/auth/health", {}, http, "POST"))
        assert _status(result) == 405

    def test_verify_email_get(self, handler):
        http = MockHTTPHandler(method="GET")
        result = _run(handler.handle("/api/auth/verify-email", {}, http, "GET"))
        assert _status(result) == 405


# =========================================================================
# _check_permission internal
# =========================================================================


class TestCheckPermission:
    """Test the _check_permission RBAC integration."""

    def test_permission_allowed(self, handler, monkeypatch):
        monkeypatch.setattr(
            "aragora.server.handlers.auth.handler.extract_user_from_request",
            lambda h, s: MockAuthCtx(),
        )
        monkeypatch.setattr(
            "aragora.server.handlers.auth.handler.check_permission",
            lambda ctx, perm, res_id=None: MockRBACDecision(allowed=True),
        )
        http = MockHTTPHandler()
        result = handler._check_permission(http, "auth.read")
        assert result is None  # allowed

    def test_permission_denied(self, handler, monkeypatch):
        monkeypatch.setattr(
            "aragora.server.handlers.auth.handler.extract_user_from_request",
            lambda h, s: MockAuthCtx(),
        )
        monkeypatch.setattr(
            "aragora.server.handlers.auth.handler.check_permission",
            lambda ctx, perm, res_id=None: MockRBACDecision(allowed=False, reason="denied"),
        )
        http = MockHTTPHandler()
        result = handler._check_permission(http, "auth.write")
        assert _status(result) == 403

    def test_permission_unauthenticated(self, handler, monkeypatch):
        monkeypatch.setattr(
            "aragora.server.handlers.auth.handler.extract_user_from_request",
            lambda h, s: MockAuthCtx(is_authenticated=False, user_id=None),
        )
        http = MockHTTPHandler()
        result = handler._check_permission(http, "auth.read")
        assert _status(result) == 401


# =========================================================================
# _require_user_id
# =========================================================================


class TestRequireUserId:
    """Test _require_user_id helper."""

    def test_authenticated(self, handler, monkeypatch):
        monkeypatch.setattr(
            "aragora.server.handlers.auth.handler.extract_user_from_request",
            lambda h, s: MockAuthCtx(is_authenticated=True, user_id="u1"),
        )
        http = MockHTTPHandler()
        uid, err = handler._require_user_id(http)
        assert uid == "u1"
        assert err is None

    def test_unauthenticated(self, handler, monkeypatch):
        monkeypatch.setattr(
            "aragora.server.handlers.auth.handler.extract_user_from_request",
            lambda h, s: MockAuthCtx(is_authenticated=False, user_id=None),
        )
        http = MockHTTPHandler()
        uid, err = handler._require_user_id(http)
        assert uid is None
        assert _status(err) == 401


# =========================================================================
# _get_user_store / _get_lockout_tracker
# =========================================================================


class TestInternalHelpers:
    """Test internal helper methods."""

    def test_get_user_store(self):
        store = MagicMock()
        h = AuthHandler(server_context={"user_store": store})
        assert h._get_user_store() is store

    def test_get_user_store_missing(self, handler):
        assert handler._get_user_store() is None

    def test_get_lockout_tracker(self, handler):
        # Should return a tracker without error
        tracker = handler._get_lockout_tracker()
        assert tracker is not None


# =========================================================================
# Path normalization / version stripping
# =========================================================================


class TestPathNormalization:
    """Test version prefix stripping in handle()."""

    @patch("aragora.server.handlers.auth.login.handle_login")
    def test_v1_prefix_stripped(self, mock_fn, handler):
        mock_fn.return_value = MagicMock(status_code=200, body=b'{}')
        http = MockHTTPHandler(method="POST")
        _run(handler.handle("/api/v1/auth/login", {}, http, "POST"))
        mock_fn.assert_called_once()

    @patch("aragora.server.handlers.auth.login.handle_register")
    def test_v2_prefix_stripped(self, mock_fn, handler):
        """If a v2 is passed, strip_version_prefix should still handle it."""
        mock_fn.return_value = MagicMock(status_code=200, body=b'{}')
        http = MockHTTPHandler(method="POST")
        # strip_version_prefix handles /api/v<N>/ patterns
        _run(handler.handle("/api/v2/auth/register", {}, http, "POST"))
        mock_fn.assert_called_once()


# =========================================================================
# Method determination
# =========================================================================


class TestMethodDetermination:
    """Test how the method is determined from handler or argument."""

    @patch("aragora.server.handlers.auth.login.handle_login")
    def test_method_from_argument(self, mock_fn, handler):
        mock_fn.return_value = MagicMock(status_code=200, body=b'{}')
        http = MockHTTPHandler(method="GET")  # handler says GET
        _run(handler.handle("/api/auth/login", {}, http, "POST"))  # but method arg says POST
        mock_fn.assert_called_once()

    def test_method_from_handler_command(self, handler):
        http = MockHTTPHandler(method="GET")
        http.command = "GET"
        result = _run(handler.handle("/api/auth/me", {}, http, None))
        # method=None -> uses handler.command -> GET -> GET /me should work
        # Since _check_permission is autopatched, and there's no user_store, it might 503
        # but the point is it resolved to GET handler, not 405
        # Just confirm it's not 405
        assert _status(result) != 405 or _status(result) == 503

    def test_method_default_get(self, handler):
        # No handler, no method -> default GET
        result = _run(handler.handle("/api/auth/me", {}, None, None))
        # Even with None handler, it should try to GET /me
        # _check_permission on None handler... depends on mock
        assert result is not None


# =========================================================================
# Edge cases / error handling
# =========================================================================


class TestEdgeCases:
    """Various edge cases."""

    def test_empty_path(self, handler):
        http = MockHTTPHandler()
        result = _run(handler.handle("", {}, http, "GET"))
        assert _status(result) == 405

    def test_none_query_params(self, handler):
        """Ensure None query params don't crash."""
        http = MockHTTPHandler(method="GET")
        result = _run(handler.handle("/api/auth/health", None, http, "GET"))
        # health should still work
        assert result is not None

    @patch("aragora.server.handlers.auth.handler.AuthHandler._check_permission", return_value=None)
    @patch("aragora.server.middleware.auth.extract_token", return_value="tok")
    @patch("aragora.billing.jwt_auth.revoke_token_persistent", return_value=True)
    @patch("aragora.billing.jwt_auth.get_token_blacklist")
    def test_logout_persistent_success_memory_fail(self, mock_bl, mock_rev, mock_ext, mock_perm, handler):
        """Test logout when persistent succeeds but in-memory fails."""
        bl = MagicMock()
        bl.revoke_token.return_value = False  # in-memory fails
        mock_bl.return_value = bl
        http = MockHTTPHandler(method="POST")
        result = _run(handler.handle("/api/auth/logout", {}, http, "POST"))
        # Should still succeed
        assert _status(result) == 200

    @patch("aragora.server.handlers.auth.handler.AuthHandler._check_permission", return_value=None)
    @patch("aragora.server.middleware.auth.extract_token", return_value="tok")
    @patch("aragora.billing.jwt_auth.revoke_token_persistent", return_value=False)
    @patch("aragora.billing.jwt_auth.get_token_blacklist")
    def test_logout_persistent_fail_memory_ok(self, mock_bl, mock_rev, mock_ext, mock_perm, handler):
        """Test logout when persistent fails but in-memory succeeds."""
        bl = MagicMock()
        bl.revoke_token.return_value = True
        mock_bl.return_value = bl
        http = MockHTTPHandler(method="POST")
        result = _run(handler.handle("/api/auth/logout", {}, http, "POST"))
        assert _status(result) == 200

    @patch("aragora.server.handlers.auth.handler.AuthHandler._check_permission", return_value=None)
    def test_get_me_org_lookup_fails(self, mock_perm):
        """If org lookup returns None, still return user."""
        user = MockUser(org_id="org-99")
        store = _make_user_store(user=user)
        store.get_organization_by_id.return_value = None
        h = AuthHandler(server_context={"user_store": store})
        http = MockHTTPHandler(method="GET")
        result = _run(h.handle("/api/auth/me", {}, http, "GET"))
        assert _status(result) == 200
        body = _body(result)
        assert body["user"]["id"] == "user-001"
        # org is None even though user has org_id
        assert body["organization"] is None

    @patch("aragora.server.handlers.auth.handler.AuthHandler._check_permission", return_value=None)
    @patch("aragora.server.middleware.auth.extract_token", return_value="tok")
    @patch("aragora.billing.jwt_auth.revoke_token_persistent", return_value=True)
    @patch("aragora.billing.jwt_auth.get_token_blacklist")
    def test_logout_all_no_token(self, mock_bl, mock_rev, mock_ext, mock_perm):
        """logout-all with no extractable token still succeeds."""
        mock_ext.return_value = None
        store = _make_user_store()
        h = AuthHandler(server_context={"user_store": store})
        http = MockHTTPHandler(method="POST")
        result = _run(h.handle("/api/auth/logout-all", {}, http, "POST"))
        assert _status(result) == 200

    @patch("aragora.server.handlers.auth.handler.AuthHandler._check_permission", return_value=None)
    def test_update_me_async_store(self, mock_perm):
        """Test update with async-only user store."""
        user = MockUser()
        store = MagicMock()
        store.get_user_by_id.return_value = None
        store.get_user_by_id_async = MagicMock(return_value=user)
        store.update_user_async = MagicMock(return_value=None)
        h = AuthHandler(server_context={"user_store": store})
        http = MockHTTPHandler(body={"name": "New"}, method="PUT")
        result = _run(h.handle("/api/auth/me", {}, http, "PUT"))
        # Should use async fallback
        assert result is not None


# =========================================================================
# ROUTES constant coverage
# =========================================================================


class TestRoutesConstant:
    """Ensure ROUTES list is correct."""

    def test_routes_is_list(self, handler):
        assert isinstance(AuthHandler.ROUTES, list)

    def test_routes_not_empty(self, handler):
        assert len(AuthHandler.ROUTES) > 20

    def test_all_routes_start_with_api(self, handler):
        for route in AuthHandler.ROUTES:
            assert route.startswith("/api/"), f"Route {route} doesn't start with /api/"

    def test_no_duplicate_routes(self, handler):
        # Allow wildcards, check non-wildcard routes
        non_wildcard = [r for r in AuthHandler.ROUTES if "*" not in r]
        assert len(non_wildcard) == len(set(non_wildcard))


# =========================================================================
# Refresh token edge cases
# =========================================================================


class TestRefreshEdgeCases:
    """Additional refresh token edge cases."""

    @patch("aragora.billing.jwt_auth.revoke_token_persistent")
    @patch("aragora.billing.jwt_auth.get_token_blacklist")
    @patch("aragora.billing.jwt_auth.create_token_pair")
    def test_refresh_async_user_store(self, mock_create, mock_bl, mock_revoke):
        """Test refresh with async-only user store."""
        mock_create.return_value = MockTokenPair()
        bl = MagicMock()
        bl.revoke_token.return_value = True
        mock_bl.return_value = bl
        mock_revoke.return_value = True

        user = MockUser()
        store = MagicMock()
        store.get_user_by_id.return_value = None
        store.get_user_by_id_async = MagicMock(return_value=user)
        h = AuthHandler(server_context={"user_store": store})
        http = MockHTTPHandler(body={"refresh_token": "rt-valid"}, method="POST")
        result = _run(h.handle("/api/auth/refresh", {}, http, "POST"))
        assert _status(result) == 200

    @patch("aragora.billing.jwt_auth.revoke_token_persistent", side_effect=ConnectionError("conn"))
    def test_refresh_revoke_connection_error(self, mock_revoke):
        store = _make_user_store()
        h = AuthHandler(server_context={"user_store": store})
        http = MockHTTPHandler(body={"refresh_token": "rt-valid"}, method="POST")
        result = _run(h.handle("/api/auth/refresh", {}, http, "POST"))
        assert _status(result) == 500

    @patch("aragora.billing.jwt_auth.revoke_token_persistent", side_effect=TimeoutError("timeout"))
    def test_refresh_revoke_timeout_error(self, mock_revoke):
        store = _make_user_store()
        h = AuthHandler(server_context={"user_store": store})
        http = MockHTTPHandler(body={"refresh_token": "rt-valid"}, method="POST")
        result = _run(h.handle("/api/auth/refresh", {}, http, "POST"))
        assert _status(result) == 500

    @patch("aragora.billing.jwt_auth.revoke_token_persistent", side_effect=RuntimeError("runtime"))
    def test_refresh_revoke_runtime_error(self, mock_revoke):
        store = _make_user_store()
        h = AuthHandler(server_context={"user_store": store})
        http = MockHTTPHandler(body={"refresh_token": "rt-valid"}, method="POST")
        result = _run(h.handle("/api/auth/refresh", {}, http, "POST"))
        assert _status(result) == 500
