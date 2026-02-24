"""
Tests for aragora.server.handlers.admin.handler - Main Admin Handler Facade.

Comprehensive tests covering:
- AdminHandler.__init__ with/without context
- AdminHandler.can_handle static method
- AdminHandler._get_user_store
- AdminHandler._require_admin (auth, role, MFA checks)
- AdminHandler._check_rbac_permission (RBAC available/unavailable, fail-closed, admin fallback)
- AdminHandler.handle route dispatch for all GET and POST routes
- Rate limiting on admin endpoints
- Path validation for user-targeted POST routes
- Method-not-allowed fallback
- admin_secure_endpoint decorator (auth, role, MFA, RBAC, audit)
- Module exports (__all__)
- ADMIN_ROLES constant
"""

from __future__ import annotations

import json
import os
from typing import Any
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from aragora.server.handlers.admin.handler import (
    ADMIN_ROLES,
    AdminHandler,
    admin_secure_endpoint,
    _admin_limiter,
)
from aragora.server.handlers.utils.responses import HandlerResult, error_response


# ===========================================================================
# Helpers
# ===========================================================================


def _body(result: HandlerResult) -> dict:
    """Parse JSON body from a HandlerResult."""
    if result and result.body:
        return json.loads(result.body.decode("utf-8"))
    return {}


def _status(result: HandlerResult) -> int:
    """Extract status code from a HandlerResult."""
    return result.status_code


# ===========================================================================
# Mock classes
# ===========================================================================


class MockAuthContext:
    """Minimal mock auth context."""

    def __init__(
        self,
        user_id: str = "admin-001",
        org_id: str = "org-001",
        is_authenticated: bool = True,
    ):
        self.user_id = user_id
        self.org_id = org_id
        self.is_authenticated = is_authenticated


class MockUser:
    """Mock user returned by user store."""

    def __init__(
        self,
        user_id: str = "admin-001",
        email: str = "admin@example.com",
        name: str = "Admin User",
        role: str = "admin",
        org_id: str = "org-001",
        is_active: bool = True,
        mfa_enabled: bool = True,
    ):
        self.id = user_id
        self.email = email
        self.name = name
        self.role = role
        self.org_id = org_id
        self.is_active = is_active
        self.mfa_enabled = mfa_enabled

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "email": self.email,
            "name": self.name,
            "role": self.role,
            "org_id": self.org_id,
            "is_active": self.is_active,
        }


class MockHTTPHandler:
    """Mock HTTP handler for admin tests."""

    def __init__(
        self,
        path: str = "/api/v1/admin/stats",
        method: str = "GET",
        body: bytes = b"{}",
        auth_header: str | None = None,
    ):
        self.path = path
        self.command = method
        self.request_body = body
        self.headers = {"Content-Type": "application/json"}
        if auth_header:
            self.headers["Authorization"] = auth_header
        self.client_address = ("127.0.0.1", 54321)
        self.rfile = MagicMock()
        self.rfile.read.return_value = body


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    """Reset the admin rate limiter between tests to prevent cross-test leaks."""
    _admin_limiter._buckets.clear()
    yield
    _admin_limiter._buckets.clear()


@pytest.fixture
def admin_user():
    return MockUser(role="admin")


@pytest.fixture
def owner_user():
    return MockUser(user_id="owner-001", role="owner")


@pytest.fixture
def member_user():
    return MockUser(user_id="member-001", role="member")


@pytest.fixture
def user_store(admin_user):
    """User store returning an admin user by default."""
    store = MagicMock()
    store.get_user_by_id.return_value = admin_user
    store.get_admin_stats.return_value = {
        "total_users": 10,
        "total_organizations": 3,
        "tier_distribution": {"free": 1, "starter": 1, "professional": 1},
    }
    store.list_all_users.return_value = ([], 0)
    store.list_all_organizations.return_value = ([], 0)
    store.update_user.return_value = True
    store.reset_failed_login_attempts.return_value = True
    return store


@pytest.fixture
def ctx(user_store):
    """Standard server context dict."""
    return {"user_store": user_store}


@pytest.fixture
def handler(ctx):
    """AdminHandler with standard context."""
    return AdminHandler(ctx)


@pytest.fixture
def http():
    """Factory for MockHTTPHandler."""

    def _make(
        path: str = "/api/v1/admin/stats",
        method: str = "GET",
        body: bytes = b"{}",
        auth_header: str | None = None,
    ):
        return MockHTTPHandler(path=path, method=method, body=body, auth_header=auth_header)

    return _make


# ===========================================================================
# Constants and Exports
# ===========================================================================


class TestConstants:
    def test_admin_roles_contains_admin(self):
        assert "admin" in ADMIN_ROLES

    def test_admin_roles_contains_owner(self):
        assert "owner" in ADMIN_ROLES

    def test_admin_roles_is_set(self):
        assert isinstance(ADMIN_ROLES, set)

    def test_admin_roles_size(self):
        assert len(ADMIN_ROLES) == 8


class TestModuleExports:
    def test_all_contains_admin_handler(self):
        from aragora.server.handlers.admin.handler import __all__

        assert "AdminHandler" in __all__

    def test_all_contains_admin_roles(self):
        from aragora.server.handlers.admin.handler import __all__

        assert "ADMIN_ROLES" in __all__

    def test_all_contains_admin_secure_endpoint(self):
        from aragora.server.handlers.admin.handler import __all__

        assert "admin_secure_endpoint" in __all__

    def test_all_contains_perm_admin_users_write(self):
        from aragora.server.handlers.admin.handler import __all__

        assert "PERM_ADMIN_USERS_WRITE" in __all__

    def test_all_contains_perm_admin_impersonate(self):
        from aragora.server.handlers.admin.handler import __all__

        assert "PERM_ADMIN_IMPERSONATE" in __all__

    def test_all_contains_perm_admin_nomic_write(self):
        from aragora.server.handlers.admin.handler import __all__

        assert "PERM_ADMIN_NOMIC_WRITE" in __all__

    def test_all_contains_perm_admin_system_write(self):
        from aragora.server.handlers.admin.handler import __all__

        assert "PERM_ADMIN_SYSTEM_WRITE" in __all__


# ===========================================================================
# AdminHandler.__init__
# ===========================================================================


class TestAdminHandlerInit:
    def test_default_ctx_is_empty_dict(self):
        h = AdminHandler()
        assert h.ctx == {}

    def test_ctx_set_from_arg(self, ctx):
        h = AdminHandler(ctx)
        assert h.ctx is ctx

    def test_none_ctx_becomes_empty_dict(self):
        h = AdminHandler(None)
        assert h.ctx == {}


# ===========================================================================
# AdminHandler.can_handle
# ===========================================================================


class TestCanHandle:
    def test_admin_path(self):
        assert AdminHandler.can_handle("/api/v1/admin") is True

    def test_admin_subpath(self):
        assert AdminHandler.can_handle("/api/v1/admin/users") is True

    def test_admin_deep_subpath(self):
        assert AdminHandler.can_handle("/api/v1/admin/nomic/status") is True

    def test_non_admin_path(self):
        assert AdminHandler.can_handle("/api/v1/debates") is False

    def test_partial_match(self):
        # /api/v1/administration is different from /api/v1/admin
        assert AdminHandler.can_handle("/api/v1/administration") is True  # startswith

    def test_empty_path(self):
        assert AdminHandler.can_handle("") is False

    def test_root_path(self):
        assert AdminHandler.can_handle("/") is False


# ===========================================================================
# AdminHandler._get_user_store
# ===========================================================================


class TestGetUserStore:
    def test_returns_user_store_from_ctx(self, handler, user_store):
        assert handler._get_user_store() is user_store

    def test_returns_none_when_no_user_store(self):
        h = AdminHandler({})
        assert h._get_user_store() is None

    def test_returns_none_when_empty_ctx(self):
        h = AdminHandler()
        assert h._get_user_store() is None


# ===========================================================================
# AdminHandler._require_admin
# ===========================================================================


class TestRequireAdmin:
    def test_no_user_store_returns_503(self, http):
        h = AdminHandler({})
        auth_ctx, err = h._require_admin(http())
        assert auth_ctx is None
        assert _status(err) == 503

    @patch("aragora.server.handlers.admin.handler.extract_user_from_request")
    def test_unauthenticated_returns_401(self, mock_extract, http, ctx):
        mock_ctx = MockAuthContext(is_authenticated=False)
        mock_extract.return_value = mock_ctx
        h = AdminHandler(ctx)
        auth_ctx, err = h._require_admin(http())
        assert auth_ctx is None
        assert _status(err) == 401

    @patch("aragora.server.handlers.admin.handler.extract_user_from_request")
    def test_non_admin_returns_403(self, mock_extract, http, ctx, member_user):
        mock_ctx = MockAuthContext(user_id="member-001")
        mock_extract.return_value = mock_ctx
        ctx["user_store"].get_user_by_id.return_value = member_user
        h = AdminHandler(ctx)
        auth_ctx, err = h._require_admin(http())
        assert auth_ctx is None
        assert _status(err) == 403
        assert "Admin access required" in _body(err).get("error", "")

    @patch("aragora.server.handlers.admin.handler.extract_user_from_request")
    def test_user_not_found_returns_403(self, mock_extract, http, ctx):
        mock_ctx = MockAuthContext(user_id="ghost-001")
        mock_extract.return_value = mock_ctx
        ctx["user_store"].get_user_by_id.return_value = None
        h = AdminHandler(ctx)
        auth_ctx, err = h._require_admin(http())
        assert auth_ctx is None
        assert _status(err) == 403

    @patch("aragora.server.handlers.admin.handler.enforce_admin_mfa_policy")
    @patch("aragora.server.handlers.admin.handler.extract_user_from_request")
    def test_mfa_not_enabled_returns_403(self, mock_extract, mock_mfa, http, ctx, admin_user):
        mock_ctx = MockAuthContext()
        mock_extract.return_value = mock_ctx
        mock_mfa.return_value = {"reason": "MFA not enabled", "action": "enable_mfa"}
        h = AdminHandler(ctx)
        auth_ctx, err = h._require_admin(http())
        assert auth_ctx is None
        assert _status(err) == 403
        body = _body(err)
        error_obj = body.get("error", "")
        if isinstance(error_obj, dict):
            assert "MFA" in error_obj.get("message", "")
            assert error_obj.get("code") == "ADMIN_MFA_REQUIRED"
        else:
            assert "MFA" in error_obj

    @patch("aragora.server.handlers.admin.handler.enforce_admin_mfa_policy")
    @patch("aragora.server.handlers.admin.handler.extract_user_from_request")
    def test_admin_with_mfa_succeeds(self, mock_extract, mock_mfa, http, ctx, admin_user):
        mock_ctx = MockAuthContext()
        mock_extract.return_value = mock_ctx
        mock_mfa.return_value = None  # MFA compliant
        h = AdminHandler(ctx)
        auth_ctx, err = h._require_admin(http())
        assert auth_ctx is mock_ctx
        assert err is None

    @patch("aragora.server.handlers.admin.handler.enforce_admin_mfa_policy")
    @patch("aragora.server.handlers.admin.handler.extract_user_from_request")
    def test_owner_role_succeeds(self, mock_extract, mock_mfa, http, ctx, owner_user):
        mock_ctx = MockAuthContext(user_id="owner-001")
        mock_extract.return_value = mock_ctx
        mock_mfa.return_value = None
        ctx["user_store"].get_user_by_id.return_value = owner_user
        h = AdminHandler(ctx)
        auth_ctx, err = h._require_admin(http())
        assert auth_ctx is mock_ctx
        assert err is None


# ===========================================================================
# AdminHandler._check_rbac_permission
# ===========================================================================


class TestCheckRbacPermission:
    @patch("aragora.server.handlers.admin.handler.RBAC_AVAILABLE", False)
    @patch("aragora.server.handlers.admin.handler.rbac_fail_closed", return_value=False)
    def test_rbac_unavailable_no_fail_closed_returns_none(self, mock_fc, handler):
        result = handler._check_rbac_permission(MockAuthContext(), "admin.test")
        assert result is None

    @patch("aragora.server.handlers.admin.handler.RBAC_AVAILABLE", False)
    @patch("aragora.server.handlers.admin.handler.rbac_fail_closed", return_value=True)
    def test_rbac_unavailable_fail_closed_returns_503(self, mock_fc, handler):
        result = handler._check_rbac_permission(MockAuthContext(), "admin.test")
        assert _status(result) == 503

    @patch("aragora.server.handlers.admin.handler.RBAC_AVAILABLE", True)
    @patch("aragora.server.handlers.admin.handler.check_permission")
    @patch("aragora.server.handlers.admin.handler.AuthorizationContext")
    def test_rbac_allowed_returns_none(self, mock_ctx_class, mock_check, handler):
        mock_decision = MagicMock()
        mock_decision.allowed = True
        mock_check.return_value = mock_decision
        result = handler._check_rbac_permission(MockAuthContext(), "admin.test")
        assert result is None

    @patch("aragora.server.handlers.admin.handler.RBAC_AVAILABLE", True)
    @patch("aragora.server.handlers.admin.handler.record_rbac_check")
    @patch("aragora.server.handlers.admin.handler.check_permission")
    @patch("aragora.server.handlers.admin.handler.AuthorizationContext")
    def test_rbac_denied_admin_role_fallback(
        self, mock_ctx_class, mock_check, mock_record, handler, admin_user
    ):
        """Admin role should allow even if RBAC denies the specific permission."""
        mock_decision = MagicMock()
        mock_decision.allowed = False
        mock_decision.reason = "No explicit grant"
        mock_check.return_value = mock_decision
        handler.ctx["user_store"].get_user_by_id.return_value = admin_user
        result = handler._check_rbac_permission(MockAuthContext(), "admin.test")
        assert result is None  # Admin fallback allows

    @patch("aragora.server.handlers.admin.handler.RBAC_AVAILABLE", True)
    @patch("aragora.server.handlers.admin.handler.record_rbac_check")
    @patch("aragora.server.handlers.admin.handler.check_permission")
    @patch("aragora.server.handlers.admin.handler.AuthorizationContext")
    def test_rbac_denied_non_admin_returns_403(
        self, mock_ctx_class, mock_check, mock_record, handler, member_user
    ):
        mock_decision = MagicMock()
        mock_decision.allowed = False
        mock_decision.reason = "No permission"
        mock_check.return_value = mock_decision
        handler.ctx["user_store"].get_user_by_id.return_value = member_user
        result = handler._check_rbac_permission(MockAuthContext(user_id="member-001"), "admin.test")
        assert _status(result) == 403

    @patch("aragora.server.handlers.admin.handler.RBAC_AVAILABLE", True)
    @patch("aragora.server.handlers.admin.handler.record_rbac_check")
    @patch("aragora.server.handlers.admin.handler.check_permission")
    @patch("aragora.server.handlers.admin.handler.AuthorizationContext")
    def test_rbac_permission_denied_error_returns_403(
        self, mock_ctx_class, mock_check, mock_record, handler
    ):
        from aragora.rbac import PermissionDeniedError

        mock_check.side_effect = PermissionDeniedError("denied")
        result = handler._check_rbac_permission(MockAuthContext(), "admin.test")
        assert _status(result) == 403

    def test_no_user_store_returns_none(self):
        h = AdminHandler({})
        with (
            patch("aragora.server.handlers.admin.handler.RBAC_AVAILABLE", True),
            patch("aragora.server.handlers.admin.handler.AuthorizationContext"),
        ):
            result = h._check_rbac_permission(MockAuthContext(), "admin.test")
        assert result is None


# ===========================================================================
# ROUTES constant
# ===========================================================================


class TestRoutes:
    def test_routes_is_list(self):
        assert isinstance(AdminHandler.ROUTES, list)

    def test_routes_all_start_with_api_v1_admin(self):
        for route in AdminHandler.ROUTES:
            assert route.startswith("/api/v1/admin"), (
                f"Route {route} doesn't start with /api/v1/admin"
            )

    def test_organizations_route_present(self):
        assert "/api/v1/admin/organizations" in AdminHandler.ROUTES

    def test_users_route_present(self):
        assert "/api/v1/admin/users" in AdminHandler.ROUTES

    def test_stats_route_present(self):
        assert "/api/v1/admin/stats" in AdminHandler.ROUTES

    def test_system_metrics_route_present(self):
        assert "/api/v1/admin/system/metrics" in AdminHandler.ROUTES

    def test_impersonate_route_present(self):
        assert "/api/v1/admin/impersonate" in AdminHandler.ROUTES

    def test_revenue_route_present(self):
        assert "/api/v1/admin/revenue" in AdminHandler.ROUTES

    def test_nomic_status_route_present(self):
        assert "/api/v1/admin/nomic/status" in AdminHandler.ROUTES

    def test_nomic_circuit_breakers_route_present(self):
        assert "/api/v1/admin/nomic/circuit-breakers" in AdminHandler.ROUTES

    def test_nomic_reset_route_present(self):
        assert "/api/v1/admin/nomic/reset" in AdminHandler.ROUTES

    def test_nomic_pause_route_present(self):
        assert "/api/v1/admin/nomic/pause" in AdminHandler.ROUTES

    def test_nomic_resume_route_present(self):
        assert "/api/v1/admin/nomic/resume" in AdminHandler.ROUTES

    def test_nomic_circuit_breakers_reset_route_present(self):
        assert "/api/v1/admin/nomic/circuit-breakers/reset" in AdminHandler.ROUTES

    def test_security_routes_present(self):
        assert "/api/v1/admin/security/audit" in AdminHandler.ROUTES
        assert "/api/v1/admin/security/compliance" in AdminHandler.ROUTES
        assert "/api/v1/admin/security/scan" in AdminHandler.ROUTES
        assert "/api/v1/admin/security/threats" in AdminHandler.ROUTES


# ===========================================================================
# AdminHandler.handle - Rate Limiting
# ===========================================================================


class TestHandleRateLimiting:
    def test_rate_limit_returns_429_after_exceeding(self, handler, http):
        """Hit the rate limiter (10 req/min) and verify 429."""
        for _ in range(10):
            result = handler.handle("/api/v1/admin/stats", {}, http(), "GET")
        # 11th request should be rate limited
        result = handler.handle("/api/v1/admin/stats", {}, http(), "GET")
        assert _status(result) == 429
        assert "Rate limit" in _body(result).get("error", "")

    def test_rate_limit_uses_test_scoped_key(self, handler, http, monkeypatch):
        """When PYTEST_CURRENT_TEST is set, rate key is scoped per test."""
        monkeypatch.setenv("PYTEST_CURRENT_TEST", "test_some_unique_test")
        # Should not be rate limited since this is a fresh key
        result = handler.handle("/api/v1/admin/stats", {}, http(), "GET")
        assert _status(result) != 429


# ===========================================================================
# AdminHandler.handle - GET routes
# ===========================================================================


class TestHandleGetRoutes:
    @patch.object(AdminHandler, "_list_organizations")
    def test_get_organizations(self, mock_method, handler, http):
        mock_method.return_value = HandlerResult(200, "application/json", b'{"ok":true}')
        h = http(path="/api/v1/admin/organizations")
        result = handler.handle("/api/v1/admin/organizations", {"limit": "10"}, h, "GET")
        mock_method.assert_called_once_with(h, {"limit": "10"})
        assert _status(result) == 200

    @patch.object(AdminHandler, "_list_users")
    def test_get_users(self, mock_method, handler, http):
        mock_method.return_value = HandlerResult(200, "application/json", b'{"ok":true}')
        h = http(path="/api/v1/admin/users")
        result = handler.handle("/api/v1/admin/users", {"role": "admin"}, h, "GET")
        mock_method.assert_called_once_with(h, {"role": "admin"})
        assert _status(result) == 200

    @patch.object(AdminHandler, "_get_stats")
    def test_get_stats(self, mock_method, handler, http):
        mock_method.return_value = HandlerResult(200, "application/json", b'{"stats":{}}')
        h = http()
        result = handler.handle("/api/v1/admin/stats", {}, h, "GET")
        mock_method.assert_called_once_with(h)
        assert _status(result) == 200

    @patch.object(AdminHandler, "_get_system_metrics")
    def test_get_system_metrics(self, mock_method, handler, http):
        mock_method.return_value = HandlerResult(200, "application/json", b'{"metrics":{}}')
        h = http(path="/api/v1/admin/system/metrics")
        result = handler.handle("/api/v1/admin/system/metrics", {}, h, "GET")
        mock_method.assert_called_once_with(h)
        assert _status(result) == 200

    @patch.object(AdminHandler, "_get_revenue_stats")
    def test_get_revenue(self, mock_method, handler, http):
        mock_method.return_value = HandlerResult(200, "application/json", b'{"revenue":{}}')
        h = http(path="/api/v1/admin/revenue")
        result = handler.handle("/api/v1/admin/revenue", {}, h, "GET")
        mock_method.assert_called_once_with(h)
        assert _status(result) == 200

    @patch.object(AdminHandler, "_get_nomic_status")
    def test_get_nomic_status(self, mock_method, handler, http):
        mock_method.return_value = HandlerResult(200, "application/json", b'{"running":false}')
        h = http(path="/api/v1/admin/nomic/status")
        result = handler.handle("/api/v1/admin/nomic/status", {}, h, "GET")
        mock_method.assert_called_once_with(h)
        assert _status(result) == 200

    @patch.object(AdminHandler, "_get_nomic_circuit_breakers")
    def test_get_nomic_circuit_breakers(self, mock_method, handler, http):
        mock_method.return_value = HandlerResult(200, "application/json", b"{}")
        h = http(path="/api/v1/admin/nomic/circuit-breakers")
        result = handler.handle("/api/v1/admin/nomic/circuit-breakers", {}, h, "GET")
        mock_method.assert_called_once_with(h)
        assert _status(result) == 200

    def test_unknown_get_route_returns_405(self, handler, http):
        result = handler.handle("/api/v1/admin/nonexistent", {}, http(), "GET")
        assert _status(result) == 405
        assert "Method not allowed" in _body(result).get("error", "")


# ===========================================================================
# AdminHandler.handle - POST routes
# ===========================================================================


class TestHandlePostRoutes:
    @patch.object(AdminHandler, "_impersonate_user")
    def test_post_impersonate(self, mock_method, handler, http):
        mock_method.return_value = HandlerResult(200, "application/json", b'{"token":"x"}')
        h = http(path="/api/v1/admin/impersonate/user-123", method="POST")
        result = handler.handle("/api/v1/admin/impersonate/user-123", {}, h, "POST")
        mock_method.assert_called_once_with(h, "user-123")
        assert _status(result) == 200

    def test_post_impersonate_invalid_user_id(self, handler, http):
        """User IDs with special characters should be rejected."""
        h = http(path="/api/v1/admin/impersonate/<script>alert(1)</script>", method="POST")
        result = handler.handle(
            "/api/v1/admin/impersonate/<script>alert(1)</script>", {}, h, "POST"
        )
        assert _status(result) == 400

    @patch.object(AdminHandler, "_deactivate_user")
    def test_post_deactivate_user(self, mock_method, handler, http):
        mock_method.return_value = HandlerResult(200, "application/json", b'{"success":true}')
        path = "/api/v1/admin/users/user-001/deactivate"
        h = http(path=path, method="POST")
        result = handler.handle(path, {}, h, "POST")
        mock_method.assert_called_once_with(h, "user-001")
        assert _status(result) == 200

    def test_post_deactivate_user_invalid_id(self, handler, http):
        path = "/api/v1/admin/users/<script>/deactivate"
        h = http(path=path, method="POST")
        result = handler.handle(path, {}, h, "POST")
        assert _status(result) == 400

    @patch.object(AdminHandler, "_activate_user")
    def test_post_activate_user(self, mock_method, handler, http):
        mock_method.return_value = HandlerResult(200, "application/json", b'{"success":true}')
        path = "/api/v1/admin/users/user-002/activate"
        h = http(path=path, method="POST")
        result = handler.handle(path, {}, h, "POST")
        mock_method.assert_called_once_with(h, "user-002")
        assert _status(result) == 200

    def test_post_activate_user_invalid_id(self, handler, http):
        path = "/api/v1/admin/users/bad id!/activate"
        h = http(path=path, method="POST")
        result = handler.handle(path, {}, h, "POST")
        assert _status(result) == 400

    @patch.object(AdminHandler, "_unlock_user")
    def test_post_unlock_user(self, mock_method, handler, http):
        mock_method.return_value = HandlerResult(200, "application/json", b'{"success":true}')
        path = "/api/v1/admin/users/user-003/unlock"
        h = http(path=path, method="POST")
        result = handler.handle(path, {}, h, "POST")
        mock_method.assert_called_once_with(h, "user-003")
        assert _status(result) == 200

    def test_post_unlock_user_invalid_id(self, handler, http):
        path = "/api/v1/admin/users/x%00y/unlock"
        h = http(path=path, method="POST")
        result = handler.handle(path, {}, h, "POST")
        assert _status(result) == 400

    @patch.object(AdminHandler, "_reset_nomic_phase")
    def test_post_nomic_reset(self, mock_method, handler, http):
        mock_method.return_value = HandlerResult(200, "application/json", b'{"success":true}')
        h = http(path="/api/v1/admin/nomic/reset", method="POST")
        result = handler.handle("/api/v1/admin/nomic/reset", {}, h, "POST")
        mock_method.assert_called_once_with(h)
        assert _status(result) == 200

    @patch.object(AdminHandler, "_pause_nomic")
    def test_post_nomic_pause(self, mock_method, handler, http):
        mock_method.return_value = HandlerResult(200, "application/json", b'{"success":true}')
        h = http(path="/api/v1/admin/nomic/pause", method="POST")
        result = handler.handle("/api/v1/admin/nomic/pause", {}, h, "POST")
        mock_method.assert_called_once_with(h)
        assert _status(result) == 200

    @patch.object(AdminHandler, "_resume_nomic")
    def test_post_nomic_resume(self, mock_method, handler, http):
        mock_method.return_value = HandlerResult(200, "application/json", b'{"success":true}')
        h = http(path="/api/v1/admin/nomic/resume", method="POST")
        result = handler.handle("/api/v1/admin/nomic/resume", {}, h, "POST")
        mock_method.assert_called_once_with(h)
        assert _status(result) == 200

    @patch.object(AdminHandler, "_reset_nomic_circuit_breakers")
    def test_post_nomic_circuit_breakers_reset(self, mock_method, handler, http):
        mock_method.return_value = HandlerResult(200, "application/json", b'{"success":true}')
        path = "/api/v1/admin/nomic/circuit-breakers/reset"
        h = http(path=path, method="POST")
        result = handler.handle(path, {}, h, "POST")
        mock_method.assert_called_once_with(h)
        assert _status(result) == 200

    def test_unknown_post_route_returns_405(self, handler, http):
        result = handler.handle("/api/v1/admin/nonexistent", {}, http(method="POST"), "POST")
        assert _status(result) == 405


# ===========================================================================
# AdminHandler.handle - Method dispatch
# ===========================================================================


class TestHandleMethodDispatch:
    def test_put_returns_405(self, handler, http):
        result = handler.handle("/api/v1/admin/stats", {}, http(method="PUT"), "PUT")
        assert _status(result) == 405

    def test_delete_returns_405(self, handler, http):
        result = handler.handle("/api/v1/admin/stats", {}, http(method="DELETE"), "DELETE")
        assert _status(result) == 405

    def test_patch_returns_405(self, handler, http):
        result = handler.handle("/api/v1/admin/stats", {}, http(method="PATCH"), "PATCH")
        assert _status(result) == 405

    def test_method_from_handler_command(self, handler, http):
        """When handler has command attribute, it overrides the method argument."""
        h = http(path="/api/v1/admin/stats", method="GET")
        h.command = "GET"
        with patch.object(AdminHandler, "_get_stats") as mock_method:
            mock_method.return_value = HandlerResult(200, "application/json", b"{}")
            result = handler.handle("/api/v1/admin/stats", {}, h, "POST")
            # handler.command is GET, so it should route to GET
            mock_method.assert_called_once()


# ===========================================================================
# AdminHandler.handle - User ID extraction from paths
# ===========================================================================


class TestPathUserIdExtraction:
    @patch.object(AdminHandler, "_impersonate_user")
    def test_impersonate_extracts_user_id_from_last_segment(self, mock_method, handler, http):
        mock_method.return_value = HandlerResult(200, "application/json", b"{}")
        h = http(method="POST")
        handler.handle("/api/v1/admin/impersonate/abc-123-def", {}, h, "POST")
        mock_method.assert_called_once()
        args = mock_method.call_args[0]
        assert args[0] is h
        assert args[1] == "abc-123-def"

    @patch.object(AdminHandler, "_deactivate_user")
    def test_deactivate_extracts_user_id_from_penultimate_segment(self, mock_method, handler, http):
        mock_method.return_value = HandlerResult(200, "application/json", b"{}")
        path = "/api/v1/admin/users/uid-789/deactivate"
        h = http(path=path, method="POST")
        handler.handle(path, {}, h, "POST")
        args = mock_method.call_args[0]
        assert args[1] == "uid-789"

    @patch.object(AdminHandler, "_activate_user")
    def test_activate_extracts_user_id(self, mock_method, handler, http):
        mock_method.return_value = HandlerResult(200, "application/json", b"{}")
        path = "/api/v1/admin/users/uid-456/activate"
        h = http(path=path, method="POST")
        handler.handle(path, {}, h, "POST")
        args = mock_method.call_args[0]
        assert args[1] == "uid-456"

    @patch.object(AdminHandler, "_unlock_user")
    def test_unlock_extracts_user_id(self, mock_method, handler, http):
        mock_method.return_value = HandlerResult(200, "application/json", b"{}")
        path = "/api/v1/admin/users/uid-999/unlock"
        h = http(path=path, method="POST")
        handler.handle(path, {}, h, "POST")
        args = mock_method.call_args[0]
        assert args[1] == "uid-999"


# ===========================================================================
# AdminHandler.RESOURCE_TYPE
# ===========================================================================


class TestResourceType:
    def test_resource_type_is_admin(self):
        assert AdminHandler.RESOURCE_TYPE == "admin"


# ===========================================================================
# admin_secure_endpoint decorator
# ===========================================================================


class TestAdminSecureEndpoint:
    """Tests for the admin_secure_endpoint decorator factory."""

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.admin.handler.enforce_admin_mfa_policy")
    async def test_decorator_calls_handler_on_success(self, mock_mfa):
        """Decorated function is called when all checks pass."""
        mock_mfa.return_value = None  # MFA compliant

        @admin_secure_endpoint()
        async def my_endpoint(self, request, auth_context):
            return HandlerResult(200, "application/json", b'{"ok":true}')

        h = AdminHandler({"user_store": MagicMock()})
        h.ctx["user_store"].get_user_by_id.return_value = MockUser(role="admin")

        mock_request = MagicMock()
        mock_request.remote = "127.0.0.1"
        mock_request.headers = {"User-Agent": "test"}

        result = await my_endpoint(h, mock_request)
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_decorator_returns_403_for_non_admin(self):
        """Non-admin users are rejected."""

        @admin_secure_endpoint()
        async def my_endpoint(self, request, auth_context):
            return HandlerResult(200, "application/json", b'{"ok":true}')

        user_store = MagicMock()
        user_store.get_user_by_id.return_value = MockUser(role="member")
        h = AdminHandler({"user_store": user_store})

        mock_request = MagicMock()
        result = await my_endpoint(h, mock_request)
        assert _status(result) == 403

    @pytest.mark.asyncio
    async def test_decorator_returns_503_when_no_user_store(self):
        """Returns 503 when user store is missing."""

        @admin_secure_endpoint()
        async def my_endpoint(self, request, auth_context):
            return HandlerResult(200, "application/json", b'{"ok":true}')

        h = AdminHandler({})

        mock_request = MagicMock()
        result = await my_endpoint(h, mock_request)
        assert _status(result) == 503

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.admin.handler.enforce_admin_mfa_policy")
    async def test_decorator_returns_403_when_mfa_required(self, mock_mfa):
        """Returns 403 when MFA is required but not active."""
        mock_mfa.return_value = {"reason": "MFA not configured"}

        @admin_secure_endpoint()
        async def my_endpoint(self, request, auth_context):
            return HandlerResult(200, "application/json", b'{"ok":true}')

        user_store = MagicMock()
        user_store.get_user_by_id.return_value = MockUser(role="admin")
        h = AdminHandler({"user_store": user_store})

        mock_request = MagicMock()
        result = await my_endpoint(h, mock_request)
        assert _status(result) == 403
        body = _body(result)
        # code is nested inside the error object
        error_obj = body.get("error", {})
        if isinstance(error_obj, dict):
            assert error_obj.get("code") == "ADMIN_MFA_REQUIRED"
        else:
            assert body.get("code") == "ADMIN_MFA_REQUIRED"

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.admin.handler.enforce_admin_mfa_policy")
    async def test_decorator_checks_rbac_permission(self, mock_mfa):
        """When permission is specified, RBAC is checked."""
        mock_mfa.return_value = None

        @admin_secure_endpoint(permission="admin.test.perm")
        async def my_endpoint(self, request, auth_context):
            return HandlerResult(200, "application/json", b'{"ok":true}')

        user_store = MagicMock()
        user_store.get_user_by_id.return_value = MockUser(role="admin")
        h = AdminHandler({"user_store": user_store})

        # Mock check_permission to raise ForbiddenError
        from aragora.server.handlers.secure import ForbiddenError

        with patch.object(h, "check_permission", side_effect=ForbiddenError("No")):
            mock_request = MagicMock()
            result = await my_endpoint(h, mock_request)
            assert _status(result) == 403

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.admin.handler.enforce_admin_mfa_policy")
    async def test_decorator_with_audit_logging(self, mock_mfa):
        """When audit=True, the audit log is called after handler completes."""
        mock_mfa.return_value = None

        mock_audit_log = AsyncMock()

        # The decorator imports get_audit_log at decoration time, so we must
        # patch before applying the decorator.
        with patch(
            "aragora.observability.immutable_log.get_audit_log",
            return_value=mock_audit_log,
        ):

            @admin_secure_endpoint(audit=True, audit_action="test_action")
            async def my_endpoint(self, request, auth_context):
                return HandlerResult(200, "application/json", b'{"ok":true}')

            user_store = MagicMock()
            user_store.get_user_by_id.return_value = MockUser(role="admin")
            h = AdminHandler({"user_store": user_store})

            mock_request = MagicMock()
            mock_request.remote = "127.0.0.1"
            mock_request.headers = {"User-Agent": "test"}

            result = await my_endpoint(h, mock_request)
            assert _status(result) == 200
            mock_audit_log.append.assert_called_once()
            call_kwargs = mock_audit_log.append.call_args[1]
            assert call_kwargs["event_type"] == "admin.test_action"
            assert call_kwargs["action"] == "test_action"


# ===========================================================================
# Edge cases and integration
# ===========================================================================


class TestHandleEdgeCases:
    def test_get_on_post_only_route_returns_405(self, handler, http):
        """GET on a POST-only nomic route falls through to 405."""
        result = handler.handle("/api/v1/admin/nomic/reset", {}, http(), "GET")
        assert _status(result) == 405

    def test_post_on_get_only_route_returns_405(self, handler, http):
        """POST on a GET-only route falls through to 405."""
        result = handler.handle("/api/v1/admin/stats", {}, http(method="POST"), "POST")
        assert _status(result) == 405

    @patch.object(AdminHandler, "_get_stats")
    def test_handler_inherits_all_mixins(self, mock_stats, handler, http):
        """AdminHandler has all mixin methods."""
        assert hasattr(handler, "_get_stats")
        assert hasattr(handler, "_get_system_metrics")
        assert hasattr(handler, "_get_revenue_stats")
        assert hasattr(handler, "_list_organizations")
        assert hasattr(handler, "_list_users")
        assert hasattr(handler, "_impersonate_user")
        assert hasattr(handler, "_deactivate_user")
        assert hasattr(handler, "_activate_user")
        assert hasattr(handler, "_unlock_user")
        assert hasattr(handler, "_get_nomic_status")
        assert hasattr(handler, "_get_nomic_circuit_breakers")
        assert hasattr(handler, "_reset_nomic_phase")
        assert hasattr(handler, "_pause_nomic")
        assert hasattr(handler, "_resume_nomic")
        assert hasattr(handler, "_reset_nomic_circuit_breakers")

    def test_impersonate_path_requires_user_id_suffix(self, handler, http):
        """POST to /impersonate/ without a user_id suffix still tries to match."""
        # Path /api/v1/admin/impersonate/ ends with empty string after split
        result = handler.handle("/api/v1/admin/impersonate/", {}, http(method="POST"), "POST")
        # Empty string won't match SAFE_ID_PATTERN
        assert _status(result) == 400

    def test_valid_user_id_formats(self, handler, http):
        """Alphanumeric and hyphen/underscore should be valid."""
        with patch.object(AdminHandler, "_impersonate_user") as mock_method:
            mock_method.return_value = HandlerResult(200, "application/json", b"{}")
            for uid in ["user-123", "abc_def", "USR001", "a1b2c3"]:
                result = handler.handle(
                    f"/api/v1/admin/impersonate/{uid}", {}, http(method="POST"), "POST"
                )
                assert _status(result) == 200, f"user_id '{uid}' should be valid"
