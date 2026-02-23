"""
Tests for aragora.server.handlers.admin.users - Admin User and Organization Management.

Comprehensive tests covering:
- GET /api/v1/admin/organizations - List organizations with pagination/filtering
- GET /api/v1/admin/users - List users with pagination/filtering
- POST /api/v1/admin/impersonate/:user_id - Create impersonation token
- POST /api/v1/admin/users/:user_id/deactivate - Deactivate a user
- POST /api/v1/admin/users/:user_id/activate - Activate a user
- POST /api/v1/admin/users/:user_id/unlock - Unlock a locked user account
- RBAC/auth gate tests for every endpoint
- Error handling, edge cases, input validation
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.admin.users import (
    MFA_IMPERSONATION_MAX_AGE_SECONDS,
    PERM_ADMIN_IMPERSONATE,
    PERM_ADMIN_USERS_WRITE,
    UserManagementMixin,
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
    """Minimal mock auth context for the mixin tests."""

    def __init__(self, user_id: str = "admin-001", org_id: str = "org-001"):
        self.user_id = user_id
        self.org_id = org_id


class MockUser:
    """Mock user object matching what the user store returns."""

    def __init__(
        self,
        user_id: str = "user-001",
        email: str = "user@example.com",
        name: str = "Test User",
        role: str = "member",
        org_id: str = "org-001",
        is_active: bool = True,
    ):
        self.id = user_id
        self.email = email
        self.name = name
        self.role = role
        self.org_id = org_id
        self.is_active = is_active

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "email": self.email,
            "name": self.name,
            "role": self.role,
            "org_id": self.org_id,
            "is_active": self.is_active,
        }


class MockOrganization:
    """Mock organization object."""

    def __init__(
        self,
        org_id: str = "org-001",
        name: str = "Test Org",
        tier: str = "professional",
    ):
        self.id = org_id
        self.name = name
        self.tier = tier

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "tier": self.tier,
        }


class MockHTTPHandler:
    """Mock HTTP handler with headers and client_address."""

    def __init__(self, path: str = "/api/v1/admin/users", method: str = "GET"):
        self.path = path
        self.command = method
        self.headers = {"Content-Type": "application/json"}
        self.client_address = ("127.0.0.1", 12345)


class MockSessionManager:
    """Mock session manager for MFA freshness checks."""

    def __init__(self, mfa_fresh: bool = False):
        self._mfa_fresh = mfa_fresh

    def is_session_mfa_fresh(self, user_id: str, token_jti: str, max_age: int) -> bool:
        return self._mfa_fresh


class TestableHandler(UserManagementMixin):
    """Concrete class wiring the mixin to controllable auth stubs."""

    def __init__(
        self,
        ctx: dict[str, Any] | None = None,
        admin_result: tuple[MockAuthContext | None, HandlerResult | None] | None = None,
        rbac_result: HandlerResult | None = None,
        user_store: Any | None = None,
    ):
        self.ctx = ctx or {}
        self._admin_result = admin_result or (MockAuthContext(), None)
        self._rbac_result = rbac_result
        self._user_store = user_store or MagicMock()

    def _require_admin(self, handler: Any) -> tuple[MockAuthContext | None, HandlerResult | None]:
        return self._admin_result

    def _check_rbac_permission(
        self, auth_ctx: Any, permission: str, resource_id: str | None = None
    ) -> HandlerResult | None:
        return self._rbac_result

    def _get_user_store(self) -> Any:
        return self._user_store


# ===========================================================================
# Fixtures
# ===========================================================================


def _make_user_store(
    users: list[MockUser] | None = None,
    orgs: list[MockOrganization] | None = None,
    total_users: int | None = None,
    total_orgs: int | None = None,
):
    """Create a mock user store with controllable return values."""
    store = MagicMock()
    user_list = users or []
    org_list = orgs or []
    store.list_all_users.return_value = (
        user_list,
        total_users if total_users is not None else len(user_list),
    )
    store.list_all_organizations.return_value = (
        org_list,
        total_orgs if total_orgs is not None else len(org_list),
    )
    store.get_user_by_id.return_value = user_list[0] if user_list else None
    store.update_user.return_value = True
    store.record_audit_event.return_value = None
    store.reset_failed_login_attempts.return_value = True
    store.log_audit_event.return_value = None
    return store


@pytest.fixture
def user_store():
    """Default user store with one user and one org."""
    return _make_user_store(
        users=[MockUser()],
        orgs=[MockOrganization()],
    )


@pytest.fixture
def handler(user_store):
    """Handler with a default user store."""
    return TestableHandler(user_store=user_store)


@pytest.fixture
def http():
    """Factory for MockHTTPHandler."""

    def _make(path: str = "/api/v1/admin/users", method: str = "GET"):
        return MockHTTPHandler(path=path, method=method)

    return _make


# ===========================================================================
# Permission Constants
# ===========================================================================


class TestPermissionConstants:
    def test_admin_users_write_value(self):
        assert PERM_ADMIN_USERS_WRITE == "admin:users:write"

    def test_admin_impersonate_value(self):
        assert PERM_ADMIN_IMPERSONATE == "admin:impersonate"

    def test_mfa_max_age_value(self):
        assert MFA_IMPERSONATION_MAX_AGE_SECONDS == 300


# ===========================================================================
# Module Exports
# ===========================================================================


class TestModuleExports:
    def test_all_exports(self):
        from aragora.server.handlers.admin.users import __all__

        assert "UserManagementMixin" in __all__
        assert "PERM_ADMIN_USERS_WRITE" in __all__
        assert "PERM_ADMIN_IMPERSONATE" in __all__

    def test_mixin_is_importable(self):
        assert UserManagementMixin is not None

    def test_mixin_has_expected_methods(self):
        assert hasattr(UserManagementMixin, "_list_organizations")
        assert hasattr(UserManagementMixin, "_list_users")
        assert hasattr(UserManagementMixin, "_impersonate_user")
        assert hasattr(UserManagementMixin, "_deactivate_user")
        assert hasattr(UserManagementMixin, "_activate_user")
        assert hasattr(UserManagementMixin, "_unlock_user")


# ===========================================================================
# GET /api/v1/admin/organizations
# ===========================================================================


class TestListOrganizations:
    """Tests for _list_organizations."""

    def test_returns_401_when_admin_check_fails(self, http):
        h = TestableHandler(admin_result=(None, error_response("Unauthorized", 401)))
        result = h._list_organizations(http(), {})
        assert _status(result) == 401

    def test_returns_403_when_rbac_denied(self, http):
        h = TestableHandler(rbac_result=error_response("Forbidden", 403))
        result = h._list_organizations(http(), {})
        assert _status(result) == 403

    def test_success_empty_list(self, http):
        store = _make_user_store(orgs=[], total_orgs=0)
        h = TestableHandler(user_store=store)
        result = h._list_organizations(http(), {})
        assert _status(result) == 200
        data = _body(result)
        assert data["organizations"] == []
        assert data["total"] == 0

    def test_success_with_organizations(self, http):
        orgs = [
            MockOrganization("org-1", "Alpha Corp", "enterprise"),
            MockOrganization("org-2", "Beta Inc", "starter"),
        ]
        store = _make_user_store(orgs=orgs, total_orgs=2)
        h = TestableHandler(user_store=store)
        result = h._list_organizations(http(), {})
        data = _body(result)
        assert len(data["organizations"]) == 2
        assert data["organizations"][0]["name"] == "Alpha Corp"
        assert data["organizations"][1]["tier"] == "starter"
        assert data["total"] == 2

    def test_default_pagination(self, http):
        store = _make_user_store(orgs=[], total_orgs=0)
        h = TestableHandler(user_store=store)
        result = h._list_organizations(http(), {})
        data = _body(result)
        assert data["limit"] == 50
        assert data["offset"] == 0

    def test_custom_pagination(self, http):
        store = _make_user_store(orgs=[], total_orgs=0)
        h = TestableHandler(user_store=store)
        result = h._list_organizations(http(), {"limit": "25", "offset": "10"})
        data = _body(result)
        assert data["limit"] == 25
        assert data["offset"] == 10

    def test_limit_capped_at_100(self, http):
        store = _make_user_store(orgs=[], total_orgs=0)
        h = TestableHandler(user_store=store)
        result = h._list_organizations(http(), {"limit": "200"})
        data = _body(result)
        assert data["limit"] == 100

    def test_tier_filter_passed(self, http):
        store = _make_user_store(orgs=[], total_orgs=0)
        h = TestableHandler(user_store=store)
        h._list_organizations(http(), {"tier": "enterprise"})
        store.list_all_organizations.assert_called_once_with(
            limit=50, offset=0, tier_filter="enterprise"
        )

    def test_tier_filter_none_by_default(self, http):
        store = _make_user_store(orgs=[], total_orgs=0)
        h = TestableHandler(user_store=store)
        h._list_organizations(http(), {})
        store.list_all_organizations.assert_called_once_with(limit=50, offset=0, tier_filter=None)

    def test_pagination_values_in_response(self, http):
        store = _make_user_store(orgs=[], total_orgs=150)
        h = TestableHandler(user_store=store)
        result = h._list_organizations(http(), {"limit": "30", "offset": "60"})
        data = _body(result)
        assert data["total"] == 150
        assert data["limit"] == 30
        assert data["offset"] == 60


# ===========================================================================
# GET /api/v1/admin/users
# ===========================================================================


class TestListUsers:
    """Tests for _list_users."""

    def test_returns_401_when_admin_check_fails(self, http):
        h = TestableHandler(admin_result=(None, error_response("Unauthorized", 401)))
        result = h._list_users(http(), {})
        assert _status(result) == 401

    def test_returns_403_when_rbac_denied(self, http):
        h = TestableHandler(rbac_result=error_response("Forbidden", 403))
        result = h._list_users(http(), {})
        assert _status(result) == 403

    def test_success_empty_list(self, http):
        store = _make_user_store(users=[], total_users=0)
        h = TestableHandler(user_store=store)
        result = h._list_users(http(), {})
        assert _status(result) == 200
        data = _body(result)
        assert data["users"] == []
        assert data["total"] == 0

    def test_success_with_users(self, http):
        users = [
            MockUser("u1", "alice@test.com", "Alice", "admin"),
            MockUser("u2", "bob@test.com", "Bob", "member"),
        ]
        store = _make_user_store(users=users, total_users=2)
        h = TestableHandler(user_store=store)
        result = h._list_users(http(), {})
        data = _body(result)
        assert len(data["users"]) == 2
        assert data["total"] == 2

    def test_default_pagination(self, http):
        store = _make_user_store(users=[], total_users=0)
        h = TestableHandler(user_store=store)
        result = h._list_users(http(), {})
        data = _body(result)
        assert data["limit"] == 50
        assert data["offset"] == 0

    def test_custom_pagination(self, http):
        store = _make_user_store(users=[], total_users=0)
        h = TestableHandler(user_store=store)
        result = h._list_users(http(), {"limit": "20", "offset": "5"})
        data = _body(result)
        assert data["limit"] == 20
        assert data["offset"] == 5

    def test_limit_capped_at_100(self, http):
        store = _make_user_store(users=[], total_users=0)
        h = TestableHandler(user_store=store)
        result = h._list_users(http(), {"limit": "500"})
        data = _body(result)
        assert data["limit"] == 100

    def test_org_id_filter(self, http):
        store = _make_user_store(users=[], total_users=0)
        h = TestableHandler(user_store=store)
        h._list_users(http(), {"org_id": "org-42"})
        store.list_all_users.assert_called_once_with(
            limit=50, offset=0, org_id_filter="org-42", role_filter=None, active_only=False
        )

    def test_role_filter(self, http):
        store = _make_user_store(users=[], total_users=0)
        h = TestableHandler(user_store=store)
        h._list_users(http(), {"role": "admin"})
        store.list_all_users.assert_called_once_with(
            limit=50, offset=0, org_id_filter=None, role_filter="admin", active_only=False
        )

    def test_active_only_true(self, http):
        store = _make_user_store(users=[], total_users=0)
        h = TestableHandler(user_store=store)
        h._list_users(http(), {"active_only": "true"})
        store.list_all_users.assert_called_once_with(
            limit=50, offset=0, org_id_filter=None, role_filter=None, active_only=True
        )

    def test_active_only_false_by_default(self, http):
        store = _make_user_store(users=[], total_users=0)
        h = TestableHandler(user_store=store)
        h._list_users(http(), {})
        store.list_all_users.assert_called_once_with(
            limit=50, offset=0, org_id_filter=None, role_filter=None, active_only=False
        )

    def test_active_only_case_insensitive(self, http):
        store = _make_user_store(users=[], total_users=0)
        h = TestableHandler(user_store=store)
        h._list_users(http(), {"active_only": "TRUE"})
        store.list_all_users.assert_called_once_with(
            limit=50, offset=0, org_id_filter=None, role_filter=None, active_only=True
        )

    def test_sanitizes_user_response(self, http):
        """Verify sensitive fields are stripped from user data."""
        user = MockUser()
        # Add sensitive fields to the to_dict output
        original_to_dict = user.to_dict

        def to_dict_with_sensitive():
            d = original_to_dict()
            d["password_hash"] = "secret_hash"
            d["api_key"] = "secret_key"
            return d

        user.to_dict = to_dict_with_sensitive
        store = _make_user_store(users=[user], total_users=1)
        h = TestableHandler(user_store=store)
        result = h._list_users(http(), {})
        data = _body(result)
        assert len(data["users"]) == 1
        # Sanitization should remove sensitive fields
        assert "password_hash" not in data["users"][0]
        assert "api_key" not in data["users"][0]

    def test_combined_filters(self, http):
        store = _make_user_store(users=[], total_users=0)
        h = TestableHandler(user_store=store)
        h._list_users(
            http(),
            {
                "org_id": "org-99",
                "role": "viewer",
                "active_only": "true",
                "limit": "10",
                "offset": "20",
            },
        )
        store.list_all_users.assert_called_once_with(
            limit=10, offset=20, org_id_filter="org-99", role_filter="viewer", active_only=True
        )


# ===========================================================================
# POST /api/v1/admin/impersonate/:user_id
# ===========================================================================


class TestImpersonateUser:
    """Tests for _impersonate_user."""

    def test_returns_401_when_admin_check_fails(self, http):
        h = TestableHandler(admin_result=(None, error_response("Unauthorized", 401)))
        result = h._impersonate_user(http(), "user-001")
        assert _status(result) == 401

    def test_returns_403_when_rbac_denied(self, http):
        h = TestableHandler(rbac_result=error_response("Forbidden", 403))
        result = h._impersonate_user(http(), "user-001")
        assert _status(result) == 403

    def test_invalid_user_id_format(self, http):
        h = TestableHandler()
        result = h._impersonate_user(http(), "../etc/passwd")
        assert _status(result) == 400
        assert "Invalid" in _body(result)["error"]

    def test_empty_user_id(self, http):
        h = TestableHandler()
        result = h._impersonate_user(http(), "")
        assert _status(result) == 400

    def test_user_id_too_long(self, http):
        h = TestableHandler()
        result = h._impersonate_user(http(), "a" * 100)
        assert _status(result) == 400

    def test_returns_403_when_mfa_not_fresh(self, http):
        """MFA verification required for impersonation."""
        target = MockUser("target-001", "target@test.com")
        store = _make_user_store(users=[target])
        h = TestableHandler(user_store=store)
        # No session manager -> mfa_fresh stays False
        result = h._impersonate_user(http(), "target-001")
        assert _status(result) == 403
        data = _body(result)
        # error_response with code= uses structured format: {"error": {"message": ..., "code": ...}}
        err = data["error"]
        assert "MFA" in err["message"]
        assert err["code"] == "MFA_STEP_UP_REQUIRED"

    def test_returns_404_when_user_not_found(self, http):
        """Target user must exist."""
        store = _make_user_store(users=[])
        store.get_user_by_id.return_value = None
        # Need MFA fresh to get past the MFA check
        session_mgr = MockSessionManager(mfa_fresh=True)
        mock_http = http()
        mock_http.headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer test-token-123",
        }
        mock_http.ctx = {"session_manager": session_mgr}

        h = TestableHandler(user_store=store)
        with patch(
            "aragora.server.handlers.admin.users._get_session_manager_from_handler",
            return_value=session_mgr,
        ):
            result = h._impersonate_user(mock_http, "nonexistent-user")
        assert _status(result) == 404
        assert "not found" in _body(result)["error"]

    def test_success_returns_token(self, http):
        """Successful impersonation returns a token."""
        target = MockUser("target-001", "target@test.com", "Target User", "member", "org-001")
        store = _make_user_store(users=[target])
        session_mgr = MockSessionManager(mfa_fresh=True)
        mock_http = http()
        mock_http.headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer test-token-123",
        }

        h = TestableHandler(user_store=store)
        # The function does: from aragora.server.handlers.admin import handler as admin_handler
        # Then calls admin_handler.create_access_token(...)
        # Patch directly on the handler module that gets imported
        with (
            patch(
                "aragora.server.handlers.admin.users._get_session_manager_from_handler",
                return_value=session_mgr,
            ),
            patch(
                "aragora.server.handlers.admin.handler.create_access_token",
                return_value="impersonation-jwt-token",
            ),
        ):
            result = h._impersonate_user(mock_http, "target-001")

        assert _status(result) == 200
        data = _body(result)
        assert data["token"] == "impersonation-jwt-token"
        assert data["expires_in"] == 3600
        assert data["target_user"]["id"] == "target-001"
        assert data["target_user"]["email"] == "target@test.com"
        assert data["target_user"]["name"] == "Target User"
        assert data["target_user"]["role"] == "member"
        assert "warning" in data

    def test_mfa_check_uses_bearer_token(self, http):
        """MFA freshness is checked with the token JTI derived from Authorization header."""
        target = MockUser("target-001", "target@test.com")
        store = _make_user_store(users=[target])
        session_mgr = MagicMock()
        session_mgr.is_session_mfa_fresh.return_value = True
        mock_http = http()
        mock_http.headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer my-test-token",
        }

        h = TestableHandler(user_store=store)
        with (
            patch(
                "aragora.server.handlers.admin.users._get_session_manager_from_handler",
                return_value=session_mgr,
            ),
            patch(
                "aragora.server.handlers.admin.handler.create_access_token",
                return_value="token",
            ),
        ):
            h._impersonate_user(mock_http, "target-001")

        # Verify session manager was called with correct args
        session_mgr.is_session_mfa_fresh.assert_called_once()
        call_args = session_mgr.is_session_mfa_fresh.call_args
        assert call_args[0][0] == "admin-001"  # user_id from auth_ctx
        assert call_args[0][2] == MFA_IMPERSONATION_MAX_AGE_SECONDS

    def test_mfa_check_exception_treated_as_not_fresh(self, http):
        """If MFA check throws, it should be treated as not fresh."""
        target = MockUser("target-001", "target@test.com")
        store = _make_user_store(users=[target])
        session_mgr = MagicMock()
        session_mgr.is_session_mfa_fresh.side_effect = RuntimeError("Redis down")
        mock_http = http()
        mock_http.headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer test-token",
        }

        h = TestableHandler(user_store=store)
        with patch(
            "aragora.server.handlers.admin.users._get_session_manager_from_handler",
            return_value=session_mgr,
        ):
            result = h._impersonate_user(mock_http, "target-001")

        assert _status(result) == 403
        err = _body(result)["error"]
        assert "MFA" in err["message"]

    def test_no_authorization_header_means_no_mfa(self, http):
        """Without Authorization header, MFA check fails gracefully."""
        store = _make_user_store(users=[MockUser()])
        session_mgr = MockSessionManager(mfa_fresh=True)
        mock_http = http()
        mock_http.headers = {"Content-Type": "application/json"}

        h = TestableHandler(user_store=store)
        with patch(
            "aragora.server.handlers.admin.users._get_session_manager_from_handler",
            return_value=session_mgr,
        ):
            result = h._impersonate_user(mock_http, "user-001")

        # Without "Bearer " prefix, mfa_fresh stays False
        assert _status(result) == 403

    def test_audit_event_failure_does_not_break_response(self, http):
        """Audit event recording failure should not prevent success."""
        target = MockUser("target-001", "target@test.com")
        store = _make_user_store(users=[target])
        store.record_audit_event.side_effect = RuntimeError("Audit DB down")
        session_mgr = MockSessionManager(mfa_fresh=True)
        mock_http = http()
        mock_http.headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer test-token-123",
        }

        h = TestableHandler(user_store=store)
        with (
            patch(
                "aragora.server.handlers.admin.users._get_session_manager_from_handler",
                return_value=session_mgr,
            ),
            patch(
                "aragora.server.handlers.admin.handler.create_access_token",
                return_value="token",
            ),
        ):
            result = h._impersonate_user(mock_http, "target-001")

        # Should still succeed despite audit failure
        assert _status(result) == 200

    def test_no_session_manager_means_no_mfa(self, http):
        """If no session manager is available, MFA freshness check fails."""
        store = _make_user_store(users=[MockUser()])
        mock_http = http()

        h = TestableHandler(user_store=store)
        with patch(
            "aragora.server.handlers.admin.users._get_session_manager_from_handler",
            return_value=None,
        ):
            result = h._impersonate_user(mock_http, "user-001")

        assert _status(result) == 403

    def test_special_characters_in_user_id_rejected(self, http):
        h = TestableHandler()
        result = h._impersonate_user(http(), "user;DROP TABLE")
        assert _status(result) == 400


# ===========================================================================
# POST /api/v1/admin/users/:user_id/deactivate
# ===========================================================================


class TestDeactivateUser:
    """Tests for _deactivate_user."""

    def test_returns_401_when_admin_check_fails(self, http):
        h = TestableHandler(admin_result=(None, error_response("Unauthorized", 401)))
        result = h._deactivate_user(http(), "user-001")
        assert _status(result) == 401

    def test_returns_403_when_rbac_denied(self, http):
        h = TestableHandler(rbac_result=error_response("Forbidden", 403))
        result = h._deactivate_user(http(), "user-001")
        assert _status(result) == 403

    def test_invalid_user_id_format(self, http):
        h = TestableHandler()
        result = h._deactivate_user(http(), "../bad/path")
        assert _status(result) == 400

    def test_empty_user_id(self, http):
        h = TestableHandler()
        result = h._deactivate_user(http(), "")
        assert _status(result) == 400

    def test_user_not_found(self, http):
        store = _make_user_store(users=[])
        store.get_user_by_id.return_value = None
        h = TestableHandler(user_store=store)
        result = h._deactivate_user(http(), "nonexistent")
        assert _status(result) == 404
        assert "not found" in _body(result)["error"]

    def test_cannot_deactivate_yourself(self, http):
        """Admin cannot deactivate their own account."""
        admin_user = MockUser("admin-001", "admin@test.com", "Admin", "admin")
        store = _make_user_store(users=[admin_user])
        h = TestableHandler(user_store=store)
        result = h._deactivate_user(http(), "admin-001")
        assert _status(result) == 400
        assert "yourself" in _body(result)["error"].lower()

    def test_success(self, http):
        target = MockUser("user-002", "user@test.com", "User", "member")
        store = _make_user_store(users=[target])
        h = TestableHandler(user_store=store)
        with (
            patch("aragora.server.handlers.admin.users.audit_admin"),
            patch("aragora.server.handlers.admin.users.emit_handler_event"),
        ):
            result = h._deactivate_user(http(), "user-002")

        assert _status(result) == 200
        data = _body(result)
        assert data["success"] is True
        assert data["user_id"] == "user-002"
        assert data["is_active"] is False

    def test_calls_update_user(self, http):
        target = MockUser("user-002", "user@test.com")
        store = _make_user_store(users=[target])
        h = TestableHandler(user_store=store)
        with (
            patch("aragora.server.handlers.admin.users.audit_admin"),
            patch("aragora.server.handlers.admin.users.emit_handler_event"),
        ):
            h._deactivate_user(http(), "user-002")

        store.update_user.assert_called_once_with("user-002", is_active=False)

    def test_audit_called(self, http):
        target = MockUser("user-002", "user@test.com", role="member")
        store = _make_user_store(users=[target])
        h = TestableHandler(user_store=store)
        with (
            patch("aragora.server.handlers.admin.users.audit_admin") as mock_audit,
            patch("aragora.server.handlers.admin.users.emit_handler_event"),
        ):
            h._deactivate_user(http(), "user-002")

        mock_audit.assert_called_once()
        kw = mock_audit.call_args[1]
        assert kw["admin_id"] == "admin-001"
        assert kw["action"] == "deactivate_user"
        assert kw["target_type"] == "user"
        assert kw["target_id"] == "user-002"
        assert kw["target_email"] == "user@test.com"

    def test_emits_handler_event(self, http):
        target = MockUser("user-002", "user@test.com")
        store = _make_user_store(users=[target])
        h = TestableHandler(user_store=store)
        with (
            patch("aragora.server.handlers.admin.users.audit_admin"),
            patch("aragora.server.handlers.admin.users.emit_handler_event") as mock_emit,
        ):
            h._deactivate_user(http(), "user-002")

        mock_emit.assert_called_once()
        args = mock_emit.call_args
        assert args[0][0] == "admin"
        assert args[0][2]["action"] == "deactivate_user"
        assert args[0][2]["target_user_id"] == "user-002"

    def test_rbac_permission_checked_with_target_user_id(self, http):
        """RBAC permission check includes the target user ID as resource."""
        target = MockUser("user-002", "user@test.com")
        store = _make_user_store(users=[target])
        rbac_calls = []

        class TrackingHandler(TestableHandler):
            def _check_rbac_permission(self, auth_ctx, permission, resource_id=None):
                rbac_calls.append((permission, resource_id))
                return None

        h = TrackingHandler(user_store=store)
        with (
            patch("aragora.server.handlers.admin.users.audit_admin"),
            patch("aragora.server.handlers.admin.users.emit_handler_event"),
        ):
            h._deactivate_user(http(), "user-002")

        assert (PERM_ADMIN_USERS_WRITE, "user-002") in rbac_calls


# ===========================================================================
# POST /api/v1/admin/users/:user_id/activate
# ===========================================================================


class TestActivateUser:
    """Tests for _activate_user."""

    def test_returns_401_when_admin_check_fails(self, http):
        h = TestableHandler(admin_result=(None, error_response("Unauthorized", 401)))
        result = h._activate_user(http(), "user-001")
        assert _status(result) == 401

    def test_returns_403_when_rbac_denied(self, http):
        h = TestableHandler(rbac_result=error_response("Forbidden", 403))
        result = h._activate_user(http(), "user-001")
        assert _status(result) == 403

    def test_invalid_user_id_format(self, http):
        h = TestableHandler()
        result = h._activate_user(http(), "../../etc")
        assert _status(result) == 400

    def test_empty_user_id(self, http):
        h = TestableHandler()
        result = h._activate_user(http(), "")
        assert _status(result) == 400

    def test_user_not_found(self, http):
        store = _make_user_store(users=[])
        store.get_user_by_id.return_value = None
        h = TestableHandler(user_store=store)
        result = h._activate_user(http(), "nonexistent")
        assert _status(result) == 404
        assert "not found" in _body(result)["error"]

    def test_success(self, http):
        target = MockUser("user-002", "user@test.com", is_active=False)
        store = _make_user_store(users=[target])
        h = TestableHandler(user_store=store)
        with patch("aragora.server.handlers.admin.users.audit_admin"):
            result = h._activate_user(http(), "user-002")

        assert _status(result) == 200
        data = _body(result)
        assert data["success"] is True
        assert data["user_id"] == "user-002"
        assert data["is_active"] is True

    def test_calls_update_user(self, http):
        target = MockUser("user-002", "user@test.com", is_active=False)
        store = _make_user_store(users=[target])
        h = TestableHandler(user_store=store)
        with patch("aragora.server.handlers.admin.users.audit_admin"):
            h._activate_user(http(), "user-002")

        store.update_user.assert_called_once_with("user-002", is_active=True)

    def test_audit_called(self, http):
        target = MockUser("user-002", "user@test.com")
        store = _make_user_store(users=[target])
        h = TestableHandler(user_store=store)
        with patch("aragora.server.handlers.admin.users.audit_admin") as mock_audit:
            h._activate_user(http(), "user-002")

        mock_audit.assert_called_once()
        kw = mock_audit.call_args[1]
        assert kw["admin_id"] == "admin-001"
        assert kw["action"] == "activate_user"
        assert kw["target_type"] == "user"
        assert kw["target_id"] == "user-002"
        assert kw["target_email"] == "user@test.com"

    def test_rbac_permission_checked_with_target_user_id(self, http):
        target = MockUser("user-002", "user@test.com")
        store = _make_user_store(users=[target])
        rbac_calls = []

        class TrackingHandler(TestableHandler):
            def _check_rbac_permission(self, auth_ctx, permission, resource_id=None):
                rbac_calls.append((permission, resource_id))
                return None

        h = TrackingHandler(user_store=store)
        with patch("aragora.server.handlers.admin.users.audit_admin"):
            h._activate_user(http(), "user-002")

        assert (PERM_ADMIN_USERS_WRITE, "user-002") in rbac_calls

    def test_activating_already_active_user_succeeds(self, http):
        """Activating an already active user should succeed (idempotent)."""
        target = MockUser("user-002", "user@test.com", is_active=True)
        store = _make_user_store(users=[target])
        h = TestableHandler(user_store=store)
        with patch("aragora.server.handlers.admin.users.audit_admin"):
            result = h._activate_user(http(), "user-002")

        assert _status(result) == 200
        assert _body(result)["is_active"] is True


# ===========================================================================
# POST /api/v1/admin/users/:user_id/unlock
# ===========================================================================


class TestUnlockUser:
    """Tests for _unlock_user."""

    def test_returns_401_when_admin_check_fails(self, http):
        h = TestableHandler(admin_result=(None, error_response("Unauthorized", 401)))
        result = h._unlock_user(http(), "user-001")
        assert _status(result) == 401

    def test_returns_403_when_rbac_denied(self, http):
        h = TestableHandler(rbac_result=error_response("Forbidden", 403))
        result = h._unlock_user(http(), "user-001")
        assert _status(result) == 403

    def test_invalid_user_id_format(self, http):
        h = TestableHandler()
        result = h._unlock_user(http(), "../../etc")
        assert _status(result) == 400

    def test_empty_user_id(self, http):
        h = TestableHandler()
        result = h._unlock_user(http(), "")
        assert _status(result) == 400

    def test_user_not_found(self, http):
        store = _make_user_store(users=[])
        store.get_user_by_id.return_value = None
        h = TestableHandler(user_store=store)
        result = h._unlock_user(http(), "nonexistent")
        assert _status(result) == 404
        assert "not found" in _body(result)["error"]

    def test_success(self, http):
        target = MockUser("user-002", "locked@test.com")
        store = _make_user_store(users=[target])
        h = TestableHandler(user_store=store)
        mock_tracker = MagicMock()
        mock_tracker.get_info.return_value = {"locked": True, "attempts": 10}
        mock_tracker.admin_unlock.return_value = True

        with patch(
            "aragora.server.handlers.admin.users.get_lockout_tracker",
            return_value=mock_tracker,
        ):
            result = h._unlock_user(http(), "user-002")

        assert _status(result) == 200
        data = _body(result)
        assert data["success"] is True
        assert data["user_id"] == "user-002"
        assert data["email"] == "locked@test.com"
        assert data["lockout_cleared"] is True
        assert data["previous_lockout_info"] == {"locked": True, "attempts": 10}
        assert "locked@test.com" in data["message"]

    def test_calls_lockout_tracker(self, http):
        target = MockUser("user-002", "locked@test.com")
        store = _make_user_store(users=[target])
        h = TestableHandler(user_store=store)
        mock_tracker = MagicMock()
        mock_tracker.get_info.return_value = {}
        mock_tracker.admin_unlock.return_value = True

        with patch(
            "aragora.server.handlers.admin.users.get_lockout_tracker",
            return_value=mock_tracker,
        ):
            h._unlock_user(http(), "user-002")

        mock_tracker.get_info.assert_called_once_with(email="locked@test.com")
        mock_tracker.admin_unlock.assert_called_once_with(
            email="locked@test.com", user_id="user-002"
        )

    def test_clears_db_lockout_when_supported(self, http):
        target = MockUser("user-002", "locked@test.com")
        store = _make_user_store(users=[target])
        h = TestableHandler(user_store=store)
        mock_tracker = MagicMock()
        mock_tracker.get_info.return_value = {}
        mock_tracker.admin_unlock.return_value = False

        with patch(
            "aragora.server.handlers.admin.users.get_lockout_tracker",
            return_value=mock_tracker,
        ):
            result = h._unlock_user(http(), "user-002")

        store.reset_failed_login_attempts.assert_called_once_with("locked@test.com")
        # lockout_cleared should be True because db_cleared is True
        assert _body(result)["lockout_cleared"] is True

    def test_lockout_cleared_false_when_neither_cleared(self, http):
        target = MockUser("user-002", "locked@test.com")
        store = _make_user_store(users=[target])
        store.reset_failed_login_attempts.return_value = False
        h = TestableHandler(user_store=store)
        mock_tracker = MagicMock()
        mock_tracker.get_info.return_value = {}
        mock_tracker.admin_unlock.return_value = False

        with patch(
            "aragora.server.handlers.admin.users.get_lockout_tracker",
            return_value=mock_tracker,
        ):
            result = h._unlock_user(http(), "user-002")

        assert _body(result)["lockout_cleared"] is False

    def test_db_clear_skipped_when_not_supported(self, http):
        """If user store doesn't have reset_failed_login_attempts, skip it."""
        target = MockUser("user-002", "locked@test.com")
        store = _make_user_store(users=[target])
        # Remove the method to simulate an older store
        del store.reset_failed_login_attempts
        h = TestableHandler(user_store=store)
        mock_tracker = MagicMock()
        mock_tracker.get_info.return_value = {}
        mock_tracker.admin_unlock.return_value = True

        with patch(
            "aragora.server.handlers.admin.users.get_lockout_tracker",
            return_value=mock_tracker,
        ):
            result = h._unlock_user(http(), "user-002")

        assert _status(result) == 200
        assert _body(result)["lockout_cleared"] is True

    def test_audit_event_logged_when_supported(self, http):
        target = MockUser("user-002", "locked@test.com")
        store = _make_user_store(users=[target])
        h = TestableHandler(user_store=store)
        mock_tracker = MagicMock()
        mock_tracker.get_info.return_value = {"locked": True}
        mock_tracker.admin_unlock.return_value = True

        with patch(
            "aragora.server.handlers.admin.users.get_lockout_tracker",
            return_value=mock_tracker,
        ):
            h._unlock_user(http(), "user-002")

        store.log_audit_event.assert_called_once()
        call_kwargs = store.log_audit_event.call_args[1]
        assert call_kwargs["action"] == "admin_unlock_user"
        assert call_kwargs["resource_type"] == "user"
        assert call_kwargs["resource_id"] == "user-002"
        assert call_kwargs["user_id"] == "admin-001"

    def test_audit_event_failure_does_not_break_response(self, http):
        target = MockUser("user-002", "locked@test.com")
        store = _make_user_store(users=[target])
        store.log_audit_event.side_effect = RuntimeError("Audit DB down")
        h = TestableHandler(user_store=store)
        mock_tracker = MagicMock()
        mock_tracker.get_info.return_value = {}
        mock_tracker.admin_unlock.return_value = True

        with patch(
            "aragora.server.handlers.admin.users.get_lockout_tracker",
            return_value=mock_tracker,
        ):
            result = h._unlock_user(http(), "user-002")

        assert _status(result) == 200

    def test_audit_event_skipped_when_not_supported(self, http):
        """If user store doesn't have log_audit_event, skip it."""
        target = MockUser("user-002", "locked@test.com")
        store = _make_user_store(users=[target])
        del store.log_audit_event
        h = TestableHandler(user_store=store)
        mock_tracker = MagicMock()
        mock_tracker.get_info.return_value = {}
        mock_tracker.admin_unlock.return_value = True

        with patch(
            "aragora.server.handlers.admin.users.get_lockout_tracker",
            return_value=mock_tracker,
        ):
            result = h._unlock_user(http(), "user-002")

        # Should succeed without audit logging
        assert _status(result) == 200

    def test_rbac_permission_checked_with_target_user_id(self, http):
        target = MockUser("user-002", "locked@test.com")
        store = _make_user_store(users=[target])
        rbac_calls = []

        class TrackingHandler(TestableHandler):
            def _check_rbac_permission(self, auth_ctx, permission, resource_id=None):
                rbac_calls.append((permission, resource_id))
                return None

        h = TrackingHandler(user_store=store)
        mock_tracker = MagicMock()
        mock_tracker.get_info.return_value = {}
        mock_tracker.admin_unlock.return_value = True

        with patch(
            "aragora.server.handlers.admin.users.get_lockout_tracker",
            return_value=mock_tracker,
        ):
            h._unlock_user(http(), "user-002")

        assert (PERM_ADMIN_USERS_WRITE, "user-002") in rbac_calls

    def test_client_address_used_for_audit_ip(self, http):
        target = MockUser("user-002", "locked@test.com")
        store = _make_user_store(users=[target])
        h = TestableHandler(user_store=store)
        mock_http = http()
        mock_http.client_address = ("10.0.0.5", 54321)
        mock_tracker = MagicMock()
        mock_tracker.get_info.return_value = {}
        mock_tracker.admin_unlock.return_value = True

        with patch(
            "aragora.server.handlers.admin.users.get_lockout_tracker",
            return_value=mock_tracker,
        ):
            h._unlock_user(mock_http, "user-002")

        call_kwargs = store.log_audit_event.call_args[1]
        assert call_kwargs["ip_address"] == "10.0.0.5"

    def test_missing_client_address_defaults_to_unknown(self, http):
        target = MockUser("user-002", "locked@test.com")
        store = _make_user_store(users=[target])
        h = TestableHandler(user_store=store)
        mock_http = http()
        # Remove client_address
        del mock_http.client_address
        mock_tracker = MagicMock()
        mock_tracker.get_info.return_value = {}
        mock_tracker.admin_unlock.return_value = True

        with patch(
            "aragora.server.handlers.admin.users.get_lockout_tracker",
            return_value=mock_tracker,
        ):
            h._unlock_user(mock_http, "user-002")

        call_kwargs = store.log_audit_event.call_args[1]
        assert call_kwargs["ip_address"] == "unknown"


# ===========================================================================
# Input validation edge cases
# ===========================================================================


class TestInputValidation:
    """Cross-endpoint input validation tests."""

    @pytest.mark.parametrize(
        "user_id",
        [
            "valid-id",
            "user_001",
            "abc123",
            "A" * 64,
        ],
    )
    def test_valid_user_ids_accepted(self, http, user_id):
        store = _make_user_store(users=[MockUser(user_id)])
        h = TestableHandler(user_store=store)
        with (
            patch("aragora.server.handlers.admin.users.audit_admin"),
            patch("aragora.server.handlers.admin.users.emit_handler_event"),
        ):
            result = h._deactivate_user(http(), user_id)
        # Should not get 400 (might get 400 for self-deactivation, but not for invalid ID)
        # user_id != "admin-001" so it won't be self-deactivation
        assert _status(result) != 400 or "yourself" in _body(result).get("error", "")

    @pytest.mark.parametrize(
        "user_id",
        [
            "",
            "../traversal",
            "a" * 65,
            "user id with spaces",
            "user;injection",
            "user<script>",
        ],
    )
    def test_invalid_user_ids_rejected_deactivate(self, http, user_id):
        h = TestableHandler()
        result = h._deactivate_user(http(), user_id)
        assert _status(result) == 400

    @pytest.mark.parametrize(
        "user_id",
        [
            "",
            "../traversal",
            "a" * 65,
        ],
    )
    def test_invalid_user_ids_rejected_activate(self, http, user_id):
        h = TestableHandler()
        result = h._activate_user(http(), user_id)
        assert _status(result) == 400

    @pytest.mark.parametrize(
        "user_id",
        [
            "",
            "../traversal",
            "a" * 65,
        ],
    )
    def test_invalid_user_ids_rejected_unlock(self, http, user_id):
        h = TestableHandler()
        result = h._unlock_user(http(), user_id)
        assert _status(result) == 400

    @pytest.mark.parametrize(
        "user_id",
        [
            "",
            "../traversal",
            "a" * 65,
        ],
    )
    def test_invalid_user_ids_rejected_impersonate(self, http, user_id):
        h = TestableHandler()
        result = h._impersonate_user(http(), user_id)
        assert _status(result) == 400
