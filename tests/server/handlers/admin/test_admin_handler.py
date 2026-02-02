"""
Tests for aragora.server.handlers.admin.admin - Admin API Handlers.

Tests cover:
- AdminHandler initialization and routing
- Authentication and admin role checks
- RBAC permission checks
- Organization listing
- User listing, activation, deactivation, unlock
- Stats and metrics endpoints
- User impersonation
- Nomic admin endpoints (status, pause, resume, reset)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from aragora.server.handlers.admin.admin import (
    AdminHandler,
    ADMIN_ROLES,
    admin_secure_endpoint,
)
from aragora.server.handlers.utils.responses import HandlerResult


def get_response_data(result: HandlerResult) -> dict:
    """Extract JSON data from HandlerResult."""
    if result and result.body:
        return json.loads(result.body.decode("utf-8"))
    return {}


# ===========================================================================
# Mock Classes
# ===========================================================================


class MockUser:
    """Mock user object for testing."""

    def __init__(
        self,
        id: str = "user-001",
        email: str = "admin@example.com",
        name: str = "Test Admin",
        role: str = "admin",
        org_id: str = "org-001",
        is_active: bool = True,
        mfa_enabled: bool = True,
    ):
        self.id = id
        self.email = email
        self.name = name
        self.role = role
        self.org_id = org_id
        self.is_active = is_active
        self.mfa_enabled = mfa_enabled
        self.mfa_secret = "secret" if mfa_enabled else None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "email": self.email,
            "name": self.name,
            "role": self.role,
            "org_id": self.org_id,
            "is_active": self.is_active,
            "mfa_enabled": self.mfa_enabled,
        }


class MockOrganization:
    """Mock organization object for testing."""

    def __init__(
        self,
        id: str = "org-001",
        name: str = "Test Org",
        tier: str = "professional",
    ):
        self.id = id
        self.name = name
        self.tier = tier

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "tier": self.tier,
        }


class MockAuthContext:
    """Mock authentication context."""

    def __init__(
        self,
        user_id: str = "user-001",
        is_authenticated: bool = True,
        org_id: str = "org-001",
        workspace_id: str = "ws-001",
    ):
        self.user_id = user_id
        self.is_authenticated = is_authenticated
        self.org_id = org_id
        self.workspace_id = workspace_id


class MockUserStore:
    """Mock user store for testing."""

    def __init__(
        self,
        users: list[MockUser] | None = None,
        organizations: list[MockOrganization] | None = None,
    ):
        self._users = users or [MockUser()]
        self._organizations = organizations or [MockOrganization()]
        self._audit_events: list[dict] = []

    def get_user_by_id(self, user_id: str) -> MockUser | None:
        for user in self._users:
            if user.id == user_id:
                return user
        return None

    def list_all_users(
        self,
        limit: int = 50,
        offset: int = 0,
        org_id_filter: str | None = None,
        role_filter: str | None = None,
        active_only: bool = False,
    ) -> tuple[list[MockUser], int]:
        filtered = self._users
        if org_id_filter:
            filtered = [u for u in filtered if u.org_id == org_id_filter]
        if role_filter:
            filtered = [u for u in filtered if u.role == role_filter]
        if active_only:
            filtered = [u for u in filtered if u.is_active]
        return filtered[offset : offset + limit], len(filtered)

    def list_all_organizations(
        self,
        limit: int = 50,
        offset: int = 0,
        tier_filter: str | None = None,
    ) -> tuple[list[MockOrganization], int]:
        filtered = self._organizations
        if tier_filter:
            filtered = [o for o in filtered if o.tier == tier_filter]
        return filtered[offset : offset + limit], len(filtered)

    def get_admin_stats(self) -> dict:
        return {
            "total_users": len(self._users),
            "total_organizations": len(self._organizations),
            "active_users": sum(1 for u in self._users if u.is_active),
            "tier_distribution": {"free": 5, "professional": 3, "enterprise": 2},
        }

    def update_user(self, user_id: str, **kwargs) -> bool:
        for user in self._users:
            if user.id == user_id:
                for key, value in kwargs.items():
                    setattr(user, key, value)
                return True
        return False

    def reset_failed_login_attempts(self, email: str) -> bool:
        return True

    def record_audit_event(self, **kwargs) -> None:
        self._audit_events.append(kwargs)


class MockHandler:
    """Mock HTTP handler for testing."""

    def __init__(
        self,
        headers: dict | None = None,
        body: bytes = b"",
        path: str = "/",
        method: str = "GET",
    ):
        self.headers = headers or {}
        self._body = body
        self.request_body = body
        self.path = path
        self.command = method
        self.client_address = ("127.0.0.1", 12345)


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def mock_admin_user() -> MockUser:
    """Create a mock admin user."""
    return MockUser(role="admin", mfa_enabled=True)


@pytest.fixture
def mock_regular_user() -> MockUser:
    """Create a mock regular user."""
    return MockUser(id="user-002", email="user@example.com", role="user")


@pytest.fixture
def mock_user_store(mock_admin_user, mock_regular_user) -> MockUserStore:
    """Create mock user store with test users."""
    return MockUserStore(
        users=[mock_admin_user, mock_regular_user],
        organizations=[
            MockOrganization(),
            MockOrganization(id="org-002", name="Another Org", tier="enterprise"),
        ],
    )


@pytest.fixture
def mock_server_context(mock_user_store) -> dict:
    """Create mock server context."""
    return {
        "user_store": mock_user_store,
        "debate_storage": MagicMock(),
        "nomic_dir": "/tmp/nomic",
    }


@pytest.fixture
def admin_handler(mock_server_context) -> AdminHandler:
    """Create AdminHandler instance."""
    return AdminHandler(mock_server_context)


@pytest.fixture
def mock_http_handler() -> MockHandler:
    """Create mock HTTP handler."""
    return MockHandler(
        headers={"Content-Type": "application/json"},
        path="/api/v1/admin/stats",
        method="GET",
    )


# ===========================================================================
# Admin Role Tests
# ===========================================================================


class TestAdminRoles:
    """Tests for admin role constants."""

    def test_admin_roles_contains_admin(self):
        """Test ADMIN_ROLES contains 'admin'."""
        assert "admin" in ADMIN_ROLES

    def test_admin_roles_contains_owner(self):
        """Test ADMIN_ROLES contains 'owner'."""
        assert "owner" in ADMIN_ROLES

    def test_admin_roles_does_not_contain_user(self):
        """Test ADMIN_ROLES does not contain 'user'."""
        assert "user" not in ADMIN_ROLES


# ===========================================================================
# Routing Tests
# ===========================================================================


class TestRouting:
    """Tests for route handling."""

    def test_can_handle_admin_routes(self, admin_handler):
        """Test handler recognizes admin routes."""
        assert admin_handler.can_handle("/api/v1/admin/organizations") is True
        assert admin_handler.can_handle("/api/v1/admin/users") is True
        assert admin_handler.can_handle("/api/v1/admin/stats") is True
        assert admin_handler.can_handle("/api/v1/admin/system/metrics") is True

    def test_can_handle_nomic_routes(self, admin_handler):
        """Test handler recognizes nomic admin routes."""
        assert admin_handler.can_handle("/api/v1/admin/nomic/status") is True
        assert admin_handler.can_handle("/api/v1/admin/nomic/circuit-breakers") is True
        assert admin_handler.can_handle("/api/v1/admin/nomic/reset") is True
        assert admin_handler.can_handle("/api/v1/admin/nomic/pause") is True
        assert admin_handler.can_handle("/api/v1/admin/nomic/resume") is True

    def test_cannot_handle_non_admin_routes(self, admin_handler):
        """Test handler rejects non-admin routes."""
        assert admin_handler.can_handle("/api/v1/users") is False
        assert admin_handler.can_handle("/api/v1/debates") is False
        assert admin_handler.can_handle("/health") is False


# ===========================================================================
# Authentication Tests
# ===========================================================================


class TestRequireAdmin:
    """Tests for _require_admin authentication check."""

    def test_require_admin_with_valid_admin(
        self, admin_handler, mock_http_handler, mock_admin_user
    ):
        """Test _require_admin passes for admin user with MFA."""
        with patch(
            "aragora.server.handlers.admin.handler.extract_user_from_request"
        ) as mock_extract:
            mock_extract.return_value = MockAuthContext(
                user_id=mock_admin_user.id, is_authenticated=True
            )
            with patch(
                "aragora.server.handlers.admin.handler.enforce_admin_mfa_policy"
            ) as mock_mfa:
                mock_mfa.return_value = None  # MFA compliant

                auth_ctx, err = admin_handler._require_admin(mock_http_handler)

                assert err is None
                assert auth_ctx is not None
                assert auth_ctx.user_id == mock_admin_user.id

    def test_require_admin_with_unauthenticated_user(self, admin_handler, mock_http_handler):
        """Test _require_admin rejects unauthenticated user."""
        with patch(
            "aragora.server.handlers.admin.handler.extract_user_from_request"
        ) as mock_extract:
            mock_extract.return_value = MockAuthContext(is_authenticated=False)

            auth_ctx, err = admin_handler._require_admin(mock_http_handler)

            assert auth_ctx is None
            assert err is not None
            assert err.status_code == 401  # Unauthorized

    def test_require_admin_with_regular_user(
        self, admin_handler, mock_http_handler, mock_regular_user
    ):
        """Test _require_admin rejects non-admin user."""
        with patch(
            "aragora.server.handlers.admin.handler.extract_user_from_request"
        ) as mock_extract:
            mock_extract.return_value = MockAuthContext(
                user_id=mock_regular_user.id, is_authenticated=True
            )

            auth_ctx, err = admin_handler._require_admin(mock_http_handler)

            assert auth_ctx is None
            assert err is not None
            assert err.status_code == 403  # Forbidden

    def test_require_admin_without_mfa(self, admin_handler, mock_http_handler, mock_admin_user):
        """Test _require_admin rejects admin without MFA."""
        with patch(
            "aragora.server.handlers.admin.handler.extract_user_from_request"
        ) as mock_extract:
            mock_extract.return_value = MockAuthContext(
                user_id=mock_admin_user.id, is_authenticated=True
            )
            with patch(
                "aragora.server.handlers.admin.handler.enforce_admin_mfa_policy"
            ) as mock_mfa:
                mock_mfa.return_value = {"reason": "MFA not enabled", "action": "enable_mfa"}

                auth_ctx, err = admin_handler._require_admin(mock_http_handler)

                assert auth_ctx is None
                assert err is not None
                assert err.status_code == 403  # Forbidden
                data = get_response_data(err)
                # Error can be a string or a structured dict
                error_content = data.get("error", "")
                if isinstance(error_content, dict):
                    assert "MFA" in error_content.get("message", "")
                else:
                    assert "MFA" in error_content


# ===========================================================================
# Organization Endpoint Tests
# ===========================================================================


class TestListOrganizations:
    """Tests for _list_organizations endpoint."""

    def test_list_organizations_success(self, admin_handler, mock_http_handler, mock_admin_user):
        """Test listing organizations as admin."""
        with patch.object(admin_handler, "_require_admin") as mock_require:
            mock_require.return_value = (
                MockAuthContext(user_id=mock_admin_user.id),
                None,
            )
            with patch.object(admin_handler, "_check_rbac_permission") as mock_rbac:
                mock_rbac.return_value = None  # Permission granted

                result = admin_handler._list_organizations(mock_http_handler, {})

                assert result is not None
                assert result.status_code == 200
                data = get_response_data(result)
                assert "organizations" in data
                assert len(data["organizations"]) == 2

    def test_list_organizations_with_pagination(
        self, admin_handler, mock_http_handler, mock_admin_user
    ):
        """Test listing organizations with pagination."""
        with patch.object(admin_handler, "_require_admin") as mock_require:
            mock_require.return_value = (
                MockAuthContext(user_id=mock_admin_user.id),
                None,
            )
            with patch.object(admin_handler, "_check_rbac_permission") as mock_rbac:
                mock_rbac.return_value = None

                result = admin_handler._list_organizations(
                    mock_http_handler, {"limit": "1", "offset": "0"}
                )

                assert result.status_code == 200
                data = get_response_data(result)
                assert data["limit"] == 1
                assert data["offset"] == 0

    def test_list_organizations_with_tier_filter(
        self, admin_handler, mock_http_handler, mock_admin_user
    ):
        """Test listing organizations filtered by tier."""
        with patch.object(admin_handler, "_require_admin") as mock_require:
            mock_require.return_value = (
                MockAuthContext(user_id=mock_admin_user.id),
                None,
            )
            with patch.object(admin_handler, "_check_rbac_permission") as mock_rbac:
                mock_rbac.return_value = None

                result = admin_handler._list_organizations(
                    mock_http_handler, {"tier": "enterprise"}
                )

                assert result.status_code == 200
                data = get_response_data(result)
                assert all(o["tier"] == "enterprise" for o in data["organizations"])


# ===========================================================================
# User Endpoint Tests
# ===========================================================================


class TestListUsers:
    """Tests for _list_users endpoint."""

    def test_list_users_success(self, admin_handler, mock_http_handler, mock_admin_user):
        """Test listing users as admin."""
        with patch.object(admin_handler, "_require_admin") as mock_require:
            mock_require.return_value = (
                MockAuthContext(user_id=mock_admin_user.id),
                None,
            )
            with patch.object(admin_handler, "_check_rbac_permission") as mock_rbac:
                mock_rbac.return_value = None

                result = admin_handler._list_users(mock_http_handler, {})

                assert result.status_code == 200
                data = get_response_data(result)
                assert "users" in data
                assert len(data["users"]) == 2

    def test_list_users_excludes_sensitive_fields(
        self, admin_handler, mock_http_handler, mock_admin_user
    ):
        """Test that sensitive fields are excluded from user list."""
        with patch.object(admin_handler, "_require_admin") as mock_require:
            mock_require.return_value = (
                MockAuthContext(user_id=mock_admin_user.id),
                None,
            )
            with patch.object(admin_handler, "_check_rbac_permission") as mock_rbac:
                mock_rbac.return_value = None

                result = admin_handler._list_users(mock_http_handler, {})

                data = get_response_data(result)
                for user in data["users"]:
                    assert "password_hash" not in user
                    assert "password_salt" not in user
                    assert "api_key" not in user
                    assert "api_key_hash" not in user


class TestDeactivateUser:
    """Tests for _deactivate_user endpoint."""

    def test_deactivate_user_success(self, admin_handler, mock_http_handler, mock_admin_user):
        """Test deactivating a user as admin."""
        with patch.object(admin_handler, "_require_admin") as mock_require:
            mock_require.return_value = (
                MockAuthContext(user_id=mock_admin_user.id),
                None,
            )
            with patch.object(admin_handler, "_check_rbac_permission") as mock_rbac:
                mock_rbac.return_value = None
                with patch("aragora.server.handlers.admin.handler.audit_admin"):
                    result = admin_handler._deactivate_user(mock_http_handler, "user-002")

                    assert result.status_code == 200
                    data = get_response_data(result)
                    assert data["success"] is True
                    assert data["is_active"] is False

    def test_deactivate_user_not_found(self, admin_handler, mock_http_handler, mock_admin_user):
        """Test deactivating a non-existent user."""
        with patch.object(admin_handler, "_require_admin") as mock_require:
            mock_require.return_value = (
                MockAuthContext(user_id=mock_admin_user.id),
                None,
            )
            with patch.object(admin_handler, "_check_rbac_permission") as mock_rbac:
                mock_rbac.return_value = None

                result = admin_handler._deactivate_user(mock_http_handler, "non-existent")

                assert result.status_code == 404

    def test_cannot_deactivate_self(self, admin_handler, mock_http_handler, mock_admin_user):
        """Test admin cannot deactivate themselves."""
        with patch.object(admin_handler, "_require_admin") as mock_require:
            mock_require.return_value = (
                MockAuthContext(user_id=mock_admin_user.id),
                None,
            )
            with patch.object(admin_handler, "_check_rbac_permission") as mock_rbac:
                mock_rbac.return_value = None

                result = admin_handler._deactivate_user(mock_http_handler, mock_admin_user.id)

                assert result.status_code == 400
                data = get_response_data(result)
                assert "yourself" in data.get("error", "").lower()


class TestActivateUser:
    """Tests for _activate_user endpoint."""

    def test_activate_user_success(self, admin_handler, mock_http_handler, mock_admin_user):
        """Test activating a user as admin."""
        with patch.object(admin_handler, "_require_admin") as mock_require:
            mock_require.return_value = (
                MockAuthContext(user_id=mock_admin_user.id),
                None,
            )
            with patch.object(admin_handler, "_check_rbac_permission") as mock_rbac:
                mock_rbac.return_value = None
                with patch("aragora.server.handlers.admin.handler.audit_admin"):
                    result = admin_handler._activate_user(mock_http_handler, "user-002")

                    assert result.status_code == 200
                    data = get_response_data(result)
                    assert data["success"] is True
                    assert data["is_active"] is True


class TestUnlockUser:
    """Tests for _unlock_user endpoint."""

    def test_unlock_user_success(self, admin_handler, mock_http_handler, mock_admin_user):
        """Test unlocking a user as admin."""
        with patch.object(admin_handler, "_require_admin") as mock_require:
            mock_require.return_value = (
                MockAuthContext(user_id=mock_admin_user.id),
                None,
            )
            with patch.object(admin_handler, "_check_rbac_permission") as mock_rbac:
                mock_rbac.return_value = None
                with patch(
                    "aragora.server.handlers.admin.handler.get_lockout_tracker"
                ) as mock_tracker:
                    tracker = MagicMock()
                    tracker.get_info.return_value = {"locked": True, "attempts": 5}
                    tracker.admin_unlock.return_value = True
                    mock_tracker.return_value = tracker

                    result = admin_handler._unlock_user(mock_http_handler, "user-002")

                    assert result.status_code == 200
                    data = get_response_data(result)
                    assert data["success"] is True
                    assert data["lockout_cleared"] is True


# ===========================================================================
# Stats Endpoint Tests
# ===========================================================================


class TestGetStats:
    """Tests for _get_stats endpoint."""

    def test_get_stats_success(self, admin_handler, mock_http_handler, mock_admin_user):
        """Test getting admin stats."""
        with patch.object(admin_handler, "_require_admin") as mock_require:
            mock_require.return_value = (
                MockAuthContext(user_id=mock_admin_user.id),
                None,
            )
            with patch.object(admin_handler, "_check_rbac_permission") as mock_rbac:
                mock_rbac.return_value = None

                result = admin_handler._get_stats(mock_http_handler)

                assert result.status_code == 200
                data = get_response_data(result)
                assert "stats" in data
                assert "total_users" in data["stats"]
                assert "total_organizations" in data["stats"]


class TestGetSystemMetrics:
    """Tests for _get_system_metrics endpoint."""

    def test_get_system_metrics_success(self, admin_handler, mock_http_handler, mock_admin_user):
        """Test getting system metrics."""
        with patch.object(admin_handler, "_require_admin") as mock_require:
            mock_require.return_value = (
                MockAuthContext(user_id=mock_admin_user.id),
                None,
            )
            with patch.object(admin_handler, "_check_rbac_permission") as mock_rbac:
                mock_rbac.return_value = None

                result = admin_handler._get_system_metrics(mock_http_handler)

                assert result.status_code == 200
                data = get_response_data(result)
                assert "metrics" in data
                assert "timestamp" in data["metrics"]
                assert "users" in data["metrics"]


# ===========================================================================
# Impersonation Tests
# ===========================================================================


class TestImpersonateUser:
    """Tests for _impersonate_user endpoint."""

    def test_impersonate_user_success(self, admin_handler, mock_http_handler, mock_admin_user):
        """Test impersonating a user as admin."""
        with patch.object(admin_handler, "_require_admin") as mock_require:
            mock_require.return_value = (
                MockAuthContext(user_id=mock_admin_user.id),
                None,
            )
            with patch.object(admin_handler, "_check_rbac_permission") as mock_rbac:
                mock_rbac.return_value = None
                with patch(
                    "aragora.server.handlers.admin.handler.create_access_token"
                ) as mock_token:
                    mock_token.return_value = "impersonation-token-123"

                    result = admin_handler._impersonate_user(mock_http_handler, "user-002")

                    assert result.status_code == 200
                    data = get_response_data(result)
                    assert data["token"] == "impersonation-token-123"
                    assert data["expires_in"] == 3600
                    assert "warning" in data

    def test_impersonate_user_not_found(self, admin_handler, mock_http_handler, mock_admin_user):
        """Test impersonating a non-existent user."""
        with patch.object(admin_handler, "_require_admin") as mock_require:
            mock_require.return_value = (
                MockAuthContext(user_id=mock_admin_user.id),
                None,
            )
            with patch.object(admin_handler, "_check_rbac_permission") as mock_rbac:
                mock_rbac.return_value = None

                result = admin_handler._impersonate_user(mock_http_handler, "non-existent")

                assert result.status_code == 404


# ===========================================================================
# Nomic Admin Endpoint Tests
# ===========================================================================


class TestNomicAdminEndpoints:
    """Tests for nomic admin endpoints."""

    def test_get_nomic_status(self, admin_handler, mock_http_handler, mock_admin_user):
        """Test getting nomic status."""
        with patch.object(admin_handler, "_require_admin") as mock_require:
            mock_require.return_value = (
                MockAuthContext(user_id=mock_admin_user.id),
                None,
            )
            with patch.object(admin_handler, "_check_rbac_permission") as mock_rbac:
                mock_rbac.return_value = None

                result = admin_handler._get_nomic_status(mock_http_handler)

                assert result.status_code == 200
                data = get_response_data(result)
                assert "running" in data
                assert "current_phase" in data

    def test_pause_nomic(self, admin_handler, mock_http_handler, mock_admin_user):
        """Test pausing nomic loop."""
        mock_http_handler.request_body = json.dumps({"reason": "Testing"}).encode()

        with patch.object(admin_handler, "_require_admin") as mock_require:
            mock_require.return_value = (
                MockAuthContext(user_id=mock_admin_user.id),
                None,
            )
            with patch.object(admin_handler, "_check_rbac_permission") as mock_rbac:
                mock_rbac.return_value = None
                with patch("aragora.server.handlers.admin.handler.audit_admin"):
                    with patch("builtins.open", MagicMock()):
                        with patch("pathlib.Path.exists") as mock_exists:
                            mock_exists.return_value = False
                            with patch("pathlib.Path.mkdir"):
                                result = admin_handler._pause_nomic(mock_http_handler)

                                assert result.status_code == 200
                                data = get_response_data(result)
                                assert data["status"] == "paused"

    def test_reset_nomic_phase_invalid(self, admin_handler, mock_http_handler, mock_admin_user):
        """Test resetting nomic to invalid phase."""
        mock_http_handler.request_body = json.dumps({"target_phase": "invalid_phase"}).encode()

        with patch.object(admin_handler, "_require_admin") as mock_require:
            mock_require.return_value = (
                MockAuthContext(user_id=mock_admin_user.id),
                None,
            )
            with patch.object(admin_handler, "_check_rbac_permission") as mock_rbac:
                mock_rbac.return_value = None

                result = admin_handler._reset_nomic_phase(mock_http_handler)

                assert result.status_code == 400
                data = get_response_data(result)
                assert "Invalid target phase" in data.get("error", "")

    def test_reset_nomic_circuit_breakers(self, admin_handler, mock_http_handler, mock_admin_user):
        """Test resetting nomic circuit breakers."""
        with patch.object(admin_handler, "_require_admin") as mock_require:
            mock_require.return_value = (
                MockAuthContext(user_id=mock_admin_user.id),
                None,
            )
            with patch.object(admin_handler, "_check_rbac_permission") as mock_rbac:
                mock_rbac.return_value = None
                with patch("aragora.server.handlers.admin.handler.audit_admin"):
                    with patch("aragora.nomic.recovery.CircuitBreakerRegistry") as mock_registry:
                        registry = MagicMock()
                        registry.all_open.return_value = ["phase-1"]
                        mock_registry.return_value = registry

                        result = admin_handler._reset_nomic_circuit_breakers(mock_http_handler)

                        assert result.status_code == 200
                        data = get_response_data(result)
                        assert data["success"] is True
                        registry.reset_all.assert_called_once()


# ===========================================================================
# Handle Method Routing Tests
# ===========================================================================


class TestHandleRouting:
    """Tests for the handle method routing."""

    def test_handle_method_not_allowed(self, admin_handler, mock_http_handler):
        """Test handle returns 405 for unsupported methods."""
        mock_http_handler.command = "DELETE"

        result = admin_handler.handle("/api/v1/admin/stats", {}, mock_http_handler, "DELETE")

        assert result.status_code == 405

    def test_handle_routes_to_stats(self, admin_handler, mock_http_handler):
        """Test handle routes GET stats correctly."""
        from aragora.server.handlers.utils.responses import json_response as jr

        with patch.object(admin_handler, "_get_stats") as mock_stats:
            mock_stats.return_value = jr({"stats": {}}, 200)

            admin_handler.handle("/api/v1/admin/stats", {}, mock_http_handler, "GET")

            mock_stats.assert_called_once()

    def test_handle_routes_to_impersonate(self, admin_handler):
        """Test handle routes POST impersonate correctly."""
        from aragora.server.handlers.utils.responses import json_response as jr

        # Create a POST handler
        post_handler = MockHandler(method="POST")

        with patch.object(admin_handler, "_impersonate_user") as mock_impersonate:
            mock_impersonate.return_value = jr({"token": "test"}, 200)

            admin_handler.handle(
                "/api/v1/admin/impersonate/user-123",
                {},
                post_handler,
                "POST",
            )

            mock_impersonate.assert_called_once_with(post_handler, "user-123")


__all__ = ["TestAdminRoles", "TestRouting", "TestRequireAdmin"]
