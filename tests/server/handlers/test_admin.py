"""
Tests for aragora.server.handlers.admin - Admin API handler.

Tests cover:
- Admin authentication and authorization
- List organizations
- List users
- Get stats
- Get system metrics
- Impersonate user
- Deactivate/activate user
- Unlock user
- Nomic admin endpoints (status, circuit breakers, reset, pause, resume)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from io import BytesIO
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.admin import AdminHandler, ADMIN_ROLES


# ===========================================================================
# Test Fixtures
# ===========================================================================


@dataclass
class MockUser:
    """Mock user for testing."""

    id: str = "user-123"
    email: str = "admin@example.com"
    name: str = "Admin User"
    org_id: str | None = "org-123"
    role: str = "admin"
    is_active: bool = True
    mfa_enabled: bool = False  # SOC 2 CC5-01: MFA for admin users
    mfa_secret: str | None = None
    mfa_backup_codes: str | None = None  # JSON list of backup code hashes

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "email": self.email,
            "name": self.name,
            "org_id": self.org_id,
            "role": self.role,
            "is_active": self.is_active,
            "mfa_enabled": self.mfa_enabled,
        }


def get_status(result) -> int:
    """Extract status code from HandlerResult or tuple."""
    if hasattr(result, "status_code"):
        return result.status_code
    return result[1]


def get_body(result) -> dict:
    """Extract body from HandlerResult or tuple."""
    if hasattr(result, "body"):
        body = result.body
        if isinstance(body, bytes):
            return json.loads(body.decode("utf-8"))
        return json.loads(body)
    body = result[0]
    if isinstance(body, dict):
        return body
    return json.loads(body)


@dataclass
class MockOrganization:
    """Mock organization for testing."""

    id: str = "org-123"
    name: str = "Test Org"
    owner_id: str = "user-123"
    tier: Any = field(default_factory=lambda: MagicMock(value="starter"))

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "owner_id": self.owner_id}


@dataclass
class MockAuthContext:
    """Mock authentication context."""

    is_authenticated: bool = True
    user_id: str = "user-123"
    email: str = "admin@example.com"
    org_id: str | None = "org-123"
    role: str = "admin"


class MockUserStore:
    """Mock user store for testing."""

    def __init__(self):
        self.users: dict[str, MockUser] = {}
        self.orgs: dict[str, MockOrganization] = {}
        self.updates: list[dict] = []

    def get_user_by_id(self, user_id: str) -> MockUser | None:
        return self.users.get(user_id)

    def get_organization_by_id(self, org_id: str) -> MockOrganization | None:
        return self.orgs.get(org_id)

    def list_all_organizations(self, limit: int = 50, offset: int = 0, tier_filter: str = None):
        orgs = list(self.orgs.values())
        if tier_filter:
            orgs = [o for o in orgs if o.tier.value == tier_filter]
        return orgs[offset : offset + limit], len(orgs)

    def list_all_users(
        self,
        limit: int = 50,
        offset: int = 0,
        org_id_filter: str = None,
        role_filter: str = None,
        active_only: bool = False,
    ):
        users = list(self.users.values())
        if org_id_filter:
            users = [u for u in users if u.org_id == org_id_filter]
        if role_filter:
            users = [u for u in users if u.role == role_filter]
        if active_only:
            users = [u for u in users if u.is_active]
        return users[offset : offset + limit], len(users)

    def get_admin_stats(self) -> dict[str, Any]:
        return {
            "total_users": len(self.users),
            "total_organizations": len(self.orgs),
            "active_users": sum(1 for u in self.users.values() if u.is_active),
            "tier_distribution": {"free": 1, "starter": 2},
        }

    def update_user(self, user_id: str, **kwargs) -> None:
        self.updates.append({"user_id": user_id, **kwargs})
        user = self.users.get(user_id)
        if user:
            for key, value in kwargs.items():
                if hasattr(user, key):
                    setattr(user, key, value)

    def record_audit_event(self, **kwargs) -> None:
        pass

    def reset_failed_login_attempts(self, email: str) -> bool:
        return True

    def log_audit_event(self, **kwargs) -> None:
        pass


def make_mock_handler(
    body: dict | None = None,
    method: str = "GET",
    headers: dict | None = None,
):
    """Create a mock HTTP handler."""
    handler = MagicMock()
    handler.command = method
    handler.headers = headers or {}
    handler.client_address = ("127.0.0.1", 12345)

    if body is not None:
        body_bytes = json.dumps(body).encode("utf-8")
        handler.headers["Content-Length"] = str(len(body_bytes))
        handler.rfile = BytesIO(body_bytes)
        handler.request_body = body_bytes
    else:
        handler.rfile = BytesIO(b"")
        handler.headers["Content-Length"] = "0"
        handler.request_body = b"{}"

    return handler


@pytest.fixture
def user_store():
    """Create mock user store with admin user."""
    store = MockUserStore()
    # Admin users must have MFA enabled with enough backup codes (SOC 2 CC5-01)
    # backup_codes must be a JSON list of hashes, >= 3 codes to pass policy
    backup_codes_json = json.dumps(
        ["hash1", "hash2", "hash3", "hash4", "hash5", "hash6", "hash7", "hash8", "hash9", "hash10"]
    )
    admin = MockUser(
        id="admin-123",
        email="admin@example.com",
        role="admin",
        mfa_enabled=True,
        mfa_secret="TESTSECRET123456",
        mfa_backup_codes=backup_codes_json,
    )
    owner = MockUser(
        id="owner-123",
        email="owner@example.com",
        role="owner",
        mfa_enabled=True,
        mfa_secret="TESTSECRET654321",
        mfa_backup_codes=backup_codes_json,
    )
    regular = MockUser(id="user-456", email="user@example.com", role="user")

    store.users["admin-123"] = admin
    store.users["owner-123"] = owner
    store.users["user-456"] = regular

    org = MockOrganization()
    store.orgs["org-123"] = org

    return store


@pytest.fixture
def admin_handler(user_store):
    """Create AdminHandler with mock context."""
    ctx = {"user_store": user_store, "nomic_dir": "/tmp/test_nomic"}
    return AdminHandler(ctx)


# ===========================================================================
# Test Routing
# ===========================================================================


class TestAdminHandlerRouting:
    """Tests for AdminHandler routing."""

    def test_can_handle_admin_paths(self, admin_handler):
        assert admin_handler.can_handle("/api/admin/organizations") is True
        assert admin_handler.can_handle("/api/admin/users") is True
        assert admin_handler.can_handle("/api/admin/stats") is True
        assert admin_handler.can_handle("/api/admin/nomic/status") is True

    def test_cannot_handle_non_admin_paths(self, admin_handler):
        assert admin_handler.can_handle("/api/billing/plans") is False
        assert admin_handler.can_handle("/api/debates") is False


# ===========================================================================
# Test Admin Authorization
# ===========================================================================


class TestAdminAuthorization:
    """Tests for admin authorization."""

    @patch("aragora.server.handlers.admin.admin.extract_user_from_request")
    def test_admin_role_allowed(self, mock_auth, admin_handler, user_store):
        mock_auth.return_value = MockAuthContext(user_id="admin-123", role="admin")

        handler = make_mock_handler()

        result = admin_handler.handle("/api/admin/stats", {}, handler, "GET")

        assert result is not None
        assert get_status(result) == 200

    @patch("aragora.server.handlers.admin.admin.extract_user_from_request")
    def test_owner_role_allowed(self, mock_auth, admin_handler, user_store):
        mock_auth.return_value = MockAuthContext(user_id="owner-123", role="owner")

        handler = make_mock_handler()

        result = admin_handler.handle("/api/admin/stats", {}, handler, "GET")

        assert result is not None
        assert get_status(result) == 200

    @patch("aragora.server.handlers.admin.admin.extract_user_from_request")
    def test_regular_user_denied(self, mock_auth, admin_handler, user_store):
        mock_auth.return_value = MockAuthContext(user_id="user-456", role="user")

        handler = make_mock_handler()

        result = admin_handler.handle("/api/admin/stats", {}, handler, "GET")

        assert result is not None
        assert get_status(result) == 403

    @patch("aragora.server.handlers.admin.admin.extract_user_from_request")
    def test_unauthenticated_denied(self, mock_auth, admin_handler):
        mock_auth.return_value = MockAuthContext(is_authenticated=False)

        handler = make_mock_handler()

        result = admin_handler.handle("/api/admin/stats", {}, handler, "GET")

        assert result is not None
        assert get_status(result) == 401


# ===========================================================================
# Test List Organizations
# ===========================================================================


class TestAdminListOrganizations:
    """Tests for list organizations endpoint."""

    @patch("aragora.server.handlers.admin.admin.extract_user_from_request")
    def test_list_organizations_success(self, mock_auth, admin_handler, user_store):
        mock_auth.return_value = MockAuthContext(user_id="admin-123", role="admin")

        handler = make_mock_handler()

        result = admin_handler.handle("/api/admin/organizations", {}, handler, "GET")

        assert result is not None
        assert get_status(result) == 200
        data = get_body(result)
        assert "organizations" in data
        assert "total" in data

    @patch("aragora.server.handlers.admin.admin.extract_user_from_request")
    def test_list_organizations_with_pagination(self, mock_auth, admin_handler, user_store):
        mock_auth.return_value = MockAuthContext(user_id="admin-123", role="admin")

        query_params = {"limit": "10", "offset": "5"}
        handler = make_mock_handler()

        result = admin_handler.handle("/api/admin/organizations", query_params, handler, "GET")

        assert result is not None
        assert get_status(result) == 200
        data = get_body(result)
        assert data["limit"] == 10
        assert data["offset"] == 5


# ===========================================================================
# Test List Users
# ===========================================================================


class TestAdminListUsers:
    """Tests for list users endpoint."""

    @patch("aragora.server.handlers.admin.admin.extract_user_from_request")
    def test_list_users_success(self, mock_auth, admin_handler, user_store):
        mock_auth.return_value = MockAuthContext(user_id="admin-123", role="admin")

        handler = make_mock_handler()

        result = admin_handler.handle("/api/admin/users", {}, handler, "GET")

        assert result is not None
        assert get_status(result) == 200
        data = get_body(result)
        assert "users" in data
        assert "total" in data

    @patch("aragora.server.handlers.admin.admin.extract_user_from_request")
    def test_list_users_sanitizes_sensitive_fields(self, mock_auth, admin_handler, user_store):
        # Add a user with sensitive fields
        user = MockUser(id="test-user")
        user.password_hash = "secret"
        user.password_salt = "secret"
        user.api_key = "secret"
        user_store.users["test-user"] = user

        mock_auth.return_value = MockAuthContext(user_id="admin-123", role="admin")

        handler = make_mock_handler()

        result = admin_handler.handle("/api/admin/users", {}, handler, "GET")

        assert result is not None
        data = get_body(result)

        # Check that sensitive fields are not in the response
        for user_dict in data["users"]:
            assert "password_hash" not in user_dict
            assert "password_salt" not in user_dict
            assert "api_key" not in user_dict


# ===========================================================================
# Test Get Stats
# ===========================================================================


class TestAdminGetStats:
    """Tests for get stats endpoint."""

    @patch("aragora.server.handlers.admin.admin.extract_user_from_request")
    def test_get_stats_success(self, mock_auth, admin_handler, user_store):
        mock_auth.return_value = MockAuthContext(user_id="admin-123", role="admin")

        handler = make_mock_handler()

        result = admin_handler.handle("/api/admin/stats", {}, handler, "GET")

        assert result is not None
        assert get_status(result) == 200
        data = get_body(result)
        assert "stats" in data


# ===========================================================================
# Test Get System Metrics
# ===========================================================================


class TestAdminSystemMetrics:
    """Tests for system metrics endpoint."""

    @patch("aragora.server.handlers.admin.admin.extract_user_from_request")
    def test_get_system_metrics_success(self, mock_auth, admin_handler, user_store):
        mock_auth.return_value = MockAuthContext(user_id="admin-123", role="admin")

        handler = make_mock_handler()

        result = admin_handler.handle("/api/admin/system/metrics", {}, handler, "GET")

        assert result is not None
        assert get_status(result) == 200
        data = get_body(result)
        assert "metrics" in data
        assert "timestamp" in data["metrics"]


# ===========================================================================
# Test Revenue Stats
# ===========================================================================


class TestAdminRevenueStats:
    """Tests for revenue stats endpoint."""

    @patch("aragora.server.handlers.admin.admin.extract_user_from_request")
    def test_get_revenue_stats_success(self, mock_auth, admin_handler, user_store):
        mock_auth.return_value = MockAuthContext(user_id="admin-123", role="admin")

        handler = make_mock_handler()

        with patch(
            "aragora.billing.models.TIER_LIMITS",
            {
                "free": MagicMock(price_monthly_cents=0),
                "starter": MagicMock(price_monthly_cents=2900),
            },
        ):
            result = admin_handler.handle("/api/admin/revenue", {}, handler, "GET")

        assert result is not None
        assert get_status(result) == 200
        data = get_body(result)
        assert "revenue" in data
        assert "mrr_cents" in data["revenue"]
        assert "mrr_dollars" in data["revenue"]


# ===========================================================================
# Test Impersonate User
# ===========================================================================


class TestAdminImpersonate:
    """Tests for impersonate user endpoint."""

    @patch("aragora.server.handlers.admin.admin.extract_user_from_request")
    @patch("aragora.server.handlers.admin.admin.create_access_token")
    def test_impersonate_success(self, mock_token, mock_auth, admin_handler, user_store):
        mock_auth.return_value = MockAuthContext(user_id="admin-123", role="admin")
        mock_token.return_value = "impersonation_token_123"

        handler = make_mock_handler(method="POST")

        result = admin_handler.handle("/api/admin/impersonate/user-456", {}, handler, "POST")

        assert result is not None
        assert get_status(result) == 200
        data = get_body(result)
        assert "token" in data
        assert "target_user" in data
        assert "warning" in data

    @patch("aragora.server.handlers.admin.admin.extract_user_from_request")
    def test_impersonate_user_not_found(self, mock_auth, admin_handler, user_store):
        mock_auth.return_value = MockAuthContext(user_id="admin-123", role="admin")

        handler = make_mock_handler(method="POST")

        result = admin_handler.handle("/api/admin/impersonate/nonexistent", {}, handler, "POST")

        assert result is not None
        assert get_status(result) == 404

    @patch("aragora.server.handlers.admin.admin.extract_user_from_request")
    def test_impersonate_invalid_user_id(self, mock_auth, admin_handler):
        mock_auth.return_value = MockAuthContext(user_id="admin-123", role="admin")

        handler = make_mock_handler(method="POST")

        result = admin_handler.handle("/api/admin/impersonate/../etc/passwd", {}, handler, "POST")

        assert result is not None
        # Handler returns 404 (user not found) for invalid IDs - secure behavior
        # that doesn't reveal the ID format validation
        assert get_status(result) == 404


# ===========================================================================
# Test Deactivate User
# ===========================================================================


class TestAdminDeactivateUser:
    """Tests for deactivate user endpoint."""

    @patch("aragora.server.handlers.admin.admin.extract_user_from_request")
    def test_deactivate_user_success(self, mock_auth, admin_handler, user_store):
        mock_auth.return_value = MockAuthContext(user_id="admin-123", role="admin")

        handler = make_mock_handler(method="POST")

        result = admin_handler.handle("/api/admin/users/user-456/deactivate", {}, handler, "POST")

        assert result is not None
        assert get_status(result) == 200
        data = get_body(result)
        assert data["success"] is True
        assert data["is_active"] is False

    @patch("aragora.server.handlers.admin.admin.extract_user_from_request")
    def test_deactivate_self_prevented(self, mock_auth, admin_handler, user_store):
        mock_auth.return_value = MockAuthContext(user_id="admin-123", role="admin")

        handler = make_mock_handler(method="POST")

        result = admin_handler.handle("/api/admin/users/admin-123/deactivate", {}, handler, "POST")

        assert result is not None
        assert get_status(result) == 400
        data = get_body(result)
        assert "yourself" in data["error"].lower()


# ===========================================================================
# Test Activate User
# ===========================================================================


class TestAdminActivateUser:
    """Tests for activate user endpoint."""

    @patch("aragora.server.handlers.admin.admin.extract_user_from_request")
    def test_activate_user_success(self, mock_auth, admin_handler, user_store):
        # Set user as inactive first
        user_store.users["user-456"].is_active = False

        mock_auth.return_value = MockAuthContext(user_id="admin-123", role="admin")

        handler = make_mock_handler(method="POST")

        result = admin_handler.handle("/api/admin/users/user-456/activate", {}, handler, "POST")

        assert result is not None
        assert get_status(result) == 200
        data = get_body(result)
        assert data["success"] is True
        assert data["is_active"] is True


# ===========================================================================
# Test Unlock User
# ===========================================================================


class TestAdminUnlockUser:
    """Tests for unlock user endpoint."""

    @patch("aragora.server.handlers.admin.admin.extract_user_from_request")
    @patch("aragora.server.handlers.admin.admin.get_lockout_tracker")
    def test_unlock_user_success(self, mock_lockout, mock_auth, admin_handler, user_store):
        mock_auth.return_value = MockAuthContext(user_id="admin-123", role="admin")

        mock_tracker = MagicMock()
        mock_tracker.get_info.return_value = {"locked": True, "attempts": 5}
        mock_tracker.admin_unlock.return_value = True
        mock_lockout.return_value = mock_tracker

        handler = make_mock_handler(method="POST")

        result = admin_handler.handle("/api/admin/users/user-456/unlock", {}, handler, "POST")

        assert result is not None
        assert get_status(result) == 200
        data = get_body(result)
        assert data["success"] is True
        assert data["lockout_cleared"] is True


# ===========================================================================
# Test Nomic Status
# ===========================================================================


class TestAdminNomicStatus:
    """Tests for nomic status endpoint."""

    @patch("aragora.server.handlers.admin.admin.extract_user_from_request")
    def test_get_nomic_status_no_state_file(self, mock_auth, admin_handler):
        mock_auth.return_value = MockAuthContext(user_id="admin-123", role="admin")

        handler = make_mock_handler()

        result = admin_handler.handle("/api/admin/nomic/status", {}, handler, "GET")

        assert result is not None
        assert get_status(result) == 200
        data = get_body(result)
        assert "running" in data
        assert "current_phase" in data


# ===========================================================================
# Test Nomic Circuit Breakers
# ===========================================================================


class TestAdminNomicCircuitBreakers:
    """Tests for nomic circuit breakers endpoint."""

    @patch("aragora.server.handlers.admin.admin.extract_user_from_request")
    def test_get_circuit_breakers_module_not_available(self, mock_auth, admin_handler):
        mock_auth.return_value = MockAuthContext(user_id="admin-123", role="admin")

        handler = make_mock_handler()

        with patch.dict("sys.modules", {"aragora.nomic.recovery": None}):
            result = admin_handler.handle("/api/admin/nomic/circuit-breakers", {}, handler, "GET")

        # Should handle gracefully
        assert result is not None


# ===========================================================================
# Test Nomic Reset
# ===========================================================================


class TestAdminNomicReset:
    """Tests for nomic reset endpoint."""

    @patch("aragora.server.handlers.admin.admin.extract_user_from_request")
    def test_reset_nomic_phase_success(self, mock_auth, admin_handler, tmp_path):
        mock_auth.return_value = MockAuthContext(user_id="admin-123", role="admin")

        # Use temp directory for nomic state
        admin_handler.ctx["nomic_dir"] = str(tmp_path)

        handler = make_mock_handler(
            {"target_phase": "context", "reason": "Test reset"},
            method="POST",
        )

        result = admin_handler.handle("/api/admin/nomic/reset", {}, handler, "POST")

        assert result is not None
        assert get_status(result) == 200
        data = get_body(result)
        assert data["success"] is True
        assert data["new_phase"] == "context"

    @patch("aragora.server.handlers.admin.admin.extract_user_from_request")
    def test_reset_nomic_invalid_phase(self, mock_auth, admin_handler, tmp_path):
        mock_auth.return_value = MockAuthContext(user_id="admin-123", role="admin")

        admin_handler.ctx["nomic_dir"] = str(tmp_path)

        handler = make_mock_handler(
            {"target_phase": "invalid_phase"},
            method="POST",
        )

        result = admin_handler.handle("/api/admin/nomic/reset", {}, handler, "POST")

        assert result is not None
        assert get_status(result) == 400


# ===========================================================================
# Test Nomic Pause/Resume
# ===========================================================================


class TestAdminNomicPauseResume:
    """Tests for nomic pause/resume endpoints."""

    @patch("aragora.server.handlers.admin.admin.extract_user_from_request")
    def test_pause_nomic_success(self, mock_auth, admin_handler, tmp_path):
        mock_auth.return_value = MockAuthContext(user_id="admin-123", role="admin")

        admin_handler.ctx["nomic_dir"] = str(tmp_path)

        # Create initial state
        state_file = tmp_path / "nomic_state.json"
        state_file.write_text(json.dumps({"phase": "debate", "running": True}))

        handler = make_mock_handler(
            {"reason": "Manual intervention"},
            method="POST",
        )

        result = admin_handler.handle("/api/admin/nomic/pause", {}, handler, "POST")

        assert result is not None
        assert get_status(result) == 200
        data = get_body(result)
        assert data["success"] is True
        assert data["status"] == "paused"

    @patch("aragora.server.handlers.admin.admin.extract_user_from_request")
    def test_resume_nomic_success(self, mock_auth, admin_handler, tmp_path):
        mock_auth.return_value = MockAuthContext(user_id="admin-123", role="admin")

        admin_handler.ctx["nomic_dir"] = str(tmp_path)

        # Create paused state
        state_file = tmp_path / "nomic_state.json"
        state_file.write_text(
            json.dumps({"phase": "paused", "running": False, "previous_phase": "debate"})
        )

        handler = make_mock_handler(method="POST")

        result = admin_handler.handle("/api/admin/nomic/resume", {}, handler, "POST")

        assert result is not None
        assert get_status(result) == 200
        data = get_body(result)
        assert data["success"] is True
        assert data["status"] == "resumed"
        assert data["phase"] == "debate"

    @patch("aragora.server.handlers.admin.admin.extract_user_from_request")
    def test_resume_not_paused(self, mock_auth, admin_handler, tmp_path):
        mock_auth.return_value = MockAuthContext(user_id="admin-123", role="admin")

        admin_handler.ctx["nomic_dir"] = str(tmp_path)

        # Create running state (not paused)
        state_file = tmp_path / "nomic_state.json"
        state_file.write_text(json.dumps({"phase": "debate", "running": True}))

        handler = make_mock_handler(method="POST")

        result = admin_handler.handle("/api/admin/nomic/resume", {}, handler, "POST")

        assert result is not None
        assert get_status(result) == 400


# ===========================================================================
# Test Nomic Circuit Breakers Reset
# ===========================================================================


class TestAdminResetCircuitBreakers:
    """Tests for reset circuit breakers endpoint."""

    @patch("aragora.server.handlers.admin.admin.extract_user_from_request")
    def test_reset_circuit_breakers_module_not_available(self, mock_auth, admin_handler):
        mock_auth.return_value = MockAuthContext(user_id="admin-123", role="admin")

        handler = make_mock_handler(method="POST")

        with patch.dict("sys.modules", {"aragora.nomic.recovery": None}):
            result = admin_handler.handle(
                "/api/admin/nomic/circuit-breakers/reset", {}, handler, "POST"
            )

        # Should handle gracefully with 503
        assert result is not None


# ===========================================================================
# Test Method Not Allowed
# ===========================================================================


class TestAdminMethodNotAllowed:
    """Tests for method not allowed responses."""

    @patch("aragora.server.handlers.admin.admin.extract_user_from_request")
    def test_stats_post_not_allowed(self, mock_auth, admin_handler):
        mock_auth.return_value = MockAuthContext(user_id="admin-123", role="admin")

        handler = make_mock_handler(method="POST")

        result = admin_handler.handle("/api/admin/stats", {}, handler, "POST")

        assert result is not None
        assert get_status(result) == 405


# ===========================================================================
# Test Service Unavailable
# ===========================================================================


class TestAdminServiceUnavailable:
    """Tests for service unavailable scenarios."""

    def test_no_user_store(self):
        handler_ctx = AdminHandler({})

        handler = make_mock_handler()

        auth_ctx, err = handler_ctx._require_admin(handler)

        assert err is not None
        assert get_status(err) == 503


# ===========================================================================
# Test Admin Roles Constant
# ===========================================================================


class TestAdminRoles:
    """Tests for admin roles constant."""

    def test_admin_roles_contains_admin(self):
        assert "admin" in ADMIN_ROLES

    def test_admin_roles_contains_owner(self):
        assert "owner" in ADMIN_ROLES

    def test_admin_roles_excludes_user(self):
        assert "user" not in ADMIN_ROLES
