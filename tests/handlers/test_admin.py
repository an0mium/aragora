"""Tests for admin handler endpoints.

Tests the administrative API endpoints including:
- Authentication and authorization (admin role + MFA)
- Organization listing
- User listing and management
- System statistics and metrics
- User impersonation (audit-logged)
- User activation/deactivation
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.admin.admin import AdminHandler, ADMIN_ROLES


def parse_body(result) -> dict:
    """Parse JSON body from HandlerResult."""
    return json.loads(result.body.decode("utf-8"))


class MockUser:
    """Mock user object."""

    def __init__(
        self,
        id: str,
        email: str,
        name: str = "Test User",
        role: str = "member",
        org_id: str = "org_1",
        is_active: bool = True,
        mfa_enabled: bool = False,
    ):
        self.id = id
        self.email = email
        self.name = name
        self.role = role
        self.org_id = org_id
        self.is_active = is_active
        self.mfa_enabled = mfa_enabled

    def to_dict(self) -> dict:
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
        id: str,
        name: str,
        tier: str = "free",
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


class MockUserStore:
    """Mock user store for testing."""

    def __init__(self):
        self._users: Dict[str, MockUser] = {}
        self._orgs: Dict[str, MockOrganization] = {}
        self._audit_events: List[dict] = []

    def add_user(self, user: MockUser):
        self._users[user.id] = user

    def add_organization(self, org: MockOrganization):
        self._orgs[org.id] = org

    def get_user_by_id(self, user_id: str) -> Optional[MockUser]:
        return self._users.get(user_id)

    def list_all_users(
        self,
        limit: int = 50,
        offset: int = 0,
        org_id_filter: Optional[str] = None,
        role_filter: Optional[str] = None,
        active_only: bool = False,
    ) -> tuple[List[MockUser], int]:
        users = list(self._users.values())
        if org_id_filter:
            users = [u for u in users if u.org_id == org_id_filter]
        if role_filter:
            users = [u for u in users if u.role == role_filter]
        if active_only:
            users = [u for u in users if u.is_active]
        total = len(users)
        return users[offset : offset + limit], total

    def list_all_organizations(
        self,
        limit: int = 50,
        offset: int = 0,
        tier_filter: Optional[str] = None,
    ) -> tuple[List[MockOrganization], int]:
        orgs = list(self._orgs.values())
        if tier_filter:
            orgs = [o for o in orgs if o.tier == tier_filter]
        total = len(orgs)
        return orgs[offset : offset + limit], total

    def get_admin_stats(self) -> dict:
        return {
            "total_users": len(self._users),
            "total_organizations": len(self._orgs),
            "tier_distribution": {"free": 1, "pro": 0, "enterprise": 0},
        }

    def update_user(self, user_id: str, **kwargs) -> Optional[MockUser]:
        user = self._users.get(user_id)
        if user:
            for key, value in kwargs.items():
                if hasattr(user, key):
                    setattr(user, key, value)
        return user

    def record_audit_event(self, **kwargs):
        self._audit_events.append(kwargs)


class MockAuthContext:
    """Mock authentication context."""

    def __init__(
        self,
        user_id: str,
        is_authenticated: bool = True,
        org_id: Optional[str] = None,
    ):
        self.user_id = user_id
        self.is_authenticated = is_authenticated
        self.org_id = org_id


class MockHandler:
    """Mock HTTP handler."""

    def __init__(
        self,
        body: Optional[dict] = None,
        command: str = "GET",
    ):
        self.command = command
        self.headers = {}
        self.rfile = MagicMock()
        self.client_address = ("127.0.0.1", 12345)

        if body:
            body_bytes = json.dumps(body).encode()
            self.rfile.read.return_value = body_bytes
            self.headers["Content-Length"] = str(len(body_bytes))
        else:
            self.rfile.read.return_value = b"{}"
            self.headers["Content-Length"] = "2"


@pytest.fixture
def user_store():
    """Create a mock user store with test data."""
    store = MockUserStore()

    # Add admin user with MFA
    admin = MockUser(
        id="admin_1",
        email="admin@example.com",
        name="Admin User",
        role="admin",
        mfa_enabled=True,
    )
    store.add_user(admin)

    # Add regular users
    for i in range(5):
        user = MockUser(
            id=f"user_{i}",
            email=f"user{i}@example.com",
            name=f"User {i}",
            role="member",
        )
        store.add_user(user)

    # Add organizations
    for i in range(3):
        org = MockOrganization(
            id=f"org_{i}",
            name=f"Organization {i}",
            tier=["free", "pro", "enterprise"][i],
        )
        store.add_organization(org)

    return store


@pytest.fixture
def admin_handler(user_store):
    """Create an admin handler with mock context."""
    ctx = {"user_store": user_store}
    return AdminHandler(ctx)


class TestAdminHandlerRouting:
    """Tests for admin handler routing."""

    def test_can_handle_admin_paths(self):
        """Test can_handle identifies admin paths."""
        assert AdminHandler.can_handle("/api/admin/users")
        assert AdminHandler.can_handle("/api/admin/organizations")
        assert AdminHandler.can_handle("/api/admin/stats")
        assert AdminHandler.can_handle("/api/admin/nomic/status")

    def test_cannot_handle_non_admin_paths(self):
        """Test can_handle rejects non-admin paths."""
        assert not AdminHandler.can_handle("/api/debates")
        assert not AdminHandler.can_handle("/api/users")
        assert not AdminHandler.can_handle("/api/health")


class TestAdminAuthentication:
    """Tests for admin authentication and authorization."""

    def test_unauthenticated_request_rejected(self, admin_handler):
        """Test that unauthenticated requests are rejected."""
        mock_handler = MockHandler()

        with patch("aragora.server.handlers.admin.admin.extract_user_from_request") as mock_extract:
            mock_ctx = MockAuthContext("", is_authenticated=False)
            mock_extract.return_value = mock_ctx

            result = admin_handler.handle("/api/admin/stats", {}, mock_handler)

            assert result.status_code == 401
            assert "authenticated" in parse_body(result)["error"].lower()

    def test_non_admin_user_rejected(self, admin_handler, user_store):
        """Test that non-admin users are rejected."""
        mock_handler = MockHandler()

        with patch("aragora.server.handlers.admin.admin.extract_user_from_request") as mock_extract:
            mock_ctx = MockAuthContext("user_0", is_authenticated=True)
            mock_extract.return_value = mock_ctx

            result = admin_handler.handle("/api/admin/stats", {}, mock_handler)

            assert result.status_code == 403
            assert "admin" in parse_body(result)["error"].lower()

    def test_admin_without_mfa_rejected(self, admin_handler, user_store):
        """Test that admin without MFA is rejected."""
        # Create admin without MFA
        admin_no_mfa = MockUser(
            id="admin_no_mfa",
            email="admin_no_mfa@example.com",
            role="admin",
            mfa_enabled=False,
        )
        user_store.add_user(admin_no_mfa)

        mock_handler = MockHandler()

        with patch("aragora.server.handlers.admin.admin.extract_user_from_request") as mock_extract:
            mock_ctx = MockAuthContext("admin_no_mfa", is_authenticated=True)
            mock_extract.return_value = mock_ctx

            with patch("aragora.server.handlers.admin.admin.enforce_admin_mfa_policy") as mock_mfa:
                mock_mfa.return_value = {"reason": "MFA not enabled", "action": "enable_mfa"}

                result = admin_handler.handle("/api/admin/stats", {}, mock_handler)
                body = parse_body(result)

                assert result.status_code == 403
                # Error response uses "message" field with ADMIN_MFA_REQUIRED code
                assert body.get("code") == "ADMIN_MFA_REQUIRED" or "MFA" in str(body)

    def test_admin_with_mfa_allowed(self, admin_handler, user_store):
        """Test that admin with MFA is allowed."""
        mock_handler = MockHandler()

        with patch("aragora.server.handlers.admin.admin.extract_user_from_request") as mock_extract:
            mock_ctx = MockAuthContext("admin_1", is_authenticated=True)
            mock_extract.return_value = mock_ctx

            with patch("aragora.server.handlers.admin.admin.enforce_admin_mfa_policy") as mock_mfa:
                mock_mfa.return_value = None  # MFA check passed

                result = admin_handler.handle("/api/admin/stats", {}, mock_handler)

                assert result.status_code == 200


class TestListOrganizations:
    """Tests for list organizations endpoint."""

    def setup_admin_auth(self, mock_extract, mock_mfa):
        """Helper to set up admin authentication."""
        mock_ctx = MockAuthContext("admin_1", is_authenticated=True)
        mock_extract.return_value = mock_ctx
        mock_mfa.return_value = None

    def test_list_organizations_success(self, admin_handler):
        """Test successful organization listing."""
        mock_handler = MockHandler()

        with patch("aragora.server.handlers.admin.admin.extract_user_from_request") as mock_extract:
            with patch("aragora.server.handlers.admin.admin.enforce_admin_mfa_policy") as mock_mfa:
                self.setup_admin_auth(mock_extract, mock_mfa)

                result = admin_handler.handle("/api/admin/organizations", {}, mock_handler)
                body = parse_body(result)

                assert result.status_code == 200
                assert "organizations" in body
                assert "total" in body
                assert len(body["organizations"]) == 3

    def test_list_organizations_with_pagination(self, admin_handler):
        """Test organization listing with pagination."""
        mock_handler = MockHandler()

        with patch("aragora.server.handlers.admin.admin.extract_user_from_request") as mock_extract:
            with patch("aragora.server.handlers.admin.admin.enforce_admin_mfa_policy") as mock_mfa:
                self.setup_admin_auth(mock_extract, mock_mfa)

                result = admin_handler.handle(
                    "/api/admin/organizations",
                    {"limit": "2", "offset": "1"},
                    mock_handler,
                )
                body = parse_body(result)

                assert result.status_code == 200
                assert body["limit"] == 2
                assert body["offset"] == 1

    def test_list_organizations_with_tier_filter(self, admin_handler):
        """Test organization listing with tier filter."""
        mock_handler = MockHandler()

        with patch("aragora.server.handlers.admin.admin.extract_user_from_request") as mock_extract:
            with patch("aragora.server.handlers.admin.admin.enforce_admin_mfa_policy") as mock_mfa:
                self.setup_admin_auth(mock_extract, mock_mfa)

                result = admin_handler.handle(
                    "/api/admin/organizations",
                    {"tier": "pro"},
                    mock_handler,
                )
                body = parse_body(result)

                assert result.status_code == 200
                # All returned orgs should be pro tier
                for org in body["organizations"]:
                    assert org["tier"] == "pro"


class TestListUsers:
    """Tests for list users endpoint."""

    def setup_admin_auth(self, mock_extract, mock_mfa):
        """Helper to set up admin authentication."""
        mock_ctx = MockAuthContext("admin_1", is_authenticated=True)
        mock_extract.return_value = mock_ctx
        mock_mfa.return_value = None

    def test_list_users_success(self, admin_handler):
        """Test successful user listing."""
        mock_handler = MockHandler()

        with patch("aragora.server.handlers.admin.admin.extract_user_from_request") as mock_extract:
            with patch("aragora.server.handlers.admin.admin.enforce_admin_mfa_policy") as mock_mfa:
                self.setup_admin_auth(mock_extract, mock_mfa)

                result = admin_handler.handle("/api/admin/users", {}, mock_handler)
                body = parse_body(result)

                assert result.status_code == 200
                assert "users" in body
                assert "total" in body

    def test_list_users_excludes_sensitive_fields(self, admin_handler):
        """Test that user listing excludes sensitive fields."""
        mock_handler = MockHandler()

        with patch("aragora.server.handlers.admin.admin.extract_user_from_request") as mock_extract:
            with patch("aragora.server.handlers.admin.admin.enforce_admin_mfa_policy") as mock_mfa:
                self.setup_admin_auth(mock_extract, mock_mfa)

                result = admin_handler.handle("/api/admin/users", {}, mock_handler)
                body = parse_body(result)

                for user in body["users"]:
                    assert "password_hash" not in user
                    assert "password_salt" not in user
                    assert "api_key" not in user

    def test_list_users_with_role_filter(self, admin_handler):
        """Test user listing with role filter."""
        mock_handler = MockHandler()

        with patch("aragora.server.handlers.admin.admin.extract_user_from_request") as mock_extract:
            with patch("aragora.server.handlers.admin.admin.enforce_admin_mfa_policy") as mock_mfa:
                self.setup_admin_auth(mock_extract, mock_mfa)

                result = admin_handler.handle(
                    "/api/admin/users",
                    {"role": "admin"},
                    mock_handler,
                )
                body = parse_body(result)

                assert result.status_code == 200
                for user in body["users"]:
                    assert user["role"] == "admin"


class TestGetStats:
    """Tests for get admin stats endpoint."""

    def test_get_stats_success(self, admin_handler):
        """Test successful stats retrieval."""
        mock_handler = MockHandler()

        with patch("aragora.server.handlers.admin.admin.extract_user_from_request") as mock_extract:
            with patch("aragora.server.handlers.admin.admin.enforce_admin_mfa_policy") as mock_mfa:
                mock_ctx = MockAuthContext("admin_1", is_authenticated=True)
                mock_extract.return_value = mock_ctx
                mock_mfa.return_value = None

                result = admin_handler.handle("/api/admin/stats", {}, mock_handler)
                body = parse_body(result)

                assert result.status_code == 200
                assert "stats" in body
                assert "total_users" in body["stats"]
                assert "total_organizations" in body["stats"]


class TestUserManagement:
    """Tests for user management endpoints."""

    def setup_admin_auth(self, mock_extract, mock_mfa):
        """Helper to set up admin authentication."""
        mock_ctx = MockAuthContext("admin_1", is_authenticated=True)
        mock_extract.return_value = mock_ctx
        mock_mfa.return_value = None

    def test_deactivate_user_success(self, admin_handler, user_store):
        """Test successful user deactivation."""
        mock_handler = MockHandler(command="POST")

        with patch("aragora.server.handlers.admin.admin.extract_user_from_request") as mock_extract:
            with patch("aragora.server.handlers.admin.admin.enforce_admin_mfa_policy") as mock_mfa:
                self.setup_admin_auth(mock_extract, mock_mfa)

                result = admin_handler.handle(
                    "/api/admin/users/user_0/deactivate",
                    {},
                    mock_handler,
                    method="POST",
                )
                body = parse_body(result)

                assert result.status_code == 200
                assert body["success"] is True
                assert body["is_active"] is False

                # Verify user was actually deactivated
                user = user_store.get_user_by_id("user_0")
                assert user.is_active is False

    def test_deactivate_self_rejected(self, admin_handler):
        """Test that admin cannot deactivate themselves."""
        mock_handler = MockHandler(command="POST")

        with patch("aragora.server.handlers.admin.admin.extract_user_from_request") as mock_extract:
            with patch("aragora.server.handlers.admin.admin.enforce_admin_mfa_policy") as mock_mfa:
                self.setup_admin_auth(mock_extract, mock_mfa)

                result = admin_handler.handle(
                    "/api/admin/users/admin_1/deactivate",
                    {},
                    mock_handler,
                    method="POST",
                )

                assert result.status_code == 400
                assert "yourself" in parse_body(result)["error"].lower()

    def test_deactivate_nonexistent_user(self, admin_handler):
        """Test deactivating non-existent user."""
        mock_handler = MockHandler(command="POST")

        with patch("aragora.server.handlers.admin.admin.extract_user_from_request") as mock_extract:
            with patch("aragora.server.handlers.admin.admin.enforce_admin_mfa_policy") as mock_mfa:
                self.setup_admin_auth(mock_extract, mock_mfa)

                result = admin_handler.handle(
                    "/api/admin/users/nonexistent/deactivate",
                    {},
                    mock_handler,
                    method="POST",
                )

                assert result.status_code == 404

    def test_activate_user_success(self, admin_handler, user_store):
        """Test successful user activation."""
        # First deactivate the user
        user = user_store.get_user_by_id("user_0")
        user.is_active = False

        mock_handler = MockHandler(command="POST")

        with patch("aragora.server.handlers.admin.admin.extract_user_from_request") as mock_extract:
            with patch("aragora.server.handlers.admin.admin.enforce_admin_mfa_policy") as mock_mfa:
                self.setup_admin_auth(mock_extract, mock_mfa)

                result = admin_handler.handle(
                    "/api/admin/users/user_0/activate",
                    {},
                    mock_handler,
                    method="POST",
                )
                body = parse_body(result)

                assert result.status_code == 200
                assert body["success"] is True

                # Verify user was activated
                user = user_store.get_user_by_id("user_0")
                assert user.is_active is True

    def test_invalid_user_id_format_rejected(self, admin_handler):
        """Test that invalid user ID format is rejected."""
        mock_handler = MockHandler(command="POST")

        with patch("aragora.server.handlers.admin.admin.extract_user_from_request") as mock_extract:
            with patch("aragora.server.handlers.admin.admin.enforce_admin_mfa_policy") as mock_mfa:
                self.setup_admin_auth(mock_extract, mock_mfa)

                # Use path traversal characters that should fail validation
                result = admin_handler.handle(
                    "/api/admin/users/<script>alert(1)</script>/deactivate",
                    {},
                    mock_handler,
                    method="POST",
                )

                # Should be rejected with 400 for invalid format
                assert result.status_code == 400
                assert "Invalid" in parse_body(result)["error"]


class TestImpersonation:
    """Tests for user impersonation endpoint."""

    def test_impersonate_user_success(self, admin_handler, user_store):
        """Test successful user impersonation."""
        mock_handler = MockHandler(command="POST")

        with patch("aragora.server.handlers.admin.admin.extract_user_from_request") as mock_extract:
            with patch("aragora.server.handlers.admin.admin.enforce_admin_mfa_policy") as mock_mfa:
                with patch("aragora.server.handlers.admin.admin.create_access_token") as mock_token:
                    mock_ctx = MockAuthContext("admin_1", is_authenticated=True)
                    mock_extract.return_value = mock_ctx
                    mock_mfa.return_value = None
                    mock_token.return_value = "impersonation_token_123"

                    result = admin_handler.handle(
                        "/api/admin/impersonate/user_0",
                        {},
                        mock_handler,
                        method="POST",
                    )
                    body = parse_body(result)

                    assert result.status_code == 200
                    assert "token" in body
                    assert body["expires_in"] == 3600
                    assert body["target_user"]["id"] == "user_0"
                    assert "warning" in body

    def test_impersonate_records_audit(self, admin_handler, user_store):
        """Test that impersonation is recorded in audit log."""
        mock_handler = MockHandler(command="POST")

        with patch("aragora.server.handlers.admin.admin.extract_user_from_request") as mock_extract:
            with patch("aragora.server.handlers.admin.admin.enforce_admin_mfa_policy") as mock_mfa:
                with patch("aragora.server.handlers.admin.admin.create_access_token") as mock_token:
                    mock_ctx = MockAuthContext("admin_1", is_authenticated=True)
                    mock_extract.return_value = mock_ctx
                    mock_mfa.return_value = None
                    mock_token.return_value = "token"

                    result = admin_handler.handle(
                        "/api/admin/impersonate/user_0",
                        {},
                        mock_handler,
                        method="POST",
                    )

                    assert result.status_code == 200
                    # Check audit event was recorded
                    assert len(user_store._audit_events) == 1
                    event = user_store._audit_events[0]
                    assert event["event_type"] == "admin_impersonate"
                    assert event["resource_id"] == "user_0"

    def test_impersonate_nonexistent_user(self, admin_handler):
        """Test impersonating non-existent user."""
        mock_handler = MockHandler(command="POST")

        with patch("aragora.server.handlers.admin.admin.extract_user_from_request") as mock_extract:
            with patch("aragora.server.handlers.admin.admin.enforce_admin_mfa_policy") as mock_mfa:
                mock_ctx = MockAuthContext("admin_1", is_authenticated=True)
                mock_extract.return_value = mock_ctx
                mock_mfa.return_value = None

                result = admin_handler.handle(
                    "/api/admin/impersonate/nonexistent",
                    {},
                    mock_handler,
                    method="POST",
                )

                assert result.status_code == 404


class TestMethodNotAllowed:
    """Tests for method not allowed handling."""

    def test_unsupported_method_returns_405(self, admin_handler):
        """Test that unsupported methods return 405."""
        mock_handler = MockHandler(command="DELETE")

        with patch("aragora.server.handlers.admin.admin.extract_user_from_request") as mock_extract:
            with patch("aragora.server.handlers.admin.admin.enforce_admin_mfa_policy") as mock_mfa:
                mock_ctx = MockAuthContext("admin_1", is_authenticated=True)
                mock_extract.return_value = mock_ctx
                mock_mfa.return_value = None

                result = admin_handler.handle(
                    "/api/admin/unknown",
                    {},
                    mock_handler,
                    method="DELETE",
                )

                assert result.status_code == 405


class TestSystemMetrics:
    """Tests for system metrics endpoint."""

    def test_get_system_metrics(self, admin_handler):
        """Test system metrics retrieval."""
        mock_handler = MockHandler()

        with patch("aragora.server.handlers.admin.admin.extract_user_from_request") as mock_extract:
            with patch("aragora.server.handlers.admin.admin.enforce_admin_mfa_policy") as mock_mfa:
                mock_ctx = MockAuthContext("admin_1", is_authenticated=True)
                mock_extract.return_value = mock_ctx
                mock_mfa.return_value = None

                result = admin_handler.handle("/api/admin/system/metrics", {}, mock_handler)
                body = parse_body(result)

                assert result.status_code == 200
                assert "metrics" in body
                assert "timestamp" in body["metrics"]
                assert "users" in body["metrics"]
