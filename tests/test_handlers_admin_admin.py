"""
Tests for AdminHandler endpoints.

Endpoints tested:
- GET /api/admin/organizations - List all organizations
- GET /api/admin/users - List all users
- GET /api/admin/stats - Get system-wide statistics
- POST /api/admin/impersonate/:user_id - Create impersonation token
- POST /api/admin/users/:user_id/deactivate - Deactivate a user
- POST /api/admin/users/:user_id/activate - Activate a user
- POST /api/admin/users/:user_id/unlock - Unlock a locked user account
"""

import json
import pytest
from unittest.mock import MagicMock, Mock, patch

from aragora.server.handlers.admin.admin import AdminHandler, ADMIN_ROLES
from aragora.server.handlers.base import clear_cache


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def mock_user_store():
    """Create a mock user store."""
    store = MagicMock()

    # Mock get_user_by_id for admin user
    admin_user = MagicMock()
    admin_user.id = "admin-user-123"
    admin_user.email = "admin@example.com"
    admin_user.name = "Admin User"
    admin_user.role = "admin"
    admin_user.org_id = "org-123"
    admin_user.is_active = True
    admin_user.mfa_enabled = True
    admin_user.to_dict.return_value = {
        "id": "admin-user-123",
        "email": "admin@example.com",
        "name": "Admin User",
        "role": "admin",
    }

    # Mock target user for operations
    target_user = MagicMock()
    target_user.id = "target-user-456"
    target_user.email = "target@example.com"
    target_user.name = "Target User"
    target_user.role = "member"
    target_user.org_id = "org-123"
    target_user.is_active = True
    target_user.to_dict.return_value = {
        "id": "target-user-456",
        "email": "target@example.com",
        "name": "Target User",
        "role": "member",
    }

    def get_user_by_id(user_id):
        if user_id == "admin-user-123":
            return admin_user
        elif user_id == "target-user-456":
            return target_user
        return None

    store.get_user_by_id = MagicMock(side_effect=get_user_by_id)
    store.list_all_organizations = MagicMock(return_value=([], 0))
    store.list_all_users = MagicMock(return_value=([], 0))
    store.get_admin_stats = MagicMock(return_value={"total_users": 10})
    store.update_user = MagicMock()
    store.record_audit_event = MagicMock()
    store.reset_failed_login_attempts = MagicMock(return_value=True)

    return store


@pytest.fixture
def mock_handler():
    """Create a mock HTTP handler with authentication context."""
    handler = MagicMock()
    handler.command = "GET"
    handler.headers = {"Authorization": "Bearer test-token"}
    handler.client_address = ("127.0.0.1", 12345)
    handler.request_body = b"{}"
    return handler


@pytest.fixture
def mock_auth_context():
    """Create a mock authenticated admin context."""
    ctx = MagicMock()
    ctx.is_authenticated = True
    ctx.user_id = "admin-user-123"
    ctx.email = "admin@example.com"
    ctx.org_id = "org-123"
    return ctx


@pytest.fixture
def admin_handler(mock_user_store):
    """Create an AdminHandler with mock dependencies."""
    ctx = {
        "user_store": mock_user_store,
        "nomic_dir": ".nomic",
    }
    return AdminHandler(ctx)


@pytest.fixture
def admin_handler_no_store():
    """Create an AdminHandler without user store."""
    ctx = {
        "user_store": None,
    }
    return AdminHandler(ctx)


@pytest.fixture(autouse=True)
def clear_caches():
    """Clear caches before and after each test."""
    clear_cache()
    yield
    clear_cache()


# ============================================================================
# Route Matching Tests
# ============================================================================


class TestAdminRouting:
    """Tests for route matching."""

    def test_can_handle_admin_organizations(self, admin_handler):
        """Handler can handle /api/admin/organizations."""
        assert admin_handler.can_handle("/api/admin/organizations") is True

    def test_can_handle_admin_users(self, admin_handler):
        """Handler can handle /api/admin/users."""
        assert admin_handler.can_handle("/api/admin/users") is True

    def test_can_handle_admin_stats(self, admin_handler):
        """Handler can handle /api/admin/stats."""
        assert admin_handler.can_handle("/api/admin/stats") is True

    def test_can_handle_admin_impersonate(self, admin_handler):
        """Handler can handle /api/admin/impersonate/:user_id."""
        assert admin_handler.can_handle("/api/admin/impersonate/user-123") is True

    def test_can_handle_admin_nomic_status(self, admin_handler):
        """Handler can handle /api/admin/nomic/status."""
        assert admin_handler.can_handle("/api/admin/nomic/status") is True

    def test_cannot_handle_non_admin_routes(self, admin_handler):
        """Handler doesn't handle non-admin routes."""
        assert admin_handler.can_handle("/api/debates") is False
        assert admin_handler.can_handle("/api/users") is False
        assert admin_handler.can_handle("/api/agents") is False

    def test_admin_roles_defined(self):
        """ADMIN_ROLES contains expected roles."""
        assert "admin" in ADMIN_ROLES
        assert "owner" in ADMIN_ROLES
        assert "member" not in ADMIN_ROLES


# ============================================================================
# Authentication Tests
# ============================================================================


class TestAdminAuth:
    """Tests for admin authentication and authorization."""

    def test_returns_503_when_no_user_store(self, admin_handler_no_store, mock_handler):
        """Returns 503 when user store is unavailable."""
        result = admin_handler_no_store.handle("/api/admin/stats", {}, mock_handler)

        assert result is not None
        assert result.status_code == 503
        data = json.loads(result.body)
        assert "unavailable" in data["error"].lower()

    @patch("aragora.server.handlers.admin.admin.extract_user_from_request")
    def test_returns_401_when_not_authenticated(self, mock_extract, admin_handler, mock_handler):
        """Returns 401 when user is not authenticated."""
        mock_auth = MagicMock()
        mock_auth.is_authenticated = False
        mock_extract.return_value = mock_auth

        result = admin_handler.handle("/api/admin/stats", {}, mock_handler)

        assert result is not None
        assert result.status_code == 401
        data = json.loads(result.body)
        assert "authenticated" in data["error"].lower()

    @patch("aragora.server.handlers.admin.admin.enforce_admin_mfa_policy")
    @patch("aragora.server.handlers.admin.admin.extract_user_from_request")
    def test_returns_403_when_not_admin(
        self, mock_extract, mock_mfa, admin_handler, mock_handler, mock_user_store
    ):
        """Returns 403 when user does not have admin role."""
        mock_auth = MagicMock()
        mock_auth.is_authenticated = True
        mock_auth.user_id = "regular-user"
        mock_extract.return_value = mock_auth

        # Create non-admin user
        non_admin = MagicMock()
        non_admin.role = "member"
        mock_user_store.get_user_by_id.return_value = non_admin

        result = admin_handler.handle("/api/admin/stats", {}, mock_handler)

        assert result is not None
        assert result.status_code == 403
        data = json.loads(result.body)
        assert "admin" in data["error"].lower()

    @patch("aragora.server.handlers.admin.admin.enforce_admin_mfa_policy")
    @patch("aragora.server.handlers.admin.admin.extract_user_from_request")
    def test_returns_403_when_mfa_not_enabled(
        self, mock_extract, mock_mfa, admin_handler, mock_handler, mock_auth_context
    ):
        """Returns 403 when admin does not have MFA enabled."""
        mock_extract.return_value = mock_auth_context
        mock_mfa.return_value = {"reason": "MFA not enabled", "action": "enable_mfa"}

        result = admin_handler.handle("/api/admin/stats", {}, mock_handler)

        assert result is not None
        assert result.status_code == 403
        data = json.loads(result.body)
        # Response uses structured error format when code is provided
        # {"error": {"code": "...", "message": "..."}} or {"code": "...", "message": "..."}
        error_obj = data.get("error", data)
        if isinstance(error_obj, dict):
            error_text = error_obj.get("message", "")
            error_code = error_obj.get("code") or data.get("code")
        else:
            error_text = str(error_obj)
            error_code = data.get("code")
        assert "MFA" in error_text
        assert error_code == "ADMIN_MFA_REQUIRED"


# ============================================================================
# GET /api/admin/stats Tests
# ============================================================================


class TestGetAdminStats:
    """Tests for GET /api/admin/stats endpoint."""

    @patch("aragora.server.handlers.admin.admin.enforce_admin_mfa_policy")
    @patch("aragora.server.handlers.admin.admin.extract_user_from_request")
    def test_get_stats_success(
        self, mock_extract, mock_mfa, admin_handler, mock_handler, mock_auth_context
    ):
        """Successfully returns admin stats."""
        mock_extract.return_value = mock_auth_context
        mock_mfa.return_value = None  # MFA compliant

        result = admin_handler.handle("/api/admin/stats", {}, mock_handler)

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert "stats" in data
        assert data["stats"]["total_users"] == 10


# ============================================================================
# GET /api/admin/users Tests
# ============================================================================


class TestListUsers:
    """Tests for GET /api/admin/users endpoint."""

    @patch("aragora.server.handlers.admin.admin.enforce_admin_mfa_policy")
    @patch("aragora.server.handlers.admin.admin.extract_user_from_request")
    def test_list_users_success(
        self,
        mock_extract,
        mock_mfa,
        admin_handler,
        mock_handler,
        mock_auth_context,
        mock_user_store,
    ):
        """Successfully returns paginated user list."""
        mock_extract.return_value = mock_auth_context
        mock_mfa.return_value = None

        user = MagicMock()
        user.to_dict.return_value = {
            "id": "user-1",
            "email": "user@example.com",
            "password_hash": "secret",
            "api_key": "api-key",
        }
        mock_user_store.list_all_users.return_value = ([user], 1)

        result = admin_handler.handle("/api/admin/users", {}, mock_handler)

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert "users" in data
        assert data["total"] == 1
        # Sensitive fields should be removed
        assert "password_hash" not in data["users"][0]
        assert "api_key" not in data["users"][0]

    @patch("aragora.server.handlers.admin.admin.enforce_admin_mfa_policy")
    @patch("aragora.server.handlers.admin.admin.extract_user_from_request")
    def test_list_users_with_pagination(
        self,
        mock_extract,
        mock_mfa,
        admin_handler,
        mock_handler,
        mock_auth_context,
        mock_user_store,
    ):
        """Respects pagination parameters."""
        mock_extract.return_value = mock_auth_context
        mock_mfa.return_value = None
        mock_user_store.list_all_users.return_value = ([], 0)

        result = admin_handler.handle(
            "/api/admin/users", {"limit": "25", "offset": "10", "role": "admin"}, mock_handler
        )

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["limit"] == 25
        assert data["offset"] == 10
        mock_user_store.list_all_users.assert_called_once()


# ============================================================================
# POST /api/admin/users/:user_id/deactivate Tests
# ============================================================================


def _mock_allowed_decision():
    """Create a mock decision that allows access."""
    decision = MagicMock()
    decision.allowed = True
    decision.reason = "Test mock: allowed"
    return decision


class TestDeactivateUser:
    """Tests for POST /api/admin/users/:user_id/deactivate endpoint."""

    @patch("aragora.server.handlers.admin.admin.check_permission")
    @patch("aragora.server.handlers.admin.admin.enforce_admin_mfa_policy")
    @patch("aragora.server.handlers.admin.admin.extract_user_from_request")
    def test_deactivate_user_success(
        self,
        mock_extract,
        mock_mfa,
        mock_check_permission,
        admin_handler,
        mock_handler,
        mock_auth_context,
        mock_user_store,
    ):
        """Successfully deactivates a user."""
        mock_extract.return_value = mock_auth_context
        mock_mfa.return_value = None
        mock_check_permission.return_value = _mock_allowed_decision()
        mock_handler.command = "POST"

        result = admin_handler.handle(
            "/api/admin/users/target-user-456/deactivate", {}, mock_handler, method="POST"
        )

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["success"] is True
        assert data["user_id"] == "target-user-456"
        assert data["is_active"] is False
        mock_user_store.update_user.assert_called_with("target-user-456", is_active=False)

    @patch("aragora.server.handlers.admin.admin.check_permission")
    @patch("aragora.server.handlers.admin.admin.enforce_admin_mfa_policy")
    @patch("aragora.server.handlers.admin.admin.extract_user_from_request")
    def test_cannot_deactivate_self(
        self, mock_extract, mock_mfa, mock_check_permission, admin_handler, mock_handler, mock_auth_context
    ):
        """Cannot deactivate yourself."""
        mock_extract.return_value = mock_auth_context
        mock_mfa.return_value = None
        mock_check_permission.return_value = _mock_allowed_decision()
        mock_handler.command = "POST"

        result = admin_handler.handle(
            "/api/admin/users/admin-user-123/deactivate", {}, mock_handler, method="POST"
        )

        assert result is not None
        assert result.status_code == 400
        data = json.loads(result.body)
        assert "yourself" in data["error"].lower()

    @patch("aragora.server.handlers.admin.admin.check_permission")
    @patch("aragora.server.handlers.admin.admin.enforce_admin_mfa_policy")
    @patch("aragora.server.handlers.admin.admin.extract_user_from_request")
    def test_deactivate_user_not_found(
        self, mock_extract, mock_mfa, mock_check_permission, admin_handler, mock_handler, mock_auth_context
    ):
        """Returns 404 when user does not exist."""
        mock_extract.return_value = mock_auth_context
        mock_mfa.return_value = None
        mock_check_permission.return_value = _mock_allowed_decision()
        mock_handler.command = "POST"

        result = admin_handler.handle(
            "/api/admin/users/nonexistent-user/deactivate", {}, mock_handler, method="POST"
        )

        assert result is not None
        assert result.status_code == 404
        data = json.loads(result.body)
        assert "not found" in data["error"].lower()


# ============================================================================
# POST /api/admin/users/:user_id/activate Tests
# ============================================================================


class TestActivateUser:
    """Tests for POST /api/admin/users/:user_id/activate endpoint."""

    @patch("aragora.server.handlers.admin.admin.check_permission")
    @patch("aragora.server.handlers.admin.admin.enforce_admin_mfa_policy")
    @patch("aragora.server.handlers.admin.admin.extract_user_from_request")
    def test_activate_user_success(
        self,
        mock_extract,
        mock_mfa,
        mock_check_permission,
        admin_handler,
        mock_handler,
        mock_auth_context,
        mock_user_store,
    ):
        """Successfully activates a user."""
        mock_extract.return_value = mock_auth_context
        mock_mfa.return_value = None
        mock_check_permission.return_value = _mock_allowed_decision()
        mock_handler.command = "POST"

        result = admin_handler.handle(
            "/api/admin/users/target-user-456/activate", {}, mock_handler, method="POST"
        )

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["success"] is True
        assert data["user_id"] == "target-user-456"
        assert data["is_active"] is True
        mock_user_store.update_user.assert_called_with("target-user-456", is_active=True)


# ============================================================================
# Input Validation Tests
# ============================================================================


class TestInputValidation:
    """Tests for input validation."""

    def test_invalid_user_id_format_impersonate(self, admin_handler, mock_handler):
        """Rejects invalid user ID format for impersonate."""
        mock_handler.command = "POST"

        result = admin_handler.handle(
            "/api/admin/impersonate/invalid<script>id", {}, mock_handler, method="POST"
        )

        assert result is not None
        assert result.status_code == 400
        data = json.loads(result.body)
        assert "invalid" in data["error"].lower()

    @patch("aragora.server.handlers.admin.admin.enforce_admin_mfa_policy")
    @patch("aragora.server.handlers.admin.admin.extract_user_from_request")
    def test_invalid_user_id_format_deactivate(
        self, mock_extract, mock_mfa, admin_handler, mock_handler, mock_auth_context
    ):
        """Rejects invalid user ID format for deactivate."""
        mock_extract.return_value = mock_auth_context
        mock_mfa.return_value = None
        mock_handler.command = "POST"

        # Use a user ID with special characters that violate SAFE_ID_PATTERN
        result = admin_handler.handle(
            "/api/admin/users/invalid!user@id/deactivate", {}, mock_handler, method="POST"
        )

        assert result is not None
        # Invalid user ID format is rejected with 400
        assert result.status_code == 400
        data = json.loads(result.body)
        assert "invalid" in data["error"].lower()


# ============================================================================
# Method Not Allowed Tests
# ============================================================================


class TestMethodNotAllowed:
    """Tests for method not allowed responses."""

    @patch("aragora.server.handlers.admin.admin.enforce_admin_mfa_policy")
    @patch("aragora.server.handlers.admin.admin.extract_user_from_request")
    def test_post_to_get_endpoint_returns_405(
        self, mock_extract, mock_mfa, admin_handler, mock_handler, mock_auth_context
    ):
        """POST to GET-only endpoint returns 405."""
        mock_extract.return_value = mock_auth_context
        mock_mfa.return_value = None
        mock_handler.command = "POST"

        # /api/admin/stats is GET only
        result = admin_handler.handle("/api/admin/stats", {}, mock_handler, method="POST")

        assert result is not None
        assert result.status_code == 405
        data = json.loads(result.body)
        assert "not allowed" in data["error"].lower()


# ============================================================================
# Handler Import Tests
# ============================================================================


class TestAdminHandlerImport:
    """Test AdminHandler import and export."""

    def test_handler_importable(self):
        """AdminHandler can be imported from admin module."""
        from aragora.server.handlers.admin.admin import AdminHandler

        assert AdminHandler is not None

    def test_admin_roles_exported(self):
        """ADMIN_ROLES constant is exported."""
        from aragora.server.handlers.admin.admin import ADMIN_ROLES

        assert ADMIN_ROLES is not None
        assert isinstance(ADMIN_ROLES, set)
