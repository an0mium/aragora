"""
Tests for AdminHandler - administrative endpoints.

Tests cover:
- GET /api/admin/organizations - List all organizations
- GET /api/admin/users - List all users
- GET /api/admin/stats - Get system-wide statistics
- GET /api/admin/system/metrics - Get aggregated system metrics
- GET /api/admin/revenue - Get revenue statistics
- POST /api/admin/impersonate/:user_id - Create impersonation token
- POST /api/admin/users/:user_id/deactivate - Deactivate a user
- POST /api/admin/users/:user_id/activate - Activate a user

Security tests:
- Admin role validation
- Authentication required
- Input validation
- Audit logging
"""

import json
import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime

from aragora.server.handlers.admin import AdminHandler, ADMIN_ROLES


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def mock_admin_user():
    """Create a mock admin user object."""
    user = Mock()
    user.id = "admin-123"
    user.email = "admin@example.com"
    user.name = "Admin User"
    user.org_id = "org-456"
    user.role = "admin"
    user.is_active = True
    user.to_dict = Mock(
        return_value={
            "id": "admin-123",
            "email": "admin@example.com",
            "name": "Admin User",
            "org_id": "org-456",
            "role": "admin",
        }
    )
    return user


@pytest.fixture
def mock_owner_user():
    """Create a mock owner user object."""
    user = Mock()
    user.id = "owner-123"
    user.email = "owner@example.com"
    user.name = "Owner User"
    user.org_id = "org-456"
    user.role = "owner"
    user.is_active = True
    user.to_dict = Mock(
        return_value={
            "id": "owner-123",
            "email": "owner@example.com",
            "name": "Owner User",
            "org_id": "org-456",
            "role": "owner",
        }
    )
    return user


@pytest.fixture
def mock_regular_user():
    """Create a mock regular user object (non-admin)."""
    user = Mock()
    user.id = "user-123"
    user.email = "user@example.com"
    user.name = "Regular User"
    user.org_id = "org-456"
    user.role = "member"
    user.is_active = True
    user.to_dict = Mock(
        return_value={
            "id": "user-123",
            "email": "user@example.com",
            "name": "Regular User",
            "org_id": "org-456",
            "role": "member",
        }
    )
    return user


@pytest.fixture
def mock_target_user():
    """Create a mock target user for impersonation/activation tests."""
    user = Mock()
    user.id = "target-456"
    user.email = "target@example.com"
    user.name = "Target User"
    user.org_id = "org-789"
    user.role = "member"
    user.is_active = True
    user.to_dict = Mock(
        return_value={
            "id": "target-456",
            "email": "target@example.com",
            "name": "Target User",
            "org_id": "org-789",
            "role": "member",
        }
    )
    return user


@pytest.fixture
def mock_organization():
    """Create a mock organization object."""
    org = Mock()
    org.id = "org-456"
    org.name = "Test Org"
    org.slug = "test-org"
    org.tier = "pro"
    org.to_dict = Mock(
        return_value={
            "id": "org-456",
            "name": "Test Org",
            "slug": "test-org",
            "tier": "pro",
        }
    )
    return org


@pytest.fixture
def mock_user_store(mock_admin_user, mock_target_user, mock_organization):
    """Create a mock user store with admin-related methods."""
    store = Mock()

    # User lookups
    store.get_user_by_id = Mock(side_effect=lambda uid: {
        "admin-123": mock_admin_user,
        "target-456": mock_target_user,
    }.get(uid))
    store.get_user_by_email = Mock(return_value=None)

    # Organization listing
    store.list_all_organizations = Mock(return_value=([mock_organization], 1))

    # User listing
    store.list_all_users = Mock(return_value=([mock_admin_user, mock_target_user], 2))

    # Admin stats
    store.get_admin_stats = Mock(return_value={
        "total_users": 100,
        "total_organizations": 20,
        "tier_distribution": {
            "free": 10,
            "pro": 8,
            "enterprise": 2,
        },
        "active_users": 85,
    })

    # User updates
    store.update_user = Mock(return_value=True)

    # Audit logging
    store.record_audit_event = Mock(return_value=1)

    return store


@pytest.fixture
def mock_handler():
    """Create a mock HTTP handler."""
    handler = Mock()
    handler.command = "GET"
    handler.headers = {"Content-Type": "application/json"}
    handler.client_address = ("192.168.1.1", 12345)
    return handler


@pytest.fixture
def admin_handler(mock_user_store):
    """Create AdminHandler with mock dependencies."""
    ctx = {"user_store": mock_user_store}
    return AdminHandler(ctx)


@pytest.fixture
def admin_handler_no_store():
    """Create AdminHandler without user store."""
    ctx = {}
    return AdminHandler(ctx)


@pytest.fixture(autouse=True)
def clear_rate_limiters():
    """Clear rate limiters before each test."""
    try:
        from aragora.server.handlers.utils.rate_limit import _limiters
        for limiter in _limiters.values():
            limiter._buckets.clear()
    except (ImportError, AttributeError):
        pass
    yield
    try:
        from aragora.server.handlers.utils.rate_limit import _limiters
        for limiter in _limiters.values():
            limiter._buckets.clear()
    except (ImportError, AttributeError):
        pass


# ============================================================================
# Authentication and Authorization Tests
# ============================================================================


class TestAdminAuthorization:
    """Tests for admin role validation."""

    def test_unauthenticated_request_rejected(self, admin_handler, mock_handler):
        """Test that unauthenticated requests are rejected."""
        with patch("aragora.server.handlers.admin.extract_user_from_request") as mock_extract:
            mock_auth_ctx = Mock()
            mock_auth_ctx.is_authenticated = False
            mock_extract.return_value = mock_auth_ctx

            result = admin_handler.handle("/api/admin/organizations", {}, mock_handler)

            assert result.status_code == 401
            body = json.loads(result.body)
            assert "authenticated" in body.get("error", "").lower()

    def test_non_admin_user_rejected(self, admin_handler, mock_handler, mock_user_store, mock_regular_user):
        """Test that non-admin users are rejected with 403."""
        mock_user_store.get_user_by_id = Mock(return_value=mock_regular_user)

        with patch("aragora.server.handlers.admin.extract_user_from_request") as mock_extract:
            mock_auth_ctx = Mock()
            mock_auth_ctx.is_authenticated = True
            mock_auth_ctx.user_id = "user-123"
            mock_extract.return_value = mock_auth_ctx

            result = admin_handler.handle("/api/admin/organizations", {}, mock_handler)

            assert result.status_code == 403
            body = json.loads(result.body)
            assert "admin" in body.get("error", "").lower()

    def test_admin_user_allowed(self, admin_handler, mock_handler, mock_admin_user):
        """Test that admin users can access admin endpoints."""
        with patch("aragora.server.handlers.admin.extract_user_from_request") as mock_extract:
            mock_auth_ctx = Mock()
            mock_auth_ctx.is_authenticated = True
            mock_auth_ctx.user_id = "admin-123"
            mock_extract.return_value = mock_auth_ctx

            result = admin_handler.handle("/api/admin/organizations", {}, mock_handler)

            assert result.status_code == 200

    def test_owner_user_allowed(self, admin_handler, mock_handler, mock_user_store, mock_owner_user):
        """Test that owner users can access admin endpoints."""
        mock_user_store.get_user_by_id = Mock(return_value=mock_owner_user)

        with patch("aragora.server.handlers.admin.extract_user_from_request") as mock_extract:
            mock_auth_ctx = Mock()
            mock_auth_ctx.is_authenticated = True
            mock_auth_ctx.user_id = "owner-123"
            mock_extract.return_value = mock_auth_ctx

            result = admin_handler.handle("/api/admin/organizations", {}, mock_handler)

            assert result.status_code == 200

    def test_service_unavailable_without_user_store(self, admin_handler_no_store, mock_handler):
        """Test 503 response when user store is unavailable."""
        result = admin_handler_no_store.handle("/api/admin/organizations", {}, mock_handler)

        assert result.status_code == 503
        body = json.loads(result.body)
        assert "unavailable" in body.get("error", "").lower()


class TestAdminRoles:
    """Tests for ADMIN_ROLES configuration."""

    def test_admin_roles_contains_admin(self):
        """Test that admin role is in ADMIN_ROLES."""
        assert "admin" in ADMIN_ROLES

    def test_admin_roles_contains_owner(self):
        """Test that owner role is in ADMIN_ROLES."""
        assert "owner" in ADMIN_ROLES

    def test_member_not_in_admin_roles(self):
        """Test that member role is not in ADMIN_ROLES."""
        assert "member" not in ADMIN_ROLES


# ============================================================================
# Organization Listing Tests
# ============================================================================


class TestListOrganizations:
    """Tests for GET /api/admin/organizations."""

    def test_list_organizations_success(self, admin_handler, mock_handler, mock_user_store):
        """Test successful organization listing."""
        with patch("aragora.server.handlers.admin.extract_user_from_request") as mock_extract:
            mock_auth_ctx = Mock()
            mock_auth_ctx.is_authenticated = True
            mock_auth_ctx.user_id = "admin-123"
            mock_extract.return_value = mock_auth_ctx

            result = admin_handler.handle("/api/admin/organizations", {}, mock_handler)

            assert result.status_code == 200
            body = json.loads(result.body)
            assert "organizations" in body
            assert "total" in body
            assert body["total"] == 1

    def test_list_organizations_with_pagination(self, admin_handler, mock_handler, mock_user_store):
        """Test organization listing with pagination parameters."""
        with patch("aragora.server.handlers.admin.extract_user_from_request") as mock_extract:
            mock_auth_ctx = Mock()
            mock_auth_ctx.is_authenticated = True
            mock_auth_ctx.user_id = "admin-123"
            mock_extract.return_value = mock_auth_ctx

            query_params = {"limit": "25", "offset": "10"}
            result = admin_handler.handle("/api/admin/organizations", query_params, mock_handler)

            assert result.status_code == 200
            body = json.loads(result.body)
            assert body["limit"] == 25
            assert body["offset"] == 10

            # Verify store was called with correct params
            mock_user_store.list_all_organizations.assert_called_with(
                limit=25, offset=10, tier_filter=None
            )

    def test_list_organizations_max_limit(self, admin_handler, mock_handler, mock_user_store):
        """Test that limit is capped at 100."""
        with patch("aragora.server.handlers.admin.extract_user_from_request") as mock_extract:
            mock_auth_ctx = Mock()
            mock_auth_ctx.is_authenticated = True
            mock_auth_ctx.user_id = "admin-123"
            mock_extract.return_value = mock_auth_ctx

            query_params = {"limit": "500"}
            result = admin_handler.handle("/api/admin/organizations", query_params, mock_handler)

            assert result.status_code == 200
            body = json.loads(result.body)
            assert body["limit"] == 100  # Capped at max

    def test_list_organizations_tier_filter(self, admin_handler, mock_handler, mock_user_store):
        """Test organization listing with tier filter."""
        with patch("aragora.server.handlers.admin.extract_user_from_request") as mock_extract:
            mock_auth_ctx = Mock()
            mock_auth_ctx.is_authenticated = True
            mock_auth_ctx.user_id = "admin-123"
            mock_extract.return_value = mock_auth_ctx

            query_params = {"tier": "pro"}
            result = admin_handler.handle("/api/admin/organizations", query_params, mock_handler)

            assert result.status_code == 200
            mock_user_store.list_all_organizations.assert_called_with(
                limit=50, offset=0, tier_filter="pro"
            )


# ============================================================================
# User Listing Tests
# ============================================================================


class TestListUsers:
    """Tests for GET /api/admin/users."""

    def test_list_users_success(self, admin_handler, mock_handler, mock_user_store):
        """Test successful user listing."""
        with patch("aragora.server.handlers.admin.extract_user_from_request") as mock_extract:
            mock_auth_ctx = Mock()
            mock_auth_ctx.is_authenticated = True
            mock_auth_ctx.user_id = "admin-123"
            mock_extract.return_value = mock_auth_ctx

            result = admin_handler.handle("/api/admin/users", {}, mock_handler)

            assert result.status_code == 200
            body = json.loads(result.body)
            assert "users" in body
            assert "total" in body

    def test_list_users_excludes_sensitive_fields(self, admin_handler, mock_handler, mock_user_store, mock_admin_user):
        """Test that sensitive fields are excluded from user listing."""
        mock_admin_user.to_dict = Mock(return_value={
            "id": "admin-123",
            "email": "admin@example.com",
            "password_hash": "secret",
            "password_salt": "salt",
            "api_key": "key123",
            "api_key_hash": "hash123",
        })

        with patch("aragora.server.handlers.admin.extract_user_from_request") as mock_extract:
            mock_auth_ctx = Mock()
            mock_auth_ctx.is_authenticated = True
            mock_auth_ctx.user_id = "admin-123"
            mock_extract.return_value = mock_auth_ctx

            result = admin_handler.handle("/api/admin/users", {}, mock_handler)

            assert result.status_code == 200
            body = json.loads(result.body)
            for user in body["users"]:
                assert "password_hash" not in user
                assert "password_salt" not in user
                assert "api_key" not in user
                assert "api_key_hash" not in user

    def test_list_users_with_org_filter(self, admin_handler, mock_handler, mock_user_store):
        """Test user listing with organization filter."""
        with patch("aragora.server.handlers.admin.extract_user_from_request") as mock_extract:
            mock_auth_ctx = Mock()
            mock_auth_ctx.is_authenticated = True
            mock_auth_ctx.user_id = "admin-123"
            mock_extract.return_value = mock_auth_ctx

            query_params = {"org_id": "org-456"}
            result = admin_handler.handle("/api/admin/users", query_params, mock_handler)

            assert result.status_code == 200
            mock_user_store.list_all_users.assert_called_with(
                limit=50, offset=0, org_id_filter="org-456",
                role_filter=None, active_only=False
            )

    def test_list_users_with_role_filter(self, admin_handler, mock_handler, mock_user_store):
        """Test user listing with role filter."""
        with patch("aragora.server.handlers.admin.extract_user_from_request") as mock_extract:
            mock_auth_ctx = Mock()
            mock_auth_ctx.is_authenticated = True
            mock_auth_ctx.user_id = "admin-123"
            mock_extract.return_value = mock_auth_ctx

            query_params = {"role": "admin"}
            result = admin_handler.handle("/api/admin/users", query_params, mock_handler)

            assert result.status_code == 200
            mock_user_store.list_all_users.assert_called_with(
                limit=50, offset=0, org_id_filter=None,
                role_filter="admin", active_only=False
            )

    def test_list_users_active_only(self, admin_handler, mock_handler, mock_user_store):
        """Test user listing with active_only filter."""
        with patch("aragora.server.handlers.admin.extract_user_from_request") as mock_extract:
            mock_auth_ctx = Mock()
            mock_auth_ctx.is_authenticated = True
            mock_auth_ctx.user_id = "admin-123"
            mock_extract.return_value = mock_auth_ctx

            query_params = {"active_only": "true"}
            result = admin_handler.handle("/api/admin/users", query_params, mock_handler)

            assert result.status_code == 200
            mock_user_store.list_all_users.assert_called_with(
                limit=50, offset=0, org_id_filter=None,
                role_filter=None, active_only=True
            )


# ============================================================================
# Admin Stats Tests
# ============================================================================


class TestGetStats:
    """Tests for GET /api/admin/stats."""

    def test_get_stats_success(self, admin_handler, mock_handler, mock_user_store):
        """Test successful stats retrieval."""
        with patch("aragora.server.handlers.admin.extract_user_from_request") as mock_extract:
            mock_auth_ctx = Mock()
            mock_auth_ctx.is_authenticated = True
            mock_auth_ctx.user_id = "admin-123"
            mock_extract.return_value = mock_auth_ctx

            result = admin_handler.handle("/api/admin/stats", {}, mock_handler)

            assert result.status_code == 200
            body = json.loads(result.body)
            assert "stats" in body
            assert body["stats"]["total_users"] == 100
            assert body["stats"]["total_organizations"] == 20


# ============================================================================
# System Metrics Tests
# ============================================================================


class TestGetSystemMetrics:
    """Tests for GET /api/admin/system/metrics."""

    def test_get_system_metrics_success(self, admin_handler, mock_handler, mock_user_store):
        """Test successful system metrics retrieval."""
        with patch("aragora.server.handlers.admin.extract_user_from_request") as mock_extract:
            mock_auth_ctx = Mock()
            mock_auth_ctx.is_authenticated = True
            mock_auth_ctx.user_id = "admin-123"
            mock_extract.return_value = mock_auth_ctx

            result = admin_handler.handle("/api/admin/system/metrics", {}, mock_handler)

            assert result.status_code == 200
            body = json.loads(result.body)
            assert "metrics" in body
            assert "timestamp" in body["metrics"]
            assert "users" in body["metrics"]

    def test_system_metrics_includes_users(self, admin_handler, mock_handler, mock_user_store):
        """Test that system metrics include user statistics."""
        with patch("aragora.server.handlers.admin.extract_user_from_request") as mock_extract:
            mock_auth_ctx = Mock()
            mock_auth_ctx.is_authenticated = True
            mock_auth_ctx.user_id = "admin-123"
            mock_extract.return_value = mock_auth_ctx

            result = admin_handler.handle("/api/admin/system/metrics", {}, mock_handler)

            body = json.loads(result.body)
            assert body["metrics"]["users"]["total_users"] == 100

    def test_system_metrics_handles_missing_debate_storage(self, admin_handler, mock_handler):
        """Test graceful handling when debate storage is unavailable."""
        with patch("aragora.server.handlers.admin.extract_user_from_request") as mock_extract:
            mock_auth_ctx = Mock()
            mock_auth_ctx.is_authenticated = True
            mock_auth_ctx.user_id = "admin-123"
            mock_extract.return_value = mock_auth_ctx

            result = admin_handler.handle("/api/admin/system/metrics", {}, mock_handler)

            assert result.status_code == 200
            body = json.loads(result.body)
            # Should not have debates key or should have error
            assert "metrics" in body


# ============================================================================
# Revenue Stats Tests
# ============================================================================


class TestGetRevenueStats:
    """Tests for GET /api/admin/revenue."""

    def test_get_revenue_stats_success(self, admin_handler, mock_handler, mock_user_store):
        """Test successful revenue stats retrieval."""
        with patch("aragora.server.handlers.admin.extract_user_from_request") as mock_extract:
            mock_auth_ctx = Mock()
            mock_auth_ctx.is_authenticated = True
            mock_auth_ctx.user_id = "admin-123"
            mock_extract.return_value = mock_auth_ctx

            result = admin_handler.handle("/api/admin/revenue", {}, mock_handler)

            assert result.status_code == 200
            body = json.loads(result.body)
            assert "revenue" in body
            assert "mrr_cents" in body["revenue"]
            assert "mrr_dollars" in body["revenue"]
            assert "arr_dollars" in body["revenue"]

    def test_revenue_stats_calculates_mrr(self, admin_handler, mock_handler, mock_user_store):
        """Test that MRR is calculated from tier distribution."""
        with patch("aragora.server.handlers.admin.extract_user_from_request") as mock_extract:
            mock_auth_ctx = Mock()
            mock_auth_ctx.is_authenticated = True
            mock_auth_ctx.user_id = "admin-123"
            mock_extract.return_value = mock_auth_ctx

            result = admin_handler.handle("/api/admin/revenue", {}, mock_handler)

            body = json.loads(result.body)
            # MRR should be sum of tier prices * counts
            assert body["revenue"]["mrr_cents"] >= 0
            assert body["revenue"]["mrr_dollars"] == body["revenue"]["mrr_cents"] / 100

    def test_revenue_stats_calculates_arr(self, admin_handler, mock_handler, mock_user_store):
        """Test that ARR is calculated as MRR * 12."""
        with patch("aragora.server.handlers.admin.extract_user_from_request") as mock_extract:
            mock_auth_ctx = Mock()
            mock_auth_ctx.is_authenticated = True
            mock_auth_ctx.user_id = "admin-123"
            mock_extract.return_value = mock_auth_ctx

            result = admin_handler.handle("/api/admin/revenue", {}, mock_handler)

            body = json.loads(result.body)
            expected_arr = body["revenue"]["mrr_dollars"] * 12
            assert body["revenue"]["arr_dollars"] == expected_arr


# ============================================================================
# Impersonation Tests
# ============================================================================


class TestImpersonateUser:
    """Tests for POST /api/admin/impersonate/:user_id."""

    def test_impersonate_user_success(self, admin_handler, mock_handler, mock_user_store, mock_target_user):
        """Test successful user impersonation."""
        mock_handler.command = "POST"
        mock_user_store.get_user_by_id = Mock(side_effect=lambda uid: {
            "admin-123": Mock(id="admin-123", role="admin"),
            "target-456": mock_target_user,
        }.get(uid))

        with patch("aragora.server.handlers.admin.extract_user_from_request") as mock_extract:
            mock_auth_ctx = Mock()
            mock_auth_ctx.is_authenticated = True
            mock_auth_ctx.user_id = "admin-123"
            mock_extract.return_value = mock_auth_ctx

            with patch("aragora.server.handlers.admin.create_access_token") as mock_token:
                mock_token.return_value = "impersonation-token-123"

                result = admin_handler.handle(
                    "/api/admin/impersonate/target-456", {}, mock_handler, "POST"
                )

                assert result.status_code == 200
                body = json.loads(result.body)
                assert "token" in body
                assert body["token"] == "impersonation-token-123"
                assert body["expires_in"] == 3600
                assert "warning" in body

    def test_impersonate_user_not_found(self, admin_handler, mock_handler, mock_user_store):
        """Test impersonation of non-existent user."""
        mock_handler.command = "POST"
        mock_user_store.get_user_by_id = Mock(side_effect=lambda uid: {
            "admin-123": Mock(id="admin-123", role="admin"),
        }.get(uid))

        with patch("aragora.server.handlers.admin.extract_user_from_request") as mock_extract:
            mock_auth_ctx = Mock()
            mock_auth_ctx.is_authenticated = True
            mock_auth_ctx.user_id = "admin-123"
            mock_extract.return_value = mock_auth_ctx

            result = admin_handler.handle(
                "/api/admin/impersonate/nonexistent-user", {}, mock_handler, "POST"
            )

            assert result.status_code == 404
            body = json.loads(result.body)
            assert "not found" in body.get("error", "").lower()

    def test_impersonate_invalid_user_id_format(self, admin_handler, mock_handler):
        """Test impersonation with invalid user ID format."""
        mock_handler.command = "POST"

        # Invalid user ID with special characters
        result = admin_handler.handle(
            "/api/admin/impersonate/../../etc/passwd", {}, mock_handler, "POST"
        )

        # Should reject with either 400 (validation) or 401 (auth)
        assert result.status_code in (400, 401)

    def test_impersonate_logs_audit_event(self, admin_handler, mock_handler, mock_user_store, mock_target_user):
        """Test that impersonation is logged for audit."""
        mock_handler.command = "POST"
        mock_user_store.get_user_by_id = Mock(side_effect=lambda uid: {
            "admin-123": Mock(id="admin-123", role="admin"),
            "target-456": mock_target_user,
        }.get(uid))

        with patch("aragora.server.handlers.admin.extract_user_from_request") as mock_extract:
            mock_auth_ctx = Mock()
            mock_auth_ctx.is_authenticated = True
            mock_auth_ctx.user_id = "admin-123"
            mock_extract.return_value = mock_auth_ctx

            with patch("aragora.server.handlers.admin.create_access_token") as mock_token:
                mock_token.return_value = "impersonation-token-123"

                admin_handler.handle(
                    "/api/admin/impersonate/target-456", {}, mock_handler, "POST"
                )

                # Verify audit event was recorded
                mock_user_store.record_audit_event.assert_called_once()


# ============================================================================
# User Activation/Deactivation Tests
# ============================================================================


class TestDeactivateUser:
    """Tests for POST /api/admin/users/:user_id/deactivate."""

    def test_deactivate_user_success(self, admin_handler, mock_handler, mock_user_store, mock_target_user):
        """Test successful user deactivation."""
        mock_handler.command = "POST"
        mock_user_store.get_user_by_id = Mock(side_effect=lambda uid: {
            "admin-123": Mock(id="admin-123", role="admin"),
            "target-456": mock_target_user,
        }.get(uid))

        with patch("aragora.server.handlers.admin.extract_user_from_request") as mock_extract:
            mock_auth_ctx = Mock()
            mock_auth_ctx.is_authenticated = True
            mock_auth_ctx.user_id = "admin-123"
            mock_extract.return_value = mock_auth_ctx

            result = admin_handler.handle(
                "/api/admin/users/target-456/deactivate", {}, mock_handler, "POST"
            )

            assert result.status_code == 200
            body = json.loads(result.body)
            assert body["success"] is True
            assert body["is_active"] is False

            # Verify update was called
            mock_user_store.update_user.assert_called_with("target-456", is_active=False)

    def test_deactivate_nonexistent_user(self, admin_handler, mock_handler, mock_user_store):
        """Test deactivating a non-existent user."""
        mock_handler.command = "POST"
        mock_user_store.get_user_by_id = Mock(side_effect=lambda uid: {
            "admin-123": Mock(id="admin-123", role="admin"),
        }.get(uid))

        with patch("aragora.server.handlers.admin.extract_user_from_request") as mock_extract:
            mock_auth_ctx = Mock()
            mock_auth_ctx.is_authenticated = True
            mock_auth_ctx.user_id = "admin-123"
            mock_extract.return_value = mock_auth_ctx

            result = admin_handler.handle(
                "/api/admin/users/nonexistent/deactivate", {}, mock_handler, "POST"
            )

            assert result.status_code == 404

    def test_cannot_deactivate_self(self, admin_handler, mock_handler, mock_user_store, mock_admin_user):
        """Test that admin cannot deactivate themselves."""
        mock_handler.command = "POST"
        mock_user_store.get_user_by_id = Mock(return_value=mock_admin_user)

        with patch("aragora.server.handlers.admin.extract_user_from_request") as mock_extract:
            mock_auth_ctx = Mock()
            mock_auth_ctx.is_authenticated = True
            mock_auth_ctx.user_id = "admin-123"
            mock_extract.return_value = mock_auth_ctx

            result = admin_handler.handle(
                "/api/admin/users/admin-123/deactivate", {}, mock_handler, "POST"
            )

            assert result.status_code == 400
            body = json.loads(result.body)
            assert "yourself" in body.get("error", "").lower()


class TestActivateUser:
    """Tests for POST /api/admin/users/:user_id/activate."""

    def test_activate_user_success(self, admin_handler, mock_handler, mock_user_store, mock_target_user):
        """Test successful user activation."""
        mock_handler.command = "POST"
        mock_target_user.is_active = False  # User is currently deactivated
        mock_user_store.get_user_by_id = Mock(side_effect=lambda uid: {
            "admin-123": Mock(id="admin-123", role="admin"),
            "target-456": mock_target_user,
        }.get(uid))

        with patch("aragora.server.handlers.admin.extract_user_from_request") as mock_extract:
            mock_auth_ctx = Mock()
            mock_auth_ctx.is_authenticated = True
            mock_auth_ctx.user_id = "admin-123"
            mock_extract.return_value = mock_auth_ctx

            result = admin_handler.handle(
                "/api/admin/users/target-456/activate", {}, mock_handler, "POST"
            )

            assert result.status_code == 200
            body = json.loads(result.body)
            assert body["success"] is True
            assert body["is_active"] is True

            # Verify update was called
            mock_user_store.update_user.assert_called_with("target-456", is_active=True)

    def test_activate_nonexistent_user(self, admin_handler, mock_handler, mock_user_store):
        """Test activating a non-existent user."""
        mock_handler.command = "POST"
        mock_user_store.get_user_by_id = Mock(side_effect=lambda uid: {
            "admin-123": Mock(id="admin-123", role="admin"),
        }.get(uid))

        with patch("aragora.server.handlers.admin.extract_user_from_request") as mock_extract:
            mock_auth_ctx = Mock()
            mock_auth_ctx.is_authenticated = True
            mock_auth_ctx.user_id = "admin-123"
            mock_extract.return_value = mock_auth_ctx

            result = admin_handler.handle(
                "/api/admin/users/nonexistent/activate", {}, mock_handler, "POST"
            )

            assert result.status_code == 404


# ============================================================================
# Route Handling Tests
# ============================================================================


class TestRouteHandling:
    """Tests for route handling and method validation."""

    def test_can_handle_admin_paths(self):
        """Test that handler can handle admin paths."""
        assert AdminHandler.can_handle("/api/admin/organizations")
        assert AdminHandler.can_handle("/api/admin/users")
        assert AdminHandler.can_handle("/api/admin/stats")
        assert AdminHandler.can_handle("/api/admin/impersonate/user-123")

    def test_cannot_handle_non_admin_paths(self):
        """Test that handler rejects non-admin paths."""
        assert not AdminHandler.can_handle("/api/users")
        assert not AdminHandler.can_handle("/api/debates")
        assert not AdminHandler.can_handle("/api/health")

    def test_method_not_allowed(self, admin_handler, mock_handler, mock_user_store):
        """Test 405 response for unsupported methods."""
        mock_handler.command = "DELETE"

        with patch("aragora.server.handlers.admin.extract_user_from_request") as mock_extract:
            mock_auth_ctx = Mock()
            mock_auth_ctx.is_authenticated = True
            mock_auth_ctx.user_id = "admin-123"
            mock_extract.return_value = mock_auth_ctx

            result = admin_handler.handle("/api/admin/organizations", {}, mock_handler, "DELETE")

            assert result.status_code == 405

    def test_routes_list(self):
        """Test that ROUTES contains expected paths."""
        assert "/api/admin/organizations" in AdminHandler.ROUTES
        assert "/api/admin/users" in AdminHandler.ROUTES
        assert "/api/admin/stats" in AdminHandler.ROUTES


# ============================================================================
# Input Validation Tests
# ============================================================================


class TestInputValidation:
    """Tests for input parameter validation.

    Note: Different endpoints have different validation/auth ordering.
    Some validate input first (400), others check auth first (401).
    """

    def test_invalid_user_id_in_impersonate(self, admin_handler, mock_handler):
        """Test rejection of invalid user ID in impersonate path."""
        mock_handler.command = "POST"

        invalid_ids = [
            "../../../etc/passwd",
            "user'; DROP TABLE users;--",
            "<script>alert('xss')</script>",
        ]

        for invalid_id in invalid_ids:
            result = admin_handler.handle(
                f"/api/admin/impersonate/{invalid_id}", {}, mock_handler, "POST"
            )
            # Should reject with either 400 (validation) or 401 (auth)
            assert result.status_code in (400, 401), f"Should reject: {invalid_id}"

    def test_invalid_user_id_in_deactivate(self, admin_handler, mock_handler):
        """Test rejection of invalid user ID in deactivate path."""
        mock_handler.command = "POST"

        result = admin_handler.handle(
            "/api/admin/users/../../../etc/passwd/deactivate", {}, mock_handler, "POST"
        )
        # Should reject with either 400 (validation) or 401 (auth)
        assert result.status_code in (400, 401)

    def test_invalid_user_id_in_activate(self, admin_handler, mock_handler):
        """Test rejection of invalid user ID in activate path."""
        mock_handler.command = "POST"

        result = admin_handler.handle(
            "/api/admin/users/'; DROP TABLE--/activate", {}, mock_handler, "POST"
        )
        # Should reject with either 400 (validation) or 401 (auth)
        assert result.status_code in (400, 401)


__all__ = [
    "TestAdminAuthorization",
    "TestAdminRoles",
    "TestListOrganizations",
    "TestListUsers",
    "TestGetStats",
    "TestGetSystemMetrics",
    "TestGetRevenueStats",
    "TestImpersonateUser",
    "TestDeactivateUser",
    "TestActivateUser",
    "TestRouteHandling",
    "TestInputValidation",
]
