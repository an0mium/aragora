"""
Tests for RBAC (Role-Based Access Control) system.

Tests the permission matrix, has_permission function, and require_permission decorator.
"""

import pytest
from unittest.mock import MagicMock, patch

from aragora.server.handlers.base import (
    PERMISSION_MATRIX,
    has_permission,
    require_permission,
    error_response,
)


class TestPermissionMatrix:
    """Tests for the permission matrix structure."""

    def test_permission_matrix_has_all_categories(self):
        """Verify all expected permission categories exist."""
        categories = set()
        for perm in PERMISSION_MATRIX.keys():
            category = perm.split(":")[0]
            categories.add(category)

        expected = {"debates", "agents", "org", "plugins", "admin", "apikeys"}
        assert expected.issubset(categories), f"Missing categories: {expected - categories}"

    def test_all_permissions_have_valid_roles(self):
        """Verify all permissions map to valid roles."""
        valid_roles = {"member", "admin", "owner"}
        for perm, roles in PERMISSION_MATRIX.items():
            assert isinstance(roles, list), f"Permission {perm} roles should be a list"
            for role in roles:
                assert role in valid_roles, f"Permission {perm} has invalid role: {role}"

    def test_owner_has_most_permissions(self):
        """Owner should have the most permissions."""
        owner_perms = sum(1 for roles in PERMISSION_MATRIX.values() if "owner" in roles)
        admin_perms = sum(1 for roles in PERMISSION_MATRIX.values() if "admin" in roles)
        member_perms = sum(1 for roles in PERMISSION_MATRIX.values() if "member" in roles)

        assert owner_perms >= admin_perms >= member_perms

    def test_specific_owner_only_permissions(self):
        """Certain permissions should be owner-only."""
        owner_only = ["org:billing", "org:delete", "admin:system", "admin:users"]
        for perm in owner_only:
            roles = PERMISSION_MATRIX.get(perm, [])
            assert roles == ["owner"], f"{perm} should be owner-only, got {roles}"


class TestHasPermission:
    """Tests for the has_permission function."""

    def test_member_can_read_debates(self):
        """Members should be able to read debates."""
        assert has_permission("member", "debates:read") is True

    def test_member_can_create_debates(self):
        """Members should be able to create debates."""
        assert has_permission("member", "debates:create") is True

    def test_member_cannot_delete_debates(self):
        """Members should not be able to delete debates."""
        assert has_permission("member", "debates:delete") is False

    def test_admin_can_delete_debates(self):
        """Admins should be able to delete debates."""
        assert has_permission("admin", "debates:delete") is True

    def test_owner_can_delete_debates(self):
        """Owners should be able to delete debates."""
        assert has_permission("owner", "debates:delete") is True

    def test_owner_can_access_billing(self):
        """Only owners can access billing."""
        assert has_permission("owner", "org:billing") is True

    def test_admin_cannot_access_billing(self):
        """Admins should not be able to access billing."""
        assert has_permission("admin", "org:billing") is False

    def test_member_cannot_access_billing(self):
        """Members should not be able to access billing."""
        assert has_permission("member", "org:billing") is False

    def test_empty_role_returns_false(self):
        """Empty role should return False."""
        assert has_permission("", "debates:read") is False
        assert has_permission(None, "debates:read") is False

    def test_empty_permission_returns_false(self):
        """Empty permission should return False."""
        assert has_permission("member", "") is False
        assert has_permission("member", None) is False

    def test_unknown_permission_returns_false(self):
        """Unknown permissions should return False."""
        assert has_permission("owner", "unknown:permission") is False

    def test_unknown_role_returns_false(self):
        """Unknown roles should return False."""
        assert has_permission("superuser", "debates:read") is False


class TestRequirePermissionDecorator:
    """Tests for the require_permission decorator."""

    def test_decorator_returns_401_when_no_handler(self):
        """Decorator should return 401 when no handler is provided."""
        @require_permission("debates:read")
        def test_func():
            return "success"

        result = test_func()
        # HandlerResult is a dataclass with body, status_code, headers
        assert result.status_code == 401

    def test_decorator_returns_401_when_not_authenticated(self):
        """Decorator should return 401 when user is not authenticated."""
        mock_handler = MagicMock()
        mock_handler.headers = {"Authorization": ""}
        mock_handler.user_store = None

        @require_permission("debates:read")
        def test_func(handler):
            return "success"

        with patch(
            "aragora.billing.jwt_auth.extract_user_from_request"
        ) as mock_extract:
            mock_ctx = MagicMock()
            mock_ctx.is_authenticated = False
            mock_ctx.error_reason = None
            mock_extract.return_value = mock_ctx

            result = test_func(handler=mock_handler)
            assert result.status_code == 401

    def test_decorator_returns_403_when_not_authorized(self):
        """Decorator should return 403 when user lacks permission."""
        mock_handler = MagicMock()
        mock_handler.headers = {"Authorization": "Bearer test_token"}
        mock_handler.user_store = None

        @require_permission("org:billing")
        def test_func(handler, user=None):
            return "success"

        with patch(
            "aragora.billing.jwt_auth.extract_user_from_request"
        ) as mock_extract:
            mock_ctx = MagicMock()
            mock_ctx.is_authenticated = True
            mock_ctx.user_id = "user123"
            mock_ctx.role = "member"  # Members can't access billing
            mock_extract.return_value = mock_ctx

            result = test_func(handler=mock_handler)
            assert result.status_code == 403
            assert b"org:billing" in result.body

    def test_decorator_allows_authorized_user(self):
        """Decorator should allow access when user has permission."""
        mock_handler = MagicMock()
        mock_handler.headers = {"Authorization": "Bearer test_token"}
        mock_handler.user_store = None

        @require_permission("debates:read")
        def test_func(handler, user=None):
            return ("success", 200)

        with patch(
            "aragora.billing.jwt_auth.extract_user_from_request"
        ) as mock_extract:
            mock_ctx = MagicMock()
            mock_ctx.is_authenticated = True
            mock_ctx.user_id = "user123"
            mock_ctx.role = "member"
            mock_extract.return_value = mock_ctx

            result = test_func(handler=mock_handler)
            assert result == ("success", 200)

    def test_decorator_injects_user_context(self):
        """Decorator should inject user context into kwargs."""
        mock_handler = MagicMock()
        mock_handler.headers = {"Authorization": "Bearer test_token"}
        mock_handler.user_store = None

        received_user = None

        @require_permission("debates:read")
        def test_func(handler, user=None):
            nonlocal received_user
            received_user = user
            return ("success", 200)

        with patch(
            "aragora.billing.jwt_auth.extract_user_from_request"
        ) as mock_extract:
            mock_ctx = MagicMock()
            mock_ctx.is_authenticated = True
            mock_ctx.user_id = "user123"
            mock_ctx.email = "test@example.com"
            mock_ctx.role = "admin"
            mock_extract.return_value = mock_ctx

            test_func(handler=mock_handler)
            assert received_user is mock_ctx

    def test_decorator_owner_has_all_access(self):
        """Owner role should have access to owner-only permissions."""
        mock_handler = MagicMock()
        mock_handler.headers = {"Authorization": "Bearer test_token"}
        mock_handler.user_store = None

        @require_permission("admin:system")
        def test_func(handler, user=None):
            return ("success", 200)

        with patch(
            "aragora.billing.jwt_auth.extract_user_from_request"
        ) as mock_extract:
            mock_ctx = MagicMock()
            mock_ctx.is_authenticated = True
            mock_ctx.user_id = "owner123"
            mock_ctx.role = "owner"
            mock_extract.return_value = mock_ctx

            result = test_func(handler=mock_handler)
            assert result == ("success", 200)


class TestRoleHierarchy:
    """Tests for role hierarchy consistency."""

    def test_owner_has_all_admin_permissions(self):
        """Owner should have all permissions that admin has."""
        for perm, roles in PERMISSION_MATRIX.items():
            if "admin" in roles:
                assert "owner" in roles, f"Owner missing permission that admin has: {perm}"

    def test_admin_has_all_member_permissions(self):
        """Admin should have all permissions that member has."""
        for perm, roles in PERMISSION_MATRIX.items():
            if "member" in roles:
                assert "admin" in roles, f"Admin missing permission that member has: {perm}"


class TestPermissionCategories:
    """Tests for specific permission categories."""

    def test_debate_permissions(self):
        """Test debate permission configuration."""
        assert has_permission("member", "debates:create")
        assert has_permission("member", "debates:read")
        assert not has_permission("member", "debates:update")
        assert not has_permission("member", "debates:delete")
        assert has_permission("admin", "debates:delete")

    def test_org_permissions(self):
        """Test organization permission configuration."""
        assert has_permission("member", "org:read")
        assert not has_permission("member", "org:settings")
        assert has_permission("admin", "org:settings")
        assert not has_permission("admin", "org:billing")
        assert has_permission("owner", "org:billing")

    def test_plugin_permissions(self):
        """Test plugin permission configuration."""
        assert has_permission("member", "plugins:read")
        assert has_permission("member", "plugins:run")
        assert not has_permission("member", "plugins:install")
        assert has_permission("admin", "plugins:install")
