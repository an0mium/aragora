"""
Tests for OAuth-to-RBAC permission auto-provisioning.

Verifies that:
1. New OAuth users get default 'member' permissions resolved from RBAC
2. Existing users keep their permissions
3. AuthContext.to_authorization_context() resolves permissions from roles
4. The auto-provisioning is idempotent
"""

from __future__ import annotations

import pytest

from aragora.server.middleware.auth import AuthContext


class TestAuthContextToAuthorizationContext:
    """Test AuthContext -> AuthorizationContext conversion with permission resolution."""

    def test_member_role_gets_default_permissions(self):
        """New OAuth user with 'member' role gets resolved RBAC permissions."""
        ctx = AuthContext(
            authenticated=True,
            user_id="oauth-user-123",
            roles={"member"},
            permissions=None,  # Not yet resolved
        )

        auth_ctx = ctx.to_authorization_context()

        assert auth_ctx.user_id == "oauth-user-123"
        assert "member" in auth_ctx.roles
        # Member should have basic debate permissions
        assert auth_ctx.has_permission("debates.create")
        assert auth_ctx.has_permission("debates.read")
        assert auth_ctx.has_permission("debates.run")

    def test_admin_role_gets_admin_permissions(self):
        """Admin users get full admin permissions."""
        ctx = AuthContext(
            authenticated=True,
            user_id="admin-user",
            roles={"admin"},
            permissions=None,
        )

        auth_ctx = ctx.to_authorization_context()

        assert auth_ctx.has_permission("debates.create")
        assert auth_ctx.has_permission("debates.delete")
        assert auth_ctx.has_permission("agents.create")

    def test_explicit_permissions_preserved(self):
        """If permissions are already set, they are not overwritten."""
        custom_perms = {"custom.read", "custom.write"}
        ctx = AuthContext(
            authenticated=True,
            user_id="custom-user",
            roles={"member"},
            permissions=custom_perms,
        )

        auth_ctx = ctx.to_authorization_context()

        # Should use the explicit permissions, not resolve from roles
        assert auth_ctx.permissions == custom_perms

    def test_no_roles_no_permissions(self):
        """Unauthenticated context with no roles gets empty permissions."""
        ctx = AuthContext(
            authenticated=False,
            user_id=None,
            roles=None,
            permissions=None,
        )

        auth_ctx = ctx.to_authorization_context()

        assert auth_ctx.user_id == "anonymous"
        assert auth_ctx.roles == set()
        assert auth_ctx.permissions == set()

    def test_viewer_role_gets_read_only(self):
        """Viewer role should only get read permissions."""
        ctx = AuthContext(
            authenticated=True,
            user_id="viewer-user",
            roles={"viewer"},
            permissions=None,
        )

        auth_ctx = ctx.to_authorization_context()

        assert auth_ctx.has_permission("debates.read")
        assert auth_ctx.has_permission("agents.read")
        # Viewer should NOT have create/delete
        assert not auth_ctx.has_permission("debates.create")
        assert not auth_ctx.has_permission("debates.delete")

    def test_idempotent_permission_resolution(self):
        """Calling to_authorization_context() multiple times gives consistent results."""
        ctx = AuthContext(
            authenticated=True,
            user_id="user-1",
            roles={"member"},
            permissions=None,
        )

        auth_ctx_1 = ctx.to_authorization_context()
        auth_ctx_2 = ctx.to_authorization_context()

        assert auth_ctx_1.permissions == auth_ctx_2.permissions
        assert auth_ctx_1.roles == auth_ctx_2.roles

    def test_multiple_roles_union_permissions(self):
        """Multiple roles should have union of all permissions."""
        ctx = AuthContext(
            authenticated=True,
            user_id="multi-role-user",
            roles={"member", "compliance_officer"},
            permissions=None,
        )

        auth_ctx = ctx.to_authorization_context()

        # Should have member permissions
        assert auth_ctx.has_permission("debates.create")
        # Should also have compliance officer permissions
        assert auth_ctx.has_permission("compliance_policy.read")


class TestOAuthUserCreationAudit:
    """Test that OAuth user creation logs audit info."""

    def test_new_user_gets_member_role(self):
        """Verify User model defaults to 'member' role."""
        from aragora.billing.models import User

        user = User(email="test@example.com")
        assert user.role == "member"

    def test_member_role_includes_debate_permissions(self):
        """Verify RBAC 'member' role includes required debate permissions."""
        from aragora.rbac.defaults import get_role_permissions

        perms = get_role_permissions("member", include_inherited=True)

        # Required minimum permissions for usability
        assert "debates.create" in perms
        assert "debates.read" in perms
        assert "debates.run" in perms

    def test_member_role_includes_inherited_viewer_permissions(self):
        """Member inherits from viewer via role hierarchy."""
        from aragora.rbac.defaults import get_role_permissions

        member_perms = get_role_permissions("member", include_inherited=True)
        viewer_perms = get_role_permissions("viewer", include_inherited=True)

        # Member should include all viewer permissions
        assert viewer_perms.issubset(member_perms)


class TestHandlerPermissionBridge:
    """Test that the handler-level permission check works with JWT roles."""

    def test_permission_matrix_has_member_debate_access(self):
        """PERMISSION_MATRIX should grant member role debate access."""
        from aragora.server.handlers.utils.decorators import has_permission

        assert has_permission("member", "debates:read")
        assert has_permission("member", "debates:create")
        assert has_permission("member", "debates:write")

    def test_permission_matrix_denies_member_admin_access(self):
        """PERMISSION_MATRIX should deny member role admin access."""
        from aragora.server.handlers.utils.decorators import has_permission

        assert not has_permission("member", "admin:*")
        assert not has_permission("member", "admin:system")

    def test_empty_role_denied(self):
        """Empty or None role should be denied."""
        from aragora.server.handlers.utils.decorators import has_permission

        assert not has_permission("", "debates:read")
        assert not has_permission(None, "debates:read")


class TestOptionalAuthPermissionPopulation:
    """Test that optional_auth populates permissions when authenticated."""

    def test_auth_context_has_permission_method(self):
        """AuthContext.has_permission works with populated permissions."""
        ctx = AuthContext(
            authenticated=True,
            user_id="test",
            permissions={"debates.read", "debates.create"},
        )

        assert ctx.has_permission("debates.read")
        assert ctx.has_permission("debates.create")
        assert not ctx.has_permission("admin.system")

    def test_auth_context_wildcard_permission(self):
        """Wildcard permission grants all access."""
        ctx = AuthContext(
            authenticated=True,
            user_id="test",
            permissions={"*"},
        )

        assert ctx.has_permission("anything")
        assert ctx.has_permission("debates.read")
