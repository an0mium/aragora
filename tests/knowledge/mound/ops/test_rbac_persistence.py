"""
Tests for RBAC persistence and management.

Phase 7: KM Governance Test Gaps - RBAC persistence tests.

Tests:
- test_role_creation_and_persistence - Creating custom roles
- test_permission_inheritance_admin - Admin permission grants all
- test_workspace_scoped_roles - Workspace-specific roles
- test_role_assignment_expiration - Time-limited assignments
- test_multiple_roles_per_user - Users with multiple roles
- test_role_revocation_cascade - Revocation effects
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Set

import pytest

from aragora.knowledge.mound.ops.governance import (
    BUILTIN_ROLES,
    BuiltinRole,
    Permission,
    RBACManager,
    Role,
    RoleAssignment,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def rbac_manager():
    """Create a fresh RBAC manager."""
    return RBACManager()


# ============================================================================
# Test: Role Creation and Persistence
# ============================================================================


class TestRoleCreationAndPersistence:
    """Test custom role creation."""

    @pytest.mark.asyncio
    async def test_role_creation_and_persistence(self, rbac_manager):
        """Test creating a custom role."""
        role = await rbac_manager.create_role(
            name="Custom Editor",
            permissions={Permission.READ, Permission.UPDATE},
            description="Can read and edit items",
            created_by="admin-1",
        )

        assert role.id is not None
        assert role.name == "Custom Editor"
        assert Permission.READ in role.permissions
        assert Permission.UPDATE in role.permissions
        assert not role.is_builtin

    @pytest.mark.asyncio
    async def test_role_stored_in_manager(self, rbac_manager):
        """Test that created role is stored and retrievable."""
        role = await rbac_manager.create_role(
            name="Test Role",
            permissions={Permission.READ},
        )

        # Role should be in manager's internal storage
        assert role.id in rbac_manager._roles
        assert rbac_manager._roles[role.id].name == "Test Role"

    @pytest.mark.asyncio
    async def test_builtin_roles_exist(self, rbac_manager):
        """Test that builtin roles are initialized."""
        # All builtin roles should exist
        for builtin in BuiltinRole:
            role_id = f"builtin:{builtin.value}"
            assert role_id in rbac_manager._roles

    @pytest.mark.asyncio
    async def test_role_to_dict_serialization(self, rbac_manager):
        """Test role serialization."""
        role = await rbac_manager.create_role(
            name="Serializable Role",
            permissions={Permission.READ, Permission.CREATE},
            description="Test description",
            workspace_id="ws-1",
            created_by="user-1",
        )

        d = role.to_dict()
        assert d["id"] == role.id
        assert d["name"] == "Serializable Role"
        assert "read" in d["permissions"]
        assert "create" in d["permissions"]
        assert d["workspace_id"] == "ws-1"
        assert d["created_by"] == "user-1"


# ============================================================================
# Test: Permission Inheritance - Admin
# ============================================================================


class TestPermissionInheritanceAdmin:
    """Test admin permission grants all permissions."""

    @pytest.mark.asyncio
    async def test_permission_inheritance_admin(self, rbac_manager):
        """Test that admin role has all permissions."""
        admin_role = BUILTIN_ROLES[BuiltinRole.ADMIN]

        # Admin should have all permissions
        for permission in Permission:
            assert admin_role.has_permission(permission)

    @pytest.mark.asyncio
    async def test_admin_user_has_all_permissions(self, rbac_manager):
        """Test that user with admin role has all permissions."""
        user_id = "admin-user"

        # Assign admin role
        await rbac_manager.assign_role(
            user_id=user_id,
            role_id="builtin:admin",
        )

        # Check all permissions
        for permission in Permission:
            has_perm = await rbac_manager.check_permission(user_id, permission)
            assert has_perm, f"Admin should have {permission}"

    @pytest.mark.asyncio
    async def test_admin_permissions_returned_for_user(self, rbac_manager):
        """Test getting all permissions for admin user."""
        user_id = "admin-user"

        await rbac_manager.assign_role(
            user_id=user_id,
            role_id="builtin:admin",
        )

        permissions = await rbac_manager.get_user_permissions(user_id)

        # Should have all permissions
        assert permissions == set(Permission)


# ============================================================================
# Test: Workspace-Scoped Roles
# ============================================================================


class TestWorkspaceScopedRoles:
    """Test workspace-specific role scoping."""

    @pytest.mark.asyncio
    async def test_workspace_scoped_roles(self, rbac_manager):
        """Test that workspace-scoped roles only apply to that workspace."""
        user_id = "user-1"

        # Create workspace-scoped role
        role = await rbac_manager.create_role(
            name="WS1 Editor",
            permissions={Permission.READ, Permission.UPDATE},
            workspace_id="ws-1",
        )

        # Assign to user
        await rbac_manager.assign_role(
            user_id=user_id,
            role_id=role.id,
            workspace_id="ws-1",
        )

        # Should have permission in ws-1
        has_perm_ws1 = await rbac_manager.check_permission(
            user_id, Permission.READ, workspace_id="ws-1"
        )
        assert has_perm_ws1

        # Should not have permission in ws-2 (no role there)
        has_perm_ws2 = await rbac_manager.check_permission(
            user_id, Permission.READ, workspace_id="ws-2"
        )
        assert not has_perm_ws2

    @pytest.mark.asyncio
    async def test_global_role_applies_to_all_workspaces(self, rbac_manager):
        """Test that global roles apply to all workspaces."""
        user_id = "user-1"

        # Assign global viewer role
        await rbac_manager.assign_role(
            user_id=user_id,
            role_id="builtin:viewer",
            workspace_id=None,  # Global
        )

        # Should have read permission in any workspace
        for ws_id in ["ws-1", "ws-2", "ws-3"]:
            has_perm = await rbac_manager.check_permission(
                user_id, Permission.READ, workspace_id=ws_id
            )
            assert has_perm, f"Global viewer should have read in {ws_id}"


# ============================================================================
# Test: Role Assignment Expiration
# ============================================================================


class TestRoleAssignmentExpiration:
    """Test time-limited role assignments."""

    @pytest.mark.asyncio
    async def test_role_assignment_expiration(self, rbac_manager):
        """Test that expired assignments are detected."""
        user_id = "temp-user"

        # Create assignment with past expiration
        past_expiration = datetime.now() - timedelta(hours=1)
        assignment = await rbac_manager.assign_role(
            user_id=user_id,
            role_id="builtin:editor",
            expires_at=past_expiration,
        )

        assert assignment.is_expired()

    @pytest.mark.asyncio
    async def test_role_assignment_not_expired(self, rbac_manager):
        """Test that future-expiring assignments are valid."""
        user_id = "temp-user"

        # Create assignment with future expiration
        future_expiration = datetime.now() + timedelta(hours=24)
        assignment = await rbac_manager.assign_role(
            user_id=user_id,
            role_id="builtin:editor",
            expires_at=future_expiration,
        )

        assert not assignment.is_expired()

    @pytest.mark.asyncio
    async def test_assignment_with_no_expiration(self, rbac_manager):
        """Test that assignments without expiration never expire."""
        user_id = "perm-user"

        assignment = await rbac_manager.assign_role(
            user_id=user_id,
            role_id="builtin:viewer",
            expires_at=None,  # No expiration
        )

        assert not assignment.is_expired()


# ============================================================================
# Test: Multiple Roles Per User
# ============================================================================


class TestMultipleRolesPerUser:
    """Test users with multiple roles."""

    @pytest.mark.asyncio
    async def test_multiple_roles_per_user(self, rbac_manager):
        """Test that users can have multiple roles."""
        user_id = "multi-role-user"

        # Assign multiple roles
        await rbac_manager.assign_role(user_id=user_id, role_id="builtin:viewer")
        await rbac_manager.assign_role(user_id=user_id, role_id="builtin:contributor")

        # User should have permissions from both roles
        roles = await rbac_manager.get_user_roles(user_id)
        assert len(roles) == 2

        permissions = await rbac_manager.get_user_permissions(user_id)
        # Viewer has READ, Contributor has READ + CREATE
        assert Permission.READ in permissions
        assert Permission.CREATE in permissions

    @pytest.mark.asyncio
    async def test_permission_union_from_multiple_roles(self, rbac_manager):
        """Test that permissions are unioned from multiple roles."""
        user_id = "union-user"

        # Create two custom roles with different permissions
        role1 = await rbac_manager.create_role(
            name="Role A",
            permissions={Permission.READ, Permission.UPDATE},
        )
        role2 = await rbac_manager.create_role(
            name="Role B",
            permissions={Permission.CREATE, Permission.DELETE},
        )

        # Assign both
        await rbac_manager.assign_role(user_id=user_id, role_id=role1.id)
        await rbac_manager.assign_role(user_id=user_id, role_id=role2.id)

        permissions = await rbac_manager.get_user_permissions(user_id)

        # Should have union of all permissions
        assert Permission.READ in permissions
        assert Permission.UPDATE in permissions
        assert Permission.CREATE in permissions
        assert Permission.DELETE in permissions


# ============================================================================
# Test: Role Revocation Cascade
# ============================================================================


class TestRoleRevocationCascade:
    """Test role revocation effects."""

    @pytest.mark.asyncio
    async def test_role_revocation_cascade(self, rbac_manager):
        """Test that revoking a role removes user's permissions from that role."""
        user_id = "revoke-user"

        # Assign editor role (has READ, CREATE, UPDATE)
        await rbac_manager.assign_role(user_id=user_id, role_id="builtin:editor")

        # Verify has permissions
        assert await rbac_manager.check_permission(user_id, Permission.UPDATE)

        # Revoke the role
        revoked = await rbac_manager.revoke_role(user_id, "builtin:editor")
        assert revoked

        # Should no longer have update permission
        assert not await rbac_manager.check_permission(user_id, Permission.UPDATE)

    @pytest.mark.asyncio
    async def test_revoke_one_role_keeps_other(self, rbac_manager):
        """Test that revoking one role doesn't affect others."""
        user_id = "multi-revoke-user"

        # Assign two roles
        await rbac_manager.assign_role(user_id=user_id, role_id="builtin:viewer")
        await rbac_manager.assign_role(user_id=user_id, role_id="builtin:contributor")

        # Revoke contributor
        await rbac_manager.revoke_role(user_id, "builtin:contributor")

        # Should still have viewer permissions (READ)
        assert await rbac_manager.check_permission(user_id, Permission.READ)

        # Should not have create (was from contributor)
        # Viewer doesn't have create
        assert not await rbac_manager.check_permission(user_id, Permission.CREATE)

    @pytest.mark.asyncio
    async def test_revoke_nonexistent_returns_false(self, rbac_manager):
        """Test that revoking a non-assigned role returns False."""
        result = await rbac_manager.revoke_role("nonexistent-user", "builtin:admin")
        assert result is False


# ============================================================================
# Test: Concurrent Role Operations
# ============================================================================


class TestConcurrentRoleOperations:
    """Test thread safety of role operations."""

    @pytest.mark.asyncio
    async def test_concurrent_role_assignments(self, rbac_manager):
        """Test concurrent role assignments don't corrupt state."""
        num_users = 20

        async def assign_role_to_user(user_idx: int):
            user_id = f"user-{user_idx}"
            await rbac_manager.assign_role(
                user_id=user_id,
                role_id="builtin:viewer",
            )
            return user_id

        # Assign roles concurrently
        tasks = [assign_role_to_user(i) for i in range(num_users)]
        user_ids = await asyncio.gather(*tasks)

        # All users should have been assigned
        assert len(user_ids) == num_users

        # Verify all have the role
        for user_id in user_ids:
            has_read = await rbac_manager.check_permission(user_id, Permission.READ)
            assert has_read


__all__ = [
    "TestRoleCreationAndPersistence",
    "TestPermissionInheritanceAdmin",
    "TestWorkspaceScopedRoles",
    "TestRoleAssignmentExpiration",
    "TestMultipleRolesPerUser",
    "TestRoleRevocationCascade",
    "TestConcurrentRoleOperations",
]
