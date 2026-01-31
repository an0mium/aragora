"""
Comprehensive tests for Knowledge Mound Governance module.

Tests the governance features including:
- Role-Based Access Control (RBAC)
- Audit trail logging
- Policy enforcement
- Permission evaluation
- Access decisions
- GovernanceMixin integration

Test Categories:
1. Permission enum tests
2. Role dataclass tests
3. RoleAssignment tests
4. RBACManager policy evaluation tests
5. AuditAction/AuditEntry tests
6. AuditTrail tests
7. GovernanceMixin integration tests
8. Edge cases and error handling
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from datetime import datetime, timedelta
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.knowledge.mound.ops.governance import (
    BUILTIN_ROLES,
    AuditAction,
    AuditEntry,
    AuditTrail,
    BuiltinRole,
    GovernanceMixin,
    Permission,
    RBACManager,
    Role,
    RoleAssignment,
    get_audit_trail,
    get_rbac_manager,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def rbac_manager():
    """Create a fresh RBAC manager for testing."""
    return RBACManager()


@pytest.fixture
def audit_trail():
    """Create an audit trail without persistence for testing."""
    return AuditTrail(max_entries=1000, enable_persistence=False)


@pytest.fixture
def temp_db_path():
    """Create a temporary database path for persistence tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield os.path.join(tmpdir, "test_governance.db")


@pytest.fixture
def governance_mixin():
    """Create a class with GovernanceMixin for testing."""

    class TestGovernanceClass(GovernanceMixin):
        pass

    return TestGovernanceClass()


# ============================================================================
# Test: Permission Enum
# ============================================================================


class TestPermissionEnum:
    """Test Permission enum values and coverage."""

    def test_all_permissions_defined(self):
        """Test that all expected permissions are defined."""
        expected_permissions = {
            "read",
            "create",
            "update",
            "delete",
            "manage_workspace",
            "share",
            "export",
            "manage_users",
            "manage_roles",
            "view_audit",
            "manage_policies",
            "admin",
        }
        actual_permissions = {p.value for p in Permission}
        assert actual_permissions == expected_permissions

    def test_permission_string_enum(self):
        """Test that Permission is a string enum."""
        assert isinstance(Permission.READ, str)
        assert Permission.READ == "read"
        assert Permission.ADMIN.value == "admin"

    def test_permission_categories(self):
        """Test permission categorization for documentation."""
        item_permissions = {
            Permission.READ,
            Permission.CREATE,
            Permission.UPDATE,
            Permission.DELETE,
        }
        workspace_permissions = {Permission.MANAGE_WORKSPACE, Permission.SHARE, Permission.EXPORT}
        admin_permissions = {
            Permission.MANAGE_USERS,
            Permission.MANAGE_ROLES,
            Permission.VIEW_AUDIT,
            Permission.MANAGE_POLICIES,
        }

        # Verify categories are disjoint
        assert len(item_permissions & workspace_permissions) == 0
        assert len(item_permissions & admin_permissions) == 0
        assert len(workspace_permissions & admin_permissions) == 0


# ============================================================================
# Test: Role Dataclass
# ============================================================================


class TestRoleDataclass:
    """Test Role dataclass functionality."""

    def test_role_creation_with_defaults(self):
        """Test creating a role with default values."""
        role = Role(
            id="test-role",
            name="Test Role",
            description="A test role",
            permissions={Permission.READ},
        )

        assert role.id == "test-role"
        assert role.name == "Test Role"
        assert role.workspace_id is None
        assert role.is_builtin is False
        assert role.created_by is None
        assert isinstance(role.created_at, datetime)

    def test_role_has_permission_direct(self):
        """Test direct permission check."""
        role = Role(
            id="r1",
            name="Reader",
            description="",
            permissions={Permission.READ, Permission.CREATE},
        )

        assert role.has_permission(Permission.READ)
        assert role.has_permission(Permission.CREATE)
        assert not role.has_permission(Permission.DELETE)

    def test_role_has_permission_admin_grants_all(self):
        """Test that admin permission grants all other permissions."""
        admin_role = Role(
            id="admin",
            name="Admin",
            description="",
            permissions={Permission.ADMIN},
        )

        # Admin should have all permissions
        for perm in Permission:
            assert admin_role.has_permission(perm)

    def test_role_to_dict_serialization(self):
        """Test role serialization to dictionary."""
        now = datetime.now()
        role = Role(
            id="r1",
            name="Test",
            description="Test role",
            permissions={Permission.READ, Permission.UPDATE},
            workspace_id="ws-1",
            created_at=now,
            created_by="admin",
            is_builtin=False,
        )

        d = role.to_dict()
        assert d["id"] == "r1"
        assert d["name"] == "Test"
        assert d["description"] == "Test role"
        assert set(d["permissions"]) == {"read", "update"}
        assert d["workspace_id"] == "ws-1"
        assert d["created_by"] == "admin"
        assert d["is_builtin"] is False

    def test_builtin_roles_configuration(self):
        """Test builtin roles are correctly configured."""
        # Viewer should only have READ
        viewer = BUILTIN_ROLES[BuiltinRole.VIEWER]
        assert viewer.permissions == {Permission.READ}
        assert viewer.is_builtin

        # Contributor should have READ + CREATE
        contributor = BUILTIN_ROLES[BuiltinRole.CONTRIBUTOR]
        assert Permission.READ in contributor.permissions
        assert Permission.CREATE in contributor.permissions

        # Manager should have sharing permissions
        manager = BUILTIN_ROLES[BuiltinRole.MANAGER]
        assert Permission.SHARE in manager.permissions
        assert Permission.MANAGE_WORKSPACE in manager.permissions


# ============================================================================
# Test: RoleAssignment Dataclass
# ============================================================================


class TestRoleAssignmentDataclass:
    """Test RoleAssignment dataclass functionality."""

    def test_assignment_creation_defaults(self):
        """Test creating an assignment with defaults."""
        assignment = RoleAssignment(
            id="a1",
            user_id="user-1",
            role_id="role-1",
        )

        assert assignment.id == "a1"
        assert assignment.user_id == "user-1"
        assert assignment.workspace_id is None
        assert assignment.expires_at is None
        assert isinstance(assignment.assigned_at, datetime)

    def test_assignment_not_expired_without_expiry(self):
        """Test that assignments without expiry never expire."""
        assignment = RoleAssignment(
            id="a1",
            user_id="u1",
            role_id="r1",
            expires_at=None,
        )
        assert not assignment.is_expired()

    def test_assignment_expired_past_date(self):
        """Test that past expiry dates are detected."""
        past = datetime.now() - timedelta(hours=1)
        assignment = RoleAssignment(
            id="a1",
            user_id="u1",
            role_id="r1",
            expires_at=past,
        )
        assert assignment.is_expired()

    def test_assignment_not_expired_future_date(self):
        """Test that future expiry dates are valid."""
        future = datetime.now() + timedelta(days=30)
        assignment = RoleAssignment(
            id="a1",
            user_id="u1",
            role_id="r1",
            expires_at=future,
        )
        assert not assignment.is_expired()

    def test_assignment_to_dict_serialization(self):
        """Test assignment serialization."""
        now = datetime.now()
        expires = now + timedelta(days=7)
        assignment = RoleAssignment(
            id="a1",
            user_id="user-1",
            role_id="role-1",
            workspace_id="ws-1",
            assigned_at=now,
            assigned_by="admin",
            expires_at=expires,
        )

        d = assignment.to_dict()
        assert d["id"] == "a1"
        assert d["user_id"] == "user-1"
        assert d["role_id"] == "role-1"
        assert d["workspace_id"] == "ws-1"
        assert d["assigned_by"] == "admin"
        assert d["expires_at"] is not None


# ============================================================================
# Test: RBACManager Policy Evaluation
# ============================================================================


class TestRBACManagerPolicyEvaluation:
    """Test RBACManager permission and policy evaluation."""

    @pytest.mark.asyncio
    async def test_check_permission_no_roles(self, rbac_manager):
        """Test permission check for user with no roles."""
        # User with no roles should have no permissions
        has_perm = await rbac_manager.check_permission("unknown-user", Permission.READ)
        assert not has_perm

    @pytest.mark.asyncio
    async def test_check_permission_with_role(self, rbac_manager):
        """Test permission check for user with assigned role."""
        await rbac_manager.assign_role(user_id="user-1", role_id="builtin:viewer")

        # Should have READ but not CREATE
        assert await rbac_manager.check_permission("user-1", Permission.READ)
        assert not await rbac_manager.check_permission("user-1", Permission.CREATE)

    @pytest.mark.asyncio
    async def test_check_permission_workspace_scoping(self, rbac_manager):
        """Test permission check respects workspace scoping."""
        # Create workspace-scoped role
        role = await rbac_manager.create_role(
            name="WS Editor",
            permissions={Permission.READ, Permission.UPDATE},
            workspace_id="ws-1",
        )

        await rbac_manager.assign_role(
            user_id="user-1",
            role_id=role.id,
            workspace_id="ws-1",
        )

        # Should have permission in ws-1
        assert await rbac_manager.check_permission("user-1", Permission.READ, workspace_id="ws-1")

        # Should NOT have permission in ws-2 (different workspace)
        assert not await rbac_manager.check_permission(
            "user-1", Permission.READ, workspace_id="ws-2"
        )

    @pytest.mark.asyncio
    async def test_get_user_permissions_union(self, rbac_manager):
        """Test getting all permissions for user with multiple roles."""
        # Assign multiple roles
        await rbac_manager.assign_role(user_id="user-1", role_id="builtin:viewer")
        await rbac_manager.assign_role(user_id="user-1", role_id="builtin:contributor")

        permissions = await rbac_manager.get_user_permissions("user-1")

        # Should have union of permissions
        assert Permission.READ in permissions
        assert Permission.CREATE in permissions

    @pytest.mark.asyncio
    async def test_get_user_permissions_admin_returns_all(self, rbac_manager):
        """Test that admin role returns all permissions."""
        await rbac_manager.assign_role(user_id="admin-user", role_id="builtin:admin")

        permissions = await rbac_manager.get_user_permissions("admin-user")

        # Should have all permissions
        assert permissions == set(Permission)

    @pytest.mark.asyncio
    async def test_assign_role_invalid_role_raises(self, rbac_manager):
        """Test assigning non-existent role raises ValueError."""
        with pytest.raises(ValueError, match="Role not found"):
            await rbac_manager.assign_role(user_id="user-1", role_id="nonexistent-role")

    @pytest.mark.asyncio
    async def test_revoke_role_success(self, rbac_manager):
        """Test successful role revocation."""
        await rbac_manager.assign_role(user_id="user-1", role_id="builtin:editor")

        # Verify has permission
        assert await rbac_manager.check_permission("user-1", Permission.UPDATE)

        # Revoke
        result = await rbac_manager.revoke_role("user-1", "builtin:editor")
        assert result is True

        # Should no longer have permission
        assert not await rbac_manager.check_permission("user-1", Permission.UPDATE)

    @pytest.mark.asyncio
    async def test_revoke_role_not_found(self, rbac_manager):
        """Test revoking non-assigned role returns False."""
        result = await rbac_manager.revoke_role("user-1", "builtin:admin")
        assert result is False

    @pytest.mark.asyncio
    async def test_get_user_roles_empty(self, rbac_manager):
        """Test getting roles for user with no assignments."""
        roles = await rbac_manager.get_user_roles("unknown-user")
        assert roles == []

    @pytest.mark.asyncio
    async def test_get_user_roles_multiple(self, rbac_manager):
        """Test getting multiple roles for a user."""
        await rbac_manager.assign_role(user_id="user-1", role_id="builtin:viewer")
        await rbac_manager.assign_role(user_id="user-1", role_id="builtin:contributor")

        roles = await rbac_manager.get_user_roles("user-1")
        role_ids = {r.id for r in roles}

        assert "builtin:viewer" in role_ids
        assert "builtin:contributor" in role_ids


# ============================================================================
# Test: AuditAction and AuditEntry
# ============================================================================


class TestAuditActionAndEntry:
    """Test AuditAction enum and AuditEntry dataclass."""

    def test_audit_action_categories(self):
        """Test audit action categorization."""
        item_actions = {
            AuditAction.ITEM_CREATE,
            AuditAction.ITEM_READ,
            AuditAction.ITEM_UPDATE,
            AuditAction.ITEM_DELETE,
        }
        sharing_actions = {AuditAction.SHARE_GRANT, AuditAction.SHARE_REVOKE}
        admin_actions = {AuditAction.ROLE_CREATE, AuditAction.ROLE_ASSIGN, AuditAction.ROLE_REVOKE}
        policy_actions = {
            AuditAction.POLICY_CREATE,
            AuditAction.POLICY_UPDATE,
            AuditAction.POLICY_DELETE,
        }

        # Verify all categories exist
        assert len(item_actions) == 4
        assert len(sharing_actions) == 2
        assert len(admin_actions) == 3
        assert len(policy_actions) == 3

    def test_audit_entry_creation(self):
        """Test creating an audit entry."""
        entry = AuditEntry(
            id="entry-1",
            action=AuditAction.ITEM_CREATE,
            actor_id="user-1",
            resource_type="knowledge_item",
            resource_id="item-1",
        )

        assert entry.id == "entry-1"
        assert entry.action == AuditAction.ITEM_CREATE
        assert entry.success is True
        assert entry.error_message is None
        assert isinstance(entry.timestamp, datetime)

    def test_audit_entry_failure_tracking(self):
        """Test audit entry with failure information."""
        entry = AuditEntry(
            id="entry-1",
            action=AuditAction.ITEM_DELETE,
            actor_id="user-1",
            resource_type="item",
            resource_id="item-1",
            success=False,
            error_message="Permission denied",
        )

        assert not entry.success
        assert entry.error_message == "Permission denied"

    def test_audit_entry_to_dict(self):
        """Test audit entry serialization."""
        entry = AuditEntry(
            id="e1",
            action=AuditAction.ITEM_CREATE,
            actor_id="user-1",
            resource_type="item",
            resource_id="item-1",
            workspace_id="ws-1",
            details={"key": "value"},
            ip_address="192.168.1.1",
            user_agent="TestAgent/1.0",
        )

        d = entry.to_dict()
        assert d["id"] == "e1"
        assert d["action"] == "item.create"
        assert d["actor_id"] == "user-1"
        assert d["details"] == {"key": "value"}
        assert d["ip_address"] == "192.168.1.1"


# ============================================================================
# Test: AuditTrail
# ============================================================================


class TestAuditTrail:
    """Test AuditTrail logging and querying."""

    @pytest.mark.asyncio
    async def test_log_entry_basic(self, audit_trail):
        """Test basic audit entry logging."""
        entry = await audit_trail.log(
            action=AuditAction.ITEM_CREATE,
            actor_id="user-1",
            resource_type="knowledge_item",
            resource_id="item-1",
        )

        assert entry.id is not None
        assert entry.action == AuditAction.ITEM_CREATE

    @pytest.mark.asyncio
    async def test_query_by_actor(self, audit_trail):
        """Test querying audit entries by actor."""
        await audit_trail.log(
            action=AuditAction.ITEM_CREATE,
            actor_id="alice",
            resource_type="item",
            resource_id="item-1",
        )
        await audit_trail.log(
            action=AuditAction.ITEM_UPDATE,
            actor_id="bob",
            resource_type="item",
            resource_id="item-2",
        )

        alice_entries = await audit_trail.query(actor_id="alice")
        assert len(alice_entries) == 1
        assert alice_entries[0].actor_id == "alice"

    @pytest.mark.asyncio
    async def test_query_by_action(self, audit_trail):
        """Test querying audit entries by action type."""
        await audit_trail.log(
            action=AuditAction.ITEM_CREATE,
            actor_id="user-1",
            resource_type="item",
            resource_id="item-1",
        )
        await audit_trail.log(
            action=AuditAction.ITEM_DELETE,
            actor_id="user-1",
            resource_type="item",
            resource_id="item-2",
        )

        creates = await audit_trail.query(action=AuditAction.ITEM_CREATE)
        assert len(creates) == 1
        assert creates[0].action == AuditAction.ITEM_CREATE

    @pytest.mark.asyncio
    async def test_query_by_resource(self, audit_trail):
        """Test querying by resource type and ID."""
        await audit_trail.log(
            action=AuditAction.ITEM_READ,
            actor_id="user-1",
            resource_type="knowledge_item",
            resource_id="item-123",
        )
        await audit_trail.log(
            action=AuditAction.ITEM_READ,
            actor_id="user-1",
            resource_type="role",
            resource_id="role-1",
        )

        items = await audit_trail.query(resource_type="knowledge_item")
        assert len(items) == 1
        assert items[0].resource_type == "knowledge_item"

    @pytest.mark.asyncio
    async def test_query_success_only(self, audit_trail):
        """Test filtering for successful operations only."""
        await audit_trail.log(
            action=AuditAction.ITEM_CREATE,
            actor_id="user-1",
            resource_type="item",
            resource_id="item-1",
            success=True,
        )
        await audit_trail.log(
            action=AuditAction.ITEM_DELETE,
            actor_id="user-1",
            resource_type="item",
            resource_id="item-2",
            success=False,
            error_message="Access denied",
        )

        success_entries = await audit_trail.query(success_only=True)
        assert len(success_entries) == 1
        assert success_entries[0].success is True

    @pytest.mark.asyncio
    async def test_query_time_range(self, audit_trail):
        """Test querying within a time range."""
        now = datetime.now()

        await audit_trail.log(
            action=AuditAction.ITEM_CREATE,
            actor_id="user-1",
            resource_type="item",
            resource_id="item-1",
        )

        # Query with range that includes now
        entries = await audit_trail.query(
            start_time=now - timedelta(minutes=5),
            end_time=now + timedelta(minutes=5),
        )
        assert len(entries) >= 1

        # Query with range in the past
        old_entries = await audit_trail.query(
            start_time=now - timedelta(days=10),
            end_time=now - timedelta(days=5),
        )
        assert len(old_entries) == 0

    @pytest.mark.asyncio
    async def test_query_limit_and_offset(self, audit_trail):
        """Test query pagination with limit and offset."""
        # Create 10 entries
        for i in range(10):
            await audit_trail.log(
                action=AuditAction.ITEM_READ,
                actor_id=f"user-{i}",
                resource_type="item",
                resource_id=f"item-{i}",
            )

        # Query with limit
        entries = await audit_trail.query(limit=5)
        assert len(entries) == 5

        # Query with offset
        entries_offset = await audit_trail.query(limit=5, offset=5)
        assert len(entries_offset) == 5

        # Ensure different entries
        first_ids = {e.id for e in entries}
        second_ids = {e.id for e in entries_offset}
        assert len(first_ids & second_ids) == 0

    @pytest.mark.asyncio
    async def test_get_user_activity_summary(self, audit_trail):
        """Test user activity summary generation."""
        # Create various activities
        for action in [AuditAction.ITEM_CREATE, AuditAction.ITEM_READ, AuditAction.ITEM_UPDATE]:
            await audit_trail.log(
                action=action,
                actor_id="user-1",
                resource_type="item",
                resource_id="item-1",
            )

        activity = await audit_trail.get_user_activity("user-1", days=30)

        assert activity["user_id"] == "user-1"
        assert activity["total_actions"] == 3
        assert "item.create" in activity["by_action"]
        assert "item.read" in activity["by_action"]
        assert activity["success_rate"] == 1.0

    def test_get_stats(self, audit_trail):
        """Test audit trail statistics."""
        # Initially empty
        stats = audit_trail.get_stats()
        assert stats["total_entries"] == 0
        assert stats["success_rate"] == 1.0

    @pytest.mark.asyncio
    async def test_max_entries_trimming(self):
        """Test that in-memory entries are trimmed when max is exceeded."""
        small_trail = AuditTrail(max_entries=5, enable_persistence=False)

        for i in range(10):
            await small_trail.log(
                action=AuditAction.ITEM_READ,
                actor_id=f"user-{i}",
                resource_type="item",
                resource_id=f"item-{i}",
            )

        stats = small_trail.get_stats()
        assert stats["total_entries"] == 5


# ============================================================================
# Test: GovernanceMixin
# ============================================================================


class TestGovernanceMixin:
    """Test GovernanceMixin integration with knowledge classes."""

    @pytest.mark.asyncio
    async def test_mixin_creates_rbac_manager(self, governance_mixin):
        """Test that mixin creates RBAC manager on demand."""
        assert governance_mixin._rbac_manager is None

        # Call a method that uses RBAC
        await governance_mixin.check_permission("user-1", Permission.READ)

        # Manager should now exist
        assert governance_mixin._rbac_manager is not None

    @pytest.mark.asyncio
    async def test_mixin_creates_audit_trail(self, governance_mixin):
        """Test that mixin creates audit trail on demand."""
        assert governance_mixin._audit_trail is None

        # Call audit method
        await governance_mixin.log_audit(
            action=AuditAction.ITEM_CREATE,
            actor_id="user-1",
            resource_type="item",
            resource_id="item-1",
        )

        # Trail should now exist
        assert governance_mixin._audit_trail is not None

    @pytest.mark.asyncio
    async def test_mixin_role_workflow(self, governance_mixin):
        """Test complete role workflow via mixin."""
        # Create a custom role
        role = await governance_mixin.create_role(
            name="Custom Reader",
            permissions={Permission.READ, Permission.EXPORT},
            description="Can read and export",
        )

        # Assign to user
        await governance_mixin.assign_role(user_id="user-1", role_id=role.id)

        # Check permissions
        assert await governance_mixin.check_permission("user-1", Permission.READ)
        assert await governance_mixin.check_permission("user-1", Permission.EXPORT)
        assert not await governance_mixin.check_permission("user-1", Permission.DELETE)

        # Revoke
        await governance_mixin.revoke_role("user-1", role.id)
        assert not await governance_mixin.check_permission("user-1", Permission.READ)

    @pytest.mark.asyncio
    async def test_mixin_audit_workflow(self, governance_mixin):
        """Test audit logging via mixin."""
        # Log an action
        entry = await governance_mixin.log_audit(
            action=AuditAction.ITEM_CREATE,
            actor_id="user-1",
            resource_type="knowledge_item",
            resource_id="item-123",
            workspace_id="ws-1",
            details={"source": "import"},
        )

        assert entry.id is not None

        # Query it back
        entries = await governance_mixin.query_audit(actor_id="user-1")
        assert len(entries) >= 1

    @pytest.mark.asyncio
    async def test_mixin_get_user_activity(self, governance_mixin):
        """Test getting user activity via mixin."""
        # Log some activities
        await governance_mixin.log_audit(
            action=AuditAction.ITEM_READ,
            actor_id="user-1",
            resource_type="item",
            resource_id="item-1",
        )

        activity = await governance_mixin.get_user_activity("user-1", days=7)
        assert activity["total_actions"] >= 1

    def test_mixin_get_governance_stats(self, governance_mixin):
        """Test getting governance stats via mixin."""
        stats = governance_mixin.get_governance_stats()
        assert "audit" in stats


# ============================================================================
# Test: Singleton Functions
# ============================================================================


class TestSingletonFunctions:
    """Test global singleton accessor functions."""

    def test_get_rbac_manager_singleton(self):
        """Test that get_rbac_manager returns a singleton."""
        manager1 = get_rbac_manager()
        manager2 = get_rbac_manager()
        assert manager1 is manager2

    def test_get_audit_trail_singleton(self):
        """Test that get_audit_trail returns a singleton."""
        trail1 = get_audit_trail()
        trail2 = get_audit_trail()
        assert trail1 is trail2


# ============================================================================
# Test: Edge Cases and Error Handling
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_empty_permissions_role(self, rbac_manager):
        """Test creating a role with no permissions."""
        role = await rbac_manager.create_role(
            name="Empty Role",
            permissions=set(),
            description="Has no permissions",
        )

        await rbac_manager.assign_role(user_id="user-1", role_id=role.id)

        # Should have no permissions
        for perm in Permission:
            if perm != Permission.ADMIN:  # Skip admin as it's a special case
                assert not await rbac_manager.check_permission("user-1", perm)

    @pytest.mark.asyncio
    async def test_audit_entry_with_empty_details(self, audit_trail):
        """Test audit entry with empty details dict."""
        entry = await audit_trail.log(
            action=AuditAction.ITEM_READ,
            actor_id="user-1",
            resource_type="item",
            resource_id="item-1",
            details=None,
        )

        assert entry.details == {}

    @pytest.mark.asyncio
    async def test_query_nonexistent_workspace(self, audit_trail):
        """Test querying with non-existent workspace."""
        await audit_trail.log(
            action=AuditAction.ITEM_CREATE,
            actor_id="user-1",
            resource_type="item",
            resource_id="item-1",
            workspace_id="ws-1",
        )

        entries = await audit_trail.query(workspace_id="nonexistent-ws")
        assert len(entries) == 0

    @pytest.mark.asyncio
    async def test_concurrent_role_operations(self, rbac_manager):
        """Test concurrent role operations don't corrupt state."""
        num_users = 20

        async def assign_to_user(i: int):
            user_id = f"concurrent-user-{i}"
            await rbac_manager.assign_role(user_id=user_id, role_id="builtin:viewer")
            return user_id

        # Run concurrently
        tasks = [assign_to_user(i) for i in range(num_users)]
        user_ids = await asyncio.gather(*tasks)

        # Verify all succeeded
        for user_id in user_ids:
            assert await rbac_manager.check_permission(user_id, Permission.READ)

    @pytest.mark.asyncio
    async def test_concurrent_audit_logging(self, audit_trail):
        """Test concurrent audit logging doesn't lose entries."""
        num_entries = 50

        async def log_entry(i: int):
            return await audit_trail.log(
                action=AuditAction.ITEM_READ,
                actor_id=f"user-{i}",
                resource_type="item",
                resource_id=f"item-{i}",
            )

        # Log concurrently
        tasks = [log_entry(i) for i in range(num_entries)]
        entries = await asyncio.gather(*tasks)

        assert len(entries) == num_entries

        # Verify all are in memory
        all_entries = await audit_trail.query(limit=num_entries + 10)
        assert len(all_entries) >= num_entries


# ============================================================================
# Test: Workspace Scoped Permissions
# ============================================================================


class TestWorkspaceScopedPermissions:
    """Test workspace-scoped permission evaluation."""

    @pytest.mark.asyncio
    async def test_global_role_applies_everywhere(self, rbac_manager):
        """Test that global roles work in all workspaces."""
        await rbac_manager.assign_role(
            user_id="global-user",
            role_id="builtin:editor",
            workspace_id=None,  # Global
        )

        # Should have permissions in any workspace
        for ws_id in ["ws-1", "ws-2", "ws-3", None]:
            assert await rbac_manager.check_permission(
                "global-user", Permission.UPDATE, workspace_id=ws_id
            )

    @pytest.mark.asyncio
    async def test_mixed_global_and_workspace_roles(self, rbac_manager):
        """Test user with both global and workspace-scoped roles."""
        # Assign global viewer role
        await rbac_manager.assign_role(
            user_id="mixed-user",
            role_id="builtin:viewer",
            workspace_id=None,
        )

        # Create and assign workspace-scoped editor role
        ws_role = await rbac_manager.create_role(
            name="WS1 Editor",
            permissions={Permission.UPDATE, Permission.DELETE},
            workspace_id="ws-1",
        )
        await rbac_manager.assign_role(
            user_id="mixed-user",
            role_id=ws_role.id,
            workspace_id="ws-1",
        )

        # In ws-1: should have READ (global) + UPDATE/DELETE (workspace)
        assert await rbac_manager.check_permission(
            "mixed-user", Permission.READ, workspace_id="ws-1"
        )
        assert await rbac_manager.check_permission(
            "mixed-user", Permission.UPDATE, workspace_id="ws-1"
        )

        # In ws-2: should only have READ (global)
        assert await rbac_manager.check_permission(
            "mixed-user", Permission.READ, workspace_id="ws-2"
        )
        assert not await rbac_manager.check_permission(
            "mixed-user", Permission.UPDATE, workspace_id="ws-2"
        )


__all__ = [
    "TestPermissionEnum",
    "TestRoleDataclass",
    "TestRoleAssignmentDataclass",
    "TestRBACManagerPolicyEvaluation",
    "TestAuditActionAndEntry",
    "TestAuditTrail",
    "TestGovernanceMixin",
    "TestSingletonFunctions",
    "TestEdgeCases",
    "TestWorkspaceScopedPermissions",
]
