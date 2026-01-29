"""
Tests for RBAC Types module.

Tests cover:
- ResourceType enum
- Action enum
- Scope enum
- Permission dataclass (matching, serialization)
- Role dataclass (permission checking, serialization)
- RoleAssignment dataclass
- IsolationContext dataclass
- SYSTEM_ROLES definitions
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from aragora.rbac.types import (
    Action,
    IsolationContext,
    Permission,
    ResourceType,
    Role,
    RoleAssignment,
    Scope,
    SYSTEM_ROLES,
)


# =============================================================================
# Test Enums
# =============================================================================


class TestResourceType:
    """Tests for ResourceType enum."""

    def test_core_resource_types(self):
        """Core resource types exist."""
        assert ResourceType.DEBATE.value == "debate"
        assert ResourceType.AGENT.value == "agent"
        assert ResourceType.WORKFLOW.value == "workflow"
        assert ResourceType.DOCUMENT.value == "document"

    def test_knowledge_resource_types(self):
        """Knowledge resource types exist."""
        assert ResourceType.MEMORY.value == "memory"
        assert ResourceType.CULTURE.value == "culture"
        assert ResourceType.KNOWLEDGE_NODE.value == "knowledge_node"

    def test_audit_resource_types(self):
        """Audit resource types exist."""
        assert ResourceType.AUDIT_SESSION.value == "audit_session"
        assert ResourceType.AUDIT_FINDING.value == "audit_finding"

    def test_admin_resource_types(self):
        """Administrative resource types exist."""
        assert ResourceType.WORKSPACE.value == "workspace"
        assert ResourceType.ORGANIZATION.value == "organization"
        assert ResourceType.BILLING.value == "billing"
        assert ResourceType.API_KEY.value == "api_key"

    def test_resource_type_is_string_enum(self):
        """ResourceType is string enum for JSON serialization."""
        assert isinstance(ResourceType.DEBATE, str)
        assert ResourceType.DEBATE == "debate"


class TestAction:
    """Tests for Action enum."""

    def test_crud_actions(self):
        """CRUD actions exist."""
        assert Action.CREATE.value == "create"
        assert Action.READ.value == "read"
        assert Action.UPDATE.value == "update"
        assert Action.DELETE.value == "delete"

    def test_additional_actions(self):
        """Additional actions exist."""
        assert Action.EXECUTE.value == "execute"
        assert Action.EXPORT.value == "export"
        assert Action.SHARE.value == "share"
        assert Action.ADMIN.value == "admin"


class TestScope:
    """Tests for Scope enum."""

    def test_scope_values(self):
        """All scope values exist."""
        assert Scope.GLOBAL.value == "global"
        assert Scope.ORGANIZATION.value == "organization"
        assert Scope.WORKSPACE.value == "workspace"
        assert Scope.RESOURCE.value == "resource"


# =============================================================================
# Test Permission
# =============================================================================


class TestPermission:
    """Tests for Permission dataclass."""

    def test_create_permission(self):
        """Create a permission."""
        perm = Permission(
            resource=ResourceType.DEBATE,
            action=Action.READ,
            scope=Scope.WORKSPACE,
        )

        assert perm.resource == ResourceType.DEBATE
        assert perm.action == Action.READ
        assert perm.scope == Scope.WORKSPACE
        assert perm.conditions == {}

    def test_permission_with_conditions(self):
        """Create permission with conditions."""
        perm = Permission(
            resource=ResourceType.DEBATE,
            action=Action.DELETE,
            scope=Scope.WORKSPACE,
            conditions={"owner_id": "user-123"},
        )

        assert perm.conditions["owner_id"] == "user-123"

    def test_permission_default_scope(self):
        """Permission defaults to workspace scope."""
        perm = Permission(
            resource=ResourceType.DEBATE,
            action=Action.READ,
        )

        assert perm.scope == Scope.WORKSPACE

    def test_permission_hash(self):
        """Permissions are hashable for use in sets."""
        perm1 = Permission(
            resource=ResourceType.DEBATE,
            action=Action.READ,
            scope=Scope.WORKSPACE,
        )
        perm2 = Permission(
            resource=ResourceType.DEBATE,
            action=Action.READ,
            scope=Scope.WORKSPACE,
        )

        assert hash(perm1) == hash(perm2)
        assert {perm1, perm2} == {perm1}

    def test_permission_equality(self):
        """Permissions compare equal based on resource/action/scope."""
        perm1 = Permission(
            resource=ResourceType.DEBATE,
            action=Action.READ,
            scope=Scope.WORKSPACE,
        )
        perm2 = Permission(
            resource=ResourceType.DEBATE,
            action=Action.READ,
            scope=Scope.WORKSPACE,
        )
        perm3 = Permission(
            resource=ResourceType.DEBATE,
            action=Action.UPDATE,  # Different action
            scope=Scope.WORKSPACE,
        )

        assert perm1 == perm2
        assert perm1 != perm3

    def test_permission_matches_exact(self):
        """Permission matches exact resource and action."""
        perm = Permission(
            resource=ResourceType.DEBATE,
            action=Action.READ,
        )

        assert perm.matches(ResourceType.DEBATE, Action.READ)
        assert not perm.matches(ResourceType.DEBATE, Action.UPDATE)
        assert not perm.matches(ResourceType.AGENT, Action.READ)

    def test_permission_admin_matches_any_action(self):
        """Admin permission matches any action on resource."""
        perm = Permission(
            resource=ResourceType.DEBATE,
            action=Action.ADMIN,
        )

        assert perm.matches(ResourceType.DEBATE, Action.READ)
        assert perm.matches(ResourceType.DEBATE, Action.UPDATE)
        assert perm.matches(ResourceType.DEBATE, Action.DELETE)
        assert perm.matches(ResourceType.DEBATE, Action.ADMIN)

    def test_permission_matches_with_conditions(self):
        """Permission respects conditions when matching."""
        perm = Permission(
            resource=ResourceType.DEBATE,
            action=Action.DELETE,
            conditions={"owner_id": "user-123"},
        )

        # Matches with correct context
        assert perm.matches(ResourceType.DEBATE, Action.DELETE, context={"owner_id": "user-123"})

        # Doesn't match with wrong context
        assert not perm.matches(
            ResourceType.DEBATE, Action.DELETE, context={"owner_id": "user-456"}
        )

    def test_permission_matches_no_context_with_conditions(self):
        """Permission with conditions matches when no context provided."""
        perm = Permission(
            resource=ResourceType.DEBATE,
            action=Action.DELETE,
            conditions={"owner_id": "user-123"},
        )

        # No context provided, conditions not checked
        assert perm.matches(ResourceType.DEBATE, Action.DELETE)

    def test_permission_to_dict(self):
        """Permission serializes to dict."""
        perm = Permission(
            resource=ResourceType.DEBATE,
            action=Action.READ,
            scope=Scope.WORKSPACE,
            conditions={"key": "value"},
        )

        result = perm.to_dict()
        assert result["resource"] == "debate"
        assert result["action"] == "read"
        assert result["scope"] == "workspace"
        assert result["conditions"] == {"key": "value"}

    def test_permission_from_dict(self):
        """Permission deserializes from dict."""
        data = {
            "resource": "debate",
            "action": "read",
            "scope": "workspace",
            "conditions": {"key": "value"},
        }

        perm = Permission.from_dict(data)
        assert perm.resource == ResourceType.DEBATE
        assert perm.action == Action.READ
        assert perm.scope == Scope.WORKSPACE
        assert perm.conditions == {"key": "value"}


# =============================================================================
# Test Role
# =============================================================================


class TestRole:
    """Tests for Role dataclass."""

    def test_create_role(self):
        """Create a role."""
        role = Role(
            id="role-123",
            name="Test Role",
            description="A test role",
            permissions={
                Permission(resource=ResourceType.DEBATE, action=Action.READ),
            },
            scope=Scope.WORKSPACE,
        )

        assert role.id == "role-123"
        assert role.name == "Test Role"
        assert len(role.permissions) == 1
        assert not role.is_system

    def test_role_has_permission_exact(self):
        """Role checks permission grants."""
        role = Role(
            id="role-123",
            name="Viewer",
            description="Read only",
            permissions={
                Permission(resource=ResourceType.DEBATE, action=Action.READ),
                Permission(resource=ResourceType.DOCUMENT, action=Action.READ),
            },
            scope=Scope.WORKSPACE,
        )

        assert role.has_permission(ResourceType.DEBATE, Action.READ)
        assert role.has_permission(ResourceType.DOCUMENT, Action.READ)
        assert not role.has_permission(ResourceType.DEBATE, Action.UPDATE)
        assert not role.has_permission(ResourceType.AGENT, Action.READ)

    def test_role_has_permission_admin(self):
        """Admin permission grants all actions."""
        role = Role(
            id="role-admin",
            name="Admin",
            description="Full access",
            permissions={
                Permission(resource=ResourceType.DEBATE, action=Action.ADMIN),
            },
            scope=Scope.WORKSPACE,
        )

        assert role.has_permission(ResourceType.DEBATE, Action.READ)
        assert role.has_permission(ResourceType.DEBATE, Action.UPDATE)
        assert role.has_permission(ResourceType.DEBATE, Action.DELETE)

    def test_role_has_permission_with_context(self):
        """Role respects conditions in permission check."""
        role = Role(
            id="role-owner",
            name="Owner",
            description="Can delete own debates",
            permissions={
                Permission(
                    resource=ResourceType.DEBATE,
                    action=Action.DELETE,
                    conditions={"owner_id": "user-123"},
                ),
            },
            scope=Scope.WORKSPACE,
        )

        assert role.has_permission(
            ResourceType.DEBATE, Action.DELETE, context={"owner_id": "user-123"}
        )
        assert not role.has_permission(
            ResourceType.DEBATE, Action.DELETE, context={"owner_id": "user-456"}
        )

    def test_role_to_dict(self):
        """Role serializes to dict."""
        role = Role(
            id="role-123",
            name="Test Role",
            description="A test role",
            permissions={
                Permission(resource=ResourceType.DEBATE, action=Action.READ),
            },
            scope=Scope.WORKSPACE,
            is_system=True,
            created_by="admin",
        )

        result = role.to_dict()
        assert result["id"] == "role-123"
        assert result["name"] == "Test Role"
        assert result["scope"] == "workspace"
        assert result["is_system"] is True
        assert len(result["permissions"]) == 1


# =============================================================================
# Test RoleAssignment
# =============================================================================


class TestRoleAssignment:
    """Tests for RoleAssignment dataclass."""

    def test_create_role_assignment(self):
        """Create a role assignment."""
        assignment = RoleAssignment(
            actor_id="user-123",
            role_id="role-456",
            scope=Scope.WORKSPACE,
            scope_id="ws-789",
        )

        assert assignment.actor_id == "user-123"
        assert assignment.role_id == "role-456"
        assert assignment.scope == Scope.WORKSPACE
        assert assignment.scope_id == "ws-789"

    def test_role_assignment_not_expired(self):
        """Assignment is not expired without expiration."""
        assignment = RoleAssignment(
            actor_id="user-123",
            role_id="role-456",
            scope=Scope.WORKSPACE,
            scope_id="ws-789",
        )

        assert not assignment.is_expired

    def test_role_assignment_not_expired_future(self):
        """Assignment not expired with future expiration."""
        assignment = RoleAssignment(
            actor_id="user-123",
            role_id="role-456",
            scope=Scope.WORKSPACE,
            scope_id="ws-789",
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        )

        assert not assignment.is_expired

    def test_role_assignment_expired(self):
        """Assignment is expired with past expiration."""
        assignment = RoleAssignment(
            actor_id="user-123",
            role_id="role-456",
            scope=Scope.WORKSPACE,
            scope_id="ws-789",
            expires_at=datetime.now(timezone.utc) - timedelta(hours=1),
        )

        assert assignment.is_expired

    def test_role_assignment_to_dict(self):
        """Assignment serializes to dict."""
        assignment = RoleAssignment(
            actor_id="user-123",
            role_id="role-456",
            scope=Scope.WORKSPACE,
            scope_id="ws-789",
            assigned_by="admin",
        )

        result = assignment.to_dict()
        assert result["actor_id"] == "user-123"
        assert result["role_id"] == "role-456"
        assert result["scope"] == "workspace"
        assert result["assigned_by"] == "admin"
        assert result["expires_at"] is None


# =============================================================================
# Test IsolationContext
# =============================================================================


class TestIsolationContext:
    """Tests for IsolationContext dataclass."""

    def test_create_isolation_context(self):
        """Create an isolation context."""
        ctx = IsolationContext(
            actor_id="user-123",
            organization_id="org-456",
            workspace_id="ws-789",
        )

        assert ctx.actor_id == "user-123"
        assert ctx.actor_type == "user"
        assert ctx.organization_id == "org-456"
        assert ctx.workspace_id == "ws-789"

    def test_isolation_context_service_type(self):
        """Context can be for service actors."""
        ctx = IsolationContext(
            actor_id="service-123",
            actor_type="service",
        )

        assert ctx.actor_type == "service"

    def test_isolation_context_with_workspace(self):
        """Create new context scoped to workspace."""
        ctx = IsolationContext(
            actor_id="user-123",
            organization_id="org-456",
        )

        new_ctx = ctx.with_workspace("ws-789")

        assert new_ctx.workspace_id == "ws-789"
        assert new_ctx.actor_id == "user-123"
        assert new_ctx.organization_id == "org-456"

    def test_isolation_context_with_organization(self):
        """Create new context scoped to organization."""
        ctx = IsolationContext(
            actor_id="user-123",
            workspace_id="ws-789",
        )

        new_ctx = ctx.with_organization("org-456")

        assert new_ctx.organization_id == "org-456"
        assert new_ctx.actor_id == "user-123"
        assert new_ctx.workspace_id == "ws-789"

    def test_isolation_context_to_dict(self):
        """Context serializes to dict."""
        ctx = IsolationContext(
            actor_id="user-123",
            actor_type="user",
            organization_id="org-456",
            workspace_id="ws-789",
            request_id="req-123",
            correlation_id="corr-456",
        )

        result = ctx.to_dict()
        assert result["actor_id"] == "user-123"
        assert result["actor_type"] == "user"
        assert result["organization_id"] == "org-456"
        assert result["workspace_id"] == "ws-789"
        assert result["request_id"] == "req-123"


# =============================================================================
# Test SYSTEM_ROLES
# =============================================================================


class TestSystemRoles:
    """Tests for predefined system roles."""

    def test_superadmin_exists(self):
        """Superadmin role exists."""
        assert "superadmin" in SYSTEM_ROLES
        role = SYSTEM_ROLES["superadmin"]
        assert role.id == "role_superadmin"
        assert role.is_system is True
        assert role.scope == Scope.GLOBAL

    def test_superadmin_has_all_permissions(self):
        """Superadmin has admin on all resources."""
        role = SYSTEM_ROLES["superadmin"]

        for resource_type in ResourceType:
            assert role.has_permission(resource_type, Action.ADMIN)
            assert role.has_permission(resource_type, Action.READ)
            assert role.has_permission(resource_type, Action.DELETE)

    def test_org_admin_exists(self):
        """Organization admin role exists."""
        assert "org_admin" in SYSTEM_ROLES
        role = SYSTEM_ROLES["org_admin"]
        assert role.scope == Scope.ORGANIZATION
        assert role.is_system is True

    def test_workspace_admin_exists(self):
        """Workspace admin role exists."""
        assert "workspace_admin" in SYSTEM_ROLES
        role = SYSTEM_ROLES["workspace_admin"]
        assert role.scope == Scope.WORKSPACE

    def test_workspace_editor_exists(self):
        """Workspace editor role exists."""
        assert "workspace_editor" in SYSTEM_ROLES
        role = SYSTEM_ROLES["workspace_editor"]
        assert role.scope == Scope.WORKSPACE

        # Should have CRUD on debates
        assert role.has_permission(ResourceType.DEBATE, Action.CREATE)
        assert role.has_permission(ResourceType.DEBATE, Action.READ)
        assert role.has_permission(ResourceType.DEBATE, Action.UPDATE)
        assert role.has_permission(ResourceType.DEBATE, Action.DELETE)

    def test_workspace_viewer_exists(self):
        """Workspace viewer role exists."""
        assert "workspace_viewer" in SYSTEM_ROLES
        role = SYSTEM_ROLES["workspace_viewer"]

        # Should have read on all resources
        assert role.has_permission(ResourceType.DEBATE, Action.READ)
        assert role.has_permission(ResourceType.DOCUMENT, Action.READ)

        # Should not have write
        assert not role.has_permission(ResourceType.DEBATE, Action.CREATE)
        assert not role.has_permission(ResourceType.DEBATE, Action.DELETE)

    def test_auditor_exists(self):
        """Auditor role exists."""
        assert "auditor" in SYSTEM_ROLES
        role = SYSTEM_ROLES["auditor"]

        # Should have audit permissions
        assert role.has_permission(ResourceType.AUDIT_SESSION, Action.CREATE)
        assert role.has_permission(ResourceType.AUDIT_SESSION, Action.READ)
        assert role.has_permission(ResourceType.AUDIT_FINDING, Action.READ)

    def test_ml_engineer_exists(self):
        """ML Engineer role exists."""
        assert "ml_engineer" in SYSTEM_ROLES
        role = SYSTEM_ROLES["ml_engineer"]
        assert role.scope == Scope.ORGANIZATION

        # Should have training permissions
        assert role.has_permission(ResourceType.TRAINING_JOB, Action.CREATE)
        assert role.has_permission(ResourceType.SPECIALIST_MODEL, Action.READ)

    def test_all_system_roles_are_system(self):
        """All system roles have is_system=True."""
        for name, role in SYSTEM_ROLES.items():
            assert role.is_system, f"Role {name} should have is_system=True"
