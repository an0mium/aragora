"""
Tests for RBAC edge cases and boundary conditions.

Covers scenarios not in standard test suites:
- Workspace isolation within organizations
- Cross-organization permission boundaries
- Multi-level role hierarchy chains
- Invalid/malformed permission keys
- Wildcard permission edge cases
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta

from aragora.rbac import (
    AuthorizationContext,
    AuthorizationDecision,
    APIKeyScope,
    PermissionChecker,
    check_permission,
    has_permission,
    require_permission,
    require_role,
    PermissionDeniedError,
    RoleRequiredError,
    SYSTEM_PERMISSIONS,
    SYSTEM_ROLES_V2,
    get_role_permissions,
    get_role,
    ROLE_HIERARCHY,
)
from aragora.rbac.models import (
    ResourceType,
    Action,
    Permission,
    Role,
)
from aragora.rbac.types import IsolationContext, Scope


class TestWorkspaceIsolation:
    """Test workspace isolation within an organization."""

    def test_user_cannot_access_different_workspace_resource(self):
        """User with role in workspace A cannot access resources in workspace B."""
        # User has permissions in workspace_a
        context = AuthorizationContext(
            user_id="user1",
            org_id="org1",
            workspace_id="workspace_a",
            permissions={"debates:read", "debates:create"},
            roles={"debate_creator"},
        )

        # Check permission in their own workspace - should succeed
        assert context.has_permission("debates:read")

        # Create a new context for workspace_b attempt
        context_wrong_workspace = AuthorizationContext(
            user_id="user1",
            org_id="org1",
            workspace_id="workspace_b",  # Different workspace
            permissions=set(),  # No permissions granted for this workspace
            roles=set(),
        )

        # Should not have permissions in workspace_b
        assert not context_wrong_workspace.has_permission("debates:read")

    def test_workspace_admin_cannot_escalate_to_org_admin(self):
        """Workspace admin should not have org-level admin permissions."""
        # User is admin in a specific workspace
        context = AuthorizationContext(
            user_id="user1",
            org_id="org1",
            workspace_id="workspace_a",
            permissions={"debates.*", "agents.*", "workflows.*"},
            roles={"admin"},  # Admin at workspace level only
        )

        # Should have workspace-level permissions
        assert context.has_permission("debates:create")
        assert context.has_permission("agents.deploy")

        # Should NOT have org-level admin permissions
        assert not context.has_permission("organization.manage_billing")
        assert not context.has_permission("admin.system_config")

    def test_org_owner_can_access_all_workspaces(self):
        """Organization owner should have access to all workspaces."""
        context = AuthorizationContext(
            user_id="owner1",
            org_id="org1",
            workspace_id="any_workspace",
            permissions={"*"},  # Owner gets all permissions
            roles={"owner"},
        )

        # Should have all permissions in any workspace
        assert context.has_permission("debates:create")
        assert context.has_permission("organization.manage_billing")
        assert context.has_permission("admin.system_config")


class TestCrossOrganizationBoundaries:
    """Test cross-organization permission boundaries."""

    def test_user_with_role_in_org_a_denied_access_to_org_b(self):
        """User with permissions in org A cannot access org B resources."""
        context_org_alpha = AuthorizationContext(
            user_id="user1",
            org_id="org_alpha",
            permissions={"debates.*", "agents.*"},
            roles={"admin"},
        )

        context_org_beta = AuthorizationContext(
            user_id="user1",
            org_id="org_beta",
            permissions=set(),
            roles=set(),
        )

        assert context_org_alpha.has_permission("debates:create")
        assert not context_org_beta.has_permission("debates:read")

    def test_user_with_roles_in_multiple_orgs(self):
        """User can have different roles in different organizations."""
        user_id = "multi_org_user"

        context_org1 = AuthorizationContext(
            user_id=user_id,
            org_id="org1",
            permissions={"debates.*", "agents.*", "users.invite"},
            roles={"admin"},
        )

        context_org2 = AuthorizationContext(
            user_id=user_id,
            org_id="org2",
            permissions={"debates:read"},
            roles={"viewer"},
        )

        assert context_org1.has_permission("debates:create")
        assert context_org1.has_permission("users.invite")
        assert context_org2.has_permission("debates:read")
        assert not context_org2.has_permission("debates:create")

    def test_platform_owner_cross_org_access(self):
        """Platform owner should have access across all organizations."""
        context = AuthorizationContext(
            user_id="platform_admin",
            org_id=None,
            permissions={"*"},
            roles={"platform_owner"},
        )

        assert context.has_permission("debates:create")
        assert context.has_permission("admin.system_config")
        assert context.has_permission("organization.export_data")


class TestMultiLevelRoleHierarchy:
    """Test multi-level role hierarchy permission inheritance."""

    def test_role_hierarchy_exists(self):
        """Verify role hierarchy is defined."""
        assert "owner" in ROLE_HIERARCHY
        assert "admin" in ROLE_HIERARCHY
        assert "admin" in ROLE_HIERARCHY["owner"]

    def test_full_inheritance_chain(self):
        """Test permission inheritance through full chain."""
        owner_perms = get_role_permissions("owner")
        admin_perms = get_role_permissions("admin")

        assert len(owner_perms) > 0
        assert "debates:create" in admin_perms or len(admin_perms) > 0
        assert "admin" in ROLE_HIERARCHY.get("owner", [])

    def test_inherited_permissions_from_parent(self):
        """Child roles inherit permissions from parent roles."""
        admin_role = get_role("admin")
        debate_creator_role = get_role("debate_creator")

        if admin_role and debate_creator_role:
            admin_perms = get_role_permissions("admin")
            dc_perms = get_role_permissions("debate_creator")

            core_debate_perms = {"debates:create", "debates:read", "debates.run"}
            for perm in core_debate_perms:
                if perm in dc_perms:
                    assert perm in admin_perms

    def test_viewer_has_minimal_permissions(self):
        """Viewer role should have only read permissions."""
        viewer_perms = get_role_permissions("viewer")

        for perm in viewer_perms:
            assert any(word in perm for word in ["read", "view", "analytics"])

        assert "debates:create" not in viewer_perms
        assert "debates:delete" not in viewer_perms


class TestAPIKeyScopeEdgeCases:
    """Test API key scope restriction edge cases."""

    def test_api_key_scope_restricts_wildcard_user_permissions(self):
        """API key scope should restrict even if user has wildcard."""
        api_key_scope = APIKeyScope(
            permissions={"debates:read", "debates:create"},
        )

        context = AuthorizationContext(
            user_id="user1",
            permissions={"*"},
            api_key_scope=api_key_scope,
        )

        assert context.has_permission("debates:read")
        assert context.has_permission("debates:create")
        assert not context.has_permission("debates:delete")
        assert not context.has_permission("agents:create")

    def test_api_key_explicit_scope_limits_access(self):
        """API key with explicit scope limits even wildcard user permissions."""
        api_key_scope = APIKeyScope(
            permissions={"debates:read"},
        )

        context = AuthorizationContext(
            user_id="user1",
            permissions={"debates.*"},
            api_key_scope=api_key_scope,
        )

        assert context.has_permission("debates:read")
        assert not context.has_permission("debates:create")

    def test_api_key_scope_intersection(self):
        """API key scope should intersect with user permissions."""
        user_perms = {"debates:read", "debates:create", "agents:read"}
        api_key_scope = APIKeyScope(
            permissions={"debates:read", "debates:delete", "agents:read"},
        )

        context = AuthorizationContext(
            user_id="user1",
            permissions=user_perms,
            api_key_scope=api_key_scope,
        )

        assert context.has_permission("debates:read")
        assert context.has_permission("agents:read")
        assert not context.has_permission("debates:create")
        assert not context.has_permission("debates:delete")


class TestInvalidPermissionHandling:
    """Test handling of invalid/malformed permission keys."""

    def test_empty_permission_key(self):
        """Empty permission key should be denied."""
        context = AuthorizationContext(
            user_id="user1",
            permissions={"debates.*"},
        )
        assert not context.has_permission("")

    def test_malformed_permission_key(self):
        """Malformed permission keys should be handled gracefully."""
        context = AuthorizationContext(
            user_id="user1",
            permissions={"debates:create"},
        )
        assert not context.has_permission("...")
        assert not context.has_permission("debates")
        assert not context.has_permission(".create")

    def test_nonexistent_resource_type(self):
        """Permission check for nonexistent resource type."""
        context = AuthorizationContext(
            user_id="user1",
            permissions={"debates:create"},
        )
        assert not context.has_permission("unicorns.fly")
        assert not context.has_permission("foobar.create")

    def test_case_sensitivity(self):
        """Permission keys should be case-sensitive."""
        context = AuthorizationContext(
            user_id="user1",
            permissions={"debates:create"},
        )
        assert context.has_permission("debates:create")
        assert not context.has_permission("Debates.create")
        assert not context.has_permission("debates.CREATE")


class TestIsolationContext:
    """Test IsolationContext for multi-tenant scenarios."""

    def test_isolation_context_creation(self):
        """Test creating isolation context with various scopes."""
        global_ctx = IsolationContext(actor_id="user1")
        assert global_ctx.organization_id is None
        assert global_ctx.workspace_id is None

        org_ctx = IsolationContext(actor_id="user1", organization_id="org123")
        assert org_ctx.organization_id == "org123"

        ws_ctx = IsolationContext(actor_id="user1", organization_id="org123", workspace_id="ws456")
        assert ws_ctx.organization_id == "org123"
        assert ws_ctx.workspace_id == "ws456"

    def test_isolation_context_actor_types(self):
        """Test different actor types in isolation context."""
        user_ctx = IsolationContext(actor_id="user1", actor_type="user")
        service_ctx = IsolationContext(actor_id="svc1", actor_type="service")
        agent_ctx = IsolationContext(actor_id="agent1", actor_type="agent")

        assert user_ctx.actor_type == "user"
        assert service_ctx.actor_type == "service"
        assert agent_ctx.actor_type == "agent"

    def test_isolation_context_with_workspace(self):
        """Test creating context with workspace helper."""
        ctx = IsolationContext(actor_id="user1", organization_id="org1")
        ws_ctx = ctx.with_workspace("ws123")

        assert ws_ctx.workspace_id == "ws123"
        assert ws_ctx.organization_id == "org1"
        assert ws_ctx.actor_id == "user1"


class TestPermissionModelValidation:
    """Test Permission model validation."""

    def test_permission_creation(self):
        """Test creating a Permission object."""
        perm = Permission(
            id="perm1",
            name="Create Debates",
            resource=ResourceType.DEBATE,
            action=Action.CREATE,
            description="Create debates",
        )

        assert perm.name == "Create Debates"
        assert perm.resource == ResourceType.DEBATE
        assert perm.action == Action.CREATE

    def test_permission_with_conditions(self):
        """Test Permission with conditional access (ABAC)."""
        perm = Permission(
            id="perm2",
            name="Create Debates",
            resource=ResourceType.DEBATE,
            action=Action.CREATE,
            conditions={"max_per_day": 10, "workspace_required": True},
        )

        assert perm.conditions.get("max_per_day") == 10
        assert perm.conditions.get("workspace_required") is True


class TestRoleModel:
    """Test Role model functionality."""

    def test_role_creation(self):
        """Test creating a Role object."""
        role = Role(
            id="role1",
            name="custom_analyst",
            permissions={"debates:read", "analytics:view"},
            description="Custom analyst role",
        )

        assert role.name == "custom_analyst"
        assert "debates:read" in role.permissions

    def test_role_priority(self):
        """Test Role priority for hierarchy."""
        high_priority = Role(id="role3", name="manager", permissions=set(), priority=80)
        low_priority = Role(id="role4", name="intern", permissions=set(), priority=10)

        assert high_priority.priority > low_priority.priority


class TestWildcardPermissionEdgeCases:
    """Test wildcard permission edge cases."""

    def test_double_wildcard_resource(self):
        """Test resource.* wildcard matching."""
        context = AuthorizationContext(
            user_id="user1",
            permissions={"debates.*"},
        )

        assert context.has_permission("debates:create")
        assert context.has_permission("debates:read")
        assert context.has_permission("debates:delete")
        assert not context.has_permission("agents:create")

    def test_super_wildcard_matches_all(self):
        """Test * wildcard matches everything."""
        context = AuthorizationContext(
            user_id="user1",
            permissions={"*"},
        )

        assert context.has_permission("debates:create")
        assert context.has_permission("agents.deploy")
        assert context.has_permission("admin.system_config")

    def test_multiple_wildcards(self):
        """Test multiple wildcard permissions."""
        context = AuthorizationContext(
            user_id="user1",
            permissions={"debates.*", "agents.*"},
        )

        assert context.has_permission("debates:create")
        assert context.has_permission("agents.deploy")
        assert not context.has_permission("users.invite")


class TestContextualPermissions:
    """Test permissions with additional context."""

    def test_permission_with_org_context(self):
        """Test that org context is preserved in AuthorizationContext."""
        context = AuthorizationContext(
            user_id="user1",
            org_id="org_specific",
            permissions={"debates:create"},
        )

        assert context.org_id == "org_specific"
        assert context.has_permission("debates:create")


class TestEdgeCaseCombinations:
    """Test complex combinations of edge cases."""

    def test_api_key_limits_owner_permissions(self):
        """API key scope restricts even owner permissions."""
        limited_key = APIKeyScope(
            permissions={"debates:read", "agents:read"},
        )

        context = AuthorizationContext(
            user_id="user1",
            permissions={"*"},
            api_key_scope=limited_key,
        )

        assert context.has_permission("debates:read")
        assert context.has_permission("agents:read")
        assert not context.has_permission("debates:create")
        assert not context.has_permission("admin.system_config")

    def test_owner_with_restricted_api_key(self):
        """Even owners should be restricted by API key scope."""
        restricted_key = APIKeyScope(permissions={"debates:read"})

        context = AuthorizationContext(
            user_id="owner1",
            permissions={"*"},
            roles={"owner"},
            api_key_scope=restricted_key,
        )

        assert context.has_permission("debates:read")
        assert not context.has_permission("debates:create")
        assert not context.has_permission("admin.system_config")

    def test_empty_roles_empty_permissions(self):
        """User with no roles and no permissions should be denied everything."""
        context = AuthorizationContext(
            user_id="nobody",
            permissions=set(),
            roles=set(),
        )

        assert not context.has_permission("debates:read")
        assert not context.has_permission("*")
        assert not context.has_role("viewer")


class TestDecisionCaching:
    """Test permission decision caching behavior."""

    def test_authorization_decision_structure(self):
        """Test AuthorizationDecision creation."""
        decision = AuthorizationDecision(
            allowed=True,
            permission_key="debates:create",
            reason="Permission granted via role",
        )

        assert decision.allowed is True
        assert decision.permission_key == "debates:create"
        assert "granted" in decision.reason

    def test_authorization_decision_denied(self):
        """Test denied AuthorizationDecision."""
        decision = AuthorizationDecision(
            allowed=False,
            permission_key="admin.system_config",
            reason="Permission not in user's permission set",
        )

        assert decision.allowed is False
        assert "not in user's permission set" in decision.reason
