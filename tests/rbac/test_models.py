"""
Tests for RBAC Models module (aragora/rbac/models.py).

Tests cover:
- ResourceType enum: All resource types and string enum behavior
- Action enum: All actions and string enum behavior
- Permission dataclass: Creation, key generation, from_key parsing, matches logic
- Role dataclass: Permission management, hierarchy, custom roles
- RoleAssignment: Expiration handling, validity checks
- APIKeyScope: Permission and resource access control
- AuthorizationContext: Permission checks, role checks, API key scope integration
- AuthorizationDecision: Serialization
- Helper functions: _permission_candidates, _resource_candidates
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import UUID

import pytest

from aragora.rbac.models import (
    Action,
    APIKeyScope,
    AuthorizationContext,
    AuthorizationDecision,
    Permission,
    ResourceType,
    Role,
    RoleAssignment,
    _permission_candidates,
    _resource_candidates,
)


# =============================================================================
# Test ResourceType Enum
# =============================================================================


class TestResourceType:
    """Tests for ResourceType enum."""

    def test_core_resource_types_exist(self):
        """Core resource types should exist with correct values."""
        assert ResourceType.DEBATE.value == "debates"
        assert ResourceType.AGENT.value == "agents"
        assert ResourceType.USER.value == "users"
        assert ResourceType.ORGANIZATION.value == "organization"
        assert ResourceType.API.value == "api"
        assert ResourceType.MEMORY.value == "memory"
        assert ResourceType.WORKFLOW.value == "workflows"

    def test_knowledge_resource_types(self):
        """Knowledge-related resource types should exist."""
        assert ResourceType.EVIDENCE.value == "evidence"
        assert ResourceType.DOCUMENTS.value == "documents"
        assert ResourceType.KNOWLEDGE.value == "knowledge"
        assert ResourceType.PROVENANCE.value == "provenance"

    def test_enterprise_resource_types(self):
        """Enterprise resource types should exist."""
        assert ResourceType.BILLING.value == "billing"
        assert ResourceType.BACKUP.value == "backup"
        assert ResourceType.DISASTER_RECOVERY.value == "disaster_recovery"
        assert ResourceType.COMPLIANCE.value == "compliance"
        assert ResourceType.POLICY.value == "policies"

    def test_gauntlet_resource_types(self):
        """Gauntlet and related resource types should exist."""
        assert ResourceType.GAUNTLET.value == "gauntlet"
        assert ResourceType.EXPLAINABILITY.value == "explainability"
        assert ResourceType.FINDINGS.value == "findings"

    def test_data_governance_resource_types(self):
        """Data governance resource types should exist."""
        assert ResourceType.DATA_CLASSIFICATION.value == "data_classification"
        assert ResourceType.DATA_RETENTION.value == "data_retention"
        assert ResourceType.DATA_LINEAGE.value == "data_lineage"
        assert ResourceType.PII.value == "pii"

    def test_resource_type_is_string_enum(self):
        """ResourceType should be a string enum for JSON serialization."""
        assert isinstance(ResourceType.DEBATE, str)
        assert ResourceType.DEBATE == "debates"

    def test_resource_type_from_value(self):
        """ResourceType can be created from string value."""
        rt = ResourceType("debates")
        assert rt == ResourceType.DEBATE

    def test_resource_type_invalid_value_raises(self):
        """Invalid ResourceType value should raise ValueError."""
        with pytest.raises(ValueError):
            ResourceType("invalid_resource")

    def test_all_resource_types_have_values(self):
        """All ResourceType enum members should have string values."""
        for rt in ResourceType:
            assert isinstance(rt.value, str)
            assert len(rt.value) > 0


# =============================================================================
# Test Action Enum
# =============================================================================


class TestAction:
    """Tests for Action enum."""

    def test_crud_actions_exist(self):
        """CRUD actions should exist."""
        assert Action.CREATE.value == "create"
        assert Action.READ.value == "read"
        assert Action.UPDATE.value == "update"
        assert Action.WRITE.value == "write"
        assert Action.DELETE.value == "delete"

    def test_debate_actions_exist(self):
        """Debate-specific actions should exist."""
        assert Action.RUN.value == "run"
        assert Action.STOP.value == "stop"
        assert Action.PAUSE.value == "pause"
        assert Action.RESUME.value == "resume"
        assert Action.FORK.value == "fork"

    def test_agent_actions_exist(self):
        """Agent-specific actions should exist."""
        assert Action.DEPLOY.value == "deploy"
        assert Action.CONFIGURE.value == "configure"

    def test_user_management_actions_exist(self):
        """User management actions should exist."""
        assert Action.INVITE.value == "invite"
        assert Action.REMOVE.value == "remove"
        assert Action.CHANGE_ROLE.value == "change_role"
        assert Action.IMPERSONATE.value == "impersonate"

    def test_admin_actions_exist(self):
        """Admin actions should exist."""
        assert Action.SYSTEM_CONFIG.value == "system_config"
        assert Action.VIEW_METRICS.value == "view_metrics"
        assert Action.MANAGE_FEATURES.value == "manage_features"

    def test_gauntlet_actions_exist(self):
        """Gauntlet-specific actions should exist."""
        assert Action.SIGN.value == "sign"
        assert Action.COMPARE.value == "compare"
        assert Action.VERIFY.value == "verify"

    def test_wildcard_action_exists(self):
        """Wildcard action should exist."""
        assert Action.ALL.value == "*"

    def test_control_plane_sub_actions_exist(self):
        """Control plane sub-operation actions should exist."""
        assert Action.AGENTS_READ.value == "agents.read"
        assert Action.AGENTS_REGISTER.value == "agents.register"
        assert Action.TASKS_SUBMIT.value == "tasks.submit"
        assert Action.HEALTH_READ.value == "health.read"

    def test_computer_use_actions_exist(self):
        """Computer-use actions should exist."""
        assert Action.BROWSER.value == "browser"
        assert Action.SHELL.value == "shell"
        assert Action.FILE_READ.value == "file_read"
        assert Action.FILE_WRITE.value == "file_write"
        assert Action.SCREENSHOT.value == "screenshot"

    def test_action_is_string_enum(self):
        """Action should be a string enum for JSON serialization."""
        assert isinstance(Action.CREATE, str)
        assert Action.CREATE == "create"


# =============================================================================
# Test Permission Dataclass
# =============================================================================


class TestPermission:
    """Tests for Permission dataclass."""

    def test_create_permission(self):
        """Create a permission with required fields."""
        perm = Permission(
            id="perm-1",
            name="Create Debates",
            resource=ResourceType.DEBATE,
            action=Action.CREATE,
        )
        assert perm.id == "perm-1"
        assert perm.name == "Create Debates"
        assert perm.resource == ResourceType.DEBATE
        assert perm.action == Action.CREATE
        assert perm.description == ""
        assert perm.conditions == {}

    def test_permission_with_description(self):
        """Permission can have a description."""
        perm = Permission(
            id="perm-1",
            name="Create Debates",
            resource=ResourceType.DEBATE,
            action=Action.CREATE,
            description="Allows creating new debates",
        )
        assert perm.description == "Allows creating new debates"

    def test_permission_with_conditions(self):
        """Permission can have conditions for ABAC."""
        perm = Permission(
            id="perm-1",
            name="Delete Own Debates",
            resource=ResourceType.DEBATE,
            action=Action.DELETE,
            conditions={"owner_id": "${user_id}"},
        )
        assert perm.conditions == {"owner_id": "${user_id}"}

    def test_permission_key_property(self):
        """Permission key should be resource.action format."""
        perm = Permission(
            id="perm-1",
            name="Create Debates",
            resource=ResourceType.DEBATE,
            action=Action.CREATE,
        )
        assert perm.key == "debates.create"

    def test_permission_key_with_wildcard(self):
        """Permission key with wildcard action."""
        perm = Permission(
            id="perm-1",
            name="All Debate Actions",
            resource=ResourceType.DEBATE,
            action=Action.ALL,
        )
        assert perm.key == "debates.*"

    def test_permission_from_key_basic(self):
        """Create permission from key string."""
        perm = Permission.from_key("debates.create")
        assert perm.resource == ResourceType.DEBATE
        assert perm.action == Action.CREATE
        # ID should be a UUID
        UUID(perm.id)  # Will raise if invalid
        assert perm.name == "Debates Create"

    def test_permission_from_key_with_name(self):
        """Create permission from key with custom name."""
        perm = Permission.from_key("agents.deploy", name="Deploy Agents")
        assert perm.name == "Deploy Agents"
        assert perm.resource == ResourceType.AGENT
        assert perm.action == Action.DEPLOY

    def test_permission_from_key_with_description(self):
        """Create permission from key with description."""
        perm = Permission.from_key(
            "memory.read",
            name="Read Memory",
            description="Access memory contents",
        )
        assert perm.description == "Access memory contents"

    def test_permission_from_key_wildcard(self):
        """Create permission from key with wildcard action."""
        perm = Permission.from_key("admin.*")
        assert perm.resource == ResourceType.ADMIN
        assert perm.action == Action.ALL

    def test_permission_from_key_complex_action(self):
        """Create permission from key with complex action like agents.read."""
        perm = Permission.from_key("control_plane.agents.read")
        assert perm.resource == ResourceType.CONTROL_PLANE
        assert perm.action == Action.AGENTS_READ

    def test_permission_from_key_invalid_resource_raises(self):
        """Invalid resource in key should raise ValueError."""
        with pytest.raises(ValueError):
            Permission.from_key("invalid_resource.create")

    def test_permission_from_key_invalid_action_raises(self):
        """Invalid action in key should raise ValueError."""
        with pytest.raises(ValueError):
            Permission.from_key("debates.invalid_action")

    def test_permission_matches_exact(self):
        """Permission matches exact resource and action."""
        perm = Permission(
            id="perm-1",
            name="Read Debates",
            resource=ResourceType.DEBATE,
            action=Action.READ,
        )
        assert perm.matches(ResourceType.DEBATE, Action.READ) is True
        assert perm.matches(ResourceType.DEBATE, Action.CREATE) is False
        assert perm.matches(ResourceType.AGENT, Action.READ) is False

    def test_permission_matches_wildcard(self):
        """Wildcard permission matches all actions on resource."""
        perm = Permission(
            id="perm-1",
            name="All Debates",
            resource=ResourceType.DEBATE,
            action=Action.ALL,
        )
        assert perm.matches(ResourceType.DEBATE, Action.READ) is True
        assert perm.matches(ResourceType.DEBATE, Action.CREATE) is True
        assert perm.matches(ResourceType.DEBATE, Action.DELETE) is True
        # Should not match different resource
        assert perm.matches(ResourceType.AGENT, Action.READ) is False


# =============================================================================
# Test Role Dataclass
# =============================================================================


class TestRole:
    """Tests for Role dataclass."""

    def test_create_role_minimal(self):
        """Create a role with minimal fields."""
        role = Role(id="role-1", name="viewer")
        assert role.id == "role-1"
        assert role.name == "viewer"
        assert role.display_name == "Viewer"  # Auto-generated
        assert role.description == ""
        assert role.permissions == set()
        assert role.parent_roles == []
        assert role.is_system is True
        assert role.is_custom is False
        assert role.org_id is None
        assert role.priority == 0
        assert role.metadata == {}

    def test_create_role_full(self):
        """Create a role with all fields."""
        role = Role(
            id="role-1",
            name="debate_creator",
            display_name="Debate Creator",
            description="Can create debates",
            permissions={"debates.create", "debates.read"},
            parent_roles=["viewer"],
            is_system=False,
            is_custom=True,
            org_id="org-123",
            priority=50,
            metadata={"max_debates": 10},
        )
        assert role.display_name == "Debate Creator"
        assert role.description == "Can create debates"
        assert role.permissions == {"debates.create", "debates.read"}
        assert role.parent_roles == ["viewer"]
        assert role.is_system is False
        assert role.is_custom is True
        assert role.org_id == "org-123"
        assert role.priority == 50
        assert role.metadata == {"max_debates": 10}

    def test_role_display_name_auto_generated(self):
        """Role display_name is auto-generated from name."""
        role = Role(id="role-1", name="debate_creator")
        assert role.display_name == "Debate Creator"

        role2 = Role(id="role-2", name="super_admin_role")
        assert role2.display_name == "Super Admin Role"

    def test_role_display_name_not_overwritten(self):
        """Provided display_name is not overwritten."""
        role = Role(id="role-1", name="debate_creator", display_name="Custom Name")
        assert role.display_name == "Custom Name"

    def test_role_has_permission(self):
        """Role.has_permission checks direct permission."""
        role = Role(
            id="role-1",
            name="editor",
            permissions={"debates.create", "debates.read"},
        )
        assert role.has_permission("debates.create") is True
        assert role.has_permission("debates.read") is True
        assert role.has_permission("debates.delete") is False

    def test_role_add_permission(self):
        """Role.add_permission adds a permission."""
        role = Role(id="role-1", name="editor", permissions={"debates.read"})
        role.add_permission("debates.create")
        assert "debates.create" in role.permissions
        assert "debates.read" in role.permissions

    def test_role_add_permission_idempotent(self):
        """Adding same permission twice is idempotent."""
        role = Role(id="role-1", name="editor", permissions={"debates.read"})
        role.add_permission("debates.read")
        role.add_permission("debates.read")
        assert role.permissions == {"debates.read"}

    def test_role_remove_permission(self):
        """Role.remove_permission removes a permission."""
        role = Role(
            id="role-1",
            name="editor",
            permissions={"debates.create", "debates.read"},
        )
        role.remove_permission("debates.create")
        assert "debates.create" not in role.permissions
        assert "debates.read" in role.permissions

    def test_role_remove_nonexistent_permission(self):
        """Removing nonexistent permission does not raise."""
        role = Role(id="role-1", name="editor", permissions={"debates.read"})
        role.remove_permission("debates.delete")  # Should not raise
        assert role.permissions == {"debates.read"}

    def test_role_empty_permissions(self):
        """Role with empty permissions."""
        role = Role(id="role-1", name="guest")
        assert role.permissions == set()
        assert role.has_permission("anything") is False


# =============================================================================
# Test RoleAssignment Dataclass
# =============================================================================


class TestRoleAssignment:
    """Tests for RoleAssignment dataclass."""

    def test_create_role_assignment_minimal(self):
        """Create role assignment with minimal fields."""
        assignment = RoleAssignment(
            id="assign-1",
            user_id="user-1",
            role_id="editor",
        )
        assert assignment.id == "assign-1"
        assert assignment.user_id == "user-1"
        assert assignment.role_id == "editor"
        assert assignment.org_id is None
        assert assignment.assigned_by is None
        assert assignment.expires_at is None
        assert assignment.is_active is True
        assert assignment.conditions == {}
        assert assignment.metadata == {}

    def test_create_role_assignment_full(self):
        """Create role assignment with all fields."""
        assigned_at = datetime.now(timezone.utc)
        expires_at = assigned_at + timedelta(days=30)
        assignment = RoleAssignment(
            id="assign-1",
            user_id="user-1",
            role_id="editor",
            org_id="org-1",
            assigned_by="admin-1",
            assigned_at=assigned_at,
            expires_at=expires_at,
            is_active=True,
            conditions={"max_projects": 5},
            metadata={"reason": "Promotion"},
        )
        assert assignment.org_id == "org-1"
        assert assignment.assigned_by == "admin-1"
        assert assignment.expires_at == expires_at
        assert assignment.conditions == {"max_projects": 5}
        assert assignment.metadata == {"reason": "Promotion"}

    def test_role_assignment_not_expired_no_expiry(self):
        """Assignment without expiry is not expired."""
        assignment = RoleAssignment(
            id="assign-1",
            user_id="user-1",
            role_id="editor",
        )
        assert assignment.is_expired is False

    def test_role_assignment_not_expired_future_date(self):
        """Assignment with future expiry is not expired."""
        assignment = RoleAssignment(
            id="assign-1",
            user_id="user-1",
            role_id="editor",
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        )
        assert assignment.is_expired is False

    def test_role_assignment_expired_past_date(self):
        """Assignment with past expiry is expired."""
        assignment = RoleAssignment(
            id="assign-1",
            user_id="user-1",
            role_id="editor",
            expires_at=datetime.now(timezone.utc) - timedelta(hours=1),
        )
        assert assignment.is_expired is True

    def test_role_assignment_is_valid_active_not_expired(self):
        """Active, non-expired assignment is valid."""
        assignment = RoleAssignment(
            id="assign-1",
            user_id="user-1",
            role_id="editor",
            is_active=True,
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        )
        assert assignment.is_valid is True

    def test_role_assignment_is_valid_inactive(self):
        """Inactive assignment is not valid."""
        assignment = RoleAssignment(
            id="assign-1",
            user_id="user-1",
            role_id="editor",
            is_active=False,
        )
        assert assignment.is_valid is False

    def test_role_assignment_is_valid_expired(self):
        """Expired assignment is not valid."""
        assignment = RoleAssignment(
            id="assign-1",
            user_id="user-1",
            role_id="editor",
            is_active=True,
            expires_at=datetime.now(timezone.utc) - timedelta(hours=1),
        )
        assert assignment.is_valid is False

    def test_role_assignment_is_valid_inactive_and_expired(self):
        """Inactive and expired assignment is not valid."""
        assignment = RoleAssignment(
            id="assign-1",
            user_id="user-1",
            role_id="editor",
            is_active=False,
            expires_at=datetime.now(timezone.utc) - timedelta(hours=1),
        )
        assert assignment.is_valid is False


# =============================================================================
# Test APIKeyScope Dataclass
# =============================================================================


class TestAPIKeyScope:
    """Tests for APIKeyScope dataclass."""

    def test_create_api_key_scope_empty(self):
        """Empty APIKeyScope means full access."""
        scope = APIKeyScope()
        assert scope.permissions == set()
        assert scope.resources is None
        assert scope.rate_limit is None
        assert scope.expires_at is None
        assert scope.ip_whitelist is None

    def test_api_key_scope_with_permissions(self):
        """APIKeyScope with specific permissions."""
        scope = APIKeyScope(permissions={"debates.read", "debates.create"})
        assert scope.permissions == {"debates.read", "debates.create"}

    def test_api_key_scope_with_resources(self):
        """APIKeyScope with resource restrictions."""
        scope = APIKeyScope(
            resources={
                ResourceType.DEBATE: {"debate-1", "debate-2"},
                ResourceType.AGENT: {"agent-1"},
            }
        )
        assert ResourceType.DEBATE in scope.resources
        assert scope.resources[ResourceType.DEBATE] == {"debate-1", "debate-2"}

    def test_allows_permission_empty_scope(self):
        """Empty scope allows all permissions."""
        scope = APIKeyScope()
        assert scope.allows_permission("debates.create") is True
        assert scope.allows_permission("admin.system_config") is True

    def test_allows_permission_wildcard_scope(self):
        """Wildcard in scope allows all permissions."""
        scope = APIKeyScope(permissions={"*"})
        assert scope.allows_permission("debates.create") is True
        assert scope.allows_permission("admin.delete") is True

    def test_allows_permission_exact_match(self):
        """Exact permission match."""
        scope = APIKeyScope(permissions={"debates.read", "debates.create"})
        assert scope.allows_permission("debates.read") is True
        assert scope.allows_permission("debates.create") is True
        assert scope.allows_permission("debates.delete") is False

    def test_allows_permission_resource_wildcard(self):
        """Resource wildcard allows all actions on resource."""
        scope = APIKeyScope(permissions={"debates.*"})
        assert scope.allows_permission("debates.read") is True
        assert scope.allows_permission("debates.create") is True
        assert scope.allows_permission("debates.delete") is True
        assert scope.allows_permission("agents.read") is False

    def test_allows_permission_colon_format(self):
        """Permission with colon format is compatible."""
        scope = APIKeyScope(permissions={"debates:read"})
        # Should allow both dot and colon formats
        assert scope.allows_permission("debates.read") is True
        assert scope.allows_permission("debates:read") is True

    def test_allows_permission_dot_format_in_scope(self):
        """Dot format in scope matches colon format request."""
        scope = APIKeyScope(permissions={"debates.read"})
        assert scope.allows_permission("debates:read") is True

    def test_allows_permission_resource_wildcard_colon(self):
        """Resource wildcard with colon format."""
        scope = APIKeyScope(permissions={"debates:*"})
        assert scope.allows_permission("debates.read") is True
        assert scope.allows_permission("debates:create") is True

    def test_allows_resource_no_restrictions(self):
        """No resource restrictions allows all."""
        scope = APIKeyScope()
        assert scope.allows_resource(ResourceType.DEBATE, "debate-123") is True

    def test_allows_resource_type_not_restricted(self):
        """Resource type not in restrictions is allowed."""
        scope = APIKeyScope(resources={ResourceType.AGENT: {"agent-1"}})
        assert scope.allows_resource(ResourceType.DEBATE, "debate-123") is True

    def test_allows_resource_in_allowed_set(self):
        """Resource in allowed set is allowed."""
        scope = APIKeyScope(resources={ResourceType.DEBATE: {"debate-1", "debate-2"}})
        assert scope.allows_resource(ResourceType.DEBATE, "debate-1") is True
        assert scope.allows_resource(ResourceType.DEBATE, "debate-2") is True

    def test_allows_resource_not_in_allowed_set(self):
        """Resource not in allowed set is denied."""
        scope = APIKeyScope(resources={ResourceType.DEBATE: {"debate-1", "debate-2"}})
        assert scope.allows_resource(ResourceType.DEBATE, "debate-3") is False


# =============================================================================
# Test AuthorizationContext Dataclass
# =============================================================================


class TestAuthorizationContext:
    """Tests for AuthorizationContext dataclass."""

    def test_create_context_minimal(self):
        """Create context with minimal fields."""
        ctx = AuthorizationContext(user_id="user-1")
        assert ctx.user_id == "user-1"
        assert ctx.user_email is None
        assert ctx.org_id is None
        assert ctx.workspace_id is None
        assert ctx.roles == set()
        assert ctx.permissions == set()
        assert ctx.api_key_scope is None
        assert ctx.ip_address is None
        assert ctx.user_agent is None
        # request_id should be auto-generated UUID
        UUID(ctx.request_id)

    def test_create_context_full(self):
        """Create context with all fields."""
        scope = APIKeyScope(permissions={"debates.read"})
        ctx = AuthorizationContext(
            user_id="user-1",
            user_email="user@example.com",
            org_id="org-1",
            workspace_id="ws-1",
            roles={"editor", "viewer"},
            permissions={"debates.read", "debates.create"},
            api_key_scope=scope,
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0",
            request_id="req-123",
        )
        assert ctx.user_email == "user@example.com"
        assert ctx.org_id == "org-1"
        assert ctx.workspace_id == "ws-1"
        assert ctx.roles == {"editor", "viewer"}
        assert ctx.permissions == {"debates.read", "debates.create"}
        assert ctx.api_key_scope == scope
        assert ctx.ip_address == "192.168.1.1"
        assert ctx.request_id == "req-123"

    def test_has_permission_exact_match(self):
        """has_permission with exact match."""
        ctx = AuthorizationContext(
            user_id="user-1",
            permissions={"debates.read", "debates.create"},
        )
        assert ctx.has_permission("debates.read") is True
        assert ctx.has_permission("debates.create") is True
        assert ctx.has_permission("debates.delete") is False

    def test_has_permission_wildcard(self):
        """has_permission with resource wildcard."""
        ctx = AuthorizationContext(
            user_id="user-1",
            permissions={"debates.*"},
        )
        assert ctx.has_permission("debates.read") is True
        assert ctx.has_permission("debates.create") is True
        assert ctx.has_permission("debates.delete") is True
        assert ctx.has_permission("agents.read") is False

    def test_has_permission_super_wildcard(self):
        """has_permission with super wildcard."""
        ctx = AuthorizationContext(
            user_id="user-1",
            permissions={"*"},
        )
        assert ctx.has_permission("debates.read") is True
        assert ctx.has_permission("agents.delete") is True
        assert ctx.has_permission("admin.system_config") is True

    def test_has_permission_colon_format(self):
        """has_permission handles colon format."""
        ctx = AuthorizationContext(
            user_id="user-1",
            permissions={"debates:read"},
        )
        assert ctx.has_permission("debates.read") is True
        assert ctx.has_permission("debates:read") is True

    def test_has_permission_dot_to_colon(self):
        """has_permission converts dot format to colon."""
        ctx = AuthorizationContext(
            user_id="user-1",
            permissions={"debates.read"},
        )
        assert ctx.has_permission("debates:read") is True

    def test_has_permission_wildcard_colon(self):
        """has_permission with colon wildcard."""
        ctx = AuthorizationContext(
            user_id="user-1",
            permissions={"debates:*"},
        )
        assert ctx.has_permission("debates.read") is True
        assert ctx.has_permission("debates:create") is True

    def test_has_permission_respects_api_key_scope(self):
        """has_permission respects API key scope restrictions."""
        scope = APIKeyScope(permissions={"debates.read"})
        ctx = AuthorizationContext(
            user_id="user-1",
            permissions={"debates.*"},  # Has wildcard
            api_key_scope=scope,  # But scope limits to read only
        )
        assert ctx.has_permission("debates.read") is True
        assert ctx.has_permission("debates.create") is False

    def test_has_permission_empty_permissions(self):
        """has_permission with no permissions."""
        ctx = AuthorizationContext(user_id="user-1")
        assert ctx.has_permission("debates.read") is False

    def test_has_role(self):
        """has_role checks for specific role."""
        ctx = AuthorizationContext(
            user_id="user-1",
            roles={"editor", "viewer"},
        )
        assert ctx.has_role("editor") is True
        assert ctx.has_role("viewer") is True
        assert ctx.has_role("admin") is False

    def test_has_any_role(self):
        """has_any_role checks for any of the specified roles."""
        ctx = AuthorizationContext(
            user_id="user-1",
            roles={"viewer"},
        )
        assert ctx.has_any_role("admin", "viewer") is True
        assert ctx.has_any_role("admin", "owner") is False

    def test_has_any_role_empty_roles(self):
        """has_any_role with no roles returns False."""
        ctx = AuthorizationContext(user_id="user-1")
        assert ctx.has_any_role("admin", "viewer") is False

    def test_has_any_role_empty_check(self):
        """has_any_role with no roles to check."""
        ctx = AuthorizationContext(
            user_id="user-1",
            roles={"viewer"},
        )
        assert ctx.has_any_role() is False


# =============================================================================
# Test AuthorizationDecision Dataclass
# =============================================================================


class TestAuthorizationDecision:
    """Tests for AuthorizationDecision dataclass."""

    def test_create_decision_allowed(self):
        """Create an allowed decision."""
        decision = AuthorizationDecision(
            allowed=True,
            reason="Permission granted",
            permission_key="debates.read",
        )
        assert decision.allowed is True
        assert decision.reason == "Permission granted"
        assert decision.permission_key == "debates.read"
        assert decision.resource_id is None
        assert decision.context is None
        assert decision.cached is False

    def test_create_decision_denied(self):
        """Create a denied decision."""
        decision = AuthorizationDecision(
            allowed=False,
            reason="Permission not granted",
            permission_key="debates.delete",
        )
        assert decision.allowed is False
        assert decision.reason == "Permission not granted"

    def test_decision_with_resource_id(self):
        """Decision with resource ID."""
        decision = AuthorizationDecision(
            allowed=True,
            reason="Granted",
            permission_key="debates.read",
            resource_id="debate-123",
        )
        assert decision.resource_id == "debate-123"

    def test_decision_with_context(self):
        """Decision with authorization context."""
        ctx = AuthorizationContext(
            user_id="user-1",
            org_id="org-1",
            request_id="req-123",
        )
        decision = AuthorizationDecision(
            allowed=True,
            reason="Granted",
            permission_key="debates.read",
            context=ctx,
        )
        assert decision.context == ctx

    def test_decision_cached_flag(self):
        """Decision with cached flag."""
        decision = AuthorizationDecision(
            allowed=True,
            reason="Cached grant",
            permission_key="debates.read",
            cached=True,
        )
        assert decision.cached is True

    def test_decision_to_dict_minimal(self):
        """to_dict with minimal decision."""
        decision = AuthorizationDecision(
            allowed=True,
            reason="Granted",
            permission_key="debates.read",
        )
        result = decision.to_dict()
        assert result["allowed"] is True
        assert result["reason"] == "Granted"
        assert result["permission_key"] == "debates.read"
        assert result["resource_id"] is None
        assert result["user_id"] is None
        assert result["org_id"] is None
        assert result["request_id"] is None
        assert result["cached"] is False
        assert "checked_at" in result

    def test_decision_to_dict_with_context(self):
        """to_dict includes context information."""
        ctx = AuthorizationContext(
            user_id="user-1",
            org_id="org-1",
            request_id="req-123",
        )
        decision = AuthorizationDecision(
            allowed=True,
            reason="Granted",
            permission_key="debates.read",
            resource_id="debate-456",
            context=ctx,
            cached=True,
        )
        result = decision.to_dict()
        assert result["user_id"] == "user-1"
        assert result["org_id"] == "org-1"
        assert result["request_id"] == "req-123"
        assert result["resource_id"] == "debate-456"
        assert result["cached"] is True

    def test_decision_checked_at_timestamp(self):
        """Decision has checked_at timestamp."""
        before = datetime.now(timezone.utc)
        decision = AuthorizationDecision(
            allowed=True,
            reason="Granted",
            permission_key="debates.read",
        )
        after = datetime.now(timezone.utc)
        assert before <= decision.checked_at <= after


# =============================================================================
# Test Helper Functions
# =============================================================================


class TestPermissionCandidates:
    """Tests for _permission_candidates helper function."""

    def test_dot_format(self):
        """Dot format generates both formats."""
        candidates = _permission_candidates("debates.read")
        assert "debates.read" in candidates
        assert "debates:read" in candidates

    def test_colon_format(self):
        """Colon format generates both formats."""
        candidates = _permission_candidates("debates:read")
        assert "debates:read" in candidates
        assert "debates.read" in candidates

    def test_no_separator(self):
        """Key without separator returns just itself."""
        candidates = _permission_candidates("debates")
        assert candidates == {"debates"}

    def test_wildcard(self):
        """Wildcard permission generates candidates."""
        candidates = _permission_candidates("debates.*")
        assert "debates.*" in candidates
        assert "debates:*" in candidates


class TestResourceCandidates:
    """Tests for _resource_candidates helper function."""

    def test_dot_format(self):
        """Dot format extracts resource."""
        resources = _resource_candidates("debates.read")
        assert "debates" in resources

    def test_colon_format(self):
        """Colon format extracts resource."""
        resources = _resource_candidates("debates:read")
        assert "debates" in resources

    def test_both_formats(self):
        """Both formats produce same resource."""
        dot_resources = _resource_candidates("debates.read")
        colon_resources = _resource_candidates("debates:read")
        assert dot_resources == colon_resources == {"debates"}

    def test_complex_permission(self):
        """Complex permission with multiple dots."""
        resources = _resource_candidates("control_plane.agents.read")
        assert "control_plane" in resources


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestEdgeCases:
    """Edge case tests for RBAC models."""

    def test_permission_matches_different_resource_types(self):
        """Permission does not match different resource types."""
        perm = Permission(
            id="perm-1",
            name="Read Debates",
            resource=ResourceType.DEBATE,
            action=Action.READ,
        )
        # Try all other resource types
        for rt in ResourceType:
            if rt != ResourceType.DEBATE:
                assert perm.matches(rt, Action.READ) is False

    def test_role_with_many_permissions(self):
        """Role can have many permissions."""
        perms = {f"resource{i}.action{j}" for i in range(10) for j in range(5)}
        role = Role(id="role-1", name="super", permissions=perms)
        assert len(role.permissions) == 50
        assert role.has_permission("resource5.action3") is True

    def test_assignment_expiry_boundary(self):
        """Assignment expiry at exact boundary."""
        # Create assignment that expires "now"
        now = datetime.now(timezone.utc)
        assignment = RoleAssignment(
            id="assign-1",
            user_id="user-1",
            role_id="editor",
            expires_at=now,
        )
        # Due to timing, might be expired or just at boundary
        # The key is it should not raise an error
        _ = assignment.is_expired
        _ = assignment.is_valid

    def test_api_key_scope_multiple_wildcards(self):
        """APIKeyScope with multiple resource wildcards."""
        scope = APIKeyScope(permissions={"debates.*", "agents.*", "admin.*"})
        assert scope.allows_permission("debates.read") is True
        assert scope.allows_permission("agents.deploy") is True
        assert scope.allows_permission("admin.system_config") is True
        assert scope.allows_permission("users.read") is False

    def test_context_with_mixed_permission_formats(self):
        """Context with both dot and colon permission formats."""
        ctx = AuthorizationContext(
            user_id="user-1",
            permissions={"debates.read", "agents:create", "memory.*"},
        )
        assert ctx.has_permission("debates.read") is True
        assert ctx.has_permission("debates:read") is True
        assert ctx.has_permission("agents.create") is True
        assert ctx.has_permission("agents:create") is True
        assert ctx.has_permission("memory.read") is True
        assert ctx.has_permission("memory:delete") is True

    def test_role_priority_ordering(self):
        """Roles can be sorted by priority."""
        roles = [
            Role(id="r1", name="viewer", priority=10),
            Role(id="r2", name="admin", priority=100),
            Role(id="r3", name="editor", priority=50),
        ]
        sorted_roles = sorted(roles, key=lambda r: r.priority, reverse=True)
        assert sorted_roles[0].name == "admin"
        assert sorted_roles[1].name == "editor"
        assert sorted_roles[2].name == "viewer"

    def test_decision_timestamp_is_utc(self):
        """Decision timestamp should be in UTC."""
        decision = AuthorizationDecision(
            allowed=True,
            reason="Granted",
            permission_key="debates.read",
        )
        # checked_at should be close to current UTC time
        now = datetime.now(timezone.utc)
        diff = abs((now - decision.checked_at).total_seconds())
        assert diff < 1  # Within 1 second

    def test_assignment_with_conditions(self):
        """Assignment can have conditions."""
        assignment = RoleAssignment(
            id="assign-1",
            user_id="user-1",
            role_id="editor",
            conditions={
                "ip_range": "10.0.0.0/8",
                "time_window": {"start": "09:00", "end": "17:00"},
            },
        )
        assert assignment.conditions["ip_range"] == "10.0.0.0/8"
        assert assignment.conditions["time_window"]["start"] == "09:00"

    def test_context_timestamp_auto_generated(self):
        """Context timestamp is auto-generated."""
        before = datetime.now(timezone.utc)
        ctx = AuthorizationContext(user_id="user-1")
        after = datetime.now(timezone.utc)
        assert before <= ctx.timestamp <= after

    def test_empty_role_name_display_name(self):
        """Empty role name generates empty display name."""
        role = Role(id="role-1", name="")
        assert role.display_name == ""

    def test_permission_from_key_default_name_format(self):
        """Permission.from_key generates readable default name."""
        perm = Permission.from_key("debates.create")
        assert perm.name == "Debates Create"

        perm2 = Permission.from_key("admin.system_config")
        assert perm2.name == "Admin System_Config"  # Note: underscore preserved


# =============================================================================
# Test Role Hierarchy
# =============================================================================


class TestRoleHierarchy:
    """Tests for role hierarchy support."""

    def test_role_with_parent_roles(self):
        """Role can specify parent roles."""
        role = Role(
            id="role-1",
            name="admin",
            parent_roles=["editor", "viewer"],
        )
        assert "editor" in role.parent_roles
        assert "viewer" in role.parent_roles

    def test_role_without_parent_roles(self):
        """Role without parent roles has empty list."""
        role = Role(id="role-1", name="viewer")
        assert role.parent_roles == []

    def test_custom_role_org_id(self):
        """Custom role has org_id."""
        role = Role(
            id="org-1:custom-role",
            name="custom-role",
            is_system=False,
            is_custom=True,
            org_id="org-1",
        )
        assert role.org_id == "org-1"
        assert role.is_custom is True
        assert role.is_system is False

    def test_system_role_no_org_id(self):
        """System role has no org_id."""
        role = Role(
            id="role-admin",
            name="admin",
            is_system=True,
            is_custom=False,
        )
        assert role.org_id is None
        assert role.is_system is True


# =============================================================================
# Test Integration Scenarios
# =============================================================================


class TestIntegrationScenarios:
    """Integration tests for common RBAC scenarios."""

    def test_api_key_with_limited_scope_and_context(self):
        """API key limits permissions even when context has more."""
        scope = APIKeyScope(permissions={"debates.read"})
        ctx = AuthorizationContext(
            user_id="api-user",
            permissions={"debates.*", "agents.*"},  # Wide permissions
            api_key_scope=scope,  # Limited scope
        )
        # Only debates.read should be allowed
        assert ctx.has_permission("debates.read") is True
        assert ctx.has_permission("debates.create") is False
        assert ctx.has_permission("agents.read") is False

    def test_expired_assignment_not_valid(self):
        """Expired role assignment is not valid."""
        assignment = RoleAssignment(
            id="assign-1",
            user_id="user-1",
            role_id="editor",
            is_active=True,
            expires_at=datetime.now(timezone.utc) - timedelta(days=1),
        )
        assert assignment.is_expired is True
        assert assignment.is_valid is False

    def test_inactive_assignment_not_valid(self):
        """Inactive role assignment is not valid even if not expired."""
        assignment = RoleAssignment(
            id="assign-1",
            user_id="user-1",
            role_id="editor",
            is_active=False,
            expires_at=datetime.now(timezone.utc) + timedelta(days=30),
        )
        assert assignment.is_expired is False
        assert assignment.is_valid is False

    def test_context_role_and_permission_checks(self):
        """Context supports both role and permission checks."""
        ctx = AuthorizationContext(
            user_id="user-1",
            roles={"editor"},
            permissions={"debates.create", "debates.read"},
        )
        # Role check
        assert ctx.has_role("editor") is True
        assert ctx.has_role("admin") is False
        assert ctx.has_any_role("admin", "editor") is True
        # Permission check
        assert ctx.has_permission("debates.create") is True
        assert ctx.has_permission("debates.delete") is False

    def test_decision_audit_trail(self):
        """Decision to_dict provides audit trail information."""
        ctx = AuthorizationContext(
            user_id="user-123",
            org_id="org-456",
            request_id="req-789",
        )
        decision = AuthorizationDecision(
            allowed=False,
            reason="Permission not granted: debates.delete",
            permission_key="debates.delete",
            resource_id="debate-abc",
            context=ctx,
        )
        audit = decision.to_dict()

        # All audit fields should be present
        assert audit["allowed"] is False
        assert audit["permission_key"] == "debates.delete"
        assert audit["resource_id"] == "debate-abc"
        assert audit["user_id"] == "user-123"
        assert audit["org_id"] == "org-456"
        assert audit["request_id"] == "req-789"
        assert "checked_at" in audit
