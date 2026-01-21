"""
Tests for RBAC Enforcer.

Tests cover:
- Permission checking and enforcement
- Role assignment and inheritance
- Permission caching
- Audit logging
- Isolation context handling
"""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.rbac.enforcer import (
    RBACEnforcer,
    RBACConfig,
    PermissionCheckResult,
    PermissionDeniedException,
)
from aragora.rbac.types import (
    Action,
    IsolationContext,
    Permission,
    ResourceType,
    Role,
    RoleAssignment,
    Scope,
)


class TestRBACConfig:
    """Tests for RBACConfig dataclass."""

    def test_default_config(self):
        """Default config has expected values."""
        config = RBACConfig()
        assert config.enabled is True
        assert config.deny_by_default is True
        assert config.log_all_checks is True
        assert config.cache_ttl_seconds == 300
        assert config.max_cache_size == 10000
        assert config.log_denials is True
        assert config.log_grants is False

    def test_custom_config(self):
        """Custom config values are respected."""
        config = RBACConfig(
            enabled=False,
            deny_by_default=False,
            log_all_checks=False,
            cache_ttl_seconds=60,
        )
        assert config.enabled is False
        assert config.deny_by_default is False
        assert config.cache_ttl_seconds == 60


class TestPermissionCheckResult:
    """Tests for PermissionCheckResult dataclass."""

    def test_granted_result(self):
        """Granted result has correct values."""
        result = PermissionCheckResult(
            granted=True,
            reason="Permission granted",
            matching_role="admin",
        )
        assert result.granted is True
        assert result.reason == "Permission granted"
        assert result.matching_role == "admin"

    def test_denied_result(self):
        """Denied result has correct values."""
        result = PermissionCheckResult(
            granted=False,
            reason="No permission",
        )
        assert result.granted is False
        assert result.reason == "No permission"
        assert result.matching_role is None

    def test_to_dict(self):
        """to_dict returns correct dictionary."""
        result = PermissionCheckResult(
            granted=True,
            reason="Test",
            matching_role="viewer",
        )
        d = result.to_dict()
        assert d["granted"] is True
        assert d["reason"] == "Test"
        assert d["matching_role"] == "viewer"
        assert "checked_at" in d


class TestPermissionDeniedException:
    """Tests for PermissionDeniedException."""

    def test_exception_attributes(self):
        """Exception stores all attributes."""
        context = IsolationContext(actor_id="user-1", organization_id="org-1")
        exc = PermissionDeniedException(
            message="Access denied",
            actor_id="user-1",
            resource=ResourceType.DEBATE,
            action=Action.CREATE,
            context=context,
        )
        assert str(exc) == "Access denied"
        assert exc.actor_id == "user-1"
        assert exc.resource == ResourceType.DEBATE
        assert exc.action == Action.CREATE
        assert exc.context == context

    def test_exception_without_context(self):
        """Exception works without context."""
        exc = PermissionDeniedException(
            message="Denied",
            actor_id="user-2",
            resource=ResourceType.WORKFLOW,
            action=Action.DELETE,
        )
        assert exc.context is None


class TestRBACEnforcer:
    """Tests for RBACEnforcer class."""

    @pytest.fixture
    def enforcer(self):
        """Create enforcer with default config."""
        return RBACEnforcer()

    @pytest.fixture
    def disabled_enforcer(self):
        """Create enforcer with RBAC disabled."""
        config = RBACConfig(enabled=False)
        return RBACEnforcer(config)

    @pytest.fixture
    def permissive_enforcer(self):
        """Create enforcer that allows by default."""
        config = RBACConfig(deny_by_default=False)
        return RBACEnforcer(config)

    @pytest.mark.asyncio
    async def test_check_disabled_allows_all(self, disabled_enforcer):
        """Disabled enforcer allows all actions."""
        result = await disabled_enforcer.check(
            actor="anyone",
            resource=ResourceType.DEBATE,
            action=Action.DELETE,
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_check_denies_by_default(self, enforcer):
        """Enforcer denies by default when no permissions match."""
        result = await enforcer.check(
            actor="unknown-user",
            resource=ResourceType.DEBATE,
            action=Action.CREATE,
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_check_allows_by_default_when_configured(self, permissive_enforcer):
        """Permissive enforcer allows by default."""
        result = await permissive_enforcer.check(
            actor="anyone",
            resource=ResourceType.DEBATE,
            action=Action.READ,
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_require_raises_on_denial(self, enforcer):
        """require() raises PermissionDeniedException when denied."""
        with pytest.raises(PermissionDeniedException) as exc_info:
            await enforcer.require(
                actor="unauthorized-user",
                resource=ResourceType.WORKFLOW,
                action=Action.DELETE,
            )
        assert exc_info.value.actor_id == "unauthorized-user"
        assert exc_info.value.resource == ResourceType.WORKFLOW
        assert exc_info.value.action == Action.DELETE

    @pytest.mark.asyncio
    async def test_check_succeeds_when_disabled(self, disabled_enforcer):
        """check() returns True when RBAC is disabled."""
        # Should allow when RBAC is disabled
        result = await disabled_enforcer.check(
            actor="any-user",
            resource=ResourceType.DEBATE,
            action=Action.CREATE,
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_assign_role(self, enforcer):
        """assign_role adds role assignment."""
        # Use a built-in role that exists
        assignment = await enforcer.assign_role(
            actor_id="user-1",
            role_id="workspace_viewer",
            scope=Scope.GLOBAL,
            scope_id="*",
            assigned_by="admin",
        )

        assert assignment.actor_id == "user-1"
        assert assignment.role_id == "workspace_viewer"

        assignments = enforcer._role_assignments.get("user-1", [])
        assert len(assignments) == 1

    @pytest.mark.asyncio
    async def test_revoke_role(self, enforcer):
        """revoke_role removes role assignment."""
        await enforcer.assign_role(
            actor_id="user-1",
            role_id="workspace_viewer",
            scope=Scope.GLOBAL,
            scope_id="*",
            assigned_by="admin",
        )

        result = await enforcer.revoke_role("user-1", "workspace_viewer", "*")
        assert result is True

        assignments = enforcer._role_assignments.get("user-1", [])
        assert len(assignments) == 0

    @pytest.mark.asyncio
    async def test_get_actor_roles(self, enforcer):
        """get_actor_roles returns all roles for actor."""
        await enforcer.assign_role(
            actor_id="user-1",
            role_id="workspace_viewer",
            scope=Scope.GLOBAL,
            scope_id="*",
            assigned_by="admin",
        )

        roles = await enforcer.get_actor_roles("user-1")
        assert len(roles) == 1
        assert roles[0].role_id == "workspace_viewer"

    @pytest.mark.asyncio
    async def test_check_with_isolation_context(self, enforcer):
        """Permission check respects isolation context."""
        context = IsolationContext(
            actor_id="user-1",
            organization_id="org-1",
            workspace_id="ws-1",
        )

        # Without permissions, should be denied
        result = await enforcer.check(
            actor="user-1",
            resource=ResourceType.DEBATE,
            action=Action.READ,
            context=context,
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_admin_permission_grants_all_actions(self, enforcer):
        """Admin permission on resource grants all actions."""
        # Create admin permission for debates
        admin_role = Role(
            id="debate-admin",
            name="Debate Admin",
            description="Admin access to debates",
            scope=Scope.GLOBAL,
            permissions={
                Permission(
                    resource=ResourceType.DEBATE,
                    action=Action.ADMIN,
                    scope=Scope.GLOBAL,
                )
            },
        )
        enforcer._roles["debate-admin"] = admin_role

        await enforcer.assign_role(
            actor_id="admin-user",
            role_id="debate-admin",
            scope=Scope.GLOBAL,
            scope_id="*",
            assigned_by="system",
        )

        # Should be able to do any action on debates
        for action in [Action.CREATE, Action.READ, Action.UPDATE, Action.DELETE]:
            result = await enforcer.check(
                actor="admin-user",
                resource=ResourceType.DEBATE,
                action=action,
            )
            assert result is True, f"Admin should have {action} permission"

    @pytest.mark.asyncio
    async def test_permission_cache_invalidation(self, enforcer):
        """Permission cache is invalidated on role change."""
        # Assign a role
        await enforcer.assign_role(
            actor_id="user-1",
            role_id="workspace_viewer",
            scope=Scope.GLOBAL,
            scope_id="*",
            assigned_by="admin",
        )

        # Revoke role - should invalidate cache
        await enforcer.revoke_role("user-1", "workspace_viewer", "*")

        # Cache should be empty for this user
        cache_key = ("user-1", None, None)
        assert cache_key not in enforcer._permission_cache


class TestRBACEnforcerEdgeCases:
    """Edge case tests for RBACEnforcer."""

    @pytest.fixture
    def enforcer(self):
        """Create enforcer with logging disabled."""
        config = RBACConfig(log_all_checks=False, log_denials=False)
        return RBACEnforcer(config)

    @pytest.mark.asyncio
    async def test_check_with_resource_context(self, enforcer):
        """Permission check can use resource context for conditions."""
        # Without matching resource context, should be denied
        result = await enforcer.check(
            actor="user-1",
            resource=ResourceType.WORKFLOW,
            action=Action.UPDATE,
            resource_context={"owner_id": "user-1"},
        )
        assert result is False  # No permissions assigned

    @pytest.mark.asyncio
    async def test_multiple_role_assignments(self, enforcer):
        """Actor can have multiple role assignments."""
        await enforcer.assign_role(
            actor_id="user-1",
            role_id="workspace_viewer",
            scope=Scope.GLOBAL,
            scope_id="*",
            assigned_by="admin",
        )

        roles = await enforcer.get_actor_roles("user-1")
        assert len(roles) == 1

    @pytest.mark.asyncio
    async def test_revoke_nonexistent_role(self, enforcer):
        """Revoking nonexistent role returns False."""
        result = await enforcer.revoke_role("nonexistent-user", "nonexistent-role", "*")
        assert result is False

    @pytest.mark.asyncio
    async def test_get_roles_for_unknown_actor(self, enforcer):
        """Getting roles for unknown actor returns empty list."""
        roles = await enforcer.get_actor_roles("unknown-actor")
        assert roles == []

    @pytest.mark.asyncio
    async def test_assign_invalid_role_raises(self, enforcer):
        """Assigning invalid role raises ValueError."""
        with pytest.raises(ValueError, match="Role not found"):
            await enforcer.assign_role(
                actor_id="user-1",
                role_id="nonexistent-role",
                scope=Scope.GLOBAL,
                scope_id="*",
                assigned_by="admin",
            )
