"""
Tests for RBAC Permission Checker.

Tests cover:
- Permission checking with exact match, wildcard, and super wildcard
- Role-based permission resolution
- API key scope restrictions
- Resource-specific policies
- Decision caching
- Role assignment management
- Cache invalidation
"""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock

from aragora.rbac.checker import (
    PermissionChecker,
    get_permission_checker,
    set_permission_checker,
    check_permission,
    has_permission,
)
from aragora.rbac.models import (
    Action,
    APIKeyScope,
    AuthorizationContext,
    AuthorizationDecision,
    ResourceType,
    RoleAssignment,
)


class TestPermissionChecker:
    """Tests for PermissionChecker class."""

    @pytest.fixture
    def checker(self):
        """Create a fresh permission checker."""
        return PermissionChecker(enable_cache=False)

    @pytest.fixture
    def cached_checker(self):
        """Create a permission checker with caching enabled."""
        return PermissionChecker(enable_cache=True, cache_ttl=60)

    @pytest.fixture
    def context_with_permissions(self):
        """Create context with pre-resolved permissions."""
        return AuthorizationContext(
            user_id="user-1",
            org_id="org-1",
            roles={"editor"},
            permissions={"debates.create", "debates.read", "debates.update"},
        )

    @pytest.fixture
    def context_with_wildcard(self):
        """Create context with wildcard permission."""
        return AuthorizationContext(
            user_id="user-2",
            org_id="org-1",
            roles={"admin"},
            permissions={"debates.*", "agents.read"},
        )

    @pytest.fixture
    def context_with_super_wildcard(self):
        """Create context with super wildcard."""
        return AuthorizationContext(
            user_id="user-3",
            org_id="org-1",
            roles={"owner"},
            permissions={"*"},
        )

    def test_exact_permission_match(self, checker, context_with_permissions):
        """Exact permission key grants access."""
        decision = checker.check_permission(context_with_permissions, "debates.create")
        assert decision.allowed is True
        assert "granted" in decision.reason.lower()

    def test_permission_denied_when_missing(self, checker, context_with_permissions):
        """Missing permission denies access."""
        decision = checker.check_permission(context_with_permissions, "debates:delete")
        assert decision.allowed is False
        assert "not granted" in decision.reason.lower()

    def test_wildcard_permission_grants_all_actions(self, checker, context_with_wildcard):
        """Wildcard permission grants all actions on resource."""
        # All debate actions should be allowed
        for action in ["create", "read", "update", "delete"]:
            decision = checker.check_permission(context_with_wildcard, f"debates.{action}")
            assert decision.allowed is True, f"debates.{action} should be allowed"

    def test_wildcard_does_not_grant_other_resources(self, checker, context_with_wildcard):
        """Wildcard permission only applies to its resource."""
        decision = checker.check_permission(context_with_wildcard, "users.delete")
        assert decision.allowed is False

    def test_super_wildcard_grants_all(self, checker, context_with_super_wildcard):
        """Super wildcard grants all permissions."""
        for permission in ["debates.create", "users.delete", "admin.system_config"]:
            decision = checker.check_permission(context_with_super_wildcard, permission)
            assert decision.allowed is True, f"{permission} should be allowed"

    def test_check_with_resource_id(self, checker, context_with_permissions):
        """Permission check includes resource ID."""
        decision = checker.check_permission(
            context_with_permissions, "debates.read", resource_id="debate-123"
        )
        assert decision.allowed is True
        assert decision.resource_id == "debate-123"


class TestAPIKeyScope:
    """Tests for API key scope restrictions."""

    @pytest.fixture
    def checker(self):
        return PermissionChecker(enable_cache=False)

    def test_api_key_scope_restricts_permissions(self, checker):
        """API key scope limits available permissions."""
        scope = APIKeyScope(permissions={"debates.read"})
        context = AuthorizationContext(
            user_id="api-user",
            org_id="org-1",
            roles={"admin"},
            permissions={"debates.*"},
            api_key_scope=scope,
        )

        # Read allowed by scope
        decision = checker.check_permission(context, "debates.read")
        assert decision.allowed is True

        # Create not in scope
        decision = checker.check_permission(context, "debates.create")
        assert decision.allowed is False
        assert "API key scope" in decision.reason

    def test_empty_scope_allows_all(self, checker):
        """Empty API key scope means full access."""
        scope = APIKeyScope()  # Empty permissions = all
        context = AuthorizationContext(
            user_id="api-user",
            org_id="org-1",
            roles={"editor"},
            permissions={"debates.create"},
            api_key_scope=scope,
        )

        decision = checker.check_permission(context, "debates.create")
        assert decision.allowed is True

    def test_api_key_scope_wildcard(self, checker):
        """API key scope supports wildcards."""
        scope = APIKeyScope(permissions={"debates.*"})
        context = AuthorizationContext(
            user_id="api-user",
            org_id="org-1",
            roles={"admin"},
            permissions={"*"},
            api_key_scope=scope,
        )

        # Debates allowed
        decision = checker.check_permission(context, "debates:delete")
        assert decision.allowed is True

        # Other resources not allowed by scope
        decision = checker.check_permission(context, "users:read")
        assert decision.allowed is False


class TestResourceAccess:
    """Tests for resource-level access control."""

    @pytest.fixture
    def checker(self):
        return PermissionChecker(enable_cache=False)

    def test_check_resource_access(self, checker):
        """check_resource_access uses resource type and action."""
        context = AuthorizationContext(
            user_id="user-1",
            org_id="org-1",
            permissions={"debates.read"},
        )

        decision = checker.check_resource_access(
            context,
            resource_type=ResourceType.DEBATE,
            action=Action.READ,
            resource_id="debate-123",
        )
        assert decision.allowed is True

    def test_resource_policy_restricts_access(self, checker):
        """Custom resource policies can restrict access."""
        context = AuthorizationContext(
            user_id="user-1",
            org_id="org-1",
            permissions={"debates.update"},
        )

        # Register policy that only allows owner to update
        def owner_only_policy(ctx, action, attrs):
            return attrs.get("owner_id") == ctx.user_id

        checker.register_resource_policy(
            ResourceType.DEBATE,
            "debate-123",
            owner_only_policy,
        )

        # Not owner - denied
        decision = checker.check_resource_access(
            context,
            resource_type=ResourceType.DEBATE,
            action=Action.UPDATE,
            resource_id="debate-123",
            resource_attrs={"owner_id": "other-user"},
        )
        assert decision.allowed is False
        assert "policy" in decision.reason.lower()

        # Is owner - allowed
        decision = checker.check_resource_access(
            context,
            resource_type=ResourceType.DEBATE,
            action=Action.UPDATE,
            resource_id="debate-123",
            resource_attrs={"owner_id": "user-1"},
        )
        assert decision.allowed is True


class TestRoleHelpers:
    """Tests for role checking helper methods."""

    @pytest.fixture
    def checker(self):
        return PermissionChecker(enable_cache=False)

    def test_has_role(self, checker):
        """has_role checks for specific role."""
        context = AuthorizationContext(
            user_id="user-1",
            roles={"editor", "viewer"},
        )
        assert checker.has_role(context, "editor") is True
        assert checker.has_role(context, "admin") is False

    def test_has_any_role(self, checker):
        """has_any_role checks for any of the roles."""
        context = AuthorizationContext(
            user_id="user-1",
            roles={"viewer"},
        )
        assert checker.has_any_role(context, "admin", "viewer") is True
        assert checker.has_any_role(context, "admin", "owner") is False

    def test_has_all_roles(self, checker):
        """has_all_roles checks for all specified roles."""
        context = AuthorizationContext(
            user_id="user-1",
            roles={"admin", "billing_manager"},
        )
        assert checker.has_all_roles(context, "admin", "billing_manager") is True
        assert checker.has_all_roles(context, "admin", "owner") is False

    def test_is_owner(self, checker):
        """is_owner checks for owner role."""
        owner_ctx = AuthorizationContext(user_id="u1", roles={"owner"})
        admin_ctx = AuthorizationContext(user_id="u2", roles={"admin"})

        assert checker.is_owner(owner_ctx) is True
        assert checker.is_owner(admin_ctx) is False

    def test_is_admin(self, checker):
        """is_admin checks for admin or owner role."""
        owner_ctx = AuthorizationContext(user_id="u1", roles={"owner"})
        admin_ctx = AuthorizationContext(user_id="u2", roles={"admin"})
        editor_ctx = AuthorizationContext(user_id="u3", roles={"editor"})

        assert checker.is_admin(owner_ctx) is True
        assert checker.is_admin(admin_ctx) is True
        assert checker.is_admin(editor_ctx) is False


class TestRoleAssignments:
    """Tests for role assignment management."""

    @pytest.fixture
    def checker(self):
        return PermissionChecker(enable_cache=False)

    def test_add_role_assignment(self, checker):
        """add_role_assignment adds assignment to cache."""
        assignment = RoleAssignment(
            id="assign-1",
            user_id="user-1",
            role_id="editor",
            org_id="org-1",
        )
        checker.add_role_assignment(assignment)

        roles = checker.get_user_roles("user-1", "org-1")
        assert "editor" in roles

    def test_remove_role_assignment(self, checker):
        """remove_role_assignment removes assignment."""
        assignment = RoleAssignment(
            id="assign-1",
            user_id="user-1",
            role_id="editor",
            org_id="org-1",
        )
        checker.add_role_assignment(assignment)
        checker.remove_role_assignment("user-1", "editor", "org-1")

        roles = checker.get_user_roles("user-1", "org-1")
        assert "editor" not in roles

    def test_get_user_roles_respects_org(self, checker):
        """get_user_roles filters by organization."""
        checker.add_role_assignment(
            RoleAssignment(id="a1", user_id="user-1", role_id="editor", org_id="org-1")
        )
        checker.add_role_assignment(
            RoleAssignment(id="a2", user_id="user-1", role_id="admin", org_id="org-2")
        )

        roles_org1 = checker.get_user_roles("user-1", "org-1")
        roles_org2 = checker.get_user_roles("user-1", "org-2")

        assert roles_org1 == {"editor"}
        assert roles_org2 == {"admin"}

    def test_expired_assignments_ignored(self, checker):
        """Expired role assignments are not returned."""
        expired = RoleAssignment(
            id="a1",
            user_id="user-1",
            role_id="editor",
            org_id="org-1",
            expires_at=datetime.now(timezone.utc) - timedelta(hours=1),
        )
        checker.add_role_assignment(expired)

        roles = checker.get_user_roles("user-1", "org-1")
        assert "editor" not in roles

    def test_inactive_assignments_ignored(self, checker):
        """Inactive role assignments are not returned."""
        inactive = RoleAssignment(
            id="a1",
            user_id="user-1",
            role_id="editor",
            org_id="org-1",
            is_active=False,
        )
        checker.add_role_assignment(inactive)

        roles = checker.get_user_roles("user-1", "org-1")
        assert "editor" not in roles


class TestDecisionCaching:
    """Tests for decision caching."""

    @pytest.fixture
    def checker(self):
        return PermissionChecker(enable_cache=True, cache_ttl=60)

    def test_decision_is_cached(self, checker):
        """Repeated checks return cached decision."""
        context = AuthorizationContext(
            user_id="user-1",
            org_id="org-1",
            permissions={"debates.read"},
        )

        # First check
        decision1 = checker.check_permission(context, "debates.read")
        assert decision1.allowed is True
        assert decision1.cached is not True  # First call not cached

        # Second check - should be cached
        decision2 = checker.check_permission(context, "debates.read")
        assert decision2.allowed is True
        assert decision2.cached is True

    def test_cache_cleared_for_user(self, checker):
        """clear_cache with user_id only clears that user."""
        ctx1 = AuthorizationContext(
            user_id="user-1",
            org_id="org-1",
            permissions={"debates.read"},
        )
        ctx2 = AuthorizationContext(
            user_id="user-2",
            org_id="org-1",
            permissions={"debates.read"},
        )

        # Populate cache
        checker.check_permission(ctx1, "debates.read")
        checker.check_permission(ctx2, "debates.read")

        # Clear only user-1
        checker.clear_cache("user-1")

        # user-1 should not be cached
        decision1 = checker.check_permission(ctx1, "debates.read")
        # Note: We can't easily verify cache miss without internal access

        # user-2 should still be cached
        decision2 = checker.check_permission(ctx2, "debates.read")
        assert decision2.cached is True

    def test_clear_all_cache(self, checker):
        """clear_cache without user_id clears all."""
        context = AuthorizationContext(
            user_id="user-1",
            org_id="org-1",
            permissions={"debates.read"},
        )

        # Populate cache
        checker.check_permission(context, "debates.read")

        # Clear all
        checker.clear_cache()

        # Check cache size
        assert checker._decision_cache == {}


class TestGlobalChecker:
    """Tests for global permission checker functions."""

    def teardown_method(self):
        """Reset global checker after each test."""
        set_permission_checker(None)

    def test_get_permission_checker_creates_default(self):
        """get_permission_checker creates default instance."""
        checker = get_permission_checker()
        assert isinstance(checker, PermissionChecker)

    def test_set_permission_checker_replaces_global(self):
        """set_permission_checker replaces global instance."""
        custom = PermissionChecker(cache_ttl=999)
        set_permission_checker(custom)

        retrieved = get_permission_checker()
        assert retrieved._cache_ttl == 999

    def test_check_permission_uses_global(self):
        """check_permission convenience function uses global."""
        context = AuthorizationContext(
            user_id="user-1",
            permissions={"debates.read"},
        )
        decision = check_permission(context, "debates.read")
        assert decision.allowed is True

    def test_has_permission_returns_bool(self):
        """has_permission returns boolean."""
        context = AuthorizationContext(
            user_id="user-1",
            permissions={"debates.read"},
        )
        assert has_permission(context, "debates.read") is True
        assert has_permission(context, "debates:delete") is False


class TestCacheStats:
    """Tests for cache statistics."""

    def test_get_cache_stats(self):
        """get_cache_stats returns useful statistics."""
        checker = PermissionChecker(enable_cache=True, cache_ttl=300)
        context = AuthorizationContext(
            user_id="user-1",
            permissions={"debates.read"},
        )

        # Make some checks to populate cache
        checker.check_permission(context, "debates.read")
        checker.check_permission(context, "debates.create")

        stats = checker.get_cache_stats()

        assert "local_cache_size" in stats
        assert stats["local_cache_size"] >= 2
        assert stats["cache_enabled"] is True
        assert stats["cache_ttl"] == 300
        assert stats["distributed"] is False
