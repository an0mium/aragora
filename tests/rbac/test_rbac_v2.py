"""
Tests for the RBAC v2 module.

Tests the new fine-grained permission system including:
- Permission checking
- Role hierarchy
- Decorators
- Middleware
- Audit logging
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from aragora.rbac import (
    # Models
    AuthorizationContext,
    AuthorizationDecision,
    APIKeyScope,
    # Checker
    PermissionChecker,
    check_permission,
    has_permission,
    # Decorators
    require_permission,
    require_role,
    require_admin,
    require_owner,
    PermissionDeniedError,
    RoleRequiredError,
    # Middleware
    RBACMiddleware,
    RoutePermission,
    # Defaults
    SYSTEM_PERMISSIONS,
    SYSTEM_ROLES_V2,
    get_role_permissions,
    get_role,
    # Audit
    AuthorizationAuditor,
    AuditEventType,
)


class TestAuthorizationContext:
    """Tests for AuthorizationContext."""

    def test_has_permission_exact(self):
        """Test exact permission match."""
        context = AuthorizationContext(
            user_id="user1",
            permissions={"debates:create", "debates:read"},
        )
        assert context.has_permission("debates:create")
        assert context.has_permission("debates:read")
        assert not context.has_permission("debates:delete")

    def test_has_permission_wildcard(self):
        """Test wildcard permission matching."""
        context = AuthorizationContext(
            user_id="user1",
            permissions={"debates.*"},
        )
        assert context.has_permission("debates:create")
        assert context.has_permission("debates:delete")
        assert not context.has_permission("agents:create")

    def test_has_permission_super_wildcard(self):
        """Test super wildcard grants all."""
        context = AuthorizationContext(
            user_id="user1",
            permissions={"*"},
        )
        assert context.has_permission("debates:create")
        assert context.has_permission("agents:delete")
        assert context.has_permission("admin.system_config")

    def test_has_role(self):
        """Test role checking."""
        context = AuthorizationContext(
            user_id="user1",
            roles={"admin", "debate_creator"},
        )
        assert context.has_role("admin")
        assert context.has_role("debate_creator")
        assert not context.has_role("owner")

    def test_has_any_role(self):
        """Test any role checking."""
        context = AuthorizationContext(
            user_id="user1",
            roles={"member"},
        )
        assert context.has_any_role("admin", "member")
        assert not context.has_any_role("admin", "owner")


class TestAPIKeyScope:
    """Tests for APIKeyScope."""

    def test_allows_permission_empty_scope(self):
        """Empty permissions means full access."""
        scope = APIKeyScope()
        assert scope.allows_permission("debates:create")
        assert scope.allows_permission("anything.else")

    def test_allows_permission_limited(self):
        """Limited scope only allows specified permissions."""
        scope = APIKeyScope(permissions={"debates:read", "debates:create"})
        assert scope.allows_permission("debates:read")
        assert scope.allows_permission("debates:create")
        assert not scope.allows_permission("debates:delete")

    def test_allows_permission_wildcard(self):
        """Wildcard scope allows all under resource."""
        scope = APIKeyScope(permissions={"debates.*"})
        assert scope.allows_permission("debates:read")
        assert scope.allows_permission("debates:delete")
        assert not scope.allows_permission("agents:create")


class TestPermissionChecker:
    """Tests for PermissionChecker."""

    def test_check_permission_allowed(self):
        """Test permission check when allowed."""
        checker = PermissionChecker()
        context = AuthorizationContext(
            user_id="user1",
            roles={"admin"},
        )
        # Manually resolve permissions for test
        context = AuthorizationContext(
            user_id="user1",
            roles={"admin"},
            permissions=get_role_permissions("admin"),
        )

        decision = checker.check_permission(context, "debates:create")
        assert decision.allowed
        assert "granted" in decision.reason.lower()

    def test_check_permission_denied(self):
        """Test permission check when denied."""
        checker = PermissionChecker()
        context = AuthorizationContext(
            user_id="user1",
            roles={"viewer"},
            permissions=get_role_permissions("viewer"),
        )

        decision = checker.check_permission(context, "debates:create")
        assert not decision.allowed
        assert "not granted" in decision.reason.lower()

    def test_check_permission_with_api_key_scope(self):
        """Test permission check with API key scope limiting."""
        checker = PermissionChecker()
        context = AuthorizationContext(
            user_id="user1",
            roles={"admin"},
            permissions=get_role_permissions("admin"),
            api_key_scope=APIKeyScope(permissions={"debates:read"}),
        )

        # Admin has debates.create but API key scope doesn't
        decision = checker.check_permission(context, "debates:create")
        assert not decision.allowed
        assert "scope" in decision.reason.lower()

    def test_caching(self):
        """Test decision caching."""
        checker = PermissionChecker(enable_cache=True)
        context = AuthorizationContext(
            user_id="user1",
            roles={"admin"},
            permissions=get_role_permissions("admin"),
        )

        # First check
        decision1 = checker.check_permission(context, "debates:create")
        assert not decision1.cached

        # Second check should be cached
        decision2 = checker.check_permission(context, "debates:create")
        assert decision2.cached
        assert decision2.allowed == decision1.allowed


class TestRoleHierarchy:
    """Tests for role hierarchy and permission inheritance."""

    def test_owner_has_all_permissions(self):
        """Owner role should have all permissions."""
        permissions = get_role_permissions("owner")
        assert len(permissions) == len(SYSTEM_PERMISSIONS)

    def test_admin_inherits_from_debate_creator(self):
        """Admin should have all debate_creator permissions."""
        admin_perms = get_role_permissions("admin")
        creator_perms = get_role_permissions("debate_creator")

        # All creator permissions should be in admin
        for perm in creator_perms:
            if perm in SYSTEM_PERMISSIONS:
                assert perm in admin_perms, f"Admin missing: {perm}"

    def test_viewer_has_minimal_permissions(self):
        """Viewer should have minimal read-only permissions."""
        permissions = get_role_permissions("viewer")
        assert "debates:read" in permissions
        assert "debates:create" not in permissions
        assert "debates:delete" not in permissions


class TestDecorators:
    """Tests for permission decorators."""

    def test_require_permission_allowed(self):
        """Test require_permission decorator when allowed."""

        @require_permission("debates:read")
        def handler(context: AuthorizationContext):
            return "success"

        context = AuthorizationContext(
            user_id="user1",
            permissions={"debates:read"},
        )

        result = handler(context=context)
        assert result == "success"

    def test_require_permission_denied(self):
        """Test require_permission decorator when denied."""

        @require_permission("debates:delete")
        def handler(context: AuthorizationContext):
            return "success"

        context = AuthorizationContext(
            user_id="user1",
            permissions={"debates:read"},
        )

        with pytest.raises(PermissionDeniedError):
            handler(context=context)

    def test_require_role_single(self):
        """Test require_role with single role."""

        @require_role("admin")
        def handler(context: AuthorizationContext):
            return "success"

        # With admin role
        context = AuthorizationContext(user_id="user1", roles={"admin"})
        assert handler(context=context) == "success"

        # Without admin role
        context = AuthorizationContext(user_id="user1", roles={"viewer"})
        with pytest.raises(RoleRequiredError):
            handler(context=context)

    def test_require_role_any(self):
        """Test require_role with multiple roles (any)."""

        @require_role("admin", "owner")
        def handler(context: AuthorizationContext):
            return "success"

        # With admin
        context = AuthorizationContext(user_id="user1", roles={"admin"})
        assert handler(context=context) == "success"

        # With owner
        context = AuthorizationContext(user_id="user1", roles={"owner"})
        assert handler(context=context) == "success"

        # With neither
        context = AuthorizationContext(user_id="user1", roles={"member"})
        with pytest.raises(RoleRequiredError):
            handler(context=context)

    def test_require_admin(self):
        """Test require_admin shorthand."""

        @require_admin()
        def handler(context: AuthorizationContext):
            return "success"

        # With admin
        context = AuthorizationContext(user_id="user1", roles={"admin"})
        assert handler(context=context) == "success"

        # With owner (should also work)
        context = AuthorizationContext(user_id="user1", roles={"owner"})
        assert handler(context=context) == "success"

        # With member
        context = AuthorizationContext(user_id="user1", roles={"member"})
        with pytest.raises(RoleRequiredError):
            handler(context=context)


class TestAsyncDecorators:
    """Tests for async decorators."""

    @pytest.mark.asyncio
    async def test_require_permission_async(self):
        """Test require_permission on async function."""

        @require_permission("debates:create")
        async def handler(context: AuthorizationContext):
            return "success"

        context = AuthorizationContext(
            user_id="user1",
            permissions={"debates:create"},
        )

        result = await handler(context=context)
        assert result == "success"

    @pytest.mark.asyncio
    async def test_require_role_async(self):
        """Test require_role on async function."""

        @require_role("admin")
        async def handler(context: AuthorizationContext):
            return "success"

        context = AuthorizationContext(user_id="user1", roles={"admin"})
        result = await handler(context=context)
        assert result == "success"


class TestMiddleware:
    """Tests for RBAC middleware."""

    def test_route_permission_matching(self):
        """Test route permission pattern matching."""
        rule = RoutePermission(
            r"^/api/debates/([^/]+)$",
            "GET",
            "debates:read",
            resource_id_group=1,
        )

        matches, resource_id = rule.matches("/api/debates/123", "GET")
        assert matches
        assert resource_id == "123"

        matches, _ = rule.matches("/api/debates", "GET")
        assert not matches

        matches, _ = rule.matches("/api/debates/123", "POST")
        assert not matches

    def test_middleware_check_bypass_paths(self):
        """Test middleware bypasses health check paths."""
        middleware = RBACMiddleware()

        allowed, reason, _ = middleware.check_request("/health", "GET", None)
        assert allowed
        assert "Bypass" in reason

    def test_middleware_check_authenticated(self):
        """Test middleware requires auth for protected routes."""
        middleware = RBACMiddleware()

        allowed, reason, _ = middleware.check_request("/api/debates", "POST", None)
        assert not allowed
        assert "Authentication" in reason

    def test_middleware_check_permission(self):
        """Test middleware checks permission for routes."""
        middleware = RBACMiddleware()

        context = AuthorizationContext(
            user_id="user1",
            permissions={"debates:create"},
        )

        allowed, reason, perm = middleware.check_request("/api/debates", "POST", context)
        assert allowed
        assert perm == "debates:create"


class TestAudit:
    """Tests for authorization audit logging."""

    def test_audit_decision_logging(self):
        """Test audit logs authorization decisions."""
        events = []

        def capture_handler(event):
            events.append(event)

        auditor = AuthorizationAuditor(handlers=[capture_handler])

        context = AuthorizationContext(user_id="user1", org_id="org1")
        decision = AuthorizationDecision(
            allowed=True,
            reason="Permission granted",
            permission_key="debates:create",
            context=context,
        )

        auditor.log_decision(decision)

        assert len(events) >= 1
        event = events[-1]
        assert event.event_type == AuditEventType.PERMISSION_GRANTED
        assert event.user_id == "user1"
        assert event.permission_key == "debates:create"

    def test_audit_denied_logging(self):
        """Test audit logs denied decisions."""
        events = []

        def capture_handler(event):
            events.append(event)

        auditor = AuthorizationAuditor(handlers=[capture_handler])

        context = AuthorizationContext(user_id="user1")
        decision = AuthorizationDecision(
            allowed=False,
            reason="Permission denied",
            permission_key="admin.all",
            context=context,
        )

        auditor.log_decision(decision)

        assert len(events) >= 1
        event = events[-1]
        assert event.event_type == AuditEventType.PERMISSION_DENIED
        assert not event.decision


class TestSystemRoles:
    """Tests for system role definitions."""

    def test_all_system_roles_exist(self):
        """Verify all expected system roles exist."""
        expected = {
            "owner",
            "admin",
            "compliance_officer",
            "debate_creator",
            "team_lead",
            "analyst",
            "viewer",
            "member",
        }
        actual = set(SYSTEM_ROLES_V2.keys())
        assert expected == actual

    def test_role_priority_ordering(self):
        """Verify role priorities are correctly ordered."""
        owner = get_role("owner")
        admin = get_role("admin")
        member = get_role("member")
        viewer = get_role("viewer")

        assert owner.priority > admin.priority
        assert admin.priority > member.priority
        assert member.priority > viewer.priority

    def test_permissions_count(self):
        """Verify reasonable permission counts per role."""
        assert len(SYSTEM_PERMISSIONS) >= 50

        owner_perms = get_role_permissions("owner")
        viewer_perms = get_role_permissions("viewer")

        assert len(owner_perms) > len(viewer_perms)
        assert len(viewer_perms) >= 3  # At least read access to core resources
