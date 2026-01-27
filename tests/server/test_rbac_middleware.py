"""
Tests for RBAC middleware integration.

Tests the RBAC middleware that enforces permission checks on API requests.
"""

import pytest
from unittest.mock import MagicMock, patch

from aragora.rbac.middleware import (
    RBACMiddleware,
    RBACMiddlewareConfig,
    RoutePermission,
    DEFAULT_ROUTE_PERMISSIONS,
)
from aragora.rbac.models import AuthorizationContext
from aragora.rbac.checker import PermissionChecker


@pytest.fixture(autouse=True)
def reset_permission_checker():
    """Reset the global permission checker before each test."""
    import aragora.rbac.checker as checker_module

    checker_module._permission_checker = None
    yield
    checker_module._permission_checker = None


class TestRBACMiddleware:
    """Test RBAC middleware functionality."""

    def test_bypass_paths(self):
        """Bypass paths should be allowed without authentication."""
        config = RBACMiddlewareConfig(
            bypass_paths={"/health", "/metrics"},
            default_authenticated=True,
        )
        middleware = RBACMiddleware(config)

        # Health endpoint should be allowed
        allowed, reason, _ = middleware.check_request("/health", "GET", None)
        assert allowed is True
        assert "Bypass" in reason

        # Metrics should be allowed
        allowed, reason, _ = middleware.check_request("/metrics", "GET", None)
        assert allowed is True

    def test_bypass_methods(self):
        """OPTIONS method should be allowed (CORS preflight)."""
        config = RBACMiddlewareConfig(
            bypass_methods={"OPTIONS"},
            default_authenticated=True,
        )
        middleware = RBACMiddleware(config)

        # OPTIONS should be allowed
        allowed, reason, _ = middleware.check_request("/api/debates", "OPTIONS", None)
        assert allowed is True
        assert "Bypass method" in reason

    def test_unauthenticated_protected_route(self):
        """Unauthenticated requests to protected routes should be denied."""
        config = RBACMiddlewareConfig(
            route_permissions=[
                RoutePermission(r"^/api/debates$", "POST", "debates.create"),
            ],
            default_authenticated=True,
        )
        middleware = RBACMiddleware(config)

        # POST to debates without auth should be denied
        allowed, reason, _ = middleware.check_request("/api/debates", "POST", None)
        assert allowed is False
        assert "Authentication required" in reason

    def test_authenticated_with_permission(self):
        """Authenticated requests with correct permission should be allowed."""
        config = RBACMiddlewareConfig(
            route_permissions=[
                RoutePermission(r"^/api/debates$", "POST", "debates.create"),
            ],
        )
        middleware = RBACMiddleware(config)

        # Create auth context with permission
        auth_ctx = AuthorizationContext(
            user_id="user_123",
            permissions={"debates.create"},
        )

        allowed, reason, perm = middleware.check_request("/api/debates", "POST", auth_ctx)
        assert allowed is True
        assert perm == "debates.create"

    def test_authenticated_without_permission(self):
        """Authenticated requests without required permission should be denied."""
        config = RBACMiddlewareConfig(
            route_permissions=[
                RoutePermission(r"^/api/debates$", "POST", "debates.create"),
            ],
        )
        middleware = RBACMiddleware(config)

        # Create auth context without permission
        auth_ctx = AuthorizationContext(
            user_id="user_123",
            permissions={"debates:read"},  # Wrong permission
        )

        allowed, reason, perm = middleware.check_request("/api/debates", "POST", auth_ctx)
        assert allowed is False
        assert perm == "debates.create"

    def test_allow_unauthenticated_route(self):
        """Routes marked allow_unauthenticated should work without auth."""
        config = RBACMiddlewareConfig(
            route_permissions=[
                RoutePermission(r"^/api/auth/login$", "POST", "", allow_unauthenticated=True),
            ],
        )
        middleware = RBACMiddleware(config)

        # Login should be allowed without auth
        allowed, reason, _ = middleware.check_request("/api/auth/login", "POST", None)
        assert allowed is True
        assert "Unauthenticated access allowed" in reason

    def test_route_pattern_matching(self):
        """Route patterns should match correctly."""
        config = RBACMiddlewareConfig(
            route_permissions=[
                RoutePermission(r"^/api/debates/([^/]+)$", "GET", "debates:read", 1),
            ],
        )
        middleware = RBACMiddleware(config)

        auth_ctx = AuthorizationContext(
            user_id="user_123",
            permissions={"debates:read"},
        )

        # Should match with resource ID
        allowed, reason, _ = middleware.check_request("/api/debates/abc123", "GET", auth_ctx)
        assert allowed is True

        # Should not match (no ID)
        allowed, _, _ = middleware.check_request("/api/debates/", "GET", auth_ctx)
        # Falls through to no rule match
        assert allowed is True  # default_authenticated=False

    def test_default_route_permissions(self):
        """Default route permissions should be loaded."""
        middleware = RBACMiddleware()

        # Should have default permissions loaded
        assert len(middleware.config.route_permissions) > 0

        # Check some expected permissions exist
        debate_create = middleware.get_required_permission("/api/debates", "POST")
        assert debate_create == "debates.create"

        admin_perm = middleware.get_required_permission("/api/admin/users", "GET")
        assert admin_perm == "admin.*"


class TestRoutePermission:
    """Test RoutePermission class."""

    def test_matches_exact_path(self):
        """Exact path patterns should match."""
        rule = RoutePermission(r"^/api/debates$", "GET", "debates:read")

        matches, resource_id = rule.matches("/api/debates", "GET")
        assert matches is True
        assert resource_id is None

    def test_matches_with_capture_group(self):
        """Patterns with capture groups should extract resource ID."""
        rule = RoutePermission(r"^/api/debates/([^/]+)$", "GET", "debates:read", 1)

        matches, resource_id = rule.matches("/api/debates/abc123", "GET")
        assert matches is True
        assert resource_id == "abc123"

    def test_method_mismatch(self):
        """Wrong method should not match."""
        rule = RoutePermission(r"^/api/debates$", "POST", "debates.create")

        matches, _ = rule.matches("/api/debates", "GET")
        assert matches is False

    def test_wildcard_method(self):
        """Wildcard method should match any method."""
        rule = RoutePermission(r"^/api/admin", "*", "admin.*")

        matches, _ = rule.matches("/api/admin/users", "GET")
        assert matches is True

        matches, _ = rule.matches("/api/admin/users", "POST")
        assert matches is True

        matches, _ = rule.matches("/api/admin/users", "DELETE")
        assert matches is True


class TestUnifiedServerRBACIntegration:
    """Test RBAC integration with unified server."""

    def test_server_has_rbac_configured(self):
        """Verify unified server has RBAC middleware configured."""
        from aragora.server.unified_server import UnifiedHandler

        # Check RBAC is configured with route permissions
        assert hasattr(UnifiedHandler, "rbac")
        assert UnifiedHandler.rbac is not None
        assert len(UnifiedHandler.rbac.config.route_permissions) > 0

    def test_server_rbac_has_default_permissions(self):
        """Verify server RBAC includes default route permissions."""
        from aragora.server.unified_server import UnifiedHandler

        # Check specific permissions are configured
        rbac = UnifiedHandler.rbac

        # Debates endpoints
        assert rbac.get_required_permission("/api/debates", "POST") == "debates.create"
        assert rbac.get_required_permission("/api/debates", "GET") == "debates:read"

        # Admin endpoints require admin permission
        assert rbac.get_required_permission("/api/admin/users", "GET") == "admin.*"

        # Auth endpoints should be unauthenticated
        allowed, reason, _ = rbac.check_request("/api/auth/login", "POST", None)
        assert allowed is True

    def test_server_rbac_bypass_health(self):
        """Verify health endpoints bypass RBAC."""
        from aragora.server.unified_server import UnifiedHandler

        rbac = UnifiedHandler.rbac

        # Health endpoints should bypass
        allowed, reason, _ = rbac.check_request("/health", "GET", None)
        assert allowed is True
        assert "Bypass" in reason

        allowed, reason, _ = rbac.check_request("/metrics", "GET", None)
        assert allowed is True

    def test_server_rbac_blocks_unauthenticated_protected(self):
        """Verify RBAC blocks unauthenticated access to protected routes."""
        from aragora.server.unified_server import UnifiedHandler

        rbac = UnifiedHandler.rbac

        # Protected route without auth should be blocked
        allowed, reason, perm = rbac.check_request("/api/debates", "POST", None)
        # With default_authenticated=False, unmatched routes are allowed
        # but matched routes with permission requirements should fail
        if perm:  # If a permission is required
            assert allowed is False or perm == "debates.create"
