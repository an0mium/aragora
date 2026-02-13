"""
Security tests for RBAC default-deny policy.

Verifies that the RBAC middleware denies access to routes that are not
explicitly listed in the route permission rules, preventing unauthorized
access to unlisted endpoints.

Finding 1: RBAC default-allow for unmatched routes.
"""

import pytest
from unittest.mock import Mock, patch

from aragora.rbac.middleware import (
    RBACMiddleware,
    RBACMiddlewareConfig,
    RoutePermission,
)
from aragora.rbac.models import AuthorizationContext


def _make_auth_context(
    user_id: str = "user-1",
    roles: list[str] | None = None,
) -> AuthorizationContext:
    """Create a minimal AuthorizationContext for testing."""
    ctx = Mock(spec=AuthorizationContext)
    ctx.user_id = user_id
    ctx.roles = roles or ["member"]
    ctx.tenant_id = "tenant-1"
    return ctx


class TestRBACDefaultDeny:
    """Verify that unmatched routes are denied by default."""

    def _make_middleware(
        self,
        rules: list[RoutePermission] | None = None,
        bypass_paths: set[str] | None = None,
    ) -> RBACMiddleware:
        """Create a middleware with a small set of rules for testing."""
        config = RBACMiddlewareConfig(
            route_permissions=rules
            or [
                RoutePermission(r"^/api/debates$", "GET", "debates.read"),
                RoutePermission(r"^/api/auth/login$", "POST", "", allow_unauthenticated=True),
            ],
            bypass_paths=bypass_paths or {"/health", "/healthz", "/ready"},
        )
        return RBACMiddleware(config=config, validate_permissions=False)

    # -----------------------------------------------------------------
    # Core default-deny tests
    # -----------------------------------------------------------------

    def test_unmatched_route_denied_for_authenticated_user(self):
        """An authenticated user hitting an unlisted route must be denied."""
        mw = self._make_middleware()
        ctx = _make_auth_context()

        allowed, reason, _ = mw.check_request("/api/secret-admin-panel", "GET", ctx)

        assert allowed is False
        assert "default-deny" in reason.lower() or "no permission rule matched" in reason.lower()

    def test_unmatched_route_denied_for_unauthenticated_user(self):
        """An unauthenticated user hitting an unlisted route must be denied."""
        mw = self._make_middleware()

        allowed, reason, _ = mw.check_request("/api/secret-admin-panel", "GET", None)

        assert allowed is False

    def test_unmatched_route_denied_for_various_methods(self):
        """Default-deny applies regardless of HTTP method."""
        mw = self._make_middleware()
        ctx = _make_auth_context()

        for method in ["GET", "POST", "PUT", "PATCH", "DELETE"]:
            allowed, reason, _ = mw.check_request("/api/unlisted-endpoint", method, ctx)
            assert allowed is False, f"Expected deny for {method} on unmatched route"

    def test_unmatched_route_with_path_traversal_variants(self):
        """Various path tricks should still be denied."""
        mw = self._make_middleware()
        ctx = _make_auth_context()

        tricky_paths = [
            "/api/debates/../admin",
            "/api/internal/secret",
            "/api/v2/debates",  # v2 not covered by rules
            "/api/debates%2F..%2Fadmin",
            "/api/DEBATES",  # case mismatch
        ]

        for path in tricky_paths:
            allowed, reason, _ = mw.check_request(path, "GET", ctx)
            assert allowed is False, f"Expected deny for tricky path: {path}"

    # -----------------------------------------------------------------
    # Legitimate access still works
    # -----------------------------------------------------------------

    def test_matched_route_allowed_with_permission(self):
        """A matched route with correct permissions should still be allowed."""
        mw = self._make_middleware()
        ctx = _make_auth_context()

        # Mock the permission checker to approve
        mw._checker = Mock()
        mw._checker.check_permission.return_value = Mock(allowed=True, reason="OK")

        allowed, reason, perm = mw.check_request("/api/debates", "GET", ctx)

        assert allowed is True
        assert perm == "debates.read"

    def test_bypass_paths_still_work(self):
        """Health check and other bypass paths should still be accessible."""
        mw = self._make_middleware()

        for path in ["/health", "/healthz", "/ready"]:
            allowed, reason, _ = mw.check_request(path, "GET", None)
            assert allowed is True, f"Bypass path {path} should still be allowed"

    def test_unauthenticated_routes_still_work(self):
        """Routes marked allow_unauthenticated should still work."""
        mw = self._make_middleware()

        allowed, reason, _ = mw.check_request("/api/auth/login", "POST", None)

        assert allowed is True

    def test_bypass_methods_still_work(self):
        """OPTIONS (CORS preflight) should still be allowed."""
        mw = self._make_middleware()

        allowed, reason, _ = mw.check_request("/api/anything", "OPTIONS", None)

        assert allowed is True

    # -----------------------------------------------------------------
    # Default middleware (with DEFAULT_ROUTE_PERMISSIONS)
    # -----------------------------------------------------------------

    def test_default_middleware_denies_unregistered_api_route(self):
        """The default middleware (full rule set) should deny unknown routes."""
        mw = RBACMiddleware(validate_permissions=False)
        ctx = _make_auth_context()

        allowed, reason, _ = mw.check_request(
            "/api/v1/totally-made-up-endpoint", "GET", ctx
        )

        assert allowed is False

    def test_default_middleware_allows_health(self):
        """The default middleware should allow health check endpoints."""
        mw = RBACMiddleware(validate_permissions=False)

        allowed, reason, _ = mw.check_request("/health", "GET", None)
        assert allowed is True

    def test_default_middleware_allows_login_unauthenticated(self):
        """The default middleware should allow login without auth."""
        mw = RBACMiddleware(validate_permissions=False)

        allowed, reason, _ = mw.check_request("/api/auth/login", "POST", None)
        assert allowed is True


class TestRBACDefaultDenyLogging:
    """Verify that default-deny events are logged for audit."""

    def test_default_deny_logs_warning(self):
        """Default-deny should log a warning with path and method details."""
        config = RBACMiddlewareConfig(
            route_permissions=[
                RoutePermission(r"^/api/debates$", "GET", "debates.read"),
            ],
            bypass_paths={"/health"},
        )
        mw = RBACMiddleware(config=config, validate_permissions=False)
        ctx = _make_auth_context()

        with patch("aragora.rbac.middleware.logger") as mock_logger:
            allowed, _, _ = mw.check_request("/api/secret", "POST", ctx)

            assert allowed is False
            mock_logger.warning.assert_called_once()
            log_msg = mock_logger.warning.call_args[0][0]
            assert "default-deny" in log_msg.lower()
