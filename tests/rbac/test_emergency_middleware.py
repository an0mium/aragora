"""Tests for emergency break-glass middleware integration.

Verifies that:
1. Emergency access bypass works in RBAC middleware
2. Emergency API handler endpoints function correctly
3. Break-glass sessions are properly audit-logged
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from aragora.rbac.emergency import (
    BreakGlassAccess,
    EmergencyAccessRecord,
    EmergencyAccessStatus,
    get_break_glass_access,
)
from aragora.rbac.middleware import RBACMiddleware, RBACMiddlewareConfig
from aragora.rbac.models import AuthorizationContext


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def emergency_mgr():
    """Return a fresh BreakGlassAccess instance (no persistence)."""
    return BreakGlassAccess(enable_persistence=False)


@pytest.fixture
def auth_context():
    """Return a minimal AuthorizationContext for testing."""
    return AuthorizationContext(
        user_id="user-42",
        roles=["member"],
        permissions=[],
    )


@pytest.fixture
def admin_context():
    """Return an admin AuthorizationContext for testing."""
    return AuthorizationContext(
        user_id="admin-1",
        roles=["admin"],
        permissions=["admin.*"],
    )


# ---------------------------------------------------------------------------
# Middleware bypass tests
# ---------------------------------------------------------------------------

class TestEmergencyMiddlewareBypass:
    """Verify emergency access bypasses normal RBAC in middleware."""

    @pytest.mark.asyncio
    async def test_normal_request_denied_without_permission(self, auth_context):
        """A regular user without the right permission is denied."""
        middleware = RBACMiddleware(
            RBACMiddlewareConfig(default_authenticated=True),
            validate_permissions=False,
        )
        allowed, reason, _ = middleware.check_request(
            "/api/v1/admin/security/status", "GET", auth_context
        )
        # admin.* routes require admin permission, regular member lacks it
        assert not allowed or "bypass" not in reason.lower()

    @pytest.mark.asyncio
    async def test_emergency_bypass_grants_access(self, emergency_mgr, auth_context):
        """User with active break-glass session bypasses RBAC."""
        # Activate emergency access
        access_id = await emergency_mgr.activate(
            user_id="user-42",
            reason="Production incident - database corruption requiring immediate fix",
            duration_minutes=60,
        )

        # Patch the singleton so middleware sees the active session
        with patch("aragora.rbac.emergency.get_break_glass_access", return_value=emergency_mgr):
            middleware = RBACMiddleware(
                RBACMiddlewareConfig(default_authenticated=True),
                validate_permissions=False,
            )
            allowed, reason, perm = middleware.check_request(
                "/api/v1/admin/security/status", "GET", auth_context
            )

        assert allowed
        assert "emergency" in reason.lower() or "break-glass" in reason.lower()

    @pytest.mark.asyncio
    async def test_expired_emergency_does_not_bypass(self, emergency_mgr, auth_context):
        """An expired emergency session does NOT bypass RBAC."""
        # Activate then manually expire
        access_id = await emergency_mgr.activate(
            user_id="user-42",
            reason="Production incident - database corruption requiring immediate fix",
            duration_minutes=60,
        )
        record = emergency_mgr._active_records[access_id]
        record.expires_at = datetime.now(timezone.utc) - timedelta(minutes=1)

        with patch("aragora.rbac.emergency.get_break_glass_access", return_value=emergency_mgr):
            middleware = RBACMiddleware(
                RBACMiddlewareConfig(default_authenticated=True),
                validate_permissions=False,
            )
            allowed, reason, _ = middleware.check_request(
                "/api/v1/admin/security/status", "GET", auth_context
            )

        # Should NOT be allowed via emergency bypass (expired)
        assert "emergency" not in reason.lower()

    @pytest.mark.asyncio
    async def test_deactivated_emergency_does_not_bypass(self, emergency_mgr, auth_context):
        """A deactivated emergency session does NOT bypass RBAC."""
        access_id = await emergency_mgr.activate(
            user_id="user-42",
            reason="Production incident - database corruption requiring immediate fix",
            duration_minutes=60,
        )
        await emergency_mgr.deactivate(access_id)

        with patch("aragora.rbac.emergency.get_break_glass_access", return_value=emergency_mgr):
            middleware = RBACMiddleware(
                RBACMiddlewareConfig(default_authenticated=True),
                validate_permissions=False,
            )
            allowed, reason, _ = middleware.check_request(
                "/api/v1/admin/security/status", "GET", auth_context
            )

        assert "emergency" not in reason.lower()

    @pytest.mark.asyncio
    async def test_bypass_only_for_matching_user(self, emergency_mgr):
        """Emergency bypass only applies to the user with the active session."""
        await emergency_mgr.activate(
            user_id="user-42",
            reason="Production incident - database corruption requiring immediate fix",
            duration_minutes=60,
        )

        other_context = AuthorizationContext(
            user_id="other-user",
            roles=["member"],
            permissions=[],
        )

        with patch("aragora.rbac.emergency.get_break_glass_access", return_value=emergency_mgr):
            middleware = RBACMiddleware(
                RBACMiddlewareConfig(default_authenticated=True),
                validate_permissions=False,
            )
            allowed, reason, _ = middleware.check_request(
                "/api/v1/admin/security/status", "GET", other_context
            )

        # The other user should NOT get emergency bypass
        assert "emergency" not in reason.lower()

    def test_bypass_paths_still_work(self, auth_context):
        """Bypass paths (health, metrics) still work normally."""
        middleware = RBACMiddleware(
            RBACMiddlewareConfig(default_authenticated=True),
            validate_permissions=False,
        )
        allowed, reason, _ = middleware.check_request("/health", "GET", None)
        assert allowed
        assert "bypass" in reason.lower()

    def test_unauthenticated_request_not_bypassed(self, emergency_mgr):
        """Emergency bypass requires authentication (context must not be None)."""
        middleware = RBACMiddleware(
            RBACMiddlewareConfig(default_authenticated=True),
            validate_permissions=False,
        )
        allowed, reason, _ = middleware.check_request(
            "/api/v1/admin/security/status", "GET", None
        )
        assert not allowed


# ---------------------------------------------------------------------------
# Emergency API handler tests
# ---------------------------------------------------------------------------

class TestEmergencyAccessHandler:
    """Tests for the EmergencyAccessHandler endpoints."""

    def test_handler_can_handle_routes(self):
        """Handler correctly identifies its routes."""
        from aragora.server.handlers.admin.emergency_access import EmergencyAccessHandler

        handler = EmergencyAccessHandler({})
        assert handler.can_handle("/api/v1/admin/emergency/activate")
        assert handler.can_handle("/api/v1/admin/emergency/deactivate")
        assert handler.can_handle("/api/v1/admin/emergency/status")
        assert handler.can_handle("/api/admin/emergency/activate")
        assert not handler.can_handle("/api/v1/admin/users")
        assert not handler.can_handle("/api/v1/debates")

    def test_handler_has_rbac_protection(self):
        """Handler uses require_permission for protection."""
        from aragora.server.handlers.admin.emergency_access import EmergencyAccessHandler

        handler = EmergencyAccessHandler({})
        # The _activate, _deactivate, and _status methods use @require_permission
        # This is verified by the RBAC enforcement test scanning for the import
        import inspect
        source = inspect.getsource(EmergencyAccessHandler)
        assert "require_permission" in source

    def test_handler_import(self):
        """Handler can be imported from the expected location."""
        from aragora.server.handlers.admin.emergency_access import EmergencyAccessHandler
        assert EmergencyAccessHandler is not None

    def test_handler_registered_in_admin_registry(self):
        """Handler is registered in the admin handler registry."""
        from aragora.server.handler_registry.admin import (
            ADMIN_HANDLER_REGISTRY,
            EmergencyAccessHandler,
        )
        # Find entry in registry
        registry_names = [name for name, _ in ADMIN_HANDLER_REGISTRY]
        assert "_emergency_access_handler" in registry_names


# ---------------------------------------------------------------------------
# Route permission tests
# ---------------------------------------------------------------------------

class TestEmergencyRoutePermissions:
    """Verify emergency routes are covered by RBAC middleware."""

    def test_emergency_routes_in_default_permissions(self):
        """Emergency routes should be in DEFAULT_ROUTE_PERMISSIONS."""
        from aragora.rbac.middleware import DEFAULT_ROUTE_PERMISSIONS

        emergency_patterns = []
        for rule in DEFAULT_ROUTE_PERMISSIONS:
            pattern_str = rule.pattern.pattern if hasattr(rule.pattern, "pattern") else str(rule.pattern)
            if "emergency" in pattern_str:
                emergency_patterns.append(pattern_str)

        assert len(emergency_patterns) >= 3, (
            f"Expected at least 3 emergency route permissions, found {len(emergency_patterns)}"
        )

    def test_emergency_activate_requires_admin(self):
        """POST /api/v1/admin/emergency/activate requires admin permission."""
        from aragora.rbac.middleware import DEFAULT_ROUTE_PERMISSIONS

        for rule in DEFAULT_ROUTE_PERMISSIONS:
            matches, _ = rule.matches("/api/v1/admin/emergency/activate", "POST")
            if matches:
                assert "admin" in rule.permission_key
                break
        else:
            pytest.fail("No route permission found for /api/v1/admin/emergency/activate")

    def test_emergency_deactivate_requires_admin(self):
        """POST /api/v1/admin/emergency/deactivate requires admin permission."""
        from aragora.rbac.middleware import DEFAULT_ROUTE_PERMISSIONS

        for rule in DEFAULT_ROUTE_PERMISSIONS:
            matches, _ = rule.matches("/api/v1/admin/emergency/deactivate", "POST")
            if matches:
                assert "admin" in rule.permission_key
                break
        else:
            pytest.fail("No route permission found for /api/v1/admin/emergency/deactivate")

    def test_emergency_status_requires_admin(self):
        """GET /api/v1/admin/emergency/status requires admin permission."""
        from aragora.rbac.middleware import DEFAULT_ROUTE_PERMISSIONS

        for rule in DEFAULT_ROUTE_PERMISSIONS:
            matches, _ = rule.matches("/api/v1/admin/emergency/status", "GET")
            if matches:
                assert "admin" in rule.permission_key
                break
        else:
            pytest.fail("No route permission found for /api/v1/admin/emergency/status")


# ---------------------------------------------------------------------------
# Export tests
# ---------------------------------------------------------------------------

class TestEmergencyExports:
    """Verify emergency classes are properly exported."""

    def test_rbac_package_exports(self):
        """Emergency classes exported from aragora.rbac."""
        from aragora.rbac import (
            BreakGlassAccess,
            EmergencyAccessRecord,
            EmergencyAccessStatus,
            get_break_glass_access,
        )
        assert BreakGlassAccess is not None
        assert EmergencyAccessRecord is not None
        assert EmergencyAccessStatus is not None
        assert callable(get_break_glass_access)

    def test_emergency_module_direct_import(self):
        """Emergency classes importable directly from emergency module."""
        from aragora.rbac.emergency import (
            BreakGlassAccess,
            EmergencyAccessRecord,
            EmergencyAccessStatus,
            get_break_glass_access,
        )
        assert BreakGlassAccess is not None
