"""
Tests for signup handler RBAC enforcement.

Tests cover:
- Middleware correctly marks public vs protected signup endpoints
- Permission keys are correctly configured
"""

import pytest


class TestSignupMiddlewareIntegration:
    """Tests verifying middleware route permissions match expected behavior."""

    def test_middleware_marks_signup_as_public(self):
        """Middleware should mark signup as allow_unauthenticated."""
        from aragora.rbac.middleware import DEFAULT_ROUTE_PERMISSIONS

        signup_routes = [
            rp
            for rp in DEFAULT_ROUTE_PERMISSIONS
            if "auth/signup" in rp.pattern.pattern and rp.allow_unauthenticated
        ]
        assert len(signup_routes) > 0, "Signup should be marked as public in middleware"

    def test_middleware_marks_register_as_public(self):
        """Middleware should mark register as allow_unauthenticated."""
        from aragora.rbac.middleware import DEFAULT_ROUTE_PERMISSIONS

        register_routes = [
            rp
            for rp in DEFAULT_ROUTE_PERMISSIONS
            if "auth/register" in rp.pattern.pattern and rp.allow_unauthenticated
        ]
        assert len(register_routes) > 0, "Register should be marked as public in middleware"

    def test_middleware_marks_verify_email_as_public(self):
        """Middleware should mark verify-email as allow_unauthenticated."""
        from aragora.rbac.middleware import DEFAULT_ROUTE_PERMISSIONS

        verify_routes = [
            rp
            for rp in DEFAULT_ROUTE_PERMISSIONS
            if "verify-email" in rp.pattern.pattern and rp.allow_unauthenticated
        ]
        assert len(verify_routes) > 0, "Verify email should be marked as public in middleware"

    def test_middleware_marks_check_invite_as_public(self):
        """Middleware should mark check-invite as allow_unauthenticated."""
        from aragora.rbac.middleware import DEFAULT_ROUTE_PERMISSIONS

        invite_routes = [
            rp
            for rp in DEFAULT_ROUTE_PERMISSIONS
            if "check-invite" in rp.pattern.pattern and rp.allow_unauthenticated
        ]
        assert len(invite_routes) > 0, "Check invite should be marked as public in middleware"

    def test_middleware_marks_accept_invite_as_public(self):
        """Middleware should mark accept-invite as allow_unauthenticated."""
        from aragora.rbac.middleware import DEFAULT_ROUTE_PERMISSIONS

        accept_routes = [
            rp
            for rp in DEFAULT_ROUTE_PERMISSIONS
            if "accept-invite" in rp.pattern.pattern and rp.allow_unauthenticated
        ]
        assert len(accept_routes) > 0, "Accept invite should be marked as public in middleware"

    def test_middleware_protects_setup_organization(self):
        """Middleware should protect setup-organization with organization.update."""
        from aragora.rbac.middleware import DEFAULT_ROUTE_PERMISSIONS

        setup_routes = [
            rp
            for rp in DEFAULT_ROUTE_PERMISSIONS
            if "setup-organization" in rp.pattern.pattern and rp.permission_key
        ]
        assert len(setup_routes) > 0, "Setup organization should be protected"
        assert any("organization" in rp.permission_key for rp in setup_routes), (
            "Should require organization permission"
        )

    def test_middleware_protects_invite_endpoint(self):
        """Middleware should protect /auth/invite with organization.invite."""
        from aragora.rbac.middleware import DEFAULT_ROUTE_PERMISSIONS

        invite_routes = [
            rp
            for rp in DEFAULT_ROUTE_PERMISSIONS
            if "auth/invite" in rp.pattern.pattern and rp.permission_key
        ]
        assert len(invite_routes) > 0, "Invite should be protected"
        assert any("organization.invite" in rp.permission_key for rp in invite_routes), (
            "Should require organization.invite permission"
        )

    def test_middleware_protects_onboarding(self):
        """Middleware should protect onboarding endpoints with authentication.read."""
        from aragora.rbac.middleware import DEFAULT_ROUTE_PERMISSIONS

        onboarding_routes = [
            rp
            for rp in DEFAULT_ROUTE_PERMISSIONS
            if "onboarding" in rp.pattern.pattern and rp.permission_key
        ]
        assert len(onboarding_routes) > 0, "Onboarding should be protected"
        assert any("authentication" in rp.permission_key for rp in onboarding_routes), (
            "Should require authentication permission"
        )


class TestSignupHandlerExistence:
    """Tests verifying signup handlers exist and are async."""

    def test_signup_handler_is_async(self):
        """Signup handler should be async."""
        import inspect
        from aragora.server.handlers.auth.signup_handlers import handle_signup

        assert inspect.iscoroutinefunction(handle_signup)

    def test_verify_email_handler_is_async(self):
        """Verify email handler should be async."""
        import inspect
        from aragora.server.handlers.auth.signup_handlers import handle_verify_email

        assert inspect.iscoroutinefunction(handle_verify_email)

    def test_setup_organization_handler_is_async(self):
        """Setup organization handler should be async."""
        import inspect
        from aragora.server.handlers.auth.signup_handlers import handle_setup_organization

        assert inspect.iscoroutinefunction(handle_setup_organization)

    def test_invite_handler_is_async(self):
        """Invite handler should be async."""
        import inspect
        from aragora.server.handlers.auth.signup_handlers import handle_invite

        assert inspect.iscoroutinefunction(handle_invite)
