"""
Tests for @require_mfa RBAC decorator (GitHub issue #275).

Tests cover:
- Admin users blocked without MFA
- Admin users with MFA allowed
- Non-admin users pass through without MFA
- Grace period for newly promoted admins
- Service account bypass
- Custom policy configuration
- Async handler support
- Stacking with @require_permission
- MFARequiredError exception attributes
- Integration with MFAEnforcementMiddleware
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from aragora.auth.mfa_enforcement import (
    MFAEnforcementPolicy,
    MFAStatus,
)
from aragora.rbac.decorators import (
    MFARequiredError,
    PermissionDeniedError,
    require_mfa,
    require_permission,
)
from aragora.rbac.models import AuthorizationContext


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_context(
    user_id: str = "user-1",
    roles: set[str] | None = None,
    permissions: set[str] | None = None,
    metadata: dict | None = None,
) -> AuthorizationContext:
    """Create an AuthorizationContext for testing."""
    ctx = AuthorizationContext(
        user_id=user_id,
        roles=roles or {"member"},
        permissions=permissions or set(),
    )
    if metadata:
        # AuthorizationContext doesn't have a metadata field, so we
        # attach it as an attribute for the MFA enforcement middleware
        # which reads getattr(user_context, "metadata", {}).
        ctx.metadata = metadata  # type: ignore[attr-defined]
    return ctx


# ---------------------------------------------------------------------------
# Admin users blocked without MFA
# ---------------------------------------------------------------------------


class TestAdminBlockedWithoutMFA:
    """Admin users must be denied if MFA is not enabled."""

    def test_admin_without_mfa_denied(self):
        ctx = _make_context(user_id="admin-1", roles={"admin"})

        @require_mfa()
        def admin_action(context: AuthorizationContext) -> str:
            return "success"

        with pytest.raises(MFARequiredError) as exc_info:
            admin_action(ctx)

        assert "MFA" in str(exc_info.value)
        assert exc_info.value.user_id == "admin-1"
        assert exc_info.value.required_action == "/api/auth/mfa/setup"

    def test_owner_without_mfa_denied(self):
        ctx = _make_context(user_id="owner-1", roles={"owner"})

        @require_mfa()
        def owner_action(context: AuthorizationContext) -> str:
            return "success"

        with pytest.raises(MFARequiredError):
            owner_action(ctx)

    def test_superadmin_without_mfa_denied(self):
        ctx = _make_context(user_id="sa-1", roles={"superadmin"})

        @require_mfa()
        def superadmin_action(context: AuthorizationContext) -> str:
            return "success"

        with pytest.raises(MFARequiredError):
            superadmin_action(ctx)

    def test_org_admin_without_mfa_denied(self):
        ctx = _make_context(user_id="org-1", roles={"org_admin"})

        @require_mfa()
        def org_admin_action(context: AuthorizationContext) -> str:
            return "success"

        with pytest.raises(MFARequiredError):
            org_admin_action(ctx)

    def test_workspace_admin_without_mfa_denied(self):
        ctx = _make_context(user_id="ws-1", roles={"workspace_admin"})

        @require_mfa()
        def ws_admin_action(context: AuthorizationContext) -> str:
            return "success"

        with pytest.raises(MFARequiredError):
            ws_admin_action(ctx)

    def test_security_admin_without_mfa_denied(self):
        ctx = _make_context(user_id="sec-1", roles={"security_admin"})

        @require_mfa()
        def sec_admin_action(context: AuthorizationContext) -> str:
            return "success"

        with pytest.raises(MFARequiredError):
            sec_admin_action(ctx)

    def test_compliance_officer_without_mfa_denied(self):
        ctx = _make_context(user_id="co-1", roles={"compliance_officer"})

        @require_mfa()
        def compliance_action(context: AuthorizationContext) -> str:
            return "success"

        with pytest.raises(MFARequiredError):
            compliance_action(ctx)

    def test_multiple_roles_one_requires_mfa_denied(self):
        """User has both member and admin roles -- MFA still required."""
        ctx = _make_context(user_id="multi-1", roles={"member", "admin"})

        @require_mfa()
        def multi_role_action(context: AuthorizationContext) -> str:
            return "success"

        with pytest.raises(MFARequiredError):
            multi_role_action(ctx)


# ---------------------------------------------------------------------------
# Admin users with MFA allowed
# ---------------------------------------------------------------------------


class TestAdminWithMFAAllowed:
    """Admin users with MFA enabled should be allowed through."""

    def test_admin_with_mfa_via_status(self):
        ctx = _make_context(
            user_id="admin-mfa",
            roles={"admin"},
            metadata={"mfa_enabled": True},
        )

        @require_mfa()
        def admin_action(context: AuthorizationContext) -> str:
            return "success"

        result = admin_action(ctx)
        assert result == "success"

    def test_owner_with_mfa_allowed(self):
        ctx = _make_context(
            user_id="owner-mfa",
            roles={"owner"},
            metadata={"mfa_enabled": True},
        )

        @require_mfa()
        def owner_action(context: AuthorizationContext) -> str:
            return "allowed"

        result = owner_action(ctx)
        assert result == "allowed"

    def test_admin_with_mfa_via_user_store(self):
        """MFA status resolved via user_store."""
        store = MagicMock()
        stored_user = MagicMock()
        stored_user.mfa_enabled = True
        stored_user.is_service_account = False
        store.get_user_by_id.return_value = stored_user

        ctx = _make_context(user_id="admin-store", roles={"admin"})

        @require_mfa(user_store=store)
        def admin_action(context: AuthorizationContext) -> str:
            return "via-store"

        result = admin_action(ctx)
        assert result == "via-store"
        store.get_user_by_id.assert_called_once_with("admin-store")


# ---------------------------------------------------------------------------
# Non-admin users pass through
# ---------------------------------------------------------------------------


class TestNonAdminPassThrough:
    """Non-admin users should not be affected by MFA enforcement."""

    def test_member_allowed_without_mfa(self):
        ctx = _make_context(user_id="user-1", roles={"member"})

        @require_mfa()
        def member_action(context: AuthorizationContext) -> str:
            return "ok"

        result = member_action(ctx)
        assert result == "ok"

    def test_viewer_allowed_without_mfa(self):
        ctx = _make_context(user_id="user-2", roles={"viewer"})

        @require_mfa()
        def viewer_action(context: AuthorizationContext) -> str:
            return "ok"

        result = viewer_action(ctx)
        assert result == "ok"

    def test_debate_creator_allowed_without_mfa(self):
        ctx = _make_context(user_id="user-3", roles={"debate_creator"})

        @require_mfa()
        def creator_action(context: AuthorizationContext) -> str:
            return "ok"

        result = creator_action(ctx)
        assert result == "ok"

    def test_empty_roles_allowed(self):
        ctx = _make_context(user_id="user-4", roles=set())

        @require_mfa()
        def no_role_action(context: AuthorizationContext) -> str:
            return "ok"

        result = no_role_action(ctx)
        assert result == "ok"

    def test_analyst_allowed_without_mfa(self):
        ctx = _make_context(user_id="user-5", roles={"analyst"})

        @require_mfa()
        def analyst_action(context: AuthorizationContext) -> str:
            return "ok"

        result = analyst_action(ctx)
        assert result == "ok"


# ---------------------------------------------------------------------------
# Grace period for newly promoted admins
# ---------------------------------------------------------------------------


class TestGracePeriod:
    """Newly-promoted admins get a grace period to set up MFA."""

    def test_within_grace_period_allowed(self):
        now = datetime.now(timezone.utc)
        recently_promoted = now - timedelta(hours=1)

        ctx = _make_context(
            user_id="new-admin",
            roles={"admin"},
            metadata={"role_assigned_at": recently_promoted},
        )

        @require_mfa()
        def admin_action(context: AuthorizationContext) -> str:
            return "grace"

        result = admin_action(ctx)
        assert result == "grace"

    def test_expired_grace_period_denied(self):
        long_ago = datetime.now(timezone.utc) - timedelta(hours=100)

        ctx = _make_context(
            user_id="old-admin",
            roles={"admin"},
            metadata={"role_assigned_at": long_ago},
        )

        @require_mfa()
        def admin_action(context: AuthorizationContext) -> str:
            return "should-not-reach"

        with pytest.raises(MFARequiredError):
            admin_action(ctx)

    def test_custom_grace_period(self):
        now = datetime.now(timezone.utc)
        promoted_5h_ago = now - timedelta(hours=5)

        policy = MFAEnforcementPolicy(grace_period_hours=4)
        ctx = _make_context(
            user_id="admin-gp",
            roles={"admin"},
            metadata={"role_assigned_at": promoted_5h_ago},
        )

        @require_mfa(policy=policy)
        def admin_action(context: AuthorizationContext) -> str:
            return "should-not-reach"

        with pytest.raises(MFARequiredError):
            admin_action(ctx)

    def test_zero_grace_period(self):
        now = datetime.now(timezone.utc)
        just_promoted = now - timedelta(seconds=10)

        policy = MFAEnforcementPolicy(grace_period_hours=0)
        ctx = _make_context(
            user_id="admin-no-gp",
            roles={"admin"},
            metadata={"role_assigned_at": just_promoted},
        )

        @require_mfa(policy=policy)
        def admin_action(context: AuthorizationContext) -> str:
            return "should-not-reach"

        with pytest.raises(MFARequiredError):
            admin_action(ctx)


# ---------------------------------------------------------------------------
# Service account bypass
# ---------------------------------------------------------------------------


class TestServiceAccountBypass:
    """Service accounts with valid bypass should be allowed."""

    def test_service_account_with_bypass_allowed(self):
        ctx = _make_context(user_id="svc-1", roles={"admin"})
        status = MFAStatus(
            user_id="svc-1",
            mfa_enabled=False,
            is_service_account=True,
            has_valid_bypass=True,
        )

        # Use a custom middleware that returns our pre-built status
        from aragora.auth.mfa_enforcement import MFAEnforcementMiddleware

        mw = MFAEnforcementMiddleware()
        decision = mw.enforce(ctx, mfa_status=status)
        assert decision.allowed

    def test_service_account_without_bypass_denied(self):
        ctx = _make_context(user_id="svc-2", roles={"admin"})
        status = MFAStatus(
            user_id="svc-2",
            mfa_enabled=False,
            is_service_account=True,
            has_valid_bypass=False,
        )

        from aragora.auth.mfa_enforcement import MFAEnforcementMiddleware

        mw = MFAEnforcementMiddleware()
        decision = mw.enforce(ctx, mfa_status=status)
        assert not decision.allowed


# ---------------------------------------------------------------------------
# Policy configuration
# ---------------------------------------------------------------------------


class TestPolicyConfiguration:
    """Test policy customization and the enabled toggle."""

    def test_enforcement_disabled(self):
        policy = MFAEnforcementPolicy(enabled=False)
        ctx = _make_context(user_id="admin-1", roles={"admin"})

        @require_mfa(policy=policy)
        def admin_action(context: AuthorizationContext) -> str:
            return "allowed-when-disabled"

        result = admin_action(ctx)
        assert result == "allowed-when-disabled"

    def test_custom_required_roles(self):
        policy = MFAEnforcementPolicy(
            required_roles=frozenset({"custom_admin"}),
        )
        ctx = _make_context(user_id="ca-1", roles={"custom_admin"})

        @require_mfa(policy=policy)
        def custom_admin_action(context: AuthorizationContext) -> str:
            return "should-not-reach"

        with pytest.raises(MFARequiredError):
            custom_admin_action(ctx)

    def test_standard_admin_not_required_with_custom_roles(self):
        policy = MFAEnforcementPolicy(
            required_roles=frozenset({"custom_admin"}),
        )
        ctx = _make_context(user_id="admin-1", roles={"admin"})

        @require_mfa(policy=policy)
        def action(context: AuthorizationContext) -> str:
            return "pass"

        result = action(ctx)
        assert result == "pass"


# ---------------------------------------------------------------------------
# Async handler support
# ---------------------------------------------------------------------------


class TestAsyncHandlerSupport:
    """Test that @require_mfa works with async handlers."""

    def test_async_admin_without_mfa_denied(self):
        ctx = _make_context(user_id="async-admin", roles={"admin"})

        @require_mfa()
        async def async_admin_action(context: AuthorizationContext) -> str:
            return "success"

        with pytest.raises(MFARequiredError):
            asyncio.get_event_loop().run_until_complete(async_admin_action(ctx))

    def test_async_admin_with_mfa_allowed(self):
        ctx = _make_context(
            user_id="async-admin-mfa",
            roles={"admin"},
            metadata={"mfa_enabled": True},
        )

        @require_mfa()
        async def async_admin_action(context: AuthorizationContext) -> str:
            return "async-success"

        result = asyncio.get_event_loop().run_until_complete(async_admin_action(ctx))
        assert result == "async-success"

    def test_async_non_admin_allowed(self):
        ctx = _make_context(user_id="async-member", roles={"member"})

        @require_mfa()
        async def async_member_action(context: AuthorizationContext) -> str:
            return "async-member"

        result = asyncio.get_event_loop().run_until_complete(async_member_action(ctx))
        assert result == "async-member"


# ---------------------------------------------------------------------------
# Stacking with @require_permission
# ---------------------------------------------------------------------------


class TestDecoratorStacking:
    """Test stacking @require_mfa with @require_permission."""

    @pytest.fixture(autouse=True)
    def setup_checker(self):
        from aragora.rbac.checker import PermissionChecker, set_permission_checker

        checker = PermissionChecker(enable_cache=False)
        set_permission_checker(checker)
        yield
        set_permission_checker(None)

    def test_permission_and_mfa_both_pass(self):
        ctx = _make_context(
            user_id="admin-stack",
            roles={"admin"},
            permissions={"admin:system_config"},
            metadata={"mfa_enabled": True},
        )

        @require_permission("admin:system_config")
        @require_mfa()
        def sensitive_action(context: AuthorizationContext) -> str:
            return "stacked-success"

        result = sensitive_action(ctx)
        assert result == "stacked-success"

    def test_permission_passes_mfa_fails(self):
        ctx = _make_context(
            user_id="admin-no-mfa",
            roles={"admin"},
            permissions={"admin:system_config"},
        )

        @require_permission("admin:system_config")
        @require_mfa()
        def sensitive_action(context: AuthorizationContext) -> str:
            return "should-not-reach"

        with pytest.raises(MFARequiredError):
            sensitive_action(ctx)

    def test_permission_fails_mfa_not_checked(self):
        ctx = _make_context(
            user_id="admin-no-perm",
            roles={"admin"},
            permissions=set(),
        )

        @require_permission("admin:system_config")
        @require_mfa()
        def sensitive_action(context: AuthorizationContext) -> str:
            return "should-not-reach"

        with pytest.raises(PermissionDeniedError):
            sensitive_action(ctx)


# ---------------------------------------------------------------------------
# MFARequiredError exception attributes
# ---------------------------------------------------------------------------


class TestMFARequiredError:
    """Test MFARequiredError exception structure."""

    def test_error_fields(self):
        err = MFARequiredError(
            "MFA required",
            user_id="u-1",
            roles={"admin"},
            required_action="/api/auth/mfa/setup",
            grace_period_remaining_hours=24,
        )
        assert str(err) == "MFA required"
        assert err.user_id == "u-1"
        assert err.roles == {"admin"}
        assert err.required_action == "/api/auth/mfa/setup"
        assert err.grace_period_remaining_hours == 24

    def test_error_default_fields(self):
        err = MFARequiredError("MFA needed")
        assert err.user_id is None
        assert err.roles == set()
        assert err.required_action == "/api/auth/mfa/setup"
        assert err.grace_period_remaining_hours is None

    def test_error_is_catchable_as_exception(self):
        with pytest.raises(Exception):
            raise MFARequiredError("test")

    def test_error_attributes_on_denied(self):
        ctx = _make_context(user_id="admin-err", roles={"admin"})

        @require_mfa()
        def admin_action(context: AuthorizationContext) -> str:
            return "success"

        with pytest.raises(MFARequiredError) as exc_info:
            admin_action(ctx)

        assert exc_info.value.user_id == "admin-err"
        assert "admin" in exc_info.value.roles
        assert exc_info.value.required_action == "/api/auth/mfa/setup"


# ---------------------------------------------------------------------------
# No context handling
# ---------------------------------------------------------------------------


class TestNoContextHandling:
    """Test behavior when no AuthorizationContext is provided."""

    def test_no_context_raises_mfa_error(self):
        @require_mfa()
        def action() -> str:
            return "success"

        with patch("aragora.server.auth.auth_config") as mock_auth:
            mock_auth.enabled = True
            with pytest.raises(MFARequiredError) as exc_info:
                action()

            assert "No AuthorizationContext" in str(exc_info.value)

    def test_no_context_auth_disabled_allows(self):
        @require_mfa()
        def action() -> str:
            return "no-auth"

        with patch("aragora.server.auth.auth_config") as mock_auth:
            mock_auth.enabled = False
            result = action()

        assert result == "no-auth"


# ---------------------------------------------------------------------------
# Decorator preserves function metadata
# ---------------------------------------------------------------------------


class TestDecoratorMetadata:
    """Test that @require_mfa preserves function metadata."""

    def test_preserves_name(self):
        @require_mfa()
        def my_admin_endpoint():
            """My admin endpoint docstring."""
            pass

        assert my_admin_endpoint.__name__ == "my_admin_endpoint"
        assert my_admin_endpoint.__doc__ == "My admin endpoint docstring."

    def test_preserves_async_name(self):
        @require_mfa()
        async def my_async_endpoint():
            """Async admin endpoint."""
            pass

        assert my_async_endpoint.__name__ == "my_async_endpoint"
        assert my_async_endpoint.__doc__ == "Async admin endpoint."


# ---------------------------------------------------------------------------
# Context extraction from kwargs
# ---------------------------------------------------------------------------


class TestContextExtraction:
    """Test context extraction from different argument positions."""

    def test_context_from_kwargs(self):
        ctx = _make_context(user_id="kw-user", roles={"member"})

        @require_mfa()
        def action(context: AuthorizationContext) -> str:
            return "ok"

        result = action(context=ctx)
        assert result == "ok"

    def test_context_from_positional(self):
        ctx = _make_context(user_id="pos-user", roles={"member"})

        @require_mfa()
        def action(context: AuthorizationContext) -> str:
            return "ok"

        result = action(ctx)
        assert result == "ok"

    def test_context_from_second_arg_with_self(self):
        """When used on a method, context is the second arg."""
        ctx = _make_context(user_id="method-user", roles={"member"})

        class Handler:
            @require_mfa()
            def action(self, context: AuthorizationContext) -> str:
                return "method-ok"

        handler = Handler()
        result = handler.action(ctx)
        assert result == "method-ok"


# ---------------------------------------------------------------------------
# Exception mapping in handler error decorator
# ---------------------------------------------------------------------------


class TestExceptionMapping:
    """Test that MFARequiredError is mapped to 403 in handler error handling."""

    def test_mfa_error_maps_to_403(self):
        from aragora.server.handlers.utils.decorators import map_exception_to_status

        err = MFARequiredError("MFA required", user_id="u-1")
        status = map_exception_to_status(err)
        assert status == 403
