"""
Tests for MFA enforcement middleware (GitHub issue #275).

Tests cover:
- Admin users blocked without MFA
- Non-admin users pass through freely
- Grace period logic for newly-promoted admins
- Exempt path handling
- Service account bypass
- Policy configuration
- check_mfa_required utility
- get_mfa_status utility
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from aragora.auth.mfa_enforcement import (
    DEFAULT_MFA_REQUIRED_ROLES,
    MFAEnforcementDecision,
    MFAEnforcementMiddleware,
    MFAEnforcementPolicy,
    MFAEnforcementResult,
    MFAStatus,
    check_mfa_required,
    get_mfa_status,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class FakeUserContext:
    """Minimal stand-in for AuthorizationContext."""

    user_id: str = "user-1"
    roles: set[str] = field(default_factory=lambda: {"member"})
    metadata: dict = field(default_factory=dict)


@dataclass
class FakeStoredUser:
    """Minimal stand-in for a user record from the store."""

    id: str = "user-1"
    mfa_enabled: bool = False
    mfa_verified_at: datetime | None = None
    mfa_grace_period_started_at: datetime | None = None
    created_at: datetime | None = None
    is_service_account: bool = False
    mfa_bypass_approved_at: datetime | None = None


def _make_middleware(
    policy: MFAEnforcementPolicy | None = None,
    user_store: MagicMock | None = None,
) -> MFAEnforcementMiddleware:
    return MFAEnforcementMiddleware(policy=policy, user_store=user_store)


# ---------------------------------------------------------------------------
# Admin users blocked without MFA
# ---------------------------------------------------------------------------

class TestAdminBlockedWithoutMFA:
    """Admin users must be denied if MFA is not enabled."""

    def test_admin_without_mfa_denied(self):
        ctx = FakeUserContext(user_id="admin-1", roles={"admin"})
        mw = _make_middleware()
        decision = mw.enforce(ctx, path="/api/v1/admin/users")

        assert not decision.allowed
        assert decision.result == MFAEnforcementResult.DENIED
        assert "MFA" in decision.reason
        assert decision.required_action == "/api/auth/mfa/setup"

    def test_owner_without_mfa_denied(self):
        ctx = FakeUserContext(user_id="owner-1", roles={"owner"})
        mw = _make_middleware()
        decision = mw.enforce(ctx, path="/api/v1/admin/settings")

        assert not decision.allowed
        assert decision.result == MFAEnforcementResult.DENIED

    def test_superadmin_without_mfa_denied(self):
        ctx = FakeUserContext(user_id="sa-1", roles={"superadmin"})
        mw = _make_middleware()
        decision = mw.enforce(ctx, path="/api/v1/users")

        assert not decision.allowed
        assert decision.result == MFAEnforcementResult.DENIED

    def test_security_admin_without_mfa_denied(self):
        ctx = FakeUserContext(user_id="sec-1", roles={"security_admin"})
        mw = _make_middleware()
        decision = mw.enforce(ctx, path="/api/v1/admin/security")

        assert not decision.allowed
        assert decision.result == MFAEnforcementResult.DENIED

    def test_compliance_officer_without_mfa_denied(self):
        ctx = FakeUserContext(user_id="co-1", roles={"compliance_officer"})
        mw = _make_middleware()
        decision = mw.enforce(ctx, path="/api/v1/compliance")

        assert not decision.allowed
        assert decision.result == MFAEnforcementResult.DENIED

    def test_admin_with_mfa_allowed(self):
        ctx = FakeUserContext(user_id="admin-1", roles={"admin"})
        status = MFAStatus(user_id="admin-1", mfa_enabled=True)
        mw = _make_middleware()
        decision = mw.enforce(ctx, path="/api/v1/admin/users", mfa_status=status)

        assert decision.allowed
        assert decision.result == MFAEnforcementResult.ALLOWED

    def test_admin_with_mfa_via_store(self):
        store = MagicMock()
        stored_user = FakeStoredUser(id="admin-1", mfa_enabled=True)
        store.get_user_by_id.return_value = stored_user

        ctx = FakeUserContext(user_id="admin-1", roles={"admin"})
        mw = _make_middleware(user_store=store)
        decision = mw.enforce(ctx, path="/api/v1/admin/users")

        assert decision.allowed
        assert decision.result == MFAEnforcementResult.ALLOWED
        store.get_user_by_id.assert_called_once_with("admin-1")

    def test_multiple_roles_one_requires_mfa(self):
        """User has both member and admin roles -- MFA still required."""
        ctx = FakeUserContext(user_id="u-1", roles={"member", "admin"})
        mw = _make_middleware()
        decision = mw.enforce(ctx, path="/api/v1/some-path")

        assert not decision.allowed
        assert decision.result == MFAEnforcementResult.DENIED

    def test_case_insensitive_role_matching(self):
        ctx = FakeUserContext(user_id="u-1", roles={"Admin"})
        mw = _make_middleware()
        decision = mw.enforce(ctx, path="/api/v1/admin/users")

        assert not decision.allowed
        assert decision.result == MFAEnforcementResult.DENIED


# ---------------------------------------------------------------------------
# Non-admin users pass through
# ---------------------------------------------------------------------------

class TestNonAdminPassThrough:
    """Non-admin users should not be affected by MFA enforcement."""

    def test_member_allowed_without_mfa(self):
        ctx = FakeUserContext(user_id="user-1", roles={"member"})
        mw = _make_middleware()
        decision = mw.enforce(ctx, path="/api/v1/debates")

        assert decision.allowed
        assert decision.result == MFAEnforcementResult.NOT_REQUIRED

    def test_viewer_allowed_without_mfa(self):
        ctx = FakeUserContext(user_id="user-2", roles={"viewer"})
        mw = _make_middleware()
        decision = mw.enforce(ctx, path="/api/v1/debates")

        assert decision.allowed
        assert decision.result == MFAEnforcementResult.NOT_REQUIRED

    def test_empty_roles_allowed(self):
        ctx = FakeUserContext(user_id="user-3", roles=set())
        mw = _make_middleware()
        decision = mw.enforce(ctx, path="/api/v1/debates")

        assert decision.allowed
        assert decision.result == MFAEnforcementResult.NOT_REQUIRED

    def test_debate_creator_allowed(self):
        ctx = FakeUserContext(user_id="user-4", roles={"debate_creator"})
        mw = _make_middleware()
        decision = mw.enforce(ctx, path="/api/v1/debates")

        assert decision.allowed

    def test_list_roles_also_work(self):
        """Roles passed as a list (not set) should also work."""
        ctx = FakeUserContext(user_id="user-5")
        ctx.roles = ["member", "viewer"]  # type: ignore[assignment]
        mw = _make_middleware()
        decision = mw.enforce(ctx, path="/api/v1/debates")

        assert decision.allowed
        assert decision.result == MFAEnforcementResult.NOT_REQUIRED


# ---------------------------------------------------------------------------
# Grace period logic
# ---------------------------------------------------------------------------

class TestGracePeriod:
    """Newly-promoted admins get a grace period to set up MFA."""

    def test_within_grace_period_allowed(self):
        now = datetime.now(timezone.utc)
        recently_promoted = now - timedelta(hours=1)

        ctx = FakeUserContext(user_id="new-admin", roles={"admin"})
        status = MFAStatus(
            user_id="new-admin",
            mfa_enabled=False,
            role_assigned_at=recently_promoted,
        )
        mw = _make_middleware()
        decision = mw.enforce(ctx, path="/api/v1/admin/users", mfa_status=status)

        assert decision.allowed
        assert decision.result == MFAEnforcementResult.GRACE_PERIOD
        assert decision.grace_period_remaining_hours is not None
        assert decision.grace_period_remaining_hours > 0
        assert "grace period" in decision.reason.lower()

    def test_expired_grace_period_denied(self):
        long_ago = datetime.now(timezone.utc) - timedelta(hours=100)

        ctx = FakeUserContext(user_id="old-admin", roles={"admin"})
        status = MFAStatus(
            user_id="old-admin",
            mfa_enabled=False,
            role_assigned_at=long_ago,
        )
        mw = _make_middleware()
        decision = mw.enforce(ctx, path="/api/v1/admin/users", mfa_status=status)

        assert not decision.allowed
        assert decision.result == MFAEnforcementResult.DENIED

    def test_custom_grace_period_hours(self):
        now = datetime.now(timezone.utc)
        promoted_5h_ago = now - timedelta(hours=5)

        policy = MFAEnforcementPolicy(grace_period_hours=4)
        ctx = FakeUserContext(user_id="admin-gp", roles={"admin"})
        status = MFAStatus(
            user_id="admin-gp",
            mfa_enabled=False,
            role_assigned_at=promoted_5h_ago,
        )
        mw = _make_middleware(policy=policy)
        decision = mw.enforce(ctx, path="/api/v1/admin/users", mfa_status=status)

        assert not decision.allowed
        assert decision.result == MFAEnforcementResult.DENIED

    def test_zero_grace_period_no_grace(self):
        now = datetime.now(timezone.utc)
        just_promoted = now - timedelta(seconds=10)

        policy = MFAEnforcementPolicy(grace_period_hours=0)
        ctx = FakeUserContext(user_id="admin-no-gp", roles={"admin"})
        status = MFAStatus(
            user_id="admin-no-gp",
            mfa_enabled=False,
            role_assigned_at=just_promoted,
        )
        mw = _make_middleware(policy=policy)
        decision = mw.enforce(ctx, path="/api/v1/admin/users", mfa_status=status)

        assert not decision.allowed
        assert decision.result == MFAEnforcementResult.DENIED

    def test_no_role_assigned_at_no_grace(self):
        """Without role_assigned_at, grace period cannot be calculated."""
        ctx = FakeUserContext(user_id="admin-nots", roles={"admin"})
        status = MFAStatus(
            user_id="admin-nots",
            mfa_enabled=False,
            role_assigned_at=None,
        )
        mw = _make_middleware()
        decision = mw.enforce(ctx, path="/api/v1/admin/users", mfa_status=status)

        assert not decision.allowed
        assert decision.result == MFAEnforcementResult.DENIED

    def test_grace_period_with_string_datetime(self):
        """role_assigned_at as ISO string should be parsed correctly."""
        now = datetime.now(timezone.utc)
        recent = (now - timedelta(hours=2)).isoformat()

        ctx = FakeUserContext(user_id="admin-str", roles={"admin"})
        status = MFAStatus(
            user_id="admin-str",
            mfa_enabled=False,
            role_assigned_at=recent,  # type: ignore[arg-type]
        )
        mw = _make_middleware()
        decision = mw.enforce(ctx, path="/api/v1/admin/users", mfa_status=status)

        assert decision.allowed
        assert decision.result == MFAEnforcementResult.GRACE_PERIOD

    def test_grace_period_via_store(self):
        """Grace period should work when status is resolved from user store."""
        now = datetime.now(timezone.utc)
        recently_promoted = now - timedelta(hours=1)

        store = MagicMock()
        stored_user = FakeStoredUser(
            id="new-admin-2",
            mfa_enabled=False,
            mfa_grace_period_started_at=recently_promoted,
        )
        store.get_user_by_id.return_value = stored_user

        ctx = FakeUserContext(user_id="new-admin-2", roles={"admin"})
        mw = _make_middleware(user_store=store)
        decision = mw.enforce(ctx, path="/api/v1/admin/users")

        assert decision.allowed
        assert decision.result == MFAEnforcementResult.GRACE_PERIOD


# ---------------------------------------------------------------------------
# Exempt paths
# ---------------------------------------------------------------------------

class TestExemptPaths:
    """Certain paths should bypass MFA enforcement entirely."""

    @pytest.mark.parametrize("path", [
        "/health",
        "/healthz",
        "/ready",
        "/metrics",
        "/api/docs",
        "/openapi.json",
        "/api/auth/login",
        "/api/auth/mfa/verify",
        "/api/auth/mfa/setup",
    ])
    def test_default_exempt_paths(self, path: str):
        ctx = FakeUserContext(user_id="admin-1", roles={"admin"})
        mw = _make_middleware()
        decision = mw.enforce(ctx, path=path)

        assert decision.allowed
        assert decision.result == MFAEnforcementResult.EXEMPT_PATH

    @pytest.mark.parametrize("path", [
        "/api/auth/oauth/google",
        "/api/auth/oauth/github/callback",
        "/api/auth/sso/metadata",
    ])
    def test_default_exempt_prefixes(self, path: str):
        ctx = FakeUserContext(user_id="admin-1", roles={"admin"})
        mw = _make_middleware()
        decision = mw.enforce(ctx, path=path)

        assert decision.allowed
        assert decision.result == MFAEnforcementResult.EXEMPT_PATH

    def test_non_exempt_admin_path_enforced(self):
        ctx = FakeUserContext(user_id="admin-1", roles={"admin"})
        mw = _make_middleware()
        decision = mw.enforce(ctx, path="/api/v1/admin/users")

        assert not decision.allowed

    def test_custom_exempt_paths(self):
        policy = MFAEnforcementPolicy(
            exempt_paths=frozenset({"/custom/exempt"}),
            exempt_prefixes=frozenset(),
        )
        ctx = FakeUserContext(user_id="admin-1", roles={"admin"})
        mw = _make_middleware(policy=policy)

        decision = mw.enforce(ctx, path="/custom/exempt")
        assert decision.allowed
        assert decision.result == MFAEnforcementResult.EXEMPT_PATH

        decision2 = mw.enforce(ctx, path="/health")
        assert not decision2.allowed  # /health not in custom exempt set

    def test_empty_path_not_exempt(self):
        ctx = FakeUserContext(user_id="admin-1", roles={"admin"})
        mw = _make_middleware()
        decision = mw.enforce(ctx, path="")

        # Empty path with admin role still requires MFA
        assert not decision.allowed


# ---------------------------------------------------------------------------
# Service account bypass
# ---------------------------------------------------------------------------

class TestServiceAccountBypass:
    """Service accounts with valid bypass should be allowed."""

    def test_service_account_with_bypass_allowed(self):
        ctx = FakeUserContext(user_id="svc-1", roles={"admin"})
        status = MFAStatus(
            user_id="svc-1",
            mfa_enabled=False,
            is_service_account=True,
            has_valid_bypass=True,
        )
        mw = _make_middleware()
        decision = mw.enforce(ctx, path="/api/v1/admin/users", mfa_status=status)

        assert decision.allowed
        assert decision.result == MFAEnforcementResult.ALLOWED

    def test_service_account_without_bypass_denied(self):
        ctx = FakeUserContext(user_id="svc-2", roles={"admin"})
        status = MFAStatus(
            user_id="svc-2",
            mfa_enabled=False,
            is_service_account=True,
            has_valid_bypass=False,
        )
        mw = _make_middleware()
        decision = mw.enforce(ctx, path="/api/v1/admin/users", mfa_status=status)

        assert not decision.allowed
        assert decision.result == MFAEnforcementResult.DENIED


# ---------------------------------------------------------------------------
# Policy configuration
# ---------------------------------------------------------------------------

class TestPolicyConfiguration:
    """Test policy customization and the enabled toggle."""

    def test_enforcement_disabled(self):
        policy = MFAEnforcementPolicy(enabled=False)
        ctx = FakeUserContext(user_id="admin-1", roles={"admin"})
        mw = _make_middleware(policy=policy)
        decision = mw.enforce(ctx, path="/api/v1/admin/users")

        assert decision.allowed
        assert decision.result == MFAEnforcementResult.NOT_REQUIRED
        assert "disabled" in decision.reason.lower()

    def test_custom_required_roles(self):
        policy = MFAEnforcementPolicy(
            required_roles=frozenset({"custom_admin"}),
        )
        ctx = FakeUserContext(user_id="ca-1", roles={"custom_admin"})
        mw = _make_middleware(policy=policy)
        decision = mw.enforce(ctx, path="/api/v1/admin/users")

        assert not decision.allowed
        assert decision.result == MFAEnforcementResult.DENIED

    def test_standard_admin_not_required_with_custom_roles(self):
        policy = MFAEnforcementPolicy(
            required_roles=frozenset({"custom_admin"}),
        )
        ctx = FakeUserContext(user_id="admin-1", roles={"admin"})
        mw = _make_middleware(policy=policy)
        decision = mw.enforce(ctx, path="/api/v1/admin/users")

        assert decision.allowed
        assert decision.result == MFAEnforcementResult.NOT_REQUIRED

    def test_policy_is_immutable(self):
        policy = MFAEnforcementPolicy()
        with pytest.raises(AttributeError):
            policy.enabled = False  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

class TestCheckMFARequired:
    """Tests for the check_mfa_required helper."""

    def test_admin_requires_mfa(self):
        ctx = FakeUserContext(roles={"admin"})
        assert check_mfa_required(ctx) is True

    def test_owner_requires_mfa(self):
        ctx = FakeUserContext(roles={"owner"})
        assert check_mfa_required(ctx) is True

    def test_member_does_not_require(self):
        ctx = FakeUserContext(roles={"member"})
        assert check_mfa_required(ctx) is False

    def test_empty_roles(self):
        ctx = FakeUserContext(roles=set())
        assert check_mfa_required(ctx) is False

    def test_list_roles(self):
        ctx = FakeUserContext()
        ctx.roles = ["admin"]  # type: ignore[assignment]
        assert check_mfa_required(ctx) is True


class TestGetMFAStatus:
    """Tests for the get_mfa_status helper."""

    def test_no_store_returns_default(self):
        status = get_mfa_status("user-1")
        assert status.user_id == "user-1"
        assert status.mfa_enabled is False

    def test_with_store_returns_status(self):
        store = MagicMock()
        stored_user = FakeStoredUser(id="user-1", mfa_enabled=True)
        store.get_user_by_id.return_value = stored_user

        status = get_mfa_status("user-1", user_store=store)
        assert status.user_id == "user-1"
        assert status.mfa_enabled is True

    def test_store_user_not_found(self):
        store = MagicMock()
        store.get_user_by_id.return_value = None

        status = get_mfa_status("ghost-user", user_store=store)
        assert status.user_id == "ghost-user"
        assert status.mfa_enabled is False

    def test_store_raises_exception(self):
        store = MagicMock()
        store.get_user_by_id.side_effect = RuntimeError("DB down")

        status = get_mfa_status("user-1", user_store=store)
        assert status.user_id == "user-1"
        assert status.mfa_enabled is False

    def test_service_account_bypass_detection(self):
        store = MagicMock()
        stored_user = FakeStoredUser(
            id="svc-1",
            mfa_enabled=False,
            is_service_account=True,
            mfa_bypass_approved_at=datetime.now(timezone.utc),
        )
        store.get_user_by_id.return_value = stored_user

        status = get_mfa_status("svc-1", user_store=store)
        assert status.is_service_account is True
        assert status.has_valid_bypass is True


# ---------------------------------------------------------------------------
# Decision dataclass
# ---------------------------------------------------------------------------

class TestMFAEnforcementDecision:
    """Test decision defaults and structure."""

    def test_denied_decision_fields(self):
        decision = MFAEnforcementDecision(
            result=MFAEnforcementResult.DENIED,
            allowed=False,
            reason="MFA required",
            user_id="u-1",
            required_action="/api/auth/mfa/setup",
        )
        assert not decision.allowed
        assert decision.result == MFAEnforcementResult.DENIED
        assert decision.grace_period_remaining_hours is None

    def test_grace_period_decision_fields(self):
        decision = MFAEnforcementDecision(
            result=MFAEnforcementResult.GRACE_PERIOD,
            allowed=True,
            reason="Grace period active",
            grace_period_remaining_hours=24,
            user_id="u-1",
        )
        assert decision.allowed
        assert decision.grace_period_remaining_hours == 24


class TestDefaultMFARoles:
    """Verify the default required roles set."""

    def test_contains_expected_roles(self):
        expected = {"admin", "owner", "superadmin", "super_admin", "security_admin", "compliance_officer"}
        assert expected == set(DEFAULT_MFA_REQUIRED_ROLES)

    def test_is_frozenset(self):
        assert isinstance(DEFAULT_MFA_REQUIRED_ROLES, frozenset)
