"""
Tests for Admin MFA Enforcement via admin_secure_endpoint.

Validates SOC 2 Control CC5-01: Enforce MFA for administrative access.

Tests the full admin_secure_endpoint flow:
  auth → admin role check → MFA policy enforcement → RBAC permission check

Also covers:
  - Service account MFA bypass
  - Grace period for new admins
  - MFA enforcement disabled via config
  - Low backup code warnings
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Shared test fixtures / mocks
# ---------------------------------------------------------------------------


@dataclass
class MockFullUser:
    """Mock user as returned by user store (billing model)."""

    id: str
    email: str
    role: str = "user"
    mfa_enabled: bool = False
    mfa_secret: str | None = None
    mfa_backup_codes: str | None = None
    created_at: str | None = None
    is_service_account: bool = False
    mfa_bypass_approved_at: str | None = None
    mfa_bypass_expires_at: str | None = None
    mfa_grace_period_started_at: str | None = None

    @property
    def is_admin(self) -> bool:
        return self.role in {"admin", "owner", "superadmin"}

    def is_mfa_bypass_valid(self) -> bool:
        if not self.is_service_account:
            return False
        if not self.mfa_bypass_approved_at:
            return False
        if self.mfa_bypass_expires_at:
            now = datetime.now(timezone.utc)
            expires = datetime.fromisoformat(self.mfa_bypass_expires_at)
            if now >= expires:
                return False
        return True


class MockUserStore:
    """Mock user store."""

    def __init__(self):
        self.users: dict[str, MockFullUser] = {}

    def add_user(self, user: MockFullUser):
        self.users[user.id] = user

    def get_user_by_id(self, user_id: str) -> MockFullUser | None:
        return self.users.get(user_id)


@dataclass
class MockLightUser:
    """Lightweight user as returned by get_current_user / middleware."""

    id: str
    email: str
    role: str = "user"
    plan: str = "free"
    metadata: dict = field(default_factory=dict)

    @property
    def is_admin(self) -> bool:
        return self.role in {"admin", "owner", "superadmin"}


class MockHandler:
    """Mock HTTP handler."""

    def __init__(
        self,
        user: MockLightUser | None = None,
        user_store: MockUserStore | None = None,
    ):
        self.headers = {"Authorization": f"Bearer test_token_{user.id}" if user else ""}
        self._user = user
        self.ctx = {"user_store": user_store}


def _get_status(result):
    if hasattr(result, "status_code"):
        return result.status_code
    return result[1]


def _get_body(result):
    if hasattr(result, "body"):
        return json.loads(result.body.decode("utf-8"))
    return result[0] if isinstance(result[0], dict) else json.loads(result[0])


# ---------------------------------------------------------------------------
# Tests: _has_valid_mfa_bypass (service account bypass)
# ---------------------------------------------------------------------------


class TestServiceAccountMFABypass:
    """Test MFA bypass for service accounts."""

    def test_none_user_returns_false(self):
        from aragora.server.middleware.mfa import _has_valid_mfa_bypass

        assert _has_valid_mfa_bypass(None) is False

    def test_regular_user_returns_false(self):
        from aragora.server.middleware.mfa import _has_valid_mfa_bypass

        user = MockFullUser(id="u1", email="u@e.com", is_service_account=False)
        assert _has_valid_mfa_bypass(user) is False

    def test_service_account_without_approval_returns_false(self):
        from aragora.server.middleware.mfa import _has_valid_mfa_bypass

        user = MockFullUser(
            id="svc1",
            email="svc@e.com",
            is_service_account=True,
            mfa_bypass_approved_at=None,
        )
        assert _has_valid_mfa_bypass(user) is False

    def test_service_account_with_valid_bypass(self):
        from aragora.server.middleware.mfa import _has_valid_mfa_bypass

        user = MockFullUser(
            id="svc1",
            email="svc@e.com",
            is_service_account=True,
            mfa_bypass_approved_at=datetime.now(timezone.utc).isoformat(),
            mfa_bypass_expires_at=None,
        )
        assert _has_valid_mfa_bypass(user) is True

    def test_service_account_with_expired_bypass(self):
        from aragora.server.middleware.mfa import _has_valid_mfa_bypass

        past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        user = MockFullUser(
            id="svc1",
            email="svc@e.com",
            is_service_account=True,
            mfa_bypass_approved_at=past,
            mfa_bypass_expires_at=past,
        )
        assert _has_valid_mfa_bypass(user) is False

    def test_service_account_with_future_expiry(self):
        from aragora.server.middleware.mfa import _has_valid_mfa_bypass

        future = (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()
        user = MockFullUser(
            id="svc1",
            email="svc@e.com",
            is_service_account=True,
            mfa_bypass_approved_at=datetime.now(timezone.utc).isoformat(),
            mfa_bypass_expires_at=future,
        )
        assert _has_valid_mfa_bypass(user) is True


# ---------------------------------------------------------------------------
# Tests: require_admin_mfa decorator with service account bypass
# ---------------------------------------------------------------------------


class TestRequireAdminMFAWithBypass:
    """Test require_admin_mfa decorator with service account bypass."""

    def test_service_account_admin_bypasses_mfa(self):
        """Service account admins with valid bypass should access admin endpoints."""
        from aragora.server.middleware.mfa import require_admin_mfa

        @require_admin_mfa
        def admin_endpoint(handler, user):
            return {"success": True}, 200, {}

        svc_user = MockFullUser(
            id="svc_admin",
            email="svc@example.com",
            role="admin",
            mfa_enabled=False,
            is_service_account=True,
            mfa_bypass_approved_at=datetime.now(timezone.utc).isoformat(),
        )
        light_user = MockLightUser(id="svc_admin", email="svc@example.com", role="admin")
        user_store = MockUserStore()
        user_store.add_user(svc_user)
        handler = MockHandler(user=light_user, user_store=user_store)

        with patch("aragora.server.middleware.mfa.get_current_user") as mock_get_user:
            mock_get_user.return_value = light_user
            with patch("aragora.server.middleware.mfa._get_user_store_from_handler") as mock_store:
                mock_store.return_value = user_store
                result = admin_endpoint(handler)

        assert result[1] == 200

    def test_service_account_with_expired_bypass_blocked(self):
        """Service accounts with expired bypass should be blocked like normal users."""
        from aragora.server.middleware.mfa import require_admin_mfa

        @require_admin_mfa
        def admin_endpoint(handler, user):
            return {"success": True}

        past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        svc_user = MockFullUser(
            id="svc_admin",
            email="svc@example.com",
            role="admin",
            mfa_enabled=False,
            is_service_account=True,
            mfa_bypass_approved_at=past,
            mfa_bypass_expires_at=past,
        )
        light_user = MockLightUser(id="svc_admin", email="svc@example.com", role="admin")
        user_store = MockUserStore()
        user_store.add_user(svc_user)
        handler = MockHandler(user=light_user, user_store=user_store)

        with patch("aragora.server.middleware.mfa.get_current_user") as mock_get_user:
            mock_get_user.return_value = light_user
            with patch("aragora.server.middleware.mfa._get_user_store_from_handler") as mock_store:
                mock_store.return_value = user_store
                result = admin_endpoint(handler)

        assert _get_status(result) == 403


# ---------------------------------------------------------------------------
# Tests: enforce_admin_mfa_policy (grace period + config)
# ---------------------------------------------------------------------------


class TestEnforceAdminMFAPolicyExtended:
    """Extended tests for enforce_admin_mfa_policy covering grace period and config."""

    def _mock_settings(self, *, admin_mfa_required: bool = True, grace_days: int = 7):
        settings = MagicMock()
        settings.security.admin_mfa_required = admin_mfa_required
        settings.security.admin_mfa_grace_period_days = grace_days
        return settings

    def test_mfa_enforcement_disabled_returns_none(self):
        """When admin_mfa_required is False, all users pass."""
        from aragora.server.middleware.mfa import enforce_admin_mfa_policy

        user = MockLightUser(id="admin_1", email="admin@e.com", role="admin")
        full_user = MockFullUser(id="admin_1", email="admin@e.com", role="admin", mfa_enabled=False)
        user_store = MockUserStore()
        user_store.add_user(full_user)

        settings = self._mock_settings(admin_mfa_required=False)
        with patch("aragora.config.settings.get_settings", return_value=settings):
            result = enforce_admin_mfa_policy(user, user_store)

        assert result is None

    def test_admin_within_grace_period_not_enforced(self):
        """Admin within grace period should get non-enforced result."""
        from aragora.server.middleware.mfa import enforce_admin_mfa_policy

        # User created 2 days ago, grace period is 7 days
        created = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
        user = MockLightUser(id="admin_1", email="admin@e.com", role="admin")
        full_user = MockFullUser(
            id="admin_1",
            email="admin@e.com",
            role="admin",
            mfa_enabled=False,
            created_at=created,
        )
        user_store = MockUserStore()
        user_store.add_user(full_user)

        settings = self._mock_settings(admin_mfa_required=True, grace_days=7)
        with patch("aragora.config.settings.get_settings", return_value=settings):
            result = enforce_admin_mfa_policy(user, user_store)

        assert result is not None
        assert result["compliant"] is False
        assert result["enforced"] is False
        assert result["grace_period_remaining_days"] >= 4

    def test_admin_past_grace_period_enforced(self):
        """Admin past grace period should have MFA enforced."""
        from aragora.server.middleware.mfa import enforce_admin_mfa_policy

        # User created 30 days ago, grace period is 7 days
        created = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        user = MockLightUser(id="admin_1", email="admin@e.com", role="admin")
        full_user = MockFullUser(
            id="admin_1",
            email="admin@e.com",
            role="admin",
            mfa_enabled=False,
            created_at=created,
        )
        user_store = MockUserStore()
        user_store.add_user(full_user)

        settings = self._mock_settings(admin_mfa_required=True, grace_days=7)
        with patch("aragora.config.settings.get_settings", return_value=settings):
            result = enforce_admin_mfa_policy(user, user_store)

        assert result is not None
        assert result["compliant"] is False
        assert result["enforced"] is True

    def test_grace_period_uses_dedicated_field(self):
        """Grace period should prefer mfa_grace_period_started_at over created_at."""
        from aragora.server.middleware.mfa import enforce_admin_mfa_policy

        # created_at is 30 days ago (past grace period)
        # but mfa_grace_period_started_at is 1 day ago (within grace period)
        old = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        recent = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()

        user = MockLightUser(id="admin_1", email="admin@e.com", role="admin")
        full_user = MockFullUser(
            id="admin_1",
            email="admin@e.com",
            role="admin",
            mfa_enabled=False,
            created_at=old,
            mfa_grace_period_started_at=recent,
        )
        user_store = MockUserStore()
        user_store.add_user(full_user)

        settings = self._mock_settings(admin_mfa_required=True, grace_days=7)
        with patch("aragora.config.settings.get_settings", return_value=settings):
            result = enforce_admin_mfa_policy(user, user_store)

        assert result is not None
        assert result["compliant"] is False
        assert result["enforced"] is False  # Within grace period
        assert result["grace_period_remaining_days"] >= 5

    def test_non_admin_roles_always_pass(self):
        """Non-admin roles (member, team_lead, viewer) should always pass."""
        from aragora.server.middleware.mfa import enforce_admin_mfa_policy

        settings = self._mock_settings(admin_mfa_required=True)
        for role in ["user", "member", "team_lead", "viewer"]:
            user = MockLightUser(id=f"{role}_1", email=f"{role}@e.com", role=role)
            user_store = MockUserStore()

            with patch("aragora.config.settings.get_settings", return_value=settings):
                result = enforce_admin_mfa_policy(user, user_store)

            assert result is None, f"Role '{role}' should always be compliant"

    def test_superadmin_without_mfa_enforced(self):
        """Superadmin role should also require MFA."""
        from aragora.server.middleware.mfa import enforce_admin_mfa_policy

        user = MockLightUser(id="sa_1", email="sa@e.com", role="superadmin")
        full_user = MockFullUser(
            id="sa_1",
            email="sa@e.com",
            role="superadmin",
            mfa_enabled=False,
        )
        user_store = MockUserStore()
        user_store.add_user(full_user)

        settings = self._mock_settings(admin_mfa_required=True, grace_days=0)
        with patch("aragora.config.settings.get_settings", return_value=settings):
            result = enforce_admin_mfa_policy(user, user_store)

        assert result is not None
        assert result["compliant"] is False
        assert result["enforced"] is True

    def test_admin_with_mfa_and_enough_backup_codes(self):
        """Admin with MFA and sufficient backup codes is fully compliant."""
        from aragora.server.middleware.mfa import enforce_admin_mfa_policy

        codes = json.dumps(["h1", "h2", "h3", "h4", "h5"])
        user = MockLightUser(id="admin_1", email="admin@e.com", role="admin")
        full_user = MockFullUser(
            id="admin_1",
            email="admin@e.com",
            role="admin",
            mfa_enabled=True,
            mfa_backup_codes=codes,
        )
        user_store = MockUserStore()
        user_store.add_user(full_user)

        settings = self._mock_settings(admin_mfa_required=True)
        with patch("aragora.config.settings.get_settings", return_value=settings):
            result = enforce_admin_mfa_policy(user, user_store)

        assert result is None  # Fully compliant

    def test_admin_with_mfa_low_backup_codes_warning(self):
        """Admin with MFA but <3 backup codes gets a warning (still compliant)."""
        from aragora.server.middleware.mfa import enforce_admin_mfa_policy

        codes = json.dumps(["h1"])
        user = MockLightUser(id="admin_1", email="admin@e.com", role="admin")
        full_user = MockFullUser(
            id="admin_1",
            email="admin@e.com",
            role="admin",
            mfa_enabled=True,
            mfa_backup_codes=codes,
        )
        user_store = MockUserStore()
        user_store.add_user(full_user)

        settings = self._mock_settings(admin_mfa_required=True)
        with patch("aragora.config.settings.get_settings", return_value=settings):
            result = enforce_admin_mfa_policy(user, user_store)

        assert result is not None
        assert result["compliant"] is True
        assert result["warning"] == "Low backup codes"


# ---------------------------------------------------------------------------
# Tests: require_mfa decorator with service account bypass
# ---------------------------------------------------------------------------


class TestRequireMFAWithBypass:
    """Test require_mfa decorator with service account bypass."""

    def test_service_account_bypasses_mfa_requirement(self):
        """Service account with valid bypass should pass require_mfa."""
        from aragora.server.middleware.mfa import require_mfa

        @require_mfa
        def protected(handler, user):
            return {"ok": True}, 200, {}

        svc_user = MockFullUser(
            id="svc1",
            email="svc@e.com",
            mfa_enabled=False,
            is_service_account=True,
            mfa_bypass_approved_at=datetime.now(timezone.utc).isoformat(),
        )
        light_user = MockLightUser(id="svc1", email="svc@e.com")
        user_store = MockUserStore()
        user_store.add_user(svc_user)
        handler = MockHandler(user=light_user, user_store=user_store)

        with patch("aragora.server.middleware.mfa.get_current_user") as mock_get_user:
            mock_get_user.return_value = light_user
            with patch("aragora.server.middleware.mfa._get_user_store_from_handler") as mock_store:
                mock_store.return_value = user_store
                result = protected(handler)

        assert result[1] == 200


# ---------------------------------------------------------------------------
# Tests: require_mfa_fresh decorator
# ---------------------------------------------------------------------------


class TestRequireMFAFresh:
    """Test the require_mfa_fresh step-up auth decorator."""

    def test_user_without_mfa_rejected(self):
        """Users without MFA enabled are rejected."""
        from aragora.server.middleware.mfa import require_mfa_fresh

        @require_mfa_fresh(max_age_minutes=15)
        def sensitive(handler, user):
            return {"ok": True}

        user = MockFullUser(id="u1", email="u@e.com", mfa_enabled=False)
        light_user = MockLightUser(id="u1", email="u@e.com")
        user_store = MockUserStore()
        user_store.add_user(user)
        handler = MockHandler(user=light_user, user_store=user_store)

        with patch("aragora.server.middleware.mfa.get_current_user") as mock_gu:
            mock_gu.return_value = light_user
            with patch("aragora.server.middleware.mfa._get_user_store_from_handler") as mock_store:
                mock_store.return_value = user_store
                result = sensitive(handler)

        assert _get_status(result) == 403
        body = _get_body(result)
        assert body["error"]["code"] == "MFA_REQUIRED"

    def test_service_account_bypasses_freshness(self):
        """Service accounts with valid bypass skip MFA freshness check."""
        from aragora.server.middleware.mfa import require_mfa_fresh

        @require_mfa_fresh(max_age_minutes=5)
        def sensitive(handler, user):
            return {"ok": True}, 200, {}

        svc_user = MockFullUser(
            id="svc1",
            email="svc@e.com",
            mfa_enabled=False,
            is_service_account=True,
            mfa_bypass_approved_at=datetime.now(timezone.utc).isoformat(),
        )
        light_user = MockLightUser(id="svc1", email="svc@e.com")
        user_store = MockUserStore()
        user_store.add_user(svc_user)
        handler = MockHandler(user=light_user, user_store=user_store)

        with patch("aragora.server.middleware.mfa.get_current_user") as mock_gu:
            mock_gu.return_value = light_user
            with patch("aragora.server.middleware.mfa._get_user_store_from_handler") as mock_store:
                mock_store.return_value = user_store
                result = sensitive(handler)

        assert result[1] == 200
