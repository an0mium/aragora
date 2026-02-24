"""
Tests for MFA enforcement on admin sub-handlers.

Verifies that write operations on credits, feature flags, emergency access,
and security handlers reject requests when MFA is not verified for admin users.

SOC 2 Control: CC5-01 - Enforce MFA for administrative access.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@dataclass
class FakeAdminUser:
    """Mock admin user for MFA enforcement tests."""

    id: str = "admin-1"
    role: str = "admin"
    mfa_enabled: bool = False
    email: str = "admin@test.com"
    is_active: bool = True
    mfa_secret: str | None = None
    mfa_backup_codes: str | None = None
    mfa_grace_period_started_at: str | None = None
    created_at: str | None = None


class FakeUserStore:
    """User store that returns a single fake admin."""

    def __init__(self, user: FakeAdminUser | None = None):
        self._user = user or FakeAdminUser()

    def get_user_by_id(self, user_id: str) -> FakeAdminUser | None:
        if self._user.id == user_id:
            return self._user
        return None


def _make_settings(mfa_required: bool = True, grace_period_days: int = 2) -> MagicMock:
    """Create mock settings with MFA enforcement config."""
    settings = MagicMock()
    settings.security.admin_mfa_required = mfa_required
    settings.security.admin_mfa_grace_period_days = grace_period_days
    return settings


# -------------------------------------------------------------------------
# Credits handler MFA enforcement
# -------------------------------------------------------------------------


class TestCreditsHandlerMFA:
    """Credits admin handler rejects when MFA not verified."""

    @pytest.fixture()
    def _patch_settings(self):
        with patch(
            "aragora.config.settings.get_settings",
            return_value=_make_settings(),
        ):
            yield

    @pytest.fixture()
    def _patch_credit_manager(self):
        mock_manager = MagicMock()
        mock_tx = MagicMock()
        mock_tx.to_dict.return_value = {"id": "txn-1", "amount_cents": 100}
        mock_manager.issue_credit = AsyncMock(return_value=mock_tx)
        mock_manager.adjust_balance = AsyncMock(return_value=mock_tx)
        with patch(
            "aragora.server.handlers.admin.credits.get_credit_manager", return_value=mock_manager
        ):
            yield

    @pytest.mark.usefixtures("_patch_settings")
    @pytest.mark.asyncio
    async def test_issue_credit_rejects_without_mfa(self):
        from aragora.server.handlers.admin.credits import CreditsAdminHandler

        user = FakeAdminUser(mfa_enabled=False)
        store = FakeUserStore(user)
        handler = CreditsAdminHandler(ctx={"user_store": store})

        result = await handler.issue_credit(
            org_id="org-1",
            data={"amount_cents": 100, "type": "promotional", "description": "test credit"},
            user_id="admin-1",
        )
        assert result.status_code == 403

    @pytest.mark.usefixtures("_patch_settings", "_patch_credit_manager")
    @pytest.mark.asyncio
    async def test_issue_credit_allows_with_mfa(self):
        from aragora.server.handlers.admin.credits import CreditsAdminHandler

        user = FakeAdminUser(mfa_enabled=True)
        store = FakeUserStore(user)
        handler = CreditsAdminHandler(ctx={"user_store": store})

        result = await handler.issue_credit(
            org_id="org-1",
            data={"amount_cents": 100, "type": "promotional", "description": "test credit"},
            user_id="admin-1",
        )
        # Should proceed past MFA check (may fail on credit manager mock, but not 403)
        assert result.status_code != 403

    @pytest.mark.usefixtures("_patch_settings")
    @pytest.mark.asyncio
    async def test_adjust_balance_rejects_without_mfa(self):
        from aragora.server.handlers.admin.credits import CreditsAdminHandler

        user = FakeAdminUser(mfa_enabled=False)
        store = FakeUserStore(user)
        handler = CreditsAdminHandler(ctx={"user_store": store})

        result = await handler.adjust_balance(
            org_id="org-1",
            data={"amount_cents": 50, "description": "adjustment"},
            user_id="admin-1",
        )
        assert result.status_code == 403


# -------------------------------------------------------------------------
# Feature flags handler MFA enforcement
# -------------------------------------------------------------------------


class TestFeatureFlagsHandlerMFA:
    """Feature flags handler rejects PUT when MFA not verified."""

    @pytest.fixture()
    def _patch_settings(self):
        with patch(
            "aragora.config.settings.get_settings",
            return_value=_make_settings(),
        ):
            yield

    @pytest.mark.usefixtures("_patch_settings")
    def test_handle_put_rejects_without_mfa(self):
        from aragora.server.handlers.admin.feature_flags import FeatureFlagAdminHandler

        user = FakeAdminUser(mfa_enabled=False)
        store = FakeUserStore(user)
        handler = FeatureFlagAdminHandler(ctx={"user_store": store})

        result = handler.handle_put(
            path="/api/v1/admin/feature-flags/test_flag",
            query_params={},
            handler=MagicMock(),
            user=user,
        )
        assert result is not None
        assert result.status_code == 403


# -------------------------------------------------------------------------
# Emergency access handler MFA enforcement
# -------------------------------------------------------------------------


class TestEmergencyAccessHandlerMFA:
    """Emergency access handler rejects activate/deactivate when MFA not verified."""

    @pytest.fixture()
    def _patch_settings(self):
        with patch(
            "aragora.config.settings.get_settings",
            return_value=_make_settings(),
        ):
            yield

    @pytest.mark.usefixtures("_patch_settings")
    def test_activate_rejects_without_mfa(self):
        from aragora.server.handlers.admin.emergency_access import EmergencyAccessHandler

        user = FakeAdminUser(mfa_enabled=False)
        store = FakeUserStore(user)
        handler = EmergencyAccessHandler(ctx={"user_store": store})

        result = handler._activate(handler=MagicMock(), user=user)
        assert result.status_code == 403

    @pytest.mark.usefixtures("_patch_settings")
    def test_deactivate_rejects_without_mfa(self):
        from aragora.server.handlers.admin.emergency_access import EmergencyAccessHandler

        user = FakeAdminUser(mfa_enabled=False)
        store = FakeUserStore(user)
        handler = EmergencyAccessHandler(ctx={"user_store": store})

        result = handler._deactivate(handler=MagicMock(), user=user)
        assert result.status_code == 403


# -------------------------------------------------------------------------
# Security handler MFA enforcement
# -------------------------------------------------------------------------


class TestSecurityHandlerMFA:
    """Security handler rejects POST when MFA not verified."""

    @pytest.fixture()
    def _patch_settings(self):
        with patch(
            "aragora.config.settings.get_settings",
            return_value=_make_settings(),
        ):
            yield

    @pytest.mark.usefixtures("_patch_settings")
    def test_handle_post_rejects_without_mfa(self):
        from aragora.server.handlers.admin.security import SecurityHandler

        user = FakeAdminUser(mfa_enabled=False)
        store = FakeUserStore(user)
        mock_handler = MagicMock()
        mock_handler.auth_context = MagicMock()
        mock_handler.auth_context.user_id = "admin-1"

        sec_handler = SecurityHandler(ctx={"user_store": store})
        result = sec_handler.handle_post(
            path="/api/v1/admin/security/rotate-key",
            data={"dry_run": True},
            handler=mock_handler,
        )
        assert result is not None
        assert result.status_code == 403
