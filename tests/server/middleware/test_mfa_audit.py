"""Tests for MFA bypass audit logging.

Verifies that:
1. _audit_mfa_bypass emits events via the unified audit system
2. _audit_mfa_bypass falls back to logger.error when audit module is unavailable
3. Each MFA decorator emits an audit event on service-account bypass
4. Audit event details contain the expected fields
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.middleware.mfa import (
    _audit_mfa_bypass,
    _has_valid_mfa_bypass,
    require_admin_mfa,
    require_admin_with_mfa,
    require_mfa,
    require_mfa_fresh,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@dataclass
class MockUser:
    """Mock user for testing."""

    id: str = "svc-001"
    email: str = "svc@example.com"
    role: str = "admin"
    is_admin: bool = True
    mfa_enabled: bool = False
    mfa_secret: str | None = None
    mfa_backup_codes: str | None = None
    metadata: dict = field(default_factory=dict)
    is_service_account: bool = True
    mfa_bypass_approved_at: datetime | None = None
    mfa_bypass_expires_at: datetime | None = None


def _make_bypass_user(**kwargs) -> MockUser:
    """Create a service-account user with a valid MFA bypass."""
    defaults = {
        "id": "svc-001",
        "is_service_account": True,
        "mfa_bypass_approved_at": datetime.now(timezone.utc) - timedelta(days=1),
        "mfa_bypass_expires_at": datetime.now(timezone.utc) + timedelta(days=30),
    }
    defaults.update(kwargs)
    return MockUser(**defaults)


class MockUserStore:
    def __init__(self):
        self.users: dict[str, MockUser] = {}

    def get_user_by_id(self, user_id: str) -> MockUser | None:
        return self.users.get(user_id)


def _make_handler(user_store: MockUserStore):
    """Build a mock handler with a user store attached."""
    handler = MagicMock()
    handler.ctx = {"user_store": user_store}
    handler.headers = {}
    return handler


# ---------------------------------------------------------------------------
# _audit_mfa_bypass – unit tests
# ---------------------------------------------------------------------------


class TestAuditMfaBypass:
    """Tests for the _audit_mfa_bypass helper."""

    def test_dispatches_to_unified_audit(self):
        mock_audit = MagicMock()
        with patch(
            "aragora.server.middleware.mfa.audit_security",
            mock_audit,
            create=True,
        ):
            # Patch the import inside _audit_mfa_bypass
            with patch.dict(
                "sys.modules",
                {"aragora.audit.unified": MagicMock(audit_security=mock_audit)},
            ):
                _audit_mfa_bypass("svc-001", "require_mfa decorator bypass")

        mock_audit.assert_called_once()
        call_kwargs = mock_audit.call_args[1]
        assert call_kwargs["event_type"] == "mfa_bypass"
        assert call_kwargs["actor_id"] == "svc-001"
        assert call_kwargs["reason"] == "service_account_bypass"
        assert "require_mfa" in call_kwargs["details"]["operation"]

    def test_fallback_logs_error_when_audit_unavailable(self, caplog):
        """When audit module cannot be imported, a logger.error is emitted."""
        with patch.dict("sys.modules", {"aragora.audit.unified": None}):
            with caplog.at_level(logging.ERROR):
                _audit_mfa_bypass("svc-002", "test operation")

        error_messages = [r.message for r in caplog.records if r.levelno >= logging.ERROR]
        assert any("audit" in msg.lower() for msg in error_messages)
        assert any("svc-002" in msg for msg in error_messages)

    def test_event_details_contain_required_fields(self):
        mock_audit = MagicMock()
        with patch.dict(
            "sys.modules",
            {"aragora.audit.unified": MagicMock(audit_security=mock_audit)},
        ):
            _audit_mfa_bypass("svc-003", "sensitive_op")

        details = mock_audit.call_args[1]["details"]
        assert details["actor_id"] == "svc-003"
        assert details["operation"] == "sensitive_op"
        assert details["reason"] == "service_account_bypass"


# ---------------------------------------------------------------------------
# Decorator-level audit emission
# ---------------------------------------------------------------------------


class TestRequireMfaAuditEmission:
    """Verify that require_mfa emits an audit event when a bypass occurs."""

    def test_audit_emitted_on_bypass(self):
        user_store = MockUserStore()
        user_store.users["svc-001"] = _make_bypass_user()

        mock_audit_fn = MagicMock()

        @require_mfa
        def endpoint(handler=None, user=None):
            return {"ok": True}

        handler = _make_handler(user_store)

        with patch(
            "aragora.server.middleware.mfa.get_current_user",
            return_value=MagicMock(id="svc-001", role="user", metadata={}),
        ):
            with patch("aragora.server.middleware.mfa._audit_mfa_bypass", mock_audit_fn):
                result = endpoint(handler=handler)

        mock_audit_fn.assert_called_once_with("svc-001", "require_mfa decorator bypass")
        assert result == {"ok": True}


class TestRequireAdminMfaAuditEmission:
    """Verify that require_admin_mfa emits an audit event when a bypass occurs."""

    def test_audit_emitted_on_bypass(self):
        user_store = MockUserStore()
        user_store.users["svc-admin"] = _make_bypass_user(id="svc-admin", role="admin")

        mock_audit_fn = MagicMock()

        @require_admin_mfa
        def endpoint(handler=None, user=None):
            return {"ok": True}

        handler = _make_handler(user_store)

        with patch(
            "aragora.server.middleware.mfa.get_current_user",
            return_value=MagicMock(id="svc-admin", role="admin", metadata={}),
        ):
            with patch("aragora.server.middleware.mfa._audit_mfa_bypass", mock_audit_fn):
                result = endpoint(handler=handler)

        mock_audit_fn.assert_called_once_with("svc-admin", "require_admin_mfa decorator bypass")
        assert result == {"ok": True}


class TestRequireAdminWithMfaAuditEmission:
    """Verify that require_admin_with_mfa emits an audit event when a bypass occurs."""

    def test_audit_emitted_on_bypass(self):
        user_store = MockUserStore()
        user_store.users["svc-admin"] = _make_bypass_user(id="svc-admin", role="admin")

        mock_audit_fn = MagicMock()

        @require_admin_with_mfa
        def endpoint(handler=None, user=None):
            return {"ok": True}

        handler = _make_handler(user_store)

        with patch(
            "aragora.server.middleware.mfa.get_current_user",
            return_value=MagicMock(id="svc-admin", role="admin", is_admin=True, metadata={}),
        ):
            with patch("aragora.server.middleware.mfa._audit_mfa_bypass", mock_audit_fn):
                result = endpoint(handler=handler)

        mock_audit_fn.assert_called_once_with(
            "svc-admin",
            "require_admin_with_mfa decorator bypass for sensitive operation",
        )
        assert result == {"ok": True}


class TestRequireMfaFreshAuditEmission:
    """Verify that require_mfa_fresh emits an audit event when a bypass occurs."""

    def test_audit_emitted_on_bypass(self):
        user_store = MockUserStore()
        user_store.users["svc-fresh"] = _make_bypass_user(id="svc-fresh")

        mock_audit_fn = MagicMock()

        @require_mfa_fresh(max_age_minutes=10)
        def endpoint(handler=None, user=None):
            return {"ok": True}

        handler = _make_handler(user_store)

        with patch(
            "aragora.server.middleware.mfa.get_current_user",
            return_value=MagicMock(id="svc-fresh", role="user", metadata={}),
        ):
            with patch("aragora.server.middleware.mfa._audit_mfa_bypass", mock_audit_fn):
                result = endpoint(handler=handler)

        mock_audit_fn.assert_called_once_with("svc-fresh", "require_mfa_fresh decorator bypass")
        assert result == {"ok": True}


# ---------------------------------------------------------------------------
# No audit when bypass is NOT used
# ---------------------------------------------------------------------------


class TestNoAuditWhenNoBypass:
    """Verify that audit events are NOT emitted when MFA is enabled normally."""

    def test_no_audit_for_normal_mfa_user(self):
        user_store = MockUserStore()
        user_store.users["user-123"] = MockUser(
            id="user-123",
            is_service_account=False,
            mfa_enabled=True,
        )

        mock_audit_fn = MagicMock()

        @require_mfa
        def endpoint(handler=None, user=None):
            return {"ok": True}

        handler = _make_handler(user_store)

        with patch(
            "aragora.server.middleware.mfa.get_current_user",
            return_value=MagicMock(id="user-123", role="user", metadata={}),
        ):
            with patch("aragora.server.middleware.mfa._audit_mfa_bypass", mock_audit_fn):
                result = endpoint(handler=handler)

        mock_audit_fn.assert_not_called()
        assert result == {"ok": True}
