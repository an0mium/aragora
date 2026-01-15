"""
Tests for aragora.server.middleware.mfa - MFA enforcement middleware.

Tests cover:
- require_mfa decorator
- require_admin_mfa decorator
- require_admin_with_mfa decorator
- check_mfa_status function
- enforce_admin_mfa_policy function
- User store extraction

SOC 2 Control: CC5-01 - Enforce MFA for administrative access
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.middleware.mfa import (
    require_mfa,
    require_admin_mfa,
    require_admin_with_mfa,
    check_mfa_status,
    enforce_admin_mfa_policy,
    _get_user_store_from_handler,
)


# ===========================================================================
# Test Fixtures
# ===========================================================================


@dataclass
class MockUser:
    """Mock user for testing."""

    id: str = "user-123"
    email: str = "user@example.com"
    role: str = "user"
    is_admin: bool = False
    mfa_enabled: bool = False
    mfa_secret: Optional[str] = None
    mfa_backup_codes: Optional[str] = None
    created_at: Optional[datetime] = None
    metadata: dict = field(default_factory=dict)


class MockUserStore:
    """Mock user store for testing."""

    def __init__(self):
        self.users: dict[str, MockUser] = {}

    def get_user_by_id(self, user_id: str) -> Optional[MockUser]:
        return self.users.get(user_id)


def make_mock_handler(ctx: dict = None, headers: dict = None):
    """Create mock HTTP handler."""
    handler = MagicMock()
    handler.ctx = ctx or {}
    handler.headers = headers or {}
    return handler


def get_status(result) -> int:
    """Extract status code from result."""
    if hasattr(result, "status_code"):
        return result.status_code
    if isinstance(result, tuple):
        return result[1]
    return 0


def get_body(result) -> dict:
    """Extract body from result."""
    if hasattr(result, "body"):
        body = result.body
        if isinstance(body, bytes):
            return json.loads(body.decode("utf-8"))
        if isinstance(body, str):
            return json.loads(body)
        return body
    if isinstance(result, tuple):
        body = result[0]
        if isinstance(body, dict):
            return body
        return json.loads(body)
    return {}


def get_error_code(result) -> Optional[str]:
    """Extract error code from result body."""
    body = get_body(result)
    # Handle nested error structure: {"error": {"code": "...", "message": "..."}}
    if "error" in body and isinstance(body["error"], dict):
        return body["error"].get("code")
    return body.get("code")


# ===========================================================================
# Test _get_user_store_from_handler
# ===========================================================================


class TestGetUserStoreFromHandler:
    """Tests for _get_user_store_from_handler function."""

    def test_from_ctx_dict(self):
        user_store = MockUserStore()
        handler = MagicMock()
        handler.ctx = {"user_store": user_store}

        result = _get_user_store_from_handler(handler)
        assert result is user_store

    def test_from_ctx_object(self):
        user_store = MockUserStore()
        handler = MagicMock()
        handler.ctx = MagicMock()
        handler.ctx.user_store = user_store

        result = _get_user_store_from_handler(handler)
        assert result is user_store

    def test_from_server_ctx(self):
        user_store = MockUserStore()
        handler = MagicMock(spec=[])  # No ctx attribute
        handler.server = MagicMock()
        handler.server.ctx = {"user_store": user_store}

        result = _get_user_store_from_handler(handler)
        assert result is user_store

    def test_from_app_ctx(self):
        user_store = MockUserStore()
        handler = MagicMock(spec=[])  # No ctx attribute
        handler.app = MagicMock()
        handler.app.ctx = {"user_store": user_store}

        result = _get_user_store_from_handler(handler)
        assert result is user_store

    def test_returns_none_when_not_found(self):
        handler = MagicMock()
        handler.ctx = {}
        handler.server = MagicMock(spec=[])
        handler.app = MagicMock(spec=[])

        result = _get_user_store_from_handler(handler)
        assert result is None


# ===========================================================================
# Test check_mfa_status
# ===========================================================================


class TestCheckMfaStatus:
    """Tests for check_mfa_status function."""

    def test_no_user_store(self):
        status = check_mfa_status("user-123", None)

        assert status["mfa_enabled"] is False
        assert status["mfa_secret_set"] is False
        assert status["backup_codes_remaining"] == 0
        assert "error" in status

    def test_user_not_found(self):
        user_store = MockUserStore()

        status = check_mfa_status("nonexistent", user_store)

        assert status["mfa_enabled"] is False
        assert "error" in status

    def test_mfa_disabled(self):
        user_store = MockUserStore()
        user_store.users["user-123"] = MockUser(mfa_enabled=False)

        status = check_mfa_status("user-123", user_store)

        assert status["mfa_enabled"] is False
        assert status["mfa_secret_set"] is False

    def test_mfa_enabled_with_secret(self):
        user_store = MockUserStore()
        user_store.users["user-123"] = MockUser(
            mfa_enabled=True,
            mfa_secret="TESTSECRET123456"
        )

        status = check_mfa_status("user-123", user_store)

        assert status["mfa_enabled"] is True
        assert status["mfa_secret_set"] is True

    def test_mfa_with_backup_codes(self):
        user_store = MockUserStore()
        backup_codes = json.dumps(["hash1", "hash2", "hash3", "hash4", "hash5"])
        user_store.users["user-123"] = MockUser(
            mfa_enabled=True,
            mfa_secret="SECRET",
            mfa_backup_codes=backup_codes
        )

        status = check_mfa_status("user-123", user_store)

        assert status["mfa_enabled"] is True
        assert status["backup_codes_remaining"] == 5

    def test_invalid_backup_codes_json(self):
        user_store = MockUserStore()
        user_store.users["user-123"] = MockUser(
            mfa_enabled=True,
            mfa_backup_codes="invalid json"
        )

        status = check_mfa_status("user-123", user_store)

        assert status["backup_codes_remaining"] == 0


# ===========================================================================
# Test enforce_admin_mfa_policy
# ===========================================================================


class TestEnforceAdminMfaPolicy:
    """Tests for enforce_admin_mfa_policy function."""

    def test_non_admin_always_compliant(self):
        user = MagicMock()
        user.id = "user-123"
        user.role = "user"

        result = enforce_admin_mfa_policy(user, None)

        assert result is None  # Compliant

    def test_admin_with_mfa_compliant(self):
        user = MagicMock()
        user.id = "admin-123"
        user.role = "admin"

        user_store = MockUserStore()
        backup_codes = json.dumps(["h1", "h2", "h3", "h4", "h5"])
        user_store.users["admin-123"] = MockUser(
            id="admin-123",
            mfa_enabled=True,
            mfa_secret="SECRET",
            mfa_backup_codes=backup_codes
        )

        result = enforce_admin_mfa_policy(user, user_store)

        assert result is None  # Fully compliant

    def test_admin_low_backup_codes_warning(self):
        user = MagicMock()
        user.id = "admin-123"
        user.role = "admin"

        user_store = MockUserStore()
        backup_codes = json.dumps(["h1", "h2"])  # Only 2 codes
        user_store.users["admin-123"] = MockUser(
            id="admin-123",
            mfa_enabled=True,
            mfa_backup_codes=backup_codes
        )

        result = enforce_admin_mfa_policy(user, user_store)

        assert result is not None
        assert result["compliant"] is True
        assert "warning" in result
        assert result["backup_codes_remaining"] == 2

    def test_owner_role_checked(self):
        user = MagicMock()
        user.id = "owner-123"
        user.role = "owner"

        user_store = MockUserStore()
        user_store.users["owner-123"] = MockUser(
            id="owner-123",
            mfa_enabled=False
        )

        result = enforce_admin_mfa_policy(user, user_store)

        assert result is not None
        assert result["compliant"] is False

    def test_superadmin_role_checked(self):
        user = MagicMock()
        user.id = "super-123"
        user.role = "superadmin"

        user_store = MockUserStore()
        user_store.users["super-123"] = MockUser(
            id="super-123",
            mfa_enabled=False
        )

        result = enforce_admin_mfa_policy(user, user_store)

        assert result is not None
        assert result["compliant"] is False

    def test_admin_in_grace_period(self):
        user = MagicMock()
        user.id = "new-admin"
        user.role = "admin"

        user_store = MockUserStore()
        # Created 3 days ago
        created = datetime.now(timezone.utc) - timedelta(days=3)
        user_store.users["new-admin"] = MockUser(
            id="new-admin",
            mfa_enabled=False,
            created_at=created
        )

        result = enforce_admin_mfa_policy(user, user_store, grace_period_days=7)

        assert result is not None
        assert result["compliant"] is False
        assert result["enforced"] is False
        assert "grace_period_remaining_days" in result

    def test_admin_grace_period_expired(self):
        user = MagicMock()
        user.id = "old-admin"
        user.role = "admin"

        user_store = MockUserStore()
        # Created 10 days ago
        created = datetime.now(timezone.utc) - timedelta(days=10)
        user_store.users["old-admin"] = MockUser(
            id="old-admin",
            mfa_enabled=False,
            created_at=created
        )

        result = enforce_admin_mfa_policy(user, user_store, grace_period_days=7)

        assert result is not None
        assert result["compliant"] is False
        assert result["enforced"] is True

    def test_iso_string_created_at(self):
        user = MagicMock()
        user.id = "admin-iso"
        user.role = "admin"

        user_store = MockUserStore()
        # Created 3 days ago as ISO string
        created = (datetime.now(timezone.utc) - timedelta(days=3)).isoformat()
        mock_user = MockUser(id="admin-iso", mfa_enabled=False)
        mock_user.created_at = created
        user_store.users["admin-iso"] = mock_user

        result = enforce_admin_mfa_policy(user, user_store, grace_period_days=7)

        assert result is not None
        assert result["enforced"] is False  # Still in grace period


# ===========================================================================
# Test require_mfa Decorator
# ===========================================================================


class TestRequireMfaDecorator:
    """Tests for require_mfa decorator."""

    def test_no_handler_returns_500(self):
        @require_mfa
        def endpoint():
            return {"success": True}

        result = endpoint()

        assert get_status(result) == 500

    @patch("aragora.server.middleware.mfa.get_current_user")
    def test_unauthenticated_returns_401(self, mock_get_user):
        mock_get_user.return_value = None

        @require_mfa
        def endpoint(handler):
            return {"success": True}

        handler = make_mock_handler()
        result = endpoint(handler)

        assert get_status(result) == 401

    @patch("aragora.server.middleware.mfa.get_current_user")
    def test_mfa_disabled_returns_403(self, mock_get_user):
        mock_user = MagicMock()
        mock_user.id = "user-123"
        mock_user.metadata = {"mfa_enabled": False}
        mock_get_user.return_value = mock_user

        @require_mfa
        def endpoint(handler, user=None):
            return {"success": True}

        user_store = MockUserStore()
        user_store.users["user-123"] = MockUser(mfa_enabled=False)
        handler = make_mock_handler(ctx={"user_store": user_store})

        result = endpoint(handler)

        assert get_status(result) == 403
        assert get_error_code(result) == "MFA_REQUIRED"

    @patch("aragora.server.middleware.mfa.get_current_user")
    def test_mfa_enabled_allows_access(self, mock_get_user):
        mock_user = MagicMock()
        mock_user.id = "user-123"
        mock_user.metadata = {"mfa_enabled": True}
        mock_get_user.return_value = mock_user

        @require_mfa
        def endpoint(handler, user=None):
            return {"success": True, "user_id": user.id}

        user_store = MockUserStore()
        user_store.users["user-123"] = MockUser(mfa_enabled=True)
        handler = make_mock_handler(ctx={"user_store": user_store})

        result = endpoint(handler)

        assert result["success"] is True
        assert result["user_id"] == "user-123"


# ===========================================================================
# Test require_admin_mfa Decorator
# ===========================================================================


class TestRequireAdminMfaDecorator:
    """Tests for require_admin_mfa decorator."""

    @patch("aragora.server.middleware.mfa.get_current_user")
    def test_regular_user_allowed_without_mfa(self, mock_get_user):
        mock_user = MagicMock()
        mock_user.id = "user-123"
        mock_user.role = "user"
        mock_user.metadata = {}
        mock_get_user.return_value = mock_user

        @require_admin_mfa
        def endpoint(handler, user=None):
            return {"success": True}

        handler = make_mock_handler()
        result = endpoint(handler)

        # Regular users don't need MFA for this decorator
        assert result["success"] is True

    @patch("aragora.server.middleware.mfa.get_current_user")
    def test_admin_without_mfa_blocked(self, mock_get_user):
        mock_user = MagicMock()
        mock_user.id = "admin-123"
        mock_user.role = "admin"
        mock_user.metadata = {}
        mock_get_user.return_value = mock_user

        @require_admin_mfa
        def endpoint(handler, user=None):
            return {"success": True}

        user_store = MockUserStore()
        user_store.users["admin-123"] = MockUser(
            id="admin-123",
            role="admin",
            mfa_enabled=False
        )
        handler = make_mock_handler(ctx={"user_store": user_store})

        result = endpoint(handler)

        assert get_status(result) == 403
        assert get_error_code(result) == "ADMIN_MFA_REQUIRED"

    @patch("aragora.server.middleware.mfa.get_current_user")
    def test_admin_with_mfa_allowed(self, mock_get_user):
        mock_user = MagicMock()
        mock_user.id = "admin-123"
        mock_user.role = "admin"
        mock_user.metadata = {}
        mock_get_user.return_value = mock_user

        @require_admin_mfa
        def endpoint(handler, user=None):
            return {"success": True, "admin": True}

        user_store = MockUserStore()
        user_store.users["admin-123"] = MockUser(
            id="admin-123",
            role="admin",
            mfa_enabled=True
        )
        handler = make_mock_handler(ctx={"user_store": user_store})

        result = endpoint(handler)

        assert result["success"] is True
        assert result["admin"] is True

    @patch("aragora.server.middleware.mfa.get_current_user")
    def test_owner_without_mfa_blocked(self, mock_get_user):
        mock_user = MagicMock()
        mock_user.id = "owner-123"
        mock_user.role = "owner"
        mock_user.metadata = {}
        mock_get_user.return_value = mock_user

        @require_admin_mfa
        def endpoint(handler, user=None):
            return {"success": True}

        user_store = MockUserStore()
        user_store.users["owner-123"] = MockUser(
            id="owner-123",
            role="owner",
            mfa_enabled=False
        )
        handler = make_mock_handler(ctx={"user_store": user_store})

        result = endpoint(handler)

        assert get_status(result) == 403


# ===========================================================================
# Test require_admin_with_mfa Decorator
# ===========================================================================


class TestRequireAdminWithMfaDecorator:
    """Tests for require_admin_with_mfa decorator."""

    @patch("aragora.server.middleware.mfa.get_current_user")
    def test_non_admin_blocked(self, mock_get_user):
        mock_user = MagicMock()
        mock_user.id = "user-123"
        mock_user.is_admin = False
        mock_get_user.return_value = mock_user

        @require_admin_with_mfa
        def endpoint(handler, user=None):
            return {"success": True}

        handler = make_mock_handler()
        result = endpoint(handler)

        assert get_status(result) == 403

    @patch("aragora.server.middleware.mfa.get_current_user")
    def test_admin_without_mfa_blocked(self, mock_get_user):
        mock_user = MagicMock()
        mock_user.id = "admin-123"
        mock_user.is_admin = True
        mock_user.metadata = {}
        mock_get_user.return_value = mock_user

        @require_admin_with_mfa
        def endpoint(handler, user=None):
            return {"success": True}

        user_store = MockUserStore()
        user_store.users["admin-123"] = MockUser(
            id="admin-123",
            mfa_enabled=False
        )
        handler = make_mock_handler(ctx={"user_store": user_store})

        result = endpoint(handler)

        assert get_status(result) == 403
        assert get_error_code(result) == "MFA_REQUIRED"

    @patch("aragora.server.middleware.mfa.get_current_user")
    def test_admin_with_mfa_allowed(self, mock_get_user):
        mock_user = MagicMock()
        mock_user.id = "admin-123"
        mock_user.is_admin = True
        mock_user.metadata = {}
        mock_get_user.return_value = mock_user

        @require_admin_with_mfa
        def endpoint(handler, user=None):
            return {"success": True, "sensitive_op": True}

        user_store = MockUserStore()
        user_store.users["admin-123"] = MockUser(
            id="admin-123",
            mfa_enabled=True
        )
        handler = make_mock_handler(ctx={"user_store": user_store})

        result = endpoint(handler)

        assert result["success"] is True
        assert result["sensitive_op"] is True


# ===========================================================================
# Test Handler Argument Extraction
# ===========================================================================


class TestHandlerArgumentExtraction:
    """Tests for handler extraction from args/kwargs."""

    @patch("aragora.server.middleware.mfa.get_current_user")
    def test_handler_from_kwargs(self, mock_get_user):
        mock_user = MagicMock()
        mock_user.id = "user-123"
        mock_user.role = "user"
        mock_user.metadata = {}
        mock_get_user.return_value = mock_user

        @require_admin_mfa
        def endpoint(handler=None, user=None):
            return {"success": True}

        handler = make_mock_handler()
        result = endpoint(handler=handler)

        assert result["success"] is True

    @patch("aragora.server.middleware.mfa.get_current_user")
    def test_handler_from_args(self, mock_get_user):
        mock_user = MagicMock()
        mock_user.id = "user-123"
        mock_user.role = "user"
        mock_user.metadata = {}
        mock_get_user.return_value = mock_user

        @require_admin_mfa
        def endpoint(self_arg, handler, user=None):
            return {"success": True}

        handler = make_mock_handler()
        result = endpoint(object(), handler)

        assert result["success"] is True
