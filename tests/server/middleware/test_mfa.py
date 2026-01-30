"""
Tests for aragora.server.middleware.mfa - MFA enforcement middleware.

Comprehensive tests covering:
- _has_valid_mfa_bypass: Service account bypass logic
- require_mfa decorator: MFA enforcement for all users
- require_admin_mfa decorator: MFA enforcement for admin roles only
- require_admin_with_mfa decorator: Combined admin + MFA gate
- require_mfa_fresh decorator: Step-up MFA freshness checks
- check_mfa_status function: MFA status inspection
- enforce_admin_mfa_policy function: SOC 2 CC5-01 compliance policy
- _get_user_store_from_handler: Handler context extraction
- _get_session_manager_from_handler: Session manager extraction

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
    _get_user_store_from_handler,
    _has_valid_mfa_bypass,
    check_mfa_status,
    enforce_admin_mfa_policy,
    require_admin_mfa,
    require_admin_with_mfa,
    require_mfa,
    require_mfa_fresh,
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
    mfa_grace_period_started_at: Optional[datetime] = None
    metadata: dict = field(default_factory=dict)
    # Service account bypass fields
    is_service_account: bool = False
    mfa_bypass_approved_at: Optional[datetime] = None
    mfa_bypass_expires_at: Optional[datetime] = None


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
# Test _has_valid_mfa_bypass
# ===========================================================================


class TestHasValidMfaBypass:
    """Tests for _has_valid_mfa_bypass helper."""

    def test_none_user_returns_false(self):
        assert _has_valid_mfa_bypass(None) is False

    def test_non_service_account_returns_false(self):
        user = MockUser(is_service_account=False)
        assert _has_valid_mfa_bypass(user) is False

    def test_service_account_without_approval_returns_false(self):
        user = MockUser(is_service_account=True, mfa_bypass_approved_at=None)
        assert _has_valid_mfa_bypass(user) is False

    def test_service_account_with_approval_no_expiry(self):
        user = MockUser(
            is_service_account=True,
            mfa_bypass_approved_at=datetime.now(timezone.utc),
            mfa_bypass_expires_at=None,
        )
        assert _has_valid_mfa_bypass(user) is True

    def test_service_account_with_future_expiry(self):
        user = MockUser(
            is_service_account=True,
            mfa_bypass_approved_at=datetime.now(timezone.utc) - timedelta(days=1),
            mfa_bypass_expires_at=datetime.now(timezone.utc) + timedelta(days=30),
        )
        assert _has_valid_mfa_bypass(user) is True

    def test_service_account_with_expired_bypass(self):
        user = MockUser(
            is_service_account=True,
            mfa_bypass_approved_at=datetime.now(timezone.utc) - timedelta(days=60),
            mfa_bypass_expires_at=datetime.now(timezone.utc) - timedelta(days=1),
        )
        assert _has_valid_mfa_bypass(user) is False

    def test_service_account_with_string_expiry(self):
        future = (datetime.now(timezone.utc) + timedelta(days=10)).isoformat()
        user = MockUser(
            is_service_account=True,
            mfa_bypass_approved_at=datetime.now(timezone.utc),
        )
        user.mfa_bypass_expires_at = future
        assert _has_valid_mfa_bypass(user) is True

    def test_service_account_with_expired_string_expiry(self):
        past = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
        user = MockUser(
            is_service_account=True,
            mfa_bypass_approved_at=datetime.now(timezone.utc) - timedelta(days=30),
        )
        user.mfa_bypass_expires_at = past
        assert _has_valid_mfa_bypass(user) is False

    def test_service_account_with_is_mfa_bypass_valid_method(self):
        """Test that the model method is preferred when available."""
        user = MagicMock()
        user.is_service_account = True
        user.is_mfa_bypass_valid.return_value = True
        assert _has_valid_mfa_bypass(user) is True

    def test_service_account_model_method_returns_false(self):
        user = MagicMock()
        user.is_service_account = True
        user.is_mfa_bypass_valid.return_value = False
        assert _has_valid_mfa_bypass(user) is False

    def test_user_without_service_account_attr(self):
        """Test object that doesn't have is_service_account at all."""
        user = MagicMock(spec=[])  # No attributes
        assert _has_valid_mfa_bypass(user) is False


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

    def test_from_app_ctx_dict(self):
        user_store = MockUserStore()
        handler = MagicMock(spec=[])  # No ctx attribute
        handler.app = MagicMock()
        handler.app.ctx = {"user_store": user_store}

        result = _get_user_store_from_handler(handler)
        assert result is user_store

    def test_from_app_user_store(self):
        user_store = MockUserStore()
        handler = MagicMock(spec=[])
        handler.app = MagicMock()
        handler.app.ctx = MagicMock(spec=[])  # ctx without get
        handler.app.user_store = user_store

        result = _get_user_store_from_handler(handler)
        assert result is user_store

    def test_returns_none_when_not_found(self):
        handler = MagicMock()
        handler.ctx = {}
        handler.server = MagicMock(spec=[])
        handler.app = MagicMock(spec=[])

        result = _get_user_store_from_handler(handler)
        assert result is None

    def test_ctx_dict_without_user_store_key(self):
        handler = MagicMock()
        handler.ctx = {"other_key": "value"}
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
        user_store.users["user-123"] = MockUser(mfa_enabled=True, mfa_secret="TESTSECRET123456")

        status = check_mfa_status("user-123", user_store)

        assert status["mfa_enabled"] is True
        assert status["mfa_secret_set"] is True

    def test_mfa_enabled_without_secret(self):
        user_store = MockUserStore()
        user_store.users["user-123"] = MockUser(mfa_enabled=True, mfa_secret=None)

        status = check_mfa_status("user-123", user_store)

        assert status["mfa_enabled"] is True
        assert status["mfa_secret_set"] is False

    def test_mfa_with_backup_codes(self):
        user_store = MockUserStore()
        backup_codes = json.dumps(["hash1", "hash2", "hash3", "hash4", "hash5"])
        user_store.users["user-123"] = MockUser(
            mfa_enabled=True, mfa_secret="SECRET", mfa_backup_codes=backup_codes
        )

        status = check_mfa_status("user-123", user_store)

        assert status["mfa_enabled"] is True
        assert status["backup_codes_remaining"] == 5

    def test_mfa_with_empty_backup_codes(self):
        user_store = MockUserStore()
        user_store.users["user-123"] = MockUser(mfa_enabled=True, mfa_backup_codes=json.dumps([]))

        status = check_mfa_status("user-123", user_store)

        assert status["backup_codes_remaining"] == 0

    def test_invalid_backup_codes_json(self):
        user_store = MockUserStore()
        user_store.users["user-123"] = MockUser(mfa_enabled=True, mfa_backup_codes="invalid json")

        status = check_mfa_status("user-123", user_store)

        assert status["backup_codes_remaining"] == 0

    def test_null_backup_codes(self):
        user_store = MockUserStore()
        user_store.users["user-123"] = MockUser(mfa_enabled=True, mfa_backup_codes=None)

        status = check_mfa_status("user-123", user_store)

        assert status["backup_codes_remaining"] == 0

    def test_no_error_field_on_success(self):
        user_store = MockUserStore()
        user_store.users["user-123"] = MockUser(mfa_enabled=True)

        status = check_mfa_status("user-123", user_store)

        assert "error" not in status


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
            id="admin-123", mfa_enabled=True, mfa_secret="SECRET", mfa_backup_codes=backup_codes
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
            id="admin-123", mfa_enabled=True, mfa_backup_codes=backup_codes
        )

        result = enforce_admin_mfa_policy(user, user_store)

        assert result is not None
        assert result["compliant"] is True
        assert "warning" in result
        assert result["backup_codes_remaining"] == 2

    def test_admin_zero_backup_codes_warning(self):
        user = MagicMock()
        user.id = "admin-123"
        user.role = "admin"

        user_store = MockUserStore()
        user_store.users["admin-123"] = MockUser(
            id="admin-123", mfa_enabled=True, mfa_backup_codes=json.dumps([])
        )

        result = enforce_admin_mfa_policy(user, user_store)

        assert result is not None
        assert result["compliant"] is True
        assert result["backup_codes_remaining"] == 0

    def test_owner_role_checked(self):
        user = MagicMock()
        user.id = "owner-123"
        user.role = "owner"

        user_store = MockUserStore()
        user_store.users["owner-123"] = MockUser(id="owner-123", mfa_enabled=False)

        result = enforce_admin_mfa_policy(user, user_store)

        assert result is not None
        assert result["compliant"] is False

    def test_superadmin_role_checked(self):
        user = MagicMock()
        user.id = "super-123"
        user.role = "superadmin"

        user_store = MockUserStore()
        user_store.users["super-123"] = MockUser(id="super-123", mfa_enabled=False)

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
            id="new-admin", mfa_enabled=False, created_at=created
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
            id="old-admin", mfa_enabled=False, created_at=created
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

    def test_iso_string_with_z_suffix(self):
        user = MagicMock()
        user.id = "admin-z"
        user.role = "admin"

        user_store = MockUserStore()
        created = (datetime.now(timezone.utc) - timedelta(days=2)).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        mock_user = MockUser(id="admin-z", mfa_enabled=False)
        mock_user.created_at = created
        user_store.users["admin-z"] = mock_user

        result = enforce_admin_mfa_policy(user, user_store, grace_period_days=7)

        assert result is not None
        assert result["enforced"] is False

    def test_mfa_grace_period_started_at_preferred_over_created_at(self):
        """mfa_grace_period_started_at should be used when available."""
        user = MagicMock()
        user.id = "admin-gp"
        user.role = "admin"

        user_store = MockUserStore()
        # created_at was long ago, but grace period started recently
        mock_user = MockUser(
            id="admin-gp",
            mfa_enabled=False,
            created_at=datetime.now(timezone.utc) - timedelta(days=365),
            mfa_grace_period_started_at=datetime.now(timezone.utc) - timedelta(days=1),
        )
        user_store.users["admin-gp"] = mock_user

        result = enforce_admin_mfa_policy(user, user_store, grace_period_days=7)

        assert result is not None
        assert result["enforced"] is False  # Still within grace from the recent start

    @patch("aragora.server.middleware.mfa.get_settings")
    def test_mfa_enforcement_disabled_in_settings(self, mock_settings):
        settings = MagicMock()
        settings.security.admin_mfa_required = False
        mock_settings.return_value = settings

        user = MagicMock()
        user.id = "admin-123"
        user.role = "admin"

        result = enforce_admin_mfa_policy(user, None)

        assert result is None  # Enforcement disabled

    def test_admin_no_user_store_no_grace(self):
        """Admin without user store gets enforced result."""
        user = MagicMock()
        user.id = "admin-no-store"
        user.role = "admin"

        result = enforce_admin_mfa_policy(user, None)

        assert result is not None
        assert result["compliant"] is False
        assert result["enforced"] is True


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

    @patch("aragora.server.middleware.mfa.get_current_user")
    def test_mfa_fallback_to_metadata_enabled(self, mock_get_user):
        """When no user store is available, falls back to user metadata."""
        mock_user = MagicMock()
        mock_user.id = "user-meta"
        mock_user.metadata = {"mfa_enabled": True}
        mock_get_user.return_value = mock_user

        @require_mfa
        def endpoint(handler, user=None):
            return {"success": True}

        # No user_store in context
        handler = make_mock_handler(ctx={})
        result = endpoint(handler)

        assert result["success"] is True

    @patch("aragora.server.middleware.mfa.get_current_user")
    def test_mfa_fallback_to_metadata_disabled(self, mock_get_user):
        """When no user store and metadata shows mfa_enabled=False."""
        mock_user = MagicMock()
        mock_user.id = "user-meta"
        mock_user.metadata = {"mfa_enabled": False}
        mock_get_user.return_value = mock_user

        @require_mfa
        def endpoint(handler, user=None):
            return {"success": True}

        handler = make_mock_handler(ctx={})
        result = endpoint(handler)

        assert get_status(result) == 403

    @patch("aragora.server.middleware.mfa.get_current_user")
    def test_service_account_bypass_in_require_mfa(self, mock_get_user):
        """Service accounts with valid bypass skip MFA requirement."""
        mock_user = MagicMock()
        mock_user.id = "svc-123"
        mock_user.metadata = {}
        mock_get_user.return_value = mock_user

        @require_mfa
        def endpoint(handler, user=None):
            return {"success": True, "user_id": user.id}

        user_store = MockUserStore()
        user_store.users["svc-123"] = MockUser(
            id="svc-123",
            mfa_enabled=False,
            is_service_account=True,
            mfa_bypass_approved_at=datetime.now(timezone.utc),
            mfa_bypass_expires_at=None,
        )
        handler = make_mock_handler(ctx={"user_store": user_store})

        result = endpoint(handler)

        assert result["success"] is True
        assert result["user_id"] == "svc-123"

    @patch("aragora.server.middleware.mfa.get_current_user")
    def test_user_in_store_but_mfa_not_set(self, mock_get_user):
        """User exists in store but doesn't have mfa_enabled attribute."""
        mock_user = MagicMock()
        mock_user.id = "user-no-attr"
        mock_user.metadata = {}
        mock_get_user.return_value = mock_user

        @require_mfa
        def endpoint(handler, user=None):
            return {"success": True}

        user_store = MockUserStore()
        # MockUser defaults to mfa_enabled=False
        user_store.users["user-no-attr"] = MockUser(id="user-no-attr")
        handler = make_mock_handler(ctx={"user_store": user_store})

        result = endpoint(handler)

        assert get_status(result) == 403


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
        user_store.users["admin-123"] = MockUser(id="admin-123", role="admin", mfa_enabled=False)
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
        user_store.users["admin-123"] = MockUser(id="admin-123", role="admin", mfa_enabled=True)
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
        user_store.users["owner-123"] = MockUser(id="owner-123", role="owner", mfa_enabled=False)
        handler = make_mock_handler(ctx={"user_store": user_store})

        result = endpoint(handler)

        assert get_status(result) == 403

    @patch("aragora.server.middleware.mfa.get_current_user")
    def test_superadmin_without_mfa_blocked(self, mock_get_user):
        mock_user = MagicMock()
        mock_user.id = "super-123"
        mock_user.role = "superadmin"
        mock_user.metadata = {}
        mock_get_user.return_value = mock_user

        @require_admin_mfa
        def endpoint(handler, user=None):
            return {"success": True}

        user_store = MockUserStore()
        user_store.users["super-123"] = MockUser(
            id="super-123", role="superadmin", mfa_enabled=False
        )
        handler = make_mock_handler(ctx={"user_store": user_store})

        result = endpoint(handler)

        assert get_status(result) == 403

    @patch("aragora.server.middleware.mfa.get_current_user")
    def test_admin_service_account_bypass(self, mock_get_user):
        """Admin service account with approved bypass gets through."""
        mock_user = MagicMock()
        mock_user.id = "svc-admin"
        mock_user.role = "admin"
        mock_user.metadata = {}
        mock_get_user.return_value = mock_user

        @require_admin_mfa
        def endpoint(handler, user=None):
            return {"success": True, "bypassed": True}

        user_store = MockUserStore()
        user_store.users["svc-admin"] = MockUser(
            id="svc-admin",
            role="admin",
            mfa_enabled=False,
            is_service_account=True,
            mfa_bypass_approved_at=datetime.now(timezone.utc),
        )
        handler = make_mock_handler(ctx={"user_store": user_store})

        result = endpoint(handler)

        assert result["success"] is True
        assert result["bypassed"] is True

    @patch("aragora.server.middleware.mfa.get_current_user")
    def test_admin_fallback_to_metadata(self, mock_get_user):
        """Admin without user store falls back to metadata."""
        mock_user = MagicMock()
        mock_user.id = "admin-meta"
        mock_user.role = "admin"
        mock_user.metadata = {"mfa_enabled": True}
        mock_get_user.return_value = mock_user

        @require_admin_mfa
        def endpoint(handler, user=None):
            return {"success": True}

        handler = make_mock_handler(ctx={})
        result = endpoint(handler)

        assert result["success"] is True

    @patch("aragora.server.middleware.mfa.get_current_user")
    def test_admin_fallback_metadata_disabled(self, mock_get_user):
        mock_user = MagicMock()
        mock_user.id = "admin-meta"
        mock_user.role = "admin"
        mock_user.metadata = {"mfa_enabled": False}
        mock_get_user.return_value = mock_user

        @require_admin_mfa
        def endpoint(handler, user=None):
            return {"success": True}

        handler = make_mock_handler(ctx={})
        result = endpoint(handler)

        assert get_status(result) == 403

    @patch("aragora.server.middleware.mfa.get_current_user")
    def test_no_handler_returns_500(self, mock_get_user):
        @require_admin_mfa
        def endpoint():
            return {"success": True}

        result = endpoint()

        assert get_status(result) == 500

    @patch("aragora.server.middleware.mfa.get_current_user")
    def test_unauthenticated_returns_401(self, mock_get_user):
        mock_get_user.return_value = None

        @require_admin_mfa
        def endpoint(handler):
            return {"success": True}

        handler = make_mock_handler()
        result = endpoint(handler)

        assert get_status(result) == 401


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
        user_store.users["admin-123"] = MockUser(id="admin-123", mfa_enabled=False)
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
        user_store.users["admin-123"] = MockUser(id="admin-123", mfa_enabled=True)
        handler = make_mock_handler(ctx={"user_store": user_store})

        result = endpoint(handler)

        assert result["success"] is True
        assert result["sensitive_op"] is True

    @patch("aragora.server.middleware.mfa.get_current_user")
    def test_admin_service_account_bypass(self, mock_get_user):
        mock_user = MagicMock()
        mock_user.id = "svc-admin"
        mock_user.is_admin = True
        mock_user.metadata = {}
        mock_get_user.return_value = mock_user

        @require_admin_with_mfa
        def endpoint(handler, user=None):
            return {"success": True}

        user_store = MockUserStore()
        user_store.users["svc-admin"] = MockUser(
            id="svc-admin",
            mfa_enabled=False,
            is_service_account=True,
            mfa_bypass_approved_at=datetime.now(timezone.utc),
        )
        handler = make_mock_handler(ctx={"user_store": user_store})

        result = endpoint(handler)

        assert result["success"] is True

    @patch("aragora.server.middleware.mfa.get_current_user")
    def test_admin_fallback_metadata_mfa_enabled(self, mock_get_user):
        mock_user = MagicMock()
        mock_user.id = "admin-fallback"
        mock_user.is_admin = True
        mock_user.metadata = {"mfa_enabled": True}
        mock_get_user.return_value = mock_user

        @require_admin_with_mfa
        def endpoint(handler, user=None):
            return {"success": True}

        handler = make_mock_handler(ctx={})
        result = endpoint(handler)

        assert result["success"] is True

    @patch("aragora.server.middleware.mfa.get_current_user")
    def test_admin_fallback_metadata_mfa_disabled(self, mock_get_user):
        mock_user = MagicMock()
        mock_user.id = "admin-fallback"
        mock_user.is_admin = True
        mock_user.metadata = {"mfa_enabled": False}
        mock_get_user.return_value = mock_user

        @require_admin_with_mfa
        def endpoint(handler, user=None):
            return {"success": True}

        handler = make_mock_handler(ctx={})
        result = endpoint(handler)

        assert get_status(result) == 403

    @patch("aragora.server.middleware.mfa.get_current_user")
    def test_no_handler_returns_500(self, mock_get_user):
        @require_admin_with_mfa
        def endpoint():
            return {"success": True}

        result = endpoint()

        assert get_status(result) == 500

    @patch("aragora.server.middleware.mfa.get_current_user")
    def test_unauthenticated_returns_401(self, mock_get_user):
        mock_get_user.return_value = None

        @require_admin_with_mfa
        def endpoint(handler):
            return {"success": True}

        handler = make_mock_handler()
        result = endpoint(handler)

        assert get_status(result) == 401


# ===========================================================================
# Test require_mfa_fresh Decorator
# ===========================================================================


class TestRequireMfaFreshDecorator:
    """Tests for require_mfa_fresh step-up authentication decorator."""

    @patch("aragora.server.middleware.mfa._get_session_manager_from_handler")
    @patch("aragora.server.middleware.mfa._get_user_store_from_handler")
    @patch("aragora.server.middleware.mfa.get_current_user")
    def test_fresh_mfa_allows_access(self, mock_get_user, mock_store, mock_session):
        mock_user = MagicMock()
        mock_user.id = "user-123"
        mock_user.token_jti = "jti-abc"
        mock_user.metadata = {"jti": "jti-abc"}
        mock_get_user.return_value = mock_user

        full_user = MagicMock()
        full_user.mfa_enabled = True
        full_user.is_service_account = False
        mock_user_store = MagicMock()
        mock_user_store.get_user_by_id.return_value = full_user
        mock_store.return_value = mock_user_store

        mock_session_mgr = MagicMock()
        mock_session_mgr.is_session_mfa_fresh.return_value = True
        mock_session.return_value = mock_session_mgr

        @require_mfa_fresh(max_age_minutes=15)
        def sensitive_op(handler, user=None):
            return {"success": True}

        handler = make_mock_handler()
        result = sensitive_op(handler)

        assert result["success"] is True
        mock_session_mgr.is_session_mfa_fresh.assert_called_once_with(
            "user-123", "jti-abc", 15 * 60
        )

    @patch("aragora.server.middleware.mfa._get_session_manager_from_handler")
    @patch("aragora.server.middleware.mfa._get_user_store_from_handler")
    @patch("aragora.server.middleware.mfa.get_current_user")
    def test_stale_mfa_returns_403(self, mock_get_user, mock_store, mock_session):
        mock_user = MagicMock()
        mock_user.id = "user-123"
        mock_user.token_jti = "jti-abc"
        mock_user.metadata = {"jti": "jti-abc"}
        mock_get_user.return_value = mock_user

        full_user = MagicMock()
        full_user.mfa_enabled = True
        full_user.is_service_account = False
        mock_user_store = MagicMock()
        mock_user_store.get_user_by_id.return_value = full_user
        mock_store.return_value = mock_user_store

        mock_session_mgr = MagicMock()
        mock_session_mgr.is_session_mfa_fresh.return_value = False
        mock_session.return_value = mock_session_mgr

        @require_mfa_fresh(max_age_minutes=15)
        def sensitive_op(handler, user=None):
            return {"success": True}

        handler = make_mock_handler()
        result = sensitive_op(handler)

        assert get_status(result) == 403
        assert get_error_code(result) == "MFA_STEP_UP_REQUIRED"

    @patch("aragora.server.middleware.mfa._get_user_store_from_handler")
    @patch("aragora.server.middleware.mfa.get_current_user")
    def test_mfa_not_enabled_returns_403(self, mock_get_user, mock_store):
        mock_user = MagicMock()
        mock_user.id = "user-123"
        mock_user.metadata = {}
        mock_get_user.return_value = mock_user

        full_user = MagicMock()
        full_user.mfa_enabled = False
        full_user.is_service_account = False
        mock_user_store = MagicMock()
        mock_user_store.get_user_by_id.return_value = full_user
        mock_store.return_value = mock_user_store

        @require_mfa_fresh(max_age_minutes=10)
        def sensitive_op(handler, user=None):
            return {"success": True}

        handler = make_mock_handler()
        result = sensitive_op(handler)

        assert get_status(result) == 403
        assert get_error_code(result) == "MFA_REQUIRED"

    @patch("aragora.server.middleware.mfa._get_user_store_from_handler")
    @patch("aragora.server.middleware.mfa.get_current_user")
    def test_service_account_bypass_skips_freshness(self, mock_get_user, mock_store):
        mock_user = MagicMock()
        mock_user.id = "svc-123"
        mock_user.metadata = {}
        mock_get_user.return_value = mock_user

        full_user = MagicMock()
        full_user.mfa_enabled = False
        full_user.is_service_account = True
        full_user.is_mfa_bypass_valid.return_value = True
        mock_user_store = MagicMock()
        mock_user_store.get_user_by_id.return_value = full_user
        mock_store.return_value = mock_user_store

        @require_mfa_fresh(max_age_minutes=5)
        def sensitive_op(handler, user=None):
            return {"success": True, "bypassed": True}

        handler = make_mock_handler()
        result = sensitive_op(handler)

        assert result["success"] is True
        assert result["bypassed"] is True

    @patch("aragora.server.middleware.mfa._get_session_manager_from_handler")
    @patch("aragora.server.middleware.mfa._get_user_store_from_handler")
    @patch("aragora.server.middleware.mfa.get_current_user")
    def test_no_session_manager_returns_step_up(self, mock_get_user, mock_store, mock_session):
        mock_user = MagicMock()
        mock_user.id = "user-123"
        mock_user.token_jti = None
        mock_user.metadata = {}
        mock_get_user.return_value = mock_user

        full_user = MagicMock()
        full_user.mfa_enabled = True
        full_user.is_service_account = False
        mock_user_store = MagicMock()
        mock_user_store.get_user_by_id.return_value = full_user
        mock_store.return_value = mock_user_store

        mock_session.return_value = None

        @require_mfa_fresh(max_age_minutes=15)
        def sensitive_op(handler, user=None):
            return {"success": True}

        handler = make_mock_handler()
        result = sensitive_op(handler)

        assert get_status(result) == 403
        assert get_error_code(result) == "MFA_STEP_UP_REQUIRED"

    @patch("aragora.server.middleware.mfa._get_session_manager_from_handler")
    @patch("aragora.server.middleware.mfa._get_user_store_from_handler")
    @patch("aragora.server.middleware.mfa.get_current_user")
    def test_jti_from_metadata_fallback(self, mock_get_user, mock_store, mock_session):
        """When token_jti is None, JTI falls back to user.metadata['jti']."""
        mock_user = MagicMock()
        mock_user.id = "user-456"
        mock_user.token_jti = None
        mock_user.metadata = {"jti": "fallback-jti"}
        mock_get_user.return_value = mock_user

        full_user = MagicMock()
        full_user.mfa_enabled = True
        full_user.is_service_account = False
        mock_user_store = MagicMock()
        mock_user_store.get_user_by_id.return_value = full_user
        mock_store.return_value = mock_user_store

        mock_session_mgr = MagicMock()
        mock_session_mgr.is_session_mfa_fresh.return_value = True
        mock_session.return_value = mock_session_mgr

        @require_mfa_fresh(max_age_minutes=15)
        def sensitive_op(handler, user=None):
            return {"success": True}

        handler = make_mock_handler()
        result = sensitive_op(handler)

        assert result["success"] is True
        mock_session_mgr.is_session_mfa_fresh.assert_called_once_with(
            "user-456", "fallback-jti", 900
        )

    @patch("aragora.server.middleware.mfa.get_current_user")
    def test_no_handler_returns_500(self, mock_get_user):
        @require_mfa_fresh(max_age_minutes=15)
        def endpoint():
            return {"success": True}

        result = endpoint()

        assert get_status(result) == 500

    @patch("aragora.server.middleware.mfa.get_current_user")
    def test_unauthenticated_returns_401(self, mock_get_user):
        mock_get_user.return_value = None

        @require_mfa_fresh(max_age_minutes=15)
        def endpoint(handler):
            return {"success": True}

        handler = make_mock_handler()
        result = endpoint(handler)

        assert get_status(result) == 401

    @patch("aragora.server.middleware.mfa._get_user_store_from_handler")
    @patch("aragora.server.middleware.mfa.get_current_user")
    def test_custom_max_age_minutes(self, mock_get_user, mock_store):
        """Custom max_age_minutes is converted correctly to seconds."""
        mock_user = MagicMock()
        mock_user.id = "user-123"
        mock_user.token_jti = "jti-test"
        mock_user.metadata = {}
        mock_get_user.return_value = mock_user

        full_user = MagicMock()
        full_user.mfa_enabled = True
        full_user.is_service_account = False
        mock_user_store = MagicMock()
        mock_user_store.get_user_by_id.return_value = full_user
        mock_store.return_value = mock_user_store

        mock_session_mgr = MagicMock()
        mock_session_mgr.is_session_mfa_fresh.return_value = True

        with patch(
            "aragora.server.middleware.mfa._get_session_manager_from_handler",
            return_value=mock_session_mgr,
        ):

            @require_mfa_fresh(max_age_minutes=5)
            def sensitive_op(handler, user=None):
                return {"success": True}

            handler = make_mock_handler()
            result = sensitive_op(handler)

        mock_session_mgr.is_session_mfa_fresh.assert_called_once_with("user-123", "jti-test", 300)

    @patch("aragora.server.middleware.mfa._get_user_store_from_handler")
    @patch("aragora.server.middleware.mfa.get_current_user")
    def test_fallback_metadata_mfa_disabled(self, mock_get_user, mock_store):
        """When no user store, check metadata for mfa_enabled."""
        mock_user = MagicMock()
        mock_user.id = "user-123"
        mock_user.metadata = {"mfa_enabled": False}
        mock_get_user.return_value = mock_user

        mock_store.return_value = None

        @require_mfa_fresh(max_age_minutes=15)
        def sensitive_op(handler, user=None):
            return {"success": True}

        handler = make_mock_handler()
        result = sensitive_op(handler)

        assert get_status(result) == 403
        assert get_error_code(result) == "MFA_REQUIRED"


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

    @patch("aragora.server.middleware.mfa.get_current_user")
    def test_handler_detected_by_headers_attr(self, mock_get_user):
        """Handler is detected by looking for args with 'headers' attribute."""
        mock_user = MagicMock()
        mock_user.id = "user-123"
        mock_user.role = "user"
        mock_user.metadata = {}
        mock_get_user.return_value = mock_user

        @require_mfa
        def endpoint(request, user=None):
            return {"success": True}

        # Object with headers attribute will be detected as handler
        request = MagicMock()
        request.headers = {"Authorization": "Bearer test"}
        request.ctx = {"user_store": MockUserStore()}

        user_store = MockUserStore()
        user_store.users["user-123"] = MockUser(id="user-123", mfa_enabled=True)
        request.ctx = {"user_store": user_store}

        result = endpoint(request)

        assert result["success"] is True


# ===========================================================================
# Test Decorator Preserves Function Metadata
# ===========================================================================


class TestDecoratorMetadata:
    """Test that decorators preserve function metadata via functools.wraps."""

    def test_require_mfa_preserves_name(self):
        @require_mfa
        def my_endpoint():
            """My endpoint docstring."""
            pass

        assert my_endpoint.__name__ == "my_endpoint"
        assert my_endpoint.__doc__ == "My endpoint docstring."

    def test_require_admin_mfa_preserves_name(self):
        @require_admin_mfa
        def admin_endpoint():
            """Admin endpoint docstring."""
            pass

        assert admin_endpoint.__name__ == "admin_endpoint"
        assert admin_endpoint.__doc__ == "Admin endpoint docstring."

    def test_require_admin_with_mfa_preserves_name(self):
        @require_admin_with_mfa
        def sensitive_endpoint():
            """Sensitive endpoint docstring."""
            pass

        assert sensitive_endpoint.__name__ == "sensitive_endpoint"
        assert sensitive_endpoint.__doc__ == "Sensitive endpoint docstring."

    def test_require_mfa_fresh_preserves_name(self):
        @require_mfa_fresh(max_age_minutes=10)
        def fresh_endpoint():
            """Fresh endpoint docstring."""
            pass

        assert fresh_endpoint.__name__ == "fresh_endpoint"
        assert fresh_endpoint.__doc__ == "Fresh endpoint docstring."
