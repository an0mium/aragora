"""
Tests for MFA Enforcement Middleware.

Validates SOC 2 Control CC5-01: Enforce MFA for administrative access.
"""

import json
from dataclasses import dataclass
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import pytest


@dataclass
class MockUser:
    """Mock user for testing."""

    id: str
    email: str
    role: str = "user"
    plan: str = "free"
    mfa_enabled: bool = False
    mfa_secret: Optional[str] = None
    mfa_backup_codes: Optional[str] = None
    metadata: dict = None
    created_at: Optional[str] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"


class MockUserStore:
    """Mock user store for testing."""

    def __init__(self):
        self.users = {}

    def add_user(self, user: MockUser):
        self.users[user.id] = user

    def get_user_by_id(self, user_id: str) -> Optional[MockUser]:
        return self.users.get(user_id)


class MockHandler:
    """Mock HTTP handler for testing."""

    def __init__(self, user: Optional[MockUser] = None, user_store: Optional[MockUserStore] = None):
        self.headers = {"Authorization": f"Bearer test_token_{user.id}" if user else ""}
        self._user = user
        self.ctx = {"user_store": user_store}


def get_status(result):
    """Extract status code from result (HandlerResult or tuple)."""
    if hasattr(result, "status_code"):
        return result.status_code
    return result[1]


def get_body(result):
    """Extract body from result (HandlerResult or tuple)."""
    if hasattr(result, "body"):
        return json.loads(result.body.decode("utf-8"))
    return result[0] if isinstance(result[0], dict) else json.loads(result[0])


class TestRequireMFA:
    """Test the require_mfa decorator."""

    def test_unauthenticated_request_rejected(self):
        """Unauthenticated requests should be rejected with 401."""
        from aragora.server.middleware.mfa import require_mfa

        @require_mfa
        def protected_endpoint(handler, user):
            return {"success": True}

        handler = MockHandler()

        with patch("aragora.server.middleware.mfa.get_current_user") as mock_get_user:
            mock_get_user.return_value = None
            result = protected_endpoint(handler)

        assert get_status(result) == 401

    def test_user_without_mfa_rejected(self):
        """Users without MFA enabled should be rejected with 403."""
        from aragora.server.middleware.mfa import require_mfa

        @require_mfa
        def protected_endpoint(handler, user):
            return {"success": True}

        user = MockUser(id="user_1", email="test@example.com", mfa_enabled=False)
        user_store = MockUserStore()
        user_store.add_user(user)
        handler = MockHandler(user=user, user_store=user_store)

        with patch("aragora.server.middleware.mfa.get_current_user") as mock_get_user:
            mock_get_user.return_value = user
            with patch("aragora.server.middleware.mfa._get_user_store_from_handler") as mock_store:
                mock_store.return_value = user_store
                result = protected_endpoint(handler)

        assert get_status(result) == 403
        body = get_body(result)
        assert body["error"]["code"] == "MFA_REQUIRED"

    def test_user_with_mfa_allowed(self):
        """Users with MFA enabled should be allowed."""
        from aragora.server.middleware.mfa import require_mfa

        @require_mfa
        def protected_endpoint(handler, user):
            return {"user_id": user.id}, 200, {}

        user = MockUser(id="user_1", email="test@example.com", mfa_enabled=True)
        user_store = MockUserStore()
        user_store.add_user(user)
        handler = MockHandler(user=user, user_store=user_store)

        with patch("aragora.server.middleware.mfa.get_current_user") as mock_get_user:
            mock_get_user.return_value = user
            with patch("aragora.server.middleware.mfa._get_user_store_from_handler") as mock_store:
                mock_store.return_value = user_store
                result = protected_endpoint(handler)

        assert result[1] == 200
        assert result[0]["user_id"] == "user_1"


class TestRequireAdminMFA:
    """Test the require_admin_mfa decorator."""

    def test_regular_user_allowed_without_mfa(self):
        """Regular users should be allowed without MFA."""
        from aragora.server.middleware.mfa import require_admin_mfa

        @require_admin_mfa
        def admin_endpoint(handler, user):
            return {"success": True}, 200, {}

        user = MockUser(id="user_1", email="test@example.com", role="user", mfa_enabled=False)
        user_store = MockUserStore()
        user_store.add_user(user)
        handler = MockHandler(user=user, user_store=user_store)

        with patch("aragora.server.middleware.mfa.get_current_user") as mock_get_user:
            mock_get_user.return_value = user
            result = admin_endpoint(handler)

        assert result[1] == 200

    def test_admin_without_mfa_rejected(self):
        """Admin users without MFA should be rejected with 403."""
        from aragora.server.middleware.mfa import require_admin_mfa

        @require_admin_mfa
        def admin_endpoint(handler, user):
            return {"success": True}

        user = MockUser(id="admin_1", email="admin@example.com", role="admin", mfa_enabled=False)
        user_store = MockUserStore()
        user_store.add_user(user)
        handler = MockHandler(user=user, user_store=user_store)

        with patch("aragora.server.middleware.mfa.get_current_user") as mock_get_user:
            mock_get_user.return_value = user
            with patch("aragora.server.middleware.mfa._get_user_store_from_handler") as mock_store:
                mock_store.return_value = user_store
                result = admin_endpoint(handler)

        assert get_status(result) == 403
        body = get_body(result)
        assert body["error"]["code"] == "ADMIN_MFA_REQUIRED"

    def test_admin_with_mfa_allowed(self):
        """Admin users with MFA should be allowed."""
        from aragora.server.middleware.mfa import require_admin_mfa

        @require_admin_mfa
        def admin_endpoint(handler, user):
            return {"admin": True}, 200, {}

        user = MockUser(id="admin_1", email="admin@example.com", role="admin", mfa_enabled=True)
        user_store = MockUserStore()
        user_store.add_user(user)
        handler = MockHandler(user=user, user_store=user_store)

        with patch("aragora.server.middleware.mfa.get_current_user") as mock_get_user:
            mock_get_user.return_value = user
            with patch("aragora.server.middleware.mfa._get_user_store_from_handler") as mock_store:
                mock_store.return_value = user_store
                result = admin_endpoint(handler)

        assert result[1] == 200
        assert result[0]["admin"] is True

    def test_owner_without_mfa_rejected(self):
        """Owner users without MFA should be rejected."""
        from aragora.server.middleware.mfa import require_admin_mfa

        @require_admin_mfa
        def admin_endpoint(handler, user):
            return {"success": True}

        user = MockUser(id="owner_1", email="owner@example.com", role="owner", mfa_enabled=False)
        user_store = MockUserStore()
        user_store.add_user(user)
        handler = MockHandler(user=user, user_store=user_store)

        with patch("aragora.server.middleware.mfa.get_current_user") as mock_get_user:
            mock_get_user.return_value = user
            with patch("aragora.server.middleware.mfa._get_user_store_from_handler") as mock_store:
                mock_store.return_value = user_store
                result = admin_endpoint(handler)

        assert get_status(result) == 403


class TestRequireAdminWithMFA:
    """Test the require_admin_with_mfa decorator."""

    def test_non_admin_rejected(self):
        """Non-admin users should be rejected even with MFA."""
        from aragora.server.middleware.mfa import require_admin_with_mfa

        @require_admin_with_mfa
        def sensitive_endpoint(handler, user):
            return {"success": True}

        user = MockUser(id="user_1", email="test@example.com", role="user", mfa_enabled=True)
        handler = MockHandler(user=user)

        with patch("aragora.server.middleware.mfa.get_current_user") as mock_get_user:
            mock_get_user.return_value = user
            result = sensitive_endpoint(handler)

        assert get_status(result) == 403
        body = get_body(result)
        assert "Admin access required" in body["error"]

    def test_admin_without_mfa_rejected(self):
        """Admin users without MFA should be rejected."""
        from aragora.server.middleware.mfa import require_admin_with_mfa

        @require_admin_with_mfa
        def sensitive_endpoint(handler, user):
            return {"success": True}

        user = MockUser(id="admin_1", email="admin@example.com", role="admin", mfa_enabled=False)
        user_store = MockUserStore()
        user_store.add_user(user)
        handler = MockHandler(user=user, user_store=user_store)

        with patch("aragora.server.middleware.mfa.get_current_user") as mock_get_user:
            mock_get_user.return_value = user
            with patch("aragora.server.middleware.mfa._get_user_store_from_handler") as mock_store:
                mock_store.return_value = user_store
                result = sensitive_endpoint(handler)

        assert get_status(result) == 403
        body = get_body(result)
        assert body["error"]["code"] == "MFA_REQUIRED"

    def test_admin_with_mfa_allowed(self):
        """Admin users with MFA should be allowed."""
        from aragora.server.middleware.mfa import require_admin_with_mfa

        @require_admin_with_mfa
        def sensitive_endpoint(handler, user):
            return {"sensitive": True}, 200, {}

        user = MockUser(id="admin_1", email="admin@example.com", role="admin", mfa_enabled=True)
        user_store = MockUserStore()
        user_store.add_user(user)
        handler = MockHandler(user=user, user_store=user_store)

        with patch("aragora.server.middleware.mfa.get_current_user") as mock_get_user:
            mock_get_user.return_value = user
            with patch("aragora.server.middleware.mfa._get_user_store_from_handler") as mock_store:
                mock_store.return_value = user_store
                result = sensitive_endpoint(handler)

        assert result[1] == 200
        assert result[0]["sensitive"] is True


class TestCheckMFAStatus:
    """Test the check_mfa_status function."""

    def test_user_not_found(self):
        """Should handle missing users gracefully."""
        from aragora.server.middleware.mfa import check_mfa_status

        user_store = MockUserStore()
        result = check_mfa_status("nonexistent", user_store)

        assert result["mfa_enabled"] is False
        assert result["error"] == "User not found"

    def test_no_user_store(self):
        """Should handle missing user store gracefully."""
        from aragora.server.middleware.mfa import check_mfa_status

        result = check_mfa_status("user_1", None)

        assert result["mfa_enabled"] is False
        assert result["error"] == "User store not available"

    def test_mfa_disabled(self):
        """Should correctly report MFA disabled status."""
        from aragora.server.middleware.mfa import check_mfa_status

        user = MockUser(id="user_1", email="test@example.com", mfa_enabled=False)
        user_store = MockUserStore()
        user_store.add_user(user)

        result = check_mfa_status("user_1", user_store)

        assert result["mfa_enabled"] is False
        assert result["mfa_secret_set"] is False
        assert result["backup_codes_remaining"] == 0

    def test_mfa_enabled_with_backup_codes(self):
        """Should correctly count backup codes."""
        from aragora.server.middleware.mfa import check_mfa_status

        backup_codes = json.dumps(["hash1", "hash2", "hash3"])
        user = MockUser(
            id="user_1",
            email="test@example.com",
            mfa_enabled=True,
            mfa_secret="ABCD1234",
            mfa_backup_codes=backup_codes,
        )
        user_store = MockUserStore()
        user_store.add_user(user)

        result = check_mfa_status("user_1", user_store)

        assert result["mfa_enabled"] is True
        assert result["mfa_secret_set"] is True
        assert result["backup_codes_remaining"] == 3


class TestEnforceAdminMFAPolicy:
    """Test the enforce_admin_mfa_policy function."""

    def test_non_admin_always_compliant(self):
        """Non-admin users should always be compliant."""
        from aragora.server.middleware.mfa import enforce_admin_mfa_policy

        user = MockUser(id="user_1", email="test@example.com", role="user")
        user_store = MockUserStore()
        user_store.add_user(user)

        result = enforce_admin_mfa_policy(user, user_store)

        assert result is None  # None means compliant

    def test_admin_with_mfa_compliant(self):
        """Admin users with MFA should be compliant."""
        from aragora.server.middleware.mfa import enforce_admin_mfa_policy

        backup_codes = json.dumps(["h1", "h2", "h3", "h4", "h5"])
        user = MockUser(
            id="admin_1",
            email="admin@example.com",
            role="admin",
            mfa_enabled=True,
            mfa_backup_codes=backup_codes,
        )
        user_store = MockUserStore()
        user_store.add_user(user)

        result = enforce_admin_mfa_policy(user, user_store)

        assert result is None  # Compliant

    def test_admin_low_backup_codes_warning(self):
        """Admin with low backup codes should get warning."""
        from aragora.server.middleware.mfa import enforce_admin_mfa_policy

        backup_codes = json.dumps(["h1", "h2"])
        user = MockUser(
            id="admin_1",
            email="admin@example.com",
            role="admin",
            mfa_enabled=True,
            mfa_backup_codes=backup_codes,
        )
        user_store = MockUserStore()
        user_store.add_user(user)

        result = enforce_admin_mfa_policy(user, user_store)

        assert result is not None
        assert result["compliant"] is True
        assert result["warning"] == "Low backup codes"
        assert result["backup_codes_remaining"] == 2

    def test_admin_without_mfa_not_compliant(self):
        """Admin without MFA should be non-compliant and enforced."""
        from aragora.server.middleware.mfa import enforce_admin_mfa_policy

        user = MockUser(id="admin_1", email="admin@example.com", role="admin", mfa_enabled=False)
        user_store = MockUserStore()
        user_store.add_user(user)

        result = enforce_admin_mfa_policy(user, user_store)

        assert result is not None
        assert result["compliant"] is False
        assert result["enforced"] is True


class TestMFAIntegration:
    """Integration tests for MFA enforcement."""

    def test_decorator_chain_works(self):
        """Multiple decorators should work together correctly."""
        from aragora.server.middleware.mfa import require_admin_mfa

        call_count = [0]

        @require_admin_mfa
        def counted_endpoint(handler, user):
            call_count[0] += 1
            return {"count": call_count[0]}, 200, {}

        # Admin with MFA should increment count
        admin = MockUser(id="admin_1", email="admin@example.com", role="admin", mfa_enabled=True)
        user_store = MockUserStore()
        user_store.add_user(admin)
        handler = MockHandler(user=admin, user_store=user_store)

        with patch("aragora.server.middleware.mfa.get_current_user") as mock_get_user:
            mock_get_user.return_value = admin
            with patch("aragora.server.middleware.mfa._get_user_store_from_handler") as mock_store:
                mock_store.return_value = user_store
                result = counted_endpoint(handler)

        assert call_count[0] == 1
        assert result[1] == 200

    def test_all_admin_roles_enforced(self):
        """All admin-type roles should have MFA enforced."""
        from aragora.server.middleware.mfa import require_admin_mfa

        @require_admin_mfa
        def admin_endpoint(handler, user):
            return {"success": True}

        admin_roles = ["admin", "owner", "superadmin"]

        for role in admin_roles:
            user = MockUser(
                id=f"{role}_1", email=f"{role}@example.com", role=role, mfa_enabled=False
            )
            user_store = MockUserStore()
            user_store.add_user(user)
            handler = MockHandler(user=user, user_store=user_store)

            with patch("aragora.server.middleware.mfa.get_current_user") as mock_get_user:
                mock_get_user.return_value = user
                with patch(
                    "aragora.server.middleware.mfa._get_user_store_from_handler"
                ) as mock_store:
                    mock_store.return_value = user_store
                    result = admin_endpoint(handler)

            assert get_status(result) == 403, f"Role {role} should be rejected without MFA"
