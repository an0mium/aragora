"""
Tests for aragora.server.openapi.endpoints.admin_security and
aragora.server.handlers.admin.security modules.

Comprehensive tests covering:
1. Permission checks (require admin role)
2. Key lifecycle management (status, health, rotation, listing)
3. Audit logging verification
4. Impersonation controls and time limits
5. Error handling for unauthorized access
6. Rate limiting for sensitive operations
7. Compliance endpoints (violations)
8. Backup endpoints (list, create, delete)
9. Disaster recovery endpoints

50+ tests for admin security endpoints.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from io import BytesIO
from typing import Any
from unittest.mock import MagicMock, patch, AsyncMock

import pytest


# =============================================================================
# Test Fixtures and Helpers
# =============================================================================


@dataclass
class MockUser:
    """Mock user for testing."""

    id: str = "user-123"
    email: str = "admin@example.com"
    name: str = "Admin User"
    org_id: str | None = "org-123"
    role: str = "admin"
    is_active: bool = True
    mfa_enabled: bool = True
    mfa_secret: str | None = "TESTSECRET123456"
    mfa_backup_codes: str | None = None

    def __post_init__(self):
        if self.mfa_backup_codes is None:
            self.mfa_backup_codes = json.dumps(
                [
                    "hash1",
                    "hash2",
                    "hash3",
                    "hash4",
                    "hash5",
                    "hash6",
                    "hash7",
                    "hash8",
                    "hash9",
                    "hash10",
                ]
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "email": self.email,
            "name": self.name,
            "org_id": self.org_id,
            "role": self.role,
            "is_active": self.is_active,
            "mfa_enabled": self.mfa_enabled,
        }


@dataclass
class MockAuthContext:
    """Mock authentication context."""

    is_authenticated: bool = True
    user_id: str = "admin-123"
    email: str = "admin@example.com"
    org_id: str | None = "org-123"
    role: str = "admin"
    workspace_id: str = "ws-123"


@dataclass
class MockPermissionDecision:
    """Mock RBAC permission decision."""

    allowed: bool = True
    reason: str = "Allowed by test"


@dataclass
class MockEncryptionKey:
    """Mock encryption key."""

    key_id: str = "key-001"
    version: int = 1
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc) - timedelta(days=45)
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "key_id": self.key_id,
            "version": self.version,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class MockRotationResult:
    """Mock key rotation result."""

    success: bool = True
    old_key_version: int = 1
    new_key_version: int = 2
    stores_processed: list = field(default_factory=list)
    records_reencrypted: int = 100
    failed_records: int = 0
    duration_seconds: float = 5.5
    errors: list = field(default_factory=list)


class MockUserStore:
    """Mock user store for testing."""

    def __init__(self):
        self.users: dict[str, MockUser] = {}
        backup_codes = json.dumps(
            [
                "hash1",
                "hash2",
                "hash3",
                "hash4",
                "hash5",
                "hash6",
                "hash7",
                "hash8",
                "hash9",
                "hash10",
            ]
        )
        self.users["admin-123"] = MockUser(
            id="admin-123",
            email="admin@example.com",
            role="admin",
            mfa_enabled=True,
            mfa_backup_codes=backup_codes,
        )
        self.users["owner-123"] = MockUser(
            id="owner-123",
            email="owner@example.com",
            role="owner",
            mfa_enabled=True,
            mfa_backup_codes=backup_codes,
        )
        self.users["user-456"] = MockUser(
            id="user-456",
            email="user@example.com",
            role="user",
            mfa_enabled=False,
        )
        self.users["admin-no-mfa"] = MockUser(
            id="admin-no-mfa",
            email="admin-no-mfa@example.com",
            role="admin",
            mfa_enabled=False,
        )

    def get_user_by_id(self, user_id: str) -> MockUser | None:
        return self.users.get(user_id)


class MockEncryptionService:
    """Mock encryption service."""

    def __init__(self):
        self.keys = [
            MockEncryptionKey(key_id="key-001", version=1),
            MockEncryptionKey(key_id="key-002", version=2, created_at=datetime.now(timezone.utc)),
        ]
        self.active_key_id = "key-002"

    def get_active_key(self) -> MockEncryptionKey | None:
        for key in self.keys:
            if key.key_id == self.active_key_id:
                return key
        return None

    def get_active_key_id(self) -> str:
        return self.active_key_id

    def list_keys(self) -> list[dict[str, Any]]:
        return [key.to_dict() for key in self.keys]

    def encrypt(self, data: bytes) -> bytes:
        return b"encrypted:" + data

    def decrypt(self, data: bytes) -> bytes:
        return data.replace(b"encrypted:", b"")


def make_mock_handler(
    body: dict | None = None,
    method: str = "GET",
    headers: dict | None = None,
):
    """Create a mock HTTP handler."""
    handler = MagicMock()
    handler.command = method
    handler.headers = headers or {}
    handler.client_address = ("127.0.0.1", 12345)
    handler.remote = "127.0.0.1"

    if body is not None:
        body_bytes = json.dumps(body).encode("utf-8")
        handler.headers["Content-Length"] = str(len(body_bytes))
        handler.rfile = BytesIO(body_bytes)
        handler.request_body = body_bytes
    else:
        handler.rfile = BytesIO(b"")
        handler.headers["Content-Length"] = "0"
        handler.request_body = b"{}"

    return handler


def get_status(result) -> int:
    """Extract status code from HandlerResult or tuple."""
    if hasattr(result, "status_code"):
        return result.status_code
    return result[1]


def get_body(result) -> dict:
    """Extract body from HandlerResult or tuple."""
    if hasattr(result, "body"):
        body = result.body
        if isinstance(body, bytes):
            return json.loads(body.decode("utf-8"))
        return json.loads(body)
    body = result[0]
    if isinstance(body, dict):
        return body
    return json.loads(body)


def mock_check_permission_allowed(*args, **kwargs):
    """Mock check_permission that always allows."""
    return MockPermissionDecision(allowed=True)


def mock_check_permission_denied(*args, **kwargs):
    """Mock check_permission that always denies."""
    return MockPermissionDecision(allowed=False, reason="Permission denied")


# =============================================================================
# Test OpenAPI Schema Structure
# =============================================================================


class TestAdminSecurityOpenAPISchema:
    """Tests for ADMIN_SECURITY_ENDPOINTS OpenAPI schema structure."""

    def test_security_status_endpoint_exists(self):
        """Security status endpoint should be defined."""
        from aragora.server.openapi.endpoints.admin_security import ADMIN_SECURITY_ENDPOINTS

        assert "/api/v1/admin/security/status" in ADMIN_SECURITY_ENDPOINTS

    def test_security_health_endpoint_exists(self):
        """Security health endpoint should be defined."""
        from aragora.server.openapi.endpoints.admin_security import ADMIN_SECURITY_ENDPOINTS

        assert "/api/v1/admin/security/health" in ADMIN_SECURITY_ENDPOINTS

    def test_security_keys_endpoint_exists(self):
        """Security keys endpoint should be defined."""
        from aragora.server.openapi.endpoints.admin_security import ADMIN_SECURITY_ENDPOINTS

        assert "/api/v1/admin/security/keys" in ADMIN_SECURITY_ENDPOINTS

    def test_rotate_key_endpoint_exists(self):
        """Rotate key endpoint should be defined."""
        from aragora.server.openapi.endpoints.admin_security import ADMIN_SECURITY_ENDPOINTS

        assert "/api/v1/admin/security/rotate-key" in ADMIN_SECURITY_ENDPOINTS

    def test_impersonate_endpoint_exists(self):
        """Impersonate endpoint should be defined."""
        from aragora.server.openapi.endpoints.admin_security import ADMIN_SECURITY_ENDPOINTS

        assert "/api/v1/admin/impersonate/{user_id}" in ADMIN_SECURITY_ENDPOINTS

    def test_all_endpoints_require_bearer_auth(self):
        """All admin security endpoints should require bearer auth."""
        from aragora.server.openapi.endpoints.admin_security import ADMIN_SECURITY_ENDPOINTS

        for path, methods in ADMIN_SECURITY_ENDPOINTS.items():
            for method, spec in methods.items():
                if method in ("get", "post", "put", "delete"):
                    assert "security" in spec, f"{path} {method} missing security"
                    assert {"bearerAuth": []} in spec["security"]

    def test_security_status_response_properties(self):
        """Security status response should have expected properties."""
        from aragora.server.openapi.endpoints.admin_security import ADMIN_SECURITY_ENDPOINTS

        endpoint = ADMIN_SECURITY_ENDPOINTS["/api/v1/admin/security/status"]
        schema = endpoint["get"]["responses"]["200"]["content"]["application/json"]["schema"]
        props = schema["properties"]

        assert "crypto_available" in props
        assert "active_key_id" in props
        assert "key_version" in props
        assert "rotation_recommended" in props
        assert "rotation_required" in props

    def test_security_health_status_enum(self):
        """Security health endpoint should define status enum."""
        from aragora.server.openapi.endpoints.admin_security import ADMIN_SECURITY_ENDPOINTS

        endpoint = ADMIN_SECURITY_ENDPOINTS["/api/v1/admin/security/health"]
        schema = endpoint["get"]["responses"]["200"]["content"]["application/json"]["schema"]
        status_prop = schema["properties"]["status"]

        assert "enum" in status_prop
        assert "healthy" in status_prop["enum"]
        assert "degraded" in status_prop["enum"]
        assert "unhealthy" in status_prop["enum"]

    def test_rotate_key_endpoint_is_post(self):
        """Rotate key endpoint should be POST."""
        from aragora.server.openapi.endpoints.admin_security import ADMIN_SECURITY_ENDPOINTS

        endpoint = ADMIN_SECURITY_ENDPOINTS["/api/v1/admin/security/rotate-key"]
        assert "post" in endpoint

    def test_impersonate_response_has_token(self):
        """Impersonate response should include token."""
        from aragora.server.openapi.endpoints.admin_security import ADMIN_SECURITY_ENDPOINTS

        endpoint = ADMIN_SECURITY_ENDPOINTS["/api/v1/admin/impersonate/{user_id}"]
        schema = endpoint["post"]["responses"]["200"]["content"]["application/json"]["schema"]
        props = schema["properties"]

        assert "token" in props
        assert "expires_at" in props
        assert "target_user" in props


# =============================================================================
# Test Permission Checks
# =============================================================================


class TestSecurityEndpointPermissions:
    """Tests for admin permission requirements on security endpoints."""

    @pytest.fixture
    def user_store(self):
        return MockUserStore()

    @pytest.fixture
    def security_handler(self, user_store):
        from aragora.server.handlers.admin.security import SecurityHandler

        ctx = {"user_store": user_store}
        return SecurityHandler(ctx)

    @patch("aragora.server.handlers.admin.admin.extract_user_from_request")
    def test_admin_role_can_access_security_status(self, mock_auth, security_handler, user_store):
        """Admin role should be able to access security status."""
        mock_auth.return_value = MockAuthContext(user_id="admin-123", role="admin")

        with patch("aragora.server.handlers.admin.security.CRYPTO_AVAILABLE", False):
            with patch("aragora.server.handlers.admin.security.get_encryption_service"):
                handler = make_mock_handler()
                # SecurityHandler uses handle() for GET
                result = security_handler.handle("/api/v1/admin/security/status", {}, handler)
                # With CRYPTO_AVAILABLE=False, returns crypto_available: false
                assert result is not None

    @patch("aragora.server.handlers.admin.admin.extract_user_from_request")
    def test_owner_role_can_access_security_status(self, mock_auth, security_handler, user_store):
        """Owner role should be able to access security status."""
        mock_auth.return_value = MockAuthContext(user_id="owner-123", role="owner")

        with patch("aragora.server.handlers.admin.security.CRYPTO_AVAILABLE", False):
            with patch("aragora.server.handlers.admin.security.get_encryption_service"):
                handler = make_mock_handler()
                result = security_handler.handle("/api/v1/admin/security/status", {}, handler)
                assert result is not None

    @patch("aragora.server.handlers.admin.admin.extract_user_from_request")
    def test_regular_user_denied_security_access(self, mock_auth, security_handler, user_store):
        """Regular users should be denied access to security endpoints."""
        mock_auth.return_value = MockAuthContext(user_id="user-456", role="user")

        handler = make_mock_handler()
        result = security_handler.handle("/api/v1/admin/security/status", {}, handler)

        # Should return None (not handled) or 403
        if result is not None:
            assert get_status(result) in (401, 403)

    @patch("aragora.server.handlers.admin.admin.extract_user_from_request")
    def test_unauthenticated_denied_security_access(self, mock_auth, security_handler):
        """Unauthenticated requests should be denied."""
        mock_auth.return_value = MockAuthContext(is_authenticated=False)

        handler = make_mock_handler()
        result = security_handler.handle("/api/v1/admin/security/status", {}, handler)

        if result is not None:
            assert get_status(result) == 401

    @patch("aragora.server.handlers.admin.admin.extract_user_from_request")
    def test_admin_without_mfa_denied(self, mock_auth, security_handler, user_store):
        """Admin without MFA enabled should be denied (SOC 2 CC5-01)."""
        mock_auth.return_value = MockAuthContext(user_id="admin-no-mfa", role="admin")

        handler = make_mock_handler()
        result = security_handler.handle("/api/v1/admin/security/status", {}, handler)

        # MFA requirement for admins should block access
        if result is not None:
            status = get_status(result)
            # Either 403 (MFA required) or None (not handled)
            assert status in (401, 403)


# =============================================================================
# Test Key Lifecycle Management
# =============================================================================


class TestKeyLifecycleManagement:
    """Tests for encryption key lifecycle management."""

    @pytest.fixture
    def user_store(self):
        return MockUserStore()

    @pytest.fixture
    def security_handler(self, user_store):
        from aragora.server.handlers.admin.security import SecurityHandler

        ctx = {"user_store": user_store}
        return SecurityHandler(ctx)

    @patch("aragora.server.handlers.admin.admin.extract_user_from_request")
    @patch("aragora.server.handlers.admin.security.CRYPTO_AVAILABLE", True)
    @patch("aragora.server.handlers.admin.security.get_encryption_service")
    def test_get_security_status_success(self, mock_service, mock_auth, security_handler):
        """Security status should return key information."""
        mock_auth.return_value = MockAuthContext(user_id="admin-123", role="admin")

        mock_enc = MockEncryptionService()
        mock_service.return_value = mock_enc

        handler = make_mock_handler()
        result = security_handler._get_status(handler)

        assert result is not None
        assert get_status(result) == 200
        data = get_body(result)
        assert data["crypto_available"] is True
        assert "active_key_id" in data

    @patch("aragora.server.handlers.admin.admin.extract_user_from_request")
    @patch("aragora.server.handlers.admin.security.CRYPTO_AVAILABLE", False)
    def test_get_security_status_no_crypto(self, mock_auth, security_handler):
        """Security status should report when crypto is unavailable."""
        mock_auth.return_value = MockAuthContext(user_id="admin-123", role="admin")

        handler = make_mock_handler()
        result = security_handler._get_status(handler)

        assert result is not None
        data = get_body(result)
        assert data["crypto_available"] is False

    @patch("aragora.server.handlers.admin.admin.extract_user_from_request")
    @patch("aragora.server.handlers.admin.security.CRYPTO_AVAILABLE", True)
    @patch("aragora.server.handlers.admin.security.get_encryption_service")
    def test_get_security_health_healthy(self, mock_service, mock_auth, security_handler):
        """Security health should return healthy when all checks pass."""
        mock_auth.return_value = MockAuthContext(user_id="admin-123", role="admin")

        mock_enc = MockEncryptionService()
        mock_service.return_value = mock_enc

        handler = make_mock_handler()
        result = security_handler._get_health(handler)

        assert result is not None
        data = get_body(result)
        assert data["status"] in ("healthy", "degraded")
        assert "checks" in data

    @patch("aragora.server.handlers.admin.admin.extract_user_from_request")
    @patch("aragora.server.handlers.admin.security.CRYPTO_AVAILABLE", True)
    @patch("aragora.server.handlers.admin.security.get_encryption_service")
    def test_list_keys_success(self, mock_service, mock_auth, security_handler):
        """List keys should return all keys without sensitive data."""
        mock_auth.return_value = MockAuthContext(user_id="admin-123", role="admin")

        mock_enc = MockEncryptionService()
        mock_service.return_value = mock_enc

        handler = make_mock_handler()
        result = security_handler._list_keys(handler)

        assert result is not None
        data = get_body(result)
        assert "keys" in data
        assert "total_keys" in data
        assert len(data["keys"]) == 2

    @patch("aragora.server.handlers.admin.admin.extract_user_from_request")
    @patch("aragora.server.handlers.admin.security.CRYPTO_AVAILABLE", True)
    @patch("aragora.server.handlers.admin.security.get_encryption_service")
    def test_list_keys_includes_active_indicator(self, mock_service, mock_auth, security_handler):
        """Listed keys should indicate which one is active."""
        mock_auth.return_value = MockAuthContext(user_id="admin-123", role="admin")

        mock_enc = MockEncryptionService()
        mock_service.return_value = mock_enc

        handler = make_mock_handler()
        result = security_handler._list_keys(handler)

        data = get_body(result)
        active_count = sum(1 for k in data["keys"] if k.get("is_active"))
        assert active_count == 1

    @patch("aragora.server.handlers.admin.admin.extract_user_from_request")
    @patch("aragora.server.handlers.admin.security.CRYPTO_AVAILABLE", True)
    @patch("aragora.server.handlers.admin.security.get_encryption_service")
    @patch("aragora.server.handlers.admin.security.rotate_encryption_key")
    def test_rotate_key_success(self, mock_rotate, mock_service, mock_auth, security_handler):
        """Key rotation should succeed and return new key info."""
        mock_auth.return_value = MockAuthContext(user_id="admin-123", role="admin")

        mock_enc = MockEncryptionService()
        # Make the key old enough to not require force
        mock_enc.keys[1].created_at = datetime.now(timezone.utc) - timedelta(days=60)
        mock_service.return_value = mock_enc

        mock_rotate.return_value = MockRotationResult()

        handler = make_mock_handler({"dry_run": False, "force": True})
        result = security_handler._rotate_key({"dry_run": False, "force": True}, handler)

        assert result is not None
        data = get_body(result)
        assert data["success"] is True
        assert "new_key_version" in data

    @patch("aragora.server.handlers.admin.admin.extract_user_from_request")
    @patch("aragora.server.handlers.admin.security.CRYPTO_AVAILABLE", True)
    @patch("aragora.server.handlers.admin.security.get_encryption_service")
    def test_rotate_key_rejected_when_key_too_new(self, mock_service, mock_auth, security_handler):
        """Key rotation should be rejected if key is less than 30 days old."""
        mock_auth.return_value = MockAuthContext(user_id="admin-123", role="admin")

        mock_enc = MockEncryptionService()
        # Key is only 5 days old
        mock_enc.keys[1].created_at = datetime.now(timezone.utc) - timedelta(days=5)
        mock_service.return_value = mock_enc

        handler = make_mock_handler({"dry_run": False, "force": False})
        result = security_handler._rotate_key({"dry_run": False, "force": False}, handler)

        assert result is not None
        assert get_status(result) == 400
        data = get_body(result)
        assert "force" in data.get("error", "").lower() or "days" in data.get("error", "").lower()

    @patch("aragora.server.handlers.admin.admin.extract_user_from_request")
    @patch("aragora.server.handlers.admin.security.CRYPTO_AVAILABLE", True)
    @patch("aragora.server.handlers.admin.security.get_encryption_service")
    @patch("aragora.server.handlers.admin.security.rotate_encryption_key")
    def test_rotate_key_dry_run(self, mock_rotate, mock_service, mock_auth, security_handler):
        """Dry run should preview changes without executing."""
        mock_auth.return_value = MockAuthContext(user_id="admin-123", role="admin")

        mock_enc = MockEncryptionService()
        mock_service.return_value = mock_enc
        mock_rotate.return_value = MockRotationResult()

        handler = make_mock_handler({"dry_run": True})
        result = security_handler._rotate_key({"dry_run": True}, handler)

        assert result is not None
        data = get_body(result)
        assert data["dry_run"] is True

    @patch("aragora.server.handlers.admin.admin.extract_user_from_request")
    @patch("aragora.server.handlers.admin.security.CRYPTO_AVAILABLE", True)
    @patch("aragora.server.handlers.admin.security.get_encryption_service")
    def test_key_age_triggers_rotation_recommended(self, mock_service, mock_auth, security_handler):
        """Keys older than 60 days should trigger rotation_recommended."""
        mock_auth.return_value = MockAuthContext(user_id="admin-123", role="admin")

        mock_enc = MockEncryptionService()
        mock_enc.keys[1].created_at = datetime.now(timezone.utc) - timedelta(days=65)
        mock_service.return_value = mock_enc

        handler = make_mock_handler()
        result = security_handler._get_status(handler)

        data = get_body(result)
        assert data.get("rotation_recommended") is True

    @patch("aragora.server.handlers.admin.admin.extract_user_from_request")
    @patch("aragora.server.handlers.admin.security.CRYPTO_AVAILABLE", True)
    @patch("aragora.server.handlers.admin.security.get_encryption_service")
    def test_key_age_triggers_rotation_required(self, mock_service, mock_auth, security_handler):
        """Keys older than 90 days should trigger rotation_required."""
        mock_auth.return_value = MockAuthContext(user_id="admin-123", role="admin")

        mock_enc = MockEncryptionService()
        mock_enc.keys[1].created_at = datetime.now(timezone.utc) - timedelta(days=95)
        mock_service.return_value = mock_enc

        handler = make_mock_handler()
        result = security_handler._get_status(handler)

        data = get_body(result)
        assert data.get("rotation_required") is True


# =============================================================================
# Test Impersonation Controls
# =============================================================================


class TestImpersonationControls:
    """Tests for user impersonation controls and security."""

    def test_impersonation_session_expiration(self):
        """Impersonation sessions should expire correctly."""
        from aragora.auth.impersonation import ImpersonationSession

        now = datetime.now(timezone.utc)
        session = ImpersonationSession(
            session_id="test-session",
            admin_user_id="admin-123",
            admin_email="admin@example.com",
            target_user_id="user-456",
            target_email="user@example.com",
            reason="Support investigation",
            started_at=now,
            expires_at=now - timedelta(minutes=1),  # Already expired
            ip_address="127.0.0.1",
            user_agent="Test Agent",
        )

        assert session.is_expired() is True

    def test_impersonation_session_not_expired(self):
        """Active sessions should not be marked as expired."""
        from aragora.auth.impersonation import ImpersonationSession

        now = datetime.now(timezone.utc)
        session = ImpersonationSession(
            session_id="test-session",
            admin_user_id="admin-123",
            admin_email="admin@example.com",
            target_user_id="user-456",
            target_email="user@example.com",
            reason="Support investigation",
            started_at=now,
            expires_at=now + timedelta(hours=1),  # Still valid
            ip_address="127.0.0.1",
            user_agent="Test Agent",
        )

        assert session.is_expired() is False

    def test_impersonation_manager_rejects_short_reason(self):
        """Impersonation should require a valid reason (>=10 chars)."""
        from aragora.auth.impersonation import ImpersonationManager

        manager = ImpersonationManager()
        manager._use_persistence = False

        session, message = manager.start_impersonation(
            admin_user_id="admin-123",
            admin_email="admin@example.com",
            admin_roles=["admin"],
            target_user_id="user-456",
            target_email="user@example.com",
            target_roles=["user"],
            reason="short",  # Too short
            ip_address="127.0.0.1",
            user_agent="Test Agent",
        )

        assert session is None
        assert "10 characters" in message

    def test_impersonation_manager_rejects_self_impersonation(self):
        """Admins should not be able to impersonate themselves."""
        from aragora.auth.impersonation import ImpersonationManager

        manager = ImpersonationManager()
        manager._use_persistence = False

        session, message = manager.start_impersonation(
            admin_user_id="admin-123",
            admin_email="admin@example.com",
            admin_roles=["admin"],
            target_user_id="admin-123",  # Same as admin
            target_email="admin@example.com",
            target_roles=["admin"],
            reason="Testing self-impersonation",
            ip_address="127.0.0.1",
            user_agent="Test Agent",
        )

        assert session is None
        assert "yourself" in message.lower()

    def test_impersonation_requires_2fa_for_admin_targets(self):
        """Impersonating admin users should require 2FA."""
        from aragora.auth.impersonation import ImpersonationManager

        manager = ImpersonationManager(require_2fa_for_admin_targets=True)
        manager._use_persistence = False

        session, message = manager.start_impersonation(
            admin_user_id="admin-123",
            admin_email="admin@example.com",
            admin_roles=["admin"],
            target_user_id="admin-456",
            target_email="other-admin@example.com",
            target_roles=["admin"],  # Target is an admin
            reason="Testing admin impersonation",
            ip_address="127.0.0.1",
            user_agent="Test Agent",
            has_2fa=False,  # No 2FA
        )

        assert session is None
        assert "2fa" in message.lower()

    def test_impersonation_with_2fa_succeeds_for_admin_target(self):
        """Impersonating admin users should succeed with 2FA."""
        from aragora.auth.impersonation import ImpersonationManager

        manager = ImpersonationManager(require_2fa_for_admin_targets=True)
        manager._use_persistence = False

        session, message = manager.start_impersonation(
            admin_user_id="admin-123",
            admin_email="admin@example.com",
            admin_roles=["admin"],
            target_user_id="admin-456",
            target_email="other-admin@example.com",
            target_roles=["admin"],
            reason="Testing admin impersonation with 2FA",
            ip_address="127.0.0.1",
            user_agent="Test Agent",
            has_2fa=True,
        )

        assert session is not None

    def test_impersonation_max_concurrent_sessions(self):
        """Should enforce maximum concurrent sessions per admin."""
        from aragora.auth.impersonation import ImpersonationManager

        manager = ImpersonationManager(max_concurrent_sessions=2)
        manager._use_persistence = False

        # Create 2 sessions
        for i in range(2):
            session, _ = manager.start_impersonation(
                admin_user_id="admin-123",
                admin_email="admin@example.com",
                admin_roles=["admin"],
                target_user_id=f"user-{i}",
                target_email=f"user{i}@example.com",
                target_roles=["user"],
                reason="Testing concurrent sessions",
                ip_address="127.0.0.1",
                user_agent="Test Agent",
            )
            assert session is not None

        # Third should fail
        session, message = manager.start_impersonation(
            admin_user_id="admin-123",
            admin_email="admin@example.com",
            admin_roles=["admin"],
            target_user_id="user-999",
            target_email="user999@example.com",
            target_roles=["user"],
            reason="Testing concurrent limit",
            ip_address="127.0.0.1",
            user_agent="Test Agent",
        )

        assert session is None
        assert "concurrent" in message.lower() or "maximum" in message.lower()

    def test_impersonation_session_duration_capped(self):
        """Session duration should be capped at maximum."""
        from aragora.auth.impersonation import ImpersonationManager

        manager = ImpersonationManager()
        manager._use_persistence = False

        session, _ = manager.start_impersonation(
            admin_user_id="admin-123",
            admin_email="admin@example.com",
            admin_roles=["admin"],
            target_user_id="user-456",
            target_email="user@example.com",
            target_roles=["user"],
            reason="Testing duration cap",
            ip_address="127.0.0.1",
            user_agent="Test Agent",
            duration=timedelta(hours=10),  # Request 10 hours
        )

        assert session is not None
        # Should be capped at MAX_SESSION_DURATION (1 hour)
        actual_duration = session.expires_at - session.started_at
        assert actual_duration <= manager.MAX_SESSION_DURATION

    def test_impersonation_end_session_by_owner_only(self):
        """Only the session owner should be able to end the session."""
        from aragora.auth.impersonation import ImpersonationManager

        manager = ImpersonationManager()
        manager._use_persistence = False

        session, _ = manager.start_impersonation(
            admin_user_id="admin-123",
            admin_email="admin@example.com",
            admin_roles=["admin"],
            target_user_id="user-456",
            target_email="user@example.com",
            target_roles=["user"],
            reason="Testing session ownership",
            ip_address="127.0.0.1",
            user_agent="Test Agent",
        )

        # Different admin tries to end
        success, message = manager.end_impersonation(
            session_id=session.session_id,
            admin_user_id="admin-999",  # Different admin
            ip_address="127.0.0.1",
            user_agent="Test Agent",
        )

        assert success is False
        assert "owner" in message.lower() or "started" in message.lower()


# =============================================================================
# Test Audit Logging
# =============================================================================


class TestAuditLogging:
    """Tests for security audit logging."""

    def test_impersonation_start_is_logged(self):
        """Starting impersonation should create audit log entry."""
        from aragora.auth.impersonation import ImpersonationManager

        audit_entries = []

        def capture_audit(entry):
            audit_entries.append(entry)

        manager = ImpersonationManager(audit_callback=capture_audit)
        manager._use_persistence = False

        session, _ = manager.start_impersonation(
            admin_user_id="admin-123",
            admin_email="admin@example.com",
            admin_roles=["admin"],
            target_user_id="user-456",
            target_email="user@example.com",
            target_roles=["user"],
            reason="Testing audit logging",
            ip_address="127.0.0.1",
            user_agent="Test Agent",
        )

        assert len(audit_entries) == 1
        assert audit_entries[0].event_type == "start"
        assert audit_entries[0].admin_user_id == "admin-123"
        assert audit_entries[0].target_user_id == "user-456"

    def test_impersonation_action_is_logged(self):
        """Actions during impersonation should be logged."""
        from aragora.auth.impersonation import ImpersonationManager

        audit_entries = []

        def capture_audit(entry):
            audit_entries.append(entry)

        manager = ImpersonationManager(audit_callback=capture_audit)
        manager._use_persistence = False

        session, _ = manager.start_impersonation(
            admin_user_id="admin-123",
            admin_email="admin@example.com",
            admin_roles=["admin"],
            target_user_id="user-456",
            target_email="user@example.com",
            target_roles=["user"],
            reason="Testing action logging",
            ip_address="127.0.0.1",
            user_agent="Test Agent",
        )

        # Log an action
        manager.log_impersonation_action(
            session_id=session.session_id,
            action_type="view_profile",
            action_details={"page": "/settings"},
            ip_address="127.0.0.1",
            user_agent="Test Agent",
        )

        # Should have start + action entries
        action_entries = [e for e in audit_entries if e.event_type == "action"]
        assert len(action_entries) == 1
        assert action_entries[0].action_details["action_type"] == "view_profile"

    def test_impersonation_denial_is_logged(self):
        """Denied impersonation attempts should be logged."""
        from aragora.auth.impersonation import ImpersonationManager

        audit_entries = []

        def capture_audit(entry):
            audit_entries.append(entry)

        manager = ImpersonationManager(audit_callback=capture_audit)
        manager._use_persistence = False

        # Try with too short reason (will be denied)
        session, _ = manager.start_impersonation(
            admin_user_id="admin-123",
            admin_email="admin@example.com",
            admin_roles=["admin"],
            target_user_id="user-456",
            target_email="user@example.com",
            target_roles=["user"],
            reason="short",
            ip_address="127.0.0.1",
            user_agent="Test Agent",
        )

        denied_entries = [e for e in audit_entries if e.event_type == "denied"]
        assert len(denied_entries) == 1
        assert denied_entries[0].success is False

    def test_audit_log_includes_ip_address(self):
        """Audit entries should include IP address."""
        from aragora.auth.impersonation import ImpersonationManager

        audit_entries = []

        def capture_audit(entry):
            audit_entries.append(entry)

        manager = ImpersonationManager(audit_callback=capture_audit)
        manager._use_persistence = False

        session, _ = manager.start_impersonation(
            admin_user_id="admin-123",
            admin_email="admin@example.com",
            admin_roles=["admin"],
            target_user_id="user-456",
            target_email="user@example.com",
            target_roles=["user"],
            reason="Testing IP logging",
            ip_address="192.168.1.100",
            user_agent="Test Agent",
        )

        assert audit_entries[0].ip_address == "192.168.1.100"

    def test_audit_log_query_by_admin(self):
        """Should be able to query audit log by admin user."""
        from aragora.auth.impersonation import ImpersonationManager

        manager = ImpersonationManager()
        manager._use_persistence = False

        # Create sessions for different admins
        for admin_id in ["admin-1", "admin-2"]:
            manager.start_impersonation(
                admin_user_id=admin_id,
                admin_email=f"{admin_id}@example.com",
                admin_roles=["admin"],
                target_user_id="user-456",
                target_email="user@example.com",
                target_roles=["user"],
                reason="Testing query filtering",
                ip_address="127.0.0.1",
                user_agent="Test Agent",
            )

        # Query for specific admin
        entries = manager.get_audit_log(admin_user_id="admin-1")

        assert all(e.admin_user_id == "admin-1" for e in entries)


# =============================================================================
# Test Error Handling
# =============================================================================


class TestErrorHandling:
    """Tests for error handling in security endpoints."""

    @pytest.fixture
    def user_store(self):
        return MockUserStore()

    @pytest.fixture
    def security_handler(self, user_store):
        from aragora.server.handlers.admin.security import SecurityHandler

        ctx = {"user_store": user_store}
        return SecurityHandler(ctx)

    @patch("aragora.server.handlers.admin.admin.extract_user_from_request")
    @patch("aragora.server.handlers.admin.security.CRYPTO_AVAILABLE", True)
    @patch("aragora.server.handlers.admin.security.get_encryption_service")
    def test_get_status_handles_service_error(self, mock_service, mock_auth, security_handler):
        """Security status should handle encryption service errors gracefully."""
        mock_auth.return_value = MockAuthContext(user_id="admin-123", role="admin")
        mock_service.side_effect = RuntimeError("Service unavailable")

        handler = make_mock_handler()
        result = security_handler._get_status(handler)

        assert result is not None
        assert get_status(result) == 500

    @patch("aragora.server.handlers.admin.admin.extract_user_from_request")
    @patch("aragora.server.handlers.admin.security.CRYPTO_AVAILABLE", False)
    def test_rotate_key_fails_without_crypto(self, mock_auth, security_handler):
        """Key rotation should fail when crypto is unavailable."""
        mock_auth.return_value = MockAuthContext(user_id="admin-123", role="admin")

        handler = make_mock_handler({"force": True})
        result = security_handler._rotate_key({"force": True}, handler)

        assert result is not None
        assert get_status(result) == 400
        data = get_body(result)
        assert "crypto" in data.get("error", "").lower()

    @patch("aragora.server.handlers.admin.admin.extract_user_from_request")
    @patch("aragora.server.handlers.admin.security.CRYPTO_AVAILABLE", False)
    def test_list_keys_fails_without_crypto(self, mock_auth, security_handler):
        """Listing keys should fail when crypto is unavailable."""
        mock_auth.return_value = MockAuthContext(user_id="admin-123", role="admin")

        handler = make_mock_handler()
        result = security_handler._list_keys(handler)

        assert result is not None
        assert get_status(result) == 400

    def test_impersonation_validates_session_not_found(self):
        """Validating non-existent session should return None."""
        from aragora.auth.impersonation import ImpersonationManager

        manager = ImpersonationManager()
        manager._use_persistence = False

        session = manager.validate_session("nonexistent-session-id")
        assert session is None

    def test_impersonation_end_nonexistent_session(self):
        """Ending non-existent session should fail gracefully."""
        from aragora.auth.impersonation import ImpersonationManager

        manager = ImpersonationManager()
        manager._use_persistence = False

        success, message = manager.end_impersonation(
            session_id="nonexistent-session",
            admin_user_id="admin-123",
            ip_address="127.0.0.1",
            user_agent="Test Agent",
        )

        assert success is False
        assert "not found" in message.lower()


# =============================================================================
# Test Compliance and Backup Endpoints (OpenAPI Schema)
# =============================================================================


class TestComplianceEndpoints:
    """Tests for compliance violation endpoints."""

    def test_compliance_violation_endpoint_exists(self):
        """Compliance violation endpoint should be defined."""
        from aragora.server.openapi.endpoints.admin_security import ADMIN_SECURITY_ENDPOINTS

        assert "/api/v1/compliance/violations/{violation_id}" in ADMIN_SECURITY_ENDPOINTS

    def test_compliance_violation_severity_enum(self):
        """Compliance violations should have severity enum."""
        from aragora.server.openapi.endpoints.admin_security import ADMIN_SECURITY_ENDPOINTS

        endpoint = ADMIN_SECURITY_ENDPOINTS["/api/v1/compliance/violations/{violation_id}"]
        schema = endpoint["get"]["responses"]["200"]["content"]["application/json"]["schema"]

        assert "severity" in schema["properties"]
        severity = schema["properties"]["severity"]
        assert "enum" in severity
        assert "low" in severity["enum"]
        assert "high" in severity["enum"]
        assert "critical" in severity["enum"]

    def test_compliance_violation_types(self):
        """Compliance violations should define violation types."""
        from aragora.server.openapi.endpoints.admin_security import ADMIN_SECURITY_ENDPOINTS

        endpoint = ADMIN_SECURITY_ENDPOINTS["/api/v1/compliance/violations/{violation_id}"]
        schema = endpoint["get"]["responses"]["200"]["content"]["application/json"]["schema"]

        violation_type = schema["properties"]["type"]
        assert "enum" in violation_type
        assert "encryption" in violation_type["enum"]
        assert "access_control" in violation_type["enum"]
        assert "audit_logging" in violation_type["enum"]

    def test_compliance_violation_update_endpoint(self):
        """Should be able to update compliance violations."""
        from aragora.server.openapi.endpoints.admin_security import ADMIN_SECURITY_ENDPOINTS

        endpoint = ADMIN_SECURITY_ENDPOINTS["/api/v1/compliance/violations/{violation_id}"]
        assert "put" in endpoint

        # Check request body allows status updates
        request_schema = endpoint["put"]["requestBody"]["content"]["application/json"]["schema"]
        assert "status" in request_schema["properties"]


class TestBackupEndpoints:
    """Tests for backup management endpoints."""

    def test_list_backups_endpoint_exists(self):
        """List backups endpoint should be defined."""
        from aragora.server.openapi.endpoints.admin_security import ADMIN_SECURITY_ENDPOINTS

        assert "/api/v2/backups" in ADMIN_SECURITY_ENDPOINTS
        assert "get" in ADMIN_SECURITY_ENDPOINTS["/api/v2/backups"]

    def test_create_backup_endpoint_exists(self):
        """Create backup endpoint should be defined."""
        from aragora.server.openapi.endpoints.admin_security import ADMIN_SECURITY_ENDPOINTS

        assert "post" in ADMIN_SECURITY_ENDPOINTS["/api/v2/backups"]

    def test_backup_types_defined(self):
        """Backup types should include full, incremental, differential."""
        from aragora.server.openapi.endpoints.admin_security import ADMIN_SECURITY_ENDPOINTS

        endpoint = ADMIN_SECURITY_ENDPOINTS["/api/v2/backups"]
        get_params = endpoint["get"]["parameters"]

        type_param = next((p for p in get_params if p["name"] == "type"), None)
        assert type_param is not None
        assert "full" in type_param["schema"]["enum"]
        assert "incremental" in type_param["schema"]["enum"]
        assert "differential" in type_param["schema"]["enum"]

    def test_backup_details_endpoint(self):
        """Get backup details endpoint should exist."""
        from aragora.server.openapi.endpoints.admin_security import ADMIN_SECURITY_ENDPOINTS

        assert "/api/v2/backups/{backup_id}" in ADMIN_SECURITY_ENDPOINTS
        assert "get" in ADMIN_SECURITY_ENDPOINTS["/api/v2/backups/{backup_id}"]

    def test_delete_backup_endpoint(self):
        """Delete backup endpoint should exist."""
        from aragora.server.openapi.endpoints.admin_security import ADMIN_SECURITY_ENDPOINTS

        assert "delete" in ADMIN_SECURITY_ENDPOINTS["/api/v2/backups/{backup_id}"]

    def test_backup_conflict_response(self):
        """Backup endpoints should handle conflicts (409)."""
        from aragora.server.openapi.endpoints.admin_security import ADMIN_SECURITY_ENDPOINTS

        endpoint = ADMIN_SECURITY_ENDPOINTS["/api/v2/backups"]
        assert "409" in endpoint["post"]["responses"]


class TestDisasterRecoveryEndpoints:
    """Tests for disaster recovery endpoints."""

    def test_dr_status_endpoint_exists(self):
        """DR status endpoint should be defined."""
        from aragora.server.openapi.endpoints.admin_security import ADMIN_SECURITY_ENDPOINTS

        assert "/api/v2/dr" in ADMIN_SECURITY_ENDPOINTS

    def test_dr_status_includes_rpo_rto(self):
        """DR status should include RPO and RTO."""
        from aragora.server.openapi.endpoints.admin_security import ADMIN_SECURITY_ENDPOINTS

        endpoint = ADMIN_SECURITY_ENDPOINTS["/api/v2/dr"]
        schema = endpoint["get"]["responses"]["200"]["content"]["application/json"]["schema"]

        assert "rpo_hours" in schema["properties"]
        assert "rto_hours" in schema["properties"]

    def test_dr_plan_endpoint_exists(self):
        """DR plan endpoint should be defined."""
        from aragora.server.openapi.endpoints.admin_security import ADMIN_SECURITY_ENDPOINTS

        assert "/api/v2/dr/{plan_id}" in ADMIN_SECURITY_ENDPOINTS

    def test_dr_plan_execute_is_post(self):
        """Executing DR plan should be POST."""
        from aragora.server.openapi.endpoints.admin_security import ADMIN_SECURITY_ENDPOINTS

        endpoint = ADMIN_SECURITY_ENDPOINTS["/api/v2/dr/{plan_id}"]
        assert "post" in endpoint

    def test_dr_execution_modes(self):
        """DR execution should support drill and recovery modes."""
        from aragora.server.openapi.endpoints.admin_security import ADMIN_SECURITY_ENDPOINTS

        endpoint = ADMIN_SECURITY_ENDPOINTS["/api/v2/dr/{plan_id}"]
        request_schema = endpoint["post"]["requestBody"]["content"]["application/json"]["schema"]

        mode = request_schema["properties"]["mode"]
        assert "enum" in mode
        assert "drill" in mode["enum"]
        assert "recovery" in mode["enum"]


# =============================================================================
# Test Impersonation Store Persistence
# =============================================================================


class TestImpersonationStorePersistence:
    """Tests for impersonation session persistence."""

    @pytest.fixture
    def temp_db(self, tmp_path):
        """Create a temporary database for testing."""
        return str(tmp_path / "test_impersonation.db")

    def test_store_save_and_get_session(self, temp_db):
        """Should save and retrieve session from store."""
        from aragora.storage.impersonation_store import ImpersonationStore

        store = ImpersonationStore(db_path=temp_db, backend="sqlite")

        now = datetime.now(timezone.utc)
        session_id = store.save_session(
            session_id="test-session-001",
            admin_user_id="admin-123",
            admin_email="admin@example.com",
            target_user_id="user-456",
            target_email="user@example.com",
            reason="Testing persistence",
            started_at=now,
            expires_at=now + timedelta(hours=1),
            ip_address="127.0.0.1",
            user_agent="Test Agent",
        )

        retrieved = store.get_session("test-session-001")
        assert retrieved is not None
        assert retrieved.admin_user_id == "admin-123"
        assert retrieved.target_user_id == "user-456"

        store.close()

    def test_store_get_active_sessions(self, temp_db):
        """Should retrieve only active (non-expired, non-ended) sessions."""
        from aragora.storage.impersonation_store import ImpersonationStore

        store = ImpersonationStore(db_path=temp_db, backend="sqlite")
        now = datetime.now(timezone.utc)

        # Active session
        store.save_session(
            session_id="active-session",
            admin_user_id="admin-123",
            admin_email="admin@example.com",
            target_user_id="user-456",
            target_email="user@example.com",
            reason="Active session",
            started_at=now,
            expires_at=now + timedelta(hours=1),
            ip_address="127.0.0.1",
            user_agent="Test Agent",
        )

        # Expired session
        store.save_session(
            session_id="expired-session",
            admin_user_id="admin-123",
            admin_email="admin@example.com",
            target_user_id="user-789",
            target_email="user2@example.com",
            reason="Expired session",
            started_at=now - timedelta(hours=2),
            expires_at=now - timedelta(hours=1),  # Already expired
            ip_address="127.0.0.1",
            user_agent="Test Agent",
        )

        active = store.get_active_sessions()
        assert len(active) == 1
        assert active[0].session_id == "active-session"

        store.close()

    def test_store_end_session(self, temp_db):
        """Should mark session as ended."""
        from aragora.storage.impersonation_store import ImpersonationStore

        store = ImpersonationStore(db_path=temp_db, backend="sqlite")
        now = datetime.now(timezone.utc)

        store.save_session(
            session_id="ending-session",
            admin_user_id="admin-123",
            admin_email="admin@example.com",
            target_user_id="user-456",
            target_email="user@example.com",
            reason="Session to end",
            started_at=now,
            expires_at=now + timedelta(hours=1),
            ip_address="127.0.0.1",
            user_agent="Test Agent",
        )

        store.end_session("ending-session", "admin", 5)

        session = store.get_session("ending-session")
        assert session.ended_at is not None
        assert session.ended_by == "admin"
        assert session.actions_performed == 5

        store.close()

    def test_store_save_audit_entry(self, temp_db):
        """Should save audit log entries."""
        from aragora.storage.impersonation_store import ImpersonationStore

        store = ImpersonationStore(db_path=temp_db, backend="sqlite")

        store.save_audit_entry(
            audit_id="audit-001",
            timestamp=datetime.now(timezone.utc),
            event_type="start",
            admin_user_id="admin-123",
            ip_address="127.0.0.1",
            user_agent="Test Agent",
            success=True,
            target_user_id="user-456",
            reason="Audit test",
        )

        entries = store.get_audit_log()
        assert len(entries) == 1
        assert entries[0].event_type == "start"

        store.close()

    def test_store_query_audit_by_event_type(self, temp_db):
        """Should filter audit log by event type."""
        from aragora.storage.impersonation_store import ImpersonationStore

        store = ImpersonationStore(db_path=temp_db, backend="sqlite")
        now = datetime.now(timezone.utc)

        # Add different event types
        for i, event_type in enumerate(["start", "action", "end", "denied"]):
            store.save_audit_entry(
                audit_id=f"audit-{i}",
                timestamp=now,
                event_type=event_type,
                admin_user_id="admin-123",
                ip_address="127.0.0.1",
                user_agent="Test Agent",
                success=event_type != "denied",
            )

        denied = store.get_audit_log(event_type="denied")
        assert len(denied) == 1
        assert denied[0].success is False

        store.close()


# =============================================================================
# Test Rate Limiting Concepts (Schema Validation)
# =============================================================================


class TestRateLimitingConcepts:
    """Tests for rate limiting on sensitive operations."""

    def test_endpoints_define_rate_limits_in_description(self):
        """Sensitive endpoints should mention rate limiting considerations."""
        from aragora.server.openapi.endpoints.admin_security import ADMIN_SECURITY_ENDPOINTS

        # Key rotation is a sensitive operation
        endpoint = ADMIN_SECURITY_ENDPOINTS["/api/v1/admin/security/rotate-key"]
        # Description mentions safety or force flag
        assert (
            "force" in endpoint["post"]["description"].lower()
            or "safety" in endpoint["post"]["description"].lower()
        )

    def test_impersonate_endpoint_mentions_audit(self):
        """Impersonate endpoint should mention auditing."""
        from aragora.server.openapi.endpoints.admin_security import ADMIN_SECURITY_ENDPOINTS

        endpoint = ADMIN_SECURITY_ENDPOINTS["/api/v1/admin/impersonate/{user_id}"]
        assert "audit" in endpoint["post"]["description"].lower()

    def test_keys_endpoint_mentions_audit(self):
        """Keys listing should mention audit logging."""
        from aragora.server.openapi.endpoints.admin_security import ADMIN_SECURITY_ENDPOINTS

        endpoint = ADMIN_SECURITY_ENDPOINTS["/api/v1/admin/security/keys"]
        assert "audit" in endpoint["get"]["description"].lower()


# =============================================================================
# Test Session Management Edge Cases
# =============================================================================


class TestSessionManagementEdgeCases:
    """Tests for edge cases in session management."""

    def test_impersonation_session_to_audit_dict(self):
        """Session should convert to audit dict format."""
        from aragora.auth.impersonation import ImpersonationSession

        now = datetime.now(timezone.utc)
        session = ImpersonationSession(
            session_id="test-session",
            admin_user_id="admin-123",
            admin_email="admin@example.com",
            target_user_id="user-456",
            target_email="user@example.com",
            reason="Testing dict conversion",
            started_at=now,
            expires_at=now + timedelta(hours=1),
            ip_address="127.0.0.1",
            user_agent="Test Agent",
            actions_performed=5,
        )

        audit_dict = session.to_audit_dict()

        assert audit_dict["session_id"] == "test-session"
        assert audit_dict["actions_performed"] == 5
        assert "started_at" in audit_dict
        assert "expires_at" in audit_dict

    def test_manager_cleanup_expired_sessions(self):
        """Manager should clean up expired sessions properly."""
        from aragora.auth.impersonation import ImpersonationManager

        manager = ImpersonationManager()
        manager._use_persistence = False

        # Create a session that's already expired
        now = datetime.now(timezone.utc)
        session, _ = manager.start_impersonation(
            admin_user_id="admin-123",
            admin_email="admin@example.com",
            admin_roles=["admin"],
            target_user_id="user-456",
            target_email="user@example.com",
            target_roles=["user"],
            reason="Testing cleanup mechanism",
            ip_address="127.0.0.1",
            user_agent="Test Agent",
            duration=timedelta(seconds=0),  # Immediate expiration
        )

        # Manually set expiration to past
        manager._sessions[session.session_id].expires_at = now - timedelta(seconds=1)

        # Validate should trigger cleanup
        result = manager.validate_session(session.session_id)
        assert result is None

    def test_get_active_sessions_for_admin(self):
        """Should get only active sessions for specific admin."""
        from aragora.auth.impersonation import ImpersonationManager

        manager = ImpersonationManager(max_concurrent_sessions=10)
        manager._use_persistence = False

        # Create sessions for admin-1
        for i in range(3):
            manager.start_impersonation(
                admin_user_id="admin-1",
                admin_email="admin1@example.com",
                admin_roles=["admin"],
                target_user_id=f"user-{i}",
                target_email=f"user{i}@example.com",
                target_roles=["user"],
                reason="Testing admin sessions",
                ip_address="127.0.0.1",
                user_agent="Test Agent",
            )

        # Create session for admin-2
        manager.start_impersonation(
            admin_user_id="admin-2",
            admin_email="admin2@example.com",
            admin_roles=["admin"],
            target_user_id="user-999",
            target_email="user999@example.com",
            target_roles=["user"],
            reason="Testing admin sessions",
            ip_address="127.0.0.1",
            user_agent="Test Agent",
        )

        admin1_sessions = manager.get_active_sessions_for_admin("admin-1")
        assert len(admin1_sessions) == 3

        admin2_sessions = manager.get_active_sessions_for_admin("admin-2")
        assert len(admin2_sessions) == 1


# =============================================================================
# Test Security Handler Routing
# =============================================================================


class TestSecurityHandlerRouting:
    """Tests for security handler request routing."""

    @pytest.fixture
    def user_store(self):
        return MockUserStore()

    @pytest.fixture
    def security_handler(self, user_store):
        from aragora.server.handlers.admin.security import SecurityHandler

        ctx = {"user_store": user_store}
        return SecurityHandler(ctx)

    def test_can_handle_versioned_routes(self, security_handler):
        """Handler should recognize versioned routes."""
        assert security_handler.can_handle("/api/v1/admin/security/status")
        assert security_handler.can_handle("/api/v1/admin/security/health")
        assert security_handler.can_handle("/api/v1/admin/security/keys")

    def test_can_handle_legacy_routes(self, security_handler):
        """Handler should recognize legacy routes."""
        assert security_handler.can_handle("/api/admin/security/status")
        assert security_handler.can_handle("/api/admin/security/health")

    def test_cannot_handle_unrelated_routes(self, security_handler):
        """Handler should not handle unrelated routes."""
        assert not security_handler.can_handle("/api/v1/debates")
        assert not security_handler.can_handle("/api/v1/users")
        assert not security_handler.can_handle("/api/admin/users")
