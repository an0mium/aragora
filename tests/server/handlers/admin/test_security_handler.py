"""
Tests for admin security handler.

Tests cover:
- Encryption status retrieval
- Key rotation (dry run and actual)
- Encryption health checks
- Key listing
- Authorization requirements
- Error handling when crypto unavailable
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.admin.security import SecurityHandler


class MockEncryptionKey:
    """Mock encryption key for testing."""

    def __init__(
        self,
        key_id: str = "key-001",
        version: int = 1,
        created_at: datetime | None = None,
    ):
        self.key_id = key_id
        self.version = version
        self.created_at = created_at or datetime.now(timezone.utc)


class MockEncryptionService:
    """Mock encryption service for testing."""

    def __init__(
        self,
        active_key: MockEncryptionKey | None = None,
        keys: list[MockEncryptionKey] | None = None,
    ):
        self._active_key = active_key or MockEncryptionKey()
        self._keys = keys or [self._active_key]

    def get_active_key(self) -> MockEncryptionKey | None:
        return self._active_key

    def get_active_key_id(self) -> str | None:
        return self._active_key.key_id if self._active_key else None

    def list_keys(self) -> list[dict[str, Any]]:
        """Return keys as dictionaries, matching real EncryptionService.list_keys()."""
        return [
            {
                "key_id": key.key_id,
                "version": key.version,
                "created_at": key.created_at.isoformat(),
                "is_active": key == self._active_key,
            }
            for key in self._keys
        ]

    def encrypt(self, data: bytes) -> bytes:
        return b"encrypted:" + data

    def decrypt(self, data: bytes) -> bytes:
        if data.startswith(b"encrypted:"):
            return data[10:]
        return data


class MockRotationResult:
    """Mock result for key rotation."""

    def __init__(
        self,
        success: bool = True,
        old_version: int = 1,
        new_version: int = 2,
    ):
        self.success = success
        self.old_key_version = old_version
        self.new_key_version = new_version
        self.stores_processed = ["users", "credentials"]
        self.records_reencrypted = 150
        self.failed_records = 0 if success else 5
        self.duration_seconds = 2.5
        self.errors = [] if success else ["Sample error"]


@pytest.fixture
def mock_server_context() -> MagicMock:
    """Create mock server context."""
    return MagicMock()


@pytest.fixture
def security_handler(mock_server_context: MagicMock) -> SecurityHandler:
    """Create security handler for tests."""
    return SecurityHandler(mock_server_context)


@pytest.fixture
def mock_handler() -> MagicMock:
    """Create mock request handler with auth context."""
    handler = MagicMock()
    handler._context = {"user": {"id": "admin-1", "roles": ["admin"]}}
    handler._auth_context = MagicMock()
    handler._auth_context.user_id = "admin-1"
    handler._auth_context.roles = ["admin"]
    return handler


class TestCanHandle:
    """Tests for route matching."""

    def test_handles_versioned_status_route(self, security_handler: SecurityHandler):
        """Handler matches versioned status route."""
        assert security_handler.can_handle("/api/v1/admin/security/status")

    def test_handles_unversioned_status_route(self, security_handler: SecurityHandler):
        """Handler matches unversioned status route."""
        assert security_handler.can_handle("/api/admin/security/status")

    def test_handles_health_route(self, security_handler: SecurityHandler):
        """Handler matches health route."""
        assert security_handler.can_handle("/api/v1/admin/security/health")
        assert security_handler.can_handle("/api/admin/security/health")

    def test_handles_rotate_key_route(self, security_handler: SecurityHandler):
        """Handler matches rotate-key route."""
        assert security_handler.can_handle("/api/v1/admin/security/rotate-key")
        assert security_handler.can_handle("/api/admin/security/rotate-key")

    def test_handles_keys_route(self, security_handler: SecurityHandler):
        """Handler matches keys route."""
        assert security_handler.can_handle("/api/v1/admin/security/keys")
        assert security_handler.can_handle("/api/admin/security/keys")

    def test_rejects_unknown_route(self, security_handler: SecurityHandler):
        """Handler rejects unknown routes."""
        assert not security_handler.can_handle("/api/admin/other")
        assert not security_handler.can_handle("/api/v1/security/status")


# Patch path for runtime imports
ENCRYPTION_MODULE = "aragora.security.encryption"
MIGRATION_MODULE = "aragora.security.migration"


class TestGetStatus:
    """Tests for _get_status endpoint."""

    def test_status_with_active_key(
        self, security_handler: SecurityHandler, mock_handler: MagicMock
    ):
        """Status returns key information when active key exists."""
        service = MockEncryptionService()

        with patch(f"{ENCRYPTION_MODULE}.get_encryption_service", return_value=service):
            with patch(f"{ENCRYPTION_MODULE}.CRYPTO_AVAILABLE", True):
                result = security_handler._get_status.__wrapped__(security_handler, mock_handler)

        assert result.status_code == 200
        body = json.loads(result.body.decode())
        assert body["crypto_available"] is True
        assert body["active_key_id"] == "key-001"
        assert body["key_version"] == 1
        assert "key_age_days" in body
        assert "rotation_recommended" in body

    def test_status_crypto_unavailable(
        self, security_handler: SecurityHandler, mock_handler: MagicMock
    ):
        """Status handles missing cryptography library."""
        with patch(f"{ENCRYPTION_MODULE}.CRYPTO_AVAILABLE", False):
            result = security_handler._get_status.__wrapped__(security_handler, mock_handler)

        assert result.status_code == 200
        body = json.loads(result.body.decode())
        assert body["crypto_available"] is False
        assert "error" in body

    def test_status_no_active_key(self, security_handler: SecurityHandler, mock_handler: MagicMock):
        """Status warns when no active key found."""
        service = MockEncryptionService(active_key=None)
        service._active_key = None

        with patch(f"{ENCRYPTION_MODULE}.get_encryption_service", return_value=service):
            with patch(f"{ENCRYPTION_MODULE}.CRYPTO_AVAILABLE", True):
                result = security_handler._get_status.__wrapped__(security_handler, mock_handler)

        assert result.status_code == 200
        body = json.loads(result.body.decode())
        assert body["warning"] == "No active encryption key found"

    def test_status_rotation_recommended(
        self, security_handler: SecurityHandler, mock_handler: MagicMock
    ):
        """Status recommends rotation for old keys (>60 days)."""
        old_key = MockEncryptionKey(created_at=datetime.now(timezone.utc) - timedelta(days=65))
        service = MockEncryptionService(active_key=old_key)

        with patch(f"{ENCRYPTION_MODULE}.get_encryption_service", return_value=service):
            with patch(f"{ENCRYPTION_MODULE}.CRYPTO_AVAILABLE", True):
                result = security_handler._get_status.__wrapped__(security_handler, mock_handler)

        body = json.loads(result.body.decode())
        assert body["rotation_recommended"] is True
        assert body["rotation_required"] is False

    def test_status_rotation_required(
        self, security_handler: SecurityHandler, mock_handler: MagicMock
    ):
        """Status requires rotation for very old keys (>90 days)."""
        old_key = MockEncryptionKey(created_at=datetime.now(timezone.utc) - timedelta(days=95))
        service = MockEncryptionService(active_key=old_key)

        with patch(f"{ENCRYPTION_MODULE}.get_encryption_service", return_value=service):
            with patch(f"{ENCRYPTION_MODULE}.CRYPTO_AVAILABLE", True):
                result = security_handler._get_status.__wrapped__(security_handler, mock_handler)

        body = json.loads(result.body.decode())
        assert body["rotation_required"] is True


class TestRotateKey:
    """Tests for _rotate_key endpoint."""

    def test_rotate_key_dry_run(self, security_handler: SecurityHandler, mock_handler: MagicMock):
        """Dry run rotation returns preview without changes."""
        service = MockEncryptionService()
        result_obj = MockRotationResult()

        with patch(f"{ENCRYPTION_MODULE}.get_encryption_service", return_value=service):
            with patch(f"{ENCRYPTION_MODULE}.CRYPTO_AVAILABLE", True):
                with patch(
                    f"{MIGRATION_MODULE}.rotate_encryption_key",
                    return_value=result_obj,
                ):
                    result = security_handler._rotate_key.__wrapped__(
                        security_handler, {"dry_run": True}, mock_handler
                    )

        assert result.status_code == 200
        body = json.loads(result.body.decode())
        assert body["dry_run"] is True
        assert body["success"] is True

    def test_rotate_key_actual(self, security_handler: SecurityHandler, mock_handler: MagicMock):
        """Actual rotation rotates key and re-encrypts data."""
        old_key = MockEncryptionKey(created_at=datetime.now(timezone.utc) - timedelta(days=45))
        service = MockEncryptionService(active_key=old_key)
        result_obj = MockRotationResult()

        with patch(f"{ENCRYPTION_MODULE}.get_encryption_service", return_value=service):
            with patch(f"{ENCRYPTION_MODULE}.CRYPTO_AVAILABLE", True):
                with patch(
                    f"{MIGRATION_MODULE}.rotate_encryption_key",
                    return_value=result_obj,
                ):
                    result = security_handler._rotate_key.__wrapped__(
                        security_handler, {"force": True}, mock_handler
                    )

        assert result.status_code == 200
        body = json.loads(result.body.decode())
        assert body["dry_run"] is False
        assert body["success"] is True
        assert body["new_key_version"] == 2
        assert body["records_reencrypted"] == 150

    def test_rotate_key_too_recent(
        self, security_handler: SecurityHandler, mock_handler: MagicMock
    ):
        """Rotation fails for keys <30 days old without force."""
        recent_key = MockEncryptionKey(created_at=datetime.now(timezone.utc) - timedelta(days=15))
        service = MockEncryptionService(active_key=recent_key)

        with patch(f"{ENCRYPTION_MODULE}.get_encryption_service", return_value=service):
            with patch(f"{ENCRYPTION_MODULE}.CRYPTO_AVAILABLE", True):
                result = security_handler._rotate_key.__wrapped__(
                    security_handler, {}, mock_handler
                )

        assert result.status_code == 400
        body = json.loads(result.body.decode())
        assert "15 days old" in body["error"]

    def test_rotate_key_crypto_unavailable(
        self, security_handler: SecurityHandler, mock_handler: MagicMock
    ):
        """Rotation fails when crypto unavailable."""
        with patch(f"{ENCRYPTION_MODULE}.CRYPTO_AVAILABLE", False):
            result = security_handler._rotate_key.__wrapped__(security_handler, {}, mock_handler)

        assert result.status_code == 400


class TestGetHealth:
    """Tests for _get_health endpoint."""

    def test_health_healthy(self, security_handler: SecurityHandler, mock_handler: MagicMock):
        """Health returns healthy when all checks pass."""
        service = MockEncryptionService()

        with patch(f"{ENCRYPTION_MODULE}.get_encryption_service", return_value=service):
            with patch(f"{ENCRYPTION_MODULE}.CRYPTO_AVAILABLE", True):
                result = security_handler._get_health.__wrapped__(security_handler, mock_handler)

        assert result.status_code == 200
        body = json.loads(result.body.decode())
        assert body["status"] == "healthy"
        assert body["checks"]["crypto_available"] is True
        assert body["checks"]["service_initialized"] is True
        assert body["checks"]["active_key"] is True
        assert body["checks"]["round_trip"] is True

    def test_health_crypto_unavailable(
        self, security_handler: SecurityHandler, mock_handler: MagicMock
    ):
        """Health returns unhealthy when crypto unavailable."""
        with patch(f"{ENCRYPTION_MODULE}.CRYPTO_AVAILABLE", False):
            result = security_handler._get_health.__wrapped__(security_handler, mock_handler)

        assert result.status_code == 200
        body = json.loads(result.body.decode())
        assert body["status"] == "unhealthy"
        assert "Cryptography library not installed" in body["issues"]

    def test_health_degraded_old_key(
        self, security_handler: SecurityHandler, mock_handler: MagicMock
    ):
        """Health returns degraded for old keys."""
        old_key = MockEncryptionKey(created_at=datetime.now(timezone.utc) - timedelta(days=95))
        service = MockEncryptionService(active_key=old_key)

        with patch(f"{ENCRYPTION_MODULE}.get_encryption_service", return_value=service):
            with patch(f"{ENCRYPTION_MODULE}.CRYPTO_AVAILABLE", True):
                result = security_handler._get_health.__wrapped__(security_handler, mock_handler)

        assert result.status_code == 200
        body = json.loads(result.body.decode())
        assert body["status"] == "degraded"
        assert any("95 days old" in w for w in body["warnings"])

    def test_health_no_active_key(self, security_handler: SecurityHandler, mock_handler: MagicMock):
        """Health returns unhealthy when no active key."""
        service = MockEncryptionService()
        service._active_key = None

        with patch(f"{ENCRYPTION_MODULE}.get_encryption_service", return_value=service):
            with patch(f"{ENCRYPTION_MODULE}.CRYPTO_AVAILABLE", True):
                result = security_handler._get_health.__wrapped__(security_handler, mock_handler)

        assert result.status_code == 200
        body = json.loads(result.body.decode())
        assert body["status"] == "unhealthy"
        assert "No active encryption key" in body["issues"]


class TestListKeys:
    """Tests for _list_keys endpoint."""

    def test_list_keys_single(self, security_handler: SecurityHandler, mock_handler: MagicMock):
        """Lists single active key."""
        service = MockEncryptionService()

        with patch(f"{ENCRYPTION_MODULE}.get_encryption_service", return_value=service):
            with patch(f"{ENCRYPTION_MODULE}.CRYPTO_AVAILABLE", True):
                result = security_handler._list_keys.__wrapped__(security_handler, mock_handler)

        assert result.status_code == 200
        body = json.loads(result.body.decode())
        assert body["total_keys"] == 1
        assert body["keys"][0]["is_active"] is True
        assert body["keys"][0]["key_id"] == "key-001"

    def test_list_keys_multiple(self, security_handler: SecurityHandler, mock_handler: MagicMock):
        """Lists multiple keys with active indicator."""
        old_key = MockEncryptionKey(
            key_id="key-000",
            version=0,
            created_at=datetime.now(timezone.utc) - timedelta(days=120),
        )
        active_key = MockEncryptionKey(
            key_id="key-001",
            version=1,
            created_at=datetime.now(timezone.utc) - timedelta(days=30),
        )
        service = MockEncryptionService(
            active_key=active_key,
            keys=[old_key, active_key],
        )

        with patch(f"{ENCRYPTION_MODULE}.get_encryption_service", return_value=service):
            with patch(f"{ENCRYPTION_MODULE}.CRYPTO_AVAILABLE", True):
                result = security_handler._list_keys.__wrapped__(security_handler, mock_handler)

        assert result.status_code == 200
        body = json.loads(result.body.decode())
        assert body["total_keys"] == 2
        assert body["active_key_id"] == "key-001"

        # Find active and inactive keys
        active = next(k for k in body["keys"] if k["is_active"])
        inactive = next(k for k in body["keys"] if not k["is_active"])

        assert active["key_id"] == "key-001"
        assert inactive["key_id"] == "key-000"

    def test_list_keys_crypto_unavailable(
        self, security_handler: SecurityHandler, mock_handler: MagicMock
    ):
        """List keys fails when crypto unavailable."""
        with patch(f"{ENCRYPTION_MODULE}.CRYPTO_AVAILABLE", False):
            result = security_handler._list_keys.__wrapped__(security_handler, mock_handler)

        assert result.status_code == 400


class TestRouting:
    """Tests for handle and handle_post routing."""

    def test_handle_routes_to_status(
        self, security_handler: SecurityHandler, mock_handler: MagicMock
    ):
        """GET request routes to status handler."""
        service = MockEncryptionService()

        with patch(f"{ENCRYPTION_MODULE}.get_encryption_service", return_value=service):
            with patch(f"{ENCRYPTION_MODULE}.CRYPTO_AVAILABLE", True):
                with patch.object(security_handler, "_get_status") as mock_status:
                    mock_status.return_value = MagicMock(status_code=200)
                    security_handler.handle("/api/v1/admin/security/status", {}, mock_handler)
                    mock_status.assert_called_once()

    def test_handle_post_routes_to_rotate(
        self, security_handler: SecurityHandler, mock_handler: MagicMock
    ):
        """POST request routes to rotate handler."""
        with patch.object(security_handler, "_rotate_key") as mock_rotate:
            mock_rotate.return_value = MagicMock(status_code=200)
            security_handler.handle_post(
                "/api/v1/admin/security/rotate-key", {"dry_run": True}, mock_handler
            )
            mock_rotate.assert_called_once()

    def test_handle_returns_none_for_unknown(
        self, security_handler: SecurityHandler, mock_handler: MagicMock
    ):
        """Unknown path returns None."""
        result = security_handler.handle("/api/v1/unknown", {}, mock_handler)
        assert result is None

    def test_handle_post_returns_none_for_unknown(
        self, security_handler: SecurityHandler, mock_handler: MagicMock
    ):
        """Unknown POST path returns None."""
        result = security_handler.handle_post("/api/v1/unknown", {}, mock_handler)
        assert result is None


class TestIntegration:
    """Integration tests for security handler flow."""

    def test_status_then_rotate_flow(
        self, security_handler: SecurityHandler, mock_handler: MagicMock
    ):
        """Complete flow: check status, rotate key, verify."""
        # Initial key is 45 days old
        initial_key = MockEncryptionKey(created_at=datetime.now(timezone.utc) - timedelta(days=45))
        service = MockEncryptionService(active_key=initial_key)

        with patch(f"{ENCRYPTION_MODULE}.get_encryption_service", return_value=service):
            with patch(f"{ENCRYPTION_MODULE}.CRYPTO_AVAILABLE", True):
                # Step 1: Check status
                status_result = security_handler._get_status.__wrapped__(
                    security_handler, mock_handler
                )
                status_body = json.loads(status_result.body.decode())
                assert status_body["rotation_recommended"] is False
                assert status_body["key_age_days"] == 45

                # Step 2: Dry run rotation
                with patch(
                    f"{MIGRATION_MODULE}.rotate_encryption_key",
                    return_value=MockRotationResult(),
                ):
                    dry_result = security_handler._rotate_key.__wrapped__(
                        security_handler, {"dry_run": True}, mock_handler
                    )
                    dry_body = json.loads(dry_result.body.decode())
                    assert dry_body["dry_run"] is True

                # Step 3: Actual rotation
                with patch(
                    f"{MIGRATION_MODULE}.rotate_encryption_key",
                    return_value=MockRotationResult(),
                ):
                    rotate_result = security_handler._rotate_key.__wrapped__(
                        security_handler, {"force": True}, mock_handler
                    )
                    rotate_body = json.loads(rotate_result.body.decode())
                    assert rotate_body["success"] is True
