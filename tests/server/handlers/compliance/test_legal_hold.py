"""
Tests for Legal Hold Management Handler.

Tests cover:
- Legal hold CRUD operations (list, create, release)
- User ID extraction from headers
- Audit logging for legal hold operations
- RBAC permission enforcement
- Error handling and edge cases
"""

from __future__ import annotations

import json
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.server.handlers.compliance.handler import ComplianceHandler
from aragora.server.handlers.compliance.legal_hold import _extract_user_id_from_headers
from aragora.server.handlers.base import HandlerResult


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def compliance_handler():
    """Create a compliance handler instance."""
    return ComplianceHandler(server_context={})


@pytest.fixture
def mock_legal_hold_manager():
    """Create a mock legal hold manager."""
    manager = MagicMock()
    manager.is_user_on_hold = MagicMock(return_value=False)
    manager.get_active_holds = MagicMock(return_value=[])
    manager.create_hold = MagicMock()
    manager.release_hold = MagicMock()
    manager._store = MagicMock()
    manager._store._holds = {}
    return manager


@pytest.fixture
def mock_audit_store():
    """Create a mock audit store."""
    store = MagicMock()
    store.log_event = MagicMock()
    return store


# ============================================================================
# User ID Extraction Tests
# ============================================================================


class TestUserIdExtraction:
    """Tests for user ID extraction from headers."""

    def test_no_headers_returns_default(self):
        """No headers returns compliance_api."""
        result = _extract_user_id_from_headers(None)
        assert result == "compliance_api"

    def test_empty_headers_returns_default(self):
        """Empty headers returns compliance_api."""
        result = _extract_user_id_from_headers({})
        assert result == "compliance_api"

    def test_no_auth_header_returns_default(self):
        """Missing Authorization header returns compliance_api."""
        result = _extract_user_id_from_headers({"Content-Type": "application/json"})
        assert result == "compliance_api"

    def test_invalid_auth_format_returns_default(self):
        """Non-Bearer token returns compliance_api."""
        result = _extract_user_id_from_headers({"Authorization": "Basic dXNlcjpwYXNz"})
        assert result == "compliance_api"

    def test_api_key_extracts_prefix(self):
        """API key extracts truncated prefix."""
        result = _extract_user_id_from_headers({"Authorization": "Bearer ara_test123456789"})
        assert result.startswith("api_key:")
        assert "..." in result

    def test_lowercase_authorization_header(self):
        """Handles lowercase authorization header."""
        result = _extract_user_id_from_headers({"authorization": "Bearer ara_test123456789"})
        assert result.startswith("api_key:")

    def test_jwt_extraction_with_valid_token(self):
        """JWT token extracts user_id."""
        mock_payload = MagicMock()
        mock_payload.user_id = "user-123"

        with patch(
            "aragora.billing.auth.tokens.validate_access_token",
            return_value=mock_payload,
        ):
            result = _extract_user_id_from_headers({"Authorization": "Bearer valid.jwt.token"})

        assert result == "user-123"

    def test_jwt_extraction_failure_returns_default(self):
        """Failed JWT validation returns compliance_api."""
        with patch(
            "aragora.billing.auth.tokens.validate_access_token",
            side_effect=ValueError("Invalid token"),
        ):
            result = _extract_user_id_from_headers({"Authorization": "Bearer invalid.jwt"})

        assert result == "compliance_api"

    def test_jwt_import_error_returns_default(self):
        """Missing JWT module returns compliance_api."""
        with patch(
            "aragora.billing.auth.tokens.validate_access_token",
            side_effect=ImportError("Module not found"),
        ):
            result = _extract_user_id_from_headers({"Authorization": "Bearer some.token"})

        assert result == "compliance_api"


# ============================================================================
# List Legal Holds Tests
# ============================================================================


class TestListLegalHolds:
    """Tests for listing legal holds."""

    @pytest.mark.asyncio
    async def test_list_holds_active_only_default(
        self, compliance_handler, mock_legal_hold_manager
    ):
        """List holds defaults to active only."""
        mock_hold = MagicMock()
        mock_hold.to_dict.return_value = {
            "hold_id": "hold-001",
            "user_ids": ["user-123"],
            "reason": "Investigation",
            "is_active": True,
        }
        mock_legal_hold_manager.get_active_holds.return_value = [mock_hold]

        with patch(
            "aragora.server.handlers.compliance_handler.get_legal_hold_manager",
            return_value=mock_legal_hold_manager,
        ):
            result = await compliance_handler._list_legal_holds({})

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["count"] == 1
        assert body["filters"]["active_only"] is True
        mock_legal_hold_manager.get_active_holds.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_holds_all(self, compliance_handler, mock_legal_hold_manager):
        """List holds returns all holds when active_only=false."""
        mock_hold = MagicMock()
        mock_hold.to_dict.return_value = {"hold_id": "hold-001"}
        mock_legal_hold_manager._store._holds = {"hold-001": mock_hold}

        with patch(
            "aragora.server.handlers.compliance_handler.get_legal_hold_manager",
            return_value=mock_legal_hold_manager,
        ):
            result = await compliance_handler._list_legal_holds({"active_only": "false"})

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["filters"]["active_only"] is False

    @pytest.mark.asyncio
    async def test_list_holds_empty(self, compliance_handler, mock_legal_hold_manager):
        """List holds returns empty list when no holds exist."""
        mock_legal_hold_manager.get_active_holds.return_value = []

        with patch(
            "aragora.server.handlers.compliance_handler.get_legal_hold_manager",
            return_value=mock_legal_hold_manager,
        ):
            result = await compliance_handler._list_legal_holds({})

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["count"] == 0
        assert body["legal_holds"] == []

    @pytest.mark.asyncio
    async def test_list_holds_error_handling(self, compliance_handler):
        """List holds handles manager errors."""
        with patch(
            "aragora.server.handlers.compliance_handler.get_legal_hold_manager",
            side_effect=RuntimeError("Database error"),
        ):
            result = await compliance_handler._list_legal_holds({})

        assert result.status_code == 500
        body = json.loads(result.body)
        assert "error" in body


# ============================================================================
# Create Legal Hold Tests
# ============================================================================


class TestCreateLegalHold:
    """Tests for creating legal holds."""

    @pytest.mark.asyncio
    async def test_create_hold_requires_user_ids(self, compliance_handler):
        """Create hold fails without user_ids."""
        result = await compliance_handler._create_legal_hold(
            {"reason": "Investigation"},
            headers=None,
        )

        assert result.status_code == 400
        body = json.loads(result.body)
        assert "user_ids is required" in body.get("error", "")

    @pytest.mark.asyncio
    async def test_create_hold_requires_reason(self, compliance_handler):
        """Create hold fails without reason."""
        result = await compliance_handler._create_legal_hold(
            {"user_ids": ["user-123"]},
            headers=None,
        )

        assert result.status_code == 400
        body = json.loads(result.body)
        assert "reason is required" in body.get("error", "")

    @pytest.mark.asyncio
    async def test_create_hold_success(
        self, compliance_handler, mock_legal_hold_manager, mock_audit_store
    ):
        """Create hold successfully creates a legal hold."""
        mock_hold = MagicMock()
        mock_hold.hold_id = "hold-001"
        mock_hold.to_dict.return_value = {
            "hold_id": "hold-001",
            "user_ids": ["user-123"],
            "reason": "Legal investigation",
            "created_by": "compliance_api",
        }
        mock_legal_hold_manager.create_hold.return_value = mock_hold

        with (
            patch(
                "aragora.server.handlers.compliance_handler.get_legal_hold_manager",
                return_value=mock_legal_hold_manager,
            ),
            patch(
                "aragora.server.handlers.compliance_handler.get_audit_store",
                return_value=mock_audit_store,
            ),
        ):
            result = await compliance_handler._create_legal_hold(
                {"user_ids": ["user-123"], "reason": "Legal investigation"},
                headers=None,
            )

        assert result.status_code == 201
        body = json.loads(result.body)
        assert body["message"] == "Legal hold created successfully"
        assert body["legal_hold"]["hold_id"] == "hold-001"

    @pytest.mark.asyncio
    async def test_create_hold_with_case_reference(
        self, compliance_handler, mock_legal_hold_manager, mock_audit_store
    ):
        """Create hold includes case reference."""
        mock_hold = MagicMock()
        mock_hold.hold_id = "hold-001"
        mock_hold.to_dict.return_value = {
            "hold_id": "hold-001",
            "case_reference": "CASE-2025-001",
        }
        mock_legal_hold_manager.create_hold.return_value = mock_hold

        with (
            patch(
                "aragora.server.handlers.compliance_handler.get_legal_hold_manager",
                return_value=mock_legal_hold_manager,
            ),
            patch(
                "aragora.server.handlers.compliance_handler.get_audit_store",
                return_value=mock_audit_store,
            ),
        ):
            result = await compliance_handler._create_legal_hold(
                {
                    "user_ids": ["user-123"],
                    "reason": "Investigation",
                    "case_reference": "CASE-2025-001",
                },
                headers=None,
            )

        assert result.status_code == 201
        mock_legal_hold_manager.create_hold.assert_called_once()
        call_kwargs = mock_legal_hold_manager.create_hold.call_args.kwargs
        assert call_kwargs["case_reference"] == "CASE-2025-001"

    @pytest.mark.asyncio
    async def test_create_hold_with_expiration(
        self, compliance_handler, mock_legal_hold_manager, mock_audit_store
    ):
        """Create hold includes expiration date."""
        mock_hold = MagicMock()
        mock_hold.hold_id = "hold-001"
        mock_hold.to_dict.return_value = {"hold_id": "hold-001"}
        mock_legal_hold_manager.create_hold.return_value = mock_hold

        with (
            patch(
                "aragora.server.handlers.compliance_handler.get_legal_hold_manager",
                return_value=mock_legal_hold_manager,
            ),
            patch(
                "aragora.server.handlers.compliance_handler.get_audit_store",
                return_value=mock_audit_store,
            ),
        ):
            result = await compliance_handler._create_legal_hold(
                {
                    "user_ids": ["user-123"],
                    "reason": "Investigation",
                    "expires_at": "2025-12-31T00:00:00Z",
                },
                headers=None,
            )

        assert result.status_code == 201
        call_kwargs = mock_legal_hold_manager.create_hold.call_args.kwargs
        assert call_kwargs["expires_at"] is not None

    @pytest.mark.asyncio
    async def test_create_hold_invalid_expiration_format(self, compliance_handler):
        """Create hold fails with invalid expiration format."""
        result = await compliance_handler._create_legal_hold(
            {
                "user_ids": ["user-123"],
                "reason": "Investigation",
                "expires_at": "not-a-date",
            },
            headers=None,
        )

        assert result.status_code == 400
        body = json.loads(result.body)
        assert "invalid" in body.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_create_hold_extracts_user_from_headers(
        self, compliance_handler, mock_legal_hold_manager, mock_audit_store
    ):
        """Create hold extracts user from auth headers."""
        mock_hold = MagicMock()
        mock_hold.hold_id = "hold-001"
        mock_hold.to_dict.return_value = {"hold_id": "hold-001"}
        mock_legal_hold_manager.create_hold.return_value = mock_hold

        with (
            patch(
                "aragora.server.handlers.compliance_handler.get_legal_hold_manager",
                return_value=mock_legal_hold_manager,
            ),
            patch(
                "aragora.server.handlers.compliance_handler.get_audit_store",
                return_value=mock_audit_store,
            ),
        ):
            result = await compliance_handler._create_legal_hold(
                {"user_ids": ["user-123"], "reason": "Investigation"},
                headers={"Authorization": "Bearer ara_admin123456789"},
            )

        assert result.status_code == 201
        call_kwargs = mock_legal_hold_manager.create_hold.call_args.kwargs
        assert "api_key:" in call_kwargs["created_by"]

    @pytest.mark.asyncio
    async def test_create_hold_logs_audit_event(
        self, compliance_handler, mock_legal_hold_manager, mock_audit_store
    ):
        """Create hold logs audit event."""
        mock_hold = MagicMock()
        mock_hold.hold_id = "hold-001"
        mock_hold.to_dict.return_value = {"hold_id": "hold-001"}
        mock_legal_hold_manager.create_hold.return_value = mock_hold

        with (
            patch(
                "aragora.server.handlers.compliance_handler.get_legal_hold_manager",
                return_value=mock_legal_hold_manager,
            ),
            patch(
                "aragora.server.handlers.compliance_handler.get_audit_store",
                return_value=mock_audit_store,
            ),
        ):
            await compliance_handler._create_legal_hold(
                {"user_ids": ["user-123"], "reason": "Investigation"},
                headers=None,
            )

        mock_audit_store.log_event.assert_called_once()
        call_kwargs = mock_audit_store.log_event.call_args.kwargs
        assert call_kwargs["action"] == "legal_hold_created"
        assert call_kwargs["resource_type"] == "legal_hold"
        assert call_kwargs["resource_id"] == "hold-001"

    @pytest.mark.asyncio
    async def test_create_hold_error_handling(self, compliance_handler, mock_legal_hold_manager):
        """Create hold handles manager errors."""
        mock_legal_hold_manager.create_hold.side_effect = RuntimeError("Database error")

        with patch(
            "aragora.server.handlers.compliance_handler.get_legal_hold_manager",
            return_value=mock_legal_hold_manager,
        ):
            result = await compliance_handler._create_legal_hold(
                {"user_ids": ["user-123"], "reason": "Investigation"},
                headers=None,
            )

        assert result.status_code == 500


# ============================================================================
# Release Legal Hold Tests
# ============================================================================


class TestReleaseLegalHold:
    """Tests for releasing legal holds."""

    @pytest.mark.asyncio
    async def test_release_hold_success(
        self, compliance_handler, mock_legal_hold_manager, mock_audit_store
    ):
        """Release hold successfully releases a legal hold."""
        mock_released = MagicMock()
        mock_released.released_at = datetime.now(timezone.utc)
        mock_released.user_ids = ["user-123"]
        mock_released.to_dict.return_value = {
            "hold_id": "hold-001",
            "is_active": False,
            "released_at": mock_released.released_at.isoformat(),
        }
        mock_legal_hold_manager.release_hold.return_value = mock_released

        with (
            patch(
                "aragora.server.handlers.compliance_handler.get_legal_hold_manager",
                return_value=mock_legal_hold_manager,
            ),
            patch(
                "aragora.server.handlers.compliance_handler.get_audit_store",
                return_value=mock_audit_store,
            ),
        ):
            result = await compliance_handler._release_legal_hold(
                "hold-001",
                {"released_by": "admin@example.com"},
            )

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["message"] == "Legal hold released successfully"

    @pytest.mark.asyncio
    async def test_release_hold_not_found(self, compliance_handler, mock_legal_hold_manager):
        """Release hold returns 404 when hold not found."""
        mock_legal_hold_manager.release_hold.return_value = None

        with patch(
            "aragora.server.handlers.compliance_handler.get_legal_hold_manager",
            return_value=mock_legal_hold_manager,
        ):
            result = await compliance_handler._release_legal_hold("nonexistent", {})

        assert result.status_code == 404
        body = json.loads(result.body)
        assert "not found" in body.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_release_hold_default_released_by(
        self, compliance_handler, mock_legal_hold_manager, mock_audit_store
    ):
        """Release hold uses default released_by."""
        mock_released = MagicMock()
        mock_released.released_at = datetime.now(timezone.utc)
        mock_released.user_ids = ["user-123"]
        mock_released.to_dict.return_value = {"hold_id": "hold-001"}
        mock_legal_hold_manager.release_hold.return_value = mock_released

        with (
            patch(
                "aragora.server.handlers.compliance_handler.get_legal_hold_manager",
                return_value=mock_legal_hold_manager,
            ),
            patch(
                "aragora.server.handlers.compliance_handler.get_audit_store",
                return_value=mock_audit_store,
            ),
        ):
            await compliance_handler._release_legal_hold("hold-001", {})

        mock_legal_hold_manager.release_hold.assert_called_once_with("hold-001", "compliance_api")

    @pytest.mark.asyncio
    async def test_release_hold_logs_audit_event(
        self, compliance_handler, mock_legal_hold_manager, mock_audit_store
    ):
        """Release hold logs audit event."""
        mock_released = MagicMock()
        mock_released.released_at = datetime.now(timezone.utc)
        mock_released.user_ids = ["user-123", "user-456"]
        mock_released.to_dict.return_value = {"hold_id": "hold-001"}
        mock_legal_hold_manager.release_hold.return_value = mock_released

        with (
            patch(
                "aragora.server.handlers.compliance_handler.get_legal_hold_manager",
                return_value=mock_legal_hold_manager,
            ),
            patch(
                "aragora.server.handlers.compliance_handler.get_audit_store",
                return_value=mock_audit_store,
            ),
        ):
            await compliance_handler._release_legal_hold(
                "hold-001",
                {"released_by": "admin@example.com"},
            )

        mock_audit_store.log_event.assert_called_once()
        call_kwargs = mock_audit_store.log_event.call_args.kwargs
        assert call_kwargs["action"] == "legal_hold_released"
        assert call_kwargs["resource_id"] == "hold-001"
        assert call_kwargs["metadata"]["released_by"] == "admin@example.com"
        assert call_kwargs["metadata"]["user_ids"] == ["user-123", "user-456"]

    @pytest.mark.asyncio
    async def test_release_hold_error_handling(self, compliance_handler, mock_legal_hold_manager):
        """Release hold handles manager errors."""
        mock_legal_hold_manager.release_hold.side_effect = RuntimeError("Database error")

        with patch(
            "aragora.server.handlers.compliance_handler.get_legal_hold_manager",
            return_value=mock_legal_hold_manager,
        ):
            result = await compliance_handler._release_legal_hold("hold-001", {})

        assert result.status_code == 500


# ============================================================================
# RBAC Permission Tests
# ============================================================================


class TestLegalHoldPermissions:
    """Tests for legal hold RBAC permission enforcement."""

    def test_list_holds_has_permission_decorator(self):
        """List holds requires compliance:legal permission."""
        import inspect

        source = inspect.getsource(ComplianceHandler._list_legal_holds)
        assert "require_permission" in source
        assert "compliance:legal" in source

    def test_create_hold_has_permission_decorator(self):
        """Create hold requires compliance:legal permission."""
        import inspect

        source = inspect.getsource(ComplianceHandler._create_legal_hold)
        assert "require_permission" in source
        assert "compliance:legal" in source

    def test_release_hold_has_permission_decorator(self):
        """Release hold requires compliance:legal permission."""
        import inspect

        source = inspect.getsource(ComplianceHandler._release_legal_hold)
        assert "require_permission" in source
        assert "compliance:legal" in source


# ============================================================================
# Audit Logging Edge Cases
# ============================================================================


class TestLegalHoldAuditLogging:
    """Tests for audit logging edge cases."""

    @pytest.mark.asyncio
    async def test_create_hold_continues_on_audit_failure(
        self, compliance_handler, mock_legal_hold_manager
    ):
        """Create hold succeeds even if audit logging fails."""
        mock_hold = MagicMock()
        mock_hold.hold_id = "hold-001"
        mock_hold.to_dict.return_value = {"hold_id": "hold-001"}
        mock_legal_hold_manager.create_hold.return_value = mock_hold

        mock_audit_store = MagicMock()
        mock_audit_store.log_event.side_effect = RuntimeError("Audit store unavailable")

        with (
            patch(
                "aragora.server.handlers.compliance_handler.get_legal_hold_manager",
                return_value=mock_legal_hold_manager,
            ),
            patch(
                "aragora.server.handlers.compliance_handler.get_audit_store",
                return_value=mock_audit_store,
            ),
        ):
            result = await compliance_handler._create_legal_hold(
                {"user_ids": ["user-123"], "reason": "Investigation"},
                headers=None,
            )

        # Should still succeed
        assert result.status_code == 201

    @pytest.mark.asyncio
    async def test_release_hold_continues_on_audit_failure(
        self, compliance_handler, mock_legal_hold_manager
    ):
        """Release hold succeeds even if audit logging fails."""
        mock_released = MagicMock()
        mock_released.released_at = datetime.now(timezone.utc)
        mock_released.user_ids = ["user-123"]
        mock_released.to_dict.return_value = {"hold_id": "hold-001"}
        mock_legal_hold_manager.release_hold.return_value = mock_released

        mock_audit_store = MagicMock()
        mock_audit_store.log_event.side_effect = RuntimeError("Audit store unavailable")

        with (
            patch(
                "aragora.server.handlers.compliance_handler.get_legal_hold_manager",
                return_value=mock_legal_hold_manager,
            ),
            patch(
                "aragora.server.handlers.compliance_handler.get_audit_store",
                return_value=mock_audit_store,
            ),
        ):
            result = await compliance_handler._release_legal_hold("hold-001", {})

        # Should still succeed
        assert result.status_code == 200


# ============================================================================
# Edge Cases Tests
# ============================================================================


class TestLegalHoldEdgeCases:
    """Tests for edge cases in legal hold operations."""

    @pytest.mark.asyncio
    async def test_create_hold_multiple_users(
        self, compliance_handler, mock_legal_hold_manager, mock_audit_store
    ):
        """Create hold with multiple users."""
        mock_hold = MagicMock()
        mock_hold.hold_id = "hold-001"
        mock_hold.to_dict.return_value = {
            "hold_id": "hold-001",
            "user_ids": ["user-1", "user-2", "user-3"],
        }
        mock_legal_hold_manager.create_hold.return_value = mock_hold

        with (
            patch(
                "aragora.server.handlers.compliance_handler.get_legal_hold_manager",
                return_value=mock_legal_hold_manager,
            ),
            patch(
                "aragora.server.handlers.compliance_handler.get_audit_store",
                return_value=mock_audit_store,
            ),
        ):
            result = await compliance_handler._create_legal_hold(
                {"user_ids": ["user-1", "user-2", "user-3"], "reason": "Multi-user hold"},
                headers=None,
            )

        assert result.status_code == 201
        call_kwargs = mock_legal_hold_manager.create_hold.call_args.kwargs
        assert len(call_kwargs["user_ids"]) == 3

    @pytest.mark.asyncio
    async def test_create_hold_empty_user_ids_list(self, compliance_handler):
        """Create hold fails with empty user_ids list."""
        result = await compliance_handler._create_legal_hold(
            {"user_ids": [], "reason": "Investigation"},
            headers=None,
        )

        assert result.status_code == 400
        body = json.loads(result.body)
        assert "user_ids is required" in body.get("error", "")

    @pytest.mark.asyncio
    async def test_release_hold_with_null_released_at(
        self, compliance_handler, mock_legal_hold_manager, mock_audit_store
    ):
        """Release hold handles null released_at in response."""
        mock_released = MagicMock()
        mock_released.released_at = None  # Shouldn't happen, but test edge case
        mock_released.user_ids = ["user-123"]
        mock_released.to_dict.return_value = {"hold_id": "hold-001"}
        mock_legal_hold_manager.release_hold.return_value = mock_released

        with (
            patch(
                "aragora.server.handlers.compliance_handler.get_legal_hold_manager",
                return_value=mock_legal_hold_manager,
            ),
            patch(
                "aragora.server.handlers.compliance_handler.get_audit_store",
                return_value=mock_audit_store,
            ),
        ):
            result = await compliance_handler._release_legal_hold("hold-001", {})

        assert result.status_code == 200
        call_kwargs = mock_audit_store.log_event.call_args.kwargs
        assert call_kwargs["metadata"]["released_at"] is None


# ============================================================================
# Handler Tracking Tests
# ============================================================================


class TestLegalHoldTracking:
    """Tests for handler metrics tracking."""

    def test_create_hold_has_track_handler_decorator(self):
        """Create hold has metrics tracking."""
        import inspect

        source = inspect.getsource(ComplianceHandler._create_legal_hold)
        assert "track_handler" in source
        assert "compliance/legal-hold-create" in source
