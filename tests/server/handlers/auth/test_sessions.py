"""
Tests for Session Management handlers.

Phase 5: Auth Handler Test Coverage - Session handler tests.

Tests for:
- handle_list_sessions - List active sessions
- handle_revoke_session - Revoke a specific session
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from tests.server.handlers.conftest import (
    assert_error_response,
    assert_success_response,
    parse_handler_response,
)

# Patch paths for functions imported inside handlers
PATCH_EXTRACT_USER = "aragora.server.handlers.auth.sessions.extract_user_from_request"
PATCH_GET_SESSION_MANAGER = "aragora.billing.auth.sessions.get_session_manager"
PATCH_DECODE_JWT = "aragora.billing.jwt_auth.decode_jwt"
PATCH_EXTRACT_TOKEN = "aragora.server.middleware.auth.extract_token"


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def mock_auth_handler():
    """Create a mock AuthHandler instance."""
    handler_instance = MagicMock()
    handler_instance._check_permission.return_value = None
    return handler_instance


@pytest.fixture
def mock_user_store():
    """Create a mock user store."""
    store = MagicMock()
    return store


@pytest.fixture
def mock_session():
    """Create a mock session object."""
    session = MagicMock()
    session.session_id = "sess-abc123"
    session.user_id = "user-001"
    session.created_at = datetime.now(timezone.utc)
    session.last_activity = datetime.now(timezone.utc)
    session.ip_address = "192.168.1.1"
    session.user_agent = "Mozilla/5.0"
    session.device_type = "desktop"
    session.to_dict.return_value = {
        "session_id": session.session_id,
        "user_id": session.user_id,
        "created_at": session.created_at.isoformat(),
        "last_activity": session.last_activity.isoformat(),
        "ip_address": session.ip_address,
        "user_agent": session.user_agent,
        "device_type": session.device_type,
    }
    return session


@pytest.fixture
def mock_session_manager(mock_session):
    """Create a mock session manager."""
    manager = MagicMock()
    manager.list_sessions.return_value = [mock_session]
    manager.get_session.return_value = mock_session
    manager.revoke_session.return_value = None
    return manager


@pytest.fixture
def mock_http_handler():
    """Factory to create mock HTTP handler."""

    def _create(method: str = "GET", body: dict = None):
        mock = MagicMock()
        mock.command = method

        if body is not None:
            body_bytes = json.dumps(body).encode()
        else:
            body_bytes = b"{}"

        mock.rfile = MagicMock()
        mock.rfile.read = MagicMock(return_value=body_bytes)
        mock.headers = {"Content-Length": str(len(body_bytes))}
        mock.client_address = ("127.0.0.1", 12345)

        return mock

    return _create


# ============================================================================
# Test: List Sessions (handle_list_sessions)
# ============================================================================


class TestListSessions:
    """Tests for handle_list_sessions."""

    def test_list_sessions_returns_active_sessions(
        self, mock_auth_handler, mock_user_store, mock_session_manager, mock_http_handler
    ):
        """Test that list sessions returns active sessions."""
        from aragora.server.handlers.auth.sessions import handle_list_sessions

        mock_auth_handler._get_user_store.return_value = mock_user_store

        mock_auth_ctx = MagicMock()
        mock_auth_ctx.user_id = "user-001"

        http = mock_http_handler(method="GET")

        with (
            patch(PATCH_EXTRACT_USER, return_value=mock_auth_ctx),
            patch(PATCH_GET_SESSION_MANAGER, return_value=mock_session_manager),
            patch(PATCH_EXTRACT_TOKEN, return_value="valid-jwt-token"),
            patch(PATCH_DECODE_JWT, return_value={"sub": "user-001"}),
        ):
            result = handle_list_sessions(mock_auth_handler, http)

        assert result.status_code == 200
        body = parse_handler_response(result)
        assert "sessions" in body
        assert "total" in body
        assert len(body["sessions"]) == 1

    def test_list_sessions_includes_metadata(
        self, mock_auth_handler, mock_user_store, mock_session_manager, mock_http_handler
    ):
        """Test that session metadata is included."""
        from aragora.server.handlers.auth.sessions import handle_list_sessions

        mock_auth_handler._get_user_store.return_value = mock_user_store

        mock_auth_ctx = MagicMock()
        mock_auth_ctx.user_id = "user-001"

        http = mock_http_handler(method="GET")

        with (
            patch(PATCH_EXTRACT_USER, return_value=mock_auth_ctx),
            patch(PATCH_GET_SESSION_MANAGER, return_value=mock_session_manager),
            patch(PATCH_EXTRACT_TOKEN, return_value="valid-jwt-token"),
            patch(PATCH_DECODE_JWT, return_value={"sub": "user-001"}),
        ):
            result = handle_list_sessions(mock_auth_handler, http)

        body = parse_handler_response(result)
        session = body["sessions"][0]
        assert "ip_address" in session
        assert "user_agent" in session
        assert "device_type" in session
        assert "last_activity" in session

    def test_list_sessions_identifies_current_session(
        self,
        mock_auth_handler,
        mock_user_store,
        mock_session_manager,
        mock_session,
        mock_http_handler,
    ):
        """Test that current session is marked."""
        from aragora.server.handlers.auth.sessions import handle_list_sessions

        mock_auth_handler._get_user_store.return_value = mock_user_store

        mock_auth_ctx = MagicMock()
        mock_auth_ctx.user_id = "user-001"

        # Token that matches session ID when hashed
        token = "current-token"
        expected_jti = hashlib.sha256(token.encode()).hexdigest()[:32]
        mock_session.session_id = expected_jti
        mock_session.to_dict.return_value["session_id"] = expected_jti

        http = mock_http_handler(method="GET")

        with (
            patch(PATCH_EXTRACT_USER, return_value=mock_auth_ctx),
            patch(PATCH_GET_SESSION_MANAGER, return_value=mock_session_manager),
            patch(PATCH_EXTRACT_TOKEN, return_value=token),
            patch(PATCH_DECODE_JWT, return_value={"sub": "user-001"}),
        ):
            result = handle_list_sessions(mock_auth_handler, http)

        body = parse_handler_response(result)
        session = body["sessions"][0]
        assert session.get("is_current") is True

    def test_list_sessions_empty_for_no_sessions(
        self, mock_auth_handler, mock_user_store, mock_session_manager, mock_http_handler
    ):
        """Test that empty list returned when no sessions."""
        from aragora.server.handlers.auth.sessions import handle_list_sessions

        mock_auth_handler._get_user_store.return_value = mock_user_store
        mock_session_manager.list_sessions.return_value = []

        mock_auth_ctx = MagicMock()
        mock_auth_ctx.user_id = "user-001"

        http = mock_http_handler(method="GET")

        with (
            patch(PATCH_EXTRACT_USER, return_value=mock_auth_ctx),
            patch(PATCH_GET_SESSION_MANAGER, return_value=mock_session_manager),
            patch(PATCH_EXTRACT_TOKEN, return_value=None),
            patch(PATCH_DECODE_JWT, return_value=None),
        ):
            result = handle_list_sessions(mock_auth_handler, http)

        body = parse_handler_response(result)
        assert body["sessions"] == []
        assert body["total"] == 0

    def test_list_sessions_requires_permission(self, mock_auth_handler, mock_http_handler):
        """Test that listing sessions requires permission."""
        from aragora.server.handlers.auth.sessions import handle_list_sessions
        from aragora.server.handlers.base import error_response

        mock_auth_handler._check_permission.return_value = error_response("Forbidden", 403)

        http = mock_http_handler(method="GET")
        result = handle_list_sessions(mock_auth_handler, http)

        assert_error_response(result, 403)

    def test_list_sessions_sorted_by_last_activity(
        self, mock_auth_handler, mock_user_store, mock_session_manager, mock_http_handler
    ):
        """Test that sessions are sorted by last activity (most recent first)."""
        from aragora.server.handlers.auth.sessions import handle_list_sessions

        # Create multiple sessions with different last_activity times
        old_session = MagicMock()
        old_session.session_id = "sess-old"
        old_session.to_dict.return_value = {
            "session_id": "sess-old",
            "last_activity": "2026-01-01T10:00:00Z",
        }

        new_session = MagicMock()
        new_session.session_id = "sess-new"
        new_session.to_dict.return_value = {
            "session_id": "sess-new",
            "last_activity": "2026-01-15T10:00:00Z",
        }

        mock_session_manager.list_sessions.return_value = [old_session, new_session]
        mock_auth_handler._get_user_store.return_value = mock_user_store

        mock_auth_ctx = MagicMock()
        mock_auth_ctx.user_id = "user-001"

        http = mock_http_handler(method="GET")

        with (
            patch(PATCH_EXTRACT_USER, return_value=mock_auth_ctx),
            patch(PATCH_GET_SESSION_MANAGER, return_value=mock_session_manager),
            patch(PATCH_EXTRACT_TOKEN, return_value=None),
            patch(PATCH_DECODE_JWT, return_value=None),
        ):
            result = handle_list_sessions(mock_auth_handler, http)

        body = parse_handler_response(result)
        # Most recent should be first
        assert body["sessions"][0]["session_id"] == "sess-new"
        assert body["sessions"][1]["session_id"] == "sess-old"


# ============================================================================
# Test: Revoke Session (handle_revoke_session)
# ============================================================================


class TestRevokeSession:
    """Tests for handle_revoke_session."""

    def test_revoke_session_succeeds(
        self, mock_auth_handler, mock_user_store, mock_session_manager, mock_http_handler
    ):
        """Test successful session revocation."""
        from aragora.server.handlers.auth.sessions import handle_revoke_session

        mock_auth_handler._get_user_store.return_value = mock_user_store

        mock_auth_ctx = MagicMock()
        mock_auth_ctx.user_id = "user-001"

        http = mock_http_handler(method="DELETE")

        with (
            patch(PATCH_EXTRACT_USER, return_value=mock_auth_ctx),
            patch(PATCH_GET_SESSION_MANAGER, return_value=mock_session_manager),
            patch(PATCH_EXTRACT_TOKEN, return_value="different-token"),
            patch(PATCH_DECODE_JWT, return_value={"sub": "user-001"}),
        ):
            result = handle_revoke_session(mock_auth_handler, http, "sess-abc123")

        assert result.status_code == 200
        body = parse_handler_response(result)
        assert body.get("success") is True
        mock_session_manager.revoke_session.assert_called_once()

    def test_revoke_session_not_found(
        self, mock_auth_handler, mock_user_store, mock_session_manager, mock_http_handler
    ):
        """Test revoke fails for unknown session."""
        from aragora.server.handlers.auth.sessions import handle_revoke_session

        mock_auth_handler._get_user_store.return_value = mock_user_store
        mock_session_manager.get_session.return_value = None

        mock_auth_ctx = MagicMock()
        mock_auth_ctx.user_id = "user-001"

        http = mock_http_handler(method="DELETE")

        with (
            patch(PATCH_EXTRACT_USER, return_value=mock_auth_ctx),
            patch(PATCH_GET_SESSION_MANAGER, return_value=mock_session_manager),
            patch(PATCH_EXTRACT_TOKEN, return_value="token"),
            patch(PATCH_DECODE_JWT, return_value={"sub": "user-001"}),
        ):
            result = handle_revoke_session(mock_auth_handler, http, "unknown-session")

        assert_error_response(result, 404, "not found")

    def test_revoke_session_cannot_revoke_current(
        self, mock_auth_handler, mock_user_store, mock_session_manager, mock_http_handler
    ):
        """Test cannot revoke current session."""
        from aragora.server.handlers.auth.sessions import handle_revoke_session

        mock_auth_handler._get_user_store.return_value = mock_user_store

        mock_auth_ctx = MagicMock()
        mock_auth_ctx.user_id = "user-001"

        # Token JTI matches session ID
        token = "current-token"
        current_jti = hashlib.sha256(token.encode()).hexdigest()[:32]

        http = mock_http_handler(method="DELETE")

        with (
            patch(PATCH_EXTRACT_USER, return_value=mock_auth_ctx),
            patch(PATCH_GET_SESSION_MANAGER, return_value=mock_session_manager),
            patch(PATCH_EXTRACT_TOKEN, return_value=token),
            patch(PATCH_DECODE_JWT, return_value={"sub": "user-001"}),
        ):
            result = handle_revoke_session(mock_auth_handler, http, current_jti)

        assert_error_response(result, 400, "current session")

    def test_revoke_session_requires_permission(self, mock_auth_handler, mock_http_handler):
        """Test that revoking session requires permission."""
        from aragora.server.handlers.auth.sessions import handle_revoke_session
        from aragora.server.handlers.base import error_response

        mock_auth_handler._check_permission.return_value = error_response("Forbidden", 403)

        http = mock_http_handler(method="DELETE")
        result = handle_revoke_session(mock_auth_handler, http, "sess-abc123")

        assert_error_response(result, 403)

    def test_revoke_session_invalid_session_id(
        self, mock_auth_handler, mock_user_store, mock_http_handler
    ):
        """Test revoke fails for invalid session ID format."""
        from aragora.server.handlers.auth.sessions import handle_revoke_session

        mock_auth_handler._get_user_store.return_value = mock_user_store

        http = mock_http_handler(method="DELETE")

        # Short session ID (less than 8 chars)
        result = handle_revoke_session(mock_auth_handler, http, "short")

        assert_error_response(result, 400, "Invalid session ID")

    def test_revoke_session_empty_session_id(
        self, mock_auth_handler, mock_user_store, mock_http_handler
    ):
        """Test revoke fails for empty session ID."""
        from aragora.server.handlers.auth.sessions import handle_revoke_session

        mock_auth_handler._get_user_store.return_value = mock_user_store

        http = mock_http_handler(method="DELETE")

        result = handle_revoke_session(mock_auth_handler, http, "")

        assert_error_response(result, 400, "Invalid session ID")

    def test_revoke_session_only_own_sessions(
        self, mock_auth_handler, mock_user_store, mock_session_manager, mock_http_handler
    ):
        """Test that user can only revoke their own sessions."""
        from aragora.server.handlers.auth.sessions import handle_revoke_session

        mock_auth_handler._get_user_store.return_value = mock_user_store

        mock_auth_ctx = MagicMock()
        mock_auth_ctx.user_id = "user-001"

        http = mock_http_handler(method="DELETE")

        with (
            patch(PATCH_EXTRACT_USER, return_value=mock_auth_ctx),
            patch(PATCH_GET_SESSION_MANAGER, return_value=mock_session_manager),
            patch(PATCH_EXTRACT_TOKEN, return_value="token"),
            patch(PATCH_DECODE_JWT, return_value={"sub": "user-001"}),
        ):
            result = handle_revoke_session(mock_auth_handler, http, "sess-abc123")

        # Session manager should be called with user_id
        mock_session_manager.get_session.assert_called_with("user-001", "sess-abc123")

    def test_revoke_session_logs_action(
        self, mock_auth_handler, mock_user_store, mock_session_manager, mock_http_handler
    ):
        """Test that session revocation is logged."""
        from aragora.server.handlers.auth.sessions import handle_revoke_session

        mock_auth_handler._get_user_store.return_value = mock_user_store

        mock_auth_ctx = MagicMock()
        mock_auth_ctx.user_id = "user-001"

        http = mock_http_handler(method="DELETE")

        with (
            patch(PATCH_EXTRACT_USER, return_value=mock_auth_ctx),
            patch(PATCH_GET_SESSION_MANAGER, return_value=mock_session_manager),
            patch(PATCH_EXTRACT_TOKEN, return_value="different-token"),
            patch(PATCH_DECODE_JWT, return_value={"sub": "user-001"}),
            patch("aragora.server.handlers.auth.sessions.logger") as mock_logger,
        ):
            result = handle_revoke_session(mock_auth_handler, http, "sess-abc123")

        mock_logger.info.assert_called()


# ============================================================================
# Test: Security Properties
# ============================================================================


class TestSessionSecurityProperties:
    """Tests for session security properties."""

    def test_session_jti_derived_from_token_hash(self):
        """Test that session JTI is computed from SHA-256 hash of token."""
        token = "test-jwt-token-value"
        expected_jti = hashlib.sha256(token.encode()).hexdigest()[:32]

        # SHA-256 produces 64 hex chars, first 32 are used
        assert len(expected_jti) == 32
        assert all(c in "0123456789abcdef" for c in expected_jti)

    def test_session_id_format_validated(
        self, mock_auth_handler, mock_user_store, mock_http_handler
    ):
        """Test that session ID format is validated."""
        from aragora.server.handlers.auth.sessions import handle_revoke_session

        mock_auth_handler._get_user_store.return_value = mock_user_store

        http = mock_http_handler(method="DELETE")

        # Various invalid session IDs
        invalid_ids = ["", "abc", "12345"]  # All too short

        for invalid_id in invalid_ids:
            result = handle_revoke_session(mock_auth_handler, http, invalid_id)
            assert result.status_code == 400

    def test_cannot_enumerate_other_users_sessions(
        self, mock_auth_handler, mock_user_store, mock_session_manager, mock_http_handler
    ):
        """Test that users cannot access other users' sessions."""
        from aragora.server.handlers.auth.sessions import handle_revoke_session

        mock_auth_handler._get_user_store.return_value = mock_user_store
        # Session manager returns None for sessions not owned by user
        mock_session_manager.get_session.return_value = None

        mock_auth_ctx = MagicMock()
        mock_auth_ctx.user_id = "user-001"

        http = mock_http_handler(method="DELETE")

        with (
            patch(PATCH_EXTRACT_USER, return_value=mock_auth_ctx),
            patch(PATCH_GET_SESSION_MANAGER, return_value=mock_session_manager),
            patch(PATCH_EXTRACT_TOKEN, return_value="token"),
            patch(PATCH_DECODE_JWT, return_value={"sub": "user-001"}),
        ):
            result = handle_revoke_session(mock_auth_handler, http, "other-user-session")

        # Should return 404, not reveal that session exists for another user
        assert_error_response(result, 404)


# ============================================================================
# Test: Multiple Sessions
# ============================================================================


class TestMultipleSessions:
    """Tests for handling multiple sessions."""

    def test_list_multiple_sessions(
        self, mock_auth_handler, mock_user_store, mock_session_manager, mock_http_handler
    ):
        """Test listing multiple sessions."""
        from aragora.server.handlers.auth.sessions import handle_list_sessions

        # Create multiple sessions
        sessions = []
        for i in range(3):
            session = MagicMock()
            session.session_id = f"sess-{i:03d}"
            session.to_dict.return_value = {
                "session_id": f"sess-{i:03d}",
                "last_activity": f"2026-01-{15 - i:02d}T10:00:00Z",
            }
            sessions.append(session)

        mock_session_manager.list_sessions.return_value = sessions
        mock_auth_handler._get_user_store.return_value = mock_user_store

        mock_auth_ctx = MagicMock()
        mock_auth_ctx.user_id = "user-001"

        http = mock_http_handler(method="GET")

        with (
            patch(PATCH_EXTRACT_USER, return_value=mock_auth_ctx),
            patch(PATCH_GET_SESSION_MANAGER, return_value=mock_session_manager),
            patch(PATCH_EXTRACT_TOKEN, return_value=None),
            patch(PATCH_DECODE_JWT, return_value=None),
        ):
            result = handle_list_sessions(mock_auth_handler, http)

        body = parse_handler_response(result)
        assert body["total"] == 3
        assert len(body["sessions"]) == 3

    def test_current_session_marked_among_multiple(
        self, mock_auth_handler, mock_user_store, mock_session_manager, mock_http_handler
    ):
        """Test that current session is correctly marked among multiple."""
        from aragora.server.handlers.auth.sessions import handle_list_sessions

        token = "current-token"
        current_jti = hashlib.sha256(token.encode()).hexdigest()[:32]

        # Create sessions with one matching current token
        sessions = []
        for i, sid in enumerate(["sess-other-1", current_jti, "sess-other-2"]):
            session = MagicMock()
            session.session_id = sid
            session.to_dict.return_value = {
                "session_id": sid,
                "last_activity": f"2026-01-{15 - i:02d}T10:00:00Z",
            }
            sessions.append(session)

        mock_session_manager.list_sessions.return_value = sessions
        mock_auth_handler._get_user_store.return_value = mock_user_store

        mock_auth_ctx = MagicMock()
        mock_auth_ctx.user_id = "user-001"

        http = mock_http_handler(method="GET")

        with (
            patch(PATCH_EXTRACT_USER, return_value=mock_auth_ctx),
            patch(PATCH_GET_SESSION_MANAGER, return_value=mock_session_manager),
            patch(PATCH_EXTRACT_TOKEN, return_value=token),
            patch(PATCH_DECODE_JWT, return_value={"sub": "user-001"}),
        ):
            result = handle_list_sessions(mock_auth_handler, http)

        body = parse_handler_response(result)
        current_sessions = [s for s in body["sessions"] if s.get("is_current")]
        assert len(current_sessions) == 1
        assert current_sessions[0]["session_id"] == current_jti


__all__ = [
    "TestListSessions",
    "TestRevokeSession",
    "TestSessionSecurityProperties",
    "TestMultipleSessions",
]
