"""Tests for session management handler functions (aragora/server/handlers/auth/sessions.py).

Covers both session endpoints:
- GET /api/auth/sessions       -> handle_list_sessions
- DELETE /api/auth/sessions/:id -> handle_revoke_session

Tests exercise: success paths, permission checks, empty sessions, sorting,
current-session marking, session ID validation, revoke-current-session guard,
session-not-found, audit event emission, edge cases (short IDs, missing tokens,
no JTI in payload, etc.).
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.auth.sessions import (
    handle_list_sessions,
    handle_revoke_session,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result: object) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    if isinstance(result, dict):
        return result
    return json.loads(result.body)


def _status(result: object) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


class MockHTTPHandler:
    """Lightweight mock HTTP request handler."""

    def __init__(self, body: dict | None = None, method: str = "GET"):
        self.command = method
        self.client_address = ("127.0.0.1", 12345)
        self.headers: dict[str, str] = {
            "User-Agent": "test-agent",
            "Authorization": "Bearer test-token-abc",
        }
        self.rfile = MagicMock()
        if body is not None:
            body_bytes = json.dumps(body).encode()
            self.rfile.read.return_value = body_bytes
            self.headers["Content-Length"] = str(len(body_bytes))
        else:
            self.rfile.read.return_value = b"{}"
            self.headers["Content-Length"] = "2"


@dataclass
class MockAuthCtx:
    """Mock auth context from extract_user_from_request."""

    is_authenticated: bool = True
    user_id: str = "user-001"
    email: str = "test@example.com"
    org_id: str = "org-001"
    role: str = "admin"
    client_ip: str = "127.0.0.1"


@dataclass
class MockJWTPayload:
    """Mock JWT payload returned by decode_jwt."""

    user_id: str = "user-001"
    jti: str | None = None
    sub: str = "user-001"


@dataclass
class MockSession:
    """Mock session object matching JWTSession interface."""

    session_id: str
    user_id: str = "user-001"
    created_at: float = field(default_factory=lambda: time.time() - 3600)
    last_activity: float = field(default_factory=time.time)
    ip_address: str | None = "127.0.0.1"
    user_agent: str | None = "test-agent"
    device_name: str | None = "Test Device"
    expires_at: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "created_at": datetime.fromtimestamp(self.created_at, timezone.utc).isoformat(),
            "last_activity": datetime.fromtimestamp(self.last_activity, timezone.utc).isoformat(),
            "ip_address": self.ip_address,
            "device_name": self.device_name,
            "is_current": False,
            "expires_at": (
                datetime.fromtimestamp(self.expires_at, timezone.utc).isoformat()
                if self.expires_at
                else None
            ),
        }


class MockSessionManager:
    """Mock session manager with controllable return values."""

    def __init__(self):
        self.sessions: list[MockSession] = []
        self._get_session_return: MockSession | None = None
        self.revoke_calls: list[tuple[str, str]] = []

    def list_sessions(self, user_id: str) -> list[MockSession]:
        return [s for s in self.sessions if s.user_id == user_id]

    def get_session(self, user_id: str, session_id: str) -> MockSession | None:
        if self._get_session_return is not None:
            return self._get_session_return
        for s in self.sessions:
            if s.user_id == user_id and s.session_id == session_id:
                return s
        return None

    def revoke_session(self, user_id: str, session_id: str) -> bool:
        self.revoke_calls.append((user_id, session_id))
        return True


def _make_user_store():
    """Create a mock user store."""
    store = MagicMock()
    return store


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _patch_session_deps(monkeypatch):
    """Patch dependencies common to all session handler functions."""
    mock_auth_ctx = MockAuthCtx()

    # Patch extract_user_from_request used at module level by sessions.py
    monkeypatch.setattr(
        "aragora.server.handlers.auth.sessions.extract_user_from_request",
        lambda handler, user_store: mock_auth_ctx,
    )

    # Patch emit_handler_event to no-op
    monkeypatch.setattr(
        "aragora.server.handlers.auth.sessions.emit_handler_event",
        lambda *args, **kwargs: None,
    )


@pytest.fixture
def session_manager():
    """Create a mock session manager."""
    return MockSessionManager()


@pytest.fixture
def handler_instance(session_manager):
    """Create an AuthHandler with mocked dependencies."""
    from aragora.server.handlers.auth.handler import AuthHandler

    store = _make_user_store()
    h = AuthHandler(server_context={"user_store": store})
    # Always grant permissions
    h._check_permission = MagicMock(return_value=None)
    return h, store, session_manager


@pytest.fixture
def http():
    """Factory for creating mock HTTP handlers."""

    def _create(body: dict | None = None, method: str = "GET") -> MockHTTPHandler:
        return MockHTTPHandler(body=body, method=method)

    return _create


def _patch_local_imports(monkeypatch, mgr, extract_token_return=None, decode_jwt_return=None):
    """Patch the locally-imported dependencies (get_session_manager, extract_token, decode_jwt).

    These are imported inside the function body, so we patch at their source modules.
    """
    monkeypatch.setattr(
        "aragora.billing.auth.sessions.get_session_manager",
        lambda: mgr,
    )
    monkeypatch.setattr(
        "aragora.server.middleware.auth.extract_token",
        lambda h: extract_token_return,
    )
    monkeypatch.setattr(
        "aragora.billing.jwt_auth.decode_jwt",
        lambda t: decode_jwt_return,
    )


# =========================================================================
# handle_list_sessions
# =========================================================================


class TestListSessions:
    """GET /api/auth/sessions."""

    def test_list_sessions_success_empty(self, handler_instance, http, monkeypatch):
        hi, store, mgr = handler_instance
        _patch_local_imports(monkeypatch, mgr)
        result = handle_list_sessions(hi, http())
        assert _status(result) == 200
        body = _body(result)
        assert body["sessions"] == []
        assert body["total"] == 0

    def test_list_sessions_returns_sessions(self, handler_instance, http, monkeypatch):
        hi, store, mgr = handler_instance
        now = time.time()
        mgr.sessions = [
            MockSession(session_id="sess-aaa", last_activity=now - 100),
            MockSession(session_id="sess-bbb", last_activity=now - 50),
        ]
        _patch_local_imports(monkeypatch, mgr)
        result = handle_list_sessions(hi, http())
        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 2
        assert len(body["sessions"]) == 2

    def test_list_sessions_sorted_by_last_activity_desc(self, handler_instance, http, monkeypatch):
        hi, store, mgr = handler_instance
        now = time.time()
        mgr.sessions = [
            MockSession(session_id="old", last_activity=now - 3600),
            MockSession(session_id="newest", last_activity=now),
            MockSession(session_id="middle", last_activity=now - 1800),
        ]
        _patch_local_imports(monkeypatch, mgr)
        result = handle_list_sessions(hi, http())
        body = _body(result)
        session_ids = [s["session_id"] for s in body["sessions"]]
        assert session_ids[0] == "newest"
        assert session_ids[1] == "middle"
        assert session_ids[2] == "old"

    def test_list_sessions_marks_current_session_via_jti(self, handler_instance, http, monkeypatch):
        hi, store, mgr = handler_instance
        now = time.time()
        mgr.sessions = [
            MockSession(session_id="other-session", last_activity=now - 100),
            MockSession(session_id="my-jti-123", last_activity=now),
        ]
        payload = MockJWTPayload(jti="my-jti-123")
        _patch_local_imports(monkeypatch, mgr, extract_token_return="some-token", decode_jwt_return=payload)
        result = handle_list_sessions(hi, http())
        body = _body(result)
        for s in body["sessions"]:
            if s["session_id"] == "my-jti-123":
                assert s["is_current"] is True
            else:
                assert s["is_current"] is False

    def test_list_sessions_marks_current_via_token_hash_fallback(self, handler_instance, http, monkeypatch):
        """When JWT payload has no jti, falls back to sha256 hash of token."""
        hi, store, mgr = handler_instance
        now = time.time()
        token = "my-bearer-token"
        token_hash = hashlib.sha256(token.encode()).hexdigest()[:32]
        mgr.sessions = [
            MockSession(session_id=token_hash, last_activity=now),
            MockSession(session_id="other-sess", last_activity=now - 100),
        ]
        # Payload without jti
        payload = MockJWTPayload(jti=None)
        _patch_local_imports(monkeypatch, mgr, extract_token_return=token, decode_jwt_return=payload)
        result = handle_list_sessions(hi, http())
        body = _body(result)
        for s in body["sessions"]:
            if s["session_id"] == token_hash:
                assert s["is_current"] is True
            else:
                assert s["is_current"] is False

    def test_list_sessions_no_current_when_no_token(self, handler_instance, http, monkeypatch):
        hi, store, mgr = handler_instance
        now = time.time()
        mgr.sessions = [
            MockSession(session_id="sess-1", last_activity=now),
        ]
        _patch_local_imports(monkeypatch, mgr)
        result = handle_list_sessions(hi, http())
        body = _body(result)
        assert body["sessions"][0]["is_current"] is False

    def test_list_sessions_no_current_when_decode_fails(self, handler_instance, http, monkeypatch):
        """When decode_jwt returns None, current_jti stays None."""
        hi, store, mgr = handler_instance
        now = time.time()
        mgr.sessions = [
            MockSession(session_id="sess-1", last_activity=now),
        ]
        _patch_local_imports(monkeypatch, mgr, extract_token_return="some-token", decode_jwt_return=None)
        result = handle_list_sessions(hi, http())
        body = _body(result)
        # No session should be marked as current since decode failed
        assert body["sessions"][0]["is_current"] is False

    def test_list_sessions_permission_denied(self, handler_instance, http, monkeypatch):
        from aragora.server.handlers.base import error_response

        hi, store, mgr = handler_instance
        hi._check_permission = MagicMock(
            return_value=error_response("Permission denied", 403)
        )
        _patch_local_imports(monkeypatch, mgr)
        result = handle_list_sessions(hi, http())
        assert _status(result) == 403

    def test_list_sessions_checks_correct_permission(self, handler_instance, http, monkeypatch):
        """Verify the handler checks 'session.list_active' permission."""
        hi, store, mgr = handler_instance
        permission_checked = []

        def track_permission(handler, perm):
            permission_checked.append(perm)
            return None  # Allow

        hi._check_permission = track_permission
        _patch_local_imports(monkeypatch, mgr)
        handle_list_sessions(hi, http())
        assert "session.list_active" in permission_checked

    def test_list_sessions_session_contains_expected_fields(self, handler_instance, http, monkeypatch):
        hi, store, mgr = handler_instance
        now = time.time()
        mgr.sessions = [
            MockSession(
                session_id="sess-field-check",
                last_activity=now,
                ip_address="10.0.0.1",
                device_name="Chrome on macOS",
            ),
        ]
        _patch_local_imports(monkeypatch, mgr)
        result = handle_list_sessions(hi, http())
        body = _body(result)
        session = body["sessions"][0]
        assert "session_id" in session
        assert "user_id" in session
        assert "created_at" in session
        assert "last_activity" in session
        assert "ip_address" in session
        assert "device_name" in session
        assert "is_current" in session

    def test_list_sessions_total_matches_count(self, handler_instance, http, monkeypatch):
        hi, store, mgr = handler_instance
        now = time.time()
        mgr.sessions = [
            MockSession(session_id=f"sess-{i}", last_activity=now - i)
            for i in range(5)
        ]
        _patch_local_imports(monkeypatch, mgr)
        result = handle_list_sessions(hi, http())
        body = _body(result)
        assert body["total"] == 5
        assert len(body["sessions"]) == 5

    def test_list_sessions_only_returns_user_sessions(self, handler_instance, http, monkeypatch):
        """Sessions belonging to other users should not appear."""
        hi, store, mgr = handler_instance
        now = time.time()
        mgr.sessions = [
            MockSession(session_id="mine-1", user_id="user-001", last_activity=now),
            MockSession(session_id="other-1", user_id="user-999", last_activity=now),
        ]
        _patch_local_imports(monkeypatch, mgr)
        result = handle_list_sessions(hi, http())
        body = _body(result)
        assert body["total"] == 1
        assert body["sessions"][0]["session_id"] == "mine-1"


# =========================================================================
# handle_revoke_session
# =========================================================================


class TestRevokeSession:
    """DELETE /api/auth/sessions/:id."""

    def test_revoke_session_success(self, handler_instance, http, monkeypatch):
        hi, store, mgr = handler_instance
        target_session = MockSession(session_id="sess-to-revoke-12345678")
        mgr.sessions = [target_session]
        _patch_local_imports(
            monkeypatch, mgr,
            extract_token_return="some-other-token",
            decode_jwt_return=MockJWTPayload(jti="current-jti"),
        )
        result = handle_revoke_session(hi, http(method="DELETE"), "sess-to-revoke-12345678")
        assert _status(result) == 200
        body = _body(result)
        assert body["success"] is True
        assert body["message"] == "Session revoked successfully"
        assert body["session_id"] == "sess-to-revoke-12345678"

    def test_revoke_session_calls_manager_revoke(self, handler_instance, http, monkeypatch):
        hi, store, mgr = handler_instance
        target_session = MockSession(session_id="sess-revoke-check")
        mgr.sessions = [target_session]
        _patch_local_imports(
            monkeypatch, mgr,
            extract_token_return="token-xyz",
            decode_jwt_return=MockJWTPayload(jti="different-jti"),
        )
        handle_revoke_session(hi, http(method="DELETE"), "sess-revoke-check")
        assert ("user-001", "sess-revoke-check") in mgr.revoke_calls

    def test_revoke_session_permission_denied(self, handler_instance, http, monkeypatch):
        from aragora.server.handlers.base import error_response

        hi, store, mgr = handler_instance
        hi._check_permission = MagicMock(
            return_value=error_response("Permission denied", 403)
        )
        result = handle_revoke_session(hi, http(method="DELETE"), "sess-12345678")
        assert _status(result) == 403

    def test_revoke_session_checks_correct_permission(self, handler_instance, http, monkeypatch):
        """Verify the handler checks 'session.revoke' permission."""
        hi, store, mgr = handler_instance
        permission_checked = []

        def track_permission(handler, perm):
            permission_checked.append(perm)
            return None  # Allow

        hi._check_permission = track_permission
        mgr.sessions = [MockSession(session_id="sess-perm-check")]
        _patch_local_imports(monkeypatch, mgr)
        handle_revoke_session(hi, http(method="DELETE"), "sess-perm-check")
        assert "session.revoke" in permission_checked

    def test_revoke_session_invalid_id_empty(self, handler_instance, http, monkeypatch):
        hi, store, mgr = handler_instance
        _patch_local_imports(monkeypatch, mgr)
        result = handle_revoke_session(hi, http(method="DELETE"), "")
        assert _status(result) == 400
        assert "invalid session id" in _body(result)["error"].lower()

    def test_revoke_session_invalid_id_too_short(self, handler_instance, http, monkeypatch):
        hi, store, mgr = handler_instance
        _patch_local_imports(monkeypatch, mgr)
        result = handle_revoke_session(hi, http(method="DELETE"), "short")
        assert _status(result) == 400
        assert "invalid session id" in _body(result)["error"].lower()

    def test_revoke_session_invalid_id_exactly_7_chars(self, handler_instance, http, monkeypatch):
        """Session IDs < 8 chars are invalid."""
        hi, store, mgr = handler_instance
        _patch_local_imports(monkeypatch, mgr)
        result = handle_revoke_session(hi, http(method="DELETE"), "1234567")
        assert _status(result) == 400

    def test_revoke_session_valid_id_exactly_8_chars(self, handler_instance, http, monkeypatch):
        """Session IDs of exactly 8 chars should pass validation."""
        hi, store, mgr = handler_instance
        mgr.sessions = [MockSession(session_id="12345678")]
        _patch_local_imports(monkeypatch, mgr)
        result = handle_revoke_session(hi, http(method="DELETE"), "12345678")
        assert _status(result) == 200

    def test_revoke_current_session_blocked_by_jti(self, handler_instance, http, monkeypatch):
        """Cannot revoke current session when matching via JWT jti."""
        hi, store, mgr = handler_instance
        current_jti = "current-session-jti-12345"
        _patch_local_imports(
            monkeypatch, mgr,
            extract_token_return="my-token",
            decode_jwt_return=MockJWTPayload(jti=current_jti),
        )
        result = handle_revoke_session(hi, http(method="DELETE"), current_jti)
        assert _status(result) == 400
        assert "cannot revoke current session" in _body(result)["error"].lower()
        assert "/api/auth/logout" in _body(result)["error"].lower()

    def test_revoke_current_session_blocked_by_token_hash(self, handler_instance, http, monkeypatch):
        """Cannot revoke current session when matching via token SHA256 hash."""
        hi, store, mgr = handler_instance
        token = "my-special-bearer-token"
        token_hash = hashlib.sha256(token.encode()).hexdigest()[:32]
        _patch_local_imports(
            monkeypatch, mgr,
            extract_token_return=token,
            # Payload without jti so only the hash is used
            decode_jwt_return=MockJWTPayload(jti=None),
        )
        result = handle_revoke_session(hi, http(method="DELETE"), token_hash)
        assert _status(result) == 400
        assert "cannot revoke current session" in _body(result)["error"].lower()

    def test_revoke_session_not_found(self, handler_instance, http, monkeypatch):
        hi, store, mgr = handler_instance
        # No sessions in manager
        _patch_local_imports(
            monkeypatch, mgr,
            extract_token_return="other-token",
            decode_jwt_return=MockJWTPayload(jti="different-jti"),
        )
        result = handle_revoke_session(hi, http(method="DELETE"), "nonexistent-session-id")
        assert _status(result) == 404
        assert "not found" in _body(result)["error"].lower()

    def test_revoke_session_emits_handler_event(self, handler_instance, http, monkeypatch):
        hi, store, mgr = handler_instance
        target = MockSession(session_id="sess-event-test")
        mgr.sessions = [target]
        events = []

        _patch_local_imports(
            monkeypatch, mgr,
            extract_token_return="other-tok",
            decode_jwt_return=MockJWTPayload(jti="other-jti"),
        )
        monkeypatch.setattr(
            "aragora.server.handlers.auth.sessions.emit_handler_event",
            lambda *args, **kwargs: events.append((args, kwargs)),
        )
        handle_revoke_session(hi, http(method="DELETE"), "sess-event-test")
        assert len(events) == 1
        args, kwargs = events[0]
        assert args[0] == "auth"
        assert kwargs.get("user_id") == "user-001"

    def test_revoke_session_no_token_still_works(self, handler_instance, http, monkeypatch):
        """When extract_token returns None, current_jtis is empty and revoke proceeds."""
        hi, store, mgr = handler_instance
        target = MockSession(session_id="sess-no-token-test")
        mgr.sessions = [target]
        _patch_local_imports(monkeypatch, mgr)
        result = handle_revoke_session(hi, http(method="DELETE"), "sess-no-token-test")
        assert _status(result) == 200
        assert _body(result)["success"] is True

    def test_revoke_session_decode_returns_none(self, handler_instance, http, monkeypatch):
        """When decode_jwt returns None, only token hash is in current_jtis."""
        hi, store, mgr = handler_instance
        token = "my-token-for-none-decode"
        token_hash = hashlib.sha256(token.encode()).hexdigest()[:32]
        # Use a session ID that is NOT the token hash
        target = MockSession(session_id="sess-other-12345678")
        mgr.sessions = [target]
        _patch_local_imports(monkeypatch, mgr, extract_token_return=token, decode_jwt_return=None)
        result = handle_revoke_session(hi, http(method="DELETE"), "sess-other-12345678")
        assert _status(result) == 200

    def test_revoke_session_dict_payload_with_jti(self, handler_instance, http, monkeypatch):
        """When decode_jwt returns a dict with 'jti', that jti is added to current_jtis."""
        hi, store, mgr = handler_instance
        target = MockSession(session_id="other-session-12345")
        mgr.sessions = [target]
        _patch_local_imports(
            monkeypatch, mgr,
            extract_token_return="token-value",
            # Return a plain dict (the code checks isinstance(payload, dict))
            decode_jwt_return={"jti": "dict-jti-value", "sub": "user-001"},
        )
        # Try revoking the dict jti - should be blocked
        result = handle_revoke_session(hi, http(method="DELETE"), "dict-jti-value")
        assert _status(result) == 400
        assert "cannot revoke current session" in _body(result)["error"].lower()

    def test_revoke_session_dict_payload_without_jti(self, handler_instance, http, monkeypatch):
        """When decode_jwt returns a dict without 'jti', only token hash is tracked."""
        hi, store, mgr = handler_instance
        target = MockSession(session_id="normal-session-123")
        mgr.sessions = [target]
        _patch_local_imports(
            monkeypatch, mgr,
            extract_token_return="token-value",
            decode_jwt_return={"sub": "user-001"},  # no jti
        )
        result = handle_revoke_session(hi, http(method="DELETE"), "normal-session-123")
        assert _status(result) == 200

    def test_revoke_session_both_jti_and_hash_blocked(self, handler_instance, http, monkeypatch):
        """Both jti from payload and token hash should be in current_jtis."""
        hi, store, mgr = handler_instance
        token = "my-token-for-double-block"
        token_hash = hashlib.sha256(token.encode()).hexdigest()[:32]
        _patch_local_imports(
            monkeypatch, mgr,
            extract_token_return=token,
            decode_jwt_return=MockJWTPayload(jti="payload-jti-value"),
        )
        # Block by jti
        result1 = handle_revoke_session(hi, http(method="DELETE"), "payload-jti-value")
        assert _status(result1) == 400

        # Block by token hash
        result2 = handle_revoke_session(hi, http(method="DELETE"), token_hash)
        assert _status(result2) == 400


# =========================================================================
# handle_revoke_session - edge cases
# =========================================================================


class TestRevokeSessionEdgeCases:
    """Edge cases for session revocation."""

    def test_revoke_session_long_session_id(self, handler_instance, http, monkeypatch):
        hi, store, mgr = handler_instance
        long_id = "a" * 256
        target = MockSession(session_id=long_id)
        mgr.sessions = [target]
        _patch_local_imports(monkeypatch, mgr)
        result = handle_revoke_session(hi, http(method="DELETE"), long_id)
        assert _status(result) == 200
        assert _body(result)["session_id"] == long_id

    def test_revoke_session_special_chars_in_id(self, handler_instance, http, monkeypatch):
        hi, store, mgr = handler_instance
        special_id = "sess-abc!@#$%^&*()"
        target = MockSession(session_id=special_id)
        mgr.sessions = [target]
        _patch_local_imports(monkeypatch, mgr)
        result = handle_revoke_session(hi, http(method="DELETE"), special_id)
        assert _status(result) == 200

    def test_revoke_session_returns_session_id_in_response(self, handler_instance, http, monkeypatch):
        hi, store, mgr = handler_instance
        session_id = "sess-return-check-1234"
        target = MockSession(session_id=session_id)
        mgr.sessions = [target]
        _patch_local_imports(monkeypatch, mgr)
        result = handle_revoke_session(hi, http(method="DELETE"), session_id)
        body = _body(result)
        assert body["session_id"] == session_id

    def test_revoke_session_manager_called_with_correct_user_id(self, handler_instance, http, monkeypatch):
        """Verify revoke_session is called with the authenticated user's ID."""
        hi, store, mgr = handler_instance
        target = MockSession(session_id="sess-user-check-12")
        mgr.sessions = [target]
        _patch_local_imports(monkeypatch, mgr)
        handle_revoke_session(hi, http(method="DELETE"), "sess-user-check-12")
        assert len(mgr.revoke_calls) == 1
        user_id, session_id = mgr.revoke_calls[0]
        assert user_id == "user-001"
        assert session_id == "sess-user-check-12"


# =========================================================================
# Routing integration tests via AuthHandler.handle()
# =========================================================================


class TestSessionRouting:
    """Test that AuthHandler routes session endpoints correctly."""

    @pytest.fixture
    def auth_handler(self, session_manager, monkeypatch):
        from aragora.server.handlers.auth.handler import AuthHandler

        store = _make_user_store()
        h = AuthHandler(server_context={"user_store": store})
        h._check_permission = MagicMock(return_value=None)
        _patch_local_imports(monkeypatch, session_manager)
        return h

    @pytest.mark.asyncio
    async def test_route_list_sessions(self, auth_handler):
        result = await auth_handler.handle("/api/auth/sessions", {}, MockHTTPHandler(), "GET")
        assert _status(result) == 200
        body = _body(result)
        assert "sessions" in body
        assert "total" in body

    @pytest.mark.asyncio
    async def test_route_list_sessions_with_v1_prefix(self, auth_handler):
        result = await auth_handler.handle("/api/v1/auth/sessions", {}, MockHTTPHandler(), "GET")
        assert _status(result) == 200
        body = _body(result)
        assert "sessions" in body

    @pytest.mark.asyncio
    async def test_route_revoke_session(self, auth_handler, session_manager):
        target = MockSession(session_id="sess-route-test")
        session_manager.sessions = [target]
        result = await auth_handler.handle(
            "/api/auth/sessions/sess-route-test", {}, MockHTTPHandler(method="DELETE"), "DELETE"
        )
        assert _status(result) == 200
        assert _body(result)["success"] is True

    @pytest.mark.asyncio
    async def test_route_revoke_session_with_v1_prefix(self, auth_handler, session_manager):
        target = MockSession(session_id="sess-v1-route-test")
        session_manager.sessions = [target]
        result = await auth_handler.handle(
            "/api/v1/auth/sessions/sess-v1-route-test", {}, MockHTTPHandler(method="DELETE"), "DELETE"
        )
        assert _status(result) == 200

    def test_can_handle_sessions_path(self, auth_handler):
        assert auth_handler.can_handle("/api/auth/sessions") is True
        assert auth_handler.can_handle("/api/v1/auth/sessions") is True
        assert auth_handler.can_handle("/api/auth/sessions/some-id") is True
        assert auth_handler.can_handle("/api/v1/auth/sessions/some-id") is True

    @pytest.mark.asyncio
    async def test_route_sessions_wrong_method(self, auth_handler):
        """POST to /api/auth/sessions is not a valid route - should 405."""
        result = await auth_handler.handle("/api/auth/sessions", {}, MockHTTPHandler(method="POST"), "POST")
        # The handler falls through to the 405 at the end
        assert _status(result) == 405


# =========================================================================
# Module exports
# =========================================================================


class TestModuleExports:
    """Verify __all__ exports."""

    def test_all_exports(self):
        from aragora.server.handlers.auth import sessions

        assert "handle_list_sessions" in sessions.__all__
        assert "handle_revoke_session" in sessions.__all__

    def test_all_exports_count(self):
        from aragora.server.handlers.auth import sessions

        assert len(sessions.__all__) == 2
