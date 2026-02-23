"""Comprehensive tests for OpenClaw SessionOrchestrationMixin.

Covers all handler methods defined in
aragora/server/handlers/openclaw/orchestrator.py:

Session handlers:
- _handle_list_sessions  (GET /api/v1/openclaw/sessions)
- _handle_get_session    (GET /api/v1/openclaw/sessions/:id)
- _handle_create_session (POST /api/v1/openclaw/sessions)
- _handle_close_session  (DELETE /api/v1/openclaw/sessions/:id)
- _handle_end_session    (POST /api/v1/openclaw/sessions/:id/end)

Action handlers:
- _handle_get_action     (GET /api/v1/openclaw/actions/:id)
- _handle_execute_action (POST /api/v1/openclaw/actions)
- _handle_cancel_action  (POST /api/v1/openclaw/actions/:id/cancel)

Test categories:
- Happy paths for every endpoint
- Access control (ownership, admin bypass)
- Validation errors (missing fields, invalid config, invalid metadata)
- Not found (404) responses
- Session status checks (inactive session, non-cancellable action)
- Store error handling (exceptions -> 500)
- Query parameter parsing (status filter, pagination)
- Audit logging side effects
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.openclaw.gateway import OpenClawGatewayHandler
from aragora.server.handlers.openclaw.models import (
    Action,
    ActionStatus,
    Session,
    SessionStatus,
)
from aragora.server.handlers.openclaw.store import OpenClawGatewayStore


# ============================================================================
# Helpers
# ============================================================================


def _body(result) -> dict[str, Any]:
    """Decode a HandlerResult body to dict."""
    if result is None:
        return {}
    if hasattr(result, "body"):
        return json.loads(result.body)
    return result


def _status(result) -> int:
    """Extract status code from HandlerResult."""
    if result is None:
        return 0
    if hasattr(result, "status_code"):
        return result.status_code
    return 0


class MockHTTPHandler:
    """Minimal mock HTTP handler for BaseHandler methods."""

    def __init__(self, body: dict | None = None, method: str = "GET"):
        self.rfile = MagicMock()
        self.command = method
        self.client_address = ("127.0.0.1", 54321)
        if body is not None:
            body_bytes = json.dumps(body).encode()
            self.rfile.read.return_value = body_bytes
            self.headers = {
                "Content-Length": str(len(body_bytes)),
                "Content-Type": "application/json",
            }
        else:
            self.rfile.read.return_value = b"{}"
            self.headers = {
                "Content-Length": "2",
                "Content-Type": "application/json",
            }


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture()
def store():
    """Fresh in-memory OpenClaw store for each test."""
    return OpenClawGatewayStore()


class _MockUser:
    """Minimal mock user for get_current_user."""

    def __init__(self, user_id="test-user-001", org_id="test-org-001", role="admin"):
        self.user_id = user_id
        self.org_id = org_id
        self.role = role


@pytest.fixture()
def mock_user():
    """Default mock user returned by get_current_user."""
    return _MockUser()


@pytest.fixture()
def handler(store, mock_user):
    """OpenClawGatewayHandler with _get_store and get_current_user patched."""
    with (
        patch(
            "aragora.server.handlers.openclaw.orchestrator._get_store",
            return_value=store,
        ),
        patch(
            "aragora.server.handlers.openclaw.credentials._get_store",
            return_value=store,
        ),
        patch(
            "aragora.server.handlers.openclaw.policies._get_store",
            return_value=store,
        ),
    ):
        h = OpenClawGatewayHandler(server_context={})
        # Override get_current_user on the instance so _get_user_id / _get_tenant_id
        # resolve correctly (OpenClawMixinBase.get_current_user is a stub that raises).
        h.get_current_user = lambda handler: mock_user
        yield h


@pytest.fixture()
def mock_http():
    """Factory for MockHTTPHandler."""

    def _make(body: dict | None = None, method: str = "GET") -> MockHTTPHandler:
        return MockHTTPHandler(body=body, method=method)

    return _make


@pytest.fixture()
def active_session(store) -> Session:
    """Pre-create an active session owned by 'test-user-001'."""
    return store.create_session(
        user_id="test-user-001",
        tenant_id="test-org-001",
        config={"timeout": 300},
        metadata={"source": "test"},
    )


@pytest.fixture()
def other_user_session(store) -> Session:
    """Pre-create an active session owned by a different user."""
    return store.create_session(
        user_id="other-user-999",
        tenant_id="other-org",
        config={},
        metadata={},
    )


@pytest.fixture()
def pending_action(store, active_session) -> Action:
    """Pre-create a pending action in the active session."""
    return store.create_action(
        session_id=active_session.id,
        action_type="code.execute",
        input_data={"code": "print('hello')"},
        metadata={"lang": "python"},
    )


@pytest.fixture()
def running_action(store, active_session) -> Action:
    """Pre-create a running action."""
    action = store.create_action(
        session_id=active_session.id,
        action_type="code.execute",
        input_data={"code": "print('running')"},
    )
    store.update_action(action.id, status=ActionStatus.RUNNING)
    return action


# ============================================================================
# Session: List Sessions
# ============================================================================


class TestListSessions:
    """Tests for _handle_list_sessions (GET /sessions)."""

    def test_list_sessions_empty(self, handler, mock_http):
        result = handler.handle("/api/v1/openclaw/sessions", {}, mock_http())
        assert _status(result) == 200
        body = _body(result)
        assert body["sessions"] == []
        assert body["total"] == 0
        assert body["limit"] == 50
        assert body["offset"] == 0

    def test_list_sessions_returns_sessions(self, handler, mock_http, active_session):
        result = handler.handle("/api/v1/openclaw/sessions", {}, mock_http())
        assert _status(result) == 200
        body = _body(result)
        assert body["total"] >= 1
        ids = [s["id"] for s in body["sessions"]]
        assert active_session.id in ids

    def test_list_sessions_status_filter(self, handler, mock_http, store, active_session):
        # Close one session, create another active
        store.update_session_status(active_session.id, SessionStatus.CLOSED)
        store.create_session(user_id="test-user-001")

        result = handler.handle("/api/v1/openclaw/sessions", {"status": "active"}, mock_http())
        assert _status(result) == 200
        body = _body(result)
        for s in body["sessions"]:
            assert s["status"] == "active"

    def test_list_sessions_pagination(self, handler, mock_http, store):
        # Create 5 sessions with matching tenant_id (handler sends tenant_id from mock_user)
        for _ in range(5):
            store.create_session(user_id="test-user-001", tenant_id="test-org-001")

        result = handler.handle(
            "/api/v1/openclaw/sessions", {"limit": "2", "offset": "0"}, mock_http()
        )
        assert _status(result) == 200
        body = _body(result)
        assert len(body["sessions"]) == 2
        assert body["total"] == 5
        assert body["limit"] == 2
        assert body["offset"] == 0

    def test_list_sessions_invalid_status_returns_400(self, handler, mock_http):
        result = handler.handle("/api/v1/openclaw/sessions", {"status": "bogus"}, mock_http())
        assert _status(result) == 400

    def test_list_sessions_legacy_path(self, handler, mock_http, active_session):
        result = handler.handle("/api/gateway/openclaw/sessions", {}, mock_http())
        assert _status(result) == 200
        body = _body(result)
        assert body["total"] >= 1

    def test_list_sessions_store_error_returns_500(self, handler, mock_http):
        with patch(
            "aragora.server.handlers.openclaw.orchestrator._get_store",
            side_effect=OSError("db down"),
        ):
            result = handler.handle("/api/v1/openclaw/sessions", {}, mock_http())
            assert _status(result) == 500


# ============================================================================
# Session: Get Session
# ============================================================================


class TestGetSession:
    """Tests for _handle_get_session (GET /sessions/:id)."""

    def test_get_session_success(self, handler, mock_http, active_session):
        result = handler.handle(f"/api/v1/openclaw/sessions/{active_session.id}", {}, mock_http())
        assert _status(result) == 200
        body = _body(result)
        assert body["id"] == active_session.id
        assert body["user_id"] == "test-user-001"
        assert body["status"] == "active"

    def test_get_session_not_found(self, handler, mock_http):
        result = handler.handle("/api/v1/openclaw/sessions/nonexistent-id", {}, mock_http())
        assert _status(result) == 404

    def test_get_session_other_user_denied(self, handler, mock_http, other_user_session):
        """Non-admin user cannot read another user's session."""
        # The conftest patches get_current_user to return a user with user_id='test-user-001'
        # other_user_session is owned by 'other-user-999', so access should be denied
        # But conftest gives admin role, so we need to mock _has_permission to return False
        with patch(
            "aragora.server.handlers.openclaw.orchestrator._has_permission",
            return_value=False,
        ):
            result = handler.handle(
                f"/api/v1/openclaw/sessions/{other_user_session.id}", {}, mock_http()
            )
            assert _status(result) == 403

    def test_get_session_admin_can_read_any(self, handler, mock_http, other_user_session):
        """Admin can read any session, even if not the owner."""
        with patch(
            "aragora.server.handlers.openclaw.orchestrator._has_permission",
            return_value=True,
        ):
            result = handler.handle(
                f"/api/v1/openclaw/sessions/{other_user_session.id}", {}, mock_http()
            )
            assert _status(result) == 200
            body = _body(result)
            assert body["id"] == other_user_session.id

    def test_get_session_store_error(self, handler, mock_http, active_session):
        with patch(
            "aragora.server.handlers.openclaw.orchestrator._get_store",
        ) as mock_store:
            mock_store.return_value.get_session.side_effect = TypeError("boom")
            result = handler.handle(
                f"/api/v1/openclaw/sessions/{active_session.id}", {}, mock_http()
            )
            assert _status(result) == 500


# ============================================================================
# Session: Create Session
# ============================================================================


class TestCreateSession:
    """Tests for _handle_create_session (POST /sessions)."""

    def test_create_session_success(self, handler, mock_http, store):
        http = mock_http(
            body={"config": {"timeout": 600}, "metadata": {"env": "test"}},
            method="POST",
        )
        result = handler.handle_post("/api/v1/openclaw/sessions", {}, http)
        assert _status(result) == 201
        body = _body(result)
        assert body["status"] == "active"
        assert body["user_id"] == "test-user-001"
        assert body["config"]["timeout"] == 600
        # Verify persisted in store
        assert store.get_session(body["id"]) is not None

    def test_create_session_empty_body(self, handler, mock_http, store):
        http = mock_http(body={}, method="POST")
        result = handler.handle_post("/api/v1/openclaw/sessions", {}, http)
        assert _status(result) == 201
        body = _body(result)
        assert body["status"] == "active"

    def test_create_session_invalid_config(self, handler, mock_http):
        # Config with too deep nesting (> MAX_SESSION_CONFIG_DEPTH=5)
        nested = {"a": {"b": {"c": {"d": {"e": {"f": {"g": 1}}}}}}}
        http = mock_http(body={"config": nested}, method="POST")
        result = handler.handle_post("/api/v1/openclaw/sessions", {}, http)
        assert _status(result) == 400

    def test_create_session_invalid_metadata(self, handler, mock_http):
        # Metadata must be a dict
        http = mock_http(body={"metadata": "not-a-dict"}, method="POST")
        result = handler.handle_post("/api/v1/openclaw/sessions", {}, http)
        assert _status(result) == 400

    def test_create_session_audit_logged(self, handler, mock_http, store):
        http = mock_http(body={}, method="POST")
        handler.handle_post("/api/v1/openclaw/sessions", {}, http)
        entries, total = store.get_audit_log(action="session.create")
        assert total >= 1
        assert entries[0].action == "session.create"
        assert entries[0].result == "success"

    def test_create_session_store_error(self, handler, mock_http):
        with patch(
            "aragora.server.handlers.openclaw.orchestrator._get_store",
        ) as mock_store:
            mock_store.return_value.create_session.side_effect = OSError("disk full")
            http = mock_http(body={}, method="POST")
            result = handler.handle_post("/api/v1/openclaw/sessions", {}, http)
            assert _status(result) == 500


# ============================================================================
# Session: Close Session
# ============================================================================


class TestCloseSession:
    """Tests for _handle_close_session (DELETE /sessions/:id)."""

    def test_close_session_success(self, handler, mock_http, active_session, store):
        result = handler.handle_delete(
            f"/api/v1/openclaw/sessions/{active_session.id}", {}, mock_http()
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["closed"] is True
        assert body["session_id"] == active_session.id
        # Verify session is now closed in store
        session = store.get_session(active_session.id)
        assert session.status == SessionStatus.CLOSED

    def test_close_session_not_found(self, handler, mock_http):
        result = handler.handle_delete("/api/v1/openclaw/sessions/nonexistent", {}, mock_http())
        assert _status(result) == 404

    def test_close_session_other_user_denied(self, handler, mock_http, other_user_session):
        with patch(
            "aragora.server.handlers.openclaw.orchestrator._has_permission",
            return_value=False,
        ):
            result = handler.handle_delete(
                f"/api/v1/openclaw/sessions/{other_user_session.id}", {}, mock_http()
            )
            assert _status(result) == 403

    def test_close_session_admin_can_close_any(self, handler, mock_http, other_user_session, store):
        with patch(
            "aragora.server.handlers.openclaw.orchestrator._has_permission",
            return_value=True,
        ):
            result = handler.handle_delete(
                f"/api/v1/openclaw/sessions/{other_user_session.id}", {}, mock_http()
            )
            assert _status(result) == 200
            session = store.get_session(other_user_session.id)
            assert session.status == SessionStatus.CLOSED

    def test_close_session_audit_logged(self, handler, mock_http, active_session, store):
        handler.handle_delete(f"/api/v1/openclaw/sessions/{active_session.id}", {}, mock_http())
        entries, total = store.get_audit_log(action="session.close")
        assert total >= 1

    def test_close_session_store_error(self, handler, mock_http, active_session):
        with patch(
            "aragora.server.handlers.openclaw.orchestrator._get_store",
        ) as mock_store:
            mock_store.return_value.get_session.side_effect = AttributeError("oops")
            result = handler.handle_delete(
                f"/api/v1/openclaw/sessions/{active_session.id}", {}, mock_http()
            )
            assert _status(result) == 500


# ============================================================================
# Session: End Session (POST SDK endpoint)
# ============================================================================


class TestEndSession:
    """Tests for _handle_end_session (POST /sessions/:id/end)."""

    def test_end_session_success(self, handler, mock_http, active_session, store):
        http = mock_http(body={}, method="POST")
        result = handler.handle_post(f"/api/v1/openclaw/sessions/{active_session.id}/end", {}, http)
        assert _status(result) == 200
        body = _body(result)
        assert body["success"] is True
        assert body["session_id"] == active_session.id
        session = store.get_session(active_session.id)
        assert session.status == SessionStatus.CLOSED

    def test_end_session_not_found(self, handler, mock_http):
        http = mock_http(body={}, method="POST")
        result = handler.handle_post("/api/v1/openclaw/sessions/nonexistent/end", {}, http)
        assert _status(result) == 404

    def test_end_session_other_user_denied(self, handler, mock_http, other_user_session):
        with patch(
            "aragora.server.handlers.openclaw.orchestrator._has_permission",
            return_value=False,
        ):
            http = mock_http(body={}, method="POST")
            result = handler.handle_post(
                f"/api/v1/openclaw/sessions/{other_user_session.id}/end", {}, http
            )
            assert _status(result) == 403

    def test_end_session_admin_can_end_any(self, handler, mock_http, other_user_session, store):
        with patch(
            "aragora.server.handlers.openclaw.orchestrator._has_permission",
            return_value=True,
        ):
            http = mock_http(body={}, method="POST")
            result = handler.handle_post(
                f"/api/v1/openclaw/sessions/{other_user_session.id}/end", {}, http
            )
            assert _status(result) == 200

    def test_end_session_audit_logged(self, handler, mock_http, active_session, store):
        http = mock_http(body={}, method="POST")
        handler.handle_post(f"/api/v1/openclaw/sessions/{active_session.id}/end", {}, http)
        entries, total = store.get_audit_log(action="session.end")
        assert total >= 1


# ============================================================================
# Action: Get Action
# ============================================================================


class TestGetAction:
    """Tests for _handle_get_action (GET /actions/:id)."""

    def test_get_action_success(self, handler, mock_http, pending_action):
        result = handler.handle(f"/api/v1/openclaw/actions/{pending_action.id}", {}, mock_http())
        assert _status(result) == 200
        body = _body(result)
        assert body["id"] == pending_action.id
        assert body["action_type"] == "code.execute"
        assert body["status"] == "pending"

    def test_get_action_not_found(self, handler, mock_http):
        result = handler.handle("/api/v1/openclaw/actions/nonexistent", {}, mock_http())
        assert _status(result) == 404

    def test_get_action_other_user_denied(self, handler, mock_http, store, other_user_session):
        """Non-admin cannot read actions from another user's session."""
        action = store.create_action(
            session_id=other_user_session.id,
            action_type="code.run",
            input_data={},
        )
        with patch(
            "aragora.server.handlers.openclaw.orchestrator._has_permission",
            return_value=False,
        ):
            result = handler.handle(f"/api/v1/openclaw/actions/{action.id}", {}, mock_http())
            assert _status(result) == 403

    def test_get_action_admin_can_read_any(self, handler, mock_http, store, other_user_session):
        action = store.create_action(
            session_id=other_user_session.id,
            action_type="code.run",
            input_data={},
        )
        with patch(
            "aragora.server.handlers.openclaw.orchestrator._has_permission",
            return_value=True,
        ):
            result = handler.handle(f"/api/v1/openclaw/actions/{action.id}", {}, mock_http())
            assert _status(result) == 200

    def test_get_action_store_error(self, handler, mock_http, pending_action):
        with patch(
            "aragora.server.handlers.openclaw.orchestrator._get_store",
        ) as mock_store:
            mock_store.return_value.get_action.side_effect = KeyError("gone")
            result = handler.handle(
                f"/api/v1/openclaw/actions/{pending_action.id}", {}, mock_http()
            )
            assert _status(result) == 500


# ============================================================================
# Action: Execute Action
# ============================================================================


class TestExecuteAction:
    """Tests for _handle_execute_action (POST /actions)."""

    def test_execute_action_success(self, handler, mock_http, active_session, store):
        http = mock_http(
            body={
                "session_id": active_session.id,
                "action_type": "code.execute",
                "input": {"code": "1+1"},
                "metadata": {"lang": "python"},
            },
            method="POST",
        )
        result = handler.handle_post("/api/v1/openclaw/actions", {}, http)
        assert _status(result) == 202
        body = _body(result)
        assert body["action_type"] == "code.execute"
        assert body["session_id"] == active_session.id
        # Action should be RUNNING (handler updates from PENDING to RUNNING)
        action = store.get_action(body["id"])
        assert action.status == ActionStatus.RUNNING

    def test_execute_action_missing_session_id(self, handler, mock_http):
        http = mock_http(
            body={"action_type": "code.execute", "input": {}},
            method="POST",
        )
        result = handler.handle_post("/api/v1/openclaw/actions", {}, http)
        assert _status(result) == 400
        body = _body(result)
        assert "session_id" in body.get("error", "").lower()

    def test_execute_action_missing_action_type(self, handler, mock_http, active_session):
        http = mock_http(
            body={"session_id": active_session.id, "input": {}},
            method="POST",
        )
        result = handler.handle_post("/api/v1/openclaw/actions", {}, http)
        assert _status(result) == 400

    def test_execute_action_invalid_action_type(self, handler, mock_http, active_session):
        http = mock_http(
            body={
                "session_id": active_session.id,
                "action_type": "!invalid;type",
                "input": {},
            },
            method="POST",
        )
        result = handler.handle_post("/api/v1/openclaw/actions", {}, http)
        assert _status(result) == 400

    def test_execute_action_session_not_found(self, handler, mock_http):
        http = mock_http(
            body={
                "session_id": "nonexistent",
                "action_type": "code.execute",
                "input": {},
            },
            method="POST",
        )
        result = handler.handle_post("/api/v1/openclaw/actions", {}, http)
        assert _status(result) == 404

    def test_execute_action_session_not_active(self, handler, mock_http, active_session, store):
        store.update_session_status(active_session.id, SessionStatus.CLOSED)
        http = mock_http(
            body={
                "session_id": active_session.id,
                "action_type": "code.execute",
                "input": {},
            },
            method="POST",
        )
        result = handler.handle_post("/api/v1/openclaw/actions", {}, http)
        assert _status(result) == 400
        body = _body(result)
        assert "not active" in body.get("error", "").lower()

    def test_execute_action_other_user_denied(self, handler, mock_http, other_user_session):
        with patch(
            "aragora.server.handlers.openclaw.orchestrator._has_permission",
            return_value=False,
        ):
            http = mock_http(
                body={
                    "session_id": other_user_session.id,
                    "action_type": "code.execute",
                    "input": {},
                },
                method="POST",
            )
            result = handler.handle_post("/api/v1/openclaw/actions", {}, http)
            assert _status(result) == 403

    def test_execute_action_admin_can_execute_any(self, handler, mock_http, other_user_session):
        with patch(
            "aragora.server.handlers.openclaw.orchestrator._has_permission",
            return_value=True,
        ):
            http = mock_http(
                body={
                    "session_id": other_user_session.id,
                    "action_type": "code.execute",
                    "input": {},
                },
                method="POST",
            )
            result = handler.handle_post("/api/v1/openclaw/actions", {}, http)
            assert _status(result) == 202

    def test_execute_action_sanitizes_input(self, handler, mock_http, active_session, store):
        """Shell metacharacters in input are escaped."""
        http = mock_http(
            body={
                "session_id": active_session.id,
                "action_type": "code.execute",
                "input": {"cmd": "ls; rm -rf /"},
            },
            method="POST",
        )
        result = handler.handle_post("/api/v1/openclaw/actions", {}, http)
        assert _status(result) == 202
        body = _body(result)
        action = store.get_action(body["id"])
        # The semicolon should be escaped with a backslash
        cmd = action.input_data.get("cmd", "")
        assert cmd != "ls; rm -rf /"  # Must differ from raw input
        assert "\\;" in cmd  # Semicolon escaped

    def test_execute_action_invalid_input(self, handler, mock_http, active_session):
        http = mock_http(
            body={
                "session_id": active_session.id,
                "action_type": "code.execute",
                "input": "not-a-dict",
            },
            method="POST",
        )
        result = handler.handle_post("/api/v1/openclaw/actions", {}, http)
        assert _status(result) == 400

    def test_execute_action_invalid_metadata(self, handler, mock_http, active_session):
        http = mock_http(
            body={
                "session_id": active_session.id,
                "action_type": "code.execute",
                "input": {},
                "metadata": "not-a-dict",
            },
            method="POST",
        )
        result = handler.handle_post("/api/v1/openclaw/actions", {}, http)
        assert _status(result) == 400

    def test_execute_action_audit_logged(self, handler, mock_http, active_session, store):
        http = mock_http(
            body={
                "session_id": active_session.id,
                "action_type": "code.execute",
                "input": {},
            },
            method="POST",
        )
        handler.handle_post("/api/v1/openclaw/actions", {}, http)
        entries, total = store.get_audit_log(action="action.execute")
        assert total >= 1
        assert entries[0].result == "pending"
        assert entries[0].details["action_type"] == "code.execute"

    def test_execute_action_store_error(self, handler, mock_http, active_session):
        with patch(
            "aragora.server.handlers.openclaw.orchestrator._get_store",
        ) as mock_store:
            mock_store.return_value.get_session.side_effect = ValueError("bad data")
            http = mock_http(
                body={
                    "session_id": active_session.id,
                    "action_type": "code.execute",
                    "input": {},
                },
                method="POST",
            )
            result = handler.handle_post("/api/v1/openclaw/actions", {}, http)
            assert _status(result) == 500


# ============================================================================
# Action: Cancel Action
# ============================================================================


class TestCancelAction:
    """Tests for _handle_cancel_action (POST /actions/:id/cancel)."""

    def test_cancel_pending_action(self, handler, mock_http, pending_action, store):
        http = mock_http(body={}, method="POST")
        result = handler.handle_post(
            f"/api/v1/openclaw/actions/{pending_action.id}/cancel", {}, http
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["cancelled"] is True
        assert body["action_id"] == pending_action.id
        action = store.get_action(pending_action.id)
        assert action.status == ActionStatus.CANCELLED

    def test_cancel_running_action(self, handler, mock_http, running_action, store):
        http = mock_http(body={}, method="POST")
        result = handler.handle_post(
            f"/api/v1/openclaw/actions/{running_action.id}/cancel", {}, http
        )
        assert _status(result) == 200
        action = store.get_action(running_action.id)
        assert action.status == ActionStatus.CANCELLED

    def test_cancel_action_not_found(self, handler, mock_http):
        http = mock_http(body={}, method="POST")
        result = handler.handle_post("/api/v1/openclaw/actions/nonexistent/cancel", {}, http)
        assert _status(result) == 404

    def test_cancel_completed_action_rejected(self, handler, mock_http, pending_action, store):
        """Cannot cancel a completed action."""
        store.update_action(pending_action.id, status=ActionStatus.COMPLETED)
        http = mock_http(body={}, method="POST")
        result = handler.handle_post(
            f"/api/v1/openclaw/actions/{pending_action.id}/cancel", {}, http
        )
        assert _status(result) == 400
        body = _body(result)
        assert "cannot be cancelled" in body.get("error", "").lower()

    def test_cancel_failed_action_rejected(self, handler, mock_http, pending_action, store):
        """Cannot cancel a failed action."""
        store.update_action(pending_action.id, status=ActionStatus.FAILED)
        http = mock_http(body={}, method="POST")
        result = handler.handle_post(
            f"/api/v1/openclaw/actions/{pending_action.id}/cancel", {}, http
        )
        assert _status(result) == 400

    def test_cancel_already_cancelled_action_rejected(
        self, handler, mock_http, pending_action, store
    ):
        store.update_action(pending_action.id, status=ActionStatus.CANCELLED)
        http = mock_http(body={}, method="POST")
        result = handler.handle_post(
            f"/api/v1/openclaw/actions/{pending_action.id}/cancel", {}, http
        )
        assert _status(result) == 400

    def test_cancel_action_other_user_denied(self, handler, mock_http, store, other_user_session):
        action = store.create_action(
            session_id=other_user_session.id,
            action_type="code.run",
            input_data={},
        )
        with patch(
            "aragora.server.handlers.openclaw.orchestrator._has_permission",
            return_value=False,
        ):
            http = mock_http(body={}, method="POST")
            result = handler.handle_post(f"/api/v1/openclaw/actions/{action.id}/cancel", {}, http)
            assert _status(result) == 403

    def test_cancel_action_admin_can_cancel_any(
        self, handler, mock_http, store, other_user_session
    ):
        action = store.create_action(
            session_id=other_user_session.id,
            action_type="code.run",
            input_data={},
        )
        with patch(
            "aragora.server.handlers.openclaw.orchestrator._has_permission",
            return_value=True,
        ):
            http = mock_http(body={}, method="POST")
            result = handler.handle_post(f"/api/v1/openclaw/actions/{action.id}/cancel", {}, http)
            assert _status(result) == 200

    def test_cancel_action_audit_logged(self, handler, mock_http, pending_action, store):
        http = mock_http(body={}, method="POST")
        handler.handle_post(f"/api/v1/openclaw/actions/{pending_action.id}/cancel", {}, http)
        entries, total = store.get_audit_log(action="action.cancel")
        assert total >= 1
        assert entries[0].result == "success"

    def test_cancel_action_store_error(self, handler, mock_http, pending_action):
        with patch(
            "aragora.server.handlers.openclaw.orchestrator._get_store",
        ) as mock_store:
            mock_store.return_value.get_action.side_effect = OSError("timeout")
            http = mock_http(body={}, method="POST")
            result = handler.handle_post(
                f"/api/v1/openclaw/actions/{pending_action.id}/cancel", {}, http
            )
            assert _status(result) == 500


# ============================================================================
# Routing: Path Normalization
# ============================================================================


class TestRouting:
    """Verify path normalization across all supported prefixes."""

    def test_api_v1_openclaw_prefix(self, handler, mock_http, active_session):
        result = handler.handle(f"/api/v1/openclaw/sessions/{active_session.id}", {}, mock_http())
        assert _status(result) == 200

    def test_api_gateway_openclaw_prefix(self, handler, mock_http, active_session):
        result = handler.handle(
            f"/api/gateway/openclaw/sessions/{active_session.id}", {}, mock_http()
        )
        assert _status(result) == 200

    def test_api_v1_gateway_openclaw_prefix(self, handler, mock_http, active_session):
        result = handler.handle(
            f"/api/v1/gateway/openclaw/sessions/{active_session.id}", {}, mock_http()
        )
        assert _status(result) == 200

    def test_api_openclaw_prefix(self, handler, mock_http, active_session):
        result = handler.handle(f"/api/openclaw/sessions/{active_session.id}", {}, mock_http())
        assert _status(result) == 200

    def test_can_handle_matches_openclaw_paths(self, handler):
        assert handler.can_handle("/api/v1/openclaw/sessions") is True
        assert handler.can_handle("/api/gateway/openclaw/actions") is True
        assert handler.can_handle("/api/v1/gateway/openclaw/health") is True
        assert handler.can_handle("/api/openclaw/metrics") is True

    def test_can_handle_rejects_non_openclaw(self, handler):
        assert handler.can_handle("/api/v1/debates") is False
        assert handler.can_handle("/api/v1/agents") is False

    def test_unknown_get_path_returns_none(self, handler, mock_http):
        result = handler.handle("/api/v1/openclaw/unknown-endpoint", {}, mock_http())
        assert result is None

    def test_unknown_post_path_returns_none(self, handler, mock_http):
        http = mock_http(body={}, method="POST")
        result = handler.handle_post("/api/v1/openclaw/unknown-endpoint", {}, http)
        assert result is None


# ============================================================================
# End-to-end: Full session lifecycle
# ============================================================================


class TestSessionLifecycle:
    """End-to-end test covering session create -> execute action -> cancel -> close."""

    def test_full_lifecycle(self, handler, mock_http, store):
        # 1. Create session
        http = mock_http(body={"config": {"timeout": 120}}, method="POST")
        result = handler.handle_post("/api/v1/openclaw/sessions", {}, http)
        assert _status(result) == 201
        session_id = _body(result)["id"]

        # 2. Verify session is listed
        result = handler.handle("/api/v1/openclaw/sessions", {}, mock_http())
        assert _status(result) == 200
        ids = [s["id"] for s in _body(result)["sessions"]]
        assert session_id in ids

        # 3. Execute action
        http = mock_http(
            body={
                "session_id": session_id,
                "action_type": "code.execute",
                "input": {"code": "2+2"},
            },
            method="POST",
        )
        result = handler.handle_post("/api/v1/openclaw/actions", {}, http)
        assert _status(result) == 202
        action_id = _body(result)["id"]

        # 4. Get action status
        result = handler.handle(f"/api/v1/openclaw/actions/{action_id}", {}, mock_http())
        assert _status(result) == 200
        assert _body(result)["status"] == "running"

        # 5. Create another action and cancel it
        http = mock_http(
            body={
                "session_id": session_id,
                "action_type": "code.lint",
                "input": {},
            },
            method="POST",
        )
        result = handler.handle_post("/api/v1/openclaw/actions", {}, http)
        assert _status(result) == 202
        action_id_2 = _body(result)["id"]

        http = mock_http(body={}, method="POST")
        result = handler.handle_post(f"/api/v1/openclaw/actions/{action_id_2}/cancel", {}, http)
        assert _status(result) == 200
        assert _body(result)["cancelled"] is True

        # 6. Close session
        result = handler.handle_delete(f"/api/v1/openclaw/sessions/{session_id}", {}, mock_http())
        assert _status(result) == 200
        assert _body(result)["closed"] is True

        # 7. Verify audit log has entries for all operations
        entries, total = store.get_audit_log()
        actions_logged = {e.action for e in entries}
        assert "session.create" in actions_logged
        assert "action.execute" in actions_logged
        assert "action.cancel" in actions_logged
        assert "session.close" in actions_logged
