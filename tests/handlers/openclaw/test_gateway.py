"""Comprehensive tests for OpenClawGatewayHandler routing and dispatch.

Tests the top-level gateway handler at
aragora/server/handlers/openclaw/gateway.py (358 lines).

This file exercises the routing layer that:
- Normalizes versioned/shorthand paths to base form
- Dispatches GET/POST/DELETE requests to the correct mixin methods
- Returns None for unrecognized paths

Test categories:
- can_handle: path prefix recognition
- _normalize_path: path rewriting
- GET routing: sessions, actions, credentials, health, metrics, audit, policy, approvals, stats
- POST routing: sessions, actions, credentials, policy rules, approvals, cancel, end, rotate
- DELETE routing: sessions, credentials, policy rules
- Path normalization across all 4 prefix variants
- Error handling on store failures
- Edge cases (trailing slashes, unknown paths, empty bodies)
- Circuit breaker helpers
- Handler registration factory
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.openclaw.gateway import (
    OpenClawGatewayHandler,
    get_openclaw_circuit_breaker,
    get_openclaw_circuit_breaker_status,
    get_openclaw_gateway_handler,
)
from aragora.server.handlers.openclaw.models import (
    Action,
    ActionStatus,
    Credential,
    CredentialType,
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


class _MockUser:
    """Minimal mock user for get_current_user."""

    def __init__(self, user_id="test-user-001", org_id="test-org-001", role="admin"):
        self.user_id = user_id
        self.org_id = org_id
        self.role = role


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture()
def store():
    """Fresh in-memory OpenClaw store for each test."""
    return OpenClawGatewayStore()


@pytest.fixture()
def mock_user():
    """Default mock user returned by get_current_user."""
    return _MockUser()


@pytest.fixture()
def handler(store, mock_user):
    """OpenClawGatewayHandler with _get_store and get_current_user patched."""
    with patch(
        "aragora.server.handlers.openclaw.orchestrator._get_store",
        return_value=store,
    ), patch(
        "aragora.server.handlers.openclaw.credentials._get_store",
        return_value=store,
    ), patch(
        "aragora.server.handlers.openclaw.policies._get_store",
        return_value=store,
    ):
        h = OpenClawGatewayHandler(server_context={})
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


@pytest.fixture()
def credential(store) -> Credential:
    """Pre-create a credential owned by 'test-user-001'."""
    return store.store_credential(
        name="test_api_key",
        credential_type=CredentialType.API_KEY,
        secret_value="sk-test-1234567890abcdef",
        user_id="test-user-001",
        tenant_id="test-org-001",
    )


@pytest.fixture()
def other_user_credential(store) -> Credential:
    """Pre-create a credential owned by a different user."""
    return store.store_credential(
        name="other_key",
        credential_type=CredentialType.API_KEY,
        secret_value="sk-other-1234567890abcdef",
        user_id="other-user-999",
        tenant_id="other-org",
    )


# ============================================================================
# can_handle
# ============================================================================


class TestCanHandle:
    """Tests for can_handle path prefix matching."""

    def test_gateway_openclaw_prefix(self, handler):
        assert handler.can_handle("/api/gateway/openclaw/sessions") is True

    def test_v1_gateway_openclaw_prefix(self, handler):
        assert handler.can_handle("/api/v1/gateway/openclaw/sessions") is True

    def test_v1_openclaw_prefix(self, handler):
        assert handler.can_handle("/api/v1/openclaw/sessions") is True

    def test_openclaw_prefix(self, handler):
        assert handler.can_handle("/api/openclaw/sessions") is True

    def test_unrelated_path_rejected(self, handler):
        assert handler.can_handle("/api/v1/debates") is False

    def test_partial_match_rejected(self, handler):
        assert handler.can_handle("/api/gateway/other/sessions") is False

    def test_root_path_rejected(self, handler):
        assert handler.can_handle("/") is False


# ============================================================================
# _normalize_path
# ============================================================================


class TestNormalizePath:
    """Tests for path normalization."""

    def test_v1_gateway_openclaw_normalized(self, handler):
        result = handler._normalize_path("/api/v1/gateway/openclaw/sessions")
        assert result == "/api/gateway/openclaw/sessions"

    def test_v1_openclaw_normalized(self, handler):
        result = handler._normalize_path("/api/v1/openclaw/sessions")
        assert result == "/api/gateway/openclaw/sessions"

    def test_api_openclaw_normalized(self, handler):
        result = handler._normalize_path("/api/openclaw/sessions")
        assert result == "/api/gateway/openclaw/sessions"

    def test_base_path_unchanged(self, handler):
        result = handler._normalize_path("/api/gateway/openclaw/sessions")
        assert result == "/api/gateway/openclaw/sessions"

    def test_unrelated_path_unchanged(self, handler):
        result = handler._normalize_path("/api/v1/debates")
        assert result == "/api/v1/debates"

    def test_deep_path_normalized(self, handler):
        result = handler._normalize_path("/api/v1/openclaw/actions/abc/cancel")
        assert result == "/api/gateway/openclaw/actions/abc/cancel"


# ============================================================================
# GET Routing: Sessions
# ============================================================================


class TestGetListSessions:
    """Tests for GET /sessions via handler.handle."""

    def test_list_sessions_via_v1_openclaw(self, handler, mock_http):
        result = handler.handle("/api/v1/openclaw/sessions", {}, mock_http())
        assert _status(result) == 200
        body = _body(result)
        assert "sessions" in body
        assert body["total"] == 0

    def test_list_sessions_via_gateway_path(self, handler, mock_http):
        result = handler.handle("/api/gateway/openclaw/sessions", {}, mock_http())
        assert _status(result) == 200

    def test_list_sessions_via_v1_gateway(self, handler, mock_http):
        result = handler.handle("/api/v1/gateway/openclaw/sessions", {}, mock_http())
        assert _status(result) == 200

    def test_list_sessions_via_api_openclaw(self, handler, mock_http):
        result = handler.handle("/api/openclaw/sessions", {}, mock_http())
        assert _status(result) == 200

    def test_list_sessions_with_data(self, handler, mock_http, active_session):
        result = handler.handle("/api/v1/openclaw/sessions", {}, mock_http())
        assert _status(result) == 200
        body = _body(result)
        assert body["total"] >= 1

    def test_list_sessions_with_status_filter(self, handler, mock_http, store, active_session):
        store.update_session_status(active_session.id, SessionStatus.CLOSED)
        result = handler.handle(
            "/api/v1/openclaw/sessions", {"status": "closed"}, mock_http()
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["total"] >= 1

    def test_list_sessions_with_pagination(self, handler, mock_http, active_session):
        result = handler.handle(
            "/api/v1/openclaw/sessions", {"limit": "10", "offset": "0"}, mock_http()
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["limit"] == 10
        assert body["offset"] == 0


class TestGetSession:
    """Tests for GET /sessions/:id."""

    def test_get_session_found(self, handler, mock_http, active_session):
        path = f"/api/v1/openclaw/sessions/{active_session.id}"
        result = handler.handle(path, {}, mock_http())
        assert _status(result) == 200
        body = _body(result)
        assert body["id"] == active_session.id

    def test_get_session_not_found(self, handler, mock_http):
        result = handler.handle("/api/v1/openclaw/sessions/nonexistent", {}, mock_http())
        assert _status(result) == 404

    def test_get_session_access_denied_other_user(self, handler, mock_http, other_user_session):
        """Non-admin user cannot access another user's session."""
        # Make the user non-admin
        handler.get_current_user = lambda h: _MockUser(role="viewer")
        path = f"/api/v1/openclaw/sessions/{other_user_session.id}"
        result = handler.handle(path, {}, mock_http())
        assert _status(result) == 403

    def test_get_session_via_gateway_path(self, handler, mock_http, active_session):
        path = f"/api/gateway/openclaw/sessions/{active_session.id}"
        result = handler.handle(path, {}, mock_http())
        assert _status(result) == 200


# ============================================================================
# GET Routing: Actions
# ============================================================================


class TestGetAction:
    """Tests for GET /actions/:id."""

    def test_get_action_found(self, handler, mock_http, pending_action):
        path = f"/api/v1/openclaw/actions/{pending_action.id}"
        result = handler.handle(path, {}, mock_http())
        assert _status(result) == 200
        body = _body(result)
        assert body["id"] == pending_action.id

    def test_get_action_not_found(self, handler, mock_http):
        result = handler.handle("/api/v1/openclaw/actions/nonexistent", {}, mock_http())
        assert _status(result) == 404

    def test_get_action_access_denied(self, handler, mock_http, store, other_user_session):
        action = store.create_action(
            session_id=other_user_session.id,
            action_type="code.execute",
            input_data={"code": "x"},
        )
        handler.get_current_user = lambda h: _MockUser(role="viewer")
        path = f"/api/v1/openclaw/actions/{action.id}"
        result = handler.handle(path, {}, mock_http())
        assert _status(result) == 403


# ============================================================================
# GET Routing: Credentials
# ============================================================================


class TestGetListCredentials:
    """Tests for GET /credentials."""

    def test_list_credentials_empty(self, handler, mock_http):
        result = handler.handle("/api/v1/openclaw/credentials", {}, mock_http())
        assert _status(result) == 200
        body = _body(result)
        assert body["credentials"] == []
        assert body["total"] == 0

    def test_list_credentials_with_data(self, handler, mock_http, credential):
        result = handler.handle("/api/v1/openclaw/credentials", {}, mock_http())
        assert _status(result) == 200
        body = _body(result)
        assert body["total"] >= 1

    def test_list_credentials_type_filter(self, handler, mock_http, credential):
        result = handler.handle(
            "/api/v1/openclaw/credentials", {"type": "api_key"}, mock_http()
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["total"] >= 1


# ============================================================================
# GET Routing: Admin
# ============================================================================


class TestGetHealth:
    """Tests for GET /health."""

    def test_health_healthy(self, handler, mock_http):
        result = handler.handle("/api/v1/openclaw/health", {}, mock_http())
        assert _status(result) == 200
        body = _body(result)
        assert body["status"] == "healthy"
        assert body["healthy"] is True
        assert "timestamp" in body

    def test_health_via_gateway_path(self, handler, mock_http):
        result = handler.handle("/api/gateway/openclaw/health", {}, mock_http())
        assert _status(result) == 200

    def test_health_degraded_many_running(self, handler, mock_http, store, active_session):
        """Health degrades when >100 running actions."""
        for i in range(101):
            a = store.create_action(
                session_id=active_session.id,
                action_type="code.execute",
                input_data={"i": i},
            )
            store.update_action(a.id, status=ActionStatus.RUNNING)
        result = handler.handle("/api/v1/openclaw/health", {}, mock_http())
        assert _status(result) == 200
        body = _body(result)
        assert body["status"] == "degraded"


class TestGetMetrics:
    """Tests for GET /metrics."""

    def test_metrics(self, handler, mock_http):
        result = handler.handle("/api/v1/openclaw/metrics", {}, mock_http())
        assert _status(result) == 200
        body = _body(result)
        assert "sessions" in body
        assert "actions" in body
        assert "timestamp" in body

    def test_metrics_via_gateway_path(self, handler, mock_http):
        result = handler.handle("/api/gateway/openclaw/metrics", {}, mock_http())
        assert _status(result) == 200


class TestGetAudit:
    """Tests for GET /audit."""

    def test_audit_empty(self, handler, mock_http):
        result = handler.handle("/api/v1/openclaw/audit", {}, mock_http())
        assert _status(result) == 200
        body = _body(result)
        assert body["entries"] == []
        assert body["total"] == 0

    def test_audit_with_entries(self, handler, mock_http, store):
        store.add_audit_entry(
            action="session.create",
            actor_id="test-user-001",
            resource_type="session",
            resource_id="sess-1",
            result="success",
        )
        result = handler.handle("/api/v1/openclaw/audit", {}, mock_http())
        assert _status(result) == 200
        body = _body(result)
        assert body["total"] >= 1

    def test_audit_with_action_filter(self, handler, mock_http, store):
        store.add_audit_entry(
            action="session.create", actor_id="u1", resource_type="session"
        )
        store.add_audit_entry(
            action="credential.delete", actor_id="u1", resource_type="credential"
        )
        result = handler.handle(
            "/api/v1/openclaw/audit", {"action": "session.create"}, mock_http()
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 1

    def test_audit_via_gateway_path(self, handler, mock_http):
        result = handler.handle("/api/gateway/openclaw/audit", {}, mock_http())
        assert _status(result) == 200


class TestGetPolicyRules:
    """Tests for GET /policy/rules."""

    def test_get_policy_rules(self, handler, mock_http):
        result = handler.handle("/api/v1/openclaw/policy/rules", {}, mock_http())
        assert _status(result) == 200
        body = _body(result)
        assert "rules" in body
        assert body["total"] == 0


class TestGetApprovals:
    """Tests for GET /approvals."""

    def test_list_approvals_empty(self, handler, mock_http):
        result = handler.handle("/api/v1/openclaw/approvals", {}, mock_http())
        assert _status(result) == 200
        body = _body(result)
        assert body["approvals"] == []
        assert body["total"] == 0


class TestGetStats:
    """Tests for GET /stats."""

    def test_stats(self, handler, mock_http):
        result = handler.handle("/api/v1/openclaw/stats", {}, mock_http())
        assert _status(result) == 200
        body = _body(result)
        assert "active_sessions" in body
        assert "actions_allowed" in body
        assert "timestamp" in body

    def test_stats_with_data(self, handler, mock_http, active_session):
        result = handler.handle("/api/v1/openclaw/stats", {}, mock_http())
        assert _status(result) == 200
        body = _body(result)
        assert body["active_sessions"] >= 1


# ============================================================================
# GET Routing: Unknown paths
# ============================================================================


class TestGetUnknownPath:
    """Tests for GET on unrecognized paths."""

    def test_unknown_path_returns_none(self, handler, mock_http):
        result = handler.handle("/api/v1/openclaw/unknown", {}, mock_http())
        assert result is None

    def test_deep_unknown_path_returns_none(self, handler, mock_http):
        result = handler.handle("/api/v1/openclaw/foo/bar/baz", {}, mock_http())
        assert result is None


# ============================================================================
# POST Routing: Sessions
# ============================================================================


class TestPostCreateSession:
    """Tests for POST /sessions."""

    def test_create_session(self, handler, mock_http):
        body = {"config": {"timeout": 600}, "metadata": {"source": "test"}}
        result = handler.handle_post(
            "/api/v1/openclaw/sessions", {}, mock_http(body=body, method="POST")
        )
        assert _status(result) == 201
        resp = _body(result)
        assert "id" in resp
        assert resp["status"] == "active"

    def test_create_session_minimal(self, handler, mock_http):
        result = handler.handle_post(
            "/api/v1/openclaw/sessions", {}, mock_http(body={}, method="POST")
        )
        assert _status(result) == 201

    def test_create_session_invalid_config(self, handler, mock_http):
        body = {"config": "not_a_dict"}
        result = handler.handle_post(
            "/api/v1/openclaw/sessions", {}, mock_http(body=body, method="POST")
        )
        assert _status(result) == 400

    def test_create_session_via_gateway_path(self, handler, mock_http):
        result = handler.handle_post(
            "/api/gateway/openclaw/sessions", {}, mock_http(body={}, method="POST")
        )
        assert _status(result) == 201


class TestPostEndSession:
    """Tests for POST /sessions/:id/end."""

    def test_end_session(self, handler, mock_http, active_session):
        path = f"/api/v1/openclaw/sessions/{active_session.id}/end"
        result = handler.handle_post(
            path, {}, mock_http(body={}, method="POST")
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["success"] is True

    def test_end_session_not_found(self, handler, mock_http):
        result = handler.handle_post(
            "/api/v1/openclaw/sessions/nonexistent/end",
            {},
            mock_http(body={}, method="POST"),
        )
        assert _status(result) == 404

    def test_end_session_access_denied(self, handler, mock_http, other_user_session):
        handler.get_current_user = lambda h: _MockUser(role="viewer")
        path = f"/api/v1/openclaw/sessions/{other_user_session.id}/end"
        result = handler.handle_post(path, {}, mock_http(body={}, method="POST"))
        assert _status(result) == 403


# ============================================================================
# POST Routing: Actions
# ============================================================================


class TestPostExecuteAction:
    """Tests for POST /actions."""

    def test_execute_action(self, handler, mock_http, active_session):
        body = {
            "session_id": active_session.id,
            "action_type": "code.execute",
            "input": {"code": "print('hello')"},
        }
        result = handler.handle_post(
            "/api/v1/openclaw/actions", {}, mock_http(body=body, method="POST")
        )
        assert _status(result) == 202
        resp = _body(result)
        assert "id" in resp
        assert resp["action_type"] == "code.execute"

    def test_execute_action_missing_session_id(self, handler, mock_http):
        body = {"action_type": "code.execute", "input": {}}
        result = handler.handle_post(
            "/api/v1/openclaw/actions", {}, mock_http(body=body, method="POST")
        )
        assert _status(result) == 400

    def test_execute_action_missing_action_type(self, handler, mock_http, active_session):
        body = {"session_id": active_session.id, "input": {}}
        result = handler.handle_post(
            "/api/v1/openclaw/actions", {}, mock_http(body=body, method="POST")
        )
        assert _status(result) == 400

    def test_execute_action_session_not_found(self, handler, mock_http):
        body = {"session_id": "nonexistent", "action_type": "code.execute"}
        result = handler.handle_post(
            "/api/v1/openclaw/actions", {}, mock_http(body=body, method="POST")
        )
        assert _status(result) == 404

    def test_execute_action_session_not_active(self, handler, mock_http, store, active_session):
        store.update_session_status(active_session.id, SessionStatus.CLOSED)
        body = {
            "session_id": active_session.id,
            "action_type": "code.execute",
            "input": {},
        }
        result = handler.handle_post(
            "/api/v1/openclaw/actions", {}, mock_http(body=body, method="POST")
        )
        assert _status(result) == 400

    def test_execute_action_access_denied(self, handler, mock_http, other_user_session):
        handler.get_current_user = lambda h: _MockUser(role="viewer")
        body = {
            "session_id": other_user_session.id,
            "action_type": "code.execute",
            "input": {},
        }
        result = handler.handle_post(
            "/api/v1/openclaw/actions", {}, mock_http(body=body, method="POST")
        )
        assert _status(result) == 403


class TestPostCancelAction:
    """Tests for POST /actions/:id/cancel."""

    def test_cancel_running_action(self, handler, mock_http, running_action):
        path = f"/api/v1/openclaw/actions/{running_action.id}/cancel"
        result = handler.handle_post(path, {}, mock_http(body={}, method="POST"))
        assert _status(result) == 200
        body = _body(result)
        assert body["cancelled"] is True

    def test_cancel_pending_action(self, handler, mock_http, pending_action):
        path = f"/api/v1/openclaw/actions/{pending_action.id}/cancel"
        result = handler.handle_post(path, {}, mock_http(body={}, method="POST"))
        assert _status(result) == 200

    def test_cancel_action_not_found(self, handler, mock_http):
        result = handler.handle_post(
            "/api/v1/openclaw/actions/nonexistent/cancel",
            {},
            mock_http(body={}, method="POST"),
        )
        assert _status(result) == 404

    def test_cancel_completed_action_rejected(self, handler, mock_http, store, active_session):
        action = store.create_action(
            session_id=active_session.id,
            action_type="code.execute",
            input_data={"code": "x"},
        )
        store.update_action(action.id, status=ActionStatus.COMPLETED)
        path = f"/api/v1/openclaw/actions/{action.id}/cancel"
        result = handler.handle_post(path, {}, mock_http(body={}, method="POST"))
        assert _status(result) == 400

    def test_cancel_action_access_denied(
        self, handler, mock_http, store, other_user_session
    ):
        action = store.create_action(
            session_id=other_user_session.id,
            action_type="code.execute",
            input_data={"code": "x"},
        )
        store.update_action(action.id, status=ActionStatus.RUNNING)
        handler.get_current_user = lambda h: _MockUser(role="viewer")
        path = f"/api/v1/openclaw/actions/{action.id}/cancel"
        result = handler.handle_post(path, {}, mock_http(body={}, method="POST"))
        assert _status(result) == 403


# ============================================================================
# POST Routing: Credentials
# ============================================================================


class TestPostStoreCredential:
    """Tests for POST /credentials."""

    def test_store_credential(self, handler, mock_http):
        body = {
            "name": "my_api_key",
            "type": "api_key",
            "secret": "sk-test-1234567890abcdef",
        }
        result = handler.handle_post(
            "/api/v1/openclaw/credentials", {}, mock_http(body=body, method="POST")
        )
        assert _status(result) == 201
        resp = _body(result)
        assert resp["name"] == "my_api_key"
        assert resp["credential_type"] == "api_key"

    def test_store_credential_missing_name(self, handler, mock_http):
        body = {"type": "api_key", "secret": "sk-test-1234567890abcdef"}
        result = handler.handle_post(
            "/api/v1/openclaw/credentials", {}, mock_http(body=body, method="POST")
        )
        assert _status(result) == 400

    def test_store_credential_missing_type(self, handler, mock_http):
        body = {"name": "my_key", "secret": "sk-test-1234567890abcdef"}
        result = handler.handle_post(
            "/api/v1/openclaw/credentials", {}, mock_http(body=body, method="POST")
        )
        assert _status(result) == 400

    def test_store_credential_invalid_type(self, handler, mock_http):
        body = {
            "name": "my_key",
            "type": "invalid_type",
            "secret": "sk-test-1234567890abcdef",
        }
        result = handler.handle_post(
            "/api/v1/openclaw/credentials", {}, mock_http(body=body, method="POST")
        )
        assert _status(result) == 400

    def test_store_credential_missing_secret(self, handler, mock_http):
        body = {"name": "my_key", "type": "api_key"}
        result = handler.handle_post(
            "/api/v1/openclaw/credentials", {}, mock_http(body=body, method="POST")
        )
        assert _status(result) == 400

    def test_store_credential_with_expiration(self, handler, mock_http):
        body = {
            "name": "expiring_key",
            "type": "api_key",
            "secret": "sk-test-1234567890abcdef",
            "expires_at": "2030-12-31T23:59:59+00:00",
        }
        result = handler.handle_post(
            "/api/v1/openclaw/credentials", {}, mock_http(body=body, method="POST")
        )
        assert _status(result) == 201

    def test_store_credential_invalid_expiration(self, handler, mock_http):
        body = {
            "name": "bad_expiry",
            "type": "api_key",
            "secret": "sk-test-1234567890abcdef",
            "expires_at": "not-a-date",
        }
        result = handler.handle_post(
            "/api/v1/openclaw/credentials", {}, mock_http(body=body, method="POST")
        )
        assert _status(result) == 400


class TestPostRotateCredential:
    """Tests for POST /credentials/:id/rotate."""

    def test_rotate_credential(self, handler, mock_http, credential):
        path = f"/api/v1/openclaw/credentials/{credential.id}/rotate"
        body = {"secret": "new-secret-value-123456"}
        result = handler.handle_post(path, {}, mock_http(body=body, method="POST"))
        assert _status(result) == 200
        resp = _body(result)
        assert resp["rotated"] is True
        assert resp["credential_id"] == credential.id

    def test_rotate_credential_not_found(self, handler, mock_http):
        body = {"secret": "new-secret-value-123456"}
        result = handler.handle_post(
            "/api/v1/openclaw/credentials/nonexistent/rotate",
            {},
            mock_http(body=body, method="POST"),
        )
        assert _status(result) == 404

    def test_rotate_credential_access_denied(
        self, handler, mock_http, other_user_credential
    ):
        handler.get_current_user = lambda h: _MockUser(role="viewer")
        path = f"/api/v1/openclaw/credentials/{other_user_credential.id}/rotate"
        body = {"secret": "new-secret-value-123456"}
        result = handler.handle_post(path, {}, mock_http(body=body, method="POST"))
        assert _status(result) == 403

    def test_rotate_credential_invalid_secret(self, handler, mock_http, credential):
        path = f"/api/v1/openclaw/credentials/{credential.id}/rotate"
        body = {"secret": ""}  # empty secret
        result = handler.handle_post(path, {}, mock_http(body=body, method="POST"))
        assert _status(result) == 400


# ============================================================================
# POST Routing: Policy Rules
# ============================================================================


class TestPostAddPolicyRule:
    """Tests for POST /policy/rules."""

    def test_add_policy_rule(self, handler, mock_http):
        body = {
            "name": "block_destructive",
            "action_types": ["file.delete", "db.drop"],
            "decision": "deny",
            "priority": 10,
        }
        result = handler.handle_post(
            "/api/v1/openclaw/policy/rules", {}, mock_http(body=body, method="POST")
        )
        assert _status(result) == 201
        resp = _body(result)
        assert resp["name"] == "block_destructive"

    def test_add_policy_rule_missing_name(self, handler, mock_http):
        body = {"action_types": ["file.delete"], "decision": "deny"}
        result = handler.handle_post(
            "/api/v1/openclaw/policy/rules", {}, mock_http(body=body, method="POST")
        )
        assert _status(result) == 400


# ============================================================================
# POST Routing: Approvals
# ============================================================================


class TestPostApproveAction:
    """Tests for POST /approvals/:id/approve."""

    def test_approve_action(self, handler, mock_http):
        body = {"reason": "Looks safe"}
        result = handler.handle_post(
            "/api/v1/openclaw/approvals/approval-1/approve",
            {},
            mock_http(body=body, method="POST"),
        )
        assert _status(result) == 200
        resp = _body(result)
        assert resp["success"] is True
        assert resp["approval_id"] == "approval-1"

    def test_approve_action_no_reason(self, handler, mock_http):
        result = handler.handle_post(
            "/api/v1/openclaw/approvals/approval-2/approve",
            {},
            mock_http(body={}, method="POST"),
        )
        assert _status(result) == 200


class TestPostDenyAction:
    """Tests for POST /approvals/:id/deny."""

    def test_deny_action(self, handler, mock_http):
        body = {"reason": "Too risky"}
        result = handler.handle_post(
            "/api/v1/openclaw/approvals/approval-1/deny",
            {},
            mock_http(body=body, method="POST"),
        )
        assert _status(result) == 200
        resp = _body(result)
        assert resp["success"] is True

    def test_deny_action_no_reason(self, handler, mock_http):
        result = handler.handle_post(
            "/api/v1/openclaw/approvals/approval-2/deny",
            {},
            mock_http(body={}, method="POST"),
        )
        assert _status(result) == 200


# ============================================================================
# POST Routing: Unknown paths
# ============================================================================


class TestPostUnknownPath:
    """Tests for POST on unrecognized paths."""

    def test_unknown_post_returns_none(self, handler, mock_http):
        result = handler.handle_post(
            "/api/v1/openclaw/unknown", {}, mock_http(body={}, method="POST")
        )
        assert result is None


# ============================================================================
# DELETE Routing: Sessions
# ============================================================================


class TestDeleteCloseSession:
    """Tests for DELETE /sessions/:id."""

    def test_close_session(self, handler, mock_http, active_session):
        path = f"/api/v1/openclaw/sessions/{active_session.id}"
        result = handler.handle_delete(path, {}, mock_http())
        assert _status(result) == 200
        body = _body(result)
        assert body["closed"] is True

    def test_close_session_not_found(self, handler, mock_http):
        result = handler.handle_delete(
            "/api/v1/openclaw/sessions/nonexistent", {}, mock_http()
        )
        assert _status(result) == 404

    def test_close_session_access_denied(self, handler, mock_http, other_user_session):
        handler.get_current_user = lambda h: _MockUser(role="viewer")
        path = f"/api/v1/openclaw/sessions/{other_user_session.id}"
        result = handler.handle_delete(path, {}, mock_http())
        assert _status(result) == 403


# ============================================================================
# DELETE Routing: Credentials
# ============================================================================


class TestDeleteCredential:
    """Tests for DELETE /credentials/:id."""

    def test_delete_credential(self, handler, mock_http, credential):
        path = f"/api/v1/openclaw/credentials/{credential.id}"
        result = handler.handle_delete(path, {}, mock_http())
        assert _status(result) == 200
        body = _body(result)
        assert body["deleted"] is True

    def test_delete_credential_not_found(self, handler, mock_http):
        result = handler.handle_delete(
            "/api/v1/openclaw/credentials/nonexistent", {}, mock_http()
        )
        assert _status(result) == 404

    def test_delete_credential_access_denied(
        self, handler, mock_http, other_user_credential
    ):
        handler.get_current_user = lambda h: _MockUser(role="viewer")
        path = f"/api/v1/openclaw/credentials/{other_user_credential.id}"
        result = handler.handle_delete(path, {}, mock_http())
        assert _status(result) == 403


# ============================================================================
# DELETE Routing: Policy Rules
# ============================================================================


class TestDeletePolicyRule:
    """Tests for DELETE /policy/rules/:name."""

    def test_remove_policy_rule(self, handler, mock_http):
        result = handler.handle_delete(
            "/api/v1/openclaw/policy/rules/block_destructive", {}, mock_http()
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["name"] == "block_destructive"


# ============================================================================
# DELETE Routing: Unknown paths
# ============================================================================


class TestDeleteUnknownPath:
    """Tests for DELETE on unrecognized paths."""

    def test_unknown_delete_returns_none(self, handler, mock_http):
        result = handler.handle_delete(
            "/api/v1/openclaw/unknown", {}, mock_http()
        )
        assert result is None


# ============================================================================
# Path Normalization: All 4 prefix variants
# ============================================================================


class TestAllPathVariants:
    """Verify that all 4 URL prefix variants reach the same endpoint."""

    PREFIXES = [
        "/api/gateway/openclaw",
        "/api/v1/gateway/openclaw",
        "/api/v1/openclaw",
        "/api/openclaw",
    ]

    def test_list_sessions_all_prefixes(self, handler, mock_http):
        for prefix in self.PREFIXES:
            result = handler.handle(f"{prefix}/sessions", {}, mock_http())
            assert _status(result) == 200, f"Failed for prefix: {prefix}"

    def test_health_all_prefixes(self, handler, mock_http):
        for prefix in self.PREFIXES:
            result = handler.handle(f"{prefix}/health", {}, mock_http())
            assert _status(result) == 200, f"Failed for prefix: {prefix}"

    def test_metrics_all_prefixes(self, handler, mock_http):
        for prefix in self.PREFIXES:
            result = handler.handle(f"{prefix}/metrics", {}, mock_http())
            assert _status(result) == 200, f"Failed for prefix: {prefix}"

    def test_audit_all_prefixes(self, handler, mock_http):
        for prefix in self.PREFIXES:
            result = handler.handle(f"{prefix}/audit", {}, mock_http())
            assert _status(result) == 200, f"Failed for prefix: {prefix}"

    def test_create_session_all_prefixes(self, handler, mock_http):
        for prefix in self.PREFIXES:
            result = handler.handle_post(
                f"{prefix}/sessions", {}, mock_http(body={}, method="POST")
            )
            assert _status(result) == 201, f"Failed for prefix: {prefix}"


# ============================================================================
# Error Handling: Store Failures
# ============================================================================


class TestStoreErrors:
    """Tests for error handling when the store raises exceptions."""

    def test_list_sessions_store_error(self, handler, mock_http, store):
        store.list_sessions = MagicMock(side_effect=OSError("db unavailable"))
        result = handler.handle("/api/v1/openclaw/sessions", {}, mock_http())
        assert _status(result) == 500

    def test_get_session_store_error(self, handler, mock_http, store):
        store.get_session = MagicMock(side_effect=OSError("connection lost"))
        result = handler.handle("/api/v1/openclaw/sessions/some-id", {}, mock_http())
        assert _status(result) == 500

    def test_health_store_error_returns_503(self, handler, mock_http, store):
        store.get_metrics = MagicMock(side_effect=OSError("db down"))
        result = handler.handle("/api/v1/openclaw/health", {}, mock_http())
        assert _status(result) == 503
        body = _body(result)
        assert body["status"] == "error"
        assert body["healthy"] is False

    def test_metrics_store_error(self, handler, mock_http, store):
        store.get_metrics = MagicMock(side_effect=ValueError("bad data"))
        result = handler.handle("/api/v1/openclaw/metrics", {}, mock_http())
        assert _status(result) == 500

    def test_audit_store_error(self, handler, mock_http, store):
        store.get_audit_log = MagicMock(side_effect=OSError("db error"))
        result = handler.handle("/api/v1/openclaw/audit", {}, mock_http())
        assert _status(result) == 500

    def test_create_session_store_error(self, handler, mock_http, store):
        store.create_session = MagicMock(side_effect=RuntimeError("write failed"))
        body = {"config": {}, "metadata": {}}
        result = handler.handle_post(
            "/api/v1/openclaw/sessions", {}, mock_http(body=body, method="POST")
        )
        assert _status(result) == 500

    def test_execute_action_store_error(self, handler, mock_http, store, active_session):
        store.create_action = MagicMock(side_effect=OSError("write failed"))
        body = {
            "session_id": active_session.id,
            "action_type": "code.execute",
            "input": {},
        }
        result = handler.handle_post(
            "/api/v1/openclaw/actions", {}, mock_http(body=body, method="POST")
        )
        assert _status(result) == 500

    def test_stats_store_error(self, handler, mock_http, store):
        store.get_metrics = MagicMock(side_effect=TypeError("bad metric"))
        result = handler.handle("/api/v1/openclaw/stats", {}, mock_http())
        assert _status(result) == 500


# ============================================================================
# Circuit Breaker Helpers
# ============================================================================


class TestCircuitBreakerHelpers:
    """Tests for module-level circuit breaker helper functions."""

    def test_get_circuit_breaker(self):
        cb = get_openclaw_circuit_breaker()
        assert cb is not None
        assert cb.name == "openclaw_gateway_handler"

    def test_get_circuit_breaker_status(self):
        status = get_openclaw_circuit_breaker_status()
        assert isinstance(status, dict)
        assert "config" in status

    def test_circuit_breaker_singleton(self):
        cb1 = get_openclaw_circuit_breaker()
        cb2 = get_openclaw_circuit_breaker()
        assert cb1 is cb2


# ============================================================================
# Handler Registration Factory
# ============================================================================


class TestHandlerFactory:
    """Tests for get_openclaw_gateway_handler factory function."""

    def test_factory_returns_handler(self):
        h = get_openclaw_gateway_handler(server_context={})
        assert isinstance(h, OpenClawGatewayHandler)

    def test_factory_with_context(self):
        ctx = {"storage": MagicMock()}
        h = get_openclaw_gateway_handler(server_context=ctx)
        assert h.ctx is ctx


# ============================================================================
# ROUTES class attribute
# ============================================================================


class TestRoutes:
    """Tests for ROUTES class attribute completeness."""

    def test_routes_not_empty(self):
        assert len(OpenClawGatewayHandler.ROUTES) > 0

    def test_routes_include_sessions(self):
        routes = OpenClawGatewayHandler.ROUTES
        assert "/api/v1/openclaw/sessions" in routes

    def test_routes_include_actions(self):
        routes = OpenClawGatewayHandler.ROUTES
        assert "/api/v1/openclaw/actions" in routes

    def test_routes_include_credentials(self):
        routes = OpenClawGatewayHandler.ROUTES
        assert "/api/v1/openclaw/credentials" in routes

    def test_routes_include_health(self):
        routes = OpenClawGatewayHandler.ROUTES
        assert "/api/v1/openclaw/health" in routes

    def test_routes_include_legacy_paths(self):
        routes = OpenClawGatewayHandler.ROUTES
        assert "/api/gateway/openclaw/sessions" in routes
        assert "/api/gateway/openclaw/health" in routes

    def test_routes_include_policy(self):
        routes = OpenClawGatewayHandler.ROUTES
        assert "/api/v1/openclaw/policy/rules" in routes

    def test_routes_include_approvals(self):
        routes = OpenClawGatewayHandler.ROUTES
        assert "/api/v1/openclaw/approvals" in routes

    def test_routes_include_stats(self):
        routes = OpenClawGatewayHandler.ROUTES
        assert "/api/v1/openclaw/stats" in routes


# ============================================================================
# _get_user_id / _get_tenant_id
# ============================================================================


class TestUserAndTenantExtraction:
    """Tests for _get_user_id and _get_tenant_id."""

    def test_get_user_id_from_user(self, handler, mock_http):
        http = mock_http()
        uid = handler._get_user_id(http)
        assert uid == "test-user-001"

    def test_get_tenant_id_from_user(self, handler, mock_http):
        http = mock_http()
        tid = handler._get_tenant_id(http)
        assert tid == "test-org-001"

    def test_get_user_id_anonymous(self, handler, mock_http):
        handler.get_current_user = lambda h: None
        uid = handler._get_user_id(mock_http())
        assert uid == "anonymous"

    def test_get_tenant_id_none_when_no_user(self, handler, mock_http):
        handler.get_current_user = lambda h: None
        tid = handler._get_tenant_id(mock_http())
        assert tid is None

    def test_get_tenant_id_none_when_no_org_id(self, handler, mock_http):
        class SimpleUser:
            user_id = "u1"

        handler.get_current_user = lambda h: SimpleUser()
        tid = handler._get_tenant_id(mock_http())
        assert tid is None


# ============================================================================
# Audit Side Effects
# ============================================================================


class TestAuditSideEffects:
    """Verify that mutating operations create audit log entries."""

    def test_create_session_creates_audit(self, handler, mock_http, store):
        handler.handle_post(
            "/api/v1/openclaw/sessions", {}, mock_http(body={}, method="POST")
        )
        entries, _ = store.get_audit_log(action="session.create")
        assert len(entries) >= 1

    def test_close_session_creates_audit(self, handler, mock_http, store, active_session):
        path = f"/api/v1/openclaw/sessions/{active_session.id}"
        handler.handle_delete(path, {}, mock_http())
        entries, _ = store.get_audit_log(action="session.close")
        assert len(entries) >= 1

    def test_end_session_creates_audit(self, handler, mock_http, store, active_session):
        path = f"/api/v1/openclaw/sessions/{active_session.id}/end"
        handler.handle_post(path, {}, mock_http(body={}, method="POST"))
        entries, _ = store.get_audit_log(action="session.end")
        assert len(entries) >= 1

    def test_execute_action_creates_audit(self, handler, mock_http, store, active_session):
        body = {
            "session_id": active_session.id,
            "action_type": "code.execute",
            "input": {},
        }
        handler.handle_post(
            "/api/v1/openclaw/actions", {}, mock_http(body=body, method="POST")
        )
        entries, _ = store.get_audit_log(action="action.execute")
        assert len(entries) >= 1

    def test_cancel_action_creates_audit(self, handler, mock_http, store, running_action):
        path = f"/api/v1/openclaw/actions/{running_action.id}/cancel"
        handler.handle_post(path, {}, mock_http(body={}, method="POST"))
        entries, _ = store.get_audit_log(action="action.cancel")
        assert len(entries) >= 1

    def test_store_credential_creates_audit(self, handler, mock_http, store):
        body = {
            "name": "audit_key",
            "type": "api_key",
            "secret": "sk-test-1234567890abcdef",
        }
        handler.handle_post(
            "/api/v1/openclaw/credentials", {}, mock_http(body=body, method="POST")
        )
        entries, _ = store.get_audit_log(action="credential.create")
        assert len(entries) >= 1

    def test_delete_credential_creates_audit(self, handler, mock_http, store, credential):
        path = f"/api/v1/openclaw/credentials/{credential.id}"
        handler.handle_delete(path, {}, mock_http())
        entries, _ = store.get_audit_log(action="credential.delete")
        assert len(entries) >= 1

    def test_add_policy_rule_creates_audit(self, handler, mock_http, store):
        body = {"name": "test_rule", "decision": "deny"}
        handler.handle_post(
            "/api/v1/openclaw/policy/rules", {}, mock_http(body=body, method="POST")
        )
        entries, _ = store.get_audit_log(action="policy.rule.add")
        assert len(entries) >= 1

    def test_remove_policy_rule_creates_audit(self, handler, mock_http, store):
        handler.handle_delete(
            "/api/v1/openclaw/policy/rules/some_rule", {}, mock_http()
        )
        entries, _ = store.get_audit_log(action="policy.rule.remove")
        assert len(entries) >= 1

    def test_approve_action_creates_audit(self, handler, mock_http, store):
        handler.handle_post(
            "/api/v1/openclaw/approvals/appr-1/approve",
            {},
            mock_http(body={"reason": "ok"}, method="POST"),
        )
        entries, _ = store.get_audit_log(action="approval.approve")
        assert len(entries) >= 1

    def test_deny_action_creates_audit(self, handler, mock_http, store):
        handler.handle_post(
            "/api/v1/openclaw/approvals/appr-2/deny",
            {},
            mock_http(body={"reason": "nope"}, method="POST"),
        )
        entries, _ = store.get_audit_log(action="approval.deny")
        assert len(entries) >= 1
