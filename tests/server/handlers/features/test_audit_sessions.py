"""
Comprehensive tests for the Audit Sessions handler.

Covers: can_handle, routing, session CRUD, audit lifecycle (start/pause/resume/cancel),
SSE event streaming, human intervention, RBAC permissions, findings, error handling,
and report export.
"""

import sys
import types as _types_mod

# Pre-stub Slack modules to avoid circular ImportError
_SLACK_ATTRS = [
    "SlackHandler",
    "get_slack_handler",
    "get_slack_integration",
    "get_workspace_store",
    "resolve_workspace",
    "create_tracked_task",
    "_validate_slack_url",
    "SLACK_SIGNING_SECRET",
    "SLACK_BOT_TOKEN",
    "SLACK_WEBHOOK_URL",
    "SLACK_ALLOWED_DOMAINS",
    "SignatureVerifierMixin",
    "CommandsMixin",
    "EventsMixin",
    "init_slack_handler",
]
for _mod_name in (
    "aragora.server.handlers.social.slack.handler",
    "aragora.server.handlers.social.slack",
):
    if _mod_name not in sys.modules:
        _m = _types_mod.ModuleType(_mod_name)
        for _a in _SLACK_ATTRS:
            setattr(_m, _a, None)
        sys.modules[_mod_name] = _m

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass, field

import pytest

from aragora.server.handlers.features import audit_sessions as audit_mod
from aragora.server.handlers.features.audit_sessions import (
    AuditSessionsHandler,
    AUDIT_READ_PERMISSION,
    AUDIT_CREATE_PERMISSION,
    AUDIT_EXECUTE_PERMISSION,
    AUDIT_DELETE_PERMISSION,
    AUDIT_INTERVENE_PERMISSION,
    _sessions,
    _findings,
    _event_queues,
    _cancellation_tokens,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeQuery(dict):
    """Dict-like query object that supports .get()."""

    pass


class FakeRequest:
    """Minimal request object matching what AuditSessionsHandler expects."""

    def __init__(
        self,
        method: str = "GET",
        path: str = "/api/v1/audit/sessions",
        body_data: dict | None = None,
        query_params: dict | None = None,
        headers: dict | None = None,
    ):
        self.method = method
        self.path = path
        self._body_data = body_data or {}
        self.query = FakeQuery(query_params or {})
        self.headers = headers or {"Authorization": "Bearer fake-token"}

    async def json(self):
        return self._body_data

    async def body(self):
        return json.dumps(self._body_data).encode()


@dataclass
class FakeAuthContext:
    user_id: str = "test-user"
    workspace_id: str = "ws-1"
    roles: list = field(default_factory=lambda: ["admin"])
    permissions: list = field(
        default_factory=lambda: [
            "audit:read",
            "audit:create",
            "audit:execute",
            "audit:delete",
            "audit:intervene",
        ]
    )
    is_authenticated: bool = True


def _make_handler() -> AuditSessionsHandler:
    """Create an AuditSessionsHandler with a minimal server context."""
    ctx: Any = {}
    handler = AuditSessionsHandler(ctx)
    return handler


def _parse_body(response: dict) -> dict:
    """Parse the JSON body from a handler response dict."""
    body = response.get("body", "{}")
    if isinstance(body, str):
        return json.loads(body)
    return body


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_state():
    """Clear all module-level state before each test."""
    _sessions.clear()
    _findings.clear()
    _event_queues.clear()
    _cancellation_tokens.clear()
    yield
    _sessions.clear()
    _findings.clear()
    _event_queues.clear()
    _cancellation_tokens.clear()


@pytest.fixture()
def handler():
    """Return a handler with auth/permission checks patched to pass."""
    h = _make_handler()
    h.get_auth_context = AsyncMock(return_value=FakeAuthContext())
    h.check_permission = MagicMock(return_value=True)
    return h


@pytest.fixture()
def _seed_session():
    """Seed a single pending session and return its id."""
    sid = "sess-001"
    _sessions[sid] = {
        "id": sid,
        "document_ids": ["doc-a", "doc-b"],
        "audit_types": ["security"],
        "config": {},
        "status": "pending",
        "created_at": "2025-01-01T00:00:00+00:00",
        "updated_at": "2025-01-01T00:00:00+00:00",
        "started_at": None,
        "completed_at": None,
        "progress": {
            "total_documents": 2,
            "processed_documents": 0,
            "total_chunks": 0,
            "processed_chunks": 0,
            "findings_count": 0,
        },
        "agents": [],
        "error": None,
    }
    _findings[sid] = []
    _event_queues[sid] = []
    return sid


# ---------------------------------------------------------------------------
# can_handle
# ---------------------------------------------------------------------------


class TestCanHandle:
    def test_matches_audit_path(self):
        h = _make_handler()
        assert h.can_handle("/api/v1/audit/sessions") is True

    def test_matches_audit_subpath(self):
        h = _make_handler()
        assert h.can_handle("/api/v1/audit/sessions/abc/start") is True

    def test_rejects_non_audit_path(self):
        h = _make_handler()
        assert h.can_handle("/api/v1/debates") is False

    def test_rejects_empty_path(self):
        h = _make_handler()
        assert h.can_handle("") is False


# ---------------------------------------------------------------------------
# ROUTES class attribute
# ---------------------------------------------------------------------------


class TestRoutes:
    def test_routes_defined(self):
        assert isinstance(AuditSessionsHandler.ROUTES, list)
        assert len(AuditSessionsHandler.ROUTES) == 10

    def test_routes_contain_session_id_placeholder(self):
        routes_with_id = [r for r in AuditSessionsHandler.ROUTES if "{session_id}" in r]
        assert len(routes_with_id) == 9


# ---------------------------------------------------------------------------
# Permission mapping
# ---------------------------------------------------------------------------


class TestPermissionMapping:
    def setup_method(self):
        self.handler = _make_handler()

    def test_create_requires_audit_create(self):
        perm = self.handler._get_required_permission("/api/v1/audit/sessions", "POST")
        assert perm == AUDIT_CREATE_PERMISSION

    def test_list_requires_audit_read(self):
        perm = self.handler._get_required_permission("/api/v1/audit/sessions", "GET")
        assert perm == AUDIT_READ_PERMISSION

    def test_get_session_requires_audit_read(self):
        perm = self.handler._get_required_permission("/api/v1/audit/sessions/abc", "GET")
        assert perm == AUDIT_READ_PERMISSION

    def test_delete_requires_audit_delete(self):
        perm = self.handler._get_required_permission("/api/v1/audit/sessions/abc", "DELETE")
        assert perm == AUDIT_DELETE_PERMISSION

    def test_start_requires_audit_execute(self):
        perm = self.handler._get_required_permission("/api/v1/audit/sessions/abc/start", "POST")
        assert perm == AUDIT_EXECUTE_PERMISSION

    def test_pause_requires_audit_execute(self):
        perm = self.handler._get_required_permission("/api/v1/audit/sessions/abc/pause", "POST")
        assert perm == AUDIT_EXECUTE_PERMISSION

    def test_resume_requires_audit_execute(self):
        perm = self.handler._get_required_permission("/api/v1/audit/sessions/abc/resume", "POST")
        assert perm == AUDIT_EXECUTE_PERMISSION

    def test_cancel_requires_audit_execute(self):
        perm = self.handler._get_required_permission("/api/v1/audit/sessions/abc/cancel", "POST")
        assert perm == AUDIT_EXECUTE_PERMISSION

    def test_intervene_requires_audit_intervene(self):
        perm = self.handler._get_required_permission("/api/v1/audit/sessions/abc/intervene", "POST")
        assert perm == AUDIT_INTERVENE_PERMISSION

    def test_findings_requires_audit_read(self):
        perm = self.handler._get_required_permission("/api/v1/audit/sessions/abc/findings", "GET")
        assert perm == AUDIT_READ_PERMISSION

    def test_events_requires_audit_read(self):
        perm = self.handler._get_required_permission("/api/v1/audit/sessions/abc/events", "GET")
        assert perm == AUDIT_READ_PERMISSION

    def test_report_requires_audit_read(self):
        perm = self.handler._get_required_permission("/api/v1/audit/sessions/abc/report", "GET")
        assert perm == AUDIT_READ_PERMISSION


# ---------------------------------------------------------------------------
# RBAC enforcement (auth failures)
# ---------------------------------------------------------------------------


class TestRBACEnforcement:
    @pytest.mark.asyncio
    async def test_unauthenticated_returns_401(self):
        from aragora.server.handlers.utils.auth import UnauthorizedError

        h = _make_handler()
        h.get_auth_context = AsyncMock(side_effect=UnauthorizedError("no token"))
        req = FakeRequest(method="GET", path="/api/v1/audit/sessions")
        resp = await h.handle_request(req)
        assert resp["status"] == 401
        body = _parse_body(resp)
        assert "Authentication required" in body["error"]

    @pytest.mark.asyncio
    async def test_forbidden_returns_403(self):
        from aragora.server.handlers.utils.auth import ForbiddenError

        h = _make_handler()
        h.get_auth_context = AsyncMock(return_value=FakeAuthContext())
        h.check_permission = MagicMock(
            side_effect=ForbiddenError("denied", permission="audit:read")
        )
        req = FakeRequest(method="GET", path="/api/v1/audit/sessions")
        resp = await h.handle_request(req)
        assert resp["status"] == 403


# ---------------------------------------------------------------------------
# Session CRUD
# ---------------------------------------------------------------------------


class TestCreateSession:
    @pytest.mark.asyncio
    async def test_create_session_success(self, handler):
        req = FakeRequest(
            method="POST",
            path="/api/v1/audit/sessions",
            body_data={"document_ids": ["doc-1", "doc-2"], "audit_types": ["security"]},
        )
        resp = await handler.handle_request(req)
        assert resp["status"] == 201
        body = _parse_body(resp)
        assert body["status"] == "pending"
        assert body["document_ids"] == ["doc-1", "doc-2"]
        assert body["id"] in _sessions

    @pytest.mark.asyncio
    async def test_create_session_no_document_ids(self, handler):
        req = FakeRequest(method="POST", path="/api/v1/audit/sessions", body_data={})
        resp = await handler.handle_request(req)
        assert resp["status"] == 400
        body = _parse_body(resp)
        assert "document_ids" in body["error"]

    @pytest.mark.asyncio
    async def test_create_session_invalid_json(self, handler):
        req = FakeRequest(method="POST", path="/api/v1/audit/sessions")
        # Make json() raise
        req.json = AsyncMock(side_effect=json.JSONDecodeError("bad", "", 0))
        req.body = AsyncMock(return_value=b"not json")
        resp = await handler.handle_request(req)
        assert resp["status"] == 400

    @pytest.mark.asyncio
    async def test_create_session_default_audit_types(self, handler):
        req = FakeRequest(
            method="POST",
            path="/api/v1/audit/sessions",
            body_data={"document_ids": ["d1"]},
        )
        resp = await handler.handle_request(req)
        body = _parse_body(resp)
        assert "security" in body["audit_types"]
        assert "compliance" in body["audit_types"]

    @pytest.mark.asyncio
    async def test_create_session_initialises_event_queue(self, handler):
        req = FakeRequest(
            method="POST",
            path="/api/v1/audit/sessions",
            body_data={"document_ids": ["d1"]},
        )
        resp = await handler.handle_request(req)
        body = _parse_body(resp)
        assert body["id"] in _event_queues

    @pytest.mark.asyncio
    async def test_create_session_initialises_findings(self, handler):
        req = FakeRequest(
            method="POST",
            path="/api/v1/audit/sessions",
            body_data={"document_ids": ["d1"]},
        )
        resp = await handler.handle_request(req)
        body = _parse_body(resp)
        assert _findings[body["id"]] == []


class TestListSessions:
    @pytest.mark.asyncio
    async def test_list_empty(self, handler):
        req = FakeRequest(method="GET", path="/api/v1/audit/sessions")
        resp = await handler.handle_request(req)
        assert resp["status"] == 200
        body = _parse_body(resp)
        assert body["sessions"] == []
        assert body["total"] == 0

    @pytest.mark.asyncio
    async def test_list_returns_sessions(self, handler, _seed_session):
        req = FakeRequest(method="GET", path="/api/v1/audit/sessions")
        resp = await handler.handle_request(req)
        body = _parse_body(resp)
        assert body["total"] == 1
        assert body["sessions"][0]["id"] == _seed_session

    @pytest.mark.asyncio
    async def test_list_filter_by_status(self, handler, _seed_session):
        req = FakeRequest(
            method="GET",
            path="/api/v1/audit/sessions",
            query_params={"status": "running"},
        )
        resp = await handler.handle_request(req)
        body = _parse_body(resp)
        assert body["total"] == 0

    @pytest.mark.asyncio
    async def test_list_pagination(self, handler):
        # Create 3 sessions
        for i in range(3):
            sid = f"s-{i}"
            _sessions[sid] = {
                "id": sid,
                "document_ids": [],
                "audit_types": [],
                "config": {},
                "status": "pending",
                "created_at": f"2025-01-0{i + 1}T00:00:00+00:00",
                "updated_at": "",
                "started_at": None,
                "completed_at": None,
                "progress": {},
                "agents": [],
                "error": None,
            }
        req = FakeRequest(
            method="GET",
            path="/api/v1/audit/sessions",
            query_params={"limit": "2", "offset": "0"},
        )
        resp = await handler.handle_request(req)
        body = _parse_body(resp)
        assert body["total"] == 3
        assert len(body["sessions"]) == 2


class TestGetSession:
    @pytest.mark.asyncio
    async def test_get_existing_session(self, handler, _seed_session):
        req = FakeRequest(method="GET", path=f"/api/v1/audit/sessions/{_seed_session}")
        resp = await handler.handle_request(req)
        assert resp["status"] == 200
        body = _parse_body(resp)
        assert body["id"] == _seed_session

    @pytest.mark.asyncio
    async def test_get_nonexistent_session(self, handler):
        req = FakeRequest(method="GET", path="/api/v1/audit/sessions/no-such")
        resp = await handler.handle_request(req)
        assert resp["status"] == 404


class TestDeleteSession:
    @pytest.mark.asyncio
    async def test_delete_pending_session(self, handler, _seed_session):
        req = FakeRequest(method="DELETE", path=f"/api/v1/audit/sessions/{_seed_session}")
        resp = await handler.handle_request(req)
        assert resp["status"] == 200
        body = _parse_body(resp)
        assert body["deleted"] == _seed_session
        assert _seed_session not in _sessions

    @pytest.mark.asyncio
    async def test_delete_running_session_fails(self, handler, _seed_session):
        _sessions[_seed_session]["status"] = "running"
        req = FakeRequest(method="DELETE", path=f"/api/v1/audit/sessions/{_seed_session}")
        resp = await handler.handle_request(req)
        assert resp["status"] == 400
        body = _parse_body(resp)
        assert "running" in body["error"].lower() or "Cannot delete" in body["error"]

    @pytest.mark.asyncio
    async def test_delete_nonexistent_session(self, handler):
        req = FakeRequest(method="DELETE", path="/api/v1/audit/sessions/missing")
        resp = await handler.handle_request(req)
        assert resp["status"] == 404

    @pytest.mark.asyncio
    async def test_delete_cleans_up_findings_and_queues(self, handler, _seed_session):
        _findings[_seed_session] = [{"id": "f1"}]
        _event_queues[_seed_session] = []
        req = FakeRequest(method="DELETE", path=f"/api/v1/audit/sessions/{_seed_session}")
        await handler.handle_request(req)
        assert _seed_session not in _findings
        assert _seed_session not in _event_queues


# ---------------------------------------------------------------------------
# Audit lifecycle
# ---------------------------------------------------------------------------


class TestStartAudit:
    @pytest.mark.asyncio
    async def test_start_pending_session(self, handler, _seed_session):
        req = FakeRequest(method="POST", path=f"/api/v1/audit/sessions/{_seed_session}/start")
        with patch.object(handler, "_run_audit_background", new_callable=AsyncMock):
            resp = await handler.handle_request(req)
        assert resp["status"] == 200
        body = _parse_body(resp)
        assert body["status"] == "running"
        assert body["started_at"] is not None

    @pytest.mark.asyncio
    async def test_start_paused_session(self, handler, _seed_session):
        _sessions[_seed_session]["status"] = "paused"
        req = FakeRequest(method="POST", path=f"/api/v1/audit/sessions/{_seed_session}/start")
        with patch.object(handler, "_run_audit_background", new_callable=AsyncMock):
            resp = await handler.handle_request(req)
        assert resp["status"] == 200

    @pytest.mark.asyncio
    async def test_start_completed_session_fails(self, handler, _seed_session):
        _sessions[_seed_session]["status"] = "completed"
        req = FakeRequest(method="POST", path=f"/api/v1/audit/sessions/{_seed_session}/start")
        resp = await handler.handle_request(req)
        assert resp["status"] == 400

    @pytest.mark.asyncio
    async def test_start_nonexistent_session(self, handler):
        req = FakeRequest(method="POST", path="/api/v1/audit/sessions/nope/start")
        resp = await handler.handle_request(req)
        assert resp["status"] == 404

    @pytest.mark.asyncio
    async def test_start_emits_event(self, handler, _seed_session):
        queue = asyncio.Queue()
        _event_queues[_seed_session] = [queue]
        req = FakeRequest(method="POST", path=f"/api/v1/audit/sessions/{_seed_session}/start")
        with patch.object(handler, "_run_audit_background", new_callable=AsyncMock):
            await handler.handle_request(req)
        event = queue.get_nowait()
        assert event["type"] == "audit_started"


class TestPauseAudit:
    @pytest.mark.asyncio
    async def test_pause_running_session(self, handler, _seed_session):
        _sessions[_seed_session]["status"] = "running"
        req = FakeRequest(method="POST", path=f"/api/v1/audit/sessions/{_seed_session}/pause")
        resp = await handler.handle_request(req)
        assert resp["status"] == 200
        body = _parse_body(resp)
        assert body["status"] == "paused"

    @pytest.mark.asyncio
    async def test_pause_pending_session_fails(self, handler, _seed_session):
        req = FakeRequest(method="POST", path=f"/api/v1/audit/sessions/{_seed_session}/pause")
        resp = await handler.handle_request(req)
        assert resp["status"] == 400

    @pytest.mark.asyncio
    async def test_pause_nonexistent(self, handler):
        req = FakeRequest(method="POST", path="/api/v1/audit/sessions/xyz/pause")
        resp = await handler.handle_request(req)
        assert resp["status"] == 404

    @pytest.mark.asyncio
    async def test_pause_emits_event(self, handler, _seed_session):
        _sessions[_seed_session]["status"] = "running"
        queue = asyncio.Queue()
        _event_queues[_seed_session] = [queue]
        req = FakeRequest(method="POST", path=f"/api/v1/audit/sessions/{_seed_session}/pause")
        await handler.handle_request(req)
        event = queue.get_nowait()
        assert event["type"] == "audit_paused"


class TestResumeAudit:
    @pytest.mark.asyncio
    async def test_resume_paused_session(self, handler, _seed_session):
        _sessions[_seed_session]["status"] = "paused"
        req = FakeRequest(method="POST", path=f"/api/v1/audit/sessions/{_seed_session}/resume")
        resp = await handler.handle_request(req)
        assert resp["status"] == 200
        body = _parse_body(resp)
        assert body["status"] == "running"

    @pytest.mark.asyncio
    async def test_resume_pending_session_fails(self, handler, _seed_session):
        req = FakeRequest(method="POST", path=f"/api/v1/audit/sessions/{_seed_session}/resume")
        resp = await handler.handle_request(req)
        assert resp["status"] == 400

    @pytest.mark.asyncio
    async def test_resume_nonexistent(self, handler):
        req = FakeRequest(method="POST", path="/api/v1/audit/sessions/xyz/resume")
        resp = await handler.handle_request(req)
        assert resp["status"] == 404

    @pytest.mark.asyncio
    async def test_resume_emits_event(self, handler, _seed_session):
        _sessions[_seed_session]["status"] = "paused"
        queue = asyncio.Queue()
        _event_queues[_seed_session] = [queue]
        req = FakeRequest(method="POST", path=f"/api/v1/audit/sessions/{_seed_session}/resume")
        await handler.handle_request(req)
        event = queue.get_nowait()
        assert event["type"] == "audit_resumed"


class TestCancelAudit:
    @pytest.mark.asyncio
    async def test_cancel_running_session(self, handler, _seed_session):
        _sessions[_seed_session]["status"] = "running"
        req = FakeRequest(
            method="POST",
            path=f"/api/v1/audit/sessions/{_seed_session}/cancel",
            body_data={"reason": "no longer needed"},
        )
        resp = await handler.handle_request(req)
        assert resp["status"] == 200
        body = _parse_body(resp)
        assert body["status"] == "cancelled"
        assert body["cancel_reason"] == "no longer needed"

    @pytest.mark.asyncio
    async def test_cancel_pending_session(self, handler, _seed_session):
        req = FakeRequest(
            method="POST",
            path=f"/api/v1/audit/sessions/{_seed_session}/cancel",
            body_data={},
        )
        resp = await handler.handle_request(req)
        assert resp["status"] == 200
        body = _parse_body(resp)
        assert body["status"] == "cancelled"

    @pytest.mark.asyncio
    async def test_cancel_completed_session_fails(self, handler, _seed_session):
        _sessions[_seed_session]["status"] = "completed"
        req = FakeRequest(method="POST", path=f"/api/v1/audit/sessions/{_seed_session}/cancel")
        resp = await handler.handle_request(req)
        assert resp["status"] == 400

    @pytest.mark.asyncio
    async def test_cancel_already_cancelled_fails(self, handler, _seed_session):
        _sessions[_seed_session]["status"] = "cancelled"
        req = FakeRequest(method="POST", path=f"/api/v1/audit/sessions/{_seed_session}/cancel")
        resp = await handler.handle_request(req)
        assert resp["status"] == 400

    @pytest.mark.asyncio
    async def test_cancel_nonexistent(self, handler):
        req = FakeRequest(method="POST", path="/api/v1/audit/sessions/nope/cancel")
        resp = await handler.handle_request(req)
        assert resp["status"] == 404

    @pytest.mark.asyncio
    async def test_cancel_triggers_cancellation_token(self, handler, _seed_session):
        _sessions[_seed_session]["status"] = "running"
        token = MagicMock()
        _cancellation_tokens[_seed_session] = token
        req = FakeRequest(
            method="POST",
            path=f"/api/v1/audit/sessions/{_seed_session}/cancel",
            body_data={"reason": "abort"},
        )
        await handler.handle_request(req)
        token.cancel.assert_called_once_with("abort")

    @pytest.mark.asyncio
    async def test_cancel_default_reason(self, handler, _seed_session):
        _sessions[_seed_session]["status"] = "running"
        req = FakeRequest(method="POST", path=f"/api/v1/audit/sessions/{_seed_session}/cancel")
        resp = await handler.handle_request(req)
        body = _parse_body(resp)
        assert body["cancel_reason"] == "User requested cancellation"

    @pytest.mark.asyncio
    async def test_cancel_emits_event(self, handler, _seed_session):
        _sessions[_seed_session]["status"] = "running"
        queue = asyncio.Queue()
        _event_queues[_seed_session] = [queue]
        req = FakeRequest(
            method="POST",
            path=f"/api/v1/audit/sessions/{_seed_session}/cancel",
            body_data={"reason": "test"},
        )
        await handler.handle_request(req)
        event = queue.get_nowait()
        assert event["type"] == "audit_cancelled"
        assert event["reason"] == "test"

    @pytest.mark.asyncio
    async def test_cancel_token_failure_still_cancels(self, handler, _seed_session):
        _sessions[_seed_session]["status"] = "running"
        token = MagicMock()
        token.cancel.side_effect = RuntimeError("boom")
        _cancellation_tokens[_seed_session] = token
        req = FakeRequest(
            method="POST",
            path=f"/api/v1/audit/sessions/{_seed_session}/cancel",
            body_data={},
        )
        resp = await handler.handle_request(req)
        assert resp["status"] == 200
        body = _parse_body(resp)
        assert body["status"] == "cancelled"


# ---------------------------------------------------------------------------
# Findings
# ---------------------------------------------------------------------------


class TestFindings:
    @pytest.mark.asyncio
    async def test_get_findings_empty(self, handler, _seed_session):
        req = FakeRequest(method="GET", path=f"/api/v1/audit/sessions/{_seed_session}/findings")
        resp = await handler.handle_request(req)
        assert resp["status"] == 200
        body = _parse_body(resp)
        assert body["findings"] == []
        assert body["total"] == 0

    @pytest.mark.asyncio
    async def test_get_findings_with_data(self, handler, _seed_session):
        _findings[_seed_session] = [
            {"id": "f1", "severity": "high", "audit_type": "security", "status": "open"},
            {"id": "f2", "severity": "low", "audit_type": "compliance", "status": "open"},
        ]
        req = FakeRequest(method="GET", path=f"/api/v1/audit/sessions/{_seed_session}/findings")
        resp = await handler.handle_request(req)
        body = _parse_body(resp)
        assert body["total"] == 2
        # high before low
        assert body["findings"][0]["id"] == "f1"

    @pytest.mark.asyncio
    async def test_filter_findings_by_severity(self, handler, _seed_session):
        _findings[_seed_session] = [
            {"id": "f1", "severity": "critical", "audit_type": "security", "status": "open"},
            {"id": "f2", "severity": "low", "audit_type": "security", "status": "open"},
        ]
        req = FakeRequest(
            method="GET",
            path=f"/api/v1/audit/sessions/{_seed_session}/findings",
            query_params={"severity": "critical"},
        )
        resp = await handler.handle_request(req)
        body = _parse_body(resp)
        assert body["total"] == 1
        assert body["findings"][0]["id"] == "f1"

    @pytest.mark.asyncio
    async def test_filter_findings_by_audit_type(self, handler, _seed_session):
        _findings[_seed_session] = [
            {"id": "f1", "severity": "medium", "audit_type": "security", "status": "open"},
            {"id": "f2", "severity": "medium", "audit_type": "compliance", "status": "open"},
        ]
        req = FakeRequest(
            method="GET",
            path=f"/api/v1/audit/sessions/{_seed_session}/findings",
            query_params={"audit_type": "compliance"},
        )
        resp = await handler.handle_request(req)
        body = _parse_body(resp)
        assert body["total"] == 1
        assert body["findings"][0]["id"] == "f2"

    @pytest.mark.asyncio
    async def test_filter_findings_by_status(self, handler, _seed_session):
        _findings[_seed_session] = [
            {"id": "f1", "severity": "high", "audit_type": "security", "status": "open"},
            {"id": "f2", "severity": "high", "audit_type": "security", "status": "acknowledged"},
        ]
        req = FakeRequest(
            method="GET",
            path=f"/api/v1/audit/sessions/{_seed_session}/findings",
            query_params={"status": "acknowledged"},
        )
        resp = await handler.handle_request(req)
        body = _parse_body(resp)
        assert body["total"] == 1
        assert body["findings"][0]["id"] == "f2"

    @pytest.mark.asyncio
    async def test_findings_pagination(self, handler, _seed_session):
        _findings[_seed_session] = [{"id": f"f{i}", "severity": "medium"} for i in range(5)]
        req = FakeRequest(
            method="GET",
            path=f"/api/v1/audit/sessions/{_seed_session}/findings",
            query_params={"limit": "2", "offset": "1"},
        )
        resp = await handler.handle_request(req)
        body = _parse_body(resp)
        assert body["total"] == 5
        assert len(body["findings"]) == 2

    @pytest.mark.asyncio
    async def test_findings_sorted_by_severity(self, handler, _seed_session):
        _findings[_seed_session] = [
            {"id": "f-info", "severity": "info"},
            {"id": "f-crit", "severity": "critical"},
            {"id": "f-med", "severity": "medium"},
        ]
        req = FakeRequest(method="GET", path=f"/api/v1/audit/sessions/{_seed_session}/findings")
        resp = await handler.handle_request(req)
        body = _parse_body(resp)
        ids = [f["id"] for f in body["findings"]]
        assert ids == ["f-crit", "f-med", "f-info"]

    @pytest.mark.asyncio
    async def test_findings_nonexistent_session(self, handler):
        req = FakeRequest(method="GET", path="/api/v1/audit/sessions/missing/findings")
        resp = await handler.handle_request(req)
        assert resp["status"] == 404


# ---------------------------------------------------------------------------
# SSE Event Streaming
# ---------------------------------------------------------------------------


class TestStreamEvents:
    @pytest.mark.asyncio
    async def test_stream_events_returns_sse_response(self, handler, _seed_session):
        req = FakeRequest(method="GET", path=f"/api/v1/audit/sessions/{_seed_session}/events")
        resp = await handler.handle_request(req)
        assert resp["status"] == 200
        assert resp["headers"]["Content-Type"] == "text/event-stream"

    @pytest.mark.asyncio
    async def test_stream_events_nonexistent(self, handler):
        req = FakeRequest(method="GET", path="/api/v1/audit/sessions/missing/events")
        resp = await handler.handle_request(req)
        assert resp["status"] == 404

    @pytest.mark.asyncio
    async def test_stream_events_generator_sends_connected(self, handler, _seed_session):
        req = FakeRequest(method="GET", path=f"/api/v1/audit/sessions/{_seed_session}/events")
        resp = await handler.handle_request(req)
        gen = resp["body"]
        first_msg = await gen.__anext__()
        data = json.loads(first_msg.replace("data: ", "").strip())
        assert data["type"] == "connected"
        assert data["session_id"] == _seed_session
        # Clean up generator
        await gen.aclose()

    @pytest.mark.asyncio
    async def test_stream_events_receives_emitted_event(self, handler, _seed_session):
        req = FakeRequest(method="GET", path=f"/api/v1/audit/sessions/{_seed_session}/events")
        resp = await handler.handle_request(req)
        gen = resp["body"]
        # Consume connected event
        await gen.__anext__()

        # Emit an event to the queue
        queues = _event_queues.get(_seed_session, [])
        assert len(queues) == 1
        await queues[0].put({"type": "test_event", "data": "hello"})

        second_msg = await gen.__anext__()
        data = json.loads(second_msg.replace("data: ", "").strip())
        assert data["type"] == "test_event"
        await gen.aclose()

    @pytest.mark.asyncio
    async def test_stream_events_registers_queue(self, handler, _seed_session):
        req = FakeRequest(method="GET", path=f"/api/v1/audit/sessions/{_seed_session}/events")
        resp = await handler.handle_request(req)
        assert len(_event_queues[_seed_session]) == 1
        # Clean up
        gen = resp["body"]
        await gen.aclose()


# ---------------------------------------------------------------------------
# Human Intervention
# ---------------------------------------------------------------------------


class TestHumanIntervention:
    @pytest.mark.asyncio
    async def test_approve_finding(self, handler, _seed_session):
        _findings[_seed_session] = [
            {"id": "f1", "severity": "high", "status": "open"},
        ]
        req = FakeRequest(
            method="POST",
            path=f"/api/v1/audit/sessions/{_seed_session}/intervene",
            body_data={"action": "approve_finding", "finding_id": "f1", "reason": "confirmed"},
        )
        resp = await handler.handle_request(req)
        assert resp["status"] == 200
        body = _parse_body(resp)
        assert body["success"] is True
        assert body["action"] == "approve_finding"
        # Verify the finding was updated
        f = _findings[_seed_session][0]
        assert f["status"] == "acknowledged"
        assert f["human_review"]["action"] == "approved"
        assert f["human_review"]["reason"] == "confirmed"

    @pytest.mark.asyncio
    async def test_reject_finding(self, handler, _seed_session):
        _findings[_seed_session] = [
            {"id": "f1", "severity": "low", "status": "open"},
        ]
        req = FakeRequest(
            method="POST",
            path=f"/api/v1/audit/sessions/{_seed_session}/intervene",
            body_data={"action": "reject_finding", "finding_id": "f1", "reason": "false positive"},
        )
        resp = await handler.handle_request(req)
        assert resp["status"] == 200
        f = _findings[_seed_session][0]
        assert f["status"] == "false_positive"
        assert f["human_review"]["action"] == "rejected"

    @pytest.mark.asyncio
    async def test_intervention_no_action(self, handler, _seed_session):
        req = FakeRequest(
            method="POST",
            path=f"/api/v1/audit/sessions/{_seed_session}/intervene",
            body_data={},
        )
        resp = await handler.handle_request(req)
        assert resp["status"] == 400
        body = _parse_body(resp)
        assert "action" in body["error"]

    @pytest.mark.asyncio
    async def test_intervention_invalid_json(self, handler, _seed_session):
        req = FakeRequest(
            method="POST",
            path=f"/api/v1/audit/sessions/{_seed_session}/intervene",
        )
        req.json = AsyncMock(side_effect=json.JSONDecodeError("bad", "", 0))
        req.body = AsyncMock(return_value=b"not json")
        resp = await handler.handle_request(req)
        assert resp["status"] == 400

    @pytest.mark.asyncio
    async def test_intervention_nonexistent_session(self, handler):
        req = FakeRequest(
            method="POST",
            path="/api/v1/audit/sessions/missing/intervene",
            body_data={"action": "approve_finding", "finding_id": "f1"},
        )
        resp = await handler.handle_request(req)
        assert resp["status"] == 404

    @pytest.mark.asyncio
    async def test_intervention_add_context_action(self, handler, _seed_session):
        req = FakeRequest(
            method="POST",
            path=f"/api/v1/audit/sessions/{_seed_session}/intervene",
            body_data={"action": "add_context", "context": "Extra info"},
        )
        resp = await handler.handle_request(req)
        assert resp["status"] == 200
        body = _parse_body(resp)
        assert body["action"] == "add_context"

    @pytest.mark.asyncio
    async def test_intervention_emits_event(self, handler, _seed_session):
        queue = asyncio.Queue()
        _event_queues[_seed_session] = [queue]
        req = FakeRequest(
            method="POST",
            path=f"/api/v1/audit/sessions/{_seed_session}/intervene",
            body_data={"action": "override_decision", "finding_id": "f1"},
        )
        await handler.handle_request(req)
        event = queue.get_nowait()
        assert event["type"] == "human_intervention"
        assert event["action"] == "override_decision"

    @pytest.mark.asyncio
    async def test_approve_finding_not_in_list(self, handler, _seed_session):
        """Approving a non-existent finding_id should still succeed (no match, no crash)."""
        _findings[_seed_session] = [
            {"id": "other", "severity": "high", "status": "open"},
        ]
        req = FakeRequest(
            method="POST",
            path=f"/api/v1/audit/sessions/{_seed_session}/intervene",
            body_data={"action": "approve_finding", "finding_id": "nonexistent"},
        )
        resp = await handler.handle_request(req)
        assert resp["status"] == 200
        # Original finding unchanged
        assert _findings[_seed_session][0]["status"] == "open"


# ---------------------------------------------------------------------------
# Report Export (fallback path -- aragora.reports not available in tests)
# ---------------------------------------------------------------------------


class TestExportReport:
    @pytest.mark.asyncio
    async def test_export_report_fallback(self, handler, _seed_session):
        """When aragora.reports is unavailable, handler falls back to JSON export."""
        req = FakeRequest(
            method="GET",
            path=f"/api/v1/audit/sessions/{_seed_session}/report",
        )
        resp = await handler.handle_request(req)
        assert resp["status"] == 200
        body = _parse_body(resp)
        assert "session" in body
        assert "findings" in body
        assert "generated_at" in body

    @pytest.mark.asyncio
    async def test_export_report_nonexistent(self, handler):
        req = FakeRequest(method="GET", path="/api/v1/audit/sessions/missing/report")
        resp = await handler.handle_request(req)
        assert resp["status"] == 404


# ---------------------------------------------------------------------------
# Routing edge cases
# ---------------------------------------------------------------------------


class TestRouting:
    @pytest.mark.asyncio
    async def test_unknown_endpoint(self, handler):
        req = FakeRequest(method="POST", path="/api/v1/audit/unknown")
        resp = await handler.handle_request(req)
        assert resp["status"] == 404
        body = _parse_body(resp)
        assert "not found" in body["error"].lower()

    @pytest.mark.asyncio
    async def test_session_id_parsing(self, handler, _seed_session):
        """Ensure session id is correctly extracted from path."""
        req = FakeRequest(method="GET", path=f"/api/v1/audit/sessions/{_seed_session}")
        resp = await handler.handle_request(req)
        assert resp["status"] == 200
        body = _parse_body(resp)
        assert body["id"] == _seed_session


# ---------------------------------------------------------------------------
# _emit_event
# ---------------------------------------------------------------------------


class TestEmitEvent:
    @pytest.mark.asyncio
    async def test_emit_to_multiple_queues(self, handler, _seed_session):
        q1 = asyncio.Queue()
        q2 = asyncio.Queue()
        _event_queues[_seed_session] = [q1, q2]
        await handler._emit_event(_seed_session, {"type": "test"})
        assert q1.get_nowait()["type"] == "test"
        assert q2.get_nowait()["type"] == "test"

    @pytest.mark.asyncio
    async def test_emit_skips_full_queue(self, handler, _seed_session):
        q = asyncio.Queue(maxsize=1)
        await q.put({"type": "old"})
        _event_queues[_seed_session] = [q]
        # Should not raise
        await handler._emit_event(_seed_session, {"type": "new"})
        # Still the old event
        assert q.get_nowait()["type"] == "old"

    @pytest.mark.asyncio
    async def test_emit_no_queues(self, handler):
        # Should not raise even when no queues exist
        await handler._emit_event("nonexistent", {"type": "test"})


# ---------------------------------------------------------------------------
# Response helpers
# ---------------------------------------------------------------------------


class TestResponseHelpers:
    def test_json_response(self):
        h = _make_handler()
        resp = h._json_response(200, {"key": "val"})
        assert resp["status"] == 200
        assert resp["headers"]["Content-Type"] == "application/json"
        body = json.loads(resp["body"])
        assert body["key"] == "val"

    def test_error_response(self):
        h = _make_handler()
        resp = h._error_response(404, "Not found")
        assert resp["status"] == 404
        body = json.loads(resp["body"])
        assert body["error"] == "Not found"

    def test_sse_response(self):
        h = _make_handler()
        resp = h._sse_response("gen")
        assert resp["status"] == 200
        assert resp["headers"]["Content-Type"] == "text/event-stream"
        assert resp["headers"]["Cache-Control"] == "no-cache"
        assert resp["body"] == "gen"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_permission_constants(self):
        assert AUDIT_READ_PERMISSION == "audit:read"
        assert AUDIT_CREATE_PERMISSION == "audit:create"
        assert AUDIT_EXECUTE_PERMISSION == "audit:execute"
        assert AUDIT_DELETE_PERMISSION == "audit:delete"
        assert AUDIT_INTERVENE_PERMISSION == "audit:intervene"


# ---------------------------------------------------------------------------
# Circuit Breaker
# ---------------------------------------------------------------------------


class TestCircuitBreaker:
    def test_get_circuit_breaker_returns_instance(self):
        from aragora.server.handlers.features.audit_sessions import (
            get_audit_sessions_circuit_breaker,
        )

        cb = get_audit_sessions_circuit_breaker()
        assert cb is not None
        assert cb.name == "audit_sessions_handler"

    def test_get_circuit_breaker_status_returns_dict(self):
        from aragora.server.handlers.features.audit_sessions import (
            get_audit_sessions_circuit_breaker_status,
        )

        status = get_audit_sessions_circuit_breaker_status()
        assert isinstance(status, dict)

    def test_circuit_breaker_singleton(self):
        from aragora.server.handlers.features.audit_sessions import (
            get_audit_sessions_circuit_breaker,
        )

        cb1 = get_audit_sessions_circuit_breaker()
        cb2 = get_audit_sessions_circuit_breaker()
        assert cb1 is cb2


# ---------------------------------------------------------------------------
# Handler Initialization
# ---------------------------------------------------------------------------


class TestHandlerInit:
    def test_init_with_server_context(self):
        ctx = {"storage": "mock"}
        h = AuditSessionsHandler(server_context=ctx)
        assert h.ctx == ctx

    def test_init_with_ctx(self):
        ctx = {"key": "value"}
        h = AuditSessionsHandler(ctx=ctx)
        assert h.ctx == ctx

    def test_init_with_none(self):
        h = AuditSessionsHandler()
        assert h.ctx == {}

    def test_init_server_context_takes_priority(self):
        ctx1 = {"a": 1}
        ctx2 = {"b": 2}
        h = AuditSessionsHandler(ctx=ctx1, server_context=ctx2)
        assert h.ctx == ctx2


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------


class TestModuleExports:
    def test_all_exports(self):
        assert "AuditSessionsHandler" in audit_mod.__all__
        assert "get_audit_sessions_circuit_breaker" in audit_mod.__all__
        assert "get_audit_sessions_circuit_breaker_status" in audit_mod.__all__
        assert len(audit_mod.__all__) == 3


# ---------------------------------------------------------------------------
# Permission mapping edge case
# ---------------------------------------------------------------------------


class TestPermissionMappingEdgeCases:
    def test_default_fallback_returns_read(self):
        h = _make_handler()
        perm = h._get_required_permission("/api/v1/audit/sessions/abc/unknown", "PATCH")
        assert perm == AUDIT_READ_PERMISSION


# ---------------------------------------------------------------------------
# Create session additional validation
# ---------------------------------------------------------------------------


class TestCreateSessionDetails:
    @pytest.mark.asyncio
    async def test_create_session_with_custom_config(self, handler):
        config = {"use_debate": True, "min_confidence": 0.8, "parallel_agents": 5}
        req = FakeRequest(
            method="POST",
            path="/api/v1/audit/sessions",
            body_data={
                "document_ids": ["doc-x"],
                "audit_types": ["quality"],
                "config": config,
            },
        )
        resp = await handler.handle_request(req)
        assert resp["status"] == 201
        body = _parse_body(resp)
        assert body["config"] == config
        assert body["audit_types"] == ["quality"]

    @pytest.mark.asyncio
    async def test_create_session_progress_fields(self, handler):
        req = FakeRequest(
            method="POST",
            path="/api/v1/audit/sessions",
            body_data={"document_ids": ["d1", "d2", "d3"]},
        )
        resp = await handler.handle_request(req)
        body = _parse_body(resp)
        progress = body["progress"]
        assert progress["total_documents"] == 3
        assert progress["processed_documents"] == 0
        assert progress["total_chunks"] == 0
        assert progress["processed_chunks"] == 0
        assert progress["findings_count"] == 0

    @pytest.mark.asyncio
    async def test_create_session_has_timestamps(self, handler):
        req = FakeRequest(
            method="POST",
            path="/api/v1/audit/sessions",
            body_data={"document_ids": ["d1"]},
        )
        resp = await handler.handle_request(req)
        body = _parse_body(resp)
        assert body["created_at"] is not None
        assert body["updated_at"] is not None
        assert body["started_at"] is None
        assert body["completed_at"] is None

    @pytest.mark.asyncio
    async def test_create_session_empty_document_ids_list(self, handler):
        req = FakeRequest(
            method="POST",
            path="/api/v1/audit/sessions",
            body_data={"document_ids": []},
        )
        resp = await handler.handle_request(req)
        assert resp["status"] == 400
        body = _parse_body(resp)
        assert "document_ids" in body["error"]


# ---------------------------------------------------------------------------
# List sessions sorting
# ---------------------------------------------------------------------------


class TestListSessionsSorting:
    @pytest.mark.asyncio
    async def test_list_sorted_descending_by_created_at(self, handler):
        _sessions["s-old"] = {
            "id": "s-old",
            "document_ids": [],
            "audit_types": [],
            "config": {},
            "status": "pending",
            "created_at": "2025-01-01T00:00:00+00:00",
            "updated_at": "",
            "started_at": None,
            "completed_at": None,
            "progress": {},
            "agents": [],
            "error": None,
        }
        _sessions["s-new"] = {
            "id": "s-new",
            "document_ids": [],
            "audit_types": [],
            "config": {},
            "status": "pending",
            "created_at": "2025-06-01T00:00:00+00:00",
            "updated_at": "",
            "started_at": None,
            "completed_at": None,
            "progress": {},
            "agents": [],
            "error": None,
        }
        req = FakeRequest(method="GET", path="/api/v1/audit/sessions")
        resp = await handler.handle_request(req)
        body = _parse_body(resp)
        assert body["sessions"][0]["id"] == "s-new"
        assert body["sessions"][1]["id"] == "s-old"

    @pytest.mark.asyncio
    async def test_list_filter_matches_exact_status(self, handler, _seed_session):
        req = FakeRequest(
            method="GET",
            path="/api/v1/audit/sessions",
            query_params={"status": "pending"},
        )
        resp = await handler.handle_request(req)
        body = _parse_body(resp)
        assert body["total"] == 1

    @pytest.mark.asyncio
    async def test_list_filter_no_match(self, handler, _seed_session):
        req = FakeRequest(
            method="GET",
            path="/api/v1/audit/sessions",
            query_params={"status": "completed"},
        )
        resp = await handler.handle_request(req)
        body = _parse_body(resp)
        assert body["total"] == 0


# ---------------------------------------------------------------------------
# Report Export with findings data
# ---------------------------------------------------------------------------


class TestExportReportWithFindings:
    @pytest.mark.asyncio
    async def test_export_report_fallback_includes_findings(self, handler, _seed_session):
        _findings[_seed_session] = [
            {"id": "f1", "severity": "high", "status": "open"},
            {"id": "f2", "severity": "low", "status": "acknowledged"},
        ]
        req = FakeRequest(
            method="GET",
            path=f"/api/v1/audit/sessions/{_seed_session}/report",
        )
        resp = await handler.handle_request(req)
        assert resp["status"] == 200
        body = _parse_body(resp)
        assert len(body["findings"]) == 2
        assert body["session"]["id"] == _seed_session

    @pytest.mark.asyncio
    async def test_export_report_query_params_ignored_in_fallback(self, handler, _seed_session):
        """Query params like format, template don't affect fallback JSON export."""
        req = FakeRequest(
            method="GET",
            path=f"/api/v1/audit/sessions/{_seed_session}/report",
            query_params={"format": "html", "template": "executive_summary"},
        )
        resp = await handler.handle_request(req)
        assert resp["status"] == 200
        body = _parse_body(resp)
        assert "session" in body
        assert "findings" in body


# ---------------------------------------------------------------------------
# _parse_json_body branches
# ---------------------------------------------------------------------------


class TestParseJsonBody:
    @pytest.mark.asyncio
    async def test_parse_json_body_via_json_method(self):
        h = _make_handler()
        req = MagicMock()
        req.json = AsyncMock(return_value={"key": "val"})
        result = await h._parse_json_body(req)
        assert result == {"key": "val"}

    @pytest.mark.asyncio
    async def test_parse_json_body_via_body_method(self):
        h = _make_handler()
        req = MagicMock(spec=[])  # No .json attribute

        async def fake_body():
            return json.dumps({"from": "body"}).encode()

        req.body = fake_body
        result = await h._parse_json_body(req)
        assert result == {"from": "body"}

    @pytest.mark.asyncio
    async def test_parse_json_body_empty_request(self):
        h = _make_handler()
        req = MagicMock(spec=[])  # No .json, no .body
        result = await h._parse_json_body(req)
        assert result == {}


# ---------------------------------------------------------------------------
# Lifecycle state transitions additional tests
# ---------------------------------------------------------------------------


class TestLifecycleTransitions:
    @pytest.mark.asyncio
    async def test_start_running_session_fails(self, handler, _seed_session):
        _sessions[_seed_session]["status"] = "running"
        req = FakeRequest(method="POST", path=f"/api/v1/audit/sessions/{_seed_session}/start")
        resp = await handler.handle_request(req)
        assert resp["status"] == 400
        body = _parse_body(resp)
        assert "running" in body["error"]

    @pytest.mark.asyncio
    async def test_start_cancelled_session_fails(self, handler, _seed_session):
        _sessions[_seed_session]["status"] = "cancelled"
        req = FakeRequest(method="POST", path=f"/api/v1/audit/sessions/{_seed_session}/start")
        resp = await handler.handle_request(req)
        assert resp["status"] == 400

    @pytest.mark.asyncio
    async def test_pause_completed_session_fails(self, handler, _seed_session):
        _sessions[_seed_session]["status"] = "completed"
        req = FakeRequest(method="POST", path=f"/api/v1/audit/sessions/{_seed_session}/pause")
        resp = await handler.handle_request(req)
        assert resp["status"] == 400

    @pytest.mark.asyncio
    async def test_resume_running_session_fails(self, handler, _seed_session):
        _sessions[_seed_session]["status"] = "running"
        req = FakeRequest(method="POST", path=f"/api/v1/audit/sessions/{_seed_session}/resume")
        resp = await handler.handle_request(req)
        assert resp["status"] == 400

    @pytest.mark.asyncio
    async def test_resume_completed_session_fails(self, handler, _seed_session):
        _sessions[_seed_session]["status"] = "completed"
        req = FakeRequest(method="POST", path=f"/api/v1/audit/sessions/{_seed_session}/resume")
        resp = await handler.handle_request(req)
        assert resp["status"] == 400

    @pytest.mark.asyncio
    async def test_delete_completed_session_succeeds(self, handler, _seed_session):
        _sessions[_seed_session]["status"] = "completed"
        req = FakeRequest(method="DELETE", path=f"/api/v1/audit/sessions/{_seed_session}")
        resp = await handler.handle_request(req)
        assert resp["status"] == 200

    @pytest.mark.asyncio
    async def test_delete_cancelled_session_succeeds(self, handler, _seed_session):
        _sessions[_seed_session]["status"] = "cancelled"
        req = FakeRequest(method="DELETE", path=f"/api/v1/audit/sessions/{_seed_session}")
        resp = await handler.handle_request(req)
        assert resp["status"] == 200

    @pytest.mark.asyncio
    async def test_start_updates_timestamps(self, handler, _seed_session):
        original_updated = _sessions[_seed_session]["updated_at"]
        req = FakeRequest(method="POST", path=f"/api/v1/audit/sessions/{_seed_session}/start")
        with patch.object(handler, "_run_audit_background", new_callable=AsyncMock):
            await handler.handle_request(req)
        assert _sessions[_seed_session]["updated_at"] != original_updated
        assert _sessions[_seed_session]["started_at"] is not None

    @pytest.mark.asyncio
    async def test_pause_updates_timestamp(self, handler, _seed_session):
        _sessions[_seed_session]["status"] = "running"
        original_updated = _sessions[_seed_session]["updated_at"]
        req = FakeRequest(method="POST", path=f"/api/v1/audit/sessions/{_seed_session}/pause")
        await handler.handle_request(req)
        assert _sessions[_seed_session]["updated_at"] != original_updated

    @pytest.mark.asyncio
    async def test_cancel_sets_completed_at(self, handler, _seed_session):
        _sessions[_seed_session]["status"] = "running"
        req = FakeRequest(
            method="POST",
            path=f"/api/v1/audit/sessions/{_seed_session}/cancel",
            body_data={},
        )
        await handler.handle_request(req)
        assert _sessions[_seed_session]["completed_at"] is not None


# ---------------------------------------------------------------------------
# Findings edge cases
# ---------------------------------------------------------------------------


class TestFindingsEdgeCases:
    @pytest.mark.asyncio
    async def test_findings_unknown_severity_sorted_last(self, handler, _seed_session):
        _findings[_seed_session] = [
            {"id": "f-unknown", "severity": "unknown_level"},
            {"id": "f-critical", "severity": "critical"},
        ]
        req = FakeRequest(method="GET", path=f"/api/v1/audit/sessions/{_seed_session}/findings")
        resp = await handler.handle_request(req)
        body = _parse_body(resp)
        assert body["findings"][0]["id"] == "f-critical"
        assert body["findings"][1]["id"] == "f-unknown"

    @pytest.mark.asyncio
    async def test_findings_multiple_filters_combined(self, handler, _seed_session):
        _findings[_seed_session] = [
            {"id": "f1", "severity": "high", "audit_type": "security", "status": "open"},
            {"id": "f2", "severity": "high", "audit_type": "compliance", "status": "open"},
            {"id": "f3", "severity": "low", "audit_type": "security", "status": "open"},
            {"id": "f4", "severity": "high", "audit_type": "security", "status": "acknowledged"},
        ]
        req = FakeRequest(
            method="GET",
            path=f"/api/v1/audit/sessions/{_seed_session}/findings",
            query_params={"severity": "high", "audit_type": "security", "status": "open"},
        )
        resp = await handler.handle_request(req)
        body = _parse_body(resp)
        assert body["total"] == 1
        assert body["findings"][0]["id"] == "f1"


# ---------------------------------------------------------------------------
# Background audit run
# ---------------------------------------------------------------------------


class TestBackgroundAuditRun:
    """Tests for _run_audit_background.

    We replace the asyncio attribute *on the handler module* so that
    asyncio.sleep becomes an instant no-op without contaminating the real
    event loop (which would break hive_mind monitor loops, etc.).
    """

    @staticmethod
    def _fake_asyncio():
        """Build a fake asyncio namespace that only overrides sleep."""
        fake = MagicMock()
        fake.sleep = AsyncMock(return_value=None)
        fake.create_task = asyncio.get_event_loop().create_task
        fake.Queue = asyncio.Queue
        fake.QueueFull = asyncio.QueueFull
        fake.wait_for = asyncio.wait_for
        fake.TimeoutError = asyncio.TimeoutError
        return fake

    @pytest.mark.asyncio
    async def test_background_completes_session(self, handler, _seed_session):
        _sessions[_seed_session]["status"] = "running"
        with patch.object(audit_mod, "asyncio", self._fake_asyncio()):
            await handler._run_audit_background(_seed_session)
        assert _sessions[_seed_session]["status"] == "completed"
        assert _sessions[_seed_session]["completed_at"] is not None

    @pytest.mark.asyncio
    async def test_background_nonexistent_session_returns(self, handler):
        # Should not raise for missing session
        await handler._run_audit_background("nonexistent-id")

    @pytest.mark.asyncio
    async def test_background_emits_completed_event(self, handler, _seed_session):
        _sessions[_seed_session]["status"] = "running"
        queue = asyncio.Queue()
        _event_queues[_seed_session] = [queue]
        with patch.object(audit_mod, "asyncio", self._fake_asyncio()):
            await handler._run_audit_background(_seed_session)
        events = []
        while not queue.empty():
            events.append(queue.get_nowait())
        event_types = [e["type"] for e in events]
        assert "audit_completed" in event_types

    @pytest.mark.asyncio
    async def test_background_respects_cancelled_status(self, handler, _seed_session):
        _sessions[_seed_session]["status"] = "cancelled"
        with patch.object(audit_mod, "asyncio", self._fake_asyncio()):
            await handler._run_audit_background(_seed_session)
        # Should remain cancelled, not set to completed
        assert _sessions[_seed_session]["status"] == "cancelled"

    @pytest.mark.asyncio
    async def test_background_cleans_up_cancellation_token(self, handler, _seed_session):
        _sessions[_seed_session]["status"] = "running"
        _cancellation_tokens[_seed_session] = MagicMock(is_cancelled=False)
        with patch.object(audit_mod, "asyncio", self._fake_asyncio()):
            await handler._run_audit_background(_seed_session)
        assert _seed_session not in _cancellation_tokens

    @pytest.mark.asyncio
    async def test_background_error_sets_failed_status(self, handler, _seed_session):
        _sessions[_seed_session]["status"] = "running"
        # Trigger the outer except by making _emit_event raise
        original_emit = handler._emit_event

        async def failing_emit(sid, event):
            raise RuntimeError("simulated error")

        handler._emit_event = failing_emit
        await handler._run_audit_background(_seed_session)
        assert _sessions[_seed_session]["status"] == "failed"
        assert _sessions[_seed_session]["error"] == "simulated error"
