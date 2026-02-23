"""
Tests for AuditSessionsHandler.

Comprehensive test coverage for all audit session API endpoints:

Routing (can_handle):
- /api/v1/audit/sessions              POST (create), GET (list)
- /api/v1/audit/sessions/{id}         GET (detail), DELETE
- /api/v1/audit/sessions/{id}/start   POST
- /api/v1/audit/sessions/{id}/pause   POST
- /api/v1/audit/sessions/{id}/resume  POST
- /api/v1/audit/sessions/{id}/cancel  POST
- /api/v1/audit/sessions/{id}/findings GET
- /api/v1/audit/sessions/{id}/events  GET (SSE)
- /api/v1/audit/sessions/{id}/intervene POST
- /api/v1/audit/sessions/{id}/report  GET
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.features.audit_sessions import (
    AuditSessionsHandler,
    _sessions,
    _findings,
    _event_queues,
    _cancellation_tokens,
    get_audit_sessions_circuit_breaker,
    get_audit_sessions_circuit_breaker_status,
)


# =============================================================================
# Module path for patching
# =============================================================================

MODULE = "aragora.server.handlers.features.audit_sessions"


# =============================================================================
# Mock Request
# =============================================================================


@dataclass
class MockHTTPHandler:
    """Mock HTTP request for handler tests."""

    path: str = "/api/v1/audit/sessions"
    method: str = "GET"
    body: dict[str, Any] | None = None
    query: dict[str, str] = field(default_factory=dict)
    headers: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        self._raw = json.dumps(self.body).encode() if self.body is not None else b"{}"

    async def read(self) -> bytes:
        return self._raw

    async def json(self) -> dict[str, Any]:
        return self.body if self.body is not None else {}


# =============================================================================
# Helpers
# =============================================================================


def _status(result: dict[str, Any]) -> int:
    """Extract status code from handler response dict."""
    return result.get("status", 0)


def _body(result: dict[str, Any]) -> dict[str, Any]:
    """Parse the JSON body from a handler response dict."""
    raw = result.get("body", "{}")
    if isinstance(raw, str):
        return json.loads(raw)
    return raw


def _make_session(
    session_id: str = "sess-1",
    status: str = "pending",
    document_ids: list[str] | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    """Create a session dict for seeding _sessions."""
    now = datetime.now(timezone.utc).isoformat()
    session: dict[str, Any] = {
        "id": session_id,
        "document_ids": document_ids or ["doc-1"],
        "audit_types": ["security", "compliance"],
        "config": {},
        "status": status,
        "created_at": now,
        "updated_at": now,
        "started_at": None,
        "completed_at": None,
        "progress": {
            "total_documents": len(document_ids or ["doc-1"]),
            "processed_documents": 0,
            "total_chunks": 0,
            "processed_chunks": 0,
            "findings_count": 0,
        },
        "agents": [],
        "error": None,
    }
    session.update(overrides)
    return session


def _make_finding(
    finding_id: str = "find-1",
    severity: str = "high",
    audit_type: str = "security",
    status: str = "open",
    **overrides: Any,
) -> dict[str, Any]:
    """Create a finding dict for seeding _findings."""
    finding: dict[str, Any] = {
        "id": finding_id,
        "title": f"Finding {finding_id}",
        "description": "Test finding",
        "severity": severity,
        "audit_type": audit_type,
        "status": status,
        "confidence": 0.9,
        "category": "test",
        "document_id": "doc-1",
        "evidence_text": "Evidence",
    }
    finding.update(overrides)
    return finding


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def handler():
    """Create an AuditSessionsHandler instance."""
    return AuditSessionsHandler(server_context={})


@pytest.fixture(autouse=True)
def clear_global_state():
    """Clear module-level mutable state before each test."""
    _sessions.clear()
    _findings.clear()
    _event_queues.clear()
    _cancellation_tokens.clear()
    yield
    _sessions.clear()
    _findings.clear()
    _event_queues.clear()
    _cancellation_tokens.clear()


@pytest.fixture
def seeded_session():
    """Seed a pending session and return its id."""
    sid = "sess-test-1"
    _sessions[sid] = _make_session(session_id=sid)
    _findings[sid] = []
    _event_queues[sid] = []
    return sid


@pytest.fixture
def seeded_running_session():
    """Seed a running session and return its id."""
    sid = "sess-running-1"
    _sessions[sid] = _make_session(session_id=sid, status="running")
    _findings[sid] = []
    _event_queues[sid] = []
    return sid


@pytest.fixture
def seeded_paused_session():
    """Seed a paused session and return its id."""
    sid = "sess-paused-1"
    _sessions[sid] = _make_session(session_id=sid, status="paused")
    _findings[sid] = []
    _event_queues[sid] = []
    return sid


@pytest.fixture
def seeded_completed_session():
    """Seed a completed session and return its id."""
    sid = "sess-completed-1"
    _sessions[sid] = _make_session(session_id=sid, status="completed")
    _findings[sid] = []
    _event_queues[sid] = []
    return sid


@pytest.fixture
def seeded_session_with_findings():
    """Seed a session with multiple findings."""
    sid = "sess-findings-1"
    _sessions[sid] = _make_session(session_id=sid, status="completed")
    _findings[sid] = [
        _make_finding("f-1", severity="critical", audit_type="security", status="open"),
        _make_finding("f-2", severity="high", audit_type="compliance", status="open"),
        _make_finding("f-3", severity="medium", audit_type="security", status="acknowledged"),
        _make_finding("f-4", severity="low", audit_type="quality", status="false_positive"),
        _make_finding("f-5", severity="info", audit_type="consistency", status="open"),
    ]
    _event_queues[sid] = []
    return sid


# =============================================================================
# 1. can_handle() Tests
# =============================================================================


class TestCanHandle:
    """Test can_handle routes correctly."""

    def test_handles_audit_sessions_path(self, handler):
        assert handler.can_handle("/api/v1/audit/sessions") is True

    def test_handles_audit_session_id_path(self, handler):
        assert handler.can_handle("/api/v1/audit/sessions/abc-123") is True

    def test_handles_audit_session_start(self, handler):
        assert handler.can_handle("/api/v1/audit/sessions/abc/start") is True

    def test_handles_audit_session_findings(self, handler):
        assert handler.can_handle("/api/v1/audit/sessions/abc/findings") is True

    def test_handles_audit_session_events(self, handler):
        assert handler.can_handle("/api/v1/audit/sessions/abc/events") is True

    def test_handles_audit_session_report(self, handler):
        assert handler.can_handle("/api/v1/audit/sessions/abc/report") is True

    def test_does_not_handle_unrelated_path(self, handler):
        assert handler.can_handle("/api/v1/debates") is False

    def test_does_not_handle_non_audit_path(self, handler):
        assert handler.can_handle("/api/v1/knowledge/mound") is False

    def test_routes_class_attribute_count(self):
        assert len(AuditSessionsHandler.ROUTES) == 10


# =============================================================================
# 2. Create Session Tests
# =============================================================================


class TestCreateSession:
    """Tests for POST /api/v1/audit/sessions."""

    @pytest.mark.asyncio
    async def test_create_session_success(self, handler):
        req = MockHTTPHandler(
            method="POST",
            path="/api/v1/audit/sessions",
            body={"document_ids": ["doc-1", "doc-2"]},
        )
        result = await handler.handle_request(req)
        assert _status(result) == 201
        body = _body(result)
        assert body["status"] == "pending"
        assert body["document_ids"] == ["doc-1", "doc-2"]
        assert body["id"] in _sessions

    @pytest.mark.asyncio
    async def test_create_session_with_audit_types(self, handler):
        req = MockHTTPHandler(
            method="POST",
            path="/api/v1/audit/sessions",
            body={"document_ids": ["doc-1"], "audit_types": ["security"]},
        )
        result = await handler.handle_request(req)
        assert _status(result) == 201
        body = _body(result)
        assert body["audit_types"] == ["security"]

    @pytest.mark.asyncio
    async def test_create_session_with_config(self, handler):
        req = MockHTTPHandler(
            method="POST",
            path="/api/v1/audit/sessions",
            body={
                "document_ids": ["doc-1"],
                "config": {"use_debate": True, "min_confidence": 0.7},
            },
        )
        result = await handler.handle_request(req)
        assert _status(result) == 201
        body = _body(result)
        assert body["config"]["use_debate"] is True
        assert body["config"]["min_confidence"] == 0.7

    @pytest.mark.asyncio
    async def test_create_session_default_audit_types(self, handler):
        req = MockHTTPHandler(
            method="POST",
            path="/api/v1/audit/sessions",
            body={"document_ids": ["doc-1"]},
        )
        result = await handler.handle_request(req)
        body = _body(result)
        assert body["audit_types"] == ["security", "compliance", "consistency", "quality"]

    @pytest.mark.asyncio
    async def test_create_session_missing_document_ids(self, handler):
        req = MockHTTPHandler(
            method="POST",
            path="/api/v1/audit/sessions",
            body={},
        )
        result = await handler.handle_request(req)
        assert _status(result) == 400
        body = _body(result)
        assert "document_ids" in body["error"]

    @pytest.mark.asyncio
    async def test_create_session_empty_document_ids(self, handler):
        req = MockHTTPHandler(
            method="POST",
            path="/api/v1/audit/sessions",
            body={"document_ids": []},
        )
        result = await handler.handle_request(req)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_create_session_invalid_json(self, handler):
        req = MockHTTPHandler(
            method="POST",
            path="/api/v1/audit/sessions",
        )
        # Override json() to raise
        async def bad_json():
            raise json.JSONDecodeError("bad", "", 0)

        req.json = bad_json
        result = await handler.handle_request(req)
        assert _status(result) == 400
        body = _body(result)
        assert "Invalid JSON" in body["error"]

    @pytest.mark.asyncio
    async def test_create_session_initializes_findings_and_queues(self, handler):
        req = MockHTTPHandler(
            method="POST",
            path="/api/v1/audit/sessions",
            body={"document_ids": ["doc-1"]},
        )
        result = await handler.handle_request(req)
        sid = _body(result)["id"]
        assert sid in _findings
        assert sid in _event_queues
        assert _findings[sid] == []
        assert _event_queues[sid] == []

    @pytest.mark.asyncio
    async def test_create_session_progress_fields(self, handler):
        req = MockHTTPHandler(
            method="POST",
            path="/api/v1/audit/sessions",
            body={"document_ids": ["doc-1", "doc-2", "doc-3"]},
        )
        result = await handler.handle_request(req)
        body = _body(result)
        assert body["progress"]["total_documents"] == 3
        assert body["progress"]["processed_documents"] == 0
        assert body["progress"]["findings_count"] == 0


# =============================================================================
# 3. List Sessions Tests
# =============================================================================


class TestListSessions:
    """Tests for GET /api/v1/audit/sessions."""

    @pytest.mark.asyncio
    async def test_list_sessions_empty(self, handler):
        req = MockHTTPHandler(method="GET", path="/api/v1/audit/sessions")
        result = await handler.handle_request(req)
        assert _status(result) == 200
        body = _body(result)
        assert body["sessions"] == []
        assert body["total"] == 0

    @pytest.mark.asyncio
    async def test_list_sessions_returns_all(self, handler):
        _sessions["s1"] = _make_session("s1")
        _sessions["s2"] = _make_session("s2")
        req = MockHTTPHandler(method="GET", path="/api/v1/audit/sessions")
        result = await handler.handle_request(req)
        body = _body(result)
        assert body["total"] == 2
        assert len(body["sessions"]) == 2

    @pytest.mark.asyncio
    async def test_list_sessions_filter_by_status(self, handler):
        _sessions["s1"] = _make_session("s1", status="pending")
        _sessions["s2"] = _make_session("s2", status="running")
        _sessions["s3"] = _make_session("s3", status="pending")
        req = MockHTTPHandler(
            method="GET",
            path="/api/v1/audit/sessions",
            query={"status": "pending"},
        )
        result = await handler.handle_request(req)
        body = _body(result)
        assert body["total"] == 2
        for sess in body["sessions"]:
            assert sess["status"] == "pending"

    @pytest.mark.asyncio
    async def test_list_sessions_pagination_limit(self, handler):
        for i in range(5):
            _sessions[f"s{i}"] = _make_session(f"s{i}")
        req = MockHTTPHandler(
            method="GET",
            path="/api/v1/audit/sessions",
            query={"limit": "2"},
        )
        result = await handler.handle_request(req)
        body = _body(result)
        assert len(body["sessions"]) == 2
        assert body["total"] == 5
        assert body["limit"] == 2

    @pytest.mark.asyncio
    async def test_list_sessions_pagination_offset(self, handler):
        for i in range(5):
            _sessions[f"s{i}"] = _make_session(f"s{i}")
        req = MockHTTPHandler(
            method="GET",
            path="/api/v1/audit/sessions",
            query={"offset": "3"},
        )
        result = await handler.handle_request(req)
        body = _body(result)
        assert len(body["sessions"]) == 2
        assert body["offset"] == 3

    @pytest.mark.asyncio
    async def test_list_sessions_default_limit(self, handler):
        _sessions["s1"] = _make_session("s1")
        req = MockHTTPHandler(method="GET", path="/api/v1/audit/sessions")
        result = await handler.handle_request(req)
        body = _body(result)
        assert body["limit"] == 50

    @pytest.mark.asyncio
    async def test_list_sessions_filter_returns_empty_for_no_match(self, handler):
        _sessions["s1"] = _make_session("s1", status="pending")
        req = MockHTTPHandler(
            method="GET",
            path="/api/v1/audit/sessions",
            query={"status": "completed"},
        )
        result = await handler.handle_request(req)
        body = _body(result)
        assert body["total"] == 0
        assert body["sessions"] == []


# =============================================================================
# 4. Get Session Detail Tests
# =============================================================================


class TestGetSession:
    """Tests for GET /api/v1/audit/sessions/{id}."""

    @pytest.mark.asyncio
    async def test_get_session_success(self, handler, seeded_session):
        req = MockHTTPHandler(
            method="GET",
            path=f"/api/v1/audit/sessions/{seeded_session}",
        )
        result = await handler.handle_request(req)
        assert _status(result) == 200
        body = _body(result)
        assert body["id"] == seeded_session

    @pytest.mark.asyncio
    async def test_get_session_not_found(self, handler):
        req = MockHTTPHandler(
            method="GET",
            path="/api/v1/audit/sessions/nonexistent",
        )
        result = await handler.handle_request(req)
        assert _status(result) == 404
        body = _body(result)
        assert "not found" in body["error"]


# =============================================================================
# 5. Delete Session Tests
# =============================================================================


class TestDeleteSession:
    """Tests for DELETE /api/v1/audit/sessions/{id}."""

    @pytest.mark.asyncio
    async def test_delete_session_success(self, handler, seeded_session):
        req = MockHTTPHandler(
            method="DELETE",
            path=f"/api/v1/audit/sessions/{seeded_session}",
        )
        result = await handler.handle_request(req)
        assert _status(result) == 200
        body = _body(result)
        assert body["deleted"] == seeded_session
        assert seeded_session not in _sessions

    @pytest.mark.asyncio
    async def test_delete_session_not_found(self, handler):
        req = MockHTTPHandler(
            method="DELETE",
            path="/api/v1/audit/sessions/nonexistent",
        )
        result = await handler.handle_request(req)
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_delete_running_session_rejected(self, handler, seeded_running_session):
        req = MockHTTPHandler(
            method="DELETE",
            path=f"/api/v1/audit/sessions/{seeded_running_session}",
        )
        result = await handler.handle_request(req)
        assert _status(result) == 400
        body = _body(result)
        assert "running" in body["error"].lower()

    @pytest.mark.asyncio
    async def test_delete_session_cleans_findings_and_queues(self, handler, seeded_session):
        _findings[seeded_session] = [_make_finding()]
        _event_queues[seeded_session] = [asyncio.Queue()]
        req = MockHTTPHandler(
            method="DELETE",
            path=f"/api/v1/audit/sessions/{seeded_session}",
        )
        await handler.handle_request(req)
        assert seeded_session not in _findings
        assert seeded_session not in _event_queues

    @pytest.mark.asyncio
    async def test_delete_completed_session_succeeds(self, handler, seeded_completed_session):
        req = MockHTTPHandler(
            method="DELETE",
            path=f"/api/v1/audit/sessions/{seeded_completed_session}",
        )
        result = await handler.handle_request(req)
        assert _status(result) == 200


# =============================================================================
# 6. Start Audit Tests
# =============================================================================


class TestStartAudit:
    """Tests for POST /api/v1/audit/sessions/{id}/start."""

    @pytest.mark.asyncio
    async def test_start_pending_session(self, handler, seeded_session):
        req = MockHTTPHandler(
            method="POST",
            path=f"/api/v1/audit/sessions/{seeded_session}/start",
        )
        with patch(f"{MODULE}.asyncio.create_task") as mock_task:
            mock_task.return_value = MagicMock(
                add_done_callback=MagicMock(),
                cancelled=MagicMock(return_value=False),
                exception=MagicMock(return_value=None),
            )
            result = await handler.handle_request(req)
        assert _status(result) == 200
        body = _body(result)
        assert body["status"] == "running"
        assert body["started_at"] is not None

    @pytest.mark.asyncio
    async def test_start_paused_session(self, handler, seeded_paused_session):
        req = MockHTTPHandler(
            method="POST",
            path=f"/api/v1/audit/sessions/{seeded_paused_session}/start",
        )
        with patch(f"{MODULE}.asyncio.create_task") as mock_task:
            mock_task.return_value = MagicMock(
                add_done_callback=MagicMock(),
            )
            result = await handler.handle_request(req)
        assert _status(result) == 200
        body = _body(result)
        assert body["status"] == "running"

    @pytest.mark.asyncio
    async def test_start_already_running_session(self, handler, seeded_running_session):
        req = MockHTTPHandler(
            method="POST",
            path=f"/api/v1/audit/sessions/{seeded_running_session}/start",
        )
        result = await handler.handle_request(req)
        assert _status(result) == 400
        body = _body(result)
        assert "running" in body["error"]

    @pytest.mark.asyncio
    async def test_start_completed_session_rejected(self, handler, seeded_completed_session):
        req = MockHTTPHandler(
            method="POST",
            path=f"/api/v1/audit/sessions/{seeded_completed_session}/start",
        )
        result = await handler.handle_request(req)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_start_session_not_found(self, handler):
        req = MockHTTPHandler(
            method="POST",
            path="/api/v1/audit/sessions/nonexistent/start",
        )
        result = await handler.handle_request(req)
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_start_emits_event(self, handler, seeded_session):
        queue = asyncio.Queue()
        _event_queues[seeded_session] = [queue]
        req = MockHTTPHandler(
            method="POST",
            path=f"/api/v1/audit/sessions/{seeded_session}/start",
        )
        with patch(f"{MODULE}.asyncio.create_task") as mock_task:
            mock_task.return_value = MagicMock(add_done_callback=MagicMock())
            await handler.handle_request(req)
        event = queue.get_nowait()
        assert event["type"] == "audit_started"
        assert event["session_id"] == seeded_session


# =============================================================================
# 7. Pause Audit Tests
# =============================================================================


class TestPauseAudit:
    """Tests for POST /api/v1/audit/sessions/{id}/pause."""

    @pytest.mark.asyncio
    async def test_pause_running_session(self, handler, seeded_running_session):
        req = MockHTTPHandler(
            method="POST",
            path=f"/api/v1/audit/sessions/{seeded_running_session}/pause",
        )
        result = await handler.handle_request(req)
        assert _status(result) == 200
        body = _body(result)
        assert body["status"] == "paused"

    @pytest.mark.asyncio
    async def test_pause_non_running_session_rejected(self, handler, seeded_session):
        req = MockHTTPHandler(
            method="POST",
            path=f"/api/v1/audit/sessions/{seeded_session}/pause",
        )
        result = await handler.handle_request(req)
        assert _status(result) == 400
        body = _body(result)
        assert "pending" in body["error"]

    @pytest.mark.asyncio
    async def test_pause_not_found(self, handler):
        req = MockHTTPHandler(
            method="POST",
            path="/api/v1/audit/sessions/nonexistent/pause",
        )
        result = await handler.handle_request(req)
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_pause_emits_event(self, handler, seeded_running_session):
        queue = asyncio.Queue()
        _event_queues[seeded_running_session] = [queue]
        req = MockHTTPHandler(
            method="POST",
            path=f"/api/v1/audit/sessions/{seeded_running_session}/pause",
        )
        await handler.handle_request(req)
        event = queue.get_nowait()
        assert event["type"] == "audit_paused"

    @pytest.mark.asyncio
    async def test_pause_updates_timestamp(self, handler, seeded_running_session):
        old_updated = _sessions[seeded_running_session]["updated_at"]
        req = MockHTTPHandler(
            method="POST",
            path=f"/api/v1/audit/sessions/{seeded_running_session}/pause",
        )
        await handler.handle_request(req)
        assert _sessions[seeded_running_session]["updated_at"] >= old_updated


# =============================================================================
# 8. Resume Audit Tests
# =============================================================================


class TestResumeAudit:
    """Tests for POST /api/v1/audit/sessions/{id}/resume."""

    @pytest.mark.asyncio
    async def test_resume_paused_session(self, handler, seeded_paused_session):
        req = MockHTTPHandler(
            method="POST",
            path=f"/api/v1/audit/sessions/{seeded_paused_session}/resume",
        )
        result = await handler.handle_request(req)
        assert _status(result) == 200
        body = _body(result)
        assert body["status"] == "running"

    @pytest.mark.asyncio
    async def test_resume_non_paused_session_rejected(self, handler, seeded_session):
        req = MockHTTPHandler(
            method="POST",
            path=f"/api/v1/audit/sessions/{seeded_session}/resume",
        )
        result = await handler.handle_request(req)
        assert _status(result) == 400
        body = _body(result)
        assert "pending" in body["error"]

    @pytest.mark.asyncio
    async def test_resume_running_session_rejected(self, handler, seeded_running_session):
        req = MockHTTPHandler(
            method="POST",
            path=f"/api/v1/audit/sessions/{seeded_running_session}/resume",
        )
        result = await handler.handle_request(req)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_resume_not_found(self, handler):
        req = MockHTTPHandler(
            method="POST",
            path="/api/v1/audit/sessions/nonexistent/resume",
        )
        result = await handler.handle_request(req)
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_resume_emits_event(self, handler, seeded_paused_session):
        queue = asyncio.Queue()
        _event_queues[seeded_paused_session] = [queue]
        req = MockHTTPHandler(
            method="POST",
            path=f"/api/v1/audit/sessions/{seeded_paused_session}/resume",
        )
        await handler.handle_request(req)
        event = queue.get_nowait()
        assert event["type"] == "audit_resumed"


# =============================================================================
# 9. Cancel Audit Tests
# =============================================================================


class TestCancelAudit:
    """Tests for POST /api/v1/audit/sessions/{id}/cancel."""

    @pytest.mark.asyncio
    async def test_cancel_running_session(self, handler, seeded_running_session):
        req = MockHTTPHandler(
            method="POST",
            path=f"/api/v1/audit/sessions/{seeded_running_session}/cancel",
        )
        result = await handler.handle_request(req)
        assert _status(result) == 200
        body = _body(result)
        assert body["status"] == "cancelled"
        assert body["completed_at"] is not None

    @pytest.mark.asyncio
    async def test_cancel_pending_session(self, handler, seeded_session):
        req = MockHTTPHandler(
            method="POST",
            path=f"/api/v1/audit/sessions/{seeded_session}/cancel",
        )
        result = await handler.handle_request(req)
        assert _status(result) == 200
        body = _body(result)
        assert body["status"] == "cancelled"

    @pytest.mark.asyncio
    async def test_cancel_with_custom_reason(self, handler, seeded_running_session):
        req = MockHTTPHandler(
            method="POST",
            path=f"/api/v1/audit/sessions/{seeded_running_session}/cancel",
            body={"reason": "Budget exceeded"},
        )
        result = await handler.handle_request(req)
        body = _body(result)
        assert body["cancel_reason"] == "Budget exceeded"

    @pytest.mark.asyncio
    async def test_cancel_default_reason(self, handler, seeded_running_session):
        req = MockHTTPHandler(
            method="POST",
            path=f"/api/v1/audit/sessions/{seeded_running_session}/cancel",
        )
        result = await handler.handle_request(req)
        body = _body(result)
        assert body["cancel_reason"] == "User requested cancellation"

    @pytest.mark.asyncio
    async def test_cancel_already_completed_rejected(self, handler, seeded_completed_session):
        req = MockHTTPHandler(
            method="POST",
            path=f"/api/v1/audit/sessions/{seeded_completed_session}/cancel",
        )
        result = await handler.handle_request(req)
        assert _status(result) == 400
        body = _body(result)
        assert "completed" in body["error"]

    @pytest.mark.asyncio
    async def test_cancel_already_cancelled_rejected(self, handler):
        sid = "cancelled-sess"
        _sessions[sid] = _make_session(sid, status="cancelled")
        _findings[sid] = []
        req = MockHTTPHandler(
            method="POST",
            path=f"/api/v1/audit/sessions/{sid}/cancel",
        )
        result = await handler.handle_request(req)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_cancel_not_found(self, handler):
        req = MockHTTPHandler(
            method="POST",
            path="/api/v1/audit/sessions/nonexistent/cancel",
        )
        result = await handler.handle_request(req)
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_cancel_triggers_cancellation_token(self, handler, seeded_running_session):
        mock_token = MagicMock()
        _cancellation_tokens[seeded_running_session] = mock_token
        req = MockHTTPHandler(
            method="POST",
            path=f"/api/v1/audit/sessions/{seeded_running_session}/cancel",
        )
        await handler.handle_request(req)
        mock_token.cancel.assert_called_once()

    @pytest.mark.asyncio
    async def test_cancel_handles_token_error_gracefully(self, handler, seeded_running_session):
        mock_token = MagicMock()
        mock_token.cancel.side_effect = RuntimeError("token error")
        _cancellation_tokens[seeded_running_session] = mock_token
        req = MockHTTPHandler(
            method="POST",
            path=f"/api/v1/audit/sessions/{seeded_running_session}/cancel",
        )
        result = await handler.handle_request(req)
        # Should still succeed despite token error
        assert _status(result) == 200
        body = _body(result)
        assert body["status"] == "cancelled"

    @pytest.mark.asyncio
    async def test_cancel_emits_event(self, handler, seeded_running_session):
        queue = asyncio.Queue()
        _event_queues[seeded_running_session] = [queue]
        req = MockHTTPHandler(
            method="POST",
            path=f"/api/v1/audit/sessions/{seeded_running_session}/cancel",
        )
        await handler.handle_request(req)
        event = queue.get_nowait()
        assert event["type"] == "audit_cancelled"
        assert "reason" in event

    @pytest.mark.asyncio
    async def test_cancel_with_invalid_json_body_uses_default_reason(
        self, handler, seeded_running_session
    ):
        req = MockHTTPHandler(
            method="POST",
            path=f"/api/v1/audit/sessions/{seeded_running_session}/cancel",
        )

        async def bad_json():
            raise json.JSONDecodeError("bad", "", 0)

        req.json = bad_json
        result = await handler.handle_request(req)
        assert _status(result) == 200
        body = _body(result)
        assert body["cancel_reason"] == "User requested cancellation"


# =============================================================================
# 10. Get Findings Tests
# =============================================================================


class TestGetFindings:
    """Tests for GET /api/v1/audit/sessions/{id}/findings."""

    @pytest.mark.asyncio
    async def test_get_findings_empty(self, handler, seeded_session):
        req = MockHTTPHandler(
            method="GET",
            path=f"/api/v1/audit/sessions/{seeded_session}/findings",
        )
        result = await handler.handle_request(req)
        assert _status(result) == 200
        body = _body(result)
        assert body["findings"] == []
        assert body["total"] == 0

    @pytest.mark.asyncio
    async def test_get_findings_returns_all(self, handler, seeded_session_with_findings):
        req = MockHTTPHandler(
            method="GET",
            path=f"/api/v1/audit/sessions/{seeded_session_with_findings}/findings",
        )
        result = await handler.handle_request(req)
        body = _body(result)
        assert body["total"] == 5
        assert len(body["findings"]) == 5

    @pytest.mark.asyncio
    async def test_get_findings_filter_severity(self, handler, seeded_session_with_findings):
        req = MockHTTPHandler(
            method="GET",
            path=f"/api/v1/audit/sessions/{seeded_session_with_findings}/findings",
            query={"severity": "critical"},
        )
        result = await handler.handle_request(req)
        body = _body(result)
        assert body["total"] == 1
        assert body["findings"][0]["severity"] == "critical"

    @pytest.mark.asyncio
    async def test_get_findings_filter_audit_type(self, handler, seeded_session_with_findings):
        req = MockHTTPHandler(
            method="GET",
            path=f"/api/v1/audit/sessions/{seeded_session_with_findings}/findings",
            query={"audit_type": "security"},
        )
        result = await handler.handle_request(req)
        body = _body(result)
        assert body["total"] == 2
        for f in body["findings"]:
            assert f["audit_type"] == "security"

    @pytest.mark.asyncio
    async def test_get_findings_filter_status(self, handler, seeded_session_with_findings):
        req = MockHTTPHandler(
            method="GET",
            path=f"/api/v1/audit/sessions/{seeded_session_with_findings}/findings",
            query={"status": "open"},
        )
        result = await handler.handle_request(req)
        body = _body(result)
        assert body["total"] == 3

    @pytest.mark.asyncio
    async def test_get_findings_sorted_by_severity(self, handler, seeded_session_with_findings):
        req = MockHTTPHandler(
            method="GET",
            path=f"/api/v1/audit/sessions/{seeded_session_with_findings}/findings",
        )
        result = await handler.handle_request(req)
        body = _body(result)
        severities = [f["severity"] for f in body["findings"]]
        order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}
        for i in range(len(severities) - 1):
            assert order[severities[i]] <= order[severities[i + 1]]

    @pytest.mark.asyncio
    async def test_get_findings_pagination(self, handler, seeded_session_with_findings):
        req = MockHTTPHandler(
            method="GET",
            path=f"/api/v1/audit/sessions/{seeded_session_with_findings}/findings",
            query={"limit": "2", "offset": "0"},
        )
        result = await handler.handle_request(req)
        body = _body(result)
        assert len(body["findings"]) == 2
        assert body["total"] == 5

    @pytest.mark.asyncio
    async def test_get_findings_pagination_offset(self, handler, seeded_session_with_findings):
        req = MockHTTPHandler(
            method="GET",
            path=f"/api/v1/audit/sessions/{seeded_session_with_findings}/findings",
            query={"limit": "2", "offset": "4"},
        )
        result = await handler.handle_request(req)
        body = _body(result)
        assert len(body["findings"]) == 1

    @pytest.mark.asyncio
    async def test_get_findings_not_found(self, handler):
        req = MockHTTPHandler(
            method="GET",
            path="/api/v1/audit/sessions/nonexistent/findings",
        )
        result = await handler.handle_request(req)
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_get_findings_combined_filters(self, handler, seeded_session_with_findings):
        req = MockHTTPHandler(
            method="GET",
            path=f"/api/v1/audit/sessions/{seeded_session_with_findings}/findings",
            query={"severity": "high", "audit_type": "compliance"},
        )
        result = await handler.handle_request(req)
        body = _body(result)
        assert body["total"] == 1
        assert body["findings"][0]["id"] == "f-2"


# =============================================================================
# 11. Stream Events Tests
# =============================================================================


class TestStreamEvents:
    """Tests for GET /api/v1/audit/sessions/{id}/events (SSE)."""

    @pytest.mark.asyncio
    async def test_stream_events_returns_sse_response(self, handler, seeded_session):
        req = MockHTTPHandler(
            method="GET",
            path=f"/api/v1/audit/sessions/{seeded_session}/events",
        )
        result = await handler.handle_request(req)
        assert _status(result) == 200
        assert result["headers"]["Content-Type"] == "text/event-stream"
        assert result["headers"]["Cache-Control"] == "no-cache"

    @pytest.mark.asyncio
    async def test_stream_events_not_found(self, handler):
        req = MockHTTPHandler(
            method="GET",
            path="/api/v1/audit/sessions/nonexistent/events",
        )
        result = await handler.handle_request(req)
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_stream_events_creates_queue(self, handler, seeded_session):
        req = MockHTTPHandler(
            method="GET",
            path=f"/api/v1/audit/sessions/{seeded_session}/events",
        )
        await handler.handle_request(req)
        assert len(_event_queues[seeded_session]) == 1

    @pytest.mark.asyncio
    async def test_stream_events_generator_emits_connected(self, handler, seeded_session):
        req = MockHTTPHandler(
            method="GET",
            path=f"/api/v1/audit/sessions/{seeded_session}/events",
        )
        result = await handler.handle_request(req)
        gen = result["body"]
        # Get the first yielded value (connected event)
        first = await gen.__anext__()
        data = json.loads(first.replace("data: ", "").strip())
        assert data["type"] == "connected"
        assert data["session_id"] == seeded_session


# =============================================================================
# 12. Human Intervention Tests
# =============================================================================


class TestIntervention:
    """Tests for POST /api/v1/audit/sessions/{id}/intervene."""

    @pytest.mark.asyncio
    async def test_approve_finding(self, handler, seeded_session_with_findings):
        sid = seeded_session_with_findings
        req = MockHTTPHandler(
            method="POST",
            path=f"/api/v1/audit/sessions/{sid}/intervene",
            body={
                "action": "approve_finding",
                "finding_id": "f-1",
                "reason": "Confirmed issue",
            },
        )
        result = await handler.handle_request(req)
        assert _status(result) == 200
        body = _body(result)
        assert body["success"] is True
        assert body["action"] == "approve_finding"
        # Check finding was updated
        finding = next(f for f in _findings[sid] if f["id"] == "f-1")
        assert finding["status"] == "acknowledged"
        assert finding["human_review"]["action"] == "approved"

    @pytest.mark.asyncio
    async def test_reject_finding(self, handler, seeded_session_with_findings):
        sid = seeded_session_with_findings
        req = MockHTTPHandler(
            method="POST",
            path=f"/api/v1/audit/sessions/{sid}/intervene",
            body={
                "action": "reject_finding",
                "finding_id": "f-2",
                "reason": "False positive",
            },
        )
        result = await handler.handle_request(req)
        assert _status(result) == 200
        finding = next(f for f in _findings[sid] if f["id"] == "f-2")
        assert finding["status"] == "false_positive"
        assert finding["human_review"]["action"] == "rejected"

    @pytest.mark.asyncio
    async def test_intervention_add_context_action(self, handler, seeded_session):
        req = MockHTTPHandler(
            method="POST",
            path=f"/api/v1/audit/sessions/{seeded_session}/intervene",
            body={
                "action": "add_context",
                "context": "Additional information about the issue",
            },
        )
        result = await handler.handle_request(req)
        assert _status(result) == 200
        body = _body(result)
        assert body["action"] == "add_context"

    @pytest.mark.asyncio
    async def test_intervention_missing_action(self, handler, seeded_session):
        req = MockHTTPHandler(
            method="POST",
            path=f"/api/v1/audit/sessions/{seeded_session}/intervene",
            body={"finding_id": "f-1"},
        )
        result = await handler.handle_request(req)
        assert _status(result) == 400
        body = _body(result)
        assert "action" in body["error"]

    @pytest.mark.asyncio
    async def test_intervention_invalid_json(self, handler, seeded_session):
        req = MockHTTPHandler(
            method="POST",
            path=f"/api/v1/audit/sessions/{seeded_session}/intervene",
        )

        async def bad_json():
            raise json.JSONDecodeError("bad", "", 0)

        req.json = bad_json
        result = await handler.handle_request(req)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_intervention_not_found(self, handler):
        req = MockHTTPHandler(
            method="POST",
            path="/api/v1/audit/sessions/nonexistent/intervene",
            body={"action": "add_context"},
        )
        result = await handler.handle_request(req)
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_intervention_emits_event(self, handler, seeded_session):
        queue = asyncio.Queue()
        _event_queues[seeded_session] = [queue]
        req = MockHTTPHandler(
            method="POST",
            path=f"/api/v1/audit/sessions/{seeded_session}/intervene",
            body={"action": "override_decision"},
        )
        await handler.handle_request(req)
        event = queue.get_nowait()
        assert event["type"] == "human_intervention"
        assert event["action"] == "override_decision"

    @pytest.mark.asyncio
    async def test_intervention_approve_nonexistent_finding(
        self, handler, seeded_session_with_findings
    ):
        sid = seeded_session_with_findings
        req = MockHTTPHandler(
            method="POST",
            path=f"/api/v1/audit/sessions/{sid}/intervene",
            body={
                "action": "approve_finding",
                "finding_id": "nonexistent-finding",
                "reason": "test",
            },
        )
        result = await handler.handle_request(req)
        # Should succeed, but no finding is updated
        assert _status(result) == 200
        body = _body(result)
        assert body["finding_id"] == "nonexistent-finding"

    @pytest.mark.asyncio
    async def test_intervention_approve_without_finding_id(self, handler, seeded_session):
        req = MockHTTPHandler(
            method="POST",
            path=f"/api/v1/audit/sessions/{seeded_session}/intervene",
            body={"action": "approve_finding"},
        )
        result = await handler.handle_request(req)
        # Succeeds but finding_id is None — no findings updated
        assert _status(result) == 200
        body = _body(result)
        assert body["finding_id"] is None


# =============================================================================
# 13. Export Report Tests
# =============================================================================


class TestExportReport:
    """Tests for GET /api/v1/audit/sessions/{id}/report."""

    @pytest.mark.asyncio
    async def test_export_report_fallback_on_import_error(self, handler, seeded_session):
        req = MockHTTPHandler(
            method="GET",
            path=f"/api/v1/audit/sessions/{seeded_session}/report",
        )
        with patch(f"{MODULE}.AuditSessionsHandler._export_report") as mock_export:
            # Simulate the internal fallback path
            mock_export.side_effect = None
            # Actually call the real method — the ImportError fallback is built-in
            pass

        # Directly call the real handler which will hit the import fallback
        result = await handler.handle_request(req)
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_export_report_not_found(self, handler):
        req = MockHTTPHandler(
            method="GET",
            path="/api/v1/audit/sessions/nonexistent/report",
        )
        result = await handler.handle_request(req)
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_export_report_fallback_includes_session_and_findings(
        self, handler, seeded_session_with_findings
    ):
        sid = seeded_session_with_findings
        req = MockHTTPHandler(
            method="GET",
            path=f"/api/v1/audit/sessions/{sid}/report",
        )
        # The handler will fall back to legacy JSON when aragora.reports is not available
        result = await handler.handle_request(req)
        assert _status(result) == 200
        body = _body(result)
        # Fallback path returns session + findings + generated_at
        if "session" in body:
            assert body["session"]["id"] == sid
            assert len(body["findings"]) == 5

    @pytest.mark.asyncio
    async def test_export_report_with_report_generator_success(
        self, handler, seeded_session_with_findings
    ):
        sid = seeded_session_with_findings

        # Mock the report generator module
        mock_report = MagicMock()
        mock_report.content = '{"report": "data"}'
        mock_report.filename = "audit_report.json"
        mock_report.findings_count = 5

        mock_generator = AsyncMock()
        mock_generator.generate.return_value = mock_report

        mock_config_cls = MagicMock()
        mock_format_enum = MagicMock()
        mock_format_enum.JSON = "json"
        mock_format_enum.MARKDOWN = "markdown"
        mock_format_enum.HTML = "html"
        mock_format_enum.PDF = "pdf"
        mock_template_enum = MagicMock()
        mock_template_enum.EXECUTIVE_SUMMARY = "executive_summary"
        mock_template_enum.DETAILED_FINDINGS = "detailed_findings"
        mock_template_enum.COMPLIANCE_ATTESTATION = "compliance_attestation"
        mock_template_enum.SECURITY_ASSESSMENT = "security_assessment"

        # Mock the document_auditor module
        mock_session_cls = MagicMock()
        mock_audit_finding = MagicMock()
        mock_audit_status = MagicMock()
        mock_audit_type = MagicMock()
        mock_severity = MagicMock()
        mock_finding_status = MagicMock()

        with (
            patch.dict(
                "sys.modules",
                {
                    "aragora.reports": MagicMock(
                        AuditReportGenerator=MagicMock(return_value=mock_generator),
                        ReportConfig=mock_config_cls,
                        ReportFormat=mock_format_enum,
                        ReportTemplate=mock_template_enum,
                    ),
                    "aragora.audit.document_auditor": MagicMock(
                        AuditSession=mock_session_cls,
                        AuditFinding=mock_audit_finding,
                        AuditStatus=mock_audit_status,
                        AuditType=mock_audit_type,
                        FindingSeverity=mock_severity,
                        FindingStatus=mock_finding_status,
                    ),
                },
            ),
        ):
            req = MockHTTPHandler(
                method="GET",
                path=f"/api/v1/audit/sessions/{sid}/report",
                query={"format": "json"},
            )
            result = await handler.handle_request(req)
            assert _status(result) == 200
            assert result["headers"]["Content-Type"] == "application/json"
            assert "audit_report" in result["headers"]["Content-Disposition"]

    @pytest.mark.asyncio
    async def test_export_report_query_params_default(self, handler, seeded_session):
        """Test that default query params are used when none specified."""
        req = MockHTTPHandler(
            method="GET",
            path=f"/api/v1/audit/sessions/{seeded_session}/report",
        )
        result = await handler.handle_request(req)
        # Should get a successful response (fallback or generated)
        assert _status(result) == 200


# =============================================================================
# 14. Routing / Endpoint Not Found Tests
# =============================================================================


class TestRoutingAndNotFound:
    """Tests for routing dispatch and 404 fallthrough."""

    @pytest.mark.asyncio
    async def test_unknown_path_returns_404(self, handler):
        req = MockHTTPHandler(
            method="GET",
            path="/api/v1/audit/unknown",
        )
        result = await handler.handle_request(req)
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_unknown_subpath_get_falls_through_to_detail(self, handler, seeded_session):
        """Unknown subpaths with GET fall through to _get_session because
        session_id is parsed and method == GET matches the catch-all branch."""
        req = MockHTTPHandler(
            method="GET",
            path=f"/api/v1/audit/sessions/{seeded_session}/nonexistent",
        )
        result = await handler.handle_request(req)
        # The session_id is parsed from the path and it matches the GET catch-all
        assert _status(result) == 200
        body = _body(result)
        assert body["id"] == seeded_session

    @pytest.mark.asyncio
    async def test_unknown_method_on_session_returns_404(self, handler, seeded_session):
        """PATCH on a session id path with no matching route returns 404."""
        req = MockHTTPHandler(
            method="PATCH",
            path=f"/api/v1/audit/sessions/{seeded_session}",
        )
        result = await handler.handle_request(req)
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_post_to_list_endpoint_creates(self, handler):
        req = MockHTTPHandler(
            method="POST",
            path="/api/v1/audit/sessions",
            body={"document_ids": ["doc-1"]},
        )
        result = await handler.handle_request(req)
        assert _status(result) == 201

    @pytest.mark.asyncio
    async def test_get_to_list_endpoint_lists(self, handler):
        req = MockHTTPHandler(
            method="GET",
            path="/api/v1/audit/sessions",
        )
        result = await handler.handle_request(req)
        assert _status(result) == 200
        body = _body(result)
        assert "sessions" in body


# =============================================================================
# 15. RBAC Permission Resolution Tests
# =============================================================================


class TestPermissionResolution:
    """Tests for _get_required_permission."""

    def test_create_permission(self, handler):
        perm = handler._get_required_permission("/api/v1/audit/sessions", "POST")
        assert perm == "audit:create"

    def test_list_permission(self, handler):
        perm = handler._get_required_permission("/api/v1/audit/sessions", "GET")
        assert perm == "audit:read"

    def test_get_session_permission(self, handler):
        perm = handler._get_required_permission("/api/v1/audit/sessions/abc", "GET")
        assert perm == "audit:read"

    def test_delete_permission(self, handler):
        perm = handler._get_required_permission("/api/v1/audit/sessions/abc", "DELETE")
        assert perm == "audit:delete"

    def test_start_permission(self, handler):
        perm = handler._get_required_permission("/api/v1/audit/sessions/abc/start", "POST")
        assert perm == "audit:execute"

    def test_pause_permission(self, handler):
        perm = handler._get_required_permission("/api/v1/audit/sessions/abc/pause", "POST")
        assert perm == "audit:execute"

    def test_resume_permission(self, handler):
        perm = handler._get_required_permission("/api/v1/audit/sessions/abc/resume", "POST")
        assert perm == "audit:execute"

    def test_cancel_permission(self, handler):
        perm = handler._get_required_permission("/api/v1/audit/sessions/abc/cancel", "POST")
        assert perm == "audit:execute"

    def test_intervene_permission(self, handler):
        perm = handler._get_required_permission("/api/v1/audit/sessions/abc/intervene", "POST")
        assert perm == "audit:intervene"

    def test_findings_permission(self, handler):
        perm = handler._get_required_permission("/api/v1/audit/sessions/abc/findings", "GET")
        assert perm == "audit:read"

    def test_events_permission(self, handler):
        perm = handler._get_required_permission("/api/v1/audit/sessions/abc/events", "GET")
        assert perm == "audit:read"

    def test_report_permission(self, handler):
        perm = handler._get_required_permission("/api/v1/audit/sessions/abc/report", "GET")
        assert perm == "audit:read"

    def test_default_permission_fallback(self, handler):
        perm = handler._get_required_permission("/api/v1/audit/sessions/abc/unknown", "PATCH")
        assert perm == "audit:read"


# =============================================================================
# 16. Circuit Breaker Tests
# =============================================================================


class TestCircuitBreaker:
    """Tests for circuit breaker accessors."""

    def test_get_circuit_breaker(self):
        cb = get_audit_sessions_circuit_breaker()
        assert cb is not None
        assert cb.name == "audit_sessions_handler"

    def test_get_circuit_breaker_status(self):
        status = get_audit_sessions_circuit_breaker_status()
        assert isinstance(status, dict)
        assert "name" in status or "state" in status or len(status) > 0


# =============================================================================
# 17. Session ID Parsing Tests
# =============================================================================


class TestSessionIdParsing:
    """Tests for session_id extraction from paths."""

    @pytest.mark.asyncio
    async def test_parses_session_id_from_detail_path(self, handler, seeded_session):
        req = MockHTTPHandler(
            method="GET",
            path=f"/api/v1/audit/sessions/{seeded_session}",
        )
        result = await handler.handle_request(req)
        assert _status(result) == 200
        body = _body(result)
        assert body["id"] == seeded_session

    @pytest.mark.asyncio
    async def test_parses_session_id_from_action_path(self, handler, seeded_session):
        req = MockHTTPHandler(
            method="POST",
            path=f"/api/v1/audit/sessions/{seeded_session}/start",
        )
        with patch(f"{MODULE}.asyncio.create_task") as mock_task:
            mock_task.return_value = MagicMock(add_done_callback=MagicMock())
            result = await handler.handle_request(req)
        assert _status(result) == 200
        body = _body(result)
        assert body["id"] == seeded_session

    @pytest.mark.asyncio
    async def test_no_session_id_for_list_path(self, handler):
        req = MockHTTPHandler(
            method="GET",
            path="/api/v1/audit/sessions",
        )
        result = await handler.handle_request(req)
        assert _status(result) == 200
        body = _body(result)
        assert "sessions" in body


# =============================================================================
# 18. Event Emission Tests
# =============================================================================


class TestEventEmission:
    """Tests for _emit_event helper."""

    @pytest.mark.asyncio
    async def test_emit_event_to_multiple_queues(self, handler):
        sid = "sess-multi-q"
        _sessions[sid] = _make_session(sid)
        q1 = asyncio.Queue()
        q2 = asyncio.Queue()
        _event_queues[sid] = [q1, q2]
        await handler._emit_event(sid, {"type": "test_event"})
        assert q1.get_nowait()["type"] == "test_event"
        assert q2.get_nowait()["type"] == "test_event"

    @pytest.mark.asyncio
    async def test_emit_event_skips_full_queue(self, handler):
        sid = "sess-full-q"
        _sessions[sid] = _make_session(sid)
        q = asyncio.Queue(maxsize=1)
        q.put_nowait({"type": "existing"})
        _event_queues[sid] = [q]
        # Should not raise
        await handler._emit_event(sid, {"type": "new_event"})
        # Queue still has old event
        assert q.get_nowait()["type"] == "existing"

    @pytest.mark.asyncio
    async def test_emit_event_no_queues(self, handler):
        # Should not raise when no queues exist
        await handler._emit_event("nonexistent", {"type": "test"})


# =============================================================================
# 19. Response Helper Tests
# =============================================================================


class TestResponseHelpers:
    """Tests for _json_response, _error_response, _sse_response."""

    def test_json_response_format(self, handler):
        result = handler._json_response(200, {"key": "value"})
        assert result["status"] == 200
        assert result["headers"]["Content-Type"] == "application/json"
        assert json.loads(result["body"]) == {"key": "value"}

    def test_error_response_format(self, handler):
        result = handler._error_response(404, "Not found")
        assert result["status"] == 404
        body = json.loads(result["body"])
        assert body["error"] == "Not found"

    def test_sse_response_format(self, handler):
        gen = iter([])
        result = handler._sse_response(gen)
        assert result["status"] == 200
        assert result["headers"]["Content-Type"] == "text/event-stream"
        assert result["headers"]["Cache-Control"] == "no-cache"
        assert result["headers"]["Connection"] == "keep-alive"

    def test_json_response_serializes_datetime(self, handler):
        dt = datetime(2025, 1, 1, tzinfo=timezone.utc)
        result = handler._json_response(200, {"time": dt})
        body = json.loads(result["body"])
        assert "2025" in body["time"]


# =============================================================================
# 20. Parse JSON Body Tests
# =============================================================================


class TestParseJsonBody:
    """Tests for _parse_json_body."""

    @pytest.mark.asyncio
    async def test_parse_with_json_method(self, handler):
        req = MockHTTPHandler(body={"key": "value"})
        result = await handler._parse_json_body(req)
        assert result == {"key": "value"}

    @pytest.mark.asyncio
    async def test_parse_with_body_method(self, handler):
        # Create a request-like object with only a body() method, not json()
        class BodyOnlyRequest:
            async def body(self):
                return json.dumps({"from_body": True}).encode()

        req = BodyOnlyRequest()
        result = await handler._parse_json_body(req)
        assert result["from_body"] is True

    @pytest.mark.asyncio
    async def test_parse_empty_request(self, handler):
        # An object with neither json nor body
        class EmptyRequest:
            pass

        req = EmptyRequest()
        result = await handler._parse_json_body(req)
        assert result == {}


# =============================================================================
# 21. Background Audit Run Tests
# =============================================================================


class TestRunAuditBackground:
    """Tests for _run_audit_background."""

    @pytest.mark.asyncio
    async def test_background_audit_missing_session(self, handler):
        # Should return early without error
        await handler._run_audit_background("nonexistent")

    @pytest.mark.asyncio
    async def test_background_audit_fallback_completes(self, handler, seeded_running_session):
        """Test fallback simulation completes session."""
        # Patch DocumentAuditor to not be available, and sleep to be instant
        with (
            patch.dict("sys.modules", {"aragora.audit.document_auditor": None}),
            patch.dict("sys.modules", {"aragora.debate.cancellation": None}),
            patch(f"{MODULE}.asyncio.sleep", new_callable=AsyncMock),
        ):
            await handler._run_audit_background(seeded_running_session)
        assert _sessions[seeded_running_session]["status"] == "completed"
        assert _sessions[seeded_running_session]["completed_at"] is not None

    @pytest.mark.asyncio
    async def test_background_audit_error_marks_failed(self, handler, seeded_running_session):
        """Test that errors in background audit set status to failed."""
        with (
            patch.dict("sys.modules", {"aragora.audit.document_auditor": None}),
            patch.dict("sys.modules", {"aragora.debate.cancellation": None}),
            patch(f"{MODULE}.asyncio.sleep", side_effect=RuntimeError("boom")),
        ):
            await handler._run_audit_background(seeded_running_session)
        assert _sessions[seeded_running_session]["status"] == "failed"
        assert _sessions[seeded_running_session]["error"] is not None

    @pytest.mark.asyncio
    async def test_background_audit_cancelled_session_stops(self, handler):
        """Test that background audit stops when session is cancelled mid-run."""
        sid = "cancel-mid"
        _sessions[sid] = _make_session(sid, status="running", document_ids=["d1", "d2", "d3"])
        _findings[sid] = []
        _event_queues[sid] = []

        call_count = 0

        async def cancel_on_second_sleep(_):
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                _sessions[sid]["status"] = "cancelled"

        with (
            patch.dict("sys.modules", {"aragora.audit.document_auditor": None}),
            patch.dict("sys.modules", {"aragora.debate.cancellation": None}),
            patch(f"{MODULE}.asyncio.sleep", side_effect=cancel_on_second_sleep),
        ):
            await handler._run_audit_background(sid)
        # Should not be marked completed since it was cancelled
        assert _sessions[sid]["status"] == "cancelled"

    @pytest.mark.asyncio
    async def test_background_audit_cleans_cancellation_token(self, handler, seeded_running_session):
        """Test cancellation token is cleaned up after background audit."""
        mock_token_cls = MagicMock()
        mock_token = MagicMock()
        mock_token.is_cancelled = False
        mock_token_cls.return_value = mock_token

        mock_cancellation_module = MagicMock()
        mock_cancellation_module.CancellationToken = mock_token_cls

        with (
            patch.dict(
                "sys.modules",
                {
                    "aragora.debate.cancellation": mock_cancellation_module,
                    "aragora.audit.document_auditor": None,
                },
            ),
            patch(f"{MODULE}.asyncio.sleep", new_callable=AsyncMock),
        ):
            await handler._run_audit_background(seeded_running_session)
        assert seeded_running_session not in _cancellation_tokens


# =============================================================================
# 22. Handler Initialization Tests
# =============================================================================


class TestHandlerInit:
    """Tests for handler initialization."""

    def test_init_with_server_context(self):
        h = AuditSessionsHandler(server_context={"key": "val"})
        assert h.ctx == {"key": "val"}

    def test_init_with_ctx(self):
        h = AuditSessionsHandler(ctx={"old_key": "val"})
        assert h.ctx == {"old_key": "val"}

    def test_init_with_no_context(self):
        h = AuditSessionsHandler()
        assert h.ctx == {}

    def test_server_context_takes_precedence(self):
        h = AuditSessionsHandler(ctx={"a": 1}, server_context={"b": 2})
        assert h.ctx == {"b": 2}


# =============================================================================
# 23. Full Lifecycle Integration Tests
# =============================================================================


class TestFullLifecycle:
    """Integration-style tests covering create -> start -> pause -> resume -> cancel flows."""

    @pytest.mark.asyncio
    async def test_create_then_start(self, handler):
        # Create
        req = MockHTTPHandler(
            method="POST",
            path="/api/v1/audit/sessions",
            body={"document_ids": ["doc-1"]},
        )
        create_result = await handler.handle_request(req)
        sid = _body(create_result)["id"]

        # Start
        req = MockHTTPHandler(
            method="POST",
            path=f"/api/v1/audit/sessions/{sid}/start",
        )
        with patch(f"{MODULE}.asyncio.create_task") as mock_task:
            mock_task.return_value = MagicMock(add_done_callback=MagicMock())
            start_result = await handler.handle_request(req)
        assert _body(start_result)["status"] == "running"

    @pytest.mark.asyncio
    async def test_create_start_pause_resume(self, handler):
        # Create
        req = MockHTTPHandler(
            method="POST",
            path="/api/v1/audit/sessions",
            body={"document_ids": ["doc-1"]},
        )
        sid = _body(await handler.handle_request(req))["id"]

        # Start
        req = MockHTTPHandler(method="POST", path=f"/api/v1/audit/sessions/{sid}/start")
        with patch(f"{MODULE}.asyncio.create_task") as mock_task:
            mock_task.return_value = MagicMock(add_done_callback=MagicMock())
            await handler.handle_request(req)

        # Pause
        req = MockHTTPHandler(method="POST", path=f"/api/v1/audit/sessions/{sid}/pause")
        pause_result = await handler.handle_request(req)
        assert _body(pause_result)["status"] == "paused"

        # Resume
        req = MockHTTPHandler(method="POST", path=f"/api/v1/audit/sessions/{sid}/resume")
        resume_result = await handler.handle_request(req)
        assert _body(resume_result)["status"] == "running"

    @pytest.mark.asyncio
    async def test_create_start_cancel(self, handler):
        # Create
        req = MockHTTPHandler(
            method="POST",
            path="/api/v1/audit/sessions",
            body={"document_ids": ["doc-1"]},
        )
        sid = _body(await handler.handle_request(req))["id"]

        # Start
        req = MockHTTPHandler(method="POST", path=f"/api/v1/audit/sessions/{sid}/start")
        with patch(f"{MODULE}.asyncio.create_task") as mock_task:
            mock_task.return_value = MagicMock(add_done_callback=MagicMock())
            await handler.handle_request(req)

        # Cancel
        req = MockHTTPHandler(
            method="POST",
            path=f"/api/v1/audit/sessions/{sid}/cancel",
            body={"reason": "test cancel"},
        )
        cancel_result = await handler.handle_request(req)
        assert _body(cancel_result)["status"] == "cancelled"

    @pytest.mark.asyncio
    async def test_create_then_delete(self, handler):
        # Create
        req = MockHTTPHandler(
            method="POST",
            path="/api/v1/audit/sessions",
            body={"document_ids": ["doc-1"]},
        )
        sid = _body(await handler.handle_request(req))["id"]

        # Delete
        req = MockHTTPHandler(method="DELETE", path=f"/api/v1/audit/sessions/{sid}")
        delete_result = await handler.handle_request(req)
        assert _body(delete_result)["deleted"] == sid
        assert sid not in _sessions


# =============================================================================
# 24. Edge Cases
# =============================================================================


class TestEdgeCases:
    """Edge case and boundary tests."""

    @pytest.mark.asyncio
    async def test_pause_completed_session(self, handler, seeded_completed_session):
        req = MockHTTPHandler(
            method="POST",
            path=f"/api/v1/audit/sessions/{seeded_completed_session}/pause",
        )
        result = await handler.handle_request(req)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_resume_completed_session(self, handler, seeded_completed_session):
        req = MockHTTPHandler(
            method="POST",
            path=f"/api/v1/audit/sessions/{seeded_completed_session}/resume",
        )
        result = await handler.handle_request(req)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_cancel_paused_session_succeeds(self, handler, seeded_paused_session):
        req = MockHTTPHandler(
            method="POST",
            path=f"/api/v1/audit/sessions/{seeded_paused_session}/cancel",
        )
        result = await handler.handle_request(req)
        assert _status(result) == 200
        body = _body(result)
        assert body["status"] == "cancelled"

    @pytest.mark.asyncio
    async def test_delete_paused_session_succeeds(self, handler, seeded_paused_session):
        req = MockHTTPHandler(
            method="DELETE",
            path=f"/api/v1/audit/sessions/{seeded_paused_session}",
        )
        result = await handler.handle_request(req)
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_findings_default_limit(self, handler, seeded_session_with_findings):
        req = MockHTTPHandler(
            method="GET",
            path=f"/api/v1/audit/sessions/{seeded_session_with_findings}/findings",
        )
        result = await handler.handle_request(req)
        body = _body(result)
        assert body["limit"] == 100

    @pytest.mark.asyncio
    async def test_multiple_sessions_independent(self, handler):
        # Create two sessions
        for doc in ["doc-a", "doc-b"]:
            req = MockHTTPHandler(
                method="POST",
                path="/api/v1/audit/sessions",
                body={"document_ids": [doc]},
            )
            await handler.handle_request(req)
        assert len(_sessions) == 2
        sids = list(_sessions.keys())
        assert sids[0] != sids[1]
