"""
Tests for FindingWorkflowHandler.

Comprehensive test coverage for all finding workflow API endpoints:

Routing (can_handle):
- /api/v1/audit/findings/{id}/status        PATCH
- /api/v1/audit/findings/{id}/assign        PATCH
- /api/v1/audit/findings/{id}/unassign      POST
- /api/v1/audit/findings/{id}/comments      POST / GET
- /api/v1/audit/findings/{id}/history       GET
- /api/v1/audit/findings/{id}/priority      PATCH
- /api/v1/audit/findings/{id}/due-date      PATCH
- /api/v1/audit/findings/{id}/link          POST
- /api/v1/audit/findings/{id}/duplicate     POST
- /api/v1/audit/findings/bulk-action        POST
- /api/v1/audit/findings/my-assignments     GET
- /api/v1/audit/findings/overdue            GET
- /api/v1/audit/workflow/states             GET
- /api/v1/audit/presets                     GET
- /api/v1/audit/types                       GET
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.features.finding_workflow import FindingWorkflowHandler


# =============================================================================
# Module path for patching
# =============================================================================

MODULE = "aragora.server.handlers.features.finding_workflow"


# =============================================================================
# Mock Request
# =============================================================================


@dataclass
class MockHTTPHandler:
    """Mock HTTP request for handler tests."""

    path: str = "/api/v1/audit/findings/f-1/status"
    method: str = "GET"
    body: dict[str, Any] | None = None
    headers: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        self._raw = json.dumps(self.body).encode() if self.body is not None else b"{}"

    async def read(self) -> bytes:
        return self._raw

    async def json(self) -> dict[str, Any]:
        return self.body if self.body is not None else {}


# =============================================================================
# Mock auth contexts
# =============================================================================


class _AuthenticatedJWT:
    """Simulates an authenticated admin user from JWT."""

    authenticated = True
    user_id = "admin-001"
    role = "admin"
    org_id = "org-1"
    client_ip = "127.0.0.1"
    email = "admin@example.com"


class _UnauthenticatedJWT:
    """Simulates an unauthenticated request."""

    authenticated = False
    user_id = None
    role = None
    org_id = None
    client_ip = None
    email = None


# =============================================================================
# Helpers
# =============================================================================


def _status(result: dict[str, Any]) -> int:
    """Extract status code from handler response dict."""
    return result.get("status", 0)


def _body(result: dict[str, Any]) -> dict[str, Any]:
    """Parse the JSON body from a handler response dict."""
    return json.loads(result.get("body", "{}"))


def _default_workflow(finding_id: str = "f-1", **overrides: Any) -> dict[str, Any]:
    """Create a default workflow dict for testing."""
    wf: dict[str, Any] = {
        "finding_id": finding_id,
        "current_state": "open",
        "history": [],
        "assigned_to": None,
        "assigned_by": None,
        "assigned_at": None,
        "priority": 3,
        "due_date": None,
        "linked_findings": [],
        "parent_finding_id": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    wf.update(overrides)
    return wf


def _make_store(
    existing: dict[str, Any] | None = None,
    assignee_list: list[dict[str, Any]] | None = None,
    overdue_list: list[dict[str, Any]] | None = None,
) -> MagicMock:
    """Build a mock FindingWorkflowStoreBackend."""
    store = MagicMock()
    store.get = AsyncMock(return_value=existing)
    store.save = AsyncMock()
    store.list_by_assignee = AsyncMock(return_value=assignee_list or [])
    store.list_overdue = AsyncMock(return_value=overdue_list or [])
    return store


def _auth_patch():
    """Return a patch that makes extract_user_from_request return admin context."""
    return patch(f"{MODULE}.extract_user_from_request", return_value=_AuthenticatedJWT())


def _import_error_patch():
    """Force ImportError for aragora.audit.findings.workflow (fallback path)."""
    return patch.dict("sys.modules", {"aragora.audit.findings.workflow": None})


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def handler():
    """Create a FindingWorkflowHandler instance."""
    return FindingWorkflowHandler(server_context={})


@pytest.fixture
def store():
    """Create a mock store with no existing data."""
    return _make_store(existing=None)


@pytest.fixture
def store_with_workflow():
    """Create a mock store with an existing open workflow."""
    return _make_store(existing=_default_workflow())


# =============================================================================
# 1. can_handle() Routing Tests
# =============================================================================


class TestCanHandle:
    """Test can_handle routes all expected paths."""

    def test_can_handle_finding_workflow_prefix(self, handler):
        assert handler.can_handle("/api/v1/finding-workflow/something") is True

    def test_cannot_handle_unrelated_path(self, handler):
        assert handler.can_handle("/api/v1/debates") is False

    def test_cannot_handle_audit_path_without_prefix(self, handler):
        # can_handle checks startswith("/api/v1/finding-workflow/")
        assert handler.can_handle("/api/v1/audit/findings/f-1/status") is False

    def test_cannot_handle_empty_path(self, handler):
        assert handler.can_handle("") is False

    def test_cannot_handle_partial_prefix(self, handler):
        assert handler.can_handle("/api/v1/finding-workflow") is False

    def test_routes_class_attribute_has_all_endpoints(self):
        assert len(FindingWorkflowHandler.ROUTES) == 15
        expected_routes = [
            "/api/v1/audit/findings/{finding_id}/status",
            "/api/v1/audit/findings/{finding_id}/assign",
            "/api/v1/audit/findings/{finding_id}/unassign",
            "/api/v1/audit/findings/{finding_id}/comments",
            "/api/v1/audit/findings/{finding_id}/history",
            "/api/v1/audit/findings/{finding_id}/priority",
            "/api/v1/audit/findings/{finding_id}/due-date",
            "/api/v1/audit/findings/{finding_id}/link",
            "/api/v1/audit/findings/{finding_id}/duplicate",
            "/api/v1/audit/findings/bulk-action",
            "/api/v1/audit/findings/my-assignments",
            "/api/v1/audit/findings/overdue",
            "/api/v1/audit/workflow/states",
            "/api/v1/audit/presets",
            "/api/v1/audit/types",
        ]
        for route in expected_routes:
            assert route in FindingWorkflowHandler.ROUTES, f"Missing route: {route}"


# =============================================================================
# 2. handle_request Routing Tests
# =============================================================================


class TestHandleRequestRouting:
    """Test that handle_request dispatches to the correct internal method."""

    @pytest.mark.asyncio
    async def test_routes_to_workflow_states(self, handler):
        req = MockHTTPHandler(method="GET", path="/api/v1/audit/workflow/states")
        with _auth_patch(), patch.object(
            handler,
            "_get_workflow_states",
            new_callable=AsyncMock,
            return_value={"status": 200, "body": "{}"},
        ) as mock_fn:
            await handler.handle_request(req)
        mock_fn.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_routes_to_presets(self, handler):
        req = MockHTTPHandler(method="GET", path="/api/v1/audit/presets")
        with _auth_patch(), patch.object(
            handler,
            "_get_presets",
            new_callable=AsyncMock,
            return_value={"status": 200, "body": "{}"},
        ) as mock_fn:
            await handler.handle_request(req)
        mock_fn.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_routes_to_audit_types(self, handler):
        req = MockHTTPHandler(method="GET", path="/api/v1/audit/types")
        with _auth_patch(), patch.object(
            handler,
            "_get_audit_types",
            new_callable=AsyncMock,
            return_value={"status": 200, "body": "{}"},
        ) as mock_fn:
            await handler.handle_request(req)
        mock_fn.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_routes_to_bulk_action(self, handler):
        req = MockHTTPHandler(method="POST", path="/api/v1/audit/findings/bulk-action")
        with _auth_patch(), patch.object(
            handler,
            "_bulk_action",
            new_callable=AsyncMock,
            return_value={"status": 200, "body": "{}"},
        ) as mock_fn:
            await handler.handle_request(req)
        mock_fn.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_routes_to_my_assignments(self, handler):
        req = MockHTTPHandler(method="GET", path="/api/v1/audit/findings/my-assignments")
        with _auth_patch(), patch.object(
            handler,
            "_get_my_assignments",
            new_callable=AsyncMock,
            return_value={"status": 200, "body": "{}"},
        ) as mock_fn:
            await handler.handle_request(req)
        mock_fn.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_routes_to_overdue(self, handler):
        req = MockHTTPHandler(method="GET", path="/api/v1/audit/findings/overdue")
        with _auth_patch(), patch.object(
            handler,
            "_get_overdue",
            new_callable=AsyncMock,
            return_value={"status": 200, "body": "{}"},
        ) as mock_fn:
            await handler.handle_request(req)
        mock_fn.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_routes_to_update_status(self, handler):
        req = MockHTTPHandler(
            method="PATCH",
            path="/api/v1/audit/findings/f-42/status",
            body={"status": "triaging"},
        )
        with _auth_patch(), patch.object(
            handler,
            "_update_status",
            new_callable=AsyncMock,
            return_value={"status": 200, "body": "{}"},
        ) as mock_fn:
            await handler.handle_request(req)
        mock_fn.assert_awaited_once()
        assert mock_fn.call_args[0][1] == "f-42"

    @pytest.mark.asyncio
    async def test_routes_to_assign(self, handler):
        req = MockHTTPHandler(
            method="PATCH",
            path="/api/v1/audit/findings/f-99/assign",
            body={"user_id": "u-1"},
        )
        with _auth_patch(), patch.object(
            handler,
            "_assign",
            new_callable=AsyncMock,
            return_value={"status": 200, "body": "{}"},
        ) as mock_fn:
            await handler.handle_request(req)
        mock_fn.assert_awaited_once()
        assert mock_fn.call_args[0][1] == "f-99"

    @pytest.mark.asyncio
    async def test_routes_to_unassign(self, handler):
        req = MockHTTPHandler(method="POST", path="/api/v1/audit/findings/f-7/unassign")
        with _auth_patch(), patch.object(
            handler,
            "_unassign",
            new_callable=AsyncMock,
            return_value={"status": 200, "body": "{}"},
        ) as mock_fn:
            await handler.handle_request(req)
        mock_fn.assert_awaited_once()
        assert mock_fn.call_args[0][1] == "f-7"

    @pytest.mark.asyncio
    async def test_routes_to_add_comment(self, handler):
        req = MockHTTPHandler(
            method="POST",
            path="/api/v1/audit/findings/f-1/comments",
            body={"comment": "test"},
        )
        with _auth_patch(), patch.object(
            handler,
            "_add_comment",
            new_callable=AsyncMock,
            return_value={"status": 201, "body": "{}"},
        ) as mock_fn:
            await handler.handle_request(req)
        mock_fn.assert_awaited_once()
        assert mock_fn.call_args[0][1] == "f-1"

    @pytest.mark.asyncio
    async def test_routes_to_get_comments(self, handler):
        req = MockHTTPHandler(method="GET", path="/api/v1/audit/findings/f-1/comments")
        with _auth_patch(), patch.object(
            handler,
            "_get_comments",
            new_callable=AsyncMock,
            return_value={"status": 200, "body": "{}"},
        ) as mock_fn:
            await handler.handle_request(req)
        mock_fn.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_routes_to_history(self, handler):
        req = MockHTTPHandler(method="GET", path="/api/v1/audit/findings/f-1/history")
        with _auth_patch(), patch.object(
            handler,
            "_get_history",
            new_callable=AsyncMock,
            return_value={"status": 200, "body": "{}"},
        ) as mock_fn:
            await handler.handle_request(req)
        mock_fn.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_routes_to_set_priority(self, handler):
        req = MockHTTPHandler(
            method="PATCH",
            path="/api/v1/audit/findings/f-1/priority",
            body={"priority": 2},
        )
        with _auth_patch(), patch.object(
            handler,
            "_set_priority",
            new_callable=AsyncMock,
            return_value={"status": 200, "body": "{}"},
        ) as mock_fn:
            await handler.handle_request(req)
        mock_fn.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_routes_to_set_due_date(self, handler):
        req = MockHTTPHandler(
            method="PATCH",
            path="/api/v1/audit/findings/f-1/due-date",
            body={"due_date": "2026-12-31T00:00:00Z"},
        )
        with _auth_patch(), patch.object(
            handler,
            "_set_due_date",
            new_callable=AsyncMock,
            return_value={"status": 200, "body": "{}"},
        ) as mock_fn:
            await handler.handle_request(req)
        mock_fn.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_routes_to_link_finding(self, handler):
        req = MockHTTPHandler(
            method="POST",
            path="/api/v1/audit/findings/f-1/link",
            body={"linked_finding_id": "f-2"},
        )
        with _auth_patch(), patch.object(
            handler,
            "_link_finding",
            new_callable=AsyncMock,
            return_value={"status": 200, "body": "{}"},
        ) as mock_fn:
            await handler.handle_request(req)
        mock_fn.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_routes_to_mark_duplicate(self, handler):
        req = MockHTTPHandler(
            method="POST",
            path="/api/v1/audit/findings/f-1/duplicate",
            body={"parent_finding_id": "f-0"},
        )
        with _auth_patch(), patch.object(
            handler,
            "_mark_duplicate",
            new_callable=AsyncMock,
            return_value={"status": 200, "body": "{}"},
        ) as mock_fn:
            await handler.handle_request(req)
        mock_fn.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_unknown_endpoint_returns_404(self, handler):
        req = MockHTTPHandler(method="GET", path="/api/v1/audit/unknown/endpoint")
        result = await handler.handle_request(req)
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_unknown_method_on_known_path_returns_404(self, handler):
        # DELETE is not supported on /status
        req = MockHTTPHandler(method="DELETE", path="/api/v1/audit/findings/f-1/status")
        result = await handler.handle_request(req)
        assert _status(result) == 404


# =============================================================================
# 3. Path Parameter Extraction
# =============================================================================


class TestPathParameterExtraction:
    """Verify finding_id is correctly extracted from various paths."""

    @pytest.mark.asyncio
    async def test_extracts_simple_finding_id(self, handler):
        req = MockHTTPHandler(method="PATCH", path="/api/v1/audit/findings/abc-123/status")
        with _auth_patch(), patch.object(
            handler, "_update_status", new_callable=AsyncMock, return_value={"status": 200, "body": "{}"}
        ) as mock_fn:
            await handler.handle_request(req)
        assert mock_fn.call_args[0][1] == "abc-123"

    @pytest.mark.asyncio
    async def test_extracts_uuid_finding_id(self, handler):
        fid = "550e8400-e29b-41d4-a716-446655440000"
        req = MockHTTPHandler(method="GET", path=f"/api/v1/audit/findings/{fid}/history")
        with _auth_patch(), patch.object(
            handler, "_get_history", new_callable=AsyncMock, return_value={"status": 200, "body": "{}"}
        ) as mock_fn:
            await handler.handle_request(req)
        assert mock_fn.call_args[0][1] == fid

    @pytest.mark.asyncio
    async def test_bulk_action_does_not_extract_finding_id(self, handler):
        req = MockHTTPHandler(
            method="POST",
            path="/api/v1/audit/findings/bulk-action",
            body={"finding_ids": ["f-1"], "action": "assign", "params": {"user_id": "u-1"}},
        )
        with _auth_patch(), patch.object(
            handler, "_bulk_action", new_callable=AsyncMock, return_value={"status": 200, "body": "{}"}
        ) as mock_fn:
            await handler.handle_request(req)
        # _bulk_action is called with just (request,), no finding_id
        assert len(mock_fn.call_args[0]) == 1

    @pytest.mark.asyncio
    async def test_my_assignments_does_not_extract_finding_id(self, handler):
        req = MockHTTPHandler(method="GET", path="/api/v1/audit/findings/my-assignments")
        with _auth_patch(), patch.object(
            handler, "_get_my_assignments", new_callable=AsyncMock, return_value={"status": 200, "body": "{}"}
        ) as mock_fn:
            await handler.handle_request(req)
        assert len(mock_fn.call_args[0]) == 1


# =============================================================================
# 4. _update_status Tests
# =============================================================================


class TestUpdateStatus:
    """Test PATCH /api/v1/audit/findings/{id}/status."""

    @pytest.mark.asyncio
    async def test_success_fallback_path(self, handler):
        existing = _default_workflow(current_state="open")
        store = _make_store(existing=existing)
        req = MockHTTPHandler(body={"status": "triaging", "comment": "triage"})
        with (
            _auth_patch(),
            patch(f"{MODULE}.get_finding_workflow_store", return_value=store),
            patch(f"{MODULE}._finding_workflow_circuit_breaker") as cb,
            _import_error_patch(),
        ):
            cb.can_proceed.return_value = True
            result = await handler._update_status(req, "f-1")
        assert _status(result) == 200
        body = _body(result)
        assert body["success"] is True
        assert body["previous_state"] == "open"
        assert body["current_state"] == "triaging"

    @pytest.mark.asyncio
    async def test_missing_status_returns_400(self, handler):
        req = MockHTTPHandler(body={"comment": "no status"})
        with _auth_patch(), patch(f"{MODULE}._finding_workflow_circuit_breaker") as cb:
            cb.can_proceed.return_value = True
            result = await handler._update_status(req, "f-1")
        assert _status(result) == 400
        assert "status is required" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_invalid_json_returns_400(self, handler):
        req = MockHTTPHandler()
        req._raw = b"<<<not json>>>"
        # Override json() to raise
        async def bad_json():
            raise ValueError("bad json")
        req.json = bad_json
        with _auth_patch(), patch(f"{MODULE}._finding_workflow_circuit_breaker") as cb:
            cb.can_proceed.return_value = True
            result = await handler._update_status(req, "f-1")
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_circuit_breaker_open_returns_503(self, handler):
        req = MockHTTPHandler(body={"status": "triaging"})
        with patch(f"{MODULE}._finding_workflow_circuit_breaker") as cb:
            cb.can_proceed.return_value = False
            result = await handler._update_status(req, "f-1")
        assert _status(result) == 503
        assert "temporarily unavailable" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_status_history_appended(self, handler):
        existing = _default_workflow(current_state="open")
        store = _make_store(existing=existing)
        req = MockHTTPHandler(body={"status": "investigating"})
        with (
            _auth_patch(),
            patch(f"{MODULE}.get_finding_workflow_store", return_value=store),
            patch(f"{MODULE}._finding_workflow_circuit_breaker") as cb,
            _import_error_patch(),
        ):
            cb.can_proceed.return_value = True
            await handler._update_status(req, "f-1")
        # Verify save was called and history was appended
        saved_data = store.save.call_args[0][0]
        assert len(saved_data["history"]) == 1
        assert saved_data["history"][0]["event_type"] == "state_change"
        assert saved_data["history"][0]["from_state"] == "open"
        assert saved_data["history"][0]["to_state"] == "investigating"


# =============================================================================
# 5. _assign Tests
# =============================================================================


class TestAssign:
    """Test PATCH /api/v1/audit/findings/{id}/assign."""

    @pytest.mark.asyncio
    async def test_assign_success(self, handler):
        store = _make_store(existing=None)
        req = MockHTTPHandler(body={"user_id": "user-2", "comment": "review please"})
        with _auth_patch(), patch(f"{MODULE}.get_finding_workflow_store", return_value=store), _import_error_patch():
            result = await handler._assign(req, "f-1")
        assert _status(result) == 200
        body = _body(result)
        assert body["success"] is True
        assert body["assigned_to"] == "user-2"

    @pytest.mark.asyncio
    async def test_assign_missing_user_id_returns_400(self, handler):
        req = MockHTTPHandler(body={"comment": "no user"})
        with _auth_patch():
            result = await handler._assign(req, "f-1")
        assert _status(result) == 400
        assert "user_id is required" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_assign_invalid_json_returns_400(self, handler):
        req = MockHTTPHandler()
        req._raw = b"not json"
        async def bad_json():
            raise ValueError("bad")
        req.json = bad_json
        with _auth_patch():
            result = await handler._assign(req, "f-1")
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_assign_updates_store(self, handler):
        store = _make_store(existing=None)
        req = MockHTTPHandler(body={"user_id": "user-5"})
        with _auth_patch(), patch(f"{MODULE}.get_finding_workflow_store", return_value=store), _import_error_patch():
            await handler._assign(req, "f-1")
        saved = store.save.call_args[0][0]
        assert saved["assigned_to"] == "user-5"
        assert saved["assigned_by"] == "admin-001"
        assert saved["assigned_at"] is not None

    @pytest.mark.asyncio
    async def test_assign_with_comment_in_history(self, handler):
        existing = _default_workflow()
        store = _make_store(existing=existing)
        req = MockHTTPHandler(body={"user_id": "user-3", "comment": "urgent"})
        with _auth_patch(), patch(f"{MODULE}.get_finding_workflow_store", return_value=store), _import_error_patch():
            await handler._assign(req, "f-1")
        saved = store.save.call_args[0][0]
        assert len(saved["history"]) == 1
        assert saved["history"][0]["event_type"] == "assignment"
        assert saved["history"][0]["comment"] == "urgent"


# =============================================================================
# 6. _unassign Tests
# =============================================================================


class TestUnassign:
    """Test POST /api/v1/audit/findings/{id}/unassign."""

    @pytest.mark.asyncio
    async def test_unassign_success(self, handler):
        existing = _default_workflow(assigned_to="user-2")
        store = _make_store(existing=existing)
        req = MockHTTPHandler(body={"comment": "freeing"})
        with _auth_patch(), patch(f"{MODULE}.get_finding_workflow_store", return_value=store), _import_error_patch():
            result = await handler._unassign(req, "f-1")
        assert _status(result) == 200
        assert _body(result)["success"] is True

    @pytest.mark.asyncio
    async def test_unassign_clears_assignment_fields(self, handler):
        existing = _default_workflow(assigned_to="user-2", assigned_by="admin-001")
        store = _make_store(existing=existing)
        req = MockHTTPHandler(body={})
        with _auth_patch(), patch(f"{MODULE}.get_finding_workflow_store", return_value=store), _import_error_patch():
            await handler._unassign(req, "f-1")
        saved = store.save.call_args[0][0]
        assert saved["assigned_to"] is None
        assert saved["assigned_by"] is None
        assert saved["assigned_at"] is None

    @pytest.mark.asyncio
    async def test_unassign_without_body_still_succeeds(self, handler):
        existing = _default_workflow(assigned_to="user-2")
        store = _make_store(existing=existing)
        req = MockHTTPHandler()
        req._raw = b"<<<invalid>>>"
        async def bad_json():
            raise ValueError("bad")
        req.json = bad_json
        with _auth_patch(), patch(f"{MODULE}.get_finding_workflow_store", return_value=store), _import_error_patch():
            result = await handler._unassign(req, "f-1")
        assert _status(result) == 200


# =============================================================================
# 7. _add_comment / _get_comments Tests
# =============================================================================


class TestComments:
    """Test POST/GET /api/v1/audit/findings/{id}/comments."""

    @pytest.mark.asyncio
    async def test_add_comment_success(self, handler):
        existing = _default_workflow()
        store = _make_store(existing=existing)
        req = MockHTTPHandler(body={"comment": "Needs investigation"})
        with _auth_patch(), patch(f"{MODULE}.get_finding_workflow_store", return_value=store), _import_error_patch():
            result = await handler._add_comment(req, "f-1")
        assert _status(result) == 201
        body = _body(result)
        assert body["success"] is True
        assert body["comment"]["comment"] == "Needs investigation"

    @pytest.mark.asyncio
    async def test_add_comment_missing_comment_returns_400(self, handler):
        req = MockHTTPHandler(body={})
        with _auth_patch():
            result = await handler._add_comment(req, "f-1")
        assert _status(result) == 400
        assert "comment is required" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_add_comment_empty_string_returns_400(self, handler):
        req = MockHTTPHandler(body={"comment": ""})
        with _auth_patch():
            result = await handler._add_comment(req, "f-1")
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_add_comment_invalid_json_returns_400(self, handler):
        req = MockHTTPHandler()
        req._raw = b"not valid json"
        async def bad_json():
            raise ValueError("bad")
        req.json = bad_json
        with _auth_patch():
            result = await handler._add_comment(req, "f-1")
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_add_comment_appends_to_history(self, handler):
        existing = _default_workflow()
        store = _make_store(existing=existing)
        req = MockHTTPHandler(body={"comment": "a note"})
        with _auth_patch(), patch(f"{MODULE}.get_finding_workflow_store", return_value=store), _import_error_patch():
            await handler._add_comment(req, "f-1")
        saved = store.save.call_args[0][0]
        assert len(saved["history"]) == 1
        assert saved["history"][0]["event_type"] == "comment"
        assert saved["history"][0]["user_id"] == "admin-001"

    @pytest.mark.asyncio
    async def test_get_comments_filters_history(self, handler):
        existing = _default_workflow(
            history=[
                {"event_type": "comment", "comment": "first"},
                {"event_type": "state_change"},
                {"event_type": "comment", "comment": "second"},
                {"event_type": "assignment"},
            ]
        )
        store = _make_store(existing=existing)
        req = MockHTTPHandler()
        with _auth_patch(), patch(f"{MODULE}.get_finding_workflow_store", return_value=store):
            result = await handler._get_comments(req, "f-1")
        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 2
        assert len(body["comments"]) == 2
        assert all(c["event_type"] == "comment" for c in body["comments"])

    @pytest.mark.asyncio
    async def test_get_comments_empty_history(self, handler):
        existing = _default_workflow()
        store = _make_store(existing=existing)
        req = MockHTTPHandler()
        with _auth_patch(), patch(f"{MODULE}.get_finding_workflow_store", return_value=store):
            result = await handler._get_comments(req, "f-1")
        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 0
        assert body["comments"] == []


# =============================================================================
# 8. _get_history Tests
# =============================================================================


class TestGetHistory:
    """Test GET /api/v1/audit/findings/{id}/history."""

    @pytest.mark.asyncio
    async def test_get_history_returns_full_data(self, handler):
        existing = _default_workflow(
            assigned_to="user-x",
            priority=1,
            due_date="2026-03-01T00:00:00+00:00",
            history=[{"event_type": "state_change"}, {"event_type": "comment"}],
        )
        store = _make_store(existing=existing)
        req = MockHTTPHandler()
        with _auth_patch(), patch(f"{MODULE}.get_finding_workflow_store", return_value=store):
            result = await handler._get_history(req, "f-1")
        assert _status(result) == 200
        body = _body(result)
        assert body["finding_id"] == "f-1"
        assert body["current_state"] == "open"
        assert body["assigned_to"] == "user-x"
        assert body["priority"] == 1
        assert body["due_date"] == "2026-03-01T00:00:00+00:00"
        assert len(body["history"]) == 2

    @pytest.mark.asyncio
    async def test_get_history_new_finding(self, handler):
        store = _make_store(existing=None)
        req = MockHTTPHandler()
        with _auth_patch(), patch(f"{MODULE}.get_finding_workflow_store", return_value=store):
            result = await handler._get_history(req, "new-finding")
        assert _status(result) == 200
        body = _body(result)
        assert body["finding_id"] == "new-finding"
        assert body["current_state"] == "open"
        assert body["history"] == []


# =============================================================================
# 9. _set_priority Tests
# =============================================================================


class TestSetPriority:
    """Test PATCH /api/v1/audit/findings/{id}/priority."""

    @pytest.mark.asyncio
    async def test_set_priority_success(self, handler):
        existing = _default_workflow()
        store = _make_store(existing=existing)
        req = MockHTTPHandler(body={"priority": 1, "comment": "urgent"})
        with _auth_patch(), patch(f"{MODULE}.get_finding_workflow_store", return_value=store), _import_error_patch():
            result = await handler._set_priority(req, "f-1")
        assert _status(result) == 200
        assert _body(result)["priority"] == 1

    @pytest.mark.asyncio
    async def test_set_priority_boundary_low(self, handler):
        existing = _default_workflow()
        store = _make_store(existing=existing)
        req = MockHTTPHandler(body={"priority": 1})
        with _auth_patch(), patch(f"{MODULE}.get_finding_workflow_store", return_value=store), _import_error_patch():
            result = await handler._set_priority(req, "f-1")
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_set_priority_boundary_high(self, handler):
        existing = _default_workflow()
        store = _make_store(existing=existing)
        req = MockHTTPHandler(body={"priority": 5})
        with _auth_patch(), patch(f"{MODULE}.get_finding_workflow_store", return_value=store), _import_error_patch():
            result = await handler._set_priority(req, "f-1")
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_set_priority_zero_returns_400(self, handler):
        req = MockHTTPHandler(body={"priority": 0})
        with _auth_patch():
            result = await handler._set_priority(req, "f-1")
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_set_priority_six_returns_400(self, handler):
        req = MockHTTPHandler(body={"priority": 6})
        with _auth_patch():
            result = await handler._set_priority(req, "f-1")
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_set_priority_negative_returns_400(self, handler):
        req = MockHTTPHandler(body={"priority": -1})
        with _auth_patch():
            result = await handler._set_priority(req, "f-1")
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_set_priority_string_returns_400(self, handler):
        req = MockHTTPHandler(body={"priority": "high"})
        with _auth_patch():
            result = await handler._set_priority(req, "f-1")
        assert _status(result) == 400
        assert "priority must be integer 1-5" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_set_priority_missing_returns_400(self, handler):
        req = MockHTTPHandler(body={})
        with _auth_patch():
            result = await handler._set_priority(req, "f-1")
        assert _status(result) == 400
        assert "priority is required" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_set_priority_records_history(self, handler):
        existing = _default_workflow(priority=3)
        store = _make_store(existing=existing)
        req = MockHTTPHandler(body={"priority": 1, "comment": "escalation"})
        with _auth_patch(), patch(f"{MODULE}.get_finding_workflow_store", return_value=store), _import_error_patch():
            await handler._set_priority(req, "f-1")
        saved = store.save.call_args[0][0]
        assert saved["priority"] == 1
        event = saved["history"][-1]
        assert event["event_type"] == "priority_change"
        assert event["old_value"] == 3
        assert event["new_value"] == 1
        assert event["comment"] == "escalation"


# =============================================================================
# 10. _set_due_date Tests
# =============================================================================


class TestSetDueDate:
    """Test PATCH /api/v1/audit/findings/{id}/due-date."""

    @pytest.mark.asyncio
    async def test_set_due_date_success(self, handler):
        existing = _default_workflow()
        store = _make_store(existing=existing)
        req = MockHTTPHandler(body={"due_date": "2026-06-01T12:00:00Z"})
        with _auth_patch(), patch(f"{MODULE}.get_finding_workflow_store", return_value=store):
            result = await handler._set_due_date(req, "f-1")
        assert _status(result) == 200
        body = _body(result)
        assert body["due_date"] is not None
        assert "2026-06-01" in body["due_date"]

    @pytest.mark.asyncio
    async def test_clear_due_date(self, handler):
        existing = _default_workflow(due_date="2026-01-01T00:00:00+00:00")
        store = _make_store(existing=existing)
        req = MockHTTPHandler(body={"due_date": None})
        with _auth_patch(), patch(f"{MODULE}.get_finding_workflow_store", return_value=store):
            result = await handler._set_due_date(req, "f-1")
        assert _status(result) == 200
        assert _body(result)["due_date"] is None

    @pytest.mark.asyncio
    async def test_set_due_date_invalid_format_returns_400(self, handler):
        req = MockHTTPHandler(body={"due_date": "next-tuesday"})
        with _auth_patch():
            result = await handler._set_due_date(req, "f-1")
        assert _status(result) == 400
        assert "ISO 8601" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_set_due_date_invalid_json_returns_400(self, handler):
        req = MockHTTPHandler()
        req._raw = b"not json"
        async def bad_json():
            raise ValueError("bad")
        req.json = bad_json
        with _auth_patch():
            result = await handler._set_due_date(req, "f-1")
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_set_due_date_records_history(self, handler):
        existing = _default_workflow()
        store = _make_store(existing=existing)
        req = MockHTTPHandler(body={"due_date": "2026-12-31T23:59:59Z", "comment": "deadline"})
        with _auth_patch(), patch(f"{MODULE}.get_finding_workflow_store", return_value=store):
            await handler._set_due_date(req, "f-1")
        saved = store.save.call_args[0][0]
        event = saved["history"][-1]
        assert event["event_type"] == "due_date_change"
        assert event["comment"] == "deadline"

    @pytest.mark.asyncio
    async def test_set_due_date_iso_with_timezone(self, handler):
        existing = _default_workflow()
        store = _make_store(existing=existing)
        req = MockHTTPHandler(body={"due_date": "2026-06-15T10:30:00+05:00"})
        with _auth_patch(), patch(f"{MODULE}.get_finding_workflow_store", return_value=store):
            result = await handler._set_due_date(req, "f-1")
        assert _status(result) == 200
        assert _body(result)["due_date"] is not None


# =============================================================================
# 11. _link_finding Tests
# =============================================================================


class TestLinkFinding:
    """Test POST /api/v1/audit/findings/{id}/link."""

    @pytest.mark.asyncio
    async def test_link_finding_success(self, handler):
        existing = _default_workflow()
        store = _make_store(existing=existing)
        req = MockHTTPHandler(body={"linked_finding_id": "f-2", "comment": "related"})
        with _auth_patch(), patch(f"{MODULE}.get_finding_workflow_store", return_value=store):
            result = await handler._link_finding(req, "f-1")
        assert _status(result) == 200
        body = _body(result)
        assert "f-2" in body["linked_findings"]

    @pytest.mark.asyncio
    async def test_link_finding_no_duplicate_entries(self, handler):
        existing = _default_workflow(linked_findings=["f-2"])
        store = _make_store(existing=existing)
        req = MockHTTPHandler(body={"linked_finding_id": "f-2"})
        with _auth_patch(), patch(f"{MODULE}.get_finding_workflow_store", return_value=store):
            result = await handler._link_finding(req, "f-1")
        assert _status(result) == 200
        assert _body(result)["linked_findings"].count("f-2") == 1

    @pytest.mark.asyncio
    async def test_link_finding_missing_id_returns_400(self, handler):
        req = MockHTTPHandler(body={})
        with _auth_patch():
            result = await handler._link_finding(req, "f-1")
        assert _status(result) == 400
        assert "linked_finding_id is required" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_link_finding_invalid_json_returns_400(self, handler):
        req = MockHTTPHandler()
        req._raw = b"bad"
        async def bad_json():
            raise ValueError("bad")
        req.json = bad_json
        with _auth_patch():
            result = await handler._link_finding(req, "f-1")
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_link_finding_records_history(self, handler):
        existing = _default_workflow()
        store = _make_store(existing=existing)
        req = MockHTTPHandler(body={"linked_finding_id": "f-3", "comment": "see also"})
        with _auth_patch(), patch(f"{MODULE}.get_finding_workflow_store", return_value=store):
            await handler._link_finding(req, "f-1")
        saved = store.save.call_args[0][0]
        event = saved["history"][-1]
        assert event["event_type"] == "linked"
        assert event["new_value"] == "f-3"


# =============================================================================
# 12. _mark_duplicate Tests
# =============================================================================


class TestMarkDuplicate:
    """Test POST /api/v1/audit/findings/{id}/duplicate."""

    @pytest.mark.asyncio
    async def test_mark_duplicate_success(self, handler):
        existing = _default_workflow()
        store = _make_store(existing=existing)
        req = MockHTTPHandler(body={"parent_finding_id": "f-original"})
        with _auth_patch(), patch(f"{MODULE}.get_finding_workflow_store", return_value=store):
            result = await handler._mark_duplicate(req, "f-1")
        assert _status(result) == 200
        body = _body(result)
        assert body["current_state"] == "duplicate"
        assert body["parent_finding_id"] == "f-original"

    @pytest.mark.asyncio
    async def test_mark_duplicate_missing_parent_returns_400(self, handler):
        req = MockHTTPHandler(body={})
        with _auth_patch():
            result = await handler._mark_duplicate(req, "f-1")
        assert _status(result) == 400
        assert "parent_finding_id is required" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_mark_duplicate_invalid_json_returns_400(self, handler):
        req = MockHTTPHandler()
        req._raw = b"not json"
        async def bad_json():
            raise ValueError("bad")
        req.json = bad_json
        with _auth_patch():
            result = await handler._mark_duplicate(req, "f-1")
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_mark_duplicate_sets_state_and_parent(self, handler):
        existing = _default_workflow(current_state="investigating")
        store = _make_store(existing=existing)
        req = MockHTTPHandler(body={"parent_finding_id": "f-0"})
        with _auth_patch(), patch(f"{MODULE}.get_finding_workflow_store", return_value=store):
            await handler._mark_duplicate(req, "f-1")
        saved = store.save.call_args[0][0]
        assert saved["current_state"] == "duplicate"
        assert saved["parent_finding_id"] == "f-0"

    @pytest.mark.asyncio
    async def test_mark_duplicate_links_parent(self, handler):
        existing = _default_workflow()
        store = _make_store(existing=existing)
        req = MockHTTPHandler(body={"parent_finding_id": "f-parent"})
        with _auth_patch(), patch(f"{MODULE}.get_finding_workflow_store", return_value=store):
            await handler._mark_duplicate(req, "f-1")
        saved = store.save.call_args[0][0]
        assert "f-parent" in saved["linked_findings"]

    @pytest.mark.asyncio
    async def test_mark_duplicate_default_comment(self, handler):
        existing = _default_workflow()
        store = _make_store(existing=existing)
        req = MockHTTPHandler(body={"parent_finding_id": "f-parent"})
        with _auth_patch(), patch(f"{MODULE}.get_finding_workflow_store", return_value=store):
            await handler._mark_duplicate(req, "f-1")
        saved = store.save.call_args[0][0]
        event = saved["history"][-1]
        assert "Duplicate of f-parent" in event["comment"]

    @pytest.mark.asyncio
    async def test_mark_duplicate_custom_comment(self, handler):
        existing = _default_workflow()
        store = _make_store(existing=existing)
        req = MockHTTPHandler(body={"parent_finding_id": "f-p", "comment": "exact same issue"})
        with _auth_patch(), patch(f"{MODULE}.get_finding_workflow_store", return_value=store):
            await handler._mark_duplicate(req, "f-1")
        saved = store.save.call_args[0][0]
        event = saved["history"][-1]
        assert event["comment"] == "exact same issue"


# =============================================================================
# 13. _bulk_action Tests
# =============================================================================


class TestBulkAction:
    """Test POST /api/v1/audit/findings/bulk-action."""

    @pytest.mark.asyncio
    async def test_bulk_assign_success(self, handler):
        store = _make_store(existing=None)
        req = MockHTTPHandler(
            body={
                "finding_ids": ["f-1", "f-2"],
                "action": "assign",
                "params": {"user_id": "user-10"},
                "comment": "bulk",
            }
        )
        with (
            _auth_patch(),
            patch(f"{MODULE}.get_finding_workflow_store", return_value=store),
            patch(f"{MODULE}._finding_workflow_circuit_breaker") as cb,
        ):
            cb.can_proceed.return_value = True
            result = await handler._bulk_action(req)
        assert _status(result) == 200
        body = _body(result)
        assert body["success_count"] == 2
        assert body["failed_count"] == 0
        assert body["action"] == "assign"
        assert body["total"] == 2

    @pytest.mark.asyncio
    async def test_bulk_update_status(self, handler):
        store = _make_store(existing=None)
        req = MockHTTPHandler(
            body={
                "finding_ids": ["f-1"],
                "action": "update_status",
                "params": {"status": "investigating"},
            }
        )
        with (
            _auth_patch(),
            patch(f"{MODULE}.get_finding_workflow_store", return_value=store),
            patch(f"{MODULE}._finding_workflow_circuit_breaker") as cb,
        ):
            cb.can_proceed.return_value = True
            result = await handler._bulk_action(req)
        assert _status(result) == 200
        assert _body(result)["success_count"] == 1

    @pytest.mark.asyncio
    async def test_bulk_set_priority(self, handler):
        store = _make_store(existing=None)
        req = MockHTTPHandler(
            body={
                "finding_ids": ["f-1"],
                "action": "set_priority",
                "params": {"priority": 2},
            }
        )
        with (
            _auth_patch(),
            patch(f"{MODULE}.get_finding_workflow_store", return_value=store),
            patch(f"{MODULE}._finding_workflow_circuit_breaker") as cb,
        ):
            cb.can_proceed.return_value = True
            result = await handler._bulk_action(req)
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_bulk_unassign(self, handler):
        existing = _default_workflow(assigned_to="user-5")
        store = _make_store(existing=existing)
        req = MockHTTPHandler(
            body={
                "finding_ids": ["f-1"],
                "action": "unassign",
            }
        )
        with (
            _auth_patch(),
            patch(f"{MODULE}.get_finding_workflow_store", return_value=store),
            patch(f"{MODULE}._finding_workflow_circuit_breaker") as cb,
        ):
            cb.can_proceed.return_value = True
            result = await handler._bulk_action(req)
        assert _status(result) == 200
        saved = store.save.call_args[0][0]
        assert saved["assigned_to"] is None

    @pytest.mark.asyncio
    async def test_bulk_missing_finding_ids_returns_400(self, handler):
        req = MockHTTPHandler(body={"action": "assign"})
        with (
            _auth_patch(),
            patch(f"{MODULE}._finding_workflow_circuit_breaker") as cb,
        ):
            cb.can_proceed.return_value = True
            result = await handler._bulk_action(req)
        assert _status(result) == 400
        assert "finding_ids is required" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_bulk_missing_action_returns_400(self, handler):
        req = MockHTTPHandler(body={"finding_ids": ["f-1"]})
        with (
            _auth_patch(),
            patch(f"{MODULE}._finding_workflow_circuit_breaker") as cb,
        ):
            cb.can_proceed.return_value = True
            result = await handler._bulk_action(req)
        assert _status(result) == 400
        assert "action is required" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_bulk_invalid_json_returns_400(self, handler):
        req = MockHTTPHandler()
        req._raw = b"not json"
        async def bad_json():
            raise ValueError("bad")
        req.json = bad_json
        with (
            _auth_patch(),
            patch(f"{MODULE}._finding_workflow_circuit_breaker") as cb,
        ):
            cb.can_proceed.return_value = True
            result = await handler._bulk_action(req)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_bulk_circuit_breaker_open_returns_503(self, handler):
        req = MockHTTPHandler(body={"finding_ids": ["f-1"], "action": "assign"})
        with patch(f"{MODULE}._finding_workflow_circuit_breaker") as cb:
            cb.can_proceed.return_value = False
            result = await handler._bulk_action(req)
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_bulk_partial_failure(self, handler):
        """When one finding fails to process, others should still succeed."""
        call_count = 0

        async def flaky_get(fid):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("store error")
            return None  # Will create new workflow

        store = MagicMock()
        store.get = AsyncMock(side_effect=flaky_get)
        store.save = AsyncMock()

        req = MockHTTPHandler(
            body={
                "finding_ids": ["f-1", "f-2", "f-3"],
                "action": "assign",
                "params": {"user_id": "u-1"},
            }
        )
        with (
            _auth_patch(),
            patch(f"{MODULE}.get_finding_workflow_store", return_value=store),
            patch(f"{MODULE}._finding_workflow_circuit_breaker") as cb,
        ):
            cb.can_proceed.return_value = True
            result = await handler._bulk_action(req)
        body = _body(result)
        assert body["success_count"] == 2
        assert body["failed_count"] == 1

    @pytest.mark.asyncio
    async def test_bulk_empty_finding_ids_returns_400(self, handler):
        req = MockHTTPHandler(body={"finding_ids": [], "action": "assign"})
        with (
            _auth_patch(),
            patch(f"{MODULE}._finding_workflow_circuit_breaker") as cb,
        ):
            cb.can_proceed.return_value = True
            result = await handler._bulk_action(req)
        assert _status(result) == 400


# =============================================================================
# 14. _get_my_assignments Tests
# =============================================================================


class TestGetMyAssignments:
    """Test GET /api/v1/audit/findings/my-assignments."""

    @pytest.mark.asyncio
    async def test_get_my_assignments_success(self, handler):
        assignments = [
            _default_workflow("f-a", priority=2, due_date="2026-02-01"),
            _default_workflow("f-b", priority=1, due_date="2026-03-01"),
        ]
        store = _make_store(assignee_list=assignments)
        req = MockHTTPHandler()
        with _auth_patch(), patch(f"{MODULE}.get_finding_workflow_store", return_value=store):
            result = await handler._get_my_assignments(req)
        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 2
        assert body["user_id"] == "admin-001"

    @pytest.mark.asyncio
    async def test_get_my_assignments_sorted_by_priority(self, handler):
        assignments = [
            _default_workflow("f-low", priority=5),
            _default_workflow("f-high", priority=1),
            _default_workflow("f-med", priority=3),
        ]
        store = _make_store(assignee_list=assignments)
        req = MockHTTPHandler()
        with _auth_patch(), patch(f"{MODULE}.get_finding_workflow_store", return_value=store):
            result = await handler._get_my_assignments(req)
        body = _body(result)
        priorities = [f["priority"] for f in body["findings"]]
        assert priorities == sorted(priorities)

    @pytest.mark.asyncio
    async def test_get_my_assignments_empty(self, handler):
        store = _make_store(assignee_list=[])
        req = MockHTTPHandler()
        with _auth_patch(), patch(f"{MODULE}.get_finding_workflow_store", return_value=store):
            result = await handler._get_my_assignments(req)
        assert _status(result) == 200
        assert _body(result)["total"] == 0


# =============================================================================
# 15. _get_overdue Tests
# =============================================================================


class TestGetOverdue:
    """Test GET /api/v1/audit/findings/overdue."""

    @pytest.mark.asyncio
    async def test_get_overdue_success(self, handler):
        overdue = [
            _default_workflow("f-old", due_date="2025-01-01"),
            _default_workflow("f-older", due_date="2024-06-01"),
        ]
        store = _make_store(overdue_list=overdue)
        req = MockHTTPHandler()
        with _auth_patch(), patch(f"{MODULE}.get_finding_workflow_store", return_value=store):
            result = await handler._get_overdue(req)
        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 2

    @pytest.mark.asyncio
    async def test_get_overdue_sorted_by_due_date(self, handler):
        overdue = [
            _default_workflow("f-2", due_date="2025-06-01"),
            _default_workflow("f-1", due_date="2025-01-01"),
        ]
        store = _make_store(overdue_list=overdue)
        req = MockHTTPHandler()
        with _auth_patch(), patch(f"{MODULE}.get_finding_workflow_store", return_value=store):
            result = await handler._get_overdue(req)
        body = _body(result)
        dates = [f["due_date"] for f in body["findings"]]
        assert dates == sorted(dates)

    @pytest.mark.asyncio
    async def test_get_overdue_empty(self, handler):
        store = _make_store(overdue_list=[])
        req = MockHTTPHandler()
        with _auth_patch(), patch(f"{MODULE}.get_finding_workflow_store", return_value=store):
            result = await handler._get_overdue(req)
        assert _status(result) == 200
        assert _body(result)["total"] == 0


# =============================================================================
# 16. _get_workflow_states Tests
# =============================================================================


class TestGetWorkflowStates:
    """Test GET /api/v1/audit/workflow/states."""

    @pytest.mark.asyncio
    async def test_fallback_states_returned(self, handler):
        """When workflow module is not available, fallback states are returned."""
        req = MockHTTPHandler()
        with _auth_patch(), _import_error_patch():
            result = await handler._get_workflow_states(req)
        assert _status(result) == 200
        body = _body(result)
        assert "states" in body
        state_ids = [s["id"] for s in body["states"]]
        assert "open" in state_ids
        assert "resolved" in state_ids
        assert "duplicate" in state_ids
        assert "false_positive" in state_ids

    @pytest.mark.asyncio
    async def test_fallback_terminal_states_marked(self, handler):
        req = MockHTTPHandler()
        with _auth_patch(), _import_error_patch():
            result = await handler._get_workflow_states(req)
        body = _body(result)
        terminal_states = [s for s in body["states"] if s.get("is_terminal")]
        terminal_ids = [s["id"] for s in terminal_states]
        assert "resolved" in terminal_ids
        assert "false_positive" in terminal_ids
        assert "accepted_risk" in terminal_ids
        assert "duplicate" in terminal_ids

    @pytest.mark.asyncio
    async def test_fallback_states_have_transitions(self, handler):
        req = MockHTTPHandler()
        with _auth_patch(), _import_error_patch():
            result = await handler._get_workflow_states(req)
        body = _body(result)
        for state in body["states"]:
            assert "valid_transitions" in state
            assert isinstance(state["valid_transitions"], list)


# =============================================================================
# 17. _get_presets Tests
# =============================================================================


class TestGetPresets:
    """Test GET /api/v1/audit/presets."""

    @pytest.mark.asyncio
    async def test_fallback_returns_empty_presets(self, handler):
        req = MockHTTPHandler()
        with _auth_patch(), patch.dict("sys.modules", {"aragora.audit.registry": None}):
            result = await handler._get_presets(req)
        assert _status(result) == 200
        body = _body(result)
        assert body["presets"] == []
        assert body["total"] == 0


# =============================================================================
# 18. _get_audit_types Tests
# =============================================================================


class TestGetAuditTypes:
    """Test GET /api/v1/audit/types."""

    @pytest.mark.asyncio
    async def test_fallback_returns_default_types(self, handler):
        req = MockHTTPHandler()
        with _auth_patch(), patch.dict("sys.modules", {"aragora.audit.registry": None}):
            result = await handler._get_audit_types(req)
        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 4
        type_ids = [t["id"] for t in body["audit_types"]]
        assert "security" in type_ids
        assert "compliance" in type_ids
        assert "consistency" in type_ids
        assert "quality" in type_ids

    @pytest.mark.asyncio
    async def test_fallback_types_have_display_name(self, handler):
        req = MockHTTPHandler()
        with _auth_patch(), patch.dict("sys.modules", {"aragora.audit.registry": None}):
            result = await handler._get_audit_types(req)
        body = _body(result)
        for t in body["audit_types"]:
            assert "display_name" in t
            assert len(t["display_name"]) > 0


# =============================================================================
# 19. Initialization Tests
# =============================================================================


class TestHandlerInit:
    """Test handler instantiation."""

    def test_init_with_server_context(self):
        h = FindingWorkflowHandler(server_context={"key": "val"})
        assert h.ctx == {"key": "val"}

    def test_init_with_ctx_kwarg(self):
        h = FindingWorkflowHandler(ctx={"key2": "val2"})
        assert h.ctx == {"key2": "val2"}

    def test_init_defaults_to_empty_dict(self):
        h = FindingWorkflowHandler()
        assert h.ctx == {}

    def test_server_context_takes_precedence_over_ctx(self):
        h = FindingWorkflowHandler(ctx={"old": 1}, server_context={"new": 2})
        assert h.ctx == {"new": 2}


# =============================================================================
# 20. _get_or_create_workflow Tests
# =============================================================================


class TestGetOrCreateWorkflow:
    """Test internal _get_or_create_workflow method."""

    @pytest.mark.asyncio
    async def test_creates_new_when_not_found(self, handler):
        store = _make_store(existing=None)
        with patch(f"{MODULE}.get_finding_workflow_store", return_value=store):
            wf = await handler._get_or_create_workflow("new-finding")
        assert wf["finding_id"] == "new-finding"
        assert wf["current_state"] == "open"
        assert wf["priority"] == 3
        assert wf["history"] == []
        assert wf["assigned_to"] is None
        store.save.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_returns_existing_workflow(self, handler):
        existing = _default_workflow("existing-1", priority=1, assigned_to="user-x")
        store = _make_store(existing=existing)
        with patch(f"{MODULE}.get_finding_workflow_store", return_value=store):
            wf = await handler._get_or_create_workflow("existing-1")
        assert wf["priority"] == 1
        assert wf["assigned_to"] == "user-x"
        store.save.assert_not_awaited()


# =============================================================================
# 21. _get_user_from_request Tests
# =============================================================================


class TestGetUserFromRequest:
    """Test _get_user_from_request behavior."""

    def test_returns_admin_user(self, handler):
        with _auth_patch():
            user_id, user_name = handler._get_user_from_request(MagicMock())
        assert user_id == "admin-001"

    def test_returns_anonymous_for_unauthenticated(self, handler):
        with patch(f"{MODULE}.extract_user_from_request", return_value=_UnauthenticatedJWT()):
            user_id, user_name = handler._get_user_from_request(MagicMock())
        assert user_id == "anonymous"
        assert user_name == "Anonymous User"

    def test_strips_email_domain_for_display(self, handler):
        class _EmailJWT:
            authenticated = True
            user_id = "user-1"
            role = "admin"
            org_id = "org-1"
            client_ip = "127.0.0.1"
            email = "johndoe@example.com"

        with patch(f"{MODULE}.extract_user_from_request", return_value=_EmailJWT()):
            _, user_name = handler._get_user_from_request(MagicMock())
        assert user_name == "johndoe"


# =============================================================================
# 22. Error Handling Patterns
# =============================================================================


class TestErrorHandling:
    """Test error response formatting."""

    def test_error_response_format(self, handler):
        result = handler._error_response(404, "Not found")
        assert _status(result) == 404
        body = _body(result)
        assert body["error"] == "Not found"

    def test_json_response_format(self, handler):
        result = handler._json_response(200, {"key": "value"})
        assert _status(result) == 200
        assert result["headers"]["Content-Type"] == "application/json"
        body = _body(result)
        assert body["key"] == "value"

    def test_error_response_has_json_content_type(self, handler):
        result = handler._error_response(500, "Internal error")
        assert result["headers"]["Content-Type"] == "application/json"


# =============================================================================
# 23. _parse_json_body Tests
# =============================================================================


class TestParseJsonBody:
    """Test _parse_json_body handles different request formats."""

    @pytest.mark.asyncio
    async def test_parses_json_method(self, handler):
        req = MockHTTPHandler(body={"key": "value"})
        result = await handler._parse_json_body(req)
        assert result["key"] == "value"

    @pytest.mark.asyncio
    async def test_parses_read_method(self, handler):
        req = MagicMock()
        del req.json  # Remove json method so read is used
        req.read = AsyncMock(return_value=b'{"key": "value2"}')
        result = await handler._parse_json_body(req)
        assert result["key"] == "value2"

    @pytest.mark.asyncio
    async def test_parses_body_method(self, handler):
        req = MagicMock()
        del req.json
        del req.read
        req.body = AsyncMock(return_value=b'{"key": "value3"}')
        result = await handler._parse_json_body(req)
        assert result["key"] == "value3"

    @pytest.mark.asyncio
    async def test_returns_empty_dict_when_no_method(self, handler):
        req = MagicMock(spec=[])  # No attributes at all
        result = await handler._parse_json_body(req)
        assert result == {}


# =============================================================================
# 24. Authentication Denial Tests (no_auto_auth)
# =============================================================================


@pytest.mark.no_auto_auth
class TestAuthenticationDenials:
    """Test that unauthenticated requests are rejected."""

    @pytest.mark.asyncio
    async def test_unauthenticated_update_status_returns_401(self):
        h = FindingWorkflowHandler(server_context={})
        req = MockHTTPHandler(body={"status": "triaging"})
        with (
            patch(f"{MODULE}.extract_user_from_request", return_value=_UnauthenticatedJWT()),
            patch(f"{MODULE}._finding_workflow_circuit_breaker") as cb,
        ):
            cb.can_proceed.return_value = True
            result = await h._update_status(req, "f-1")
        assert _status(result) == 401

    @pytest.mark.asyncio
    async def test_unauthenticated_assign_returns_401(self):
        h = FindingWorkflowHandler(server_context={})
        req = MockHTTPHandler(body={"user_id": "u-1"})
        with patch(f"{MODULE}.extract_user_from_request", return_value=_UnauthenticatedJWT()):
            result = await h._assign(req, "f-1")
        assert _status(result) == 401

    @pytest.mark.asyncio
    async def test_unauthenticated_get_history_returns_401(self):
        h = FindingWorkflowHandler(server_context={})
        req = MockHTTPHandler()
        with patch(f"{MODULE}.extract_user_from_request", return_value=_UnauthenticatedJWT()):
            result = await h._get_history(req, "f-1")
        assert _status(result) == 401

    @pytest.mark.asyncio
    async def test_unauthenticated_bulk_action_returns_401(self):
        h = FindingWorkflowHandler(server_context={})
        req = MockHTTPHandler(body={"finding_ids": ["f-1"], "action": "assign"})
        with (
            patch(f"{MODULE}.extract_user_from_request", return_value=_UnauthenticatedJWT()),
            patch(f"{MODULE}._finding_workflow_circuit_breaker") as cb,
        ):
            cb.can_proceed.return_value = True
            result = await h._bulk_action(req)
        assert _status(result) == 401

    @pytest.mark.asyncio
    async def test_unauthenticated_my_assignments_returns_401(self):
        h = FindingWorkflowHandler(server_context={})
        req = MockHTTPHandler()
        with patch(f"{MODULE}.extract_user_from_request", return_value=_UnauthenticatedJWT()):
            result = await h._get_my_assignments(req)
        assert _status(result) == 401

    @pytest.mark.asyncio
    async def test_unauthenticated_overdue_returns_401(self):
        h = FindingWorkflowHandler(server_context={})
        req = MockHTTPHandler()
        with patch(f"{MODULE}.extract_user_from_request", return_value=_UnauthenticatedJWT()):
            result = await h._get_overdue(req)
        assert _status(result) == 401
