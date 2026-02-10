"""
Comprehensive tests for FindingWorkflowHandler.

Covers all handler endpoints with success cases, error cases, edge cases,
workflow state transitions, routing, and circuit breaker / rate limit behavior.

Target: 25-35 tests covering all handler endpoints.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.features.finding_workflow import FindingWorkflowHandler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MODULE = "aragora.server.handlers.features.finding_workflow"


class _MockRequest:
    """Lightweight mock HTTP request used across tests."""

    def __init__(
        self,
        method: str = "GET",
        path: str = "/api/v1/audit/findings/f-1/status",
        body: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ):
        self.method = method
        self.path = path
        self.headers = headers or {}
        self._body = json.dumps(body).encode() if body is not None else b"{}"

    async def read(self) -> bytes:
        return self._body


class _AdminJWT:
    """Simulates an authenticated admin JWT context."""

    authenticated = True
    user_id = "admin-001"
    role = "admin"
    org_id = "org-1"
    client_ip = "127.0.0.1"
    email = "admin@example.com"


class _AnonJWT:
    """Simulates an unauthenticated JWT context."""

    authenticated = False
    user_id = None
    role = None
    org_id = None
    client_ip = None
    email = None


def _make_handler() -> FindingWorkflowHandler:
    return FindingWorkflowHandler(server_context={})


def _make_store_mock(
    existing: dict[str, Any] | None = None,
    assignee_list: list[dict] | None = None,
    overdue_list: list[dict] | None = None,
) -> MagicMock:
    """Build a mock FindingWorkflowStoreBackend."""
    store = MagicMock()
    store.get = AsyncMock(return_value=existing)
    store.save = AsyncMock()
    store.list_by_assignee = AsyncMock(return_value=assignee_list or [])
    store.list_overdue = AsyncMock(return_value=overdue_list or [])
    return store


def _default_workflow(finding_id: str = "f-1", **overrides: Any) -> dict[str, Any]:
    """Return a minimal but complete workflow dict."""
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


def _ctx(*patches):
    """Convenience wrapper combining multiple context manager patches."""
    from contextlib import ExitStack

    stack = ExitStack()
    for p in patches:
        stack.enter_context(p)
    return stack


def _body(resp: dict) -> dict:
    """Parse the JSON body from a handler response dict."""
    return json.loads(resp.get("body", "{}"))


# ---------------------------------------------------------------------------
# Tests: Initialization and routing
# ---------------------------------------------------------------------------


class TestHandlerInit:
    """Handler instantiation and basic attributes."""

    def test_init_with_server_context(self):
        h = FindingWorkflowHandler(server_context={"key": "val"})
        assert h.ctx == {"key": "val"}

    def test_init_with_ctx_kwarg(self):
        h = FindingWorkflowHandler(ctx={"key2": "val2"})
        assert h.ctx == {"key2": "val2"}

    def test_init_defaults_to_empty(self):
        h = FindingWorkflowHandler()
        assert h.ctx == {}

    def test_routes_list_populated(self):
        assert len(FindingWorkflowHandler.ROUTES) >= 15


class TestRequestRouting:
    """Routing within handle_request."""

    @pytest.mark.asyncio
    async def test_route_workflow_states(self):
        h = _make_handler()
        req = _MockRequest(method="GET", path="/api/v1/audit/workflow/states")
        with (
            patch(f"{MODULE}.extract_user_from_request", return_value=_AdminJWT()),
            patch.object(h, "_get_workflow_states", new_callable=AsyncMock, return_value={"status": 200, "body": "{}"}),
        ):
            result = await h.handle_request(req)
        assert result["status"] == 200

    @pytest.mark.asyncio
    async def test_route_presets(self):
        h = _make_handler()
        req = _MockRequest(method="GET", path="/api/v1/audit/presets")
        with (
            patch(f"{MODULE}.extract_user_from_request", return_value=_AdminJWT()),
            patch.object(h, "_get_presets", new_callable=AsyncMock, return_value={"status": 200, "body": "{}"}),
        ):
            result = await h.handle_request(req)
        assert result["status"] == 200

    @pytest.mark.asyncio
    async def test_route_types(self):
        h = _make_handler()
        req = _MockRequest(method="GET", path="/api/v1/audit/types")
        with (
            patch(f"{MODULE}.extract_user_from_request", return_value=_AdminJWT()),
            patch.object(h, "_get_audit_types", new_callable=AsyncMock, return_value={"status": 200, "body": "{}"}),
        ):
            result = await h.handle_request(req)
        assert result["status"] == 200

    @pytest.mark.asyncio
    async def test_route_unknown_returns_404(self):
        h = _make_handler()
        req = MagicMock()
        req.method = "GET"
        req.path = "/api/v1/audit/unknown/endpoint"
        result = await h.handle_request(req)
        assert result["status"] == 404

    @pytest.mark.asyncio
    async def test_finding_id_parsing_for_status(self):
        h = _make_handler()
        req = _MockRequest(method="PATCH", path="/api/v1/audit/findings/my-finding/status", body={"status": "open"})
        with (
            patch(f"{MODULE}.extract_user_from_request", return_value=_AdminJWT()),
            patch.object(h, "_update_status", new_callable=AsyncMock, return_value={"status": 200, "body": "{}"}) as mock_us,
        ):
            await h.handle_request(req)
        mock_us.assert_awaited_once()
        # Verify finding_id was passed correctly
        assert mock_us.call_args[0][1] == "my-finding"


# ---------------------------------------------------------------------------
# Tests: Update status (fallback path -- ImportError branch)
# ---------------------------------------------------------------------------


class TestUpdateStatusSuccess:
    """Tests the update_status handler with a real workflow module."""

    @pytest.mark.asyncio
    async def test_update_status_success_via_workflow(self):
        """Test successful status update using the real FindingWorkflow module."""
        h = _make_handler()
        existing = _default_workflow(current_state="open")
        store = _make_store_mock(existing=existing)
        req = _MockRequest(body={"status": "triaging", "comment": "triage now"})

        with (
            patch(f"{MODULE}.extract_user_from_request", return_value=_AdminJWT()),
            patch(f"{MODULE}.get_finding_workflow_store", return_value=store),
            patch(f"{MODULE}._finding_workflow_circuit_breaker") as cb,
        ):
            cb.can_proceed.return_value = True
            result = await h._update_status(req, "f-1")

        assert result["status"] == 200
        body = _body(result)
        assert body["success"] is True
        # Verify the store was used (save called at least once -- for get_or_create or for final persist)
        assert store.save.await_count >= 1

    @pytest.mark.asyncio
    async def test_update_status_missing_status_field(self):
        h = _make_handler()
        req = _MockRequest(body={})

        with (
            patch(f"{MODULE}.extract_user_from_request", return_value=_AdminJWT()),
            patch(f"{MODULE}._finding_workflow_circuit_breaker") as cb,
        ):
            cb.can_proceed.return_value = True
            result = await h._update_status(req, "f-1")

        assert result["status"] == 400

    @pytest.mark.asyncio
    async def test_update_status_invalid_json(self):
        h = _make_handler()
        req = _MockRequest()
        req._body = b"<<<not json>>>"

        with (
            patch(f"{MODULE}.extract_user_from_request", return_value=_AdminJWT()),
            patch(f"{MODULE}._finding_workflow_circuit_breaker") as cb,
        ):
            cb.can_proceed.return_value = True
            result = await h._update_status(req, "f-1")

        assert result["status"] == 400

    @pytest.mark.asyncio
    async def test_update_status_circuit_breaker_open(self):
        h = _make_handler()
        req = _MockRequest(body={"status": "triaging"})

        with patch(f"{MODULE}._finding_workflow_circuit_breaker") as cb:
            cb.can_proceed.return_value = False
            result = await h._update_status(req, "f-1")

        assert result["status"] == 503


# ---------------------------------------------------------------------------
# Tests: Assign / Unassign
# ---------------------------------------------------------------------------


class TestAssignment:

    @pytest.mark.asyncio
    async def test_assign_success(self):
        h = _make_handler()
        store = _make_store_mock(existing=None)
        req = _MockRequest(body={"user_id": "user-2", "comment": "please review"})

        with (
            patch(f"{MODULE}.extract_user_from_request", return_value=_AdminJWT()),
            patch(f"{MODULE}.get_finding_workflow_store", return_value=store),
            # Force ImportError fallback
            patch.dict("sys.modules", {"aragora.audit.findings.workflow": None}),
        ):
            result = await h._assign(req, "f-1")

        assert result["status"] == 200
        body = _body(result)
        assert body["success"] is True
        assert body["assigned_to"] == "user-2"

    @pytest.mark.asyncio
    async def test_assign_missing_user_id(self):
        h = _make_handler()
        req = _MockRequest(body={"comment": "no user"})

        with patch(f"{MODULE}.extract_user_from_request", return_value=_AdminJWT()):
            result = await h._assign(req, "f-1")

        assert result["status"] == 400

    @pytest.mark.asyncio
    async def test_unassign_success(self):
        h = _make_handler()
        existing = _default_workflow(assigned_to="user-2")
        store = _make_store_mock(existing=existing)
        req = _MockRequest(body={"comment": "freeing up"})

        with (
            patch(f"{MODULE}.extract_user_from_request", return_value=_AdminJWT()),
            patch(f"{MODULE}.get_finding_workflow_store", return_value=store),
            patch.dict("sys.modules", {"aragora.audit.findings.workflow": None}),
        ):
            result = await h._unassign(req, "f-1")

        assert result["status"] == 200
        body = _body(result)
        assert body["success"] is True

    @pytest.mark.asyncio
    async def test_unassign_without_body_still_succeeds(self):
        """Unassign should work even if no JSON body is provided."""
        h = _make_handler()
        existing = _default_workflow(assigned_to="user-2")
        store = _make_store_mock(existing=existing)
        req = _MockRequest()
        req._body = b"<<<invalid>>>"  # Body parsing will fail but is optional

        with (
            patch(f"{MODULE}.extract_user_from_request", return_value=_AdminJWT()),
            patch(f"{MODULE}.get_finding_workflow_store", return_value=store),
            patch.dict("sys.modules", {"aragora.audit.findings.workflow": None}),
        ):
            result = await h._unassign(req, "f-1")

        assert result["status"] == 200


# ---------------------------------------------------------------------------
# Tests: Comments
# ---------------------------------------------------------------------------


class TestComments:

    @pytest.mark.asyncio
    async def test_add_comment_success(self):
        h = _make_handler()
        existing = _default_workflow()
        store = _make_store_mock(existing=existing)
        req = _MockRequest(body={"comment": "Needs investigation"})

        with (
            patch(f"{MODULE}.extract_user_from_request", return_value=_AdminJWT()),
            patch(f"{MODULE}.get_finding_workflow_store", return_value=store),
            patch.dict("sys.modules", {"aragora.audit.findings.workflow": None}),
        ):
            result = await h._add_comment(req, "f-1")

        assert result["status"] == 201
        body = _body(result)
        assert body["success"] is True

    @pytest.mark.asyncio
    async def test_add_comment_missing_comment(self):
        h = _make_handler()
        req = _MockRequest(body={})

        with patch(f"{MODULE}.extract_user_from_request", return_value=_AdminJWT()):
            result = await h._add_comment(req, "f-1")

        assert result["status"] == 400

    @pytest.mark.asyncio
    async def test_get_comments_filters_by_event_type(self):
        h = _make_handler()
        existing = _default_workflow(
            history=[
                {"event_type": "comment", "comment": "first"},
                {"event_type": "state_change"},
                {"event_type": "comment", "comment": "second"},
            ]
        )
        store = _make_store_mock(existing=existing)
        req = _MockRequest()

        with (
            patch(f"{MODULE}.extract_user_from_request", return_value=_AdminJWT()),
            patch(f"{MODULE}.get_finding_workflow_store", return_value=store),
        ):
            result = await h._get_comments(req, "f-1")

        assert result["status"] == 200
        body = _body(result)
        assert body["total"] == 2
        assert len(body["comments"]) == 2


# ---------------------------------------------------------------------------
# Tests: History
# ---------------------------------------------------------------------------


class TestHistory:

    @pytest.mark.asyncio
    async def test_get_history_returns_full_data(self):
        h = _make_handler()
        existing = _default_workflow(
            assigned_to="user-x",
            priority=1,
            due_date="2026-03-01T00:00:00+00:00",
            history=[{"event_type": "state_change"}],
        )
        store = _make_store_mock(existing=existing)
        req = _MockRequest()

        with (
            patch(f"{MODULE}.extract_user_from_request", return_value=_AdminJWT()),
            patch(f"{MODULE}.get_finding_workflow_store", return_value=store),
        ):
            result = await h._get_history(req, "f-1")

        assert result["status"] == 200
        body = _body(result)
        assert body["finding_id"] == "f-1"
        assert body["current_state"] == "open"
        assert body["assigned_to"] == "user-x"
        assert body["priority"] == 1
        assert len(body["history"]) == 1


# ---------------------------------------------------------------------------
# Tests: Priority
# ---------------------------------------------------------------------------


class TestPriority:

    @pytest.mark.asyncio
    async def test_set_priority_valid(self):
        h = _make_handler()
        existing = _default_workflow()
        store = _make_store_mock(existing=existing)
        req = _MockRequest(body={"priority": 1, "comment": "urgent"})

        with (
            patch(f"{MODULE}.extract_user_from_request", return_value=_AdminJWT()),
            patch(f"{MODULE}.get_finding_workflow_store", return_value=store),
            patch.dict("sys.modules", {"aragora.audit.findings.workflow": None}),
        ):
            result = await h._set_priority(req, "f-1")

        assert result["status"] == 200
        body = _body(result)
        assert body["priority"] == 1

    @pytest.mark.asyncio
    async def test_set_priority_out_of_range(self):
        h = _make_handler()
        req = _MockRequest(body={"priority": 0})

        with patch(f"{MODULE}.extract_user_from_request", return_value=_AdminJWT()):
            result = await h._set_priority(req, "f-1")

        assert result["status"] == 400

    @pytest.mark.asyncio
    async def test_set_priority_non_integer(self):
        h = _make_handler()
        req = _MockRequest(body={"priority": "high"})

        with patch(f"{MODULE}.extract_user_from_request", return_value=_AdminJWT()):
            result = await h._set_priority(req, "f-1")

        assert result["status"] == 400


# ---------------------------------------------------------------------------
# Tests: Due date
# ---------------------------------------------------------------------------


class TestDueDate:

    @pytest.mark.asyncio
    async def test_set_due_date_success(self):
        h = _make_handler()
        existing = _default_workflow()
        store = _make_store_mock(existing=existing)
        req = _MockRequest(body={"due_date": "2026-06-01T12:00:00Z"})

        with (
            patch(f"{MODULE}.extract_user_from_request", return_value=_AdminJWT()),
            patch(f"{MODULE}.get_finding_workflow_store", return_value=store),
        ):
            result = await h._set_due_date(req, "f-1")

        assert result["status"] == 200
        body = _body(result)
        assert body["due_date"] is not None

    @pytest.mark.asyncio
    async def test_clear_due_date(self):
        h = _make_handler()
        existing = _default_workflow(due_date="2026-01-01T00:00:00+00:00")
        store = _make_store_mock(existing=existing)
        req = _MockRequest(body={"due_date": None})

        with (
            patch(f"{MODULE}.extract_user_from_request", return_value=_AdminJWT()),
            patch(f"{MODULE}.get_finding_workflow_store", return_value=store),
        ):
            result = await h._set_due_date(req, "f-1")

        assert result["status"] == 200
        body = _body(result)
        assert body["due_date"] is None

    @pytest.mark.asyncio
    async def test_set_due_date_invalid_format(self):
        h = _make_handler()
        req = _MockRequest(body={"due_date": "next-tuesday"})

        with patch(f"{MODULE}.extract_user_from_request", return_value=_AdminJWT()):
            result = await h._set_due_date(req, "f-1")

        assert result["status"] == 400


# ---------------------------------------------------------------------------
# Tests: Link finding / Mark duplicate
# ---------------------------------------------------------------------------


class TestLinkAndDuplicate:

    @pytest.mark.asyncio
    async def test_link_finding_success(self):
        h = _make_handler()
        existing = _default_workflow()
        store = _make_store_mock(existing=existing)
        req = _MockRequest(body={"linked_finding_id": "f-2", "comment": "related"})

        with (
            patch(f"{MODULE}.extract_user_from_request", return_value=_AdminJWT()),
            patch(f"{MODULE}.get_finding_workflow_store", return_value=store),
        ):
            result = await h._link_finding(req, "f-1")

        assert result["status"] == 200
        body = _body(result)
        assert "f-2" in body["linked_findings"]

    @pytest.mark.asyncio
    async def test_link_finding_no_duplicate_entries(self):
        """Linking the same finding twice should not create duplicates."""
        h = _make_handler()
        existing = _default_workflow(linked_findings=["f-2"])
        store = _make_store_mock(existing=existing)
        req = _MockRequest(body={"linked_finding_id": "f-2"})

        with (
            patch(f"{MODULE}.extract_user_from_request", return_value=_AdminJWT()),
            patch(f"{MODULE}.get_finding_workflow_store", return_value=store),
        ):
            result = await h._link_finding(req, "f-1")

        assert result["status"] == 200
        body = _body(result)
        assert body["linked_findings"].count("f-2") == 1

    @pytest.mark.asyncio
    async def test_link_finding_missing_id(self):
        h = _make_handler()
        req = _MockRequest(body={})

        with patch(f"{MODULE}.extract_user_from_request", return_value=_AdminJWT()):
            result = await h._link_finding(req, "f-1")

        assert result["status"] == 400

    @pytest.mark.asyncio
    async def test_mark_duplicate_success(self):
        h = _make_handler()
        existing = _default_workflow()
        store = _make_store_mock(existing=existing)
        req = _MockRequest(body={"parent_finding_id": "f-original"})

        with (
            patch(f"{MODULE}.extract_user_from_request", return_value=_AdminJWT()),
            patch(f"{MODULE}.get_finding_workflow_store", return_value=store),
        ):
            result = await h._mark_duplicate(req, "f-1")

        assert result["status"] == 200
        body = _body(result)
        assert body["current_state"] == "duplicate"
        assert body["parent_finding_id"] == "f-original"

    @pytest.mark.asyncio
    async def test_mark_duplicate_missing_parent(self):
        h = _make_handler()
        req = _MockRequest(body={})

        with patch(f"{MODULE}.extract_user_from_request", return_value=_AdminJWT()):
            result = await h._mark_duplicate(req, "f-1")

        assert result["status"] == 400


# ---------------------------------------------------------------------------
# Tests: Bulk actions
# ---------------------------------------------------------------------------


class TestBulkActions:

    @pytest.mark.asyncio
    async def test_bulk_assign_success(self):
        h = _make_handler()
        store = _make_store_mock(existing=None)  # will auto-create
        req = _MockRequest(
            body={
                "finding_ids": ["f-1", "f-2"],
                "action": "assign",
                "params": {"user_id": "user-10"},
                "comment": "bulk assign",
            }
        )

        with (
            patch(f"{MODULE}.extract_user_from_request", return_value=_AdminJWT()),
            patch(f"{MODULE}.get_finding_workflow_store", return_value=store),
            patch(f"{MODULE}._finding_workflow_circuit_breaker") as cb,
        ):
            cb.can_proceed.return_value = True
            result = await h._bulk_action(req)

        assert result["status"] == 200
        body = _body(result)
        assert body["success_count"] == 2
        assert body["failed_count"] == 0

    @pytest.mark.asyncio
    async def test_bulk_action_missing_finding_ids(self):
        h = _make_handler()
        req = _MockRequest(body={"action": "assign"})

        with (
            patch(f"{MODULE}.extract_user_from_request", return_value=_AdminJWT()),
            patch(f"{MODULE}._finding_workflow_circuit_breaker") as cb,
        ):
            cb.can_proceed.return_value = True
            result = await h._bulk_action(req)

        assert result["status"] == 400

    @pytest.mark.asyncio
    async def test_bulk_action_missing_action(self):
        h = _make_handler()
        req = _MockRequest(body={"finding_ids": ["f-1"]})

        with (
            patch(f"{MODULE}.extract_user_from_request", return_value=_AdminJWT()),
            patch(f"{MODULE}._finding_workflow_circuit_breaker") as cb,
        ):
            cb.can_proceed.return_value = True
            result = await h._bulk_action(req)

        assert result["status"] == 400

    @pytest.mark.asyncio
    async def test_bulk_action_circuit_open(self):
        h = _make_handler()
        req = _MockRequest(body={"finding_ids": ["f-1"], "action": "assign"})

        with patch(f"{MODULE}._finding_workflow_circuit_breaker") as cb:
            cb.can_proceed.return_value = False
            result = await h._bulk_action(req)

        assert result["status"] == 503


# ---------------------------------------------------------------------------
# Tests: My assignments / Overdue
# ---------------------------------------------------------------------------


class TestAssignmentsAndOverdue:

    @pytest.mark.asyncio
    async def test_get_my_assignments(self):
        h = _make_handler()
        assignments = [
            _default_workflow("f-a", priority=2, due_date="2026-02-01"),
            _default_workflow("f-b", priority=1, due_date="2026-03-01"),
        ]
        store = _make_store_mock(assignee_list=assignments)
        req = _MockRequest()

        with (
            patch(f"{MODULE}.extract_user_from_request", return_value=_AdminJWT()),
            patch(f"{MODULE}.get_finding_workflow_store", return_value=store),
        ):
            result = await h._get_my_assignments(req)

        assert result["status"] == 200
        body = _body(result)
        assert body["total"] == 2
        # Should be sorted by priority then due_date
        assert body["findings"][0]["priority"] <= body["findings"][1]["priority"]

    @pytest.mark.asyncio
    async def test_get_overdue(self):
        h = _make_handler()
        overdue = [
            _default_workflow("f-old", due_date="2025-01-01"),
            _default_workflow("f-older", due_date="2024-06-01"),
        ]
        store = _make_store_mock(overdue_list=overdue)
        req = _MockRequest()

        with (
            patch(f"{MODULE}.extract_user_from_request", return_value=_AdminJWT()),
            patch(f"{MODULE}.get_finding_workflow_store", return_value=store),
        ):
            result = await h._get_overdue(req)

        assert result["status"] == 200
        body = _body(result)
        assert body["total"] == 2


# ---------------------------------------------------------------------------
# Tests: Static endpoints (states, presets, audit types)
# ---------------------------------------------------------------------------


class TestStaticEndpoints:

    @pytest.mark.asyncio
    async def test_get_workflow_states_fallback(self):
        """When workflow module is unavailable, fallback states are returned."""
        h = _make_handler()
        req = _MockRequest()

        with patch(f"{MODULE}.extract_user_from_request", return_value=_AdminJWT()):
            result = await h._get_workflow_states(req)

        assert result["status"] == 200
        body = _body(result)
        assert "states" in body
        state_ids = [s["id"] for s in body["states"]]
        assert "open" in state_ids
        assert "resolved" in state_ids

    @pytest.mark.asyncio
    async def test_get_presets_fallback(self):
        h = _make_handler()
        req = _MockRequest()

        with patch(f"{MODULE}.extract_user_from_request", return_value=_AdminJWT()):
            result = await h._get_presets(req)

        assert result["status"] == 200
        body = _body(result)
        assert "presets" in body

    @pytest.mark.asyncio
    async def test_get_audit_types_fallback(self):
        h = _make_handler()
        req = _MockRequest()

        with patch(f"{MODULE}.extract_user_from_request", return_value=_AdminJWT()):
            result = await h._get_audit_types(req)

        assert result["status"] == 200
        body = _body(result)
        assert "audit_types" in body
        assert body["total"] >= 4


# ---------------------------------------------------------------------------
# Tests: Permission checks (unauthenticated)
# ---------------------------------------------------------------------------


class TestPermissionDenials:

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_unauthenticated_user_gets_401(self):
        h = _make_handler()
        req = _MockRequest()

        with patch(f"{MODULE}.extract_user_from_request", return_value=_AnonJWT()):
            result = await h._get_history(req, "f-1")

        assert result["status"] == 401


# ---------------------------------------------------------------------------
# Tests: _get_or_create_workflow
# ---------------------------------------------------------------------------


class TestGetOrCreateWorkflow:

    @pytest.mark.asyncio
    async def test_creates_new_when_not_found(self):
        h = _make_handler()
        store = _make_store_mock(existing=None)

        with patch(f"{MODULE}.get_finding_workflow_store", return_value=store):
            wf = await h._get_or_create_workflow("new-finding")

        assert wf["finding_id"] == "new-finding"
        assert wf["current_state"] == "open"
        store.save.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_returns_existing(self):
        h = _make_handler()
        existing = _default_workflow("existing-1", priority=1)
        store = _make_store_mock(existing=existing)

        with patch(f"{MODULE}.get_finding_workflow_store", return_value=store):
            wf = await h._get_or_create_workflow("existing-1")

        assert wf["priority"] == 1
        store.save.assert_not_awaited()
