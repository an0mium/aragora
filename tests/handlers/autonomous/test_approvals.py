"""Comprehensive tests for ApprovalHandler.

Tests cover:
- list_pending (GET /api/v1/autonomous/approvals/pending)
- get_request (GET /api/v1/autonomous/approvals/{request_id})
- approve (POST /api/v1/autonomous/approvals/{request_id}/approve)
- reject (POST /api/v1/autonomous/approvals/{request_id}/reject)
- Auth / permission checks (401, 403)
- Circuit breaker behaviour (503)
- Error-handling paths (500)
- RBAC permission escalation (_ensure_auth_context)
- Admin bypass (_is_admin)
- Global accessors (get/set_approval_flow)
- register_routes
- Handler init
- Security edge cases (path traversal, injection)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import Enum
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web

from aragora.server.handlers.autonomous.approvals import (
    ApprovalHandler,
    get_approval_flow,
    set_approval_flow,
    _get_circuit_breaker,
    _ensure_auth_context,
    _is_admin,
    AUTONOMOUS_READ_PERMISSION,
    AUTONOMOUS_APPROVE_PERMISSION,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _parse(response: web.Response) -> dict:
    """Extract JSON dict from an aiohttp json_response."""
    return json.loads(response.body)


def _make_request(
    method: str = "GET",
    query: dict | None = None,
    match_info: dict | None = None,
    body: dict | None = None,
) -> MagicMock:
    """Build a MagicMock that mimics an aiohttp web.Request."""
    req = MagicMock()
    req.method = method
    req.query = query or {}

    mi_data = match_info or {}
    mi_mock = MagicMock()
    mi_mock.get = MagicMock(side_effect=lambda k, default=None: mi_data.get(k, default))
    req.match_info = mi_mock

    if body is not None:
        req.json = AsyncMock(return_value=body)
        raw = json.dumps(body).encode()
        req.read = AsyncMock(return_value=raw)
        req.text = AsyncMock(return_value=json.dumps(body))
        req.content_type = "application/json"
        req.content_length = len(raw)
        req.can_read_body = True
    else:
        req.json = AsyncMock(return_value={})
        req.read = AsyncMock(return_value=b"{}")
        req.text = AsyncMock(return_value="{}")
        req.content_type = "application/json"
        req.content_length = 2
        req.can_read_body = True

    req.remote = "127.0.0.1"
    req.transport = MagicMock()
    req.transport.get_extra_info.return_value = ("127.0.0.1", 12345)

    return req


class _MockApprovalStatus(Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    AUTO_APPROVED = "auto_approved"
    TIMEOUT = "timeout"


def _make_approval_request(
    request_id: str = "req-001",
    title: str = "Refactor module X",
    description: str = "Restructure module X for better performance",
    changes: list | None = None,
    risk_level: str = "medium",
    requested_at: datetime | None = None,
    requested_by: str = "agent-1",
    timeout_seconds: int = 3600,
    status: _MockApprovalStatus = _MockApprovalStatus.PENDING,
    approved_by: str | None = None,
    approved_at: datetime | None = None,
    rejection_reason: str | None = None,
    metadata: dict | None = None,
) -> MagicMock:
    """Build a mock ApprovalRequest."""
    obj = MagicMock()
    obj.id = request_id
    obj.title = title
    obj.description = description
    obj.changes = changes or [{"file": "module_x.py", "action": "refactor"}]
    obj.risk_level = risk_level
    obj.requested_at = requested_at or datetime(2026, 2, 15, 10, 0, 0, tzinfo=timezone.utc)
    obj.requested_by = requested_by
    obj.timeout_seconds = timeout_seconds
    obj.status = status
    obj.approved_by = approved_by
    obj.approved_at = approved_at
    obj.rejection_reason = rejection_reason
    obj.metadata = metadata or {}
    return obj


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_approval_globals():
    """Reset global approval flow and circuit breaker between tests."""
    import aragora.server.handlers.autonomous.approvals as mod

    old_flow = mod._approval_flow
    old_cb = mod._approval_circuit_breaker
    mod._approval_flow = None
    mod._approval_circuit_breaker = None
    yield
    mod._approval_flow = old_flow
    mod._approval_circuit_breaker = old_cb


@pytest.fixture
def mock_flow():
    """Create a mock ApprovalFlow instance."""
    flow = MagicMock()
    flow.list_pending = MagicMock(return_value=[])
    flow._load_request = MagicMock(return_value=None)
    flow.approve = MagicMock(
        return_value=_make_approval_request(
            status=_MockApprovalStatus.APPROVED,
            approved_by="test-user-001",
            approved_at=datetime(2026, 2, 15, 11, 0, 0, tzinfo=timezone.utc),
        )
    )
    flow.reject = MagicMock(
        return_value=_make_approval_request(
            status=_MockApprovalStatus.REJECTED,
            approved_by="test-user-001",
            rejection_reason="Too risky",
        )
    )
    return flow


@pytest.fixture
def install_flow(mock_flow):
    """Set mock flow as the global singleton."""
    set_approval_flow(mock_flow)
    return mock_flow


@pytest.fixture
def mock_cb():
    """Create a mock circuit breaker that allows execution."""
    cb = MagicMock()
    cb.can_execute.return_value = True
    return cb


@pytest.fixture
def install_cb(mock_cb):
    """Patch _get_circuit_breaker to return our mock."""
    with patch(
        "aragora.server.handlers.autonomous.approvals._get_circuit_breaker",
        return_value=mock_cb,
    ):
        yield mock_cb


# ---------------------------------------------------------------------------
# list_pending endpoint
# ---------------------------------------------------------------------------


class TestListPending:
    @pytest.mark.asyncio
    async def test_list_empty(self, install_flow, install_cb):
        install_flow.list_pending.return_value = []
        req = _make_request()
        resp = await ApprovalHandler.list_pending(req)

        assert resp.status == 200
        data = await _parse(resp)
        assert data["success"] is True
        assert data["pending"] == []
        assert data["count"] == 0

    @pytest.mark.asyncio
    async def test_list_single_pending(self, install_flow, install_cb):
        pending_req = _make_approval_request()
        install_flow.list_pending.return_value = [pending_req]

        req = _make_request()
        resp = await ApprovalHandler.list_pending(req)

        assert resp.status == 200
        data = await _parse(resp)
        assert data["success"] is True
        assert data["count"] == 1
        assert data["pending"][0]["id"] == "req-001"
        assert data["pending"][0]["title"] == "Refactor module X"
        assert data["pending"][0]["risk_level"] == "medium"
        assert data["pending"][0]["requested_by"] == "agent-1"

    @pytest.mark.asyncio
    async def test_list_multiple_pending(self, install_flow, install_cb):
        reqs = [
            _make_approval_request(request_id="req-001", title="First"),
            _make_approval_request(request_id="req-002", title="Second", risk_level="high"),
            _make_approval_request(request_id="req-003", title="Third", risk_level="critical"),
        ]
        install_flow.list_pending.return_value = reqs

        req = _make_request()
        resp = await ApprovalHandler.list_pending(req)

        assert resp.status == 200
        data = await _parse(resp)
        assert data["count"] == 3
        assert data["pending"][0]["id"] == "req-001"
        assert data["pending"][1]["risk_level"] == "high"
        assert data["pending"][2]["risk_level"] == "critical"

    @pytest.mark.asyncio
    async def test_list_response_keys(self, install_flow, install_cb):
        pending_req = _make_approval_request(metadata={"priority": "high", "source": "nomic"})
        install_flow.list_pending.return_value = [pending_req]

        req = _make_request()
        resp = await ApprovalHandler.list_pending(req)

        data = await _parse(resp)
        item = data["pending"][0]
        expected_keys = {
            "id",
            "title",
            "description",
            "changes",
            "risk_level",
            "requested_at",
            "requested_by",
            "timeout_seconds",
            "metadata",
        }
        assert expected_keys == set(item.keys())

    @pytest.mark.asyncio
    async def test_list_requested_at_iso_format(self, install_flow, install_cb):
        dt = datetime(2026, 3, 1, 15, 30, 0, tzinfo=timezone.utc)
        pending_req = _make_approval_request(requested_at=dt)
        install_flow.list_pending.return_value = [pending_req]

        req = _make_request()
        resp = await ApprovalHandler.list_pending(req)

        data = await _parse(resp)
        assert data["pending"][0]["requested_at"] == dt.isoformat()

    @pytest.mark.asyncio
    async def test_list_metadata_preserved(self, install_flow, install_cb):
        pending_req = _make_approval_request(
            metadata={"topic": "security", "agents": 5, "nested": {"key": "val"}}
        )
        install_flow.list_pending.return_value = [pending_req]

        req = _make_request()
        resp = await ApprovalHandler.list_pending(req)

        data = await _parse(resp)
        assert data["pending"][0]["metadata"]["topic"] == "security"
        assert data["pending"][0]["metadata"]["agents"] == 5
        assert data["pending"][0]["metadata"]["nested"]["key"] == "val"

    @pytest.mark.asyncio
    async def test_list_changes_preserved(self, install_flow, install_cb):
        changes = [
            {"file": "a.py", "action": "modify"},
            {"file": "b.py", "action": "create"},
        ]
        pending_req = _make_approval_request(changes=changes)
        install_flow.list_pending.return_value = [pending_req]

        req = _make_request()
        resp = await ApprovalHandler.list_pending(req)

        data = await _parse(resp)
        assert len(data["pending"][0]["changes"]) == 2
        assert data["pending"][0]["changes"][0]["file"] == "a.py"

    @pytest.mark.asyncio
    async def test_list_circuit_breaker_open(self):
        cb = MagicMock()
        cb.can_execute.return_value = False

        with patch(
            "aragora.server.handlers.autonomous.approvals._get_circuit_breaker",
            return_value=cb,
        ):
            req = _make_request()
            resp = await ApprovalHandler.list_pending(req)

        assert resp.status == 503
        data = await _parse(resp)
        assert data["success"] is False
        assert "unavailable" in data["error"].lower()

    @pytest.mark.asyncio
    async def test_list_runtime_error(self, install_flow, install_cb):
        install_flow.list_pending.side_effect = RuntimeError("db down")

        req = _make_request()
        resp = await ApprovalHandler.list_pending(req)

        assert resp.status == 500
        data = await _parse(resp)
        assert data["success"] is False
        assert "Failed to list pending approvals" in data["error"]

    @pytest.mark.asyncio
    async def test_list_key_error(self, install_flow, install_cb):
        install_flow.list_pending.side_effect = KeyError("missing key")

        req = _make_request()
        resp = await ApprovalHandler.list_pending(req)

        assert resp.status == 500

    @pytest.mark.asyncio
    async def test_list_value_error(self, install_flow, install_cb):
        install_flow.list_pending.side_effect = ValueError("bad value")

        req = _make_request()
        resp = await ApprovalHandler.list_pending(req)

        assert resp.status == 500

    @pytest.mark.asyncio
    async def test_list_type_error(self, install_flow, install_cb):
        install_flow.list_pending.side_effect = TypeError("bad type")

        req = _make_request()
        resp = await ApprovalHandler.list_pending(req)

        assert resp.status == 500

    @pytest.mark.asyncio
    async def test_list_attribute_error(self, install_flow, install_cb):
        install_flow.list_pending.side_effect = AttributeError("missing attr")

        req = _make_request()
        resp = await ApprovalHandler.list_pending(req)

        assert resp.status == 500

    @pytest.mark.asyncio
    @pytest.mark.no_auto_auth
    async def test_list_unauthorized(self, install_flow, install_cb):
        from aragora.server.handlers.utils.auth import UnauthorizedError

        with patch(
            "aragora.server.handlers.autonomous.approvals.get_auth_context",
            side_effect=UnauthorizedError("no token"),
        ):
            req = _make_request()
            resp = await ApprovalHandler.list_pending(req)

        assert resp.status == 401
        data = await _parse(resp)
        assert "Authentication required" in data["error"]

    @pytest.mark.asyncio
    @pytest.mark.no_auto_auth
    async def test_list_forbidden(self, install_flow, install_cb):
        mock_ctx = MagicMock()
        mock_ctx.user_id = "non-admin"
        mock_ctx.roles = ["viewer"]
        mock_ctx.permissions = {"some:read"}
        mock_checker = MagicMock()
        mock_decision = MagicMock()
        mock_decision.allowed = False

        with (
            patch(
                "aragora.server.handlers.autonomous.approvals.get_auth_context",
                new_callable=AsyncMock,
                return_value=mock_ctx,
            ),
            patch(
                "aragora.server.handlers.autonomous.approvals.get_permission_checker",
                return_value=mock_checker,
            ),
        ):
            mock_checker.check_permission.return_value = mock_decision
            req = _make_request()
            resp = await ApprovalHandler.list_pending(req)

        assert resp.status == 403
        data = await _parse(resp)
        assert "Permission denied" in data["error"]

    @pytest.mark.asyncio
    async def test_list_timeout_seconds_in_response(self, install_flow, install_cb):
        pending_req = _make_approval_request(timeout_seconds=7200)
        install_flow.list_pending.return_value = [pending_req]

        req = _make_request()
        resp = await ApprovalHandler.list_pending(req)

        data = await _parse(resp)
        assert data["pending"][0]["timeout_seconds"] == 7200


# ---------------------------------------------------------------------------
# get_request endpoint
# ---------------------------------------------------------------------------


class TestGetRequest:
    @pytest.mark.asyncio
    async def test_get_request_found(self, install_flow, install_cb):
        approval_req = _make_approval_request()
        install_flow._load_request.return_value = approval_req

        req = _make_request(match_info={"request_id": "req-001"})
        resp = await ApprovalHandler.get_request(req)

        assert resp.status == 200
        data = await _parse(resp)
        assert data["success"] is True
        assert data["request"]["id"] == "req-001"
        assert data["request"]["title"] == "Refactor module X"
        assert data["request"]["status"] == "pending"

    @pytest.mark.asyncio
    async def test_get_request_not_found(self, install_flow, install_cb):
        install_flow._load_request.return_value = None

        req = _make_request(match_info={"request_id": "nonexistent"})
        resp = await ApprovalHandler.get_request(req)

        assert resp.status == 404
        data = await _parse(resp)
        assert data["success"] is False
        assert "not found" in data["error"].lower()

    @pytest.mark.asyncio
    async def test_get_request_response_keys(self, install_flow, install_cb):
        approval_req = _make_approval_request()
        install_flow._load_request.return_value = approval_req

        req = _make_request(match_info={"request_id": "req-001"})
        resp = await ApprovalHandler.get_request(req)

        data = await _parse(resp)
        request_data = data["request"]
        expected_keys = {
            "id",
            "title",
            "description",
            "changes",
            "risk_level",
            "requested_at",
            "requested_by",
            "timeout_seconds",
            "status",
            "approved_by",
            "approved_at",
            "rejection_reason",
            "metadata",
        }
        assert expected_keys == set(request_data.keys())

    @pytest.mark.asyncio
    async def test_get_request_approved(self, install_flow, install_cb):
        approved_at = datetime(2026, 2, 15, 12, 0, 0, tzinfo=timezone.utc)
        approval_req = _make_approval_request(
            status=_MockApprovalStatus.APPROVED,
            approved_by="admin-user",
            approved_at=approved_at,
        )
        install_flow._load_request.return_value = approval_req

        req = _make_request(match_info={"request_id": "req-001"})
        resp = await ApprovalHandler.get_request(req)

        data = await _parse(resp)
        assert data["request"]["status"] == "approved"
        assert data["request"]["approved_by"] == "admin-user"
        assert data["request"]["approved_at"] == approved_at.isoformat()

    @pytest.mark.asyncio
    async def test_get_request_rejected(self, install_flow, install_cb):
        approval_req = _make_approval_request(
            status=_MockApprovalStatus.REJECTED,
            approved_by="admin-user",
            rejection_reason="Too dangerous",
        )
        install_flow._load_request.return_value = approval_req

        req = _make_request(match_info={"request_id": "req-001"})
        resp = await ApprovalHandler.get_request(req)

        data = await _parse(resp)
        assert data["request"]["status"] == "rejected"
        assert data["request"]["rejection_reason"] == "Too dangerous"

    @pytest.mark.asyncio
    async def test_get_request_pending_no_approved_at(self, install_flow, install_cb):
        approval_req = _make_approval_request(approved_at=None)
        install_flow._load_request.return_value = approval_req

        req = _make_request(match_info={"request_id": "req-001"})
        resp = await ApprovalHandler.get_request(req)

        data = await _parse(resp)
        assert data["request"]["approved_at"] is None

    @pytest.mark.asyncio
    async def test_get_request_circuit_breaker_open(self):
        cb = MagicMock()
        cb.can_execute.return_value = False

        with patch(
            "aragora.server.handlers.autonomous.approvals._get_circuit_breaker",
            return_value=cb,
        ):
            req = _make_request(match_info={"request_id": "req-001"})
            resp = await ApprovalHandler.get_request(req)

        assert resp.status == 503
        data = await _parse(resp)
        assert "unavailable" in data["error"].lower()

    @pytest.mark.asyncio
    async def test_get_request_runtime_error(self, install_flow, install_cb):
        install_flow._load_request.side_effect = RuntimeError("db failure")

        req = _make_request(match_info={"request_id": "req-001"})
        resp = await ApprovalHandler.get_request(req)

        assert resp.status == 500
        data = await _parse(resp)
        assert "Failed to retrieve approval request" in data["error"]

    @pytest.mark.asyncio
    async def test_get_request_key_error(self, install_flow, install_cb):
        install_flow._load_request.side_effect = KeyError("missing")

        req = _make_request(match_info={"request_id": "req-001"})
        resp = await ApprovalHandler.get_request(req)

        assert resp.status == 500

    @pytest.mark.asyncio
    async def test_get_request_type_error(self, install_flow, install_cb):
        install_flow._load_request.side_effect = TypeError("bad type")

        req = _make_request(match_info={"request_id": "req-001"})
        resp = await ApprovalHandler.get_request(req)

        assert resp.status == 500

    @pytest.mark.asyncio
    async def test_get_request_attribute_error(self, install_flow, install_cb):
        install_flow._load_request.side_effect = AttributeError("no attr")

        req = _make_request(match_info={"request_id": "req-001"})
        resp = await ApprovalHandler.get_request(req)

        assert resp.status == 500

    @pytest.mark.asyncio
    @pytest.mark.no_auto_auth
    async def test_get_request_unauthorized(self, install_flow, install_cb):
        from aragora.server.handlers.utils.auth import UnauthorizedError

        with patch(
            "aragora.server.handlers.autonomous.approvals.get_auth_context",
            side_effect=UnauthorizedError("no token"),
        ):
            req = _make_request(match_info={"request_id": "req-001"})
            resp = await ApprovalHandler.get_request(req)

        assert resp.status == 401
        data = await _parse(resp)
        assert "Authentication required" in data["error"]

    @pytest.mark.asyncio
    @pytest.mark.no_auto_auth
    async def test_get_request_forbidden(self, install_flow, install_cb):
        mock_ctx = MagicMock()
        mock_ctx.user_id = "non-admin"
        mock_ctx.roles = ["viewer"]
        mock_ctx.permissions = {"some:read"}
        mock_checker = MagicMock()
        mock_decision = MagicMock()
        mock_decision.allowed = False

        with (
            patch(
                "aragora.server.handlers.autonomous.approvals.get_auth_context",
                new_callable=AsyncMock,
                return_value=mock_ctx,
            ),
            patch(
                "aragora.server.handlers.autonomous.approvals.get_permission_checker",
                return_value=mock_checker,
            ),
        ):
            mock_checker.check_permission.return_value = mock_decision
            req = _make_request(match_info={"request_id": "req-001"})
            resp = await ApprovalHandler.get_request(req)

        assert resp.status == 403
        data = await _parse(resp)
        assert "Permission denied" in data["error"]

    @pytest.mark.asyncio
    async def test_get_request_with_none_id(self, install_flow, install_cb):
        """When match_info returns None for request_id."""
        install_flow._load_request.return_value = None

        req = _make_request(match_info={})
        resp = await ApprovalHandler.get_request(req)

        assert resp.status == 404

    @pytest.mark.asyncio
    async def test_get_request_with_metadata(self, install_flow, install_cb):
        approval_req = _make_approval_request(metadata={"source": "nomic_loop", "cycle": 3})
        install_flow._load_request.return_value = approval_req

        req = _make_request(match_info={"request_id": "req-001"})
        resp = await ApprovalHandler.get_request(req)

        data = await _parse(resp)
        assert data["request"]["metadata"]["source"] == "nomic_loop"
        assert data["request"]["metadata"]["cycle"] == 3


# ---------------------------------------------------------------------------
# approve endpoint
# ---------------------------------------------------------------------------


class TestApprove:
    @pytest.mark.asyncio
    async def test_approve_success(self, install_flow, install_cb):
        approved_at = datetime(2026, 2, 15, 11, 0, 0, tzinfo=timezone.utc)
        approved_req = _make_approval_request(
            status=_MockApprovalStatus.APPROVED,
            approved_by="test-user-001",
            approved_at=approved_at,
        )
        install_flow.approve.return_value = approved_req

        req = _make_request(
            method="POST",
            match_info={"request_id": "req-001"},
            body={"approved_by": "test-user-001"},
        )
        resp = await ApprovalHandler.approve(req)

        assert resp.status == 200
        data = await _parse(resp)
        assert data["success"] is True
        assert data["request"]["id"] == "req-001"
        assert data["request"]["status"] == "approved"
        assert data["request"]["approved_by"] == "test-user-001"
        assert data["request"]["approved_at"] == approved_at.isoformat()

    @pytest.mark.asyncio
    async def test_approve_defaults_to_auth_user(self, install_flow, install_cb):
        """When approved_by is not in body, use auth user."""
        approved_req = _make_approval_request(
            status=_MockApprovalStatus.APPROVED,
            approved_by="test-user-001",
            approved_at=datetime(2026, 2, 15, 11, 0, 0, tzinfo=timezone.utc),
        )
        install_flow.approve.return_value = approved_req

        req = _make_request(
            method="POST",
            match_info={"request_id": "req-001"},
            body={},
        )
        resp = await ApprovalHandler.approve(req)

        assert resp.status == 200
        # The handler calls flow.approve(request_id, approved_by)
        # where approved_by falls back to auth_ctx.user_id
        install_flow.approve.assert_called_once()
        call_args = install_flow.approve.call_args
        # Second positional arg is approved_by
        assert call_args[0][1] == "test-user-001"

    @pytest.mark.asyncio
    async def test_approve_override_approved_by(self, install_flow, install_cb):
        """Explicit approved_by in body overrides auth user."""
        approved_req = _make_approval_request(
            status=_MockApprovalStatus.APPROVED,
            approved_by="custom-user",
        )
        install_flow.approve.return_value = approved_req

        req = _make_request(
            method="POST",
            match_info={"request_id": "req-001"},
            body={"approved_by": "custom-user"},
        )
        resp = await ApprovalHandler.approve(req)

        assert resp.status == 200
        call_args = install_flow.approve.call_args
        assert call_args[0][1] == "custom-user"

    @pytest.mark.asyncio
    async def test_approve_not_found(self, install_flow, install_cb):
        install_flow.approve.side_effect = ValueError("Unknown approval request")

        req = _make_request(
            method="POST",
            match_info={"request_id": "nonexistent"},
            body={},
        )
        resp = await ApprovalHandler.approve(req)

        assert resp.status == 404
        data = await _parse(resp)
        assert "not found" in data["error"].lower()

    @pytest.mark.asyncio
    async def test_approve_already_approved(self, install_flow, install_cb):
        """Approving a non-pending request raises ValueError."""
        install_flow.approve.side_effect = ValueError("not pending")

        req = _make_request(
            method="POST",
            match_info={"request_id": "req-001"},
            body={},
        )
        resp = await ApprovalHandler.approve(req)

        assert resp.status == 404

    @pytest.mark.asyncio
    async def test_approve_circuit_breaker_open(self):
        cb = MagicMock()
        cb.can_execute.return_value = False

        with patch(
            "aragora.server.handlers.autonomous.approvals._get_circuit_breaker",
            return_value=cb,
        ):
            req = _make_request(
                method="POST",
                match_info={"request_id": "req-001"},
                body={},
            )
            resp = await ApprovalHandler.approve(req)

        assert resp.status == 503
        data = await _parse(resp)
        assert "unavailable" in data["error"].lower()

    @pytest.mark.asyncio
    async def test_approve_runtime_error(self, install_flow, install_cb):
        install_flow.approve.side_effect = RuntimeError("boom")

        req = _make_request(
            method="POST",
            match_info={"request_id": "req-001"},
            body={},
        )
        resp = await ApprovalHandler.approve(req)

        assert resp.status == 500
        data = await _parse(resp)
        assert "Failed to approve request" in data["error"]

    @pytest.mark.asyncio
    async def test_approve_key_error(self, install_flow, install_cb):
        install_flow.approve.side_effect = KeyError("missing")

        req = _make_request(
            method="POST",
            match_info={"request_id": "req-001"},
            body={},
        )
        resp = await ApprovalHandler.approve(req)

        assert resp.status == 500

    @pytest.mark.asyncio
    async def test_approve_type_error(self, install_flow, install_cb):
        install_flow.approve.side_effect = TypeError("bad type")

        req = _make_request(
            method="POST",
            match_info={"request_id": "req-001"},
            body={},
        )
        resp = await ApprovalHandler.approve(req)

        assert resp.status == 500

    @pytest.mark.asyncio
    async def test_approve_attribute_error(self, install_flow, install_cb):
        install_flow.approve.side_effect = AttributeError("missing")

        req = _make_request(
            method="POST",
            match_info={"request_id": "req-001"},
            body={},
        )
        resp = await ApprovalHandler.approve(req)

        assert resp.status == 500

    @pytest.mark.asyncio
    @pytest.mark.no_auto_auth
    async def test_approve_unauthorized(self, install_flow, install_cb):
        from aragora.server.handlers.utils.auth import UnauthorizedError

        with patch(
            "aragora.server.handlers.autonomous.approvals.get_auth_context",
            side_effect=UnauthorizedError("no token"),
        ):
            req = _make_request(
                method="POST",
                match_info={"request_id": "req-001"},
                body={},
            )
            resp = await ApprovalHandler.approve(req)

        assert resp.status == 401
        data = await _parse(resp)
        assert "Authentication required" in data["error"]

    @pytest.mark.asyncio
    @pytest.mark.no_auto_auth
    async def test_approve_forbidden(self, install_flow, install_cb):
        mock_ctx = MagicMock()
        mock_ctx.user_id = "non-admin"
        mock_ctx.roles = ["viewer"]
        mock_ctx.permissions = {"some:read"}
        mock_checker = MagicMock()
        mock_decision = MagicMock()
        mock_decision.allowed = False

        with (
            patch(
                "aragora.server.handlers.autonomous.approvals.get_auth_context",
                new_callable=AsyncMock,
                return_value=mock_ctx,
            ),
            patch(
                "aragora.server.handlers.autonomous.approvals.get_permission_checker",
                return_value=mock_checker,
            ),
        ):
            mock_checker.check_permission.return_value = mock_decision
            req = _make_request(
                method="POST",
                match_info={"request_id": "req-001"},
                body={},
            )
            resp = await ApprovalHandler.approve(req)

        assert resp.status == 403
        data = await _parse(resp)
        assert "Permission denied" in data["error"]

    @pytest.mark.asyncio
    async def test_approve_invalid_json(self, install_flow, install_cb):
        """Malformed JSON body returns 400 via parse_json_body."""
        req = MagicMock()
        req.method = "POST"
        req.json = AsyncMock(side_effect=ValueError("bad json"))
        req.read = AsyncMock(return_value=b"not json")
        req.text = AsyncMock(return_value="not json")
        req.content_type = "application/json"
        req.content_length = 8
        req.can_read_body = True
        req.remote = "127.0.0.1"
        req.transport = MagicMock()
        req.transport.get_extra_info.return_value = ("127.0.0.1", 12345)
        mi = MagicMock()
        mi.get = MagicMock(return_value="req-001")
        req.match_info = mi

        resp = await ApprovalHandler.approve(req)
        assert resp.status == 400

    @pytest.mark.asyncio
    async def test_approve_with_none_match_info(self, install_flow, install_cb):
        """When match_info returns None for request_id."""
        install_flow.approve.side_effect = ValueError("not found")

        req = _make_request(
            method="POST",
            match_info={},
            body={},
        )
        resp = await ApprovalHandler.approve(req)

        assert resp.status == 404

    @pytest.mark.asyncio
    async def test_approve_response_has_approved_at_null_when_none(self, install_flow, install_cb):
        """approved_at is None when not set."""
        approved_req = _make_approval_request(
            status=_MockApprovalStatus.APPROVED,
            approved_by="user-1",
            approved_at=None,
        )
        install_flow.approve.return_value = approved_req

        req = _make_request(
            method="POST",
            match_info={"request_id": "req-001"},
            body={},
        )
        resp = await ApprovalHandler.approve(req)

        data = await _parse(resp)
        assert data["request"]["approved_at"] is None


# ---------------------------------------------------------------------------
# reject endpoint
# ---------------------------------------------------------------------------


class TestReject:
    @pytest.mark.asyncio
    async def test_reject_success(self, install_flow, install_cb):
        rejected_req = _make_approval_request(
            status=_MockApprovalStatus.REJECTED,
            approved_by="test-user-001",
            rejection_reason="Too risky",
        )
        install_flow.reject.return_value = rejected_req

        req = _make_request(
            method="POST",
            match_info={"request_id": "req-001"},
            body={"reason": "Too risky"},
        )
        resp = await ApprovalHandler.reject(req)

        assert resp.status == 200
        data = await _parse(resp)
        assert data["success"] is True
        assert data["request"]["id"] == "req-001"
        assert data["request"]["status"] == "rejected"
        assert data["request"]["rejection_reason"] == "Too risky"

    @pytest.mark.asyncio
    async def test_reject_defaults_to_auth_user(self, install_flow, install_cb):
        rejected_req = _make_approval_request(
            status=_MockApprovalStatus.REJECTED,
            approved_by="test-user-001",
            rejection_reason="No reason provided",
        )
        install_flow.reject.return_value = rejected_req

        req = _make_request(
            method="POST",
            match_info={"request_id": "req-001"},
            body={},
        )
        resp = await ApprovalHandler.reject(req)

        assert resp.status == 200
        call_args = install_flow.reject.call_args
        # flow.reject(request_id, rejected_by, reason)
        assert call_args[0][1] == "test-user-001"
        assert call_args[0][2] == "No reason provided"

    @pytest.mark.asyncio
    async def test_reject_with_custom_rejected_by(self, install_flow, install_cb):
        rejected_req = _make_approval_request(
            status=_MockApprovalStatus.REJECTED,
            approved_by="custom-rejecter",
        )
        install_flow.reject.return_value = rejected_req

        req = _make_request(
            method="POST",
            match_info={"request_id": "req-001"},
            body={"rejected_by": "custom-rejecter", "reason": "Policy violation"},
        )
        resp = await ApprovalHandler.reject(req)

        assert resp.status == 200
        call_args = install_flow.reject.call_args
        assert call_args[0][1] == "custom-rejecter"
        assert call_args[0][2] == "Policy violation"

    @pytest.mark.asyncio
    async def test_reject_default_reason(self, install_flow, install_cb):
        """When no reason is given, default to 'No reason provided'."""
        rejected_req = _make_approval_request(
            status=_MockApprovalStatus.REJECTED,
        )
        install_flow.reject.return_value = rejected_req

        req = _make_request(
            method="POST",
            match_info={"request_id": "req-001"},
            body={},
        )
        resp = await ApprovalHandler.reject(req)

        assert resp.status == 200
        call_args = install_flow.reject.call_args
        assert call_args[0][2] == "No reason provided"

    @pytest.mark.asyncio
    async def test_reject_not_found(self, install_flow, install_cb):
        install_flow.reject.side_effect = ValueError("Unknown approval request")

        req = _make_request(
            method="POST",
            match_info={"request_id": "nonexistent"},
            body={"reason": "test"},
        )
        resp = await ApprovalHandler.reject(req)

        assert resp.status == 404
        data = await _parse(resp)
        assert "not found" in data["error"].lower()

    @pytest.mark.asyncio
    async def test_reject_already_rejected(self, install_flow, install_cb):
        install_flow.reject.side_effect = ValueError("not pending")

        req = _make_request(
            method="POST",
            match_info={"request_id": "req-001"},
            body={"reason": "test"},
        )
        resp = await ApprovalHandler.reject(req)

        assert resp.status == 404

    @pytest.mark.asyncio
    async def test_reject_circuit_breaker_open(self):
        cb = MagicMock()
        cb.can_execute.return_value = False

        with patch(
            "aragora.server.handlers.autonomous.approvals._get_circuit_breaker",
            return_value=cb,
        ):
            req = _make_request(
                method="POST",
                match_info={"request_id": "req-001"},
                body={"reason": "test"},
            )
            resp = await ApprovalHandler.reject(req)

        assert resp.status == 503

    @pytest.mark.asyncio
    async def test_reject_runtime_error(self, install_flow, install_cb):
        install_flow.reject.side_effect = RuntimeError("boom")

        req = _make_request(
            method="POST",
            match_info={"request_id": "req-001"},
            body={"reason": "test"},
        )
        resp = await ApprovalHandler.reject(req)

        assert resp.status == 500
        data = await _parse(resp)
        assert "Failed to reject request" in data["error"]

    @pytest.mark.asyncio
    async def test_reject_key_error(self, install_flow, install_cb):
        install_flow.reject.side_effect = KeyError("missing")

        req = _make_request(
            method="POST",
            match_info={"request_id": "req-001"},
            body={"reason": "test"},
        )
        resp = await ApprovalHandler.reject(req)

        assert resp.status == 500

    @pytest.mark.asyncio
    async def test_reject_type_error(self, install_flow, install_cb):
        install_flow.reject.side_effect = TypeError("bad type")

        req = _make_request(
            method="POST",
            match_info={"request_id": "req-001"},
            body={"reason": "test"},
        )
        resp = await ApprovalHandler.reject(req)

        assert resp.status == 500

    @pytest.mark.asyncio
    async def test_reject_attribute_error(self, install_flow, install_cb):
        install_flow.reject.side_effect = AttributeError("no attr")

        req = _make_request(
            method="POST",
            match_info={"request_id": "req-001"},
            body={"reason": "test"},
        )
        resp = await ApprovalHandler.reject(req)

        assert resp.status == 500

    @pytest.mark.asyncio
    @pytest.mark.no_auto_auth
    async def test_reject_unauthorized(self, install_flow, install_cb):
        from aragora.server.handlers.utils.auth import UnauthorizedError

        with patch(
            "aragora.server.handlers.autonomous.approvals.get_auth_context",
            side_effect=UnauthorizedError("no token"),
        ):
            req = _make_request(
                method="POST",
                match_info={"request_id": "req-001"},
                body={"reason": "test"},
            )
            resp = await ApprovalHandler.reject(req)

        assert resp.status == 401
        data = await _parse(resp)
        assert "Authentication required" in data["error"]

    @pytest.mark.asyncio
    @pytest.mark.no_auto_auth
    async def test_reject_forbidden(self, install_flow, install_cb):
        mock_ctx = MagicMock()
        mock_ctx.user_id = "non-admin"
        mock_ctx.roles = ["viewer"]
        mock_ctx.permissions = {"some:read"}
        mock_checker = MagicMock()
        mock_decision = MagicMock()
        mock_decision.allowed = False

        with (
            patch(
                "aragora.server.handlers.autonomous.approvals.get_auth_context",
                new_callable=AsyncMock,
                return_value=mock_ctx,
            ),
            patch(
                "aragora.server.handlers.autonomous.approvals.get_permission_checker",
                return_value=mock_checker,
            ),
        ):
            mock_checker.check_permission.return_value = mock_decision
            req = _make_request(
                method="POST",
                match_info={"request_id": "req-001"},
                body={"reason": "test"},
            )
            resp = await ApprovalHandler.reject(req)

        assert resp.status == 403
        data = await _parse(resp)
        assert "Permission denied" in data["error"]

    @pytest.mark.asyncio
    async def test_reject_invalid_json(self, install_flow, install_cb):
        """Malformed JSON body returns 400."""
        req = MagicMock()
        req.method = "POST"
        req.json = AsyncMock(side_effect=ValueError("bad json"))
        req.read = AsyncMock(return_value=b"not json")
        req.text = AsyncMock(return_value="not json")
        req.content_type = "application/json"
        req.content_length = 8
        req.can_read_body = True
        req.remote = "127.0.0.1"
        req.transport = MagicMock()
        req.transport.get_extra_info.return_value = ("127.0.0.1", 12345)
        mi = MagicMock()
        mi.get = MagicMock(return_value="req-001")
        req.match_info = mi

        resp = await ApprovalHandler.reject(req)
        assert resp.status == 400

    @pytest.mark.asyncio
    async def test_reject_with_none_match_info(self, install_flow, install_cb):
        install_flow.reject.side_effect = ValueError("not found")

        req = _make_request(
            method="POST",
            match_info={},
            body={"reason": "test"},
        )
        resp = await ApprovalHandler.reject(req)

        assert resp.status == 404

    @pytest.mark.asyncio
    async def test_reject_response_keys(self, install_flow, install_cb):
        rejected_req = _make_approval_request(
            status=_MockApprovalStatus.REJECTED,
            approved_by="user-1",
            rejection_reason="Bad idea",
        )
        install_flow.reject.return_value = rejected_req

        req = _make_request(
            method="POST",
            match_info={"request_id": "req-001"},
            body={"reason": "Bad idea"},
        )
        resp = await ApprovalHandler.reject(req)

        data = await _parse(resp)
        expected_keys = {"id", "status", "approved_by", "rejection_reason"}
        assert expected_keys == set(data["request"].keys())


# ---------------------------------------------------------------------------
# register_routes
# ---------------------------------------------------------------------------


class TestRegisterRoutes:
    def test_register_routes_default_prefix(self):
        app = web.Application()
        ApprovalHandler.register_routes(app)

        route_paths = [
            r.resource.canonical
            for r in app.router.routes()
            if hasattr(r, "resource") and r.resource
        ]

        assert any("/api/v1/autonomous/approvals/pending" == p for p in route_paths)
        assert any("request_id" in p and "/approve" in p for p in route_paths)
        assert any("request_id" in p and "/reject" in p for p in route_paths)
        # The GET for specific request by ID
        assert any(
            "request_id" in p and "/approve" not in p and "/reject" not in p for p in route_paths
        )

    def test_register_routes_custom_prefix(self):
        app = web.Application()
        ApprovalHandler.register_routes(app, prefix="/custom/api")

        route_paths = [
            r.resource.canonical
            for r in app.router.routes()
            if hasattr(r, "resource") and r.resource
        ]

        assert any("/custom/api/approvals/pending" in p for p in route_paths)
        assert any("/custom/api/approvals/" in p and "/approve" in p for p in route_paths)
        assert any("/custom/api/approvals/" in p and "/reject" in p for p in route_paths)

    def test_register_routes_count(self):
        app = web.Application()
        ApprovalHandler.register_routes(app)

        route_count = sum(1 for r in app.router.routes() if hasattr(r, "resource") and r.resource)
        # 4 explicit routes: GET pending, GET {request_id}, POST approve, POST reject
        # aiohttp auto-adds HEAD for each GET = 2 more
        assert route_count == 6

    def test_register_routes_methods(self):
        app = web.Application()
        ApprovalHandler.register_routes(app)

        method_map = {}
        for r in app.router.routes():
            if hasattr(r, "resource") and r.resource:
                canonical = r.resource.canonical
                if canonical not in method_map:
                    method_map[canonical] = []
                method_map[canonical].append(r.method)

        pending_path = "/api/v1/autonomous/approvals/pending"
        assert "GET" in method_map.get(pending_path, [])


# ---------------------------------------------------------------------------
# Handler init
# ---------------------------------------------------------------------------


class TestHandlerInit:
    def test_default_ctx(self):
        handler = ApprovalHandler()
        assert handler.ctx == {}

    def test_custom_ctx(self):
        handler = ApprovalHandler(ctx={"env": "production"})
        assert handler.ctx == {"env": "production"}

    def test_none_ctx_becomes_empty(self):
        handler = ApprovalHandler(ctx=None)
        assert handler.ctx == {}


# ---------------------------------------------------------------------------
# Global accessors
# ---------------------------------------------------------------------------


class TestGlobalAccessors:
    def test_get_approval_flow_creates_instance(self):
        flow = get_approval_flow()
        assert flow is not None

    def test_get_approval_flow_singleton(self):
        f1 = get_approval_flow()
        f2 = get_approval_flow()
        assert f1 is f2

    def test_set_and_get_approval_flow(self):
        custom = MagicMock()
        set_approval_flow(custom)
        assert get_approval_flow() is custom

    def test_set_approval_flow_replaces(self):
        first = MagicMock()
        second = MagicMock()
        set_approval_flow(first)
        set_approval_flow(second)
        assert get_approval_flow() is second

    def test_get_circuit_breaker_creates_instance(self):
        cb = _get_circuit_breaker()
        assert cb is not None

    def test_get_circuit_breaker_singleton(self):
        cb1 = _get_circuit_breaker()
        cb2 = _get_circuit_breaker()
        assert cb1 is cb2

    def test_permission_constants(self):
        assert AUTONOMOUS_READ_PERMISSION == "autonomous:read"
        assert AUTONOMOUS_APPROVE_PERMISSION == "autonomous:approve"


# ---------------------------------------------------------------------------
# _ensure_auth_context
# ---------------------------------------------------------------------------


class TestEnsureAuthContext:
    def test_sets_missing_roles(self):
        ctx = MagicMock(spec=[])
        _ensure_auth_context(ctx)
        assert ctx.roles == []

    def test_sets_none_roles(self):
        ctx = MagicMock(spec=[])
        ctx.roles = None
        _ensure_auth_context(ctx)
        assert ctx.roles == []

    def test_sets_missing_org_id(self):
        ctx = MagicMock(spec=[])
        _ensure_auth_context(ctx)
        assert ctx.org_id is None

    def test_sets_missing_api_key_scope(self):
        ctx = MagicMock(spec=[])
        _ensure_auth_context(ctx)
        assert ctx.api_key_scope is None

    def test_sets_missing_workspace_id(self):
        ctx = MagicMock(spec=[])
        _ensure_auth_context(ctx)
        assert ctx.workspace_id is None

    def test_sets_empty_permissions(self):
        ctx = MagicMock(spec=[])
        _ensure_auth_context(ctx)
        assert ctx.permissions == set()

    def test_permissions_escalation_from_approval_grant(self):
        ctx = MagicMock()
        ctx.roles = ["user"]
        ctx.permissions = {"approval.grant"}
        _ensure_auth_context(ctx)
        assert AUTONOMOUS_READ_PERMISSION in ctx.permissions
        assert AUTONOMOUS_APPROVE_PERMISSION in ctx.permissions

    def test_permissions_escalation_from_approvals_manage(self):
        ctx = MagicMock()
        ctx.roles = ["user"]
        ctx.permissions = {"approvals.manage"}
        _ensure_auth_context(ctx)
        assert AUTONOMOUS_READ_PERMISSION in ctx.permissions
        assert AUTONOMOUS_APPROVE_PERMISSION in ctx.permissions

    def test_permissions_escalation_from_approvals_colon_manage(self):
        ctx = MagicMock()
        ctx.roles = ["user"]
        ctx.permissions = {"approvals:manage"}
        _ensure_auth_context(ctx)
        assert AUTONOMOUS_READ_PERMISSION in ctx.permissions
        assert AUTONOMOUS_APPROVE_PERMISSION in ctx.permissions

    def test_no_escalation_without_matching_permission(self):
        ctx = MagicMock()
        ctx.roles = ["user"]
        ctx.permissions = {"other:read"}
        _ensure_auth_context(ctx)
        assert AUTONOMOUS_READ_PERMISSION not in ctx.permissions
        assert AUTONOMOUS_APPROVE_PERMISSION not in ctx.permissions

    def test_permissions_converted_from_list_to_set(self):
        ctx = MagicMock()
        ctx.roles = ["user"]
        ctx.permissions = ["approvals:manage", "other:read"]
        _ensure_auth_context(ctx)
        assert isinstance(ctx.permissions, set)
        assert AUTONOMOUS_READ_PERMISSION in ctx.permissions

    def test_permissions_none_becomes_set(self):
        """When permissions attr exists but is None."""
        ctx = MagicMock()
        ctx.roles = ["user"]
        ctx.permissions = None
        _ensure_auth_context(ctx)
        # None is falsy, so set() union doesn't trigger, permissions = set()
        assert ctx.permissions == set()


# ---------------------------------------------------------------------------
# _is_admin
# ---------------------------------------------------------------------------


class TestIsAdmin:
    def test_admin_role(self):
        ctx = MagicMock()
        ctx.roles = ["admin"]
        assert _is_admin(ctx) is True

    def test_owner_role(self):
        ctx = MagicMock()
        ctx.roles = ["owner"]
        assert _is_admin(ctx) is True

    def test_admin_and_owner_roles(self):
        ctx = MagicMock()
        ctx.roles = ["admin", "owner"]
        assert _is_admin(ctx) is True

    def test_non_admin_role(self):
        ctx = MagicMock()
        ctx.roles = ["viewer", "editor"]
        assert _is_admin(ctx) is False

    def test_empty_roles(self):
        ctx = MagicMock()
        ctx.roles = []
        assert _is_admin(ctx) is False

    def test_none_roles(self):
        ctx = MagicMock()
        ctx.roles = None
        assert _is_admin(ctx) is False

    def test_missing_roles_attribute(self):
        ctx = MagicMock(spec=[])
        assert _is_admin(ctx) is False


# ---------------------------------------------------------------------------
# RBAC with non-admin user and valid permissions
# ---------------------------------------------------------------------------


class TestRBACNonAdmin:
    @pytest.mark.asyncio
    @pytest.mark.no_auto_auth
    async def test_list_pending_with_read_permission(self, install_flow, install_cb):
        """Non-admin user with autonomous:read permission can list pending."""
        mock_ctx = MagicMock()
        mock_ctx.user_id = "regular-user"
        mock_ctx.roles = ["viewer"]
        mock_ctx.permissions = {"autonomous:read"}

        mock_checker = MagicMock()
        mock_decision = MagicMock()
        mock_decision.allowed = True

        install_flow.list_pending.return_value = []

        with (
            patch(
                "aragora.server.handlers.autonomous.approvals.get_auth_context",
                new_callable=AsyncMock,
                return_value=mock_ctx,
            ),
            patch(
                "aragora.server.handlers.autonomous.approvals.get_permission_checker",
                return_value=mock_checker,
            ),
        ):
            mock_checker.check_permission.return_value = mock_decision
            req = _make_request()
            resp = await ApprovalHandler.list_pending(req)

        assert resp.status == 200
        mock_checker.check_permission.assert_called_once_with(mock_ctx, AUTONOMOUS_READ_PERMISSION)

    @pytest.mark.asyncio
    @pytest.mark.no_auto_auth
    async def test_approve_with_approve_permission(self, install_flow, install_cb):
        """Non-admin user with autonomous:approve permission can approve."""
        mock_ctx = MagicMock()
        mock_ctx.user_id = "regular-user"
        mock_ctx.roles = ["approver"]
        mock_ctx.permissions = {"autonomous:approve"}

        mock_checker = MagicMock()
        mock_decision = MagicMock()
        mock_decision.allowed = True

        approved_req = _make_approval_request(
            status=_MockApprovalStatus.APPROVED,
            approved_by="regular-user",
        )
        install_flow.approve.return_value = approved_req

        with (
            patch(
                "aragora.server.handlers.autonomous.approvals.get_auth_context",
                new_callable=AsyncMock,
                return_value=mock_ctx,
            ),
            patch(
                "aragora.server.handlers.autonomous.approvals.get_permission_checker",
                return_value=mock_checker,
            ),
        ):
            mock_checker.check_permission.return_value = mock_decision
            req = _make_request(
                method="POST",
                match_info={"request_id": "req-001"},
                body={},
            )
            resp = await ApprovalHandler.approve(req)

        assert resp.status == 200
        mock_checker.check_permission.assert_called_once_with(
            mock_ctx, AUTONOMOUS_APPROVE_PERMISSION
        )

    @pytest.mark.asyncio
    @pytest.mark.no_auto_auth
    async def test_admin_bypasses_rbac(self, install_flow, install_cb):
        """Admin user bypasses RBAC check even with permissions set."""
        mock_ctx = MagicMock()
        mock_ctx.user_id = "admin-user"
        mock_ctx.roles = ["admin"]
        mock_ctx.permissions = {"some:permission"}

        install_flow.list_pending.return_value = []

        with patch(
            "aragora.server.handlers.autonomous.approvals.get_auth_context",
            new_callable=AsyncMock,
            return_value=mock_ctx,
        ):
            req = _make_request()
            resp = await ApprovalHandler.list_pending(req)

        assert resp.status == 200

    @pytest.mark.asyncio
    @pytest.mark.no_auto_auth
    async def test_empty_permissions_allows_legacy_access(self, install_flow, install_cb):
        """User with empty permissions (legacy context) bypasses RBAC check."""
        mock_ctx = MagicMock()
        mock_ctx.user_id = "legacy-user"
        mock_ctx.roles = ["viewer"]
        mock_ctx.permissions = set()

        install_flow.list_pending.return_value = []

        with patch(
            "aragora.server.handlers.autonomous.approvals.get_auth_context",
            new_callable=AsyncMock,
            return_value=mock_ctx,
        ):
            req = _make_request()
            resp = await ApprovalHandler.list_pending(req)

        assert resp.status == 200


# ---------------------------------------------------------------------------
# Security edge cases
# ---------------------------------------------------------------------------


class TestSecurityEdgeCases:
    @pytest.mark.asyncio
    async def test_path_traversal_request_id(self, install_flow, install_cb):
        """Path traversal in request_id should not cause issues."""
        install_flow._load_request.return_value = None

        req = _make_request(match_info={"request_id": "../../etc/passwd"})
        resp = await ApprovalHandler.get_request(req)

        assert resp.status == 404

    @pytest.mark.asyncio
    async def test_xss_in_request_id(self, install_flow, install_cb):
        """XSS attempt in request_id should not be reflected."""
        install_flow._load_request.return_value = None

        req = _make_request(match_info={"request_id": "<script>alert(1)</script>"})
        resp = await ApprovalHandler.get_request(req)

        assert resp.status == 404
        data = await _parse(resp)
        # The error message should not contain the injected script
        assert "<script>" not in data["error"]

    @pytest.mark.asyncio
    async def test_sql_injection_request_id(self, install_flow, install_cb):
        """SQL injection in request_id should not cause issues."""
        install_flow._load_request.return_value = None

        req = _make_request(match_info={"request_id": "'; DROP TABLE approvals; --"})
        resp = await ApprovalHandler.get_request(req)

        assert resp.status == 404

    @pytest.mark.asyncio
    async def test_very_long_request_id(self, install_flow, install_cb):
        """Very long request_id should not cause issues."""
        install_flow._load_request.return_value = None

        long_id = "a" * 10000
        req = _make_request(match_info={"request_id": long_id})
        resp = await ApprovalHandler.get_request(req)

        assert resp.status == 404

    @pytest.mark.asyncio
    async def test_approve_with_xss_in_approved_by(self, install_flow, install_cb):
        """XSS in approved_by field should be handled safely."""
        approved_req = _make_approval_request(
            status=_MockApprovalStatus.APPROVED,
            approved_by="<script>alert(1)</script>",
        )
        install_flow.approve.return_value = approved_req

        req = _make_request(
            method="POST",
            match_info={"request_id": "req-001"},
            body={"approved_by": "<script>alert(1)</script>"},
        )
        resp = await ApprovalHandler.approve(req)

        # The handler should process it, the XSS value is just stored as data
        assert resp.status == 200

    @pytest.mark.asyncio
    async def test_reject_with_long_reason(self, install_flow, install_cb):
        """Very long rejection reason should not cause issues."""
        long_reason = "x" * 10000
        rejected_req = _make_approval_request(
            status=_MockApprovalStatus.REJECTED,
            rejection_reason=long_reason,
        )
        install_flow.reject.return_value = rejected_req

        req = _make_request(
            method="POST",
            match_info={"request_id": "req-001"},
            body={"reason": long_reason},
        )
        resp = await ApprovalHandler.reject(req)

        assert resp.status == 200

    @pytest.mark.asyncio
    async def test_unicode_in_request_fields(self, install_flow, install_cb):
        """Unicode characters in fields should be handled properly."""
        approval_req = _make_approval_request(
            title="Refactoriser le module \u00e9v\u00e9nements",
            description="\u65e5\u672c\u8a9e\u306e\u8aac\u660e",
        )
        install_flow._load_request.return_value = approval_req

        req = _make_request(match_info={"request_id": "req-001"})
        resp = await ApprovalHandler.get_request(req)

        assert resp.status == 200
        data = await _parse(resp)
        assert "\u00e9" in data["request"]["title"]

    @pytest.mark.asyncio
    async def test_empty_string_request_id(self, install_flow, install_cb):
        """Empty string request_id should be handled gracefully."""
        install_flow._load_request.return_value = None

        req = _make_request(match_info={"request_id": ""})
        resp = await ApprovalHandler.get_request(req)

        assert resp.status == 404

    @pytest.mark.asyncio
    async def test_null_bytes_in_request_id(self, install_flow, install_cb):
        """Null bytes in request_id should be handled gracefully."""
        install_flow._load_request.return_value = None

        req = _make_request(match_info={"request_id": "req\x00001"})
        resp = await ApprovalHandler.get_request(req)

        assert resp.status == 404


# ---------------------------------------------------------------------------
# Integration: flows through global accessor
# ---------------------------------------------------------------------------


class TestIntegration:
    @pytest.mark.asyncio
    async def test_list_pending_uses_get_approval_flow(self, install_cb):
        """Verify list_pending goes through get_approval_flow()."""
        mock_flow = MagicMock()
        mock_flow.list_pending.return_value = []
        set_approval_flow(mock_flow)

        req = _make_request()
        resp = await ApprovalHandler.list_pending(req)

        assert resp.status == 200
        mock_flow.list_pending.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_request_uses_get_approval_flow(self, install_cb):
        """Verify get_request goes through get_approval_flow()."""
        mock_flow = MagicMock()
        mock_flow._load_request.return_value = None
        set_approval_flow(mock_flow)

        req = _make_request(match_info={"request_id": "req-001"})
        resp = await ApprovalHandler.get_request(req)

        assert resp.status == 404
        mock_flow._load_request.assert_called_once_with("req-001")

    @pytest.mark.asyncio
    async def test_approve_uses_get_approval_flow(self, install_cb):
        """Verify approve goes through get_approval_flow()."""
        approved_req = _make_approval_request(
            status=_MockApprovalStatus.APPROVED,
        )
        mock_flow = MagicMock()
        mock_flow.approve.return_value = approved_req
        set_approval_flow(mock_flow)

        req = _make_request(
            method="POST",
            match_info={"request_id": "req-001"},
            body={},
        )
        resp = await ApprovalHandler.approve(req)

        assert resp.status == 200
        mock_flow.approve.assert_called_once()

    @pytest.mark.asyncio
    async def test_reject_uses_get_approval_flow(self, install_cb):
        """Verify reject goes through get_approval_flow()."""
        rejected_req = _make_approval_request(
            status=_MockApprovalStatus.REJECTED,
        )
        mock_flow = MagicMock()
        mock_flow.reject.return_value = rejected_req
        set_approval_flow(mock_flow)

        req = _make_request(
            method="POST",
            match_info={"request_id": "req-001"},
            body={"reason": "test"},
        )
        resp = await ApprovalHandler.reject(req)

        assert resp.status == 200
        mock_flow.reject.assert_called_once()
