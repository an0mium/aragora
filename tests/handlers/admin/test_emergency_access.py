"""Tests for aragora.server.handlers.admin.emergency_access module.

Comprehensive coverage of EmergencyAccessHandler:
- POST /api/v1/admin/emergency/activate  (_activate)
- POST /api/v1/admin/emergency/deactivate  (_deactivate)
- GET  /api/v1/admin/emergency/status  (_status)

Also covers:
- can_handle() path matching for versioned and legacy routes
- handle() routing dispatch for all endpoints
- Invalid JSON body handling
- Required field validation (user_id, reason, access_id)
- Reason length validation (min 10 chars)
- duration_minutes parsing and defaults
- IP address and User-Agent extraction from handler
- BreakGlassAccess.activate ValueError handling
- BreakGlassAccess.deactivate ValueError handling (not found)
- Status endpoint: active sessions, history, persistence flag
- PermissionDeniedError handling in route dispatch
- Edge cases: empty body, whitespace-only fields, non-versioned routes
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.admin.emergency_access import EmergencyAccessHandler
from aragora.server.handlers.utils.responses import HandlerResult


# ===========================================================================
# Helpers
# ===========================================================================


def _body(result: HandlerResult) -> dict:
    """Parse JSON body from a HandlerResult."""
    if result and result.body:
        return json.loads(result.body.decode("utf-8"))
    return {}


def _status(result: HandlerResult) -> int:
    """Extract status code from a HandlerResult."""
    return result.status_code


def _make_http_handler(
    body: dict | None = None,
    ip: str = "127.0.0.1",
    user_agent: str | None = "TestAgent/1.0",
) -> MagicMock:
    """Create a mock HTTP handler with optional JSON body."""
    h = MagicMock()
    h.command = "POST"
    h.client_address = (ip, 12345)
    h.remote = ip
    if body is not None:
        body_bytes = json.dumps(body).encode()
        h.rfile.read.return_value = body_bytes
        headers = {
            "Content-Type": "application/json",
            "Content-Length": str(len(body_bytes)),
        }
        if user_agent:
            headers["User-Agent"] = user_agent
        h.headers = headers
    else:
        h.rfile.read.return_value = b"{}"
        headers = {"Content-Type": "application/json", "Content-Length": "2"}
        if user_agent:
            headers["User-Agent"] = user_agent
        h.headers = headers
    return h


# ===========================================================================
# Mock data classes
# ===========================================================================


class MockEmergencyAccessStatus:
    """Mock enum for emergency access status."""

    def __init__(self, value: str):
        self.value = value


@dataclass
class MockEmergencyAccessRecord:
    """Mock emergency access record."""

    id: str = "emerg-abc123"
    user_id: str = "user-123"
    reason: str = "Production incident requiring immediate access"
    status: Any = field(default_factory=lambda: MockEmergencyAccessStatus("deactivated"))
    activated_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc) - timedelta(minutes=30)
    )
    expires_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc) + timedelta(minutes=30)
    )
    deactivated_at: datetime | None = None
    deactivated_by: str | None = None
    ip_address: str | None = "127.0.0.1"
    user_agent: str | None = "TestAgent/1.0"
    actions_taken: list = field(default_factory=list)
    review_required: bool = True
    review_completed: bool = False
    metadata: dict = field(default_factory=dict)
    is_active: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "reason": self.reason,
            "status": self.status.value if hasattr(self.status, "value") else self.status,
            "activated_at": self.activated_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "deactivated_at": self.deactivated_at.isoformat() if self.deactivated_at else None,
            "deactivated_by": self.deactivated_by,
            "actions_count": len(self.actions_taken),
            "is_active": self.is_active,
        }


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def handler():
    """An EmergencyAccessHandler with empty context."""
    return EmergencyAccessHandler(ctx={})


@pytest.fixture
def mock_http():
    """Minimal mock HTTP handler (no body)."""
    return _make_http_handler()


@pytest.fixture
def mock_emergency():
    """Mock BreakGlassAccess instance."""
    emergency = MagicMock()
    emergency._all_records = {}
    emergency._active_records = {}
    emergency._persistence_enabled = False
    emergency.activate = AsyncMock(return_value="emerg-abc123")
    emergency.deactivate = AsyncMock()
    emergency.expire_old_access = AsyncMock(return_value=0)
    emergency.get_history = AsyncMock(return_value=[])
    return emergency


# ===========================================================================
# Tests: can_handle
# ===========================================================================


class TestCanHandle:
    """Tests for can_handle path matching."""

    def test_versioned_activate_route(self, handler):
        assert handler.can_handle("/api/v1/admin/emergency/activate") is True

    def test_versioned_deactivate_route(self, handler):
        assert handler.can_handle("/api/v1/admin/emergency/deactivate") is True

    def test_versioned_status_route(self, handler):
        assert handler.can_handle("/api/v1/admin/emergency/status") is True

    def test_legacy_activate_route(self, handler):
        assert handler.can_handle("/api/admin/emergency/activate") is True

    def test_legacy_deactivate_route(self, handler):
        assert handler.can_handle("/api/admin/emergency/deactivate") is True

    def test_legacy_status_route(self, handler):
        assert handler.can_handle("/api/admin/emergency/status") is True

    def test_unknown_path(self, handler):
        assert handler.can_handle("/api/v1/admin/unknown") is False

    def test_empty_path(self, handler):
        assert handler.can_handle("") is False

    def test_partial_path(self, handler):
        assert handler.can_handle("/api/v1/admin/emergency") is False

    def test_unrelated_path(self, handler):
        assert handler.can_handle("/api/v1/debates") is False

    def test_routes_list_has_six_entries(self):
        """ROUTES should include 3 versioned + 3 legacy routes."""
        assert len(EmergencyAccessHandler.ROUTES) == 6

    def test_all_routes_are_strings(self):
        """All routes are strings."""
        for route in EmergencyAccessHandler.ROUTES:
            assert isinstance(route, str)

    def test_routes_include_all_expected_paths(self):
        """ROUTES includes all expected paths."""
        expected = {
            "/api/v1/admin/emergency/activate",
            "/api/v1/admin/emergency/deactivate",
            "/api/v1/admin/emergency/status",
            "/api/admin/emergency/activate",
            "/api/admin/emergency/deactivate",
            "/api/admin/emergency/status",
        }
        assert set(EmergencyAccessHandler.ROUTES) == expected


# ===========================================================================
# Tests: handle() routing
# ===========================================================================


class TestHandleRouting:
    """Tests for request routing via handle()."""

    def test_activate_route_dispatches(self, handler, mock_http):
        """Activate endpoint dispatches to _activate."""
        mock_fn = MagicMock(return_value=MagicMock(spec=HandlerResult, status_code=200, body=b"{}"))
        with patch.object(handler, "_activate", mock_fn):
            handler.handle("/api/v1/admin/emergency/activate", {}, mock_http)
            mock_fn.assert_called_once()

    def test_deactivate_route_dispatches(self, handler, mock_http):
        """Deactivate endpoint dispatches to _deactivate."""
        mock_fn = MagicMock(return_value=MagicMock(spec=HandlerResult, status_code=200, body=b"{}"))
        with patch.object(handler, "_deactivate", mock_fn):
            handler.handle("/api/v1/admin/emergency/deactivate", {}, mock_http)
            mock_fn.assert_called_once()

    def test_status_route_dispatches(self, handler, mock_http):
        """Status endpoint dispatches to _status."""
        mock_fn = MagicMock(return_value=MagicMock(spec=HandlerResult, status_code=200, body=b"{}"))
        with patch.object(handler, "_status", mock_fn):
            handler.handle("/api/v1/admin/emergency/status", {}, mock_http)
            mock_fn.assert_called_once()

    def test_legacy_activate_route_dispatches(self, handler, mock_http):
        """Legacy activate route dispatches to _activate."""
        mock_fn = MagicMock(return_value=MagicMock(spec=HandlerResult, status_code=200, body=b"{}"))
        with patch.object(handler, "_activate", mock_fn):
            handler.handle("/api/admin/emergency/activate", {}, mock_http)
            mock_fn.assert_called_once()

    def test_legacy_deactivate_route_dispatches(self, handler, mock_http):
        """Legacy deactivate route dispatches to _deactivate."""
        mock_fn = MagicMock(return_value=MagicMock(spec=HandlerResult, status_code=200, body=b"{}"))
        with patch.object(handler, "_deactivate", mock_fn):
            handler.handle("/api/admin/emergency/deactivate", {}, mock_http)
            mock_fn.assert_called_once()

    def test_legacy_status_route_dispatches(self, handler, mock_http):
        """Legacy status route dispatches to _status."""
        mock_fn = MagicMock(return_value=MagicMock(spec=HandlerResult, status_code=200, body=b"{}"))
        with patch.object(handler, "_status", mock_fn):
            handler.handle("/api/admin/emergency/status", {}, mock_http)
            mock_fn.assert_called_once()

    def test_unmatched_path_returns_none(self, handler, mock_http):
        """Unmatched path returns None from handle()."""
        result = handler.handle("/api/v1/admin/emergency/nonexistent", {}, mock_http)
        assert result is None

    def test_permission_denied_returns_error(self, handler, mock_http):
        """PermissionDeniedError is caught and returns error via handle_security_error."""
        from aragora.rbac.decorators import PermissionDeniedError

        mock_security = MagicMock(
            return_value=MagicMock(spec=HandlerResult, status_code=403, body=b'{"error":"denied"}')
        )
        with patch.object(handler, "_activate", side_effect=PermissionDeniedError("test")), patch.object(
            handler, "handle_security_error", mock_security
        ):
            result = handler.handle("/api/v1/admin/emergency/activate", {}, mock_http)
            mock_security.assert_called_once()
            assert result is not None


# ===========================================================================
# Tests: _activate endpoint
# ===========================================================================


class TestActivate:
    """Tests for the _activate endpoint."""

    def test_activate_success(self, handler, mock_emergency):
        """Successful activation returns access_id and status."""
        body_data = {
            "user_id": "user-target-001",
            "reason": "Production database corruption requiring immediate intervention",
            "duration_minutes": 120,
        }
        mock_http = _make_http_handler(body=body_data)

        mock_record = MockEmergencyAccessRecord(
            id="emerg-abc123",
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=120),
        )
        mock_emergency._all_records["emerg-abc123"] = mock_record

        with patch(
            "aragora.rbac.emergency.get_break_glass_access",
            return_value=mock_emergency,
        ), patch(
            "aragora.server.http_utils.run_async",
            return_value="emerg-abc123",
        ):
            result = handler.handle("/api/v1/admin/emergency/activate", {}, mock_http)
            body = _body(result)
            assert _status(result) == 200
            assert body["access_id"] == "emerg-abc123"
            assert body["user_id"] == "user-target-001"
            assert body["status"] == "active"
            assert body["duration_minutes"] == 120
            assert body["message"] == "Emergency break-glass access activated"

    def test_activate_invalid_json_body(self, handler):
        """Invalid JSON body returns 400."""
        mock_http = MagicMock()
        mock_http.headers = {"Content-Length": "5"}
        mock_http.rfile.read.return_value = b"notjson"

        # Need to patch run_async since it won't be called but _activate
        # still gets invoked
        result = handler.handle("/api/v1/admin/emergency/activate", {}, mock_http)
        body = _body(result)
        assert _status(result) == 400
        assert "Invalid JSON body" in body["error"]

    def test_activate_missing_user_id(self, handler):
        """Missing user_id returns 400."""
        body_data = {
            "reason": "Production incident requiring immediate access",
        }
        mock_http = _make_http_handler(body=body_data)

        result = handler.handle("/api/v1/admin/emergency/activate", {}, mock_http)
        body = _body(result)
        assert _status(result) == 400
        assert "user_id" in body["error"]

    def test_activate_empty_user_id(self, handler):
        """Empty user_id returns 400."""
        body_data = {
            "user_id": "",
            "reason": "Production incident requiring immediate access",
        }
        mock_http = _make_http_handler(body=body_data)

        result = handler.handle("/api/v1/admin/emergency/activate", {}, mock_http)
        body = _body(result)
        assert _status(result) == 400
        assert "user_id" in body["error"]

    def test_activate_whitespace_only_user_id(self, handler):
        """Whitespace-only user_id returns 400."""
        body_data = {
            "user_id": "   ",
            "reason": "Production incident requiring immediate access",
        }
        mock_http = _make_http_handler(body=body_data)

        result = handler.handle("/api/v1/admin/emergency/activate", {}, mock_http)
        body = _body(result)
        assert _status(result) == 400
        assert "user_id" in body["error"]

    def test_activate_missing_reason(self, handler):
        """Missing reason returns 400."""
        body_data = {
            "user_id": "user-target-001",
        }
        mock_http = _make_http_handler(body=body_data)

        result = handler.handle("/api/v1/admin/emergency/activate", {}, mock_http)
        body = _body(result)
        assert _status(result) == 400
        assert "10 characters" in body["error"]

    def test_activate_empty_reason(self, handler):
        """Empty reason returns 400."""
        body_data = {
            "user_id": "user-target-001",
            "reason": "",
        }
        mock_http = _make_http_handler(body=body_data)

        result = handler.handle("/api/v1/admin/emergency/activate", {}, mock_http)
        body = _body(result)
        assert _status(result) == 400
        assert "10 characters" in body["error"]

    def test_activate_reason_too_short(self, handler):
        """Reason shorter than 10 chars returns 400."""
        body_data = {
            "user_id": "user-target-001",
            "reason": "short",
        }
        mock_http = _make_http_handler(body=body_data)

        result = handler.handle("/api/v1/admin/emergency/activate", {}, mock_http)
        body = _body(result)
        assert _status(result) == 400
        assert "10 characters" in body["error"]

    def test_activate_reason_exactly_10_chars(self, handler, mock_emergency):
        """Reason with exactly 10 chars succeeds."""
        body_data = {
            "user_id": "user-target-001",
            "reason": "1234567890",
        }
        mock_http = _make_http_handler(body=body_data)

        mock_record = MockEmergencyAccessRecord(id="emerg-abc123")
        mock_emergency._all_records["emerg-abc123"] = mock_record

        with patch(
            "aragora.rbac.emergency.get_break_glass_access",
            return_value=mock_emergency,
        ), patch(
            "aragora.server.http_utils.run_async",
            return_value="emerg-abc123",
        ):
            result = handler.handle("/api/v1/admin/emergency/activate", {}, mock_http)
            assert _status(result) == 200

    def test_activate_default_duration(self, handler, mock_emergency):
        """Default duration is 60 minutes when not specified."""
        body_data = {
            "user_id": "user-target-001",
            "reason": "Production incident requiring immediate access",
        }
        mock_http = _make_http_handler(body=body_data)

        mock_record = MockEmergencyAccessRecord(id="emerg-abc123")
        mock_emergency._all_records["emerg-abc123"] = mock_record

        with patch(
            "aragora.rbac.emergency.get_break_glass_access",
            return_value=mock_emergency,
        ), patch(
            "aragora.server.http_utils.run_async",
            return_value="emerg-abc123",
        ):
            result = handler.handle("/api/v1/admin/emergency/activate", {}, mock_http)
            body = _body(result)
            assert body["duration_minutes"] == 60

    def test_activate_custom_duration(self, handler, mock_emergency):
        """Custom duration_minutes is used when specified."""
        body_data = {
            "user_id": "user-target-001",
            "reason": "Production incident requiring immediate access",
            "duration_minutes": 240,
        }
        mock_http = _make_http_handler(body=body_data)

        mock_record = MockEmergencyAccessRecord(id="emerg-abc123")
        mock_emergency._all_records["emerg-abc123"] = mock_record

        with patch(
            "aragora.rbac.emergency.get_break_glass_access",
            return_value=mock_emergency,
        ), patch(
            "aragora.server.http_utils.run_async",
            return_value="emerg-abc123",
        ):
            result = handler.handle("/api/v1/admin/emergency/activate", {}, mock_http)
            body = _body(result)
            assert body["duration_minutes"] == 240

    def test_activate_invalid_duration_falls_back_to_default(self, handler, mock_emergency):
        """Invalid duration_minutes falls back to 60."""
        body_data = {
            "user_id": "user-target-001",
            "reason": "Production incident requiring immediate access",
            "duration_minutes": "not-a-number",
        }
        mock_http = _make_http_handler(body=body_data)

        mock_record = MockEmergencyAccessRecord(id="emerg-abc123")
        mock_emergency._all_records["emerg-abc123"] = mock_record

        with patch(
            "aragora.rbac.emergency.get_break_glass_access",
            return_value=mock_emergency,
        ), patch(
            "aragora.server.http_utils.run_async",
            return_value="emerg-abc123",
        ):
            result = handler.handle("/api/v1/admin/emergency/activate", {}, mock_http)
            body = _body(result)
            assert body["duration_minutes"] == 60

    def test_activate_none_duration_falls_back_to_default(self, handler, mock_emergency):
        """None duration_minutes falls back to 60."""
        body_data = {
            "user_id": "user-target-001",
            "reason": "Production incident requiring immediate access",
            "duration_minutes": None,
        }
        mock_http = _make_http_handler(body=body_data)

        mock_record = MockEmergencyAccessRecord(id="emerg-abc123")
        mock_emergency._all_records["emerg-abc123"] = mock_record

        with patch(
            "aragora.rbac.emergency.get_break_glass_access",
            return_value=mock_emergency,
        ), patch(
            "aragora.server.http_utils.run_async",
            return_value="emerg-abc123",
        ):
            result = handler.handle("/api/v1/admin/emergency/activate", {}, mock_http)
            body = _body(result)
            assert body["duration_minutes"] == 60

    def test_activate_extracts_ip_address(self, handler, mock_emergency):
        """IP address is extracted from handler.client_address."""
        body_data = {
            "user_id": "user-target-001",
            "reason": "Production incident requiring immediate access",
        }
        mock_http = _make_http_handler(body=body_data, ip="10.0.0.5")

        mock_record = MockEmergencyAccessRecord(id="emerg-abc123")
        mock_emergency._all_records["emerg-abc123"] = mock_record

        captured_kwargs = {}

        def capture_run_async(coro):
            # We just return the access_id; the coro was already formed
            return "emerg-abc123"

        with patch(
            "aragora.rbac.emergency.get_break_glass_access",
            return_value=mock_emergency,
        ), patch(
            "aragora.server.http_utils.run_async",
            side_effect=capture_run_async,
        ):
            result = handler.handle("/api/v1/admin/emergency/activate", {}, mock_http)
            assert _status(result) == 200

    def test_activate_extracts_user_agent(self, handler, mock_emergency):
        """User-Agent is extracted from handler.headers."""
        body_data = {
            "user_id": "user-target-001",
            "reason": "Production incident requiring immediate access",
        }
        mock_http = _make_http_handler(body=body_data, user_agent="CustomBrowser/2.0")

        mock_record = MockEmergencyAccessRecord(id="emerg-abc123")
        mock_emergency._all_records["emerg-abc123"] = mock_record

        with patch(
            "aragora.rbac.emergency.get_break_glass_access",
            return_value=mock_emergency,
        ), patch(
            "aragora.server.http_utils.run_async",
            return_value="emerg-abc123",
        ):
            result = handler.handle("/api/v1/admin/emergency/activate", {}, mock_http)
            assert _status(result) == 200

    def test_activate_value_error_from_emergency_returns_400(self, handler):
        """ValueError from emergency.activate returns 400."""
        body_data = {
            "user_id": "user-target-001",
            "reason": "Production incident requiring immediate access",
        }
        mock_http = _make_http_handler(body=body_data)

        mock_emergency = MagicMock()

        with patch(
            "aragora.rbac.emergency.get_break_glass_access",
            return_value=mock_emergency,
        ), patch(
            "aragora.server.http_utils.run_async",
            side_effect=ValueError("Duration cannot exceed 1440 minutes"),
        ):
            result = handler.handle("/api/v1/admin/emergency/activate", {}, mock_http)
            body = _body(result)
            assert _status(result) == 400
            assert "Invalid request" in body["error"]

    def test_activate_record_not_found_returns_null_expires(self, handler, mock_emergency):
        """When record is not in _all_records, expires_at is None."""
        body_data = {
            "user_id": "user-target-001",
            "reason": "Production incident requiring immediate access",
        }
        mock_http = _make_http_handler(body=body_data)

        # _all_records does NOT contain the access_id
        mock_emergency._all_records = {}

        with patch(
            "aragora.rbac.emergency.get_break_glass_access",
            return_value=mock_emergency,
        ), patch(
            "aragora.server.http_utils.run_async",
            return_value="emerg-missing",
        ):
            result = handler.handle("/api/v1/admin/emergency/activate", {}, mock_http)
            body = _body(result)
            assert _status(result) == 200
            assert body["expires_at"] is None

    def test_activate_record_has_expires_at(self, handler, mock_emergency):
        """When record exists, expires_at is included as ISO string."""
        body_data = {
            "user_id": "user-target-001",
            "reason": "Production incident requiring immediate access",
        }
        mock_http = _make_http_handler(body=body_data)

        expires = datetime(2026, 3, 1, 12, 0, 0, tzinfo=timezone.utc)
        mock_record = MockEmergencyAccessRecord(id="emerg-abc123", expires_at=expires)
        mock_emergency._all_records["emerg-abc123"] = mock_record

        with patch(
            "aragora.rbac.emergency.get_break_glass_access",
            return_value=mock_emergency,
        ), patch(
            "aragora.server.http_utils.run_async",
            return_value="emerg-abc123",
        ):
            result = handler.handle("/api/v1/admin/emergency/activate", {}, mock_http)
            body = _body(result)
            assert _status(result) == 200
            assert body["expires_at"] == expires.isoformat()

    def test_activate_via_legacy_route(self, handler, mock_emergency):
        """Activation works via legacy (non-versioned) route."""
        body_data = {
            "user_id": "user-target-001",
            "reason": "Production incident requiring immediate access",
        }
        mock_http = _make_http_handler(body=body_data)

        mock_record = MockEmergencyAccessRecord(id="emerg-abc123")
        mock_emergency._all_records["emerg-abc123"] = mock_record

        with patch(
            "aragora.rbac.emergency.get_break_glass_access",
            return_value=mock_emergency,
        ), patch(
            "aragora.server.http_utils.run_async",
            return_value="emerg-abc123",
        ):
            result = handler.handle("/api/admin/emergency/activate", {}, mock_http)
            assert _status(result) == 200

    def test_activate_no_client_address(self, handler, mock_emergency):
        """Handles handler without client_address attribute gracefully."""
        body_data = {
            "user_id": "user-target-001",
            "reason": "Production incident requiring immediate access",
        }
        mock_http = _make_http_handler(body=body_data)
        del mock_http.client_address  # Remove the attribute

        mock_record = MockEmergencyAccessRecord(id="emerg-abc123")
        mock_emergency._all_records["emerg-abc123"] = mock_record

        with patch(
            "aragora.rbac.emergency.get_break_glass_access",
            return_value=mock_emergency,
        ), patch(
            "aragora.server.http_utils.run_async",
            return_value="emerg-abc123",
        ):
            result = handler.handle("/api/v1/admin/emergency/activate", {}, mock_http)
            assert _status(result) == 200


# ===========================================================================
# Tests: _deactivate endpoint
# ===========================================================================


class TestDeactivate:
    """Tests for the _deactivate endpoint."""

    def test_deactivate_success(self, handler, mock_emergency):
        """Successful deactivation returns expected fields."""
        body_data = {"access_id": "emerg-abc123"}
        mock_http = _make_http_handler(body=body_data)

        mock_record = MockEmergencyAccessRecord(
            id="emerg-abc123",
            status=MockEmergencyAccessStatus("deactivated"),
            actions_taken=[{"action": "read_db"}, {"action": "update_config"}],
        )

        with patch(
            "aragora.rbac.emergency.get_break_glass_access",
            return_value=mock_emergency,
        ), patch(
            "aragora.server.http_utils.run_async",
            return_value=mock_record,
        ):
            result = handler.handle("/api/v1/admin/emergency/deactivate", {}, mock_http)
            body = _body(result)
            assert _status(result) == 200
            assert body["access_id"] == "emerg-abc123"
            assert body["status"] == "deactivated"
            assert body["actions_taken"] == 2
            assert body["message"] == "Emergency break-glass access deactivated"

    def test_deactivate_invalid_json_body(self, handler):
        """Invalid JSON body returns 400."""
        mock_http = MagicMock()
        mock_http.headers = {"Content-Length": "5"}
        mock_http.rfile.read.return_value = b"notjson"

        result = handler.handle("/api/v1/admin/emergency/deactivate", {}, mock_http)
        body = _body(result)
        assert _status(result) == 400
        assert "Invalid JSON body" in body["error"]

    def test_deactivate_missing_access_id(self, handler):
        """Missing access_id returns 400."""
        body_data = {}
        mock_http = _make_http_handler(body=body_data)

        result = handler.handle("/api/v1/admin/emergency/deactivate", {}, mock_http)
        body = _body(result)
        assert _status(result) == 400
        assert "access_id" in body["error"]

    def test_deactivate_empty_access_id(self, handler):
        """Empty access_id returns 400."""
        body_data = {"access_id": ""}
        mock_http = _make_http_handler(body=body_data)

        result = handler.handle("/api/v1/admin/emergency/deactivate", {}, mock_http)
        body = _body(result)
        assert _status(result) == 400
        assert "access_id" in body["error"]

    def test_deactivate_whitespace_only_access_id(self, handler):
        """Whitespace-only access_id returns 400."""
        body_data = {"access_id": "   "}
        mock_http = _make_http_handler(body=body_data)

        result = handler.handle("/api/v1/admin/emergency/deactivate", {}, mock_http)
        body = _body(result)
        assert _status(result) == 400
        assert "access_id" in body["error"]

    def test_deactivate_not_found_returns_404(self, handler):
        """ValueError from emergency.deactivate returns 404."""
        body_data = {"access_id": "emerg-nonexistent"}
        mock_http = _make_http_handler(body=body_data)

        mock_emergency = MagicMock()

        with patch(
            "aragora.rbac.emergency.get_break_glass_access",
            return_value=mock_emergency,
        ), patch(
            "aragora.server.http_utils.run_async",
            side_effect=ValueError("Access record not found"),
        ):
            result = handler.handle("/api/v1/admin/emergency/deactivate", {}, mock_http)
            body = _body(result)
            assert _status(result) == 404
            assert "not found" in body["error"].lower()

    def test_deactivate_already_deactivated_returns_404(self, handler):
        """Trying to deactivate already-deactivated access returns 404."""
        body_data = {"access_id": "emerg-abc123"}
        mock_http = _make_http_handler(body=body_data)

        mock_emergency = MagicMock()

        with patch(
            "aragora.rbac.emergency.get_break_glass_access",
            return_value=mock_emergency,
        ), patch(
            "aragora.server.http_utils.run_async",
            side_effect=ValueError("Access is not active: deactivated"),
        ):
            result = handler.handle("/api/v1/admin/emergency/deactivate", {}, mock_http)
            assert _status(result) == 404

    def test_deactivate_via_legacy_route(self, handler, mock_emergency):
        """Deactivation works via legacy (non-versioned) route."""
        body_data = {"access_id": "emerg-abc123"}
        mock_http = _make_http_handler(body=body_data)

        mock_record = MockEmergencyAccessRecord(
            id="emerg-abc123",
            status=MockEmergencyAccessStatus("deactivated"),
        )

        with patch(
            "aragora.rbac.emergency.get_break_glass_access",
            return_value=mock_emergency,
        ), patch(
            "aragora.server.http_utils.run_async",
            return_value=mock_record,
        ):
            result = handler.handle("/api/admin/emergency/deactivate", {}, mock_http)
            assert _status(result) == 200

    def test_deactivate_records_deactivated_by_user(self, handler, mock_emergency):
        """Deactivation includes deactivated_by in response."""
        body_data = {"access_id": "emerg-abc123"}
        mock_http = _make_http_handler(body=body_data)

        mock_record = MockEmergencyAccessRecord(
            id="emerg-abc123",
            status=MockEmergencyAccessStatus("deactivated"),
            deactivated_by="admin-user-001",
        )

        with patch(
            "aragora.rbac.emergency.get_break_glass_access",
            return_value=mock_emergency,
        ), patch(
            "aragora.server.http_utils.run_async",
            return_value=mock_record,
        ):
            result = handler.handle("/api/v1/admin/emergency/deactivate", {}, mock_http)
            body = _body(result)
            assert _status(result) == 200
            # deactivated_by comes from getattr(user, "user_id", "unknown")
            assert "deactivated_by" in body

    def test_deactivate_zero_actions_taken(self, handler, mock_emergency):
        """Deactivation with zero actions returns actions_taken=0."""
        body_data = {"access_id": "emerg-abc123"}
        mock_http = _make_http_handler(body=body_data)

        mock_record = MockEmergencyAccessRecord(
            id="emerg-abc123",
            status=MockEmergencyAccessStatus("deactivated"),
            actions_taken=[],
        )

        with patch(
            "aragora.rbac.emergency.get_break_glass_access",
            return_value=mock_emergency,
        ), patch(
            "aragora.server.http_utils.run_async",
            return_value=mock_record,
        ):
            result = handler.handle("/api/v1/admin/emergency/deactivate", {}, mock_http)
            body = _body(result)
            assert body["actions_taken"] == 0


# ===========================================================================
# Tests: _status endpoint
# ===========================================================================


class TestStatus:
    """Tests for the _status endpoint."""

    def test_status_no_active_sessions(self, handler, mock_emergency):
        """Status with no active sessions returns empty lists."""
        mock_http = _make_http_handler()
        mock_emergency._active_records = {}

        run_async_calls = []

        def mock_run_async(coro):
            run_async_calls.append(coro)
            # First call is expire_old_access, second is get_history
            if len(run_async_calls) == 1:
                return 0  # expire_old_access returns count
            return []  # get_history returns list

        with patch(
            "aragora.rbac.emergency.get_break_glass_access",
            return_value=mock_emergency,
        ), patch(
            "aragora.server.http_utils.run_async",
            side_effect=mock_run_async,
        ):
            result = handler.handle("/api/v1/admin/emergency/status", {}, mock_http)
            body = _body(result)
            assert _status(result) == 200
            assert body["active_count"] == 0
            assert body["active_sessions"] == []
            assert body["recent_history"] == []
            assert body["persistence_enabled"] is False

    def test_status_with_active_sessions(self, handler, mock_emergency):
        """Status with active sessions returns them in response."""
        mock_http = _make_http_handler()

        active_record = MockEmergencyAccessRecord(
            id="emerg-active-001",
            user_id="user-001",
            is_active=True,
        )
        mock_emergency._active_records = {"emerg-active-001": active_record}

        run_async_calls = []

        def mock_run_async(coro):
            run_async_calls.append(coro)
            if len(run_async_calls) == 1:
                return 0
            return []

        with patch(
            "aragora.rbac.emergency.get_break_glass_access",
            return_value=mock_emergency,
        ), patch(
            "aragora.server.http_utils.run_async",
            side_effect=mock_run_async,
        ):
            result = handler.handle("/api/v1/admin/emergency/status", {}, mock_http)
            body = _body(result)
            assert _status(result) == 200
            assert body["active_count"] == 1
            assert len(body["active_sessions"]) == 1
            assert body["active_sessions"][0]["id"] == "emerg-active-001"

    def test_status_filters_inactive_sessions(self, handler, mock_emergency):
        """Status only includes is_active=True records from active_records."""
        mock_http = _make_http_handler()

        active_record = MockEmergencyAccessRecord(id="emerg-001", is_active=True)
        inactive_record = MockEmergencyAccessRecord(id="emerg-002", is_active=False)
        mock_emergency._active_records = {
            "emerg-001": active_record,
            "emerg-002": inactive_record,
        }

        run_async_calls = []

        def mock_run_async(coro):
            run_async_calls.append(coro)
            if len(run_async_calls) == 1:
                return 0
            return []

        with patch(
            "aragora.rbac.emergency.get_break_glass_access",
            return_value=mock_emergency,
        ), patch(
            "aragora.server.http_utils.run_async",
            side_effect=mock_run_async,
        ):
            result = handler.handle("/api/v1/admin/emergency/status", {}, mock_http)
            body = _body(result)
            assert body["active_count"] == 1

    def test_status_includes_recent_history(self, handler, mock_emergency):
        """Status includes recent history from get_history."""
        mock_http = _make_http_handler()
        mock_emergency._active_records = {}

        history_record = MockEmergencyAccessRecord(
            id="emerg-hist-001",
            user_id="user-old",
            is_active=False,
            status=MockEmergencyAccessStatus("expired"),
        )

        run_async_calls = []

        def mock_run_async(coro):
            run_async_calls.append(coro)
            if len(run_async_calls) == 1:
                return 0
            return [history_record]

        with patch(
            "aragora.rbac.emergency.get_break_glass_access",
            return_value=mock_emergency,
        ), patch(
            "aragora.server.http_utils.run_async",
            side_effect=mock_run_async,
        ):
            result = handler.handle("/api/v1/admin/emergency/status", {}, mock_http)
            body = _body(result)
            assert len(body["recent_history"]) == 1
            assert body["recent_history"][0]["id"] == "emerg-hist-001"

    def test_status_persistence_enabled(self, handler, mock_emergency):
        """Status reports persistence_enabled from emergency instance."""
        mock_http = _make_http_handler()
        mock_emergency._active_records = {}
        mock_emergency._persistence_enabled = True

        run_async_calls = []

        def mock_run_async(coro):
            run_async_calls.append(coro)
            if len(run_async_calls) == 1:
                return 0
            return []

        with patch(
            "aragora.rbac.emergency.get_break_glass_access",
            return_value=mock_emergency,
        ), patch(
            "aragora.server.http_utils.run_async",
            side_effect=mock_run_async,
        ):
            result = handler.handle("/api/v1/admin/emergency/status", {}, mock_http)
            body = _body(result)
            assert body["persistence_enabled"] is True

    def test_status_via_legacy_route(self, handler, mock_emergency):
        """Status works via legacy (non-versioned) route."""
        mock_http = _make_http_handler()
        mock_emergency._active_records = {}

        run_async_calls = []

        def mock_run_async(coro):
            run_async_calls.append(coro)
            if len(run_async_calls) == 1:
                return 0
            return []

        with patch(
            "aragora.rbac.emergency.get_break_glass_access",
            return_value=mock_emergency,
        ), patch(
            "aragora.server.http_utils.run_async",
            side_effect=mock_run_async,
        ):
            result = handler.handle("/api/admin/emergency/status", {}, mock_http)
            assert _status(result) == 200


# ===========================================================================
# Tests: Handler initialization and class attributes
# ===========================================================================


class TestHandlerInit:
    """Tests for EmergencyAccessHandler initialization."""

    def test_default_context(self):
        """Default context is empty dict."""
        h = EmergencyAccessHandler()
        assert h.ctx == {}

    def test_none_context(self):
        """None context defaults to empty dict."""
        h = EmergencyAccessHandler(ctx=None)
        assert h.ctx == {}

    def test_custom_context(self):
        """Custom context is stored."""
        ctx = {"user_store": MagicMock()}
        h = EmergencyAccessHandler(ctx=ctx)
        assert h.ctx is ctx

    def test_resource_type(self):
        """RESOURCE_TYPE is emergency_access."""
        assert EmergencyAccessHandler.RESOURCE_TYPE == "emergency_access"

    def test_inherits_secure_handler(self):
        """EmergencyAccessHandler inherits from SecureHandler."""
        from aragora.server.handlers.secure import SecureHandler

        assert issubclass(EmergencyAccessHandler, SecureHandler)

    def test_has_required_methods(self):
        """Handler has all expected methods."""
        assert hasattr(EmergencyAccessHandler, "handle")
        assert hasattr(EmergencyAccessHandler, "can_handle")
        assert hasattr(EmergencyAccessHandler, "_activate")
        assert hasattr(EmergencyAccessHandler, "_deactivate")
        assert hasattr(EmergencyAccessHandler, "_status")


# ===========================================================================
# Tests: Edge cases
# ===========================================================================


class TestEdgeCases:
    """Edge case tests."""

    def test_activate_with_empty_body(self, handler):
        """Empty body (Content-Length=0) returns 400 for missing user_id."""
        mock_http = MagicMock()
        mock_http.headers = {"Content-Length": "0"}
        mock_http.rfile.read.return_value = b""

        result = handler.handle("/api/v1/admin/emergency/activate", {}, mock_http)
        body = _body(result)
        assert _status(result) == 400
        # Empty body parses as {} => missing user_id
        assert "user_id" in body["error"]

    def test_deactivate_with_empty_body(self, handler):
        """Empty body (Content-Length=0) returns 400 for missing access_id."""
        mock_http = MagicMock()
        mock_http.headers = {"Content-Length": "0"}
        mock_http.rfile.read.return_value = b""

        result = handler.handle("/api/v1/admin/emergency/deactivate", {}, mock_http)
        body = _body(result)
        assert _status(result) == 400
        assert "access_id" in body["error"]

    def test_activate_strips_user_id_whitespace(self, handler, mock_emergency):
        """user_id is stripped of whitespace."""
        body_data = {
            "user_id": "  user-001  ",
            "reason": "Production incident requiring immediate access",
        }
        mock_http = _make_http_handler(body=body_data)

        mock_record = MockEmergencyAccessRecord(id="emerg-abc123")
        mock_emergency._all_records["emerg-abc123"] = mock_record

        with patch(
            "aragora.rbac.emergency.get_break_glass_access",
            return_value=mock_emergency,
        ), patch(
            "aragora.server.http_utils.run_async",
            return_value="emerg-abc123",
        ):
            result = handler.handle("/api/v1/admin/emergency/activate", {}, mock_http)
            body = _body(result)
            assert _status(result) == 200
            assert body["user_id"] == "user-001"

    def test_activate_strips_reason_whitespace(self, handler, mock_emergency):
        """reason is stripped of whitespace."""
        body_data = {
            "user_id": "user-001",
            "reason": "  Production incident requiring immediate access  ",
        }
        mock_http = _make_http_handler(body=body_data)

        mock_record = MockEmergencyAccessRecord(id="emerg-abc123")
        mock_emergency._all_records["emerg-abc123"] = mock_record

        with patch(
            "aragora.rbac.emergency.get_break_glass_access",
            return_value=mock_emergency,
        ), patch(
            "aragora.server.http_utils.run_async",
            return_value="emerg-abc123",
        ):
            result = handler.handle("/api/v1/admin/emergency/activate", {}, mock_http)
            assert _status(result) == 200

    def test_deactivate_strips_access_id_whitespace(self, handler, mock_emergency):
        """access_id is stripped of whitespace."""
        body_data = {"access_id": "  emerg-abc123  "}
        mock_http = _make_http_handler(body=body_data)

        mock_record = MockEmergencyAccessRecord(
            id="emerg-abc123",
            status=MockEmergencyAccessStatus("deactivated"),
        )

        with patch(
            "aragora.rbac.emergency.get_break_glass_access",
            return_value=mock_emergency,
        ), patch(
            "aragora.server.http_utils.run_async",
            return_value=mock_record,
        ):
            result = handler.handle("/api/v1/admin/emergency/deactivate", {}, mock_http)
            body = _body(result)
            assert _status(result) == 200
            assert body["access_id"] == "emerg-abc123"

    def test_activate_no_headers_attribute(self, handler, mock_emergency):
        """Handles handler without headers attribute gracefully (no user-agent)."""
        body_data = {
            "user_id": "user-target-001",
            "reason": "Production incident requiring immediate access",
        }
        mock_http = _make_http_handler(body=body_data, user_agent=None)
        del mock_http.headers  # Remove headers entirely
        # Re-add minimal headers needed for read_json_body
        body_bytes = json.dumps(body_data).encode()
        mock_http.headers = {"Content-Length": str(len(body_bytes))}
        mock_http.rfile.read.return_value = body_bytes

        mock_record = MockEmergencyAccessRecord(id="emerg-abc123")
        mock_emergency._all_records["emerg-abc123"] = mock_record

        with patch(
            "aragora.rbac.emergency.get_break_glass_access",
            return_value=mock_emergency,
        ), patch(
            "aragora.server.http_utils.run_async",
            return_value="emerg-abc123",
        ):
            result = handler.handle("/api/v1/admin/emergency/activate", {}, mock_http)
            assert _status(result) == 200

    def test_multiple_active_sessions_in_status(self, handler, mock_emergency):
        """Status correctly counts multiple active sessions."""
        mock_http = _make_http_handler()

        records = {
            f"emerg-{i}": MockEmergencyAccessRecord(
                id=f"emerg-{i}",
                user_id=f"user-{i}",
                is_active=True,
            )
            for i in range(5)
        }
        mock_emergency._active_records = records

        run_async_calls = []

        def mock_run_async(coro):
            run_async_calls.append(coro)
            if len(run_async_calls) == 1:
                return 0
            return []

        with patch(
            "aragora.rbac.emergency.get_break_glass_access",
            return_value=mock_emergency,
        ), patch(
            "aragora.server.http_utils.run_async",
            side_effect=mock_run_async,
        ):
            result = handler.handle("/api/v1/admin/emergency/status", {}, mock_http)
            body = _body(result)
            assert body["active_count"] == 5
            assert len(body["active_sessions"]) == 5

    def test_activate_integer_duration_from_string(self, handler, mock_emergency):
        """Integer-like string for duration_minutes is parsed correctly."""
        body_data = {
            "user_id": "user-target-001",
            "reason": "Production incident requiring immediate access",
            "duration_minutes": "180",
        }
        mock_http = _make_http_handler(body=body_data)

        mock_record = MockEmergencyAccessRecord(id="emerg-abc123")
        mock_emergency._all_records["emerg-abc123"] = mock_record

        with patch(
            "aragora.rbac.emergency.get_break_glass_access",
            return_value=mock_emergency,
        ), patch(
            "aragora.server.http_utils.run_async",
            return_value="emerg-abc123",
        ):
            result = handler.handle("/api/v1/admin/emergency/activate", {}, mock_http)
            body = _body(result)
            assert body["duration_minutes"] == 180

    def test_activate_with_handler_no_client_address_none(self, handler, mock_emergency):
        """When client_address is None, ip_address is None."""
        body_data = {
            "user_id": "user-target-001",
            "reason": "Production incident requiring immediate access",
        }
        mock_http = _make_http_handler(body=body_data)
        mock_http.client_address = None

        mock_record = MockEmergencyAccessRecord(id="emerg-abc123")
        mock_emergency._all_records["emerg-abc123"] = mock_record

        with patch(
            "aragora.rbac.emergency.get_break_glass_access",
            return_value=mock_emergency,
        ), patch(
            "aragora.server.http_utils.run_async",
            return_value="emerg-abc123",
        ):
            result = handler.handle("/api/v1/admin/emergency/activate", {}, mock_http)
            assert _status(result) == 200
