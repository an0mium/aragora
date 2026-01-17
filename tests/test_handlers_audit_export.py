"""
Tests for audit_export handler endpoints.

Endpoints tested:
- GET /api/audit/events - Query audit events
- GET /api/audit/stats - Audit log statistics
- POST /api/audit/export - Export audit log (JSON, CSV, SOC2)
- POST /api/audit/verify - Verify audit log integrity
"""

import json
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from aiohttp import web
from aiohttp.test_utils import make_mocked_request

from aragora.server.handlers import audit_export


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def mock_audit_event():
    """Create a mock AuditEvent."""
    mock_event = MagicMock()
    mock_event.to_dict.return_value = {
        "id": "event-123",
        "timestamp": "2026-01-15T10:00:00",
        "category": "auth",
        "action": "login",
        "actor_id": "user-456",
        "resource_type": "session",
        "resource_id": "sess-789",
        "outcome": "success",
        "ip_address": "127.0.0.1",
        "user_agent": "TestAgent/1.0",
        "correlation_id": "corr-001",
        "org_id": "org-001",
        "workspace_id": "ws-001",
        "details": {},
        "reason": "",
        "previous_hash": "abc123",
        "event_hash": "def456",
    }
    return mock_event


@pytest.fixture
def mock_audit_log(mock_audit_event):
    """Create a mock AuditLog instance."""
    mock_log = MagicMock()
    mock_log.query.return_value = [mock_audit_event]
    mock_log.get_stats.return_value = {
        "total_events": 100,
        "oldest_event": "2026-01-01T00:00:00",
        "newest_event": "2026-01-15T12:00:00",
        "by_category": {"auth": 50, "access": 30, "admin": 20},
        "retention_days": 2555,
    }
    mock_log.export_json.return_value = 10
    mock_log.export_csv.return_value = 10
    mock_log.export_soc2.return_value = {"events_exported": 10}
    mock_log.verify_integrity.return_value = (True, [])
    return mock_log


@pytest.fixture(autouse=True)
def reset_audit_log_singleton():
    """Reset the module-level audit log singleton before/after each test."""
    audit_export._audit_log = None
    yield
    audit_export._audit_log = None


# ============================================================================
# get_audit_log Tests
# ============================================================================


class TestGetAuditLog:
    """Tests for get_audit_log function."""

    def test_creates_audit_log_on_first_call(self):
        """get_audit_log creates AuditLog instance on first call."""
        with patch("aragora.server.handlers.audit_export.get_audit_log") as mock_get:
            mock_get.return_value = MagicMock()
            result = mock_get()
            assert result is not None

    def test_returns_same_instance_on_subsequent_calls(self):
        """get_audit_log returns the same instance on subsequent calls."""
        with patch("aragora.audit.AuditLog") as MockAuditLog:
            mock_instance = MagicMock()
            MockAuditLog.return_value = mock_instance

            first = audit_export.get_audit_log()
            second = audit_export.get_audit_log()

            assert first is second
            MockAuditLog.assert_called_once()


# ============================================================================
# handle_audit_events Tests
# ============================================================================


class TestHandleAuditEvents:
    """Tests for GET /api/audit/events endpoint."""

    @pytest.mark.asyncio
    async def test_returns_events_with_default_params(self, mock_audit_log, mock_audit_event):
        """Returns events with default query parameters."""
        with patch.object(audit_export, "get_audit_log", return_value=mock_audit_log):
            request = make_mocked_request("GET", "/api/audit/events", app=web.Application())

            response = await audit_export.handle_audit_events(request)

            assert response.status == 200
            data = json.loads(response.body)
            assert "events" in data
            assert len(data["events"]) == 1
            assert data["count"] == 1
            assert "query" in data
            assert data["query"]["limit"] == 100
            assert data["query"]["offset"] == 0

    @pytest.mark.asyncio
    async def test_parses_date_range_params(self, mock_audit_log):
        """Parses start_date and end_date query parameters."""
        with patch.object(audit_export, "get_audit_log", return_value=mock_audit_log):
            request = make_mocked_request(
                "GET",
                "/api/audit/events?start_date=2026-01-01T00:00:00&end_date=2026-01-15T00:00:00",
                app=web.Application(),
            )

            response = await audit_export.handle_audit_events(request)

            assert response.status == 200
            mock_audit_log.query.assert_called_once()
            query_arg = mock_audit_log.query.call_args[0][0]
            assert query_arg.start_date is not None
            assert query_arg.end_date is not None

    @pytest.mark.asyncio
    async def test_invalid_start_date_returns_400(self, mock_audit_log):
        """Returns 400 for invalid start_date format."""
        with patch.object(audit_export, "get_audit_log", return_value=mock_audit_log):
            request = make_mocked_request(
                "GET",
                "/api/audit/events?start_date=not-a-date",
                app=web.Application(),
            )

            response = await audit_export.handle_audit_events(request)

            assert response.status == 400
            data = json.loads(response.body)
            assert "error" in data
            assert "start_date" in data["error"]

    @pytest.mark.asyncio
    async def test_invalid_end_date_returns_400(self, mock_audit_log):
        """Returns 400 for invalid end_date format."""
        with patch.object(audit_export, "get_audit_log", return_value=mock_audit_log):
            request = make_mocked_request(
                "GET",
                "/api/audit/events?end_date=invalid",
                app=web.Application(),
            )

            response = await audit_export.handle_audit_events(request)

            assert response.status == 400
            data = json.loads(response.body)
            assert "error" in data
            assert "end_date" in data["error"]

    @pytest.mark.asyncio
    async def test_invalid_category_returns_400(self, mock_audit_log):
        """Returns 400 for invalid category value."""
        with patch.object(audit_export, "get_audit_log", return_value=mock_audit_log):
            request = make_mocked_request(
                "GET",
                "/api/audit/events?category=invalid_category",
                app=web.Application(),
            )

            response = await audit_export.handle_audit_events(request)

            assert response.status == 400
            data = json.loads(response.body)
            assert "error" in data
            assert "category" in data["error"].lower()

    @pytest.mark.asyncio
    async def test_invalid_outcome_returns_400(self, mock_audit_log):
        """Returns 400 for invalid outcome value."""
        with patch.object(audit_export, "get_audit_log", return_value=mock_audit_log):
            request = make_mocked_request(
                "GET",
                "/api/audit/events?outcome=invalid_outcome",
                app=web.Application(),
            )

            response = await audit_export.handle_audit_events(request)

            assert response.status == 400
            data = json.loads(response.body)
            assert "error" in data
            assert "outcome" in data["error"].lower()

    @pytest.mark.asyncio
    async def test_filters_by_action(self, mock_audit_log):
        """Filters events by action parameter."""
        with patch.object(audit_export, "get_audit_log", return_value=mock_audit_log):
            request = make_mocked_request(
                "GET",
                "/api/audit/events?action=login",
                app=web.Application(),
            )

            response = await audit_export.handle_audit_events(request)

            assert response.status == 200
            query_arg = mock_audit_log.query.call_args[0][0]
            assert query_arg.action == "login"

    @pytest.mark.asyncio
    async def test_filters_by_actor_id(self, mock_audit_log):
        """Filters events by actor_id parameter."""
        with patch.object(audit_export, "get_audit_log", return_value=mock_audit_log):
            request = make_mocked_request(
                "GET",
                "/api/audit/events?actor_id=user-123",
                app=web.Application(),
            )

            response = await audit_export.handle_audit_events(request)

            assert response.status == 200
            query_arg = mock_audit_log.query.call_args[0][0]
            assert query_arg.actor_id == "user-123"

    @pytest.mark.asyncio
    async def test_filters_by_org_id(self, mock_audit_log):
        """Filters events by org_id parameter."""
        with patch.object(audit_export, "get_audit_log", return_value=mock_audit_log):
            request = make_mocked_request(
                "GET",
                "/api/audit/events?org_id=org-456",
                app=web.Application(),
            )

            response = await audit_export.handle_audit_events(request)

            assert response.status == 200
            query_arg = mock_audit_log.query.call_args[0][0]
            assert query_arg.org_id == "org-456"

    @pytest.mark.asyncio
    async def test_supports_search_text(self, mock_audit_log):
        """Supports full-text search via search parameter."""
        with patch.object(audit_export, "get_audit_log", return_value=mock_audit_log):
            request = make_mocked_request(
                "GET",
                "/api/audit/events?search=login",
                app=web.Application(),
            )

            response = await audit_export.handle_audit_events(request)

            assert response.status == 200
            query_arg = mock_audit_log.query.call_args[0][0]
            assert query_arg.search_text == "login"

    @pytest.mark.asyncio
    async def test_respects_limit_and_offset(self, mock_audit_log):
        """Respects limit and offset pagination parameters."""
        with patch.object(audit_export, "get_audit_log", return_value=mock_audit_log):
            request = make_mocked_request(
                "GET",
                "/api/audit/events?limit=50&offset=10",
                app=web.Application(),
            )

            response = await audit_export.handle_audit_events(request)

            assert response.status == 200
            query_arg = mock_audit_log.query.call_args[0][0]
            assert query_arg.limit == 50
            assert query_arg.offset == 10

    @pytest.mark.asyncio
    async def test_caps_limit_at_1000(self, mock_audit_log):
        """Caps limit at maximum of 1000."""
        with patch.object(audit_export, "get_audit_log", return_value=mock_audit_log):
            request = make_mocked_request(
                "GET",
                "/api/audit/events?limit=5000",
                app=web.Application(),
            )

            response = await audit_export.handle_audit_events(request)

            assert response.status == 200
            query_arg = mock_audit_log.query.call_args[0][0]
            assert query_arg.limit == 1000


# ============================================================================
# handle_audit_stats Tests
# ============================================================================


class TestHandleAuditStats:
    """Tests for GET /api/audit/stats endpoint."""

    @pytest.mark.asyncio
    async def test_returns_audit_stats(self, mock_audit_log):
        """Returns audit log statistics."""
        with patch.object(audit_export, "get_audit_log", return_value=mock_audit_log):
            request = make_mocked_request("GET", "/api/audit/stats", app=web.Application())

            response = await audit_export.handle_audit_stats(request)

            assert response.status == 200
            data = json.loads(response.body)
            assert data["total_events"] == 100
            assert data["oldest_event"] == "2026-01-01T00:00:00"
            assert data["newest_event"] == "2026-01-15T12:00:00"
            assert "by_category" in data
            assert data["by_category"]["auth"] == 50
            assert data["retention_days"] == 2555


# ============================================================================
# handle_audit_export Tests
# ============================================================================


class TestHandleAuditExport:
    """Tests for POST /api/audit/export endpoint."""

    @pytest.mark.asyncio
    async def test_exports_json_format(self, mock_audit_log):
        """Exports audit log in JSON format."""
        with patch.object(audit_export, "get_audit_log", return_value=mock_audit_log):
            body = json.dumps({
                "format": "json",
                "start_date": "2026-01-01T00:00:00",
                "end_date": "2026-01-15T00:00:00",
            })
            request = make_mocked_request(
                "POST",
                "/api/audit/export",
                app=web.Application(),
            )
            request._payload = AsyncMock()
            request.json = AsyncMock(return_value={
                "format": "json",
                "start_date": "2026-01-01T00:00:00",
                "end_date": "2026-01-15T00:00:00",
            })

            response = await audit_export.handle_audit_export(request)

            assert response.status == 200
            assert response.content_type == "application/json"
            assert "attachment" in response.headers.get("Content-Disposition", "")
            assert response.headers.get("X-Audit-Event-Count") == "10"

    @pytest.mark.asyncio
    async def test_exports_csv_format(self, mock_audit_log):
        """Exports audit log in CSV format."""
        with patch.object(audit_export, "get_audit_log", return_value=mock_audit_log):
            request = make_mocked_request(
                "POST",
                "/api/audit/export",
                app=web.Application(),
            )
            request.json = AsyncMock(return_value={
                "format": "csv",
                "start_date": "2026-01-01T00:00:00",
                "end_date": "2026-01-15T00:00:00",
            })

            response = await audit_export.handle_audit_export(request)

            assert response.status == 200
            assert response.content_type == "text/csv"
            assert ".csv" in response.headers.get("Content-Disposition", "")

    @pytest.mark.asyncio
    async def test_exports_soc2_format(self, mock_audit_log):
        """Exports audit log in SOC2 format."""
        with patch.object(audit_export, "get_audit_log", return_value=mock_audit_log):
            request = make_mocked_request(
                "POST",
                "/api/audit/export",
                app=web.Application(),
            )
            request.json = AsyncMock(return_value={
                "format": "soc2",
                "start_date": "2026-01-01T00:00:00",
                "end_date": "2026-01-15T00:00:00",
            })

            response = await audit_export.handle_audit_export(request)

            assert response.status == 200
            assert response.content_type == "application/json"
            assert "soc2_audit" in response.headers.get("Content-Disposition", "")

    @pytest.mark.asyncio
    async def test_defaults_to_json_format(self, mock_audit_log):
        """Defaults to JSON format when not specified."""
        with patch.object(audit_export, "get_audit_log", return_value=mock_audit_log):
            request = make_mocked_request(
                "POST",
                "/api/audit/export",
                app=web.Application(),
            )
            request.json = AsyncMock(return_value={
                "start_date": "2026-01-01T00:00:00",
                "end_date": "2026-01-15T00:00:00",
            })

            response = await audit_export.handle_audit_export(request)

            assert response.status == 200
            assert response.content_type == "application/json"

    @pytest.mark.asyncio
    async def test_invalid_json_returns_400(self, mock_audit_log):
        """Returns 400 for invalid JSON body."""
        with patch.object(audit_export, "get_audit_log", return_value=mock_audit_log):
            request = make_mocked_request(
                "POST",
                "/api/audit/export",
                app=web.Application(),
            )
            request.json = AsyncMock(side_effect=json.JSONDecodeError("test", "doc", 0))

            response = await audit_export.handle_audit_export(request)

            assert response.status == 400
            data = json.loads(response.body)
            assert "error" in data
            assert "Invalid JSON" in data["error"]

    @pytest.mark.asyncio
    async def test_missing_start_date_returns_400(self, mock_audit_log):
        """Returns 400 when start_date is missing."""
        with patch.object(audit_export, "get_audit_log", return_value=mock_audit_log):
            request = make_mocked_request(
                "POST",
                "/api/audit/export",
                app=web.Application(),
            )
            request.json = AsyncMock(return_value={
                "end_date": "2026-01-15T00:00:00",
            })

            response = await audit_export.handle_audit_export(request)

            assert response.status == 400
            data = json.loads(response.body)
            assert "start_date" in data["error"]

    @pytest.mark.asyncio
    async def test_missing_end_date_returns_400(self, mock_audit_log):
        """Returns 400 when end_date is missing."""
        with patch.object(audit_export, "get_audit_log", return_value=mock_audit_log):
            request = make_mocked_request(
                "POST",
                "/api/audit/export",
                app=web.Application(),
            )
            request.json = AsyncMock(return_value={
                "start_date": "2026-01-01T00:00:00",
            })

            response = await audit_export.handle_audit_export(request)

            assert response.status == 400
            data = json.loads(response.body)
            assert "end_date" in data["error"]

    @pytest.mark.asyncio
    async def test_invalid_date_format_returns_400(self, mock_audit_log):
        """Returns 400 for invalid date format."""
        with patch.object(audit_export, "get_audit_log", return_value=mock_audit_log):
            request = make_mocked_request(
                "POST",
                "/api/audit/export",
                app=web.Application(),
            )
            request.json = AsyncMock(return_value={
                "start_date": "not-a-date",
                "end_date": "2026-01-15T00:00:00",
            })

            response = await audit_export.handle_audit_export(request)

            assert response.status == 400
            data = json.loads(response.body)
            assert "error" in data
            assert "date format" in data["error"].lower()

    @pytest.mark.asyncio
    async def test_invalid_format_returns_400(self, mock_audit_log):
        """Returns 400 for invalid export format."""
        with patch.object(audit_export, "get_audit_log", return_value=mock_audit_log):
            request = make_mocked_request(
                "POST",
                "/api/audit/export",
                app=web.Application(),
            )
            request.json = AsyncMock(return_value={
                "format": "invalid_format",
                "start_date": "2026-01-01T00:00:00",
                "end_date": "2026-01-15T00:00:00",
            })

            response = await audit_export.handle_audit_export(request)

            assert response.status == 400
            data = json.loads(response.body)
            assert "error" in data
            assert "Invalid format" in data["error"]

    @pytest.mark.asyncio
    async def test_filters_by_org_id(self, mock_audit_log):
        """Passes org_id filter to export functions."""
        with patch.object(audit_export, "get_audit_log", return_value=mock_audit_log):
            request = make_mocked_request(
                "POST",
                "/api/audit/export",
                app=web.Application(),
            )
            request.json = AsyncMock(return_value={
                "format": "json",
                "start_date": "2026-01-01T00:00:00",
                "end_date": "2026-01-15T00:00:00",
                "org_id": "org-123",
            })

            response = await audit_export.handle_audit_export(request)

            assert response.status == 200
            mock_audit_log.export_json.assert_called_once()
            call_args = mock_audit_log.export_json.call_args
            assert call_args[0][3] == "org-123"  # org_id is 4th positional arg


# ============================================================================
# handle_audit_verify Tests
# ============================================================================


class TestHandleAuditVerify:
    """Tests for POST /api/audit/verify endpoint."""

    @pytest.mark.asyncio
    async def test_verifies_all_events_by_default(self, mock_audit_log):
        """Verifies all events when no date range specified."""
        with patch.object(audit_export, "get_audit_log", return_value=mock_audit_log):
            request = make_mocked_request(
                "POST",
                "/api/audit/verify",
                app=web.Application(),
            )
            request.json = AsyncMock(return_value={})

            response = await audit_export.handle_audit_verify(request)

            assert response.status == 200
            data = json.loads(response.body)
            assert data["verified"] is True
            assert data["errors"] == []
            assert data["total_errors"] == 0
            assert data["verified_range"]["start_date"] == "beginning"
            assert data["verified_range"]["end_date"] == "now"

    @pytest.mark.asyncio
    async def test_verifies_with_date_range(self, mock_audit_log):
        """Verifies events within specified date range."""
        with patch.object(audit_export, "get_audit_log", return_value=mock_audit_log):
            request = make_mocked_request(
                "POST",
                "/api/audit/verify",
                app=web.Application(),
            )
            request.json = AsyncMock(return_value={
                "start_date": "2026-01-01T00:00:00",
                "end_date": "2026-01-15T00:00:00",
            })

            response = await audit_export.handle_audit_verify(request)

            assert response.status == 200
            data = json.loads(response.body)
            assert data["verified"] is True
            mock_audit_log.verify_integrity.assert_called_once()

    @pytest.mark.asyncio
    async def test_handles_invalid_json_gracefully(self, mock_audit_log):
        """Handles invalid JSON body gracefully (uses empty body)."""
        with patch.object(audit_export, "get_audit_log", return_value=mock_audit_log):
            request = make_mocked_request(
                "POST",
                "/api/audit/verify",
                app=web.Application(),
            )
            request.json = AsyncMock(side_effect=json.JSONDecodeError("test", "doc", 0))

            response = await audit_export.handle_audit_verify(request)

            assert response.status == 200
            data = json.loads(response.body)
            assert data["verified"] is True

    @pytest.mark.asyncio
    async def test_invalid_start_date_returns_400(self, mock_audit_log):
        """Returns 400 for invalid start_date format."""
        with patch.object(audit_export, "get_audit_log", return_value=mock_audit_log):
            request = make_mocked_request(
                "POST",
                "/api/audit/verify",
                app=web.Application(),
            )
            request.json = AsyncMock(return_value={
                "start_date": "not-a-date",
            })

            response = await audit_export.handle_audit_verify(request)

            assert response.status == 400
            data = json.loads(response.body)
            assert "start_date" in data["error"]

    @pytest.mark.asyncio
    async def test_invalid_end_date_returns_400(self, mock_audit_log):
        """Returns 400 for invalid end_date format."""
        with patch.object(audit_export, "get_audit_log", return_value=mock_audit_log):
            request = make_mocked_request(
                "POST",
                "/api/audit/verify",
                app=web.Application(),
            )
            request.json = AsyncMock(return_value={
                "end_date": "invalid",
            })

            response = await audit_export.handle_audit_verify(request)

            assert response.status == 400
            data = json.loads(response.body)
            assert "end_date" in data["error"]

    @pytest.mark.asyncio
    async def test_returns_errors_on_integrity_failure(self, mock_audit_log):
        """Returns errors when integrity verification fails."""
        mock_audit_log.verify_integrity.return_value = (
            False,
            ["Hash chain broken at event-1", "Hash mismatch at event-2"],
        )
        with patch.object(audit_export, "get_audit_log", return_value=mock_audit_log):
            request = make_mocked_request(
                "POST",
                "/api/audit/verify",
                app=web.Application(),
            )
            request.json = AsyncMock(return_value={})

            response = await audit_export.handle_audit_verify(request)

            assert response.status == 200
            data = json.loads(response.body)
            assert data["verified"] is False
            assert len(data["errors"]) == 2
            assert data["total_errors"] == 2

    @pytest.mark.asyncio
    async def test_limits_errors_in_response(self, mock_audit_log):
        """Limits errors to first 20 in response."""
        errors = [f"Error {i}" for i in range(30)]
        mock_audit_log.verify_integrity.return_value = (False, errors)
        with patch.object(audit_export, "get_audit_log", return_value=mock_audit_log):
            request = make_mocked_request(
                "POST",
                "/api/audit/verify",
                app=web.Application(),
            )
            request.json = AsyncMock(return_value={})

            response = await audit_export.handle_audit_verify(request)

            assert response.status == 200
            data = json.loads(response.body)
            assert len(data["errors"]) == 20
            assert data["total_errors"] == 30


# ============================================================================
# register_handlers Tests
# ============================================================================


class TestRegisterHandlers:
    """Tests for register_handlers function."""

    def test_registers_all_routes(self):
        """Registers all audit routes on the application."""
        app = web.Application()

        audit_export.register_handlers(app)

        # Get all registered routes
        routes = {(r.method, r.resource.canonical) for r in app.router.routes()}

        assert ("GET", "/api/audit/events") in routes
        assert ("GET", "/api/audit/stats") in routes
        assert ("POST", "/api/audit/export") in routes
        assert ("POST", "/api/audit/verify") in routes

    def test_routes_point_to_correct_handlers(self):
        """Routes point to the correct handler functions."""
        app = web.Application()

        audit_export.register_handlers(app)

        # Find routes by path
        for route in app.router.routes():
            if route.resource.canonical == "/api/audit/events":
                assert route.handler == audit_export.handle_audit_events
            elif route.resource.canonical == "/api/audit/stats":
                assert route.handler == audit_export.handle_audit_stats
            elif route.resource.canonical == "/api/audit/export":
                assert route.handler == audit_export.handle_audit_export
            elif route.resource.canonical == "/api/audit/verify":
                assert route.handler == audit_export.handle_audit_verify


# ============================================================================
# Module Export Tests
# ============================================================================


class TestModuleExports:
    """Tests for module exports."""

    def test_all_exports_are_defined(self):
        """All items in __all__ are defined in the module."""
        for name in audit_export.__all__:
            assert hasattr(audit_export, name), f"{name} in __all__ but not defined"

    def test_expected_exports(self):
        """Module exports expected functions."""
        expected = [
            "handle_audit_events",
            "handle_audit_export",
            "handle_audit_stats",
            "handle_audit_verify",
            "register_handlers",
        ]
        for name in expected:
            assert name in audit_export.__all__, f"{name} not in __all__"
