"""Tests for Backup Offsite handler.

Covers all routes and behaviour of the BackupOffsiteHandler class:
- GET  /api/v1/backup/status  - Current backup status
- GET  /api/v1/backup/drills  - List restore drill results
- POST /api/v1/backup/drill   - Trigger a manual restore drill

SOC 2 Compliance: CC9.1, CC9.2 (Business Continuity)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.backup_offsite_handler import (
    BackupOffsiteHandler,
    create_backup_offsite_handler,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    if result is None:
        return {}
    if isinstance(result, dict):
        return result
    return json.loads(result.body)


def _status(result) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if result is None:
        return 0
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


def _data(result) -> dict:
    """Extract the 'data' envelope from a response."""
    body = _body(result)
    if isinstance(body, dict) and "data" in body:
        return body["data"]
    return body


# ---------------------------------------------------------------------------
# Mock objects
# ---------------------------------------------------------------------------


@dataclass
class MockDrillReport:
    """Mock restore drill report."""

    drill_id: str = "drill-001"
    backup_id: str = "backup-001"
    status: str = "passed"
    started_at: str = "2026-02-24T10:00:00Z"
    completed_at: str = "2026-02-24T10:05:00Z"
    duration_seconds: float = 300.0
    tables_restored: int = 15
    records_verified: int = 5000
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "drill_id": self.drill_id,
            "backup_id": self.backup_id,
            "status": self.status,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_seconds": self.duration_seconds,
            "tables_restored": self.tables_restored,
            "records_verified": self.records_verified,
            "errors": self.errors,
        }


class MockBackupManager:
    """Mock backup manager."""

    def __init__(self):
        self.get_backup_status = MagicMock(return_value={
            "last_backup": "2026-02-24T09:00:00Z",
            "total_backups": 42,
            "total_size_bytes": 1073741824,
            "latest_drill": {
                "status": "passed",
                "timestamp": "2026-02-24T10:05:00Z",
            },
        })
        self.get_drill_history = MagicMock(return_value=[
            MockDrillReport(drill_id="drill-001"),
            MockDrillReport(drill_id="drill-002", status="failed", errors=["Checksum mismatch"]),
        ])
        self.restore_drill = MagicMock(return_value=MockDrillReport())


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_manager():
    """Create a mock backup manager."""
    return MockBackupManager()


@pytest.fixture
def handler(mock_manager):
    """Create a BackupOffsiteHandler with a mock manager."""
    h = BackupOffsiteHandler(server_context={})
    h._manager = mock_manager
    return h


@pytest.fixture
def mock_http_handler():
    """Create a mock HTTP handler."""
    def _make(method: str = "GET", body: dict | None = None):
        h = MagicMock()
        h.command = method
        if body:
            h.rfile.read.return_value = json.dumps(body).encode("utf-8")
            h.headers = {"Content-Length": str(len(json.dumps(body)))}
        else:
            h.rfile.read.return_value = b"{}"
            h.headers = {"Content-Length": "2"}
        return h
    return _make


# ---------------------------------------------------------------------------
# ROUTES and can_handle
# ---------------------------------------------------------------------------


class TestRoutes:
    """Test ROUTES class attribute and can_handle."""

    def test_routes_contains_all_endpoints(self):
        expected = [
            "/api/v1/backup/status",
            "/api/v1/backup/drills",
            "/api/v1/backup/drill",
        ]
        for route in expected:
            assert route in BackupOffsiteHandler.ROUTES, f"Missing route: {route}"

    def test_can_handle_get_status(self, handler):
        assert handler.can_handle("/api/v1/backup/status", method="GET")

    def test_can_handle_get_drills(self, handler):
        assert handler.can_handle("/api/v1/backup/drills", method="GET")

    def test_can_handle_post_drill(self, handler):
        assert handler.can_handle("/api/v1/backup/drill", method="POST")

    def test_can_handle_rejects_wrong_method(self, handler):
        assert not handler.can_handle("/api/v1/backup/status", method="POST")
        assert not handler.can_handle("/api/v1/backup/drill", method="GET")
        assert not handler.can_handle("/api/v1/backup/drills", method="POST")

    def test_can_handle_rejects_unknown_paths(self, handler):
        assert not handler.can_handle("/api/v1/backup/other")
        assert not handler.can_handle("/api/v1/other")
        assert not handler.can_handle("/api/v1/backup/status", method="DELETE")


# ---------------------------------------------------------------------------
# GET /api/v1/backup/status
# ---------------------------------------------------------------------------


class TestGetStatus:
    """Test the backup status endpoint."""

    @pytest.mark.asyncio
    async def test_get_status_returns_data(self, handler, mock_manager):
        result = await handler._get_status()

        body = _body(result)
        data = body.get("data", body)
        assert "last_backup" in data
        assert "total_backups" in data
        assert data["total_backups"] == 42
        assert _status(result) == 200

        mock_manager.get_backup_status.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_status_via_handle(self, handler, mock_http_handler):
        h = mock_http_handler(method="GET")
        result = await handler.handle("/api/v1/backup/status", {}, h)

        assert _status(result) == 200
        body = _body(result)
        data = body.get("data", body)
        assert "last_backup" in data


# ---------------------------------------------------------------------------
# GET /api/v1/backup/drills
# ---------------------------------------------------------------------------


class TestListDrills:
    """Test the drills listing endpoint."""

    @pytest.mark.asyncio
    async def test_list_drills_returns_data(self, handler, mock_manager):
        result = await handler._list_drills({"limit": "50"})

        body = _body(result)
        data = body.get("data", body)
        assert "drills" in data
        assert "total" in data
        assert data["total"] == 2
        assert len(data["drills"]) == 2
        assert _status(result) == 200

        mock_manager.get_drill_history.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_drills_default_limit(self, handler, mock_manager):
        result = await handler._list_drills({})

        # safe_query_int should default to 50
        mock_manager.get_drill_history.assert_called_once_with(limit=50)

    @pytest.mark.asyncio
    async def test_list_drills_custom_limit(self, handler, mock_manager):
        result = await handler._list_drills({"limit": "10"})

        mock_manager.get_drill_history.assert_called_once_with(limit=10)

    @pytest.mark.asyncio
    async def test_list_drills_drill_entries_have_expected_keys(self, handler, mock_manager):
        result = await handler._list_drills({})

        data = _data(result)
        drill = data["drills"][0]
        assert "drill_id" in drill
        assert "backup_id" in drill
        assert "status" in drill

    @pytest.mark.asyncio
    async def test_list_drills_via_handle(self, handler, mock_http_handler):
        h = mock_http_handler(method="GET")
        result = await handler.handle("/api/v1/backup/drills", {}, h)

        assert _status(result) == 200
        data = _data(result)
        assert "drills" in data


# ---------------------------------------------------------------------------
# POST /api/v1/backup/drill
# ---------------------------------------------------------------------------


class TestTriggerDrill:
    """Test the trigger drill endpoint."""

    @pytest.mark.asyncio
    async def test_trigger_drill_success_returns_201(self, handler, mock_manager):
        result = await handler._trigger_drill({})

        body = _body(result)
        data = body.get("data", body)
        assert "drill_id" in data
        assert data["status"] == "passed"
        assert _status(result) == 201

        mock_manager.restore_drill.assert_called_once_with(backup_id=None)

    @pytest.mark.asyncio
    async def test_trigger_drill_with_backup_id(self, handler, mock_manager):
        result = await handler._trigger_drill({"backup_id": "backup-123"})

        mock_manager.restore_drill.assert_called_once_with(backup_id="backup-123")

    @pytest.mark.asyncio
    async def test_trigger_drill_failed_returns_200(self, handler, mock_manager):
        mock_manager.restore_drill.return_value = MockDrillReport(
            status="failed",
            errors=["Integrity check failed"],
        )

        result = await handler._trigger_drill({})

        data = _data(result)
        assert data["status"] == "failed"
        # Failed drill returns 200, not 201
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_trigger_drill_report_includes_errors(self, handler, mock_manager):
        mock_manager.restore_drill.return_value = MockDrillReport(
            status="failed",
            errors=["Table users: row count mismatch"],
        )

        result = await handler._trigger_drill({})

        data = _data(result)
        assert "errors" in data
        assert len(data["errors"]) == 1


# ---------------------------------------------------------------------------
# handle() routing
# ---------------------------------------------------------------------------


class TestHandleRouting:
    """Test the main handle method routing."""

    @pytest.mark.asyncio
    async def test_handle_routes_to_status(self, handler, mock_http_handler, mock_manager):
        h = mock_http_handler(method="GET")
        result = await handler.handle("/api/v1/backup/status", {}, h)

        assert _status(result) == 200
        mock_manager.get_backup_status.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_routes_to_drills(self, handler, mock_http_handler, mock_manager):
        h = mock_http_handler(method="GET")
        result = await handler.handle("/api/v1/backup/drills", {}, h)

        assert _status(result) == 200
        mock_manager.get_drill_history.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_routes_to_trigger_drill(self, handler, mock_http_handler, mock_manager):
        h = mock_http_handler(method="POST", body={})
        result = await handler.handle("/api/v1/backup/drill", {}, h)

        assert _status(result) in (200, 201)
        mock_manager.restore_drill.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_unknown_path_returns_404(self, handler, mock_http_handler):
        h = mock_http_handler(method="GET")
        result = await handler.handle("/api/v1/backup/unknown", {}, h)

        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_handle_none_handler(self, handler):
        result = await handler.handle("/api/v1/backup/status", {}, None)

        # Should still work - handler defaults to GET
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_handle_none_query_params(self, handler, mock_http_handler):
        h = mock_http_handler(method="GET")
        result = await handler.handle("/api/v1/backup/drills", None, h)

        assert _status(result) == 200


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Test error handling within the handler."""

    @pytest.mark.asyncio
    async def test_get_status_handles_manager_error(self, handler, mock_manager):
        mock_manager.get_backup_status.side_effect = RuntimeError("Connection lost")

        result = await handler.handle(
            "/api/v1/backup/status", {}, MagicMock(command="GET")
        )

        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_list_drills_handles_manager_error(self, handler, mock_manager):
        mock_manager.get_drill_history.side_effect = OSError("Disk error")

        result = await handler.handle(
            "/api/v1/backup/drills", {}, MagicMock(command="GET")
        )

        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_trigger_drill_handles_value_error(self, handler, mock_manager):
        mock_manager.restore_drill.side_effect = ValueError("Invalid backup ID")

        result = await handler.handle(
            "/api/v1/backup/drill", {}, MagicMock(command="POST")
        )

        # ValueError inside _trigger_drill is caught by @handle_errors (returns 400)
        # or by the outer handle() exception handler (returns 500)
        assert _status(result) in (400, 500)


# ---------------------------------------------------------------------------
# Lazy manager initialization
# ---------------------------------------------------------------------------


class TestLazyManager:
    """Test the lazy manager initialization."""

    def test_manager_factory_used_when_none(self):
        h = BackupOffsiteHandler(server_context={})
        assert h._manager is None

        mock_factory = MagicMock()
        mock_factory.get.return_value = MagicMock()
        h._manager_factory = mock_factory

        manager = h._get_manager()
        mock_factory.get.assert_called_once()
        assert manager is not None

    def test_manager_cached_after_first_call(self):
        h = BackupOffsiteHandler(server_context={})
        mock_mgr = MagicMock()
        h._manager = mock_mgr

        assert h._get_manager() is mock_mgr


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


class TestFactory:
    """Test the create_backup_offsite_handler factory function."""

    def test_factory_creates_handler(self):
        handler = create_backup_offsite_handler({"key": "value"})
        assert isinstance(handler, BackupOffsiteHandler)

    def test_factory_passes_context(self):
        ctx = {"storage": MagicMock()}
        handler = create_backup_offsite_handler(ctx)
        # BaseHandler stores server_context as self.ctx
        assert handler.ctx is ctx


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------


class TestInit:
    """Test handler initialization."""

    def test_init_with_server_context(self):
        ctx = {"key": "value"}
        handler = BackupOffsiteHandler(server_context=ctx)
        assert handler._manager is None
