"""Tests for SystemHandler in aragora/server/handlers/admin/system.py.

Comprehensive coverage of all endpoints:
- GET /api/debug/test            (_handle_debug_test)
- GET /api/history/cycles        (_get_history_cycles)
- GET /api/history/events        (_get_history_events)
- GET /api/history/debates       (_get_history_debates)
- GET /api/history/summary       (_get_history_summary)
- GET /api/system/maintenance    (_handle_maintenance)
- GET /api/auth/stats            (_get_auth_stats)
- POST /api/auth/revoke          (_revoke_token)
- GET /api/circuit-breakers      (_get_circuit_breaker_metrics)
- GET /metrics                   (_get_prometheus_metrics)
- GET /api/v1/diagnostics/handlers (_get_handler_diagnostics)

Also covers:
- Routing via handle() and handle_post()
- Version prefix stripping (/api/v1/... -> /api/...)
- can_handle() path matching
- History auth checks
- loop_id validation
- Maintenance task validation
- Error handling for ImportError, StorageError, OSError, etc.
- TTL cache interaction
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.admin.cache import clear_cache
from aragora.server.handlers.admin.system import SystemHandler
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


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture(autouse=True)
def _clear_ttl_cache():
    """Clear TTL cache before every test to avoid cross-test pollution."""
    clear_cache()
    yield
    clear_cache()


@pytest.fixture
def handler():
    """A SystemHandler with empty context."""
    return SystemHandler(ctx={})


@pytest.fixture
def mock_http():
    """Minimal mock HTTP handler (simulates BaseHTTPRequestHandler)."""
    h = MagicMock()
    h.path = "/api/debug/test"
    h.command = "GET"
    h.headers = {"Content-Type": "application/json", "Content-Length": "0"}
    h.client_address = ("127.0.0.1", 12345)
    h.rfile = MagicMock()
    h.rfile.read.return_value = b"{}"
    return h


def _make_http_handler(body: dict | None = None) -> MagicMock:
    """Create a mock HTTP handler with optional JSON body."""
    h = MagicMock()
    h.command = "GET"
    h.client_address = ("127.0.0.1", 12345)
    if body is not None:
        body_bytes = json.dumps(body).encode()
        h.rfile.read.return_value = body_bytes
        h.headers = {
            "Content-Type": "application/json",
            "Content-Length": str(len(body_bytes)),
        }
    else:
        h.rfile.read.return_value = b"{}"
        h.headers = {"Content-Type": "application/json", "Content-Length": "0"}
    return h


# ===========================================================================
# Tests: can_handle
# ===========================================================================


class TestCanHandle:
    """Tests for can_handle path matching."""

    def test_can_handle_debug_test(self, handler):
        assert handler.can_handle("/api/debug/test") is True

    def test_can_handle_history_cycles(self, handler):
        assert handler.can_handle("/api/history/cycles") is True

    def test_can_handle_history_events(self, handler):
        assert handler.can_handle("/api/history/events") is True

    def test_can_handle_history_debates(self, handler):
        assert handler.can_handle("/api/history/debates") is True

    def test_can_handle_history_summary(self, handler):
        assert handler.can_handle("/api/history/summary") is True

    def test_can_handle_system_maintenance(self, handler):
        assert handler.can_handle("/api/system/maintenance") is True

    def test_can_handle_auth_stats(self, handler):
        assert handler.can_handle("/api/auth/stats") is True

    def test_can_handle_auth_revoke(self, handler):
        assert handler.can_handle("/api/auth/revoke") is True

    def test_can_handle_circuit_breakers(self, handler):
        assert handler.can_handle("/api/circuit-breakers") is True

    def test_can_handle_metrics(self, handler):
        assert handler.can_handle("/metrics") is True

    def test_can_handle_diagnostics_in_routes(self, handler):
        """Diagnostics path is in ROUTES list."""
        assert "/api/v1/diagnostics/handlers" in SystemHandler.ROUTES

    def test_can_handle_with_version_prefix(self, handler):
        """Version prefix is stripped, so /api/v1/debug/test matches /api/debug/test."""
        assert handler.can_handle("/api/v1/debug/test") is True

    def test_can_handle_unknown_path(self, handler):
        assert handler.can_handle("/api/unknown/path") is False

    def test_can_handle_empty_path(self, handler):
        assert handler.can_handle("") is False

    def test_can_handle_partial_match(self, handler):
        assert handler.can_handle("/api/debug") is False


# ===========================================================================
# Tests: handle() routing - debug test
# ===========================================================================


class TestDebugTest:
    """Tests for the /api/debug/test endpoint."""

    def test_debug_test_returns_ok(self, handler, mock_http):
        result = handler.handle("/api/debug/test", {}, mock_http)
        assert _status(result) == 200
        body = _body(result)
        assert body["status"] == "ok"
        assert body["message"] == "Modular handler works"

    def test_debug_test_includes_method(self, handler, mock_http):
        mock_http.command = "GET"
        result = handler.handle("/api/debug/test", {}, mock_http)
        body = _body(result)
        assert body["method"] == "GET"

    def test_debug_test_with_version_prefix(self, handler, mock_http):
        result = handler.handle("/api/v1/debug/test", {}, mock_http)
        assert _status(result) == 200
        body = _body(result)
        assert body["status"] == "ok"

    def test_debug_test_handler_without_command(self, handler):
        """When handler has no command attr, defaults to GET."""
        h = MagicMock(spec=[])  # No attributes
        result = handler.handle("/api/debug/test", {}, h)
        assert _status(result) == 200
        body = _body(result)
        assert body["method"] == "GET"


# ===========================================================================
# Tests: handle() routing - maintenance
# ===========================================================================


class TestMaintenance:
    """Tests for the /api/system/maintenance endpoint."""

    def test_maintenance_invalid_task(self, handler, mock_http):
        result = handler.handle("/api/system/maintenance", {"task": "invalid"}, mock_http)
        assert _status(result) == 400
        body = _body(result)
        assert "Invalid task" in body["error"]

    def test_maintenance_default_task_is_status(self, handler, mock_http):
        """When no task param given, defaults to 'status'."""
        with patch.object(handler, "get_nomic_dir", return_value=Path("/tmp/test_nomic")):
            with patch(
                "aragora.maintenance.DatabaseMaintenance"
            ) as mock_maint:
                instance = MagicMock()
                instance.get_stats.return_value = {"db_count": 2}
                mock_maint.return_value = instance
                result = handler.handle("/api/system/maintenance", {}, mock_http)
                assert _status(result) == 200
                body = _body(result)
                assert body["task"] == "status"
                assert body["stats"] == {"db_count": 2}

    def test_maintenance_status_task(self, handler, mock_http):
        with patch.object(handler, "get_nomic_dir", return_value=Path("/tmp/nomic")):
            with patch(
                "aragora.maintenance.DatabaseMaintenance"
            ) as mock_cls:
                inst = MagicMock()
                inst.get_stats.return_value = {"total": 5}
                mock_cls.return_value = inst
                result = handler.handle(
                    "/api/system/maintenance", {"task": "status"}, mock_http
                )
                assert _status(result) == 200
                assert _body(result)["task"] == "status"

    def test_maintenance_vacuum_task(self, handler, mock_http):
        with patch.object(handler, "get_nomic_dir", return_value=Path("/tmp/nomic")):
            with patch(
                "aragora.maintenance.DatabaseMaintenance"
            ) as mock_cls:
                inst = MagicMock()
                inst.vacuum_all.return_value = {"freed": 100}
                inst.get_stats.return_value = {"total": 3}
                mock_cls.return_value = inst
                result = handler.handle(
                    "/api/system/maintenance", {"task": "vacuum"}, mock_http
                )
                assert _status(result) == 200
                body = _body(result)
                assert body["task"] == "vacuum"
                assert body["vacuum"] == {"freed": 100}
                assert "stats" in body

    def test_maintenance_analyze_task(self, handler, mock_http):
        with patch.object(handler, "get_nomic_dir", return_value=Path("/tmp/nomic")):
            with patch(
                "aragora.maintenance.DatabaseMaintenance"
            ) as mock_cls:
                inst = MagicMock()
                inst.analyze_all.return_value = {"tables": 5}
                inst.get_stats.return_value = {}
                mock_cls.return_value = inst
                result = handler.handle(
                    "/api/system/maintenance", {"task": "analyze"}, mock_http
                )
                assert _status(result) == 200
                body = _body(result)
                assert body["task"] == "analyze"
                assert body["analyze"] == {"tables": 5}

    def test_maintenance_checkpoint_task(self, handler, mock_http):
        with patch.object(handler, "get_nomic_dir", return_value=Path("/tmp/nomic")):
            with patch(
                "aragora.maintenance.DatabaseMaintenance"
            ) as mock_cls:
                inst = MagicMock()
                inst.checkpoint_all_wal.return_value = {"wal_size": 0}
                inst.get_stats.return_value = {}
                mock_cls.return_value = inst
                result = handler.handle(
                    "/api/system/maintenance", {"task": "checkpoint"}, mock_http
                )
                assert _status(result) == 200
                body = _body(result)
                assert body["task"] == "checkpoint"
                assert "checkpoint" in body

    def test_maintenance_full_task(self, handler, mock_http):
        with patch.object(handler, "get_nomic_dir", return_value=Path("/tmp/nomic")):
            with patch(
                "aragora.maintenance.DatabaseMaintenance"
            ) as mock_cls:
                inst = MagicMock()
                inst.checkpoint_all_wal.return_value = {"wal": True}
                inst.analyze_all.return_value = {"analyze": True}
                inst.vacuum_all.return_value = {"vacuum": True}
                inst.get_stats.return_value = {"all": True}
                mock_cls.return_value = inst
                result = handler.handle(
                    "/api/system/maintenance", {"task": "full"}, mock_http
                )
                assert _status(result) == 200
                body = _body(result)
                assert body["task"] == "full"
                assert "checkpoint" in body
                assert "analyze" in body
                assert "vacuum" in body
                assert "stats" in body

    def test_maintenance_no_nomic_dir(self, handler, mock_http):
        with patch.object(handler, "get_nomic_dir", return_value=None):
            result = handler.handle(
                "/api/system/maintenance", {"task": "status"}, mock_http
            )
            assert _status(result) == 503
            assert "not configured" in _body(result)["error"]

    def test_maintenance_import_error(self, handler, mock_http):
        with patch.object(handler, "get_nomic_dir", return_value=Path("/tmp/nomic")):
            with patch.dict(
                "sys.modules", {"aragora.maintenance": None}
            ):
                result = handler.handle(
                    "/api/system/maintenance", {"task": "status"}, mock_http
                )
                assert _status(result) == 503
                assert "not available" in _body(result)["error"]

    def test_maintenance_os_error(self, handler, mock_http):
        with patch.object(handler, "get_nomic_dir", return_value=Path("/tmp/nomic")):
            with patch(
                "aragora.maintenance.DatabaseMaintenance",
                side_effect=OSError("disk full"),
            ):
                result = handler.handle(
                    "/api/system/maintenance", {"task": "status"}, mock_http
                )
                assert _status(result) == 500

    def test_maintenance_storage_error(self, handler, mock_http):
        from aragora.exceptions import StorageError

        with patch.object(handler, "get_nomic_dir", return_value=Path("/tmp/nomic")):
            with patch(
                "aragora.maintenance.DatabaseMaintenance",
                side_effect=StorageError("corrupt"),
            ):
                result = handler.handle(
                    "/api/system/maintenance", {"task": "status"}, mock_http
                )
                assert _status(result) == 500
                assert "Database error" in _body(result)["error"]

    def test_maintenance_database_error(self, handler, mock_http):
        from aragora.exceptions import DatabaseError

        with patch.object(handler, "get_nomic_dir", return_value=Path("/tmp/nomic")):
            with patch(
                "aragora.maintenance.DatabaseMaintenance",
                side_effect=DatabaseError("lock"),
            ):
                result = handler.handle(
                    "/api/system/maintenance", {"task": "status"}, mock_http
                )
                assert _status(result) == 500
                assert "Database error" in _body(result)["error"]

    def test_maintenance_value_error(self, handler, mock_http):
        with patch.object(handler, "get_nomic_dir", return_value=Path("/tmp/nomic")):
            with patch(
                "aragora.maintenance.DatabaseMaintenance",
                side_effect=ValueError("bad value"),
            ):
                result = handler.handle(
                    "/api/system/maintenance", {"task": "status"}, mock_http
                )
                assert _status(result) == 500

    def test_maintenance_valid_tasks_enumerated(self, handler, mock_http):
        """Error message for invalid task includes all valid options."""
        result = handler.handle(
            "/api/system/maintenance", {"task": "drop_tables"}, mock_http
        )
        body = _body(result)
        for task_name in ("status", "vacuum", "analyze", "checkpoint", "full"):
            assert task_name in body["error"]

    def test_maintenance_runtime_error(self, handler, mock_http):
        """RuntimeError during maintenance is caught."""
        with patch.object(handler, "get_nomic_dir", return_value=Path("/tmp/nomic")):
            with patch(
                "aragora.maintenance.DatabaseMaintenance",
                side_effect=RuntimeError("unexpected"),
            ):
                result = handler.handle(
                    "/api/system/maintenance", {"task": "status"}, mock_http
                )
                assert _status(result) == 500


# ===========================================================================
# Tests: History endpoints
# ===========================================================================


class TestHistoryCycles:
    """Tests for /api/history/cycles endpoint."""

    def test_cycles_no_nomic_dir(self, handler, mock_http):
        with patch.object(handler, "get_nomic_dir", return_value=None):
            result = handler.handle("/api/history/cycles", {}, mock_http)
            assert _status(result) == 200
            assert _body(result)["cycles"] == []

    def test_cycles_file_not_exists(self, handler, mock_http, tmp_path):
        with patch.object(handler, "get_nomic_dir", return_value=tmp_path):
            result = handler.handle("/api/history/cycles", {}, mock_http)
            assert _status(result) == 200
            assert _body(result)["cycles"] == []

    def test_cycles_with_data(self, handler, mock_http, tmp_path):
        cycles = [{"loop_id": "loop1", "phase": "debate"}, {"loop_id": "loop2", "phase": "verify"}]
        (tmp_path / "cycles.json").write_text(json.dumps(cycles))
        with patch.object(handler, "get_nomic_dir", return_value=tmp_path):
            result = handler.handle("/api/history/cycles", {}, mock_http)
            assert _status(result) == 200
            body = _body(result)
            assert len(body["cycles"]) == 2

    def test_cycles_with_limit(self, handler, mock_http, tmp_path):
        cycles = [{"loop_id": f"l{i}"} for i in range(100)]
        (tmp_path / "cycles.json").write_text(json.dumps(cycles))
        with patch.object(handler, "get_nomic_dir", return_value=tmp_path):
            result = handler.handle("/api/history/cycles", {"limit": "10"}, mock_http)
            assert _status(result) == 200
            body = _body(result)
            assert len(body["cycles"]) == 10

    def test_cycles_with_loop_id_filter(self, handler, mock_http, tmp_path):
        cycles = [
            {"loop_id": "target", "data": "a"},
            {"loop_id": "other", "data": "b"},
            {"loop_id": "target", "data": "c"},
        ]
        (tmp_path / "cycles.json").write_text(json.dumps(cycles))
        with patch.object(handler, "get_nomic_dir", return_value=tmp_path):
            result = handler.handle(
                "/api/history/cycles", {"loop_id": "target"}, mock_http
            )
            assert _status(result) == 200
            body = _body(result)
            assert len(body["cycles"]) == 2
            assert all(c["loop_id"] == "target" for c in body["cycles"])

    def test_cycles_invalid_loop_id(self, handler, mock_http):
        """loop_id with special characters is rejected."""
        result = handler.handle(
            "/api/history/cycles", {"loop_id": "../../etc/passwd"}, mock_http
        )
        assert _status(result) == 400

    def test_cycles_default_limit_is_50(self, handler, mock_http, tmp_path):
        cycles = [{"loop_id": f"l{i}"} for i in range(100)]
        (tmp_path / "cycles.json").write_text(json.dumps(cycles))
        with patch.object(handler, "get_nomic_dir", return_value=tmp_path):
            result = handler.handle("/api/history/cycles", {}, mock_http)
            body = _body(result)
            assert len(body["cycles"]) == 50

    def test_cycles_max_limit_is_200(self, handler, mock_http, tmp_path):
        cycles = [{"loop_id": f"l{i}"} for i in range(300)]
        (tmp_path / "cycles.json").write_text(json.dumps(cycles))
        with patch.object(handler, "get_nomic_dir", return_value=tmp_path):
            result = handler.handle(
                "/api/history/cycles", {"limit": "999"}, mock_http
            )
            body = _body(result)
            # Clamped to max 200
            assert len(body["cycles"]) == 200

    def test_cycles_with_version_prefix(self, handler, mock_http, tmp_path):
        """History cycles route works via /api/v1/history/cycles."""
        cycles = [{"loop_id": "l1"}]
        (tmp_path / "cycles.json").write_text(json.dumps(cycles))
        with patch.object(handler, "get_nomic_dir", return_value=tmp_path):
            result = handler.handle("/api/v1/history/cycles", {}, mock_http)
            assert _status(result) == 200
            assert len(_body(result)["cycles"]) == 1


class TestHistoryEvents:
    """Tests for /api/history/events endpoint."""

    def test_events_no_nomic_dir(self, handler, mock_http):
        with patch.object(handler, "get_nomic_dir", return_value=None):
            result = handler.handle("/api/history/events", {}, mock_http)
            assert _status(result) == 200
            assert _body(result)["events"] == []

    def test_events_with_data(self, handler, mock_http, tmp_path):
        events = [{"type": "start"}, {"type": "end"}]
        (tmp_path / "events.json").write_text(json.dumps(events))
        with patch.object(handler, "get_nomic_dir", return_value=tmp_path):
            result = handler.handle("/api/history/events", {}, mock_http)
            assert _status(result) == 200
            assert len(_body(result)["events"]) == 2

    def test_events_default_limit_is_100(self, handler, mock_http, tmp_path):
        events = [{"type": f"e{i}"} for i in range(200)]
        (tmp_path / "events.json").write_text(json.dumps(events))
        with patch.object(handler, "get_nomic_dir", return_value=tmp_path):
            result = handler.handle("/api/history/events", {}, mock_http)
            assert len(_body(result)["events"]) == 100

    def test_events_max_limit_is_500(self, handler, mock_http, tmp_path):
        events = [{"type": f"e{i}"} for i in range(600)]
        (tmp_path / "events.json").write_text(json.dumps(events))
        with patch.object(handler, "get_nomic_dir", return_value=tmp_path):
            result = handler.handle(
                "/api/history/events", {"limit": "9999"}, mock_http
            )
            assert len(_body(result)["events"]) == 500

    def test_events_with_loop_id_filter(self, handler, mock_http, tmp_path):
        events = [
            {"loop_id": "a", "val": 1},
            {"loop_id": "b", "val": 2},
            {"loop_id": "a", "val": 3},
        ]
        (tmp_path / "events.json").write_text(json.dumps(events))
        with patch.object(handler, "get_nomic_dir", return_value=tmp_path):
            result = handler.handle(
                "/api/history/events", {"loop_id": "a"}, mock_http
            )
            body = _body(result)
            assert len(body["events"]) == 2

    def test_events_file_not_exists(self, handler, mock_http, tmp_path):
        with patch.object(handler, "get_nomic_dir", return_value=tmp_path):
            result = handler.handle("/api/history/events", {}, mock_http)
            assert _status(result) == 200
            assert _body(result)["events"] == []


class TestHistoryDebates:
    """Tests for /api/history/debates endpoint."""

    def test_debates_no_storage(self, handler, mock_http):
        result = handler.handle("/api/history/debates", {}, mock_http)
        assert _status(result) == 200
        assert _body(result)["debates"] == []

    def test_debates_with_storage(self, handler, mock_http):
        mock_storage = MagicMock()
        mock_item = MagicMock()
        mock_item.__dict__ = {"id": "d1", "status": "completed"}
        mock_storage.list_recent.return_value = [mock_item]
        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = handler.handle("/api/history/debates", {}, mock_http)
            assert _status(result) == 200

    def test_debates_with_loop_id_filter(self, handler, mock_http):
        items = []
        for i, lid in enumerate(["target", "other", "target"]):
            m = MagicMock()
            m.__dict__ = {"id": f"d{i}", "loop_id": lid}
            items.append(m)
        mock_storage = MagicMock()
        mock_storage.list_recent.return_value = items
        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = handler.handle(
                "/api/history/debates", {"loop_id": "target"}, mock_http
            )
            body = _body(result)
            assert _status(result) == 200
            assert all(d["loop_id"] == "target" for d in body["debates"])

    def test_debates_default_limit_is_50(self, handler, mock_http):
        items = []
        for i in range(100):
            m = MagicMock()
            m.__dict__ = {"id": f"d{i}"}
            items.append(m)
        mock_storage = MagicMock()
        mock_storage.list_recent.return_value = items
        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = handler.handle("/api/history/debates", {}, mock_http)
            body = _body(result)
            assert len(body["debates"]) == 50

    def test_debates_fetch_multiplier_with_loop_id(self, handler, mock_http):
        """When loop_id is provided, fetch_limit is multiplied by 3."""
        mock_storage = MagicMock()
        mock_storage.list_recent.return_value = []
        with patch.object(handler, "get_storage", return_value=mock_storage):
            handler.handle(
                "/api/history/debates", {"loop_id": "test", "limit": "10"}, mock_http
            )
            # Should call list_recent with limit=30 (10*3)
            mock_storage.list_recent.assert_called_once()
            call_kwargs = mock_storage.list_recent.call_args
            assert call_kwargs[1]["limit"] == 30


class TestHistorySummary:
    """Tests for /api/history/summary endpoint."""

    def test_summary_no_storage_no_elo(self, handler, mock_http):
        result = handler.handle("/api/history/summary", {}, mock_http)
        assert _status(result) == 200
        body = _body(result)
        assert body["total_debates"] == 0
        assert body["total_agents"] == 0
        assert body["total_matches"] == 0

    def test_summary_with_storage(self, handler, mock_http):
        mock_storage = MagicMock()
        mock_storage.list_recent.return_value = [MagicMock()] * 42
        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = handler.handle("/api/history/summary", {}, mock_http)
            assert _status(result) == 200
            assert _body(result)["total_debates"] == 42

    def test_summary_with_elo(self, handler, mock_http):
        mock_elo = MagicMock()
        mock_elo.get_leaderboard.return_value = [MagicMock()] * 10
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = handler.handle("/api/history/summary", {}, mock_http)
            assert _status(result) == 200
            assert _body(result)["total_agents"] == 10

    def test_summary_storage_error(self, handler, mock_http):
        from aragora.exceptions import StorageError

        mock_storage = MagicMock()
        mock_storage.list_recent.side_effect = StorageError("corrupt")
        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = handler.handle("/api/history/summary", {}, mock_http)
            assert _status(result) == 500
            assert "Database error" in _body(result)["error"]

    def test_summary_database_error(self, handler, mock_http):
        from aragora.exceptions import DatabaseError

        mock_storage = MagicMock()
        mock_storage.list_recent.side_effect = DatabaseError("locked")
        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = handler.handle("/api/history/summary", {}, mock_http)
            assert _status(result) == 500
            assert "Database error" in _body(result)["error"]

    def test_summary_generic_error(self, handler, mock_http):
        mock_storage = MagicMock()
        mock_storage.list_recent.side_effect = ValueError("bad")
        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = handler.handle("/api/history/summary", {}, mock_http)
            assert _status(result) == 500

    def test_summary_ignores_loop_id(self, handler, mock_http):
        """Summary doesn't use limit, but accepts loop_id."""
        result = handler.handle("/api/history/summary", {"loop_id": "test-loop"}, mock_http)
        assert _status(result) == 200

    def test_summary_with_both_storage_and_elo(self, handler, mock_http):
        mock_storage = MagicMock()
        mock_storage.list_recent.return_value = [MagicMock()] * 5
        mock_elo = MagicMock()
        mock_elo.get_leaderboard.return_value = [MagicMock()] * 3
        with patch.object(handler, "get_storage", return_value=mock_storage):
            with patch.object(handler, "get_elo_system", return_value=mock_elo):
                result = handler.handle("/api/history/summary", {}, mock_http)
                body = _body(result)
                assert body["total_debates"] == 5
                assert body["total_agents"] == 3


# ===========================================================================
# Tests: Auth stats
# ===========================================================================


class TestAuthStats:
    """Tests for /api/auth/stats endpoint."""

    def test_auth_stats_returns_config(self, handler, mock_http):
        mock_config = MagicMock()
        mock_config.enabled = True
        mock_config.rate_limit_per_minute = 60
        mock_config.ip_rate_limit_per_minute = 30
        mock_config.token_ttl = 3600
        mock_config.get_rate_limit_stats.return_value = {"total": 100}
        with patch("aragora.server.auth.auth_config", mock_config):
            result = handler.handle("/api/auth/stats", {}, mock_http)
            assert _status(result) == 200
            body = _body(result)
            assert body["enabled"] is True
            assert body["rate_limit_per_minute"] == 60
            assert body["ip_rate_limit_per_minute"] == 30
            assert body["token_ttl_seconds"] == 3600
            assert body["stats"] == {"total": 100}

    def test_auth_stats_disabled(self, handler, mock_http):
        mock_config = MagicMock()
        mock_config.enabled = False
        mock_config.rate_limit_per_minute = 0
        mock_config.ip_rate_limit_per_minute = 0
        mock_config.token_ttl = 0
        mock_config.get_rate_limit_stats.return_value = {}
        with patch("aragora.server.auth.auth_config", mock_config):
            result = handler.handle("/api/auth/stats", {}, mock_http)
            assert _status(result) == 200
            assert _body(result)["enabled"] is False


# ===========================================================================
# Tests: Token revocation (POST)
# ===========================================================================


class TestRevokeToken:
    """Tests for POST /api/auth/revoke endpoint."""

    def test_revoke_missing_body(self, handler, mock_http):
        """Invalid JSON body returns 400."""
        mock_http.headers = {"Content-Length": "5"}
        mock_http.rfile.read.return_value = b"xxxxx"
        result = handler.handle_post("/api/auth/revoke", {}, mock_http)
        assert _status(result) == 400
        assert "Invalid JSON" in _body(result)["error"]

    def test_revoke_missing_token(self, handler):
        """Body without token field returns 400."""
        h = _make_http_handler(body={"reason": "test"})
        result = handler.handle_post("/api/auth/revoke", {}, h)
        assert _status(result) == 400
        assert "Token is required" in _body(result)["error"]

    def test_revoke_success(self, handler):
        h = _make_http_handler(body={"token": "abc123", "reason": "compromised"})
        mock_config = MagicMock()
        mock_config.revoke_token.return_value = True
        mock_config.get_revocation_count.return_value = 5
        with patch("aragora.server.auth.auth_config", mock_config):
            with patch(
                "aragora.billing.jwt_auth.revoke_token_persistent",
                return_value=True,
            ):
                result = handler.handle_post("/api/auth/revoke", {}, h)
                assert _status(result) == 200
                body = _body(result)
                assert body["success"] is True
                assert body["revoked_count"] == 5

    def test_revoke_failure(self, handler):
        h = _make_http_handler(body={"token": "bad-token"})
        mock_config = MagicMock()
        mock_config.revoke_token.return_value = False
        mock_config.get_revocation_count.return_value = 0
        with patch("aragora.server.auth.auth_config", mock_config):
            result = handler.handle_post("/api/auth/revoke", {}, h)
            assert _status(result) == 200
            body = _body(result)
            assert body["success"] is False

    def test_revoke_persistent_import_error(self, handler):
        """When persistent revocation module unavailable, still succeeds in-memory."""
        h = _make_http_handler(body={"token": "tok123"})
        mock_config = MagicMock()
        mock_config.revoke_token.return_value = True
        mock_config.get_revocation_count.return_value = 1
        with patch("aragora.server.auth.auth_config", mock_config):
            with patch.dict("sys.modules", {"aragora.billing.jwt_auth": None}):
                # Force re-import to hit ImportError
                result = handler.handle_post("/api/auth/revoke", {}, h)
                # May succeed or fail depending on import caching; check status is 200
                assert _status(result) == 200
                assert _body(result)["success"] is True

    def test_revoke_persistent_failure(self, handler):
        """When persistent revocation returns False, token is still revoked in-memory."""
        h = _make_http_handler(body={"token": "tok456", "reason": "test"})
        mock_config = MagicMock()
        mock_config.revoke_token.return_value = True
        mock_config.get_revocation_count.return_value = 2
        with patch("aragora.server.auth.auth_config", mock_config):
            with patch(
                "aragora.billing.jwt_auth.revoke_token_persistent",
                return_value=False,
            ):
                result = handler.handle_post("/api/auth/revoke", {}, h)
                assert _status(result) == 200
                assert _body(result)["success"] is True

    def test_revoke_with_empty_token(self, handler):
        """Empty string token is rejected."""
        h = _make_http_handler(body={"token": ""})
        result = handler.handle_post("/api/auth/revoke", {}, h)
        assert _status(result) == 400
        assert "Token is required" in _body(result)["error"]

    def test_handle_post_unknown_path(self, handler, mock_http):
        """POST to unknown path returns None."""
        result = handler.handle_post("/api/unknown", {}, mock_http)
        assert result is None

    def test_handle_post_with_version_prefix(self, handler):
        """POST with /api/v1/ prefix routes correctly."""
        h = _make_http_handler(body={"token": "tok"})
        mock_config = MagicMock()
        mock_config.revoke_token.return_value = True
        mock_config.get_revocation_count.return_value = 1
        with patch("aragora.server.auth.auth_config", mock_config):
            with patch(
                "aragora.billing.jwt_auth.revoke_token_persistent",
                return_value=True,
            ):
                result = handler.handle_post("/api/v1/auth/revoke", {}, h)
                assert _status(result) == 200

    def test_revoke_no_reason(self, handler):
        """Revoke without reason field succeeds (reason defaults to empty)."""
        h = _make_http_handler(body={"token": "abc"})
        mock_config = MagicMock()
        mock_config.revoke_token.return_value = True
        mock_config.get_revocation_count.return_value = 1
        with patch("aragora.server.auth.auth_config", mock_config):
            with patch(
                "aragora.billing.jwt_auth.revoke_token_persistent",
                return_value=True,
            ):
                result = handler.handle_post("/api/auth/revoke", {}, h)
                assert _status(result) == 200
                assert _body(result)["success"] is True


# ===========================================================================
# Tests: Prometheus metrics
# ===========================================================================


class TestPrometheusMetrics:
    """Tests for /metrics endpoint."""

    def test_metrics_success(self, handler, mock_http):
        with patch(
            "aragora.server.metrics.generate_metrics",
            return_value="# HELP test\ntest_metric 42\n",
        ):
            result = handler.handle("/metrics", {}, mock_http)
            assert _status(result) == 200
            assert result.content_type == "text/plain; version=0.0.4; charset=utf-8"
            assert b"test_metric 42" in result.body

    def test_metrics_import_error(self, handler, mock_http):
        with patch.dict("sys.modules", {"aragora.server.metrics": None}):
            result = handler.handle("/metrics", {}, mock_http)
            assert _status(result) == 503
            assert "not available" in _body(result)["error"]

    def test_metrics_runtime_error(self, handler, mock_http):
        with patch(
            "aragora.server.metrics.generate_metrics",
            side_effect=RuntimeError("metrics broke"),
        ):
            result = handler.handle("/metrics", {}, mock_http)
            assert _status(result) == 500

    def test_metrics_type_error(self, handler, mock_http):
        with patch(
            "aragora.server.metrics.generate_metrics",
            side_effect=TypeError("bad type"),
        ):
            result = handler.handle("/metrics", {}, mock_http)
            assert _status(result) == 500


# ===========================================================================
# Tests: Circuit breaker metrics
# ===========================================================================


class TestCircuitBreakerMetrics:
    """Tests for /api/circuit-breakers endpoint."""

    def test_circuit_breakers_success(self, handler, mock_http):
        metrics_data = {
            "summary": {"total": 5, "open": 1, "closed": 3, "half_open": 1},
            "circuit_breakers": [],
            "health": {"status": "degraded"},
        }
        with patch(
            "aragora.resilience.get_circuit_breaker_metrics",
            return_value=metrics_data,
        ):
            result = handler.handle("/api/circuit-breakers", {}, mock_http)
            assert _status(result) == 200
            body = _body(result)
            assert body["summary"]["total"] == 5
            assert body["health"]["status"] == "degraded"

    def test_circuit_breakers_import_error(self, handler, mock_http):
        with patch.dict("sys.modules", {"aragora.resilience": None}):
            result = handler.handle("/api/circuit-breakers", {}, mock_http)
            assert _status(result) == 503
            assert "not available" in _body(result)["error"]

    def test_circuit_breakers_runtime_error(self, handler, mock_http):
        with patch(
            "aragora.resilience.get_circuit_breaker_metrics",
            side_effect=RuntimeError("circuit broke"),
        ):
            result = handler.handle("/api/circuit-breakers", {}, mock_http)
            assert _status(result) == 500

    def test_circuit_breakers_value_error(self, handler, mock_http):
        with patch(
            "aragora.resilience.get_circuit_breaker_metrics",
            side_effect=ValueError("bad metrics"),
        ):
            result = handler.handle("/api/circuit-breakers", {}, mock_http)
            assert _status(result) == 500


# ===========================================================================
# Tests: Handler diagnostics (called directly since route has v1 in dict key)
# ===========================================================================


class TestHandlerDiagnostics:
    """Tests for _get_handler_diagnostics called directly."""

    def test_diagnostics_success(self, handler, mock_http):
        mock_registry = [
            ("handler_a", type("HandlerA", (), {"ROUTES": ["/api/a", "/api/b"], "__name__": "HandlerA"})),
            ("handler_b", None),
        ]
        with patch(
            "aragora.server.handler_registry.HANDLER_REGISTRY",
            mock_registry,
        ):
            result = handler._get_handler_diagnostics(mock_http)
            assert _status(result) == 200
            body = _body(result)
            assert body["handlers_count"] == 2
            assert body["loaded_count"] == 1
            assert body["handlers"][0]["status"] == "loaded"
            assert body["handlers"][0]["routes_count"] == 2
            assert body["handlers"][1]["status"] == "not_loaded"

    def test_diagnostics_import_error(self, handler, mock_http):
        with patch.dict("sys.modules", {"aragora.server.handler_registry": None}):
            result = handler._get_handler_diagnostics(mock_http)
            assert _status(result) == 503

    def test_diagnostics_oauth_handler_present(self, handler, mock_http):
        """When OAuth handler is in registry, diagnostics checks it."""
        oauth_cls = MagicMock()
        oauth_cls.__name__ = "OAuthHandler"
        oauth_cls.ROUTES = ["/api/auth/oauth/google/callback"]
        oauth_instance = MagicMock()
        oauth_instance.can_handle.return_value = True
        oauth_cls.return_value = oauth_instance
        mock_registry = [
            ("_oauth_handler", oauth_cls),
        ]
        with patch(
            "aragora.server.handler_registry.HANDLER_REGISTRY",
            mock_registry,
        ):
            result = handler._get_handler_diagnostics(mock_http)
            assert _status(result) == 200
            body = _body(result)
            assert body["oauth_handler"]["registered"] is True
            assert body["oauth_handler"]["can_handle_google_callback"] is True

    def test_diagnostics_oauth_init_error(self, handler, mock_http):
        """When OAuth handler init fails, diagnostics reports error."""
        oauth_cls = MagicMock()
        oauth_cls.__name__ = "OAuthHandler"
        oauth_cls.ROUTES = []
        oauth_cls.side_effect = TypeError("init failed")
        mock_registry = [
            ("_oauth_handler", oauth_cls),
        ]
        with patch(
            "aragora.server.handler_registry.HANDLER_REGISTRY",
            mock_registry,
        ):
            result = handler._get_handler_diagnostics(mock_http)
            assert _status(result) == 200
            body = _body(result)
            assert body["oauth_handler"]["registered"] is True
            assert "error" in body["oauth_handler"]

    def test_diagnostics_no_oauth_handler(self, handler, mock_http):
        """When no OAuth handler in registry, oauth_handler.registered is False."""
        mock_registry = [
            ("some_handler", type("SomeHandler", (), {"ROUTES": [], "__name__": "SomeHandler"})),
        ]
        with patch(
            "aragora.server.handler_registry.HANDLER_REGISTRY",
            mock_registry,
        ):
            result = handler._get_handler_diagnostics(mock_http)
            body = _body(result)
            assert body["oauth_handler"]["registered"] is False
            assert body["oauth_handler"]["can_handle_google_callback"] is False

    def test_diagnostics_sample_routes_max_5(self, handler, mock_http):
        """Sample routes are limited to 5."""
        routes = [f"/api/route/{i}" for i in range(20)]
        handler_cls = type("BigHandler", (), {"ROUTES": routes, "__name__": "BigHandler"})
        mock_registry = [("big_handler", handler_cls)]
        with patch(
            "aragora.server.handler_registry.HANDLER_REGISTRY",
            mock_registry,
        ):
            result = handler._get_handler_diagnostics(mock_http)
            body = _body(result)
            assert len(body["handlers"][0]["sample_routes"]) == 5

    def test_diagnostics_handler_without_routes_attr(self, handler, mock_http):
        """Handler class without ROUTES attr gets empty routes list."""
        handler_cls = type("MinimalHandler", (), {"__name__": "MinimalHandler"})
        mock_registry = [("minimal_handler", handler_cls)]
        with patch(
            "aragora.server.handler_registry.HANDLER_REGISTRY",
            mock_registry,
        ):
            result = handler._get_handler_diagnostics(mock_http)
            body = _body(result)
            assert body["handlers"][0]["routes_count"] == 0
            assert body["handlers"][0]["sample_routes"] == []

    def test_diagnostics_runtime_error(self, handler, mock_http):
        """RuntimeError during diagnostics is caught."""
        mock_registry = MagicMock()
        mock_registry.__iter__ = MagicMock(side_effect=RuntimeError("boom"))
        with patch(
            "aragora.server.handler_registry.HANDLER_REGISTRY",
            mock_registry,
        ):
            result = handler._get_handler_diagnostics(mock_http)
            assert _status(result) == 500


# ===========================================================================
# Tests: handle() routing returns None for unmatched paths
# ===========================================================================


class TestHandleRouting:
    """Tests for routing logic in handle()."""

    def test_handle_returns_none_for_unknown_path(self, handler, mock_http):
        result = handler.handle("/api/unknown", {}, mock_http)
        assert result is None

    def test_handle_version_prefix_stripping(self, handler, mock_http):
        """Routes with /api/v1/ prefix are stripped and matched."""
        result = handler.handle("/api/v1/debug/test", {}, mock_http)
        assert result is not None
        assert _status(result) == 200

    def test_handle_v2_prefix_stripping(self, handler, mock_http):
        """Routes with /api/v2/ prefix are also stripped."""
        result = handler.handle("/api/v2/debug/test", {}, mock_http)
        assert result is not None
        assert _status(result) == 200

    def test_handle_routes_to_correct_handler(self, handler, mock_http):
        """Each path routes to the correct method."""
        # debug test returns specific status/message
        result = handler.handle("/api/debug/test", {}, mock_http)
        body = _body(result)
        assert body["status"] == "ok"
        assert body["message"] == "Modular handler works"


# ===========================================================================
# Tests: _load_filtered_json helper
# ===========================================================================


class TestLoadFilteredJson:
    """Tests for the _load_filtered_json helper."""

    def test_file_not_exists(self, handler, tmp_path):
        result = handler._load_filtered_json(tmp_path / "missing.json")
        assert result == []

    def test_loads_all_without_filter(self, handler, tmp_path):
        data = [{"id": 1}, {"id": 2}, {"id": 3}]
        f = tmp_path / "data.json"
        f.write_text(json.dumps(data))
        result = handler._load_filtered_json(f, limit=10)
        assert len(result) == 3

    def test_respects_limit(self, handler, tmp_path):
        data = [{"id": i} for i in range(50)]
        f = tmp_path / "data.json"
        f.write_text(json.dumps(data))
        result = handler._load_filtered_json(f, limit=5)
        assert len(result) == 5

    def test_filters_by_loop_id(self, handler, tmp_path):
        data = [
            {"loop_id": "a", "val": 1},
            {"loop_id": "b", "val": 2},
            {"loop_id": "a", "val": 3},
        ]
        f = tmp_path / "data.json"
        f.write_text(json.dumps(data))
        result = handler._load_filtered_json(f, loop_id="a", limit=10)
        assert len(result) == 2
        assert all(item["loop_id"] == "a" for item in result)

    def test_filter_with_limit_terminates_early(self, handler, tmp_path):
        data = [{"loop_id": "a"} for _ in range(100)]
        f = tmp_path / "data.json"
        f.write_text(json.dumps(data))
        result = handler._load_filtered_json(f, loop_id="a", limit=3)
        assert len(result) == 3

    def test_empty_file(self, handler, tmp_path):
        f = tmp_path / "data.json"
        f.write_text("[]")
        result = handler._load_filtered_json(f, limit=10)
        assert result == []

    def test_no_matching_loop_id(self, handler, tmp_path):
        data = [{"loop_id": "a"}, {"loop_id": "b"}]
        f = tmp_path / "data.json"
        f.write_text(json.dumps(data))
        result = handler._load_filtered_json(f, loop_id="nonexistent", limit=10)
        assert result == []


# ===========================================================================
# Tests: History auth check
# ===========================================================================


class TestHistoryAuthCheck:
    """Tests for _check_history_auth method."""

    def test_auth_disabled_passes(self, handler, mock_http):
        mock_config = MagicMock()
        mock_config.enabled = False
        with patch("aragora.server.auth.auth_config", mock_config):
            result = handler._check_history_auth(mock_http)
            assert result is None  # No error

    def test_auth_enabled_no_token_returns_401(self, handler):
        mock_config = MagicMock()
        mock_config.enabled = True
        mock_config.api_token = None
        h = MagicMock()
        h.headers = {}
        mock_user_ctx = MagicMock()
        mock_user_ctx.is_authenticated = False
        with patch("aragora.server.auth.auth_config", mock_config):
            with patch(
                "aragora.server.handlers.admin.system.extract_user_from_request",
                return_value=mock_user_ctx,
            ):
                result = handler._check_history_auth(h)
                assert result is not None
                assert _status(result) == 401

    def test_auth_enabled_valid_jwt_passes(self, handler):
        mock_config = MagicMock()
        mock_config.enabled = True
        h = MagicMock()
        h.headers = {}
        mock_user_ctx = MagicMock()
        mock_user_ctx.is_authenticated = True
        with patch("aragora.server.auth.auth_config", mock_config):
            with patch(
                "aragora.server.handlers.admin.system.extract_user_from_request",
                return_value=mock_user_ctx,
            ):
                result = handler._check_history_auth(h)
                assert result is None

    def test_auth_enabled_valid_legacy_token_passes(self, handler):
        mock_config = MagicMock()
        mock_config.enabled = True
        mock_config.api_token = "secret"
        mock_config.validate_token.return_value = True
        h = MagicMock()
        h.headers = {"Authorization": "Bearer secret"}
        mock_user_ctx = MagicMock()
        mock_user_ctx.is_authenticated = False
        with patch("aragora.server.auth.auth_config", mock_config):
            with patch(
                "aragora.server.handlers.admin.system.extract_user_from_request",
                return_value=mock_user_ctx,
            ):
                result = handler._check_history_auth(h)
                assert result is None

    def test_auth_enabled_invalid_legacy_token_fails(self, handler):
        mock_config = MagicMock()
        mock_config.enabled = True
        mock_config.api_token = "secret"
        mock_config.validate_token.return_value = False
        h = MagicMock()
        h.headers = {"Authorization": "Bearer wrong"}
        mock_user_ctx = MagicMock()
        mock_user_ctx.is_authenticated = False
        with patch("aragora.server.auth.auth_config", mock_config):
            with patch(
                "aragora.server.handlers.admin.system.extract_user_from_request",
                return_value=mock_user_ctx,
            ):
                result = handler._check_history_auth(h)
                assert result is not None
                assert _status(result) == 401

    def test_auth_no_authorization_header(self, handler):
        """Handler without Authorization header fails when auth is required."""
        mock_config = MagicMock()
        mock_config.enabled = True
        mock_config.api_token = "secret"
        h = MagicMock()
        h.headers = {}  # No Authorization header
        mock_user_ctx = MagicMock()
        mock_user_ctx.is_authenticated = False
        with patch("aragora.server.auth.auth_config", mock_config):
            with patch(
                "aragora.server.handlers.admin.system.extract_user_from_request",
                return_value=mock_user_ctx,
            ):
                result = handler._check_history_auth(h)
                assert result is not None
                assert _status(result) == 401


# ===========================================================================
# Tests: Handler initialization
# ===========================================================================


class TestHandlerInit:
    """Tests for handler initialization."""

    def test_default_ctx_is_empty_dict(self):
        h = SystemHandler()
        assert h.ctx == {}

    def test_ctx_passed_through(self):
        ctx = {"storage": MagicMock(), "nomic_dir": Path("/tmp")}
        h = SystemHandler(ctx=ctx)
        assert h.ctx is ctx

    def test_routes_list_complete(self):
        expected = {
            "/api/debug/test",
            "/api/history/cycles",
            "/api/history/events",
            "/api/history/debates",
            "/api/history/summary",
            "/api/system/maintenance",
            "/api/auth/stats",
            "/api/auth/revoke",
            "/api/circuit-breakers",
            "/metrics",
            "/api/v1/diagnostics/handlers",
        }
        assert set(SystemHandler.ROUTES) == expected

    def test_history_endpoints_list(self):
        expected = {
            "/api/history/cycles",
            "/api/history/events",
            "/api/history/debates",
            "/api/history/summary",
        }
        assert set(SystemHandler.HISTORY_ENDPOINTS) == expected

    def test_history_config_entries(self):
        """_HISTORY_CONFIG has correct method names and limits."""
        config = SystemHandler._HISTORY_CONFIG
        assert config["/api/history/cycles"] == ("_get_history_cycles", 50, 200)
        assert config["/api/history/events"] == ("_get_history_events", 100, 500)
        assert config["/api/history/debates"] == ("_get_history_debates", 50, 200)
        assert config["/api/history/summary"] == ("_get_history_summary", 0, 0)


# ===========================================================================
# Tests: Edge cases and integration
# ===========================================================================


class TestEdgeCases:
    """Edge case and integration tests."""

    def test_history_endpoint_with_empty_loop_id(self, handler, mock_http):
        """Empty loop_id string is treated as no filter."""
        with patch.object(handler, "get_nomic_dir", return_value=None):
            result = handler.handle("/api/history/cycles", {"loop_id": ""}, mock_http)
            assert _status(result) == 200

    def test_all_simple_routes_return_results(self, handler, mock_http):
        """Simple routes return HandlerResult (not None)."""
        result = handler.handle("/api/debug/test", {}, mock_http)
        assert result is not None

    def test_metrics_response_is_text_plain(self, handler, mock_http):
        """Metrics endpoint returns text/plain content type."""
        with patch(
            "aragora.server.metrics.generate_metrics",
            return_value="# test\n",
        ):
            result = handler.handle("/metrics", {}, mock_http)
            assert "text/plain" in result.content_type

    def test_json_responses_are_application_json(self, handler, mock_http):
        """Non-metrics endpoints return application/json."""
        result = handler.handle("/api/debug/test", {}, mock_http)
        assert result.content_type == "application/json"

    def test_history_config_has_correct_entries(self):
        """_HISTORY_CONFIG covers all history endpoints."""
        for endpoint in SystemHandler.HISTORY_ENDPOINTS:
            assert endpoint in SystemHandler._HISTORY_CONFIG

    def test_history_summary_has_zero_limits(self):
        """Summary endpoint has 0/0 limits indicating no limit param."""
        _, default_limit, max_limit = SystemHandler._HISTORY_CONFIG["/api/history/summary"]
        assert default_limit == 0
        assert max_limit == 0

    def test_loop_id_with_spaces_rejected(self, handler, mock_http):
        """loop_id with spaces is rejected by validation."""
        result = handler.handle(
            "/api/history/cycles", {"loop_id": "has spaces"}, mock_http
        )
        assert _status(result) == 400

    def test_loop_id_valid_format(self, handler, mock_http, tmp_path):
        """Valid loop_id (alphanumeric with dashes) is accepted."""
        (tmp_path / "cycles.json").write_text("[]")
        with patch.object(handler, "get_nomic_dir", return_value=tmp_path):
            result = handler.handle(
                "/api/history/cycles", {"loop_id": "valid-loop-id"}, mock_http
            )
            assert _status(result) == 200

    def test_maintenance_with_version_prefix(self, handler, mock_http):
        """Maintenance route works via /api/v1/system/maintenance."""
        result = handler.handle(
            "/api/v1/system/maintenance", {"task": "invalid"}, mock_http
        )
        assert _status(result) == 400

    def test_auth_stats_with_version_prefix(self, handler, mock_http):
        """Auth stats route works via /api/v1/auth/stats."""
        mock_config = MagicMock()
        mock_config.enabled = False
        mock_config.rate_limit_per_minute = 0
        mock_config.ip_rate_limit_per_minute = 0
        mock_config.token_ttl = 0
        mock_config.get_rate_limit_stats.return_value = {}
        with patch("aragora.server.auth.auth_config", mock_config):
            result = handler.handle("/api/v1/auth/stats", {}, mock_http)
            assert _status(result) == 200

    def test_circuit_breakers_with_version_prefix(self, handler, mock_http):
        """Circuit breakers via /api/v1/circuit-breakers."""
        metrics_data = {"summary": {}, "circuit_breakers": [], "health": {}}
        with patch(
            "aragora.resilience.get_circuit_breaker_metrics",
            return_value=metrics_data,
        ):
            result = handler.handle("/api/v1/circuit-breakers", {}, mock_http)
            assert _status(result) == 200
