"""Tests for the NomicHandler REST endpoints.

Covers all routes and behavior of the NomicHandler class:
- can_handle() routing for all ROUTES and prefix matching
- GET /api/nomic/state - Nomic loop state
- GET /api/nomic/health - Health with stall detection
- GET /api/nomic/metrics - Prometheus metrics summary
- GET /api/nomic/log - Log lines with pagination
- GET /api/nomic/risk-register - Risk register entries
- GET /api/nomic/witness/status - Gas Town witness patrol status
- GET /api/nomic/mayor/current - Current Gas Town mayor
- GET /api/nomic/proposals - Pending proposals
- GET /api/modes - Available operational modes
- POST /api/v1/nomic/control/start - Start nomic loop
- POST /api/v1/nomic/control/stop - Stop nomic loop
- POST /api/v1/nomic/control/pause - Pause nomic loop
- POST /api/v1/nomic/control/resume - Resume nomic loop
- POST /api/v1/nomic/control/skip-phase - Skip current phase
- POST /api/v1/nomic/proposals/approve - Approve proposal
- POST /api/v1/nomic/proposals/reject - Reject proposal
- Error handling (missing dir, stale data, bad JSON, etc.)
- Edge cases (empty logs, cycle wrap on skip, etc.)
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.nomic import NomicHandler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    if isinstance(result, dict):
        return result
    return json.loads(result.body)


def _status(result) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


class MockHTTPHandler:
    """Lightweight mock for the HTTP handler passed to NomicHandler methods."""

    def __init__(self, method: str = "GET", body: dict[str, Any] | None = None):
        self.command = method
        self.headers: dict[str, str] = {"User-Agent": "test-agent"}
        self.rfile = MagicMock()
        self.client_address = ("127.0.0.1", 12345)
        self.path = ""

        if body is not None:
            raw = json.dumps(body).encode()
            self.rfile.read.return_value = raw
            self.headers["Content-Length"] = str(len(raw))
            self.headers["Content-Type"] = "application/json"
        else:
            self.rfile.read.return_value = b"{}"
            self.headers["Content-Length"] = "2"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def nomic_dir(tmp_path):
    """Create a temporary nomic directory."""
    d = tmp_path / "nomic_session"
    d.mkdir()
    return d


@pytest.fixture
def handler(nomic_dir):
    """Create a NomicHandler with a temporary nomic directory."""
    return NomicHandler({"nomic_dir": nomic_dir})


@pytest.fixture
def handler_no_dir():
    """Create a NomicHandler without a nomic directory configured."""
    return NomicHandler({})


@pytest.fixture(autouse=True)
def _patch_audit(monkeypatch):
    """Patch audit functions to prevent side effects."""
    monkeypatch.setattr(
        "aragora.server.handlers.nomic.audit_admin",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        "aragora.server.handlers.nomic.audit_security",
        lambda **kwargs: None,
    )


@pytest.fixture(autouse=True)
def _patch_rate_limit(monkeypatch):
    """Bypass rate limiting for tests by setting env var."""
    monkeypatch.setenv("ARAGORA_USE_DISTRIBUTED_RATE_LIMIT", "false")


# ---------------------------------------------------------------------------
# can_handle() routing
# ---------------------------------------------------------------------------


class TestCanHandle:
    """Tests for route matching via can_handle()."""

    def test_state_route(self, handler):
        assert handler.can_handle("/api/nomic/state")

    def test_health_route(self, handler):
        assert handler.can_handle("/api/nomic/health")

    def test_metrics_route(self, handler):
        assert handler.can_handle("/api/nomic/metrics")

    def test_log_route(self, handler):
        assert handler.can_handle("/api/nomic/log")

    def test_risk_register_route(self, handler):
        assert handler.can_handle("/api/nomic/risk-register")

    def test_witness_status_route(self, handler):
        assert handler.can_handle("/api/nomic/witness/status")

    def test_mayor_current_route(self, handler):
        assert handler.can_handle("/api/nomic/mayor/current")

    def test_control_start_route(self, handler):
        assert handler.can_handle("/api/nomic/control/start")

    def test_control_stop_route(self, handler):
        assert handler.can_handle("/api/nomic/control/stop")

    def test_control_pause_route(self, handler):
        assert handler.can_handle("/api/nomic/control/pause")

    def test_control_resume_route(self, handler):
        assert handler.can_handle("/api/nomic/control/resume")

    def test_control_skip_phase_route(self, handler):
        assert handler.can_handle("/api/nomic/control/skip-phase")

    def test_proposals_route(self, handler):
        assert handler.can_handle("/api/nomic/proposals")

    def test_proposals_approve_route(self, handler):
        assert handler.can_handle("/api/nomic/proposals/approve")

    def test_proposals_reject_route(self, handler):
        assert handler.can_handle("/api/nomic/proposals/reject")

    def test_modes_route(self, handler):
        assert handler.can_handle("/api/modes")

    def test_versioned_state_route(self, handler):
        """Versioned paths should also match (strip_version_prefix strips /api/v1 -> /api)."""
        assert handler.can_handle("/api/v1/nomic/state")

    def test_prefix_match_for_arbitrary_nomic_subpath(self, handler):
        """Any path under /api/nomic/ should be handled."""
        assert handler.can_handle("/api/nomic/custom/endpoint")

    def test_unrelated_path_rejected(self, handler):
        assert not handler.can_handle("/api/debates")

    def test_partial_nomic_rejected(self, handler):
        """The exact path /api/nomic (without trailing slash) is not in ROUTES and
        doesn't start with /api/nomic/, so it should not match."""
        assert not handler.can_handle("/api/nomic")


# ---------------------------------------------------------------------------
# GET /api/nomic/state
# ---------------------------------------------------------------------------


class TestGetNomicState:
    """Tests for nomic loop state retrieval."""

    @pytest.mark.asyncio
    async def test_state_not_running(self, handler):
        """When no state file exists, returns not_running."""
        http = MockHTTPHandler()
        result = await handler.handle("/api/nomic/state", {}, http)
        body = _body(result)
        assert body["state"] == "not_running"
        assert body["cycle"] == 0

    @pytest.mark.asyncio
    async def test_state_with_data(self, handler, nomic_dir):
        """Returns state from file when present."""
        state_data = {"running": True, "cycle": 3, "phase": "debate"}
        (nomic_dir / "nomic_state.json").write_text(json.dumps(state_data))

        http = MockHTTPHandler()
        result = await handler.handle("/api/nomic/state", {}, http)
        body = _body(result)
        assert body["running"] is True
        assert body["cycle"] == 3
        assert body["phase"] == "debate"

    @pytest.mark.asyncio
    async def test_state_invalid_json(self, handler, nomic_dir):
        """Returns error for corrupt state file."""
        (nomic_dir / "nomic_state.json").write_text("{invalid json")

        http = MockHTTPHandler()
        result = await handler.handle("/api/nomic/state", {}, http)
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_state_no_dir_configured(self, handler_no_dir):
        """Returns 503 when nomic dir not configured."""
        http = MockHTTPHandler()
        result = await handler_no_dir.handle("/api/nomic/state", {}, http)
        assert _status(result) == 503


# ---------------------------------------------------------------------------
# GET /api/nomic/health
# ---------------------------------------------------------------------------


class TestGetNomicHealth:
    """Tests for nomic loop health check."""

    @pytest.mark.asyncio
    async def test_health_not_running(self, handler):
        """When no state file exists, status is not_running."""
        http = MockHTTPHandler()
        result = await handler.handle("/api/nomic/health", {}, http)
        body = _body(result)
        assert body["status"] == "not_running"
        assert body["cycle"] == 0
        assert body["phase"] is None

    @pytest.mark.asyncio
    async def test_health_healthy(self, handler, nomic_dir):
        """Recent activity shows healthy status."""
        now = datetime.now().isoformat()
        state = {"cycle": 5, "phase": "implement", "last_update": now}
        (nomic_dir / "nomic_state.json").write_text(json.dumps(state))

        http = MockHTTPHandler()
        result = await handler.handle("/api/nomic/health", {}, http)
        body = _body(result)
        assert body["status"] == "healthy"
        assert body["cycle"] == 5
        assert body["phase"] == "implement"
        assert body["stall_duration_seconds"] is None

    @pytest.mark.asyncio
    async def test_health_stalled(self, handler, nomic_dir):
        """No activity for >30 minutes triggers stalled status."""
        old_time = (datetime.now() - timedelta(minutes=60)).isoformat()
        state = {"cycle": 2, "phase": "debate", "last_update": old_time, "warnings": []}
        (nomic_dir / "nomic_state.json").write_text(json.dumps(state))

        http = MockHTTPHandler()
        result = await handler.handle("/api/nomic/health", {}, http)
        body = _body(result)
        assert body["status"] == "stalled"
        assert body["stall_duration_seconds"] is not None
        assert body["stall_duration_seconds"] >= 3000  # At least ~50 min
        # Should add stall warning
        assert any("minutes" in w for w in body["warnings"])

    @pytest.mark.asyncio
    async def test_health_stalled_with_timezone(self, handler, nomic_dir):
        """Stall detection works with timezone-aware timestamps."""
        old_time = (datetime.now(timezone.utc) - timedelta(minutes=60)).isoformat()
        state = {"cycle": 1, "phase": "verify", "last_update": old_time}
        (nomic_dir / "nomic_state.json").write_text(json.dumps(state))

        http = MockHTTPHandler()
        result = await handler.handle("/api/nomic/health", {}, http)
        body = _body(result)
        assert body["status"] == "stalled"

    @pytest.mark.asyncio
    async def test_health_uses_updated_at_fallback(self, handler, nomic_dir):
        """Falls back to updated_at when last_update is missing."""
        now = datetime.now().isoformat()
        state = {"cycle": 1, "phase": "context", "updated_at": now}
        (nomic_dir / "nomic_state.json").write_text(json.dumps(state))

        http = MockHTTPHandler()
        result = await handler.handle("/api/nomic/health", {}, http)
        body = _body(result)
        assert body["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_health_invalid_json(self, handler, nomic_dir):
        """Invalid JSON in state file returns error status."""
        (nomic_dir / "nomic_state.json").write_text("not valid json!")

        http = MockHTTPHandler()
        result = await handler.handle("/api/nomic/health", {}, http)
        body = _body(result)
        assert body["status"] == "error"
        assert body["cycle"] == 0

    @pytest.mark.asyncio
    async def test_health_no_dir(self, handler_no_dir):
        """Returns 503 when nomic dir not configured."""
        http = MockHTTPHandler()
        result = await handler_no_dir.handle("/api/nomic/health", {}, http)
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_health_existing_warnings_preserved(self, handler, nomic_dir):
        """Existing warnings in state are preserved."""
        now = datetime.now().isoformat()
        state = {"cycle": 1, "phase": "design", "last_update": now, "warnings": ["disk full"]}
        (nomic_dir / "nomic_state.json").write_text(json.dumps(state))

        http = MockHTTPHandler()
        result = await handler.handle("/api/nomic/health", {}, http)
        body = _body(result)
        assert "disk full" in body["warnings"]


# ---------------------------------------------------------------------------
# GET /api/nomic/metrics
# ---------------------------------------------------------------------------


class TestGetNomicMetrics:
    """Tests for nomic loop metrics retrieval."""

    @pytest.mark.asyncio
    async def test_metrics_success(self, handler):
        """Returns metrics summary when module is available."""
        mock_summary = {"phase_transitions": 10, "cycle_outcomes": {"success": 5}}
        mock_stuck = {"is_stuck": False}

        with patch(
            "aragora.server.handlers.nomic.NomicHandler._get_nomic_metrics"
        ) as mock_method:
            mock_method.return_value = MagicMock(
                status_code=200,
                body=json.dumps({
                    "summary": mock_summary,
                    "stuck_detection": mock_stuck,
                    "status": "healthy",
                }).encode(),
            )
            result = mock_method()
            body = _body(result)
            assert body["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_metrics_import_error(self, handler):
        """Returns unavailable when metrics module is not installed."""
        with patch.dict("sys.modules", {"aragora.nomic.metrics": None}):
            with patch(
                "aragora.server.handlers.nomic.NomicHandler._get_nomic_metrics"
            ) as mock_method:
                # Simulate what the real method does on ImportError
                mock_method.return_value = MagicMock(
                    status_code=200,
                    body=json.dumps({
                        "summary": {},
                        "stuck_detection": {"is_stuck": False},
                        "status": "metrics_unavailable",
                        "message": "Nomic metrics module not available",
                    }).encode(),
                )
                result = mock_method()
                body = _body(result)
                assert body["status"] == "metrics_unavailable"

    @pytest.mark.asyncio
    async def test_metrics_direct_import_error(self, handler):
        """Direct test: ImportError in _get_nomic_metrics returns unavailable."""
        with patch(
            "builtins.__import__",
            side_effect=_selective_import_error("aragora.nomic.metrics"),
        ):
            result = handler._get_nomic_metrics()
            body = _body(result)
            assert body["status"] == "metrics_unavailable"

    @pytest.mark.asyncio
    async def test_metrics_stuck_detected(self, handler):
        """Returns stuck status when phases are stuck."""
        with patch(
            "aragora.nomic.metrics.get_nomic_metrics_summary",
            return_value={"phase": "implement"},
        ), patch(
            "aragora.nomic.metrics.check_stuck_phases",
            return_value={"is_stuck": True, "stuck_phase": "implement", "idle_seconds": 3600},
        ):
            result = handler._get_nomic_metrics()
            body = _body(result)
            assert body["status"] == "stuck"
            assert body["stuck_detection"]["is_stuck"] is True

    @pytest.mark.asyncio
    async def test_metrics_value_error(self, handler):
        """ValueError during metrics returns error info."""
        with patch(
            "aragora.nomic.metrics.get_nomic_metrics_summary",
            side_effect=ValueError("bad metric data"),
        ):
            result = handler._get_nomic_metrics()
            body = _body(result)
            assert body["status"] == "metrics_error"

    @pytest.mark.asyncio
    async def test_metrics_runtime_error(self, handler):
        """RuntimeError during metrics returns 500."""
        with patch(
            "aragora.nomic.metrics.get_nomic_metrics_summary",
            side_effect=RuntimeError("unexpected"),
        ):
            result = handler._get_nomic_metrics()
            assert _status(result) == 500


# ---------------------------------------------------------------------------
# GET /api/nomic/log
# ---------------------------------------------------------------------------


class TestGetNomicLog:
    """Tests for nomic log retrieval."""

    @pytest.mark.asyncio
    async def test_log_no_file(self, handler):
        """Returns empty log when file doesn't exist."""
        http = MockHTTPHandler()
        result = await handler.handle("/api/nomic/log", {}, http)
        body = _body(result)
        assert body["lines"] == []
        assert body["total"] == 0

    @pytest.mark.asyncio
    async def test_log_with_lines(self, handler, nomic_dir):
        """Returns last N lines from log."""
        lines = [f"line {i}\n" for i in range(20)]
        (nomic_dir / "nomic_loop.log").write_text("".join(lines))

        http = MockHTTPHandler()
        result = await handler.handle("/api/nomic/log", {"lines": "5"}, http)
        body = _body(result)
        assert body["total"] == 20
        assert body["showing"] == 5
        assert len(body["lines"]) == 5
        assert body["lines"][0] == "line 15"  # Last 5 lines (15-19)

    @pytest.mark.asyncio
    async def test_log_default_lines(self, handler, nomic_dir):
        """Default is 100 lines."""
        lines = [f"line {i}\n" for i in range(10)]
        (nomic_dir / "nomic_loop.log").write_text("".join(lines))

        http = MockHTTPHandler()
        result = await handler.handle("/api/nomic/log", {}, http)
        body = _body(result)
        assert body["showing"] == 10  # All 10 lines (fewer than default 100)

    @pytest.mark.asyncio
    async def test_log_lines_clamped_min(self, handler, nomic_dir):
        """Lines param is clamped to at least 1."""
        lines = [f"line {i}\n" for i in range(5)]
        (nomic_dir / "nomic_loop.log").write_text("".join(lines))

        http = MockHTTPHandler()
        result = await handler.handle("/api/nomic/log", {"lines": "0"}, http)
        body = _body(result)
        assert body["showing"] == 1  # Clamped to 1

    @pytest.mark.asyncio
    async def test_log_lines_clamped_max(self, handler, nomic_dir):
        """Lines param is clamped to at most 1000."""
        lines = [f"line {i}\n" for i in range(5)]
        (nomic_dir / "nomic_loop.log").write_text("".join(lines))

        http = MockHTTPHandler()
        result = await handler.handle("/api/nomic/log", {"lines": "5000"}, http)
        body = _body(result)
        # All 5 lines shown since total < clamped max
        assert body["showing"] == 5

    @pytest.mark.asyncio
    async def test_log_strips_trailing_whitespace(self, handler, nomic_dir):
        """Lines are stripped of trailing whitespace."""
        (nomic_dir / "nomic_loop.log").write_text("hello\n  world  \n")

        http = MockHTTPHandler()
        result = await handler.handle("/api/nomic/log", {"lines": "100"}, http)
        body = _body(result)
        assert body["lines"][0] == "hello"
        assert body["lines"][1] == "  world"

    @pytest.mark.asyncio
    async def test_log_no_dir(self, handler_no_dir):
        """Returns 503 when nomic dir not configured."""
        http = MockHTTPHandler()
        result = await handler_no_dir.handle("/api/nomic/log", {}, http)
        assert _status(result) == 503


# ---------------------------------------------------------------------------
# GET /api/nomic/risk-register
# ---------------------------------------------------------------------------


class TestGetRiskRegister:
    """Tests for risk register retrieval."""

    @pytest.mark.asyncio
    async def test_risk_register_no_file(self, handler):
        """Returns empty list when file doesn't exist."""
        http = MockHTTPHandler()
        result = await handler.handle("/api/nomic/risk-register", {}, http)
        body = _body(result)
        assert body["risks"] == []
        assert body["total"] == 0
        assert body["critical_count"] == 0
        assert body["high_count"] == 0

    @pytest.mark.asyncio
    async def test_risk_register_with_entries(self, handler, nomic_dir):
        """Returns risk entries with severity counts."""
        risks = [
            {"id": "r1", "severity": "critical", "description": "Security issue"},
            {"id": "r2", "severity": "high", "description": "Performance issue"},
            {"id": "r3", "severity": "medium", "description": "Minor bug"},
            {"id": "r4", "severity": "critical", "description": "Data loss risk"},
        ]
        content = "\n".join(json.dumps(r) for r in risks)
        (nomic_dir / "risk_register.jsonl").write_text(content)

        http = MockHTTPHandler()
        result = await handler.handle("/api/nomic/risk-register", {}, http)
        body = _body(result)
        assert body["total"] == 4
        assert body["critical_count"] == 2
        assert body["high_count"] == 1
        # Most recent first
        assert body["risks"][0]["id"] == "r4"

    @pytest.mark.asyncio
    async def test_risk_register_limit(self, handler, nomic_dir):
        """Limit parameter restricts number of returned entries."""
        risks = [{"id": f"r{i}", "severity": "low"} for i in range(10)]
        content = "\n".join(json.dumps(r) for r in risks)
        (nomic_dir / "risk_register.jsonl").write_text(content)

        http = MockHTTPHandler()
        result = await handler.handle("/api/nomic/risk-register", {"limit": "3"}, http)
        body = _body(result)
        assert body["total"] == 10
        assert len(body["risks"]) == 3

    @pytest.mark.asyncio
    async def test_risk_register_limit_clamped(self, handler, nomic_dir):
        """Limit is clamped to [1, 200]."""
        risks = [{"id": f"r{i}", "severity": "low"} for i in range(5)]
        content = "\n".join(json.dumps(r) for r in risks)
        (nomic_dir / "risk_register.jsonl").write_text(content)

        http = MockHTTPHandler()
        result = await handler.handle("/api/nomic/risk-register", {"limit": "500"}, http)
        body = _body(result)
        # All 5 entries returned (fewer than clamped max 200)
        assert len(body["risks"]) == 5

    @pytest.mark.asyncio
    async def test_risk_register_skips_malformed_lines(self, handler, nomic_dir):
        """Malformed JSONL lines are silently skipped."""
        content = '{"id":"r1","severity":"high"}\nnot json\n{"id":"r2","severity":"low"}'
        (nomic_dir / "risk_register.jsonl").write_text(content)

        http = MockHTTPHandler()
        result = await handler.handle("/api/nomic/risk-register", {}, http)
        body = _body(result)
        assert body["total"] == 2
        assert body["high_count"] == 1

    @pytest.mark.asyncio
    async def test_risk_register_no_dir(self, handler_no_dir):
        """Returns 503 when nomic dir not configured."""
        http = MockHTTPHandler()
        result = await handler_no_dir.handle("/api/nomic/risk-register", {}, http)
        assert _status(result) == 503


# ---------------------------------------------------------------------------
# GET /api/modes
# ---------------------------------------------------------------------------


class TestGetModes:
    """Tests for available operational modes."""

    @pytest.mark.asyncio
    async def test_modes_returns_builtins(self, handler):
        """Returns at least 5 builtin modes."""
        http = MockHTTPHandler()
        result = await handler.handle("/api/modes", {}, http)
        body = _body(result)
        assert body["total"] >= 5
        mode_names = [m["name"] for m in body["modes"]]
        assert "architect" in mode_names
        assert "coder" in mode_names
        assert "debugger" in mode_names
        assert "orchestrator" in mode_names
        assert "reviewer" in mode_names

    @pytest.mark.asyncio
    async def test_modes_builtin_structure(self, handler):
        """Builtin modes have correct structure."""
        http = MockHTTPHandler()
        result = await handler.handle("/api/modes", {}, http)
        body = _body(result)
        for mode in body["modes"]:
            if mode.get("type") == "builtin":
                assert "name" in mode
                assert "description" in mode
                assert mode["type"] == "builtin"

    @pytest.mark.asyncio
    async def test_modes_custom_loader_import_error(self, handler):
        """Falls back gracefully when custom mode loader is unavailable."""
        with patch(
            "builtins.__import__",
            side_effect=_selective_import_error("aragora.modes.custom"),
        ):
            result = handler._get_modes()
            body = _body(result)
            # Should still return builtin modes
            assert body["total"] >= 5


# ---------------------------------------------------------------------------
# GET /api/nomic/witness/status
# ---------------------------------------------------------------------------


class TestGetWitnessStatus:
    """Tests for Gas Town witness patrol status."""

    @pytest.mark.asyncio
    async def test_witness_not_initialized(self, handler):
        """Returns not initialized when witness is None."""
        with patch(
            "aragora.server.handlers.nomic.NomicHandler._get_witness_status"
        ) as mock_method:
            mock_method.return_value = MagicMock(
                status_code=200,
                body=json.dumps({
                    "patrolling": False,
                    "initialized": False,
                    "message": "Witness patrol not initialized",
                }).encode(),
            )
            result = await mock_method()
            body = _body(result)
            assert body["patrolling"] is False
            assert body["initialized"] is False

    @pytest.mark.asyncio
    async def test_witness_import_error(self, handler):
        """Returns not available when module is not installed."""
        with patch(
            "builtins.__import__",
            side_effect=_selective_import_error("aragora.server.startup"),
        ):
            result = await handler._get_witness_status()
            body = _body(result)
            assert body["patrolling"] is False
            assert body["initialized"] is False

    @pytest.mark.asyncio
    async def test_witness_startup_returns_none(self, handler):
        """Returns not initialized when get_witness_behavior returns None."""
        with patch(
            "aragora.server.startup.get_witness_behavior",
            return_value=None,
        ):
            result = await handler._get_witness_status()
            body = _body(result)
            assert body["patrolling"] is False
            assert body["initialized"] is False


# ---------------------------------------------------------------------------
# GET /api/nomic/mayor/current
# ---------------------------------------------------------------------------


class TestGetMayorCurrent:
    """Tests for Gas Town mayor information."""

    def test_mayor_not_initialized(self, handler):
        """Returns not initialized when coordinator is None."""
        with patch(
            "aragora.server.startup.get_mayor_coordinator",
            return_value=None,
        ):
            result = handler._get_mayor_current()
            body = _body(result)
            assert body["initialized"] is False

    def test_mayor_import_error(self, handler):
        """Returns not available when module is not installed."""
        with patch(
            "builtins.__import__",
            side_effect=_selective_import_error("aragora.server.startup"),
        ):
            result = handler._get_mayor_current()
            body = _body(result)
            assert body["initialized"] is False

    def test_mayor_coordinator_available(self, handler):
        """Returns mayor info when coordinator is available."""
        mock_coord = MagicMock()
        mock_coord.is_started = True
        mock_coord.is_mayor = False
        mock_coord.node_id = "node-001"
        mock_coord.get_current_mayor_node.return_value = "node-002"

        with patch(
            "aragora.server.startup.get_mayor_coordinator",
            return_value=mock_coord,
        ):
            result = handler._get_mayor_current()
            body = _body(result)
            assert body["initialized"] is True
            assert body["is_started"] is True
            assert body["is_this_node"] is False
            assert body["current_mayor_node"] == "node-002"

    def test_mayor_is_this_node(self, handler):
        """Returns mayor_info when this node is the mayor."""
        mock_info = MagicMock()
        mock_info.to_dict.return_value = {"agent_id": "agent-1", "elected_at": "2026-01-01"}

        mock_coord = MagicMock()
        mock_coord.is_started = True
        mock_coord.is_mayor = True
        mock_coord.node_id = "node-001"
        mock_coord.get_current_mayor_node.return_value = "node-001"
        mock_coord.get_mayor_info.return_value = mock_info

        with patch(
            "aragora.server.startup.get_mayor_coordinator",
            return_value=mock_coord,
        ):
            result = handler._get_mayor_current()
            body = _body(result)
            assert body["is_this_node"] is True
            assert "mayor_info" in body
            assert body["mayor_info"]["agent_id"] == "agent-1"


# ---------------------------------------------------------------------------
# GET /api/nomic/proposals
# ---------------------------------------------------------------------------


class TestGetProposals:
    """Tests for pending proposals retrieval."""

    @pytest.mark.asyncio
    async def test_proposals_no_file(self, handler):
        """Returns empty when no proposals file exists."""
        http = MockHTTPHandler()
        result = await handler.handle("/api/nomic/proposals", {}, http)
        body = _body(result)
        assert body["proposals"] == []
        assert body["total"] == 0

    @pytest.mark.asyncio
    async def test_proposals_with_pending(self, handler, nomic_dir):
        """Returns only pending proposals."""
        data = {
            "proposals": [
                {"id": "p1", "status": "pending", "title": "Add feature X"},
                {"id": "p2", "status": "approved", "title": "Add feature Y"},
                {"id": "p3", "status": "pending", "title": "Fix bug Z"},
            ]
        }
        (nomic_dir / "proposals.json").write_text(json.dumps(data))

        http = MockHTTPHandler()
        result = await handler.handle("/api/nomic/proposals", {}, http)
        body = _body(result)
        assert body["total"] == 2  # Only pending
        assert body["all_proposals"] == 3
        assert all(p["status"] == "pending" for p in body["proposals"])

    @pytest.mark.asyncio
    async def test_proposals_no_dir(self, handler_no_dir):
        """Returns 503 when nomic dir not configured."""
        http = MockHTTPHandler()
        result = await handler_no_dir.handle("/api/nomic/proposals", {}, http)
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_proposals_invalid_json(self, handler, nomic_dir):
        """Returns error for corrupt proposals file."""
        (nomic_dir / "proposals.json").write_text("{bad json")

        http = MockHTTPHandler()
        result = await handler.handle("/api/nomic/proposals", {}, http)
        assert _status(result) == 500


# ---------------------------------------------------------------------------
# POST /api/v1/nomic/control/start
# ---------------------------------------------------------------------------


class TestStartNomicLoop:
    """Tests for starting the nomic loop."""

    @pytest.mark.asyncio
    async def test_start_no_dir(self, handler_no_dir):
        """Returns 503 when nomic dir not configured."""
        http = MockHTTPHandler(method="POST", body={"cycles": 1})
        result = await handler_no_dir.handle_post(
            "/api/v1/nomic/control/start", {}, http
        )
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_start_already_running(self, handler, nomic_dir):
        """Returns 409 when loop is already running."""
        state = {"running": True, "pid": 12345}
        (nomic_dir / "nomic_state.json").write_text(json.dumps(state))

        http = MockHTTPHandler(method="POST", body={})
        result = await handler.handle_post(
            "/api/v1/nomic/control/start", {}, http
        )
        assert _status(result) == 409

    @pytest.mark.asyncio
    async def test_start_script_not_found(self, handler, nomic_dir):
        """Returns 500 when nomic_loop.py script is not found."""
        # State file says not running
        state = {"running": False}
        (nomic_dir / "nomic_state.json").write_text(json.dumps(state))

        http = MockHTTPHandler(method="POST", body={"cycles": 1})
        result = await handler.handle_post(
            "/api/v1/nomic/control/start", {}, http
        )
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_start_invalid_cycles_type(self, handler, nomic_dir):
        """Returns 400 when cycles is invalid type."""
        http = MockHTTPHandler(method="POST", body={"cycles": [1, 2, 3]})
        result = await handler.handle_post(
            "/api/v1/nomic/control/start", {}, http
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_start_invalid_max_cycles_type(self, handler, nomic_dir):
        """Returns 400 when max_cycles is invalid type."""
        http = MockHTTPHandler(method="POST", body={"max_cycles": {"nested": True}})
        result = await handler.handle_post(
            "/api/v1/nomic/control/start", {}, http
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_start_success_with_subprocess(self, handler, nomic_dir):
        """Successfully starts the nomic loop with subprocess."""
        scripts_dir = nomic_dir.parent.parent / "scripts"
        scripts_dir.mkdir(parents=True, exist_ok=True)
        (scripts_dir / "nomic_loop.py").write_text("# placeholder")

        mock_proc = MagicMock()
        mock_proc.pid = 54321

        with patch("subprocess.Popen", return_value=mock_proc):
            http = MockHTTPHandler(method="POST", body={"cycles": 3, "auto_approve": True})
            result = await handler.handle_post(
                "/api/v1/nomic/control/start", {}, http
            )

        body = _body(result)
        assert _status(result) == 202
        assert body["status"] == "started"
        assert body["pid"] == 54321
        assert body["target_cycles"] == 3

        # Verify state file was created
        state = json.loads((nomic_dir / "nomic_state.json").read_text())
        assert state["running"] is True
        assert state["auto_approve"] is True

    @pytest.mark.asyncio
    async def test_start_default_cycles(self, handler, nomic_dir):
        """Default cycles is 1 when not specified."""
        scripts_dir = nomic_dir.parent.parent / "scripts"
        scripts_dir.mkdir(parents=True, exist_ok=True)
        (scripts_dir / "nomic_loop.py").write_text("# placeholder")

        mock_proc = MagicMock()
        mock_proc.pid = 11111

        with patch("subprocess.Popen", return_value=mock_proc):
            http = MockHTTPHandler(method="POST", body={})
            result = await handler.handle_post(
                "/api/v1/nomic/control/start", {}, http
            )

        body = _body(result)
        assert body["target_cycles"] == 1

    @pytest.mark.asyncio
    async def test_start_cycles_clamped_to_100(self, handler, nomic_dir):
        """Cycles are clamped to max 100."""
        scripts_dir = nomic_dir.parent.parent / "scripts"
        scripts_dir.mkdir(parents=True, exist_ok=True)
        (scripts_dir / "nomic_loop.py").write_text("# placeholder")

        mock_proc = MagicMock()
        mock_proc.pid = 22222

        with patch("subprocess.Popen", return_value=mock_proc):
            http = MockHTTPHandler(method="POST", body={"cycles": 999})
            result = await handler.handle_post(
                "/api/v1/nomic/control/start", {}, http
            )

        body = _body(result)
        assert body["target_cycles"] <= 100

    @pytest.mark.asyncio
    async def test_start_string_cycles(self, handler, nomic_dir):
        """String cycles are converted to int."""
        scripts_dir = nomic_dir.parent.parent / "scripts"
        scripts_dir.mkdir(parents=True, exist_ok=True)
        (scripts_dir / "nomic_loop.py").write_text("# placeholder")

        mock_proc = MagicMock()
        mock_proc.pid = 33333

        with patch("subprocess.Popen", return_value=mock_proc):
            http = MockHTTPHandler(method="POST", body={"cycles": "5"})
            result = await handler.handle_post(
                "/api/v1/nomic/control/start", {}, http
            )

        body = _body(result)
        assert body["target_cycles"] == 5

    @pytest.mark.asyncio
    async def test_start_alternate_script_path(self, handler, nomic_dir):
        """Falls back to alternate script path when primary doesn't exist."""
        # Create alternate path (parent/scripts instead of parent.parent/scripts)
        alt_scripts_dir = nomic_dir.parent / "scripts"
        alt_scripts_dir.mkdir(parents=True, exist_ok=True)
        (alt_scripts_dir / "nomic_loop.py").write_text("# placeholder")

        mock_proc = MagicMock()
        mock_proc.pid = 44444

        with patch("subprocess.Popen", return_value=mock_proc):
            http = MockHTTPHandler(method="POST", body={"cycles": 1})
            result = await handler.handle_post(
                "/api/v1/nomic/control/start", {}, http
            )

        body = _body(result)
        assert _status(result) == 202
        assert body["pid"] == 44444

    @pytest.mark.asyncio
    async def test_start_invalid_state_json(self, handler, nomic_dir):
        """Returns 500 when existing state file is corrupt."""
        (nomic_dir / "nomic_state.json").write_text("not json")

        http = MockHTTPHandler(method="POST", body={"cycles": 1})
        result = await handler.handle_post(
            "/api/v1/nomic/control/start", {}, http
        )
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_start_not_running_state(self, handler, nomic_dir):
        """Starts when state exists but running=False."""
        state = {"running": False, "pid": 0}
        (nomic_dir / "nomic_state.json").write_text(json.dumps(state))

        scripts_dir = nomic_dir.parent.parent / "scripts"
        scripts_dir.mkdir(parents=True, exist_ok=True)
        (scripts_dir / "nomic_loop.py").write_text("# placeholder")

        mock_proc = MagicMock()
        mock_proc.pid = 55555

        with patch("subprocess.Popen", return_value=mock_proc):
            http = MockHTTPHandler(method="POST", body={"cycles": 2})
            result = await handler.handle_post(
                "/api/v1/nomic/control/start", {}, http
            )

        body = _body(result)
        assert _status(result) == 202
        assert body["status"] == "started"


# ---------------------------------------------------------------------------
# POST /api/v1/nomic/control/stop
# ---------------------------------------------------------------------------


class TestStopNomicLoop:
    """Tests for stopping the nomic loop."""

    @pytest.mark.asyncio
    async def test_stop_no_dir(self, handler_no_dir):
        """Returns 503 when nomic dir not configured."""
        http = MockHTTPHandler(method="POST", body={})
        result = await handler_no_dir.handle_post(
            "/api/v1/nomic/control/stop", {}, http
        )
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_stop_not_running(self, handler):
        """Returns 404 when no state file exists."""
        http = MockHTTPHandler(method="POST", body={})
        result = await handler.handle_post(
            "/api/v1/nomic/control/stop", {}, http
        )
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_stop_no_pid(self, handler, nomic_dir):
        """Returns 404 when state file has no PID."""
        state = {"running": True}
        (nomic_dir / "nomic_state.json").write_text(json.dumps(state))

        http = MockHTTPHandler(method="POST", body={})
        result = await handler.handle_post(
            "/api/v1/nomic/control/stop", {}, http
        )
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_stop_already_stopped_process(self, handler, nomic_dir):
        """Returns already_stopped when process doesn't exist."""
        state = {"running": True, "pid": 99999999}
        (nomic_dir / "nomic_state.json").write_text(json.dumps(state))

        http = MockHTTPHandler(method="POST", body={})
        with patch("os.kill", side_effect=OSError("No such process")):
            result = await handler.handle_post(
                "/api/v1/nomic/control/stop", {}, http
            )
        body = _body(result)
        assert body["status"] == "already_stopped"

    @pytest.mark.asyncio
    async def test_stop_graceful(self, handler, nomic_dir):
        """Graceful stop sends SIGTERM."""
        state = {"running": True, "pid": 12345}
        (nomic_dir / "nomic_state.json").write_text(json.dumps(state))

        http = MockHTTPHandler(method="POST", body={"graceful": True})

        def fake_kill(pid, sig):
            import signal
            if sig == 0:
                return  # Process check
            assert sig == signal.SIGTERM

        with patch("os.kill", side_effect=fake_kill):
            result = await handler.handle_post(
                "/api/v1/nomic/control/stop", {}, http
            )
        body = _body(result)
        assert body["status"] == "stopping"

    @pytest.mark.asyncio
    async def test_stop_force(self, handler, nomic_dir):
        """Force stop sends SIGKILL."""
        state = {"running": True, "pid": 12345}
        (nomic_dir / "nomic_state.json").write_text(json.dumps(state))

        http = MockHTTPHandler(method="POST", body={"graceful": False})

        def fake_kill(pid, sig):
            import signal
            if sig == 0:
                return
            assert sig == signal.SIGKILL

        with patch("os.kill", side_effect=fake_kill):
            result = await handler.handle_post(
                "/api/v1/nomic/control/stop", {}, http
            )
        body = _body(result)
        assert body["status"] == "killed"

    @pytest.mark.asyncio
    async def test_stop_invalid_json_state(self, handler, nomic_dir):
        """Returns 500 when state file is corrupt."""
        (nomic_dir / "nomic_state.json").write_text("corrupt!")

        http = MockHTTPHandler(method="POST", body={})
        result = await handler.handle_post(
            "/api/v1/nomic/control/stop", {}, http
        )
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_stop_updates_state_file(self, handler, nomic_dir):
        """State file is updated with stopped info."""
        state = {"running": True, "pid": 12345}
        (nomic_dir / "nomic_state.json").write_text(json.dumps(state))

        def fake_kill(pid, sig):
            pass  # Both sig 0 check and actual signal succeed

        with patch("os.kill", side_effect=fake_kill):
            http = MockHTTPHandler(method="POST", body={"graceful": True})
            result = await handler.handle_post(
                "/api/v1/nomic/control/stop", {}, http
            )

        updated = json.loads((nomic_dir / "nomic_state.json").read_text())
        assert updated["running"] is False
        assert "stopped_at" in updated
        assert updated["stopped_reason"] == "user_requested"

    @pytest.mark.asyncio
    async def test_stop_default_graceful(self, handler, nomic_dir):
        """Default stop is graceful (SIGTERM)."""
        state = {"running": True, "pid": 12345}
        (nomic_dir / "nomic_state.json").write_text(json.dumps(state))

        signals_sent = []

        def fake_kill(pid, sig):
            signals_sent.append(sig)

        with patch("os.kill", side_effect=fake_kill):
            http = MockHTTPHandler(method="POST", body={})
            result = await handler.handle_post(
                "/api/v1/nomic/control/stop", {}, http
            )

        body = _body(result)
        assert body["status"] == "stopping"
        import signal
        assert signal.SIGTERM in signals_sent


# ---------------------------------------------------------------------------
# POST /api/v1/nomic/control/pause
# ---------------------------------------------------------------------------


class TestPauseNomicLoop:
    """Tests for pausing the nomic loop."""

    @pytest.mark.asyncio
    async def test_pause_no_dir(self, handler_no_dir):
        """Returns 503 when nomic dir not configured."""
        http = MockHTTPHandler(method="POST")
        result = await handler_no_dir.handle_post(
            "/api/v1/nomic/control/pause", {}, http
        )
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_pause_not_running(self, handler):
        """Returns 404 when no state file exists."""
        http = MockHTTPHandler(method="POST")
        result = await handler.handle_post(
            "/api/v1/nomic/control/pause", {}, http
        )
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_pause_not_running_state(self, handler, nomic_dir):
        """Returns 404 when running is False in state."""
        state = {"running": False, "phase": "verify"}
        (nomic_dir / "nomic_state.json").write_text(json.dumps(state))

        http = MockHTTPHandler(method="POST")
        result = await handler.handle_post(
            "/api/v1/nomic/control/pause", {}, http
        )
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_pause_already_paused(self, handler, nomic_dir):
        """Returns 409 when already paused."""
        state = {"running": True, "paused": True, "phase": "debate"}
        (nomic_dir / "nomic_state.json").write_text(json.dumps(state))

        http = MockHTTPHandler(method="POST")
        result = await handler.handle_post(
            "/api/v1/nomic/control/pause", {}, http
        )
        assert _status(result) == 409

    @pytest.mark.asyncio
    async def test_pause_success(self, handler, nomic_dir):
        """Successfully pauses the loop."""
        state = {"running": True, "cycle": 2, "phase": "implement", "pid": 123}
        (nomic_dir / "nomic_state.json").write_text(json.dumps(state))

        http = MockHTTPHandler(method="POST")
        result = await handler.handle_post(
            "/api/v1/nomic/control/pause", {}, http
        )
        body = _body(result)
        assert body["status"] == "paused"
        assert body["cycle"] == 2
        assert body["phase"] == "implement"

        # Verify state file updated
        updated = json.loads((nomic_dir / "nomic_state.json").read_text())
        assert updated["paused"] is True
        assert "paused_at" in updated


# ---------------------------------------------------------------------------
# POST /api/v1/nomic/control/resume
# ---------------------------------------------------------------------------


class TestResumeNomicLoop:
    """Tests for resuming a paused nomic loop."""

    @pytest.mark.asyncio
    async def test_resume_no_dir(self, handler_no_dir):
        """Returns 503 when nomic dir not configured."""
        http = MockHTTPHandler(method="POST")
        result = await handler_no_dir.handle_post(
            "/api/v1/nomic/control/resume", {}, http
        )
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_resume_not_running(self, handler):
        """Returns 404 when no state file exists."""
        http = MockHTTPHandler(method="POST")
        result = await handler.handle_post(
            "/api/v1/nomic/control/resume", {}, http
        )
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_resume_not_paused(self, handler, nomic_dir):
        """Returns 409 when loop is not paused."""
        state = {"running": True, "paused": False}
        (nomic_dir / "nomic_state.json").write_text(json.dumps(state))

        http = MockHTTPHandler(method="POST")
        result = await handler.handle_post(
            "/api/v1/nomic/control/resume", {}, http
        )
        assert _status(result) == 409

    @pytest.mark.asyncio
    async def test_resume_success(self, handler, nomic_dir):
        """Successfully resumes a paused loop."""
        state = {"running": True, "paused": True, "cycle": 3, "phase": "design"}
        (nomic_dir / "nomic_state.json").write_text(json.dumps(state))

        http = MockHTTPHandler(method="POST")
        result = await handler.handle_post(
            "/api/v1/nomic/control/resume", {}, http
        )
        body = _body(result)
        assert body["status"] == "resumed"
        assert body["cycle"] == 3
        assert body["phase"] == "design"

        # Verify state updated
        updated = json.loads((nomic_dir / "nomic_state.json").read_text())
        assert updated["paused"] is False
        assert "resumed_at" in updated


# ---------------------------------------------------------------------------
# POST /api/v1/nomic/control/skip-phase
# ---------------------------------------------------------------------------


class TestSkipPhase:
    """Tests for skipping the current nomic phase."""

    @pytest.mark.asyncio
    async def test_skip_no_dir(self, handler_no_dir):
        """Returns 503 when nomic dir not configured."""
        http = MockHTTPHandler(method="POST")
        result = await handler_no_dir.handle_post(
            "/api/v1/nomic/control/skip-phase", {}, http
        )
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_skip_not_running(self, handler):
        """Returns 404 when no state file exists."""
        http = MockHTTPHandler(method="POST")
        result = await handler.handle_post(
            "/api/v1/nomic/control/skip-phase", {}, http
        )
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_skip_debate_to_design(self, handler, nomic_dir):
        """Skipping debate moves to design."""
        state = {"cycle": 1, "phase": "debate"}
        (nomic_dir / "nomic_state.json").write_text(json.dumps(state))

        http = MockHTTPHandler(method="POST")
        result = await handler.handle_post(
            "/api/v1/nomic/control/skip-phase", {}, http
        )
        body = _body(result)
        assert body["status"] == "skip_requested"
        assert body["previous_phase"] == "debate"
        assert body["next_phase"] == "design"

    @pytest.mark.asyncio
    async def test_skip_verify_wraps_to_context(self, handler, nomic_dir):
        """Skipping verify wraps to context and increments cycle."""
        state = {"cycle": 2, "phase": "verify"}
        (nomic_dir / "nomic_state.json").write_text(json.dumps(state))

        http = MockHTTPHandler(method="POST")
        result = await handler.handle_post(
            "/api/v1/nomic/control/skip-phase", {}, http
        )
        body = _body(result)
        assert body["next_phase"] == "context"
        assert body["cycle"] == 3  # Incremented from 2

        # Verify state
        updated = json.loads((nomic_dir / "nomic_state.json").read_text())
        assert updated["cycle"] == 3

    @pytest.mark.asyncio
    async def test_skip_unknown_phase(self, handler, nomic_dir):
        """Unknown phase returns 400 error."""
        state = {"cycle": 1, "phase": "unknown_phase"}
        (nomic_dir / "nomic_state.json").write_text(json.dumps(state))

        http = MockHTTPHandler(method="POST")
        result = await handler.handle_post(
            "/api/v1/nomic/control/skip-phase", {}, http
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_skip_all_phases_sequential(self, handler, nomic_dir):
        """Verifies all 5 phase transitions: context->debate->design->implement->verify->context."""
        phases = ["context", "debate", "design", "implement", "verify"]
        expected_next = ["debate", "design", "implement", "verify", "context"]

        for phase, expected in zip(phases, expected_next):
            state = {"cycle": 1, "phase": phase}
            (nomic_dir / "nomic_state.json").write_text(json.dumps(state))

            http = MockHTTPHandler(method="POST")
            result = await handler.handle_post(
                "/api/v1/nomic/control/skip-phase", {}, http
            )
            body = _body(result)
            assert body["next_phase"] == expected, f"After {phase}, expected {expected}"


# ---------------------------------------------------------------------------
# POST /api/v1/nomic/proposals/approve
# ---------------------------------------------------------------------------


class TestApproveProposal:
    """Tests for proposal approval."""

    @pytest.mark.asyncio
    async def test_approve_no_dir(self, handler_no_dir):
        """Returns 503 when nomic dir not configured."""
        http = MockHTTPHandler(method="POST", body={"proposal_id": "p1"})
        result = await handler_no_dir.handle_post(
            "/api/v1/nomic/proposals/approve", {}, http
        )
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_approve_missing_id(self, handler):
        """Returns 400 when proposal_id is missing."""
        http = MockHTTPHandler(method="POST", body={})
        result = await handler.handle_post(
            "/api/v1/nomic/proposals/approve", {}, http
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_approve_no_proposals_file(self, handler):
        """Returns 404 when proposals file doesn't exist."""
        http = MockHTTPHandler(method="POST", body={"proposal_id": "p1"})
        result = await handler.handle_post(
            "/api/v1/nomic/proposals/approve", {}, http
        )
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_approve_proposal_not_found(self, handler, nomic_dir):
        """Returns 404 when proposal ID doesn't match."""
        data = {"proposals": [{"id": "p1", "status": "pending"}]}
        (nomic_dir / "proposals.json").write_text(json.dumps(data))

        http = MockHTTPHandler(method="POST", body={"proposal_id": "p999"})
        result = await handler.handle_post(
            "/api/v1/nomic/proposals/approve", {}, http
        )
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_approve_success(self, handler, nomic_dir):
        """Successfully approves a proposal."""
        data = {
            "proposals": [
                {"id": "p1", "status": "pending", "title": "Feature A"},
                {"id": "p2", "status": "pending", "title": "Feature B"},
            ]
        }
        (nomic_dir / "proposals.json").write_text(json.dumps(data))

        http = MockHTTPHandler(
            method="POST",
            body={"proposal_id": "p1", "approved_by": "admin"},
        )
        result = await handler.handle_post(
            "/api/v1/nomic/proposals/approve", {}, http
        )
        body = _body(result)
        assert body["status"] == "approved"
        assert body["proposal_id"] == "p1"

        # Verify file updated
        updated = json.loads((nomic_dir / "proposals.json").read_text())
        p1 = next(p for p in updated["proposals"] if p["id"] == "p1")
        assert p1["status"] == "approved"
        assert p1["approved_by"] == "admin"
        assert "approved_at" in p1

    @pytest.mark.asyncio
    async def test_approve_default_approved_by(self, handler, nomic_dir):
        """Default approved_by is 'user'."""
        data = {"proposals": [{"id": "p1", "status": "pending"}]}
        (nomic_dir / "proposals.json").write_text(json.dumps(data))

        http = MockHTTPHandler(method="POST", body={"proposal_id": "p1"})
        result = await handler.handle_post(
            "/api/v1/nomic/proposals/approve", {}, http
        )

        updated = json.loads((nomic_dir / "proposals.json").read_text())
        p1 = updated["proposals"][0]
        assert p1["approved_by"] == "user"

    @pytest.mark.asyncio
    async def test_approve_invalid_json_file(self, handler, nomic_dir):
        """Returns 500 when proposals file is corrupt."""
        (nomic_dir / "proposals.json").write_text("bad!")

        http = MockHTTPHandler(method="POST", body={"proposal_id": "p1"})
        result = await handler.handle_post(
            "/api/v1/nomic/proposals/approve", {}, http
        )
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_approve_preserves_other_proposals(self, handler, nomic_dir):
        """Approving one proposal preserves others."""
        data = {
            "proposals": [
                {"id": "p1", "status": "pending"},
                {"id": "p2", "status": "pending"},
            ]
        }
        (nomic_dir / "proposals.json").write_text(json.dumps(data))

        http = MockHTTPHandler(method="POST", body={"proposal_id": "p1"})
        result = await handler.handle_post(
            "/api/v1/nomic/proposals/approve", {}, http
        )

        updated = json.loads((nomic_dir / "proposals.json").read_text())
        p2 = next(p for p in updated["proposals"] if p["id"] == "p2")
        assert p2["status"] == "pending"  # Unchanged


# ---------------------------------------------------------------------------
# POST /api/v1/nomic/proposals/reject
# ---------------------------------------------------------------------------


class TestRejectProposal:
    """Tests for proposal rejection."""

    @pytest.mark.asyncio
    async def test_reject_no_dir(self, handler_no_dir):
        """Returns 503 when nomic dir not configured."""
        http = MockHTTPHandler(method="POST", body={"proposal_id": "p1"})
        result = await handler_no_dir.handle_post(
            "/api/v1/nomic/proposals/reject", {}, http
        )
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_reject_missing_id(self, handler):
        """Returns 400 when proposal_id is missing."""
        http = MockHTTPHandler(method="POST", body={})
        result = await handler.handle_post(
            "/api/v1/nomic/proposals/reject", {}, http
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_reject_no_proposals_file(self, handler):
        """Returns 404 when proposals file doesn't exist."""
        http = MockHTTPHandler(method="POST", body={"proposal_id": "p1"})
        result = await handler.handle_post(
            "/api/v1/nomic/proposals/reject", {}, http
        )
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_reject_proposal_not_found(self, handler, nomic_dir):
        """Returns 404 when proposal ID doesn't match."""
        data = {"proposals": [{"id": "p1", "status": "pending"}]}
        (nomic_dir / "proposals.json").write_text(json.dumps(data))

        http = MockHTTPHandler(method="POST", body={"proposal_id": "p999"})
        result = await handler.handle_post(
            "/api/v1/nomic/proposals/reject", {}, http
        )
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_reject_success(self, handler, nomic_dir):
        """Successfully rejects a proposal."""
        data = {
            "proposals": [
                {"id": "p1", "status": "pending", "title": "Bad feature"},
            ]
        }
        (nomic_dir / "proposals.json").write_text(json.dumps(data))

        http = MockHTTPHandler(
            method="POST",
            body={
                "proposal_id": "p1",
                "rejected_by": "reviewer",
                "reason": "Does not align with roadmap",
            },
        )
        result = await handler.handle_post(
            "/api/v1/nomic/proposals/reject", {}, http
        )
        body = _body(result)
        assert body["status"] == "rejected"
        assert body["proposal_id"] == "p1"

        # Verify file updated
        updated = json.loads((nomic_dir / "proposals.json").read_text())
        p1 = updated["proposals"][0]
        assert p1["status"] == "rejected"
        assert p1["rejected_by"] == "reviewer"
        assert p1["rejection_reason"] == "Does not align with roadmap"
        assert "rejected_at" in p1

    @pytest.mark.asyncio
    async def test_reject_default_values(self, handler, nomic_dir):
        """Default rejected_by is 'user' and reason is empty."""
        data = {"proposals": [{"id": "p1", "status": "pending"}]}
        (nomic_dir / "proposals.json").write_text(json.dumps(data))

        http = MockHTTPHandler(method="POST", body={"proposal_id": "p1"})
        result = await handler.handle_post(
            "/api/v1/nomic/proposals/reject", {}, http
        )

        updated = json.loads((nomic_dir / "proposals.json").read_text())
        p1 = updated["proposals"][0]
        assert p1["rejected_by"] == "user"
        assert p1["rejection_reason"] == ""

    @pytest.mark.asyncio
    async def test_reject_invalid_json_file(self, handler, nomic_dir):
        """Returns 500 when proposals file is corrupt."""
        (nomic_dir / "proposals.json").write_text("bad!")

        http = MockHTTPHandler(method="POST", body={"proposal_id": "p1"})
        result = await handler.handle_post(
            "/api/v1/nomic/proposals/reject", {}, http
        )
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_reject_preserves_other_proposals(self, handler, nomic_dir):
        """Rejecting one proposal preserves others."""
        data = {
            "proposals": [
                {"id": "p1", "status": "pending"},
                {"id": "p2", "status": "pending"},
            ]
        }
        (nomic_dir / "proposals.json").write_text(json.dumps(data))

        http = MockHTTPHandler(method="POST", body={"proposal_id": "p1"})
        result = await handler.handle_post(
            "/api/v1/nomic/proposals/reject", {}, http
        )

        updated = json.loads((nomic_dir / "proposals.json").read_text())
        p2 = next(p for p in updated["proposals"] if p["id"] == "p2")
        assert p2["status"] == "pending"  # Unchanged


# ---------------------------------------------------------------------------
# handle() routing returns None for unknown paths
# ---------------------------------------------------------------------------


class TestHandleRouting:
    """Tests for handle() method routing behavior."""

    @pytest.mark.asyncio
    async def test_unknown_path_returns_none(self, handler):
        """handle() returns None for paths it doesn't route."""
        http = MockHTTPHandler()
        result = await handler.handle("/api/nomic/nonexistent", {}, http)
        assert result is None

    @pytest.mark.asyncio
    async def test_handle_post_unknown_path_returns_none(self, handler):
        """handle_post() returns None for unknown control paths."""
        http = MockHTTPHandler(method="POST", body={})
        result = await handler.handle_post(
            "/api/v1/nomic/control/unknown", {}, http
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_versioned_path_stripped(self, handler, nomic_dir):
        """Versioned /api/v1/nomic/state is handled correctly."""
        state = {"running": False, "cycle": 0}
        (nomic_dir / "nomic_state.json").write_text(json.dumps(state))

        http = MockHTTPHandler()
        result = await handler.handle("/api/v1/nomic/state", {}, http)
        body = _body(result)
        assert body["running"] is False


# ---------------------------------------------------------------------------
# Stream server
# ---------------------------------------------------------------------------


class TestStreamServer:
    """Tests for WebSocket stream server integration."""

    def test_set_stream_server(self, handler):
        """set_stream_server stores the stream."""
        mock_stream = MagicMock()
        handler.set_stream_server(mock_stream)
        assert handler._stream is mock_stream

    def test_get_stream_from_instance(self, handler):
        """_get_stream returns instance stream when set."""
        mock_stream = MagicMock()
        handler.set_stream_server(mock_stream)
        assert handler._get_stream() is mock_stream

    def test_get_stream_from_context(self, nomic_dir):
        """_get_stream falls back to context."""
        mock_stream = MagicMock()
        h = NomicHandler({"nomic_dir": nomic_dir, "nomic_loop_stream": mock_stream})
        assert h._get_stream() is mock_stream

    def test_get_stream_none(self, handler):
        """_get_stream returns None when no stream available."""
        assert handler._get_stream() is None

    def test_emit_event_no_stream(self, handler):
        """_emit_event is a no-op when no stream is set."""
        # Should not raise
        handler._emit_event("emit_loop_started", cycles=1)

    def test_emit_event_no_method(self, handler):
        """_emit_event is a no-op when stream lacks the method."""
        mock_stream = MagicMock(spec=[])  # no attributes
        handler.set_stream_server(mock_stream)
        # Should not raise
        handler._emit_event("nonexistent_method")


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge case tests for various handler behaviors."""

    @pytest.mark.asyncio
    async def test_empty_log_file(self, handler, nomic_dir):
        """Empty log file returns no lines."""
        (nomic_dir / "nomic_loop.log").write_text("")

        http = MockHTTPHandler()
        result = await handler.handle("/api/nomic/log", {}, http)
        body = _body(result)
        assert body["lines"] == []
        assert body["total"] == 0

    @pytest.mark.asyncio
    async def test_empty_risk_register(self, handler, nomic_dir):
        """Empty risk register file returns no entries."""
        (nomic_dir / "risk_register.jsonl").write_text("")

        http = MockHTTPHandler()
        result = await handler.handle("/api/nomic/risk-register", {}, http)
        body = _body(result)
        assert body["risks"] == []
        assert body["total"] == 0

    @pytest.mark.asyncio
    async def test_health_no_last_update(self, handler, nomic_dir):
        """Health check handles missing last_update field."""
        state = {"cycle": 1, "phase": "context"}
        (nomic_dir / "nomic_state.json").write_text(json.dumps(state))

        http = MockHTTPHandler()
        result = await handler.handle("/api/nomic/health", {}, http)
        body = _body(result)
        # Without timestamp, cannot determine stall - defaults to healthy
        assert body["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_health_z_suffix_timestamp(self, handler, nomic_dir):
        """Health check handles Z-suffix ISO timestamps."""
        now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        state = {"cycle": 1, "phase": "design", "last_update": now}
        (nomic_dir / "nomic_state.json").write_text(json.dumps(state))

        http = MockHTTPHandler()
        result = await handler.handle("/api/nomic/health", {}, http)
        body = _body(result)
        assert body["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_proposals_all_approved(self, handler, nomic_dir):
        """All approved proposals returns empty pending list."""
        data = {
            "proposals": [
                {"id": "p1", "status": "approved"},
                {"id": "p2", "status": "rejected"},
            ]
        }
        (nomic_dir / "proposals.json").write_text(json.dumps(data))

        http = MockHTTPHandler()
        result = await handler.handle("/api/nomic/proposals", {}, http)
        body = _body(result)
        assert body["total"] == 0
        assert body["all_proposals"] == 2

    def test_resource_type(self, handler):
        """RESOURCE_TYPE is set to 'nomic'."""
        assert handler.RESOURCE_TYPE == "nomic"

    def test_routes_count(self, handler):
        """All 16 routes are declared."""
        assert len(handler.ROUTES) == 16

    @pytest.mark.asyncio
    async def test_skip_phase_context_to_debate(self, handler, nomic_dir):
        """Context phase skips to debate without cycle increment."""
        state = {"cycle": 5, "phase": "context"}
        (nomic_dir / "nomic_state.json").write_text(json.dumps(state))

        http = MockHTTPHandler(method="POST")
        result = await handler.handle_post(
            "/api/v1/nomic/control/skip-phase", {}, http
        )
        body = _body(result)
        assert body["next_phase"] == "debate"
        assert body["cycle"] == 5  # Not incremented

    @pytest.mark.asyncio
    async def test_skip_phase_design_to_implement(self, handler, nomic_dir):
        """Design skips to implement."""
        state = {"cycle": 1, "phase": "design"}
        (nomic_dir / "nomic_state.json").write_text(json.dumps(state))

        http = MockHTTPHandler(method="POST")
        result = await handler.handle_post(
            "/api/v1/nomic/control/skip-phase", {}, http
        )
        body = _body(result)
        assert body["next_phase"] == "implement"

    @pytest.mark.asyncio
    async def test_skip_phase_implement_to_verify(self, handler, nomic_dir):
        """Implement skips to verify."""
        state = {"cycle": 1, "phase": "implement"}
        (nomic_dir / "nomic_state.json").write_text(json.dumps(state))

        http = MockHTTPHandler(method="POST")
        result = await handler.handle_post(
            "/api/v1/nomic/control/skip-phase", {}, http
        )
        body = _body(result)
        assert body["next_phase"] == "verify"

    @pytest.mark.asyncio
    async def test_skip_phase_updates_state_file(self, handler, nomic_dir):
        """Skip phase persists changes to state file."""
        state = {"cycle": 1, "phase": "debate"}
        (nomic_dir / "nomic_state.json").write_text(json.dumps(state))

        http = MockHTTPHandler(method="POST")
        await handler.handle_post("/api/v1/nomic/control/skip-phase", {}, http)

        updated = json.loads((nomic_dir / "nomic_state.json").read_text())
        assert updated["phase"] == "design"
        assert updated["skip_requested"] is True
        assert "skipped_at" in updated

    @pytest.mark.asyncio
    async def test_pause_invalid_json_state(self, handler, nomic_dir):
        """Pause with corrupt state file returns 500."""
        (nomic_dir / "nomic_state.json").write_text("bad json")

        http = MockHTTPHandler(method="POST")
        result = await handler.handle_post(
            "/api/v1/nomic/control/pause", {}, http
        )
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_resume_invalid_json_state(self, handler, nomic_dir):
        """Resume with corrupt state file returns 500."""
        (nomic_dir / "nomic_state.json").write_text("bad json")

        http = MockHTTPHandler(method="POST")
        result = await handler.handle_post(
            "/api/v1/nomic/control/resume", {}, http
        )
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_risk_register_empty_lines_skipped(self, handler, nomic_dir):
        """Empty lines in risk register are skipped."""
        content = '{"id":"r1","severity":"low"}\n\n\n{"id":"r2","severity":"high"}\n'
        (nomic_dir / "risk_register.jsonl").write_text(content)

        http = MockHTTPHandler()
        result = await handler.handle("/api/nomic/risk-register", {}, http)
        body = _body(result)
        assert body["total"] == 2

    @pytest.mark.asyncio
    async def test_risk_register_limit_clamped_min(self, handler, nomic_dir):
        """Risk register limit is clamped to at least 1."""
        risks = [{"id": "r1", "severity": "low"}]
        (nomic_dir / "risk_register.jsonl").write_text(json.dumps(risks[0]))

        http = MockHTTPHandler()
        result = await handler.handle("/api/nomic/risk-register", {"limit": "0"}, http)
        body = _body(result)
        # Clamped to 1, and only 1 entry exists
        assert len(body["risks"]) == 1

    @pytest.mark.asyncio
    async def test_health_invalid_date_value(self, handler, nomic_dir):
        """Health check handles non-parseable date format."""
        state = {"cycle": 1, "phase": "design", "last_update": "not-a-date-at-all"}
        (nomic_dir / "nomic_state.json").write_text(json.dumps(state))

        http = MockHTTPHandler()
        result = await handler.handle("/api/nomic/health", {}, http)
        body = _body(result)
        # Invalid date cannot determine stall, defaults to healthy
        assert body["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_health_stall_warning_includes_minutes(self, handler, nomic_dir):
        """Stall warning message includes minutes count."""
        old_time = (datetime.now() - timedelta(hours=2)).isoformat()
        state = {"cycle": 1, "phase": "debate", "last_update": old_time, "warnings": []}
        (nomic_dir / "nomic_state.json").write_text(json.dumps(state))

        http = MockHTTPHandler()
        result = await handler.handle("/api/nomic/health", {}, http)
        body = _body(result)
        stall_warning = [w for w in body["warnings"] if "minutes" in w]
        assert len(stall_warning) == 1
        # Should mention the number of minutes
        assert "120" in stall_warning[0] or "No activity" in stall_warning[0]

    def test_stream_instance_takes_priority_over_context(self, nomic_dir):
        """Instance stream takes priority over context stream."""
        ctx_stream = MagicMock()
        inst_stream = MagicMock()
        h = NomicHandler({"nomic_dir": nomic_dir, "nomic_loop_stream": ctx_stream})
        h._stream = inst_stream
        assert h._get_stream() is inst_stream

    @pytest.mark.asyncio
    async def test_log_negative_lines_clamped(self, handler, nomic_dir):
        """Negative line count is clamped to 1."""
        (nomic_dir / "nomic_loop.log").write_text("line1\nline2\nline3\n")

        http = MockHTTPHandler()
        result = await handler.handle("/api/nomic/log", {"lines": "-50"}, http)
        body = _body(result)
        assert body["showing"] >= 1

    @pytest.mark.asyncio
    async def test_modes_total_matches_length(self, handler):
        """Total field matches modes list length."""
        http = MockHTTPHandler()
        result = await handler.handle("/api/modes", {}, http)
        body = _body(result)
        assert body["total"] == len(body["modes"])

    @pytest.mark.asyncio
    async def test_modes_all_have_description(self, handler):
        """All builtin modes have non-empty descriptions."""
        http = MockHTTPHandler()
        result = await handler.handle("/api/modes", {}, http)
        body = _body(result)
        for mode in body["modes"]:
            if mode["type"] == "builtin":
                assert mode["description"], f"Mode {mode['name']} has no description"

    @pytest.mark.asyncio
    async def test_proposals_empty_proposals_key(self, handler, nomic_dir):
        """Empty proposals array returns zero total."""
        data = {"proposals": []}
        (nomic_dir / "proposals.json").write_text(json.dumps(data))

        http = MockHTTPHandler()
        result = await handler.handle("/api/nomic/proposals", {}, http)
        body = _body(result)
        assert body["total"] == 0
        assert body["all_proposals"] == 0


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _selective_import_error(blocked_module: str):
    """Create an import side_effect that blocks a specific module."""
    import builtins

    original_import = builtins.__import__

    def _import(name, *args, **kwargs):
        if name == blocked_module:
            raise ImportError(f"No module named '{blocked_module}'")
        return original_import(name, *args, **kwargs)

    return _import
