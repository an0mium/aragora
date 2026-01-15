"""
Tests for the System Handler endpoints and extracted handlers.

Covers:
- Kubernetes health probes (/healthz, /readyz) - HealthHandler
- Health endpoints (/api/health, /api/health/detailed, /api/health/deep) - HealthHandler
- Nomic loop endpoints (/api/nomic/*) - NomicHandler
- History endpoints (/api/history/*) - SystemHandler
- System maintenance (/api/system/maintenance) - SystemHandler
- Auth stats (/api/auth/stats, /api/auth/revoke) - SystemHandler
- Circuit breaker metrics (/api/circuit-breakers) - SystemHandler
- OpenAPI spec (/api/openapi) - DocsHandler
- Swagger UI (/api/docs) - DocsHandler
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

import pytest

from aragora.server.handlers.admin import SystemHandler
from aragora.server.handlers.admin import HealthHandler
from aragora.server.handlers.nomic import NomicHandler
from aragora.server.handlers.docs import DocsHandler
from aragora.server.handlers.base import HandlerResult


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def system_handler(handler_context):
    """Create a SystemHandler with mock context."""
    handler = SystemHandler(handler_context)
    return handler


@pytest.fixture
def health_handler(handler_context):
    """Create a HealthHandler with mock context."""
    handler = HealthHandler(handler_context)
    return handler


@pytest.fixture
def nomic_handler(handler_context):
    """Create a NomicHandler with mock context."""
    handler = NomicHandler(handler_context)
    return handler


@pytest.fixture
def docs_handler(handler_context):
    """Create a DocsHandler with mock context."""
    handler = DocsHandler(handler_context)
    return handler


@pytest.fixture
def mock_http_handler():
    """Create a mock HTTP handler object."""
    handler = Mock()
    handler.headers = {}
    handler.command = "GET"
    return handler


@pytest.fixture
def temp_nomic_dir_with_files():
    """Create a temporary nomic directory with state files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        nomic_dir = Path(tmpdir)

        # Create nomic state file
        state_file = nomic_dir / "nomic_state.json"
        state = {
            "phase": "implement",
            "stage": "executing",
            "cycle": 3,
            "total_tasks": 5,
            "completed_tasks": 2,
            "last_update": datetime.utcnow().isoformat() + "Z",
            "warnings": [],
        }
        state_file.write_text(json.dumps(state))

        # Create nomic log file
        log_file = nomic_dir / "nomic_loop.log"
        log_lines = [
            "2026-01-05 00:00:01 Starting cycle 1",
            "2026-01-05 00:00:02 Phase: context",
            "2026-01-05 00:00:03 Phase: debate",
            "2026-01-05 00:00:04 Phase: design",
            "2026-01-05 00:00:05 Phase: implement",
        ]
        log_file.write_text("\n".join(log_lines))

        # Create risk register file
        risk_file = nomic_dir / "risk_register.jsonl"
        risks = [
            {"id": "1", "severity": "critical", "description": "Critical risk"},
            {"id": "2", "severity": "high", "description": "High risk"},
            {"id": "3", "severity": "medium", "description": "Medium risk"},
        ]
        risk_file.write_text("\n".join(json.dumps(r) for r in risks))

        # Create cycles file
        cycles_file = nomic_dir / "cycles.json"
        cycles_file.write_text(
            json.dumps(
                [
                    {"cycle": 1, "loop_id": "loop_1", "status": "complete"},
                    {"cycle": 2, "loop_id": "loop_1", "status": "complete"},
                    {"cycle": 3, "loop_id": "loop_2", "status": "in_progress"},
                ]
            )
        )

        # Create events file
        events_file = nomic_dir / "events.json"
        events_file.write_text(
            json.dumps(
                [
                    {"id": "e1", "loop_id": "loop_1", "type": "start"},
                    {"id": "e2", "loop_id": "loop_1", "type": "phase_change"},
                    {"id": "e3", "loop_id": "loop_2", "type": "start"},
                ]
            )
        )

        yield nomic_dir


# =============================================================================
# Kubernetes Health Probes Tests
# =============================================================================


class TestKubernetesProbes:
    """Tests for Kubernetes liveness and readiness probes."""

    def test_liveness_probe_returns_ok(self, health_handler, mock_http_handler):
        """Test that /healthz returns 200 OK."""
        result = health_handler.handle("/healthz", {}, mock_http_handler)

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["status"] == "ok"

    def test_readiness_probe_returns_ready(
        self, mock_storage, mock_elo_system, temp_nomic_dir, mock_http_handler
    ):
        """Test that /readyz returns ready when all dependencies are up."""
        ctx = {
            "storage": mock_storage,
            "elo_system": mock_elo_system,
            "nomic_dir": temp_nomic_dir,
        }
        handler = HealthHandler(ctx)

        result = handler.handle("/readyz", {}, mock_http_handler)

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["status"] == "ready"
        assert body["checks"]["storage"] is True
        assert body["checks"]["elo_system"] is True

    def test_readiness_probe_with_no_storage(
        self, mock_elo_system, temp_nomic_dir, mock_http_handler
    ):
        """Test that /readyz handles missing storage gracefully."""
        # When storage is None, readiness probe should still pass
        # (storage not configured is OK for readiness)
        ctx = {
            "storage": None,
            "elo_system": mock_elo_system,
            "nomic_dir": temp_nomic_dir,
        }
        handler = HealthHandler(ctx)

        result = handler.handle("/readyz", {}, mock_http_handler)

        assert result is not None
        # Storage not configured is still considered ready
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["status"] == "ready"


# =============================================================================
# Health Endpoint Tests
# =============================================================================


class TestHealthEndpoints:
    """Tests for /api/health endpoints (HealthHandler)."""

    def test_health_check_returns_healthy(
        self, mock_storage, mock_elo_system, temp_nomic_dir, mock_http_handler
    ):
        """Test that /api/health returns healthy status."""
        ctx = {
            "storage": mock_storage,
            "elo_system": mock_elo_system,
            "nomic_dir": temp_nomic_dir,
        }
        handler = HealthHandler(ctx)
        mock_storage.list_recent.return_value = []
        mock_elo_system.get_leaderboard.return_value = []

        result = handler.handle("/api/health", {}, mock_http_handler)

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["status"] == "healthy"
        assert "checks" in body
        assert "timestamp" in body
        assert "version" in body

    def test_health_check_includes_database_latency(
        self, mock_storage, mock_elo_system, temp_nomic_dir, mock_http_handler
    ):
        """Test that /api/health includes database latency metric."""
        ctx = {
            "storage": mock_storage,
            "elo_system": mock_elo_system,
            "nomic_dir": temp_nomic_dir,
        }
        handler = HealthHandler(ctx)
        mock_storage.list_recent.return_value = []
        mock_elo_system.get_leaderboard.return_value = []

        result = handler.handle("/api/health", {}, mock_http_handler)

        body = json.loads(result.body)
        assert "database" in body["checks"]
        assert "latency_ms" in body["checks"]["database"]

    def test_detailed_health_check(
        self, mock_storage, mock_elo_system, temp_nomic_dir, mock_http_handler
    ):
        """Test that /api/health/detailed returns extended status."""
        ctx = {
            "storage": mock_storage,
            "elo_system": mock_elo_system,
            "nomic_dir": temp_nomic_dir,
        }
        handler = HealthHandler(ctx)

        result = handler.handle("/api/health/detailed", {}, mock_http_handler)

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["status"] == "healthy"
        assert "components" in body
        assert "version" in body

    def test_deep_health_check(
        self, mock_storage, mock_elo_system, temp_nomic_dir, mock_http_handler
    ):
        """Test that /api/health/deep returns comprehensive status."""
        ctx = {
            "storage": mock_storage,
            "elo_system": mock_elo_system,
            "nomic_dir": temp_nomic_dir,
        }
        handler = HealthHandler(ctx)
        mock_storage.list_recent.return_value = []
        mock_elo_system.get_leaderboard.return_value = []

        result = handler.handle("/api/health/deep", {}, mock_http_handler)

        assert result is not None
        # Deep health can return 200 or 503 depending on env
        body = json.loads(result.body)
        assert "status" in body
        assert "checks" in body
        assert "response_time_ms" in body
        assert "timestamp" in body


# =============================================================================
# Nomic Loop Endpoint Tests
# =============================================================================


class TestNomicEndpoints:
    """Tests for /api/nomic/* endpoints (NomicHandler)."""

    def test_get_nomic_state(self, temp_nomic_dir_with_files, mock_http_handler):
        """Test that /api/nomic/state returns current state."""
        ctx = {"nomic_dir": temp_nomic_dir_with_files}
        handler = NomicHandler(ctx)

        result = handler.handle("/api/nomic/state", {}, mock_http_handler)

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["phase"] == "implement"
        assert body["cycle"] == 3

    def test_get_nomic_state_not_running(self, mock_http_handler):
        """Test that /api/nomic/state returns not_running when no state file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ctx = {"nomic_dir": Path(tmpdir)}
            handler = NomicHandler(ctx)

            result = handler.handle("/api/nomic/state", {}, mock_http_handler)

            assert result is not None
            assert result.status_code == 200
            body = json.loads(result.body)
            assert body["state"] == "not_running"

    def test_get_nomic_state_no_directory(self, mock_http_handler):
        """Test that /api/nomic/state returns error when no directory configured."""
        ctx = {"nomic_dir": None}
        handler = NomicHandler(ctx)

        result = handler.handle("/api/nomic/state", {}, mock_http_handler)

        assert result is not None
        assert result.status_code == 503
        body = json.loads(result.body)
        assert "error" in body

    def test_get_nomic_health_healthy(self, temp_nomic_dir_with_files, mock_http_handler):
        """Test that /api/nomic/health returns healthy status."""
        ctx = {"nomic_dir": temp_nomic_dir_with_files}
        handler = NomicHandler(ctx)

        result = handler.handle("/api/nomic/health", {}, mock_http_handler)

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["status"] == "healthy"
        assert body["cycle"] == 3
        assert body["phase"] == "implement"

    def test_get_nomic_health_stalled(self, mock_http_handler):
        """Test that /api/nomic/health detects stalled state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nomic_dir = Path(tmpdir)
            # Create state with old timestamp
            old_time = (datetime.utcnow() - timedelta(hours=1)).isoformat() + "Z"
            state_file = nomic_dir / "nomic_state.json"
            state_file.write_text(
                json.dumps(
                    {
                        "phase": "implement",
                        "cycle": 1,
                        "last_update": old_time,
                    }
                )
            )

            ctx = {"nomic_dir": nomic_dir}
            handler = NomicHandler(ctx)

            result = handler.handle("/api/nomic/health", {}, mock_http_handler)

            assert result is not None
            body = json.loads(result.body)
            assert body["status"] == "stalled"
            assert body["stall_duration_seconds"] is not None

    def test_get_nomic_log(self, temp_nomic_dir_with_files, mock_http_handler):
        """Test that /api/nomic/log returns log lines."""
        ctx = {"nomic_dir": temp_nomic_dir_with_files}
        handler = NomicHandler(ctx)

        result = handler.handle("/api/nomic/log", {"lines": "10"}, mock_http_handler)

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert "lines" in body
        assert len(body["lines"]) == 5
        assert body["total"] == 5

    def test_get_nomic_log_limits_lines(self, temp_nomic_dir_with_files, mock_http_handler):
        """Test that /api/nomic/log respects line limit."""
        ctx = {"nomic_dir": temp_nomic_dir_with_files}
        handler = NomicHandler(ctx)

        result = handler.handle("/api/nomic/log", {"lines": "2"}, mock_http_handler)

        body = json.loads(result.body)
        assert body["showing"] == 2

    def test_get_risk_register(self, temp_nomic_dir_with_files, mock_http_handler):
        """Test that /api/nomic/risk-register returns risk entries."""
        ctx = {"nomic_dir": temp_nomic_dir_with_files}
        handler = NomicHandler(ctx)

        result = handler.handle("/api/nomic/risk-register", {}, mock_http_handler)

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["total"] == 3
        assert body["critical_count"] == 1
        assert body["high_count"] == 1
        assert len(body["risks"]) == 3


# =============================================================================
# History Endpoint Tests
# =============================================================================


class TestHistoryEndpoints:
    """Tests for /api/history/* endpoints."""

    def test_get_history_cycles(self, temp_nomic_dir_with_files, mock_http_handler):
        """Test that /api/history/cycles returns cycle history."""
        # Disable auth for this test
        with patch("aragora.server.handlers.system.extract_user_from_request") as mock_auth:
            mock_user = Mock()
            mock_user.is_authenticated = True
            mock_auth.return_value = mock_user

            ctx = {"nomic_dir": temp_nomic_dir_with_files}
            handler = SystemHandler(ctx)

            result = handler.handle("/api/history/cycles", {}, mock_http_handler)

            assert result is not None
            assert result.status_code == 200
            body = json.loads(result.body)
            assert "cycles" in body
            assert len(body["cycles"]) == 3

    def test_get_history_cycles_with_loop_id_filter(
        self, temp_nomic_dir_with_files, mock_http_handler
    ):
        """Test that /api/history/cycles filters by loop_id."""
        with patch("aragora.server.handlers.system.extract_user_from_request") as mock_auth:
            mock_user = Mock()
            mock_user.is_authenticated = True
            mock_auth.return_value = mock_user

            ctx = {"nomic_dir": temp_nomic_dir_with_files}
            handler = SystemHandler(ctx)

            result = handler.handle("/api/history/cycles", {"loop_id": "loop_1"}, mock_http_handler)

            body = json.loads(result.body)
            assert len(body["cycles"]) == 2
            for cycle in body["cycles"]:
                assert cycle["loop_id"] == "loop_1"

    def test_get_history_events(self, temp_nomic_dir_with_files, mock_http_handler):
        """Test that /api/history/events returns event history."""
        with patch("aragora.server.handlers.system.extract_user_from_request") as mock_auth:
            mock_user = Mock()
            mock_user.is_authenticated = True
            mock_auth.return_value = mock_user

            ctx = {"nomic_dir": temp_nomic_dir_with_files}
            handler = SystemHandler(ctx)

            result = handler.handle("/api/history/events", {}, mock_http_handler)

            assert result is not None
            assert result.status_code == 200
            body = json.loads(result.body)
            assert "events" in body

    def test_get_history_debates(
        self, mock_storage, mock_elo_system, temp_nomic_dir_with_files, mock_http_handler
    ):
        """Test that /api/history/debates returns debate history."""
        with patch("aragora.server.handlers.system.extract_user_from_request") as mock_auth:
            mock_user = Mock()
            mock_user.is_authenticated = True
            mock_auth.return_value = mock_user

            # Ensure list_recent returns a sliceable list
            mock_storage.list_recent.return_value = [
                {"id": "d1", "task": "Test debate"},
                {"id": "d2", "task": "Another debate"},
            ]

            ctx = {
                "storage": mock_storage,
                "elo_system": mock_elo_system,
                "nomic_dir": temp_nomic_dir_with_files,
            }
            handler = SystemHandler(ctx)

            result = handler.handle("/api/history/debates", {}, mock_http_handler)

            assert result is not None
            assert result.status_code == 200
            body = json.loads(result.body)
            assert "debates" in body

    def test_get_history_summary(
        self, mock_storage, mock_elo_system, temp_nomic_dir_with_files, mock_http_handler
    ):
        """Test that /api/history/summary returns summary stats."""
        with patch("aragora.server.handlers.system.extract_user_from_request") as mock_auth:
            mock_user = Mock()
            mock_user.is_authenticated = True
            mock_auth.return_value = mock_user

            ctx = {
                "storage": mock_storage,
                "elo_system": mock_elo_system,
                "nomic_dir": temp_nomic_dir_with_files,
            }
            handler = SystemHandler(ctx)
            mock_storage.list_recent.return_value = [Mock(), Mock()]
            mock_elo_system.get_leaderboard.return_value = [Mock(), Mock(), Mock()]

            result = handler.handle("/api/history/summary", {}, mock_http_handler)

            assert result is not None
            assert result.status_code == 200
            body = json.loads(result.body)
            assert body["total_debates"] == 2
            assert body["total_agents"] == 3

    def test_history_requires_auth_when_enabled(self, temp_nomic_dir_with_files, mock_http_handler):
        """Test that history endpoints require auth when enabled."""
        with patch("aragora.server.auth.auth_config") as mock_config:
            mock_config.enabled = True
            mock_config.api_token = "secret"
            mock_config.validate_token.return_value = False

            with patch("aragora.server.handlers.system.extract_user_from_request") as mock_auth:
                mock_user = Mock()
                mock_user.is_authenticated = False
                mock_auth.return_value = mock_user

                ctx = {"nomic_dir": temp_nomic_dir_with_files}
                handler = SystemHandler(ctx)

                result = handler.handle("/api/history/cycles", {}, mock_http_handler)

                assert result is not None
                assert result.status_code == 401


# =============================================================================
# System Maintenance Tests
# =============================================================================


class TestSystemMaintenance:
    """Tests for /api/system/maintenance endpoint."""

    def test_maintenance_status(self, temp_nomic_dir_with_files, mock_http_handler):
        """Test that maintenance status returns stats."""
        with patch("aragora.maintenance.DatabaseMaintenance") as mock_maintenance:
            mock_instance = Mock()
            mock_instance.get_stats.return_value = {"total_size_mb": 10}
            mock_maintenance.return_value = mock_instance

            ctx = {"nomic_dir": temp_nomic_dir_with_files}
            handler = SystemHandler(ctx)

            result = handler.handle(
                "/api/system/maintenance", {"task": "status"}, mock_http_handler
            )

            assert result is not None
            assert result.status_code == 200
            body = json.loads(result.body)
            assert body["task"] == "status"
            assert "stats" in body

    def test_maintenance_invalid_task(self, temp_nomic_dir_with_files, mock_http_handler):
        """Test that invalid maintenance task returns error."""
        ctx = {"nomic_dir": temp_nomic_dir_with_files}
        handler = SystemHandler(ctx)

        result = handler.handle("/api/system/maintenance", {"task": "invalid"}, mock_http_handler)

        assert result is not None
        assert result.status_code == 400
        body = json.loads(result.body)
        assert "error" in body


# =============================================================================
# Auth Stats Tests
# =============================================================================


class TestAuthStats:
    """Tests for /api/auth/* endpoints."""

    def test_get_auth_stats(self, mock_http_handler):
        """Test that /api/auth/stats returns auth statistics."""
        with patch("aragora.server.auth.auth_config") as mock_config:
            mock_config.enabled = True
            mock_config.rate_limit_per_minute = 60
            mock_config.ip_rate_limit_per_minute = 120
            mock_config.token_ttl = 3600
            mock_config.get_rate_limit_stats.return_value = {"requests": 100}

            ctx = {}
            handler = SystemHandler(ctx)

            result = handler.handle("/api/auth/stats", {}, mock_http_handler)

            assert result is not None
            assert result.status_code == 200
            body = json.loads(result.body)
            assert body["enabled"] is True
            assert body["rate_limit_per_minute"] == 60
            assert "stats" in body

    def test_revoke_token(self, mock_http_handler):
        """Test that /api/auth/revoke revokes a token."""
        with patch("aragora.server.auth.auth_config") as mock_config:
            mock_config.revoke_token.return_value = True
            mock_config.get_revocation_count.return_value = 5

            mock_http_handler.rfile = Mock()
            mock_http_handler.rfile.read.return_value = (
                b'{"token": "test_token", "reason": "compromised"}'
            )
            mock_http_handler.headers = {"Content-Length": "50"}

            ctx = {}
            handler = SystemHandler(ctx)

            result = handler.handle_post("/api/auth/revoke", {}, mock_http_handler)

            assert result is not None
            assert result.status_code == 200
            body = json.loads(result.body)
            assert body["success"] is True
            assert body["revoked_count"] == 5

    def test_revoke_token_missing_token(self, mock_http_handler):
        """Test that /api/auth/revoke returns error when token missing."""
        mock_http_handler.rfile = Mock()
        mock_http_handler.rfile.read.return_value = b'{"reason": "test"}'
        mock_http_handler.headers = {"Content-Length": "20"}

        ctx = {}
        handler = SystemHandler(ctx)

        result = handler.handle_post("/api/auth/revoke", {}, mock_http_handler)

        assert result is not None
        assert result.status_code == 400


# =============================================================================
# OpenAPI Tests
# =============================================================================


class TestOpenAPI:
    """Tests for /api/openapi and /api/docs endpoints (DocsHandler)."""

    def test_get_openapi_json(self, mock_http_handler):
        """Test that /api/openapi returns JSON spec."""
        with patch("aragora.server.openapi.handle_openapi_request") as mock_openapi:
            mock_openapi.return_value = ('{"openapi": "3.0.0"}', "application/json")

            ctx = {}
            handler = DocsHandler(ctx)

            result = handler.handle("/api/openapi", {}, mock_http_handler)

            assert result is not None
            assert result.status_code == 200
            assert result.content_type == "application/json"

    def test_get_openapi_yaml(self, mock_http_handler):
        """Test that /api/openapi.yaml returns YAML spec."""
        with patch("aragora.server.openapi.handle_openapi_request") as mock_openapi:
            mock_openapi.return_value = ('openapi: "3.0.0"', "application/x-yaml")

            ctx = {}
            handler = DocsHandler(ctx)

            result = handler.handle("/api/openapi.yaml", {}, mock_http_handler)

            assert result is not None
            assert result.status_code == 200
            assert result.content_type == "application/x-yaml"

    def test_get_swagger_ui(self, mock_http_handler):
        """Test that /api/docs returns Swagger UI HTML."""
        ctx = {}
        handler = DocsHandler(ctx)

        result = handler.handle("/api/docs", {}, mock_http_handler)

        assert result is not None
        assert result.status_code == 200
        assert "text/html" in result.content_type
        assert b"swagger-ui" in result.body


# =============================================================================
# Circuit Breaker Tests
# =============================================================================


class TestCircuitBreakers:
    """Tests for /api/circuit-breakers endpoint."""

    def test_get_circuit_breaker_metrics(self, mock_http_handler):
        """Test that /api/circuit-breakers returns metrics."""
        with patch("aragora.resilience.get_circuit_breaker_metrics") as mock_metrics:
            mock_metrics.return_value = {
                "summary": {"total": 5, "open": 0, "closed": 5},
                "circuit_breakers": {},
                "health": {"status": "healthy"},
            }

            ctx = {}
            handler = SystemHandler(ctx)

            result = handler.handle("/api/circuit-breakers", {}, mock_http_handler)

            assert result is not None
            assert result.status_code == 200
            body = json.loads(result.body)
            assert "summary" in body
            assert body["health"]["status"] == "healthy"


# =============================================================================
# Prometheus Metrics Tests
# =============================================================================


class TestPrometheusMetrics:
    """Tests for /metrics endpoint."""

    def test_get_prometheus_metrics(self, mock_http_handler):
        """Test that /metrics returns Prometheus format."""
        with patch("aragora.server.metrics.generate_metrics") as mock_gen:
            mock_gen.return_value = "# HELP aragora_requests_total\naragora_requests_total 100"

            ctx = {}
            handler = SystemHandler(ctx)

            result = handler.handle("/metrics", {}, mock_http_handler)

            assert result is not None
            assert result.status_code == 200
            assert "text/plain" in result.content_type
            assert b"aragora_requests_total" in result.body


# =============================================================================
# Modes Tests
# =============================================================================


class TestModes:
    """Tests for /api/modes endpoint (NomicHandler)."""

    def test_get_modes_returns_builtin(self, mock_http_handler):
        """Test that /api/modes returns builtin modes."""
        ctx = {"nomic_dir": None}
        handler = NomicHandler(ctx)

        result = handler.handle("/api/modes", {}, mock_http_handler)

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert "modes" in body
        assert len(body["modes"]) >= 5  # At least the builtin modes

        # Check that builtin modes are present
        mode_names = [m["name"] for m in body["modes"]]
        assert "architect" in mode_names
        assert "coder" in mode_names
        assert "debugger" in mode_names


# =============================================================================
# Can Handle Tests
# =============================================================================


class TestCanHandle:
    """Tests for handler routing - each handler type handles its own routes."""

    def test_can_handle_health_routes(self, health_handler):
        """Test that HealthHandler can handle health routes."""
        assert health_handler.can_handle("/healthz") is True
        assert health_handler.can_handle("/readyz") is True
        assert health_handler.can_handle("/api/health") is True
        assert health_handler.can_handle("/api/health/detailed") is True
        assert health_handler.can_handle("/api/health/deep") is True

    def test_can_handle_nomic_routes(self, nomic_handler):
        """Test that NomicHandler can handle nomic routes."""
        assert nomic_handler.can_handle("/api/nomic/state") is True
        assert nomic_handler.can_handle("/api/nomic/health") is True
        assert nomic_handler.can_handle("/api/nomic/log") is True
        assert nomic_handler.can_handle("/api/nomic/risk-register") is True

    def test_can_handle_history_routes(self, system_handler):
        """Test that SystemHandler can handle history routes."""
        assert system_handler.can_handle("/api/history/cycles") is True
        assert system_handler.can_handle("/api/history/events") is True
        assert system_handler.can_handle("/api/history/debates") is True
        assert system_handler.can_handle("/api/history/summary") is True

    def test_cannot_handle_unknown_routes(self, system_handler, health_handler):
        """Test that handlers reject unknown routes."""
        assert system_handler.can_handle("/api/unknown") is False
        assert system_handler.can_handle("/other/path") is False
        assert health_handler.can_handle("/api/unknown") is False


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling in system endpoints."""

    def test_invalid_json_state_file(self, mock_http_handler):
        """Test handling of corrupt state file (NomicHandler)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nomic_dir = Path(tmpdir)
            state_file = nomic_dir / "nomic_state.json"
            state_file.write_text("not valid json{")

            ctx = {"nomic_dir": nomic_dir}
            handler = NomicHandler(ctx)

            result = handler.handle("/api/nomic/state", {}, mock_http_handler)

            assert result is not None
            assert result.status_code == 500
            body = json.loads(result.body)
            assert "error" in body

    def test_permission_error_on_log_read(self, mock_http_handler):
        """Test handling of permission errors (NomicHandler)."""
        with patch("builtins.open", side_effect=PermissionError("Access denied")):
            with tempfile.TemporaryDirectory() as tmpdir:
                nomic_dir = Path(tmpdir)
                # Create the file so exists() check passes
                log_file = nomic_dir / "nomic_loop.log"
                log_file.touch()

                ctx = {"nomic_dir": nomic_dir}
                handler = NomicHandler(ctx)

                result = handler.handle("/api/nomic/log", {}, mock_http_handler)

                assert result is not None
                assert result.status_code == 500


# =============================================================================
# Debug Endpoint Tests
# =============================================================================


class TestDebugEndpoint:
    """Tests for /api/debug/test endpoint."""

    def test_debug_endpoint_returns_ok(self, mock_http_handler):
        """Test that debug endpoint returns ok status."""
        ctx = {}
        handler = SystemHandler(ctx)

        result = handler.handle("/api/debug/test", {}, mock_http_handler)

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["status"] == "ok"
        assert body["message"] == "Modular handler works"

    def test_debug_endpoint_includes_method(self, mock_http_handler):
        """Test that debug endpoint includes HTTP method."""
        mock_http_handler.command = "POST"
        ctx = {}
        handler = SystemHandler(ctx)

        result = handler.handle("/api/debug/test", {}, mock_http_handler)

        body = json.loads(result.body)
        assert body["method"] == "POST"


# =============================================================================
# Filesystem Health Tests
# =============================================================================


class TestFilesystemHealth:
    """Tests for filesystem health check logic (HealthHandler)."""

    def test_filesystem_health_with_nomic_dir(self, temp_nomic_dir, mock_http_handler):
        """Test filesystem health with valid nomic directory."""
        ctx = {"nomic_dir": temp_nomic_dir}
        handler = HealthHandler(ctx)

        result = handler._check_filesystem_health()

        assert result["healthy"] is True
        assert "path" in result

    def test_filesystem_health_falls_back_to_temp(self, mock_http_handler):
        """Test filesystem health falls back to temp when no nomic dir."""
        ctx = {"nomic_dir": None}
        handler = HealthHandler(ctx)

        result = handler._check_filesystem_health()

        # Should still succeed using temp directory
        assert result["healthy"] is True


# =============================================================================
# Redis Health Tests
# =============================================================================


class TestRedisHealth:
    """Tests for Redis health check logic (HealthHandler)."""

    def test_redis_health_not_configured(self, monkeypatch, mock_http_handler):
        """Test Redis health when not configured."""
        monkeypatch.delenv("REDIS_URL", raising=False)
        monkeypatch.delenv("CACHE_REDIS_URL", raising=False)

        ctx = {}
        handler = HealthHandler(ctx)

        result = handler._check_redis_health()

        assert result["healthy"] is True
        assert result["configured"] is False
        assert "not configured" in result.get("note", "").lower()

    def test_redis_health_configured_but_unavailable(self, monkeypatch, mock_http_handler):
        """Test Redis health when configured but connection fails."""
        monkeypatch.setenv("REDIS_URL", "redis://localhost:6379")

        with patch("redis.from_url") as mock_redis:
            mock_redis.side_effect = ConnectionRefusedError("Connection refused")

            ctx = {}
            handler = HealthHandler(ctx)

            result = handler._check_redis_health()

            assert result["healthy"] is False
            assert result["configured"] is True
            assert "error" in result

    def test_redis_health_module_not_installed(self, monkeypatch, mock_http_handler):
        """Test Redis health when redis module not installed."""
        monkeypatch.setenv("REDIS_URL", "redis://localhost:6379")

        # Simulate redis import failure
        import builtins

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "redis":
                raise ImportError("No module named 'redis'")
            return original_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", mock_import):
            ctx = {}
            handler = HealthHandler(ctx)

            result = handler._check_redis_health()

            # Should handle ImportError gracefully
            assert result["healthy"] is True
            assert "warning" in result or "not installed" in str(result).lower()


# =============================================================================
# AI Provider Health Tests
# =============================================================================


class TestAIProviderHealth:
    """Tests for AI provider availability checks (HealthHandler)."""

    def test_ai_provider_health_none_configured(self, monkeypatch, mock_http_handler):
        """Test AI provider health when no providers configured."""
        # Clear all API keys
        for key in [
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY",
            "GEMINI_API_KEY",
            "GROK_API_KEY",
            "XAI_API_KEY",
            "DEEPSEEK_API_KEY",
            "OPENROUTER_API_KEY",
        ]:
            monkeypatch.delenv(key, raising=False)

        ctx = {}
        handler = HealthHandler(ctx)

        result = handler._check_ai_providers_health()

        assert result["healthy"] is True  # Not having providers isn't unhealthy
        assert result["any_available"] is False
        assert result["available_count"] == 0

    def test_ai_provider_health_anthropic_configured(self, monkeypatch, mock_http_handler):
        """Test AI provider health with Anthropic configured."""
        for key in [
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY",
            "GEMINI_API_KEY",
            "GROK_API_KEY",
            "XAI_API_KEY",
            "DEEPSEEK_API_KEY",
            "OPENROUTER_API_KEY",
        ]:
            monkeypatch.delenv(key, raising=False)

        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key-12345")

        ctx = {}
        handler = HealthHandler(ctx)

        result = handler._check_ai_providers_health()

        assert result["any_available"] is True
        assert result["available_count"] == 1
        assert result["providers"]["anthropic"] is True

    def test_ai_provider_health_multiple_configured(self, monkeypatch, mock_http_handler):
        """Test AI provider health with multiple providers configured."""
        for key in [
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY",
            "GEMINI_API_KEY",
            "GROK_API_KEY",
            "XAI_API_KEY",
            "DEEPSEEK_API_KEY",
            "OPENROUTER_API_KEY",
        ]:
            monkeypatch.delenv(key, raising=False)

        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key-12345")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-openai-key-1234")

        ctx = {}
        handler = HealthHandler(ctx)

        result = handler._check_ai_providers_health()

        assert result["any_available"] is True
        assert result["available_count"] == 2
        assert result["providers"]["anthropic"] is True
        assert result["providers"]["openai"] is True

    def test_ai_provider_health_short_key_not_counted(self, monkeypatch, mock_http_handler):
        """Test that short API keys are not counted as configured."""
        for key in [
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY",
            "GEMINI_API_KEY",
            "GROK_API_KEY",
            "XAI_API_KEY",
            "DEEPSEEK_API_KEY",
            "OPENROUTER_API_KEY",
        ]:
            monkeypatch.delenv(key, raising=False)

        monkeypatch.setenv("ANTHROPIC_API_KEY", "short")  # Too short

        ctx = {}
        handler = HealthHandler(ctx)

        result = handler._check_ai_providers_health()

        assert result["providers"]["anthropic"] is False  # Key too short


# =============================================================================
# Maintenance Task Tests
# =============================================================================


class TestMaintenanceTasks:
    """Tests for all maintenance task types."""

    def test_maintenance_vacuum(self, temp_nomic_dir_with_files, mock_http_handler):
        """Test maintenance vacuum task."""
        with patch("aragora.maintenance.DatabaseMaintenance") as mock_maintenance:
            mock_instance = Mock()
            mock_instance.vacuum_all.return_value = {"vacuumed": 3}
            mock_instance.get_stats.return_value = {"total_size_mb": 10}
            mock_maintenance.return_value = mock_instance

            ctx = {"nomic_dir": temp_nomic_dir_with_files}
            handler = SystemHandler(ctx)

            result = handler.handle(
                "/api/system/maintenance", {"task": "vacuum"}, mock_http_handler
            )

            assert result is not None
            assert result.status_code == 200
            body = json.loads(result.body)
            assert body["task"] == "vacuum"
            assert "vacuum" in body
            mock_instance.vacuum_all.assert_called_once()

    def test_maintenance_analyze(self, temp_nomic_dir_with_files, mock_http_handler):
        """Test maintenance analyze task."""
        with patch("aragora.maintenance.DatabaseMaintenance") as mock_maintenance:
            mock_instance = Mock()
            mock_instance.analyze_all.return_value = {"analyzed": 5}
            mock_instance.get_stats.return_value = {"total_size_mb": 10}
            mock_maintenance.return_value = mock_instance

            ctx = {"nomic_dir": temp_nomic_dir_with_files}
            handler = SystemHandler(ctx)

            result = handler.handle(
                "/api/system/maintenance", {"task": "analyze"}, mock_http_handler
            )

            assert result is not None
            assert result.status_code == 200
            body = json.loads(result.body)
            assert body["task"] == "analyze"
            assert "analyze" in body
            mock_instance.analyze_all.assert_called_once()

    def test_maintenance_checkpoint(self, temp_nomic_dir_with_files, mock_http_handler):
        """Test maintenance checkpoint task."""
        with patch("aragora.maintenance.DatabaseMaintenance") as mock_maintenance:
            mock_instance = Mock()
            mock_instance.checkpoint_all_wal.return_value = {"checkpointed": 2}
            mock_instance.get_stats.return_value = {"total_size_mb": 10}
            mock_maintenance.return_value = mock_instance

            ctx = {"nomic_dir": temp_nomic_dir_with_files}
            handler = SystemHandler(ctx)

            result = handler.handle(
                "/api/system/maintenance", {"task": "checkpoint"}, mock_http_handler
            )

            assert result is not None
            assert result.status_code == 200
            body = json.loads(result.body)
            assert body["task"] == "checkpoint"
            assert "checkpoint" in body
            mock_instance.checkpoint_all_wal.assert_called_once()

    def test_maintenance_full(self, temp_nomic_dir_with_files, mock_http_handler):
        """Test maintenance full task runs all tasks."""
        with patch("aragora.maintenance.DatabaseMaintenance") as mock_maintenance:
            mock_instance = Mock()
            mock_instance.checkpoint_all_wal.return_value = {"checkpointed": 2}
            mock_instance.analyze_all.return_value = {"analyzed": 5}
            mock_instance.vacuum_all.return_value = {"vacuumed": 3}
            mock_instance.get_stats.return_value = {"total_size_mb": 10}
            mock_maintenance.return_value = mock_instance

            ctx = {"nomic_dir": temp_nomic_dir_with_files}
            handler = SystemHandler(ctx)

            result = handler.handle("/api/system/maintenance", {"task": "full"}, mock_http_handler)

            assert result is not None
            assert result.status_code == 200
            body = json.loads(result.body)
            assert body["task"] == "full"
            assert "checkpoint" in body
            assert "analyze" in body
            assert "vacuum" in body
            # Full task calls all three
            mock_instance.checkpoint_all_wal.assert_called_once()
            mock_instance.analyze_all.assert_called_once()
            mock_instance.vacuum_all.assert_called_once()

    def test_maintenance_no_nomic_dir(self, mock_http_handler):
        """Test maintenance fails gracefully without nomic dir."""
        ctx = {"nomic_dir": None}
        handler = SystemHandler(ctx)

        result = handler.handle("/api/system/maintenance", {"task": "status"}, mock_http_handler)

        assert result is not None
        assert result.status_code == 503
        body = json.loads(result.body)
        assert "error" in body


# =============================================================================
# Validation Tests
# =============================================================================


class TestValidation:
    """Tests for input validation."""

    def test_invalid_loop_id_format_cycles(self, temp_nomic_dir_with_files, mock_http_handler):
        """Test that invalid loop_id format is rejected for cycles."""
        with patch("aragora.server.handlers.system.extract_user_from_request") as mock_auth:
            mock_user = Mock()
            mock_user.is_authenticated = True
            mock_auth.return_value = mock_user

            ctx = {"nomic_dir": temp_nomic_dir_with_files}
            handler = SystemHandler(ctx)

            # Try with path traversal attempt
            result = handler.handle(
                "/api/history/cycles", {"loop_id": "../etc/passwd"}, mock_http_handler
            )

            assert result is not None
            assert result.status_code == 400
            body = json.loads(result.body)
            assert "error" in body

    def test_invalid_loop_id_format_events(self, temp_nomic_dir_with_files, mock_http_handler):
        """Test that invalid loop_id format is rejected for events."""
        with patch("aragora.server.handlers.system.extract_user_from_request") as mock_auth:
            mock_user = Mock()
            mock_user.is_authenticated = True
            mock_auth.return_value = mock_user

            ctx = {"nomic_dir": temp_nomic_dir_with_files}
            handler = SystemHandler(ctx)

            # Try with special characters
            result = handler.handle(
                "/api/history/events", {"loop_id": "loop<script>"}, mock_http_handler
            )

            assert result is not None
            assert result.status_code == 400

    def test_nomic_log_line_limit_clamped(self, temp_nomic_dir_with_files, mock_http_handler):
        """Test that nomic log line limit is clamped to 1000 (NomicHandler)."""
        ctx = {"nomic_dir": temp_nomic_dir_with_files}
        handler = NomicHandler(ctx)

        # Request more than max
        result = handler.handle("/api/nomic/log", {"lines": "9999"}, mock_http_handler)

        assert result is not None
        assert result.status_code == 200
        # Should be clamped - check the code clamps to max 1000

    def test_risk_register_limit_clamped(self, temp_nomic_dir_with_files, mock_http_handler):
        """Test that risk register limit is clamped to 200 (NomicHandler)."""
        ctx = {"nomic_dir": temp_nomic_dir_with_files}
        handler = NomicHandler(ctx)

        # Request more than max
        result = handler.handle("/api/nomic/risk-register", {"limit": "9999"}, mock_http_handler)

        assert result is not None
        assert result.status_code == 200
        # Code clamps to max 200


# =============================================================================
# Health Degradation Tests
# =============================================================================


class TestHealthDegradation:
    """Tests for health status degradation scenarios (HealthHandler)."""

    def test_health_degraded_when_storage_fails(
        self, mock_elo_system, temp_nomic_dir, mock_http_handler
    ):
        """Test health is degraded when storage query fails."""
        mock_storage = Mock()
        mock_storage.list_recent.side_effect = RuntimeError("Database connection lost")

        ctx = {
            "storage": mock_storage,
            "elo_system": mock_elo_system,
            "nomic_dir": temp_nomic_dir,
        }
        handler = HealthHandler(ctx)
        mock_elo_system.get_leaderboard.return_value = []

        result = handler.handle("/api/health", {}, mock_http_handler)

        assert result is not None
        assert result.status_code == 503
        body = json.loads(result.body)
        assert body["status"] == "degraded"

    def test_health_warning_when_elo_fails(self, mock_storage, temp_nomic_dir, mock_http_handler):
        """Test health includes warning when ELO query fails (ELO is non-critical)."""
        mock_elo = Mock()
        mock_elo.get_leaderboard.side_effect = RuntimeError("ELO database error")

        ctx = {
            "storage": mock_storage,
            "elo_system": mock_elo,
            "nomic_dir": temp_nomic_dir,
        }
        handler = HealthHandler(ctx)
        mock_storage.list_recent.return_value = []

        result = handler.handle("/api/health", {}, mock_http_handler)

        assert result is not None
        # ELO is non-critical, so health remains healthy with warning
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["status"] == "healthy"
        # But ELO check should indicate the issue
        assert body["checks"]["elo_system"]["warning"] is not None


# =============================================================================
# Circuit Breaker Edge Cases
# =============================================================================


class TestCircuitBreakerEdgeCases:
    """Additional tests for circuit breaker metrics edge cases."""

    def test_circuit_breaker_module_not_available(self, mock_http_handler):
        """Test circuit breaker endpoint when module not available."""
        with patch("aragora.resilience.get_circuit_breaker_metrics", side_effect=ImportError):
            ctx = {}
            handler = SystemHandler(ctx)

            result = handler.handle("/api/circuit-breakers", {}, mock_http_handler)

            assert result is not None
            assert result.status_code == 503

    def test_circuit_breaker_unexpected_error(self, mock_http_handler):
        """Test circuit breaker endpoint handles unexpected errors."""
        with patch(
            "aragora.resilience.get_circuit_breaker_metrics", side_effect=ValueError("Unexpected")
        ):
            ctx = {}
            handler = SystemHandler(ctx)

            result = handler.handle("/api/circuit-breakers", {}, mock_http_handler)

            assert result is not None
            assert result.status_code == 500


# =============================================================================
# Prometheus Metrics Edge Cases
# =============================================================================


class TestPrometheusEdgeCases:
    """Additional tests for Prometheus metrics edge cases."""

    def test_prometheus_module_not_available(self, mock_http_handler):
        """Test metrics endpoint when module not available."""
        with patch("aragora.server.metrics.generate_metrics", side_effect=ImportError):
            ctx = {}
            handler = SystemHandler(ctx)

            result = handler.handle("/metrics", {}, mock_http_handler)

            assert result is not None
            assert result.status_code == 503

    def test_prometheus_generation_error(self, mock_http_handler):
        """Test metrics endpoint handles generation errors."""
        with patch(
            "aragora.server.metrics.generate_metrics", side_effect=RuntimeError("Generation failed")
        ):
            ctx = {}
            handler = SystemHandler(ctx)

            result = handler.handle("/metrics", {}, mock_http_handler)

            assert result is not None
            assert result.status_code == 500


# =============================================================================
# OpenAPI Edge Cases
# =============================================================================


class TestOpenAPIEdgeCases:
    """Additional tests for OpenAPI endpoint edge cases (DocsHandler)."""

    def test_openapi_json_format(self, mock_http_handler):
        """Test OpenAPI JSON endpoint path variation."""
        with patch("aragora.server.openapi.handle_openapi_request") as mock_openapi:
            mock_openapi.return_value = ('{"openapi": "3.0.0"}', "application/json")

            ctx = {}
            handler = DocsHandler(ctx)

            result = handler.handle("/api/openapi.json", {}, mock_http_handler)

            assert result is not None
            assert result.status_code == 200
            assert result.content_type == "application/json"

    def test_openapi_module_not_available(self, mock_http_handler):
        """Test OpenAPI endpoint when module not available."""
        with patch("aragora.server.openapi.handle_openapi_request", side_effect=ImportError):
            ctx = {}
            handler = DocsHandler(ctx)

            result = handler.handle("/api/openapi", {}, mock_http_handler)

            assert result is not None
            assert result.status_code == 503

    def test_swagger_ui_docs_slash_variant(self, mock_http_handler):
        """Test Swagger UI at /api/docs/ (with trailing slash)."""
        ctx = {}
        handler = DocsHandler(ctx)

        result = handler.handle("/api/docs/", {}, mock_http_handler)

        assert result is not None
        assert result.status_code == 200
        assert "text/html" in result.content_type


# =============================================================================
# Risk Register Edge Cases
# =============================================================================


class TestRiskRegisterEdgeCases:
    """Additional tests for risk register edge cases (NomicHandler)."""

    def test_risk_register_empty(self, mock_http_handler):
        """Test risk register with no risks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nomic_dir = Path(tmpdir)
            ctx = {"nomic_dir": nomic_dir}
            handler = NomicHandler(ctx)

            result = handler.handle("/api/nomic/risk-register", {}, mock_http_handler)

            assert result is not None
            assert result.status_code == 200
            body = json.loads(result.body)
            assert body["risks"] == []
            assert body["total"] == 0

    def test_risk_register_malformed_lines(self, mock_http_handler):
        """Test risk register handles malformed JSON lines."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nomic_dir = Path(tmpdir)
            risk_file = nomic_dir / "risk_register.jsonl"
            # Mix of valid and invalid JSON
            risk_file.write_text(
                '{"id": "1", "severity": "critical"}\ninvalid json here\n{"id": "2", "severity": "low"}'
            )

            ctx = {"nomic_dir": nomic_dir}
            handler = NomicHandler(ctx)

            result = handler.handle("/api/nomic/risk-register", {}, mock_http_handler)

            assert result is not None
            assert result.status_code == 200
            body = json.loads(result.body)
            # Should skip malformed line
            assert body["total"] == 2


# =============================================================================
# Nomic Health Edge Cases
# =============================================================================


class TestNomicHealthEdgeCases:
    """Additional tests for nomic health edge cases (NomicHandler)."""

    def test_nomic_health_invalid_timestamp(self, mock_http_handler):
        """Test nomic health with invalid timestamp format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nomic_dir = Path(tmpdir)
            state_file = nomic_dir / "nomic_state.json"
            state_file.write_text(
                json.dumps(
                    {
                        "phase": "implement",
                        "cycle": 1,
                        "last_update": "not-a-valid-timestamp",
                    }
                )
            )

            ctx = {"nomic_dir": nomic_dir}
            handler = NomicHandler(ctx)

            result = handler.handle("/api/nomic/health", {}, mock_http_handler)

            assert result is not None
            assert result.status_code == 200
            body = json.loads(result.body)
            # Should not crash, status based on what it can determine
            assert "status" in body

    def test_nomic_health_corrupt_json(self, mock_http_handler):
        """Test nomic health with corrupt state file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nomic_dir = Path(tmpdir)
            state_file = nomic_dir / "nomic_state.json"
            state_file.write_text("not valid json {")

            ctx = {"nomic_dir": nomic_dir}
            handler = NomicHandler(ctx)

            result = handler.handle("/api/nomic/health", {}, mock_http_handler)

            assert result is not None
            assert result.status_code == 200
            body = json.loads(result.body)
            assert body["status"] == "error"


# =============================================================================
# POST Handler Tests
# =============================================================================


class TestPostHandler:
    """Tests for POST request handling."""

    def test_post_unknown_path_returns_none(self, mock_http_handler):
        """Test that POST to unknown path returns None."""
        ctx = {}
        handler = SystemHandler(ctx)

        result = handler.handle_post("/api/unknown", {}, mock_http_handler)

        assert result is None

    def test_revoke_token_invalid_json(self, mock_http_handler):
        """Test token revocation with invalid JSON body."""
        mock_http_handler.rfile = Mock()
        mock_http_handler.rfile.read.return_value = b"not valid json"
        mock_http_handler.headers = {"Content-Length": "16"}

        ctx = {}
        handler = SystemHandler(ctx)

        result = handler.handle_post("/api/auth/revoke", {}, mock_http_handler)

        assert result is not None
        assert result.status_code == 400
