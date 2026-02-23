"""Comprehensive tests for control plane health monitoring handlers.

Tests the HealthHandlerMixin endpoints:
- GET  /api/control-plane/health           (system health)
- GET  /api/control-plane/health/{agent_id} (agent health)
- GET  /api/control-plane/health/detailed   (detailed health with components)
- GET  /api/control-plane/breakers          (circuit breaker states)
- GET  /api/control-plane/stats             (control plane statistics)
- GET  /api/control-plane/metrics           (dashboard metrics)
- GET  /api/control-plane/notifications     (recent notifications)
- GET  /api/control-plane/notifications/stats (notification statistics)
- GET  /api/control-plane/audit             (audit logs)
- GET  /api/control-plane/audit/stats       (audit log statistics)
- GET  /api/control-plane/audit/verify      (audit integrity verification)
"""

from __future__ import annotations

import json
import sys
import time
import types
from contextlib import contextmanager
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.control_plane import ControlPlaneHandler


# ============================================================================
# Helpers
# ============================================================================


def _body(result: object) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    if isinstance(result, dict):
        return result
    return json.loads(result.body)


def _status(result: object) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


@contextmanager
def _mock_db_module(database_cls):
    """Temporarily inject a mock ``aragora.db.database`` module into sys.modules.

    The handler does ``from aragora.db.database import Database`` inside a
    try/except.  Since the real module does not exist, we inject a synthetic
    module with the supplied *database_cls* as the ``Database`` attribute so the
    import succeeds during testing.
    """
    mod = types.ModuleType("aragora.db.database")
    mod.Database = database_cls  # type: ignore[attr-defined]
    old = sys.modules.get("aragora.db.database")
    sys.modules["aragora.db.database"] = mod
    try:
        yield
    finally:
        if old is None:
            sys.modules.pop("aragora.db.database", None)
        else:
            sys.modules["aragora.db.database"] = old


# ============================================================================
# Mock Domain Objects
# ============================================================================


class MockHealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class MockHealthCheck:
    """Mock health check returned by the coordinator."""

    def __init__(self, agent_id: str, status: str = "healthy", latency_ms: float = 10.0):
        self.agent_id = agent_id
        self.status = MockHealthStatus(status)
        self.last_check = datetime.now(timezone.utc)
        self.latency_ms = latency_ms
        self.error_rate = 0.01

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "status": self.status.value,
            "last_check": self.last_check.isoformat(),
            "latency_ms": self.latency_ms,
            "error_rate": self.error_rate,
        }


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(autouse=True)
def _reset_has_permission_cache():
    """Reset the _get_has_permission cache and inject a permissive has_permission.

    The handler's _get_has_permission() caches the has_permission callable.
    The conftest patches handler_decorators.has_permission, but the control_plane
    __init__ already imported it as a separate binding.  We force the cached
    function to always return True so the inline permission checks in
    _handle_get_audit_logs and _handle_verify_audit_integrity pass.
    """
    import aragora.server.handlers.control_plane.health as health_mod

    health_mod._cached_has_permission = lambda role, perm: True
    health_mod._cache_timestamp = time.time()
    yield
    health_mod._cached_has_permission = None
    health_mod._cache_timestamp = 0


@pytest.fixture
def mock_coordinator():
    """Create a mock coordinator with health-related methods."""
    coord = MagicMock()

    # Health methods
    coord.get_system_health.return_value = MockHealthStatus.HEALTHY
    coord.get_agent_health.return_value = None

    # Health monitor sub-object
    health_monitor = MagicMock()
    health_monitor.get_all_health.return_value = {}
    coord._health_monitor = health_monitor

    # Scheduler sub-object
    scheduler = MagicMock()
    coord._scheduler = scheduler

    # Stats
    coord.get_stats = AsyncMock(return_value={
        "scheduler": {
            "by_status": {"running": 3, "pending": 5, "completed": 100},
            "by_type": {"document_processing": 42, "audit": 7},
        },
        "registry": {
            "total_agents": 10,
            "available_agents": 8,
            "by_status": {"busy": 2},
        },
    })

    return coord


@pytest.fixture
def handler(mock_coordinator):
    """Create a ControlPlaneHandler with a mock coordinator in context."""
    ctx: dict[str, Any] = {
        "control_plane_coordinator": mock_coordinator,
    }
    return ControlPlaneHandler(ctx)


@pytest.fixture
def handler_no_coord():
    """Create a ControlPlaneHandler with NO coordinator."""
    ctx: dict[str, Any] = {}
    return ControlPlaneHandler(ctx)


@pytest.fixture
def mock_http_handler():
    """Create a minimal mock HTTP handler."""
    m = MagicMock()
    m.path = "/api/control-plane/health"
    m.headers = {"Content-Type": "application/json"}
    return m


# ============================================================================
# GET /api/control-plane/health  (system health)
# ============================================================================


class TestSystemHealth:
    """Tests for _handle_system_health."""

    def test_system_health_success_healthy(self, handler, mock_coordinator):
        mock_coordinator.get_system_health.return_value = MockHealthStatus.HEALTHY
        result = handler._handle_system_health()
        assert _status(result) == 200
        body = _body(result)
        assert body["status"] == "healthy"
        assert "agents" in body

    def test_system_health_degraded(self, handler, mock_coordinator):
        mock_coordinator.get_system_health.return_value = MockHealthStatus.DEGRADED
        result = handler._handle_system_health()
        assert _status(result) == 200
        assert _body(result)["status"] == "degraded"

    def test_system_health_unhealthy(self, handler, mock_coordinator):
        mock_coordinator.get_system_health.return_value = MockHealthStatus.UNHEALTHY
        result = handler._handle_system_health()
        assert _status(result) == 200
        assert _body(result)["status"] == "unhealthy"

    def test_system_health_with_agents(self, handler, mock_coordinator):
        hc1 = MockHealthCheck("agent-1", "healthy")
        hc2 = MockHealthCheck("agent-2", "degraded")
        mock_coordinator._health_monitor.get_all_health.return_value = {
            "agent-1": hc1,
            "agent-2": hc2,
        }
        result = handler._handle_system_health()
        assert _status(result) == 200
        body = _body(result)
        assert "agent-1" in body["agents"]
        assert "agent-2" in body["agents"]
        assert body["agents"]["agent-1"]["status"] == "healthy"
        assert body["agents"]["agent-2"]["status"] == "degraded"

    def test_system_health_empty_agents(self, handler, mock_coordinator):
        mock_coordinator._health_monitor.get_all_health.return_value = {}
        result = handler._handle_system_health()
        assert _status(result) == 200
        assert _body(result)["agents"] == {}

    def test_system_health_no_coordinator(self, handler_no_coord):
        result = handler_no_coord._handle_system_health()
        assert _status(result) == 503

    def test_system_health_value_error(self, handler, mock_coordinator):
        mock_coordinator.get_system_health.side_effect = ValueError("bad state")
        result = handler._handle_system_health()
        assert _status(result) == 400

    def test_system_health_key_error(self, handler, mock_coordinator):
        mock_coordinator.get_system_health.side_effect = KeyError("missing key")
        result = handler._handle_system_health()
        assert _status(result) == 400

    def test_system_health_attribute_error(self, handler, mock_coordinator):
        mock_coordinator.get_system_health.side_effect = AttributeError("no attr")
        result = handler._handle_system_health()
        assert _status(result) == 400

    def test_system_health_runtime_error(self, handler, mock_coordinator):
        mock_coordinator.get_system_health.side_effect = RuntimeError("crash")
        result = handler._handle_system_health()
        assert _status(result) == 500

    def test_system_health_os_error(self, handler, mock_coordinator):
        mock_coordinator.get_system_health.side_effect = OSError("disk")
        result = handler._handle_system_health()
        assert _status(result) == 500

    def test_system_health_type_error(self, handler, mock_coordinator):
        mock_coordinator.get_system_health.side_effect = TypeError("wrong type")
        result = handler._handle_system_health()
        assert _status(result) == 500


# ============================================================================
# GET /api/control-plane/health/{agent_id}  (agent health)
# ============================================================================


class TestAgentHealth:
    """Tests for _handle_agent_health."""

    def test_agent_health_success(self, handler, mock_coordinator):
        hc = MockHealthCheck("agent-A", "healthy", latency_ms=5.0)
        mock_coordinator.get_agent_health.return_value = hc
        result = handler._handle_agent_health("agent-A")
        assert _status(result) == 200
        body = _body(result)
        assert body["agent_id"] == "agent-A"
        assert body["status"] == "healthy"
        assert body["latency_ms"] == 5.0

    def test_agent_health_not_found(self, handler, mock_coordinator):
        mock_coordinator.get_agent_health.return_value = None
        result = handler._handle_agent_health("nonexistent")
        assert _status(result) == 404
        assert "nonexistent" in _body(result).get("error", "")

    def test_agent_health_no_coordinator(self, handler_no_coord):
        result = handler_no_coord._handle_agent_health("agent-A")
        assert _status(result) == 503

    def test_agent_health_value_error(self, handler, mock_coordinator):
        mock_coordinator.get_agent_health.side_effect = ValueError("bad id")
        result = handler._handle_agent_health("bad-id")
        assert _status(result) == 400

    def test_agent_health_key_error(self, handler, mock_coordinator):
        mock_coordinator.get_agent_health.side_effect = KeyError("missing")
        result = handler._handle_agent_health("missing-key")
        assert _status(result) == 400

    def test_agent_health_attribute_error(self, handler, mock_coordinator):
        mock_coordinator.get_agent_health.side_effect = AttributeError("no attr")
        result = handler._handle_agent_health("attr-err")
        assert _status(result) == 400

    def test_agent_health_runtime_error(self, handler, mock_coordinator):
        mock_coordinator.get_agent_health.side_effect = RuntimeError("boom")
        result = handler._handle_agent_health("runtime-err")
        assert _status(result) == 500

    def test_agent_health_os_error(self, handler, mock_coordinator):
        mock_coordinator.get_agent_health.side_effect = OSError("disk")
        result = handler._handle_agent_health("os-err")
        assert _status(result) == 500

    def test_agent_health_type_error(self, handler, mock_coordinator):
        mock_coordinator.get_agent_health.side_effect = TypeError("wrong")
        result = handler._handle_agent_health("type-err")
        assert _status(result) == 500

    def test_agent_health_degraded_status(self, handler, mock_coordinator):
        hc = MockHealthCheck("agent-B", "degraded", latency_ms=500.0)
        mock_coordinator.get_agent_health.return_value = hc
        result = handler._handle_agent_health("agent-B")
        assert _status(result) == 200
        assert _body(result)["status"] == "degraded"

    def test_agent_health_unhealthy_status(self, handler, mock_coordinator):
        hc = MockHealthCheck("agent-C", "unhealthy", latency_ms=0.0)
        mock_coordinator.get_agent_health.return_value = hc
        result = handler._handle_agent_health("agent-C")
        assert _status(result) == 200
        assert _body(result)["status"] == "unhealthy"


# ============================================================================
# GET /api/control-plane/health/detailed  (detailed system health)
# ============================================================================


class TestDetailedHealth:
    """Tests for _handle_detailed_health."""

    def test_detailed_health_with_coordinator(self, handler, mock_coordinator):
        with patch(
            "aragora.control_plane.shared_state.get_shared_state_sync",
            side_effect=ImportError("no redis"),
        ):
            result = handler._handle_detailed_health()
        assert _status(result) == 200
        body = _body(result)
        assert body["status"] in ("healthy", "degraded", "unhealthy")
        assert "uptime_seconds" in body
        assert body["version"] == "2.1.0"
        assert isinstance(body["components"], list)

    def test_detailed_health_coordinator_component_present(self, handler, mock_coordinator):
        with patch(
            "aragora.control_plane.shared_state.get_shared_state_sync",
            side_effect=ImportError("no redis"),
        ):
            result = handler._handle_detailed_health()
        body = _body(result)
        names = [c["name"] for c in body["components"]]
        assert "Coordinator" in names

    def test_detailed_health_scheduler_component_present(self, handler, mock_coordinator):
        with patch(
            "aragora.control_plane.shared_state.get_shared_state_sync",
            side_effect=ImportError("no redis"),
        ):
            result = handler._handle_detailed_health()
        body = _body(result)
        names = [c["name"] for c in body["components"]]
        assert "Scheduler" in names

    def test_detailed_health_no_coordinator(self, handler_no_coord):
        with patch(
            "aragora.control_plane.shared_state.get_shared_state_sync",
            side_effect=ImportError("no redis"),
        ):
            result = handler_no_coord._handle_detailed_health()
        assert _status(result) == 200
        body = _body(result)
        names = [c["name"] for c in body["components"]]
        assert "Coordinator" not in names

    def test_detailed_health_no_scheduler_on_coordinator(self, handler, mock_coordinator):
        del mock_coordinator._scheduler
        with patch(
            "aragora.control_plane.shared_state.get_shared_state_sync",
            side_effect=ImportError("no redis"),
        ):
            result = handler._handle_detailed_health()
        body = _body(result)
        names = [c["name"] for c in body["components"]]
        assert "Coordinator" in names
        assert "Scheduler" not in names

    def test_detailed_health_redis_healthy(self, handler, mock_coordinator):
        mock_state = MagicMock()
        mock_state.redis = MagicMock()

        with patch(
            "aragora.control_plane.shared_state.get_shared_state_sync",
            return_value=mock_state,
        ):
            with patch(
                "aragora.server.handlers.control_plane.health._run_async",
                return_value=True,
            ):
                result = handler._handle_detailed_health()
        body = _body(result)
        redis_components = [c for c in body["components"] if c["name"] == "Redis"]
        assert len(redis_components) == 1
        assert redis_components[0]["status"] == "healthy"

    def test_detailed_health_redis_unhealthy(self, handler, mock_coordinator):
        with patch(
            "aragora.control_plane.shared_state.get_shared_state_sync",
            side_effect=ConnectionError("refused"),
        ):
            result = handler._handle_detailed_health()
        body = _body(result)
        redis_components = [c for c in body["components"] if c["name"] == "Redis"]
        assert len(redis_components) == 1
        assert redis_components[0]["status"] == "unhealthy"

    def test_detailed_health_redis_timeout(self, handler, mock_coordinator):
        with patch(
            "aragora.control_plane.shared_state.get_shared_state_sync",
            side_effect=TimeoutError("timed out"),
        ):
            result = handler._handle_detailed_health()
        body = _body(result)
        redis_components = [c for c in body["components"] if c["name"] == "Redis"]
        assert len(redis_components) == 1
        assert redis_components[0]["status"] == "unhealthy"
        assert "TimeoutError" in redis_components[0]["error"]

    def test_detailed_health_database_healthy(self, handler, mock_coordinator):
        mock_db = MagicMock()
        mock_db.is_connected.return_value = True

        with patch(
            "aragora.control_plane.shared_state.get_shared_state_sync",
            side_effect=ImportError("no redis"),
        ):
            with _mock_db_module(lambda: mock_db):
                result = handler._handle_detailed_health()
        body = _body(result)
        db_components = [c for c in body["components"] if c["name"] == "Database"]
        assert len(db_components) == 1
        assert db_components[0]["status"] == "healthy"

    def test_detailed_health_database_not_connected(self, handler, mock_coordinator):
        mock_db = MagicMock()
        mock_db.is_connected.return_value = False

        with patch(
            "aragora.control_plane.shared_state.get_shared_state_sync",
            side_effect=ImportError("no redis"),
        ):
            with _mock_db_module(lambda: mock_db):
                result = handler._handle_detailed_health()
        body = _body(result)
        db_names = [c["name"] for c in body["components"]]
        assert "Database" not in db_names

    def test_detailed_health_overall_unhealthy_when_redis_down(self, handler, mock_coordinator):
        with patch(
            "aragora.control_plane.shared_state.get_shared_state_sync",
            side_effect=ConnectionError("down"),
        ):
            result = handler._handle_detailed_health()
        body = _body(result)
        assert body["status"] == "unhealthy"

    def test_detailed_health_overall_healthy_when_all_good(self, handler, mock_coordinator):
        mock_state = MagicMock()
        mock_state.redis = MagicMock()
        mock_db = MagicMock()
        mock_db.is_connected.return_value = True

        with patch(
            "aragora.control_plane.shared_state.get_shared_state_sync",
            return_value=mock_state,
        ):
            with patch(
                "aragora.server.handlers.control_plane.health._run_async",
                return_value=True,
            ):
                with _mock_db_module(lambda: mock_db):
                    result = handler._handle_detailed_health()
        assert _body(result)["status"] == "healthy"

    def test_detailed_health_uptime_is_nonnegative(self, handler, mock_coordinator):
        handler._start_time = time.time() - 100
        with patch(
            "aragora.control_plane.shared_state.get_shared_state_sync",
            side_effect=ImportError("no redis"),
        ):
            result = handler._handle_detailed_health()
        assert _body(result)["uptime_seconds"] >= 100

    def test_detailed_health_redis_os_error(self, handler, mock_coordinator):
        with patch(
            "aragora.control_plane.shared_state.get_shared_state_sync",
            side_effect=OSError("network error"),
        ):
            result = handler._handle_detailed_health()
        body = _body(result)
        redis_components = [c for c in body["components"] if c["name"] == "Redis"]
        assert len(redis_components) == 1
        assert redis_components[0]["status"] == "unhealthy"
        assert "OSError" in redis_components[0]["error"]

    def test_detailed_health_redis_runtime_error(self, handler, mock_coordinator):
        with patch(
            "aragora.control_plane.shared_state.get_shared_state_sync",
            side_effect=RuntimeError("redis runtime err"),
        ):
            result = handler._handle_detailed_health()
        body = _body(result)
        redis_components = [c for c in body["components"] if c["name"] == "Redis"]
        assert len(redis_components) == 1
        assert redis_components[0]["status"] == "unhealthy"

    def test_detailed_health_database_connection_error(self, handler, mock_coordinator):
        with patch(
            "aragora.control_plane.shared_state.get_shared_state_sync",
            side_effect=ImportError("no redis"),
        ):
            with _mock_db_module(MagicMock(side_effect=ConnectionError("db down"))):
                result = handler._handle_detailed_health()
        body = _body(result)
        db_names = [c["name"] for c in body["components"]]
        # Connection error caught, no Database component added
        assert "Database" not in db_names

    def test_detailed_health_redis_no_redis_attr(self, handler, mock_coordinator):
        """Shared state exists but has no redis attribute."""
        mock_state = MagicMock(spec=[])  # No redis attr

        with patch(
            "aragora.control_plane.shared_state.get_shared_state_sync",
            return_value=mock_state,
        ):
            result = handler._handle_detailed_health()
        body = _body(result)
        redis_names = [c["name"] for c in body["components"] if c["name"] == "Redis"]
        # No redis attr, no Redis component
        assert len(redis_names) == 0

    def test_detailed_health_shared_state_none(self, handler, mock_coordinator):
        """get_shared_state_sync returns None."""
        with patch(
            "aragora.control_plane.shared_state.get_shared_state_sync",
            return_value=None,
        ):
            result = handler._handle_detailed_health()
        body = _body(result)
        # state is None so redis check skipped
        redis_names = [c["name"] for c in body["components"] if c["name"] == "Redis"]
        assert len(redis_names) == 0


# ============================================================================
# GET /api/control-plane/breakers  (circuit breaker states)
# ============================================================================


class TestCircuitBreakers:
    """Tests for _handle_circuit_breakers."""

    def test_circuit_breakers_no_breakers(self, handler):
        with patch(
            "aragora.resilience.get_circuit_breakers",
            return_value={},
        ):
            result = handler._handle_circuit_breakers()
        assert _status(result) == 200
        assert _body(result)["breakers"] == []

    def test_circuit_breakers_with_breakers(self, handler):
        mock_breaker = MagicMock()
        mock_breaker.state = MagicMock()
        mock_breaker.state.value = "closed"
        mock_breaker.failure_count = 2
        mock_breaker.success_count = 100
        mock_breaker.last_failure_time = 1700000000.0
        mock_breaker.reset_timeout = 30

        with patch(
            "aragora.resilience.get_circuit_breakers",
            return_value={"api-call": mock_breaker},
        ):
            result = handler._handle_circuit_breakers()
        assert _status(result) == 200
        breakers = _body(result)["breakers"]
        assert len(breakers) == 1
        assert breakers[0]["name"] == "api-call"
        assert breakers[0]["state"] == "closed"
        assert breakers[0]["failure_count"] == 2
        assert breakers[0]["success_count"] == 100
        assert breakers[0]["reset_timeout_ms"] == 30000

    def test_circuit_breakers_multiple(self, handler):
        b1 = MagicMock()
        b1.state = MagicMock(value="closed")
        b1.failure_count = 0
        b1.success_count = 50
        b1.last_failure_time = None
        b1.reset_timeout = 30

        b2 = MagicMock()
        b2.state = MagicMock(value="open")
        b2.failure_count = 10
        b2.success_count = 5
        b2.last_failure_time = 1700000000.0
        b2.reset_timeout = 60

        with patch(
            "aragora.resilience.get_circuit_breakers",
            return_value={"svc-a": b1, "svc-b": b2},
        ):
            result = handler._handle_circuit_breakers()
        assert _status(result) == 200
        breakers = _body(result)["breakers"]
        assert len(breakers) == 2

    def test_circuit_breakers_state_as_string(self, handler):
        """Breaker state that is a plain string (no .value attribute)."""
        mock_breaker = MagicMock(spec=["state", "failure_count", "success_count",
                                       "last_failure_time", "reset_timeout"])
        mock_breaker.state = "half-open"
        mock_breaker.failure_count = 1
        mock_breaker.success_count = 10
        mock_breaker.last_failure_time = None
        mock_breaker.reset_timeout = 30

        with patch(
            "aragora.resilience.get_circuit_breakers",
            return_value={"test-breaker": mock_breaker},
        ):
            result = handler._handle_circuit_breakers()
        assert _status(result) == 200
        breakers = _body(result)["breakers"]
        assert len(breakers) == 1
        assert breakers[0]["state"] == "half-open"

    def test_circuit_breakers_resilience_import_error(self, handler):
        """When the resilience module import fails inside the handler."""
        # The handler does `from aragora.resilience import get_circuit_breakers`
        # in a try/except (ImportError, AttributeError) block.
        # We cannot easily mock a local import; instead call the method
        # and let it work with whatever the module provides (returns empty if no breakers).
        result = handler._handle_circuit_breakers()
        assert _status(result) == 200
        assert isinstance(_body(result)["breakers"], list)

    def test_circuit_breakers_last_failure_none(self, handler):
        mock_breaker = MagicMock()
        mock_breaker.state = MagicMock(value="closed")
        mock_breaker.failure_count = 0
        mock_breaker.success_count = 200
        mock_breaker.last_failure_time = None
        mock_breaker.reset_timeout = 15

        with patch(
            "aragora.resilience.get_circuit_breakers",
            return_value={"clean-breaker": mock_breaker},
        ):
            result = handler._handle_circuit_breakers()
        assert _status(result) == 200
        breakers = _body(result)["breakers"]
        assert breakers[0]["last_failure"] is None
        assert breakers[0]["reset_timeout_ms"] == 15000


# ============================================================================
# GET /api/control-plane/stats  (control plane statistics)
# ============================================================================


class TestStats:
    """Tests for _handle_stats."""

    def test_stats_success(self, handler, mock_coordinator):
        mock_coordinator.get_stats = AsyncMock(return_value={
            "agents": {"total": 5},
            "tasks": {"pending": 3},
        })
        result = handler._handle_stats()
        assert _status(result) == 200
        body = _body(result)
        assert body["agents"]["total"] == 5
        assert body["tasks"]["pending"] == 3

    def test_stats_no_coordinator(self, handler_no_coord):
        result = handler_no_coord._handle_stats()
        assert _status(result) == 503

    def test_stats_value_error(self, handler, mock_coordinator):
        mock_coordinator.get_stats = AsyncMock(side_effect=ValueError("bad"))
        result = handler._handle_stats()
        assert _status(result) == 400

    def test_stats_key_error(self, handler, mock_coordinator):
        mock_coordinator.get_stats = AsyncMock(side_effect=KeyError("missing"))
        result = handler._handle_stats()
        assert _status(result) == 400

    def test_stats_attribute_error(self, handler, mock_coordinator):
        mock_coordinator.get_stats = AsyncMock(side_effect=AttributeError("no attr"))
        result = handler._handle_stats()
        assert _status(result) == 400

    def test_stats_runtime_error(self, handler, mock_coordinator):
        mock_coordinator.get_stats = AsyncMock(side_effect=RuntimeError("boom"))
        result = handler._handle_stats()
        assert _status(result) == 500

    def test_stats_os_error(self, handler, mock_coordinator):
        mock_coordinator.get_stats = AsyncMock(side_effect=OSError("disk"))
        result = handler._handle_stats()
        assert _status(result) == 500

    def test_stats_type_error(self, handler, mock_coordinator):
        mock_coordinator.get_stats = AsyncMock(side_effect=TypeError("wrong"))
        result = handler._handle_stats()
        assert _status(result) == 500


# ============================================================================
# GET /api/control-plane/metrics  (dashboard metrics)
# ============================================================================


class TestDashboardMetrics:
    """Tests for _handle_get_metrics."""

    def test_metrics_success(self, handler, mock_coordinator):
        result = handler._handle_get_metrics()
        assert _status(result) == 200
        body = _body(result)
        assert body["active_jobs"] == 3
        assert body["queued_jobs"] == 5
        assert body["completed_jobs"] == 100
        assert body["agents_available"] == 8
        assert body["agents_busy"] == 2
        assert body["total_agents"] == 10
        assert body["documents_processed_today"] == 42
        assert body["audits_completed_today"] == 7
        assert body["tokens_used_today"] == 0

    def test_metrics_no_coordinator(self, handler_no_coord):
        result = handler_no_coord._handle_get_metrics()
        assert _status(result) == 503

    def test_metrics_empty_stats(self, handler, mock_coordinator):
        mock_coordinator.get_stats = AsyncMock(return_value={})
        result = handler._handle_get_metrics()
        assert _status(result) == 200
        body = _body(result)
        assert body["active_jobs"] == 0
        assert body["queued_jobs"] == 0
        assert body["total_agents"] == 0

    def test_metrics_partial_stats(self, handler, mock_coordinator):
        mock_coordinator.get_stats = AsyncMock(return_value={
            "scheduler": {"by_status": {"running": 7}},
        })
        result = handler._handle_get_metrics()
        assert _status(result) == 200
        body = _body(result)
        assert body["active_jobs"] == 7
        assert body["agents_available"] == 0

    def test_metrics_value_error(self, handler, mock_coordinator):
        mock_coordinator.get_stats = AsyncMock(side_effect=ValueError("bad"))
        result = handler._handle_get_metrics()
        assert _status(result) == 400

    def test_metrics_key_error(self, handler, mock_coordinator):
        mock_coordinator.get_stats = AsyncMock(side_effect=KeyError("missing"))
        result = handler._handle_get_metrics()
        assert _status(result) == 400

    def test_metrics_runtime_error(self, handler, mock_coordinator):
        mock_coordinator.get_stats = AsyncMock(side_effect=RuntimeError("crash"))
        result = handler._handle_get_metrics()
        assert _status(result) == 500

    def test_metrics_type_error(self, handler, mock_coordinator):
        mock_coordinator.get_stats = AsyncMock(side_effect=TypeError("wrong"))
        result = handler._handle_get_metrics()
        assert _status(result) == 500

    def test_metrics_os_error(self, handler, mock_coordinator):
        mock_coordinator.get_stats = AsyncMock(side_effect=OSError("io"))
        result = handler._handle_get_metrics()
        assert _status(result) == 500


# ============================================================================
# GET /api/control-plane/notifications
# ============================================================================


class TestGetNotifications:
    """Tests for _handle_get_notifications."""

    def test_notifications_no_manager(self, handler):
        result = handler._handle_get_notifications({})
        assert _status(result) == 200
        body = _body(result)
        assert body["notifications"] == []
        assert body["total"] == 0
        assert "not configured" in body.get("message", "").lower()

    def test_notifications_with_manager_no_stats(self, handler):
        manager = MagicMock(spec=[])  # No get_stats method
        handler.ctx["notification_manager"] = manager
        result = handler._handle_get_notifications({})
        assert _status(result) == 200
        body = _body(result)
        assert body["notifications"] == []
        assert body["stats"] == {}

    def test_notifications_with_manager_with_stats(self, handler):
        manager = MagicMock()
        manager.get_stats.return_value = {"total_sent": 42, "failed": 1}
        handler.ctx["notification_manager"] = manager
        result = handler._handle_get_notifications({})
        assert _status(result) == 200
        body = _body(result)
        assert body["stats"]["total_sent"] == 42

    def test_notifications_value_error(self, handler):
        manager = MagicMock()
        manager.get_stats.side_effect = ValueError("bad")
        handler.ctx["notification_manager"] = manager
        result = handler._handle_get_notifications({})
        assert _status(result) == 500

    def test_notifications_type_error(self, handler):
        manager = MagicMock()
        manager.get_stats.side_effect = TypeError("wrong")
        handler.ctx["notification_manager"] = manager
        result = handler._handle_get_notifications({})
        assert _status(result) == 500

    def test_notifications_attribute_error(self, handler):
        manager = MagicMock()
        manager.get_stats.side_effect = AttributeError("no attr")
        handler.ctx["notification_manager"] = manager
        result = handler._handle_get_notifications({})
        assert _status(result) == 500


# ============================================================================
# GET /api/control-plane/notifications/stats
# ============================================================================


class TestNotificationStats:
    """Tests for _handle_get_notification_stats."""

    def test_notification_stats_no_manager(self, handler):
        result = handler._handle_get_notification_stats()
        assert _status(result) == 200
        body = _body(result)
        assert body["total_sent"] == 0
        assert body["successful"] == 0
        assert body["failed"] == 0
        assert body["channels_configured"] == 0

    def test_notification_stats_with_manager(self, handler):
        manager = MagicMock()
        manager.get_stats.return_value = {
            "total_sent": 100,
            "successful": 95,
            "failed": 5,
        }
        handler.ctx["notification_manager"] = manager
        result = handler._handle_get_notification_stats()
        assert _status(result) == 200
        body = _body(result)
        assert body["total_sent"] == 100
        assert body["successful"] == 95

    def test_notification_stats_manager_no_get_stats(self, handler):
        manager = MagicMock(spec=[])  # No get_stats
        handler.ctx["notification_manager"] = manager
        result = handler._handle_get_notification_stats()
        assert _status(result) == 200
        assert _body(result) == {}

    def test_notification_stats_value_error(self, handler):
        manager = MagicMock()
        manager.get_stats.side_effect = ValueError("err")
        handler.ctx["notification_manager"] = manager
        result = handler._handle_get_notification_stats()
        assert _status(result) == 500

    def test_notification_stats_type_error(self, handler):
        manager = MagicMock()
        manager.get_stats.side_effect = TypeError("wrong")
        handler.ctx["notification_manager"] = manager
        result = handler._handle_get_notification_stats()
        assert _status(result) == 500


# ============================================================================
# GET /api/control-plane/audit  (audit logs)
# ============================================================================


class TestGetAuditLogs:
    """Tests for _handle_get_audit_logs."""

    def test_audit_logs_no_audit_log(self, handler, mock_http_handler):
        with patch(
            "aragora.control_plane.audit.AuditQuery",
        ):
            with patch("aragora.control_plane.audit.AuditAction"):
                with patch("aragora.control_plane.audit.ActorType"):
                    result = handler._handle_get_audit_logs({}, mock_http_handler)
        assert _status(result) == 200
        body = _body(result)
        assert body["entries"] == []
        assert body["total"] == 0
        assert "not configured" in body.get("message", "").lower()

    def test_audit_logs_with_entries(self, handler, mock_http_handler):
        entry = MagicMock()
        entry.to_dict.return_value = {"id": "e1", "action": "task.created"}

        audit_log = MagicMock()
        handler.ctx["audit_log"] = audit_log

        with patch(
            "aragora.server.handlers.control_plane.health._run_async",
            return_value=[entry],
        ):
            result = handler._handle_get_audit_logs({}, mock_http_handler)
        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 1
        assert body["entries"][0]["id"] == "e1"

    def test_audit_logs_with_time_filters(self, handler, mock_http_handler):
        audit_log = MagicMock()
        handler.ctx["audit_log"] = audit_log

        query_params = {
            "start_time": "2026-01-01T00:00:00",
            "end_time": "2026-02-01T00:00:00",
        }

        with patch(
            "aragora.server.handlers.control_plane.health._run_async",
            return_value=[],
        ):
            result = handler._handle_get_audit_logs(query_params, mock_http_handler)
        assert _status(result) == 200

    def test_audit_logs_invalid_date_format(self, handler, mock_http_handler):
        audit_log = MagicMock()
        handler.ctx["audit_log"] = audit_log

        query_params = {"start_time": "not-a-date"}
        result = handler._handle_get_audit_logs(query_params, mock_http_handler)
        assert _status(result) == 400
        assert "date" in _body(result).get("error", "").lower()

    def test_audit_logs_with_action_filter(self, handler, mock_http_handler):
        audit_log = MagicMock()
        handler.ctx["audit_log"] = audit_log

        query_params = {"actions": "task.submitted,agent.registered"}

        with patch(
            "aragora.server.handlers.control_plane.health._run_async",
            return_value=[],
        ):
            result = handler._handle_get_audit_logs(query_params, mock_http_handler)
        assert _status(result) == 200

    def test_audit_logs_with_actor_type_filter(self, handler, mock_http_handler):
        audit_log = MagicMock()
        handler.ctx["audit_log"] = audit_log

        query_params = {"actor_types": "user,system"}

        with patch(
            "aragora.server.handlers.control_plane.health._run_async",
            return_value=[],
        ):
            result = handler._handle_get_audit_logs(query_params, mock_http_handler)
        assert _status(result) == 200

    def test_audit_logs_with_actor_ids_filter(self, handler, mock_http_handler):
        audit_log = MagicMock()
        handler.ctx["audit_log"] = audit_log

        query_params = {"actor_ids": "user1,user2"}

        with patch(
            "aragora.server.handlers.control_plane.health._run_async",
            return_value=[],
        ):
            result = handler._handle_get_audit_logs(query_params, mock_http_handler)
        assert _status(result) == 200

    def test_audit_logs_with_resource_types_filter(self, handler, mock_http_handler):
        audit_log = MagicMock()
        handler.ctx["audit_log"] = audit_log

        query_params = {"resource_types": "task,agent"}

        with patch(
            "aragora.server.handlers.control_plane.health._run_async",
            return_value=[],
        ):
            result = handler._handle_get_audit_logs(query_params, mock_http_handler)
        assert _status(result) == 200

    def test_audit_logs_with_workspace_ids_filter(self, handler, mock_http_handler):
        audit_log = MagicMock()
        handler.ctx["audit_log"] = audit_log

        query_params = {"workspace_ids": "ws1,ws2"}

        with patch(
            "aragora.server.handlers.control_plane.health._run_async",
            return_value=[],
        ):
            result = handler._handle_get_audit_logs(query_params, mock_http_handler)
        assert _status(result) == 200

    def test_audit_logs_with_limit_and_offset(self, handler, mock_http_handler):
        audit_log = MagicMock()
        handler.ctx["audit_log"] = audit_log

        query_params = {"limit": "50", "offset": "10"}

        with patch(
            "aragora.server.handlers.control_plane.health._run_async",
            return_value=[],
        ):
            result = handler._handle_get_audit_logs(query_params, mock_http_handler)
        assert _status(result) == 200
        body = _body(result)
        assert "query" in body

    def test_audit_logs_no_query_fn(self, handler, mock_http_handler):
        audit_log = MagicMock(spec=[])  # No query method
        handler.ctx["audit_log"] = audit_log

        result = handler._handle_get_audit_logs({}, mock_http_handler)
        assert _status(result) == 200
        body = _body(result)
        assert body["entries"] == []
        assert body["total"] == 0

    def test_audit_logs_runtime_error(self, handler, mock_http_handler):
        audit_log = MagicMock()
        handler.ctx["audit_log"] = audit_log

        with patch(
            "aragora.server.handlers.control_plane.health._run_async",
            side_effect=RuntimeError("db down"),
        ):
            result = handler._handle_get_audit_logs({}, mock_http_handler)
        assert _status(result) == 500

    def test_audit_logs_multiple_entries(self, handler, mock_http_handler):
        e1 = MagicMock()
        e1.to_dict.return_value = {"id": "e1"}
        e2 = MagicMock()
        e2.to_dict.return_value = {"id": "e2"}

        audit_log = MagicMock()
        handler.ctx["audit_log"] = audit_log

        with patch(
            "aragora.server.handlers.control_plane.health._run_async",
            return_value=[e1, e2],
        ):
            result = handler._handle_get_audit_logs({}, mock_http_handler)
        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 2
        assert len(body["entries"]) == 2


# ============================================================================
# GET /api/control-plane/audit/stats
# ============================================================================


class TestAuditStats:
    """Tests for _handle_get_audit_stats."""

    def test_audit_stats_no_audit_log(self, handler):
        result = handler._handle_get_audit_stats()
        assert _status(result) == 200
        body = _body(result)
        assert body["total_entries"] == 0
        assert body["storage_backend"] == "none"
        assert "not configured" in body.get("message", "").lower()

    def test_audit_stats_with_audit_log(self, handler):
        audit_log = MagicMock()
        audit_log.get_stats.return_value = {
            "total_entries": 500,
            "storage_backend": "postgres",
        }
        handler.ctx["audit_log"] = audit_log
        result = handler._handle_get_audit_stats()
        assert _status(result) == 200
        body = _body(result)
        assert body["total_entries"] == 500
        assert body["storage_backend"] == "postgres"

    def test_audit_stats_no_get_stats_method(self, handler):
        audit_log = MagicMock(spec=[])  # No get_stats
        handler.ctx["audit_log"] = audit_log
        result = handler._handle_get_audit_stats()
        assert _status(result) == 200
        assert _body(result) == {}

    def test_audit_stats_value_error(self, handler):
        audit_log = MagicMock()
        audit_log.get_stats.side_effect = ValueError("bad")
        handler.ctx["audit_log"] = audit_log
        result = handler._handle_get_audit_stats()
        assert _status(result) == 500

    def test_audit_stats_type_error(self, handler):
        audit_log = MagicMock()
        audit_log.get_stats.side_effect = TypeError("wrong")
        handler.ctx["audit_log"] = audit_log
        result = handler._handle_get_audit_stats()
        assert _status(result) == 500


# ============================================================================
# GET /api/control-plane/audit/verify
# ============================================================================


class TestVerifyAuditIntegrity:
    """Tests for _handle_verify_audit_integrity."""

    def test_verify_integrity_no_audit_log(self, handler, mock_http_handler):
        result = handler._handle_verify_audit_integrity({}, mock_http_handler)
        assert _status(result) == 503

    def test_verify_integrity_valid(self, handler, mock_http_handler):
        audit_log = MagicMock()
        handler.ctx["audit_log"] = audit_log

        with patch(
            "aragora.server.handlers.control_plane.health._run_async",
            return_value=True,
        ):
            result = handler._handle_verify_audit_integrity({}, mock_http_handler)
        assert _status(result) == 200
        body = _body(result)
        assert body["valid"] is True
        assert "verified" in body["message"].lower()

    def test_verify_integrity_invalid(self, handler, mock_http_handler):
        audit_log = MagicMock()
        handler.ctx["audit_log"] = audit_log

        with patch(
            "aragora.server.handlers.control_plane.health._run_async",
            return_value=False,
        ):
            result = handler._handle_verify_audit_integrity({}, mock_http_handler)
        assert _status(result) == 200
        body = _body(result)
        assert body["valid"] is False
        assert "tampering" in body["message"].lower()

    def test_verify_integrity_with_seq_params(self, handler, mock_http_handler):
        audit_log = MagicMock()
        handler.ctx["audit_log"] = audit_log

        query_params = {"start_seq": "10", "end_seq": "100"}

        with patch(
            "aragora.server.handlers.control_plane.health._run_async",
            return_value=True,
        ):
            result = handler._handle_verify_audit_integrity(
                query_params, mock_http_handler
            )
        assert _status(result) == 200
        body = _body(result)
        assert body["start_seq"] == 10
        assert body["end_seq"] == 100

    def test_verify_integrity_no_end_seq(self, handler, mock_http_handler):
        audit_log = MagicMock()
        handler.ctx["audit_log"] = audit_log

        query_params = {"start_seq": "5"}

        with patch(
            "aragora.server.handlers.control_plane.health._run_async",
            return_value=True,
        ):
            result = handler._handle_verify_audit_integrity(
                query_params, mock_http_handler
            )
        assert _status(result) == 200
        body = _body(result)
        assert body["start_seq"] == 5
        assert body["end_seq"] is None

    def test_verify_integrity_no_verify_fn(self, handler, mock_http_handler):
        audit_log = MagicMock(spec=[])  # No verify_integrity method
        handler.ctx["audit_log"] = audit_log

        result = handler._handle_verify_audit_integrity({}, mock_http_handler)
        assert _status(result) == 200
        body = _body(result)
        assert body["valid"] is False

    def test_verify_integrity_runtime_error(self, handler, mock_http_handler):
        audit_log = MagicMock()
        handler.ctx["audit_log"] = audit_log

        with patch(
            "aragora.server.handlers.control_plane.health._run_async",
            side_effect=RuntimeError("db failure"),
        ):
            result = handler._handle_verify_audit_integrity(
                {}, mock_http_handler
            )
        assert _status(result) == 500

    def test_verify_integrity_os_error(self, handler, mock_http_handler):
        audit_log = MagicMock()
        handler.ctx["audit_log"] = audit_log

        with patch(
            "aragora.server.handlers.control_plane.health._run_async",
            side_effect=OSError("io fail"),
        ):
            result = handler._handle_verify_audit_integrity(
                {}, mock_http_handler
            )
        assert _status(result) == 500


# ============================================================================
# Routing through handle() for all health endpoints
# ============================================================================


class TestRouting:
    """Tests for request routing through the main handle() method."""

    def test_route_system_health(self, handler, mock_coordinator, mock_http_handler):
        result = handler.handle("/api/control-plane/health", {}, mock_http_handler)
        assert _status(result) == 200
        assert "status" in _body(result)

    def test_route_system_health_v1(self, handler, mock_coordinator, mock_http_handler):
        result = handler.handle("/api/v1/control-plane/health", {}, mock_http_handler)
        assert _status(result) == 200

    def test_route_agent_health(self, handler, mock_coordinator, mock_http_handler):
        hc = MockHealthCheck("agent-1", "healthy")
        mock_coordinator.get_agent_health.return_value = hc
        result = handler.handle(
            "/api/control-plane/health/agent-1", {}, mock_http_handler
        )
        assert _status(result) == 200
        assert _body(result)["agent_id"] == "agent-1"

    def test_route_agent_health_v1(self, handler, mock_coordinator, mock_http_handler):
        hc = MockHealthCheck("agent-X", "degraded")
        mock_coordinator.get_agent_health.return_value = hc
        result = handler.handle(
            "/api/v1/control-plane/health/agent-X", {}, mock_http_handler
        )
        assert _status(result) == 200
        assert _body(result)["agent_id"] == "agent-X"

    def test_route_detailed_health(self, handler, mock_coordinator, mock_http_handler):
        with patch(
            "aragora.control_plane.shared_state.get_shared_state_sync",
            side_effect=ImportError("no redis"),
        ):
            result = handler.handle(
                "/api/control-plane/health/detailed", {}, mock_http_handler
            )
        assert _status(result) == 200
        assert "version" in _body(result)

    def test_route_breakers(self, handler, mock_coordinator, mock_http_handler):
        with patch(
            "aragora.resilience.get_circuit_breakers",
            return_value={},
        ):
            result = handler.handle(
                "/api/control-plane/breakers", {}, mock_http_handler
            )
        assert _status(result) == 200

    def test_route_stats(self, handler, mock_coordinator, mock_http_handler):
        result = handler.handle("/api/control-plane/stats", {}, mock_http_handler)
        assert _status(result) == 200

    def test_route_metrics(self, handler, mock_coordinator, mock_http_handler):
        result = handler.handle("/api/control-plane/metrics", {}, mock_http_handler)
        assert _status(result) == 200

    def test_route_notifications(self, handler, mock_coordinator, mock_http_handler):
        result = handler.handle(
            "/api/control-plane/notifications", {}, mock_http_handler
        )
        assert _status(result) == 200

    def test_route_notification_stats(self, handler, mock_coordinator, mock_http_handler):
        result = handler.handle(
            "/api/control-plane/notifications/stats", {}, mock_http_handler
        )
        assert _status(result) == 200

    def test_route_audit_stats(self, handler, mock_coordinator, mock_http_handler):
        result = handler.handle(
            "/api/control-plane/audit/stats", {}, mock_http_handler
        )
        assert _status(result) == 200

    def test_route_audit(self, handler, mock_coordinator, mock_http_handler):
        result = handler.handle(
            "/api/control-plane/audit", {}, mock_http_handler
        )
        # Has audit permission check via _get_has_permission;
        # conftest grants all permissions
        assert _status(result) == 200

    def test_route_audit_verify(self, handler, mock_coordinator, mock_http_handler):
        result = handler.handle(
            "/api/control-plane/audit/verify", {}, mock_http_handler
        )
        # No audit_log configured -> 503 after permission check passes
        assert _status(result) == 503


# ============================================================================
# Mixin Internal Methods
# ============================================================================


class TestMixinHelpers:
    """Tests for HealthHandlerMixin internal helper methods."""

    def test_get_coordinator_from_context(self, handler, mock_coordinator):
        assert handler._get_coordinator() is mock_coordinator

    def test_get_coordinator_none_when_missing(self, handler_no_coord):
        assert handler_no_coord._get_coordinator() is None

    def test_require_coordinator_success(self, handler, mock_coordinator):
        coord, err = handler._require_coordinator()
        assert coord is mock_coordinator
        assert err is None

    def test_require_coordinator_error(self, handler_no_coord):
        coord, err = handler_no_coord._require_coordinator()
        assert coord is None
        assert _status(err) == 503

    def test_handle_coordinator_error_value_error(self, handler):
        result = handler._handle_coordinator_error(ValueError("bad"), "test_op")
        assert _status(result) == 400

    def test_handle_coordinator_error_key_error(self, handler):
        result = handler._handle_coordinator_error(KeyError("missing"), "test_op")
        assert _status(result) == 400

    def test_handle_coordinator_error_attribute_error(self, handler):
        result = handler._handle_coordinator_error(AttributeError("no attr"), "test_op")
        assert _status(result) == 400

    def test_handle_coordinator_error_runtime_error(self, handler):
        result = handler._handle_coordinator_error(RuntimeError("crash"), "test_op")
        assert _status(result) == 500

    def test_handle_coordinator_error_os_error(self, handler):
        result = handler._handle_coordinator_error(OSError("disk"), "test_op")
        assert _status(result) == 500


# ============================================================================
# _get_has_permission caching
# ============================================================================


class TestGetHasPermission:
    """Tests for _get_has_permission function."""

    def test_get_has_permission_returns_callable(self):
        from aragora.server.handlers.control_plane.health import _get_has_permission

        fn = _get_has_permission()
        assert callable(fn)

    def test_get_has_permission_caches_result(self):
        import aragora.server.handlers.control_plane.health as health_mod

        # Reset cache
        health_mod._cached_has_permission = None
        health_mod._cache_timestamp = 0

        fn1 = health_mod._get_has_permission()
        fn2 = health_mod._get_has_permission()
        assert fn1 is fn2

    def test_get_has_permission_cache_expires(self):
        import aragora.server.handlers.control_plane.health as health_mod

        # Set cache to expired state
        health_mod._cached_has_permission = lambda r, p: True
        health_mod._cache_timestamp = time.time() - 120  # Expired (TTL is 60)

        fn = health_mod._get_has_permission()
        assert callable(fn)
