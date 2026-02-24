"""Tests for System Health Dashboard handler.

Covers all routes and behaviour of the SystemHealthDashboardHandler class:
- GET /api/admin/system-health              - Aggregated health overview
- GET /api/admin/system-health/circuit-breakers - Circuit breaker states
- GET /api/admin/system-health/slos          - SLO compliance status
- GET /api/admin/system-health/adapters      - KM adapter health
- GET /api/admin/system-health/agents        - Agent pool health
- GET /api/admin/system-health/budget        - Budget utilization
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.system_health import SystemHealthDashboardHandler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_response(result) -> dict:
    """Extract data from json_response HandlerResult."""
    if hasattr(result, "body"):
        body = result.body
        if isinstance(body, (bytes, bytearray)):
            body = body.decode("utf-8")
        if isinstance(body, str):
            body = json.loads(body)
        if isinstance(body, dict):
            return body
    if isinstance(result, tuple):
        body = result[0] if len(result) > 0 else {}
        if isinstance(body, str):
            body = json.loads(body)
        return body
    if isinstance(result, dict):
        return result
    return {}


def _get_data(result) -> dict:
    """Extract the 'data' envelope from a response."""
    body = _parse_response(result)
    if isinstance(body, dict) and "data" in body:
        return body["data"]
    return body


def _get_status_code(result) -> int:
    """Extract HTTP status code from HandlerResult."""
    if hasattr(result, "status_code"):
        return result.status_code
    if isinstance(result, tuple) and len(result) > 1:
        return result[1]
    return 200


# ---------------------------------------------------------------------------
# Mock data helpers
# ---------------------------------------------------------------------------


def _make_mock_circuit_breaker(
    name: str = "test-cb",
    state: str = "closed",
    failures: int = 0,
    threshold: int = 3,
    cooldown: int = 60,
):
    """Create a mock circuit breaker object."""
    cb = MagicMock()
    cb.get_status.return_value = state
    cb._single_failures = failures
    cb.failure_threshold = threshold
    cb.cooldown_seconds = cooldown
    return cb


@dataclass
class MockAdapterSpec:
    """Mock KM adapter spec."""

    enabled_by_default: bool = True
    priority: int = 100
    reverse_method: str | None = "sync_back"


@dataclass
class MockSloResult:
    """Mock SLO result entry."""

    name: str = "availability"
    target: float = 0.999
    current: float = 0.9995
    compliant: bool = True
    compliance_percentage: float = 99.95
    error_budget_remaining: float = 50.0
    burn_rate: float = 0.5


@dataclass
class MockSloStatus:
    """Mock SLO status collection."""

    availability: MockSloResult = field(default_factory=MockSloResult)
    latency_p99: MockSloResult = field(
        default_factory=lambda: MockSloResult(name="latency_p99")
    )
    debate_success: MockSloResult = field(
        default_factory=lambda: MockSloResult(name="debate_success")
    )
    overall_healthy: bool = True
    timestamp: Any = None

    def __post_init__(self):
        if self.timestamp is None:
            from datetime import datetime, timezone

            self.timestamp = datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def handler():
    """Create a SystemHealthDashboardHandler instance."""
    return SystemHealthDashboardHandler(server_context={})


@pytest.fixture
def mock_http_handler():
    """Create a mock HTTP handler for passing to handle()."""
    h = MagicMock()
    h.rfile = MagicMock()
    h.rfile.read.return_value = b"{}"
    h.headers = {"Content-Length": "2"}
    return h


# ---------------------------------------------------------------------------
# ROUTES / can_handle
# ---------------------------------------------------------------------------


class TestRoutes:
    """Test ROUTES class attribute and can_handle method."""

    def test_routes_contains_expected(self):
        expected = [
            "/api/admin/system-health",
            "/api/admin/system-health/*",
        ]
        for route in expected:
            assert route in SystemHealthDashboardHandler.ROUTES, f"Missing route: {route}"

    def test_can_handle_main_path(self, handler):
        assert handler.can_handle("/api/admin/system-health")

    def test_can_handle_sub_paths(self, handler):
        sub_paths = [
            "/api/admin/system-health/circuit-breakers",
            "/api/admin/system-health/slos",
            "/api/admin/system-health/adapters",
            "/api/admin/system-health/agents",
            "/api/admin/system-health/budget",
        ]
        for path in sub_paths:
            assert handler.can_handle(path), f"Should handle: {path}"

    def test_can_handle_rejects_unknown_paths(self, handler):
        assert not handler.can_handle("/api/admin/system-health-other")
        assert not handler.can_handle("/api/admin/other")
        assert not handler.can_handle("/api/v1/system-health")

    def test_can_handle_only_accepts_get(self, handler):
        assert handler.can_handle("/api/admin/system-health", "GET")
        assert not handler.can_handle("/api/admin/system-health", "POST")
        assert not handler.can_handle("/api/admin/system-health", "PUT")
        assert not handler.can_handle("/api/admin/system-health", "DELETE")


# ---------------------------------------------------------------------------
# Route dispatch
# ---------------------------------------------------------------------------


class TestRouteDispatch:
    """Test that handle() routes to the correct method."""

    @pytest.mark.asyncio
    async def test_dispatch_overview(self, handler, mock_http_handler):
        result = await handler.handle("/api/admin/system-health", {}, mock_http_handler)
        data = _get_data(result)
        assert "overall_status" in data
        assert "subsystems" in data

    @pytest.mark.asyncio
    async def test_dispatch_circuit_breakers(self, handler, mock_http_handler):
        result = await handler.handle(
            "/api/admin/system-health/circuit-breakers", {}, mock_http_handler
        )
        data = _get_data(result)
        assert "breakers" in data

    @pytest.mark.asyncio
    async def test_dispatch_slos(self, handler, mock_http_handler):
        result = await handler.handle(
            "/api/admin/system-health/slos", {}, mock_http_handler
        )
        data = _get_data(result)
        assert "slos" in data

    @pytest.mark.asyncio
    async def test_dispatch_adapters(self, handler, mock_http_handler):
        result = await handler.handle(
            "/api/admin/system-health/adapters", {}, mock_http_handler
        )
        data = _get_data(result)
        assert "adapters" in data

    @pytest.mark.asyncio
    async def test_dispatch_agents(self, handler, mock_http_handler):
        result = await handler.handle(
            "/api/admin/system-health/agents", {}, mock_http_handler
        )
        data = _get_data(result)
        assert "agents" in data

    @pytest.mark.asyncio
    async def test_dispatch_budget(self, handler, mock_http_handler):
        result = await handler.handle(
            "/api/admin/system-health/budget", {}, mock_http_handler
        )
        data = _get_data(result)
        assert "available" in data

    @pytest.mark.asyncio
    async def test_dispatch_unknown_returns_404(self, handler, mock_http_handler):
        result = await handler.handle(
            "/api/admin/system-health/nonexistent", {}, mock_http_handler
        )
        status = _get_status_code(result)
        assert status == 404


# ---------------------------------------------------------------------------
# Circuit breakers endpoint
# ---------------------------------------------------------------------------


class TestCircuitBreakers:
    """Test the circuit breakers sub-endpoint."""

    def test_collect_circuit_breakers_available(self, handler):
        breakers = {
            "api-cb": _make_mock_circuit_breaker("api-cb", "closed", 0, 5),
            "db-cb": _make_mock_circuit_breaker("db-cb", "open", 3, 3),
        }
        with patch(
            "aragora.server.handlers.system_health.SystemHealthDashboardHandler._collect_circuit_breakers",
            wraps=handler._collect_circuit_breakers,
        ):
            with patch(
                "aragora.resilience.registry.get_circuit_breakers",
                return_value=breakers,
            ):
                result = handler._collect_circuit_breakers()

        assert result["available"] is True
        assert result["total"] == 2
        assert len(result["breakers"]) == 2

    def test_collect_circuit_breakers_entry_shape(self, handler):
        breakers = {
            "test-cb": _make_mock_circuit_breaker("test-cb", "closed", 1, 5, 30),
        }
        with patch(
            "aragora.resilience.registry.get_circuit_breakers",
            return_value=breakers,
        ):
            result = handler._collect_circuit_breakers()

        entry = result["breakers"][0]
        assert entry["name"] == "test-cb"
        assert entry["state"] == "closed"
        assert entry["failure_count"] == 1
        assert entry["failure_threshold"] == 5
        assert entry["cooldown_seconds"] == 30
        assert 0.0 <= entry["success_rate"] <= 1.0
        assert entry["last_failure"] is None

    def test_collect_circuit_breakers_success_rate_calculation(self, handler):
        breakers = {
            "cb-full": _make_mock_circuit_breaker("cb-full", "open", 3, 3),
        }
        with patch(
            "aragora.resilience.registry.get_circuit_breakers",
            return_value=breakers,
        ):
            result = handler._collect_circuit_breakers()

        # failures == threshold => success_rate = max(0, 1 - 3/3) = 0
        assert result["breakers"][0]["success_rate"] == 0.0

    def test_collect_circuit_breakers_zero_failures(self, handler):
        breakers = {
            "cb-ok": _make_mock_circuit_breaker("cb-ok", "closed", 0, 5),
        }
        with patch(
            "aragora.resilience.registry.get_circuit_breakers",
            return_value=breakers,
        ):
            result = handler._collect_circuit_breakers()

        # No failures => success_rate = 1.0
        assert result["breakers"][0]["success_rate"] == 1.0

    def test_collect_circuit_breakers_import_error(self, handler):
        with patch(
            "builtins.__import__",
            side_effect=_selective_import_error("aragora.resilience.registry"),
        ):
            result = handler._collect_circuit_breakers()

        assert result["available"] is False
        assert result["breakers"] == []

    def test_collect_circuit_breakers_no_get_status(self, handler):
        """Circuit breaker without get_status method uses 'unknown'."""
        cb = MagicMock(spec=[])  # no attributes
        cb._single_failures = 0
        cb.failure_threshold = 3
        cb.cooldown_seconds = 60
        breakers = {"bare-cb": cb}
        with patch(
            "aragora.resilience.registry.get_circuit_breakers",
            return_value=breakers,
        ):
            result = handler._collect_circuit_breakers()

        assert result["breakers"][0]["state"] == "unknown"


# ---------------------------------------------------------------------------
# SLOs endpoint
# ---------------------------------------------------------------------------


class TestSLOs:
    """Test the SLO sub-endpoint."""

    def test_collect_slos_available(self, handler):
        with patch(
            "aragora.observability.slo.get_slo_status",
            return_value=MockSloStatus(),
        ):
            result = handler._collect_slos()

        assert result["available"] is True
        assert len(result["slos"]) == 3
        assert "timestamp" in result
        assert "overall_healthy" in result

    def test_collect_slos_entry_shape(self, handler):
        with patch(
            "aragora.observability.slo.get_slo_status",
            return_value=MockSloStatus(),
        ):
            result = handler._collect_slos()

        slo = result["slos"][0]
        expected_keys = [
            "name", "key", "target", "current",
            "compliant", "compliance_percentage",
            "error_budget_remaining", "burn_rate",
        ]
        for key in expected_keys:
            assert key in slo, f"Missing key: {key}"

    def test_collect_slos_non_compliant(self, handler):
        bad_slo = MockSloResult(compliant=False, compliance_percentage=90.0)
        status = MockSloStatus(
            availability=bad_slo,
            overall_healthy=False,
        )
        with patch(
            "aragora.observability.slo.get_slo_status",
            return_value=status,
        ):
            result = handler._collect_slos()

        assert result["overall_healthy"] is False

    def test_collect_slos_import_error(self, handler):
        with patch(
            "builtins.__import__",
            side_effect=_selective_import_error("aragora.observability.slo"),
        ):
            result = handler._collect_slos()

        assert result["available"] is False
        assert result["slos"] == []
        assert result["overall_healthy"] is True


# ---------------------------------------------------------------------------
# Adapters endpoint
# ---------------------------------------------------------------------------


class TestAdapters:
    """Test the KM adapter health sub-endpoint."""

    def test_collect_adapters_available(self, handler):
        specs = {
            "continuum": MockAdapterSpec(enabled_by_default=True, priority=100),
            "consensus": MockAdapterSpec(enabled_by_default=True, priority=90),
            "disabled_one": MockAdapterSpec(enabled_by_default=False, priority=10),
        }
        with patch(
            "aragora.knowledge.mound.adapters.factory.ADAPTER_SPECS",
            specs,
        ):
            result = handler._collect_adapters()

        assert result["available"] is True
        assert result["total"] == 3
        assert result["active"] == 2  # 2 enabled by default

    def test_collect_adapters_sorted_by_priority_desc(self, handler):
        specs = {
            "low": MockAdapterSpec(priority=10),
            "high": MockAdapterSpec(priority=200),
            "mid": MockAdapterSpec(priority=50),
        }
        with patch(
            "aragora.knowledge.mound.adapters.factory.ADAPTER_SPECS",
            specs,
        ):
            result = handler._collect_adapters()

        priorities = [a["priority"] for a in result["adapters"]]
        assert priorities == sorted(priorities, reverse=True)

    def test_collect_adapters_entry_shape(self, handler):
        specs = {
            "test": MockAdapterSpec(
                enabled_by_default=True,
                priority=100,
                reverse_method="sync_back",
            ),
        }
        with patch(
            "aragora.knowledge.mound.adapters.factory.ADAPTER_SPECS",
            specs,
        ):
            result = handler._collect_adapters()

        adapter = result["adapters"][0]
        assert adapter["name"] == "test"
        assert adapter["enabled_by_default"] is True
        assert adapter["priority"] == 100
        assert adapter["has_reverse_sync"] is True

    def test_collect_adapters_no_reverse_method(self, handler):
        specs = {
            "no_reverse": MockAdapterSpec(reverse_method=None),
        }
        with patch(
            "aragora.knowledge.mound.adapters.factory.ADAPTER_SPECS",
            specs,
        ):
            result = handler._collect_adapters()

        assert result["adapters"][0]["has_reverse_sync"] is False

    def test_collect_adapters_import_error(self, handler):
        with patch(
            "builtins.__import__",
            side_effect=_selective_import_error(
                "aragora.knowledge.mound.adapters.factory"
            ),
        ):
            result = handler._collect_adapters()

        assert result["available"] is False
        assert result["adapters"] == []
        assert result["total"] == 0
        assert result["active"] == 0


# ---------------------------------------------------------------------------
# Agents endpoint
# ---------------------------------------------------------------------------


class TestAgents:
    """Test the agent pool health sub-endpoint."""

    def test_collect_agents_with_dict_agents(self, handler):
        registry = MagicMock()
        registry.list_agents.return_value = [
            {"agent_id": "a1", "type": "claude", "status": "active", "last_heartbeat": "2026-02-01T00:00:00Z"},
            {"agent_id": "a2", "type": "gpt4", "status": "idle", "last_heartbeat": "2026-02-01T00:00:00Z"},
            {"agent_id": "a3", "type": "gemini", "status": "failed", "last_heartbeat": "2026-02-01T00:00:00Z"},
        ]
        with patch(
            "aragora.control_plane.registry.get_default_registry",
            create=True,
            return_value=registry,
        ):
            result = handler._collect_agents()

        assert result["available"] is True
        assert result["total"] == 3
        assert result["active"] == 2  # active + idle

    def test_collect_agents_with_object_agents(self, handler):
        agent_obj = MagicMock()
        agent_obj.agent_id = "obj-1"
        agent_obj.type = "claude"
        agent_obj.status = "active"
        agent_obj.last_heartbeat = "2026-01-01"

        registry = MagicMock()
        registry.list_agents.return_value = [agent_obj]
        with patch(
            "aragora.control_plane.registry.get_default_registry",
            create=True,
            return_value=registry,
        ):
            result = handler._collect_agents()

        assert result["total"] == 1
        assert result["active"] == 1
        agent = result["agents"][0]
        assert agent["agent_id"] == "obj-1"
        assert agent["type"] == "claude"
        assert agent["status"] == "active"

    def test_collect_agents_registry_none(self, handler):
        with patch(
            "aragora.control_plane.registry.get_default_registry",
            create=True,
            return_value=None,
        ):
            result = handler._collect_agents()

        assert result["available"] is False
        assert result["agents"] == []
        assert result["total"] == 0

    def test_collect_agents_import_error(self, handler):
        with patch(
            "builtins.__import__",
            side_effect=_selective_import_error("aragora.control_plane.registry"),
        ):
            result = handler._collect_agents()

        assert result["available"] is False
        assert result["agents"] == []

    def test_collect_agents_empty_registry(self, handler):
        registry = MagicMock()
        registry.list_agents.return_value = []
        with patch(
            "aragora.control_plane.registry.get_default_registry",
            create=True,
            return_value=registry,
        ):
            result = handler._collect_agents()

        assert result["available"] is True
        assert result["total"] == 0
        assert result["active"] == 0

    def test_collect_agents_fallback_id_key(self, handler):
        """Test that 'id' key is used as fallback when 'agent_id' is absent."""
        registry = MagicMock()
        registry.list_agents.return_value = [
            {"id": "fallback-id", "agent_type": "claude", "status": "active"},
        ]
        with patch(
            "aragora.control_plane.registry.get_default_registry",
            create=True,
            return_value=registry,
        ):
            result = handler._collect_agents()

        assert result["agents"][0]["agent_id"] == "fallback-id"
        assert result["agents"][0]["type"] == "claude"


# ---------------------------------------------------------------------------
# Budget endpoint
# ---------------------------------------------------------------------------


class TestBudget:
    """Test the budget utilization sub-endpoint."""

    def test_collect_budget_with_dict_summary(self, handler):
        tracker = MagicMock()
        tracker.get_summary.return_value = {
            "budget_usd": 100.0,
            "total_cost_usd": 45.0,
        }
        with patch(
            "aragora.billing.cost_tracker.get_cost_tracker",
            return_value=tracker,
        ):
            with patch(
                "aragora.billing.forecaster.get_cost_forecaster", create=True,
                return_value=None,
            ):
                result = handler._collect_budget()

        assert result["available"] is True
        assert result["total_budget"] == 100.0
        assert result["spent"] == 45.0
        assert result["utilization"] == 0.45

    def test_collect_budget_with_object_summary(self, handler):
        summary_obj = MagicMock()
        summary_obj.budget_usd = 200.0
        summary_obj.total_cost_usd = 180.0
        # Make isinstance(summary, dict) return False
        summary_obj.__class__ = type("BudgetSummary", (), {})

        tracker = MagicMock()
        tracker.get_summary.return_value = summary_obj
        with patch(
            "aragora.billing.cost_tracker.get_cost_tracker",
            return_value=tracker,
        ):
            with patch(
                "aragora.billing.forecaster.get_cost_forecaster", create=True,
                return_value=None,
            ):
                result = handler._collect_budget()

        assert result["available"] is True
        assert result["total_budget"] == 200.0
        assert result["spent"] == 180.0
        assert result["utilization"] == 0.9

    def test_collect_budget_with_forecast_dict(self, handler):
        tracker = MagicMock()
        tracker.get_summary.return_value = {
            "budget_usd": 100.0,
            "total_cost_usd": 50.0,
        }
        forecaster = MagicMock()
        forecaster.forecast_eom.return_value = {
            "projected": 75.0,
            "trend": "increasing",
        }
        with patch(
            "aragora.billing.cost_tracker.get_cost_tracker",
            return_value=tracker,
        ):
            with patch(
                "aragora.billing.forecaster.get_cost_forecaster", create=True,
                return_value=forecaster,
            ):
                result = handler._collect_budget()

        assert result["forecast"] is not None
        assert result["forecast"]["eom"] == 75.0
        assert result["forecast"]["trend"] == "increasing"

    def test_collect_budget_with_forecast_numeric(self, handler):
        tracker = MagicMock()
        tracker.get_summary.return_value = {
            "budget_usd": 100.0,
            "total_cost_usd": 50.0,
        }
        forecaster = MagicMock()
        # numeric forecast: eom > spent * 1.2 => "increasing"
        forecaster.forecast_eom.return_value = 80.0
        with patch(
            "aragora.billing.cost_tracker.get_cost_tracker",
            return_value=tracker,
        ):
            with patch(
                "aragora.billing.forecaster.get_cost_forecaster", create=True,
                return_value=forecaster,
            ):
                result = handler._collect_budget()

        assert result["forecast"]["eom"] == 80.0
        assert result["forecast"]["trend"] == "increasing"

    def test_collect_budget_numeric_forecast_decreasing(self, handler):
        tracker = MagicMock()
        tracker.get_summary.return_value = {
            "budget_usd": 100.0,
            "total_cost_usd": 50.0,
        }
        forecaster = MagicMock()
        # eom < spent * 0.8 = 40 => "decreasing"
        forecaster.forecast_eom.return_value = 35.0
        with patch(
            "aragora.billing.cost_tracker.get_cost_tracker",
            return_value=tracker,
        ):
            with patch(
                "aragora.billing.forecaster.get_cost_forecaster", create=True,
                return_value=forecaster,
            ):
                result = handler._collect_budget()

        assert result["forecast"]["trend"] == "decreasing"

    def test_collect_budget_numeric_forecast_stable(self, handler):
        tracker = MagicMock()
        tracker.get_summary.return_value = {
            "budget_usd": 100.0,
            "total_cost_usd": 50.0,
        }
        forecaster = MagicMock()
        # eom between spent*0.8 (40) and spent*1.2 (60) => "stable"
        forecaster.forecast_eom.return_value = 55.0
        with patch(
            "aragora.billing.cost_tracker.get_cost_tracker",
            return_value=tracker,
        ):
            with patch(
                "aragora.billing.forecaster.get_cost_forecaster", create=True,
                return_value=forecaster,
            ):
                result = handler._collect_budget()

        assert result["forecast"]["trend"] == "stable"

    def test_collect_budget_tracker_none(self, handler):
        with patch(
            "aragora.billing.cost_tracker.get_cost_tracker",
            return_value=None,
        ):
            result = handler._collect_budget()

        assert result["available"] is False
        assert result["total_budget"] == 0
        assert result["spent"] == 0

    def test_collect_budget_import_error(self, handler):
        with patch(
            "builtins.__import__",
            side_effect=_selective_import_error("aragora.billing.cost_tracker"),
        ):
            result = handler._collect_budget()

        assert result["available"] is False
        assert result["forecast"] is None

    def test_collect_budget_forecaster_import_error(self, handler):
        """Forecast import failure should still return budget data without forecast."""
        tracker = MagicMock()
        tracker.get_summary.return_value = {
            "budget_usd": 100.0,
            "total_cost_usd": 60.0,
        }
        with patch(
            "aragora.billing.cost_tracker.get_cost_tracker",
            return_value=tracker,
        ):
            with patch(
                "builtins.__import__",
                side_effect=_selective_import_error("aragora.billing.forecaster"),
            ):
                result = handler._collect_budget()

        assert result["available"] is True
        assert result["total_budget"] == 100.0
        assert result["forecast"] is None

    def test_collect_budget_zero_total(self, handler):
        """Zero total budget => utilization = 0 (no division by zero)."""
        tracker = MagicMock()
        tracker.get_summary.return_value = {
            "budget_usd": 0,
            "total_cost_usd": 0,
        }
        with patch(
            "aragora.billing.cost_tracker.get_cost_tracker",
            return_value=tracker,
        ):
            with patch(
                "aragora.billing.forecaster.get_cost_forecaster", create=True,
                return_value=None,
            ):
                result = handler._collect_budget()

        assert result["utilization"] == 0

    def test_collect_budget_forecast_eom_none(self, handler):
        """forecast_eom returning None => no forecast populated."""
        tracker = MagicMock()
        tracker.get_summary.return_value = {
            "budget_usd": 100.0,
            "total_cost_usd": 20.0,
        }
        forecaster = MagicMock()
        forecaster.forecast_eom.return_value = None
        with patch(
            "aragora.billing.cost_tracker.get_cost_tracker",
            return_value=tracker,
        ):
            with patch(
                "aragora.billing.forecaster.get_cost_forecaster", create=True,
                return_value=forecaster,
            ):
                result = handler._collect_budget()

        assert result["forecast"] is None


# ---------------------------------------------------------------------------
# Overview endpoint (aggregated)
# ---------------------------------------------------------------------------


class TestOverview:
    """Test the aggregated health overview endpoint."""

    def test_overview_all_healthy(self, handler):
        with (
            patch.object(
                handler, "_collect_circuit_breakers",
                return_value={"breakers": [], "total": 0, "available": True},
            ),
            patch.object(
                handler, "_collect_slos",
                return_value={"slos": [], "overall_healthy": True, "available": True},
            ),
            patch.object(
                handler, "_collect_adapters",
                return_value={"adapters": [], "active": 10, "total": 10, "available": True},
            ),
            patch.object(
                handler, "_collect_agents",
                return_value={"agents": [], "total": 5, "active": 5, "available": True},
            ),
            patch.object(
                handler, "_collect_budget",
                return_value={
                    "total_budget": 100, "spent": 50, "utilization": 0.5,
                    "forecast": None, "available": True,
                },
            ),
        ):
            result = handler._get_overview()

        data = _get_data(result)
        assert data["overall_status"] == "healthy"
        assert data["subsystems"]["circuit_breakers"] == "healthy"
        assert data["subsystems"]["slos"] == "healthy"
        assert data["subsystems"]["adapters"] == "healthy"
        assert data["subsystems"]["agents"] == "healthy"
        assert data["subsystems"]["budget"] == "healthy"
        assert "last_check" in data
        assert "collection_time_ms" in data

    def test_overview_critical_from_open_circuit_breaker(self, handler):
        with (
            patch.object(
                handler, "_collect_circuit_breakers",
                return_value={
                    "breakers": [{"state": "open"}],
                    "total": 1, "available": True,
                },
            ),
            patch.object(
                handler, "_collect_slos",
                return_value={"slos": [], "overall_healthy": True, "available": True},
            ),
            patch.object(
                handler, "_collect_adapters",
                return_value={"adapters": [], "active": 5, "total": 5, "available": True},
            ),
            patch.object(
                handler, "_collect_agents",
                return_value={"agents": [], "total": 5, "active": 5, "available": True},
            ),
            patch.object(
                handler, "_collect_budget",
                return_value={
                    "total_budget": 100, "spent": 50, "utilization": 0.5,
                    "forecast": None, "available": True,
                },
            ),
        ):
            result = handler._get_overview()

        data = _get_data(result)
        assert data["overall_status"] == "critical"
        assert data["subsystems"]["circuit_breakers"] == "critical"

    def test_overview_degraded_from_half_open_cb(self, handler):
        with (
            patch.object(
                handler, "_collect_circuit_breakers",
                return_value={
                    "breakers": [{"state": "half-open"}],
                    "total": 1, "available": True,
                },
            ),
            patch.object(
                handler, "_collect_slos",
                return_value={"slos": [], "overall_healthy": True, "available": True},
            ),
            patch.object(
                handler, "_collect_adapters",
                return_value={"adapters": [], "active": 5, "total": 5, "available": True},
            ),
            patch.object(
                handler, "_collect_agents",
                return_value={"agents": [], "total": 5, "active": 5, "available": True},
            ),
            patch.object(
                handler, "_collect_budget",
                return_value={
                    "total_budget": 100, "spent": 50, "utilization": 0.5,
                    "forecast": None, "available": True,
                },
            ),
        ):
            result = handler._get_overview()

        data = _get_data(result)
        assert data["overall_status"] == "degraded"

    def test_overview_critical_from_slos(self, handler):
        """Multiple non-compliant SLOs => critical."""
        with (
            patch.object(
                handler, "_collect_circuit_breakers",
                return_value={"breakers": [], "total": 0, "available": True},
            ),
            patch.object(
                handler, "_collect_slos",
                return_value={
                    "slos": [
                        {"compliant": False},
                        {"compliant": False},
                    ],
                    "overall_healthy": False,
                    "available": True,
                },
            ),
            patch.object(
                handler, "_collect_adapters",
                return_value={"adapters": [], "active": 5, "total": 5, "available": True},
            ),
            patch.object(
                handler, "_collect_agents",
                return_value={"agents": [], "total": 5, "active": 5, "available": True},
            ),
            patch.object(
                handler, "_collect_budget",
                return_value={
                    "total_budget": 100, "spent": 50, "utilization": 0.5,
                    "forecast": None, "available": True,
                },
            ),
        ):
            result = handler._get_overview()

        data = _get_data(result)
        assert data["subsystems"]["slos"] == "critical"
        assert data["overall_status"] == "critical"

    def test_overview_degraded_from_single_slo(self, handler):
        """One non-compliant SLO => degraded."""
        with (
            patch.object(
                handler, "_collect_circuit_breakers",
                return_value={"breakers": [], "total": 0, "available": True},
            ),
            patch.object(
                handler, "_collect_slos",
                return_value={
                    "slos": [{"compliant": False}, {"compliant": True}],
                    "overall_healthy": False,
                    "available": True,
                },
            ),
            patch.object(
                handler, "_collect_adapters",
                return_value={"adapters": [], "active": 5, "total": 5, "available": True},
            ),
            patch.object(
                handler, "_collect_agents",
                return_value={"agents": [], "total": 5, "active": 5, "available": True},
            ),
            patch.object(
                handler, "_collect_budget",
                return_value={
                    "total_budget": 100, "spent": 50, "utilization": 0.5,
                    "forecast": None, "available": True,
                },
            ),
        ):
            result = handler._get_overview()

        data = _get_data(result)
        assert data["subsystems"]["slos"] == "degraded"

    def test_overview_degraded_from_adapters(self, handler):
        """Less than 50% active adapters => degraded."""
        with (
            patch.object(
                handler, "_collect_circuit_breakers",
                return_value={"breakers": [], "total": 0, "available": True},
            ),
            patch.object(
                handler, "_collect_slos",
                return_value={"slos": [], "overall_healthy": True, "available": True},
            ),
            patch.object(
                handler, "_collect_adapters",
                return_value={"adapters": [], "active": 2, "total": 10, "available": True},
            ),
            patch.object(
                handler, "_collect_agents",
                return_value={"agents": [], "total": 5, "active": 5, "available": True},
            ),
            patch.object(
                handler, "_collect_budget",
                return_value={
                    "total_budget": 100, "spent": 50, "utilization": 0.5,
                    "forecast": None, "available": True,
                },
            ),
        ):
            result = handler._get_overview()

        data = _get_data(result)
        assert data["subsystems"]["adapters"] == "degraded"

    def test_overview_critical_from_agents(self, handler):
        """More than 30% failed agents => critical."""
        with (
            patch.object(
                handler, "_collect_circuit_breakers",
                return_value={"breakers": [], "total": 0, "available": True},
            ),
            patch.object(
                handler, "_collect_slos",
                return_value={"slos": [], "overall_healthy": True, "available": True},
            ),
            patch.object(
                handler, "_collect_adapters",
                return_value={"adapters": [], "active": 5, "total": 5, "available": True},
            ),
            patch.object(
                handler, "_collect_agents",
                return_value={
                    "agents": [
                        {"status": "failed"},
                        {"status": "failed"},
                        {"status": "active"},
                    ],
                    "total": 3,
                    "active": 1,
                    "available": True,
                },
            ),
            patch.object(
                handler, "_collect_budget",
                return_value={
                    "total_budget": 100, "spent": 50, "utilization": 0.5,
                    "forecast": None, "available": True,
                },
            ),
        ):
            result = handler._get_overview()

        data = _get_data(result)
        # 2/3 > 0.3 => critical
        assert data["subsystems"]["agents"] == "critical"

    def test_overview_degraded_from_agents(self, handler):
        """Some failed agents but <= 30% => degraded."""
        with (
            patch.object(
                handler, "_collect_circuit_breakers",
                return_value={"breakers": [], "total": 0, "available": True},
            ),
            patch.object(
                handler, "_collect_slos",
                return_value={"slos": [], "overall_healthy": True, "available": True},
            ),
            patch.object(
                handler, "_collect_adapters",
                return_value={"adapters": [], "active": 5, "total": 5, "available": True},
            ),
            patch.object(
                handler, "_collect_agents",
                return_value={
                    "agents": [
                        {"status": "failed"},
                        {"status": "active"},
                        {"status": "active"},
                        {"status": "active"},
                    ],
                    "total": 4,
                    "active": 3,
                    "available": True,
                },
            ),
            patch.object(
                handler, "_collect_budget",
                return_value={
                    "total_budget": 100, "spent": 50, "utilization": 0.5,
                    "forecast": None, "available": True,
                },
            ),
        ):
            result = handler._get_overview()

        data = _get_data(result)
        # 1/4 = 0.25 <= 0.3, but > 0 => degraded
        assert data["subsystems"]["agents"] == "degraded"

    def test_overview_critical_from_budget(self, handler):
        """Utilization > 0.95 => critical."""
        with (
            patch.object(
                handler, "_collect_circuit_breakers",
                return_value={"breakers": [], "total": 0, "available": True},
            ),
            patch.object(
                handler, "_collect_slos",
                return_value={"slos": [], "overall_healthy": True, "available": True},
            ),
            patch.object(
                handler, "_collect_adapters",
                return_value={"adapters": [], "active": 5, "total": 5, "available": True},
            ),
            patch.object(
                handler, "_collect_agents",
                return_value={"agents": [], "total": 5, "active": 5, "available": True},
            ),
            patch.object(
                handler, "_collect_budget",
                return_value={
                    "total_budget": 100, "spent": 98, "utilization": 0.98,
                    "forecast": None, "available": True,
                },
            ),
        ):
            result = handler._get_overview()

        data = _get_data(result)
        assert data["subsystems"]["budget"] == "critical"

    def test_overview_degraded_from_budget(self, handler):
        """Utilization > 0.8 but <= 0.95 => degraded."""
        with (
            patch.object(
                handler, "_collect_circuit_breakers",
                return_value={"breakers": [], "total": 0, "available": True},
            ),
            patch.object(
                handler, "_collect_slos",
                return_value={"slos": [], "overall_healthy": True, "available": True},
            ),
            patch.object(
                handler, "_collect_adapters",
                return_value={"adapters": [], "active": 5, "total": 5, "available": True},
            ),
            patch.object(
                handler, "_collect_agents",
                return_value={"agents": [], "total": 5, "active": 5, "available": True},
            ),
            patch.object(
                handler, "_collect_budget",
                return_value={
                    "total_budget": 100, "spent": 85, "utilization": 0.85,
                    "forecast": None, "available": True,
                },
            ),
        ):
            result = handler._get_overview()

        data = _get_data(result)
        assert data["subsystems"]["budget"] == "degraded"

    def test_overview_unknown_when_subsystem_unavailable(self, handler):
        """Subsystems that are unavailable => 'unknown'."""
        with (
            patch.object(
                handler, "_collect_circuit_breakers",
                return_value={"breakers": [], "available": False},
            ),
            patch.object(
                handler, "_collect_slos",
                return_value={"slos": [], "overall_healthy": True, "available": False},
            ),
            patch.object(
                handler, "_collect_adapters",
                return_value={"adapters": [], "active": 0, "total": 0, "available": False},
            ),
            patch.object(
                handler, "_collect_agents",
                return_value={"agents": [], "total": 0, "active": 0, "available": False},
            ),
            patch.object(
                handler, "_collect_budget",
                return_value={
                    "total_budget": 0, "spent": 0, "utilization": 0,
                    "forecast": None, "available": False,
                },
            ),
        ):
            result = handler._get_overview()

        data = _get_data(result)
        for key in ["circuit_breakers", "slos", "adapters", "agents", "budget"]:
            assert data["subsystems"][key] == "unknown"
        # All unknown => healthy (no critical/degraded)
        assert data["overall_status"] == "healthy"

    def test_overview_collection_time_present(self, handler):
        with (
            patch.object(handler, "_collect_circuit_breakers", return_value={"breakers": [], "available": False}),
            patch.object(handler, "_collect_slos", return_value={"slos": [], "overall_healthy": True, "available": False}),
            patch.object(handler, "_collect_adapters", return_value={"adapters": [], "active": 0, "total": 0, "available": False}),
            patch.object(handler, "_collect_agents", return_value={"agents": [], "total": 0, "active": 0, "available": False}),
            patch.object(handler, "_collect_budget", return_value={"total_budget": 0, "spent": 0, "utilization": 0, "forecast": None, "available": False}),
        ):
            result = handler._get_overview()

        data = _get_data(result)
        assert isinstance(data["collection_time_ms"], float)
        assert data["collection_time_ms"] >= 0


# ---------------------------------------------------------------------------
# JSON response wrappers (_get_* methods)
# ---------------------------------------------------------------------------


class TestJsonWrappers:
    """Test that _get_* wrappers return proper data envelopes."""

    @pytest.mark.asyncio
    async def test_get_circuit_breakers_wraps_data(self, handler):
        with patch(
            "aragora.resilience.registry.get_circuit_breakers",
            return_value={},
        ):
            result = handler._get_circuit_breakers()

        body = _parse_response(result)
        assert "data" in body

    @pytest.mark.asyncio
    async def test_get_slos_wraps_data(self, handler):
        with patch(
            "aragora.observability.slo.get_slo_status",
            return_value=MockSloStatus(),
        ):
            result = handler._get_slos()

        body = _parse_response(result)
        assert "data" in body

    @pytest.mark.asyncio
    async def test_get_adapters_wraps_data(self, handler):
        with patch(
            "aragora.knowledge.mound.adapters.factory.ADAPTER_SPECS",
            {},
        ):
            result = handler._get_adapters()

        body = _parse_response(result)
        assert "data" in body

    @pytest.mark.asyncio
    async def test_get_agents_wraps_data(self, handler):
        with patch(
            "aragora.control_plane.registry.get_default_registry",
            create=True,
            return_value=None,
        ):
            result = handler._get_agents()

        body = _parse_response(result)
        assert "data" in body

    @pytest.mark.asyncio
    async def test_get_budget_wraps_data(self, handler):
        with patch(
            "builtins.__import__",
            side_effect=_selective_import_error("aragora.billing.cost_tracker"),
        ):
            result = handler._get_budget()

        body = _parse_response(result)
        assert "data" in body


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _selective_import_error(blocked_module: str):
    """Create a side_effect for builtins.__import__ that blocks a specific module."""
    real_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

    def _import(name, *args, **kwargs):
        if name == blocked_module:
            raise ImportError(f"Mocked import error for {name}")
        return real_import(name, *args, **kwargs)

    return _import
