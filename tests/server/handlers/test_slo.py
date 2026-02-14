"""
Tests for aragora.server.handlers.slo - SLO HTTP Handlers.

Tests cover:
- SLOHandler: instantiation, ROUTES, can_handle
- GET /api/slos/status: success, exception
- GET /api/slos/{slo_name}: availability, latency, debate-success, unknown
- GET /api/slos/error-budget: success
- GET /api/slos/violations: success
- GET /api/slos/targets: success
- GET /api/slos/{name}/{sub_route}: error-budget, violations, compliant, alerts
- handle routing: rate limiting, returns None for unmatched paths
- Path normalization: /api/slo/ -> /api/slos/
- Version prefix stripping
- _calculate_exhaustion_time: sustainable, exhausting, zero budget
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.slo import SLOHandler
from aragora.server.handlers.utils.responses import HandlerResult


# ===========================================================================
# Helpers
# ===========================================================================


def _parse_body(result: HandlerResult) -> dict[str, Any]:
    """Parse JSON body from HandlerResult."""
    return json.loads(result.body)


def _make_mock_handler(
    method: str = "GET",
    body: bytes = b"",
    content_type: str = "application/json",
) -> MagicMock:
    """Create a mock HTTP handler object."""
    handler = MagicMock()
    handler.command = method
    handler.client_address = ("127.0.0.1", 12345)
    handler.headers = {
        "Content-Length": str(len(body)),
        "Content-Type": content_type,
        "Host": "localhost:8080",
    }
    handler.rfile = MagicMock()
    handler.rfile.read.return_value = body
    return handler


# ===========================================================================
# Mock SLO Objects
# ===========================================================================


class MockSLOResult:
    """Mock SLO result."""

    def __init__(
        self,
        name: str = "Availability",
        target: float = 99.9,
        current: float = 99.95,
        compliant: bool = True,
    ):
        self.name = name
        self.target = target
        self.current = current
        self.compliant = compliant
        self.compliance_percentage = 100.0 if compliant else 95.0
        self.error_budget_remaining = 80.0
        self.burn_rate = 0.5
        self.window_start = datetime(2026, 2, 14, 0, 0, tzinfo=timezone.utc)
        self.window_end = datetime(2026, 2, 14, 1, 0, tzinfo=timezone.utc)


class MockSLOStatus:
    """Mock SLO overall status."""

    def __init__(self):
        self.availability = MockSLOResult("Availability", 99.9, 99.95, True)
        self.latency_p99 = MockSLOResult("Latency P99", 500.0, 320.0, True)
        self.debate_success = MockSLOResult("Debate Success", 95.0, 96.5, True)
        self.overall_healthy = True
        self.timestamp = datetime(2026, 2, 14, 12, 0, tzinfo=timezone.utc)


class MockAlert:
    """Mock SLO alert."""

    def __init__(self, slo_name: str = "Availability"):
        self.slo_name = slo_name
        self.severity = "warning"
        self.message = "Error budget burning faster than expected"


class MockSLOTarget:
    """Mock SLO target."""

    def __init__(
        self,
        name: str = "Availability",
        target: float = 99.9,
        unit: str = "%",
        description: str = "Service availability",
        comparison: str = ">=",
    ):
        self.name = name
        self.target = target
        self.unit = unit
        self.description = description
        self.comparison = comparison


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    """Reset the rate limiter between tests."""
    from aragora.server.handlers.slo import _slo_limiter

    _slo_limiter._buckets.clear()


@pytest.fixture
def handler():
    """Create a SLOHandler."""
    return SLOHandler(ctx={})


@pytest.fixture
def mock_slo_status():
    """Create a mock SLO status."""
    return MockSLOStatus()


# ===========================================================================
# Test Instantiation and Basics
# ===========================================================================


class TestSLOHandlerBasics:
    """Basic instantiation and attribute tests."""

    def test_instantiation(self, handler):
        assert handler is not None
        assert isinstance(handler, SLOHandler)

    def test_routes(self, handler):
        assert "/api/slos/status" in handler.ROUTES
        assert "/api/slos/error-budget" in handler.ROUTES
        assert "/api/slos/violations" in handler.ROUTES
        assert "/api/slos/targets" in handler.ROUTES

    def test_can_handle_status(self, handler):
        assert handler.can_handle("/api/slos/status") is True

    def test_can_handle_error_budget(self, handler):
        assert handler.can_handle("/api/slos/error-budget") is True

    def test_can_handle_violations(self, handler):
        assert handler.can_handle("/api/slos/violations") is True

    def test_can_handle_targets(self, handler):
        assert handler.can_handle("/api/slos/targets") is True

    def test_can_handle_availability(self, handler):
        assert handler.can_handle("/api/slos/availability") is True

    def test_can_handle_latency(self, handler):
        assert handler.can_handle("/api/slos/latency") is True

    def test_can_handle_debate_success(self, handler):
        assert handler.can_handle("/api/slos/debate-success") is True

    def test_can_handle_singular_slo_path(self, handler):
        """Test /api/slo/ normalization to /api/slos/."""
        assert handler.can_handle("/api/slo/status") is True

    def test_can_handle_versioned_path(self, handler):
        assert handler.can_handle("/api/v1/slos/status") is True

    def test_can_handle_v2_slo_path(self, handler):
        assert handler.can_handle("/api/v2/slo/status") is True

    def test_can_handle_sub_route(self, handler):
        assert handler.can_handle("/api/slos/availability/error-budget") is True

    def test_can_handle_sub_route_alerts(self, handler):
        assert handler.can_handle("/api/slos/availability/alerts") is True

    def test_cannot_handle_other_path(self, handler):
        assert handler.can_handle("/api/debates") is False

    def test_cannot_handle_unknown_slo(self, handler):
        assert handler.can_handle("/api/slos/unknown-metric") is False

    def test_cannot_handle_unknown_sub_route(self, handler):
        assert handler.can_handle("/api/slos/availability/unknown") is False

    def test_default_context(self):
        h = SLOHandler()
        assert h.ctx == {}


# ===========================================================================
# Test GET /api/slos/status
# ===========================================================================


class TestGetStatus:
    """Tests for the SLO status endpoint."""

    def test_get_status_success(self, handler):
        with patch(
            "aragora.server.handlers.slo.get_slo_status_json",
            return_value={"healthy": True, "slos": []},
        ):
            result = handler._handle_slo_status()
            assert result.status_code == 200
            data = _parse_body(result)
            assert data["healthy"] is True

    def test_get_status_exception(self, handler):
        with patch(
            "aragora.server.handlers.slo.get_slo_status_json",
            side_effect=RuntimeError("DB down"),
        ):
            result = handler._handle_slo_status()
            assert result.status_code == 500


# ===========================================================================
# Test GET /api/slos/{slo_name}
# ===========================================================================


class TestGetSLODetail:
    """Tests for individual SLO detail endpoint."""

    def test_get_availability(self, handler):
        mock_result = MockSLOResult("Availability", 99.9, 99.95, True)
        with patch(
            "aragora.server.handlers.slo.check_availability_slo",
            return_value=mock_result,
        ):
            result = handler._handle_slo_detail("availability")
            assert result.status_code == 200
            data = _parse_body(result)
            assert data["name"] == "Availability"
            assert data["compliant"] is True
            assert "window" in data

    def test_get_latency(self, handler):
        mock_result = MockSLOResult("Latency P99", 500.0, 320.0, True)
        with patch(
            "aragora.server.handlers.slo.check_latency_slo",
            return_value=mock_result,
        ):
            result = handler._handle_slo_detail("latency_p99")
            assert result.status_code == 200
            data = _parse_body(result)
            assert data["name"] == "Latency P99"

    def test_get_debate_success(self, handler):
        mock_result = MockSLOResult("Debate Success", 95.0, 96.5, True)
        with patch(
            "aragora.server.handlers.slo.check_debate_success_slo",
            return_value=mock_result,
        ):
            result = handler._handle_slo_detail("debate_success")
            assert result.status_code == 200
            data = _parse_body(result)
            assert data["name"] == "Debate Success"

    def test_get_unknown_slo(self, handler):
        result = handler._handle_slo_detail("nonexistent")
        assert result.status_code == 404

    def test_get_detail_exception(self, handler):
        with patch(
            "aragora.server.handlers.slo.check_availability_slo",
            side_effect=RuntimeError("Service error"),
        ):
            result = handler._handle_slo_detail("availability")
            assert result.status_code == 500


# ===========================================================================
# Test GET /api/slos/error-budget
# ===========================================================================


class TestGetErrorBudget:
    """Tests for the error budget endpoint."""

    def test_error_budget_success(self, handler, mock_slo_status):
        with patch(
            "aragora.server.handlers.slo.get_slo_status",
            return_value=mock_slo_status,
        ):
            result = handler._handle_error_budget()
            assert result.status_code == 200
            data = _parse_body(result)
            assert "budgets" in data
            assert len(data["budgets"]) == 3
            budget = data["budgets"][0]
            assert "error_budget_remaining" in budget
            assert "burn_rate" in budget
            assert "error_budget_consumed" in budget

    def test_error_budget_exception(self, handler):
        with patch(
            "aragora.server.handlers.slo.get_slo_status",
            side_effect=RuntimeError("Failed"),
        ):
            result = handler._handle_error_budget()
            assert result.status_code == 500


# ===========================================================================
# Test GET /api/slos/violations
# ===========================================================================


class TestGetViolations:
    """Tests for the violations endpoint."""

    def test_violations_no_alerts(self, handler, mock_slo_status):
        with patch(
            "aragora.server.handlers.slo.get_slo_status",
            return_value=mock_slo_status,
        ):
            with patch(
                "aragora.server.handlers.slo.check_alerts",
                return_value=[],
            ):
                result = handler._handle_violations()
                assert result.status_code == 200
                data = _parse_body(result)
                assert data["violation_count"] == 0
                assert data["overall_healthy"] is True

    def test_violations_with_alerts(self, handler, mock_slo_status):
        alert = MockAlert("Availability")
        slo_result = mock_slo_status.availability

        with patch(
            "aragora.server.handlers.slo.get_slo_status",
            return_value=mock_slo_status,
        ):
            with patch(
                "aragora.server.handlers.slo.check_alerts",
                return_value=[(alert, slo_result)],
            ):
                result = handler._handle_violations()
                assert result.status_code == 200
                data = _parse_body(result)
                assert data["violation_count"] == 1
                violation = data["violations"][0]
                assert violation["slo_name"] == "Availability"
                assert violation["severity"] == "warning"

    def test_violations_exception(self, handler):
        with patch(
            "aragora.server.handlers.slo.get_slo_status",
            side_effect=RuntimeError("Failed"),
        ):
            result = handler._handle_violations()
            assert result.status_code == 500


# ===========================================================================
# Test GET /api/slos/targets
# ===========================================================================


class TestGetTargets:
    """Tests for the targets endpoint."""

    def test_targets_success(self, handler):
        mock_targets = {
            "availability": MockSLOTarget("Availability", 99.9, "%", "Uptime", ">="),
            "latency_p99": MockSLOTarget("Latency P99", 500, "ms", "Response time", "<="),
        }
        with patch(
            "aragora.server.handlers.slo.get_slo_targets",
            return_value=mock_targets,
        ):
            result = handler._handle_targets()
            assert result.status_code == 200
            data = _parse_body(result)
            assert "targets" in data
            assert len(data["targets"]) == 2
            target = data["targets"][0]
            assert "name" in target
            assert "target" in target
            assert "unit" in target

    def test_targets_exception(self, handler):
        with patch(
            "aragora.server.handlers.slo.get_slo_targets",
            side_effect=RuntimeError("Failed"),
        ):
            result = handler._handle_targets()
            assert result.status_code == 500


# ===========================================================================
# Test GET /api/slos/{name}/{sub_route}
# ===========================================================================


class TestSLOSubRoutes:
    """Tests for per-SLO sub-routes."""

    def test_sub_route_error_budget(self, handler):
        mock_result = MockSLOResult("Availability")
        with patch(
            "aragora.server.handlers.slo.check_availability_slo",
            return_value=mock_result,
        ):
            result = handler._handle_slo_sub_route("availability", "error-budget")
            assert result.status_code == 200
            data = _parse_body(result)
            assert data["slo_name"] == "Availability"
            assert "error_budget_remaining" in data

    def test_sub_route_compliant(self, handler):
        mock_result = MockSLOResult("Availability", compliant=True)
        with patch(
            "aragora.server.handlers.slo.check_availability_slo",
            return_value=mock_result,
        ):
            result = handler._handle_slo_sub_route("availability", "compliant")
            assert result.status_code == 200
            data = _parse_body(result)
            assert data["compliant"] is True

    def test_sub_route_violations(self, handler, mock_slo_status):
        mock_result = MockSLOResult("Availability")
        alert = MockAlert("Availability")

        with patch(
            "aragora.server.handlers.slo.check_availability_slo",
            return_value=mock_result,
        ):
            with patch(
                "aragora.server.handlers.slo.get_slo_status",
                return_value=mock_slo_status,
            ):
                with patch(
                    "aragora.server.handlers.slo.check_alerts",
                    return_value=[(alert, mock_result)],
                ):
                    result = handler._handle_slo_sub_route("availability", "violations")
                    assert result.status_code == 200
                    data = _parse_body(result)
                    assert data["slo_name"] == "Availability"
                    assert len(data["violations"]) == 1

    def test_sub_route_alerts(self, handler, mock_slo_status):
        mock_result = MockSLOResult("Availability")
        alert = MockAlert("Availability")

        with patch(
            "aragora.server.handlers.slo.check_availability_slo",
            return_value=mock_result,
        ):
            with patch(
                "aragora.server.handlers.slo.get_slo_status",
                return_value=mock_slo_status,
            ):
                with patch(
                    "aragora.server.handlers.slo.check_alerts",
                    return_value=[(alert, mock_result)],
                ):
                    result = handler._handle_slo_sub_route("availability", "alerts")
                    assert result.status_code == 200
                    data = _parse_body(result)
                    assert data["slo_name"] == "Availability"

    def test_sub_route_unknown_slo(self, handler):
        result = handler._handle_slo_sub_route("nonexistent", "error-budget")
        assert result.status_code == 404

    def test_sub_route_unknown_route(self, handler):
        mock_result = MockSLOResult("Availability")
        with patch(
            "aragora.server.handlers.slo.check_availability_slo",
            return_value=mock_result,
        ):
            result = handler._handle_slo_sub_route("availability", "unknown")
            assert result.status_code == 404

    def test_sub_route_exception(self, handler):
        with patch(
            "aragora.server.handlers.slo.check_availability_slo",
            side_effect=RuntimeError("Service error"),
        ):
            result = handler._handle_slo_sub_route("availability", "error-budget")
            assert result.status_code == 500


# ===========================================================================
# Test handle() Routing (GET)
# ===========================================================================


class TestHandleRouting:
    """Tests for the top-level handle() method routing."""

    def test_handle_status(self, handler):
        mock_handler = _make_mock_handler()
        with patch(
            "aragora.server.handlers.slo.get_slo_status_json",
            return_value={"healthy": True},
        ):
            result = handler.handle("/api/slos/status", {}, mock_handler)
            assert result is not None
            assert result.status_code == 200

    def test_handle_availability(self, handler):
        mock_handler = _make_mock_handler()
        mock_result = MockSLOResult("Availability")
        with patch(
            "aragora.server.handlers.slo.check_availability_slo",
            return_value=mock_result,
        ):
            result = handler.handle("/api/slos/availability", {}, mock_handler)
            assert result is not None
            assert result.status_code == 200

    def test_handle_latency(self, handler):
        mock_handler = _make_mock_handler()
        mock_result = MockSLOResult("Latency P99")
        with patch(
            "aragora.server.handlers.slo.check_latency_slo",
            return_value=mock_result,
        ):
            result = handler.handle("/api/slos/latency", {}, mock_handler)
            assert result is not None
            assert result.status_code == 200

    def test_handle_debate_success(self, handler):
        mock_handler = _make_mock_handler()
        mock_result = MockSLOResult("Debate Success")
        with patch(
            "aragora.server.handlers.slo.check_debate_success_slo",
            return_value=mock_result,
        ):
            result = handler.handle("/api/slos/debate-success", {}, mock_handler)
            assert result is not None
            assert result.status_code == 200

    def test_handle_error_budget(self, handler, mock_slo_status):
        mock_handler = _make_mock_handler()
        with patch(
            "aragora.server.handlers.slo.get_slo_status",
            return_value=mock_slo_status,
        ):
            result = handler.handle("/api/slos/error-budget", {}, mock_handler)
            assert result is not None
            assert result.status_code == 200

    def test_handle_violations(self, handler, mock_slo_status):
        mock_handler = _make_mock_handler()
        with patch(
            "aragora.server.handlers.slo.get_slo_status",
            return_value=mock_slo_status,
        ):
            with patch(
                "aragora.server.handlers.slo.check_alerts",
                return_value=[],
            ):
                result = handler.handle("/api/slos/violations", {}, mock_handler)
                assert result is not None
                assert result.status_code == 200

    def test_handle_targets(self, handler):
        mock_handler = _make_mock_handler()
        with patch(
            "aragora.server.handlers.slo.get_slo_targets",
            return_value={},
        ):
            result = handler.handle("/api/slos/targets", {}, mock_handler)
            assert result is not None
            assert result.status_code == 200

    def test_handle_singular_slo_path(self, handler):
        mock_handler = _make_mock_handler()
        with patch(
            "aragora.server.handlers.slo.get_slo_status_json",
            return_value={"healthy": True},
        ):
            result = handler.handle("/api/slo/status", {}, mock_handler)
            assert result is not None
            assert result.status_code == 200

    def test_handle_versioned_path(self, handler):
        mock_handler = _make_mock_handler()
        with patch(
            "aragora.server.handlers.slo.get_slo_status_json",
            return_value={"healthy": True},
        ):
            result = handler.handle("/api/v1/slos/status", {}, mock_handler)
            assert result is not None
            assert result.status_code == 200

    def test_handle_rate_limited(self, handler):
        from aragora.server.handlers.slo import _slo_limiter

        mock_handler = _make_mock_handler()
        with patch.object(_slo_limiter, "is_allowed", return_value=False):
            result = handler.handle("/api/slos/status", {}, mock_handler)
            assert result.status_code == 429

    def test_handle_sub_route(self, handler):
        mock_handler = _make_mock_handler()
        mock_result = MockSLOResult("Availability")
        with patch(
            "aragora.server.handlers.slo.check_availability_slo",
            return_value=mock_result,
        ):
            result = handler.handle(
                "/api/slos/availability/compliant", {}, mock_handler
            )
            assert result is not None
            assert result.status_code == 200

    def test_handle_exception(self, handler):
        mock_handler = _make_mock_handler()
        with patch(
            "aragora.server.handlers.slo.get_slo_status_json",
            side_effect=RuntimeError("Unexpected"),
        ):
            result = handler.handle("/api/slos/status", {}, mock_handler)
            assert result.status_code == 500


# ===========================================================================
# Test _calculate_exhaustion_time
# ===========================================================================


class TestCalculateExhaustionTime:
    """Tests for error budget exhaustion calculation."""

    def test_sustainable_burn_rate(self, handler):
        mock_result = MockSLOResult()
        mock_result.burn_rate = 0.5  # Sustainable
        assert handler._calculate_exhaustion_time(mock_result) is None

    def test_exactly_one_burn_rate(self, handler):
        mock_result = MockSLOResult()
        mock_result.burn_rate = 1.0  # Exact
        assert handler._calculate_exhaustion_time(mock_result) is None

    def test_high_burn_rate_returns_time(self, handler):
        mock_result = MockSLOResult()
        mock_result.burn_rate = 2.0
        mock_result.error_budget_remaining = 50.0
        result = handler._calculate_exhaustion_time(mock_result)
        assert result is not None
        # Should be an ISO format string
        assert "T" in result

    def test_zero_budget_remaining(self, handler):
        mock_result = MockSLOResult()
        mock_result.burn_rate = 2.0
        mock_result.error_budget_remaining = 0.0
        assert handler._calculate_exhaustion_time(mock_result) is None

    def test_zero_burn_rate(self, handler):
        mock_result = MockSLOResult()
        mock_result.burn_rate = 0.0
        assert handler._calculate_exhaustion_time(mock_result) is None


# ===========================================================================
# Test Path Normalization
# ===========================================================================


class TestPathNormalization:
    """Tests for path normalization logic."""

    def test_normalize_singular_to_plural(self):
        assert SLOHandler._normalize_slo_path("/api/slo/status") == "/api/slos/status"

    def test_normalize_already_plural(self):
        assert SLOHandler._normalize_slo_path("/api/slos/status") == "/api/slos/status"

    def test_normalize_strips_version(self):
        assert SLOHandler._normalize_slo_path("/api/v1/slos/status") == "/api/slos/status"

    def test_normalize_v2_singular(self):
        assert SLOHandler._normalize_slo_path("/api/v2/slo/status") == "/api/slos/status"
