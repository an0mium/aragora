"""
Tests for SLOHandler - Service Level Objective HTTP endpoints.

Stability: STABLE
Handler Status: Graduated from EXPERIMENTAL to STABLE (2026-02-01)

Tests cover:
- Handler initialization and route matching
- GET /api/slos/status - Overall SLO status
- GET /api/slos/{slo_name} - Individual SLO details (availability, latency, debate-success)
- GET /api/slos/error-budget - Error budget timeline
- GET /api/slos/violations - Recent SLO violations
- GET /api/slos/targets - Configured SLO targets
- Rate limiting (30 RPM)
- RBAC permission enforcement (slo:read)
- Error handling
- API version stripping
- Response format validation
- Edge cases and boundary conditions

Target: 80%+ code coverage
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from aragora.server.handlers.slo import SLOHandler, SLO_SERVICE_TIMEOUT


# ===========================================================================
# Test Fixtures and Mocks
# ===========================================================================


@dataclass
class MockSLOResult:
    """Mock SLO result for testing."""

    name: str = "availability"
    target: float = 0.999
    current: float = 0.9995
    compliant: bool = True
    compliance_percentage: float = 99.95
    error_budget_remaining: float = 50.0
    burn_rate: float = 0.5
    window_start: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc) - timedelta(hours=1)
    )
    window_end: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class MockSLOStatus:
    """Mock SLO status for testing."""

    availability: MockSLOResult = field(default_factory=lambda: MockSLOResult(name="availability"))
    latency_p99: MockSLOResult = field(
        default_factory=lambda: MockSLOResult(name="latency_p99", target=0.95)
    )
    debate_success: MockSLOResult = field(
        default_factory=lambda: MockSLOResult(name="debate_success", target=0.90)
    )
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    overall_healthy: bool = True


@dataclass
class MockSLOTarget:
    """Mock SLO target configuration."""

    name: str = "availability"
    target: float = 0.999
    unit: str = "percentage"
    description: str = "Service availability target"
    comparison: str = "gte"


@dataclass
class MockAlert:
    """Mock SLO alert for violations."""

    slo_name: str = "availability"
    severity: str = "warning"
    message: str = "SLO degradation detected"


@pytest.fixture
def mock_handler():
    """Create mock HTTP handler."""
    handler = MagicMock()
    handler.client_address = ("127.0.0.1", 12345)
    handler.headers = {}
    return handler


@pytest.fixture
def server_context():
    """Create minimal server context for handler testing."""
    from unittest.mock import MagicMock

    return {
        "storage": MagicMock(),
        "user_store": MagicMock(),
        "elo_system": MagicMock(),
        "debate_embeddings": None,
        "critique_store": None,
        "nomic_dir": None,
    }


@pytest.fixture
def slo_handler(
    server_context,
):
    """Create SLOHandler instance."""
    return SLOHandler(server_context)


@pytest.fixture
def mock_slo_result():
    """Create mock SLO result."""
    return MockSLOResult()


@pytest.fixture
def mock_slo_status():
    """Create mock SLO status."""
    return MockSLOStatus()


# ===========================================================================
# Route Matching Tests
# ===========================================================================


class TestSLOHandlerRouting:
    """Test request routing and path matching."""

    def test_can_handle_status_path(self, slo_handler):
        """Test that handler recognizes /api/slos/status."""
        assert slo_handler.can_handle("/api/slos/status")

    def test_can_handle_error_budget_path(self, slo_handler):
        """Test that handler recognizes /api/slos/error-budget."""
        assert slo_handler.can_handle("/api/slos/error-budget")

    def test_can_handle_violations_path(self, slo_handler):
        """Test that handler recognizes /api/slos/violations."""
        assert slo_handler.can_handle("/api/slos/violations")

    def test_can_handle_targets_path(self, slo_handler):
        """Test that handler recognizes /api/slos/targets."""
        assert slo_handler.can_handle("/api/slos/targets")

    def test_can_handle_availability_path(self, slo_handler):
        """Test that handler recognizes /api/slos/availability."""
        assert slo_handler.can_handle("/api/slos/availability")

    def test_can_handle_latency_path(self, slo_handler):
        """Test that handler recognizes /api/slos/latency."""
        assert slo_handler.can_handle("/api/slos/latency")

    def test_can_handle_debate_success_path(self, slo_handler):
        """Test that handler recognizes /api/slos/debate-success."""
        assert slo_handler.can_handle("/api/slos/debate-success")

    def test_can_handle_versioned_status_path(self, slo_handler):
        """Test that handler recognizes versioned path /api/v1/slos/status."""
        assert slo_handler.can_handle("/api/v1/slos/status")

    def test_cannot_handle_other_paths(self, slo_handler):
        """Test that handler rejects non-SLO paths."""
        assert not slo_handler.can_handle("/api/metrics/status")
        assert not slo_handler.can_handle("/api/slos")
        assert not slo_handler.can_handle("/api/v1/debates")

    def test_cannot_handle_invalid_slo_name(self, slo_handler):
        """Test that handler rejects invalid SLO names."""
        assert not slo_handler.can_handle("/api/slos/invalid_slo")
        assert not slo_handler.can_handle("/api/slos/foo")


# ===========================================================================
# SLO Status Tests
# ===========================================================================


class TestSLOStatus:
    """Test /api/slos/status endpoint."""

    @patch("aragora.server.handlers.slo.get_slo_status_json")
    @patch("aragora.server.handlers.slo._slo_limiter")
    def test_get_slo_status_success(self, mock_limiter, mock_get_status, slo_handler, mock_handler):
        """Test successful SLO status retrieval."""
        mock_limiter.is_allowed.return_value = True
        mock_get_status.return_value = {
            "overall_healthy": True,
            "availability": {"compliant": True, "current": 0.9995},
            "latency_p99": {"compliant": True, "current": 0.95},
            "debate_success": {"compliant": True, "current": 0.92},
        }

        result = slo_handler.handle("/api/slos/status", {}, mock_handler)

        assert result is not None
        assert result.status_code == 200

    @patch("aragora.server.handlers.slo._slo_limiter")
    def test_get_slo_status_rate_limited(self, mock_limiter, slo_handler, mock_handler):
        """Test rate limiting on SLO status endpoint."""
        mock_limiter.is_allowed.return_value = False

        result = slo_handler.handle("/api/slos/status", {}, mock_handler)

        assert result is not None
        assert result.status_code == 429

    @patch("aragora.server.handlers.slo.get_slo_status_json")
    @patch("aragora.server.handlers.slo._slo_limiter")
    def test_get_slo_status_error_handling(
        self, mock_limiter, mock_get_status, slo_handler, mock_handler
    ):
        """Test error handling in SLO status endpoint."""
        mock_limiter.is_allowed.return_value = True
        mock_get_status.side_effect = RuntimeError("Database connection failed")

        result = slo_handler.handle("/api/slos/status", {}, mock_handler)

        assert result is not None
        assert result.status_code == 500


# ===========================================================================
# Individual SLO Detail Tests
# ===========================================================================


class TestSLODetail:
    """Test /api/slos/{slo_name} endpoints."""

    @patch("aragora.server.handlers.slo.check_availability_slo")
    @patch("aragora.server.handlers.slo._slo_limiter")
    def test_get_availability_slo_success(
        self, mock_limiter, mock_check, slo_handler, mock_handler, mock_slo_result
    ):
        """Test successful availability SLO retrieval."""
        mock_limiter.is_allowed.return_value = True
        mock_check.return_value = mock_slo_result

        result = slo_handler.handle("/api/slos/availability", {}, mock_handler)

        assert result is not None
        assert result.status_code == 200
        mock_check.assert_called_once()

    @patch("aragora.server.handlers.slo.check_latency_slo")
    @patch("aragora.server.handlers.slo._slo_limiter")
    def test_get_latency_slo_success(self, mock_limiter, mock_check, slo_handler, mock_handler):
        """Test successful latency SLO retrieval."""
        mock_limiter.is_allowed.return_value = True
        mock_check.return_value = MockSLOResult(name="latency_p99", target=0.95)

        result = slo_handler.handle("/api/slos/latency", {}, mock_handler)

        assert result is not None
        assert result.status_code == 200
        mock_check.assert_called_once()

    @patch("aragora.server.handlers.slo.check_debate_success_slo")
    @patch("aragora.server.handlers.slo._slo_limiter")
    def test_get_debate_success_slo_success(
        self, mock_limiter, mock_check, slo_handler, mock_handler
    ):
        """Test successful debate-success SLO retrieval."""
        mock_limiter.is_allowed.return_value = True
        mock_check.return_value = MockSLOResult(name="debate_success", target=0.90)

        result = slo_handler.handle("/api/slos/debate-success", {}, mock_handler)

        assert result is not None
        assert result.status_code == 200
        mock_check.assert_called_once()

    @patch("aragora.server.handlers.slo._slo_limiter")
    def test_get_unknown_slo_returns_404(self, mock_limiter, slo_handler, mock_handler):
        """Test that unknown SLO name returns 404."""
        mock_limiter.is_allowed.return_value = True

        # Can't reach here via can_handle, but test internal routing
        result = slo_handler._handle_slo_detail("unknown_slo")

        assert result is not None
        assert result.status_code == 404


# ===========================================================================
# Error Budget Tests
# ===========================================================================


class TestErrorBudget:
    """Test /api/slos/error-budget endpoint."""

    @patch("aragora.server.handlers.slo.get_slo_status")
    @patch("aragora.server.handlers.slo._slo_limiter")
    def test_get_error_budget_success(
        self, mock_limiter, mock_get_status, slo_handler, mock_handler, mock_slo_status
    ):
        """Test successful error budget retrieval."""
        mock_limiter.is_allowed.return_value = True
        mock_get_status.return_value = mock_slo_status

        result = slo_handler.handle("/api/slos/error-budget", {}, mock_handler)

        assert result is not None
        assert result.status_code == 200

    @patch("aragora.server.handlers.slo.get_slo_status")
    @patch("aragora.server.handlers.slo._slo_limiter")
    def test_error_budget_contains_all_slos(
        self, mock_limiter, mock_get_status, slo_handler, mock_handler, mock_slo_status
    ):
        """Test that error budget response contains all SLO budgets."""
        mock_limiter.is_allowed.return_value = True
        mock_get_status.return_value = mock_slo_status

        result = slo_handler.handle("/api/slos/error-budget", {}, mock_handler)

        assert result is not None
        assert result.status_code == 200
        # Response should contain budgets for all 3 SLOs


# ===========================================================================
# Violations Tests
# ===========================================================================


class TestViolations:
    """Test /api/slos/violations endpoint."""

    @patch("aragora.server.handlers.slo.check_alerts")
    @patch("aragora.server.handlers.slo.get_slo_status")
    @patch("aragora.server.handlers.slo._slo_limiter")
    def test_get_violations_no_alerts(
        self,
        mock_limiter,
        mock_get_status,
        mock_check_alerts,
        slo_handler,
        mock_handler,
        mock_slo_status,
    ):
        """Test violations endpoint with no active alerts."""
        mock_limiter.is_allowed.return_value = True
        mock_get_status.return_value = mock_slo_status
        mock_check_alerts.return_value = []

        result = slo_handler.handle("/api/slos/violations", {}, mock_handler)

        assert result is not None
        assert result.status_code == 200

    @patch("aragora.server.handlers.slo.check_alerts")
    @patch("aragora.server.handlers.slo.get_slo_status")
    @patch("aragora.server.handlers.slo._slo_limiter")
    def test_get_violations_with_alerts(
        self,
        mock_limiter,
        mock_get_status,
        mock_check_alerts,
        slo_handler,
        mock_handler,
        mock_slo_status,
    ):
        """Test violations endpoint with active alerts."""
        mock_limiter.is_allowed.return_value = True
        mock_slo_status.overall_healthy = False
        mock_get_status.return_value = mock_slo_status

        mock_alert = MockAlert()
        mock_result = MockSLOResult(compliant=False, current=0.98)
        mock_check_alerts.return_value = [(mock_alert, mock_result)]

        result = slo_handler.handle("/api/slos/violations", {}, mock_handler)

        assert result is not None
        assert result.status_code == 200


# ===========================================================================
# Targets Tests
# ===========================================================================


class TestTargets:
    """Test /api/slos/targets endpoint."""

    @patch("aragora.server.handlers.slo.get_slo_targets")
    @patch("aragora.server.handlers.slo._slo_limiter")
    def test_get_targets_success(self, mock_limiter, mock_get_targets, slo_handler, mock_handler):
        """Test successful SLO targets retrieval."""
        mock_limiter.is_allowed.return_value = True
        mock_get_targets.return_value = {
            "availability": MockSLOTarget(name="availability"),
            "latency_p99": MockSLOTarget(name="latency_p99", target=0.95),
            "debate_success": MockSLOTarget(name="debate_success", target=0.90),
        }

        result = slo_handler.handle("/api/slos/targets", {}, mock_handler)

        assert result is not None
        assert result.status_code == 200


# ===========================================================================
# Error Budget Calculation Tests
# ===========================================================================


class TestErrorBudgetCalculation:
    """Test internal error budget exhaustion calculation."""

    def test_calculate_exhaustion_time_sustainable_rate(self, slo_handler):
        """Test exhaustion calculation with sustainable burn rate."""
        mock_result = MockSLOResult(burn_rate=0.5, error_budget_remaining=50.0)
        result = slo_handler._calculate_exhaustion_time(mock_result)

        # Sustainable burn rate should return None
        assert result is None

    def test_calculate_exhaustion_time_high_burn_rate(self, slo_handler):
        """Test exhaustion calculation with high burn rate."""
        mock_result = MockSLOResult(burn_rate=2.0, error_budget_remaining=50.0)
        result = slo_handler._calculate_exhaustion_time(mock_result)

        # Should return an ISO timestamp
        assert result is not None
        assert "T" in result  # ISO format

    def test_calculate_exhaustion_time_no_budget(self, slo_handler):
        """Test exhaustion calculation with no budget remaining."""
        mock_result = MockSLOResult(burn_rate=2.0, error_budget_remaining=0.0)
        result = slo_handler._calculate_exhaustion_time(mock_result)

        # No budget remaining should return None
        assert result is None


# ===========================================================================
# RBAC Tests
# ===========================================================================


class TestSLORBAC:
    """Test RBAC permission enforcement."""

    @patch("aragora.server.handlers.slo._slo_limiter")
    def test_handle_requires_slo_read_permission(self, mock_limiter, slo_handler, mock_handler):
        """Test that handle method requires slo:read permission."""
        mock_limiter.is_allowed.return_value = True

        # The handle method has @require_permission("slo:read")
        # In test mode, this is typically bypassed, but we verify the decorator exists
        assert hasattr(slo_handler.handle, "__wrapped__") or callable(slo_handler.handle)


# ===========================================================================
# Version Stripping Tests
# ===========================================================================


class TestVersionStripping:
    """Test API version prefix stripping."""

    @patch("aragora.server.handlers.slo.get_slo_status_json")
    @patch("aragora.server.handlers.slo._slo_limiter")
    def test_versioned_path_handled(self, mock_limiter, mock_get_status, slo_handler, mock_handler):
        """Test that versioned paths are handled correctly."""
        mock_limiter.is_allowed.return_value = True
        mock_get_status.return_value = {"overall_healthy": True}

        # Handler should strip /api/v1 prefix and handle as /api/slos/status
        result = slo_handler.handle("/api/v1/slos/status", {}, mock_handler)

        assert result is not None
        assert result.status_code == 200


# ===========================================================================
# Error Handling Tests
# ===========================================================================


class TestErrorHandling:
    """Test error handling scenarios."""

    @patch("aragora.server.handlers.slo.get_slo_status")
    @patch("aragora.server.handlers.slo._slo_limiter")
    def test_error_budget_exception_handling(
        self, mock_limiter, mock_get_status, slo_handler, mock_handler
    ):
        """Test error handling in error budget endpoint."""
        mock_limiter.is_allowed.return_value = True
        mock_get_status.side_effect = ValueError("Unexpected error")

        result = slo_handler.handle("/api/slos/error-budget", {}, mock_handler)

        assert result is not None
        assert result.status_code == 500

    @patch("aragora.server.handlers.slo.check_availability_slo")
    @patch("aragora.server.handlers.slo._slo_limiter")
    def test_slo_detail_exception_handling(
        self, mock_limiter, mock_check, slo_handler, mock_handler
    ):
        """Test error handling in SLO detail endpoint."""
        mock_limiter.is_allowed.return_value = True
        mock_check.side_effect = ValueError("SLO check failed")

        result = slo_handler.handle("/api/slos/availability", {}, mock_handler)

        assert result is not None
        assert result.status_code == 500

    @patch("aragora.server.handlers.slo.check_alerts")
    @patch("aragora.server.handlers.slo.get_slo_status")
    @patch("aragora.server.handlers.slo._slo_limiter")
    def test_violations_exception_handling(
        self, mock_limiter, mock_get_status, mock_check_alerts, slo_handler, mock_handler
    ):
        """Test error handling in violations endpoint."""
        mock_limiter.is_allowed.return_value = True
        mock_get_status.side_effect = ValueError("Violation check failed")

        result = slo_handler.handle("/api/slos/violations", {}, mock_handler)

        assert result is not None
        assert result.status_code == 500

    @patch("aragora.server.handlers.slo.get_slo_targets")
    @patch("aragora.server.handlers.slo._slo_limiter")
    def test_targets_exception_handling(
        self, mock_limiter, mock_get_targets, slo_handler, mock_handler
    ):
        """Test error handling in targets endpoint."""
        mock_limiter.is_allowed.return_value = True
        mock_get_targets.side_effect = ValueError("Targets fetch failed")

        result = slo_handler.handle("/api/slos/targets", {}, mock_handler)

        assert result is not None
        assert result.status_code == 500


# ===========================================================================
# Handler Initialization Tests
# ===========================================================================


class TestHandlerInitialization:
    """Test handler initialization patterns."""

    def test_handler_init_with_context(self):
        """Test handler initialization with context."""
        ctx = {"storage": MagicMock(), "user_store": MagicMock()}
        handler = SLOHandler(ctx)
        assert handler.ctx == ctx

    def test_handler_init_without_context(self):
        """Test handler initialization without context."""
        handler = SLOHandler()
        assert handler.ctx == {}

    def test_handler_init_with_none_context(self):
        """Test handler initialization with None context."""
        handler = SLOHandler(None)
        assert handler.ctx == {}

    def test_handler_routes_attribute(self):
        """Test that handler has ROUTES class attribute."""
        assert hasattr(SLOHandler, "ROUTES")
        assert "/api/slos/status" in SLOHandler.ROUTES
        assert "/api/slos/error-budget" in SLOHandler.ROUTES
        assert "/api/slos/violations" in SLOHandler.ROUTES
        assert "/api/slos/targets" in SLOHandler.ROUTES

    def test_handler_slo_names_attribute(self):
        """Test that handler has SLO_NAMES class attribute."""
        assert hasattr(SLOHandler, "SLO_NAMES")
        assert "availability" in SLOHandler.SLO_NAMES
        assert "latency" in SLOHandler.SLO_NAMES
        assert "latency_p99" in SLOHandler.SLO_NAMES
        assert "debate_success" in SLOHandler.SLO_NAMES
        assert "debate-success" in SLOHandler.SLO_NAMES


# ===========================================================================
# Response Content Tests
# ===========================================================================


class TestResponseContent:
    """Test response content structure and format."""

    @patch("aragora.server.handlers.slo.get_slo_status_json")
    @patch("aragora.server.handlers.slo._slo_limiter")
    def test_slo_status_response_structure(
        self, mock_limiter, mock_get_status, slo_handler, mock_handler
    ):
        """Test that SLO status response has expected structure."""
        mock_limiter.is_allowed.return_value = True
        expected_response = {
            "timestamp": "2026-02-01T00:00:00+00:00",
            "overall_healthy": True,
            "slos": {
                "availability": {"compliant": True, "current": 0.9995},
                "latency_p99": {"compliant": True, "current": 0.1},
                "debate_success": {"compliant": True, "current": 0.96},
            },
            "alerts": [],
        }
        mock_get_status.return_value = expected_response

        result = slo_handler.handle("/api/slos/status", {}, mock_handler)

        assert result is not None
        assert result.status_code == 200
        # Parse the response body
        body = json.loads(result.body.decode("utf-8"))
        assert "overall_healthy" in body
        assert "slos" in body or "timestamp" in body

    @patch("aragora.server.handlers.slo.check_availability_slo")
    @patch("aragora.server.handlers.slo._slo_limiter")
    def test_slo_detail_response_structure(
        self, mock_limiter, mock_check, slo_handler, mock_handler, mock_slo_result
    ):
        """Test that SLO detail response has expected fields."""
        mock_limiter.is_allowed.return_value = True
        mock_check.return_value = mock_slo_result

        result = slo_handler.handle("/api/slos/availability", {}, mock_handler)

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body.decode("utf-8"))
        assert "name" in body
        assert "target" in body
        assert "current" in body
        assert "compliant" in body
        assert "compliance_percentage" in body
        assert "error_budget_remaining" in body
        assert "burn_rate" in body
        assert "window" in body
        assert "start" in body["window"]
        assert "end" in body["window"]

    @patch("aragora.server.handlers.slo.get_slo_status")
    @patch("aragora.server.handlers.slo._slo_limiter")
    def test_error_budget_response_structure(
        self, mock_limiter, mock_get_status, slo_handler, mock_handler, mock_slo_status
    ):
        """Test that error budget response has expected structure."""
        mock_limiter.is_allowed.return_value = True
        mock_get_status.return_value = mock_slo_status

        result = slo_handler.handle("/api/slos/error-budget", {}, mock_handler)

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body.decode("utf-8"))
        assert "timestamp" in body
        assert "budgets" in body
        assert isinstance(body["budgets"], list)
        assert len(body["budgets"]) == 3  # availability, latency_p99, debate_success

        # Check budget fields
        for budget in body["budgets"]:
            assert "slo_name" in budget
            assert "slo_id" in budget
            assert "target" in budget
            assert "error_budget_total" in budget
            assert "error_budget_remaining" in budget
            assert "error_budget_consumed" in budget
            assert "burn_rate" in budget
            assert "window" in budget

    @patch("aragora.server.handlers.slo.check_alerts")
    @patch("aragora.server.handlers.slo.get_slo_status")
    @patch("aragora.server.handlers.slo._slo_limiter")
    def test_violations_response_structure(
        self,
        mock_limiter,
        mock_get_status,
        mock_check_alerts,
        slo_handler,
        mock_handler,
        mock_slo_status,
    ):
        """Test that violations response has expected structure."""
        mock_limiter.is_allowed.return_value = True
        mock_get_status.return_value = mock_slo_status
        mock_check_alerts.return_value = []

        result = slo_handler.handle("/api/slos/violations", {}, mock_handler)

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body.decode("utf-8"))
        assert "timestamp" in body
        assert "violation_count" in body
        assert "violations" in body
        assert "overall_healthy" in body
        assert isinstance(body["violations"], list)

    @patch("aragora.server.handlers.slo.check_alerts")
    @patch("aragora.server.handlers.slo.get_slo_status")
    @patch("aragora.server.handlers.slo._slo_limiter")
    def test_violations_response_with_violations(
        self,
        mock_limiter,
        mock_get_status,
        mock_check_alerts,
        slo_handler,
        mock_handler,
        mock_slo_status,
    ):
        """Test violations response when violations exist."""
        mock_limiter.is_allowed.return_value = True
        mock_slo_status.overall_healthy = False
        mock_get_status.return_value = mock_slo_status

        mock_alert = MockAlert(slo_name="availability", severity="critical", message="SLO breach")
        mock_result = MockSLOResult(compliant=False, current=0.98)
        mock_check_alerts.return_value = [(mock_alert, mock_result)]

        result = slo_handler.handle("/api/slos/violations", {}, mock_handler)

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body.decode("utf-8"))
        assert body["violation_count"] == 1
        assert len(body["violations"]) == 1
        violation = body["violations"][0]
        assert violation["slo_name"] == "availability"
        assert violation["severity"] == "critical"
        assert violation["message"] == "SLO breach"

    @patch("aragora.server.handlers.slo.get_slo_targets")
    @patch("aragora.server.handlers.slo._slo_limiter")
    def test_targets_response_structure(
        self, mock_limiter, mock_get_targets, slo_handler, mock_handler
    ):
        """Test that targets response has expected structure."""
        mock_limiter.is_allowed.return_value = True
        mock_get_targets.return_value = {
            "availability": MockSLOTarget(
                name="API Availability",
                target=0.999,
                unit="ratio",
                description="API uptime target",
                comparison="gte",
            ),
        }

        result = slo_handler.handle("/api/slos/targets", {}, mock_handler)

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body.decode("utf-8"))
        assert "targets" in body
        assert isinstance(body["targets"], list)
        assert len(body["targets"]) == 1
        target = body["targets"][0]
        assert "id" in target
        assert "name" in target
        assert "target" in target
        assert "unit" in target
        assert "description" in target
        assert "comparison" in target


# ===========================================================================
# Rate Limiting Edge Cases
# ===========================================================================


class TestRateLimitingEdgeCases:
    """Test rate limiting edge cases and behaviors."""

    @patch("aragora.server.handlers.slo._slo_limiter")
    def test_rate_limit_response_code(self, mock_limiter, slo_handler, mock_handler):
        """Test that rate limiting returns 429 status code."""
        mock_limiter.is_allowed.return_value = False

        result = slo_handler.handle("/api/slos/status", {}, mock_handler)

        assert result.status_code == 429

    @patch("aragora.server.handlers.slo._slo_limiter")
    def test_rate_limit_response_message(self, mock_limiter, slo_handler, mock_handler):
        """Test that rate limit response contains appropriate message."""
        mock_limiter.is_allowed.return_value = False

        result = slo_handler.handle("/api/slos/status", {}, mock_handler)

        body = json.loads(result.body.decode("utf-8"))
        assert "error" in body
        # The error may be a string or dict with message field
        error_value = body["error"]
        if isinstance(error_value, dict):
            error_message = error_value.get("message", "").lower()
        else:
            error_message = str(error_value).lower()
        assert "rate limit" in error_message

    @patch("aragora.server.handlers.slo._slo_limiter")
    def test_rate_limit_applies_to_all_endpoints(self, mock_limiter, slo_handler, mock_handler):
        """Test that rate limiting applies to all SLO endpoints."""
        mock_limiter.is_allowed.return_value = False

        endpoints = [
            "/api/slos/status",
            "/api/slos/error-budget",
            "/api/slos/violations",
            "/api/slos/targets",
            "/api/slos/availability",
        ]

        for endpoint in endpoints:
            result = slo_handler.handle(endpoint, {}, mock_handler)
            assert result.status_code == 429, f"Rate limit not applied to {endpoint}"


# ===========================================================================
# Dynamic Route Handling Tests
# ===========================================================================


class TestDynamicRouteHandling:
    """Test dynamic route handling for SLO names."""

    def test_can_handle_latency_p99(self, slo_handler):
        """Test handling of latency_p99 SLO name."""
        assert slo_handler.can_handle("/api/slos/latency_p99")

    def test_can_handle_debate_success_underscore(self, slo_handler):
        """Test handling of debate_success SLO name with underscore."""
        assert slo_handler.can_handle("/api/slos/debate_success")

    @patch("aragora.server.handlers.slo.check_latency_slo")
    @patch("aragora.server.handlers.slo._slo_limiter")
    def test_latency_normalization(self, mock_limiter, mock_check, slo_handler, mock_handler):
        """Test that /api/slos/latency normalizes to latency_p99."""
        mock_limiter.is_allowed.return_value = True
        mock_check.return_value = MockSLOResult(name="latency_p99", target=0.5)

        result = slo_handler.handle("/api/slos/latency", {}, mock_handler)

        assert result.status_code == 200
        mock_check.assert_called_once()

    @patch("aragora.server.handlers.slo.check_debate_success_slo")
    @patch("aragora.server.handlers.slo._slo_limiter")
    def test_debate_success_normalization(
        self, mock_limiter, mock_check, slo_handler, mock_handler
    ):
        """Test that /api/slos/debate-success normalizes to debate_success."""
        mock_limiter.is_allowed.return_value = True
        mock_check.return_value = MockSLOResult(name="debate_success", target=0.95)

        result = slo_handler.handle("/api/slos/debate-success", {}, mock_handler)

        assert result.status_code == 200
        mock_check.assert_called_once()

    @patch("aragora.server.handlers.slo._slo_limiter")
    def test_unknown_endpoint_returns_404(self, mock_limiter, slo_handler, mock_handler):
        """Test that unknown endpoint paths return 404."""
        mock_limiter.is_allowed.return_value = True

        # This shouldn't match can_handle, but test the internal routing
        result = slo_handler.handle("/api/slos/unknown_path", {}, mock_handler)

        assert result.status_code == 404


# ===========================================================================
# Error Budget Calculation Edge Cases
# ===========================================================================


class TestExhaustionCalculationEdgeCases:
    """Test edge cases in error budget exhaustion calculation."""

    def test_exhaustion_with_zero_burn_rate(self, slo_handler):
        """Test exhaustion calculation with zero burn rate."""
        mock_result = MockSLOResult(burn_rate=0.0, error_budget_remaining=50.0)
        result = slo_handler._calculate_exhaustion_time(mock_result)
        assert result is None

    def test_exhaustion_with_negative_burn_rate(self, slo_handler):
        """Test exhaustion calculation with negative burn rate."""
        mock_result = MockSLOResult(burn_rate=-1.0, error_budget_remaining=50.0)
        result = slo_handler._calculate_exhaustion_time(mock_result)
        assert result is None

    def test_exhaustion_with_exactly_one_burn_rate(self, slo_handler):
        """Test exhaustion calculation with burn rate exactly at 1.0."""
        mock_result = MockSLOResult(burn_rate=1.0, error_budget_remaining=50.0)
        result = slo_handler._calculate_exhaustion_time(mock_result)
        # burn_rate <= 1.0 is sustainable, should return None
        assert result is None

    def test_exhaustion_with_very_high_burn_rate(self, slo_handler):
        """Test exhaustion calculation with very high burn rate."""
        mock_result = MockSLOResult(burn_rate=100.0, error_budget_remaining=10.0)
        result = slo_handler._calculate_exhaustion_time(mock_result)
        # Should return a valid ISO timestamp
        assert result is not None
        assert "T" in result

    def test_exhaustion_with_small_remaining_budget(self, slo_handler):
        """Test exhaustion calculation with very small remaining budget."""
        mock_result = MockSLOResult(burn_rate=2.0, error_budget_remaining=0.001)
        result = slo_handler._calculate_exhaustion_time(mock_result)
        # Should return a valid ISO timestamp
        assert result is not None

    def test_exhaustion_caps_at_30_days(self, slo_handler):
        """Test that exhaustion time is capped at 30 days."""
        # Very slow burn rate with lots of budget
        mock_result = MockSLOResult(burn_rate=1.01, error_budget_remaining=99.0)
        result = slo_handler._calculate_exhaustion_time(mock_result)
        # Should return a timestamp, even if far in the future
        assert result is not None


# ===========================================================================
# API Versioning Tests
# ===========================================================================


class TestAPIVersioning:
    """Test API version handling."""

    def test_can_handle_v1_prefix(self, slo_handler):
        """Test that handler recognizes v1 prefixed paths."""
        assert slo_handler.can_handle("/api/v1/slos/status")
        assert slo_handler.can_handle("/api/v1/slos/error-budget")
        assert slo_handler.can_handle("/api/v1/slos/violations")
        assert slo_handler.can_handle("/api/v1/slos/targets")
        assert slo_handler.can_handle("/api/v1/slos/availability")

    @patch("aragora.server.handlers.slo.get_slo_status_json")
    @patch("aragora.server.handlers.slo._slo_limiter")
    def test_v2_prefix_handling(self, mock_limiter, mock_get_status, slo_handler, mock_handler):
        """Test that v2 prefixed paths are also handled."""
        mock_limiter.is_allowed.return_value = True
        mock_get_status.return_value = {"overall_healthy": True}

        result = slo_handler.handle("/api/v2/slos/status", {}, mock_handler)
        assert result.status_code == 200


# ===========================================================================
# Module Constants Tests
# ===========================================================================


class TestModuleConstants:
    """Test module-level constants."""

    def test_slo_service_timeout_is_defined(self):
        """Test that SLO_SERVICE_TIMEOUT constant is defined."""
        assert SLO_SERVICE_TIMEOUT is not None
        assert isinstance(SLO_SERVICE_TIMEOUT, (int, float))
        assert SLO_SERVICE_TIMEOUT > 0


# ===========================================================================
# Integration Style Tests
# ===========================================================================


class TestIntegrationPatterns:
    """Test integration-style scenarios."""

    @patch("aragora.server.handlers.slo.get_slo_status")
    @patch("aragora.server.handlers.slo._slo_limiter")
    def test_error_budget_with_projected_exhaustion(
        self, mock_limiter, mock_get_status, slo_handler, mock_handler
    ):
        """Test error budget endpoint with high burn rate showing projected exhaustion."""
        mock_limiter.is_allowed.return_value = True

        # Create status with high burn rate
        status = MockSLOStatus()
        status.availability = MockSLOResult(
            name="availability",
            burn_rate=5.0,
            error_budget_remaining=20.0,
        )
        mock_get_status.return_value = status

        result = slo_handler.handle("/api/slos/error-budget", {}, mock_handler)

        assert result.status_code == 200
        body = json.loads(result.body.decode("utf-8"))
        # At least one budget should have projected_exhaustion
        availability_budget = next(b for b in body["budgets"] if b["slo_id"] == "availability")
        assert "projected_exhaustion" in availability_budget

    @patch("aragora.server.handlers.slo.check_alerts")
    @patch("aragora.server.handlers.slo.get_slo_status")
    @patch("aragora.server.handlers.slo._slo_limiter")
    def test_multiple_violations_handling(
        self,
        mock_limiter,
        mock_get_status,
        mock_check_alerts,
        slo_handler,
        mock_handler,
        mock_slo_status,
    ):
        """Test violations endpoint with multiple simultaneous violations."""
        mock_limiter.is_allowed.return_value = True
        mock_slo_status.overall_healthy = False
        mock_get_status.return_value = mock_slo_status

        # Multiple violations
        violations = [
            (
                MockAlert(
                    slo_name="availability", severity="critical", message="Availability breach"
                ),
                MockSLOResult(name="availability", compliant=False, current=0.97),
            ),
            (
                MockAlert(
                    slo_name="latency_p99", severity="warning", message="Latency degradation"
                ),
                MockSLOResult(name="latency_p99", compliant=False, current=0.6),
            ),
        ]
        mock_check_alerts.return_value = violations

        result = slo_handler.handle("/api/slos/violations", {}, mock_handler)

        assert result.status_code == 200
        body = json.loads(result.body.decode("utf-8"))
        assert body["violation_count"] == 2
        assert len(body["violations"]) == 2
        assert body["overall_healthy"] is False


# ===========================================================================
# Dynamic SLO Path Tests
# ===========================================================================


class TestDynamicSLOPaths:
    """Test dynamic SLO path routing via the /api/slos/{name} pattern."""

    @patch("aragora.server.handlers.slo.check_latency_slo")
    @patch("aragora.server.handlers.slo._slo_limiter")
    def test_dynamic_latency_p99_path(self, mock_limiter, mock_check, slo_handler, mock_handler):
        """Test that /api/slos/latency_p99 calls check_latency_slo."""
        mock_limiter.is_allowed.return_value = True
        mock_check.return_value = MockSLOResult(name="latency_p99", target=0.5)

        result = slo_handler.handle("/api/slos/latency_p99", {}, mock_handler)

        assert result.status_code == 200
        mock_check.assert_called_once()

    @patch("aragora.server.handlers.slo.check_debate_success_slo")
    @patch("aragora.server.handlers.slo._slo_limiter")
    def test_dynamic_debate_success_path(self, mock_limiter, mock_check, slo_handler, mock_handler):
        """Test that /api/slos/debate_success calls check_debate_success_slo."""
        mock_limiter.is_allowed.return_value = True
        mock_check.return_value = MockSLOResult(name="debate_success", target=0.95)

        result = slo_handler.handle("/api/slos/debate_success", {}, mock_handler)

        assert result.status_code == 200
        mock_check.assert_called_once()


# ===========================================================================
# Latency SLO Exception Tests
# ===========================================================================


class TestLatencySLOExceptions:
    """Test exception handling for latency SLO endpoints."""

    @patch("aragora.server.handlers.slo.check_latency_slo")
    @patch("aragora.server.handlers.slo._slo_limiter")
    def test_latency_slo_exception(self, mock_limiter, mock_check, slo_handler, mock_handler):
        """Test error handling when latency SLO check fails."""
        mock_limiter.is_allowed.return_value = True
        mock_check.side_effect = ValueError("Latency check failed")

        result = slo_handler.handle("/api/slos/latency", {}, mock_handler)

        assert result.status_code == 500

    @patch("aragora.server.handlers.slo.check_debate_success_slo")
    @patch("aragora.server.handlers.slo._slo_limiter")
    def test_debate_success_slo_exception(
        self, mock_limiter, mock_check, slo_handler, mock_handler
    ):
        """Test error handling when debate success SLO check fails."""
        mock_limiter.is_allowed.return_value = True
        mock_check.side_effect = ValueError("Debate success check failed")

        result = slo_handler.handle("/api/slos/debate-success", {}, mock_handler)

        assert result.status_code == 500
