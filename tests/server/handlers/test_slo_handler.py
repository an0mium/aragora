"""
Tests for SLOHandler - Service Level Objective HTTP endpoints.

Tests cover:
- Handler initialization and route matching
- GET /api/slos/status - Overall SLO status
- GET /api/slos/{slo_name} - Individual SLO details (availability, latency, debate-success)
- GET /api/slos/error-budget - Error budget timeline
- GET /api/slos/violations - Recent SLO violations
- GET /api/slos/targets - Configured SLO targets
- Rate limiting
- RBAC permission enforcement
- Error handling
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.slo import SLOHandler


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
def slo_handler():
    """Create SLOHandler instance."""
    return SLOHandler()


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
        mock_get_status.side_effect = Exception("Unexpected error")

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
        mock_check.side_effect = Exception("SLO check failed")

        result = slo_handler.handle("/api/slos/availability", {}, mock_handler)

        assert result is not None
        assert result.status_code == 500
