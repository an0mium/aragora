"""Tests for SLO Handler.

Tests the SLOHandler which provides REST API endpoints for SLO monitoring,
error budgets, and violation tracking.
"""

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.base import HandlerResult
from aragora.server.handlers.slo import SLOHandler


def parse_body(result: HandlerResult) -> dict:
    """Parse JSON body from HandlerResult."""
    return json.loads(result.body.decode("utf-8"))


def get_error_fields(body: dict) -> tuple[str | None, str | None]:
    """Extract error message/code from either legacy or structured format."""
    error = body.get("error")
    if isinstance(error, dict):
        return error.get("message"), error.get("code")
    return error, body.get("code")


@pytest.fixture
def server_context():
    """Create mock server context."""
    return {
        "storage": MagicMock(),
        "elo_system": MagicMock(),
    }


@pytest.fixture
def handler(server_context):
    """Create SLO handler with mock context."""
    return SLOHandler(server_context)


class TestSLOHandlerCanHandle:
    """Tests for can_handle method."""

    def test_handles_slo_status(self, handler):
        """Test that handler can handle SLO status path."""
        assert handler.can_handle("/api/slos/status") is True

    def test_handles_slo_error_budget(self, handler):
        """Test that handler can handle error budget path."""
        assert handler.can_handle("/api/slos/error-budget") is True

    def test_handles_slo_violations(self, handler):
        """Test that handler can handle violations path."""
        assert handler.can_handle("/api/slos/violations") is True

    def test_handles_slo_targets(self, handler):
        """Test that handler can handle targets path."""
        assert handler.can_handle("/api/slos/targets") is True

    def test_handles_slo_availability(self, handler):
        """Test that handler can handle specific SLO path."""
        assert handler.can_handle("/api/slos/availability") is True

    def test_handles_slo_latency(self, handler):
        """Test that handler can handle latency SLO path."""
        assert handler.can_handle("/api/slos/latency") is True

    def test_handles_versioned_path(self, handler):
        """Test that handler can handle versioned paths."""
        assert handler.can_handle("/api/v1/slos/status") is True

    def test_rejects_unrelated_path(self, handler):
        """Test that handler rejects unrelated paths."""
        assert handler.can_handle("/api/debates") is False
        assert handler.can_handle("/api/agents") is False


class TestSLOHandlerRateLimiting:
    """Tests for rate limiting in SLO handler."""

    def test_rate_limit_exceeded(self, handler):
        """Test that rate limit exceeded returns proper error code."""
        mock_handler = MagicMock()
        mock_handler.client_address = ("127.0.0.1", 12345)

        with patch("aragora.server.handlers.slo._slo_limiter") as mock_limiter:
            mock_limiter.is_allowed.return_value = False
            result = handler.handle("/api/slos/status", {}, mock_handler)

        assert result.status_code == 429
        body = parse_body(result)
        _, code = get_error_fields(body)
        assert code == "RATE_LIMITED"


class TestSLOStatusEndpoint:
    """Tests for SLO status endpoint."""

    def test_slo_status_success(self, handler):
        """Test that SLO status returns successfully."""
        mock_handler = MagicMock()
        mock_handler.client_address = ("127.0.0.1", 12345)

        mock_status = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "overall_healthy": True,
            "slos": [],
        }

        with patch("aragora.server.handlers.slo._slo_limiter") as mock_limiter:
            mock_limiter.is_allowed.return_value = True
            with patch("aragora.server.handlers.slo.get_slo_status_json") as mock_get:
                mock_get.return_value = mock_status
                result = handler.handle("/api/slos/status", {}, mock_handler)

        assert result.status_code == 200
        body = parse_body(result)
        assert "overall_healthy" in body

    def test_slo_status_error(self, handler):
        """Test that SLO status error returns proper error code."""
        mock_handler = MagicMock()
        mock_handler.client_address = ("127.0.0.1", 12345)

        with patch("aragora.server.handlers.slo._slo_limiter") as mock_limiter:
            mock_limiter.is_allowed.return_value = True
            with patch("aragora.server.handlers.slo.get_slo_status_json") as mock_get:
                mock_get.side_effect = Exception("Database error")
                result = handler.handle("/api/slos/status", {}, mock_handler)

        assert result.status_code == 500
        body = parse_body(result)
        _, code = get_error_fields(body)
        assert code == "SLO_STATUS_ERROR"


class TestSLODetailEndpoint:
    """Tests for individual SLO detail endpoint."""

    def test_slo_detail_unknown_slo(self, handler):
        """Test that unknown SLO returns proper error code."""
        with patch("aragora.server.handlers.slo._slo_limiter") as mock_limiter:
            mock_limiter.is_allowed.return_value = True
            result = handler._handle_slo_detail("unknown_slo")

        assert result.status_code == 404
        body = parse_body(result)
        _, code = get_error_fields(body)
        assert code == "UNKNOWN_SLO"

    def test_slo_detail_availability_success(self, handler):
        """Test that availability SLO returns successfully."""
        mock_result = MagicMock()
        mock_result.name = "Availability"
        mock_result.target = 99.9
        mock_result.current = 99.95
        mock_result.compliant = True
        mock_result.compliance_percentage = 100.0
        mock_result.error_budget_remaining = 95.0
        mock_result.burn_rate = 0.5
        mock_result.window_start = datetime.now(timezone.utc)
        mock_result.window_end = datetime.now(timezone.utc)

        with patch("aragora.server.handlers.slo.check_availability_slo") as mock_check:
            mock_check.return_value = mock_result
            result = handler._handle_slo_detail("availability")

        assert result.status_code == 200
        body = parse_body(result)
        assert body["name"] == "Availability"
        assert body["compliant"] is True


class TestUnknownEndpoint:
    """Tests for unknown endpoint handling."""

    def test_unknown_endpoint(self, handler):
        """Test that unknown endpoint returns proper error code."""
        mock_handler = MagicMock()
        mock_handler.client_address = ("127.0.0.1", 12345)

        with patch("aragora.server.handlers.slo._slo_limiter") as mock_limiter:
            mock_limiter.is_allowed.return_value = True
            result = handler.handle("/api/slos/unknown", {}, mock_handler)

        # Unknown SLO name should return 404
        assert result.status_code == 404
        body = parse_body(result)
        _, code = get_error_fields(body)
        assert code == "UNKNOWN_SLO"


class TestErrorBudgetEndpoint:
    """Tests for error budget endpoint."""

    def test_error_budget_success(self, handler):
        """Test that error budget returns successfully."""
        mock_handler = MagicMock()
        mock_handler.client_address = ("127.0.0.1", 12345)

        mock_status = MagicMock()
        mock_status.timestamp = datetime.now(timezone.utc)

        # Create mock SLO results
        for attr in ["availability", "latency_p99", "debate_success"]:
            mock_result = MagicMock()
            mock_result.name = attr.replace("_", " ").title()
            mock_result.target = 99.0
            mock_result.error_budget_remaining = 50.0
            mock_result.burn_rate = 1.0
            mock_result.window_start = datetime.now(timezone.utc)
            mock_result.window_end = datetime.now(timezone.utc)
            setattr(mock_status, attr, mock_result)

        with patch("aragora.server.handlers.slo._slo_limiter") as mock_limiter:
            mock_limiter.is_allowed.return_value = True
            with patch("aragora.server.handlers.slo.get_slo_status") as mock_get:
                mock_get.return_value = mock_status
                result = handler.handle("/api/slos/error-budget", {}, mock_handler)

        assert result.status_code == 200
        body = parse_body(result)
        assert "budgets" in body
        assert len(body["budgets"]) == 3

    def test_error_budget_error(self, handler):
        """Test that error budget error returns proper error code."""
        mock_handler = MagicMock()
        mock_handler.client_address = ("127.0.0.1", 12345)

        with patch("aragora.server.handlers.slo._slo_limiter") as mock_limiter:
            mock_limiter.is_allowed.return_value = True
            with patch("aragora.server.handlers.slo.get_slo_status") as mock_get:
                mock_get.side_effect = Exception("Metrics error")
                result = handler.handle("/api/slos/error-budget", {}, mock_handler)

        assert result.status_code == 500
        body = parse_body(result)
        _, code = get_error_fields(body)
        assert code == "ERROR_BUDGET_ERROR"


class TestViolationsEndpoint:
    """Tests for violations endpoint."""

    def test_violations_error(self, handler):
        """Test that violations error returns proper error code."""
        mock_handler = MagicMock()
        mock_handler.client_address = ("127.0.0.1", 12345)

        with patch("aragora.server.handlers.slo._slo_limiter") as mock_limiter:
            mock_limiter.is_allowed.return_value = True
            with patch("aragora.server.handlers.slo.get_slo_status") as mock_get:
                mock_get.side_effect = Exception("Database error")
                result = handler.handle("/api/slos/violations", {}, mock_handler)

        assert result.status_code == 500
        body = parse_body(result)
        _, code = get_error_fields(body)
        assert code == "VIOLATIONS_ERROR"


class TestTargetsEndpoint:
    """Tests for targets endpoint."""

    def test_targets_error(self, handler):
        """Test that targets error returns proper error code."""
        mock_handler = MagicMock()
        mock_handler.client_address = ("127.0.0.1", 12345)

        with patch("aragora.server.handlers.slo._slo_limiter") as mock_limiter:
            mock_limiter.is_allowed.return_value = True
            with patch("aragora.server.handlers.slo.get_slo_targets") as mock_get:
                mock_get.side_effect = Exception("Config error")
                result = handler.handle("/api/slos/targets", {}, mock_handler)

        assert result.status_code == 500
        body = parse_body(result)
        _, code = get_error_fields(body)
        assert code == "TARGETS_ERROR"
