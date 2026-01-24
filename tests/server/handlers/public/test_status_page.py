"""Tests for public status page handler."""

import json
import pytest
from datetime import datetime, timezone
from unittest.mock import Mock, patch

from aragora.server.handlers.public.status_page import (
    StatusPageHandler,
    ServiceStatus,
    ComponentHealth,
)


def _parse_json_result(result):
    """Parse JSON body from HandlerResult."""
    if result is None:
        return None
    # HandlerResult has body as bytes
    body = result.body if hasattr(result, "body") else result.get("body", result)
    if isinstance(body, bytes):
        return json.loads(body.decode("utf-8"))
    elif isinstance(body, str):
        return json.loads(body)
    return body


class TestStatusPageHandler:
    """Tests for StatusPageHandler."""

    @pytest.fixture
    def handler(self):
        """Create a status page handler."""
        return StatusPageHandler({})

    def test_can_handle_status_routes(self, handler):
        """Test handler recognizes status routes."""
        assert handler.can_handle("/status")
        assert handler.can_handle("/api/status")
        assert handler.can_handle("/api/status/summary")
        assert handler.can_handle("/api/status/history")
        assert handler.can_handle("/api/status/components")
        assert handler.can_handle("/api/status/incidents")

    def test_cannot_handle_non_status_routes(self, handler):
        """Test handler rejects non-status routes."""
        assert not handler.can_handle("/api/health")
        assert not handler.can_handle("/api/debates")
        assert not handler.can_handle("/")

    def test_json_status_summary(self, handler):
        """Test JSON status summary endpoint."""
        result = handler.handle("/api/status", {}, Mock())

        assert result is not None
        body = _parse_json_result(result)

        assert "status" in body
        assert "components" in body
        assert "timestamp" in body
        assert body["status"] in [s.value for s in ServiceStatus]

    def test_component_status(self, handler):
        """Test component status endpoint."""
        result = handler.handle("/api/status/components", {}, Mock())

        assert result is not None
        body = _parse_json_result(result)

        assert "components" in body
        assert len(body["components"]) > 0

        for component in body["components"]:
            assert "id" in component
            assert "name" in component
            assert "status" in component

    def test_uptime_history(self, handler):
        """Test uptime history endpoint."""
        result = handler.handle("/api/status/history", {}, Mock())

        assert result is not None
        body = _parse_json_result(result)

        assert "current" in body
        assert "periods" in body
        assert "24h" in body["periods"]
        assert "7d" in body["periods"]
        assert "30d" in body["periods"]

    def test_incidents(self, handler):
        """Test incidents endpoint."""
        result = handler.handle("/api/status/incidents", {}, Mock())

        assert result is not None
        body = _parse_json_result(result)

        assert "active" in body
        assert "recent" in body
        assert "scheduled_maintenance" in body

    def test_html_status_page(self, handler):
        """Test HTML status page endpoint."""
        result = handler.handle("/status", {}, Mock())

        assert result is not None
        assert isinstance(result, dict)
        assert "body" in result
        assert "headers" in result
        assert result["headers"]["Content-Type"] == "text/html; charset=utf-8"

        html = result["body"]
        assert "<!DOCTYPE html>" in html
        assert "Aragora Status" in html
        assert "All Systems" in html or "System" in html


class TestServiceStatus:
    """Tests for ServiceStatus enum."""

    def test_status_values(self):
        """Test all status values exist."""
        assert ServiceStatus.OPERATIONAL.value == "operational"
        assert ServiceStatus.DEGRADED.value == "degraded"
        assert ServiceStatus.PARTIAL_OUTAGE.value == "partial_outage"
        assert ServiceStatus.MAJOR_OUTAGE.value == "major_outage"
        assert ServiceStatus.MAINTENANCE.value == "maintenance"


class TestComponentHealth:
    """Tests for ComponentHealth dataclass."""

    def test_create_component_health(self):
        """Test creating ComponentHealth."""
        health = ComponentHealth(
            name="API",
            status=ServiceStatus.OPERATIONAL,
            response_time_ms=5.2,
            message="All good",
        )

        assert health.name == "API"
        assert health.status == ServiceStatus.OPERATIONAL
        assert health.response_time_ms == 5.2
        assert health.message == "All good"

    def test_component_health_defaults(self):
        """Test ComponentHealth default values."""
        health = ComponentHealth(
            name="Test",
            status=ServiceStatus.DEGRADED,
        )

        assert health.response_time_ms is None
        assert health.last_check is None
        assert health.message is None


class TestOverallStatus:
    """Tests for overall status calculation."""

    @pytest.fixture
    def handler(self):
        return StatusPageHandler({})

    def test_all_operational_returns_operational(self, handler):
        """Test all operational components = operational overall."""
        with patch.object(
            handler,
            "_check_all_components",
            return_value=[
                ComponentHealth("API", ServiceStatus.OPERATIONAL),
                ComponentHealth("DB", ServiceStatus.OPERATIONAL),
            ],
        ):
            assert handler._get_overall_status() == ServiceStatus.OPERATIONAL

    def test_one_degraded_returns_degraded(self, handler):
        """Test one degraded component = degraded overall."""
        with patch.object(
            handler,
            "_check_all_components",
            return_value=[
                ComponentHealth("API", ServiceStatus.OPERATIONAL),
                ComponentHealth("Cache", ServiceStatus.DEGRADED),
            ],
        ):
            assert handler._get_overall_status() == ServiceStatus.DEGRADED

    def test_one_partial_outage_returns_partial(self, handler):
        """Test one partial outage = partial outage overall."""
        with patch.object(
            handler,
            "_check_all_components",
            return_value=[
                ComponentHealth("API", ServiceStatus.OPERATIONAL),
                ComponentHealth("DB", ServiceStatus.PARTIAL_OUTAGE),
            ],
        ):
            assert handler._get_overall_status() == ServiceStatus.PARTIAL_OUTAGE

    def test_multiple_partial_returns_major(self, handler):
        """Test multiple partial outages = major outage overall."""
        with patch.object(
            handler,
            "_check_all_components",
            return_value=[
                ComponentHealth("API", ServiceStatus.PARTIAL_OUTAGE),
                ComponentHealth("DB", ServiceStatus.PARTIAL_OUTAGE),
            ],
        ):
            assert handler._get_overall_status() == ServiceStatus.MAJOR_OUTAGE

    def test_one_major_outage_returns_major(self, handler):
        """Test one major outage = major outage overall."""
        with patch.object(
            handler,
            "_check_all_components",
            return_value=[
                ComponentHealth("API", ServiceStatus.OPERATIONAL),
                ComponentHealth("DB", ServiceStatus.MAJOR_OUTAGE),
            ],
        ):
            assert handler._get_overall_status() == ServiceStatus.MAJOR_OUTAGE


class TestFormatUptime:
    """Tests for uptime formatting."""

    @pytest.fixture
    def handler(self):
        return StatusPageHandler({})

    def test_format_seconds(self, handler):
        """Test formatting seconds."""
        assert handler._format_uptime(30) == "< 1m"

    def test_format_minutes(self, handler):
        """Test formatting minutes."""
        assert handler._format_uptime(300) == "5m"

    def test_format_hours(self, handler):
        """Test formatting hours."""
        assert handler._format_uptime(7200) == "2h"

    def test_format_days(self, handler):
        """Test formatting days."""
        assert handler._format_uptime(172800) == "2d"

    def test_format_mixed(self, handler):
        """Test formatting mixed duration."""
        # 1 day, 2 hours, 30 minutes
        seconds = 86400 + 7200 + 1800
        assert handler._format_uptime(seconds) == "1d 2h 30m"
