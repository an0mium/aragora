"""Tests for compliance monitor event emissions."""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest

# Import the module to check what's available
from aragora.compliance.monitor import ComplianceMonitor, ComplianceHealth, ViolationTrend


@pytest.fixture
def monitor():
    # ComplianceMonitor may require config - check constructor
    try:
        return ComplianceMonitor()
    except TypeError:
        # Try with a config object
        from aragora.compliance.monitor import ComplianceMonitorConfig
        return ComplianceMonitor(config=ComplianceMonitorConfig())


class TestComplianceEventEmission:
    """Tests for _emit_compliance_status_event."""

    def test_emits_compliance_status_event(self, monitor) -> None:
        # Create a mock status object
        status = MagicMock()
        status.overall_health = ComplianceHealth.HEALTHY
        status.overall_score = 95.5
        status.open_violations = 2
        status.trend = ViolationTrend.IMPROVING
        status.frameworks = {}

        with patch("aragora.events.dispatcher.dispatch_event") as mock_dispatch:
            monitor._emit_compliance_status_event(status)

        mock_dispatch.assert_called_once()
        assert mock_dispatch.call_args[0][0] == "compliance_status_updated"
        data = mock_dispatch.call_args[0][1]
        assert data["overall_health"] == "healthy"
        assert data["overall_score"] == 95.5
        assert data["open_violations"] == 2

    def test_emits_with_framework_details(self, monitor) -> None:
        status = MagicMock()
        status.overall_health = ComplianceHealth.DEGRADED
        status.overall_score = 72.0
        status.open_violations = 5
        status.trend = ViolationTrend.WORSENING

        # Mock framework status
        fs = MagicMock()
        fs.health = ComplianceHealth.AT_RISK
        fs.score = 60
        fs.critical_violations = 1
        status.frameworks = {"soc2": fs}

        with patch("aragora.events.dispatcher.dispatch_event") as mock_dispatch:
            monitor._emit_compliance_status_event(status)

        data = mock_dispatch.call_args[0][1]
        assert "soc2" in data["frameworks"]
        assert data["frameworks"]["soc2"]["critical"] == 1

    def test_handles_import_error(self, monitor) -> None:
        status = MagicMock()
        status.overall_health = ComplianceHealth.HEALTHY
        status.overall_score = 100.0
        status.open_violations = 0
        status.trend = ViolationTrend.STABLE
        status.frameworks = {}

        with patch(
            "aragora.events.dispatcher.dispatch_event",
            side_effect=ImportError("no module"),
        ):
            # Should not raise
            monitor._emit_compliance_status_event(status)

    def test_handles_empty_frameworks(self, monitor) -> None:
        status = MagicMock()
        status.overall_health = ComplianceHealth.HEALTHY
        status.overall_score = 100.0
        status.open_violations = 0
        status.trend = ViolationTrend.STABLE
        status.frameworks = {}

        with patch("aragora.events.dispatcher.dispatch_event") as mock_dispatch:
            monitor._emit_compliance_status_event(status)

        data = mock_dispatch.call_args[0][1]
        assert data["frameworks"] == {}

    def test_critical_health_emits_event(self, monitor) -> None:
        status = MagicMock()
        status.overall_health = ComplianceHealth.CRITICAL
        status.overall_score = 15.0
        status.open_violations = 20
        status.trend = ViolationTrend.WORSENING
        status.frameworks = {}

        with patch("aragora.events.dispatcher.dispatch_event") as mock_dispatch:
            monitor._emit_compliance_status_event(status)

        data = mock_dispatch.call_args[0][1]
        assert data["overall_health"] == "critical"
        assert data["open_violations"] == 20
