"""Tests for continuous compliance monitoring."""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.compliance.monitor import (
    ComplianceHealth,
    ComplianceMonitor,
    ComplianceMonitorConfig,
    ComplianceStatus,
    DriftEvent,
    FrameworkStatus,
    ViolationTrend,
    get_compliance_monitor,
    get_compliance_status,
    init_compliance_monitoring,
    start_compliance_monitoring,
    stop_compliance_monitoring,
)


class TestComplianceHealth:
    """Test ComplianceHealth enum."""

    def test_health_values(self):
        """Test all health values exist."""
        assert ComplianceHealth.HEALTHY.value == "healthy"
        assert ComplianceHealth.DEGRADED.value == "degraded"
        assert ComplianceHealth.AT_RISK.value == "at_risk"
        assert ComplianceHealth.CRITICAL.value == "critical"


class TestViolationTrend:
    """Test ViolationTrend enum."""

    def test_trend_values(self):
        """Test all trend values exist."""
        assert ViolationTrend.IMPROVING.value == "improving"
        assert ViolationTrend.STABLE.value == "stable"
        assert ViolationTrend.WORSENING.value == "worsening"


class TestFrameworkStatus:
    """Test FrameworkStatus dataclass."""

    def test_default_values(self):
        """Test default framework status."""
        status = FrameworkStatus(framework="soc2")
        assert status.framework == "soc2"
        assert status.enabled
        assert status.score == 100.0
        assert status.critical_violations == 0
        assert status.health == ComplianceHealth.HEALTHY

    def test_critical_health(self):
        """Test critical health detection."""
        status = FrameworkStatus(framework="soc2", critical_violations=1)
        assert status.health == ComplianceHealth.CRITICAL

    def test_at_risk_health(self):
        """Test at-risk health detection."""
        status = FrameworkStatus(framework="soc2", major_violations=2)
        assert status.health == ComplianceHealth.AT_RISK

    def test_degraded_health(self):
        """Test degraded health detection."""
        status = FrameworkStatus(framework="soc2", moderate_violations=3)
        assert status.health == ComplianceHealth.DEGRADED


class TestComplianceStatus:
    """Test ComplianceStatus dataclass."""

    def test_default_values(self):
        """Test default compliance status."""
        status = ComplianceStatus()
        assert status.overall_health == ComplianceHealth.HEALTHY
        assert status.overall_score == 100.0
        assert status.open_violations == 0
        assert status.audit_trail_verified
        assert isinstance(status.timestamp, datetime)


class TestComplianceMonitorConfig:
    """Test ComplianceMonitorConfig dataclass."""

    def test_default_values(self):
        """Test default configuration."""
        config = ComplianceMonitorConfig()
        assert config.enabled
        assert config.check_interval_seconds == 300.0
        assert config.alert_on_critical
        assert "soc2" in config.enabled_frameworks

    def test_custom_values(self):
        """Test custom configuration."""
        config = ComplianceMonitorConfig(
            check_interval_seconds=60.0,
            alert_on_critical=False,
            enabled_frameworks={"gdpr"},
        )
        assert config.check_interval_seconds == 60.0
        assert not config.alert_on_critical
        assert config.enabled_frameworks == {"gdpr"}


class TestComplianceMonitor:
    """Test ComplianceMonitor class."""

    def test_initialization(self):
        """Test monitor initialization."""
        config = ComplianceMonitorConfig()
        monitor = ComplianceMonitor(config)
        assert monitor.config == config
        assert not monitor._running
        assert monitor.get_status() is None

    def test_callback_registration(self):
        """Test callback registration."""
        config = ComplianceMonitorConfig()
        monitor = ComplianceMonitor(config)

        callback = MagicMock()
        monitor.register_violation_callback(callback)
        assert callback in monitor._violation_callbacks

        drift_callback = MagicMock()
        monitor.register_drift_callback(drift_callback)
        assert drift_callback in monitor._drift_callbacks

    @pytest.mark.asyncio
    async def test_start_stop(self):
        """Test starting and stopping monitor."""
        config = ComplianceMonitorConfig(check_interval_seconds=0.1)
        monitor = ComplianceMonitor(config)

        await monitor.start()
        assert monitor._running
        assert monitor._task is not None

        # Wait briefly for a check
        await asyncio.sleep(0.2)

        await monitor.stop()
        assert not monitor._running

    @pytest.mark.asyncio
    async def test_record_drift(self):
        """Test recording drift events."""
        config = ComplianceMonitorConfig(alert_on_drift=False)
        monitor = ComplianceMonitor(config)

        await monitor.record_drift(
            drift_type="policy_version",
            framework="soc2",
            description="Policy version changed",
            severity="moderate",
            current_value="v2",
            expected_value="v1",
        )

        events = monitor.get_drift_events()
        assert len(events) == 1
        assert events[0].drift_type == "policy_version"
        assert events[0].framework == "soc2"

    def test_calculate_overall_health_healthy(self):
        """Test health calculation with no violations."""
        config = ComplianceMonitorConfig()
        monitor = ComplianceMonitor(config)

        status = ComplianceStatus()
        health = monitor._calculate_overall_health(status)
        assert health == ComplianceHealth.HEALTHY

    def test_calculate_overall_health_critical(self):
        """Test health calculation with critical violations."""
        config = ComplianceMonitorConfig()
        monitor = ComplianceMonitor(config)

        status = ComplianceStatus()
        status.frameworks["soc2"] = FrameworkStatus(framework="soc2", critical_violations=1)
        health = monitor._calculate_overall_health(status)
        assert health == ComplianceHealth.CRITICAL

    def test_calculate_overall_score(self):
        """Test score calculation."""
        config = ComplianceMonitorConfig()
        monitor = ComplianceMonitor(config)

        status = ComplianceStatus()
        status.frameworks["soc2"] = FrameworkStatus(framework="soc2", score=90.0)
        status.frameworks["gdpr"] = FrameworkStatus(framework="gdpr", score=80.0)

        score = monitor._calculate_overall_score(status)
        assert score == 85.0

    def test_calculate_trend_stable(self):
        """Test trend calculation with insufficient history."""
        config = ComplianceMonitorConfig()
        monitor = ComplianceMonitor(config)

        trend = monitor._calculate_trend()
        assert trend == ViolationTrend.STABLE


class TestModuleFunctions:
    """Test module-level functions."""

    def test_init_compliance_monitoring(self):
        """Test init function."""
        monitor = init_compliance_monitoring(
            check_interval_seconds=60,
            alert_on_critical=True,
        )
        assert monitor is not None
        assert monitor.config.check_interval_seconds == 60

        # Verify global accessor works
        global_monitor = get_compliance_monitor()
        assert global_monitor is monitor

    @pytest.mark.asyncio
    async def test_get_compliance_status(self):
        """Test status getter."""
        init_compliance_monitoring(check_interval_seconds=60)
        status = await get_compliance_status()
        # May be None if no checks have run yet
        assert status is None or isinstance(status, ComplianceStatus)


class TestDriftEvent:
    """Test DriftEvent dataclass."""

    def test_drift_event_creation(self):
        """Test drift event creation."""
        event = DriftEvent(
            timestamp=datetime.now(timezone.utc),
            drift_type="config_change",
            framework="hipaa",
            description="Encryption config changed",
            severity="major",
            current_value="AES-128",
            expected_value="AES-256",
        )
        assert event.drift_type == "config_change"
        assert event.framework == "hipaa"
        assert event.severity == "major"
