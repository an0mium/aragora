"""
End-to-End SLO Alerting Flow Tests.

Verifies the complete pipeline:
  SLO violation → Alert callback → Notification/Persistence

Tests cover:
  - SLO violation triggers callback
  - Multiple callbacks invoked in order
  - Alert cooldown prevents spam
  - Error budget depletion warning
  - SLO recovery detection
  - Integration with SLO history store
  - Integration with incident store
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.observability.slo import (
    SLOAlertMonitor,
    SLOBreach,
    SLOResult,
    SLOStatus,
    check_alerts,
    get_default_alerts,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_slo_status_healthy() -> SLOStatus:
    """Create a healthy SLO status (all compliant)."""
    now = datetime.now(timezone.utc)
    window_start = now - timedelta(hours=24)

    return SLOStatus(
        availability=SLOResult(
            name="API Availability",
            target=0.999,
            current=0.9995,
            compliant=True,
            compliance_percentage=100.0,
            window_start=window_start,
            window_end=now,
            error_budget_remaining=80.0,
            burn_rate=0.5,
        ),
        latency_p99=SLOResult(
            name="p99 Latency",
            target=500.0,
            current=350.0,
            compliant=True,
            compliance_percentage=100.0,
            window_start=window_start,
            window_end=now,
            error_budget_remaining=75.0,
            burn_rate=0.8,
        ),
        debate_success=SLOResult(
            name="Debate Success Rate",
            target=0.95,
            current=0.98,
            compliant=True,
            compliance_percentage=100.0,
            window_start=window_start,
            window_end=now,
            error_budget_remaining=90.0,
            burn_rate=0.3,
        ),
    )


@pytest.fixture
def mock_slo_status_degraded() -> SLOStatus:
    """Create a degraded SLO status (some violations)."""
    now = datetime.now(timezone.utc)
    window_start = now - timedelta(hours=24)

    return SLOStatus(
        availability=SLOResult(
            name="API Availability",
            target=0.999,
            current=0.995,
            compliant=False,
            compliance_percentage=99.5,
            window_start=window_start,
            window_end=now,
            error_budget_remaining=25.0,  # Triggers warning (<50%)
            burn_rate=3.0,  # Triggers warning (>2.0)
        ),
        latency_p99=SLOResult(
            name="p99 Latency",
            target=500.0,
            current=650.0,
            compliant=False,
            compliance_percentage=76.9,
            window_start=window_start,
            window_end=now,
            error_budget_remaining=5.0,  # Triggers critical (<10%)
            burn_rate=12.0,  # Triggers critical (>10.0)
        ),
        debate_success=SLOResult(
            name="Debate Success Rate",
            target=0.95,
            current=0.97,
            compliant=True,
            compliance_percentage=100.0,
            window_start=window_start,
            window_end=now,
            error_budget_remaining=70.0,
            burn_rate=0.5,
        ),
    )


@pytest.fixture
def monitor() -> SLOAlertMonitor:
    """Create a fresh SLO alert monitor."""
    return SLOAlertMonitor(check_interval_seconds=1.0, cooldown_seconds=5.0)


# ---------------------------------------------------------------------------
# Tests: check_alerts function
# ---------------------------------------------------------------------------


class TestCheckAlerts:
    """Test the check_alerts function for triggering alerts."""

    def test_healthy_status_triggers_no_alerts(self, mock_slo_status_healthy: SLOStatus):
        """Healthy SLO status should trigger no alerts."""
        triggered = check_alerts(mock_slo_status_healthy)
        assert len(triggered) == 0

    def test_degraded_status_triggers_alerts(self, mock_slo_status_degraded: SLOStatus):
        """Degraded SLO status should trigger appropriate alerts."""
        triggered = check_alerts(mock_slo_status_degraded)

        # Should have multiple alerts for availability and latency
        assert len(triggered) >= 2

        slo_names = [alert.slo_name for alert, _ in triggered]
        assert "API Availability" in slo_names
        assert "p99 Latency" in slo_names

    def test_critical_severity_triggered(self, mock_slo_status_degraded: SLOStatus):
        """Critical alerts should trigger when error budget < 10%."""
        triggered = check_alerts(mock_slo_status_degraded)

        critical_alerts = [(a, r) for a, r in triggered if a.severity == "critical"]
        assert len(critical_alerts) >= 1

        # p99 Latency has error_budget_remaining=5%, should be critical
        latency_critical = [(a, r) for a, r in critical_alerts if a.slo_name == "p99 Latency"]
        assert len(latency_critical) == 1

    def test_warning_severity_triggered(self, mock_slo_status_degraded: SLOStatus):
        """Warning alerts should trigger when error budget < 50%."""
        triggered = check_alerts(mock_slo_status_degraded)

        warning_alerts = [(a, r) for a, r in triggered if a.severity == "warning"]
        assert len(warning_alerts) >= 1

    def test_default_alerts_exist(self):
        """Default alert configurations should be defined."""
        alerts = get_default_alerts()
        assert len(alerts) >= 6  # 2 per SLO (warning + critical)

        slo_names = {a.slo_name for a in alerts}
        assert "API Availability" in slo_names
        assert "p99 Latency" in slo_names
        assert "Debate Success Rate" in slo_names


# ---------------------------------------------------------------------------
# Tests: SLOAlertMonitor callbacks
# ---------------------------------------------------------------------------


class TestSLOAlertMonitorCallbacks:
    """Test SLOAlertMonitor callback invocation."""

    @pytest.mark.asyncio
    async def test_callback_invoked_on_breach(
        self, monitor: SLOAlertMonitor, mock_slo_status_degraded: SLOStatus
    ):
        """Callbacks should be invoked when SLO breach detected."""
        callback = AsyncMock()
        monitor.add_callback(callback)

        with patch(
            "aragora.observability.slo.get_slo_status",
            return_value=mock_slo_status_degraded,
        ):
            breaches = await monitor.check_and_alert()

        assert len(breaches) >= 1
        assert callback.call_count >= 1

        # Verify breach object passed to callback
        breach = callback.call_args_list[0][0][0]
        assert isinstance(breach, SLOBreach)
        assert breach.slo_name in ["API Availability", "p99 Latency", "Debate Success Rate"]

    @pytest.mark.asyncio
    async def test_multiple_callbacks_invoked(
        self, monitor: SLOAlertMonitor, mock_slo_status_degraded: SLOStatus
    ):
        """All registered callbacks should be invoked for each breach."""
        callback1 = AsyncMock()
        callback2 = MagicMock()  # Sync callback
        monitor.add_callback(callback1)
        monitor.add_callback(callback2)

        with patch(
            "aragora.observability.slo.get_slo_status",
            return_value=mock_slo_status_degraded,
        ):
            breaches = await monitor.check_and_alert()

        # Both callbacks should be called for each breach
        assert callback1.call_count == len(breaches)
        assert callback2.call_count == len(breaches)

    @pytest.mark.asyncio
    async def test_sync_callback_supported(
        self, monitor: SLOAlertMonitor, mock_slo_status_degraded: SLOStatus
    ):
        """Synchronous callbacks should work alongside async ones."""
        results = []

        def sync_callback(breach: SLOBreach):
            results.append(breach.slo_name)

        monitor.add_callback(sync_callback)

        with patch(
            "aragora.observability.slo.get_slo_status",
            return_value=mock_slo_status_degraded,
        ):
            await monitor.check_and_alert()

        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_callback_error_does_not_break_flow(
        self, monitor: SLOAlertMonitor, mock_slo_status_degraded: SLOStatus
    ):
        """Callback errors should not prevent other callbacks from running."""
        erroring_callback = AsyncMock(side_effect=Exception("Callback failed"))
        working_callback = AsyncMock()

        monitor.add_callback(erroring_callback)
        monitor.add_callback(working_callback)

        with patch(
            "aragora.observability.slo.get_slo_status",
            return_value=mock_slo_status_degraded,
        ):
            breaches = await monitor.check_and_alert()

        # Despite error in first callback, second should still be called
        assert working_callback.call_count == len(breaches)

    @pytest.mark.asyncio
    async def test_no_callbacks_on_healthy(
        self, monitor: SLOAlertMonitor, mock_slo_status_healthy: SLOStatus
    ):
        """No callbacks should be invoked when SLOs are healthy."""
        callback = AsyncMock()
        monitor.add_callback(callback)

        with patch(
            "aragora.observability.slo.get_slo_status",
            return_value=mock_slo_status_healthy,
        ):
            breaches = await monitor.check_and_alert()

        assert len(breaches) == 0
        assert callback.call_count == 0


# ---------------------------------------------------------------------------
# Tests: Alert cooldown
# ---------------------------------------------------------------------------


class TestAlertCooldown:
    """Test alert cooldown prevents alert spam."""

    @pytest.mark.asyncio
    async def test_cooldown_prevents_repeated_alerts(self, mock_slo_status_degraded: SLOStatus):
        """Same alert should not fire repeatedly within cooldown period."""
        monitor = SLOAlertMonitor(check_interval_seconds=0.1, cooldown_seconds=10.0)
        callback = AsyncMock()
        monitor.add_callback(callback)

        with patch(
            "aragora.observability.slo.get_slo_status",
            return_value=mock_slo_status_degraded,
        ):
            # First check triggers alerts
            breaches1 = await monitor.check_and_alert()
            first_count = callback.call_count

            # Second check within cooldown should not re-trigger
            breaches2 = await monitor.check_and_alert()

        # First run should have triggered alerts
        assert len(breaches1) >= 1
        # Second run should have been skipped due to cooldown
        assert len(breaches2) == 0
        # Callback count should not have increased
        assert callback.call_count == first_count

    @pytest.mark.asyncio
    async def test_cooldown_expires(self, mock_slo_status_degraded: SLOStatus):
        """Alerts should fire again after cooldown expires."""
        monitor = SLOAlertMonitor(check_interval_seconds=0.01, cooldown_seconds=0.05)
        callback = AsyncMock()
        monitor.add_callback(callback)

        with patch(
            "aragora.observability.slo.get_slo_status",
            return_value=mock_slo_status_degraded,
        ):
            # First check
            breaches1 = await monitor.check_and_alert()
            first_count = callback.call_count

            # Wait for cooldown to expire
            await asyncio.sleep(0.1)

            # Second check after cooldown
            breaches2 = await monitor.check_and_alert()

        assert len(breaches1) >= 1
        assert len(breaches2) >= 1
        assert callback.call_count > first_count


# ---------------------------------------------------------------------------
# Tests: Integration with persistence stores
# ---------------------------------------------------------------------------


class TestSLOHistoryIntegration:
    """Test integration with SLO history persistence."""

    @pytest.mark.asyncio
    async def test_slo_history_callback_records_breach(
        self, tmp_path: Path, mock_slo_status_degraded: SLOStatus
    ):
        """slo_history_callback should persist breaches to the store."""
        from aragora.observability.slo_history import (
            get_slo_history_store,
            reset_slo_history_store,
            slo_history_callback,
        )

        reset_slo_history_store()
        store = get_slo_history_store(db_path=str(tmp_path / "slo.db"))

        monitor = SLOAlertMonitor()
        monitor.add_callback(slo_history_callback)

        with patch(
            "aragora.observability.slo.get_slo_status",
            return_value=mock_slo_status_degraded,
        ):
            breaches = await monitor.check_and_alert()

        # Verify breaches were persisted
        records = store.query()
        assert len(records) == len(breaches)

        for record in records:
            assert record.slo_name in ["API Availability", "p99 Latency", "Debate Success Rate"]

        reset_slo_history_store()


class TestIncidentIntegration:
    """Test integration with incident store for auto-creating incidents."""

    @pytest.mark.asyncio
    async def test_incident_created_from_critical_breach(
        self, tmp_path: Path, mock_slo_status_degraded: SLOStatus
    ):
        """Critical SLO breaches should auto-create incidents."""
        from aragora.observability.incident_store import (
            get_incident_store,
            reset_incident_store,
        )

        reset_incident_store()
        store = get_incident_store(db_path=str(tmp_path / "incidents.db"))

        # Create a callback that creates incidents for critical breaches
        async def incident_callback(breach: SLOBreach):
            if breach.severity == "critical":
                store.create_from_slo_violation(
                    slo_name=breach.slo_name,
                    severity=breach.severity,
                    message=breach.message,
                )

        monitor = SLOAlertMonitor()
        monitor.add_callback(incident_callback)

        with patch(
            "aragora.observability.slo.get_slo_status",
            return_value=mock_slo_status_degraded,
        ):
            await monitor.check_and_alert()

        # Verify incident was created
        active = store.get_active_incidents()
        assert len(active) >= 1
        assert any("slo" in i.title.lower() for i in active)

        reset_incident_store()


# ---------------------------------------------------------------------------
# Tests: Background monitoring
# ---------------------------------------------------------------------------


class TestBackgroundMonitoring:
    """Test background SLO monitoring."""

    @pytest.mark.asyncio
    async def test_start_stop_monitoring(self, monitor: SLOAlertMonitor):
        """Background monitoring should start and stop cleanly."""
        assert not monitor._running

        await monitor.start_background_monitoring()
        assert monitor._running
        assert monitor._task is not None

        await monitor.stop_background_monitoring()
        assert not monitor._running
        assert monitor._task is None

    @pytest.mark.asyncio
    async def test_monitoring_invokes_check(self, mock_slo_status_healthy: SLOStatus):
        """Background monitoring should periodically check SLOs."""
        monitor = SLOAlertMonitor(check_interval_seconds=0.05, cooldown_seconds=0.01)
        check_count = [0]

        original_check = monitor.check_and_alert

        async def counting_check():
            check_count[0] += 1
            return await original_check()

        monitor.check_and_alert = counting_check

        with patch(
            "aragora.observability.slo.get_slo_status",
            return_value=mock_slo_status_healthy,
        ):
            await monitor.start_background_monitoring()
            await asyncio.sleep(0.15)
            await monitor.stop_background_monitoring()

        # Should have run at least 2 checks in 150ms with 50ms interval
        assert check_count[0] >= 2


# ---------------------------------------------------------------------------
# Tests: SLO recovery detection
# ---------------------------------------------------------------------------


class TestSLORecovery:
    """Test SLO recovery detection scenarios."""

    @pytest.mark.asyncio
    async def test_recovery_after_breach(
        self,
        monitor: SLOAlertMonitor,
        mock_slo_status_degraded: SLOStatus,
        mock_slo_status_healthy: SLOStatus,
    ):
        """SLO recovery should result in no new alerts."""
        callback = AsyncMock()
        monitor.add_callback(callback)

        # First: degraded state triggers alerts
        with patch(
            "aragora.observability.slo.get_slo_status",
            return_value=mock_slo_status_degraded,
        ):
            breaches1 = await monitor.check_and_alert()

        initial_count = callback.call_count
        assert len(breaches1) >= 1

        # Second: recovered state triggers no new alerts
        with patch(
            "aragora.observability.slo.get_slo_status",
            return_value=mock_slo_status_healthy,
        ):
            breaches2 = await monitor.check_and_alert()

        assert len(breaches2) == 0
        assert callback.call_count == initial_count
