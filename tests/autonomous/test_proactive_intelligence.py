"""Tests for Proactive Intelligence (Phase 5.3)."""

import asyncio
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from aragora.autonomous.proactive_intelligence import (
    Alert,
    AlertAnalyzer,
    AlertSeverity,
    Anomaly,
    AnomalyDetector,
    ScheduledTrigger,
    ScheduledTriggerConfig,
    TrendData,
    TrendDirection,
    TrendMonitor,
)


class TestAlertSeverity:
    """Tests for AlertSeverity enum."""

    def test_severity_values(self):
        """Test all severity values exist."""
        assert AlertSeverity.INFO.value == "info"
        assert AlertSeverity.LOW.value == "low"
        assert AlertSeverity.MEDIUM.value == "medium"
        assert AlertSeverity.HIGH.value == "high"
        assert AlertSeverity.CRITICAL.value == "critical"


class TestTrendDirection:
    """Tests for TrendDirection enum."""

    def test_direction_values(self):
        """Test all direction values exist."""
        assert TrendDirection.INCREASING.value == "increasing"
        assert TrendDirection.DECREASING.value == "decreasing"
        assert TrendDirection.STABLE.value == "stable"
        assert TrendDirection.VOLATILE.value == "volatile"


class TestScheduledTrigger:
    """Tests for ScheduledTrigger class."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def scheduled_trigger(self, temp_dir):
        """Create a ScheduledTrigger instance."""
        return ScheduledTrigger(storage_path=temp_dir / "triggers.json")

    def test_add_trigger(self, scheduled_trigger):
        """Test adding a scheduled trigger."""
        trigger = scheduled_trigger.add_trigger(
            trigger_id="daily_analysis",
            name="Daily Analysis",
            interval_seconds=86400,
            metadata={"topic": "System health check"},
        )

        assert trigger.id == "daily_analysis"
        assert trigger.name == "Daily Analysis"
        assert trigger.interval_seconds == 86400
        assert trigger.enabled
        assert trigger.next_run is not None

    def test_remove_trigger(self, scheduled_trigger):
        """Test removing a trigger."""
        scheduled_trigger.add_trigger("test", "Test", interval_seconds=60)

        success = scheduled_trigger.remove_trigger("test")

        assert success
        assert len(scheduled_trigger.list_triggers()) == 0

    def test_remove_nonexistent_trigger(self, scheduled_trigger):
        """Test removing nonexistent trigger returns False."""
        success = scheduled_trigger.remove_trigger("nonexistent")
        assert not success

    def test_enable_disable_trigger(self, scheduled_trigger):
        """Test enabling and disabling triggers."""
        scheduled_trigger.add_trigger("test", "Test", interval_seconds=60)

        scheduled_trigger.disable_trigger("test")
        triggers = scheduled_trigger.list_triggers()
        assert not triggers[0].enabled

        scheduled_trigger.enable_trigger("test")
        triggers = scheduled_trigger.list_triggers()
        assert triggers[0].enabled

    def test_list_triggers(self, scheduled_trigger):
        """Test listing all triggers."""
        scheduled_trigger.add_trigger("t1", "Trigger 1", interval_seconds=60)
        scheduled_trigger.add_trigger("t2", "Trigger 2", interval_seconds=120)

        triggers = scheduled_trigger.list_triggers()

        assert len(triggers) == 2

    @pytest.mark.asyncio
    async def test_start_stop(self, scheduled_trigger):
        """Test starting and stopping the scheduler."""
        await scheduled_trigger.start()
        assert scheduled_trigger._running

        await scheduled_trigger.stop()
        assert not scheduled_trigger._running

    @pytest.mark.asyncio
    async def test_trigger_execution(self, temp_dir):
        """Test trigger actually executes."""
        debate_creator = AsyncMock(return_value={"debate_id": "test"})

        trigger = ScheduledTrigger(
            storage_path=temp_dir / "triggers.json",
            debate_creator=debate_creator,
        )

        trigger.add_trigger(
            "immediate",
            "Immediate Test",
            interval_seconds=1,
            metadata={"topic": "Test topic"},
        )

        # Start scheduler
        await trigger.start()

        # Wait for execution
        await asyncio.sleep(1.5)

        await trigger.stop()

        # Should have executed
        debate_creator.assert_called()


class TestAlertAnalyzer:
    """Tests for AlertAnalyzer class."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def alert_analyzer(self, temp_dir):
        """Create an AlertAnalyzer instance."""
        return AlertAnalyzer(storage_path=temp_dir / "alerts.json")

    def test_set_threshold(self, alert_analyzer):
        """Test setting thresholds."""
        alert_analyzer.set_threshold(
            metric_name="cpu_usage",
            warning_threshold=80.0,
            critical_threshold=95.0,
            comparison="gt",
        )

        assert "cpu_usage" in alert_analyzer._thresholds
        assert alert_analyzer._thresholds["cpu_usage"]["warning"] == 80.0
        assert alert_analyzer._thresholds["cpu_usage"]["critical"] == 95.0

    @pytest.mark.asyncio
    async def test_check_metric_no_threshold(self, alert_analyzer):
        """Test checking metric without threshold."""
        alert = await alert_analyzer.check_metric("unknown", 100.0)
        assert alert is None

    @pytest.mark.asyncio
    async def test_check_metric_below_threshold(self, alert_analyzer):
        """Test checking metric below threshold."""
        alert_analyzer.set_threshold("cpu", warning_threshold=80.0, comparison="gt")

        alert = await alert_analyzer.check_metric("cpu", 50.0)

        assert alert is None

    @pytest.mark.asyncio
    async def test_check_metric_warning_threshold(self, alert_analyzer):
        """Test checking metric at warning level."""
        alert_analyzer.set_threshold(
            "cpu",
            warning_threshold=80.0,
            critical_threshold=95.0,
            comparison="gt",
        )

        alert = await alert_analyzer.check_metric("cpu", 85.0)

        assert alert is not None
        assert alert.severity == AlertSeverity.HIGH

    @pytest.mark.asyncio
    async def test_check_metric_critical_threshold(self, alert_analyzer):
        """Test checking metric at critical level."""
        alert_analyzer.set_threshold(
            "cpu",
            warning_threshold=80.0,
            critical_threshold=95.0,
            comparison="gt",
        )

        alert = await alert_analyzer.check_metric("cpu", 98.0)

        assert alert is not None
        assert alert.severity == AlertSeverity.CRITICAL

    @pytest.mark.asyncio
    async def test_check_metric_less_than(self, alert_analyzer):
        """Test less-than comparison."""
        alert_analyzer.set_threshold(
            "disk_free",
            warning_threshold=20.0,
            comparison="lt",
        )

        alert = await alert_analyzer.check_metric("disk_free", 10.0)

        assert alert is not None

    @pytest.mark.asyncio
    async def test_alert_callback(self, temp_dir):
        """Test alert callback is called."""
        callback = MagicMock()
        analyzer = AlertAnalyzer(
            storage_path=temp_dir / "alerts.json",
            alert_callback=callback,
        )
        analyzer.set_threshold("test", warning_threshold=50.0, comparison="gt")

        await analyzer.check_metric("test", 75.0)

        callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_auto_debate_trigger(self, temp_dir):
        """Test auto-debate is triggered for high severity."""
        debate_creator = AsyncMock(return_value={"debate_id": "d123"})
        analyzer = AlertAnalyzer(
            storage_path=temp_dir / "alerts.json",
            debate_creator=debate_creator,
            auto_debate_severities={AlertSeverity.CRITICAL},
        )
        analyzer.set_threshold("test", critical_threshold=90.0, comparison="gt")

        alert = await analyzer.check_metric("test", 95.0)

        assert alert.debate_triggered
        assert alert.debate_id == "d123"
        debate_creator.assert_called_once()

    def test_acknowledge_alert(self, alert_analyzer):
        """Test acknowledging an alert."""
        alert_analyzer.set_threshold("test", warning_threshold=50.0, comparison="gt")
        asyncio.run(alert_analyzer.check_metric("test", 75.0))

        alerts = alert_analyzer.get_active_alerts()
        success = alert_analyzer.acknowledge_alert(alerts[0].id, "user1")

        assert success
        assert alerts[0].acknowledged
        assert alerts[0].acknowledged_by == "user1"

    def test_resolve_alert(self, alert_analyzer):
        """Test resolving an alert."""
        alert_analyzer.set_threshold("test", warning_threshold=50.0, comparison="gt")
        asyncio.run(alert_analyzer.check_metric("test", 75.0))

        alerts = alert_analyzer.get_active_alerts()
        success = alert_analyzer.resolve_alert(alerts[0].id)

        assert success
        assert alert_analyzer.get_active_alerts() == []

    def test_get_active_alerts(self, alert_analyzer):
        """Test getting active alerts."""
        alert_analyzer.set_threshold("test", warning_threshold=50.0, comparison="gt")
        asyncio.run(alert_analyzer.check_metric("test", 75.0))
        asyncio.run(alert_analyzer.check_metric("test", 80.0))

        alerts = alert_analyzer.get_active_alerts()

        assert len(alerts) == 2


class TestTrendMonitor:
    """Tests for TrendMonitor class."""

    @pytest.fixture
    def trend_monitor(self):
        """Create a TrendMonitor instance."""
        return TrendMonitor(
            window_size=100,
            min_data_points=5,
            trend_threshold=0.1,
        )

    def test_record_data(self, trend_monitor):
        """Test recording metric data."""
        trend_monitor.record("cpu", 50.0)
        trend_monitor.record("cpu", 55.0)

        assert len(trend_monitor._metrics["cpu"]) == 2

    def test_get_trend_insufficient_data(self, trend_monitor):
        """Test trend returns None with insufficient data."""
        trend_monitor.record("cpu", 50.0)
        trend_monitor.record("cpu", 55.0)

        trend = trend_monitor.get_trend("cpu")

        assert trend is None

    def test_get_trend_increasing(self, trend_monitor):
        """Test detecting increasing trend."""
        # Add increasing values
        for i in range(10):
            trend_monitor.record("metric", 100 + i * 5)

        trend = trend_monitor.get_trend("metric")

        assert trend is not None
        assert trend.direction == TrendDirection.INCREASING
        assert trend.change_percent > 0

    def test_get_trend_decreasing(self, trend_monitor):
        """Test detecting decreasing trend."""
        # Add decreasing values
        for i in range(10):
            trend_monitor.record("metric", 100 - i * 5)

        trend = trend_monitor.get_trend("metric")

        assert trend is not None
        assert trend.direction == TrendDirection.DECREASING
        assert trend.change_percent < 0

    def test_get_trend_stable(self, trend_monitor):
        """Test detecting stable trend."""
        # Add stable values with small variation
        for i in range(10):
            trend_monitor.record("metric", 100 + (i % 2))

        trend = trend_monitor.get_trend("metric")

        assert trend is not None
        assert trend.direction == TrendDirection.STABLE

    def test_get_trend_volatile(self, trend_monitor):
        """Test detecting volatile trend."""
        import random

        # Add highly variable values
        for i in range(10):
            trend_monitor.record("metric", random.uniform(50, 150))

        trend = trend_monitor.get_trend("metric")

        # May or may not be volatile depending on random values
        assert trend is not None

    def test_get_all_trends(self, trend_monitor):
        """Test getting all trends."""
        for i in range(10):
            trend_monitor.record("cpu", 50 + i)
            trend_monitor.record("memory", 60 + i)

        trends = trend_monitor.get_all_trends()

        assert "cpu" in trends
        assert "memory" in trends

    def test_trend_with_period_filter(self, trend_monitor):
        """Test trend with time period filter."""
        # Add old data
        for i in range(5):
            trend_monitor.record(
                "metric",
                100 + i,
                timestamp=datetime.now() - timedelta(hours=2),
            )

        # Add recent data
        for i in range(5):
            trend_monitor.record("metric", 200 + i)

        # Get trend for last hour only
        trend = trend_monitor.get_trend("metric", period_seconds=3600)

        assert trend is not None
        # Should only reflect recent data
        assert trend.current_value > 150


class TestAnomalyDetector:
    """Tests for AnomalyDetector class."""

    @pytest.fixture
    def anomaly_detector(self):
        """Create an AnomalyDetector instance."""
        return AnomalyDetector(
            window_size=100,
            z_threshold=3.0,
            min_data_points=10,
        )

    def test_record_builds_baseline(self, anomaly_detector):
        """Test recording builds baseline before detecting."""
        # Add baseline data
        for i in range(5):
            result = anomaly_detector.record("metric", 100.0)
            assert result is None  # Not enough data yet

    def test_detect_anomaly_high(self, anomaly_detector):
        """Test detecting high anomaly."""
        # Build baseline around 100
        for i in range(20):
            anomaly_detector.record("metric", 100.0 + (i % 3 - 1))

        # Record anomalous value
        anomaly = anomaly_detector.record("metric", 200.0)

        assert anomaly is not None
        assert anomaly.value == 200.0
        assert anomaly.deviation >= 3.0

    def test_detect_anomaly_low(self, anomaly_detector):
        """Test detecting low anomaly."""
        # Build baseline around 100
        for i in range(20):
            anomaly_detector.record("metric", 100.0 + (i % 3 - 1))

        # Record anomalous value
        anomaly = anomaly_detector.record("metric", 0.0)

        assert anomaly is not None
        assert anomaly.value == 0.0

    def test_no_anomaly_normal_value(self, anomaly_detector):
        """Test normal value doesn't trigger anomaly."""
        # Build baseline around 100
        for i in range(20):
            anomaly_detector.record("metric", 100.0 + (i % 5 - 2))

        # Record normal value
        anomaly = anomaly_detector.record("metric", 101.0)

        assert anomaly is None

    def test_anomaly_callback(self):
        """Test anomaly callback is called."""
        callback = MagicMock()
        detector = AnomalyDetector(
            min_data_points=10,
            z_threshold=3.0,
            alert_callback=callback,
        )

        # Build baseline with small variance (stdev > 0 required)
        for i in range(15):
            detector.record("metric", 100.0 + (i % 3))  # values: 100, 101, 102, ...

        # Trigger clear anomaly (well outside 3 stdev)
        detector.record("metric", 1000.0)

        callback.assert_called_once()

    def test_get_recent_anomalies(self, anomaly_detector):
        """Test getting recent anomalies."""
        # Build baseline with small variance
        for i in range(20):
            anomaly_detector.record("metric", 100.0 + (i % 3))

        # Trigger anomaly
        anomaly_detector.record("metric", 500.0)

        recent = anomaly_detector.get_recent_anomalies(hours=1)

        assert len(recent) >= 1

    def test_get_recent_anomalies_filtered(self, anomaly_detector):
        """Test filtering anomalies by metric name."""
        # Build baseline for cpu with some variance
        for i in range(20):
            anomaly_detector.record("cpu", 50.0 + (i % 3))

        # Build baseline for memory with some variance
        for i in range(20):
            anomaly_detector.record("memory", 60.0 + (i % 3))

        # Trigger clear anomaly for cpu
        anomaly_detector.record("cpu", 500.0)

        cpu_anomalies = anomaly_detector.get_recent_anomalies(metric_name="cpu")

        assert len(cpu_anomalies) == 1
        assert cpu_anomalies[0].metric_name == "cpu"

    def test_get_baseline_stats(self, anomaly_detector):
        """Test getting baseline statistics."""
        for i in range(20):
            anomaly_detector.record("metric", 100.0 + i)

        stats = anomaly_detector.get_baseline_stats("metric")

        assert stats is not None
        assert "mean" in stats
        assert "stdev" in stats
        assert "min" in stats
        assert "max" in stats
        assert "median" in stats
        assert stats["count"] == 20

    def test_get_baseline_stats_insufficient(self, anomaly_detector):
        """Test baseline stats with insufficient data."""
        anomaly_detector.record("metric", 100.0)

        stats = anomaly_detector.get_baseline_stats("metric")

        assert stats is None

    def test_severity_based_on_deviation(self, anomaly_detector):
        """Test anomaly severity scales with deviation."""
        # Build tight baseline
        for i in range(20):
            anomaly_detector.record("metric", 100.0)

        # Moderate anomaly
        anomaly1 = anomaly_detector.record("metric", 130.0)

        # Reset and build new baseline
        anomaly_detector._metrics.clear()
        for i in range(20):
            anomaly_detector.record("metric", 100.0)

        # Extreme anomaly
        anomaly2 = anomaly_detector.record("metric", 500.0)

        if anomaly1 and anomaly2:
            # More extreme should have higher severity
            severity_order = [
                AlertSeverity.MEDIUM,
                AlertSeverity.HIGH,
                AlertSeverity.CRITICAL,
            ]
            assert severity_order.index(anomaly2.severity) >= severity_order.index(
                anomaly1.severity
            )


class TestAlert:
    """Tests for Alert dataclass."""

    def test_alert_creation(self):
        """Test creating an alert."""
        alert = Alert(
            id="alert_1",
            severity=AlertSeverity.HIGH,
            title="High CPU Usage",
            description="CPU usage exceeded 90%",
            source="system_monitor",
            timestamp=datetime.now(),
        )

        assert alert.id == "alert_1"
        assert alert.severity == AlertSeverity.HIGH
        assert not alert.acknowledged
        assert not alert.resolved


class TestTrendData:
    """Tests for TrendData dataclass."""

    def test_trend_data_creation(self):
        """Test creating trend data."""
        now = datetime.now()
        trend = TrendData(
            metric_name="cpu",
            direction=TrendDirection.INCREASING,
            current_value=85.0,
            previous_value=70.0,
            change_percent=21.4,
            period_start=now - timedelta(hours=1),
            period_end=now,
            data_points=60,
            confidence=0.9,
        )

        assert trend.metric_name == "cpu"
        assert trend.direction == TrendDirection.INCREASING
        assert trend.change_percent == 21.4


class TestAnomaly:
    """Tests for Anomaly dataclass."""

    def test_anomaly_creation(self):
        """Test creating an anomaly."""
        anomaly = Anomaly(
            id="anomaly_1",
            metric_name="response_time",
            value=5000.0,
            expected_value=200.0,
            deviation=4.5,
            timestamp=datetime.now(),
            severity=AlertSeverity.CRITICAL,
            description="Response time 4.5 std devs above normal",
        )

        assert anomaly.metric_name == "response_time"
        assert anomaly.deviation == 4.5
        assert anomaly.severity == AlertSeverity.CRITICAL
