"""
Tests for replication health monitoring and recovery progress monitoring.

Tests cover:
- Replication lag monitoring
- Alert thresholds and callbacks
- Health check endpoints
- Recovery progress tracking
- Rate estimation
- Prometheus metrics integration
"""

import time
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from aragora.backup.replication_monitor import (
    ReplicationConfig,
    ReplicationHealth,
    ReplicationHealthMonitor,
    ReplicationMetrics,
    ReplicationStatus,
    create_replication_monitor,
    get_replication_monitor,
    set_replication_monitor,
)
from aragora.backup.monitoring import (
    RecoveryPhase,
    RecoveryProgress,
    RecoveryProgressMonitor,
    get_recovery_monitor,
    record_recovery_completed,
    record_recovery_progress,
    set_recovery_monitor,
)


class TestReplicationStatus:
    """Tests for ReplicationStatus enum."""

    def test_status_values(self):
        """Test all expected status values exist."""
        assert ReplicationStatus.HEALTHY.value == "healthy"
        assert ReplicationStatus.DEGRADED.value == "degraded"
        assert ReplicationStatus.CRITICAL.value == "critical"
        assert ReplicationStatus.UNKNOWN.value == "unknown"
        assert ReplicationStatus.STOPPED.value == "stopped"


class TestReplicationConfig:
    """Tests for ReplicationConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ReplicationConfig()
        assert config.warning_lag_seconds == 10.0
        assert config.critical_lag_seconds == 60.0
        assert config.max_time_since_replication_seconds == 300.0
        assert config.check_interval_seconds == 10.0
        assert config.enable_alerts is True
        assert config.alert_cooldown_seconds == 300.0

    def test_custom_values(self):
        """Test custom configuration values."""
        config = ReplicationConfig(
            warning_lag_seconds=5.0,
            critical_lag_seconds=30.0,
            max_time_since_replication_seconds=60.0,
            enable_alerts=False,
        )
        assert config.warning_lag_seconds == 5.0
        assert config.critical_lag_seconds == 30.0
        assert config.max_time_since_replication_seconds == 60.0
        assert config.enable_alerts is False


class TestReplicationHealth:
    """Tests for ReplicationHealth dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        health = ReplicationHealth(
            status=ReplicationStatus.HEALTHY,
            healthy=True,
            lag_seconds=0.5,
            standby_connected=True,
            primary_host="primary",
            standby_host="standby",
        )
        data = health.to_dict()

        assert data["status"] == "healthy"
        assert data["healthy"] is True
        assert data["lag_seconds"] == 0.5
        assert data["standby_connected"] is True
        assert data["primary_host"] == "primary"
        assert data["standby_host"] == "standby"

    def test_to_dict_with_none_values(self):
        """Test conversion with None values."""
        health = ReplicationHealth(
            status=ReplicationStatus.UNKNOWN,
            healthy=False,
        )
        data = health.to_dict()

        assert data["status"] == "unknown"
        assert data["healthy"] is False
        assert data["lag_seconds"] is None
        assert data["last_replication_at"] is None


class TestReplicationHealthMonitor:
    """Tests for ReplicationHealthMonitor class."""

    def test_initialization(self):
        """Test monitor initialization."""
        monitor = ReplicationHealthMonitor(
            primary_host="primary-db",
            standby_host="standby-db",
        )
        assert monitor.primary_host == "primary-db"
        assert monitor.standby_host == "standby-db"
        assert monitor.config is not None

    def test_record_replication_healthy(self):
        """Test recording a healthy replication event."""
        monitor = ReplicationHealthMonitor()
        monitor.record_replication(lag_seconds=0.5)

        health = monitor.get_health()
        assert health.status == ReplicationStatus.HEALTHY
        assert health.healthy is True
        assert health.lag_seconds == 0.5
        assert health.standby_connected is True

    def test_record_replication_degraded(self):
        """Test recording a degraded replication event."""
        config = ReplicationConfig(warning_lag_seconds=5.0, critical_lag_seconds=30.0)
        monitor = ReplicationHealthMonitor(config=config)

        # Lag above warning but below critical
        monitor.record_replication(lag_seconds=10.0)

        health = monitor.get_health()
        assert health.status == ReplicationStatus.DEGRADED
        assert health.healthy is False

    def test_record_replication_critical(self):
        """Test recording a critical replication event."""
        config = ReplicationConfig(warning_lag_seconds=5.0, critical_lag_seconds=30.0)
        monitor = ReplicationHealthMonitor(config=config)

        # Lag above critical threshold
        monitor.record_replication(lag_seconds=60.0)

        health = monitor.get_health()
        assert health.status == ReplicationStatus.CRITICAL
        assert health.healthy is False

    def test_record_replication_failure(self):
        """Test recording a failed replication."""
        monitor = ReplicationHealthMonitor()
        monitor.record_replication(lag_seconds=0.0, success=False, error_type="network_error")

        metrics = monitor.get_metrics()
        assert metrics.failed_replications == 1

    def test_standby_disconnect(self):
        """Test recording standby disconnect."""
        monitor = ReplicationHealthMonitor()
        monitor.record_replication(lag_seconds=0.5)
        monitor.record_standby_disconnect()

        health = monitor.get_health()
        assert health.status == ReplicationStatus.STOPPED
        assert health.standby_connected is False

    def test_standby_reconnect(self):
        """Test recording standby reconnect."""
        monitor = ReplicationHealthMonitor()
        monitor.record_standby_disconnect()
        monitor.record_standby_connect()

        health = monitor.get_health()
        assert health.standby_connected is True

    def test_alert_callback(self):
        """Test alert callback is invoked."""
        callback = MagicMock()
        config = ReplicationConfig(
            warning_lag_seconds=5.0,
            critical_lag_seconds=30.0,
            alert_cooldown_seconds=0,  # Disable cooldown for test
        )
        monitor = ReplicationHealthMonitor(
            config=config,
            alert_callback=callback,
        )

        # Trigger critical alert
        monitor.record_replication(lag_seconds=60.0)

        callback.assert_called_once()
        alert_type, health = callback.call_args[0]
        assert alert_type == "critical_lag"
        assert health.status == ReplicationStatus.CRITICAL

    def test_alert_cooldown(self):
        """Test that alerts respect cooldown period."""
        callback = MagicMock()
        config = ReplicationConfig(
            critical_lag_seconds=30.0,
            alert_cooldown_seconds=300.0,  # 5 minute cooldown
        )
        monitor = ReplicationHealthMonitor(
            config=config,
            alert_callback=callback,
        )

        # First critical event should trigger alert
        monitor.record_replication(lag_seconds=60.0)
        assert callback.call_count == 1

        # Second critical event within cooldown should not trigger
        monitor.record_replication(lag_seconds=70.0)
        assert callback.call_count == 1  # Still 1

    def test_get_metrics(self):
        """Test getting metrics snapshot."""
        monitor = ReplicationHealthMonitor()
        monitor.record_replication(lag_seconds=1.0)
        monitor.record_replication(lag_seconds=2.0)
        monitor.record_replication(lag_seconds=3.0)

        metrics = monitor.get_metrics()
        assert metrics.lag_seconds == 3.0
        assert metrics.total_replications == 3
        assert metrics.failed_replications == 0
        assert metrics.average_lag_seconds is not None

    def test_health_check_endpoint(self):
        """Test health check endpoint format."""
        monitor = ReplicationHealthMonitor()
        monitor.record_replication(lag_seconds=0.5)

        result = monitor.health_check()

        assert result["service"] == "replication_monitor"
        assert result["status"] == "healthy"
        assert "details" in result
        assert result["details"]["lag_seconds"] == 0.5

    def test_health_check_unhealthy(self):
        """Test health check with unhealthy status."""
        config = ReplicationConfig(critical_lag_seconds=30.0)
        monitor = ReplicationHealthMonitor(config=config)
        monitor.record_replication(lag_seconds=60.0)

        result = monitor.health_check()

        assert result["status"] == "unhealthy"


class TestReplicationMonitorGlobal:
    """Tests for global replication monitor functions."""

    def test_get_replication_monitor_creates_instance(self):
        """Test that get_replication_monitor creates instance if none exists."""
        set_replication_monitor(None)  # Reset
        monitor = get_replication_monitor()
        assert monitor is not None
        assert isinstance(monitor, ReplicationHealthMonitor)

    def test_set_replication_monitor(self):
        """Test setting the global monitor."""
        custom_monitor = ReplicationHealthMonitor(primary_host="custom")
        set_replication_monitor(custom_monitor)

        monitor = get_replication_monitor()
        assert monitor.primary_host == "custom"

        # Cleanup
        set_replication_monitor(None)

    def test_create_replication_monitor(self):
        """Test create_replication_monitor factory function."""
        config = ReplicationConfig(warning_lag_seconds=20.0)
        monitor = create_replication_monitor(
            config=config,
            primary_host="prod-primary",
            standby_host="prod-standby",
        )

        assert monitor.primary_host == "prod-primary"
        assert monitor.standby_host == "prod-standby"
        assert monitor.config.warning_lag_seconds == 20.0

        # Cleanup
        set_replication_monitor(None)


class TestRecoveryPhase:
    """Tests for RecoveryPhase enum."""

    def test_phase_values(self):
        """Test all expected phase values exist."""
        assert RecoveryPhase.INITIALIZING.value == "initializing"
        assert RecoveryPhase.DOWNLOADING.value == "downloading"
        assert RecoveryPhase.DECOMPRESSING.value == "decompressing"
        assert RecoveryPhase.VERIFYING.value == "verifying"
        assert RecoveryPhase.RESTORING.value == "restoring"
        assert RecoveryPhase.VALIDATING.value == "validating"
        assert RecoveryPhase.COMPLETED.value == "completed"
        assert RecoveryPhase.FAILED.value == "failed"


class TestRecoveryProgress:
    """Tests for RecoveryProgress dataclass."""

    def test_time_remaining_calculation(self):
        """Test time remaining calculation."""
        progress = RecoveryProgress(
            recovery_id="test-1",
            backup_id="backup-1",
            phase=RecoveryPhase.RESTORING,
            bytes_processed=500,
            total_bytes=1000,
            rate_bytes_per_sec=100.0,
        )

        time_remaining = progress.time_remaining_seconds()
        assert time_remaining == 5.0  # 500 remaining / 100 rate

    def test_time_remaining_no_rate(self):
        """Test time remaining with zero rate."""
        progress = RecoveryProgress(
            recovery_id="test-1",
            backup_id="backup-1",
            phase=RecoveryPhase.RESTORING,
            rate_bytes_per_sec=0.0,
        )

        assert progress.time_remaining_seconds() is None

    def test_to_dict(self):
        """Test conversion to dictionary."""
        progress = RecoveryProgress(
            recovery_id="test-1",
            backup_id="backup-1",
            phase=RecoveryPhase.RESTORING,
            percent_complete=50.0,
            bytes_processed=500,
            total_bytes=1000,
            rate_bytes_per_sec=100.0,
        )

        data = progress.to_dict()

        assert data["recovery_id"] == "test-1"
        assert data["backup_id"] == "backup-1"
        assert data["phase"] == "restoring"
        assert data["percent_complete"] == 50.0
        assert data["bytes_processed"] == 500
        assert data["total_bytes"] == 1000
        assert data["time_remaining_seconds"] == 5.0


class TestRecoveryProgressMonitor:
    """Tests for RecoveryProgressMonitor class."""

    def test_start_recovery(self):
        """Test starting a recovery operation."""
        monitor = RecoveryProgressMonitor()
        progress = monitor.start_recovery(
            recovery_id="test-1",
            backup_id="backup-1",
            total_bytes=1000,
        )

        assert progress.recovery_id == "test-1"
        assert progress.backup_id == "backup-1"
        assert progress.phase == RecoveryPhase.INITIALIZING
        assert progress.total_bytes == 1000
        assert progress.started_at is not None

    def test_update_progress(self):
        """Test updating recovery progress."""
        monitor = RecoveryProgressMonitor()
        monitor.start_recovery(
            recovery_id="test-1",
            backup_id="backup-1",
            total_bytes=1000,
        )

        # Allow some time for rate calculation
        time.sleep(0.01)

        progress = monitor.update_progress(
            recovery_id="test-1",
            phase=RecoveryPhase.RESTORING,
            bytes_processed=500,
        )

        assert progress is not None
        assert progress.phase == RecoveryPhase.RESTORING
        assert progress.bytes_processed == 500
        assert progress.percent_complete == 50.0
        assert progress.rate_bytes_per_sec > 0

    def test_update_nonexistent_recovery(self):
        """Test updating a non-existent recovery."""
        monitor = RecoveryProgressMonitor()
        result = monitor.update_progress(
            recovery_id="nonexistent",
            bytes_processed=100,
        )
        assert result is None

    def test_complete_recovery_success(self):
        """Test completing a recovery successfully."""
        monitor = RecoveryProgressMonitor()
        monitor.start_recovery(
            recovery_id="test-1",
            backup_id="backup-1",
        )

        progress = monitor.complete_recovery("test-1", success=True)

        assert progress is not None
        assert progress.phase == RecoveryPhase.COMPLETED
        assert progress.percent_complete == 100.0

    def test_complete_recovery_failure(self):
        """Test completing a recovery with failure."""
        monitor = RecoveryProgressMonitor()
        monitor.start_recovery(
            recovery_id="test-1",
            backup_id="backup-1",
        )

        progress = monitor.complete_recovery(
            "test-1",
            success=False,
            error="Disk space exhausted",
        )

        assert progress is not None
        assert progress.phase == RecoveryPhase.FAILED
        assert progress.error == "Disk space exhausted"

    def test_get_active_recoveries(self):
        """Test getting all active recoveries."""
        monitor = RecoveryProgressMonitor()
        monitor.start_recovery(recovery_id="test-1", backup_id="backup-1")
        monitor.start_recovery(recovery_id="test-2", backup_id="backup-2")

        active = monitor.get_active_recoveries()
        assert len(active) == 2

    def test_get_recovery_history(self):
        """Test getting completed recovery history."""
        monitor = RecoveryProgressMonitor()
        monitor.start_recovery(recovery_id="test-1", backup_id="backup-1")
        monitor.complete_recovery("test-1", success=True)

        history = monitor.get_recovery_history()
        assert len(history) == 1
        assert history[0].recovery_id == "test-1"

    def test_progress_callback(self):
        """Test progress callback is invoked."""
        callback = MagicMock()
        monitor = RecoveryProgressMonitor(progress_callback=callback)

        monitor.start_recovery(recovery_id="test-1", backup_id="backup-1")
        assert callback.call_count == 1

        monitor.update_progress(recovery_id="test-1", bytes_processed=100)
        assert callback.call_count == 2

        monitor.complete_recovery("test-1")
        assert callback.call_count == 3

    def test_health_check(self):
        """Test health check endpoint format."""
        monitor = RecoveryProgressMonitor()
        monitor.start_recovery(recovery_id="test-1", backup_id="backup-1")

        result = monitor.health_check()

        assert result["service"] == "recovery_monitor"
        assert result["status"] == "healthy"
        assert result["active_recoveries"] == 1
        assert len(result["recoveries"]) == 1


class TestRecoveryMonitorGlobal:
    """Tests for global recovery monitor functions."""

    def test_get_recovery_monitor_creates_instance(self):
        """Test that get_recovery_monitor creates instance if none exists."""
        set_recovery_monitor(None)  # Reset
        monitor = get_recovery_monitor()
        assert monitor is not None
        assert isinstance(monitor, RecoveryProgressMonitor)

    def test_record_recovery_progress_starts_new(self):
        """Test record_recovery_progress starts new recovery if needed."""
        set_recovery_monitor(None)  # Reset

        progress = record_recovery_progress(
            recovery_id="test-1",
            backup_id="backup-1",
            total_bytes=1000,
        )

        assert progress is not None
        assert progress.recovery_id == "test-1"

        # Cleanup
        set_recovery_monitor(None)

    def test_record_recovery_progress_updates_existing(self):
        """Test record_recovery_progress updates existing recovery."""
        set_recovery_monitor(None)  # Reset

        # Start recovery
        record_recovery_progress(
            recovery_id="test-1",
            backup_id="backup-1",
            total_bytes=1000,
        )

        # Update progress
        time.sleep(0.01)
        progress = record_recovery_progress(
            recovery_id="test-1",
            backup_id="backup-1",
            phase=RecoveryPhase.RESTORING,
            bytes_processed=500,
        )

        assert progress is not None
        assert progress.phase == RecoveryPhase.RESTORING
        assert progress.bytes_processed == 500

        # Cleanup
        set_recovery_monitor(None)

    def test_record_recovery_completed(self):
        """Test record_recovery_completed function."""
        set_recovery_monitor(None)  # Reset

        record_recovery_progress(
            recovery_id="test-1",
            backup_id="backup-1",
        )

        progress = record_recovery_completed("test-1", success=True)

        assert progress is not None
        assert progress.phase == RecoveryPhase.COMPLETED

        # Cleanup
        set_recovery_monitor(None)


class TestRecoveryProgressRateEstimation:
    """Tests for rate estimation in recovery progress."""

    def test_bytes_rate_calculation(self):
        """Test bytes per second rate calculation."""
        monitor = RecoveryProgressMonitor()
        monitor.start_recovery(
            recovery_id="test-1",
            backup_id="backup-1",
            total_bytes=10000,
        )

        # Simulate time passing
        time.sleep(0.1)  # 100ms

        progress = monitor.update_progress(
            recovery_id="test-1",
            bytes_processed=1000,
        )

        # Rate should be approximately 1000 bytes / 0.1 seconds = 10000 bytes/sec
        # Allow some tolerance for timing variations
        assert progress is not None
        assert progress.rate_bytes_per_sec > 0
        assert 5000 < progress.rate_bytes_per_sec < 20000

    def test_objects_rate_calculation(self):
        """Test objects per second rate calculation."""
        monitor = RecoveryProgressMonitor()
        monitor.start_recovery(
            recovery_id="test-1",
            backup_id="backup-1",
            total_objects=100,
        )

        # Simulate time passing
        time.sleep(0.1)

        progress = monitor.update_progress(
            recovery_id="test-1",
            objects_processed=10,
        )

        assert progress is not None
        assert progress.rate_objects_per_sec > 0

    def test_estimated_completion_time(self):
        """Test estimated completion time calculation."""
        monitor = RecoveryProgressMonitor()
        monitor.start_recovery(
            recovery_id="test-1",
            backup_id="backup-1",
            total_bytes=10000,
        )

        time.sleep(0.05)

        progress = monitor.update_progress(
            recovery_id="test-1",
            bytes_processed=5000,
        )

        assert progress is not None
        if progress.estimated_completion:
            # Should be in the future
            assert progress.estimated_completion > datetime.now(timezone.utc)


class TestRecoveryProgressPhaseTransitions:
    """Tests for phase transitions in recovery progress."""

    def test_phase_transitions(self):
        """Test typical phase transition sequence."""
        monitor = RecoveryProgressMonitor()
        monitor.start_recovery(recovery_id="test-1", backup_id="backup-1")

        phases = [
            RecoveryPhase.DOWNLOADING,
            RecoveryPhase.DECOMPRESSING,
            RecoveryPhase.VERIFYING,
            RecoveryPhase.RESTORING,
            RecoveryPhase.VALIDATING,
        ]

        for phase in phases:
            progress = monitor.update_progress(recovery_id="test-1", phase=phase)
            assert progress is not None
            assert progress.phase == phase

    def test_error_sets_failed_phase(self):
        """Test that recording an error sets phase to FAILED."""
        monitor = RecoveryProgressMonitor()
        monitor.start_recovery(recovery_id="test-1", backup_id="backup-1")

        progress = monitor.update_progress(
            recovery_id="test-1",
            error="Connection lost",
        )

        assert progress is not None
        assert progress.phase == RecoveryPhase.FAILED
        assert progress.error == "Connection lost"
