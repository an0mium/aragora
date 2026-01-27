"""
Tests for backup monitoring metrics.

Tests Prometheus metrics recording and SLA compliance tracking.
"""

import time
from unittest.mock import patch, MagicMock

import pytest

from aragora.backup.monitoring import (
    BackupMetrics,
    SLA_TARGETS,
    record_backup_created,
    record_backup_verified,
    record_restore_completed,
    get_backup_age_seconds,
    get_current_metrics,
    check_rpo_breach,
    check_rto_breach,
)


class TestBackupMetricsDataclass:
    """Tests for BackupMetrics dataclass."""

    def test_default_values(self):
        """Test default initialization."""
        metrics = BackupMetrics()
        assert metrics.last_backup_timestamp is None
        assert metrics.backup_age_seconds is None
        assert metrics.last_backup_size_bytes is None
        assert metrics.rpo_compliant == {}
        assert metrics.rto_compliant == {}

    def test_with_values(self):
        """Test initialization with values."""
        now = time.time()
        metrics = BackupMetrics(
            last_backup_timestamp=now,
            backup_age_seconds=60.0,
            last_backup_size_bytes=1024,
            rpo_compliant={"free": True},
            rto_compliant={"free": True},
        )
        assert metrics.last_backup_timestamp == now
        assert metrics.backup_age_seconds == 60.0
        assert metrics.last_backup_size_bytes == 1024


class TestSLATargets:
    """Tests for SLA target configuration."""

    def test_free_tier_targets(self):
        """Free tier has 24h RTO and 24h RPO."""
        assert SLA_TARGETS["free"]["rto"] == 24 * 3600
        assert SLA_TARGETS["free"]["rpo"] == 24 * 3600

    def test_pro_tier_targets(self):
        """Pro tier has 4h RTO and 1h RPO."""
        assert SLA_TARGETS["pro"]["rto"] == 4 * 3600
        assert SLA_TARGETS["pro"]["rpo"] == 1 * 3600

    def test_enterprise_tier_targets(self):
        """Enterprise tier has 1h RTO and 15m RPO."""
        assert SLA_TARGETS["enterprise"]["rto"] == 1 * 3600
        assert SLA_TARGETS["enterprise"]["rpo"] == 15 * 60


class TestRecordBackupCreated:
    """Tests for record_backup_created function."""

    def test_records_timestamp_and_size(self):
        """Recording backup updates internal state."""
        record_backup_created(size_bytes=1024 * 1024, duration_seconds=30.0)

        age = get_backup_age_seconds()
        # Just created, should be < 1 second
        assert age is not None
        assert age < 1.0

    def test_records_backup_type(self):
        """Different backup types are supported."""
        for backup_type in ["full", "incremental", "differential"]:
            record_backup_created(
                size_bytes=1024,
                duration_seconds=1.0,
                backup_type=backup_type,
            )
            # Should not raise


class TestRecordBackupVerified:
    """Tests for record_backup_verified function."""

    def test_record_success(self):
        """Record successful verification."""
        record_backup_verified(success=True)
        # Should not raise

    def test_record_failure_with_type(self):
        """Record failed verification with failure type."""
        record_backup_verified(success=False, failure_type="checksum_mismatch")
        # Should not raise


class TestRecordRestoreCompleted:
    """Tests for record_restore_completed function."""

    def test_record_successful_restore(self):
        """Record successful restore."""
        record_restore_completed(success=True, duration_seconds=120.0)
        # Should not raise

    def test_record_failed_restore(self):
        """Record failed restore."""
        record_restore_completed(
            success=False,
            duration_seconds=60.0,
            failure_type="corruption",
        )
        # Should not raise


class TestGetBackupAge:
    """Tests for get_backup_age_seconds function."""

    def test_returns_none_when_no_backup(self):
        """Returns None when no backup has been created."""
        # Reset internal state by importing fresh
        import aragora.backup.monitoring as m

        m._last_backup_timestamp = None

        age = get_backup_age_seconds()
        assert age is None

    def test_returns_age_after_backup(self):
        """Returns correct age after backup."""
        record_backup_created(size_bytes=1024, duration_seconds=1.0)

        age = get_backup_age_seconds()
        assert age is not None
        assert age >= 0
        assert age < 2.0  # Should be very recent


class TestRPOBreach:
    """Tests for RPO breach checking."""

    def test_no_breach_when_backup_fresh(self):
        """Fresh backup has no breach for all tiers."""
        record_backup_created(size_bytes=1024, duration_seconds=1.0)

        # No breach means compliant
        assert check_rpo_breach("free") is False
        assert check_rpo_breach("pro") is False
        assert check_rpo_breach("enterprise") is False

    def test_breach_when_backup_stale(self):
        """Stale backup may breach RPO for stricter tiers."""
        import aragora.backup.monitoring as m

        # Simulate backup from 2 hours ago
        m._last_backup_timestamp = time.time() - (2 * 3600)

        # Free tier (24h RPO) should not breach
        assert check_rpo_breach("free") is False

        # Pro tier (1h RPO) should breach
        assert check_rpo_breach("pro") is True

        # Enterprise tier (15m RPO) should breach
        assert check_rpo_breach("enterprise") is True


class TestRTOBreach:
    """Tests for RTO breach checking."""

    def test_no_breach_with_fast_restore(self):
        """Fast restore has no breach."""
        # Record a quick restore (10 minutes)
        record_restore_completed(success=True, duration_seconds=600)

        # No breach means compliant
        assert check_rto_breach("free") is False
        assert check_rto_breach("pro") is False
        assert check_rto_breach("enterprise") is False

    def test_breach_with_slow_restore(self):
        """Slow restore may breach RTO for stricter tiers."""
        import aragora.backup.monitoring as m

        # Simulate a 2-hour restore
        m._last_restore_duration = 2 * 3600

        # Free tier (24h RTO) should not breach
        assert check_rto_breach("free") is False

        # Pro tier (4h RTO) should not breach
        assert check_rto_breach("pro") is False

        # Enterprise tier (1h RTO) should breach
        assert check_rto_breach("enterprise") is True


class TestGetCurrentMetrics:
    """Tests for get_current_metrics function."""

    def test_returns_metrics_object(self):
        """Returns BackupMetrics instance."""
        record_backup_created(size_bytes=2048, duration_seconds=5.0)

        metrics = get_current_metrics()

        assert isinstance(metrics, BackupMetrics)
        assert metrics.last_backup_size_bytes == 2048
        assert metrics.backup_age_seconds is not None
        assert metrics.backup_age_seconds < 2.0
