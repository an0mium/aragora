"""
Tests for the key rotation scheduler and operations.

Tests cover:
- KeyRotationConfig dataclass
- KeyRotationResult dataclass
- RotationSchedule dataclass
- KeyRotationScheduler lifecycle
- Manual rotation
- Rotation scheduling
- Re-encryption operations
- Alert callbacks
"""

from __future__ import annotations

import asyncio
import os
import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.operations.key_rotation import (
    KeyRotationConfig,
    KeyRotationResult,
    KeyRotationScheduler,
    RotationSchedule,
    RotationStatus,
    get_key_rotation_scheduler,
    reset_key_rotation_scheduler,
)


# =============================================================================
# KeyRotationConfig Tests
# =============================================================================


class TestKeyRotationConfig:
    """Tests for KeyRotationConfig dataclass."""

    def test_default_values(self):
        """Should have sensible default values."""
        config = KeyRotationConfig()

        assert config.rotation_interval_days == 90
        assert config.key_overlap_days == 7
        assert config.re_encrypt_on_rotation is False
        assert config.alert_days_before == 7
        assert config.re_encrypt_batch_size == 100
        assert config.re_encrypt_timeout == 3600.0

    def test_custom_values(self):
        """Should accept custom values."""
        config = KeyRotationConfig(
            rotation_interval_days=30,
            key_overlap_days=14,
            re_encrypt_on_rotation=True,
            alert_days_before=3,
        )

        assert config.rotation_interval_days == 30
        assert config.key_overlap_days == 14
        assert config.re_encrypt_on_rotation is True
        assert config.alert_days_before == 3

    def test_stores_to_re_encrypt_default(self):
        """Should have default stores list."""
        config = KeyRotationConfig()

        assert "integrations" in config.stores_to_re_encrypt
        assert "webhooks" in config.stores_to_re_encrypt
        assert "gmail_tokens" in config.stores_to_re_encrypt
        assert "enterprise_sync" in config.stores_to_re_encrypt

    def test_from_env(self):
        """Should create config from environment variables."""
        with patch.dict(
            os.environ,
            {
                "ARAGORA_KEY_ROTATION_INTERVAL_DAYS": "60",
                "ARAGORA_KEY_ROTATION_OVERLAP_DAYS": "14",
                "ARAGORA_KEY_ROTATION_RE_ENCRYPT": "true",
                "ARAGORA_KEY_ROTATION_ALERT_DAYS": "5",
            },
        ):
            config = KeyRotationConfig.from_env()

            assert config.rotation_interval_days == 60
            assert config.key_overlap_days == 14
            assert config.re_encrypt_on_rotation is True
            assert config.alert_days_before == 5

    def test_from_env_defaults(self):
        """Should use defaults when env vars not set."""
        with patch.dict(os.environ, {}, clear=True):
            config = KeyRotationConfig.from_env()

            assert config.rotation_interval_days == 90
            assert config.key_overlap_days == 7


# =============================================================================
# KeyRotationResult Tests
# =============================================================================


class TestKeyRotationResult:
    """Tests for KeyRotationResult dataclass."""

    def test_success_result(self):
        """Should create successful result."""
        now = datetime.now(timezone.utc)
        result = KeyRotationResult(
            success=True,
            key_id="master",
            old_version=1,
            new_version=2,
            started_at=now,
            completed_at=now + timedelta(seconds=5),
            duration_seconds=5.0,
            records_re_encrypted=100,
        )

        assert result.success is True
        assert result.key_id == "master"
        assert result.new_version == 2
        assert result.error is None

    def test_failure_result(self):
        """Should create failed result."""
        now = datetime.now(timezone.utc)
        result = KeyRotationResult(
            success=False,
            key_id="unknown",
            old_version=0,
            new_version=0,
            started_at=now,
            completed_at=now,
            duration_seconds=0.5,
            error="Encryption service unavailable",
        )

        assert result.success is False
        assert result.error is not None
        assert "unavailable" in result.error

    def test_to_dict(self):
        """Should convert to dictionary."""
        now = datetime.now(timezone.utc)
        result = KeyRotationResult(
            success=True,
            key_id="master",
            old_version=1,
            new_version=2,
            started_at=now,
            completed_at=now,
            duration_seconds=1.5,
            records_re_encrypted=50,
            re_encryption_errors=["store1: error"],
        )

        data = result.to_dict()

        assert data["success"] is True
        assert data["key_id"] == "master"
        assert data["old_version"] == 1
        assert data["new_version"] == 2
        assert data["records_re_encrypted"] == 50
        assert len(data["re_encryption_errors"]) == 1
        assert "started_at" in data
        assert "completed_at" in data


# =============================================================================
# RotationSchedule Tests
# =============================================================================


class TestRotationSchedule:
    """Tests for RotationSchedule dataclass."""

    def test_default_values(self):
        """Should have sensible defaults."""
        schedule = RotationSchedule()

        assert schedule.next_rotation is None
        assert schedule.last_rotation is None
        assert schedule.last_result is None
        assert schedule.rotation_count == 0
        assert schedule.status == RotationStatus.IDLE

    def test_custom_values(self):
        """Should accept custom values."""
        now = datetime.now(timezone.utc)
        schedule = RotationSchedule(
            next_rotation=now + timedelta(days=30),
            last_rotation=now - timedelta(days=60),
            rotation_count=5,
            status=RotationStatus.COMPLETED,
        )

        assert schedule.rotation_count == 5
        assert schedule.status == RotationStatus.COMPLETED


# =============================================================================
# RotationStatus Tests
# =============================================================================


class TestRotationStatus:
    """Tests for RotationStatus enum."""

    def test_all_statuses(self):
        """Should have all expected statuses."""
        assert RotationStatus.IDLE.value == "idle"
        assert RotationStatus.SCHEDULED.value == "scheduled"
        assert RotationStatus.ROTATING.value == "rotating"
        assert RotationStatus.RE_ENCRYPTING.value == "re_encrypting"
        assert RotationStatus.COMPLETED.value == "completed"
        assert RotationStatus.FAILED.value == "failed"


# =============================================================================
# KeyRotationScheduler Tests
# =============================================================================


class TestKeyRotationScheduler:
    """Tests for KeyRotationScheduler class."""

    def setup_method(self):
        """Reset scheduler before each test."""
        reset_key_rotation_scheduler()

    def test_init_default(self):
        """Should initialize with defaults."""
        scheduler = KeyRotationScheduler()

        assert scheduler.config is not None
        assert scheduler.config.rotation_interval_days == 90
        assert scheduler.alert_callback is None
        assert scheduler.status == RotationStatus.IDLE

    def test_init_with_config(self):
        """Should initialize with custom config."""
        config = KeyRotationConfig(rotation_interval_days=30)
        scheduler = KeyRotationScheduler(config=config)

        assert scheduler.config.rotation_interval_days == 30

    def test_init_with_alert_callback(self):
        """Should accept alert callback."""
        callback = MagicMock()
        scheduler = KeyRotationScheduler(alert_callback=callback)

        assert scheduler.alert_callback is callback

    @pytest.mark.asyncio
    async def test_start_stop(self):
        """Should start and stop scheduler."""
        scheduler = KeyRotationScheduler()

        assert not scheduler._running

        await scheduler.start()
        assert scheduler._running
        assert scheduler._task is not None

        await scheduler.stop()
        assert not scheduler._running
        assert scheduler._task is None

    @pytest.mark.asyncio
    async def test_start_twice_is_noop(self):
        """Starting twice should not create second task."""
        scheduler = KeyRotationScheduler()

        await scheduler.start()
        task1 = scheduler._task

        await scheduler.start()  # Second start
        task2 = scheduler._task

        assert task1 is task2  # Same task

        await scheduler.stop()

    @pytest.mark.asyncio
    async def test_get_status(self):
        """Should return status dictionary."""
        scheduler = KeyRotationScheduler()
        status = await scheduler.get_status()

        assert "status" in status
        assert "running" in status
        assert "next_rotation" in status
        assert "last_rotation" in status
        assert "rotation_count" in status
        assert "config" in status
        assert status["status"] == "idle"
        assert status["running"] is False

    @pytest.mark.asyncio
    async def test_check_rotation_due_no_key(self):
        """Should return True when no key exists."""
        scheduler = KeyRotationScheduler()

        with patch("aragora.security.encryption.get_encryption_service") as mock_service:
            mock_service.return_value.get_active_key.return_value = None
            result = await scheduler.check_rotation_due()

        assert result is True

    @pytest.mark.asyncio
    async def test_check_rotation_due_old_key(self):
        """Should return True when key is old."""
        scheduler = KeyRotationScheduler(config=KeyRotationConfig(rotation_interval_days=30))

        old_key = MagicMock()
        old_key.created_at = datetime.now(timezone.utc) - timedelta(days=60)

        with patch("aragora.security.encryption.get_encryption_service") as mock_service:
            mock_service.return_value.get_active_key.return_value = old_key
            result = await scheduler.check_rotation_due()

        assert result is True

    @pytest.mark.asyncio
    async def test_check_rotation_due_recent_key(self):
        """Should return False when key is recent."""
        scheduler = KeyRotationScheduler(config=KeyRotationConfig(rotation_interval_days=90))

        recent_key = MagicMock()
        recent_key.created_at = datetime.now(timezone.utc) - timedelta(days=10)

        with patch("aragora.security.encryption.get_encryption_service") as mock_service:
            mock_service.return_value.get_active_key.return_value = recent_key
            result = await scheduler.check_rotation_due()

        assert result is False

    @pytest.mark.asyncio
    async def test_check_rotation_due_error_handling(self):
        """Should return False on error."""
        scheduler = KeyRotationScheduler()

        with patch("aragora.security.encryption.get_encryption_service") as mock_service:
            mock_service.side_effect = Exception("Service unavailable")
            result = await scheduler.check_rotation_due()

        assert result is False

    @pytest.mark.asyncio
    async def test_get_key_age_days(self):
        """Should return key age in days."""
        scheduler = KeyRotationScheduler()

        key = MagicMock()
        key.created_at = datetime.now(timezone.utc) - timedelta(days=45)

        with patch("aragora.security.encryption.get_encryption_service") as mock_service:
            mock_service.return_value.get_active_key.return_value = key
            age = await scheduler.get_key_age_days()

        assert age == 45

    @pytest.mark.asyncio
    async def test_get_key_age_days_no_key(self):
        """Should return None when no key."""
        scheduler = KeyRotationScheduler()

        with patch("aragora.security.encryption.get_encryption_service") as mock_service:
            mock_service.return_value.get_active_key.return_value = None
            age = await scheduler.get_key_age_days()

        assert age is None


class TestRotateNow:
    """Tests for rotate_now method."""

    def setup_method(self):
        """Reset scheduler before each test."""
        reset_key_rotation_scheduler()

    @pytest.mark.asyncio
    async def test_rotate_now_success(self):
        """Should perform rotation successfully."""
        scheduler = KeyRotationScheduler()

        # Mock encryption service
        mock_key = MagicMock()
        mock_key.key_id = "master"
        mock_key.version = 2

        old_key = MagicMock()
        old_key.version = 1

        with patch("aragora.security.encryption.get_encryption_service") as mock_service:
            mock_service.return_value.get_active_key.return_value = old_key
            mock_service.return_value.rotate_key.return_value = mock_key

            # Mock audit and metrics
            with patch(
                "aragora.observability.security_audit.audit_key_rotation", new_callable=AsyncMock
            ):
                with patch("aragora.observability.metrics.security.record_key_rotation"):
                    with patch.object(scheduler, "_send_alert", new_callable=AsyncMock):
                        result = await scheduler.rotate_now()

        assert result.success is True
        assert result.key_id == "master"
        assert result.old_version == 1
        assert result.new_version == 2
        assert scheduler.status == RotationStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_rotate_now_failure(self):
        """Should handle rotation failure."""
        scheduler = KeyRotationScheduler()

        with patch("aragora.security.encryption.get_encryption_service") as mock_service:
            mock_service.return_value.get_active_key.side_effect = Exception("No key")

            with patch(
                "aragora.observability.security_audit.audit_key_rotation", new_callable=AsyncMock
            ):
                with patch("aragora.observability.metrics.security.record_key_rotation"):
                    with patch.object(scheduler, "_send_alert", new_callable=AsyncMock):
                        result = await scheduler.rotate_now()

        assert result.success is False
        assert result.error is not None
        assert scheduler.status == RotationStatus.FAILED

    @pytest.mark.asyncio
    async def test_rotate_now_with_re_encrypt(self):
        """Should re-encrypt data when configured."""
        config = KeyRotationConfig(
            re_encrypt_on_rotation=True,
            stores_to_re_encrypt=["integrations"],
        )
        scheduler = KeyRotationScheduler(config=config)

        mock_key = MagicMock()
        mock_key.key_id = "master"
        mock_key.version = 2

        old_key = MagicMock()
        old_key.version = 1

        with patch("aragora.security.encryption.get_encryption_service") as mock_service:
            mock_service.return_value.get_active_key.return_value = old_key
            mock_service.return_value.rotate_key.return_value = mock_key

            with patch(
                "aragora.observability.security_audit.audit_key_rotation", new_callable=AsyncMock
            ):
                with patch(
                    "aragora.observability.security_audit.audit_migration_started",
                    new_callable=AsyncMock,
                ):
                    with patch(
                        "aragora.observability.security_audit.audit_migration_completed",
                        new_callable=AsyncMock,
                    ):
                        with patch("aragora.observability.metrics.security.record_key_rotation"):
                            with patch("aragora.observability.metrics.security.track_migration"):
                                with patch.object(
                                    scheduler, "_re_encrypt_store", new_callable=AsyncMock
                                ) as mock_re_encrypt:
                                    mock_re_encrypt.return_value = 10
                                    with patch.object(
                                        scheduler, "_send_alert", new_callable=AsyncMock
                                    ):
                                        result = await scheduler.rotate_now()

        assert result.success is True
        assert result.records_re_encrypted == 10
        mock_re_encrypt.assert_called_once_with("integrations")


class TestSchedulerAlerts:
    """Tests for alert functionality."""

    def setup_method(self):
        """Reset scheduler before each test."""
        reset_key_rotation_scheduler()

    @pytest.mark.asyncio
    async def test_alert_callback_called(self):
        """Should call alert callback."""
        callback = MagicMock()
        scheduler = KeyRotationScheduler(alert_callback=callback)

        with patch(
            "aragora.observability.security_audit.audit_security_alert", new_callable=AsyncMock
        ):
            await scheduler._send_alert(
                "warning",
                "Test alert",
                {"key": "value"},
            )

        callback.assert_called_once_with("warning", "Test alert", {"key": "value"})

    @pytest.mark.asyncio
    async def test_alert_callback_error_handling(self):
        """Should handle callback errors gracefully."""
        callback = MagicMock(side_effect=Exception("Callback failed"))
        scheduler = KeyRotationScheduler(alert_callback=callback)

        with patch(
            "aragora.observability.security_audit.audit_security_alert", new_callable=AsyncMock
        ):
            # Should not raise
            await scheduler._send_alert("warning", "Test", {})

    @pytest.mark.asyncio
    async def test_check_alerts_sends_warning(self):
        """Should send alert when rotation is approaching."""
        config = KeyRotationConfig(
            rotation_interval_days=30,
            alert_days_before=7,
        )
        scheduler = KeyRotationScheduler(config=config)

        # Key is 25 days old, 5 days until rotation
        with patch.object(scheduler, "get_key_age_days", new_callable=AsyncMock) as mock_age:
            mock_age.return_value = 25
            with patch.object(scheduler, "_send_alert", new_callable=AsyncMock) as mock_alert:
                await scheduler._check_alerts()

        mock_alert.assert_called_once()
        call_args = mock_alert.call_args
        assert call_args[0][0] == "warning"
        assert "5 days" in call_args[0][1]


class TestReEncryptStores:
    """Tests for re-encryption operations."""

    def setup_method(self):
        """Reset scheduler before each test."""
        reset_key_rotation_scheduler()

    @pytest.mark.asyncio
    async def test_re_encrypt_unknown_store(self):
        """Should return 0 for unknown store."""
        scheduler = KeyRotationScheduler()

        count = await scheduler._re_encrypt_store("unknown_store")

        assert count == 0

    @pytest.mark.asyncio
    async def test_re_encrypt_integrations_no_db(self):
        """Should return 0 when database doesn't exist."""
        scheduler = KeyRotationScheduler()

        with patch("pathlib.Path.exists", return_value=False):
            count = await scheduler._re_encrypt_integrations()

        assert count == 0

    @pytest.mark.asyncio
    async def test_re_encrypt_webhooks_no_db(self):
        """Should return 0 when database doesn't exist."""
        scheduler = KeyRotationScheduler()

        with patch("pathlib.Path.exists", return_value=False):
            count = await scheduler._re_encrypt_webhooks()

        assert count == 0

    @pytest.mark.asyncio
    async def test_re_encrypt_gmail_tokens_no_db(self):
        """Should return 0 when database doesn't exist."""
        scheduler = KeyRotationScheduler()

        with patch("pathlib.Path.exists", return_value=False):
            count = await scheduler._re_encrypt_gmail_tokens()

        assert count == 0

    @pytest.mark.asyncio
    async def test_re_encrypt_sync_configs_no_db(self):
        """Should return 0 when database doesn't exist."""
        scheduler = KeyRotationScheduler()

        with patch("pathlib.Path.exists", return_value=False):
            count = await scheduler._re_encrypt_sync_configs()

        assert count == 0


# =============================================================================
# Global Functions Tests
# =============================================================================


class TestGlobalFunctions:
    """Tests for global scheduler functions."""

    def setup_method(self):
        """Reset scheduler before each test."""
        reset_key_rotation_scheduler()

    def test_get_key_rotation_scheduler_singleton(self):
        """Should return same instance."""
        scheduler1 = get_key_rotation_scheduler()
        scheduler2 = get_key_rotation_scheduler()

        assert scheduler1 is scheduler2

    def test_reset_key_rotation_scheduler(self):
        """Should reset singleton."""
        scheduler1 = get_key_rotation_scheduler()
        reset_key_rotation_scheduler()
        scheduler2 = get_key_rotation_scheduler()

        assert scheduler1 is not scheduler2


# =============================================================================
# Scheduler Loop Tests
# =============================================================================


class TestSchedulerLoop:
    """Tests for scheduler loop behavior."""

    def setup_method(self):
        """Reset scheduler before each test."""
        reset_key_rotation_scheduler()

    @pytest.mark.asyncio
    async def test_scheduler_loop_runs(self):
        """Should run scheduler loop until stopped."""
        scheduler = KeyRotationScheduler()

        # Mock the inner methods
        check_count = 0

        async def mock_check_and_rotate():
            nonlocal check_count
            check_count += 1

        async def mock_check_alerts():
            pass

        with patch.object(scheduler, "_check_and_rotate", side_effect=mock_check_and_rotate):
            with patch.object(scheduler, "_check_alerts", side_effect=mock_check_alerts):
                with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
                    # Make sleep raise CancelledError after first call
                    mock_sleep.side_effect = [None, asyncio.CancelledError()]

                    scheduler._running = True
                    try:
                        await scheduler._scheduler_loop()
                    except asyncio.CancelledError:
                        pass

        assert check_count >= 1

    @pytest.mark.asyncio
    async def test_scheduler_loop_handles_errors(self):
        """Should continue after errors."""
        scheduler = KeyRotationScheduler()

        call_count = 0

        async def mock_check_and_rotate():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("First call fails")

        with patch.object(scheduler, "_check_and_rotate", side_effect=mock_check_and_rotate):
            with patch.object(scheduler, "_check_alerts", new_callable=AsyncMock):
                with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
                    # Stop after second iteration
                    mock_sleep.side_effect = [None, asyncio.CancelledError()]

                    scheduler._running = True
                    try:
                        await scheduler._scheduler_loop()
                    except asyncio.CancelledError:
                        pass

        assert call_count >= 1


# =============================================================================
# Integration Tests
# =============================================================================


class TestKeyRotationIntegration:
    """Integration tests for key rotation."""

    def setup_method(self):
        """Reset scheduler before each test."""
        reset_key_rotation_scheduler()

    def test_all_components_importable(self):
        """Should import all components."""
        from aragora.operations.key_rotation import (
            KeyRotationConfig,
            KeyRotationResult,
            KeyRotationScheduler,
            RotationSchedule,
            RotationStatus,
            get_key_rotation_scheduler,
            reset_key_rotation_scheduler,
        )

        assert KeyRotationConfig is not None
        assert KeyRotationResult is not None
        assert KeyRotationScheduler is not None
        assert RotationSchedule is not None
        assert RotationStatus is not None

    @pytest.mark.asyncio
    async def test_full_rotation_flow(self):
        """Should complete full rotation flow."""
        config = KeyRotationConfig(
            rotation_interval_days=30,
            re_encrypt_on_rotation=False,
        )
        scheduler = KeyRotationScheduler(config=config)

        mock_key = MagicMock()
        mock_key.key_id = "master"
        mock_key.version = 2

        old_key = MagicMock()
        old_key.version = 1

        with patch("aragora.security.encryption.get_encryption_service") as mock_service:
            mock_service.return_value.get_active_key.return_value = old_key
            mock_service.return_value.rotate_key.return_value = mock_key

            with patch(
                "aragora.observability.security_audit.audit_key_rotation", new_callable=AsyncMock
            ):
                with patch("aragora.observability.metrics.security.record_key_rotation"):
                    with patch.object(scheduler, "_send_alert", new_callable=AsyncMock):
                        # Perform rotation
                        result = await scheduler.rotate_now()

                        # Check status
                        status = await scheduler.get_status()

        assert result.success is True
        assert status["rotation_count"] == 1
        assert status["last_result"]["success"] is True
