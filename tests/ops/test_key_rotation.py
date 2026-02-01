"""
Tests for Key Rotation Scheduler.

Tests cover:
- Enum types: RotationStatus
- Dataclasses: KeyRotationConfig, KeyRotationResult, RotationSchedule
- KeyRotationScheduler: init, get_status, check_rotation_due, get_key_age_days
- rotate_now: manual rotation with mocked encryption service
- Scheduler lifecycle: start/stop
- Singleton pattern: get_key_rotation_scheduler, reset_key_rotation_scheduler
- Edge cases: rotation when already rotating, force rotation, failed rotations
"""

from __future__ import annotations

import asyncio
import os
import sys
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.ops.key_rotation import (
    KeyRotationConfig,
    KeyRotationResult,
    KeyRotationScheduler,
    RotationSchedule,
    RotationStatus,
    get_key_rotation_scheduler,
    reset_key_rotation_scheduler,
)


# ============================================================================
# Helpers
# ============================================================================


def _make_mock_key(version: int = 1, age_days: int = 10, key_id: str = "key-test"):
    """Create a mock encryption key object."""
    created_at = datetime.now(timezone.utc) - timedelta(days=age_days)
    return SimpleNamespace(version=version, created_at=created_at, key_id=key_id)


def _make_mock_new_key(version: int = 2, key_id: str = "key-new"):
    """Create a mock new key returned by rotate_key()."""
    return SimpleNamespace(version=version, key_id=key_id)


def _make_mock_encryption_module(mock_service):
    """Create a mock module for aragora.security.encryption."""
    mod = MagicMock()
    mod.get_encryption_service = MagicMock(return_value=mock_service)
    return mod


def _make_mock_audit_module():
    """Create a mock module for aragora.observability.security_audit."""
    mod = MagicMock()
    mod.audit_key_rotation = AsyncMock()
    mod.audit_migration_started = AsyncMock()
    mod.audit_migration_completed = AsyncMock()
    mod.audit_security_alert = AsyncMock()
    return mod


def _make_mock_metrics_module():
    """Create a mock module for aragora.observability.metrics.security."""
    mod = MagicMock()
    mod.record_key_rotation = MagicMock()
    mod.track_migration = MagicMock()
    return mod


@contextmanager
def _patch_rotation_imports(
    encryption_service=None,
    encryption_side_effect=None,
):
    """Context manager that patches all external modules used by key_rotation.

    The key_rotation module uses local imports inside methods, so we must
    patch the modules in sys.modules before the methods call their imports.
    """
    mock_encryption_mod = MagicMock()
    if encryption_side_effect:
        mock_encryption_mod.get_encryption_service = MagicMock(side_effect=encryption_side_effect)
    elif encryption_service is not None:
        mock_encryption_mod.get_encryption_service = MagicMock(return_value=encryption_service)
    else:
        mock_encryption_mod.get_encryption_service = MagicMock(return_value=MagicMock())

    mock_audit_mod = MagicMock()
    mock_audit_mod.audit_key_rotation = AsyncMock()
    mock_audit_mod.audit_migration_started = AsyncMock()
    mock_audit_mod.audit_migration_completed = AsyncMock()
    mock_audit_mod.audit_security_alert = AsyncMock()

    mock_metrics_mod = MagicMock()
    mock_metrics_mod.record_key_rotation = MagicMock()
    mock_metrics_mod.track_migration = MagicMock()

    patches = {
        "aragora.security.encryption": mock_encryption_mod,
        "aragora.observability.security_audit": mock_audit_mod,
        "aragora.observability.metrics.security": mock_metrics_mod,
    }

    originals = {}
    for mod_name, mock_mod in patches.items():
        originals[mod_name] = sys.modules.get(mod_name)
        sys.modules[mod_name] = mock_mod

    try:
        yield {
            "encryption": mock_encryption_mod,
            "audit": mock_audit_mod,
            "metrics": mock_metrics_mod,
        }
    finally:
        for mod_name in patches:
            if originals[mod_name] is None:
                sys.modules.pop(mod_name, None)
            else:
                sys.modules[mod_name] = originals[mod_name]


# ============================================================================
# RotationStatus Enum Tests
# ============================================================================


class TestRotationStatus:
    """Tests for RotationStatus enum."""

    def test_all_values(self):
        assert RotationStatus.IDLE.value == "idle"
        assert RotationStatus.SCHEDULED.value == "scheduled"
        assert RotationStatus.ROTATING.value == "rotating"
        assert RotationStatus.RE_ENCRYPTING.value == "re_encrypting"
        assert RotationStatus.COMPLETED.value == "completed"
        assert RotationStatus.FAILED.value == "failed"

    def test_member_count(self):
        assert len(RotationStatus) == 6


# ============================================================================
# KeyRotationConfig Tests
# ============================================================================


class TestKeyRotationConfig:
    """Tests for KeyRotationConfig dataclass."""

    def test_default_values(self):
        config = KeyRotationConfig()
        assert config.rotation_interval_days == 90
        assert config.key_overlap_days == 7
        assert config.re_encrypt_on_rotation is False
        assert config.alert_days_before == 7
        assert config.re_encrypt_batch_size == 100
        assert config.re_encrypt_timeout == 3600.0
        assert config.stores_to_re_encrypt == [
            "integrations",
            "webhooks",
            "gmail_tokens",
            "enterprise_sync",
        ]

    def test_custom_values(self):
        config = KeyRotationConfig(
            rotation_interval_days=30,
            key_overlap_days=14,
            re_encrypt_on_rotation=True,
            alert_days_before=3,
            re_encrypt_batch_size=50,
            re_encrypt_timeout=1800.0,
            stores_to_re_encrypt=["integrations"],
        )
        assert config.rotation_interval_days == 30
        assert config.key_overlap_days == 14
        assert config.re_encrypt_on_rotation is True
        assert config.alert_days_before == 3
        assert config.re_encrypt_batch_size == 50
        assert config.re_encrypt_timeout == 1800.0
        assert config.stores_to_re_encrypt == ["integrations"]

    def test_from_env_defaults(self):
        """from_env should use defaults when env vars are not set."""
        with patch.dict(os.environ, {}, clear=True):
            config = KeyRotationConfig.from_env()
            assert config.rotation_interval_days == 90
            assert config.key_overlap_days == 7
            assert config.re_encrypt_on_rotation is False
            assert config.alert_days_before == 7

    def test_from_env_custom(self):
        """from_env should read from environment variables."""
        env = {
            "ARAGORA_KEY_ROTATION_INTERVAL_DAYS": "30",
            "ARAGORA_KEY_ROTATION_OVERLAP_DAYS": "14",
            "ARAGORA_KEY_ROTATION_RE_ENCRYPT": "true",
            "ARAGORA_KEY_ROTATION_ALERT_DAYS": "3",
        }
        with patch.dict(os.environ, env, clear=True):
            config = KeyRotationConfig.from_env()
            assert config.rotation_interval_days == 30
            assert config.key_overlap_days == 14
            assert config.re_encrypt_on_rotation is True
            assert config.alert_days_before == 3

    def test_from_env_re_encrypt_false_values(self):
        """from_env should treat non-'true' strings as False for re_encrypt."""
        for val in ("false", "False", "0", "no", ""):
            with patch.dict(os.environ, {"ARAGORA_KEY_ROTATION_RE_ENCRYPT": val}, clear=True):
                config = KeyRotationConfig.from_env()
                assert config.re_encrypt_on_rotation is False

    def test_stores_default_is_independent_per_instance(self):
        """Each config instance should have its own stores list."""
        config_a = KeyRotationConfig()
        config_b = KeyRotationConfig()
        config_a.stores_to_re_encrypt.append("extra_store")
        assert "extra_store" not in config_b.stores_to_re_encrypt


# ============================================================================
# KeyRotationResult Tests
# ============================================================================


class TestKeyRotationResult:
    """Tests for KeyRotationResult dataclass."""

    def test_success_result(self):
        now = datetime.now(timezone.utc)
        result = KeyRotationResult(
            success=True,
            key_id="key-123",
            old_version=1,
            new_version=2,
            started_at=now,
            completed_at=now + timedelta(seconds=5),
            duration_seconds=5.0,
            records_re_encrypted=42,
        )
        assert result.success is True
        assert result.key_id == "key-123"
        assert result.old_version == 1
        assert result.new_version == 2
        assert result.records_re_encrypted == 42
        assert result.re_encryption_errors == []
        assert result.error is None

    def test_failure_result(self):
        now = datetime.now(timezone.utc)
        result = KeyRotationResult(
            success=False,
            key_id="unknown",
            old_version=0,
            new_version=0,
            started_at=now,
            completed_at=now + timedelta(seconds=1),
            duration_seconds=1.0,
            error="Encryption service unavailable",
        )
        assert result.success is False
        assert result.error == "Encryption service unavailable"

    def test_to_dict(self):
        started = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        completed = datetime(2025, 1, 1, 12, 0, 5, tzinfo=timezone.utc)
        result = KeyRotationResult(
            success=True,
            key_id="key-abc",
            old_version=3,
            new_version=4,
            started_at=started,
            completed_at=completed,
            duration_seconds=5.0,
            records_re_encrypted=10,
            re_encryption_errors=["store_x: timeout"],
        )
        d = result.to_dict()
        assert d["success"] is True
        assert d["key_id"] == "key-abc"
        assert d["old_version"] == 3
        assert d["new_version"] == 4
        assert d["started_at"] == started.isoformat()
        assert d["completed_at"] == completed.isoformat()
        assert d["duration_seconds"] == 5.0
        assert d["records_re_encrypted"] == 10
        assert d["re_encryption_errors"] == ["store_x: timeout"]
        assert d["error"] is None

    def test_to_dict_with_error(self):
        now = datetime.now(timezone.utc)
        result = KeyRotationResult(
            success=False,
            key_id="unknown",
            old_version=0,
            new_version=0,
            started_at=now,
            completed_at=now,
            duration_seconds=0.1,
            error="something broke",
        )
        d = result.to_dict()
        assert d["success"] is False
        assert d["error"] == "something broke"


# ============================================================================
# RotationSchedule Tests
# ============================================================================


class TestRotationSchedule:
    """Tests for RotationSchedule dataclass."""

    def test_default_values(self):
        schedule = RotationSchedule()
        assert schedule.next_rotation is None
        assert schedule.last_rotation is None
        assert schedule.last_result is None
        assert schedule.rotation_count == 0
        assert schedule.status == RotationStatus.IDLE

    def test_status_transition(self):
        schedule = RotationSchedule()
        schedule.status = RotationStatus.SCHEDULED
        assert schedule.status == RotationStatus.SCHEDULED
        schedule.status = RotationStatus.ROTATING
        assert schedule.status == RotationStatus.ROTATING
        schedule.status = RotationStatus.COMPLETED
        assert schedule.status == RotationStatus.COMPLETED

    def test_tracking_rotation_result(self):
        now = datetime.now(timezone.utc)
        result = KeyRotationResult(
            success=True,
            key_id="key-1",
            old_version=1,
            new_version=2,
            started_at=now,
            completed_at=now,
            duration_seconds=1.0,
        )
        schedule = RotationSchedule()
        schedule.last_result = result
        schedule.last_rotation = now
        schedule.rotation_count = 1
        assert schedule.last_result.success is True
        assert schedule.rotation_count == 1


# ============================================================================
# KeyRotationScheduler Tests
# ============================================================================


class TestKeyRotationSchedulerInit:
    """Tests for KeyRotationScheduler initialization."""

    def test_init_with_default_config(self):
        with patch.dict(os.environ, {}, clear=True):
            scheduler = KeyRotationScheduler()
            assert scheduler.config.rotation_interval_days == 90
            assert scheduler.alert_callback is None
            assert scheduler.status == RotationStatus.IDLE
            assert scheduler._running is False
            assert scheduler._task is None

    def test_init_with_custom_config(self):
        config = KeyRotationConfig(rotation_interval_days=30, alert_days_before=5)
        scheduler = KeyRotationScheduler(config=config)
        assert scheduler.config.rotation_interval_days == 30
        assert scheduler.config.alert_days_before == 5

    def test_init_with_alert_callback(self):
        callback = MagicMock()
        scheduler = KeyRotationScheduler(
            config=KeyRotationConfig(),
            alert_callback=callback,
        )
        assert scheduler.alert_callback is callback

    def test_status_property(self):
        scheduler = KeyRotationScheduler(config=KeyRotationConfig())
        assert scheduler.status == RotationStatus.IDLE
        scheduler._schedule.status = RotationStatus.ROTATING
        assert scheduler.status == RotationStatus.ROTATING


class TestGetStatus:
    """Tests for KeyRotationScheduler.get_status()."""

    @pytest.mark.asyncio
    async def test_initial_status(self):
        scheduler = KeyRotationScheduler(config=KeyRotationConfig())
        status = await scheduler.get_status()

        assert status["status"] == "idle"
        assert status["running"] is False
        assert status["next_rotation"] is None
        assert status["last_rotation"] is None
        assert status["rotation_count"] == 0
        assert status["last_result"] is None
        assert "config" in status
        assert status["config"]["rotation_interval_days"] == 90
        assert status["config"]["key_overlap_days"] == 7
        assert status["config"]["re_encrypt_on_rotation"] is False
        assert status["config"]["alert_days_before"] == 7

    @pytest.mark.asyncio
    async def test_status_with_scheduled_rotation(self):
        scheduler = KeyRotationScheduler(config=KeyRotationConfig())
        next_time = datetime(2025, 6, 1, 0, 0, 0, tzinfo=timezone.utc)
        scheduler._schedule.next_rotation = next_time
        scheduler._schedule.status = RotationStatus.SCHEDULED

        status = await scheduler.get_status()
        assert status["status"] == "scheduled"
        assert status["next_rotation"] == next_time.isoformat()

    @pytest.mark.asyncio
    async def test_status_with_last_result(self):
        scheduler = KeyRotationScheduler(config=KeyRotationConfig())
        now = datetime.now(timezone.utc)
        result = KeyRotationResult(
            success=True,
            key_id="key-1",
            old_version=1,
            new_version=2,
            started_at=now,
            completed_at=now,
            duration_seconds=2.5,
        )
        scheduler._schedule.last_result = result
        scheduler._schedule.last_rotation = now
        scheduler._schedule.rotation_count = 3

        status = await scheduler.get_status()
        assert status["rotation_count"] == 3
        assert status["last_result"]["success"] is True
        assert status["last_result"]["key_id"] == "key-1"
        assert status["last_rotation"] == now.isoformat()


class TestCheckRotationDue:
    """Tests for KeyRotationScheduler.check_rotation_due()."""

    @pytest.mark.asyncio
    async def test_rotation_due_when_key_is_old(self):
        config = KeyRotationConfig(rotation_interval_days=90)
        scheduler = KeyRotationScheduler(config=config)

        mock_key = _make_mock_key(version=1, age_days=100)
        mock_service = MagicMock()
        mock_service.get_active_key.return_value = mock_key

        with _patch_rotation_imports(encryption_service=mock_service):
            assert await scheduler.check_rotation_due() is True

    @pytest.mark.asyncio
    async def test_rotation_not_due_when_key_is_fresh(self):
        config = KeyRotationConfig(rotation_interval_days=90)
        scheduler = KeyRotationScheduler(config=config)

        mock_key = _make_mock_key(version=1, age_days=10)
        mock_service = MagicMock()
        mock_service.get_active_key.return_value = mock_key

        with _patch_rotation_imports(encryption_service=mock_service):
            assert await scheduler.check_rotation_due() is False

    @pytest.mark.asyncio
    async def test_rotation_due_when_no_active_key(self):
        config = KeyRotationConfig(rotation_interval_days=90)
        scheduler = KeyRotationScheduler(config=config)

        mock_service = MagicMock()
        mock_service.get_active_key.return_value = None

        with _patch_rotation_imports(encryption_service=mock_service):
            assert await scheduler.check_rotation_due() is True

    @pytest.mark.asyncio
    async def test_rotation_due_returns_false_on_exception(self):
        scheduler = KeyRotationScheduler(config=KeyRotationConfig())

        with _patch_rotation_imports(
            encryption_side_effect=RuntimeError("service unavailable"),
        ):
            assert await scheduler.check_rotation_due() is False


class TestGetKeyAgeDays:
    """Tests for KeyRotationScheduler.get_key_age_days()."""

    @pytest.mark.asyncio
    async def test_returns_key_age(self):
        scheduler = KeyRotationScheduler(config=KeyRotationConfig())

        mock_key = _make_mock_key(version=1, age_days=45)
        mock_service = MagicMock()
        mock_service.get_active_key.return_value = mock_key

        with _patch_rotation_imports(encryption_service=mock_service):
            age = await scheduler.get_key_age_days()
            assert age == 45

    @pytest.mark.asyncio
    async def test_returns_none_when_no_key(self):
        scheduler = KeyRotationScheduler(config=KeyRotationConfig())

        mock_service = MagicMock()
        mock_service.get_active_key.return_value = None

        with _patch_rotation_imports(encryption_service=mock_service):
            assert await scheduler.get_key_age_days() is None

    @pytest.mark.asyncio
    async def test_returns_none_on_exception(self):
        scheduler = KeyRotationScheduler(config=KeyRotationConfig())

        with _patch_rotation_imports(
            encryption_side_effect=RuntimeError("boom"),
        ):
            assert await scheduler.get_key_age_days() is None


# ============================================================================
# rotate_now Tests
# ============================================================================


class TestRotateNow:
    """Tests for KeyRotationScheduler.rotate_now()."""

    @pytest.mark.asyncio
    async def test_successful_rotation(self):
        config = KeyRotationConfig(re_encrypt_on_rotation=False)
        alert_cb = MagicMock()
        scheduler = KeyRotationScheduler(config=config, alert_callback=alert_cb)

        mock_active_key = _make_mock_key(version=1, age_days=91)
        mock_new_key = _make_mock_new_key(version=2, key_id="key-rotated")
        mock_service = MagicMock()
        mock_service.get_active_key.return_value = mock_active_key
        mock_service.rotate_key.return_value = mock_new_key

        with _patch_rotation_imports(encryption_service=mock_service) as mocks:
            result = await scheduler.rotate_now()

        assert result.success is True
        assert result.key_id == "key-rotated"
        assert result.old_version == 1
        assert result.new_version == 2
        assert result.records_re_encrypted == 0
        assert result.re_encryption_errors == []
        assert result.error is None
        assert result.duration_seconds > 0

        # Verify schedule was updated
        assert scheduler._schedule.rotation_count == 1
        assert scheduler._schedule.last_rotation is not None
        assert scheduler._schedule.next_rotation is not None
        assert scheduler._schedule.last_result is result
        assert scheduler.status == RotationStatus.COMPLETED

        # Verify metrics were recorded
        mocks["metrics"].record_key_rotation.assert_called_once()
        mocks["audit"].audit_key_rotation.assert_called_once()

        # Verify alert callback was called (info alert for success)
        alert_cb.assert_called_once()
        call_args = alert_cb.call_args
        assert call_args[0][0] == "info"

    @pytest.mark.asyncio
    async def test_failed_rotation(self):
        config = KeyRotationConfig()
        scheduler = KeyRotationScheduler(config=config)

        with _patch_rotation_imports(
            encryption_side_effect=RuntimeError("encryption service down"),
        ) as mocks:
            result = await scheduler.rotate_now()

        assert result.success is False
        assert result.key_id == "unknown"
        assert result.old_version == 0
        assert result.new_version == 0
        assert result.error == "encryption service down"
        assert scheduler.status == RotationStatus.FAILED
        assert scheduler._schedule.last_result is result

    @pytest.mark.asyncio
    async def test_rotation_with_no_existing_key(self):
        """Rotation should succeed even when there is no prior active key."""
        config = KeyRotationConfig(re_encrypt_on_rotation=False)
        scheduler = KeyRotationScheduler(config=config)

        mock_new_key = _make_mock_new_key(version=1, key_id="key-first")
        mock_service = MagicMock()
        mock_service.get_active_key.return_value = None
        mock_service.rotate_key.return_value = mock_new_key

        with _patch_rotation_imports(encryption_service=mock_service):
            result = await scheduler.rotate_now()

        assert result.success is True
        assert result.old_version == 0
        assert result.new_version == 1
        assert result.key_id == "key-first"

    @pytest.mark.asyncio
    async def test_rotation_next_rotation_set_correctly(self):
        """After rotation, next_rotation should be interval days in the future."""
        config = KeyRotationConfig(rotation_interval_days=60, re_encrypt_on_rotation=False)
        scheduler = KeyRotationScheduler(config=config)

        mock_active_key = _make_mock_key(version=1, age_days=61)
        mock_new_key = _make_mock_new_key(version=2)
        mock_service = MagicMock()
        mock_service.get_active_key.return_value = mock_active_key
        mock_service.rotate_key.return_value = mock_new_key

        with _patch_rotation_imports(encryption_service=mock_service):
            result = await scheduler.rotate_now()

        assert result.success is True
        # next_rotation should be approximately 60 days from the completed_at time
        expected_next = result.completed_at + timedelta(days=60)
        delta = abs((scheduler._schedule.next_rotation - expected_next).total_seconds())
        assert delta < 5  # within 5 seconds tolerance


# ============================================================================
# Scheduler Lifecycle Tests
# ============================================================================


class TestSchedulerLifecycle:
    """Tests for start/stop lifecycle management."""

    @pytest.mark.asyncio
    async def test_start_sets_running(self):
        scheduler = KeyRotationScheduler(config=KeyRotationConfig())

        # Patch the scheduler loop so it doesn't actually run
        with patch.object(scheduler, "_scheduler_loop", new_callable=AsyncMock):
            await scheduler.start()
            assert scheduler._running is True
            assert scheduler._task is not None

            # Cleanup
            await scheduler.stop()

    @pytest.mark.asyncio
    async def test_stop_clears_state(self):
        scheduler = KeyRotationScheduler(config=KeyRotationConfig())

        with patch.object(scheduler, "_scheduler_loop", new_callable=AsyncMock):
            await scheduler.start()
            await scheduler.stop()

            assert scheduler._running is False
            assert scheduler._task is None

    @pytest.mark.asyncio
    async def test_double_start_is_noop(self):
        """Calling start() twice should not create a second task."""
        scheduler = KeyRotationScheduler(config=KeyRotationConfig())

        with patch.object(scheduler, "_scheduler_loop", new_callable=AsyncMock):
            await scheduler.start()
            first_task = scheduler._task

            await scheduler.start()  # second call
            assert scheduler._task is first_task  # same task, no duplicate

            await scheduler.stop()

    @pytest.mark.asyncio
    async def test_stop_without_start(self):
        """Stopping a scheduler that was never started should be safe."""
        scheduler = KeyRotationScheduler(config=KeyRotationConfig())
        await scheduler.stop()
        assert scheduler._running is False
        assert scheduler._task is None


# ============================================================================
# Singleton Pattern Tests
# ============================================================================


class TestSingletonPattern:
    """Tests for get_key_rotation_scheduler and reset_key_rotation_scheduler."""

    def test_get_returns_same_instance(self):
        reset_key_rotation_scheduler()
        try:
            scheduler_a = get_key_rotation_scheduler()
            scheduler_b = get_key_rotation_scheduler()
            assert scheduler_a is scheduler_b
        finally:
            reset_key_rotation_scheduler()

    def test_reset_clears_instance(self):
        reset_key_rotation_scheduler()
        try:
            scheduler_a = get_key_rotation_scheduler()
            reset_key_rotation_scheduler()
            scheduler_b = get_key_rotation_scheduler()
            assert scheduler_a is not scheduler_b
        finally:
            reset_key_rotation_scheduler()

    def test_reset_when_no_instance(self):
        """reset should be safe to call even when no instance exists."""
        reset_key_rotation_scheduler()
        reset_key_rotation_scheduler()  # second call, should not raise


# ============================================================================
# Edge Cases and Alert Callback Tests
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases in key rotation."""

    @pytest.mark.asyncio
    async def test_alert_callback_exception_is_caught(self):
        """An exception in the alert callback should not propagate."""
        callback = MagicMock(side_effect=ValueError("callback error"))
        config = KeyRotationConfig(re_encrypt_on_rotation=False)
        scheduler = KeyRotationScheduler(config=config, alert_callback=callback)

        mock_active_key = _make_mock_key(version=1, age_days=91)
        mock_new_key = _make_mock_new_key(version=2)
        mock_service = MagicMock()
        mock_service.get_active_key.return_value = mock_active_key
        mock_service.rotate_key.return_value = mock_new_key

        with _patch_rotation_imports(encryption_service=mock_service):
            # Should not raise despite callback failure
            result = await scheduler.rotate_now()

        assert result.success is True
        callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_concurrent_rotate_now_serialized_by_lock(self):
        """Two concurrent rotate_now calls should be serialized by the lock."""
        config = KeyRotationConfig(re_encrypt_on_rotation=False)
        scheduler = KeyRotationScheduler(config=config)

        call_count = 0

        mock_active_key = _make_mock_key(version=1, age_days=91)
        mock_service = MagicMock()
        mock_service.get_active_key.return_value = mock_active_key

        def rotating_key():
            nonlocal call_count
            call_count += 1
            return _make_mock_new_key(version=1 + call_count)

        mock_service.rotate_key.side_effect = rotating_key

        with _patch_rotation_imports(encryption_service=mock_service):
            results = await asyncio.gather(
                scheduler.rotate_now(),
                scheduler.rotate_now(),
            )

        # Both should succeed (serialized, not interleaved)
        assert all(r.success for r in results)
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_check_rotation_due_at_exact_boundary(self):
        """Key exactly at the rotation interval boundary should be due."""
        config = KeyRotationConfig(rotation_interval_days=90)
        scheduler = KeyRotationScheduler(config=config)

        mock_key = _make_mock_key(version=1, age_days=90)
        mock_service = MagicMock()
        mock_service.get_active_key.return_value = mock_key

        with _patch_rotation_imports(encryption_service=mock_service):
            assert await scheduler.check_rotation_due() is True

    @pytest.mark.asyncio
    async def test_check_rotation_due_one_day_before_boundary(self):
        """Key one day before the rotation interval should not be due."""
        config = KeyRotationConfig(rotation_interval_days=90)
        scheduler = KeyRotationScheduler(config=config)

        mock_key = _make_mock_key(version=1, age_days=89)
        mock_service = MagicMock()
        mock_service.get_active_key.return_value = mock_key

        with _patch_rotation_imports(encryption_service=mock_service):
            assert await scheduler.check_rotation_due() is False

    @pytest.mark.asyncio
    async def test_rotation_increments_count_each_time(self):
        """Each successful rotation should increment the rotation count."""
        config = KeyRotationConfig(re_encrypt_on_rotation=False)
        scheduler = KeyRotationScheduler(config=config)

        mock_active_key = _make_mock_key(version=1, age_days=91)
        mock_service = MagicMock()
        mock_service.get_active_key.return_value = mock_active_key

        version_counter = [1]

        def make_new_key():
            version_counter[0] += 1
            return _make_mock_new_key(version=version_counter[0])

        mock_service.rotate_key.side_effect = make_new_key

        with _patch_rotation_imports(encryption_service=mock_service):
            await scheduler.rotate_now()
            assert scheduler._schedule.rotation_count == 1

            await scheduler.rotate_now()
            assert scheduler._schedule.rotation_count == 2

            await scheduler.rotate_now()
            assert scheduler._schedule.rotation_count == 3

    @pytest.mark.asyncio
    async def test_failed_rotation_does_not_increment_count(self):
        """A failed rotation should NOT increment the rotation count."""
        config = KeyRotationConfig()
        scheduler = KeyRotationScheduler(config=config)

        with _patch_rotation_imports(
            encryption_side_effect=RuntimeError("fail"),
        ):
            result = await scheduler.rotate_now()

        assert result.success is False
        assert scheduler._schedule.rotation_count == 0

    @pytest.mark.asyncio
    async def test_status_transitions_during_successful_rotation(self):
        """Verify status transitions through ROTATING -> COMPLETED on success."""
        config = KeyRotationConfig(re_encrypt_on_rotation=False)
        scheduler = KeyRotationScheduler(config=config)
        observed_statuses = []

        mock_active_key = _make_mock_key(version=1, age_days=91)
        mock_service = MagicMock()
        mock_service.get_active_key.return_value = mock_active_key

        original_rotate_key = MagicMock(return_value=_make_mock_new_key(version=2))

        def spying_rotate_key():
            observed_statuses.append(scheduler.status)
            return original_rotate_key()

        mock_service.rotate_key.side_effect = spying_rotate_key

        with _patch_rotation_imports(encryption_service=mock_service):
            result = await scheduler.rotate_now()

        assert result.success is True
        # During rotate_key(), the status should have been ROTATING
        assert RotationStatus.ROTATING in observed_statuses
        # After completion, the status should be COMPLETED
        assert scheduler.status == RotationStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_status_transitions_to_failed_on_error(self):
        """Verify status transitions to FAILED on rotation error."""
        config = KeyRotationConfig()
        scheduler = KeyRotationScheduler(config=config)

        assert scheduler.status == RotationStatus.IDLE

        with _patch_rotation_imports(
            encryption_side_effect=RuntimeError("boom"),
        ):
            await scheduler.rotate_now()

        assert scheduler.status == RotationStatus.FAILED
