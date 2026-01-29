"""
Tests for aragora.backup.scheduler module.

Tests cover:
- BackupSchedule configuration and next-time calculation
- BackupJob dataclass and serialization
- SchedulerStats tracking
- BackupScheduler lifecycle (start, stop, pause, resume)
- Backup execution with verification and cleanup
- Event emission and metrics recording
- DR drill execution
- Job history tracking
- Global scheduler instance management
"""

from __future__ import annotations

import asyncio
from datetime import datetime, time, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.backup.scheduler import (
    BackupJob,
    BackupSchedule,
    BackupScheduler,
    ScheduleType,
    SchedulerStats,
    SchedulerStatus,
    get_backup_scheduler,
    set_backup_scheduler,
)


# =============================================================================
# TestScheduleType
# =============================================================================


class TestScheduleType:
    """Tests for ScheduleType enum."""

    def test_values(self):
        """Should have all expected schedule types."""
        assert ScheduleType.HOURLY == "hourly"
        assert ScheduleType.DAILY == "daily"
        assert ScheduleType.WEEKLY == "weekly"
        assert ScheduleType.MONTHLY == "monthly"
        assert ScheduleType.CUSTOM == "custom"


class TestSchedulerStatus:
    """Tests for SchedulerStatus enum."""

    def test_values(self):
        """Should have all expected status values."""
        assert SchedulerStatus.STOPPED == "stopped"
        assert SchedulerStatus.RUNNING == "running"
        assert SchedulerStatus.PAUSED == "paused"
        assert SchedulerStatus.ERROR == "error"


# =============================================================================
# TestBackupSchedule
# =============================================================================


class TestBackupScheduleInit:
    """Tests for BackupSchedule initialization."""

    def test_default_values(self):
        """Should initialize with sensible defaults."""
        schedule = BackupSchedule()

        assert schedule.hourly_minute is None
        assert schedule.daily is None
        assert schedule.weekly_day is None
        assert schedule.monthly_day is None
        assert schedule.max_retries == 3
        assert schedule.retry_delay_seconds == 60
        assert schedule.verify_after_backup is True
        assert schedule.retention_cleanup_after is True
        assert schedule.enable_dr_drills is True
        assert schedule.dr_drill_interval_days == 30

    def test_custom_values(self):
        """Should accept custom values."""
        schedule = BackupSchedule(
            hourly_minute=30,
            daily=time(2, 0),
            weekly_day=6,
            weekly_time=time(3, 0),
            monthly_day=15,
            monthly_time=time(4, 0),
            max_retries=5,
        )

        assert schedule.hourly_minute == 30
        assert schedule.daily == time(2, 0)
        assert schedule.weekly_day == 6
        assert schedule.monthly_day == 15
        assert schedule.max_retries == 5


class TestBackupScheduleNextTime:
    """Tests for BackupSchedule.get_next_backup_time()."""

    def test_hourly_next_time(self):
        """Should calculate next hourly backup time."""
        schedule = BackupSchedule(hourly_minute=30)
        now = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)

        next_time = schedule.get_next_backup_time(ScheduleType.HOURLY, now)

        assert next_time is not None
        assert next_time.minute == 30
        assert next_time > now

    def test_hourly_rolls_over_when_past(self):
        """Should roll to next hour if minute is past."""
        schedule = BackupSchedule(hourly_minute=15)
        now = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)

        next_time = schedule.get_next_backup_time(ScheduleType.HOURLY, now)

        assert next_time is not None
        assert next_time.hour == 11
        assert next_time.minute == 15

    def test_daily_next_time(self):
        """Should calculate next daily backup time."""
        schedule = BackupSchedule(daily=time(2, 0))
        now = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)

        next_time = schedule.get_next_backup_time(ScheduleType.DAILY, now)

        assert next_time is not None
        assert next_time.hour == 2
        assert next_time.minute == 0
        assert next_time.day == 16  # Next day since 10 AM > 2 AM

    def test_daily_today_if_before_time(self):
        """Should schedule for today if before daily time."""
        schedule = BackupSchedule(daily=time(14, 0))
        now = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)

        next_time = schedule.get_next_backup_time(ScheduleType.DAILY, now)

        assert next_time is not None
        assert next_time.day == 15  # Today
        assert next_time.hour == 14

    def test_weekly_next_time(self):
        """Should calculate next weekly backup time."""
        schedule = BackupSchedule(weekly_day=6, weekly_time=time(3, 0))
        # Monday Jan 15
        now = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)

        next_time = schedule.get_next_backup_time(ScheduleType.WEEKLY, now)

        assert next_time is not None
        assert next_time.weekday() == 6  # Sunday

    def test_monthly_next_time(self):
        """Should calculate next monthly backup time."""
        schedule = BackupSchedule(monthly_day=20, monthly_time=time(4, 0))
        now = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)

        next_time = schedule.get_next_backup_time(ScheduleType.MONTHLY, now)

        assert next_time is not None
        assert next_time.day == 20
        assert next_time.month == 1  # This month since day 20 is ahead

    def test_monthly_rolls_to_next_month(self):
        """Should roll to next month if day is past."""
        schedule = BackupSchedule(monthly_day=10, monthly_time=time(4, 0))
        now = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)

        next_time = schedule.get_next_backup_time(ScheduleType.MONTHLY, now)

        assert next_time is not None
        assert next_time.month == 2  # Next month

    def test_custom_interval(self):
        """Should calculate custom interval time."""
        schedule = BackupSchedule(custom_interval_seconds=3600)
        now = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)

        next_time = schedule.get_next_backup_time(ScheduleType.CUSTOM, now)

        assert next_time is not None
        assert next_time == now + timedelta(seconds=3600)

    def test_returns_none_for_disabled_schedule(self):
        """Should return None when schedule type is not configured."""
        schedule = BackupSchedule()  # No daily configured

        next_time = schedule.get_next_backup_time(ScheduleType.DAILY)

        assert next_time is None

    def test_returns_none_for_weekly_without_config(self):
        """Should return None for weekly when not configured."""
        schedule = BackupSchedule()

        next_time = schedule.get_next_backup_time(ScheduleType.WEEKLY)

        assert next_time is None


# =============================================================================
# TestBackupJob
# =============================================================================


class TestBackupJobInit:
    """Tests for BackupJob initialization."""

    def test_create_basic(self):
        """Should create with required fields."""
        job = BackupJob(
            id="job-001",
            schedule_type=ScheduleType.DAILY,
            scheduled_at=datetime(2024, 1, 15, 2, 0, tzinfo=timezone.utc),
        )

        assert job.id == "job-001"
        assert job.schedule_type == ScheduleType.DAILY
        assert job.status == "scheduled"
        assert job.backup_id is None
        assert job.verified is False
        assert job.retries == 0

    def test_default_values(self):
        """Should have correct default values."""
        job = BackupJob(
            id="job",
            schedule_type=ScheduleType.HOURLY,
            scheduled_at=datetime.now(timezone.utc),
        )

        assert job.started_at is None
        assert job.completed_at is None
        assert job.error is None
        assert job.metadata == {}


class TestBackupJobToDict:
    """Tests for BackupJob.to_dict()."""

    def test_returns_dict(self):
        """Should return a dictionary."""
        job = BackupJob(
            id="job-001",
            schedule_type=ScheduleType.DAILY,
            scheduled_at=datetime(2024, 1, 15, 2, 0, tzinfo=timezone.utc),
        )

        result = job.to_dict()

        assert isinstance(result, dict)
        assert result["id"] == "job-001"
        assert result["schedule_type"] == "daily"
        assert result["status"] == "scheduled"

    def test_includes_optional_fields(self):
        """Should include optional fields when set."""
        now = datetime.now(timezone.utc)
        job = BackupJob(
            id="job-001",
            schedule_type=ScheduleType.DAILY,
            scheduled_at=now,
            started_at=now,
            completed_at=now + timedelta(minutes=5),
            backup_id="backup-123",
            verified=True,
        )

        result = job.to_dict()

        assert result["backup_id"] == "backup-123"
        assert result["verified"] is True
        assert result["started_at"] is not None
        assert result["completed_at"] is not None


# =============================================================================
# TestSchedulerStats
# =============================================================================


class TestSchedulerStats:
    """Tests for SchedulerStats dataclass."""

    def test_initial_values(self):
        """Should have correct initial values."""
        stats = SchedulerStats(status=SchedulerStatus.STOPPED)

        assert stats.status == SchedulerStatus.STOPPED
        assert stats.total_backups == 0
        assert stats.successful_backups == 0
        assert stats.failed_backups == 0
        assert stats.last_backup_at is None
        assert stats.dr_drills_completed == 0
        assert stats.uptime_seconds == 0.0


# =============================================================================
# TestBackupSchedulerInit
# =============================================================================


class TestBackupSchedulerInit:
    """Tests for BackupScheduler initialization."""

    def test_default_init(self):
        """Should initialize with defaults."""
        manager = MagicMock()
        scheduler = BackupScheduler(backup_manager=manager)

        assert scheduler.status == SchedulerStatus.STOPPED
        assert scheduler.schedule.daily == time(2, 0)

    def test_custom_schedule(self):
        """Should accept custom schedule."""
        manager = MagicMock()
        schedule = BackupSchedule(
            daily=time(3, 0),
            weekly_day=5,
            weekly_time=time(4, 0),
        )
        scheduler = BackupScheduler(
            backup_manager=manager,
            schedule=schedule,
        )

        assert scheduler.schedule.daily == time(3, 0)
        assert scheduler.schedule.weekly_day == 5

    def test_accepts_event_callback(self):
        """Should accept event callback."""
        manager = MagicMock()
        callback = MagicMock()
        scheduler = BackupScheduler(
            backup_manager=manager,
            event_callback=callback,
        )

        assert scheduler._event_callback is callback


# =============================================================================
# TestBackupSchedulerLifecycle
# =============================================================================


class TestBackupSchedulerLifecycle:
    """Tests for BackupScheduler lifecycle methods."""

    @pytest.mark.asyncio
    async def test_start_sets_running(self):
        """Should set status to RUNNING on start."""
        manager = MagicMock()
        scheduler = BackupScheduler(backup_manager=manager)

        await scheduler.start()

        assert scheduler.status == SchedulerStatus.RUNNING

        await scheduler.stop()

    @pytest.mark.asyncio
    async def test_stop_sets_stopped(self):
        """Should set status to STOPPED on stop."""
        manager = MagicMock()
        scheduler = BackupScheduler(backup_manager=manager)

        await scheduler.start()
        await scheduler.stop()

        assert scheduler.status == SchedulerStatus.STOPPED

    @pytest.mark.asyncio
    async def test_pause_sets_paused(self):
        """Should set status to PAUSED on pause."""
        manager = MagicMock()
        scheduler = BackupScheduler(backup_manager=manager)

        await scheduler.start()
        await scheduler.pause()

        assert scheduler.status == SchedulerStatus.PAUSED

        await scheduler.stop()

    @pytest.mark.asyncio
    async def test_resume_sets_running(self):
        """Should set status back to RUNNING on resume."""
        manager = MagicMock()
        scheduler = BackupScheduler(backup_manager=manager)

        await scheduler.start()
        await scheduler.pause()
        await scheduler.resume()

        assert scheduler.status == SchedulerStatus.RUNNING

        await scheduler.stop()

    @pytest.mark.asyncio
    async def test_resume_only_from_paused(self):
        """Should only resume from PAUSED state."""
        manager = MagicMock()
        scheduler = BackupScheduler(backup_manager=manager)

        # Not paused, so resume should not change status
        await scheduler.resume()

        assert scheduler.status == SchedulerStatus.STOPPED

    @pytest.mark.asyncio
    async def test_start_emits_event(self):
        """Should emit event on start."""
        manager = MagicMock()
        callback = MagicMock()
        scheduler = BackupScheduler(
            backup_manager=manager,
            event_callback=callback,
        )

        await scheduler.start()

        callback.assert_called()
        call_args = callback.call_args_list[0]
        assert call_args[0][0] == "backup_scheduler_started"

        await scheduler.stop()

    @pytest.mark.asyncio
    async def test_stop_emits_event(self):
        """Should emit event on stop."""
        manager = MagicMock()
        callback = MagicMock()
        scheduler = BackupScheduler(
            backup_manager=manager,
            event_callback=callback,
        )

        await scheduler.start()
        callback.reset_mock()
        await scheduler.stop()

        callback.assert_called()
        call_args = callback.call_args_list[0]
        assert call_args[0][0] == "backup_scheduler_stopped"


# =============================================================================
# TestBackupSchedulerExecution
# =============================================================================


class TestBackupSchedulerExecution:
    """Tests for backup execution."""

    @pytest.mark.asyncio
    async def test_backup_now_creates_job(self):
        """Should create a backup job on backup_now."""
        mock_metadata = MagicMock()
        mock_metadata.id = "backup-001"

        mock_verify = MagicMock()
        mock_verify.verified = True

        manager = MagicMock()
        manager.create_backup.return_value = mock_metadata
        manager.verify_backup.return_value = mock_verify
        manager.cleanup_expired.return_value = 0

        scheduler = BackupScheduler(backup_manager=manager)

        job = await scheduler.backup_now()

        assert job is not None
        assert job.status == "completed"
        assert job.backup_id == "backup-001"
        assert job.verified is True

    @pytest.mark.asyncio
    async def test_backup_now_handles_failure(self):
        """Should handle backup failure gracefully."""
        manager = MagicMock()
        manager.create_backup.side_effect = RuntimeError("Disk full")

        schedule = BackupSchedule(max_retries=0)
        scheduler = BackupScheduler(
            backup_manager=manager,
            schedule=schedule,
        )

        job = await scheduler.backup_now()

        assert job.status == "failed"
        assert "Disk full" in job.error

    @pytest.mark.asyncio
    async def test_backup_updates_stats(self):
        """Should update statistics after backup."""
        mock_metadata = MagicMock()
        mock_metadata.id = "backup-001"

        mock_verify = MagicMock()
        mock_verify.verified = True

        manager = MagicMock()
        manager.create_backup.return_value = mock_metadata
        manager.verify_backup.return_value = mock_verify
        manager.cleanup_expired.return_value = 0

        scheduler = BackupScheduler(backup_manager=manager)

        await scheduler.backup_now()

        stats = scheduler.get_stats()
        assert stats.total_backups == 1
        assert stats.successful_backups == 1
        assert stats.last_backup_status == "success"

    @pytest.mark.asyncio
    async def test_failed_backup_updates_stats(self):
        """Should update failure stats on backup failure."""
        manager = MagicMock()
        manager.create_backup.side_effect = RuntimeError("Failure")

        schedule = BackupSchedule(max_retries=0)
        scheduler = BackupScheduler(
            backup_manager=manager,
            schedule=schedule,
        )

        await scheduler.backup_now()

        stats = scheduler.get_stats()
        assert stats.total_backups == 1
        assert stats.failed_backups == 1
        assert stats.last_backup_status == "failed"

    @pytest.mark.asyncio
    async def test_backup_emits_completed_event(self):
        """Should emit backup_completed event on success."""
        mock_metadata = MagicMock()
        mock_metadata.id = "backup-001"

        mock_verify = MagicMock()
        mock_verify.verified = True

        manager = MagicMock()
        manager.create_backup.return_value = mock_metadata
        manager.verify_backup.return_value = mock_verify
        manager.cleanup_expired.return_value = 0

        callback = MagicMock()
        scheduler = BackupScheduler(
            backup_manager=manager,
            event_callback=callback,
        )

        await scheduler.backup_now()

        # Find the backup_completed call
        event_names = [call[0][0] for call in callback.call_args_list]
        assert "backup_completed" in event_names

    @pytest.mark.asyncio
    async def test_backup_emits_failed_event(self):
        """Should emit backup_failed event on failure."""
        manager = MagicMock()
        manager.create_backup.side_effect = RuntimeError("Failure")

        callback = MagicMock()
        schedule = BackupSchedule(max_retries=0)
        scheduler = BackupScheduler(
            backup_manager=manager,
            event_callback=callback,
            schedule=schedule,
        )

        await scheduler.backup_now()

        event_names = [call[0][0] for call in callback.call_args_list]
        assert "backup_failed" in event_names


# =============================================================================
# TestBackupSchedulerDRDrill
# =============================================================================


class TestBackupSchedulerDRDrill:
    """Tests for DR drill execution."""

    @pytest.mark.asyncio
    async def test_dr_drill_success(self):
        """Should execute DR drill successfully."""
        mock_backup = MagicMock()
        mock_backup.id = "backup-001"
        mock_backup.created_at = datetime.now(timezone.utc) - timedelta(hours=1)

        mock_verify = MagicMock()
        mock_verify.verified = True
        mock_verify.checksum_valid = True

        manager = MagicMock()
        manager.list_backups.return_value = [mock_backup]
        manager.verify_backup.return_value = mock_verify
        manager.restore_backup.return_value = True

        scheduler = BackupScheduler(backup_manager=manager)

        result = await scheduler._execute_dr_drill()

        assert result["success"] is True
        assert "rto_seconds" in result
        assert "rpo_seconds" in result
        assert len(result["steps"]) >= 3

    @pytest.mark.asyncio
    async def test_dr_drill_no_backups(self):
        """Should handle case with no backups available."""
        manager = MagicMock()
        manager.list_backups.return_value = []

        scheduler = BackupScheduler(backup_manager=manager)

        result = await scheduler._execute_dr_drill()

        assert result["success"] is False
        assert "No backups available" in result.get("error", "")

    @pytest.mark.asyncio
    async def test_dr_drill_verification_failure(self):
        """Should handle verification failure during DR drill."""
        mock_backup = MagicMock()
        mock_backup.id = "backup-001"
        mock_backup.created_at = datetime.now(timezone.utc) - timedelta(hours=1)

        mock_verify = MagicMock()
        mock_verify.verified = False
        mock_verify.checksum_valid = False
        mock_verify.errors = ["Checksum mismatch"]

        manager = MagicMock()
        manager.list_backups.return_value = [mock_backup]
        manager.verify_backup.return_value = mock_verify

        scheduler = BackupScheduler(backup_manager=manager)

        result = await scheduler._execute_dr_drill()

        assert result["success"] is False
        assert "verification failed" in result.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_dr_drill_updates_stats(self):
        """Should update DR drill stats on success."""
        mock_backup = MagicMock()
        mock_backup.id = "backup-001"
        mock_backup.created_at = datetime.now(timezone.utc)

        mock_verify = MagicMock()
        mock_verify.verified = True
        mock_verify.checksum_valid = True

        manager = MagicMock()
        manager.list_backups.return_value = [mock_backup]
        manager.verify_backup.return_value = mock_verify
        manager.restore_backup.return_value = True

        scheduler = BackupScheduler(backup_manager=manager)

        await scheduler._execute_dr_drill()

        stats = scheduler.get_stats()
        assert stats.dr_drills_completed == 1
        assert stats.last_dr_drill_at is not None

    @pytest.mark.asyncio
    async def test_dr_drill_emits_event(self):
        """Should emit dr_drill_completed event."""
        mock_backup = MagicMock()
        mock_backup.id = "backup-001"
        mock_backup.created_at = datetime.now(timezone.utc)

        mock_verify = MagicMock()
        mock_verify.verified = True
        mock_verify.checksum_valid = True

        manager = MagicMock()
        manager.list_backups.return_value = [mock_backup]
        manager.verify_backup.return_value = mock_verify
        manager.restore_backup.return_value = True

        callback = MagicMock()
        scheduler = BackupScheduler(
            backup_manager=manager,
            event_callback=callback,
        )

        await scheduler._execute_dr_drill()

        event_names = [call[0][0] for call in callback.call_args_list]
        assert "dr_drill_completed" in event_names


# =============================================================================
# TestBackupSchedulerJobHistory
# =============================================================================


class TestBackupSchedulerJobHistory:
    """Tests for job history tracking."""

    @pytest.mark.asyncio
    async def test_job_history_recorded(self):
        """Should record completed jobs in history."""
        mock_metadata = MagicMock()
        mock_metadata.id = "backup-001"

        mock_verify = MagicMock()
        mock_verify.verified = True

        manager = MagicMock()
        manager.create_backup.return_value = mock_metadata
        manager.verify_backup.return_value = mock_verify
        manager.cleanup_expired.return_value = 0

        scheduler = BackupScheduler(backup_manager=manager)

        await scheduler.backup_now()
        await scheduler.backup_now()

        history = scheduler.get_job_history()
        assert len(history) == 2

    @pytest.mark.asyncio
    async def test_job_history_limited(self):
        """Should limit history to requested count."""
        mock_metadata = MagicMock()
        mock_metadata.id = "backup"

        mock_verify = MagicMock()
        mock_verify.verified = True

        manager = MagicMock()
        manager.create_backup.return_value = mock_metadata
        manager.verify_backup.return_value = mock_verify
        manager.cleanup_expired.return_value = 0

        scheduler = BackupScheduler(backup_manager=manager)

        for _ in range(5):
            await scheduler.backup_now()

        history = scheduler.get_job_history(limit=3)
        assert len(history) == 3


# =============================================================================
# TestBackupSchedulerEventCallback
# =============================================================================


class TestBackupSchedulerEventCallback:
    """Tests for event callback handling."""

    def test_set_event_callback(self):
        """Should set event callback."""
        manager = MagicMock()
        scheduler = BackupScheduler(backup_manager=manager)

        callback = MagicMock()
        scheduler.set_event_callback(callback)

        assert scheduler._event_callback is callback

    def test_emit_event_handles_callback_error(self):
        """Should handle callback errors gracefully."""
        manager = MagicMock()
        callback = MagicMock(side_effect=RuntimeError("Callback failed"))
        scheduler = BackupScheduler(
            backup_manager=manager,
            event_callback=callback,
        )

        # Should not raise
        scheduler._emit_event("test_event", {"key": "value"})


# =============================================================================
# TestGlobalSchedulerInstance
# =============================================================================


class TestGlobalSchedulerInstance:
    """Tests for global scheduler instance management."""

    def test_get_returns_none_initially(self):
        """Should return None when no scheduler is set."""
        set_backup_scheduler(None)
        assert get_backup_scheduler() is None

    def test_set_and_get(self):
        """Should set and get scheduler instance."""
        manager = MagicMock()
        scheduler = BackupScheduler(backup_manager=manager)

        set_backup_scheduler(scheduler)
        assert get_backup_scheduler() is scheduler

        # Cleanup
        set_backup_scheduler(None)
