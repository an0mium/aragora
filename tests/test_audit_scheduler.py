"""
Tests for AuditScheduler - scheduled and triggered audit execution.

Tests cover:
- Schedule creation and management
- Cron expression parsing
- Job execution lifecycle
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta, timezone

from aragora.scheduler.audit_scheduler import (
    AuditScheduler,
    ScheduleConfig,
    ScheduledJob,
    JobRun,
    TriggerType,
    ScheduleStatus,
    CronParser,
)


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def scheduler():
    """Create an AuditScheduler instance."""
    return AuditScheduler()


@pytest.fixture
def mock_auditor():
    """Create a mock document auditor."""
    auditor = Mock()
    auditor.start_session = AsyncMock(return_value=Mock(
        id="session-123",
        status="completed",
        findings=[],
    ))
    auditor.run_audit = AsyncMock(return_value=Mock(
        id="session-123",
        status="completed",
        findings=[Mock(id="f-1", severity="high")],
    ))
    return auditor


@pytest.fixture
def schedule_config():
    """Create a basic schedule configuration."""
    return ScheduleConfig(
        name="Daily Security Audit",
        cron="0 2 * * *",  # 2 AM daily
        workspace_id="ws-123",
        preset="code_security",
    )


# ============================================================================
# ScheduleConfig Tests
# ============================================================================


class TestScheduleConfig:
    """Tests for schedule configuration."""

    def test_schedule_config_creation(self):
        """Test creating a schedule configuration."""
        config = ScheduleConfig(
            name="Test Schedule",
            cron="0 0 * * *",
            workspace_id="ws-123",
            preset="security",
        )

        assert config.name == "Test Schedule"
        assert config.cron == "0 0 * * *"
        assert config.preset == "security"
        assert config.trigger_type == TriggerType.CRON

    def test_schedule_config_with_audit_types(self):
        """Test schedule with audit types configuration."""
        config = ScheduleConfig(
            name="Multi-type Schedule",
            cron="0 0 * * *",
            workspace_id="ws-123",
            preset="security",
            audit_types=["security", "compliance"],
        )

        assert config.audit_types == ["security", "compliance"]
        assert config.notify_on_complete is True  # Default

    def test_schedule_config_interval_trigger(self):
        """Test schedule with interval trigger."""
        config = ScheduleConfig(
            name="Interval Schedule",
            trigger_type=TriggerType.INTERVAL,
            interval_minutes=60,
            workspace_id="ws-123",
            preset="security",
        )

        assert config.trigger_type == TriggerType.INTERVAL
        assert config.interval_minutes == 60


# ============================================================================
# ScheduledJob Tests
# ============================================================================


class TestScheduledJob:
    """Tests for scheduled job objects."""

    def test_job_creation(self, scheduler, schedule_config):
        """Test creating a scheduled job."""
        job = scheduler.add_schedule(schedule_config)

        assert job is not None
        assert job.job_id is not None
        assert job.schedule_id is not None
        assert job.config == schedule_config
        assert job.status == ScheduleStatus.ACTIVE

    def test_job_has_next_run(self, scheduler, schedule_config):
        """Test that cron job has next_run calculated."""
        job = scheduler.add_schedule(schedule_config)

        # Cron schedule should have next_run set
        assert job.next_run is not None

    def test_job_to_dict(self, scheduler, schedule_config):
        """Test converting job to dictionary."""
        job = scheduler.add_schedule(schedule_config)

        data = job.to_dict()

        assert "job_id" in data
        assert "schedule_id" in data
        assert "name" in data  # name instead of config
        assert "status" in data


# ============================================================================
# JobRun Tests
# ============================================================================


class TestJobRun:
    """Tests for job run records."""

    def test_job_run_creation(self):
        """Test creating a job run record."""
        run = JobRun(
            run_id="run-123",
            job_id="job-456",
            started_at=datetime.now(timezone.utc),
            status="running",
        )

        assert run.run_id == "run-123"
        assert run.status == "running"

    def test_job_run_completion(self):
        """Test completing a job run."""
        run = JobRun(
            run_id="run-123",
            job_id="job-456",
            started_at=datetime.now(timezone.utc),
            status="running",
        )

        run.status = "completed"
        run.completed_at = datetime.now(timezone.utc)
        run.session_id = "session-789"
        run.findings_count = 5

        assert run.status == "completed"
        assert run.findings_count == 5

    def test_job_run_to_dict(self):
        """Test converting run to dictionary."""
        run = JobRun(
            run_id="run-123",
            job_id="job-456",
            started_at=datetime.now(timezone.utc),
            status="completed",
        )

        data = run.to_dict()

        assert data["run_id"] == "run-123"
        assert data["status"] == "completed"


# ============================================================================
# AuditScheduler Schedule Management Tests
# ============================================================================


class TestSchedulerScheduleManagement:
    """Tests for schedule management."""

    def test_add_schedule(self, scheduler, schedule_config):
        """Test adding a schedule."""
        job = scheduler.add_schedule(schedule_config)

        assert job is not None
        assert job.job_id is not None
        assert job.config == schedule_config

    def test_add_schedule_generates_id(self, scheduler, schedule_config):
        """Test that adding schedule generates unique ID."""
        job1 = scheduler.add_schedule(schedule_config)
        job2 = scheduler.add_schedule(schedule_config)

        assert job1.job_id != job2.job_id

    def test_get_job(self, scheduler, schedule_config):
        """Test retrieving a job by ID."""
        job = scheduler.add_schedule(schedule_config)

        retrieved = scheduler.get_job(job.job_id)

        assert retrieved is not None
        assert retrieved.job_id == job.job_id

    def test_get_nonexistent_job(self, scheduler):
        """Test getting non-existent job returns None."""
        result = scheduler.get_job("nonexistent")

        assert result is None

    def test_list_jobs(self, scheduler, schedule_config):
        """Test listing all jobs."""
        scheduler.add_schedule(schedule_config)
        scheduler.add_schedule(schedule_config)

        jobs = scheduler.list_jobs()

        assert len(jobs) == 2

    def test_list_jobs_by_workspace(self, scheduler):
        """Test listing jobs filtered by workspace."""
        config1 = ScheduleConfig(
            name="Job 1",
            cron="0 0 * * *",
            workspace_id="ws-1",
            preset="security",
        )
        config2 = ScheduleConfig(
            name="Job 2",
            cron="0 0 * * *",
            workspace_id="ws-2",
            preset="security",
        )

        scheduler.add_schedule(config1)
        scheduler.add_schedule(config2)

        ws1_jobs = scheduler.list_jobs(workspace_id="ws-1")

        assert len(ws1_jobs) == 1
        assert ws1_jobs[0].config.workspace_id == "ws-1"

    def test_remove_schedule(self, scheduler, schedule_config):
        """Test removing a schedule."""
        job = scheduler.add_schedule(schedule_config)

        result = scheduler.remove_schedule(job.job_id)

        assert result is True
        assert scheduler.get_job(job.job_id) is None

    def test_pause_schedule(self, scheduler, schedule_config):
        """Test pausing a schedule."""
        job = scheduler.add_schedule(schedule_config)

        result = scheduler.pause_schedule(job.job_id)

        assert result is True
        paused = scheduler.get_job(job.job_id)
        assert paused.status == ScheduleStatus.PAUSED

    def test_resume_schedule(self, scheduler, schedule_config):
        """Test resuming a paused schedule."""
        job = scheduler.add_schedule(schedule_config)
        scheduler.pause_schedule(job.job_id)

        result = scheduler.resume_schedule(job.job_id)

        assert result is True
        resumed = scheduler.get_job(job.job_id)
        assert resumed.status == ScheduleStatus.ACTIVE


# ============================================================================
# Cron Expression Tests
# ============================================================================


class TestCronExpressions:
    """Tests for cron expression handling."""

    def test_parse_daily_cron(self):
        """Test parsing daily cron expression."""
        # Daily at midnight
        next_run = CronParser.next_run("0 0 * * *")
        assert next_run is not None
        assert isinstance(next_run, datetime)

    def test_parse_hourly_cron(self):
        """Test parsing hourly cron expression."""
        next_run = CronParser.next_run("0 * * * *")
        assert next_run is not None

    def test_parse_weekly_cron(self):
        """Test parsing weekly cron expression."""
        # Weekly on Monday at midnight
        next_run = CronParser.next_run("0 0 * * 1")
        assert next_run is not None

    def test_next_run_in_future(self):
        """Test that next run is in the future."""
        next_run = CronParser.next_run("0 0 * * *")
        if next_run:
            # Should be within 24 hours for daily schedule
            now = datetime.utcnow()
            assert next_run > now
            assert next_run < now + timedelta(days=2)


# ============================================================================
# Scheduler Lifecycle Tests
# ============================================================================


class TestSchedulerLifecycle:
    """Tests for scheduler lifecycle management."""

    @pytest.mark.asyncio
    async def test_start_scheduler(self, scheduler):
        """Test starting the scheduler."""
        # Start the scheduler (async method)
        await scheduler.start()

        # Give it a moment to start
        import asyncio
        await asyncio.sleep(0.1)

        assert scheduler._running is True

        # Clean up
        await scheduler.stop()

    @pytest.mark.asyncio
    async def test_stop_scheduler(self, scheduler):
        """Test stopping the scheduler."""
        await scheduler.start()
        import asyncio
        await asyncio.sleep(0.1)

        await scheduler.stop()

        assert scheduler._running is False

    def test_list_jobs_by_status(self, scheduler, schedule_config):
        """Test listing jobs filtered by status."""
        job1 = scheduler.add_schedule(schedule_config)
        job2 = scheduler.add_schedule(schedule_config)
        scheduler.pause_schedule(job2.job_id)

        active_jobs = scheduler.list_jobs(status=ScheduleStatus.ACTIVE)
        paused_jobs = scheduler.list_jobs(status=ScheduleStatus.PAUSED)

        assert len(active_jobs) == 1
        assert len(paused_jobs) == 1
