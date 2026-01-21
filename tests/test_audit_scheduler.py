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
    auditor.start_session = AsyncMock(
        return_value=Mock(
            id="session-123",
            status="completed",
            findings=[],
        )
    )
    auditor.run_audit = AsyncMock(
        return_value=Mock(
            id="session-123",
            status="completed",
            findings=[Mock(id="f-1", severity="high")],
        )
    )
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
            now = datetime.now(timezone.utc)
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


# ============================================================================
# Cron Parsing Edge Cases Tests
# ============================================================================


class TestCronParsingEdgeCases:
    """Tests for cron expression edge cases."""

    def test_parse_step_expression(self):
        """Test parsing step expressions like */15."""
        parsed = CronParser.parse("*/15 * * * *")

        # Every 15 minutes: 0, 15, 30, 45
        assert parsed["minute"] == [0, 15, 30, 45]

    def test_parse_range_expression(self):
        """Test parsing range expressions like 1-5."""
        parsed = CronParser.parse("0 9-17 * * *")

        # Hours 9 through 17
        assert parsed["hour"] == [9, 10, 11, 12, 13, 14, 15, 16, 17]

    def test_parse_list_expression(self):
        """Test parsing list expressions like 1,15,30."""
        parsed = CronParser.parse("0,15,30,45 * * * *")

        assert parsed["minute"] == [0, 15, 30, 45]

    def test_parse_range_with_step(self):
        """Test parsing range with step like 0/10."""
        parsed = CronParser.parse("0/10 * * * *")

        # Starts at 0, steps by 10 until max (59)
        assert 0 in parsed["minute"]
        assert 10 in parsed["minute"]
        assert 20 in parsed["minute"]

    def test_parse_specific_value(self):
        """Test parsing specific value like 0."""
        parsed = CronParser.parse("30 14 * * *")

        assert parsed["minute"] == [30]
        assert parsed["hour"] == [14]

    def test_parse_weekday_range(self):
        """Test parsing weekday range (Monday-Friday)."""
        parsed = CronParser.parse("0 9 * * 1-5")

        # Monday(1) through Friday(5)
        assert parsed["weekday"] == [1, 2, 3, 4, 5]

    def test_invalid_cron_expression(self):
        """Test that invalid cron expression raises error."""
        with pytest.raises(ValueError) as exc_info:
            CronParser.parse("0 0 * *")  # Only 4 fields

        assert "Invalid cron expression" in str(exc_info.value)

    def test_complex_expression(self):
        """Test complex cron expression."""
        # At minute 0 and 30, at hours 9-17, on weekdays
        parsed = CronParser.parse("0,30 9-17 * * 1-5")

        assert parsed["minute"] == [0, 30]
        assert 9 in parsed["hour"]
        assert 17 in parsed["hour"]
        assert 1 in parsed["weekday"]


# ============================================================================
# Job Execution Tests
# ============================================================================


class TestJobExecution:
    """Tests for job execution flow."""

    @pytest.mark.asyncio
    async def test_execute_job_success(self, scheduler, schedule_config):
        """Test successful job execution."""
        job = scheduler.add_schedule(schedule_config)

        # Mock the auditor
        mock_session = Mock(id="session-123")
        mock_result = Mock(findings=[])

        with patch("aragora.audit.get_document_auditor") as mock_get:
            mock_auditor = Mock()
            mock_auditor.create_session = AsyncMock(return_value=mock_session)
            mock_auditor.run_audit = AsyncMock(return_value=mock_result)
            mock_get.return_value = mock_auditor

            run = await scheduler._execute_job(job)

        assert run is not None
        assert run.status == "completed"
        assert run.session_id == "session-123"
        assert job.run_count == 1

    @pytest.mark.asyncio
    async def test_execute_job_with_findings(self, scheduler, schedule_config):
        """Test job execution with findings."""
        job = scheduler.add_schedule(schedule_config)

        mock_session = Mock(id="session-456")
        mock_result = Mock(findings=[Mock(), Mock(), Mock()])

        with patch("aragora.audit.get_document_auditor") as mock_get:
            mock_auditor = Mock()
            mock_auditor.create_session = AsyncMock(return_value=mock_session)
            mock_auditor.run_audit = AsyncMock(return_value=mock_result)
            mock_get.return_value = mock_auditor

            run = await scheduler._execute_job(job)

        assert run.findings_count == 3

    @pytest.mark.asyncio
    async def test_execute_job_error(self, scheduler, schedule_config):
        """Test job execution with error."""
        job = scheduler.add_schedule(schedule_config)

        with patch("aragora.audit.get_document_auditor") as mock_get:
            mock_auditor = Mock()
            mock_auditor.create_session = AsyncMock(side_effect=Exception("Audit failed"))
            mock_get.return_value = mock_auditor

            run = await scheduler._execute_job(job)

        assert run.status == "error"
        assert "Audit failed" in run.error_message
        assert job.error_count == 1

    @pytest.mark.asyncio
    async def test_execute_job_error_status_transition(self, scheduler, schedule_config):
        """Test job status transitions to ERROR after max retries."""
        schedule_config.max_retries = 2
        job = scheduler.add_schedule(schedule_config)

        with patch("aragora.audit.get_document_auditor") as mock_get:
            mock_auditor = Mock()
            mock_auditor.create_session = AsyncMock(side_effect=Exception("Fail"))
            mock_get.return_value = mock_auditor

            # First failure
            await scheduler._execute_job(job)
            assert job.status == ScheduleStatus.ACTIVE  # Still active

            # Second failure
            await scheduler._execute_job(job)
            assert job.status == ScheduleStatus.ACTIVE  # Still active

            # Third failure (exceeds max_retries)
            await scheduler._execute_job(job)
            assert job.status == ScheduleStatus.ERROR

    @pytest.mark.asyncio
    async def test_execute_job_timeout(self, scheduler):
        """Test job execution timeout."""
        config = ScheduleConfig(
            name="Timeout Test",
            cron="0 0 * * *",
            timeout_minutes=0,  # Immediate timeout
        )
        job = scheduler.add_schedule(config)

        import asyncio

        async def slow_audit(*args, **kwargs):
            await asyncio.sleep(10)  # Longer than timeout
            return Mock(findings=[])

        with patch("aragora.audit.get_document_auditor") as mock_get:
            mock_auditor = Mock()
            mock_auditor.create_session = AsyncMock(return_value=Mock(id="s-1"))
            mock_auditor.run_audit = slow_audit
            mock_get.return_value = mock_auditor

            run = await scheduler._execute_job(job)

        assert run.status == "timeout"
        assert "timed out" in run.error_message

    @pytest.mark.asyncio
    async def test_execute_job_with_context(self, scheduler, schedule_config):
        """Test job execution with context override."""
        job = scheduler.add_schedule(schedule_config)

        mock_session = Mock(id="session-ctx")
        mock_result = Mock(findings=[])

        with patch("aragora.audit.get_document_auditor") as mock_get:
            mock_auditor = Mock()
            mock_auditor.create_session = AsyncMock(return_value=mock_session)
            mock_auditor.run_audit = AsyncMock(return_value=mock_result)
            mock_get.return_value = mock_auditor

            context = {"document_ids": ["doc-1", "doc-2"]}
            run = await scheduler._execute_job(job, context=context)

        assert run is not None
        # Verify create_session was called with overridden document_ids
        call_kwargs = mock_auditor.create_session.call_args[1]
        assert call_kwargs["document_ids"] == ["doc-1", "doc-2"]


# ============================================================================
# Trigger Job Tests
# ============================================================================


class TestTriggerJob:
    """Tests for manual job triggering."""

    @pytest.mark.asyncio
    async def test_trigger_job_success(self, scheduler, schedule_config):
        """Test manually triggering a job."""
        job = scheduler.add_schedule(schedule_config)

        with patch("aragora.audit.get_document_auditor") as mock_get:
            mock_auditor = Mock()
            mock_auditor.create_session = AsyncMock(return_value=Mock(id="s-1"))
            mock_auditor.run_audit = AsyncMock(return_value=Mock(findings=[]))
            mock_get.return_value = mock_auditor

            run = await scheduler.trigger_job(job.job_id)

        assert run is not None
        assert run.job_id == job.job_id

    @pytest.mark.asyncio
    async def test_trigger_nonexistent_job(self, scheduler):
        """Test triggering a non-existent job."""
        run = await scheduler.trigger_job("nonexistent-job")

        assert run is None


# ============================================================================
# Webhook Handler Tests
# ============================================================================


class TestWebhookHandling:
    """Tests for webhook trigger handling."""

    @pytest.fixture
    def webhook_config(self):
        """Create a webhook-triggered schedule."""
        return ScheduleConfig(
            name="Webhook Audit",
            trigger_type=TriggerType.WEBHOOK,
            webhook_secret="test-secret",
            workspace_id="ws-123",
        )

    @pytest.mark.asyncio
    async def test_handle_webhook_triggers_job(self, scheduler, webhook_config):
        """Test that webhook triggers matching jobs."""
        job = scheduler.add_schedule(webhook_config)

        with patch("aragora.audit.get_document_auditor") as mock_get:
            mock_auditor = Mock()
            mock_auditor.create_session = AsyncMock(return_value=Mock(id="s-1"))
            mock_auditor.run_audit = AsyncMock(return_value=Mock(findings=[]))
            mock_get.return_value = mock_auditor

            # Generate valid signature
            import hmac
            import hashlib
            import json

            payload = {"event": "push"}
            payload_bytes = json.dumps(payload, sort_keys=True).encode()
            signature = hmac.new(
                "test-secret".encode(),
                payload_bytes,
                hashlib.sha256,
            ).hexdigest()

            runs = await scheduler.handle_webhook("wh-1", payload, signature)

        assert len(runs) == 1
        assert runs[0].job_id == job.job_id

    @pytest.mark.asyncio
    async def test_handle_webhook_invalid_signature(self, scheduler, webhook_config):
        """Test that invalid signature rejects webhook."""
        scheduler.add_schedule(webhook_config)

        with patch("aragora.audit.get_document_auditor"):
            runs = await scheduler.handle_webhook(
                "wh-1",
                {"event": "push"},
                "invalid-signature",
            )

        assert len(runs) == 0

    @pytest.mark.asyncio
    async def test_handle_webhook_no_secret(self, scheduler):
        """Test webhook without secret configured."""
        config = ScheduleConfig(
            name="No Secret Webhook",
            trigger_type=TriggerType.WEBHOOK,
            # No webhook_secret
        )
        job = scheduler.add_schedule(config)

        with patch("aragora.audit.get_document_auditor") as mock_get:
            mock_auditor = Mock()
            mock_auditor.create_session = AsyncMock(return_value=Mock(id="s-1"))
            mock_auditor.run_audit = AsyncMock(return_value=Mock(findings=[]))
            mock_get.return_value = mock_auditor

            runs = await scheduler.handle_webhook("wh-1", {"event": "push"})

        # Should execute without signature verification
        assert len(runs) == 1

    @pytest.mark.asyncio
    async def test_handle_webhook_skips_paused_jobs(self, scheduler, webhook_config):
        """Test that paused jobs are not triggered by webhooks."""
        job = scheduler.add_schedule(webhook_config)
        scheduler.pause_schedule(job.job_id)

        with patch("aragora.audit.get_document_auditor"):
            runs = await scheduler.handle_webhook("wh-1", {"event": "push"})

        assert len(runs) == 0


# ============================================================================
# Git Push Handler Tests
# ============================================================================


class TestGitPushHandling:
    """Tests for Git push event handling."""

    @pytest.fixture
    def git_push_config(self):
        """Create a git push-triggered schedule."""
        return ScheduleConfig(
            name="Git Push Audit",
            trigger_type=TriggerType.GIT_PUSH,
            workspace_id="ws-git",
        )

    @pytest.mark.asyncio
    async def test_handle_git_push_triggers_job(self, scheduler, git_push_config):
        """Test that git push triggers matching jobs."""
        job = scheduler.add_schedule(git_push_config)

        with patch("aragora.audit.get_document_auditor") as mock_get:
            mock_auditor = Mock()
            mock_auditor.create_session = AsyncMock(return_value=Mock(id="s-git"))
            mock_auditor.run_audit = AsyncMock(return_value=Mock(findings=[]))
            mock_get.return_value = mock_auditor

            runs = await scheduler.handle_git_push(
                repository="org/repo",
                branch="main",
                commit_sha="abc123",
                changed_files=["src/main.py", "tests/test_main.py"],
            )

        assert len(runs) == 1
        assert runs[0].job_id == job.job_id

    @pytest.mark.asyncio
    async def test_handle_git_push_skips_paused(self, scheduler, git_push_config):
        """Test that paused jobs are skipped."""
        job = scheduler.add_schedule(git_push_config)
        scheduler.pause_schedule(job.job_id)

        with patch("aragora.audit.get_document_auditor"):
            runs = await scheduler.handle_git_push(
                repository="org/repo",
                branch="main",
                commit_sha="abc123",
                changed_files=[],
            )

        assert len(runs) == 0


# ============================================================================
# File Upload Handler Tests
# ============================================================================


class TestFileUploadHandling:
    """Tests for file upload event handling."""

    @pytest.fixture
    def file_upload_config(self):
        """Create a file upload-triggered schedule."""
        return ScheduleConfig(
            name="File Upload Audit",
            trigger_type=TriggerType.FILE_UPLOAD,
            workspace_id="ws-upload",
        )

    @pytest.mark.asyncio
    async def test_handle_file_upload_triggers_job(self, scheduler, file_upload_config):
        """Test that file upload triggers matching jobs."""
        job = scheduler.add_schedule(file_upload_config)

        with patch("aragora.audit.get_document_auditor") as mock_get:
            mock_auditor = Mock()
            mock_auditor.create_session = AsyncMock(return_value=Mock(id="s-upload"))
            mock_auditor.run_audit = AsyncMock(return_value=Mock(findings=[]))
            mock_get.return_value = mock_auditor

            runs = await scheduler.handle_file_upload(
                workspace_id="ws-upload",
                document_ids=["doc-1", "doc-2"],
            )

        assert len(runs) == 1

    @pytest.mark.asyncio
    async def test_handle_file_upload_filters_by_workspace(self, scheduler, file_upload_config):
        """Test that file upload filters by workspace."""
        scheduler.add_schedule(file_upload_config)

        with patch("aragora.audit.get_document_auditor"):
            # Different workspace
            runs = await scheduler.handle_file_upload(
                workspace_id="ws-other",
                document_ids=["doc-1"],
            )

        assert len(runs) == 0

    @pytest.mark.asyncio
    async def test_handle_file_upload_no_workspace_filter(self, scheduler):
        """Test file upload job without workspace filter."""
        config = ScheduleConfig(
            name="Any Workspace",
            trigger_type=TriggerType.FILE_UPLOAD,
            # No workspace_id - matches all
        )
        scheduler.add_schedule(config)

        with patch("aragora.audit.get_document_auditor") as mock_get:
            mock_auditor = Mock()
            mock_auditor.create_session = AsyncMock(return_value=Mock(id="s-any"))
            mock_auditor.run_audit = AsyncMock(return_value=Mock(findings=[]))
            mock_get.return_value = mock_auditor

            runs = await scheduler.handle_file_upload(
                workspace_id="any-workspace",
                document_ids=["doc-1"],
            )

        assert len(runs) == 1


# ============================================================================
# Event System Tests
# ============================================================================


class TestEventSystem:
    """Tests for event callback system."""

    def test_register_callback(self, scheduler):
        """Test registering an event callback."""
        callback = Mock()

        scheduler.on("job_started", callback)

        assert callback in scheduler._callbacks["job_started"]

    def test_register_multiple_callbacks(self, scheduler):
        """Test registering multiple callbacks for same event."""
        callback1 = Mock()
        callback2 = Mock()

        scheduler.on("job_completed", callback1)
        scheduler.on("job_completed", callback2)

        assert len(scheduler._callbacks["job_completed"]) == 2

    def test_register_invalid_event(self, scheduler):
        """Test registering callback for invalid event is ignored."""
        callback = Mock()

        scheduler.on("invalid_event", callback)

        # Should not crash, callback not added
        assert "invalid_event" not in scheduler._callbacks

    @pytest.mark.asyncio
    async def test_emit_sync_callback(self, scheduler, schedule_config):
        """Test emitting events to sync callbacks."""
        job = scheduler.add_schedule(schedule_config)
        callback = Mock()
        scheduler.on("job_started", callback)

        with patch("aragora.audit.get_document_auditor") as mock_get:
            mock_auditor = Mock()
            mock_auditor.create_session = AsyncMock(return_value=Mock(id="s-1"))
            mock_auditor.run_audit = AsyncMock(return_value=Mock(findings=[]))
            mock_get.return_value = mock_auditor

            await scheduler._execute_job(job)

        callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_emit_async_callback(self, scheduler, schedule_config):
        """Test emitting events to async callbacks."""
        job = scheduler.add_schedule(schedule_config)
        callback = AsyncMock()
        scheduler.on("job_completed", callback)

        with patch("aragora.audit.get_document_auditor") as mock_get:
            mock_auditor = Mock()
            mock_auditor.create_session = AsyncMock(return_value=Mock(id="s-1"))
            mock_auditor.run_audit = AsyncMock(return_value=Mock(findings=[]))
            mock_get.return_value = mock_auditor

            await scheduler._execute_job(job)

        callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_emit_callback_error_handled(self, scheduler, schedule_config):
        """Test that callback errors don't break execution."""
        job = scheduler.add_schedule(schedule_config)
        failing_callback = Mock(side_effect=Exception("Callback error"))
        scheduler.on("job_started", failing_callback)

        with patch("aragora.audit.get_document_auditor") as mock_get:
            mock_auditor = Mock()
            mock_auditor.create_session = AsyncMock(return_value=Mock(id="s-1"))
            mock_auditor.run_audit = AsyncMock(return_value=Mock(findings=[]))
            mock_get.return_value = mock_auditor

            # Should not raise
            run = await scheduler._execute_job(job)

        assert run is not None  # Job still completed

    @pytest.mark.asyncio
    async def test_findings_detected_event(self, scheduler, schedule_config):
        """Test findings_detected event is emitted."""
        job = scheduler.add_schedule(schedule_config)
        callback = Mock()
        scheduler.on("findings_detected", callback)

        mock_findings = [Mock(), Mock()]

        with patch("aragora.audit.get_document_auditor") as mock_get:
            mock_auditor = Mock()
            mock_auditor.create_session = AsyncMock(return_value=Mock(id="s-1"))
            mock_auditor.run_audit = AsyncMock(return_value=Mock(findings=mock_findings))
            mock_get.return_value = mock_auditor

            await scheduler._execute_job(job)

        callback.assert_called_once()


# ============================================================================
# Job History Tests
# ============================================================================


class TestJobHistory:
    """Tests for job run history."""

    @pytest.mark.asyncio
    async def test_get_job_history(self, scheduler, schedule_config):
        """Test retrieving job run history."""
        job = scheduler.add_schedule(schedule_config)

        with patch("aragora.audit.get_document_auditor") as mock_get:
            mock_auditor = Mock()
            mock_auditor.create_session = AsyncMock(return_value=Mock(id="s-1"))
            mock_auditor.run_audit = AsyncMock(return_value=Mock(findings=[]))
            mock_get.return_value = mock_auditor

            # Execute job multiple times
            await scheduler._execute_job(job)
            await scheduler._execute_job(job)

        history = scheduler.get_job_history(job.job_id)

        assert len(history) == 2

    @pytest.mark.asyncio
    async def test_get_job_history_limit(self, scheduler, schedule_config):
        """Test job history respects limit."""
        job = scheduler.add_schedule(schedule_config)

        with patch("aragora.audit.get_document_auditor") as mock_get:
            mock_auditor = Mock()
            mock_auditor.create_session = AsyncMock(return_value=Mock(id="s-1"))
            mock_auditor.run_audit = AsyncMock(return_value=Mock(findings=[]))
            mock_get.return_value = mock_auditor

            for _ in range(5):
                await scheduler._execute_job(job)

        history = scheduler.get_job_history(job.job_id, limit=3)

        assert len(history) == 3

    @pytest.mark.asyncio
    async def test_get_job_history_ordered(self, scheduler, schedule_config):
        """Test job history is ordered by most recent first."""
        job = scheduler.add_schedule(schedule_config)

        with patch("aragora.audit.get_document_auditor") as mock_get:
            mock_auditor = Mock()
            mock_auditor.create_session = AsyncMock(return_value=Mock(id="s-1"))
            mock_auditor.run_audit = AsyncMock(return_value=Mock(findings=[]))
            mock_get.return_value = mock_auditor

            await scheduler._execute_job(job)
            import asyncio

            await asyncio.sleep(0.01)  # Small delay
            await scheduler._execute_job(job)

        history = scheduler.get_job_history(job.job_id)

        # Most recent should be first
        assert history[0].started_at >= history[1].started_at


# ============================================================================
# Webhook Signature Verification Tests
# ============================================================================


class TestWebhookSignature:
    """Tests for webhook signature verification."""

    def test_verify_valid_signature(self, scheduler):
        """Test verifying a valid signature."""
        import hmac
        import hashlib
        import json

        payload = {"event": "push", "ref": "refs/heads/main"}
        secret = "webhook-secret"

        payload_bytes = json.dumps(payload, sort_keys=True).encode()
        signature = hmac.new(
            secret.encode(),
            payload_bytes,
            hashlib.sha256,
        ).hexdigest()

        result = scheduler._verify_webhook_signature(payload, signature, secret)

        assert result is True

    def test_verify_invalid_signature(self, scheduler):
        """Test rejecting an invalid signature."""
        payload = {"event": "push"}
        secret = "webhook-secret"

        result = scheduler._verify_webhook_signature(
            payload,
            "invalid-signature",
            secret,
        )

        assert result is False

    def test_verify_signature_wrong_secret(self, scheduler):
        """Test rejecting signature with wrong secret."""
        import hmac
        import hashlib
        import json

        payload = {"event": "push"}

        payload_bytes = json.dumps(payload, sort_keys=True).encode()
        signature = hmac.new(
            "wrong-secret".encode(),
            payload_bytes,
            hashlib.sha256,
        ).hexdigest()

        result = scheduler._verify_webhook_signature(
            payload,
            signature,
            "correct-secret",
        )

        assert result is False


# ============================================================================
# Singleton Tests
# ============================================================================


class TestSchedulerSingleton:
    """Tests for scheduler singleton."""

    def test_get_scheduler_returns_same_instance(self):
        """Test that get_scheduler returns singleton."""
        from aragora.scheduler.audit_scheduler import get_scheduler

        # Reset singleton for test
        import aragora.scheduler.audit_scheduler as module

        module._scheduler = None

        s1 = get_scheduler()
        s2 = get_scheduler()

        assert s1 is s2

        # Clean up
        module._scheduler = None
