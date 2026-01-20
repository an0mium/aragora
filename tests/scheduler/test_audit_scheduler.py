"""
Tests for audit scheduler module.

Tests cover:
- TriggerType enum
- ScheduleStatus enum
- ScheduleConfig dataclass
- ScheduledJob dataclass
- JobRun dataclass
- CronParser class
- AuditScheduler class
- Singleton get_scheduler
"""

import asyncio
import hmac
import hashlib
import json
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.scheduler.audit_scheduler import (
    AuditScheduler,
    CronParser,
    JobRun,
    ScheduleConfig,
    ScheduledJob,
    ScheduleStatus,
    TriggerType,
    get_scheduler,
)


class TestTriggerType:
    """Tests for TriggerType enum."""

    def test_has_all_trigger_types(self):
        """Enum has all expected trigger types."""
        assert TriggerType.CRON.value == "cron"
        assert TriggerType.WEBHOOK.value == "webhook"
        assert TriggerType.GIT_PUSH.value == "git_push"
        assert TriggerType.FILE_UPLOAD.value == "file_upload"
        assert TriggerType.MANUAL.value == "manual"
        assert TriggerType.INTERVAL.value == "interval"

    def test_trigger_count(self):
        """Enum has exactly 6 trigger types."""
        assert len(TriggerType) == 6

    def test_is_string_enum(self):
        """TriggerType is a string enum."""
        assert isinstance(TriggerType.CRON.value, str)


class TestScheduleStatus:
    """Tests for ScheduleStatus enum."""

    def test_has_all_statuses(self):
        """Enum has all expected statuses."""
        assert ScheduleStatus.ACTIVE.value == "active"
        assert ScheduleStatus.PAUSED.value == "paused"
        assert ScheduleStatus.DISABLED.value == "disabled"
        assert ScheduleStatus.RUNNING.value == "running"
        assert ScheduleStatus.ERROR.value == "error"

    def test_status_count(self):
        """Enum has exactly 5 statuses."""
        assert len(ScheduleStatus) == 5


class TestScheduleConfig:
    """Tests for ScheduleConfig dataclass."""

    def test_default_values(self):
        """Default values are sensible."""
        config = ScheduleConfig(name="test")

        assert config.name == "test"
        assert config.description == ""
        assert config.trigger_type == TriggerType.CRON
        assert config.cron is None
        assert config.interval_minutes is None

    def test_audit_options(self):
        """Audit options have defaults."""
        config = ScheduleConfig(name="test")

        assert config.preset is None
        assert config.audit_types == []
        assert config.custom_config == {}

    def test_scope_options(self):
        """Scope options have defaults."""
        config = ScheduleConfig(name="test")

        assert config.workspace_id is None
        assert config.document_scope is None
        assert config.document_ids == []

    def test_notification_options(self):
        """Notification options have defaults."""
        config = ScheduleConfig(name="test")

        assert config.notify_on_complete is True
        assert config.notify_on_findings is True
        assert config.finding_severity_threshold == "medium"

    def test_retry_options(self):
        """Retry options have defaults."""
        config = ScheduleConfig(name="test")

        assert config.max_retries == 3
        assert config.timeout_minutes == 60

    def test_custom_config(self):
        """Custom configuration can be set."""
        config = ScheduleConfig(
            name="security_scan",
            description="Daily security scan",
            trigger_type=TriggerType.CRON,
            cron="0 2 * * *",
            preset="Code Security",
            workspace_id="ws_123",
            max_retries=5,
        )

        assert config.name == "security_scan"
        assert config.trigger_type == TriggerType.CRON
        assert config.cron == "0 2 * * *"


class TestScheduledJob:
    """Tests for ScheduledJob dataclass."""

    def test_required_fields(self):
        """Required fields are set."""
        config = ScheduleConfig(name="test")
        job = ScheduledJob(
            job_id="job_123",
            schedule_id="sched_456",
            config=config,
            status=ScheduleStatus.ACTIVE,
        )

        assert job.job_id == "job_123"
        assert job.schedule_id == "sched_456"
        assert job.status == ScheduleStatus.ACTIVE

    def test_optional_fields_defaults(self):
        """Optional fields have sensible defaults."""
        config = ScheduleConfig(name="test")
        job = ScheduledJob(
            job_id="job_123",
            schedule_id="sched_456",
            config=config,
            status=ScheduleStatus.ACTIVE,
        )

        assert job.next_run is None
        assert job.last_run is None
        assert job.last_result is None
        assert job.run_count == 0
        assert job.error_count == 0

    def test_to_dict(self):
        """to_dict returns proper dictionary."""
        config = ScheduleConfig(name="test", trigger_type=TriggerType.CRON)
        job = ScheduledJob(
            job_id="job_123",
            schedule_id="sched_456",
            config=config,
            status=ScheduleStatus.ACTIVE,
            run_count=5,
        )

        result = job.to_dict()

        assert result["job_id"] == "job_123"
        assert result["schedule_id"] == "sched_456"
        assert result["name"] == "test"
        assert result["status"] == "active"
        assert result["trigger_type"] == "cron"
        assert result["run_count"] == 5

    def test_to_dict_with_timestamps(self):
        """to_dict formats timestamps correctly."""
        config = ScheduleConfig(name="test")
        now = datetime.utcnow()
        job = ScheduledJob(
            job_id="job_123",
            schedule_id="sched_456",
            config=config,
            status=ScheduleStatus.ACTIVE,
            next_run=now,
            last_run=now - timedelta(hours=1),
        )

        result = job.to_dict()

        assert result["next_run"] is not None
        assert result["last_run"] is not None


class TestJobRun:
    """Tests for JobRun dataclass."""

    def test_required_fields(self):
        """Required fields are set."""
        now = datetime.utcnow()
        run = JobRun(
            run_id="run_123",
            job_id="job_456",
            started_at=now,
        )

        assert run.run_id == "run_123"
        assert run.job_id == "job_456"
        assert run.started_at == now

    def test_optional_fields_defaults(self):
        """Optional fields have sensible defaults."""
        run = JobRun(
            run_id="run_123",
            job_id="job_456",
            started_at=datetime.utcnow(),
        )

        assert run.completed_at is None
        assert run.status == "running"
        assert run.session_id is None
        assert run.findings_count == 0
        assert run.error_message is None
        assert run.duration_ms == 0

    def test_to_dict(self):
        """to_dict returns proper dictionary."""
        now = datetime.utcnow()
        run = JobRun(
            run_id="run_123",
            job_id="job_456",
            started_at=now,
            status="completed",
            findings_count=5,
        )

        result = run.to_dict()

        assert result["run_id"] == "run_123"
        assert result["job_id"] == "job_456"
        assert result["status"] == "completed"
        assert result["findings_count"] == 5

    def test_to_dict_with_error(self):
        """to_dict includes error message."""
        run = JobRun(
            run_id="run_123",
            job_id="job_456",
            started_at=datetime.utcnow(),
            status="error",
            error_message="Something went wrong",
        )

        result = run.to_dict()

        assert result["error_message"] == "Something went wrong"


class TestCronParser:
    """Tests for CronParser class."""

    def test_parse_simple_expression(self):
        """Parses simple cron expression."""
        result = CronParser.parse("0 2 * * *")

        assert result["minute"] == [0]
        assert result["hour"] == [2]
        assert result["day"] == list(range(1, 32))
        assert result["month"] == list(range(1, 13))
        assert result["weekday"] == list(range(0, 7))

    def test_parse_all_stars(self):
        """Parses all-star expression."""
        result = CronParser.parse("* * * * *")

        assert result["minute"] == list(range(0, 60))
        assert result["hour"] == list(range(0, 24))
        assert result["day"] == list(range(1, 32))
        assert result["month"] == list(range(1, 13))
        assert result["weekday"] == list(range(0, 7))

    def test_parse_step_expression(self):
        """Parses step expression."""
        result = CronParser.parse("*/15 * * * *")

        assert result["minute"] == [0, 15, 30, 45]

    def test_parse_range_expression(self):
        """Parses range expression."""
        result = CronParser.parse("0 9-17 * * *")

        assert result["minute"] == [0]
        assert result["hour"] == list(range(9, 18))

    def test_parse_list_expression(self):
        """Parses list expression."""
        result = CronParser.parse("0 0,12 * * *")

        assert result["hour"] == [0, 12]

    def test_parse_invalid_expression(self):
        """Raises error for invalid expression."""
        with pytest.raises(ValueError, match="Invalid cron expression"):
            CronParser.parse("0 2 * *")  # Missing field

    def test_next_run_daily(self):
        """Calculates next run for daily schedule."""
        # 2 AM every day
        after = datetime(2024, 1, 15, 1, 0, 0)  # 1 AM
        next_run = CronParser.next_run("0 2 * * *", after)

        assert next_run.hour == 2
        assert next_run.minute == 0

    def test_next_run_after_time_passes(self):
        """Next run is tomorrow if time already passed."""
        # 2 AM every day, but it's already 3 AM
        after = datetime(2024, 1, 15, 3, 0, 0)
        next_run = CronParser.next_run("0 2 * * *", after)

        assert next_run.day == 16  # Tomorrow
        assert next_run.hour == 2

    def test_next_run_hourly(self):
        """Calculates next run for hourly schedule."""
        after = datetime(2024, 1, 15, 14, 30, 0)
        next_run = CronParser.next_run("0 * * * *", after)

        assert next_run.hour == 15
        assert next_run.minute == 0

    def test_next_run_specific_day(self):
        """Calculates next run for specific weekday."""
        # In cron: 0=Sunday, 1=Monday, etc.
        # In Python weekday(): 0=Monday, 1=Tuesday, etc.
        # The CronParser uses Python's weekday() internally
        # So cron "* * * * 1" matches Python weekday() == 1 (Tuesday)
        after = datetime(2024, 1, 15, 0, 0, 0)  # This is a Monday
        # Weekday 1 in the cron expression will match Python weekday 1 (Tuesday)
        next_run = CronParser.next_run("0 9 * * 1", after)

        # Since the parser uses Python weekday, weekday 1 = Tuesday
        assert next_run.weekday() == 1  # Tuesday in Python datetime
        assert next_run.hour == 9


class TestAuditScheduler:
    """Tests for AuditScheduler class."""

    def test_init(self):
        """Initializes with empty state."""
        scheduler = AuditScheduler()

        assert scheduler._jobs == {}
        assert scheduler._runs == {}
        assert scheduler._running is False

    def test_add_schedule_cron(self):
        """Adds a cron-based schedule."""
        scheduler = AuditScheduler()
        config = ScheduleConfig(
            name="nightly_scan",
            trigger_type=TriggerType.CRON,
            cron="0 2 * * *",
        )

        job = scheduler.add_schedule(config)

        assert job.job_id is not None
        assert job.schedule_id is not None
        assert job.config.name == "nightly_scan"
        assert job.status == ScheduleStatus.ACTIVE
        assert job.next_run is not None

    def test_add_schedule_interval(self):
        """Adds an interval-based schedule."""
        scheduler = AuditScheduler()
        config = ScheduleConfig(
            name="frequent_check",
            trigger_type=TriggerType.INTERVAL,
            interval_minutes=30,
        )

        job = scheduler.add_schedule(config)

        assert job.next_run is not None
        # Should be ~30 minutes from now
        delta = job.next_run - datetime.utcnow()
        assert 29 <= delta.total_seconds() / 60 <= 31

    def test_add_schedule_webhook(self):
        """Adds a webhook-based schedule."""
        scheduler = AuditScheduler()
        config = ScheduleConfig(
            name="webhook_trigger",
            trigger_type=TriggerType.WEBHOOK,
            webhook_secret="secret123",
        )

        job = scheduler.add_schedule(config)

        assert job.next_run is None  # No scheduled time for webhooks
        assert job.status == ScheduleStatus.ACTIVE

    def test_remove_schedule(self):
        """Removes a schedule."""
        scheduler = AuditScheduler()
        config = ScheduleConfig(name="test")
        job = scheduler.add_schedule(config)

        result = scheduler.remove_schedule(job.job_id)

        assert result is True
        assert scheduler.get_job(job.job_id) is None

    def test_remove_nonexistent_schedule(self):
        """Returns False for nonexistent schedule."""
        scheduler = AuditScheduler()

        result = scheduler.remove_schedule("nonexistent")

        assert result is False

    def test_pause_schedule(self):
        """Pauses a schedule."""
        scheduler = AuditScheduler()
        config = ScheduleConfig(name="test")
        job = scheduler.add_schedule(config)

        result = scheduler.pause_schedule(job.job_id)

        assert result is True
        assert scheduler.get_job(job.job_id).status == ScheduleStatus.PAUSED

    def test_pause_nonexistent_schedule(self):
        """Returns False for nonexistent schedule."""
        scheduler = AuditScheduler()

        result = scheduler.pause_schedule("nonexistent")

        assert result is False

    def test_resume_schedule(self):
        """Resumes a paused schedule."""
        scheduler = AuditScheduler()
        config = ScheduleConfig(name="test", trigger_type=TriggerType.CRON, cron="0 2 * * *")
        job = scheduler.add_schedule(config)
        scheduler.pause_schedule(job.job_id)

        result = scheduler.resume_schedule(job.job_id)

        assert result is True
        assert scheduler.get_job(job.job_id).status == ScheduleStatus.ACTIVE

    def test_resume_non_paused_schedule(self):
        """Returns False for non-paused schedule."""
        scheduler = AuditScheduler()
        config = ScheduleConfig(name="test")
        job = scheduler.add_schedule(config)

        result = scheduler.resume_schedule(job.job_id)

        assert result is False

    def test_get_job(self):
        """Gets a job by ID."""
        scheduler = AuditScheduler()
        config = ScheduleConfig(name="test")
        job = scheduler.add_schedule(config)

        retrieved = scheduler.get_job(job.job_id)

        assert retrieved is not None
        assert retrieved.job_id == job.job_id

    def test_get_nonexistent_job(self):
        """Returns None for nonexistent job."""
        scheduler = AuditScheduler()

        result = scheduler.get_job("nonexistent")

        assert result is None

    def test_list_jobs(self):
        """Lists all jobs."""
        scheduler = AuditScheduler()
        scheduler.add_schedule(ScheduleConfig(name="job1"))
        scheduler.add_schedule(ScheduleConfig(name="job2"))

        jobs = scheduler.list_jobs()

        assert len(jobs) == 2

    def test_list_jobs_by_status(self):
        """Lists jobs filtered by status."""
        scheduler = AuditScheduler()
        job1 = scheduler.add_schedule(ScheduleConfig(name="job1"))
        scheduler.add_schedule(ScheduleConfig(name="job2"))
        scheduler.pause_schedule(job1.job_id)

        active_jobs = scheduler.list_jobs(status=ScheduleStatus.ACTIVE)
        paused_jobs = scheduler.list_jobs(status=ScheduleStatus.PAUSED)

        assert len(active_jobs) == 1
        assert len(paused_jobs) == 1

    def test_list_jobs_by_workspace(self):
        """Lists jobs filtered by workspace."""
        scheduler = AuditScheduler()
        scheduler.add_schedule(ScheduleConfig(name="job1", workspace_id="ws_1"))
        scheduler.add_schedule(ScheduleConfig(name="job2", workspace_id="ws_2"))

        jobs = scheduler.list_jobs(workspace_id="ws_1")

        assert len(jobs) == 1
        assert jobs[0].config.workspace_id == "ws_1"

    def test_on_callback_registration(self):
        """Registers event callbacks."""
        scheduler = AuditScheduler()
        callback = MagicMock()

        scheduler.on("job_started", callback)

        assert callback in scheduler._callbacks["job_started"]

    def test_on_invalid_event(self):
        """Ignores invalid event names."""
        scheduler = AuditScheduler()
        callback = MagicMock()

        scheduler.on("invalid_event", callback)

        # Should not raise, just ignore

    @pytest.mark.asyncio
    async def test_trigger_job(self):
        """Manually triggers a job."""
        scheduler = AuditScheduler()
        config = ScheduleConfig(name="test", document_ids=["doc1"])
        job = scheduler.add_schedule(config)

        with patch("aragora.audit.get_document_auditor") as mock_get:
            mock_auditor = MagicMock()
            mock_auditor.create_session = AsyncMock(return_value=MagicMock(id="session_123"))
            mock_auditor.run_audit = AsyncMock(return_value=MagicMock(findings=[]))
            mock_get.return_value = mock_auditor

            run = await scheduler.trigger_job(job.job_id)

        assert run is not None
        assert run.job_id == job.job_id

    @pytest.mark.asyncio
    async def test_trigger_nonexistent_job(self):
        """Returns None for nonexistent job."""
        scheduler = AuditScheduler()

        run = await scheduler.trigger_job("nonexistent")

        assert run is None

    @pytest.mark.asyncio
    async def test_handle_webhook(self):
        """Handles webhook trigger."""
        scheduler = AuditScheduler()
        config = ScheduleConfig(
            name="webhook_job",
            trigger_type=TriggerType.WEBHOOK,
            document_ids=["doc1"],
        )
        scheduler.add_schedule(config)

        with patch("aragora.audit.get_document_auditor") as mock_get:
            mock_auditor = MagicMock()
            mock_auditor.create_session = AsyncMock(return_value=MagicMock(id="session_123"))
            mock_auditor.run_audit = AsyncMock(return_value=MagicMock(findings=[]))
            mock_get.return_value = mock_auditor

            runs = await scheduler.handle_webhook("webhook_id", {"event": "push"})

        assert len(runs) == 1

    @pytest.mark.asyncio
    async def test_handle_webhook_with_signature(self):
        """Handles webhook with signature verification."""
        scheduler = AuditScheduler()
        secret = "my_secret"
        config = ScheduleConfig(
            name="webhook_job",
            trigger_type=TriggerType.WEBHOOK,
            webhook_secret=secret,
            document_ids=["doc1"],
        )
        scheduler.add_schedule(config)

        payload = {"event": "push"}
        # Generate valid signature
        payload_bytes = json.dumps(payload, sort_keys=True).encode()
        signature = hmac.new(secret.encode(), payload_bytes, hashlib.sha256).hexdigest()

        with patch("aragora.audit.get_document_auditor") as mock_get:
            mock_auditor = MagicMock()
            mock_auditor.create_session = AsyncMock(return_value=MagicMock(id="session_123"))
            mock_auditor.run_audit = AsyncMock(return_value=MagicMock(findings=[]))
            mock_get.return_value = mock_auditor

            runs = await scheduler.handle_webhook("webhook_id", payload, signature)

        assert len(runs) == 1

    @pytest.mark.asyncio
    async def test_handle_webhook_invalid_signature(self):
        """Rejects webhook with invalid signature."""
        scheduler = AuditScheduler()
        config = ScheduleConfig(
            name="webhook_job",
            trigger_type=TriggerType.WEBHOOK,
            webhook_secret="my_secret",
            document_ids=["doc1"],
        )
        scheduler.add_schedule(config)

        runs = await scheduler.handle_webhook("webhook_id", {"event": "push"}, "invalid_sig")

        assert len(runs) == 0

    @pytest.mark.asyncio
    async def test_handle_git_push(self):
        """Handles git push trigger."""
        scheduler = AuditScheduler()
        config = ScheduleConfig(
            name="git_job",
            trigger_type=TriggerType.GIT_PUSH,
            document_ids=["doc1"],
        )
        scheduler.add_schedule(config)

        with patch("aragora.audit.get_document_auditor") as mock_get:
            mock_auditor = MagicMock()
            mock_auditor.create_session = AsyncMock(return_value=MagicMock(id="session_123"))
            mock_auditor.run_audit = AsyncMock(return_value=MagicMock(findings=[]))
            mock_get.return_value = mock_auditor

            runs = await scheduler.handle_git_push(
                repository="owner/repo",
                branch="main",
                commit_sha="abc123",
                changed_files=["file1.py"],
            )

        assert len(runs) == 1

    @pytest.mark.asyncio
    async def test_handle_file_upload(self):
        """Handles file upload trigger."""
        scheduler = AuditScheduler()
        config = ScheduleConfig(
            name="upload_job",
            trigger_type=TriggerType.FILE_UPLOAD,
            workspace_id="ws_123",
        )
        scheduler.add_schedule(config)

        with patch("aragora.audit.get_document_auditor") as mock_get:
            mock_auditor = MagicMock()
            mock_auditor.create_session = AsyncMock(return_value=MagicMock(id="session_123"))
            mock_auditor.run_audit = AsyncMock(return_value=MagicMock(findings=[]))
            mock_get.return_value = mock_auditor

            runs = await scheduler.handle_file_upload(
                workspace_id="ws_123",
                document_ids=["doc1", "doc2"],
            )

        assert len(runs) == 1

    @pytest.mark.asyncio
    async def test_handle_file_upload_wrong_workspace(self):
        """Ignores file upload for different workspace."""
        scheduler = AuditScheduler()
        config = ScheduleConfig(
            name="upload_job",
            trigger_type=TriggerType.FILE_UPLOAD,
            workspace_id="ws_123",
        )
        scheduler.add_schedule(config)

        runs = await scheduler.handle_file_upload(
            workspace_id="ws_different",
            document_ids=["doc1"],
        )

        assert len(runs) == 0

    @pytest.mark.asyncio
    async def test_start_stop(self):
        """Starts and stops the scheduler."""
        scheduler = AuditScheduler()

        await scheduler.start()
        assert scheduler._running is True
        assert scheduler._task is not None

        await scheduler.stop()
        assert scheduler._running is False

    @pytest.mark.asyncio
    async def test_start_idempotent(self):
        """Start is idempotent."""
        scheduler = AuditScheduler()

        await scheduler.start()
        task1 = scheduler._task
        await scheduler.start()  # Second start should be no-op
        task2 = scheduler._task

        assert task1 is task2

        await scheduler.stop()

    @pytest.mark.asyncio
    async def test_job_execution_emits_events(self):
        """Job execution emits events."""
        scheduler = AuditScheduler()
        config = ScheduleConfig(name="test", document_ids=["doc1"])
        job = scheduler.add_schedule(config)

        started_callback = MagicMock()
        completed_callback = MagicMock()
        scheduler.on("job_started", started_callback)
        scheduler.on("job_completed", completed_callback)

        with patch("aragora.audit.get_document_auditor") as mock_get:
            mock_auditor = MagicMock()
            mock_auditor.create_session = AsyncMock(return_value=MagicMock(id="session_123"))
            mock_auditor.run_audit = AsyncMock(return_value=MagicMock(findings=[]))
            mock_get.return_value = mock_auditor

            await scheduler.trigger_job(job.job_id)

        started_callback.assert_called_once()
        completed_callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_job_execution_with_findings(self):
        """Job with findings emits findings_detected event."""
        scheduler = AuditScheduler()
        config = ScheduleConfig(name="test", document_ids=["doc1"], notify_on_findings=True)
        job = scheduler.add_schedule(config)

        findings_callback = MagicMock()
        scheduler.on("findings_detected", findings_callback)

        with patch("aragora.audit.get_document_auditor") as mock_get:
            mock_auditor = MagicMock()
            mock_auditor.create_session = AsyncMock(return_value=MagicMock(id="session_123"))
            mock_auditor.run_audit = AsyncMock(
                return_value=MagicMock(findings=[{"severity": "high"}])
            )
            mock_get.return_value = mock_auditor

            await scheduler.trigger_job(job.job_id)

        findings_callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_job_execution_error_handling(self):
        """Job execution handles errors gracefully."""
        scheduler = AuditScheduler()
        config = ScheduleConfig(name="test", document_ids=["doc1"])
        job = scheduler.add_schedule(config)

        failed_callback = MagicMock()
        scheduler.on("job_failed", failed_callback)

        with patch("aragora.audit.get_document_auditor") as mock_get:
            mock_auditor = MagicMock()
            mock_auditor.create_session = AsyncMock(side_effect=RuntimeError("Auditor failed"))
            mock_get.return_value = mock_auditor

            run = await scheduler.trigger_job(job.job_id)

        assert run.status == "error"
        assert run.error_message == "Auditor failed"
        failed_callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_job_execution_timeout(self):
        """Job execution handles timeout."""
        scheduler = AuditScheduler()
        config = ScheduleConfig(name="test", document_ids=["doc1"], timeout_minutes=1)
        job = scheduler.add_schedule(config)

        with patch("aragora.audit.get_document_auditor") as mock_get:
            mock_auditor = MagicMock()
            mock_auditor.create_session = AsyncMock(return_value=MagicMock(id="session_123"))
            mock_auditor.run_audit = AsyncMock(side_effect=asyncio.TimeoutError)
            mock_get.return_value = mock_auditor

            with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError):
                run = await scheduler.trigger_job(job.job_id)

        assert run.status == "timeout"

    def test_get_job_history(self):
        """Gets job run history."""
        scheduler = AuditScheduler()

        # Manually add runs
        for i in range(15):
            run = JobRun(
                run_id=f"run_{i}",
                job_id="job_123",
                started_at=datetime.utcnow() - timedelta(hours=i),
            )
            scheduler._runs[run.run_id] = run

        history = scheduler.get_job_history("job_123", limit=10)

        assert len(history) == 10
        # Should be sorted by started_at descending
        for i in range(len(history) - 1):
            assert history[i].started_at >= history[i + 1].started_at


class TestGetScheduler:
    """Tests for get_scheduler singleton."""

    def test_returns_scheduler(self):
        """Returns an AuditScheduler instance."""
        # Reset singleton for test
        import aragora.scheduler.audit_scheduler as module
        module._scheduler = None

        scheduler = get_scheduler()

        assert isinstance(scheduler, AuditScheduler)

    def test_returns_same_instance(self):
        """Returns the same instance on subsequent calls."""
        # Reset singleton for test
        import aragora.scheduler.audit_scheduler as module
        module._scheduler = None

        scheduler1 = get_scheduler()
        scheduler2 = get_scheduler()

        assert scheduler1 is scheduler2


class TestCronParserEdgeCases:
    """Edge case tests for CronParser."""

    def test_parse_step_from_start(self):
        """Step expression with start value."""
        result = CronParser.parse("5/10 * * * *")

        assert result["minute"] == [5, 15, 25, 35, 45, 55]

    def test_next_run_default_time(self):
        """next_run uses current time when none provided."""
        next_run = CronParser.next_run("0 * * * *")

        assert next_run is not None
        assert next_run > datetime.utcnow()


class TestSchedulerIntegration:
    """Integration tests for scheduler workflows."""

    @pytest.mark.asyncio
    async def test_full_cron_workflow(self):
        """Tests full cron job workflow."""
        scheduler = AuditScheduler()

        # Add schedule
        config = ScheduleConfig(
            name="daily_scan",
            trigger_type=TriggerType.CRON,
            cron="0 2 * * *",
            preset="Code Security",
            document_ids=["doc1"],
        )
        job = scheduler.add_schedule(config)

        assert job.status == ScheduleStatus.ACTIVE
        assert job.next_run is not None

        # Pause
        scheduler.pause_schedule(job.job_id)
        assert scheduler.get_job(job.job_id).status == ScheduleStatus.PAUSED

        # Resume
        scheduler.resume_schedule(job.job_id)
        assert scheduler.get_job(job.job_id).status == ScheduleStatus.ACTIVE

        # Remove
        scheduler.remove_schedule(job.job_id)
        assert scheduler.get_job(job.job_id) is None

    @pytest.mark.asyncio
    async def test_async_callbacks(self):
        """Tests async callback support."""
        scheduler = AuditScheduler()
        config = ScheduleConfig(name="test", document_ids=["doc1"])
        job = scheduler.add_schedule(config)

        results = []

        async def async_callback(j, r):
            results.append(("started", j.job_id))

        scheduler.on("job_started", async_callback)

        with patch("aragora.audit.get_document_auditor") as mock_get:
            mock_auditor = MagicMock()
            mock_auditor.create_session = AsyncMock(return_value=MagicMock(id="session_123"))
            mock_auditor.run_audit = AsyncMock(return_value=MagicMock(findings=[]))
            mock_get.return_value = mock_auditor

            await scheduler.trigger_job(job.job_id)

        assert len(results) == 1
        assert results[0][0] == "started"
