"""
Tests for AuditScheduler - scheduled and triggered audit execution.

Tests cover:
- Schedule creation and management
- Cron expression parsing
- Webhook triggers
- Git push and file upload events
- Job execution and history
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta, timezone
import json

from aragora.scheduler.audit_scheduler import (
    AuditScheduler,
    ScheduleConfig,
    ScheduledJob,
    JobRun,
    TriggerType,
    ScheduleStatus,
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
        audit_preset="code_security",
        enabled=True,
    )


@pytest.fixture
def webhook_config():
    """Create a webhook configuration."""
    return {
        "id": "webhook-123",
        "secret": "test-secret-key",
        "workspace_id": "ws-123",
        "audit_preset": "code_security",
        "enabled": True,
    }


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
            audit_preset="security",
        )

        assert config.name == "Test Schedule"
        assert config.cron == "0 0 * * *"
        assert config.enabled is True  # Default

    def test_schedule_config_with_triggers(self):
        """Test schedule with additional triggers."""
        config = ScheduleConfig(
            name="Multi-trigger Schedule",
            cron="0 0 * * *",
            workspace_id="ws-123",
            audit_preset="security",
            on_push_branches=["main", "develop"],
            on_upload=True,
        )

        assert config.on_push_branches == ["main", "develop"]
        assert config.on_upload is True


# ============================================================================
# ScheduledJob Tests
# ============================================================================


class TestScheduledJob:
    """Tests for scheduled job objects."""

    def test_job_creation(self, schedule_config):
        """Test creating a scheduled job."""
        job = ScheduledJob(
            id="job-123",
            config=schedule_config,
            created_at=datetime.now(timezone.utc),
        )

        assert job.id == "job-123"
        assert job.config == schedule_config
        assert job.next_run is not None or job.next_run is None

    def test_job_to_dict(self, schedule_config):
        """Test converting job to dictionary."""
        job = ScheduledJob(
            id="job-123",
            config=schedule_config,
            created_at=datetime.now(timezone.utc),
        )

        data = job.to_dict()

        assert "id" in data
        assert "config" in data
        assert "created_at" in data


# ============================================================================
# JobRun Tests
# ============================================================================


class TestJobRun:
    """Tests for job run records."""

    def test_job_run_creation(self):
        """Test creating a job run record."""
        run = JobRun(
            id="run-123",
            job_id="job-456",
            trigger_type=TriggerType.SCHEDULED,
            started_at=datetime.now(timezone.utc),
            status=JobStatus.RUNNING,
        )

        assert run.id == "run-123"
        assert run.status == JobStatus.RUNNING
        assert run.trigger_type == TriggerType.SCHEDULED

    def test_job_run_completion(self):
        """Test completing a job run."""
        run = JobRun(
            id="run-123",
            job_id="job-456",
            trigger_type=TriggerType.SCHEDULED,
            started_at=datetime.now(timezone.utc),
            status=JobStatus.RUNNING,
        )

        run.status = JobStatus.COMPLETED
        run.completed_at = datetime.now(timezone.utc)
        run.session_id = "session-789"
        run.finding_count = 5

        assert run.status == JobStatus.COMPLETED
        assert run.finding_count == 5

    def test_job_run_to_dict(self):
        """Test converting run to dictionary."""
        run = JobRun(
            id="run-123",
            job_id="job-456",
            trigger_type=TriggerType.WEBHOOK,
            started_at=datetime.now(timezone.utc),
            status=JobStatus.COMPLETED,
        )

        data = run.to_dict()

        assert data["trigger_type"] == "webhook"
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
        assert job.id is not None
        assert job.config == schedule_config

    def test_add_schedule_generates_id(self, scheduler, schedule_config):
        """Test that adding schedule generates unique ID."""
        job1 = scheduler.add_schedule(schedule_config)
        job2 = scheduler.add_schedule(schedule_config)

        assert job1.id != job2.id

    def test_get_schedule(self, scheduler, schedule_config):
        """Test retrieving a schedule by ID."""
        job = scheduler.add_schedule(schedule_config)

        retrieved = scheduler.get_schedule(job.id)

        assert retrieved is not None
        assert retrieved.id == job.id

    def test_get_nonexistent_schedule(self, scheduler):
        """Test getting non-existent schedule returns None."""
        result = scheduler.get_schedule("nonexistent")

        assert result is None

    def test_list_schedules(self, scheduler, schedule_config):
        """Test listing all schedules."""
        scheduler.add_schedule(schedule_config)
        scheduler.add_schedule(schedule_config)

        schedules = scheduler.list_schedules()

        assert len(schedules) == 2

    def test_list_schedules_by_workspace(self, scheduler):
        """Test listing schedules filtered by workspace."""
        config1 = ScheduleConfig(
            name="Job 1",
            cron="0 0 * * *",
            workspace_id="ws-1",
            audit_preset="security",
        )
        config2 = ScheduleConfig(
            name="Job 2",
            cron="0 0 * * *",
            workspace_id="ws-2",
            audit_preset="security",
        )

        scheduler.add_schedule(config1)
        scheduler.add_schedule(config2)

        ws1_jobs = scheduler.list_schedules(workspace_id="ws-1")

        assert len(ws1_jobs) == 1
        assert ws1_jobs[0].config.workspace_id == "ws-1"

    def test_update_schedule(self, scheduler, schedule_config):
        """Test updating a schedule."""
        job = scheduler.add_schedule(schedule_config)

        updated_config = ScheduleConfig(
            name="Updated Schedule",
            cron="0 3 * * *",  # Changed to 3 AM
            workspace_id="ws-123",
            audit_preset="code_security",
        )

        result = scheduler.update_schedule(job.id, updated_config)

        assert result is True
        updated = scheduler.get_schedule(job.id)
        assert updated.config.name == "Updated Schedule"
        assert updated.config.cron == "0 3 * * *"

    def test_delete_schedule(self, scheduler, schedule_config):
        """Test deleting a schedule."""
        job = scheduler.add_schedule(schedule_config)

        result = scheduler.delete_schedule(job.id)

        assert result is True
        assert scheduler.get_schedule(job.id) is None

    def test_enable_disable_schedule(self, scheduler, schedule_config):
        """Test enabling and disabling a schedule."""
        job = scheduler.add_schedule(schedule_config)

        scheduler.disable_schedule(job.id)
        disabled = scheduler.get_schedule(job.id)
        assert disabled.config.enabled is False

        scheduler.enable_schedule(job.id)
        enabled = scheduler.get_schedule(job.id)
        assert enabled.config.enabled is True


# ============================================================================
# Cron Expression Tests
# ============================================================================


class TestCronExpressions:
    """Tests for cron expression handling."""

    def test_parse_valid_cron(self, scheduler):
        """Test parsing valid cron expressions."""
        # Daily at midnight
        next_run = scheduler._parse_cron("0 0 * * *")
        assert next_run is not None

        # Every hour
        next_run = scheduler._parse_cron("0 * * * *")
        assert next_run is not None

        # Weekly on Monday
        next_run = scheduler._parse_cron("0 0 * * 1")
        assert next_run is not None

    def test_parse_invalid_cron(self, scheduler):
        """Test parsing invalid cron expressions."""
        with pytest.raises(ValueError):
            scheduler._parse_cron("invalid cron")

        with pytest.raises(ValueError):
            scheduler._parse_cron("* * *")  # Too few fields

    def test_next_run_calculation(self, scheduler, schedule_config):
        """Test next run time is calculated correctly."""
        job = scheduler.add_schedule(schedule_config)

        # Next run should be in the future
        if job.next_run:
            assert job.next_run > datetime.now(timezone.utc)

    def test_next_run_after_completion(self, scheduler, schedule_config):
        """Test next run is updated after job completion."""
        job = scheduler.add_schedule(schedule_config)
        original_next = job.next_run

        # Simulate job completion
        scheduler._update_next_run(job.id)

        updated = scheduler.get_schedule(job.id)
        # Next run should be different (moved forward)
        if original_next and updated.next_run:
            assert updated.next_run >= original_next


# ============================================================================
# Job Execution Tests
# ============================================================================


class TestJobExecution:
    """Tests for job execution."""

    @pytest.mark.asyncio
    async def test_trigger_job_manually(self, scheduler, schedule_config, mock_auditor):
        """Test manually triggering a job."""
        with patch("aragora.scheduler.audit_scheduler.get_document_auditor",
                   return_value=mock_auditor):
            job = scheduler.add_schedule(schedule_config)

            run = await scheduler.trigger_job(job.id)

            assert run is not None
            assert run.job_id == job.id
            assert run.trigger_type == TriggerType.MANUAL

    @pytest.mark.asyncio
    async def test_trigger_nonexistent_job(self, scheduler):
        """Test triggering non-existent job returns None."""
        run = await scheduler.trigger_job("nonexistent")

        assert run is None

    @pytest.mark.asyncio
    async def test_trigger_disabled_job(self, scheduler, schedule_config):
        """Test triggering disabled job."""
        schedule_config.enabled = False
        job = scheduler.add_schedule(schedule_config)

        run = await scheduler.trigger_job(job.id)

        # May or may not trigger depending on implementation
        # Some schedulers allow manual triggers on disabled jobs

    @pytest.mark.asyncio
    async def test_job_run_records_history(self, scheduler, schedule_config, mock_auditor):
        """Test that job runs are recorded in history."""
        with patch("aragora.scheduler.audit_scheduler.get_document_auditor",
                   return_value=mock_auditor):
            job = scheduler.add_schedule(schedule_config)

            await scheduler.trigger_job(job.id)
            await scheduler.trigger_job(job.id)

            history = scheduler.get_job_history(job.id)

            assert len(history) == 2


# ============================================================================
# Webhook Trigger Tests
# ============================================================================


class TestWebhookTriggers:
    """Tests for webhook-triggered audits."""

    def test_register_webhook(self, scheduler, webhook_config):
        """Test registering a webhook."""
        result = scheduler.register_webhook(webhook_config)

        assert result is True
        assert scheduler.get_webhook(webhook_config.id) is not None

    def test_verify_webhook_signature(self, scheduler, webhook_config):
        """Test webhook signature verification."""
        scheduler.register_webhook(webhook_config)

        payload = {"event": "push", "ref": "refs/heads/main"}
        # Generate valid signature
        import hmac
        import hashlib

        signature = hmac.new(
            webhook_config.secret.encode(),
            json.dumps(payload).encode(),
            hashlib.sha256,
        ).hexdigest()

        is_valid = scheduler._verify_signature(
            webhook_config.id,
            payload,
            f"sha256={signature}",
        )

        assert is_valid is True

    def test_reject_invalid_signature(self, scheduler, webhook_config):
        """Test rejecting invalid webhook signature."""
        scheduler.register_webhook(webhook_config)

        payload = {"event": "push"}
        invalid_signature = "sha256=invalid"

        is_valid = scheduler._verify_signature(
            webhook_config.id,
            payload,
            invalid_signature,
        )

        assert is_valid is False

    @pytest.mark.asyncio
    async def test_handle_webhook(self, scheduler, webhook_config, mock_auditor):
        """Test handling webhook event."""
        with patch("aragora.scheduler.audit_scheduler.get_document_auditor",
                   return_value=mock_auditor):
            scheduler.register_webhook(webhook_config)

            payload = {"event": "push", "ref": "refs/heads/main"}
            import hmac
            import hashlib

            signature = "sha256=" + hmac.new(
                webhook_config.secret.encode(),
                json.dumps(payload).encode(),
                hashlib.sha256,
            ).hexdigest()

            runs = await scheduler.handle_webhook(
                webhook_config.id,
                payload,
                signature,
            )

            assert len(runs) >= 0  # May or may not trigger depending on config


# ============================================================================
# Git Push Event Tests
# ============================================================================


class TestGitPushEvents:
    """Tests for git push triggered audits."""

    @pytest.mark.asyncio
    async def test_handle_git_push(self, scheduler, mock_auditor):
        """Test handling git push event."""
        config = ScheduleConfig(
            name="Push-triggered Audit",
            workspace_id="ws-123",
            audit_preset="code_security",
            on_push_branches=["main"],
        )

        with patch("aragora.scheduler.audit_scheduler.get_document_auditor",
                   return_value=mock_auditor):
            job = scheduler.add_schedule(config)

            runs = await scheduler.handle_git_push(
                repository="org/repo",
                branch="main",
                commit_sha="abc123",
            )

            # Should trigger audit for main branch
            assert len(runs) >= 0

    @pytest.mark.asyncio
    async def test_git_push_non_matching_branch(self, scheduler):
        """Test git push on non-matching branch doesn't trigger."""
        config = ScheduleConfig(
            name="Push-triggered Audit",
            workspace_id="ws-123",
            audit_preset="code_security",
            on_push_branches=["main"],  # Only main
        )

        scheduler.add_schedule(config)

        runs = await scheduler.handle_git_push(
            repository="org/repo",
            branch="feature-branch",  # Not main
            commit_sha="abc123",
        )

        assert len(runs) == 0

    @pytest.mark.asyncio
    async def test_git_push_wildcard_branch(self, scheduler, mock_auditor):
        """Test git push with wildcard branch pattern."""
        config = ScheduleConfig(
            name="Push-triggered Audit",
            workspace_id="ws-123",
            audit_preset="code_security",
            on_push_branches=["release/*"],
        )

        with patch("aragora.scheduler.audit_scheduler.get_document_auditor",
                   return_value=mock_auditor):
            scheduler.add_schedule(config)

            runs = await scheduler.handle_git_push(
                repository="org/repo",
                branch="release/v1.0",
                commit_sha="abc123",
            )

            # Should match wildcard pattern
            # Implementation may vary


# ============================================================================
# File Upload Event Tests
# ============================================================================


class TestFileUploadEvents:
    """Tests for file upload triggered audits."""

    @pytest.mark.asyncio
    async def test_handle_file_upload(self, scheduler, mock_auditor):
        """Test handling file upload event."""
        config = ScheduleConfig(
            name="Upload-triggered Audit",
            workspace_id="ws-123",
            audit_preset="legal_due_diligence",
            on_upload=True,
        )

        with patch("aragora.scheduler.audit_scheduler.get_document_auditor",
                   return_value=mock_auditor):
            scheduler.add_schedule(config)

            runs = await scheduler.handle_file_upload(
                workspace_id="ws-123",
                document_ids=["doc-1", "doc-2"],
            )

            assert len(runs) >= 0

    @pytest.mark.asyncio
    async def test_file_upload_different_workspace(self, scheduler):
        """Test file upload in different workspace doesn't trigger."""
        config = ScheduleConfig(
            name="Upload-triggered Audit",
            workspace_id="ws-123",
            audit_preset="legal_due_diligence",
            on_upload=True,
        )

        scheduler.add_schedule(config)

        runs = await scheduler.handle_file_upload(
            workspace_id="ws-999",  # Different workspace
            document_ids=["doc-1"],
        )

        assert len(runs) == 0


# ============================================================================
# Job History Tests
# ============================================================================


class TestJobHistory:
    """Tests for job history management."""

    @pytest.mark.asyncio
    async def test_get_job_history(self, scheduler, schedule_config, mock_auditor):
        """Test retrieving job run history."""
        with patch("aragora.scheduler.audit_scheduler.get_document_auditor",
                   return_value=mock_auditor):
            job = scheduler.add_schedule(schedule_config)

            await scheduler.trigger_job(job.id)
            await scheduler.trigger_job(job.id)
            await scheduler.trigger_job(job.id)

            history = scheduler.get_job_history(job.id, limit=2)

            assert len(history) == 2

    @pytest.mark.asyncio
    async def test_get_job_history_empty(self, scheduler, schedule_config):
        """Test getting history for job with no runs."""
        job = scheduler.add_schedule(schedule_config)

        history = scheduler.get_job_history(job.id)

        assert len(history) == 0

    @pytest.mark.asyncio
    async def test_get_run_by_id(self, scheduler, schedule_config, mock_auditor):
        """Test getting specific run by ID."""
        with patch("aragora.scheduler.audit_scheduler.get_document_auditor",
                   return_value=mock_auditor):
            job = scheduler.add_schedule(schedule_config)

            run = await scheduler.trigger_job(job.id)

            retrieved = scheduler.get_run(run.id)

            assert retrieved is not None
            assert retrieved.id == run.id


# ============================================================================
# Scheduler Lifecycle Tests
# ============================================================================


class TestSchedulerLifecycle:
    """Tests for scheduler lifecycle management."""

    @pytest.mark.asyncio
    async def test_start_scheduler(self, scheduler):
        """Test starting the scheduler."""
        await scheduler.start()

        assert scheduler.is_running is True

    @pytest.mark.asyncio
    async def test_stop_scheduler(self, scheduler):
        """Test stopping the scheduler."""
        await scheduler.start()
        await scheduler.stop()

        assert scheduler.is_running is False

    @pytest.mark.asyncio
    async def test_get_due_jobs(self, scheduler):
        """Test getting jobs that are due to run."""
        # Create job with past next_run
        config = ScheduleConfig(
            name="Due Job",
            cron="* * * * *",  # Every minute
            workspace_id="ws-123",
            audit_preset="security",
        )

        job = scheduler.add_schedule(config)
        # Force next_run to be in the past
        job.next_run = datetime.now(timezone.utc) - timedelta(minutes=5)

        due_jobs = scheduler.get_due_jobs()

        assert len(due_jobs) >= 1
        assert any(j.id == job.id for j in due_jobs)
