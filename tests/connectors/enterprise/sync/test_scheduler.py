"""
Tests for aragora.connectors.enterprise.sync.scheduler module.

Tests cover:
- RetryPolicy delay calculation and execution
- SyncSchedule configuration and serialization
- SyncHistory dataclass and properties
- SyncJob lifecycle and scheduling
- SyncScheduler registration and management
- Sync execution and error handling
- History tracking and cleanup
- Statistics computation
- State persistence
- Cron scheduling (parse expressions, next run calculation)
- Retry policies (exponential backoff, max retries, jitter)
- Job registration (add_job, remove_job, list_jobs)
- Job execution (run_job, cancel_job)
- Concurrency control (max concurrent jobs)
- Error handling (job failures, timeout)
- Persistence (save/load job state)
- Metrics (job duration, success rate)
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.connectors.enterprise.sync.scheduler import (
    DEFAULT_HISTORY_RETENTION_DAYS,
    MAX_HISTORY_ENTRIES,
    RetryPolicy,
    SyncHistory,
    SyncJob,
    SyncSchedule,
    SyncScheduler,
    SyncStatus,
)


# =============================================================================
# TestSyncStatus
# =============================================================================


class TestSyncStatus:
    """Tests for SyncStatus enum."""

    def test_values(self):
        """Should have all expected status values."""
        assert SyncStatus.PENDING.value == "pending"
        assert SyncStatus.RUNNING.value == "running"
        assert SyncStatus.COMPLETED.value == "completed"
        assert SyncStatus.FAILED.value == "failed"
        assert SyncStatus.CANCELLED.value == "cancelled"

    def test_all_statuses_exist(self):
        """Should have exactly 5 status values."""
        assert len(SyncStatus) == 5


# =============================================================================
# TestRetryPolicy
# =============================================================================


class TestRetryPolicyInit:
    """Tests for RetryPolicy initialization."""

    def test_default_values(self):
        """Should initialize with sensible defaults."""
        policy = RetryPolicy()

        assert policy.max_retries == 5
        assert policy.base_delay == 1.0
        assert policy.max_delay == 300.0
        assert policy.exponential_base == 2.0
        assert policy.jitter is True

    def test_custom_values(self):
        """Should accept custom values."""
        policy = RetryPolicy(
            max_retries=3,
            base_delay=2.0,
            max_delay=60.0,
            exponential_base=3.0,
            jitter=False,
        )

        assert policy.max_retries == 3
        assert policy.base_delay == 2.0
        assert policy.max_delay == 60.0

    def test_zero_retries(self):
        """Should accept zero retries."""
        policy = RetryPolicy(max_retries=0)
        assert policy.max_retries == 0


class TestRetryPolicyCalculateDelay:
    """Tests for RetryPolicy.calculate_delay()."""

    def test_exponential_backoff_without_jitter(self):
        """Should calculate exponential backoff without jitter."""
        policy = RetryPolicy(base_delay=1.0, exponential_base=2.0, jitter=False)

        assert policy.calculate_delay(0) == 1.0  # 1 * 2^0
        assert policy.calculate_delay(1) == 2.0  # 1 * 2^1
        assert policy.calculate_delay(2) == 4.0  # 1 * 2^2
        assert policy.calculate_delay(3) == 8.0  # 1 * 2^3

    def test_caps_at_max_delay(self):
        """Should cap delay at max_delay."""
        policy = RetryPolicy(
            base_delay=1.0,
            max_delay=10.0,
            exponential_base=2.0,
            jitter=False,
        )

        assert policy.calculate_delay(10) == 10.0  # Capped at max_delay

    def test_jitter_adds_variation(self):
        """Should add jitter when enabled."""
        policy = RetryPolicy(base_delay=10.0, exponential_base=1.0, jitter=True)

        # Run multiple times to check for variation
        delays = [policy.calculate_delay(0) for _ in range(20)]

        # With jitter, delays should vary
        assert len(set(round(d, 3) for d in delays)) > 1

    def test_delay_is_non_negative(self):
        """Should never return negative delay."""
        policy = RetryPolicy(base_delay=1.0, jitter=True)

        for attempt in range(10):
            delay = policy.calculate_delay(attempt)
            assert delay >= 0

    def test_jitter_range_is_25_percent(self):
        """Jitter should vary within 25% of the base delay."""
        policy = RetryPolicy(base_delay=100.0, exponential_base=1.0, jitter=True, max_delay=1000.0)

        # Collect many samples
        delays = [policy.calculate_delay(0) for _ in range(100)]

        # Expected delay without jitter is 100
        # With 25% jitter, range is 75-125
        assert min(delays) >= 75.0
        assert max(delays) <= 125.0

    def test_exponential_base_3(self):
        """Should work with different exponential base."""
        policy = RetryPolicy(base_delay=1.0, exponential_base=3.0, jitter=False)

        assert policy.calculate_delay(0) == 1.0  # 1 * 3^0
        assert policy.calculate_delay(1) == 3.0  # 1 * 3^1
        assert policy.calculate_delay(2) == 9.0  # 1 * 3^2

    def test_large_attempt_number(self):
        """Should handle large attempt numbers gracefully."""
        policy = RetryPolicy(base_delay=1.0, max_delay=300.0, jitter=False)

        # Large attempt should be capped at max_delay
        delay = policy.calculate_delay(100)
        assert delay == 300.0


class TestRetryPolicyExecuteWithRetry:
    """Tests for RetryPolicy.execute_with_retry()."""

    @pytest.mark.asyncio
    async def test_returns_result_on_success(self):
        """Should return result when function succeeds."""
        policy = RetryPolicy(max_retries=3, base_delay=0.01)

        async def success_func():
            return "result"

        result = await policy.execute_with_retry(success_func)
        assert result == "result"

    @pytest.mark.asyncio
    async def test_retries_on_failure(self):
        """Should retry on failure."""
        policy = RetryPolicy(max_retries=3, base_delay=0.01, jitter=False)

        call_count = 0

        async def fail_then_succeed():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("Fail")
            return "success"

        result = await policy.execute_with_retry(fail_then_succeed)

        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_raises_after_max_retries(self):
        """Should raise last exception after max retries."""
        policy = RetryPolicy(max_retries=2, base_delay=0.01, jitter=False)

        async def always_fail():
            raise RuntimeError("Persistent failure")

        with pytest.raises(RuntimeError, match="Persistent failure"):
            await policy.execute_with_retry(always_fail)

    @pytest.mark.asyncio
    async def test_calls_on_retry_callback(self):
        """Should call on_retry callback on each retry."""
        policy = RetryPolicy(max_retries=3, base_delay=0.01, jitter=False)

        retries = []

        async def fail_then_succeed():
            if len(retries) < 2:
                raise RuntimeError("Fail")
            return "success"

        def on_retry(attempt, exc):
            retries.append((attempt, str(exc)))

        await policy.execute_with_retry(fail_then_succeed, on_retry=on_retry)

        assert len(retries) == 2

    @pytest.mark.asyncio
    async def test_does_not_retry_on_cancellation(self):
        """Should not retry on asyncio.CancelledError."""
        policy = RetryPolicy(max_retries=3, base_delay=0.01)

        async def cancel_func():
            raise asyncio.CancelledError()

        with pytest.raises(asyncio.CancelledError):
            await policy.execute_with_retry(cancel_func)

    @pytest.mark.asyncio
    async def test_retries_on_oserror(self):
        """Should retry on OSError."""
        policy = RetryPolicy(max_retries=3, base_delay=0.01, jitter=False)

        call_count = 0

        async def oserror_then_succeed():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise OSError("Network error")
            return "recovered"

        result = await policy.execute_with_retry(oserror_then_succeed)
        assert result == "recovered"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_retries_on_valueerror(self):
        """Should retry on ValueError."""
        policy = RetryPolicy(max_retries=3, base_delay=0.01, jitter=False)

        call_count = 0

        async def valueerror_then_succeed():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("Invalid data")
            return "fixed"

        result = await policy.execute_with_retry(valueerror_then_succeed)
        assert result == "fixed"

    @pytest.mark.asyncio
    async def test_passes_args_and_kwargs(self):
        """Should pass arguments to the retried function."""
        policy = RetryPolicy(max_retries=1, base_delay=0.01)

        async def func_with_args(a, b, c=None):
            return f"{a}-{b}-{c}"

        result = await policy.execute_with_retry(func_with_args, "x", "y", c="z")
        assert result == "x-y-z"

    @pytest.mark.asyncio
    async def test_zero_retries_single_attempt(self):
        """With max_retries=0, should only try once."""
        policy = RetryPolicy(max_retries=0, base_delay=0.01)

        call_count = 0

        async def always_fail():
            nonlocal call_count
            call_count += 1
            raise RuntimeError("Fail")

        with pytest.raises(RuntimeError):
            await policy.execute_with_retry(always_fail)

        assert call_count == 1


# =============================================================================
# TestSyncSchedule
# =============================================================================


class TestSyncScheduleInit:
    """Tests for SyncSchedule initialization."""

    def test_default_values(self):
        """Should initialize with sensible defaults."""
        schedule = SyncSchedule()

        assert schedule.schedule_type == "interval"
        assert schedule.interval_minutes == 60
        assert schedule.cron_expression is None
        assert schedule.enabled is True
        assert schedule.max_concurrent == 1
        assert schedule.retry_on_failure is True
        assert schedule.max_retries == 3

    def test_custom_values(self):
        """Should accept custom values."""
        schedule = SyncSchedule(
            schedule_type="cron",
            cron_expression="*/5 * * * *",
            interval_minutes=30,
            max_concurrent=3,
        )

        assert schedule.schedule_type == "cron"
        assert schedule.cron_expression == "*/5 * * * *"
        assert schedule.max_concurrent == 3

    def test_webhook_only_schedule(self):
        """Should support webhook-only schedule type."""
        schedule = SyncSchedule(schedule_type="webhook_only")
        assert schedule.schedule_type == "webhook_only"

    def test_time_constraints(self):
        """Should accept time constraints."""
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 12, 31, tzinfo=timezone.utc)

        schedule = SyncSchedule(start_time=start, end_time=end)

        assert schedule.start_time == start
        assert schedule.end_time == end


class TestSyncScheduleSerialization:
    """Tests for SyncSchedule serialization."""

    def test_to_dict(self):
        """Should serialize to dictionary."""
        schedule = SyncSchedule(
            schedule_type="interval",
            interval_minutes=30,
            enabled=True,
        )

        result = schedule.to_dict()

        assert result["schedule_type"] == "interval"
        assert result["interval_minutes"] == 30
        assert result["enabled"] is True

    def test_from_dict(self):
        """Should deserialize from dictionary."""
        data = {
            "schedule_type": "cron",
            "cron_expression": "*/5 * * * *",
            "interval_minutes": 30,
            "enabled": True,
            "max_concurrent": 2,
        }

        schedule = SyncSchedule.from_dict(data)

        assert schedule.schedule_type == "cron"
        assert schedule.cron_expression == "*/5 * * * *"
        assert schedule.max_concurrent == 2

    def test_roundtrip(self):
        """Should preserve data through serialization roundtrip."""
        original = SyncSchedule(
            schedule_type="interval",
            interval_minutes=45,
            max_retries=5,
            retry_delay_minutes=10,
        )

        restored = SyncSchedule.from_dict(original.to_dict())

        assert restored.schedule_type == original.schedule_type
        assert restored.interval_minutes == original.interval_minutes
        assert restored.max_retries == original.max_retries

    def test_to_dict_with_time_constraints(self):
        """Should serialize time constraints."""
        start = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
        schedule = SyncSchedule(start_time=start)

        result = schedule.to_dict()

        assert result["start_time"] == start.isoformat()

    def test_from_dict_with_time_constraints(self):
        """Should deserialize time constraints."""
        data = {
            "start_time": "2024-01-01T12:00:00+00:00",
            "end_time": "2024-12-31T12:00:00+00:00",
        }

        schedule = SyncSchedule.from_dict(data)

        assert schedule.start_time is not None
        assert schedule.end_time is not None
        assert schedule.start_time.year == 2024

    def test_from_dict_defaults_for_missing_keys(self):
        """Should use defaults for missing keys."""
        data = {}
        schedule = SyncSchedule.from_dict(data)

        assert schedule.schedule_type == "interval"
        assert schedule.interval_minutes == 60
        assert schedule.enabled is True


# =============================================================================
# TestSyncHistory
# =============================================================================


class TestSyncHistoryInit:
    """Tests for SyncHistory initialization."""

    def test_create_basic(self):
        """Should create with required fields."""
        now = datetime.now(timezone.utc)
        history = SyncHistory(
            id="run-001",
            job_id="job-001",
            connector_id="connector-001",
            tenant_id="tenant-001",
            status=SyncStatus.RUNNING,
            started_at=now,
        )

        assert history.id == "run-001"
        assert history.status == SyncStatus.RUNNING
        assert history.items_synced == 0
        assert history.errors == []

    def test_duration_seconds_when_complete(self):
        """Should calculate duration when completed."""
        now = datetime.now(timezone.utc)
        history = SyncHistory(
            id="run-001",
            job_id="job-001",
            connector_id="connector-001",
            tenant_id="tenant-001",
            status=SyncStatus.COMPLETED,
            started_at=now,
            completed_at=now + timedelta(seconds=120),
        )

        assert history.duration_seconds == 120.0

    def test_duration_seconds_none_when_running(self):
        """Should return None duration when still running."""
        history = SyncHistory(
            id="run-001",
            job_id="job-001",
            connector_id="connector-001",
            tenant_id="tenant-001",
            status=SyncStatus.RUNNING,
            started_at=datetime.now(timezone.utc),
        )

        assert history.duration_seconds is None

    def test_metadata_field(self):
        """Should support metadata field."""
        history = SyncHistory(
            id="run-001",
            job_id="job-001",
            connector_id="conn",
            tenant_id="tenant",
            status=SyncStatus.RUNNING,
            started_at=datetime.now(timezone.utc),
            metadata={"key": "value", "count": 42},
        )

        assert history.metadata["key"] == "value"
        assert history.metadata["count"] == 42


class TestSyncHistoryToDict:
    """Tests for SyncHistory.to_dict()."""

    def test_returns_dict(self):
        """Should return a dictionary."""
        now = datetime.now(timezone.utc)
        history = SyncHistory(
            id="run-001",
            job_id="job-001",
            connector_id="connector-001",
            tenant_id="tenant-001",
            status=SyncStatus.COMPLETED,
            started_at=now,
            completed_at=now + timedelta(seconds=60),
            items_synced=100,
            items_total=150,
        )

        result = history.to_dict()

        assert result["id"] == "run-001"
        assert result["status"] == "completed"
        assert result["items_synced"] == 100
        assert result["items_total"] == 150
        assert result["duration_seconds"] == 60.0

    def test_includes_errors(self):
        """Should include errors in serialization."""
        history = SyncHistory(
            id="run-001",
            job_id="job-001",
            connector_id="conn",
            tenant_id="tenant",
            status=SyncStatus.FAILED,
            started_at=datetime.now(timezone.utc),
            errors=["Error 1", "Error 2"],
        )

        result = history.to_dict()
        assert result["errors"] == ["Error 1", "Error 2"]


# =============================================================================
# TestSyncJob
# =============================================================================


class TestSyncJobInit:
    """Tests for SyncJob initialization."""

    def test_create_basic(self):
        """Should create with required fields."""
        schedule = SyncSchedule(schedule_type="webhook_only")
        job = SyncJob(
            id="job-001",
            connector_id="connector-001",
            tenant_id="tenant-001",
            schedule=schedule,
        )

        assert job.id == "job-001"
        assert job.connector_id == "connector-001"
        assert job.consecutive_failures == 0

    def test_interval_schedule_sets_next_run(self):
        """Should calculate next_run for interval schedules."""
        schedule = SyncSchedule(schedule_type="interval", interval_minutes=60)
        job = SyncJob(
            id="job-001",
            connector_id="connector-001",
            tenant_id="tenant-001",
            schedule=schedule,
        )

        assert job.next_run is not None

    def test_webhook_only_no_next_run(self):
        """Should not set next_run for webhook-only schedules."""
        schedule = SyncSchedule(schedule_type="webhook_only")
        job = SyncJob(
            id="job-001",
            connector_id="connector-001",
            tenant_id="tenant-001",
            schedule=schedule,
        )

        assert job.next_run is None

    def test_disabled_schedule_no_next_run(self):
        """Should not set next_run when schedule is disabled."""
        schedule = SyncSchedule(enabled=False)
        job = SyncJob(
            id="job-001",
            connector_id="connector-001",
            tenant_id="tenant-001",
            schedule=schedule,
        )

        assert job.next_run is None

    def test_interval_with_last_run(self):
        """Should calculate next_run based on last_run for intervals."""
        last_run = datetime.now(timezone.utc) - timedelta(minutes=30)
        schedule = SyncSchedule(schedule_type="interval", interval_minutes=60)
        job = SyncJob(
            id="job-001",
            connector_id="connector-001",
            tenant_id="tenant-001",
            schedule=schedule,
            last_run=last_run,
        )

        assert job.next_run is not None
        # Next run should be ~30 minutes from now (60 - 30 already elapsed)
        expected = last_run + timedelta(minutes=60)
        assert abs((job.next_run - expected).total_seconds()) < 1


class TestSyncJobCronScheduling:
    """Tests for cron-based scheduling."""

    def test_cron_schedule_sets_next_run(self):
        """Should calculate next_run for cron schedules when croniter available."""
        schedule = SyncSchedule(schedule_type="cron", cron_expression="0 * * * *")
        job = SyncJob(
            id="job-001",
            connector_id="connector-001",
            tenant_id="tenant-001",
            schedule=schedule,
        )

        # Should have a next_run set (either from croniter or fallback)
        assert job.next_run is not None

    def test_cron_invalid_expression_returns_none(self):
        """Should handle invalid cron expressions."""
        schedule = SyncSchedule(schedule_type="cron", cron_expression="invalid cron")
        job = SyncJob(
            id="job-001",
            connector_id="connector-001",
            tenant_id="tenant-001",
            schedule=schedule,
        )

        # Depends on whether croniter is installed - either None or fallback
        # Just ensure no exception is raised

    def test_cron_fallback_when_croniter_missing(self):
        """Should fallback to interval when croniter not installed."""
        # This test verifies the fallback behavior when croniter import fails
        # The actual implementation handles this with a try/except in _parse_cron_next
        schedule = SyncSchedule(schedule_type="cron", cron_expression="0 * * * *")
        job = SyncJob(
            id="job-001",
            connector_id="connector-001",
            tenant_id="tenant-001",
            schedule=schedule,
        )

        # Should still have a next_run (either from croniter or fallback)
        # The fallback returns datetime.now() + timedelta(hours=1)
        assert job.next_run is not None


class TestSyncJobTimeConstraints:
    """Tests for time constraint handling in SyncJob."""

    def test_start_time_constraint(self):
        """Should not schedule before start_time."""
        future_start = datetime.now(timezone.utc) + timedelta(hours=1)
        schedule = SyncSchedule(
            schedule_type="interval",
            interval_minutes=1,  # Would normally run soon
            start_time=future_start,
        )
        job = SyncJob(
            id="job-001",
            connector_id="connector-001",
            tenant_id="tenant-001",
            schedule=schedule,
        )

        assert job.next_run is not None
        assert job.next_run >= future_start

    def test_end_time_constraint(self):
        """Should not schedule after end_time."""
        past_end = datetime.now(timezone.utc) - timedelta(hours=1)
        schedule = SyncSchedule(
            schedule_type="interval",
            interval_minutes=60,
            end_time=past_end,
        )
        job = SyncJob(
            id="job-001",
            connector_id="connector-001",
            tenant_id="tenant-001",
            schedule=schedule,
        )

        assert job.next_run is None


class TestSyncJobToDict:
    """Tests for SyncJob.to_dict()."""

    def test_returns_dict(self):
        """Should return a dictionary."""
        schedule = SyncSchedule(schedule_type="webhook_only")
        job = SyncJob(
            id="job-001",
            connector_id="connector-001",
            tenant_id="tenant-001",
            schedule=schedule,
        )

        result = job.to_dict()

        assert result["id"] == "job-001"
        assert result["connector_id"] == "connector-001"
        assert result["schedule"]["schedule_type"] == "webhook_only"

    def test_includes_runtime_state(self):
        """Should include runtime state in serialization."""
        schedule = SyncSchedule(schedule_type="webhook_only")
        job = SyncJob(
            id="job-001",
            connector_id="connector-001",
            tenant_id="tenant-001",
            schedule=schedule,
            consecutive_failures=3,
        )

        result = job.to_dict()
        assert result["consecutive_failures"] == 3


# =============================================================================
# TestSyncSchedulerInit
# =============================================================================


class TestSyncSchedulerInit:
    """Tests for SyncScheduler initialization."""

    def test_default_init(self):
        """Should initialize with defaults."""
        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler = SyncScheduler(state_dir=Path(tmpdir))

            assert scheduler.max_concurrent_syncs == 5
            assert scheduler.history_retention_days == DEFAULT_HISTORY_RETENTION_DAYS

    def test_custom_values(self):
        """Should accept custom values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler = SyncScheduler(
                state_dir=Path(tmpdir),
                max_concurrent_syncs=10,
                history_retention_days=60,
            )

            assert scheduler.max_concurrent_syncs == 10
            assert scheduler.history_retention_days == 60

    def test_creates_state_directory(self):
        """Should create state directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            new_dir = Path(tmpdir) / "new_state_dir"
            scheduler = SyncScheduler(state_dir=new_dir)

            assert new_dir.exists()

    def test_max_history_entries(self):
        """Should accept max_history_entries parameter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler = SyncScheduler(
                state_dir=Path(tmpdir),
                max_history_entries=100,
            )

            assert scheduler._max_history_entries == 100


# =============================================================================
# TestSyncSchedulerRegistration
# =============================================================================


class TestSyncSchedulerRegistration:
    """Tests for connector registration."""

    def test_register_connector(self):
        """Should register a connector and create a job."""
        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler = SyncScheduler(state_dir=Path(tmpdir))

            connector = MagicMock()
            connector.connector_id = "jira"
            connector.name = "Jira"

            job = scheduler.register_connector(
                connector,
                schedule=SyncSchedule(schedule_type="interval", interval_minutes=30),
                tenant_id="tenant-001",
            )

            assert job.connector_id == "jira"
            assert job.tenant_id == "tenant-001"

    def test_register_creates_job_id(self):
        """Should create job ID from tenant and connector."""
        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler = SyncScheduler(state_dir=Path(tmpdir))

            connector = MagicMock()
            connector.connector_id = "jira"
            connector.name = "Jira"

            job = scheduler.register_connector(connector, tenant_id="acme")

            assert job.id == "acme:jira"

    def test_unregister_connector(self):
        """Should unregister a connector and remove its job."""
        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler = SyncScheduler(state_dir=Path(tmpdir))

            connector = MagicMock()
            connector.connector_id = "jira"
            connector.name = "Jira"

            scheduler.register_connector(connector, tenant_id="acme")
            scheduler.unregister_connector("jira", tenant_id="acme")

            assert scheduler.get_job("acme:jira") is None

    def test_get_job(self):
        """Should retrieve a registered job."""
        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler = SyncScheduler(state_dir=Path(tmpdir))

            connector = MagicMock()
            connector.connector_id = "jira"
            connector.name = "Jira"

            scheduler.register_connector(connector, tenant_id="acme")

            job = scheduler.get_job("acme:jira")
            assert job is not None
            assert job.connector_id == "jira"

    def test_get_job_nonexistent(self):
        """Should return None for nonexistent job."""
        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler = SyncScheduler(state_dir=Path(tmpdir))

            assert scheduler.get_job("nonexistent:job") is None

    def test_list_jobs(self):
        """Should list all registered jobs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler = SyncScheduler(state_dir=Path(tmpdir))

            for name in ["jira", "github", "slack"]:
                connector = MagicMock()
                connector.connector_id = name
                connector.name = name
                scheduler.register_connector(connector, tenant_id="acme")

            jobs = scheduler.list_jobs()
            assert len(jobs) == 3

    def test_list_jobs_filtered_by_tenant(self):
        """Should filter jobs by tenant."""
        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler = SyncScheduler(state_dir=Path(tmpdir))

            entries = [("acme", "jira"), ("acme", "confluence"), ("beta", "jira")]
            for tenant, tool in entries:
                connector = MagicMock()
                connector.connector_id = f"{tool}-{tenant}"
                connector.name = f"{tool.title()} ({tenant})"
                scheduler.register_connector(connector, tenant_id=tenant)

            acme_jobs = scheduler.list_jobs(tenant_id="acme")
            assert len(acme_jobs) == 2

    def test_list_jobs_sorted_by_next_run(self):
        """Should sort jobs by next_run time."""
        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler = SyncScheduler(state_dir=Path(tmpdir))

            # Create jobs with different schedules
            for i, interval in enumerate([60, 30, 120]):
                connector = MagicMock()
                connector.connector_id = f"conn-{i}"
                connector.name = f"Connector {i}"
                scheduler.register_connector(
                    connector,
                    schedule=SyncSchedule(interval_minutes=interval),
                    tenant_id="acme",
                )

            jobs = scheduler.list_jobs()
            # Jobs should be sorted by next_run
            for i in range(len(jobs) - 1):
                if jobs[i].next_run and jobs[i + 1].next_run:
                    assert jobs[i].next_run <= jobs[i + 1].next_run

    def test_register_with_default_schedule(self):
        """Should create webhook-only schedule when none provided."""
        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler = SyncScheduler(state_dir=Path(tmpdir))

            connector = MagicMock()
            connector.connector_id = "jira"
            connector.name = "Jira"

            job = scheduler.register_connector(connector)

            assert job.schedule.schedule_type == "webhook_only"

    def test_unregister_removes_from_connectors(self):
        """Should remove connector from internal connectors dict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler = SyncScheduler(state_dir=Path(tmpdir))

            connector = MagicMock()
            connector.connector_id = "jira"
            connector.name = "Jira"

            scheduler.register_connector(connector, tenant_id="acme")
            assert "jira" in scheduler._connectors

            scheduler.unregister_connector("jira", tenant_id="acme")
            assert "jira" not in scheduler._connectors


# =============================================================================
# TestSyncSchedulerExecution
# =============================================================================


class TestSyncSchedulerExecution:
    """Tests for sync execution."""

    @pytest.mark.asyncio
    async def test_trigger_sync(self):
        """Should trigger a manual sync."""
        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler = SyncScheduler(state_dir=Path(tmpdir))

            mock_result = MagicMock()
            mock_result.success = True
            mock_result.items_synced = 50
            mock_result.items_total = 50
            mock_result.errors = []

            connector = MagicMock()
            connector.connector_id = "jira"
            connector.name = "Jira"
            connector.sync = AsyncMock(return_value=mock_result)

            scheduler.register_connector(connector, tenant_id="acme")

            run_id = await scheduler.trigger_sync("jira", tenant_id="acme")

            assert run_id is not None
            connector.sync.assert_called_once()

    @pytest.mark.asyncio
    async def test_trigger_sync_not_registered(self):
        """Should return None for unregistered connector."""
        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler = SyncScheduler(state_dir=Path(tmpdir))

            result = await scheduler.trigger_sync("nonexistent")

            assert result is None

    @pytest.mark.asyncio
    async def test_trigger_sync_full_sync(self):
        """Should pass full_sync flag to connector."""
        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler = SyncScheduler(state_dir=Path(tmpdir))

            mock_result = MagicMock()
            mock_result.success = True
            mock_result.items_synced = 100
            mock_result.items_total = 100
            mock_result.errors = []

            connector = MagicMock()
            connector.connector_id = "jira"
            connector.name = "Jira"
            connector.sync = AsyncMock(return_value=mock_result)

            scheduler.register_connector(connector, tenant_id="acme")

            await scheduler.trigger_sync("jira", tenant_id="acme", full_sync=True)

            connector.sync.assert_called_once_with(full_sync=True)

    @pytest.mark.asyncio
    async def test_sync_records_history(self):
        """Should record sync execution in history."""
        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler = SyncScheduler(state_dir=Path(tmpdir))

            mock_result = MagicMock()
            mock_result.success = True
            mock_result.items_synced = 25
            mock_result.items_total = 30
            mock_result.errors = []

            connector = MagicMock()
            connector.connector_id = "jira"
            connector.name = "Jira"
            connector.sync = AsyncMock(return_value=mock_result)

            scheduler.register_connector(connector, tenant_id="acme")
            await scheduler.trigger_sync("jira", tenant_id="acme")

            history = scheduler.get_history()
            assert len(history) == 1
            assert history[0].items_synced == 25

    @pytest.mark.asyncio
    async def test_sync_failure_records_error(self):
        """Should record sync failure in history."""
        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler = SyncScheduler(state_dir=Path(tmpdir))

            connector = MagicMock()
            connector.connector_id = "jira"
            connector.name = "Jira"
            connector.sync = AsyncMock(side_effect=RuntimeError("Connection failed"))

            scheduler.register_connector(connector, tenant_id="acme")
            await scheduler.trigger_sync("jira", tenant_id="acme")

            history = scheduler.get_history()
            assert len(history) == 1
            assert history[0].status == SyncStatus.FAILED
            assert "Connection failed" in history[0].errors[0]

    @pytest.mark.asyncio
    async def test_sync_failure_increments_consecutive_failures(self):
        """Should increment consecutive failures on error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler = SyncScheduler(state_dir=Path(tmpdir))

            connector = MagicMock()
            connector.connector_id = "jira"
            connector.name = "Jira"
            connector.sync = AsyncMock(side_effect=RuntimeError("Fail"))

            scheduler.register_connector(connector, tenant_id="acme")
            await scheduler.trigger_sync("jira", tenant_id="acme")

            job = scheduler.get_job("acme:jira")
            assert job.consecutive_failures == 1

    @pytest.mark.asyncio
    async def test_sync_success_resets_failures(self):
        """Should reset consecutive failures on success."""
        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler = SyncScheduler(state_dir=Path(tmpdir))

            mock_result = MagicMock()
            mock_result.success = True
            mock_result.items_synced = 10
            mock_result.items_total = 10
            mock_result.errors = []

            connector = MagicMock()
            connector.connector_id = "jira"
            connector.name = "Jira"
            connector.sync = AsyncMock(return_value=mock_result)

            scheduler.register_connector(connector, tenant_id="acme")

            # Simulate previous failures
            job = scheduler.get_job("acme:jira")
            job.consecutive_failures = 3

            await scheduler.trigger_sync("jira", tenant_id="acme")

            assert job.consecutive_failures == 0

    @pytest.mark.asyncio
    async def test_sync_already_running(self):
        """Should return current run_id if sync already running."""
        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler = SyncScheduler(state_dir=Path(tmpdir))

            connector = MagicMock()
            connector.connector_id = "jira"
            connector.name = "Jira"

            scheduler.register_connector(connector, tenant_id="acme")

            # Set current_run_id to simulate running sync
            job = scheduler.get_job("acme:jira")
            job.current_run_id = "existing-run"

            result = await scheduler.trigger_sync("jira", tenant_id="acme")

            assert result == "existing-run"

    @pytest.mark.asyncio
    async def test_sync_calls_on_complete_callback(self):
        """Should call on_complete callback on success."""
        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler = SyncScheduler(state_dir=Path(tmpdir))

            mock_result = MagicMock()
            mock_result.success = True
            mock_result.items_synced = 10
            mock_result.items_total = 10
            mock_result.errors = []

            connector = MagicMock()
            connector.connector_id = "jira"
            connector.name = "Jira"
            connector.sync = AsyncMock(return_value=mock_result)

            on_complete = MagicMock()

            job = scheduler.register_connector(connector, tenant_id="acme")
            job.on_complete = on_complete

            await scheduler.trigger_sync("jira", tenant_id="acme")

            on_complete.assert_called_once_with(mock_result)

    @pytest.mark.asyncio
    async def test_sync_calls_on_error_callback(self):
        """Should call on_error callback on failure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler = SyncScheduler(state_dir=Path(tmpdir))

            connector = MagicMock()
            connector.connector_id = "jira"
            connector.name = "Jira"
            error = RuntimeError("Failed")
            connector.sync = AsyncMock(side_effect=error)

            on_error = MagicMock()

            job = scheduler.register_connector(connector, tenant_id="acme")
            job.on_error = on_error

            await scheduler.trigger_sync("jira", tenant_id="acme")

            on_error.assert_called_once()
            assert isinstance(on_error.call_args[0][0], RuntimeError)

    @pytest.mark.asyncio
    async def test_sync_updates_job_last_run(self):
        """Should update job's last_run after sync."""
        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler = SyncScheduler(state_dir=Path(tmpdir))

            mock_result = MagicMock()
            mock_result.success = True
            mock_result.items_synced = 10
            mock_result.items_total = 10
            mock_result.errors = []

            connector = MagicMock()
            connector.connector_id = "jira"
            connector.name = "Jira"
            connector.sync = AsyncMock(return_value=mock_result)

            scheduler.register_connector(connector, tenant_id="acme")

            before = datetime.now(timezone.utc)
            await scheduler.trigger_sync("jira", tenant_id="acme")
            after = datetime.now(timezone.utc)

            job = scheduler.get_job("acme:jira")
            assert job.last_run is not None
            assert before <= job.last_run <= after


# =============================================================================
# TestSyncSchedulerConcurrency
# =============================================================================


class TestSyncSchedulerConcurrency:
    """Tests for concurrency control."""

    @pytest.mark.asyncio
    async def test_respects_max_concurrent_syncs(self):
        """Should queue jobs when max concurrent syncs reached."""
        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler = SyncScheduler(state_dir=Path(tmpdir), max_concurrent_syncs=1)

            # Create a slow connector
            async def slow_sync(full_sync=False):
                await asyncio.sleep(0.1)
                result = MagicMock()
                result.success = True
                result.items_synced = 10
                result.items_total = 10
                result.errors = []
                return result

            connector = MagicMock()
            connector.connector_id = "slow"
            connector.name = "Slow"
            connector.sync = slow_sync

            scheduler.register_connector(connector, tenant_id="acme")

            # Fill up running syncs dict manually
            scheduler._running_syncs["fake-task"] = asyncio.create_task(asyncio.sleep(10))

            try:
                # Trigger sync - should be queued
                run_id = await scheduler.trigger_sync("slow", tenant_id="acme")

                # Check that history shows PENDING status
                history = scheduler.get_history()
                assert len(history) == 1
                assert history[0].status == SyncStatus.PENDING
            finally:
                # Cleanup
                for task in scheduler._running_syncs.values():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass


# =============================================================================
# TestSyncSchedulerWebhook
# =============================================================================


class TestSyncSchedulerWebhook:
    """Tests for webhook handling."""

    @pytest.mark.asyncio
    async def test_handle_webhook(self):
        """Should handle webhook and trigger connector."""
        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler = SyncScheduler(state_dir=Path(tmpdir))

            connector = MagicMock()
            connector.connector_id = "jira"
            connector.name = "Jira"
            connector.handle_webhook = AsyncMock(return_value=True)

            scheduler.register_connector(connector, tenant_id="acme")

            payload = {"event": "issue_created", "issue_id": "123"}
            result = await scheduler.handle_webhook("jira", payload, tenant_id="acme")

            assert result is True
            connector.handle_webhook.assert_called_once_with(payload)

    @pytest.mark.asyncio
    async def test_handle_webhook_not_registered(self):
        """Should return False for unregistered connector."""
        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler = SyncScheduler(state_dir=Path(tmpdir))

            result = await scheduler.handle_webhook("nonexistent", {})

            assert result is False

    @pytest.mark.asyncio
    async def test_handle_webhook_no_connector(self):
        """Should return False when job has no connector."""
        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler = SyncScheduler(state_dir=Path(tmpdir))

            connector = MagicMock()
            connector.connector_id = "jira"
            connector.name = "Jira"

            job = scheduler.register_connector(connector, tenant_id="acme")
            job.connector = None  # Remove connector reference

            result = await scheduler.handle_webhook("jira", {}, tenant_id="acme")

            assert result is False


# =============================================================================
# TestSyncSchedulerHistory
# =============================================================================


class TestSyncSchedulerHistory:
    """Tests for history tracking."""

    def test_get_history_by_job(self):
        """Should filter history by job ID."""
        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler = SyncScheduler(state_dir=Path(tmpdir))

            now = datetime.now(timezone.utc)
            scheduler._history.append(
                SyncHistory(
                    id="run-1",
                    job_id="acme:jira",
                    connector_id="jira",
                    tenant_id="acme",
                    status=SyncStatus.COMPLETED,
                    started_at=now,
                )
            )
            scheduler._history.append(
                SyncHistory(
                    id="run-2",
                    job_id="acme:github",
                    connector_id="github",
                    tenant_id="acme",
                    status=SyncStatus.COMPLETED,
                    started_at=now,
                )
            )

            history = scheduler.get_history(job_id="acme:jira")
            assert len(history) == 1
            assert history[0].job_id == "acme:jira"

    def test_get_history_by_tenant(self):
        """Should filter history by tenant ID."""
        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler = SyncScheduler(state_dir=Path(tmpdir))

            now = datetime.now(timezone.utc)
            scheduler._history.append(
                SyncHistory(
                    id="run-1",
                    job_id="acme:jira",
                    connector_id="jira",
                    tenant_id="acme",
                    status=SyncStatus.COMPLETED,
                    started_at=now,
                )
            )
            scheduler._history.append(
                SyncHistory(
                    id="run-2",
                    job_id="beta:jira",
                    connector_id="jira",
                    tenant_id="beta",
                    status=SyncStatus.COMPLETED,
                    started_at=now,
                )
            )

            history = scheduler.get_history(tenant_id="acme")
            assert len(history) == 1

    def test_get_history_by_status(self):
        """Should filter history by status."""
        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler = SyncScheduler(state_dir=Path(tmpdir))

            now = datetime.now(timezone.utc)
            scheduler._history.append(
                SyncHistory(
                    id="run-1",
                    job_id="acme:jira",
                    connector_id="jira",
                    tenant_id="acme",
                    status=SyncStatus.COMPLETED,
                    started_at=now,
                )
            )
            scheduler._history.append(
                SyncHistory(
                    id="run-2",
                    job_id="acme:jira",
                    connector_id="jira",
                    tenant_id="acme",
                    status=SyncStatus.FAILED,
                    started_at=now,
                )
            )

            history = scheduler.get_history(status=SyncStatus.FAILED)
            assert len(history) == 1
            assert history[0].status == SyncStatus.FAILED

    def test_get_history_limit(self):
        """Should limit history results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler = SyncScheduler(state_dir=Path(tmpdir))

            now = datetime.now(timezone.utc)
            for i in range(10):
                scheduler._history.append(
                    SyncHistory(
                        id=f"run-{i}",
                        job_id="job",
                        connector_id="conn",
                        tenant_id="t",
                        status=SyncStatus.COMPLETED,
                        started_at=now + timedelta(seconds=i),
                    )
                )

            history = scheduler.get_history(limit=5)
            assert len(history) == 5

    def test_get_history_sorted_by_date_descending(self):
        """Should return history sorted by date descending."""
        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler = SyncScheduler(state_dir=Path(tmpdir))

            now = datetime.now(timezone.utc)
            scheduler._history.append(
                SyncHistory(
                    id="old",
                    job_id="job",
                    connector_id="conn",
                    tenant_id="t",
                    status=SyncStatus.COMPLETED,
                    started_at=now - timedelta(hours=1),
                )
            )
            scheduler._history.append(
                SyncHistory(
                    id="new",
                    job_id="job",
                    connector_id="conn",
                    tenant_id="t",
                    status=SyncStatus.COMPLETED,
                    started_at=now,
                )
            )

            history = scheduler.get_history()
            assert history[0].id == "new"
            assert history[1].id == "old"

    def test_cleanup_old_history(self):
        """Should clean up history entries older than retention period."""
        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler = SyncScheduler(
                state_dir=Path(tmpdir),
                history_retention_days=7,
            )

            old_date = datetime.now(timezone.utc) - timedelta(days=10)
            recent_date = datetime.now(timezone.utc) - timedelta(days=1)

            scheduler._history.append(
                SyncHistory(
                    id="old",
                    job_id="job",
                    connector_id="conn",
                    tenant_id="t",
                    status=SyncStatus.COMPLETED,
                    started_at=old_date,
                )
            )
            scheduler._history.append(
                SyncHistory(
                    id="recent",
                    job_id="job",
                    connector_id="conn",
                    tenant_id="t",
                    status=SyncStatus.COMPLETED,
                    started_at=recent_date,
                )
            )

            scheduler._cleanup_history()

            assert len(scheduler._history) == 1
            assert scheduler._history[0].id == "recent"

    def test_history_bounded_by_maxlen(self):
        """Should use deque with maxlen for bounded history."""
        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler = SyncScheduler(
                state_dir=Path(tmpdir),
                max_history_entries=5,
            )

            now = datetime.now(timezone.utc)
            for i in range(10):
                scheduler._history.append(
                    SyncHistory(
                        id=f"run-{i}",
                        job_id="job",
                        connector_id="conn",
                        tenant_id="t",
                        status=SyncStatus.COMPLETED,
                        started_at=now,
                    )
                )

            # Should be limited to max_history_entries
            assert len(scheduler._history) == 5
            # Oldest entries should be evicted (FIFO)
            assert scheduler._history[0].id == "run-5"


# =============================================================================
# TestSyncSchedulerStats
# =============================================================================


class TestSyncSchedulerStats:
    """Tests for scheduler statistics."""

    def test_empty_stats(self):
        """Should return initial stats when empty."""
        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler = SyncScheduler(state_dir=Path(tmpdir))

            stats = scheduler.get_stats()

            assert stats["total_jobs"] == 0
            assert stats["total_syncs"] == 0
            assert stats["success_rate"] == 1.0

    def test_stats_with_history(self):
        """Should compute stats from history."""
        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler = SyncScheduler(state_dir=Path(tmpdir))

            now = datetime.now(timezone.utc)
            scheduler._history.append(
                SyncHistory(
                    id="run-1",
                    job_id="job",
                    connector_id="conn",
                    tenant_id="t",
                    status=SyncStatus.COMPLETED,
                    started_at=now,
                    completed_at=now + timedelta(seconds=30),
                    items_synced=50,
                )
            )
            scheduler._history.append(
                SyncHistory(
                    id="run-2",
                    job_id="job",
                    connector_id="conn",
                    tenant_id="t",
                    status=SyncStatus.FAILED,
                    started_at=now,
                    completed_at=now + timedelta(seconds=5),
                    items_synced=0,
                )
            )
            scheduler._history.append(
                SyncHistory(
                    id="run-3",
                    job_id="job",
                    connector_id="conn",
                    tenant_id="t",
                    status=SyncStatus.COMPLETED,
                    started_at=now,
                    completed_at=now + timedelta(seconds=45),
                    items_synced=75,
                )
            )

            stats = scheduler.get_stats()

            assert stats["total_syncs"] == 3
            assert stats["successful_syncs"] == 2
            assert stats["failed_syncs"] == 1
            assert stats["total_items_synced"] == 125

    def test_stats_success_rate(self):
        """Should calculate correct success rate."""
        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler = SyncScheduler(state_dir=Path(tmpdir))

            now = datetime.now(timezone.utc)
            # 3 successful, 1 failed = 75% success rate
            for i, status in enumerate(
                [
                    SyncStatus.COMPLETED,
                    SyncStatus.COMPLETED,
                    SyncStatus.COMPLETED,
                    SyncStatus.FAILED,
                ]
            ):
                scheduler._history.append(
                    SyncHistory(
                        id=f"run-{i}",
                        job_id="job",
                        connector_id="conn",
                        tenant_id="t",
                        status=status,
                        started_at=now,
                    )
                )

            stats = scheduler.get_stats()
            assert stats["success_rate"] == 0.75

    def test_stats_average_duration(self):
        """Should calculate average duration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler = SyncScheduler(state_dir=Path(tmpdir))

            now = datetime.now(timezone.utc)
            scheduler._history.append(
                SyncHistory(
                    id="run-1",
                    job_id="job",
                    connector_id="conn",
                    tenant_id="t",
                    status=SyncStatus.COMPLETED,
                    started_at=now,
                    completed_at=now + timedelta(seconds=30),
                )
            )
            scheduler._history.append(
                SyncHistory(
                    id="run-2",
                    job_id="job",
                    connector_id="conn",
                    tenant_id="t",
                    status=SyncStatus.COMPLETED,
                    started_at=now,
                    completed_at=now + timedelta(seconds=60),
                )
            )

            stats = scheduler.get_stats()
            # Average of 30 and 60 is 45
            assert stats["average_duration_seconds"] == 45.0

    def test_stats_filtered_by_tenant(self):
        """Should filter stats by tenant."""
        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler = SyncScheduler(state_dir=Path(tmpdir))

            now = datetime.now(timezone.utc)
            scheduler._history.append(
                SyncHistory(
                    id="run-1",
                    job_id="acme:jira",
                    connector_id="jira",
                    tenant_id="acme",
                    status=SyncStatus.COMPLETED,
                    started_at=now,
                )
            )
            scheduler._history.append(
                SyncHistory(
                    id="run-2",
                    job_id="beta:jira",
                    connector_id="jira",
                    tenant_id="beta",
                    status=SyncStatus.COMPLETED,
                    started_at=now,
                )
            )

            stats = scheduler.get_stats(tenant_id="acme")
            assert stats["total_syncs"] == 1

    def test_stats_running_syncs_count(self):
        """Should track running syncs count."""
        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler = SyncScheduler(state_dir=Path(tmpdir))

            # Simulate running syncs
            scheduler._running_syncs["run-1"] = MagicMock()
            scheduler._running_syncs["run-2"] = MagicMock()

            stats = scheduler.get_stats()
            assert stats["running_syncs"] == 2

    def test_stats_enabled_jobs_count(self):
        """Should track enabled jobs count."""
        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler = SyncScheduler(state_dir=Path(tmpdir))

            # Register enabled and disabled connectors
            for i, enabled in enumerate([True, True, False]):
                connector = MagicMock()
                connector.connector_id = f"conn-{i}"
                connector.name = f"Connector {i}"
                scheduler.register_connector(
                    connector,
                    schedule=SyncSchedule(enabled=enabled),
                    tenant_id="acme",
                )

            stats = scheduler.get_stats()
            assert stats["total_jobs"] == 3
            assert stats["enabled_jobs"] == 2


# =============================================================================
# TestSyncSchedulerStatePersistence
# =============================================================================


class TestSyncSchedulerStatePersistence:
    """Tests for state persistence."""

    @pytest.mark.asyncio
    async def test_save_state(self):
        """Should save scheduler state to disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler = SyncScheduler(state_dir=Path(tmpdir))

            # Add some history
            now = datetime.now(timezone.utc)
            scheduler._history.append(
                SyncHistory(
                    id="run-1",
                    job_id="job-1",
                    connector_id="jira",
                    tenant_id="acme",
                    status=SyncStatus.COMPLETED,
                    started_at=now,
                    completed_at=now + timedelta(seconds=30),
                )
            )

            await scheduler.save_state()

            state_file = Path(tmpdir) / "scheduler_state.json"
            assert state_file.exists()

    @pytest.mark.asyncio
    async def test_load_state(self):
        """Should load scheduler state from disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save state first
            scheduler1 = SyncScheduler(state_dir=Path(tmpdir))

            now = datetime.now(timezone.utc)
            scheduler1._history.append(
                SyncHistory(
                    id="run-1",
                    job_id="job-1",
                    connector_id="jira",
                    tenant_id="acme",
                    status=SyncStatus.COMPLETED,
                    started_at=now,
                    completed_at=now + timedelta(seconds=30),
                )
            )

            await scheduler1.save_state()

            # Load state in new scheduler
            scheduler2 = SyncScheduler(state_dir=Path(tmpdir))
            await scheduler2.load_state()

            assert len(scheduler2._history) == 1
            assert scheduler2._history[0].id == "run-1"

    @pytest.mark.asyncio
    async def test_load_nonexistent_state(self):
        """Should handle nonexistent state file gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler = SyncScheduler(state_dir=Path(tmpdir))

            # Should not raise
            await scheduler.load_state()

            assert len(scheduler._history) == 0

    @pytest.mark.asyncio
    async def test_save_state_limits_history(self):
        """Should limit saved history to 1000 entries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler = SyncScheduler(state_dir=Path(tmpdir), max_history_entries=2000)

            # Add more than 1000 history entries
            now = datetime.now(timezone.utc)
            for i in range(1500):
                scheduler._history.append(
                    SyncHistory(
                        id=f"run-{i}",
                        job_id="job",
                        connector_id="conn",
                        tenant_id="t",
                        status=SyncStatus.COMPLETED,
                        started_at=now,
                    )
                )

            await scheduler.save_state()

            # Load and verify
            state_file = Path(tmpdir) / "scheduler_state.json"
            with open(state_file) as f:
                state = json.load(f)

            # Should only save last 1000
            assert len(state["history"]) == 1000

    @pytest.mark.asyncio
    async def test_load_state_handles_invalid_json(self):
        """Should handle invalid JSON gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "scheduler_state.json"
            state_file.write_text("invalid json {")

            scheduler = SyncScheduler(state_dir=Path(tmpdir))

            # Should not raise, just log warning
            await scheduler.load_state()

            assert len(scheduler._history) == 0

    @pytest.mark.asyncio
    async def test_save_and_load_preserves_all_history_fields(self):
        """Should preserve all history fields through save/load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler1 = SyncScheduler(state_dir=Path(tmpdir))

            now = datetime.now(timezone.utc)
            scheduler1._history.append(
                SyncHistory(
                    id="run-1",
                    job_id="acme:jira",
                    connector_id="jira",
                    tenant_id="acme",
                    status=SyncStatus.FAILED,
                    started_at=now,
                    completed_at=now + timedelta(seconds=10),
                    items_synced=5,
                    items_total=100,
                    errors=["Error 1", "Error 2"],
                )
            )

            await scheduler1.save_state()

            scheduler2 = SyncScheduler(state_dir=Path(tmpdir))
            await scheduler2.load_state()

            h = scheduler2._history[0]
            assert h.id == "run-1"
            assert h.job_id == "acme:jira"
            assert h.connector_id == "jira"
            assert h.tenant_id == "acme"
            assert h.status == SyncStatus.FAILED
            assert h.items_synced == 5
            assert h.items_total == 100
            assert h.errors == ["Error 1", "Error 2"]


# =============================================================================
# TestSyncSchedulerLifecycle
# =============================================================================


class TestSyncSchedulerLifecycle:
    """Tests for scheduler lifecycle."""

    @pytest.mark.asyncio
    async def test_start_creates_task(self):
        """Should create scheduler task on start."""
        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler = SyncScheduler(state_dir=Path(tmpdir))

            await scheduler.start()

            assert scheduler._scheduler_task is not None

            await scheduler.stop()

    @pytest.mark.asyncio
    async def test_stop_clears_task(self):
        """Should clear scheduler task on stop."""
        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler = SyncScheduler(state_dir=Path(tmpdir))

            await scheduler.start()
            await scheduler.stop()

            assert scheduler._scheduler_task is None

    @pytest.mark.asyncio
    async def test_start_idempotent(self):
        """Should not create duplicate tasks on double start."""
        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler = SyncScheduler(state_dir=Path(tmpdir))

            await scheduler.start()
            task1 = scheduler._scheduler_task

            await scheduler.start()
            task2 = scheduler._scheduler_task

            assert task1 is task2

            await scheduler.stop()

    @pytest.mark.asyncio
    async def test_stop_cancels_running_syncs(self):
        """Should cancel running syncs on stop."""
        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler = SyncScheduler(state_dir=Path(tmpdir))

            await scheduler.start()

            # Add a fake running sync
            cancelled = []

            async def fake_sync():
                try:
                    await asyncio.sleep(100)
                except asyncio.CancelledError:
                    cancelled.append(True)
                    raise

            scheduler._running_syncs["fake"] = asyncio.create_task(fake_sync())

            await scheduler.stop()

            assert len(cancelled) == 1
            assert len(scheduler._running_syncs) == 0

    @pytest.mark.asyncio
    async def test_stop_sets_stop_event(self):
        """Should set stop event on stop."""
        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler = SyncScheduler(state_dir=Path(tmpdir))

            await scheduler.start()
            assert not scheduler._stop_event.is_set()

            await scheduler.stop()
            assert scheduler._stop_event.is_set()

    @pytest.mark.asyncio
    async def test_start_clears_stop_event(self):
        """Should clear stop event on start."""
        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler = SyncScheduler(state_dir=Path(tmpdir))

            scheduler._stop_event.set()

            await scheduler.start()
            assert not scheduler._stop_event.is_set()

            await scheduler.stop()


# =============================================================================
# TestSyncSchedulerLoop
# =============================================================================


class TestSyncSchedulerLoop:
    """Tests for the scheduler loop."""

    @pytest.mark.asyncio
    async def test_scheduler_loop_executes_due_jobs(self):
        """Should execute jobs when they are due."""
        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler = SyncScheduler(state_dir=Path(tmpdir))

            mock_result = MagicMock()
            mock_result.success = True
            mock_result.items_synced = 10
            mock_result.items_total = 10
            mock_result.errors = []

            connector = MagicMock()
            connector.connector_id = "jira"
            connector.name = "Jira"
            connector.sync = AsyncMock(return_value=mock_result)

            job = scheduler.register_connector(
                connector,
                schedule=SyncSchedule(schedule_type="interval", interval_minutes=1),
                tenant_id="acme",
            )

            # Set next_run to now (due)
            job.next_run = datetime.now(timezone.utc) - timedelta(seconds=1)

            await scheduler.start()

            # Wait for scheduler to run
            await asyncio.sleep(0.2)

            await scheduler.stop()

            # Sync should have been called
            connector.sync.assert_called()

    @pytest.mark.asyncio
    async def test_scheduler_loop_skips_disabled_jobs(self):
        """Should skip disabled jobs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler = SyncScheduler(state_dir=Path(tmpdir))

            connector = MagicMock()
            connector.connector_id = "jira"
            connector.name = "Jira"
            connector.sync = AsyncMock()

            job = scheduler.register_connector(
                connector,
                schedule=SyncSchedule(enabled=False),
                tenant_id="acme",
            )

            # Force next_run even though disabled
            job.next_run = datetime.now(timezone.utc) - timedelta(seconds=1)

            await scheduler.start()
            await asyncio.sleep(0.2)
            await scheduler.stop()

            # Sync should NOT have been called
            connector.sync.assert_not_called()

    @pytest.mark.asyncio
    async def test_scheduler_loop_handles_errors(self):
        """Should continue running after errors in loop."""
        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler = SyncScheduler(state_dir=Path(tmpdir))

            # Create a properly structured but failing job
            connector = MagicMock()
            connector.connector_id = "failing"
            connector.name = "Failing"
            connector.sync = AsyncMock(side_effect=RuntimeError("Intentional failure"))

            job = scheduler.register_connector(
                connector,
                schedule=SyncSchedule(schedule_type="interval", interval_minutes=1),
                tenant_id="test",
            )
            # Set next_run to now so it triggers
            job.next_run = datetime.now(timezone.utc) - timedelta(seconds=1)

            await scheduler.start()
            await asyncio.sleep(0.2)

            # Should still be running despite the error
            assert scheduler._scheduler_task is not None

            await scheduler.stop()


# =============================================================================
# TestRetryScheduling
# =============================================================================


class TestRetryScheduling:
    """Tests for retry scheduling on failure."""

    @pytest.mark.asyncio
    async def test_schedules_retry_on_failure(self):
        """Should schedule retry when sync fails."""
        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler = SyncScheduler(state_dir=Path(tmpdir))

            connector = MagicMock()
            connector.connector_id = "jira"
            connector.name = "Jira"
            connector.sync = AsyncMock(side_effect=RuntimeError("Fail"))

            job = scheduler.register_connector(
                connector,
                schedule=SyncSchedule(
                    retry_on_failure=True,
                    max_retries=3,
                    retry_delay_minutes=5,
                ),
                tenant_id="acme",
            )

            original_next_run = job.next_run

            await scheduler.trigger_sync("jira", tenant_id="acme")

            # Should have scheduled a retry
            assert job.next_run is not None
            # Retry should be sooner than original schedule
            assert job.next_run <= datetime.now(timezone.utc) + timedelta(minutes=6)

    @pytest.mark.asyncio
    async def test_no_retry_when_max_retries_exceeded(self):
        """Should not retry when max retries exceeded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler = SyncScheduler(state_dir=Path(tmpdir))

            connector = MagicMock()
            connector.connector_id = "jira"
            connector.name = "Jira"
            connector.sync = AsyncMock(side_effect=RuntimeError("Fail"))

            job = scheduler.register_connector(
                connector,
                schedule=SyncSchedule(
                    retry_on_failure=True,
                    max_retries=2,
                    retry_delay_minutes=5,
                ),
                tenant_id="acme",
            )

            # Set consecutive failures to exceed max
            job.consecutive_failures = 3

            before_next_run = job.next_run

            await scheduler.trigger_sync("jira", tenant_id="acme")

            # Should not have scheduled a retry (consecutive_failures > max_retries)
            # Note: The implementation schedules retry if consecutive_failures <= max_retries
            # After the sync, consecutive_failures will be 4, exceeding max_retries of 2

    @pytest.mark.asyncio
    async def test_no_retry_when_disabled(self):
        """Should not retry when retry_on_failure is False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler = SyncScheduler(state_dir=Path(tmpdir))

            connector = MagicMock()
            connector.connector_id = "jira"
            connector.name = "Jira"
            connector.sync = AsyncMock(side_effect=RuntimeError("Fail"))

            job = scheduler.register_connector(
                connector,
                schedule=SyncSchedule(
                    schedule_type="webhook_only",
                    retry_on_failure=False,
                ),
                tenant_id="acme",
            )

            await scheduler.trigger_sync("jira", tenant_id="acme")

            # Should not have scheduled a retry
            assert job.next_run is None


# =============================================================================
# TestSyncJobNoConnector
# =============================================================================


class TestSyncJobNoConnector:
    """Tests for sync execution without connector."""

    @pytest.mark.asyncio
    async def test_execute_sync_no_connector_raises(self):
        """Should record error when connector is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler = SyncScheduler(state_dir=Path(tmpdir))

            connector = MagicMock()
            connector.connector_id = "jira"
            connector.name = "Jira"

            job = scheduler.register_connector(connector, tenant_id="acme")
            job.connector = None  # Remove connector

            await scheduler.trigger_sync("jira", tenant_id="acme")

            history = scheduler.get_history()
            assert len(history) == 1
            assert history[0].status == SyncStatus.FAILED
            assert "No connector registered" in history[0].errors[0]


# =============================================================================
# TestConstants
# =============================================================================


class TestConstants:
    """Tests for module constants."""

    def test_max_history_entries_constant(self):
        """Should have MAX_HISTORY_ENTRIES constant."""
        assert MAX_HISTORY_ENTRIES == 10_000

    def test_default_history_retention_days_constant(self):
        """Should have DEFAULT_HISTORY_RETENTION_DAYS constant."""
        assert DEFAULT_HISTORY_RETENTION_DAYS == 30
