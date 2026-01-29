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
"""

from __future__ import annotations

import asyncio
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.connectors.enterprise.sync.scheduler import (
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
            assert scheduler.history_retention_days == 30

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
            scheduler._history = [
                SyncHistory(
                    id="run-1",
                    job_id="acme:jira",
                    connector_id="jira",
                    tenant_id="acme",
                    status=SyncStatus.COMPLETED,
                    started_at=now,
                ),
                SyncHistory(
                    id="run-2",
                    job_id="acme:github",
                    connector_id="github",
                    tenant_id="acme",
                    status=SyncStatus.COMPLETED,
                    started_at=now,
                ),
            ]

            history = scheduler.get_history(job_id="acme:jira")
            assert len(history) == 1
            assert history[0].job_id == "acme:jira"

    def test_get_history_by_tenant(self):
        """Should filter history by tenant ID."""
        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler = SyncScheduler(state_dir=Path(tmpdir))

            now = datetime.now(timezone.utc)
            scheduler._history = [
                SyncHistory(
                    id="run-1",
                    job_id="acme:jira",
                    connector_id="jira",
                    tenant_id="acme",
                    status=SyncStatus.COMPLETED,
                    started_at=now,
                ),
                SyncHistory(
                    id="run-2",
                    job_id="beta:jira",
                    connector_id="jira",
                    tenant_id="beta",
                    status=SyncStatus.COMPLETED,
                    started_at=now,
                ),
            ]

            history = scheduler.get_history(tenant_id="acme")
            assert len(history) == 1

    def test_get_history_by_status(self):
        """Should filter history by status."""
        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler = SyncScheduler(state_dir=Path(tmpdir))

            now = datetime.now(timezone.utc)
            scheduler._history = [
                SyncHistory(
                    id="run-1",
                    job_id="acme:jira",
                    connector_id="jira",
                    tenant_id="acme",
                    status=SyncStatus.COMPLETED,
                    started_at=now,
                ),
                SyncHistory(
                    id="run-2",
                    job_id="acme:jira",
                    connector_id="jira",
                    tenant_id="acme",
                    status=SyncStatus.FAILED,
                    started_at=now,
                ),
            ]

            history = scheduler.get_history(status=SyncStatus.FAILED)
            assert len(history) == 1
            assert history[0].status == SyncStatus.FAILED

    def test_cleanup_old_history(self):
        """Should clean up history entries older than retention period."""
        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler = SyncScheduler(
                state_dir=Path(tmpdir),
                history_retention_days=7,
            )

            old_date = datetime.now(timezone.utc) - timedelta(days=10)
            recent_date = datetime.now(timezone.utc) - timedelta(days=1)

            scheduler._history = [
                SyncHistory(
                    id="old",
                    job_id="job",
                    connector_id="conn",
                    tenant_id="t",
                    status=SyncStatus.COMPLETED,
                    started_at=old_date,
                ),
                SyncHistory(
                    id="recent",
                    job_id="job",
                    connector_id="conn",
                    tenant_id="t",
                    status=SyncStatus.COMPLETED,
                    started_at=recent_date,
                ),
            ]

            scheduler._cleanup_history()

            assert len(scheduler._history) == 1
            assert scheduler._history[0].id == "recent"


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
            scheduler._history = [
                SyncHistory(
                    id="run-1",
                    job_id="job",
                    connector_id="conn",
                    tenant_id="t",
                    status=SyncStatus.COMPLETED,
                    started_at=now,
                    completed_at=now + timedelta(seconds=30),
                    items_synced=50,
                ),
                SyncHistory(
                    id="run-2",
                    job_id="job",
                    connector_id="conn",
                    tenant_id="t",
                    status=SyncStatus.FAILED,
                    started_at=now,
                    completed_at=now + timedelta(seconds=5),
                    items_synced=0,
                ),
                SyncHistory(
                    id="run-3",
                    job_id="job",
                    connector_id="conn",
                    tenant_id="t",
                    status=SyncStatus.COMPLETED,
                    started_at=now,
                    completed_at=now + timedelta(seconds=45),
                    items_synced=75,
                ),
            ]

            stats = scheduler.get_stats()

            assert stats["total_syncs"] == 3
            assert stats["successful_syncs"] == 2
            assert stats["failed_syncs"] == 1
            assert stats["total_items_synced"] == 125


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
