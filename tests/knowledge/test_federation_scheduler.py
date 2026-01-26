"""
Tests for Federation Scheduler.

Tests the automatic scheduling of federation sync operations:
- Cron-based scheduling
- Interval-based scheduling
- Manual triggers
- Sync history tracking
- Error handling and retry logic
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.knowledge.mound.ops.federation_scheduler import (
    SyncMode,
    SyncScope,
    ScheduleStatus,
    TriggerType,
    FederationScheduleConfig,
    ScheduledSync,
    SyncRun,
    CronParser,
    FederationScheduler,
    get_federation_scheduler,
    init_federation_scheduler,
)


class TestSyncMode:
    """Test SyncMode enum."""

    def test_all_modes(self):
        """Test that all expected modes exist."""
        assert SyncMode.PUSH.value == "push"
        assert SyncMode.PULL.value == "pull"
        assert SyncMode.BIDIRECTIONAL.value == "bidirectional"


class TestSyncScope:
    """Test SyncScope enum."""

    def test_all_scopes(self):
        """Test that all expected scopes exist."""
        assert SyncScope.FULL.value == "full"
        assert SyncScope.SUMMARY.value == "summary"
        assert SyncScope.METADATA.value == "metadata"


class TestScheduleStatus:
    """Test ScheduleStatus enum."""

    def test_all_statuses(self):
        """Test that all expected statuses exist."""
        assert ScheduleStatus.ACTIVE.value == "active"
        assert ScheduleStatus.PAUSED.value == "paused"
        assert ScheduleStatus.DISABLED.value == "disabled"
        assert ScheduleStatus.RUNNING.value == "running"
        assert ScheduleStatus.ERROR.value == "error"


class TestCronParser:
    """Test CronParser."""

    def test_parse_every_minute(self):
        """Test parsing '* * * * *'."""
        fields = CronParser.parse("* * * * *")
        assert len(fields["minute"]) == 60
        assert len(fields["hour"]) == 24

    def test_parse_every_15_minutes(self):
        """Test parsing '*/15 * * * *'."""
        fields = CronParser.parse("*/15 * * * *")
        assert fields["minute"] == [0, 15, 30, 45]

    def test_parse_specific_hour(self):
        """Test parsing '0 2 * * *' (2 AM)."""
        fields = CronParser.parse("0 2 * * *")
        assert fields["minute"] == [0]
        assert fields["hour"] == [2]

    def test_parse_range(self):
        """Test parsing minute range '0-30 * * * *'."""
        fields = CronParser.parse("0-30 * * * *")
        assert fields["minute"] == list(range(0, 31))

    def test_parse_list(self):
        """Test parsing minute list '0,15,30,45 * * * *'."""
        fields = CronParser.parse("0,15,30,45 * * * *")
        assert fields["minute"] == [0, 15, 30, 45]

    def test_parse_invalid_expression(self):
        """Test parsing invalid expression raises error."""
        with pytest.raises(ValueError):
            CronParser.parse("invalid")

    def test_next_run_every_minute(self):
        """Test calculating next run for every minute."""
        now = datetime(2024, 1, 15, 10, 30, 30)
        next_run = CronParser.next_run("* * * * *", from_time=now)

        # Should be the next minute
        assert next_run.minute == 31
        assert next_run.hour == 10

    def test_next_run_specific_time(self):
        """Test calculating next run for specific time."""
        now = datetime(2024, 1, 15, 10, 0, 0)
        next_run = CronParser.next_run("0 14 * * *", from_time=now)

        # Should be 2 PM same day
        assert next_run.hour == 14
        assert next_run.minute == 0


class TestFederationScheduleConfig:
    """Test FederationScheduleConfig."""

    def test_default_config(self):
        """Test creating config with defaults."""
        config = FederationScheduleConfig(
            name="test_sync",
            region_id="us-west-2",
            workspace_id="ws_123",
        )

        assert config.name == "test_sync"
        assert config.region_id == "us-west-2"
        assert config.workspace_id == "ws_123"
        assert config.sync_mode == SyncMode.BIDIRECTIONAL
        assert config.sync_scope == SyncScope.SUMMARY
        assert config.trigger_type == TriggerType.CRON
        assert config.interval_minutes == 15
        assert config.max_items_per_sync == 1000
        assert config.enabled is True

    def test_custom_config(self):
        """Test creating config with custom values."""
        config = FederationScheduleConfig(
            name="custom_sync",
            region_id="eu-central-1",
            workspace_id="ws_456",
            sync_mode=SyncMode.PUSH,
            sync_scope=SyncScope.METADATA,
            cron="0 */6 * * *",
            max_items_per_sync=500,
            notify_on_error=False,
        )

        assert config.sync_mode == SyncMode.PUSH
        assert config.sync_scope == SyncScope.METADATA
        assert config.cron == "0 */6 * * *"
        assert config.max_items_per_sync == 500
        assert config.notify_on_error is False


class TestScheduledSync:
    """Test ScheduledSync."""

    def test_to_dict(self):
        """Test converting schedule to dict."""
        config = FederationScheduleConfig(
            name="test",
            region_id="us-west-2",
            workspace_id="ws_123",
            cron="*/15 * * * *",
        )

        schedule = ScheduledSync(
            schedule_id="fed_sched_123",
            config=config,
            status=ScheduleStatus.ACTIVE,
            run_count=5,
            error_count=1,
        )

        data = schedule.to_dict()
        assert data["schedule_id"] == "fed_sched_123"
        assert data["name"] == "test"
        assert data["region_id"] == "us-west-2"
        assert data["status"] == "active"
        assert data["run_count"] == 5
        assert data["error_count"] == 1


class TestSyncRun:
    """Test SyncRun."""

    def test_to_dict(self):
        """Test converting run to dict."""
        run = SyncRun(
            run_id="sync_run_123",
            schedule_id="fed_sched_456",
            region_id="us-west-2",
            workspace_id="ws_789",
            sync_mode=SyncMode.BIDIRECTIONAL,
            started_at=datetime(2024, 1, 15, 10, 0, 0),
            completed_at=datetime(2024, 1, 15, 10, 0, 5),
            status="completed",
            items_pushed=50,
            items_pulled=30,
            duration_ms=5000,
        )

        data = run.to_dict()
        assert data["run_id"] == "sync_run_123"
        assert data["sync_mode"] == "bidirectional"
        assert data["items_pushed"] == 50
        assert data["items_pulled"] == 30
        assert data["status"] == "completed"


class TestFederationScheduler:
    """Test FederationScheduler."""

    def test_add_schedule_cron(self):
        """Test adding a cron-based schedule."""
        scheduler = FederationScheduler()

        config = FederationScheduleConfig(
            name="hourly_sync",
            region_id="us-west-2",
            workspace_id="ws_123",
            cron="0 * * * *",
        )

        schedule = scheduler.add_schedule(config)

        assert schedule.schedule_id.startswith("fed_sched_")
        assert schedule.config.name == "hourly_sync"
        assert schedule.status == ScheduleStatus.ACTIVE
        assert schedule.next_run is not None

    def test_add_schedule_interval(self):
        """Test adding an interval-based schedule."""
        scheduler = FederationScheduler()

        config = FederationScheduleConfig(
            name="frequent_sync",
            region_id="us-west-2",
            workspace_id="ws_123",
            trigger_type=TriggerType.INTERVAL,
            interval_minutes=5,
        )

        schedule = scheduler.add_schedule(config)

        assert schedule.next_run is not None
        # Next run should be approximately 5 minutes from now
        expected = datetime.now() + timedelta(minutes=5)
        assert abs((schedule.next_run - expected).total_seconds()) < 10

    def test_add_schedule_disabled(self):
        """Test adding a disabled schedule."""
        scheduler = FederationScheduler()

        config = FederationScheduleConfig(
            name="disabled_sync",
            region_id="us-west-2",
            workspace_id="ws_123",
            enabled=False,
        )

        schedule = scheduler.add_schedule(config)
        assert schedule.status == ScheduleStatus.PAUSED

    def test_remove_schedule(self):
        """Test removing a schedule."""
        scheduler = FederationScheduler()

        config = FederationScheduleConfig(
            name="temp_sync",
            region_id="us-west-2",
            workspace_id="ws_123",
        )

        schedule = scheduler.add_schedule(config)
        schedule_id = schedule.schedule_id

        assert scheduler.get_schedule(schedule_id) is not None

        result = scheduler.remove_schedule(schedule_id)
        assert result is True
        assert scheduler.get_schedule(schedule_id) is None

    def test_remove_nonexistent_schedule(self):
        """Test removing a nonexistent schedule."""
        scheduler = FederationScheduler()
        result = scheduler.remove_schedule("nonexistent")
        assert result is False

    def test_pause_and_resume_schedule(self):
        """Test pausing and resuming a schedule."""
        scheduler = FederationScheduler()

        config = FederationScheduleConfig(
            name="pausable_sync",
            region_id="us-west-2",
            workspace_id="ws_123",
            cron="0 * * * *",
        )

        schedule = scheduler.add_schedule(config)
        schedule_id = schedule.schedule_id

        # Pause
        result = scheduler.pause_schedule(schedule_id)
        assert result is True
        assert scheduler.get_schedule(schedule_id).status == ScheduleStatus.PAUSED

        # Resume
        result = scheduler.resume_schedule(schedule_id)
        assert result is True
        assert scheduler.get_schedule(schedule_id).status == ScheduleStatus.ACTIVE

    def test_list_schedules(self):
        """Test listing schedules with filters."""
        scheduler = FederationScheduler()

        # Add multiple schedules
        for i, region in enumerate(["us-west-2", "eu-central-1", "us-west-2"]):
            config = FederationScheduleConfig(
                name=f"sync_{i}",
                region_id=region,
                workspace_id=f"ws_{i}",
            )
            scheduler.add_schedule(config)

        # List all
        all_schedules = scheduler.list_schedules()
        assert len(all_schedules) == 3

        # Filter by region
        us_west_schedules = scheduler.list_schedules(region_id="us-west-2")
        assert len(us_west_schedules) == 2

        # Filter by workspace
        ws_0_schedules = scheduler.list_schedules(workspace_id="ws_0")
        assert len(ws_0_schedules) == 1

    @pytest.mark.asyncio
    async def test_trigger_sync(self):
        """Test manually triggering a sync."""
        mock_callback = AsyncMock(
            return_value={
                "items_pushed": 10,
                "items_pulled": 5,
                "items_conflicted": 0,
            }
        )

        scheduler = FederationScheduler(sync_callback=mock_callback)

        config = FederationScheduleConfig(
            name="manual_sync",
            region_id="us-west-2",
            workspace_id="ws_123",
        )

        schedule = scheduler.add_schedule(config)

        run = await scheduler.trigger_sync(schedule.schedule_id)

        assert run is not None
        assert run.status == "completed"
        assert run.items_pushed == 10
        assert run.items_pulled == 5
        mock_callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_trigger_sync_error(self):
        """Test sync error handling."""
        mock_callback = AsyncMock(side_effect=Exception("Sync failed"))

        scheduler = FederationScheduler(sync_callback=mock_callback)

        config = FederationScheduleConfig(
            name="error_sync",
            region_id="us-west-2",
            workspace_id="ws_123",
            max_retries=2,
        )

        schedule = scheduler.add_schedule(config)

        run = await scheduler.trigger_sync(schedule.schedule_id)

        assert run.status == "error"
        assert "Sync failed" in run.error_message
        assert schedule.error_count == 1
        assert schedule.consecutive_errors == 1

    @pytest.mark.asyncio
    async def test_trigger_nonexistent_schedule(self):
        """Test triggering nonexistent schedule."""
        scheduler = FederationScheduler()
        run = await scheduler.trigger_sync("nonexistent")
        assert run is None

    def test_get_history(self):
        """Test getting sync history."""
        scheduler = FederationScheduler()

        # Manually add some history
        for i in range(5):
            run = SyncRun(
                run_id=f"run_{i}",
                schedule_id=f"sched_{i % 2}",
                region_id="us-west-2",
                workspace_id="ws_123",
                sync_mode=SyncMode.PUSH,
                started_at=datetime.now(),
                status="completed",
            )
            scheduler._sync_history.append(run)

        # Get all history
        history = scheduler.get_history(limit=10)
        assert len(history) == 5

        # Get filtered history
        filtered = scheduler.get_history(schedule_id="sched_0")
        assert len(filtered) == 3

    def test_get_stats(self):
        """Test getting scheduler statistics."""
        scheduler = FederationScheduler()

        # Add some schedules with different statuses
        for i, status in enumerate(
            [ScheduleStatus.ACTIVE, ScheduleStatus.ACTIVE, ScheduleStatus.PAUSED]
        ):
            config = FederationScheduleConfig(
                name=f"sync_{i}",
                region_id="us-west-2",
                workspace_id=f"ws_{i}",
                enabled=(status == ScheduleStatus.ACTIVE),
            )
            schedule = scheduler.add_schedule(config)
            schedule.run_count = i * 10
            schedule.error_count = i

        stats = scheduler.get_stats()

        assert stats["schedules"]["total"] == 3
        assert stats["schedules"]["active"] == 2
        assert stats["schedules"]["paused"] == 1
        assert stats["runs"]["total"] == 30  # 0 + 10 + 20
        assert stats["runs"]["total_errors"] == 3  # 0 + 1 + 2
        assert stats["running"] is False

    @pytest.mark.asyncio
    async def test_scheduler_start_stop(self):
        """Test starting and stopping the scheduler."""
        scheduler = FederationScheduler()

        assert scheduler._running is False

        await scheduler.start()
        assert scheduler._running is True
        assert scheduler._task is not None

        await scheduler.stop()
        assert scheduler._running is False

    @pytest.mark.asyncio
    async def test_consecutive_error_disables_schedule(self):
        """Test that consecutive errors disable the schedule."""
        mock_callback = AsyncMock(side_effect=Exception("Persistent error"))

        scheduler = FederationScheduler(sync_callback=mock_callback)

        config = FederationScheduleConfig(
            name="failing_sync",
            region_id="us-west-2",
            workspace_id="ws_123",
            max_retries=3,
        )

        schedule = scheduler.add_schedule(config)

        # Trigger multiple failures
        for _ in range(3):
            await scheduler.trigger_sync(schedule.schedule_id)

        assert schedule.consecutive_errors == 3
        assert schedule.status == ScheduleStatus.ERROR


class TestGlobalScheduler:
    """Test global scheduler functions."""

    def test_get_federation_scheduler(self):
        """Test getting global scheduler instance."""
        # Reset global
        import aragora.knowledge.mound.ops.federation_scheduler as fed_module

        fed_module._federation_scheduler = None

        scheduler1 = get_federation_scheduler()
        scheduler2 = get_federation_scheduler()

        # Should be same instance
        assert scheduler1 is scheduler2

    @pytest.mark.asyncio
    async def test_init_federation_scheduler(self):
        """Test initializing global scheduler."""
        # Reset global
        import aragora.knowledge.mound.ops.federation_scheduler as fed_module

        fed_module._federation_scheduler = None

        scheduler = await init_federation_scheduler()
        assert scheduler._running is True

        await scheduler.stop()
