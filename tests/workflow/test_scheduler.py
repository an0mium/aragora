"""Tests for WorkflowScheduler with cron integration."""

from __future__ import annotations

import asyncio
import tempfile
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.workflow.scheduler import (
    ScheduleEntry,
    WorkflowScheduler,
    _ScheduleStore,
    cron_matches,
    get_workflow_scheduler,
    parse_cron_expression,
    reset_workflow_scheduler,
)


# ---------------------------------------------------------------------------
# Cron parsing tests
# ---------------------------------------------------------------------------


class TestParseCronExpression:
    def test_every_minute(self):
        parsed = parse_cron_expression("* * * * *")
        assert parsed[0] == set(range(0, 60))  # minute
        assert parsed[1] == set(range(0, 24))  # hour
        assert parsed[2] == set(range(1, 32))  # day of month
        assert parsed[3] == set(range(1, 13))  # month
        assert parsed[4] == set(range(0, 7))   # day of week

    def test_specific_values(self):
        parsed = parse_cron_expression("30 9 1 6 3")
        assert parsed[0] == {30}
        assert parsed[1] == {9}
        assert parsed[2] == {1}
        assert parsed[3] == {6}
        assert parsed[4] == {3}

    def test_step_values(self):
        parsed = parse_cron_expression("*/5 */2 * * *")
        assert parsed[0] == {0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55}
        assert parsed[1] == {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22}

    def test_ranges(self):
        parsed = parse_cron_expression("0 9 * * 1-5")
        assert parsed[0] == {0}
        assert parsed[1] == {9}
        assert parsed[4] == {1, 2, 3, 4, 5}

    def test_comma_list(self):
        parsed = parse_cron_expression("0,30 * * * *")
        assert parsed[0] == {0, 30}

    def test_every_hour(self):
        parsed = parse_cron_expression("0 * * * *")
        assert parsed[0] == {0}
        assert parsed[1] == set(range(0, 24))

    def test_daily_at_midnight(self):
        parsed = parse_cron_expression("0 0 * * *")
        assert parsed[0] == {0}
        assert parsed[1] == {0}

    def test_invalid_too_few_fields(self):
        with pytest.raises(ValueError, match="5 fields"):
            parse_cron_expression("* * *")

    def test_invalid_too_many_fields(self):
        with pytest.raises(ValueError, match="5 fields"):
            parse_cron_expression("* * * * * *")

    def test_invalid_value_out_of_range(self):
        with pytest.raises(ValueError):
            parse_cron_expression("60 * * * *")

    def test_invalid_step_zero(self):
        with pytest.raises(ValueError):
            parse_cron_expression("*/0 * * * *")


class TestCronMatches:
    def test_every_minute_matches(self):
        parsed = parse_cron_expression("* * * * *")
        dt = datetime(2026, 2, 14, 10, 30, 0, tzinfo=timezone.utc)
        assert cron_matches(parsed, dt) is True

    def test_specific_time_matches(self):
        parsed = parse_cron_expression("30 10 14 2 *")
        dt = datetime(2026, 2, 14, 10, 30, 0, tzinfo=timezone.utc)
        assert cron_matches(parsed, dt) is True

    def test_specific_time_no_match_minute(self):
        parsed = parse_cron_expression("0 10 14 2 *")
        dt = datetime(2026, 2, 14, 10, 30, 0, tzinfo=timezone.utc)
        assert cron_matches(parsed, dt) is False

    def test_weekday_match(self):
        # 2026-02-14 is a Saturday (Python weekday=5, cron dow=6)
        parsed = parse_cron_expression("* * * * 6")
        dt = datetime(2026, 2, 14, 10, 30, 0, tzinfo=timezone.utc)
        assert cron_matches(parsed, dt) is True

    def test_weekday_no_match(self):
        # 2026-02-14 is Saturday, cron 1-5 = Mon-Fri
        parsed = parse_cron_expression("* * * * 1-5")
        dt = datetime(2026, 2, 14, 10, 30, 0, tzinfo=timezone.utc)
        assert cron_matches(parsed, dt) is False

    def test_every_5_minutes_match(self):
        parsed = parse_cron_expression("*/5 * * * *")
        dt = datetime(2026, 2, 14, 10, 15, 0, tzinfo=timezone.utc)
        assert cron_matches(parsed, dt) is True

    def test_every_5_minutes_no_match(self):
        parsed = parse_cron_expression("*/5 * * * *")
        dt = datetime(2026, 2, 14, 10, 13, 0, tzinfo=timezone.utc)
        assert cron_matches(parsed, dt) is False


# ---------------------------------------------------------------------------
# Schedule store tests
# ---------------------------------------------------------------------------


class TestScheduleStore:
    def test_insert_and_get(self, tmp_path):
        store = _ScheduleStore(str(tmp_path / "test.db"))
        entry = ScheduleEntry(
            id="sched_abc",
            workflow_id="wf-1",
            cron_expr="* * * * *",
            inputs={"key": "val"},
            enabled=True,
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
        )
        store.insert(entry)
        result = store.get("sched_abc")
        assert result is not None
        assert result.workflow_id == "wf-1"
        assert result.inputs == {"key": "val"}
        store.close()

    def test_list_all(self, tmp_path):
        store = _ScheduleStore(str(tmp_path / "test.db"))
        for i in range(3):
            store.insert(ScheduleEntry(
                id=f"sched_{i}",
                workflow_id=f"wf-{i}",
                cron_expr="* * * * *",
                created_at=f"2026-01-0{i+1}T00:00:00+00:00",
                updated_at=f"2026-01-0{i+1}T00:00:00+00:00",
            ))
        assert len(store.list_all()) == 3
        store.close()

    def test_list_enabled(self, tmp_path):
        store = _ScheduleStore(str(tmp_path / "test.db"))
        store.insert(ScheduleEntry(
            id="s1", workflow_id="w1", cron_expr="* * * * *",
            enabled=True, created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
        ))
        store.insert(ScheduleEntry(
            id="s2", workflow_id="w2", cron_expr="* * * * *",
            enabled=False, created_at="2026-01-02T00:00:00+00:00",
            updated_at="2026-01-02T00:00:00+00:00",
        ))
        enabled = store.list_enabled()
        assert len(enabled) == 1
        assert enabled[0].id == "s1"
        store.close()

    def test_update(self, tmp_path):
        store = _ScheduleStore(str(tmp_path / "test.db"))
        entry = ScheduleEntry(
            id="s1", workflow_id="w1", cron_expr="* * * * *",
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
        )
        store.insert(entry)
        entry.cron_expr = "0 9 * * *"
        entry.enabled = False
        store.update(entry)
        result = store.get("s1")
        assert result.cron_expr == "0 9 * * *"
        assert result.enabled is False
        store.close()

    def test_delete(self, tmp_path):
        store = _ScheduleStore(str(tmp_path / "test.db"))
        store.insert(ScheduleEntry(
            id="s1", workflow_id="w1", cron_expr="* * * * *",
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
        ))
        assert store.delete("s1") is True
        assert store.get("s1") is None
        assert store.delete("s1") is False
        store.close()


# ---------------------------------------------------------------------------
# WorkflowScheduler CRUD tests
# ---------------------------------------------------------------------------


class TestWorkflowSchedulerCRUD:
    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        reset_workflow_scheduler()
        self.db_path = str(tmp_path / "schedules.db")
        self.scheduler = WorkflowScheduler(db_path=self.db_path)

    def test_add_schedule(self):
        sid = self.scheduler.add_schedule("wf-1", "*/5 * * * *")
        assert sid.startswith("sched_")
        entry = self.scheduler.get_schedule(sid)
        assert entry is not None
        assert entry.workflow_id == "wf-1"
        assert entry.cron_expr == "*/5 * * * *"
        assert entry.enabled is True

    def test_add_schedule_with_inputs(self):
        sid = self.scheduler.add_schedule(
            "wf-1", "0 9 * * 1-5",
            inputs={"report_type": "daily"},
            enabled=False,
        )
        entry = self.scheduler.get_schedule(sid)
        assert entry.inputs == {"report_type": "daily"}
        assert entry.enabled is False

    def test_add_schedule_invalid_cron(self):
        with pytest.raises(ValueError):
            self.scheduler.add_schedule("wf-1", "bad cron")

    def test_list_schedules(self):
        self.scheduler.add_schedule("wf-1", "* * * * *")
        self.scheduler.add_schedule("wf-2", "0 * * * *")
        entries = self.scheduler.list_schedules()
        assert len(entries) == 2

    def test_update_schedule_cron(self):
        sid = self.scheduler.add_schedule("wf-1", "* * * * *")
        updated = self.scheduler.update_schedule(sid, cron_expr="0 9 * * *")
        assert updated is not None
        assert updated.cron_expr == "0 9 * * *"

    def test_update_schedule_enabled(self):
        sid = self.scheduler.add_schedule("wf-1", "* * * * *")
        updated = self.scheduler.update_schedule(sid, enabled=False)
        assert updated.enabled is False

    def test_update_schedule_inputs(self):
        sid = self.scheduler.add_schedule("wf-1", "* * * * *")
        updated = self.scheduler.update_schedule(sid, inputs={"a": 1})
        assert updated.inputs == {"a": 1}

    def test_update_schedule_invalid_cron(self):
        sid = self.scheduler.add_schedule("wf-1", "* * * * *")
        with pytest.raises(ValueError):
            self.scheduler.update_schedule(sid, cron_expr="nope")

    def test_update_nonexistent(self):
        result = self.scheduler.update_schedule("no_such_id", enabled=False)
        assert result is None

    def test_remove_schedule(self):
        sid = self.scheduler.add_schedule("wf-1", "* * * * *")
        assert self.scheduler.remove_schedule(sid) is True
        assert self.scheduler.get_schedule(sid) is None

    def test_remove_nonexistent(self):
        assert self.scheduler.remove_schedule("no_such_id") is False

    def test_get_nonexistent(self):
        assert self.scheduler.get_schedule("no_such_id") is None


# ---------------------------------------------------------------------------
# Persistence tests
# ---------------------------------------------------------------------------


class TestSchedulerPersistence:
    def test_persists_across_instances(self, tmp_path):
        db_path = str(tmp_path / "persist.db")

        # Create schedule with first instance
        s1 = WorkflowScheduler(db_path=db_path)
        sid = s1.add_schedule("wf-persist", "0 9 * * *", inputs={"x": 42})
        s1._store.close()

        # Load with second instance
        s2 = WorkflowScheduler(db_path=db_path)
        entry = s2.get_schedule(sid)
        assert entry is not None
        assert entry.workflow_id == "wf-persist"
        assert entry.cron_expr == "0 9 * * *"
        assert entry.inputs == {"x": 42}
        s2._store.close()


# ---------------------------------------------------------------------------
# Tick / execution tests
# ---------------------------------------------------------------------------


class TestSchedulerTick:
    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        reset_workflow_scheduler()
        self.db_path = str(tmp_path / "tick.db")
        self.scheduler = WorkflowScheduler(db_path=self.db_path, tick_interval=0.1)

    @pytest.mark.asyncio
    async def test_tick_enqueues_due_workflow(self):
        self.scheduler.add_schedule("wf-due", "* * * * *")

        with patch.object(
            self.scheduler, "_enqueue_execution", new_callable=AsyncMock
        ) as mock_enqueue:
            await self.scheduler._tick()
            assert mock_enqueue.call_count == 1
            call_entry = mock_enqueue.call_args[0][0]
            assert call_entry.workflow_id == "wf-due"

    @pytest.mark.asyncio
    async def test_tick_skips_disabled_schedule(self):
        sid = self.scheduler.add_schedule("wf-disabled", "* * * * *", enabled=False)

        with patch.object(
            self.scheduler, "_enqueue_execution", new_callable=AsyncMock
        ) as mock_enqueue:
            await self.scheduler._tick()
            mock_enqueue.assert_not_called()

    @pytest.mark.asyncio
    async def test_tick_skips_non_matching_cron(self):
        # Schedule for a specific minute that is unlikely to be "now"
        self.scheduler.add_schedule("wf-future", "59 23 31 12 *")

        with patch.object(
            self.scheduler, "_enqueue_execution", new_callable=AsyncMock
        ) as mock_enqueue:
            # Patch datetime to a time that won't match
            with patch(
                "aragora.workflow.scheduler.datetime"
            ) as mock_dt:
                mock_dt.now.return_value = datetime(2026, 6, 15, 10, 0, 0, tzinfo=timezone.utc)
                mock_dt.fromisoformat = datetime.fromisoformat
                await self.scheduler._tick()
            mock_enqueue.assert_not_called()

    @pytest.mark.asyncio
    async def test_tick_prevents_double_fire(self):
        sid = self.scheduler.add_schedule("wf-once", "* * * * *")

        with patch.object(
            self.scheduler, "_enqueue_execution", new_callable=AsyncMock
        ) as mock_enqueue:
            # First tick fires
            await self.scheduler._tick()
            assert mock_enqueue.call_count == 1

            # Second tick within the same minute should not fire again
            await self.scheduler._tick()
            assert mock_enqueue.call_count == 1

    @pytest.mark.asyncio
    async def test_tick_updates_run_count(self):
        sid = self.scheduler.add_schedule("wf-count", "* * * * *")

        with patch.object(
            self.scheduler, "_enqueue_execution", new_callable=AsyncMock
        ):
            await self.scheduler._tick()

        entry = self.scheduler.get_schedule(sid)
        assert entry.run_count == 1
        assert entry.last_run_at is not None


# ---------------------------------------------------------------------------
# Lifecycle tests
# ---------------------------------------------------------------------------


class TestSchedulerLifecycle:
    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        reset_workflow_scheduler()
        self.db_path = str(tmp_path / "lifecycle.db")

    @pytest.mark.asyncio
    async def test_start_and_stop(self):
        scheduler = WorkflowScheduler(db_path=self.db_path, tick_interval=0.05)
        await scheduler.start()
        assert scheduler._running is True
        assert scheduler._task is not None

        await scheduler.stop()
        assert scheduler._running is False

    @pytest.mark.asyncio
    async def test_double_start_is_safe(self):
        scheduler = WorkflowScheduler(db_path=self.db_path, tick_interval=0.05)
        await scheduler.start()
        await scheduler.start()  # should warn but not error
        assert scheduler._running is True
        await scheduler.stop()

    @pytest.mark.asyncio
    async def test_double_stop_is_safe(self):
        scheduler = WorkflowScheduler(db_path=self.db_path, tick_interval=0.05)
        await scheduler.start()
        await scheduler.stop()
        await scheduler.stop()  # should not error

    @pytest.mark.asyncio
    async def test_stop_without_start(self):
        scheduler = WorkflowScheduler(db_path=self.db_path)
        await scheduler.stop()  # no-op, no error


# ---------------------------------------------------------------------------
# Singleton tests
# ---------------------------------------------------------------------------


class TestSingleton:
    def test_get_returns_same_instance(self, tmp_path):
        reset_workflow_scheduler()
        db = str(tmp_path / "singleton.db")
        s1 = get_workflow_scheduler(db_path=db)
        s2 = get_workflow_scheduler()
        assert s1 is s2
        reset_workflow_scheduler()

    def test_reset_clears_singleton(self, tmp_path):
        reset_workflow_scheduler()
        db = str(tmp_path / "singleton.db")
        s1 = get_workflow_scheduler(db_path=db)
        reset_workflow_scheduler()
        db2 = str(tmp_path / "singleton2.db")
        s2 = get_workflow_scheduler(db_path=db2)
        assert s1 is not s2
        reset_workflow_scheduler()


# ---------------------------------------------------------------------------
# ScheduleEntry serialization
# ---------------------------------------------------------------------------


class TestScheduleEntry:
    def test_to_dict(self):
        entry = ScheduleEntry(
            id="sched_abc",
            workflow_id="wf-1",
            cron_expr="0 9 * * 1-5",
            inputs={"report": True},
            enabled=True,
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
            run_count=5,
        )
        d = entry.to_dict()
        assert d["id"] == "sched_abc"
        assert d["workflow_id"] == "wf-1"
        assert d["cron_expr"] == "0 9 * * 1-5"
        assert d["inputs"] == {"report": True}
        assert d["enabled"] is True
        assert d["run_count"] == 5
