"""Tests for the settlement review scheduler."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from aragora.debate.settlement_scheduler import (
    DEFAULT_CHECK_INTERVAL_SECONDS,
    SchedulerStats,
    SettlementReviewEvent,
    SettlementReviewScheduler,
    get_scheduler,
    reset_scheduler,
    set_scheduler,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Reset the global scheduler singleton between tests."""
    reset_scheduler()
    yield
    reset_scheduler()


@pytest.fixture
def mock_tracker():
    """Create a mock EpistemicSettlementTracker."""
    tracker = MagicMock()
    tracker.get_due_settlements.return_value = []
    tracker.get_settlement.return_value = None
    return tracker


@pytest.fixture
def scheduler(mock_tracker):
    """Create a scheduler instance with a mock tracker."""
    return SettlementReviewScheduler(
        tracker=mock_tracker,
        check_interval_seconds=1,
    )


# ---------------------------------------------------------------------------
# SettlementReviewEvent tests
# ---------------------------------------------------------------------------


class TestSettlementReviewEvent:
    """Tests for the SettlementReviewEvent dataclass."""

    def test_creation(self):
        event = SettlementReviewEvent(
            debate_id="d1",
            settled_at="2026-01-01T00:00:00Z",
            review_horizon="2026-06-01T00:00:00Z",
            confidence=0.85,
            falsifier_count=3,
        )
        assert event.debate_id == "d1"
        assert event.confidence == 0.85
        assert event.falsifier_count == 3
        assert event.flagged_at  # auto-generated

    def test_to_dict(self):
        event = SettlementReviewEvent(
            debate_id="d1",
            settled_at="2026-01-01",
            review_horizon="2026-06-01",
            confidence=0.9,
            falsifier_count=2,
        )
        d = event.to_dict()
        assert d["debate_id"] == "d1"
        assert d["confidence"] == 0.9
        assert d["falsifier_count"] == 2
        assert "flagged_at" in d


# ---------------------------------------------------------------------------
# SchedulerStats tests
# ---------------------------------------------------------------------------


class TestSchedulerStats:
    """Tests for the SchedulerStats dataclass."""

    def test_defaults(self):
        stats = SchedulerStats()
        assert stats.total_checks == 0
        assert stats.total_flagged == 0
        assert stats.total_errors == 0
        assert stats.last_check_at is None

    def test_to_dict(self):
        stats = SchedulerStats(total_checks=5, total_flagged=2)
        d = stats.to_dict()
        assert d["total_checks"] == 5
        assert d["total_flagged"] == 2
        assert d["last_check_at"] is None

    def test_to_dict_with_datetime(self):
        now = datetime.now(timezone.utc)
        stats = SchedulerStats(last_check_at=now)
        d = stats.to_dict()
        assert d["last_check_at"] == now.isoformat()


# ---------------------------------------------------------------------------
# Scheduler lifecycle tests
# ---------------------------------------------------------------------------


class TestSchedulerLifecycle:
    """Tests for start/stop lifecycle."""

    @pytest.mark.asyncio
    async def test_start_creates_task(self, scheduler):
        assert scheduler.is_running is False
        await scheduler.start()
        assert scheduler.is_running is True
        await scheduler.stop()

    @pytest.mark.asyncio
    async def test_stop_cancels_task(self, scheduler):
        await scheduler.start()
        assert scheduler.is_running is True
        await scheduler.stop()
        assert scheduler.is_running is False

    @pytest.mark.asyncio
    async def test_double_start_is_noop(self, scheduler):
        await scheduler.start()
        task1 = scheduler._task
        await scheduler.start()  # should warn but not create new task
        assert scheduler._task is task1
        await scheduler.stop()

    @pytest.mark.asyncio
    async def test_stop_when_not_running(self, scheduler):
        # Should not raise
        await scheduler.stop()
        assert scheduler.is_running is False


# ---------------------------------------------------------------------------
# Explicit scheduling tests
# ---------------------------------------------------------------------------


class TestExplicitScheduling:
    """Tests for schedule_review() and get_scheduled()."""

    def test_schedule_review_default_time(self, scheduler):
        scheduler.schedule_review("debate-abc")
        scheduled = scheduler.get_scheduled()
        assert len(scheduled) == 1
        assert scheduled[0][0] == "debate-abc"

    def test_schedule_review_with_time(self, scheduler):
        future = datetime.now(timezone.utc) + timedelta(hours=1)
        scheduler.schedule_review("debate-xyz", review_at=future)
        scheduled = scheduler.get_scheduled()
        assert len(scheduled) == 1
        assert scheduled[0][1] == future

    def test_schedule_review_naive_datetime(self, scheduler):
        naive = datetime(2026, 6, 1, 12, 0)
        scheduler.schedule_review("debate-naive", review_at=naive)
        scheduled = scheduler.get_scheduled()
        assert scheduled[0][1].tzinfo is not None  # Should be made aware

    def test_multiple_schedules(self, scheduler):
        scheduler.schedule_review("d1")
        scheduler.schedule_review("d2")
        scheduler.schedule_review("d3")
        assert len(scheduler.get_scheduled()) == 3


# ---------------------------------------------------------------------------
# check_due_settlements tests
# ---------------------------------------------------------------------------


class TestCheckDueSettlements:
    """Tests for the core check logic."""

    @pytest.mark.asyncio
    async def test_no_tracker_no_scheduled(self):
        scheduler = SettlementReviewScheduler(tracker=None)
        events = await scheduler.check_due_settlements()
        assert events == []
        assert scheduler.stats.total_checks == 1

    @pytest.mark.asyncio
    async def test_tracker_returns_due_settlements(self, scheduler, mock_tracker):
        metadata = MagicMock()
        metadata.debate_id = "d1"
        metadata.status = "settled"
        metadata.settled_at = "2026-01-01"
        metadata.review_horizon = "2026-01-15"
        metadata.confidence = 0.9
        metadata.falsifiers = ["f1", "f2"]
        mock_tracker.get_due_settlements.return_value = [metadata]

        events = await scheduler.check_due_settlements()
        assert len(events) == 1
        assert events[0].debate_id == "d1"
        assert events[0].confidence == 0.9
        assert events[0].falsifier_count == 2

    @pytest.mark.asyncio
    async def test_skips_already_flagged(self, scheduler, mock_tracker):
        metadata = MagicMock()
        metadata.debate_id = "d1"
        metadata.status = "due_review"  # Already flagged
        mock_tracker.get_due_settlements.return_value = [metadata]

        events = await scheduler.check_due_settlements()
        assert len(events) == 0

    @pytest.mark.asyncio
    async def test_skips_empty_debate_id(self, scheduler, mock_tracker):
        metadata = MagicMock()
        metadata.debate_id = ""
        mock_tracker.get_due_settlements.return_value = [metadata]

        events = await scheduler.check_due_settlements()
        assert len(events) == 0

    @pytest.mark.asyncio
    async def test_processes_explicit_schedule(self, scheduler, mock_tracker):
        past = datetime.now(timezone.utc) - timedelta(hours=1)
        scheduler.schedule_review("scheduled-debate", review_at=past)
        mock_tracker.get_settlement.return_value = None

        events = await scheduler.check_due_settlements()
        assert len(events) == 1
        assert events[0].debate_id == "scheduled-debate"
        # Explicit schedule should be consumed
        assert len(scheduler.get_scheduled()) == 0

    @pytest.mark.asyncio
    async def test_future_schedule_kept(self, scheduler):
        future = datetime.now(timezone.utc) + timedelta(hours=24)
        scheduler.schedule_review("future-debate", review_at=future)

        events = await scheduler.check_due_settlements()
        assert len(events) == 0
        assert len(scheduler.get_scheduled()) == 1

    @pytest.mark.asyncio
    async def test_tracker_error_handled(self, scheduler, mock_tracker):
        mock_tracker.get_due_settlements.side_effect = RuntimeError("DB down")

        events = await scheduler.check_due_settlements()
        assert events == []
        assert scheduler.stats.total_checks == 1

    @pytest.mark.asyncio
    async def test_stats_updated(self, scheduler, mock_tracker):
        metadata = MagicMock()
        metadata.debate_id = "d1"
        metadata.status = "settled"
        metadata.settled_at = "2026-01-01"
        metadata.review_horizon = "2026-01-15"
        metadata.confidence = 0.8
        metadata.falsifiers = []
        mock_tracker.get_due_settlements.return_value = [metadata]

        await scheduler.check_due_settlements()
        assert scheduler.stats.total_checks == 1
        assert scheduler.stats.total_flagged == 1
        assert scheduler.stats.last_flagged_count == 1
        assert scheduler.stats.last_check_at is not None

    @pytest.mark.asyncio
    async def test_on_review_due_callback(self, mock_tracker):
        callback = MagicMock()
        scheduler = SettlementReviewScheduler(
            tracker=mock_tracker,
            on_review_due=callback,
        )
        metadata = MagicMock()
        metadata.debate_id = "d1"
        metadata.status = "settled"
        metadata.settled_at = "2026-01-01"
        metadata.review_horizon = "2026-01-15"
        metadata.confidence = 0.8
        metadata.falsifiers = []
        mock_tracker.get_due_settlements.return_value = [metadata]

        await scheduler.check_due_settlements()
        assert callback.call_count == 1
        event = callback.call_args[0][0]
        assert isinstance(event, SettlementReviewEvent)
        assert event.debate_id == "d1"


# ---------------------------------------------------------------------------
# Status and events tests
# ---------------------------------------------------------------------------


class TestStatusAndEvents:
    """Tests for get_status() and event history."""

    def test_get_status_initial(self, scheduler):
        status = scheduler.get_status()
        assert status["running"] is False
        assert status["check_interval_seconds"] == 1
        assert status["scheduled_count"] == 0
        assert status["events_emitted"] == 0

    @pytest.mark.asyncio
    async def test_events_recorded(self, scheduler, mock_tracker):
        metadata = MagicMock()
        metadata.debate_id = "d1"
        metadata.status = "settled"
        metadata.settled_at = "2026-01-01"
        metadata.review_horizon = "2026-01-15"
        metadata.confidence = 0.7
        metadata.falsifiers = ["f1"]
        mock_tracker.get_due_settlements.return_value = [metadata]

        await scheduler.check_due_settlements()
        assert len(scheduler.events) == 1
        assert scheduler.events[0].debate_id == "d1"

    @pytest.mark.asyncio
    async def test_events_bounded(self, scheduler, mock_tracker):
        scheduler._max_events = 5
        for i in range(10):
            metadata = MagicMock()
            metadata.debate_id = f"d{i}"
            metadata.status = "settled"
            metadata.settled_at = "2026-01-01"
            metadata.review_horizon = "2026-01-15"
            metadata.confidence = 0.5
            metadata.falsifiers = []
            mock_tracker.get_due_settlements.return_value = [metadata]
            await scheduler.check_due_settlements()
            mock_tracker.get_due_settlements.return_value = []

        assert len(scheduler.events) == 5


# ---------------------------------------------------------------------------
# Singleton accessor tests
# ---------------------------------------------------------------------------


class TestSingletonAccessor:
    """Tests for get_scheduler/set_scheduler/reset_scheduler."""

    def test_get_scheduler_creates_instance(self):
        s = get_scheduler()
        assert isinstance(s, SettlementReviewScheduler)

    def test_get_scheduler_returns_same_instance(self):
        s1 = get_scheduler()
        s2 = get_scheduler()
        assert s1 is s2

    def test_set_scheduler(self):
        custom = SettlementReviewScheduler()
        set_scheduler(custom)
        assert get_scheduler() is custom

    def test_set_scheduler_none(self):
        get_scheduler()  # Create
        set_scheduler(None)
        s = get_scheduler()  # Should create a new one
        assert isinstance(s, SettlementReviewScheduler)

    def test_reset_scheduler(self):
        s1 = get_scheduler()
        reset_scheduler()
        s2 = get_scheduler()
        assert s1 is not s2


# ---------------------------------------------------------------------------
# Module exports tests
# ---------------------------------------------------------------------------


class TestModuleExports:
    """Tests for __all__ exports."""

    def test_all_exports_exist(self):
        import aragora.debate.settlement_scheduler as mod

        for name in mod.__all__:
            assert hasattr(mod, name), f"Missing export: {name}"

    def test_key_classes_exported(self):
        import aragora.debate.settlement_scheduler as mod

        expected = {
            "SettlementReviewScheduler",
            "SettlementReviewEvent",
            "SchedulerStats",
            "get_scheduler",
            "set_scheduler",
            "reset_scheduler",
        }
        assert expected.issubset(set(mod.__all__))
