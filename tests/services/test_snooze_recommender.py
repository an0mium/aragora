"""Tests for the Snooze Recommender service."""

from __future__ import annotations

import pytest
from datetime import datetime, time, timedelta
from unittest.mock import AsyncMock, MagicMock

from aragora.services.snooze_recommender import (
    SnoozeRecommendation,
    SnoozeReason,
    SnoozeRecommender,
    SnoozeSuggestion,
    WorkSchedule,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_email(**kw) -> MagicMock:
    email = MagicMock()
    email.id = kw.get("id", "email_123")
    email.from_address = kw.get("from_address", "sender@example.com")
    email.sender = kw.get("sender", "sender@example.com")
    email.subject = kw.get("subject", "Test email")
    return email


def _make_priority_result(priority: int = 3) -> MagicMock:
    result = MagicMock()
    result.priority.value = priority
    return result


# ---------------------------------------------------------------------------
# Dataclass tests
# ---------------------------------------------------------------------------


class TestSnoozeDataclasses:
    def test_snooze_reason_values(self):
        assert SnoozeReason.CALENDAR_FREE.value == "calendar_free"
        assert SnoozeReason.SENDER_PATTERN.value == "sender_pattern"
        assert SnoozeReason.WORK_HOURS.value == "work_hours"
        assert SnoozeReason.PRIORITY_DECAY.value == "priority_decay"
        assert SnoozeReason.WEEKEND_SKIP.value == "weekend_skip"
        assert SnoozeReason.END_OF_DAY.value == "end_of_day"
        assert SnoozeReason.TOMORROW_MORNING.value == "tomorrow_morning"

    def test_snooze_suggestion_to_dict(self):
        suggestion = SnoozeSuggestion(
            snooze_until=datetime(2025, 7, 1, 9, 0),
            reason=SnoozeReason.TOMORROW_MORNING,
            label="Tomorrow morning",
            confidence=0.85,
            rationale="Fresh start",
            is_recommended=True,
        )
        d = suggestion.to_dict()
        assert d["reason"] == "tomorrow_morning"
        assert d["label"] == "Tomorrow morning"
        assert d["confidence"] == 0.85
        assert d["is_recommended"] is True

    def test_snooze_recommendation_to_dict(self):
        suggestion = SnoozeSuggestion(
            snooze_until=datetime(2025, 7, 1, 9, 0),
            reason=SnoozeReason.WORK_HOURS,
            label="Start of day",
        )
        rec = SnoozeRecommendation(
            email_id="email_1",
            suggestions=[suggestion],
            recommended=suggestion,
            priority_level=3,
            can_safely_snooze=True,
            warning=None,
        )
        d = rec.to_dict()
        assert d["email_id"] == "email_1"
        assert len(d["suggestions"]) == 1
        assert d["recommended"] is not None
        assert d["can_safely_snooze"] is True
        assert d["warning"] is None

    def test_snooze_recommendation_no_recommended(self):
        rec = SnoozeRecommendation(
            email_id="email_1",
            suggestions=[],
        )
        d = rec.to_dict()
        assert d["recommended"] is None

    def test_work_schedule_defaults(self):
        ws = WorkSchedule()
        assert ws.work_start == time(9, 0)
        assert ws.work_end == time(17, 0)
        assert ws.work_days == [0, 1, 2, 3, 4]  # Mon-Fri
        assert ws.prefer_morning is True


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestSnoozeRecommenderInit:
    def test_defaults(self):
        recommender = SnoozeRecommender()
        assert recommender.sender_history is None
        assert recommender.calendar_service is None
        assert isinstance(recommender.work_schedule, WorkSchedule)

    def test_custom_work_schedule(self):
        ws = WorkSchedule(work_start=time(8, 0), work_end=time(18, 0))
        recommender = SnoozeRecommender(work_schedule=ws)
        assert recommender.work_schedule.work_start == time(8, 0)


# ---------------------------------------------------------------------------
# recommend_snooze
# ---------------------------------------------------------------------------


class TestRecommendSnooze:
    @pytest.mark.asyncio
    async def test_basic_recommendation(self):
        recommender = SnoozeRecommender()
        email = _make_email()
        rec = await recommender.recommend_snooze(email)
        assert isinstance(rec, SnoozeRecommendation)
        assert rec.email_id == "email_123"
        assert len(rec.suggestions) > 0
        assert rec.can_safely_snooze is True

    @pytest.mark.asyncio
    async def test_high_priority_warning(self):
        recommender = SnoozeRecommender()
        email = _make_email()
        priority = _make_priority_result(priority=1)  # Critical
        rec = await recommender.recommend_snooze(email, priority_result=priority)
        assert rec.warning is not None
        assert "high-priority" in rec.warning.lower()

    @pytest.mark.asyncio
    async def test_max_suggestions_limit(self):
        recommender = SnoozeRecommender()
        email = _make_email()
        rec = await recommender.recommend_snooze(email, max_suggestions=2)
        assert len(rec.suggestions) <= 2

    @pytest.mark.asyncio
    async def test_has_recommended(self):
        recommender = SnoozeRecommender()
        email = _make_email()
        rec = await recommender.recommend_snooze(email)
        assert rec.recommended is not None
        assert rec.recommended.is_recommended is True

    @pytest.mark.asyncio
    async def test_suggestions_sorted_by_time(self):
        recommender = SnoozeRecommender()
        email = _make_email()
        rec = await recommender.recommend_snooze(email)
        if len(rec.suggestions) > 1:
            times = [s.snooze_until for s in rec.suggestions]
            assert times == sorted(times)


# ---------------------------------------------------------------------------
# _get_quick_suggestions
# ---------------------------------------------------------------------------


class TestQuickSuggestions:
    def test_includes_tomorrow_morning(self):
        recommender = SnoozeRecommender()
        now = datetime(2025, 7, 1, 10, 0)  # Tuesday 10 AM
        suggestions = recommender._get_quick_suggestions(now, priority=3)
        labels = [s.label for s in suggestions]
        assert "Tomorrow morning" in labels

    def test_later_today_before_4pm(self):
        recommender = SnoozeRecommender()
        now = datetime(2025, 7, 1, 10, 0)  # 10 AM
        suggestions = recommender._get_quick_suggestions(now, priority=3)
        labels = [s.label for s in suggestions]
        assert "Later today" in labels

    def test_no_later_today_after_4pm(self):
        recommender = SnoozeRecommender()
        now = datetime(2025, 7, 1, 17, 0)  # 5 PM
        suggestions = recommender._get_quick_suggestions(now, priority=3)
        labels = [s.label for s in suggestions]
        assert "Later today" not in labels

    def test_low_priority_includes_weekend(self):
        recommender = SnoozeRecommender()
        now = datetime(2025, 7, 1, 10, 0)  # Tuesday
        suggestions = recommender._get_quick_suggestions(now, priority=4)
        labels = [s.label for s in suggestions]
        assert "This weekend" in labels

    def test_next_week_for_medium_priority(self):
        recommender = SnoozeRecommender()
        now = datetime(2025, 7, 1, 10, 0)
        suggestions = recommender._get_quick_suggestions(now, priority=3)
        labels = [s.label for s in suggestions]
        assert "Next week" in labels

    def test_no_next_week_for_high_priority(self):
        recommender = SnoozeRecommender()
        now = datetime(2025, 7, 1, 10, 0)
        suggestions = recommender._get_quick_suggestions(now, priority=2)
        labels = [s.label for s in suggestions]
        assert "Next week" not in labels


# ---------------------------------------------------------------------------
# _get_sender_pattern_suggestions
# ---------------------------------------------------------------------------


class TestSenderPatternSuggestions:
    @pytest.mark.asyncio
    async def test_no_history_service(self):
        recommender = SnoozeRecommender()
        suggestions = await recommender._get_sender_pattern_suggestions(
            "test@example.com",
            datetime.now(),
        )
        assert suggestions == []

    @pytest.mark.asyncio
    async def test_quick_responder(self):
        mock_history = AsyncMock()
        mock_stats = MagicMock()
        mock_stats.avg_response_time_minutes = 60.0  # 1 hour => avg_response < 4
        mock_history.get_stats = AsyncMock(return_value=mock_stats)

        recommender = SnoozeRecommender(sender_history=mock_history)
        suggestions = await recommender._get_sender_pattern_suggestions(
            "fast@example.com",
            datetime.now(),
        )
        assert len(suggestions) == 1
        assert suggestions[0].reason == SnoozeReason.SENDER_PATTERN
        assert "1 hour" in suggestions[0].label.lower()

    @pytest.mark.asyncio
    async def test_slow_responder(self):
        mock_history = AsyncMock()
        mock_stats = MagicMock()
        mock_stats.avg_response_time_minutes = 60.0 * 72  # 72 hours => > 48
        mock_history.get_stats = AsyncMock(return_value=mock_stats)

        recommender = SnoozeRecommender(sender_history=mock_history)
        suggestions = await recommender._get_sender_pattern_suggestions(
            "slow@example.com",
            datetime.now(),
        )
        assert len(suggestions) == 1
        assert "2 days" in suggestions[0].label.lower()


# ---------------------------------------------------------------------------
# _get_priority_decay_suggestions
# ---------------------------------------------------------------------------


class TestPriorityDecaySuggestions:
    def test_critical_priority_short_window(self):
        recommender = SnoozeRecommender()
        now = datetime(2025, 7, 1, 10, 0)  # Tuesday
        suggestions = recommender._get_priority_decay_suggestions(1, now)
        assert len(suggestions) == 1
        assert suggestions[0].label == "Soon (2 hours)"

    def test_defer_priority_long_window(self):
        recommender = SnoozeRecommender()
        now = datetime(2025, 7, 1, 10, 0)
        suggestions = recommender._get_priority_decay_suggestions(5, now)
        assert len(suggestions) == 1
        assert suggestions[0].label == "Next week"


# ---------------------------------------------------------------------------
# _get_work_schedule_suggestions
# ---------------------------------------------------------------------------


class TestWorkScheduleSuggestions:
    def test_end_of_day_during_work_hours(self):
        recommender = SnoozeRecommender()
        now = datetime(2025, 7, 1, 10, 0)  # 10 AM
        suggestions = recommender._get_work_schedule_suggestions(now)
        labels = [s.label for s in suggestions]
        assert "End of day" in labels

    def test_no_end_of_day_after_hours(self):
        recommender = SnoozeRecommender()
        now = datetime(2025, 7, 1, 17, 0)  # After work_end - 1
        suggestions = recommender._get_work_schedule_suggestions(now)
        labels = [s.label for s in suggestions]
        assert "End of day" not in labels


# ---------------------------------------------------------------------------
# _next_work_morning / _adjust_to_work_hours
# ---------------------------------------------------------------------------


class TestScheduleHelpers:
    def test_next_work_morning_weekday(self):
        recommender = SnoozeRecommender()
        # Tuesday -> Wednesday
        now = datetime(2025, 7, 1, 10, 0)  # Tuesday
        next_morning = recommender._next_work_morning(now)
        assert next_morning.hour == 9
        assert next_morning.weekday() == 2  # Wednesday

    def test_next_work_morning_friday_skips_weekend(self):
        recommender = SnoozeRecommender()
        now = datetime(2025, 7, 4, 10, 0)  # Friday
        next_morning = recommender._next_work_morning(now)
        assert next_morning.weekday() == 0  # Monday

    def test_adjust_to_work_hours_during_work(self):
        recommender = SnoozeRecommender()
        dt = datetime(2025, 7, 1, 14, 0)  # 2 PM Tuesday
        adjusted = recommender._adjust_to_work_hours(dt)
        assert adjusted == dt  # No change needed

    def test_adjust_to_work_hours_before_start(self):
        recommender = SnoozeRecommender()
        dt = datetime(2025, 7, 1, 6, 0)  # 6 AM
        adjusted = recommender._adjust_to_work_hours(dt)
        assert adjusted.hour == 9

    def test_adjust_to_work_hours_weekend(self):
        recommender = SnoozeRecommender()
        dt = datetime(2025, 7, 5, 10, 0)  # Saturday
        adjusted = recommender._adjust_to_work_hours(dt)
        assert adjusted.weekday() == 0  # Monday

    def test_format_slot_label_today(self):
        recommender = SnoozeRecommender()
        now = datetime.now()
        slot = now.replace(hour=14, minute=0)
        label = recommender._format_slot_label(slot)
        assert "Today" in label or "14:00" in label

    def test_format_slot_label_tomorrow(self):
        recommender = SnoozeRecommender()
        tomorrow = datetime.now() + timedelta(days=1)
        slot = tomorrow.replace(hour=9, minute=0)
        label = recommender._format_slot_label(slot)
        assert "Tomorrow" in label


# ---------------------------------------------------------------------------
# _deduplicate_suggestions / _pick_recommended
# ---------------------------------------------------------------------------


class TestSuggestionHelpers:
    def test_deduplicate_same_time(self):
        recommender = SnoozeRecommender()
        t = datetime(2025, 7, 1, 9, 0)
        suggestions = [
            SnoozeSuggestion(snooze_until=t, reason=SnoozeReason.WORK_HOURS, label="A"),
            SnoozeSuggestion(snooze_until=t, reason=SnoozeReason.TOMORROW_MORNING, label="B"),
        ]
        result = recommender._deduplicate_suggestions(suggestions)
        assert len(result) == 1

    def test_deduplicate_different_times(self):
        recommender = SnoozeRecommender()
        suggestions = [
            SnoozeSuggestion(
                snooze_until=datetime(2025, 7, 1, 9, 0),
                reason=SnoozeReason.WORK_HOURS,
                label="A",
            ),
            SnoozeSuggestion(
                snooze_until=datetime(2025, 7, 1, 14, 0),
                reason=SnoozeReason.CALENDAR_FREE,
                label="B",
            ),
        ]
        result = recommender._deduplicate_suggestions(suggestions)
        assert len(result) == 2

    def test_pick_recommended_high_priority_soonest(self):
        recommender = SnoozeRecommender()
        suggestions = [
            SnoozeSuggestion(
                snooze_until=datetime(2025, 7, 1, 14, 0),
                reason=SnoozeReason.END_OF_DAY,
                label="Later",
                confidence=0.9,
            ),
            SnoozeSuggestion(
                snooze_until=datetime(2025, 7, 1, 10, 0),
                reason=SnoozeReason.PRIORITY_DECAY,
                label="Soon",
                confidence=0.6,
            ),
        ]
        result = recommender._pick_recommended(suggestions, priority=1)
        assert result.label == "Soon"

    def test_pick_recommended_calendar_for_medium(self):
        recommender = SnoozeRecommender()
        suggestions = [
            SnoozeSuggestion(
                snooze_until=datetime(2025, 7, 1, 10, 0),
                reason=SnoozeReason.CALENDAR_FREE,
                label="Calendar slot",
                confidence=0.9,
            ),
            SnoozeSuggestion(
                snooze_until=datetime(2025, 7, 1, 14, 0),
                reason=SnoozeReason.END_OF_DAY,
                label="End of day",
                confidence=0.7,
            ),
        ]
        result = recommender._pick_recommended(suggestions, priority=3)
        assert result.label == "Calendar slot"

    def test_pick_recommended_highest_confidence(self):
        recommender = SnoozeRecommender()
        suggestions = [
            SnoozeSuggestion(
                snooze_until=datetime(2025, 7, 1, 10, 0),
                reason=SnoozeReason.END_OF_DAY,
                label="A",
                confidence=0.6,
            ),
            SnoozeSuggestion(
                snooze_until=datetime(2025, 7, 1, 14, 0),
                reason=SnoozeReason.TOMORROW_MORNING,
                label="B",
                confidence=0.9,
            ),
        ]
        result = recommender._pick_recommended(suggestions, priority=3)
        assert result.label == "B"

    def test_pick_recommended_empty(self):
        recommender = SnoozeRecommender()
        result = recommender._pick_recommended([], priority=3)
        assert result is None
