"""Tests for the Follow-Up Tracker service."""

from __future__ import annotations

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

from aragora.services.followup_tracker import (
    FollowUpItem,
    FollowUpPriority,
    FollowUpStats,
    FollowUpStatus,
    FollowUpTracker,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tracker(**kw) -> FollowUpTracker:
    return FollowUpTracker(**kw)


async def _add_item(
    tracker: FollowUpTracker,
    email_id: str = "email_1",
    thread_id: str = "thread_1",
    subject: str = "Test Subject",
    recipient: str = "bob@example.com",
    **kw,
) -> FollowUpItem:
    return await tracker.mark_awaiting_reply(
        email_id=email_id,
        thread_id=thread_id,
        subject=subject,
        recipient=recipient,
        **kw,
    )


# ---------------------------------------------------------------------------
# Dataclass / enum tests
# ---------------------------------------------------------------------------


class TestFollowUpDataclasses:
    def test_status_values(self):
        assert FollowUpStatus.AWAITING.value == "awaiting"
        assert FollowUpStatus.OVERDUE.value == "overdue"
        assert FollowUpStatus.RECEIVED.value == "received"
        assert FollowUpStatus.CANCELLED.value == "cancelled"

    def test_priority_values(self):
        assert FollowUpPriority.URGENT.value == "urgent"
        assert FollowUpPriority.LOW.value == "low"

    def test_item_days_waiting(self):
        item = FollowUpItem(
            id="t",
            email_id="e",
            thread_id="th",
            subject="S",
            recipient="r@example.com",
            sent_at=datetime.now() - timedelta(days=5),
        )
        assert item.days_waiting >= 4

    def test_item_not_overdue_no_expected_by(self):
        item = FollowUpItem(
            id="t",
            email_id="e",
            thread_id="th",
            subject="S",
            recipient="r@example.com",
            sent_at=datetime.now(),
        )
        assert item.is_overdue is False
        assert item.days_overdue == 0

    def test_item_is_overdue(self):
        item = FollowUpItem(
            id="t",
            email_id="e",
            thread_id="th",
            subject="S",
            recipient="r@example.com",
            sent_at=datetime.now() - timedelta(days=10),
            expected_by=datetime.now() - timedelta(days=3),
        )
        assert item.is_overdue is True
        assert item.days_overdue >= 2

    def test_item_not_overdue_future_expected_by(self):
        item = FollowUpItem(
            id="t",
            email_id="e",
            thread_id="th",
            subject="S",
            recipient="r@example.com",
            sent_at=datetime.now(),
            expected_by=datetime.now() + timedelta(days=5),
        )
        assert item.is_overdue is False
        assert item.days_overdue == 0

    def test_item_to_dict(self):
        item = FollowUpItem(
            id="fu_1",
            email_id="e_1",
            thread_id="th_1",
            subject="Hello",
            recipient="bob@example.com",
            sent_at=datetime.now(),
            status=FollowUpStatus.AWAITING,
            priority=FollowUpPriority.HIGH,
        )
        d = item.to_dict()
        assert d["id"] == "fu_1"
        assert d["recipient"] == "bob@example.com"
        assert d["status"] == "awaiting"
        assert d["priority"] == "high"
        assert "days_waiting" in d
        assert "is_overdue" in d

    def test_stats_to_dict(self):
        stats = FollowUpStats(
            total_pending=5,
            overdue_count=2,
            urgent_count=1,
            avg_wait_days=3.456,
        )
        d = stats.to_dict()
        assert d["total_pending"] == 5
        assert d["avg_wait_days"] == 3.5  # Rounded to 1 decimal


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestFollowUpTrackerInit:
    def test_init_defaults(self):
        tracker = _make_tracker()
        assert tracker.gmail is None
        assert tracker.default_followup_days == 3
        assert tracker.auto_detect_threshold_days == 2
        assert len(tracker._followups) == 0

    def test_init_custom(self):
        tracker = _make_tracker(
            default_followup_days=5,
            auto_detect_threshold_days=1,
        )
        assert tracker.default_followup_days == 5
        assert tracker.auto_detect_threshold_days == 1


# ---------------------------------------------------------------------------
# mark_awaiting_reply
# ---------------------------------------------------------------------------


class TestMarkAwaitingReply:
    @pytest.mark.asyncio
    async def test_creates_followup(self):
        tracker = _make_tracker()
        item = await _add_item(tracker)
        assert item.id.startswith("fu_")
        assert item.email_id == "email_1"
        assert item.thread_id == "thread_1"
        assert item.recipient == "bob@example.com"
        assert item.status == FollowUpStatus.AWAITING

    @pytest.mark.asyncio
    async def test_stores_followup(self):
        tracker = _make_tracker()
        item = await _add_item(tracker)
        assert item.id in tracker._followups

    @pytest.mark.asyncio
    async def test_indexes_by_thread(self):
        tracker = _make_tracker()
        item = await _add_item(tracker, thread_id="th_123")
        assert item.id in tracker._by_thread["th_123"]

    @pytest.mark.asyncio
    async def test_indexes_by_recipient(self):
        tracker = _make_tracker()
        item = await _add_item(tracker, recipient="Alice@Example.com")
        assert item.id in tracker._by_recipient["alice@example.com"]

    @pytest.mark.asyncio
    async def test_default_expected_by(self):
        tracker = _make_tracker(default_followup_days=5)
        sent = datetime.now()
        item = await _add_item(tracker, sent_at=sent)
        expected = sent + timedelta(days=5)
        # Allow 2 second tolerance
        assert abs((item.expected_by - expected).total_seconds()) < 2

    @pytest.mark.asyncio
    async def test_custom_expected_by(self):
        tracker = _make_tracker()
        exp = datetime.now() + timedelta(days=10)
        item = await _add_item(tracker, expected_by=exp)
        assert item.expected_by == exp

    @pytest.mark.asyncio
    async def test_priority_set(self):
        tracker = _make_tracker()
        item = await _add_item(tracker, priority=FollowUpPriority.URGENT)
        assert item.priority == FollowUpPriority.URGENT


# ---------------------------------------------------------------------------
# get_pending_followups
# ---------------------------------------------------------------------------


class TestGetPendingFollowups:
    @pytest.mark.asyncio
    async def test_empty(self):
        tracker = _make_tracker()
        items = await tracker.get_pending_followups()
        assert items == []

    @pytest.mark.asyncio
    async def test_filters_resolved(self):
        tracker = _make_tracker()
        item = await _add_item(tracker, email_id="e1", thread_id="t1")
        await tracker.resolve_followup(item.id)
        items = await tracker.get_pending_followups(include_resolved=False)
        assert len(items) == 0

    @pytest.mark.asyncio
    async def test_includes_resolved(self):
        tracker = _make_tracker()
        item = await _add_item(tracker, email_id="e1", thread_id="t1")
        await tracker.resolve_followup(item.id)
        items = await tracker.get_pending_followups(include_resolved=True)
        assert len(items) == 1

    @pytest.mark.asyncio
    async def test_auto_updates_overdue(self):
        tracker = _make_tracker()
        item = await _add_item(
            tracker,
            email_id="e1",
            thread_id="t1",
            sent_at=datetime.now() - timedelta(days=10),
            expected_by=datetime.now() - timedelta(days=3),
        )
        items = await tracker.get_pending_followups()
        assert len(items) == 1
        assert items[0].status == FollowUpStatus.OVERDUE

    @pytest.mark.asyncio
    async def test_sort_by_priority(self):
        tracker = _make_tracker()
        await _add_item(tracker, email_id="e1", thread_id="t1", priority=FollowUpPriority.LOW)
        await _add_item(tracker, email_id="e2", thread_id="t2", priority=FollowUpPriority.URGENT)
        items = await tracker.get_pending_followups(sort_by="priority")
        assert items[0].priority == FollowUpPriority.URGENT


# ---------------------------------------------------------------------------
# check_for_replies
# ---------------------------------------------------------------------------


class TestCheckForReplies:
    @pytest.mark.asyncio
    async def test_no_gmail_returns_empty(self):
        tracker = _make_tracker()
        result = await tracker.check_for_replies()
        assert result == []

    @pytest.mark.asyncio
    async def test_with_gmail_detects_reply(self):
        mock_gmail = AsyncMock()

        # Create a mock thread with reply
        sent_at = datetime.now() - timedelta(hours=5)
        reply_msg = MagicMock()
        reply_msg.date = datetime.now() - timedelta(hours=1)
        reply_msg.from_address = "bob@example.com"

        sent_msg = MagicMock()
        sent_msg.date = sent_at
        sent_msg.from_address = "me@example.com"

        mock_thread = MagicMock()
        mock_thread.messages = [sent_msg, reply_msg]
        mock_gmail.get_thread = AsyncMock(return_value=mock_thread)

        tracker = _make_tracker(gmail_connector=mock_gmail)
        item = await _add_item(
            tracker,
            email_id="e1",
            thread_id="t1",
            recipient="bob@example.com",
            sent_at=sent_at,
        )

        result = await tracker.check_for_replies()
        assert len(result) == 1
        assert result[0].status == FollowUpStatus.RECEIVED


# ---------------------------------------------------------------------------
# auto_detect_sent_emails
# ---------------------------------------------------------------------------


class TestAutoDetectSentEmails:
    @pytest.mark.asyncio
    async def test_no_gmail_returns_empty(self):
        tracker = _make_tracker()
        result = await tracker.auto_detect_sent_emails()
        assert result == []


# ---------------------------------------------------------------------------
# resolve_followup
# ---------------------------------------------------------------------------


class TestResolveFollowup:
    @pytest.mark.asyncio
    async def test_resolve(self):
        tracker = _make_tracker()
        item = await _add_item(tracker)
        resolved = await tracker.resolve_followup(item.id)
        assert resolved is not None
        assert resolved.status == FollowUpStatus.RESOLVED

    @pytest.mark.asyncio
    async def test_resolve_with_notes(self):
        tracker = _make_tracker()
        item = await _add_item(tracker)
        resolved = await tracker.resolve_followup(item.id, notes="Done manually")
        assert resolved.notes == "Done manually"

    @pytest.mark.asyncio
    async def test_resolve_cancelled(self):
        tracker = _make_tracker()
        item = await _add_item(tracker)
        resolved = await tracker.resolve_followup(item.id, status=FollowUpStatus.CANCELLED)
        assert resolved.status == FollowUpStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_resolve_not_found(self):
        tracker = _make_tracker()
        result = await tracker.resolve_followup("nonexistent")
        assert result is None


# ---------------------------------------------------------------------------
# update_priority / update_expected_date / record_reminder_sent
# ---------------------------------------------------------------------------


class TestUpdateFollowup:
    @pytest.mark.asyncio
    async def test_update_priority(self):
        tracker = _make_tracker()
        item = await _add_item(tracker)
        result = await tracker.update_priority(item.id, FollowUpPriority.URGENT)
        assert result.priority == FollowUpPriority.URGENT

    @pytest.mark.asyncio
    async def test_update_priority_not_found(self):
        tracker = _make_tracker()
        result = await tracker.update_priority("nonexistent", FollowUpPriority.HIGH)
        assert result is None

    @pytest.mark.asyncio
    async def test_update_expected_date(self):
        tracker = _make_tracker()
        item = await _add_item(tracker)
        new_date = datetime.now() + timedelta(days=10)
        result = await tracker.update_expected_date(item.id, new_date)
        assert result.expected_by == new_date

    @pytest.mark.asyncio
    async def test_update_expected_date_clears_overdue(self):
        tracker = _make_tracker()
        item = await _add_item(
            tracker,
            sent_at=datetime.now() - timedelta(days=10),
            expected_by=datetime.now() - timedelta(days=3),
        )
        # Force overdue status
        item.status = FollowUpStatus.OVERDUE
        new_date = datetime.now() + timedelta(days=5)
        result = await tracker.update_expected_date(item.id, new_date)
        assert result.status == FollowUpStatus.AWAITING

    @pytest.mark.asyncio
    async def test_record_reminder_sent(self):
        tracker = _make_tracker()
        item = await _add_item(tracker)
        result = await tracker.record_reminder_sent(item.id)
        assert result.reminder_count == 1
        assert result.last_reminder is not None

    @pytest.mark.asyncio
    async def test_record_reminder_increments(self):
        tracker = _make_tracker()
        item = await _add_item(tracker)
        await tracker.record_reminder_sent(item.id)
        result = await tracker.record_reminder_sent(item.id)
        assert result.reminder_count == 2

    @pytest.mark.asyncio
    async def test_record_reminder_not_found(self):
        tracker = _make_tracker()
        result = await tracker.record_reminder_sent("nonexistent")
        assert result is None


# ---------------------------------------------------------------------------
# get_stats
# ---------------------------------------------------------------------------


class TestFollowUpStats:
    def test_empty_stats(self):
        tracker = _make_tracker()
        stats = tracker.get_stats()
        assert stats.total_pending == 0
        assert stats.overdue_count == 0
        assert stats.avg_wait_days == 0.0

    @pytest.mark.asyncio
    async def test_stats_with_data(self):
        tracker = _make_tracker()
        await _add_item(tracker, email_id="e1", thread_id="t1", priority=FollowUpPriority.URGENT)
        await _add_item(tracker, email_id="e2", thread_id="t2", recipient="bob@example.com")
        stats = tracker.get_stats()
        assert stats.total_pending == 2
        assert stats.urgent_count == 1


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------


class TestQueryHelpers:
    @pytest.mark.asyncio
    async def test_get_followups_by_recipient(self):
        tracker = _make_tracker()
        await _add_item(tracker, email_id="e1", thread_id="t1", recipient="bob@example.com")
        await _add_item(tracker, email_id="e2", thread_id="t2", recipient="alice@example.com")
        result = await tracker.get_followups_by_recipient("bob@example.com")
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_get_followups_by_recipient_case_insensitive(self):
        tracker = _make_tracker()
        await _add_item(tracker, email_id="e1", thread_id="t1", recipient="Bob@Example.com")
        result = await tracker.get_followups_by_recipient("bob@example.com")
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_get_overdue_followups(self):
        tracker = _make_tracker()
        await _add_item(
            tracker,
            email_id="e1",
            thread_id="t1",
            sent_at=datetime.now() - timedelta(days=10),
            expected_by=datetime.now() - timedelta(days=3),
        )
        await _add_item(tracker, email_id="e2", thread_id="t2")
        overdue = await tracker.get_overdue_followups()
        assert len(overdue) == 1

    @pytest.mark.asyncio
    async def test_get_followups_due_soon(self):
        tracker = _make_tracker()
        await _add_item(
            tracker,
            email_id="e1",
            thread_id="t1",
            expected_by=datetime.now() + timedelta(hours=12),
        )
        await _add_item(
            tracker,
            email_id="e2",
            thread_id="t2",
            expected_by=datetime.now() + timedelta(days=5),
        )
        due_soon = await tracker.get_followups_due_soon(days=1)
        assert len(due_soon) == 1
