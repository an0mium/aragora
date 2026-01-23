"""
Tests for Outlook Calendar connector data classes.
"""

from datetime import datetime, timedelta, timezone

import pytest

from aragora.connectors.calendar.outlook_calendar import (
    OutlookCalendarEvent,
    OutlookCalendarInfo,
    OutlookFreeBusySlot,
    CALENDAR_SCOPES,
    CALENDAR_SCOPES_READONLY,
    CALENDAR_SCOPES_FULL,
)


class TestOutlookCalendarEvent:
    """Tests for OutlookCalendarEvent dataclass."""

    def test_basic_event(self):
        """Test creating a basic event."""
        event = OutlookCalendarEvent(
            id="event123",
            calendar_id="primary",
            subject="Test Meeting",
        )

        assert event.id == "event123"
        assert event.calendar_id == "primary"
        assert event.subject == "Test Meeting"
        assert event.body_preview is None
        assert event.all_day is False
        assert event.show_as == "busy"

    def test_event_with_times(self):
        """Test event with start/end times."""
        start = datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 15, 11, 0, tzinfo=timezone.utc)

        event = OutlookCalendarEvent(
            id="event123",
            calendar_id="primary",
            subject="Meeting",
            start=start,
            end=end,
        )

        assert event.start == start
        assert event.end == end

    def test_event_to_dict(self):
        """Test converting event to dictionary."""
        start = datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 15, 11, 0, tzinfo=timezone.utc)

        event = OutlookCalendarEvent(
            id="event123",
            calendar_id="primary",
            subject="Meeting",
            body_preview="Team sync discussion",
            location="Conference Room A",
            start=start,
            end=end,
            organizer_email="organizer@example.com",
            attendees=[{"email": "attendee@example.com"}],
            is_online_meeting=True,
            online_meeting_url="https://teams.microsoft.com/...",
        )

        result = event.to_dict()

        assert result["id"] == "event123"
        assert result["calendar_id"] == "primary"
        assert result["subject"] == "Meeting"
        assert result["body_preview"] == "Team sync discussion"
        assert result["location"] == "Conference Room A"
        assert result["start"] == start.isoformat()
        assert result["end"] == end.isoformat()
        assert result["is_online_meeting"] is True
        assert result["online_meeting_url"] == "https://teams.microsoft.com/..."

    def test_all_day_event(self):
        """Test all-day event."""
        event = OutlookCalendarEvent(
            id="event123",
            calendar_id="primary",
            subject="Holiday",
            all_day=True,
        )

        assert event.all_day is True

    def test_event_with_recurrence(self):
        """Test event with recurrence rules."""
        event = OutlookCalendarEvent(
            id="event123",
            calendar_id="primary",
            subject="Weekly Meeting",
            recurrence={
                "pattern": {"type": "weekly", "daysOfWeek": ["monday"]},
                "range": {"type": "noEnd"},
            },
        )

        assert event.recurrence is not None
        assert event.recurrence["pattern"]["type"] == "weekly"

    def test_event_with_teams_meeting(self):
        """Test event with Teams meeting info."""
        event = OutlookCalendarEvent(
            id="event123",
            calendar_id="primary",
            subject="Teams Call",
            is_online_meeting=True,
            online_meeting_url="https://teams.microsoft.com/l/meetup-join/...",
            online_meeting_provider="teamsForBusiness",
        )

        assert event.is_online_meeting is True
        assert "teams.microsoft.com" in event.online_meeting_url
        assert event.online_meeting_provider == "teamsForBusiness"

    def test_event_sensitivity_levels(self):
        """Test various event sensitivity levels."""
        for sensitivity in ["normal", "personal", "private", "confidential"]:
            event = OutlookCalendarEvent(
                id="event123",
                calendar_id="primary",
                subject="Meeting",
                sensitivity=sensitivity,
            )
            assert event.sensitivity == sensitivity

    def test_event_importance_levels(self):
        """Test various event importance levels."""
        for importance in ["low", "normal", "high"]:
            event = OutlookCalendarEvent(
                id="event123",
                calendar_id="primary",
                subject="Meeting",
                importance=importance,
            )
            assert event.importance == importance

    def test_show_as_values(self):
        """Test various show_as values."""
        for show_as in ["free", "tentative", "busy", "oof", "workingElsewhere"]:
            event = OutlookCalendarEvent(
                id="event123",
                calendar_id="primary",
                subject="Meeting",
                show_as=show_as,
            )
            assert event.show_as == show_as

    def test_event_with_categories(self):
        """Test event with categories."""
        event = OutlookCalendarEvent(
            id="event123",
            calendar_id="primary",
            subject="Categorized Meeting",
            categories=["Work", "Important"],
        )

        assert "Work" in event.categories
        assert "Important" in event.categories

    def test_cancelled_event(self):
        """Test cancelled event."""
        event = OutlookCalendarEvent(
            id="event123",
            calendar_id="primary",
            subject="Cancelled Meeting",
            is_cancelled=True,
        )

        assert event.is_cancelled is True


class TestOutlookFreeBusySlot:
    """Tests for OutlookFreeBusySlot dataclass."""

    def test_free_busy_slot(self):
        """Test creating a free/busy slot."""
        start = datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 15, 11, 0, tzinfo=timezone.utc)

        slot = OutlookFreeBusySlot(start=start, end=end)

        assert slot.start == start
        assert slot.end == end
        assert slot.status == "busy"

    def test_to_dict(self):
        """Test converting to dictionary."""
        start = datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 15, 11, 0, tzinfo=timezone.utc)

        slot = OutlookFreeBusySlot(start=start, end=end, status="tentative")
        result = slot.to_dict()

        assert result["start"] == start.isoformat()
        assert result["end"] == end.isoformat()
        assert result["status"] == "tentative"

    def test_slot_duration(self):
        """Test calculating slot duration."""
        start = datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 15, 11, 30, tzinfo=timezone.utc)

        slot = OutlookFreeBusySlot(start=start, end=end)
        duration = slot.end - slot.start

        assert duration == timedelta(hours=1, minutes=30)

    def test_slot_status_values(self):
        """Test various slot status values."""
        start = datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 15, 11, 0, tzinfo=timezone.utc)

        for status in ["free", "tentative", "busy", "oof", "workingElsewhere"]:
            slot = OutlookFreeBusySlot(start=start, end=end, status=status)
            assert slot.status == status


class TestOutlookCalendarInfo:
    """Tests for OutlookCalendarInfo dataclass."""

    def test_basic_calendar(self):
        """Test creating a basic calendar."""
        cal = OutlookCalendarInfo(
            id="AAMk...",
            name="My Calendar",
        )

        assert cal.id == "AAMk..."
        assert cal.name == "My Calendar"
        assert cal.is_default is False
        assert cal.can_edit is True

    def test_default_calendar(self):
        """Test default calendar."""
        cal = OutlookCalendarInfo(
            id="AAMk...",
            name="Calendar",
            is_default=True,
            owner_email="user@example.com",
        )

        assert cal.is_default is True
        assert cal.owner_email == "user@example.com"

    def test_to_dict(self):
        """Test converting to dictionary."""
        cal = OutlookCalendarInfo(
            id="AAMk...",
            name="Work Calendar",
            color="#0000ff",
            is_default=True,
            can_edit=True,
            can_share=True,
            owner_email="user@example.com",
            owner_name="Test User",
        )

        result = cal.to_dict()

        assert result["id"] == "AAMk..."
        assert result["name"] == "Work Calendar"
        assert result["color"] == "#0000ff"
        assert result["is_default"] is True
        assert result["can_edit"] is True
        assert result["owner_email"] == "user@example.com"

    def test_shared_calendar(self):
        """Test shared calendar with limited permissions."""
        cal = OutlookCalendarInfo(
            id="AAMk...shared",
            name="Shared Calendar",
            is_default=False,
            can_edit=False,
            can_share=False,
            can_view_private_items=False,
            owner_email="other@example.com",
            owner_name="Other User",
        )

        assert cal.can_edit is False
        assert cal.can_share is False
        assert cal.can_view_private_items is False


class TestCalendarScopes:
    """Tests for calendar API scopes."""

    def test_readonly_scopes(self):
        """Test read-only scopes are defined."""
        assert len(CALENDAR_SCOPES_READONLY) >= 2
        # Should contain Calendars.Read
        assert any("Calendars.Read" in scope for scope in CALENDAR_SCOPES_READONLY)
        # Should contain User.Read
        assert any("User.Read" in scope for scope in CALENDAR_SCOPES_READONLY)

    def test_full_scopes(self):
        """Test full access scopes are defined."""
        assert len(CALENDAR_SCOPES_FULL) >= 2
        # Full scopes should have ReadWrite
        assert any("Calendars.ReadWrite" in scope for scope in CALENDAR_SCOPES_FULL)

    def test_default_scopes(self):
        """Test default scopes are read-only."""
        assert CALENDAR_SCOPES == CALENDAR_SCOPES_READONLY

    def test_scopes_are_microsoft_graph_urls(self):
        """Test scopes are Microsoft Graph format."""
        for scope in CALENDAR_SCOPES:
            assert "graph.microsoft.com" in scope or scope.startswith("Calendars.") or scope.startswith("User.")
