"""
Tests for Google Calendar connector data classes.
"""

from datetime import datetime, timedelta, timezone

import pytest

from aragora.connectors.calendar.google_calendar import (
    CalendarEvent,
    CalendarInfo,
    FreeBusySlot,
    CALENDAR_SCOPES,
    CALENDAR_SCOPES_READONLY,
    CALENDAR_SCOPES_FULL,
)


class TestCalendarEvent:
    """Tests for CalendarEvent dataclass."""

    def test_basic_event(self):
        """Test creating a basic event."""
        event = CalendarEvent(
            id="event123",
            calendar_id="primary",
            summary="Test Meeting",
        )

        assert event.id == "event123"
        assert event.calendar_id == "primary"
        assert event.summary == "Test Meeting"
        assert event.description is None
        assert event.all_day is False
        assert event.status == "confirmed"

    def test_event_with_times(self):
        """Test event with start/end times."""
        start = datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 15, 11, 0, tzinfo=timezone.utc)

        event = CalendarEvent(
            id="event123",
            calendar_id="primary",
            summary="Meeting",
            start=start,
            end=end,
        )

        assert event.start == start
        assert event.end == end

    def test_event_to_dict(self):
        """Test converting event to dictionary."""
        start = datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 15, 11, 0, tzinfo=timezone.utc)

        event = CalendarEvent(
            id="event123",
            calendar_id="primary",
            summary="Meeting",
            description="Team sync",
            location="Conference Room A",
            start=start,
            end=end,
            organizer_email="organizer@example.com",
            attendees=[{"email": "attendee@example.com"}],
        )

        result = event.to_dict()

        assert result["id"] == "event123"
        assert result["calendar_id"] == "primary"
        assert result["summary"] == "Meeting"
        assert result["description"] == "Team sync"
        assert result["location"] == "Conference Room A"
        assert result["start"] == start.isoformat()
        assert result["end"] == end.isoformat()
        assert result["organizer_email"] == "organizer@example.com"
        assert len(result["attendees"]) == 1

    def test_all_day_event(self):
        """Test all-day event."""
        event = CalendarEvent(
            id="event123",
            calendar_id="primary",
            summary="Holiday",
            all_day=True,
        )

        assert event.all_day is True

    def test_event_with_recurrence(self):
        """Test event with recurrence rules."""
        event = CalendarEvent(
            id="event123",
            calendar_id="primary",
            summary="Weekly Meeting",
            recurrence=["RRULE:FREQ=WEEKLY;BYDAY=MO"],
        )

        assert event.recurrence is not None
        assert "WEEKLY" in event.recurrence[0]

    def test_event_with_conference(self):
        """Test event with conference data."""
        event = CalendarEvent(
            id="event123",
            calendar_id="primary",
            summary="Video Call",
            hangout_link="https://meet.google.com/abc-defg-hij",
            conference_data={"entryPoints": [{"entryPointType": "video"}]},
        )

        assert event.hangout_link is not None
        assert event.conference_data is not None

    def test_event_status_types(self):
        """Test various event statuses."""
        for status in ["confirmed", "tentative", "cancelled"]:
            event = CalendarEvent(
                id="event123",
                calendar_id="primary",
                summary="Meeting",
                status=status,
            )
            assert event.status == status


class TestFreeBusySlot:
    """Tests for FreeBusySlot dataclass."""

    def test_free_busy_slot(self):
        """Test creating a free/busy slot."""
        start = datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 15, 11, 0, tzinfo=timezone.utc)

        slot = FreeBusySlot(start=start, end=end)

        assert slot.start == start
        assert slot.end == end

    def test_to_dict(self):
        """Test converting to dictionary."""
        start = datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 15, 11, 0, tzinfo=timezone.utc)

        slot = FreeBusySlot(start=start, end=end)
        result = slot.to_dict()

        assert result["start"] == start.isoformat()
        assert result["end"] == end.isoformat()

    def test_slot_duration(self):
        """Test calculating slot duration."""
        start = datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 15, 11, 30, tzinfo=timezone.utc)

        slot = FreeBusySlot(start=start, end=end)
        duration = slot.end - slot.start

        assert duration == timedelta(hours=1, minutes=30)


class TestCalendarInfo:
    """Tests for CalendarInfo dataclass."""

    def test_basic_calendar(self):
        """Test creating a basic calendar."""
        cal = CalendarInfo(
            id="primary",
            summary="My Calendar",
        )

        assert cal.id == "primary"
        assert cal.summary == "My Calendar"
        assert cal.primary is False
        assert cal.access_role == "reader"

    def test_primary_calendar(self):
        """Test primary calendar."""
        cal = CalendarInfo(
            id="user@example.com",
            summary="Personal Calendar",
            primary=True,
            access_role="owner",
            timezone="America/New_York",
        )

        assert cal.primary is True
        assert cal.access_role == "owner"
        assert cal.timezone == "America/New_York"

    def test_to_dict(self):
        """Test converting to dictionary."""
        cal = CalendarInfo(
            id="primary",
            summary="Work Calendar",
            description="Work meetings",
            timezone="UTC",
            background_color="#0000ff",
        )

        result = cal.to_dict()

        assert result["id"] == "primary"
        assert result["summary"] == "Work Calendar"
        assert result["description"] == "Work meetings"
        assert result["timezone"] == "UTC"
        assert result["background_color"] == "#0000ff"

    def test_access_roles(self):
        """Test various access roles."""
        for role in ["reader", "writer", "owner", "freeBusyReader"]:
            cal = CalendarInfo(
                id="cal123",
                summary="Test Calendar",
                access_role=role,
            )
            assert cal.access_role == role


class TestCalendarScopes:
    """Tests for calendar API scopes."""

    def test_readonly_scopes(self):
        """Test read-only scopes are defined."""
        assert len(CALENDAR_SCOPES_READONLY) >= 2
        for scope in CALENDAR_SCOPES_READONLY:
            assert "readonly" in scope.lower() or "read" in scope.lower()

    def test_full_scopes(self):
        """Test full access scopes are defined."""
        assert len(CALENDAR_SCOPES_FULL) >= 2
        # Full scopes should not have 'readonly'
        for scope in CALENDAR_SCOPES_FULL:
            assert "readonly" not in scope.lower()

    def test_default_scopes(self):
        """Test default scopes are read-only."""
        assert CALENDAR_SCOPES == CALENDAR_SCOPES_READONLY
