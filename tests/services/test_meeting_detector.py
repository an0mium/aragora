"""
Tests for Meeting Detection Service.

Tests the meeting detection functionality including:
- Meeting type detection (invite, update, cancel, etc.)
- Meeting link extraction (Zoom, Meet, Teams, etc.)
- Time parsing and duration extraction
- Participant extraction
- Location detection
- Calendar conflict checking
- Batch processing
"""

import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.services.meeting_detector import (
    MeetingDetector,
    MeetingDetectionResult,
    MeetingType,
    MeetingPlatform,
    MeetingLink,
    MeetingParticipant,
    ConflictInfo,
    detect_meeting_quick,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def detector():
    """Create meeting detector without calendar connectors."""
    return MeetingDetector()


@pytest.fixture
def detector_with_google():
    """Create meeting detector with Google Calendar."""
    google_calendar = MagicMock()
    google_calendar.find_conflicts = AsyncMock(return_value=[])
    return MeetingDetector(google_calendar=google_calendar)


@pytest.fixture
def detector_with_outlook():
    """Create meeting detector with Outlook Calendar."""
    outlook_calendar = MagicMock()
    outlook_calendar.find_conflicts = AsyncMock(return_value=[])
    return MeetingDetector(outlook_calendar=outlook_calendar)


@pytest.fixture
def detector_with_calendars():
    """Create meeting detector with both calendars."""
    google = MagicMock()
    google.find_conflicts = AsyncMock(return_value=[])
    outlook = MagicMock()
    outlook.find_conflicts = AsyncMock(return_value=[])
    return MeetingDetector(google_calendar=google, outlook_calendar=outlook)


def make_email(
    subject: str = "Meeting",
    body: str = "",
    sender: str = "sender@example.com",
    email_id: str = "test-123",
):
    """Create a mock email object."""
    email = MagicMock()
    email.id = email_id
    email.subject = subject
    email.body_text = body
    email.body = body
    email.from_address = sender
    email.sender = sender
    email.body_html = ""
    return email


# =============================================================================
# Data Model Tests
# =============================================================================


class TestDataModels:
    """Test data model creation."""

    def test_meeting_type_values(self):
        """Should have all meeting type values."""
        assert MeetingType.INVITE.value == "invite"
        assert MeetingType.UPDATE.value == "update"
        assert MeetingType.RESCHEDULE.value == "reschedule"
        assert MeetingType.CANCELLATION.value == "cancellation"
        assert MeetingType.CONFIRMATION.value == "confirmation"
        assert MeetingType.REMINDER.value == "reminder"
        assert MeetingType.RESPONSE.value == "response"
        assert MeetingType.AGENDA.value == "agenda"
        assert MeetingType.FOLLOWUP.value == "followup"
        assert MeetingType.NOT_MEETING.value == "not_meeting"

    def test_meeting_platform_values(self):
        """Should have all platform values."""
        assert MeetingPlatform.ZOOM.value == "zoom"
        assert MeetingPlatform.GOOGLE_MEET.value == "google_meet"
        assert MeetingPlatform.MICROSOFT_TEAMS.value == "microsoft_teams"
        assert MeetingPlatform.WEBEX.value == "webex"
        assert MeetingPlatform.SKYPE.value == "skype"
        assert MeetingPlatform.BLUEJEANS.value == "bluejeans"
        assert MeetingPlatform.GOTOMEETING.value == "gotomeeting"

    def test_meeting_participant_creation(self):
        """Should create MeetingParticipant."""
        participant = MeetingParticipant(
            email="john@example.com",
            name="John Doe",
            role="organizer",
            response_status="accepted",
        )
        assert participant.email == "john@example.com"
        assert participant.name == "John Doe"
        assert participant.role == "organizer"
        assert participant.response_status == "accepted"

    def test_meeting_participant_to_dict(self):
        """Should convert participant to dict."""
        participant = MeetingParticipant(email="test@example.com", name="Test")
        d = participant.to_dict()
        assert d["email"] == "test@example.com"
        assert d["name"] == "Test"
        assert d["role"] == "attendee"

    def test_meeting_link_creation(self):
        """Should create MeetingLink."""
        link = MeetingLink(
            url="https://zoom.us/j/123456789",
            platform=MeetingPlatform.ZOOM,
            meeting_id="123456789",
            password="abc123",
        )
        assert link.url == "https://zoom.us/j/123456789"
        assert link.platform == MeetingPlatform.ZOOM
        assert link.meeting_id == "123456789"
        assert link.password == "abc123"

    def test_meeting_link_to_dict(self):
        """Should convert link to dict."""
        link = MeetingLink(
            url="https://meet.google.com/abc-def", platform=MeetingPlatform.GOOGLE_MEET
        )
        d = link.to_dict()
        assert d["url"] == "https://meet.google.com/abc-def"
        assert d["platform"] == "google_meet"

    def test_conflict_info_creation(self):
        """Should create ConflictInfo."""
        conflict = ConflictInfo(
            event_id="event-123",
            title="Other Meeting",
            start=datetime(2025, 1, 15, 14, 0, tzinfo=timezone.utc),
            end=datetime(2025, 1, 15, 15, 0, tzinfo=timezone.utc),
            calendar_source="google",
            severity="hard",
        )
        assert conflict.event_id == "event-123"
        assert conflict.calendar_source == "google"
        assert conflict.severity == "hard"

    def test_conflict_info_to_dict(self):
        """Should convert conflict to dict."""
        conflict = ConflictInfo(
            event_id="e1",
            title="Test",
            start=datetime(2025, 1, 15, 14, 0, tzinfo=timezone.utc),
            end=datetime(2025, 1, 15, 15, 0, tzinfo=timezone.utc),
            calendar_source="outlook",
        )
        d = conflict.to_dict()
        assert d["event_id"] == "e1"
        assert d["calendar_source"] == "outlook"
        assert "2025-01-15" in d["start"]

    def test_meeting_detection_result_creation(self):
        """Should create MeetingDetectionResult."""
        result = MeetingDetectionResult(
            email_id="email-123",
            is_meeting=True,
            meeting_type=MeetingType.INVITE,
            confidence=0.9,
            title="Team Sync",
        )
        assert result.email_id == "email-123"
        assert result.is_meeting is True
        assert result.meeting_type == MeetingType.INVITE
        assert result.confidence == 0.9
        assert result.title == "Team Sync"

    def test_meeting_detection_result_to_dict(self):
        """Should convert result to dict."""
        result = MeetingDetectionResult(
            email_id="email-456",
            is_meeting=True,
            meeting_type=MeetingType.UPDATE,
            confidence=0.85,
            start_time=datetime(2025, 1, 15, 14, 0, tzinfo=timezone.utc),
        )
        d = result.to_dict()
        assert d["email_id"] == "email-456"
        assert d["is_meeting"] is True
        assert d["meeting_type"] == "update"
        assert d["confidence"] == 0.85
        assert "2025-01-15" in d["start_time"]


# =============================================================================
# Initialization Tests
# =============================================================================


class TestInitialization:
    """Test MeetingDetector initialization."""

    def test_init_without_calendars(self):
        """Should initialize without calendar connectors."""
        detector = MeetingDetector()
        assert detector.google_calendar is None
        assert detector.outlook_calendar is None

    def test_init_with_google_calendar(self):
        """Should initialize with Google Calendar."""
        google = MagicMock()
        detector = MeetingDetector(google_calendar=google)
        assert detector.google_calendar is google
        assert detector.outlook_calendar is None

    def test_init_with_outlook_calendar(self):
        """Should initialize with Outlook Calendar."""
        outlook = MagicMock()
        detector = MeetingDetector(outlook_calendar=outlook)
        assert detector.google_calendar is None
        assert detector.outlook_calendar is outlook

    def test_init_with_both_calendars(self):
        """Should initialize with both calendars."""
        google = MagicMock()
        outlook = MagicMock()
        detector = MeetingDetector(google_calendar=google, outlook_calendar=outlook)
        assert detector.google_calendar is google
        assert detector.outlook_calendar is outlook

    def test_patterns_compiled(self):
        """Should compile patterns on init."""
        detector = MeetingDetector()
        assert len(detector._compiled_type_patterns) > 0
        # All meeting types except NOT_MEETING should have patterns
        assert MeetingType.INVITE in detector._compiled_type_patterns


# =============================================================================
# Meeting Type Detection Tests
# =============================================================================


class TestMeetingTypeDetection:
    """Test meeting type detection."""

    @pytest.mark.asyncio
    async def test_detect_invite(self, detector):
        """Should detect meeting invitation."""
        email = make_email(
            subject="Meeting Invitation: Project Review",
            body="You've been invited to a meeting to discuss the project.",
        )
        result = await detector.detect_meeting(email, check_calendar=False)

        assert result.is_meeting is True
        assert result.meeting_type == MeetingType.INVITE
        assert result.confidence > 0.8

    @pytest.mark.asyncio
    async def test_detect_update(self, detector):
        """Should detect meeting update."""
        email = make_email(
            subject="Meeting Updated: Team Sync",
            body="The meeting time has been changed.",
        )
        result = await detector.detect_meeting(email, check_calendar=False)

        assert result.is_meeting is True
        assert result.meeting_type == MeetingType.UPDATE

    @pytest.mark.asyncio
    async def test_detect_reschedule(self, detector):
        """Should detect meeting reschedule."""
        email = make_email(
            subject="Meeting Rescheduled",
            body="The meeting has been rescheduled to a new time.",
        )
        result = await detector.detect_meeting(email, check_calendar=False)

        assert result.is_meeting is True
        assert result.meeting_type == MeetingType.RESCHEDULE

    @pytest.mark.asyncio
    async def test_detect_cancellation(self, detector):
        """Should detect meeting cancellation."""
        email = make_email(
            subject="Meeting Cancelled: Weekly Review",
            body="The meeting has been cancelled.",
        )
        result = await detector.detect_meeting(email, check_calendar=False)

        assert result.is_meeting is True
        assert result.meeting_type == MeetingType.CANCELLATION

    @pytest.mark.asyncio
    async def test_detect_confirmation(self, detector):
        """Should detect meeting confirmation."""
        email = make_email(
            subject="Meeting Confirmed",
            body="Your meeting has been confirmed. You're all set!",
        )
        result = await detector.detect_meeting(email, check_calendar=False)

        assert result.is_meeting is True
        assert result.meeting_type == MeetingType.CONFIRMATION

    @pytest.mark.asyncio
    async def test_detect_reminder(self, detector):
        """Should detect meeting reminder."""
        email = make_email(
            subject="Reminder: Meeting starting soon",
            body="Your meeting is starting in 15 minutes.",
        )
        result = await detector.detect_meeting(email, check_calendar=False)

        assert result.is_meeting is True
        assert result.meeting_type == MeetingType.REMINDER

    @pytest.mark.asyncio
    async def test_detect_response(self, detector):
        """Should detect meeting response/RSVP."""
        email = make_email(
            subject="Accepted: Team Meeting",
            body="John accepted your invitation to the meeting.",
        )
        result = await detector.detect_meeting(email, check_calendar=False)

        assert result.is_meeting is True
        assert result.meeting_type == MeetingType.RESPONSE

    @pytest.mark.asyncio
    async def test_detect_agenda(self, detector):
        """Should detect meeting agenda."""
        email = make_email(
            subject="Meeting Agenda: Q1 Planning",
            body="Here's the agenda for our meeting:\n1. Review Q4\n2. Plan Q1",
        )
        result = await detector.detect_meeting(email, check_calendar=False)

        assert result.is_meeting is True
        assert result.meeting_type == MeetingType.AGENDA

    @pytest.mark.asyncio
    async def test_detect_followup(self, detector):
        """Should detect meeting follow-up."""
        email = make_email(
            subject="Follow-up from our meeting",
            body="Meeting notes from our discussion:\n- Action item 1",
        )
        result = await detector.detect_meeting(email, check_calendar=False)

        assert result.is_meeting is True
        assert result.meeting_type == MeetingType.FOLLOWUP

    @pytest.mark.asyncio
    async def test_detect_not_meeting(self, detector):
        """Should detect non-meeting email."""
        email = make_email(
            subject="Invoice #12345",
            body="Please find attached invoice for services rendered.",
        )
        result = await detector.detect_meeting(email, check_calendar=False)

        assert result.is_meeting is False
        assert result.meeting_type == MeetingType.NOT_MEETING


# =============================================================================
# Meeting Link Extraction Tests
# =============================================================================


class TestMeetingLinkExtraction:
    """Test meeting link extraction."""

    @pytest.mark.asyncio
    async def test_extract_zoom_link(self, detector):
        """Should extract Zoom meeting link."""
        email = make_email(
            subject="Meeting Invitation",
            body="Join Zoom: https://zoom.us/j/123456789?pwd=abc123",
        )
        result = await detector.detect_meeting(email, check_calendar=False)

        assert len(result.meeting_links) == 1
        assert result.meeting_links[0].platform == MeetingPlatform.ZOOM
        assert "123456789" in result.meeting_links[0].url
        assert result.meeting_links[0].meeting_id == "123456789"

    @pytest.mark.asyncio
    async def test_extract_google_meet_link(self, detector):
        """Should extract Google Meet link."""
        email = make_email(
            subject="Meeting Invitation",
            body="Join via Google Meet: https://meet.google.com/abc-def-ghi",
        )
        result = await detector.detect_meeting(email, check_calendar=False)

        assert len(result.meeting_links) == 1
        assert result.meeting_links[0].platform == MeetingPlatform.GOOGLE_MEET

    @pytest.mark.asyncio
    async def test_extract_teams_link(self, detector):
        """Should extract Microsoft Teams link."""
        email = make_email(
            subject="Meeting Invitation",
            body="Join Teams: https://teams.microsoft.com/l/meetup-join/19%3ameeting_xyz",
        )
        result = await detector.detect_meeting(email, check_calendar=False)

        assert len(result.meeting_links) == 1
        assert result.meeting_links[0].platform == MeetingPlatform.MICROSOFT_TEAMS

    @pytest.mark.asyncio
    async def test_extract_webex_link(self, detector):
        """Should extract Webex link."""
        email = make_email(
            subject="Meeting Invitation",
            body="Join Webex: https://company.webex.com/meet/john",
        )
        result = await detector.detect_meeting(email, check_calendar=False)

        assert len(result.meeting_links) == 1
        assert result.meeting_links[0].platform == MeetingPlatform.WEBEX

    @pytest.mark.asyncio
    async def test_extract_skype_link(self, detector):
        """Should extract Skype link."""
        email = make_email(
            subject="Meeting Invitation",
            body="Join Skype: https://join.skype.com/abcdefg",
        )
        result = await detector.detect_meeting(email, check_calendar=False)

        assert len(result.meeting_links) == 1
        assert result.meeting_links[0].platform == MeetingPlatform.SKYPE

    @pytest.mark.asyncio
    async def test_extract_bluejeans_link(self, detector):
        """Should extract BlueJeans link."""
        email = make_email(
            subject="Meeting Invitation",
            body="Join BlueJeans: https://bluejeans.com/123456789",
        )
        result = await detector.detect_meeting(email, check_calendar=False)

        assert len(result.meeting_links) == 1
        assert result.meeting_links[0].platform == MeetingPlatform.BLUEJEANS
        assert result.meeting_links[0].meeting_id == "123456789"

    @pytest.mark.asyncio
    async def test_extract_gotomeeting_link(self, detector):
        """Should extract GoToMeeting link."""
        email = make_email(
            subject="Meeting Invitation",
            body="Join GoToMeeting: https://gotomeeting.com/join/123456789",
        )
        result = await detector.detect_meeting(email, check_calendar=False)

        assert len(result.meeting_links) == 1
        assert result.meeting_links[0].platform == MeetingPlatform.GOTOMEETING
        assert result.meeting_links[0].meeting_id == "123456789"

    @pytest.mark.asyncio
    async def test_extract_multiple_links(self, detector):
        """Should extract multiple meeting links."""
        email = make_email(
            subject="Meeting Invitation",
            body="""
            Join via Zoom: https://zoom.us/j/111111111
            Or Google Meet: https://meet.google.com/xyz-abc
            """,
        )
        result = await detector.detect_meeting(email, check_calendar=False)

        assert len(result.meeting_links) == 2
        platforms = {link.platform for link in result.meeting_links}
        assert MeetingPlatform.ZOOM in platforms
        assert MeetingPlatform.GOOGLE_MEET in platforms

    @pytest.mark.asyncio
    async def test_primary_meeting_link(self, detector):
        """Should set primary meeting link."""
        email = make_email(
            subject="Meeting Invitation",
            body="Join: https://zoom.us/j/123456789",
        )
        result = await detector.detect_meeting(email, check_calendar=False)

        assert result.primary_meeting_link is not None
        assert result.primary_meeting_link == result.meeting_links[0]


# =============================================================================
# Time Extraction Tests
# =============================================================================


class TestTimeExtraction:
    """Test time extraction from emails."""

    @pytest.mark.asyncio
    async def test_extract_time_range(self, detector):
        """Should extract time range."""
        email = make_email(
            subject="Meeting Invitation",
            body="Meeting scheduled from 2:00 PM - 3:00 PM",
        )
        result = await detector.detect_meeting(email, check_calendar=False)

        assert result.duration_minutes == 60

    @pytest.mark.asyncio
    async def test_extract_date_with_time(self, detector):
        """Should extract full date with time."""
        email = make_email(
            subject="Meeting Invitation",
            body="Meeting on January 15, 2025 at 2:00 PM",
        )
        result = await detector.detect_meeting(email, check_calendar=False)

        assert result.start_time is not None
        assert result.start_time.month == 1
        assert result.start_time.day == 15

    @pytest.mark.asyncio
    async def test_extract_numeric_date(self, detector):
        """Should extract numeric date format."""
        email = make_email(
            subject="Meeting Invitation",
            body="Meeting on 1/15/2025 2:00 PM",
        )
        result = await detector.detect_meeting(email, check_calendar=False)

        assert result.start_time is not None
        assert result.start_time.month == 1
        assert result.start_time.day == 15

    @pytest.mark.asyncio
    async def test_extract_today_time(self, detector):
        """Should extract 'today at' time."""
        email = make_email(
            subject="Meeting Invitation",
            body="Meeting today at 3:00 PM",
        )
        result = await detector.detect_meeting(email, check_calendar=False)

        assert result.start_time is not None
        assert result.start_time.date() == datetime.now(timezone.utc).date()

    @pytest.mark.asyncio
    async def test_extract_tomorrow_time(self, detector):
        """Should extract 'tomorrow at' time."""
        email = make_email(
            subject="Meeting Invitation",
            body="Meeting tomorrow at 10:00 AM",
        )
        result = await detector.detect_meeting(email, check_calendar=False)

        assert result.start_time is not None
        expected_date = (datetime.now(timezone.utc) + timedelta(days=1)).date()
        assert result.start_time.date() == expected_date

    @pytest.mark.asyncio
    async def test_extract_duration_hours(self, detector):
        """Should extract duration in hours."""
        email = make_email(
            subject="Meeting Invitation",
            body="The meeting will be 2 hours long.",
        )
        result = await detector.detect_meeting(email, check_calendar=False)

        assert result.duration_minutes == 120

    @pytest.mark.asyncio
    async def test_extract_duration_minutes(self, detector):
        """Should extract duration in minutes."""
        email = make_email(
            subject="Meeting Invitation",
            body="Quick 30 minute sync.",
        )
        result = await detector.detect_meeting(email, check_calendar=False)

        assert result.duration_minutes == 30

    @pytest.mark.asyncio
    async def test_extract_duration_hours_and_minutes(self, detector):
        """Should extract duration with hours and minutes."""
        email = make_email(
            subject="Meeting Invitation",
            body="Meeting duration: 1 hour 30 minutes.",
        )
        result = await detector.detect_meeting(email, check_calendar=False)

        assert result.duration_minutes == 90


# =============================================================================
# Participant Extraction Tests
# =============================================================================


class TestParticipantExtraction:
    """Test participant extraction."""

    @pytest.mark.asyncio
    async def test_extract_participants_from_body(self, detector):
        """Should extract participant emails from body."""
        email = make_email(
            subject="Meeting Invitation",
            body="Attendees: john@example.com, jane@example.com",
            sender="organizer@example.com",
        )
        result = await detector.detect_meeting(email, check_calendar=False)

        participant_emails = [p.email for p in result.participants]
        assert "john@example.com" in participant_emails
        assert "jane@example.com" in participant_emails
        # Sender should not be duplicated in participants
        assert "organizer@example.com" not in participant_emails

    @pytest.mark.asyncio
    async def test_organizer_set_from_sender(self, detector):
        """Should set organizer from sender."""
        email = make_email(
            subject="Meeting Invitation",
            body="Join us for a meeting.",
            sender="boss@company.com",
        )
        result = await detector.detect_meeting(email, check_calendar=False)

        assert result.organizer is not None
        assert result.organizer.email == "boss@company.com"
        assert result.organizer.role == "organizer"


# =============================================================================
# Location Extraction Tests
# =============================================================================


class TestLocationExtraction:
    """Test location extraction."""

    @pytest.mark.asyncio
    async def test_extract_location_explicit(self, detector):
        """Should extract explicit location."""
        email = make_email(
            subject="Meeting Invitation",
            body="Location: Conference Room A",
        )
        result = await detector.detect_meeting(email, check_calendar=False)

        assert result.location == "Conference Room A"

    @pytest.mark.asyncio
    async def test_extract_location_where(self, detector):
        """Should extract location from 'where' field."""
        email = make_email(
            subject="Meeting Invitation",
            body="Where: Building 5, Room 302",
        )
        result = await detector.detect_meeting(email, check_calendar=False)

        assert result.location == "Building 5, Room 302"

    @pytest.mark.asyncio
    async def test_extract_location_room(self, detector):
        """Should extract room location."""
        email = make_email(
            subject="Meeting Invitation",
            body="Room: Board Room 3",
        )
        result = await detector.detect_meeting(email, check_calendar=False)

        assert result.location == "Board Room 3"

    @pytest.mark.asyncio
    async def test_no_url_as_location(self, detector):
        """Should not extract URL as location."""
        email = make_email(
            subject="Meeting Invitation",
            body="Location: https://zoom.us/j/123",
        )
        result = await detector.detect_meeting(email, check_calendar=False)

        # URL should not be the location
        if result.location:
            assert not result.location.startswith("http")


# =============================================================================
# Title Extraction Tests
# =============================================================================


class TestTitleExtraction:
    """Test title extraction."""

    @pytest.mark.asyncio
    async def test_title_from_subject(self, detector):
        """Should use subject as title."""
        email = make_email(
            subject="Team Sync Meeting",
            body="You've been invited to a meeting. Join us for the weekly team sync.",
        )
        result = await detector.detect_meeting(email, check_calendar=False)

        assert result.is_meeting is True
        assert result.title == "Team Sync Meeting"

    @pytest.mark.asyncio
    async def test_title_removes_re_prefix(self, detector):
        """Should remove Re: prefix from title."""
        email = make_email(
            subject="Re: Team Sync Meeting",
            body="Meeting invitation - join us for a discussion.",
        )
        result = await detector.detect_meeting(email, check_calendar=False)

        assert result.is_meeting is True
        assert result.title == "Team Sync Meeting"

    @pytest.mark.asyncio
    async def test_title_removes_fwd_prefix(self, detector):
        """Should remove Fwd: prefix from title."""
        email = make_email(
            subject="Fwd: Project Review",
            body="Forwarded meeting invitation - please join us.",
        )
        result = await detector.detect_meeting(email, check_calendar=False)

        assert result.is_meeting is True
        assert result.title == "Project Review"

    @pytest.mark.asyncio
    async def test_title_from_body(self, detector):
        """Should extract title from body if specified."""
        email = make_email(
            subject="Calendar Event",
            body="Meeting invitation\nMeeting title: Q1 Planning Session\nPlease join us.",
        )
        result = await detector.detect_meeting(email, check_calendar=False)

        assert result.is_meeting is True
        assert result.title == "Q1 Planning Session"


# =============================================================================
# Calendar Conflict Tests
# =============================================================================


class TestCalendarConflicts:
    """Test calendar conflict checking."""

    @pytest.mark.asyncio
    async def test_check_google_calendar_conflicts(self, detector_with_google):
        """Should check Google Calendar for conflicts."""
        google_event = MagicMock()
        google_event.id = "g-event-1"
        google_event.summary = "Existing Meeting"
        google_event.start = datetime(2025, 1, 15, 14, 0, tzinfo=timezone.utc)
        google_event.end = datetime(2025, 1, 15, 15, 0, tzinfo=timezone.utc)

        detector_with_google.google_calendar.find_conflicts.return_value = [google_event]

        email = make_email(
            subject="Meeting Invitation",
            body="Meeting on January 15, 2025 at 2:00 PM",
        )
        result = await detector_with_google.detect_meeting(email, check_calendar=True)

        assert result.availability_checked is True
        assert result.has_conflicts is True
        assert len(result.conflicts) == 1
        assert result.conflicts[0].calendar_source == "google"

    @pytest.mark.asyncio
    async def test_check_outlook_calendar_conflicts(self, detector_with_outlook):
        """Should check Outlook Calendar for conflicts."""
        outlook_event = MagicMock()
        outlook_event.id = "o-event-1"
        outlook_event.subject = "Team Meeting"
        outlook_event.start = datetime(2025, 1, 15, 14, 0, tzinfo=timezone.utc)
        outlook_event.end = datetime(2025, 1, 15, 15, 0, tzinfo=timezone.utc)

        detector_with_outlook.outlook_calendar.find_conflicts.return_value = [outlook_event]

        email = make_email(
            subject="Meeting Invitation",
            body="Meeting on January 15, 2025 at 2:00 PM",
        )
        result = await detector_with_outlook.detect_meeting(email, check_calendar=True)

        assert result.availability_checked is True
        assert result.has_conflicts is True
        assert len(result.conflicts) == 1
        assert result.conflicts[0].calendar_source == "outlook"

    @pytest.mark.asyncio
    async def test_no_conflicts(self, detector_with_calendars):
        """Should report no conflicts when calendar is free."""
        email = make_email(
            subject="Meeting Invitation",
            body="Meeting on January 15, 2025 at 2:00 PM",
        )
        result = await detector_with_calendars.detect_meeting(email, check_calendar=True)

        assert result.availability_checked is True
        assert result.has_conflicts is False
        assert len(result.conflicts) == 0

    @pytest.mark.asyncio
    async def test_skip_conflict_check(self, detector_with_calendars):
        """Should skip conflict check when disabled."""
        email = make_email(
            subject="Meeting Invitation",
            body="Meeting on January 15, 2025 at 2:00 PM",
        )
        result = await detector_with_calendars.detect_meeting(email, check_calendar=False)

        assert result.availability_checked is False
        detector_with_calendars.google_calendar.find_conflicts.assert_not_called()

    @pytest.mark.asyncio
    async def test_calendar_error_handling(self, detector_with_google):
        """Should handle calendar API errors gracefully."""
        detector_with_google.google_calendar.find_conflicts.side_effect = ConnectionError("API Error")

        email = make_email(
            subject="Meeting Invitation",
            body="Meeting on January 15, 2025 at 2:00 PM",
        )
        # Should not raise
        result = await detector_with_google.detect_meeting(email, check_calendar=True)

        assert result.is_meeting is True


# =============================================================================
# Snooze Suggestion Tests
# =============================================================================


class TestSnoozeSuggestion:
    """Test snooze time suggestions."""

    @pytest.mark.asyncio
    async def test_suggest_snooze_time(self, detector):
        """Should suggest snooze time before meeting."""
        future_time = datetime.now(timezone.utc) + timedelta(hours=2)
        time_str = future_time.strftime("%I:%M %p")
        date_str = future_time.strftime("%B %d, %Y")

        email = make_email(
            subject="Meeting Invitation",
            body=f"Meeting on {date_str} at {time_str}",
        )
        result = await detector.detect_meeting(email, check_calendar=False)

        if result.start_time and result.suggested_snooze_until:
            # Snooze should be 30 min before start
            expected_snooze = result.start_time - timedelta(minutes=30)
            assert abs((result.suggested_snooze_until - expected_snooze).total_seconds()) < 60


# =============================================================================
# Batch Processing Tests
# =============================================================================


class TestBatchProcessing:
    """Test batch email processing."""

    @pytest.mark.asyncio
    async def test_detect_batch(self, detector):
        """Should process multiple emails."""
        emails = [
            make_email("Meeting Invitation: Team Sync", "Join us for sync", email_id="e1"),
            make_email("Invoice #123", "Please pay invoice", email_id="e2"),
            make_email("Meeting Cancelled", "Meeting has been cancelled", email_id="e3"),
        ]

        results = await detector.detect_batch(emails, check_calendar=False)

        assert len(results) == 3
        assert results[0].is_meeting is True
        assert results[1].is_meeting is False
        assert results[2].is_meeting is True

    @pytest.mark.asyncio
    async def test_get_upcoming_meetings(self, detector):
        """Should get upcoming meetings sorted by time."""
        now = datetime.now(timezone.utc)

        # Create emails with different meeting times
        emails = [
            make_email(
                "Later Meeting",
                f"Meeting on {(now + timedelta(hours=5)).strftime('%B %d, %Y at %I:%M %p')}",
                email_id="e1",
            ),
            make_email(
                "Sooner Meeting",
                f"Meeting on {(now + timedelta(hours=2)).strftime('%B %d, %Y at %I:%M %p')}",
                email_id="e2",
            ),
            make_email("Not a meeting", "Just a regular email", email_id="e3"),
        ]

        results = await detector.get_upcoming_meetings_from_emails(emails, hours_ahead=24)

        # Only meeting emails with detected times within 24h window
        meeting_results = [r for r in results if r.start_time is not None]
        # Results should be sorted by start time
        for i in range(len(meeting_results) - 1):
            assert meeting_results[i].start_time <= meeting_results[i + 1].start_time


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestConvenienceFunction:
    """Test detect_meeting_quick function."""

    @pytest.mark.asyncio
    async def test_quick_detection(self):
        """Should detect meeting with quick function."""
        result = await detect_meeting_quick(
            subject="Meeting Invitation: Team Review",
            body="Join us on January 15 at 2:00 PM\nhttps://zoom.us/j/123456",
            sender="organizer@example.com",
        )

        assert result.is_meeting is True
        assert result.meeting_type == MeetingType.INVITE
        assert len(result.meeting_links) == 1

    @pytest.mark.asyncio
    async def test_quick_non_meeting(self):
        """Should detect non-meeting with quick function."""
        result = await detect_meeting_quick(
            subject="Your Order Confirmation",
            body="Thank you for your order #12345",
            sender="store@example.com",
        )

        assert result.is_meeting is False


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases."""

    @pytest.mark.asyncio
    async def test_empty_email(self, detector):
        """Should handle empty email."""
        email = make_email(subject="", body="")
        result = await detector.detect_meeting(email, check_calendar=False)

        assert result.is_meeting is False

    @pytest.mark.asyncio
    async def test_missing_attributes(self, detector):
        """Should handle email with missing attributes."""
        email = MagicMock()
        email.id = "test"
        # Remove common attributes
        del email.subject
        del email.body_text
        del email.from_address

        # Should not raise
        result = await detector.detect_meeting(email, check_calendar=False)
        assert result is not None

    @pytest.mark.asyncio
    async def test_malformed_link(self, detector):
        """Should handle malformed meeting links."""
        email = make_email(
            subject="Meeting Invitation",
            body="Join at zoom.us/invalid",
        )
        result = await detector.detect_meeting(email, check_calendar=False)

        # Should not crash, may or may not detect as link
        assert result is not None

    @pytest.mark.asyncio
    async def test_rationale_generation(self, detector):
        """Should generate rationale for detection."""
        email = make_email(
            subject="Meeting Invitation",
            body="You've been invited to a meeting.",
        )
        result = await detector.detect_meeting(email, check_calendar=False)

        assert result.rationale != ""
        assert "Detected" in result.rationale
        assert "pattern" in result.rationale.lower()
