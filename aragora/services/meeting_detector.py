"""
Meeting Detection Service.

Intelligent meeting detection from email content with calendar integration:
- Detect meeting invites, confirmations, reschedules, cancellations
- Extract meeting time, participants, location, meeting links
- Check calendar for conflicts
- Auto-snooze emails until meeting time
- Parse natural language time expressions

Features:
- Pattern-based detection for common meeting email formats
- Natural language datetime extraction
- Integration with Google Calendar and Outlook Calendar connectors
- Meeting link detection (Zoom, Meet, Teams, etc.)
- Conflict detection with existing calendar events

Usage:
    from aragora.services.meeting_detector import MeetingDetector

    detector = MeetingDetector(
        google_calendar=google_connector,
        outlook_calendar=outlook_connector,
    )

    # Detect meeting from email
    result = await detector.detect_meeting(email)
    if result.is_meeting:
        print(f"Meeting: {result.title}")
        print(f"Time: {result.start_time} - {result.end_time}")
        print(f"Conflicts: {len(result.conflicts)}")
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from aragora.connectors.calendar.google_calendar import (
        GoogleCalendarConnector,
    )
    from aragora.connectors.calendar.outlook_calendar import (
        OutlookCalendarConnector,
    )
    from aragora.connectors.enterprise.communication.models import EmailMessage

logger = logging.getLogger(__name__)


class MeetingType(Enum):
    """Types of meeting-related emails."""

    INVITE = "invite"  # New meeting invitation
    UPDATE = "update"  # Meeting update/change
    RESCHEDULE = "reschedule"  # Meeting rescheduled
    CANCELLATION = "cancellation"  # Meeting cancelled
    CONFIRMATION = "confirmation"  # Meeting confirmed
    REMINDER = "reminder"  # Meeting reminder
    RESPONSE = "response"  # RSVP response
    AGENDA = "agenda"  # Meeting agenda
    FOLLOWUP = "followup"  # Post-meeting follow-up
    NOT_MEETING = "not_meeting"  # Not meeting-related


class MeetingPlatform(Enum):
    """Video conferencing platforms."""

    ZOOM = "zoom"
    GOOGLE_MEET = "google_meet"
    MICROSOFT_TEAMS = "microsoft_teams"
    WEBEX = "webex"
    SKYPE = "skype"
    BLUEJEANS = "bluejeans"
    GOTOMEETING = "gotomeeting"
    SLACK_HUDDLE = "slack_huddle"
    UNKNOWN = "unknown"


@dataclass
class MeetingParticipant:
    """Participant in a detected meeting."""

    email: str
    name: Optional[str] = None
    role: str = "attendee"  # organizer, required, optional
    response_status: Optional[str] = None  # accepted, declined, tentative, pending

    def to_dict(self) -> Dict[str, Any]:
        return {
            "email": self.email,
            "name": self.name,
            "role": self.role,
            "response_status": self.response_status,
        }


@dataclass
class MeetingLink:
    """Video conferencing link."""

    url: str
    platform: MeetingPlatform
    meeting_id: Optional[str] = None
    password: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "url": self.url,
            "platform": self.platform.value,
            "meeting_id": self.meeting_id,
            "password": self.password,
        }


@dataclass
class ConflictInfo:
    """Information about a calendar conflict."""

    event_id: str
    title: str
    start: datetime
    end: datetime
    calendar_source: str  # "google" or "outlook"
    severity: str = "hard"  # hard (overlap), soft (adjacent)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "title": self.title,
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "calendar_source": self.calendar_source,
            "severity": self.severity,
        }


@dataclass
class MeetingDetectionResult:
    """Result of meeting detection from an email."""

    email_id: str
    is_meeting: bool
    meeting_type: MeetingType
    confidence: float

    # Meeting details (if detected)
    title: Optional[str] = None
    description: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_minutes: Optional[int] = None
    timezone_str: Optional[str] = None
    location: Optional[str] = None
    is_all_day: bool = False

    # Participants
    organizer: Optional[MeetingParticipant] = None
    participants: List[MeetingParticipant] = field(default_factory=list)

    # Video conferencing
    meeting_links: List[MeetingLink] = field(default_factory=list)
    primary_meeting_link: Optional[MeetingLink] = None

    # Calendar integration
    conflicts: List[ConflictInfo] = field(default_factory=list)
    has_conflicts: bool = False
    availability_checked: bool = False

    # Snooze suggestion
    suggested_snooze_until: Optional[datetime] = None

    # Detection metadata
    matched_patterns: List[str] = field(default_factory=list)
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "email_id": self.email_id,
            "is_meeting": self.is_meeting,
            "meeting_type": self.meeting_type.value,
            "confidence": self.confidence,
            "title": self.title,
            "description": self.description,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_minutes": self.duration_minutes,
            "timezone": self.timezone_str,
            "location": self.location,
            "is_all_day": self.is_all_day,
            "organizer": self.organizer.to_dict() if self.organizer else None,
            "participants": [p.to_dict() for p in self.participants],
            "meeting_links": [ml.to_dict() for ml in self.meeting_links],
            "primary_meeting_link": (
                self.primary_meeting_link.to_dict() if self.primary_meeting_link else None
            ),
            "conflicts": [c.to_dict() for c in self.conflicts],
            "has_conflicts": self.has_conflicts,
            "availability_checked": self.availability_checked,
            "suggested_snooze_until": (
                self.suggested_snooze_until.isoformat() if self.suggested_snooze_until else None
            ),
            "matched_patterns": self.matched_patterns,
            "rationale": self.rationale,
        }


# Meeting detection patterns
MEETING_TYPE_PATTERNS = {
    MeetingType.INVITE: [
        (r"\binvit(e|ation|ed)\s+(to|for)\s+(a\s+)?meet", 0.9),
        (r"\bmeeting\s+invit(e|ation)", 0.9),
        (r"\bcalendar\s+invit(e|ation)", 0.85),
        (r"\byou('ve| have)\s+been\s+invited", 0.85),
        (r"\bplease\s+join\s+(us|me)\s+(for|at)", 0.7),
        (r"\bschedule[ds]?\s+a\s+meet", 0.8),
        (r"\bmeeting\s+request", 0.85),
    ],
    MeetingType.UPDATE: [
        (r"\bmeeting\s+(update|updated|changed)", 0.9),
        (r"\bevent\s+update", 0.8),
        (r"\btime\s+(changed|updated)", 0.7),
        (r"\blocation\s+(changed|updated)", 0.7),
    ],
    MeetingType.RESCHEDULE: [
        (r"\breschedule[d]?", 0.9),
        (r"\bmoved\s+to\s+(a\s+)?new\s+(time|date)", 0.85),
        (r"\bpostponed", 0.8),
        (r"\bpush(ed)?\s+back", 0.7),
    ],
    MeetingType.CANCELLATION: [
        (r"\bcancel(led|ed)", 0.9),
        (r"\bmeeting\s+cancel", 0.9),
        (r"\bno\s+longer\s+(taking|happening|occurring)", 0.8),
        (r"\bdeclined", 0.6),
    ],
    MeetingType.CONFIRMATION: [
        (r"\bmeeting\s+confirm(ed|ation)", 0.9),
        (r"\bconfirm(ed|ing)\s+(your\s+)?attendance", 0.85),
        (r"\byou('re|r| are)\s+all\s+set", 0.6),
    ],
    MeetingType.REMINDER: [
        (r"\breminder:?\s+(your\s+)?meet", 0.9),
        (r"\bstarting\s+(soon|in\s+\d+)", 0.8),
        (r"\bmeeting\s+reminder", 0.9),
        (r"\bdon('t|t)\s+forget.*(meeting|call)", 0.7),
    ],
    MeetingType.RESPONSE: [
        (r"\b(accepted|declined|tentative)\s+(your\s+)?invit", 0.9),
        (r"\bwill\s+(be\s+)?(attending|joining)", 0.7),
        (r"\bcan('t|not)\s+(make\s+it|attend)", 0.7),
        (r"\brsvp", 0.6),
    ],
    MeetingType.AGENDA: [
        (r"\bmeeting\s+agenda", 0.9),
        (r"\bagenda\s+for\s+(our|the|today)", 0.85),
        (r"\bhere('s| is)\s+(the\s+)?agenda", 0.85),
        (r"\btopics\s+to\s+(discuss|cover)", 0.6),
    ],
    MeetingType.FOLLOWUP: [
        (r"\bfollow(-| )?up\s+(from|on|to)\s+(our|the)\s+meet", 0.9),
        (r"\bmeeting\s+(notes|recap|summary)", 0.85),
        (r"\baction\s+items\s+from", 0.8),
        (r"\bper\s+our\s+(discussion|meeting)", 0.7),
    ],
}

# Meeting link patterns
MEETING_LINK_PATTERNS = {
    MeetingPlatform.ZOOM: [
        r"(https?://)?(\w+\.)?zoom\.us/j/\d+(\?pwd=\w+)?",
        r"(https?://)?(\w+\.)?zoom\.us/my/\w+",
    ],
    MeetingPlatform.GOOGLE_MEET: [
        r"(https?://)?meet\.google\.com/[\w-]+",
    ],
    MeetingPlatform.MICROSOFT_TEAMS: [
        r"(https?://)?teams\.microsoft\.com/l/meetup-join/[^\s]+",
        r"(https?://)?teams\.live\.com/meet/[^\s]+",
    ],
    MeetingPlatform.WEBEX: [
        r"(https?://)?(\w+\.)?webex\.com/\w+/j\.php\?MTID=\w+",
        r"(https?://)?(\w+\.)?webex\.com/meet/\w+",
    ],
    MeetingPlatform.SKYPE: [
        r"(https?://)?join\.skype\.com/\w+",
    ],
    MeetingPlatform.BLUEJEANS: [
        r"(https?://)?(\w+\.)?bluejeans\.com/\d+",
    ],
    MeetingPlatform.GOTOMEETING: [
        r"(https?://)?gotomeeting\.com/join/\d+",
        r"(https?://)?global\.gotomeeting\.com/join/\d+",
    ],
}

# Time patterns for natural language parsing
TIME_PATTERNS = [
    # Explicit datetime patterns
    (
        r"(?:on\s+)?(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})(?:\s+at\s+)?(\d{1,2}:\d{2}\s*(?:am|pm)?)?",
        "date_time",
    ),
    (
        r"(\w+day),?\s+(\w+\s+\d{1,2}),?\s+(\d{4})(?:\s+at\s+)?(\d{1,2}:\d{2}\s*(?:am|pm)?)?",
        "full_date",
    ),
    (r"(\d{1,2}:\d{2}\s*(?:am|pm)?)\s*-\s*(\d{1,2}:\d{2}\s*(?:am|pm)?)", "time_range"),
    # Relative patterns
    (r"\b(today|tomorrow|next\s+\w+day)\s+at\s+(\d{1,2}:\d{2}\s*(?:am|pm)?)", "relative"),
    (r"\bin\s+(\d+)\s+(hour|minute|day)s?", "relative_delta"),
    # Common phrases
    (r"\b(this|next)\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)", "weekday"),
    (r"\b(morning|afternoon|evening)\s+(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)?", "time_of_day"),
]

# Duration patterns
DURATION_PATTERNS = [
    (r"(\d+)\s*(?:hour|hr)s?(?:\s+(\d+)\s*(?:min|minute)s?)?", "hours_minutes"),
    (r"(\d+)\s*(?:min|minute)s?", "minutes"),
    (r"(\d+(?:\.\d+)?)\s*(?:hour|hr)s?", "hours_decimal"),
]


class MeetingDetector:
    """
    Intelligent meeting detection from emails with calendar integration.

    Detects meeting-related emails and extracts structured meeting information
    including time, participants, location, and video conferencing links.
    """

    def __init__(
        self,
        google_calendar: Optional[GoogleCalendarConnector] = None,
        outlook_calendar: Optional[OutlookCalendarConnector] = None,
    ):
        """
        Initialize meeting detector.

        Args:
            google_calendar: Optional Google Calendar connector for conflict checking
            outlook_calendar: Optional Outlook Calendar connector for conflict checking
        """
        self.google_calendar = google_calendar
        self.outlook_calendar = outlook_calendar

        # Compile patterns
        self._compiled_type_patterns: Dict[MeetingType, List[tuple]] = {}
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Pre-compile regex patterns for performance."""
        for meeting_type, patterns in MEETING_TYPE_PATTERNS.items():
            self._compiled_type_patterns[meeting_type] = [
                (re.compile(p[0], re.IGNORECASE), p[1]) for p in patterns
            ]

    async def detect_meeting(
        self,
        email: EmailMessage,
        check_calendar: bool = True,
    ) -> MeetingDetectionResult:
        """
        Detect meeting information from an email.

        Args:
            email: Email message to analyze
            check_calendar: Whether to check calendar for conflicts

        Returns:
            MeetingDetectionResult with detection details
        """
        email_id = getattr(email, "id", str(hash(str(email))))
        subject = getattr(email, "subject", "")
        body = getattr(email, "body_text", getattr(email, "body", getattr(email, "snippet", "")))
        sender = getattr(
            email, "from_address", getattr(email, "sender", getattr(email, "from_", ""))
        )
        html_body = getattr(email, "body_html", "")

        # Combine text for analysis
        full_text = f"{subject}\n{body}"

        # Detect meeting type
        meeting_type, type_confidence, matched_patterns = self._detect_meeting_type(full_text)

        if meeting_type == MeetingType.NOT_MEETING:
            return MeetingDetectionResult(
                email_id=email_id,
                is_meeting=False,
                meeting_type=MeetingType.NOT_MEETING,
                confidence=1.0 - type_confidence,
                rationale="No meeting patterns detected",
            )

        # Extract meeting details
        title = self._extract_title(subject, body)
        meeting_links = self._extract_meeting_links(full_text + " " + html_body)
        start_time, end_time, duration = self._extract_time(full_text)
        participants = self._extract_participants(body, sender)
        location = self._extract_location(full_text)

        # Build result
        result = MeetingDetectionResult(
            email_id=email_id,
            is_meeting=True,
            meeting_type=meeting_type,
            confidence=type_confidence,
            title=title or subject,
            start_time=start_time,
            end_time=end_time,
            duration_minutes=duration,
            location=location,
            meeting_links=meeting_links,
            primary_meeting_link=meeting_links[0] if meeting_links else None,
            matched_patterns=matched_patterns,
            rationale=self._generate_rationale(meeting_type, matched_patterns),
        )

        # Extract organizer
        if sender:
            result.organizer = MeetingParticipant(
                email=sender,
                role="organizer",
            )

        # Add participants
        result.participants = participants

        # Check calendar for conflicts
        if check_calendar and start_time and end_time:
            conflicts = await self._check_conflicts(start_time, end_time)
            result.conflicts = conflicts
            result.has_conflicts = len(conflicts) > 0
            result.availability_checked = True

        # Calculate suggested snooze time (30 min before meeting)
        if start_time:
            snooze_time = start_time - timedelta(minutes=30)
            if snooze_time > datetime.now(timezone.utc):
                result.suggested_snooze_until = snooze_time

        return result

    def _detect_meeting_type(
        self,
        text: str,
    ) -> tuple[MeetingType, float, List[str]]:
        """Detect the type of meeting email and confidence."""
        scores: Dict[MeetingType, float] = {}
        matches: Dict[MeetingType, List[str]] = {}

        for meeting_type, patterns in self._compiled_type_patterns.items():
            type_score = 0.0
            type_matches = []

            for regex, weight in patterns:
                if regex.search(text):
                    type_score = max(type_score, weight)
                    type_matches.append(regex.pattern)

            if type_score > 0:
                scores[meeting_type] = type_score
                matches[meeting_type] = type_matches

        if not scores:
            return MeetingType.NOT_MEETING, 0.0, []

        # Get best match
        best_type = max(scores.items(), key=lambda x: x[1])
        return best_type[0], best_type[1], matches.get(best_type[0], [])

    def _extract_meeting_links(self, text: str) -> List[MeetingLink]:
        """Extract video conferencing links from text."""
        links = []
        seen_urls = set()

        for platform, patterns in MEETING_LINK_PATTERNS.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    url = match.group(0)
                    if not url.startswith("http"):
                        url = "https://" + url

                    if url not in seen_urls:
                        seen_urls.add(url)
                        meeting_id = self._extract_meeting_id(url, platform)
                        links.append(
                            MeetingLink(
                                url=url,
                                platform=platform,
                                meeting_id=meeting_id,
                            )
                        )

        return links

    def _extract_meeting_id(self, url: str, platform: MeetingPlatform) -> Optional[str]:
        """Extract meeting ID from URL."""
        if platform == MeetingPlatform.ZOOM:
            match = re.search(r"/j/(\d+)", url)
            if match:
                return match.group(1)
        elif platform == MeetingPlatform.BLUEJEANS:
            match = re.search(r"/(\d+)$", url)
            if match:
                return match.group(1)
        elif platform == MeetingPlatform.GOTOMEETING:
            match = re.search(r"/join/(\d+)", url)
            if match:
                return match.group(1)
        return None

    def _extract_title(self, subject: str, body: str) -> Optional[str]:
        """Extract meeting title from email."""
        # Remove common prefixes
        title = subject
        for prefix in ["Re:", "Fwd:", "FW:", "RE:", "Invitation:", "Reminder:", "Updated:"]:
            if title.startswith(prefix):
                title = title[len(prefix) :].strip()

        # Try to find explicit meeting title in body
        title_patterns = [
            r"(?:meeting|event)(?:\s+title)?:\s*(.+?)(?:\n|$)",
            r"subject:\s*(.+?)(?:\n|$)",
            r"regarding:\s*(.+?)(?:\n|$)",
        ]

        for pattern in title_patterns:
            match = re.search(pattern, body, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        return title if title else None

    def _extract_time(
        self,
        text: str,
    ) -> tuple[Optional[datetime], Optional[datetime], Optional[int]]:
        """Extract meeting time from text."""
        now = datetime.now(timezone.utc)

        # Try to find time range
        time_range_pattern = (
            r"(\d{1,2}:\d{2}\s*(?:am|pm)?)\s*[-–to]+\s*(\d{1,2}:\d{2}\s*(?:am|pm)?)"
        )
        range_match = re.search(time_range_pattern, text, re.IGNORECASE)

        if range_match:
            start_str = range_match.group(1)
            end_str = range_match.group(2)

            start_time = self._parse_time_string(start_str, now)
            end_time = self._parse_time_string(end_str, now)

            if start_time and end_time:
                if end_time <= start_time:
                    end_time += timedelta(hours=12)  # Assume PM for end time
                duration = int((end_time - start_time).total_seconds() / 60)
                return start_time, end_time, duration

        # Try date + time patterns
        date_time_patterns = [
            # January 15, 2025 at 2:00 PM
            r"(\w+)\s+(\d{1,2}),?\s+(\d{4})\s+(?:at\s+)?(\d{1,2}:\d{2}\s*(?:am|pm)?)",
            # 1/15/2025 2:00 PM
            r"(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\s+(\d{1,2}:\d{2}\s*(?:am|pm)?)",
            # Today at 2:00 PM
            r"(today|tomorrow)\s+at\s+(\d{1,2}:\d{2}\s*(?:am|pm)?)",
        ]

        for pattern in date_time_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                start_time = self._parse_datetime_match(match, now)
                if start_time:
                    # Default 1 hour duration
                    return start_time, start_time + timedelta(hours=1), 60

        # Try duration
        extracted_duration: Optional[int] = self._extract_duration(text)

        return None, None, extracted_duration

    def _parse_time_string(self, time_str: str, base_date: datetime) -> Optional[datetime]:
        """Parse a time string like '2:00 pm' into datetime."""
        time_str = time_str.strip().lower()

        try:
            # Parse with AM/PM
            if "pm" in time_str:
                time_str = time_str.replace("pm", "").strip()
                parts = time_str.split(":")
                hour = int(parts[0])
                minute = int(parts[1]) if len(parts) > 1 else 0
                if hour != 12:
                    hour += 12
            elif "am" in time_str:
                time_str = time_str.replace("am", "").strip()
                parts = time_str.split(":")
                hour = int(parts[0])
                minute = int(parts[1]) if len(parts) > 1 else 0
                if hour == 12:
                    hour = 0
            else:
                parts = time_str.split(":")
                hour = int(parts[0])
                minute = int(parts[1]) if len(parts) > 1 else 0

            return base_date.replace(hour=hour, minute=minute, second=0, microsecond=0)

        except (ValueError, IndexError):
            return None

    def _parse_datetime_match(
        self,
        match: re.Match,
        now: datetime,
    ) -> Optional[datetime]:
        """Parse datetime from regex match."""
        groups = match.groups()

        try:
            # Handle "today/tomorrow at time" pattern
            if groups[0].lower() in ("today", "tomorrow"):
                base = now
                if groups[0].lower() == "tomorrow":
                    base = now + timedelta(days=1)
                return self._parse_time_string(groups[1], base)

            # Handle month name format
            month_names = {
                "january": 1,
                "february": 2,
                "march": 3,
                "april": 4,
                "may": 5,
                "june": 6,
                "july": 7,
                "august": 8,
                "september": 9,
                "october": 10,
                "november": 11,
                "december": 12,
                "jan": 1,
                "feb": 2,
                "mar": 3,
                "apr": 4,
                "jun": 6,
                "jul": 7,
                "aug": 8,
                "sep": 9,
                "oct": 10,
                "nov": 11,
                "dec": 12,
            }

            if groups[0].lower() in month_names:
                month = month_names[groups[0].lower()]
                day = int(groups[1])
                year = int(groups[2])
                time_str = groups[3] if len(groups) > 3 else "9:00am"

                base = datetime(year, month, day, tzinfo=timezone.utc)
                return self._parse_time_string(time_str, base)

            # Handle numeric date format
            if len(groups) >= 3:
                month = int(groups[0])
                day = int(groups[1])
                year = int(groups[2])
                if year < 100:
                    year += 2000

                time_str = groups[3] if len(groups) > 3 else "9:00am"
                base = datetime(year, month, day, tzinfo=timezone.utc)
                return self._parse_time_string(time_str, base)

        except (ValueError, TypeError):
            pass

        return None

    def _extract_duration(self, text: str) -> Optional[int]:
        """Extract meeting duration in minutes from text."""
        for pattern, pattern_type in DURATION_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                if pattern_type == "hours_minutes":
                    hours = int(match.group(1))
                    minutes = int(match.group(2)) if match.group(2) else 0
                    return hours * 60 + minutes
                elif pattern_type == "minutes":
                    return int(match.group(1))
                elif pattern_type == "hours_decimal":
                    return int(float(match.group(1)) * 60)

        return None

    def _extract_participants(
        self,
        text: str,
        sender: str,
    ) -> List[MeetingParticipant]:
        """Extract meeting participants from text."""
        participants = []
        seen_emails = {sender.lower()} if sender else set()

        # Email pattern
        email_pattern = r"[\w\.-]+@[\w\.-]+\.\w+"

        for match in re.finditer(email_pattern, text):
            email = match.group(0).lower()
            if email not in seen_emails:
                seen_emails.add(email)
                participants.append(MeetingParticipant(email=email))

        return participants

    def _extract_location(self, text: str) -> Optional[str]:
        """Extract meeting location from text."""
        location_patterns = [
            r"location:\s*(.+?)(?:\n|$)",
            r"where:\s*(.+?)(?:\n|$)",
            r"room:\s*(.+?)(?:\n|$)",
            r"conference\s+room:\s*(.+?)(?:\n|$)",
            r"at\s+the\s+(.+?(?:room|office|building|floor).*?)(?:\n|$)",
        ]

        for pattern in location_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                location = match.group(1).strip()
                # Don't return URLs as locations
                if not location.startswith("http"):
                    return location

        return None

    async def _check_conflicts(
        self,
        start_time: datetime,
        end_time: datetime,
    ) -> List[ConflictInfo]:
        """Check calendar for conflicts with proposed meeting time."""
        conflicts = []

        # Check Google Calendar
        if self.google_calendar:
            try:
                google_conflicts = await self.google_calendar.find_conflicts(
                    proposed_start=start_time,
                    proposed_end=end_time,
                )
                for event in google_conflicts:
                    if event.start and event.end:
                        conflicts.append(
                            ConflictInfo(
                                event_id=event.id,
                                title=event.summary,
                                start=event.start,
                                end=event.end,
                                calendar_source="google",
                            )
                        )
            except Exception as e:
                logger.warning(f"Failed to check Google Calendar: {e}")

        # Check Outlook Calendar
        if self.outlook_calendar:
            try:
                outlook_conflicts = await self.outlook_calendar.find_conflicts(
                    proposed_start=start_time,
                    proposed_end=end_time,
                )
                for outlook_event in outlook_conflicts:
                    if outlook_event.start and outlook_event.end:
                        conflicts.append(
                            ConflictInfo(
                                event_id=outlook_event.id,
                                title=outlook_event.subject,
                                start=outlook_event.start,
                                end=outlook_event.end,
                                calendar_source="outlook",
                            )
                        )
            except Exception as e:
                logger.warning(f"Failed to check Outlook Calendar: {e}")

        return conflicts

    def _generate_rationale(
        self,
        meeting_type: MeetingType,
        patterns: List[str],
    ) -> str:
        """Generate human-readable rationale."""
        type_descriptions = {
            MeetingType.INVITE: "meeting invitation",
            MeetingType.UPDATE: "meeting update",
            MeetingType.RESCHEDULE: "meeting reschedule",
            MeetingType.CANCELLATION: "meeting cancellation",
            MeetingType.CONFIRMATION: "meeting confirmation",
            MeetingType.REMINDER: "meeting reminder",
            MeetingType.RESPONSE: "meeting response/RSVP",
            MeetingType.AGENDA: "meeting agenda",
            MeetingType.FOLLOWUP: "meeting follow-up",
        }

        desc = type_descriptions.get(meeting_type, "meeting-related content")
        pattern_count = len(patterns)

        return f"Detected {desc}. Matched {pattern_count} pattern(s)."

    async def detect_batch(
        self,
        emails: List[EmailMessage],
        check_calendar: bool = True,
    ) -> List[MeetingDetectionResult]:
        """
        Detect meetings from multiple emails.

        Args:
            emails: List of emails to analyze
            check_calendar: Whether to check calendar for conflicts

        Returns:
            List of detection results
        """
        import asyncio

        tasks = [self.detect_meeting(email, check_calendar) for email in emails]
        return await asyncio.gather(*tasks)

    async def get_upcoming_meetings_from_emails(
        self,
        emails: List[EmailMessage],
        hours_ahead: int = 24,
    ) -> List[MeetingDetectionResult]:
        """
        Get meeting emails for upcoming events.

        Args:
            emails: List of emails to scan
            hours_ahead: How far ahead to look

        Returns:
            List of meeting results sorted by start time
        """
        now = datetime.now(timezone.utc)
        cutoff = now + timedelta(hours=hours_ahead)

        results = await self.detect_batch(emails, check_calendar=False)

        upcoming = [
            r for r in results if r.is_meeting and r.start_time and now <= r.start_time <= cutoff
        ]

        # Sort by start time
        upcoming.sort(key=lambda r: r.start_time or datetime.max.replace(tzinfo=timezone.utc))

        return upcoming


# Convenience function
async def detect_meeting_quick(
    subject: str,
    body: str,
    sender: str = "",
) -> MeetingDetectionResult:
    """
    Quick meeting detection without full email object.

    Args:
        subject: Email subject
        body: Email body text
        sender: Sender email address

    Returns:
        MeetingDetectionResult
    """

    class SimpleEmail:
        def __init__(self, subject: str, body: str, sender: str):
            self.id = f"quick_{hash((subject, body, sender))}"
            self.subject = subject
            self.body_text = body
            self.from_address = sender

    email = SimpleEmail(subject, body, sender)
    detector = MeetingDetector()
    return await detector.detect_meeting(email, check_calendar=False)  # type: ignore


__all__ = [
    "MeetingDetector",
    "MeetingDetectionResult",
    "MeetingType",
    "MeetingPlatform",
    "MeetingLink",
    "MeetingParticipant",
    "ConflictInfo",
    "detect_meeting_quick",
]
