"""
Google Calendar Connector.

Provides integration with Google Calendar:
- OAuth2 authentication flow
- Event fetching and creation
- Free/busy availability checking
- Calendar list management
- Watch for changes (push notifications)

Requires Google Cloud OAuth2 credentials with Calendar scopes.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import aiohttp

from aragora.connectors.enterprise.base import EnterpriseConnector, SyncState
from aragora.reasoning.provenance import SourceType
from aragora.resilience import CircuitBreaker

logger = logging.getLogger(__name__)


# Google Calendar API scopes
CALENDAR_SCOPES_READONLY = [
    "https://www.googleapis.com/auth/calendar.readonly",
    "https://www.googleapis.com/auth/calendar.events.readonly",
]

CALENDAR_SCOPES_FULL = [
    "https://www.googleapis.com/auth/calendar",
    "https://www.googleapis.com/auth/calendar.events",
]

# Default to read-only for inbox integration
CALENDAR_SCOPES = CALENDAR_SCOPES_READONLY


@dataclass
class CalendarEvent:
    """Represents a calendar event."""

    id: str
    calendar_id: str
    summary: str
    description: Optional[str] = None
    location: Optional[str] = None
    start: Optional[datetime] = None
    end: Optional[datetime] = None
    all_day: bool = False
    status: str = "confirmed"  # confirmed, tentative, cancelled
    organizer_email: Optional[str] = None
    organizer_name: Optional[str] = None
    attendees: List[Dict[str, Any]] = field(default_factory=list)
    html_link: Optional[str] = None
    hangout_link: Optional[str] = None
    conference_data: Optional[Dict[str, Any]] = None
    recurrence: Optional[List[str]] = None
    recurring_event_id: Optional[str] = None
    created: Optional[datetime] = None
    updated: Optional[datetime] = None
    etag: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "calendar_id": self.calendar_id,
            "summary": self.summary,
            "description": self.description,
            "location": self.location,
            "start": self.start.isoformat() if self.start else None,
            "end": self.end.isoformat() if self.end else None,
            "all_day": self.all_day,
            "status": self.status,
            "organizer_email": self.organizer_email,
            "organizer_name": self.organizer_name,
            "attendees": self.attendees,
            "html_link": self.html_link,
            "hangout_link": self.hangout_link,
            "conference_data": self.conference_data,
            "recurrence": self.recurrence,
            "recurring_event_id": self.recurring_event_id,
            "created": self.created.isoformat() if self.created else None,
            "updated": self.updated.isoformat() if self.updated else None,
        }


@dataclass
class FreeBusySlot:
    """Represents a busy time slot."""

    start: datetime
    end: datetime

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
        }


@dataclass
class CalendarInfo:
    """Represents a calendar."""

    id: str
    summary: str
    description: Optional[str] = None
    timezone: Optional[str] = None
    primary: bool = False
    access_role: str = "reader"
    background_color: Optional[str] = None
    foreground_color: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "summary": self.summary,
            "description": self.description,
            "timezone": self.timezone,
            "primary": self.primary,
            "access_role": self.access_role,
            "background_color": self.background_color,
            "foreground_color": self.foreground_color,
        }


class GoogleCalendarConnector(EnterpriseConnector):
    """
    Enterprise connector for Google Calendar.

    Features:
    - OAuth2 authentication with refresh tokens
    - Event listing with time range filtering
    - Free/busy availability checking
    - Calendar list management
    - Meeting conflict detection

    Authentication:
    - OAuth2 with refresh token (required)

    Usage:
        connector = GoogleCalendarConnector()

        # Get OAuth URL for user authorization
        url = connector.get_oauth_url(redirect_uri, state)

        # After user authorizes, exchange code for tokens
        await connector.authenticate(code=auth_code, redirect_uri=redirect_uri)

        # Get events
        events = await connector.get_events(
            time_min=datetime.now(),
            time_max=datetime.now() + timedelta(days=7)
        )

        # Check availability
        busy = await connector.get_free_busy(
            time_min=datetime.now(),
            time_max=datetime.now() + timedelta(hours=2)
        )
    """

    API_BASE = "https://www.googleapis.com/calendar/v3"
    TOKEN_URL = "https://oauth2.googleapis.com/token"
    AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"

    def __init__(
        self,
        calendar_ids: Optional[List[str]] = None,
        max_results: int = 250,
        **kwargs,
    ):
        """
        Initialize Google Calendar connector.

        Args:
            calendar_ids: Specific calendars to sync (None = primary only)
            max_results: Max events per request
        """
        super().__init__(connector_id="google_calendar", **kwargs)

        self.calendar_ids = calendar_ids
        self.max_results = max_results

        # OAuth tokens
        self._access_token: Optional[str] = None
        self._refresh_token: Optional[str] = None
        self._token_expiry: Optional[datetime] = None
        self._token_lock: asyncio.Lock = asyncio.Lock()

        # Circuit breaker for API calls
        self._circuit_breaker = CircuitBreaker(
            name="google_calendar",
            failure_threshold=5,
            recovery_timeout=60,
        )

        # HTTP session
        self._session: Optional[aiohttp.ClientSession] = None

    @property
    def source_type(self) -> SourceType:
        return SourceType.DOCUMENT

    @property
    def name(self) -> str:
        return "Google Calendar"

    def is_configured(self) -> bool:
        """Check if connector has required configuration."""
        import os

        return bool(
            os.environ.get("GOOGLE_CALENDAR_CLIENT_ID") or os.environ.get("GOOGLE_CLIENT_ID")
        )

    def get_oauth_url(self, redirect_uri: str, state: str = "") -> str:
        """
        Generate OAuth2 authorization URL.

        Args:
            redirect_uri: URL to redirect after authorization
            state: Optional state parameter for CSRF protection

        Returns:
            Authorization URL for user to visit
        """
        import os
        from urllib.parse import urlencode

        client_id = os.environ.get("GOOGLE_CALENDAR_CLIENT_ID") or os.environ.get(
            "GOOGLE_CLIENT_ID", ""
        )

        params = {
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": " ".join(CALENDAR_SCOPES),
            "access_type": "offline",
            "prompt": "consent",
        }
        if state:
            params["state"] = state

        return f"{self.AUTH_URL}?{urlencode(params)}"

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def _ensure_token(self) -> str:
        """Ensure we have a valid access token, refreshing if needed."""
        async with self._token_lock:
            now = datetime.now(timezone.utc)

            # Check if token is valid
            if self._access_token and self._token_expiry:
                if now < self._token_expiry - timedelta(minutes=5):
                    return self._access_token

            # Need to refresh
            if not self._refresh_token:
                raise ValueError("No refresh token available. Re-authenticate required.")

            await self._refresh_access_token()
            return self._access_token

    async def _refresh_access_token(self) -> None:
        """Refresh the access token using refresh token."""
        import os

        client_id = os.environ.get("GOOGLE_CALENDAR_CLIENT_ID") or os.environ.get(
            "GOOGLE_CLIENT_ID", ""
        )
        client_secret = os.environ.get("GOOGLE_CALENDAR_CLIENT_SECRET") or os.environ.get(
            "GOOGLE_CLIENT_SECRET", ""
        )

        session = await self._get_session()

        async with session.post(
            self.TOKEN_URL,
            data={
                "client_id": client_id,
                "client_secret": client_secret,
                "refresh_token": self._refresh_token,
                "grant_type": "refresh_token",
            },
        ) as resp:
            if resp.status != 200:
                error = await resp.text()
                logger.error(f"Token refresh failed: {error}")
                raise ValueError(f"Token refresh failed: {resp.status}")

            data = await resp.json()
            self._access_token = data["access_token"]
            expires_in = data.get("expires_in", 3600)
            self._token_expiry = datetime.now(timezone.utc) + timedelta(seconds=expires_in)

    async def authenticate(
        self,
        code: Optional[str] = None,
        redirect_uri: Optional[str] = None,
        refresh_token: Optional[str] = None,
    ) -> bool:
        """
        Authenticate with Google Calendar.

        Args:
            code: Authorization code from OAuth flow
            redirect_uri: Redirect URI used in OAuth flow
            refresh_token: Existing refresh token

        Returns:
            True if authentication successful
        """
        import os

        if refresh_token:
            self._refresh_token = refresh_token
            await self._refresh_access_token()
            return True

        if not code or not redirect_uri:
            raise ValueError("Either refresh_token or (code, redirect_uri) required")

        client_id = os.environ.get("GOOGLE_CALENDAR_CLIENT_ID") or os.environ.get(
            "GOOGLE_CLIENT_ID", ""
        )
        client_secret = os.environ.get("GOOGLE_CALENDAR_CLIENT_SECRET") or os.environ.get(
            "GOOGLE_CLIENT_SECRET", ""
        )

        session = await self._get_session()

        async with session.post(
            self.TOKEN_URL,
            data={
                "client_id": client_id,
                "client_secret": client_secret,
                "code": code,
                "redirect_uri": redirect_uri,
                "grant_type": "authorization_code",
            },
        ) as resp:
            if resp.status != 200:
                error = await resp.text()
                logger.error(f"Token exchange failed: {error}")
                return False

            data = await resp.json()
            self._access_token = data["access_token"]
            self._refresh_token = data.get("refresh_token")
            expires_in = data.get("expires_in", 3600)
            self._token_expiry = datetime.now(timezone.utc) + timedelta(seconds=expires_in)

            return True

    async def _api_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        json_data: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Make authenticated API request."""
        token = await self._ensure_token()
        session = await self._get_session()

        url = f"{self.API_BASE}{endpoint}"
        headers = {"Authorization": f"Bearer {token}"}

        async def _make_request():
            async with session.request(
                method,
                url,
                headers=headers,
                params=params,
                json=json_data,
            ) as resp:
                if resp.status == 401:
                    # Token expired, refresh and retry
                    await self._refresh_access_token()
                    headers["Authorization"] = f"Bearer {self._access_token}"
                    async with session.request(
                        method,
                        url,
                        headers=headers,
                        params=params,
                        json=json_data,
                    ) as retry_resp:
                        if retry_resp.status >= 400:
                            error = await retry_resp.text()
                            raise ValueError(f"API error: {retry_resp.status} - {error}")
                        return await retry_resp.json()

                if resp.status >= 400:
                    error = await resp.text()
                    raise ValueError(f"API error: {resp.status} - {error}")

                return await resp.json()

        return await self._circuit_breaker.call(_make_request)

    async def get_calendars(self) -> List[CalendarInfo]:
        """Get list of user's calendars."""
        data = await self._api_request("GET", "/users/me/calendarList")

        calendars = []
        for item in data.get("items", []):
            calendars.append(
                CalendarInfo(
                    id=item["id"],
                    summary=item.get("summary", ""),
                    description=item.get("description"),
                    timezone=item.get("timeZone"),
                    primary=item.get("primary", False),
                    access_role=item.get("accessRole", "reader"),
                    background_color=item.get("backgroundColor"),
                    foreground_color=item.get("foregroundColor"),
                )
            )

        return calendars

    async def get_events(
        self,
        calendar_id: str = "primary",
        time_min: Optional[datetime] = None,
        time_max: Optional[datetime] = None,
        query: Optional[str] = None,
        single_events: bool = True,
        order_by: str = "startTime",
        max_results: Optional[int] = None,
    ) -> List[CalendarEvent]:
        """
        Get events from a calendar.

        Args:
            calendar_id: Calendar ID (default: primary)
            time_min: Start of time range
            time_max: End of time range
            query: Free text search query
            single_events: Expand recurring events
            order_by: Sort order (startTime or updated)
            max_results: Maximum events to return

        Returns:
            List of calendar events
        """
        params: Dict[str, Any] = {
            "singleEvents": str(single_events).lower(),
            "orderBy": order_by,
            "maxResults": max_results or self.max_results,
        }

        if time_min:
            params["timeMin"] = time_min.isoformat()
        if time_max:
            params["timeMax"] = time_max.isoformat()
        if query:
            params["q"] = query

        data = await self._api_request("GET", f"/calendars/{calendar_id}/events", params=params)

        events = []
        for item in data.get("items", []):
            event = self._parse_event(item, calendar_id)
            if event:
                events.append(event)

        return events

    def _parse_event(self, item: Dict[str, Any], calendar_id: str) -> Optional[CalendarEvent]:
        """Parse API response into CalendarEvent."""
        try:
            # Parse start time
            start_data = item.get("start", {})
            end_data = item.get("end", {})

            if "dateTime" in start_data:
                start = datetime.fromisoformat(start_data["dateTime"].replace("Z", "+00:00"))
                all_day = False
            elif "date" in start_data:
                start = datetime.strptime(start_data["date"], "%Y-%m-%d").replace(
                    tzinfo=timezone.utc
                )
                all_day = True
            else:
                start = None
                all_day = False

            if "dateTime" in end_data:
                end = datetime.fromisoformat(end_data["dateTime"].replace("Z", "+00:00"))
            elif "date" in end_data:
                end = datetime.strptime(end_data["date"], "%Y-%m-%d").replace(tzinfo=timezone.utc)
            else:
                end = None

            # Parse organizer
            organizer = item.get("organizer", {})
            organizer_email = organizer.get("email")
            organizer_name = organizer.get("displayName")

            # Parse attendees
            attendees = []
            for att in item.get("attendees", []):
                attendees.append(
                    {
                        "email": att.get("email"),
                        "name": att.get("displayName"),
                        "response_status": att.get("responseStatus"),
                        "optional": att.get("optional", False),
                        "organizer": att.get("organizer", False),
                        "self": att.get("self", False),
                    }
                )

            # Parse timestamps
            created = None
            if "created" in item:
                created = datetime.fromisoformat(item["created"].replace("Z", "+00:00"))
            updated = None
            if "updated" in item:
                updated = datetime.fromisoformat(item["updated"].replace("Z", "+00:00"))

            return CalendarEvent(
                id=item["id"],
                calendar_id=calendar_id,
                summary=item.get("summary", "(No title)"),
                description=item.get("description"),
                location=item.get("location"),
                start=start,
                end=end,
                all_day=all_day,
                status=item.get("status", "confirmed"),
                organizer_email=organizer_email,
                organizer_name=organizer_name,
                attendees=attendees,
                html_link=item.get("htmlLink"),
                hangout_link=item.get("hangoutLink"),
                conference_data=item.get("conferenceData"),
                recurrence=item.get("recurrence"),
                recurring_event_id=item.get("recurringEventId"),
                created=created,
                updated=updated,
                etag=item.get("etag"),
            )
        except Exception as e:
            logger.warning(f"Failed to parse event: {e}")
            return None

    async def get_free_busy(
        self,
        time_min: datetime,
        time_max: datetime,
        calendar_ids: Optional[List[str]] = None,
    ) -> Dict[str, List[FreeBusySlot]]:
        """
        Get free/busy information for calendars.

        Args:
            time_min: Start of time range
            time_max: End of time range
            calendar_ids: Calendar IDs to check (default: primary)

        Returns:
            Dict mapping calendar ID to list of busy slots
        """
        if not calendar_ids:
            calendar_ids = self.calendar_ids or ["primary"]

        request_body = {
            "timeMin": time_min.isoformat(),
            "timeMax": time_max.isoformat(),
            "items": [{"id": cal_id} for cal_id in calendar_ids],
        }

        data = await self._api_request("POST", "/freeBusy", json_data=request_body)

        result: Dict[str, List[FreeBusySlot]] = {}
        for cal_id, cal_data in data.get("calendars", {}).items():
            busy_slots = []
            for busy in cal_data.get("busy", []):
                start = datetime.fromisoformat(busy["start"].replace("Z", "+00:00"))
                end = datetime.fromisoformat(busy["end"].replace("Z", "+00:00"))
                busy_slots.append(FreeBusySlot(start=start, end=end))
            result[cal_id] = busy_slots

        return result

    async def check_availability(
        self,
        time_min: datetime,
        time_max: datetime,
        calendar_ids: Optional[List[str]] = None,
    ) -> bool:
        """
        Check if user is available during a time range.

        Args:
            time_min: Start of time range
            time_max: End of time range
            calendar_ids: Calendar IDs to check

        Returns:
            True if user is free during the entire time range
        """
        busy = await self.get_free_busy(time_min, time_max, calendar_ids)

        for cal_id, slots in busy.items():
            if slots:
                return False

        return True

    async def get_upcoming_events(
        self,
        hours: int = 24,
        calendar_ids: Optional[List[str]] = None,
    ) -> List[CalendarEvent]:
        """
        Get upcoming events within a time window.

        Args:
            hours: Number of hours to look ahead
            calendar_ids: Calendar IDs to check (default: all accessible)

        Returns:
            List of upcoming events sorted by start time
        """
        now = datetime.now(timezone.utc)
        time_max = now + timedelta(hours=hours)

        if not calendar_ids:
            calendar_ids = self.calendar_ids or ["primary"]

        all_events = []
        for cal_id in calendar_ids:
            try:
                events = await self.get_events(
                    calendar_id=cal_id,
                    time_min=now,
                    time_max=time_max,
                )
                all_events.extend(events)
            except Exception as e:
                logger.warning(f"Failed to get events from {cal_id}: {e}")

        # Sort by start time
        all_events.sort(key=lambda e: e.start or datetime.max.replace(tzinfo=timezone.utc))

        return all_events

    async def find_conflicts(
        self,
        proposed_start: datetime,
        proposed_end: datetime,
        calendar_ids: Optional[List[str]] = None,
    ) -> List[CalendarEvent]:
        """
        Find events that conflict with a proposed time.

        Args:
            proposed_start: Proposed event start
            proposed_end: Proposed event end
            calendar_ids: Calendar IDs to check

        Returns:
            List of conflicting events
        """
        events = await self.get_events(
            time_min=proposed_start,
            time_max=proposed_end,
            calendar_ids=calendar_ids[0] if calendar_ids else "primary",
        )

        conflicts = []
        for event in events:
            if event.status == "cancelled":
                continue
            if event.start and event.end:
                # Check for overlap
                if event.start < proposed_end and event.end > proposed_start:
                    conflicts.append(event)

        return conflicts

    async def close(self) -> None:
        """Close the connector and release resources."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def sync(
        self,
        state: Optional[SyncState] = None,
        full_sync: bool = False,
    ):
        """
        Sync events from calendars.

        For inbox integration, we primarily use get_events() and get_free_busy()
        directly rather than full sync.
        """
        # Get upcoming events from all calendars
        calendar_ids = self.calendar_ids or ["primary"]
        events = await self.get_upcoming_events(hours=168, calendar_ids=calendar_ids)  # 1 week
        return {"events": [e.to_dict() for e in events]}
