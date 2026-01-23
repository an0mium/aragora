"""
Outlook/Microsoft 365 Calendar Connector.

Provides integration with Outlook Calendar via Microsoft Graph API:
- OAuth2 authentication flow
- Event fetching and creation
- Free/busy availability checking
- Calendar list management
- Meeting conflict detection

Requires Azure AD app registration with Calendar scopes.
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


# Microsoft Graph Calendar API scopes
CALENDAR_SCOPES_READONLY = [
    "https://graph.microsoft.com/Calendars.Read",
    "https://graph.microsoft.com/User.Read",
]

CALENDAR_SCOPES_FULL = [
    "https://graph.microsoft.com/Calendars.ReadWrite",
    "https://graph.microsoft.com/User.Read",
]

# Default to read-only for inbox integration
CALENDAR_SCOPES = CALENDAR_SCOPES_READONLY


@dataclass
class OutlookCalendarEvent:
    """Represents an Outlook calendar event."""

    id: str
    calendar_id: str
    subject: str
    body_preview: Optional[str] = None
    body_content: Optional[str] = None
    location: Optional[str] = None
    start: Optional[datetime] = None
    end: Optional[datetime] = None
    all_day: bool = False
    show_as: str = "busy"  # free, tentative, busy, oof, workingElsewhere
    is_cancelled: bool = False
    is_organizer: bool = False
    organizer_email: Optional[str] = None
    organizer_name: Optional[str] = None
    attendees: List[Dict[str, Any]] = field(default_factory=list)
    web_link: Optional[str] = None
    online_meeting_url: Optional[str] = None
    online_meeting_provider: Optional[str] = None
    is_online_meeting: bool = False
    recurrence: Optional[Dict[str, Any]] = None
    series_master_id: Optional[str] = None
    response_status: Optional[str] = None
    sensitivity: str = "normal"  # normal, personal, private, confidential
    importance: str = "normal"  # low, normal, high
    categories: List[str] = field(default_factory=list)
    created: Optional[datetime] = None
    last_modified: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "calendar_id": self.calendar_id,
            "subject": self.subject,
            "body_preview": self.body_preview,
            "body_content": self.body_content,
            "location": self.location,
            "start": self.start.isoformat() if self.start else None,
            "end": self.end.isoformat() if self.end else None,
            "all_day": self.all_day,
            "show_as": self.show_as,
            "is_cancelled": self.is_cancelled,
            "is_organizer": self.is_organizer,
            "organizer_email": self.organizer_email,
            "organizer_name": self.organizer_name,
            "attendees": self.attendees,
            "web_link": self.web_link,
            "online_meeting_url": self.online_meeting_url,
            "online_meeting_provider": self.online_meeting_provider,
            "is_online_meeting": self.is_online_meeting,
            "recurrence": self.recurrence,
            "series_master_id": self.series_master_id,
            "response_status": self.response_status,
            "sensitivity": self.sensitivity,
            "importance": self.importance,
            "categories": self.categories,
            "created": self.created.isoformat() if self.created else None,
            "last_modified": self.last_modified.isoformat() if self.last_modified else None,
        }


@dataclass
class OutlookFreeBusySlot:
    """Represents a busy time slot from schedule info."""

    start: datetime
    end: datetime
    status: str = "busy"  # free, tentative, busy, oof, workingElsewhere

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "status": self.status,
        }


@dataclass
class OutlookCalendarInfo:
    """Represents an Outlook calendar."""

    id: str
    name: str
    color: Optional[str] = None
    is_default: bool = False
    can_edit: bool = True
    can_share: bool = True
    can_view_private_items: bool = True
    owner_email: Optional[str] = None
    owner_name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "color": self.color,
            "is_default": self.is_default,
            "can_edit": self.can_edit,
            "can_share": self.can_share,
            "can_view_private_items": self.can_view_private_items,
            "owner_email": self.owner_email,
            "owner_name": self.owner_name,
        }


class OutlookCalendarConnector(EnterpriseConnector):
    """
    Enterprise connector for Outlook/Microsoft 365 Calendar.

    Features:
    - OAuth2 authentication with refresh tokens
    - Event listing with time range filtering
    - Free/busy availability checking via getSchedule
    - Calendar list management
    - Meeting conflict detection
    - Teams/Skype meeting integration

    Authentication:
    - OAuth2 with refresh token (required)

    Usage:
        connector = OutlookCalendarConnector()

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
        available = await connector.check_availability(
            time_min=datetime.now(),
            time_max=datetime.now() + timedelta(hours=2)
        )
    """

    API_BASE = "https://graph.microsoft.com/v1.0"
    AUTH_URL_TEMPLATE = "https://login.microsoftonline.com/{tenant}/oauth2/v2.0/authorize"
    TOKEN_URL_TEMPLATE = "https://login.microsoftonline.com/{tenant}/oauth2/v2.0/token"

    def __init__(
        self,
        calendar_ids: Optional[List[str]] = None,
        max_results: int = 250,
        user_id: str = "me",
        **kwargs,
    ):
        """
        Initialize Outlook Calendar connector.

        Args:
            calendar_ids: Specific calendars to sync (None = default calendar)
            max_results: Max events per request
            user_id: User ID ("me" for authenticated user)
        """
        super().__init__(connector_id="outlook_calendar", **kwargs)

        self.calendar_ids = calendar_ids
        self.max_results = max_results
        self.user_id = user_id

        # OAuth tokens
        self._access_token: Optional[str] = None
        self._refresh_token: Optional[str] = None
        self._token_expiry: Optional[datetime] = None
        self._token_lock: asyncio.Lock = asyncio.Lock()

        # Circuit breaker for API calls
        self._circuit_breaker = CircuitBreaker(
            name="outlook_calendar",
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
        return "Outlook Calendar"

    def _get_tenant(self) -> str:
        """Get Azure AD tenant ID."""
        import os

        return os.environ.get("OUTLOOK_TENANT_ID", "common")

    def is_configured(self) -> bool:
        """Check if connector has required configuration."""
        import os

        return bool(
            os.environ.get("OUTLOOK_CALENDAR_CLIENT_ID")
            or os.environ.get("OUTLOOK_CLIENT_ID")
            or os.environ.get("AZURE_CLIENT_ID")
            or os.environ.get("MICROSOFT_CLIENT_ID")
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

        client_id = (
            os.environ.get("OUTLOOK_CALENDAR_CLIENT_ID")
            or os.environ.get("OUTLOOK_CLIENT_ID")
            or os.environ.get("AZURE_CLIENT_ID")
            or os.environ.get("MICROSOFT_CLIENT_ID", "")
        )

        tenant = self._get_tenant()

        params = {
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": " ".join(CALENDAR_SCOPES + ["offline_access"]),
            "response_mode": "query",
            "prompt": "consent",
        }
        if state:
            params["state"] = state

        auth_url = self.AUTH_URL_TEMPLATE.format(tenant=tenant)
        return f"{auth_url}?{urlencode(params)}"

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

        client_id = (
            os.environ.get("OUTLOOK_CALENDAR_CLIENT_ID")
            or os.environ.get("OUTLOOK_CLIENT_ID")
            or os.environ.get("AZURE_CLIENT_ID")
            or os.environ.get("MICROSOFT_CLIENT_ID", "")
        )
        client_secret = (
            os.environ.get("OUTLOOK_CALENDAR_CLIENT_SECRET")
            or os.environ.get("OUTLOOK_CLIENT_SECRET")
            or os.environ.get("AZURE_CLIENT_SECRET")
            or os.environ.get("MICROSOFT_CLIENT_SECRET", "")
        )

        tenant = self._get_tenant()
        token_url = self.TOKEN_URL_TEMPLATE.format(tenant=tenant)

        session = await self._get_session()

        async with session.post(
            token_url,
            data={
                "client_id": client_id,
                "client_secret": client_secret,
                "refresh_token": self._refresh_token,
                "grant_type": "refresh_token",
                "scope": " ".join(CALENDAR_SCOPES + ["offline_access"]),
            },
        ) as resp:
            if resp.status != 200:
                error = await resp.text()
                logger.error(f"Token refresh failed: {error}")
                raise ValueError(f"Token refresh failed: {resp.status}")

            data = await resp.json()
            self._access_token = data["access_token"]
            self._refresh_token = data.get("refresh_token", self._refresh_token)
            expires_in = data.get("expires_in", 3600)
            self._token_expiry = datetime.now(timezone.utc) + timedelta(seconds=expires_in)

    async def authenticate(
        self,
        code: Optional[str] = None,
        redirect_uri: Optional[str] = None,
        refresh_token: Optional[str] = None,
    ) -> bool:
        """
        Authenticate with Outlook Calendar.

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

        client_id = (
            os.environ.get("OUTLOOK_CALENDAR_CLIENT_ID")
            or os.environ.get("OUTLOOK_CLIENT_ID")
            or os.environ.get("AZURE_CLIENT_ID")
            or os.environ.get("MICROSOFT_CLIENT_ID", "")
        )
        client_secret = (
            os.environ.get("OUTLOOK_CALENDAR_CLIENT_SECRET")
            or os.environ.get("OUTLOOK_CLIENT_SECRET")
            or os.environ.get("AZURE_CLIENT_SECRET")
            or os.environ.get("MICROSOFT_CLIENT_SECRET", "")
        )

        tenant = self._get_tenant()
        token_url = self.TOKEN_URL_TEMPLATE.format(tenant=tenant)

        session = await self._get_session()

        async with session.post(
            token_url,
            data={
                "client_id": client_id,
                "client_secret": client_secret,
                "code": code,
                "redirect_uri": redirect_uri,
                "grant_type": "authorization_code",
                "scope": " ".join(CALENDAR_SCOPES + ["offline_access"]),
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

        # Handle absolute URLs (for pagination)
        if endpoint.startswith("https://"):
            url = endpoint
        else:
            url = f"{self.API_BASE}/{self.user_id}{endpoint}"

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

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
                        return await retry_resp.json() if retry_resp.content else {}

                if resp.status >= 400:
                    error = await resp.text()
                    raise ValueError(f"API error: {resp.status} - {error}")

                return await resp.json() if resp.content else {}

        if not self._circuit_breaker.can_proceed():
            raise ValueError("Circuit breaker is open - API temporarily unavailable")
        try:
            result = await _make_request()
            self._circuit_breaker.record_success()
            return result
        except Exception:
            self._circuit_breaker.record_failure()
            raise

    async def get_calendars(self) -> List[OutlookCalendarInfo]:
        """Get list of user's calendars."""
        data = await self._api_request("GET", "/calendars")

        calendars = []
        for item in data.get("value", []):
            owner = item.get("owner", {})
            calendars.append(
                OutlookCalendarInfo(
                    id=item["id"],
                    name=item.get("name", ""),
                    color=item.get("color"),
                    is_default=item.get("isDefaultCalendar", False),
                    can_edit=item.get("canEdit", True),
                    can_share=item.get("canShare", True),
                    can_view_private_items=item.get("canViewPrivateItems", True),
                    owner_email=owner.get("address"),
                    owner_name=owner.get("name"),
                )
            )

        return calendars

    async def get_events(
        self,
        calendar_id: Optional[str] = None,
        time_min: Optional[datetime] = None,
        time_max: Optional[datetime] = None,
        query: Optional[str] = None,
        max_results: Optional[int] = None,
    ) -> List[OutlookCalendarEvent]:
        """
        Get events from a calendar.

        Args:
            calendar_id: Calendar ID (None = default calendar)
            time_min: Start of time range
            time_max: End of time range
            query: Free text search query
            max_results: Maximum events to return

        Returns:
            List of calendar events
        """
        params: Dict[str, Any] = {
            "$top": max_results or self.max_results,
            "$orderby": "start/dateTime",
        }

        # Build filter for time range
        filters = []
        if time_min:
            filters.append(f"start/dateTime ge '{time_min.isoformat()}'")
        if time_max:
            filters.append(f"end/dateTime le '{time_max.isoformat()}'")

        if filters:
            params["$filter"] = " and ".join(filters)

        if query:
            params["$search"] = f'"{query}"'

        # Select fields to return
        params["$select"] = ",".join(
            [
                "id",
                "subject",
                "bodyPreview",
                "body",
                "start",
                "end",
                "location",
                "isAllDay",
                "showAs",
                "isCancelled",
                "isOrganizer",
                "organizer",
                "attendees",
                "webLink",
                "onlineMeeting",
                "isOnlineMeeting",
                "onlineMeetingProvider",
                "recurrence",
                "seriesMasterId",
                "responseStatus",
                "sensitivity",
                "importance",
                "categories",
                "createdDateTime",
                "lastModifiedDateTime",
            ]
        )

        # Choose endpoint based on calendar_id
        if calendar_id:
            endpoint = f"/calendars/{calendar_id}/events"
        else:
            endpoint = "/calendar/events"

        data = await self._api_request("GET", endpoint, params=params)

        events = []
        for item in data.get("value", []):
            event = self._parse_event(item, calendar_id or "default")
            if event:
                events.append(event)

        return events

    def _parse_event(
        self, item: Dict[str, Any], calendar_id: str
    ) -> Optional[OutlookCalendarEvent]:
        """Parse API response into OutlookCalendarEvent."""
        try:
            # Parse start time
            start_data = item.get("start", {})
            end_data = item.get("end", {})

            if start_data.get("dateTime"):
                start_str = start_data["dateTime"]
                # Handle timezone
                if "Z" in start_str:
                    start = datetime.fromisoformat(start_str.replace("Z", "+00:00"))
                elif "+" in start_str or start_str.count("-") > 2:
                    start = datetime.fromisoformat(start_str)
                else:
                    # No timezone, assume UTC
                    start = datetime.fromisoformat(start_str).replace(tzinfo=timezone.utc)
            else:
                start = None

            if end_data.get("dateTime"):
                end_str = end_data["dateTime"]
                if "Z" in end_str:
                    end = datetime.fromisoformat(end_str.replace("Z", "+00:00"))
                elif "+" in end_str or end_str.count("-") > 2:
                    end = datetime.fromisoformat(end_str)
                else:
                    end = datetime.fromisoformat(end_str).replace(tzinfo=timezone.utc)
            else:
                end = None

            # Parse organizer
            organizer = item.get("organizer", {}).get("emailAddress", {})
            organizer_email = organizer.get("address")
            organizer_name = organizer.get("name")

            # Parse attendees
            attendees = []
            for att in item.get("attendees", []):
                email_addr = att.get("emailAddress", {})
                attendees.append(
                    {
                        "email": email_addr.get("address"),
                        "name": email_addr.get("name"),
                        "type": att.get("type"),  # required, optional, resource
                        "response_status": att.get("status", {}).get("response"),
                        "response_time": att.get("status", {}).get("time"),
                    }
                )

            # Parse location
            location_data = item.get("location", {})
            location = location_data.get("displayName")

            # Parse online meeting info
            online_meeting = item.get("onlineMeeting", {})
            online_meeting_url = online_meeting.get("joinUrl")

            # Parse body
            body_data = item.get("body", {})
            body_content = (
                body_data.get("content") if body_data.get("contentType") == "text" else None
            )

            # Parse timestamps
            created = None
            if item.get("createdDateTime"):
                created = datetime.fromisoformat(item["createdDateTime"].replace("Z", "+00:00"))
            last_modified = None
            if item.get("lastModifiedDateTime"):
                last_modified = datetime.fromisoformat(
                    item["lastModifiedDateTime"].replace("Z", "+00:00")
                )

            # Parse response status
            response_status = None
            if item.get("responseStatus"):
                response_status = item["responseStatus"].get("response")

            return OutlookCalendarEvent(
                id=item["id"],
                calendar_id=calendar_id,
                subject=item.get("subject", "(No subject)"),
                body_preview=item.get("bodyPreview"),
                body_content=body_content,
                location=location,
                start=start,
                end=end,
                all_day=item.get("isAllDay", False),
                show_as=item.get("showAs", "busy"),
                is_cancelled=item.get("isCancelled", False),
                is_organizer=item.get("isOrganizer", False),
                organizer_email=organizer_email,
                organizer_name=organizer_name,
                attendees=attendees,
                web_link=item.get("webLink"),
                online_meeting_url=online_meeting_url,
                online_meeting_provider=item.get("onlineMeetingProvider"),
                is_online_meeting=item.get("isOnlineMeeting", False),
                recurrence=item.get("recurrence"),
                series_master_id=item.get("seriesMasterId"),
                response_status=response_status,
                sensitivity=item.get("sensitivity", "normal"),
                importance=item.get("importance", "normal"),
                categories=item.get("categories", []),
                created=created,
                last_modified=last_modified,
            )
        except Exception as e:
            logger.warning(f"Failed to parse event: {e}")
            return None

    async def get_schedule(
        self,
        time_min: datetime,
        time_max: datetime,
        email_addresses: Optional[List[str]] = None,
    ) -> Dict[str, List[OutlookFreeBusySlot]]:
        """
        Get free/busy schedule information.

        Note: This requires Calendars.Read.Shared or higher permissions
        when checking other users' schedules.

        Args:
            time_min: Start of time range
            time_max: End of time range
            email_addresses: Email addresses to check (default: current user)

        Returns:
            Dict mapping email to list of busy slots
        """
        if not email_addresses:
            # Get current user's email
            user_info = await self._api_request("GET", "")
            email_addresses = [user_info.get("mail") or user_info.get("userPrincipalName")]

        request_body = {
            "schedules": email_addresses,
            "startTime": {
                "dateTime": time_min.isoformat(),
                "timeZone": "UTC",
            },
            "endTime": {
                "dateTime": time_max.isoformat(),
                "timeZone": "UTC",
            },
            "availabilityViewInterval": 30,  # 30-minute intervals
        }

        data = await self._api_request("POST", "/calendar/getSchedule", json_data=request_body)

        result: Dict[str, List[OutlookFreeBusySlot]] = {}
        for schedule in data.get("value", []):
            email = schedule.get("scheduleId", "")
            busy_slots = []

            for item in schedule.get("scheduleItems", []):
                start_data = item.get("start", {})
                end_data = item.get("end", {})

                start_str = start_data.get("dateTime", "")
                end_str = end_data.get("dateTime", "")

                if start_str and end_str:
                    start = datetime.fromisoformat(start_str.replace("Z", "+00:00"))
                    end = datetime.fromisoformat(end_str.replace("Z", "+00:00"))
                    status = item.get("status", "busy")
                    busy_slots.append(OutlookFreeBusySlot(start=start, end=end, status=status))

            result[email] = busy_slots

        return result

    async def get_free_busy(
        self,
        time_min: datetime,
        time_max: datetime,
        calendar_ids: Optional[List[str]] = None,
    ) -> Dict[str, List[OutlookFreeBusySlot]]:
        """
        Get free/busy information for calendars.

        Alternative to get_schedule that uses events directly.

        Args:
            time_min: Start of time range
            time_max: End of time range
            calendar_ids: Calendar IDs to check

        Returns:
            Dict mapping calendar ID to list of busy slots
        """
        if not calendar_ids:
            calendar_ids = self.calendar_ids or [None]  # None = default calendar

        result: Dict[str, List[OutlookFreeBusySlot]] = {}

        for cal_id in calendar_ids:
            try:
                events = await self.get_events(
                    calendar_id=cal_id,
                    time_min=time_min,
                    time_max=time_max,
                )

                busy_slots = []
                for event in events:
                    if event.is_cancelled:
                        continue
                    if event.show_as in ("free",):
                        continue
                    if event.start and event.end:
                        busy_slots.append(
                            OutlookFreeBusySlot(
                                start=event.start,
                                end=event.end,
                                status=event.show_as,
                            )
                        )

                result[cal_id or "default"] = busy_slots

            except Exception as e:
                logger.warning(f"Failed to get events for calendar {cal_id}: {e}")
                result[cal_id or "default"] = []

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
            for slot in slots:
                # Check for overlap with requested time range
                if slot.start < time_max and slot.end > time_min:
                    if slot.status not in ("free", "workingElsewhere"):
                        return False

        return True

    async def get_upcoming_events(
        self,
        hours: int = 24,
        calendar_ids: Optional[List[str]] = None,
    ) -> List[OutlookCalendarEvent]:
        """
        Get upcoming events within a time window.

        Args:
            hours: Number of hours to look ahead
            calendar_ids: Calendar IDs to check

        Returns:
            List of upcoming events sorted by start time
        """
        now = datetime.now(timezone.utc)
        time_max = now + timedelta(hours=hours)

        if not calendar_ids:
            calendar_ids = self.calendar_ids or [None]

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
    ) -> List[OutlookCalendarEvent]:
        """
        Find events that conflict with a proposed time.

        Args:
            proposed_start: Proposed event start
            proposed_end: Proposed event end
            calendar_ids: Calendar IDs to check

        Returns:
            List of conflicting events
        """
        if not calendar_ids:
            calendar_ids = self.calendar_ids or [None]

        conflicts = []
        for cal_id in calendar_ids:
            events = await self.get_events(
                calendar_id=cal_id,
                time_min=proposed_start,
                time_max=proposed_end,
            )

            for event in events:
                if event.is_cancelled:
                    continue
                if event.show_as == "free":
                    continue
                if event.start and event.end:
                    # Check for overlap
                    if event.start < proposed_end and event.end > proposed_start:
                        conflicts.append(event)

        return conflicts

    async def get_event(
        self,
        event_id: str,
        calendar_id: Optional[str] = None,
    ) -> Optional[OutlookCalendarEvent]:
        """
        Get a specific event by ID.

        Args:
            event_id: Event ID
            calendar_id: Calendar ID (None = default)

        Returns:
            Calendar event or None if not found
        """
        try:
            if calendar_id:
                endpoint = f"/calendars/{calendar_id}/events/{event_id}"
            else:
                endpoint = f"/calendar/events/{event_id}"

            data = await self._api_request("GET", endpoint)
            return self._parse_event(data, calendar_id or "default")
        except Exception as e:
            logger.warning(f"Failed to get event {event_id}: {e}")
            return None

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
        calendar_ids = self.calendar_ids or [None]
        events = await self.get_upcoming_events(hours=168, calendar_ids=calendar_ids)  # 1 week
        return {"events": [e.to_dict() for e in events]}


__all__ = [
    "OutlookCalendarConnector",
    "OutlookCalendarEvent",
    "OutlookFreeBusySlot",
    "OutlookCalendarInfo",
    "CALENDAR_SCOPES",
    "CALENDAR_SCOPES_READONLY",
    "CALENDAR_SCOPES_FULL",
]
