"""
Calendar Connectors.

Provides calendar integration for meeting detection and availability checking:
- Google Calendar (via Google Calendar API)
- Outlook Calendar (via Microsoft Graph)
"""

from .google_calendar import GoogleCalendarConnector
from .outlook_calendar import OutlookCalendarConnector

__all__ = [
    "GoogleCalendarConnector",
    "OutlookCalendarConnector",
]
