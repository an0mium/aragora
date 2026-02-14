"""
Outlook Namespace API

Provides Microsoft Outlook integration:
- Connect Outlook accounts via OAuth
- Fetch and sync emails
- Calendar integration
- Contact sync

Features:
- Microsoft Graph API integration
- Email sync and prioritization
- Calendar event access
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient

OutlookScopes = Literal["mail.read", "mail.readwrite", "calendars.read", "contacts.read"]

class OutlookAPI:
    """
    Synchronous Outlook API.

    Provides methods for Microsoft Outlook integration:
    - Connect Outlook accounts via OAuth
    - Fetch and sync emails
    - Calendar integration

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai", api_key="...")
        >>> status = client.outlook.get_status()
        >>> if not status["connected"]:
        ...     auth_url = client.outlook.get_oauth_url(redirect_uri="...")
    """

    def __init__(self, client: AragoraClient):
        self._client = client

class AsyncOutlookAPI:
    """
    Asynchronous Outlook API.

    Example:
        >>> async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
        ...     status = await client.outlook.get_status()
        ...     emails = await client.outlook.fetch_emails(limit=10)
    """

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

