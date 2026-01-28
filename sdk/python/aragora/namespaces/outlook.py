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

    def get_status(self) -> dict[str, Any]:
        """
        Get Outlook connection status.

        Returns:
            Dict with:
            - connected: Whether connected
            - email: Connected email address
            - scopes: Granted scopes
            - expires_at: Token expiration
        """
        return self._client.request("GET", "/api/v1/outlook/status")

    def get_oauth_url(
        self,
        redirect_uri: str,
        state: str | None = None,
        scopes: list[OutlookScopes] | None = None,
    ) -> dict[str, Any]:
        """
        Get Outlook OAuth authorization URL.

        Args:
            redirect_uri: OAuth callback URL
            state: Optional state parameter
            scopes: Requested scopes

        Returns:
            Dict with oauth_url
        """
        data: dict[str, Any] = {"redirect_uri": redirect_uri}
        if state:
            data["state"] = state
        if scopes:
            data["scopes"] = scopes
        return self._client.request("POST", "/api/v1/outlook/oauth/url", json=data)

    def handle_callback(
        self,
        code: str,
        redirect_uri: str,
    ) -> dict[str, Any]:
        """
        Handle Outlook OAuth callback.

        Args:
            code: Authorization code
            redirect_uri: Original redirect URI

        Returns:
            Dict with connection status
        """
        return self._client.request(
            "POST",
            "/api/v1/outlook/oauth/callback",
            json={"code": code, "redirect_uri": redirect_uri},
        )

    def disconnect(self) -> dict[str, Any]:
        """
        Disconnect Outlook account.

        Returns:
            Dict with success status
        """
        return self._client.request("POST", "/api/v1/outlook/disconnect")

    def fetch_emails(
        self,
        folder: str = "inbox",
        limit: int | None = None,
        offset: int | None = None,
        unread_only: bool | None = None,
    ) -> dict[str, Any]:
        """
        Fetch emails from Outlook.

        Args:
            folder: Mail folder (inbox, sent, etc.)
            limit: Maximum emails
            offset: Skip count
            unread_only: Only unread emails

        Returns:
            Dict with emails list
        """
        params: dict[str, Any] = {"folder": folder}
        if limit:
            params["limit"] = limit
        if offset:
            params["offset"] = offset
        if unread_only is not None:
            params["unread_only"] = unread_only
        return self._client.request("GET", "/api/v1/outlook/emails", params=params)

    def get_email(self, email_id: str) -> dict[str, Any]:
        """
        Get a specific email.

        Args:
            email_id: The email ID

        Returns:
            Dict with email details
        """
        return self._client.request("GET", f"/api/v1/outlook/emails/{email_id}")

    def fetch_calendar(
        self,
        start: str | None = None,
        end: str | None = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """
        Fetch calendar events.

        Args:
            start: Start date (ISO 8601)
            end: End date (ISO 8601)
            limit: Maximum events

        Returns:
            Dict with events list
        """
        params: dict[str, Any] = {}
        if start:
            params["start"] = start
        if end:
            params["end"] = end
        if limit:
            params["limit"] = limit
        return self._client.request(
            "GET", "/api/v1/outlook/calendar", params=params if params else None
        )

    def sync(self) -> dict[str, Any]:
        """
        Trigger a sync with Outlook.

        Returns:
            Dict with sync status
        """
        return self._client.request("POST", "/api/v1/outlook/sync")


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

    async def get_status(self) -> dict[str, Any]:
        """Get Outlook connection status."""
        return await self._client.request("GET", "/api/v1/outlook/status")

    async def get_oauth_url(
        self,
        redirect_uri: str,
        state: str | None = None,
        scopes: list[OutlookScopes] | None = None,
    ) -> dict[str, Any]:
        """Get Outlook OAuth authorization URL."""
        data: dict[str, Any] = {"redirect_uri": redirect_uri}
        if state:
            data["state"] = state
        if scopes:
            data["scopes"] = scopes
        return await self._client.request("POST", "/api/v1/outlook/oauth/url", json=data)

    async def handle_callback(
        self,
        code: str,
        redirect_uri: str,
    ) -> dict[str, Any]:
        """Handle Outlook OAuth callback."""
        return await self._client.request(
            "POST",
            "/api/v1/outlook/oauth/callback",
            json={"code": code, "redirect_uri": redirect_uri},
        )

    async def disconnect(self) -> dict[str, Any]:
        """Disconnect Outlook account."""
        return await self._client.request("POST", "/api/v1/outlook/disconnect")

    async def fetch_emails(
        self,
        folder: str = "inbox",
        limit: int | None = None,
        offset: int | None = None,
        unread_only: bool | None = None,
    ) -> dict[str, Any]:
        """Fetch emails from Outlook."""
        params: dict[str, Any] = {"folder": folder}
        if limit:
            params["limit"] = limit
        if offset:
            params["offset"] = offset
        if unread_only is not None:
            params["unread_only"] = unread_only
        return await self._client.request("GET", "/api/v1/outlook/emails", params=params)

    async def get_email(self, email_id: str) -> dict[str, Any]:
        """Get a specific email."""
        return await self._client.request("GET", f"/api/v1/outlook/emails/{email_id}")

    async def fetch_calendar(
        self,
        start: str | None = None,
        end: str | None = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """Fetch calendar events."""
        params: dict[str, Any] = {}
        if start:
            params["start"] = start
        if end:
            params["end"] = end
        if limit:
            params["limit"] = limit
        return await self._client.request(
            "GET", "/api/v1/outlook/calendar", params=params if params else None
        )

    async def sync(self) -> dict[str, Any]:
        """Trigger a sync with Outlook."""
        return await self._client.request("POST", "/api/v1/outlook/sync")
