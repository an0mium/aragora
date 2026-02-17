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

    # ===========================================================================
    # OAuth
    # ===========================================================================

    def get_oauth_url(self) -> dict[str, Any]:
        """Get OAuth authorization URL for Outlook."""
        return self._client.request("GET", "/api/v1/outlook/oauth/url")

    def get_oauth_callback(self) -> dict[str, Any]:
        """Handle OAuth callback."""
        return self._client.request("GET", "/api/v1/outlook/oauth/callback")

    def get_status(self) -> dict[str, Any]:
        """Get Outlook connection status."""
        return self._client.request("GET", "/api/v1/outlook/status")

    # ===========================================================================
    # Messages
    # ===========================================================================

    def list_messages(self) -> dict[str, Any]:
        """List email messages."""
        return self._client.request("GET", "/api/v1/outlook/messages")

    def get_message(self, message_id: str) -> dict[str, Any]:
        """Get a specific email message."""
        return self._client.request("GET", f"/api/v1/outlook/messages/{message_id}")

    def delete_message(self, message_id: str) -> dict[str, Any]:
        """Delete an email message."""
        return self._client.request("DELETE", f"/api/v1/outlook/messages/{message_id}")

    def search_messages(self) -> dict[str, Any]:
        """Search email messages."""
        return self._client.request("GET", "/api/v1/outlook/messages/search")

    def send_message(self, **kwargs: Any) -> dict[str, Any]:
        """Send an email message."""
        return self._client.request("POST", "/api/v1/outlook/messages/send", json=kwargs)

    def reply_to_message(self, **kwargs: Any) -> dict[str, Any]:
        """Reply to an email message."""
        return self._client.request("POST", "/api/v1/outlook/messages/reply", json=kwargs)

    def move_message(self, message_id: str, **kwargs: Any) -> dict[str, Any]:
        """Move a message to a different folder."""
        return self._client.request("POST", f"/api/v1/outlook/messages/{message_id}/move", json=kwargs)

    def mark_read(self, message_id: str, **kwargs: Any) -> dict[str, Any]:
        """Mark a message as read/unread."""
        return self._client.request("PATCH", f"/api/v1/outlook/messages/{message_id}/read", json=kwargs)

    # ===========================================================================
    # Folders & Search
    # ===========================================================================

    def list_folders(self) -> dict[str, Any]:
        """List email folders."""
        return self._client.request("GET", "/api/v1/outlook/folders")

    def search(self) -> dict[str, Any]:
        """Search across Outlook."""
        return self._client.request("GET", "/api/v1/outlook/search")

    # ===========================================================================
    # Conversations
    # ===========================================================================

    def get_conversation(self, conversation_id: str) -> dict[str, Any]:
        """Get an email conversation thread."""
        return self._client.request("GET", f"/api/v1/outlook/conversations/{conversation_id}")

    # ===========================================================================
    # Send / Reply (top-level)
    # ===========================================================================

    def send(self, **kwargs: Any) -> dict[str, Any]:
        """Send an email (top-level endpoint)."""
        return self._client.request("POST", "/api/v1/outlook/send", json=kwargs)

    def reply(self, **kwargs: Any) -> dict[str, Any]:
        """Reply to an email (top-level endpoint)."""
        return self._client.request("POST", "/api/v1/outlook/reply", json=kwargs)


class AsyncOutlookAPI:
    """
    Asynchronous Outlook API.

    Example:
        >>> async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
        ...     status = await client.outlook.get_status()
        ...     emails = await client.outlook.list_messages()
    """

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def get_oauth_url(self) -> dict[str, Any]:
        """Get OAuth authorization URL for Outlook."""
        return await self._client.request("GET", "/api/v1/outlook/oauth/url")

    async def get_oauth_callback(self) -> dict[str, Any]:
        """Handle OAuth callback."""
        return await self._client.request("GET", "/api/v1/outlook/oauth/callback")

    async def get_status(self) -> dict[str, Any]:
        """Get Outlook connection status."""
        return await self._client.request("GET", "/api/v1/outlook/status")

    async def list_messages(self) -> dict[str, Any]:
        """List email messages."""
        return await self._client.request("GET", "/api/v1/outlook/messages")

    async def get_message(self, message_id: str) -> dict[str, Any]:
        """Get a specific email message."""
        return await self._client.request("GET", f"/api/v1/outlook/messages/{message_id}")

    async def delete_message(self, message_id: str) -> dict[str, Any]:
        """Delete an email message."""
        return await self._client.request("DELETE", f"/api/v1/outlook/messages/{message_id}")

    async def search_messages(self) -> dict[str, Any]:
        """Search email messages."""
        return await self._client.request("GET", "/api/v1/outlook/messages/search")

    async def send_message(self, **kwargs: Any) -> dict[str, Any]:
        """Send an email message."""
        return await self._client.request("POST", "/api/v1/outlook/messages/send", json=kwargs)

    async def reply_to_message(self, **kwargs: Any) -> dict[str, Any]:
        """Reply to an email message."""
        return await self._client.request("POST", "/api/v1/outlook/messages/reply", json=kwargs)

    async def move_message(self, message_id: str, **kwargs: Any) -> dict[str, Any]:
        """Move a message to a different folder."""
        return await self._client.request("POST", f"/api/v1/outlook/messages/{message_id}/move", json=kwargs)

    async def mark_read(self, message_id: str, **kwargs: Any) -> dict[str, Any]:
        """Mark a message as read/unread."""
        return await self._client.request("PATCH", f"/api/v1/outlook/messages/{message_id}/read", json=kwargs)

    async def list_folders(self) -> dict[str, Any]:
        """List email folders."""
        return await self._client.request("GET", "/api/v1/outlook/folders")

    async def search(self) -> dict[str, Any]:
        """Search across Outlook."""
        return await self._client.request("GET", "/api/v1/outlook/search")

    async def get_conversation(self, conversation_id: str) -> dict[str, Any]:
        """Get an email conversation thread."""
        return await self._client.request("GET", f"/api/v1/outlook/conversations/{conversation_id}")

    async def send(self, **kwargs: Any) -> dict[str, Any]:
        """Send an email (top-level endpoint)."""
        return await self._client.request("POST", "/api/v1/outlook/send", json=kwargs)

    async def reply(self, **kwargs: Any) -> dict[str, Any]:
        """Reply to an email (top-level endpoint)."""
        return await self._client.request("POST", "/api/v1/outlook/reply", json=kwargs)
