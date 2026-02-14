"""
Gmail Namespace API

Provides methods for Gmail integration:
- Message operations (labels, read status, archiving)
- Attachment handling
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient

class GmailAPI:
    """
    Synchronous Gmail API.

    Provides methods for managing Gmail messages through the Aragora platform.

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai")
        >>> client.gmail.list_labels()
        >>> client.gmail.list_threads()
    """

    def __init__(self, client: AragoraClient):
        self._client = client

    # ===========================================================================
    # Labels and Filters
    # ===========================================================================

    def list_labels(self) -> dict[str, Any]:
        """
        List Gmail labels.

        Returns:
            Dict with labels array
        """
        return self._client.request("GET", "/api/v1/gmail/labels")

    def list_filters(self) -> dict[str, Any]:
        """
        List Gmail filters.

        Returns:
            Dict with filters array
        """
        return self._client.request("GET", "/api/v1/gmail/filters")

    # ===========================================================================
    # Threads and Drafts
    # ===========================================================================

    def list_threads(self, **kwargs: Any) -> dict[str, Any]:
        """
        List Gmail threads.

        Returns:
            Dict with threads array
        """
        return self._client.request("GET", "/api/v1/gmail/threads", params=kwargs)

    def list_drafts(self, **kwargs: Any) -> dict[str, Any]:
        """
        List Gmail drafts.

        Returns:
            Dict with drafts array
        """
        return self._client.request("GET", "/api/v1/gmail/drafts", params=kwargs)

    def archive_message(self, message_id: str) -> dict[str, Any]:
        """Archive a message."""
        return self._client.request("POST", f"/api/v1/gmail/messages/{message_id}/archive")

    def get_attachment(self, message_id: str, attachment_id: str) -> dict[str, Any]:
        """Get an attachment."""
        return self._client.request("GET", f"/api/v1/gmail/messages/{message_id}/attachments/{attachment_id}")

    def update_labels(self, message_id: str) -> dict[str, Any]:
        """Update message labels."""
        return self._client.request("POST", f"/api/v1/gmail/messages/{message_id}/labels")

    def mark_read(self, message_id: str) -> dict[str, Any]:
        """Mark message as read."""
        return self._client.request("POST", f"/api/v1/gmail/messages/{message_id}/read")

    def star_message(self, message_id: str) -> dict[str, Any]:
        """Star a message."""
        return self._client.request("POST", f"/api/v1/gmail/messages/{message_id}/star")

    def trash_message(self, message_id: str) -> dict[str, Any]:
        """Move message to trash."""
        return self._client.request("POST", f"/api/v1/gmail/messages/{message_id}/trash")


class AsyncGmailAPI:
    """
    Asynchronous Gmail API.

    Example:
        >>> async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
        ...     labels = await client.gmail.list_labels()
    """

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    # Labels and Filters

    async def list_labels(self) -> dict[str, Any]:
        """List Gmail labels."""
        return await self._client.request("GET", "/api/v1/gmail/labels")

    async def list_filters(self) -> dict[str, Any]:
        """List Gmail filters."""
        return await self._client.request("GET", "/api/v1/gmail/filters")

    # Threads and Drafts

    async def list_threads(self, **kwargs: Any) -> dict[str, Any]:
        """List Gmail threads."""
        return await self._client.request("GET", "/api/v1/gmail/threads", params=kwargs)

    async def list_drafts(self, **kwargs: Any) -> dict[str, Any]:
        """List Gmail drafts."""
        return await self._client.request("GET", "/api/v1/gmail/drafts", params=kwargs)

    async def archive_message(self, message_id: str) -> dict[str, Any]:
        """Archive a message."""
        return await self._client.request("POST", f"/api/v1/gmail/messages/{message_id}/archive")

    async def get_attachment(self, message_id: str, attachment_id: str) -> dict[str, Any]:
        """Get an attachment."""
        return await self._client.request("GET", f"/api/v1/gmail/messages/{message_id}/attachments/{attachment_id}")

    async def update_labels(self, message_id: str) -> dict[str, Any]:
        """Update message labels."""
        return await self._client.request("POST", f"/api/v1/gmail/messages/{message_id}/labels")

    async def mark_read(self, message_id: str) -> dict[str, Any]:
        """Mark message as read."""
        return await self._client.request("POST", f"/api/v1/gmail/messages/{message_id}/read")

    async def star_message(self, message_id: str) -> dict[str, Any]:
        """Star a message."""
        return await self._client.request("POST", f"/api/v1/gmail/messages/{message_id}/star")

    async def trash_message(self, message_id: str) -> dict[str, Any]:
        """Move message to trash."""
        return await self._client.request("POST", f"/api/v1/gmail/messages/{message_id}/trash")
