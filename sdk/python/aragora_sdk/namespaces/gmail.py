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
        >>> client.gmail.mark_read("message_id")
        >>> client.gmail.add_labels("message_id", ["Important", "Work"])
    """

    def __init__(self, client: AragoraClient):
        self._client = client

    # ===========================================================================
    # Message Operations
    # ===========================================================================

    def add_labels(self, message_id: str, labels: list[str]) -> dict[str, Any]:
        """
        Add labels to a Gmail message.

        Args:
            message_id: Gmail message ID
            labels: List of label names to add

        Returns:
            Updated message with labels
        """
        return self._client.request(
            "POST",
            f"/api/v1/gmail/messages/{message_id}/labels",
            json={"labels": labels},
        )

    def remove_labels(self, message_id: str, labels: list[str]) -> dict[str, Any]:
        """
        Remove labels from a Gmail message.

        Args:
            message_id: Gmail message ID
            labels: List of label names to remove

        Returns:
            Updated message
        """
        return self._client.request(
            "DELETE",
            f"/api/v1/gmail/messages/{message_id}/labels",
            json={"labels": labels},
        )

    def mark_read(self, message_id: str) -> dict[str, Any]:
        """
        Mark a Gmail message as read.

        Args:
            message_id: Gmail message ID

        Returns:
            Updated message status
        """
        return self._client.request("POST", f"/api/v1/gmail/messages/{message_id}/read")

    def mark_unread(self, message_id: str) -> dict[str, Any]:
        """
        Mark a Gmail message as unread.

        Args:
            message_id: Gmail message ID

        Returns:
            Updated message status
        """
        return self._client.request("DELETE", f"/api/v1/gmail/messages/{message_id}/read")

    def star(self, message_id: str) -> dict[str, Any]:
        """
        Star a Gmail message.

        Args:
            message_id: Gmail message ID

        Returns:
            Updated message status
        """
        return self._client.request("POST", f"/api/v1/gmail/messages/{message_id}/star")

    def unstar(self, message_id: str) -> dict[str, Any]:
        """
        Remove star from a Gmail message.

        Args:
            message_id: Gmail message ID

        Returns:
            Updated message status
        """
        return self._client.request("DELETE", f"/api/v1/gmail/messages/{message_id}/star")

    def archive(self, message_id: str) -> dict[str, Any]:
        """
        Archive a Gmail message.

        Args:
            message_id: Gmail message ID

        Returns:
            Archive result
        """
        return self._client.request("POST", f"/api/v1/gmail/messages/{message_id}/archive")

    def trash(self, message_id: str) -> dict[str, Any]:
        """
        Move a Gmail message to trash.

        Args:
            message_id: Gmail message ID

        Returns:
            Trash result
        """
        return self._client.request("POST", f"/api/v1/gmail/messages/{message_id}/trash")

    # ===========================================================================
    # Attachments
    # ===========================================================================

    def get_attachment(self, message_id: str, attachment_id: str) -> dict[str, Any]:
        """
        Get an attachment from a Gmail message.

        Args:
            message_id: Gmail message ID
            attachment_id: Attachment ID

        Returns:
            Attachment data with filename, mimeType, and data (base64)
        """
        return self._client.request(
            "GET", f"/api/v1/gmail/messages/{message_id}/attachments/{attachment_id}"
        )


class AsyncGmailAPI:
    """
    Asynchronous Gmail API.

    Example:
        >>> async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
        ...     await client.gmail.mark_read("message_id")
    """

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    # Message Operations
    async def add_labels(self, message_id: str, labels: list[str]) -> dict[str, Any]:
        """Add labels to a Gmail message."""
        return await self._client.request(
            "POST",
            f"/api/v1/gmail/messages/{message_id}/labels",
            json={"labels": labels},
        )

    async def remove_labels(self, message_id: str, labels: list[str]) -> dict[str, Any]:
        """Remove labels from a Gmail message."""
        return await self._client.request(
            "DELETE",
            f"/api/v1/gmail/messages/{message_id}/labels",
            json={"labels": labels},
        )

    async def mark_read(self, message_id: str) -> dict[str, Any]:
        """Mark a Gmail message as read."""
        return await self._client.request("POST", f"/api/v1/gmail/messages/{message_id}/read")

    async def mark_unread(self, message_id: str) -> dict[str, Any]:
        """Mark a Gmail message as unread."""
        return await self._client.request("DELETE", f"/api/v1/gmail/messages/{message_id}/read")

    async def star(self, message_id: str) -> dict[str, Any]:
        """Star a Gmail message."""
        return await self._client.request("POST", f"/api/v1/gmail/messages/{message_id}/star")

    async def unstar(self, message_id: str) -> dict[str, Any]:
        """Remove star from a Gmail message."""
        return await self._client.request("DELETE", f"/api/v1/gmail/messages/{message_id}/star")

    async def archive(self, message_id: str) -> dict[str, Any]:
        """Archive a Gmail message."""
        return await self._client.request("POST", f"/api/v1/gmail/messages/{message_id}/archive")

    async def trash(self, message_id: str) -> dict[str, Any]:
        """Move a Gmail message to trash."""
        return await self._client.request("POST", f"/api/v1/gmail/messages/{message_id}/trash")

    # Attachments
    async def get_attachment(self, message_id: str, attachment_id: str) -> dict[str, Any]:
        """Get an attachment from a Gmail message."""
        return await self._client.request(
            "GET", f"/api/v1/gmail/messages/{message_id}/attachments/{attachment_id}"
        )
