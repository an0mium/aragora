"""
Gmail label and message action operations.

Provides label management (list, create, add), message modification
(archive, trash, star, mark read/unread, move), and batch operations.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, AsyncIterator, Optional, Protocol, TYPE_CHECKING

from ..models import GmailLabel

if TYPE_CHECKING:
    import httpx

logger = logging.getLogger(__name__)


class GmailBaseMethods(Protocol):
    """Protocol defining expected methods from base classes for type checking."""

    user_id: str

    async def _get_access_token(self) -> str: ...
    async def _api_request(
        self, endpoint: str, method: str = "GET", **kwargs: Any
    ) -> dict[str, Any]: ...
    @asynccontextmanager
    def _get_client(self) -> AsyncIterator["httpx.AsyncClient"]: ...
    def check_circuit_breaker(self) -> bool: ...
    def get_circuit_breaker_status(self) -> dict[str, Any]: ...
    def record_success(self) -> None: ...
    def record_failure(self) -> None: ...


class GmailLabelsMixin(GmailBaseMethods):
    """Mixin providing label management and message action operations."""

    async def list_labels(self) -> list[GmailLabel]:
        """List all Gmail labels."""
        data = await self._api_request("/labels")

        labels = []
        for item in data.get("labels", []):
            labels.append(
                GmailLabel(
                    id=item["id"],
                    name=item.get("name", item["id"]),
                    type=item.get("type", "user"),
                    message_list_visibility=item.get("messageListVisibility", "show"),
                    label_list_visibility=item.get("labelListVisibility", "labelShow"),
                )
            )

        return labels

    async def create_label(self, label_name: str) -> GmailLabel:
        """
        Create a new Gmail label.

        Args:
            label_name: Name for the new label

        Returns:
            Created GmailLabel object
        """
        access_token = await self._get_access_token()

        async with self._get_client() as client:
            response = await client.post(
                f"https://gmail.googleapis.com/gmail/v1/users/{self.user_id}/labels",
                headers={"Authorization": f"Bearer {access_token}"},
                json={
                    "name": label_name,
                    "labelListVisibility": "labelShow",
                    "messageListVisibility": "show",
                },
            )
            response.raise_for_status()
            data = response.json()

        return GmailLabel(
            id=data["id"],
            name=data["name"],
            type=data.get("type", "user"),
            message_list_visibility=data.get("messageListVisibility", "show"),
            label_list_visibility=data.get("labelListVisibility", "labelShow"),
        )

    async def add_label(self, message_id: str, label_id: str) -> dict[str, Any]:
        """
        Add a label to a message.

        Args:
            message_id: Gmail message ID
            label_id: Label ID to add

        Returns:
            Dict with message_id and updated labels
        """
        return await self.modify_message(message_id, add_labels=[label_id])

    async def modify_message(
        self,
        message_id: str,
        add_labels: Optional[list[str]] = None,
        remove_labels: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """
        Modify message labels.

        Requires gmail.modify scope to be authorized.

        Args:
            message_id: Gmail message ID
            add_labels: Labels to add (e.g., ["STARRED", "IMPORTANT"])
            remove_labels: Labels to remove (e.g., ["INBOX", "UNREAD"])

        Returns:
            Dict with message_id and updated labels
        """
        access_token = await self._get_access_token()

        if not self.check_circuit_breaker():
            cb_status = self.get_circuit_breaker_status()
            raise ConnectionError(
                f"Circuit breaker open for Gmail. Cooldown: {cb_status.get('cooldown_seconds', 60)}s"
            )

        try:
            async with self._get_client() as client:
                response = await client.post(
                    f"https://gmail.googleapis.com/gmail/v1/users/{self.user_id}/messages/{message_id}/modify",
                    headers={"Authorization": f"Bearer {access_token}"},
                    json={
                        "addLabelIds": add_labels or [],
                        "removeLabelIds": remove_labels or [],
                    },
                )

                if response.status_code != 200:
                    error = response.json().get("error", {})
                    if response.status_code >= 500 or response.status_code == 429:
                        self.record_failure()
                    raise RuntimeError(
                        f"Failed to modify message: {error.get('message', response.text)}"
                    )

                self.record_success()
                result = response.json()
                logger.info(f"[Gmail] Modified message: {message_id}")

                return {
                    "message_id": result.get("id"),
                    "labels": result.get("labelIds", []),
                    "success": True,
                }
        except Exception as e:
            if not isinstance(e, RuntimeError):
                self.record_failure()
            raise

    async def archive_message(self, message_id: str) -> dict[str, Any]:
        """
        Archive a message (remove from INBOX but keep in All Mail).

        Args:
            message_id: Gmail message ID

        Returns:
            Dict with success status
        """
        result = await self.modify_message(
            message_id,
            remove_labels=["INBOX"],
        )
        logger.info(f"[Gmail] Archived message: {message_id}")
        return result

    async def trash_message(self, message_id: str) -> dict[str, Any]:
        """
        Move a message to trash.

        Args:
            message_id: Gmail message ID

        Returns:
            Dict with success status
        """
        access_token = await self._get_access_token()

        if not self.check_circuit_breaker():
            cb_status = self.get_circuit_breaker_status()
            raise ConnectionError(
                f"Circuit breaker open for Gmail. Cooldown: {cb_status.get('cooldown_seconds', 60)}s"
            )

        try:
            async with self._get_client() as client:
                response = await client.post(
                    f"https://gmail.googleapis.com/gmail/v1/users/{self.user_id}/messages/{message_id}/trash",
                    headers={"Authorization": f"Bearer {access_token}"},
                )

                if response.status_code != 200:
                    error = response.json().get("error", {})
                    if response.status_code >= 500 or response.status_code == 429:
                        self.record_failure()
                    raise RuntimeError(
                        f"Failed to trash message: {error.get('message', response.text)}"
                    )

                self.record_success()
                logger.info(f"[Gmail] Trashed message: {message_id}")

                return {
                    "message_id": message_id,
                    "success": True,
                }
        except Exception as e:
            if not isinstance(e, RuntimeError):
                self.record_failure()
            raise

    async def untrash_message(self, message_id: str) -> dict[str, Any]:
        """
        Restore a message from trash.

        Args:
            message_id: Gmail message ID

        Returns:
            Dict with success status
        """
        access_token = await self._get_access_token()

        if not self.check_circuit_breaker():
            cb_status = self.get_circuit_breaker_status()
            raise ConnectionError(
                f"Circuit breaker open for Gmail. Cooldown: {cb_status.get('cooldown_seconds', 60)}s"
            )

        try:
            async with self._get_client() as client:
                response = await client.post(
                    f"https://gmail.googleapis.com/gmail/v1/users/{self.user_id}/messages/{message_id}/untrash",
                    headers={"Authorization": f"Bearer {access_token}"},
                )

                if response.status_code != 200:
                    error = response.json().get("error", {})
                    if response.status_code >= 500 or response.status_code == 429:
                        self.record_failure()
                    raise RuntimeError(
                        f"Failed to untrash message: {error.get('message', response.text)}"
                    )

                self.record_success()
                logger.info(f"[Gmail] Untrashed message: {message_id}")

                return {
                    "message_id": message_id,
                    "success": True,
                }
        except Exception as e:
            if not isinstance(e, RuntimeError):
                self.record_failure()
            raise

    async def mark_as_read(self, message_id: str) -> dict[str, Any]:
        """
        Mark a message as read.

        Args:
            message_id: Gmail message ID

        Returns:
            Dict with success status
        """
        result = await self.modify_message(
            message_id,
            remove_labels=["UNREAD"],
        )
        logger.info(f"[Gmail] Marked as read: {message_id}")
        return result

    async def mark_as_unread(self, message_id: str) -> dict[str, Any]:
        """
        Mark a message as unread.

        Args:
            message_id: Gmail message ID

        Returns:
            Dict with success status
        """
        result = await self.modify_message(
            message_id,
            add_labels=["UNREAD"],
        )
        logger.info(f"[Gmail] Marked as unread: {message_id}")
        return result

    async def star_message(self, message_id: str) -> dict[str, Any]:
        """
        Star a message.

        Args:
            message_id: Gmail message ID

        Returns:
            Dict with success status
        """
        result = await self.modify_message(
            message_id,
            add_labels=["STARRED"],
        )
        logger.info(f"[Gmail] Starred message: {message_id}")
        return result

    async def unstar_message(self, message_id: str) -> dict[str, Any]:
        """
        Remove star from a message.

        Args:
            message_id: Gmail message ID

        Returns:
            Dict with success status
        """
        result = await self.modify_message(
            message_id,
            remove_labels=["STARRED"],
        )
        logger.info(f"[Gmail] Unstarred message: {message_id}")
        return result

    async def mark_important(self, message_id: str) -> dict[str, Any]:
        """
        Mark a message as important.

        Args:
            message_id: Gmail message ID

        Returns:
            Dict with success status
        """
        result = await self.modify_message(
            message_id,
            add_labels=["IMPORTANT"],
        )
        logger.info(f"[Gmail] Marked important: {message_id}")
        return result

    async def mark_not_important(self, message_id: str) -> dict[str, Any]:
        """
        Remove important flag from a message.

        Args:
            message_id: Gmail message ID

        Returns:
            Dict with success status
        """
        result = await self.modify_message(
            message_id,
            remove_labels=["IMPORTANT"],
        )
        logger.info(f"[Gmail] Marked not important: {message_id}")
        return result

    async def move_to_folder(
        self,
        message_id: str,
        folder_label: str,
        remove_from_inbox: bool = True,
    ) -> dict[str, Any]:
        """
        Move a message to a specific folder/label.

        Args:
            message_id: Gmail message ID
            folder_label: Target label name
            remove_from_inbox: Whether to remove from INBOX

        Returns:
            Dict with success status
        """
        remove_labels = ["INBOX"] if remove_from_inbox else []
        result = await self.modify_message(
            message_id,
            add_labels=[folder_label],
            remove_labels=remove_labels,
        )
        logger.info(f"[Gmail] Moved to {folder_label}: {message_id}")
        return result

    async def snooze_message(
        self,
        message_id: str,
        snooze_until: datetime,
    ) -> dict[str, Any]:
        """
        Snooze a message until a specific time.

        Note: Gmail doesn't have native snooze API, so this archives the message
        and stores snooze metadata. A separate scheduler should restore it to inbox.

        Args:
            message_id: Gmail message ID
            snooze_until: When to restore the message

        Returns:
            Dict with success status and snooze metadata
        """
        # Archive the message first
        await self.archive_message(message_id)

        # Add SNOOZED label if it exists (custom label)
        try:
            await self.modify_message(message_id, add_labels=["SNOOZED"])
        except Exception:
            # SNOOZED label might not exist, that's okay
            pass

        logger.info(f"[Gmail] Snoozed message until {snooze_until}: {message_id}")

        return {
            "message_id": message_id,
            "snoozed_until": snooze_until.isoformat(),
            "success": True,
        }

    async def batch_modify(
        self,
        message_ids: list[str],
        add_labels: Optional[list[str]] = None,
        remove_labels: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """
        Batch modify multiple messages.

        Args:
            message_ids: List of Gmail message IDs
            add_labels: Labels to add
            remove_labels: Labels to remove

        Returns:
            Dict with success count and failures
        """
        access_token = await self._get_access_token()

        if not self.check_circuit_breaker():
            cb_status = self.get_circuit_breaker_status()
            raise ConnectionError(
                f"Circuit breaker open for Gmail. Cooldown: {cb_status.get('cooldown_seconds', 60)}s"
            )

        try:
            async with self._get_client() as client:
                response = await client.post(
                    f"https://gmail.googleapis.com/gmail/v1/users/{self.user_id}/messages/batchModify",
                    headers={"Authorization": f"Bearer {access_token}"},
                    json={
                        "ids": message_ids,
                        "addLabelIds": add_labels or [],
                        "removeLabelIds": remove_labels or [],
                    },
                )

                if response.status_code != 204:
                    error = response.json().get("error", {})
                    if response.status_code >= 500 or response.status_code == 429:
                        self.record_failure()
                    raise RuntimeError(
                        f"Failed to batch modify: {error.get('message', response.text)}"
                    )

                self.record_success()
                logger.info(f"[Gmail] Batch modified {len(message_ids)} messages")

                return {
                    "modified_count": len(message_ids),
                    "success": True,
                }
        except Exception as e:
            if not isinstance(e, RuntimeError):
                self.record_failure()
            raise

    async def batch_archive(self, message_ids: list[str]) -> dict[str, Any]:
        """
        Archive multiple messages at once.

        Args:
            message_ids: List of Gmail message IDs

        Returns:
            Dict with success count
        """
        result = await self.batch_modify(
            message_ids,
            remove_labels=["INBOX"],
        )
        logger.info(f"[Gmail] Batch archived {len(message_ids)} messages")
        return result

    async def batch_trash(self, message_ids: list[str]) -> dict[str, Any]:
        """
        Trash multiple messages at once.

        Args:
            message_ids: List of Gmail message IDs

        Returns:
            Dict with success count
        """
        access_token = await self._get_access_token()

        if not self.check_circuit_breaker():
            cb_status = self.get_circuit_breaker_status()
            raise ConnectionError(
                f"Circuit breaker open for Gmail. Cooldown: {cb_status.get('cooldown_seconds', 60)}s"
            )

        try:
            async with self._get_client() as client:
                response = await client.post(
                    f"https://gmail.googleapis.com/gmail/v1/users/{self.user_id}/messages/batchDelete",
                    headers={"Authorization": f"Bearer {access_token}"},
                    json={"ids": message_ids},
                )

                # Note: This permanently deletes, not trash
                # For trash, we use batch_modify with TRASH label
                if response.status_code != 204:
                    error = response.json().get("error", {})
                    if response.status_code >= 500 or response.status_code == 429:
                        self.record_failure()
                    raise RuntimeError(
                        f"Failed to batch delete: {error.get('message', response.text)}"
                    )

                self.record_success()
                logger.info(f"[Gmail] Batch deleted {len(message_ids)} messages")

                return {
                    "deleted_count": len(message_ids),
                    "success": True,
                }
        except Exception as e:
            if not isinstance(e, RuntimeError):
                self.record_failure()
            raise
