"""
Microsoft Teams thread management.

Provides the TeamsThreadManager class for managing threaded
conversations in Teams channels using the Graph API.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING

from aragora.connectors.chat.models import (
    ChatChannel,
    ChatMessage,
    ChatUser,
)

if TYPE_CHECKING:
    from aragora.connectors.chat.thread_manager import ThreadInfo, ThreadStats
    from aragora.connectors.chat.teams.connector import TeamsConnector

logger = logging.getLogger(__name__)


class TeamsThreadManager:
    """
    Microsoft Teams thread management using Graph API.

    Teams threads are message-based - replies reference a parent message ID.
    Requires team_id for all Graph API operations on channels.
    """

    def __init__(self, connector: TeamsConnector, team_id: str):
        """
        Initialize Teams thread manager.

        Args:
            connector: TeamsConnector instance for API calls
            team_id: Team ID (required for Graph API channel operations)
        """
        self.connector = connector
        self.team_id = team_id

    @property
    def platform_name(self) -> str:
        """Return platform identifier."""
        return "teams"

    async def get_thread(
        self,
        thread_id: str,
        channel_id: str,
    ) -> "ThreadInfo":
        """
        Get thread metadata for a Teams message thread.

        Args:
            thread_id: Message ID of the thread root
            channel_id: Channel ID

        Returns:
            ThreadInfo with thread metadata

        Raises:
            ThreadNotFoundError: If thread doesn't exist
        """
        from aragora.connectors.chat.thread_manager import ThreadInfo, ThreadNotFoundError

        endpoint = f"/teams/{self.team_id}/channels/{channel_id}/messages/{thread_id}"
        success, data, error = await self.connector._graph_api_request(
            endpoint=endpoint,
            method="GET",
            operation="get_thread",
        )

        if not success or not data:
            raise ThreadNotFoundError(
                thread_id=thread_id,
                channel_id=channel_id,
                platform="teams",
            )

        # Get reply count
        replies_endpoint = f"{endpoint}/replies"
        _, replies_data, _ = await self.connector._graph_api_request(
            endpoint=replies_endpoint,
            method="GET",
            operation="get_thread_replies",
        )

        reply_count = len(replies_data.get("value", [])) if replies_data else 0
        participants = set()

        if replies_data and replies_data.get("value"):
            for reply in replies_data["value"]:
                if reply.get("from", {}).get("user", {}).get("id"):
                    participants.add(reply["from"]["user"]["id"])

        # Add original author
        if data.get("from", {}).get("user", {}).get("id"):
            participants.add(data["from"]["user"]["id"])

        created_at = data.get("createdDateTime", "")
        last_modified = data.get("lastModifiedDateTime", created_at)

        return ThreadInfo(
            id=thread_id,
            channel_id=channel_id,
            platform="teams",
            created_by=data.get("from", {}).get("user", {}).get("id", "unknown"),
            created_at=datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            if created_at
            else datetime.now(),
            updated_at=datetime.fromisoformat(last_modified.replace("Z", "+00:00"))
            if last_modified
            else datetime.now(),
            message_count=reply_count + 1,  # Include root message
            participant_count=len(participants),
            title=data.get("subject"),
            metadata={
                "team_id": self.team_id,
                "importance": data.get("importance"),
                "message_type": data.get("messageType"),
            },
        )

    async def get_thread_messages(
        self,
        thread_id: str,
        channel_id: str,
        limit: int = 50,
        cursor: str | None = None,
    ) -> tuple[list[ChatMessage], str | None]:
        """
        Get messages in a thread with pagination.

        Args:
            thread_id: Message ID of the thread root
            channel_id: Channel ID
            limit: Maximum messages to retrieve
            cursor: Pagination cursor (nextLink URL)

        Returns:
            Tuple of (messages list, next cursor or None)
        """
        if cursor:
            # Use provided nextLink for pagination
            success, data, error = await self.connector._graph_api_request(
                endpoint=cursor,
                method="GET",
                operation="get_thread_messages_page",
                use_full_url=True,
            )
        else:
            endpoint = f"/teams/{self.team_id}/channels/{channel_id}/messages/{thread_id}/replies"
            params = {"$top": str(limit)}
            success, data, error = await self.connector._graph_api_request(
                endpoint=endpoint,
                method="GET",
                operation="get_thread_messages",
                params=params,
            )

        if not success or not data:
            return [], None

        messages = []
        for msg in data.get("value", []):
            user_data = msg.get("from", {}).get("user", {})
            messages.append(
                ChatMessage(
                    id=msg.get("id", ""),
                    platform="teams",
                    channel=ChatChannel(
                        id=channel_id,
                        platform="teams",
                        name=channel_id,
                    ),
                    author=ChatUser(
                        id=user_data.get("id", "unknown"),
                        platform="teams",
                        display_name=user_data.get("displayName"),
                    ),
                    content=msg.get("body", {}).get("content", ""),
                    thread_id=thread_id,
                    timestamp=datetime.fromisoformat(
                        msg.get("createdDateTime", "").replace("Z", "+00:00")
                    )
                    if msg.get("createdDateTime")
                    else datetime.now(),
                    metadata={
                        "importance": msg.get("importance"),
                        "content_type": msg.get("body", {}).get("contentType"),
                    },
                )
            )

        next_cursor = data.get("@odata.nextLink")
        return messages, next_cursor

    async def list_threads(
        self,
        channel_id: str,
        limit: int = 20,
    ) -> list["ThreadInfo"]:
        """
        List recent threads (root messages) in a channel.

        Args:
            channel_id: Channel ID
            limit: Maximum threads to retrieve

        Returns:
            List of ThreadInfo objects for threads with replies
        """
        from aragora.connectors.chat.thread_manager import ThreadInfo

        endpoint = f"/teams/{self.team_id}/channels/{channel_id}/messages"
        params = {"$top": str(limit)}

        success, data, error = await self.connector._graph_api_request(
            endpoint=endpoint,
            method="GET",
            operation="list_threads",
            params=params,
        )

        if not success or not data:
            return []

        threads = []
        for msg in data.get("value", []):
            # Only include messages that have replies (are threads)
            # Teams doesn't directly expose reply count in list, so we include all root messages
            created_at = msg.get("createdDateTime", "")
            last_modified = msg.get("lastModifiedDateTime", created_at)

            threads.append(
                ThreadInfo(
                    id=msg.get("id", ""),
                    channel_id=channel_id,
                    platform="teams",
                    created_by=msg.get("from", {}).get("user", {}).get("id", "unknown"),
                    created_at=datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                    if created_at
                    else datetime.now(),
                    updated_at=datetime.fromisoformat(last_modified.replace("Z", "+00:00"))
                    if last_modified
                    else datetime.now(),
                    message_count=1,  # Would need separate call for accurate count
                    participant_count=1,
                    title=msg.get("subject"),
                    metadata={
                        "team_id": self.team_id,
                        "importance": msg.get("importance"),
                    },
                )
            )

        return threads

    async def reply_to_thread(
        self,
        thread_id: str,
        channel_id: str,
        message: str,
    ) -> ChatMessage:
        """
        Reply to an existing thread.

        Args:
            thread_id: Message ID of the thread root
            channel_id: Channel ID
            message: Reply message content

        Returns:
            ChatMessage representing the sent reply
        """
        endpoint = f"/teams/{self.team_id}/channels/{channel_id}/messages/{thread_id}/replies"

        success, data, error = await self.connector._graph_api_request(
            endpoint=endpoint,
            method="POST",
            operation="reply_to_thread",
            json_data={
                "body": {
                    "contentType": "text",
                    "content": message,
                }
            },
        )

        if not success or not data:
            raise RuntimeError(f"Failed to reply to thread: {error}")

        user_data = data.get("from", {}).get("user", {})
        return ChatMessage(
            id=data.get("id", ""),
            platform="teams",
            channel=ChatChannel(
                id=channel_id,
                platform="teams",
                name=channel_id,
            ),
            author=ChatUser(
                id=user_data.get("id", "bot"),
                platform="teams",
                display_name=user_data.get("displayName", "Bot"),
            ),
            content=message,
            thread_id=thread_id,
            timestamp=datetime.now(),
        )

    async def get_thread_stats(
        self,
        thread_id: str,
        channel_id: str,
    ) -> "ThreadStats":
        """
        Get statistics for a thread.

        Args:
            thread_id: Message ID of the thread root
            channel_id: Channel ID

        Returns:
            ThreadStats with thread metrics
        """
        from aragora.connectors.chat.thread_manager import ThreadStats

        # Get thread info which includes counts
        thread_info = await self.get_thread(thread_id, channel_id)

        return ThreadStats(
            thread_id=thread_id,
            message_count=thread_info.message_count,
            participant_count=thread_info.participant_count,
            last_activity=thread_info.updated_at,
        )

    async def get_thread_participants(
        self,
        thread_id: str,
        channel_id: str,
    ) -> list[str]:
        """
        Get list of user IDs who participated in a thread.

        Args:
            thread_id: Message ID of the thread root
            channel_id: Channel ID

        Returns:
            List of user IDs
        """
        messages, _ = await self.get_thread_messages(thread_id, channel_id, limit=100)
        participants = set()

        for msg in messages:
            if msg.author and msg.author.id:
                participants.add(msg.author.id)

        return list(participants)
