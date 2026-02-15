"""
Slack thread management: thread info, replies, participants, stats.

Contains the SlackThreadManager class for Slack-specific thread operations.
"""

from __future__ import annotations

import importlib.util
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

logger = logging.getLogger(__name__)

_MAX_PAGES = 1000  # Safety cap for pagination loops

HTTPX_AVAILABLE = importlib.util.find_spec("httpx") is not None

if TYPE_CHECKING:
    from aragora.connectors.chat.thread_manager import ThreadInfo, ThreadParticipant, ThreadStats
    from . import SlackConnector

from aragora.connectors.chat.models import (
    ChatChannel,
    ChatMessage,
    ChatUser,
)


class SlackThreadManager:
    """
    Slack-specific thread management using thread_ts.

    Slack's threading model uses the timestamp of the parent message
    (thread_ts) as the thread identifier.
    """

    def __init__(self, connector: SlackConnector):
        self.connector = connector

    @property
    def platform_name(self) -> str:
        return "slack"

    async def get_thread(self, thread_ts: str, channel_id: str) -> ThreadInfo:
        """Get thread information using conversations.replies."""
        from aragora.connectors.chat.thread_manager import ThreadInfo, ThreadNotFoundError

        if not HTTPX_AVAILABLE:
            raise RuntimeError("httpx required for thread operations")

        success, data, error = await self.connector._slack_api_request(
            "conversations.replies",
            operation="get_thread",
            method="GET",
            params={"channel": channel_id, "ts": thread_ts, "limit": 1, "inclusive": True},
        )
        if not success or not data:
            error_str = error or "unknown"
            if error_str in ("thread_not_found", "message_not_found"):
                raise ThreadNotFoundError(thread_ts, channel_id, "slack")
            raise Exception(f"Slack API error: {error_str}")

        messages = data.get("messages", [])
        if not messages:
            raise ThreadNotFoundError(thread_ts, channel_id, "slack")

        root_msg = messages[0]
        reply_count = int(root_msg.get("reply_count", 0))
        reply_users_count = len(root_msg.get("reply_users", []))
        created_ts = float(thread_ts.split(".")[0])
        created_at = datetime.utcfromtimestamp(created_ts)
        latest_reply = root_msg.get("latest_reply")
        updated_at = (
            datetime.utcfromtimestamp(float(latest_reply.split(".")[0]))
            if latest_reply
            else created_at
        )

        return ThreadInfo(
            id=thread_ts,
            channel_id=channel_id,
            platform="slack",
            created_by=root_msg.get("user", ""),
            created_at=created_at,
            updated_at=updated_at,
            message_count=reply_count + 1,
            participant_count=reply_users_count + 1,
            title=root_msg.get("text", "")[:100],
            root_message_id=thread_ts,
        )

    async def get_thread_messages(
        self, thread_ts: str, channel_id: str, limit: int = 50, cursor: str | None = None
    ) -> tuple[list[ChatMessage], str | None]:
        """Get all messages in a thread with pagination."""
        if not HTTPX_AVAILABLE:
            return [], None

        params: dict[str, Any] = {
            "channel": channel_id,
            "ts": thread_ts,
            "limit": min(limit, 1000),
            "inclusive": True,
        }
        if cursor:
            params["cursor"] = cursor

        success, data, _ = await self.connector._slack_api_request(
            "conversations.replies",
            operation="get_thread_messages",
            method="GET",
            params=params,
        )
        if not success or not data:
            return [], None

        messages = []
        for msg in data.get("messages", []):
            channel = ChatChannel(id=channel_id, platform="slack")
            user = ChatUser(
                id=msg.get("user", msg.get("bot_id", "")),
                platform="slack",
                is_bot="bot_id" in msg,
            )
            messages.append(
                ChatMessage(
                    id=msg.get("ts", ""),
                    platform="slack",
                    channel=channel,
                    author=user,
                    content=msg.get("text", ""),
                    thread_id=thread_ts,
                    timestamp=datetime.fromtimestamp(float(msg.get("ts", "0").split(".")[0])),
                )
            )

        next_cursor = data.get("response_metadata", {}).get("next_cursor")
        return messages, next_cursor if next_cursor else None

    async def list_threads(
        self, channel_id: str, limit: int = 20, cursor: str | None = None
    ) -> tuple[list[ThreadInfo], str | None]:
        """List threads in a channel."""
        from aragora.connectors.chat.thread_manager import ThreadInfo

        if not HTTPX_AVAILABLE:
            return [], None

        threads: list[ThreadInfo] = []
        params: dict[str, Any] = {"channel": channel_id, "limit": min(limit * 5, 200)}
        if cursor:
            params["cursor"] = cursor

        success, data, _ = await self.connector._slack_api_request(
            "conversations.history",
            operation="list_threads",
            method="GET",
            params=params,
        )
        if not success or not data:
            return [], None

        for msg in data.get("messages", []):
            reply_count = int(msg.get("reply_count", 0))
            if reply_count == 0:
                continue
            ts = msg.get("ts", "")
            created_ts = float(ts.split(".")[0]) if ts else 0
            created_at = datetime.utcfromtimestamp(created_ts)
            latest_reply = msg.get("latest_reply")
            updated_at = (
                datetime.utcfromtimestamp(float(latest_reply.split(".")[0]))
                if latest_reply
                else created_at
            )

            threads.append(
                ThreadInfo(
                    id=ts,
                    channel_id=channel_id,
                    platform="slack",
                    created_by=msg.get("user", ""),
                    created_at=created_at,
                    updated_at=updated_at,
                    message_count=reply_count + 1,
                    participant_count=len(msg.get("reply_users", [])) + 1,
                    title=msg.get("text", "")[:100],
                    root_message_id=ts,
                )
            )
            if len(threads) >= limit:
                break

        next_cursor = data.get("response_metadata", {}).get("next_cursor")
        return threads, next_cursor if next_cursor else None

    async def reply_to_thread(
        self, thread_ts: str, channel_id: str, message: str, **kwargs: Any
    ) -> ChatMessage:
        """Reply to an existing thread."""
        response = await self.connector.send_message(
            channel_id=channel_id, text=message, thread_id=thread_ts, blocks=kwargs.get("blocks")
        )
        channel = ChatChannel(id=channel_id, platform="slack")
        # Bot messages don't include author_id in response; use empty string for bot user
        user = ChatUser(id="", platform="slack", is_bot=True)
        return ChatMessage(
            id=response.message_id,
            platform="slack",
            channel=channel,
            author=user,
            content=message,
            thread_id=thread_ts,
            timestamp=datetime.now(timezone.utc),
        )

    async def broadcast_thread_reply(
        self, thread_ts: str, channel_id: str, message: str, **kwargs: Any
    ) -> ChatMessage:
        """Reply to thread and broadcast to channel."""
        if not HTTPX_AVAILABLE:
            raise RuntimeError("httpx required for broadcast reply")

        payload: dict[str, Any] = {
            "channel": channel_id,
            "text": message,
            "thread_ts": thread_ts,
            "reply_broadcast": True,
        }
        if kwargs.get("blocks"):
            payload["blocks"] = kwargs["blocks"]

        success, data, error = await self.connector._slack_api_request(
            "chat.postMessage",
            operation="broadcast_thread_reply",
            json_data=payload,
        )
        if not success or not data:
            raise Exception(f"Slack broadcast error: {error}")

        msg_data = data.get("message", {})
        channel = ChatChannel(id=channel_id, platform="slack")
        user = ChatUser(
            id=msg_data.get("user", msg_data.get("bot_id", "")), platform="slack", is_bot=True
        )
        return ChatMessage(
            id=data.get("ts", ""),
            platform="slack",
            channel=channel,
            author=user,
            content=message,
            thread_id=thread_ts,
            timestamp=datetime.now(timezone.utc),
        )

    async def get_thread_stats(self, thread_ts: str, channel_id: str) -> ThreadStats:
        """Get statistics for a thread."""
        from aragora.connectors.chat.thread_manager import ThreadStats

        all_messages: list[ChatMessage] = []
        cursor: str | None = None
        for _page in range(_MAX_PAGES):
            messages, next_cursor = await self.get_thread_messages(
                thread_ts, channel_id, limit=200, cursor=cursor
            )
            all_messages.extend(messages)
            if not next_cursor:
                break
            cursor = next_cursor
        else:
            logger.warning(
                f"Pagination safety cap reached for thread stats "
                f"(thread_ts={thread_ts}), stopping with {len(all_messages)} messages"
            )

        if not all_messages:
            return ThreadStats(
                thread_id=thread_ts,
                message_count=0,
                participant_count=0,
                last_activity=datetime.now(timezone.utc),
            )

        participants = list(set(m.author.id for m in all_messages))
        last_activity = max(m.timestamp for m in all_messages)
        return ThreadStats(
            thread_id=thread_ts,
            message_count=len(all_messages),
            participant_count=len(participants),
            last_activity=last_activity,
        )

    async def get_thread_participants(
        self, thread_ts: str, channel_id: str
    ) -> list[ThreadParticipant]:
        """Get participants in a thread."""
        from aragora.connectors.chat.thread_manager import ThreadParticipant

        all_messages: list[ChatMessage] = []
        cursor: str | None = None
        for _page in range(_MAX_PAGES):
            messages, next_cursor = await self.get_thread_messages(
                thread_ts, channel_id, limit=200, cursor=cursor
            )
            all_messages.extend(messages)
            if not next_cursor or len(all_messages) > 1000:
                break
            cursor = next_cursor
        else:
            logger.warning(
                f"Pagination safety cap reached for thread participants "
                f"(thread_ts={thread_ts}), stopping with {len(all_messages)} messages"
            )

        participant_data: dict[str, dict[str, Any]] = {}
        for msg in all_messages:
            user_id = msg.author.id
            if user_id not in participant_data:
                participant_data[user_id] = {
                    "user_id": user_id,
                    "message_count": 0,
                    "first_message_at": msg.timestamp,
                    "last_message_at": msg.timestamp,
                    "is_bot": msg.author.is_bot,
                }
            participant_data[user_id]["message_count"] += 1
            if msg.timestamp < participant_data[user_id]["first_message_at"]:
                participant_data[user_id]["first_message_at"] = msg.timestamp
            if msg.timestamp > participant_data[user_id]["last_message_at"]:
                participant_data[user_id]["last_message_at"] = msg.timestamp

        return [
            ThreadParticipant(
                user_id=p["user_id"],
                message_count=p["message_count"],
                first_message_at=p["first_message_at"],
                last_message_at=p["last_message_at"],
                is_bot=p["is_bot"],
            )
            for p in participant_data.values()
        ]
