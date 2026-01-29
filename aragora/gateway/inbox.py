"""
Inbox Aggregator - Unified inbox across all channels.

Aggregates messages from multiple communication channels (Slack, Teams,
Telegram, WhatsApp, etc.) into a single inbox with threading, priority,
and read/unread tracking.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MessagePriority(Enum):
    """Message priority levels."""

    URGENT = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3


@dataclass
class InboxMessage:
    """A unified inbox message from any channel."""

    message_id: str
    channel: str
    sender: str
    content: str
    timestamp: float = field(default_factory=time.time)
    thread_id: str | None = None
    priority: MessagePriority = MessagePriority.NORMAL
    is_read: bool = False
    is_replied: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    attachments: list[str] = field(default_factory=list)


@dataclass
class InboxThread:
    """A threaded conversation in the inbox."""

    thread_id: str
    channel: str
    subject: str
    messages: list[InboxMessage] = field(default_factory=list)
    last_activity: float = field(default_factory=time.time)
    participants: list[str] = field(default_factory=list)


class InboxAggregator:
    """
    Unified inbox that aggregates messages from all channels.

    Features:
    - Add messages from any channel
    - Get messages with channel/read/priority filters
    - Threading support
    - Read/unread tracking
    - Size limits with oldest-first eviction
    """

    def __init__(self, max_size: int = 10000) -> None:
        self._max_size = max_size
        self._messages: deque[InboxMessage] = deque(maxlen=max_size)
        self._threads: dict[str, InboxThread] = {}
        self._by_id: dict[str, InboxMessage] = {}

    async def add_message(self, message: InboxMessage) -> None:
        """Add a message to the inbox."""
        self._messages.append(message)
        self._by_id[message.message_id] = message

        # Track threading
        if message.thread_id:
            if message.thread_id not in self._threads:
                self._threads[message.thread_id] = InboxThread(
                    thread_id=message.thread_id,
                    channel=message.channel,
                    subject=message.content[:100],
                )
            thread = self._threads[message.thread_id]
            thread.messages.append(message)
            thread.last_activity = message.timestamp
            if message.sender not in thread.participants:
                thread.participants.append(message.sender)

    async def get_messages(
        self,
        channel: str | None = None,
        is_read: bool | None = None,
        priority: MessagePriority | None = None,
        limit: int = 50,
        since: float | None = None,
    ) -> list[InboxMessage]:
        """Get messages with optional filters."""
        results = []
        for msg in reversed(self._messages):
            if channel and msg.channel != channel:
                continue
            if is_read is not None and msg.is_read != is_read:
                continue
            if priority and msg.priority != priority:
                continue
            if since and msg.timestamp < since:
                continue
            results.append(msg)
            if len(results) >= limit:
                break
        return results

    async def get_message(self, message_id: str) -> InboxMessage | None:
        """Get a specific message by ID."""
        return self._by_id.get(message_id)

    async def mark_read(self, message_ids: list[str]) -> int:
        """Mark messages as read. Returns count of messages marked."""
        count = 0
        for msg_id in message_ids:
            msg = self._by_id.get(msg_id)
            if msg and not msg.is_read:
                msg.is_read = True
                count += 1
        return count

    async def mark_replied(self, message_id: str) -> bool:
        """Mark a message as replied."""
        msg = self._by_id.get(message_id)
        if msg:
            msg.is_replied = True
            return True
        return False

    async def get_threads(
        self,
        channel: str | None = None,
        limit: int = 20,
    ) -> list[InboxThread]:
        """Get threads sorted by last activity."""
        threads = list(self._threads.values())
        if channel:
            threads = [t for t in threads if t.channel == channel]
        threads.sort(key=lambda t: t.last_activity, reverse=True)
        return threads[:limit]

    async def get_unread_count(self, channel: str | None = None) -> int:
        """Get count of unread messages."""
        count = 0
        for msg in self._messages:
            if channel and msg.channel != channel:
                continue
            if not msg.is_read:
                count += 1
        return count

    async def get_size(self) -> int:
        """Get current inbox size."""
        return len(self._messages)

    async def clear(self, channel: str | None = None) -> int:
        """Clear messages from the inbox. Returns count removed."""
        if channel is None:
            count = len(self._messages)
            self._messages.clear()
            self._by_id.clear()
            self._threads.clear()
            return count

        to_keep = deque(maxlen=self._max_size)
        removed = 0
        for msg in self._messages:
            if msg.channel == channel:
                self._by_id.pop(msg.message_id, None)
                removed += 1
            else:
                to_keep.append(msg)
        self._messages = to_keep
        return removed
