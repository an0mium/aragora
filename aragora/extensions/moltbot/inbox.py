"""
Inbox Manager - Unified Multi-Channel Message Aggregation.

Manages messages across multiple communication channels with
unified threading, intent detection, and response tracking.
"""

from __future__ import annotations

import asyncio
import os
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, cast
from collections.abc import Callable

from aragora.gateway.inbox import (
    InboxAggregator as GatewayInbox,
    InboxMessage as GatewayInboxMessage,
    MessagePriority,
)

from .models import (
    Channel,
    ChannelConfig,
    ChannelType,
    InboxMessage,
    InboxMessageStatus,
)

logger = logging.getLogger(__name__)


class InboxManager:
    """
    Unified inbox for multi-channel message management.

    Aggregates messages from multiple channels (SMS, email, WhatsApp,
    Telegram, etc.) into a single inbox with threading and analytics.
    """

    def __init__(
        self,
        storage_path: str | Path | None = None,
        gateway_inbox: GatewayInbox | None = None,
        gateway_max_size: int | None = None,
    ) -> None:
        """
        Initialize the inbox manager.

        Args:
            storage_path: Path for message storage
        """
        self._storage_path = Path(storage_path) if storage_path else None
        self._channels: dict[str, Channel] = {}
        self._messages: dict[str, InboxMessage] = {}
        self._threads: dict[str, list[str]] = {}  # thread_id -> message_ids
        self._lock = asyncio.Lock()
        self._gateway_inbox = gateway_inbox
        if self._gateway_inbox is None:
            enable_gateway = os.environ.get("MOLTBOT_GATEWAY_INBOX", "0") == "1"
            if enable_gateway or gateway_max_size is not None:
                # Opt-in gateway mirroring when explicitly configured.
                self._gateway_inbox = GatewayInbox(max_size=gateway_max_size or 10000)

        # Message handlers by channel type
        self._handlers: dict[ChannelType, Callable] = {}

        if self._storage_path:
            self._storage_path.mkdir(parents=True, exist_ok=True)

    def _map_priority(self, channel: Channel) -> MessagePriority:
        priority = channel.config.priority
        if priority >= 8:
            return MessagePriority.URGENT
        if priority >= 6:
            return MessagePriority.HIGH
        if priority >= 3:
            return MessagePriority.NORMAL
        return MessagePriority.LOW

    async def _mirror_to_gateway(self, message: InboxMessage, channel: Channel) -> None:
        if not self._gateway_inbox:
            return
        gateway_message = GatewayInboxMessage(
            message_id=message.id,
            channel=channel.config.type.value,
            sender=message.user_id,
            content=message.content,
            timestamp=message.created_at.timestamp(),
            thread_id=message.thread_id,
            priority=self._map_priority(channel),
            metadata={
                "channel_id": message.channel_id,
                "direction": message.direction,
                "content_type": message.content_type,
                "status": message.status.value,
                "intent": message.intent,
                "reply_to": message.reply_to,
                "external_id": message.external_id,
            },
        )
        await self._gateway_inbox.add_message(gateway_message)

    # ========== Channel Management ==========

    async def register_channel(
        self,
        config: ChannelConfig,
        user_id: str,
        tenant_id: str | None = None,
    ) -> Channel:
        """
        Register a new communication channel.

        Args:
            config: Channel configuration
            user_id: Owner user ID
            tenant_id: Tenant ID for multi-tenancy

        Returns:
            Registered channel
        """
        async with self._lock:
            channel_id = str(uuid.uuid4())

            channel = Channel(
                id=channel_id,
                config=config,
                user_id=user_id,
                tenant_id=tenant_id,
            )

            self._channels[channel_id] = channel
            logger.info(f"Registered {config.type.value} channel ({channel_id})")

            return channel

    async def get_channel(self, channel_id: str) -> Channel | None:
        """Get a channel by ID."""
        return self._channels.get(channel_id)

    async def list_channels(
        self,
        user_id: str | None = None,
        channel_type: ChannelType | None = None,
        tenant_id: str | None = None,
    ) -> list[Channel]:
        """List channels with optional filters."""
        channels = list(self._channels.values())

        if user_id:
            channels = [c for c in channels if c.user_id == user_id]
        if channel_type:
            channels = [c for c in channels if c.config.type == channel_type]
        if tenant_id:
            channels = [c for c in channels if c.tenant_id == tenant_id]

        return channels

    async def update_channel_status(
        self,
        channel_id: str,
        status: str,
    ) -> Channel | None:
        """Update a channel's status."""
        async with self._lock:
            channel = self._channels.get(channel_id)
            if not channel:
                return None

            channel.status = cast(Literal["active", "paused", "disconnected"], status)
            channel.updated_at = datetime.now(timezone.utc)
            return channel

    async def unregister_channel(self, channel_id: str) -> bool:
        """Unregister a channel."""
        async with self._lock:
            if channel_id not in self._channels:
                return False

            del self._channels[channel_id]
            logger.info(f"Unregistered channel {channel_id}")
            return True

    # ========== Message Management ==========

    async def receive_message(
        self,
        channel_id: str,
        user_id: str,
        content: str,
        content_type: str = "text",
        thread_id: str | None = None,
        reply_to: str | None = None,
        external_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> InboxMessage:
        """
        Receive an inbound message from a channel.

        Args:
            channel_id: Source channel
            user_id: Sender user ID
            content: Message content
            content_type: Content type (text, image, audio, etc.)
            thread_id: Thread ID for grouping
            reply_to: ID of message being replied to
            external_id: Provider's message ID
            metadata: Additional metadata

        Returns:
            Created message
        """
        async with self._lock:
            channel = self._channels.get(channel_id)
            if not channel:
                raise ValueError(f"Channel {channel_id} not found")

            message_id = str(uuid.uuid4())

            # Auto-create thread if replying
            if reply_to and not thread_id:
                original = self._messages.get(reply_to)
                if original:
                    thread_id = original.thread_id or reply_to

            # Create new thread if none specified
            if not thread_id:
                thread_id = message_id

            message = InboxMessage(
                id=message_id,
                channel_id=channel_id,
                user_id=user_id,
                direction="inbound",
                content=content,
                content_type=content_type,
                thread_id=thread_id,
                reply_to=reply_to,
                external_id=external_id,
                metadata=metadata or {},
            )

            self._messages[message_id] = message

            # Track thread
            if thread_id not in self._threads:
                self._threads[thread_id] = []
            self._threads[thread_id].append(message_id)

            # Update channel stats
            channel.last_message_at = datetime.now(timezone.utc)
            channel.message_count += 1

            logger.debug(f"Received message {message_id} on channel {channel_id}")

            # Process message (intent detection, etc.)
            await self._process_message(message)
            await self._mirror_to_gateway(message, channel)

            return message

    async def send_message(
        self,
        channel_id: str,
        user_id: str,
        content: str,
        content_type: str = "text",
        thread_id: str | None = None,
        reply_to: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> InboxMessage:
        """
        Send an outbound message through a channel.

        Args:
            channel_id: Target channel
            user_id: Recipient user ID
            content: Message content
            content_type: Content type
            thread_id: Thread ID
            reply_to: ID of message being replied to
            metadata: Additional metadata

        Returns:
            Created message
        """
        async with self._lock:
            channel = self._channels.get(channel_id)
            if not channel:
                raise ValueError(f"Channel {channel_id} not found")

            message_id = str(uuid.uuid4())

            # Auto-thread from reply
            if reply_to and not thread_id:
                original = self._messages.get(reply_to)
                if original:
                    thread_id = original.thread_id or reply_to

            message = InboxMessage(
                id=message_id,
                channel_id=channel_id,
                user_id=user_id,
                direction="outbound",
                content=content,
                content_type=content_type,
                status=InboxMessageStatus.PROCESSING,
                thread_id=thread_id,
                reply_to=reply_to,
                metadata=metadata or {},
            )

            self._messages[message_id] = message

            # Track thread
            if thread_id:
                if thread_id not in self._threads:
                    self._threads[thread_id] = []
                self._threads[thread_id].append(message_id)

            # Update channel stats
            channel.message_count += 1

        # Deliver message (would integrate with channel provider)
        delivered = await self._deliver_message(message, channel)
        if delivered:
            message.status = InboxMessageStatus.DELIVERED
            message.delivered_at = datetime.now(timezone.utc)
        else:
            message.status = InboxMessageStatus.FAILED

        message.updated_at = datetime.now(timezone.utc)

        logger.debug(f"Sent message {message_id} via channel {channel_id}")
        await self._mirror_to_gateway(message, channel)
        if self._gateway_inbox and reply_to:
            await self._gateway_inbox.mark_replied(reply_to)
        return message

    async def get_message(self, message_id: str) -> InboxMessage | None:
        """Get a message by ID."""
        return self._messages.get(message_id)

    async def list_messages(
        self,
        channel_id: str | None = None,
        user_id: str | None = None,
        thread_id: str | None = None,
        status: InboxMessageStatus | None = None,
        direction: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[InboxMessage]:
        """List messages with optional filters."""
        messages = list(self._messages.values())

        if channel_id:
            messages = [m for m in messages if m.channel_id == channel_id]
        if user_id:
            messages = [m for m in messages if m.user_id == user_id]
        if thread_id:
            messages = [m for m in messages if m.thread_id == thread_id]
        if status:
            messages = [m for m in messages if m.status == status]
        if direction:
            messages = [m for m in messages if m.direction == direction]

        # Sort by created_at descending
        messages.sort(key=lambda m: m.created_at, reverse=True)

        return messages[offset : offset + limit]

    async def get_thread(self, thread_id: str) -> list[InboxMessage]:
        """Get all messages in a thread."""
        message_ids = self._threads.get(thread_id, [])
        messages = [self._messages[mid] for mid in message_ids if mid in self._messages]
        messages.sort(key=lambda m: m.created_at)
        return messages

    async def mark_read(self, message_id: str) -> InboxMessage | None:
        """Mark a message as read."""
        async with self._lock:
            message = self._messages.get(message_id)
            if not message:
                return None

            message.status = InboxMessageStatus.READ
            message.read_at = datetime.now(timezone.utc)
            message.updated_at = datetime.now(timezone.utc)
            if self._gateway_inbox:
                await self._gateway_inbox.mark_read([message_id])
            return message

    async def archive_message(self, message_id: str) -> InboxMessage | None:
        """Archive a message."""
        async with self._lock:
            message = self._messages.get(message_id)
            if not message:
                return None

            message.status = InboxMessageStatus.ARCHIVED
            message.updated_at = datetime.now(timezone.utc)
            return message

    # ========== Message Processing ==========

    async def _process_message(self, message: InboxMessage) -> None:
        """Process an inbound message (intent detection, sentiment, etc.)."""
        # Simple intent detection (would integrate with NLU)
        content_lower = message.content.lower()

        if any(word in content_lower for word in ["help", "support", "assist"]):
            message.intent = "help_request"
        elif any(word in content_lower for word in ["buy", "purchase", "order"]):
            message.intent = "purchase_intent"
        elif any(word in content_lower for word in ["cancel", "stop", "unsubscribe"]):
            message.intent = "cancellation"
        elif "?" in message.content:
            message.intent = "question"
        else:
            message.intent = "general"

        message.status = InboxMessageStatus.PROCESSING
        message.updated_at = datetime.now(timezone.utc)

    async def _deliver_message(
        self,
        message: InboxMessage,
        channel: Channel,
    ) -> bool:
        """Deliver a message through its channel."""
        # Would integrate with channel-specific providers
        # For now, simulate successful delivery
        handler = self._handlers.get(channel.config.type)
        if handler:
            try:
                await handler(message, channel)
                return True
            except (RuntimeError, ValueError, AttributeError) as e:  # delivery handler
                logger.error(f"Failed to deliver message {message.id}: {e}")
                return False

        # Default: assume success for mock
        return True

    def register_handler(
        self,
        channel_type: ChannelType,
        handler: Callable,
    ) -> None:
        """Register a message handler for a channel type."""
        self._handlers[channel_type] = handler
        logger.info(f"Registered handler for {channel_type.value}")

    # ========== Analytics ==========

    async def get_stats(self) -> dict[str, Any]:
        """Get inbox statistics."""
        async with self._lock:
            by_channel: dict[str, int] = {}
            by_status: dict[str, int] = {}
            inbound = 0
            outbound = 0

            for message in self._messages.values():
                # By channel
                channel = self._channels.get(message.channel_id)
                if channel:
                    ch_type = channel.config.type.value
                    by_channel[ch_type] = by_channel.get(ch_type, 0) + 1

                # By status
                status = message.status.value
                by_status[status] = by_status.get(status, 0) + 1

                # Direction
                if message.direction == "inbound":
                    inbound += 1
                else:
                    outbound += 1

            return {
                "channels_total": len(self._channels),
                "channels_active": sum(1 for c in self._channels.values() if c.status == "active"),
                "messages_total": len(self._messages),
                "messages_inbound": inbound,
                "messages_outbound": outbound,
                "threads_total": len(self._threads),
                "by_channel": by_channel,
                "by_status": by_status,
            }
