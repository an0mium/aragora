"""
Agent Channel - Peer-to-peer messaging for debates.

Enables direct communication between agents during debate rounds:
- Agents can broadcast messages to all peers
- Agents can send private messages to specific peers
- Channels maintain message history for context
- Supports async message queues (mailbox pattern)

This complements the A2A protocol (for external agent invocation)
by providing internal messaging during debates.

Usage:
    from aragora.debate.agent_channel import ChannelManager

    # Create channel for a debate
    manager = ChannelManager()
    channel = await manager.create_channel("debate_123")

    # Agents join
    await channel.join("claude")
    await channel.join("gpt4")

    # Send messages
    await channel.broadcast("claude", "I propose we use token bucket algorithm")
    await channel.send("claude", "gpt4", "What do you think about this approach?")

    # Receive messages
    messages = await channel.receive("gpt4")
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Coroutine, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """Types of inter-agent messages."""

    BROADCAST = "broadcast"  # Message to all agents
    DIRECT = "direct"  # Private message to specific agent
    PROPOSAL = "proposal"  # Proposal announcement
    CRITIQUE = "critique"  # Critique of a proposal
    QUERY = "query"  # Question to other agents
    RESPONSE = "response"  # Response to a query
    SIGNAL = "signal"  # Control signal (ready, done, etc.)


@dataclass
class ChannelMessage:
    """A message in an agent channel."""

    message_id: str
    channel_id: str
    sender: str
    message_type: MessageType
    content: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    recipient: Optional[str] = None  # None for broadcast
    metadata: dict[str, Any] = field(default_factory=dict)
    reply_to: Optional[str] = None  # For threaded conversations

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "message_id": self.message_id,
            "channel_id": self.channel_id,
            "sender": self.sender,
            "message_type": self.message_type.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "recipient": self.recipient,
            "metadata": self.metadata,
            "reply_to": self.reply_to,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChannelMessage":
        """Create from dictionary."""
        return cls(
            message_id=data["message_id"],
            channel_id=data["channel_id"],
            sender=data["sender"],
            message_type=MessageType(data["message_type"]),
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            recipient=data.get("recipient"),
            metadata=data.get("metadata", {}),
            reply_to=data.get("reply_to"),
        )


class AgentChannel:
    """
    Communication channel for agents in a debate.

    Features:
    - Async message queues per agent (mailbox pattern)
    - Broadcast and direct messaging
    - Message history for context
    - Event handlers for message receipt
    """

    def __init__(self, channel_id: str, max_history: int = 1000):
        """
        Initialize an agent channel.

        Args:
            channel_id: Unique channel identifier (typically debate_id)
            max_history: Maximum messages to retain in history
        """
        self._channel_id = channel_id
        self._max_history = max_history

        # Agent mailboxes (async queues)
        self._mailboxes: dict[str, asyncio.Queue[ChannelMessage]] = {}

        # Message history
        self._history: list[ChannelMessage] = []

        # Message handlers
        self._handlers: dict[str, list[Callable]] = {}

        # Lock for thread-safe operations
        self._lock = asyncio.Lock()

        # Channel state
        self._created_at = datetime.now(timezone.utc)
        self._closed = False

    @property
    def channel_id(self) -> str:
        """Get channel ID."""
        return self._channel_id

    @property
    def agents(self) -> list[str]:
        """List of joined agents."""
        return list(self._mailboxes.keys())

    @property
    def history(self) -> list[ChannelMessage]:
        """Get message history (readonly copy)."""
        return list(self._history)

    async def join(self, agent_name: str) -> bool:
        """
        Add an agent to the channel.

        Args:
            agent_name: Name of the agent joining

        Returns:
            True if joined successfully
        """
        if self._closed:
            return False

        async with self._lock:
            if agent_name not in self._mailboxes:
                self._mailboxes[agent_name] = asyncio.Queue()
                logger.debug(f"Agent {agent_name} joined channel {self._channel_id}")
                return True
            return False

    async def leave(self, agent_name: str) -> bool:
        """
        Remove an agent from the channel.

        Args:
            agent_name: Name of the agent leaving

        Returns:
            True if left successfully
        """
        async with self._lock:
            if agent_name in self._mailboxes:
                del self._mailboxes[agent_name]
                if agent_name in self._handlers:
                    del self._handlers[agent_name]
                logger.debug(f"Agent {agent_name} left channel {self._channel_id}")
                return True
            return False

    async def broadcast(
        self,
        sender: str,
        content: str,
        message_type: MessageType = MessageType.BROADCAST,
        metadata: Optional[dict[str, Any]] = None,
        reply_to: Optional[str] = None,
    ) -> ChannelMessage:
        """
        Broadcast a message to all agents in the channel.

        Args:
            sender: Agent sending the message
            content: Message content
            message_type: Type of message
            metadata: Optional metadata
            reply_to: Optional message ID being replied to

        Returns:
            The sent message
        """
        message = ChannelMessage(
            message_id=f"msg_{uuid4().hex[:12]}",
            channel_id=self._channel_id,
            sender=sender,
            message_type=message_type,
            content=content,
            metadata=metadata or {},
            reply_to=reply_to,
        )

        async with self._lock:
            # Add to history
            self._history.append(message)
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history :]

            # Deliver to all mailboxes except sender
            recipients = []
            for agent_name, mailbox in self._mailboxes.items():
                if agent_name != sender:
                    await mailbox.put(message)
                    recipients.append(agent_name)

        # Trigger handlers for all recipients
        for recipient in recipients:
            await self._trigger_handlers(recipient, message)

        logger.debug(
            f"[{self._channel_id}] {sender} broadcast: {content[:50]}..."
            if len(content) > 50
            else f"[{self._channel_id}] {sender} broadcast: {content}"
        )

        return message

    async def send(
        self,
        sender: str,
        recipient: str,
        content: str,
        message_type: MessageType = MessageType.DIRECT,
        metadata: Optional[dict[str, Any]] = None,
        reply_to: Optional[str] = None,
    ) -> Optional[ChannelMessage]:
        """
        Send a direct message to a specific agent.

        Args:
            sender: Agent sending the message
            recipient: Target agent
            content: Message content
            message_type: Type of message
            metadata: Optional metadata
            reply_to: Optional message ID being replied to

        Returns:
            The sent message, or None if recipient not in channel
        """
        if recipient not in self._mailboxes:
            logger.warning(f"Cannot send to {recipient}: not in channel {self._channel_id}")
            return None

        message = ChannelMessage(
            message_id=f"msg_{uuid4().hex[:12]}",
            channel_id=self._channel_id,
            sender=sender,
            message_type=message_type,
            content=content,
            recipient=recipient,
            metadata=metadata or {},
            reply_to=reply_to,
        )

        async with self._lock:
            # Add to history
            self._history.append(message)
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history :]

            # Deliver to recipient mailbox
            await self._mailboxes[recipient].put(message)

        # Trigger handlers
        await self._trigger_handlers(recipient, message)

        logger.debug(f"[{self._channel_id}] {sender} -> {recipient}: {content[:50]}...")

        return message

    async def receive(
        self,
        agent_name: str,
        timeout: Optional[float] = None,
    ) -> Optional[ChannelMessage]:
        """
        Receive next message for an agent.

        Args:
            agent_name: Agent receiving messages
            timeout: Optional timeout in seconds

        Returns:
            Next message, or None if timeout/not joined
        """
        if agent_name not in self._mailboxes:
            return None

        mailbox = self._mailboxes[agent_name]

        try:
            if timeout is not None:
                return await asyncio.wait_for(mailbox.get(), timeout=timeout)
            else:
                return await mailbox.get()
        except asyncio.TimeoutError:
            return None

    async def receive_all(
        self,
        agent_name: str,
        max_messages: int = 100,
    ) -> list[ChannelMessage]:
        """
        Receive all pending messages for an agent.

        Args:
            agent_name: Agent receiving messages
            max_messages: Maximum messages to return

        Returns:
            List of pending messages
        """
        if agent_name not in self._mailboxes:
            return []

        mailbox = self._mailboxes[agent_name]
        messages: list[ChannelMessage] = []

        while len(messages) < max_messages and not mailbox.empty():
            try:
                msg = mailbox.get_nowait()
                messages.append(msg)
            except asyncio.QueueEmpty:
                break

        return messages

    def pending_count(self, agent_name: str) -> int:
        """Get count of pending messages for an agent."""
        if agent_name not in self._mailboxes:
            return 0
        return self._mailboxes[agent_name].qsize()

    def on_message(
        self,
        agent_name: str,
        handler: Callable[[ChannelMessage], Coroutine[Any, Any, None]],
    ) -> None:
        """
        Register a message handler for an agent.

        Args:
            agent_name: Agent to handle messages for
            handler: Async handler function
        """
        if agent_name not in self._handlers:
            self._handlers[agent_name] = []
        self._handlers[agent_name].append(handler)

    async def _trigger_handlers(self, agent_name: str, message: ChannelMessage) -> None:
        """Trigger message handlers for an agent."""
        handlers = self._handlers.get(agent_name, [])
        for handler in handlers:
            try:
                await handler(message)
            except Exception as e:
                logger.error(f"Handler error for {agent_name}: {e}")

    def get_history(
        self,
        limit: int = 100,
        sender: Optional[str] = None,
        message_type: Optional[MessageType] = None,
    ) -> list[ChannelMessage]:
        """
        Get filtered message history.

        Args:
            limit: Maximum messages to return
            sender: Filter by sender
            message_type: Filter by message type

        Returns:
            Filtered message list
        """
        messages = self._history

        if sender:
            messages = [m for m in messages if m.sender == sender]

        if message_type:
            messages = [m for m in messages if m.message_type == message_type]

        return messages[-limit:]

    def get_thread(self, root_message_id: str) -> list[ChannelMessage]:
        """
        Get a conversation thread starting from a message.

        Args:
            root_message_id: ID of the root message

        Returns:
            Messages in the thread
        """
        thread = []
        thread_ids = {root_message_id}

        for msg in self._history:
            if msg.message_id in thread_ids or msg.reply_to in thread_ids:
                thread.append(msg)
                thread_ids.add(msg.message_id)

        return thread

    async def close(self) -> None:
        """Close the channel."""
        self._closed = True
        async with self._lock:
            self._mailboxes.clear()
            self._handlers.clear()
        logger.debug(f"Channel {self._channel_id} closed")

    def to_context(self, limit: int = 10) -> str:
        """
        Convert recent history to context string for prompts.

        Args:
            limit: Maximum messages to include

        Returns:
            Formatted context string
        """
        recent = self._history[-limit:]
        if not recent:
            return ""

        lines = ["## Recent Agent Discussion"]
        for msg in recent:
            prefix = f"[{msg.sender}]"
            if msg.recipient:
                prefix = f"[{msg.sender} -> {msg.recipient}]"
            lines.append(f"{prefix}: {msg.content}")

        return "\n".join(lines)


class ChannelManager:
    """
    Manager for agent communication channels.

    Handles lifecycle of channels across debates.
    """

    def __init__(self):
        """Initialize channel manager."""
        self._channels: dict[str, AgentChannel] = {}
        self._lock = asyncio.Lock()

    async def create_channel(
        self,
        channel_id: str,
        max_history: int = 1000,
    ) -> AgentChannel:
        """
        Create a new channel.

        Args:
            channel_id: Unique channel identifier
            max_history: Maximum message history

        Returns:
            Created channel
        """
        async with self._lock:
            if channel_id in self._channels:
                return self._channels[channel_id]

            channel = AgentChannel(channel_id, max_history)
            self._channels[channel_id] = channel
            logger.info(f"Created channel: {channel_id}")
            return channel

    async def get_channel(self, channel_id: str) -> Optional[AgentChannel]:
        """Get a channel by ID."""
        return self._channels.get(channel_id)

    async def close_channel(self, channel_id: str) -> bool:
        """
        Close and remove a channel.

        Args:
            channel_id: Channel to close

        Returns:
            True if closed successfully
        """
        async with self._lock:
            if channel_id in self._channels:
                await self._channels[channel_id].close()
                del self._channels[channel_id]
                logger.info(f"Closed channel: {channel_id}")
                return True
            return False

    def list_channels(self) -> list[str]:
        """List all active channel IDs."""
        return list(self._channels.keys())

    async def broadcast_to_all(
        self,
        sender: str,
        content: str,
        message_type: MessageType = MessageType.BROADCAST,
    ) -> int:
        """
        Broadcast a message to all channels.

        Args:
            sender: Sender name
            content: Message content
            message_type: Message type

        Returns:
            Number of channels messaged
        """
        count = 0
        for channel in self._channels.values():
            if sender in channel.agents:
                await channel.broadcast(sender, content, message_type)
                count += 1
        return count


# Default manager instance
_default_manager: Optional[ChannelManager] = None


def get_channel_manager() -> ChannelManager:
    """Get or create the default channel manager."""
    global _default_manager
    if _default_manager is None:
        _default_manager = ChannelManager()
    return _default_manager


def reset_channel_manager() -> None:
    """Reset the default channel manager (for testing)."""
    global _default_manager
    _default_manager = None


__all__ = [
    "AgentChannel",
    "ChannelManager",
    "ChannelMessage",
    "MessageType",
    "get_channel_manager",
    "reset_channel_manager",
]
