"""
Nudge - Inter-agent messaging system.

Pattern: Inter-agent Mail
Inspired by: Gastown (https://github.com/gastown)
Aragora adaptation: Async message routing with priority queues and delivery guarantees

Implements Gastown's agent-to-agent communication pattern. Agents can send
"nudges" to other agents asynchronously, enabling:
- Task delegation between agents
- Status updates and notifications
- Collaborative workflows
- Broadcast announcements

Key concepts:
- NudgeMessage: A message from one agent to another
- NudgeRouter: Routes and delivers messages between agents
- DeliveryStatus: Tracks message delivery state

Usage:
    from aragora.fabric.nudge import NudgeRouter, NudgeMessage

    router = NudgeRouter()

    # Send a message
    msg = NudgeMessage(
        from_agent="planner-1",
        to_agent="coder-2",
        content="Please implement the auth module",
        priority=5,
    )
    await router.send(msg)

    # Receive messages
    messages = await router.receive("coder-2")

    # Broadcast to all agents
    await router.broadcast("planner-1", "Standup in 5 minutes")
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from collections.abc import Callable

logger = logging.getLogger(__name__)


class DeliveryStatus(Enum):
    """Message delivery status."""

    PENDING = "pending"
    DELIVERED = "delivered"
    READ = "read"
    EXPIRED = "expired"
    FAILED = "failed"


class MessagePriority(Enum):
    """Message priority levels."""

    LOW = 0
    NORMAL = 5
    HIGH = 10
    URGENT = 20


@dataclass
class NudgeMessage:
    """
    A message from one agent to another.

    Attributes:
        message_id: Unique message identifier.
        from_agent: Sender agent ID.
        to_agent: Recipient agent ID (or "*" for broadcast).
        content: Message content.
        priority: Message priority (higher = more urgent).
        metadata: Additional message metadata.
        created_at: Timestamp when message was created.
        expires_at: Optional expiration timestamp.
        delivery_status: Current delivery status.
        delivered_at: Timestamp when delivered.
        read_at: Timestamp when read/acknowledged.
    """

    from_agent: str
    to_agent: str
    content: str
    message_id: str = field(default_factory=lambda: f"nudge-{uuid.uuid4().hex[:12]}")
    priority: int = 5
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    expires_at: float | None = None
    delivery_status: DeliveryStatus = DeliveryStatus.PENDING
    delivered_at: float | None = None
    read_at: float | None = None

    def is_expired(self) -> bool:
        """Check if the message has expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "message_id": self.message_id,
            "from_agent": self.from_agent,
            "to_agent": self.to_agent,
            "content": self.content,
            "priority": self.priority,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "delivery_status": self.delivery_status.value,
            "delivered_at": self.delivered_at,
            "read_at": self.read_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NudgeMessage:
        """Deserialize from dictionary."""
        status = data.get("delivery_status", "pending")
        if isinstance(status, str):
            status = DeliveryStatus(status)

        return cls(
            message_id=data.get("message_id", f"nudge-{uuid.uuid4().hex[:12]}"),
            from_agent=data["from_agent"],
            to_agent=data["to_agent"],
            content=data["content"],
            priority=data.get("priority", 5),
            metadata=data.get("metadata", {}),
            created_at=data.get("created_at", time.time()),
            expires_at=data.get("expires_at"),
            delivery_status=status,
            delivered_at=data.get("delivered_at"),
            read_at=data.get("read_at"),
        )


@dataclass
class NudgeRouterConfig:
    """Configuration for NudgeRouter."""

    max_queue_size: int = 1000
    default_ttl_seconds: float = 86400  # 24 hours
    cleanup_interval_seconds: float = 60.0
    enable_persistence: bool = False
    persistence_path: str | None = None


class NudgeRouter:
    """
    Routes and delivers messages between agents.

    Provides:
    - Priority-based message queuing
    - Delivery tracking
    - Message expiration
    - Broadcast support
    - Optional persistence

    Usage:
        router = NudgeRouter()
        await router.send(NudgeMessage(from_agent="a", to_agent="b", content="hi"))
        messages = await router.receive("b")
    """

    def __init__(self, config: NudgeRouterConfig | None = None) -> None:
        self._config = config or NudgeRouterConfig()
        self._queues: dict[str, list[NudgeMessage]] = {}
        self._all_messages: dict[str, NudgeMessage] = {}
        self._lock = asyncio.Lock()
        self._cleanup_task: asyncio.Task[None] | None = None
        self._callbacks: dict[str, list[Callable[[NudgeMessage], Any]]] = {}

        # Stats
        self._messages_sent = 0
        self._messages_delivered = 0
        self._messages_expired = 0
        self._broadcasts_sent = 0

    async def start(self) -> None:
        """Start the router (begins cleanup task)."""
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            logger.debug("Nudge router started")

    async def stop(self) -> None:
        """Stop the router."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None
        logger.debug("Nudge router stopped")

    async def send(self, message: NudgeMessage) -> bool:
        """
        Send a message to an agent.

        Args:
            message: The message to send.

        Returns:
            True if queued successfully, False if queue is full or expired.
        """
        if message.is_expired():
            logger.warning(f"Message {message.message_id} already expired")
            return False

        # Set default expiration if not set
        if message.expires_at is None and self._config.default_ttl_seconds > 0:
            message.expires_at = time.time() + self._config.default_ttl_seconds

        async with self._lock:
            # Get or create queue for recipient
            if message.to_agent not in self._queues:
                self._queues[message.to_agent] = []

            queue = self._queues[message.to_agent]

            # Check queue size
            if len(queue) >= self._config.max_queue_size:
                logger.warning(f"Queue full for agent {message.to_agent}")
                return False

            # Insert in priority order (higher priority first)
            inserted = False
            for i, existing in enumerate(queue):
                if message.priority > existing.priority:
                    queue.insert(i, message)
                    inserted = True
                    break
            if not inserted:
                queue.append(message)

            # Track message
            self._all_messages[message.message_id] = message
            self._messages_sent += 1

        logger.debug(f"Nudge sent: {message.from_agent} -> {message.to_agent}")

        # Trigger callbacks
        await self._trigger_callbacks(message.to_agent, message)

        return True

    async def receive(
        self,
        agent_id: str,
        limit: int = 50,
        mark_delivered: bool = True,
    ) -> list[NudgeMessage]:
        """
        Receive messages for an agent.

        Args:
            agent_id: The recipient agent ID.
            limit: Maximum messages to return.
            mark_delivered: Whether to mark messages as delivered.

        Returns:
            List of messages, ordered by priority (highest first).
        """
        async with self._lock:
            if agent_id not in self._queues:
                return []

            queue = self._queues[agent_id]
            messages: list[Any] = []

            # Collect non-expired messages
            remaining = []
            for msg in queue:
                if msg.is_expired():
                    msg.delivery_status = DeliveryStatus.EXPIRED
                    self._messages_expired += 1
                elif len(messages) < limit:
                    if mark_delivered:
                        msg.delivery_status = DeliveryStatus.DELIVERED
                        msg.delivered_at = time.time()
                        self._messages_delivered += 1
                        messages.append(msg)
                    else:
                        # Peek mode: add to messages but keep in queue
                        messages.append(msg)
                        remaining.append(msg)
                else:
                    remaining.append(msg)

            # Only modify queue when consuming (not peeking)
            if mark_delivered:
                self._queues[agent_id] = remaining

        return messages

    async def peek(
        self,
        agent_id: str,
        limit: int = 50,
    ) -> list[NudgeMessage]:
        """
        Peek at messages without marking them as delivered.

        Args:
            agent_id: The recipient agent ID.
            limit: Maximum messages to return.

        Returns:
            List of pending messages.
        """
        return await self.receive(agent_id, limit=limit, mark_delivered=False)

    async def acknowledge(self, message_id: str) -> bool:
        """
        Mark a message as read/acknowledged.

        Args:
            message_id: The message ID to acknowledge.

        Returns:
            True if acknowledged, False if not found.
        """
        async with self._lock:
            msg = self._all_messages.get(message_id)
            if not msg:
                return False

            msg.delivery_status = DeliveryStatus.READ
            msg.read_at = time.time()
            return True

    async def broadcast(
        self,
        from_agent: str,
        content: str,
        priority: int = 5,
        exclude: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """
        Broadcast a message to all known agents.

        Args:
            from_agent: Sender agent ID.
            content: Message content.
            priority: Message priority.
            exclude: Agent IDs to exclude from broadcast.
            metadata: Additional metadata.

        Returns:
            Number of agents the message was sent to.
        """
        exclude = exclude or []
        metadata = metadata or {}
        metadata["broadcast"] = True

        async with self._lock:
            recipients = [aid for aid in self._queues.keys() if aid not in exclude]

        count = 0
        for recipient in recipients:
            msg = NudgeMessage(
                from_agent=from_agent,
                to_agent=recipient,
                content=content,
                priority=priority,
                metadata=metadata,
            )
            if await self.send(msg):
                count += 1

        self._broadcasts_sent += 1
        logger.debug(f"Broadcast from {from_agent} sent to {count} agents")
        return count

    async def register_agent(self, agent_id: str) -> None:
        """Register an agent to receive messages."""
        async with self._lock:
            if agent_id not in self._queues:
                self._queues[agent_id] = []
                logger.debug(f"Agent {agent_id} registered for nudges")

    async def unregister_agent(self, agent_id: str) -> int:
        """
        Unregister an agent and clear their queue.

        Returns:
            Number of messages that were pending.
        """
        async with self._lock:
            if agent_id not in self._queues:
                return 0

            count = len(self._queues[agent_id])
            del self._queues[agent_id]
            logger.debug(f"Agent {agent_id} unregistered, {count} messages cleared")
            return count

    def on_message(
        self,
        agent_id: str,
        callback: Callable[[NudgeMessage], Any],
    ) -> None:
        """
        Register a callback for incoming messages.

        Args:
            agent_id: Agent to receive callbacks for.
            callback: Function to call when message arrives.
        """
        if agent_id not in self._callbacks:
            self._callbacks[agent_id] = []
        self._callbacks[agent_id].append(callback)

    async def _trigger_callbacks(self, agent_id: str, message: NudgeMessage) -> None:
        """Trigger registered callbacks for a message."""
        if agent_id not in self._callbacks:
            return

        for callback in self._callbacks[agent_id]:
            try:
                result = callback(message)
                if asyncio.iscoroutine(result):
                    await result
            except (RuntimeError, ValueError, AttributeError) as e:  # user-supplied callback
                logger.error(f"Callback error for agent {agent_id}: {e}")

    async def get_pending_count(self, agent_id: str) -> int:
        """Get count of pending messages for an agent."""
        async with self._lock:
            if agent_id not in self._queues:
                return 0
            return len([m for m in self._queues[agent_id] if not m.is_expired()])

    async def get_stats(self) -> dict[str, Any]:
        """Get router statistics."""
        async with self._lock:
            total_pending = sum(len(q) for q in self._queues.values())
            return {
                "messages_sent": self._messages_sent,
                "messages_delivered": self._messages_delivered,
                "messages_expired": self._messages_expired,
                "broadcasts_sent": self._broadcasts_sent,
                "agents_registered": len(self._queues),
                "total_pending": total_pending,
            }

    async def _cleanup_loop(self) -> None:
        """Background task to clean up expired messages."""
        while True:
            try:
                await asyncio.sleep(self._config.cleanup_interval_seconds)
                await self._cleanup_expired()
            except asyncio.CancelledError:
                break
            except (RuntimeError, OSError, ValueError) as e:
                logger.error(f"Cleanup error: {e}")

    async def _cleanup_expired(self) -> int:
        """Remove expired messages from all queues."""
        expired_count = 0

        async with self._lock:
            for agent_id, queue in self._queues.items():
                remaining = []
                for msg in queue:
                    if msg.is_expired():
                        msg.delivery_status = DeliveryStatus.EXPIRED
                        expired_count += 1
                    else:
                        remaining.append(msg)
                self._queues[agent_id] = remaining

        if expired_count > 0:
            self._messages_expired += expired_count
            logger.debug(f"Cleaned up {expired_count} expired messages")

        return expired_count
