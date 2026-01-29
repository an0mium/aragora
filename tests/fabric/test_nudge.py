"""
Tests for Nudge inter-agent messaging.

Tests cover:
- NudgeMessage creation and serialization
- NudgeRouter send/receive operations
- Priority ordering
- Message expiration
- Broadcast functionality
- Agent registration/unregistration
- Callbacks
"""

from __future__ import annotations

import asyncio
import time

import pytest

from aragora.fabric.nudge import (
    DeliveryStatus,
    MessagePriority,
    NudgeMessage,
    NudgeRouter,
    NudgeRouterConfig,
)


# =============================================================================
# NudgeMessage Tests
# =============================================================================


class TestNudgeMessage:
    """Tests for NudgeMessage dataclass."""

    def test_message_creation(self):
        msg = NudgeMessage(
            from_agent="agent-1",
            to_agent="agent-2",
            content="Hello, agent!",
        )
        assert msg.from_agent == "agent-1"
        assert msg.to_agent == "agent-2"
        assert msg.content == "Hello, agent!"
        assert msg.priority == 5  # Default
        assert msg.delivery_status == DeliveryStatus.PENDING
        assert msg.message_id.startswith("nudge-")

    def test_message_with_priority(self):
        msg = NudgeMessage(
            from_agent="a",
            to_agent="b",
            content="Urgent!",
            priority=MessagePriority.URGENT.value,
        )
        assert msg.priority == 20

    def test_message_with_metadata(self):
        msg = NudgeMessage(
            from_agent="a",
            to_agent="b",
            content="Task",
            metadata={"task_id": "t-123", "type": "code_review"},
        )
        assert msg.metadata["task_id"] == "t-123"
        assert msg.metadata["type"] == "code_review"

    def test_message_expiration(self):
        # Not expired
        msg = NudgeMessage(
            from_agent="a",
            to_agent="b",
            content="test",
            expires_at=time.time() + 3600,
        )
        assert not msg.is_expired()

        # Expired
        msg_expired = NudgeMessage(
            from_agent="a",
            to_agent="b",
            content="test",
            expires_at=time.time() - 1,
        )
        assert msg_expired.is_expired()

        # No expiration
        msg_no_expire = NudgeMessage(
            from_agent="a",
            to_agent="b",
            content="test",
        )
        assert not msg_no_expire.is_expired()

    def test_message_to_dict(self):
        msg = NudgeMessage(
            from_agent="a",
            to_agent="b",
            content="test",
            priority=10,
        )
        d = msg.to_dict()
        assert d["from_agent"] == "a"
        assert d["to_agent"] == "b"
        assert d["content"] == "test"
        assert d["priority"] == 10
        assert d["delivery_status"] == "pending"

    def test_message_from_dict(self):
        d = {
            "from_agent": "a",
            "to_agent": "b",
            "content": "test",
            "priority": 15,
            "delivery_status": "delivered",
        }
        msg = NudgeMessage.from_dict(d)
        assert msg.from_agent == "a"
        assert msg.to_agent == "b"
        assert msg.priority == 15
        assert msg.delivery_status == DeliveryStatus.DELIVERED


# =============================================================================
# NudgeRouter Send/Receive Tests
# =============================================================================


class TestNudgeRouterSendReceive:
    """Tests for NudgeRouter send and receive operations."""

    @pytest.fixture
    def router(self):
        return NudgeRouter()

    @pytest.mark.asyncio
    async def test_send_message(self, router):
        msg = NudgeMessage(from_agent="a", to_agent="b", content="hello")
        result = await router.send(msg)
        assert result is True

        stats = await router.get_stats()
        assert stats["messages_sent"] == 1

    @pytest.mark.asyncio
    async def test_receive_message(self, router):
        msg = NudgeMessage(from_agent="a", to_agent="b", content="hello")
        await router.send(msg)

        messages = await router.receive("b")
        assert len(messages) == 1
        assert messages[0].content == "hello"
        assert messages[0].delivery_status == DeliveryStatus.DELIVERED

    @pytest.mark.asyncio
    async def test_receive_marks_delivered(self, router):
        msg = NudgeMessage(from_agent="a", to_agent="b", content="hello")
        await router.send(msg)

        messages = await router.receive("b")
        assert messages[0].delivered_at is not None

        stats = await router.get_stats()
        assert stats["messages_delivered"] == 1

    @pytest.mark.asyncio
    async def test_receive_empty_queue(self, router):
        messages = await router.receive("nonexistent")
        assert messages == []

    @pytest.mark.asyncio
    async def test_receive_with_limit(self, router):
        for i in range(5):
            await router.send(NudgeMessage(from_agent="a", to_agent="b", content=f"msg-{i}"))

        messages = await router.receive("b", limit=3)
        assert len(messages) == 3

        # Remaining messages
        remaining = await router.receive("b")
        assert len(remaining) == 2

    @pytest.mark.asyncio
    async def test_peek_does_not_remove(self, router):
        msg = NudgeMessage(from_agent="a", to_agent="b", content="hello")
        await router.send(msg)

        # Peek
        peeked = await router.peek("b")
        assert len(peeked) == 1
        assert peeked[0].delivery_status == DeliveryStatus.PENDING

        # Messages still there
        messages = await router.receive("b")
        assert len(messages) == 1


# =============================================================================
# Priority Tests
# =============================================================================


class TestNudgeRouterPriority:
    """Tests for message priority ordering."""

    @pytest.fixture
    def router(self):
        return NudgeRouter()

    @pytest.mark.asyncio
    async def test_priority_ordering(self, router):
        # Send in random order
        await router.send(NudgeMessage(from_agent="a", to_agent="b", content="low", priority=0))
        await router.send(NudgeMessage(from_agent="a", to_agent="b", content="urgent", priority=20))
        await router.send(NudgeMessage(from_agent="a", to_agent="b", content="normal", priority=5))
        await router.send(NudgeMessage(from_agent="a", to_agent="b", content="high", priority=10))

        messages = await router.receive("b")
        assert len(messages) == 4
        assert messages[0].content == "urgent"
        assert messages[1].content == "high"
        assert messages[2].content == "normal"
        assert messages[3].content == "low"

    @pytest.mark.asyncio
    async def test_same_priority_fifo(self, router):
        # Same priority should be FIFO
        for i in range(3):
            await router.send(
                NudgeMessage(from_agent="a", to_agent="b", content=f"msg-{i}", priority=5)
            )

        messages = await router.receive("b")
        assert messages[0].content == "msg-0"
        assert messages[1].content == "msg-1"
        assert messages[2].content == "msg-2"


# =============================================================================
# Expiration Tests
# =============================================================================


class TestNudgeRouterExpiration:
    """Tests for message expiration."""

    @pytest.fixture
    def router(self):
        return NudgeRouter(NudgeRouterConfig(default_ttl_seconds=0))  # No default TTL

    @pytest.mark.asyncio
    async def test_expired_message_not_sent(self, router):
        msg = NudgeMessage(
            from_agent="a",
            to_agent="b",
            content="expired",
            expires_at=time.time() - 1,
        )
        result = await router.send(msg)
        assert result is False

    @pytest.mark.asyncio
    async def test_expired_message_filtered_on_receive(self, router):
        # Send message that will expire
        msg = NudgeMessage(
            from_agent="a",
            to_agent="b",
            content="will expire",
            expires_at=time.time() + 0.1,
        )
        await router.send(msg)

        # Wait for expiration
        await asyncio.sleep(0.2)

        messages = await router.receive("b")
        assert len(messages) == 0

        stats = await router.get_stats()
        assert stats["messages_expired"] == 1

    @pytest.mark.asyncio
    async def test_default_ttl_applied(self):
        router = NudgeRouter(NudgeRouterConfig(default_ttl_seconds=3600))
        msg = NudgeMessage(from_agent="a", to_agent="b", content="test")
        await router.send(msg)

        # Check that expires_at was set
        messages = await router.peek("b")
        assert messages[0].expires_at is not None
        assert messages[0].expires_at > time.time()


# =============================================================================
# Broadcast Tests
# =============================================================================


class TestNudgeRouterBroadcast:
    """Tests for broadcast functionality."""

    @pytest.fixture
    def router(self):
        return NudgeRouter()

    @pytest.mark.asyncio
    async def test_broadcast_to_all(self, router):
        # Register agents
        await router.register_agent("agent-1")
        await router.register_agent("agent-2")
        await router.register_agent("agent-3")

        # Broadcast
        count = await router.broadcast("sender", "Hello everyone!")
        assert count == 3

        # Each agent should have the message
        for agent in ["agent-1", "agent-2", "agent-3"]:
            messages = await router.receive(agent)
            assert len(messages) == 1
            assert messages[0].content == "Hello everyone!"
            assert messages[0].metadata.get("broadcast") is True

    @pytest.mark.asyncio
    async def test_broadcast_with_exclusion(self, router):
        await router.register_agent("agent-1")
        await router.register_agent("agent-2")
        await router.register_agent("agent-3")

        count = await router.broadcast("sender", "Not for agent-2", exclude=["agent-2"])
        assert count == 2

        # agent-2 should not have the message
        messages = await router.receive("agent-2")
        assert len(messages) == 0

    @pytest.mark.asyncio
    async def test_broadcast_with_priority(self, router):
        await router.register_agent("agent-1")

        await router.broadcast("sender", "Urgent!", priority=MessagePriority.URGENT.value)

        messages = await router.receive("agent-1")
        assert messages[0].priority == 20


# =============================================================================
# Agent Registration Tests
# =============================================================================


class TestNudgeRouterAgentManagement:
    """Tests for agent registration and unregistration."""

    @pytest.fixture
    def router(self):
        return NudgeRouter()

    @pytest.mark.asyncio
    async def test_register_agent(self, router):
        await router.register_agent("new-agent")

        stats = await router.get_stats()
        assert stats["agents_registered"] == 1

    @pytest.mark.asyncio
    async def test_register_agent_idempotent(self, router):
        await router.register_agent("agent-1")
        await router.register_agent("agent-1")

        stats = await router.get_stats()
        assert stats["agents_registered"] == 1

    @pytest.mark.asyncio
    async def test_unregister_agent(self, router):
        await router.register_agent("agent-1")
        await router.send(NudgeMessage(from_agent="a", to_agent="agent-1", content="test"))

        count = await router.unregister_agent("agent-1")
        assert count == 1

        stats = await router.get_stats()
        assert stats["agents_registered"] == 0

    @pytest.mark.asyncio
    async def test_unregister_nonexistent(self, router):
        count = await router.unregister_agent("nonexistent")
        assert count == 0


# =============================================================================
# Callback Tests
# =============================================================================


class TestNudgeRouterCallbacks:
    """Tests for message callbacks."""

    @pytest.fixture
    def router(self):
        return NudgeRouter()

    @pytest.mark.asyncio
    async def test_sync_callback(self, router):
        received = []

        def callback(msg):
            received.append(msg)

        router.on_message("agent-1", callback)

        await router.send(NudgeMessage(from_agent="a", to_agent="agent-1", content="test"))

        assert len(received) == 1
        assert received[0].content == "test"

    @pytest.mark.asyncio
    async def test_async_callback(self, router):
        received = []

        async def callback(msg):
            received.append(msg)

        router.on_message("agent-1", callback)

        await router.send(NudgeMessage(from_agent="a", to_agent="agent-1", content="test"))

        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_multiple_callbacks(self, router):
        count = [0]

        def callback1(msg):
            count[0] += 1

        def callback2(msg):
            count[0] += 10

        router.on_message("agent-1", callback1)
        router.on_message("agent-1", callback2)

        await router.send(NudgeMessage(from_agent="a", to_agent="agent-1", content="test"))

        assert count[0] == 11


# =============================================================================
# Acknowledge Tests
# =============================================================================


class TestNudgeRouterAcknowledge:
    """Tests for message acknowledgement."""

    @pytest.fixture
    def router(self):
        return NudgeRouter()

    @pytest.mark.asyncio
    async def test_acknowledge_message(self, router):
        msg = NudgeMessage(from_agent="a", to_agent="b", content="test")
        await router.send(msg)

        # Receive (marks as delivered)
        messages = await router.receive("b")
        assert messages[0].delivery_status == DeliveryStatus.DELIVERED

        # Acknowledge (marks as read)
        result = await router.acknowledge(msg.message_id)
        assert result is True
        assert msg.delivery_status == DeliveryStatus.READ
        assert msg.read_at is not None

    @pytest.mark.asyncio
    async def test_acknowledge_nonexistent(self, router):
        result = await router.acknowledge("nonexistent")
        assert result is False


# =============================================================================
# Stats Tests
# =============================================================================


class TestNudgeRouterStats:
    """Tests for router statistics."""

    @pytest.fixture
    def router(self):
        return NudgeRouter()

    @pytest.mark.asyncio
    async def test_stats(self, router):
        await router.register_agent("a")
        await router.register_agent("b")

        await router.send(NudgeMessage(from_agent="x", to_agent="a", content="1"))
        await router.send(NudgeMessage(from_agent="x", to_agent="a", content="2"))
        await router.send(NudgeMessage(from_agent="x", to_agent="b", content="3"))

        await router.receive("a")

        stats = await router.get_stats()
        assert stats["messages_sent"] == 3
        assert stats["messages_delivered"] == 2
        assert stats["agents_registered"] == 2
        assert stats["total_pending"] == 1

    @pytest.mark.asyncio
    async def test_pending_count(self, router):
        await router.send(NudgeMessage(from_agent="a", to_agent="b", content="1"))
        await router.send(NudgeMessage(from_agent="a", to_agent="b", content="2"))

        count = await router.get_pending_count("b")
        assert count == 2

        await router.receive("b", limit=1)

        count = await router.get_pending_count("b")
        assert count == 1


# =============================================================================
# Queue Size Tests
# =============================================================================


class TestNudgeRouterQueueSize:
    """Tests for queue size limits."""

    @pytest.mark.asyncio
    async def test_queue_full(self):
        router = NudgeRouter(NudgeRouterConfig(max_queue_size=2))

        await router.send(NudgeMessage(from_agent="a", to_agent="b", content="1"))
        await router.send(NudgeMessage(from_agent="a", to_agent="b", content="2"))

        # Queue full
        result = await router.send(NudgeMessage(from_agent="a", to_agent="b", content="3"))
        assert result is False

        stats = await router.get_stats()
        assert stats["messages_sent"] == 2


# =============================================================================
# Lifecycle Tests
# =============================================================================


class TestNudgeRouterLifecycle:
    """Tests for router start/stop lifecycle."""

    @pytest.mark.asyncio
    async def test_start_stop(self):
        router = NudgeRouter()
        await router.start()
        assert router._cleanup_task is not None

        await router.stop()
        assert router._cleanup_task is None

    @pytest.mark.asyncio
    async def test_start_idempotent(self):
        router = NudgeRouter()
        await router.start()
        task1 = router._cleanup_task

        await router.start()
        task2 = router._cleanup_task

        assert task1 is task2
        await router.stop()
