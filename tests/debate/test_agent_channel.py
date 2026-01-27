"""Tests for agent communication channels."""

import asyncio

import pytest

from aragora.debate.agent_channel import (
    AgentChannel,
    ChannelManager,
    ChannelMessage,
    MessageType,
    get_channel_manager,
    reset_channel_manager,
)


class TestChannelMessage:
    """Tests for ChannelMessage."""

    def test_create_message(self):
        """Test creating a channel message."""
        msg = ChannelMessage(
            message_id="msg_123",
            channel_id="debate_456",
            sender="claude",
            message_type=MessageType.PROPOSAL,
            content="I propose we use token bucket",
        )

        assert msg.message_id == "msg_123"
        assert msg.sender == "claude"
        assert msg.message_type == MessageType.PROPOSAL
        assert msg.recipient is None

    def test_message_to_dict(self):
        """Test message serialization."""
        msg = ChannelMessage(
            message_id="msg_123",
            channel_id="debate_456",
            sender="claude",
            message_type=MessageType.DIRECT,
            content="Private message",
            recipient="gpt4",
            metadata={"priority": "high"},
        )

        data = msg.to_dict()

        assert data["message_id"] == "msg_123"
        assert data["sender"] == "claude"
        assert data["recipient"] == "gpt4"
        assert data["message_type"] == "direct"
        assert data["metadata"]["priority"] == "high"

    def test_message_from_dict(self):
        """Test message deserialization."""
        data = {
            "message_id": "msg_123",
            "channel_id": "debate_456",
            "sender": "claude",
            "message_type": "broadcast",
            "content": "Hello everyone",
            "timestamp": "2024-01-15T10:00:00+00:00",
            "metadata": {},
        }

        msg = ChannelMessage.from_dict(data)

        assert msg.message_id == "msg_123"
        assert msg.sender == "claude"
        assert msg.message_type == MessageType.BROADCAST


class TestAgentChannel:
    """Tests for AgentChannel."""

    @pytest.fixture
    def channel(self):
        """Create a test channel."""
        return AgentChannel("test_debate", max_history=100)

    @pytest.mark.asyncio
    async def test_join_channel(self, channel):
        """Test agents joining a channel."""
        await channel.join("claude")
        await channel.join("gpt4")

        assert "claude" in channel.agents
        assert "gpt4" in channel.agents
        assert len(channel.agents) == 2

    @pytest.mark.asyncio
    async def test_join_already_joined(self, channel):
        """Test joining when already in channel."""
        await channel.join("claude")
        result = await channel.join("claude")

        assert result is False  # Already joined
        assert channel.agents.count("claude") == 1

    @pytest.mark.asyncio
    async def test_leave_channel(self, channel):
        """Test leaving a channel."""
        await channel.join("claude")
        await channel.join("gpt4")

        await channel.leave("claude")

        assert "claude" not in channel.agents
        assert "gpt4" in channel.agents

    @pytest.mark.asyncio
    async def test_broadcast_message(self, channel):
        """Test broadcasting a message."""
        await channel.join("claude")
        await channel.join("gpt4")
        await channel.join("gemini")

        msg = await channel.broadcast(
            sender="claude",
            content="I propose token bucket algorithm",
            message_type=MessageType.PROPOSAL,
        )

        assert msg.sender == "claude"
        assert msg.message_type == MessageType.PROPOSAL
        assert msg.recipient is None

        # Other agents should receive
        gpt4_pending = channel.pending_count("gpt4")
        gemini_pending = channel.pending_count("gemini")
        claude_pending = channel.pending_count("claude")

        assert gpt4_pending == 1
        assert gemini_pending == 1
        assert claude_pending == 0  # Sender doesn't receive own broadcast

    @pytest.mark.asyncio
    async def test_direct_message(self, channel):
        """Test sending a direct message."""
        await channel.join("claude")
        await channel.join("gpt4")
        await channel.join("gemini")

        msg = await channel.send(
            sender="claude",
            recipient="gpt4",
            content="What do you think?",
            message_type=MessageType.QUERY,
        )

        assert msg is not None
        assert msg.recipient == "gpt4"

        # Only gpt4 should receive
        assert channel.pending_count("gpt4") == 1
        assert channel.pending_count("gemini") == 0
        assert channel.pending_count("claude") == 0

    @pytest.mark.asyncio
    async def test_direct_message_to_unknown_agent(self, channel):
        """Test sending to unknown agent."""
        await channel.join("claude")

        msg = await channel.send(
            sender="claude",
            recipient="unknown_agent",
            content="Hello?",
        )

        assert msg is None

    @pytest.mark.asyncio
    async def test_receive_message(self, channel):
        """Test receiving a message."""
        await channel.join("claude")
        await channel.join("gpt4")

        await channel.broadcast("claude", "Hello!")

        msg = await channel.receive("gpt4", timeout=1.0)

        assert msg is not None
        assert msg.sender == "claude"
        assert msg.content == "Hello!"

    @pytest.mark.asyncio
    async def test_receive_timeout(self, channel):
        """Test receive timeout."""
        await channel.join("claude")

        msg = await channel.receive("claude", timeout=0.1)

        assert msg is None

    @pytest.mark.asyncio
    async def test_receive_all_messages(self, channel):
        """Test receiving all pending messages."""
        await channel.join("claude")
        await channel.join("gpt4")

        # Send multiple messages
        await channel.broadcast("claude", "Message 1")
        await channel.broadcast("claude", "Message 2")
        await channel.broadcast("claude", "Message 3")

        messages = await channel.receive_all("gpt4")

        assert len(messages) == 3
        assert messages[0].content == "Message 1"
        assert messages[2].content == "Message 3"

    @pytest.mark.asyncio
    async def test_message_history(self, channel):
        """Test message history."""
        await channel.join("claude")
        await channel.join("gpt4")

        await channel.broadcast("claude", "First message")
        await channel.send("gpt4", "claude", "Reply")
        await channel.broadcast("claude", "Second broadcast")

        history = channel.history
        assert len(history) == 3

    @pytest.mark.asyncio
    async def test_history_filtering(self, channel):
        """Test filtered history retrieval."""
        await channel.join("claude")
        await channel.join("gpt4")

        await channel.broadcast("claude", "Proposal 1", MessageType.PROPOSAL)
        await channel.send("gpt4", "claude", "Query", MessageType.QUERY)
        await channel.broadcast("claude", "Proposal 2", MessageType.PROPOSAL)

        # Filter by sender
        claude_msgs = channel.get_history(sender="claude")
        assert len(claude_msgs) == 2

        # Filter by type
        proposals = channel.get_history(message_type=MessageType.PROPOSAL)
        assert len(proposals) == 2

    @pytest.mark.asyncio
    async def test_reply_to_threading(self, channel):
        """Test message threading with reply_to."""
        await channel.join("claude")
        await channel.join("gpt4")

        root = await channel.broadcast("claude", "Original proposal")
        await channel.broadcast("gpt4", "I have concerns", reply_to=root.message_id)
        await channel.broadcast("claude", "Let me address that", reply_to=root.message_id)

        thread = channel.get_thread(root.message_id)
        assert len(thread) == 3

    @pytest.mark.asyncio
    async def test_message_handler(self, channel):
        """Test message handler callback."""
        await channel.join("claude")
        await channel.join("gpt4")

        received_messages = []

        async def handler(msg: ChannelMessage):
            received_messages.append(msg)

        channel.on_message("gpt4", handler)

        await channel.broadcast("claude", "Test message")

        # Give handler time to run
        await asyncio.sleep(0.01)

        assert len(received_messages) == 1
        assert received_messages[0].sender == "claude"

    @pytest.mark.asyncio
    async def test_to_context(self, channel):
        """Test converting history to context string."""
        await channel.join("claude")
        await channel.join("gpt4")

        await channel.broadcast("claude", "I propose X")
        await channel.send("gpt4", "claude", "What about Y?")

        context = channel.to_context(limit=10)

        assert "## Recent Agent Discussion" in context
        assert "[claude]" in context
        assert "[gpt4 -> claude]" in context
        assert "I propose X" in context

    @pytest.mark.asyncio
    async def test_close_channel(self, channel):
        """Test closing a channel."""
        await channel.join("claude")
        await channel.join("gpt4")

        await channel.close()

        # Can't join after close
        result = await channel.join("gemini")
        assert result is False

    @pytest.mark.asyncio
    async def test_max_history_limit(self):
        """Test history limit enforcement."""
        channel = AgentChannel("test", max_history=5)
        await channel.join("claude")
        await channel.join("gpt4")

        # Send more than max_history messages
        for i in range(10):
            await channel.broadcast("claude", f"Message {i}")

        # Should only keep last 5
        assert len(channel.history) == 5
        assert channel.history[0].content == "Message 5"
        assert channel.history[4].content == "Message 9"


class TestChannelManager:
    """Tests for ChannelManager."""

    @pytest.fixture
    def manager(self):
        """Create a test manager."""
        return ChannelManager()

    @pytest.mark.asyncio
    async def test_create_channel(self, manager):
        """Test creating a channel."""
        channel = await manager.create_channel("debate_123")

        assert channel.channel_id == "debate_123"
        assert "debate_123" in manager.list_channels()

    @pytest.mark.asyncio
    async def test_get_existing_channel(self, manager):
        """Test getting an existing channel returns same instance."""
        channel1 = await manager.create_channel("debate_123")
        channel2 = await manager.create_channel("debate_123")

        assert channel1 is channel2

    @pytest.mark.asyncio
    async def test_get_channel(self, manager):
        """Test getting a channel by ID."""
        await manager.create_channel("debate_123")

        channel = await manager.get_channel("debate_123")
        assert channel is not None
        assert channel.channel_id == "debate_123"

        missing = await manager.get_channel("nonexistent")
        assert missing is None

    @pytest.mark.asyncio
    async def test_close_channel(self, manager):
        """Test closing a channel."""
        await manager.create_channel("debate_123")

        result = await manager.close_channel("debate_123")
        assert result is True
        assert "debate_123" not in manager.list_channels()

    @pytest.mark.asyncio
    async def test_close_nonexistent_channel(self, manager):
        """Test closing nonexistent channel."""
        result = await manager.close_channel("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_list_channels(self, manager):
        """Test listing channels."""
        await manager.create_channel("debate_1")
        await manager.create_channel("debate_2")
        await manager.create_channel("debate_3")

        channels = manager.list_channels()

        assert len(channels) == 3
        assert "debate_1" in channels
        assert "debate_2" in channels
        assert "debate_3" in channels

    @pytest.mark.asyncio
    async def test_broadcast_to_all(self, manager):
        """Test broadcasting to all channels."""
        channel1 = await manager.create_channel("debate_1")
        channel2 = await manager.create_channel("debate_2")

        await channel1.join("claude")
        await channel1.join("gpt4")
        await channel2.join("claude")
        await channel2.join("gemini")

        count = await manager.broadcast_to_all(
            sender="claude",
            content="Hello all channels!",
        )

        assert count == 2


class TestSingleton:
    """Tests for singleton pattern."""

    def setup_method(self):
        """Reset singleton before each test."""
        reset_channel_manager()

    def test_get_channel_manager(self):
        """Test singleton getter."""
        mgr1 = get_channel_manager()
        mgr2 = get_channel_manager()

        assert mgr1 is mgr2

    def test_reset_channel_manager(self):
        """Test singleton reset."""
        mgr1 = get_channel_manager()
        reset_channel_manager()
        mgr2 = get_channel_manager()

        assert mgr1 is not mgr2


class TestMessageTypes:
    """Tests for different message types."""

    @pytest.mark.asyncio
    async def test_proposal_message(self):
        """Test sending proposal messages."""
        channel = AgentChannel("test")
        await channel.join("claude")
        await channel.join("gpt4")

        msg = await channel.broadcast(
            sender="claude",
            content="I propose using leaky bucket",
            message_type=MessageType.PROPOSAL,
            metadata={"confidence": 0.85},
        )

        assert msg.message_type == MessageType.PROPOSAL
        assert msg.metadata["confidence"] == 0.85

    @pytest.mark.asyncio
    async def test_critique_message(self):
        """Test sending critique messages."""
        channel = AgentChannel("test")
        await channel.join("claude")
        await channel.join("gpt4")

        proposal = await channel.broadcast("claude", "Proposal", MessageType.PROPOSAL)
        critique = await channel.broadcast(
            "gpt4",
            "This has issues",
            MessageType.CRITIQUE,
            reply_to=proposal.message_id,
        )

        assert critique.message_type == MessageType.CRITIQUE
        assert critique.reply_to == proposal.message_id

    @pytest.mark.asyncio
    async def test_query_response_flow(self):
        """Test query-response message flow."""
        channel = AgentChannel("test")
        await channel.join("claude")
        await channel.join("gpt4")

        query = await channel.send(
            "claude",
            "gpt4",
            "What about edge cases?",
            MessageType.QUERY,
        )

        response = await channel.send(
            "gpt4",
            "claude",
            "Good point, let me address that",
            MessageType.RESPONSE,
            reply_to=query.message_id,
        )

        assert query.message_type == MessageType.QUERY
        assert response.message_type == MessageType.RESPONSE
        assert response.reply_to == query.message_id

    @pytest.mark.asyncio
    async def test_signal_message(self):
        """Test signal messages."""
        channel = AgentChannel("test")
        await channel.join("claude")
        await channel.join("gpt4")

        signal = await channel.broadcast(
            "claude",
            "ready",
            MessageType.SIGNAL,
            metadata={"phase": "voting"},
        )

        assert signal.message_type == MessageType.SIGNAL
        assert signal.metadata["phase"] == "voting"
