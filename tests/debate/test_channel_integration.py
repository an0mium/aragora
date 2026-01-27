"""Tests for Arena channel integration."""

import pytest

from aragora.debate.agent_channel import MessageType, reset_channel_manager
from aragora.debate.channel_integration import (
    ChannelIntegration,
    create_channel_integration,
)


class SimpleAgent:
    """Simple agent for testing."""

    def __init__(self, name: str):
        self.name = name


class SimpleProtocol:
    """Simple protocol for testing."""

    def __init__(
        self,
        enable_agent_channels: bool = True,
        agent_channel_max_history: int = 100,
    ):
        self.enable_agent_channels = enable_agent_channels
        self.agent_channel_max_history = agent_channel_max_history


class TestChannelIntegration:
    """Tests for ChannelIntegration."""

    @pytest.fixture(autouse=True)
    def reset_channels(self):
        """Reset channel manager before each test."""
        reset_channel_manager()
        yield
        reset_channel_manager()

    @pytest.fixture
    def agents(self):
        """Create test agents."""
        return [
            SimpleAgent("claude"),
            SimpleAgent("gpt4"),
            SimpleAgent("gemini"),
        ]

    @pytest.fixture
    def protocol(self):
        """Create test protocol."""
        return SimpleProtocol()

    @pytest.fixture
    def integration(self, agents, protocol):
        """Create channel integration."""
        return ChannelIntegration(
            debate_id="test_debate",
            agents=agents,
            protocol=protocol,
        )

    @pytest.mark.asyncio
    async def test_setup(self, integration):
        """Test channel setup."""
        result = await integration.setup()

        assert result is True
        assert integration.channel is not None
        assert len(integration.channel.agents) == 3

    @pytest.mark.asyncio
    async def test_setup_disabled(self, agents):
        """Test setup when channels are disabled."""
        protocol = SimpleProtocol(enable_agent_channels=False)
        integration = ChannelIntegration(
            debate_id="test",
            agents=agents,
            protocol=protocol,
        )

        result = await integration.setup()

        assert result is False
        assert integration.channel is None

    @pytest.mark.asyncio
    async def test_teardown(self, integration):
        """Test channel teardown."""
        await integration.setup()
        assert integration.channel is not None

        await integration.teardown()
        assert integration.channel is None

    @pytest.mark.asyncio
    async def test_broadcast_proposal(self, integration):
        """Test broadcasting a proposal."""
        await integration.setup()

        msg = await integration.broadcast_proposal(
            agent_name="claude",
            proposal_content="I propose using token bucket algorithm",
            round_number=1,
            metadata={"confidence": 0.85},
        )

        assert msg is not None
        assert msg.sender == "claude"
        assert msg.message_type == MessageType.PROPOSAL
        assert msg.metadata["round"] == 1
        assert msg.metadata["confidence"] == 0.85

    @pytest.mark.asyncio
    async def test_broadcast_critique(self, integration):
        """Test broadcasting a critique."""
        await integration.setup()

        # First broadcast a proposal
        proposal = await integration.broadcast_proposal(
            agent_name="claude",
            proposal_content="Original proposal",
            round_number=1,
        )

        # Then critique it
        critique = await integration.broadcast_critique(
            agent_name="gpt4",
            critique_content="This has issues with edge cases",
            target_proposal_id=proposal.message_id,
            round_number=1,
        )

        assert critique is not None
        assert critique.sender == "gpt4"
        assert critique.message_type == MessageType.CRITIQUE
        assert critique.reply_to == proposal.message_id

    @pytest.mark.asyncio
    async def test_send_query(self, integration):
        """Test sending a direct query."""
        await integration.setup()

        msg = await integration.send_query(
            sender="claude",
            recipient="gpt4",
            query="What do you think about this approach?",
        )

        assert msg is not None
        assert msg.sender == "claude"
        assert msg.recipient == "gpt4"
        assert msg.message_type == MessageType.QUERY

    @pytest.mark.asyncio
    async def test_signal_ready(self, integration):
        """Test signaling readiness."""
        await integration.setup()

        msg = await integration.signal_ready("claude", "voting")

        assert msg is not None
        assert msg.sender == "claude"
        assert msg.message_type == MessageType.SIGNAL
        assert msg.metadata["phase"] == "voting"

    @pytest.mark.asyncio
    async def test_get_context_for_prompt(self, integration):
        """Test getting context for prompts."""
        await integration.setup()

        await integration.broadcast_proposal("claude", "Proposal 1", 1)
        await integration.broadcast_critique("gpt4", "Critique 1", round_number=1)

        context = integration.get_context_for_prompt(limit=5)

        assert "## Recent Agent Discussion" in context
        assert "[claude]" in context
        assert "Proposal 1" in context

    @pytest.mark.asyncio
    async def test_get_context_empty(self, integration):
        """Test getting context when empty."""
        await integration.setup()

        context = integration.get_context_for_prompt()
        assert context == ""

    @pytest.mark.asyncio
    async def test_get_agent_messages(self, integration):
        """Test getting messages from specific agent."""
        await integration.setup()

        await integration.broadcast_proposal("claude", "Proposal 1", 1)
        await integration.broadcast_proposal("claude", "Proposal 2", 2)
        await integration.broadcast_critique("gpt4", "Critique", round_number=1)

        claude_msgs = integration.get_agent_messages("claude", limit=10)

        assert len(claude_msgs) == 2
        assert all(m.sender == "claude" for m in claude_msgs)

    @pytest.mark.asyncio
    async def test_get_proposals(self, integration):
        """Test getting proposal messages."""
        await integration.setup()

        await integration.broadcast_proposal("claude", "Proposal 1", 1)
        await integration.broadcast_critique("gpt4", "Critique", round_number=1)
        await integration.broadcast_proposal("gemini", "Proposal 2", 1)

        proposals = integration.get_proposals(limit=10)

        assert len(proposals) == 2
        assert all(m.message_type == MessageType.PROPOSAL for m in proposals)

    @pytest.mark.asyncio
    async def test_get_critiques(self, integration):
        """Test getting critique messages."""
        await integration.setup()

        await integration.broadcast_proposal("claude", "Proposal", 1)
        await integration.broadcast_critique("gpt4", "Critique 1", round_number=1)
        await integration.broadcast_critique("gemini", "Critique 2", round_number=1)

        critiques = integration.get_critiques(limit=10)

        assert len(critiques) == 2
        assert all(m.message_type == MessageType.CRITIQUE for m in critiques)

    @pytest.mark.asyncio
    async def test_inject_discussion_context(self, integration):
        """Test injecting discussion context into prompt."""
        await integration.setup()

        await integration.broadcast_proposal("claude", "My proposal", 1)

        original_prompt = "You are a helpful assistant."
        enhanced = await integration.inject_discussion_context(
            original_prompt,
            max_messages=5,
        )

        assert "## Recent Agent Discussion" in enhanced
        assert "My proposal" in enhanced
        assert "You are a helpful assistant." in enhanced

    @pytest.mark.asyncio
    async def test_inject_discussion_empty(self, integration):
        """Test injecting context when no messages."""
        await integration.setup()

        original_prompt = "You are a helpful assistant."
        enhanced = await integration.inject_discussion_context(original_prompt)

        assert enhanced == original_prompt

    @pytest.mark.asyncio
    async def test_methods_without_setup(self, integration):
        """Test methods return None/empty when channel not set up."""
        # Don't call setup

        proposal = await integration.broadcast_proposal("claude", "Test", 1)
        assert proposal is None

        critique = await integration.broadcast_critique("gpt4", "Test", round_number=1)
        assert critique is None

        query = await integration.send_query("claude", "gpt4", "Test")
        assert query is None

        signal = await integration.signal_ready("claude", "voting")
        assert signal is None

        context = integration.get_context_for_prompt()
        assert context == ""

        messages = integration.get_agent_messages("claude")
        assert messages == []


class TestChannelIntegrationFactory:
    """Tests for factory function."""

    @pytest.fixture(autouse=True)
    def reset_channels(self):
        """Reset channel manager before each test."""
        reset_channel_manager()
        yield
        reset_channel_manager()

    def test_create_channel_integration(self):
        """Test factory function."""
        agents = [SimpleAgent("claude"), SimpleAgent("gpt4")]
        protocol = SimpleProtocol()

        integration = create_channel_integration(
            debate_id="test_debate",
            agents=agents,
            protocol=protocol,
        )

        assert integration is not None
        assert integration.enabled is True

    def test_create_without_protocol(self):
        """Test factory without protocol."""
        agents = [SimpleAgent("claude")]

        integration = create_channel_integration(
            debate_id="test",
            agents=agents,
            protocol=None,
        )

        assert integration is not None
        assert integration.enabled is True  # Default to enabled


class TestChannelIntegrationWithMaxHistory:
    """Tests for max history configuration."""

    @pytest.fixture(autouse=True)
    def reset_channels(self):
        """Reset channel manager before each test."""
        reset_channel_manager()
        yield
        reset_channel_manager()

    @pytest.mark.asyncio
    async def test_custom_max_history(self):
        """Test custom max history from protocol."""
        agents = [SimpleAgent("claude"), SimpleAgent("gpt4")]
        protocol = SimpleProtocol(agent_channel_max_history=5)

        integration = ChannelIntegration(
            debate_id="test",
            agents=agents,
            protocol=protocol,
        )

        await integration.setup()

        # Send more than max_history messages
        for i in range(10):
            await integration.broadcast_proposal("claude", f"Message {i}", i)

        # Should only keep last 5
        assert len(integration.channel.history) == 5
