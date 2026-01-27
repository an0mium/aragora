"""
Channel Integration for Arena.

Provides integration between AgentChannel and the Arena orchestrator,
enabling peer-to-peer messaging between agents during debates.

Usage:
    from aragora.debate.channel_integration import ChannelIntegration

    # In Arena initialization
    self._channel_integration = ChannelIntegration(
        debate_id=self.debate_id,
        agents=self.agents,
        protocol=self.protocol,
    )

    # Before debate rounds
    await self._channel_integration.setup()

    # Get channel context for prompts
    context = self._channel_integration.get_context_for_prompt()

    # After debate
    await self._channel_integration.teardown()
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

from aragora.debate.agent_channel import (
    AgentChannel,
    ChannelManager,
    ChannelMessage,
    MessageType,
    get_channel_manager,
)

if TYPE_CHECKING:
    from aragora.core import Agent
    from aragora.debate.protocol import DebateProtocol

logger = logging.getLogger(__name__)


class ChannelIntegration:
    """
    Integrates AgentChannel with Arena for peer messaging.

    Provides:
    - Automatic channel setup/teardown
    - Agent registration
    - Context injection for prompts
    - Message broadcasting hooks
    """

    def __init__(
        self,
        debate_id: str,
        agents: list["Agent"],
        protocol: Optional["DebateProtocol"] = None,
    ):
        """
        Initialize channel integration.

        Args:
            debate_id: Unique debate identifier
            agents: List of participating agents
            protocol: DebateProtocol with channel settings
        """
        self._debate_id = debate_id
        self._agents = agents
        self._protocol = protocol
        self._channel: Optional[AgentChannel] = None
        self._manager: ChannelManager = get_channel_manager()

        # Check if channels are enabled
        self._enabled = True
        if protocol and hasattr(protocol, "enable_agent_channels"):
            self._enabled = protocol.enable_agent_channels

        # Get max history from protocol
        self._max_history = 100
        if protocol and hasattr(protocol, "agent_channel_max_history"):
            self._max_history = protocol.agent_channel_max_history

    @property
    def enabled(self) -> bool:
        """Check if channel integration is enabled."""
        return self._enabled

    @property
    def channel(self) -> Optional[AgentChannel]:
        """Get the debate channel."""
        return self._channel

    async def setup(self) -> bool:
        """
        Set up the channel and register agents.

        Returns:
            True if setup succeeded
        """
        if not self._enabled:
            logger.debug(f"Channel integration disabled for debate {self._debate_id}")
            return False

        try:
            # Create channel
            self._channel = await self._manager.create_channel(
                self._debate_id,
                max_history=self._max_history,
            )

            # Register all agents
            for agent in self._agents:
                agent_name = getattr(agent, "name", str(agent))
                await self._channel.join(agent_name)

            logger.info(
                f"Channel setup complete for debate {self._debate_id} "
                f"with {len(self._agents)} agents"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to setup channel for {self._debate_id}: {e}")
            return False

    async def teardown(self) -> None:
        """Tear down the channel after debate."""
        if self._channel:
            await self._manager.close_channel(self._debate_id)
            self._channel = None
            logger.debug(f"Channel teardown complete for debate {self._debate_id}")

    async def broadcast_proposal(
        self,
        agent_name: str,
        proposal_content: str,
        round_number: int,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Optional[ChannelMessage]:
        """
        Broadcast a proposal to all agents.

        Args:
            agent_name: Agent making the proposal
            proposal_content: Proposal text
            round_number: Current round number
            metadata: Optional additional metadata

        Returns:
            The sent message, or None if channel not available
        """
        if not self._channel:
            return None

        msg_metadata = {"round": round_number, **(metadata or {})}

        return await self._channel.broadcast(
            sender=agent_name,
            content=proposal_content,
            message_type=MessageType.PROPOSAL,
            metadata=msg_metadata,
        )

    async def broadcast_critique(
        self,
        agent_name: str,
        critique_content: str,
        target_proposal_id: Optional[str] = None,
        round_number: int = 0,
    ) -> Optional[ChannelMessage]:
        """
        Broadcast a critique to all agents.

        Args:
            agent_name: Agent making the critique
            critique_content: Critique text
            target_proposal_id: Optional ID of proposal being critiqued
            round_number: Current round number

        Returns:
            The sent message
        """
        if not self._channel:
            return None

        return await self._channel.broadcast(
            sender=agent_name,
            content=critique_content,
            message_type=MessageType.CRITIQUE,
            metadata={"round": round_number},
            reply_to=target_proposal_id,
        )

    async def send_query(
        self,
        sender: str,
        recipient: str,
        query: str,
    ) -> Optional[ChannelMessage]:
        """
        Send a direct query to another agent.

        Args:
            sender: Agent asking the question
            recipient: Agent being asked
            query: Question text

        Returns:
            The sent message
        """
        if not self._channel:
            return None

        return await self._channel.send(
            sender=sender,
            recipient=recipient,
            content=query,
            message_type=MessageType.QUERY,
        )

    async def signal_ready(self, agent_name: str, phase: str) -> Optional[ChannelMessage]:
        """
        Signal that an agent is ready for a phase.

        Args:
            agent_name: Agent signaling
            phase: Phase name (e.g., "voting", "revision")

        Returns:
            The sent signal
        """
        if not self._channel:
            return None

        return await self._channel.broadcast(
            sender=agent_name,
            content="ready",
            message_type=MessageType.SIGNAL,
            metadata={"phase": phase},
        )

    def get_context_for_prompt(self, limit: int = 5) -> str:
        """
        Get recent channel messages as context for prompts.

        Args:
            limit: Maximum messages to include

        Returns:
            Formatted context string
        """
        if not self._channel:
            return ""

        return self._channel.to_context(limit=limit)

    def get_agent_messages(
        self,
        agent_name: str,
        limit: int = 10,
    ) -> list[ChannelMessage]:
        """
        Get messages from a specific agent.

        Args:
            agent_name: Agent to get messages from
            limit: Maximum messages

        Returns:
            List of messages
        """
        if not self._channel:
            return []

        return self._channel.get_history(limit=limit, sender=agent_name)

    def get_proposals(self, limit: int = 10) -> list[ChannelMessage]:
        """Get recent proposal messages."""
        if not self._channel:
            return []

        return self._channel.get_history(
            limit=limit,
            message_type=MessageType.PROPOSAL,
        )

    def get_critiques(self, limit: int = 10) -> list[ChannelMessage]:
        """Get recent critique messages."""
        if not self._channel:
            return []

        return self._channel.get_history(
            limit=limit,
            message_type=MessageType.CRITIQUE,
        )

    async def inject_discussion_context(
        self,
        prompt: str,
        max_messages: int = 5,
    ) -> str:
        """
        Inject recent discussion context into a prompt.

        Args:
            prompt: Original prompt
            max_messages: Maximum messages to include

        Returns:
            Prompt with discussion context prepended
        """
        context = self.get_context_for_prompt(limit=max_messages)

        if not context:
            return prompt

        return f"{context}\n\n---\n\n{prompt}"


# Factory function for Arena integration
def create_channel_integration(
    debate_id: str,
    agents: list["Agent"],
    protocol: Optional["DebateProtocol"] = None,
) -> ChannelIntegration:
    """
    Create a channel integration instance.

    Args:
        debate_id: Debate identifier
        agents: Participating agents
        protocol: Optional protocol with settings

    Returns:
        ChannelIntegration instance
    """
    return ChannelIntegration(debate_id, agents, protocol)


__all__ = [
    "ChannelIntegration",
    "create_channel_integration",
]
