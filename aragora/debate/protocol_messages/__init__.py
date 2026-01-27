"""
Protocol Messages for Aragora Debates.

Provides typed, queryable structured messages for debate protocol events.
Inspired by gastown's protocol message pattern for inter-agent coordination.

Usage:
    from aragora.debate.protocol_messages import (
        ProtocolMessage,
        ProtocolMessageType,
        ProtocolMessageStore,
    )

    # Create and store a protocol message
    store = ProtocolMessageStore()
    msg = ProtocolMessage(
        message_type=ProtocolMessageType.PROPOSAL_SUBMITTED,
        debate_id="debate-123",
        agent_id="claude-opus",
        payload={"content": "My proposal..."},
    )
    await store.record(msg)

    # Query messages
    proposals = await store.query(
        debate_id="debate-123",
        message_type=ProtocolMessageType.PROPOSAL_SUBMITTED,
    )
"""

from .messages import (
    ProtocolMessage,
    ProtocolMessageType,
    ProtocolPayload,
    ProposalPayload,
    CritiquePayload,
    VotePayload,
    ConsensusPayload,
    RoundPayload,
    AgentEventPayload,
)
from .store import ProtocolMessageStore, get_protocol_store
from .handlers import ProtocolHandler, ProtocolHandlerRegistry

__all__ = [
    # Message types
    "ProtocolMessage",
    "ProtocolMessageType",
    "ProtocolPayload",
    "ProposalPayload",
    "CritiquePayload",
    "VotePayload",
    "ConsensusPayload",
    "RoundPayload",
    "AgentEventPayload",
    # Store
    "ProtocolMessageStore",
    "get_protocol_store",
    # Handlers
    "ProtocolHandler",
    "ProtocolHandlerRegistry",
]
