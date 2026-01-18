"""
Protocol Integrations for Aragora.

Supports:
- MCP (Model Context Protocol) for tool access
- A2A (Agent-to-Agent Protocol) for inter-agent communication
- Core protocols for debate, agents, and memory
"""

from aragora.protocols.a2a import (
    A2AClient,
    A2AServer,
    AgentCard,
    TaskRequest,
    TaskResult,
    TaskStatus,
)
from aragora.protocols.bridge import ProtocolBridge

# Re-export core protocols from aragora.core_protocols
from aragora.core_protocols import (
    Agent,
    AgentRating,
    AuthenticatedUser,
    ConsensusBackend,
    CritiqueBackend,
    EloBackend,
    EmbeddingBackend,
    GenesisBackend,
    HTTPHeaders,
    HTTPRequestHandler,
    MemoryBackend,
    PersonaBackend,
    StorageBackend,
)

__all__ = [
    # A2A
    "A2AClient",
    "A2AServer",
    "AgentCard",
    "TaskRequest",
    "TaskResult",
    "TaskStatus",
    # Bridge
    "ProtocolBridge",
    # Core protocols
    "Agent",
    "AgentRating",
    "AuthenticatedUser",
    "ConsensusBackend",
    "CritiqueBackend",
    "EloBackend",
    "EmbeddingBackend",
    "GenesisBackend",
    "HTTPHeaders",
    "HTTPRequestHandler",
    "MemoryBackend",
    "PersonaBackend",
    "StorageBackend",
]
