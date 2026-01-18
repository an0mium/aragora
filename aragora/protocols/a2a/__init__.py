"""
A2A (Agent-to-Agent Protocol) Implementation.

Implements the A2A protocol for inter-agent communication,
allowing Aragora agents to be discovered and invoked by external systems.

Based on the Linux Foundation A2A specification:
https://github.com/a2aproject/A2A

Key concepts:
- AgentCard: Describes an agent's capabilities
- TaskRequest: Request for an agent to perform work
- TaskResult: Result of agent work
"""

from aragora.protocols.a2a.types import (
    AgentCard,
    AgentCapability,
    TaskRequest,
    TaskResult,
    TaskStatus,
    TaskPriority,
    ContextItem,
    SecurityCard,
)
from aragora.protocols.a2a.client import A2AClient
from aragora.protocols.a2a.server import A2AServer

__all__ = [
    # Types
    "AgentCard",
    "AgentCapability",
    "TaskRequest",
    "TaskResult",
    "TaskStatus",
    "TaskPriority",
    "ContextItem",
    "SecurityCard",
    # Client/Server
    "A2AClient",
    "A2AServer",
]
