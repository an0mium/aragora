"""Core utilities and shared services for Aragora.

This package re-exports core types from aragora.core_types and adds
additional utilities like the unified embedding service and decision router.

Core types (from aragora/core_types.py):
    - Message, Critique, Vote, DebateResult, Environment, Agent
    - TaskComplexity, AgentRole, AgentStance
    - DisagreementReport

Embeddings (from aragora.core.embeddings):
    - UnifiedEmbeddingService, get_embedding_service
    - EmbeddingConfig, EmbeddingResult, EmbeddingBackend

Decision (from aragora.core.decision):
    - DecisionRequest, DecisionResult, DecisionRouter
    - DecisionType, InputSource, Priority
    - ResponseChannel, RequestContext, DecisionConfig
"""

from typing import Any

# Import all exports from the core_types module (formerly core.py)
from aragora.core_types import (
    AgentRole,
    AgentStance,
    TaskComplexity,
    Message,
    Critique,
    Vote,
    DisagreementReport,
    DebateResult,
    Environment,
    Agent,
)

# Import embeddings
from aragora.core.embeddings import (
    EmbeddingBackend,
    EmbeddingConfig,
    EmbeddingResult,
    UnifiedEmbeddingService,
    get_embedding_service,
)

# Import decision routing
from aragora.core.decision import (
    DecisionConfig,
    DecisionRequest,
    DecisionResult,
    DecisionRouter,
    DecisionType,
    InputSource,
    Priority,
    RequestContext,
    ResponseChannel,
    ResponseFormat,
    get_decision_router,
)

__all__ = [
    # Core types
    "AgentRole",
    "AgentStance",
    "TaskComplexity",
    "Message",
    "Critique",
    "Vote",
    "DisagreementReport",
    "DebateResult",
    "Environment",
    "Agent",
    "DebateProtocol",  # Lazy import
    # Embeddings
    "EmbeddingBackend",
    "EmbeddingConfig",
    "EmbeddingResult",
    "UnifiedEmbeddingService",
    "get_embedding_service",
    # Decision routing
    "DecisionConfig",
    "DecisionRequest",
    "DecisionResult",
    "DecisionRouter",
    "DecisionType",
    "InputSource",
    "Priority",
    "RequestContext",
    "ResponseChannel",
    "ResponseFormat",
    "get_decision_router",
]


# Lazy imports for backwards compatibility
def __getattr__(name: str) -> Any:
    if name == "DebateProtocol":
        from aragora.debate.protocol import DebateProtocol

        return DebateProtocol
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
