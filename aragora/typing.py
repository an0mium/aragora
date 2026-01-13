"""
Type definitions and protocols for Aragora.

Provides Protocol classes for duck-typed interfaces, enabling better
type checking without requiring concrete inheritance relationships.

Usage:
    from aragora.typing import AgentProtocol, MemoryProtocol

    def process_agent(agent: AgentProtocol) -> str:
        return agent.respond("prompt")

    # Any object with name attribute and respond method will satisfy the protocol
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Protocol,
    TypeVar,
    runtime_checkable,
)

# =============================================================================
# Type Variables
# =============================================================================

T = TypeVar("T")
AgentT = TypeVar("AgentT", bound="AgentProtocol")
MemoryT = TypeVar("MemoryT", bound="MemoryProtocol")


# =============================================================================
# Agent Protocols
# =============================================================================


@runtime_checkable
class AgentProtocol(Protocol):
    """Protocol for agent implementations.

    All agents must have a name and be able to respond to prompts.
    Streaming is optional but recommended for long responses.

    Example:
        class MyAgent:
            name = "my-agent"

            async def respond(self, prompt: str, context: Optional[str] = None) -> str:
                return "Response"

        agent: AgentProtocol = MyAgent()  # Type checks correctly
    """

    name: str

    async def respond(self, prompt: str, context: Optional[str] = None) -> str:
        """Generate a response to the given prompt."""
        ...


@runtime_checkable
class StreamingAgentProtocol(AgentProtocol, Protocol):
    """Protocol for agents that support streaming responses."""

    async def stream(self, prompt: str, context: Optional[str] = None) -> AsyncIterator[str]:
        """Stream response tokens."""
        ...


@runtime_checkable
class ToolUsingAgentProtocol(AgentProtocol, Protocol):
    """Protocol for agents that can use tools."""

    available_tools: List[str]

    async def respond_with_tools(
        self,
        prompt: str,
        tools: List[Dict[str, Any]],
        context: Optional[str] = None,
    ) -> str:
        """Generate response with tool use."""
        ...


# =============================================================================
# Memory Protocols
# =============================================================================


@runtime_checkable
class MemoryProtocol(Protocol):
    """Protocol for memory backends.

    Provides basic store/query interface for memory systems.
    """

    def store(self, content: str, **kwargs: Any) -> str:
        """Store content and return an identifier."""
        ...

    def query(self, **kwargs: Any) -> List[Any]:
        """Query stored content."""
        ...


@runtime_checkable
class TieredMemoryProtocol(MemoryProtocol, Protocol):
    """Protocol for tiered memory systems like ContinuumMemory."""

    def store(
        self,
        content: str,
        tier: Any = None,  # MemoryTier, optional for protocol compatibility
        importance: float = 0.5,
        **kwargs: Any,
    ) -> str:
        """Store content in a specific tier."""
        ...

    def query(
        self,
        tier: Optional[Any] = None,
        limit: int = 10,
        min_importance: float = 0.0,
        **kwargs: Any,
    ) -> List[Any]:
        """Query content from specified tier."""
        ...

    def promote(self, entry_id: str, target_tier: Any) -> bool:
        """Promote entry to a faster tier."""
        ...

    def demote(self, entry_id: str, target_tier: Any) -> bool:
        """Demote entry to a slower tier."""
        ...


@runtime_checkable
class CritiqueStoreProtocol(Protocol):
    """Protocol for critique/pattern storage."""

    def store_pattern(self, critique: Any, resolution: str) -> str:
        """Store a critique pattern with its resolution."""
        ...

    def retrieve_patterns(
        self,
        issue_type: Optional[str] = None,
        limit: int = 10,
    ) -> List[Any]:
        """Retrieve stored patterns."""
        ...

    def get_reputation(self, agent: str) -> Dict[str, Any]:
        """Get reputation data for an agent."""
        ...


# =============================================================================
# Event Protocols
# =============================================================================


@runtime_checkable
class EventEmitterProtocol(Protocol):
    """Protocol for event emission systems."""

    def emit(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit an event with data."""
        ...

    def on(self, event_type: str, callback: Callable[..., Any]) -> None:
        """Register an event listener."""
        ...


@runtime_checkable
class AsyncEventEmitterProtocol(Protocol):
    """Protocol for async event emission."""

    async def emit_async(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit an event asynchronously."""
        ...


# =============================================================================
# Handler Protocols
# =============================================================================


@runtime_checkable
class HandlerProtocol(Protocol):
    """Protocol for HTTP endpoint handlers."""

    def can_handle(self, path: str) -> bool:
        """Check if handler can process this path."""
        ...

    def handle(
        self,
        path: str,
        query: Dict[str, Any],
        request_handler: Any,
    ) -> Optional[Any]:
        """Handle the request and return result."""
        ...


@runtime_checkable
class BaseHandlerProtocol(HandlerProtocol, Protocol):
    """Extended handler protocol with common patterns."""

    ROUTES: List[str]
    ctx: Dict[str, Any]

    def read_json_body(self, handler: Any) -> Optional[Dict[str, Any]]:
        """Read and parse JSON body from request."""
        ...


# =============================================================================
# Debate Protocols
# =============================================================================


@runtime_checkable
class DebateResultProtocol(Protocol):
    """Protocol for debate results."""

    rounds: int
    consensus_reached: bool
    final_answer: Optional[str]
    messages: List[Any]


@runtime_checkable
class ConsensusDetectorProtocol(Protocol):
    """Protocol for consensus detection."""

    def check_consensus(
        self,
        votes: List[Any],
        threshold: float = 0.5,
    ) -> bool:
        """Check if consensus has been reached."""
        ...

    def get_winner(self, votes: List[Any]) -> Optional[str]:
        """Get the winning choice if any."""
        ...


# =============================================================================
# Ranking Protocols
# =============================================================================


@runtime_checkable
class RankingSystemProtocol(Protocol):
    """Protocol for agent ranking systems."""

    def get_rating(self, agent: str) -> float:
        """Get rating for an agent."""
        ...

    def record_match(
        self,
        agent_a: str,
        agent_b: str,
        scores: Dict[str, float],
        context: str,
    ) -> None:
        """Record a match result."""
        ...

    def get_leaderboard(self, limit: int = 10) -> List[Any]:
        """Get top agents by rating."""
        ...


# =============================================================================
# Storage Protocols
# =============================================================================


@runtime_checkable
class DebateStorageProtocol(Protocol):
    """Protocol for debate storage backends."""

    def save_debate(self, debate_id: str, data: Dict[str, Any]) -> None:
        """Save debate data."""
        ...

    def load_debate(self, debate_id: str) -> Optional[Dict[str, Any]]:
        """Load debate data."""
        ...

    def list_debates(self, limit: int = 100, org_id: Optional[str] = None) -> List[Any]:
        """List available debates. Returns list of debate metadata objects."""
        ...

    def delete_debate(self, debate_id: str) -> bool:
        """Delete a debate."""
        ...

    def get_debate(self, debate_id: str) -> Optional[Dict[str, Any]]:
        """Get debate by ID."""
        ...

    def get_debate_by_slug(self, slug: str) -> Optional[Dict[str, Any]]:
        """Get debate by slug."""
        ...

    def get_by_id(self, debate_id: str) -> Optional[Dict[str, Any]]:
        """Get debate by ID (alias)."""
        ...

    def get_by_slug(self, slug: str) -> Optional[Dict[str, Any]]:
        """Get debate by slug (alias)."""
        ...

    def list_recent(self, limit: int = 20, org_id: Optional[str] = None) -> List[Any]:
        """List recent debates."""
        ...

    def search(
        self,
        query: Optional[str] = None,
        agent: Optional[str] = None,
        min_confidence: Optional[float] = None,
        limit: int = 20,
        org_id: Optional[str] = None,
    ) -> List[Any]:
        """Search debates."""
        ...


@runtime_checkable
class UserStoreProtocol(Protocol):
    """Protocol for user storage backends."""

    def get_user_by_id(self, user_id: str) -> Optional[Any]:
        """Get user by ID."""
        ...

    def get_user_by_email(self, email: str) -> Optional[Any]:
        """Get user by email."""
        ...

    def create_user(
        self,
        email: str,
        password_hash: str,
        password_salt: str,
        **kwargs: Any,
    ) -> Any:
        """Create a new user."""
        ...

    def update_user(self, user_id: str, **kwargs: Any) -> bool:
        """Update user attributes."""
        ...


# =============================================================================
# Verification Protocols
# =============================================================================


@runtime_checkable
class VerificationBackendProtocol(Protocol):
    """Protocol for formal verification backends."""

    @property
    def is_available(self) -> bool:
        """Check if backend is available."""
        ...

    def can_verify(self, claim: str, claim_type: Optional[str] = None) -> bool:
        """Check if backend can verify this claim type."""
        ...

    async def translate(self, claim: str) -> str:
        """Translate natural language claim to formal statement."""
        ...

    async def prove(self, formal_statement: str) -> Any:
        """Attempt to prove the formal statement."""
        ...


# =============================================================================
# Callback Types
# =============================================================================

# Event callback type
EventCallback = Callable[[str, Dict[str, Any]], None]

# Async event callback type
AsyncEventCallback = Callable[[str, Dict[str, Any]], Any]

# Response filter type
ResponseFilter = Callable[[str], str]

# Vote callback type
VoteCallback = Callable[[Any], None]


# =============================================================================
# Result Types
# =============================================================================


@dataclass
class Result(Generic[T]):
    """Generic result type for operations that can fail.

    Example:
        def get_user(id: str) -> Result[User]:
            user = db.get(id)
            if user:
                return Result(success=True, value=user)
            return Result(success=False, error="User not found")
    """

    success: bool
    value: Optional[T] = None
    error: Optional[str] = None

    @classmethod
    def ok(cls, value: T) -> "Result[T]":
        """Create successful result."""
        return cls(success=True, value=value)

    @classmethod
    def fail(cls, error: str) -> "Result[T]":
        """Create failed result."""
        return cls(success=False, error=error)




__all__ = [
    # Type variables
    "T",
    "AgentT",
    "MemoryT",
    # Agent protocols
    "AgentProtocol",
    "StreamingAgentProtocol",
    "ToolUsingAgentProtocol",
    # Memory protocols
    "MemoryProtocol",
    "TieredMemoryProtocol",
    "CritiqueStoreProtocol",
    # Event protocols
    "EventEmitterProtocol",
    "AsyncEventEmitterProtocol",
    # Handler protocols
    "HandlerProtocol",
    "BaseHandlerProtocol",
    # Debate protocols
    "DebateResultProtocol",
    "ConsensusDetectorProtocol",
    # Ranking protocols
    "RankingSystemProtocol",
    # Storage protocols
    "DebateStorageProtocol",
    "UserStoreProtocol",
    # Verification protocols
    "VerificationBackendProtocol",
    # Callback types
    "EventCallback",
    "AsyncEventCallback",
    "ResponseFilter",
    "VoteCallback",
    # Result types
    "Result",
]
