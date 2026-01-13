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
# Tracker Protocols
# =============================================================================


@runtime_checkable
class EloSystemProtocol(Protocol):
    """Protocol for ELO rating systems.

    More specific than RankingSystemProtocol, tailored for ELO-style ratings
    with win/loss recording and domain-specific tracking.
    """

    def get_rating(self, agent: str, domain: str = "") -> float:
        """Get ELO rating for an agent, optionally in a specific domain."""
        ...

    def record_match(
        self,
        winner: str,
        loser: str,
        domain: str = "",
        margin: float = 1.0,
    ) -> None:
        """Record a match result between two agents."""
        ...

    def get_leaderboard(self, limit: int = 10, domain: str = "") -> List[Any]:
        """Get top agents by ELO rating."""
        ...

    def get_match_history(self, agent: str, limit: int = 20) -> List[Any]:
        """Get recent match history for an agent."""
        ...


@runtime_checkable
class CalibrationTrackerProtocol(Protocol):
    """Protocol for prediction calibration tracking.

    Tracks how well-calibrated an agent's confidence scores are
    by comparing predicted confidence to actual outcomes.
    """

    def get_calibration(self, agent: str) -> Optional[Dict[str, Any]]:
        """Get calibration data for an agent."""
        ...

    def record_prediction(
        self,
        agent: str,
        confidence: float,
        correct: bool,
        domain: str = "",
    ) -> None:
        """Record a prediction with its outcome."""
        ...

    def get_calibration_score(self, agent: str) -> float:
        """Get overall calibration score (0-1, lower is better calibrated)."""
        ...


@runtime_checkable
class PositionLedgerProtocol(Protocol):
    """Protocol for tracking agent positions across debates.

    Records what positions agents take on claims, enabling
    consistency tracking and position evolution analysis.
    """

    def record_position(
        self,
        agent_name: str,
        claim: str,
        stance: str,
        confidence: float,
        debate_id: str,
        round_num: int,
        domain: Optional[str] = None,
    ) -> None:
        """Record an agent's position on a claim."""
        ...

    def get_positions(
        self,
        agent_name: str,
        limit: int = 10,
        claim_filter: Optional[str] = None,
    ) -> List[Any]:
        """Get recent positions for an agent."""
        ...

    def get_consistency_score(self, agent_name: str) -> float:
        """Get position consistency score for an agent."""
        ...


@runtime_checkable
class RelationshipTrackerProtocol(Protocol):
    """Protocol for agent relationship tracking.

    Tracks agreement patterns between agent pairs,
    enabling alliance detection and relationship analysis.
    """

    def get_relationship(self, agent_a: str, agent_b: str) -> Optional[Dict[str, Any]]:
        """Get relationship data between two agents."""
        ...

    def update_relationship(
        self,
        agent_a: str,
        agent_b: str,
        outcome: str,
        debate_id: str = "",
    ) -> None:
        """Update relationship based on debate outcome."""
        ...

    def get_allies(self, agent: str, threshold: float = 0.6) -> List[str]:
        """Get agents that frequently agree with the given agent."""
        ...

    def get_adversaries(self, agent: str, threshold: float = 0.6) -> List[str]:
        """Get agents that frequently disagree with the given agent."""
        ...


@runtime_checkable
class MomentDetectorProtocol(Protocol):
    """Protocol for significant moment detection.

    Identifies important moments in debates such as
    breakthroughs, conflicts, or consensus shifts.
    """

    def detect_moment(
        self,
        content: str,
        context: Dict[str, Any],
        threshold: float = 0.7,
    ) -> Optional[Dict[str, Any]]:
        """Detect if content represents a significant moment."""
        ...

    def get_moment_types(self) -> List[str]:
        """Get list of moment types this detector can identify."""
        ...


@runtime_checkable
class PersonaManagerProtocol(Protocol):
    """Protocol for agent persona management.

    Manages persistent personas for agents, including
    communication style, expertise areas, and traits.
    """

    def get_persona(self, agent_name: str) -> Optional[Dict[str, Any]]:
        """Get persona configuration for an agent."""
        ...

    def update_persona(self, agent_name: str, updates: Dict[str, Any]) -> None:
        """Update persona attributes for an agent."""
        ...

    def get_context_for_prompt(self, agent_name: str) -> str:
        """Get persona context string for prompt injection."""
        ...


@runtime_checkable
class DissentRetrieverProtocol(Protocol):
    """Protocol for retrieving dissenting positions.

    Finds historical dissenting opinions relevant to current debates.
    """

    def retrieve_dissent(
        self,
        topic: str,
        limit: int = 5,
        min_relevance: float = 0.5,
    ) -> List[Any]:
        """Retrieve relevant dissenting positions."""
        ...

    def store_dissent(
        self,
        agent: str,
        position: str,
        debate_id: str,
        context: str = "",
    ) -> str:
        """Store a dissenting position for future retrieval."""
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
    # Tracker protocols
    "EloSystemProtocol",
    "CalibrationTrackerProtocol",
    "PositionLedgerProtocol",
    "RelationshipTrackerProtocol",
    "MomentDetectorProtocol",
    "PersonaManagerProtocol",
    "DissentRetrieverProtocol",
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
