"""
Type definitions and protocols for Aragora.

Provides Protocol classes for duck-typed interfaces, enabling better
type checking without requiring concrete inheritance relationships.

Usage:
    from aragora.type_protocols import AgentProtocol, MemoryProtocol

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
    Generic,
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

            async def respond(self, prompt: str, context: str | None = None) -> str:
                return "Response"

        agent: AgentProtocol = MyAgent()  # Type checks correctly
    """

    name: str

    async def respond(self, prompt: str, context: str | None = None) -> str:
        """Generate a response to the given prompt."""
        ...


@runtime_checkable
class StreamingAgentProtocol(AgentProtocol, Protocol):
    """Protocol for agents that support streaming responses."""

    async def stream(self, prompt: str, context: str | None = None) -> AsyncIterator[str]:
        """Stream response tokens."""
        ...


@runtime_checkable
class ToolUsingAgentProtocol(AgentProtocol, Protocol):
    """Protocol for agents that can use tools."""

    available_tools: list[str]

    async def respond_with_tools(
        self,
        prompt: str,
        tools: list[dict[str, Any]],
        context: str | None = None,
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

    def query(self, **kwargs: Any) -> list[Any]:
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
        tier: Any | None = None,
        limit: int = 10,
        min_importance: float = 0.0,
        **kwargs: Any,
    ) -> list[Any]:
        """Query content from specified tier."""
        ...

    def promote(self, entry_id: str, target_tier: Any) -> bool:
        """Promote entry to a faster tier."""
        ...

    def demote(self, entry_id: str, target_tier: Any) -> bool:
        """Demote entry to a slower tier."""
        ...

    def cleanup_expired_memories(self) -> int:
        """Clean up expired memories. Returns count of cleaned entries."""
        ...

    def enforce_tier_limits(self) -> None:
        """Enforce tier size limits by evicting excess entries."""
        ...


@runtime_checkable
class CritiqueStoreProtocol(Protocol):
    """Protocol for critique/pattern storage."""

    def store_pattern(self, critique: Any, resolution: str) -> str:
        """Store a critique pattern with its resolution."""
        ...

    def retrieve_patterns(
        self,
        issue_type: str | None = None,
        limit: int = 10,
    ) -> list[Any]:
        """Retrieve stored patterns."""
        ...

    def get_reputation(self, agent: str) -> dict[str, Any]:
        """Get reputation data for an agent."""
        ...


# =============================================================================
# Event Protocols
# =============================================================================


@runtime_checkable
class EventEmitterProtocol(Protocol):
    """Protocol for event emission systems."""

    def emit(self, event: Any, data: Optional[dict[str, Any]] = None) -> None:
        """Emit an event. Can be called with event object or (event_type, data)."""
        ...

    def on(self, event_type: str, callback: Callable[..., Any]) -> None:
        """Register an event listener."""
        ...


@runtime_checkable
class AsyncEventEmitterProtocol(Protocol):
    """Protocol for async event emission."""

    async def emit_async(self, event_type: str, data: dict[str, Any]) -> None:
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
        query: dict[str, Any],
        request_handler: Any,
    ) -> Any | None:
        """Handle the request and return result."""
        ...


@runtime_checkable
class BaseHandlerProtocol(HandlerProtocol, Protocol):
    """Extended handler protocol with common patterns."""

    ROUTES: list[str]
    ctx: dict[str, Any]

    def read_json_body(self, handler: Any) -> Optional[dict[str, Any]]:
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
    final_answer: str | None
    messages: list[Any]


@runtime_checkable
class ConsensusDetectorProtocol(Protocol):
    """Protocol for consensus detection."""

    def check_consensus(
        self,
        votes: list[Any],
        threshold: float = 0.5,
    ) -> bool:
        """Check if consensus has been reached."""
        ...

    def get_winner(self, votes: list[Any]) -> str | None:
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
        scores: dict[str, float],
        context: str,
    ) -> None:
        """Record a match result."""
        ...

    def get_leaderboard(self, limit: int = 10) -> list[Any]:
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
        debate_id: str,
        participants: list[str],
        scores: dict[str, float],
        domain: str = "",
        winner: str | None = None,
        loser: str | None = None,
        margin: float = 1.0,
    ) -> None:
        """Record a match result. Can use debate_id/participants/scores or winner/loser."""
        ...

    def get_leaderboard(self, limit: int = 10, domain: str = "") -> list[Any]:
        """Get top agents by ELO rating."""
        ...

    def get_match_history(self, agent: str, limit: int = 20) -> list[Any]:
        """Get recent match history for an agent."""
        ...

    def get_ratings_batch(self, agents: list[str]) -> dict[str, Any]:
        """Get ratings for multiple agents in a single call."""
        ...

    def update_voting_accuracy(
        self,
        agent_name: str,
        voted_for_consensus: bool,
        domain: str = "general",
        debate_id: str | None = None,
        apply_elo_bonus: bool = True,
        bonus_k_factor: float = 4.0,
    ) -> float:
        """Update an agent's voting accuracy and optionally apply ELO bonus."""
        ...

    def apply_learning_bonus(
        self,
        agent_name: str,
        domain: str = "general",
        debate_id: str | None = None,
        bonus_factor: float = 0.5,
    ) -> float:
        """Apply ELO bonus based on agent's learning efficiency."""
        ...


@runtime_checkable
class CalibrationTrackerProtocol(Protocol):
    """Protocol for prediction calibration tracking.

    Tracks how well-calibrated an agent's confidence scores are
    by comparing predicted confidence to actual outcomes.
    """

    def get_calibration(self, agent: str) -> Optional[dict[str, Any]]:
        """Get calibration data for an agent."""
        ...

    def record_prediction(
        self,
        agent: str,
        confidence: float,
        correct: bool,
        domain: str = "",
        debate_id: str | None = None,
        prediction_type: str | None = None,
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
        domain: str | None = None,
    ) -> None:
        """Record an agent's position on a claim."""
        ...

    def get_positions(
        self,
        agent_name: str,
        limit: int = 10,
        claim_filter: str | None = None,
    ) -> list[Any]:
        """Get recent positions for an agent."""
        ...

    def get_agent_positions(
        self,
        agent_name: str,
        limit: int = 100,
        outcome_filter: str | None = None,
    ) -> list[Any]:
        """Get positions for an agent with optional outcome filter."""
        ...

    def get_consistency_score(self, agent_name: str) -> float:
        """Get position consistency score for an agent."""
        ...

    def resolve_position(
        self,
        position_id: str | None = None,
        outcome: str | None = None,
        resolution_source: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Record a position resolution outcome."""
        ...


@runtime_checkable
class RelationshipTrackerProtocol(Protocol):
    """Protocol for agent relationship tracking.

    Tracks agreement patterns between agent pairs,
    enabling alliance detection and relationship analysis.
    """

    def get_relationship(self, agent_a: str, agent_b: str) -> Optional[dict[str, Any]]:
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

    def get_allies(self, agent: str, threshold: float = 0.6) -> list[str]:
        """Get agents that frequently agree with the given agent."""
        ...

    def get_adversaries(self, agent: str, threshold: float = 0.6) -> list[str]:
        """Get agents that frequently disagree with the given agent."""
        ...

    def update_from_debate(
        self,
        debate_id: str = "",
        participants: Optional[list[str]] = None,
        winner: str | None = None,
        votes: Optional[dict[str, Any]] = None,
        critiques: Optional[list[Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Update relationships based on debate voting patterns."""
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
        context: dict[str, Any],
        threshold: float = 0.7,
    ) -> Optional[dict[str, Any]]:
        """Detect if content represents a significant moment."""
        ...

    def get_moment_types(self) -> list[str]:
        """Get list of moment types this detector can identify."""
        ...

    def detect_upset_victory(
        self,
        winner: str = "",
        loser: str = "",
        debate_id: str = "",
        **kwargs: Any,
    ) -> Optional[dict[str, Any]]:
        """Detect if outcome represents an upset victory."""
        ...

    def detect_calibration_vindication(
        self,
        agent_name: str = "",
        prediction_confidence: float = 0.0,
        was_correct: bool = False,
        domain: str = "",
        debate_id: str = "",
        **kwargs: Any,
    ) -> Optional[dict[str, Any]]:
        """Detect if a prediction was vindicated."""
        ...

    def record_moment(
        self,
        moment: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> str | None:
        """Record a significant moment. Returns moment ID."""
        ...


@runtime_checkable
class PersonaManagerProtocol(Protocol):
    """Protocol for agent persona management.

    Manages persistent personas for agents, including
    communication style, expertise areas, and traits.
    """

    def get_persona(self, agent_name: str) -> Optional[dict[str, Any]]:
        """Get persona configuration for an agent."""
        ...

    def update_persona(self, agent_name: str, updates: dict[str, Any]) -> None:
        """Update persona attributes for an agent."""
        ...

    def get_context_for_prompt(self, agent_name: str) -> str:
        """Get persona context string for prompt injection."""
        ...

    def record_performance(
        self,
        agent_name: str,
        domain: str,
        success: bool,
        action: str = "critique",
        debate_id: str | None = None,
    ) -> None:
        """Record a performance event to update expertise."""
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
    ) -> list[Any]:
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
# Infrastructure Protocols
# =============================================================================


@runtime_checkable
class RedisClientProtocol(Protocol):
    """Protocol for Redis client implementations.

    Covers standard Redis operations used by RedisClusterManager.
    This allows typing Redis clients without requiring the redis library.

    Note: Return types use Any to accommodate both bytes (default)
    and str (when decode_responses=True) modes.
    """

    # Connection management
    def close(self) -> None:
        """Close the connection."""
        ...

    def ping(self) -> bool:
        """Ping the server to check connectivity."""
        ...

    def info(self, section: str | None = None) -> dict[str, Any]:
        """Get server information."""
        ...

    def execute_command(self, *args: Any, **kwargs: Any) -> Any:
        """Execute an arbitrary Redis command."""
        ...

    # Basic key-value operations
    def get(self, key: str) -> Any:
        """Get value for key. Returns bytes or str depending on decode_responses."""
        ...

    def set(
        self,
        key: str,
        value: Any,
        ex: int | None = None,
        px: int | None = None,
        nx: bool = False,
        xx: bool = False,
    ) -> bool | None:
        """Set key to value with optional expiration."""
        ...

    def delete(self, *keys: str) -> int:
        """Delete one or more keys."""
        ...

    def exists(self, *keys: str) -> int:
        """Check how many keys exist."""
        ...

    def expire(self, key: str, seconds: int) -> bool:
        """Set TTL on key."""
        ...

    def ttl(self, key: str) -> int:
        """Get TTL of key in seconds."""
        ...

    def incr(self, key: str) -> int:
        """Increment key by 1."""
        ...

    def decr(self, key: str) -> int:
        """Decrement key by 1."""
        ...

    # Hash operations
    def hget(self, name: str, key: str) -> Any:
        """Get field from hash. Returns bytes or str depending on decode_responses."""
        ...

    def hset(self, name: str, key: str, value: Any) -> int:
        """Set field in hash."""
        ...

    def hgetall(self, name: str) -> dict[str, Any]:
        """Get all fields from hash."""
        ...

    def hdel(self, name: str, *keys: str) -> int:
        """Delete fields from hash."""
        ...

    # Sorted set operations
    def zadd(self, name: str, mapping: dict[str, float]) -> int:
        """Add members to sorted set."""
        ...

    def zrem(self, name: str, *members: str) -> int:
        """Remove members from sorted set."""
        ...

    def zcard(self, name: str) -> int:
        """Get sorted set cardinality."""
        ...

    def zrangebyscore(
        self,
        name: str,
        min: Any,
        max: Any,
        withscores: bool = False,
    ) -> list[Any]:
        """Get members by score range."""
        ...

    def zremrangebyscore(self, name: str, min: Any, max: Any) -> int:
        """Remove members by score range."""
        ...

    # Pipeline support
    def pipeline(self, transaction: bool = True) -> Any:
        """Get pipeline for batch operations."""
        ...


# =============================================================================
# Storage Protocols
# =============================================================================


@runtime_checkable
class DebateStorageProtocol(Protocol):
    """Protocol for debate storage backends."""

    def save_debate(self, debate_id: str, data: dict[str, Any]) -> None:
        """Save debate data."""
        ...

    def load_debate(self, debate_id: str) -> Optional[dict[str, Any]]:
        """Load debate data."""
        ...

    def list_debates(self, limit: int = 100, org_id: str | None = None) -> list[Any]:
        """List available debates. Returns list of debate metadata objects."""
        ...

    def delete_debate(self, debate_id: str) -> bool:
        """Delete a debate."""
        ...

    def get_debate(self, debate_id: str) -> Optional[dict[str, Any]]:
        """Get debate by ID."""
        ...

    def get_debate_by_slug(self, slug: str) -> Optional[dict[str, Any]]:
        """Get debate by slug."""
        ...

    def get_by_id(self, debate_id: str) -> Optional[dict[str, Any]]:
        """Get debate by ID (alias)."""
        ...

    def get_by_slug(self, slug: str) -> Optional[dict[str, Any]]:
        """Get debate by slug (alias)."""
        ...

    def list_recent(self, limit: int = 20, org_id: str | None = None) -> list[Any]:
        """List recent debates."""
        ...

    def search(
        self,
        query: str | None = None,
        agent: str | None = None,
        min_confidence: float | None = None,
        limit: int = 20,
        org_id: str | None = None,
    ) -> list[Any]:
        """Search debates."""
        ...


@runtime_checkable
class UserStoreProtocol(Protocol):
    """Protocol for user storage backends."""

    def get_user_by_id(self, user_id: str) -> Any | None:
        """Get user by ID."""
        ...

    def get_user_by_email(self, email: str) -> Any | None:
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

    def can_verify(self, claim: str, claim_type: str | None = None) -> bool:
        """Check if backend can verify this claim type."""
        ...

    async def translate(self, claim: str) -> str:
        """Translate natural language claim to formal statement."""
        ...

    async def prove(self, formal_statement: str) -> Any:
        """Attempt to prove the formal statement."""
        ...


# =============================================================================
# Feedback Phase Protocols
# =============================================================================


@runtime_checkable
class DebateEmbeddingsProtocol(Protocol):
    """Protocol for debate embedding/indexing systems.

    Provides semantic indexing of debates for similarity search
    and retrieval-augmented debate preparation.
    """

    def embed(self, text: str) -> list[float]:
        """Generate embedding vector for text."""
        ...

    def index_debate(
        self,
        debate_id: str,
        content: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Index a debate for future retrieval."""
        ...

    def search_similar(
        self,
        query: str,
        limit: int = 10,
        threshold: float = 0.7,
    ) -> list[dict[str, Any]]:
        """Find debates similar to query."""
        ...


@runtime_checkable
class FlipDetectorProtocol(Protocol):
    """Protocol for position flip/change detection.

    Identifies when agents significantly change their positions
    during or between debates.
    """

    def detect_flip(
        self,
        agent: str,
        old_position: str,
        new_position: str,
        threshold: float = 0.3,
    ) -> Optional[dict[str, Any]]:
        """Detect if positions represent a significant flip."""
        ...

    def get_flip_history(
        self,
        agent: str,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Get recent position flips for an agent."""
        ...

    def get_consistency_score(self, agent: str) -> float:
        """Get position consistency score (1.0 = fully consistent)."""
        ...

    def detect_flips_for_agent(
        self,
        agent: str,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Detect all position flips for an agent in a debate."""
        ...


@runtime_checkable
class ConsensusMemoryProtocol(Protocol):
    """Protocol for storing and retrieving historical consensus outcomes.

    Maintains a knowledge base of settled debates and consensus
    positions for reference in future debates.
    """

    def store_outcome(
        self,
        topic: str,
        position: str,
        confidence: float,
        supporting_agents: list[str],
        debate_id: str,
        domain: str | None = None,
    ) -> str:
        """Store a consensus outcome. Returns outcome ID."""
        ...

    def get_consensus(
        self,
        topic: str,
        domain: str | None = None,
    ) -> Optional[dict[str, Any]]:
        """Get current consensus on a topic."""
        ...

    def search_similar_topics(
        self,
        query: str,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Find similar previously debated topics."""
        ...

    def store_consensus(
        self,
        topic: str = "",
        conclusion: str = "",
        strength: str = "",
        confidence: float = 0.0,
        participating_agents: Optional[list[str]] = None,
        agreeing_agents: Optional[list[str]] = None,
        dissenting_agents: Optional[list[str]] = None,
        key_claims: Optional[list[str]] = None,
        domain: str = "",
        tags: Optional[list[str]] = None,
        debate_duration: float = 0.0,
        rounds: int = 0,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Store a consensus outcome. Returns consensus record."""
        ...

    def update_cruxes(
        self,
        consensus_id: Any,
        cruxes: list[dict[str, Any]],
        **kwargs: Any,
    ) -> None:
        """Update crux information for a consensus record."""
        ...

    def store_vote(
        self,
        debate_id: str = "",
        vote_data: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Store vote data for a debate."""
        ...

    def store_dissent(
        self,
        debate_id: str = "",
        agent_id: str = "",
        dissent_type: Any = None,
        content: str = "",
        reasoning: str = "",
        confidence: float = 0.5,
        **kwargs: Any,
    ) -> None:
        """Store a dissenting opinion."""
        ...


@runtime_checkable
class PopulationManagerProtocol(Protocol):
    """Protocol for Genesis agent population management.

    Manages the population of agent genomes for evolutionary
    optimization of debate strategies.
    """

    def get_population(self, limit: int = 100) -> list[dict[str, Any]]:
        """Get current population of genomes."""
        ...

    def update_fitness(
        self,
        genome_id: str,
        fitness_delta: float = 0.0,
        context: str | None = None,
        consensus_win: bool | None = None,
        prediction_correct: bool | None = None,
        **kwargs: Any,
    ) -> None:
        """Update fitness score for a genome."""
        ...

    def breed(
        self,
        parent_a: str,
        parent_b: str,
        mutation_rate: float = 0.1,
    ) -> str:
        """Create offspring from two parent genomes. Returns new genome ID."""
        ...

    def select_for_breeding(
        self,
        count: int = 2,
        threshold: float = 0.8,
    ) -> list[str]:
        """Select top genomes for breeding."""
        ...

    def get_or_create_population(
        self,
        agent_names: list[str],
        **kwargs: Any,
    ) -> Any:
        """Get or create a population for a group of agents."""
        ...

    def evolve_population(
        self,
        population: Any,
        **kwargs: Any,
    ) -> Any:
        """Evolve a population to the next generation."""
        ...


@runtime_checkable
class PulseManagerProtocol(Protocol):
    """Protocol for Pulse trending topic management.

    Tracks trending topics and manages automatic debate scheduling
    based on current events.
    """

    def get_trending(
        self,
        sources: Optional[list[str]] = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Get trending topics from specified sources."""
        ...

    def record_debate_outcome(
        self,
        topic: str = "",
        platform: str = "",
        debate_id: str = "",
        consensus_reached: bool = False,
        confidence: float = 0.0,
        rounds_used: int = 0,
        category: str = "",
        volume: int = 0,
        **kwargs: Any,
    ) -> None:
        """Record outcome of a debate on a trending topic."""
        ...

    def get_topic_analytics(
        self,
        days: int = 30,
    ) -> dict[str, Any]:
        """Get analytics on debated trending topics."""
        ...


@runtime_checkable
class PromptEvolverProtocol(Protocol):
    """Protocol for prompt evolution/optimization.

    Learns from debate outcomes to improve agent prompts
    over time.
    """

    def get_current_prompt(self, agent: str) -> str:
        """Get current evolved prompt for an agent."""
        ...

    def record_outcome(
        self,
        agent: str,
        prompt_variant: str,
        success: bool,
        score: float,
        context: str | None = None,
    ) -> None:
        """Record outcome for prompt variant."""
        ...

    def evolve(self, agent: str) -> str | None:
        """Generate new prompt variant based on learnings."""
        ...

    def get_evolution_history(
        self,
        agent: str,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Get prompt evolution history for an agent."""
        ...

    def extract_winning_patterns(
        self,
        debate_results: list[Any],
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Extract winning patterns from debate results."""
        ...

    def store_patterns(
        self,
        patterns: list[dict[str, Any]],
        **kwargs: Any,
    ) -> None:
        """Store extracted patterns."""
        ...

    def update_performance(
        self,
        agent_name: str = "",
        version: Any | None = None,
        debate_result: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """Update performance metrics for prompt evolution."""
        ...


@runtime_checkable
class InsightStoreProtocol(Protocol):
    """Protocol for storing and tracking insight application.

    Tracks which insights have been applied from debates
    and their effectiveness.
    """

    def store_insight(
        self,
        insight_type: str,
        content: str,
        source_debate_id: str,
        confidence: float,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """Store an insight. Returns insight ID."""
        ...

    def mark_applied(
        self,
        insight_id: str,
        target_debate_id: str,
        success: bool | None = None,
    ) -> None:
        """Mark an insight as applied to a debate."""
        ...

    def get_recent_insights(
        self,
        insight_type: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Get recent insights."""
        ...

    def get_effectiveness(
        self,
        insight_type: str | None = None,
    ) -> dict[str, Any]:
        """Get effectiveness metrics for insights."""
        ...

    async def record_insight_usage(
        self,
        insight_id: str = "",
        debate_id: str = "",
        was_successful: bool = False,
        **kwargs: Any,
    ) -> None:
        """Record usage of an insight in a debate."""
        ...


@runtime_checkable
class BroadcastPipelineProtocol(Protocol):
    """Protocol for debate broadcast/publication pipeline.

    Handles automatic broadcasting of high-quality debates
    to various platforms.
    """

    def should_broadcast(
        self,
        debate_result: Any,
        min_confidence: float = 0.8,
    ) -> bool:
        """Check if debate qualifies for broadcast."""
        ...

    def queue_broadcast(
        self,
        debate_id: str,
        platforms: Optional[list[str]] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> str:
        """Queue a debate for broadcast. Returns job ID."""
        ...

    def get_broadcast_status(
        self,
        job_id: str,
    ) -> dict[str, Any]:
        """Get status of a broadcast job."""
        ...

    def get_supported_platforms(self) -> list[str]:
        """Get list of supported broadcast platforms."""
        ...

    async def run(
        self,
        debate_id: str,
        options: Any = None,
    ) -> Any:
        """Run the broadcast pipeline for a debate."""
        ...


@runtime_checkable
class ContinuumMemoryProtocol(Protocol):
    """Protocol for cross-debate learning memory.

    ContinuumMemory provides multi-tier memory for long-term learning across debates.
    Used by Arena to provide historical context and cross-debate learning.
    """

    def store(
        self,
        key: str,
        value: Any,
        tier: str = "medium",
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Store a value in the specified memory tier."""
        ...

    def retrieve(
        self,
        key: str,
        tier: str | None = None,
    ) -> Any | None:
        """Retrieve a value, searching tiers if not specified."""
        ...

    def search(
        self,
        query: str,
        limit: int = 10,
        tier: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search for relevant memories."""
        ...

    def get_context(
        self,
        task: str,
        limit: int = 5,
    ) -> str:
        """Get formatted context for a task from historical memories."""
        ...


@runtime_checkable
class PositionTrackerProtocol(Protocol):
    """Protocol for tracking agent positions over time.

    PositionTracker monitors agent stances and belief changes during debates.
    Used for understanding how agents evolve their positions.
    """

    def record_position(
        self,
        agent_name: str,
        position: str,
        confidence: float = 1.0,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Record an agent's position."""
        ...

    def get_position(
        self,
        agent_name: str,
    ) -> Optional[dict[str, Any]]:
        """Get an agent's current position."""
        ...

    def get_position_history(
        self,
        agent_name: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Get history of an agent's positions."""
        ...

    def has_changed(
        self,
        agent_name: str,
        threshold: float = 0.3,
    ) -> bool:
        """Check if agent's position has changed significantly."""
        ...


@runtime_checkable
class EvidenceCollectorProtocol(Protocol):
    """Protocol for automatic evidence collection.

    EvidenceCollector gathers supporting evidence from various sources
    during debates to support claims.
    """

    def collect(
        self,
        query: str,
        sources: Optional[list[str]] = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Collect evidence for a query."""
        ...

    def verify(
        self,
        claim: str,
        evidence: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Verify a claim against collected evidence."""
        ...

    def get_sources(self) -> list[str]:
        """Get list of available evidence sources."""
        ...


# =============================================================================
# Callback Types
# =============================================================================

# Event callback type
EventCallback = Callable[[str, dict[str, Any]], None]

# Async event callback type
AsyncEventCallback = Callable[[str, dict[str, Any]], Any]

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
    value: T | None = None
    error: str | None = None

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
    # Infrastructure protocols
    "RedisClientProtocol",
    # Storage protocols
    "DebateStorageProtocol",
    "UserStoreProtocol",
    # Verification protocols
    "VerificationBackendProtocol",
    # Feedback phase protocols
    "DebateEmbeddingsProtocol",
    "FlipDetectorProtocol",
    "ConsensusMemoryProtocol",
    "PopulationManagerProtocol",
    "PulseManagerProtocol",
    "PromptEvolverProtocol",
    "InsightStoreProtocol",
    "BroadcastPipelineProtocol",
    # Arena config protocols
    "ContinuumMemoryProtocol",
    "PositionTrackerProtocol",
    "EvidenceCollectorProtocol",
    # Callback types
    "EventCallback",
    "AsyncEventCallback",
    "ResponseFilter",
    "VoteCallback",
    # Result types
    "Result",
]
