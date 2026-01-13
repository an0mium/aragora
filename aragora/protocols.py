"""
Protocol definitions for Aragora storage and memory backends.

These protocols define the interfaces that storage implementations must follow,
enabling better type checking and easier testing with mock implementations.

Usage:
    from aragora.protocols import StorageBackend, MemoryBackend

    def save_debate(storage: StorageBackend, debate: DebateResult) -> str:
        return storage.save(debate)
"""

from __future__ import annotations

from typing import Any, Optional, Protocol, Sequence, runtime_checkable


@runtime_checkable
class StorageBackend(Protocol):
    """Protocol for debate storage backends (SQLite, Supabase, etc.)."""

    def save(
        self,
        debate_id: str,
        task: str,
        agents: list[str],
        artifact: dict[str, Any],
        consensus_reached: bool = False,
        confidence: float = 0.0,
    ) -> str:
        """Save a debate result. Returns the slug."""
        ...

    def get(self, slug: str) -> Optional[dict[str, Any]]:
        """Get a debate by slug."""
        ...

    def list_debates(
        self,
        limit: int = 20,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """List debates with pagination."""
        ...

    def search(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """Search debates by query."""
        ...


@runtime_checkable
class MemoryBackend(Protocol):
    """Protocol for multi-tier memory backends."""

    def store(
        self,
        content: str,
        importance: float = 0.5,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """Store a memory item. Returns the memory ID."""
        ...

    def retrieve(
        self,
        query: str,
        limit: int = 10,
        tier: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Retrieve memories matching query."""
        ...

    def promote(self, memory_id: str, reason: str) -> bool:
        """Promote a memory to a longer-lived tier."""
        ...

    def decay(self) -> int:
        """Run decay cycle. Returns number of memories affected."""
        ...


@runtime_checkable
class EloBackend(Protocol):
    """Protocol for ELO rating system backends."""

    def get_rating(self, agent: str) -> float:
        """Get current ELO rating for an agent."""
        ...

    def update_ratings(
        self,
        debate_id: str,
        winner: Optional[str],
        participants: list[str],
        scores: dict[str, float],
    ) -> dict[str, float]:
        """Update ratings after a debate. Returns ELO changes."""
        ...

    def get_leaderboard(
        self,
        limit: int = 20,
        domain: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Get leaderboard rankings."""
        ...

    def get_history(
        self,
        agent: str,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Get ELO history for an agent."""
        ...


@runtime_checkable
class EmbeddingBackend(Protocol):
    """Protocol for embedding/vector storage backends."""

    def embed(self, text: str) -> list[float]:
        """Generate embedding vector for text."""
        ...

    def store_embedding(
        self,
        id: str,
        text: str,
        embedding: list[float],
    ) -> None:
        """Store an embedding."""
        ...

    def search_similar(
        self,
        query_embedding: list[float],
        limit: int = 10,
        threshold: float = 0.7,
    ) -> list[tuple[str, float]]:
        """Search for similar embeddings. Returns (id, similarity) pairs."""
        ...


@runtime_checkable
class ConsensusBackend(Protocol):
    """Protocol for consensus memory backends."""

    def record_consensus(
        self,
        topic: str,
        position: str,
        confidence: float,
        supporting_agents: list[str],
        evidence: list[str],
        debate_id: Optional[str] = None,
    ) -> int:
        """Record a new consensus. Returns consensus ID."""
        ...

    def get_consensus(self, topic: str) -> Optional[dict[str, Any]]:
        """Get the current consensus on a topic."""
        ...

    def record_dissent(
        self,
        consensus_id: int,
        agent: str,
        position: str,
        reasoning: str,
    ) -> int:
        """Record dissent from a consensus. Returns dissent ID."""
        ...

    def get_dissents(self, consensus_id: int) -> list[dict[str, Any]]:
        """Get all dissents for a consensus."""
        ...


@runtime_checkable
class CritiqueBackend(Protocol):
    """Protocol for critique storage backends."""

    def record_critique(
        self,
        debate_id: str,
        critic: str,
        target: str,
        critique_type: str,
        content: str,
        accepted: bool = False,
    ) -> int:
        """Record a critique. Returns critique ID."""
        ...

    def get_patterns(
        self,
        agent: Optional[str] = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Get critique patterns."""
        ...

    def get_reputation(self, agent: str) -> Optional[dict[str, Any]]:
        """Get reputation scores for an agent."""
        ...


@runtime_checkable
class PersonaBackend(Protocol):
    """Protocol for agent persona backends."""

    def get_persona(self, agent_name: str) -> Optional[dict[str, Any]]:
        """Get persona for an agent."""
        ...

    def save_persona(
        self,
        agent_name: str,
        traits: list[str],
        expertise: dict[str, float],
        description: Optional[str] = None,
    ) -> None:
        """Save or update an agent persona."""
        ...

    def record_performance(
        self,
        agent_name: str,
        debate_id: str,
        domain: str,
        action: str,
        success: bool,
    ) -> None:
        """Record performance event for learning."""
        ...


@runtime_checkable
class GenesisBackend(Protocol):
    """Protocol for agent genome/evolution backends."""

    def save_genome(
        self,
        genome_id: str,
        name: str,
        traits: list[str],
        expertise: dict[str, float],
        parent_genomes: Optional[list[str]] = None,
        generation: int = 0,
    ) -> None:
        """Save an agent genome."""
        ...

    def get_genome(self, genome_id: str) -> Optional[dict[str, Any]]:
        """Get a genome by ID."""
        ...

    def get_top_genomes(
        self,
        limit: int = 10,
        min_fitness: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Get top-performing genomes."""
        ...

    def record_event(
        self,
        event_type: str,
        data: dict[str, Any],
        parent_event_id: Optional[str] = None,
    ) -> str:
        """Record a genesis event. Returns event ID."""
        ...


# =============================================================================
# HTTP Handler Protocols
# =============================================================================


@runtime_checkable
class HTTPHeaders(Protocol):
    """Protocol for HTTP headers dictionary-like interface."""

    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get a header value."""
        ...


@runtime_checkable
class HTTPRequestHandler(Protocol):
    """
    Protocol for HTTP request handlers.

    Used to type the `handler` parameter in endpoint handlers,
    replacing `Any` with a proper structural type.
    """

    headers: HTTPHeaders
    client_address: tuple[str, int]
    command: str  # HTTP method (GET, POST, etc.)
    path: str  # Request path

    @property
    def rfile(self) -> Any:
        """Readable file-like object for request body."""
        ...


@runtime_checkable
class AuthenticatedUser(Protocol):
    """Protocol for authenticated user context."""

    user_id: str
    email: Optional[str]
    is_authenticated: bool
    org_id: Optional[str]

    @property
    def is_admin(self) -> bool:
        """Check if user has admin role."""
        ...


@runtime_checkable
class Agent(Protocol):
    """Protocol for debate agent interface."""

    name: str

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate a response for the given prompt."""
        ...


@runtime_checkable
class AgentRating(Protocol):
    """Protocol for agent rating/ELO data."""

    name: str
    elo: float
    wins: int
    losses: int
    draws: int


# Type aliases for common return types
DebateRecord = dict[str, Any]
MemoryRecord = dict[str, Any]
AgentRecord = dict[str, Any]
QueryParams = dict[str, Any]
PathSegments = list[str]
