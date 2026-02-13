"""Debate and consensus protocol definitions.

Provides Protocol classes for debate results, consensus detection,
ranking systems, consensus memory, debate embeddings, and flip detection.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


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
    ) -> dict[str, Any] | None:
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
        participating_agents: list[str] | None = None,
        agreeing_agents: list[str] | None = None,
        dissenting_agents: list[str] | None = None,
        key_claims: list[str] | None = None,
        domain: str = "",
        tags: list[str] | None = None,
        debate_duration: float = 0.0,
        rounds: int = 0,
        metadata: dict[str, Any] | None = None,
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
        vote_data: dict[str, Any] | None = None,
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
        metadata: dict[str, Any] | None = None,
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
    ) -> dict[str, Any] | None:
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


__all__ = [
    "DebateResultProtocol",
    "ConsensusDetectorProtocol",
    "RankingSystemProtocol",
    "ConsensusMemoryProtocol",
    "DebateEmbeddingsProtocol",
    "FlipDetectorProtocol",
]
