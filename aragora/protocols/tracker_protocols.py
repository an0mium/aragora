"""Tracker protocol definitions.

Provides Protocol classes for ELO systems, calibration tracking,
position tracking, relationship tracking, moment detection,
persona management, and dissent retrieval.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


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

    def get_calibration(self, agent: str) -> dict[str, Any] | None:
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

    def get_relationship(self, agent_a: str, agent_b: str) -> dict[str, Any] | None:
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
        participants: list[str] | None = None,
        winner: str | None = None,
        votes: dict[str, Any] | None = None,
        critiques: list[Any] | None = None,
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
    ) -> dict[str, Any] | None:
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
    ) -> dict[str, Any] | None:
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
    ) -> dict[str, Any] | None:
        """Detect if a prediction was vindicated."""
        ...

    def record_moment(
        self,
        moment: dict[str, Any] | None = None,
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

    def get_persona(self, agent_name: str) -> dict[str, Any] | None:
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
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record an agent's position."""
        ...

    def get_position(
        self,
        agent_name: str,
    ) -> dict[str, Any] | None:
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


__all__ = [
    "EloSystemProtocol",
    "CalibrationTrackerProtocol",
    "PositionLedgerProtocol",
    "RelationshipTrackerProtocol",
    "MomentDetectorProtocol",
    "PersonaManagerProtocol",
    "DissentRetrieverProtocol",
    "PositionTrackerProtocol",
]
