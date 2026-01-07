"""
Protocol interfaces for Arena dependencies.

These Protocol classes allow Arena to depend on interfaces rather than
concrete implementations, breaking circular import chains. Concrete
implementations are injected at runtime via ArenaFactory.
"""

from typing import Any, Optional, Protocol, runtime_checkable


@runtime_checkable
class PositionTrackerProtocol(Protocol):
    """Interface for tracking agent positions on claims."""

    def record_position(
        self,
        agent_name: str,
        claim_text: str,
        stance: str,
        confidence: float,
        context: Optional[str] = None,
    ) -> None:
        """Record an agent's position on a claim."""
        ...

    def get_positions(self, agent_name: str) -> list[dict]:
        """Get all positions recorded for an agent."""
        ...

    def get_position_history(self, claim_text: str) -> list[dict]:
        """Get position history for a specific claim."""
        ...


@runtime_checkable
class CalibrationTrackerProtocol(Protocol):
    """Interface for tracking agent prediction calibration."""

    def record_prediction(
        self,
        agent_name: str,
        prediction: str,
        confidence: float,
        category: Optional[str] = None,
    ) -> str:
        """Record a prediction for later resolution."""
        ...

    def resolve_prediction(
        self,
        prediction_id: str,
        outcome: bool,
    ) -> None:
        """Resolve a prediction as correct or incorrect."""
        ...

    def get_calibration_score(self, agent_name: str) -> Optional[float]:
        """Get an agent's calibration score."""
        ...


@runtime_checkable
class BeliefNetworkProtocol(Protocol):
    """Interface for belief network operations."""

    def add_claim(
        self,
        claim_id: str,
        text: str,
        confidence: float,
        agent: Optional[str] = None,
    ) -> None:
        """Add a claim to the belief network."""
        ...

    def add_support(
        self,
        claim_id: str,
        supporting_claim_id: str,
        strength: float = 1.0,
    ) -> None:
        """Add a support relationship between claims."""
        ...

    def add_attack(
        self,
        claim_id: str,
        attacking_claim_id: str,
        strength: float = 1.0,
    ) -> None:
        """Add an attack relationship between claims."""
        ...

    def propagate(self) -> dict[str, float]:
        """Propagate beliefs through the network."""
        ...


@runtime_checkable
class BeliefPropagationAnalyzerProtocol(Protocol):
    """Interface for analyzing belief propagation in debates."""

    def analyze_debate(
        self,
        messages: list,
        network: Optional[BeliefNetworkProtocol] = None,
    ) -> dict:
        """Analyze belief changes during a debate."""
        ...


@runtime_checkable
class CitationExtractorProtocol(Protocol):
    """Interface for extracting citations from text."""

    def extract(self, text: str) -> list[dict]:
        """Extract citations from text."""
        ...

    def validate_citations(
        self,
        citations: list[dict],
        evidence_store: Optional[Any] = None,
    ) -> list[dict]:
        """Validate extracted citations."""
        ...


@runtime_checkable
class InsightExtractorProtocol(Protocol):
    """Interface for extracting insights from debate content."""

    def extract_insights(
        self,
        text: str,
        context: Optional[dict] = None,
    ) -> list[dict]:
        """Extract insights from debate text."""
        ...


@runtime_checkable
class InsightStoreProtocol(Protocol):
    """Interface for storing and retrieving insights."""

    def store_insight(
        self,
        insight: dict,
        debate_id: Optional[str] = None,
    ) -> str:
        """Store an insight and return its ID."""
        ...

    def get_insights(
        self,
        debate_id: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict]:
        """Retrieve stored insights."""
        ...

    def search_insights(
        self,
        query: str,
        limit: int = 10,
    ) -> list[dict]:
        """Search insights by query."""
        ...


@runtime_checkable
class CritiqueStoreProtocol(Protocol):
    """Interface for storing and retrieving critique patterns."""

    def store_critique(
        self,
        critique: Any,
        debate_id: Optional[str] = None,
    ) -> None:
        """Store a critique."""
        ...

    def get_patterns(
        self,
        task: str,
        limit: int = 10,
    ) -> list[dict]:
        """Get critique patterns relevant to a task."""
        ...


@runtime_checkable
class ArgumentCartographerProtocol(Protocol):
    """Interface for mapping argument structures."""

    def map_arguments(
        self,
        messages: list,
        context: Optional[dict] = None,
    ) -> dict:
        """Map argument structure from messages."""
        ...

    def visualize(
        self,
        argument_map: dict,
        format: str = "json",
    ) -> Any:
        """Visualize the argument map."""
        ...
