"""Feature protocol definitions.

Provides Protocol classes for various platform features including
verification, population management, pulse, prompt evolution,
insights, broadcasting, evidence collection, and cross-cutting
protocols that break circular imports.
"""

from __future__ import annotations

from typing import Any, Optional, Protocol, runtime_checkable


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
# Cross-cutting protocols (break circular imports)
# =============================================================================


class EvidenceProtocol(Protocol):
    """Protocol for Evidence objects.

    Use this for typing Evidence parameters without importing from
    aragora.connectors.base, which causes circular imports in 43+ files.
    """

    id: str
    source_type: Any
    source_id: str
    content: str
    title: str
    confidence: float
    freshness: float
    authority: float
    metadata: dict[str, Any]


class StreamEventProtocol(Protocol):
    """Protocol for stream events.

    Use this for typing StreamEvent parameters without importing from
    aragora.events.types or aragora.server.stream, which causes
    circular imports in the events dispatcher and server handlers.
    """

    type: Any
    data: dict[str, Any]
    timestamp: float
    round: int
    agent: str


class WebhookConfigProtocol(Protocol):
    """Protocol for webhook configuration.

    Use this for typing WebhookConfig parameters without importing from
    aragora.integrations.webhooks or aragora.storage.webhook_config_store.
    """

    name: str
    url: str
    secret: str
    event_types: set[str]
    timeout_s: float
    max_retries: int


__all__ = [
    "VerificationBackendProtocol",
    "PopulationManagerProtocol",
    "PulseManagerProtocol",
    "PromptEvolverProtocol",
    "InsightStoreProtocol",
    "BroadcastPipelineProtocol",
    "EvidenceCollectorProtocol",
    "EvidenceProtocol",
    "StreamEventProtocol",
    "WebhookConfigProtocol",
]
