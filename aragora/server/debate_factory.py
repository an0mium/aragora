"""
Factory for creating and configuring debate arenas.

Extracts agent creation and arena setup logic from unified_server.py
for better modularity and testability.
"""

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Optional

from aragora.config import ALLOWED_AGENT_TYPES, MAX_AGENTS_PER_DEBATE
from aragora.exceptions import ConfigurationError

logger = logging.getLogger(__name__)

# Import create_agent for agent creation
try:
    from aragora.agents.base import create_agent
except ImportError:
    create_agent = None  # type: ignore

if TYPE_CHECKING:
    from aragora.agents.grounded import MomentDetector, PositionLedger  # type: ignore[attr-defined]
    from aragora.agents.personas import PersonaManager
    from aragora.agents.truth_grounding import PositionTracker  # type: ignore[attr-defined]
    from aragora.debate.embeddings import (
        DebateEmbeddingsDatabase as DebateEmbeddings,  # type: ignore[attr-defined]
    )
    from aragora.debate.orchestrator import Arena
    from aragora.insights.flip_detector import FlipDetector
    from aragora.memory.consensus import DissentRetriever  # type: ignore[attr-defined]
    from aragora.pulse.ingestor import TrendingTopic
    from aragora.ranking.elo import EloSystem
    from aragora.server.stream import SyncEventEmitter  # type: ignore[attr-defined]


@dataclass
class AgentSpec:
    """Specification for an agent to create."""

    agent_type: str
    role: Optional[str] = None
    name: Optional[str] = None

    def __post_init__(self):
        """Validate agent type."""
        if self.agent_type.lower() not in ALLOWED_AGENT_TYPES:
            raise ValueError(
                f"Invalid agent type: {self.agent_type}. "
                f"Allowed: {', '.join(sorted(ALLOWED_AGENT_TYPES))}"
            )
        if self.name is None:
            self.name = f"{self.agent_type}_{self.role or 'proposer'}"


@dataclass
class AgentCreationResult:
    """Result of agent creation attempt."""

    agents: list = field(default_factory=list)
    failed: list = field(default_factory=list)  # List of (agent_type, error_msg)

    @property
    def success_count(self) -> int:
        return len(self.agents)

    @property
    def failure_count(self) -> int:
        return len(self.failed)

    @property
    def has_minimum(self) -> bool:
        """Check if minimum number of agents were created."""
        return self.success_count >= 2


@dataclass
class DebateConfig:
    """Configuration for debate creation."""

    question: str
    agents_str: str = "anthropic-api,openai-api"
    rounds: int = 3
    consensus: str = "majority"
    debate_id: Optional[str] = None
    trending_topic: Optional["TrendingTopic"] = None  # TrendingTopic from pulse

    def parse_agent_specs(self) -> list[AgentSpec]:
        """Parse agent specifications from comma-separated string.

        Returns:
            List of AgentSpec objects

        Raises:
            ValueError: If agent count exceeds maximum
        """
        agent_list = [s.strip() for s in self.agents_str.split(",") if s.strip()]

        if len(agent_list) > MAX_AGENTS_PER_DEBATE:
            raise ValueError(f"Too many agents. Maximum: {MAX_AGENTS_PER_DEBATE}")
        if len(agent_list) < 2:
            raise ValueError("At least 2 agents required for a debate")

        specs = []
        for spec_str in agent_list:
            if ":" in spec_str:
                agent_type, role = spec_str.split(":", 1)
            else:
                agent_type = spec_str
                role = None
            specs.append(AgentSpec(agent_type=agent_type.strip(), role=role))

        return specs


class DebateFactory:
    """
    Factory for creating and configuring debates.

    Handles agent creation, validation, and arena setup with all
    required subsystems (ELO, personas, embeddings, etc.).

    Usage:
        factory = DebateFactory(
            elo_system=elo_system,
            persona_manager=persona_manager,
            stream_emitter=emitter,
        )

        config = DebateConfig(
            question="What is the best sorting algorithm?",
            agents_str="anthropic-api,openai-api,gemini",
            rounds=3,
        )

        arena = factory.create_arena(config)
        result = await arena.run()
    """

    def __init__(
        self,
        elo_system: Optional["EloSystem"] = None,
        persona_manager: Optional["PersonaManager"] = None,
        debate_embeddings: Optional["DebateEmbeddings"] = None,
        position_tracker: Optional["PositionTracker"] = None,
        position_ledger: Optional["PositionLedger"] = None,
        flip_detector: Optional["FlipDetector"] = None,
        dissent_retriever: Optional["DissentRetriever"] = None,
        moment_detector: Optional["MomentDetector"] = None,
        stream_emitter: Optional["SyncEventEmitter"] = None,
    ):
        """Initialize the debate factory.

        Args:
            elo_system: ELO rating system for agent rankings
            persona_manager: Manager for agent personas
            debate_embeddings: Embedding store for semantic search
            position_tracker: Tracks agent positions during debate
            position_ledger: Historical position ledger
            flip_detector: Detects agent position flips
            dissent_retriever: Retrieves past dissent patterns
            moment_detector: Detects key debate moments
            stream_emitter: Event stream emitter for live updates
        """
        self.elo_system = elo_system
        self.persona_manager = persona_manager
        self.debate_embeddings = debate_embeddings
        self.position_tracker = position_tracker
        self.position_ledger = position_ledger
        self.flip_detector = flip_detector
        self.dissent_retriever = dissent_retriever
        self.moment_detector = moment_detector
        self.stream_emitter = stream_emitter

    def create_agents(
        self,
        specs: list[AgentSpec],
        stream_wrapper: Optional[Callable[..., Any]] = None,
        debate_id: Optional[str] = None,
    ) -> AgentCreationResult:
        """Create agents from specifications.

        Args:
            specs: List of agent specifications
            stream_wrapper: Optional function to wrap agents for streaming
            debate_id: Optional debate ID for error reporting

        Returns:
            AgentCreationResult with created agents and failures
        """
        if create_agent is None:
            raise ConfigurationError(
                component="DebateFactory",
                reason="create_agent not available - agents module failed to import",
            )

        result = AgentCreationResult()

        for spec in specs:
            role = spec.role or "proposer"
            try:
                agent = create_agent(
                    model_type=spec.agent_type,  # type: ignore[arg-type]
                    name=spec.name,
                    role=role,
                )

                # Validate API key for API-based agents
                if hasattr(agent, "api_key") and not agent.api_key:
                    raise ValueError(f"Missing API key for {spec.agent_type}")

                # Apply streaming wrapper if provided
                if stream_wrapper is not None:
                    agent = stream_wrapper(agent, self.stream_emitter, debate_id)

                result.agents.append(agent)
                logger.debug(f"Created agent {spec.agent_type} successfully")

            except Exception as e:
                error_msg = f"Failed to create agent {spec.agent_type}: {e}"
                logger.error(error_msg)
                result.failed.append((spec.agent_type, str(e)))

                # Emit error event if emitter available
                if self.stream_emitter and debate_id:
                    self._emit_agent_error(spec.agent_type, str(e), debate_id)

        return result

    def _emit_agent_error(self, agent_type: str, error: str, debate_id: str) -> None:
        """Emit an error event for agent creation failure."""
        try:
            from aragora.server.stream.events import StreamEvent, StreamEventType

            self.stream_emitter.emit(
                StreamEvent(
                    type=StreamEventType.ERROR,
                    data={
                        "agent": agent_type,
                        "error": error,
                        "phase": "initialization",
                    },
                    loop_id=debate_id,
                )
            )
        except Exception as e:
            logger.warning(f"Failed to emit agent error event: {e}")

    def create_arena(
        self,
        config: DebateConfig,
        event_hooks: Optional[dict] = None,
        stream_wrapper: Optional[Callable[..., Any]] = None,
    ) -> "Arena":
        """Create a fully configured debate arena.

        Uses ArenaBuilder internally for cleaner configuration.

        Args:
            config: Debate configuration
            event_hooks: Optional event hooks for the arena
            stream_wrapper: Optional function to wrap agents for streaming

        Returns:
            Configured Arena ready to run

        Raises:
            ValueError: If not enough agents could be created
        """
        from aragora.core import Environment
        from aragora.debate.arena_builder import ArenaBuilder
        from aragora.debate.protocol import DebateProtocol

        # Parse and create agents
        specs = config.parse_agent_specs()
        agent_result = self.create_agents(
            specs,
            stream_wrapper=stream_wrapper,
            debate_id=config.debate_id,
        )

        if not agent_result.has_minimum:
            failed_names = [a for a, _ in agent_result.failed]
            raise ValueError(
                f"Only {agent_result.success_count} agents initialized "
                f"(need at least 2). Failed: {', '.join(failed_names)}"
            )

        # Create environment and protocol
        env = Environment(
            task=config.question,
            context="",
            max_rounds=config.rounds,
        )
        protocol = DebateProtocol(
            rounds=config.rounds,
            consensus=config.consensus,  # type: ignore[arg-type]
            proposer_count=len(agent_result.agents),
            topology="all-to-all",
        )

        # Build arena using ArenaBuilder for cleaner configuration
        builder = (
            ArenaBuilder(env, agent_result.agents)
            .with_protocol(protocol)
            .with_event_hooks(event_hooks or {})
            .with_event_emitter(self.stream_emitter)
            .with_loop_id(config.debate_id or "")
        )

        # Add optional subsystems if available
        if self.persona_manager:
            builder = builder.with_persona_manager(self.persona_manager)
        if self.debate_embeddings:
            builder = builder.with_debate_embeddings(self.debate_embeddings)
        if self.elo_system:
            builder = builder.with_elo_system(self.elo_system)
        if self.position_tracker:
            builder = builder.with_position_tracker(self.position_tracker)
        if self.position_ledger:
            builder = builder.with_position_ledger(self.position_ledger)
        if self.flip_detector:
            builder = builder.with_flip_detector(self.flip_detector)
        if self.dissent_retriever:
            builder = builder.with_dissent_retriever(self.dissent_retriever)
        if self.moment_detector:
            builder = builder.with_moment_detector(self.moment_detector)
        if config.trending_topic:
            builder = builder.with_trending_topic(config.trending_topic)

        return builder.build()

    def reset_circuit_breakers(self, arena: "Arena") -> None:
        """Reset circuit breakers for fresh debate.

        For ad-hoc debates, we want all agents to have a fresh start.

        Args:
            arena: The arena whose circuit breakers to reset
        """
        cb_status = arena.circuit_breaker.get_all_status()
        if cb_status:
            logger.debug(f"Agent status before debate: {cb_status}")
            open_circuits = [
                name for name, status in cb_status.items() if status["status"] == "open"
            ]
            if open_circuits:
                logger.debug(f"Resetting open circuits for: {open_circuits}")
                arena.circuit_breaker.reset()
