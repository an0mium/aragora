"""
Builder pattern for Arena construction.

Provides a fluent interface for configuring and constructing Arena instances
with many optional components. This simplifies Arena creation and makes the
configuration more readable.

Example:
    arena = (
        ArenaBuilder(environment, agents)
        .with_protocol(protocol)
        .with_memory(critique_store)
        .with_elo_system(elo)
        .with_spectator(spectator_stream)
        .build()
    )
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from aragora.core import Agent, Environment
from aragora.debate.protocol import CircuitBreaker, DebateProtocol
from aragora.spectate.stream import SpectatorStream

if TYPE_CHECKING:
    from aragora.agents.calibration import CalibrationTracker
    from aragora.agents.grounded import MomentDetector
    from aragora.agents.personas import PersonaManager
    from aragora.agents.truth_grounding import PositionLedger, PositionTracker  # type: ignore[attr-defined]
    from aragora.insights.store import InsightStore
    from aragora.memory.continuum import ContinuumMemory
    from aragora.memory.embeddings import DebateEmbeddingsDatabase  # type: ignore[attr-defined]
    from aragora.memory.store import CritiqueStore
    from aragora.ranking.dissent import DissentRetriever
    from aragora.ranking.elo import EloSystem
    from aragora.ranking.relationship import RelationshipTracker
    from aragora.reasoning.flip import FlipDetector
    from aragora.connectors.evidence import EvidenceCollector
    from aragora.replay.recorder import ReplayRecorder
    from aragora.pulse.topics import TrendingTopic
    from aragora.debate.orchestrator import Arena

logger = logging.getLogger(__name__)


class ArenaBuilder:
    """Fluent builder for Arena construction.

    Simplifies the creation of Arena instances by providing a clear,
    chainable interface for setting optional components.

    Required parameters are provided in the constructor.
    Optional parameters are set via fluent methods.

    Usage:
        # Minimal setup
        arena = ArenaBuilder(env, agents).build()

        # Full configuration
        arena = (
            ArenaBuilder(env, agents)
            .with_protocol(DebateProtocol(rounds=5))
            .with_memory(critique_store)
            .with_elo_system(elo)
            .with_spectator(spectator)
            .with_recorder(recorder)
            .build()
        )
    """

    def __init__(
        self,
        environment: Environment,
        agents: list[Agent],
    ):
        """Initialize builder with required parameters.

        Args:
            environment: The debate environment (task, constraints, etc.)
            agents: List of agents participating in the debate
        """
        self._environment = environment
        self._agents = agents

        # Protocol configuration
        self._protocol: Optional[DebateProtocol] = None

        # Memory and persistence
        self._memory: Optional["CritiqueStore"] = None
        self._debate_embeddings: Optional["DebateEmbeddingsDatabase"] = None
        self._insight_store: Optional["InsightStore"] = None
        self._continuum_memory: Optional["ContinuumMemory"] = None

        # Event handling
        self._event_hooks: dict = {}
        self._event_emitter = None
        self._spectator: Optional[SpectatorStream] = None
        self._recorder: Optional["ReplayRecorder"] = None

        # Agent tracking and ranking
        self._agent_weights: dict[str, float] = {}
        self._elo_system: Optional["EloSystem"] = None
        self._persona_manager: Optional["PersonaManager"] = None
        self._calibration_tracker: Optional["CalibrationTracker"] = None
        self._relationship_tracker: Optional["RelationshipTracker"] = None

        # Position and truth grounding
        self._position_tracker: Optional["PositionTracker"] = None
        self._position_ledger: Optional["PositionLedger"] = None
        self._flip_detector: Optional["FlipDetector"] = None
        self._moment_detector: Optional["MomentDetector"] = None

        # Historical context
        self._dissent_retriever: Optional["DissentRetriever"] = None
        self._evidence_collector: Optional["EvidenceCollector"] = None
        self._trending_topic: Optional["TrendingTopic"] = None

        # Loop configuration
        self._loop_id: str = ""
        self._strict_loop_scoping: bool = False
        self._circuit_breaker: Optional[CircuitBreaker] = None
        self._initial_messages: list = []

    # =========================================================================
    # Protocol Configuration
    # =========================================================================

    def with_protocol(self, protocol: DebateProtocol) -> ArenaBuilder:
        """Set the debate protocol.

        Args:
            protocol: DebateProtocol instance with rounds, consensus settings, etc.
        """
        self._protocol = protocol
        return self

    def with_rounds(self, rounds: int) -> ArenaBuilder:
        """Set the number of debate rounds (creates protocol if needed).

        Args:
            rounds: Number of critique/revision rounds
        """
        if self._protocol is None:
            self._protocol = DebateProtocol(rounds=rounds)
        else:
            self._protocol.rounds = rounds
        return self

    # =========================================================================
    # Memory and Persistence
    # =========================================================================

    def with_memory(self, memory: "CritiqueStore") -> ArenaBuilder:
        """Set the critique store for memory persistence.

        Args:
            memory: CritiqueStore instance for storing debate outcomes
        """
        self._memory = memory
        return self

    def with_debate_embeddings(
        self, embeddings: "DebateEmbeddingsDatabase"
    ) -> ArenaBuilder:
        """Set the debate embeddings database for historical context.

        Args:
            embeddings: DebateEmbeddingsDatabase for semantic search
        """
        self._debate_embeddings = embeddings
        return self

    def with_insight_store(self, store: "InsightStore") -> ArenaBuilder:
        """Set the insight store for extracting learnings.

        Args:
            store: InsightStore for debate insights
        """
        self._insight_store = store
        return self

    def with_continuum_memory(self, memory: "ContinuumMemory") -> ArenaBuilder:
        """Set continuum memory for cross-debate learning.

        Args:
            memory: ContinuumMemory instance
        """
        self._continuum_memory = memory
        return self

    # =========================================================================
    # Event Handling
    # =========================================================================

    def with_event_hooks(self, hooks: dict) -> ArenaBuilder:
        """Set event hooks for streaming events.

        Args:
            hooks: Dict mapping event names to handler functions
        """
        self._event_hooks = hooks
        return self

    def with_event_emitter(self, emitter) -> ArenaBuilder:
        """Set event emitter for subscribing to user events.

        Args:
            emitter: Event emitter instance
        """
        self._event_emitter = emitter
        return self

    def with_spectator(self, spectator: SpectatorStream) -> ArenaBuilder:
        """Set spectator stream for real-time events.

        Args:
            spectator: SpectatorStream instance
        """
        self._spectator = spectator
        return self

    def with_recorder(self, recorder: "ReplayRecorder") -> ArenaBuilder:
        """Set replay recorder for debate recording.

        Args:
            recorder: ReplayRecorder instance
        """
        self._recorder = recorder
        return self

    # =========================================================================
    # Agent Tracking and Ranking
    # =========================================================================

    def with_agent_weights(self, weights: dict[str, float]) -> ArenaBuilder:
        """Set reliability weights from capability probing.

        Args:
            weights: Dict mapping agent names to reliability weights
        """
        self._agent_weights = weights
        return self

    def with_elo_system(self, elo: "EloSystem") -> ArenaBuilder:
        """Set ELO system for relationship tracking.

        Args:
            elo: EloSystem instance
        """
        self._elo_system = elo
        return self

    def with_persona_manager(self, manager: "PersonaManager") -> ArenaBuilder:
        """Set persona manager for agent specialization.

        Args:
            manager: PersonaManager instance
        """
        self._persona_manager = manager
        return self

    def with_calibration_tracker(
        self, tracker: "CalibrationTracker"
    ) -> ArenaBuilder:
        """Set calibration tracker for prediction accuracy.

        Args:
            tracker: CalibrationTracker instance
        """
        self._calibration_tracker = tracker
        return self

    def with_relationship_tracker(
        self, tracker: "RelationshipTracker"
    ) -> ArenaBuilder:
        """Set relationship tracker for agent relationships.

        Args:
            tracker: RelationshipTracker instance
        """
        self._relationship_tracker = tracker
        return self

    # =========================================================================
    # Position and Truth Grounding
    # =========================================================================

    def with_position_tracker(self, tracker: "PositionTracker") -> ArenaBuilder:
        """Set position tracker for truth-grounded personas.

        Args:
            tracker: PositionTracker instance
        """
        self._position_tracker = tracker
        return self

    def with_position_ledger(self, ledger: "PositionLedger") -> ArenaBuilder:
        """Set position ledger for grounded personas.

        Args:
            ledger: PositionLedger instance
        """
        self._position_ledger = ledger
        return self

    def with_flip_detector(self, detector: "FlipDetector") -> ArenaBuilder:
        """Set flip detector for position reversal detection.

        Args:
            detector: FlipDetector instance
        """
        self._flip_detector = detector
        return self

    def with_moment_detector(self, detector: "MomentDetector") -> ArenaBuilder:
        """Set moment detector for significant moments.

        Args:
            detector: MomentDetector instance
        """
        self._moment_detector = detector
        return self

    # =========================================================================
    # Historical Context
    # =========================================================================

    def with_dissent_retriever(self, retriever: "DissentRetriever") -> ArenaBuilder:
        """Set dissent retriever for historical minority views.

        Args:
            retriever: DissentRetriever instance
        """
        self._dissent_retriever = retriever
        return self

    def with_evidence_collector(
        self, collector: "EvidenceCollector"
    ) -> ArenaBuilder:
        """Set evidence collector for auto-collecting evidence.

        Args:
            collector: EvidenceCollector instance
        """
        self._evidence_collector = collector
        return self

    def with_trending_topic(self, topic: "TrendingTopic") -> ArenaBuilder:
        """Set trending topic to seed debate context.

        Args:
            topic: TrendingTopic instance
        """
        self._trending_topic = topic
        return self

    # =========================================================================
    # Loop Configuration
    # =========================================================================

    def with_loop_id(self, loop_id: str) -> ArenaBuilder:
        """Set loop ID for multi-loop scoping.

        Args:
            loop_id: Unique identifier for this loop
        """
        self._loop_id = loop_id
        return self

    def with_strict_loop_scoping(self, strict: bool = True) -> ArenaBuilder:
        """Enable strict loop scoping (drop events without loop_id).

        Args:
            strict: Whether to enforce strict scoping
        """
        self._strict_loop_scoping = strict
        return self

    def with_circuit_breaker(self, breaker: CircuitBreaker) -> ArenaBuilder:
        """Set circuit breaker for agent failure handling.

        Args:
            breaker: CircuitBreaker instance
        """
        self._circuit_breaker = breaker
        return self

    def with_initial_messages(self, messages: list) -> ArenaBuilder:
        """Set initial conversation history (for fork debates).

        Args:
            messages: List of initial messages
        """
        self._initial_messages = messages
        return self

    # =========================================================================
    # Composite Configuration
    # =========================================================================

    def with_full_tracking(
        self,
        elo_system: "EloSystem",
        persona_manager: Optional["PersonaManager"] = None,
        calibration_tracker: Optional["CalibrationTracker"] = None,
        relationship_tracker: Optional["RelationshipTracker"] = None,
    ) -> ArenaBuilder:
        """Configure all tracking components at once.

        Args:
            elo_system: EloSystem instance (required)
            persona_manager: Optional PersonaManager
            calibration_tracker: Optional CalibrationTracker
            relationship_tracker: Optional RelationshipTracker
        """
        self._elo_system = elo_system
        if persona_manager:
            self._persona_manager = persona_manager
        if calibration_tracker:
            self._calibration_tracker = calibration_tracker
        if relationship_tracker:
            self._relationship_tracker = relationship_tracker
        return self

    def with_full_memory(
        self,
        memory: "CritiqueStore",
        debate_embeddings: Optional["DebateEmbeddingsDatabase"] = None,
        continuum_memory: Optional["ContinuumMemory"] = None,
        insight_store: Optional["InsightStore"] = None,
    ) -> ArenaBuilder:
        """Configure all memory components at once.

        Args:
            memory: CritiqueStore instance (required)
            debate_embeddings: Optional DebateEmbeddingsDatabase
            continuum_memory: Optional ContinuumMemory
            insight_store: Optional InsightStore
        """
        self._memory = memory
        if debate_embeddings:
            self._debate_embeddings = debate_embeddings
        if continuum_memory:
            self._continuum_memory = continuum_memory
        if insight_store:
            self._insight_store = insight_store
        return self

    # =========================================================================
    # Build
    # =========================================================================

    def build(self) -> "Arena":
        """Build and return the configured Arena instance.

        Returns:
            Configured Arena instance
        """
        # Import here to avoid circular dependency
        from aragora.debate.orchestrator import Arena

        return Arena(
            environment=self._environment,
            agents=self._agents,
            protocol=self._protocol,
            memory=self._memory,
            event_hooks=self._event_hooks,
            event_emitter=self._event_emitter,
            spectator=self._spectator,
            debate_embeddings=self._debate_embeddings,
            insight_store=self._insight_store,
            recorder=self._recorder,
            agent_weights=self._agent_weights,
            position_tracker=self._position_tracker,
            position_ledger=self._position_ledger,
            elo_system=self._elo_system,
            persona_manager=self._persona_manager,
            dissent_retriever=self._dissent_retriever,
            flip_detector=self._flip_detector,
            calibration_tracker=self._calibration_tracker,
            continuum_memory=self._continuum_memory,
            relationship_tracker=self._relationship_tracker,
            moment_detector=self._moment_detector,
            loop_id=self._loop_id,
            strict_loop_scoping=self._strict_loop_scoping,
            circuit_breaker=self._circuit_breaker,
            initial_messages=self._initial_messages,
            trending_topic=self._trending_topic,
            evidence_collector=self._evidence_collector,
        )


# Convenience function for minimal Arena creation
def create_arena(
    environment: Environment,
    agents: list[Agent],
    protocol: Optional[DebateProtocol] = None,
    memory: Optional["CritiqueStore"] = None,
    elo_system: Optional["EloSystem"] = None,
) -> "Arena":
    """Create an Arena with commonly used options.

    For more complex configurations, use ArenaBuilder directly.

    Args:
        environment: The debate environment
        agents: List of participating agents
        protocol: Optional debate protocol
        memory: Optional critique store
        elo_system: Optional ELO system

    Returns:
        Configured Arena instance
    """
    builder = ArenaBuilder(environment, agents)

    if protocol:
        builder.with_protocol(protocol)
    if memory:
        builder.with_memory(memory)
    if elo_system:
        builder.with_elo_system(elo_system)

    return builder.build()
