"""
Multi-agent debate orchestrator.

Implements the propose -> critique -> revise loop with configurable
debate protocols and consensus mechanisms.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from functools import lru_cache
from types import TracebackType
from typing import TYPE_CHECKING, Any, Optional

from aragora.core import Agent, Critique, DebateResult, Environment, Message, Vote
from aragora.debate.agent_pool import AgentPool, AgentPoolConfig
from aragora.debate.arena_config import ArenaConfig
from aragora.debate.arena_phases import create_phase_executor, init_phases
from aragora.debate.audience_manager import AudienceManager
from aragora.debate.autonomic_executor import AutonomicExecutor
from aragora.debate.chaos_theater import DramaLevel, get_chaos_director
from aragora.debate.complexity_governor import (
    classify_task_complexity,
    get_complexity_governor,
)
from aragora.debate.context import DebateContext
from aragora.debate.context_delegation import ContextDelegator
from aragora.debate.convergence import (
    ConvergenceDetector,
    cleanup_embedding_cache,
)
from aragora.debate.event_bridge import EventEmitterBridge
from aragora.debate.grounded_operations import GroundedOperations
from aragora.debate.prompt_context import PromptContextBuilder
from aragora.debate.event_bus import EventBus
from aragora.debate.extensions import ArenaExtensions
from aragora.debate.immune_system import get_immune_system
from aragora.debate.judge_selector import JudgeSelector
from aragora.debate.optional_imports import OptionalImports
from aragora.debate.protocol import CircuitBreaker, DebateProtocol
from aragora.debate.result_formatter import ResultFormatter
from aragora.debate.roles_manager import RolesManager
from aragora.debate.safety import resolve_auto_evolve, resolve_prompt_evolution
from aragora.debate.sanitization import OutputSanitizer
from aragora.debate.state_cache import DebateStateCache
from aragora.debate.subsystem_coordinator import SubsystemCoordinator
from aragora.debate.termination_checker import TerminationChecker
from aragora.exceptions import EarlyStopError
from aragora.observability.logging import correlation_context
from aragora.observability.logging import get_logger as get_structured_logger
from aragora.observability.tracing import add_span_attributes, get_tracer
from aragora.server.metrics import (
    ACTIVE_DEBATES,
    track_circuit_breaker_state,
    track_debate_outcome,
)
from aragora.spectate.stream import SpectatorStream
from aragora.utils.cache_registry import register_lru_cache

# Optional evolution import for prompt self-improvement
try:
    from aragora.evolution.evolver import PromptEvolver

    PROMPT_EVOLVER_AVAILABLE = True
except ImportError:
    PromptEvolver = None  # type: ignore[misc, assignment]
    PROMPT_EVOLVER_AVAILABLE = False

logger = logging.getLogger(__name__)
# Structured logger for key debate events (JSON-formatted in production)
slog = get_structured_logger(__name__)

# TYPE_CHECKING imports for type hints without runtime import overhead
if TYPE_CHECKING:
    from aragora.debate.context_gatherer import ContextGatherer
    from aragora.debate.memory_manager import MemoryManager
    from aragora.debate.phases import (
        AnalyticsPhase,
        ConsensusPhase,
        ContextInitializer,
        DebateRoundsPhase,
        FeedbackPhase,
        ProposalPhase,
        VotingPhase,
    )
    from aragora.debate.prompt_builder import PromptBuilder
    from aragora.reasoning.citations import CitationExtractor
    from aragora.reasoning.evidence_grounding import EvidenceGrounder
    from aragora.types.protocols import EventEmitterProtocol


@register_lru_cache
@lru_cache(maxsize=256)
def _compute_domain_from_task(task_lower: str) -> str:
    """Compute domain from lowercased task string.

    Module-level cached helper to avoid O(n) string matching
    for repeated task strings across debate instances.
    """
    if any(w in task_lower for w in ("security", "hack", "vulnerability", "auth", "encrypt")):
        return "security"
    if any(w in task_lower for w in ("performance", "speed", "optimize", "cache", "latency")):
        return "performance"
    if any(w in task_lower for w in ("test", "testing", "coverage", "regression")):
        return "testing"
    if any(w in task_lower for w in ("design", "architecture", "pattern", "structure")):
        return "architecture"
    if any(w in task_lower for w in ("bug", "error", "fix", "crash", "exception")):
        return "debugging"
    if any(w in task_lower for w in ("api", "endpoint", "rest", "graphql")):
        return "api"
    if any(w in task_lower for w in ("database", "sql", "query", "schema")):
        return "database"
    if any(w in task_lower for w in ("ui", "frontend", "react", "css", "layout")):
        return "frontend"
    return "general"


class Arena:
    """
    Orchestrates multi-agent debates.

    The Arena manages the flow of a debate:
    1. Proposers generate initial proposals
    2. Critics critique each proposal
    3. Proposers revise based on critique
    4. Repeat for configured rounds
    5. Consensus mechanism selects final answer
    """

    # Phase class attributes (initialized by init_phases)
    voting_phase: "VotingPhase"
    citation_extractor: Optional["CitationExtractor"]
    evidence_grounder: "EvidenceGrounder"
    prompt_builder: "PromptBuilder"
    memory_manager: "MemoryManager"
    context_gatherer: "ContextGatherer"
    context_initializer: "ContextInitializer"
    proposal_phase: "ProposalPhase"
    debate_rounds_phase: "DebateRoundsPhase"
    consensus_phase: "ConsensusPhase"
    analytics_phase: "AnalyticsPhase"
    feedback_phase: "FeedbackPhase"

    def __init__(
        self,
        environment: Environment,
        agents: list[Agent],
        protocol: DebateProtocol = None,
        memory=None,  # CritiqueStore instance
        event_hooks: dict = None,  # Optional hooks for streaming events
        event_emitter: Optional[
            "EventEmitterProtocol"
        ] = None,  # Optional event emitter for subscribing to user events
        spectator: SpectatorStream = None,  # Optional spectator stream for real-time events
        debate_embeddings=None,  # DebateEmbeddingsDatabase for historical context
        insight_store=None,  # Optional InsightStore for extracting learnings from debates
        recorder=None,  # Optional ReplayRecorder for debate recording
        agent_weights: (
            dict[str, float] | None
        ) = None,  # Optional reliability weights from capability probing
        position_tracker=None,  # Optional PositionTracker for truth-grounded personas
        position_ledger=None,  # Optional PositionLedger for grounded personas
        enable_position_ledger: bool = False,  # Auto-create PositionLedger if True
        elo_system=None,  # Optional EloSystem for relationship tracking
        persona_manager=None,  # Optional PersonaManager for agent specialization
        dissent_retriever=None,  # Optional DissentRetriever for historical minority views
        consensus_memory=None,  # Optional ConsensusMemory for historical outcomes
        flip_detector=None,  # Optional FlipDetector for position reversal detection
        calibration_tracker=None,  # Optional CalibrationTracker for prediction accuracy
        continuum_memory=None,  # Optional ContinuumMemory for cross-debate learning
        relationship_tracker=None,  # Optional RelationshipTracker for agent relationships
        moment_detector=None,  # Optional MomentDetector for significant moments
        tier_analytics_tracker=None,  # Optional TierAnalyticsTracker for memory ROI
        loop_id: str = "",  # Loop ID for multi-loop scoping
        strict_loop_scoping: bool = False,  # Drop events without loop_id when True
        circuit_breaker: CircuitBreaker = None,  # Optional CircuitBreaker for agent failure handling
        initial_messages: list = None,  # Optional initial conversation history (for fork debates)
        trending_topic=None,  # Optional TrendingTopic to seed debate context
        pulse_manager=None,  # Optional PulseManager for auto-fetching trending topics
        auto_fetch_trending: bool = False,  # Auto-fetch trending topics if none provided
        population_manager=None,  # Optional PopulationManager for genome evolution
        auto_evolve: bool = False,  # Trigger evolution after high-quality debates
        breeding_threshold: float = 0.8,  # Min confidence to trigger evolution
        evidence_collector=None,  # Optional EvidenceCollector for auto-collecting evidence
        breakpoint_manager=None,  # Optional BreakpointManager for human-in-the-loop
        checkpoint_manager=None,  # Optional CheckpointManager for debate resume
        enable_checkpointing: bool = False,  # Auto-create CheckpointManager if True
        performance_monitor=None,  # Optional AgentPerformanceMonitor for telemetry
        enable_performance_monitor: bool = True,  # Auto-create PerformanceMonitor if True
        enable_telemetry: bool = False,  # Enable Prometheus/Blackbox telemetry emission
        use_airlock: bool = False,  # Wrap agents with AirlockProxy for timeout protection
        airlock_config=None,  # Optional AirlockConfig for customization
        agent_selector=None,  # Optional AgentSelector for performance-based team selection
        use_performance_selection: bool = False,  # Enable ELO/calibration-based agent selection
        prompt_evolver=None,  # Optional PromptEvolver for extracting winning patterns
        enable_prompt_evolution: bool = False,  # Auto-create PromptEvolver if True
        # Billing/usage tracking
        org_id: str = "",  # Organization ID for multi-tenancy
        user_id: str = "",  # User ID for usage attribution
        usage_tracker=None,  # UsageTracker instance for recording token usage
        # Broadcast auto-trigger
        broadcast_pipeline=None,  # BroadcastPipeline for audio/video generation
        auto_broadcast: bool = False,  # Auto-trigger broadcast after high-quality debates
        broadcast_min_confidence: float = 0.8,  # Minimum confidence to trigger broadcast
        # Training data export (Tinker integration)
        training_exporter=None,  # DebateTrainingExporter for auto-export
        auto_export_training: bool = False,  # Auto-export training data after debates
        training_export_min_confidence: float = 0.75,  # Min confidence to export
    ):
        """Initialize the Arena with environment, agents, and optional subsystems.

        Args:
            environment: Task definition including topic and constraints
            agents: List of Agent instances to participate in debate
            protocol: Debate rules (rounds, consensus type, critique format)

        Optional subsystems (all default to None):
            memory: CritiqueStore for persisting critiques
            event_hooks: Dict of callbacks for debate events
            event_emitter: EventEmitter for user event subscriptions
            spectator: SpectatorStream for real-time event broadcasting
            debate_embeddings: Historical debate vector database
            insight_store: InsightStore for extracting debate learnings
            recorder: ReplayRecorder for debate playback
            agent_weights: Reliability scores from capability probing
            position_tracker/position_ledger: Truth-grounded position tracking
            elo_system: ELO skill ratings for agents
            persona_manager: Agent specialization management
            dissent_retriever: Historical minority opinion retrieval
            flip_detector: Position reversal detection
            calibration_tracker: Prediction accuracy tracking
            continuum_memory: Cross-debate learning memory
            relationship_tracker: Agent interaction tracking
            moment_detector: Significant moment detection
            loop_id: ID for multi-loop scoping
            circuit_breaker: Failure handling for agent errors
            initial_messages: Pre-existing conversation (for forked debates)
            trending_topic: Topic context from trending analysis
            evidence_collector: Automatic evidence gathering
            breakpoint_manager: Human-in-the-loop breakpoint handling
            consensus_memory: Historical debate outcomes storage

        Initialization flow:
            1. _init_core() - Core config, protocol, agents, event system
            2. _init_trackers() - All tracking subsystems
            3. _init_user_participation() - Vote/suggestion handling
            4. _init_roles_and_stances() - Agent role assignment
            5. _init_convergence() - Semantic similarity detection
            6. _init_caches() - Cache initialization
            7. _init_phases() - Phase manager setup
        """
        # Initialize core configuration
        self._init_core(
            environment=environment,
            agents=agents,
            protocol=protocol,
            memory=memory,
            event_hooks=event_hooks,
            event_emitter=event_emitter,
            spectator=spectator,
            debate_embeddings=debate_embeddings,
            insight_store=insight_store,
            recorder=recorder,
            agent_weights=agent_weights,
            loop_id=loop_id,
            strict_loop_scoping=strict_loop_scoping,
            circuit_breaker=circuit_breaker,
            initial_messages=initial_messages,
            trending_topic=trending_topic,
            pulse_manager=pulse_manager,
            auto_fetch_trending=auto_fetch_trending,
            population_manager=population_manager,
            auto_evolve=auto_evolve,
            breeding_threshold=breeding_threshold,
            evidence_collector=evidence_collector,
            breakpoint_manager=breakpoint_manager,
            checkpoint_manager=checkpoint_manager,
            enable_checkpointing=enable_checkpointing,
            performance_monitor=performance_monitor,
            enable_performance_monitor=enable_performance_monitor,
            enable_telemetry=enable_telemetry,
            use_airlock=use_airlock,
            airlock_config=airlock_config,
            agent_selector=agent_selector,
            use_performance_selection=use_performance_selection,
            prompt_evolver=prompt_evolver,
            enable_prompt_evolution=enable_prompt_evolution,
            org_id=org_id,
            user_id=user_id,
            usage_tracker=usage_tracker,
            broadcast_pipeline=broadcast_pipeline,
            auto_broadcast=auto_broadcast,
            broadcast_min_confidence=broadcast_min_confidence,
            training_exporter=training_exporter,
            auto_export_training=auto_export_training,
            training_export_min_confidence=training_export_min_confidence,
        )

        # Initialize tracking subsystems
        self._init_trackers(
            position_tracker=position_tracker,
            position_ledger=position_ledger,
            enable_position_ledger=enable_position_ledger,
            elo_system=elo_system,
            persona_manager=persona_manager,
            dissent_retriever=dissent_retriever,
            consensus_memory=consensus_memory,
            flip_detector=flip_detector,
            calibration_tracker=calibration_tracker,
            continuum_memory=continuum_memory,
            relationship_tracker=relationship_tracker,
            moment_detector=moment_detector,
            tier_analytics_tracker=tier_analytics_tracker,
        )

        # Initialize user participation and roles
        self._init_user_participation()
        self._init_event_bus()
        self._init_roles_and_stances()

        # Initialize convergence detection and caches
        self._init_convergence()
        self._init_caches()

        # Initialize grounded operations helper (uses position_ledger, elo_system)
        self._init_grounded_operations()

        # Initialize phase classes for orchestrator decomposition
        self._init_phases()

        # Initialize prompt context builder (uses persona_manager, flip_detector, etc.)
        self._init_prompt_context_builder()

        # Initialize context delegator (after phases since it needs evidence_grounder)
        self._init_context_delegator()

        # Initialize termination checker
        self._init_termination_checker()

    @classmethod
    def from_config(
        cls,
        environment: Environment,
        agents: list[Agent],
        protocol: DebateProtocol = None,
        config: ArenaConfig = None,
    ) -> "Arena":
        """Create an Arena from an ArenaConfig.

        This factory method provides a cleaner way to create Arena instances
        when using dependency injection or configuration objects.

        Args:
            environment: The debate environment with task description
            agents: List of agents participating in the debate
            protocol: Optional debate protocol (defaults to DebateProtocol())
            config: Optional ArenaConfig with dependencies and settings

        Returns:
            Configured Arena instance

        Example:
            config = ArenaConfig(
                loop_id="debate-123",
                elo_system=elo,
                memory=critique_store,
            )
            arena = Arena.from_config(env, agents, protocol, config)
        """
        config = config or ArenaConfig()
        return cls(
            environment=environment,
            agents=agents,
            protocol=protocol,
            **config.to_arena_kwargs(),
        )

    def _init_core(
        self,
        environment: Environment,
        agents: list[Agent],
        protocol: DebateProtocol | None,
        memory,
        event_hooks: dict | None,
        event_emitter: Optional["EventEmitterProtocol"],
        spectator: SpectatorStream | None,
        debate_embeddings,
        insight_store,
        recorder,
        agent_weights: dict[str, float] | None,
        loop_id: str,
        strict_loop_scoping: bool,
        circuit_breaker: CircuitBreaker | None,
        initial_messages: list | None,
        trending_topic,
        pulse_manager,
        auto_fetch_trending: bool,
        population_manager,
        auto_evolve: bool,
        breeding_threshold: float,
        evidence_collector,
        breakpoint_manager,
        checkpoint_manager,
        enable_checkpointing: bool,
        performance_monitor,
        enable_performance_monitor: bool,
        enable_telemetry: bool,
        use_airlock: bool,
        airlock_config,
        agent_selector,
        use_performance_selection: bool,
        prompt_evolver,
        enable_prompt_evolution: bool,
        org_id: str = "",
        user_id: str = "",
        usage_tracker=None,
        broadcast_pipeline=None,
        auto_broadcast: bool = False,
        broadcast_min_confidence: float = 0.8,
        training_exporter=None,
        auto_export_training: bool = False,
        training_export_min_confidence: float = 0.75,
    ) -> None:
        """Initialize core Arena configuration."""
        auto_evolve = resolve_auto_evolve(auto_evolve)
        enable_prompt_evolution = resolve_prompt_evolution(enable_prompt_evolution)
        self.env = environment
        self.agents = agents
        self.protocol = protocol or DebateProtocol()

        # Wrap agents with airlock protection if enabled
        if use_airlock:
            from aragora.agents.airlock import AirlockConfig, wrap_agents

            airlock_cfg = airlock_config or AirlockConfig()
            self.agents = wrap_agents(self.agents, airlock_cfg)  # type: ignore[assignment]
            logger.debug(f"[airlock] Wrapped {len(self.agents)} agents with resilience layer")
        self.memory = memory
        self.hooks = event_hooks or {}
        self.event_emitter: Optional["EventEmitterProtocol"] = event_emitter
        self.spectator = spectator or SpectatorStream(enabled=False)
        self.debate_embeddings = debate_embeddings
        self.insight_store = insight_store
        self.recorder = recorder
        self.agent_weights = agent_weights or {}
        self.loop_id = loop_id
        self.strict_loop_scoping = strict_loop_scoping
        self.circuit_breaker = circuit_breaker or CircuitBreaker()

        # Agent pool for lifecycle management and team selection
        self.agent_pool = AgentPool(
            agents=self.agents,
            config=AgentPoolConfig(circuit_breaker=self.circuit_breaker),
        )

        # Transparent immune system for health monitoring and broadcasting
        self.immune_system = get_immune_system()

        # Chaos director for theatrical failure messages
        self.chaos_director = get_chaos_director(DramaLevel.MODERATE)

        # Performance monitor for agent telemetry
        if performance_monitor:
            self.performance_monitor = performance_monitor
        elif enable_performance_monitor:
            from aragora.agents.performance_monitor import AgentPerformanceMonitor

            self.performance_monitor = AgentPerformanceMonitor()
        else:
            self.performance_monitor = None

        # Prompt evolver for self-improvement via pattern extraction
        if prompt_evolver:
            self.prompt_evolver = prompt_evolver
        elif enable_prompt_evolution and PROMPT_EVOLVER_AVAILABLE:
            self.prompt_evolver = PromptEvolver()
            logger.debug("[evolution] Auto-created PromptEvolver for pattern extraction")
        else:
            self.prompt_evolver = None

        self.autonomic = AutonomicExecutor(
            circuit_breaker=self.circuit_breaker,
            immune_system=self.immune_system,
            chaos_director=self.chaos_director,
            performance_monitor=self.performance_monitor,
            enable_telemetry=enable_telemetry,
            event_hooks=self.hooks,  # Pass hooks for agent error events
        )
        self.initial_messages = initial_messages or []
        self.trending_topic = trending_topic
        self.pulse_manager = pulse_manager
        self.auto_fetch_trending = auto_fetch_trending
        self.population_manager = population_manager
        self.auto_evolve = auto_evolve
        self.breeding_threshold = breeding_threshold
        self.evidence_collector = evidence_collector
        self.breakpoint_manager = breakpoint_manager
        self.agent_selector = agent_selector
        self.use_performance_selection = use_performance_selection

        # Checkpoint manager for debate resume support
        if checkpoint_manager:
            self.checkpoint_manager = checkpoint_manager
        elif enable_checkpointing:
            from aragora.debate.checkpoint import CheckpointManager, DatabaseCheckpointStore

            self.checkpoint_manager = CheckpointManager(store=DatabaseCheckpointStore())
            logger.debug("[checkpoint] Auto-created CheckpointManager with database store")
        else:
            self.checkpoint_manager = None

        # Billing identity - kept for context metadata
        self.org_id = org_id
        self.user_id = user_id

        # Create extensions handler (billing, broadcast, training)
        # Extensions are triggered after debate completion - all settings stored there
        self.extensions = ArenaExtensions(
            org_id=org_id,
            user_id=user_id,
            usage_tracker=usage_tracker,
            broadcast_pipeline=broadcast_pipeline,
            auto_broadcast=auto_broadcast,
            broadcast_min_confidence=broadcast_min_confidence,
            training_exporter=training_exporter,
            auto_export_training=auto_export_training,
            training_export_min_confidence=training_export_min_confidence,
        )

        # Auto-initialize BreakpointManager if enable_breakpoints is True
        if self.protocol.enable_breakpoints and self.breakpoint_manager is None:
            self._auto_init_breakpoint_manager()

        # ArgumentCartographer for debate graph visualization
        AC = OptionalImports.get_argument_cartographer()
        self.cartographer = AC() if AC else None

        # Event bridge for coordinating spectator/websocket/cartographer
        self.event_bridge = EventEmitterBridge(
            spectator=self.spectator,
            event_emitter=self.event_emitter,
            cartographer=self.cartographer,
            loop_id=self.loop_id,
        )

        # Event bus initialized later in _init_event_bus() after audience_manager exists
        self.event_bus: Optional[EventBus] = None

        # Connect immune system to event bridge for WebSocket broadcasting
        self.immune_system.set_broadcast_callback(self._broadcast_health_event)

    def _extract_health_event_data(self, event: dict) -> dict:
        """Extract and filter health event data for broadcasting.

        Removes event_type and debate_id keys to avoid duplicate kwargs.
        """
        data = event.get("data", event)
        if isinstance(data, dict):
            return {k: v for k, v in data.items() if k not in ("event_type", "debate_id")}
        return data if isinstance(data, dict) else {}

    def _emit_health_event(self, data: dict) -> None:
        """Emit health event via EventBus or fallback to event_bridge."""
        if self.event_bus:
            self.event_bus.emit_sync(
                event_type="health_event",
                debate_id=getattr(self, "_current_debate_id", ""),
                **data,
            )
        else:
            self.event_bridge.notify(event_type="health_event", **data)

    def _broadcast_health_event(self, event: dict) -> None:
        """Broadcast health events to WebSocket clients via EventBus."""
        try:
            data = self._extract_health_event_data(event)
            self._emit_health_event(data)
        except (KeyError, TypeError, AttributeError, RuntimeError) as e:
            logger.debug(f"health_broadcast_failed error={e}")

    def _init_trackers(
        self,
        position_tracker,
        position_ledger,
        enable_position_ledger: bool,
        elo_system,
        persona_manager,
        dissent_retriever,
        consensus_memory,
        flip_detector,
        calibration_tracker,
        continuum_memory,
        relationship_tracker,
        moment_detector,
        tier_analytics_tracker=None,
    ) -> None:
        """Initialize tracking subsystems for positions, relationships, and learning."""
        self.position_tracker = position_tracker
        self.position_ledger = position_ledger
        self._enable_position_ledger = enable_position_ledger
        self.elo_system = elo_system
        self.persona_manager = persona_manager
        self.dissent_retriever = dissent_retriever
        self.consensus_memory = consensus_memory
        self.flip_detector = flip_detector
        self.calibration_tracker = calibration_tracker
        self.continuum_memory = continuum_memory
        self.relationship_tracker = relationship_tracker
        self.moment_detector = moment_detector
        self.tier_analytics_tracker = tier_analytics_tracker

        # Auto-initialize MomentDetector when elo_system available but no detector provided
        if self.moment_detector is None and self.elo_system:
            self._auto_init_moment_detector()

        # Auto-upgrade to ELO-ranked judge selection when elo_system is available
        # Only upgrade from default "random" - don't override explicit user choice
        if self.elo_system and self.protocol.judge_selection == "random":
            self.protocol.judge_selection = "elo_ranked"

        # Auto-initialize CalibrationTracker when enable_calibration is True
        if self.protocol.enable_calibration and self.calibration_tracker is None:
            self._auto_init_calibration_tracker()

        # Auto-initialize DissentRetriever when consensus_memory is available
        if self.consensus_memory and self.dissent_retriever is None:
            self._auto_init_dissent_retriever()

        # Update AgentPool with ELO and calibration systems if available
        if self.elo_system or self.calibration_tracker:
            self.agent_pool.set_scoring_systems(
                elo_system=self.elo_system,
                calibration_tracker=self.calibration_tracker,
            )

        # Sync topology setting from protocol to AgentPool
        topology = getattr(self.protocol, "topology", "full_mesh")
        self.agent_pool._config.topology = topology

        # Auto-initialize PositionLedger when enable_position_ledger is True
        if self._enable_position_ledger and self.position_ledger is None:
            self._auto_init_position_ledger()

        # Create SubsystemCoordinator with all initialized subsystems
        # This provides centralized lifecycle management while maintaining
        # backward compatibility with direct field access
        self._trackers = SubsystemCoordinator(
            protocol=self.protocol,
            loop_id=self.loop_id,
            position_tracker=self.position_tracker,
            position_ledger=self.position_ledger,
            elo_system=self.elo_system,
            calibration_tracker=self.calibration_tracker,
            consensus_memory=self.consensus_memory,
            dissent_retriever=self.dissent_retriever,
            continuum_memory=self.continuum_memory,
            flip_detector=self.flip_detector,
            moment_detector=self.moment_detector,
            relationship_tracker=self.relationship_tracker,
            tier_analytics_tracker=self.tier_analytics_tracker,
            # Disable auto-init since we already initialized everything
            enable_position_ledger=False,
            enable_calibration=False,
            enable_moment_detection=False,
        )

    def _auto_init_position_ledger(self) -> None:
        """Auto-initialize PositionLedger. See SubsystemCoordinator for details."""
        from aragora.debate.subsystem_coordinator import SubsystemCoordinator

        temp = SubsystemCoordinator(enable_position_ledger=True)
        self.position_ledger = temp.position_ledger

    def _auto_init_calibration_tracker(self) -> None:
        """Auto-initialize CalibrationTracker. See SubsystemCoordinator for details."""
        from aragora.debate.subsystem_coordinator import SubsystemCoordinator

        temp = SubsystemCoordinator(enable_calibration=True)
        self.calibration_tracker = temp.calibration_tracker

    def _auto_init_dissent_retriever(self) -> None:
        """Auto-initialize DissentRetriever. See SubsystemCoordinator for details."""
        from aragora.debate.subsystem_coordinator import SubsystemCoordinator

        temp = SubsystemCoordinator(consensus_memory=self.consensus_memory)
        self.dissent_retriever = temp.dissent_retriever

    def _auto_init_moment_detector(self) -> None:
        """Auto-initialize MomentDetector. See SubsystemCoordinator for details."""
        from aragora.debate.subsystem_coordinator import SubsystemCoordinator

        temp = SubsystemCoordinator(
            elo_system=self.elo_system,
            position_ledger=self.position_ledger,
            relationship_tracker=self.relationship_tracker,
            enable_moment_detection=True,
        )
        self.moment_detector = temp.moment_detector

    def _auto_init_breakpoint_manager(self) -> None:
        """Auto-initialize BreakpointManager when enable_breakpoints is True."""
        try:
            from aragora.debate.breakpoints import BreakpointConfig, BreakpointManager

            config = self.protocol.breakpoint_config or BreakpointConfig()
            self.breakpoint_manager = BreakpointManager(config=config)
            logger.debug("Auto-initialized BreakpointManager for human-in-the-loop breakpoints")
        except ImportError:
            logger.warning("BreakpointManager not available - breakpoints disabled")
        except (TypeError, ValueError, RuntimeError) as e:
            logger.warning("BreakpointManager auto-init failed: %s", e)

    def _init_user_participation(self) -> None:
        """Initialize user participation tracking and event subscription."""
        # Create AudienceManager for thread-safe event handling
        self.audience_manager = AudienceManager(
            loop_id=self.loop_id,
            strict_loop_scoping=self.strict_loop_scoping,
        )
        self.audience_manager.set_notify_callback(self._notify_spectator)  # type: ignore[arg-type]

        # Subscribe to user participation events if emitter provided
        if self.event_emitter:
            self.audience_manager.subscribe_to_emitter(self.event_emitter)

    def _init_event_bus(self) -> None:
        """Initialize EventBus for pub/sub event handling.

        Must be called after _init_user_participation() so audience_manager exists.
        """
        self.event_bus = EventBus(
            event_bridge=self.event_bridge,
            audience_manager=self.audience_manager,
            immune_system=self.immune_system,
            spectator=self.spectator,
        )

    @property
    def user_votes(self) -> deque[dict]:
        """Get user votes from AudienceManager (backward compatibility)."""
        return self.audience_manager._votes

    @property
    def user_suggestions(self) -> deque[dict]:
        """Get user suggestions from AudienceManager (backward compatibility)."""
        return self.audience_manager._suggestions

    def _init_roles_and_stances(self) -> None:
        """Initialize cognitive role rotation and agent stances.

        Delegates to RolesManager for role assignment, stance assignment,
        and agreement intensity application.
        """
        # Create roles manager
        self.roles_manager = RolesManager(
            agents=self.agents,
            protocol=self.protocol,
            prompt_builder=self.prompt_builder if hasattr(self, "prompt_builder") else None,
            calibration_tracker=(
                self.calibration_tracker if hasattr(self, "calibration_tracker") else None
            ),
            persona_manager=self.persona_manager if hasattr(self, "persona_manager") else None,
        )

        # Expose internal state for backwards compatibility
        self.role_rotator = self.roles_manager.role_rotator
        self.role_matcher = self.roles_manager.role_matcher
        self.current_role_assignments = self.roles_manager.current_role_assignments

        # Assign roles, stances, and agreement intensity
        self.roles_manager.assign_initial_roles()
        self.roles_manager.assign_stances(round_num=0)
        self.roles_manager.apply_agreement_intensity()

    def _init_convergence(self, debate_id: Optional[str] = None) -> None:
        """Initialize convergence detection if enabled.

        Args:
            debate_id: Debate ID for scoped caching (prevents cross-debate contamination).
                       If not provided, will be set later via _reinit_convergence_for_debate.
        """
        self.convergence_detector = None
        self._convergence_debate_id = debate_id
        if self.protocol.convergence_detection:
            self.convergence_detector = ConvergenceDetector(
                convergence_threshold=self.protocol.convergence_threshold,
                divergence_threshold=self.protocol.divergence_threshold,
                min_rounds_before_check=1,
                debate_id=debate_id,
            )

        # Track responses for convergence detection
        self._previous_round_responses: dict[str, str] = {}

    def _reinit_convergence_for_debate(self, debate_id: str) -> None:
        """Reinitialize convergence detector with debate-specific cache.

        Called at the start of run() to ensure embedding cache is isolated
        per debate, preventing cross-debate topic contamination.

        Args:
            debate_id: Unique ID for this debate run
        """
        if self._convergence_debate_id == debate_id:
            return  # Already initialized for this debate

        self._convergence_debate_id = debate_id
        if self.protocol.convergence_detection:
            self.convergence_detector = ConvergenceDetector(
                convergence_threshold=self.protocol.convergence_threshold,
                divergence_threshold=self.protocol.divergence_threshold,
                min_rounds_before_check=1,
                debate_id=debate_id,
            )
            logger.debug(f"Reinitialized convergence detector for debate {debate_id}")

    def _cleanup_convergence_cache(self) -> None:
        """Cleanup embedding cache for the current debate.

        Called at the end of run() to free memory and prevent leaks.
        """
        if self._convergence_debate_id:
            cleanup_embedding_cache(self._convergence_debate_id)
            logger.debug(f"Cleaned up embedding cache for debate {self._convergence_debate_id}")

    def _init_caches(self) -> None:
        """Initialize caches for computed values.

        Uses DebateStateCache to centralize all per-debate cached values.
        """
        self._cache = DebateStateCache()

    def _init_grounded_operations(self) -> None:
        """Initialize GroundedOperations helper for verdict and relationship management."""
        # Note: evidence_grounder is set later in _init_phases
        self._grounded_ops = GroundedOperations(
            position_ledger=self.position_ledger,
            elo_system=self.elo_system,
            evidence_grounder=None,  # Set after _init_phases
        )

    def _init_prompt_context_builder(self) -> None:
        """Initialize PromptContextBuilder for agent prompt context."""
        self._prompt_context = PromptContextBuilder(
            persona_manager=self.persona_manager,
            flip_detector=self.flip_detector,
            protocol=self.protocol,
            prompt_builder=self.prompt_builder,
            audience_manager=self.audience_manager,
            spectator=self.spectator,
            notify_callback=self._notify_spectator,
        )

    def _init_context_delegator(self) -> None:
        """Initialize ContextDelegator for context gathering operations."""
        self._context_delegator = ContextDelegator(
            context_gatherer=self.context_gatherer,
            memory_manager=self.memory_manager,
            cache=self._cache,
            evidence_grounder=getattr(self, "evidence_grounder", None),
            continuum_memory=self.continuum_memory,
            env=self.env,
            extract_domain_fn=self._extract_debate_domain,
        )

    def _init_phases(self) -> None:
        """Initialize phase classes for orchestrator decomposition."""
        init_phases(self)

        # Create PhaseExecutor with all phases for centralized execution
        self.phase_executor = create_phase_executor(self)

        # Now that phases are initialized, set evidence_grounder on _grounded_ops
        if hasattr(self, "_grounded_ops") and self._grounded_ops:
            self._grounded_ops.evidence_grounder = self.evidence_grounder

    def _init_termination_checker(self) -> None:
        """Initialize the termination checker for early debate termination."""

        async def generate_fn(agent: Agent, prompt: str, ctx: list[Message]) -> str:
            return await self.autonomic.generate(agent, prompt, ctx)

        async def select_judge_fn(proposals: dict[str, str], context: list[Message]) -> Agent:
            return await self._select_judge(proposals, context)

        self.termination_checker = TerminationChecker(
            protocol=self.protocol,
            agents=self._require_agents() if self.agents else [],
            generate_fn=generate_fn,
            task=self.env.task if self.env else "",
            select_judge_fn=select_judge_fn,
            hooks=self.hooks,
        )

    def _require_agents(self) -> list[Agent]:
        """Return agents list, raising error if empty.

        Use this helper before accessing self.agents[0], self.agents[-1],
        or random.choice(self.agents) to prevent IndexError on empty lists.
        """
        if not self.agents:
            raise ValueError("No agents available - Arena requires at least one agent")
        return self.agents

    def _sync_prompt_builder_state(self) -> None:
        """Sync Arena state to PromptBuilder before building prompts.

        This ensures PromptBuilder has access to dynamic state computed by Arena:
        - Role assignments (updated per round)
        - Historical context cache (computed once per debate)
        - Continuum memory context (computed once per debate)
        - User suggestions (accumulated from audience)
        """
        self.prompt_builder.current_role_assignments = self.current_role_assignments
        self.prompt_builder._historical_context_cache = self._cache.historical_context
        self.prompt_builder._continuum_context_cache = self._get_continuum_context()
        self.prompt_builder.user_suggestions = self.user_suggestions  # type: ignore[assignment]

    def _get_continuum_context(self) -> str:
        """Retrieve relevant memories from ContinuumMemory for debate context."""
        return self._context_delegator.get_continuum_context()

    def _store_debate_outcome_as_memory(self, result: "DebateResult") -> None:
        """Store debate outcome in ContinuumMemory for future retrieval."""
        # Extract belief cruxes from result if set by AnalyticsPhase
        belief_cruxes = getattr(result, "belief_cruxes", None)
        if belief_cruxes:
            belief_cruxes = [str(c) for c in belief_cruxes[:10]]
        self.memory_manager.store_debate_outcome(result, self.env.task, belief_cruxes=belief_cruxes)

    def _store_evidence_in_memory(self, evidence_snippets: list, task: str) -> None:
        """Store collected evidence snippets in ContinuumMemory for future retrieval."""
        self.memory_manager.store_evidence(evidence_snippets, task)

    def _update_continuum_memory_outcomes(self, result: "DebateResult") -> None:
        """Update retrieved memories based on debate outcome."""
        # Sync tracked IDs and tier info to memory manager
        self.memory_manager.track_retrieved_ids(
            self._cache.continuum_retrieved_ids,
            tiers=self._cache.continuum_retrieved_tiers,
        )
        self.memory_manager.update_memory_outcomes(result)
        # Clear local tracking
        self._cache.clear_continuum_tracking()

    def _has_high_priority_needs(self, needs: list[dict]) -> list[dict]:
        """Filter citation needs to high-priority items only."""
        return [n for n in needs if n["priority"] == "high"]

    def _log_citation_needs(self, agent_name: str, needs: list[dict]) -> None:
        """Log high-priority citation needs for an agent if any exist."""
        high_priority = self._has_high_priority_needs(needs)
        if high_priority:
            logger.debug(f"citations_needed agent={agent_name} count={len(high_priority)}")

    def _extract_citation_needs(self, proposals: dict[str, str]) -> dict[str, list[dict]]:
        """Extract claims that need citations from all proposals.

        Heavy3-inspired: Identifies statements that should be backed by evidence.
        """
        if not self.citation_extractor:
            return {}

        citation_needs = {}
        for agent_name, proposal in proposals.items():
            needs = self.citation_extractor.identify_citation_needs(proposal)
            if needs:
                citation_needs[agent_name] = needs
                self._log_citation_needs(agent_name, needs)

        return citation_needs

    def _extract_debate_domain(self) -> str:
        """Extract domain from the debate task for calibration tracking.

        Uses heuristics to categorize the debate topic.
        Result is cached at both instance level (for this debate) and
        module level (for repeated tasks across debates).
        """
        # Return instance-level cached domain if available
        if self._cache.has_debate_domain():
            # has_debate_domain() guarantees debate_domain is not None
            assert self._cache.debate_domain is not None
            return self._cache.debate_domain

        # Use module-level LRU cache for the actual computation
        domain = _compute_domain_from_task(self.env.task.lower())

        # Cache at instance level and return
        self._cache.debate_domain = domain
        return domain

    def _select_debate_team(self, requested_agents: list[Agent]) -> list[Agent]:
        """Select debate team using performance metrics if enabled.

        Delegates to AgentPool for scoring based on ELO, calibration,
        and circuit breaker filtering.

        Args:
            requested_agents: Original list of agents requested for the debate

        Returns:
            Sorted list of agents, prioritized by performance if enabled,
            otherwise the original list unchanged.
        """
        if not self.use_performance_selection:
            return requested_agents

        # Delegate to AgentPool for centralized team selection
        domain = self._extract_debate_domain()
        return self.agent_pool.select_team(
            domain=domain,
            team_size=len(requested_agents),
        )

    def _get_calibration_weight(self, agent_name: str) -> float:
        """Get agent weight based on calibration score (0.5-1.5 range).

        Delegates to AgentPool for centralized scoring.
        """
        return self.agent_pool._get_calibration_weight(agent_name)

    def _compute_composite_judge_score(self, agent_name: str) -> float:
        """Compute composite judge score (ELO + calibration).

        Delegates to AgentPool for centralized scoring.
        """
        domain = self._extract_debate_domain()
        return self.agent_pool._compute_composite_score(agent_name, domain)

    def _select_critics_for_proposal(
        self, proposal_agent: str, all_critics: list[Agent]
    ) -> list[Agent]:
        """Select which critics should critique the given proposal based on topology.

        Delegates to AgentPool for centralized critic selection.
        """
        # Find the proposer agent object
        proposer = None
        for agent in all_critics:
            if getattr(agent, "name", str(agent)) == proposal_agent:
                proposer = agent
                break

        if proposer is None:
            proposer = all_critics[0] if all_critics else None

        return self.agent_pool.select_critics(
            proposer=proposer,
            candidates=all_critics,
        )

    def _handle_user_event(self, event) -> None:
        """Handle incoming user participation events (thread-safe).

        Delegates to AudienceManager for thread-safe event queuing.
        """
        self.audience_manager.handle_event(event)

    def _drain_user_events(self) -> None:
        """Drain pending user events from queue into working lists.

        Delegates to AudienceManager for the 'digest' phase of the
        Stadium Mailbox pattern.
        """
        self.audience_manager.drain_events()

    def _notify_spectator(self, event_type: str, **kwargs) -> None:
        """Delegate to EventBus for spectator/websocket emission."""
        if self.event_bus:
            debate_id = kwargs.pop("debate_id", getattr(self, "_current_debate_id", ""))
            self.event_bus.emit_sync(event_type, debate_id=debate_id, **kwargs)
        else:
            # Fallback to event_bridge if event_bus not initialized yet
            self.event_bridge.notify(event_type, **kwargs)

    def _emit_moment_event(self, moment) -> None:
        """Delegate to EventBus for moment emission."""
        if self.event_bus and hasattr(moment, "to_dict"):
            # Use EventBus for unified event handling with tracing
            moment_data = moment.to_dict()
            self.event_bus.emit_sync(
                event_type="moment",
                debate_id=getattr(self, "_current_debate_id", ""),
                moment_type=getattr(moment, "moment_type", "unknown"),
                agent=getattr(moment, "agent_name", None),
                **moment_data,
            )
        else:
            # Fallback to event_bridge for backward compatibility
            self.event_bridge.emit_moment(moment)

    def _should_emit_preview(self) -> bool:
        """Check if agent preview hook is registered."""
        return "on_agent_preview" in self.hooks

    def _get_agent_role(self, agent: Agent) -> str:
        """Get the role string for an agent from current assignments."""
        role_data = self.current_role_assignments.get(agent.name, {})
        return str(role_data.get("role", "proposer"))

    def _get_agent_stance(self, agent: Agent) -> str:
        """Get the stance string for an agent from current assignments."""
        role_data = self.current_role_assignments.get(agent.name, {})
        return role_data.get("stance", "neutral")

    def _get_agent_description(self, agent: Agent) -> str:
        """Get the persona description for an agent."""
        if not self.persona_manager:
            return ""
        persona = self.persona_manager.get_persona(agent.name)
        return getattr(persona, "brief_description", "")

    def _build_agent_preview(self, agent: Agent) -> dict:
        """Build preview data for a single agent."""
        return {
            "name": agent.name,
            "role": self._get_agent_role(agent),
            "stance": self._get_agent_stance(agent),
            "description": self._get_agent_description(agent),
            "strengths": [],
        }

    def _emit_agent_preview(self) -> None:
        """Emit agent preview for quick UI feedback.

        Shows agent roles and stances while proposals are being generated.
        """
        if not self._should_emit_preview():
            return
        try:
            previews = [self._build_agent_preview(a) for a in self.agents]
            self.hooks["on_agent_preview"](previews)
        except Exception as e:
            logger.debug(f"Agent preview emission failed: {e}")

    def _record_grounded_position(
        self,
        agent_name: str,
        content: str,
        debate_id: str,
        round_num: int,
        confidence: float = 0.7,
        domain: Optional[str] = None,
    ):
        """Record a position to the grounded persona ledger.

        Delegates to GroundedOperations.record_position().
        """
        self._grounded_ops.record_position(
            agent_name=agent_name,
            content=content,
            debate_id=debate_id,
            round_num=round_num,
            confidence=confidence,
            domain=domain,
        )

    def _update_agent_relationships(
        self, debate_id: str, participants: list[str], winner: Optional[str], votes: list
    ):
        """Update agent relationships after debate completion.

        Delegates to GroundedOperations.update_relationships().
        """
        self._grounded_ops.update_relationships(debate_id, participants, winner, votes)

    def _create_grounded_verdict(self, result: "DebateResult") -> Any:
        """Create a GroundedVerdict for the final answer.

        Delegates to GroundedOperations.create_grounded_verdict().
        """
        return self._grounded_ops.create_grounded_verdict(result)

    async def _verify_claims_formally(self, result: "DebateResult") -> None:
        """Verify decidable claims using Z3 SMT solver.

        Delegates to GroundedOperations.verify_claims_formally().
        """
        await self._grounded_ops.verify_claims_formally(result)

    async def _fetch_historical_context(self, task: str, limit: int = 3) -> str:
        """Fetch similar past debates for historical context."""
        return await self._context_delegator.fetch_historical_context(task, limit)

    def _format_patterns_for_prompt(self, patterns: list[dict]) -> str:
        """Format learned patterns as prompt context for agents."""
        return self._context_delegator.format_patterns_for_prompt(patterns)

    def _get_successful_patterns_from_memory(self, limit: int = 5) -> str:
        """Retrieve successful patterns from CritiqueStore memory."""
        return self._context_delegator.get_successful_patterns(limit)

    async def _perform_research(self, task: str) -> str:
        """Perform multi-source research for the debate topic."""
        return await self._context_delegator.perform_research(task)

    async def _gather_aragora_context(self, task: str) -> Optional[str]:
        """Gather Aragora-specific documentation context if relevant to task."""
        return await self._context_delegator.gather_aragora_context(task)

    async def _gather_evidence_context(self, task: str) -> Optional[str]:
        """Gather evidence from web, GitHub, and local docs connectors."""
        return await self._context_delegator.gather_evidence_context(task)

    async def _gather_trending_context(self) -> Optional[str]:
        """Gather pulse/trending context from social platforms."""
        return await self._context_delegator.gather_trending_context()

    async def _refresh_evidence_for_round(
        self, combined_text: str, ctx: "DebateContext", round_num: int
    ) -> int:
        """Refresh evidence based on claims made during a debate round."""
        return await self._context_delegator.refresh_evidence_for_round(
            combined_text=combined_text,
            evidence_collector=self.evidence_collector,
            task=self.env.task if self.env else "",
            evidence_store_callback=self._store_evidence_in_memory,
            prompt_builder=self.prompt_builder,
        )

    def _format_conclusion(self, result: "DebateResult") -> str:
        """Format a clear, readable debate conclusion with full context.

        Delegates to ResultFormatter for the actual formatting.
        """
        formatter = ResultFormatter()
        return formatter.format_conclusion(result)

    def _assign_roles(self) -> None:
        """Assign roles to agents based on protocol. Delegates to RolesManager."""
        self.roles_manager.assign_initial_roles()

    def _apply_agreement_intensity(self) -> None:
        """Apply agreement intensity guidance. Delegates to RolesManager."""
        self.roles_manager.apply_agreement_intensity()

    def _assign_stances(self, round_num: int = 0) -> None:
        """Assign debate stances to agents. Delegates to RolesManager."""
        self.roles_manager.assign_stances(round_num)

    def _get_stance_guidance(self, agent) -> str:
        """Get stance guidance for agent. Delegates to RolesManager."""
        return self.roles_manager.get_stance_guidance(agent)

    async def _create_checkpoint(self, ctx, round_num: int) -> None:
        """Create a checkpoint after a debate round.

        Called by DebateRoundsPhase after each round completes.
        Only checkpoints if should_checkpoint returns True.

        Args:
            ctx: DebateContext with current debate state
            round_num: The round number that just completed
        """
        if not self.checkpoint_manager:
            return

        if not self.checkpoint_manager.should_checkpoint(ctx.debate_id, round_num):
            return

        try:
            await self.checkpoint_manager.create_checkpoint(
                debate_id=ctx.debate_id,
                task=self.env.task,
                current_round=round_num,
                total_rounds=self.protocol.rounds,
                phase="revision",
                messages=ctx.result.messages,
                critiques=ctx.result.critiques,
                votes=ctx.result.votes,
                agents=self.agents,
                current_consensus=getattr(ctx.result, "final_answer", None),
            )
            logger.debug(f"[checkpoint] Saved checkpoint after round {round_num}")
        except (IOError, OSError, TypeError, ValueError, RuntimeError) as e:
            logger.warning(f"[checkpoint] Failed to create checkpoint: {e}")

    # =========================================================================
    # Async Context Manager Protocol
    # =========================================================================

    async def __aenter__(self) -> "Arena":
        """Enter async context - prepare for debate.

        Enables usage pattern:
            async with Arena(env, agents, protocol) as arena:
                result = await arena.run()
        """
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        """Exit async context - cleanup resources.

        Cancels any pending arena-related tasks and clears caches.
        This ensures clean teardown even when tests timeout or fail.
        """
        await self._cleanup()

    def _is_arena_task(self, task: asyncio.Task) -> bool:
        """Check if an asyncio task is arena-related and should be cancelled."""
        task_name = task.get_name() if hasattr(task, "get_name") else ""
        return bool(task_name and task_name.startswith(("arena_", "debate_")))

    async def _cancel_arena_task(self, task: asyncio.Task) -> None:
        """Cancel and await a single arena-related task with timeout."""
        task.cancel()
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=1.0)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass

    async def _cancel_arena_tasks(self) -> None:
        """Cancel all pending arena-related asyncio tasks."""
        try:
            for task in asyncio.all_tasks():
                if self._is_arena_task(task):
                    await self._cancel_arena_task(task)
        except Exception as e:
            logger.debug(f"Error cancelling tasks during cleanup: {e}")

    async def _close_checkpoint_manager(self) -> None:
        """Close the checkpoint manager if it exists and has a close method."""
        if not self.checkpoint_manager or not hasattr(self.checkpoint_manager, "close"):
            return
        try:
            close_result = self.checkpoint_manager.close()
            if asyncio.iscoroutine(close_result):
                await close_result
        except Exception as e:
            logger.debug(f"Error closing checkpoint manager: {e}")

    def _count_open_circuit_breakers(self) -> int:
        """Count the number of open circuit breakers across all agents."""
        if not self.circuit_breaker:
            return 0
        agent_states = getattr(self.circuit_breaker, "_agent_states", {})
        return sum(1 for state in agent_states.values() if getattr(state, "is_open", False))

    def _track_circuit_breaker_metrics(self) -> None:
        """Track circuit breaker state in metrics if circuit breaker is enabled."""
        if self.circuit_breaker:
            track_circuit_breaker_state(self._count_open_circuit_breakers())

    def _log_phase_failures(self, execution_result) -> None:
        """Log any failed phases from the execution result."""
        if execution_result.success:
            return
        error_phases = [p.phase_name for p in execution_result.phases if p.status.value == "failed"]
        if error_phases:
            logger.warning(f"Phase failures: {error_phases}")

    async def _cleanup(self) -> None:
        """Internal cleanup implementation.

        Can be called directly if not using context manager.
        """
        await self._cancel_arena_tasks()
        self._cache.clear()
        await self._close_checkpoint_manager()

    async def run(self, correlation_id: str = "") -> DebateResult:
        """Run the full debate and return results.

        Args:
            correlation_id: Optional request correlation ID for distributed tracing.
                           If not provided, one will be generated.

        If timeout_seconds is set in protocol, the debate will be terminated
        after the specified time with partial results.
        """
        if self.protocol.timeout_seconds > 0:
            try:
                # Use wait_for for Python 3.10 compatibility (asyncio.timeout is 3.11+)
                return await asyncio.wait_for(
                    self._run_inner(correlation_id=correlation_id),
                    timeout=self.protocol.timeout_seconds,
                )
            except asyncio.TimeoutError:
                logger.warning(f"debate_timeout timeout_seconds={self.protocol.timeout_seconds}")
                # Return partial result with timeout indicator
                return DebateResult(
                    task=self.env.task,
                    messages=getattr(self, "_partial_messages", []),
                    critiques=getattr(self, "_partial_critiques", []),
                    votes=[],
                    dissenting_views=[],
                    rounds_used=getattr(self, "_partial_rounds", 0),
                )
        return await self._run_inner(correlation_id=correlation_id)

    async def _run_inner(self, correlation_id: str = "") -> DebateResult:
        """Internal debate execution orchestrator.

        Args:
            correlation_id: Request correlation ID for distributed tracing.

        This method coordinates the debate phases:
        0. Context Initialization - inject history, patterns, research
        1. Proposals - generate initial proposer responses
        2. Debate Rounds - critique/revision loop
        3. Consensus - voting and resolution
        4-6. Analytics - metrics, insights, verdict
        7. Feedback - ELO, persona, position updates
        """
        import uuid

        debate_id = str(uuid.uuid4())
        # Generate correlation_id if not provided (prefix with 'corr-' to distinguish)
        if not correlation_id:
            correlation_id = f"corr-{debate_id[:8]}"

        # Reinitialize convergence detector with debate-scoped cache
        # This prevents cross-debate embedding contamination
        self._reinit_convergence_for_debate(debate_id)

        # Extract domain early for metrics
        domain = self._extract_debate_domain()

        # Create shared context for all phases
        ctx = DebateContext(
            env=self.env,
            agents=self.agents,
            start_time=time.time(),
            debate_id=debate_id,
            correlation_id=correlation_id,
            domain=domain,
        )

        # Classify task complexity and configure adaptive timeouts
        task_complexity = classify_task_complexity(self.env.task)
        governor = get_complexity_governor()
        governor.set_task_complexity(task_complexity)

        # Classify question domain using LLM for accurate persona selection
        # This runs once and caches the result for get_persona_context() calls
        if self.prompt_builder:
            try:
                await self.prompt_builder.classify_question_async(use_llm=True)
            except Exception as e:
                logger.warning(f"Question classification failed, using keyword fallback: {e}")

        # Apply performance-based agent selection if enabled
        if self.use_performance_selection:
            self.agents = self._select_debate_team(self.agents)
            ctx.agents = self.agents  # Update context with selected agents

        # Structured logging for debate lifecycle (JSON in production)
        with correlation_context(correlation_id):
            slog.info(
                "debate_start",
                debate_id=debate_id,
                complexity=task_complexity.value,
                agent_count=len(self.agents),
                agents=[a.name for a in self.agents],
                domain=domain,
                task_length=len(self.env.task),
            )

        # Notify subsystem coordinator of debate start
        self._trackers.on_debate_start(ctx)

        # Emit agent preview for quick UI feedback
        self._emit_agent_preview()

        # Initialize result early for timeout recovery
        ctx.result = DebateResult(
            task=self.env.task,
            consensus_reached=False,
            confidence=0.0,
            messages=[],
            critiques=[],
            votes=[],
            rounds_used=0,
            final_answer="",
        )

        # Track active debates metric
        ACTIVE_DEBATES.inc()
        debate_start_time = time.perf_counter()
        debate_status = "completed"

        # Initialize OpenTelemetry tracer for distributed tracing
        tracer = get_tracer()

        with tracer.start_as_current_span("debate") as span:
            # Add debate attributes to span
            add_span_attributes(
                span,
                {
                    "debate.id": debate_id,
                    "debate.correlation_id": correlation_id,
                    "debate.domain": domain,
                    "debate.complexity": task_complexity.value,
                    "debate.agent_count": len(self.agents),
                    "debate.agents": ",".join(a.name for a in self.agents),
                    "debate.task_length": len(self.env.task),
                },
            )

            try:
                # Execute all phases via PhaseExecutor with OpenTelemetry tracing
                execution_result = await self.phase_executor.execute(
                    ctx,
                    debate_id=debate_id,
                )

                self._log_phase_failures(execution_result)

            except asyncio.TimeoutError:
                # Timeout recovery - use partial results from context
                ctx.result.messages = ctx.partial_messages
                ctx.result.critiques = ctx.partial_critiques
                ctx.result.rounds_used = ctx.partial_rounds
                debate_status = "timeout"
                span.set_attribute("debate.status", "timeout")
                logger.warning("Debate timed out, returning partial results")

            except EarlyStopError:
                # Early stop is intentional, not an error
                debate_status = "aborted"
                span.set_attribute("debate.status", "aborted")
                raise

            except Exception as e:
                debate_status = "error"
                span.set_attribute("debate.status", "error")
                span.record_exception(e)
                raise

            finally:
                # Track metrics regardless of outcome
                ACTIVE_DEBATES.dec()
                duration = time.perf_counter() - debate_start_time

                # Get consensus info from result
                consensus_reached = getattr(ctx.result, "consensus_reached", False)
                confidence = getattr(ctx.result, "confidence", 0.0)

                # Add final attributes to span
                add_span_attributes(
                    span,
                    {
                        "debate.status": debate_status,
                        "debate.duration_seconds": duration,
                        "debate.consensus_reached": consensus_reached,
                        "debate.confidence": confidence,
                        "debate.message_count": len(ctx.result.messages) if ctx.result else 0,
                    },
                )

                track_debate_outcome(
                    status=debate_status,
                    domain=domain,
                    duration_seconds=duration,
                    consensus_reached=consensus_reached,
                    confidence=confidence,
                )

                # Structured logging for debate completion
                slog.info(
                    "debate_end",
                    debate_id=debate_id,
                    status=debate_status,
                    duration_seconds=round(duration, 3),
                    consensus_reached=consensus_reached,
                    confidence=round(confidence, 3),
                    rounds_used=ctx.result.rounds_used if ctx.result else 0,
                    message_count=len(ctx.result.messages) if ctx.result else 0,
                    domain=domain,
                )

                self._track_circuit_breaker_metrics()

        # Notify subsystem coordinator of debate completion
        if ctx.result:
            self._trackers.on_debate_complete(ctx, ctx.result)

        # Trigger extensions (billing, training export)
        # Extensions handle their own error handling and won't fail the debate
        self.extensions.on_debate_complete(ctx, ctx.result, self.agents)

        # Cleanup debate-scoped embedding cache to free memory
        self._cleanup_convergence_cache()

        return ctx.finalize_result()

    # NOTE: Legacy _run_inner code (1,300+ lines) removed after successful phase integration.
    # The debate execution is now handled by phase classes:
    # - ContextInitializer (Phase 0)
    # - ProposalPhase (Phase 1)
    # - DebateRoundsPhase (Phase 2)
    # - ConsensusPhase (Phase 3)
    # - AnalyticsPhase (Phases 4-6)
    # - FeedbackPhase (Phase 7)
    #
    # NOTE: Token usage recording and training export are now handled by
    # ArenaExtensions.on_debate_complete() - see debate/extensions.py

    async def _index_debate_async(self, artifact: dict) -> None:
        """Index debate asynchronously to avoid blocking."""
        try:
            if self.debate_embeddings:
                await self.debate_embeddings.index_debate(artifact)
        except (AttributeError, TypeError, ValueError, RuntimeError, OSError, ConnectionError) as e:
            logger.warning("Async debate indexing failed: %s", e)

    def _group_similar_votes(self, votes: list[Vote]) -> dict[str, list[str]]:
        """
        Group semantically similar vote choices.

        This prevents artificial disagreement when agents vote for the
        same thing using different wording (e.g., "Vector DB" vs "Use vector database").

        Delegates to VotingPhase for implementation.

        Returns:
            Dict mapping canonical choice -> list of original choices that map to it
        """
        return self.voting_phase.group_similar_votes(votes)

    async def _check_judge_termination(
        self, round_num: int, proposals: dict[str, str], context: list[Message]
    ) -> tuple[bool, str]:
        """Have a judge evaluate if the debate is conclusive.

        Delegates to TerminationChecker.

        Returns:
            Tuple of (should_continue: bool, reason: str)
        """
        return await self.termination_checker.check_judge_termination(round_num, proposals, context)

    async def _check_early_stopping(
        self, round_num: int, proposals: dict[str, str], context: list[Message]
    ) -> bool:
        """Check if agents want to stop debate early.

        Delegates to TerminationChecker.

        Returns True if debate should continue, False if it should stop.
        """
        return await self.termination_checker.check_early_stopping(round_num, proposals, context)

    async def _select_judge(self, proposals: dict[str, str], context: list[Message]) -> Agent:
        """Select judge based on protocol.judge_selection setting.

        Delegates to JudgeSelector. For new code, use:
            selector = JudgeSelector.from_protocol(protocol, agents, elo_system, ...)
            judge = await selector.select_judge(proposals, context)
        """

        async def generate_wrapper(agent: Agent, prompt: str, ctx: list[Message]) -> str:
            return await agent.generate(prompt, ctx)

        selector = JudgeSelector(
            agents=self._require_agents(),
            elo_system=self.elo_system,
            judge_selection=self.protocol.judge_selection,
            generate_fn=generate_wrapper,
            build_vote_prompt_fn=lambda candidates, props: self.prompt_builder.build_judge_vote_prompt(
                candidates, props
            ),
            sanitize_fn=OutputSanitizer.sanitize_agent_output,
            consensus_memory=self.consensus_memory,
        )
        return await selector.select_judge(proposals, context)

    def _get_agreement_intensity_guidance(self) -> str:
        """Get agreement intensity guidance. Delegates to RolesManager."""
        return self.roles_manager._get_agreement_intensity_guidance()

    def _format_role_assignments_for_log(self) -> str:
        """Format current role assignments as a log-friendly string."""
        return ", ".join(
            f"{name}: {assign.role.value}" for name, assign in self.current_role_assignments.items()
        )

    def _log_role_assignments(self, round_num: int) -> None:
        """Log current role assignments if any exist."""
        if self.current_role_assignments:
            roles_str = self._format_role_assignments_for_log()
            logger.debug(f"role_assignments round={round_num} roles={roles_str}")

    def _update_role_assignments(self, round_num: int) -> None:
        """Update cognitive role assignments for the current round.

        Delegates to RolesManager for centralized role management.
        """
        debate_domain = self._extract_debate_domain()
        self.roles_manager.update_role_assignments(round_num, debate_domain)

        # Sync role assignments back to orchestrator for backward compatibility
        self.current_role_assignments = self.roles_manager.current_role_assignments
        self._log_role_assignments(round_num)

    def _get_role_context(self, agent: Agent) -> str:
        """Get cognitive role context for an agent in the current round.

        Delegates to RolesManager for centralized role management.
        """
        return self.roles_manager.get_role_context(agent)

    def _get_persona_context(self, agent: Agent) -> str:
        """Get persona context for agent specialization.

        Delegates to PromptContextBuilder.get_persona_context().
        """
        return self._prompt_context.get_persona_context(agent)

    def _get_flip_context(self, agent: Agent) -> str:
        """Get flip/consistency context for agent self-awareness.

        Delegates to PromptContextBuilder.get_flip_context().
        """
        return self._prompt_context.get_flip_context(agent)

    def _prepare_audience_context(self, emit_event: bool = False) -> str:
        """Prepare audience context for prompt building.

        Handles the shared pre-processing for prompt building:
        1. Drain pending audience events
        2. Sync Arena state to PromptBuilder
        3. Compute audience section from suggestions

        Args:
            emit_event: Whether to emit spectator event for dashboard

        Returns:
            Formatted audience section string (empty if no suggestions)
        """
        # Sync state to PromptBuilder first (audience_manager.drain_events is called by _prompt_context)
        self._sync_prompt_builder_state()

        return self._prompt_context.prepare_audience_context(emit_event=emit_event)

    def _build_proposal_prompt(self, agent: Agent) -> str:
        """Build the initial proposal prompt.

        Delegates to PromptContextBuilder.build_proposal_prompt().
        """
        self._sync_prompt_builder_state()
        return self._prompt_context.build_proposal_prompt(agent)

    def _build_revision_prompt(
        self, agent: Agent, original: str, critiques: list[Critique], round_number: int = 0
    ) -> str:
        """Build the revision prompt including critiques and round-specific phase context.

        Delegates to PromptContextBuilder.build_revision_prompt().
        """
        self._sync_prompt_builder_state()
        return self._prompt_context.build_revision_prompt(
            agent, original, critiques, round_number=round_number
        )
